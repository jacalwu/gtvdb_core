# CUDA 加速方案（C0 已实测）

> 版本：2026-11-09 ｜ 前置文档：[`design-review-2026-11.md`](design-review-2026-11.md)
> 触发：第三方评审意见（§1–§6）＋ 本机 C0 探针实测
> 状态：**C0 / C1 / C2 已交付**（探针 + 边界 + 驻留句柄 + flat batch KNN），C3 起未开工。

---

## 0. 结论先行

六条，按重要性排序：

1. **GPU 在本项目的唯一可行形态是「设备常驻」，不是「每次查询传一次」。** 实测：1e7 边的图（153 MiB）单向上传 **13.3 ms**，而 CPU 全量 valid-time 扫描只要 **6.9 ms**。冷路径上 GPU 一定输，且 96% 的时间花在传输上。
2. **因此 C0 的交付物不是 kernel，而是「驻留句柄 + 失效路径 + 容量预算」。** 没有这一层，后面每个 kernel 的加速比都会被传输吃掉。
3. **评审给出的优先级我大部分接受，但有两处要下调**：批量 k-hop（P4）在 Q1 当前规模（N=32–128）上不成立；Compaction prepare（P7）在数据未常驻时同样不成立。两处都加了量化门禁，而不是直接排进里程碑。
4. **容量比带宽更早成为墙，而容量可以靠"少传"买回来。** 8 GiB 卡上 budget 是 4.8 GiB；五列 23 B/edge 只装 224M 边，只传漫步要用的拓扑（5 B/edge）装 **1.03B 边**，上载同时快约 5×（§9.5，实测）。这一步不需要 kernel，已完成。
5. **"超过显存就溢出到主机内存"在图上不自动成立。** 传一条边的 5 B（0.625 ns）与 CPU 扫一条边（0.69 ns）同价，而图漫步不像 LLM 的权重那样天然被复用；分层的全部前提是"搬过去的字节会被 reread"。所以 slab 换入换出排在 kernel 之后，并配一个 spill guard 决定何时降级（§9.6）。
6. **第一个 kernel 落地后，瓶颈从传输换成了计算，而这正是想要的结果。** C2 的 flat KNN 在 100k×128 上：batch 1 赢 **3.5×**（上传全算进去），batch 32 赢 **14.3×**，盈亏点 **5 次**。阶段表里 **kernel 占 99.1%**，upload 0.2%、download 0.5%（§9.7）。也就是说 §9.5 的列投影 + `k·12 B` 的回传把传输压到可以忽略，接下来值得做的只有"让 kernel 更快"（访存形态，约 2×，见 §9.7 末尾）和"再落一个 kernel"（C3），而不是继续省传输。

---

## 1. 边界：GPU 接在哪一层

评审 §1.1 / §1.2 的判断正确，采纳：

```
        ┌─────────────────────────────────────────────┐
        │  System Time 解析 / Snapshot 选择 / RETRACT  │  CPU（存储层）
        │  live_rows_at() → Materialized TemporalCSR  │
        └───────────────────┬─────────────────────────┘
                            │  不可变快照（snapshot_id）
                            ▼
        ┌─────────────────────────────────────────────┐
        │  Device Resident Graph                      │  H2D 一次，多次复用
        │  offsets / dst / valid_from / valid_to /    │
        │  edge_type / endpoint_mask                  │
        └───────────────────┬─────────────────────────┘
                            ▼
        ┌─────────────────────────────────────────────┐
        │  GPU: valid-time filter · kNN · window      │
        │       reduce · BFS frontier                 │
        └───────────────────┬─────────────────────────┘
                            ▼
        ┌─────────────────────────────────────────────┐
        │  Arrow RecordBatch 组装 / SQL 出口          │  CPU
        └─────────────────────────────────────────────┘
```

**硬边界**（评审第 9 条，采纳）：

| 永不上 GPU | 理由 |
|:--|:--|
| WAL append / replay / 帧校验 | 持久性边界，失败即数据损坏 |
| `write_atomic` / `fsync` / `rename` / manifest | 同上 |
| `live_rows_at()` 的版本折叠 | 控制逻辑 + 随机访问；先做 streaming / 谓词下推（见设计文档 §13 Q6） |
| `SegmentMeta::classify()`（Skip/FullHit/Scan） | 纳秒级标量判定，已是 1–3 ns/段 |
| 单点 `neighbors(src, t)` | run 极短，kernel launch 成本 > 收益 |
| GC / Tag pin / ReadScope | 引用计数语义，与设备内存无关 |

---

## 2. 数据模型前提（已按评审 1.3 修订）

评审指出设计文档 §4.2 与 §9 自相矛盾，**这一条成立**，已修 `design-review-2026-11.md` §4.2。事实是：

- **业务时间**：行级，`[valid_from, valid_to)`，在 GTVSeg 列区和 CSR 里；
- **系统时间**：版本级，`SegmentEntry.system_ts`（注释原文 *"Global snapshot point"*）+ `superseded_at`；
- `BitemporalRange` 的四字段是**值类型 / 列式表面**，`migrate_legacy_edges` 给一批数据盖同一个 `system_valid_from`，等价于版本戳。

对加速器的直接后果：**kernel 只做两个 valid-time 比较 + 一个 endpoint 掩码**，签名里不出现 `system_from/system_to`。这条如果写错，整个 kernel API 会被错误的数据模型固化——评审的措辞是准确的。

---

## 3. C0 实测（本机，不是估算）

设备：`NVIDIA GeForce RTX 4060 Laptop`，24 SM，8.0 GiB，cc 8.9，128-bit 显存总线，32 MiB L2，CUDA 13.0。
源码与复现命令：[`tools/cuda-probe/`](../tools/cuda-probe/)。

### 3.1 传输

| 方向 | 载荷 | 时间 | 带宽 |
|:--|--:|--:|--:|
| H2D pageable | 153 MiB（1e7 边 × 2 × i64） | 15.0 ms | 10.7 GB/s |
| **H2D pinned** | 153 MiB | **13.3 ms** | **12.0 GB/s** |
| D2H | 153 MiB | 15.7 ms | 10.2 GB/s |

pinned 只比 pageable 快 12%——这条链路是**总线限制**，不是驱动限制。128-bit 的笔记本 4060 不是数据中心卡，别按 A100 的 2 TB/s 做预算。

### 3.2 valid-time filter（1e7 边）

| 路径 | 时间 | 吞吐 |
|:--|--:|--:|
| CPU 单线程扫描 | 6.86 ms | 1.46 Gedges/s |
| GPU kernel | 0.556 ms | 17.98 Gedges/s |
| GPU 冷路径（H2D + kernel） | **13.86 ms** | — |

- kernel 加速 **12.3×**，但冷路径 **比 CPU 慢 2×**（传输占 96%）；
- **break-even = 2.1 次**：同一快照上重复查询 ≥3 次，GPU 才开始赢。

### 3.3 512 维点积（TC4 的算术核）

`100,000 向量 × 1,000 query = 102.4 GFLOP`，语料 195 MiB。

| 路径 | 时间 | 吞吐 |
|:--|--:|--:|
| CPU（4 累加器循环，`/O2 /arch:AVX2`） | 14.17 s | 7.2 GFLOP/s |
| GPU（合并访存，1024 blocks × 256 threads） | 0.912 s | 112.3 GFLOP/s |
| GPU 有效带宽 | — | **225 GB/s** |
| 常驻加速比 | **15.5×** | 含一次 H2D 仍 15.3×（17.0 ms） |

**两个必须说清的保留**：

1. CPU 基线是朴素 4 累加器循环，**不是手调 AVX2/GEMM**。一个良好实现大约能到 40–60 GFLOP/s，届时加速比会降到 **2–4×**。15.5× 应读作上界。
2. GPU 端 225 GB/s ≈ 128-bit GDDR6 峰值的 88%，**说明这个 kernel 是带宽墙，不是算力墙**。112 GFLOP/s 相对 4060 的 FP32 峰值（约 13 TFLOPS）只用了 1%。所以 TC4 的 GPU 收益只能来自**减少读取字节**（量化 / 更小候选集），加 FLOP 没有意义。

### 3.4 成本模型（用来判每个候选值不值得做）

```
T_cpu > T_gpu + T_xfer            （单次才可能赢）
复用次数 R* = T_xfer / (T_cpu - T_gpu)   （常驻才赢，R ≥ R*）
```

代入 filter：`R* = 13.3 / (6.86 - 0.556) = 2.1`。所有后续候选都按这个式子先算一遍，再决定是否开工。

---

## 4. 核心设计：常驻句柄与失效

这是本文与评审意见的**主要增量**。评审说「先测传输」，测得的结果是：传输成本高到只有常驻才成立。那么第一优先就不是 kernel，而是**设备侧快照的生命周期**。

### 4.1 两个驻留对象，两种键

```rust
/// 图结构只随快照变化，与 business_t 无关。
pub struct DeviceGraph {
    snapshot_id: u64,        // = manifest 的 system_ts / commit
    offsets: DeviceBuf<u32>,
    dst:     DeviceBuf<u32>,
    valid_from: DeviceBuf<i64>,
    valid_to:   DeviceBuf<i64>,
    edge_type:  DeviceBuf<u16>,
    bytes: usize,
}

/// 活跃边位图随 (快照, 业务时间, 端点策略) 变化。
pub struct DeviceActiveEdges {
    snapshot_id: u64,
    business_t: i64,
    policy_mask: u64,
    bits: DeviceBuf<u64>,
}
```

键的设计要点：**不要把 `business_t` 放进 `DeviceGraph` 的键**。图结构和时间谓词是正交的，混在一起会让「同一快照上换一个 `t` 查询」变成整图重传——这正是当前 CPU 路径能免费做到、GPU 路径最容易做错的地方。

### 4.2 设备内存是缓存，不是事实源

| 规则 | 后果 |
|:--|:--|
| CPU（mmap 段）永远是权威副本 | GPU 崩溃 / 驱动重置 / OOM 不影响存储正确性 |
| `snapshot_id` 变更 → 旧句柄立即失效 | 新段发布后不会读到旧图 |
| 驻留期间用 Tag pin 住相关段 | GC 不会在设备持有引用时删文件 |
| 超出预算按 LRU 驱逐，驱逐即降级 CPU | 无「显存不足」这种错误路径，只有快慢 |
| 默认 `max_device_bytes = 60% VRAM` | 留出 workspace 与 top-k 缓冲 |

以本机 8 GiB 为例：1e7 边 × 16 B = 160 MB，1e8 边 = 1.6 GB，**AML 规模的真实图完全可以常驻**。这不是理论上的可行性，是这条路线唯一站得住的理由。

### 4.3 失效路径必须可测

`DeviceGraph` 的失效有三个来源，每个都要有测试：

1. `publish_delta()` / `flush()` 产生新 `snapshot_id`；
2. Tag 被 drop 后 GC 回收了段（pin 计数归零）；
3. 显存压力触发驱逐。

对应断言：失效后下一次查询必须**要么命中 CPU 路径且结果一致，要么重新上传**，不能返回旧图的结果。这条比任何 kernel 的加速比都重要。

---

## 5. 后端抽象

评审建议的 `Accelerator` trait 方向正确，但**无状态 trait 表达不了唯一能赢的模式**。加驻留句柄：

```rust
pub trait Accelerator: Send + Sync {
    fn name(&self) -> &'static str;
    fn caps(&self) -> Caps;                      // device_bytes, resident, max_k

    /// 把不可变快照上传成设备句柄；不支持的后端返回 None（调用方回落 CPU）。
    fn load(&self, view: SnapshotView<'_>) -> Result<Option<DeviceHandle>>;
    fn release(&self, h: DeviceHandle);

    fn valid_time_filter(&self, h: &DeviceHandle, t: i64, policy: EndpointMask)
        -> Result<EdgeBitmap>;
    fn knn_flat(&self, h: &DeviceHandle, queries: &[f32], k: usize)
        -> Result<Vec<NeighborList>>;
    fn window_reduce(&self, h: &DeviceHandle, window: usize, op: WindowOp)
        -> Result<Vec<f64>>;
    fn batch_khop(&self, h: &DeviceHandle, seeds: &[u64], hops: u8, at: i64)
        -> Result<Vec<u64>>;
}
```

后端：`CpuAccelerator`（永远可用、永远是 reference）、`CudaAccelerator`（feature 门控）。

**数值一致性**：整数路径要求**逐位相等**；浮点路径定义容差（`|Δ| ≤ 1e-5 * max(1,|x|)`）并额外断言 **top-k 集合相等**——距离值差一点可以接受，返回不同的邻居不行。

---

## 6. 优先级与门禁（与评审的差异）

| 优先级 | 候选 | 评审 | 本文 | 判据 |
|:--|:--|:--|:--|:--|
| **C1** | TC4 KNN（flat 全扫、批量 query、向量常驻） | P1 | **同意最高** | 实测 15.5×（上界）／乐观估计 3–5×；IVF 路径不在此列，理由见 §6.1 |
| **C2** | TC2 OFI / rolling（prefix sum + window diff） | P2 | **同意** | 行宽约 16 B（价 + 量），1e7 行传输仍约 13 ms，而 CPU fused 在同规模约 19 ms，`R*` 预计 **< 1**（冷路径都不亏）；但必须实测，不能从点积外推 |
| **C3** | valid-time filter + ActiveEdgeBitmap | P3 | **同意，但要求链式** | 单独使用必然亏（§3.2）；只有「filter → 后续 GPU 算子」不回传才有意义 |
| **C4** | 批量 k-hop / BFS | P4 | **下调** | 见 §6.2：N=32–128 时 CPU 已 316 µs，GPU 不成立 |
| **C5** | TC3 ring3 候选生成 | P5 | **同意，范围收缩** | 只做候选生成，语义验证留 CPU |
| **C6** | Compaction prepare（sort/scan/scatter） | P7 | **下调** | 见 §6.3：数据在磁盘上，传输 13 ms 起步 |

### 6.1 为什么 TC4 只做 flat，不做 IVF

仓库现有 TC4 两条路：flat（CI gate 6.23 ms）与 IVF（369.8 µs，`nlist=1024`、`nprobe=32`）。

IVF 扫的是 `32/1024 ≈ 3.1%` 的向量：100k 语料 × 3.1% × 512 × 4 B ≈ **6.4 MiB/query**。它已经是亚毫秒级，GPU 要先把这 6.4 MiB **按索引 gather** 过来——gather 的随机访存比顺序扫描慢数倍，再加上 launch 与回传，没有赢面。

**结论**：GPU 的 TC4 只对 **flat 全扫 + 大批量 query + 向量常驻** 成立。IVF 留在 CPU。评审把 TC4 列为 P1 是对的，但需要补这一条限定，否则会把力气花在错的形态上。

### 6.2 为什么下调批量 k-hop

Q1 当前基线（M6-a，Windows release，shared 路径）：N=32 时 **268.7 µs**，N=128 时 **316.6 µs**。

GPU 要赢，需要：

```
T_kernel + T_seeds + T_result < 316 µs
```

1e7 边的 2-hop：kernel 本身在随机访存下**估计** 0.2–0.8 ms（随 frontier 规模变化，未经实测），再加 seed 上传与结果回传。**在 N ≤ 128 时没有赢面**，而这正是当前 Q1 的测量口径。

评审说「Q1 是核心差异化，迟早要有 GPU 路径」——这个判断对，但**「迟早」不该写进 C3 之前的里程碑**。门禁：

- 触发条件：`seeds ≥ 10^4` **或** 单层 frontier ≥ `10^6`；
- 届时先做 **Push BFS parity**，再谈 direction-optimizing（评审在这一点上是对的：不要一上来移植 `khop_directed()` 的全部策略）。

### 6.3 为什么下调 compaction prepare

评审把它排在 P7（较后期），我同意顺序，但理由要更硬：compaction 的输入是**磁盘上的段**，不是设备内存。若不复用常驻副本，起手就是 13 ms 传输；而一次 1e7 边段的 CPU 端 prepare 是几十毫秒量级，**收益窗口很窄**。

只有当「ingest 直接写设备缓冲 → 常驻 → compaction 在设备上完成」这条链整体成立时，C6 才有意义。这也意味着 C6 依赖 C1/C3 的驻留基础设施，不能单独做。

### 6.4 不值得做（与评审一致）

| 项 | 理由 |
|:--|:--|
| TC1 字符串 join | 瓶颈是哈希与带宽；166 MiB 过 PCIe 就 13 ms，输出还膨胀 4.2 倍 |
| TC5 独立 PIT 查询 | 已 0.070 µs（bin 路径），GPU 调度开销比它大三个数量级 |
| 单点 `neighbors()` | run 太短，launch 成本主导 |
| L2 `classify()` | 1–3 ns/段，已是标量极限 |
| `live_rows_at()` 首版 | 控制逻辑；先做 streaming / 谓词下推 |
| 全部控制平面 | WAL / manifest / fsync / publish / GC |

---

## 7. 语言与工具链

评审建议「Rust host + CUDA C++ kernel」，**采纳**。但 CUB 的引入方式要给一个可落地的方案，否则会与「默认构建不依赖 CUDA」冲突。

### 7.1 三种接法，选第三种

| 方案 | 默认构建 | CI | 问题 |
|:--|:--|:--|:--|
| A. `build.rs` + `nvcc` | 需要 toolkit | 每个 job 装 CUDA | 违背「无 GPU 的 CI 必须全绿」 |
| B. `cudarc` + NVRTC 运行时编译 | 不需要 | 只需驱动 | CUB 头文件经 NVRTC 编译很脆（`-std=c++17`、include 路径、模板实例化） |
| **C. `xtask` 预编译 PTX，`include_bytes!` 嵌入** | **不需要** | 默认 job 不碰 | 需要一条手动/定时的 kernel 构建流水线 |

选 C：kernel 用 CUDA C++ 写（可用 CUB），由 `cargo xtask cuda-ptx` 在有 toolkit 的机器上编成 PTX，产物入库（体积小、可审阅）；运行时 `cudarc` 只做驱动 API + 模块加载。PTX 的 JIT 向前兼容，Ada 之后的新卡也能跑。

**代价**：PTX 需要针对每个 compute capability 编一份（当前 `sm_89`），并且 kernel 改动要走两步（改 `.cu` → 跑 xtask → 提交 PTX）。这个代价换来的是「99% 的开发者 `cargo test` 完全不碰 CUDA」。值得。

### 7.2 Rust CUDA 的定位

`cuda-oxide` 与 `rust-cuda` 都仍是 early-stage（官方文档自己这么写）。用于 **原型**（filter、bitmap、简单 reduction）可以，**不用于** 存储正确性相关路径，也不让公共 API 绑定某个 kernel 语言。这与评审 §4 的判断一致。

---

## 8. 准入门禁（采纳评审 10 条，补 4 条）

任一 CUDA 实现必须同时满足：

1. 与 CPU reference **逐项一致**（整数逐位；浮点按 §5 容差 + top-k 集合相等）；
2. 同时报告 **kernel-only** 与 **端到端** 时间；
3. 报告 H2D / D2H 的**字节数与时间**；
4. 报告**冷启动**与**常驻**两条曲线（不是只报好看的那条）；
5. 小规模 / 显存不足**自动回落 CPU**，不报错；
6. GPU 任何失败**不影响存储正确性**；
7. `cuda` feature **默认关闭**；
8. 无 GPU 的 CI **必须通过**；
9. CUDA 代码**不得进入** WAL / manifest / publish 的正确性边界；
10. 只有**端到端 p99 改善**才进主路径；
11. **（补）** 驻留对象必须绑定 `snapshot_id`，并有失效测试（§4.3 三个来源）；
12. **（补）** 必须声明 `max_device_bytes` 与驱逐策略，且驱逐后行为与冷启动一致；
13. **（补）** 必须报告**结果回传字节数**——bitmap / 邻居列表的体积常被忽略，却是 N 较小时的主要成本；
14. **（补）** 基准必须能一键跑 CPU-only 对照，并输出两边的比值（否则无法判断退化）。

---

## 9. 路线图

| 阶段 | 内容 | 验收 |
|:--|:--|:--|
| **C0** ✅ 已交付 | 设备探针（带宽 / filter / dot）+ 成本模型 + 本文档 | 本机数据可复现；`R*` 公式与三个候选的判定 |
| **C1** ✅ 已交付 | `cuda` feature + `Accelerator` trait + 驻留缓存 + CPU 回落（**无 kernel**） | 无 GPU 机器 `cargo test` 全绿（571 项）；失效场景见 §9.1 |
| **C2** ✅ 已交付 | TC4 flat batch KNN（向量常驻） | 实测 batch 1 → **3.5×**、batch 32 → **14.3×**（100k×128，含摊销后的首次上传），top-k 集合相等；盈亏 **5 次**（见 §9.7） |
| C3 | TC2 OFI（prefix sum + window diff） | vs CPU fused ≥ 2×，逐值相等 |
| C4 | valid-time filter + `ActiveEdgeBitmap`（链式，不回传） | 「filter → 聚合」端到端改善；单独 filter 不单独立项 |
| C5 | 批量 k-hop（Push BFS） | 仅在 `seeds ≥ 1e4` 或 frontier ≥ 1e6 的场景验收 |
| C6 | TC3 ring3 候选生成 | 候选集与 CPU 一致；语义验证仍在 CPU |
| C7 | compaction prepare | 仅在「ingest → 设备常驻 → compaction」整链成立后评估 |

顺序理由：C1 是所有后续的前置（没有驻留就没有赢面）；C2/C3 是数值密集、语义最简单、最容易验证；C5/C6 有分支与语义，放后面。

### 9.1 C1 交付后的实测修正

C1 把 §3.4 的成本模型从"估算"换成"测量"，并推翻了两处：

| 项 | C0 的估算 | C1 实测（RTX 4060 Laptop，2026-11） |
|:--|:--|:--|
| 每边上载字节 | 16 B | **26–27 B**（`dst` u64 + 2×i64 + u16；C0 按 u32 `dst` 算） |
| 1e7 边上载 | 153 MiB / 13.3 ms（pinned，12.0 GB/s） | **257.5 MiB / 46.3 ms 冷、35.7 ms 暖（5.6–7.6 GB/s，pageable）** |
| filter break-even | 2.1 次 | **7.6 次（冷）/ 5.9 次（暖）** |

两处修正的含义：

1. **`dst` 的宽度直接决定盈亏点。** 现在按源宽度 `u64` 上载，零转换；改成 `u32` 可降到 22 B/edge，break-even 回到约 6.4 次。**这一条已在 §9.2 实测**：降幅是真的，但只能由上游提供窄列，不能在 `load_graph` 里转换。

2. **pageable 与 pinned 差 1.6×。** 上载走 `Vec` 直接 H2D，实测 7.0–7.6 GB/s；C0 的 12.0 GB/s 是 pinned。是否加 staging 池，取决于 C2 能否把传输摊销掉——在 7.6 次 break-even 面前，1.6× 传输改善只把门槛降到 4.7 次，不改变结论。

另有一条只在实测中出现的事实：**显存按 2 MiB 粒度计费**。300 KB 的图首次上载实测占用 2 097 152 B，第二个同尺寸快照为 0 B（落在同一池内）。所以 `max_device_bytes` 按"申请字节"记账会低估真实占用，预算不能设成刚好等于显存。

C1 刻意**不**注册任何算子（`Caps::supports` 全 false），因此 `ExecutionPolicy` 在 C1 期间总是把查询判给 CPU，`AcceleratorRegistry::route` 报 `Unsupported` 而非 `NoDevice`——"这台机器没有 GPU"和"GPU 还没移植这个算子"是两种结论。C2 每落一个 kernel 翻一个 bit，policy 不用改。

### 9.2 `dst` 收窄：0.9 次的门槛，和一个被删掉的实现

§9.1 把"改成 u32 可降到 22 B/edge"列为 C2 第一个要测的量。测完了。两个宽度在**各自独立的进程**里量，否则第二个宽度继承第一个已经热好的池子，冷启动就不叫冷启动。

命令：`bench_residency 10000000 wide|narrow`（RTX 4060 Laptop，2026-11-10，pageable）

| 项 | `dst` u64（源宽度） | `dst` u32（现成窄列） |
|:--|--:|--:|
| 每边上载字节 | **27.0 B** | **23.0 B** |
| 1e7 边一份拷贝 | 257.5 MiB | 219.3 MiB |
| 冷上载 | 45.9 ms（7.3 次盈亏） | 40.4 ms（6.4 次盈亏） |
| 稳态上载 | 34.8–37.0 ms，7.3–7.9 GB/s | 30.9–31.0 ms，7.45 GB/s |
| 稳态盈亏点 | **5.9 次** | **4.9 次** |

比 26/22 高 1 B，是因为 offsets 列（4 B/节点）摊到每条边上；平均出度 4 时正好 1.0 B。

三条结论，按重要性：

1. **收益是真的，但比预想小。** 冷路径 7.3 → 6.4 次，稳态 5.9 → 4.9 次，即 **0.9–1.0 次**，不是 §9.1 推算的 1.2 次（那是拿 7.6 做基线的结果，本次冷测重跑得 7.3）。显存省 15%：8 GiB 卡、4.8 GiB 预算下，1e7 边的快照从 19 个变 22 个。
2. **不能在 `load_graph` 里转换——这是第一版实现，测完删掉了。** 一次 8→4 的转换要搬 12 B/edge（读 8、写 4），1e7 边实测 **9.5–11.2 ms**（新分配缓冲 14–19 ms，首次触碰 40 MB 页占大头），而它省下的传输只有 **5.7 ms**。转换比它省下的字节更贵，因为主机内存带宽（约 12 GB/s）和 PCIe（7.5 GB/s）是同一个量级，而转换搬的字节是省下的 3 倍。任何"上载时顺手压一下"的写法都是净亏。
3. **所以 22 B/edge 只能由生产方提供。** `SnapshotView::dst32` 就是那个入口：谁在构建 CSR 时顺手产出窄列，谁就能零成本拿到 4 B/edge。一次转换 9.5 ms，只要同一张图会被上载 **≥1.7 次**（换模型重传、多卡、驱逐后重载）就回本；融合进 CSR 构建（单遍、复用缓冲）会更快。

对 kernel 的影响（C2 起）：`ResidentGraph::dst_layout()` 记录这份拷贝的宽度，kernel 必须读它而不是假设。读错宽度不会报错——`u32` 列按 `u64` 读，每条边变成两个垃圾节点 id，答案错而系统沉默。这是本次改动里唯一"防事故"的部分，比那 4 B 重要。

还有一条比算术更硬的约束，顺手算出来写在这里，因为它决定 C2/C4 的 kernel 形状：**回传格式**。filter 的结果若以"边 id 列表"回到主机，30% 选择率下是 3e6 × 8 B = 24 MB，按 C0 实测的 D2H 10.2 GB/s 需要 **2.35 ms**，而 kernel 本身只跑 0.556 ms——省下的 6.30 ms 被吃掉一大半，break-even 反而升到 **7.8 次**，比不收窄还差。换成位图回传（1e7 bit = 1.25 MB）只要 **0.12 ms**，并且 `Sel::from_bitmap` 能直接接住，不用转稠密。所以 device 算子的输出必须是位图或仍驻留的句柄，不能是稠密 id 列表；这条比 block 开多大重要得多。

### 9.3 阶段实测：瓶颈不在 H2D，在 materialize

§9.1/§9.2 报的都是"上载要多久"。评审追问的是另一件事：**上载在整条冷路径里到底占多少**。这次把它拆成四段，每段单独计时。

命令：`bench_residency 10000000 wide|narrow`（阶段表）、`bench_graph_load 10000000`（materialize）；RTX 4060 Laptop，2026-11-10。

#### 上载内部：97% 是 PCIe，没有隐性复制

`GraphLoadMetrics` 逐列记账，1e7 边窄列（`bench_residency 10000000 narrow`）：

| 阶段 | 字节 | ms | 占比 | 带宽 |
|:--|--:|--:|--:|--:|
| offsets | 10 000 004 | 1.30 | 4.4% | 7.68 GB/s |
| dst (u32) | 40 000 000 | 5.07 | 17.1% | 7.89 GB/s |
| valid_from | 80 000 000 | 10.23 | 34.5% | 7.82 GB/s |
| valid_to | 80 000 000 | 10.24 | 34.5% | 7.81 GB/s |
| edge_type | 20 000 000 | 2.84 | 9.6% | 7.03 GB/s |
| host pack / transform / staging | **0** | **0.00** | 0% | — |
| device sync | — | 0.088 | — | — |
| 上载合计 | 230 000 004 | 29.68 | | 7.75 GB/s |
| 设备分配 | — | 0.71 | | |
| 整次 load | 230 000 004 | 30.49 | 97.3% 是 PCIe | |

宽列（`dst` u64）同构，1e7 边 33.84 ms / 7.98 GB/s，各行带宽 7.5–8.1 GB/s 一致——没有哪一列异常，**没有 AoS 中间态、没有逐边转换、没有 staging 缓冲**。这三项在结构里是字段而不是"不存在"，所以只要有人加进来，`is_copy_free()` 立刻变 false，`the_upload_reports_where_its_milliseconds_went` 立刻红。这是对评审"检查是否存在隐性多次复制"的回答：**上载路径里没有，一条都没有**。

顺带确认 `cuMemcpyHtoD` 确实是同步的：之后显式 `cuCtxSynchronize` 只花 0.088 ms，说明逐列数字是传输时间而不是排队时间。

#### pinned 端到端 A/B：**9% 更慢**，否决

评审说 pinned 只快 12%，C0 说快 58%（12.0 vs 7.6 GB/s）。两边都对了一半——差在测的是哪一段。端到端 A/B（219.3 MiB，五列各一块 pinned，缓冲复用）：

| 路径 | 第 1 次 | 第 3 次 | 稳态带宽 |
|:--|--:|--:|--:|
| pageable `cuMemcpyHtoD` | 40.53 ms | **30.64 ms** | 7.51 GB/s |
| pinned `fill`（主机搬进固定页） | 14.10 | 14.10 | 16.31 |
| pinned `cuMemcpyHtoDAsync` | 19.03 | 19.25 | **11.95** |
| pinned 合计 `fill + copy` | 33.13 | **33.35 ms** | 6.90 GB/s |

纯拷贝 pinned 是 **11.95 GB/s，比 pageable 快 59%**——C0 的 12.0 是这一段，没错。但 pinned 必须先 `fill`：14.10 ms 的主机拷贝 > 11.4 ms 的 DMA 收益，**净亏 2.7 ms**。评审的判断对，理由也对（"如果 B 的总时间包含 CPU staging copy，可能并不优于 A"），只是快慢比例不是 12%。

另有一条只在本机出现的成本：五列 pinned 一次性分配要 **33.44 ms**，等于一整次上载。所以 pinned 池不是"免费加速"，它的回本周期要单独算；在 `fill` 仍存在的前提下，任何周期都回不了本。

结论：**上载保持 pageable**。唯一能让 pinned 成立的做法是让生产方直接把 CSR 列建在固定页里（评审的方案 C），省掉 `fill`：那会得到 19.25 ms（−11.4 ms，break-even 4.8 → 3.4 次），代价是每快照 219 MiB 不可换出内存。记账上不划算，等下面 materialize 的问题解决后再复议。

#### 上载之前：materialize 是上载的 51 倍

`graph_at(T)` 的实测（真实 `SegmentStore`，磁盘上的 GTVSeg，1e7 边）：

| 规模 | `live_rows_at` | `graph_at` 合计 | ns/edge | LiveRow 峰值 |
|:--|--:|--:|--:|--:|
| 1e6 | 40.05 ms | 88.92 ms | 88.9 | 68 MiB |
| 2e6 | 79.16 | 174.70 | 87.3 | 137 MiB |
| 4e6 | 165.24 | 375.45 | 93.9 | 274 MiB |
| **1e7** | **497.83** | **1 579.06** | **157.9** | **686 MiB** |

1e7 与 4e6 之间斜率变大，因为 1e7 边被 `ingest` 自动切成 10 个段，版本遍历要跨段。冷/暖几乎一样（1570 vs 1579 ms），所以**这不是缺页，是重算**。

拆开看，1e7 的 1579 ms 里：
- **498 ms** `live_rows_at`：扫段 + 版本折叠，产出一个 `Vec<LiveRow>`；
- **1081 ms** 其余：`Vec<EdgeRecord>`（AoS，686 MiB）再折成五列 SoA（257 MiB）。

也就是说，评审担心的那条链 **确实存在，只是在 materialize 里，不在上载里**：

```
GTVSeg mmap → Vec<LiveRow>(686 MiB) → Vec<EdgeRecord>(AoS) → TemporalCSR(SoA, 257 MiB) → 各列直传
                                └─ 1081 ms，2.7× 内存放大 ─┘
```

于是 1e7 边的冷路径是 **1 579 + 41 = 1 620 ms**，其中上载占 **2.5%**。对照 CPU 路径的 6.86 ms 全扫：

| | CPU 每查询 | GPU 冷路径 | GPU 稳态（kernel 0.556 ms） |
|:--|--:|--:|--:|
| 1e7 边 | 6.86 ms | **1 620 ms（236×）** | 0.556 ms |

**上载内盈亏 4.8 次是上载内部的账；端到端盈亏是 255 次。** 两个都对，把前者当结论就错了——这是本次测量推翻的第三个数（前两个是 26 B/edge 和 pinned 12 GB/s）。

#### 对 C2 的影响：先修 materialize，再写 kernel

1. **C2 的门禁从"能上载"改成"能常驻且不被重建"。** 设备路径的前提是同一 `snapshot_id` 的 CSR 被构造一次、`Arc` 住、复用到底（`GtvContext::register_neighbors` 已经是这个形状）。若每次查询都 `graph_at`，GPU 永远不可能赢。
2. **`graph_at` 的 AoS 往返必须消掉，这是存储层的事。** 逐列从段里读出来直接追加进 SoA，省掉 686 MiB 中间态和 1081 ms 折叠；上限是把 1579 压到接近 `live_rows_at` 的 500 ms，但**这是待测的，不是承诺**。它同时也让 CPU 侧的 pattern/khop 变快，所以不依赖 CUDA 也值得做。→ **已做完，比上面这个上限好得多，见 §9.4。**
3. **`snapshot_materialize_ns` 留在 `GraphLoadMetrics` 里，由调用方填。** 后端无法知道自己输入的构造耗时；留 `None` 是"没测"，填 0 是撒谎。
4. **时间列窄化（评审 §9）现在的优先级高于 `dst` 窄化。** 1e7 边里两列时间占 **69.6%** 的线（152 MiB），按实测 7.75 GB/s 折算是 **10.3 ms**，而 `dst` 收窄只省 5.07 ms。但注意：这只在 materialize 修好之后才有意义——现在省 10 ms 和亏 1500 ms 是同一句话里的两个数。

---

### 9.4 列式 materialize：`graph_at` 不再重建，而是归并

§9.3 把瓶颈指到了 `graph_at`（1 579 ms，是上载的 51 倍）。改动只有一件事：**不再经过 AoS**。

发布出去的段，它的 CSR 在 flush 时就已经按 `(src, valid_from, valid_to, dst)` 排好序了。旧实现把这个序丢掉（摊平成 `LiveRow`），再用 `from_edge_columns` 排一遍——同一份数据排两次，还多背 686 MiB 的中间态。新实现把每个存活段当成一条有序 run，k 路归并时直接写进五列；输出 `offsets` 随归并推进即时收口，不再数一遍度数再做前缀和。

它和 `live_rows_from` 必须逐字节一致（同一套存活判定、同一套 retract 命名空间、同一套 `RowPos` 编号），所以两者放在一起，`tests/graph_at_columnar.rs` 用**行路径本身**当 oracle 逐列比对，并额外直接断言 CSR 文档里的排序不变量——行路径的序来自排序，归并没有排序，这条只能自己守着。

1e7 边，同一台机器，各自独立进程：

| 形态 | `graph_at` | 相对旧实现 |
|:--|--:|--:|
| 旧实现（行路径 + 重排） | 1 508–1 579 ms | 1.0× |
| 新实现，3 个存活段（归并） | **393 ms** | 3.8× |
| 新实现，压实后 1 个段（原样拷贝） | **176 ms** | 8.6× |

第二条是重点：**未压实和压实后的差别比"新实现 vs 旧实现"还大**，因为材料化的成本不是图的属性，是段数的属性。

#### 393 ms 里的 390 ms 花在哪

用探针把归并拆成三段（1e7 边，3 条 run）：

| 阶段 | 时间 | 说明 |
|:--|--:|:--|
| 只迭代、不写列 | 102 ms | 读五列 + 判 retract，10 ns/edge |
| 顺序 drain + 写列（无选择） | 175 ms | 写 26 B/edge，+72 ms |
| 完整归并（含选最小 run） | 390 ms | **选择器本身约 215 ms** |

选择器是 21 ns/edge，不是因为比较次数多（3 条 run），而是因为**每次比较的走向都由数据决定**：三条 run 在 src 上高度交错，每输出一行都要重新决定赢家，分支预测器基本必错。这是归并这一类算法固有的代价，靠换堆或换循环形状收益有限。

对照 `fs::read` 读同样 260 MiB：**78 ms（3.3 GB/s）**；读+写 260 MiB 的拷贝下限约 157 ms。所以压实后的 176 ms **已经贴在这台机器的拷贝上限上**，没有剩余空间；而未压实的 393 ms 是"拷贝下限 + 归并选择"，压下去的办法是**压实，不是把归并写得更快**。

#### 对盈亏的影响

1e7 边、未压实：冷路径 393 + 41 = **434 ms**，端到端盈亏 **69 次/快照**（§9.3 记的是 255 次）。
压实后：冷路径 176 + 41 = **217 ms**，盈亏 **34 次/快照**。

也就是说 §9.3 的结论"先修 materialize 再写 kernel"已经兑现了一半：materialize 从"不可能摊销"降到"比上载贵但可摊销"。**剩下的那一半不是存储层能解决的**——217 ms 里 41 ms 是 PCIe、176 ms 是把 257 MiB 复制一遍，而复制之所以必要，只因为 `graph_at` 的返回类型是**拥有**的 CSR。真要再降一个量级，得让查询直接读段视图、根本不 materialize（§8 的第二阶段），那是 C4 之后的事。

顺带两条：`live_rows_at` 保留不动（Arrow 表面和 `visible_edges_at` 还在用），它现在比 `graph_at` 慢（477 vs 393 ms），这是对的——它是行形态，`graph_at` 是列形态，后者才是 CSRs 的用途。

---

### 9.5 垂直分层：只把漫步要读的列放进显存（已实现）

§9.2 和 §9.3 一直在问"怎么把 23 B/edge 传得更快"。这一节换问题：**为什么要传这 23 B**。

23 B/edge 里 16 B 是 `valid_from` + `valid_to`，2 B 是 `edge_type`。一次图漫步（沿 `offsets` 取区间、读 `dst` 继续走）**一个字节都不读这三列**。列存数据库早就在做这件事（投影下推），GPU 侧只是把它从"扫描时少读"变成"根本不传"。

#### 实现

- `ColumnSet`：五个列名的位集合，带 `TOPOLOGY`（offsets+dst）、`TEMPORAL`、`LABELS`、`ALL` 四个命名组合，以及 `bytes(nodes, edges, layout)` ——**字节模型的唯一出处**，`SnapshotView::bytes`、容量判断、bench 三处都调它。
- `SnapshotView::with_columns(set)`：投影是**算子的属性**，不是图的属性。投影里没有的列，`validate` 不要求数组存在——调用方不必为了传拓扑去借一份它故意留下的时间列。
- `validate`：拓扑不是"某一列"，而是"这份拷贝是什么"。缺 `offsets`/`dst` 直接拒绝，因为 `edge_count` 从 `dst` 读，缺了就会把一份 offsets 价成 0 条边。
- 缓存键变成 `(snapshot_id, ColumnSet)`。这一条是正确性，不是性能：只按 snapshot 做键，一次拓扑请求会命中一份没有时间列的拷贝，然后越界读——错答案，不是 miss。`ResidentGraph` 因此多了 `columns()` 与 `check_columns(required)`，读 `valid_from` 的算子会拿到"这份拷贝只有 offsets+dst"的错误，而不是一个越界指针。
- `GraphLoadMetrics` 加 `projection` 与 `deferred_bytes`：**留着没传的部分要被记账**，否则"省了"和"还没传"分不清。`priced_bytes == uploaded_bytes` 这条 ratchet 保留，投影改变等式两边，不改变等式。
- `Caps::admit(needed)`：上载前先问预算。`cuMemAlloc` 传到一半失败只给一个驱动码和一片碎片化的池子；预检给的是"23 B/edge × 1.2e9 装不进 4.8 GiB"——这句话才是查询规划器需要的（`Fallback::Cpu`，永远不是查询失败）。
- `DeviceGraph.columns` 从 `Vec<u64>` 变成 `Vec<(Column, u64)>`：少传两列之后，指针表按下标推断列身份就不再成立了，kernel 必须按名字取。

#### 实测（RTX 4060 Laptop 8 GiB，budget 4.8 GiB，1e7 边，out-degree 4，各自稳定值）

`cargo run --release -p gtv-accelerator-cuda --example bench_residency -- 10000000 topology`

| 列 | 列数 | 字节 | 留在主机 | B/edge | 上载 | 带宽 | 装得下的边 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 全部五列 | 5 | 230 000 004 | 0 | 23.00 | 29.2–31.1 ms | 7.5–8.1 GB/s | 224M |
| 去掉 label | 4 | 210 000 004 | 20 000 000 | 21.00 | 25.4–26.4 ms | 8.1–8.5 GB/s | 245M |
| 只有拓扑 | 2 | 50 000 004 | 180 000 000 | 5.00 | **5.9–6.4 ms** | 8.0–8.8 GB/s | **1.03B** |

三次带宽都落在 7.5–8.8 GB/s 的同一档（差别在噪声内）：**省下的时间是省下的字节，不是传输效率变好了**。4.6× 的容量、约 5× 的上载，代价是零——这三列本来就躺在调用方的内存里。

#### 代价，说清楚

- 投影不是免费的午餐：`temporal` 单独占全量的 **69.6%**，反过来看就是"时间过滤拿不到任何投影收益"。一个既漫步又按 `t` 过滤的算子要么用全量拷贝，要么先投影走拓扑、再为时间列付第二次 16 B/edge。
- 两份拷贝可以同时驻留（`(snapshot, projection)` 键），代价是 5 + 23 = 28 B/edge，比一份多 5 B/edge。只有两类算子都反复跑时才划算；只跑一类时，冷的那份会被 LRU 换出去，这是对的默认。

---

### 9.6 水平分层：数据大于显存时怎么办（设计，未实现）

第二问是"超过 8 GB 之后还能不能留在 GPU 路径上"。参考 [Strata](https://github.com/Niko1221/Strata)（llama.cpp 分支的推理引擎：模型大于显存时，权重留在 mmap 文件/页缓存里，VRAM 只放当前层的热切片，CPU lane 兜底，并有 spill guard 监控是否真的在溢出）。它的做法能搬，但**先要有一个可以换入换出的单位**——今天这个单位是"整张快照"，而"整张快照放不下"正是问题本身。

#### 一一对应的那几层

| Strata（LLM） | 这里（时序图） | 现状 |
|:--|:--|:--|
| 权重 mmap 文件 + 页缓存 | `GTVSeg` 的 mmap 段（已经是 mmap） | 有 |
| 主机侧 pinned staging | 段 → 主机 pinned slab 池 | pinned 已测（§9.3）：端到端慢 9%，**不做**，除非后面有异步流配合 |
| VRAM 里"当前层的切片" | 按 `node_range` 切的列 slab | **缺** |
| 层间预取（算第 k 层时拉第 k+1 层） | hop k 算完前把 hop k+1 的 frontier slab 入队 | **缺**（也缺 kernel） |
| 量化（int4 权重） | `dst` u32、时间列 delta-u32、`edge_type` u8 | 部分（`dst` 已能 u32；时间列要在写段时窄化，§9.3 测过 10.3 ms/1e7） |
| spill guard：溢出就换 lane | `Caps::admit` + 命中率/每 hop 时间在线判断，预测不划算就整条降级 CPU | 预检已有；在线判断**缺** |

#### 一条冷水算术：PCIe 与 CPU 扫描同价

本机实测的两条独立数字：

- CPU 扫一边 `valid_time_filter`：1e7 边 6.86 ms → **0.69 ns/edge**。
- PCIe：8.0 GB/s。拓扑投影 5 B/edge → 1e7 边 50 MB → **6.25 ms**。

**传一条边的 5 B，和 CPU 扫一条边，是同一个数量级，而且 PCIe 略贵。** 这不奇怪：8 GB/s ÷ 5 B = 1.6e9 edge/s，CPU 是 1.45e9 edge/s。所以"数据放不下就搬到 GPU"这个思路在图上不成立——除非搬过去的字节**被复用**。Strata 之所以成立，是因为一层权重在生成几百个 token 时被反复读；图漫步里对应的是"同一批节点在多个 hop / 多次查询里被重复访问"。这就是分层的全部前提，也是 spill guard 要测的东西。

#### 8 GiB 的实际墙在哪

budget 是显存的 60%（4.8 GiB），剩下 3.2 GiB 给驱动、上下文、kernel 工作区和结果缓冲——不是保守，是 §3.1 量出来的：

| 每边成本 | 形成方式 | 4.8 GiB 装得下 |
|:--|:--|--:|
| 23 B | 现状（u32 dst + 两个 i64 + u16） | 224M 边 |
| 9 B | 写段时窄化（u32 dst + u32 delta×2 + u8） | 573M 边 |
| 5 B | 拓扑投影 | 1.03B 边 |
| 1 B | 只放 `dst`，`offsets` 按需从主机取 | 5.1B 边 |

也就是说：**10 亿边以内，投影就够了，不需要分层**；10 亿边以上才需要 slab。这个顺序很重要——先做便宜的（投影，已完成），再谈复杂的（slab + 预取），因为 slab 化只能把"装不下"变成"换入换出"，而换入换出的价格刚刚算过：和 CPU 扫描同价。

#### 要做的话，顺序是

1. **P1 slab 化**：`ResidentSlab { snapshot, columns, node_range }`，键三元组，复用现有 `Lru`（它本来就是泛型的，`Arc` 就是 pin）。`load_slab` 只传 `offsets[node_range]` 覆盖的边区间——段内 CSR 已经是按 src 排好的，切一个节点区间是一段连续字节，不需要重排。
2. **P2 两流预取**：hop k 的 kernel 在 stream A 上跑，hop k+1 的 slab 在 stream B 上 `cuMemcpyHtoDAsync`（驱动已接好这个入口），用 event 而不是 `cuCtxSynchronize` 做交接。§9.3 已经证明 pinned 单独用是亏的（fill 14.1 ms > DMA 省下的 11.4 ms），但**配合异步重叠**才有意义：被隐藏的是整个 hop 的延迟，不是一次拷贝。
3. **P3 spill guard**：每个 slab 记 `(命中率, 每 hop 时间)`，与 CPU 的 `0.69 ns/edge` 比。预测不划算就整条查询降级——`ExecutionPolicy` 已经有 `OverBudget` / `AfterError` 两条通道，只缺"预测"这一路。
4. **P4 窄化**：把时间列在**写段时**写成 u32 delta（写路径一次，读路径永远是 9 B/edge），这是 2.5× 容量，且和 slab 正交。

前三条都要 kernel 才测得准（没有 kernel，slab 只是"更小的上载"）。**所以 §9.5 是现在能做的全部，§9.6 是 C2 之后的第一件事。**

---

### 9.7 C2：flat 批量 KNN（已交付）

§9 的门禁是「vs CPU flat ≥ 2×（端到端、含摊销后的首次上传），top-k 集合相等」。实测 **3.5×（batch=1）到 14.3×（batch=32）**，集合相等，盈亏点 5 次。

#### 交付物

| 文件 | 作用 |
|:--|:--|
| `kernels/knn_flat.cu` | 内核源码：一块处理一个 query，8 个 warp 各持一个 shared 候选堆 |
| `kernels/knn_flat.ptx` | 入库的 `compute_80` PTX（25 KB，ASCII，带源码 BLAKE3 章） |
| `src/knn.rs` | 模块加载、`MAX_K`、split 启发式、`Launch` 参数表 |
| `src/driver.rs` | `cuModuleLoadDataEx`（带 JIT 错误日志）、`cuLaunchKernel`、`cuMemcpyDtoH`、event 三件套 |
| `src/lib.rs` | `load_vectors` / `knn_flat` / `release_vectors`、`VectorLoadMetrics`、`KnnMetrics` |
| `tests/device.rs` | 三种 metric 的 top-k parity、过滤、退化输入、上传账目、失效 |
| `tests/ptx_stamp.rs` | 无 GPU 也能跑：入库 PTX 是否由入库 `.cu` 编译而来 |
| `examples/bench_knn.rs` | parity 闸门 → 冷/常驻 → 批量曲线 → 阶段表 → 盈亏点 |
| `crates/xtask` | `cargo xtask cuda-ptx [--check]`，唯一需要 toolkit 的一步 |

#### 为什么是入库 PTX，而不是 NVRTC

§7.1 选的是「预编译 PTX 入库」，理由是 CUB 头文件过 NVRTC 很脆。C2 的内核**一个 CUB/头文件都不用**（只有 `float4` 和算术），所以那条理由不成立，需要重新判一次。重判的结论仍然是入库 PTX，但理由换成了更硬的一条：

| 方案 | 构建期 | **运行期** | 结论 |
|:--|:--|:--|:--|
| `build.rs` + `nvcc` | 需要 toolkit | 不需要 | 无 GPU 的 CI 直接挂，否决 |
| NVRTC 运行时编译 | 不需要 | **需要 `libnvrtc`** | 只装驱动、不装 toolkit 的机器/容器是常规部署，那里 C2 会静默不可用 |
| 入库 PTX（`include_str!`） | 不需要 | 只要驱动（驱动自己 JIT PTX） | **采纳** |

代价是"制品可能与源码漂移"。这一条用机械手段堵住：`cargo xtask cuda-ptx` 把 `.cu` 的 BLAKE3 写进 PTX 头部，`tests/ptx_stamp.rs` 在每次 `cargo test` 重算——**在没有 GPU 的 CI 上，这是关于内核唯一能被验证的东西**。

一份 `compute_80` 覆盖 Ampere 及之后（PTX 向前兼容）；不兼容的只有"比生成它的 toolkit 更老的驱动"，那时 `cuModuleLoadDataEx` 报 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`，用那台机器的 toolkit 重跑一次 xtask 即可。

#### 内核形状

```
grid  = (batch, splits)      一块 = 一个 query 的一段行
block = 256                  8 warps x 32 lanes
heap  = WARPS x k x 12 B     每个 warp 一个升序候选堆，放 shared
```

四条设计决定，每条都对应一个会出错的方式：

1. **每个 warp 一个堆，不是一个线程一个、也不是一块一个。** 线程级要 `k` 个寄存器（k=100 放不下）；块级要么一把锁、要么原子。warp 级是中间那个：8 个独立堆，每个用 `__ballot_sync` + leader 选举在 warp 内串行，**没有锁**。
2. **`splits` 维度存在，是因为一块扫一百万行只用一个 SM。** 主机按 SM 数和 batch 定 `splits`（目标 4 块/SM，上限 64），kernel 留下每块的有序部分表，主机侧 k 路合并。batch=1 时实测 8 块 0.50 ms、24 块 0.30 ms——这就是它存在的理由。
3. **ties 按 id 升序，和 CPU 的 `cmp_scored` 一致。** 否则同一距离的两行谁在前面取决于调度，"top-k 集合相等"就变成掷硬币。
4. **`float4` 路径的门是 `dim % 4 == 0`，不是 `i + 4 <= dim`。** 一行从 `row * dim * 4` 字节开始，`dim` 不是 4 的倍数时行首只有 4 字节对齐，向量化读是 UB——而且只在"行足够多"时才炸。这个 bug 在探针阶段被 `DIM=130` 抓到。

`k` 的上限 256 是 shared 算出来的（8 x 256 x 12 B = 24 KB，一块不额外申请就是 48 KB）。它不是"没实现"，所以既进 `Caps::max_k` 让 planner 提前避开，也在 `knn_flat` 里返回 `LimitExceeded`（`Fallback::Cpu`）——CPU 参考没有这个上限。

#### 实测（RTX 4060 Laptop 8 GiB，budget 4.8 GiB，release，三次取最小）

`100 000 x 128`，k=10，51.2 MB 语料，上传 6.8 ms（7.91 GB/s）：

| batch | CPU ms | GPU ms | 加速 | GPU µs/query |
|--:|--:|--:|--:|--:|
| 1 | 1.402 | 0.398 | 3.5x | 397 |
| 2 | 3.017 | 0.404 | 7.5x | 202 |
| 4 | 6.390 | 0.526 | 12.1x | 132 |
| 8 | 12.786 | 0.961 | 13.3x | 120 |
| 16 | 24.939 | 1.777 | 14.0x | 111 |
| 32 | 49.195 | 3.437 | **14.3x** | 107 |
| 64 | 99.752 | 7.569 | 13.2x | 118 |

`1 000 000 x 128`（512 MB，超过 L2 也不影响结论），上传 65.0 ms：

| batch | CPU ms | GPU ms | 加速 |
|--:|--:|--:|--:|
| 1 | 10.538 | 3.960 | 2.7x |
| 8 | 103.374 | 8.868 | 11.7x |
| 32 | 316.780 | 34.241 | 9.3x |
| 64 | 634.679 | 76.379 | 8.3x |

阶段表（100k，batch=64）：**kernel 99.1%**（7.493 ms），upload 0.2%，download 0.5%（15.9 KB），merge 0.1%。传输在这里彻底不是问题了——这是 §9.5 的投影 + `k·12 B` 的回传共同的结果，也是 §8 第 13 条（必须报回传字节）的价值：它现在是 0.5%，一年后如果变成 30%，表里会写着。

盈亏点：**5 次**（100k）/ **7 次**（1M）。契约里的 2× 在 batch=1、把上传全部算进去时就已经达到（3.5×），摊销后是 13–14×。`ExecutionPolicy::min_knn_work = 1e5` 的默认值因此仍然成立——它拦的是"行数 x query 数太少"，而那种情形下上传确实吃不回来。

#### 两个必须记下来的坑

1. **PTX 里不能有非 ASCII 字节，注释也不行。** 生成的头里放了一个 em dash（U+2014），`cuModuleLoadData` 报 `a PTX JIT compilation failed`，没有任何行号。修法三层：xtask 头部改 ASCII、xtask 读回时校验 `is_ascii()`、`knn.rs` 单测直接断言 `PTX.is_ascii()`。顺带把 `cuModuleLoadData` 换成 `cuModuleLoadDataEx` + `CU_JIT_ERROR_LOG_BUFFER`，以后这类失败会带行号。
2. **内核模块挂在 context 上，context 会被 `Drop` 拆掉。** 模块是进程级 `OnceLock`（一次 JIT，不是一次一个），但它属于设备 0 的 primary context；第一个 `CudaAccelerator` drop 时 `cuDevicePrimaryCtxRelease` 把引用计数降到 0，context 销毁，缓存的模块句柄变成悬空——下一次 launch 报 `CUDA_ERROR_INVALID_HANDLE`。修法：**一旦加载过内核，就保留这次 context retain**（context 升级为进程级），代价是退出时泄漏一个 context。`tests/device.rs::a_second_accelerator_reuses_the_loaded_kernel` 就是这个 bug 的回归测试。

#### 数值一致性：能保证什么，不能保证什么

整数路径逐位相等；浮点路径 `|Δ| ≤ 1e-5·max(1,|d|)`。**top-k 集合相等有前提**：参考侧第 k 与第 k+1 名距离之差要大于 `2 x 容差`，否则 f32 的求和顺序（GPU 按 4 个一组、CPU AVX2 按 8 个一组）就能合法地翻转名次。测试把这个前提写成了断言（先跑 `k+1` 拿到间隙，间隙不够就报"语料在边界有并列"），所以"集合不等"永远不会是"语料恰好有并列"的马甲。实测在 `dim=512, k=100` 时确实出现过这种边界并列（第 49 名差 7.7e-6），两条距离都在容差内——这不是 bug，是 f32 的极限，写进契约比藏起来好。

#### 已知的下一根杠杆（不在 C2 范围内）

单 query 的带宽只有 **129 GB/s**（1M x 128），而这张卡的 DRAM 是 272 GB/s。原因是"一线程一行"的访存形态：一条 warp load 命中 32 个不同的 32 B sector，每次只用 16 B。换成"shared 转置 + 8 lane 一行 + 3 次 shuffle 归约"能贴到 DRAM 上限，大约 2×。它只影响 batch ≤ 2（那里已经赢 2.7–3.5×），而 batch ≥ 4 时跨 query 的 L2 复用把有效带宽推到 430–490 GB/s，已经不是瓶颈——**所以先记下来，不动手**。

---

## 10. 风险

| 风险 | 影响 | 缓解 |
|:--|:--|:--|
| 笔记本 GPU 与生产卡差异大 | 本机结论不可外推 | 所有数字标机器；生产卡上重跑 C0 |
| 225 GB/s 已是带宽墙 | 加算力无用，只能减字节 | TC4 走量化 / 更小候选；不追 FLOP |
| PTX 需要按 arch 编 | 新卡要重编 | PTX JIT 向前兼容；xtask 一条命令 |
| 驻留缓存与 GC/Tag 交互 | 悬空引用 | 驻留期间 pin；失效测试（§4.3） |
| GPU 路径难调试 | 错误难定位 | Compute Sanitizer + CPU reference 逐项对照；kernel 只做纯函数 |

---

## 11. 复现

```bash
# C0 探针（需要 CUDA toolkit 与 MSVC，产物不入仓）
cd tools/cuda-probe
nvcc -O3 --use_fast_math -Xcompiler "/O2 /arch:AVX2" -arch=sm_89 -o probe.exe probe.cu
./probe.exe
```

Windows 上若 `cl.exe` 不在 PATH，先 `vcvars64.bat`，或给 `nvcc` 传 `-ccbin <MSVC bin dir>` 并设置 `INCLUDE` / `LIB`。

C1 之后的阶段测量不需要 toolkit，只需驱动：

```bash
# 上载内部：逐列阶段表（wide / narrow / stage）
cargo run --release -p gtv-accelerator-cuda --example bench_residency -- 10000000 narrow
# pageable vs pinned 端到端 A/B
cargo run --release -p gtv-accelerator-cuda --example bench_residency -- 10000000 pinned
# 投影：五列 / 去 label / 只有拓扑，三种拷贝的字节、时间、容量
cargo run --release -p gtv-accelerator-cuda --example bench_residency -- 10000000 topology
# A–D 全链路：真实 SegmentStore → graph_at → view → upload
cargo run --release -p gtv-engine --example bench_graph_load -- 10000000
cargo run --release -p gtv-engine --example bench_graph_load --features cuda -- 10000000
```

三类数字都要落在报表里：**传输 / kernel-only / 端到端**，缺任何一项的结论不接受。第四条从本次起加进去：**materialize**——端到端里不含它，报出来的永远是"传输占 97%"，而那个 97% 只对整条路径的 2.5% 成立。C2 起再加第五条：**阶段表**（upload / kernel / download / merge 各占多少），因为 kernel 占 99.1% 和传输占 97% 是两条完全不同的优化方向。

C2 的内核多一步构建期依赖，其余不需要 toolkit：

```bash
# 只有这一条要 toolkit（nvcc + MSVC），产物 `kernels/*.ptx` 入库
cargo xtask cuda-ptx            # 生成并写源码 BLAKE3
cargo xtask cuda-ptx --check    # 只校验：入库 PTX 是否由入库 .cu 编译而来
# kernel 侧：parity 闸门 → 冷/常驻 → 批量曲线 → 阶段表 → 盈亏点
cargo run --release -p gtv-accelerator-cuda --example bench_knn -- 100000 128 10
cargo run --release -p gtv-accelerator-cuda --example bench_knn -- 1000000 128 10
```

`bench_knn` 第一段是 **parity 闸门**（三种 metric 的 top-k 与 CPU 逐个比），闸门不过就不计时——一个跑得很快但答案错的 kernel 没有加速比可言。

`bench_graph_load` 会打**两行** `graph_at`：一行是未压实的（归并），一行是它自己 `spawn_compaction` 之后的（原样拷贝），并在两行之间逐列断言新路径等于行路径。§9.4 的两个数就是这两行——只报其中一行，报的是"这台机器上恰好有几个段"，不是材料化的成本。

---

## 12. 对评审意见的逐条回应

| 评审条目 | 结论 | 落在哪 |
|:--|:--|:--|
| §1.1 不要把 `neighbors()` 热循环搬 GPU | **接受** | §1 硬边界表；单点 `neighbors` 列入「永不上 GPU」 |
| §1.2 CUDA 入口在 Materialized Snapshot 之后 | **接受** | §1 数据流图；`live_rows_at()` 明确留在 CPU |
| §1.3 双时间描述不一致，会固化错误 Kernel API | **接受，已修文档** | 设计文档 §4.2 重写；本条是评审中最有价值的发现，因为它改的是前提不是结论 |
| §2 P0 传输探针升级为门禁 | **接受并已执行** | §3 全部为本机实测；§8 第 2–4 条把它写进准入 |
| P1 TC4 KNN | **接受，补一条限定** | §6.1：只做 flat 全扫 + 批量 query；IVF（3.1% 候选、亚毫秒）留 CPU |
| P2 TC2 OFI | **接受** | §6 表；`R*` 预计 <1，但要求实测，不从点积外推 |
| P3 批量 Valid-Time 过滤 | **接受，加「链式」前提** | §3.2 实测单独用必亏；§6 C4 要求与下游算子串起来、不回传位图 |
| P4 批量 k-hop / BFS | **部分接受：下调** | §6.2：N=32–128 时 CPU 已 268–316 µs，无赢面；门禁为 `seeds ≥ 1e4` 或 frontier ≥ 1e6。评审「迟早要有」的判断对，但「迟早」不进 C3 之前的里程碑 |
| P5 motif 候选生成 | **接受，范围收缩** | §6 C6：只生成候选，时间序 / 边类型序列 / 语义验证留 CPU |
| P6 批量统计构建 | **接受，降为融合项** | 不单独立项，只在 flush / compaction 已扫描全段且数据已常驻时顺带算 |
| P7 Compaction prepare | **部分接受：下调** | §6.3：输入在磁盘上，起手 13 ms 传输；依赖 C1 驻留基础设施，不能单独做 |
| §3 不建议 CUDA 的项 | **全部接受** | §6.4 / §1 硬边界表，逐项列出理由 |
| §4 Rust host + CUDA C++ | **接受，补落地方式** | §7：`xtask` 预编译 PTX + `cudarc` 加载，避免默认构建依赖 toolkit；CUB 可用，但不进 NVRTC |
| §5 C0–C6 路线图 | **接受，调整为 C0–C7** | §9：C1 由「设备抽象」细化为「常驻句柄 + 失效路径」，并把 compaction 拆到 C7 |
| §6 十条准入规则 | **接受，补四条** | §8 第 11–14 条：失效测试、显存预算、回传字节数、CPU-only 对照 |

### 第二轮评审（12 条：阶段计时 / 缓存键 / SoA / pinned / 增量）

| 评审条目 | 结论 | 落在哪 |
|:--|:--|:--|
| §1 给 `load_graph` 加 `GraphLoadMetrics`，测 A–G 七段 | **接受并已执行** | `gtv-accelerator/src/metrics.rs` 的 `GraphLoadMetrics`（逐列 + 九类字节计数）；`bench_residency` 打阶段表；`bench_graph_load` 打 A–D。**结论与评审的猜测一致：H2D 不是最大单项**（§9.3） |
| §1 检查隐性多次复制 | **接受，已证伪上载侧、证实 materialize 侧** | 上载侧 `host_pack_ns`/`layout_transform_ns`/`staging_bytes` 三项结构上为 0 并有测试守护；materialize 侧确有 AoS 往返，1081 ms / 686 MiB（§9.3） |
| §1 "SoA → AoS → SoA 是危险实现" | **接受，正是 `graph_at` 的实现** | `store.rs:1503`：`live_rows_at` → `Vec<EdgeRecord>` → `to_csr`。列为 C2 前置项 |
| §2 不按 `business_t` 重传 | **已是现状** | `ResidentGraph` 只绑 `snapshot_id`（§4.1）；`residency.rs` 有测试断言 `business_t` 不入键 |
| §2 `ActiveEdgeKey(snapshot, t, endpoint_mask)` | **部分接受** | `Sel` 已携带 `snapshot_id` 并跨快照报 `SnapshotMismatch`；`endpoint_mask` 待 C4（C1 视图只有出边） |
| §3 GPU 侧保持 SoA，不要 `DeviceEdge` | **已是现状** | 全仓无 AoS 设备结构；`load_graph` 五列逐列 `alloc` + `memcpy` |
| §4 `DeviceUploadView` 构造必须 O(1) | **已是现状，并加了断言** | `SnapshotView::from_csr` 借用 `csr.columns()`；`bench_graph_load` 断言 < 1000 µs（实测 0.1–0.3 µs） |
| §5 pinned 池 / 端到端 A/B | **A/B 已做，否决** | §9.3：pinned 纯拷贝快 59%，但 `fill` 14.10 ms 抵掉 DMA 收益，端到端 **慢 9%** |
| §6 多 stream 并行上传 | **暂不做** | 评审自己给的保留意见（共享 PCIe）就是理由；且 `cuMemcpyHtoD` 在本机已跑满 7.5–8.0 GB/s，没有可重叠的空隙。等有 kernel 能与上载重叠时再测 |
| §7 按算子上传最少列 | **已实现（§9.5）** | 不再"推迟到 C2"：投影是上载路径的决定，不是 kernel 的决定。`ColumnSet` + `(snapshot, projection)` 缓存键 + `Caps::admit` 预检。实测 1e7 边：29–31 → **5.9–6.4 ms**，容量 224M → **1.03B 边（4.6×）** |
| §8 段级增量设备缓存 | **接受其顺序建议** | 与本文一致："先整图常驻与精确计时，再按更新频率决定"。本次测量把这一条的门槛量化了：materialize 修不好时，增量上载省下的 30 ms 毫无意义 |
| §9 时间列 i64 → u32 delta、edge_type u16 → u8 | **接受，优先级上调** | 152 MiB → 76 MiB，按实测带宽折 **10.3 ms**，是 `dst` 窄化（5.07 ms）的两倍。仍须"写入段时窄化"，查询前转换是 §9.2 已经证伪的做法 |
| §10 不要优先 Unified Memory / zero-copy | **接受** | 与 §6.4 一致，未纳入路线 |
| §11 P0–P4 实施顺序 | **接受，插入一条 P0.5；P0.5 已完成** | 顺序改为：P0 测量 → **P0.5 修 materialize（AoS 往返，已完成，§9.4：1 579 → 176–393 ms）** → P1 消除查询级重复 → P2 重叠 → P3 减字节 → P4 增量 |
| §12 阶段 C1 Profiling 的硬断言 | **大部分已落地** | 同快照不重传、`business_t` 不触发重传、快照变更失效、CUDA 失败回落 CPU 均有测试；"不得按 edge_count 分配 Vec"由 `from_csr` 的 O(1) 借用 + 断言保证 |

### 评审未提、但本文新增的一条

**驻留对象的键必须把 `business_t` 排除在外**（§4.1）。图结构只随快照变化，活跃位图才随业务时间变化。如果实现时把 `t` 拼进图缓存的键，「同一快照换一个 `t`」就会触发整图重传——而 CPU 路径在这件事上是免费的。这是把 CUDA 做错的最高频方式，故写进设计而不是留到编码时决定。
