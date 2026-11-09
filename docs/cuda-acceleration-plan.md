# CUDA 加速方案（C0 已实测）

> 版本：2026-11-09 ｜ 前置文档：[`design-review-2026-11.md`](design-review-2026-11.md)
> 触发：第三方评审意见（§1–§6）＋ 本机 C0 探针实测
> 状态：**C0 已交付**（探针 + 结论），C1 起未开工。本文只定义边界与准入门槛，不含 kernel 实现。

---

## 0. 结论先行

三条，按重要性排序：

1. **GPU 在本项目的唯一可行形态是「设备常驻」，不是「每次查询传一次」。** 实测：1e7 边的图（153 MiB）单向上传 **13.3 ms**，而 CPU 全量 valid-time 扫描只要 **6.9 ms**。冷路径上 GPU 一定输，且 96% 的时间花在传输上。
2. **因此 C0 的交付物不是 kernel，而是「驻留句柄 + 失效路径 + 容量预算」。** 没有这一层，后面每个 kernel 的加速比都会被传输吃掉。
3. **评审给出的优先级我大部分接受，但有两处要下调**：批量 k-hop（P4）在 Q1 当前规模（N=32–128）上不成立；Compaction prepare（P7）在数据未常驻时同样不成立。两处都加了量化门禁，而不是直接排进里程碑。

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
| C2 | TC4 flat batch KNN（向量常驻） | vs CPU flat ≥ 2×（端到端、含摊销后的首次上传），top-k 集合相等 |
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

三类数字都要落在报表里：**传输 / kernel-only / 端到端**，缺任何一项的结论不接受。

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

### 评审未提、但本文新增的一条

**驻留对象的键必须把 `business_t` 排除在外**（§4.1）。图结构只随快照变化，活跃位图才随业务时间变化。如果实现时把 `t` 拼进图缓存的键，「同一快照换一个 `t`」就会触发整图重传——而 CPU 路径在这件事上是免费的。这是把 CUDA 做错的最高频方式，故写进设计而不是留到编码时决定。
