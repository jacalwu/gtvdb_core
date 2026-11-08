# gtvdb 存储层设计：从磁盘字节到 CPU 缓存的零拷贝路径

> 目标：让数据从持久化介质**直接映射**成可查询的内存结构，并在遍历时贴合 CPU 缓存行为。
> 本文基于 2026-11-08 的代码事实（见「证据」列），是对 `docs/storage_design.md` 的**落地补完**，不是替换。
> 姊妹篇：[docs/oceanbase-storage-lessons.md](oceanbase-storage-lessons.md) — OceanBase 压缩 / LSM-Tree 的适用性评估，为 §3.5 写路径、§6 路线与「热层不压缩」约束提供依据。
> 状态：**P0 读路径闭环 + P1 写路径闭环均已实现并通过全工作区测试**（2026-11-08）；P2–P3 仍为设计提案。实现清单、验据与相对本文的工程偏差见 §8（P0）/ §9（P1）。

---

## 0. 结论先行

当前 gtvdb 的存储**不是慢在磁盘 I/O，而是断在「格式不兼容」**：

- Parquet 是**压缩 + 编码的交换格式**，读取必须解码物化成 `RecordBatch`；
- `TemporalCSR` 是**纯 owned `Vec` 的内存结构**，没有磁盘格式、没有反序列化路径；
- 两者之间的桥是 `from_arrays()`：**抽列 → 组装 `EdgeRow` → `sort_unstable_by_key` → 拆回 5 个 Vec**，即每次启动都要全量重建。

所以「快速从存储映射进内存」目前**在原理上就不可能**——不是没调好，是没有可映射的东西。

**推荐方案**：引入一个**定宽、cache-line 对齐、不可变的段格式 `GTVSeg`**，让

```
磁盘字节布局 == 内存 CSR 布局 == CPU 缓存访问单元
```

Parquet 降级为**归档 / 跨系统交换格式**，不再是热查询路径。核心改造只有一处：把 `TemporalCSR` 的 `Vec<T>` 换成 `Deref<Target=[T]>` 的存储后端枚举，使全部算子代码零改动即可跑在 mmap 上。

---

## 1. 核心功能盘点（代码事实）

| 能力 | 实现位置 | 状态 | 与存储的关系 |
|---|---|---|---|
| 时序图索引 `TemporalCSR` | `crates/gtv-core/src/csr.rs:90-109` | ✅ 成熟 | **纯内存，不落地** |
| 半开区间 + 二分 + zone map | `csr.rs:326-338`（手写二分）、`:31` `CHUNK_SIZE=64`、`:177-196` | ✅ | zone map 仅 `max_valid_to` 单值 |
| 方向优化 BFS / transpose | `csr.rs:481-575`、`:687-712` | ✅ | transpose 是**全量重建** |
| Arrow ↔ Parquet 读写 | `gtv-storage/src/parquet.rs:57-77` | ✅ | ZSTD 默认压缩（`:41`），row group 256k（`:22`） |
| mmap 读 Parquet 原始字节 | `gtv-storage/src/hdb.rs:402-417` | ⚠️ 半程 | `Bytes::from_owner(mmap)` 后 **parquet-rs 解码即断链** |
| 原子发布 `temp→fsync→rename→fsync(dir)` | `gtv-storage/src/atomic.rs:31-46` | ✅ 可复用 | 已具备 crash-safe 段发布能力 |
| kdb 风格 HDB 分区 + symbol 字典 | `gtv-storage/src/hdb.rs:64-105,214-267` | ✅ | 分区剪枝思路可直接沿用 |
| 向量索引持久化 `.gtvidx` | `crates/gtv-index-store/src/{container,store}.rs` | ✅ | `fs::read` 整文件 + 反序列化（**非 mmap**），容器格式设计可复用 |
| 内存时间旅行 `SnapshotStore` / `BitemporalStore` | `gtv-storage/src/{snapshot,bitemporal}.rs` | ✅ | `HashMap<String, TableVersions>`，纯内存 |
| 量子状态来源森林 | `gtv-core/src/{provenance,quantum_state}.rs` | ✅ | 需纳入持久化范围（当前只有 JSONL） |
| **WAL** | — | ❌ **不存在** | `ROADMAP.md:18` M6、`README.md:172` 均列为缺 |
| **CSR 磁盘格式 / mmap CSR** | — | ❌ **不存在** | `docs/assessment-2026-11.md:78-80` 已明确判定 |
| **CPU 缓存层设计** | — | ❌ **不存在** | 全库无 cache-line 对齐、Eytzinger、大页读路径 |

一句话：**计算内核（时序 CSR、ANN、SIMD 窗口算子）是强的，存储层是缺失的**。这与路线 A「嵌入式时序图 + 向量分析引擎」的定位冲突——嵌入式引擎要求「打开即用」，而现状是「每次启动重编译索引」。

---

## 2. 断点诊断：为什么现在到不了 CPU 缓存

```
[磁盘 Parquet]  ──ZSTD 解压──▶ [编码页]
      │                            │
      │  parquet-rs 解码(物化)      │  ① 压缩 = 必须解码，zero-copy 不可能
      ▼                            ▼
[RecordBatch]  ──抽列/拷贝──▶ [Vec<u64> ×2 + Vec<i64> ×2 + Vec<u16>]
      │                            │
      │  from_arrays()              │  ② 第二次全量拷贝
      ▼                            ▼
[Vec<EdgeRow>] ──sort O(E log E)──▶ ③ 第三次拷贝 + 全量排序
      ▼
[TemporalCSR: 5 个 owned Vec]  ────▶ ④ 无法 mmap，进程退出即消失
```

四个断点，逐个说明：

1. **压缩与 zero-copy 互斥**。`parquet.rs:41` 默认 ZSTD；即使 `hdb.rs:408` 用 `Bytes::from_owner(mmap)` 跳过了「文件→用户缓冲区」这一次拷贝，`ParquetRecordBatchReaderBuilder` 仍会解码/解压并物化 `RecordBatch`（`hdb.rs:409-415`）。零拷贝只覆盖了第一段。
   - **精化**（见 `oceanbase-storage-lessons.md` §2）：这个互斥对**语义编码**（字典 / RLE / delta + bit-packing）同样成立，但理由不是「需要解压」——OceanBase 的 encoding 数据可以随机访问、可以不解压直接过滤。真正的问题是**编码后不再是定宽 POD 数组**，且访问时需执行解码指令、字典查表会打断顺序预取，因而无法 `cast_slice` 成 `&[u32]`。判断准则：任何改变元素位宽或引入间接引用的技术都不能进热层，无论它叫压缩还是编码。
2. **数据形态不兼容**。Parquet 是「行组 + 列块 + 字典/RLE」，CSR 是「按 `(src, valid_from, valid_to, dst)` 排序的并行数组」。两者不是同一个字节序。
3. **`from_arrays` 是构建器不是加载器**。`csr.rs:138-152` 先物化 `Vec<EdgeRow>` 再 `sort_unstable_by_key`。2.5M 边（`bench_traversal.rs:25`）的排序 + 5 次分配在启动路径上，无法摊掉。
4. **`.gtvidx` 同理**。`store.rs:264` `fs::read` 整文件入 `Vec<u8>` → `decode`（含 `payload.to_vec()`）→ `from_bytes`。索引越大，启动越慢。

另外两个**缓存层**问题（现有代码没做错，只是没做）：

- `CHUNK_SIZE = 64` 的单位是**边数**（`csr.rs:31`）。以当前 26 B/边估算（`u64 + i64 + i64 + u16`），一个 chunk ≈ 1664 B ≈ 26 条 cache line，与 64 B 缓存行**没有对齐关系**，zone map 粒度是经验值而非硬件粒度。
- `valid_from_upper_bound`（`csr.rs:326-338`）是标准二分，每一步都是随机跳转；高出度节点（如 `bench_csr_adaptive.rs:24` 的 100 万度单源）会产生约 20 次跨 cache line 的 miss，且硬件预取器对这种跳跃无能为力。

---

## 3. 推荐设计

### 3.1 三层职责（关键：把 Parquet 从热路径移走）

| 层 | 介质 | 格式 | 加载方式 | 用途 |
|---|---|---|---|---|
| **L0 热** | 常驻 RAM | GTVSeg（不可变） | 启动时 mmap + `MADV_WILLNEED` + 大页 | 近期时间分区，在线分析 |
| **L1 温** | OS page cache | GTVSeg | 按需 mmap，LRU 管理 | 历史分区，首次访问触发 page fault |
| **L2 冷** | 磁盘 | **Parquet** | DataFusion `ListingTable` | 归档、跨系统交换、>内存数据集 |

**原则**：查询只走 GTVSeg；Parquet 只在 ingest（外部数据进来）和 archive（段淘汰）时出现。`hdb.rs` 的 `date/table/symbol` 分区与分区剪枝逻辑完全保留，只是分区内的文件换成段。

### 3.2 GTVSeg 字节格式（v1）

一个段 = 一个时间分区内、一个不可变文件。所有多字节字段 **little-endian**，所有列区 **64 B 对齐**（正好一条 x86-64 cache line）。

```
offset 0     ┌────────────────────────────────────────────┐
             │ Header (固定 4096 B)                        │
             │  magic        "GTVSEG\0\0"  [u8;8]          │
             │  version      u16 = 1                       │
             │  flags        u16  (窄化/压缩/校验类型)      │
             │  node_count   u64                           │
             │  edge_count   u64                           │
             │  chunk_size   u32  (边数/块)                │
             │  col_dir[8]   { offset:u64, len:u64,        │
             │                 elem:u8, reserved[7] }     │
             │  time_bounds  { max_valid_from:i64,         │
             │                 min_valid_to:i64 }          │
             │  checksum     blake3 [u8;32]                │
             └────────────────────────────────────────────┘
offset 4096  ┌─ offsets              u32  (node_count+1)  ┐  ← 最热，单独一页
             ├─ chunk_index          u32                   ┤
             ├─ chunk_max_valid_to   i64                   ┤  ← zone map 热数组
             ├─ dst                  u32                   ┤
             ├─ valid_from           i64 / u32(delta)      ┤
             ├─ valid_to             i64 / u32(delta)      ┤
             ├─ edge_type            u16                   ┤
             └─ (可选) provenance / vector sidecar          ┘
```

设计约束（这些是「能 mmap 」的必要条件，缺一不可）：

- **不压缩**。整段是定宽 POD 数组，`mmap` 后经 `bytemuck::cast_slice` 直接得到 `&[u32]` / `&[i64]`，零解码。压缩留给 L2 Parquet。
- **不可变**。写入永远生成**新段**（copy-on-write），发布走现成的 `write_atomic`（`atomic.rs:31-46`）。这样读侧无需锁、无需担心映射被并发改写。
- **校验与版本**。magic + version + blake3 沿用 `.gtvidx` 容器（`container.rs:14-32`）已验证过的做法，加载时校验一次，避免映射损坏文件造成 UB。
- **对齐检查**。`cast_slice` 前必须断言 `ptr as usize % align_of::<T>() == 0` 且 `len % size_of::<T>() == 0`；Header 定长 4096 保证所有列起点 64 B 对齐。
- **字节序**。固定 LE；大端平台走一次字节交换（当前目标平台 x86-64 无需）。

**窄化**（可选，flags 标记）：节点数 < 2³² 时 `dst: u64 → u32`；时间戳改为相对分区起点的 `u32` delta（秒/毫秒精度）。每边从 26 B 降到 **16 B**，一条 64 B cache line 从装 2.4 条边提升到 4 条。这是收益最直接的缓存优化。

**Skip Index 预留（P2 落地，P0 只留槽位）**：`col_dir[8]` 目前未用满，Header 的 `time_bounds` 也还有空间。建议为分层聚合元数据预留定宽列，使 `as_of(t)` 能按「段级 → src 级 → chunk 级」逐层下钻（借鉴 OceanBase 的 microblock → macroblock → SSTable 上卷，见姊妹篇 §4③）：

| 层级 | 当前 | 建议补齐 |
|---|---|---|
| 段（≈SSTable） | `max_valid_from` / `min_valid_to` 2 个标量 | + `seg_min_valid_from` / `seg_max_valid_to` / `edge_count` |
| src（≈宏块） | **缺失** | `src_min_valid_from[s]` / `src_max_valid_to[s]`，可 O(1) 判定整个 src 在 t 时刻无存活边 |
| chunk（≈微块） | `chunk_max_valid_to` 单值 | + `min_valid_from` / `max_valid_from` / `min_valid_to` / `edge_count` |

这些元数据**本身是定宽的**，放进热元数据区后仍可 `cast_slice`、仍零拷贝，不破坏本节任何约束。它们直接增强 `csr.rs:734-736` 的 `zone_map_skips_expired_chunks`。P0 只需保证 `col_dir` 有空位与 `flags` 有标志位，**不实现**这些列。

### 3.3 CPU 缓存路径（你这次的核心诉求）

按**收益从大到小**排序，避免把花哨技术当银弹：

| 优先级 | 手段 | 解决什么 | 预期收益 |
|---|---|---|---|
| **P0** | mmap 直接映射，消除解码与重建 | 启动从 `O(E log E)` 排序 → `O(1)` open+mmap；内存从 3–4 份拷贝 → 1 份（页缓存即数据集） | **最大**，数量级 |
| **P0** | 大页 + `MADV_WILLNEED` | 减少 TLB miss 与首次遍历的 page fault 风暴 | 大（已有 `madvise_hugepages` 可复用，`hdb.rs:391-400`） |
| **P1** | 窄化边表示 26 B → 16 B | 同样 cache line 装更多边，顺序扫描带宽减半 | 大 |
| **P1** | `offsets` + `chunk_max_valid_to` 前置且独立对齐 | 这两组是每次查询必访的「热元数据」，不与冷边数据争 cache line | 中 |
| **P2** | chunk 粒度改为**按字节**：`chunk_bytes = 256 B`（4 条 cache line） | 让 zone map 剪枝跳过整数条 cache line，而非 26 条 | 中 |
| **P2** | **Eytzinger（BFS 序）布局的 `valid_from` 旁路索引** | 把二分的随机跳转改成预取友好的顺序布局，20 次 miss → 少数几条 cache line | **仅高出度节点显著**（平均度 50 时二分仅 ~6 步，收益有限） |
| **P3** | 混合 AoS：`#[repr(C, align(16))] Edge` 主数组 + SoA 时间列旁路 | 图遍历一次加载 4 条完整边 vs 跨 4 个数组 | 需实测；会动 `edge_slices()` API，谨慎 |
| **P3** | NUMA 分区、SIMD zone map 过滤 | 多_socket 机器 | 视硬件 |

**关于 AoS 的诚实权衡**：现有 SoA 对「只按时间剪枝」很省带宽（不加载 `dst`），对「遍历需同时取 dst + 时间」则是 4 条 cache line。AoS 反过来。因为 `csr.rs` 的主要算子是遍历，**P3 值得做 A/B benchmark**，但不该在 P0 冒险——它会改 `edge_slices()` 返回的四元组切片签名。

### 3.4 最小侵入的代码改造（这是能不能落地的关键）

不要给 `TemporalCSR` 加生命周期参数 `'a`——那会传染到 `TemporalGraphIndex` trait、所有 UDF 和 CLI。改用**存储后端枚举 + `Deref`**：

```rust
// crates/gtv-core/src/csr.rs（新增）
pub enum Col<T: Pod> {
    Owned(Vec<T>),
    Mapped { map: Arc<Mmap>, range: Range<usize> },
}
impl<T: Pod> Deref for Col<T> {
    type Target = [T];
    fn deref(&self) -> &[T] { /* cast_slice over map[range] */ }
}

pub struct TemporalCSR {
    offsets: Col<u32>,
    dst: Col<u64>,          // 窄化时为 Col<u32>，见 flags
    valid_from: Col<i64>,
    // ... 其余字段不变
    _segment: Option<Arc<Segment>>,   // 持有 mmap 生命周期
}
```

因为 `self.dst[e]` / `self.valid_from[mid]` 全部走 `Deref<Target=[T]>`，**`csr.rs` 里 1180 行算子代码一行都不用改**。新增两个构造入口：

```rust
TemporalCSR::from_arrays(...)        // 保留：owned，兼容现有 14 处调用
TemporalCSR::open_segment(path)      // 新增：mmap，O(1)，校验 magic/version/checksum
```

`.gtvidx` 索引同理可升级为 mmap 加载（`store.rs:264` 的 `fs::read` → `Mmap`），但优先级低于 CSR，因为索引体积通常小一个数量级。

**放哪个 crate**：建议**在 `crates/gtv-storage` 新增 `segment.rs` 模块**，不新建 crate。`docs/assessment-2026-11.md:170` 曾建议新建 `gtv-store`，但目录刚重构为「平台 14 crate + 扩展树」并加了 CI 边界守卫，再加 crate 会破坏刚稳定的结构；而 `gtv-storage` 已经持有 `memmap2`、`atomic.rs`、`hdb.rs`，职责正好吻合。`Col<T>` / `Pod` 放在 `gtv-core`（供 `gtv-index` 复用），`Segment` 的 I/O 放在 `gtv-storage`。

### 3.5 写路径（P1 已实现，实现细节与验据见 §9）

> 本节依据 `oceanbase-storage-lessons.md` 修订：吸收 OceanBase 的「写路径与压缩解耦」「宏块复用 / 渐进重写」，但**只读半套 LSM**。

```
ingest → WAL(append-only, 复用 atomic 语义) → MemSegment(原生定宽 Vec，零编码零压缩)
       → flush: 排序一次 → 写新 GTVSeg → write_atomic 发布 → 更新 meta
       → compaction:
            逐段检查 (time_boundary, checksum, version, system_ts_range)
            未受增量触及的段 → 直接复用，不重读不重写     ← 宏块复用
            受触及的段     → 多路归并生成新段
            格式升级       → 后台渐进重写，不停机          ← 渐进合并
       → 归档淘汰段 → Parquet（编码 + 压缩，仅在 L2）
```

WAL 格式可直接采用 `docs/storage_design.md:91-96` 的定义（`[txn_id:u64][op:u8][table_id:u32][partition_id:u32][payload_len:u32][payload]`），那份文档这部分是可直接用的。

**三条硬约束**（来自 OceanBase 适用性评估，详见姊妹篇 §4/§5.3）：

1. **写路径零编码零压缩**。OceanBase 敢用高压缩比 zstd 的前提正是「LSM 让压缩只发生在 compaction，不碰写路径」；我们借这个架构决策，但把产物换成定宽段。ingest 里出现任何编码逻辑都是设计错误。
2. **按段复用**。`hdb.rs:6,59` 的 `<root>/<date>/<table>/<symbol>` 时间分区天然是不可变宏块——历史日期片不会有新边（除非 bitemporal 的 system-time 修正触及）。绝大多数 compaction 只需重写活跃片，写放大从「全量重建」降到 O(活跃片)，直接解决 `csr.rs:687` `transpose()` 全量重建。这个收益应作为 P1 的显式验收指标（历史段复用率）。
3. **「LSM 写、CSR 读」——读侧必须退化成单一 CSR**。

第 3 点是对早期草稿的**修正**：原稿写「查询时 `base ∪ delta` 归并」，这对图遍历是错的。`edge_slices(src)`（`csr.rs:313`）返回的是一个连续区间，BFS 每扩展一个节点都要拿它；若改成多层查找 + k 路归并，邻接定位会从 `offsets[src]..offsets[src+1]` 的 **O(1)** 退化，且 delta 稀疏却要为每次遍历付固定成本。正确分界是：

| | 写侧（吸收 LSM） | 读侧（必须退化成 CSR） |
|---|---|---|
| 形态 | MemSegment + L0 delta 段 + L1 近期段 + L2 基线段 | **单一连续 CSR**，一个 `offsets[src]..offsets[src+1]` 区间 |
| 多层何时存在 | 仅在活跃写入期间 | compaction 完成后只留基线，delta 归零 |
| delta 组织 | 按 `src` 分区的小段，最多一次额外查找 | **禁止**把多路归并放进 `neighbor_iter` 热循环 |
| 版本融合 | 由 `BitemporalStore::as_of_system`（`bitemporal.rs:156`）在**版本查询**层做 | 图遍历不做版本融合 |

另外，**全局快照点可与 bitemporal 统一**：把 major compaction 的快照点定义为 `system_ts = T`，产出的基线 GTVSeg 就是 `as_of_system(T)` 的物化，「合并」与「时间旅行」共用一套版本语义，基线同时充当 checksum 校验锚点（OceanBase 也用 major compaction 做物理校验，但那里没有双时间轴的业务含义）。

**注意**：`gtv-shm`（`gtv_ext_lib/gtv-shm/src/lib.rs:140-168`）已经有一条成熟的「mmap → Arrow `Buffer` 零拷贝」实现，可作为 ingest 通道复用，但它是**进程间 IPC**，不是持久化格式，不要混用。

### 3.6 SQL 接入面

在 `context.rs:447-471` 的 `register_parquet` 旁并列新增：

```rust
ctx.register_segment(path, name)   // 打开 GTVSeg，注册为可查询表 / 图
SELECT * FROM gtv_segments();      // 段清单、驻留状态、冷热度（对齐 gtv_extensions() 风格）
```

保持 `read_parquet` / `register_parquet` 不变，L2 冷数据仍走 DataFusion `ListingTable`。

---

## 4. 与已有 `docs/storage_design.md` 的关系

那份文档（约 500 行）方向是对的：WAL、delta、CSR 分档、mmap、zone map、向量 sidecar 都写到了，且 `docs/assessment-2026-11.md:78-80` 正确判定它「只是设计文件」。**不要重写它**。但它有 5 个缺口，正好卡住落地：

| 缺口 | 说明 | 本文对应 |
|---|---|---|
| ① 未解决 `TemporalCSR` owned `Vec` 与 mmap 借用的生命周期冲突 | 这是最硬的阻塞；只说「能直接 mmap 成为 in-memory CSR」没给 Rust 层面怎么做 | §3.4 `Col<T>` + `Deref` |
| ② 没有字节级格式定义 | `row_ptr.bin` / `col_idx.bin` 只给了文件名，没给对齐、字节序、magic、校验、版本、不可变约束 | §3.2 |
| ③ 没处理 Parquet 压缩与 zero-copy 的根本矛盾 | 文档里 `delta.arrow`、`block_*.col` 与 mmap 并列，但没说清压缩列不可映射 | §2 断点①、§3.1 分层 |
| ④ 读路径没有最小闭环 | WAL / delta / compaction / recovery 一齐上，第一个可验收里程碑太远 | §6 P0 只做读路径 |
| ⑤ **完全没有 CPU 缓存层设计** | 只有 `madvise(MADV_WILLNEED)` 与 prefetch 一句；无 cache line、无对齐、无窄化、无 Eytzinger、无 TLB/大页 | §3.3 |

另外，该文档附录 A（`:302-314`）的 WAL demo 用 `serde + bincode` 序列化——**与 zero-copy 目标相反**。那段代码自注「並非生產就緒」，实现时不要沿用。

---

## 5. 量化验收（当前所有压测都没有这一项）

`PT_RESULT_V1.MD:11-18` 明确「Not counted: one-time index build, network I/O」，TC1–TC15 全部是**纯内存合成数据的计算延迟**，`docs/storage_design.md:257` 把 "Cold mmap load time" 列为**待测**。所以存储性能目前是一片空白，必须先建基线。

建议新增 `bench_segment.rs`，对 1M / 10M / 100M 边三档测：

| 指标 | 现状（Parquet → CSR） | GTVSeg 目标 |
|---|---|---|
| Cold open（首次可用） | 全量解码 + 排序，随 E 线性/超线性 | `O(1)`：open + mmap + header 校验，< 10 ms |
| First traversal（触发 page fault） | 已在 open 时付过 | 显式 `MADV_WILLNEED` + 大页，可测、可优化 |
| Steady traversal | 基线 | **不低于现状**（`Col<T>` 不得引入间接开销） |
| RSS | 3–4 份数据拷贝 | ≈ 页缓存 1 份 |
| 内存带宽 / cache miss | 未测 | `perf stat` 采 `cache-misses`、`dTLB-load-misses` |

**准入门槛**：Steady traversal 不退步（这是 `Deref` 方案能否成立的前提），Cold open 至少一个数量级改善。

---

## 6. 实施路线（每阶段独立可验收）

| 阶段 | 交付 | 本次增修（OceanBase 借鉴） | 验收 |
|---|---|---|---|
| **P0 读路径闭环** ✅ 已实现 | GTVSeg v1 writer（离线工具）+ `Col<T>` + `TemporalCSR::open_segment()` + `register_segment()` | `col_dir` / `flags` **预留** skip index 槽位；不改闭环范围 | 现有 13 个 CSR 单元测试全部在 mmap 后端再跑一遍；`bench_segment` 基线出炉 |
| **P1 写路径** ✅ 已实现 | WAL + MemSegment + flush + crash recovery | **写路径零编码零压缩**；**段级复用 + 渐进重写**；**全局快照点绑定 `system_ts`**；落实「LSM 写、CSR 读」 | kill -9 后重放一致；`write_atomic` 原子性；**历史段复用率**（应近 100%，只重写活跃片）；`as_of_system(T)` 与基线内容一致 |
| **P2 缓存优化** | 窄化 16 B/边、chunk 按字节、Eytzinger 旁路、大页读路径 | **分层 Skip Index 落地**（段/src/chunk）；**SIMD 时间谓词下推**（在未编码定宽数组上，比 OceanBase 的编码域 SIMD 更简单） | `perf stat` 对比 P0；新增「整 src 跳过」命中率指标 |
| **P3 分层与扩展** | 冷热淘汰、compaction、`.gtvidx` mmap 化、量子状态段、NUMA | **L2 与 Parquet row-group statistics 共用剪枝谓词**；把 `as_of` 下推接线到 `ListingTable`（当前全库无 `PruningPredicate` 接线） | 冷热两层 `as_of` 剪枝结果一致；100M 边常驻内存可控 |

P0 **不碰** WAL、不碰 `from_arrays`、不碰任何算子，风险最低，且单独就能回答「能不能打开即用」。本次 OceanBase 评估**不改变**这个最小闭环的风险边界。

> L2 冷层不要再造压缩格式：`parquet.rs:49-53` 已经 `set_dictionary_enabled(true)` 且用 row-group min/max statistics 做 as-of 剪枝——OceanBase 第一层编码 + skip index 的思路在那里已经部分在用，缺的只是把剪枝谓词与 GTVSeg 统一、并真正接到查询上。

---

## 7. 风险与明确不做的事

- **unsafe 面积**：`Mmap → &[T]` 需要 `unsafe`。必须集中封装在 `Col<T>` 一处，配齐对齐/长度/校验断言 + 文件不可变约束；禁止在算子层散落裸指针。
- **Windows/Linux 差异**：现有 `madvise_hugepages`（`hdb.rs:391`）是 `#[cfg(target_os="linux")]`。Windows 侧对应 `PrefetchVirtualMemory` / large pages，需单独实现，别假设 Linux 结论能直接搬。
- **`u32` offset 上限**：`offsets: Vec<u32>` 意味着单段 ≤ 2³² 边。超出要分段时间片（本来就该分），不要悄悄升到 `u64` 破坏对齐。
- **文件不可变**：更新必须 copy-on-write 新段；mmap 期间改写同文件会直接导致读侧读到撕裂数据。
- **热层不压缩**：这是 §3.2 的「不压缩」约束上升为架构红线（论证见 `oceanbase-storage-lessons.md` §2/§3）。若将来在 `flags` 里加入压缩位，必须同时放弃 `cast_slice` 路径并补一套用户态 block/row cache——那正是 OceanBase 因格式不可映射而付出的代价。任何「给热段加个 zstd / 字典编码」的提议，先回答「省下的 I/O 在哪」：mmap 场景下答案是 **0**。压缩只属于 L2 Parquet，且只在归档时发生。
- **不做**：分布式存储、事务隔离、行级 MVCC、把 Parquet 从仓库移除。这些超出「嵌入式分析引擎」定位，且 `README.md:172` 已把 transactions 明确列为 out of scope。
- **不做**：在 P0 把 `edge_slices()` 改成 AoS。先用 `Deref` 拿到零改动收益，AoS 留到 P3 用数据说话。

---

## 8. P0 实现记录（2026-11-08）

### 8.1 交付清单

| 文件 | 内容 |
|---|---|
| `crates/gtv-core/src/column.rs`（新增） | sealed `Pod`、`MappedRegion`、`Col<T>` / `MappedCol<T>` + `Deref<Target=[T]>`；全库唯一一处 `unsafe`。5 个单测 |
| `crates/gtv-core/src/csr.rs` | 7 个 `Vec` → `Col`；`from_arrays()` 保留；新增 `columns()` / `from_columns()`；**1180 行算子零改动** |
| `crates/gtv-storage/src/segment.rs`（新增） | GTVSeg v1 header / 列布局、`encode_segment` / `write_segment` / `open_segment` / `Segment::verify_data()`、`MappedRegion for Mmap` |
| `crates/gtv-storage/tests/segment.rs`（新增） | 10 个集成测试：往返、双后端等价、COW 重写、损坏 / 截断 / 对齐 / 版本 / flags 错误路径、空图 |
| `crates/gtv-storage/examples/bench_segment.rs`（新增） | cold open / first traversal / steady traversal 基线 |
| `crates/gtv-engine/src/segment.rs`（新增） | `gtv_segments()` / `gtv_segment_edges(name)` UDTF |
| `crates/gtv-engine/src/context.rs` | `register_segment()` / `segment()` / `segments()` + 两个 UDTF 注册 |
| `crates/gtv-engine/tests/segment_sql.rs`（新增） | 3 个 SQL 端到端测试 |
| `.gitignore` | `*.seg`（生成物）、`/.workbuddy/`（本机工作日志） |

### 8.2 验据

`cargo test --workspace --no-fail-fast` → `REAL_EXIT=0`，**84 个测试目标全 ok，685 passed / 0 failed / 2 ignored**，零编译 / 链接错误、零 warning。上一轮基线 667 passed，本轮净增 18（5 `column` + 10 `segment` + 3 `segment_sql`）。

等价性验证比「把 13 个 CSR 单测在 mmap 后端再跑一遍」更强：同一份图在 owned / mmap 两种后端上，对 12 个时间点 × 6 个源节点逐项比对 `neighbors`、`edge_slices`、`degree`、`stats`、`degree_histogram`、`all_active_at`、`transpose()`，并逐列比对全部 11 个列 / 标量字段；遍历 checksum 两边一致。

`bench_segment --release`：

| 指标 | 200k 边 / 100k 节点 | 1M 边 / 100k 节点 |
|---|---|---|
| `from_arrays` (build) | 13.065 ms | 73.391 ms |
| `open_segment` (cold) | 4.351 ms | 3.732 ms |
| **cold-open speedup** | **3.0×** | **19.7×**（5 次取样 18.4–28.9×） |
| first traversal owned / mapped | 3.317 / 3.162 ms | 8.543 / 9.839 ms |
| **steady mapped / owned** | **0.983×** | **0.995–1.078×**（5 次取样，中位 ≈1.04×） |
| `verify_data()`（不在 open 路径） | 3.146 ms | 14.486 ms |

**两个必须讲清楚的结论**：

1. **debug 数字会骗人**。同一基准在 dev profile 下 `mapped/owned = 1.934×`，看起来像严重回归；release 下 `Deref::deref` 与 `from_raw_parts` 被完全内联，差距归零。debug 不内联时，每个 `self.dst[e]` 都要走一次未内联的函数调用。**任何关于 `Col<T>` 性能的结论都必须在 `--release` 下取得**。基准本身也据此修正过一次：初版把「首次遍历的 page fault」混进稳态数字，现已拆成独立指标（对应 §5 的 First traversal 行）。
2. **准入门槛基本通过，但不是零成本**。200k 边 mapped 略快（0.983×）；1M 边中位约 **+4%** 的轻微、波动开销（0.995–1.078×），疑似 TLB miss / 页表遍历随工作集增大而上升。`Deref` 方案成立（没有数量级回归），但 §5 的「Steady traversal 不退步」在大规模下应表述为「≤5%」。这一项需要 P2 的大页 / `MADV_WILLNEED`（Windows 对应 `PrefetchVirtualMemory`）与 `perf stat` 的 `dTLB-load-misses` 采样来归因，本轮**未做**。

诚实边界：200k 边时 cold-open 只快 3×，未达 §5 设的「一个数量级」门槛；1M 边达到 19.7×，印证 `O(1)` open 对 `O(E log E)` build 的规模效应。10M / 100M 边、`perf stat` cache-miss 采样、NUMA 均待测。

### 8.3 相对本文的 6 处工程偏差

1. **未引入 `bytemuck`**：改用自己的 sealed `Pod`（只允许 `u16/u32/u64/i64`）。`bytemuck` 当时既不在依赖树也不在离线缓存，为一次 `cast_slice` 拉新 crate 不划算；sealed trait 同样防住 `bool` / 引用 / 带 padding 的类型，unsafe 面积不变。
2. **`MappedCol` 缓存 `*const T` + `len`**：§3.4 的伪码在 `Deref` 里才做 `cast_slice over map[range]`。实现改为在 `new()` 一次性完成 bounds / alignment / overflow 校验并缓存指针，`Deref` 只剩一次 `from_raw_parts` —— 避免热循环里的 trait-object vtable 调用，这是 steady traversal 不退步的前提。空列用 `NonNull::dangling()` 保证 `from_raw_parts` 的指针对齐合法。
3. **checksum 拆成两级**：§3.2 只写一个 blake3。若覆盖全文件，open 就变成 O(file size)，直接毁掉「O(1) cold open」。改为 `header_checksum`（覆盖 0..256，强制、O(1)，保护所有决定列位置的字段）+ `data_checksum`（writer 计算、reader 不默认验，由 `Segment::verify_data()` 显式做）。
4. **`open_segment` 属于 `gtv-storage` 而非 `TemporalCSR` 的方法**：`gtv-storage` 已依赖 `gtv-core`，反向依赖会成环。`gtv-core` 提供 `from_columns()` / `columns()` 作为跨 crate 组装边界，I/O 与 header 解析留在 storage 层。`Segment` 实现 `Deref<Target=TemporalCSR>`，调用侧写法与文档几乎一致。
5. **v1 列宽保持原生**：§3.2 表格里 `dst u32` 是窄化后的 P2 形态。P0 用 `dst u64` / `i64` 时间戳，与现有 CSR 完全一致；`flags` 的 P2 位已定义且 reader 明确拒绝，避免把未实现格式误读成合法数据。大端主机同样明确报错。
6. **SQL 边界**：新增 `gtv_segment_edges(name)` 是**物化的** Arrow 表，仅用于检视 / 导出，代码注释已标明不是零拷贝热路径；热路径仍是 Rust 侧 `edge_slices()` / `neighbors()`。P0 没有把段伪装成 DataFusion `ListingTable`。

### 8.4 P0 当时明确未做（P1 已补上前四项）

窄化 16 B、分层 Skip Index、SIMD 谓词下推、Eytzinger、大页 / `MADV_WILLNEED` 的 Windows 对应实现（P2）；冷热淘汰、`.gtvidx` mmap 化、量子状态段、`perf stat` 采样（P3）。P0 时段只能由离线 writer 产生；P1 起已接入 ingest（见 §9）。

---

## 9. P1 实现记录（2026-11-08）

### 9.1 交付清单

| 模块 | 内容 |
|:--|:--|
| `crates/gtv-core/src/csr.rs` | 新增 `TemporalCSR::from_edge_columns()`：`from_arrays` 的原生定宽列版本，`from_arrays` 委托之。整个 flush 只排序一次；P0 的 `from_columns()` / `columns()` 及 1180 行算子零改动 |
| `crates/gtv-storage/src/wal.rs` | append-only WAL：64 B header + `[frame_header 24 B][payload n×42 B][blake3 32 B]`；`EdgeRecord` 定宽 42 B，只有小端字节序转换，无 varint / 字典 / 压缩；帧校验和检测 kill -9 的半写尾部 |
| `crates/gtv-storage/src/mem_segment.rs` | 原生 `Vec<IngestedEdge>`，`groups()` 按 `(table_id, partition_id)` 分组，`to_csr()` 一次性生成单一连续 CSR |
| `crates/gtv-storage/src/store.rs` | `SegmentStore`：manifest（JSON，`write_atomic` 发布）+ 回放 + flush + `graph_at(T)` / `entries_at_system(T)` / `verify_all()` |
| `crates/gtv-engine/src/context.rs` | `register_segment_store(store, prefix)`：把已发布段注册进 P0 的 segment registry，SQL 可直接查询 |
| 测试 | `crates/gtv-storage/tests/p1_write_path.rs`（6 个集成用例 + `wal.rs` 2 个单测）、`crates/gtv-engine/tests/p1_segment_store_sql.rs` |

### 9.2 崩溃一致性顺序（相对 §3.5 的具体化）

```
ingest:   WAL.append + fsync  →  MemSegment.push_batch
flush(T): ① 按活跃分区分组、各排序一次、encode_segment
          ② write_atomic 发布新 .seg（同 tmp + rename，历史段不重读不重写）
          ③ write_atomic 发布 manifest（checkpoint_txn = txn_end，system_ts = T）
          ④ WAL.checkpoint() 截断到 64 B header
```

* 崩在 ③ 之前：checkpoint 未动，重开时重放全部帧；孤儿 `.seg` 不在 manifest 中，被忽略且下次 flush 以相同 id 覆盖。
* 崩在 ③ 与 ④ 之间：帧的 `txn_id <= checkpoint_txn` 被跳过，不重复应用。
* 崩在 ② 中：`write_atomic` 保证读者只看到完整文件。
* WAL 尾部损坏：`scan_frames` 在最后一个完整帧处停止并 `set_len`，后续 append 落在干净边界。

### 9.3 验据

| 验收项（§6 P1 行） | 结果 |
|:--|:--|
| kill -9 后重放一致 | `crash_before_flush_replays_wal_then_flushes_exactly_once`：3 帧 / 90 边全量重放，flush 后重开 `frames_replayed = 0`、边数仍 90（无重复） |
| `write_atomic` 原子性 + 断尾 | `torn_wal_tail_is_truncated_and_ingest_continues`：人工半写 18 B，`torn_tail_bytes = 18`，截断后继续 ingest / flush / 重开成功 |
| checkpoint 幂等 | `checkpoint_skips_stale_frames_and_orphan_segments_are_ignored`：人为写入 `txn_id == checkpoint` 的旧帧 + 孤儿段，重开后跳过且不重复计边 |
| 历史段复用率 | `flush_publishes_mmap_segments_and_reuses_history`：第二次 flush 的 `reuse_ratio = 0.5`（1 保留 / 1 新建），首段文件字节完全未变；随 flush 次数趋近 1.0 |
| `as_of_system(T)` 与基线一致 | `graph_at(15) = 100` 边、`graph_at(25) = 150` 边；`entries_at_system(T)` 分别为 1 / 2 个段；返回的是**单一 CSR**，不在 `neighbor_iter` 里做多路归并 |
| SQL 可见性 | `ingested_edges_are_queryable_as_mmap_segments`：`gtv_segments()` 显示 `mmap` / 16 节点 / 24 边，`gtv_segment_edges` 计数 24 |

### 9.4 工程偏差与边界

1. **manifest 用 JSON 而非二进制**。它每次 flush 才写一次，不在热路径；可读性好于自造格式，且 `write_atomic` 的原子性不受影响。段格式本身仍是纯定宽二进制。
2. **WAL 每帧 blake3 32 B**。§3.5 引用的 `storage_design.md` 帧格式没有校验和；加上它才能把「半写尾部」与「合法帧」区分开。写路径仍无编码/压缩，只是多一次流式哈希。
3. **`compaction()` 尚未实现**。P1 只做「段级复用 + 追加新段」，读侧的版本融合由 `graph_at(T)` 一次性物化成单一 CSR；把多个段归并回一个基线属 P3。因此 `reuse_ratio` 是「未重写比例」，不是 compaction 效率。
4. **全局快照点**：`flush_at(T)` 拒绝 `T < max(edge.system_ts)`，保证段内所有边在 T 时已可见；`flush()` 用当前墙上时钟。manifest 记录 `system_ts`，`entries_at_system(T)` 据此剪枝。
5. **Windows 权限细节**：WAL 以 `read+write` 而非 `append` 打开，因为 append-only 句柄在 Windows 上没有 `set_len` 所需的通用写权限；`append()` 每次显式 `seek(End)`，语义不变。这是首轮测试 `os error 5` 的根因，已修。
6. **数据纪律（硬性）**：P1 所有测试数据在内存构造、只写 tempdir，**不产生也不提交 CSV**。`.gitignore` 全局忽略 `*.csv`，CI 新增 `generated-data` job 检查 `git ls-files` 中没有 `.csv` / `.seg`。测试里的 `assert_no_csv()` 会在运行期守住这条线。

### 9.5 P1 明确未做

compaction / 渐进重写、L2 Parquet 归档淘汰、孤儿段 GC、并发写入（当前 `&mut self` 串行）、WAL 批量 fsync、真实 `kill -9` 进程级测试、P2 缓存优化与 P3 分层。段内每边的 `system_ts` 不单独存放，由段级 `system_ts` 作为快照锚点，因此 `graph_at(T)` 的时间粒度是 flush 快照粒度。

