# gtvdb 设计文档（第三方评审版）

> 版本：2026-11-09 ｜ 代码基线：`gtvdb_core@554ad9f`（P1-4 后）、`gtvdb_ext@f64311c`（SQL/CSV 基准后）
> 文档目的：让**没有参与开发**的评审人能在不读全部源码的前提下，判断这套设计是否成立、边界是否诚实、风险是否可控。
> 本文只描述**已实现**的东西。未实现或已冻结的项，统一写在 §12「明确不做 / 已冻结」，不与现状混写。

---

## 0. 给评审人的导读

如果你只有 20 分钟，按这个顺序看：

| 顺序 | 章节 | 你要判断的问题 |
|:--|:--|:--|
| 1 | §2 定位与非目标 | 这个产品定义是否清晰、是否在打一场不必打的仗 |
| 2 | §4 数据模型 | 把「时间」和「来源」做进存储格式，是否真的比外挂一列 `valid_from/valid_to` 强 |
| 3 | §6 存储格式 + §7 读路径 | 定宽不可变段 + mmap 零解码，是否撑得住「媲美 kdb」的目标 |
| 4 | §9 过去更新 | 快照级 PUBLISH/RETRACT/AS OF 三原语，是否够用、代价是否被说清 |
| 5 | §12 已知限制 | 这些限制是不是你能接受的 |
| 6 | §13 评审问题清单 | 六个我认为最可能被推翻的设计选择 |

全文的**数字都标了机器与口径**。凡是没有同机 A/B 的跨机器对比，本文一律只谈趋势，不当作性能声明。

---

## 1. 一句话与三句话

**一句话**：gtvdb 是嵌入式时序图分析引擎，在同一个 Arrow 表上回答「`T` 时刻这张图长什么样」和「这个状态是怎么一步步变成现在的」。

**三句话**：

1. 边带半开有效区间 `[valid_from, valid_to)`，`as-of T` 是一次**索引操作**，不是全表扫描；
2. 每个状态能沿父链回溯到产生它的路径与概率，代价 O(depth)；
3. 它嵌在你的进程里（Arrow 零拷贝 + DataFusion SQL + 可 mmap 的定宽段文件），不是又一个要运维的集群。

**性能目标不是「列式算子带宽」**，而是「图 + 时间」类查询的**端到端延迟**——kdb+ 表达同一件事需要多次 join 或多段 q 脚本。这一点决定了后面所有的取舍（§2.1）。

---

## 2. 定位与非目标

### 2.1 为什么不做「通用图数据库」也不做「通用列存」

| 赛道 | 已有的评估清单 | gtvdb 的选择 |
|:--|:--|:--|
| 图数据库（Neo4j / Kuzu / Memgraph） | Cypher、ACID、HA、生态、云服务 | 不参加。没有 Cypher，不做服务端 |
| 时序数据库（kdb+ / Timescale / Influx） | 单表向量化扫描、asof join 带宽 | 不正面比。只在「图 + 时间」的端到端查询上比 |
| 湖仓 / 列存（DuckDB / Parquet 生态） | 扫描吞吐、压缩率 | 热层**不压缩**，因为 mmap 后必须能零解码直接算 |

**判定规则（写进了 PR 模板与 CI）**：任何进主干的改动必须回答「它让 Q1/Q2/Q3 中哪一条的 p99 降了多少，基准在哪个文件」。答不上来的进冻结清单，不进里程碑。

三条目标查询（下文的 Q1/Q2/Q3 恒指这三条）：

| 代号 | 形态 | 规模口径 |
|:--|:--|:--|
| **Q1** | N 个种子、2–3 hop、指定 `valid_at` | 1e7 边级图上的 p50/p99 |
| **Q2** | `[T1,T2)` 活跃边 / 敞口聚合 | p99，含 `FullHit` 短路比例 |
| **Q3** | 摄入 | 单进程 M edges/s、publish→可见延迟 p99 |

### 2.2 业务锚点

主用于金融 **AML / CRM / LE** 这三类「一批种子 → 时序图求解 → 聚合」的场景：

- **AML**：时序资金流追踪、环路 / motif 检测（k-hop + 时间窗）；
- **CRM**：担保 / 抵押链传递、客户集扩散；
- **LE**：**Large Exposure（大额风险敞口，MA(BS)28）**，客户群 → 集团 → 敞口聚合。**不指 Law Enforcement**——这条在早期文档里有歧义，此处固定。

选这三类的理由不是「市场大」，而是它们**共享同一个查询形状**：种子集 + 时间点/窗口 + 多跳 + 聚合。同一个内核能同时服务，才不会变成功能堆叠。

### 2.3 非目标（明确不做）

| 不做 | 理由 |
|:--|:--|
| 分布式（shard / planner / worker） | 目前只是远程执行 SQL；没有真实多机负载前不承诺 |
| 通用图查询语言（Cypher / GQL） | 语义面积大、与 SQL 表面重复；核心只做 SQL + 图算子 |
| 热层压缩 | `mmap` 后必须零解码，压缩破坏这个恒等式（压缩留给 L2 Parquet 归档） |
| 事务隔离级别、MVCC 多写者 | 单写者 `&mut self`，见 §12.2 |
| 把领域功能塞进核心 | 银行 / HFT / 量子全部在 `gtvdb_ext`，核心只保留注册接口 |
| 提交任何生成数据集 | 历史教训：两个 >50MB CSV 曾拖慢每次 clone。改为生成器 + CI guard |

---

## 3. 仓库边界与依赖方向

```
jacalwu/gtvdb_core        （内核，12 个 lib crate + 1 个零扩展 REPL）
        ▲
        │  git dependency，钉单一 commit SHA，单向
        │
jacalwu/gtvdb_ext         （13 个领域扩展 crate + full-pack CLI + gRPC server）
```

这不是目录约定，是**依赖源约束**：`gtvdb_ext` 的根 `Cargo.toml` 用固定 40 位 SHA 引入内核 crate，核心仓的 `Cargo.toml` 里不可能出现任何领域 crate。CI 的 `core-boundary` job 额外检查三件事：不内嵌内核源码、不用本地 `path` 回退、所有内核 crate 钉在同一个 SHA。

```
gtvdb_core/crates/
├── gtv-core        Temporal CSR、bitemporal、endpoint policy、provenance、量子状态元数据
├── gtv-array       列式 / 时序 kernel（asof、window、window_simd、crm、cr_ead…）
├── gtv-index       ANN：flat / IVF / HNSW，统一 Metric 契约
├── gtv-index-store .gtvidx 容器
├── gtv-storage     GTVSeg（mmap）、WAL、MemSegment、manifest、delta、observation、GC
├── gtv-delta       LSM delta 层
├── gtv-pattern     时序 motif / 环路匹配（ring3/4、path2-4、diamond）
├── gtv-catalog     L1 catalog：schema / stats / lineage / partition / manifest
├── gtv-ingest      批量摄入：envelope、watermark、dedup、DLQ
├── gtv-udf / gtv-proto / gtv-engine / gtv-cli
└── gtv-engine      DataFusion 表面 + 扩展 ABI（ext.rs）+ 批量图算子 + motif SQL
```

**为什么拆两个仓**：核心的承诺是「读路径零回归」，一旦领域代码混进来，任何一次领域改动都可能碰 `csr.rs`。物理隔离让这条纪律由 Cargo 保证，而不是靠自律。

---

## 4. 数据模型：一张表，三个维度

### 4.1 不是「三种数据库合一」，是一张表三个维度

早期文档说「Graph + Temporal + Vector 三合一」，这句话既难验证也招打。实际的数据布局是：

| 维度 | 在表里是什么 | 谁在用 |
|:--|:--|:--|
| **G** | 边：`src → dst` | 遍历、k-hop、motif |
| **T** | 边的有效区间 `[valid_from, valid_to)` + 系统时间 | `as_of(T)`、时间窗聚合 |
| **V** | 节点 / 边上的向量列 | KNN、语义检索 |

三者在同一个 `RecordBatch` 里。所以「时间切片 → 向量检索 → 图遍历」不需要跨系统搬数据。护城河不是功能，是**这个数据布局**。

### 4.2 双时间：业务时间 vs 系统时间

```rust
pub struct BitemporalRange {
    business_from: i64, business_to: i64,   // 这件事在现实世界里何时成立
    system_from:   i64, system_to:   i64,   // 我们何时知道 / 何时不再这么认为
}
```

两轴的语义差异是**更正（correction）与回填（backfill）的差别**：

- 业务时间改动 = 「当时的事实不是这样」→ 必须留痕，不能覆盖；
- 系统时间推进 = 「我们现在才知道」→ 是正常追加。

因此系统时间只**追加**，不修改；业务时间的修改走 §9 的快照级更正原语。

### 4.3 半开区间与「端点策略」

边的区间是半开的 `[valid_from, valid_to)`。选半开而非闭区间，是为了让相邻区间可以首尾相接而不重叠（`[1,5)` 与 `[5,9)` 在 `t=5` 恰好只有一个成立）。

节点还有一层：**端点存活期**。

```rust
pub struct EndpointPolicy { /* 每个 edge_type 1 bit */ }
pub struct NodeLifetime   { /* 每节点定宽 (valid_from, valid_to) 对 */ }
```

`EndpointView::active_neighbors(src, t)` = 边的时间谓词 ∩ 两端节点的存活期 ∩ edge_type 掩码。这三层过滤全部是**定宽数组上的标量比较**，没有分支预测灾难，也没有额外分配。

设计选择：**用一个 1-bit 掩码而不是 per-type 的复杂策略对象**。理由是热路径上每多一层间接就多一次 cache miss，而实际配置里「某类边要不要参与端点过滤」几乎是二值的。代价是不支持 per-type 的复杂规则——这是有意为之。

### 4.4 为什么「时间是一等字段」不是文案

如果时间只是普通列，`as_of(T)` 就必须全表扫描 + 过滤。这里的差别在存储格式里：

- 段头存 `max_valid_from` / `min_valid_to`，一次比较就能判定**整段**与 `T` 的关系；
- 段内按 `src` 分 run，每个 run 有 `chunk_max_valid_to` zone map；
- 因为 `valid_to` 在 run 内**非单调**（可以随时终止），所以剪枝用「`max_valid_to` 上界 + run 内二分」，而不是简单的二分查找。

这条「非单调所以用 max 上界」是读路径最关键的一个细节，也是最容易被评审误判为「为什么不用普通二分」的地方。

---

## 5. 架构总览

```
                     ┌──────────────────────────────────────────┐
   SQL / DataFusion  │  gtv-engine: TableProvider / UDTF / UDF   │
                     │  neighbors · khop · motif · asof · knn    │
                     └───────────────┬──────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
┌───────▼────────┐        ┌──────────▼─────────┐       ┌──────────▼─────────┐
│  TemporalCSR   │        │   SegmentStore     │       │   Vector indexes   │
│  内存态遍历     │        │   WAL → MemSeg →   │       │  flat / IVF / HNSW │
│  neighbors()   │        │   GTVSeg → manifest│       │  filter-aware      │
│  khop()        │        │   delta / obs / GC │       │  exact rerank      │
└───────┬────────┘        └──────────┬─────────┘       └────────────────────┘
        │                            │
        │                   ┌────────▼─────────┐
        └───────────────────►  mmap 零拷贝读取  │
                            │  GTVSeg v1 定宽段 │
                            └──────────────────┘
```

两条读路径，共用一个物理布局：

- **内存态**：`graph_at(T)` 把可见边折叠成 `TemporalCSR`，之后 `neighbors` / `khop` 全在内存里跑；
- **落盘态**：`register_segment()` 直接 `mmap` 段文件，`neighbors_record_batch()` 从映射区切片构造 Arrow 数组，**零解码**。

两条路径的正确性由同一组 reference 断言约束（§10）。

---

## 6. 存储格式：GTVSeg v1

### 6.1 字节布局

一个段 = 一个时间分区内的一个**不可变**文件。所有多字节字段 little-endian，所有列区 **64 B 对齐**（正好一条 x86-64 cache line）。

```
offset 0     ┌────────────────────────────────────────────┐
             │ Header (固定 4096 B)                        │
             │  magic        "GTVSEG\0\0"  [u8;8]          │
             │  version      u16 = 1                       │
             │  flags        u16  (窄化 / 校验类型)         │
             │  node_count   u64                           │
             │  edge_count   u64                           │
             │  chunk_size   u32  (边数/块)                 │
             │  col_dir[8]   { offset:u64, len:u64,        │
             │                 elem:u8, reserved[7] }      │
             │  time_bounds  { max_valid_from:i64,          │
             │                 min_valid_to:i64 }           │
             │  checksum     blake3 [u8;32]                │
             └────────────────────────────────────────────┘
offset 4096  ┌─ offsets              u32  (node_count+1)  ┐ ← 最热，单独一页
             ├─ chunk_index          u32                   ┤
             ├─ chunk_max_valid_to   i64                   ┤ ← zone map 热数组
             ├─ dst                  u32                   ┤
             ├─ valid_from           i64 / u32(delta)      ┤
             ├─ valid_to             i64 / u32(delta)      ┤
             ├─ edge_type            u16                   ┤
             └─ (可选) provenance / vector sidecar          ┘
```

### 6.2 三条不可协商的约束

| 约束 | 为什么 |
|:--|:--|
| **不压缩** | 定宽 POD 数组 `mmap` 后经 `bytemuck::cast_slice` 直接得到 `&[u32]` / `&[i64]`。任何压缩都要解码，解码就要分配，就破坏了「页缓存即数据集」 |
| **不可变** | 写入永远生成新段（copy-on-write），发布走 `write_atomic`。读侧因此无需加锁，也不用担心映射被并发改写 |
| **定长 header + 64 B 对齐** | 保证 `disk == memory == cache line`：磁盘上的字节布局与内存里的数组布局完全相同，偏移可用整数算出，不需要反序列化 |

校验：加载时对 `bytes[4096..]` 做一次 blake3，失败即拒绝映射——避免损坏文件导致 UB。对齐在 `cast_slice` 前断言 `ptr % align_of::<T>() == 0` 且 `len % size_of::<T>() == 0`。

### 6.3 窄化（只留了位，未实现）

节点数 < 2³² 时 `dst: u64 → u32`；时间戳改成相对分区起点的 `u32` delta。每边 26 B → **16 B**，一条 cache line 从装 2.4 条边变成 4 条。

这是**收益最直接的缓存优化**，但它改的是数据表示，必须走 A/B 且不能破坏 §7 的红线。当前状态是明确的：`FLAG_NARROW_DST` / `FLAG_DELTA_TIME` / `FLAG_SKIP_INDEX` 三个位已定义，reader 把它们列进 `FLAG_UNSUPPORTED`，**遇到即拒绝加载并报「P2 encoding this v1 reader cannot decode」**。也就是说不是「半实现」，而是「显式拒绝」——不会出现读了一半才失败的情况。

### 6.4 为什么是「段」而不是 LSM

| | 段（当前） | LSM |
|:--|:--|:--|
| 读 | 一次 mmap，零合并 | 多层合并读，可能要读多个 level |
| 写 | 追加新段 + manifest | memtable → flush → compaction |
| 空间放大 | 无（不可变段直接映射） | 有（多层重复） |
| 更新 | 位置化 retract + 两阶段 compaction | 墓碑 + compaction |

选段的理由是**读放大优先**：目标查询是「读多写少、读要 p99 稳定」。LSM 的写优化在这里买不到东西，反而把读路径变成多层查找。

---

## 7. 读路径

### 7.1 TemporalCSR

```rust
pub struct TemporalCSR { /* offsets / dst / valid_from / valid_to / chunk_max_valid_to */ }

impl TemporalCSR {
    pub fn neighbors(&self, src: u64, valid_at: i64) -> Result<Neighbors<'_>>;
    pub fn neighbors_record_batch(&self, seeds: &UInt64Array, valid_at: i64)
        -> Result<RecordBatch>;
    pub fn khop(&self, seeds: &UInt64Array, k: usize, valid_at: i64)
        -> Result<Vec<UInt64Array>>;
    pub fn khop_bounded(...) -> ...;   // 带 TraversalBudget / CancelToken
    pub fn khop_directed(...) -> ...;  // direction-optimizing
}
```

`neighbors()` 的执行顺序：

1. `offsets[src] .. offsets[src+1]` 定位 run；
2. 用 `chunk_max_valid_to` 跳过整块都已过期的 chunk（**先看 zone map，再逐边判定**）；
3. 在未跳过的 chunk 内二分 `valid_from`，因为 `valid_to` 非单调，只能用 `max` 上界而不是精确二分；
4. 逐边判定 `valid_from <= t < valid_to`。

**这条循环是红线**：U0–U3 四个里程碑、M6-a、P1-4 全部**一行未改** `csr.rs` 的热循环。批量算子（§8.3）是在它外面套了一层，不是重写它。

### 7.2 L2 段级剪枝

```rust
pub enum PruneOutcome {
    Skip,      // 整段不可能贡献，连文件都不打开
    FullHit,   // 整段全活跃，可以走零扫描短路
    Scan,      // 必须逐边判定
}
```

判定发生在 `SegmentMeta::classify(t1, t2)`，用的是段级时间盒：

- `valid_from_min` / `valid_to_max`：**并集**盒，回答「有没有可能相关」；
- `max_valid_from` / `min_valid_to`：**交集**盒，回答「是不是全都相关」；
- `node_id_min/max`（src+dst 并集）与 `src_id_min/max`（仅 src）：回答「这个 src 在不在段里」。

为什么要分 src-only 盒：`neighbors(src, t)` 永远不需要打开一个 `src` run 为空的段，而 src∪dst 的并集盒表达不了这件事。

### 7.3 批量图算子

逐点 UDTF 的真实瓶颈**不是图循环，是 N 次 SQL 规划**。所以把「一次规划、一次执行、一个 RecordBatch」做成了两个 `TableProvider`：

| 表 | 对应内核 | 用途 |
|:--|:--|:--|
| `gtv_batch_neighbors` | `neighbors_record_batch()` | 可 join / 投影 / 过滤的批量邻居 |
| `gtv_batch_khop` | `khop(&seeds, k, t)` | 批量 k-hop |
| `gtv_motif_matches` | `gtv_pattern::find*` | 一行一 match |
| `gtv_motif_edges` | 同上（同一次 `Vec<Match>`） | 一行一 matched edge |

`register_batch_graph()` 是 owned（每次整图 clone），`register_batch_graph_shared()` 共享一个 `Arc<TemporalCSR>`。后者是生产路径，前者留作对照——因为 clone 一张 1e7 边的图约 1.2 ms，在 N 小时比规划开销还大。

---

## 8. 写路径与摄入

### 8.1 链路

```
ingest(边) ──► WAL 追加（42 B/边 + 24 B 帧头 + 32 B 尾校验）
                 │
                 ├─► MemSegment（原生 Vec delta，零编码）
                 │
                 └─► flush：编码成 GTVSeg → write_atomic → 更新 manifest
                                    │
                                    └─► checkpoint：截断 WAL
```

崩溃恢复：读 manifest 的 `checkpoint_txn`，重放其后的 WAL 帧；帧校验和失败即停止重放（截断而不是报错）。空 store 的 manifest 也要写 `node_count`（早期 bug，已修）。

### 8.2 崩溃一致性顺序

`write_atomic` 的语义是「要么旧文件，要么新文件」，顺序固定：

1. 写临时文件 + `fsync(file)`；
2. `rename(tmp, final)`；
3. `fsync(dir)`（仅 Unix；Windows 上目录 fsync 不可用，用 rename 的原子性兜底）。

**跨平台差异是真的**：M2 实测 Windows 与 Linux 的 fsync 成本相差约一个数量级，M6-a 复现为 4–6×。所以本文所有 fsync 相关结论只谈趋势。

### 8.3 为什么写路径零压缩零编码

写路径的每一层都在做同一件事：**让落盘格式等于内存格式**。MemSegment 是 `Vec<EdgeRecord>`，flush 只是把 `Vec` 按列重排 + 对齐 + 算校验和，没有序列化框架、没有压缩、没有 schema 推断。

代价是磁盘占用大（约 16–26 B/边）。这个代价被显式接受：目标场景是「热数据放内存 / mmap，冷数据归档到 Parquet」，而归档层才做压缩。

---

## 9. 过去更新：三原语

### 9.1 问题

「这条边在 `T` 时刻是对的，但我们后来发现它错了」——这既不是插入也不是删除，是**对历史的重述**。传统做法是整段重写，代价 O(段大小)。

### 9.2 三个原语

| 原语 | 语义 | 实现 |
|:--|:--|:--|
| **PUBLISH** | 发布一个新的快照级 delta | `SnapshotDelta` 追加，WAL `OP_MUTATION` v2 |
| **RETRACT** | 撤回某些**位置**（不是按值匹配） | `RetractSet` = 排序的 `RowPos { segment_id, row }` |
| **AS OF** | 在某个系统时间点看这张图 | `live_rows_at(system_t)` |

关键设计：**位置化撤回**。`RowPos` 是 `(segment_id, row)` 的 12 字节定宽对，不是「按 src/dst/时间匹配」。理由是后者在重复边（同一 src/dst/时间窗出现多次）上语义不确定，而位置永远唯一。

代价：位置会因 compaction 失效。解法是把撤回的落点推迟到 compaction **之后**——`DeltaDraft` 内部先跑 compaction，再做区间 sweep 把 `RowPos` 映射到新段。这是 M1 阶段踩过的坑（旧 `(segment_id,row)` 悬空），现在的实现顺序是固定的。

### 9.3 唯一的 retract fold

```rust
fn live_rows_at(&self, t: i64) -> Result<Vec<LiveRow>>;
```

这是全库**唯一**一处把 retract 折叠进可见集的地方。所有上层（`graph_at`、`visible_edges_at`、聚合、GC、compaction 校验）都调它，不各自实现一遍。好处是「Overlay == Materialized」这个等式只需要在一个地方证明。

### 9.4 三个存储产物

| 文件 | 内容 | 大小 |
|:--|:--|:--|
| `*.gtvd` | 快照 delta：header 96 B + retract 行 + invalidation 记录 48 B + blake3 尾 | 定宽 |
| `nodes_c<commit>.bin` | 节点存活期的版本化快照 | 定宽 pair |
| `*.gtvobs` | Arrow IPC 的观测版本（表级快照） | IPC |

`.gtvd` 的 header 是 96 B（早期 92 B 会导致 padding 错位，已修）。

### 9.5 两阶段 compaction

```
prepare(system_ts) ──► CompactionPlan（读旧段、算新段、不写盘、不持锁）
        │
        ▼
apply(plan)        ──► 写新段 + 新 manifest + 原子切换
```

两阶段的理由：`prepare` 可以放到维护线程里跑（`MaintenanceService` 的 phase-1 只 prepare），`apply` 必须在写者线程上做。这样长耗时的重排在后台，切换点很短。

GC 是**延迟**的：段被 compaction 替换后不立即删除，等所有活跃 reader（`ReadScope`）退出、且没有被任何 Tag pin 住，才回收。

---

## 10. 正确性策略

这一节是本文最值得评审的部分，因为**它决定了上面所有性能数字是否可信**。

### 10.1 每个性能数字都伴随一次独立重算

| 组件 | reference 实现 | 断言 |
|:--|:--|:--|
| 批量图算子 | 逐点 UDTF 跑 N 次 | 逐条一致，不一致 CI 红 |
| motif SQL | `gtv_pattern::find_from` 直接调用 | `pointwise == batch == kernel` 三重相等 |
| TC1 as-of join | 朴素 nested-loop | 逐行相等 |
| TC2 OFI | `msum` 窗口函数的标量实现 | 逐行相等 |
| TC3 洗舱环 | 通用 `find` ring(3) | 逐 match 相等 |
| TC4 KNN | 全量 exact 距离 | recall 与命中集合校验 |
| TC5 PIT | 全表 linear scan + zone mask | 逐行相等 |
| SQL/CSV 基准 | **纯 Arrow 列扫描**（不经 SQL 引擎） | 逐组比对，不一致直接中止不出数 |

原则：**fast-but-wrong 不允许存在**。任何基准在输出延迟之前先跑 reference；断言失败就不出数字。

### 10.2 CI 门禁

`gtvdb_core`：

- 全量回归（54 suites / 503 passed / 0 failed / 1 ignored）；
- M6 批量图算子 parity（step 17）；
- M6 准入基准（step 18，Q1/Q2/Q3 现状表 + 硬断言）；
- P1-4 motif SQL（step 19，9 项 parity + Q1 准入）；
- generated-data guard（禁止任何 `*.csv / *.seg / *.gtvd / *.gtvobs` 入仓）；
- core-boundary（目录树 + manifest 依赖 + `cargo tree` 三重检查）。

`gtvdb_ext`：workload regression + TC1–TC5 parity + CBOM drift + core-boundary。

### 10.3 为什么「生成数据不入仓」是硬纪律

历史上有两个 >50MB 的 CSV 被提交，导致每次 clone 都很慢。现在的做法：

- `.gitignore` 覆盖 `*.csv` 等；
- CI 有独立 job 检查 tracked 文件；
- 测试里有 `assert_no_generated_data()`，且基准只写 OS 临时目录；
- 所有基准数据放仓库外（`C:/ai/gtvdb/tmp/`、`_benchdata/`）。

这条纪律是用户明确要求的（「P1 不要提交 CSV」），且已经沉淀成 CI 守卫，不依赖人的记忆。

---

## 11. 扩展机制

### 11.1 ABI

```rust
pub trait GtvExtension: Any + Debug + Send + Sync {
    fn name(&self) -> &'static str;
    fn version(&self) -> &'static str;
    fn description(&self) -> &'static str { "" }
    fn objects(&self) -> &'static [&'static str] { &[] }
    fn register(&self, ctx: &GtvContext) -> Result<()>;
}
```

和 PostgreSQL 扩展的差别：**静态链接，没有 `CREATE EXTENSION`**。安装就是 `GtvContext::with_extensions(...)`。

**为什么不是 `dlopen`**：Rust trait object 是胖指针，布局不保证跨 dylib 边界。真要做动态 ABI，必须传 `#[repr(C)]` 的 `extern "C"` 函数指针表 + 不透明状态指针。当前的 `ExtensionRegistry` 就是「将来那种 vtable 会填充的那一层」，所以以后加 loader 不需要改任何扩展的实现。

传整个 `GtvContext` 而不是 `SessionContext`，因为扩展几乎总要共享核心状态：

| 存取子 | 用途 |
|:--|:--|
| `ctx.session()` | 注册 UDF / UDAF / UDWF / UDTF |
| `ctx.knn_collections()` | 与核心 `knn()` 共用向量集合 |
| `ctx.resources()` | `pit` / `asof` / `wash` / `tables` 资源 |
| `ctx.extension::<T>()` | 反向取回扩展实例（配置 policy、读状态） |

### 11.2 边界验证

`cargo check -p gtv-engine` 不拉进任何领域 crate。这不是靠人看，是 `core-boundary` job 用 `cargo tree` 断言。

### 11.3 扩展仓里有什么（不在核心的承诺内）

| crate | 内容 |
|:--|:--|
| `gtv-ext-banking` | `crm_alloc` / `crm_audit` / `crm_rating_map` / `cr_ead_rw` |
| `gtv-ext-hft` | `tick_to_trade` / `match_orders` / `backtest` / `var` / 指标 / 行情 |
| `gtv-ext-quantum` | `quantum_knn` / `pca` / `walk` / `optimize`（算，不存） |
| `gtv-ext-usecases` | TC1–TC5 工作负载 + motif/CSV 基准（本文 §11.4） |
| `gtv-largeexposure` / `gtv-scenario` / `gtv-governance` / `gtv-enterprise-sql` / `gtv-refdata` | 大额风险、情景、治理、企业 SQL 面、参考数据 |
| `gtv-quantum` / `gtv-crypto` / `gtv-cbom` / `gtv-q` / `gtv-shm` | 量子计算 MCP、PQC、CBOM、q 前端、共享内存传输 |

### 11.4 TC1–TC5：把工作负载做成可复用资产

这五个 case 原本是 2183 行的 example，现在抽成 crate，可被库调用、可出 SQL、可出报表。

| case | 内容 | 门槛 |
|:--|:--|:--|
| TC1 | 跨资产 as-of join | < 5 ms @ 1M |
| TC2 | OFI 滚动 100 | < 2 ms @ 1M |
| TC3 | 洗舱环 ring(3) | < 10 ms @ 500k |
| TC4 | 512 维 KNN + 时序波动率 | < 8 ms |
| TC5 | point-in-time 快照 | < 1 ms @ 5M |

三个设计点值得注意：

1. **`SuiteMode::{Gate, Full}`**：Gate 只跑有门槛的规模，Full 跑全矩阵。
2. **报表内嵌历史基线**（27 行 Rust `RESULTS.md` + 5 行 kdb+），每行都带 `Δ vs baseline` 与 `Δ vs kdb`，避免「数字好看但不知道跟谁比」。
3. **不把数值门槛当 CI 硬断言**。CI 只断正确性、latency 有限、TC5 bin ≤ zone reference、以及每条门槛行必须有历史基线。原因见 §12.4。

---

## 12. 已知限制与技术债

**这一节请重点看。** 设计文档如果只写优点，评审就没有意义。

### 12.1 单机、单写者

`SegmentStore` 的写方法接收 `&mut self`。没有多写者并发控制，没有跨进程锁。当前假设是「一个进程一个 store」。

- 影响：无法多实例同时写同一目录；
- 缓解：读侧可以多进程（mmap 只读）；
- 未做：并发写入、WAL 批量 fsync。触发条件是 Q3 在 Linux 上证明单写者或每帧 fsync 确实是瓶颈——**先测再做**。

### 12.2 K=1 的 overlay 上限

更正层最多叠 1 层 delta（`delta_depth()`）。超过就触发 compaction。

- 影响：高频更正场景（如批量重述）会频繁触发 compaction；
- 缓解：compaction 是两阶段的，`prepare` 在维护线程；
- 未做：Parent+Diff manifest。触发条件写明是「>5000 段且 serialize > 2 ms/发布」，不主动重启。

### 12.3 段数带来的按边成本

实测（跨平台几何常量）：1 / 10 / 100 / 500 段 → 0.8 / 8 / 80 / **400** B/edge。目标是从 1,439 降到 ~145 B/edge。

- 未做：段内节点区间窄化、`(time_bucket, src_range)` 分片；
- 触发条件：段数进入 100–500 **且** B/edge > 256；
- 这是**冻结项**，不是漏项。

### 12.4 性能数字的适用范围

| 数字 | 能说明什么 | 不能说明什么 |
|:--|:--|:--|
| `bench_m6_gate` 毫秒 | 同机 A/B 的加速比 | 对外性能声明 |
| TC1–TC5 gate | 本机 8/9 门槛通过 | 通用硬件上的表现 |
| Linux CI（2 vCPU） | 7/9，趋势一致 | 性能准入（共享 runner 噪声大） |
| SQL/CSV 基准 | DataFusion 55 + 本仓组合根的下限 | **核心仓的性能**（图算子一行未参与） |

历史 `RESULTS.md` 的 3–10× 与 kdb+ 的 19–93× **主要来自跨主机差异**，不能归因于某轮改动。同机 A/B 才有效：TC3/TC4/TC5 共 12 行在 ±6% 内。

### 12.5 未接出 SQL 的能力

- `find_bounded`（显式 `TraversalBudget` / `CancelToken`）未接 SQL，当前上界由 `limit` 提供；
- `MotifSpec::with_sequence()`（边类型序列）只在 Rust API，没有 SQL 数组字面量；
- 段级 `FullHit` 零扫描聚合短路未做（U2 已有 `gtv_range_agg()` / `gtv_active_edge_count()`，缺的是段级短路本身）。

### 12.6 文档时点漂移

三处历史文档仍停在早期时点（`metadata-benchmark.md` §6.2、`storage-mmap-design.md` §9.5、`oceanbase-storage-lessons.md` §6）。它们不影响代码事实，但**评审时请以本文与 `ROADMAP.md` 为准**。

### 12.7 CUDA 尚未开始

GPU 环境已探明（RTX 4060 Laptop，8 GiB，CC 8.9，CUDA 13.0），但没有一行 GPU 代码。

**先给一个可能不中听的判断**：这份基线里最贵的 join（41.7 ms / 1M 行）恰恰最不适合上 GPU。它的成本在字符串哈希与内存带宽，166 MiB 经 PCIe 传一次约 10 ms，还要回传 4.2 倍行数。除非数据常驻显存且多次复用，单次查询不可能赢。

真正 GPU-shaped 的是 **TC4 512 维 KNN**（稠密浮点、无分支、算力受限）和 **TC2 OFI**（纯数值扫描）。所以推进顺序定为「先证伪再动手」：设备抽象层 + `cuda` feature（默认关闭，无 GPU 的 CI 必须继续全绿）→ 传输成本探针 → 按 ROI 做 TC4 → TC2 → CSV 聚合。

---

## 13. 评审问题清单

以下六个是我认为**最可能被推翻**的设计选择。请优先攻击这些，而不是细节实现。

**Q1. 快照级更正 vs 行级更正。**
当前是「快照级 PUBLISH + 位置化 RETRACT」，不是逐行 MVCC。理由是状态重建场景的更正通常是成批的，行级版本链会让存储放大数倍。但如果真实负载是「随机、稀疏、单行更正」，这个选择会让每次更正都生成一个 delta。**请判断：目标场景的更正粒度是什么？**

**Q2. GTVSeg v1 的不可变段模型是否撑得住规模。**
不压缩 + 全量重写 compaction，在 1e9 边级别会让单次 compaction 变成分钟级。当前的分片维度只有时间。**是否应该更早引入 `(time_bucket, src_range)` 二维分片，而不是等段数到 100–500 才做？**

**Q3. `EndpointPolicy` 用 1-bit 掩码。**
省 cache miss，但不支持 per-type 的复杂端点规则。如果监管场景需要「某类边用节点存活期、另一类不用、第三类用自定义窗口」，这个设计要扩容成什么形状？

**Q4. 静态链接的扩展 ABI 是否是死路。**
好处是零运行时开销、类型安全；坏处是不能第三方分发二进制扩展，必须重新编译宿主。**对「第三方评审」这个场景本身，这是不是致命缺陷？**

**Q5. 性能目标是否选错了对手。**
我们只争「图 + 时间」端到端延迟，不比单表向量化带宽。如果评审认为买方实际会用 TPC-H 类查询来评估，这个定位就是自废武功。**请判断：目标买方的第一条验证查询长什么样？**

**Q6. `live_rows_at()` 作为唯一 retract fold。**
好处是等式只在一处证明；坏处是这个函数成为所有读路径的必经点，任何性能优化都要经过它。当前它返回 `Vec<LiveRow>`（物化）。**是否应该提供 streaming / 谓词下推的变体，还是保持「一个正确入口」更值？**

---

## 14. 复现方式

```bash
# 核心：全量回归
cd gtvdb_core && cargo test --workspace --no-fail-fast

# Q1/Q2/Q3 现状表（release）
cargo run --release -p gtv-engine --example bench_m6_gate

# P1-4 motif SQL 准入
cargo run --release -p gtv-engine --example bench_motif_gate

# 零扩展 REPL
cargo run -p gtv-cli --bin gtv

# 扩展仓：TC1–TC5 准入（release）
cd gtvdb_ext
cargo run --release -p gtv-ext-usecases --example bench_tc_usecases   # 可用 TC_MODE=full

# 传统 SQL / CSV 基准（数据需自备，仓库不含 CSV）
GTV_SQLCSV_DATA=/path/to/datablist GTV_SQLCSV_SCALES=100000,500000,1000000 \
  cargo run --release -p gtv-ext-usecases --example bench_sql_csv
```

**数据集不在仓库里**。`customers-*.csv` 与 `organizations-1000.csv` 需自行获取（`datablist/sample-csv-files` 只发布生成器，CSV 走 Drive / LFS 镜像）。行数与 fnv1a-64 校验值见 `gtvdb_ext/doc/sql_csv_benchmark.md` §2，可据此确认拿到的是同一份数据。

---

## 15. 附录

### 15.1 实测数据汇总（截至本文基线）

**A. 传统 SQL / CSV**（Windows / 22 逻辑核 / 31.4 GiB / DataFusion 55）

| 行数 | load | count(CSV) | count | aggregation | join | 峰值 RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 20.3 ms | 11.8 ms | 0.4 ms | 2.0 ms | 6.4 ms | 96 MiB |
| 500,000 | 63.5 ms | 31.5 ms | 0.4 ms | 3.4 ms | 20.3 ms | 191 MiB |
| 1,000,000 | 121.0 ms | 52.6 ms | 0.4 ms | 6.5 ms | 41.7 ms | 310 MiB |

CPU 明细：load 929%、count(CSV) 1339%、aggregation 644%、join 799%、`count` 100%。

读法：`count` 是常量 0.4 ms 且单线程，因为 DataFusion 用 `MemTable` 行数元数据短路，不是扫描——它的「24 亿行/秒」是伪吞吐。真正随行数增长的是 `count(CSV)`。内存约为 CSV 文本的 1.5 倍（166 MiB → 251 MiB 常驻）。

**B. TC1–TC5**（同机 Windows / 8 threads）

| 口径 | 结果 |
|:--|:--|
| gate（9 行） | 8/9 通过，仅 TC2 SIMD 未达门槛 |
| full（24 行） | 9/11，新增 TC4 flat @1M 未达 |
| Linux CI 2 vCPU | 7/9，多一项 TC1 v3 |
| 同机 A/B（library 化前后） | TC3/TC4/TC5 共 12 行在 ±6% 内 |

**C. 批量算子加速比**（M6-a，N=32/128）

| 路径 | N=32 | N=128 |
|:--|--:|--:|
| batch graph（owned，含整图 clone） | 5.7× | 15.8× |
| batch graph（shared，`Arc` 共享） | 27× | 55× |
| motif（owned） | 11.82× | 49.79× |
| motif（shared） | 30.49× | 111.55× |

（batch graph 为 Windows 区间下限，Linux 更高：owned 16–55×、shared 28–78×。）

结论一致：**瓶颈是 N 次 SQL 规划，不是图内核**。

**D. L2 剪枝**（元数据基准）

稀疏时间窗 skip 99.0–99.7%，提速 103–529×；冷启动与段数无关；每段剪枝判定 1–3 ns。

### 15.2 关键文件索引

| 主题 | 文件 |
|:--|:--|
| 定位与取舍 | `docs/positioning.md` |
| 存储格式与写路径 | `docs/storage-mmap-design.md` §3.2 / §8 / §9 |
| 过去更新设计 | `docs/gtvdb-minimal-temporal-update-design.md`、`docs/minimal-temporal-update-impl.md` |
| 元数据对比与基准 | `docs/metadata-design-comparison.md`、`docs/metadata-benchmark.md` |
| 性能路线核查 | `docs/perf-first-roadmap-review.md` |
| 路线图与冻结清单 | `ROADMAP.md` §3 / §4 |
| 扩展 ABI | `crates/gtv-engine/src/ext.rs`（顶部注释即设计说明） |
| 热路径 | `crates/gtv-core/src/csr.rs`、`crates/gtv-core/src/endpoint.rs` |
| TC1–TC5 | `gtvdb_ext/doc/tc1_tc5_usecases.md` |
| SQL/CSV 基准 | `gtvdb_ext/doc/sql_csv_benchmark.md` |

### 15.3 术语

| 术语 | 含义 |
|:--|:--|
| GTVSeg | 定宽、不可变、可 mmap 的段文件格式，v1 |
| run | 段内属于同一 `src` 的连续边区间 |
| zone map | 每个 chunk 的 `max_valid_to` 上界，用于跳过整块过期数据 |
| `RowPos` | `(segment_id, row)` 的 12 B 位置标识，retract 的落点 |
| retract fold | 把撤回集折叠进可见集的唯一函数 `live_rows_at()` |
| PUBLISH / RETRACT / AS OF | 快照级更正的三个原语 |
| U0–U3 | 过去更新系列（原 M0–M3，避免与平台早期 M1–M6 混淆） |
| gate / full | 基准的两种规模模式：只跑门槛规模 / 跑全矩阵 |
| `FullHit` | 整段全部活跃，可零扫描 |
| `core-boundary` | 检查扩展不内嵌内核的 CI job |
