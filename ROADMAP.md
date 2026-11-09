# gtvdb_core — 核心路线图

> 2026-11-09 重写｜依据 [`docs/perf-first-roadmap-review.md`](docs/perf-first-roadmap-review.md) §4｜代码基线 `ed7019c`（M6-a 后，53 suites / 494 passed / 0 failed / 1 ignored，CI 5/5）
> 上一版把 compaction / GC 列为未完成、且完全没有记录 L1/L2 与 M0–M3，主线编号停在 M14；本版按事实对账，并把剩余项从「6 个开放里程碑」压缩为「P0 基线 + P1 三项 + 冻结清单」。
> 同日二次纠正（剩余项盘点后）：修正 Q2 / `[T1,T2)` 聚合口径（U2 已有 SQL 出口，缺的是段级 `FullHit` 短路）；补入被评审 §4 漏掉的 **AML motif / 环路 SQL 出口缺口**；补记 CI 工程债与三处历史文档状态过时（§3.2）。
> 同日 P1-4 落地：业主批准把该 motif 缺口纳入主线，已交付逐点 UDTF + 批量表并实测 GO（§3.3）；原「待业主决策」状态已撤销。

嵌入式时序图分析平台核心：在一个 Arrow 表上回答「`T` 时刻这张图长什么样」与「这个状态是怎么变成的」。
本仓只承诺**内核主线**；领域工作负载已拆到独立扩展仓 [`jacalwu/gtvdb_ext`](https://github.com/jacalwu/gtvdb_ext)，不进本路线图。定位理由见 [`docs/positioning.md`](docs/positioning.md) §三（对位表）。

---

## 0. 目标与判定规则

**目标**：保持高性能，做速度上可媲美 kdb 的**时序图**数据库，主用于金融 **AML / CRM / LE** 这类「一批种子 → 时序图求解 → 聚合」的 use case；不把各种功能硬塞进去。

| 缩写 | 本路线图中的含义 | 典型查询形态 |
|:--|:--|:--|
| AML | 反洗钱 | 时序资金流追踪、环路 / motif 检测（k-hop + 时间窗） |
| CRM | 客户关系与担保网络 | 担保 / 抵押链传递、客户集扩散 |
| **LE** | **Large Exposure（大额风险敞口，MA(BS)28）** | 客户群 → 集团 → 敞口聚合；**不指 Law Enforcement** |

**口径**：争「图 + 时间」类查询的**端到端延迟**（kdb 需多次 join / 多段 q 脚本才能表达），不争纯列式算子的绝对带宽。对外只报可复现的查询延迟数字，不写「kdb+ replacement」一类口号。

**三条硬指标**（阈值先看现状表再定，不拍脑袋）：

| 指标 | 口径 | 当前测量出口 |
|:--|:--|:--|
| Q1 批量 k-hop | N 个种子、2–3 hop、指定 `valid_at`，1e7 边级图上的 p50/p99 | `crates/gtv-engine/examples/bench_m6_gate.rs` A 段 |
| Q2 时间窗聚合 | `[T1,T2)` 活跃边 / 敞口聚合 p99，含 `FullHit` 短路比例 | U2 已交付 `gtv_range_agg()` / `gtv_active_edge_count()` SQL 出口（在折叠后的 `AggregateView` 上计算）；`bench_m6_gate` B 段目前只测段粒度 B/edge |
| Q3 摄入 | 单进程 M edges/s、publish→可见延迟 p99 | 同基准 C 段 |

每条都必须带**正确性断言**（对照全量扫描逐条一致）；`Overlay == Materialized` 沿用 `bench_correction` 口径。跨平台数字只比趋势：M2 实测 Windows / Linux 的 fsync 相差约一个数量级，M6-a 复现为 4–6×。

> **判定规则（写进 CI 与 PR 模板）**
> 每个进主干的 item 必须能回答「它让 Q1/Q2/Q3 中哪一条的 p99 降了多少，基准在哪个文件」。答不上来的，进 §4「冻结 / 降级清单」，不进里程碑。

---

## 1. 编号命名空间

历史上有两套 `M` 编号，直接并列会误读，本路线图固定如下写法：

- `M1–M6 / M11 / M13 / M14`：**平台早期里程碑**，§2.1 保留原名不改号。
- `M0 / M1 / M2 / M3`（见 `docs/minimal-temporal-update-impl.md`）：**过去更新系列**，下文记作 **U0–U3**。
- `M6-a / M6-b`：**perf-first 系列**（沿用「做 M6」的叫法）；`-a` 已交付，`-b` 为冻结项。
- `P0 / P1 / P2`：评审 §4 的**优先级**，不是里程碑编号；**P1-4** 是业主在 §3.3 盘点后批准追加的第四项（motif SQL 出口），不属于评审 §4 原本的三项。

---

## 2. 已交付（对账到 HEAD）

### 2.1 平台主线

- [x] **M1 时序核心**：Temporal-CSR（半开区间边 + zone map 剪枝 + run 内二分）、k-hop、direction-optimizing 走访
- [x] **M2 SQL 表面**：DataFusion REPL、`as_of` / `neighbors` / `ann` table functions、`mavg`/`msum` WindowUDF
- [x] **M3 向量**：Flat（AVX2+FMA）/ IVF / HNSW，统一 `Metric` 契约、filter-aware 路由 + exact rerank
- [x] **M4 落地**：Parquet 分区 + atomic rename、SnapshotStore（多版本时间旅行）
- [x] **M5 来源图（state provenance）**：`gtv-core::provenance` + `gtv-core::quantum_state` + `QUANTUM_STATE_*` SQL
- [x] **M6-P0 读路径闭环**：GTVSeg v1 定宽可映射格式 + `Col<T>` mmap 后端 + `register_segment()` / `gtv_segments()` / `gtv_segment_edges()` + `bench_segment` 基线。详 `docs/storage-mmap-design.md` §8
- [x] **M6-P1 写路径闭环**：WAL（定宽 42B/边 + 帧校验和）+ MemSegment 原生 Vec delta + manifest / flush / crash recovery / 段级复用 / `graph_at(T)` + `register_segment_store()`。写路径零编码零压缩、不产生也不提交 CSV。详 `docs/storage-mmap-design.md` §9
- [x] **M11 扩展机制**：`gtv_engine::ext::GtvExtension` + `ExtensionRegistry` + `gtv_extensions()`；领域工作负载移出核心仓
- [x] **M13 核心仓拆分**：拆出独立的 `gtvdb_core`（干净历史、零扩展依赖、可独立 `cargo test --workspace`），并以 CI 守卫固定边界
- [x] **M14 扩展仓拆分**：拆出独立的 `gtvdb_ext`（13 个扩展 crate + full-pack CLI/server），以固定 commit 的 git dependency 单向依赖本仓；两仓各有 CI

### 2.2 元数据与过去更新系列（上一版完全漏记）

- [x] **L1 / L2 元数据**（`310ead3`）：L1 catalog surface（`gtv_tables()` / `gtv_snapshots()` / `gtv_files()`）、L2 段时间盒（`SegmentMeta` 的 `Skip / FullHit / Scan` 判定 + zone map）。稀疏窗口 skip 99.0–99.7% → 103–529×，冷启动与段数无关。详 `docs/metadata-design-comparison.md`、`docs/metadata-benchmark.md`
- [x] **U0 / M0 端点策略**（`7b812ff`）：`endpoint.rs` 的 `EndpointPolicy` 1-bit、`NodeLifetime` 定宽数组、`EndpointView::active_neighbors()`；三原语 PUBLISH / RETRACT / AS OF。热路径包围盒命中 1.00x，`csr.rs` 一行未改
- [x] **U1 / M1 位置化更正**（`e0aa33c`）：`SnapshotDelta` / `RetractSet` / `RowPos` 定宽 `.gtvd`、WAL v2 `OP_MUTATION`、`live_rows_at()` 唯一 retract fold、K=1 bounded overlay、`gtv_invalidations([cursor])`
- [x] **U2 / M2 控制面**（`2235e8f`）：**两阶段 compaction**、**延迟 GC**、观测版本落盘、L2 聚合去重、`gtv_range_agg()` / `gtv_active_edge_count()` SQL 出口、`bench_correction`
  - ⚠️ **状态纠正**：上一版把「compaction / 孤儿段 GC」列为 M6 P2–P3 未完成，实际已在 U2 交付、U3 又补了 Tag 对 GC 的 pin 与常驻 maintenance worker。**这两项不再投入。** 另有两份历史文档仍停在 P1 时点，见 §3.2。
- [x] **U3 / M3 准入裁剪**（`13f7367`）：`ActiveNodeBitmap` 可选派生缓存（`NodeLifetime` 仍是事实源）、opt-in 常驻 `MaintenanceService`（phase-1 只 prepare，不写盘 / 不改 manifest / 不持 store 锁）、轻量 Tag + GC pin + `gtv_tags()`。`bench_m3_gate` 判定：Parent+Diff **DEFER**、Branch 只做 Tag（线性 manifest 不支持）、GTVSeg 列扩展 **NO-GO**（变长属性走 sidecar）
- [x] **M6-a / P0 目标查询基准**（`0d7f918`）：`bench_m6_gate` 给出 Q1/Q2/Q3 的 A/B/C 三段现状表，每段带硬断言；CI step 18 实跑，Linux 对照见评审 §7.6
- [x] **M6-a / P1-1 批量图算子**（`0d7f918`）：`batch_graph.rs` 两个可 join / 投影 / 过滤的 `TableProvider`（`gtv_batch_neighbors` / `gtv_batch_khop`），复用 `neighbors_record_batch()` 与 `khop(&seeds,k,t)`；`register_neighbors()` 由两次整图 clone 收敛成一份 `Arc`。7 项 SQL 回归测试与逐点 UDTF **逐条一致**；N=32–128 时 Windows 5.7–15.8×（owned）/ 27–55×（shared），Linux 16–55× / 28–78×。`csr.rs` / `endpoint.rs` / GTVSeg v1 列区零改动
- [x] **P1-4 / `gtv-pattern` SQL 出口**：`crates/gtv-engine/src/motif.rs` 把已存在的 `find` / `find_from` / `find_windowed*` 接出 SQL：`motif(shape, t [, limit [, window]])` / `motif_from(shape, seed, t [, ...])` 两个逐点 UDTF，加可 join 的 `gtv_motif_matches`（一行一 match）与 `gtv_motif_edges`（一行一 matched edge）两张批量表；`register_motif()` / `register_motif_batch()` / `register_motif_batch_shared()`。9 项集成测试逐条比对 Rust 内核，`bench_motif_gate` 实测 Q1 GO。详 §3.3

> 全量回归：U0 428 → U1 448 → U2 466 → U3 487 → M6-a 494 → **P1-4 503**（54 suites，0 failed，1 ignored）。

---

## 3. 当前主线：P0 基线 → P1 三项 → 冻结清单

| 阶段 | 项 | 状态 | 判据 / 触发条件 |
|:--|:--|:--|:--|
| **P0** | 目标查询基准 Q1/Q2/Q3 + 正确性断言 | ✅ M6-a 已交付 | 产出是**一张现状表**，不是代码优化；`perf stat` 与目标规模基准仍未补完（§3.1） |
| **P1-1** | 批量图算子 | ✅ **GO**，M6-a 已交付 | 瓶颈被证明是 N 次 SQL 规划而非图循环；逐条一致后才谈加速 |
| **P1-2** | 段内节点区间窄化 / `(time_bucket, src_range)` 分片 | 🧊 冻结（M6-b） | 持续段数进入 **100–500** 且 **B/edge > 256** 才启动。实测 1/10/100/500 段 → 0.8/8/80/**400** B/edge（跨平台几何常量）；目标 1,439 → ~145 B/edge。验收须复用 `crates/gtv-storage/tests/l2_time_box.rs` 断言，不破坏 `Skip/FullHit/Scan` |
| **P1-3** | 热循环 measured-first 调优 | 🧊 冻结（M6-b） | 先做 Linux `perf stat`（`cache-misses` / `dTLB-load-misses` / `branch-misses`），再**只做被证明是瓶颈的一项**（16B 窄化 / SIMD 时间谓词 / Eytzinger / 大页 / 并行 frontier）；每项 A/B，输了回滚 |
| **P1-4** | `gtv-pattern` motif / 环路 SQL 出口（业主追加） | ✅ **GO**，已交付 | 逐点 UDTF 与批量表逐条一致；`bench_motif_gate` Q1 在 N=32/128 为 11.82× / 49.79×（owned）、30.49× / 111.55×（shared）。详 §3.3 |

**红线**：`csr.rs` 的 `neighbors()` 热循环、GTVSeg v1 的 `disk == memory == cache line` 定宽列区恒等式、L2 交集/并集两组标量语义不得因优化而合并。U0–U3 用四个里程碑换来的「读路径零回归」是当前最值钱的资产。

### 3.1 欠的是「测」，不是「做」

P0 的以下补完属于测量欠账，**不新增任何算子**：

- Linux `perf stat` 采样（补 `docs/oceanbase-storage-lessons.md` §6 自认的欠账）；
- 把基准规模从当前合成图（V=20k / E=200k）推到 **1e6 节点 / 1e7 边**的三组真实形态查询；数据只在内存 / tempdir 构造，如需落盘一律放仓库外（如 `C:/ai/gtvdb/_benchdata/`），沿用 `.gitignore` + CI generated-data guard；
- **Q2 的真实时间窗聚合 p99 与 `FullHit` 短路比例**：U2 已有聚合 UDTF，但当前 B 段只测段粒度 B/edge；段级 `FullHit` 零扫描路径属 §4 冻结项，两者不要混为一谈；
- 并发查询 p99、WAL fsync 成本拆分、同机列式基线（仅作趋势对照）。

`bench_m6_gate` 的绝对毫秒数只用于同机 A/B 判定，**不得当作对外性能声明**。

### 3.2 工程与文档欠账（不占里程碑）

| 项 | 现状 | 处理 |
|:--|:--|:--|
| CI release 编译成本 | step 18 是 kernel job 第一个需 release 编译 `gtv-engine`（DataFusion）的步骤，每个 run 约 +20 分钟；rust-cache 未跨 run 命中（评审 §7.6） | 拆独立 job + 复用 release cache，或仅在 `gtv-core` / `gtv-engine` / `gtv-storage` 路径变更时触发。属 CI 工程优化，不改变任何基准结论 |
| `docs/metadata-benchmark.md` §6.2 第 4 项 | 仍把 **compaction** 列为「下一步」，实际 U2 已交付 | 同步状态，保留历史基准数值 |
| `docs/storage-mmap-design.md` §9.5 | 仍把 **compaction / 孤儿段 GC** 列为未做，实际 U2 已交付、U3 补 GC pin | 同步状态 |
| `docs/oceanbase-storage-lessons.md` §6 / 脚注 17 | 引用的 `crates/gtv-ingest/src/dedup.rs:7` TODO 注释已不存在 | 更新引用；「全库无 bloom filter 实现」的结论仍成立 |

> 这三处是历史文档的时点快照，不影响代码事实；当前状态以本路线图与评审记录为准。修正它们**不算新增开发项**。

### 3.3 已交付（P1-4）：AML 环路 / motif 的 SQL 出口

**背景**：`gtv-pattern` 已有 `find` / `find_from` / `find_windowed` / `find_windowed_from` / `find_bounded`（时序 motif、环路、有界匹配），但 `gtv-engine` 对它零引用（评审 §3.3 第 2 点）。业主 2026-11-09 批准纳入主线；这是「核心已有、SQL 缺出口」，与 M6-a 同类，**没有新造匹配算法**。

| 出口 | 形态 | 对应内核 |
|:--|:--|:--|
| `motif(shape, valid_at [, limit [, window]])` | 逐点 UDTF，全图扫描，字面量参数 | `find` / `find_windowed` |
| `motif_from(shape, seed, valid_at [, limit [, window]])` | 逐点 UDTF，锚定单种子；Q1 的逐点基线 | `find_from` / `find_windowed_from` |
| `gtv_motif_matches` | 可 join 批量表，一行一 match：`match_id`、`n0..n{k-1}`、`event_min`、`event_max` | 同一次 `Vec<Match>` |
| `gtv_motif_edges` | 可 join 批量表，一行一 matched edge：`match_id`、`edge_pos`（路径内位置）、`edge_idx`（CSR 槽位）、`src`、`dst`、`edge_type`、有效区间 | 同一次 `Vec<Match>` |

`shape` 为 `ring3` / `ring4` / `path2` / `path3` / `path4` / `diamond`；`MotifSpec` 另支持滚动时间窗 `with_window()` 与按事件时间排序的边类型序列 `with_sequence()`（SQL 侧只暴露 `window` 可选参数）。注册走 `register_motif()` / `register_motif_batch()` / `register_motif_batch_shared()`；两张批量表由**同一次 `Vec<Match>`** 物化，不重复跑 DFS。

**验收（原 §3.3 口径逐条兑现）**：复用 `find_*` 内核 ✅；一次规划、一个 RecordBatch ✅；与 Rust API 逐条一致（`crates/gtv-engine/tests/p1_4_motif_sql.rs` 9 项：逐点 UDTF vs `find_from`、全图 UDTF vs `find`、批量 vs N 次逐点、边表重建 `Match.edges`、滚动窗口 vs `find_windowed_from`、类型序列、shape 变体、join/投影/过滤下推、空集与 limit）✅；Q1 p50/p99 A/B（`crates/gtv-engine/examples/bench_motif_gate.rs`）✅；未新增任何非图算子 ✅。

| seeds | 逐点 p50 / p99 µs | owned µs | shared µs | spd owned | spd shared |
|--:|--:|--:|--:|--:|--:|
| 1 | 302.9 / 302.9 | 717.3 | 246.1 | 0.42× | 1.23× |
| 8 | 248.1 / 296.3 | 788.8 | 258.5 | 2.59× | 7.90× |
| 32 | 241.1 / 382.2 | 693.3 | 268.7 | 11.82× | 30.49× |
| 128 | 246.4 / 454.0 | 709.3 | 316.6 | 49.79× | 111.55× |

Windows 本机 release，合成账户图 V=20,000 / `ring3` / `valid_at=100`；硬断言 `pointwise == batch == gtv_pattern::find_from`。与 M6-a 同结论：**瓶颈是 N 次 SQL 规划，不是图内核**；N=1 时 owned 仍慢（整图 clone），shared 路径已 >1。门禁用保守的 owned 路径，N≥32 必须 >1，否则 CI 失败。

**边界**：`find_bounded`（显式 `TraversalBudget` / `CancelToken`）仍未接出 SQL，当前上界由 `limit` 提供；`sequence` 只在 Rust `MotifSpec` 可用，未做 SQL 数组字面量；`neighbors()` 热循环与 GTVSeg v1 列区零改动（`csr.rs` 在 M1 中只加了 `Neighbor::edge_idx` 与预算返回语义，见 §3.4）。

### 3.4 已交付（M1）：候选集 / 边身份 / 预算语义

三项通用原语，为 AML 一类「图 + 时间 + 向量」组合查询补齐**算子之间的公共值**。方案与裁定过程见 [`docs/aml-core-ext-enablement.md`](docs/aml-core-ext-enablement.md) §3 与 §7.1。

| 项 | 交付 | 关键事实 |
|:--|:--|:--|
| `Sel` 统一候选集 | `gtv-core/src/selection.rs`：`Dense` / `Bitmap` / `Ranges` 三表示，按代价自动选；`to_mask()` 一次转 Arrow，即 CUDA 的 H2D 输入 | 携带 `snapshot_id`，跨快照喂给 `khop` 直接 `SnapshotMismatch`；`UNBOUND_SNAPSHOT` 为无快照上下文放行。`khop_bounded(&Sel, …)` 为主入口，`khop_bounded_ids(&UInt64Array, …)` 保留兼容 |
| `Neighbor::edge_idx` | `csr.rs`：`u32` 槽位，**结构体仍是 32 B**（`const` 断言守门）；`MatchedEdge` 同步带出 | 解决"同源同目标同区间的两笔转账不可区分"；转置图的槽位不是正排身份，以 `u32::MAX` 表示未知并在 `needs_edge_identity()` 时强制 push |
| 预算语义 | `traversal.rs`：`Truncation` / `TraversalOutcome` / `TriState{Found, NoMatch, Incomplete}`；`khop_*` 与 `find_bounded` 返回**残缺结果 + 截断原因**，不再丢结果报错 | `TruncReason` 含 `Edges/Nodes/Frontier/Rows/Hops/Memory/Deadline`；`max_hops` 少跑也计作截断。SQL 的 `khop` 无三态通道，遇截断直接报错而不是返回空行集 |

**验收**：`cargo test --workspace` 521 项全绿（C1 后为 571 项），其中新增 10 项（`selection::tests` 9 项 + 边身份 4 项 + 预算三态 3 项，含 `p1_4_motif_sql` 中新增的槽位回址断言）。**未做**：Top-K 证据路径、provenance 三层拆分（M2）；CUDA 一切实现（M4 之后）。

---

## 4. 冻结 / 降级清单（不占里程碑）

| 项 | 处理 | 触发条件 |
|:--|:--|:--|
| Parent+Diff Manifest、Branch、day-bucket 缓存、GTVSeg 列扩展 / sidecar、自动 GC 默认开 | 🧊 冻结 | `docs/minimal-temporal-update-impl.md` §14.8 已写死，不主动重启（Parent+Diff：>5000 段且 serialize >2 ms/发布） |
| 并发写入、WAL 批量 fsync、真实 kill -9 进程级测试 | ⏸ 先测 | Q3 在 Linux 上证明单写者 `&mut self` 或每帧 fsync 是瓶颈；现在补的是可靠性，不是速度 |
| L1 `stats.idx`、段级 `FullHit` 零扫描聚合短路（`[T1,T2)` UDTF 已于 U2 交付）、冷热淘汰 / `.gtvidx` mmap / L2 Parquet 归档 | ⏸ 先测 | `stats.idx` 属运维面，不在目标查询热路径；其余等规模超过内存再谈 |
| M7 增量 frontier | 🧊 冻结 | 与 AML / CRM / LE 无关；仅 quantum state 有真实分支集需求时再评估 |
| M8 crates.io、M9 pyo3、M10 canonical demo、M12 dylib loader | 🗑 主线下架 | 出现真实嵌入用户后按需；对速度零贡献，且会继续扩大接口面积 |
| 任何新的非图算子 UDTF / 领域功能 | ⛔ 本轮禁止 | 领域一律进 `gtvdb_ext`；核心只保留注册接口 |
| CUDA 加速（C2 起） | 🧊 C1 已交付；**C2 被 materialize 阻塞** | [`docs/cuda-acceleration-plan.md`](docs/cuda-acceleration-plan.md) §9.3：上载侧很干净（97.3% 是 PCIe、7.75 GB/s、无 AoS/无转换/无 staging，`dst` 收窄 5.9→4.9 次仅在窄列由上游提供时成立），但 `graph_at` 在 1e7 边上要 **1 579 ms**——是上载的 51 倍、CPU 全扫的 230 倍，其中 1 081 ms 是 `Vec<EdgeRecord>` 的 AoS 往返。端到端盈亏是 **255 次/快照**，不是上载内部的 4.9 次。先修 materialize，再写 kernel |

**门禁索引**：触发条件与 go/no-go 结论以 [`docs/perf-first-roadmap-review.md`](docs/perf-first-roadmap-review.md) 为准；CI 已有「M6 batch graph operator」（step 17，逐条 parity）、「M6 admission gate」（step 18，目标查询延迟 + 段粒度）与「P1-4 motif SQL」（step 19，9 项 parity + Q1 准入基准）三道门禁，generated-data guard 与 core-boundary 保持不变。§3.2 的工程/文档欠账**不占里程碑**；§3.3 已被业主批准纳入 P1-4 并交付。

---

## 5. 明确不做（核心仓）

| 项目 | 理由 |
|:--|:--|
| 银行监管 / HFT / 量子求解 / PQC / q 前端 | 工作负载，留在扩展仓；核心只保留注册接口 |
| 分布式（shard / planner / worker） | 目前只是远程执行 SQL；无真实多机工作负载前不承诺 |
| 提交任何生成数据集 | 历史教训：两个 >50MB CSV blob 曾拖慢每次 clone；改为生成器 + 校验和，CI guard 固定边界 |
| 热层压缩 | mmap 后必须能零解码直接算，见 `docs/oceanbase-storage-lessons.md` |
| **以纯列式算子带宽定义性能目标** | kdb+/q 的强项是单表向量化扫描与 asof join；用对手最强的算子定义胜负，会把自己拖进一场不需要打的仗。只争「图 + 时间」端到端延迟 |
| **为做基准而改读路径** | P0 只测，A/B 之前不改 `csr.rs`；正确性优先于速度 |

---

## 6. Workspace 边界

```
crates/                    # 平台（本仓全部内容）
├── gtv-core / gtv-array / gtv-index / gtv-index-store / gtv-storage / gtv-delta
├── gtv-pattern / gtv-catalog / gtv-ingest / gtv-udf / gtv-engine / gtv-proto
└── gtv-cli                # 精简 REPL，零扩展依赖（bin: gtv）
```

`cargo check -p gtv-engine` 不拉进任何领域 crate；CI `core-boundary` job 同时检查目录树、manifest 依赖与 `cargo tree`。
