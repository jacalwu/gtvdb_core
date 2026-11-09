# gtvdb_core — 核心路线图

> 2026-11-09 重写｜依据 [`docs/perf-first-roadmap-review.md`](docs/perf-first-roadmap-review.md) §4｜代码基线 `ed7019c`（M6-a 后，53 suites / 494 passed / 0 failed / 1 ignored，CI 5/5）
> 上一版把 compaction / GC 列为未完成、且完全没有记录 L1/L2 与 M0–M3，主线编号停在 M14；本版按事实对账，并把剩余项从「6 个开放里程碑」压缩为「P0 基线 + P1 三项 + 冻结清单」。

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
| Q2 时间窗聚合 | `[T1,T2)` 活跃边 / 敞口聚合 p99，含 `FullHit` 短路比例 | 同基准 B 段（当前为段粒度 B/edge，聚合口径待补完） |
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
- `P0 / P1 / P2`：评审 §4 的**优先级**，不是里程碑编号。

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
- [x] **U2 / M2 控制面**（`2235e8f`）：**两阶段 compaction**、**延迟 GC**、观测版本落盘、L2 聚合去重、`bench_correction`
  - ⚠️ **状态纠正**：上一版把「compaction / 孤儿段 GC」列为 M6 P2–P3 未完成，实际已在 U2 交付、U3 又补了 Tag 对 GC 的 pin 与常驻 maintenance worker。**这两项不再投入。**
- [x] **U3 / M3 准入裁剪**（`13f7367`）：`ActiveNodeBitmap` 可选派生缓存（`NodeLifetime` 仍是事实源）、opt-in 常驻 `MaintenanceService`（phase-1 只 prepare，不写盘 / 不改 manifest / 不持 store 锁）、轻量 Tag + GC pin + `gtv_tags()`。`bench_m3_gate` 判定：Parent+Diff **DEFER**、Branch 只做 Tag（线性 manifest 不支持）、GTVSeg 列扩展 **NO-GO**（变长属性走 sidecar）
- [x] **M6-a / P0 目标查询基准**（`0d7f918`）：`bench_m6_gate` 给出 Q1/Q2/Q3 的 A/B/C 三段现状表，每段带硬断言；CI step 18 实跑，Linux 对照见评审 §7.6
- [x] **M6-a / P1-1 批量图算子**（`0d7f918`）：`batch_graph.rs` 两个可 join / 投影 / 过滤的 `TableProvider`（`gtv_batch_neighbors` / `gtv_batch_khop`），复用 `neighbors_record_batch()` 与 `khop(&seeds,k,t)`；`register_neighbors()` 由两次整图 clone 收敛成一份 `Arc`。7 项 SQL 回归测试与逐点 UDTF **逐条一致**；N=32–128 时 Windows 5.7–15.8×（owned）/ 27–55×（shared），Linux 16–55× / 28–78×。`csr.rs` / `endpoint.rs` / GTVSeg v1 列区零改动

> 全量回归：U0 428 → U1 448 → U2 466 → U3 487 → **M6-a 494**（53 suites，0 failed，1 ignored）。

---

## 3. 当前主线：P0 基线 → P1 三项 → 冻结清单

| 阶段 | 项 | 状态 | 判据 / 触发条件 |
|:--|:--|:--|:--|
| **P0** | 目标查询基准 Q1/Q2/Q3 + 正确性断言 | ✅ M6-a 已交付 | 产出是**一张现状表**，不是代码优化；`perf stat` 与目标规模基准仍未补完（§3.1） |
| **P1-1** | 批量图算子 | ✅ **GO**，M6-a 已交付 | 瓶颈被证明是 N 次 SQL 规划而非图循环；逐条一致后才谈加速 |
| **P1-2** | 段内节点区间窄化 / `(time_bucket, src_range)` 分片 | 🧊 冻结（M6-b） | 持续段数进入 **100–500** 且 **B/edge > 256** 才启动。实测 1/10/100/500 段 → 0.8/8/80/**400** B/edge（跨平台几何常量）；目标 1,439 → ~145 B/edge。验收须复用 `crates/gtv-storage/tests/l2_time_box.rs` 断言，不破坏 `Skip/FullHit/Scan` |
| **P1-3** | 热循环 measured-first 调优 | 🧊 冻结（M6-b） | 先做 Linux `perf stat`（`cache-misses` / `dTLB-load-misses` / `branch-misses`），再**只做被证明是瓶颈的一项**（16B 窄化 / SIMD 时间谓词 / Eytzinger / 大页 / 并行 frontier）；每项 A/B，输了回滚 |

**红线**：`csr.rs` 的 `neighbors()` 热循环、GTVSeg v1 的 `disk == memory == cache line` 定宽列区恒等式、L2 交集/并集两组标量语义不得因优化而合并。U0–U3 用四个里程碑换来的「读路径零回归」是当前最值钱的资产。

### 3.1 欠的是「测」，不是「做」

P0 的以下补完属于测量欠账，**不新增任何算子**：

- Linux `perf stat` 采样（补 `docs/oceanbase-storage-lessons.md` §6 自认的欠账）；
- 把基准规模从当前合成图（V=20k / E=200k）推到 **1e6 节点 / 1e7 边**的三组真实形态查询；数据只在内存 / tempdir 构造，如需落盘一律放仓库外（如 `C:/ai/gtvdb/_benchdata/`），沿用 `.gitignore` + CI generated-data guard；
- 并发查询 p99、WAL fsync 成本拆分、同机列式基线（仅作趋势对照）。

`bench_m6_gate` 的绝对毫秒数只用于同机 A/B 判定，**不得当作对外性能声明**。

---

## 4. 冻结 / 降级清单（不占里程碑）

| 项 | 处理 | 触发条件 |
|:--|:--|:--|
| Parent+Diff Manifest、Branch、day-bucket 缓存、GTVSeg 列扩展 / sidecar、自动 GC 默认开 | 🧊 冻结 | `docs/minimal-temporal-update-impl.md` §14.8 已写死，不主动重启（Parent+Diff：>5000 段且 serialize >2 ms/发布） |
| 并发写入、WAL 批量 fsync、真实 kill -9 进程级测试 | ⏸ 先测 | Q3 在 Linux 上证明单写者 `&mut self` 或每帧 fsync 是瓶颈；现在补的是可靠性，不是速度 |
| L1 `stats.idx`、`[T1,T2)` 原生聚合、冷热淘汰 / `.gtvidx` mmap / L2 Parquet 归档 | ⏸ 先测 | `stats.idx` 属运维面，不在目标查询热路径；其余等规模超过内存再谈 |
| M7 增量 frontier | 🧊 冻结 | 与 AML / CRM / LE 无关；仅 quantum state 有真实分支集需求时再评估 |
| M8 crates.io、M9 pyo3、M10 canonical demo、M12 dylib loader | 🗑 主线下架 | 出现真实嵌入用户后按需；对速度零贡献，且会继续扩大接口面积 |
| 任何新的非图算子 UDTF / 领域功能 | ⛔ 本轮禁止 | 领域一律进 `gtvdb_ext`；核心只保留注册接口 |

**门禁索引**：触发条件与 go/no-go 结论以 [`docs/perf-first-roadmap-review.md`](docs/perf-first-roadmap-review.md) 为准；CI 已有「M6 batch graph operator」（step 17，逐条 parity）与「M6 admission gate」（step 18，目标查询延迟 + 段粒度）两道门禁，generated-data guard 与 core-boundary 保持不变。

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
