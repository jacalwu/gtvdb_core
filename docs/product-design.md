# gtvdb_core 产品设计文档

> **基线**：`main` @ `58013c3`（2026-11-15）｜ 188 个 `.rs` / 65,587 行 / 37 个集成测试文件 ｜ 回归 **681 passed / 66 suites / 0 failed**
> **本文回答四个问题**：这是什么产品、已经能做什么、为什么这样设计、下一步做什么。
> **事实源顺序**：代码 > `ROADMAP.md` > 其他设计文档。凡标不出代码落点的能力，一律按「未交付」处理——本仓历史上多次出现「文档宣称已接线、代码其实没有」，本文不重复这个错误。

---

## 0. 文档定位

| 文档 | 回答的问题 | 与本文的关系 |
|:--|:--|:--|
| **本文** | 产品有什么能力、给谁用、边界在哪、下一步做什么 | 产品面唯一入口 |
| [`design-review-2026-11.md`](design-review-2026-11.md) | 架构怎么实现、正确性怎么保证、评审问题清单 | 技术纵深，本文只引用结论 |
| [`ROADMAP.md`](../ROADMAP.md) | 里程碑编号、每项的判据与实测数字 | 本文 §6 的原始出处 |
| [`positioning.md`](positioning.md) | 为什么这样定位、对外怎么说 | 本文 §1 的原始出处（含一处与 ROADMAP 的冲突，见 §9） |

**状态图例**

| 标记 | 含义 |
|:--|:--|
| ✅ | 已交付，有单测 / 集成测试 / 门禁之一为证 |
| 🟡 | 已交付但带硬限制，限制写在 §7 |
| ⚪ | 库级已交付，**未接任何产品表面**（无 SQL、无 CLI、无依赖者） |
| ⏳ | 已排期或有明确触发条件 |
| 🧊 | 冻结：有触发条件，未触发不做 |
| 🗑 | 已下架，不占主线 |

---

## 1. 产品定义

### 1.1 一句话

> gtvdb 是**嵌入式时序图分析引擎**：在同一个 Arrow 表上，回答「`T` 时刻这张图长什么样」与「这个状态是怎么一步步变成现在的」。

不是图数据库（不比 Cypher/ACID/HA），不是时序数据库（不比单表扫描带宽），不是向量数据库（不比 recall/规模）。争的是**「图 + 时间」类查询的端到端延迟**——这类查询在 kdb+ 需要多段 q 脚本、在 Neo4j 需要 APOC 拼装、在「图库 + 向量库」组合里要跨系统搬两次数据。

### 1.2 目标用户与场景

| 用户 | 他们的问题 | 用到的能力（§3 落点） |
|:--|:--|:--|
| 量化 / 风控数据工程师 | point-in-time 特征 join；一批种子做 2–3 hop 资金流追踪 | `as_of` / `khop` / `motif` / `asof_join` (§3.2–3.4, §3.8) |
| 图 + 向量混合检索 | 先按时间切片，再在切片内做语义检索，再沿边扩散 | `as_of` + `knn`/`ann` + `khop` 同一 session (§3.5) |
| 审计 / 合规 / 实验调试 | 这个状态从哪来？由哪些 gate 造成的？这条更正是谁在什么时候改的 | `provenance` / `quantum_state_*` / `bitemporal_*` / lineage (§3.4, §3.9, §3.13) |
| 平台嵌入方 | 把内核放进自己的进程，不另起服务 | `GtvContext` + `GtvExtension` ABI (§3.12) |
| 单机大数据量用户 | 1e6 节点 / 1e7 边，内存装不下，但要求冷启动 O(1) | GTVSeg mmap + 三级分层 (§3.6, §3.10) |

### 1.3 三个支柱（每条都能当场跑出来）

| 支柱 | 主张 | 代码落点 |
|:--|:--|:--|
| **时间是一等字段** | 边带半开区间 `[valid_from, valid_to)`；`as-of T` 是**索引操作**，不是全表扫描 | `gtv-core/src/csr.rs`：zone map（`valid_to` 非单调 ⇒ 用 `max valid_to` 剪枝 + run 内二分）；`gtv-engine/src/bitemporal.rs` |
| **来源可解释** | 每个状态可回溯到产生它的路径与概率，代价 `O(depth)` 而非 `O(状态数)` | `gtv-core/src/provenance.rs`、`gtv-core/src/quantum_state.rs`、`gtv-engine/src/quantum_state.rs` |
| **嵌入式分析** | 进程内、Arrow 零拷贝、DataFusion SQL、不可变段 mmap，**没有集群要运维** | `gtv-engine/src/context.rs::GtvContext`、`gtv-cli` |

### 1.4 G / T / V 是三个维度，不是一个数据库

三者落在**同一条 `RecordBatch`** 上：

- **G** —— 边（谁指向谁）
- **T** —— 边的有效区间（何时成立）
- **V** —— 节点/边上的向量（语义相近）

所以「时间切片 → 向量检索 → 沿边遍历」不需要跨系统搬数据。这是护城河的一句话版本，也是竞品最难补的部分：**它不是功能，是数据布局**。

### 1.5 竞争对位

| | 原生时间区间边 | 图遍历 | 列式 / 嵌入式 | 状态来源回溯 | 向量检索 |
|:--|:--:|:--:|:--:|:--:|:--:|
| **gtvdb** | ✅ 半开区间 + 双时间 | ✅ 时序 CSR、k-hop、direction-optimizing | ✅ Arrow / Parquet，进程内 | ✅ 路径 + 概率 | ✅ Flat / IVF / HNSW + filter-aware |
| Neo4j | ❌（靠属性 / APOC 模拟） | ✅ | ❌（服务） | △（需自建） | △（外挂） |
| kdb+/q | △（时间序列，非区间图） | ❌ | ✅ | ❌ | ❌ |
| DuckDB | ❌ | ❌（递归 CTE 无索引） | ✅ | ❌ | ✅（vss） |
| Zep / Graphiti | △（服务层时序知识图） | ✅ | ❌ | △ | ✅ |

**对外不写**「kdb+ replacement」——把自己最强的对手写进 tagline，等于请买方用对手的标准评估自己。

---

## 2. 产品形态与边界

### 2.1 两仓边界

| 仓库 | 内容 | 依赖方向 |
|:--|:--|:--|
| **`jacalwu/gtvdb_core`**（本仓） | **14 个 lib crate** + 精简 `gtv` REPL + 核心文档/测试 | 不依赖任何扩展 |
| `jacalwu/gtvdb_ext` | 13 个领域 crate（银行监管 / HFT / 量子求解 / PQC）、full-pack CLI、gRPC server | 钉死本仓 commit 的单向 git dependency，经 `GtvExtension` 接入 |

> README §1 仍写「12 个内核 lib crate」，实际是 14 个（M16 加了 `gtv-params`、M17 加了 `gtv-accelerator`）。属文案欠账，见 §9。

边界由 CI 的 `core-boundary` job 三重守住：目录树、manifest 依赖、`cargo tree`。

### 2.2 交付形态

```
进程内库    gtv-core / gtv-index / gtv-storage / gtv-engine …   → 被嵌入
SQL 表面    DataFusion SessionContext（31 个常驻表函数，见 §3.3）→ 被查询
REPL        gtv（本仓唯一 bin，零扩展依赖）                    → 被试用
脚本层      examples/python/gtv_repl.py（Python 客户端，驱动 REPL）→ 被编排
落盘格式    GTVSeg v1（图）/ GTVVEC / GIVFCMP（向量）/ Parquet / JSON manifest
```

脚本层是**客户端而非绑定**：没有 PyO3 模块，Python 通过 stdin/stdout 驱动 `gtv` 子进程（哨兵查询定界、`stderr` 合并、按边框列位解析表格）。因此它继承 REPL 的全部边界——图遍历、建索引、`attach_catalog` 都不可达。真绑定（M9 pyo3）已下架，见 [`user-guide.md`](user-guide.md) §12。

**没有的**：常驻服务端、gRPC server（`gtv-proto` 有传输层但未接线）、SQL 网关、集群、多写者、用户权限系统。扩展仓可以自己包一层 server，但核心不承诺。

### 2.3 明确不做（核心仓）

| 项 | 理由 |
|:--|:--|
| 银行监管 / HFT / 量子求解 / PQC / q 前端 | 是工作负载不是内核；留在扩展仓，核心只留注册接口 |
| 分布式（shard / planner / worker） | 当前「分布式」只是远程执行 SQL，没有真实多机负载前不承诺 |
| 提交任何生成数据集 | `*.csv` / `*.seg` / `*.parquet` / `*.pdf` 一律不入库；数据靠生成器 + `SHA256SUMS` 重建 |
| 热层压缩 | mmap 后必须零解码直接算，见 [`oceanbase-storage-lessons.md`](oceanbase-storage-lessons.md) |
| 以纯列式算子带宽定义性能目标 | 那是 kdb+ 的最强项；用对手最强算子定义胜负是自伤 |
| 为做基准而改读路径 | P0 只测；A/B 之前不动 `csr.rs` |

---

## 3. 已交付功能

### 3.1 能力总览

| 能力域 | 产品入口 | 状态 |
|:--|:--|:--|
| 时序图内核（Temporal CSR） | `neighbors` / `khop` / `gtv_batch_*` | ✅ |
| 双时间（业务时间 × 系统时间） | `as_of` / `bitemporal_as_of` / `bitemporal_overlaps` | ✅ |
| 时序 motif / 环路 | `motif` / `motif_from` / `gtv_motif_*` | ✅ |
| 向量检索（精确 / 近似 / 治理） | `knn` / `ann` / `embedding_search` | ✅ |
| 向量索引持久化与冷开 | `.knn-index` / `.ann-index` / `.knn-save` | ✅ |
| mmap 段存储 + WAL 写路径 | `gtv_segments` / `gtv_segment_edges` / `.segment` / `.store` | ✅ |
| 过去更新（U0–U3） | `gtv_invalidations` / `gtv_tags` / `gtv_range_agg` | ✅ |
| L1/L2 元数据 | `gtv_tables` / `gtv_snapshots` / `gtv_files` | ✅ |
| 状态来源图 | `quantum_state_summary/frontier/reconstruct` | ✅ |
| 装置加速层（VRAM / 流水线 / heat pump） | `.accel` 遥测；`ACCEL_*` 参数 | ✅ 🟡 |
| GPUDirect 直传 | — | 🟡 仅策略接缝，未实测 |
| 部署参数档 | `gtv-params` + `.params` | ✅ |
| 扩展 ABI | `GtvExtension` / `gtv_extensions()` | ✅ |
| 治理面（DQ / 健康 / 工作负载 / CBO） | `dq_*` / `health_check` / `workload_status` / `cbo_explain` | ✅ |
| LSM delta / 微批摄入 / WASM UDF / gRPC 传输 | — | ⚪ 库级已交付，**零依赖者**（§3.14） |

### 3.2 时序图内核

`gtv-core/src/csr.rs` —— 不可变 Temporal-CSR，是全仓最值钱的资产，也是红线所在。

| 能力 | 实现事实 |
|:--|:--|
| 半开区间边 | `[valid_from, valid_to)`，`valid_to` 非单调 ⇒ zone map 存 `max valid_to` 做 chunk 剪枝 + run 内二分 |
| `neighbors(src, t)` | 2 次比较 + chunk zone map 命中即返回；热路径包围盒命中 **1.00×**（零回归） |
| k-hop | frontier 迭代；`Sel` 候选集可跨算子传递；`khop_bounded(&Sel, …)` 为主入口 |
| direction-optimizing | 高入度节点切换为反向按需 push |
| 边身份 | `Neighbor::edge_idx`：`u32` CSR 槽位，**结构体仍是 32 B**（`const` 断言守门）；解决「同源同目标同区间的两笔转账不可区分」 |
| 预算语义 | `Truncation` / `TraversalOutcome` / `TriState{Found,NoMatch,Incomplete}`；截断返回**残缺结果 + 原因**（`Edges/Nodes/Frontier/Rows/Hops/Memory/Deadline`），不丢结果报错 |
| 候选集 | `Sel`：`Dense` / `Bitmap` / `Ranges` 三表示按代价自动选；携带 `snapshot_id`，跨快照喂 `khop` 直接 `SnapshotMismatch` |

**红线**：`csr.rs::neighbors()` 热循环、GTVSeg v1 的 `disk == memory == cache line` 定宽列区恒等式、L2 交集/并集两组标量语义，**不得因优化而合并**。

### 3.3 SQL 表面全清单

`gtv-engine` 在 `GtvContext::new()` 里一次性注册。点函数（UDTF）**只接受字面量参数**——这是刻意的：非字面量会被明确拒绝，避免「看起来能 join 其实每行重规划」。

**图 / 时间**

| 调用 | 语义 | 备注 |
|:--|:--|:--|
| `neighbors(src, valid_at)` | `valid_at` 时活跃的邻居，列形同 `edges` 表 | 需先 `register_neighbors(csr)` |
| `khop(seed, k, valid_at)` | k 跳可达 | 同上 |
| `gtv_batch_neighbors` | 可 join / 投影 / 过滤的批量表 `(src,dst,edge_type,valid_from,valid_to)` | `register_batch_graph(_shared)` |
| `gtv_batch_khop` | 批量表 `(seed, hop, dst)` | 同上；一次规划、一个 RecordBatch |
| `motif(shape, valid_at [, limit [, window]])` | 全图 motif / 环路扫描 | `shape` ∈ `ring3 ring4 path2 path3 path4 diamond` |
| `motif_from(shape, seed, valid_at [, …])` | 单种子锚定 | Q1 的逐点基线 |
| `gtv_motif_matches` / `gtv_motif_edges` | 一行一 match / 一行一 matched edge | 同一次 `Vec<Match>` 物化，不重跑 DFS |
| `asof_join(t0, t1, …)` / `aj` | kdb+ 式 asof join；单列 `(t,value)` 或多列 + `tolerance` | 单调归并 `O(L+R)`，非逐行二分 |
| `mavg` / `msum` / `deltas` | 窗口 UDF（WindowUDF） | `udf.rs::window_udfs()` |
| `trunc` | 时间截断标量函数 | `datetime.rs` |

**向量（三表面，分工见 §3.5）**

| 调用 | 语义 |
|:--|:--|
| `knn(name, query, k [, label])` / `vector_search` | 精确 KNN；第 4 参数是**标签字符串**过滤（KDB.AI `metadata_filtering` 对位） |
| `ann(name, query, k [, metric])` / `ann_search` | Flat / IVF / HNSW，近似 + exact rerank，可 `force_exact` |
| `ann_explain` | 策略路由 / 召回估计遥测 |
| `embedding_search(name, query, k [, tenant [, as_of]])` | 治理集合：带 provenance、`tenant_id`、`effective_from/to` |
| `cbo_explain` | 关系 / 图 / 向量三族统计合成一个计划选择（filter-first vs ANN-first） |

**存储 / 元数据 / 治理**

| 调用 | 语义 |
|:--|:--|
| `gtv_segments()` | 已打开的段（mmap 大小、checksum、reuse_ratio） |
| `gtv_segment_edges('name')` | 一个段的边物化成 Arrow 表（注意：**物化**路径，非零拷贝） |
| `gtv_tables()` / `gtv_snapshots()` / `gtv_files()` | L1：以 `FsCatalog` 为唯一权威，不再靠侧信道猜 |
| `gtv_invalidations([cursor])` | 提交日志的只读投影；事件**就是**已发布的 `SnapshotDelta`，无事件总线、天然幂等 |
| `gtv_tags()` | 命名快照标签（同一 manifest 的另一投影） |
| `gtv_active_edge_count(t [, system_t])` | `t` 时刻活跃的**去重后**逻辑边数 |
| `gtv_range_agg(t1, t2, bucket [, system_t])` | 半开桶聚合；合并相接区间，每个 `(src,dst,type)` 每桶只计一次 |
| `dq_report` / `dq_check` / `dq_gate` | 诊断 → 校验 → 发布门禁（失败产出 `DqFailure`：规则/列/观测值/阈值） |
| `health_check('t'[, max_age_days[, min_rows]])` / `strategy_stats` | 日常体检 / 命中率·Brier·ECE·PSI 漂移 |
| `workload_status()` | 工作负载类、资源组、准入与协作式抢占状态 |
| `quantum_state_summary/frontier/reconstruct` | 状态森林形状 / 边界 / 路径重建 |
| `read_csv` / `read_parquet` | 直接读文件建表 |
| `gtv_extensions()` | 已装载扩展（核心构建下为空表） |

### 3.4 双时间

- **业务时间** vs **系统时间**两轴：`as_of(table, business_ts [, system_ts])` 先选「`system_ts` 时刻已知的最新系统版本」，再保留业务区间包含 `business_ts` 的行。
- `bitemporal_overlaps(table, key_column)`：暴露互相矛盾的版本，给治理用（不是给查询用）。
- `system_ts` 是**守卫而非第二查询轴**：一个视图只覆盖一个快照，问别的快照是规划错误，而不是悄悄返回陈旧答案。

### 3.5 向量检索：三个表面，一个内核

**T5 决策（2026-11-15）：保留双表面，合并执行内核**——不 deprecate 任何一个。

| 表面 | 语义保证 | 过滤 | 硬件 | 何时用 |
|:--|:--|:--|:--|:--|
| `knn()` / `vector_search` | 精确（Flat 全扫；IVF 按注册 `nprobe` 探测） | **标签字符串** | ✅ 加速层（VRAM 常驻 / IVF probe 卸载） | 标签过滤、GPU 常驻、精确 oracle |
| `ann()` / `ann_search` | 近似 + exact rerank（可 `force_exact`） | **id allow-list** | — | Hnsw、大语料近似、要 CBO/召回估计 |
| `embedding_search` | 精确（治理集合） | tenant + `as_of` | — | 要 provenance / 租户隔离 / 过期语义 |

**为什么保留两个**：`knn` 被扩展仓（Quantum MCP）、TC-11 oracle、M18 硬件门禁依赖；`ann` 持有 Hnsw 与 CBO 规划遥测。合并表面会打断这些契约，而合并**内核**能拿到全部收益。

**执行内核边界**：三条表面的机械管线（query 解析、距离报告、`(id,distance)` schema、dim/metric 守卫、hits→batch）统一收在 `gtv-engine/src/vec_surface.rs`；CPU 搜索走 `gtv-index`（`exact_topk*` / `execute_ann`）；`knn` 的 GPU overlay 留在 engine 层。**标签过滤与 id 过滤刻意不合并**——语义不同域。

**索引与持久化**

| 索引 | 状态 | 事实 |
|:--|:--|:--|
| Flat | ✅ | AVX2+FMA 精确；`Col<f32>` 化后可 mmap |
| IVF | ✅ | `ProbePlan` + `exact_topk_positions` 宿主/装置共用同一份 cell 选择；`GIVFCMP` 复合 payload |
| HNSW | ✅ | 仅 `ann()` 表面；`knn()` 明确拒绝并指向 `ann()` |

**冷开**：`.knn-index <name> <file> [labels.txt]` / `.ann-index` / `.knn-save`。`GTVVEC`（Flat）/ `GIVFCMP`（IVF）mmap 打开，O(1) 不重解码。1e6×128 冷开 **885 ms → 0.85 ms**，常驻内存 1487.7 → 495.8 MiB。

> **关键事实**：`IvfIndex::probe_positions` 用 `mask[orig_pos[p]]`，而 `orig_pos` 随 `GIVFCMP` 持久化 ⇒ 标签域是**原始输入序**而非 cell 序，冷开无需重排。这条纠正了 V-2 曾经的过度宣称。

### 3.6 存储与元数据

| 层 | 内容 | 状态 |
|:--|:--|:--|
| **读路径 P0** | `open_segment()` 直接 mmap；header checksum O(1)，data checksum 可选全量 | ✅ |
| **写路径 P1** | WAL append-only 定宽帧（42 B/边 + blake3 帧校验）→ MemSegment 原生 Vec delta → flush 生成段 + JSON manifest 检查点 → 崩溃重放 / 历史段复用 / `graph_at(T)` | ✅ |
| **崩溃一致性** | 崩在 manifest 之前：重放全部帧（孤儿段忽略）；崩在 manifest 之后：按 `checkpoint_txn` 跳过旧帧，不重复计边 | ✅ |
| **L2 段时间盒** | `SegmentMeta` 的 `Skip / FullHit / Scan` 判定 + zone map；稀疏窗口 skip 99.0–99.7% ⇒ **103–529×** | ✅ |
| **Parquet / Snapshot** | 分区落盘 + atomic rename；`SnapshotStore` 多版本时间旅行 | ✅ |
| **LSM delta** | `DeltaBuffer` + `compact()` + `LsmStore`（不可变快照 = SSTable 层） | ⚪ 见 §3.14 |

**写路径零编码零压缩**：热层不压缩，因为 mmap 之后必须能零解码直接算。这既是性能决策，也是格式恒等式（`disk == memory == cache line`）。

### 3.7 过去更新（U0–U3）

| 项 | 交付 |
|:--|:--|
| **U0 端点策略** | `endpoint.rs`：`EndpointPolicy` 1-bit、`NodeLifetime` 定宽数组、`EndpointView::active_neighbors()`；三原语 PUBLISH / RETRACT / AS OF。热路径 1.00×，`csr.rs` 一行未改 |
| **U1 位置化更正** | `SnapshotDelta` / `RetractSet` / `RowPos` 定宽 `.gtvd`；WAL v2 `OP_MUTATION`；`live_rows_at()` 唯一 retract fold；K=1 bounded overlay |
| **U2 控制面** | **两阶段 compaction**、**延迟 GC**、观测版本落盘、L2 聚合去重、`gtv_range_agg()` / `gtv_active_edge_count()` |
| **U3 准入裁剪** | `ActiveNodeBitmap` 可选派生缓存（`NodeLifetime` 仍是事实源）、opt-in 常驻 `MaintenanceService`（只 prepare，不写盘/不改 manifest/不持 store 锁）、轻量 Tag + GC pin |

> **状态纠正**：compaction / 孤儿段 GC **不是未完成项**，U2 已交付、U3 又补了 Tag pin 与常驻 maintenance worker。三份历史文档仍停在 P1 时点，见 §9。

### 3.8 motif / 环路的 SQL 出口

`gtv-pattern` 早就有 `find` / `find_from` / `find_windowed*` / `find_bounded`，但引擎零引用——这是「核心已有、SQL 缺出口」，不是新造算法。

- 逐点：`motif(shape, t [, limit [, window]])`、`motif_from(shape, seed, t [, …])`
- 批量：`gtv_motif_matches`（一行一 match：`match_id`、`n0..nk`、`event_min/max`）、`gtv_motif_edges`（一行一 matched edge：`match_id`、`edge_pos`、`edge_idx`、`src`、`dst`、`edge_type`、有效区间）
- `MotifSpec` 另支持滚动时间窗 `with_window()` 与按事件时间排序的类型序列 `with_sequence()`（**仅 Rust API**，SQL 只暴露 `window`）

**边界**：`find_bounded`（显式 `TraversalBudget` / `CancelToken`）仍未接 SQL，当前上界只由 `limit` 提供。

### 3.9 状态来源图

- `gtv-core/src/provenance.rs`：来源森林，`O(depth)` 路径回溯。
- `gtv-core/src/quantum_state.rs` + `gtv-engine/src/quantum_state.rs`：`state_id / parent_state / gate / timestamp / probability` 五列，三个 SQL 出口（summary / frontier / reconstruct）。
- **诚实边界（必须写进任何对外材料）**：存的是**分支结构与概率**，不是振幅；**不能**从这个 log 算干涉项或重建完整 statevector。要振幅只能存 blob 引用。把这条写清楚比夸大更能建立信任。

### 3.10 装置加速层

`gtv-accelerator`，`--features cuda` 才编译 CUDA 后端，否则纯主机路径（GPU-less CI 走的就是这条）。

| 组件 | 事实 |
|:--|:--|
| `VramBudget` | 弹性显存账本：**只缩不胀**；容量不足时降级而不是 OOM |
| `ResidentRegistry` | 衰减热度 + 滞回 + **影子账本**（`MAX_SHADOW=256`、`ADMIT_MARGIN=1.25`、`PROTECT_FLOOR=4.0`、`ACCEL_HEAT_HALF_LIFE`）；`ACCEL_CACHE=SHADOW` 可在无 GPU 机器上标定命中率 |
| 算子 | 精确 KNN（连续语料 + 有界 `select_nth_unstable`）、as-of join（`O(L+R)` 单调归并）、IVF probe 卸载（融合 top-K kernel + 可选位置间接层） |
| 主机回退 | 任何一步失败都回落主机并记账；**启动永不依赖 GPU** |
| NVRTC | 运行时编译 CUDA（无预编译 PTX 路线） |
| 七条硬件门禁 | `cuda_acceptance.rs` 1–7，RTX 4060 Laptop / CUDA 13.1 / compute_89 全绿，CI `cuda-acceptance.yml` |
| 遥测 | `.accel` 输出预算、常驻块、命中率、回退、`PREFETCH`、`GPUDIRECT` |

**三级分层（P1a–P2）**

| 步骤 | 内容 | 状态 |
|:--|:--|:--|
| P1a | 量化 PCIe DMA 墙 | ✅ pinned 拿回 33–45% 稳态带宽；1 MiB 小载荷 4.37× |
| P1b | pinned 上传 | ✅ `cudaHostAlloc` write-combined staging；`cudaHostRegister` 封装（V-3，Windows WDDM 拒绝文件页 ⇒ 零拷贝仅 Linux/TCC） |
| P1c | 异步预取流水线 | ✅ 块双缓冲 `copy(b+1) ∥ transpose(b) ∥ topk(b)`（8 MiB 块）+ 页面预热 `warm.rs` + 窗口转置 kernel |
| P2 | heat 驱动空闲预热 + GPUDirect 接缝 | ✅ 见下 |

**heat pump 的设计是被实测改过的**：它读影子账本里「热但未驻留」的块，在 `knn()` 收尾跑一轮。第一版允许淘汰，基准直接否掉——见 §4 D4。现在它是 **slack-only**：只填空闲 VRAM、绝不淘汰、只消费热度不产出热度，每轮起手先 reclaim 掉已被 DROP 的集合的显存。

**GPUDirect** 是**策略接缝**而非实现：`gpudirect.rs` 无 FFI 无 `unsafe`，`probe()` 在 Linux 上查 `/dev/nvidia-fs*`，但 `DIRECT_TRANSFER_COMPILED = false` ⇒ 任何构建都**不能**宣称直传。`note_upload_path()` 把每次语料上传记成 direct/staged，`.accel` 显示 `GPUDIRECT = A direct / B staged`。

### 3.11 部署参数档

Oracle `init.ora` 风格扁平 `KEY=VALUE`（大写键、`#`/`--` 注释、空行忽略）。支持二进制后缀 `K/M/G/T` 与分数引用 `SYSTEM:0.5` / `TARGET:0.5` / `DEVICE:0.8`。

| 键 | 默认 | 作用 |
|:--|:--|:--|
| `MEMORY_TARGET` | `SYSTEM:0.5` | DataFusion 内存池上限（**仅显式设置才强制**） |
| `MEMORY_OBJ_AREA` | `TARGET:0.5` | 对象区预算 |
| `CUDA_MODE` | `AUTO` | `AUTO` 探到设备才开；`OFF` 一票否决所有加速项 |
| `CUDA_VRAM` / `CUDA_DEVICE` | `DEVICE:0.8` / `0` | 显存预算与设备序号 |
| `ACCEL_CACHE` | `ON` | `OFF` / `ON` / `SHADOW`（无 GPU 标定） |
| `ACCEL_SYNC_INTERVAL` / `ACCEL_SYNC_THRESHOLD` | `5` / — | 常驻同步节流 |
| `ACCEL_HEAT_HALF_LIFE` | `30` | 热度半衰期 |
| `ACCEL_PREFETCH` / `ACCEL_PREFETCH_BLOCKS` | `OFF` / `1` | heat pump 开关与每轮预算 |
| `ACCEL_GPUDIRECT` / `ACCEL_GPUDIRECT_THRESHOLD` | `OFF` / `1G` | 直传开关与触发阈值；被 `CUDA_MODE=OFF` 否决 |
| `DEFAULT_MODE` / `LOG_LEVEL` / `TELEMETRY` | `SQL` / `INFO` / `FALSE` | 运行形态 |
| `WORKER_THREADS` / `SPILL_DIR` | 逻辑核数 / 临时目录 | 执行并行度与落盘 |

`GTV_PARAM_<KEY>` 环境覆写；`GTV_PARAMS_FILE` 发现；`GtvContext::with_params` 挂载；`.params` 打印生效值。

### 3.12 扩展 ABI

```rust
impl GtvExtension for MyExtension {
    fn name(&self) -> &'static str { "my_workload" }
    fn version(&self) -> &'static str { "0.1.0" }
    fn register(&self, ctx: &GtvContext) -> Result<()> { /* 注册 table function / UDF */ }
}
GtvContext::with_extensions(ExtensionRegistry::new().with(MyExtension))?
```

- **静态注册**，没有 `CREATE EXTENSION`；安装顺序即优先级顺序，扩展可以刻意遮蔽核心函数。
- 扩展拿到的是**同一个** `GtvContext`（同一批 KNN collection、资源注册表、段注册表），可随时卸载不影响内核。
- `gtv_extensions()` 返回 `(name, version, objects)`；核心构建下为空表。

### 3.13 治理与可观测

| 面 | 内容 |
|:--|:--|
| 目录 / lineage | `gtv-catalog`：稳定 table/snapshot/file 身份、版本化 schema 注册表、分区规格、原子提交协议；`execute_with_lineage` 记录源快照、模型/索引版本与输出 blake3，`replay` 可钉住快照重放 |
| DQ | 声明式 `DqRule` → `GateDecision`；`publish_guarded` 让失败规则直接挡住发布 |
| 工作负载 | 单进程 best-effort：类/资源组/准入/协作式抢占（复用 `CancelToken`）；高优类可抢占低优类，预算满则显式排队或拒绝 |
| CBO | 关系 `TableStats` + 图 `GraphStats` + 向量 `VectorStats` 三族合成一个 `QueryPlanChoice`（filter-first vs ANN-first、索引建议、图源剪枝、exact rerank） |
| 监控 | 进程计数器 + SQL 延迟直方图；`health_check` / `strategy_stats` |

### 3.14 库级已交付、但未接产品表面（⚪）

这四个 crate **没有任何 workspace 内依赖者**（`grep 'path = "../gtv-delta"' crates/*/Cargo.toml` 为空），也没有 SQL/CLI 出口。它们是能力储备，不是产品功能：

| crate | 已实现 | 缺口 |
|:--|:--|:--|
| `gtv-delta` | `DeltaBuffer`（memtable）+ `compact()` + `LsmStore` | 引擎未使用；当前更新语义走 U1 的 `.gtvd` delta + U2 compaction，两套并存 |
| `gtv-ingest` | 源无关 at-least-once **微批**摄入、内容定址信封、dedup、DLQ、watermark、offset | 无调度器、无 CLI 入口 |
| `gtv-udf` | WASM（wasmtime）沙盒标量 UDF，`f64 -> f64` | 未注册进 SQL；仅标量 MVP |
| `gtv-proto` | `gtvquery` gRPC 服务 + Arrow IPC（Flight wire）序列化 | 无 server 二进制；远程序列化未接 CLI |

**这条要写进产品材料**：对外说「有 14 个 crate」会把储备算成能力。真实交付面是 §3.2–3.13 那些有入口的东西。

---

## 4. 关键设计决策

> 格式：**决策 / 被否掉的替代 / 证据 / 代价**。每一条都是「实测或硬约束逼出来的」，不是审美选择。

**D1 边的有效区间进索引，而不是进属性**
替代：把 `valid_from/to` 当普通列 + 建 B-tree。否决理由：`as_of T` 会退化成全表扫描；区间列的 B-tree 对「包含」谓词无效。
证据：zone map + run 内二分让 `neighbors()` 包围盒命中 1.00×（与无时间维度等价）。
代价：`valid_to` 非单调 ⇒ 只能存 `max valid_to` 做剪枝，剪枝率不如单调列。

**D2 热层零编码零压缩、段不可变**
替代：LSM + 块压缩（OceanBase / Parquet 路线）。否决理由：mmap 之后要能**零解码直接算**；压缩会破坏 `disk == memory == cache line` 恒等式。
证据：[`oceanbase-storage-lessons.md`](oceanbase-storage-lessons.md)；GTVSeg 定宽 64 B 对齐 LE。
代价：磁盘占用大；变长属性必须走 sidecar（U3 判定 GTVSeg 列扩展 **NO-GO**）。

**D3 准入用衰减热度 + 弹性预算，不用 LRU**
替代：LRU / LFU。否决理由：LRU 在「一批种子反复打同一集合」时会把刚上传的块挤掉；LFU 不衰减会让历史热点永久占位。
证据：`ADMIT_MARGIN=1.25` + `PROTECT_FLOOR=4.0` + 影子账本让被拒的块也积累证据；`ACCEL_CACHE=SHADOW` 可在无 GPU 机器标定命中率。
代价：参数多（半衰期、margin、floor），调参靠遥测；预算是「只缩不胀」，容量预期要保守。

**D4 heat pump 只填空闲、绝不淘汰**
替代：允许 pump 淘汰。**这是被基准当场否掉的设计**：3×99 MiB 语料 / 198 MiB 缓存下，淘汰式 pump 把查询刚换上来的块再搬一遍——上传字节 1388 → 2380 MiB（**+71%**），device 命中 20 → 14，host 回退 4 → 10；bursty 形态 p50 0.93× / p99 0.97×，上传 495 → 2182 MiB。
现行：`admit_prefetch` 走 slack-only，无空位直接 `DeclineReason::NoSlack`；pump 命中**不加热度**（否则会反超查询块）。空位唯一来源是「集合被 DROP / 重注册变小」，所以每轮起手先 `reclaim_dropped`。
证据：reclaim 场景 device 19 → 20、p99 26.9 → 21.3 ms，**传输字节完全相同 297 MiB**。
代价：收益上界很窄——只有在「空位出现后仍有查询打到该集合」时才兑现；换掉的是「可证明不回退」。

**D5 向量保留 `knn` / `ann` 双表面，合并执行内核**
替代：收敛到单一表面。否决理由：`knn` 被扩展仓 / oracle / 硬件门禁依赖，`ann` 持有 Hnsw 与 CBO 遥测；合并表面会打断外部契约。
证据：管线统一到 `vec_surface.rs`，三条表面零漂移；标签过滤与 id 过滤语义不同域，刻意不合并。
代价：用户要理解「什么时候用哪个」，靠 §3.5 的分工表补。

**D6 只争「图 + 时间」的端到端延迟**
替代：和 kdb+ 比单表向量化扫描带宽。否决理由：那是对手最强项，赢不了也不需要赢；gtvdb 的价值在「kdb 需要多段 q 脚本」的那类查询。
代价：不能拿 TPC-H 式数字做营销；对外只能报「同机 A/B + 可复现脚本」。

**D7 扩展静态注册、工作负载出仓**
替代：动态加载 dylib / `CREATE EXTENSION` / 核心内置领域功能。
证据：`cargo tree -p gtv-engine` 拉不进任何领域 crate，CI `core-boundary` 三重守卫。
代价：加扩展要重新编译；换来的是「核心边界不会腐化」——本仓最贵的资产。

**D8 GPUDirect 先交策略接缝，不交实现**
理由：没有 Linux + `nvidia-fs` + `cufile` 真机就写 FFI，等于给一个无法验证的路径发合格证。
证据：`probe()` 恒报 `Unavailable("built without cufile bindings")`；`DIRECT_TRANSFER_COMPILED=false` 是编译期常量，**任何构建都无法虚报直传**。
代价：功能上是零；价值是把「以后接进来时不用改调用点」和「现在不会骗人」同时拿到。

**D9 点函数只接受字面量，批量走 TableProvider**
理由：`neighbors(t)` 之类的标量 UDTF 每行重规划，N 个种子就是 N 次 DataFusion parse+plan+collect，而内核每边只要纳秒。
证据：N=32–128 时 Windows 5.7–15.8×（owned）/ 27–55×（shared），Linux 16–55× / 28–78×。
代价：两套入口要各自维护对等性——用「逐条 parity 测试」守着，不是靠约定。

---

## 5. 性能现状

> **口径**：全部为**同机 A/B**，只用于判断「这版比上版好/坏」。**不得当作对外性能声明**（`bench_m6_gate` 的绝对毫秒数尤其如此）。测量机：Windows / RTX 4060 Laptop / CUDA 13.1 / compute_89。

### 5.1 目标查询 Q1/Q2/Q3

| 指标 | 口径 | 现状 |
|:--|:--|:--|
| **Q1** 批量 k-hop | N 个种子、2–3 hop、指定 `valid_at`，1e7 边级图 p50/p99 | ✅ 出口在 `bench_m6_gate` A 段；批量算子已交付（见上）。**规模仍在合成图 V=20k/E=200k，未推到 1e6/1e7** |
| **Q2** 时间窗聚合 | `[T1,T2)` 活跃边/敞口聚合 p99，含 `FullHit` 短路比例 | 🟡 `gtv_range_agg` / `gtv_active_edge_count` 已交付；**B 段目前只测段粒度 B/edge**，真实窗口 p99 与段级 `FullHit` 零扫描路径都还欠 |
| **Q3** 摄入 | 单进程 M edges/s、publish→可见 p99 | ✅ C 段有基线；WAL fsync 成本未拆分，跨平台 fsync 差一个数量级（M6-a 复现 4–6×） |

### 5.2 加速层与分层

| 场景 | 数字 |
|:--|:--|
| 精确 KNN（M17，纯 CPU 路径） | **5.85×** |
| as-of join（M17，纯 CPU 路径） | **5.28×** |
| PCIe pinned vs unpinned（P1a） | 稳态（≥8 MiB）**33–45%** 带宽；1 MiB 小载荷 **4.37×**（0.69 → 3.03 GB/s）；本机墙 ≈ **12 GB/s** |
| 冷首查 195 MiB / k=10（P1c） | bulk 94.6 ms → 8 MiB 分块 **46.9 ms（2.0×）**；块扫描 64/16/8/4/1 MiB = 72.2/49.9/**46.9**/47.0/57.7 ms（拐点 8 MiB = `DEFAULT_PREFETCH_BLOCK_BYTES`） |
| 稳态 cache-hit（P1c） | **1.5 ms，无回归** |
| heat pump（P2，3×99 MiB / 198 MiB 缓存 / 50 ms 间隔） | off：device 19 / host 6，p99 26.9 ms，上传 297 MiB<br>on：device 20 / host 5，p99 21.3 ms，上传 **297 MiB（零额外搬运）**，1 upload / 4 refused |
| flat batch KNN（C2） | batch 1 **3.5×**、batch 32 **14.3×**，盈亏点 **5 次**；阶段耗时 kernel 99.1% / upload 0.2% / download 0.5% |
| rolling window（C3，**反例**） | kernel 比 CPU 快 **18×**（1.13 vs 20.6 ms @1e7），端到端只 **1.03×**（冷 0.68×）⇒ 默认关闭 |

### 5.3 冷开、检索与图论

| 场景 | 数字 |
|:--|:--|
| 1e6×128 向量冷开（V-4） | **885 ms → 0.85 ms（1040×）**；常驻 1487.7 → **495.8 MiB（3.0×）** |
| IVF 1e5×128 冷开 | 75 ms → **1.1 ms** |
| motif SQL 出口（P1-4，`ring3`，V=20k） | seeds 1/8/32/128 → owned **0.42× / 2.59× / 11.82× / 49.79×**；shared **1.23× / 7.90× / 30.49× / 111.55×** |
| 批量图算子（P1-1） | N=32–128：Windows 5.7–15.8×（owned）/ 27–55×（shared）；Linux 16–55× / 28–78× |
| `graph_at(T)` 列式化（C4 前置） | 1e7 边 **1579 ms → 393 ms**（3 段归并）/ **176 ms**（压实后单段） |
| L2 稀疏窗口 skip | skip 99.0–99.7% ⇒ **103–529×**，冷启动与段数无关 |

### 5.4 两条读法

1. **瓶颈几乎从不在图内核**。P1-1 与 P1-4 同结论：慢在 N 次 SQL 规划；C2 的 kernel 已占 99.1%，传输不是瓶颈。
2. **输出与输入同大小的算子不划算**。C3 的 18× kernel 被 `8n` 往返和 80 MB 新页缺页（占下行 53%）吃干净——这类算子只能做**链内**，不能独立上表面。

---

## 6. 路线图：准备开发的功能

> 来源：[`ROADMAP.md`](../ROADMAP.md) §3.1 / §3.2 / §4。**每个进主干的项必须能回答「它让 Q1/Q2/Q3 哪一条的 p99 降了多少，基准在哪个文件」**；答不上来的进 §6.4 冻结/下架。

### 6.1 已排期 / 条件已明确

| 项 | 内容 | 状态 | 触发与验收 |
|:--|:--|:--|:--|
| **GPUDirect 真机** | 编入 `cufile`，`probe()` 返回 `Available`，`upload_plan` 走 `Direct` | ⏳ 接缝已交付，等硬件 | 需 Linux + NVMe + `nvidia-fs`；验收 = 大表扫描省一次 RAM 往返（≥ threshold 1 GiB 路径），且 `.accel` 计数与实际传输路径一致 |
| **P0 测量补齐** | ① Linux `perf stat`（`cache-misses` / `dTLB` / `branch-misses`）② 规模推到 **1e6 节点 / 1e7 边** ③ Q2 真实时间窗 p99 + `FullHit` 短路比例 ④ 并发查询 p99 ⑤ WAL fsync 成本拆分 ⑥ 同机列式基线（仅趋势） | ⏳ 不新增算子 | 数据只在内存/tempdir 构造，落盘必须放仓库外；产出是**现状表**，不是优化 |
| **C4 CUDA 链式计划** | 把 rolling window 这类「18× kernel、1× 端到端」算子放进链内；拓扑投影只上 5 B/edge；超出显存后的 slab / 预取 / spill guard | ⏳ | 端到端必须 >1 才有资格默认开；`count/min/max` 逐位相等、`sum/mean` 容差 1e-6 的约束不变 |
| **T5.3 过滤互备** | 标签过滤 ↔ id allow-list 互相可达 | ⏳ 未排期 | 语义不同域，属「另立任务」；不做也没缺口，做了只是少一次换算 |

### 6.2 冻结（有触发条件，未触发不做）

| 项 | 触发条件 | 为什么现在不做 |
|:--|:--|:--|
| **P1-2 段内窄化 / `(time_bucket, src_range)` 分片** | 持续段数进入 **100–500** 且 **B/edge > 256** | 实测 1/10/100/500 段 → 0.8/8/80/**400** B/edge；目标 1,439 → ~145 B/edge。验收须复用 `l2_time_box.rs` 断言，不得破坏 `Skip/FullHit/Scan` |
| **P1-3 热循环 measured-first 调优** | 先拿到 Linux `perf stat`，**只做被证明是瓶颈的一项**（16 B 窄化 / SIMD 时间谓词 / Eytzinger / 大页 / 并行 frontier） | 每项都要 A/B，输了回滚；没有 profile 就动手等于赌 |
| **并发写入 / WAL 批量 fsync / 真实 `kill -9` 测试** | Q3 在 Linux 上证明单写者 `&mut self` 或每帧 fsync 是瓶颈 | 现在补的是可靠性不是速度；先测 |
| **L1 `stats.idx`、段级 `FullHit` 零扫描聚合、冷热淘汰、`L2` Parquet 归档** | 规模超过内存 / 运维面真的需要 | 都不在目标查询热路径 |

### 6.3 待决策（本次分析识别出的产品缺口，尚未排期）

| 缺口 | 现状 | 要决策什么 |
|:--|:--|:--|
| **4 个 ⚪ crate 要不要接产品面** | `gtv-delta` / `gtv-ingest` / `gtv-udf` / `gtv-proto` 零依赖者 | 接哪一个、以什么入口接（`.ingest` 指令？gRPC server？WASM UDF 注册）、还是明确标为储备并移出叙事 |
| **两套 delta 语义并存** | U1 的 `.gtvd` + U2 compaction 是实际路径；`gtv-delta` 的 LSM 是另一套 | 二选一或划清边界，否则未来加更新功能会踩错层 |
| **repo 门面与 ROADMAP 判定冲突** | `positioning.md` §8 的 90 天计划含 crates.io 发布 + pyo3，ROADMAP §4 把 M8/M9 判为 🗑 下架 | 以 ROADMAP 为准即可，但两份文档要显式对齐（见 §9） |

### 6.4 冻结 / 下架清单（不占里程碑）

| 项 | 处理 |
|:--|:--|
| Parent+Diff Manifest、Branch、day-bucket 缓存、GTVSeg 列扩展 / sidecar、自动 GC 默认开 | 🧊 冻结（Parent+Diff 触发：>5000 段且 serialize >2 ms/发布） |
| M7 增量 frontier | 🧊 冻结（与 AML/CRM/LE 无关） |
| M8 crates.io、M9 pyo3、M10 canonical demo、M12 dylib loader | 🗑 主线下架（出现真实嵌入用户再按需） |
| 任何新的非图算子 UDTF / 领域功能 | ⛔ 禁止（领域一律进扩展仓） |
| PTX 预编译路线（C1–C3 旧实现） | 🗑 已被 NVRTC 单 crate 路线整体取代并删除 |

---

## 7. 已知限制与诚实边界

**这张表比功能清单更重要。** 每一条都写明「影响什么」和「什么时候会修」。

| 限制 | 影响 | 现状 / 触发 |
|:--|:--|:--|
| 量子状态图**不存振幅** | 不能算干涉项、不能重建 statevector | 设计如此；要振幅只能存 blob 引用。**对外必须写明** |
| GPUDirect 未实测 | 大表扫描省不下那一次 RAM 往返 | 策略接缝已交付，等 Linux + `nvidia-fs` |
| Windows WDDM 拒绝 `cudaHostRegister` 文件页 | 零拷贝上传仅 Linux/TCC 可达 | 自动回落 P1b-1 staging（write-combined），功能不受影响 |
| 单写者（`&mut self`）、无分布式 | 不能多进程并发写；无 shard/planner/worker | §6.2 冻结，等 Q3 证明是瓶颈 |
| `knn()` 不支持 Hnsw | 大语料近似要走 `ann()` | 明确报错并指向 `ann()` |
| `find_bounded` 未接 SQL | 遍历预算只能靠 `limit` 粗控 | §6.1 未排期 |
| **无 Python 原生绑定** | Python 只能当客户端驱动 REPL，继承 REPL 边界（图遍历 / 建索引 / catalog 不可达） | M9 已下架；`examples/python/gtv_repl.py` 是过渡答案，见 [`user-guide.md`](user-guide.md) §12 |
| `sequence` 只在 Rust `MotifSpec` | SQL 无法表达「按事件时间排序的类型序列」 | 需要 SQL 数组字面量，未做 |
| Q2 段级 `FullHit` 零扫描 | 大窗口聚合仍走折叠视图 | §6.2 冻结（先测规模） |
| heat pump 收益窄 | 只在「有空位 + 后续仍有查询」时兑现 | 设计如此（D4）；换来零回退 |
| 热层不压缩 | 磁盘占用大 | 设计如此（D2） |
| 4 个 ⚪ crate 无入口 | 对外不能当能力算 | §6.3 待决策 |
| 段内每边 `system_ts` 不单存 | `graph_at(T)` 时间粒度 = flush 快照粒度 | 设计如此（段级锚点） |

---

## 8. 质量与验收基线

| 项 | 事实 |
|:--|:--|
| 回归 | `cargo test --workspace`：**681 passed / 66 suites / 0 failed** |
| 告警 | `cargo check --workspace --tests` 与 `-p gtv-accelerator --features cuda --tests` **均零告警** |
| CI 主干（`ci.yml`） | `generated-data`（不提交生成数据）· `core-boundary`（目录树 + manifest + `cargo tree` 三重）· `workspace-build` · `kernel-tests`（**17 个分组**，含 5 道门禁）· `full-tests`（含 doctests） |
| CI 硬件（`cuda-acceptance.yml`） | 7 条验收门禁 + `cuda_prefetch` 4 项 + `cuda_heat_pump`；门禁在 `static Mutex` 上串行（`gate5` 量的是进程级空闲 VRAM，并行会把邻居的分配当泄漏） |
| 基准门禁 | `bench_m6_gate`（Q1/Q2/Q3 现状表，每段带硬断言）· `bench_motif_gate`（N≥32 必须 >1）· `bench_m3_gate`（准入判定）· `bench_correction`（`Overlay == Materialized`） |
| 正确性铁律 | 每条加速路径都必须与主机路径**逐条一致**（batch vs pointwise、多块 vs 单块、GPU vs host oracle）；`Overlay == Materialized` 是更新路径的等价口径 |
| 数据纪律 | `*.csv` / `*.seg` / `*.parquet` / `*.pdf` 不入库；`testcase/data/gen_datasets.sh` 依 `SHA256SUMS` 重建，`.gitattributes` 钉死 LF |

---

## 9. 本次分析发现的事实偏差（建议后续修正）

分析代码与文档时发现 6 处不一致。**它们不影响代码事实**，但会让产品材料说错话，建议在下次文档同步时一并修掉：

| # | 位置 | 偏差 | 建议 |
|:--|:--|:--|:--|
| 1 | `README.md` §1 | 写「12 个内核 lib crate」，实际 **14 个**（M16 加 `gtv-params`、M17 加 `gtv-accelerator`） | 改成 14，或改成「14 个 lib + 1 个 bin」 |
| 2 | `docs/params-file.md` | **完全没有** `ACCEL_PREFETCH` / `ACCEL_PREFETCH_BLOCKS` / `ACCEL_GPUDIRECT` / `ACCEL_GPUDIRECT_THRESHOLD` | 补一节；`gtvdb.params.example` 已是准确的 |
| 3 | `docs/metadata-benchmark.md` §6.2 第 4 项 | 仍把 **compaction** 列为「下一步」，U2 已交付 | 同步状态，保留历史数值 |
| 4 | `docs/storage-mmap-design.md` §9.5 / §11 | 仍称「`compaction()` 尚未实现」「孤儿段 GC 未做」 | 标注 U2/U3 已交付 |
| 5 | `docs/oceanbase-storage-lessons.md` §6 与脚注 17 | 引用 `crates/gtv-ingest/src/dedup.rs:7` 的 TODO 注释——该行现为**说明性注释**，不是 TODO | 更新引用；「全库无 bloom filter」结论仍成立 |
| 6 | `crates/gtv-engine/src/monitor.rs` 头注释 | 提到「REPL `metrics` 命令」，当前 `gtv-cli` **无此指令** | 删掉或补实现 |

另有一处**内容层面**的冲突需要业主裁定（不是笔误）：

- `docs/positioning.md` §8 的 90 天计划包含 **crates.io 发布**与 **pyo3 绑定**；`ROADMAP.md` §4 把 M8（crates.io）/ M9（pyo3）判为 **🗑 主线下架**。本文 §6 以 ROADMAP 为准，但两份文档同时存在于仓库里，对外口径会自相矛盾。

---

## 10. 产品视角的文档地图

| 想了解 | 看这个 |
|:--|:--|
| **怎么用起来**（命令、示例、报错） | [`user-guide.md`](user-guide.md) |
| 产品有什么能力、下一步做什么 | **本文** |
| 架构、存储格式、正确性策略、评审问题 | [`design-review-2026-11.md`](design-review-2026-11.md) |
| 里程碑编号与判据、实测数字 | [`ROADMAP.md`](../ROADMAP.md) |
| 为什么这样定位、竞品对位、对外文案 | [`positioning.md`](positioning.md) |
| GTVSeg 格式与读写路径 | [`storage-mmap-design.md`](storage-mmap-design.md) |
| 三级分层（SSD→RAM→VRAM） | [`storage-tiering-ssd-ram-vram.md`](storage-tiering-ssd-ram-vram.md) |
| 装置加速层与 7 条门禁 | [`acceleration.md`](acceleration.md) |
| 向量语料上段格式 | [`vector-corpus-gtvseg.md`](vector-corpus-gtvseg.md) |
| 参数档 | [`params-file.md`](params-file.md)、`gtvdb.params.example` |
| 过去更新（U0–U3） | [`minimal-temporal-update-impl.md`](minimal-temporal-update-impl.md)、[`update-segment.md`](update-segment.md) |
| 量子状态图 | [`quantum-state-graph.md`](quantum-state-graph.md) |
| AML 就绪与核心原语 | [`aml-core-ext-enablement.md`](aml-core-ext-enablement.md) |
| 性能优先的路线评审 | [`perf-first-roadmap-review.md`](perf-first-roadmap-review.md) |
