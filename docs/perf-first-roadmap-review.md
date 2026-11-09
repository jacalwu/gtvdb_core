# 剩余开发项核查与线路修正建议（perf-first）

> 2026-11-09｜对象：`gtvdb_core` @ `7e328af`（M3 落地后，487 passed / 0 failed / 1 ignored，CI 5/5）。
> 目标（用户给定）：**保持高性能，做速度上可媲美 kdb 的时序图数据库，主用于金融 AML / CRM / LE 这类「可用图快速求解」的 use case；不把各种功能硬塞进去。**
> 本文只做核查与建议，不修改 `ROADMAP.md` 或代码；确认后再落路线图与 CI 门禁。

---

## 0. 结论先行

**需要修正，而且修正的方向是「减项」，不是「加项」。**

1. 现有剩余项清单共 **约 20 条**，但其中真正影响「目标查询延迟」的只有 **3 条**：批量图算子、段内节点区间窄化、热循环 measured-first 调优。其余都是控制面/分发/绑定/演示，或已被写死触发条件的 defer 项。
2. 过去 5 个里程碑（L1/L2 元数据、M0、M1、M2、M3）几乎全部投在**控制面与历史可更正性**（catalog、时间盒、端点策略、位置化 retract、WAL v2、delta、compaction、GC、Tag、常驻 worker、bitmap）。它们把「能不能改历史」做对了，但没有一条给出「目标 use case 的 p99 降了多少」。
3. **「媲美 kdb」目前不可证伪**：全仓没有任何以 AML / CRM / LE 命名的端到端基准，没有摄入吞吐基准，没有并发查询基准，没有 `perf stat` 采样，也没有 kdb 或等价列式基线对照。在这种状态下继续做功能，无法判断是变快还是变慢。
4. 一处结构性缺口比任何待办都关键：**核心层能批量，SQL 层把批量掐断了**。`neighbors(src,t)` / `khop(...)` 只接受字面量（`expr_util.rs:21,36,51,66` 一律 `arguments must be literals`），而 AML 批次告警、CRM 客户集、LE 债务人集全都是「一批种子 → 图求解」。现在的实现等于对每条种子付一次 DataFusion 规划 + 一次 RecordBatch 分配（毫秒级），而内核热循环是 ns/edge 级。**这是离目标最远、也最容易补的一厘米。**
5. 一份实测已经指出 10–20× 的性能债，却一直停留在「下一步」：`metadata-benchmark.md` §5⑤，50,303 条边按月分桶 → 586 段 / 69.02 MB / 1,439 B per edge / flush 9,975 ms / 全扫 535 ms；按年 → 49 段 / 6.96 MB / 145 B per edge / 857 ms / 26 ms。根因是稠密 CSR 每段都要带 `node_count` 个 `offsets`（O(V)），与边数无关。解法（§6.2，`node_id_min/max` 已存 manifest）已备好但没有排进主线。

**建议把路线改成：先建立可证伪的速度口径（P0）→ 只做 3 项被证明影响延迟的工程改动（P1）→ 其余全部冻结或降级（P2）。**

---

## 1. 口径：什么叫「媲美 kdb」

先把目标写成可测量、且不自我伤害的形式。

- **不要争纯列式算子的绝对带宽。** kdb+/q 的强项是单表向量化扫描与 asof join；本仓 `bench_fused` 的 1M asof 2.5 ms 是阵列层结果，不是图查询。用对手最强的算子定义胜负，会把自己拖进一场不需要打的仗。
- **要争的是「图 + 时间」这一类查询的端到端延迟**：kdb 需要多次 join / 多次 q 脚本才能表达的「T 时刻的 k-hop 资金链、担保链、敞口传导」，gtvdb 应在单进程内一次求解。指标应为**同量级延迟与吞吐**，而不是「同一算子更快」。
- **「LE」按本仓语境解释为 Large Exposure（MA(BS)28 大额风险敞口）**；若用户指的是 Law Enforcement 的关联/资金链路分析，结论不变——两者都是「一批种子 + 时序 k-hop + 聚合」，共用同一套图算子。

因此把目标拆成三条可写进 CI 的硬指标（**阈值必须先测现状再定，不拍脑袋**）：

| 指标 | 口径 |
|:--|:--|
| Q1 批量 k-hop | N 个种子、2–3 hop、指定 `valid_at`、1e7 边级图上的 p50/p99 |
| Q2 时间窗聚合 | `[T1,T2)` 上的活跃边 / 敞口聚合 p99，含 `FullHit` 短路比例 |
| Q3 摄入 | 单进程 M edges/s，以及 publish→可见延迟 p99 |

每条都必须带**正确性断言**（对照全量扫描），沿用 `bench_metadata.rs` 与 `bench_correction.rs` 已有的做法；否则「剪得干净但漏数据」会被基准奖励。

---

## 2. 剩余项全量对账

### 2.1 四个来源

| 来源 | 列出的未完成项 |
|:--|:--|
| `ROADMAP.md` | **M6 P2–P3**（16B 窄化 / 分层 Skip Index / SIMD 谓词下推 / compaction / 孤儿段 GC / 冷熱淘汰 / `.gtvidx` mmap / L2 Parquet 归档 / 并发写入 / kill -9）、**M7** 增量 frontier、**M8** crates.io 发布、**M9** pyo3 绑定、**M10** canonical demo、**M12** dylib 加载器 |
| `minimal-temporal-update-impl.md` §14.8 | Parent+Diff Manifest、Branch、day-bucket 缓存、GTVSeg 列扩展 / sidecar、自动 GC 默认开启 —— **每项都有写死的触发条件** |
| `metadata-benchmark.md` §6.2 | L1 每表 `stats.idx`、段内节点区间窄化、`[T1,T2)` 原生聚合、compaction |
| `storage-mmap-design.md` §9.5 / `oceanbase-storage-lessons.md` §6 | 并发写入、WAL 批量 fsync、真实 kill -9 进程级测试、冷热淘汰、L2 Parquet 归档、`perf stat` 采样、压缩 vs 不压缩 A/B |

### 2.2 状态漂移（这是必须先修的文档错误）

- `ROADMAP.md` 仍把 **compaction / 孤儿段 GC** 列为未完成，但 M2 已交付两阶段 compaction 与延迟 GC（`docs/minimal-temporal-update-impl.md` §13）、M3 又补了 Tag 对 GC 的 pin 与常驻 maintenance worker。
- `ROADMAP.md` **完全没有记录 L1/L2 元数据、M0/M1/M2/M3 这 5 个已交付里程碑**，主线编号停在 M14（拆仓）。
- 结果：读路线图的人会以为存储层还停在 P1（2026-11-08），实际 HEAD 已到 M3（2026-11-09）。**路线图本身失去指导能力，比任何单项未做都更需要先修。**

### 2.3 按「是否影响目标查询延迟」重新分类

| 类别 | 项 | 判定 |
|:--|:--|:--|
| **直接影响** | 批量图算子（新增，不在原清单里）、段内节点区间窄化、热循环 SIMD/窄化/预取（measured-first） | ✅ 进 P1 |
| **间接/条件** | 并发写入 + WAL 批量 fsync + kill -9、`stats.idx`、`[T1,T2)` 原生聚合、冷热淘汰 / L2 Parquet 归档 | ⏸ 先测，触发才做 |
| **纯正确性/运维，已在 M2/M3 完成** | compaction、GC、Tag、invalidation、observation、aggregate | ✅ 标记完成，不再投入 |
| **纯分发/包装** | M8 crates.io、M9 pyo3、M10 demo、M12 dylib loader | 🗑 降级为「有真实嵌入用户再做」，不占里程碑 |
| **与目标无关** | M7 增量 frontier（服务 quantum state 分支集） | 🗑 冻结 |
| **已写死触发条件** | Parent+Diff（>5000 段且 serialize >2 ms）、Branch、day-bucket、sidecar、自动 GC 默认开 | 🧊 冻结，不主动重启 |

---

## 3. 证据：线路为什么偏了

### 3.1 投入结构

M0→M3 的测试数从 428 → 448 → 466 → **487**，新增的 59 个用例绝大多数在控制面（`m1_correction` / `m2_control_plane` / `m3_admission` / `m3_tags_sql`）。读路径的改动只有：P0 `Col<T>` mmap、L2 段时间盒、M0 端点包围盒，且 M0–M3 明确以「`csr.rs` 与 GTVSeg v1 列格式一行未改」为荣。这条纪律在**保证正确性**上完全正确，但也意味着：**连续 4 个里程碑没有动过决定速度的那一行代码。**

### 3.2 L2 的收益边界

L2 很有效，但它的收益来自**跳过不看的数据**：`trips_x100` 1×30d 窗口 skip 99.0% → 103×、`comments_point` skip 99.7% → 529×、冷开 ~2.8 ms 与段数无关、剪枝决策 1–3 ns/段。可一旦目标查询落在热门时间窗（`full` 窗口），加速就回到 1.01–1.06×；静态图对照组干脆 0% skip。**AML / LE 的日常负载恰恰是「近 30 天全量 + 一批种子」，不是稀疏历史切片。** 所以下一倍速必须来自热循环与批量执行，而不是再加一层元数据。

### 3.3 SQL 图面的三个具体事实

1. `crates/gtv-engine/src/expr_util.rs`：`expr_to_u64` / `expr_to_i64` 只匹配 `Expr::Literal`，其余一律 `arguments must be literals`。`neighbors(src, valid_at)`、`khop(src, k, valid_at)` 因此**无法接受一列种子，也无法与子查询 join**。
2. 核心层其实已有批量能力：`TemporalCSR::neighbors_record_batch(&UInt64Array, valid_at)`（`csr.rs:532`）、`khop(&seeds: UInt64Array, k, t)`（`csr.rs:571`）、`gtv-core::traversal` 的 rayon-safe budget、`gtv-pattern::find / find_from`。**能力在 Rust 层，SQL 层没有出口**；更甚者 `gtv-pattern` 虽是 `gtv-engine` 的依赖，全仓 `gtv-engine/src/*.rs` 里没有任何 `gtv_pattern` 引用 —— AML 环路/motif 检测目前根本没有 SQL 面。
3. `context.rs:814 register_neighbors()` 会 `csr.clone()` **两次**（每个 UDTF 一份 `Arc::new(csr.clone())`），即注册时把整图复制两份；`graph.rs` 每次调用还把结果重建成 5 个 `Vec` + `RecordBatch`。对 1e7 边级图，这是实打实的内存与分配成本，且与「kdb 级内存效率」相反。

另外：`crates/gtv-engine/tests/` 14 个集成测试里**没有一条覆盖 `neighbors()` / `khop()` 的 SQL**。这条路径既没基准，也没回归测试。

### 3.4 未测的底层欠账

- `rayon` / AVX2 只出现在 `gtv-array`、`gtv-index`；图遍历主路径 `csr.rs::Neighbors::next()` 是标量循环 + chunk zone map，无 SIMD、无并行 frontier。
- `oceanbase-storage-lessons.md` §6 自己承认：「没有在 gtvdb 上跑过压缩 vs 不压缩的对照 benchmark」「P2 SIMD 下推与分层 skip index 收益需要 A/B」「需要 `perf stat` 实测」。**至今没跑。**
- 现有基准最大规模：`bench_traversal` 50k 节点 × 50 度 = 2.5M 边（单种子 k=4）、`bench_metadata` 1M 边。没有 1e8 边级、没有摄入吞吐、没有并发查询、没有 p99。

### 3.5 功能面仍在膨胀

`gtv-engine` 单 crate 24 个模块 / 约 1 万行（`cbo.rs` 936、`monitor.rs` 621、`workload.rs` 684、`dq.rs` 675…），核心仓 13 个 crate。这些是过去「平台化」留下的面积。用户明确要求「而非把各种功能硬塞进去」，因此**本轮不应再新增任何非图算子类 UDTF**；领域逻辑继续留在 `gtvdb_ext`，核心只保留注册接口。

---

## 4. 修正后的线路

判定规则（写进 CI 与 PR 模板）：

> 每个进主干的 item 必须能回答「它让 Q1/Q2/Q3 中哪一条的 p99 降了多少，基准在哪个文件」。答不上来的，进「触发条件清单」而不是进里程碑。

### P0 — 先把速度变成可证伪的（先测，不改算法）

| 交付 | 说明 |
|:--|:--|
| `crates/gtv-storage/examples/bench_fin_graph.rs`（或 `gtv-engine/examples/`） | 三组真实形态查询：① AML 时序资金流 / 环路（k-hop + 时间窗 + motif）② CRM 担保/抵押链传递 ③ LE 客户群 → 集团 → 敞口聚合。固定规模（建议 1e6 节点 / 1e7 边，合成但标注清楚），数据放仓库外 `C:/ai/gtvdb/_benchdata/`，沿用 `.gitignore` + CI generated-data guard |
| 断言 | 每条查询对照「全量扫描」逐条一致；`Overlay == Materialized` 沿用 `bench_correction` 口径 |
| 指标 | Q1/Q2/Q3 的 p50/p99、edges/s、publish→可见延迟；同机列式基线只作趋势对照，不写进 tagline |
| `perf stat`（Linux CI 可选 step） | `cache-misses` / `dTLB-load-misses` / `branch-misses`，补上 OceanBase 文档列的欠账 |
| 摄入吞吐基准 | 单进程 M edges/s + WAL fsync 成本拆分（Windows/Linux 已知差一个数量级：publish 19.5 ms vs 1.75 ms，只比趋势） |

**产出不是代码优化，是一张现状表。** 阈值在看到这张表之后再定。

### P1 — 只做三项（按预期收益排序）

1. **批量图算子（最高优先）**
   - 新增接受种子列的 `TableProvider` / 数组参数 UDTF，直接复用 `neighbors_record_batch()` 与 `khop(&seeds, k, t)`；一次规划、一次分配、输出尽量复用 buffer / 零拷贝投影。
   - 把 `register_neighbors()` 的两次整图 clone 收敛成一份 `Arc<TemporalCSR>` 共享。
   - 保留现有逐点 UDTF 兼容；补 SQL 回归测试（当前为 0）。
   - 验收：N=1e3/1e4/1e5 种子时，Q1 p99 相对「N 次逐点查询」的曲线；结果逐条一致。
2. **段内节点区间窄化 / 二维分片**
   - `node_id_min/max` 已在 manifest；写入时只覆盖 `[min,max]` 的 offsets，把 O(V) 变 O(ΔV)；或按 `(time_bucket, src_range)` 分片。
   - 验收：复刻 §5⑤ 的 50,303 边对照，目标把 1,439 B/edge 压到接近 145 B/edge，且不破坏 L2 的 `Skip / FullHit / Scan` 判定（复用 `l2_time_box.rs` 断言）。
3. **热循环 measured-first 调优**
   - 先读 P0 的 `perf stat`，再**只做被证明是瓶颈的一项**：16B 窄化 / SIMD 时间谓词 / Eytzinger / 大页 / 并行 frontier。
   - 明确不做「一整套 P2」；每项都要 A/B，输了就回滚。`csr.rs` 与 GTVSeg v1 的列格式恒等式（`disk == memory == cache line`）仍是红线，M3 的 NO-GO 判定继续有效。

### P2 — 冻结 / 降级清单

| 项 | 处理 |
|:--|:--|
| Parent+Diff、Branch、day-bucket、GTVSeg 列扩展 / sidecar、自动 GC 默认开 | 🧊 冻结。`minimal-temporal-update-impl.md` §14.8 的触发条件已写死，不主动重启 |
| M7 增量 frontier | 🧊 与 AML/CRM/LE 无关；仅 quantum state 有真实需求时再评估 |
| M8 / M9 / M10 / M12（crates.io、pyo3、demo、dylib loader） | 🗑 从主线下架，改为「出现真实嵌入用户后按需」；对速度零贡献，且会继续扩大接口面积 |
| 并发写入 + WAL 批量 fsync + kill -9 进程级测试 | ⏸ 仅当 Q3 证明单写者 `&mut self` / 每帧 fsync 是瓶颈时启动；现在补的是可靠性，不是速度 |
| L1 `stats.idx`、`[T1,T2)` 原生聚合、冷热淘汰 / L2 Parquet 归档 | ⏸ L1 目前是运维面（`gtv_tables()` 121.53 ms / 50 表、`sync_catalog` 329.40 ms），不在目标查询热路径；其余等规模超过内存时再谈 |
| 任何新的非图算子 UDTF / 领域功能 | ⛔ 本轮禁止；领域一律进 `gtvdb_ext` |

### 文档动作（需确认后执行）

- `ROADMAP.md`：把 L1/L2、M0/M1/M2/M3 补进已完成；把 compaction / GC 标为已完成；把 6 个未完成 item 压缩成「P0 基线 + P1 三项 + 冻结清单」；在「明确不做」里加一条「不以纯列式算子带宽定义性能目标」。
- `docs/perf-first-roadmap-review.md`（本文）转为主线评审记录，触发条件与门禁结论以本文为准。
- CI：新增「目标查询基准」step，与现有 `bench_correction` / `bench_m3_gate` 并列；保持 generated-data guard 不变。

---

## 5. 边界与风险

- **不要为了做基准而改读路径。** P0 只测，A/B 之前不改 `csr.rs`。M0–M3 用 5 个里程碑换来的「读路径零回归」是当前最值钱的资产。
- **正确性优先于速度。** 批量算子必须先证明与逐点查询逐条一致，再谈加速；L2 的交集/并集两组标量语义（交集短路会漏、并集短路会多算）不能因为优化而合并。
- **跨平台数字只比趋势。** Windows fsync 与 Linux 差一个数量级已实测；bitmap crossover 在本机 E/V 2–4、Linux CI 可低到 1。任何阈值要在目标部署机上重跑后再定。
- **「媲美 kdb」是目标，不是 tagline。** `docs/positioning.md` §3「不要写 kdb+ replacement」的判断仍成立；对外表述用可复现的查询延迟数字，不用口号。
- 本轮不修改代码、`ROADMAP.md`、CI 与任何用户放在 `docs/` 的未跟踪需求稿；`csr.rs` / `neighbors()` / GTVSeg v1 列格式一行未动。

---

## 6. 一句话

**当前仓库「能正确地改历史」已经做得很扎实，但「快」这件事既没有目标口径、也没有基线；把剩余 20 项砍成 3 项，先用 AML/CRM/LE 的批量图查询把速度量出来，再只修被证明慢的地方。**

---

## 7. M6-a 落地记录（2026-11-09）

按 §4 的裁剪口径执行 **M6 性能主线的第一段 M6-a**：先建口径，再只落地一项被证明有效、且不碰内核红线的改动。M6-b（段内节点窄化 / 热循环调优）仍冻结，触发条件见 §7.5。

### 7.1 交付清单

| 文件 | 内容 |
|:--|:--|
| `crates/gtv-engine/examples/bench_m6_gate.rs`（新增） | P0 准入基准：A 段 Q1 批量 k-hop、B 段 Q2 段粒度、C 段 Q3 摄入/发布；每段带正确性硬断言 |
| `crates/gtv-engine/src/batch_graph.rs`（新增） | `BatchNeighborsTable` / `BatchKhopTable` 两个 `TableProvider`，一次规划、一个 `RecordBatch`、可 join / 投影 / 过滤 |
| `crates/gtv-engine/src/context.rs` | `register_batch_graph()` / `register_batch_graph_shared()`；`register_neighbors()` 由两次整图 clone 改为共享一个 `Arc<TemporalCSR>` |
| `crates/gtv-engine/src/graph.rs` | 两个 pointwise UDTF 增加 `from_arc()`；公开 `new(&TemporalCSR)` 签名不变 |
| `crates/gtv-engine/tests/m6_batch_graph.rs`（新增） | 7 项集成测试，逐条比对批量与逐点结果 |
| `.github/workflows/ci.yml` | 新增 M6 测试（step 17）与准入基准（step 18） |

**内核零改动**：`csr.rs` / `neighbors()` / `endpoint.rs` / `segment.rs`（GTVSeg v1 定宽列区）一行未动。批量结果由同一批 kernel 入口在 Rust 循环里产出，因此可与逐点 UDTF 逐条比对。

### 7.2 A 段实测：批量算子 GO（Windows 本机 release）

合成金融图 `V=20,000 / E=200,000`，`valid_at=500,000`，`k=2`；`owned` = 每次注册都 clone 整图，`shared` = 复用 `Arc`；`spd` 相对「N 次逐点 SQL 全流程」。

| seeds | 逐点 p50 µs | 逐点 p99 µs | owned µs | shared µs | spd owned | spd shared | 逐条一致 |
|--:|--:|--:|--:|--:|--:|--:|:--:|
| 1 | 375.9 | 375.9 | 1412.3 | 215.9 | 0.27× | 1.74× | ✅ |
| 8 | 228.3 | 307.7 | 1334.6 | 211.0 | 1.38× | 8.75× | ✅ |
| 32 | 222.4 | 429.3 | 1384.3 | 290.7 | 5.70× | 27.13× | ✅ |
| 128 | 194.6 | 388.0 | 1723.4 | 496.7 | 15.76× | 54.68× | ✅ |

三条结论：

1. **门禁判定用保守的 owned 路径**，N≥32 时仍 5.7–15.8× → GO。单点查询在 owned 下亏 0.27×，符合 §4「不为单点付额外成本」的谨慎：`shared` 路径把固定成本从 ~1.4 ms 降到 ~0.2 ms 后，N=1 也变成 1.74×。
2. **整图 clone 是真实的隐藏成本**：`register_neighbors()` 原来注册一次复制两份图；新代码共享一个 `Arc`，种子集变化时用 `register_batch_graph_shared()` 零复制。基准第一次跑（未修）owned 1.47 ms，复测 shared 0.22 ms，差值 ~1.2 ms 就是这个拷贝。
3. 收益来自「少 N−1 次 parse/plan/collect」，不是算法变快 —— 与 §3.3 的判断一致，也与「不把复杂度塞进控制面」的方向一致。

### 7.3 B 段实测：段粒度债被量化，M6-b 有了触发线

同一批 200,000 条边切成不同段数，每段都付 `node_count+1` 个 `offsets` + `chunk_index`：

| 段数 | 边/段 | 构建 µs | B/edge（仅稠密索引） | 边数守恒 |
|--:|--:|--:|--:|:--:|
| 1 | 200,000 | 13,278.5 | 0.8 | ✅ |
| 10 | 20,000 | 7,992.8 | 8.0 | ✅ |
| 100 | 2,000 | 15,469.7 | 80.0 | ✅ |
| 500 | 400 | 62,339.7 | **400.0** | ✅ |

与 `metadata-benchmark.md` §5⑤ 的结论一致：段数从 100 涨到 500，构建慢 4×，稠密索引从 80 B/edge 涨到 400 B/edge。**`B/edge > 256` 的交叉点落在 100–500 段之间**，这就是 M6-b「段内节点区间窄化」的触发线；本轮不动格式，因为窄化要么改 GTVSeg 列区（M3 已判 NO-GO），要么改二维分片与 `graph_at()` 的位置语义，需要单独一轮 A/B + 兼容性测试。

### 7.4 C 段实测：只报数，不做跨平台结论

`ingest 20,000 边 = 6,458 µs ≈ 3.10 M edges/s`；`publish 22,783 µs`（Windows fsync 主导）；`visible_edges_at(publish_ts) == 20,000`。M2 已实测同代码 Linux publish 1.75 ms vs Windows 19.5 ms，**跨机器只比趋势**。`perf stat` / 冷缓存 / 并发查询仍未测，列为 M6-b 前置。

### 7.5 验收与边界

| 门禁 | 结果 |
|:--|:--|
| `cargo test -p gtv-engine --test m6_batch_graph` | **7 passed / 0 failed**（逐点 vs 批量逐条一致、时间过滤、shared/owned 等价、重复注册替换、SQL join、投影/过滤下推、空种子表） |
| `bench_m6_gate` 硬断言 | A 段所有 seeds 配置逐条一致；B 段不同分段的边集守恒 + segs=1 的活跃边集与全图一致；C 段发布前后可见边数一致 |
| 生成数据 | 全部内存构造 + tempdir；无 CSV / `.seg` / `.gtvd` / `.gtvobs` / `nodes_c*.bin` 入库，`.gitignore` 与 CI guard 不变 |
| 内核 diff | `csr.rs` / `endpoint.rs` / `segment.rs` 为空 |

**M6-b 冻结与触发条件**：

| 项 | 何时才做 |
|:--|:--|
| 段内节点区间窄化 / `(time_bucket, src_range)` 分片 | 持续段数进入 100–500 区间且 B/edge > 256；先 A/B 再决定是否动 GTVSeg 列区 |
| 热循环 16B 窄化 / SIMD 时间谓词 / Eytzinger / 并行 frontier | 需要 Linux `perf stat`（cache-misses / dTLB / branch-misses）证明瓶颈；一次只做一项，输了回滚 |
| 并发写入 / WAL 批量 fsync / 真实 kill -9 | 需要 Q3 在 Linux 上证明单写者 `&mut self` 或每帧 fsync 是瓶颈 |
| Parent+Diff / Branch / day-bucket / sidecar / 自动 GC | 继续沿用 `minimal-temporal-update-impl.md` §14.8 的触发条件，不主动重启 |
| M8/M9/M10/M12 分发与绑定 | 有真实嵌入用户再做 |

`ROADMAP.md` 本轮**仍未修改**（§4 的文档动作需使用者确认后再落）；本节作为 M6-a 的事实记录与后续对账依据。

### 7.6 Linux CI 对照（run `42572999527`，5/5 success）

同配置在 Ubuntu runner 上重跑，**结论跨平台一致，绝对值不可照搬**。

| seeds | 逐点 p50 µs | owned µs | shared µs | spd owned | spd shared | 逐条一致 |
|--:|--:|--:|--:|--:|--:|:--:|
| 1 | 328.1 | 524.0 | 280.3 | 0.63× | 1.17× | ✅ |
| 8 | 286.1 | 543.3 | 295.1 | 4.33× | 7.97× | ✅ |
| 32 | 283.6 | 565.5 | 324.0 | 16.33× | 28.51× | ✅ |
| 128 | 279.3 | 665.7 | 464.1 | 54.66× | 78.41× | ✅ |

- **A 段 GO 在 Linux 上更强**：整图 clone 的固定成本从 Windows 的 ~1.4 ms 降到 ~0.5 ms（内存分配器差异，不是算法），因此 owned 路径在 N=8 就已 4.33×，N=128 达 54.66×；shared 路径 N=1 也仍 >1。两平台共同结论：**瓶颈是 N 次规划，不是内核循环**。
- **B 段完全跨平台一致**：0.8 / 8 / 80 / 400 B/edge 是几何常量；构建 13.9 / 6.0 / 12.1 / 61.8 ms 与 Windows 同量级。`B/edge > 256` 的 100–500 段触发线不是机器相关的，可以作为 M6-b 的硬判据。
- **C 段再次证明只比趋势**：Linux ingest 1,522.8 µs ≈ **13.13 M edges/s**、publish 3,813.6 µs；Windows 为 6,458 µs / 22,783 µs，约 **4–6×** 差距。M2 的 fsync 结论在本轮复现。
- **runner 全量回归**：53 suites / **494 passed / 0 failed / 1 ignored**，与本机逐项一致；M6 step 17（7 项 parity）与 step 18（准入基准）均实际执行且 success。
- **已知 CI 成本**：step 18 是 kernel job 里第一个需要 release 编译 `gtv-engine`（DataFusion）的步骤，首次运行让该 job 从分钟级涨到约 20+ 分钟。后续应把 release 基准拆成独立 job 并复用 release cache，或仅在 `gtv-core` / `gtv-engine` / `gtv-storage` 路径变更时触发；这属于 CI 工程优化，不改变任何基准结论。
