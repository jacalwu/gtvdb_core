# 非对称双时间设计评审：能否照此实施

> 2026-11-09｜评审对象 `gtvdb-asymmetric-bitemporal-update-design.md`（1430 行）
> 对照 `gtvdb-minimal-temporal-update-design.md`（业务输入）、`minimal-temporal-update-impl.md`（上一轮收敛 + M0 实测）、`past-update-design.md`（位置化 retract）与 `gtvdb_core` 当前源码。
> 约束不变：**最简洁地支持全部业务场景，且不因实现复杂度导致性能下降。**

---

## 0. 结论先行

**方向可以照做，而且它的核心论点不是新方案，是把当前代码里已经存在的事实写清楚了。**

这份文档已经吸收了上一轮评审最关键的批评：删掉行级 `system_from / system_to`，把系统时间上移到 Snapshot / Manifest，查询固定为「先解析一次 Snapshot，再按 business_t 过滤」。这与本仓 `gtv-storage/src/bitemporal.rs:16-17` 的架构注释完全一致：

```rust
// The temporal CSR keeps indexing business time only; this store provides the
// system-time axis by versioning whole tables.
```

上一轮实测（`minimal-temporal-update-impl.md` §7）：行级双时间忠实模拟在 `neighbors()` 上是 **1.91–2.23x**，快照级方案是 **1.00x / 1.06x**。新文档 §2 的理由、§18 的复杂度模型、§21 不变量 1–3 都站在这个正确的一边。**读路径零回归的判断成立。**

但 §22.1 列的 **15 项 V1 必做**，不是「最简洁」：它把复杂度从 CSR 热循环搬到了**控制面**（commit / manifest / materialize / branch / GC）。控制面不会拖慢遍历，却会拖慢**提交、冷启动与恢复**，并且引入一批新的崩溃一致性窗口。直接全盘实施会违背用户「不要因复杂实现拖慢性能」的约束，只是风险位置换了。

**推荐：把这份文档当作目标架构，不当作 V1 工单。** 采纳 9 项、修订 6 处、推迟 4 项，分 3 个阶段落地；M0 已完成的读路径一行不改。

| 判断 | 条目 |
|:--|:--|
| ✅ 直接采纳 | §0/§2/§3 非对称模型、§7 Snapshot Delta（集合差）、**§8 Materialization**、§10 不重写 Edge、§11 端点策略、§12.3 区间切分、§14 不含行级 MVCC、§16 查询执行、§19 L2/L3、§20 GC 前提 |
| ⚠️ 需修订 | R1 `ActiveNodeBitmap`、R2 `version_id: u128` 执行面、R3 `ObservationVersion` 升格、R4 invalidation 原子性、R5 `Parent+Diff Manifest`、R6 `Branch` |
| ⏸ 建议推迟 | 后台并发 Materialize、Parent+Diff、Branch/Tag、day-bucket 缓存 |
| 🗑 可整段删除 | §24 迁移说明（本仓从未有行级 system time） |

---

## 1. 三份文档的关系：不是替代，是收敛

| | `gtvdb-minimal-...`（业务稿） | `minimal-temporal-update-impl.md`（上一轮） | 本文评审的 `asymmetric-...` |
|:--|:--|:--|:--|
| 系统时间落在 | **每行** `system_from/system_to` | 快照 / 文件集 | **Snapshot / Manifest** ✅ 已采纳 |
| 热循环判定 | 4 不等式 + retract 查表 | 2 不等式（原样） | 2 不等式（原样） ✅ |
| retract 语义 | 关闭行级 `system_to` | 位置化 `(segment_id,row)` | **REMOVE VERSION**：改变新 Snapshot 集合 ✅ 大幅改进 |
| 版本身份 | `u128 version_id` | 位置化 | `u128 version_id`（仍保留） ⚠️ R2 |
| 端点策略 | 三态枚举逐边查 NodeVersion | 1 bit + O(1) 包围盒 | 三态枚举 + `ActiveNodeBitmap` ⚠️ R1 |
| 控制面 | Delta Segment + Compaction | 3 原语 `PUBLISH/RETRACT/AS OF` | Snapshot lineage / Materialization / Branch / GC（完整但重） |

新文档修掉了上一轮 §6 表里最严重的 4 项（行级双时间、retract 推导 `system_to`、§9.2「可选」空洞、Delta Segment 含 system 列），**尤其 §7.3「REMOVE VERSION 不回写旧记录、不要求 Reader 构建 RetractMap」是本轮最有价值的语义澄清**。

保留下来的 3 处过度设计：`u128 version_id`、`ObservationVersion` 内核对象、三态策略枚举；新增的复杂度是控制面。

---

## 2. §22.1 的 15 项 V1 与代码对账

| # | §22.1 必做项 | 当前代码事实 | 状态 |
|--:|:--|:--|:--:|
| 1 | Snapshot System-Time Resolution | `store.rs:537 entries_at_system`、`:569 entries_at`；`bitemporal.rs:142 version_at` / `:162 as_of`；catalog `manifest.rs:88 Snapshot{parent,files,op,created_at}` | 🟡 部分，缺统一 lineage |
| 2 | Node/Edge/Observation Version | Node=`endpoint.rs:92 NodeLifetime`（内存）、Edge=GTVSeg 定宽列、Observation=`bitemporal.rs:43 BitemporalStore`（内存 `HashMap`） | 🟡 语义有，未落盘 |
| 3 | Valid-Time 查询 | `csr.rs:883-908`（2 比较 + chunk zone map）、`segment_meta.rs:66 PruneOutcome` | ✅ 完成 |
| 4 | SnapshotDelta | 无 | ❌ |
| 5 | REMOVE + ADD VERSION | `wal.rs:131 WalFrame` 仅 `OP_INSERT`（`:156` 拒绝其他 op） | ❌ 需 WAL v2 |
| 6 | Parent + Diff Manifest | 两层都是 **Full Manifest**：catalog 每快照 `files: Vec<DataFileId>`（`store.rs:329-346`）；storage `Manifest.segments`（`store.rs:166-175`） | ❌ 建议不做（R5） |
| 7 | Full Manifest Checkpoint | 现状即 Full Manifest，天然等价 | ✅ 完成 |
| 8 | 有界 Delta Overlay | 无 | ❌ |
| 9 | Materialization | `store.rs:616 graph_at(t)` 会物化 CSR，但是**纯并集**——更正后新旧版本会双计（`past-update-design.md` §3.1） | 🟡 最有价值的缺口 |
| 10 | EdgeType EndpointValidityPolicy | `endpoint.rs:34 EndpointPolicy`（1 bit / 0..=63）、`:201 EndpointView`、`:292 active_neighbors` | ✅ M0 完成 |
| 11 | ActiveNodeBitmap | 无；M0 用 `NodeLifetime` 定宽数组 + `all_endpoints_active_at(t)` O(1) 包围盒 | ❌ 建议降级（R1） |
| 12 | Valid-Time 区间切分 | 仅 `past-update-design.md` §4.4 设计 | ❌ |
| 13 | InvalidationEvent | 无（计划 `gtv_invalidations()` UDTF） | ❌ |
| 14 | 后台 Compaction | 无 | ❌ |
| 15 | 延迟 GC | 无 | ❌ |

**5/15 已有或基本具备，10 项待建；但 10 项全部在控制面，没有一项需要改 `csr.rs` 或 GTVSeg v1 的列。** 这是与上一轮业务稿最本质的区别，也是可以放心采纳的根据。

---

## 3. 必须修订的 6 处

### R1 — `ActiveNodeBitmap` 降级为可选加速器（§6 / 不变量 12 / §22.1#11）

文档把 bitmap 定为 `ClampToEndpoints` 的**必选**执行路径。它的缓存键是 `(snapshot_id, business_time, node_partition_id)`，意味着**每个不同的 business_t 都要 O(V) 构建一次**。

| 查询形态 | Bitmap | M0 `NodeLifetime` + 包围盒 |
|:--|:--|:--|
| 单点 / AS OF 查询（E_selected ≪ V） | 构建 O(V)，单次查询亏 10²–10³ 倍 | O(1) 包围盒；命中则 0 额外成本 |
| 全图 sweep，固定 t（E_selected ≫ V） | 构建 O(V) 摊薄，每次 1 bit test | 每 clamp 边 2×i64 读 + 2 比较 |
| 所有节点都活跃 | 仍要构建 + 每边 2 次 bit test | **0 额外成本**（直接返回原 `Neighbors`） |
| 缓存一致性 | 键含 snapshot_id；day-bucket 只能是候选集（文档 §6 已承认） | 无缓存，事实数组即真相 |

**建议**：`NodeLifetime` 是事实源（正确性唯一来源），bitmap 由 planner 按 `E_selected / V` 决定是否构建，定位为 L3 级可丢弃派生缓存——这其实正是文档 §19.2 自己写的「必须可从 Node Valid-Time 事实重建」，只是与 §21 不变量 12、§22.1#11 的措辞冲突，需统一。

另：bitmap 只能回答「t 时刻是否活跃」，无法给出 `effective_from / effective_to`；M0 的 `EndpointView` 按需显式计算，**不把遍历不消费的上界算进热循环**。

### R2 — `version_id: u128` 只做审计，Delta 的 REMOVE 用位置化执行（§4.4 / §7.3 / §8.3 / §14）

文档 §4.4 把「Snapshot Delta 精确替换」列为 `version_id` 的职责，这隐含一个全局 u128 索引。分工要拆开：

```text
removed_version_ids : Vec<u128>        → 审计 / Lineage / 冲突检测（文档 §4.4 原意，保留）
removed_positions   : Bitmap<(seg,row)> → 查询期执行（past-update-design.md I3）
```

原因只在**有界 Overlay 窗口**（§8.3）成立：Snapshot 尚未 Materialize 时，REMOVE 必须在查询期生效，此时 `HashSet<u128>` 每条边探测一次，位置化位图则是 O(1) 且 K 个 delta 直接 OR，不需要 K 路归并。一旦 Materialize 完成，两者都不进热路径——**文档的 §8 已经把这个成本移出了读路径，这点是对的**，R2 只是把过渡窗口也补干净。

代价写进契约：Compaction / Materialization **必须整体重写位置空间**，悬空 retract 必须报错而不是忽略（`past-update-design.md` §4.7）。由于 §8 本来就要求重写，这个代价已经被支付了。

### R3 — `TemporalObject` / `ObservationVersion` 不升格为内核对象（§4 / §13 / §14）

`series_id / metric_id / value_ref / quality_flags / source_id` 全是业务列，由调用方在 Arrow schema 定义；内核不需要 `TemporalObject` 枚举，也不需要新的定长结构体。`BitemporalStore::record()` / `as_of()` 已经提供「append-only 系统版本 + 行内业务区间」的完整语义。

**但必须补一句现状**：`bitemporal.rs:43` 是会话内 `HashMap`，**尚未持久化**。新文档 §13 的语义完全正确，缺的是把它接进 Snapshot / Manifest 并落盘——这是 M2 的实活，不是「零代码」。

另：§14 的「完整 Property / Value After-Image」可能是变长（字符串 / blob）。必须明确它是**独立 sidecar + offset 表**，不能混进定宽列区，否则 `segment.rs:17-23` 的 `disk == memory == cache line` 恒等式被破坏。文档 §14 只列了区块名，没写这个约束。

### R4 — InvalidationEvent 必须是 manifest 的只读投影（§13.2 / §15 step 9→11）

§15 的提交顺序是：`9 原子发布 Snapshot → 10 更新 main/branch ref → 11 发布 InvalidationEvent`。**10 与 11 之间崩溃，外部派生重算永远不会被触发，派生序列静默陈旧**——这是最难发现的一类一致性 bug，因为图查询本身完全正确。

改为：事件 = manifest 提交日志的只读投影（`gtv_invalidations(commit_from)` UDTF，沿用 L1 `gtv_tables()` 模式），随 step 9 原子可见。外部按 `commit_id` 增量拉取，重放天然幂等；**不引入事件总线、投递状态、exactly-once、重放队列**。`DerivationInfo` 的 6 个字段由外部写进自己的观测表列。

### R5 — Parent + Diff Manifest V1 不做（§9.3 / §22.1#6）

两层现状都是 Full Manifest。文档 §9.1 说它「较大」，但 `metadata-benchmark.md` 实测冷开与段数无关（49 段 2.90 ms vs 586 段 2.78 ms）；586 段 × ~200 B ≈ 120 KB，写一次 atomic rename 的成本远低于引入父链解析。

Parent+Diff 会带来：历史 Snapshot 打开需解析父链、Full Checkpoint 周期策略、Checkpoint 前后一致性测试（§23.7）、以及父链中途损坏的恢复路径。**收益未被测量，成本是确定的。** 建议 V1 保持 Full Manifest（它本身已等价于「Full Manifest Checkpoint」），只有当更正提交频率让 manifest 写入成为实测瓶颈时才切换。

### R6 — Branch 推迟，用「暂存 + 单次原子发布」满足 §13.3（§3.1 / §13.3 / §16.3 / §20）

`branch_id / sequence_number`、Branch Head、Branch GC 引用检查是一整套控制面，`past-update-design.md` §5.1 已判定现有线性 manifest 不支持分支。

但 §13.3 要解决的问题只是「基础修正与派生结果必须同时可见」，这**不需要分支**：现有 `MemSegment` + WAL 本来就是未发布暂存区，`flush_at()` + manifest `write_atomic` 本来就是单次原子发布。让外部重算结果作为同一批 delta 一起提交即可：

```text
WAL/MemSegment 暂存 base 修正 + 派生结果（对外不可见）
    → 一次 flush + 一次 manifest 原子发布
```

Branch 只服务于「多个分析者同时持有分叉认知」，不在 5 个业务需求里，放 M3。

---

## 4. 最有价值的新增：§8 Materialization（建议作为 M1 主轴）

`store.rs:616 graph_at(t)` 当前是纯并集：

```rust
for entry in self.entries_at_system(t) { append_segment_edges(...); }
MemSegment::to_csr(&edges, self.node_count)
```

更正一旦发布，新旧版本会同时出现在 CSR 里——正是 `past-update-design.md` §3.1 判定的硬阻断。文档 §8 给出的契约恰好是正确解法：

```text
Materialized Snapshot → 已解析的最终 Manifest → 折叠后的 Base CSR
                      → L2 剪枝 → Valid-Time 过滤
禁止：RetractMap / 行级 System-Time Filter / 无界 Delta Replay / 版本竞争解析
```

V1 把 `MaterializationState` 收敛成两态即可（`DeltaOnly` / `Materialized`），`Materializing` / `Failed` 等真正上后台并发时再加。`delta_depth` 起步取 **K=1**（§8.4 的「允许 Head 带一个 Delta」），由基准决定能否放宽。

**§23.6「Overlay 查询 == Materialized 查询」的随机属性测试，是整套设计里最该先写的测试**——它一条就能覆盖 §21 不变量 19、§8 的正确性、以及 R2 的位置空间重写。

---

## 5. 其他采纳项与一处顺带增强

- **§12.3 区间切分**：显式写出 `[A,start) 旧值 / [start,end) 修正 / [end,B) 旧值`，补上了上一轮指出的「`PUT 旧值 // 可选`」空洞。建议再明确「`start >= end` 时**不产生任何 Mutation**」。切分后一条逻辑边变多行，未来 `range_agg()` 必须按 `(src,dst,edge_type)` 去重（`past-update-design.md` §4.8）。
- **§19.1 L2 增强**：`src_id_min/max` 值得加。`SegmentMeta::from_csr()`（`segment_meta.rs:100`）本来就是一次线性扫描，多两个 min/max 近乎零成本；有了它，`neighbors(src,t)` 可直接跳过「不含该 src」的段，比现有 `node_id_min/max`（src∪dst 的并集）更紧。`dst_id_min/max` 优先级低，`observation count` 属于表级统计、不该进段元数据。
- **§24 迁移说明整段可删**：本仓从未写入行级 `system_from / system_to / retract_system_ts`，`bitemporal.rs:16-17` 明确系统时间靠整表版本化。所以「非对称模型」对本仓**不是迁移，是归档既有事实**。

---

## 6. 性能风险搬家了，门禁也要跟着搬

| | 上一轮业务稿 | 本设计 |
|:--|:--|:--|
| 风险位置 | **CSR 热循环**（实测 1.91–2.23x） | **提交 / 冷开 / 恢复** |
| 成因 | 行级 4 不等式 + retract 查表 | 完整 after-image、delta fsync、manifest 发布、后台 Materialize、父链解析 |
| 遍历吞吐 | 退化 | **保持 `csr.rs` 2 比较，零回归** ✅ |

所以「不会因为复杂实现导致性能下降」**对读路径成立**，但必须新增两类门禁，否则控制面的回归不会被发现：

1. **保留** `bench_endpoint`（release）作为读路径回归门禁：包围盒命中 `1.00x`、真实过滤 `1.06x`，`csr.rs` diff 必须为空。
2. **新增** `bench_correction`：单次更正提交延迟 / 每秒更正数、Materialize 前后 P99 查询延迟、`delta_depth` 从 1 到 K 的衰减曲线；用 `metadata-benchmark.md` 的仓库外数据，不提交任何生成数据。
3. **保持** `cargo test --workspace --offline` 基线 `428 passed / 0 failed / 1 ignored`。

---

## 7. 分阶段落地（建议）

| 阶段 | 内容 | 验收 |
|:--|:--|:--|
| **M0** ✅ 已完成 | `EndpointPolicy` / `NodeLifetime` / `EndpointView` / `active_neighbors`；7 项测试（含 250 组随机属性）+ release 零回归基准（提交 `7b812ff`） | 已通过 |
| **M1 更正可用**（= 文档 §8 + §7 + R2/R4） | 位置化 retract + WAL v2（after-image + REMOVE）+ `SnapshotDelta` + `graph_at()` 折叠 + 两态 `MaterializationState` + `nodes_<S>.bin` 快照文件 + `gtv_invalidations()` UDTF + L2 `src_id_min/max` | §23.2 / §23.5 / **§23.6 Overlay==Materialized 随机属性** / §23.8 崩溃点 1–4；`bench_endpoint` 不回归 |
| **M2 控制面** | 后台 Compaction、延迟 GC（含 snapshot/tag/mmap reader 引用检查）、`BitemporalStore` 落盘接入 Snapshot、观测表 checkpoint | §23.7 / §23.8 全 6 崩溃点；压缩前后同一 `(snapshot, business_t)` 结果逐条相等 |
| **M3 按需** | `ActiveNodeBitmap`（planner 按 `E_selected/V` 选）、Parent+Diff、Branch/Tag、day-bucket 缓存 | 每项必须有基准证明收益，否则不进主干 |

**dev / prod 差异**：dev 用 Full Manifest + K=1 + 同步 Materialize + 关闭自动 GC，让崩溃点可复现；prod 才开并发 Materialize、retention policy 与自动 checkpoint。

---

## 8. 风险提示与常见坑

1. **写放大**：一次更正 = 一个 delta 文件 + fsync + manifest 发布。批量更正必须合批到同一 commit，否则小更正洪峰会打爆提交路径。
2. **位置空间失效**：Compaction 只允许整段参与；悬空 `(segment_id,row)` 必须报错，忽略会静默指向新段的无关行（最危险的失败模式）。
3. **旧快照依赖**：`AS KNOWN AT S` 需要旧的 `nodes_<S>.bin` 才能重建端点有效性；GC 不能只按「无 Segment 引用」回收。
4. **事件静默陈旧**：R4 不修，图查询全对但派生指标永远不更新，且没有任何报错。
5. **bitmap 正确性**：缓存删除后查询结果必须逐条不变（§23.4）；键漏掉 `snapshot_id` 会复用节点修正前的结论。
6. **去重复口径**：区间切分后 `range_agg()` / `active_edge_count()` 必须按逻辑边去重，否则 `FullHit` 短路的 `edge_count` 会双计。
7. **定宽恒等式**：变长 Property after-image 必须走 sidecar；混进列区会破坏 mmap 零拷贝。

---

## 9. 一句话交付

**照这份文档的方向做是对的——它把系统时间留在 Snapshot、Valid Time 留在行，正是本仓已经实现并实测过零回归的路线；但不要照 §22.1 的 15 项一次全做：保留 M0 的读路径与 GTVSeg v1 一列不加，把 §8 Materialization 作为 M1 主轴，再用 R1–R6 剪掉 Bitmap 必需化、u128 执行面、Observation 内核对象、事件投递窗口、Parent+Diff 和 Branch。这样复杂度只落在控制面，遍历仍是 2 次比较。**

---

## 10. 证据索引

**本仓源码**

- `crates/gtv-storage/src/bitemporal.rs:1-17, 43, 142, 162` — 整表版本化系统时间轴；`BitemporalStore` 仍为内存 `HashMap`。
- `crates/gtv-storage/src/store.rs:82-114, 166-175` — `SegmentEntry` / Full `Manifest`；`:537 entries_at_system`、`:569 entries_at`、`:616 graph_at` 纯并集。
- `crates/gtv-storage/src/wal.rs:131-158` — `WalFrame` 仅 `OP_INSERT`。
- `crates/gtv-storage/src/segment_meta.rs:44-62, 100` — `SegmentMeta` 标量、`from_csr()` 单次线性扫描。
- `crates/gtv-storage/src/segment.rs:17-23` — GTVSeg v1 定宽 / 64B 对齐 / `disk == memory == cache line`。
- `crates/gtv-core/src/csr.rs:871-909` — `Neighbors::next()` 热循环（2 比较 + chunk zone map）。
- `crates/gtv-core/src/endpoint.rs:34, 92, 201, 292` — M0：`EndpointPolicy` 1 bit、`NodeLifetime`、`EndpointView` 包围盒、`active_neighbors()`。
- `crates/gtv-catalog/src/manifest.rs:88-101` — catalog `Snapshot{parent, files, op, created_at}`。
- `crates/gtv-catalog/src/store.rs:329-346` — 每快照写完整 `files` 清单 + `latest` 原子指针。

**文档**

- `docs/gtvdb-asymmetric-bitemporal-update-design.md` — 本次评审对象。
- `docs/gtvdb-minimal-temporal-update-design.md` — 5 个业务场景输入（行级双时间版本）。
- `docs/minimal-temporal-update-impl.md` §7 — `bench_endpoint` 实测 1.00x / 1.06x vs 1.91–2.23x。
- `docs/past-update-design.md` §3.1 / §4.7 / §5.1 — `graph_at` 双计、位置化 retract  compaction 契约、分支阻断。
- `docs/metadata-benchmark.md` — 冷开与段数无关、FullHit 短路。
- `docs/metadata-design-comparison.md` §6 — L2/L3 分离、可丢弃派生缓存。
