# 最小化时序更新：把 6 个原语收敛成 3 个

> 2026-11-09｜针对 `gtvdb-minimal-temporal-update-design.md` 的业务补充，结合 `past-update-design.md` 与 `gtvdb_core` 当前代码事实。
> 目标：**支持全部 5 个业务场景，且不因为实现复杂而让图遍历变慢。**
> 状态：收敛设计 + M0 / M1 / M2 / M3 全部落地。M3 按准入门禁实测裁剪（§14）。`csr.rs` 与 GTVSeg v1 列格式从 M0 起一行未改。

---

## 0. 结论先行

新文档的业务范围是对的，但它的**实现机制会让 gtvdb 变慢**，而且慢在最不能慢的地方——`neighbors()` 的每条边。

根因只有一个：**它把双时间从「快照级」降到了「行级」**。gtvdb 现有的性能优势恰恰来自相反的选择：

| | 新文档 §3（行级） | gtvdb 现状 / 本设计（快照级） |
|:--|:--|:--|
| 系统时间存哪 | **每行** `system_from/system_to` | 每个**快照**一个 `system_ts` |
| 查询时做什么 | 每行判 4 个不等式 + 查 retract 才能知道 `system_to` | **选一次文件集**，之后每行只判 2 个不等式 |
| `system_to` 怎么来 | RETRACT 推导 → 查询期查表 | 版本天然封闭，不需要逐行关闭 |
| CSR 热路径 | 2 比较 → 4 比较 + 1 次版本索引查找 | 保持 2 比较（`csr.rs:895-898`） |
| 是否破坏 mmap 零拷贝 | 是（变长版本对象） | 否（定宽列不变） |

代码里已经把这个选择写成了架构注释，不是推测：

```rust
// crates/gtv-storage/src/bitemporal.rs:9-13
// The temporal CSR keeps indexing business time only; this store provides the
// system-time axis by versioning whole tables.
// crates/gtv-core/src/bitemporal.rs:9-11
// two axes in one index would blow up the structure
```

**收敛后的结论：5 个业务需求只需 3 个内核原语，`neighbors()` 一行不改。** release 实测（20k 节点 / 400k 边，见 §7）：端点包围盒命中时 ACTIVE 遍历相对原 `neighbors()` 是 **1.00x**，需要真实过滤时 **1.06x**；忠实模拟新文档的行级双时间方案则是 **1.9–2.2x**。

| 业务场景（新文档 §1） | 承载机制 | 需要新内核对象？ |
|:--|:--|:--:|
| 1. Node 到期后部分 Edge 在遍历中失效 | `EndpointPolicy` 位图 + `NodeLifetime` 定宽数组 + 切片级短路 | **唯一需要新热路径机制的一条**，但增量有上界 |
| 2. Node 历史 `valid_to` 追溯修正 | 发布一份新的 `NodeLifetime` 快照文件 | ❌ 复用快照机制 |
| 3. 观测值（CPI / 波动率 / 久期…）修正 | `BitemporalStore::record()` + `as_of()` | ❌ **已存在，零代码** |
| 4. 派生时间线重算 | manifest 提交日志的只读视图 `gtv_invalidations()` | ❌ 复用 L1 UDTF 模式 |
| 5. 保留「现在正确」与「过去认知」 | `AS OF SYSTEM` 选文件集 | ❌ 已有 `entries_at_system` / `version_at` |

---

## 1. 决定性分叉：系统时间放在行上还是快照上

新文档 §3 规定「每个可版本化事实」都含 `[system_from, system_to)`，可见性判 4 个不等式；§4 的 `NodeVersion` 又只存 `system_from`，`system_to` 由 §4.4 的 `RETRACT target_version_id` 推导。两者合起来意味着：

```text
查询一行是否可见
  → 取该行的 version_id (u128)
  → 查 retract 表：这个版本被关掉了吗？在哪个 system_ts？
  → 合成 system_to
  → 再判 4 个不等式
```

这段逻辑会进入 `Neighbors::next()`（`csr.rs:883-908`）——图漫步里执行次数最多的循环。现在它只有 2 次数组读 + 2 次比较，且靠 `all_active_at()` / chunk zone map 可以整块跳过。

**快照级做法**把这一切提前到查询开始之前：

```text
AS OF SYSTEM S
  → commit_id 决定一组不可变文件：
      边段 GTVSeg[*]  +  节点数组 nodes_<S>.bin
      + 观测表版本    +  retract 位图
  → 之后所有对象都只有业务时间一维
  → CSR 热路径与今天完全相同
```

这不是简化需求，而是**换了一个更省的实现满足同一需求**：

- 「过去当时的认知」= 用旧 `commit_id` 选旧文件集（`BitemporalStore::version_at`，`bitemporal.rs:132-140`）；
- 「现在认为正确的历史」= 用最新 `commit_id`；
- 两者都不会被后续写入改变，因为文件不可变。
- §15 不变量 13「同一 Snapshot/Business/System 下最多一个可见版本」在快照级模型下**自动成立**，不需要运行时去重校验。

> 与 `past-update-design.md` 的 I1/I2 一致：冲突在提交期解决、系统时间在物化前折叠。本设计只是把这个原则从「边」推广到「节点 / 观测 / 事件」三类对象。

---

## 2. 六个原语 → 三个

新文档 §2 的六个原语里，有三个不是原语，是组合或派生视图：

| §2 原语 | 收敛后 | 落在哪 |
|:--|:--|:--|
| `PUT VERSION` | 并入 **PUBLISH** | 现有 `write_atomic()` + `flush_at()` |
| `RETRACT VERSION` | **RETRACT**（位置化位图） | `past-update-design.md` §4.2 |
| `QUERY AS OF` | **AS OF**（选文件集） | 已有 `entries_at_system` / `version_at` / `as_of` |
| `PUBLISH SNAPSHOT` | **PUBLISH**（原子发布 commit_id） | 现有 manifest `write_atomic` 顺序 |
| `EMIT INVALIDATION` | AS OF 的**派生视图** | `gtv_invalidations(commit_from)` 只读 UDTF |
| `COMPACT / GC` | RETRACT 的**重写规则**，非用户原语 | 后台任务；位置化 retract 的既有契约 |

三个原语的定义：

```text
PUBLISH(files, commit_id)  追加不可变定宽文件 + 原子发布 manifest
RETRACT(segment_id, row)   随某次 PUBLISH 一起发布的位置位图
AS OF(commit_id)           选出该 commit 可见的全部文件，冻结成一次查询的视图
```

`AS OF` 是**一次**选择，不是每行判定。这是与 §3 最本质的区别。

---

## 3. 唯一进热路径的需求：Node 到期让部分 Edge 失效

新文档 §5 / §8.3 的做法是 `edge_visible()` 每条边都取 `src`/`dst` 的 `NodeVersion`，做一次 u128 索引查找 + 双时间判定。这是最大的性能风险点。

### 3.1 三态策略在内核里坍缩成 1 bit

看新文档自己的 §8.3：

```rust
EndpointValidityPolicy::PreserveHistoricalReference => true,
EndpointValidityPolicy::External                  => true,
```

`Preserve` 和 `External` **对内核的行为完全相同**，区别只在「外部是否订阅影响事件」。因此内核不需要三态枚举，只需要知道「哪些 `edge_type` 需要 clamp」：

```rust
/// 1 = ClampToEndpoints；0 = 内核不约束端点（Preserve / External 由业务侧区分）。
pub struct EndpointPolicy { clamp_types: u64 }   // 覆盖 edge_type 0..=63
```

一次 `(clamp_types >> et) & 1` 完成门控。**非 clamp 的边零成本**。三类业务语义仍然完整保留在 schema 与外部订阅关系里。

### 3.2 三条把开销压回 O(1) 的观察

1. **`src` 在一条邻接切片里是常数**（`offsets[src]..offsets[src+1]`）。所以 src 的生命周期检查可以提到 `active_neighbors(src, t)` 入口做**一次**；若 src 在 `t` 已失效且该切片含 clamp 边，整个迭代器直接为空——这是 O(1) 剪枝，比逐边检查更快。
2. **dst 只需 1 次定宽数组读**。`NodeLifetime` 是 `valid_from: &[i64]` / `valid_to: &[i64]`，按 `node_id` 直接索引，无哈希、无 u128、无间接跳转；且只在 clamp 边上读。
3. **段级包围盒让常见情况完全退化**。`EndpointView` 在绑定时做一次顺序扫描，算出 clamp 边涉及节点的 `max(valid_from)` / `min(valid_to)`。若 `t` 落在这个窗口内，则**所有端点都活跃**，`active_neighbors()` 直接委托原 `neighbors()`——额外成本 2 个 i64 比较。

### 3.3 复杂度契约

| 情况 | 每边额外成本 | 说明 |
|:--|:--|:--|
| HISTORICAL GRAPH（§13.2） | **0** | 直接调用原 `neighbors()`，`csr.rs` 一行不改 |
| ACTIVE GRAPH，policy 为空 | **0** | 同上 |
| ACTIVE GRAPH，所有端点活跃 | **0**（每次邻居查询 2 次标量比较） | 段级包围盒短路 |
| ACTIVE GRAPH，部分端点过期，非 clamp 边 | **0**（1 次位运算） | 策略位图门控 |
| ACTIVE GRAPH，部分端点过期，clamp 边 | 1 次数组读 + 2 次比较 | 有界且 cache 友好 |

**没有哈希表、没有分配、没有行级版本查找、没有 CSR 列复制。**物化期求交（§14.1 的替代方案）会破坏 mmap 零拷贝，本设计不采用。

### 3.4 语义完整性

`active_neighbors()` 只回答「`t` 时刻哪些边可见」，返回的仍是边的固有有效期；§5.1 的 `effective_from/effective_to` 交集公式由 `EndpointView::effective_from()` / `effective_to()` **显式按需计算**。把展示用的上界算进每条边的热循环，是拿遍历成本补贴一个遍历根本不消费的值。图漫步 `khop_*` 只看可见性，因此**零改动**。

---

## 4. 需求 2 / 3 / 5：一个 `commit_id` 贯穿所有对象

### 4.1 节点生命周期修正

`NodeLifetime` 是**一份不可变定宽快照**（两个 `i64` 数组 × `node_count`）。修正 `valid_to` = 发布一份新快照 + 新 `commit_id`；旧快照仍被旧 `AS OF SYSTEM` 引用。

```text
S1: nodes_S1.bin   Company A [2010, 2029)
S2: nodes_S2.bin   Company A [2010, 2027)     ← 只改一个 u64，发布新文件
AS KNOWN AT S1 → 2029；AS KNOWN AT S2 → 2027
```

**不扫描、不重写任何 Edge**（§6.2 的这条约束保留）。生产实现用按页 COW / delta 文件避免 O(V) 全量复制，**语义不变**；M0 原型用全量复制，只验证语义。

`EdgeTypeSchema` 的 `require_src_visible_on_insert` / `require_dst_visible_on_insert` 是**提交期校验**，放写路径，不进 `neighbors()`。

### 4.2 观测值修正：零新代码

`BitemporalStore`（`crates/gtv-storage/src/bitemporal.rs:43`）已经是「append-only 系统版本 + 行内业务区间」：

```rust
record(table, system_from, batches)   // 修正 = 追加一个新系统版本
as_of(table, business_ts, system_ts)  // SQL: as_of('cpi', t, S1) vs (t, S2)
```

SQL 入口也已在 `crates/gtv-engine/src/bitemporal.rs:5-6` 注册。**`ObservationVersion` / `series_id: u128` / `value_ref` / `quality_flags` 全部是业务列**，由调用方在 Arrow schema 里定义，内核不需要认识它们，也不需要新的 `TemporalObject` 枚举。

> **范围说明**：当前 `BitemporalStore` 是会话内 `HashMap`（`bitemporal.rs:43`），**尚未持久化**。这里说「零新代码」指的是**语义与查询形态已经具备**，不需要设计新的内核对象；把版本表接进 L1 快照与 manifest、落到磁盘，属于 §9 的 M2。在 M2 之前，观测值的 `AS KNOWN AT` 只在会话生命周期内有效。

> 这一条是最大的过度设计：把「一张时序表」升格成「内核第四类持久化对象」，会让 Delta Segment 从纯定宽变成变长混合记录，直接推翻 §11 自己要求保留的定宽布局。

### 4.3 统一选择

```text
commit_id C  →  { GTVSeg 段清单,      nodes_C.bin,
                  observation 版本 C,  retract 位图 C }
```

四类对象共用**同一个** `AS OF` 选择器，因此「两种认知」天然一致，不会出现「边是 S1 的认知、节点是 S2 的认知」。这正是 §15 不变量 13 想要的保证，且不需要逐行 `system_from/system_to`。

---

## 5. 需求 4：InvalidationEvent 不需要新对象

§10 的 `InvalidationEvent`（8 个字段）与 `DerivationInfo`（6 个字段）都是**业务负载**，不是内核状态：

- 内核只在 manifest 里记录每次 commit 的 `{commit_id, object_kind, object_id, affected_valid_from/to, reason}`；这本来就是 `PUBLISH` 的必要元数据。
- 对外暴露 `gtv_invalidations(commit_from)` 只读 UDTF（L1 已有 `gtv_tables()` 等 UDTF 模式），外部按 `commit_id` 增量拉取。
- `input_fingerprint` / `algorithm_version` / `computed_at` 由外部写进自己的观测表列；内核不计算、不校验、不索引。
- 派生重算结果作为**普通新版本**写回，再一次 `PUBLISH` 原子发布——与 §10 步骤 6 一致。

**收益**：没有事件总线、没有投递状态、没有 exactly-once 语义、没有重放队列；「事件」就是提交日志的只读投影，重放天然幂等。

---

## 6. 对 `gtvdb-minimal-temporal-update-design.md` 的逐节修订

| 章节 | 问题（性能或一致性） | 收敛后 |
|:--|:--|:--|
| §3 统一双时间模型 | 行级 4 不等式 + retract 推导 `system_to`，进入热循环 | **快照级**：选一次文件集，热路径维持 2 不等式 |
| §4 `version_id: u128` × 4 类对象 | 全局版本索引、变长记录、破坏 64B 对齐 | 位置化 `(segment_id,row)` + 文件级版本；业务 id 留在业务列 |
| §4.4 RETRACT 精确指向 `version_id` | 每次查询需 u128 → 位置解析 | 位置化位图，O(1) |
| §5 `EndpointValidityPolicy` 三态 | 枚举 + 每条边查 NodeVersion | **1 bit 位图**（Preserve 与 External 在 §8.3 已同行为） |
| §8.3 `edge_visible` 逐边查端点 | src 被重复查 degree 次 | src 检查提到**切片级 O(1)**；dst 只在 clamp 边查 |
| §9 `ObservationVersion` | 新增第四类内核对象 | 复用 `BitemporalStore` + `as_of()`，零代码 |
| §9.2 区间切分「PUT 旧值 // 可选」 | 「可选」留白 → 空洞语义未定义 | 显式策略 `AssertOverWindow`（默认，重叠报错）/ `PreserveHoles` |
| §10 `InvalidationEvent` / `DerivationInfo` | 新持久化对象 + 投递语义 | manifest 提交日志 + 只读 UDTF；业务字段外部持有 |
| §11 Delta Segment 混装 4 类记录 | **最严重**：Node/Edge/Observation/Property 变长混排，破坏 `disk == memory == cache line` 恒等式（`segment.rs:17-23`） | 按对象拆**各自定宽**文件；GTVSeg v1 一列不加 |
| §14.2 effective validity 缓存带 3 个 u128 | 缓存比计算更贵，三次版本查找 | 段级 2 个 i64 包围盒 + O(1) 短路 |
| §14.3 Delta Fan-In K 多路归并 | retract 若是 u128 查找，合并才需要 K 路归并 | 位置化位图做 OR，K 可放宽，无需归并 |
| §15 不变量 13 | 需要运行时去重校验 | 快照级文件集确定性 → 自动成立 |
| §15 不变量 5 / §14.1 | 「查询期交集、不批量改写」正确 | **保留**，只是把交集实现成 O(1) 门控而非逐边版本查找 |
| §12 提交流程 14 步 | 步骤 11「发布事件」与 10 之间有一致性窗口 | 事件是 manifest 的投影，随 10 原子可见 |

**必须保留的三条**（新文档比 `past-update-design.md` 更完整的地方）：

1. §5 / §7 —— 端点有效性**按 edge_type 分策略**，不能全局一刀切；合同 / 审计关系不能被端点终止自动删除。
2. §14.1 —— Node 生命周期变更**不同步扫描或重写 Edge**，避免写放大。
3. §10 —— 内核不实现业务级联与依赖 DAG，派生重算外部化。

---

## 7. 性能契约与实测

| # | 契约 | 验证方式 | 结果 |
|:--|:--|:--|:--|
| P1 | `csr.rs::neighbors()` 源码零改动 | git diff；HISTORICAL GRAPH 直接调用原函数 | ✅ 仅 `lib.rs` 加了一行 `mod endpoint;` |
| P2 | 包围盒命中时 ACTIVE 吞吐不回归 | `bench_endpoint` release，min-of-3 × 100 sweeps | ✅ **1.00x** |
| P3 | ACTIVE 与朴素 `edge_visible()` 结果逐条一致 | 250 组随机属性测试（寿命 × 策略 × 时间 × 全部 src） | ✅ 全部一致 |
| P4 | 真实过滤时的增量有上界 | `bench_endpoint` | ✅ **1.06x**（且少返回 4.5% 已失效边） |
| P5 | 更正提交不扫描全图边 | 只发布新 `NodeLifetime` 快照，无 edge 段写入 | ✅ `with_valid_to()` 只碰节点数组 |
| P6 | 全量测试零失败 | `cargo test --workspace --no-fail-fast --offline` | ✅ **428 passed / 0 failed / 1 ignored**（新增 7 项） |

**实测**（`cargo run --release -p gtv-core --example bench_endpoint`，20,000 节点 × 20 边 = 400,000 边，其中 200,000 条为 clamp 类型；Windows / MSVC release，min-of-3）：

| 场景 | 原 `neighbors()` | 本设计 ACTIVE | 文档行级双时间方案 |
|:--|--:|--:|--:|
| `t=1000`，无节点过期，段级包围盒命中 | 2,086 µs | **2,087 µs（1.00x）** | 4,648 µs（**2.23x**） |
| `t=1600`，5% 节点已过期，需真实过滤 | 804 µs | **851 µs（1.06x）** | 1,537 µs（**1.91x**） |

- 对照组「文档行级双时间」忠实模拟新文档 §3/§8.3：`version_id` 经索引解析（用 `HashMap` 作最便宜的替身）后，对两个端点各判 4 个不等式。**不是稻草人**——这正是 u128 版本 id + 行级 `system_from/system_to` 的必要成本。
- 每条路径返回的边集合都 `assert_eq` 到同一答案（319,801 / 75,160 条），避免「剪得干净但漏数据」。
- 基线重测漂移 0.92x，说明 ±8% 内的差异是测量噪声；`active: no policy` 的 1.09–1.21x 也落在这个带内。**结论只采信量级明确的两项：包围盒命中 1.00x、行级方案 ~2x。**

> 仍有可优化空间：把 `all_endpoints_active_at(t)` 从 per-src 提到 per-query（`EndpointView::at(t)` 句柄）可再削掉一层固定开销。M0 不做的理由是它不改变结论，却会多一个生命周期参数；留给 M1 与实际查询规划器一起设计。

---

## 8. 保留的既有约束（不因本次扩展放松）

- **不提交 CSV / 生成数据**：沿用 `P1` 硬规则，所有基准数据在仓库外且内存/tempdir 生成。
- **GTVSeg v1 二进制格式不变**：端点策略、节点寿命、观测版本都在**旁路文件 / 上层表**，不碰 `col_dir` 的 7 个定宽列。
- **L2 判定式不变**：`SegmentMeta` 六个标量原样；节点包围盒是 `EndpointView` 的绑定期产物，不进 manifest（避免 manifest 膨胀，符合 `metadata-design-comparison.md` §6 的 L2/L3 分离原则）。
- **L3 不参与正确性**：§14.4 与 §15 不变量 15 完全保留。
- **区间汇总去重**：`past-update-design.md` §4.8 的坑仍在——区间切分让一条逻辑边变多行，未来的 `range_agg` 必须按 `(src,dst,edge_type)` 去重。观测值修正没有这个问题（点值替换）。

---

## 9. 落地状态与阶段

| 阶段 | 内容 | 状态 |
|:--|:--|:--:|
| **M0** | `EndpointPolicy` / `NodeLifetime` / `EndpointView` / `active_neighbors()`；7 项测试（含 250 组随机属性测试）+ release 零回归基准 | ✅ 本次完成 |
| M1 | 位置化 retract + WAL v2 + `graph_at` 折叠；`nodes_<S>.bin` 快照文件；`gtv_invalidations()` UDTF | ✅ §12 |
| M2 | 两阶段后台 compaction、延迟 GC、观测表落盘、L2 聚合去重、段内源点窄化 | ✅ §13 |
| M3 | 准入门禁实测；planner 可选 `ActiveNodeBitmap`、常驻维护线程 + 滚动 retention、轻量 Tag；Parent+Diff / Branch / 列扩展按实测判定不进主干 | ✅ §14 |

M0 落点：

- `crates/gtv-core/src/endpoint.rs` — 全部新机制，约 520 行（含测试）。
- `crates/gtv-core/src/lib.rs` — 仅加 `pub mod endpoint;` 与一行 re-export。
- `crates/gtv-core/examples/bench_endpoint.rs` — 性能契约 P2/P4 的可复现脚本。
- `csr.rs` / `segment.rs` / `store.rs` / `catalog_meta.rs` **零改动**。

M0 是**可独立丢弃**的验证：即使 M1 决定换实现，它的属性测试仍是 `edge_visible()` 的黄金对照，基准仍是「行级 vs 快照级」分歧的度量。

---

## 10. 一句话交付

**把双时间从「行」上移到「快照」，5 个业务场景就只需要 `PUBLISH / RETRACT / AS OF` 三个原语：节点到期靠 1 bit 策略位图 + 切片级 O(1) 短路，观测修正靠已有的 `BitemporalStore`，派生重算靠提交日志的只读视图；GTVSeg 一列不加、`neighbors()` 一行不改，性能优势（定宽 mmap 零拷贝 + 2 次比较的热循环）完整保留。**

---

## 11. 证据索引

**本仓源码**

- `crates/gtv-storage/src/bitemporal.rs:1-13, 43, 95-140` — 整表版本化的系统时间轴、`record()` / `version_at()` / `as_of()`。
- `crates/gtv-engine/src/bitemporal.rs:5-6` — `as_of(table, business_ts [, system_ts])` SQL 入口。
- `crates/gtv-core/src/csr.rs:871-909` — `Neighbors::next()` 热循环（2 次比较）、`all_active_at()`（`csr.rs:386-388`）。
- `crates/gtv-core/src/bitemporal.rs:9-11, 120` — 「两轴同索引会 blow up」注释；`overlaps()` 判据。
- `crates/gtv-storage/src/segment.rs:17-64` — GTVSeg v1 定宽 / 64B 对齐 / 7 列目录 / 「disk == memory == cache line」。
- `crates/gtv-storage/src/store.rs` — `Manifest`、`entries_at_system()`；**`graph_at()` 在 M1 已改为折叠 retract**（见 §12，不再是纯并集）。
- `crates/gtv-storage/src/segment_meta.rs` — `SegmentMeta` 时间盒 + 节点/源点区间标量。
- `crates/gtv-core/src/table.rs:73` — `NodeTable`（属性是业务列，无版本字段）。
- `crates/gtv-core/src/endpoint.rs` — M0 实现：`EndpointPolicy` 位图、`NodeLifetime` 快照、`EndpointView` 包围盒、`ActiveNeighbors` 三态迭代器、7 项测试。
- `crates/gtv-core/examples/bench_endpoint.rs` — §7 实测脚本，含顺序对照与行级双时间模拟。

**文档**

- `docs/gtvdb-minimal-temporal-update-design.md` — 本次业务输入（§1 五场景、§5 三态策略、§10 派生、§15 不变量）。
- `docs/past-update-design.md` — 上一轮：位置化 retract、I1/I2、跨版本 sweep、区间汇总去重。
- `docs/metadata-design-comparison.md` §6 — L2/L3 分离、只有可加/可并的量才进段级元数据。
- `docs/metadata-benchmark.md` — 冷开 O(1)、FullHit 短路、段粒度必须 ≳ 节点数。

---

## 12. M1 落地：更正可用，读路径仍然零回归

M0 证明了「快照级系统时间 + `EndpointView`」不会拖慢遍历；M1 补的是**过去时点更正**，即 `docs/asymmetric-bitemporal-review.md` 裁剪后的第一优先级工单。

### 12.1 两层边界

| | 查询面（M0，未改） | 控制面（M1，新增） |
| :-- | :-- | :-- |
| 位置 | `gtv-core`（`csr.rs` / `endpoint.rs`） | `gtv-storage`（`delta.rs` / `wal.rs` / `node_lifetime.rs` / `store.rs`） |
| 动作 | `neighbors(src, t)` 两次比较 | 提交期区间 sweep、位置化 retract、Delta 文件、manifest 发布、compaction |
| 代价 | 0（Passthrough 直接返回原 `Neighbors`） | O(E) 一次，仅发生在提交 / 版本查询 / compaction |

**`csr.rs`、`GTVSeg v1` 的 7 列目录、`neighbors()` 热循环一行未改。** 拖慢遍历的风险因此不存在；风险搬到了提交延迟、冷开与崩溃恢复，用属性测试和注入测试覆盖。

### 12.2 三个原语在 M1 的落点

| 原语 | 实现 |
| :-- | :-- |
| `PUBLISH` | `SegmentStore::flush_at(system_ts)`（原有）；新增 `publish_delta()` 作为更正的发布点 |
| `RETRACT` | `REMOVE VERSION` = 从**结果快照**的可见集合移除 `(segment_id, row)`；`RetractSet` 是排序 `Vec<RowPos>`，`contains()` 一次二分 |
| `AS OF` | `live_rows_at(system_t)` 唯一枚举入口；`graph_at()` / 更正规划 / compaction 全部复用它，无法对「可见」产生分歧 |

`u128 version_id` 只作为审计列 `removed_version_ids`（由 `RowPos::audit_id()` 派生），**任何读路径都不查它**，也不需要全局索引。

### 12.3 关键实现决策（对应评审 R1–R6）

- **区间 sweep 在发布内部执行**：第二次更正会触发强制 compaction，而 compaction 重写整个位置空间。若 `correct_edge()` 先按旧位置规划 retract，位置必然悬空。`DeltaDraft.plan` 把 sweep 推迟到 compaction 之后；低层调用者若携带显式位置则直接报错，不静默丢弃。
- **K=1 有界 overlay**：`MAX_DELTA_DEPTH = 1`。第二个 live delta 进入前先把第一个折叠成新的不可变段，因此 `neighbors()` 永远不需要 K 路归并或 RetractMap。
- **物理重写 ≠ 逻辑变更**：被折叠的段和 delta 打上 `superseded_at = M`，新的段带 `system_ts = M`。所有 `t < M` 仍看到原来的集合，所有 `t >= M` 只看到新段。
- **单发布点（R4）**：WAL 的 `OP_MUTATION` 只是**意图**；delta 文件与 `nodes_<commit>.bin` 在 manifest 写入前都是孤儿。manifest 的原子写入让 delta、节点版本、失效事件三者同时可见，不存在「更正已生效但派生序列静默陈旧」的窗口。崩溃后未发布的 mutation 由 `open()` 计入 `mutations_orphaned` 并**拒绝重放**。
- **节点寿命**：整列定宽快照 `nodes_c<commit>.bin`，按 `system_ts` 选版本；没有行级 `system_from/system_to`。
- **不做（R1/R5/R6）**：`ActiveNodeBitmap` 仍只是可选加速器，`NodeLifetime` 定宽数组 + O(1) 包围盒是事实源；Parent+Diff Manifest、Branch 均推迟。

### 12.4 落盘格式（全部定宽、LE、blake3 trailer）

```
*.gtvd   header[96] | puts[n*42] | removed[m*12] | audit[k*16]
                    | inval[j*48] | checksum[32]
nodes_c*.bin  header[64] | valid_from[i64;N] | valid_to[i64;N] | checksum[32]
wal.log  v2：OP_INSERT(1) 不变；新增 OP_MUTATION(2)，payload = 完整 *.gtvd 编码
```

WAL 头保留 v1 兼容（`WAL_VERSION_V1` 仍可读）；manifest 的 `deltas` / `node_versions` / `next_delta_id` 全部 `#[serde(default)]`，旧仓直接打开且 `src_id_min/max` 用 `u64::MAX/0` 哨兵表示「未知，绝不剪枝」。

### 12.5 SQL 面

- `gtv_invalidations()` / `gtv_invalidations(cursor)` — manifest 提交日志的只读投影，`commit_id >= cursor`；列含 `object_kind`、`reason`、`state`、`file`。`GtvContext::register_segment_store()` 会自动 attach。
- `gtv_segments()` 新增 `src_id_min` / `src_id_max`（旧 manifest 报 NULL）；`register_segment_store()` 跳过 `superseded_at` 段，避免 head 视图重复。
- `SegmentStore::entries_at_src(src, t)` 用紧源点盒在**打开任何 .seg 之前**剪掉无关段。

### 12.6 验收结果与门禁

| 项 | 结果 |
| :-- | :-- |
| 全量 `cargo test --workspace` | **448 passed / 0 failed / 1 ignored**（M0 为 428，本轮 +20） |
| `tests/m1_correction.rs` | 8 项：retract 折叠、AS KNOWN AT 稳定、属性更正、紧源点盒、旧 manifest、§23.6 随机属性、孤儿 WAL、重开与原子失效视图 |
| `tests/m1_invalidation_sql.rs` | 3 项：投影列与游标、自动 attach + 隐藏退役段、未 attach 显式报错 |
| §23.6 属性测试 | 120 组随机序列 × 3 次更正；`Overlay == 朴素重放 == Materialized`，且所有历史快照点在 compaction 前后逐条相等 |
| 生成数据 | 仅 tempdir，`*.csv / *.seg / *.gtvd / nodes_c*.bin` 全部被 `.gitignore` + CI `generated-data` 拦截 |
| 读路径基准 | `bench_endpoint` 所有路径结果一致；行级双时间方案仍为 2.0–2.6x。`csr.rs` 未改，包装开销在本机噪声区间 1.13–1.26x（M0 静默环境实测 1.00–1.06x），非代码回归 |
| CI | `Kernel + temporal graph tests` 新增 M1 两步；`full-tests` 已覆盖 |

### 12.7 M1 当时的边界

M1 交付时 `materialize()` 是同步、显式调用、关闭自动 GC 的开发模式，便于复现每个崩溃点；后台 compaction、延迟 GC、观测表落盘、段内区间窄化、L2 聚合与 `BitemporalStore` 持久化均留给 M2（已在 §13 落地）。Parent+Diff Manifest、Branch、ActiveNodeBitmap 继续留给 M3。

另一个语义边界：GTVSeg v1 只存 `src / dst / valid_from / valid_to / edge_type` 五列，因此 after-image 唯一能变更的属性是 `edge_type`（重分类）。端点 `(src, dst)` 在更正中不可变；更丰富的属性版本化属于 M3 的列扩展，而不是给每行加系统时间。

---

## 13. M2 落地：控制面完备，读路径仍然零回归

> 对应 `asymmetric-bitemporal-review.md` §7 的 M2 行、§8 的七类风险，以及 §6 的两类门禁。

### 13.1 边界与落点

M2 的六项能力**全部落在 `gtv-storage` 控制面与 `gtv-engine` SQL 面**；`crates/gtv-core/src/csr.rs`、`neighbors()` 热循环、GTVSeg v1 的五列格式从 M0 起一行未改。

| 文件 | 作用 |
|:--|:--|
| `gtv-storage/src/observation.rs`（新） | 观测版本的 Arrow IPC 编解码、`ObservationEntry`、按 system 轴选版本 |
| `gtv-storage/src/aggregate.rs`（新） | `LogicalEdge` / `EdgeInterval` / `AggregateView` / `RangeBucket`；区间合并去重 |
| `gtv-storage/src/gc.rs`（新） | `GcPolicy` / `GcReport` / `ReadScope` / 安全 unlink |
| `gtv-storage/src/store.rs` | 两阶段 compaction、GC、观测发布、聚合 API、`live_rows_from()` 重构与源点窄化 |
| `gtv-engine/src/aggregate.rs`（新） | `gtv_active_edge_count()` / `gtv_range_agg()` 两个 UDTF |
| `gtv-engine/src/context.rs` | 注册 UDTF；`register_segment_store()` 作为观测重放与聚合视图的单一同步点 |
| `gtv-storage/examples/bench_correction.rs`（新） | 控制面回归门禁 |
| `tests/m2_control_plane.rs` / `tests/m2_aggregate_sql.rs`（新） | 10 项集成验收 |

### 13.2 两阶段后台 compaction（§8.2 位置空间失效）

`live_rows_at()` 被重构为「取一个 `ReadScope` 票据 → 校验历史下限 → 调用自由函数 `live_rows_from()`」。该函数接受一个纯借用的 `LiveRowsView`（root / node_count / manifest / 已解码 deltas / 未 flush 记录），因此**同步查询零拷贝**，而 worker 只需 clone 一次 `CompactionInput` 就能调用**同一个**折叠函数——不存在第二套「可见」定义。

- **prepare**：`spawn_compaction()` 把 owned input 移进 `std::thread::spawn`，worker 在后台完成 CSR 构造与 GTVSeg 编码，产出 `CompactionPlan`（含已编码段字节、行数、三个 base 游标）。**不写任何文件、不改 manifest**，崩溃/取消零磁盘副作用，也不用给 store 加锁。
- **apply**：`apply_compaction()` 是短临界区，先校验 `system_ts / next_segment_id / checkpoint_txn` 与 prepare 时完全一致，否则直接报 `stale compaction plan`——这正是 §8.2「悬空位置必须报错」的实现；随后写段文件、原子发布 manifest、退休旧段与 Delta。
- 同步 `materialize_at()` 也走 prepare→apply 两阶段，所以 dev 的强制物化与 prod 的 worker 路径不会分叉。

### 13.3 延迟 GC（§8.3 旧快照依赖）

- `GcPolicy::default()` = `retain_from: None` + `dry_run: true`，默认**永不回收**。要真删必须显式给 `retain_from`。
- `keep_last_n` 保护最近 N 个已发布快照点；`cutoff = min(retain_from, 倒数第 N 个点)`。
- `manifest.gc_floor` 记录已回收下限；`live_rows_at()` / `graph_at()` / `node_lifetime_at()` 对 `t < gc_floor` **明确报错**，不静默返回一个不同的图。
- 保留规则不只是「无段引用」：节点寿命快照与每个观测表都要保留 `system_ts <= cutoff` 的**最新一份**，否则 `node_lifetime_at(t)` / `as_of(..., t)` 会退化到更早的错误版本（§8.3）。
- **publish-before-unlink**：先原子写新 manifest 移除所有引用，再删文件；崩溃只留下未被引用的孤儿，下一次 GC 会连同 `.seg / .gtvd / .bin / .gtvobs` 一起清掉。
- `ReadScope` 是共享原子计数：版本查询、聚合视图与 worker 持有票据，`gc()` 发现 `active_readers > 0` 直接拒绝，避免删掉正在 mmap 的文件。

### 13.4 观测表落盘（评审 R3 的实活）

`BitemporalStore` 的语义不用改，缺的是持久化：每个 `(table, system_ts)` 版本写成一个自带 schema 的 Arrow IPC 文件，由 `ObservationEntry` 登记进 **同一份 manifest**，因此图更正与观测修正共享一个原子发布点。`load_observations_into()` 用既有 `record()` 幂等重放，`register_segment_store()` 是显式同步点，`as_of('cpi', 50, 25)` 跨进程重启后仍可用。

一个容易漏的细节：观测版本没有 WAL 帧，所以 commit 游标必须写进 manifest（`next_commit_id`）并在 `open()` 时与 WAL 尾、checkpoint、Delta 的 commit_id 取 max，否则重启后会重复分配 `commit_id`，破坏 `gtv_invalidations(cursor)` 的单调性。

### 13.5 L2 聚合去重（§8.6 去重复口径）

区间更正把一条逻辑边切成 `[A,start) / [start,end) / [end,B)`，直接 `sum(edge_count)` 会数成 3 条。`AggregateView::from_rows()` 在同步时按 `(src,dst,edge_type)` 排序并合并相邻/重叠的半开区间一次；此后：

- `active_edge_count(t)` 因区间互不重叠，一次过滤天然就是 distinct；
- `range_agg(t1,t2,bucket)` 每个桶再用 `HashSet<LogicalEdge>` 去重，一条逻辑边有多个不相连寿命也只算 1；
- SQL 面为 `gtv_active_edge_count(t [, system_t])` 与 `gtv_range_agg(t1,t2,bucket [, system_t])`。可选的 `system_t` 是**守卫**：视图只覆盖一个快照，请求另一个会 plan error，而不是返回陈旧值。

`SegmentStore::active_edge_count_at()` 另外返回 L2 段级 `Skip / FullHit / Scan` 判定（O(1) 每段、不打开段文件）。但**计数仍复用唯一折叠入口**：FullHit 段里可能有被 retract 的行，直接 `edge_count` 相加会双计——正确性优先于那个 O(1) 求和。

### 13.6 段内源点窄化

`live_rows_from()` 用紧 `src_id_min/max` 把 `for s in 0..node_count` 收窄到 `[lo, hi)`，行号从 `offsets[lo]` 起算，保证位置化 `(segment_id, row)` 与未窄化扫描**完全一致**；旧 manifest 缺该字段时 `lo = 0` 保守全扫。测试用「所有边都在 src=5」的段验证 retract 仍然精确命中中间那一行。

### 13.7 验收与门禁

| 门禁 | 结果 |
|:--|:--|
| 全量 `cargo test --workspace` | **466 passed / 0 failed / 1 ignored**，50 suites（M1 为 448，本轮 +18） |
| `cargo check --workspace --all-targets` | 零 warning（rustfmt 组件未安装） |
| 新集成测试 | `m2_control_plane` 7 项 + `m2_aggregate_sql` 3 项；三个新模块单测 8 项 |
| 关键属性 | worker compaction == 同步 == 重开；stale plan 被拒且数据不损；GC 前后 head 逐条相等、旧历史明确报错；孤儿清理；reader guard；区间切分聚合不双计；观测版本跨重启可 `as_of` |
| 读路径基准 | `bench_endpoint` 所有路径结果一致，行级双时间方案仍 2.18–2.42x；`csr.rs` diff 为空 |
| 生成数据 | 仅 tempdir；`.gitignore` 与 CI guard 新增 `*.gtvobs`，tracked 生成数据为 0 |
| CI | `Kernel + temporal graph tests` 新增 M2 测试与 release `bench_correction` 两步 |

`bench_correction`（release，本机 Windows，N=512 / E=4,000 / 20 次发布 / 80 次查询）：

| 指标 | 实测 |
|:--|:--|
| 单次更正发布 | p50 **19.5 ms**（≈50 ops/s，Windows fsync 主导） |
| 强制物化（含一次 O(E) 折叠） | **48.4 ms** |
| 版本查询 P99，depth=0 | **1.10 ms** |
| 版本查询 P99，depth=1 | **2.85 ms（2.60x）** |
| 版本查询 P99，已物化 | **1.70 ms（1.55x）** |

这些是**控制面与文件系统**成本，不进入 `neighbors()`。Linux CI runner（run `42558214981`，同配置）对照：publish p50 **1.75 ms**、forced **3.69 ms**、P99 depth=0 **0.185 ms** / depth=1 **0.210 ms（1.13x）** / materialized **0.172 ms（0.93x）**。同一份代码在 Windows 上的 19.5 ms 与 2.60x 主要来自 fsync 与文件系统，而非 overlay 本身的算法开销——所以跨机器只看「所有路径结果一致」与相对趋势，不看绝对毫秒数；门禁的硬断言是 `Overlay == Materialized`。

### 13.8 M2 交付时的边界（已在 §14 处理）

M2 交付时留给 M3 的是：Parent+Diff Manifest、Branch/Tag、`ActiveNodeBitmap`（planner 按 `E_selected/V` 选）、常驻后台线程与自动 retention、GTVSeg 列扩展。M3 **没有照单全收**：评审 §7 要求「每项必须有基准证明收益，否则不进主干」，因此先做 `bench_m3_gate` 实测，再按结果裁剪，详见 §14。dev 仍默认同步物化、关闭自动 GC，让每个崩溃点可复现。

---

## 14. M3 落地：准入门禁决定范围，读路径仍然零回归

### 14.1 为什么 M3 不等于「把 5 项做完」

`asymmetric-bitemporal-review.md` §7 把 M3 定义为**按需**：「每项必须有基准证明收益，否则不进主干」；R5 判 Parent+Diff「收益未被测量，成本是确定的」，R6 判 Branch「不在 5 个业务需求里，现有线性 manifest 不支持分支」，R1 判 Bitmap 必须降级为可选加速器。

所以 M3 的第一步不是写功能，而是新增 `crates/gtv-storage/examples/bench_m3_gate.rs`：用 release 实测三条曲线，输出显式 go/no-go。实测后再决定实现哪些。**结论是 2 项 GO、1 项 DEFER、1 项只做一半、1 项 NO-GO**：

| 候选项 | 判定 | 依据 |
|:--|:--|:--|
| `ActiveNodeBitmap`（planner 可选） | ✅ GO | `E_selected/V` 实测交叉点 2–4；默认阈值 8 留 2–4x 边际 |
| 常驻维护线程 + 滚动 retention | ✅ GO（opt-in） | 前台只需 0.5 µs 入队，prepare 2.4 ms 移到 worker |
| Parent + Diff Manifest | ⏸ DEFER | 序列化成本到 ~5000 段才越过 2 ms 触发线；当前 <600 段 |
| Branch / Tag | ◐ 只做 Tag | R6：线性 manifest 不能承载分支；Tag 是纯命名引用，成本极低 |
| GTVSeg 列扩展 | ❌ NO-GO | R3：变长属性混进定宽列区会破坏 `disk == memory == cache line` |

本节数值均为 **Windows 本机 release**；M2 已证实同代码在 Linux CI 上的 fsync 成本相差一个数量级（publish 1.75 ms vs 19.5 ms），所以跨机器只看趋势与硬断言，不看绝对毫秒数。

### 14.2 A：`ActiveNodeBitmap` 是派生缓存，不是事实源

`crates/gtv-core/src/active_bitmap.rs` 落地四条约束：

- `BitmapKey = (snapshot_id, business_time, partition_id)`，**snapshot_id 必填**；复用旧快照的位图会让节点修正静默消失（设计 §19.2 / 评审 R1）。
- `NodeLifetime` 仍是唯一事实源。`BitmapCache::invalidate()` 后 `active_at()` 自动回落到事实数组，查询结果逐条不变（§23.4）。
- `BitmapPlanner::decide()` 三重门：全节点活跃 → `AllActive`（走 M0 的 O(1) 包围盒，连位图都不建）；`E_selected < 1024` 或 `< 8·V` → `TooSmall`；超出 `max_cache_bytes` → `OverBudget`。**单点查询永远不付 O(V)**。
- `verify_against()` 逐节点比对事实源，供测试与显式校验使用。

`EndpointView::active_neighbors()` / `csr.rs` / `segment.rs` 一行未改：位图是 planner 可选加速器，不是必选执行路径。

`bench_m3_gate` A 段（V=50,000，30% 节点在 t=500 到期，t=600）：

| E/V | direct µs | bitmap µs | bitmap/direct | planner |
|:--|--:|--:|--:|:--|
| 1 | 58.2 | 82.6 | 1.42 | TooSmall |
| 2 | 116.5 | 119.4 | 1.02 | TooSmall |
| 4 | 232.7 | 195.8 | 0.84 | TooSmall |
| 8 | 467.4 | 348.4 | 0.75 | Build |
| 16 | 1035.2 | 674.5 | 0.65 | Build |
| 64 | 3969.0 | 2457.6 | 0.62 | Build |

交叉点 run-to-run 在 2–4 之间；默认阈值 8 是刻意保守的（基准未计入缓存失效、内存占用与快照切换成本）。

### 14.3 B：Parent+Diff 的收益边界被实测出来了

R5 当时说「收益未被测量」。B 段补上曲线，并明确 **fsync 两边都要付，Diff 只能省掉序列化列**：

| 段数 | manifest 字节 | serialize µs | parse µs | atomic-write µs | ser / 100 段 |
|--:|--:|--:|--:|--:|--:|
| 100 | 55,192 | 52.3 | 93.8 | 5917.6 | 1.0x |
| 1,000 | 549,292 | 587.0 | 560.2 | 6296.3 | 11.2x |
| 5,000 | 2,745,292 | 3869.6 | 3681.2 | 9435.3 | 74.0x |
| 20,000 | 10,980,292 | 17250.5 | 14376.9 | 16648.2 | 329.8x |

4 KB delta `write_atomic` 基线 5.5 ms（Windows fsync 主导，与段数无关），**不是** Parent+Diff 的收益。以「serialize > 2 ms ≈ 一次 M2 Linux CI publish 的量级」为触发线，实测触发点是 **~5000 段**；当前 `metadata-benchmark.md` 的典型 store 是 586 段以下。

判定 **DEFER**：切 Parent+Diff 只能省 serialize，却要新增父链解析、Full Checkpoint 周期策略、checkpoint 前后一致性测试与父链中途损坏的恢复路径（R5 列的确定成本）。**触发条件写死**：当持续段数 > 5000 且每次更正都要重写 manifest 时再评估，且应优先选「Full Checkpoint + 有界增量 diff」而不是任意深度父链。

### 14.4 C：常驻线程只做 prepare，不接管 store

`crates/gtv-storage/src/maintenance.rs` 的关键取舍是**不给 store 套常驻锁**：

```text
前台 publish → submit(CompactionInput)  [0.5 µs 入队，有界 channel=1]
worker       → input.prepare(head+1)    [纯 CPU：不写盘、不改 manifest、不持 store 锁]
前台         → take_plan() → store.apply_compaction(plan)  [短 &mut 临界区，三游标校验]
```

- `apply_compaction()` 原有的 `base_head / next_segment_id / checkpoint_txn` 校验仍然拒绝 stale plan，所以 worker 与前台并发发布不会重写新数据（§8.2 位置空间失效）。
- worker 期间持 `ReadScope`，延迟 GC 不会删掉它正在读的段。
- `gc_policy(head)` 由 `retain_window` 生成滚动 floor；**默认 `None` + `dry_run = true`，全历史不删**，prod 才显式开 `retain_window` 与 `gc_dry_run = false`。策略生成不删任何文件，GC 仍是前台显式调用。
- `stop()` 幂等，`Drop` 自动 join；未消费的 plan 丢弃即可，下次显式 `materialize()` 会重算。

C 段实测：前台 submit **0.5 µs**、worker prepare+wait **2.4 ms**、`apply_compaction` **12.4 ms**、同步 `materialize` **13.1 ms**。所以收益是「prepare 与前台重叠」，**不是免费的 materialize**——apply 的写盘与 manifest 发布仍在前台。文档刻意不把入队耗时说成「phase-1 CPU 全部移出」，避免误读。

### 14.5 D：只做 Tag，不做 Branch

`Manifest.tags: BTreeMap<String, i64>`（`#[serde(default)]`，旧 manifest 直接可加载）。`create_tag / drop_tag / resolve_tag / tags()` 是**单次原子 manifest 写**，且不改 `system_ts / next_segment_id / checkpoint_txn`，因此在途 compaction plan 不会失效。

- **GC 保护**：tag 指向的点加入 `points`（参与 `keep_last_n`），且 `cutoff = cutoff.min(min_tag_ts)`；`GcReport.pinned_by_tags` 让「GC 什么都没删」可解释。测试验证重开后 tag 仍在、`AS KNOWN AT 10` 与 GC 前逐条相等，`drop_tag` 后 pin 才释放。
- **SQL**：`gtv_tags()` 复用 `gtv_invalidations()` 的同一个 manifest-root slot，是提交日志的只读投影，随原子发布立即可见，无事件总线。
- **Branch 不做**：R6 与 `past-update-design.md` §5.1 已判定线性 manifest 不支持分支；`branch_id / sequence_number`、Branch Head、跨分支冲突解决是一整套控制面，且不在 5 个业务需求里。§13.3 要求的「基础修正与派生结果同时可见」由 `WAL/MemSegment` 暂存 + 单次原子发布满足。

### 14.6 E：GTVSeg 列扩展 NO-GO

GTVSeg v1 的定宽列区是 `disk == memory == cache line` 恒等式的基础。R3 已写明：变长 Property after-image 必须走 **独立 sidecar + offset 表**，不能混进列区。M3 没有引入任何 GTVSeg 列改动；更丰富的属性版本化应作为独立 sidecar 设计，触发条件是出现明确的「属性级时间旅行」查询需求，而不是为了对称双时间而扩列。

### 14.7 验收与门禁

| 门禁 | 结果 |
|:--|:--|
| 全量 `cargo test --workspace` | **487 passed / 0 failed / 1 ignored**，52 suites（M2 为 466 / 50，本轮 +21） |
| `cargo check --workspace --all-targets` | 零 warning（rustfmt 组件未安装） |
| 新测试 | `active_bitmap` 7 项 + `maintenance` 3 项单测；`m3_admission` 7 项（含 200 组随机 bitmap 可丢弃性）+ `m3_tags_sql` 4 项 |
| 关键属性 | worker plan == 同步物化 == 重开；stale plan 被拒且历史不变；tag 跨重启、GC pin、旧 manifest 兼容；bitmap 丢弃后逐点一致 |
| 读路径 | `bench_endpoint` 所有路径结果一致，行级双时间方案仍 2.19–2.46x；`csr.rs` / `endpoint.rs` / `segment.rs` diff 为空 |
| 控制面 | `bench_correction` 硬断言 `Overlay == Materialized` 通过；`bench_m3_gate` 输出 A–E 五项判定且 `all paths agree` |
| 生成数据 | 仅 tempdir；无新文件类型，`.gitignore` / CI guard 保持 `*.csv / *.seg / *.gtvd / *.gtvobs / nodes_c*.bin` 拦截 |

### 14.8 剩余项与明确的触发条件

| 未做项 | 何时才做 |
|:--|:--|
| Parent + Diff Manifest | 持续段数 > 5000 且 manifest 序列化 > 2 ms/次发布；优先 Full Checkpoint + 有界 diff |
| Branch / Tag 中的 Branch | 出现「多个分析者同时持有分叉认知」的真实需求，且 L1 与 storage manifest 先统一 |
| day-bucket 缓存 | 仅在 bitmap 缓存命中率有实测数据后；键必须仍含 `snapshot_id` |
| GTVSeg 列扩展 / sidecar | 出现属性级时间旅行查询；变长值走 sidecar，不动定宽列区 |
| 自动 GC 默认开启 | 不开启。dev 永远 dry-run + 全历史，prod 由运维显式给 `retain_window` |

M0–M3 全程遵守同一条边界：**系统时间在 Snapshot、Valid Time 在行；新增复杂度只进控制面与 SQL 面；`neighbors()` 仍是 2 次比较、GTVSeg v1 一列不加。**
