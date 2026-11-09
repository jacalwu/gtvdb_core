# 最小化时序更新：把 6 个原语收敛成 3 个

> 2026-11-09｜针对 `gtvdb-minimal-temporal-update-design.md` 的业务补充，结合 `past-update-design.md` 与 `gtvdb_core` 当前代码事实。
> 目标：**支持全部 5 个业务场景，且不因为实现复杂而让图遍历变慢。**
> 状态：收敛设计 + 可验证的 M0 原型（`crates/gtv-core/src/endpoint.rs`）。M1/M2 未实现。

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
| M1 | 位置化 retract + WAL v2 + `graph_at` 折叠；`nodes_<S>.bin` 快照文件；`gtv_invalidations()` UDTF | 未做（沿用 `past-update-design.md` U0–U1） |
| M2 | compaction 重写 retract 集、延迟 GC、快照化观测表落盘 | 未做 |

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

### 12.7 明确不做 / 留给 M2

后台 compaction 与延迟 GC、观测表落盘、段内节点区间窄化、L2 `range_agg()` / `active_edge_count()`、`BitemporalStore` 持久化、Parent+Diff Manifest、Branch、ActiveNodeBitmap。当前 `materialize()` 是同步、显式调用、关闭自动 GC 的开发模式，便于复现每个崩溃点。

另一个语义边界：GTVSeg v1 只存 `src / dst / valid_from / valid_to / edge_type` 五列，因此 after-image 唯一能变更的属性是 `edge_type`（重分类）。端点 `(src, dst)` 在更正中不可变；更丰富的属性版本化属于 M2 的列扩展，而不是给每行加系统时间。
