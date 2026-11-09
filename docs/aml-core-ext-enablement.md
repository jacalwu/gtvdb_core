# core 补强 / ext 富化：AML 就绪实施方案

> 状态：Draft for owner review
> 日期：2026-11-09
> 输入：第三方评审《建议的总体边界》（下称「评审」）+ `docs/gtvdb-cpu-cuda-cursor-development-spec.md`（下称「双后端规范」）
> 对账基线：`gtvdb_core` HEAD、`gtvdb_ext` HEAD（core rev `36985c2`）
> 本文不重复评审原文；每一条都以本仓代码事实给出裁定：**同意 / 需修正 / 已有**。

---

## 0. 结论先行

1. 评审的**边界原则正确**（"删除 AML 名词后仍成立 → core"），但它的能力清单里有 4 项与本仓事实不符：不存在统一候选集类型、不存在数据血缘原语、`FullHit` 从未产生零扫描短路、向量元数据其实已存在大半。照抄会重复造轮子。
2. 真正的缺口只有 **5 个**，且它们共享同一个根因：**算子之间没有"候选集"这个值**。core 现在每个算子都吃 `&UInt64Array` 或 `Option<&BooleanArray>`，各自物化、各自重算。补上 `Sel` 之后，评审列的 10 项里有 6 项自动成立。
3. 有一个**零成本高杠杆**改动被评审漏掉：`Neighbor` 有 6 字节对齐填充，塞进 `edge_idx: u32` **不改结构体大小**，却同时解决属性谓词寻址与 AML 证据溯源两件事。这是整个方案的第一杠杆点。
4. `TraversalBudget` 语义是错的：超预算返回 `Err` 并**丢弃已找到的结果**（`gtv-pattern/src/lib.rs:481`），ext 无法区分"没找到"与"没算完"。AML 的 3–5 hop 一定撞预算，这条必须先改。
5. CPU 版与 CUDA 版不是两个产品：**同一 workspace、同一 SQL 语义、两个发行物**。默认构建零 CUDA 依赖；CUDA 只在 `ResidentGraphHandle` 常驻后才有意义（实测 break-even 2.1 次）。双后端规范的阶段顺序需要一处修正（见 §6.3）。

---

## 1. 边界：把"删掉 AML 名词仍成立"变成可执行判定

### 1.1 四问判定（写进 CR 模板）

| # | 问题 | 答案为「是」时 |
|---|---|---|
| Q1 | 删掉 AML/银行/案件名词后，这段逻辑描述的还是一个数据库行为吗？ | 候选进 core |
| Q2 | 它会不会影响 CSR 布局、GTVSeg 格式、快照选择、执行器正确性中的任意一项？ | **必须**进 core，不准在 ext 重实现 |
| Q3 | 换个领域（CRM / Large Exposure / tick / 量子）词表不变还能用吗？ | 倾向 core |
| Q4 | 它需要监管口径、产品代码、阈值、审批或解释文案吗？ | 强制 ext |

反向红线（评审未明说，但本仓踩过）：**core 的 doc 注释里不得出现业务名词**。已经有三处：

- `crates/gtv-pattern/src/lib.rs:109` — "A `Pattern` plus the **AML motif constraints** added by prod_p4 D3"
- `crates/gtv-pattern/src/lib.rs:113` — "rolling window / sequence" 注释本身是通用的，但被 AML 绑定了叙述
- `crates/gtv-engine/src/workload.rs:6` — "interactive **AML**: when the global concurrency budget is full…"

这三处不影响运行，但会持续诱导后来者把领域语义写进 core。**清理成本 0，放在第一个提交里做**（纯注释改写，无代码变更，不影响 parity）。

### 1.2 core 的不可让渡清单

以下 7 项不论谁提需求，都不允许下移到 ext：

1. 快照选择与 `live_rows_at()` 折叠（唯一真值路径）；
2. 段级 `Skip/FullHit/Scan` 判定的**定义**（可以优化，不能换语义）；
3. 时间区间半开语义 `[valid_from, valid_to)`；
4. 候选集跨算子传递的类型与不变量（本文 §3.1）；
5. 遍历预算的计量点与截断语义（§3.3）；
6. 边的物理身份（`EdgeRef`，§3.5）；
7. 加速器生命周期与失效契约（可以 CUDA 实现，不能 ext 实现）。

### 1.3 ext 的必须自有清单

AML Schema、Pattern 模板、风险分、抑制规则、embedding 生成、外部适配器、案件工作流、解释文案、AML benchmark。评审 §三 的 8 条我全部同意，不再复述。

---

## 2. 代码事实核查：对评审 10 项逐条裁定

| # | 评审主张 | 代码事实 | 裁定 |
|---|---|---|---|
| 1 | 需要统一 SnapshotView + 候选集 | 全仓 grep 无 `Sel` / `SelectionBitmap`；KNN 过滤端口是 `Option<&BooleanArray>`（`gtv-index/src/ann.rs:208,255`），k-hop 输入是 `&UInt64Array`（`gtv-core/src/csr.rs:602`），motif 只收 `limit`（`gtv-pattern/src/lib.rs:481`）。三者无共享表示 | **同意，P0** |
| 2 | 需要通用边属性谓词 | `EdgePredicate` trait 已存在（`gtv-core/src/traversal.rs:104`），k-hop 已接线（`csr.rs:608`）；但 `Neighbor` 只有 `dst/edge_type/valid_from/valid_to` 4 字段（`csr.rs:38`），谓词只能看类型和时间 | **同意，但先补寻址（§3.2）** |
| 3 | TraversalBudget 已存在、只差 SQL 出口 | 结构存在（`traversal.rs:25`），但 (a) 超限返回 `Err(BudgetExceeded)` 并丢弃结果，(b) 无 `truncated/completed` 输出，(c) `find_bounded` 在 SQL 层**未接出**（`gtv-engine/src/motif.rs:36` 明确写了） | **同意，但语义要先改（§3.3）** |
| 4 | 需要 Top-K 路径与去重 | motif 的 `Match` 有完整 `nodes + edges`，可重建路径；batch k-hop 只返回 `frontiers`（`traversal.rs:298`），无 parent pointer；无 score、无去重 | **同意，P0** |
| 5 | 向量需要版本与快照语义 | **已存在大半**：`IndexManifest` 已有 `model_id / model_version / embedding_model / feature_version / corpus_snapshot_id / dim / metric / normalized`（`gtv-index-store/src/manifest.rs:24-40`）；`.gtvidx` 已有 `activate/rollback/versions`（`store.rs:221-247`） | **需修正：不是从零建，是补 3 个字段 + 1 条强校验** |
| 6 | 向量 sidecar 与热图解耦 | 已经是分离的：GTVSeg 热数组 5 列 + `.gtvidx` 独立布局 | **已有，不动** |
| 7 | FullHit 零扫描短路未完整 | `PruneOutcome::FullHit` 只是**分类标签**；`active_edge_count_at()` 分类后仍然调 `live_rows_from()` 全量折叠（`gtv-storage/src/store.rs:1671-1698`） | **同意，P1** |
| 8 | 需要 provenance 原语 | `gtv-core/src/provenance.rs` 是**量子态分支溯源**（`ProvenanceNode/ProvenanceEdge/StateProbability`），不是数据血缘；`gtv-engine/src/lineage.rs` 是查询重放血缘；**没有任何东西能把一条边指回 (segment, row)** | **需修正：这是缺口，且比评审说的更严重** |
| 9 | 摄入契约已具备 | `gtv-ingest` 有 envelope/watermark/dedup/dlq/offsets；`gtv-storage` 有 WAL/manifest/delta | **已有，不动** |
| 10 | 加速器抽象应进 core | **`gtv-accelerator` crate 根本不存在**；`cuda-acceleration-plan.md` 是纸面设计 + 一个 C0 探针（`tools/cuda-probe/`） | **同意，但从 0 开始** |

### 2.1 三条必须纠正的判断

**(a) 把向量版本化当"新增能力"会做错。**
评审 §2.5 列的 12 个字段里，8 个已经躺在 `IndexManifest` 里。真正缺的只有三样：`vector_space_id`（跨索引的稳定空间身份）、**每行向量的业务有效区间**（今天只有整个 index 的 `build_ts`，一条向量无法"从 3 月 1 日起有效"）、以及**查询期强校验**（今天 `corpus_snapshot_id` 是 `Option`，没人拦你拿 A 快照的索引去查 B 快照的图）。做前两样是加字段，做第三样是加断言——都不是新索引类型。

**(b) provenance 的瓶颈不在"有没有 lineage 表"，在"边没有物理身份"。**
`Neighbor` 是 `Copy` 的四字段结构，遍历完只剩 `(dst, type, t1, t2)`。两条同源同目标、时间区间相同的边（AML 里很常见：同一对手方两笔等额转账）在结果里**不可区分**。所以评审 §2.8 要的 `source_record_id / row_id` 无法"输出"——它根本没被带出来。这是 §3.2 必须先做的原因。

**(c) `FullHit` 的现状比评审描述更弱。**
评审说"未完整实现"，实际是"**只作为统计计数存在**"。`store.rs:1682` 命中 `FullHit` 后只做 `summary.full_hit += 1`，随后 `:1692` 无条件 `live_rows_from()`。所以今天没有任何一条查询路径因为 `FullHit` 而少读一页。这条要写进 ROADMAP，避免被当成"已交付"。

---

## 3. P0：五项通用原语

### 3.1 `Sel`：算子之间的唯一候选集

**为什么先做这个。** 评审把"统一候选集"列在第 1 位是对的，但它没给出类型。本仓的现实是三个算子三种输入，任何组合都要在 ext 里手工转换一遍——这正是 AML 拼不起来的原因。

```rust
// crates/gtv-core/src/selection.rs（新）
/// 一次查询内有效的候选集。携带 universe 身份，跨快照使用即报错。
pub struct Sel {
    snapshot_id: u64,
    universe: u64,          // 节点数或边数，用于越界检查
    repr: SelRepr,
}

enum SelRepr {
    /// 位图，适合高选择性、需要 O(1) 成员测试
    Bitmap(arrow::buffer::BooleanBuffer),
    /// 升序去重 id，适合低选择性、需要顺序扫描
    Dense(Vec<u64>),
    /// 连续区间并集，适合 ring 候选、时间段切片
    Ranges(Vec<(u64, u64)>),
}

impl Sel {
    pub fn from_ids(snapshot_id: u64, universe: u64, ids: impl IntoIterator<Item = u64>) -> Self;
    pub fn from_mask(snapshot_id: u64, mask: &BooleanArray) -> Self;
    pub fn to_mask(&self) -> BooleanBuffer;      // O(universe)，按需
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_;
    pub fn len(&self) -> usize;
    pub fn intersect(&self, other: &Sel) -> Sel; // 端点不匹配 → Err
    pub fn check(&self, snapshot_id: u64) -> Result<()>;
}
```

**关键不变量（这是护栏，不是优化）**：`Sel` 自带 `snapshot_id`。任何算子收到不匹配的 `Sel` 直接 `GtvError::SnapshotMismatch`，不允许静默截断。AML 回放场景里"拿昨天的 KNN 结果去查今天的图"是最容易出的错，也是最难发现的错。

**接线范围（一次改完，不要分批）**：

| 算子 | 今天 | 改成 |
|---|---|---|
| `TemporalCSR::khop_bounded` | `seeds: &UInt64Array` | `seeds: &Sel` |
| `gtv_index::execute_ann` | `mask: Option<&BooleanArray>` | `filter: Option<&Sel>` |
| `gtv_pattern::find_bounded` | 隐式 `0..node_count` 全扫 | `candidates: &Sel` |
| 聚合入口 | 无过滤端口 | `filter: Option<&Sel>` |

**兼容策略**：旧签名保留为 `#[deprecated]` 的 `*_compat` 包装，内部构造 `Sel::from_ids`，语义等价；一个 minor 周期后删。这样 ext 不必与 core 同一天改。

**代价**：`Sel` 是 40 字节左右的胖指针 + 一个 `Box` 分配；对 N=32 的批量查询，分配开销肉眼可见。对策：提供 `Sel::borrowed` 变体 + 在 `gtv-engine` 侧用 arena 复用。**这条必须进 benchmark，不能凭感觉。**

**验收**：同一查询用 `Sel` 路径与旧路径结果逐位相同；`cargo bench` 显示 N=32 时回归 < 5%，否则保留内联快路径。

---

### 3.2 `Neighbor.edge_idx` + 属性 sidecar（第一杠杆点）

**发现。** `Neighbor` 现有布局在 64 位下是 32 字节：

```text
valid_from: i64  (8)   valid_to: i64  (8)   dst: u64 (8)   edge_type: u16 (2)  + 6 B padding
```

塞入 `edge_idx: u32` 后是 `8 + 8 + 8 + 4 + 2 = 30`，**对齐后仍是 32 字节**。也就是：给每条返回的边加上全局序号，**结构体零膨胀**。实现时必须用断言钉死：

```rust
const _: () = assert!(std::mem::size_of::<Neighbor>() == 32);
```

**`edge_idx` 是什么。** 边在 `TemporalCSR` 排序后数组中的位置（`0..E`）。对给定快照稳定、可变位复用为属性列下标、也可反查回 `(segment_id, row_in_segment)`（需要可选的 `origin` 列，见下）。

**怎么给。** `Neighbors` 迭代器本来就有 `pos`（`csr.rs:873`），现在丢弃了；`khop` 的 push 循环遍历的就是连续区间。两处都是**加一个字段，不加任何计算**。需要注意：`Neighbor` 只有两个构造点——`csr.rs:884` 与 `endpoint.rs:375`，都要同步改；全仓无 `transmute` 访问 `Neighbor`，所以字段位移变化是安全的。

**属性怎么挂。** 不动 `TemporalCSR` 的 5 个热数组，另立 sidecar：

```rust
// crates/gtv-core/src/edge_attr.rs（新）
pub struct EdgeAttrSchema { pub cols: Vec<AttrCol> }   // 名字 + 类型 + 语义
pub struct EdgeAttrs {
    cols: Vec<Col<f64>>,   // 全部定宽；枚举列编码为 f64 或走 u32 变体
    schema: EdgeAttrSchema,
}
impl EdgeAttrs { pub fn get(&self, col: u16, edge_idx: u32) -> f64; }
```

谓词：

```rust
pub struct AttrPred { pub col: u16, pub op: RangeOp, pub lo: f64, pub hi: f64 }
impl EdgePredicate for AttrPred { ... }   // 复用 traversal.rs:104 的既有 trait
```

因为 `EdgePredicate::keep(&Neighbor)` 已有 `edge_idx`，属性访问就是 `attrs.get(col, e.edge_idx)` —— **trait 签名一个字都不用改**。这是 `edge_idx` 的第二重收益。

**顺带解决时间顺序约束。** AML 要的 `t_next - t_previous <= duration`、`t_next >= t_previous` 属于**路径级**约束，不是单边谓词。落点是 motif 的 `EventOrder` 扩展（`gtv-pattern/src/lib.rs:28`）：现在是 `before < after` 硬编码严格小于，改成：

```rust
pub struct EventOrder { pub before: usize, pub after: usize, pub min_gap: i64, pub max_gap: Option<i64> }
```

`min_gap = 0` 即非严格，默认值保持 `min_gap = 1`（等价于原语义）。纯加法，8 个既有测试不动。

**代价与门禁**：`EdgeAttrs` 默认不构建（`TemporalCSR::from_edge_columns` 不变），只在 `Storage::with_edge_attrs(schema)` 时物化。**因此不触碰 ROADMAP §5 的"P0 只测不改 csr.rs"约束**——热数组零改动，只是多了个可选旁路。

**验收**：`size_of::<Neighbor>() == 32` 断言；带属性/不带属性两条路径的 `neighbors()` 输出逐条相同（`edge_idx` 除外）；属性谓词在 100 万边上剪枝率与手写 `filter` 一致。

---

### 3.3 预算语义：从"报错"改成"带残缺的答案"

**这是 ext 目前最痛的一条。** 现在：

```rust
// traversal.rs:233
pub fn add_edges(&self, n: u64) -> Result<()> {
    if total > self.budget.max_edges {
        return Err(GtvError::BudgetExceeded { .. });   // 已扫的结果全丢
    }
```

`find_bounded` 收到 `Err` 就直接 `?` 上抛（`gtv-pattern/src/lib.rs:496,511-512`），`out` 整个丢弃。ext 拿到的是一个错误，**无法回答"是没找到洗钱，还是没算完"**——这在监管场景里是两种完全不同的结论。

**改法**：`Err` 只保留给"参数非法"和"用户取消"。预算耗尽变成**正常返回值**：

```rust
pub struct TraversalOutcome<T> {
    pub rows: T,
    pub stats: TraversalStats,          // 已有（traversal.rs:95）
    pub truncation: Option<Truncation>,
}

pub struct Truncation {
    pub reason: TruncReason,   // Edges | Frontier | RowLimit | Memory | Deadline | Cancelled
    pub hop: usize,            // 在第几跳断的
    pub frontier_at_stop: usize,
    pub would_have_scanned: Option<u64>,  // 只在可估算时给
}
```

三态结论由调用方判定，core 不解释：

| 返回值 | 含义 |
|---|---|
| `rows == 0 && truncation.is_none()` | 真的没有 |
| `rows == n && truncation.is_none()` | 完整 n 条 |
| `truncation.is_some()` | 有结果但**不完整**，不得当作"无匹配"上报 |

**预算字段补齐**（对齐评审，但按本仓命名）：

| 评审字段 | 本仓 | 动作 |
|---|---|---|
| `max_hops` | `max_hops` | ✅ 已有 |
| `max_visited_edges` | `max_edges` | ✅ 已有（改名会破 ABI，保留） |
| `max_frontier_per_hop` | `max_frontier` | ✅ 已有 |
| `max_output_paths` | `max_rows` | ✅ 已有 |
| `max_degree_per_node` | `max_degree` | ✅ 已有（`Option`，语义更强：超限需谓词） |
| `max_elapsed_time` | `deadline: Option<Instant>` | ✅ 已有（绝对时刻比相对时长更适合多算子共享） |
| `cancel_token` | 独立参数 | 保持独立：token 是外部句柄，不该被 clone 进 budget |
| `max_visited_nodes` | — | **补**（今天只有 frontier/edges/rows） |

**验收**：三个新测试——预算正好命中（结果非空 + truncated）、预算差一跳（结果完整 + 无 truncated）、零匹配（空 + 无 truncated），三者的 SQL 输出必须可区分。

---

### 3.4 Top-K 证据路径

AML 调查员要 3 条能自圆其说的证据路径，不要 300 万条重叠环。core 提供排序、去重、重建；**分数怎么算完全由 caller 给**。

```rust
pub struct PathEvidence {
    pub nodes: Vec<u64>,
    pub edges: Vec<u64>,        // edge_idx 序列，与 nodes 等长
    pub score: f64,
}

pub enum DedupKey {
    /// 节点序列，环按最小旋转归一化（v1v2v3 与 v2v3v1 视为同一环）
    NodeSeqCanonical,
    /// 边集合无序哈希（同一组边不同顺序视为重复）
    EdgeSet,
}

pub fn topk_dedup(
    paths: impl IntoIterator<Item = PathEvidence>,
    k: usize,
    key: DedupKey,
) -> Vec<PathEvidence>;
```

排序不变量：`(score desc, nodes lexicographic asc)`——第二个键保证 CPU/GPU 与不同线程数下输出**完全一致**。这条直接抄双后端规范 §2.1 的要求，不是新增。

**路径从哪来**：`khop_bounded` 现在只回 `frontiers`（`traversal.rs:298`），没有 parent。两个选项：

- (A) 加 `parents: Vec<u32>`（每跳一层），内存 `hops × frontier`，可接受；
- (B) 由 caller 用 `edges` 反查。

选 (A) 但**默认关闭**（`KhopOpts::with_parents()`），因为不重建路径的批量 k-hop 是主路径，不该为少数场景付内存。

---

### 3.5 Provenance：把"能查回去"变成契约

三层，缺一不可：

1. **`EdgeRef`**：`{ segment_id: u32, row_in_segment: u32 }`，由 `edge_idx` + 可选 `origin: Col<u32>` 得到（`TemporalCSR::origin_of(edge_idx) -> Option<EdgeRef>`）。`origin` 列 4 B/边，当前每边约 30 B，即 **+13%**，只在开启 provenance 时构建。
2. **查询侧回执**：结果集附带 `QueryReceipt { snapshot_id, system_ts, business_t, vector_space_id: Option<u64>, budget: TraversalStats, truncation: Option<Truncation> }`。一次查询的"取证条件"必须随结果一起走，否则半年后没人能复现。
3. **列级来源**：`source_system / source_record_id / ingest_time` 不进 `TemporalCSR`——那是任意宽度的字符串。它们留在 segment 的属性列里，由 `EdgeRef` 反查。**core 只保证"指得到"，不保证"装得下"。**

这一条把评审 §2.8 的 8 个字段拆成了"定宽身份（core 热路径）+ 变宽载荷（段内属性列）"两部分，避免把字符串塞进 CSR——那正是评审自己反对的。

---

## 4. P1：两项（可延后，但会限制 AML 上限）

### 4.1 窗口/路径聚合 + `FullHit` 真短路

先修事实问题（§2.1c）：`active_edge_count_at` 目前仍全量折叠。真正要做的不是"加一个 GPU 聚合"，而是：

- 段级 `FullHit` 时，直接用段头的 `edge_count` 与 `retract_count` 推算，不打开 `live_rows_from()`；
- 只有在段被 retract 污染时退回折叠（`store.rs:1668` 的注释已经说明了这个理由，逻辑是对的，只是没走快路）。

这条**必须在 CUDA 之前做完**，否则 GPU 聚合加速的是一个本来就该 O(1) 的操作——那是纯负收益。

路径级聚合（`sum/min/max/count` over path）依赖 §3.4 的 `PathEvidence`，属于加法。

### 4.2 向量空间版本化

三件事，按序：

1. `vector_space_id: u64` 提升为一等公民（现在只有 `index name` + `version`）。规则：`vector_space_id` 由 `(model_id, model_version, feature_schema_id, metric, dim)` 哈希得到，**同维度不同模型必然不同 id**——这就是评审要的"不可混查"。
2. 每行向量加 `business_valid_from/to`：允许"模型 3 月升级，但 2 月的向量在回放 2 月交易时仍然有效"。落点：`.gtvidx` 新增可选列（格式已有版本号，走 minor bump）。
3. 查询期强校验：`KnnCollection::check_metric`（`gtv-engine/src/knn.rs:108`）扩展为 `check_binding(snapshot_id, vector_space_id)`，不匹配报错而非回落。

**索引重建期可查旧版本**：`gtv-index-store/src/store.rs:221-247` 已有 `activate/rollback/versions`，把 `activate` 从"改名指针"改成"原子切 current 指针 + 保留 n-1 个版本可查"即可，**不需要新机制**。

---

## 5. ext lib：把孤岛接上

### 5.1 现状：AML 代码已经存在，但与数据库无关

`gtv_ext_lib/gtv-governance/src/aml.rs` 有 679 行，覆盖 `Transaction`、`explain_subgraph`、`BeneficialOwnership`、`hybrid_score`、`FeedbackLedger`、`CaseSnapshot`。**`Cargo.toml` 的依赖只有 `gtv-array`、`gtv-refdata`、`thiserror`；`aml.rs` 的 `use` 只有 `std::collections`。**

也就是说：这是一个纯内存的、与 `TemporalCSR` / motif / KNN / SQL 完全无关的平行实现。它证明了业务语义想清楚了，但**一行 core 原语都没用**。这就是"core 有七成能力、AML 却跑不起来"的真正原因——不是缺算子，是缺接线。

### 5.2 建议布局（对齐评审 §4，按本仓命名）

```text
gtv_ext_lib/
  gtv-ext-aml-schema/     # 节点/边/属性 schema，OWNS/CONTROLS/TRANSFERRED_TO…
  gtv-ext-aml-ingest/     # 银行/KYC/名单 → EdgeRecord；金额币种标准化
  gtv-ext-aml-pattern/    # smurfing / fan-in / fan-out / circular / mule / shared-beneficiary
  gtv-ext-aml-features/   # 账户行为向量、子图特征、历史基线
  gtv-ext-aml-risk/       # 打分、抑制、策略版本
  gtv-ext-aml-case/       # alert/case 引用、证据包（吸收现有 gtv-governance/aml.rs）
  gtv-ext-aml-bench/      # 合成数据 + precision/recall
```

**第一刀不要新建 7 个 crate。** 建议只建 3 个（`pattern` / `features` / `case`），把现有 `aml.rs` 整体搬进 `case`，其余先以模块形式寄生在 `pattern` 里。理由：现在建 7 个空 crate 只会让 CI 变慢，没有任何验证价值。

### 5.3 四个 Pattern → core 原语的映射（这是"只组合"的证明）

| Pattern | core 调用序列 | 需要的 core 能力 |
|---|---|---|
| **fan-in** | `Sel::from(候选账户)` → `khop_bounded(seeds, 1, t, budget)` → 按 dst 分组 count → `topk_dedup` | §3.1 §3.3 §3.4 |
| **fan-out** | 同上，方向反转 | 同上（`DirectionMode` 已有） |
| **circular flow（ring3/4）** | `Sel::from(候选)` → `find_bounded(pattern, t, candidates)` → `PathEvidence` | §3.1 §3.4 |
| **smurfing** | 金额阈值过滤（`AttrPred`）→ `WindowedMotif(ring/path, window=7d, sequence)` → `Sum(amount)` over path | §3.2 §3.4 + EventOrder 扩展 |
| **shared beneficiary** | `BeneficialOwnership`（ext 已有）→ 交点集合 → `Sel::intersect` | §3.1 |

**注意最后一列**：没有一个 Pattern 需要新的 core 算子。凡是觉得"core 得再加一个"的时候，先检查是不是 §3 里某一项没做完。

### 5.4 风险分、抑制、解释

- `gtv-ext-aml-risk`：公式、权重、阈值全部版本化（`policy_version`），**core 不感知**；
- 抑制：`expected_business_score`（工资/清算/集团内部资金池）是负样本特征，属 ext；
- 解释：`explain_subgraph`（`aml.rs:151`）已有雏形，升级为消费 `PathEvidence + QueryReceipt`，输出的句子仍由 ext 拼；
- 案件：`CaseSnapshot`（`aml.rs:465`）保留引用 id，工作流不实现。

### 5.5 Benchmark

沿用 TC1–TC5 已建立的纪律：**生成数据只落 `C:/ai/gtvdb/tmp/`，绝不入仓**；core CI 只跑通用正确性/性能门禁；AML 的 precision/recall/alert-reduction 只在 ext CI 跑。合成数据六类（smurfing 环、fan-in 星、正常工资负样本、清算负样本、高重叠网络、迟到交易）与评审 §三.8 一致。

---

## 6. CPU 版与 CUDA 版怎么真正建立起来

评审说"GPU 只有设备常驻才成立"，双后端规范把这件事落成了 Phase A–G。两者不冲突，但**顺序要改一处**，而且要把 §3.1 插进去当依赖。

### 6.1 两个发行物，一套语义

| | CPU 版 | CUDA 版 |
|---|---|---|
| 构建 | `cargo build --release`（默认） | `cargo build --release --features cuda` |
| 依赖 | 零 CUDA / 零驱动 / 零 Toolkit | `gtv-accelerator-cuda` + PTX（预编译，非 build.rs） |
| 真值 | **reference** | 必须与之逐位/容差一致 |
| 失败行为 | — | 任何失败自动回落 CPU，**不向用户报错** |
| CI | 现有 20 个 step | 独立 job，无 GPU 时只验回落 |

这不是"两个产品"，是同一 workspace 的两个 features。**禁止**为 CUDA 单开分支或 fork。

### 6.2 阶段合并（双后端规范 Phase A–G + 评审 C0–C6 → 一条线）

| 阶段 | 内容 | 依赖 | 状态 |
|---|---|---|---|
| **A0** | 清理 core 里的业务名词注释（§1.1） | — | 未开始，半天 |
| **A1** | `Sel` + `edge_idx` + budget 语义（§3.1–3.3） | A0 | 未开始，**这是真正的 P0** |
| **A2** | `gtv-accelerator`（CPU reference）+ `ExecutionPolicy` 骨架 | A1 | 未开始 |
| **B** | `gtv-accelerator-cuda` 空后端：枚举设备、能力发现、全部 unsupported → 回落 | A2 | 未开始 |
| **C** | 驻留句柄 + LRU + `max_device_bytes` + Tag pin + 6 类失效测试 | B | 未开始 |
| **D** | CUDA flat batch KNN（含 batch 入口——今天 KNN 是单 query，`ann.rs:255`） | C | 未开始 |
| **E** | rolling/window reduce | D | 未开始 |
| **F** | valid-time filter + GPU 链式执行 | C | 未开始 |
| **G** | 大规模 batch k-hop（门槛 `seeds ≥ 1e4` 或 `frontier ≥ 1e6`） | C | **冻结，见 §6.5** |

### 6.3 对双后端规范的一处修正

规范 §8 把 Phase D（KNN）写在 Phase C（驻留）之后——这是对的。但它把 Phase A 定义为"抽象 + adapter"，**没有把 `Sel` 算进去**，于是 Phase D 的 `filtered KNN` 只能继续吃 `Option<&BooleanArray>`，Phase F 的 `bitmap` 又得自己造一套。结果就是规范自己在 §4.1 禁止的"创建第三套选择向量"。

**修正**：把 §3.1 的 `Sel` 并入 Phase A，`Accelerator::knn_flat(.., filter: Option<&SelectionBitmap>)` 直接改成 `Option<&Sel>`；`SelRepr::Bitmap` 天然就是设备侧要传的位图，`to_mask()` 一次转换即可 H2D。这样 Phase F 的 `valid_time_filter` 返回 `Sel` 而不是裸 bitmap，`filter → knn → agg` 三条链路共用同一个值。

### 6.4 CUDA 在 AML 链路里实际能碰三步

实测（RTX 4060 Laptop，C0 探针）：valid-time filter 1e7 边，CPU 6.86 ms vs GPU kernel 0.556 ms（12.3×），但冷路径 13.86 ms —— **传输占 96%，break-even 2.1 次**。所以：

| 链路位置 | 是否上 GPU | 理由 |
|---|---|---|
| 列式条件过滤（金额/币种） | ❌ | 基数小，传输压倒计算 |
| **flat KNN 召回（batch、驻留）** | ✅ | 唯一被实测证明有收益的 |
| **valid-time filter（驻留后）** | ✅ | 与 KNN 共享常驻，无额外传输 |
| **window reduce（链式内）** | ⚠️ | 单独跑无收益，只能作为常驻计划内部算子 |
| batch k-hop | ❌ 冻结 | 门口径未达；且 CPU 已 268 µs @ N=32 |
| motif / 聚合 / 传输 | ❌ | 字符串、小基数、正确性敏感 |

**结论：CUDA 版的第一个有意义的交付不是 kernel，是"驻留 + 失效 + 回落"这套基础设施（Phase C）。** 在没有 Phase C 之前写的任何 kernel，性能数字都是假的。

### 6.5 门禁（沿用双后端规范 §2.3，但加两条本仓约束）

1. 每个 benchmark 必须同时给 CPU / kernel-only / H2D / D2H / cold / resident / p50-p95-p99 / 回落次数；
2. 正确性断言失败**不得输出性能结论**；
3. **新增**：GPU 路径不得改变 `Skip/FullHit/Scan` 语义，也不得在 kernel 里读 system time；
4. **新增**：任何 core 热循环改动（包括 §3.2 的属性旁路）必须能回答"改善的是哪条目标查询"，否则不做（ROADMAP §5 的既有约束）。

---

## 7. 里程碑

| 里程碑 | 交付 | 验收 | 预估 |
|---|---|---|---|
| **M1** | A0 + A1（`Sel` / `edge_idx` / budget 语义） | 新旧路径 parity；`size_of::<Neighbor>()` 断言；三态截断测试 | 3–4 天 |
| **M2** | §3.4 Top-K + §3.5 provenance | 路径去重确定性（4 线程 = 1 线程输出） | 2–3 天 |
| **M3** | ext：`gtv-ext-aml-pattern` 四 Pattern 跑通 | 每个 Pattern 只调 core 原语，无自建图结构 | 4–5 天 |
| **M4** | A2（CPU 加速器抽象） | 默认构建零 CUDA；adapter 开销 < 5% | 3 天 |
| **M5** | B + C（CUDA 空后端 + 驻留） | 6 类失效测试；无 GPU CI 绿 | 5–6 天 |
| **M6** | D（flat batch KNN） | Top-K parity；resident 端到端优于 CPU 才进默认 policy | 4 天 |
| **P1 尾** | §4.1 FullHit 真短路、§4.2 向量空间版本化 | 各自独立门禁 | 待排 |

**顺序上的硬约束**：M1 之前不要动 CUDA（任何 kernel 都会因为 `Sel` 缺失而重写一次）；M3 之前不要建 7 个 AML crate（接口还没定型）。

---

## 8. 明确不做

- 在 core 里出现 `detect_smurfing()` / `smurfing_score` / 监管阈值 / 高风险国家名单；
- 为每个 AML Pattern 加一个 core UDTF（这是评审 §三.2 的红线，也是本仓最容易犯的错）；
- 把金额、币种、渠道写进 `TemporalCSR` 热数组；
- 把 KYC 文档、案件附件放进 GTVSeg；
- 为 AML 造独立查询语言；
- 在 `Sel` 落地前写 CUDA kernel；
- 在没有 GPU 的机器上声称验证了 GPU 路径；
- 把 100 万行合成的 CSV 提交进仓库（TC1–TC5 已建立的纪律）。

---

## 9. 需要拍板的三件事

1. **`Sel` 放 `gtv-core` 还是 `gtv-engine`？** 放 core 则 `gtv-pattern`/`gtv-index` 都能用，但 core 要引入 `arrow::BooleanBuffer`（已依赖 arrow，代价 0）；放 engine 则下层用不了，等于白做。**建议 core。**
2. **`Neighbor.edge_idx` 是破坏性变更吗？** 结构体字节布局会变（32 B 不变，但字段位移变），任何 `unsafe` 按偏移读的地方会坏。本仓 grep 未发现此类代码，但需要你确认 downstream 没有外部消费者。
3. **M1 是否插队到 CUDA 之前？** 若按评审的 C0–C6 直接推进 CUDA，`Sel` 会被迫在 Phase D 补，等于 KNN 端口写两遍。建议 M1 前置。

---

## 10. 与既有文档的关系

| 文档 | 关系 |
|---|---|
| `design-review-2026-11.md` | 本文承接其 §12.7（CUDA 段）与 §13（六问）；§4.2 的双时间修正仍是本文前提 |
| `cuda-acceleration-plan.md` | 本文 §6 把它的 C0–C7 与双后端规范的 Phase A–G 合并；§6.3 是对规范的一处修正 |
| `perf-first-roadmap-review.md` | 本文 §6.5 沿用其"measured-first / 改善哪条目标查询"原则 |
| `gtvdb-cpu-cuda-cursor-development-spec.md` | 本文 §6 是它的执行前提；§6.3 要求把 `Sel` 并入 Phase A |
| `ROADMAP.md` | M1–M6 待业主批准后写入 §3；P1-2/P1-3 冻结状态不变 |
