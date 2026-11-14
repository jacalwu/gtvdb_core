# GTVDB 最小化时序更新设计

## 1. 目标与范围

本设计用于让 GTVDB 以尽可能少的核心机制支持以下场景：

1. Node 生命周期到期后，与其连接的部分 Edge 在图遍历中同步失效；
2. Node 的历史 `valid_to` 被追溯修正，例如公司原计划存续至 2029-01-01，后来确认于 2027-01-01 合并终止；
3. 波动率、利率、CPI、杠杆率、仓位、久期、风险暴露等历史观测值被修正；
4. 输入修正后，依赖该输入的派生时间线需要重新计算；
5. 保留“现在认为正确的历史”和“过去当时的认知”，同时避免内核承担业务级联逻辑。

设计目标不是让 GTVDB 自动理解公司合并、贷款转移或风险公式，而是提供一个稳定的时序事实内核，让这些业务规则可以在外部实现。

---

## 2. 核心结论

GTVDB Core 只实现以下六个原语：

```text
PUT VERSION
RETRACT VERSION
QUERY AS OF
PUBLISH SNAPSHOT
EMIT INVALIDATION
COMPACT / GC
```

所有更新统一转换为：

```text
关闭旧系统认知
+
追加新的不可变事实版本
+
发布新 Snapshot
```

核心不实现：

```text
CASCADE UPDATE EDGE
TRANSFER LOAN
MERGE COMPANY
RECALCULATE RISK
PROPAGATE CPI
REBUILD PORTFOLIO
```

这些操作由外部业务处理器或计算任务完成。

---

## 3. 统一双时间模型

每个可版本化事实均包含两组左闭右开时间区间：

```text
Valid Time  = [valid_from, valid_to)
System Time = [system_from, system_to)
```

含义：

- `valid_from / valid_to`：事实在业务世界中何时成立；
- `system_from / system_to`：数据库在何时持有该认知。

一条事实版本在查询 `(business_t, system_t)` 下可见，当且仅当：

```rust
valid_from <= business_t
    && business_t < valid_to
    && system_from <= system_t
    && system_t < system_to
```

已发布 Segment 不允许原地更新。`system_to` 的关闭由 Delta Segment 中的 `RETRACT target_version_id` 表达，Compaction 时才物化为完整区间。

---

## 4. 最小对象模型

GTVDB Core 只持久化三类事实和一种撤销记录：

```rust
pub enum TemporalObject {
    NodeVersion,
    EdgeVersion,
    ObservationVersion,
    Retract,
}
```

### 4.1 NodeVersion

```rust
pub struct NodeVersion {
    pub node_id: u64,
    pub version_id: u128,
    pub label_id: u32,

    pub valid_from: i64,
    pub valid_to: i64,

    pub system_from: i64,
    pub property_ref: u64,
}
```

说明：

- `node_id` 表示稳定的逻辑实体；
- `version_id` 表示不可变事实版本；
- 已写入 PUT 的新版本逻辑上具有 `system_to = +∞`；
- 后续 RETRACT 可将其有效系统区间关闭。

### 4.2 EdgeVersion

```rust
pub struct EdgeVersion {
    pub edge_id: u128,
    pub version_id: u128,

    pub src: u64,
    pub dst: u64,
    pub edge_type: u32,

    pub valid_from: i64,
    pub valid_to: i64,

    pub system_from: i64,
    pub property_ref: u64,
}
```

Edge 保存的是关系自身的固有有效期，不直接复制端点 Node 的生命周期。

### 4.3 ObservationVersion

```rust
pub struct ObservationVersion {
    pub series_id: u128,
    pub version_id: u128,

    pub subject_id: u128,
    pub metric_id: u32,
    pub observation_time: i64,

    pub valid_from: i64,
    pub valid_to: i64,

    pub system_from: i64,
    pub value_ref: u64,

    pub source_id: u32,
    pub quality_flags: u32,
}
```

适用对象：

```text
波动率、利率、CPI、杠杆率、仓位、久期、风险暴露及其他时间序列指标
```

### 4.4 Retract

```rust
pub struct Retract {
    pub target_version_id: u128,
    pub retract_system_ts: i64,
    pub reason: RetractReason,
}
```

```rust
pub enum RetractReason {
    PropertyUpdate,
    LifetimeCorrection,
    ValidTimeCorrection,
    TopologyChange,
    LogicalDelete,
    SourceCorrection,
}
```

RETRACT 精确指向 `version_id`，不得只按 `node_id`、`edge_id` 或 `series_id` 模糊撤销。

---

## 5. Edge 有效期策略

Node 到期后，是否影响 Edge，不能对所有 Edge 使用同一种规则。Edge Type Schema 必须声明端点有效性策略：

```rust
pub enum EndpointValidityPolicy {
    ClampToEndpoints,
    PreserveHistoricalReference,
    External,
}
```

```rust
pub struct EdgeTypeSchema {
    pub edge_type: u32,
    pub endpoint_validity_policy: EndpointValidityPolicy,

    pub require_src_visible_on_insert: bool,
    pub require_dst_visible_on_insert: bool,
}
```

### 5.1 ClampToEndpoints

适用于当前状态关系：

```text
EMPLOYED_BY
CURRENT_POSITION
DEPLOYED_AT
CURRENT_CONTROL
```

Edge 的有效有效期为：

```text
effective_edge_validity
=
edge_intrinsic_validity
∩ source_node_validity
∩ target_node_validity
```

```rust
effective_from = max(edge.valid_from, src.valid_from, dst.valid_from);
effective_to   = min(edge.valid_to,   src.valid_to,   dst.valid_to);
```

查询时间 `t` 满足：

```rust
effective_from <= t && t < effective_to
```

### 5.2 PreserveHistoricalReference

适用于历史事实、合同和审计关系：

```text
LOAN_CONTRACT
HISTORICAL_TRANSACTION
PAYMENT
SIGNED_AGREEMENT
AUDIT_EVENT
MERGED_INTO
```

Edge 按自身固有有效期查询。端点后来失效，不应删除合同或历史事件。

### 5.3 External

适用于需要业务规则解释的关系：

```text
贷款转移
债权承继
公司合并后的责任归属
重组后的担保关系
```

GTVDB Core 不自动缩短、迁移或复制 Edge，只发布影响事件，由外部处理器决定后续 Mutation。

---

## 6. Node 生命周期变更算法

### 6.1 示例

旧认知：

```text
Company A
valid  = [2010-01-01, 2029-01-01)
system = [S1, ∞)
```

系统在 `S2` 确认公司于 `2027-01-01` 合并终止。

写入：

```text
RETRACT Company-A-V1 at S2

PUT Company-A-V2
valid       = [2010-01-01, 2027-01-01)
system_from = S2
```

历史查询仍可得到：

```text
AS KNOWN AT S1 → valid_to = 2029-01-01
AS KNOWN AT S2 → valid_to = 2027-01-01
```

### 6.2 核心算法

```rust
fn correct_node_valid_to(
    old: &NodeVersion,
    new_valid_to: i64,
    commit_ts: i64,
) -> MutationBatch {
    assert!(old.valid_from < new_valid_to);

    MutationBatch {
        retracts: vec![Retract::new(
            old.version_id,
            commit_ts,
            RetractReason::LifetimeCorrection,
        )],
        puts: vec![TemporalPut::Node(
            old.copy_as_new_version()
                .with_valid_to(new_valid_to)
                .with_system_from(commit_ts),
        )],
        invalidations: vec![InvalidationEvent::endpoint_lifetime_changed(
            old.node_id,
            old.version_id,
            old.valid_to,
            new_valid_to,
            commit_ts,
        )],
    }
}
```

本操作不得同步扫描或重写关联 Edge。

---

## 7. Node 到期后 Edge 的处理

假设：

```text
Company A:
[2020-01-01, 2026-01-01)

Bank B:
[2000-01-01, ∞)

Loan Edge:
[2024-01-01, 2029-01-01)
```

对于 `ClampToEndpoints`：

```text
Effective Loan Validity
=
[2024-01-01, 2026-01-01)
```

对于 `PreserveHistoricalReference`：

```text
Intrinsic Loan Validity
=
[2024-01-01, 2029-01-01)
```

因此，贷款 Edge 不应被 GTVDB 自动覆写。公司终止后，贷款是终止、转移给继承公司，还是保留原合同主体，由外部规则决定。

外部处理结果可写成：

```text
旧 Edge:
Company A -> Bank B
valid = [2024-01-01, 2027-01-01)

新 Edge:
Company C -> Bank B
valid = [2027-01-01, 2029-01-01)
```

或者保留原贷款，并新增：

```text
Company A -[MERGED_INTO]-> Company C
valid = [2027-01-01, ∞)
```

---

## 8. Edge 更新算法

### 8.1 属性更新

如果 `src`、`dst`、`edge_type` 不变：

```text
RETRACT old_edge_version
PUT complete new_edge_version
```

Delta Segment 保存完整 after-image，不保存字段 Patch Chain。

### 8.2 拓扑更新

如果修改以下任一字段：

```text
src
dst
edge_type
```

则必须执行：

```text
RETRACT old EdgeVersion
PUT new EdgeVersion
```

默认生成新的 `edge_id`，因为拓扑身份已经改变。

### 8.3 Edge 查询

```rust
fn edge_visible(
    edge: &EdgeVersion,
    src: &NodeVersion,
    dst: &NodeVersion,
    business_t: i64,
    system_t: i64,
    policy: EndpointValidityPolicy,
) -> bool {
    if !edge.visible_at(business_t, system_t) {
        return false;
    }

    match policy {
        EndpointValidityPolicy::ClampToEndpoints => {
            src.visible_at(business_t, system_t)
                && dst.visible_at(business_t, system_t)
        }
        EndpointValidityPolicy::PreserveHistoricalReference => true,
        EndpointValidityPolicy::External => true,
    }
}
```

`External` 表示核心不增加端点约束，不表示外部业务一定会认为该 Edge 有效。

---

## 9. Observation 历史修正

### 9.1 点值修正

原始值：

```text
CPI at 2026-06-01 = 105.2
system = [S1, ∞)
```

在 `S2` 修正为 `104.8`：

```text
RETRACT CPI-V1 at S2

PUT CPI-V2
observation_time = 2026-06-01
value             = 104.8
system_from       = S2
```

基础指标修正只更新对应 ObservationVersion，不在核心内同步重算其他序列。

### 9.2 区间修正

旧事实：

```text
valid = [A, B)
```

修正区间：

```text
[C, D)
```

计算：

```rust
start = max(A, C);
end   = min(B, D);
```

若 `start >= end`，无交集，不产生 Mutation。

否则最多产生：

```text
RETRACT old version
PUT [A, start) old value       // 可选
PUT [start, end) corrected     // 必须
PUT [end, B) old value         // 可选
```

该算法统一用于：

```text
Node 属性历史修正
Edge 属性历史修正
Observation 区间修正
局部逻辑删除
```

---

## 10. 派生时间线重算

GTVDB Core 不保存业务公式依赖 DAG，也不执行派生序列的同步重算。

当 Node、Edge 或 Observation 被修正时，核心发布：

```rust
pub struct InvalidationEvent {
    pub object_kind: ObjectKind,
    pub object_id: u128,

    pub old_version_id: u128,
    pub new_version_id: Option<u128>,

    pub affected_valid_from: i64,
    pub affected_valid_to: i64,

    pub commit_id: u128,
    pub result_snapshot_id: u64,
}
```

外部计算系统负责：

```text
1. 消费 InvalidationEvent；
2. 解析指标依赖关系；
3. 扩大或收窄受影响时间范围；
4. 重算派生时间线；
5. 将结果作为新的 ObservationVersion 批量写回；
6. 原子发布新的 Snapshot。
```

派生值应保存最小 Lineage：

```rust
pub struct DerivationInfo {
    pub computation_id: u128,
    pub algorithm_id: u32,
    pub algorithm_version: u32,

    pub input_snapshot_id: u64,
    pub input_fingerprint: u128,

    pub computed_at_system_ts: i64,
}
```

这样可以判定某个风险暴露、久期或杠杆率是否仍基于旧输入。

---

## 11. Delta Segment 最小内容

Delta Segment 只需要支持两种核心记录：

```text
PUT     完整新版本
RETRACT 精确撤销旧 VersionId
```

逻辑视图：

```text
Snapshot
=
Base Segments
+
Visible Delta PUT
-
Visible RETRACT
```

Delta Segment V1 应包含：

```text
固定 Header
Node PUT Columns
Edge PUT Mini-CSR
Observation PUT Columns
Version-Level Retract
完整 after-image Property / Value
Valid-Time / System-Time Bounds
L2 剪枝元数据
Checksums
```

不包含：

```text
业务级联规则
风险公式
依赖 DAG
字段 Patch Chain
预计算的 effective_edge_valid_to 真相
L3 正确性数据
```

---

## 12. Snapshot 提交流程

```text
1. 读取 base_snapshot；
2. 解析目标旧版本；
3. 验证 Update Conflict；
4. 生成 RETRACT + PUT；
5. 写 WAL；
6. 写不可变 Delta Segment；
7. fsync Delta；
8. 生成 L2 Metadata；
9. 写新 Manifest；
10. 原子发布 Snapshot；
11. 发布 InvalidationEvent；
12. 推进 WAL checkpoint；
13. 后台 Compaction；
14. 延迟 GC。
```

Manifest 是 Segment 可见性的唯一权威来源。

---

## 13. 查询模式

### 13.1 ACTIVE GRAPH

返回 Edge 自身和两个端点在指定双时间下均有效的关系：

```text
Edge ∩ Src Node ∩ Dst Node
```

适用于：

```text
当前图遍历
风险传播
当前组织结构
当前控制关系
```

### 13.2 HISTORICAL GRAPH

按 Edge 固有有效期查询，允许引用后来终止的 Node。

适用于：

```text
历史贷款
合同审计
历史交易
事件追踪
```

### 13.3 AS KNOWN AT

同时指定业务时间和系统认知时间：

```text
BUSINESS TIME 2027-01-01
AS KNOWN AT SYSTEM TIME S1
```

与：

```text
BUSINESS TIME 2027-01-01
AS KNOWN AT SYSTEM TIME S2
```

可以返回不同版本。

---

## 14. 性能策略

### 14.1 查询期交集优先于级联写

Node 生命周期变更时不扫描全部 Edge，避免写放大。端点有效性在查询时计算。

### 14.2 Effective Validity 只能作为缓存

可在 Compaction 或独立 L2 中物化：

```text
cached_effective_from
cached_effective_to
```

但必须携带：

```text
src_version_id
dst_version_id
edge_version_id
```

任一依赖版本变化即失效。缓存缺失时回退到查询期交集。

### 14.3 限制 Delta Fan-In

读取允许：

```text
1 个 Base CSR + 最多 K 个 Delta Mini-CSR
```

超过阈值触发 Compaction，避免无限多路合并。

### 14.4 L3 不参与正确性

Node／Edge／Observation 更新后：

```text
L3 Ready    → 使用统计优化
L3 Missing  → 使用保守成本估计
L3 Failed   → 正常执行
```

修正提交不等待 L3。

---

## 15. 核心不变量

```text
1. 已发布 Segment 不得原地修改。

2. 所有历史修正使用 RETRACT old VersionId + PUT new Version。

3. Node 与 Edge 各自保存固有 Valid Time。

4. Edge 是否受端点生命周期约束，由 EdgeTypeSchema 决定。

5. ClampToEndpoints 的有效期由查询期交集计算，不通过批量改写 Edge 实现。

6. 合同、事件、审计关系不得因端点终止而被自动删除。

7. 修改 Edge 的 src、dst 或 edge_type 必须执行拓扑替换。

8. Observation 修正只更新对应 ObservationVersion。

9. 派生时间线重算由外部任务完成。

10. InvalidationEvent 只描述受影响对象与时间范围，不包含业务决策。

11. Delta PUT 必须保存完整 after-image。

12. RETRACT 必须精确指向 VersionId。

13. 相同 Snapshot、Business Time、System Time 下最多存在一个可见事实版本。

14. Compaction 前后查询结果必须完全一致。

15. 删除全部 L3 文件不得改变查询结果。

16. 旧文件只有在无 Snapshot、Branch、Tag 和 mmap Reader 引用后才能 GC。
```

---

## 16. V1 实施范围

### 必须实现

```text
NodeVersion / EdgeVersion / ObservationVersion
RETRACT + PUT
双时间可见性
EdgeType EndpointValidityPolicy
Valid-Time 区间切分
Delta Segment
Copy-on-Write Manifest
Snapshot 原子发布
InvalidationEvent
后台 Compaction
延迟 GC
```

### 暂不实现

```text
同步 Cascade Update
自动贷款转移
自动公司合并规则
内核指标依赖 DAG
内核风险模型重算
自动 Branch Merge
字段级 Patch Chain
预存 traversal cost
段级 SCC / Reachability
```

---

## 17. 验收测试

### 17.1 Node 到期

- Node `valid_to` 缩短后，`ClampToEndpoints` Edge 在新截止时间后不可见；
- `PreserveHistoricalReference` Edge 的固有有效期不变；
- 不产生全量 Edge 重写。

### 17.2 双时间修正

- 在 `S1` 查询得到旧 `valid_to`；
- 在 `S2` 查询得到修正后的 `valid_to`；
- 旧 Snapshot 仍能重现旧认知。

### 17.3 Observation 修正

- 修正点前后的系统时间查询返回不同值；
- 只影响指定 ObservationVersion；
- 正确输出 `InvalidationEvent` 的时间范围。

### 17.4 区间切分

覆盖：

```text
无交集
完全覆盖
左侧交集
右侧交集
中间切分
空区间
valid_to = i64::MAX
```

### 17.5 Compaction

在随机 Node、Edge、Observation、RETRACT 集合上验证：

```text
Compaction 前结果 == Compaction 后结果
```

比较维度包括：

```text
Snapshot
Business Time
System Time
ACTIVE GRAPH / HISTORICAL GRAPH
```

---

## 18. 一句话交付

GTVDB 用最简洁方式支持复杂时序更新的关键，不是实现业务级联，而是将核心收敛为：

```text
不可变双时间版本
+
RETRACT / PUT
+
Edge 端点有效性策略
+
Observation 修正
+
Invalidation Event
+
外部派生重算
```

Node 生命周期对 Edge 的影响通过查询期有效期交集表达；合同与历史关系保留自身固有时间；公司合并、贷款承继和风险时间线重算由外部系统决定并将结果作为新版本写回。这样可在不扩大 GTVDB Core 责任范围的情况下，同时支持当前图、历史图、审计查询和历史数据修正。
