# GTVDB 非对称双时间更新与 Snapshot Delta 设计

## 0. 结论先行

GTVDB 不采用通用双时间数据库的行级双时间判定，而采用与不可变 CSR、Manifest 和 Snapshot 相匹配的非对称双时间模型：

```text
System Time = Snapshot / Manifest Version
Valid Time  = Node / Edge / Observation 的行级区间
```

查询过程固定为：

```text
AS KNOWN AT system_t
    ↓
解析 system_t 对应的 Snapshot，一次
    ↓
取得该 Snapshot 的最终可见 Segment 集合
    ↓
按 business_t 执行 Valid-Time 查询
```

因此，Node、Edge、Observation 的查询热记录不保存 `system_from/system_to`，邻接遍历热循环也不执行行级 System-Time 判断。

GTVDB Core 只实现以下原语：

```text
PUT VERSION
REMOVE VERSION
RESOLVE SNAPSHOT
QUERY VALID TIME
PUBLISH SNAPSHOT
EMIT INVALIDATION
MATERIALIZE / COMPACT / GC
```

所有历史修正统一转换为：

```text
基于一个 Base Snapshot
-
移除旧 Version
+
加入新 Version
+
原子发布新 Snapshot
```

---

## 1. 设计目标

本设计以最简洁的核心支持：

1. Node 生命周期到期后，部分 Edge 在活动图中随端点失效；
2. Node 的历史 `valid_to` 被追溯修正；
3. 合同、贷款、交易等 Edge 保留自身固有业务期限；
4. 波动率、利率、CPI、杠杆率、仓位、久期、风险暴露等历史观测值被修正；
5. 输入修正后，外部系统可以重建派生时间线；
6. 可以查询当前认知，也可以恢复过去某个 System-Time Snapshot 的认知；
7. 避免在 CSR 扫描中加入行级 MVCC、RetractMap 和 System-Time 判断。

GTVDB Core 不理解或执行：

```text
公司合并后的贷款承继
债权转移
风险公式
指标依赖 DAG
组合重算
业务级 Cascade Update
```

这些规则由外部处理器完成，并将结果作为新的不可变版本写回。

---

## 2. 为什么不采用行级双时间

通用行级双时间模型通常让每条记录具有：

```text
valid_from
valid_to
system_from
system_to
```

对应热路径判定：

```rust
valid_from <= business_t
    && business_t < valid_to
    && system_from <= system_t
    && system_t < system_to
```

这会让每次邻接扫描增加：

```text
System-Time 区间判断
版本竞争解析
Retract 查找
Delta 多路合并
```

而 GTVDB 已有：

```text
不可变 Segment
Snapshot
Manifest
Copy-on-Write 发布
```

System Time 已经可以由整表版本提供。重复放入每条 Node、Edge、Observation，会制造第二套版本机制，并破坏 CSR/mmap 热路径。

因此 GTVDB 采用：

```text
查询级 System-Time 选择
+
记录级 Valid-Time 过滤
```

---

## 3. 非对称双时间模型

### 3.1 System Time

System Time 由 Snapshot lineage 表达。

```rust
pub struct Snapshot {
    pub snapshot_id: u64,
    pub parent_snapshot_id: Option<u64>,

    pub branch_id: u64,
    pub sequence_number: u64,
    pub committed_at: i64,

    pub manifest_id: u64,
    pub operation: SnapshotOperation,
    pub materialization_state: MaterializationState,
}
```

Snapshot 回答：

```text
在 system time S，当时数据库知道什么？
```

解析规则：

```text
选择目标 Branch 中 committed_at <= system_t 的最新 Snapshot
```

若用户直接指定 `snapshot_id`，则无需进行时间查找。

### 3.2 Valid Time

Node、Edge、Observation 只保存业务有效区间：

```text
[valid_from, valid_to)
```

记录在业务时间 `t` 可见，当且仅当：

```rust
valid_from <= t && t < valid_to
```

### 3.3 完整查询语义

```text
QUERY (system_t, business_t)
=
ResolveSnapshot(system_t)
+
QueryValidTime(snapshot, business_t)
```

System Time 每个查询解析一次，Valid Time 才进入记录和边的扫描路径。

---

## 4. 最小事实模型

GTVDB Core 只保存三类事实版本：

```rust
pub enum TemporalObject {
    NodeVersion,
    EdgeVersion,
    ObservationVersion,
}
```

Version 是不可变的。版本属于哪个 System-Time 认知状态，由 Snapshot 是否引用它决定。

### 4.1 NodeVersion

```rust
pub struct NodeVersion {
    pub node_id: u64,
    pub version_id: u128,
    pub label_id: u32,

    pub valid_from: i64,
    pub valid_to: i64,

    pub property_ref: u64,
}
```

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

    pub property_ref: u64,
}
```

Edge 保存关系自身的固有业务有效期，不复制端点 Node 的生命周期。

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

    pub value_ref: u64,
    pub source_id: u32,
    pub quality_flags: u32,
}
```

适用：

```text
波动率
利率
CPI
杠杆率
仓位
久期
风险暴露
其他时序观测值或派生指标
```

### 4.4 VersionId 的职责

`version_id` 只用于：

```text
Snapshot Delta 精确替换
审计与 Lineage
冲突检测
依赖指纹
缓存失效
```

不得用于邻接热循环中的行级 System-Time 判断。

---

## 5. Edge 端点有效性策略

Node 到期后是否影响 Edge，必须由 Edge Type 明确声明。

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

有效有效期：

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

### 5.2 PreserveHistoricalReference

适用于合同、交易、付款和审计事实：

```text
LOAN_CONTRACT
HISTORICAL_TRANSACTION
PAYMENT
SIGNED_AGREEMENT
AUDIT_EVENT
MERGED_INTO
```

Edge 只按自身固有有效期查询。端点后来终止，不会自动删除历史关系。

### 5.3 External

适用于需要业务规则解释的关系：

```text
贷款转移
债权承继
公司合并后的责任归属
重组后的担保关系
```

GTVDB Core 不自动缩短、迁移或复制 Edge，只发布变更与失效事件，由外部处理器提交后续 Snapshot Delta。

---

## 6. ActiveNodeBitmap

对 `ClampToEndpoints` 的 Edge，不能在每条边上随机读取两个 NodeVersion。

建议对指定 Snapshot 和 Business Time 构建：

```rust
pub struct ActiveNodeBitmap {
    pub snapshot_id: u64,
    pub business_time: i64,
    pub node_partition_id: u64,
    pub bitmap: Bitmap,
}
```

Edge 热循环：

```rust
fn active_edge(
    edge: &EdgeVersion,
    active_nodes: &ActiveNodeBitmap,
    business_t: i64,
) -> bool {
    edge.valid_from <= business_t
        && business_t < edge.valid_to
        && active_nodes.contains(edge.src)
        && active_nodes.contains(edge.dst)
}
```

`PreserveHistoricalReference` 不检查 Bitmap：

```rust
edge.valid_from <= business_t && business_t < edge.valid_to
```

`ActiveNodeBitmap` 是派生缓存，不是事实源：

```text
缓存缺失
→ 扫描 Node Valid-Time 区间
→ 构建 Bitmap
→ 执行查询
```

缓存键必须包含 `snapshot_id`，否则 Node 生命周期修正后会复用错误结果。

如果业务时间天然按日查询，可增加 day-bucket 缓存，但只有在能保证日内无有效期边界时，缓存结果才可直接作为最终判断；否则它只能作为候选集。

---

## 7. Snapshot Delta

### 7.1 定位

Snapshot Delta 表达从一个 Snapshot 到另一个 Snapshot 的集合变化：

```text
Snapshot N+1
=
Snapshot N
-
Removed Versions
+
Added Versions
```

它不是永久的查询时 MVCC 层，也不是 WAL。

### 7.2 数据模型

```rust
pub struct SnapshotDelta {
    pub delta_id: u128,

    pub base_snapshot_id: u64,
    pub result_snapshot_id: u64,

    pub added_node_versions: Vec<NodeVersion>,
    pub added_edge_versions: Vec<EdgeVersion>,
    pub added_observations: Vec<ObservationVersion>,

    pub removed_version_ids: Vec<u128>,
}
```

### 7.3 REMOVE VERSION

原 `RETRACT` 重命名为 `REMOVE VERSION`，避免误解为行级 `system_to`。

```rust
pub struct RemoveVersion {
    pub target_version_id: u128,
    pub reason: RemoveReason,
}
```

```rust
pub enum RemoveReason {
    PropertyUpdate,
    LifetimeCorrection,
    ValidTimeCorrection,
    TopologyChange,
    LogicalDelete,
    SourceCorrection,
}
```

语义：

```text
目标 Version 不属于 result_snapshot_id 的最终可见集合
```

`REMOVE VERSION` 不回写旧记录，不要求历史 Snapshot 改变，也不要求物化 Snapshot 的 Reader 构建 RetractMap。

### 7.4 Delta 与 WAL 的区别

```text
WAL
负责崩溃恢复和未完成提交恢复

Snapshot Delta
负责表达已提交 Snapshot 的集合差异
```

Snapshot 发布后，Reader 不应依赖 WAL。

---

## 8. Snapshot Materialization

### 8.1 MaterializationState

```rust
pub enum MaterializationState {
    DeltaOnly,
    Materializing,
    Materialized,
    Failed,
}
```

### 8.2 Materialized 快速路径

```text
Snapshot
→ 已解析的最终 Manifest
→ Base / Materialized CSR
→ L2 Segment Pruning
→ Valid-Time Filter
```

不得执行：

```text
RetractMap
行级 System-Time Filter
无界 Delta Replay
版本竞争解析
```

### 8.3 有界 Delta Overlay

新 Snapshot 尚未 Materialize 时，可以临时使用：

```text
Base Snapshot + 少量 Snapshot Delta
```

必须限制：

```text
delta_depth <= K
```

K 是配置项，并由基准测试决定。设计上不允许无限父链参与邻接扫描。

超过阈值时：

```text
必须先 Materialize / Compact
```

### 8.4 Head 发布策略

V1 推荐：

```text
允许 main Head 带一个 Snapshot Delta
后台立即 Materialize
```

更保守模式：

```text
只有 Materialized Snapshot 才能成为 main Head
```

后者查询最简单，但 Update 可见延迟更高。

---

## 9. Manifest 策略

Manifest 是 Snapshot 可见性的唯一权威来源。

### 9.1 Full Manifest

每个 Snapshot 保存完整可见文件集合。

优点：

```text
打开快
查询简单
```

缺点：

```text
Manifest 较大
```

### 9.2 Parent + Diff

每个 Snapshot 保存：

```text
parent
add
remove
```

优点：

```text
提交小
```

缺点：

```text
历史 Snapshot 打开需要解析父链
```

### 9.3 推荐折中

```text
提交写 Parent + Diff
周期性生成 Full Manifest Checkpoint
内存缓存解析后的完整 Manifest
```

无论持久格式如何，查询开始后必须得到一个最终可见 Segment 集合。Manifest 差异解析不得进入 CSR 邻接热循环。

---

## 10. Node 生命周期修正

### 10.1 示例

Snapshot 100：

```text
Company A V1
valid = [2010-01-01, 2029-01-01)
```

后续确认公司于 2027-01-01 合并终止。

Snapshot Delta：

```text
REMOVE Company-A-V1
ADD    Company-A-V2
       valid = [2010-01-01, 2027-01-01)
```

发布 Snapshot 101：

```text
Snapshot 100
→ Company A valid_to = 2029-01-01

Snapshot 101
→ Company A valid_to = 2027-01-01
```

查询语义：

```text
AS KNOWN AT Snapshot 100
→ 旧认知

AS KNOWN AT Snapshot 101
→ 修正认知
```

不需要行级 `system_from/system_to`。

### 10.2 核心算法

```rust
fn correct_node_valid_to(
    base_snapshot_id: u64,
    old: &NodeVersion,
    new_valid_to: i64,
) -> SnapshotDelta {
    assert!(old.valid_from < new_valid_to);

    SnapshotDelta {
        base_snapshot_id,
        removed_version_ids: vec![old.version_id],
        added_node_versions: vec![
            old.copy_as_new_version()
                .with_valid_to(new_valid_to)
        ],
        ..SnapshotDelta::empty()
    }
}
```

不得同步扫描或重写所有关联 Edge。

---

## 11. Node 到期后 Edge 的处理

示例：

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
Effective Validity
=
[2024-01-01, 2026-01-01)
```

对于 `PreserveHistoricalReference`：

```text
Intrinsic Validity
=
[2024-01-01, 2029-01-01)
```

因此 Company 的修正不会强制覆写 Loan Edge。

外部业务处理器可以选择：

### 11.1 贷款终止

```text
REMOVE old Loan Version
ADD Loan Version valid_to = 2027-01-01
```

### 11.2 贷款转移

```text
旧 Edge:
Company A -> Bank B
valid = [2024-01-01, 2027-01-01)

新 Edge:
Company C -> Bank B
valid = [2027-01-01, 2029-01-01)
```

### 11.3 保留原合同并增加承继关系

```text
Company A -[MERGED_INTO]-> Company C
valid = [2027-01-01, ∞)
```

GTVDB Core 不替业务方选择其中一种。

---

## 12. Edge 更新

### 12.1 属性更新

如果 `src`、`dst`、`edge_type` 不变：

```text
REMOVE old EdgeVersion
ADD complete new EdgeVersion
```

Delta 保存完整 after-image，不保存字段 Patch Chain。

### 12.2 拓扑更新

如果修改：

```text
src
dst
edge_type
```

则执行：

```text
REMOVE old EdgeVersion
ADD new EdgeVersion
```

默认生成新的 `edge_id`，因为拓扑身份已经改变。

### 12.3 Valid-Time 区间修正

旧区间：

```text
[A, B)
```

修正区间：

```text
[C, D)
```

```rust
start = max(A, C);
end   = min(B, D);
```

若 `start >= end`，无交集。

否则最多生成：

```text
REMOVE old version
ADD [A, start) old payload
ADD [start, end) corrected payload
ADD [end, B) old payload
```

空区间不写入。

---

## 13. Observation 历史修正

### 13.1 点值修正

Snapshot 200：

```text
CPI 2026-06-01 = 105.2
Version = CPI-V1
```

修正：

```text
REMOVE CPI-V1
ADD CPI-V2 = 104.8
```

发布 Snapshot 201：

```text
Snapshot 200 → 105.2
Snapshot 201 → 104.8
```

不增加行级 system-time 字段。

### 13.2 派生时间线重算

基础输入修正后，GTVDB 发布：

```rust
pub struct InvalidationEvent {
    pub object_kind: ObjectKind,
    pub object_id: u128,

    pub old_version_id: u128,
    pub new_version_id: Option<u128>,

    pub affected_valid_from: i64,
    pub affected_valid_to: i64,

    pub base_snapshot_id: u64,
    pub result_snapshot_id: u64,
    pub commit_id: u128,
}
```

外部处理器负责：

```text
消费 InvalidationEvent
→ 解析依赖 DAG
→ 确定重算区间
→ 重算派生指标
→ 写入 ObservationVersion
→ 发布后续 Snapshot
```

派生结果保存最小 Lineage：

```rust
pub struct DerivationInfo {
    pub computation_id: u128,
    pub algorithm_id: u32,
    pub algorithm_version: u32,

    pub input_snapshot_id: u64,
    pub input_fingerprint: u128,

    pub computed_at: i64,
}
```

### 13.3 原子业务发布

如果基础修正与派生结果必须同时可见：

```text
1. 从 main Snapshot 创建临时 Branch；
2. 在 Branch 写入基础修正；
3. 外部系统完成派生重算；
4. Materialize Branch Head；
5. 将完整结果一次发布为新的 main Snapshot。
```

---

## 14. Delta Segment 文件职责

Delta Segment 保存 Snapshot Delta 的物理内容，而不是行级 MVCC 历史。

V1 内容：

```text
固定 Header
Added NodeVersion Columns
Added EdgeVersion Mini-CSR
Added ObservationVersion Columns
Removed VersionId Columns
完整 Property / Value After-Image
Valid-Time Bounds
L2 Metadata
Checksums
```

不包含：

```text
system_from / system_to
retract_system_ts
行级 RetractMap
业务 Cascade 规则
风险公式
依赖 DAG
预存 effective_edge_valid_to 事实
L3 正确性数据
```

### 14.1 文件身份

Header 至少包含：

```rust
pub struct DeltaHeader {
    pub format_version: u32,
    pub delta_id: u128,

    pub base_snapshot_id: u64,
    pub result_snapshot_id: u64,

    pub branch_id: u64,
    pub commit_sequence: u64,

    pub node_put_count: u64,
    pub edge_put_count: u64,
    pub observation_put_count: u64,
    pub removed_version_count: u64,

    pub valid_from_min: i64,
    pub valid_to_max: i64,

    pub data_checksum: [u8; 32],
}
```

### 14.2 Edge Mini-CSR

Added EdgeVersion 按：

```text
(src, dst, edge_type, edge_id, valid_from)
```

排序并形成 Mini-CSR，以支持新 Snapshot 尚未 Materialize 时的有界 Overlay。

---

## 15. 提交与发布流程

```text
1. 读取 base_snapshot；
2. 解析目标旧 Version；
3. 验证 Head 与 base_snapshot 是否一致；
4. 生成 Added Versions 与 Removed VersionIds；
5. 写 WAL；
6. 写不可变 Delta Segment；
7. fsync Delta；
8. 写 Parent + Diff Manifest；
9. 原子发布新 Snapshot；
10. 更新 main / branch ref；
11. 发布 InvalidationEvent；
12. 推进 WAL checkpoint；
13. 后台 Materialize / Compact；
14. 延迟 GC。
```

Manifest 发布前失败：

```text
旧 Snapshot 继续可用
新 Delta 为 Orphan
```

Manifest 发布后失败：

```text
新 Snapshot 可通过有界 Overlay 读取
后台继续 Materialize
```

---

## 16. 查询执行

### 16.1 当前 Head

```text
1. 取得 Head Snapshot；
2. 取得已解析 Manifest；
3. L2 选择 Segment；
4. CSR 定位 src；
5. 执行 Valid-Time 判断；
6. ClampToEndpoints 时执行两次 ActiveNodeBitmap 检查。
```

System Time 判断次数：

```text
每个查询一次
```

### 16.2 历史 System-Time 查询

```text
1. 根据 system_t 解析历史 Snapshot；
2. 取得该 Snapshot 的最终 Manifest；
3. 运行与 Head 相同的 Valid-Time 查询。
```

### 16.3 Branch 查询

```text
1. 解析 Branch Head 或 Branch 历史 Snapshot；
2. 取得独立 Manifest；
3. 执行普通 Valid-Time 查询。
```

BranchId 不进入 Node、Edge 或 Observation 热记录。

---

## 17. 查询模式

### 17.1 ACTIVE GRAPH

```text
Edge Valid Time
∩ Src Node Valid Time
∩ Dst Node Valid Time
```

适用于：

```text
当前图遍历
风险传播
当前组织关系
当前控制关系
```

### 17.2 HISTORICAL GRAPH

按 Edge 固有 Valid Time 查询，允许引用后来终止的 Node。

适用于：

```text
历史贷款
合同审计
历史交易
事件追踪
```

### 17.3 AS KNOWN AT

```text
BUSINESS TIME T
AS KNOWN AT SYSTEM TIME S
```

执行为：

```text
ResolveSnapshot(S)
→ QueryValidTime(T)
```

---

## 18. 性能模型

### 18.1 行级双时间模型

```text
O(E_selected × (
    valid-time check
    + system-time check
    + retract lookup
    + version resolution
))
```

### 18.2 Snapshot System-Time 模型

```text
O(snapshot_resolution)
+
O(E_selected × valid-time check)
```

对 `ClampToEndpoints`：

```text
O(snapshot_resolution)
+
O(active-node bitmap lookup/build)
+
O(E_selected × (
    valid-time check
    + 2 bitmap checks
))
```

### 18.3 Delta Fan-In

只允许：

```text
1 个 Materialized Base
+
最多 K 个 Snapshot Delta Mini-CSR
```

超过阈值必须 Materialize。

---

## 19. L2 与 L3

### 19.1 L2

每个 Base 或 Delta Segment 必须保存：

```text
valid_from_min
valid_to_max
src_id_min / max
dst_id_min / max
node / edge / observation count
checksum
```

System-Time 剪枝由 Snapshot Manifest 完成，不需要放入行级列。

### 19.2 L3

L3 仍为可丢弃派生缓存：

```text
L3 Ready   → 使用统计优化
L3 Missing → 使用保守成本估计
L3 Failed  → 正常执行
```

Snapshot 发布不等待 L3。

ActiveNodeBitmap 可以作为独立派生缓存管理，但它参与 `ClampToEndpoints` 的快速执行时必须可从 Node Valid-Time 事实重建。

---

## 20. GC 与保留

旧 Version 和旧 Segment 只有在以下条件全部满足时才能回收：

```text
无 Snapshot 引用
无 Branch 引用
无 Tag 引用
无 mmap Reader 引用
超过 Retention Policy
```

删除旧 Snapshot 会降低历史 System-Time 可查询范围，因此属于显式保留策略，不得由普通 Compaction 隐式完成。

---

## 21. 核心不变量

```text
1. System Time 由 Snapshot lineage 提供，不存入热记录。

2. 查询先解析唯一 Snapshot，再执行 Valid-Time 查询。

3. Snapshot 解析不得进入邻接迭代热循环。

4. 已发布 Segment 不得原地修改。

5. Snapshot Delta 使用 Added Versions + Removed VersionIds。

6. REMOVE VERSION 只改变新 Snapshot 的集合，不回写旧 Snapshot。

7. Materialized Snapshot 查询不得依赖 Delta Replay 或 RetractMap。

8. 未 Materialize Snapshot 只允许有界 Delta Overlay。

9. Delta 深度超过阈值必须 Materialize。

10. Node 与 Edge 各自保存固有 Valid Time。

11. Edge 的端点约束由 EndpointValidityPolicy 决定。

12. ClampToEndpoints 优先通过 ActiveNodeBitmap 执行。

13. 合同、事件和审计关系不得因端点终止而自动删除。

14. Edge 的 src、dst 或 edge_type 变化必须执行拓扑替换。

15. Observation 修正创建新 Snapshot，不增加行级 System-Time 字段。

16. 派生时间线由外部任务重算。

17. Delta PUT 必须保存完整 after-image。

18. Manifest 是 Segment 可见性的唯一权威来源。

19. Compaction / Materialization 前后，同一 Snapshot 和 Business Time 的结果必须一致。

20. 删除全部 L3 文件不得改变查询结果。
```

---

## 22. V1 实施范围

### 22.1 必须实现

```text
Snapshot System-Time Resolution
NodeVersion / EdgeVersion / ObservationVersion
Valid-Time 查询
SnapshotDelta
REMOVE VERSION + ADD VERSION
Parent + Diff Manifest
Full Manifest Checkpoint
有界 Delta Overlay
Materialization
EdgeType EndpointValidityPolicy
ActiveNodeBitmap
Valid-Time 区间切分
InvalidationEvent
后台 Compaction
延迟 GC
```

### 22.2 暂不实现

```text
行级 system_from / system_to
永久 RetractMap
无界 Delta Chain
同步 Cascade Update
自动贷款转移
自动公司合并规则
内核指标依赖 DAG
内核风险模型重算
自动 Branch Merge
字段级 Patch Chain
段级 SCC / Reachability
```

---

## 23. 验收测试

### 23.1 System Time 只由 Snapshot 提供

- 同一个 Version 不含 `system_from/system_to`；
- Snapshot 100 和 101 可返回不同认知；
- CSR 热循环没有 System-Time 分支；
- Materialized Snapshot 查询不构建 RetractMap。

### 23.2 Node 生命周期修正

- Snapshot 100 返回旧 `valid_to`；
- Snapshot 101 返回新 `valid_to`；
- 旧 Snapshot 不被修改；
- 不产生全量 Edge 重写。

### 23.3 Edge 端点策略

- `ClampToEndpoints` 在 Node 到期后不可见；
- `PreserveHistoricalReference` 保留 Edge 固有期限；
- `External` 不执行内核级联。

### 23.4 ActiveNodeBitmap

- Bitmap 结果与直接扫描 Node Valid-Time 完全一致；
- SnapshotId 不同不得共享缓存；
- 缓存删除后查询结果不变；
- 时间边界严格遵守 `[from, to)`。

### 23.5 Observation 修正

- Snapshot 200 返回旧值；
- Snapshot 201 返回修正值；
- 正确产生 InvalidationEvent；
- 派生结果写回后由后续 Snapshot 可见。

### 23.6 Materialization

随机生成 Base + Snapshot Delta，验证：

```text
Overlay 查询结果
==
Materialized 查询结果
```

比较维度：

```text
Snapshot
Business Time
ACTIVE GRAPH
HISTORICAL GRAPH
Node / Edge / Observation
```

### 23.7 Manifest Checkpoint

- Parent + Diff 解析结果与 Full Manifest 一致；
- Checkpoint 前后 Snapshot 内容不变；
- 历史 Snapshot 不因新 Checkpoint 改变。

### 23.8 崩溃一致性

覆盖：

```text
Delta 写入前崩溃
Delta fsync 后、Manifest 发布前崩溃
Manifest 发布后、Head 更新前崩溃
Head 更新后、Materialization 前崩溃
Materialization 中途崩溃
GC 中途崩溃
```

---

## 24. 迁移说明

若此前原型已加入行级：

```text
system_from
system_to
retract_system_ts
```

迁移步骤建议：

```text
1. 冻结旧格式写入；
2. 按 System-Time 提交边界生成 Snapshot；
3. 将每个 Snapshot 的最终可见行写成 Version 集合；
4. 从热列删除 system_from/system_to；
5. 将 Retract 转换为 Snapshot Removed VersionIds；
6. 生成 Full Manifest Checkpoint；
7. 验证历史 AS KNOWN AT 结果一致；
8. 切换 Reader 到 Snapshot-first 执行模型。
```

旧文件可以保留只读兼容，但新格式不得继续写行级 System-Time。

---

## 25. 一句话交付

GTVDB 的双时间不是：

```text
每条记录同时判断 Valid Time 和 System Time
```

而是：

```text
System Time 选择整表 Snapshot
+
Valid Time 过滤 Snapshot 内的事实
```

最终核心架构为：

```text
Snapshot / Manifest System-Time
+
Node / Edge / Observation Valid-Time
+
Snapshot Delta
+
有界 Overlay 与 Materialization
+
EndpointValidityPolicy
+
ActiveNodeBitmap
+
Invalidation Event
+
外部派生重算
```

这保留了历史修正与审计能力，同时避免把行级 MVCC、RetractMap 和 System-Time 判断放进 CSR 邻接热循环，维持 GTVDB 相对于通用双时间数据库的主要性能优势。
