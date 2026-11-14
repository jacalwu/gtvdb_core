GTVDB Update 数据模型
1. 设计目标

GTVDB Update 层必须同时支持：

Plain Text
Node 属性更新
Edge 属性更新
Edge 拓扑更新
历史业务时间修正
延迟到达数据
当前状态覆盖
历史审计
Snapshot Time Travel
可选 Timeline Branch

并保持以下约束：

Plain Text
Segment Immutable
LSM Write
CSR Read
Zero-Copy mmap
Business-Time Query
System-Time Query
Snapshot Consistency

Update 不得直接修改已经发布的 GTVSeg 文件。

2. 核心时间模型

每一条 Node 或 Edge 事实包含两组时间：

Plain Text
Valid Time
事实在业务世界中何时有效
 
System Time
数据库在何时认为该事实有效

采用统一的左闭右开区间：

Plain Text
[valid_from, valid_to)
[system_from, system_to)

这与 temporal table 使用 application time 和 system time 表达现实历史与数据库认知历史的模型一致。

可见性条件

一条记录在指定业务时间和系统时间可见，当且仅当：

Rust
valid_from <= business_time
&& business_time < valid_to
&& system_from <= system_time
&& system_time < system_to
3. 统一实体标识

必须区分：

Plain Text
Logical Entity ID
Physical Version ID
Node 标识
Rust
pub type NodeId = u64;
pub type NodeVersionId = u128;

其中：

Plain Text
NodeId
表示逻辑上的同一个节点
 
NodeVersionId
表示节点的某一个不可变版本
Edge 标识
Rust
pub type EdgeId = u128;
pub type EdgeVersionId = u128;

其中：

Plain Text
EdgeId
表示逻辑上的同一条边
 
EdgeVersionId
表示该边的某个不可变版本

同一个 EdgeId 可以在不同系统时间对应多个版本，但在同一 Snapshot 中只能解析出一个当前可见版本。

4. Node 数据模型
Rust
#[repr(C)]
pub struct NodeVersion {
pub node_id: u64,
pub version_id_hi: u64,
pub version_id_lo: u64,
 
pub label_id: u32,
pub flags: u32,
 
pub valid_from: i64,
pub valid_to: i64,
 
pub system_from: i64,
pub system_to: i64,
 
pub property_ref: u64,
}
Node Update 语义

Node 属性更新不得修改原版本。

例如：

Plain Text
Node 100
status = active

在系统时间 S2 更新为：

Plain Text
status = disabled

生成：

Plain Text
旧版本：
 
node_id = 100
status = active
valid = [Day1, ∞)
system = [S1, S2)
Plain Text
新版本：
 
node_id = 100
status = disabled
valid = [Day1, ∞)
system = [S2, ∞)

业务有效时间不变，系统认知时间发生变化。

5. Edge 数据模型
Rust
#[repr(C)]
pub struct EdgeVersion {
pub edge_id_hi: u64,
pub edge_id_lo: u64,
 
pub version_id_hi: u64,
pub version_id_lo: u64,
 
pub src: u64,
pub dst: u64,
 
pub edge_type: u32,
pub flags: u32,
 
pub valid_from: i64,
pub valid_to: i64,
 
pub system_from: i64,
pub system_to: i64,
 
pub property_ref: u64,
}

建议使用：

Plain Text
EdgeVersion

作为写路径对象，Compaction 后才转换成读取优化的列式 CSR：

Plain Text
offsets
dst
edge_id
valid_from
valid_to
system_from
system_to
property_ref
6. Update 操作分型

Update 必须先分类，不能把所有变化都视为普通属性更新。

Rust
pub enum UpdateKind {
NodePropertyPatch,
NodeLabelChange,
 
EdgePropertyPatch,
EdgeValidityCorrection,
EdgeTopologyChange,
 
LogicalDelete,
Restore,
}
6.1 Node 属性更新
Plain Text
Node ID 不变
Label 不变
Properties 改变

执行为：

Plain Text
关闭旧 NodeVersion 的 system interval
+
追加新 NodeVersion
6.2 Node Label 更新

Node Label 可能影响：

Plain Text
Schema
Index
Cardinality Statistics
Graph Pattern Selectivity

因此也应生成新版本：

Plain Text
旧版本 system_to = commit_system_ts
新版本 system_from = commit_system_ts

如果 Label 同时决定物理分区，则该操作必须视为：

Plain Text
旧分区 Logical Delete
+
新分区 Insert
6.3 Edge 属性更新

如果只修改：

Plain Text
weight
status
confidence
source

且不修改：

Plain Text
src
dst
edge_type

则执行：

Plain Text
关闭旧 EdgeVersion
+
追加新 EdgeVersion

例如：

Plain Text
旧版本：
 
A -> B
weight = 0.5
system = [S1, S2)
Plain Text
新版本：
 
A -> B
weight = 0.9
system = [S2, ∞)
6.4 Edge 拓扑更新

修改以下任一字段：

Plain Text
src
dst
edge_type

必须视为：

Plain Text
Expire Old Edge
+
Insert New Edge

不能视为属性 Patch。

例如：

Plain Text
A -> B

改成：

Plain Text
A -> C

应生成：

Plain Text
旧边：
 
edge_id = E1
A -> B
system = [S1, S2)
Plain Text
新边：
 
edge_id = E2
A -> C
system = [S2, ∞)

原因是端点变化会改变：

Plain Text
CSR 排序
offsets
邻接范围
Reverse CSR
Degree Statistics
Neighborhood Filter
Topology Synopsis
7. 历史业务时间修正

假设用户在系统时间 S2 修正从 Day5 开始的数据。

原记录：

Plain Text
valid = [Day1, Day10)
system = [S1, ∞)
value = A

修正为：

Plain Text
Day1 到 Day5 使用 A
Day5 到 Day10 使用 B

更新后保存三个不可变事实版本。

历史认知版本
Plain Text
valid = [Day1, Day10)
system = [S1, S2)
value = A
当前认知的前半区间
Plain Text
valid = [Day1, Day5)
system = [S2, ∞)
value = A
当前认知的修正区间
Plain Text
valid = [Day5, Day10)
system = [S2, ∞)
value = B

这称为：

Plain Text
Valid-Time Interval Split
8. 区间切分算法

对于旧记录：

Plain Text
OldValid = [A, B)
Correction = [C, D)

首先计算交集：

Rust
overlap_start = max(A, C);
overlap_end = min(B, D);

如果：

Rust
overlap_start >= overlap_end

则记录不受影响。

否则最多产生三个新业务区间：

Plain Text
左侧保留：[A, overlap_start)
修正区间：[overlap_start, overlap_end)
右侧保留：[overlap_end, B)

伪代码：

Rust
fn correct_interval(
old: FactVersion,
correction_from: i64,
correction_to: i64,
new_value: Value,
commit_ts: i64,
) -> Vec<FactVersion> {
let start = old.valid_from.max(correction_from);
let end = old.valid_to.min(correction_to);
 
if start >= end {
return vec![];
}
 
let mut versions = Vec::new();
 
// 保留旧系统认知，关闭 system interval。
versions.push(old.with_system_to(commit_ts));
 
// 修正区间之前。
if old.valid_from < start {
versions.push(FactVersion {
valid_from: old.valid_from,
valid_to: start,
system_from: commit_ts,
system_to: i64::MAX,
value: old.value.clone(),
..old.new_version()
});
}
 
// 修正区间。
versions.push(FactVersion {
valid_from: start,
valid_to: end,
system_from: commit_ts,
system_to: i64::MAX,
value: new_value,
..old.new_version()
});
 
// 修正区间之后。
if end < old.valid_to {
versions.push(FactVersion {
valid_from: end,
valid_to: old.valid_to,
system_from: commit_ts,
system_to: i64::MAX,
value: old.value,
..old.new_version()
});
}
 
versions
}

这里的“关闭旧版本”是逻辑语义。对于不可变 Segment，物理实现不是回写旧记录，而是追加覆盖信息。

9. Delta Record 数据模型

由于旧 Segment 不可修改，建议增加统一的 Update Delta。

Rust
pub struct MutationRecord {
pub mutation_id: u128,
pub entity_kind: EntityKind,
pub entity_id: u128,
 
pub operation: MutationOperation,
 
pub commit_system_ts: i64,
pub base_snapshot_id: u64,
 
pub valid_from: i64,
pub valid_to: i64,
 
pub payload_ref: u64,
}
Rust
pub enum EntityKind {
Node,
Edge,
}
Rust
pub enum MutationOperation {
Insert,
Replace,
Correct,
Delete,
}

Mutation 不直接修改 Base Segment，而是在读取当前 Snapshot 时覆盖 Base 版本。

10. Delete 数据模型

GTVDB 的 Delete 默认应是逻辑删除。

Rust
pub struct DeleteMarker {
pub entity_kind: EntityKind,
pub entity_id: u128,
 
pub valid_from: i64,
pub valid_to: i64,
 
pub system_from: i64,
}

其语义是：

Plain Text
从 system_from 开始，
该实体在指定 valid interval 不再可见。

Delete 同样可能导致 valid interval 切分。

删除整个当前实体
Plain Text
valid = [-∞, ∞)
删除业务时间局部区间
Plain Text
valid = [Day5, Day8)
物理删除

只允许由 GC 执行：

Plain Text
无 Snapshot 引用
AND
无 Branch/Tag 引用
AND
无 mmap Reader 引用
AND
超过 Retention Window
11. Snapshot Update 模型

每次提交 Update 都产生新 Snapshot。

Rust
pub struct Snapshot {
pub snapshot_id: u64,
pub parent_snapshot_id: Option<u64>,
 
pub branch_id: u64,
pub sequence_number: u64,
pub committed_at: i64,
 
pub manifest_id: u64,
pub operation: SnapshotOperation,
}
Rust
pub enum SnapshotOperation {
Append,
Update,
Correct,
Delete,
Replace,
Merge,
Compact,
}

Snapshot 指向一份完整的可见 Segment 清单。

逻辑结构：

Plain Text
Snapshot 100
├── Segment A
├── Segment B
└── Segment C
 
Update Segment B
 
Snapshot 101
├── Segment A
├── Segment B'
└── Segment C

在物理层不需要复制 A 和 C，只需要新的 Manifest 继续引用它们。

12. Commit 数据模型

建议把一次 Update 包装为：

Rust
pub struct UpdateCommit {
pub commit_id: u128,
 
pub base_snapshot_id: u64,
pub result_snapshot_id: u64,
 
pub branch_id: u64,
pub system_ts: i64,
 
pub mutation_files: Vec<FileRef>,
pub added_segments: Vec<SegmentId>,
pub removed_segments: Vec<SegmentId>,
 
pub mode: UpdateMode,
}
Rust
pub enum UpdateMode {
Correct,
Replace,
CreateBranch,
}
13. 三种用户更新模式
CORRECT

默认模式。

Plain Text
保留旧 system-time 历史
产生新的 main Snapshot
旧 Snapshot 仍可用于审计

适用于：

Plain Text
延迟数据
错误修正
历史边修正
设备数据补录
REPLACE

用户语义上覆盖当前时间线。

Plain Text
生成新 Snapshot
main 指向新 Snapshot
旧 Snapshot 不再默认可见
保留期后 GC

即使选择 REPLACE，底层也不允许立即覆盖旧 Segment。

CREATE_BRANCH

从指定 Snapshot 创建独立时间线。

Plain Text
main ── S100 ── S101
 
└── audit-fix ── S102

适用于：

Plain Text
What-if
审计待批准修正
实验重算
模型预测
多个业务解释并存

Snapshot branch 是指向独立版本谱系的命名引用，这种设计可以参考 Apache Iceberg Branching and Tagging。

14. 写入与发布流程

一次 Update 采用以下顺序：

Plain Text
1. 读取 base_snapshot
2. 解析受影响的 Node／Edge 版本
3. 验证 Update Conflict
4. 生成 Mutation Records
5. 写入 Delta Segment
6. fsync Delta Segment
7. 生成新的 L2 Metadata
8. 生成或标记待建 L3 Metadata
9. 写入新 Manifest
10. 原子发布新 Snapshot
11. 更新 main／branch reference
12. 截断或推进 WAL checkpoint
13. 后台 Compaction
14. 延迟 GC 旧 Segment

发布之前失败：

Plain Text
旧 Snapshot 继续可用
新文件视为 Orphan

发布之后失败：

Plain Text
新 Snapshot 已经是完整可读状态
15. Update Conflict

Update 必须携带：

Plain Text
base_snapshot_id

提交时检查：

Rust
current_head_snapshot == base_snapshot_id

如果不相等，表示其间已经有其他提交。

支持三种策略：

Rust
pub enum ConflictPolicy {
Abort,
Rebase,
CreateBranch,
}
Abort
Plain Text
返回 WriteConflict

第一版建议只实现这个。

Rebase

如果两次修改：

Plain Text
不同实体
或者不重叠 valid interval

可尝试重新应用。

CreateBranch

保留两个更新结果，形成两条时间线。

16. Update 对 L2 的影响

任何 Update 都必须重算受影响 Segment 的：

Plain Text
valid_from_min
valid_to_max
max_valid_from
min_valid_to
node_id_min
node_id_max
edge_count

不得沿用旧 Segment 的 L2 Metadata。

对于 Delta Segment，也必须拥有自己的 L2 包围盒，以便扫描前剪枝。

17. Update 对 L3 的影响

L3 是派生缓存，不参与 Update 的正确性。

Update 后：

Plain Text
旧 Segment L3
仍与旧 Segment checksum 绑定
 
新 Segment L3
状态为 Missing 或 Building

Planner 行为：

Plain Text
L3 Ready
→ 使用统计优化
 
L3 Missing
→ 使用保守成本估计
 
L3 Failed
→ 正常执行，不影响结果

不得因为 L3 缺失而拒绝发布 Snapshot。

18. Delta 与 Compaction

持续 Update 会产生：

Plain Text
Base Segment
+
Delta 1
+
Delta 2
+
Delete Marker
+
Delta 3

读取成本会不断上升，因此需要 Compaction。

Plain Text
Base + Deltas + Deletes
↓
解析指定 Snapshot 的可见版本
↓
生成新 CSR Segment
↓
重建 L2
↓
异步重建 L3
↓
发布 Compaction Snapshot
Compaction 不改变逻辑数据

Compaction 前后：

Plain Text
相同 business_time
+
相同 system_time
+
相同 snapshot lineage

必须返回相同结果。

19. 核心不变量

建议将以下规则写成 Update 层硬约束。

不变量一
Plain Text
任何已发布 Segment 不得修改。
不变量二

对于同一个：

Plain Text
entity_id
branch_id
business_time
system_time

最多只有一个当前可见版本。

不变量三

同一个 Snapshot 内，同一逻辑实体的有效业务区间不得非法重叠。

不变量四

修改 Edge 的 src、dst 或 edge_type 必须生成新的逻辑边。

不变量五

Update 只有在 Manifest 原子发布后才对 Reader 可见。

不变量六

删除所有 L3 文件不得改变任何 Update 查询结果。

不变量七

旧 Segment 只有在没有 Snapshot、Branch、Tag 和 mmap Reader 引用后才能回收。

20. 建议的第一版范围

GTVDB Update V1 建议只实现：

Plain Text
Node property update
Edge property update
Edge expire + insert
Valid-time interval split
Logical delete
Main snapshot correction
Write conflict abort
Delta segment
Background compaction
Delayed GC

V1 暂不实现：

Plain Text
自动 Rebase
Branch Merge
字段级三方合并
跨 Branch Conflict Resolution
任意 Snapshot 的破坏性重写
物理原地 Update
最终架构
Plain Text
Update Request
│
▼
Update Planner
│
├── Property Patch
├── Interval Split
├── Edge Expire + Insert
└── Logical Delete
│
▼
Mutation Records
│
▼
Immutable Delta Segment
│
├── L2 Metadata
└── L3 Missing / Async Build
│
▼
Copy-on-Write Manifest
│
▼
New Snapshot
│
├── main
├── branch
└── tag
│
▼
Background Compaction + GC
结论

GTVDB 应将 Update 定义为：

针对某个 Base Snapshot，在指定 System Time 创建一组新的双时间事实版本，并通过不可变 Delta Segment 与 Copy-on-Write Manifest 原子发布新 Snapshot。

其中：

Plain Text
属性变化
= 关闭旧系统版本 + 追加新版本
 
业务时间修正
= 切分 valid-time interval + 追加当前认知版本
 
Edge 端点变化
= 旧 Edge 失效 + 新 Edge 插入
 
Replace
= 新 Snapshot 成为 main + 旧历史延迟 GC
 
Branch
= 从旧 Snapshot 派生新的并行版本谱系