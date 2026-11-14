Delta Segment 文件格式设计
1. 定位与职责

Delta Segment 用于保存已经提交但尚未合并进 Base CSR Segment 的数据变化。

支持以下 Mutation：

Plain Text
Node Insert
Node Property Update
Node Label Change
Node Delete
 
Edge Insert
Edge Property Update
Edge Valid-Time Correction
Edge Topology Change
Edge Delete

Delta Segment 必须满足：

Plain Text
1. 文件发布后不可修改；
2. 可以独立校验；
3. 可以被 mmap；
4. Update 查询无需重放 WAL；
5. 可以参与 Snapshot 可见性计算；
6. 可以通过 Compaction 生成新的 Base CSR；
7. 文件缺失不能被静默忽略；
8. L3 Metadata 缺失不影响 Delta 正确性。
2. Delta Segment 与 WAL 的区别
Plain Text
WAL
负责崩溃恢复与未完成事务重放
 
Delta Segment
负责已提交 Mutation 的持久查询

生命周期：

Plain Text
Update Request
↓
WAL Append
↓
Mutation Validation
↓
Build Delta Segment
↓
fsync Delta Segment
↓
Publish Manifest / Snapshot
↓
Checkpoint WAL

Snapshot 发布后：

Plain Text
Delta Segment = 查询事实来源之一
WAL = 可以推进 checkpoint 或截断

不得要求 Reader 重放 WAL 才能理解当前 Snapshot。

3. 文件类型

建议使用独立扩展名：

Plain Text
*.gtvd

其中：

Plain Text
GTVSeg Base Segment = *.gtvs 或现有 *.seg
GTV Delta Segment = *.gtvd
GTV Topology Metadata = *.gtvt

建议文件名：

Plain Text
<partition_id>-<commit_seq>-<checksum8>.gtvd
 

例如：

Plain Text
0000042-00001857-a891c3d2.gtvd

文件名只用于运维识别。

文件身份必须由 Header 中的：

Plain Text
delta_id
data_checksum
base_snapshot_id
result_snapshot_id

共同确定，不能依赖文件名解析。

4. 核心设计选择
4.1 Delta 中存 after-image，不存属性 Patch

Update API 可以接受：

Plain Text
SET weight = 0.9
UNSET status
INCREMENT count

但写入 Delta Segment 时，必须解析成完整的新版本：

Plain Text
完整 NodeVersion
完整 EdgeVersion

而不是持久化：

Plain Text
weight += 0.4

原因：

Plain Text
避免查询时依赖任意长 Patch Chain
避免 Compaction 必须理解历史 API 语法
避免属性 Patch 顺序影响结果
让 Delta Segment 成为独立、确定的已提交事实

Patch 语义可以保留在 WAL 或 Commit Audit Log 中，但不进入查询热路径。

4.2 关闭旧版本用 Retract Record 表达

Base Segment 已不可变，因此不能直接将旧记录：

Plain Text
system_to = S2

回写到旧文件。

Delta Segment 使用两类物理操作：

Plain Text
RETRACT 使一个旧 Version 在当前 lineage 中失效
PUT 增加一个新的完整 Version

例如属性更新：

Plain Text
RETRACT old_version_id
PUT new_version

历史业务时间修正：

Plain Text
RETRACT old_version_id
PUT left_interval_version
PUT corrected_interval_version
PUT right_interval_version

拓扑更新：

Plain Text
RETRACT old_edge_version
PUT new_edge_version

这与不可变文件系统中将数据文件和 delete 文件分离的思路相近。Apache Iceberg ContentFile 明确区分 DATA、POSITION_DELETES 和 EQUALITY_DELETES 三种内容类型。

GTVDB 不需要复制这些具体格式，但可以借鉴：

Plain Text
新增事实
和
撤销旧事实
 
是两种不同的物理内容
5. 文件总体布局
Plain Text
+--------------------------------------------------+
| Delta Header 固定 4096 B |
+--------------------------------------------------+
| Column Directory 固定宽目录 |
+--------------------------------------------------+
| Node PUT Columns |
+--------------------------------------------------+
| Edge PUT Mini-CSR |
+--------------------------------------------------+
| RETRACT Columns |
+--------------------------------------------------+
| Property Reference Columns |
+--------------------------------------------------+
| Property Blob / Dictionary |
+--------------------------------------------------+
| Optional Membership Filters |
+--------------------------------------------------+
| Footer / Section Checksums |
+--------------------------------------------------+

所有 Section：

Plain Text
64-byte aligned
 

所有可 mmap 的定宽列：

Plain Text
按实际类型自然对齐

变长属性不得混入定宽结构，应通过 property_ref 指向独立属性区。

6. Delta Header

建议 Header 固定为 4096 字节，便于：

Plain Text
单页读取
版本升级
预留扩展
校验
Rust
#[repr(C)]
pub struct DeltaHeaderV1 {
// 0x000
pub magic: [u8; 8], // b"GTVDELT1"
pub format_version: u32, // 1
pub header_size: u32, // 4096
 
// 0x010
pub flags: u64,
pub entity_mask: u32,
pub operation_mask: u32,
 
// 0x020
pub delta_id_hi: u64,
pub delta_id_lo: u64,
 
// 0x030
pub table_id: u64,
pub partition_id: u64,
 
// 0x040
pub branch_id: u64,
pub commit_sequence: u64,
 
// 0x050
pub base_snapshot_id: u64,
pub result_snapshot_id: u64,
 
// 0x060
pub commit_system_ts: i64,
pub created_at_ts: i64,
 
// 0x070
pub node_put_count: u64,
pub edge_put_count: u64,
pub retract_count: u64,
pub property_count: u64,
 
// 0x090
pub valid_from_min: i64,
pub valid_to_max: i64,
 
// 0x0A0
pub system_from_min: i64,
pub system_from_max: i64,
 
// 0x0B0
pub src_id_min: u64,
pub src_id_max: u64,
pub dst_id_min: u64,
pub dst_id_max: u64,
 
// 0x0D0
pub entity_id_min_hi: u64,
pub entity_id_min_lo: u64,
pub entity_id_max_hi: u64,
pub entity_id_max_lo: u64,
 
// 0x0F0
pub column_directory_offset: u64,
pub column_directory_count: u32,
pub column_directory_entry_size: u32,
 
// 0x100
pub property_blob_offset: u64,
pub property_blob_size: u64,
 
// 0x110
pub footer_offset: u64,
pub file_size: u64,
 
// 0x120
pub source_schema_id: u64,
pub writer_version: u32,
pub compatibility_flags: u32,
 
// 0x130
pub header_checksum: [u8; 32],
pub data_checksum: [u8; 32],
 
// 0x170
pub reserved: [u8; 3728],
}
7. Header Flags

建议定义：

Rust
pub mod DeltaFlags {
pub const HAS_NODE_PUTS: u64 = 1 << 0;
pub const HAS_EDGE_PUTS: u64 = 1 << 1;
pub const HAS_RETRACTS: u64 = 1 << 2;
pub const HAS_PROPERTIES: u64 = 1 << 3;
pub const HAS_DST_FILTER: u64 = 1 << 4;
pub const HAS_ENTITY_FILTER: u64 = 1 << 5;
pub const HAS_FORWARD_CSR: u64 = 1 << 6;
pub const HAS_REVERSE_CSR: u64 = 1 << 7;
pub const HAS_SECTION_CHECKSUMS: u64 = 1 << 8;
}

entity_mask：

Rust
pub mod EntityMask {
pub const NODE: u32 = 1 << 0;
pub const EDGE: u32 = 1 << 1;
}

operation_mask：

Rust
pub mod OperationMask {
pub const INSERT: u32 = 1 << 0;
pub const UPDATE: u32 = 1 << 1;
pub const CORRECT: u32 = 1 << 2;
pub const DELETE: u32 = 1 << 3;
pub const RESTORE: u32 = 1 << 4;
}

这些 Mask 只用于快速判断文件内容，不替代行级操作语义。

8. Column Directory

每一列由目录项描述。

Rust
#[repr(C)]
pub struct DeltaColumnEntry {
pub section_kind: u16,
pub column_kind: u16,
pub physical_type: u16,
pub encoding: u16,
 
pub offset: u64,
pub length_bytes: u64,
pub value_count: u64,
 
pub null_bitmap_offset: u64,
pub null_bitmap_bytes: u64,
 
pub min_u64: u64,
pub max_u64: u64,
 
pub checksum: u64,
pub reserved: [u64; 3],
}

建议固定大小：

Plain Text
96 bytes 或 128 bytes

第一版所有查询热列只允许：

Plain Text
PLAIN_FIXED

不在热路径引入：

Plain Text
VarInt
RLE
Dictionary Decode
General Compression

属性 Blob 可以使用独立编码，但不得影响：

Plain Text
entity lookup
时间剪枝
CSR 邻接扫描
retract resolution
9. Section Kind
Rust
#[repr(u16)]
pub enum DeltaSectionKind {
NodePut = 1,
EdgePut = 2,
Retract = 3,
Property = 4,
Index = 5,
Footer = 6,
}
 
10. Node PUT 列

Node PUT 保存完整 NodeVersion。

Plain Text
node_id u64
version_id_hi u64
version_id_lo u64
label_id u32
flags u32
valid_from i64
valid_to i64
system_from i64
property_ref u64
mutation_ordinal u32

不需要保存：

Plain Text
system_to

对于新写入版本：

Plain Text
system_to = +∞

旧版本的关闭由 RETRACT 表达。

如果未来要物化完整历史 Segment，Compaction 可以将：

Plain Text
RETRACT system_ts

转换为旧版本的：

Plain Text
system_to
11. Edge PUT Mini-CSR

Edge PUT 不能只保存无序 Mutation List，否则每次邻接查询都需要扫描整个 Delta。

Edge PUT 应在文件内部构造成小型 CSR：

Plain Text
src_nodes [u64; source_count]
src_offsets [u64; source_count + 1]
 
dst [u64; edge_put_count]
edge_id_hi [u64; edge_put_count]
edge_id_lo [u64; edge_put_count]
version_id_hi [u64; edge_put_count]
version_id_lo [u64; edge_put_count]
edge_type [u32; edge_put_count]
flags [u32; edge_put_count]
valid_from [i64; edge_put_count]
valid_to [i64; edge_put_count]
system_from [i64; edge_put_count]
property_ref [u64; edge_put_count]
mutation_ordinal [u32; edge_put_count]

排序键：

Plain Text
(src, dst, edge_type, edge_id, system_from, valid_from)

其中：

Plain Text
src_nodes + src_offsets

使 Delta Segment 可以保持：

Plain Text
O(log S) source lookup
+
O(degree_delta(src)) 邻接扫描

如果 src_nodes 稠密且范围较小，可使用完整 offsets；第一版建议使用稀疏 Source Directory，以避免少量更新产生巨大 offsets 数组。

12. 为什么需要 Mini-CSR

如果 Delta 只是普通行列表：

Plain Text
Update A -> B
Update X -> Y
Update M -> N

查询：

Plain Text
neighbors(A)

就必须：

Plain Text
扫描整个 Delta
 

当多个 Delta 累积后，复杂度将变成：

Plain Text
O(number_of_deltas × delta_rows)
 

使用 Mini-CSR 后：

Plain Text
lookup src
→ 得到连续 edge slice
→ 与 Base CSR 邻接表做版本解析

第一版允许查询时合并：

Plain Text
1 个 Base CSR
+
最多 K 个 Delta Mini-CSR

当 K 超过阈值时触发 Compaction。

13. Retract Section

Retract 表示：

从某个 system time 开始，在当前 Branch Lineage 中停止使用指定的旧 Version。

Rust
#[repr(C)]
pub struct RetractRecord {
pub entity_kind: u8,
pub reason: u8,
pub flags: u16,
pub reserved0: u32,
 
pub entity_id_hi: u64,
pub entity_id_lo: u64,
 
pub target_version_id_hi: u64,
pub target_version_id_lo: u64,
 
pub retract_system_ts: i64,
 
pub valid_from: i64,
pub valid_to: i64,
 
pub mutation_ordinal: u32,
pub reserved1: u32,
}
13.1 为什么 Retract 必须指向 Version ID

不能只写：

Plain Text
DELETE edge_id = E1

因为一个 EdgeId 可能有多个历史版本。

必须准确指向：

Plain Text
target_version_id

否则历史修正可能错误关闭：

Plain Text
另一系统时间版本
另一业务时间区间
另一 Branch 中的版本
13.2 Retract 的可见性语义

目标版本原本的系统有效期：

Plain Text
[old.system_from, ∞)

Retract：

Plain Text
retract_system_ts = S2

逻辑上解释为：

Plain Text
[old.system_from, S2)

但不会回写旧 Segment。

13.3 Retract Reason
Rust
#[repr(u8)]
pub enum RetractReason {
PropertyUpdate = 1,
ValidTimeCorrection = 2,
TopologyChange = 3,
LogicalDelete = 4,
Restore = 5,
ConflictResolution = 6,
CompactionRewrite = 7,
}

CompactionRewrite 应谨慎使用。

Compaction 一般通过 Manifest 替换文件，不应改变实体语义；它不需要生成面向用户的 Retract。

14. Mutation Ordinal

同一 Commit 内可能包含多个依赖操作：

Plain Text
RETRACT old version
PUT left version
PUT corrected version
PUT right version

Header 中的 commit_sequence 用于提交级排序。

mutation_ordinal 用于提交内部稳定排序：

Plain Text
(commit_sequence, mutation_ordinal)

规则：

Plain Text
1. Retract 在对应 PUT 之前；
2. 同一实体的操作连续；
3. mutation_ordinal 在一个 Delta 内唯一；
4. ordinal 不承担跨 Commit 排序。

Commit 是原子发布的，因此 Reader 不应观察到只应用一半 Mutation 的状态。

15. Property 区
15.1 Property Reference
Rust
#[repr(C)]
pub struct PropertyRef {
pub offset: u64,
pub length: u32,
pub schema_id: u32,
pub checksum: u64,
}

Node/Edge 中：

Plain Text
property_ref

指向 PropertyRef 表，而不是直接指向字节地址。

15.2 属性存储建议

第一版可以采用：

Plain Text
Canonical Binary Property Map

要求：

Plain Text
字段按 property_key_id 排序
相同输入产生相同字节
禁止依赖 HashMap 遍历次序
支持未知字段跳过
包含 schema_id

例如：

Plain Text
property_count u32
 
重复 property_count 次：
property_key_id u32
type_tag u16
flags u16
value_offset u64
value_length u32

固定宽值可以单独放列。

字符串、数组、JSON 类值进入 Blob。

15.3 Property 完整性

每个 PUT 必须指向完整 after-image。

例如：

Plain Text
旧属性：
{weight: 0.5, source: "sensor-a"}
 
用户操作：
SET weight = 0.9

Delta 中保存：

Plain Text
{weight: 0.9, source: "sensor-a"}

而不是只保存：

Plain Text
{weight: 0.9}
16. Node 与 Edge 的逻辑 ID 宽度

为了保持 Node CSR 热路径紧凑：

Plain Text
NodeId = u64

建议保留。

Edge 与 Version 因为涉及跨 Snapshot 唯一性，建议：

Plain Text
EdgeId = u128
VersionId = u128
DeltaId = u128
CommitId = u128
 

物理存储拆成：

Plain Text
hi u64
lo u64

避免依赖编译器对 u128 的布局和对齐实现。

17. Version ID 生成

VersionId 必须稳定且全局唯一。

推荐由以下内容计算：

Plain Text
VersionId =
hash(
branch_id,
commit_sequence,
mutation_ordinal,
entity_kind,
entity_id
)

或者使用数据库分配的 128-bit ID。

不建议仅使用：

Plain Text
system timestamp
 

因为：

Plain Text
同一时间可以有多条更新
时钟可能重复
时钟不能表达提交内部顺序
18. Delta Segment L2 Metadata

Delta 文件必须具有与 Base Segment 一样的剪枝能力。

Manifest 中的 Delta Entry 建议包含：

Rust
pub struct DeltaEntry {
pub delta_id: u128,
pub file: String,
 
pub table_id: u64,
pub partition_id: u64,
pub branch_id: u64,
 
pub base_snapshot_id: u64,
pub result_snapshot_id: u64,
 
pub commit_sequence: u64,
pub commit_system_ts: i64,
 
pub valid_from_min: i64,
pub valid_to_max: i64,
 
pub src_id_min: u64,
pub src_id_max: u64,
pub dst_id_min: u64,
pub dst_id_max: u64,
 
pub node_put_count: u64,
pub edge_put_count: u64,
pub retract_count: u64,
 
pub file_size: u64,
pub data_checksum: [u8; 32],
}

Reader 可以在不打开 .gtvd 文件的情况下判断：

Plain Text
业务时间是否可能命中
system time 是否早于 commit
src 是否可能命中
该文件是否只有 Node 或只有 Edge
19. Delta 查询解析
19.1 Snapshot 选择

首先根据 Snapshot Manifest 得到：

Plain Text
Base Segments
Visible Delta Segments

只读取：

Plain Text
delta.commit_sequence <= snapshot.sequence

且该 Delta 必须被 Manifest 明确引用。

不要通过目录扫描猜测 Delta 可见性。

19.2 Retract Resolution

建议在打开 Snapshot 时，为相关 Delta 构建：

Plain Text
RetractMap<VersionId, RetractSystemTs>

查询旧 Version 时：

Rust
fn effective_system_to(
version_id: VersionId,
original_system_to: i64,
retracts: &RetractMap,
) -> i64 {
retracts
.get(version_id)
.copied()
.unwrap_or(original_system_to)
.min(original_system_to)
}
 

然后按双时间条件判断。

19.3 邻接查询
Plain Text
neighbors(src, business_t, system_t)

流程：

Plain Text
1. L2 选择可能命中的 Base Segment；
2. L2 选择可能命中的 Delta Segment；
3. 从 Base CSR 取得 src edge slice；
4. 从每个 Delta Mini-CSR 取得 src edge slice；
5. 应用 RetractMap；
6. 应用 valid-time filter；
7. 应用 system-time filter；
8. 按 EdgeId / VersionId 解析可见版本；
9. 输出结果。

排序时不要只按 dst 去重，因为可能存在：

Plain Text
A -> B 的多种 edge_type
A -> B 的多条平行边

应至少使用：

Plain Text
(edge_id, version_id)

解析版本。

20. 同一实体的胜出规则

正常情况下，设计不变量应保证：

Plain Text
同一 branch
同一 snapshot
同一 entity_id
同一 business_time
同一 system_time
最多一个可见版本

如果由于损坏或 Bug 出现多个候选版本，Reader 不应静默使用最后一条。

建议：

Rust
pub enum VersionConflictPolicy {
Error,
HighestCommitSequence,
}

生产默认：

Plain Text
Error

诊断或修复工具才允许：

Plain Text
HighestCommitSequence

避免错误数据被静默掩盖。

21. Edge Topology Change

原 Edge：

Plain Text
EdgeId E1
A -> B

修改为：

Plain Text
A -> C

建议生成：

Plain Text
RETRACT Version(E1-V1)
 
PUT EdgeId E2
A -> C
Version E2-V1

默认生成新的 EdgeId，因为图拓扑身份已经改变。

如果业务明确要求一条逻辑 Edge 在端点变化后保持同一个 EdgeId，可以允许：

Plain Text
EdgeId 不变
VersionId 改变

但这会让：

Plain Text
一条 Edge 的身份

不再稳定地表示：

Plain Text
固定的 src + dst + type

因此 GTVDB 默认建议：

Plain Text
端点或 edge_type 改变
→ 新 EdgeId
22. Delete 表达

逻辑删除由 Retract 完成。

删除当前版本
Plain Text
RETRACT target_version_id
reason = LogicalDelete

不生成新 PUT。

删除部分 Valid-Time 区间

旧版本：

Plain Text
valid = [Day1, Day10)

删除：

Plain Text
[Day4, Day7)

生成：

Plain Text
RETRACT old_version
 
PUT [Day1, Day4) old payload
PUT [Day7, Day10) old payload

删除区间不生成 PUT。

23. Update 示例
23.1 Edge 属性更新

原版本：

Plain Text
E1-V1
A -> B
weight = 0.5
valid = [Day1, ∞)
system_from = S1

在 S2 更新为 weight = 0.9：

Plain Text
Retract:
target_version = E1-V1
retract_system_ts = S2
 
PUT:
edge_id = E1
version_id = E1-V2
A -> B
weight = 0.9
valid = [Day1, ∞)
system_from = S2
23.2 历史区间修正

原版本：

Plain Text
E1-V1
valid = [Day1, Day10)
value = A
system_from = S1

在 S2 把 [Day5, Day8) 修改为 B：

Plain Text
RETRACT E1-V1 at S2
Plain Text
PUT E1-V2
valid = [Day1, Day5)
value = A
system_from = S2
Plain Text
PUT E1-V3
valid = [Day5, Day8)
value = B
system_from = S2
Plain Text
PUT E1-V4
valid = [Day8, Day10)
value = A
system_from = S2
24. 文件校验

建议使用三级校验。

Header Checksum

覆盖：

Plain Text
Header 中除 header_checksum 自身之外的字节
 

作用：

Plain Text
检测 Header 损坏
检测版本／offset 错误
Section Checksum

每个 Column Directory Entry 包含：

Plain Text
checksum

作用：

Plain Text
按 Section 定位损坏
允许诊断工具检查部分文件
Data Checksum

覆盖：

Plain Text
Header 后的所有有效数据区

Manifest 同时保存同一 checksum。

打开时：

Plain Text
Manifest checksum
==
Header data_checksum
==
实际文件 checksum
25. 崩溃一致性

Delta 提交流程：

Plain Text
1. 写入临时文件：
<delta_id>.gtvd.tmp
 
2. 写入完整 Section
 
3. 写入 Footer
 
4. 计算 data_checksum
 
5. 回填 Header
 
6. fsync(file)
 
7. rename:
.gtvd.tmp → .gtvd
 
8. fsync(directory)
 
9. 写新 Manifest 临时文件
 
10. fsync(manifest)
 
11. 原子 rename Manifest
 
12. 更新 Snapshot Ref
 
13. WAL checkpoint

如果在 Manifest 发布前崩溃：

Plain Text
Delta 是 Orphan
不对 Reader 可见
后台清理

如果在 Manifest 发布后崩溃：

Plain Text
Delta 已完整存在
新 Snapshot 可读
 
26. Delta Compaction

触发条件可以包括：

Plain Text
Delta 文件数量超过阈值
Delta Edge 数占 Base Edge 数比例过高
RetractMap 过大
同一 Entity 的版本链过长
邻接查询 Delta Fan-In 过高

Compaction 流程：

Plain Text
Base CSR
+
Visible Delta PUT
+
Visible Retract
↓
解析目标 Snapshot 的完整可见版本
↓
生成新 Base CSR
↓
生成新 L2 Metadata
↓
发布 Compaction Snapshot
↓
异步生成 L3
↓
旧 Base / Delta 延迟 GC
 

Compaction 后可以将 Retract 物化成新的：

Plain Text
system_to

但必须保留旧 Snapshot 所依赖的原文件，直到没有引用。

27. Delta Segment 与 L3

Delta Segment 自身只保存正确性所需的数据和 L2 Metadata。

可以异步生成：

Plain Text
local degree delta
temporal density
direct-membership filter
 

但这些都属于 L3。

规则：

Plain Text
Delta 发布不等待 L3
L3 缺失不影响查询结果
L3 checksum 必须绑定 Delta data_checksum
删除 L3 只会让查询变慢
28. V1 不支持的内容

第一版明确不做：

Plain Text
Delta 内字段级 Patch Chain
原地更新 Delta
Delta 共享可变 Property Blob
依赖 WAL 执行查询
任意多个 Delta 的无界合并
在 Retract 中只写 EntityId 而不写 VersionId
把 Edge 端点变化当普通属性更新
在 L3 中保存 Update 正确性信息
29. V1 推荐范围

Delta Segment V1 建议包括：

Plain Text
固定 4096 B Header
Column Directory
Node PUT Columns
Edge PUT Mini-CSR
Version-Level Retract
完整 after-image Property
Valid-Time / System-Time Bounds
Manifest DeltaEntry
Header/Data/Section Checksum
Background Compaction

V2 再考虑：

Plain Text
Reverse Delta Mini-CSR
Bloom / XOR Filters
属性 Dictionary
分区级 Retract Index
Delta Merge
更细粒度的列压缩
30. 核心不变量
Plain Text
1. 已发布 Delta Segment 永不修改。
 
2. PUT 必须保存完整的新实体版本。
 
3. RETRACT 必须指向明确的 VersionId。
 
4. 一个 Update Commit 的全部 Delta 文件必须原子可见。
 
5. Edge src、dst 或 type 改变必须执行拓扑替换。
 
6. Manifest 是 Delta 可见性的唯一权威来源。
 
7. Delta 缺失或 checksum 不匹配必须报错，不得忽略。
 
8. L3 缺失只能影响性能，不能影响结果。
 
9. Compaction 前后，在相同 Snapshot、Business Time、
System Time 下的查询结果必须完全一致。
 
10. 旧 Delta 只有在无 Snapshot、Branch、Tag、
mmap Reader 引用后才允许 GC。
最终结论

GTVDB Delta Segment 应定义为：

一个不可变、列式、可 mmap 的已提交 Mutation 文件，由完整的新版本 PUT、针对旧 VersionId 的 RETRACT，以及用于邻接查询的 Edge Mini-CSR 组成。

它的核心结构是：

Plain Text
Delta Header
+
Node PUT Columns
+
Edge PUT Mini-CSR
+
Version-Level Retract
+
Complete Property After-Image
+
L2 Pruning Metadata
+
Checksums

读取模型为：

Plain Text
Snapshot
=
Base Segments
+
Visible Delta PUT
-
Visible Retract

这样既不会把 WAL 变成查询存储，也不需要修改已经发布的 Base CSR，并为后续 Compaction、双时间历史查询、Branch 和 delayed GC 保留了清晰边界。