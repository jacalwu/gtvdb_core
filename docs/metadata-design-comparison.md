# 元数据设计选型：OceanBase 分层 Skip Index / MongoDB Catalog / Neo4j 图拓扑

> 2026-11-09 修订｜对照 `mongodb/mongo` master（2026.10）、`neo4j/neo4j` 2026.10 源码、`oceanbase/oceanbase` 既有评估（[oceanbase-storage-lessons.md](oceanbase-storage-lessons.md)）与 `gtvdb_core` 当前代码事实。  
> 触发问题一（11-08）：上一轮建议的**第 2 项「给段补时间盒」**原定按 OceanBase 宏块 skip index 落地；gtvdb 是**图数据库**，是否该改用 MongoDB 的元数据管理方式。→ 结论见 §0/§7。  
> 触发问题二（11-09）：L3「Topology Synopsis Metadata」构想（Degree / Neighborhood / Temporal / Reachability / Component / Reverse / TraversalCost）该怎么定；参考 Neo4j 的图拓扑元数据管理。→ 结论见 §6。  
> 状态：设计评估 + 实施口径修订。L1 与 L2 阶段 A **已按 §7 落地并通过测试**，实测数据见 [metadata-benchmark.md](metadata-benchmark.md)；L3 仍是设计。

---

## 0. 结论先行

**不是二选一，而是分层选型；但就「第 2 项」本身而言，OceanBase 路线更合适，应继续按它做。**

| 元数据层                                                   | 选型                          | 理由                                                                                     |
| :----------------------------------------------------- | :-------------------------- | :------------------------------------------------------------------------------------- |
| **L2 物理剪枝元数据**（段时间盒、chunk/src 上卷、skip index）＝**第 2 项** | ✅ **OceanBase**             | 定宽、独立、不可变、可 `cast_slice`、按物理块层级上卷，与 GTVSeg「磁盘字节布局 == 内存 CSR 布局 == CPU cache line」恒等式同构 |
| **L1 逻辑 catalog**（表 / Schema / 快照 / 段清单 / 版本边界）        | ⚠️ **借 MongoDB 的架构原则，不借实现** | 借「单一权威 + 内存快照视图 + ident 延迟回收」三条；不借 `_mdb_catalog` 的 B-tree/BSON、WiredTiger 事务与两阶段 DDL  |
| **L3 图拓扑元数据**（度分布、反图标记、时间密度、遍历代价）                     | ⚠️ **借 Neo4j 的三条机制，删 2 项、改 4 项** | Neo4j 是三者中唯一真有「图拓扑元数据」的参考实现，但它只长期保留**能 O(1) 增量维护的精确计数**，其余（度分布 / 连通性 / 可达性）全部外置、按需构建、允许缺失。详见 §6 |

**决定因素不是「图数据库 vs 关系型数据库」，而是「不可变定宽 mmap 段 + `offsets[src]..offsets[src+1]` 的 O(1) 邻接定位」。** 在这条轴上：

- OceanBase 的 min/max 是**独立于数据、按物理块层级组织、随 SSTable 一起只读**的聚合量 → 映射进来只需读几个 `i64`，无解码、无索引查找、无谓词重写。
- MongoDB 的 min/max 是**嵌在可变 BSON 文档里**的 `control.min/max`，靠查询优化器重写谓词后在 `COLLSCAN` / `FETCH` / `CLUSTERED_IXSCAN` 上过滤 → 需要 BSON 解析、解压桶、B-tree 事务，与 mmap 零拷贝互斥（论证同 [storage-mmap-design.md](storage-mmap-design.md) §2 对 Parquet 的判断）。

**一个有价值的反转**：MongoDB 时序集合的 `control.min/max` + `wholeBucketFilter` **概念**值得借，正好补上第 3 项「按时间段汇总」——但载体仍然是定宽 header / manifest 标量，不是 BSON 文档。详见 §7.4。

**L3 的结论与 L2 不同**：L3 不该「全按 Neo4j 做」，也不该「照构想做」。Neo4j 给的是一组**判断标准**（什么值得持久化、什么必须外置、什么根本不该有），而不是一张字段表。按这套标准，构想里的 9 个字段有 **2 个直接删、4 个要改造、3 个保留**（§6.6）。其中 `TemporalSynopsis` 是 Neo4j 帮不上忙、但 gtvdb 必须自己做的那一项。

---

## 1. 先把「metadata」拆成三层，否则比较会错位

| 层                 | 回答什么问题                             | gtvdb 现状                                                                                                                                    | MongoDB                                                             | OceanBase                                   |
| :---------------- | :--------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------ | :------------------------------------------ |
| **L1 逻辑 catalog** | 有哪些表？什么 Schema？当前版本是哪份文件清单？        | `FsCatalog`（`tables.json` / `schemas.json` / `snapshots.jsonl` / `files.jsonl`）+ `SegmentStore::manifest.json` + DataFusion 会话注册表（**两套真相**） | `_mdb_catalog`（一张普通 WT 表）+ `CollectionCatalog`（内存权威）+ 分片 `config.*` | Tablet / SSTable meta、多版本元数据表、全局快照点         |
| **L2 物理剪枝元数据**    | 这个块 / 段 / chunk 有没有可能命中 `[T1,T2)`？ | GTVSeg header `max_valid_from / min_valid_to`、`chunk_max_valid_to`；`DataFile.event_time_min/max`；`SegmentEntry` **无 business-time 边界**      | 时序桶 `control.min/max`；分片 `config.chunks.min/max`（路由用，非块统计）          | 微块每列 max/min/null count → 上卷宏块 → 上卷 SSTable |
| **L3 图拓扑元数据**     | 这个 src 有多少邻居？反图是否已物化？遍历预算多少？       | `degree_histogram()` / `GraphStats`（需手动 `set_graph_stats()`）                                                                                | 无对应物                                                                | 无对应物                                        |

「第 2 项」只落在 **L2**。把它和 MongoDB 的 `_mdb_catalog`（L1）直接对比，是层级错配。

上表的 L3 一列需要修正：**Neo4j 并非「无对应物」**，它有一整套图拓扑元数据，只是形态与直觉不同——精确计数在线增量、度外置到独立 KV、连通性完全不做（§6）。三者对 L3 的分工见 §6.10。

---

## 2. MongoDB 元数据实现（源码级事实）

> 以下路径均为 `mongodb/mongo` **master**（2026-11 版本）。该分支已把 catalog 代码从 `src/mongo/db/catalog/` 迁到 `src/mongo/db/shard_role/shard_catalog/`，并把 `_mdb_catalog` 封装成 `MDBCatalog`。旧文常引用的 `src/mongo/db/catalog/catalog_control.cpp` 在当前 master 已不存在。

### 2.1 持久化：`_mdb_catalog` 是一张普通 WiredTiger 表

`src/mongo/db/storage/mdb_catalog.h` 原文：

> *"A wrapper around the '\_mdb_catalog' storage table. Each row in the table is indexed with a 'RecordId', referred to as the 'Catalog ID', and contains a BSON document that describes a collection's properties, indexes, and the idents which map to its storage resources."*

每个 catalog entry 是 **一个 BSON 文档**，顶层四个字段：

```
{ ident: <string>,        // 主存储表名（对应一个 RecordStore）
  idxIdent: {<indexName>: <indexIdent>},
  md: <BSONObj>,          // Collection / Index 属性（CollectionOptions、索引规格…）
  ns: <string> }          // namespace
```

关键点：

- **一张表一行，键是 `RecordId`（Catalog ID），值是 BSON**；`getCursor()` / `getAllCatalogEntries()` 在启动时**扫描全表**加载。
- `DurableCatalog`（`shard_role/shard_catalog/durable_catalog.h`）是 storage 层之外的解析视图：*"A parsed view and entry point to read and modify on-disk catalog metadata (\_mdb_catalog) from outside the 'storage' layer."*，提供 `putMetaData` / `createCollection` / `dropCollection` / `renameCollection`。
- 存储引擎层还有自己的元数据：WiredTiger 的 `WiredTiger.wt` 元数据表与 `WiredTiger.turtle` 引导文件（二级来源，见 §9）。
- `ident` 是**间接层**：catalog 只存字符串标识，物理表由 WT 的 `table:<ident>` URI 管理；`directoryperdb` / `wiredTigerDirectoryForIndexes` 会改变 ident 形态。

### 2.2 内存权威：COW + 时间点 catalog 读

`CollectionCatalog`（`shard_role/shard_catalog/collection_catalog.h`）：

- 内部是 `immutable::unordered_map<UUID, shared_ptr<Collection>>` / `immutable::unordered_map<NamespaceString, ...>` / 有序的 `OrderedCollectionMap`。
- 写模型明确是 copy-on-write：
  > *"Perform a write to the catalog using copy-on-write. A catalog previously returned by get() will not be modified."*
- 读侧 `get()` 返回与当前 `RecoveryUnit` **storage snapshot 一致**的 catalog；`stash()` 把某个 catalog 指针绑定到 snapshot；`latest()` 用于判断可写性。
- 时间点读：`establishConsistentCollection(opCtx, nssOrUUID, readTimestamp)`，配合 `HistoricalCatalogIdTracker`（timestamp → catalogId）、durable catalog、drop-pending ident 重建历史 `Collection` 实例；PIT 时集合不存在则返回 `nullptr`。
- `getAllConsistentDbNames()` 也返回与 snapshot 一致的库名列表。

**这一层是 MongoDB 最值得借的部分**：catalog 是版本化的、只读快照视图，DDL 通过 COW 发布，而不是原地改一个全局 map。

### 2.3 DDL 持久性：不是 manifest，是一整套事务机制

`src/mongo/db/storage/storage_engine.h`：

- `loadMDBCatalog()` / `closeMDBCatalog()` 负责启动时填充 / 拆除内存结构，需全局排他锁；还会处理 unclean shutdown 后的 **orphaned idents**。
- 持久性由 `WriteUnitOfWork` commit + journal（`waitUntilDurable`）+ checkpoint（`getCheckpointIteration` / `hasDataBeenCheckpointed`）+ `setStableTimestamp` / `recoverToStableTimestamp` 组合保证。
- 删除类 DDL 不立即删物理表：`dropIdentTimestamped()` 保证 *"The ident will remain visible in checkpoints taken before 'timestamp' and will be dropped in checkpoints taken at or after 'timestamp'"*；`addDropPendingIdent()` + `TimestampMonitor` 在「时间足够旧 + 无活跃 holder」后由 reaper 异步清理；`markIdentInUse()` 阻止误删。

**成本提示**：这套机制依赖 WiredTiger 的 MVCC / journal / oplog / 两阶段 DDL / 复制回滚，是数十万行级基础设施。gtvdb 是嵌入式单进程分析引擎（`README` 已把 transactions / 分布式列为 out of scope），整套搬入是过度设计。

### 2.4 时序集合：min/max 存在 bucket 文档内部

`src/mongo/db/timeseries/README.md` 的 bucket schema（v1）：

```
{ _id: <ObjectId，时间部分 = control.min.<timeField>>,
  control: {
      version: 1,
      min: { <timeField>: ..., <field0>: 最小值, ... },
      max: { <timeField>: ..., <field0>: 最大值, ... },
      closed: <bool> },
  meta: <metaField 值>,
  data: { <timeField>: {'0':..., '1':...}, <field0>: {...} } }
```

- 每个 bucket 最多 `timeseriesBucketMaxCount = 1000` 条 measurement，时间跨度由 `granularity` / `bucketMaxSpanSeconds` 决定（seconds→1h、minutes→1d、hours→30d）。
- `BucketCatalog`（`timeseries/bucket_catalog/bucket_catalog.h`）是**纯内存**协调器：32 个 `Stripe`，各自有 mutex；维护 `openBucketsById` / `openBucketsByKey` / `idleBuckets` / `archivedBuckets`（`btree_map`，用 `lower_bound` 找可复用的桶）；桶有 open / archived / closed / frozen / idle 状态。
- 提交 batch 时把插入转成列式 `data`，并**增量更新 `control.min/max`**（首次插入生成完整文档，后续用 DocDiff "v2" delta 更新）。

查询侧（`src/mongo/db/query/timeseries/README.md`）：

- 用户集合是 view，所有查询被改写成针对 `system.buckets.<ns>` 的 pipeline，前置 `$_internalUnpackBucket`。
- measurement 字段的 `$match` 下推成 bucket 层 control filter：
  ```
  {control.max.measurement0: {$gte: 42}, control.min.measurement0: {$lte: 42}}
  ```
  依据是 *"the control fields are guaranteed to hold the minimum and maximum value of all fields in the bucket"*。
- `timeField` 谓词还利用 bucket `_id` 内嵌的最小时间戳生成 bounded `CLUSTERED_IXSCAN`；并按 `bucketMaxSpanSeconds` **放宽**谓词以免漏桶（`collMod` 会改变该值），解包阶段再精确过滤。
- `wholeBucketFilter` / `eventFilter`：若整桶满足谓词（如 `control.max.time < cutoff`），跳过对每条 measurement 的过滤。
- `$group` 只含 `$min`/`$max` 时，直接改写为 `{$min: '$control.min.val'}` 并**移除解包阶段**；last-point 查询也用 `control.max.time` 在桶层排序。

**这是 MongoDB 与 gtvdb 需求最接近的一点**，也是唯一值得引入的概念（见 §7.4）。但请注意它的物理形态：min/max 与数据在**同一个可变 BSON 文档**里，读它必须走 WT cursor、BSON 解析、桶解压。

### 2.5 分片层也有 min/max，但用途是路由

`shard_role/shard_catalog/README_sharding_catalog.md`：`config.chunks` 每个 chunk 一个文档，含 `min` / `max`（shard key 范围）、owning shard、`history`、`onCurrentShardSince`；`CatalogCache` 用它做 router 的 routing table，shard 侧 `CollectionMetadata` 做 ownership filter。

**这是逻辑范围路由元数据，不是块级统计**，不能拿来当 GTVSeg 的 zone map 用。

### 2.6 MongoDB 明确没有的东西

- 没有定宽、可 `cast_slice` 的 skip-index 列；
- 没有不可变段 / COW 文件（数据是可变 B-tree，靠 journal + checkpoint）；
- 没有 `offsets[src]..offsets[src+1]` 级邻接元数据；`$graphLookup` 是运行时递归 join，元数据层不携带度分布 / 反图 / 时间包围盒；
- 因此 **L3 无法从 MongoDB 借鉴**。

---

## 3. OceanBase 元数据（摘要）

详见 [oceanbase-storage-lessons.md](oceanbase-storage-lessons.md) §1.4 / §4③ / §5.1，此处只列与选型相关的三条：

1. **Skip Index 分层上卷**：微块（~16 KB，I/O / 压缩 / 编码单位）对每列聚合 `max / min / null count` → 上卷宏块（固定 2 MB）→ 上卷 SSTable；查询按扫描范围逐层下钻，先过滤再读数据。
2. **元数据随只读 SSTable 生成**：编码与压缩只发生在 compaction，写路径不碰；SSTable 不可变，宏块可按 `(time_boundary, checksum, version)` 直接复用、不重读不重写。
3. **元数据本身是定宽的**：放进 header 热元数据区后仍可 `cast_slice`、仍零拷贝，不破坏 GTVSeg 的任何约束（[storage-mmap-design.md](storage-mmap-design.md) §3.2）。

一句话：**OceanBase 的 min/max 是「物理块的聚合索引」，MongoDB 的 min/max 是「文档里的摘要字段」。** 前者能贴在 CSR 上，后者不能。

---

## 4. gtvdb 的决定性约束（代码事实）

| 约束             | 代码依据                                                                                                                                                   | 对元数据的要求                                        |
| :------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------- |
| mmap 恒等式       | `docs/storage-mmap-design.md` §2/§3.2；`segment.rs` `Col<T>` / `MappedCol<T>`                                                                           | 元数据必须定宽、无间接引用、无解码                              |
| O(1) cold open | `segment.rs:56-64` `header_checksum` 只覆盖 `[0,256)`                                                                                                     | 段剪枝必须**不打开文件**就能判定（靠 manifest），或只读固定 header 字节 |
| O(1) 邻接定位      | `csr.rs:313` `edge_slices(src)` 返回连续区间；§5.3「LSM 写、CSR 读」                                                                                               | 不能把元数据设计成需要 k 层查找 / k 路归并                      |
| 半开区间语义         | `csr.rs:386` `all_active_at(t) = t >= max_valid_from && t < min_valid_to`                                                                              | 剪枝判定必须严格用 `[valid_from, valid_to)`             |
| 三类查询共用一份 CSR   | 时间点 `neighbors/as_of`；时间段 `trunc + GROUP BY`/UDWF/`WindowedMotif`；图漫步 `khop_bounded/directed`                                                          | 元数据应服务同一份 CSR，而不是另建一套索引                        |
| 现状缺口           | `FsCatalog` 与 SQL 会话注册两套真相；`ScanFilter` 全仓仅定义 + 测试调用（无生产调用方）；`SegmentEntry` 只有 `system_ts / txn_start / txn_end / edge_count`；`graph_at(T)` 每次重排物化 CSR | 第 2 项必须先补 business-time 边界，第 1/3 项才有意义         |

---

## 5. 逐维对比

| 维度                    | OceanBase 分层 Skip Index      | MongoDB（catalog + 时序桶）                                                            | 对 gtvdb 的判定                             |
| :-------------------- | :--------------------------- | :-------------------------------------------------------------------------------- | :-------------------------------------- |
| 元数据粒度                 | 微块 → 宏块 → SSTable，**物理块层级**  | 一行/表（`_mdb_catalog`）、一文档/桶（`control.min/max`）、一文档/chunk（分片）                       | ✅ OB：可映射成 chunk → src → 段               |
| 与数据的耦合                | 独立定宽数组，随不可变 SSTable 生成       | 与数据在**同一个可变 BSON 文档**；改数据要 delta 更新 control                                       | ✅ OB：gtvdb 段不可变，无需更新                    |
| 更新模型                  | COW 新段 / compaction 时重算      | 原地 B-tree 更新 + MVCC + journal + DocDiff                                           | ✅ OB：与 P1「写新段 → 发布 manifest → 截断 WAL」一致 |
| 剪枝路径                  | 读定宽 header/数组 → 算术比较         | 优化器重写谓词 → COLLSCAN/FETCH/IXSCAN → 解析 BSON → 解压桶 → control filter                  | ✅ OB：热循环零额外成本                           |
| mmap 零拷贝兼容            | ✅ 兼容（`cast_slice`）           | ❌ 不兼容（同 Parquet 判断）                                                               | ✅ OB：硬红线                                |
| 贴合图遍历                 | 可按 `src` 上卷，O(1) 判定整条邻接表无存活边 | bucket 按 `(meta, time)` 分组 measurement，无 `offsets[src]` 概念                        | ✅ OB；L3 另按 Neo4j 口径（§6）             |
| 时间点 / 时间段查询           | skip index 天然服务范围扫描          | 时序桶 `control.min/max` + 谓词放宽 + 解包后精确过滤，机制成熟但重                                     | ✅ OB 做载体，借 Mongo 的「整桶命中」概念              |
| 事务 / 并发 / PIT catalog | 多版本 tablet meta、全局快照点、分布式    | **明显更强**：COW catalog、snapshot 绑定、`HistoricalCatalogIdTracker`、drop-pending reaper | ⚠️ Mongo：只借原则，不借实现                      |
| 工程成本                  | 中等（定宽数组 + header 扩展）         | 极高（WT 事务 / journal / oplog / 两阶段 DDL / 复制）                                        | ✅ OB：嵌入式单进程可承受                          |


## 6. L3 图拓扑元数据：Neo4j 的答案，与对构想的裁决

> 结论先行：**Neo4j 是三者中唯一真有「图拓扑元数据」的实现，但它给的是判断标准，不是字段表。** 按它的标准，构想里 9 个字段 **保留 3（其中 2 项降为派生）、改造 4、删除 2**，并且要修掉 3 个结构性错误（§6.7）。

### 6.1 事实 A：planner 只吃五类统计，且大多是估算

`community/cypher/planner-spi/src/main/scala/org/neo4j/cypher/internal/planner/spi/GraphStatistics.scala` 全文只有 8 个方法（下表把两对同类合并成一行）：

| 方法 | 回答 | 性质 |
| :-- | :-- | :-- |
| `nodesAllCardinality()` | 全库节点基数 | 估算 |
| `nodesWithLabelCardinality(label)` | 某标签节点基数 | 估算 |
| `patternStepCardinality(from, type, to)` | `(a)-[r]->(b)` 三元组基数 | 精确（counts store） |
| `mostCommonLabelGivenRelationshipType(type)` | 某类型的常见端点标签 | 精确 |
| `uniqueValueSelectivity` / `indexPropertyIsNotNullSelectivity` | 索引选择性 | 采样近似 |
| `getHistograms(labels, pk)` / `getHistograms(type, pk)` | 范围谓词直方图 | **生产不填** |

两条注释是决定性的：

- `nodesWithLabelCardinality`：*"Attention: This method does NOT return the number of nodes anymore!"* —— 返回的是**基数估算**，不是节点数。
- `getHistograms`：*"We currently only implement this for integration testing purposes."* —— **直方图管道存在，但生产环境不填**。

**没有的东西更能说明问题**：没有 `averageDegree()`、没有 `degreeDistribution()`、没有 SCC、没有 reachability、没有 Bloom、**没有任何 cost 字段**。参考实现的图数据库，把「图拓扑统计」压到了极限。

### 6.2 事实 B：计数是 O(1) 事务内增量，全量重扫只是兜底

`record-storage-engine/.../store/stats/RecordDatabaseEntityCounters.java` 展示了**双通道**：

```java
public long nodes(CursorContext ctx) { return idGeneratorFactory.get(RecordIdType.NODE).getHighId(); }      // 估算，O(1)
public long allNodesCountStore(ctx) { return countsStore.nodeCount(ANY_LABEL, ctx); }                        // 精确
public long allRelationshipsCountStore(ctx) {
    return countsStore.relationshipCount(ANY_LABEL, ANY_RELATIONSHIP_TYPE, ANY_LABEL, ctx);                  // 三元组
}
```

官方手册对更新时机的描述更关键：*"These numbers are updated whenever you set or remove a label from a node."* —— **在线、事务内、O(1)**。

另一条路是 `CountsComputer.java`：`NodeCountsStage` + `RelationshipCountsStage` **全表扫描重建**，只用于导入 / 迁移 / 修复，不是常态。

> **修正 ①**：构想「禁止在线更新 Degree Histogram」把「不可变」放错了位置。Neo4j 的做法是**维护期可增量、发布期才不可变**。gtvdb 的 `MemSegment` 本就是写缓冲，`+1/-1` 计数器在那里零成本，flush 时随段固化即可——不必为了「不可变」在 flush 时再扫一遍 E 条边。

### 6.3 事实 C：度数外置——记录里只留 3 个布尔标志

这一段最值得抄，因为它正好回答「大块拓扑元数据放哪」。

`RelationshipGroupRecord.java`（2026.10）的字段：

```java
private int type;
private long next;
private long firstOut, firstIn, firstLoop, owningNode;
private boolean externalDegreesOut;   // ← 度数是否外置
private boolean externalDegreesIn;
private boolean externalDegreesLoop;
```

**没有直观的 `count` 字段**——度数要么随记录的二级存储单元（`toString()` 里可见 `secondaryUnitToString()`）承载，要么由三个布尔标志外置到独立存储：

```java
// internal/counts/GBPTreeRelationshipGroupDegreesStore.java
static CountsKey degreeKey(long groupId, RelationshipDirection direction) {
    return new CountsKey(TYPE_DEGREE, groupId << 2 | direction.id(), 0);
}
```

`DegreesRebuildFromStore.java` 的类注释一句话说清了它的地位：

> *"Scans the store and rebuilds the `GBPTreeRelationshipGroupDegreesStore` contents if the file is missing."*

缺文件时 `log.warn("Missing relationship degrees store, rebuilding it.")`，两阶段流水（`PrepareCacheStage` 读组记录 → `CalculateDegreesStage` 扫关系 → `cache.writeTo(updater)`），1024 路 striped latch，堆外 `LongArray`，然后继续启动。

> **三条可直接复用的结论**
>
> 1. **大元数据不进记录，但记录里留一个标志位**——构想 §6.5.4 只说了「不要放进 `SegmentEntry`」，没说「那放什么、怎么引用」；答：`flags` + 独立文件 + 内容校验和。
> 2. **可丢弃**：L3 是缓存，不是真相。丢了能重建，重建不阻塞启动。
> 3. **key 按 `(对象, 方向)`**：方向是拓扑元数据的第一维，正对应 gtvdb 的 `transpose()`。

### 6.4 事实 D：重统计是阈值触发的后台采样

官方 `statistics and execution plans` 给出了一条完整的**陈旧度契约**：

| 机制 | 默认值 | 含义 |
| :-- | :-- | :-- |
| `db.index_sampling.background_enabled` | `true` | 索引选择性靠后台全索引扫描采样 |
| `db.index_sampling.update_percentage` | `5` | 某索引变更达总量 5% 才重采样 |
| `dbms.cypher.statistics_divergence_threshold` | `0.75` | `abs(a-b)/max(a,b)` 超过才认为计划过期 |
| `dbms.cypher.min_replan_interval` | `10s` | 两次重规划最小间隔 |
| 阈值衰减 | → 10% / ~7h | 变化缓慢的库也终会被重规划 |

> **修正 ②**：构想 `TopologyMetaState::{Stale, Invalid}` 只给了状态名，没给**判定者与判据**。Neo4j 给的是可执行规则：**相对偏差阈值 + 最小间隔**，而不是「写入即标 Stale」。gtvdb 应落成 `topology_divergence_threshold`，由 planner 侧判定是否重读 L3。

### 6.5 事实 E：core 里根本没有 SCC / 连通分量 / 可达性

`community/graph-algo/src/main/java/org/neo4j/graphalgo/impl/` 只有三个包：`path`、`shortestpath`、`util`。

连通分量、SCC、PageRank、社区发现全在 **GDS**（独立产品 / 独立仓库），且 GDS 自己也是**按需计算、按图缓存**——官方文档：`degreeDistribution` *"can be fairly time-consuming to compute for larger graphs. Its computation is cached per graph"*。

> 把 `ComponentSynopsis` / `ReachabilitySynopsis` 放进**每段**的常驻元数据，是三家参考实现里**没有任何一家做过**的设计。这不是创新，而是把 O(V+E)~O(E²) 的全局分析伪装成段的属性——§6.7.1 会说明它为什么是错的而不只是慢。

### 6.6 对构想的逐项裁决

| # | 构想字段 | 裁决 | 理由（含代码依据） | 落点 |
| :- | :-- | :-- | :-- | :-- |
| 1 | `DegreeSynopsis.{vertex_count, edge_count, avg_degree, max_degree}` | ✅ 保留，但降为**派生** | `csr.rs:67 TemporalCsrStats` 已有这四项；`cbo.rs:74 mean_degree()` 也有。持久化即制造第二真相 | 不落盘，`stats()` 现算 |
| 2 | `DegreeSynopsis.{p50,p95,p99}_degree` | ⚠️ 改造为**直方图派生** | 分位标量不可跨段合并（§6.7.2）；现有 `degree_histogram`（`csr.rs:74`）是 log2 桶，只能给近似分位——存成标量会制造「精确」假象 | 存 log2 直方图，读时算分位 |
| 3 | `hub_threshold` | ⚠️ 改为**构建期策略参数** | 它不是数据属性而是策略；必须与 `TraversalBudget::max_degree`（`traversal.rs:38`）对齐，否则 planner 与执行器两套阈值 | 写入 L3 header，使 `hub_count` 可解释 |
| 4 | `NeighborhoodSynopsis{dst_bloom}` | ⚠️ 改造：优先 **XOR filter**，且**只用于段内** | 段不可变 ⇒ XOR（~1.23 B/key、无假阴性）优于 Bloom（~10 bit/key）；但 XOR 不可 OR 合并，Bloom 可。跨段「是否存在 dst」用逐段 early-exit | 独立 L3 文件，不进 manifest |
| 5 | `TemporalSynopsis{active_days, p50/p95/peak/day}` | ✅ **最高价值，但需重定义** | 这是 Neo4j 完全帮不上、gtvdb 又必须有的维度（bitemporal）。单段 O(E) 可建；但标量分位仍不可合并 → 改存「日边数 log2 直方图 + 活跃日 RLE」 | 段级 L3，flush 时构建 |
| 6 | `ReachabilitySynopsis{reachable_estimate, two_hop, hll_blob}` | ❌ **删除段级版本** | 可达性不可按段分解：路径跨段，段内 HLL 无法上卷 | 降级到全局 L3，P4 |
| 7 | `ComponentSynopsis{component_count, largest_component, largest_scc}` | ❌ 同上 | 「段内分量数」没有查询能用：全局分量可由多段拼成，段内最大分量 ≠ 全局最大分量 | 降级到全局 L3，需 §6.7.1 闸门 |
| 8 | `ReverseGraphState{None, Materialized, Partial, Lazy}` | ⚠️ 改为 `reverse_of: Option<SegmentId>` | 反图在 gtvdb 里就是另一个 `TemporalCSR`（`csr.rs:840 transpose()`），理应作为**段**存在：不可变、content-address、可 GC、随 manifest 版本化。用枚举会把它排除在生命周期之外 | manifest 加一列 + 段级 `has_reverse` |
| 9 | `TraversalCostSynopsis{estimated_bfs_cost, ..., preferred_direction}` | ❌ **不要存** | `preferred_direction` 依赖运行时 frontier 比例 p：`csr.rs:698-700` 判据为 `(1-p) < p²·avg_degree`。p 是查询期变量，存成枚举必然给错计划。Neo4j 的 `GraphStatistics` 里也一个 cost 字段都没有，代价在 `DefaultCostModel` 里算 | 只存输入，cost 用纯函数算（§6.8.3） |

**汇总：保留 3（其中 2 项降为派生）／改造 4／删除 2。**

### 6.7 三个结构性错误（比字段取舍更重要）

#### 6.7.1 粒度错配：把全局属性存成段属性

构想 §6.6.2 的 `SegmentTopologyMeta` 同时装了 `degree_synopsis`（**可加**）与 `largest_scc`（**全局**）。这两类量对「段」的依赖方式根本不同：

- 度可加：`deg_global(v) = Σ_seg deg_seg(v)`；
- SCC 不可加：`SCC(G1 ∪ G2) ≠ f(SCC(G1), SCC(G2))`。

后果不是「不精确」，而是**静默错误**：planner 会据此剪掉一条真实存在的路径。

**可判定的闸门**：§7.2 阶段 A 已计划给 `SegmentEntry` 加 `node_id_min / node_id_max`。若所有段区间**两两不相交**，则不存在跨段边，全局分析才成立。O(S log S)，且在 manifest 内存里完成：

```rust
fn topology_global_safe(segs: &[SegmentEntry]) -> bool {
    let mut v: Vec<(u64, u64)> = segs.iter().map(|s| (s.node_id_min, s.node_id_max)).collect();
    v.sort_unstable();
    v.windows(2).all(|w| w[0].1 < w[1].0)   // 闭区间不重叠
}
```

注意方向：**不相交是充分条件，不是必要条件**。相交时只能说「不敢保证」，回退运行时计算——这正是 Neo4j「缺文件就重建 / 不算」的姿态。

#### 6.7.2 可合并性未分层

L3 的每个量必须按「能否跨段合并」分三类，因为**合并方式决定粒度与能否派生**：

| 类别 | 例子 | 跨段合并 | 存放 | 精度 |
| :-- | :-- | :-- | :-- | :-- |
| 可加（exact） | node/edge count、degree、`hub_count` | 求和 | 段级 | 精确 |
| 可并（sketch） | log2 度直方图、日边数直方图、Bloom | 按桶相加 / 按位 OR | 段级 | 有界近似 |
| 不可并（global） | p50/p95 标量、SCC、连通分量、可达性、XOR filter | 不能 | 全局，或禁止 | 全量重算 |

构想把三类混在同一个 struct，直接导致 §6.7.1 的错误与 §6.6 的第 2/6/7 项问题。

#### 6.7.3 把派生量当事实存

`estimated_bfs_cost`、`preferred_direction`、`p50_degree` 都能从输入算出来。存它们有三重代价：占空间、引入陈旧、**掩盖公式变更**（改了成本模型但旧值还在）。

Neo4j 的选择很干脆：`GraphStatistics` 只暴露计数与选择性，代价由 `DefaultCostModel` 算。gtvdb 照此办理——**L3 存输入，`cbo.rs` 存公式**。

#### 6.7.4 附带：L3 必须是可丢弃缓存

构想把 L3 写成「Publish 后不可修改」的权威物。正确姿态是 **derived & discardable**（Neo4j 缺文件即重建）。由此三条硬约束：

- 删除 `<root>/topology/` 后，任何查询**结果不变**，只是变慢；
- L3 构建**不在 flush 关键路径**（T1 例外，见 §6.9）；
- L3 不写 WAL、不进 manifest 的强一致边界。

### 6.8 修订后的 L3 对象模型

#### 6.8.1 段级：只放可加 / 可并的量

```rust
/// 独立文件：<root>/topology/<segment_id>-<checksum8>.gtvt
/// 定宽 header + 定宽数组，与 GTVSeg 同一套 mmap 约束（可 cast_slice）。
#[repr(C)]
pub struct TopologyHeader {
    pub magic: [u8; 8],          // b"GTVTOP01"
    pub version: u32,
    pub flags: u32,              // bit0 dst_filter, bit1 temporal, bit2 reverse
    pub segment_id: u64,
    pub segment_checksum: u64,   // 段 data_checksum 前 8 字节：内容寻址，防错配
    pub node_count: u64,
    pub edge_count: u64,
    pub max_degree: u32,
    pub hub_threshold: u32,      // 与 TraversalBudget::max_degree 对齐
    pub hub_count: u32,
    pub histogram_len: u32,      // log2 桶数，通常 32
    pub temporal_buckets: u32,   // 0 = 未构建
    pub dst_filter_bytes: u64,
    pub build_epoch: i64,        // system time
    pub reserved: [u64; 4],
}
// 紧随其后：degree_hist:   [u32; histogram_len]
//           temporal_hist: [u32; temporal_buckets]
//           temporal_rle:  [(i64 from, i64 to, u32 edges); n]
//           dst_filter:    [u8; dst_filter_bytes]
```

要点：

- `segment_checksum` 让 L3 **内容寻址**：compaction 产生新段 → checksum 变 → 旧 L3 自然不再被引用，**构想 §6.6.8 的 `Stale` 状态可以整段删掉**；
- `hub_threshold` 入 header，`hub_count` 才有定义；
- 业务时间用**直方图 + RLE**，不用三个标量分位。

#### 6.8.2 快照级：才放全局量

```rust
pub struct SnapshotTopologyMeta {
    pub generation: u64,              // == manifest.checkpoint_txn，单调且已持久化
    pub manifest_system_ts: i64,
    pub segment_count: u64,
    pub node_count: u64,
    pub edge_count: u64,
    pub degree_hist: Vec<u32>,        // 段级直方图求和
    pub node_interval_disjoint: bool, // §6.7.1 闸门
    pub components: Option<Components>,  // 仅 disjoint 时 Some
    pub scc: Option<Scc>,                // 仅 disjoint 时 Some
}
```

构想 §6.6.4 的 `TopologyGeneration { generation_id }` 与「Generation 1 → {S1,S2}」的树形结构可整个删除：**`manifest.checkpoint_txn` 已经是单调、持久、与段清单绑定的版本号**。另建一套 generation 只会产生第二个真相。

#### 6.8.3 代价模型：纯函数，不落盘

```rust
/// 与 csr.rs:690-700 同源，但把 p（frontier 比例）显式暴露为参数。
pub fn bfs_cost(topo: &SnapshotTopologyMeta, frontier_frac: f64, hops: u32) -> u64 {
    let avg = topo.edge_count as f64 / topo.node_count.max(1) as f64;
    // push: p·V·avg / hop；pull: V(1-p)/p（提前退出）
    let per_hop = (frontier_frac * topo.node_count as f64 * avg)
        .min(topo.node_count as f64 * (1.0 - frontier_frac) / frontier_frac.max(1e-9));
    (per_hop * hops as f64) as u64
}
```

`preferred_direction` 于是不再是元数据，而是 `bfs_cost(push) < bfs_cost(pull)` 的一次比较——与现有 `DirectionMode::Auto` 完全一致，不会出现「元数据说 pull、运行时却已 push」的分裂。

### 6.9 构建时机与成本分级（修订）

| Tier | 内容 | 时机 | 成本 | 阻塞 flush？ |
| :-- | :-- | :-- | :-- | :-- |
| T0 | node/edge count、degree、`hub_count` | **写入期增量**（MemSegment 计数器） | O(1)/边 | 否 |
| T1 | log2 度直方图、日边数直方图 + RLE | flush 时随 CSR 一次扫描 | O(V) + O(E) | 是（已在线性扫描内） |
| T2 | dst XOR filter | flush 后异步 | O(E) | 否 |
| T3 | 全局分量 / SCC | 显式触发（`ANALYZE` 等价物），且仅 disjoint | O(V+E) | 否 |

与构想的唯一实质差别在 T0/T1 的切分：构想把 `Edge Count / Vertex Count` 列进 flush 的 O(E)，实际上它们**写入时就已经知道**，flush 只需读计数器。这不是微优化，而是消除了 flush 时的一次全表扫描。

### 6.10 三方在 L3 上的对照

| 维度 | MongoDB | OceanBase | Neo4j | gtvdb 判定 |
| :-- | :-- | :-- | :-- | :-- |
| 计数（节点/边/度） | 无（`$collStats` 现算） | 无（优化器统计非拓扑） | ✅ 事务内 O(1) + 全表重建兜底 | 学 Neo4j，落 T0 |
| 度分布 | 无 | 无 | ❌ 仅 GDS，按需、按图缓存 | 自研 T1（已有 `degree_histogram`） |
| 连通性 / SCC | 无 | 无 | ❌ core 完全没有 | 自研 T3，且**只做全局** |
| 可达性估计 | 无 | 无 | ❌ 无 | 推迟到 P4 |
| 反图 | 无 | 无 | 按 `(group, direction)`，度数可外置 | 反图 = 段 + `reverse_of` |
| 时间维度 | 桶 `control.min/max` | 多版本 + 全局快照点 | ❌ 非时序库 | 自研 T1 `TemporalSynopsis` |
| 失效判定 | 无（DDL 即改） | compaction 生成新 SST | 偏差 0.75 + 最小间隔 10s | 学 Neo4j 的判据 |
| 缺失时行为 | — | — | 缺文件重建、继续启动 | **必须**：降级不报错 |

### 6.11 验收要点（L3 专属）

| 验收项 | 方法 |
| :-- | :-- |
| 可丢弃性 | 删除 `topology/` 全目录后 `cargo test --workspace` 全绿，查询结果逐行一致（只允许变慢） |
| 与全量重算一致 | 属性测试：随机段集 + 随机 `t`，L3 派生值与直扫 CSR 对比；直方图精确相等，分位相对误差 < 5% |
| 内容寻址 | compaction 后新段 L3 的 `segment_checksum` 与 manifest 一致；旧 L3 不再被引用且可删 |
| 闸门正确 | `node_interval_disjoint == false` 时 `components` / `scc` 必须为 `None`，planner 走运行时路径 |
| 冷启动不变 | `open()` 不读 `topology/`；`bench_segment` 的 cold-open 指标不退化 |
| 空间上限 | `Σ topology_bytes / Σ segment_bytes < 5%`（XOR filter 是主要项），超限则不建 T2 |

### 6.12 L3 里程碑（建议插在 §7 之后）

1. **P3.1**：T0 + T1（计数器 + 度直方图 + 时间直方图），`gtv_topology()` UDTF 暴露；`GraphStats` 改为从 L3 读，`set_graph_stats()` 保留为手工覆盖。
2. **P3.2**：`reverse_of` 进 manifest，`transpose()` 产物落成正式段。
3. **P3.3**：T2 XOR filter + 段内 early-exit 剪枝。
4. **P3.4**：T3 全局分量 / SCC（带 `node_interval_disjoint` 闸门），`ANALYZE` 式显式触发。

**不要在 P3.1 之前做 `ReachabilitySynopsis` / `ComponentSynopsis`**：它们依赖 §7.2 的 `node_id_min/max`，而那一项本身还没落地。

---

## 7. 修订后的「第 2 项」实施方案

### 7.1 四个标量，而不是两个（最容易踩的坑）

`TemporalCSR` 已有的两个标量是**「整段全命中」窗口**，不是「可能命中」包围盒：

```rust
// csr.rs:206-214, 386
max_valid_from = max(所有边的 valid_from)
min_valid_to   = min(所有边的 valid_to)
all_active_at(t) = t >= max_valid_from && t < min_valid_to   // 所有边都活跃
```

新增的两个标量是**「可能命中」包围盒**：

```rust
valid_from_min = min(所有边的 valid_from)   // 段内最早的起点
valid_to_max   = max(所有边的 valid_to)     // 段内最晚的终点（open edge 用 i64::MAX）
```

**混用这两组会得出错误剪枝**：`[max_valid_from, min_valid_to)` 是所有边有效期的**交集**，`[valid_from_min, valid_to_max)` 是**并集**。剪枝必须用并集，整段短路必须用交集。

半开区间 `[valid_from, valid_to)` 下的判定式：

| 场景                    | 判定                                          | 动作                                           |
| :-------------------- | :------------------------------------------ | :------------------------------------------- |
| `as_of(t)` 可跳段        | `t < valid_from_min \|\| t >= valid_to_max` | 完全不打开该段                                      |
| `[T1,T2)` 与段重叠        | `valid_from_min < T2 && valid_to_max > T1`  | `entries_overlapping(T1,T2)`                 |
| `as_of(t)` **整段全命中**  | `max_valid_from <= t < min_valid_to`        | `active_edge_count(t) = edge_count`，O(1) 不扫边 |
| `[T1,T2)` **整段区间全命中** | `[T1,T2) ⊆ [max_valid_from, min_valid_to)`  | 每个时间桶的边数恒为 `edge_count`，可直接生成聚合结果            |

第 4 行正是 MongoDB `wholeBucketFilter` / `$group` 用 control 字段跳过 unpack 的对应物——**概念借 Mongo，实现落 OB 式标量**。

### 7.2 两阶段落地：先 manifest，后 header

**阶段 A（最小切口，不改 GTVSeg v1 二进制格式，建议先做）**

只改 `SegmentEntry`（`crates/gtv-storage/src/store.rs`）与 `gtv_segments()`（`crates/gtv-engine/src/segment.rs`）：

```rust
pub struct SegmentEntry {
    // ...现有字段不变
    pub valid_from_min: i64,
    pub valid_to_max: i64,
    pub max_valid_from: i64,   // = GTVSeg header 的 max_valid_from
    pub min_valid_to: i64,     // = GTVSeg header 的 min_valid_to
    pub node_id_min: u64,
    pub node_id_max: u64,
}
```

`flush_at()` 在 `MemSegment::to_csr()` 之后、`encode_segment()` 之前从 CSR 取这四个值（`csr.columns()` 已有全部边列，一次线性扫描即可，O(E) 但只在 flush 时发生）。

```rust
pub fn entries_at_business(&self, t: i64) -> Vec<&SegmentEntry>;
pub fn entries_overlapping(&self, t1: i64, t2: i64) -> Vec<&SegmentEntry>;
pub fn entries_at(&self, business_t: i64, system_t: i64) -> Vec<&SegmentEntry>;
// 最后一个是双时间交集：entries_at_system(system_t) ∩ entries_at_business(business_t)
```

`gtv_segments()` 同步扩列：`valid_from_min / valid_to_max / all_active_from / all_active_to / node_id_min / node_id_max`。

> ✅ **已落地（2026-11-09）**：字段与选择器均在 `gtv-storage`，SQL 侧在 `gtv_segments()`；实测见 [metadata-benchmark.md](metadata-benchmark.md)，状态见 §11。

**为什么阶段 A 足够且更稳**：

- manifest 本来就在 `open()` 时整体读入内存，段选择层可以**完全不打开 `.seg` 文件**就剪枝，不影响 O(1) cold open；
- manifest 由 `write_atomic` 发布，与 P1「写新段 → 发布 manifest → 截断 WAL」的崩溃一致性顺序完全一致，无需新增事务边界；
- 不动 `VERSION`、不动 `header_checksum` 覆盖范围、不动 `col_dir`，P0/P1 的 399 个测试与既有 `.seg` 文件全部兼容。

**阶段 B（P2，与分层 skip index 一起做）**

把同样的标量写进 GTVSeg header 的 `reserved[3776]`（建议 offset 320 起，64 B 对齐），并纳入 `header_checksum`；让「脱离 manifest 单开一个 `.seg`」也能自描述。这需要版本升级（建议 `VERSION = 2`，reader 同时支持 v1 的旧 checksum 计算），走 **渐进重写**（OceanBase 借鉴点），不要求停机全量转换。

> 不要为了早点拿到 header 自描述而在阶段 A 就改二进制格式：收益是「单文件可用」，代价是版本兼容、校验范围变更和一批错误路径测试，风险边界不划算。

### 7.3 接线第 1 / 3 项（顺序不变）

1. **统一元数据出口**：`gtv_tables()` / `gtv_snapshots()` UDTF 直接把 `FsCatalog` 的 `TableMeta / SchemaRecord / Snapshot` 暴露成 Arrow 表；`register_batches` 可选写回 catalog（对应 §6.2 单一权威）。
2. **时间谓词下推**：实现 `SegmentTableProvider`（`supports_filters_pushdown`）承接 `ScanFilter`（当前无生产调用方）；新增 `range_agg(T1,T2,bucket)` / `active_edge_count(t)`，先用 `entries_overlapping` 选段，再用「整段全命中」短路，最后才逐边过滤。
3. **图漫步**：反图、度分布、时间密度要作为段元数据标记供 CBO 选方向；具体口径见 §6（`reverse_of: Option<SegmentId>` + 度/时间直方图），此处不再另立设计。

### 7.4 验收与测试要点

| 验收项      | 方法                                                                                                                 |
| :------- | :----------------------------------------------------------------------------------------------------------------- |
| 半开区间边界正确 | 单测覆盖 `t == valid_from_min`、`t == valid_to_max`、`T2 == valid_from_min`、`T1 == valid_to_max`、空段、`i64::MAX` open edge |
| 剪枝不漏结果   | 属性测试：随机生成 N 个段 + 随机 `[T1,T2)`，比对 `entries_overlapping` 结果与「全开所有段逐边枚举」完全一致                                          |
| 双时间交集    | `entries_at(business_t, system_t)` 与 `graph_at(system_t)` 后按 business_t 过滤结果一致                                     |
| 崩溃一致性    | 新增字段后重跑 P1 六个用例：断尾 WAL、checkpoint 幂等、孤儿段、历史段复用率、`as_of_system` 基线一致                                                |
| 性能       | `bench_segment` 增加「段剪枝命中率」「整段全命中短路率」两个指标；确认剪枝后 cold open 不退化                                                       |
| 格式兼容     | 阶段 A 用 v1 文件回归；阶段 B 需 v1/v2 双读测试                                                                                   |

**dev / prod 差异**：dev 可继续单写者、无跨进程锁（`FsCatalog` 现状）；prod 若要支持多进程读同一 store，至少给 manifest 加共享/排他文件锁（`flock`），并把段 GC 与 mmap 引用计数挂钩——这是 §6.2「ident 延迟回收」的最低实现，不引入 WiredTiger 那套。

### 7.5 明确不做

- 不把元数据做成 BSON / JSON 热路径结构；manifest 保持 JSON 仅限**冷路径**（每次 flush 一次）。
- 不引入压缩 / 编码到 GTVSeg 热层（架构红线，见 [storage-mmap-design.md](storage-mmap-design.md) §7）。
- 不实现 drop-pending timestamp / 两阶段 DDL / PIT catalog 重放。
- 不在 `neighbor_iter` 热循环里做多路归并或元数据查找。
- 不因为「借了 MongoDB 概念」就把 bucket 化 / measurement 分组引入 CSR。
- 不把 SCC / 连通分量 / 可达性做成**段级**元数据（必须全局，且带 §6.7.1 闸门）。
- 不持久化 cost / `preferred_direction` / 分位标量——它们都是派生量（§6.7.3）。

---

## 8. 风险与常见坑

1. **交集 / 并集混用**（最高危）：`max_valid_from/min_valid_to` 是交集（全命中窗口），`valid_from_min/valid_to_max` 是并集（剪枝包围盒）。用前者剪枝会漏数据，用后者短路会多算。
2. **`i64::MAX` sentinel**：open edge 的 `valid_to` 用 `i64::MAX`，`min_valid_to` 与 `valid_to_max` 的取值语义要写进注释与测试，避免 `t >= valid_to_max` 把开放边误判为过期。
3. **manifest 与 header 双份一致性**：阶段 B 之后以 manifest 为权威、header 为自描述冗余；`open_entry()` 现有的 `data_checksum` 校验要同步覆盖新增字段，否则静默错位。
4. **`ScanFilter` 的半开区间**：现有 `matches()` 用 `file.event_time_max < lo` / `file.event_time_min > hi`，与 `[T1,T2)` 语义对齐；下推时不要改成闭区间。
5. **单写者假设**：`FsCatalog` 明确无跨进程锁（`store.rs:22-23`），manifest 与 WAL 尚未统一事务边界；多进程场景先加锁再谈并发写。
6. **过度工程**：MongoDB 的 catalog 强度来自它要支撑复制、分片、PIT 事务；gtvdb 若照搬，会把 90% 复杂度花在不存在的需求上。
7. **L3 的段级全局量**（最高危，L3 专属）：把 `largest_scc` / `component_count` 按段存，会在多段图上**静默剪掉真实路径**。必须由 `node_interval_disjoint` 闸门拦住，`None` 时回退运行时。
8. **分位标量的假精确**：`p95_degree: u32` 一旦落盘就再也不能随段合并更新，且与 log2 直方图派生值可能不一致。存直方图，读时分位。
9. **L3 变成事实源**：只要有任何查询的正确性依赖 L3（而非「L3 只影响快慢」），它就必须进事务边界，P1 的「写段 → 发布 manifest → 截断 WAL」顺序会被打破。删掉 `topology/` 必须只是变慢。

---

## 9. 证据来源

**Neo4j（`github.com/neo4j/neo4j`，默认分支 `2026.10`，2026-11-09 核对）**

- `community/cypher/planner-spi/src/main/scala/org/neo4j/cypher/internal/planner/spi/GraphStatistics.scala` — 仅 7 个方法；`nodesWithLabelCardinality` 注释 *"does NOT return the number of nodes anymore"*；`getHistograms` 注释 *"only implement this for integration testing purposes"*；**无 degree / SCC / reachability / cost 字段**。
- `community/record-storage-engine/src/main/java/org/neo4j/kernel/impl/store/record/RelationshipGroupRecord.java` — 字段 `type/next/firstOut/firstIn/firstLoop/owningNode` + `externalDegreesOut/In/Loop`；**无 count 字段**；`prev` 不持久化。
- `community/record-storage-engine/src/main/java/org/neo4j/internal/counts/GBPTreeRelationshipGroupDegreesStore.java` — `TYPE_DEGREE = 3`；`degreeKey(groupId, direction) = new CountsKey(TYPE_DEGREE, groupId << 2 | direction.id(), 0)`；`degree(groupId, direction, ctx)`；key 布局注释 `[gggg...dd]`。
- `community/record-storage-engine/src/main/java/org/neo4j/internal/counts/DegreesRebuildFromStore.java` — 类注释 *"Scans the store and rebuilds ... if the file is missing"*；`log.warn("Missing relationship degrees store, rebuilding it.")`；`PrepareCacheStage` / `CalculateDegreesStage` / `cache.writeTo(updater)`；1024 路 `StripedLatches`；堆外 `LongArray`。
- `community/record-storage-engine/src/main/java/org/neo4j/kernel/impl/store/CountsComputer.java` — `implements CountsBuilder`；`NodeCountsStage` + `RelationshipCountsStage` 全量重扫。
- `community/record-storage-engine/src/main/java/org/neo4j/kernel/impl/store/stats/RecordDatabaseEntityCounters.java` — `nodes()` 取 `idGenerator.getHighId()`（估算）vs `allNodesCountStore()` 取 counts store（精确）；`relationshipCount(ANY_LABEL, ANY_RELATIONSHIP_TYPE, ANY_LABEL)` 三元组。
- `community/graph-algo/src/main/java/org/neo4j/graphalgo/impl/` — 仅 `path` / `shortestpath` / `util`，**无 SCC / 连通分量 / 可达性**。
- 官方手册 [Statistics and execution plans](https://neo4j.com/docs/operations-manual/5/performance/statistics-execution-plans) — 计数随标签增删即时更新；`db.index_sampling.update_percentage = 5`；`dbms.cypher.statistics_divergence_threshold = 0.75`（`abs(a-b)/max(a,b)`）；`dbms.cypher.min_replan_interval = 10s`；阈值约 7h 衰减至 10%。
- 官方 GDS 文档 [Listing graphs](https://neo4j.com/docs/graph-data-science/2026.03/management-ops/graph-list/) — `degreeDistribution` *"can be fairly time-consuming ... cached per graph"*，即按需计算而非常驻元数据。

**MongoDB（`github.com/mongodb/mongo`，master，2026-11-08 核对）**

- `src/mongo/db/storage/mdb_catalog.h` — `_mdb_catalog` 表、`RecordId` Catalog ID、`{ident, idxIdent, md, ns}` BSON entry、`getCursor` / `getAllCatalogEntries` / `putUpdatedEntry`。
- `src/mongo/db/shard_role/shard_catalog/durable_catalog.h` — durable catalog 解析视图、`putMetaData` / `createCollection` / `dropCollection` / `importCollection`。
- `src/mongo/db/shard_role/shard_catalog/collection_catalog.h` — `immutable::*` 容器、COW `write()`、`get()/stash()/latest()`、`establishConsistentCollection(..., readTimestamp)`、`HistoricalCatalogIdTracker`。
- `src/mongo/db/shard_role/shard_catalog/catalog_control.h` — `closeCatalog` / `openCatalogAfterRollbackToStable`。
- `src/mongo/db/storage/storage_engine.h` — `loadMDBCatalog` / `dropIdentTimestamped` / `addDropPendingIdent` / `TimestampMonitor` / stable timestamp 与 recovery。
- `src/mongo/db/storage/wiredtiger/wiredtiger_kv_engine.h` — ident / URI、repair / import、`getAllIdents`。
- `src/mongo/db/timeseries/README.md` — bucket schema（`control.min/max`）、1000 条上限、`granularity`、`BucketCatalog` / Stripe。
- `src/mongo/db/timeseries/bucket_catalog/bucket_catalog.h` — 32 stripes、open/archived/idle bucket 结构。
- `src/mongo/db/query/timeseries/README.md` — control 谓词下推、`wholeBucketFilter`/`eventFilter`、bounded `CLUSTERED_IXSCAN`、`$group` min/max 改写、谓词按 `bucketMaxSpanSeconds` 放宽。
- `src/mongo/db/shard_role/shard_catalog/README_sharding_catalog.md` — `config.chunks.min/max`、`CatalogCache`、ownership filter。
- 二级来源：[淘宝数据库内核月报 2020/04《MongoDB 大量集合启动加载优化原理》](http://mysql.taobao.org/monthly/2020/04/08) — `_mdb_catalog.wt`、`WiredTiger.wt` / `WiredTiger.turtle` 三级引导关系（当前 master 未逐行复核该物理文件名）。

**OceanBase / gtvdb（本仓）**

- [docs/oceanbase-storage-lessons.md](oceanbase-storage-lessons.md) §1.4 / §4③ / §5.1 — Skip Index 上卷、宏块复用、渐进重写、定宽元数据可零拷贝。
- [docs/storage-mmap-design.md](storage-mmap-design.md) §2 / §3.2 / §5.3 / §8 / §9 — GTVSeg 格式、压缩与 zero-copy 互斥、「LSM 写 CSR 读」、P0/P1 验据。
- `crates/gtv-core/src/csr.rs:89-128, 206-249, 385-386, 479, 888` — `max_valid_from` / `min_valid_to` / `chunk_max_valid_to` 语义与 `all_active_at()`。
- `crates/gtv-storage/src/store.rs:73-101, 325-407, 429-459` — `SegmentEntry` / `Manifest` / `flush_at()` / `entries_at_system()` / `graph_at()`。
- `crates/gtv-storage/src/segment.rs:19-53, 87-147, 296-433` — GTVSeg v1 header、`col_dir[8]`、header/data checksum、reserved 区。
- `crates/gtv-catalog/src/store.rs:22-23, 48-115, 493-505` — `FsCatalog` 单写者假设、`TableMeta`、`ScanFilter::matches()`、`scan()`。
- `crates/gtv-catalog/src/manifest.rs:48-101` — `DataFile.event_time_min/max` / `ColumnStat` / `Snapshot`。
- `crates/gtv-engine/src/segment.rs:52-115` — `gtv_segments()` 现有 8 列。
- `crates/gtv-core/src/csr.rs:67-75, 415` — `TemporalCsrStats { node_count, edge_count, chunk_count, max_degree, degree_histogram }`，`hist[k]` = `floor(log2(degree)) == k`（L3 的 DegreeSynopsis 已存在，只是未持久化）。
- `crates/gtv-engine/src/cbo.rs:47-81` — `GraphStats { node_count, edge_count, max_degree, degree_histogram, temporal_active_ratio }`、`with_active_ratio()`、`mean_degree()`（L3 的现有落点）。
- `crates/gtv-engine/src/context.rs:729` — `set_graph_stats()`，当前需手工注入。
- `crates/gtv-core/src/csr.rs:663-667, 690-700` — `avg_degree` 现算；push/pull 判据 `(1-p) < p²·avg_degree`（证明 `preferred_direction` 不可预存）。
- `crates/gtv-core/src/csr.rs:840` — `transpose()` 物化反图（L3 的 `reverse_of` 应指向成段后的产物）。
- `crates/gtv-core/src/traversal.rs:7, 25-41` — `VisitedSet` 每节点 4 字节；`TraversalBudget::max_degree`（`hub_threshold` 应对齐它）。
- `crates/gtv-storage/src/store.rs:75-89` — `SegmentEntry`（原 10 字段；现已按 §7.2 增加 6 个 L2 时间盒字段）。
- 全仓 grep `ScanFilter` — 仅定义、`FsCatalog::scan()` 与 `gtv-catalog/tests/catalog.rs` 两处测试调用，**无生产调用方**。

---

## 10. 一句话交付

**L2 按 OceanBase 做，且先落 manifest 不改 GTVSeg 二进制；L1 只借 MongoDB 的「单一权威 / COW 快照 / ident 延迟回收」三条原则；L3 借 Neo4j 的三条机制（计数增量维护、大拓扑外置成独立可丢弃文件、陈旧度用偏差阈值判定），并按它砍掉构想里的段级 SCC / 可达性 / 预存 cost —— 保留度直方图与时间直方图，因为只有「可加 / 可并」的量才配进段级元数据。**

---

## 11. 落地状态（2026-11-09）

### 11.1 L2 阶段 A：已完成

- `crates/gtv-storage/src/segment_meta.rs`：`SegmentMeta` + `PruneOutcome`，六组判定式（`overlaps` / `may_contain` / `all_active_at` / `fully_covers` / `classify` / `classify_point`）。
- `SegmentEntry` 增加 6 个 `Option` 字段（`valid_from_min` / `valid_to_max` / `max_valid_from` / `min_valid_to` / `node_id_min` / `node_id_max`），`flush_at` 从 CSR 一次线性扫描写入。
- `SegmentStore` 新增 `entries_at_business` / `entries_overlapping` / `entries_at` / `business_span` / `prune_summary`。
- **旧 manifest 兼容**：字段缺失时谓词一律保守选中；有测试验证「删掉 6 个 key 后仍 100% 返回」。
- `gtv_segments()` 扩 7 列；`register_segment_store` 从 manifest 注入时间盒（`prunable` 标记是否已知）。

### 11.2 L1：已完成（比 §7.3 的第 1 项更进一步）

除 `gtv_tables()` / `gtv_snapshots()` 外，增加 `gtv_files('t' [, 'snap'])` 与 `GtvContext::sync_catalog()`。后者按**快照文件清单**注册 DataFusion 视图，使「表在 SQL 里可见」由 catalog 决定，而不是由调用方的注册顺序决定 —— 这是 §6.2「单一权威」的最小实现。

### 11.3 实测（详见 [metadata-benchmark.md](metadata-benchmark.md)）

| 指标 | 实测 |
| :-- | :-- |
| 冷启动 vs 段数 | 49 段 2.90 ms / 586 段 2.78 ms —— 与段数无关 |
| 剪枝决策 | 100 段 0.10 µs；586 段 0.7–1.8 µs（1–3 ns/段） |
| 区间 / 点事件负载 | 跳过 97.8%–99.7% 的段，扫描量降到 1/40–1/529 |
| 静态图对照 | 跳过 0%，`FullHit` 正确命中，无过度剪枝 |
| 开放式 as-of | 跳过 49%–88%，收益主要来自 `FullHit`（147/149） |

### 11.4 基准暴露的新问题（原设计未覆盖）

**GTVSeg 是稠密 CSR，每段都带 `node_count` 个 `offsets` 与 `chunk_index`。** 当一段的边数远小于节点数时，元数据开销压过数据本身：`comments` 按月分 586 段是 69 MB（1,439 B/edge），按年分 49 段只有 7 MB（145 B/edge），同样的 50,303 条边、同样的 14,559 个节点。

结论：**段粒度的选择要以「每段边数 ≳ 节点数」为约束**，而不只是时间窗口。根治手段是 P2 的「段内节点区间窄化」——`node_id_min/max` 已算好落在 manifest 里，正是为此准备的输入。这条应写进 P2 的验收，而不是留到调优阶段。
