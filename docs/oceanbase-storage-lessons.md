# OceanBase 存储压缩与 LSM-Tree 分析：能否用于 gtvdb

> 2026-11-08｜基于 OceanBase 官方文档、社区技术文章与 `oceanbase/oceanbase` 源码目录，对照 gtvdb 当前代码事实。
> 配套文档：[docs/storage-mmap-design.md](storage-mmap-design.md)（GTVSeg 零拷贝段设计）。本文是**适用性评估**，不是替代。
> 状态：设计评估（2026-11-08）。其 P1 建议已在同日落地，实现与验据见 [storage-mmap-design.md §9](storage-mmap-design.md)；P2/P3 建议仍为提案。

---

## 0. 结论先行

**不能整套搬，但有 4 项值得拿过来，其中 2 项能直接改进现有设计。**

一句话理由：**OceanBase 是「磁盘 I/O 瓶颈」的系统，压缩是用 CPU 换 I/O；gtvdb 的热路径是「内存带宽 / CPU cache 瓶颈」，压缩是用 CPU 换磁盘空间——这笔账在热层是纯亏的。**

但把 OceanBase 的「压缩」和「LSM 架构」拆开看，架构部分几乎全都能用：

| OceanBase 机制 | 用于 gtvdb 热层 (L0/L1 GTVSeg) | 用于 gtvdb 冷层 (L2 Parquet) | 用于写路径 / compaction |
|---|---|---|---|
| **两级压缩**（encoding + zstd） | ❌ **禁止** | ✅ 已在用，可加强 | ❌ 不进写路径 |
| **LSM 分层 + 后台合并** | ⚠️ 只取「读侧退化成单段」 | — | ✅ **核心借鉴** |
| **写路径与压缩解耦** | — | ✅ | ✅ **核心借鉴** |
| **宏块复用 / 渐进重写** | — | — | ✅ **核心借鉴**（时间分区 = 天然宏块） |
| **分层 Skip Index** | ✅ **可直接补进 GTVSeg** | ✅ 与 Parquet statistics 对齐 | — |
| **谓词下推 / SIMD 过滤** | ✅ 在未编码定宽数组上做，更简单 | ⚠️ 当前未接线 | — |
| **BloomFilter cache** | ⚠️ 图遍历收益低 | — | — |
| **block/row/fuse row cache 体系** | ❌ **不需要**（mmap 后 page cache 即是） | — | — |

最关键的工程约束写在 §5.3：**gtvdb 只能用「半套 LSM」——写侧吸收 LSM 的增量与后台合并，读侧必须在 compaction 后退化成单一连续 CSR。把 LSM 的多路归并放进 BFS 热循环，会把 `offsets[src]..offsets[src+1]` 的 O(1) 邻接定位毁掉。**

---

## 1. OceanBase 的存储模型（事实梳理）

### 1.1 LSM-Tree 三层结构与 Tiered + Leveled 合并

OceanBase 是「基线 + 增量」的 LSM 引擎：内存 `MemTable` 可写，磁盘 `SSTable` 只读且连续存放。SSTable 分三层 [^1][^2]：

```
MemTable (内存, 可写)
   │  freeze → mini compaction (转储)
   ▼
L0  Mini SSTable   (tiered，可有多个，直接由冻结 MemTable flush)
   │  数量超阈值 → minor compaction
   ▼
L1  Minor SSTable  (leveled，通常仅一个)
   │  每日低峰 / 阈值触发 → major compaction
   ▼
L2  Major SSTable  (基线，通常仅一个，全局快照点)
```

V4.x 明确采用 **Tiered & Leveled 混合策略**：L0 用 tiered，L1/L2 用 leveled [^3]。Compaction 的实质是「按 Rowkey 递增的多路归并排序」[^3]。

**OceanBase 特有的两点**：
- **全局快照点**：major compaction 为整个租户选定一个一致快照，所有分区基于该快照生成基线，从而「提供天然的数据校验点」，可做多副本与主索引表的物理校验 [^2][^4]。
- **Medium / Meta Major Compaction**：支持分区级合并与「不含多版本/未提交事务数据」的特殊基线，不必每次全租户 [^3]。

### 1.2 两级块结构：2MB 宏块 / ~16KB 微块

| | 宏块 Macroblock | 微块 Microblock |
|---|---|---|
| 大小 | **固定 2 MB** | 变长，默认约 **16 KB**（`block_size` 可配，指压缩前大小） |
| 职责 | 空间管理 / 分配的最小粒度 | I/O、cache、压缩、编码的最小粒度 |
| 是否压缩 | 否（容器） | 是（内容） |

数据文件启动时按 2MB 切分；SSTable 就是一组宏块；宏块内装若干变长微块 [^2][^4]。微块有 `flat`（行存平铺）和 `encoding`（行列混存）两种格式。

### 1.3 两级压缩：这是全文最需要拆清楚的地方

OceanBase 官方文档明确区分**两层**，性质完全不同 [^5][^6][^7]：

| | 第一层：encoding（语义编码） | 第二层：general compression（通用压缩） |
|---|---|---|
| 是否感知数据 | **是**：按列、按值域、按 NDV 自适应选算法 | **否**：纯字节流 |
| 算法 | 字典、RLE、整型差值 + bit-packing、常量、前缀、Hex、列间等值、列间子串 | zlib / snappy / lz4 / zstd（zstd、lz4 level 1） |
| 微块布局 | 定长数据进「列存区」，变长数据按行进「变长区」 | 压缩字节流 |
| **随机访问** | ✅ **支持**：可只解码一行 / 只解码指定列 | ❌ **不支持**：*"Compressed data cannot be randomly accessed. Both compression and decompression are performed on data blocks."* [^5] |
| **直接查询** | ✅ **支持**：谓词下压到字典、AVX2 SIMD 过滤定长列 | ❌ 必须整块解压 |
| 发生时机 | 只在 **compaction** 生成 SSTable 时 | 同左，编码之后再做一次 |
| 效果 | 第一层压缩到约 **50%** | 最终平均约 **30%**（即省 ~70% 空间）[^8] |

流水线：**row padding → encoding → general compression (可选) → encryption (可选) → 写入 2MB 宏块** [^5]。

源码印证（`src/storage/blocksstable/`）[^9]：
- `encoding/` 72 个文件，每种算法 encoder/decoder 成对：`ob_dict_*`、`ob_rle_*`、`ob_const_*`、`ob_integer_base_diff_*`、`ob_string_diff_*`、`ob_string_prefix_*`、`ob_hex_string_*`、`ob_column_equal_*`、`ob_inter_column_substring_*`、`ob_bit_stream.*`、`ob_encoding_bitset.*`；另有 `ob_dict_decoder_simd.cpp`、`ob_raw_decoder_simd.cpp`、`ob_vector_decode_util.*`。
- `cs_encoding/`（4.3 列存）63 个文件：`ob_integer_stream_*`、`ob_string_stream_*`、`ob_*_vector_decoder.*`、`ob_dict_column_decoder_simd.cpp`、`ob_dict_column_decoder_neon.cpp`。
- `ob_sstable_skip_index.cpp/.h`、`ob_bloom_filter_cache.*`、`ob_micro_block_hash_index.*`、`ob_micro_block_cache.*`、`ob_row_cache.*`、`ob_fuse_row_cache.*`、`ob_macro_block_bloom_filter.*`、`ob_data_macro_block_merge_writer.*`。

**自适应选编码**：合并时分析数据类型、值域、NDV，并参考相邻微块的编码压缩历史，启发式选算法；同一列在不同微块可用不同算法 [^7][^8]。

**官方也承认代价** [^7]：
> *"There are issues associated with encoding, of which overheads on encoding and decoding are the most significant... overheads on CPUs caused by complex decoding during progressive iterations in queries, and overheads of the decoder itself... additional performance overheads caused by decoding are inevitable for specific complex encoding formats."*

并特别指出**列间编码对 CPU 不友好**（列可级联引用，解码需按引用关系跨列处理）[^7]。

### 1.4 Skip Index、下推与 Cache 体系

- **Skip Index**（4.3 列存）：微块粒度对每列聚合 **max / min / null 总数**，逐层上卷到**宏块级 → SSTable 级**；查询按扫描范围下钻，选合适粒度的聚合值先过滤再读数据 [^10][^11]。
- **查询下压**：V3.2 起简单下推，V4.x 全面向量化；所有 filter 下压并利用 skip index 与编码信息加速；`COUNT/MAX/MIN/SUM/AVG` 下压；低 NDV 列的 `GROUP BY` 下压（利用微块字典）[^10][^11]。
- **多层 cache**：block cache（微块）+ row cache（单行）+ bloom filter cache（空查询）+ fuse row cache，用来避免对基线数据的随机读 [^2][^4]。
- **宏块复用**：*"During merging, data is reused based on the granularity of macro blocks, and macro blocks containing no updated data are not reopened for reading. This approach minimizes write amplification"* [^4]。压缩方式变更时通过**渐进合并**逐步重写宏块 [^6]。

---

## 2. 关键辨析：「不解压直接查询」≠「零拷贝映射」

这是决定 OceanBase 压缩能否用于 gtvdb 的**分水岭**，必须讲精确，否则容易得出错误结论。

OceanBase 说的「编码数据可以支持不解压直接查询」[^10]，含义是：

```
跳过第二层 zstd 的整块解压
→ 直接在第一层 encoding 数据上执行过滤
→ 字典编码：把谓词转成字典下标的二分查找
→ 差值/bit-packing：SIMD 比较 + 按需 unpack
→ 只解码投影需要的列 / 需要的那一行
```

它**省掉的是「整块解压」**，**没有省掉「解码」**。而且编码后的内存形态是：

- 定长列存区里放的是**变位宽**的 bit-packed 值（例如 delta 后用 11 bit 存一个时间戳）；
- 常量编码是「常量 + exception list」；
- 字典编码是「字典 + 下标引用」；
- 变长区按行放字符串。

这些**都不是定宽 POD 数组**，无法用 `bytemuck::cast_slice` 直接得到 `&[u32]` / `&[i64]`。

而 gtvdb 的 GTVSeg 依赖的恒等式是 [^12]：

```
磁盘字节布局 == 内存 CSR 布局 == CPU cache line 访问单元
```

**编码破坏的不是「能不能读」，而是这个恒等式本身。** 所以我在 `storage-mmap-design.md` §2 写的「压缩与 zero-copy 互斥」结论**依然成立，但理由需要精化**：

| 情形 | 是否破坏可映射性 | 原因 |
|---|---|---|
| 通用压缩（zstd/lz4） | 一定破坏 | 必须整块解压，字节流与逻辑值无对应关系 |
| 语义编码（字典/RLE/delta+bit-packing） | **同样破坏** | 不是因为要解压，而是**编码后不再是定宽 POD 数组**，且需要执行解码指令 |
| 未压缩的定宽 SoA 列 | 不破坏 | `cast_slice` 直接得到切片 |

另一个常被忽略的代价：编码会**打断顺序预取**。CSR 遍历的优势是 `dst[e], dst[e+1], ...` 连续、硬件预取器能提前载入后续 cache line；字典查表、exception list 跳转、跨列引用都会把顺序访问变成随机访问。OceanBase 用 SIMD 向量化解码把这部分补回来，但那是**为压缩付的额外工程成本**——`encoding/` 目录里 72 个文件、`cs_encoding/` 里 63 个文件，绝大多数是 encoder/decoder 对，这就是代价的量级。

> **判断准则**：只要目标是「mmap 后直接当 `&[T]` 遍历」，任何改变元素位宽或引入间接引用的技术都不能进热层，无论它叫压缩还是编码。

---

## 3. 为什么 OceanBase 能压缩、gtvdb 热层不能

不是「技术高低」问题，是**瓶颈不同导致同一笔账的符号相反**。

| | OceanBase | gtvdb 热层（GTVSeg 目标） |
|---|---|---|
| 主要瓶颈 | **磁盘 I/O**（数据在 SSD/HDD，微块 16KB 是 I/O 单位） | **内存带宽 + CPU cache miss**（数据已在 page cache / mmap） |
| 压缩省下什么 | I/O 量、磁盘空间 | 只有磁盘空间（I/O ≈ 0，已无节省空间） |
| 压缩付出什么 | compaction 时的编码 CPU + 查询时的解码 CPU | **每次遍历的解码 CPU + 预取失效 + 分支预测失败** |
| 净收益 | **正**：省 70% I/O，且压缩只在后台合并发生，不碰写路径 [^5] | **负**：I/O 没省，CPU 每轮遍历都要付 |
| 配套代价 | 需要 block/row/bloom/fuse row 多级用户态 cache 缓存解压结果 [^2] | 不需要——**mmap 后 OS page cache 就是数据集本身** |

**反证很干净**：OceanBase 那套 `ob_micro_block_cache` / `ob_row_cache` / `ob_fuse_row_cache` 存在的唯一理由，就是「压缩数据不能直接映射，解压结果得找个地方放」。gtvdb 若引入热层压缩，就**被迫**要重建这套用户态 cache 体系，同时丢掉 mmap 的最大红利（1 份数据、零拷贝、零管理）。用一个复杂性去换一个自己本来不需要解决的问题，不划算。

OceanBase 之所以敢用 zstd 这类高压缩比算法，官方文档给的理由正是 LSM 架构 [^5]：
> *"in the LSM-tree architecture of OceanBase Database, data can be compressed only during compactions, which does not affect data writes. Therefore, compression methods with higher compression ratios can be used."*

**这个「写路径与压缩解耦」的架构决策值得 100% 借鉴，压缩算法本身不值得。**

---

## 4. 逐项适用性评估

### ✅ 值得借鉴（按价值排序）

#### ① 写路径与压缩/编码解耦 ★★★
写入完全不碰编码与压缩，`MemSegment` 保持原生 `Vec`；只有 flush / compaction 时才生成最终格式。这让系统能同时拿到「写快」与「省空间」，且格式升级不影响在线写入。
→ 落地：GTVSeg 的 P1 写路径必须显式写成「delta 原生定宽 → flush 生成段 → L2 归档才转 Parquet 编码压缩」三段，不要让编码逻辑出现在 ingest 路径。

#### ② 宏块复用 + 渐进重写 ★★★（对 gtvdb 尤其契合）
gtvdb 的 `hdb.rs:6,59` 已有 `<root>/<date>/<table>/<symbol>.parquet` 分区，`list_dates()` 做分区剪枝 [^13]。**时间分区天然就是不可变宏块**：历史日期片不会有新边写入（除非 bitemporal 的 system-time 修正触及它）。
→ 落地：compaction 按「段」粒度判断复用——段级 `checksum + version + 时间边界` 未变则直接复用，**不重读不重写**；只有活跃时间片与被 system-time 修正触及的片需要重写。这直接解决 `csr.rs:687` `transpose()` 全量重建的写放大。
→ 格式升级（GTVSeg v1 → v2 窄化 16B/边）走**渐进合并**：后台逐段转换，不要求停机全量重写 [^6]。

#### ③ 分层 Skip Index ★★（可直接补进 GTVSeg，且不破坏零拷贝）
gtvdb 现状是**一维、单粒度、两层级**的 zone map [^14]：
- `csr.rs:99-100` 全局 `max_valid_from` / `min_valid_to`（≈ 段级，但只有 2 个标量）
- `csr.rs:107` `chunk_max_valid_to: Vec<i64>`（chunk 级，只有「最大 valid_to」一个值）
- `csr.rs:734-736` 用它跳过整块过期边

OceanBase 的做法是「微块聚合 max/min/null count → 上卷宏块 → 上卷 SSTable → 查询按范围下钻」[^10]。对应到 gtvdb：

```
chunk 级  : min_valid_from, max_valid_from, min_valid_to, max_valid_to, edge_count
src  级   : 该 src 所有 chunk 的上卷（≈ 宏块级）      ← 当前缺失
段   级   : 全段时间边界 + 总边数（≈ SSTable 级）      ← 当前只有 2 个标量
查询      : as_of(t) 先查段级 → src 级 → chunk 级
            可在 O(1) 判定「整个 src 在 t 时刻无存活边」，跳过整条邻接表
```

**这些元数据本身是定宽的**，放进 GTVSeg header 后的热元数据区即可，仍然可 `cast_slice`、仍然零拷贝。这是「借鉴 OceanBase 但不引入压缩」最干净的一例，且直接增强现有 `zone_map_skips_expired_chunks`。

#### ④ 全局快照点 × Bitemporal ★★（一个有洞察的结合点）
OceanBase 的 major compaction 选全局一致快照，产出基线并提供校验锚点 [^2][^4]。gtvdb 有 `BitemporalStore::as_of_system(table, system_ts)` / `as_of(business_ts, system_ts)`（`gtv-storage/src/bitemporal.rs:156,162`）[^15]——**双时间轴正好需要「某个系统时刻的全库一致视图」**。
→ 落地：把 major compaction 的快照点定义为 `system_ts = T`，生成的基线 GTVSeg 就是 `as_of_system(T)` 的物化。这样「合并」与「时间旅行」共用一套版本语义，基线同时充当 checksum 校验锚点。比 OceanBase 单纯的物理校验点更有业务含义。

#### ⑤ SIMD 时间谓词下推 ★
OceanBase 在编码后的定长列上用 AVX2 做 SIMD 过滤 [^7]。gtvdb 有 SIMD 基础（`gtv-array/src/window/window_simd.rs`、`gtv-index/src/flat.rs` 的 AVX2+FMA）[^16]。
→ 落地：在**未编码的定宽 `valid_from`/`valid_to` 数组**上一次性 SIMD 比较，输出 bitset，替代逐边分支。**比 OceanBase 更简单**——不需要先 unpack，这是零拷贝路线的额外红利。列为 P2，依赖 P0 先落地。

### ⚠️ 有条件借鉴

#### ⑥ LSM 分层（只能取半套）——见 §5.3
L0 tiered / L1 L2 leveled 的三层划分对写路径完全适用，但**读侧必须截断**，否则图遍历被读放大拖垮。

#### ⑦ BloomFilter ☆
OceanBase 用宏块级 bloom filter 加速主键点查与「空查询」[^4]。gtvdb 全库**无 bloom filter 实现**（仅 `gtv-ingest/src/dedup.rs:7` 有 TODO 注释）[^17]。
但图遍历是**顺序扫邻接表**而非点查，收益低；唯一有价值的是「某 src 在 `as_of(t)` 是否还有任何存活边」，而这已被 ③ 的分层 skip index 覆盖。优先级低，不建议为此引入新结构。

#### ⑧ L2 冷层：别造轮子，Parquet 已经在做 ★★
关键发现：`gtv-storage/src/parquet.rs:49-53` 已经 `set_dictionary_enabled(true)`，`DEFAULT_MAX_ROW_GROUP_ROWS = 256*1024`，且 `:20` 注释明确写着「让 min/max statistics 更激进地剪枝 CRM / as-of filters」[^18]。也就是说 **OceanBase 第一层编码（字典）+ skip index（row-group min/max）的思想，gtvdb 在 L2 已经部分在用**（Parquet 的 dictionary/RLE/delta/bit-packing 与 OceanBase encoding 是同一套思路）。
→ 不要新造压缩格式。真正缺的是**把这套剪枝语义接线到查询**：
  - 全库无 `PruningPredicate` / 存储层谓词下推接线（`cbo.rs` 的 `Statistics` 是 CBO 估计算子，不是存储剪枝）[^17]
  - 让 GTVSeg 的分层 skip index 与 Parquet row-group statistics **共用同一套 `as_of` 剪枝谓词定义**，冷热两层行为一致
  - 归档时显式写 Parquet statistics 并把 `as_of` 条件下推到 `ListingTable`

### ❌ 明确不借鉴

| 机制 | 不用的理由 |
|---|---|
| **两级压缩用于 L0/L1 热层** | §2/§3：破坏定宽 POD 恒等式，解码 CPU 进每次遍历，且打断顺序预取 |
| **变长微块 + 块索引** | 压缩后变长 → 必须维护偏移索引；GTVSeg 定宽可纯算术定位 `&[T] + e`。引入压缩等于拿这个优势换磁盘空间，对内存分析引擎亏本 |
| **block / row / bloom / fuse row cache 整套** | mmap 后 OS page cache 即数据集，用户态多级 cache 是「格式不可映射」的补丁，gtvdb 不需要 |
| **flat 行存 / 行列混存** | gtvdb CSR 是纯 SoA，比行列混存更贴合图遍历；没有理由倒退。`storage-mmap-design.md` §3.3 的 AoS 都还要 A/B 实测，更不必引入混存 |
| **字符串 / 列间编码** | 边是定宽数值（`u64/i64/i64/u16`），无字符串列；量子状态的 `gate` 标签是 sidecar，可单独字典编码，不进热 CSR。列间编码还被官方点名「对 CPU 不友好」[^7] |
| **Paxos 多副本、租户级全局调度、分布式 major compaction** | 超出「嵌入式分析引擎」定位，`README.md` 已把 transactions / 分布式列为 out of scope |
| **B+Tree in-place update 的对比结论** | 那是 OceanBase 论证自身优势的语境，与 gtvdb 无关 |

---

## 5. 对现有设计的具体增修

以下是**建议直接写进 `docs/storage-mmap-design.md`** 的内容，不是另起炉灶。

### 5.1 GTVSeg header 增加分层 Skip Index 区

在 §3.2 的 `col_dir[8]` 之后、列区之前，增加一组定宽的聚合数组（全部 64B 对齐，可 `cast_slice`）：

```
offset 4096  ┌─ offsets                    u32  (node_count+1)  ┐
             ├─ chunk_index                u32                   ┤
             ├─ chunk_min_valid_from       i64   ← 新增           ┤
             ├─ chunk_max_valid_from       i64   ← 新增           ┤
             ├─ chunk_min_valid_to         i64   ← 新增           ┤
             ├─ chunk_max_valid_to         i64   （原有）          ┤
             ├─ src_min_valid_from/max_valid_to  ← 新增（src 级上卷）┤
             ├─ dst / valid_from / valid_to / edge_type            ┤
             └─ ...                                               ┘
Header 内    : seg_min_valid_from / seg_max_valid_to / edge_count / checksum
```

`Col<T>` 方案不受影响——这些只是新增的映射列，算子层仍走 `Deref<Target=[T]>`。

### 5.2 Compaction 按段复用（改写 §3.5）

```
ingest → WAL → MemSegment（原生定宽 Vec，零编码零压缩）
       → flush: 排序一次 → 写新 GTVSeg → write_atomic 发布
       → compaction:
            对每个段检查 (time_boundary, checksum, version, system_ts_range)
            未受增量触及的段 → 直接复用，不重读不重写   ← 借鉴宏块复用
            受触及的段     → 多路归并，生成新段
            格式升级       → 后台渐进重写，不停机        ← 借鉴渐进合并
       → 归档淘汰段 → Parquet（编码 + 压缩，仅在 L2）
```

配合 `hdb.rs` 的 `date/table/symbol` 分区：**历史日期片即不可变宏块，绝大多数 compaction 只需重写当前活跃片**，写放大从「全量重建」降到 O(活跃片)。

### 5.3 最重要的约束：「LSM 写、CSR 读」

OceanBase 的查询需要「遍历 MemTable 与 SSTable 由新到旧，融合多版本结果」[^2]。**这个模型不能进 gtvdb 的图遍历。**

原因：`TemporalCSR::edge_slices(src)`（`csr.rs:313`）返回 `(&[u64], &[i64], &[i64], &[u16])` —— 一个连续区间；BFS 每扩展一个节点都要拿这个区间。若改成「base ∪ L0 ∪ L1 delta 多路归并」，则：

- 邻接定位从 `offsets[src]..offsets[src+1]` 的 **O(1)** 变成「k 层查找 + k 路归并」；
- 每个邻居都要判定版本、去重、合并；
- delta 层通常是稀疏的，却要为每次遍历付固定成本。

**正确的半套 LSM**：

| | 写侧（吸收 LSM） | 读侧（必须退化成 CSR） |
|---|---|---|
| 形态 | MemSegment + L0 delta 段 + L1 近期段 + L2 基线段 | **单一连续 CSR**，一个 `offsets[src]..offsets[src+1]` 区间 |
| 何时存在多层 | 仅在活跃写入期间 | compaction 完成后**只留基线**，delta 归零 |
| delta 组织 | 按 `src` 分区的小段，最多一次额外查找 | 禁止把多路归并放进 `neighbor_iter` 热循环 |
| 版本融合 | 由 `BitemporalStore::as_of_system` 在**版本查询**层做 | 图遍历不做版本融合 |

一句话：**LSM 负责「怎么写得进去、怎么少重写」，CSR 负责「怎么读得快」，两者在 compaction 边界交接，不在查询路径上共存。**

### 5.4 修订后的路线图（替换 §6）

| 阶段 | 原交付 | **本次增修** | 验收 |
|---|---|---|---|
| **P0 读路径闭环** | GTVSeg v1 writer + `Col<T>` + `open_segment()` + `register_segment()` | 不变。header 预留 skip index 槽位与 `flags` 位 | 13 个 CSR 单测在 mmap 后端重跑；`bench_segment` 基线 |
| **P1 写路径** | WAL + MemSegment + flush + crash recovery | **+ 明确「写路径零编码零压缩」；+ 段级复用与渐进重写；+ 全局快照点绑定 `system_ts`** | kill -9 重放一致；**历史段复用率**（应接近 100%，只重写活跃片）；`as_of_system(T)` 与基线内容一致 |
| **P2 缓存优化** | 窄化 16B/边、chunk 按字节、Eytzinger、大页 | **+ 分层 Skip Index 落地；+ SIMD 时间谓词下推（在定宽数组上）** | `perf stat` 对比 P0；新增「整 src 跳过」命中率指标 |
| **P3 分层与扩展** | 冷热淘汰、compaction、`.gtvidx` mmap、量子状态段、NUMA | **+ L2 与 Parquet row-group statistics 共用剪枝谓词；+ 谓词下推接线到 `ListingTable`** | 冷热两层 `as_of` 剪枝结果一致；100M 边常驻内存可控 |

**P0 依然不碰 WAL、不碰 `from_arrays`、不碰任何算子**——本次评估不改变这个最小闭环的风险边界。

### 5.5 §7 风险追加一条

> **热层不压缩**：GTVSeg 的 `flags` 里若将来加入压缩位，必须同时放弃 `cast_slice` 路径并引入用户态 block cache——这等于把 §3 的账重新算一遍。任何「给热段加个 zstd」的提议，先回答「省下的 I/O 在哪」：mmap 场景下答案是 0。

---

## 6. 一个诚实的边界

本次分析基于**官方文档、社区技术文章与源码目录清单**，没有逐行读 OceanBase 的 encoder/decoder 实现，也没有在 gtvdb 上跑过压缩 vs 不压缩的对照 benchmark。所以：

- 关于「两级压缩的性质、随机访问能力、瓶颈取舍」的结论是文档级确定的；
- 关于「解码对 gtvdb 遍历具体慢多少」是**推断**，需要 `bench_segment` 出来后用 `perf stat`（`cache-misses`、`dTLB-load-misses`、branch-misses）实测；
- 建议的 P2 SIMD 下推与分层 skip index 收益，同样需要 A/B。

如果后续要做「热层是否值得某种轻量编码」的唯一严谨做法：**只测不改变位宽、不引入间接引用的编码**（例如纯 `delta + 固定 32bit 位宽`，而非 bit-packing），因为只有这类编码仍可能保住 `&[T]` 映射。但即便如此，它仍会在每次访问时多一次加法，需在 P0 基线出来后再判断。

---

## 7. 证据来源

**OceanBase 官方文档 / 技术文章**

[^1]: [Dump management overview (V4.4.2)](https://docs.oceanbase.com/oceanbase-database/V4.4.2-dump-management-overview) — L0 Mini / L1 Minor / L2 Major 三层。
[^2]: [Overview of storage architecture (V4.6.0)](https://docs.oceanbase.com/oceanbase-database/V4.6.0-storage-architecture-overview) / [V4.3.1 Overview](https://docs.oceanbase.com/oceanbase-database/V4.3.1-storage-architecture-overview) — 2MB 宏块、变长微块、宏块复用、cache 体系、全局快照点、查询融合。
[^3]: [转储和合并概述 (V4.3.5)](https://www.oceanbase.com/docs/common-oceanbase-database-standalone-1000000003578006) — Tiered & Leveled、Mini/Minor/Medium/Major/Meta Major Compaction、多路归并。
[^4]: [V4.3.1 Storage architecture overview](https://docs.oceanbase.com/oceanbase-database/V4.3.1-storage-architecture-overview) — *"macro blocks containing no updated data are not reopened for reading"*、block/row/bloom/fuse row cache、只读基线可激进压缩。
[^5]: [Compression and encoding (EN)](http://en.oceanbase.com/docs/common-oceanbase-database-10000000001031424) — 两级压缩定义、*"Compressed data cannot be randomly accessed"*、encoding 可随机访问、SIMD/向量化、编码代价、LSM 使压缩只在 compaction 发生。
[^6]: [编码和压缩（知识库）](https://www.oceanbase.com/knowledge-base/oceanbase-database-1000000000209928) — 微块 ~16KB、编码在 compaction、宏块级渐进重写、表级压缩配置。
[^7]: [压缩与编码（中文）](https://www.oceanbase.com/docs/common-oceanbase-database-cn-1000000000218988) 与 [数据压缩概述](https://www.oceanbase.com/docs/enterprise-oceanbase-database-cn-10000000000946253) — 行列混存、列存区/变长区、各编码算法、自适应选择、列间编码 CPU 代价。
[^8]: [从硬件到存储再到运维的极致降本](https://open.oceanbase.com/blog/20979233312)、[4.x 存储引擎解读](https://open.oceanbase.com/blog/5862533120) — 第一层约 50%、最终约 30%、自适应启发式。
[^9]: `oceanbase/oceanbase` 源码目录 `src/storage/blocksstable/`，经 GitHub Contents API 核对：`encoding/`（72 文件）、`cs_encoding/`（63 文件）、`ob_sstable_skip_index.*`、`ob_bloom_filter_cache.*`、`ob_micro_block_cache.*`、`ob_row_cache.*`、`ob_fuse_row_cache.*`、`ob_data_macro_block_merge_writer.*`。
[^10]: [Columnar storage (EN)](https://en.oceanbase.com/docs/common-oceanbase-database-10000000001973720) / [列存（中文）](https://www.oceanbase.com/docs/common-oceanbase-database-standalone-1000000006077750) / [Core features (V4.3.1)](https://docs.oceanbase.com/jp/oceanbase-database/V4.3.1-obap-features) — Skip Index 微块→宏块→SSTable 上卷、编码不解压直接查询、查询下压、聚合/GROUP BY 下推。
[^11]: [OceanBase 列存：HTAP 混合负载的性能利器](https://open.oceanbase.com/blog/29883533216) — Skip Index 跳过示例。

**gtvdb 代码事实**

[^12]: `docs/storage-mmap-design.md` §2/§3.2 — GTVSeg 定宽、不压缩、不可变、`cast_slice` 恒等式。
[^13]: `crates/gtv-storage/src/hdb.rs:6,59,119` — `<root>/<date>/<table>/<symbol>.parquet` 分区与 `list_dates()` 剪枝。
[^14]: `crates/gtv-core/src/csr.rs:91-109,177-196,313,734-736` — `TemporalCSR` 字段、`chunk_max_valid_to` 单值 zone map、`edge_slices()`、跳块迭代。
[^15]: `crates/gtv-storage/src/bitemporal.rs:156,162,168` — `as_of_system` / `as_of` / `business_at` 双时间轴。
[^16]: `crates/gtv-array/src/window/window_simd.rs`、`crates/gtv-index/src/flat.rs` — 现有 AVX2/FMA 基础。
[^17]: 全库 grep `PruningPredicate|pushdown|bloom` — 无存储层剪枝接线；`crates/gtv-ingest/src/dedup.rs:7` 仅 TODO 注释；`crates/gtv-engine/src/cbo.rs` 的 `Statistics` 是 CBO 估算用途。
[^18]: `crates/gtv-storage/src/parquet.rs:20,41,49-53` — ZSTD 默认、`DEFAULT_MAX_ROW_GROUP_ROWS = 256*1024`、`set_dictionary_enabled(true)`、min/max statistics 用于 as-of 剪枝的注释。
