# 立项：向量语料上 GTVSeg（GTVVEC 格式）

> 本文是 [`storage-tiering-ssd-ram-vram.md`](storage-tiering-ssd-ram-vram.md) §4.1 与 [`pinned-upload-p1b.md`](pinned-upload-p1b.md) 的**前置工程立项**：让向量语料（KNN/ANN corpus）以 GTVSeg 同款「定宽、不可变、可 mmap」格式落盘。
> 状态：**V-0 / V-1 / V-2 / V-3 / V-4 全部实现并实测**。
> 前置事实：P1b-1 已交付并提交（`264d75a`，staging 达 pinned 上限，稳态 +14~43%）；P1b-2（`cudaHostRegister` 零拷贝）的唯一阻塞项就是「语料不在 mmap 上」——V-2 已解除该阻塞，V-3 交付封装，V-4 交付 IVF 复用与端到端实测。

## 实现进度

| 阶段 | 状态 | 落点 |
|:--|:--|:--|
| V-0 | ✅ 已交付 | `gtv-core/src/column.rs`：`Pod` seal 开 `f32`，新增 mapped `f32` 列单测（NaN/±0 位模式） |
| V-1 | ✅ 已交付 | `gtv-index/src/vector_segment.rs`（格式：`encode_vector_segment`/`open_vector_segment`/`VectorSegment`，6 单测）；`gtv-index-store/src/vector_segment_io.rs`（落盘 `write_atomic` + mmap 读回 `memmap2`，3 单测） |
| V-2 | ✅ 已交付 | `FlatIndex` `Col<f32>`/`Col<u64>` 化（`flat.rs`，构造器签名不变，新增 `from_segment`）+ `IndexStore::save` 对 Flat 写 GTVVEC payload + `load` 按 payload magic 分派（GTVVEC 走 mmap 冷开 / GFLATv1 走解码）+ `verify` 对 GTVVEC 补 `verify_data`。容器 manifest pad 到 8 B（尾随空格，向后兼容） |
| V-3 | ✅ 已交付 | `cudaHostRegister` RAII 封装（`gtv-accelerator/src/cuda.rs`：`HostRegister` + `register_and_upload`，unsafe 集中一处，register 失败自动回落 P1b-1 staging）+ `Col::stable_bytes()` 暴露稳定字节区间 + `bench_pcie_transfer` 新增 register 列。**平台发现**：Windows WDDM 下 `cuMemHostRegister` 拒绝文件映射页（`CUDA_ERROR_OUT_OF_MEMORY`），register 零拷贝只在 Linux/TCC 可达 |
| V-4 | ✅ 已交付 | `gtv-index/src/ivf_segment.rs`（`GIVFCMP` compound payload：IVF cell 布局 + GTVVEC 语料段，6 单测）；`IvfIndex` `Col<f32>`/`Col<u64>` 化 + `from_columns`；`gtv-index-store/src/payload.rs`（`PayloadKind` 三路分派：GTVVEC/GIVFCMP/Legacy）+ `open_mapped_payload`；`gtv-index-store/examples/bench_cold_open.rs`（冷开对比表）；`bench_pcie_transfer` 增常驻热语料 pinned 占用段 |

> **V-1 范围收敛说明**：立项原文把「容器 payload 从 `GFLATv1` 升级」列在 V-1，但实现时确认——容器 payload 升级必须伴随 `FlatIndex` 的 `Col` 化（否则 mmap 出的 GTVVEC 无法被仍用 `Vec` 字段的 `FlatIndex` 消费，是半成品）。因此 V-1 收敛为「格式模块 + 独立落盘/读回（`VectorSegment` 类型，不破坏现有 `.gtvidx` 路径）」，容器升级并入 V-2。

---

## 0. 结论先行

向量语料当前**完全没有持久化形态**：`register_knn(Vec<Vec<f32>>)` 收到的是引擎内存里的一份 row-major `Vec<f32>`，`.gtvidx` 持久化走的是 `fs::read 整文件 → decode → payload.to_vec()` 三次拷贝。三级分层（SSD→RAM→VRAM）的「RAM warm 层」对向量语料**不存在**——不是没接上，是根本没有可映射的东西。

**推荐方案**：把索引容器的 payload 从 `GFLATv1` 升级为 `GTVVEC v1`——一个与 GTVSeg 共享全部机器（4096 B 定长头 + 列目录 + blake3 + `write_atomic` + `MappedRegion`）的**定宽、不可变、64 B 对齐**向量段格式。`FlatIndex` 的 `data: Vec<f32>` 换成 `Col<f32>`（完全镜像 `TemporalCSR` 已完成的 `Col<T>` 改造），算子零改动即可跑在 mmap 上；mmap 的稳定页地址随后成为 P1b-2 `cudaHostRegister` 的直接目标。

---

## 1. 现状断层（代码事实）

| # | 事实 | 位置 | 断层 |
|---|---|---|---|
| 1 | 语料只活在堆上 | `gtv-engine/src/context.rs:1089` `register_knn(Vec<Vec<f32>>)` → `gtv-index/src/flat.rs:48` `with_metric` → `data: Vec<f32>`（row-major） | ingest 路径物化 `Vec<Vec<f32>>` 再展平，两份拷贝；进程退出即消失 |
| 2 | 持久化 = 三次拷贝 | `flat.rs:153` `to_bytes` / `:169` `from_bytes`（`GFLATv1`）→ `gtv-index-store/src/container.rs:21` `encode/decode` → `store.rs:264` 附近 `fs::read` | 冷打开 O(文件大小)，加载物化 3 份拷贝（文件→Vec<u8>→payload→FlatIndex） |
| 3 | `Pod` 不含 `f32` | `gtv-core/src/column.rs:43-49`：seal 在 `u16/u32/u64/i64` 四个整型 | mmap 字节无法 `cast_slice` 成 `&[f32]`——**向量列上 mmap 的第一个硬约束** |
| 4 | GTVSeg v1 是图 CSR 专用 | `gtv-storage/src/segment.rs:41-53`：列序固定 7/8 槽已用（offsets/chunk_index/chunk_max_valid_to/dst/valid_from/valid_to/edge_type），头字段是 node/edge_count | 没有「向量矩阵」的语义位置；8 号槽是给 P2 skip-index 预留的，且一个槽装不下 ids+dim+metric |

一句话：**向量语料在存储层的存在感是零**。这与「嵌入式时序图 + 向量分析引擎」的定位冲突——图边已经 mmap 冷开 O(1)，向量却要整读+解码。

## 2. 为什么 KDB 思想对向量语料同样成立

`storage-mmap-design.md` 的核心主张「磁盘字节布局 == 内存布局 == CPU 缓存访问单元」对 f32 向量矩阵**更自然**：

- 向量矩阵本来就是定宽 POD（`f32 × dim` row-major），天然满足零解码；
- KNN 扫描（`flat.rs` 的 rayon 流式扫描）已经是**顺序整段读**，mmap 后 OS page cache 即语料，首次扫描的 page fault 流水线化；
- `FlatIndex::data()`（`flat.rs:137`）已提供 `&[f32]` 零拷贝出口——CUDA 上传路径（`build_corpus`）拿到的是同一个引用，**接口零改动**；
- P1b-1 实测已证明：DMA 走 pinned 快路径收益 +14~43%；P1b-2 的 `cudaHostRegister` 需要一块**地址稳定、生命周期可控**的内存——`MappedCol` 持有的 `Arc<dyn MappedRegion>`（`column.rs:88-101`，契约保证地址不变）正好满足。

## 3. 方案选型

### 3.1 备选 A：独立 `GTVVEC` 文件格式（新目录管理）

新建 corpus store（镜像 `SegmentStore`），`.gtvvec` 文件 + 版本管理自建。
**否决理由**：`gtv-index-store` 已有完整的版本化机制（`store.rs:64-298`：`build/save/load/activate/rollback/verify`），重复造轮子。

### 3.2 备选 B（推荐）：容器 payload 升级为 GTVVEC + 容器 mmap 化

`GTVIDX` 容器结构不变（magic + version + flags + manifest_len + manifest + payload），两处升级：

1. **payload 格式**从 `GFLATv1`（自带可变长头）换成 `GTVVEC v1`（定宽列布局）；
2. **manifest 长度 pad 到 64 B 对齐**，使 payload 起点 64 B 对齐——容器文件本身即可 mmap，payload 区间按 GTVVEC 列目录零拷贝取列。

读路径变成：`open → mmap 容器 → 校验 256 B header_checksum（O(1)）→ MappedCol::new(payload 区间)`。冷打开从 O(文件) 降到 O(1)（+ 一次小 manifest JSON 解析），拷贝从 3 份降到 **1 份 mmap 视图**。

**保留的兼容性**：`GFLATv1` 的 `to_bytes/from_bytes` 不删（`flat.rs:153/169`），仅在 `load` 时按 magic 分派——旧容器照常走解码路径，新容器走 mmap。版本字段（`container.rs` version u16）天然支持按版本分派。

### 3.3 否决：塞进 GTVSeg v1 的 8 号槽

8 号槽是 P2 skip-index 预留；GTVSeg 头字段（node_count/edge_count/chunk_size）全是图语义；一个文件同时是图 CSR 和向量矩阵会让 `Segment::open` 的校验逻辑二义。图边与向量语料**共享机器但不共享文件**。

## 4. GTVVEC v1 字节格式

```
Header（固定 4096 B，与 GTVSeg 同规格）
  0    magic[8]            "GTVVEC\0\0"
  8    version   u16 = 1
  10   flags     u16
        bit0 FLAG_NORMALIZED  行已在写入时按 metric 归一化（Cosine）
        bit1 FLAG_BIG_ENDIAN  大端机产物，v1 reader 拒绝
  12   row_count  u64
  20   dim        u32
  24   metric_code u8   （0=L2, 1=Cosine, 2=Ip——沿用 gtv-index/src/bytes.rs 的编码）
  25   reserved[7]
  32   data_checksum_offset 等：与 GTVSeg 完全同构
  64   col_dir[4]  { offset u64, len_bytes u64, elem_size u8, kind u8, reserved[6] } = 24 B each
  256  header_checksum[32]  blake3 over [0, 256)
  288  data_checksum[32]    blake3 over [4096, EOF)
  320  reserved[3776]
4096  列区（各自 pad 到 64 B 对齐）：
  col0  kind=0  ids      u64[row_count]
  col1  kind=1  vectors  f32[row_count * dim]   ← 唯一大列，row-major
```

设计约束（全部继承 GTVSeg，`segment.rs:124-131`，缺一不可）：

- **不压缩**：整段定宽 POD，`cast_slice` 直接得 `&[u64]` / `&[f32]`，零解码；
- **不可变**：写入走现成 `write_atomic`（`gtv-storage/src/atomic.rs:31-46`），更新 = 新版本（copy-on-write）；
- **校验与版本**：magic + version + blake3，`header_checksum` 恒校验（O(1)），`data_checksum` 只在 `verify` 时算——保持冷开 O(1)；
- **对齐**：vectors 列起点 64 B 对齐，`cast_slice` 前断言对齐与长度；
- **字节序**：固定 LE。

**Cosine 归一化语义**（`flat.rs:99-103` 的行为平移）：归一化在**写入时**完成并置 `FLAG_NORMALIZED`——mmap 不可变，读侧不能像 `from_flat_metric` 那样原地改行。Reader 信任该 flag，行为与现有 `is_normalized()` 语义一致。

## 5. 关键改造点

### 5.1 `Pod` 扩展 `f32`（V-0，一处改动）

`column.rs:43-49` 的 seal 加 `impl Sealed for f32 {}`，`Pod` 文档同步更新。

**安全论证**：`Pod` 的要求是「每个位模式都有效、无填充、Copy + Send + Sync」。`f32` 满足全部——没有无效 niche（NaN/Inf/±0 都是合法位模式），`size_of == align_of == 4`，`bytemuck` 对 `f32` 也是同样结论。`MappedCol` 的越界/对齐检查（`column.rs:122-171`）不依赖元素语义，直接复用。

### 5.2 `FlatIndex` 的 `Col<f32>` 化（V-2，镜像 TemporalCSR）

```
// flat.rs（现状）
ids:  Vec<u64>,
data: Vec<f32>,
// 改为
ids:  Col<u64>,
data: Col<f32>,
```

- 现有构造器（`new/with_metric/from_flat/from_flat_metric`）产出 `Col::Owned`，签名不变；
- 新增 `from_mapped(region, header)`：GTVVEC 列目录 → `MappedCol::new` → `Col::Mapped`；
- 扫描算子已经全部通过 `Deref<Target=[f32]>` 访问（`flat.rs:9` 自述「one streaming pass」），**零改动**；
- `Clone` 可用（`Col<T>` 是 `Clone`，Mapped 变体共享 `Arc` region）；
- `vector_for_id` / `data()` / `ids()` 的返回类型不变（`&[f32]` / `&[u64]`），CUDA 路径（`build_corpus(&[f32])`）接口零改动。

### 5.3 `gtv-index-store` 读路径 mmap 化（V-2）

`store.rs:255 load` 与 `container.rs:36 decode`：检测 payload magic——

- `GFLATv1`：现状路径（解码物化），旧容器兼容；
- `GTVVEC\0\0`：mmap 容器文件 → `MappedFile`（`segment.rs:175` 已有实现）→ payload 偏移 = `16 + manifest_len_padded` → 按列目录建 `Col` → `FlatIndex::from_mapped`。

`LoadedIndex` 的所有权模型需要从「owned `Arc<dyn VectorIndex>`」扩展为「mapped 同样成立」——`FlatIndex` 持有 `Arc<dyn MappedRegion>`，映射生命周期由它维持，与 `MappedCol` 的安全模型一致（`column.rs:96-101`）。

### 5.4 引擎接线（V-2）

- `register_knn*` 保持现状（Owned 路径，ingest 语义）；
- 新增/改造 `save_knn` / `load_knn`：save 走 GTVVEC payload 写入；load 走 mmap——引擎里 `knn_collections: HashMap<String, Arc<KnnCollection>>` 不变，`KnnCollection` 内部换成 `FlatIndex(Col)`；
- CLI `.knn save/load` 命令与 `DATA_DIR` / index-store root 参数沿用参数档（`gtvdb.params.example`）。

### 5.5 与 P1b-2 / 三级分层的衔接（V-3 解锁 P1b-2）

```
[SSD .gtvvec]  ──mmap──▶  [RAM: OS page cache = GTVVEC 列视图（Col<f32>）]
                                │
                                │ V-3: cuMemHostRegister_v2(页区间) —— 页已是稳定地址
                                ▼
                          [pinned DMA 快路径（P1b-1 已交付）]
                                ▼
                          [VRAM: DeviceCorpus]
```

- V-3 的 unsafe 面**只有一处**：`cuda.rs` 里 `cuMemHostRegister_v2` / `cuMemHostUnregister` 成对封装，注册范围 = `MappedFile` 的字节区间；生命周期由 `Arc<dyn MappedRegion>` 的 drop 钩子保证 unregister 先于 munmap；
- 收益量化路径明确：P1b-1 已证明 staging == pinned 上限，所以 V-3 的增益 = **省掉 staging 的一次 host memcpy**（P1b-1 实测显示这不是瓶颈），真正的增量价值在于**常驻热语料免重复 staging**——注册一次，多次查询直接从 mmap 页 DMA，staging 缓冲可以被其他语料复用，pinned 页占用（`staging_bytes()` 已可观测）随之下降；
- 红线继承（tiering 文档 §2.1）：三级共用同一份定宽 POD 字节布局 ✓（GTVVEC 就是这个布局）；单向流动、反向只失效 ✓（容器不可变）；每跳独立 fallback ✓（register 失败自动回落 P1b-1 staging 路径，查询不报错）。

### 5.6 IVF 语料落盘复用（V-4，`GIVFCMP`）

一个 IVF 索引是「两样东西粘在一起」：

- **语料**——`rows × dim` 的 `f32` + `u64` ids，是 payload 的大头，**正是 GTVVEC 已经存的东西**；
- **元数据**——粗量化器（`nlist × dim` 质心）、cell 偏移、原位置置换、训练溯源，都很小，`Vec` 形状，解码极便宜。

所以 IVF 不需要发明新格式，只需要**把语料那段交给 GTVVEC**：

```
GIVFCMP compound payload
  0    magic[8]        "GIVFCMP\0"
  8    meta_len u32    元数据段长度（pad 到 8 B）
  12   version u16     1
  14   reserved u16
  16   meta[meta_len]  dim/metric/nlist/nprobe/row_count/orig_pos/centroids/list_offsets/training
  ...  padding 到 8 B 边界
  ...  gtvvec[...]     一个 GTVVEC v1 段（ids + vectors）
```

- GTVVEC 段起点 8 B 对齐 → 它的 `u64` ids 列在映射里仍是元素对齐的 → `Col<u64>` 零拷贝成立；
- 元数据段里 `orig_pos` 是 `4·rows` 字节，相对于语料的 `4·rows·dim` 字节是 **1/dim**（dim=128 时 1/128），所以 IVF 的冷开代价是 **O(rows)** 而非 O(rows·dim)；
- 旧的 `GIVFv1` / `GIVFv2` 编码**不删不改**，`load` 按 payload magic 分派，旧容器照常走解码路径。

`IvfIndex` 相应地改成 `data: Col<f32>` / `ids: Col<u64>`（镜像 `FlatIndex`），并新增 `from_columns(...)` 接受已装配好的列——这就是 mmap 路径的接缝。

**store 侧的三路分派**（`gtv-index-store/src/payload.rs`）：

| payload magic | 读路径 |
|---|---|
| `GTVVEC\0\0` | `mmap` 容器 → 语料列就地重解释为 `Col` → `FlatIndex` |
| `GIVFCMP\0` | `mmap` 容器 → 解码 cell 布局 → 语料列就地重解释为 `Col` → `IvfIndex` |
| 其他（`GFLATv1` / `GIVFv1` / `GIVFv2` / HNSW） | 读整文件 → 校验容器 checksum → 解码 payload |

一个坑值得记下来：容器尾部有 32 字节的 blake3 checksum，它**不属于 payload**。如果 payload 视图不设上界，payload 自己的 `data_checksum`（覆盖 payload 全部字节）就会把这 32 字节算进去，校验必失败。修法是给 `MappedRegion` 增加**带长度**的 `subregion(region, offset, len)`——`subrange` 只管起点，不管终点。

## 6. 分阶段路线与验收

| 阶段 | 内容 | 预期产出 |
|:--|:--|:--|
| **V-0** | 本立项文档 + `Pod` 扩展 `f32`（含 mapped `f32` 列单元测试） | 格式定稿，`column.rs` 门开 |
| **V-1** | GTVVEC v1 格式模块（`gtv-storage/src/vector_segment.rs`）：writer（`write_atomic`）+ cold-open reader + manifest pad 对齐 | 格式单测：round-trip、校验、对齐、旧 payload 兼容分派 |
| **V-2** | `FlatIndex` Col 化 + `gtv-index-store` load mmap 化 + 引擎 save/load 接线 | 冷打开 O(1)；`cargo test --workspace` 全绿；旧 `.gtvidx` 照常加载 |
| **V-3** | `cudaHostRegister` 封装 + 基准（register-mmap 直接 DMA vs P1b-1 staging） | `bench_pcie_transfer` 新增 register 列；fallback 路径验证 |
| **V-4** | IVF 语料落盘复用 + 端到端基准 | 冷打开对比表（fs::read+decode vs mmap）；常驻热语料 pinned 占用对比 |

**验收硬指标**：

1. 1e6 × 128 维（约 512 MB）语料冷打开：现状（fs::read + decode + 物化）→ mmap 后应降至 **亚毫秒级 open + header 校验**（页错误推迟到首次扫描）；
2. 进程内存占用：语料从「3 份拷贝」降到「1 份 mmap 视图」（OS page cache 即数据）；
3. CUDA 正确性：`cuda_acceptance` 7 条门槛逐位不变（上传路径只改来源，不改语义）；
4. 回归：旧 `GFLATv1` 容器加载结果与新格式逐位相等。

### 6.1 实测结果（V-4，Windows / RTX 4060 Laptop，同机 A/B）

> 同机 A/B，不是外部性能声明。复现命令见各 example 的文件头注释。

**Flat / GTVVEC payload，1e6 × 128（488.3 MiB 语料）**：

| 读路径 | open | 首次查询 | 稳态查询 | load 后常驻增量 |
|---|---|---|---|---|
| legacy（`fs::read` + `decode` + 物化） | **885.155 ms** | 13.673 ms | 11.793 ms | **1487.7 MiB** |
| mmap（页缓存） | **0.850 ms** | 40.690 ms | 11.032 ms | **495.8 MiB** |

**IVF / GIVFCMP payload，1e5 × 128（48.8 MiB 语料，nlist=64）**：

| 读路径 | open | 首次查询 | 稳态查询 | load 后常驻增量 |
|---|---|---|---|---|
| legacy（`GIVFv2`） | **74.994 ms** | 1.098 ms | — | **148.9 MiB** |
| mmap（`GIVFCMP`） | **1.099 ms** | 1.610 ms | 0.589 ms | **51.3 MiB** |

对照硬指标：

1. **冷打开亚毫秒级** ✅ —— 1e6 × 128 的 open 从 885 ms 降到 **0.85 ms**（约 **1040×**）；IVF 1e5 × 128 从 75 ms 降到 **1.1 ms**（约 **68×**）；
2. **3 份拷贝 → 1 份 mmap 视图** ✅ —— 1e6 × 128 的常驻增量从 1487.7 MiB 降到 495.8 MiB（**3.0×**），差额正好是「文件字节 + payload 字节 + 物化语料」这三份；
3. **CUDA 正确性不变** ✅ —— `cuda_acceptance` 7 条门槛全绿（上传路径只改来源，不改语义）；
4. **旧格式回归** ✅ —— `GFLATv1` / `GIVFv1` / `GIVFv2` 容器照常加载，IVF 的 `to_bytes`/`from_bytes` 语义未变。

**首次查询变慢是预期内的**：legacy 路径在 decode 时已经把页读进来了，mmap 路径把 page fault 推迟到第一次扫描。对「一个索引开一次、扫很多次」的形态，mmap 的总成本更低；对「开一次就扫一次」且语料远大于内存的形态，legacy 反而没有延迟——这是 mmap 的经典权衡，不是回归。

**常驻热语料的 host pinned 占用**（`bench_pcie_transfer` 新增段）：

| 路径 | 额外 host pinned 内存 | 设备侧副本 |
|---|---|---|
| staging（P1b-1） | **= 语料大小**（64 MiB / 256 MiB 实测） | = 语料大小（列主序） |
| register（P1b-2） | **0**（就地锁页，不额外分配） | = 语料大小（列主序） |

register 路径在 Windows WDDM 上不可用（见 §7），所以本机实测行是 N/A；Linux/TCC 下才是那个「常驻热语料不额外占 host 内存」的结局。**Windows 上常驻热语料的真实 pinned 预算就是 staging 的那一行**——这是把语料放上 mmap 之后，host 内存预算唯一需要盯的数字。

## 7. 风险与边界

| 风险 | 缓解 |
|:--|:--|
| Windows 上大 mmap 的首次访问 page-fall 抖动（无 `MADV_WILLNEED`） | v1 先交给 OS page cache 自然预热；`PrefetchVirtualMemory`（Win8+）作为 V-2 可选优化，不做默认路径 |
| `cudaHostRegister` 对 mmap 页的兼容性（WDDM/驱动差异） | V-3 封装必须带回退：register 失败 → P1b-1 staging 路径，查询不报错。**已实测确认**：Windows WDDM 拒绝锁文件映射页（返回误导性的 `CUDA_ERROR_OUT_OF_MEMORY`），register 零拷贝只在 Linux/TCC 可达 |
| IVF 冷开仍是 O(rows)（`orig_pos` 置换元数据要物化） | 代价是语料的 1/dim（dim=128 时 1/128），可接受；若 dim 很小且 rows 极大，可把 `orig_pos` 也并入 GTVVEC 的保留列槽 |
| manifest JSON 解析仍在冷打开路径 | manifest 是小对象（KB 级），O(1) 可接受；若未来膨胀可改定宽 manifest 头 |
| Cosine 归一化位置从读侧移到写侧 | `FLAG_NORMALIZED` 显式标注；旧格式读侧归一化行为保留，不混淆 |
| `f32` 进 `Pod` 是 trait 语义放宽 | seal 仍是封闭集合（加一个类型），文档明示「位模式全有效」判据，后续 `f64`/`i32` 按同判据评审 |

## 8. 不做什么（红线外）

- 不做向量**写路径的在线追加**（GTVSeg/GTVVEC 均不可变，更新 = 新版本——与 COW 语义一致）；
- 不做压缩/量化（PQ/SQ 属于 `gtv-index` 的索引问题，与存储格式正交，留 IVF 后续评估）；
- 不动 GTVSeg v1 图格式本身（P2 skip-index 窄化仍是独立路线）。
