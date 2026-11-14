# gtvdb 三级数据分层设计：SSD → RAM → VRAM 的极致 IO 路径

> 目标：把「持久化数据」到「GPU 计算」之间那条最长的搬运链，压缩成一条按需、零解码、可 DMA 的流水线。
> 状态：**设计提案（未实现）**。本文是 `storage-mmap-design.md`（GTVSeg mmap，P0/P1 已实现）与 `crates/gtv-accelerator`（VRAM resident + 热度淘汰，已实现）之间的**缝合层设计**，不重复两者的既有结论，只解决「两层之间怎么接」以及「KDB 还教会我们什么」。
> 姊妹篇：[storage-mmap-design.md](storage-mmap-design.md)、[oceanbase-storage-lessons.md](oceanbase-storage-lessons.md)、[acceleration.md](acceleration.md)、[cuda-acceleration-plan.md](cuda-acceleration-plan.md)。

---

## 0. 结论先行

gtvdb 已经有两块成熟的拼图，但**中间缺一块**：

```
（已实现）GTVSeg 定宽 mmap     （缺）        （已实现）VRAM resident 缓存
  SSD 字节 → RAM 零解码         ────────→    RAM → GPU 驻留 + 热度淘汰
  (storage-mmap-design)        缝合层       (gtv-accelerator resident.rs)
```

- **左手（RAM 层）已经很强**：GTVSeg 定宽 + mmap + `MADV_WILLNEED`，cold-open 从 `O(E log E)` 降到 `O(1)`，1M 边实测 **19.7×**。这本身就是 KDB「内存映射即存储」的落地。
- **右手（VRAM 层）也已经很强**：`ResidentRegistry` 有准入控制、decayed-heat 淘汰、resident-prefix split、write-behind delta，是 pg-strom GPU Cache + Strata 热度分层的合体。
- **缺的是「RAM → VRAM」这一跳的按需化与零拷贝化**：当前 resident 上传走的是「把数据读进普通 `Vec` → memcpy 到 pinned → DMA」，等于在 mmap 页缓存之外**又复制了一份**，且上传是同步的、整块的。

本文的核心主张只有一条：

> **VRAM 里的 resident block，应当直接是 GTVSeg 的 mmap 视图（或它的 pinned 镜像），而不是第三份拷贝。** 让 `SSD 页 → RAM 页（mmap，OS 管）→ GPU（DMA）` 成为同一条数据生命周期，而不是三段各自拷贝。

KDB 教给我们的不是「再发明一个 mmap」（我们已经有了），而是三件 mmap 之外的东西——**见 §2**。

---

## 1. KDB+/q 的「极致 IO」到底极致在哪

KDB 是时序/金融分析数据库里把 IO 做到极致的标杆。它的设计可以拆成四个正交的杠杆，**只有第一个是 mmap 本身**：

| # | KDB 杠杆 | 本质 | gtvdb 现状 |
|---|---|---|---|
| ① | **内存映射即存储**：磁盘文件直接 mmap 成内存，无「加载」步骤 | 消除「读文件 → 反序列化 → 建结构」整段拷贝 | ✅ **已实现**：GTVSeg 定宽 `Col<T>` + `Deref`，`open_segment` O(1) |
| ② | **分区 + 枚举字典**：按 `date/table/sym` 分片，字符串→`sym_id` 枚举 | 让数据定宽、可寻址、可剪枝，是 mmap 生效的**前提** | ✅ **已实现**：`hdb.rs` 的 `date/table/symbol` 分区 + `sym_dictionary` 枚举 |
| ③ | **OS page cache 即缓存层**：不造用户态 block cache，靠 mmap 让内核按页管冷热 | 零额外代码获得 LRU/预读/回写，且多个进程共享同一份页 | ✅ **部分**：mmap 天然获得，但 `cache.rs` 仍走 Parquet 全量读，未统一 |
| ④ | **按需映射 + 预热**：`madvise(MADV_WILLNEED / MADV_RANDOM / MADV_SEQUENTIAL)` 显式控制内核预取 | 把「什么时候把哪些页拉进 RAM」变成显式、可调度的决策 | ⚠️ **半程**：Linux `madvise_hugepages` 已有，`MADV_WILLNEED` 按需预热、Windows `PrefetchVirtualMemory` 未做（见 `storage-mmap-design.md` §8.4） |

**关键判断**：KDB 的「极致」不是某个魔法 API，而是**让「存储」和「内存」变成同一个东西**，再用分区和枚举让这个东西变成定宽可寻址的数组。gtvdb 的 GTVSeg 已经把 ①② 做对了；真正要补的是把 ③④ 的「页级调度」思想，**向上延伸到 VRAM 那一层**——这正是 pg-strom（GPUDirect）和 Strata（分层 + 预取）要解决的同一个问题。

---

## 2. 把三个参考统一成一条数据生命周期

三个参考各解决了一段，但没有一个解决「全程」：

| 参考 | 解决的段 | 留下的洞 |
|---|---|---|
| **KDB/q** | SSD→RAM 的 mmap、分区、字典 | 完全不碰 GPU；RAM 之上没有第二级 |
| **pg-strom** | SSD→GPU 的 GPUDirect P2P DMA；GPU Cache 驻留热表 | 依赖 Linux + `nvidia-fs` + NVMe 拓扑；GPU Cache 是静态 `max_num_rows`，超了就损坏 |
| **Strata** | RAM→VRAM 的热度分级 + pinned + 异步预取流水线 | 是 LLM 权重分层，不是数据库；没有持久化语义 |

gtvdb 的机会是**把三段接成一条**，且每一段都复用已有的正确实现：

```
  SSD（GTVSeg 定宽）                   RAM（mmap 页缓存）                 VRAM（resident block）
  ┌──────────────────┐   mmap(按需)   ┌──────────────────┐  DMA(零拷贝)  ┌──────────────────┐
  │ 不可变 .seg 文件   │ ────────────▶ │ 页缓存 = 数据集    │ ────────────▶ │ hot corpus/series │
  │ date/table/sym    │  page fault   │ MADV_WILLNEED    │  pinned +      │ decayed-heat      │
  │ (hdb 分区+枚举)    │  懒加载        │ 预热(显式调度)    │  async prefetch│ 淘汰 + 回退 host   │
  └──────────────────┘               └──────────────────┘               └──────────────────┘
        KDB ①②③④                          KDB ③④ + Strata 预取             pg-strom Cache + Strata 热度
```

### 2.1 三条硬原则（决定哪些技术能进、哪些不能进）

1. **任何一层都不允许「改变位宽或引入间接引用」**（承 `storage-mmap-design.md` §7「热层不压缩」红线）。SSD 层的 GTVSeg 是定宽 POD，RAM 层是它的 `cast_slice` 视图，VRAM 层是同一布局的 `cudaMemcpy`（DMA）目标。**三级共用同一份字节布局**，这是「零拷贝全程」成立的前提。
2. **数据单向流动：SSD → RAM → VRAM**，每级都是「按需、懒、可淘汰」。反向只做「失效」，不做「回写」——GTVSeg 不可变（COW），resident block 淘汰时只是 `cudaFree`，不写回 SSD。
3. **每一跳都有独立、可观测的 fallback**（承 `AcceleratorRegistry` 的「never fails a query」约定）。缺 GPU → 退回 RAM 的 mmap 视图跑 host kernel；缺页 → 触发 page fault 从 SSD 拉；预算不足 → 整块 declined。**任何一跳失败都不该让查询报错**。

---

## 3. 目标架构：三级 + 两条 DMA 通道

### 3.1 三级职责表

| 层 | 介质 | 格式 | 加载方式 | 淘汰/失效 | 对应参考 |
|---|---|---|---|---|---|
| **L0 hot** | VRAM | 与 GTVSeg 同布局的 device buffer | pinned host + DMA，异步预取 | decayed-heat + 保护地板（`resident.rs`） | pg-strom GPU Cache + Strata 热度 |
| **L1 warm** | RAM（页缓存） | GTVSeg mmap 视图 | `MADV_WILLNEED`/`PrefetchVirtualMemory`，按需 page fault | OS 按页回收（天然 LRU） | KDB ③④ |
| **L2 cold** | SSD | GTVSeg（热） / Parquet（归档） | 懒映射；归档走 `ListingTable` | 段淘汰 → Parquet | KDB ① + `storage-mmap-design` L2 |

### 3.2 两条 DMA 通道（关键取舍）

pg-strom 用 **GPUDirect Storage**（`cufile`）实现 SSD→GPU 直传，绕过 RAM。这对 gtvdb 是**诱惑也是陷阱**：

| 通道 | 路径 | 依赖 | 建议 |
|---|---|---|---|
| **A. 经 RAM（默认）** | SSD→(mmap)→RAM→(pinned DMA)→VRAM | 无特殊依赖，跨平台 | **默认主线** |
| **B. GPUDirect（可选）** | SSD→(P2P DMA)→VRAM | Linux + `nvidia-fs` + NVMe 拓扑 | `#[cfg(target_os="linux")]` 可选 feature，P2 再做 |

**为什么默认走 A 而不是 B**：

1. gtvdb 明确支持 Windows（`budget.rs` 大量 WDDM 处理），GPUDirect 是 Linux-only，不能进默认路径。
2. **经过 RAM 的 mmap 层，恰恰是 KDB 价值的来源**——L1 页缓存是「免费」的、跨进程共享的、OS 自动管理的热层。跳过它（走 B）只在「数据只碰一次、且表巨大」时才划算，而 gtvdb 的分析负载（KNN / asof / 图遍历）有强复用，A 的 L1 命中率会很高。
3. B 的收益可量化后再加：`gpudirect_threshold`（表 > 阈值才直传）是 pg-strom 已经验证的开关形态，照搬即可。

**但 A 也不是「直接 memcpy」**——见 §4 的两个优化，那是 Strata 的预取与 pinned 思想的落点。

---

## 4. 两个关键优化（本设计的增量价值）

### 4.1 Pinned 上传：让 mmap 页成为 DMA 源

当前 resident 上传是「普通 `Vec` → memcpy 到 pinned staging → `cudaMemcpyAsync`」，等于在 RAM 里多了一份。KDB 的启发 + CUDA 的机制给出更好的路径：

```
L1 页（mmap, 可能未 pin） ── cudaHostRegister ──▶ pinned 页（零拷贝，仅改页表属性）
                                                     │
                                                     ▼  cudaMemcpyAsync (DMA, 不经 CPU)
                                              L0 device buffer
```

- **`cudaHostRegister` 把已有的 mmap 页直接注册为 pinned**，不需要 `cudaMallocHost` + memcpy 那一跳。数据字节一次都没复制，只是被标记为「DMA 可达」。
- 代价与前提：注册/反注册有驱动开销，且 pinned 页不参与 OS 换页，**只对「即将上传且会复用」的段才注册**，用完 `cudaHostUnregister`。这正对应 Strata 的「pinned 专家池」——但 gtvdb 的粒度是「段/语料」，不是「专家」。

### 4.2 异步预取流水线（Strata 的核心突破）

Strata 相对传统 offload 的质变是**把「同步串行搬运」改成「异步流水线」**：算第 N 块时第 N+1 块已在 PCIe 上流。gtvdb 现状是同步上传，等价于传统 offload 的低效形态。

吸收方案（按收益排序，不追求一步到位）：

| 阶段 | 手段 | 说明 |
|---|---|---|
| P1 | **下一个候选块的 `MADV_WILLNEED` + `cudaHostRegister` 提前发起** | 在 `AcceleratorRegistry` 里加一个「预告」接口，engine 在探测某集合时顺带预告可能下一个要用的集合。数据仍在 RAM 层预热，不占 VRAM |
| P1 | **`cudaMemcpyAsync` + CUDA stream 与 kernel 重叠** | 上传与上一个 kernel 执行重叠，隐藏 PCIe 延迟 |
| P2 | **heat 驱动的预取**：shadow ledger 里 heat 最高但未 resident 的块，空闲时主动拉入 | 复用 `resident.rs` 已有的 shadow ledger（`note_reference` 已在积累非 resident 块的热度），把「被动淘汰」升级为「主动预热」 |

预期效果：命中率不变的前提下，**隐藏 PCIe 8 GB/s 的搬运墙**；对「上传与计算重叠良好」的负载，端到端可提升 **1.5–2×**（数据说话，先 benchmark 后宣称）。

---

## 5. 与现有代码的对接点（最小侵入）

本设计**不新建 crate**，全部落在既有模块的扩展上：

| 改动 | 文件 | 性质 |
|---|---|---|
| `VramBudget` 增加「host-pinned 预算」子账 | `gtv-accelerator/src/budget.rs` | 扩展，pinned 页也是稀缺资源（`cudaHostRegister` 受页数限制） |
| `ResidentRegistry::admit` 接受 `BytesSource`（Owned / Mapped / PinnedMapped）而非强制 `Vec` | `gtv-accelerator/src/resident.rs` | 让 resident 块的来源从「第三份拷贝」变成「mmap 视图引用」 |
| `AccelBackend` 增加 `upload_from_mapped(ptr, len, kind, name)` | `gtv-accelerator/src/backend.rs` | 后端负责 `cudaHostRegister` + `cudaMemcpyAsync`，把 pin 细节封装在 CUDA 侧 |
| `Segment`（GTVSeg）暴露列地址/长度给上传路径 | `gtv-storage/src/segment.rs` | 已能 `cast_slice`，只需把 `MappedCol` 的指针暴露成可上传的 `(ptr, len)` |
| 预取预告接口 | `gtv-accelerator/src/lib.rs`（`AcceleratorRegistry`） | 新增 `prefetch_hint(kind, name)`，engine 在探测路径调用 |
| 新参数 `ACCEL_*` | `gtv-params` / `gtvdb.params.example` | `ACCEL_PIN_MODE`、`ACCEL_PREFETCH` 等，默认关，与 `ACCEL_CACHE` 同风格 |

**不碰**：`csr.rs` 1180 行算子、`edge_slices()` 签名、`Col<T>` 的 `Deref` 语义、GTVSeg 字节格式（v1 不变）。所有改动是**在两层之间加接线**，不是改任何一层的内部。

---

## 6. 实施路线（每阶段独立可验收）

| 阶段 | 交付 | 验收 |
|---|---|---|
| **P1a 测量基线** | `bench_pcie_transfer`：对 100M/1G 边测「普通 Vec 上传 vs pinned mmap 上传 vs 无上传」，量化 DMA 墙 | 一张表：三种路径的 GB/s 与延迟，作为是否值得做 pinned 的判据 |
| **P1b pinned 上传** | `upload_from_mapped` + `cudaHostRegister`，host backend 为 no-op 等价 | GPU 上大语料上传带宽提升；host/无 GPU 路径零回归 |
| **P1c 预取预告** | `prefetch_hint` + `MADV_WILLNEED`（Linux）/ `PrefetchVirtualMemory`（Windows） | 冷启动首查延迟下降；稳态无回归 |
| **P2 heat 预取 + async 上传** | shadow-ledger 驱动预热 + `cudaMemcpyAsync` stream 重叠 | 复用型负载端到端 1.5×+（benchmark 说话） |
| **P2 GPUDirect（可选）** | `#[cfg(linux)]` + `cufile`，`gpudirect_threshold` 开关 | 仅 Linux + NVMe 拓扑，大表扫描省一次 RAM 往返 |

**风险边界**：P1 全部是「在现有回退框架上加一条更快的路径」，任何失败都落到「现状路径」（普通上传 / host 执行），不改变正确性。这是与 `storage-mmap-design.md` P0「最小闭环」一致的工程哲学。

---

## 7. 风险与明确不做

- **`cudaHostRegister` 不是免费**：注册有锁开销、pinned 页有数量上限、注册大段会让 OS 换页能力下降。必须配合 `VramBudget` 的 host-pinned 子账做配额，且默认关闭，仅对「已验证复用」的段开启。
- **mmap 页 + DMA 的一致性**：GTVSeg 不可变（COW）保证映射期间文件内容不变，DMA 源稳定；但**正在被 `cudaHostRegister` 的页若被 OS 换出会导致 DMA 失败**——pinned 正是为了阻止换出，二者是一体的，实现时必须成对 register/unregister。
- **Windows 无 `madvise`/`cudaHostRegister` 语义对等物**：Windows 用 `PrefetchVirtualMemory`（预取）+ CUDA 的 portable pinned memory（注册），行为等价但参数不同，需分别实现，禁止把 Linux 结论直接搬（承 `storage-mmap-design.md` §7 的同款警告）。
- **不安全代码**：`ptr/len → cudaMemcpyAsync` 是 `unsafe` 边界，必须集中在 `cuda.rs` 一处并写 `SAFETY` 契约（对齐、长度、pin 生命周期、kernel 完成前不释放）。
- **不做**：用户态 block cache（OS page cache 就是 L1，再造一个是对 KDB 的背叛）；分布式存储；行级 MVCC；把 GPUDirect 变成默认路径。

---

## 8. 一句话总结

gtvdb 已经做对了 KDB 最难的部分——**定宽 mmap（GTVSeg）**和**分区枚举（hdb）**，也已经做对了 pg-strom/Strata 最通用部分——**VRAM 驻留 + 热度淘汰**。本设计补的是中间那根**「按需、零拷贝、可回退」的传送带**：让 `SSD 页 → RAM 页 → GPU` 成为同一条数据生命周期，用 `cudaHostRegister` 消灭第三份拷贝、用异步预取流水线隐藏 PCIe 墙。三级共用同一份字节布局，是整条设计不可动摇的地基。
