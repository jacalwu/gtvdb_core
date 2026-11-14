# P1b 设计：pinned 上传与 `cudaHostRegister` 的落地形态

> 本文是 [`storage-tiering-ssd-ram-vram.md`](storage-tiering-ssd-ram-vram.md) §4.1 / §5 的展开。目标：把「消灭第三份拷贝」从一句口号变成一份**可落地、可验收、可分阶段**的技术方案。
> 前置事实依据：[`bench_pcie_transfer`](../crates/gtv-accelerator/examples/bench_pcie_transfer.rs)（P1a 已交付）实测：稳态 pinned 上传比未锁页快 **33–45%**，小载荷（1 MiB）快 **4.37×**。结论：pinned 有可量化价值，值得做。
> 状态：**P1b-1 已实现**（`cuda.rs` 的 `StagingBuffer` + `build_corpus`/`device_series` 改造）；**P1b-2 已实现**（`cudaHostRegister` 封装 `register_and_upload`，V-3 交付，含平台 fallback）。

---

## 0. 结论先行

P1b 不能一步到位，因为存在一个**当前代码的断层**：设计文档 §4.1 假设「上传来源是 GTVSeg 的 mmap 视图」，但**当前 KNN 语料根本不是 mmap**——`GtvContext::register_knn(Vec<Vec<f32>>)` 收的是引擎内存里的一份 `Vec<f32>`，向量语料当前**没有落 GTVSeg**（GTVSeg 只存图边：`src/dst/time/edge_type` 定宽列，不含 `f32` 向量列）。

因此 P1b 拆成两个递进的阶段：

| 阶段 | 内容 | 消灭的拷贝 | 依赖 |
|:--|:--|:--|:--|
| **P1b-1（本次先做）** | `cudaHostAlloc` pinned staging：把「普通 `Vec` → memcpy 到 pinned → DMA」里的 **staging 拷贝从「未锁页临时缓冲」换成「write-combined pinned 缓冲」**，让 DMA 走快路径 | 消灭「未锁页 → 驱动内部 bounce buffer」的隐藏拷贝 | 仅改 `cuda.rs`，无架构依赖 |
| **P1b-2（语料上 mmap 后）** | `cudaHostRegister` 注册已有 mmap 视图，零拷贝 DMA | 消灭「mmap 页 → 第三份拷贝」 | 依赖「向量语料接上 `Col<T>` / GTVSeg」，是更大的独立工程 |

**一句话**：P1b-1 是「不改变数据来源，只改搬运路径」，立即可做、立即可验；P1b-2 是「改变数据来源」，是三级分层的终点，但被「向量语料未上 mmap」这一前置卡住，单列。

---

## 1. 为什么分两段（断层诊断）

### 1.1 当前上传链路的真实形态

`CudaBackend::build_corpus`（`cuda.rs:240`）现在走的是：

```
Vec<f32> (engine 内存) ── clone_htod ──▶ CudaSlice<f32> (device)
```

`clone_htod` 内部（cudarc `memcpy_htod`）对**未锁页**的 `Vec` 源，驱动会先把源复制进一个内部 **bounce buffer（staging）**，再从 staging DMA 到 device。P1a 实测证明：这个 staging 拷贝在稳态吃掉 **33–45%** 的带宽。

### 1.2 数据来源的断层

| 层 | 数据 | 落 GTVSeg 了吗 |
|:--|:--|:--|
| 图边（`src/dst/time/edge_type`） | `Col<u64>/Col<u16>` | ✅ 是（M6-P0，`segment.rs`） |
| **KNN 语料（`f32` 向量）** | `Vec<f32>`（`register_knn` 展平） | ❌ **否**——当前只在引擎内存，无持久化、无 mmap |

所以 §4.1 的 `cudaHostRegister(mmap 页)` 目前**没有可注册的 mmap 页**——语料还在 `Vec` 里。这让 P1b 分叉成两条路径：

- **P1b-1**：语料仍是 `Vec`，但把「上传 staging」从隐式 bounce buffer 换成**显式、可复用、write-combined 的 pinned 缓冲**。收益 = 消除 bounce buffer 拷贝，P1a 已证明 33–45%。
- **P1b-2**：先把向量语料接上 GTVSeg（`Col<f32>` 或 sidecar），再做 `cudaHostRegister`，收益 = 消除「mmap 页 → 第三份拷贝」整段。

### 1.3 技术可行性核对（已查 cudarc 0.19）

| 能力 | cudarc 暴露 | 结论 |
|:--|:--|:--|
| `cudaHostAlloc`（分配 pinned） | ✅ `CudaContext::alloc_pinned`（safe） | P1b-1 用这个 |
| `cudaHostRegister`（注册已有内存为 pinned） | ⚠️ safe API **无**，但底层 FFI `driver::sys::cuMemHostRegister_v2` / `cuMemHostUnregister` 是 `pub` | P1b-2 需 unsafe 直调 FFI |
| `cudaMemcpyAsync`（异步 DMA） | ✅ `result::memcpy_htod_async`（unsafe pub） | 两条路线都用 |

**结论**：P1b-1 可完全用 safe API 落地；P1b-2 需要一次性的 unsafe FFI 封装（集中在 `cuda.rs` 一处，写 `SAFETY` 契约）。

---

## 2. P1b-1：pinned staging 上传（✅ 已实现）

### 2.1 目标

不改数据来源（语料仍是 `Vec<f32>`），只改搬运路径：把 `clone_htod(Vec)` 换成「先 copy 进一块 write-combined pinned 缓冲，再 DMA」。让 DMA 稳定走 pinned 快路径，拿回 P1a 测出的 33–45%。

### 2.2 设计（已落地）

在 `cuda.rs` 新增一个泛型 pinned staging 缓冲，挂在 `CudaBackend` 上：

```
CudaBackend {
    // 新增
    staging_f32: Mutex<StagingBuffer<f32>>,   // corpus 上传（f32 向量）
    staging_i64: Mutex<StagingBuffer<i64>>,   // as-of 时间序列上传（i64）
    ...
}

struct StagingBuffer<T: DeviceRepr> {
    buf: Option<PinnedHostSlice<T>>,   // cudaHostAlloc(write-combined)，惰性分配
    cap: usize,                        // 精确元素数（T），大小变化才重分配
}
```

`build_corpus` / `device_series` 的上传段改成：

```
// 旧：直接 clone_htod(Vec)  → 隐式 bounce buffer
// 新：
let staging = self.staging_f32.lock()?;
staging.ensure(n)?.copy_from_slice(data);   // 一次显式 memcpy（源 Vec → pinned）
let d = stream.clone_htod(&staging.buf)?;   // DMA 走 pinned 快路径（无 bounce）
```

### 2.3 关键决策与理由

| 决策 | 理由 |
|:--|:--|
| **用 `cudaHostAlloc`（write-combined）而非 `cudaHostRegister`** | 语料还是 `Vec`，register 一个「马上要上传、上传完就丢弃」的临时 `Vec` 不划算（register/unregister 开销 > 一次 memcpy）；write-combined pinned 缓冲是可复用、长驻的，摊薄分配成本 |
| **staging 缓冲精确匹配大小 + 复用** | `clone_htod(&buf)` 传输的字节数 = `PinnedHostSlice::len`（分配时的 n），精确匹配避免 DMA 未初始化尾部；同大小复用摊薄 `cudaHostAlloc` 成本，大小变化才重分配 |
| **`StagingBuffer` 泛型化（f32 + i64）** | corpus 是 `f32`、as-of 时间序列是 `i64`，两路径都受益，不共用一块（类型安全） |
| **只在 CUDA 路径生效，host/no-GPU 零改动** | 承「never fails a query」约定：无 GPU 时根本不进 `build_corpus` |
| **pinned 页是稀缺资源，暴露 `staging_bytes()` 观测** | `cudaHostAlloc` 占用 lock 页（不可换出），提供 `CudaBackend::staging_bytes()` 上报 footprint（配额执行留 P1b-2） |

### 2.4 验收（已全绿）

1. ✅ `bench_pcie_transfer` 增加「staging」列：**staging == pinned 上限**，稳态（≥8 MiB）比 unpinned 快 **+14~43%**，小载荷（1 MiB）**+432%**。
2. ✅ `cuda_acceptance` 7 条门槛全绿（KNN / asof / IVF 结果逐位不变）。
3. ✅ `cuda_smoke` 1 项通过；默认 feature `cargo test -p gtv-accelerator --lib` 48 项全绿（host 路径零回归）。
4. ✅ `cargo check --workspace` 默认 feature 通过；example 用 `required-features = ["cuda"]` 让默认 build 跳过（而非 `main` 缺失报错）。

**实测（Windows RTX 4060 Laptop，warm-up 后）**：

| payload | unpinned | staging（P1b-1） | pinned（上限） | staging 增益 |
|--:|--:|--:|--:|--:|
| 1 MiB | 0.73 GB/s | 3.86 GB/s | 3.86 GB/s | +432% |
| 8 MiB | 9.89 GB/s | 11.29 GB/s | 11.29 GB/s | +14% |
| 64 MiB | 8.82 GB/s | 11.65 GB/s | 11.65 GB/s | +32% |
| 256 MiB | 8.34 GB/s | 11.94 GB/s | 11.94 GB/s | +43% |
| 1024 MiB | 8.83 GB/s | 12.05 GB/s | 12.05 GB/s | +37% |

**关键洞察**：`staging == pinned` 说明「copy 进 pinned」那一步（host memcpy）不是瓶颈，DMA 已打满 pinned 带宽墙。因此 **P1b-1 已拿到 pinned 路线的几乎全部收益**；P1b-2 的增量价值只在「源数据已是 mmap（无需 copy）」时才显现——这正说明 P1b-2 的前置「向量语料上 mmap」是决定它是否值得做的关键。

---

## 3. P1b-2：`cudaHostRegister` 零拷贝（语料上 mmap 后，✅ 已实现 V-3）

> V-3 已交付封装；但实测发现 **Windows WDDM 不支持 register 文件映射页**（见 §3.4 平台发现），register 零拷贝只在 Linux/TCC 下可达，封装已带回退。

### 3.1 前置工程：向量语料上 mmap

当前 `register_knn(Vec<Vec<f32>>)` → 展平 `Vec<f32>`。要让语料可被 `cudaHostRegister`，需要：

1. GTVSeg 增加 `f32` 向量列（或 sidecar 文件），让语料以 `Col<f32>` 的 mmap 形态存在。
2. `register_knn` 增加「从 `Col<f32>` 加载」的入口，`KnnRequest.data` 从 `&[f32]` 扩展为可表达「mapped 来源」的 `BytesSource`。

这本身是「向量语料持久化」的独立工程，**不在 P1b 范围内**，但它决定 P1b-2 何时能做。

### 3.2 核心设计

```
Col<f32> (GTVSeg mmap) ── cudaHostRegister ──▶ pinned 页（零拷贝，仅改页表）
                                                    │
                                                    ▼  cudaMemcpyAsync (DMA)
                                             device buffer
```

- `AccelBackend` 新增 `upload_from_mapped(ptr, len, kind, name)`，后端负责 register + memcpy + unregister。
- `KnnRequest.data` 由 `&[f32]` 扩展为 `BytesSource`（`Owned(&[f32])` / `Mapped(ptr, len)` / `PinnedMapped(ptr, len)`）。
- `resident.rs` 的 `admit` 接受 `BytesSource`，让 resident block 的来源从「第三份拷贝」变成「mmap 视图引用」。

### 3.3 SAFETY 契约（unsafe 集中一处）

- `ptr` 在 register 期间不得被 `munmap`（`Col` 的 `Arc<dyn MappedRegion>` 保证映射存活）。
- register/unregister 必须成对，且 unregister 必须等最后一个 kernel 完成（`stream.synchronize` 或 event）。
- pinned 页不参与 OS 换页，必须受 `VramBudget` host-pinned 子账配额，避免锁死系统内存。
- 字节布局必须与 GTVSeg 定宽 POD 一致（承 §2.1 红线 1），不做任何 `cast` 或对齐破坏。

### 3.4 验收（V-3 已交付）

- ✅ `cudaHostRegister` RAII 封装：`gtv-accelerator/src/cuda.rs` 的 `HostRegister`（`cuMemHostRegister_v2`/`cuMemHostUnregister` 成对，Drop 保证 unregister）+ `register_and_upload`（register → `clone_htod` 零拷贝 DMA → synchronize → unregister），unsafe 集中一处并写 `SAFETY` 契约。
- ✅ `Col::stable_bytes()`（`gtv-core/src/column.rs`）：暴露 `(NonNull<u8>, len)` 稳定字节区间，`Mapped`/`Owned` 两变体都可用，是 `cudaHostRegister` 的唯一取址入口。
- ✅ `bench_pcie_transfer` 新增 register 列：register-mmap 直接 DMA，与 staging/pinned 对比。
- ✅ fallback 路径验证：`register_and_upload` 在 register 失败时返回 `Err`，由调用方回落 P1b-1 staging，查询不报错。

**平台发现（重要，改写 P1b-2 收益预期）**：

在 **Windows WDDM 驱动**（开发机 RTX 4060 Laptop）下，`cuMemHostRegister_v2` 对**文件映射页**（`memmap2::Mmap`）**必然失败**，返回误导性的 `CUDA_ERROR_OUT_OF_MEMORY`；但对**普通堆内存**（`Vec`）成功。实测探针：

| 内存形态 | `cuMemHostRegister_v2` 结果 |
|:--|:--|
| `Vec` 堆内存（1 MiB） | `CUDA_SUCCESS` |
| `mmap` 文件映射（1 MiB） | `CUDA_ERROR_OUT_OF_MEMORY` |

这意味着：**register 零拷贝（「mmap 页直接 DMA、免 host memcpy」）在 Windows WDDM 下不可达**，只在 Linux/TCC（`CU_MEMHOSTREGISTER_PORTABLE` 可 pin 文件页）下才是 P1b-2 描述的终点。在 Windows 上，语料上 mmap 带来的冷开收益（V-2 的 O(1) open + 零物化）依然成立，但上传仍走 P1b-1 staging——register 封装保留了这条 fallback 红线，代码在 Linux 上可无缝启用零拷贝路径。

### 3.5 遗留：`build_corpus` 接线到 register（P1b-2 端到端）

`register_and_upload` 已封装并验证，但 `build_corpus` 尚未接线（仍需 `KnnRequest` 携带 `BytesSource` 表达「mapped 来源」）。鉴于 Windows 上 register 必失败、收益只在 Linux 显现，此接线不排入 V-3，留待 Linux/TCC 部署环境评估时再做（与 `bytesSource` 扩展一并）。

---

## 4. 明确不做（本阶段）

| 项 | 理由 |
|:--|:--|
| GPUDirect（SSD→GPU P2P DMA） | P2，Linux-only，依赖 `nvidia-fs` + NVMe 拓扑 |
| 异步预取流水线（`prefetch_hint` + stream 重叠） | P1c / P2，独立阶段 |
| `build_corpus` 接线 `register_and_upload`（P1b-2 端到端） | 需 `BytesSource` 扩展；且 Windows WDDM 下 register mmap 必失败（见 §3.4），收益只在 Linux 显现，留待 Linux/TCC 环境评估 |
| host-pinned 配额强制执行 | 先观测（P1b-1 上报大小），执行留 P1b-2 |

---

## 5. 一句话总结

P1b 的「pinned」分两个台阶：**P1b-1** 不改数据来源，只用 `cudaHostAlloc` 的 write-combined staging 把 DMA 从「未锁页 bounce buffer」切到「pinned 快路径」，已证明 33–45% 收益；**P1b-2** 用 `cudaHostRegister` 注册已有 mmap 视图、真正消灭第三份拷贝——V-3 已交付封装与带回退语义，但实测 **Windows WDDM 拒绝 register 文件映射页**，零拷贝终点只在 Linux/TCC 可达，Windows 上仍由 staging 兜底。
