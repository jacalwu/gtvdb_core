//! CUDA backend (building with `--features cuda`).
//!
//! Two properties are load-bearing here:
//!
//! * **The corpus never round-trips per query.** A corpus is transposed to
//!   column-major once on first use and kept in device memory, keyed by
//!   `(name, version)`. A query uploads its own `dim` floats and downloads
//!   `blocks × k` candidates.
//! * **Every failure is recoverable.** No `expect`, no panic: a driver that is
//!   absent, NVRTC that is missing, an allocation that does not fit — all become
//!   [`AccelError`] values that the registry records before running the same
//!   operation on the host backend.
//!
//! Ties: the device merges block candidates by `(distance, id)` on the host, so
//! the final ordering matches the CPU path. Ties *inside* a block are resolved
//! by insertion order, which only matters when two rows have bit-identical
//! distances.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow::array::BooleanArray;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use gtv_core::Metric;

use crate::backend::{
    AccelBackend, AccelError, AccelOp, AsofRequest, DeviceInfo, DeviceKind, IvfRequest, KnnRequest,
    PrefetchRequest, RawHit,
};
use crate::resident::ResidentKind;
use crate::sweep::non_decreasing;

/// Largest `k` the register-resident candidate array supports.
pub const MAX_K: usize = 32;
/// MUST match `#define THREADS` in `kernels/knn_exact_topk.cu`: that kernel's
/// shared-memory staging area is sized `THREADS * KERNEL_K`, so launching more
/// threads than it was compiled for writes past the end of shared memory.
const THREADS: u32 = 128;
/// MUST match `#define TILE` in the transpose kernel.
const TILE: u32 = 32;
/// Cap the grid so the per-block serial merge (`THREADS * k` insertions on one
/// thread) does not dominate a large scan. With a grid-stride loop every vector
/// is still visited; fewer blocks just means more rows per thread and far less
/// merge work / fewer candidates to download.
const MAX_BLOCKS: u32 = 1024;
/// Target thread count for the as-of kernel; sets the per-thread chunk size.
const ASOF_TARGET_THREADS: usize = 256 * 512;
/// P1c: default row-block size of the corpus upload pipeline, in bytes of
/// row-major `f32` payload. One block is a single H2D copy and one windowed
/// transpose; with two blocks in flight the copy of N+1 overlaps the transpose
/// and scan of N. **8 MiB is the measured knee** (`bench_cold_upload_pipeline_vs_bulk`,
/// 195 MiB corpus on the P1a host): bulk 94.6 ms → 64 MiB 72.2 → 16 MiB 49.9 →
/// **8 MiB 46.9** → 4 MiB 47.0 → 1 MiB 57.7 (launch overhead). 8 MiB keeps the
/// transient double-buffer scratch at `2 × 8` MiB of device memory and the
/// pinned staging at the same.
pub const DEFAULT_PREFETCH_BLOCK_BYTES: usize = 8 * 1024 * 1024;

const KNN_SRC: &str = include_str!("../kernels/knn_exact_topk.cu");
const ASOF_SRC: &str = include_str!("../kernels/asof_merge_chunked.cu");

#[derive(Debug)]
struct DeviceCorpus {
    /// Column-major: `data_t[d * n + i]`.
    data_t: CudaSlice<f32>,
    n: i32,
    dim: i32,
}

#[derive(Debug)]
struct DeviceSeries {
    ts: CudaSlice<i64>,
    len: i32,
}

/// A reusable write-combined pinned host buffer used as the DMA staging area.
///
/// P1b-1 (`docs/pinned-upload-p1b.md`): uploading a pageable `Vec` through the
/// driver forces a hidden copy into an internal bounce buffer before the DMA,
/// which P1a measured at a 33–45% bandwidth loss. Copying into a `cudaHostAlloc`
/// (write-combined) buffer once, then DMA-ing from the locked pages, avoids the
/// bounce and lets the transfer run at the pinned ceiling.
///
/// The buffer is exact-sized and reallocated only when the element count
/// changes: a pinned allocation is *not* swappable by the OS, so re-using one
/// buffer per size is cheaper than re-allocating, and `clone_htod(&buf)`
/// transfers exactly `cap` elements (no uninitialised tail is ever DMA'd).
struct StagingBuffer<T: cudarc::driver::DeviceRepr> {
    /// `None` until the first upload allocates it.
    buf: Option<PinnedHostSlice<T>>,
    /// Element count (in `T`) the buffer currently holds.
    cap: usize,
}

impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy> StagingBuffer<T> {
    /// Reserve a buffer holding exactly `n` `T`s, reallocating only when the
    /// size changes.
    fn ensure(&mut self, n: usize, ctx: &Arc<CudaContext>) -> Result<&mut [T], AccelError> {
        if self.buf.is_none() || self.cap != n {
            // SAFETY: `alloc_pinned` is unsafe only because the memory is
            // uninitialised; we overwrite it via `copy_from_slice` before any
            // read, and `PinnedHostSlice` owns the lifetime / free on drop.
            let buf = unsafe { ctx.alloc_pinned::<T>(n) }
                .map_err(|e| launch_err(format!("pinned staging alloc: {e}")))?;
            self.buf = Some(buf);
            self.cap = n;
        }
        let buf = self
            .buf
            .as_mut()
            .expect("staging buffer allocated by ensure");
        // `as_mut_slice` synchronises the pinned buffer's event first, so we
        // never observe a half-written staging area from a prior async op.
        let slice = buf
            .as_mut_slice()
            .map_err(|e| launch_err(format!("pinned staging slice: {e}")))?;
        Ok(slice)
    }

    /// The current capacity in `T` elements (for telemetry / budget reporting).
    fn cap(&self) -> usize {
        self.cap
    }
}

impl<T: cudarc::driver::DeviceRepr> Default for StagingBuffer<T> {
    fn default() -> Self {
        Self { buf: None, cap: 0 }
    }
}

impl<T: cudarc::driver::DeviceRepr> std::fmt::Debug for StagingBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagingBuffer")
            .field("allocated", &self.buf.is_some())
            .field("cap", &self.cap)
            .field("bytes", &(self.cap * std::mem::size_of::<T>()))
            .finish()
    }
}

/// The query half of a fused top-K launch. The corpus half is the resident
/// [`DeviceCorpus`]; keeping everything else in one value stops the kernel
/// entry point from growing a positional argument list that is easy to
/// transpose at a call site.
#[derive(Debug, Clone, Copy)]
struct TopkQuery<'a> {
    ids: &'a [u64],
    query: &'a [f32],
    metric: Metric,
    normalized: bool,
    k: usize,
    mask: Option<&'a BooleanArray>,
    /// `None` => flat scan; `Some` => IVF probe positions.
    positions: Option<&'a [u32]>,
}

/// The query half of the P1c fused cold scan ([`CudaBackend::build_and_scan`]).
///
/// Everything [`TopkQuery`] carries except the row window: the pipeline
/// derives per-block positions itself (pre-applying the flat mask on the
/// host), so the kernel runs with `has_mask = 0` on every block.
#[derive(Debug, Clone, Copy)]
struct BlockScan<'a> {
    ids: &'a [u64],
    query: &'a [f32],
    metric: Metric,
    normalized: bool,
    k: usize,
    /// Label mask for the flat scan (row domain).
    flat_mask: Option<&'a BooleanArray>,
    /// IVF probe plan (`None` for the flat scan).
    positions: Option<&'a [u32]>,
}

impl BlockScan<'_> {
    /// The same scan expressed as a whole-corpus [`TopkQuery`] (cache-hit and
    /// race-fallback paths).
    fn as_topk_query(&self) -> TopkQuery<'_> {
        TopkQuery {
            ids: self.ids,
            query: self.query,
            metric: self.metric,
            normalized: self.normalized,
            k: self.k,
            mask: self.flat_mask,
            positions: self.positions,
        }
    }
}

/// A CUDA device with both kernels compiled through NVRTC.
pub struct CudaBackend {
    device: DeviceInfo,
    ctx: Arc<CudaContext>,
    /// NVRTC architecture option computed from the device we opened.
    arch: String,
    transpose_fn: CudaFunction,
    /// P1c windowed transpose: one row-block into its rows of the full
    /// column-major corpus (see `transpose_rowmajor_to_colmajor_window`).
    transpose_window_fn: CudaFunction,
    /// P1c: dedicated H2D stream. Block N+1's copy runs here while the default
    /// stream transposes block N.
    copy_stream: Arc<CudaStream>,
    /// P1c pipeline block size in bytes of row-major `f32` payload; `0` builds
    /// the corpus as one bulk copy (pre-P1c behavior).
    prefetch_block_bytes: usize,
    /// One compiled kernel specialization per distinct `k`, so `KERNEL_K` is a
    /// compile-time constant and the candidate arrays stay in registers.
    knn_functions: Mutex<HashMap<usize, Arc<CudaFunction>>>,
    asof_fn: CudaFunction,
    corpora: Mutex<HashMap<(String, u64), Arc<DeviceCorpus>>>,
    /// Cell-ordered IVF corpora. Separate from `corpora`: a flat index and an
    /// IVF index can share a collection name, but their row order (and hence
    /// every returned position's meaning) differs.
    ivf_corpora: Mutex<HashMap<(String, u64), Arc<DeviceCorpus>>>,
    series: Mutex<HashMap<(String, u64), Arc<DeviceSeries>>>,
    /// Two pinned DMA staging buffers (P1b-1) for the corpus upload pipeline:
    /// the host fills slot `b % 2` for block `b` while the DMA of block `b-1`
    /// is still in flight from the other slot.
    staging_f32: Mutex<[StagingBuffer<f32>; 2]>,
    /// Pinned DMA staging buffer for the `i64` as-of series upload (P1b-1).
    staging_i64: Mutex<StagingBuffer<i64>>,
    /// `ACCEL_GPUDIRECT`: whether a large upload *may* skip the staging copy.
    gpudirect_enabled: bool,
    /// `ACCEL_GPUDIRECT_THRESHOLD`: the size below which staging always wins.
    gpudirect_threshold: u64,
    /// What [`crate::gpudirect::probe`] reported for this host, cached so no
    /// upload pays for the probe.
    gpudirect_status: crate::gpudirect::DirectStatus,
    /// Uploads that actually took the direct path (0 until a `cufile` binding
    /// exists — see the module docs).
    gpudirect_direct: std::sync::atomic::AtomicU64,
    /// Uploads that took the staged path, the GPUDirect-enabled ones included.
    gpudirect_staged: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let corpora = self.corpora.lock().map(|m| m.len()).unwrap_or(0);
        let ivf = self.ivf_corpora.lock().map(|m| m.len()).unwrap_or(0);
        let series = self.series.lock().map(|m| m.len()).unwrap_or(0);
        f.debug_struct("CudaBackend")
            .field("device", &self.device)
            .field("corpora", &corpora)
            .field("ivf_corpora", &ivf)
            .field("series", &series)
            .finish()
    }
}

fn launch_err<E: std::fmt::Display>(e: E) -> AccelError {
    AccelError::Launch(e.to_string())
}

impl CudaBackend {
    /// Open `ordinal` and compile both kernels.
    ///
    /// Returns [`AccelError::NoDevice`] when there is no driver or no device at
    /// that ordinal, so a caller can treat "no GPU" as an ordinary condition.
    pub fn probe(ordinal: usize) -> Result<Self, AccelError> {
        let ctx = CudaContext::new(ordinal).map_err(|e| AccelError::NoDevice(e.to_string()))?;
        let name = ctx
            .name()
            .unwrap_or_else(|_| format!("cuda:{ordinal}"));
        let total = ctx.total_mem().unwrap_or(0) as u64;

        let (major, minor) = ctx.compute_capability().unwrap_or((7, 5));
        // NVRTC's default virtual arch targets a GPU generation that a CUDA 13
        // driver no longer JITs (CUDA_ERROR_INVALID_PTX). Compile for the device
        // we actually opened. `options` (rather than `arch`) lets the string be
        // built at runtime from the probed capability.
        let arch = format!("--gpu-architecture=compute_{major}{minor}");
        let opts = || CompileOptions {
            options: vec![arch.clone()],
            ..Default::default()
        };

        let knn_ptx = compile_ptx_with_opts(KNN_SRC, opts())
            .map_err(|e| launch_err(format!("nvrtc knn: {e}")))?;
        let knn_module = ctx
            .load_module(knn_ptx)
            .map_err(|e| launch_err(format!("load knn module: {e}")))?;
        let transpose_fn = knn_module
            .load_function("transpose_rowmajor_to_colmajor")
            .map_err(|e| launch_err(format!("load transpose kernel: {e}")))?;
        let transpose_window_fn = knn_module
            .load_function("transpose_rowmajor_to_colmajor_window")
            .map_err(|e| launch_err(format!("load windowed transpose kernel: {e}")))?;

        let asof_ptx = compile_ptx_with_opts(ASOF_SRC, opts())
            .map_err(|e| launch_err(format!("nvrtc asof: {e}")))?;
        let asof_module = ctx
            .load_module(asof_ptx)
            .map_err(|e| launch_err(format!("load asof module: {e}")))?;
        let asof_fn = asof_module
            .load_function("asof_merge_chunked_kernel")
            .map_err(|e| launch_err(format!("load asof kernel: {e}")))?;

        // P1c: a second stream for H2D corpus copies. Creating it flips the
        // context into cudarc's multi-stream mode, which activates per-slice
        // event tracking — that is what makes the copy/transpose overlap safe
        // (each kernel launch waits on its input's copy event automatically).
        let copy_stream = ctx
            .new_stream()
            .map_err(|e| launch_err(format!("create copy stream: {e}")))?;

        Ok(Self {
            device: DeviceInfo {
                name,
                kind: DeviceKind::Cuda,
                total_bytes: total,
            },
            ctx,
            arch,
            transpose_fn,
            transpose_window_fn,
            copy_stream,
            prefetch_block_bytes: DEFAULT_PREFETCH_BLOCK_BYTES,
            knn_functions: Mutex::new(HashMap::new()),
            asof_fn,
            corpora: Mutex::new(HashMap::new()),
            ivf_corpora: Mutex::new(HashMap::new()),
            series: Mutex::new(HashMap::new()),
            // Allocated lazily on the first upload of each element type.
            staging_f32: Mutex::new([StagingBuffer::default(), StagingBuffer::default()]),
            staging_i64: Mutex::new(StagingBuffer::default()),
            // Off until `from_resolved` says otherwise; probing here would be a
            // filesystem hit on a path that may never be used.
            gpudirect_enabled: false,
            gpudirect_threshold: 1 << 30,
            gpudirect_status: crate::gpudirect::DirectStatus::Unavailable("not configured"),
            gpudirect_direct: std::sync::atomic::AtomicU64::new(0),
            gpudirect_staged: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Set the P1c upload-pipeline block size in bytes of row-major `f32`
    /// payload. `0` restores the pre-P1c single bulk copy. Default
    /// [`DEFAULT_PREFETCH_BLOCK_BYTES`].
    pub fn with_prefetch_block_bytes(mut self, bytes: usize) -> Self {
        self.prefetch_block_bytes = bytes;
        self
    }

    /// The configured P1c block size (bytes of row-major payload per block).
    pub fn prefetch_block_bytes(&self) -> usize {
        self.prefetch_block_bytes
    }

    /// Enable/disable the GPUDirect Decision (`ACCEL_GPUDIRECT`) and give the
    /// size threshold above which a direct transfer would be attempted.
    ///
    /// Probing happens once here rather than per upload. The probe is allowed to
    /// refuse on any host — including every host today, because no `cufile`
    /// binding is compiled in — and the upload simply stays staged.
    pub fn with_gpudirect(mut self, enabled: bool, threshold: u64) -> Self {
        self.gpudirect_enabled = enabled;
        self.gpudirect_threshold = threshold;
        self.gpudirect_status = if enabled {
            crate::gpudirect::probe()
        } else {
            crate::gpudirect::DirectStatus::Unavailable("ACCEL_GPUDIRECT is off")
        };
        self
    }

    /// `(direct, staged)` upload counts and the probe's verdict.
    pub fn gpudirect_stats(&self) -> (u64, u64) {
        (
            self.gpudirect_direct.load(std::sync::atomic::Ordering::Relaxed),
            self.gpudirect_staged.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// The cached GPUDirect probe verdict (for `.accel` output and tests).
    pub fn gpudirect_status(&self) -> crate::gpudirect::DirectStatus {
        self.gpudirect_status
    }

    /// Decide, record and return the upload path for a `bytes`-sized corpus.
    ///
    /// Every upload goes through this exactly once, so the telemetry can never
    /// claim a direct transfer that did not happen: `Direct` is only reachable
    /// when the probe reported `Available`, which requires the binding.
    fn note_upload_path(&self, bytes: u64) -> crate::gpudirect::UploadPath {
        use std::sync::atomic::Ordering;
        let path = crate::gpudirect::upload_plan(
            self.gpudirect_enabled,
            self.gpudirect_status,
            bytes,
            self.gpudirect_threshold,
        );
        match path {
            crate::gpudirect::UploadPath::Direct => {
                self.gpudirect_direct.fetch_add(1, Ordering::Relaxed);
            }
            crate::gpudirect::UploadPath::Staged => {
                self.gpudirect_staged.fetch_add(1, Ordering::Relaxed);
            }
        }
        path
    }

    /// Rows per pipeline block for a corpus of dimension `dim`. At least 1;
    /// a zero configured block size (or a corpus narrower than one block)
    /// collapses to a single block.
    fn prefetch_rows(&self, dim: usize, n: usize) -> usize {
        if self.prefetch_block_bytes == 0 || n == 0 {
            return n.max(1);
        }
        let row_bytes = dim.max(1) * std::mem::size_of::<f32>();
        (self.prefetch_block_bytes / row_bytes).clamp(1, n.max(1))
    }

    /// Compile (once) the K-NN specialization for this exact `k`.
    fn knn_function(&self, k: usize) -> Result<Arc<CudaFunction>, AccelError> {
        if let Ok(map) = self.knn_functions.lock() {
            if let Some(hit) = map.get(&k) {
                return Ok(hit.clone());
            }
        }
        // Prepending the define avoids depending on NVRTC's -D option syntax.
        let src = format!("#define KERNEL_K {k}\n{KNN_SRC}");
        let opts = CompileOptions {
            options: vec![self.arch.clone()],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(src, opts)
            .map_err(|e| launch_err(format!("nvrtc knn k={k}: {e}")))?;
        let module = self
            .ctx
            .load_module(ptx)
            .map_err(|e| launch_err(format!("load knn module k={k}: {e}")))?;
        let func = Arc::new(
            module
                .load_function("knn_exact_topk_kernel")
                .map_err(|e| launch_err(format!("load knn kernel k={k}: {e}")))?,
        );
        if let Ok(mut map) = self.knn_functions.lock() {
            map.insert(k, func.clone());
        }
        Ok(func)
    }

    /// Number of device buffers still held by this backend (leak check).
    pub fn cached_blocks(&self) -> usize {
        let corpora = self.corpora.lock().map(|m| m.len()).unwrap_or(0);
        let ivf = self.ivf_corpora.lock().map(|m| m.len()).unwrap_or(0);
        let series = self.series.lock().map(|m| m.len()).unwrap_or(0);
        corpora + ivf + series
    }

    /// Total host-pinned staging bytes currently held (P1b-1 telemetry).
    ///
    /// This is the write-combined `cudaHostAlloc` footprint: it is *not* device
    /// memory, but it is non-swappable, so it matters for the host budget even
    /// though it never shows up in `mem_info`.
    pub fn staging_bytes(&self) -> usize {
        let f32: usize = self
            .staging_f32
            .lock()
            .map(|s| s.iter().map(|b| b.cap()).sum())
            .unwrap_or(0);
        let i64 = self.staging_i64.lock().map(|s| s.cap()).unwrap_or(0);
        f32 * std::mem::size_of::<f32>() + i64 * std::mem::size_of::<i64>()
    }

    /// Wait for all queued work on this context. Used after a device copy is
    /// released so a subsequent `mem_get_info` observes the freed memory.
    pub fn synchronize(&self) -> Result<(), AccelError> {
        self.ctx
            .synchronize()
            .map_err(|e| launch_err(format!("device synchronize: {e}")))
    }

    /// Fetch or build the device copy of the flat corpus for `req`.
    fn corpus(&self, req: &KnnRequest<'_>) -> Result<Arc<DeviceCorpus>, AccelError> {
        self.build_corpus(req.name, req.version, req.data, req.dim, &self.corpora)
    }

    /// Fetch or build the device copy of the cell-ordered IVF corpus.
    fn ivf_corpus(&self, req: &IvfRequest<'_>) -> Result<Arc<DeviceCorpus>, AccelError> {
        self.build_corpus(req.name, req.version, req.data, req.dim, &self.ivf_corpora)
    }

    /// One bulk upload + device transpose, cached in `cache`.
    ///
    /// Both the flat and the IVF corpus use this identical layout (column-major
    /// over *their own* row order); only the position list differs at query
    /// time. Sharing the upload is what keeps the two device paths from drifting.
    ///
    /// P1c: corpora larger than one prefetch block upload as a **block
    /// pipeline** — block N+1's H2D copy (on [`Self::copy_stream`]) overlaps
    /// the windowed transpose of block N (on the default stream), and the
    /// pinned staging is double-buffered so the host fill of the next block
    /// never waits on the previous DMA ("算第 N 块时预取第 N+1 块"). cudarc's
    /// per-slice events order the streams: a transpose launch waits on its own
    /// block's copy, and a staging slot's reuse waits on the transpose that
    /// read it. Single-block corpora take the original one-bulk path unchanged.
    fn build_corpus(
        &self,
        name: &str,
        version: u64,
        data: &[f32],
        dim: usize,
        cache: &Mutex<HashMap<(String, u64), Arc<DeviceCorpus>>>,
    ) -> Result<Arc<DeviceCorpus>, AccelError> {
        let key = (name.to_string(), version);
        if let Ok(map) = cache.lock() {
            if let Some(hit) = map.get(&key) {
                return Ok(hit.clone());
            }
        }

        let n = data.len() / dim.max(1);
        let stream = self.ctx.default_stream();

        // GPUDirect decision for this transfer, taken before any staging work.
        // A `Direct` plan would skip the pinned buffer entirely; today's build
        // always plans `Staged` because no `cufile` binding is compiled in, and
        // the counter records which one actually ran.
        let _upload_path = self.note_upload_path((data.len() * std::mem::size_of::<f32>()) as u64);

        // P1c-1: fault cold host pages in before the upload reads them, so
        // first-touch page faults land outside the critical path.
        crate::warm::warm_slice(data);

        let block_rows = self.prefetch_rows(dim, n);

        if n <= block_rows {
            // Single block: the original one-bulk upload + transpose (P1b-1:
            // stage the pageable source through a write-combined pinned buffer
            // so DMA runs at the pinned ceiling). Behavior unchanged.
            let d_rowmajor = {
                let mut s = self
                    .staging_f32
                    .lock()
                    .map_err(|_| launch_err("pinned staging lock poisoned"))?;
                {
                    let slice = s[0].ensure(data.len(), &self.ctx)?;
                    slice.copy_from_slice(data);
                } // drop the `&mut` borrow before re-borrowing immutably
                let buf = s[0].buf.as_ref().expect("staging allocated by ensure");
                stream
                    .clone_htod(buf)
                    .map_err(|e| launch_err(format!("upload corpus: {e}")))?
            };
            let mut data_t = stream
                .alloc_zeros::<f32>(n * dim)
                .map_err(|e| launch_err(format!("alloc column-major corpus: {e}")))?;

            let tiles_x = (dim as u32).div_ceil(TILE);
            let tiles_y = (n as u32).div_ceil(TILE);
            let cfg = LaunchConfig {
                grid_dim: (tiles_x * tiles_y, 1, 1),
                block_dim: (TILE, TILE, 1),
                shared_mem_bytes: 0,
            };
            let rows = n as i32;
            let dims = dim as i32;
            let tiles_xi = tiles_x as i32;
            let mut lb = stream.launch_builder(&self.transpose_fn);
            lb.arg(&d_rowmajor)
                .arg(&mut data_t)
                .arg(&rows)
                .arg(&dims)
                .arg(&tiles_xi);
            unsafe { lb.launch(cfg) }.map_err(|e| launch_err(format!("transpose launch: {e}")))?;
            // Ensure the transpose finished before the row-major allocation can
            // be reused by the driver; then release it.
            self.ctx
                .synchronize()
                .map_err(|e| launch_err(format!("sync after transpose: {e}")))?;
            drop(d_rowmajor);

            return self.cache_corpus(cache, key, name, version, data_t, n, dim);
        }

        // P1c pipeline: double-buffered blocks.
        let mut data_t = stream
            .alloc_zeros::<f32>(n * dim)
            .map_err(|e| launch_err(format!("alloc column-major corpus: {e}")))?;
        let block_elems = block_rows * dim;
        let dims = dim as i32;
        let n_total = n as i32;
        let tiles_x = (dim as u32).div_ceil(TILE);
        let tiles_xi = tiles_x as i32;
        let mut scratch: [Option<CudaSlice<f32>>; 2] = [None, None];

        for b in 0..n.div_ceil(block_rows) {
            let r0 = b * block_rows;
            let n_in = block_rows.min(n - r0);
            let slot = b % 2;
            let rows = &data[r0 * dim..(r0 + n_in) * dim];

            {
                let mut staging = self
                    .staging_f32
                    .lock()
                    .map_err(|_| launch_err("pinned staging lock poisoned"))?;
                {
                    let slice = staging[slot].ensure(rows.len(), &self.ctx)?;
                    slice.copy_from_slice(rows);
                } // drop the `&mut` borrow before re-borrowing immutably
                let buf = staging[slot]
                    .buf
                    .as_ref()
                    .expect("staging allocated by ensure");
                // Reusable row-major scratch for this slot, full block size so
                // the final short block does not reallocate.
                if scratch[slot].as_ref().map_or(true, |s| s.len() < block_elems) {
                    scratch[slot] = Some(
                        self.copy_stream
                            .alloc_zeros::<f32>(block_elems)
                            .map_err(|e| launch_err(format!("alloc block scratch: {e}")))?,
                    );
                }
                let dst = scratch[slot].as_mut().expect("scratch allocated above");
                self.copy_stream
                    .memcpy_htod(buf, dst)
                    .map_err(|e| launch_err(format!("upload corpus block: {e}")))?;
            }

            // Windowed transpose into rows [r0, r0 + n_in) of the full
            // column-major corpus. The launch waits on this block's H2D copy
            // (cudarc records the copy as the scratch's write event), while
            // the copy of block N+1 is already being enqueued on `copy_stream`.
            let tiles_y = (n_in as u32).div_ceil(TILE);
            let cfg = LaunchConfig {
                grid_dim: (tiles_x * tiles_y, 1, 1),
                block_dim: (TILE, TILE, 1),
                shared_mem_bytes: 0,
            };
            let row0 = r0 as i32;
            let n_ini = n_in as i32;
            let d_block = scratch[slot].as_ref().expect("scratch allocated above");
            let mut lb = stream.launch_builder(&self.transpose_window_fn);
            lb.arg(d_block)
                .arg(&mut data_t)
                .arg(&n_total)
                .arg(&dims)
                .arg(&tiles_xi)
                .arg(&row0)
                .arg(&n_ini);
            unsafe { lb.launch(cfg) }
                .map_err(|e| launch_err(format!("windowed transpose launch: {e}")))?;
        }

        // Drain the pipeline before the scratch is freed and the corpus is
        // published; then the row-major blocks are released in `scratch`'s drop.
        self.ctx
            .synchronize()
            .map_err(|e| launch_err(format!("sync after block pipeline: {e}")))?;
        drop(scratch);

        self.cache_corpus(cache, key, name, version, data_t, n, dim)
    }

    /// Wrap the transposed corpus and record it in `cache`, keeping one version
    /// per name (an older version is dead weight).
    fn cache_corpus(
        &self,
        cache: &Mutex<HashMap<(String, u64), Arc<DeviceCorpus>>>,
        key: (String, u64),
        name: &str,
        version: u64,
        data_t: CudaSlice<f32>,
        n: usize,
        dim: usize,
    ) -> Result<Arc<DeviceCorpus>, AccelError> {
        let corpus = Arc::new(DeviceCorpus {
            data_t,
            n: n as i32,
            dim: dim as i32,
        });
        if let Ok(mut map) = cache.lock() {
            map.retain(|(n, v), _| !(n == name && *v != version));
            map.insert(key, corpus.clone());
        }
        Ok(corpus)
    }

    /// Fetch or build the device copy of the right-hand series for `req`.
    fn device_series(&self, req: &AsofRequest<'_>) -> Result<Arc<DeviceSeries>, AccelError> {
        let key = (req.name.to_string(), req.version);
        if let Ok(map) = self.series.lock() {
            if let Some(hit) = map.get(&key) {
                return Ok(hit.clone());
            }
        }
        let stream = self.ctx.default_stream();
        let ts = {
            let mut s = self
                .staging_i64
                .lock()
                .map_err(|_| launch_err("pinned staging lock poisoned"))?;
            {
                let slice = s.ensure(req.right_times.len(), &self.ctx)?;
                slice.copy_from_slice(req.right_times);
            }
            let buf = s.buf.as_ref().expect("staging allocated by ensure");
            stream
                .clone_htod(buf)
                .map_err(|e| launch_err(format!("upload series: {e}")))?
        };
        let series = Arc::new(DeviceSeries {
            ts,
            len: req.right_times.len() as i32,
        });
        if let Ok(mut map) = self.series.lock() {
            map.retain(|(name, version), _| !(name == req.name && *version != req.version));
            map.insert(key, series.clone());
        }
        Ok(series)
    }

    /// The one fused top-K launch, shared by the flat scan and the IVF probe.
    ///
    /// `positions: None` means "scan every row"; `Some` means "scan these
    /// corpus positions" (an IVF probe plan, already mask-filtered on the
    /// host). `normalized` triggers the same query normalization the host
    /// kernel does — without it a non-unit cosine query would rank correctly
    /// but report the wrong distance.
    fn topk(
        &self,
        corpus: Arc<DeviceCorpus>,
        req: TopkQuery<'_>,
    ) -> Result<Vec<RawHit>, AccelError> {
        let TopkQuery {
            ids,
            query,
            metric,
            normalized,
            k,
            mask,
            positions,
        } = req;
        let n = corpus.n as usize;
        let dim = corpus.dim as usize;
        if k == 0 || n == 0 {
            return Ok(Vec::new());
        }
        if k > MAX_K {
            return Err(AccelError::Invalid(format!(
                "k={k} exceeds the device MAX_K={MAX_K}; use the host path"
            )));
        }
        if dim == 0 || query.len() != dim {
            return Err(AccelError::Invalid(format!(
                "query dim {} != corpus dim {}",
                query.len(),
                dim
            )));
        }
        if ids.len() != n {
            return Err(AccelError::Invalid("corpus shape mismatch".into()));
        }

        let query_host: Cow<'_, [f32]> = if normalized {
            let mut q = query.to_vec();
            metric.normalize_in_place(&mut q);
            Cow::Owned(q)
        } else {
            Cow::Borrowed(query)
        };

        // The host scan silently drops out-of-range positions; mirror that
        // here so both surfaces see the same candidate set.
        let pos_host: Option<Vec<i32>> = positions.map(|p| {
            p.iter()
                .copied()
                .filter(|&x| (x as usize) < n)
                .map(|x| x as i32)
                .collect()
        });
        let npos = pos_host.as_ref().map_or(n, Vec::len);
        if npos == 0 {
            return Ok(Vec::new());
        }

        let knn_fn = self.knn_function(k)?;
        let stream = self.ctx.default_stream();
        let d_query = stream
            .clone_htod(query_host.as_ref())
            .map_err(|e| launch_err(format!("upload query: {e}")))?;

        // A single-element dummy keeps the kernel's mask pointer valid when
        // unfiltered; `has_mask` gates the read. IVF always passes `has_mask=0`
        // because its positions were filtered on the host.
        let (mask_bytes, has_mask) = match mask {
            Some(m) => (
                (0..m.len()).map(|i| m.value(i) as u8).collect::<Vec<u8>>(),
                1i32,
            ),
            None => (vec![1u8], 0i32),
        };
        let d_mask = stream
            .clone_htod(&mask_bytes)
            .map_err(|e| launch_err(format!("upload mask: {e}")))?;

        let d_pos = match &pos_host {
            Some(p) => Some(
                stream
                    .clone_htod(p)
                    .map_err(|e| launch_err(format!("upload probe positions: {e}")))?,
            ),
            None => None,
        };
        // Pointer-sized zero: the kernel receives a NULL `const int*`.
        let null_pos = 0u64;

        let blocks = (npos as u32).div_ceil(THREADS).clamp(1, MAX_BLOCKS);
        let candidates = blocks as usize * k;
        let mut d_out_id = stream
            .alloc_zeros::<i32>(candidates)
            .map_err(|e| launch_err(format!("alloc candidates: {e}")))?;
        let mut d_out_dist = stream
            .alloc_zeros::<f32>(candidates)
            .map_err(|e| launch_err(format!("alloc distances: {e}")))?;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = corpus.n;
        let dim_i = corpus.dim;
        let npos_i = npos as i32;
        let metric_i = metric_code(metric);
        let mut lb = stream.launch_builder(&knn_fn);
        lb.arg(&corpus.data_t)
            .arg(&d_query)
            .arg(&d_mask)
            .arg(&has_mask)
            .arg(&n_i)
            .arg(&npos_i);
        match &d_pos {
            Some(p) => {
                lb.arg(p);
            }
            None => {
                lb.arg(&null_pos);
            }
        }
        lb.arg(&dim_i)
            .arg(&metric_i)
            .arg(&mut d_out_id)
            .arg(&mut d_out_dist);
        unsafe { lb.launch(cfg) }.map_err(|e| launch_err(format!("topk launch: {e}")))?;

        let out_ids: Vec<i32> = stream
            .clone_dtoh(&d_out_id)
            .map_err(|e| launch_err(format!("download ids: {e}")))?;
        let dists: Vec<f32> = stream
            .clone_dtoh(&d_out_dist)
            .map_err(|e| launch_err(format!("download distances: {e}")))?;

        // Position -> id, then the same comparator the CPU path uses.
        let mut merged: Vec<RawHit> = Vec::with_capacity(k);
        collect_hits(ids, &out_ids, &dists, &mut merged);
        merged.sort_by(hits_order);
        merged.truncate(k);
        Ok(merged)
    }

    /// P1c-3: the fused cold path. Runs the same block pipeline as
    /// [`Self::build_corpus`], but scans each block with the top-K kernel as
    /// soon as its transpose has been enqueued — the kernel over block N
    /// overlaps the H2D copy of block N+1 ("算第 N 块时预取第 N+1 块") — and
    /// merges the per-block candidates on the host. Exact top-K decomposes
    /// over disjoint row sets, so the merged answer is identical to scanning
    /// the finished corpus (which is what the next query does via the cache).
    ///
    /// The corpus is published to `cache` exactly like [`Self::build_corpus`]
    /// would; callers use this only on a cache miss with more than one block.
    fn build_and_scan(
        &self,
        name: &str,
        version: u64,
        data: &[f32],
        dim: usize,
        cache: &Mutex<HashMap<(String, u64), Arc<DeviceCorpus>>>,
        scan: &BlockScan<'_>,
    ) -> Result<Vec<RawHit>, AccelError> {
        let key = (name.to_string(), version);
        if let Ok(map) = cache.lock() {
            if let Some(hit) = map.get(&key) {
                // Lost a race with a concurrent build: scan what is there.
                return self.topk(hit.clone(), scan.as_topk_query());
            }
        }

        let n = data.len() / dim.max(1);
        if scan.k == 0 || n == 0 {
            return Ok(Vec::new());
        }
        if scan.k > MAX_K {
            return Err(AccelError::Invalid(format!(
                "k={} exceeds the device MAX_K={}; use the host path",
                scan.k, MAX_K
            )));
        }
        if dim == 0 || scan.query.len() != dim {
            return Err(AccelError::Invalid(format!(
                "query dim {} != corpus dim {}",
                scan.query.len(),
                dim
            )));
        }
        if scan.ids.len() != n {
            return Err(AccelError::Invalid("corpus shape mismatch".into()));
        }

        let stream = self.ctx.default_stream();
        crate::warm::warm_slice(data);

        let query_host: Cow<'_, [f32]> = if scan.normalized {
            let mut q = scan.query.to_vec();
            scan.metric.normalize_in_place(&mut q);
            Cow::Owned(q)
        } else {
            Cow::Borrowed(scan.query)
        };
        let knn_fn = self.knn_function(scan.k)?;
        // The query is uploaded once; the dummy mask keeps the kernel's mask
        // pointer valid (`has_mask = 0`). Flat windows are pre-filtered by the
        // label mask on the host here, exactly like the IVF probe plan always
        // has been — so the mask uploads once per scan, not once per block.
        let d_query = stream
            .clone_htod(query_host.as_ref())
            .map_err(|e| launch_err(format!("upload query: {e}")))?;
        let d_mask = stream
            .clone_htod(&[1u8])
            .map_err(|e| launch_err(format!("upload mask: {e}")))?;
        let has_mask = 0i32;

        let block_rows = self.prefetch_rows(dim, n);
        let mut data_t = stream
            .alloc_zeros::<f32>(n * dim)
            .map_err(|e| launch_err(format!("alloc column-major corpus: {e}")))?;
        let block_elems = block_rows * dim;
        let dims = dim as i32;
        let n_total = n as i32;
        let tiles_x = (dim as u32).div_ceil(TILE);
        let tiles_xi = tiles_x as i32;
        let metric_i = metric_code(scan.metric);
        let k = scan.k;
        let mut scratch: [Option<CudaSlice<f32>>; 2] = [None, None];
        let mut pending: Vec<(CudaSlice<i32>, CudaSlice<f32>)> = Vec::new();

        for b in 0..n.div_ceil(block_rows) {
            let r0 = b * block_rows;
            let n_in = block_rows.min(n - r0);
            let slot = b % 2;
            let rows = &data[r0 * dim..(r0 + n_in) * dim];

            {
                let mut staging = self
                    .staging_f32
                    .lock()
                    .map_err(|_| launch_err("pinned staging lock poisoned"))?;
                {
                    let slice = staging[slot].ensure(rows.len(), &self.ctx)?;
                    slice.copy_from_slice(rows);
                }
                let buf = staging[slot]
                    .buf
                    .as_ref()
                    .expect("staging allocated by ensure");
                if scratch[slot].as_ref().map_or(true, |s| s.len() < block_elems) {
                    scratch[slot] = Some(
                        self.copy_stream
                            .alloc_zeros::<f32>(block_elems)
                            .map_err(|e| launch_err(format!("alloc block scratch: {e}")))?,
                    );
                }
                let dst = scratch[slot].as_mut().expect("scratch allocated above");
                self.copy_stream
                    .memcpy_htod(buf, dst)
                    .map_err(|e| launch_err(format!("upload corpus block: {e}")))?;
            }

            // Transpose block b into rows [r0, r0 + n_in) — waits on this
            // block's copy (cudarc slice events).
            let tiles_y = (n_in as u32).div_ceil(TILE);
            let cfg = LaunchConfig {
                grid_dim: (tiles_x * tiles_y, 1, 1),
                block_dim: (TILE, TILE, 1),
                shared_mem_bytes: 0,
            };
            let row0 = r0 as i32;
            let n_ini = n_in as i32;
            let d_block = scratch[slot].as_ref().expect("scratch allocated above");
            let mut lb = stream.launch_builder(&self.transpose_window_fn);
            lb.arg(d_block)
                .arg(&mut data_t)
                .arg(&n_total)
                .arg(&dims)
                .arg(&tiles_xi)
                .arg(&row0)
                .arg(&n_ini);
            unsafe { lb.launch(cfg) }
                .map_err(|e| launch_err(format!("windowed transpose launch: {e}")))?;

            // Scan block b right away on the same stream: the kernel waits for
            // the transpose, while block b+1's copy is already in flight.
            let pos = Self::block_positions(scan, r0, n_in);
            if pos.is_empty() {
                continue;
            }
            let npos = pos.len();
            let d_pos = stream
                .clone_htod(&pos)
                .map_err(|e| launch_err(format!("upload block positions: {e}")))?;
            let grid = (npos as u32).div_ceil(THREADS).clamp(1, MAX_BLOCKS);
            let candidates = grid as usize * k;
            let mut d_out_id = stream
                .alloc_zeros::<i32>(candidates)
                .map_err(|e| launch_err(format!("alloc candidates: {e}")))?;
            let mut d_out_dist = stream
                .alloc_zeros::<f32>(candidates)
                .map_err(|e| launch_err(format!("alloc distances: {e}")))?;
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let npos_i = npos as i32;
            let mut lb = stream.launch_builder(&knn_fn);
            lb.arg(&data_t)
                .arg(&d_query)
                .arg(&d_mask)
                .arg(&has_mask)
                .arg(&n_total)
                .arg(&npos_i)
                .arg(&d_pos)
                .arg(&dims)
                .arg(&metric_i)
                .arg(&mut d_out_id)
                .arg(&mut d_out_dist);
            unsafe { lb.launch(cfg) }.map_err(|e| launch_err(format!("topk launch: {e}")))?;
            // Candidate downloads are deferred to one pass after the pipeline
            // drains — a per-block `clone_dtoh` here would synchronize the
            // stream and hide the very overlap this path exists for.
            pending.push((d_out_id, d_out_dist));
        }

        self.ctx
            .synchronize()
            .map_err(|e| launch_err(format!("sync after fused scan: {e}")))?;
        let mut merged: Vec<RawHit> = Vec::new();
        for (d_out_id, d_out_dist) in &pending {
            let out_ids: Vec<i32> = stream
                .clone_dtoh(d_out_id)
                .map_err(|e| launch_err(format!("download ids: {e}")))?;
            let dists: Vec<f32> = stream
                .clone_dtoh(d_out_dist)
                .map_err(|e| launch_err(format!("download distances: {e}")))?;
            collect_hits(scan.ids, &out_ids, &dists, &mut merged);
        }
        drop(pending);
        drop(scratch);
        merged.sort_by(hits_order);
        merged.truncate(k);

        // Publish the finished corpus exactly like `build_corpus` would.
        self.cache_corpus(cache, key, name, version, data_t, n, dim)?;
        Ok(merged)
    }

    /// Rows of the block window `[r0, r0 + n_in)` as kernel positions.
    ///
    /// The flat scan pre-applies the label mask here (so the kernel runs with
    /// `has_mask = 0`); the IVF probe contributes only the planned positions
    /// that fall inside this window.
    fn block_positions(scan: &BlockScan<'_>, r0: usize, n_in: usize) -> Vec<i32> {
        match scan.positions {
            Some(all) => all
                .iter()
                .copied()
                .filter(|&p| (p as usize) >= r0 && (p as usize) < r0 + n_in)
                .map(|p| p as i32)
                .collect(),
            None => (r0..r0 + n_in)
                .filter(|&r| scan.flat_mask.map_or(true, |m| m.value(r)))
                .map(|r| r as i32)
                .collect(),
        }
    }
}

/// Map `(out_row, distance)` candidate pairs from a kernel download into
/// `RawHit`s appended to `into`. Out rows are corpus row indices; `-1` marks
/// an unused candidate slot.
fn collect_hits(ids: &[u64], out_ids: &[i32], dists: &[f32], into: &mut Vec<RawHit>) {
    into.extend(
        dists
            .iter()
            .zip(out_ids.iter())
            .filter(|(_, &row)| row >= 0)
            .filter_map(|(&distance, &row)| {
                ids.get(row as usize).map(|&id| RawHit { distance, id })
            }),
    );
}

/// The host comparator: nearest-first, ties broken by ascending id — the same
/// ordering `gtv_index::exact_topk` produces.
fn hits_order(a: &RawHit, b: &RawHit) -> std::cmp::Ordering {
    a.distance
        .partial_cmp(&b.distance)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.id.cmp(&b.id))
}

fn metric_code(metric: Metric) -> i32 {
    match metric {
        Metric::L2 => 0,
        Metric::Cosine => 1,
        Metric::Ip => 2,
    }
}

impl AccelBackend for CudaBackend {
    fn device(&self) -> &DeviceInfo {
        &self.device
    }

    fn supports(&self, op: AccelOp) -> bool {
        matches!(
            op,
            AccelOp::ExactKnn | AccelOp::IvfProbe | AccelOp::AsofJoin
        )
    }

    fn exact_knn(&self, req: &KnnRequest<'_>) -> Result<Vec<RawHit>, AccelError> {
        if req.dim == 0 || req.data.len() != req.ids.len() * req.dim {
            return Err(AccelError::Invalid("corpus shape mismatch".into()));
        }
        if req.ids.len() > i32::MAX as usize {
            return Err(AccelError::Invalid("corpus shape overflow".into()));
        }
        let scan = BlockScan {
            ids: req.ids,
            query: req.query,
            metric: req.metric,
            normalized: req.normalized,
            k: req.k,
            flat_mask: req.mask,
            positions: None,
        };
        // Cache hit -> whole-corpus scan (steady state, untouched).
        // Cold + multi-block -> P1c fused pipeline (build_and_scan).
        // Cold + single block -> plain build + whole scan (the pre-P1c path).
        if let Ok(map) = self.corpora.lock() {
            if let Some(hit) = map.get(&(req.name.to_string(), req.version)) {
                return self.topk(hit.clone(), scan.as_topk_query());
            }
        }
        let n = req.ids.len();
        if n > self.prefetch_rows(req.dim, n) {
            return self.build_and_scan(req.name, req.version, req.data, req.dim, &self.corpora, &scan);
        }
        let corpus = self.corpus(req)?;
        self.topk(corpus, scan.as_topk_query())
    }

    fn ivf_probe(&self, req: &IvfRequest<'_>) -> Result<Vec<RawHit>, AccelError> {
        if req.dim == 0 || req.data.len() != req.ids.len() * req.dim {
            return Err(AccelError::Invalid("ivf corpus shape mismatch".into()));
        }
        if req.ids.len() > i32::MAX as usize {
            return Err(AccelError::Invalid("ivf corpus shape overflow".into()));
        }
        let scan = BlockScan {
            ids: req.ids,
            query: req.query,
            metric: req.metric,
            normalized: req.normalized,
            k: req.k,
            flat_mask: None,
            positions: Some(req.positions),
        };
        if let Ok(map) = self.ivf_corpora.lock() {
            if let Some(hit) = map.get(&(req.name.to_string(), req.version)) {
                return self.topk(hit.clone(), scan.as_topk_query());
            }
        }
        let n = req.ids.len();
        if n > self.prefetch_rows(req.dim, n) {
            return self.build_and_scan(
                req.name,
                req.version,
                req.data,
                req.dim,
                &self.ivf_corpora,
                &scan,
            );
        }
        let corpus = self.ivf_corpus(req)?;
        self.topk(corpus, scan.as_topk_query())
    }

    fn mem_info(&self) -> Option<(u64, u64)> {
        self.ctx
            .mem_get_info()
            .ok()
            .map(|(free, total)| (free as u64, total as u64))
    }

    fn asof_join(&self, req: &AsofRequest<'_>) -> Result<Vec<Option<u32>>, AccelError> {
        let left_len = req.left.len();
        if left_len == 0 {
            return Ok(Vec::new());
        }
        if req.right_times.is_empty() || req.right_times.len() > i32::MAX as usize {
            return Ok(vec![None; left_len]);
        }

        let series = self.device_series(req)?;
        let stream = self.ctx.default_stream();

        // The chunked merge kernel advances a monotonic right cursor, so its
        // left input must be ascending. `AsofRequest` does not promise that
        // (the host sweep sorts an index permutation instead). Do the same here:
        // sort a host-side index, upload the ordered timestamps, then scatter
        // the result back so output order still matches the input order.
        let order: Option<Vec<u32>> = if non_decreasing(req.left) {
            None
        } else {
            let mut o: Vec<u32> = (0..left_len as u32).collect();
            o.sort_unstable_by_key(|&i| req.left[i as usize]);
            Some(o)
        };
        let left_host: Cow<'_, [i64]> = match &order {
            Some(o) => Cow::Owned(o.iter().map(|&i| req.left[i as usize]).collect()),
            None => Cow::Borrowed(req.left),
        };
        let d_left = stream
            .clone_htod(left_host.as_ref())
            .map_err(|e| launch_err(format!("upload left: {e}")))?;
        let mut d_out = stream
            .alloc_zeros::<i32>(left_len)
            .map_err(|e| launch_err(format!("alloc asof out: {e}")))?;

        let chunk = left_len.div_ceil(ASOF_TARGET_THREADS).clamp(1, 1024);
        let threads = left_len.div_ceil(chunk);
        let blocks = (threads as u32).div_ceil(THREADS);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let left_i = left_len as i32;
        let right_i = series.len;
        let tol = req.tolerance_ns.unwrap_or(i64::MAX);
        let chunk_i = chunk as i32;
        let mut lb = stream.launch_builder(&self.asof_fn);
        lb.arg(&d_left)
            .arg(&series.ts)
            .arg(&mut d_out)
            .arg(&left_i)
            .arg(&right_i)
            .arg(&tol)
            .arg(&chunk_i);
        unsafe { lb.launch(cfg) }.map_err(|e| launch_err(format!("asof launch: {e}")))?;

        let raw: Vec<i32> = stream
            .clone_dtoh(&d_out)
            .map_err(|e| launch_err(format!("download asof: {e}")))?;
        let ordered: Vec<i32> = match order {
            Some(o) => {
                let mut back = vec![-1i32; left_len];
                for (pos, &orig) in o.iter().enumerate() {
                    back[orig as usize] = raw[pos];
                }
                back
            }
            None => raw,
        };
        Ok(ordered
            .into_iter()
            .map(|i| if i < 0 { None } else { Some(i as u32) })
            .collect())
    }

    /// P2 heat pump: build (or reuse) the device copy ahead of the query that
    /// asked for it.
    ///
    /// This reuses [`Self::build_corpus`] verbatim, so the pump pays exactly
    /// what a cold query would have paid — it just pays earlier, while no query
    /// is waiting. A cache hit costs one mutex read.
    ///
    /// `RightSeries` / `BucketIndex` have no device-copy cache behind this seam
    /// (the as-of path builds its series inside the call), so they report
    /// "nothing held" instead of pretending.
    fn prefetch_corpus(&self, req: &PrefetchRequest<'_>) -> Result<bool, AccelError> {
        if req.dim == 0 || req.data.len() != req.ids.len() * req.dim {
            return Err(AccelError::Invalid("prefetch corpus shape mismatch".into()));
        }
        if req.ids.len() > i32::MAX as usize {
            return Err(AccelError::Invalid("prefetch corpus shape overflow".into()));
        }
        let cache = match req.kind {
            ResidentKind::Corpus => &self.corpora,
            ResidentKind::IvfCorpus => &self.ivf_corpora,
            ResidentKind::RightSeries | ResidentKind::BucketIndex => return Ok(false),
        };
        let corpus = self.build_corpus(req.name, req.version, req.data, req.dim, cache)?;
        Ok(corpus.n as usize == req.ids.len())
    }

    fn direct_upload_stats(&self) -> Option<(u64, u64)> {
        Some(self.gpudirect_stats())
    }

    fn release(&self, kind: ResidentKind, name: &str) {
        match kind {
            ResidentKind::Corpus => {
                if let Ok(mut map) = self.corpora.lock() {
                    map.retain(|(n, _), _| n != name);
                }
            }
            ResidentKind::IvfCorpus => {
                if let Ok(mut map) = self.ivf_corpora.lock() {
                    map.retain(|(n, _), _| n != name);
                }
            }
            ResidentKind::RightSeries | ResidentKind::BucketIndex => {
                if let Ok(mut map) = self.series.lock() {
                    map.retain(|(n, _), _| n != name);
                }
            }
        }
    }
}

/// Total VRAM of device `ordinal`, if one is present.
///
/// Used to turn `CUDA_VRAM`'s default (`DEVICE:0.8`) into a byte count during
/// parameter resolution.
pub fn device_total_bytes(ordinal: usize) -> Option<u64> {
    CudaContext::new(ordinal)
        .ok()
        .and_then(|ctx| ctx.total_mem().ok())
        .map(|b| b as u64)
}

/// `(free, total)` device memory in bytes, for leak checks and preflight.
pub fn mem_get_info(ordinal: usize) -> Option<(u64, u64)> {
    CudaContext::new(ordinal)
        .ok()
        .and_then(|ctx| ctx.mem_get_info().ok())
        .map(|(free, total)| (free as u64, total as u64))
}

/// Number of CUDA devices visible to the driver.
pub fn device_count() -> usize {
    CudaContext::device_count().unwrap_or(0).max(0) as usize
}

/// A host-memory span pinned with [`cudaHostRegister`] (P1b-2 / V-3).
///
/// This is the one place the `unsafe` FFI for `cuMemHostRegister_v2` /
/// `cuMemHostUnregister` lives. Unlike [`PinnedHostSlice`] (which *allocates*
/// locked pages with `cudaHostAlloc`), registering pins pages the OS already
/// mapped — exactly what a `Col::Mapped` backed by `memmap2` hands us. The
/// pointer is unchanged, so the mapping stays a zero-copy source: DMA reads the
/// file's pages directly, no staging copy.
///
/// # Safety contract
///
/// 1. The caller guarantees the `[ptr, ptr + len)` span is a valid, immutable,
///    mapped host range that stays alive (and un-`munmap`-ed) until this value
///    is dropped — a `Col::Mapped`'s `Arc<dyn MappedRegion>` provides exactly
///    that (see `gtv_core::column::MappedRegion`).
/// 2. `unregister` runs on drop and is safe to call only after the driver has
///    stopped reading the pages; the upload path synchronizes the stream first,
///    so by the time this guard drops no DMA is in flight.
struct HostRegister {
    ptr: *mut std::ffi::c_void,
}

impl HostRegister {
    /// Pin `[ptr, len)` so the driver may DMA from it without a bounce buffer.
    ///
    /// # Safety
    /// See the type-level contract: the span must be a live, mapped, immutable
    /// host range for the lifetime of the returned guard.
    unsafe fn register(ptr: *mut u8, len: usize) -> Result<Self, AccelError> {
        let flags = cudarc::driver::sys::CU_MEMHOSTREGISTER_PORTABLE;
        // SAFETY: `ptr`/`len` describe a live mapped range (caller's contract);
        // `register` itself only pins pages and does not dereference them.
        unsafe {
            cudarc::driver::sys::cuMemHostRegister_v2(ptr as *mut std::ffi::c_void, len, flags)
        }
        .result()
        .map_err(|e| launch_err(format!("cuMemHostRegister_v2: {e}")))?;
        Ok(Self {
            ptr: ptr as *mut std::ffi::c_void,
        })
    }
}

impl Drop for HostRegister {
    fn drop(&mut self) {
        // SAFETY: the span was registered by `register` and has not been
        // unregistered; the upload path synchronizes before this guard drops.
        let _ = unsafe { cudarc::driver::sys::cuMemHostUnregister(self.ptr) };
    }
}

/// Register a mapped host span, DMA it to the device, and unregister — the
/// zero-copy upload the P1b-1 staging path replaces with a `copy_from_slice`.
///
/// This is the **fast path** when the source is already an mmap view (a
/// `Col::Mapped` from a GTVVEC segment): `cuMemHostRegister` only rewrites page
/// tables, so the subsequent `clone_htod` reads the file's pages directly with
/// no bounce buffer *and* no host memcpy. The caller keeps the mapping alive
/// across the call (the `&[T]` borrow enforces that).
///
/// On any register failure the caller falls back to the P1b-1 staging path; the
/// query must not error just because the driver refused to pin the pages.
pub fn register_and_upload<T: cudarc::driver::DeviceRepr>(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    src: &[T],
) -> Result<(), AccelError> {
    if src.is_empty() {
        return Ok(());
    }
    let ptr = src.as_ptr() as *mut u8;
    let len = std::mem::size_of_val(src);
    // SAFETY: `src` is a live `&[T]` for the duration of this call; the mapping
    // (if any) is held by the caller and cannot be torn down while we borrow it.
    let _reg = unsafe { HostRegister::register(ptr, len) }?;
    // The pages are now pinned, so `clone_htod`'s internal `memcpy_htod_async`
    // reads them directly (no driver bounce buffer). `_reg` keeps them pinned
    // until after the synchronize flushes the DMA.
    let dst = stream
        .clone_htod(src)
        .map_err(|e| launch_err(format!("register upload DMA: {e}")))?;
    stream
        .synchronize()
        .map_err(|e| launch_err(format!("register upload sync: {e}")))?;
    drop(dst);
    Ok(())
}
