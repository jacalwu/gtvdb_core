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
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use gtv_core::Metric;

use crate::backend::{
    AccelBackend, AccelError, AccelOp, AsofRequest, DeviceInfo, DeviceKind, IvfRequest, KnnRequest,
    RawHit,
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

/// A CUDA device with both kernels compiled through NVRTC.
pub struct CudaBackend {
    device: DeviceInfo,
    ctx: Arc<CudaContext>,
    /// NVRTC architecture option computed from the device we opened.
    arch: String,
    transpose_fn: CudaFunction,
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

        let asof_ptx = compile_ptx_with_opts(ASOF_SRC, opts())
            .map_err(|e| launch_err(format!("nvrtc asof: {e}")))?;
        let asof_module = ctx
            .load_module(asof_ptx)
            .map_err(|e| launch_err(format!("load asof module: {e}")))?;
        let asof_fn = asof_module
            .load_function("asof_merge_chunked_kernel")
            .map_err(|e| launch_err(format!("load asof kernel: {e}")))?;

        Ok(Self {
            device: DeviceInfo {
                name,
                kind: DeviceKind::Cuda,
                total_bytes: total,
            },
            ctx,
            arch,
            transpose_fn,
            knn_functions: Mutex::new(HashMap::new()),
            asof_fn,
            corpora: Mutex::new(HashMap::new()),
            ivf_corpora: Mutex::new(HashMap::new()),
            series: Mutex::new(HashMap::new()),
        })
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

        // One bulk upload of the row-major corpus, then a device-side transpose.
        // The row-major buffer is dropped as soon as the transpose is enqueued
        // and the stream has consumed it (cudarc keeps the allocation alive
        // until the stream is synchronized by the next D2H copy).
        let d_rowmajor = stream
            .clone_htod(data)
            .map_err(|e| launch_err(format!("upload corpus: {e}")))?;
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
        // Ensure the transpose finished before the row-major allocation can be
        // reused by the driver; then release it.
        self.ctx
            .synchronize()
            .map_err(|e| launch_err(format!("sync after transpose: {e}")))?;
        drop(d_rowmajor);

        let corpus = Arc::new(DeviceCorpus {
            data_t,
            n: n as i32,
            dim: dim as i32,
        });
        if let Ok(mut map) = cache.lock() {
            // Keep one version per name: an older version is dead weight.
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
        let ts = stream
            .clone_htod(req.right_times)
            .map_err(|e| launch_err(format!("upload series: {e}")))?;
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
        let mut merged: Vec<RawHit> = dists
            .iter()
            .zip(out_ids.iter())
            .filter(|(_, &pos)| pos >= 0)
            .filter_map(|(&distance, &pos)| {
                ids.get(pos as usize)
                    .map(|&id| RawHit { distance, id })
            })
            .collect();
        merged.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        merged.truncate(k);
        Ok(merged)
    }
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
        let corpus = self.corpus(req)?;
        self.topk(
            corpus,
            TopkQuery {
                ids: req.ids,
                query: req.query,
                metric: req.metric,
                normalized: req.normalized,
                k: req.k,
                mask: req.mask,
                positions: None,
            },
        )
    }

    fn ivf_probe(&self, req: &IvfRequest<'_>) -> Result<Vec<RawHit>, AccelError> {
        if req.dim == 0 || req.data.len() != req.ids.len() * req.dim {
            return Err(AccelError::Invalid("ivf corpus shape mismatch".into()));
        }
        if req.ids.len() > i32::MAX as usize {
            return Err(AccelError::Invalid("ivf corpus shape overflow".into()));
        }
        let corpus = self.ivf_corpus(req)?;
        self.topk(
            corpus,
            TopkQuery {
                ids: req.ids,
                query: req.query,
                metric: req.metric,
                normalized: req.normalized,
                k: req.k,
                mask: None,
                positions: Some(req.positions),
            },
        )
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
