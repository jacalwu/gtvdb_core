//! The flat K-NN kernel: where its PTX comes from, how it is launched, and how
//! the launch is shaped.
//!
//! # The PTX is checked in, the toolkit is not
//!
//! [`PTX`] is a committed file produced by `cargo xtask cuda-ptx` from
//! `kernels/knn_flat.cu`. That is a deliberate trade against compiling at run
//! time (NVRTC) or at build time (`build.rs` + `nvcc`):
//!
//! * **against `build.rs`** — the crate would stop building on the GPU-less CI
//!   machine, and the rule in `docs/gtvdb-cpu-cuda-cursor-development-spec.md`
//!   §7.1 is that `cargo test` here never needs a toolkit;
//! * **against NVRTC** — it needs `libnvrtc` at *run* time, which a machine with
//!   the driver installed and no toolkit does not have. A container with the
//!   driver alone is the normal deployment; embedding PTX keeps C2 working there
//!   because the driver itself JITs PTX.
//!
//! The cost of a checked-in artifact is that it can drift from its source, so
//! each image carries the BLAKE3 of the `.cu` it was built from and
//! `tests/ptx_stamp.rs` verifies that hash on every `cargo test` — including on
//! machines that cannot run the kernel.
//!
//! One `compute_80` image serves Ampere and later: PTX is forward compatible, so
//! Ada and Blackwell JIT the same file. What it is *not* compatible with is an
//! older driver than the toolkit that produced it (`cuModuleLoadData` fails with
//! `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`); rebuild with that machine's toolkit.
//!
//! # Launch shape
//!
//! `grid = (queries, splits)`. The split dimension exists because a single block
//! scanning a million rows uses one SM: the host sizes `splits` from the SM count
//! and the batch, so a batch of one still fills the card, and merges the partial
//! lists afterwards. Measured shapes are in `docs/cuda-acceleration-plan.md` §9.7.

use std::ffi::c_void;
use std::sync::OnceLock;

use crate::driver::{Function, Module};

/// The committed PTX image. See the module docs for why it is a file.
const PTX: &str = include_str!("../kernels/knn_flat.ptx");

/// Threads per block. Must match `GTV_BLOCK` in the kernel.
pub(crate) const BLOCK: usize = 256;

/// Warps per block, each with its own candidate heap. Must match `GTV_WARPS`.
pub(crate) const WARPS: usize = 8;

/// Largest `k` the kernel serves.
///
/// The heaps live in shared memory: `WARPS * k * 12 B`, so 24 KB at this limit
/// out of the 48 KB a block may reserve without opting in. The CPU reference has
/// no such bound, which is why the limit is advertised in `Caps::max_k` and
/// checked at plan time rather than discovered here.
pub(crate) const MAX_K: usize = 256;

/// Cap on the split dimension, so a batch of one cannot turn a 100-neighbour
/// query into megabytes of partial results.
const MAX_SPLITS: usize = 64;

/// Blocks per SM the split heuristic aims for.
///
/// Four is a measurement, not a preference: at 1e5 x 128 one query took 0.50 ms
/// with 8 splits and 0.30 ms with 24 on a 24-SM card (`bench_knn`), because the
/// scan is latency-bound until there is enough parallelism to saturate the
/// memory system. Beyond ~4 blocks per SM the extra partial lists cost more than
/// they buy.
const TARGET_BLOCKS_PER_SM: usize = 4;

struct Kernel {
    /// Leaked on purpose: the module must outlive every `Function` borrowed from
    /// it, and it is loaded once per process.
    _module: &'static Module,
    function: Function<'static>,
}

/// The loaded kernel, once. See [`loaded`] for why the context has to outlive it.
static KERNEL: OnceLock<Result<Kernel, String>> = OnceLock::new();

/// Compile-once handle to the kernel. Requires a current context on the calling
/// thread (module loading is context-scoped), like every other driver call here.
fn kernel() -> Result<&'static Kernel, String> {
    KERNEL
        .get_or_init(|| {
            let driver = crate::driver::driver()?;
            let module: &'static Module = Box::leak(Box::new(driver.load_ptx(PTX)?));
            let function = module.function("gtv_knn_flat")?;
            Ok(Kernel {
                _module: module,
                function,
            })
        })
        .as_ref()
        .map_err(|e| e.clone())
}

/// Whether a kernel module has been loaded into this process.
///
/// [`CudaAccelerator`](crate::CudaAccelerator) asks before releasing its retain
/// of the primary context: the module lives *in* that context, so tearing the
/// context down would leave the cached handle dangling and the next launch
/// failing with `CUDA_ERROR_INVALID_HANDLE`. Once a kernel is loaded the context
/// is process-wide, exactly like the module.
pub(crate) fn loaded() -> bool {
    matches!(KERNEL.get(), Some(Ok(_)))
}

/// Everything the kernel needs, in the order it declares its parameters.
pub(crate) struct Launch {
    /// Queries in the batch: the grid's `x`, because `blockIdx.x` selects the
    /// query and `blockIdx.y` the split in the kernel.
    pub batch: usize,
    pub splits: usize,
    pub corpus: u64,
    pub ids: u64,
    pub rows: i64,
    pub queries: u64,
    pub dim: i32,
    pub k: i32,
    pub metric: i32,
    /// Candidate bitmap words, or `0` for "every row".
    pub filter: u64,
    pub out_dist: u64,
    pub out_id: u64,
    pub out_count: u64,
}

impl Launch {
    /// `WARPS * k` entries of `(f32, u64)` per block, plus nothing else: the
    /// static `__shared__ int s_count[WARPS]` is counted by the compiler.
    pub(crate) fn shared_bytes(&self) -> usize {
        WARPS * self.k as usize * (std::mem::size_of::<f32>() + std::mem::size_of::<u64>())
    }
}

/// Launch, on the legacy default stream. The caller synchronises.
pub(crate) fn launch(l: &Launch) -> Result<(), String> {
    let k = kernel()?;

    // Pointers at the values, because that is what `cuLaunchKernel` takes. The
    // `unsafe` is the contract in `Function::launch`: each pointer must name a
    // live value of the declared type. Every one of them is a stack slot in this
    // frame or a device address, and the kernel only *reads* the scalars, so the
    // array cannot be invalidated before the launch is submitted.
    let mut corpus = l.corpus;
    let mut ids = l.ids;
    let mut rows = l.rows;
    let mut queries = l.queries;
    let mut dim = l.dim;
    let mut kk = l.k;
    let mut metric = l.metric;
    let mut filter = l.filter;
    let mut out_dist = l.out_dist;
    let mut out_id = l.out_id;
    let mut out_count = l.out_count;
    let mut argv: [*mut c_void; 11] = [
        &mut corpus as *mut u64 as *mut c_void,
        &mut ids as *mut u64 as *mut c_void,
        &mut rows as *mut i64 as *mut c_void,
        &mut queries as *mut u64 as *mut c_void,
        &mut dim as *mut i32 as *mut c_void,
        &mut kk as *mut i32 as *mut c_void,
        &mut metric as *mut i32 as *mut c_void,
        &mut filter as *mut u64 as *mut c_void,
        &mut out_dist as *mut u64 as *mut c_void,
        &mut out_id as *mut u64 as *mut c_void,
        &mut out_count as *mut u64 as *mut c_void,
    ];

    // SAFETY: `argv` mirrors the kernel's parameter list one for one
    // (`kernels/knn_flat.cu`), each entry points at a live local, and the grid is
    // sized by the caller from the buffers it allocated: every block indexes
    // `rows` rows of `corpus`, `rows` ids, its own query row, and its own slot in
    // the three output arrays.
    unsafe {
        k.function.launch(
            (l.batch.max(1) as u32, l.splits.max(1) as u32, 1),
            (BLOCK as u32, 1, 1),
            l.shared_bytes(),
            &mut argv,
        )
    }
}

/// Blocks per query in the row dimension.
///
/// This is `gridDim.y`: the kernel's `x` is the query (one block per query) and
/// `y` is the slice of rows that block scans, so a batch of one still gets the
/// whole card.
pub(crate) fn splits_for(batch: usize, rows: usize, multiprocessors: i32) -> usize {
    let target = (TARGET_BLOCKS_PER_SM * multiprocessors.max(1) as usize).max(1);
    let by_occupancy = target.div_ceil(batch.max(1));
    // A block that gets less than one full tile of rows is pure overhead.
    let by_work = (rows / BLOCK).max(1);
    by_occupancy.clamp(1, MAX_SPLITS).min(by_work).max(1)
}

/// `gtv_accelerator::Metric` as the kernel's `GTV_*` code.
pub(crate) fn metric_code(m: gtv_core::Metric) -> i32 {
    match m {
        gtv_core::Metric::L2 => 0,
        gtv_core::Metric::Cosine => 1,
        gtv_core::Metric::Ip => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_fill_the_card_without_fragmenting_a_small_scan() {
        // One query on a 24-SM card: four blocks per SM, capped so the partial
        // results stay small.
        assert_eq!(splits_for(1, 1_000_000, 24), MAX_SPLITS);
        // A batch that already fills the card needs no split at all.
        assert_eq!(splits_for(1024, 1_000_000, 24), 1);
        // 8 queries x 12 splits = 96 blocks = 4 per SM.
        assert_eq!(splits_for(8, 1_000_000, 24), 12);
        // Few rows: a block must still get a full tile, so fewer splits.
        assert_eq!(splits_for(1, 256, 24), 1);
        assert_eq!(splits_for(1, 1024, 24), 4);
        // Degenerate inputs never produce a zero grid dimension.
        assert_eq!(splits_for(0, 0, 0), 1);
    }

    #[test]
    fn the_metric_codes_match_the_kernel() {
        // The kernel switches on these numbers (`GTV_L2 0`, `GTV_COSINE 1`,
        // `GTV_IP 2`); getting one wrong would silently change what a distance
        // means rather than fail.
        assert_eq!(metric_code(gtv_core::Metric::L2), 0);
        assert_eq!(metric_code(gtv_core::Metric::Cosine), 1);
        assert_eq!(metric_code(gtv_core::Metric::Ip), 2);
    }

    #[test]
    fn the_embedded_ptx_is_a_ptx_image_of_the_expected_shape() {
        // ASCII is a requirement, not a preference: the driver's PTX parser
        // rejects a non-ASCII byte anywhere, comments included, and reports only
        // `CUDA_ERROR_INVALID_PTX` when it does.
        assert!(PTX.is_ascii(), "non-ASCII byte in the PTX image");
        assert!(PTX.contains(".version"), "not PTX: {}", &PTX[..64.min(PTX.len())]);
        assert!(PTX.contains(".entry gtv_knn_flat"), "entry point renamed");
        assert!(PTX.len() > 4096, "suspiciously small image");
    }

    #[test]
    fn shared_memory_scales_with_k_and_fits_the_default_budget() {
        let l = Launch {
            batch: 0,
            splits: 1,
            corpus: 0,
            ids: 0,
            rows: 0,
            queries: 0,
            dim: 4,
            k: MAX_K as i32,
            metric: 0,
            filter: 0,
            out_dist: 0,
            out_id: 0,
            out_count: 0,
        };
        // 8 warps x 256 x 12 B = 24 KB, inside the 48 KB a block gets without
        // opting into the larger carve-out.
        assert_eq!(l.shared_bytes(), 24_576);
        assert!(l.shared_bytes() <= 48 * 1024);
    }
}
