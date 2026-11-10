//! The rolling-window kernel: where its PTX comes from and how it is launched.
//!
//! The PTX is a committed image produced by `cargo xtask cuda-ptx` from
//! `kernels/window_scan.cu`, for the same reasons as the KNN image (see
//! [`crate::knn`]): a `build.rs` + `nvcc` would stop the GPU-less CI from
//! building, and NVRTC needs `libnvrtc` at run time, which a driver-only
//! deployment does not have.
//!
//! # Launch shape
//!
//! `grid = ceil(rows / CHUNK)`, one block per chunk of output rows, and each
//! block also loads the `window-1` rows before `chunk * CHUNK` because the first
//! windows of a chunk reach backwards. That halo is what lets a single kernel
//! suffice: with it no block ever needs another block's result, so there is no
//! prefix array, no carry chain and no lookback.
//!
//! The window is therefore bounded by the halo and the chunk: `window <=
//! MAX_WINDOW`. The bound is advertised in `Caps::max_window` so a planner can
//! send a 100k-row window to the CPU before paying for a launch, and
//! `window_reduce` returns `LimitExceeded` (`Fallback::Cpu`) for callers that do
//! not consult caps.

use std::ffi::c_void;
use std::sync::OnceLock;

use gtv_accelerator::WindowOp;

use crate::driver::{Function, Module};

/// The committed PTX image.
const PTX: &str = include_str!("../kernels/window_scan.ptx");

/// Threads per block. Must match `GTV_BLOCK` in the kernel.
pub(crate) const BLOCK: usize = 256;

/// Output rows per block. Must match `GTV_CHUNK`.
///
/// 2048 doubles of chunk plus up to 1023 of halo is 24.6 KB of shared memory,
/// and the scan path adds 256 slice totals (2 KB). The monotonic-deque path
/// needs the loaded rows as `int` on top of that, so it asks for 36.6 KB — both
/// inside the 48 KB a block may reserve without opting into the larger
/// carve-out, which is deliberate: asking for more would need
/// `cuFuncSetAttribute` and a second failure mode on older cards.
pub(crate) const CHUNK: usize = 2048;

/// Largest window the kernel serves. Must match `GTV_MAX_WINDOW`.
pub(crate) const MAX_WINDOW: usize = 1024;

struct Kernel {
    /// Leaked on purpose: the module must outlive every `Function` borrowed from
    /// it, and it is loaded once per process.
    _module: &'static Module,
    function: Function<'static>,
}

static KERNEL: OnceLock<Result<Kernel, String>> = OnceLock::new();

fn kernel() -> Result<&'static Kernel, String> {
    KERNEL
        .get_or_init(|| {
            let driver = crate::driver::driver()?;
            let module: &'static Module = Box::leak(Box::new(driver.load_ptx(PTX)?));
            let function = module.function("gtv_window_scan")?;
            Ok(Kernel {
                _module: module,
                function,
            })
        })
        .as_ref()
        .map_err(|e| e.clone())
}

/// Whether this kernel has been loaded. The accelerator asks before releasing
/// its retain of the primary context: the module lives in that context, so
/// dropping the last reference would leave the cached handle dangling.
pub(crate) fn loaded() -> bool {
    matches!(KERNEL.get(), Some(Ok(_)))
}

/// `gtv_accelerator::WindowOp` as the kernel's `GTV_*` code.
pub(crate) fn op_code(op: WindowOp) -> i32 {
    match op {
        WindowOp::Sum => 0,
        WindowOp::Mean => 1,
        WindowOp::Min => 2,
        WindowOp::Max => 3,
        WindowOp::Count => 4,
    }
}

/// Everything the kernel needs, in the order it declares its parameters.
pub(crate) struct Launch {
    pub values: u64,
    pub rows: i64,
    pub window: i32,
    pub op: i32,
    pub out: u64,
}

/// Blocks in the grid: one per chunk of output rows.
pub(crate) fn blocks(rows: usize) -> usize {
    rows.div_ceil(CHUNK).max(1)
}

/// Shared memory this launch needs, in bytes.
///
/// The kernel places its scan-total array after the loaded rows and its deque
/// after that, so the reservation has to cover the worst block: a full chunk
/// plus a full halo. Block 0 loads a shorter halo — there is no history before
/// row 0 — and reserving for that would be wrong in the other direction.
pub(crate) fn shared_bytes(op: WindowOp, window: usize, rows: usize) -> usize {
    let len = rows.min(CHUNK);
    let pre = window.saturating_sub(1);
    let loaded = pre + len;
    match op {
        // Only index arithmetic; the kernel touches no shared memory.
        WindowOp::Count => 0,
        // Loaded rows, scanned in place, plus one total per thread.
        WindowOp::Sum | WindowOp::Mean => loaded * std::mem::size_of::<f64>() + BLOCK * 8,
        // Loaded rows plus the monotonic deque of positions into them.
        WindowOp::Min | WindowOp::Max => loaded * (std::mem::size_of::<f64>() + 4),
    }
}

/// Launch on the legacy default stream. The caller synchronises.
pub(crate) fn launch(l: &Launch, shared: usize) -> Result<(), String> {
    let k = kernel()?;

    // The arguments `cuLaunchKernel` dereferences. Each is a stack slot in this
    // frame and the kernel only reads the scalars, so nothing here can be
    // invalidated before the launch is submitted.
    let mut values = l.values;
    let mut rows = l.rows;
    let mut window = l.window;
    let mut op = l.op;
    let mut out = l.out;
    let mut argv: [*mut c_void; 5] = [
        &mut values as *mut u64 as *mut c_void,
        &mut rows as *mut i64 as *mut c_void,
        &mut window as *mut i32 as *mut c_void,
        &mut op as *mut i32 as *mut c_void,
        &mut out as *mut u64 as *mut c_void,
    ];

    // SAFETY: `argv` mirrors the kernel's parameter list one for one
    // (`kernels/window_scan.cu`), and every block indexes its own chunk plus the
    // halo before it, both inside the `rows` rows of `values` and `out` that the
    // caller allocated.
    unsafe {
        k.function.launch(
            (blocks(l.rows.max(0) as usize) as u32, 1, 1),
            (BLOCK as u32, 1, 1),
            shared,
            &mut argv,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_op_codes_match_the_kernel() {
        // The kernel switches on these numbers; getting one wrong would silently
        // compute a different aggregate rather than fail.
        assert_eq!(op_code(WindowOp::Sum), 0);
        assert_eq!(op_code(WindowOp::Mean), 1);
        assert_eq!(op_code(WindowOp::Min), 2);
        assert_eq!(op_code(WindowOp::Max), 3);
        assert_eq!(op_code(WindowOp::Count), 4);
    }

    #[test]
    fn the_embedded_ptx_is_a_ptx_image_of_the_expected_shape() {
        assert!(PTX.is_ascii(), "non-ASCII byte in the PTX image");
        assert!(PTX.contains(".version"), "not PTX");
        assert!(PTX.contains(".entry gtv_window_scan"), "entry point renamed");
        assert!(PTX.len() > 2048, "suspiciously small image");
    }

    #[test]
    fn shared_memory_fits_the_default_budget_at_the_largest_window() {
        for (op, expected) in [
            // chunk + halo rows, scanned in place, plus one total per thread.
            (WindowOp::Sum, (CHUNK + MAX_WINDOW - 1) * 8 + BLOCK * 8),
            (WindowOp::Mean, (CHUNK + MAX_WINDOW - 1) * 8 + BLOCK * 8),
            // Plus the deque, one `int` per loaded row.
            (WindowOp::Min, (CHUNK + MAX_WINDOW - 1) * 12),
            (WindowOp::Max, (CHUNK + MAX_WINDOW - 1) * 12),
        ] {
            let bytes = shared_bytes(op, MAX_WINDOW, 10_000_000);
            assert_eq!(bytes, expected);
            // 48 KB is what a block gets without opting into the larger
            // carve-out; crossing it turns every launch into an error.
            assert!(bytes <= 48 * 1024, "{op:?} wants {bytes} B");
        }
        assert_eq!(shared_bytes(WindowOp::Count, MAX_WINDOW, 1_000), 0);
        // A short column never reserves a full chunk.
        assert_eq!(
            shared_bytes(WindowOp::Sum, 4, 10),
            (3 + 10) * 8 + BLOCK * 8
        );
    }

    #[test]
    fn the_grid_covers_every_row_exactly_once() {
        assert_eq!(blocks(0), 1);
        assert_eq!(blocks(1), 1);
        assert_eq!(blocks(CHUNK), 1);
        assert_eq!(blocks(CHUNK + 1), 2);
        assert_eq!(blocks(10_000_000), 10_000_000usize.div_ceil(CHUNK));
    }
}
