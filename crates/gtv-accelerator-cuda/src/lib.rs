//! The CUDA backend — residency only.
//!
//! # What this crate does today (C1)
//!
//! It enumerates the device, reports its limits, and makes a snapshot *actually*
//! resident: `cuMemAlloc` + `cuMemcpyHtoD` per column, `cuMemFree` on release.
//! It implements **no operator**. Every kernel call reports
//! [`gtv_accelerator::AcceleratorError::Unsupported`] and the policy sends the
//! work to the CPU.
//!
//! That is not an unfinished C1; it is C1. Measured on this machine: pageable
//! H2D runs at 7.5–8.0 GB/s and is 97% of the load, so one upload of a 1e7-edge
//! snapshot costs 30–41 ms against a 6.9 ms CPU scan of the same graph. A kernel
//! measured before residency exists measures PCIe, not arithmetic.
//!
//! The upload, though, is not the expensive part of going to the device. The CSR
//! it consumes costs ~1.6 s to materialise at 1e7 edges (`SegmentStore::graph_at`
//! walks the segments into `Vec<LiveRow>`, then into `Vec<EdgeRecord>`, then
//! folds that into columns), so the end-to-end cold path is 1.6 s against the
//! CPU's 6.86 ms — and 98% of it happens before this crate is called. See §9.3
//! of `docs/cuda-acceleration-plan.md`. That is why the order is: measure, make
//! materialisation cheap, and only then write kernels against a copy that is
//! already resident — residency amortises the upload over 5 queries, but nothing
//! amortises a rebuild that happens on every query.
//!
//! # The load receipt
//!
//! Every upload records [`gtv_accelerator::GraphLoadMetrics`]: per column bytes,
//! allocation and copy time, plus the stages that are supposed to cost nothing
//! (host pack, layout transform, staging). The zero-cost stages are fields rather
//! than absences so that a regression is a number rather than a mystery, and
//! `tests/device.rs` asserts them. A loader that reports only "upload took 31 ms"
//! cannot tell a PCIe bottleneck from a re-pack, and will optimise the wrong one.
//!
//! # Why `Caps` advertises nothing
//!
//! [`gtv_accelerator::ExecutionPolicy`] gates on `Caps::supports`. With no
//! operator advertised, a query plan built on this backend routes everything to
//! the CPU, and `AcceleratorRegistry::route` reports `Unsupported` rather than
//! `NoDevice` — the distinction between "no GPU here" and "the GPU has not
//! ported this yet". C2 flips one bit at a time as each kernel lands, and the
//! policy needs no change to follow.
//!
//! # Absent driver is a normal answer
//!
//! [`CudaAccelerator::open`] returns `DeviceUnavailable` when the library or the
//! device is missing. No panic, no abort, no link-time dependency: this crate
//! builds and tests on the GPU-less CI machine.

pub mod driver;

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use gtv_accelerator::{
    Accelerator, AcceleratorError, BackendKind, Caps, Column, GraphLoadMetrics, ResidentGraph,
    SnapshotView,
};

use driver::{Context, DeviceInfo, Driver};

/// Fraction of total device memory the residency budget defaults to.
///
/// 60% leaves room for the driver's own allocations, the workspace a kernel
/// needs, and the top-k buffer — all of which the C0 probe showed are not
/// negligible at 1e7 edges.
const DEFAULT_BUDGET_FRACTION: f64 = 0.60;

/// One resident snapshot, as device pointers.
struct DeviceGraph {
    snapshot_id: u64,
    columns: Vec<u64>,
    bytes: usize,
}

/// The CUDA backend.
pub struct CudaAccelerator {
    driver: &'static Driver,
    device: DeviceInfo,
    context: Context,
    budget: usize,
    graphs: Mutex<HashMap<u64, DeviceGraph>>,
    /// Receipt for the most recent upload, kept so a benchmark can read the
    /// stage breakdown after the fact instead of being handed a second return
    /// value it does not want.
    last_load: Mutex<Option<GraphLoadMetrics>>,
}

impl CudaAccelerator {
    /// Open device 0.
    ///
    /// The only function in this crate that can be called before a decision is
    /// made: everything below it needs a context.
    pub fn open() -> Result<Self, AcceleratorError> {
        let driver = driver::driver().map_err(|reason| AcceleratorError::DeviceUnavailable {
            backend: "cuda",
            reason,
        })?;
        let count = driver
            .device_count()
            .map_err(|reason| AcceleratorError::DeviceUnavailable {
                backend: "cuda",
                reason,
            })?;
        if count == 0 {
            return Err(AcceleratorError::DeviceUnavailable {
                backend: "cuda",
                reason: "no CUDA device".into(),
            });
        }
        let device = driver
            .device(0)
            .map_err(|reason| AcceleratorError::DeviceUnavailable {
                backend: "cuda",
                reason,
            })?;
        let context = driver
            .create_context(device.handle)
            .map_err(|reason| AcceleratorError::DeviceUnavailable {
                backend: "cuda",
                reason,
            })?;
        let budget = (device.total_mem_bytes as f64 * DEFAULT_BUDGET_FRACTION) as usize;
        Ok(Self {
            driver,
            device,
            context,
            budget,
            graphs: Mutex::new(HashMap::new()),
            last_load: Mutex::new(None),
        })
    }

    /// Device limits without creating a context. Safe to call from a planner
    /// that only wants to fill in `Caps`.
    pub fn probe() -> Result<DeviceInfo, AcceleratorError> {
        let driver = driver::driver().map_err(|reason| AcceleratorError::DeviceUnavailable {
            backend: "cuda",
            reason,
        })?;
        if driver.device_count().unwrap_or(0) == 0 {
            return Err(AcceleratorError::DeviceUnavailable {
                backend: "cuda",
                reason: "no CUDA device".into(),
            });
        }
        driver
            .device(0)
            .map_err(|reason| AcceleratorError::DeviceUnavailable {
                backend: "cuda",
                reason,
            })
    }

    pub fn device(&self) -> &DeviceInfo {
        &self.device
    }

    /// Bytes the residency budget allows.
    pub fn budget_bytes(&self) -> usize {
        self.budget
    }

    /// `(free, total)` device memory right now.
    pub fn mem_info(&self) -> Result<(usize, usize), AcceleratorError> {
        self.driver
            .mem_info()
            .map_err(|reason| self.backend_error("cuMemGetInfo", reason))
    }

    /// Bytes currently held in device memory, as this backend accounted them.
    ///
    /// Compared against `cuMemGetInfo` in the device test: if the two disagree
    /// the accounting is wrong, and a wrong accounting is how a budget silently
    /// stops holding.
    pub fn resident_bytes(&self) -> usize {
        self.graphs
            .lock()
            .map(|g| g.values().map(|e| e.bytes).sum())
            .unwrap_or(0)
    }

    /// Resident snapshots, for tests and for a pre-warm loop.
    pub fn resident_snapshots(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self
            .graphs
            .lock()
            .map(|g| g.values().map(|e| e.snapshot_id).collect())
            .unwrap_or_default();
        v.sort_unstable();
        v
    }

    /// The stage breakdown of the last upload on this backend.
    ///
    /// `None` before the first upload. Cleared on a failed upload so a caller
    /// cannot read the previous snapshot's numbers and believe them.
    pub fn last_load_metrics(&self) -> Option<GraphLoadMetrics> {
        self.last_load.lock().ok().and_then(|m| m.clone())
    }

    fn backend_error(&self, op: &'static str, reason: String) -> AcceleratorError {
        AcceleratorError::Backend {
            backend: "cuda",
            op,
            code: -1,
            message: reason,
        }
    }

    fn forget_metrics(&self) {
        if let Ok(mut slot) = self.last_load.lock() {
            *slot = None;
        }
    }

    fn free_all(&self, columns: &mut Vec<u64>) {
        for p in columns.drain(..) {
            self.driver.free(p);
        }
    }

    fn graphs(&self) -> Result<std::sync::MutexGuard<'_, HashMap<u64, DeviceGraph>>, AcceleratorError>
    {
        self.graphs.lock().map_err(|_| AcceleratorError::Backend {
            backend: "cuda",
            op: "lock",
            code: -1,
            message: "resident table poisoned".into(),
        })
    }
}

// SAFETY: the only raw pointer held is the `CUcontext`. It is never dereferenced
// and never used without `cuCtxSetCurrent` on the calling thread first, which is
// the documented way to make a context usable from a pool thread. The resident
// table — the only mutable state — is behind a `Mutex`, and the driver API is
// itself thread-safe.
unsafe impl Send for CudaAccelerator {}
unsafe impl Sync for CudaAccelerator {}

impl Accelerator for CudaAccelerator {
    fn name(&self) -> &'static str {
        "cuda-0"
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn caps(&self) -> Caps {
        // Residency yes, operators no — see the module docs. Adding
        // `.with_op(Op::KnnFlat)` here is C2's entire integration step.
        Caps::new(BackendKind::Cuda, "cuda-0", Some(self.budget)).with_resident(true)
    }

    fn load_graph(&self, view: SnapshotView<'_>) -> Result<Option<ResidentGraph>, AcceleratorError> {
        let started = Instant::now();
        view.validate()?;
        let bytes = view.bytes();
        let mut m = GraphLoadMetrics::new(
            view.snapshot_id,
            view.node_count,
            view.edge_count(),
            view.dst_layout(),
        );
        m.priced_bytes = bytes;
        // A failed upload must not leave the previous snapshot's receipt in
        // place: a caller reading `last_load_metrics` would attribute one
        // snapshot's cost to another.
        self.forget_metrics();

        let t = Instant::now();
        self.driver
            .bind(self.context)
            .map_err(|reason| self.backend_error("cuCtxSetCurrent", reason))?;
        m.bind_ns = nanos(t.elapsed());

        // `dst` goes up at whatever width the view carries. Nothing is converted
        // here: a narrowing pass costs 9.5–11 ms at 1e7 edges to save 5.7 ms of
        // transfer (§9.2), so the 4 B/edge only arrive when a producer already
        // has the narrow column. The receipt records which width it was, and a
        // kernel must read the receipt rather than assume.
        let dst_src: (*const u8, usize) = match view.dst32 {
            Some(d) => (d.as_ptr() as *const u8, std::mem::size_of_val(d)),
            None => (
                view.dst.as_ptr() as *const u8,
                std::mem::size_of_val(view.dst),
            ),
        };

        // One allocation per column rather than one arena: an eviction frees
        // exactly what was allocated, and a partial failure frees exactly what
        // it already got. Four of the five uploads are straight memcpys of the
        // CSR's own order, so nothing is repacked on the host — and `input_bytes`
        // below is what makes that claim checkable rather than a comment.
        let cols: [(Column, *const u8, usize); 5] = [
            (
                Column::Offsets,
                view.offsets.as_ptr() as *const u8,
                std::mem::size_of_val(view.offsets),
            ),
            (Column::Dst, dst_src.0, dst_src.1),
            (
                Column::ValidFrom,
                view.valid_from.as_ptr() as *const u8,
                std::mem::size_of_val(view.valid_from),
            ),
            (
                Column::ValidTo,
                view.valid_to.as_ptr() as *const u8,
                std::mem::size_of_val(view.valid_to),
            ),
            (
                Column::EdgeType,
                view.edge_type.as_ptr() as *const u8,
                std::mem::size_of_val(view.edge_type),
            ),
        ];
        m.input_bytes = cols.iter().map(|(_, _, len)| *len).sum();

        let mut columns: Vec<u64> = Vec::with_capacity(cols.len());
        let mut allocated = 0usize;
        for (col, src, len) in cols {
            let name = col.name();
            let t = Instant::now();
            let ptr = match self.driver.alloc(len) {
                Ok(p) => p,
                Err(reason) => {
                    self.free_all(&mut columns);
                    return Err(self.backend_error(
                        "cuMemAlloc",
                        format!("{name} ({len} bytes): {reason}"),
                    ));
                }
            };
            let alloc_ns = nanos(t.elapsed());
            // SAFETY: `ptr` came from `alloc(len)` in the line above and `src`
            // points at `len` readable bytes of the borrowed view — or of the
            // caller's `dst32` slice, which the view borrows for as long.
            let t = Instant::now();
            if let Err(reason) = unsafe { self.driver.copy_htod(ptr, src, len) } {
                self.driver.free(ptr);
                self.free_all(&mut columns);
                return Err(self.backend_error(
                    "cuMemcpyHtoD",
                    format!("{name} ({len} bytes): {reason}"),
                ));
            }
            columns.push(ptr);
            allocated += len;
            m.record_column(col, len, alloc_ns, nanos(t.elapsed()));
        }

        // The ratchet on the budget: whatever a later change does to a column
        // width, an upload that allocates something other than what `bytes()`
        // priced fails here instead of silently over-admitting snapshots.
        if allocated != bytes {
            self.free_all(&mut columns);
            return Err(AcceleratorError::InvalidArgument(format!(
                "upload allocated {allocated} bytes but the view priced {bytes}"
            )));
        }

        // `cuMemcpyHtoD` is documented as synchronous, so this should measure
        // nothing. It is here to prove that: if a copy ever returned before its
        // bytes landed, the remainder shows up as sync time and the per-column
        // table stops being a transfer measurement.
        let t = Instant::now();
        if let Err(reason) = self.driver.sync() {
            self.free_all(&mut columns);
            return Err(self.backend_error("cuCtxSynchronize", reason));
        }
        m.device_sync_ns = nanos(t.elapsed());
        m.uploaded_bytes = allocated;
        m.total_ns = nanos(started.elapsed());

        let token = next_token();
        let handle = ResidentGraph::new(
            token,
            view.snapshot_id,
            view.node_count,
            view.edge_count(),
            bytes,
        )
        .with_dst_layout(view.dst_layout());
        self.graphs()?.insert(
            token,
            DeviceGraph {
                snapshot_id: view.snapshot_id,
                columns,
                bytes,
            },
        );
        if let Ok(mut slot) = self.last_load.lock() {
            *slot = Some(m);
        }
        Ok(Some(handle))
    }

    fn last_load_metrics(&self) -> Option<GraphLoadMetrics> {
        // Explicitly the inherent method: the trait's default returns `None`,
        // and a silent self-recursion here would be a stack overflow in a test.
        CudaAccelerator::last_load_metrics(self)
    }

    fn release_graph(&self, handle: &ResidentGraph) {
        let Ok(mut g) = self.graphs.lock() else {
            return;
        };
        let Some(entry) = g.remove(&handle.token()) else {
            // Releasing an unknown token is a no-op: the cache legitimately
            // retries a release, and a failed load never inserted anything.
            return;
        };
        if self.driver.bind(self.context).is_err() {
            // Without the context the frees would fail anyway; dropping the
            // entry is the best that can be done and leaks nothing else.
            return;
        }
        for p in entry.columns {
            self.driver.free(p);
        }
    }
}

impl Drop for CudaAccelerator {
    fn drop(&mut self) {
        if self.driver.bind(self.context).is_ok() {
            if let Ok(mut g) = self.graphs.lock() {
                for (_, entry) in g.drain() {
                    for p in entry.columns {
                        self.driver.free(p);
                    }
                }
            }
        }
        // Release our retain of the primary context. Other users in the same
        // process keep theirs; the context dies when the last one goes.
        self.driver.release_context(self.device.handle);
    }
}

impl std::fmt::Debug for CudaAccelerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaAccelerator")
            .field("device", &self.device)
            .field("budget_bytes", &self.budget)
            .finish()
    }
}

/// A stage time as whole nanoseconds, saturating rather than wrapping: a load
/// long enough to overflow 584 years is not one worth reporting on.
fn nanos(d: std::time::Duration) -> u64 {
    d.as_nanos().min(u64::MAX as u128) as u64
}

fn next_token() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_reports_a_reason_instead_of_panicking() {
        match CudaAccelerator::open() {
            Ok(acc) => {
                let caps = acc.caps();
                assert_eq!(caps.backend, BackendKind::Cuda);
                assert!(caps.resident);
                assert!(caps.max_device_bytes.unwrap() > 0);
                // C1: nothing is advertised until a kernel exists.
                assert!(!caps.supports(gtv_accelerator::Op::KnnFlat));
                assert!(acc.budget_bytes() > 0);
                assert!(acc.resident_snapshots().is_empty());
            }
            Err(AcceleratorError::DeviceUnavailable { reason, .. }) => {
                assert!(!reason.is_empty());
            }
            Err(other) => panic!("unexpected failure kind: {other}"),
        }
    }

    #[test]
    fn probe_and_open_agree_about_availability() {
        let probed = CudaAccelerator::probe().is_ok();
        let opened = CudaAccelerator::open().is_ok();
        // `probe` creates no context, so it can only be stricter: anything that
        // opens must have probed successfully.
        if opened {
            assert!(probed);
        }
    }
}
