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
//! That is not an unfinished C1; it is C1. The C0 probe measured 12 GB/s pinned
//! H2D and found the valid-time filter spending 96% of its cold path in the
//! transfer, with break-even at 2.1 invocations. A kernel measured before
//! residency exists measures PCIe, not arithmetic. So the order is: prove the
//! device can hold a snapshot and give it back — including under eviction, a
//! snapshot change, and a driver that is not there — then write kernels against
//! a copy that is already resident.
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

use gtv_accelerator::{
    Accelerator, AcceleratorError, BackendKind, Caps, ResidentGraph, SnapshotView,
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

    fn backend_error(&self, op: &'static str, reason: String) -> AcceleratorError {
        AcceleratorError::Backend {
            backend: "cuda",
            op,
            code: -1,
            message: reason,
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
        view.validate()?;
        let bytes = view.bytes();
        self.driver
            .bind(self.context)
            .map_err(|reason| self.backend_error("cuCtxSetCurrent", reason))?;

        // One allocation per column rather than one arena: an eviction frees
        // exactly what was allocated, and a partial failure frees exactly what
        // it already got. The source order is the CSR's own order, so each
        // upload is a straight memcpy with no repacking.
        let cols: [(&'static str, *const u8, usize); 5] = [
            (
                "offsets",
                view.offsets.as_ptr() as *const u8,
                std::mem::size_of_val(view.offsets),
            ),
            (
                "dst",
                view.dst.as_ptr() as *const u8,
                std::mem::size_of_val(view.dst),
            ),
            (
                "valid_from",
                view.valid_from.as_ptr() as *const u8,
                std::mem::size_of_val(view.valid_from),
            ),
            (
                "valid_to",
                view.valid_to.as_ptr() as *const u8,
                std::mem::size_of_val(view.valid_to),
            ),
            (
                "edge_type",
                view.edge_type.as_ptr() as *const u8,
                std::mem::size_of_val(view.edge_type),
            ),
        ];

        let mut columns: Vec<u64> = Vec::with_capacity(cols.len());
        for (name, src, len) in cols {
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
            // SAFETY: `ptr` came from `alloc(len)` in the line above and `src`
            // points at `len` readable bytes of the borrowed view.
            if let Err(reason) = unsafe { self.driver.copy_htod(ptr, src, len) } {
                self.driver.free(ptr);
                self.free_all(&mut columns);
                return Err(self.backend_error(
                    "cuMemcpyHtoD",
                    format!("{name} ({len} bytes): {reason}"),
                ));
            }
            columns.push(ptr);
        }

        let token = next_token();
        let handle = ResidentGraph::new(
            token,
            view.snapshot_id,
            view.node_count,
            view.edge_count(),
            bytes,
        );
        self.graphs()?.insert(
            token,
            DeviceGraph {
                snapshot_id: view.snapshot_id,
                columns,
                bytes,
            },
        );
        Ok(Some(handle))
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
