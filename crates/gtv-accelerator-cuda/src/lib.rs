//! The CUDA backend — residency and the flat KNN kernel.
//!
//! # What this crate does today (C3)
//!
//! It enumerates the device, reports its limits, makes a snapshot *actually*
//! resident (`cuMemAlloc` + `cuMemcpyHtoD` per column, `cuMemFree` on release),
//! and runs **two** operators: [`Accelerator::knn_flat`] over resident vectors,
//! and the rolling-window aggregates over a column that may be resident
//! ([`Accelerator::load_column`]) or uploaded per call. Everything else reports
//! [`gtv_accelerator::AcceleratorError::Unsupported`] and the policy sends it to
//! the CPU.
//!
//! KNN, measured on this machine (RTX 4060 Laptop, release, 100k x 128, k=10):
//! 3.5x against the CPU at batch 1 with the upload included, 14.3x at batch 32,
//! break-even after five queries. The stage table for batch 64 is kernel 99.1%,
//! upload 0.2%, download 0.5% (15.9 KB), merge 0.1% — the transfer has stopped
//! being the story, which is the whole point of having done residency first.
//!
//! The window aggregates are the counter-example, and they are here because the
//! negative result is worth more than the operator. The kernel is 18x the CPU's
//! arithmetic (1.13 ms against 20.6 ms at 1e7 rows) and the operator is still
//! *off* by default: its result is the same size as its input, so `8n` up and
//! `8n` down is not a cost the kernel can pay back, and returning a fresh 80 MB
//! `Vec` adds 7.8 ms of page faults on top. Read `docs/cuda-acceleration-plan.md`
//! §9.8 before touching it, and `ExecutionPolicy::min_window_rows` before
//! enabling it.
//!
//! # Why residency came before kernels
//!
//! Pageable H2D runs at 7.5–8.0 GB/s and is 97% of the load, so one upload of a
//! 1e7-edge snapshot costs 30–41 ms against a 6.9 ms CPU scan of the same graph.
//! A kernel measured before residency exists measures PCIe, not arithmetic.
//!
//! The upload, though, is not the expensive part of going to the device. The CSR
//! it consumes used to cost ~1.6 s to materialise at 1e7 edges (`graph_at` walked
//! the segments into `Vec<LiveRow>`, then into `Vec<EdgeRecord>`, then folded that
//! into columns), which put the cold path 236x above the CPU's 6.86 ms — 98% of
//! it before this crate was ever called.
//!
//! That has since been fixed at the storage layer (§9.4): `graph_at` merges the
//! segments' own sorted columns instead of rebuilding them, so the same snapshot
//! costs 393 ms while uncompacted and 176 ms once compacted into one segment.
//! The upload below (41 ms) is now 9–19% of the cold path rather than 2.5%. The
//! order still holds — measure, make materialisation cheap, then write kernels
//! against a copy that is already resident — because residency amortises a 41 ms
//! upload over five queries, but nothing amortises a rebuild on every query.
//!
//! # The load receipt
//!
//! Every upload records [`gtv_accelerator::GraphLoadMetrics`]: per column bytes,
//! allocation and copy time, plus the stages that are supposed to cost nothing
//! (host pack, layout transform, staging). The zero-cost stages are fields rather
//! than absences so that a regression is a number rather than a mystery, and
//! `tests/device.rs` asserts them. A loader that reports only "upload took 31 ms"
//! cannot tell a PCIe bottleneck from a re-pack, and will optimise the wrong one.
//! Vector loads get the same treatment ([`VectorLoadMetrics`], including the
//! normalisation that `dot`/`cosine` needs), and each KNN call reports
//! [`KnnMetrics`] split into its four stages.
//!
//! # PTX is checked in, not compiled here
//!
//! `kernels/*.ptx` are generated artefacts: `cargo xtask cuda-ptx` compiles each
//! `.cu` and stamps its BLAKE3 into the header, and `tests/ptx_stamp.rs`
//! recomputes every stamp on every `cargo test`. That test is the only thing
//! about the kernels verifiable on the GPU-less CI, which is why the stamp
//! exists rather than a comment asking people to remember.
//!
//! # `Caps` advertises exactly what landed
//!
//! [`gtv_accelerator::ExecutionPolicy`] gates on `Caps::supports`. The
//! [`gtv_accelerator::Op`] bits advertised here are the operators whose kernels
//! are in this crate, and `tests/device.rs::the_kernels_advertised_are_the_kernels_that_landed`
//! fails if a bit and a module ever disagree. `Caps::max_k` publishes the
//! kernel's shared-memory ceiling so the planner avoids `k > 256` instead of
//! paying a launch to learn it — and `knn_flat` still returns `LimitExceeded`
//! (`Fallback::Cpu`) for callers that skip the planner.
//!
//! # Absent driver is a normal answer
//!
//! [`CudaAccelerator::open`] returns `DeviceUnavailable` when the library or the
//! device is missing. No panic, no abort, no link-time dependency: this crate
//! builds and tests on the GPU-less CI machine.

pub mod driver;
pub mod knn;
pub mod window;

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use gtv_accelerator::{
    Accelerator, AcceleratorError, BackendKind, Caps, Column, GraphLoadMetrics, KnnMetrics, Op,
    ResidentColumn, ResidentGraph, ResidentVectors, SnapshotView, VectorLoadMetrics, VectorView,
    WindowMetrics, WindowOp,
};
use gtv_core::{Metric, Sel, VectorHit};

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
    /// One entry per uploaded column. The projection decides how many there
    /// are, so a pointer's position no longer says which column it is — a
    /// kernel must look the column up by name.
    columns: Vec<(Column, u64)>,
    bytes: usize,
}

/// One resident vector matrix, as device pointers.
///
/// Two allocations rather than one arena: a release then frees exactly what a
/// load allocated, and a partial failure frees exactly what it got, which is the
/// same rule `load_graph` follows per column.
struct DeviceVectors {
    vector_space_id: u32,
    snapshot_id: u64,
    rows: usize,
    dim: usize,
    metric: Metric,
    data: u64,
    ids: u64,
    bytes: usize,
}

impl DeviceVectors {
    fn free(&self, driver: &Driver) {
        driver.free(self.data);
        driver.free(self.ids);
    }
}

/// One resident value column, as a device pointer.
///
/// A single allocation, unlike vectors: the rolling-window kernel reads the
/// column and writes a separate result buffer, so there is no companion array to
/// keep together with it.
struct DeviceColumn {
    rows: usize,
    data: u64,
    bytes: usize,
}

impl DeviceColumn {
    fn free(&self, driver: &Driver) {
        driver.free(self.data);
    }
}

/// Device allocations owned for the length of one call.
///
/// The KNN path allocates five buffers and has a dozen early returns; freeing by
/// hand on each is how a leak gets introduced years later. `Drop` frees what was
/// taken, including on the error paths, and the `Mutex` is not involved.
struct Scratch<'a> {
    driver: &'a Driver,
    ptrs: Vec<u64>,
}

impl<'a> Scratch<'a> {
    fn new(driver: &'a Driver) -> Self {
        Self {
            driver,
            ptrs: Vec::new(),
        }
    }

    fn alloc(&mut self, bytes: usize) -> Result<u64, String> {
        let ptr = self.driver.alloc(bytes)?;
        self.ptrs.push(ptr);
        Ok(ptr)
    }
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        for p in self.ptrs.drain(..) {
            self.driver.free(p);
        }
    }
}

/// The CUDA backend.
pub struct CudaAccelerator {
    driver: &'static Driver,
    device: DeviceInfo,
    context: Context,
    budget: usize,
    graphs: Mutex<HashMap<u64, DeviceGraph>>,
    vectors: Mutex<HashMap<u64, DeviceVectors>>,
    columns: Mutex<HashMap<u64, DeviceColumn>>,
    /// Receipt for the most recent upload, kept so a benchmark can read the
    /// stage breakdown after the fact instead of being handed a second return
    /// value it does not want.
    last_load: Mutex<Option<GraphLoadMetrics>>,
    last_vector_load: Mutex<Option<VectorLoadMetrics>>,
    last_knn: Mutex<Option<KnnMetrics>>,
    last_window: Mutex<Option<WindowMetrics>>,
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
            vectors: Mutex::new(HashMap::new()),
            columns: Mutex::new(HashMap::new()),
            last_load: Mutex::new(None),
            last_vector_load: Mutex::new(None),
            last_knn: Mutex::new(None),
            last_window: Mutex::new(None),
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
        let graphs: usize = self
            .graphs
            .lock()
            .map(|g| g.values().map(|e| e.bytes).sum())
            .unwrap_or(0);
        // All three kinds share one budget, so all three have to be counted; a
        // report that forgets one understates what the card is holding, and the
        // admission check would then let an upload through that does not fit.
        let vectors: usize = self
            .vectors
            .lock()
            .map(|v| v.values().map(|e| e.bytes).sum())
            .unwrap_or(0);
        let columns: usize = self
            .columns
            .lock()
            .map(|c| c.values().map(|e| e.bytes).sum())
            .unwrap_or(0);
        graphs + vectors + columns
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

    /// Device vectors resident right now, for tests and for a pre-warm loop.
    pub fn resident_spaces(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self
            .vectors
            .lock()
            .map(|s| s.values().map(|e| e.vector_space_id).collect())
            .unwrap_or_default();
        v.sort_unstable();
        v
    }

    /// Stage breakdown of the last [`Accelerator::load_vectors`].
    pub fn last_vector_load_metrics(&self) -> Option<VectorLoadMetrics> {
        self.last_vector_load.lock().ok().and_then(|m| m.clone())
    }

    /// Stage breakdown of the last [`Accelerator::knn_flat`].
    pub fn last_knn_metrics(&self) -> Option<KnnMetrics> {
        self.last_knn.lock().ok().and_then(|m| m.clone())
    }

    /// Make the context current on this thread, or say why not.
    ///
    /// Every entry point below does this first: the accelerator is `Send + Sync`
    /// and may be called from a pool thread, while CUDA contexts are per-thread.
    fn bind(&self) -> Result<(), AcceleratorError> {
        self.driver
            .bind(self.context)
            .map_err(|reason| self.backend_error("cuCtxSetCurrent", reason))
    }

    fn vectors(&self) -> Result<std::sync::MutexGuard<'_, HashMap<u64, DeviceVectors>>, AcceleratorError>
    {
        self.vectors.lock().map_err(|_| AcceleratorError::Backend {
            backend: "cuda",
            op: "lock",
            code: -1,
            message: "vector table poisoned".into(),
        })
    }

    fn free_all(&self, columns: &mut Vec<(Column, u64)>) {
        for (_, p) in columns.drain(..) {
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

    fn columns(&self) -> Result<std::sync::MutexGuard<'_, HashMap<u64, DeviceColumn>>, AcceleratorError>
    {
        self.columns.lock().map_err(|_| AcceleratorError::Backend {
            backend: "cuda",
            op: "lock",
            code: -1,
            message: "column table poisoned".into(),
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
        // One bit per landed kernel. The policy needs no change to follow: it
        // gates on `supports(op)`, so `knn_flat` and `window_reduce` go to the
        // device when the work clears their thresholds and to the CPU otherwise.
        //
        // Both limits are advertised rather than only enforced, because both are
        // properties of a block's shared-memory budget (`8 warps x k x 12 B` for
        // the heap, `chunk + window - 1` rows for the scan) rather than of the
        // arithmetic. A planner should be able to route a 10 000-neighbour query
        // or a 100 000-row window to the CPU without making a call that can only
        // fail.
        Caps::new(BackendKind::Cuda, "cuda-0", Some(self.budget))
            .with_resident(true)
            .with_op(Op::KnnFlat)
            .with_op(Op::WindowReduce)
            .with_max_k(knn::MAX_K)
            .with_max_window(window::MAX_WINDOW)
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

        // Too big is a scheduling fact, not a backend failure: say so before
        // binding, so the caller gets the two numbers instead of a `cuMemAlloc`
        // code from a half-populated pool. With a projection this check is what
        // makes "topology fits, the time columns do not" an answerable question.
        self.caps().admit(bytes)?;

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
        // The projection decides the list, so a topology-only upload allocates
        // two buffers and touches no time column at all. `validate` has already
        // refused a projection without topology, which is what lets the match
        // below assume `offsets` is present when it is asked for.
        let mut cols: Vec<(Column, *const u8, usize)> = Vec::with_capacity(view.columns.len());
        for col in view.columns.iter() {
            let (src, len) = match col {
                Column::Offsets => (
                    view.offsets.as_ptr() as *const u8,
                    std::mem::size_of_val(view.offsets),
                ),
                Column::Dst => dst_src,
                Column::ValidFrom => (
                    view.valid_from.as_ptr() as *const u8,
                    std::mem::size_of_val(view.valid_from),
                ),
                Column::ValidTo => (
                    view.valid_to.as_ptr() as *const u8,
                    std::mem::size_of_val(view.valid_to),
                ),
                Column::EdgeType => (
                    view.edge_type.as_ptr() as *const u8,
                    std::mem::size_of_val(view.edge_type),
                ),
            };
            cols.push((col, src, len));
        }
        m.input_bytes = cols.iter().map(|(_, _, len)| *len).sum();
        m.projection = view.columns;
        m.deferred_bytes = view.deferred_bytes();

        let mut columns: Vec<(Column, u64)> = Vec::with_capacity(cols.len());
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
            columns.push((col, ptr));
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
        .with_dst_layout(view.dst_layout())
        .with_columns(view.columns);
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
        for (_, p) in entry.columns {
            self.driver.free(p);
        }
    }

    fn load_vectors(
        &self,
        view: VectorView<'_>,
    ) -> Result<Option<ResidentVectors>, AcceleratorError> {
        let started = Instant::now();
        view.validate()?;
        let bytes = view.bytes();
        let mut m = VectorLoadMetrics::new(
            view.vector_space_id,
            view.snapshot_id,
            view.rows(),
            view.dim,
        );
        m.priced_bytes = bytes;
        // Same rule as the graph path: a failed load must not leave the previous
        // space's numbers behind for a caller to attribute to this one.
        if let Ok(mut slot) = self.last_vector_load.lock() {
            *slot = None;
        }
        self.caps().admit(bytes)?;
        self.bind()?;

        // Cosine space is unit-normalised by the CPU reference at build time
        // (`FlatIndex::from_flat_metric`), and the device copy has to be the same
        // matrix or the two sides answer differently. It is a host pass over the
        // whole corpus, so it is timed rather than folded into "upload": at
        // 1e5 x 128 it is 51 MB of CPU work, which is not PCIe and must not be
        // blamed on it.
        let normalized: Option<Vec<f32>> = if view.metric.requires_normalization() {
            let t = Instant::now();
            let mut data = view.data.to_vec();
            for row in data.chunks_mut(view.dim) {
                view.metric.normalize_in_place(row);
            }
            m.normalize_ns = nanos(t.elapsed());
            Some(data)
        } else {
            None
        };
        let data: &[f32] = normalized.as_deref().unwrap_or(view.data);

        let t = Instant::now();
        let d_data = self
            .driver
            .alloc(std::mem::size_of_val(data))
            .map_err(|reason| self.backend_error("cuMemAlloc(vectors)", reason))?;
        let d_ids = match self.driver.alloc(std::mem::size_of_val(view.ids)) {
            Ok(p) => p,
            Err(reason) => {
                self.driver.free(d_data);
                return Err(self.backend_error("cuMemAlloc(ids)", reason));
            }
        };
        m.alloc_ns = nanos(t.elapsed());

        let t = Instant::now();
        // SAFETY: both destination pointers come from `alloc` with exactly the
        // byte count passed here, and both sources are live slices of that
        // length. Nothing else can reach the allocations until they are inserted
        // into the table below.
        let copy = unsafe {
            self.driver
                .copy_htod(d_data, data.as_ptr() as *const u8, std::mem::size_of_val(data))
                .and_then(|()| {
                    self.driver.copy_htod(
                        d_ids,
                        view.ids.as_ptr() as *const u8,
                        std::mem::size_of_val(view.ids),
                    )
                })
        };
        if let Err(reason) = copy {
            self.driver.free(d_data);
            self.driver.free(d_ids);
            return Err(self.backend_error("cuMemcpyHtoD(vectors)", reason));
        }
        m.copy_ns = nanos(t.elapsed());
        m.uploaded_bytes = bytes;
        m.total_ns = nanos(started.elapsed());

        let token = next_token();
        let handle = ResidentVectors::new(
            token,
            view.vector_space_id,
            view.snapshot_id,
            view.rows(),
            view.dim,
            view.metric,
            bytes,
        );
        self.vectors()?.insert(
            token,
            DeviceVectors {
                vector_space_id: view.vector_space_id,
                snapshot_id: view.snapshot_id,
                rows: view.rows(),
                dim: view.dim,
                metric: view.metric,
                data: d_data,
                ids: d_ids,
                bytes,
            },
        );
        if let Ok(mut slot) = self.last_vector_load.lock() {
            *slot = Some(m);
        }
        Ok(Some(handle))
    }

    fn last_vector_load_metrics(&self) -> Option<VectorLoadMetrics> {
        CudaAccelerator::last_vector_load_metrics(self)
    }

    fn release_vectors(&self, handle: &ResidentVectors) {
        let Ok(mut table) = self.vectors.lock() else {
            return;
        };
        let Some(entry) = table.remove(&handle.token()) else {
            return;
        };
        if self.driver.bind(self.context).is_err() {
            // Without the context the frees would fail anyway; dropping the
            // entry is the best that remains and leaks nothing else.
            return;
        }
        entry.free(self.driver);
    }

    fn knn_flat(
        &self,
        handle: &ResidentVectors,
        queries: &[f32],
        k: usize,
        filter: Option<&Sel>,
    ) -> Result<Vec<Vec<VectorHit>>, AcceleratorError> {
        let started = Instant::now();
        let dim = handle.dim();
        if dim == 0 || !queries.len().is_multiple_of(dim) {
            return Err(AcceleratorError::InvalidArgument(format!(
                "{} query floats is not a multiple of dim {dim}",
                queries.len()
            )));
        }
        let batch = queries.len() / dim;
        if let Some(f) = filter {
            // Same two checks as the CPU reference, for the same reason: a
            // candidate set from another snapshot would select the wrong rows
            // and say nothing.
            if f.is_bound() {
                handle.check_snapshot(f.snapshot_id(), handle.vector_space_id())?;
            }
            if f.universe() != handle.rows() as u64 {
                return Err(AcceleratorError::InvalidArgument(format!(
                    "filter covers {} rows, space has {}",
                    f.universe(),
                    handle.rows()
                )));
            }
        }
        if k > knn::MAX_K {
            // A limit, not a malformed request: the CPU reference has no such
            // bound, so the caller can still get an answer there.
            return Err(AcceleratorError::LimitExceeded {
                backend: "cuda",
                what: "k",
                requested: k,
                limit: knn::MAX_K,
            });
        }
        if let Ok(mut slot) = self.last_knn.lock() {
            *slot = None;
        }
        // `k == 0`, an empty batch and an empty space all answer the same way the
        // CPU does — a list per query, empty — without touching the device.
        if k == 0 || batch == 0 || handle.rows() == 0 {
            return Ok(vec![Vec::new(); batch]);
        }

        let (rows, metric, d_data, d_ids) = {
            let table = self.vectors()?;
            let v = table.get(&handle.token()).ok_or_else(|| {
                AcceleratorError::InvalidArgument(format!(
                    "resident vectors {} are not loaded (released?)",
                    handle.token()
                ))
            })?;
            handle.check_snapshot(v.snapshot_id, v.vector_space_id)?;
            if v.dim != dim {
                return Err(AcceleratorError::InvalidArgument(format!(
                    "resident vectors have dim {} but the handle says {dim}",
                    v.dim
                )));
            }
            (v.rows, v.metric, v.data, v.ids)
        };
        let mut m = KnnMetrics::new(batch, rows, dim, k);
        m.filtered = filter.is_some();
        m.splits = knn::splits_for(batch, rows, self.device.multiprocessors);
        self.bind()?;

        // Cosine queries are normalised exactly as `FlatIndex::prepare_query`
        // does; the rows were normalised at load, so `1 - dot` is the distance.
        let owned_queries: Option<Vec<f32>> = if metric.requires_normalization() {
            let mut q = queries.to_vec();
            for one in q.chunks_exact_mut(dim) {
                metric.normalize_in_place(one);
            }
            Some(q)
        } else {
            None
        };
        let q: &[f32] = owned_queries.as_deref().unwrap_or(queries);
        let words = filter.map(|f| f.to_bitmap_words());

        let mut scratch = Scratch::new(self.driver);
        let t = Instant::now();
        let d_q = scratch
            .alloc(std::mem::size_of_val(q))
            .map_err(|reason| self.backend_error("cuMemAlloc(queries)", reason))?;
        let d_filter = match &words {
            Some(w) => scratch
                .alloc(std::mem::size_of_val(w.as_slice()))
                .map_err(|reason| self.backend_error("cuMemAlloc(filter)", reason))?,
            None => 0,
        };
        let slots = batch * m.splits;
        // One allocation for the three result arrays, so the read back is three
        // copies out of one pool block rather than three allocations: at k=10 the
        // result is a few kilobytes and the allocation would cost more than the
        // bytes.
        let arena_bytes = slots * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>() * k)
            + slots * std::mem::size_of::<u64>() * k
            + 64;
        let d_arena = scratch
            .alloc(arena_bytes)
            .map_err(|reason| self.backend_error("cuMemAlloc(knn out)", reason))?;
        let align8 = |x: usize| x.div_ceil(8) * 8;
        let d_count = d_arena;
        let d_dist = d_arena + align8(slots * std::mem::size_of::<i32>()) as u64;
        let d_id = d_dist + align8(slots * std::mem::size_of::<f32>() * k) as u64;
        // SAFETY: destinations are the three sub-ranges of `d_arena` sized above,
        // sources are live slices of matching length.
        unsafe {
            self.driver
                .copy_htod(d_q, q.as_ptr() as *const u8, std::mem::size_of_val(q))
                .map_err(|reason| self.backend_error("cuMemcpyHtoD(queries)", reason))?;
            if let Some(w) = &words {
                self.driver
                    .copy_htod(
                        d_filter,
                        w.as_ptr() as *const u8,
                        std::mem::size_of_val(w.as_slice()),
                    )
                    .map_err(|reason| self.backend_error("cuMemcpyHtoD(filter)", reason))?;
            }
        }
        m.upload_bytes = std::mem::size_of_val(q) + words.as_ref().map_or(0, |w| w.len() * 4);
        m.upload_ns = nanos(t.elapsed());

        // Timed with events, not wall clock: the launch returns as soon as the
        // kernel is queued, so wall time around it would measure the submission
        // and wall time around the copy would measure kernel + copy together.
        let e0 = self
            .driver
            .event()
            .map_err(|reason| self.backend_error("cuEventCreate", reason))?;
        let e1 = self
            .driver
            .event()
            .map_err(|reason| self.backend_error("cuEventCreate", reason))?;
        let launch = knn::Launch {
            batch,
            splits: m.splits,
            corpus: d_data,
            ids: d_ids,
            rows: rows as i64,
            queries: d_q,
            dim: dim as i32,
            k: k as i32,
            metric: knn::metric_code(metric),
            filter: d_filter,
            out_dist: d_dist,
            out_id: d_id,
            out_count: d_count,
        };
        e0.record()
            .map_err(|reason| self.backend_error("cuEventRecord", reason))?;
        knn::launch(&launch).map_err(|reason| self.backend_error("cuLaunchKernel(knn)", reason))?;
        e1.record()
            .map_err(|reason| self.backend_error("cuEventRecord", reason))?;
        // Wait for the kernel through the event, so the duration is the kernel's
        // and not the copy's.
        e1.synchronize()
            .map_err(|reason| self.backend_error("cuEventSynchronize", reason))?;
        m.kernel_ns = (e0
            .elapsed_ms(&e1)
            .map_err(|reason| self.backend_error("cuEventElapsedTime", reason))?
            * 1e6) as u64;

        let t = Instant::now();
        let mut counts = vec![0i32; slots];
        let mut dist = vec![0f32; slots * k];
        let mut ids = vec![0u64; slots * k];
        // SAFETY: each destination is a live `Vec` of exactly the length copied,
        // and each source is a range of `d_arena` that the kernel writes and that
        // is at least that long (see `arena_bytes`).
        unsafe {
            self.driver
                .copy_dtoh(counts.as_mut_ptr() as *mut u8, d_count, std::mem::size_of_val(&counts[..]))
                .and_then(|()| {
                    self.driver.copy_dtoh(
                        dist.as_mut_ptr() as *mut u8,
                        d_dist,
                        std::mem::size_of_val(&dist[..]),
                    )
                })
                .and_then(|()| {
                    self.driver.copy_dtoh(
                        ids.as_mut_ptr() as *mut u8,
                        d_id,
                        std::mem::size_of_val(&ids[..]),
                    )
                })
                .map_err(|reason| self.backend_error("cuMemcpyDtoH(knn)", reason))?;
        }
        m.download_ns = nanos(t.elapsed());
        m.result_bytes = std::mem::size_of_val(&counts[..])
            + std::mem::size_of_val(&dist[..])
            + std::mem::size_of_val(&ids[..]);

        let t = Instant::now();
        let hits = merge_splits(&counts, &dist, &ids, batch, m.splits, k);
        m.merge_ns = nanos(t.elapsed());
        m.total_ns = nanos(started.elapsed());
        if let Ok(mut slot) = self.last_knn.lock() {
            *slot = Some(m);
        }
        Ok(hits)
    }

    fn last_knn_metrics(&self) -> Option<KnnMetrics> {
        CudaAccelerator::last_knn_metrics(self)
    }

    fn window_reduce(
        &self,
        values: &[f64],
        window: usize,
        op: WindowOp,
    ) -> Result<Vec<f64>, AcceleratorError> {
        let started = Instant::now();
        // Order matters: the limit is a limit (fall back), not a malformed
        // request. `window == 0` and an empty column are the CPU's answers,
        // computed without touching the device at all.
        self.check_window(window)?;
        if window == 0 || values.is_empty() {
            return Ok(Vec::new());
        }
        let rows = values.len();
        let bytes = std::mem::size_of_val(values);
        let mut m = WindowMetrics::new(rows, window);
        m.upload_bytes = bytes;
        m.result_bytes = bytes;
        self.bind()?;

        let mut scratch = Scratch::new(self.driver);
        let t = Instant::now();
        let d_in = scratch
            .alloc(bytes)
            .map_err(|reason| self.backend_error("cuMemAlloc(column)", reason))?;
        let d_out = scratch
            .alloc(bytes)
            .map_err(|reason| self.backend_error("cuMemAlloc(window out)", reason))?;
        // SAFETY: both sides are `bytes` long; `d_in` came from `alloc(bytes)`.
        unsafe {
            self.driver
                .copy_htod(d_in, values.as_ptr() as *const u8, bytes)
                .map_err(|reason| self.backend_error("cuMemcpyHtoD(column)", reason))?;
        }
        m.upload_ns = nanos(t.elapsed());

        let mut out = vec![0f64; rows];
        self.window_launch(d_in, d_out, rows, window, op, &mut out, &mut m)?;
        m.total_ns = nanos(started.elapsed());
        self.remember_window(m);
        Ok(out)
    }

    fn load_column(
        &self,
        values: &[f64],
    ) -> Result<Option<ResidentColumn>, AcceleratorError> {
        if values.is_empty() {
            return Err(AcceleratorError::InvalidArgument(
                "a resident column needs at least one row".into(),
            ));
        }
        let bytes = std::mem::size_of_val(values);
        // Same pre-flight the graph upload does: refuse before a byte is
        // allocated, so an over-budget column is a size, not a driver code in
        // the middle of a copy.
        self.caps().admit(bytes)?;
        self.bind()?;
        let d = self
            .driver
            .alloc(bytes)
            .map_err(|reason| self.backend_error("cuMemAlloc(resident column)", reason))?;
        // SAFETY: destination is `bytes` from `alloc(bytes)`, source is a live
        // slice of exactly that length.
        if let Err(reason) = unsafe {
            self.driver
                .copy_htod(d, values.as_ptr() as *const u8, bytes)
        } {
            self.driver.free(d);
            return Err(self.backend_error("cuMemcpyHtoD(resident column)", reason));
        }
        let token = next_token();
        let mut table = self.columns()?;
        table.insert(
            token,
            DeviceColumn {
                rows: values.len(),
                data: d,
                bytes,
            },
        );
        Ok(Some(ResidentColumn::new(token, values.len(), bytes)))
    }

    fn release_column(&self, handle: &ResidentColumn) {
        if self.bind().is_err() {
            return;
        }
        if let Ok(mut table) = self.columns() {
            if let Some(c) = table.remove(&handle.token()) {
                c.free(self.driver);
            }
        }
    }

    fn window_reduce_resident(
        &self,
        handle: &ResidentColumn,
        window: usize,
        op: WindowOp,
    ) -> Result<Vec<f64>, AcceleratorError> {
        let started = Instant::now();
        self.check_window(window)?;
        let (rows, d_in) = {
            let table = self.columns()?;
            let c = table.get(&handle.token()).ok_or_else(|| {
                AcceleratorError::InvalidArgument(format!(
                    "resident column {} is not loaded (released?)",
                    handle.token()
                ))
            })?;
            (c.rows, c.data)
        };
        if window == 0 || rows == 0 {
            return Ok(Vec::new());
        }
        let mut m = WindowMetrics::new(rows, window);
        m.resident = true;
        m.result_bytes = rows * std::mem::size_of::<f64>();
        self.bind()?;

        let mut scratch = Scratch::new(self.driver);
        let d_out = scratch
            .alloc(m.result_bytes)
            .map_err(|reason| self.backend_error("cuMemAlloc(window out)", reason))?;
        let mut out = vec![0f64; rows];
        self.window_launch(d_in, d_out, rows, window, op, &mut out, &mut m)?;
        m.total_ns = nanos(started.elapsed());
        self.remember_window(m);
        Ok(out)
    }

    fn last_window_metrics(&self) -> Option<WindowMetrics> {
        CudaAccelerator::last_window_metrics(self)
    }
}

impl CudaAccelerator {
    /// Shared between the stateless and the resident form: launch, time the
    /// kernel with events, read the result back.
    ///
    /// One implementation because the two forms differ only in whether `d_in`
    /// arrived on this call — and a second copy of this code would be a second
    /// place for the event timing to go wrong.
    #[allow(clippy::too_many_arguments)]
    fn window_launch(
        &self,
        d_in: u64,
        d_out: u64,
        rows: usize,
        window: usize,
        op: WindowOp,
        out: &mut [f64],
        m: &mut WindowMetrics,
    ) -> Result<(), AcceleratorError> {
        let shared = window::shared_bytes(op, window, rows);
        let e0 = self
            .driver
            .event()
            .map_err(|reason| self.backend_error("cuEventCreate", reason))?;
        let e1 = self
            .driver
            .event()
            .map_err(|reason| self.backend_error("cuEventCreate", reason))?;
        let launch = window::Launch {
            values: d_in,
            rows: rows as i64,
            window: window as i32,
            op: window::op_code(op),
            out: d_out,
        };
        e0.record()
            .map_err(|reason| self.backend_error("cuEventRecord", reason))?;
        window::launch(&launch, shared)
            .map_err(|reason| self.backend_error("cuLaunchKernel(window)", reason))?;
        e1.record()
            .map_err(|reason| self.backend_error("cuEventRecord", reason))?;
        e1.synchronize()
            .map_err(|reason| self.backend_error("cuEventSynchronize", reason))?;
        m.kernel_ns = (e0
            .elapsed_ms(&e1)
            .map_err(|reason| self.backend_error("cuEventElapsedTime", reason))?
            * 1e6) as u64;

        let t = Instant::now();
        // SAFETY: destination is a live `Vec` of exactly `rows` elements, source
        // is the `d_out` buffer the kernel just wrote, sized the same.
        unsafe {
            self.driver
                .copy_dtoh(out.as_mut_ptr() as *mut u8, d_out, std::mem::size_of_val(out))
                .map_err(|reason| self.backend_error("cuMemcpyDtoH(window)", reason))?;
        }
        m.download_ns = nanos(t.elapsed());
        Ok(())
    }

    /// Reject the two cases the kernel cannot serve, in the order that keeps
    /// them distinguishable: `0` is a request with a defined answer (the CPU
    /// returns an empty column), a large window is a limit.
    fn check_window(&self, window: usize) -> Result<(), AcceleratorError> {
        if window > window::MAX_WINDOW {
            return Err(AcceleratorError::LimitExceeded {
                backend: "cuda",
                what: "window",
                requested: window,
                limit: window::MAX_WINDOW,
            });
        }
        Ok(())
    }

    fn remember_window(&self, m: WindowMetrics) {
        if let Ok(mut slot) = self.last_window.lock() {
            *slot = Some(m);
        }
    }

    pub fn last_window_metrics(&self) -> Option<WindowMetrics> {
        self.last_window.lock().ok().and_then(|m| *m)
    }
}

impl Drop for CudaAccelerator {
    fn drop(&mut self) {
        if self.driver.bind(self.context).is_ok() {
            if let Ok(mut g) = self.graphs.lock() {
                for (_, entry) in g.drain() {
                    for (_, p) in entry.columns {
                        self.driver.free(p);
                    }
                }
            }
            if let Ok(mut v) = self.vectors.lock() {
                for (_, entry) in v.drain() {
                    entry.free(self.driver);
                }
            }
            if let Ok(mut c) = self.columns.lock() {
                for (_, entry) in c.drain() {
                    entry.free(self.driver);
                }
            }
        }
        // Release our retain of the primary context — unless a kernel module is
        // cached. The module lives *in* the context, and it is process-wide (one
        // JIT per process, not one per accelerator); releasing the last retain
        // would destroy the context and leave every later launch returning
        // `CUDA_ERROR_INVALID_HANDLE`. So the first accelerator to load a kernel
        // promotes the context to process lifetime. That is a leak of exactly one
        // context, at exit, in exchange for the kernel staying loaded; the
        // allocations inside it were freed above.
        if !(knn::loaded() || window::loaded()) {
            self.driver.release_context(self.device.handle);
        }
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

/// `(distance, id)` order, matching `cmp_scored` in `gtv-index`: ties go to the
/// smaller id. Without a total order the merge on this side could pick a
/// different row than the kernel did for the same distance, and "top-k parity"
/// would be a coin flip.
fn cmp_hit(a: &(f32, u64), b: &(f32, u64)) -> std::cmp::Ordering {
    a.0.partial_cmp(&b.0)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.1.cmp(&b.1))
}

/// Collapse `splits` sorted partial lists per query into the final top-k.
///
/// Each partial list is already ordered, but a k-way merge with cursors costs the
/// same as this at these sizes and is more code; what matters is that the result
/// is ordered by the *same* comparator the kernel used, so the two reductions
/// agree on ties.
fn merge_splits(
    counts: &[i32],
    dist: &[f32],
    ids: &[u64],
    batch: usize,
    splits: usize,
    k: usize,
) -> Vec<Vec<VectorHit>> {
    let mut out = Vec::with_capacity(batch);
    let mut buf: Vec<(f32, u64)> = Vec::with_capacity(splits * k);
    for q in 0..batch {
        buf.clear();
        for s in 0..splits {
            let slot = q * splits + s;
            let n = (counts[slot].max(0) as usize).min(k);
            for j in 0..n {
                buf.push((dist[slot * k + j], ids[slot * k + j]));
            }
        }
        if buf.len() > k {
            buf.select_nth_unstable_by(k, cmp_hit);
            buf.truncate(k);
        }
        buf.sort_unstable_by(cmp_hit);
        out.push(
            buf.iter()
                .map(|&(distance, id)| VectorHit { id, distance })
                .collect(),
        );
    }
    out
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
                // One bit per landed kernel, and the limit each one carries.
                assert!(caps.supports(Op::KnnFlat));
                assert!(caps.supports(Op::WindowReduce));
                assert!(!caps.supports(Op::BatchKhop));
                assert_eq!(caps.max_k, knn::MAX_K);
                assert_eq!(caps.max_window, window::MAX_WINDOW);
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
