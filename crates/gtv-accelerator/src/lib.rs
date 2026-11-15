//! GPU acceleration layer for gtVdB: device budget, resident cache, host fallback.
//!
//! ## The shape of the problem
//!
//! PCIe moves ~8 GB/s while a modern device reads its own memory at several
//! hundred GB/s. So a kernel is only worth offloading when **the data stays**
//! and **the answer is much smaller than the input**. Exact K-NN returns `k`
//! hits from an `N`-row corpus; an as-of join returns one index per left row.
//! Both qualify. Rolling windows, `deltas`, `k`-hop traversal and pattern
//! backtracking do not — their output is the input, or their access pattern is
//! pointer-chasing — and are deliberately not exposed here.
//!
//! ## Layers
//!
//! ```text
//!            AcceleratorRegistry      <- decides, accounts, falls back
//!          /          |           \
//!   VramBudget   ResidentRegistry   AccelBackend
//!   (bytes)      (what is resident)  +-- HostBackend  (always available)
//!                                    `-- CudaBackend  (--features cuda)
//! ```
//!
//! `AcceleratorRegistry` never returns an error for an operational reason: if the
//! device refuses, it records why and runs the host kernel. A query therefore has
//! one behaviour with a GPU and without one — only the latency differs. That is
//! also what keeps GPU-less CI honest: the default build runs the same code path
//! production uses when no device is present.
//!
//! ## Where the design comes from
//!
//! The resident cache is modelled on PG-Strom's GPU Cache, including its central
//! insight that moving a table per query usually costs more than computing on it.
//! Three parts are rewritten rather than copied — see [`resident`] for the table.
//! The short version: admission control with LRU eviction instead of a static
//! row cap that corrupts itself on overflow, and a resident-prefix split instead
//! of making a query block until pending writes are applied.

pub mod backend;
pub mod budget;
pub mod gpudirect;
pub mod host;
pub mod resident;
pub mod sweep;
pub mod warm;

#[cfg(feature = "cuda")]
pub mod cuda;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

pub use gtv_params::{AccelCache, ResolvedParams};

pub use backend::{
    AccelBackend, AccelError, AccelOp, AsofRequest, DeviceInfo, DeviceKind, IvfRequest, KnnRequest,
    PrefetchRequest, RawHit,
};
pub use budget::{VramBudget, DEFAULT_HEADROOM};
pub use host::HostBackend;
pub use resident::{
    Admission, BlockView, DeclineReason, ResidentKey, ResidentKind, ResidentRegistry,
    ResidentTelemetry, SyncPolicy, ADMIT_MARGIN, DEFAULT_HEAT_HALF_LIFE, PREFETCH_HEAT_FLOOR,
    PROTECT_FLOOR,
};
pub use sweep::asof_match;
pub use warm::{warm_calls, warm_pages, warm_slice};

/// Total VRAM of `ordinal`, if a device is present and the crate was built with
/// the `cuda` feature.
///
/// The engine calls this during parameter resolution so `CUDA_VRAM`'s default
/// (`DEVICE:0.8`) becomes a byte count; without the feature it reports `None`
/// and CUDA stays disengaged, exactly as on a GPU-less host.
pub fn device_total_bytes(ordinal: usize) -> Option<u64> {
    #[cfg(feature = "cuda")]
    {
        cuda::device_total_bytes(ordinal)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinal;
        None
    }
}

/// Number of visible CUDA devices (`0` without the `cuda` feature).
pub fn device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        cuda::device_count()
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// Combined view of the device, the cache and the fallback counters.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceleratorTelemetry {
    pub device: DeviceInfo,
    pub mode: AccelCache,
    pub requests: u64,
    pub executed_on_device: u64,
    pub executed_on_host: u64,
    pub fallbacks: u64,
    pub cache_declined: u64,
    /// P1c page-warming hints served ([`AcceleratorRegistry::prefetch_hint`]).
    pub prefetch_hints: u64,
    /// P2 heat-pump uploads that completed (a hot non-resident block is now
    /// resident before the query that wanted it).
    pub prefetch_uploads: u64,
    /// P2 heat-pump attempts the ledger or the backend refused. Expected and
    /// harmless: the pump competes on the same budget a query would.
    pub prefetch_failures: u64,
    /// GPUDirect Storage `(direct, staged)` corpus uploads. `None` for a backend
    /// that keeps no such counters (the host backend). `direct` is `0` until a
    /// `cufile` binding is compiled in — see [`gpudirect`].
    pub gpudirect: Option<(u64, u64)>,
    pub last_error: Option<String>,
    pub resident: ResidentTelemetry,
}

impl AcceleratorTelemetry {
    /// Fraction of requests that actually ran on the device.
    pub fn device_ratio(&self) -> f64 {
        if self.requests == 0 {
            return 0.0;
        }
        self.executed_on_device as f64 / self.requests as f64
    }

    /// The §6 invariant: every request ends on exactly one execution surface.
    /// `fallbacks` and `cache_declined` are *reasons* for a host execution, not
    /// additional outcomes, so they are deliberately not part of the sum.
    pub fn accounting_holds(&self) -> bool {
        self.requests == self.executed_on_device + self.executed_on_host
            && self.fallbacks <= self.executed_on_host
            && self.cache_declined <= self.executed_on_host
    }
}

/// Decides where an operation runs, keeps the ledger, and never fails a query.
#[derive(Debug)]
pub struct AcceleratorRegistry {
    backend: Arc<dyn AccelBackend>,
    host: HostBackend,
    resident: RwLock<ResidentRegistry>,
    next_version: AtomicU64,
    requests: AtomicU64,
    device_execs: AtomicU64,
    host_execs: AtomicU64,
    fallbacks: AtomicU64,
    cache_declined: AtomicU64,
    prefetch_hints: AtomicU64,
    prefetch_uploads: AtomicU64,
    prefetch_failures: AtomicU64,
    /// Blocks per heat-pump pass; `0` disables the pump (`ACCEL_PREFETCH=OFF`).
    heat_pump_blocks: usize,
    last_error: RwLock<Option<String>>,
}

impl AcceleratorRegistry {
    /// A registry that always runs on the host. Used by tests and by callers
    /// that want the API without any acceleration.
    pub fn host_only() -> Self {
        Self::with_backend(
            Arc::new(HostBackend::new()),
            ResidentRegistry::new(
                AccelCache::Off,
                VramBudget::new(0, 0, 0.0),
                0,
                std::time::Duration::from_secs(0),
            ),
        )
    }

    /// Build from resolved parameters: budget from `CUDA_VRAM`, policy from
    /// `ACCEL_CACHE` / `ACCEL_SYNC_*`, and a device backend when one is both
    /// configured and reachable.
    pub fn from_resolved(params: &ResolvedParams) -> Self {
        let mut last_error = None;
        let backend: Arc<dyn AccelBackend> = if params.accel_cache.uses_device()
            && params.cuda_enabled
        {
            match open_device(
                params.cuda_device,
                params.accel_gpudirect,
                params.accel_gpudirect_threshold_bytes,
            ) {
                Ok(b) => b,
                Err(e) => {
                    last_error = Some(e.to_string());
                    Arc::new(HostBackend::new())
                }
            }
        } else {
            Arc::new(HostBackend::new())
        };
        // Elastic budget: the backend owns the driver context, so it is the
        // only thing that can report live free VRAM. Host backends return None
        // and leave `CUDA_VRAM` as a rigid ceiling.
        let mut resident = ResidentRegistry::from_resolved(params);
        if backend.mem_info().is_some() {
            let probe = Arc::clone(&backend);
            resident = resident.with_mem_probe(Arc::new(move || probe.mem_info()));
        }
        Self {
            backend,
            host: HostBackend::new(),
            resident: RwLock::new(resident),
            next_version: AtomicU64::new(1),
            requests: AtomicU64::new(0),
            device_execs: AtomicU64::new(0),
            host_execs: AtomicU64::new(0),
            fallbacks: AtomicU64::new(0),
            cache_declined: AtomicU64::new(0),
            prefetch_hints: AtomicU64::new(0),
            prefetch_uploads: AtomicU64::new(0),
            prefetch_failures: AtomicU64::new(0),
            heat_pump_blocks: if params.accel_prefetch {
                params.accel_prefetch_blocks
            } else {
                0
            },
            last_error: RwLock::new(last_error),
        }
    }

    /// Wire an explicit backend and ledger (tests, embedding, custom devices).
    pub fn with_backend(backend: Arc<dyn AccelBackend>, resident: ResidentRegistry) -> Self {
        Self {
            backend,
            host: HostBackend::new(),
            resident: RwLock::new(resident),
            next_version: AtomicU64::new(1),
            requests: AtomicU64::new(0),
            device_execs: AtomicU64::new(0),
            host_execs: AtomicU64::new(0),
            fallbacks: AtomicU64::new(0),
            cache_declined: AtomicU64::new(0),
            prefetch_hints: AtomicU64::new(0),
            prefetch_uploads: AtomicU64::new(0),
            prefetch_failures: AtomicU64::new(0),
            heat_pump_blocks: 0,
            last_error: RwLock::new(None),
        }
    }

    /// Enable the heat pump with an explicit block budget (the counterpart of
    /// `ACCEL_PREFETCH=ON` + `ACCEL_PREFETCH_BLOCKS`).
    pub fn with_heat_pump(mut self, blocks: usize) -> Self {
        self.heat_pump_blocks = blocks;
        self
    }

    /// Blocks one heat-pump pass may pull in; `0` means the pump is off.
    pub fn heat_pump_blocks(&self) -> usize {
        self.heat_pump_blocks
    }

    /// Whether `(kind, name)` has a device block right now.
    ///
    /// Used by the engine probe path to skip a page-warming hint for data that
    /// is already resident — a hint is only worth its syscall when the upload
    /// (or a host scan) is actually imminent.
    pub fn is_resident(&self, kind: ResidentKind, name: &str) -> bool {
        self.resident
            .read()
            .map(|r| r.is_resident(&ResidentKey::new(kind, name)))
            .unwrap_or(false)
    }

    /// The active device description.
    pub fn device(&self) -> &DeviceInfo {
        self.backend.device()
    }

    /// Whether a device backend is in use.
    pub fn is_device_backed(&self) -> bool {
        self.backend.device().kind != DeviceKind::Cpu
    }

    /// A fresh content version. Immutable datasets (a built index, a loaded
    /// series) take one at registration; re-registering takes another, which is
    /// what invalidates the device copy.
    pub fn next_version(&self) -> u64 {
        self.next_version.fetch_add(1, Ordering::Relaxed)
    }

    /// Admission bookkeeping shared by both operations.
    fn admit(
        &self,
        key: &ResidentKey,
        bytes: u64,
        rows: u64,
        version: u64,
    ) -> Option<Admission> {
        self.admit_inner(key, bytes, rows, version, false)
    }

    /// Admission on behalf of the idle heat pump: it may only fill VRAM that is
    /// free right now, and it consumes heat instead of adding to it. The policy
    /// lives in [`ResidentRegistry::admit_prefetch`].
    fn admit_for_prefetch(
        &self,
        key: &ResidentKey,
        bytes: u64,
        rows: u64,
        version: u64,
    ) -> Option<Admission> {
        self.admit_inner(key, bytes, rows, version, true)
    }

    fn admit_inner(
        &self,
        key: &ResidentKey,
        bytes: u64,
        rows: u64,
        version: u64,
        prefetch: bool,
    ) -> Option<Admission> {
        let mut guard = self.resident.write().ok()?;
        if !guard.mode().tracks_residency() {
            return None;
        }
        let admission = if prefetch {
            guard.admit_prefetch(key, bytes, rows, version)
        } else {
            guard.admit(key, bytes, rows, version)
        };
        // Keep the backend's buffers in step with the ledger.
        for evicted in guard.take_evictions() {
            self.backend.release(evicted.kind, &evicted.name);
        }
        Some(admission)
    }

    fn note_error(&self, err: AccelError) {
        self.fallbacks.fetch_add(1, Ordering::Relaxed);
        self.record_error(err);
    }

    /// Store the diagnosis without counting a query fallback.
    ///
    /// A failed heat-pump upload is not a query outcome; counting it as a
    /// fallback would break the §6 invariant that every request ends on exactly
    /// one execution surface.
    fn record_error(&self, err: AccelError) {
        if let Ok(mut slot) = self.last_error.write() {
            *slot = Some(err.to_string());
        }
    }

    /// Exact K-NN, device first, host on any refusal or failure.
    pub fn exact_knn(&self, req: &KnnRequest<'_>) -> Vec<RawHit> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let key = ResidentKey::new(ResidentKind::Corpus, req.name);
        let admission = self.admit(&key, req.resident_bytes(), req.rows(), req.version);

        if matches!(admission, Some(Admission::Declined { .. })) {
            self.cache_declined.fetch_add(1, Ordering::Relaxed);
        }

        // A device copy is only usable when the whole corpus is resident; a
        // split means some rows are still host-only, so the host kernel runs.
        let device_ready = self.is_device_backed()
            && matches!(admission, Some(Admission::Resident));
        if device_ready {
            match self.backend.exact_knn(req) {
                Ok(hits) => {
                    self.device_execs.fetch_add(1, Ordering::Relaxed);
                    return hits;
                }
                Err(e) => self.note_error(e),
            }
        }
        self.host_execs.fetch_add(1, Ordering::Relaxed);
        self.host.exact_knn(req).unwrap_or_default()
    }

    /// IVF probe, device first, host on any refusal or failure.
    ///
    /// The whole cell-ordered corpus is the resident unit (only the position
    /// list moves per query), so admission is identical to the flat scan's —
    /// except for `ResidentKind::IvfCorpus`, which keeps the two layouts from
    /// sharing a device buffer.
    pub fn ivf_probe(&self, req: &IvfRequest<'_>) -> Vec<RawHit> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let key = ResidentKey::new(ResidentKind::IvfCorpus, req.name);
        let admission = self.admit(&key, req.resident_bytes(), req.rows(), req.version);

        if matches!(admission, Some(Admission::Declined { .. })) {
            self.cache_declined.fetch_add(1, Ordering::Relaxed);
        }
        let device_ready = self.is_device_backed()
            && matches!(admission, Some(Admission::Resident));
        if device_ready {
            match self.backend.ivf_probe(req) {
                Ok(hits) => {
                    self.device_execs.fetch_add(1, Ordering::Relaxed);
                    return hits;
                }
                Err(e) => self.note_error(e),
            }
        }
        self.host_execs.fetch_add(1, Ordering::Relaxed);
        self.host.ivf_probe(req).unwrap_or_default()
    }

    /// As-of join, device first, host on any refusal or failure.
    pub fn asof_join(&self, req: &AsofRequest<'_>) -> Vec<Option<u32>> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let key = ResidentKey::new(ResidentKind::RightSeries, req.name);
        let admission = self.admit(&key, req.resident_bytes(), req.rows(), req.version);

        if matches!(admission, Some(Admission::Declined { .. })) {
            self.cache_declined.fetch_add(1, Ordering::Relaxed);
        }

        let device_ready = self.is_device_backed()
            && matches!(admission, Some(Admission::Resident));
        if device_ready {
            match self.backend.asof_join(req) {
                Ok(idx) => {
                    self.device_execs.fetch_add(1, Ordering::Relaxed);
                    return idx;
                }
                Err(e) => self.note_error(e),
            }
        }
        self.host_execs.fetch_add(1, Ordering::Relaxed);
        self.host.asof_join(req).unwrap_or_default()
    }

    /// Register appended rows that are not on the device yet.
    pub fn note_append(&self, kind: ResidentKind, name: &str, rows: u64, bytes: u64) -> SyncPolicy {
        match self.resident.write() {
            Ok(mut r) => r.note_append(&ResidentKey::new(kind, name), rows, bytes),
            Err(_) => SyncPolicy::Deferred,
        }
    }

    /// Apply pending delta in one batched upload. Call at a batch boundary, not
    /// from a query path.
    pub fn flush_pending(&self) -> (u64, u64) {
        match self.resident.write() {
            Ok(mut r) => r.flush_pending(),
            Err(_) => (0, 0),
        }
    }

    /// Non-resident keys worth pulling in proactively, hottest first.
    ///
    /// This reads the shadow ledger, which only ever holds keys a query was
    /// *refused*: the list is therefore "what the workload keeps asking for and
    /// does not have". `min_heat` (use [`PREFETCH_HEAT_FLOOR`]) keeps one-shot
    /// scans out, and `limit` is the caller's bandwidth budget.
    pub fn prefetch_candidates(&self, min_heat: f64, limit: usize) -> Vec<ResidentKey> {
        self.resident
            .read()
            .map(|r| {
                r.hottest_shadows(min_heat, limit)
                    .into_iter()
                    .map(|(k, _)| k)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Drop a key's shadow evidence — the dataset was dropped, so no pump should
    /// ever try to upload it again.
    pub fn forget_candidate(&self, kind: ResidentKind, name: &str) {
        if let Ok(mut r) = self.resident.write() {
            r.forget_shadow(&ResidentKey::new(kind, name));
        }
    }

    /// Hand a dataset's device copy back to the budget, shadow entry included.
    ///
    /// This is the heat pump's *reclaim* step: a collection dropped from the
    /// engine map can never be a hit again, so keeping its VRAM only shrinks the
    /// working set. Returns the bytes freed (`0` when nothing was resident), and
    /// is a no-op on a host-only registry.
    pub fn release(&self, kind: ResidentKind, name: &str) -> u64 {
        self.resident
            .write()
            .map(|mut r| r.release(&ResidentKey::new(kind, name)))
            .unwrap_or(0)
    }

    /// `(kind, name)` of every resident block, for the pump's reclaim sweep.
    pub fn resident_blocks(&self) -> Vec<(ResidentKind, String)> {
        self.resident
            .read()
            .map(|r| r.blocks().into_iter().map(|b| (b.kind, b.name)).collect())
            .unwrap_or_default()
    }

    /// P2 heat pump: upload a hot non-resident corpus *before* the query that
    /// wants it, so that query's first use is a resident hit.
    ///
    /// The key goes through [`ResidentRegistry::admit_prefetch`], not the query
    /// path: the pump may only take VRAM that is free *right now*, so it can
    /// neither evict a block a query is using nor inflate a corpus's heat. A
    /// refusal is normal and expected — the pump exists to convert idle slack
    /// (typically the region a dropped or shrunk collection just released) into
    /// a warm block; it is counted in `prefetch_failures` and never fails a
    /// query. See [`Self::release`] for the reclaim half of that loop.
    ///
    /// Returns `true` when the device now holds the corpus.
    pub fn prefetch_upload(
        &self,
        kind: ResidentKind,
        name: &str,
        ids: &[u64],
        data: &[f32],
        dim: usize,
        version: u64,
    ) -> bool {
        if !self.is_device_backed() || dim == 0 || data.len() != ids.len() * dim {
            return false;
        }
        let req = PrefetchRequest {
            kind,
            name,
            version,
            ids,
            data,
            dim,
        };
        let key = ResidentKey::new(kind, name);
        let admitted = matches!(
            self.admit_for_prefetch(&key, req.resident_bytes(), req.rows(), version),
            Some(Admission::Resident)
        );
        if !admitted {
            self.prefetch_failures.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        match self.backend.prefetch_corpus(&req) {
            Ok(true) => {
                self.prefetch_uploads.fetch_add(1, Ordering::Relaxed);
                true
            }
            Ok(false) => {
                self.prefetch_failures.fetch_add(1, Ordering::Relaxed);
                false
            }
            Err(e) => {
                self.record_error(e);
                self.prefetch_failures.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// P1c 预取预告: warm `host`'s pages so an imminent upload or scan does not
    /// pay page faults on its critical path.
    ///
    /// `kind`/`name` identify the dataset the range belongs to — the P2
    /// heat-driven prefetch correlates these hints with later admissions — but
    /// the warming itself acts on the host range because only the caller holds
    /// the pointer (`storage-tiering-ssd-ram-vram.md` §4.2 row 1: "数据仍在 RAM
    /// 层预热，不占 VRAM"). The call never fails and never waits on I/O
    /// completion: every backend path must also warm at upload time on its own
    /// (`CudaBackend::build_corpus` does), so a missed hint costs latency,
    /// never correctness.
    pub fn prefetch_hint(&self, kind: ResidentKind, name: &str, host: &[f32]) {
        let _ = (kind, name);
        self.prefetch_hints.fetch_add(1, Ordering::Relaxed);
        warm::warm_slice(host);
    }

    /// Per-block view of the resident cache.
    pub fn blocks(&self) -> Vec<BlockView> {
        self.resident
            .read()
            .map(|r| r.blocks())
            .unwrap_or_default()
    }

    /// Device, cache and fallback counters.
    pub fn telemetry(&self) -> AcceleratorTelemetry {
        let resident = self
            .resident
            .read()
            .map(|r| r.telemetry())
            .unwrap_or_default();
        AcceleratorTelemetry {
            device: self.backend.device().clone(),
            mode: self
                .resident
                .read()
                .map(|r| r.mode())
                .unwrap_or(AccelCache::Off),
            requests: self.requests.load(Ordering::Relaxed),
            executed_on_device: self.device_execs.load(Ordering::Relaxed),
            executed_on_host: self.host_execs.load(Ordering::Relaxed),
            fallbacks: self.fallbacks.load(Ordering::Relaxed),
            cache_declined: self.cache_declined.load(Ordering::Relaxed),
            prefetch_hints: self.prefetch_hints.load(Ordering::Relaxed),
            prefetch_uploads: self.prefetch_uploads.load(Ordering::Relaxed),
            prefetch_failures: self.prefetch_failures.load(Ordering::Relaxed),
            gpudirect: self.backend.direct_upload_stats(),
            last_error: self.last_error.read().ok().and_then(|e| e.clone()),
            resident,
        }
    }
}

impl Default for AcceleratorRegistry {
    fn default() -> Self {
        Self::host_only()
    }
}

#[cfg(feature = "cuda")]
fn open_device(
    ordinal: usize,
    gpudirect: bool,
    gpudirect_threshold: u64,
) -> Result<Arc<dyn AccelBackend>, AccelError> {
    cuda::CudaBackend::probe(ordinal).map(|b| {
        Arc::new(b.with_gpudirect(gpudirect, gpudirect_threshold)) as Arc<dyn AccelBackend>
    })
}

#[cfg(not(feature = "cuda"))]
fn open_device(
    _ordinal: usize,
    _gpudirect: bool,
    _gpudirect_threshold: u64,
) -> Result<Arc<dyn AccelBackend>, AccelError> {
    Err(AccelError::NoDevice(
        "built without the `cuda` feature".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gtv_core::Metric;
    use std::time::Duration;

    /// A "device" that forwards to the host kernels so results stay correct,
    /// while recording calls — enough to exercise routing without hardware.
    #[derive(Debug)]
    struct FakeDevice {
        info: DeviceInfo,
        calls: AtomicU64,
        prefetches: AtomicU64,
        released: RwLock<Vec<String>>,
    }

    impl FakeDevice {
        fn new() -> Self {
            Self {
                info: DeviceInfo {
                    name: "fake-gpu".to_string(),
                    kind: DeviceKind::Cuda,
                    total_bytes: 1 << 30,
                },
                calls: AtomicU64::new(0),
                prefetches: AtomicU64::new(0),
                released: RwLock::new(Vec::new()),
            }
        }
        fn calls(&self) -> u64 {
            self.calls.load(Ordering::Relaxed)
        }
        fn prefetches(&self) -> u64 {
            self.prefetches.load(Ordering::Relaxed)
        }
        fn released(&self) -> Vec<String> {
            self.released.read().map(|v| v.clone()).unwrap_or_default()
        }
    }

    impl AccelBackend for FakeDevice {
        fn device(&self) -> &DeviceInfo {
            &self.info
        }
        fn supports(&self, _op: AccelOp) -> bool {
            true
        }
        fn exact_knn(&self, req: &KnnRequest<'_>) -> Result<Vec<RawHit>, AccelError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            HostBackend::new().exact_knn(req)
        }
        fn ivf_probe(&self, req: &IvfRequest<'_>) -> Result<Vec<RawHit>, AccelError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            HostBackend::new().ivf_probe(req)
        }
        fn asof_join(&self, req: &AsofRequest<'_>) -> Result<Vec<Option<u32>>, AccelError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            HostBackend::new().asof_join(req)
        }
        fn release(&self, _kind: ResidentKind, name: &str) {
            if let Ok(mut v) = self.released.write() {
                v.push(name.to_string());
            }
        }
        fn prefetch_corpus(&self, req: &PrefetchRequest<'_>) -> Result<bool, AccelError> {
            self.prefetches.fetch_add(1, Ordering::Relaxed);
            Ok(matches!(
                req.kind,
                ResidentKind::Corpus | ResidentKind::IvfCorpus
            ))
        }
    }

    /// A backend whose kernel always fails, to exercise the fallback.
    #[derive(Debug)]
    struct BrokenDevice {
        info: DeviceInfo,
    }

    impl BrokenDevice {
        fn new() -> Self {
            Self {
                info: DeviceInfo {
                    name: "broken".to_string(),
                    kind: DeviceKind::Cuda,
                    total_bytes: 1 << 30,
                },
            }
        }
    }

    impl AccelBackend for BrokenDevice {
        fn device(&self) -> &DeviceInfo {
            &self.info
        }
        fn supports(&self, _op: AccelOp) -> bool {
            true
        }
        fn exact_knn(&self, _req: &KnnRequest<'_>) -> Result<Vec<RawHit>, AccelError> {
            Err(AccelError::Launch("injected failure".into()))
        }
        fn ivf_probe(&self, _req: &IvfRequest<'_>) -> Result<Vec<RawHit>, AccelError> {
            Err(AccelError::Launch("injected failure".into()))
        }
        fn asof_join(&self, _req: &AsofRequest<'_>) -> Result<Vec<Option<u32>>, AccelError> {
            Err(AccelError::Launch("injected failure".into()))
        }
    }

    fn ledger(capacity: u64) -> ResidentRegistry {
        ResidentRegistry::new(
            AccelCache::On,
            VramBudget::new(capacity, capacity, 0.0),
            0,
            Duration::from_secs(5),
        )
    }

    fn corpus<'a>(
        ids: &'a [u64],
        data: &'a [f32],
        query: &'a [f32],
        version: u64,
    ) -> KnnRequest<'a> {
        KnnRequest {
            name: "songs",
            version,
            ids,
            data,
            dim: 2,
            metric: Metric::L2,
            normalized: false,
            query,
            k: 3,
            mask: None,
        }
    }

    const IDS: [u64; 3] = [0, 1, 2];
    const DATA: [f32; 6] = [0.0, 0.0, 1.0, 1.0, 5.0, 5.0];

    #[test]
    fn device_backend_is_used_when_the_block_fits() {
        let fake = Arc::new(FakeDevice::new());
        let reg = AcceleratorRegistry::with_backend(fake.clone(), ledger(1 << 20));
        let hits = reg.exact_knn(&corpus(&IDS, &DATA, &[0.1, 0.1], 1));
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(fake.calls(), 1);
        let t = reg.telemetry();
        assert_eq!(t.executed_on_device, 1);
        assert_eq!(t.fallbacks, 0);
        assert!((t.device_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_repeated_query_is_a_resident_hit() {
        let fake = Arc::new(FakeDevice::new());
        let reg = AcceleratorRegistry::with_backend(fake.clone(), ledger(1 << 20));
        reg.exact_knn(&corpus(&IDS, &DATA, &[0.1, 0.1], 1));
        reg.exact_knn(&corpus(&IDS, &DATA, &[0.1, 0.1], 1));
        let t = reg.telemetry();
        assert_eq!(fake.calls(), 2);
        assert_eq!(t.resident.hits, 1);
        assert_eq!(t.resident.misses, 1);
    }

    #[test]
    fn a_new_version_is_a_miss() {
        let fake = Arc::new(FakeDevice::new());
        let reg = AcceleratorRegistry::with_backend(fake.clone(), ledger(1 << 20));
        let v1 = reg.next_version();
        let v2 = reg.next_version();
        reg.exact_knn(&corpus(&IDS, &DATA, &[0.1, 0.1], v1));
        reg.exact_knn(&corpus(&IDS, &DATA, &[0.1, 0.1], v2));
        assert_eq!(reg.telemetry().resident.misses, 2);
    }

    #[test]
    fn a_broken_kernel_falls_back_and_records_why() {
        let broken = Arc::new(BrokenDevice::new());
        let reg = AcceleratorRegistry::with_backend(broken, ledger(1 << 20));
        let hits = reg.exact_knn(&corpus(&IDS, &DATA, &[0.1, 0.1], 1));
        assert_eq!(
            hits.iter().map(|h| h.id).collect::<Vec<_>>(),
            vec![0, 1, 2],
            "the query must still succeed on the host"
        );
        let t = reg.telemetry();
        assert_eq!(t.fallbacks, 1);
        assert_eq!(t.executed_on_device, 0);
        assert!(t.last_error.unwrap().contains("injected failure"));
    }

    #[test]
    fn a_corpus_that_does_not_fit_runs_on_the_host() {
        let fake = Arc::new(FakeDevice::new());
        let reg = AcceleratorRegistry::with_backend(fake.clone(), ledger(16));
        let hits = reg.exact_knn(&corpus(&IDS, &DATA, &[0.1, 0.1], 1));
        assert_eq!(hits.len(), 3);
        assert_eq!(fake.calls(), 0, "declined means no device call");
        assert_eq!(reg.telemetry().cache_declined, 1);
    }

    #[test]
    fn eviction_is_propagated_to_the_backend() {
        let fake = Arc::new(FakeDevice::new());
        // Room for one corpus only (3 ids * 8 + 6 floats * 4 = 48 bytes).
        let reg = AcceleratorRegistry::with_backend(fake.clone(), ledger(60));
        let other = KnnRequest {
            name: "other",
            version: 1,
            ids: &IDS,
            data: &DATA,
            dim: 2,
            metric: Metric::L2,
            normalized: false,
            query: &[0.0, 0.0],
            k: 1,
            mask: None,
        };
        reg.exact_knn(&corpus(&IDS, &DATA, &[0.1, 0.1], 1));
        reg.exact_knn(&other);
        assert_eq!(fake.released(), vec!["songs".to_string()]);
    }

    #[test]
    fn ivf_probe_routes_to_the_device_and_keeps_its_own_resident_kind() {
        let fake = Arc::new(FakeDevice::new());
        let reg = AcceleratorRegistry::with_backend(fake.clone(), ledger(1 << 20));
        let ids = [10u64, 11, 12];
        let data = [0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0];
        let req = IvfRequest {
            name: "songs",
            version: 1,
            ids: &ids,
            data: &data,
            dim: 2,
            metric: Metric::L2,
            normalized: false,
            query: &[2.0, 2.0],
            k: 2,
            positions: &[0, 1, 2],
        };
        let hits = reg.ivf_probe(&req);
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![12, 11]);
        assert_eq!(fake.calls(), 1);
        let t = reg.telemetry();
        assert_eq!(t.executed_on_device, 1);
        assert!(t.accounting_holds());
        assert_eq!(reg.blocks()[0].kind, ResidentKind::IvfCorpus);
    }

    #[test]
    fn asof_round_trips_through_the_registry() {
        let reg = AcceleratorRegistry::host_only();
        let right = [0i64, 10, 20];
        let got = reg.asof_join(&AsofRequest {
            name: "quotes",
            version: 1,
            left: &[5, 10, 25],
            right_times: &right,
            payload_bytes: 3 * 8 * 2,
            tolerance_ns: None,
        });
        assert_eq!(got, vec![Some(0), Some(1), Some(2)]);
        assert!(reg.telemetry().resident.blocks == 0, "host-only keeps no ledger entry");
    }

    #[test]
    fn appends_are_deferred_then_flushed() {
        let fake = Arc::new(FakeDevice::new());
        let reg = AcceleratorRegistry::with_backend(fake, ledger(1 << 20));
        reg.exact_knn(&corpus(&IDS, &DATA, &[0.1, 0.1], 1));
        assert_eq!(
            reg.note_append(ResidentKind::Corpus, "songs", 10, 40),
            SyncPolicy::Deferred
        );
        let (blocks, bytes) = reg.flush_pending();
        assert_eq!((blocks, bytes), (1, 40));
        let t = reg.telemetry();
        assert_eq!(t.resident.pending_rows, 0);
        assert!(t.resident.upload_bytes >= 40);
    }

    #[test]
    fn the_heat_pump_fills_vram_a_dropped_collection_released() {
        let fake = Arc::new(FakeDevice::new());
        // Room for exactly one 48-byte corpus.
        let reg = AcceleratorRegistry::with_backend(fake.clone(), ledger(60));
        let corpus_req = corpus(&IDS, &DATA, &[0.1, 0.1], 1);
        let other = KnnRequest {
            name: "other",
            version: 1,
            ids: &IDS,
            data: &DATA,
            dim: 2,
            metric: Metric::L2,
            normalized: false,
            query: &[0.0, 0.0],
            k: 1,
            mask: None,
        };

        // Five references put "songs" at heat 5.0, above PROTECT_FLOOR.
        reg.exact_knn(&corpus_req);
        for _ in 0..4 {
            reg.exact_knn(&corpus_req);
        }
        // Six refusals leave "other" at shadow heat 6.0: above the pump floor,
        // still below the 1.25x the *query* path requires (6.25), so every one
        // of those queries ran on the host and paid its own upload.
        for _ in 0..6 {
            reg.exact_knn(&other);
        }
        let t = reg.telemetry();
        assert_eq!(t.executed_on_host, 6);
        assert_eq!(t.cache_declined, 6);

        let candidates = reg.prefetch_candidates(PREFETCH_HEAT_FLOOR, 4);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].name, "other");
        assert_eq!(candidates[0].kind, ResidentKind::Corpus);

        // A saturated cache has no slack, and the pump does not create any: it
        // declines rather than evict the block the queries are using.
        assert!(!reg.prefetch_upload(ResidentKind::Corpus, "other", &IDS, &DATA, 2, 1));
        assert_eq!(fake.prefetches(), 0, "no slack, no upload");
        assert!(reg.is_resident(ResidentKind::Corpus, "songs"));

        // The collection behind the resident block is dropped: its device copy
        // can never be a hit again, so the registry hands the bytes back. This
        // is where a slack-only pump gets its room.
        assert_eq!(
            reg.release(ResidentKind::Corpus, "songs"),
            other.resident_bytes(),
            "the dropped collection's whole device copy comes back"
        );
        assert!(!reg.is_resident(ResidentKind::Corpus, "songs"));

        // With nothing waiting, the released region is filled by the corpus the
        // workload kept asking for, so the next transfer is off the critical
        // path instead of on it.
        assert!(reg.prefetch_upload(ResidentKind::Corpus, "other", &IDS, &DATA, 2, 1));
        assert_eq!(fake.prefetches(), 1);

        let before = fake.calls();
        let hits = reg.exact_knn(&other);
        assert_eq!(hits.len(), 1, "k=1");
        assert_eq!(fake.calls(), before + 1, "the pumped query runs on the device");

        let t = reg.telemetry();
        assert_eq!(t.prefetch_uploads, 1);
        assert_eq!(t.prefetch_failures, 1, "the saturated pass declined once");
        assert!(t.accounting_holds(), "a pump must not disturb query accounting");
    }

    #[test]
    fn a_saturated_cache_makes_the_pump_decline_without_evicting() {
        let fake = Arc::new(FakeDevice::new());
        let reg = AcceleratorRegistry::with_backend(fake.clone(), ledger(60));
        let cold = KnnRequest {
            name: "cold",
            version: 1,
            ids: &IDS,
            data: &DATA,
            dim: 2,
            metric: Metric::L2,
            normalized: false,
            query: &[0.0, 0.0],
            k: 1,
            mask: None,
        };
        // "songs" owns the whole budget and its heat protects it.
        reg.exact_knn(&corpus(&IDS, &DATA, &[0.1, 0.1], 1));
        for _ in 0..4 {
            reg.exact_knn(&corpus(&IDS, &DATA, &[0.1, 0.1], 1));
        }
        reg.exact_knn(&cold); // one refusal: shadow heat 1.0

        // The query path is allowed to evict here; the pump is not. It declines
        // with `no-slack` and leaves the resident set exactly as it found it.
        assert!(!reg.prefetch_upload(ResidentKind::Corpus, "cold", &IDS, &DATA, 2, 1));
        assert_eq!(fake.prefetches(), 0, "a pump never displaces a resident block");
        assert_eq!(reg.resident_blocks().len(), 1);
        assert!(reg.is_resident(ResidentKind::Corpus, "songs"));
        let t = reg.telemetry();
        assert_eq!(t.prefetch_failures, 1);
        assert_eq!(t.prefetch_uploads, 0);
        assert_eq!(t.resident.evictions, 0, "nothing was thrown out to make room");
    }

    #[test]
    fn a_refused_pump_is_counted_but_never_fatal() {
        let fake = Arc::new(FakeDevice::new());
        // Too small for the corpus, so admission declines every time.
        let reg = AcceleratorRegistry::with_backend(fake.clone(), ledger(16));
        assert!(!reg.prefetch_upload(ResidentKind::Corpus, "songs", &IDS, &DATA, 2, 1));
        assert_eq!(fake.prefetches(), 0, "a declined pump never reaches the device");
        let t = reg.telemetry();
        assert_eq!(t.prefetch_failures, 1);
        assert_eq!(t.prefetch_uploads, 0);
        assert_eq!(t.requests, 0, "prefetching is not a query");
        assert!(t.accounting_holds());
    }

    #[test]
    fn prefetch_is_a_no_op_without_a_device() {
        let reg = AcceleratorRegistry::host_only();
        assert!(!reg.prefetch_upload(ResidentKind::Corpus, "songs", &IDS, &DATA, 2, 1));
        assert!(reg.prefetch_candidates(PREFETCH_HEAT_FLOOR, 4).is_empty());
        let t = reg.telemetry();
        assert_eq!(t.prefetch_uploads, 0);
        assert_eq!(t.prefetch_failures, 0);
    }

    #[test]
    fn device_total_is_none_without_the_cuda_feature() {
        // Guards the GPU-less contract: parameter resolution must not fabricate
        // a VRAM figure.
        #[cfg(not(feature = "cuda"))]
        {
            assert_eq!(device_total_bytes(0), None);
            assert_eq!(device_count(), 0);
        }
    }
}
