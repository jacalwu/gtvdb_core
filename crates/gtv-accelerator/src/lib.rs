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
pub mod host;
pub mod resident;
pub mod sweep;

#[cfg(feature = "cuda")]
pub mod cuda;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

pub use gtv_params::{AccelCache, ResolvedParams};

pub use backend::{
    AccelBackend, AccelError, AccelOp, AsofRequest, DeviceInfo, DeviceKind, IvfRequest, KnnRequest,
    RawHit,
};
pub use budget::{VramBudget, DEFAULT_HEADROOM};
pub use host::HostBackend;
pub use resident::{
    Admission, BlockView, DeclineReason, ResidentKey, ResidentKind, ResidentRegistry,
    ResidentTelemetry, SyncPolicy, ADMIT_MARGIN, DEFAULT_HEAT_HALF_LIFE, PROTECT_FLOOR,
};
pub use sweep::asof_match;

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
            match open_device(params.cuda_device) {
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
            last_error: RwLock::new(None),
        }
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
        let mut guard = self.resident.write().ok()?;
        if !guard.mode().tracks_residency() {
            return None;
        }
        let admission = guard.admit(key, bytes, rows, version);
        // Keep the backend's buffers in step with the ledger.
        for evicted in guard.take_evictions() {
            self.backend.release(evicted.kind, &evicted.name);
        }
        Some(admission)
    }

    fn note_error(&self, err: AccelError) {
        self.fallbacks.fetch_add(1, Ordering::Relaxed);
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
fn open_device(ordinal: usize) -> Result<Arc<dyn AccelBackend>, AccelError> {
    cuda::CudaBackend::probe(ordinal).map(|b| Arc::new(b) as Arc<dyn AccelBackend>)
}

#[cfg(not(feature = "cuda"))]
fn open_device(_ordinal: usize) -> Result<Arc<dyn AccelBackend>, AccelError> {
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
                released: RwLock::new(Vec::new()),
            }
        }
        fn calls(&self) -> u64 {
            self.calls.load(Ordering::Relaxed)
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
