//! Device-memory ledger.
//!
//! Every byte a backend is allowed to hold on the device is reserved here
//! first. Two things make this worth a type of its own:
//!
//! * **One budget, many blocks.** The resident cache, a query's staging buffer
//!   and a kernel's scratch space all draw from the same VRAM figure, so the
//!   accounting has to be central or blocks will over-commit independently.
//! * **Headroom.** The configured figure (`CUDA_VRAM`, default 80% of device
//!   VRAM) is what gtVdB may *use*, not what the driver has. Keeping a slice
//!   back means a kernel launch that needs a temporary buffer does not fail
//!   with `cudaErrorMemoryAllocation` right after a cache fill reported "full".
//! * **Elasticity.** `CUDA_VRAM` is a *ceiling*, not a promise about how much
//!   memory is actually free. The device is shared (WDDM on Windows, other
//!   processes, driver bookkeeping), and a driver that quietly spills a
//!   "successful" allocation into host-backed sysmem does not fail the launch
//!   but destroys bandwidth. When a live memory probe is available, the ledger
//!   lowers its capacity to what the driver says is free, so the affected query
//!   declines to the host instead of timing out inside a kernel.

use std::sync::Arc;
use std::time::{Duration, Instant};

/// Fraction of the configured VRAM figure held back from the resident cache.
///
/// 10% of an 80%-of-device budget leaves ~72% of VRAM for resident data, which
/// keeps kernel launch scratch and driver bookkeeping inside the device.
pub const DEFAULT_HEADROOM: f64 = 0.10;

/// Minimum interval between live device-memory probes. A probe is a driver
/// round-trip; doing one per admission would be measurable on small queries.
pub const PROBE_INTERVAL: Duration = Duration::from_millis(250);

/// A live `(free, total)` device-memory sample, in bytes. `None` means the
/// device could not be queried right now (treat as "no new information").
pub type MemProbe = Arc<dyn Fn() -> Option<(u64, u64)> + Send + Sync>;

/// Byte accounting for a single device (or for the shadow ledger on a host).
#[derive(Clone)]
pub struct VramBudget {
    device_total: u64,
    /// The `CUDA_VRAM` ceiling. Never raised by a probe.
    configured_capacity: u64,
    used: u64,
    peak: u64,
    declines: u64,
    decline_bytes: u64,
    headroom: f64,
    probe: Option<MemProbe>,
    /// `min(configured_capacity, elastic_ceiling)`. `None` until the first
    /// successful probe, meaning "trust the configuration".
    elastic_ceiling: Option<u64>,
    last_probe: Option<Instant>,
    /// Live free bytes from the most recent probe, after headroom.
    probed_free: Option<u64>,
    /// Number of probes that actually tightened the ceiling.
    clamps: u64,
}

impl std::fmt::Debug for VramBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VramBudget")
            .field("device_total", &self.device_total)
            .field("configured_capacity", &self.configured_capacity)
            .field("capacity", &self.capacity())
            .field("used", &self.used)
            .field("probe_attached", &self.probe.is_some())
            .field("elastic_ceiling", &self.elastic_ceiling)
            .finish()
    }
}

impl VramBudget {
    /// Build from an explicit device total and a usable figure.
    ///
    /// `capacity` is what callers may reserve; `headroom` is shaved off it.
    /// A zero total or zero usable figure yields a zero-capacity budget, which
    /// every admission path treats as "use the host".
    pub fn new(device_total: u64, usable: u64, headroom: f64) -> Self {
        let headroom = headroom.clamp(0.0, 0.9);
        let keep = (usable as f64 * headroom) as u64;
        Self {
            device_total,
            configured_capacity: usable.saturating_sub(keep),
            used: 0,
            peak: 0,
            declines: 0,
            decline_bytes: 0,
            headroom,
            probe: None,
            elastic_ceiling: None,
            last_probe: None,
            probed_free: None,
            clamps: 0,
        }
    }

    /// Attach a live memory probe. Without one the budget stays rigid and
    /// behaves exactly as before (which is what GPU-less and shadow modes want).
    pub fn with_probe(mut self, probe: MemProbe) -> Self {
        self.probe = Some(probe);
        self
    }

    /// Whether a live probe is attached.
    pub fn is_elastic(&self) -> bool {
        self.probe.is_some()
    }

    /// Live free bytes from the last probe, after headroom (if any).
    pub fn probed_free_bytes(&self) -> Option<u64> {
        self.probed_free
    }

    /// How many probes lowered the ceiling below the configured figure.
    pub fn elastic_clamps(&self) -> u64 {
        self.clamps
    }

    /// Re-read device memory, at most once per [`PROBE_INTERVAL`].
    ///
    /// Returns whether a fresh sample was taken. A failed probe is *not* an
    /// error: it leaves the previous ceiling in place rather than disabling
    /// acceleration because one `cudaMemGetInfo` call did not succeed.
    pub fn refresh_if_due(&mut self) -> bool {
        if self.probe.is_none() {
            return false;
        }
        let due = self
            .last_probe
            .map(|t| t.elapsed() >= PROBE_INTERVAL)
            .unwrap_or(true);
        if due {
            self.refresh()
        } else {
            false
        }
    }

    /// Force a fresh sample (tests, explicit preflight, `.accel`).
    pub fn refresh(&mut self) -> bool {
        let Some(probe) = self.probe.clone() else {
            return false;
        };
        self.last_probe = Some(Instant::now());
        let Some((free, total)) = probe() else {
            return false;
        };
        if total > 0 {
            self.device_total = total;
        }
        let usable = (free as f64 * (1.0 - self.headroom)) as u64;
        self.probed_free = Some(usable);
        // What the whole cache could hold *right now*: what the driver says is
        // free plus what we are already holding. Storing the absolute ceiling
        // (rather than recomputing it from `used` on every read) is what makes
        // `release()` still increase `free()` before the next probe.
        let ceiling = usable.saturating_add(self.used);
        if ceiling < self.configured_capacity {
            self.clamps = self.clamps.saturating_add(1);
        }
        self.elastic_ceiling = Some(ceiling);
        true
    }

    /// Derive the budget from resolved parameters.
    ///
    /// `ACCEL_CACHE=OFF` yields a zero-capacity budget so nothing is ever
    /// admitted, even if a VRAM figure happens to be present.
    pub fn from_resolved(params: &gtv_params::ResolvedParams) -> Self {
        if !params.accel_cache.tracks_residency() {
            return Self::new(params.cuda_vram_bytes, 0, DEFAULT_HEADROOM);
        }
        Self::new(
            params.cuda_vram_bytes,
            params.cuda_vram_bytes,
            DEFAULT_HEADROOM,
        )
    }

    /// Reported device total in bytes (0 when unknown / host-only).
    pub fn device_total(&self) -> u64 {
        self.device_total
    }

    /// Bytes available for reservation after headroom **and** after the live
    /// device probe. `CUDA_VRAM` is a ceiling; `min(ceiling, what is free now)`
    /// is the capacity a query may actually commit to.
    pub fn capacity(&self) -> u64 {
        self.elastic_ceiling
            .map(|c| c.min(self.configured_capacity))
            .unwrap_or(self.configured_capacity)
    }

    /// The `CUDA_VRAM` figure, before any live probe tightened it.
    pub fn configured_capacity(&self) -> u64 {
        self.configured_capacity
    }

    /// Bytes currently reserved.
    pub fn used(&self) -> u64 {
        self.used
    }

    /// High-water mark since construction.
    pub fn peak(&self) -> u64 {
        self.peak
    }

    /// Unreserved capacity (`capacity - used`).
    pub fn free(&self) -> u64 {
        self.capacity().saturating_sub(self.used)
    }

    /// Admission requests rejected for lack of space.
    pub fn declines(&self) -> u64 {
        self.declines
    }

    /// Total bytes those rejections asked for.
    pub fn decline_bytes(&self) -> u64 {
        self.decline_bytes
    }

    /// Fraction of capacity in use, in `0.0..=1.0`.
    pub fn utilization(&self) -> f64 {
        let capacity = self.capacity();
        if capacity == 0 {
            return 0.0;
        }
        (self.used as f64 / capacity as f64).clamp(0.0, 1.0)
    }

    /// Reserve `bytes`, or refuse and record the shortfall.
    ///
    /// Refusal is not an error: the caller falls back to the host path. That is
    /// the deliberate difference from a static-capacity cache that corrupts
    /// itself once it overflows.
    pub fn try_reserve(&mut self, bytes: u64) -> bool {
        if bytes == 0 {
            return true;
        }
        if bytes > self.free() {
            self.declines = self.declines.saturating_add(1);
            self.decline_bytes = self.decline_bytes.saturating_add(bytes);
            return false;
        }
        self.used = self.used.saturating_add(bytes);
        self.peak = self.peak.max(self.used);
        true
    }

    /// Record a refusal that was decided above this layer (e.g. eviction could
    /// not free enough room), so the decline counters stay truthful.
    pub fn record_decline(&mut self, bytes: u64) {
        self.declines = self.declines.saturating_add(1);
        self.decline_bytes = self.decline_bytes.saturating_add(bytes);
    }

    /// Return `bytes` to the budget. Returns whether anything was accounted.
    pub fn release(&mut self, bytes: u64) -> bool {
        if bytes == 0 || self.used == 0 {
            return false;
        }
        let taken = bytes.min(self.used);
        self.used -= taken;
        taken == bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headroom_is_shaved_off_the_usable_figure() {
        let b = VramBudget::new(8 << 30, 8 << 30, 0.10);
        let keep = ((8u64 << 30) as f64 * 0.10) as u64;
        assert_eq!(b.capacity(), (8u64 << 30) - keep);
        assert_eq!(b.free(), b.capacity());
        // Sanity: within a byte of 90%.
        assert!(b.capacity() as f64 >= (8u64 << 30) as f64 * 0.9 - 1.0);
    }

    #[test]
    fn reserve_and_release_track_peak() {
        let mut b = VramBudget::new(1000, 1000, 0.0);
        assert!(b.try_reserve(400));
        assert!(b.try_reserve(300));
        assert_eq!(b.used(), 700);
        assert_eq!(b.peak(), 700);
        assert!(b.release(300));
        assert_eq!(b.used(), 400);
        assert_eq!(b.peak(), 700);
    }

    #[test]
    fn over_reservation_is_refused_and_counted() {
        let mut b = VramBudget::new(100, 100, 0.0);
        assert!(!b.try_reserve(150));
        assert_eq!(b.used(), 0);
        assert_eq!(b.declines(), 1);
        assert_eq!(b.decline_bytes(), 150);
    }

    #[test]
    fn release_never_underflows() {
        let mut b = VramBudget::new(100, 100, 0.0);
        assert!(!b.release(10));
        b.try_reserve(10);
        assert!(!b.release(11));
        assert_eq!(b.used(), 0);
    }

    #[test]
    fn zero_capacity_accepts_nothing_but_zero() {
        let mut b = VramBudget::new(0, 0, 0.0);
        assert!(b.try_reserve(0));
        assert!(!b.try_reserve(1));
        assert_eq!(b.utilization(), 0.0);
    }

    fn budget_with_probe(probe: MemProbe) -> VramBudget {
        VramBudget::new(2000, 1000, 0.0).with_probe(probe)
    }

    #[test]
    fn a_live_probe_can_only_tighten_the_configured_ceiling() {
        let free = Arc::new(std::sync::atomic::AtomicU64::new(300));
        let f = free.clone();
        let mut b = budget_with_probe(Arc::new(move || Some((f.load(std::sync::atomic::Ordering::Relaxed), 2000))));
        assert!(b.is_elastic());
        assert_eq!(b.capacity(), 1000, "no sample yet: trust the config");
        assert!(b.refresh());
        assert_eq!(b.capacity(), 300);
        assert_eq!(b.probed_free_bytes(), Some(300));
        assert_eq!(b.elastic_clamps(), 1);

        free.store(5_000, std::sync::atomic::Ordering::Relaxed);
        assert!(b.refresh());
        assert_eq!(b.capacity(), 1000, "a roomy device cannot exceed CUDA_VRAM");
        assert_eq!(b.configured_capacity(), 1000);
    }

    #[test]
    fn releasing_memory_raises_free_without_waiting_for_another_probe() {
        let mut b = budget_with_probe(Arc::new(|| Some((300, 2000))));
        assert!(b.refresh());
        assert!(b.try_reserve(200));
        assert_eq!(b.free(), 100);
        assert!(b.release(200));
        assert_eq!(b.free(), 300, "the ceiling is absolute, not recomputed from used");
        assert!(b.try_reserve(300));
        assert!(!b.try_reserve(1));
    }

    #[test]
    fn headroom_is_applied_to_the_live_sample_too() {
        let mut b = VramBudget::new(2000, 1000, 0.10).with_probe(Arc::new(|| Some((1000, 2000))));
        assert!(b.refresh());
        assert_eq!(b.capacity(), 900);
        assert_eq!(b.device_total(), 2000);
    }

    #[test]
    fn a_failed_probe_is_not_an_error_and_rate_limiting_works() {
        let mut b = budget_with_probe(Arc::new(|| None));
        assert!(!b.refresh(), "no sample taken");
        assert_eq!(b.capacity(), 1000, "keep trusting the configuration");
        assert!(!b.refresh_if_due(), "already tried recently");

        let mut b = budget_with_probe(Arc::new(|| Some((400, 2000))));
        assert!(b.refresh_if_due());
        assert_eq!(b.capacity(), 400);
        assert!(!b.refresh_if_due(), "no driver call within PROBE_INTERVAL");
    }

    #[test]
    fn a_shrunk_ceiling_declines_instead_of_overcommitting() {
        let mut b = budget_with_probe(Arc::new(|| Some((100, 2000))));
        assert!(b.refresh());
        assert!(!b.try_reserve(500));
        assert_eq!(b.used(), 0);
        assert_eq!(b.declines(), 1);
    }
}
