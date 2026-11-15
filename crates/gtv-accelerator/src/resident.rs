//! Device-resident block ledger — the gtVdB counterpart of PG-Strom's GPU Cache.
//!
//! ## What is absorbed
//!
//! PG-Strom's observation is the one that matters: for an analytical query, the
//! cost of *reading the table and moving it over PCIe* routinely dwarfs the cost
//! of the GPU arithmetic. Its answer is to keep a device copy of a hot table and
//! stop re-sending it per query.
//!
//! gtVdB absorbs the same idea at the operator level: a corpus or a right-side
//! series is uploaded once, keyed by `(kind, name, version)`, and every later
//! query against the same version is a cache hit.
//!
//! ## What is rewritten
//!
//! Three parts of the original design do not survive contact with an analytics
//! engine, and are deliberately replaced:
//!
//! | PG-Strom GPU Cache | Here | Why |
//! |---|---|---|
//! | Static `max_num_rows`; overflow ⇒ `corrupted`, cache stops working | Admission control + heat-aware eviction; overflow ⇒ that one block is declined and runs on the host | A graph/vector workload has many datasets of unpredictable size; one oversized table must not disable acceleration for everything else |
//! | Per-row AFTER trigger writes a REDO log; a query *blocks* until pending redo is applied | Write-behind delta with a `resident_rows` split: the query reads the resident prefix from the device and the un-synced tail from the host | Same consistency guarantee (a query always sees current data) with no query-time stall |
//! | Visibility map / `refcnt` for MVCC | Immutable version keys | gtVdB segments and vector corpora are immutable snapshots, so a version number is sufficient and there is no snapshot bookkeeping to get wrong |
//!
//! The delta counters (`accel_sync_threshold_bytes`, `accel_sync_interval`)
//! mirror `gpu_sync_threshold` / `gpu_sync_interval`, including the default of a
//! quarter of the buffer.
//!
//! ## Why heat, not just LRU
//!
//! LRU is *position*-based: it forgets how often a block is used. On a card
//! that holds two 2 GiB corpora, alternating queries with a third one makes
//! pure LRU evict a hot corpus every time — each query then pays a ~0.4 s
//! re-upload. The fix borrowed from Strata's tiered expert cache (and from
//! W-TinyLFU) is a decayed frequency estimate:
//!
//! * each reference adds 1 to a block's heat, which halves every
//!   `ACCEL_HEAT_HALF_LIFE`;
//! * eviction picks the coldest block, ties broken by recency;
//! * a block with proven heat is protected from a first-time newcomer, and a
//!   **declined** block still accumulates heat in a small shadow ledger — so a
//!   genuinely hot new corpus is admitted after a few queries instead of being
//!   locked out forever by the old working set.
//!
//! Setting the half-life to zero disables the policy and restores pure LRU.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use gtv_params::AccelCache;

use crate::budget::VramBudget;

/// Default half-life of a block's decayed heat. Short enough that a workload
/// shift takes over in a couple of minutes, long enough that a burst of
/// alternating queries keeps its working set.
pub const DEFAULT_HEAT_HALF_LIFE: Duration = Duration::from_secs(30);
/// A resident block must have at least this much decayed heat before the policy
/// protects it. Below it, eviction is plain LRU so one-shot blocks still make
/// room for each other.
pub const PROTECT_FLOOR: f64 = 4.0;
/// A newcomer's shadow heat must reach `victim_heat / ADMIT_MARGIN` to displace
/// a protected block. The margin is the hysteresis that stops two datasets of
/// similar heat from swapping on every query.
pub const ADMIT_MARGIN: f64 = 1.25;
/// Bounded shadow ledger: the decline window is evidence, not a second cache.
const MAX_SHADOW: usize = 256;
/// Minimum shadow heat before a non-resident block is worth pulling in
/// proactively. One reference earns 1.0 and decays below it, so a one-shot scan
/// never qualifies; two references (the point at which a workload has
/// demonstrably asked twice) do. Deliberately low: the shadow ledger is bounded
/// and the pump is rate-limited by `ACCEL_PREFETCH_BLOCKS`, so the expensive
/// mistake is *not* prefetching, not prefetching one block too many.
pub const PREFETCH_HEAT_FLOOR: f64 = 1.5;

/// Heat at `now` after exponential decay. A zero half-life disables decay (and,
/// by convention in this module, the whole heat policy).
fn decayed(heat: f64, touched: Instant, now: Instant, half_life: Duration) -> f64 {
    if half_life.is_zero() {
        return heat;
    }
    let dt = now.saturating_duration_since(touched).as_secs_f64();
    heat * 0.5f64.powf(dt / half_life.as_secs_f64())
}

/// What a resident block holds. Different kinds have different mutation
/// patterns, so they are accounted separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentKind {
    /// A row-major vector corpus (`gtv-index` flat index).
    Corpus,
    /// A cell-ordered IVF corpus. Kept separate from `Corpus` because the two
    /// layouts can coexist under the same collection name; mixing their device
    /// buffers would silently score the wrong rows.
    IvfCorpus,
    /// The right-hand time series of an as-of join.
    RightSeries,
    /// A time-bucket offset table accelerating the as-of merge.
    BucketIndex,
}

impl ResidentKind {
    /// Short label used in telemetry output.
    pub fn as_str(self) -> &'static str {
        match self {
            ResidentKind::Corpus => "corpus",
            ResidentKind::IvfCorpus => "ivf",
            ResidentKind::RightSeries => "series",
            ResidentKind::BucketIndex => "bucket",
        }
    }
}

/// Cache identity: a kind plus a caller-chosen name (index name, table name, …).
///
/// `version` is supplied per call rather than stored here, because the same
/// logical dataset can be re-registered with new content.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResidentKey {
    pub kind: ResidentKind,
    pub name: String,
}

impl ResidentKey {
    pub fn new(kind: ResidentKind, name: impl Into<String>) -> Self {
        Self {
            kind,
            name: name.into(),
        }
    }
}

/// Why an admission was refused. Distinguishing "too big" from "a hotter block
/// is protected" from "the driver has less free memory than we thought" is what
/// makes `.accel` actionable instead of just showing a decline count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclineReason {
    /// `ACCEL_CACHE=OFF`.
    Disabled,
    /// Cannot fit even with an empty cache.
    TooLarge,
    /// Eviction could not free enough room.
    NoRoom,
    /// A proven-hot resident block was protected from a cold newcomer.
    HeatProtected,
    /// The live device probe tightened the budget below this request.
    ElasticBudget,
    /// A heat-pump upload wanted to fill VRAM that is currently in use. The
    /// pump never evicts, so this is the expected answer on a saturated cache.
    NoSlack,
}

impl DeclineReason {
    pub fn as_str(self) -> &'static str {
        match self {
            DeclineReason::Disabled => "disabled",
            DeclineReason::TooLarge => "too-large",
            DeclineReason::NoRoom => "no-room",
            DeclineReason::HeatProtected => "heat-protected",
            DeclineReason::ElasticBudget => "elastic-budget",
            DeclineReason::NoSlack => "no-slack",
        }
    }
}

/// The outcome of asking to use a device copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// The whole block is on the device and current.
    Resident,
    /// The device holds the first `resident_rows`; the remaining
    /// `total_rows - resident_rows` must be served from the host. The caller
    /// stays consistent without flushing anything.
    Split {
        resident_rows: u64,
        total_rows: u64,
    },
    /// No room (or the cache is disabled). The caller runs on the host; this is
    /// what keeps one oversized dataset from disabling every other dataset.
    Declined {
        needed: u64,
        free: u64,
        reason: DeclineReason,
    },
}

impl Admission {
    /// Whether a device copy may be used at all.
    pub fn usable(&self) -> bool {
        !matches!(self, Admission::Declined { .. })
    }

    /// The refusal reason, if this is a decline.
    pub fn decline_reason(&self) -> Option<DeclineReason> {
        match self {
            Admission::Declined { reason, .. } => Some(*reason),
            _ => None,
        }
    }
}

/// Whether pending delta should be applied now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    /// Keep accumulating; the split already keeps queries consistent.
    Deferred,
    /// A batched upload is due (byte threshold or interval elapsed).
    Due,
}

#[derive(Debug, Clone)]
struct Block {
    bytes: u64,
    rows: u64,
    resident_rows: u64,
    version: u64,
    epoch: u64,
    hits: u64,
    misses: u64,
    uploads: u64,
    /// Decayed-frequency estimate; see the module header.
    heat: f64,
    touched: Instant,
}

/// Evidence kept for a block that is *not* resident. Without this a cold
/// newcomer can never accumulate enough heat to displace an old working set.
#[derive(Debug, Clone)]
struct Shadow {
    heat: f64,
    touched: Instant,
    /// Times this key was refused to protect a hotter resident block.
    deferred: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct Delta {
    rows: u64,
    bytes: u64,
}

/// Per-block view for CLI / `EXPLAIN` output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockView {
    pub kind: ResidentKind,
    pub name: String,
    pub bytes: u64,
    pub rows: u64,
    pub resident_rows: u64,
    pub hits: u64,
    pub misses: u64,
    pub uploads: u64,
    /// Decayed heat × 1000, at the time the view was taken.
    pub heat_milli: u32,
}

/// Aggregate counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentTelemetry {
    pub blocks: usize,
    pub resident_bytes: u64,
    pub capacity_bytes: u64,
    pub configured_capacity_bytes: u64,
    pub device_total_bytes: u64,
    /// Live free bytes after headroom; 0 when no probe is attached / sampled.
    pub device_free_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub uploads: u64,
    pub upload_bytes: u64,
    pub evictions: u64,
    pub evicted_bytes: u64,
    pub pending_rows: u64,
    pub pending_bytes: u64,
    pub declines: u64,
    /// Admissions refused because a hotter block was protected.
    pub heat_deferred: u64,
    /// Blocks admitted after at least one deferral (the shadow ledger won).
    pub promotions: u64,
    /// Non-resident keys currently tracked as evidence.
    pub shadow_blocks: usize,
    /// Probes that tightened the capacity below `CUDA_VRAM`.
    pub elastic_clamps: u64,
    /// Hottest resident block, heat × 1000.
    pub hottest_milli: u32,
}

impl ResidentTelemetry {
    /// Hit ratio over admission decisions; `0.0` before the first lookup.
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}

/// The registry itself: budget + blocks + write-behind delta.
#[derive(Debug)]
pub struct ResidentRegistry {
    mode: AccelCache,
    budget: VramBudget,
    blocks: HashMap<ResidentKey, Block>,
    pending: HashMap<ResidentKey, Delta>,
    /// Decayed heat for keys that were declined; bounded by `MAX_SHADOW`.
    shadow: HashMap<ResidentKey, Shadow>,
    epoch: u64,
    uploads: u64,
    upload_bytes: u64,
    evictions: u64,
    evicted_bytes: u64,
    heat_deferred: u64,
    promotions: u64,
    sync_threshold_bytes: u64,
    sync_interval: Duration,
    heat_half_life: Duration,
    last_sync: Instant,
    /// Blocks evicted since the last [`ResidentRegistry::take_evictions`], so the
    /// caller can release the backend's own device copy and keep the ledger
    /// truthful.
    recent_evictions: Vec<ResidentKey>,
}

impl ResidentRegistry {
    /// New ledger for `mode` drawing from `budget`.
    pub fn new(
        mode: AccelCache,
        budget: VramBudget,
        sync_threshold_bytes: u64,
        sync_interval: Duration,
    ) -> Self {
        Self {
            mode,
            budget,
            blocks: HashMap::new(),
            pending: HashMap::new(),
            shadow: HashMap::new(),
            epoch: 0,
            uploads: 0,
            upload_bytes: 0,
            evictions: 0,
            evicted_bytes: 0,
            heat_deferred: 0,
            promotions: 0,
            sync_threshold_bytes,
            sync_interval,
            heat_half_life: DEFAULT_HEAT_HALF_LIFE,
            last_sync: Instant::now(),
            recent_evictions: Vec::new(),
        }
    }

    /// Build from resolved parameters.
    pub fn from_resolved(params: &gtv_params::ResolvedParams) -> Self {
        Self::new(
            params.accel_cache,
            VramBudget::from_resolved(params),
            params.accel_sync_threshold_bytes,
            params.accel_sync_interval,
        )
        .with_heat_half_life(params.accel_heat_half_life)
    }

    /// Set the decay horizon. `Duration::ZERO` disables heat and restores pure
    /// LRU, which is useful for comparison runs and for workloads with no reuse.
    pub fn with_heat_half_life(mut self, half_life: Duration) -> Self {
        self.heat_half_life = half_life;
        self
    }

    /// Attach a live device-memory probe to the budget (item 4). Called by the
    /// registry once the backend exists, since only it can query the driver.
    pub fn with_mem_probe(mut self, probe: crate::budget::MemProbe) -> Self {
        self.budget = self.budget.with_probe(probe);
        self
    }

    /// The configured mode.
    pub fn mode(&self) -> AccelCache {
        self.mode
    }

    /// The underlying budget.
    pub fn budget(&self) -> &VramBudget {
        &self.budget
    }

    /// Mutable access for an explicit `refresh()` (CLI / preflight).
    pub fn budget_mut(&mut self) -> &mut VramBudget {
        &mut self.budget
    }

    fn heat_enabled(&self) -> bool {
        !self.heat_half_life.is_zero()
    }

    fn heat_of(&self, block: &Block, now: Instant) -> f64 {
        if self.heat_enabled() {
            decayed(block.heat, block.touched, now, self.heat_half_life)
        } else {
            0.0
        }
    }

    /// Whether a device copy may be allocated at all.
    pub fn permits_device_memory(&self) -> bool {
        self.mode.uses_device() && self.budget.capacity() > 0
    }

    fn tick(&mut self) -> u64 {
        self.epoch = self.epoch.saturating_add(1);
        self.epoch
    }

    /// Count a reference to a key that is *not* resident. Returns its current
    /// decayed heat and how many times it was deferred by the heat policy.
    ///
    /// This is the admission window of a TinyLFU-style cache. Without it a
    /// repeatedly requested corpus that lost its first admission would stay on
    /// the host forever, because heat can only be earned by resident hits.
    fn note_reference(&mut self, key: &ResidentKey, now: Instant) -> (f64, u64) {
        if !self.heat_enabled() {
            return (1.0, 0);
        }
        let half_life = self.heat_half_life;
        let entry = self.shadow.entry(key.clone()).or_insert(Shadow {
            heat: 0.0,
            touched: now,
            deferred: 0,
        });
        entry.heat = decayed(entry.heat, entry.touched, now, half_life) + 1.0;
        entry.touched = now;
        let result = (entry.heat, entry.deferred);

        if self.shadow.len() > MAX_SHADOW {
            let mut ranked: Vec<(f64, ResidentKey)> = self
                .shadow
                .iter()
                .filter(|(k, _)| *k != key)
                .map(|(k, s)| (decayed(s.heat, s.touched, now, half_life), k.clone()))
                .collect();
            ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let drop_n = self.shadow.len() - MAX_SHADOW / 2;
            for (_, k) in ranked.into_iter().take(drop_n) {
                self.shadow.remove(&k);
            }
        }
        result
    }

    /// Ask for a device copy of `key` covering `rows` rows / `bytes` bytes.
    ///
    /// A repeat call with the same `version` and `rows` is a hit; a new
    /// `version` is a miss that replaces the device copy in place. A first-time
    /// or evicted key competes on decayed heat, not just recency.
    pub fn admit(
        &mut self,
        key: &ResidentKey,
        bytes: u64,
        rows: u64,
        version: u64,
    ) -> Admission {
        self.admit_with(key, bytes, rows, version, ADMIT_MARGIN, true)
    }

    /// Ask for a device copy on behalf of the idle heat pump.
    ///
    /// Two things differ from [`Self::admit`]:
    ///
    /// * **It never evicts.** The pump may only fill VRAM that is *free right
    ///   now*; if the cache is saturated it declines with
    ///   [`DeclineReason::NoSlack`]. Measured reason: a pump allowed to evict
    ///   re-uploads the block a query just replaced, which on a 3×2 rotation
    ///   raised transferred bytes by 71% and p99 by 41% for no gain. Filling
    ///   slack cannot regress anything — it uses VRAM that would otherwise idle.
    /// * **It consumes heat instead of adding to it.** An unsolicited upload is
    ///   not a reference from the workload; counting it would make pumped blocks
    ///   outrank the ones queries actually asked for.
    ///
    /// The natural trigger is VRAM released by a re-registered (smaller or
    /// dropped) collection: the freed region is exactly where the previous
    /// working set wants to live, and the shadow ledger holds the blocks that
    /// had to leave.
    pub fn admit_prefetch(
        &mut self,
        key: &ResidentKey,
        bytes: u64,
        rows: u64,
        version: u64,
    ) -> Admission {
        self.admit_with(key, bytes, rows, version, 1.0, false)
    }

    fn admit_with(
        &mut self,
        key: &ResidentKey,
        bytes: u64,
        rows: u64,
        version: u64,
        margin: f64,
        count_reference: bool,
    ) -> Admission {
        if !self.mode.tracks_residency() {
            return Admission::Declined {
                needed: bytes,
                free: 0,
                reason: DeclineReason::Disabled,
            };
        }
        // Live device memory may have shrunk under us (another process, WDDM
        // sysmem fallback). Re-check before committing to a new allocation.
        self.budget.refresh_if_due();
        let now = Instant::now();

        let declined = |reason: DeclineReason, slf: &mut Self| -> Admission {
            slf.budget.record_decline(bytes);
            if reason == DeclineReason::HeatProtected {
                slf.heat_deferred = slf.heat_deferred.saturating_add(1);
                if let Some(shadow) = slf.shadow.get_mut(key) {
                    shadow.deferred = shadow.deferred.saturating_add(1);
                }
            }
            Admission::Declined {
                needed: bytes,
                free: slf.budget.free(),
                reason,
            }
        };

        let too_large = |slf: &mut Self| -> Admission {
            let reason = if bytes > slf.budget.configured_capacity() {
                DeclineReason::TooLarge
            } else {
                DeclineReason::ElasticBudget
            };
            declined(reason, slf)
        };

        if let Some((block_version, block_rows, block_bytes, resident_rows, block_heat, block_touched)) = self
            .blocks
            .get(key)
            .map(|b| (b.version, b.rows, b.bytes, b.resident_rows, b.heat, b.touched))
        {
            if block_version == version && block_rows == rows {
                let epoch = self.tick();
                let half_life = self.heat_half_life;
                if let Some(block) = self.blocks.get_mut(key) {
                    block.hits = block.hits.saturating_add(1);
                    block.epoch = epoch;
                    if count_reference && !half_life.is_zero() {
                        block.heat = decayed(block.heat, block.touched, now, half_life) + 1.0;
                    } else if !half_life.is_zero() {
                        // A pump hit is not a reference: decay, never add.
                        block.heat = decayed(block.heat, block.touched, now, half_life);
                    }
                    block.touched = now;
                }
                return if resident_rows >= rows {
                    Admission::Resident
                } else {
                    Admission::Split {
                        resident_rows,
                        total_rows: rows,
                    }
                };
            }

            // Content changed: the device copy is stale for the rewritten
            // region, so the whole block is re-sent once and the delta resets.
            // Its earned heat survives the rewrite.
            let grow = bytes.saturating_sub(block_bytes);
            if bytes > self.budget.capacity() {
                return too_large(&mut *self);
            }
            if grow > 0 {
                let standing = if self.heat_enabled() {
                    decayed(block_heat, block_touched, now, self.heat_half_life)
                } else {
                    0.0
                };
                if let Err(reason) = self.make_room(grow, Some(key), standing, now, margin, count_reference) {
                    return declined(reason, self);
                }
            }
            if grow > 0 {
                self.budget.try_reserve(grow);
            } else {
                self.budget.release(block_bytes.saturating_sub(bytes));
            }
            let epoch = self.tick();
            if let Some(block) = self.blocks.get_mut(key) {
                block.bytes = bytes;
                block.rows = rows;
                block.resident_rows = rows;
                block.version = version;
                block.uploads = block.uploads.saturating_add(1);
                block.misses = block.misses.saturating_add(1);
                block.epoch = epoch;
                block.touched = now;
            }
            self.uploads = self.uploads.saturating_add(1);
            self.upload_bytes = self.upload_bytes.saturating_add(bytes);
            self.pending.remove(key);
            return Admission::Resident;
        }

        let (candidate_heat, deferred_before) = if count_reference {
            self.note_reference(key, now)
        } else {
            // Pump admission: spend the heat the workload already earned, but do
            // not add to it (see `admit_prefetch`).
            let heat = self
                .shadow
                .get(key)
                .map(|s| decayed(s.heat, s.touched, now, self.heat_half_life))
                .unwrap_or(0.0);
            (heat, 0)
        };
        if bytes > self.budget.capacity() {
            return too_large(self);
        }
        if let Err(reason) = self.make_room(bytes, None, candidate_heat, now, margin, count_reference) {
            return declined(reason, self);
        }
        self.budget.try_reserve(bytes);
        let epoch = self.tick();
        self.blocks.insert(
            key.clone(),
            Block {
                bytes,
                rows,
                resident_rows: rows,
                version,
                epoch,
                hits: 0,
                misses: 1,
                uploads: 1,
                heat: candidate_heat,
                touched: now,
            },
        );
        if deferred_before > 0 {
            self.promotions = self.promotions.saturating_add(1);
        }
        self.shadow.remove(key);
        self.uploads = self.uploads.saturating_add(1);
        self.upload_bytes = self.upload_bytes.saturating_add(bytes);
        Admission::Resident
    }

    /// Record `rows` appended to `key` that are not on the device yet.
    ///
    /// This is the write-behind path: the device keeps serving the prefix it
    /// already has while the tail stays on the host, so a query never has to
    /// wait for a flush. Returns whether a batched upload has become due.
    pub fn note_append(&mut self, key: &ResidentKey, rows: u64, bytes: u64) -> SyncPolicy {
        if !self.mode.tracks_residency() || rows == 0 {
            return SyncPolicy::Deferred;
        }
        if let Some(block) = self.blocks.get_mut(key) {
            block.rows = block.rows.saturating_add(rows);
        }
        let delta = self.pending.entry(key.clone()).or_default();
        delta.rows = delta.rows.saturating_add(rows);
        delta.bytes = delta.bytes.saturating_add(bytes);

        let pending_bytes: u64 = self.pending.values().map(|d| d.bytes).sum();
        let overdue = self.sync_interval.is_zero() || self.last_sync.elapsed() >= self.sync_interval;
        let over_threshold = self.sync_threshold_bytes > 0 && pending_bytes >= self.sync_threshold_bytes;
        if overdue || over_threshold {
            SyncPolicy::Due
        } else {
            SyncPolicy::Deferred
        }
    }

    /// Apply every pending delta to its block. Returns `(blocks, bytes)`.
    ///
    /// Every caller of this is a *batch* boundary — the point is to amortize a
    /// transfer over many appends, never to make a single query wait.
    pub fn flush_pending(&mut self) -> (u64, u64) {
        if self.pending.is_empty() {
            self.last_sync = Instant::now();
            return (0, 0);
        }
        let drained: Vec<(ResidentKey, Delta)> = self.pending.drain().collect();
        let mut blocks = 0u64;
        let mut bytes = 0u64;
        for (key, delta) in drained {
            if let Some(block) = self.blocks.get_mut(&key) {
                block.resident_rows = block.rows;
                block.uploads = block.uploads.saturating_add(1);
                blocks = blocks.saturating_add(1);
                bytes = bytes.saturating_add(delta.bytes);
            }
        }
        self.uploads = self.uploads.saturating_add(blocks);
        self.upload_bytes = self.upload_bytes.saturating_add(bytes);
        self.last_sync = Instant::now();
        (blocks, bytes)
    }

    /// Resident prefix and total rows for `key`, if it is known.
    pub fn split(&self, key: &ResidentKey) -> Option<(u64, u64)> {
        self.blocks
            .get(key)
            .map(|b| (b.resident_rows, b.rows))
    }

    /// Whether `key` currently has a device block (any split state).
    pub fn is_resident(&self, key: &ResidentKey) -> bool {
        self.blocks.contains_key(key)
    }

    /// Drop a block and return its bytes to the budget.
    pub fn remove(&mut self, key: &ResidentKey) -> bool {
        let Some(block) = self.blocks.remove(key) else {
            return false;
        };
        self.budget.release(block.bytes);
        self.pending.remove(key);
        true
    }

    /// Drop a block, return its bytes to the budget, and report how many.
    ///
    /// Unlike an eviction this also drops the *shadow* entry: a block released
    /// this way belongs to a dataset that no longer exists, so its heat has no
    /// future use either. That distinction is the whole point — the caller is
    /// the P2 heat pump's reclaim step, which converts dead VRAM (a dropped
    /// collection's device copy) into the free slack a pump pass can fill.
    pub fn release(&mut self, key: &ResidentKey) -> u64 {
        let Some(block) = self.blocks.remove(key) else {
            return 0;
        };
        self.budget.release(block.bytes);
        self.pending.remove(key);
        self.shadow.remove(key);
        block.bytes
    }

    /// Evict the coldest blocks until `need` bytes are free.
    ///
    /// With heat enabled the victim is the lowest *decayed frequency* (ties
    /// broken by recency, i.e. LRU). A resident block whose heat has cleared
    /// [`PROTECT_FLOOR`] is not displaced by a newcomer unless that newcomer's
    /// shadow heat is at least `ADMIT_MARGIN ×` higher — the hysteresis that
    /// stops two equally hot corpora from ping-ponging across PCIe on every
    /// query. Without heat this is plain LRU.
    ///
    /// `protect` is never evicted (the block currently being admitted), which
    /// prevents an admit from evicting itself and then failing anyway.
    ///
    /// `margin` is the hysteresis between the candidate and the victim: queries
    /// pass [`ADMIT_MARGIN`], the idle heat pump passes `1.0` (see
    /// [`Self::admit_prefetch`]).
    ///
    /// `allow_evict == false` restricts the request to VRAM that is free right
    /// now: no victim is chosen at all. The heat pump runs in that mode so an
    /// unsolicited upload can never displace a block a query is using.
    fn make_room(
        &mut self,
        need: u64,
        protect: Option<&ResidentKey>,
        candidate_heat: f64,
        now: Instant,
        margin: f64,
        allow_evict: bool,
    ) -> Result<(), DeclineReason> {
        if need == 0 || self.budget.free() >= need {
            return Ok(());
        }
        if !allow_evict {
            return Err(DeclineReason::NoSlack);
        }
        while self.budget.free() < need {
            let victim = self
                .blocks
                .iter()
                .filter(|(k, _)| protect != Some(*k))
                .min_by(|(_, a), (_, b)| {
                    let ha = self.heat_of(a, now);
                    let hb = self.heat_of(b, now);
                    ha.partial_cmp(&hb)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.epoch.cmp(&b.epoch))
                })
                .map(|(k, b)| (k.clone(), b.bytes, self.heat_of(b, now)));
            let Some((key, bytes, victim_heat)) = victim else {
                return Err(DeclineReason::NoRoom);
            };
            if self.heat_enabled()
                && victim_heat >= PROTECT_FLOOR
                && candidate_heat < victim_heat * margin
            {
                return Err(DeclineReason::HeatProtected);
            }
            self.blocks.remove(&key);
            self.pending.remove(&key);
            self.budget.release(bytes);
            self.evictions = self.evictions.saturating_add(1);
            self.evicted_bytes = self.evicted_bytes.saturating_add(bytes);
            // Demotion preserves evidence: a hot block that lost its place can
            // win it back on the next few queries instead of starting cold.
            if self.heat_enabled() && victim_heat > 0.0 {
                self.shadow.entry(key.clone()).or_insert(Shadow {
                    heat: 0.0,
                    touched: now,
                    deferred: 0,
                });
                if let Some(s) = self.shadow.get_mut(&key) {
                    s.heat = s.heat.max(victim_heat);
                    s.touched = now;
                }
            }
            self.recent_evictions.push(key);
        }
        Ok(())
    }

    /// Take the keys evicted since the previous call.
    pub fn take_evictions(&mut self) -> Vec<ResidentKey> {
        std::mem::take(&mut self.recent_evictions)
    }

    /// Non-resident keys ranked by decayed heat, hottest first.
    ///
    /// This is the read side of the shadow ledger. A block that keeps losing
    /// admission still records how often it was asked for ([`Self::admit`] →
    /// [`Self::note_reference`]); the P2 heat pump uses that evidence to upload
    /// it *before* the next query needs it, instead of paying the transfer on
    /// the query's critical path.
    ///
    /// Only keys at or above `min_heat` are returned — a one-shot scan earns
    /// ~1.0 and must never get a slot — and keys that are currently resident are
    /// skipped because they need no pump. Ties break on name so the candidate
    /// order is stable between passes.
    pub fn hottest_shadows(&self, min_heat: f64, limit: usize) -> Vec<(ResidentKey, f64)> {
        if limit == 0 || self.shadow.is_empty() {
            return Vec::new();
        }
        let now = Instant::now();
        let half_life = self.heat_half_life;
        let mut ranked: Vec<(ResidentKey, f64)> = self
            .shadow
            .iter()
            .filter(|(k, _)| !self.blocks.contains_key(*k))
            .map(|(k, s)| (k.clone(), decayed(s.heat, s.touched, now, half_life)))
            .filter(|(_, heat)| *heat >= min_heat)
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.name.cmp(&b.0.name))
        });
        ranked.truncate(limit);
        ranked
    }

    /// Drop the shadow evidence for `key` (the dataset is gone, or the caller
    /// decided it will never be prefetched). Returns whether anything was
    /// removed.
    pub fn forget_shadow(&mut self, key: &ResidentKey) -> bool {
        self.shadow.remove(key).is_some()
    }

    /// One row per resident block, sorted by name for stable output.
    pub fn blocks(&self) -> Vec<BlockView> {
        let now = Instant::now();
        let mut v: Vec<BlockView> = self
            .blocks
            .iter()
            .map(|(k, b)| BlockView {
                kind: k.kind,
                name: k.name.clone(),
                bytes: b.bytes,
                rows: b.rows,
                resident_rows: b.resident_rows,
                hits: b.hits,
                misses: b.misses,
                uploads: b.uploads,
                heat_milli: ((self.heat_of(b, now) * 1000.0).max(0.0) as u32),
            })
            .collect();
        v.sort_by(|a, b| a.kind.as_str().cmp(b.kind.as_str()).then_with(|| a.name.cmp(&b.name)));
        v
    }

    /// Aggregate counters.
    pub fn telemetry(&self) -> ResidentTelemetry {
        let pending_rows = self.pending.values().map(|d| d.rows).sum();
        let pending_bytes = self.pending.values().map(|d| d.bytes).sum();
        let now = Instant::now();
        let hottest = self
            .blocks
            .values()
            .map(|b| self.heat_of(b, now))
            .fold(0.0f64, f64::max);
        ResidentTelemetry {
            blocks: self.blocks.len(),
            resident_bytes: self.budget.used(),
            capacity_bytes: self.budget.capacity(),
            configured_capacity_bytes: self.budget.configured_capacity(),
            device_total_bytes: self.budget.device_total(),
            device_free_bytes: self.budget.probed_free_bytes().unwrap_or(0),
            hits: self.blocks.values().map(|b| b.hits).sum(),
            misses: self.blocks.values().map(|b| b.misses).sum(),
            uploads: self.uploads,
            upload_bytes: self.upload_bytes,
            evictions: self.evictions,
            evicted_bytes: self.evicted_bytes,
            pending_rows,
            pending_bytes,
            declines: self.budget.declines(),
            heat_deferred: self.heat_deferred,
            promotions: self.promotions,
            shadow_blocks: self.shadow.len(),
            elastic_clamps: self.budget.elastic_clamps(),
            hottest_milli: ((hottest * 1000.0).max(0.0) as u32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn reg(mode: AccelCache, capacity: u64, threshold: u64) -> ResidentRegistry {
        ResidentRegistry::new(
            mode,
            VramBudget::new(capacity, capacity, 0.0),
            threshold,
            Duration::from_secs(5),
        )
    }

    fn key(name: &str) -> ResidentKey {
        ResidentKey::new(ResidentKind::Corpus, name)
    }

    #[test]
    fn first_admission_is_resident_and_second_is_a_hit() {
        let mut r = reg(AccelCache::On, 1000, 0);
        assert_eq!(r.admit(&key("a"), 400, 100, 1), Admission::Resident);
        assert_eq!(r.admit(&key("a"), 400, 100, 1), Admission::Resident);
        let t = r.telemetry();
        assert_eq!(t.hits, 1);
        assert_eq!(t.misses, 1);
        assert_eq!(t.resident_bytes, 400);
        assert_eq!(t.upload_bytes, 400);
        assert!((t.hit_ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn new_version_replaces_the_device_copy() {
        let mut r = reg(AccelCache::On, 1000, 0);
        r.admit(&key("a"), 400, 100, 1);
        assert_eq!(r.admit(&key("a"), 400, 100, 2), Admission::Resident);
        let t = r.telemetry();
        assert_eq!(t.upload_bytes, 800);
        assert_eq!(t.resident_bytes, 400, "replacing must not double-reserve");
    }

    #[test]
    fn appends_split_instead_of_blocking_a_query() {
        let mut r = reg(AccelCache::On, 1000, 10_000);
        r.admit(&key("a"), 400, 100, 1);
        assert_eq!(r.note_append(&key("a"), 50, 200), SyncPolicy::Deferred);
        // Same version, more rows: the device still covers the first 100.
        assert_eq!(
            r.admit(&key("a"), 400, 150, 1),
            Admission::Split {
                resident_rows: 100,
                total_rows: 150
            }
        );
        assert_eq!(r.split(&key("a")), Some((100, 150)));
        let (blocks, bytes) = r.flush_pending();
        assert_eq!((blocks, bytes), (1, 200));
        assert_eq!(
            r.admit(&key("a"), 400, 150, 1),
            Admission::Resident,
            "after the batched upload the whole block is resident"
        );
    }

    #[test]
    fn threshold_makes_a_sync_due() {
        let mut r = reg(AccelCache::On, 1000, 100);
        r.admit(&key("a"), 400, 100, 1);
        assert_eq!(r.note_append(&key("a"), 10, 60), SyncPolicy::Deferred);
        assert_eq!(r.note_append(&key("a"), 10, 60), SyncPolicy::Due);
    }

    #[test]
    fn zero_interval_makes_every_append_due() {
        let mut r = ResidentRegistry::new(
            AccelCache::On,
            VramBudget::new(1000, 1000, 0.0),
            0,
            Duration::ZERO,
        );
        r.admit(&key("a"), 100, 10, 1);
        assert_eq!(r.note_append(&key("a"), 1, 1), SyncPolicy::Due);
    }

    #[test]
    fn lru_evicts_the_coldest_block() {
        let mut r = reg(AccelCache::On, 1000, 0);
        r.admit(&key("hot"), 400, 10, 1);
        r.admit(&key("cold"), 400, 10, 1);
        // Touch "hot" so "cold" becomes the least recently used.
        r.admit(&key("hot"), 400, 10, 1);
        assert_eq!(r.admit(&key("new"), 400, 10, 1), Admission::Resident);
        assert!(r.blocks().iter().all(|b| b.name != "cold"));
        assert_eq!(r.telemetry().evictions, 1);
        assert_eq!(r.telemetry().resident_bytes, 800);
    }

    #[test]
    fn an_oversized_block_is_declined_not_corrupted() {
        let mut r = reg(AccelCache::On, 1000, 0);
        r.admit(&key("small"), 400, 10, 1);
        let a = r.admit(&key("huge"), 5000, 10, 1);
        assert!(matches!(a, Admission::Declined { .. }));
        assert!(!a.usable());
        // Nothing was evicted for an admission that could never succeed, so the
        // small block is still resident and the rest of the cache keeps working.
        assert_eq!(r.telemetry().evictions, 0);
        assert_eq!(r.admit(&key("small"), 400, 10, 1), Admission::Resident);
        assert_eq!(r.telemetry().declines, 1);
    }

    #[test]
    fn a_declined_admission_leaves_other_blocks_untouched() {
        let mut r = reg(AccelCache::On, 1000, 0);
        r.admit(&key("a"), 600, 10, 1);
        // Fits the budget on its own, but not alongside `a`: eviction is tried.
        let a = r.admit(&key("b"), 800, 10, 1);
        assert_eq!(a, Admission::Resident);
        assert_eq!(r.telemetry().evictions, 1);
        assert_eq!(r.telemetry().resident_bytes, 800);
    }

    #[test]
    fn off_mode_declines_everything() {
        let mut r = reg(AccelCache::Off, 1000, 0);
        assert!(matches!(
            r.admit(&key("a"), 10, 1, 1),
            Admission::Declined { .. }
        ));
        assert_eq!(r.note_append(&key("a"), 1, 1), SyncPolicy::Deferred);
        assert!(!r.permits_device_memory());
    }

    #[test]
    fn shadow_mode_tracks_without_device_memory() {
        let mut r = reg(AccelCache::Shadow, 1000, 0);
        assert_eq!(r.admit(&key("a"), 400, 100, 1), Admission::Resident);
        assert_eq!(r.telemetry().upload_bytes, 400);
        assert!(!r.permits_device_memory(), "shadow must not allocate");
    }

    #[test]
    fn removing_a_block_returns_its_bytes() {
        let mut r = reg(AccelCache::On, 1000, 0);
        r.admit(&key("a"), 400, 100, 1);
        assert!(r.remove(&key("a")));
        assert_eq!(r.telemetry().resident_bytes, 0);
        assert!(!r.remove(&key("a")));
    }

    #[test]
    fn a_hot_block_is_not_evicted_by_a_one_shot_scan() {
        // Room for exactly one block: the newcomer must evict to get in.
        let mut r = reg(AccelCache::On, 500, 0);
        assert_eq!(r.admit(&key("hot"), 400, 10, 1), Admission::Resident);
        // Six references clear PROTECT_FLOOR and build a real working set.
        for _ in 0..5 {
            assert_eq!(r.admit(&key("hot"), 400, 10, 1), Admission::Resident);
        }
        let a = r.admit(&key("scan"), 400, 10, 1);
        assert_eq!(a.decline_reason(), Some(DeclineReason::HeatProtected));
        assert_eq!(r.telemetry().evictions, 0, "the hot corpus stays resident");
        assert_eq!(r.telemetry().heat_deferred, 1);
        assert!(r.blocks().iter().any(|b| b.name == "hot"));
    }

    #[test]
    fn a_persistent_newcomer_earns_admission_and_a_hot_eviction_keeps_its_heat() {
        let mut r = reg(AccelCache::On, 700, 0);
        r.admit(&key("old"), 600, 10, 1);
        for _ in 0..5 {
            r.admit(&key("old"), 600, 10, 1);
        }
        // Each refusal raises the newcomer's shadow heat. Once it is 1.25x the
        // resident's heat, the admission wins rather than being locked out.
        let mut admitted_at = None;
        for attempt in 1..=12 {
            if r.admit(&key("new"), 600, 10, 1) == Admission::Resident {
                admitted_at = Some(attempt);
                break;
            }
        }
        assert!(admitted_at.is_some(), "a hot newcomer must eventually be promoted");
        let t = r.telemetry();
        assert_eq!(t.promotions, 1);
        assert_eq!(t.evictions, 1);
        assert_eq!(t.shadow_blocks, 1, "the demoted block keeps its evidence");
        assert!(t.hottest_milli >= 6_000, "promoted heat carries over");
    }

    #[test]
    fn zero_half_life_restores_plain_lru() {
        let mut r = reg(AccelCache::On, 1000, 0).with_heat_half_life(Duration::ZERO);
        r.admit(&key("hot"), 400, 10, 1);
        for _ in 0..5 {
            r.admit(&key("hot"), 400, 10, 1);
        }
        assert_eq!(r.admit(&key("scan"), 400, 10, 1), Admission::Resident);
        assert_eq!(r.telemetry().heat_deferred, 0);
        assert_eq!(r.telemetry().shadow_blocks, 0);
    }

    #[test]
    fn a_shrunk_device_declines_with_a_distinct_reason() {
        use crate::budget::MemProbe;
        use std::sync::atomic::{AtomicU64, Ordering};
        let free = Arc::new(AtomicU64::new(1000));
        let f = free.clone();
        let probe: MemProbe = Arc::new(move || Some((f.load(Ordering::Relaxed), 2000)));
        let mut r = reg(AccelCache::On, 1000, 0).with_mem_probe(probe);

        assert_eq!(r.admit(&key("a"), 400, 10, 1), Admission::Resident);
        free.store(200, Ordering::Relaxed);
        r.budget_mut().refresh();
        // 800 B fits CUDA_VRAM (1000 B) but not the live 600 B ceiling.
        let a = r.admit(&key("b"), 800, 10, 1);
        assert_eq!(a.decline_reason(), Some(DeclineReason::ElasticBudget));
        let t = r.telemetry();
        assert!(t.elastic_clamps >= 1);
        assert!(t.capacity_bytes < t.configured_capacity_bytes);
    }

    #[test]
    fn hottest_shadows_rank_by_heat_and_exclude_residents() {
        // Room for one block: every newcomer is refused by the protected hot one,
        // which is exactly the situation the heat pump exists for.
        let mut r = reg(AccelCache::On, 500, 0);
        assert_eq!(r.admit(&key("hot"), 400, 10, 1), Admission::Resident);
        for _ in 0..5 {
            r.admit(&key("hot"), 400, 10, 1);
        }
        for _ in 0..2 {
            assert_eq!(
                r.admit(&key("warm"), 400, 10, 1).decline_reason(),
                Some(DeclineReason::HeatProtected)
            );
        }
        assert!(matches!(
            r.admit(&key("cold"), 400, 10, 1),
            Admission::Declined { .. }
        ));

        let ranked = r.hottest_shadows(0.5, 8);
        let names: Vec<&str> = ranked.iter().map(|(k, _)| k.name.as_str()).collect();
        assert_eq!(names, vec!["warm", "cold"], "hottest first, residents excluded");
        assert!(ranked[0].1 > ranked[1].1, "two refusals outrank one");

        // The prefetch floor separates "asked twice" from "scanned once": only
        // the twice-refused key is worth an unsolicited upload.
        let pumpable = r.hottest_shadows(PREFETCH_HEAT_FLOOR, 8);
        assert_eq!(pumpable.len(), 1);
        assert_eq!(pumpable[0].0.name, "warm");

        // A floor above what any of them earned yields nothing, and so does a
        // zero limit — the pump must be able to say "nothing worth pulling in".
        assert!(r.hottest_shadows(PREFETCH_HEAT_FLOOR * 10.0, 8).is_empty());
        assert!(r.hottest_shadows(0.0, 0).is_empty());
        assert_eq!(r.hottest_shadows(0.0, 1).len(), 1, "limit truncates");

        // Forgetting a key drops its evidence exactly once.
        assert!(r.forget_shadow(&key("cold")));
        assert!(!r.forget_shadow(&key("cold")));
        assert_eq!(r.hottest_shadows(0.0, 8).len(), 1);
    }

    #[test]
    fn the_pump_fills_slack_only_and_never_evicts() {
        let mut r = reg(AccelCache::On, 500, 0);
        assert_eq!(r.admit(&key("hot"), 400, 10, 1), Admission::Resident);
        for _ in 0..5 {
            r.admit(&key("hot"), 400, 10, 1);
        }
        // One refusal leaves evidence, so the key is a pump candidate.
        for _ in 0..2 {
            r.admit(&key("warm"), 400, 10, 1);
        }

        // 100 B free, 400 B wanted: a query would evict a victim here, the pump
        // declines instead — an unsolicited upload must never displace a block a
        // query is using (measured cost of allowing it: +71% bytes, p99 × 1.41).
        assert_eq!(
            r.admit_prefetch(&key("warm"), 400, 10, 1).decline_reason(),
            Some(DeclineReason::NoSlack)
        );
        assert!(r.is_resident(&key("hot")), "the victim is still there");
        assert_eq!(r.telemetry().evictions, 0);

        // Releasing the dropped collection's block is where slack comes from;
        // afterwards the very same pump request succeeds.
        assert_eq!(r.release(&key("hot")), 400);
        assert!(!r.is_resident(&key("hot")));
        assert_eq!(r.release(&key("hot")), 0, "releasing twice frees nothing");
        assert_eq!(r.admit_prefetch(&key("warm"), 400, 10, 1), Admission::Resident);
        assert!(r.is_resident(&key("warm")));

        // A pump hit consumes heat without earning any: the block's stored heat
        // is what the workload earned, so pumping cannot inflate a ranking.
        let before = r.telemetry().hottest_milli;
        assert_eq!(r.admit_prefetch(&key("warm"), 400, 10, 1), Admission::Resident);
        assert_eq!(r.admit_prefetch(&key("warm"), 400, 10, 1), Admission::Resident);
        assert_eq!(r.telemetry().hottest_milli, before, "pumps never add heat");
    }

    #[test]
    fn blocks_are_reported_in_stable_order() {
        let mut r = reg(AccelCache::On, 4000, 0);
        r.admit(&ResidentKey::new(ResidentKind::RightSeries, "z"), 100, 1, 1);
        r.admit(&ResidentKey::new(ResidentKind::Corpus, "b"), 100, 1, 1);
        r.admit(&ResidentKey::new(ResidentKind::Corpus, "a"), 100, 1, 1);
        let names: Vec<_> = r.blocks().into_iter().map(|b| b.name).collect();
        assert_eq!(names, vec!["a", "b", "z"]);
    }
}
