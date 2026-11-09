//! The resident cache: what makes a device copy worth having.
//!
//! # Why a cache and not an allocation
//!
//! Measured on the C0 probe: a 153 MiB pinned H2D copy runs at 12 GB/s, so one
//! upload costs more than the kernel it feeds unless the copy is reused. Break
//! even for the valid-time filter was 2.1 invocations. Every design decision
//! here follows from that number — the cache exists to turn "upload per query"
//! into "upload per snapshot".
//!
//! # The three invariants
//!
//! 1. **Keyed by snapshot and projection.** `business_t` is not part of the key.
//!    A graph copy is a function of the snapshot *and of which columns were
//!    asked for*; the time predicate is an argument to each operator. Including
//!    `t` would re-upload the whole graph for every query — the exact opposite
//!    of the reason to have a device copy. Including the projection does the
//!    opposite: it lets a topology-only copy and a temporal copy coexist, at
//!    6 and 23 bytes per edge, instead of one of them evicting the other.
//! 2. **Pinning is reference counting.** A handle handed to a query is an
//!    `Arc<ResidentGraph>` clone; the cache holds one reference itself. An entry
//!    whose count is above one cannot be freed, so it is not evicted — the
//!    query keeps reading valid memory and the budget treats it as a soft cap
//!    rather than failing the query.
//! 3. **Invalidation is explicit, and it is the publish path's job.** A lookup
//!    is answered only for the exact snapshot it names, so an older copy can
//!    never be served to a newer query — and it is not "stale" for a replay that
//!    asks for it by id, which is why the cache does not drop old snapshots on
//!    its own. What it does do is drop them when the caller says so
//!    ([`ResidentCache::invalidate_snapshot`], called from publish / retract /
//!    GC) and evict them under budget pressure. A query already holding the
//!    `Arc` keeps a consistent view of the snapshot it started on, which is the
//!    correct answer for that query.
//!
//! Nothing here can fail a query: the worst case is a miss, and a miss is a
//! slower correct answer.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::api::{
    Accelerator, ResidentGraph, ResidentVectors, SnapshotId, SnapshotView, VectorView,
};
use crate::error::AcceleratorError;
use crate::metrics::ColumnSet;

/// A handle the cache can account for.
pub trait Resident: Send + Sync {
    fn bytes(&self) -> usize;
    fn snapshot_id(&self) -> SnapshotId;
}

impl Resident for ResidentGraph {
    fn bytes(&self) -> usize {
        ResidentGraph::bytes(self)
    }
    fn snapshot_id(&self) -> SnapshotId {
        ResidentGraph::snapshot_id(self)
    }
}

impl Resident for ResidentVectors {
    fn bytes(&self) -> usize {
        ResidentVectors::bytes(self)
    }
    fn snapshot_id(&self) -> SnapshotId {
        ResidentVectors::snapshot_id(self)
    }
}

/// Counters a benchmark or an operator test asserts on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Lookups answered without touching the backend.
    pub hits: u64,
    /// Lookups that had to load (or that the backend declined).
    pub misses: u64,
    /// Handles the backend actually produced.
    pub loads: u64,
    /// Loads the backend declined (`Ok(None)`); residency is not supported.
    pub declined: u64,
    /// Handles released back to the backend.
    pub releases: u64,
    /// Entries dropped to stay inside the budget.
    pub evictions: u64,
    /// Entries dropped because a newer snapshot arrived.
    pub evicted_stale: u64,
    /// Eviction candidates skipped because a query still holds them.
    pub skipped_pinned: u64,
    /// Times the budget was exceeded with nothing evictable. Not an error: the
    /// next release brings it back under.
    pub over_budget: u64,
    /// Bytes currently accounted for.
    pub resident_bytes: usize,
}

struct Slot<H> {
    handle: Arc<H>,
    last_used: u64,
}

/// LRU over handles, with `Arc` as the pin.
///
/// Generic because the two resident objects (graph, vectors) have different
/// keys and lifetimes but identical eviction rules; writing the rules twice is
/// how the two copies drift apart.
struct Lru<K, H> {
    map: BTreeMap<K, Slot<H>>,
    /// Bytes of entries in `map`.
    bytes: usize,
    /// Bytes of handles evicted while pinned and not yet releasable.
    pending: Vec<Arc<H>>,
    clock: u64,
    max_bytes: Option<usize>,
    stats: CacheStats,
}

impl<K: Ord + Copy + std::fmt::Debug, H: Resident> Lru<K, H> {
    fn new(max_bytes: Option<usize>) -> Self {
        Self {
            map: BTreeMap::new(),
            bytes: 0,
            pending: Vec::new(),
            clock: 0,
            max_bytes,
            stats: CacheStats::default(),
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn get(&mut self, key: &K) -> Option<Arc<H>> {
        let now = self.tick();
        let slot = self.map.get_mut(key)?;
        slot.last_used = now;
        Some(Arc::clone(&slot.handle))
    }

    /// Insert, then make room. The new entry is pinned by the caller's `Arc`,
    /// so it is never the one evicted.
    fn insert(&mut self, key: K, handle: Arc<H>) {
        let bytes = handle.bytes();
        let now = self.tick();
        if let Some(old) = self.map.insert(
            key,
            Slot {
                handle: Arc::clone(&handle),
                last_used: now,
            },
        ) {
            self.bytes = self.bytes.saturating_sub(old.handle.bytes());
            self.retire(old.handle);
        }
        self.bytes += bytes;
        self.enforce_budget();
    }

    fn enforce_budget(&mut self) {
        let Some(max) = self.max_bytes else { return };
        while self.bytes > max {
            // Oldest first, and never one a query is holding: freeing it would
            // hand the query a dangling device pointer.
            let victim = self
                .map
                .iter()
                .filter(|(_, s)| Arc::strong_count(&s.handle) == 1)
                .min_by_key(|(_, s)| s.last_used)
                .map(|(k, _)| *k);
            let Some(k) = victim else {
                self.stats.skipped_pinned += 1;
                self.stats.over_budget += 1;
                return;
            };
            if let Some(slot) = self.map.remove(&k) {
                self.bytes = self.bytes.saturating_sub(slot.handle.bytes());
                self.stats.evictions += 1;
                self.retire(slot.handle);
            }
        }
    }

    /// Drop every entry for a snapshot. Pinned handles survive as `Arc`s but
    /// leave the index, so they can never be served again.
    fn remove_snapshot(&mut self, snapshot_id: SnapshotId) -> usize {
        let victims: Vec<K> = self
            .map
            .iter()
            .filter(|(_, s)| s.handle.snapshot_id() == snapshot_id)
            .map(|(k, _)| *k)
            .collect();
        for k in &victims {
            if let Some(slot) = self.map.remove(k) {
                self.bytes = self.bytes.saturating_sub(slot.handle.bytes());
                self.retire(slot.handle);
            }
        }
        victims.len()
    }

    /// Drop every entry for keys the predicate rejects.
    fn remove_where(&mut self, drop_it: impl Fn(&K) -> bool) -> usize {
        let victims: Vec<K> = self.map.keys().copied().filter(|k| drop_it(k)).collect();
        for k in &victims {
            if let Some(slot) = self.map.remove(k) {
                self.bytes = self.bytes.saturating_sub(slot.handle.bytes());
                self.retire(slot.handle);
            }
        }
        victims.len()
    }

    /// Queue a handle for release. Deferred rather than immediate because a
    /// query may still be reading it; [`Lru::drain_releasable`] frees it as soon
    /// as the last borrower lets go.
    fn retire(&mut self, handle: Arc<H>) {
        if Arc::strong_count(&handle) > 1 {
            self.stats.skipped_pinned += 1;
        }
        self.pending.push(handle);
    }

    /// Release deferred handles whose last borrower has let go.
    ///
    /// Returns how many were released to the backend.
    fn drain_releasable(&mut self, rel: &mut impl FnMut(&H)) -> usize {
        let mut released = 0;
        let mut keep = Vec::with_capacity(self.pending.len());
        for h in self.pending.drain(..) {
            match Arc::try_unwrap(h) {
                Ok(owned) => {
                    rel(&owned);
                    released += 1;
                }
                Err(shared) => keep.push(shared),
            }
        }
        self.pending = keep;
        self.stats.releases += released as u64;
        released
    }

    /// Handles evicted-while-pinned and still referenced elsewhere.
    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            resident_bytes: self.bytes,
            ..self.stats
        }
    }
}

/// What identifies a graph copy: the snapshot *and* the columns in it.
///
/// The projection belongs in the key for the same reason the snapshot id does —
/// two copies that answer different questions are different objects. Keying on
/// the snapshot alone would hand a topology-only copy to an operator that reads
/// `valid_from`, and the failure would be a wrong answer rather than a miss.
pub type GraphKey = (SnapshotId, ColumnSet);

/// Resident copies of graphs, one per (snapshot, projection).
pub struct ResidentCache {
    backend: Arc<dyn Accelerator>,
    lru: Lru<GraphKey, ResidentGraph>,
}

impl ResidentCache {
    pub fn new(backend: Arc<dyn Accelerator>, max_device_bytes: Option<usize>) -> Self {
        Self {
            backend,
            lru: Lru::new(max_device_bytes),
        }
    }

    pub fn backend(&self) -> &Arc<dyn Accelerator> {
        &self.backend
    }

    /// Get the snapshot's handle, uploading on a miss.
    ///
    /// `Ok(None)` means the backend does not do residency: the caller runs on
    /// the CPU and does not ask again.
    pub fn acquire(
        &mut self,
        view: SnapshotView<'_>,
    ) -> Result<Option<Arc<ResidentGraph>>, AcceleratorError> {
        self.lru.drain_releasable(&mut |h| self.backend.release_graph(h));
        let key: GraphKey = (view.snapshot_id, view.required_columns());
        if let Some(h) = self.lru.get(&key) {
            self.lru.stats.hits += 1;
            // A hit must never disagree about the snapshot or the projection; if
            // it did, the key would be wrong and every answer after it would be
            // too. Checked rather than assumed because a backend is free to
            // return a *wider* copy than asked for, and that stays a hit.
            debug_assert_eq!(h.snapshot_id(), view.snapshot_id);
            debug_assert!(h.check_columns(key.1).is_ok());
            return Ok(Some(h));
        }
        self.lru.stats.misses += 1;

        let Some(handle) = self.backend.load_graph(view)? else {
            self.lru.stats.declined += 1;
            return Ok(None);
        };
        self.lru.stats.loads += 1;
        let arc = Arc::new(handle);
        self.lru.insert(key, Arc::clone(&arc));
        // `insert` may have evicted an unpinned entry; hand it back now rather
        // than waiting for the next call, so `stats().releases` reflects the
        // eviction that just happened.
        self.lru
            .drain_releasable(&mut |h| self.backend.release_graph(h));
        Ok(Some(arc))
    }

    /// Forget a snapshot. Returns how many entries left the index.
    pub fn invalidate_snapshot(&mut self, snapshot_id: SnapshotId) -> usize {
        let n = self.lru.remove_snapshot(snapshot_id);
        self.lru
            .drain_releasable(&mut |h| self.backend.release_graph(h));
        n
    }

    /// Forget everything. Used when a segment is collected or a test wants a
    /// cold cache.
    pub fn invalidate_all(&mut self) -> usize {
        let snapshots: Vec<SnapshotId> = self.lru.map.keys().map(|k| k.0).collect();
        let mut n = 0;
        for s in snapshots {
            n += self.lru.remove_snapshot(s);
        }
        self.lru
            .drain_releasable(&mut |h| self.backend.release_graph(h));
        n
    }

    /// Re-apply the budget and retry releases that a live borrower blocked.
    ///
    /// Called after a lease is dropped: the budget is a soft cap during a
    /// query (a pinned copy is not evicted) but it converges as soon as nothing
    /// is reading, which is what keeps "over budget" from becoming an error.
    pub fn trim(&mut self) -> usize {
        let released = self
            .lru
            .drain_releasable(&mut |h| self.backend.release_graph(h));
        self.lru.enforce_budget();
        released
            + self
                .lru
                .drain_releasable(&mut |h| self.backend.release_graph(h))
    }

    /// Whether *any* projection of this snapshot is resident.
    pub fn is_resident(&self, snapshot_id: SnapshotId) -> bool {
        self.lru.map.keys().any(|k| k.0 == snapshot_id)
    }

    pub fn len(&self) -> usize {
        self.lru.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lru.map.is_empty()
    }

    pub fn resident_bytes(&self) -> usize {
        self.lru.bytes
    }

    /// Handles evicted while a query held them and not yet freed.
    pub fn pending_releases(&self) -> usize {
        self.lru.pending_count()
    }

    pub fn stats(&self) -> CacheStats {
        self.lru.stats()
    }
}

impl Drop for ResidentCache {
    fn drop(&mut self) {
        // Best effort: a handle another thread still holds cannot be freed, and
        // freeing it would be worse than leaking.
        self.lru.drain_releasable(&mut |h| self.backend.release_graph(h));
    }
}

/// Resident copies of vector spaces, keyed by `(space, snapshot)`.
///
/// Separate from [`ResidentCache`] because the two objects have different
/// identities and refresh rates: graphs change per snapshot, vectors change per
/// model *and* snapshot, and a new model version must not invalidate the graph.
pub struct VectorCache {
    backend: Arc<dyn Accelerator>,
    lru: Lru<(u32, SnapshotId), ResidentVectors>,
}

impl VectorCache {
    pub fn new(backend: Arc<dyn Accelerator>, max_device_bytes: Option<usize>) -> Self {
        Self {
            backend,
            lru: Lru::new(max_device_bytes),
        }
    }

    pub fn acquire(
        &mut self,
        view: VectorView<'_>,
    ) -> Result<Option<Arc<ResidentVectors>>, AcceleratorError> {
        self.lru.drain_releasable(&mut |h| self.backend.release_vectors(h));
        let key = (view.vector_space_id, view.snapshot_id);
        if let Some(h) = self.lru.get(&key) {
            self.lru.stats.hits += 1;
            if h.dim() != view.dim || h.metric() != view.metric {
                // Same key, different shape: the key is too weak. Refuse rather
                // than answer with vectors of another model.
                return Err(AcceleratorError::InvalidArgument(format!(
                    "space {} at snapshot {} is resident as {}d {:?}, request is {}d {:?}",
                    view.vector_space_id,
                    view.snapshot_id,
                    h.dim(),
                    h.metric(),
                    view.dim,
                    view.metric
                )));
            }
            return Ok(Some(h));
        }
        self.lru.stats.misses += 1;
        let Some(handle) = self.backend.load_vectors(view)? else {
            self.lru.stats.declined += 1;
            return Ok(None);
        };
        self.lru.stats.loads += 1;
        let arc = Arc::new(handle);
        self.lru.insert(key, Arc::clone(&arc));
        Ok(Some(arc))
    }

    pub fn invalidate_snapshot(&mut self, snapshot_id: SnapshotId) -> usize {
        let n = self.lru.remove_snapshot(snapshot_id);
        self.lru
            .drain_releasable(&mut |h| self.backend.release_vectors(h));
        n
    }

    /// Drop one vector space at every snapshot. Used when a model version is
    /// retired and its space can never be queried again.
    pub fn invalidate_space(&mut self, vector_space_id: u32) -> usize {
        let n = self.lru.remove_where(|k| k.0 == vector_space_id);
        self.lru
            .drain_releasable(&mut |h| self.backend.release_vectors(h));
        n
    }

    pub fn trim(&mut self) -> usize {
        let released = self
            .lru
            .drain_releasable(&mut |h| self.backend.release_vectors(h));
        self.lru.enforce_budget();
        released
            + self
                .lru
                .drain_releasable(&mut |h| self.backend.release_vectors(h))
    }

    pub fn resident_bytes(&self) -> usize {
        self.lru.bytes
    }

    pub fn stats(&self) -> CacheStats {
        self.lru.stats()
    }
}

impl Drop for VectorCache {
    fn drop(&mut self) {
        self.lru
            .drain_releasable(&mut |h| self.backend.release_vectors(h));
    }
}
