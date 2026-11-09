//! M3: optional, discardable `ActiveNodeBitmap` accelerator.
//!
//! This module deliberately does **not** sit on the default traversal path.
//! [`crate::endpoint::NodeLifetime`] remains the single source of truth; the
//! bitmap is an L3-style derived cache that a planner may build only when a
//! query touches enough edges for its O(V) construction cost to amortize.
//!
//! Three invariants (review R1 / design §23.4):
//!
//! 1. **Key includes the snapshot.** A bitmap built from node version S must
//!    never be reused for S+1, otherwise a node correction silently disappears.
//! 2. **Discardable.** [`BitmapCache::invalidate`] removes the cache entirely;
//!    [`BitmapCache::active_at`] then answers from the fact array. Every query
//!    result must be identical with and without the cache.
//! 3. **Never built for point queries.** [`BitmapPlanner`] gates construction
//!    on `E_selected / V`. If all nodes are alive at `t`, the O(1) node-lifetime
//!    bounding box wins and no bitmap is built at all.

use crate::endpoint::NodeLifetime;
use crate::error::{GtvError, Result};

/// A bitmap is valid for exactly one `(snapshot, business time, partition)`.
///
/// `snapshot_id` is mandatory: day-bucketing `business_time` alone is only a
/// candidate set and is explicitly rejected by the design (§19.2 / §6).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BitmapKey {
    pub snapshot_id: u64,
    pub business_time: i64,
    pub partition_id: u32,
}

impl BitmapKey {
    pub fn new(snapshot_id: u64, business_time: i64, partition_id: u32) -> Self {
        Self {
            snapshot_id,
            business_time,
            partition_id,
        }
    }
}

/// Immutable bitset over node ids. One 64-bit word per 64 nodes, so a full
/// sweep pays one load + one shift per endpoint test instead of two `i64`
/// column reads and two comparisons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveNodeBitmap {
    key: BitmapKey,
    words: Vec<u64>,
    node_count: usize,
    active_count: usize,
}

impl ActiveNodeBitmap {
    /// Build from the fact source. O(V), done at most once per key.
    pub fn build(key: BitmapKey, lifetime: &NodeLifetime) -> Result<Self> {
        let n = lifetime.len();
        let (from, to) = lifetime.columns();
        let mut words = vec![0u64; n / 64 + usize::from(n % 64 != 0)];
        let mut active_count = 0usize;
        let t = key.business_time;
        for i in 0..n {
            if from[i] <= t && t < to[i] {
                words[i >> 6] |= 1u64 << (i & 63);
                active_count += 1;
            }
        }
        Ok(Self {
            key,
            words,
            node_count: n,
            active_count,
        })
    }

    pub fn key(&self) -> &BitmapKey {
        &self.key
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Bytes of resident state (header excluded).
    pub fn bytes(&self) -> usize {
        self.words.len() * 8
    }

    pub fn matches(&self, key: &BitmapKey) -> bool {
        &self.key == key
    }

    /// Hot lookup. The caller has already validated the key through
    /// [`BitmapCache::get_or_build`]; an out-of-range id is unconstrained
    /// (`true`), matching `NodeLifetime::active_at`.
    #[inline]
    pub fn active_at_unchecked(&self, node: u64) -> bool {
        let i = node as usize;
        if i >= self.node_count {
            return true;
        }
        (self.words[i >> 6] >> (i & 63)) & 1 == 1
    }

    /// Safe lookup with an explicit key check. Used by tests and by callers
    /// that cannot prove the cache key is still current.
    pub fn checked_active_at(&self, node: u64, key: &BitmapKey) -> Result<bool> {
        if !self.matches(key) {
            return Err(GtvError::InvalidArgument(format!(
                "ActiveNodeBitmap key mismatch: built for {:?}, queried with {:?}",
                self.key, key
            )));
        }
        Ok(self.active_at_unchecked(node))
    }

    /// §23.4: prove the derived cache agrees with the fact source for *every*
    /// node. Cheap enough for tests and for an explicit `VERIFY` SQL call.
    pub fn verify_against(&self, lifetime: &NodeLifetime) -> Result<()> {
        if lifetime.len() != self.node_count {
            return Err(GtvError::InvalidArgument(format!(
                "bitmap covers {} nodes but the lifetime covers {}",
                self.node_count,
                lifetime.len()
            )));
        }
        let t = self.key.business_time;
        for i in 0..self.node_count {
            let got = self.active_at_unchecked(i as u64);
            let want = lifetime.active_at(i as u64, t);
            if got != want {
                return Err(GtvError::InvalidArgument(format!(
                    "bitmap disagrees with NodeLifetime at node {i}, t={t}: {got} vs {want}"
                )));
            }
        }
        Ok(())
    }
}

/// Planner gate. The ratio is calibrated by `bench_m3_gate`; the defaults are
/// conservative so a point query can never accidentally pay O(V).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitmapPlanner {
    /// Build only when `selected_edges >= ratio * node_count`.
    pub amortization_ratio: usize,
    /// Absolute floor: even a dense query below this edge count stays direct.
    pub min_selected_edges: usize,
    /// Hard ceiling on the resident cache, 0 = unlimited.
    pub max_cache_bytes: usize,
}

impl Default for BitmapPlanner {
    fn default() -> Self {
        Self {
            amortization_ratio: 8,
            min_selected_edges: 1024,
            max_cache_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Why the planner did or did not build a bitmap. Exposed so the SQL surface
/// and the benchmark can report the decision instead of guessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitmapDecision {
    /// All nodes alive: the O(1) bounding box already short-circuits.
    AllActive,
    /// Query too small: O(V) build would lose to direct reads.
    TooSmall,
    /// Bitmap would exceed the configured resident budget.
    OverBudget,
    /// Build (or reuse) the derived cache.
    Build,
}

impl BitmapPlanner {
    pub fn decide(&self, lifetime: &NodeLifetime, t: i64, selected_edges: usize) -> BitmapDecision {
        if lifetime.all_active_at(t) {
            return BitmapDecision::AllActive;
        }
        let v = lifetime.len().max(1);
        if selected_edges < self.min_selected_edges
            || selected_edges < self.amortization_ratio.saturating_mul(v)
        {
            return BitmapDecision::TooSmall;
        }
        let bytes = (v / 64 + usize::from(v % 64 != 0)) * 8;
        if self.max_cache_bytes > 0 && bytes > self.max_cache_bytes {
            return BitmapDecision::OverBudget;
        }
        BitmapDecision::Build
    }
}

/// Single-slot derived cache. `None` means "answer from the fact source" —
/// dropping it is always semantically safe.
#[derive(Debug)]
pub struct BitmapCache {
    slot: Option<ActiveNodeBitmap>,
    planner: BitmapPlanner,
    /// Telemetry: how many times a build was actually paid for.
    builds: usize,
    hits: usize,
    misses: usize,
}

impl BitmapCache {
    pub fn new(planner: BitmapPlanner) -> Self {
        Self {
            slot: None,
            planner,
            builds: 0,
            hits: 0,
            misses: 0,
        }
    }

    pub fn planner(&self) -> &BitmapPlanner {
        &self.planner
    }

    pub fn builds(&self) -> usize {
        self.builds
    }
    pub fn hits(&self) -> usize {
        self.hits
    }
    pub fn misses(&self) -> usize {
        self.misses
    }

    /// Drop the derived state. Query results must be byte-for-byte unchanged.
    pub fn invalidate(&mut self) {
        self.slot = None;
    }

    /// Planner-gated lookup. Returns `Some(&bitmap)` only when construction
    /// amortizes; otherwise the caller uses `NodeLifetime` directly.
    pub fn get_or_build(
        &mut self,
        key: &BitmapKey,
        lifetime: &NodeLifetime,
        selected_edges: usize,
    ) -> Result<Option<&ActiveNodeBitmap>> {
        let t = key.business_time;
        if self.planner.decide(lifetime, t, selected_edges) != BitmapDecision::Build {
            self.misses += 1;
            return Ok(None);
        }
        if self.slot.as_ref().is_some_and(|b| b.matches(key)) {
            self.hits += 1;
            return Ok(self.slot.as_ref());
        }
        let bitmap = ActiveNodeBitmap::build(key.clone(), lifetime)?;
        self.slot = Some(bitmap);
        self.builds += 1;
        self.misses += 1;
        Ok(self.slot.as_ref())
    }

    /// Resolve one endpoint test through the cache, falling back to the fact
    /// source. This is the *only* correctness path: cache presence changes
    /// cost, never the answer.
    #[inline]
    pub fn active_at(
        &self,
        node: u64,
        t: i64,
        key: &BitmapKey,
        lifetime: &NodeLifetime,
    ) -> bool {
        match &self.slot {
            Some(b) if b.matches(key) => b.active_at_unchecked(node),
            _ => lifetime.active_at(node, t),
        }
    }

    /// The planner decision for a query, without touching the cache.
    pub fn decision(&self, lifetime: &NodeLifetime, t: i64, selected_edges: usize) -> BitmapDecision {
        self.planner.decide(lifetime, t, selected_edges)
    }
}

impl Default for BitmapCache {
    fn default() -> Self {
        Self::new(BitmapPlanner::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lifetime(n: usize, expire_every: usize) -> NodeLifetime {
        let from = vec![0i64; n];
        let mut to = vec![i64::MAX; n];
        for i in (0..n).step_by(expire_every) {
            to[i] = 500;
        }
        NodeLifetime::new(from, to).unwrap()
    }

    fn key(t: i64) -> BitmapKey {
        BitmapKey::new(7, t, 0)
    }

    #[test]
    fn bitmap_agrees_with_fact_source_for_every_node() {
        let lt = lifetime(1000, 7);
        for t in [0, 499, 500, 501, 9999] {
            let b = ActiveNodeBitmap::build(key(t), &lt).unwrap();
            b.verify_against(&lt).unwrap();
            assert_eq!(
                b.active_count(),
                (0u64..1000).filter(|&i| lt.active_at(i, t)).count()
            );
        }
    }

    #[test]
    fn stale_snapshot_key_is_rejected() {
        let lt = lifetime(128, 3);
        let b = ActiveNodeBitmap::build(BitmapKey::new(1, 10, 0), &lt).unwrap();
        assert!(b.checked_active_at(5, &BitmapKey::new(2, 10, 0)).is_err());
        assert!(b.checked_active_at(5, &BitmapKey::new(1, 11, 0)).is_err());
        assert!(b.checked_active_at(5, &BitmapKey::new(1, 10, 1)).is_err());
        assert!(b.checked_active_at(5, &BitmapKey::new(1, 10, 0)).is_ok());
    }

    #[test]
    fn out_of_range_node_is_unconstrained() {
        let lt = lifetime(64, 2);
        let b = ActiveNodeBitmap::build(key(10), &lt).unwrap();
        assert!(b.active_at_unchecked(9999));
    }

    #[test]
    fn planner_never_builds_for_point_queries_or_all_active() {
        let lt = lifetime(10_000, 3);
        let p = BitmapPlanner::default();
        assert_eq!(p.decide(&lt, 600, 100), BitmapDecision::TooSmall);
        assert_eq!(p.decide(&lt, 600, 10), BitmapDecision::TooSmall);
        // At t=10 every node is alive, so the O(1) bounding box wins outright.
        assert_eq!(p.decide(&lt, 10, 100_000), BitmapDecision::AllActive);
        let forever = NodeLifetime::forever(10_000);
        assert_eq!(p.decide(&forever, 10, 1_000_000), BitmapDecision::AllActive);
    }

    #[test]
    fn planner_builds_for_dense_sweeps_and_respects_budget() {
        let lt = lifetime(1000, 3);
        let p = BitmapPlanner::default();
        // t=600 expires some nodes, so the O(1) bounding box cannot win.
        assert_eq!(p.decide(&lt, 600, 100_000), BitmapDecision::Build);
        let tiny = BitmapPlanner {
            max_cache_bytes: 8,
            ..BitmapPlanner::default()
        };
        assert_eq!(tiny.decide(&lt, 600, 100_000), BitmapDecision::OverBudget);
    }

    #[test]
    fn invalidating_the_cache_changes_no_answer() {
        let lt = lifetime(2048, 5);
        let k = key(600);
        let mut cache = BitmapCache::default();
        assert!(cache.get_or_build(&k, &lt, 100_000).unwrap().is_some());
        let with_cache: Vec<bool> = (0..2048)
            .map(|i| cache.active_at(i, 600, &k, &lt))
            .collect();
        cache.invalidate();
        let without_cache: Vec<bool> = (0..2048)
            .map(|i| cache.active_at(i, 600, &k, &lt))
            .collect();
        assert_eq!(with_cache, without_cache);
        assert_eq!(
            without_cache,
            (0..2048).map(|i| lt.active_at(i, 600)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn snapshot_change_forces_a_rebuild() {
        let lt = lifetime(512, 4);
        let mut cache = BitmapCache::default();
        let k1 = BitmapKey::new(1, 600, 0);
        let k2 = BitmapKey::new(2, 600, 0);
        assert!(cache.get_or_build(&k1, &lt, 50_000).unwrap().is_some());
        assert!(cache.get_or_build(&k1, &lt, 50_000).unwrap().is_some());
        assert_eq!(cache.builds(), 1);
        assert_eq!(cache.hits(), 1);
        assert!(cache.get_or_build(&k2, &lt, 50_000).unwrap().is_some());
        assert_eq!(cache.builds(), 2);
    }
}
