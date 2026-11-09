//! Endpoint-lifetime policy for ACTIVE GRAPH traversal (minimal-update M0).
//!
//! A node's lifecycle may be corrected retroactively, and some edge types must
//! then stop being visible. This module keeps that rule **out of the CSR**:
//!
//! * `TemporalCSR` / `neighbors()` are untouched — HISTORICAL GRAPH keeps the
//!   exact zero-copy, two-comparison hot loop (`csr.rs:883-908`).
//! * System time is resolved **once per snapshot**, by choosing a
//!   [`NodeLifetime`] file, not per row. Nothing here knows about
//!   `system_from`/`system_to`.
//! * The three business policies (ClampToEndpoints /
//!   PreserveHistoricalReference / External) collapse to **one bit per edge
//!   type**: the latter two behave identically inside the kernel.
//!
//! The only per-edge cost is a single indexed read of a fixed-width `i64`
//! array, and only for clamped edge types. A segment-level bounding box usually
//! short-circuits that entirely.

use crate::csr::{Neighbor, Neighbors, TemporalCSR};
use crate::error::{GtvError, Result};

/// Sentinel for a node (or edge) that has no lower / upper time bound.
pub const OPEN_FROM: i64 = i64::MIN;
pub const OPEN_TO: i64 = i64::MAX;

/// `1` = `ClampToEndpoints`; `0` = kernel imposes no endpoint constraint
/// (`PreserveHistoricalReference` and `External` are distinguished by the
/// business layer's subscription to invalidation events, not by traversal).
///
/// Covers edge types `0..=63`. `clamp()` rejects anything above that rather
/// than silently dropping the constraint — a silent drop would return *extra*
/// edges, i.e. a correctness bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndpointPolicy {
    clamp_types: u64,
}

impl EndpointPolicy {
    pub const NONE: Self = Self { clamp_types: 0 };
    pub const ALL: Self = Self {
        clamp_types: u64::MAX,
    };

    #[inline]
    pub fn none() -> Self {
        Self::NONE
    }

    #[inline]
    pub fn all() -> Self {
        Self::ALL
    }

    /// Build a policy from the edge types that must be clamped.
    pub fn clamp(edge_types: &[u16]) -> Result<Self> {
        let mut mask = 0u64;
        for &et in edge_types {
            if et >= 64 {
                return Err(GtvError::InvalidArgument(format!(
                    "edge_type {et} exceeds the 0..=63 clamp bitmap; widen the mask in EndpointPolicy"
                )));
            }
            mask |= 1u64 << et;
        }
        Ok(Self {
            clamp_types: mask,
        })
    }

    #[inline]
    pub fn is_clamped(&self, edge_type: u16) -> bool {
        edge_type < 64 && (self.clamp_types >> edge_type) & 1 == 1
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.clamp_types == 0
    }

    pub fn mask(&self) -> u64 {
        self.clamp_types
    }
}

/// An immutable, fixed-width snapshot of node lifecycles, indexed by `node_id`.
///
/// A retroactive `valid_to` correction publishes a **new** `NodeLifetime` under
/// a new commit id; the old one stays reachable through `AS OF SYSTEM`. That is
/// the whole bitemporal mechanism for nodes — no per-row system interval, no
/// version-id index, no retract lookup on the read path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeLifetime {
    valid_from: Vec<i64>,
    valid_to: Vec<i64>,
    max_valid_from: i64,
    min_valid_to: i64,
}

impl NodeLifetime {
    pub fn new(valid_from: Vec<i64>, valid_to: Vec<i64>) -> Result<Self> {
        if valid_from.len() != valid_to.len() {
            return Err(GtvError::InvalidArgument(
                "node lifetime columns have mismatched lengths".into(),
            ));
        }
        for (f, t) in valid_from.iter().zip(&valid_to) {
            if f > t {
                return Err(GtvError::InvalidArgument(format!(
                    "node lifetime [{f}, {t}) is inverted"
                )));
            }
        }
        let max_valid_from = valid_from.iter().copied().max().unwrap_or(OPEN_FROM);
        let min_valid_to = valid_to.iter().copied().min().unwrap_or(OPEN_TO);
        Ok(Self {
            valid_from,
            valid_to,
            max_valid_from,
            min_valid_to,
        })
    }

    /// Every node is alive on the whole time axis.
    pub fn forever(node_count: usize) -> Self {
        Self {
            valid_from: vec![OPEN_FROM; node_count],
            valid_to: vec![OPEN_TO; node_count],
            max_valid_from: OPEN_FROM,
            min_valid_to: OPEN_TO,
        }
    }

    pub fn len(&self) -> usize {
        self.valid_from.len()
    }

    pub fn is_empty(&self) -> bool {
        self.valid_from.is_empty()
    }

    /// Borrow both fixed-width columns (used by the on-disk snapshot codec).
    #[inline]
    pub fn columns(&self) -> (&[i64], &[i64]) {
        (&self.valid_from, &self.valid_to)
    }

    #[inline]
    pub fn range(&self, node: u64) -> Option<(i64, i64)> {
        let i = node as usize;
        if i >= self.valid_from.len() {
            return None;
        }
        Some((self.valid_from[i], self.valid_to[i]))
    }

    /// An id with no lifetime entry is unconstrained (never silently dropped).
    #[inline]
    pub fn active_at(&self, node: u64, t: i64) -> bool {
        self.range(node)
            .map_or(true, |(f, to)| f <= t && t < to)
    }

    #[inline]
    pub fn valid_from(&self, node: u64) -> i64 {
        self.range(node).map_or(OPEN_FROM, |(f, _)| f)
    }

    #[inline]
    pub fn valid_to(&self, node: u64) -> i64 {
        self.range(node).map_or(OPEN_TO, |(_, to)| to)
    }

    /// O(1) conservative check over *all* nodes.
    #[inline]
    pub fn all_active_at(&self, t: i64) -> bool {
        t >= self.max_valid_from && t < self.min_valid_to
    }

    /// Publish a corrected `valid_to` as a new immutable snapshot.
    ///
    /// M0 copies the column (O(V)); a production store uses page-level COW or
    /// a delta file. The *semantics* — new commit, old snapshot untouched — do
    /// not change.
    pub fn with_valid_to(&self, node: u64, new_to: i64) -> Result<Self> {
        let i = node as usize;
        if i >= self.valid_to.len() {
            return Err(GtvError::NodeOutOfRange(node));
        }
        if new_to < self.valid_from[i] {
            return Err(GtvError::InvalidArgument(format!(
                "corrected valid_to {new_to} precedes valid_from {}",
                self.valid_from[i]
            )));
        }
        let mut valid_to = self.valid_to.clone();
        valid_to[i] = new_to;
        Self::new(self.valid_from.clone(), valid_to)
    }
}

/// A bound view of `(CSR, policy, node snapshot)`.
///
/// Construction does **one** sequential pass over the CSR columns to derive
/// segment-level bounding boxes. Every subsequent neighbour query is O(1) to
/// decide whether endpoint filtering is needed at all.
#[derive(Debug)]
pub struct EndpointView<'a> {
    csr: &'a TemporalCSR,
    lifetime: &'a NodeLifetime,
    policy: EndpointPolicy,
    clamp_edges: u64,
    /// `src` slices whose every edge is clamped: if `src` is dead, the whole
    /// slice is provably empty without touching a single edge.
    src_all_clamped: Vec<bool>,
    /// Bounding box over nodes touched *by clamped edges* only. Nodes that
    /// expired but are never a clamped endpoint cannot disable the fast path.
    endpoint_from_max: i64,
    endpoint_to_min: i64,
}

impl<'a> EndpointView<'a> {
    pub fn new(
        csr: &'a TemporalCSR,
        lifetime: &'a NodeLifetime,
        policy: EndpointPolicy,
    ) -> Result<Self> {
        if lifetime.len() < csr.node_count() {
            return Err(GtvError::InvalidArgument(format!(
                "node lifetime covers {} nodes but the CSR has {}",
                lifetime.len(),
                csr.node_count()
            )));
        }

        let c = csr.columns();
        let mut src_all_clamped = vec![false; c.node_count];
        let mut from_max = OPEN_FROM;
        let mut to_min = OPEN_TO;
        let mut clamp_edges = 0u64;

        for s in 0..c.node_count {
            let a = c.offsets[s] as usize;
            let b = c.offsets[s + 1] as usize;
            let degree = b - a;
            let mut n_clamped = 0usize;
            for e in a..b {
                if !policy.is_clamped(c.edge_type[e]) {
                    continue;
                }
                n_clamped += 1;
                let (df, dt) = lifetime
                    .range(c.dst[e])
                    .unwrap_or((OPEN_FROM, OPEN_TO));
                from_max = from_max.max(df);
                to_min = to_min.min(dt);
            }
            if n_clamped > 0 {
                clamp_edges += n_clamped as u64;
                let (sf, st) = lifetime
                    .range(s as u64)
                    .unwrap_or((OPEN_FROM, OPEN_TO));
                from_max = from_max.max(sf);
                to_min = to_min.min(st);
            }
            src_all_clamped[s] = degree > 0 && n_clamped == degree;
        }

        Ok(Self {
            csr,
            lifetime,
            policy,
            clamp_edges,
            src_all_clamped,
            endpoint_from_max: from_max,
            endpoint_to_min: to_min,
        })
    }

    /// True when every endpoint of every clamped edge is alive at `t`.
    /// O(1); when there are no clamped edges this is trivially true.
    #[inline]
    pub fn all_endpoints_active_at(&self, t: i64) -> bool {
        t >= self.endpoint_from_max && t < self.endpoint_to_min
    }

    pub fn clamp_edges(&self) -> u64 {
        self.clamp_edges
    }

    /// HISTORICAL GRAPH: byte-for-byte the original CSR path.
    #[inline]
    pub fn historical_neighbors(&self, src: u64, t: i64) -> Result<Neighbors<'a>> {
        self.csr.neighbors(src, t)
    }

    /// ACTIVE GRAPH: edges active at `t`, intersected with endpoint lifetimes
    /// for clamped edge types.
    pub fn active_neighbors(&self, src: u64, t: i64) -> Result<ActiveNeighbors<'a>> {
        let inner = self.csr.neighbors(src, t)?;
        if self.policy.is_empty() || self.all_endpoints_active_at(t) {
            // Fast path: literally the original CSR iterator, no wrapper state.
            return Ok(ActiveNeighbors::Passthrough(inner));
        }
        // `src` is constant for a whole adjacency slice, so check it once here
        // instead of once per edge.
        let src_active = self.lifetime.active_at(src, t);
        if !src_active
            && self
                .src_all_clamped
                .get(src as usize)
                .copied()
                .unwrap_or(false)
        {
            return Ok(ActiveNeighbors::Empty);
        }
        Ok(ActiveNeighbors::Filtered {
            inner,
            lifetime: self.lifetime,
            clamp_types: self.policy.clamp_types,
            t,
            src_active,
        })
    }

    /// §5.1 `effective_from = max(edge, src, dst)`; non-clamped types are
    /// returned unchanged. Kept as an explicit call so the visibility hot loop
    /// never computes a value that traversal does not use.
    pub fn effective_from(&self, src: u64, n: &Neighbor) -> i64 {
        if !self.policy.is_clamped(n.edge_type) {
            return n.valid_from;
        }
        n.valid_from
            .max(self.lifetime.valid_from(src))
            .max(self.lifetime.valid_from(n.dst))
    }

    /// §5.1 `effective_to = min(edge, src, dst)`.
    pub fn effective_to(&self, src: u64, n: &Neighbor) -> i64 {
        if !self.policy.is_clamped(n.edge_type) {
            return n.valid_to;
        }
        n.valid_to
            .min(self.lifetime.valid_to(src))
            .min(self.lifetime.valid_to(n.dst))
    }
}

/// Iterator returned by [`EndpointView::active_neighbors`].
///
/// Three explicit states keep the common path free of filter state: when the
/// segment bounding box proves every endpoint alive, the variant *is* the
/// original `csr::Neighbors` — no extra fields, no extra branch per edge.
///
/// No `Debug` derive: that would require adding `#[derive(Debug)]` to
/// `csr::Neighbors`, and M0 promises not to touch `csr.rs` at all.
pub enum ActiveNeighbors<'a> {
    /// Bounding box hit (or no clamped types): delegate straight to the CSR.
    Passthrough(Neighbors<'a>),
    /// Dead `src` whose entire slice is clamped: provably no visible edges.
    Empty,
    /// Per-edge endpoint test required (only clamped edge types are tested).
    Filtered {
        inner: Neighbors<'a>,
        lifetime: &'a NodeLifetime,
        clamp_types: u64,
        t: i64,
        src_active: bool,
    },
}

impl Iterator for ActiveNeighbors<'_> {
    type Item = Neighbor;

    #[inline]
    fn next(&mut self) -> Option<Neighbor> {
        match self {
            ActiveNeighbors::Empty => None,
            ActiveNeighbors::Passthrough(inner) => inner.next(),
            ActiveNeighbors::Filtered {
                inner,
                lifetime,
                clamp_types,
                t,
                src_active,
            } => {
                while let Some(n) = inner.next() {
                    let clamped = n.edge_type < 64 && (*clamp_types >> n.edge_type) & 1 == 1;
                    if clamped && (!*src_active || !lifetime.active_at(n.dst, *t)) {
                        continue;
                    }
                    return Some(n);
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{TimestampNanosecondArray, UInt16Array, UInt64Array};

    /// Small graph with mixed edge types, lifetimes and valid windows.
    /// type 1 = ClampToEndpoints, type 2 = PreserveHistoricalReference,
    /// type 3 = External (kernel behaviour identical to type 2).
    fn fixture() -> TemporalCSR {
        TemporalCSR::from_arrays(
            &UInt64Array::from(vec![0u64, 0, 0, 1, 1, 2, 3, 4, 4]),
            &UInt64Array::from(vec![1u64, 2, 3, 2, 4, 0, 4, 0, 1]),
            &TimestampNanosecondArray::from(vec![
                0i64, 10, 20, 0, 5, 0, 0, 0, 100,
            ]),
            &TimestampNanosecondArray::from(vec![
                500i64, 500, 500, 500, 500, 500, 500, 500, 500,
            ]),
            &UInt16Array::from(vec![1u16, 2, 3, 1, 1, 2, 1, 1, 1]),
            5,
        )
        .unwrap()
    }

    /// Naive reference: scan the slice and apply `edge ∩ src ∩ dst`.
    fn brute(
        csr: &TemporalCSR,
        policy: &EndpointPolicy,
        life: &NodeLifetime,
        src: u64,
        t: i64,
    ) -> Vec<u64> {
        let c = csr.columns();
        let a = c.offsets[src as usize] as usize;
        let b = c.offsets[src as usize + 1] as usize;
        (a..b)
            .filter(|&e| {
                if !(c.valid_from[e] <= t && t < c.valid_to[e]) {
                    return false;
                }
                !policy.is_clamped(c.edge_type[e])
                    || (life.active_at(src, t) && life.active_at(c.dst[e], t))
            })
            .map(|e| c.dst[e])
            .collect()
    }

    #[test]
    fn three_business_policies_collapse_to_one_bit() {
        let p = EndpointPolicy::clamp(&[1]).unwrap();
        assert!(p.is_clamped(1));
        // Preserve / External live outside the mask; the kernel treats both as
        // "no endpoint constraint".
        assert!(!p.is_clamped(2));
        assert!(!p.is_clamped(3));
        assert!(EndpointPolicy::NONE.is_empty());
        assert!(EndpointPolicy::clamp(&[64]).is_err());
        assert!(EndpointPolicy::clamp(&[u16::MAX]).is_err());
    }

    #[test]
    fn empty_policy_is_exactly_the_original_iterator() {
        let csr = fixture();
        let life = NodeLifetime::forever(csr.node_count());
        let view = EndpointView::new(&csr, &life, EndpointPolicy::none()).unwrap();
        assert_eq!(view.clamp_edges(), 0);
        for src in 0..5u64 {
            for t in [0i64, 10, 50, 499, 500] {
                let a: Vec<u64> = view.historical_neighbors(src, t).unwrap().map(|n| n.dst).collect();
                let b: Vec<u64> = view.active_neighbors(src, t).unwrap().map(|n| n.dst).collect();
                // An empty policy must agree with the naive intersection too.
                assert_eq!(a, b, "src={src} t={t}");
                assert_eq!(a, brute(&csr, &EndpointPolicy::none(), &life, src, t));
            }
        }
    }

    #[test]
    fn expired_node_hides_only_clamped_edges() {
        let csr = fixture();
        // Node 2 dies at t=100.
        let life = NodeLifetime::new(
            vec![OPEN_FROM; 5],
            vec![OPEN_TO, OPEN_TO, 100, OPEN_TO, OPEN_TO],
        )
        .unwrap();
        let view = EndpointView::new(&csr, &life, EndpointPolicy::clamp(&[1]).unwrap()).unwrap();
        assert!(!view.all_endpoints_active_at(200));

        // src=0 has one clamped edge (->1), one Preserve (->2), one External (->3).
        // At t=200, dst=2 is dead but types 2/3 ignore it; src=0 is alive.
        let mut got: Vec<u64> = view.active_neighbors(0, 200).unwrap().map(|n| n.dst).collect();
        got.sort_unstable();
        assert_eq!(got, vec![1, 2, 3]);

        // src=1 has two clamped edges: ->2 (dead dst) and ->4 (alive dst).
        let mut got: Vec<u64> = view.active_neighbors(1, 200).unwrap().map(|n| n.dst).collect();
        got.sort_unstable();
        assert_eq!(got, vec![4]);

        // At t=50 all endpoints alive -> the O(1) fast path.
        assert!(view.all_endpoints_active_at(50));
    }

    #[test]
    fn dead_source_short_circuits_a_fully_clamped_slice() {
        let csr = fixture();
        let life = NodeLifetime::new(
            vec![OPEN_FROM; 5],
            vec![OPEN_TO, 30, OPEN_TO, OPEN_TO, OPEN_TO],
        )
        .unwrap();
        // src=1 has only clamped edges and dies at 30.
        let view = EndpointView::new(&csr, &life, EndpointPolicy::clamp(&[1]).unwrap()).unwrap();
        assert_eq!(view.active_neighbors(1, 200).unwrap().count(), 0);
        assert_eq!(view.historical_neighbors(1, 200).unwrap().count(), 2);
    }

    #[test]
    fn effective_range_follows_the_intersection_formula() {
        let csr = fixture();
        let life = NodeLifetime::new(
            vec![0, 0, 0, 0, 0],
            vec![400, 250, 300, OPEN_TO, 450],
        )
        .unwrap();
        let view = EndpointView::new(&csr, &life, EndpointPolicy::clamp(&[1]).unwrap()).unwrap();
        let n = view.historical_neighbors(0, 50).unwrap().next().unwrap();
        assert_eq!(n.dst, 1);
        assert_eq!(view.effective_from(0, &n), 0);
        assert_eq!(view.effective_to(0, &n), 250); // min(500, 400, 250)

        // A PreserveHistoricalReference edge (type 2) keeps its intrinsic
        // validity even though dst=2 has an earlier lifetime bound.
        let n2 = view.historical_neighbors(0, 50).unwrap().nth(1).unwrap();
        assert_eq!(n2.dst, 2);
        assert_eq!(view.effective_to(0, &n2), 500);
    }

    #[test]
    fn correction_publishes_a_new_snapshot_without_mutating_the_old_one() {
        let old = NodeLifetime::forever(5);
        let corrected = old.with_valid_to(2, 100).unwrap();
        assert!(old.active_at(2, 200), "S1 cognition is immutable");
        assert!(!corrected.active_at(2, 200), "S2 cognition is corrected");
        assert_eq!(old.valid_to(2), OPEN_TO);
        assert_eq!(corrected.valid_to(2), 100);
    }

    /// Deterministic LCG, so the property test needs no extra dependency.
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    #[test]
    fn active_matches_naive_intersection_under_random_lifetimes() {
        let csr = fixture();
        let mut rng = 0x5eed_u64;
        for _ in 0..250 {
            let from: Vec<i64> = (0..5).map(|_| (lcg(&mut rng) % 60) as i64).collect();
            let to: Vec<i64> = from
                .iter()
                .map(|&f| f + (lcg(&mut rng) % 400) as i64 + 1)
                .collect();
            let life = NodeLifetime::new(from, to).unwrap();
            let mask = lcg(&mut rng) % 8;
            let types: Vec<u16> = (0..3).filter(|i| (mask >> i) & 1 == 1).map(|i| i as u16).collect();
            let policy = EndpointPolicy::clamp(&types).unwrap();
            let view = EndpointView::new(&csr, &life, policy).unwrap();
            let t = (lcg(&mut rng) % 560) as i64;
            for src in 0..5u64 {
                let expected = brute(&csr, &policy, &life, src, t);
                let got: Vec<u64> = view.active_neighbors(src, t).unwrap().map(|n| n.dst).collect();
                assert_eq!(got, expected, "src={src} t={t} mask={mask} life={life:?}");
            }
        }
    }
}
