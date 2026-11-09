//! L2 physical pruning metadata: the segment-level time box.
//!
//! # Why four scalars and not two
//!
//! `TemporalCSR` already carries two scalars, and they are **not** a bounding
//! box — they describe the window in which *every* edge is active:
//!
//! ```text
//! max_valid_from = max(valid_from over all edges)   // intersection start
//! min_valid_to   = min(valid_to   over all edges)   // intersection end
//! ```
//!
//! Pruning needs the opposite: the smallest interval that *may* contain an
//! answer, i.e. the union:
//!
//! ```text
//! valid_from_min = min(valid_from over all edges)   // union start
//! valid_to_max   = max(valid_to   over all edges)   // union end
//! ```
//!
//! Using the intersection to prune silently drops rows; using the union to
//! short-circuit over-counts. Both live here, side by side, and each predicate
//! documents which one it reads.
//!
//! # Half-open semantics
//!
//! Every interval is `[valid_from, valid_to)`. Open-ended edges store
//! `valid_to = i64::MAX`, so `t >= valid_to_max` never misclassifies them as
//! expired as long as `t < i64::MAX`.
//!
//! # Where this is computed
//!
//! Only at flush time, in one linear pass over the CSR columns. It is never
//! recomputed on the read path: the numbers are published in the manifest, so
//! a reader can prune **without opening a single `.seg` file**.

use gtv_core::TemporalCSR;

/// L2 pruning metadata for one immutable segment.
///
/// `Copy` on purpose: it is a handful of scalars, read straight out of the
/// manifest, and threaded through segment selection by value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentMeta {
    pub node_count: u64,
    pub edge_count: u64,
    /// Union start: earliest `valid_from` in the segment.
    pub valid_from_min: i64,
    /// Union end: latest `valid_to` in the segment (`i64::MAX` = open edges).
    pub valid_to_max: i64,
    /// Intersection start: `t >= max_valid_from` is a prerequisite for
    /// "all edges active".
    pub max_valid_from: i64,
    /// Intersection end: `t < min_valid_to` is the other prerequisite.
    pub min_valid_to: i64,
    /// Smallest node id touched as a source or a destination.
    pub node_id_min: u64,
    /// Largest node id touched as a source or a destination.
    pub node_id_max: u64,
    /// Tight box over **sources only**. `neighbors(src, t)` never needs to open
    /// a segment whose `src` run is empty, which `node_id_min/max` (a union
    /// over src and dst) cannot express.
    pub src_id_min: u64,
    pub src_id_max: u64,
    pub edge_type_min: u16,
    pub edge_type_max: u16,
}

/// What a time predicate can conclude about a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneOutcome {
    /// The segment cannot contribute: do not open the file at all.
    Skip,
    /// Every edge is active over the whole query window: the answer's shape is
    /// known without touching a column (`active_edge_count == edge_count`).
    FullHit,
    /// The segment may contribute, but edges must be tested individually.
    Scan,
}

impl SegmentMeta {
    /// Metadata of an empty segment: prunes everything, indexes nothing.
    pub const EMPTY: SegmentMeta = SegmentMeta {
        node_count: 0,
        edge_count: 0,
        valid_from_min: i64::MAX,
        valid_to_max: i64::MIN,
        max_valid_from: i64::MIN,
        min_valid_to: i64::MAX,
        node_id_min: u64::MAX,
        node_id_max: 0,
        src_id_min: u64::MAX,
        src_id_max: 0,
        edge_type_min: u16::MAX,
        edge_type_max: 0,
    };

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.edge_count == 0
    }

    /// One linear pass over the CSR columns (`O(V + E)`, flush-time only).
    ///
    /// Edge columns are already sorted by `(src, valid_from, valid_to, dst)`,
    /// so this is a sequential scan — no random access, no re-sort.
    pub fn from_csr(csr: &TemporalCSR) -> SegmentMeta {
        let c = csr.columns();
        let mut meta = SegmentMeta {
            node_count: c.node_count as u64,
            edge_count: c.dst.len() as u64,
            valid_from_min: i64::MAX,
            valid_to_max: i64::MIN,
            max_valid_from: i64::MIN,
            min_valid_to: i64::MAX,
            node_id_min: u64::MAX,
            node_id_max: 0,
            src_id_min: u64::MAX,
            src_id_max: 0,
            edge_type_min: u16::MAX,
            edge_type_max: 0,
        };
        if c.dst.is_empty() {
            return SegmentMeta::EMPTY;
        }

        // Sources: only the non-empty runs matter, so this is one pass over the
        // offsets array rather than one pass per edge.
        for s in 0..c.node_count {
            let (a, b) = (c.offsets[s] as usize, c.offsets[s + 1] as usize);
            if a == b {
                continue;
            }
            let id = s as u64;
            meta.node_id_min = meta.node_id_min.min(id);
            meta.node_id_max = meta.node_id_max.max(id);
            meta.src_id_min = meta.src_id_min.min(id);
            meta.src_id_max = meta.src_id_max.max(id);
        }

        for e in 0..c.dst.len() {
            let (vf, vt) = (c.valid_from[e], c.valid_to[e]);
            if vf < meta.valid_from_min {
                meta.valid_from_min = vf;
            }
            if vf > meta.max_valid_from {
                meta.max_valid_from = vf;
            }
            if vt > meta.valid_to_max {
                meta.valid_to_max = vt;
            }
            if vt < meta.min_valid_to {
                meta.min_valid_to = vt;
            }
            let d = c.dst[e];
            if d < meta.node_id_min {
                meta.node_id_min = d;
            }
            if d > meta.node_id_max {
                meta.node_id_max = d;
            }
            let t = c.edge_type[e];
            if t < meta.edge_type_min {
                meta.edge_type_min = t;
            }
            if t > meta.edge_type_max {
                meta.edge_type_max = t;
            }
        }
        meta
    }

    /// Does the segment possibly have an edge active somewhere in `[t1, t2)`?
    ///
    /// Reads the **union** box. `[vf, vt)` intersects `[t1, t2)` iff
    /// `vf < t2 && vt > t1`.
    #[inline]
    pub fn overlaps(&self, t1: i64, t2: i64) -> bool {
        !self.is_empty() && self.valid_from_min < t2 && self.valid_to_max > t1
    }

    /// Does the segment possibly have an edge active exactly at `t`?
    #[inline]
    pub fn may_contain(&self, t: i64) -> bool {
        !self.is_empty() && self.valid_from_min <= t && self.valid_to_max > t
    }

    /// Could this segment hold an edge **originating** at `src`?
    ///
    /// `u64::MAX`/`0` (a legacy manifest with unknown src box) is deliberately
    /// treated as "unknown" and never prunes — an old manifest must not lose
    /// rows just because a newer reader knows how to skip more.
    #[inline]
    pub fn may_contain_src(&self, src: u64) -> bool {
        if self.is_empty() {
            return false;
        }
        if self.src_id_min == u64::MAX && self.src_id_max == 0 {
            return true;
        }
        self.src_id_min <= src && src <= self.src_id_max
    }

    /// Is *every* edge in the segment active at `t`? Reads the **intersection**
    /// window, so the caller may skip the per-edge test entirely.
    #[inline]
    pub fn all_active_at(&self, t: i64) -> bool {
        !self.is_empty() && t >= self.max_valid_from && t < self.min_valid_to
    }

    /// Is `[t1, t2)` fully inside the all-active window? Then every time bucket
    /// of the range aggregation has the same edge count (`edge_count`).
    #[inline]
    pub fn fully_covers(&self, t1: i64, t2: i64) -> bool {
        !self.is_empty() && self.max_valid_from <= t1 && t2 <= self.min_valid_to
    }

    /// Three-way decision for one segment and one range query.
    #[inline]
    pub fn classify(&self, t1: i64, t2: i64) -> PruneOutcome {
        if !self.overlaps(t1, t2) {
            PruneOutcome::Skip
        } else if self.fully_covers(t1, t2) {
            PruneOutcome::FullHit
        } else {
            PruneOutcome::Scan
        }
    }

    /// Three-way decision for a point query. Implemented without `t + 1` so
    /// `i64::MAX` cannot overflow.
    #[inline]
    pub fn classify_point(&self, t: i64) -> PruneOutcome {
        if !self.may_contain(t) {
            PruneOutcome::Skip
        } else if self.all_active_at(t) {
            PruneOutcome::FullHit
        } else {
            PruneOutcome::Scan
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gtv_core::TemporalCSR;

    fn csr(edges: &[(u64, u64, i64, i64)]) -> TemporalCSR {
        let src: Vec<u64> = edges.iter().map(|e| e.0).collect();
        let dst: Vec<u64> = edges.iter().map(|e| e.1).collect();
        let vf: Vec<i64> = edges.iter().map(|e| e.2).collect();
        let vt: Vec<i64> = edges.iter().map(|e| e.3).collect();
        let et = vec![0u16; edges.len()];
        TemporalCSR::from_edge_columns(&src, &dst, &vf, &vt, &et, 8).unwrap()
    }

    #[test]
    fn union_and_intersection_are_distinct() {
        let c = csr(&[(0, 1, 10, 20), (2, 3, 30, 40)]);
        let m = SegmentMeta::from_csr(&c);
        assert_eq!(m.valid_from_min, 10, "union start");
        assert_eq!(m.valid_to_max, 40, "union end");
        assert_eq!(m.max_valid_from, 30, "intersection start");
        assert_eq!(m.min_valid_to, 20, "intersection end");
        // Nothing is active everywhere: the intersection is empty.
        assert!(!m.all_active_at(15));
        assert!(!m.all_active_at(35));
        // But the union really does overlap these windows.
        assert!(m.overlaps(0, 11));
        assert!(m.overlaps(39, 100));
        assert!(m.overlaps(0, 100));
        // A gap *inside* the union box is a **false positive**, not a bug: a
        // box cannot express holes. Pruning must never have false negatives, so
        // [20,30) sits between the two intervals and is still selected; the
        // per-edge test is what rejects it. That is why `classify` says `Scan`
        // rather than `FullHit` here.
        assert!(m.overlaps(20, 30));
        assert_eq!(m.classify(20, 30), PruneOutcome::Scan);
    }

    #[test]
    fn half_open_boundaries() {
        let c = csr(&[(0, 1, 10, 20)]);
        let m = SegmentMeta::from_csr(&c);
        assert!(m.may_contain(10));
        assert!(m.may_contain(19));
        assert!(!m.may_contain(20), "valid_to is exclusive");
        assert!(!m.may_contain(9));
        assert_eq!(m.classify(10, 20), PruneOutcome::FullHit);
        assert_eq!(m.classify(20, 30), PruneOutcome::Skip);
        assert_eq!(m.classify(15, 30), PruneOutcome::Scan);
    }

    #[test]
    fn empty_segment_prunes_everything() {
        let m = SegmentMeta::EMPTY;
        assert!(!m.overlaps(i64::MIN, i64::MAX));
        assert!(!m.may_contain(0));
        assert!(!m.all_active_at(0));
        assert_eq!(m.classify(0, 10), PruneOutcome::Skip);
    }

    #[test]
    fn open_ended_edge_never_expires() {
        let c = csr(&[(0, 1, 100, i64::MAX)]);
        let m = SegmentMeta::from_csr(&c);
        assert_eq!(m.valid_to_max, i64::MAX);
        assert!(m.may_contain(1_000_000));
        assert!(m.overlaps(1_000_000, 2_000_000));
        assert_eq!(m.classify_point(i64::MAX - 1), PruneOutcome::FullHit);
    }

    #[test]
    fn node_bounds_track_touched_nodes_only() {
        let c = csr(&[(3, 5, 0, 10), (3, 5, 0, 10), (3, 5, 0, 10), (3, 5, 0, 10)]);
        let m = SegmentMeta::from_csr(&c);
        // Node ids 0..7 exist, but the segment only touches 3 and 5.
        assert_eq!(m.node_id_min, 3);
        assert_eq!(m.node_id_max, 5);
    }
}
