//! M2: L2 time-range aggregates over the **folded** visible row set.
//!
//! An interval correction splits one logical edge into
//! `[A,start) / [start,end) / [end,B)`. If an aggregate simply counted rows,
//! that edge would be counted three times — this is the "去重复口径" trap in
//! `asymmetric-bitemporal-review.md` §8.6.
//!
//! [`AggregateView`] therefore merges touching/overlapping intervals per
//! `(src, dst, edge_type)` **once**, when the SQL surface is synchronised. Every
//! later aggregate reads the merged list, so a point count and a range count
//! are both distinct-logical-edge counts by construction.
//!
//! The view is a snapshot at one system time, deliberately like the L1 catalog
//! projection: refreshing it is an explicit `register_segment_store` /
//! `attach_aggregate_view` call, never a hidden cache invalidation.

use std::collections::{BTreeMap, HashSet};

use crate::store::LiveRow;

/// Identity of a logical edge. Business-time fragments of the same edge share
/// one key; `edge_type` participates because a correction may reclassify it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LogicalEdge {
    pub src: u64,
    pub dst: u64,
    pub edge_type: u16,
}

impl LogicalEdge {
    pub fn of(row: &LiveRow) -> Self {
        Self {
            src: row.edge.src,
            dst: row.edge.dst,
            edge_type: row.edge.edge_type,
        }
    }
}

/// One merged, half-open business interval of a logical edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeInterval {
    pub src: u64,
    pub dst: u64,
    pub edge_type: u16,
    pub valid_from: i64,
    pub valid_to: i64,
}

/// One bucket of `range_agg()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeBucket {
    pub bucket_start: i64,
    pub bucket_end: i64,
    /// Distinct logical edges active anywhere in this bucket.
    pub active_edges: u64,
}

/// Immutable projection of `as_of_system(system_ts)` for SQL aggregates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregateView {
    pub system_ts: i64,
    /// Sorted, non-overlapping intervals; one logical edge may appear more than
    /// once only when it has genuinely disjoint lifetimes.
    pub intervals: Vec<EdgeInterval>,
    /// Raw visible rows before the merge — proof that de-duplication happened.
    pub rows_before_dedup: usize,
    /// Number of baseline segments visible at `system_ts`. Segment-level
    /// FullHit short-circuiting is reported separately by
    /// `SegmentStore::active_edge_count_at()`; this projection is already
    /// folded, so it has no segments left to open.
    pub segments_total: usize,
}

impl AggregateView {
    /// Build from the single authoritative visible-row enumeration.
    pub fn from_rows(system_ts: i64, rows: &[LiveRow], segments_total: usize) -> Self {
        let mut by_edge: BTreeMap<LogicalEdge, Vec<(i64, i64)>> = BTreeMap::new();
        for r in rows {
            by_edge
                .entry(LogicalEdge::of(r))
                .or_default()
                .push((r.edge.valid_from, r.edge.valid_to));
        }

        let mut intervals = Vec::new();
        for (key, mut spans) in by_edge {
            spans.sort_unstable();
            let mut cur = spans[0];
            for &(a, b) in spans.iter().skip(1) {
                if a <= cur.1 {
                    // Touching half-open intervals ([0,10) + [10,20)) merge
                    // back into the original [0,20) after a correction split.
                    cur.1 = cur.1.max(b);
                } else {
                    intervals.push(EdgeInterval::of(key, cur));
                    cur = (a, b);
                }
            }
            intervals.push(EdgeInterval::of(key, cur));
        }

        Self {
            system_ts,
            intervals,
            rows_before_dedup: rows.len(),
            segments_total,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Distinct logical edges active at business instant `t`.
    ///
    /// Merged intervals never overlap, so a simple filter is already a
    /// distinct count — no per-query HashSet.
    pub fn active_edge_count(&self, t: i64) -> u64 {
        self.intervals
            .iter()
            .filter(|iv| iv.valid_from <= t && t < iv.valid_to)
            .count() as u64
    }

    /// Distinct logical edges active anywhere inside each `[t1,t2)` bucket.
    pub fn range_agg(&self, t1: i64, t2: i64, bucket: i64) -> Vec<RangeBucket> {
        if bucket <= 0 || t1 >= t2 {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut start = t1;
        while start < t2 {
            let end = start.saturating_add(bucket).min(t2);
            // A logical edge with several disjoint lifetimes may have several
            // merged intervals; a bucket counts the *edge* once, not each
            // lifetime fragment.
            let active_edges = self
                .intervals
                .iter()
                .filter(|iv| iv.valid_from < end && iv.valid_to > start)
                .map(|iv| LogicalEdge {
                    src: iv.src,
                    dst: iv.dst,
                    edge_type: iv.edge_type,
                })
                .collect::<HashSet<_>>()
                .len() as u64;
            out.push(RangeBucket {
                bucket_start: start,
                bucket_end: end,
                active_edges,
            });
            if end <= start {
                break;
            }
            start = end;
        }
        out
    }
}

impl EdgeInterval {
    fn of(key: LogicalEdge, (valid_from, valid_to): (i64, i64)) -> Self {
        Self {
            src: key.src,
            dst: key.dst,
            edge_type: key.edge_type,
            valid_from,
            valid_to,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::RowPos;
    use crate::wal::EdgeRecord;

    fn row(src: u64, from: i64, to: i64) -> LiveRow {
        LiveRow {
            pos: RowPos::new(1, 0),
            edge: EdgeRecord {
                src,
                dst: 9,
                valid_from: from,
                valid_to: to,
                edge_type: 0,
                system_ts: 1,
            },
            table_id: 0,
            partition_id: 0,
        }
    }

    #[test]
    fn correction_split_merges_back_to_one_logical_edge() {
        // One edge split into three fragments by [10,20) correction.
        let rows = vec![row(1, 0, 10), row(1, 10, 20), row(1, 20, 30)];
        let v = AggregateView::from_rows(7, &rows, 1);
        assert_eq!(v.rows_before_dedup, 3);
        assert_eq!(v.intervals.len(), 1);
        assert_eq!(v.intervals[0].valid_from, 0);
        assert_eq!(v.intervals[0].valid_to, 30);
        assert_eq!(v.active_edge_count(5), 1);
        assert_eq!(v.active_edge_count(15), 1);
        assert_eq!(v.active_edge_count(30), 0);
        // A range aggregate must not report 3 for the same logical edge.
        let buckets = v.range_agg(0, 30, 10);
        assert_eq!(buckets.len(), 3);
        assert!(buckets.iter().all(|b| b.active_edges == 1));
    }

    #[test]
    fn disjoint_lifetimes_stay_distinct_intervals() {
        let rows = vec![row(1, 0, 5), row(1, 10, 15), row(2, 0, 30)];
        let v = AggregateView::from_rows(7, &rows, 1);
        assert_eq!(v.intervals.len(), 3);
        assert_eq!(v.active_edge_count(2), 2);
        assert_eq!(v.active_edge_count(7), 1);
        assert_eq!(v.active_edge_count(12), 2);
        assert_eq!(v.range_agg(0, 30, 30)[0].active_edges, 2);
    }

    #[test]
    fn invalid_ranges_produce_no_buckets() {
        let v = AggregateView::from_rows(0, &[row(1, 0, 10)], 1);
        assert!(v.range_agg(10, 0, 5).is_empty());
        assert!(v.range_agg(0, 10, 0).is_empty());
        assert_eq!(v.range_agg(0, 10, 4).len(), 3);
    }
}
