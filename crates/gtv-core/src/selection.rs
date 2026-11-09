//! Candidate sets shared by every operator (M1 / A1).
//!
//! Before this module each operator took a different candidate representation:
//! k-hop took `&UInt64Array`, ANN took `Option<&BooleanArray>`, the pattern
//! matcher took an implicit `0..node_count`. Composing them (temporal filter →
//! vector recall → k-hop → motif → aggregate) meant materialising and
//! re-encoding the same candidate set five times at the call site.
//!
//! [`Sel`] is that shared value. It carries the two invariants that make
//! composition safe:
//!
//! * `snapshot_id` — a set built against one snapshot cannot silently be used
//!   against another ([`crate::GtvError::SnapshotMismatch`]);
//! * `universe` — the row count the ids index into, so out-of-range ids are
//!   rejected at construction instead of panicking inside a traversal.
//!
//! Three representations cover the access patterns that actually occur:
//!
//! | repr | built by | best for |
//! |---|---|---|
//! | [`SelRepr::Bitmap`] | `from_mask`, `from_bitmap` | high selectivity, O(1) membership, zero-copy to arrow |
//! | [`SelRepr::Dense`] | `from_ids` | low selectivity, needs ascending scan |
//! | [`SelRepr::Ranges`] | `from_ranges`, `all` | contiguous slices (time buckets, id windows) |
//!
//! Conversion is explicit: [`Sel::to_mask`] is O(1) for `Bitmap` (the
//! [`BooleanBuffer`] is refcounted) and O(universe) otherwise.

use arrow::array::BooleanArray;
use arrow::buffer::BooleanBuffer;

use crate::error::{GtvError, Result};

/// Sentinel for "this set is not bound to a snapshot".
///
/// Only for building a set from bare ids where no snapshot context exists yet
/// (a comparator, a unit test, a SQL literal list). Any real query path must
/// bind, otherwise the cross-snapshot check is silently skipped.
pub const UNBOUND_SNAPSHOT: u64 = 0;

/// Physical representation of a candidate set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelRepr {
    /// Bit-packed membership over `0..universe`.
    Bitmap { bits: BooleanBuffer },
    /// Ascending, deduplicated ids.
    Dense { ids: Vec<u64> },
    /// Ascending, disjoint, half-open `[start, end)` intervals.
    Ranges { ranges: Vec<(u64, u64)> },
}

/// A candidate set over one snapshot's id space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sel {
    snapshot_id: u64,
    universe: u64,
    repr: SelRepr,
}

impl Sel {
    // -- construction ------------------------------------------------------

    /// Empty set over `universe` rows.
    pub fn empty(snapshot_id: u64, universe: u64) -> Self {
        Self {
            snapshot_id,
            universe,
            repr: SelRepr::Dense { ids: Vec::new() },
        }
    }

    /// Every row in `universe`. O(1): one range, no materialisation.
    pub fn all(snapshot_id: u64, universe: u64) -> Self {
        Self {
            snapshot_id,
            universe,
            repr: SelRepr::Ranges {
                ranges: vec![(0, universe)],
            },
        }
    }

    /// From ids; sorted and deduplicated. Ids must be `< universe`.
    pub fn from_ids(
        snapshot_id: u64,
        universe: u64,
        ids: impl IntoIterator<Item = u64>,
    ) -> Result<Self> {
        let mut ids: Vec<u64> = ids.into_iter().collect();
        if let Some(bad) = ids.iter().find(|id| **id >= universe) {
            return Err(GtvError::NodeOutOfRange(*bad));
        }
        ids.sort_unstable();
        ids.dedup();
        // Dense costs 8 B/entry; a bitmap costs universe/8 B. Pick the cheaper.
        let dense_cost = ids.len() as u64 * 8;
        let bitmap_cost = universe.div_ceil(8);
        let repr = if dense_cost <= bitmap_cost {
            SelRepr::Dense { ids }
        } else {
            SelRepr::Bitmap {
                bits: dense_to_bits(&ids, universe),
            }
        };
        Ok(Self {
            snapshot_id,
            universe,
            repr,
        })
    }

    /// From a boolean mask; `universe` is the mask length.
    pub fn from_mask(snapshot_id: u64, mask: &BooleanArray) -> Self {
        Self {
            snapshot_id,
            universe: mask.len() as u64,
            repr: SelRepr::Bitmap {
                bits: mask.values().clone(),
            },
        }
    }

    /// From an owned buffer. Zero copy — the buffer is refcounted.
    pub fn from_bitmap(snapshot_id: u64, bits: BooleanBuffer) -> Self {
        Self {
            snapshot_id,
            universe: bits.len() as u64,
            repr: SelRepr::Bitmap { bits },
        }
    }

    /// From half-open intervals. Overlaps are merged, touching ranges coalesce,
    /// and the result is clamped to `universe`.
    pub fn from_ranges(
        snapshot_id: u64,
        universe: u64,
        ranges: impl IntoIterator<Item = (u64, u64)>,
    ) -> Self {
        let mut ranges: Vec<(u64, u64)> = ranges
            .into_iter()
            .filter_map(|(a, b)| {
                let a = a.min(universe);
                let b = b.min(universe);
                (a < b).then_some((a, b))
            })
            .collect();
        ranges.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
        for (a, b) in ranges {
            match merged.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(b),
                _ => merged.push((a, b)),
            }
        }
        Self {
            snapshot_id,
            universe,
            repr: SelRepr::Ranges { ranges: merged },
        }
    }

    // -- identity ----------------------------------------------------------

    pub fn snapshot_id(&self) -> u64 {
        self.snapshot_id
    }

    pub fn universe(&self) -> u64 {
        self.universe
    }

    pub fn repr(&self) -> &SelRepr {
        &self.repr
    }

    /// True when this set may be used against `snapshot_id`.
    ///
    /// An unbound set ([`UNBOUND_SNAPSHOT`]) matches everything, on purpose:
    /// it exists only for paths that have no snapshot context to check against.
    pub fn is_bound(&self) -> bool {
        self.snapshot_id != UNBOUND_SNAPSHOT
    }

    /// Verify that this set belongs to `snapshot_id`.
    pub fn check_against(&self, snapshot_id: u64) -> Result<()> {
        if self.snapshot_id != UNBOUND_SNAPSHOT && self.snapshot_id != snapshot_id {
            return Err(GtvError::SnapshotMismatch {
                expected: self.snapshot_id,
                got: snapshot_id,
            });
        }
        Ok(())
    }

    // -- queries -----------------------------------------------------------

    pub fn len(&self) -> usize {
        match &self.repr {
            SelRepr::Bitmap { bits } => bits.count_set_bits(),
            SelRepr::Dense { ids } => ids.len(),
            SelRepr::Ranges { ranges } => ranges.iter().map(|(a, b)| (b - a) as usize).sum(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn contains(&self, id: u64) -> bool {
        if id >= self.universe {
            return false;
        }
        match &self.repr {
            SelRepr::Bitmap { bits } => bits.value(id as usize),
            SelRepr::Dense { ids } => ids.binary_search(&id).is_ok(),
            SelRepr::Ranges { ranges } => {
                // Last range whose start is <= id; check its end.
                match ranges.binary_search_by(|(a, _)| a.cmp(&id)) {
                    Ok(_) => true,
                    Err(0) => false,
                    Err(i) => id < ranges[i - 1].1,
                }
            }
        }
    }

    pub fn iter(&self) -> SelIter<'_> {
        match &self.repr {
            SelRepr::Bitmap { bits } => SelIter {
                inner: SelIterInner::Bits {
                    bits,
                    pos: 0,
                    len: self.universe as usize,
                },
            },
            SelRepr::Dense { ids } => SelIter {
                inner: SelIterInner::Dense(ids.iter()),
            },
            SelRepr::Ranges { ranges } => SelIter {
                inner: SelIterInner::Ranges {
                    ranges: ranges.iter(),
                    cur: 0,
                    end: 0,
                },
            },
        }
    }

    /// Materialise ascending ids.
    pub fn to_ids(&self) -> Vec<u64> {
        self.iter().collect()
    }

    /// Boolean buffer view. O(1) for [`SelRepr::Bitmap`] (refcounted), O(universe)
    /// otherwise.
    pub fn to_buffer(&self) -> BooleanBuffer {
        match &self.repr {
            SelRepr::Bitmap { bits } => bits.clone(),
            _ => {
                let mut b = vec![false; self.universe as usize];
                for id in self.iter() {
                    b[id as usize] = true;
                }
                BooleanBuffer::from(b)
            }
        }
    }

    /// Arrow mask, for operators whose port is still `&BooleanArray`.
    pub fn to_mask(&self) -> BooleanArray {
        BooleanArray::new(self.to_buffer(), None)
    }

    // -- set algebra -------------------------------------------------------

    fn same_space(&self, other: &Sel) -> Result<()> {
        if self.universe != other.universe {
            return Err(GtvError::UniverseMismatch {
                expected: self.universe,
                got: other.universe,
            });
        }
        if self.snapshot_id != other.snapshot_id && self.is_bound() && other.is_bound() {
            return Err(GtvError::SnapshotMismatch {
                expected: self.snapshot_id,
                got: other.snapshot_id,
            });
        }
        Ok(())
    }

    /// Intersection, preferring the cheapest output representation.
    pub fn intersect(&self, other: &Sel) -> Result<Sel> {
        self.same_space(other)?;
        let repr = match (&self.repr, &other.repr) {
            (SelRepr::Dense { ids: a }, SelRepr::Dense { ids: b }) => SelRepr::Dense {
                ids: merge_intersect(a, b),
            },
            (SelRepr::Ranges { ranges: a }, SelRepr::Ranges { ranges: b }) => SelRepr::Ranges {
                ranges: range_intersect(a, b),
            },
            _ => {
                let a = self.to_buffer();
                let b = other.to_buffer();
                let mut out = Vec::with_capacity(self.universe as usize);
                for i in 0..self.universe as usize {
                    out.push(a.value(i) && b.value(i));
                }
                SelRepr::Bitmap {
                    bits: BooleanBuffer::from(out),
                }
            }
        };
        Ok(Sel {
            snapshot_id: self.snapshot_id.max(other.snapshot_id),
            universe: self.universe,
            repr,
        })
    }

    /// Union, preferring the cheapest output representation.
    pub fn union(&self, other: &Sel) -> Result<Sel> {
        self.same_space(other)?;
        let repr = match (&self.repr, &other.repr) {
            (SelRepr::Dense { ids: a }, SelRepr::Dense { ids: b }) => SelRepr::Dense {
                ids: merge_union(a, b),
            },
            (SelRepr::Ranges { ranges: a }, SelRepr::Ranges { ranges: b }) => {
                let mut all: Vec<(u64, u64)> = a.iter().chain(b.iter()).copied().collect();
                all.sort_unstable();
                let mut merged: Vec<(u64, u64)> = Vec::with_capacity(all.len());
                for (lo, hi) in all {
                    match merged.last_mut() {
                        Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
                        _ => merged.push((lo, hi)),
                    }
                }
                SelRepr::Ranges { ranges: merged }
            }
            _ => {
                let a = self.to_buffer();
                let b = other.to_buffer();
                let mut out = Vec::with_capacity(self.universe as usize);
                for i in 0..self.universe as usize {
                    out.push(a.value(i) || b.value(i));
                }
                SelRepr::Bitmap {
                    bits: BooleanBuffer::from(out),
                }
            }
        };
        Ok(Sel {
            snapshot_id: self.snapshot_id.max(other.snapshot_id),
            universe: self.universe,
            repr,
        })
    }

    /// Drop ids outside `keep`, keeping the cheaper representation.
    pub fn retain(&self, keep: &Sel) -> Result<Sel> {
        self.intersect(keep)
    }
}

impl<'a> IntoIterator for &'a Sel {
    type Item = u64;
    type IntoIter = SelIter<'a>;

    fn into_iter(self) -> SelIter<'a> {
        self.iter()
    }
}

/// Iterator over the ids of a [`Sel`], ascending.
pub struct SelIter<'a> {
    inner: SelIterInner<'a>,
}

enum SelIterInner<'a> {
    Bits {
        bits: &'a BooleanBuffer,
        pos: usize,
        len: usize,
    },
    Dense(std::slice::Iter<'a, u64>),
    Ranges {
        ranges: std::slice::Iter<'a, (u64, u64)>,
        cur: u64,
        end: u64,
    },
}

impl Iterator for SelIter<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        match &mut self.inner {
            SelIterInner::Bits { bits, pos, len } => {
                while *pos < *len {
                    let i = *pos;
                    *pos += 1;
                    if bits.value(i) {
                        return Some(i as u64);
                    }
                }
                None
            }
            SelIterInner::Dense(it) => it.next().copied(),
            SelIterInner::Ranges { ranges, cur, end } => loop {
                if *cur < *end {
                    let v = *cur;
                    *cur += 1;
                    return Some(v);
                }
                match ranges.next() {
                    Some((a, b)) => {
                        *cur = *a;
                        *end = *b;
                    }
                    None => return None,
                }
            },
        }
    }
}

fn dense_to_bits(ids: &[u64], universe: u64) -> BooleanBuffer {
    let mut b = vec![false; universe as usize];
    for &id in ids {
        b[id as usize] = true;
    }
    BooleanBuffer::from(b)
}

fn merge_intersect(a: &[u64], b: &[u64]) -> Vec<u64> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

fn merge_union(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

fn range_intersect(a: &[(u64, u64)], b: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        let lo = a[i].0.max(b[j].0);
        let hi = a[i].1.min(b[j].1);
        if lo < hi {
            out.push((lo, hi));
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(ids: &[u64], universe: u64) -> Sel {
        Sel::from_ids(1, universe, ids.iter().copied()).unwrap()
    }

    #[test]
    fn roundtrip_every_repr() {
        for s in [
            dense(&[0, 3, 9], 16),
            Sel::from_ranges(1, 16, [(1, 4), (10, 12)]),
            Sel::from_bitmap(
                1,
                BooleanBuffer::from(vec![
                    false, true, true, false, false, false, false, false, false, true, false,
                    false, false, false, false, false,
                ]),
            ),
        ] {
            assert_eq!(s.to_ids(), {
                let mut v: Vec<u64> = s.iter().collect();
                v.dedup();
                v
            });
            // Membership agrees with the materialised ids at every position.
            for i in 0..16u64 {
                assert_eq!(
                    s.contains(i),
                    s.to_ids().contains(&i),
                    "id {i} repr {:?}",
                    s.repr()
                );
            }
            assert_eq!(s.len(), s.to_ids().len());
        }
    }

    #[test]
    fn all_is_one_range_and_zero_cost() {
        let s = Sel::all(7, 1_000_000);
        assert_eq!(s.len(), 1_000_000);
        assert_eq!(
            s.repr(),
            &SelRepr::Ranges {
                ranges: vec![(0, 1_000_000)]
            }
        );
        assert!(s.contains(999_999));
        assert!(!s.contains(1_000_000));
    }

    #[test]
    fn out_of_range_id_is_rejected_at_construction() {
        let e = Sel::from_ids(1, 4, [0, 4]).unwrap_err();
        assert!(matches!(e, GtvError::NodeOutOfRange(4)));
        assert!(!Sel::all(1, 4).contains(4));
    }

    #[test]
    fn snapshot_mismatch_is_fatal() {
        let a = Sel::from_ids(7, 8, [1u64]).unwrap();
        assert!(a.check_against(7).is_ok());
        let e = a.check_against(8).unwrap_err();
        assert!(matches!(
            e,
            GtvError::SnapshotMismatch {
                expected: 7,
                got: 8
            }
        ));
        // Unbound sets are usable anywhere, by design.
        let u = Sel::from_ids(UNBOUND_SNAPSHOT, 8, [1u64]).unwrap();
        assert!(u.check_against(99).is_ok());
        // But two *bound* sets from different snapshots cannot be combined.
        let b = Sel::from_ids(9, 8, [1u64]).unwrap();
        assert!(a.intersect(&b).is_err());
    }

    #[test]
    fn intersect_and_union_across_reprs() {
        let a = Sel::from_ranges(1, 16, [(0, 8)]);
        let b = dense(&[4, 5, 6, 15], 16);
        let i = a.intersect(&b).unwrap();
        assert_eq!(i.to_ids(), vec![4, 5, 6]);
        let u = a.union(&b).unwrap();
        assert_eq!(u.to_ids(), vec![0, 1, 2, 3, 4, 5, 6, 7, 15]);
    }

    #[test]
    fn intersect_is_commutative_and_idempotent() {
        let a = dense(&[1, 2, 3, 9], 16);
        let b = Sel::from_ranges(1, 16, [(2, 10)]);
        assert_eq!(
            a.intersect(&b).unwrap().to_ids(),
            b.intersect(&a).unwrap().to_ids()
        );
        assert_eq!(a.intersect(&a).unwrap().to_ids(), a.to_ids());
        assert_eq!(a.union(&a).unwrap().to_ids(), a.to_ids());
    }

    #[test]
    fn empty_intersection() {
        let a = dense(&[1], 8);
        let b = dense(&[2], 8);
        assert!(a.intersect(&b).unwrap().is_empty());
        assert!(Sel::empty(1, 8).is_empty());
        assert_eq!(
            Sel::all(1, 8).intersect(&Sel::empty(1, 8)).unwrap().len(),
            0
        );
    }

    #[test]
    fn mask_roundtrip_keeps_bits() {
        let mask = BooleanArray::from(vec![true, false, true, true]);
        let s = Sel::from_mask(3, &mask);
        assert_eq!(s.snapshot_id(), 3);
        assert_eq!(s.universe(), 4);
        assert_eq!(s.to_ids(), vec![0, 2, 3]);
        assert_eq!(s.to_mask().values().clone(), mask.values().clone());
    }

    #[test]
    fn ranges_merge_and_clamp() {
        let s = Sel::from_ranges(1, 10, [(5, 20), (0, 2), (2, 3), (3, 4)]);
        assert_eq!(
            s.repr(),
            &SelRepr::Ranges {
                ranges: vec![(0, 4), (5, 10)]
            }
        );
    }
}
