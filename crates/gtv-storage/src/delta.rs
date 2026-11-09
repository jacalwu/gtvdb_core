//! M1: positional retract + Snapshot Delta (bounded overlay, no row-level MVCC).
//!
//! # What this module is
//!
//! A correction never mutates a published GTVSeg. It publishes one immutable
//! [`SnapshotDelta`] that changes the **set of visible rows** of the result
//! snapshot:
//!
//! ```text
//! Snapshot N+1 = Snapshot N - Removed Positions + Added Versions (after-image)
//! ```
//!
//! `REMOVE VERSION` therefore means "this physical row is not in the result
//! snapshot's visible set" — it does **not** write a row-level `system_to`, and
//! a materialized snapshot never needs a RetractMap at query time.
//!
//! # Why positions, not u128 version ids
//!
//! A retract targets `(segment_id, row)`, i.e. a fixed-width physical address
//! that needs no global version index and no hash lookup. `removed_version_ids`
//! is kept **only** for audit / lineage and is never consulted on a read path.
//! The price is written down in `store.rs::materialize()`: folding rewrites the
//! position space, so every participating segment must be retired as a whole
//! and no dangling retract may be silently ignored.
//!
//! # Binary layout (`*.gtvd`, all little-endian, fixed width)
//!
//! ```text
//! header[96]   magic / version / ids / four section counts
//! puts[n]      n x 42 B EdgeRecord          (complete after-image)
//! removed[m]   m x 12 B (segment_id u64, row u32)
//! audit[k]     k x 16 B u128                (audit only, never read)
//! inval[j]     j x 48 B InvalidationRecord (commit-log projection)
//! trailer[32]  blake3(header || sections)
//! ```
//!
//! GTVSeg v1 is untouched: delta puts live in their own immutable file, and the
//! `disk == memory == cache line` invariant of the baseline segments holds.

use serde::{Deserialize, Serialize};

use crate::error::{Result, StorageError};
use crate::wal::{EdgeRecord, EDGE_RECORD_SIZE};

fn err(msg: impl Into<String>) -> StorageError {
    StorageError::Msg(msg.into())
}

pub const DELTA_MAGIC: [u8; 8] = *b"GTVDLT01";
pub const DELTA_FORMAT_VERSION: u16 = 1;
pub const DELTA_HEADER_SIZE: usize = 96;
pub const ROW_POS_SIZE: usize = 12;
pub const AUDIT_ID_SIZE: usize = 16;
pub const INVALIDATION_SIZE: usize = 48;
pub const DELTA_CHECKSUM_SIZE: usize = 32;

/// Physical address of one row inside one immutable published file.
///
/// `segment_id` is allocated from the same counter as baseline GTVSeg ids, so
/// positions of baseline rows and delta puts share one namespace and cannot
/// collide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RowPos {
    pub segment_id: u64,
    pub row: u32,
}

impl RowPos {
    pub const fn new(segment_id: u64, row: u32) -> Self {
        Self { segment_id, row }
    }

    /// Stable audit id derived from the position — never used for lookup.
    pub fn audit_id(self) -> u128 {
        ((self.segment_id as u128) << 64) | self.row as u128
    }

    fn write_le(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.segment_id.to_le_bytes());
        out.extend_from_slice(&self.row.to_le_bytes());
    }

    fn read_le(b: &[u8]) -> Self {
        Self {
            segment_id: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            row: u32::from_le_bytes(b[8..12].try_into().unwrap()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemoveReason {
    PropertyUpdate,
    LifetimeCorrection,
    ValidTimeCorrection,
    TopologyChange,
    LogicalDelete,
    SourceCorrection,
}

impl RemoveReason {
    pub const fn code(self) -> u8 {
        match self {
            Self::PropertyUpdate => 0,
            Self::LifetimeCorrection => 1,
            Self::ValidTimeCorrection => 2,
            Self::TopologyChange => 3,
            Self::LogicalDelete => 4,
            Self::SourceCorrection => 5,
        }
    }

    pub fn from_code(c: u8) -> Result<Self> {
        Ok(match c {
            0 => Self::PropertyUpdate,
            1 => Self::LifetimeCorrection,
            2 => Self::ValidTimeCorrection,
            3 => Self::TopologyChange,
            4 => Self::LogicalDelete,
            5 => Self::SourceCorrection,
            _ => return Err(err(format!("unknown RemoveReason code {c}"))),
        })
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::PropertyUpdate => "property_update",
            Self::LifetimeCorrection => "lifetime_correction",
            Self::ValidTimeCorrection => "valid_time_correction",
            Self::TopologyChange => "topology_change",
            Self::LogicalDelete => "logical_delete",
            Self::SourceCorrection => "source_correction",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectKind {
    Node,
    Edge,
    Observation,
}

impl ObjectKind {
    pub const fn code(self) -> u8 {
        match self {
            Self::Node => 0,
            Self::Edge => 1,
            Self::Observation => 2,
        }
    }

    pub fn from_code(c: u8) -> Result<Self> {
        Ok(match c {
            0 => Self::Node,
            1 => Self::Edge,
            2 => Self::Observation,
            _ => return Err(err(format!("unknown ObjectKind code {c}"))),
        })
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Node => "node",
            Self::Edge => "edge",
            Self::Observation => "observation",
        }
    }
}

/// One row of the commit-log projection consumed by `gtv_invalidations()`.
///
/// Fixed width by design: no event bus, no delivery state, no replay queue.
/// The event *is* the delta file, so it becomes visible atomically with the
/// manifest publish and replay is naturally idempotent by `commit_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvalidationRecord {
    pub commit_id: u64,
    pub system_ts: i64,
    pub object_kind: ObjectKind,
    pub object_id: u64,
    pub affected_valid_from: i64,
    pub affected_valid_to: i64,
    pub reason: RemoveReason,
}

impl InvalidationRecord {
    fn write_le(&self, out: &mut Vec<u8>) {
        let start = out.len();
        out.extend_from_slice(&self.commit_id.to_le_bytes());
        out.extend_from_slice(&self.system_ts.to_le_bytes());
        out.push(self.object_kind.code());
        out.push(self.reason.code());
        out.extend_from_slice(&[0u8; 2]);
        out.extend_from_slice(&self.object_id.to_le_bytes());
        out.extend_from_slice(&self.affected_valid_from.to_le_bytes());
        out.extend_from_slice(&self.affected_valid_to.to_le_bytes());
        out.extend_from_slice(&[0u8; 4]); // pad 44 -> 48 B
        debug_assert_eq!(out.len() - start, INVALIDATION_SIZE);
    }

    fn read_le(b: &[u8]) -> Result<Self> {
        Ok(Self {
            commit_id: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            system_ts: i64::from_le_bytes(b[8..16].try_into().unwrap()),
            object_kind: ObjectKind::from_code(b[16])?,
            reason: RemoveReason::from_code(b[17])?,
            object_id: u64::from_le_bytes(b[20..28].try_into().unwrap()),
            affected_valid_from: i64::from_le_bytes(b[28..36].try_into().unwrap()),
            affected_valid_to: i64::from_le_bytes(b[36..44].try_into().unwrap()),
        })
    }
}

/// V1 keeps exactly two states. `Materializing` / `Failed` belong to M2, when
/// a background worker can actually be in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterializationState {
    DeltaOnly,
    Materialized,
}

/// An immutable, published change of the visible row set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotDelta {
    pub delta_id: u64,
    /// Highest snapshot point of the base this delta was planned against.
    pub base_system_ts: i64,
    /// Snapshot point at which the result becomes visible.
    pub system_ts: i64,
    /// Commit id (WAL txn) — also the invalidation cursor.
    pub commit_id: u64,
    /// Namespace for rows in `puts`, allocated from the segment id counter.
    pub put_segment_id: u64,
    /// Complete after-images. **Never** a field-level patch chain.
    pub puts: Vec<EdgeRecord>,
    /// Sorted, deduplicated physical positions removed from the result set.
    pub removed: Vec<RowPos>,
    /// Audit / lineage only. Not consulted on any read path.
    pub removed_version_ids: Vec<u128>,
    pub invalidations: Vec<InvalidationRecord>,
}

impl SnapshotDelta {
    pub fn empty(ctx: DeltaContext) -> Self {
        Self {
            delta_id: ctx.delta_id,
            base_system_ts: ctx.base_system_ts,
            system_ts: ctx.system_ts,
            commit_id: ctx.commit_id,
            put_segment_id: ctx.put_segment_id,
            puts: Vec::new(),
            removed: Vec::new(),
            removed_version_ids: Vec::new(),
            invalidations: Vec::new(),
        }
    }

    pub fn retracts(&self) -> RetractSet {
        RetractSet::from_sorted(self.removed.clone())
    }
}

/// Identity fields supplied by the store at commit time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaContext {
    pub delta_id: u64,
    pub base_system_ts: i64,
    pub system_ts: i64,
    pub commit_id: u64,
    pub put_segment_id: u64,
}

/// Sorted retract positions. `contains` is one binary search; merging K deltas
/// is a sort, not a K-way merge of edge streams.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetractSet {
    rows: Vec<RowPos>,
}

impl RetractSet {
    pub fn from_sorted(mut rows: Vec<RowPos>) -> Self {
        rows.sort_unstable();
        rows.dedup();
        Self { rows }
    }

    pub fn merge(sets: &[Self]) -> Self {
        let mut rows = Vec::with_capacity(sets.iter().map(|s| s.rows.len()).sum());
        for s in sets {
            rows.extend_from_slice(&s.rows);
        }
        Self::from_sorted(rows)
    }

    #[inline]
    pub fn contains(&self, pos: RowPos) -> bool {
        self.rows.binary_search(&pos).is_ok()
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn rows(&self) -> &[RowPos] {
        &self.rows
    }
}

// -- commit-time correction planning ---------------------------------------

/// Logical identity of an edge version (topology only, no business interval).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeKey {
    pub src: u64,
    pub dst: u64,
    pub edge_type: u16,
}

impl EdgeKey {
    /// Exact identity of an **existing** version: all three topology fields.
    pub fn matches(&self, e: &EdgeRecord) -> bool {
        e.src == self.src && e.dst == self.dst && e.edge_type == self.edge_type
    }

    /// Endpoints are immutable across a correction; `edge_type` is the only
    /// property GTVSeg v1 stores, so an after-image may *reclassify* it. This
    /// is what makes `RemoveReason::PropertyUpdate` observable in the CSR
    /// instead of a positional no-op.
    pub fn matches_endpoints(&self, e: &EdgeRecord) -> bool {
        e.src == self.src && e.dst == self.dst
    }
}

/// One visible row together with its physical address, as seen at commit time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisibleRow {
    pub pos: RowPos,
    pub edge: EdgeRecord,
}

/// What the caller wants the correction window `[from, to)` to contain.
/// `payload = None` is a logical delete (fragments outside the window remain).
#[derive(Debug, Clone, Copy)]
pub struct Correction {
    pub window: (i64, i64),
    pub payload: Option<EdgeRecord>,
    pub reason: RemoveReason,
    pub object_id: u64,
}

/// `AssertCovered` rejects a window that is not fully backed by existing facts
/// (prevents silently inventing history). `PreserveHoles` clips the corrected
/// payload to the covered sub-intervals instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HolePolicy {
    AssertCovered,
    PreserveHoles,
}

/// Commit-time interval sweep across **all** visible versions of one key.
///
/// This is where conflicts are resolved: the result is a set of non-overlapping
/// after-images, so the read path never has to deduplicate (invariant I1).
/// Old snapshots are untouched — we only emit positions to remove from the new
/// snapshot's visible set.
pub fn plan_correction(
    ctx: DeltaContext,
    key: EdgeKey,
    visible: &[VisibleRow],
    correction: Correction,
    policy: HolePolicy,
) -> Result<SnapshotDelta> {
    let (c, d) = correction.window;
    if c >= d {
        return Err(err(format!("invalid correction window [{c}, {d})")));
    }
    if let Some(p) = correction.payload {
        if !key.matches_endpoints(&p) {
            return Err(err("correction payload changes the edge endpoints; a correction cannot move an edge"));
        }
        p.validate_interval()?;
    }

    let mut overlapping: Vec<VisibleRow> = visible
        .iter()
        .copied()
        .filter(|r| key.matches(&r.edge) && r.edge.valid_from < d && c < r.edge.valid_to)
        .collect();
    overlapping.sort_by(|a, b| {
        (a.edge.valid_from, a.edge.valid_to, a.pos).cmp(&(
            b.edge.valid_from,
            b.edge.valid_to,
            b.pos,
        ))
    });

    let mut delta = SnapshotDelta::empty(ctx);

    // Coverage intervals: intersection of each old version with the window.
    let mut covered: Vec<(i64, i64)> = overlapping
        .iter()
        .map(|r| (r.edge.valid_from.max(c), r.edge.valid_to.min(d)))
        .filter(|(a, b)| a < b)
        .collect();
    covered.sort_by_key(|x| x.0);
    let mut merged: Vec<(i64, i64)> = Vec::new();
    for (a, b) in covered {
        if let Some(last) = merged.last_mut() {
            if a <= last.1 {
                last.1 = last.1.max(b);
                continue;
            }
        }
        merged.push((a, b));
    }

    if policy == HolePolicy::AssertCovered {
        let covers = merged.len() == 1 && merged[0].0 == c && merged[0].1 == d;
        if !covers {
            return Err(err(format!(
                "correction window [{c}, {d}) is not fully covered by visible versions (policy AssertCovered); found {merged:?}"
            )));
        }
    }

    for r in &overlapping {
        delta.removed.push(r.pos);
        delta.removed_version_ids.push(r.pos.audit_id());
        // Left fragment keeps the old payload, unchanged.
        if r.edge.valid_from < c {
            let mut frag = r.edge;
            frag.valid_to = c;
            delta.puts.push(frag);
        }
        // Right fragment keeps the old payload, unchanged.
        if r.edge.valid_to > d {
            let mut frag = r.edge;
            frag.valid_from = d;
            delta.puts.push(frag);
        }
    }
    delta.removed.sort_unstable();
    delta.removed.dedup();

    if let Some(payload) = correction.payload {
        match policy {
            HolePolicy::AssertCovered => {
                let mut p = payload;
                p.valid_from = c;
                p.valid_to = d;
                delta.puts.push(p);
            }
            HolePolicy::PreserveHoles => {
                for (a, b) in &merged {
                    let mut p = payload;
                    p.valid_from = *a;
                    p.valid_to = *b;
                    delta.puts.push(p);
                }
            }
        }
    }

    delta.invalidations.push(InvalidationRecord {
        commit_id: ctx.commit_id,
        system_ts: ctx.system_ts,
        object_kind: ObjectKind::Edge,
        object_id: correction.object_id,
        affected_valid_from: c,
        affected_valid_to: d,
        reason: correction.reason,
    });
    Ok(delta)
}

// -- binary codec -----------------------------------------------------------

fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_i64(out: &mut Vec<u8>, v: i64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn get_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap())
}
fn get_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn get_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}
fn get_i64(b: &[u8], at: usize) -> i64 {
    i64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// Self-contained, checksummed, fixed-width encoding. No serde on the hot
/// write path, and a torn tail is detected by the trailer.
pub fn encode_delta(d: &SnapshotDelta) -> Result<Vec<u8>> {
    for (name, n) in [
        ("puts", d.puts.len()),
        ("removed", d.removed.len()),
        ("audit ids", d.removed_version_ids.len()),
        ("invalidations", d.invalidations.len()),
    ] {
        if n > u32::MAX as usize {
            return Err(err(format!("delta has {n} {name}, exceeds the u32 section count")));
        }
    }
    let mut out = Vec::with_capacity(
        DELTA_HEADER_SIZE
            + d.puts.len() * EDGE_RECORD_SIZE
            + d.removed.len() * ROW_POS_SIZE
            + d.removed_version_ids.len() * AUDIT_ID_SIZE
            + d.invalidations.len() * INVALIDATION_SIZE
            + DELTA_CHECKSUM_SIZE,
    );
    out.extend_from_slice(&DELTA_MAGIC);
    put_u16(&mut out, DELTA_FORMAT_VERSION);
    put_u16(&mut out, 0); // reserved
    put_u64(&mut out, d.delta_id);
    put_i64(&mut out, d.base_system_ts);
    put_i64(&mut out, d.system_ts);
    put_u64(&mut out, d.commit_id);
    put_u64(&mut out, d.put_segment_id);
    put_u32(&mut out, d.puts.len() as u32);
    put_u32(&mut out, d.removed.len() as u32);
    put_u32(&mut out, d.removed_version_ids.len() as u32);
    put_u32(&mut out, d.invalidations.len() as u32);
    put_u32(&mut out, 0);
    out.extend_from_slice(&[0u8; 24]);
    debug_assert_eq!(out.len(), DELTA_HEADER_SIZE);

    for e in &d.puts {
        e.write_le(&mut out);
    }
    for p in &d.removed {
        p.write_le(&mut out);
    }
    for id in &d.removed_version_ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    for inv in &d.invalidations {
        inv.write_le(&mut out);
    }
    out.extend_from_slice(blake3::hash(&out).as_bytes());
    Ok(out)
}

pub fn decode_delta(bytes: &[u8]) -> Result<SnapshotDelta> {
    let min = DELTA_HEADER_SIZE + DELTA_CHECKSUM_SIZE;
    if bytes.len() < min {
        return Err(err("delta file shorter than its header + checksum"));
    }
    if bytes[0..8] != DELTA_MAGIC {
        return Err(err("delta file bad magic"));
    }
    if get_u16(bytes, 8) != DELTA_FORMAT_VERSION {
        return Err(err("unsupported delta format version"));
    }
    let body_end = bytes.len() - DELTA_CHECKSUM_SIZE;
    if blake3::hash(&bytes[..body_end]).as_bytes() != &bytes[body_end..] {
        return Err(err("delta file checksum mismatch"));
    }

    let n_puts = get_u32(bytes, 52) as usize;
    let n_removed = get_u32(bytes, 56) as usize;
    let n_audit = get_u32(bytes, 60) as usize;
    let n_inv = get_u32(bytes, 64) as usize;
    let expected = DELTA_HEADER_SIZE
        .saturating_add(n_puts.saturating_mul(EDGE_RECORD_SIZE))
        .saturating_add(n_removed.saturating_mul(ROW_POS_SIZE))
        .saturating_add(n_audit.saturating_mul(AUDIT_ID_SIZE))
        .saturating_add(n_inv.saturating_mul(INVALIDATION_SIZE));
    if expected != body_end {
        return Err(err(format!(
            "delta section sizes do not match the file length (expected {expected}, got {body_end})"
        )));
    }

    let mut at = DELTA_HEADER_SIZE;
    let mut puts = Vec::with_capacity(n_puts);
    for _ in 0..n_puts {
        puts.push(EdgeRecord::read_le(&bytes[at..at + EDGE_RECORD_SIZE]));
        at += EDGE_RECORD_SIZE;
    }
    let mut removed = Vec::with_capacity(n_removed);
    for _ in 0..n_removed {
        removed.push(RowPos::read_le(&bytes[at..at + ROW_POS_SIZE]));
        at += ROW_POS_SIZE;
    }
    let mut removed_version_ids = Vec::with_capacity(n_audit);
    for _ in 0..n_audit {
        removed_version_ids.push(u128::from_le_bytes(
            bytes[at..at + AUDIT_ID_SIZE].try_into().unwrap(),
        ));
        at += AUDIT_ID_SIZE;
    }
    let mut invalidations = Vec::with_capacity(n_inv);
    for _ in 0..n_inv {
        invalidations.push(InvalidationRecord::read_le(
            &bytes[at..at + INVALIDATION_SIZE],
        )?);
        at += INVALIDATION_SIZE;
    }
    debug_assert_eq!(at, body_end);

    Ok(SnapshotDelta {
        delta_id: get_u64(bytes, 12),
        base_system_ts: get_i64(bytes, 20),
        system_ts: get_i64(bytes, 28),
        commit_id: get_u64(bytes, 36),
        put_segment_id: get_u64(bytes, 44),
        puts,
        removed,
        removed_version_ids,
        invalidations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(src: u64, from: i64, to: i64, et: u16) -> EdgeRecord {
        EdgeRecord::new(src, 42, from, to, et, 7)
    }

    fn ctx() -> DeltaContext {
        DeltaContext {
            delta_id: 1,
            base_system_ts: 10,
            system_ts: 20,
            commit_id: 3,
            put_segment_id: 99,
        }
    }

    #[test]
    fn delta_codec_roundtrip_is_fixed_width() {
        let mut d = SnapshotDelta::empty(ctx());
        d.puts.push(edge(1, 100, 200, 2));
        d.removed.push(RowPos::new(4, 7));
        d.removed_version_ids.push(RowPos::new(4, 7).audit_id());
        d.invalidations.push(InvalidationRecord {
            commit_id: 3,
            system_ts: 20,
            object_kind: ObjectKind::Edge,
            object_id: 17,
            affected_valid_from: 100,
            affected_valid_to: 200,
            reason: RemoveReason::ValidTimeCorrection,
        });
        let bytes = encode_delta(&d).unwrap();
        assert_eq!(
            bytes.len(),
            DELTA_HEADER_SIZE
                + EDGE_RECORD_SIZE
                + ROW_POS_SIZE
                + AUDIT_ID_SIZE
                + INVALIDATION_SIZE
                + DELTA_CHECKSUM_SIZE
        );
        assert_eq!(decode_delta(&bytes).unwrap(), d);

        let mut torn = bytes.clone();
        let last = torn.len() - 1;
        torn[last] ^= 0xff;
        assert!(decode_delta(&torn).unwrap_err().to_string().contains("checksum"));
    }

    #[test]
    fn retract_set_merges_and_deduplicates() {
        let a = RetractSet::from_sorted(vec![RowPos::new(1, 3), RowPos::new(1, 1)]);
        let b = RetractSet::from_sorted(vec![RowPos::new(1, 3), RowPos::new(2, 0)]);
        let m = RetractSet::merge(&[a, b]);
        assert_eq!(m.len(), 3);
        assert!(m.contains(RowPos::new(1, 1)));
        assert!(m.contains(RowPos::new(1, 3)));
        assert!(m.contains(RowPos::new(2, 0)));
        assert!(!m.contains(RowPos::new(2, 1)));
    }

    fn key() -> EdgeKey {
        EdgeKey { src: 1, dst: 42, edge_type: 2 }
    }

    fn corr(from: i64, to: i64, payload: Option<EdgeRecord>) -> Correction {
        Correction {
            window: (from, to),
            payload,
            reason: RemoveReason::ValidTimeCorrection,
            object_id: 17,
        }
    }

    #[test]
    fn middle_split_keeps_both_outer_fragments() {
        let visible = vec![VisibleRow {
            pos: RowPos::new(5, 0),
            edge: edge(1, 100, 400, 2),
        }];
        let payload = edge(1, 150, 250, 2);
        let d = plan_correction(
            ctx(), key(), &visible, corr(150, 250, Some(payload)), HolePolicy::AssertCovered,
        )
        .unwrap();
        assert_eq!(d.removed, vec![RowPos::new(5, 0)]);
        assert_eq!(d.puts.len(), 3);
        let mut spans: Vec<(i64, i64)> = d.puts.iter().map(|e| (e.valid_from, e.valid_to)).collect();
        spans.sort();
        assert_eq!(spans, vec![(100, 150), (150, 250), (250, 400)]);
        assert_eq!(d.removed_version_ids, vec![RowPos::new(5, 0).audit_id()]);
    }

    #[test]
    fn logical_delete_produces_no_corrected_row() {
        let visible = vec![VisibleRow {
            pos: RowPos::new(5, 0),
            edge: edge(1, 100, 400, 2),
        }];
        let d = plan_correction(
            ctx(), key(), &visible, corr(200, 300, None), HolePolicy::AssertCovered,
        )
        .unwrap();
        let mut spans: Vec<(i64, i64)> = d.puts.iter().map(|e| (e.valid_from, e.valid_to)).collect();
        spans.sort();
        assert_eq!(spans, vec![(100, 200), (300, 400)]);
    }

    #[test]
    fn assert_covered_rejects_holes_but_preserve_holes_clips() {
        let visible = vec![
            VisibleRow { pos: RowPos::new(5, 0), edge: edge(1, 100, 200, 2) },
            VisibleRow { pos: RowPos::new(5, 1), edge: edge(1, 300, 400, 2) },
        ];
        let payload = edge(1, 150, 350, 2);
        assert!(plan_correction(
            ctx(), key(), &visible, corr(150, 350, Some(payload)), HolePolicy::AssertCovered,
        )
        .unwrap_err()
        .to_string()
        .contains("not fully covered"));

        let d = plan_correction(
            ctx(), key(), &visible, corr(150, 350, Some(payload)), HolePolicy::PreserveHoles,
        )
        .unwrap();
        assert_eq!(d.removed.len(), 2);
        let mut spans: Vec<(i64, i64)> = d.puts.iter().map(|e| (e.valid_from, e.valid_to)).collect();
        spans.sort();
        assert_eq!(spans, vec![(100, 150), (150, 200), (300, 350), (350, 400)]);
    }

    #[test]
    fn disjoint_window_produces_no_row_change() {
        let visible = vec![VisibleRow {
            pos: RowPos::new(5, 0),
            edge: edge(1, 100, 200, 2),
        }];
        let d = plan_correction(
            ctx(), key(), &visible, corr(500, 600, None), HolePolicy::PreserveHoles,
        )
        .unwrap();
        assert!(d.removed.is_empty());
        assert!(d.puts.is_empty());
        assert_eq!(d.invalidations.len(), 1);
    }

    #[test]
    fn after_image_can_reclassify_but_not_move_an_edge() {
        let visible = vec![VisibleRow {
            pos: RowPos::new(5, 0),
            edge: edge(1, 100, 400, 2),
        }];
        // A different edge_type is a legal property update (the only property
        // GTVSeg v1 carries). A different dst is not.
        let d = plan_correction(
            ctx(), key(), &visible,
            corr(100, 400, Some(edge(1, 100, 400, 9))),
            HolePolicy::AssertCovered,
        )
        .unwrap();
        assert_eq!(d.puts.len(), 1);
        assert_eq!(d.puts[0].edge_type, 9);
        assert_eq!(d.removed, vec![RowPos::new(5, 0)]);

        let moved = plan_correction(
            ctx(), key(), &visible,
            corr(100, 400, Some(EdgeRecord::new(1, 43, 100, 400, 2, 7))),
            HolePolicy::AssertCovered,
        );
        assert!(moved.unwrap_err().to_string().contains("endpoints"));
    }

    #[test]
    fn open_ended_version_can_be_split() {
        let visible = vec![VisibleRow {
            pos: RowPos::new(5, 0),
            edge: edge(1, 0, i64::MAX, 2),
        }];
        let d = plan_correction(
            ctx(), key(), &visible, corr(100, 200, None), HolePolicy::AssertCovered,
        )
        .unwrap();
        assert_eq!(d.puts.len(), 2);
        assert_eq!(d.puts[0].valid_to, 100);
        assert_eq!(d.puts[1].valid_from, 200);
        assert_eq!(d.puts[1].valid_to, i64::MAX);
    }
}
