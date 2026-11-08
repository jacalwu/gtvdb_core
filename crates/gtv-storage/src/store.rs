//! P1 write path: WAL + `MemSegment` + immutable GTVSeg publication.
//!
//! # Crash-consistent flush order
//!
//! ```text
//! ingest(edges)
//!   1. WAL.append(frame) + fsync          <- durable, zero encoding
//!   2. MemSegment.push_batch(edges)       <- native fixed-width Vec
//!
//! flush_at(system_ts = T)
//!   3. group delta by (table, partition)  <- one segment per active partition
//!   4. sort once -> GTVSeg bytes -> write_atomic(<new file>)
//!   5. write_atomic(manifest.json)        <- checkpoint_txn + system_ts = T
//!   6. WAL.checkpoint()                   <- truncate to the 64 B header
//! ```
//!
//! A crash at any point is recoverable:
//!
//! * before (5): the checkpoint still points at the old segments, so `open`
//!   replays every WAL frame. Unreferenced segment files are orphans and are
//!   simply overwritten by the next flush (segment ids are derived from the
//!   manifest, which was not advanced).
//! * after (5) but before (6): replay skips frames with
//!   `txn_id <= checkpoint_txn`, so nothing is applied twice.
//! * during (4): `write_atomic` uses a sibling `.tmp` + rename, so a reader
//!   never sees a partially written segment.
//!
//! # Segment-level reuse (OceanBase macroblock lesson)
//!
//! A flush **only appends** new entries to the manifest. It never reopens,
//! rereads or rewrites an existing `.seg`. `FlushReport::reuse_ratio` reports
//! `preserved / (preserved + new)`; with one active partition and N flushes it
//! tends to 1.0. The read side is still a single contiguous CSR: use
//! `graph_at(T)` to materialise one, or `register_segment_store` to expose each
//! baseline segment to the P0 SQL surface.
//!
//! # Data discipline
//!
//! No test or benchmark data is stored in the repository. P1 tests build their
//! edges in memory and write only to a tempdir; **no CSV is ever produced or
//! committed** (see `.gitignore` + the `generated-data` CI job).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use gtv_core::csr::TemporalCSR;
use serde::{Deserialize, Serialize};

use crate::atomic::write_atomic;
use crate::error::{Result, StorageError};
use crate::mem_segment::MemSegment;
use crate::segment::{encode_segment, Segment, HEADER_SIZE};
use crate::segment_meta::{PruneOutcome, SegmentMeta};
use crate::wal::{EdgeRecord, Wal, WalFrame, OP_INSERT};

const MANIFEST_VERSION: u32 = 1;
const DEFAULT_FLUSH_THRESHOLD: usize = 256 * 1024 * 1024;

fn err(msg: impl Into<String>) -> StorageError {
    StorageError::Msg(msg.into())
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// One immutable, published GTVSeg plus its bitemporal/transaction envelope.
///
/// The `*_min` / `*_max` fields are **L2 pruning metadata** — the segment's
/// business-time box and node-id box, computed once at flush time from the CSR
/// (`SegmentMeta::from_csr`). They are `Option` so a manifest written before
/// this field existed still loads: `None` means "unknown", and every predicate
/// below then conservatively selects the segment instead of dropping it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentEntry {
    pub id: u64,
    /// File name relative to `<root>/segments/`.
    pub file: String,
    pub table_id: u32,
    pub partition_id: u32,
    pub txn_start: u64,
    pub txn_end: u64,
    /// Global snapshot point: this segment is `as_of_system(system_ts)`.
    pub system_ts: i64,
    pub node_count: u64,
    pub edge_count: u64,
    /// blake3 over the segment's column bytes (GTVSeg `data_checksum`).
    pub data_checksum: String,

    // -- L2 union box: "may contain" ------------------------------------
    /// Earliest `valid_from` in the segment.
    #[serde(default)]
    pub valid_from_min: Option<i64>,
    /// Latest `valid_to` in the segment (`i64::MAX` for open-ended edges).
    #[serde(default)]
    pub valid_to_max: Option<i64>,
    // -- L2 intersection window: "all edges active" ---------------------
    #[serde(default)]
    pub max_valid_from: Option<i64>,
    #[serde(default)]
    pub min_valid_to: Option<i64>,
    /// Smallest / largest node id the segment actually touches.
    #[serde(default)]
    pub node_id_min: Option<u64>,
    #[serde(default)]
    pub node_id_max: Option<u64>,
}

impl SegmentEntry {
    /// L2 metadata, when the entry carries it.
    pub fn meta(&self) -> Option<SegmentMeta> {
        Some(SegmentMeta {
            node_count: self.node_count,
            edge_count: self.edge_count,
            valid_from_min: self.valid_from_min?,
            valid_to_max: self.valid_to_max?,
            max_valid_from: self.max_valid_from?,
            min_valid_to: self.min_valid_to?,
            node_id_min: self.node_id_min?,
            node_id_max: self.node_id_max?,
            edge_type_min: 0,
            edge_type_max: 0,
        })
    }

    /// May this segment hold an edge active somewhere in `[t1, t2)`?
    ///
    /// Unknown metadata ⇒ `true` (a legacy manifest must never lose rows).
    pub fn overlaps(&self, t1: i64, t2: i64) -> bool {
        match self.meta() {
            Some(m) => m.overlaps(t1, t2),
            None => true,
        }
    }

    /// May this segment hold an edge active exactly at `t`?
    pub fn may_contain(&self, t: i64) -> bool {
        match self.meta() {
            Some(m) => m.may_contain(t),
            None => true,
        }
    }

    /// Is every edge of this segment active at `t`? Unknown ⇒ `false`.
    pub fn all_active_at(&self, t: i64) -> bool {
        self.meta().is_some_and(|m| m.all_active_at(t))
    }

    /// Three-way decision used by segment selection.
    pub fn classify(&self, t1: i64, t2: i64) -> PruneOutcome {
        match self.meta() {
            Some(m) => m.classify(t1, t2),
            None => PruneOutcome::Scan,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub node_count: u64,
    /// All WAL frames with `txn_id <= checkpoint_txn` are already in segments.
    pub checkpoint_txn: u64,
    /// Highest published snapshot point.
    pub system_ts: i64,
    pub next_segment_id: u64,
    pub segments: Vec<SegmentEntry>,
}

impl Manifest {
    fn new(node_count: usize) -> Self {
        Self {
            version: MANIFEST_VERSION,
            node_count: node_count as u64,
            checkpoint_txn: 0,
            system_ts: 0,
            next_segment_id: 1,
            segments: Vec::new(),
        }
    }
}

/// What `open()` had to do to rebuild the in-memory delta.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    pub frames_scanned: usize,
    pub frames_replayed: usize,
    pub frames_skipped_checkpoint: usize,
    pub edges_replayed: usize,
    pub torn_tail_bytes: usize,
    pub last_txn: u64,
}

/// Outcome of one flush.
#[derive(Debug, Clone, PartialEq)]
pub struct FlushReport {
    pub new_segments: usize,
    pub reused_segments: usize,
    /// `reused_segments / (reused_segments + new_segments)`; 0.0 for the first
    /// flush of an empty store, 1.0 when nothing was rewritten.
    pub reuse_ratio: f64,
    pub edges_flushed: usize,
    pub checkpoint_txn: u64,
    pub system_ts: i64,
    pub wal_bytes_reclaimed: usize,
}

/// Segment-selection outcome for one time range (`prune_summary`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PruneSummary {
    /// Segments dropped without opening the file.
    pub skipped: usize,
    /// Segments where every edge is active over the whole window.
    pub full_hit: usize,
    /// Segments that need a per-edge test.
    pub scan: usize,
    /// `full_hit + scan`.
    pub selected: usize,
}

impl PruneSummary {
    /// Fraction of segments that never had to be opened, in `[0, 1]`.
    pub fn skip_ratio(&self) -> f64 {
        let total = self.skipped + self.selected;
        if total == 0 {
            0.0
        } else {
            self.skipped as f64 / total as f64
        }
    }
}

#[derive(Debug)]
pub struct SegmentStore {
    root: PathBuf,
    node_count: usize,
    manifest: Manifest,
    wal: Wal,
    mem: MemSegment,
    next_txn: u64,
    flush_threshold_bytes: usize,
    last_recovery: RecoveryReport,
}

impl SegmentStore {
    /// Open a store, replaying any WAL frames not covered by the manifest
    /// checkpoint. `node_count` is fixed for the lifetime of the store so all
    /// segments share the same `offsets` width.
    pub fn open(root: impl AsRef<Path>, node_count: usize) -> Result<Self> {
        if node_count == 0 {
            return Err(err("SegmentStore node_count must be > 0"));
        }
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("segments"))?;

        let manifest_path = root.join("manifest.json");
        let (manifest, fresh) = if manifest_path.exists() {
            let bytes = fs::read(&manifest_path)?;
            let m: Manifest = serde_json::from_slice(&bytes)
                .map_err(|e| err(format!("cannot parse {}: {e}", manifest_path.display())))?;
            (m, false)
        } else {
            (Manifest::new(node_count), true)
        };
        if manifest.version != MANIFEST_VERSION {
            return Err(err(format!(
                "unsupported manifest version {} (reader supports {MANIFEST_VERSION})",
                manifest.version
            )));
        }
        if manifest.node_count as usize != node_count {
            return Err(err(format!(
                "manifest node_count {} != requested {node_count}",
                manifest.node_count
            )));
        }
        // Persist the initial manifest so `node_count` is pinned even when the
        // store is opened and closed without a single ingest/flush.
        if fresh {
            let bytes = serde_json::to_vec_pretty(&manifest)
                .map_err(|e| err(format!("cannot serialize manifest: {e}")))?;
            write_atomic(&manifest_path, &bytes)?;
        }

        let (wal, scan) = Wal::open(root.join("wal.log"))?;
        let mut mem = MemSegment::new();
        let mut report = RecoveryReport {
            frames_scanned: scan.frames.len(),
            torn_tail_bytes: scan.torn_bytes,
            last_txn: scan.last_txn,
            ..RecoveryReport::default()
        };
        for frame in &scan.frames {
            if frame.txn_id <= manifest.checkpoint_txn {
                report.frames_skipped_checkpoint += 1;
                continue;
            }
            if frame.op != OP_INSERT {
                return Err(err(format!(
                    "WAL txn {} has unknown op {}",
                    frame.txn_id, frame.op
                )));
            }
            report.edges_replayed += frame.edges.len();
            mem.push_batch(
                frame.txn_id,
                frame.table_id,
                frame.partition_id,
                &frame.edges,
            );
            report.frames_replayed += 1;
        }

        let next_txn = scan.last_txn.max(manifest.checkpoint_txn) + 1;
        Ok(Self {
            root,
            node_count,
            manifest,
            wal,
            mem,
            next_txn,
            flush_threshold_bytes: DEFAULT_FLUSH_THRESHOLD,
            last_recovery: report,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn node_count(&self) -> usize {
        self.node_count
    }
    pub fn segments(&self) -> &[SegmentEntry] {
        &self.manifest.segments
    }
    pub fn checkpoint_txn(&self) -> u64 {
        self.manifest.checkpoint_txn
    }
    pub fn system_ts(&self) -> i64 {
        self.manifest.system_ts
    }
    pub fn recovery(&self) -> &RecoveryReport {
        &self.last_recovery
    }
    pub fn pending_edges(&self) -> usize {
        self.mem.len()
    }
    pub fn wal_len(&self) -> u64 {
        self.wal.len()
    }
    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("manifest.json")
    }
    pub fn wal_path(&self) -> PathBuf {
        self.root.join("wal.log")
    }
    pub fn segments_dir(&self) -> PathBuf {
        self.root.join("segments")
    }
    pub fn segment_path(&self, entry: &SegmentEntry) -> PathBuf {
        self.segments_dir().join(&entry.file)
    }

    /// Override the automatic flush threshold (bytes of resident delta).
    /// Tests that need to simulate a crash before a flush set this to `usize::MAX`.
    pub fn set_flush_threshold(&mut self, bytes: usize) {
        self.flush_threshold_bytes = bytes.max(1);
    }
    pub fn flush_threshold(&self) -> usize {
        self.flush_threshold_bytes
    }

    /// Ingest one transaction. Returns before the caller can observe the edges
    /// in a segment; call `flush()`/`flush_at()` to publish them.
    pub fn ingest(
        &mut self,
        table_id: u32,
        partition_id: u32,
        edges: &[EdgeRecord],
    ) -> Result<u64> {
        if edges.is_empty() {
            return Err(err("ingest batch is empty"));
        }
        for e in edges {
            if e.src >= self.node_count as u64 {
                return Err(err(format!(
                    "edge src {} out of range (node_count {})",
                    e.src, self.node_count
                )));
            }
        }
        let txn_id = self.next_txn;
        // WAL first: if the process dies before the Vec push, replay restores it.
        self.wal.append(&WalFrame::insert(
            txn_id,
            table_id,
            partition_id,
            edges.to_vec(),
        ))?;
        self.mem.push_batch(txn_id, table_id, partition_id, edges);
        self.next_txn += 1;

        if self.mem.approx_bytes() >= self.flush_threshold_bytes {
            let ts = now_system_ts().max(self.mem.max_system_ts().unwrap_or(0));
            self.flush_at(ts)?;
        }
        Ok(txn_id)
    }

    /// Flush at the current wall-clock system time.
    pub fn flush(&mut self) -> Result<FlushReport> {
        self.flush_at(now_system_ts())
    }

    /// Publish the current delta as new immutable segments bound to a global
    /// snapshot point `system_ts`. All pending edges must be visible at `T`.
    pub fn flush_at(&mut self, system_ts: i64) -> Result<FlushReport> {
        let edges_flushed = self.mem.len();
        let reused_segments = self.manifest.segments.len();
        if edges_flushed == 0 {
            return Ok(FlushReport {
                new_segments: 0,
                reused_segments,
                reuse_ratio: 1.0,
                edges_flushed: 0,
                checkpoint_txn: self.manifest.checkpoint_txn,
                system_ts: self.manifest.system_ts,
                wal_bytes_reclaimed: 0,
            });
        }
        if system_ts < self.manifest.system_ts {
            return Err(err(format!(
                "flush system_ts {system_ts} goes backwards (manifest is at {})",
                self.manifest.system_ts
            )));
        }
        if let Some(max_ts) = self.mem.max_system_ts() {
            if system_ts < max_ts {
                return Err(err(format!(
                    "flush system_ts {system_ts} precedes an ingested edge's system_ts {max_ts}"
                )));
            }
        }
        let (txn_start, txn_end) = self
            .mem
            .txn_range()
            .ok_or_else(|| err("non-empty delta has no txn range"))?;

        // `groups()` copies the delta; `mem` is only cleared after the manifest
        // is durable, so a failed flush leaves everything replayable.
        let groups: BTreeMap<(u32, u32), Vec<EdgeRecord>> = self.mem.groups();
        let mut new_entries = Vec::with_capacity(groups.len());
        for ((table_id, partition_id), edges) in &groups {
            let csr = MemSegment::to_csr(edges, self.node_count)?;
            // L2: one linear pass over the (already sorted) CSR columns.
            // Flush-time only — the read path never recomputes this.
            let meta = SegmentMeta::from_csr(&csr);
            let bytes = encode_segment(&csr)?;
            let checksum = to_hex(blake3::hash(&bytes[HEADER_SIZE..]).as_bytes());
            let id = self.manifest.next_segment_id;
            let file = format!("seg_{id:08}_t{table_id}_p{partition_id}.seg");
            write_atomic(&self.segments_dir().join(&file), &bytes)?;
            new_entries.push(SegmentEntry {
                id,
                file,
                table_id: *table_id,
                partition_id: *partition_id,
                txn_start,
                txn_end,
                system_ts,
                node_count: meta.node_count,
                edge_count: meta.edge_count,
                data_checksum: checksum,
                valid_from_min: Some(meta.valid_from_min),
                valid_to_max: Some(meta.valid_to_max),
                max_valid_from: Some(meta.max_valid_from),
                min_valid_to: Some(meta.min_valid_to),
                node_id_min: Some(meta.node_id_min),
                node_id_max: Some(meta.node_id_max),
            });
            self.manifest.next_segment_id += 1;
        }

        let new_segments = new_entries.len();
        self.manifest.segments.extend(new_entries);
        self.manifest.checkpoint_txn = txn_end;
        self.manifest.system_ts = system_ts.max(self.manifest.system_ts);
        self.save_manifest()?;

        // Only now is it safe to drop the WAL frames.
        let wal_bytes_reclaimed = self.wal.checkpoint()?;
        self.mem.clear();

        let total = reused_segments + new_segments;
        Ok(FlushReport {
            new_segments,
            reused_segments,
            reuse_ratio: if total == 0 {
                1.0
            } else {
                reused_segments as f64 / total as f64
            },
            edges_flushed,
            checkpoint_txn: txn_end,
            system_ts,
            wal_bytes_reclaimed,
        })
    }

    fn save_manifest(&self) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(&self.manifest)
            .map_err(|e| err(format!("cannot serialize manifest: {e}")))?;
        write_atomic(&self.manifest_path(), &bytes)
    }

    /// Open one published segment and verify it against the manifest checksum.
    pub fn open_entry(&self, entry: &SegmentEntry) -> Result<Segment> {
        let seg = Segment::open(self.segment_path(entry))?;
        let got = to_hex(&seg.header().data_checksum);
        if got != entry.data_checksum {
            return Err(err(format!(
                "segment {} checksum mismatch: manifest {}, file {got}",
                entry.id, entry.data_checksum
            )));
        }
        Ok(seg)
    }

    /// Segments that form the baseline at system time `T`.
    pub fn entries_at_system(&self, t: i64) -> Vec<&SegmentEntry> {
        self.manifest
            .segments
            .iter()
            .filter(|e| e.system_ts <= t)
            .collect()
    }

    /// Segments that may hold an edge active at business time `t`.
    ///
    /// Pure manifest read: **no `.seg` file is opened**. This is the L2 win —
    /// segment selection costs a few integer comparisons per segment instead of
    /// a page fault per candidate file.
    pub fn entries_at_business(&self, t: i64) -> Vec<&SegmentEntry> {
        self.manifest
            .segments
            .iter()
            .filter(|e| e.may_contain(t))
            .collect()
    }

    /// Segments that may hold an edge active somewhere in `[t1, t2)`.
    pub fn entries_overlapping(&self, t1: i64, t2: i64) -> Vec<&SegmentEntry> {
        self.manifest
            .segments
            .iter()
            .filter(|e| e.overlaps(t1, t2))
            .collect()
    }

    /// Bitemporal selection: visible at system time `system_t` **and** possibly
    /// active at business time `business_t`.
    pub fn entries_at(&self, business_t: i64, system_t: i64) -> Vec<&SegmentEntry> {
        self.manifest
            .segments
            .iter()
            .filter(|e| e.system_ts <= system_t && e.may_contain(business_t))
            .collect()
    }

    /// Aggregate L2 box of the whole manifest (union over segments).
    ///
    /// `None` for an empty manifest. A query whose window misses this box needs
    /// no segment work at all — the cheapest possible negative answer.
    pub fn business_span(&self) -> Option<(i64, i64)> {
        self.manifest
            .segments
            .iter()
            .filter_map(|e| Some((e.valid_from_min?, e.valid_to_max?)))
            .reduce(|(a1, b1), (a2, b2)| (a1.min(a2), b1.max(b2)))
    }

    /// How many segments a `[t1, t2)` query would skip / short-circuit / scan.
    ///
    /// Telemetry, not a correctness path: the returned counts are exactly what
    /// [`Self::entries_overlapping`] used to make its decision.
    pub fn prune_summary(&self, t1: i64, t2: i64) -> PruneSummary {
        let mut out = PruneSummary::default();
        for e in &self.manifest.segments {
            match e.classify(t1, t2) {
                PruneOutcome::Skip => out.skipped += 1,
                PruneOutcome::FullHit => {
                    out.full_hit += 1;
                    out.selected += 1;
                }
                PruneOutcome::Scan => {
                    out.scan += 1;
                    out.selected += 1;
                }
            }
        }
        out
    }

    /// Materialise `as_of_system(T)` as **one contiguous CSR**.
    ///
    /// This is the "LSM write, CSR read" boundary: merging happens here, once
    /// per version query — never inside `neighbor_iter`. The returned CSR is
    /// owned; for zero-copy traversal of a single baseline use `open_entry`.
    pub fn graph_at(&self, t: i64) -> Result<TemporalCSR> {
        let mut edges = self.mem.edges_at_system(t);
        for entry in self.entries_at_system(t) {
            let seg = self.open_entry(entry)?;
            append_segment_edges(&seg, entry.system_ts, &mut edges);
        }
        MemSegment::to_csr(&edges, self.node_count)
    }

    /// Full integrity check over every published segment (offline / tests).
    pub fn verify_all(&self) -> Result<usize> {
        let mut checked = 0usize;
        for entry in &self.manifest.segments {
            let seg = self.open_entry(entry)?;
            seg.verify_data()?;
            checked += 1;
        }
        Ok(checked)
    }
}

fn append_segment_edges(seg: &Segment, system_ts: i64, out: &mut Vec<EdgeRecord>) {
    let c = seg.csr().columns();
    out.reserve(c.dst.len());
    for s in 0..c.node_count {
        let (a, b) = (c.offsets[s] as usize, c.offsets[s + 1] as usize);
        for e in a..b {
            out.push(EdgeRecord {
                src: s as u64,
                dst: c.dst[e],
                valid_from: c.valid_from[e],
                valid_to: c.valid_to[e],
                edge_type: c.edge_type[e],
                system_ts,
            });
        }
    }
}

fn now_system_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as i64
}
