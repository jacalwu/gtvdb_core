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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::record_batch::RecordBatch;
use gtv_core::csr::{CsrColumns, TemporalCSR, CHUNK_SIZE};
use gtv_core::endpoint::NodeLifetime;
use gtv_core::GtvError;
use serde::{Deserialize, Serialize};

use crate::aggregate::{AggregateView, LogicalEdge};
use crate::atomic::write_atomic;
use crate::bitemporal::BitemporalStore;
use crate::delta::{
    decode_delta, encode_delta, plan_correction, Correction, DeltaContext, EdgeKey, HolePolicy,
    InvalidationRecord, MaterializationState, ObjectKind, RemoveReason, RetractSet, RowPos,
    SnapshotDelta, VisibleRow,
};
use crate::error::{Result, StorageError};
use crate::gc::{reader_count, remove_file, GcPolicy, GcReport, ReadScope};
use crate::mem_segment::{IngestedEdge, MemSegment};
use crate::node_lifetime::{decode_node_lifetime, encode_node_lifetime};
use crate::observation::{
    encode_observation, read_observation, version_at, ObservationEntry,
};
use crate::segment::{encode_segment, Segment, HEADER_SIZE};
use crate::segment_meta::{PruneOutcome, SegmentMeta};
use crate::wal::{EdgeRecord, Wal, WalFrame, OP_INSERT, OP_MUTATION};

const MANIFEST_VERSION: u32 = 1;
const DEFAULT_FLUSH_THRESHOLD: usize = 256 * 1024 * 1024;
/// Bounded overlay: at most one `DeltaOnly` snapshot delta may be live. The
/// second correction forces materialization first, so neither a RetractMap nor
/// a K-way merge ever reaches `neighbors()`.
pub const MAX_DELTA_DEPTH: usize = 1;
/// Namespace for unflushed WAL-replayed rows. Never a retract target.
const MEM_ROW_NAMESPACE: u64 = u64::MAX;

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
    /// Tight box over **sources** only (L2 point-query pruning).
    #[serde(default)]
    pub src_id_min: Option<u64>,
    #[serde(default)]
    pub src_id_max: Option<u64>,
    /// Snapshot point at which this segment was folded into a materialized
    /// base. `None` = still visible; `Some(M)` = only visible for `t < M`.
    ///
    /// This is what makes a physical rewrite *not* a logical change: old
    /// `AS KNOWN AT` queries still see the original segment set.
    #[serde(default)]
    pub superseded_at: Option<i64>,
}

/// One immutable, published `SnapshotDelta` file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeltaEntry {
    pub id: u64,
    /// Relative to `<root>/deltas/`.
    pub file: String,
    pub base_system_ts: i64,
    pub system_ts: i64,
    pub commit_id: u64,
    pub put_segment_id: u64,
    pub put_count: u64,
    pub removed_count: u64,
    pub invalidation_count: u64,
    pub data_checksum: String,
    pub state: MaterializationState,
    /// `None` = still part of the bounded overlay; `Some(M)` = folded at M.
    #[serde(default)]
    pub superseded_at: Option<i64>,
}

/// One immutable node-lifetime snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeVersionEntry {
    pub system_ts: i64,
    /// Relative to `<root>/nodes/`.
    pub file: String,
    pub node_count: u64,
    pub data_checksum: String,
    pub commit_id: u64,
}

/// A node-lifetime correction, written alongside the delta in the **same**
/// atomic manifest publish. There is no separate node commit, so a reader can
/// never observe an edge correction without its node version (or vice versa).
#[derive(Debug, Clone)]
pub struct NodePublish {
    pub lifetime: NodeLifetime,
    pub object_id: u64,
    pub window: (i64, i64),
}

/// One visible row at a given system time, with its stable physical address
/// and the routing keys needed to re-partition it during compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveRow {
    pub pos: RowPos,
    pub edge: EdgeRecord,
    pub table_id: u32,
    pub partition_id: u32,
}

/// A deferred interval sweep. The store runs it **after** any forced
/// compaction, so retract positions always refer to the current position
/// space — a pre-planned delta would dangle as soon as compaction rewrote
/// the segment ids.
#[derive(Debug, Clone, Copy)]
pub struct EdgeCorrectionPlan {
    pub key: EdgeKey,
    pub correction: Correction,
    pub policy: HolePolicy,
}

/// Caller-supplied body of a correction. Identity fields (`delta_id`,
/// `commit_id`, `put_segment_id`, `base_system_ts`) are assigned by the store,
/// so a caller cannot forge a position namespace or a commit cursor.
#[derive(Debug, Clone, Default)]
pub struct DeltaDraft {
    /// Requested snapshot point; the store bumps it if it is not strictly
    /// newer than the current head.
    pub system_ts: i64,
    pub puts: Vec<EdgeRecord>,
    pub removed: Vec<RowPos>,
    pub invalidations: Vec<InvalidationRecord>,
    pub node: Option<NodePublish>,
    /// When set, `puts` / `removed` are computed by `plan_correction` inside
    /// the publish (compaction-safe). Mutually exclusive with explicit ones.
    pub plan: Option<EdgeCorrectionPlan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishReport {
    pub delta_id: u64,
    pub commit_id: u64,
    pub system_ts: i64,
    pub put_count: usize,
    pub removed_count: usize,
    pub invalidation_count: usize,
    /// A second correction arrived while one overlay was live, so the store
    /// folded the first one before publishing this one (K = 1 bound).
    pub forced_materialization: bool,
    pub state: MaterializationState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializeReport {
    pub system_ts: i64,
    pub new_segments: usize,
    pub retired_segments: usize,
    pub retired_deltas: usize,
    pub rows: usize,
}

/// One row of the read-only invalidation projection exposed by
/// `gtv_invalidations()`. Derived from the manifest + delta files, never from
/// an event bus, so it becomes visible exactly at the manifest publish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidationRow {
    pub commit_id: u64,
    pub system_ts: i64,
    pub object_kind: String,
    pub object_id: u64,
    pub affected_valid_from: i64,
    pub affected_valid_to: i64,
    pub reason: String,
    pub delta_id: u64,
    pub state: String,
    pub file: String,
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
            // A legacy manifest has no src box; the sentinel makes
            // `may_contain_src` answer "unknown", i.e. never prune.
            src_id_min: self.src_id_min.unwrap_or(u64::MAX),
            src_id_max: self.src_id_max.unwrap_or(0),
            edge_type_min: 0,
            edge_type_max: 0,
        })
    }

    /// Could this segment have `src` as a source node? Unknown ⇒ `true`.
    pub fn may_contain_src(&self, src: u64) -> bool {
        self.meta()
            .map_or(true, |m| m.may_contain_src(src))
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
    /// Append-only commit log of corrections. `#[serde(default)]` keeps every
    /// pre-M1 manifest loadable (and, importantly, still correct).
    #[serde(default)]
    pub deltas: Vec<DeltaEntry>,
    #[serde(default)]
    pub node_versions: Vec<NodeVersionEntry>,
    #[serde(default)]
    pub next_delta_id: u64,
    /// M2: persisted system-time versions of observation tables.
    #[serde(default)]
    pub observations: Vec<ObservationEntry>,
    #[serde(default)]
    pub next_observation_id: u64,
    /// M2: `AS KNOWN AT t < gc_floor` is no longer answerable because the
    /// files it needs were reclaimed by delayed GC. `None` = full history.
    #[serde(default)]
    pub gc_floor: Option<i64>,
    /// Highest published commit id (graph deltas + observation versions).
    /// Observation versions have no WAL frame, so the cursor must survive a
    /// restart through the manifest rather than the WAL tail.
    #[serde(default)]
    pub next_commit_id: u64,
    /// M3: lightweight named snapshot references (`name -> system_ts`).
    ///
    /// A tag pins one published point so delayed GC cannot collect the files
    /// needed to answer `AS KNOWN AT` it. This is **not** a branch: the
    /// manifest stays a single linear history (review R6). Creating or
    /// dropping a tag never advances `system_ts`, so an in-flight compaction
    /// plan does not go stale.
    #[serde(default)]
    pub tags: BTreeMap<String, i64>,
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
            deltas: Vec::new(),
            node_versions: Vec::new(),
            next_delta_id: 1,
            observations: Vec::new(),
            next_observation_id: 1,
            gc_floor: None,
            next_commit_id: 0,
            tags: BTreeMap::new(),
        }
    }

    fn next_delta_id(&mut self) -> u64 {
        // Old manifests may carry 0 after serde(default): never reuse id 0.
        let id = self.next_delta_id.max(1);
        self.next_delta_id = id + 1;
        id
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
    /// Mutation frame whose delta is already referenced by the manifest
    /// (crash after publish, before WAL checkpoint).
    pub mutations_published: usize,
    /// Mutation frame with no manifest entry (crash before publish) — the
    /// correction is deliberately **not** replayed.
    pub mutations_orphaned: usize,
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

/// Borrowed view of every input needed to reconstruct one snapshot.
///
/// M2 keeps this as a *view* rather than an owned struct: the synchronous
/// version query pays nothing, while a background compaction can clone the
/// same data once and call the very same reconstruction function. There is
/// still only one retract fold in the codebase.
#[derive(Clone, Copy)]
struct LiveRowsView<'a> {
    root: &'a Path,
    node_count: usize,
    manifest: &'a Manifest,
    deltas: &'a HashMap<u64, SnapshotDelta>,
    mem_records: &'a [IngestedEdge],
}

/// Owned snapshot of the read-only state, `Send` so `spawn_compaction()` can
/// move it to a worker thread without locking the store.
#[derive(Debug, Clone)]
pub struct CompactionInput {
    pub root: PathBuf,
    pub node_count: usize,
    pub manifest: Manifest,
    pub deltas: HashMap<u64, SnapshotDelta>,
    pub mem_records: Vec<IngestedEdge>,
    /// Shared with the store so GC cannot delete files under a worker.
    pub readers: Arc<AtomicUsize>,
}

impl CompactionInput {
    fn view(&self) -> LiveRowsView<'_> {
        LiveRowsView {
            root: &self.root,
            node_count: self.node_count,
            manifest: &self.manifest,
            deltas: &self.deltas,
            mem_records: &self.mem_records,
        }
    }

    /// M3: run phase 1 from an owned read-only snapshot. This is the only
    /// entry point a background maintenance worker needs, so no private view
    /// type leaks out of `store.rs`. Pure CPU: no file, no manifest, no lock.
    pub fn prepare(&self, system_ts: i64) -> Result<CompactionPlan> {
        prepare_compaction(self.view(), system_ts)
    }

    /// Live (not yet materialized) delta count in this snapshot. The worker
    /// uses it as its compaction trigger without borrowing the store.
    pub fn delta_depth(&self) -> usize {
        self.manifest
            .deltas
            .iter()
            .filter(|d| d.superseded_at.is_none())
            .count()
    }

    pub fn head(&self) -> i64 {
        self.manifest.system_ts
    }
}

/// Result of the **prepare** phase: fully encoded segments, no disk writes and
/// no manifest change. `apply_compaction()` validates the three base cursors
/// before publishing, so a stale plan can never rewrite newer data.
#[derive(Debug, Clone)]
pub struct CompactionPlan {
    pub base_head: i64,
    pub base_next_segment_id: u64,
    pub base_checkpoint_txn: u64,
    pub system_ts: i64,
    pub segments: Vec<(SegmentEntry, Vec<u8>)>,
    pub rows: usize,
}

#[derive(Debug)]
pub struct SegmentStore {
    root: PathBuf,
    node_count: usize,
    manifest: Manifest,
    wal: Wal,
    mem: MemSegment,
    /// Decoded delta files, keyed by delta id. Loaded once on open so a
    /// materialisation / overlay walk never touches disk per query.
    deltas: HashMap<u64, SnapshotDelta>,
    next_txn: u64,
    flush_threshold_bytes: usize,
    last_recovery: RecoveryReport,
    /// In-flight version queries / background compactions. Delayed GC refuses
    /// to unlink anything while this is non-zero.
    readers: Arc<AtomicUsize>,
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

        // Every delta referenced by the manifest is authoritative. A missing
        // or corrupt file is a hard error, not a silent skip: the manifest is
        // the single source of truth for the visible set.
        let mut deltas = HashMap::new();
        for de in &manifest.deltas {
            let bytes = fs::read(root.join("deltas").join(&de.file)).map_err(|e| {
                err(format!("cannot read delta {} for manifest: {e}", de.file))
            })?;
            let d = decode_delta(&bytes)
                .map_err(|e| err(format!("cannot decode delta {}: {e}", de.file)))?;
            if d.delta_id != de.id {
                return Err(err(format!(
                    "delta {} carries id {} (manifest says {})",
                    de.file, d.delta_id, de.id
                )));
            }
            deltas.insert(de.id, d);
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
            match frame.op {
                OP_INSERT => {
                    report.edges_replayed += frame.edges.len();
                    mem.push_batch(
                        frame.txn_id,
                        frame.table_id,
                        frame.partition_id,
                        &frame.edges,
                    );
                    report.frames_replayed += 1;
                }
                OP_MUTATION => {
                    // WAL is an *intent* log for corrections. A published
                    // correction lives in the manifest + delta file; an
                    // unlisted one is an orphan from a crash before publish
                    // and must NOT be applied (that would resurrect an
                    // uncommitted correction).
                    let d = frame.mutation.as_ref().ok_or_else(|| {
                        err(format!("WAL txn {} is OP_MUTATION but has no delta", frame.txn_id))
                    })?;
                    if manifest.deltas.iter().any(|de| de.id == d.delta_id) {
                        report.mutations_published += 1;
                    } else {
                        report.mutations_orphaned += 1;
                    }
                }
                other => {
                    return Err(err(format!(
                        "WAL txn {} has unknown op {}",
                        frame.txn_id, other
                    )))
                }
            }
        }

        let max_delta_commit = manifest.deltas.iter().map(|d| d.commit_id).max().unwrap_or(0);
        let next_txn = scan
            .last_txn
            .max(manifest.checkpoint_txn)
            .max(manifest.next_commit_id)
            .max(max_delta_commit)
            + 1;
        Ok(Self {
            root,
            node_count,
            manifest,
            wal,
            mem,
            deltas,
            next_txn,
            flush_threshold_bytes: DEFAULT_FLUSH_THRESHOLD,
            last_recovery: report,
            readers: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Borrowed reconstruction input for a synchronous control-plane call.
    fn view(&self) -> LiveRowsView<'_> {
        LiveRowsView {
            root: &self.root,
            node_count: self.node_count,
            manifest: &self.manifest,
            deltas: &self.deltas,
            mem_records: self.mem.records(),
        }
    }

    /// Owned reconstruction input for `spawn_compaction()`.
    pub fn compaction_input(&self) -> CompactionInput {
        CompactionInput {
            root: self.root.clone(),
            node_count: self.node_count,
            manifest: self.manifest.clone(),
            deltas: self.deltas.clone(),
            mem_records: self.mem.records().to_vec(),
            readers: self.readers.clone(),
        }
    }

    /// History below the retention floor has been physically reclaimed by a
    /// delayed GC. Fail loudly instead of returning a different graph.
    fn check_history(&self, t: i64) -> Result<()> {
        if let Some(floor) = self.manifest.gc_floor {
            if t < floor {
                return Err(err(format!(
                    "AS KNOWN AT {t} is below the GC retention floor {floor}; \
                     this history has been collected"
                )));
            }
        }
        Ok(())
    }

    /// In-flight reader tickets (mmap readers + background compactions).
    pub fn active_readers(&self) -> usize {
        reader_count(&self.readers)
    }

    /// Explicit RAII ticket for an external `open_entry()` user. Holding one
    /// prevents delayed GC from unlinking the file underneath the mapping.
    pub fn read_scope(&self) -> ReadScope {
        ReadScope::new(&self.readers)
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }
    pub fn segments(&self) -> &[SegmentEntry] {
        &self.manifest.segments
    }
    pub fn deltas(&self) -> &[DeltaEntry] {
        &self.manifest.deltas
    }
    pub fn node_versions(&self) -> &[NodeVersionEntry] {
        &self.manifest.node_versions
    }
    /// Number of live (not yet materialized) snapshot deltas. Must stay
    /// `<= MAX_DELTA_DEPTH` after every publish.
    pub fn delta_depth(&self) -> usize {
        self.manifest
            .deltas
            .iter()
            .filter(|d| d.superseded_at.is_none())
            .count()
    }
    pub fn deltas_dir(&self) -> PathBuf {
        self.root.join("deltas")
    }
    pub fn nodes_dir(&self) -> PathBuf {
        self.root.join("nodes")
    }
    pub fn delta_path(&self, entry: &DeltaEntry) -> PathBuf {
        self.deltas_dir().join(&entry.file)
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
                src_id_min: Some(meta.src_id_min),
                src_id_max: Some(meta.src_id_max),
                superseded_at: None,
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
        self.save_manifest_as(&self.manifest)
    }

    /// Publish an arbitrary candidate manifest atomically. Used when a publish
    /// must be computed before any of its files become visible.
    fn save_manifest_as(&self, m: &Manifest) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(m)
            .map_err(|e| err(format!("cannot serialize manifest: {e}")))?;
        write_atomic(&self.manifest_path(), &bytes)
    }

    /// Open one published segment and verify it against the manifest checksum.
    pub fn open_entry(&self, entry: &SegmentEntry) -> Result<Segment> {
        open_entry_from(&self.root, entry)
    }

    /// Segments that form the baseline at system time `T`.
    ///
    /// A segment folded into a materialized base at `M` stays visible for all
    /// `t < M` — that is what keeps `AS KNOWN AT` stable across a purely
    /// physical compaction.
    pub fn entries_at_system(&self, t: i64) -> Vec<&SegmentEntry> {
        self.manifest
            .segments
            .iter()
            .filter(|e| e.system_ts <= t && e.superseded_at.map_or(true, |m| m > t))
            .collect()
    }

    /// Head-view segment selection with the tight src box. Used by
    /// `neighbors(src, t)` to avoid opening a segment whose src run is empty.
    pub fn entries_at_src(&self, src: u64, t: i64) -> Vec<&SegmentEntry> {
        self.manifest
            .segments
            .iter()
            .filter(|e| {
                e.system_ts <= t
                    && e.superseded_at.is_none()
                    && e.may_contain(t)
                    && e.may_contain_src(src)
            })
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
            .filter(|e| e.superseded_at.is_none() && e.may_contain(t))
            .collect()
    }

    /// Segments that may hold an edge active somewhere in `[t1, t2)`.
    pub fn entries_overlapping(&self, t1: i64, t2: i64) -> Vec<&SegmentEntry> {
        self.manifest
            .segments
            .iter()
            .filter(|e| e.superseded_at.is_none() && e.overlaps(t1, t2))
            .collect()
    }

    /// Bitemporal selection: visible at system time `system_t` **and** possibly
    /// active at business time `business_t`.
    pub fn entries_at(&self, business_t: i64, system_t: i64) -> Vec<&SegmentEntry> {
        self.manifest
            .segments
            .iter()
            .filter(|e| {
                e.system_ts <= system_t
                    && e.superseded_at.map_or(true, |m| m > system_t)
                    && e.may_contain(business_t)
            })
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
            .filter(|e| e.superseded_at.is_none())
            .filter_map(|e| Some((e.valid_from_min?, e.valid_to_max?)))
            .reduce(|(a1, b1), (a2, b2)| (a1.min(a2), b1.max(b2)))
    }

    /// How many segments a `[t1, t2)` query would skip / short-circuit / scan.
    ///
    /// Telemetry, not a correctness path: the returned counts are exactly what
    /// [`Self::entries_overlapping`] used to make its decision.
    pub fn prune_summary(&self, t1: i64, t2: i64) -> PruneSummary {
        let mut out = PruneSummary::default();
        for e in self.manifest.segments.iter().filter(|e| e.superseded_at.is_none()) {
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

    // -- M1: visible-set reconstruction (the only retract fold) ------------

    /// `superseded_at = M` means "folded into the materialized base at M":
    /// the delta is still applied for every older `t < M`, but at `t >= M`
    /// its effect is already baked into the replacement segments. The
    /// authoritative filter lives in `live_rows_from()`, shared with workers.
    fn decoded_delta(&self, id: u64) -> Result<&SnapshotDelta> {
        decoded_delta_from(&self.deltas, id)
    }

    /// Enumerate every visible row at system time `t`, with its stable physical
    /// address and routing keys.
    ///
    /// This is the **single** place the retract overlay is applied. Version
    /// queries, correction planning and compaction all call it, so they cannot
    /// disagree about what "visible" means. The merged `RetractSet` is one
    /// sorted vector; the per-row check is a binary search in the control
    /// plane, never inside `TemporalCSR::neighbors()`.
    pub fn live_rows_at(&self, t: i64) -> Result<Vec<LiveRow>> {
        // One atomic ticket for the whole reconstruction; delayed GC refuses
        // to unlink a file while this scope is alive.
        let _scope = ReadScope::new(&self.readers);
        self.check_history(t)?;
        live_rows_from(self.view(), t)
    }

    /// Sorted logical edge tuple at `t`, excluding `system_ts` (which is not a
    /// GTVSeg column and is intentionally lost when a snapshot is compacted).
    /// This is the equality oracle for the §23.6 property tests.
    pub fn visible_edges_at(&self, t: i64) -> Result<Vec<(u64, u64, i64, i64, u16)>> {
        let mut v: Vec<_> = self
            .live_rows_at(t)?
            .iter()
            .map(|r| {
                let e = r.edge;
                (e.src, e.dst, e.valid_from, e.valid_to, e.edge_type)
            })
            .collect();
        v.sort_unstable();
        Ok(v)
    }

    /// Latest node-lifetime snapshot visible at `t`. Absent ⇒ every node is
    /// alive on the whole time axis (the pre-M1 behaviour).
    pub fn node_lifetime_at(&self, t: i64) -> Result<NodeLifetime> {
        let _scope = ReadScope::new(&self.readers);
        self.check_history(t)?;
        let entry = self
            .manifest
            .node_versions
            .iter()
            .filter(|v| v.system_ts <= t)
            .max_by_key(|v| v.system_ts);
        match entry {
            None => Ok(NodeLifetime::forever(self.node_count)),
            Some(v) => {
                let bytes = fs::read(self.nodes_dir().join(&v.file)).map_err(|e| {
                    err(format!("cannot read node version {}: {e}", v.file))
                })?;
                decode_node_lifetime(&bytes)
            }
        }
    }

    // -- M1: correction publish --------------------------------------------

    /// Plan and publish one retroactive **edge** correction.
    ///
    /// `correction.window` is business time; `system_ts` is the snapshot at
    /// which the correction becomes known. Old snapshots are untouched.
    pub fn correct_edge(
        &mut self,
        key: EdgeKey,
        correction: Correction,
        policy: HolePolicy,
        system_ts: i64,
    ) -> Result<PublishReport> {
        self.publish_delta(DeltaDraft {
            system_ts,
            plan: Some(EdgeCorrectionPlan {
                key,
                correction,
                policy,
            }),
            ..DeltaDraft::default()
        })
    }

    /// Publish a **node** lifetime correction together with its invalidation
    /// event in the same atomic manifest write. `nodes_<commit>.bin` is an
    /// immutable whole-column snapshot; old versions stay reachable via
    /// [`Self::node_lifetime_at`].
    pub fn correct_node(
        &mut self,
        node: u64,
        from: i64,
        to: i64,
        system_ts: i64,
    ) -> Result<PublishReport> {
        if node >= self.node_count as u64 {
            return Err(err(format!(
                "node {node} out of range (node_count {})",
                self.node_count
            )));
        }
        if from > to {
            return Err(err(format!("node lifetime [{from}, {to}) is inverted")));
        }
        let current = self.node_lifetime_at(self.manifest.system_ts)?;
        let (f, t) = current.columns();
        if f.len() != self.node_count {
            return Err(err("node lifetime snapshot length does not match node_count"));
        }
        let mut vf = f.to_vec();
        let mut vt = t.to_vec();
        vf[node as usize] = from;
        vt[node as usize] = to;
        let lifetime = NodeLifetime::new(vf, vt)
            .map_err(|e| err(format!("invalid node lifetime: {e}")))?;

        self.publish_delta(DeltaDraft {
            system_ts,
            invalidations: vec![InvalidationRecord {
                commit_id: 0,
                system_ts: 0,
                object_kind: ObjectKind::Node,
                object_id: node,
                affected_valid_from: from,
                affected_valid_to: to,
                reason: RemoveReason::LifetimeCorrection,
            }],
            node: Some(NodePublish {
                lifetime,
                object_id: node,
                window: (from, to),
            }),
            ..DeltaDraft::default()
        })
    }

    /// Low-level publish. The **manifest write is the single publish point**:
    /// the WAL mutation frame is only an intent, and the delta/node files are
    /// unreferenced orphans until that write becomes durable. This is what
    /// makes `gtv_invalidations()` atomically visible (review R4).
    pub fn publish_delta(&mut self, draft: DeltaDraft) -> Result<PublishReport> {
        if !self.mem.is_empty() {
            return Err(err("flush pending inserts before publishing a delta: positional retract targets published segments only"));
        }
        // Snapshot points must strictly advance. +2 leaves room for the
        // forced compaction point when the K=1 overlay is already occupied.
        let target = draft.system_ts.max(self.manifest.system_ts + 2);
        let forced_materialization = self.delta_depth() >= MAX_DELTA_DEPTH;
        if forced_materialization {
            // Compaction rewrites the whole position space. A caller that
            // already computed explicit retract targets cannot be carried
            // across it; `correct_edge()` instead defers its sweep via `plan`.
            if draft.plan.is_none() && !draft.removed.is_empty() {
                return Err(err(
                    "explicit retract positions cannot survive a forced compaction; \
                     call materialize() first or use correct_edge()",
                ));
            }
            self.materialize_at(target - 1)?;
        }
        let system_ts = target.max(self.manifest.system_ts + 1);
        let base_system_ts = self.manifest.system_ts;

        let mut next_manifest = self.manifest.clone();
        let delta_id = next_manifest.next_delta_id();
        let put_segment_id = next_manifest.next_segment_id;
        if put_segment_id == MEM_ROW_NAMESPACE {
            return Err(err("segment id namespace exhausted"));
        }
        next_manifest.next_segment_id = put_segment_id + 1;
        let commit_id = self.next_txn;

        let DeltaDraft {
            mut puts,
            mut removed,
            mut invalidations,
            node,
            plan,
            ..
        } = draft;
        if let Some(p) = plan {
            if !puts.is_empty() || !removed.is_empty() {
                return Err(err(
                    "DeltaDraft.plan is mutually exclusive with explicit puts/removed",
                ));
            }
            let visible: Vec<VisibleRow> = self
                .live_rows_at(base_system_ts)?
                .iter()
                .filter(|r| {
                    r.pos.segment_id != MEM_ROW_NAMESPACE && p.key.matches(&r.edge)
                })
                .map(|r| VisibleRow {
                    pos: r.pos,
                    edge: r.edge,
                })
                .collect();
            // Identity fields are provisional: `publish_delta` assigns the
            // authoritative ids and overwrites the invalidation cursor.
            let ctx = DeltaContext {
                delta_id: 0,
                base_system_ts,
                system_ts,
                commit_id: 0,
                put_segment_id: 0,
            };
            let d = plan_correction(ctx, p.key, &visible, p.correction, p.policy)?;
            puts = d.puts;
            removed = d.removed;
            invalidations.extend(d.invalidations);
        }
        removed.sort_unstable();
        removed.dedup();
        if !removed.is_empty() {
            let live: HashSet<RowPos> = self
                .live_rows_at(base_system_ts)?
                .iter()
                .filter(|r| r.pos.segment_id != MEM_ROW_NAMESPACE)
                .map(|r| r.pos)
                .collect();
            for p in &removed {
                if p.segment_id == MEM_ROW_NAMESPACE || !live.contains(p) {
                    return Err(err(format!(
                        "retract target ({}, {}) is not a visible published row",
                        p.segment_id, p.row
                    )));
                }
            }
        }
        for e in &puts {
            e.validate_interval()?;
            if e.src >= self.node_count as u64 {
                return Err(err(format!(
                    "correction put src {} out of range (node_count {})",
                    e.src, self.node_count
                )));
            }
        }
        for inv in &mut invalidations {
            inv.commit_id = commit_id;
            inv.system_ts = system_ts;
        }
        let put_count = puts.len();
        let removed_count = removed.len();
        let invalidation_count = invalidations.len();

        let delta = SnapshotDelta {
            delta_id,
            base_system_ts,
            system_ts,
            commit_id,
            put_segment_id,
            puts,
            removed: removed.clone(),
            removed_version_ids: removed.iter().map(|p| p.audit_id()).collect(),
            invalidations,
        };
        let bytes = encode_delta(&delta)?;
        let checksum = to_hex(blake3::hash(&bytes).as_bytes());

        // 1. Durable intent. A crash after this and before step 4 leaves an
        //    *orphan* frame that `open()` deliberately refuses to apply.
        self.wal.append(&WalFrame::mutation(commit_id, delta.clone()))?;

        // 2. Immutable delta file.
        fs::create_dir_all(self.deltas_dir())?;
        let file = format!("delta_{delta_id:08}_c{commit_id}.gtvd");
        write_atomic(&self.deltas_dir().join(&file), &bytes)?;

        // 3. Immutable node snapshot, same publish transaction.
        let mut node_entry = None;
        if let Some(np) = &node {
            let nb = encode_node_lifetime(&np.lifetime)?;
            let nfile = format!("nodes_c{commit_id:012}.bin");
            fs::create_dir_all(self.nodes_dir())?;
            write_atomic(&self.nodes_dir().join(&nfile), &nb)?;
            node_entry = Some(NodeVersionEntry {
                system_ts,
                file: nfile,
                node_count: np.lifetime.len() as u64,
                data_checksum: to_hex(blake3::hash(&nb).as_bytes()),
                commit_id,
            });
        }

        // 4. Single atomic publish: delta entry, invalidation projection and
        //    node version all become visible together.
        let entry = DeltaEntry {
            id: delta_id,
            file,
            base_system_ts,
            system_ts,
            commit_id,
            put_segment_id,
            put_count: put_count as u64,
            removed_count: removed_count as u64,
            invalidation_count: invalidation_count as u64,
            data_checksum: checksum,
            state: MaterializationState::DeltaOnly,
            superseded_at: None,
        };
        next_manifest.deltas.push(entry);
        if let Some(ne) = node_entry {
            next_manifest.node_versions.push(ne);
        }
        next_manifest.next_commit_id = commit_id;
        next_manifest.system_ts = system_ts;
        let manifest_bytes = serde_json::to_vec_pretty(&next_manifest)
            .map_err(|e| err(format!("cannot serialize manifest: {e}")))?;
        write_atomic(&self.manifest_path(), &manifest_bytes)?;

        self.manifest = next_manifest;
        self.deltas.insert(delta_id, delta);
        self.next_txn = commit_id + 1;
        Ok(PublishReport {
            delta_id,
            commit_id,
            system_ts,
            put_count,
            removed_count,
            invalidation_count,
            forced_materialization,
            state: MaterializationState::DeltaOnly,
        })
    }

    /// Fold the bounded overlay into new immutable segments.
    ///
    /// A physical rewrite must not be a logical change: every participating
    /// old segment and every live delta is retired **at** `system_ts`, so all
    /// `AS KNOWN AT t < system_ts` queries still see the original set. The new
    /// segments carry exactly the folded head row set.
    pub fn materialize_at(&mut self, system_ts: i64) -> Result<MaterializeReport> {
        if !self.mem.is_empty() {
            return Err(err("flush pending inserts before materializing"));
        }
        // The synchronous path uses the very same two phases as the worker
        // thread, so forced materialization can never diverge from a spawned
        // compaction.
        let plan = prepare_compaction(self.view(), system_ts)?;
        self.apply_compaction(plan)
    }

    /// Phase 1 on a worker thread. The store is **not** locked: only the
    /// read-only reconstruction state is cloned, CPU work (CSR build + GTVSeg
    /// encode) happens on the worker, and no file or manifest is touched.
    /// The shared reader ticket keeps GC from unlinking inputs meanwhile.
    pub fn spawn_compaction(&self, system_ts: i64) -> Result<JoinHandle<Result<CompactionPlan>>> {
        let input = self.compaction_input();
        Ok(std::thread::spawn(move || {
            let _scope = ReadScope::new(&input.readers);
            prepare_compaction(input.view(), system_ts)
        }))
    }

    /// Phase 2: short critical section that writes the prepared segments and
    /// publishes one manifest. A plan prepared against an older head is
    /// rejected outright — its row positions no longer describe the store.
    pub fn apply_compaction(&mut self, plan: CompactionPlan) -> Result<MaterializeReport> {
        let _scope = ReadScope::new(&self.readers);
        if !self.mem.is_empty() {
            return Err(err("flush pending inserts before applying a compaction"));
        }
        if self.manifest.system_ts != plan.base_head
            || self.manifest.next_segment_id != plan.base_next_segment_id
            || self.manifest.checkpoint_txn != plan.base_checkpoint_txn
        {
            return Err(err(format!(
                "stale compaction plan (base head {}, next segment {}, checkpoint {}) \
                 does not match store (head {}, next segment {}, checkpoint {})",
                plan.base_head,
                plan.base_next_segment_id,
                plan.base_checkpoint_txn,
                self.manifest.system_ts,
                self.manifest.next_segment_id,
                self.manifest.checkpoint_txn
            )));
        }

        let mut next_manifest = self.manifest.clone();
        let head = plan.base_head;
        let system_ts = plan.system_ts;
        for (entry, bytes) in &plan.segments {
            write_atomic(&self.segment_path(entry), bytes)?;
            next_manifest.segments.push(entry.clone());
        }
        next_manifest.next_segment_id =
            plan.base_next_segment_id + plan.segments.len() as u64;

        let mut retired_segments = 0usize;
        for e in &mut next_manifest.segments {
            if e.superseded_at.is_none() && e.system_ts <= head {
                e.superseded_at = Some(system_ts);
                retired_segments += 1;
            }
        }
        let mut retired_deltas = 0usize;
        for d in &mut next_manifest.deltas {
            if d.superseded_at.is_none() {
                d.superseded_at = Some(system_ts);
                d.state = MaterializationState::Materialized;
                retired_deltas += 1;
            }
        }
        next_manifest.system_ts = system_ts;
        let manifest_bytes = serde_json::to_vec_pretty(&next_manifest)
            .map_err(|e| err(format!("cannot serialize manifest: {e}")))?;
        write_atomic(&self.manifest_path(), &manifest_bytes)?;
        self.manifest = next_manifest;

        Ok(MaterializeReport {
            system_ts,
            new_segments: plan.segments.len(),
            retired_segments,
            retired_deltas,
            rows: plan.rows,
        })
    }

    /// Compact at the next strictly newer snapshot point.
    pub fn materialize(&mut self) -> Result<MaterializeReport> {
        let t = self.manifest.system_ts + 1;
        self.materialize_at(t)
    }

    /// Read-only projection of the manifest commit log. No event bus, no
    /// delivery state — every caller sees exactly the published deltas.
    pub fn invalidations_since(&self, since_commit: u64) -> Result<Vec<InvalidationRow>> {
        let mut out = Vec::new();
        for de in &self.manifest.deltas {
            let d = self.decoded_delta(de.id)?;
            for inv in &d.invalidations {
                if inv.commit_id < since_commit {
                    continue;
                }
                out.push(invalidation_row(inv, de));
            }
        }
        out.sort_by_key(|r| (r.commit_id, r.delta_id));
        Ok(out)
    }

    /// Materialise `as_of_system(T)` as **one contiguous CSR**.
    ///
    /// This is the "LSM write, CSR read" boundary: the retract overlay is
    /// folded here, once per version query — never inside `neighbor_iter`.
    /// The returned CSR is owned; for zero-copy traversal of a single baseline
    /// use `open_entry`.
    ///
    /// Materialisation is columnar: each live segment's CSR is already sorted
    /// by the documented key `(src, valid_from, valid_to, dst)`, so the whole
    /// graph is a k-way merge between those sorted runs, written straight into
    /// the five output columns. It does **not** go through
    /// [`SegmentStore::live_rows_at`]: that path builds one 72 B `LiveRow` per
    /// edge and then re-sorts them, which measured 1.5 s at 1e7 edges against
    /// 489 ms for the sweep alone. See `docs/cuda-acceleration-plan.md` §9.4.
    pub fn graph_at(&self, t: i64) -> Result<TemporalCSR> {
        let _scope = ReadScope::new(&self.readers);
        self.check_history(t)?;
        live_csr_from(self.view(), t)
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

    // -- M3: named snapshot tags -------------------------------------------

    /// All named snapshot references, in name order.
    pub fn tags(&self) -> &BTreeMap<String, i64> {
        &self.manifest.tags
    }

    /// Resolve a tag to its published system-time point. A tag below the GC
    /// retention floor is rejected rather than silently returning a graph
    /// whose files no longer exist.
    pub fn resolve_tag(&self, name: &str) -> Result<i64> {
        let t = *self
            .manifest
            .tags
            .get(name)
            .ok_or_else(|| err(format!("unknown tag `{name}`")))?;
        self.check_history(t)?;
        Ok(t)
    }

    /// Create (or move) a named reference to an already published point.
    ///
    /// One atomic manifest write; `system_ts`, `next_segment_id` and
    /// `checkpoint_txn` are untouched, so an in-flight compaction plan stays
    /// valid. Tags are references, not branches: the history stays linear.
    pub fn create_tag(&mut self, name: &str, system_ts: i64) -> Result<()> {
        validate_tag_name(name)?;
        if system_ts > self.manifest.system_ts {
            return Err(err(format!(
                "cannot tag system_ts {system_ts}: head is {}",
                self.manifest.system_ts
            )));
        }
        self.check_history(system_ts)?;
        let mut next = self.manifest.clone();
        next.tags.insert(name.to_string(), system_ts);
        self.save_manifest_as(&next)?;
        self.manifest = next;
        Ok(())
    }

    /// Remove a tag. Returns whether it existed.
    pub fn drop_tag(&mut self, name: &str) -> Result<bool> {
        validate_tag_name(name)?;
        if !self.manifest.tags.contains_key(name) {
            return Ok(false);
        }
        let mut next = self.manifest.clone();
        next.tags.remove(name);
        self.save_manifest_as(&next)?;
        self.manifest = next;
        Ok(true)
    }

    // -- M2: persisted observation versions -------------------------------

    pub fn observations_dir(&self) -> PathBuf {
        self.root.join("observations")
    }

    pub fn observations(&self) -> &[ObservationEntry] {
        &self.manifest.observations
    }

    /// Newest version of `table` known at `system_t`, decoded from its IPC file.
    pub fn observation_batches(&self, table: &str, system_t: i64) -> Result<Vec<RecordBatch>> {
        self.check_history(system_t)?;
        match version_at(&self.manifest.observations, table, system_t) {
            Some(e) => read_observation(&self.root, e),
            None => Err(err(format!(
                "no observation version of `{table}` at or before {system_t}"
            ))),
        }
    }

    /// Publish one system-time version of an observation table.
    ///
    /// The file is written first, but the version only becomes visible when
    /// the manifest publish succeeds (a crash leaves an unreferenced orphan
    /// file, reclaimed by the next GC). No WAL frame: this is an append-only
    /// immutable version, not a mutation of published rows.
    pub fn publish_observation(
        &mut self,
        table: &str,
        system_ts: i64,
        batches: Vec<RecordBatch>,
    ) -> Result<ObservationEntry> {
        if table.trim().is_empty() {
            return Err(err("observation table name is empty"));
        }
        let (bytes, _schema, rows) = encode_observation(&batches)?;
        let checksum = to_hex(blake3::hash(&bytes).as_bytes());
        let mut next_manifest = self.manifest.clone();
        let id = next_manifest.next_observation_id.max(1);
        next_manifest.next_observation_id = id + 1;
        let commit_id = self.next_txn;
        let file = format!("obs_{id:08}_c{commit_id:012}.gtvobs");
        fs::create_dir_all(self.observations_dir())?;
        write_atomic(&self.observations_dir().join(&file), &bytes)?;

        let entry = ObservationEntry {
            id,
            table: table.to_string(),
            system_ts,
            file,
            row_count: rows,
            data_checksum: checksum,
            commit_id,
        };
        next_manifest.observations.push(entry.clone());
        next_manifest.next_commit_id = commit_id;
        self.save_manifest_as(&next_manifest)?;
        self.manifest = next_manifest;
        self.next_txn = commit_id + 1;
        Ok(entry)
    }

    /// Rehydrate a session `BitemporalStore` from the published versions, so
    /// `as_of('cpi', ...)` keeps working across a process restart. `record()`
    /// is idempotent per `system_ts`, so this is safe to call repeatedly.
    pub fn load_observations_into(&self, target: &mut BitemporalStore) -> Result<usize> {
        let mut loaded = 0usize;
        for e in &self.manifest.observations {
            let batches = read_observation(&self.root, e)?;
            target.record(&e.table, e.system_ts, batches)?;
            loaded += 1;
        }
        Ok(loaded)
    }

    // -- M2: L2 aggregates -------------------------------------------------

    /// Folded, de-duplicated snapshot projection for the SQL aggregate layer.
    /// `register_segment_store()` calls this once; a later correction requires
    /// an explicit re-registration, exactly like the L1 catalog projection.
    pub fn aggregate_view(&self, system_t: i64) -> Result<AggregateView> {
        let _scope = ReadScope::new(&self.readers);
        self.check_history(system_t)?;
        let rows = live_rows_from(self.view(), system_t)?;
        let segments_total = self.entries_at_system(system_t).len();
        Ok(AggregateView::from_rows(system_t, &rows, segments_total))
    }

    /// Distinct logical edges active at `(business_t, system_t)`, plus the L2
    /// segment-level pruning decision.
    ///
    /// Segment classification (`Skip` / `FullHit` / `Scan`) is O(1) per segment
    /// and never opens a segment file. The *count* still goes through the single
    /// retract fold, because a FullHit segment may contain retracted rows and
    /// interval-split fragments must be de-duplicated — correctness wins over
    /// an O(1) sum that would double count.
    pub fn active_edge_count_at(
        &self,
        system_t: i64,
        business_t: i64,
    ) -> Result<(u64, PruneSummary)> {
        let _scope = ReadScope::new(&self.readers);
        self.check_history(system_t)?;
        let mut summary = PruneSummary::default();
        for e in self.entries_at_system(system_t) {
            match e.classify(business_t, business_t.saturating_add(1)) {
                PruneOutcome::Skip => summary.skipped += 1,
                PruneOutcome::FullHit => {
                    summary.full_hit += 1;
                    summary.selected += 1;
                }
                PruneOutcome::Scan => {
                    summary.scan += 1;
                    summary.selected += 1;
                }
            }
        }
        let distinct: HashSet<LogicalEdge> = live_rows_from(self.view(), system_t)?
            .iter()
            .filter(|r| r.edge.valid_from <= business_t && business_t < r.edge.valid_to)
            .map(LogicalEdge::of)
            .collect();
        Ok((distinct.len() as u64, summary))
    }

    // -- M2: delayed GC ----------------------------------------------------

    /// Reclaim files whose only consumers are snapshots below the retention
    /// floor. Default policy is a no-op dry run; see [`GcPolicy`].
    pub fn gc(&mut self, policy: GcPolicy) -> Result<GcReport> {
        let active = reader_count(&self.readers);
        if active != 0 {
            return Err(err(format!(
                "cannot run GC while {active} reader(s) hold a store scope"
            )));
        }
        let Some(retain) = policy.retain_from else {
            return Ok(GcReport::no_op(active));
        };

        // Snapshot points, used to protect the newest N published points even
        // when they fall below the requested floor.
        let mut points: Vec<i64> = vec![self.manifest.system_ts];
        points.extend(self.manifest.segments.iter().map(|e| e.system_ts));
        points.extend(self.manifest.segments.iter().filter_map(|e| e.superseded_at));
        points.extend(self.manifest.deltas.iter().map(|d| d.system_ts));
        points.extend(self.manifest.deltas.iter().filter_map(|d| d.superseded_at));
        points.extend(self.manifest.node_versions.iter().map(|v| v.system_ts));
        points.extend(self.manifest.observations.iter().map(|v| v.system_ts));
        // M3: a tag is an explicit reference to a published snapshot, so its
        // point participates in `keep_last_n` exactly like a segment point.
        points.extend(self.manifest.tags.values().copied());
        points.sort_unstable();
        points.dedup();
        let keep_floor = if policy.keep_last_n == 0 {
            i64::MIN
        } else {
            points[points.len().saturating_sub(policy.keep_last_n)]
        };
        let mut cutoff = retain.min(keep_floor);
        // M3: never collect the history a tag still pins. Lowering the cutoff
        // is conservative — it can only keep more files, never fewer.
        let tag_floor = self.manifest.tags.values().copied().min();
        let pinned = tag_floor.is_some_and(|t| t < cutoff);
        if let Some(t) = tag_floor {
            cutoff = cutoff.min(t);
        }
        if cutoff <= self.manifest.gc_floor.unwrap_or(i64::MIN) {
            return Ok(GcReport::no_op(active));
        }

        // `node_lifetime_at(t >= cutoff)` needs the newest version at or below
        // the floor; observation tables need one per table for the same reason.
        let keep_node_ts = self
            .manifest
            .node_versions
            .iter()
            .filter(|v| v.system_ts <= cutoff)
            .map(|v| v.system_ts)
            .max();
        let mut keep_obs: HashMap<String, i64> = HashMap::new();
        for v in &self.manifest.observations {
            if v.system_ts <= cutoff {
                let slot = keep_obs.entry(v.table.clone()).or_insert(i64::MIN);
                *slot = (*slot).max(v.system_ts);
            }
        }

        let mut report = GcReport {
            cutoff,
            dry_run: policy.dry_run,
            active_readers: active,
            pinned_by_tags: pinned,
            ..GcReport::default()
        };
        // (directory, file) pairs staged for unlink *after* the manifest publish.
        let mut dropped: Vec<(&'static str, String)> = Vec::new();
        let mut next_manifest = self.manifest.clone();

        next_manifest.segments.retain(|e| {
            if e.superseded_at.map_or(false, |m| m <= cutoff) {
                dropped.push(("segments", e.file.clone()));
                report.segments_removed += 1;
                false
            } else {
                true
            }
        });
        next_manifest.deltas.retain(|d| {
            if d.superseded_at.map_or(false, |m| m <= cutoff) {
                dropped.push(("deltas", d.file.clone()));
                report.deltas_removed += 1;
                false
            } else {
                true
            }
        });
        next_manifest.node_versions.retain(|v| {
            if v.system_ts <= cutoff && keep_node_ts.map_or(true, |k| v.system_ts < k) {
                dropped.push(("nodes", v.file.clone()));
                report.node_versions_removed += 1;
                false
            } else {
                true
            }
        });
        next_manifest.observations.retain(|v| {
            let keep = keep_obs.get(&v.table).copied();
            if v.system_ts <= cutoff && keep.map_or(true, |k| v.system_ts < k) {
                dropped.push(("observations", v.file.clone()));
                report.observations_removed += 1;
                false
            } else {
                true
            }
        });

        if policy.dry_run {
            // Report candidates only: no manifest change, no unlink.
            return Ok(report);
        }

        // Publish-before-unlink. A crash here leaves orphan files, never a
        // manifest pointing at a deleted file.
        next_manifest.gc_floor = Some(cutoff);
        self.save_manifest_as(&next_manifest)?;
        self.manifest = next_manifest;
        report.manifest_rewritten = true;

        for (dir, file) in &dropped {
            report.bytes_reclaimed += remove_file(&self.root.join(dir).join(file))?;
        }

        // Unreferenced files from an interrupted flush / publish / compaction.
        let mut referenced: HashSet<String> = HashSet::new();
        referenced.extend(self.manifest.segments.iter().map(|e| e.file.clone()));
        referenced.extend(self.manifest.deltas.iter().map(|e| e.file.clone()));
        referenced.extend(self.manifest.node_versions.iter().map(|e| e.file.clone()));
        referenced.extend(self.manifest.observations.iter().map(|e| e.file.clone()));
        for (dir, ext) in [
            ("segments", ".seg"),
            ("deltas", ".gtvd"),
            ("nodes", ".bin"),
            ("observations", ".gtvobs"),
        ] {
            let path = self.root.join(dir);
            if !path.exists() {
                continue;
            }
            for item in fs::read_dir(&path)? {
                let item = item?;
                if !item.path().is_file() {
                    continue;
                }
                let name = item.file_name().to_string_lossy().into_owned();
                if !name.ends_with(ext) || referenced.contains(&name) {
                    continue;
                }
                report.bytes_reclaimed += remove_file(&item.path())?;
                report.orphan_files_removed += 1;
            }
        }
        Ok(report)
    }
}

fn open_entry_from(root: &Path, entry: &SegmentEntry) -> Result<Segment> {
    let seg = Segment::open(root.join("segments").join(&entry.file))?;
    let got = to_hex(&seg.header().data_checksum);
    if got != entry.data_checksum {
        return Err(err(format!(
            "segment {} checksum mismatch: manifest {}, file {got}",
            entry.id, entry.data_checksum
        )));
    }
    Ok(seg)
}

fn decoded_delta_from<'a>(
    deltas: &'a HashMap<u64, SnapshotDelta>,
    id: u64,
) -> Result<&'a SnapshotDelta> {
    deltas
        .get(&id)
        .ok_or_else(|| err(format!("delta {id} is in the manifest but not decoded")))
}

/// The **single** retract fold in the codebase.
///
/// Both the live store and a background compaction worker call this with the
/// same inputs, so "visible at t" can never diverge between an overlay query
/// and its materialized form. The source sweep is narrowed to the tight L2
/// `src_id_min/max` box; the absolute row id starts at `offsets[lo]` so the
/// positional `(segment_id, row)` namespace stays byte-for-byte identical to
/// the un-narrowed scan.
fn live_rows_from(v: LiveRowsView<'_>, t: i64) -> Result<Vec<LiveRow>> {
    let live_deltas: Vec<&DeltaEntry> = v
        .manifest
        .deltas
        .iter()
        .filter(|d| d.system_ts <= t && d.superseded_at.map_or(true, |m| m > t))
        .collect();
    let retracts = RetractSet::merge(
        &live_deltas
            .iter()
            .map(|de| decoded_delta_from(v.deltas, de.id).map(|d| d.retracts()))
            .collect::<Result<Vec<_>>>()?,
    );
    let mut out = Vec::new();

    for entry in v
        .manifest
        .segments
        .iter()
        .filter(|e| e.system_ts <= t && e.superseded_at.map_or(true, |m| m > t))
    {
        let seg = open_entry_from(v.root, entry)?;
        let c = seg.csr().columns();
        if c.node_count != v.node_count {
            return Err(err(format!(
                "segment {} has node_count {} but store expects {}",
                entry.id, c.node_count, v.node_count
            )));
        }
        if c.dst.len() > u32::MAX as usize {
            return Err(err(format!(
                "segment {} has {} rows; positional retract uses u32 row ids",
                entry.id,
                c.dst.len()
            )));
        }
        if c.node_count == 0 {
            continue;
        }
        let last = (c.node_count - 1) as u64;
        let lo = entry.src_id_min.unwrap_or(0).min(last) as usize;
        let hi = entry.src_id_max.unwrap_or(u64::MAX).min(last) as usize + 1;
        let mut row = c.offsets[lo] as u32;
        for s in lo..hi {
            let (a, b) = (c.offsets[s] as usize, c.offsets[s + 1] as usize);
            for e in a..b {
                let pos = RowPos::new(entry.id, row);
                row += 1;
                if retracts.contains(pos) {
                    continue;
                }
                out.push(LiveRow {
                    pos,
                    edge: EdgeRecord {
                        src: s as u64,
                        dst: c.dst[e],
                        valid_from: c.valid_from[e],
                        valid_to: c.valid_to[e],
                        edge_type: c.edge_type[e],
                        system_ts: entry.system_ts,
                    },
                    table_id: entry.table_id,
                    partition_id: entry.partition_id,
                });
            }
        }
    }

    for de in &live_deltas {
        let d = decoded_delta_from(v.deltas, de.id)?;
        for (i, &edge) in d.puts.iter().enumerate() {
            let pos = RowPos::new(d.put_segment_id, i as u32);
            if retracts.contains(pos) {
                continue;
            }
            out.push(LiveRow {
                pos,
                edge,
                table_id: 0,
                partition_id: 0,
            });
        }
    }

    // Unflushed WAL rows live in a namespace that is never a retract target.
    for (i, r) in v.mem_records.iter().enumerate() {
        if r.edge.system_ts > t {
            continue;
        }
        out.push(LiveRow {
            pos: RowPos::new(MEM_ROW_NAMESPACE, i as u32),
            edge: r.edge,
            table_id: r.table_id,
            partition_id: r.partition_id,
        });
    }
    Ok(out)
}

// -- columnar materialisation ----------------------------------------------

/// `offsets`, filled in as the merge runs.
///
/// The output is sorted by `src`, so a source is finished the moment the merge
/// moves past it and `offsets[s + 1]` is known right then. Counting degrees
/// first and prefix-summing afterwards would cost a second pass over
/// `node_count` plus an array that nothing else needs.
struct Offsets {
    counts: Vec<u32>,
    /// Index of the source currently being filled.
    open: usize,
}

impl Offsets {
    fn new(node_count: usize) -> Self {
        Self {
            counts: vec![0u32; node_count + 1],
            open: 0,
        }
    }

    /// Close every source below `src`. `at` is the number of rows written so far.
    #[inline]
    fn close_below(&mut self, src: usize, at: usize) {
        for k in self.open + 1..=src {
            self.counts[k] = at as u32;
        }
        self.open = src;
    }

    /// Close the tail once the runs are drained.
    fn finish(mut self, at: usize) -> Vec<u32> {
        let last = self.counts.len() - 1;
        for k in self.open + 1..=last {
            self.counts[k] = at as u32;
        }
        self.counts
    }
}

/// The next edge a run offers, already retract-filtered.
#[derive(Debug, Clone, Copy)]
struct Head {
    src: u64,
    dst: u64,
    valid_from: i64,
    valid_to: i64,
    edge_type: u16,
}

impl Head {
    /// The CSR's documented sort key: `(src, valid_from, valid_to, dst)`.
    ///
    /// `edge_type` is deliberately absent. Two versions of one edge can share
    /// every other field, and the index only ever promised this order.
    #[inline]
    fn key(&self) -> (u64, i64, i64, u64) {
        (self.src, self.valid_from, self.valid_to, self.dst)
    }
}

/// One sorted run feeding [`live_csr_from`].
///
/// A run is either a published segment — whose columns are already in
/// [`Head::key`] order, so it is walked in place with no copy at all — or a row
/// list (a delta's puts, the unflushed WAL) that had to be sorted first. Both
/// yield [`Head`]s at the same key, which is the point: the merge never learns
/// which kind it is looking at.
enum Cursor<'a> {
    /// A segment's own columns, narrowed to the `src` box its L2 metadata
    /// claims and walked in row-id order so retracts stay positional.
    Columns {
        cols: &'a CsrColumns<'a>,
        retracts: &'a RetractSet,
        /// Segment id — the high half of a [`RowPos`].
        entry_id: u64,
        /// Exclusive bound on `src`.
        hi: usize,
        src: usize,
        /// Index of the next edge within the segment.
        edge: usize,
        /// Row id of `edge`, kept in lockstep with it: a retract position is
        /// `(segment, row)`, and deriving it per edge would mean a search.
        row: u32,
    },
    /// A materialised row list.
    Rows {
        rows: &'a [EdgeRecord],
        retracts: &'a RetractSet,
        /// `Some(put_segment_id)` when the positions are retractable, `None`
        /// for the WAL namespace, which is never a retract target.
        namespace: Option<u64>,
        next: usize,
    },
}

impl Cursor<'_> {
    fn next(&mut self) -> Option<Head> {
        match self {
            Cursor::Columns {
                cols,
                retracts,
                entry_id,
                hi,
                src,
                edge,
                row,
            } => loop {
                // A source with no edges left has `edge == offsets[src]`, so
                // `offsets[src + 1]` is where the next non-empty one begins.
                while *src < *hi && *edge == cols.offsets[*src + 1] as usize {
                    *src += 1;
                }
                if *src >= *hi {
                    return None;
                }
                let head = Head {
                    src: *src as u64,
                    dst: cols.dst[*edge],
                    valid_from: cols.valid_from[*edge],
                    valid_to: cols.valid_to[*edge],
                    edge_type: cols.edge_type[*edge],
                };
                let pos = RowPos::new(*entry_id, *row);
                *row += 1;
                *edge += 1;
                if !retracts.contains(pos) {
                    return Some(head);
                }
            },
            Cursor::Rows {
                rows,
                retracts,
                namespace,
                next,
            } => loop {
                let i = *next;
                if i >= rows.len() {
                    return None;
                }
                *next = i + 1;
                if namespace.is_some_and(|ns| retracts.contains(RowPos::new(ns, i as u32))) {
                    continue;
                }
                let e = rows[i];
                return Some(Head {
                    src: e.src,
                    dst: e.dst,
                    valid_from: e.valid_from,
                    valid_to: e.valid_to,
                    edge_type: e.edge_type,
                });
            },
        }
    }
}

impl Cursor<'_> {
    /// This run's own CSR, when the run *is* the whole answer.
    ///
    /// True for a single segment that covers every source node and that no
    /// retract touches: its columns already are the visible graph, so copying
    /// them is the only work left, and building it edge by edge would reproduce
    /// 257 MiB of bytes it already has. `Ok(None)` means "not this shape", never
    /// "failed" — the structural checks still run inside `from_columns` so the
    /// shortcut cannot skip them.
    fn verbatim(&self, node_count: usize) -> Result<Option<TemporalCSR>> {
        let Cursor::Columns {
            cols,
            retracts,
            hi,
            src,
            edge,
            ..
        } = self
        else {
            return Ok(None);
        };
        if !retracts.is_empty() || *src != 0 || *edge != 0 || *hi != node_count {
            return Ok(None);
        }
        if cols.node_count != node_count || cols.offsets.len() != node_count + 1 {
            return Ok(None);
        }
        TemporalCSR::from_columns(
            cols.node_count,
            cols.offsets.to_vec().into(),
            cols.dst.to_vec().into(),
            cols.valid_from.to_vec().into(),
            cols.valid_to.to_vec().into(),
            cols.edge_type.to_vec().into(),
            cols.max_valid_from,
            cols.min_valid_to,
            cols.chunk_size,
            cols.chunk_index.to_vec().into(),
            cols.chunk_max_valid_to.to_vec().into(),
        )
        .map(Some)
        .map_err(|e| err(format!("invalid delta CSR: {e}")))
    }
}

/// The five CSR columns, plus the two scalars derived while writing them.
///
/// A sink rather than five loose `Vec`s so the merge and the single-run fast
/// path below write through exactly one `push`: if a column is ever added or
/// reordered, it is one place, and the fast path cannot silently disagree with
/// the general one.
struct ColumnSink {
    offsets: Offsets,
    dst: Vec<u64>,
    valid_from: Vec<i64>,
    valid_to: Vec<i64>,
    edge_type: Vec<u16>,
    node_count: usize,
    max_valid_from: i64,
    min_valid_to: i64,
}

impl ColumnSink {
    fn new(node_count: usize, capacity: usize) -> Self {
        Self {
            offsets: Offsets::new(node_count),
            dst: Vec::with_capacity(capacity),
            valid_from: Vec::with_capacity(capacity),
            valid_to: Vec::with_capacity(capacity),
            edge_type: Vec::with_capacity(capacity),
            node_count,
            // The same identity values `from_edge_columns` uses, so an empty
            // graph gets the same bounding box from both paths.
            max_valid_from: i64::MIN,
            min_valid_to: i64::MAX,
        }
    }

    #[inline]
    fn push(&mut self, h: Head) -> Result<()> {
        let src = h.src as usize;
        if src >= self.node_count {
            // The row path reports this out of `from_edge_columns`; the column
            // path has to say it itself, because nothing downstream re-sorts.
            return Err(err(format!(
                "invalid delta CSR: {}",
                GtvError::NodeOutOfRange(h.src)
            )));
        }
        self.offsets.close_below(src, self.dst.len());
        self.dst.push(h.dst);
        self.valid_from.push(h.valid_from);
        self.valid_to.push(h.valid_to);
        self.edge_type.push(h.edge_type);
        self.max_valid_from = self.max_valid_from.max(h.valid_from);
        self.min_valid_to = self.min_valid_to.min(h.valid_to);
        Ok(())
    }

    fn finish(self) -> Result<TemporalCSR> {
        let offsets = self.offsets.finish(self.dst.len());

        // Zone maps, identical to `from_edge_columns`: one `max(valid_to)` per
        // CHUNK_SIZE edges of each source run.
        let chunk_size = CHUNK_SIZE;
        let mut chunk_index = Vec::with_capacity(self.node_count + 1);
        let mut chunk_max_valid_to = Vec::new();
        chunk_index.push(0u32);
        for s in 0..self.node_count {
            let mut c = offsets[s] as usize;
            let end = offsets[s + 1] as usize;
            while c < end {
                let ce = (c + chunk_size).min(end);
                chunk_max_valid_to.push(
                    self.valid_to[c..ce]
                        .iter()
                        .copied()
                        .max()
                        .unwrap_or(i64::MIN),
                );
                c = ce;
            }
            chunk_index.push(chunk_max_valid_to.len() as u32);
        }

        TemporalCSR::from_columns(
            self.node_count,
            offsets.into(),
            self.dst.into(),
            self.valid_from.into(),
            self.valid_to.into(),
            self.edge_type.into(),
            self.max_valid_from,
            self.min_valid_to,
            chunk_size,
            chunk_index.into(),
            chunk_max_valid_to.into(),
        )
        .map_err(|e| err(format!("invalid delta CSR: {e}")))
    }
}

/// The column form of [`live_rows_from`].
///
/// Both functions answer "which edges are visible at `t`" and must agree
/// exactly, which is why they sit next to each other and share the fold's
/// shape: the same live-set predicate, the same retract namespaces, the same
/// [`RowPos`] numbering. Only the shape of the answer differs — rows for the
/// Arrow surface, columns for the CSR — and that difference is a second of CPU
/// per query at 1e7 edges.
///
/// It is a merge because a published segment's CSR is already sorted by
/// `(src, valid_from, valid_to, dst)`: the order was paid for at flush time, and
/// collecting the edges into 72 B rows only to sort them again pays for it
/// twice. Each segment is a sorted run, the answer is their k-way merge written
/// straight into the output columns, and peak host memory is the output itself.
fn live_csr_from(v: LiveRowsView<'_>, t: i64) -> Result<TemporalCSR> {
    let live_deltas: Vec<&DeltaEntry> = v
        .manifest
        .deltas
        .iter()
        .filter(|d| d.system_ts <= t && d.superseded_at.is_none_or(|m| m > t))
        .collect();
    let retracts = RetractSet::merge(
        &live_deltas
            .iter()
            .map(|de| decoded_delta_from(v.deltas, de.id).map(|d| d.retracts()))
            .collect::<Result<Vec<_>>>()?,
    );

    let entries: Vec<&SegmentEntry> = v
        .manifest
        .segments
        .iter()
        .filter(|e| e.system_ts <= t && e.superseded_at.is_none_or(|m| m > t))
        .collect();

    // Opened first, borrowed second: `CsrColumns` borrows its `Segment`, so the
    // owned segments have to outlive the column list. Nothing mutates them
    // afterwards, which is what lets the borrow survive the merge — and why the
    // columns cost one mmap each instead of one copy per edge.
    let mut segments = Vec::with_capacity(entries.len());
    for entry in &entries {
        segments.push(open_entry_from(v.root, entry)?);
    }
    let cols: Vec<CsrColumns<'_>> = segments.iter().map(|s| s.csr().columns()).collect();

    let mut cursors: Vec<Cursor<'_>> = Vec::with_capacity(entries.len() + live_deltas.len() + 1);
    // Upper bound on the output row count, summed from the sources before
    // anything is allocated. Retracts only ever shrink it.
    let mut bound = 0usize;

    for (i, entry) in entries.iter().enumerate() {
        let c = &cols[i];
        if c.node_count != v.node_count {
            return Err(err(format!(
                "segment {} has node_count {} but store expects {}",
                entry.id, c.node_count, v.node_count
            )));
        }
        if c.dst.len() > u32::MAX as usize {
            return Err(err(format!(
                "segment {} has {} rows; positional retract uses u32 row ids",
                entry.id,
                c.dst.len()
            )));
        }
        if c.node_count == 0 {
            continue;
        }
        let last = (c.node_count - 1) as u64;
        let lo = entry.src_id_min.unwrap_or(0).min(last) as usize;
        let hi = entry.src_id_max.unwrap_or(u64::MAX).min(last) as usize + 1;
        bound += c.offsets[hi] as usize - c.offsets[lo] as usize;
        cursors.push(Cursor::Columns {
            cols: c,
            retracts: &retracts,
            entry_id: entry.id,
            hi,
            src: lo,
            edge: c.offsets[lo] as usize,
            row: c.offsets[lo],
        });
    }

    // Delta puts and WAL rows arrive as row lists in no particular order, so
    // each gets one sort before joining. They are copied because the stored
    // delta and the WAL are immutable here: sorting them in place would reorder
    // records another reader holds. Both are bounded — the overlay depth by
    // `MAX_DELTA_DEPTH`, the WAL by the flush threshold.
    let mut delta_runs: Vec<(u64, Vec<EdgeRecord>)> = Vec::with_capacity(live_deltas.len());
    for de in &live_deltas {
        let d = decoded_delta_from(v.deltas, de.id)?;
        let mut puts = d.puts.clone();
        puts.sort_unstable_by_key(|e| (e.src, e.valid_from, e.valid_to, e.dst));
        bound += puts.len();
        delta_runs.push((d.put_segment_id, puts));
    }
    let mut wal: Vec<EdgeRecord> = v
        .mem_records
        .iter()
        .filter(|r| r.edge.system_ts <= t)
        .map(|r| r.edge)
        .collect();
    wal.sort_unstable_by_key(|e| (e.src, e.valid_from, e.valid_to, e.dst));
    bound += wal.len();

    if bound > u32::MAX as usize {
        return Err(err(format!(
            "{} live edges exceed the u32 address space ({} edges)",
            bound,
            u32::MAX
        )));
    }

    for (ns, puts) in &delta_runs {
        cursors.push(Cursor::Rows {
            rows: puts,
            retracts: &retracts,
            namespace: Some(*ns),
            next: 0,
        });
    }
    if !wal.is_empty() {
        cursors.push(Cursor::Rows {
            rows: &wal,
            retracts: &retracts,
            namespace: None,
            next: 0,
        });
    }

    // Fast path: exactly one run. This is the shape that matters — a compacted
    // store has one baseline segment, no live delta and an empty WAL — and then
    // there is nothing to merge and, if no retract touches it, nothing to
    // rewrite: the answer *is* the segment's own columns.
    if cursors.len() <= 1 {
        let Some(mut only) = cursors.pop() else {
            return ColumnSink::new(v.node_count, 0).finish();
        };
        if let Some(csr) = only.verbatim(v.node_count)? {
            return Ok(csr);
        }
        let mut sink = ColumnSink::new(v.node_count, bound);
        while let Some(h) = only.next() {
            sink.push(h)?;
        }
        return sink.finish();
    }

    let mut sink = ColumnSink::new(v.node_count, bound);
    let mut heads: Vec<Option<Head>> = cursors.iter_mut().map(|c| c.next()).collect();
    loop {
        // Smallest head wins. Linear over the runs on purpose: a store holds a
        // handful (compaction merges them away), so a heap would add a
        // logarithmic term per edge to save a handful of comparisons and a
        // pointer chase. Ties go to the earliest run, which keeps the output
        // deterministic for rows that share a key.
        let mut pick: Option<usize> = None;
        for (i, head) in heads.iter().enumerate() {
            let Some(h) = head else { continue };
            pick = match pick {
                None => Some(i),
                Some(j) => {
                    let held = heads[j].as_ref().expect("picked heads are set");
                    if h.key() < held.key() {
                        Some(i)
                    } else {
                        Some(j)
                    }
                }
            };
        }
        let Some(i) = pick else { break };
        sink.push(heads[i].expect("picked heads are set"))?;
        heads[i] = cursors[i].next();
    }
    sink.finish()
}

/// Phase 1 of compaction, shared by the synchronous and worker paths.
///
/// Pure CPU over a read-only view: no file is written and the manifest is not
/// advanced. A crash / cancellation here has **zero** on-disk effect, which is
/// what makes §23.8's compaction crash points safe.
fn prepare_compaction(v: LiveRowsView<'_>, system_ts: i64) -> Result<CompactionPlan> {
    if system_ts <= v.manifest.system_ts {
        return Err(err(format!(
            "compaction system_ts {system_ts} does not advance past {}",
            v.manifest.system_ts
        )));
    }
    let head = v.manifest.system_ts;
    let rows = live_rows_from(v, head)?;
    let mut groups: BTreeMap<(u32, u32), Vec<EdgeRecord>> = BTreeMap::new();
    for r in &rows {
        groups
            .entry((r.table_id, r.partition_id))
            .or_default()
            .push(r.edge);
    }

    let checkpoint_txn = v.manifest.checkpoint_txn;
    let mut next_id = v.manifest.next_segment_id;
    let mut segments = Vec::with_capacity(groups.len());
    for ((table_id, partition_id), edges) in &groups {
        if edges.is_empty() {
            continue;
        }
        let csr = MemSegment::to_csr(edges, v.node_count)?;
        let meta = SegmentMeta::from_csr(&csr);
        let bytes = encode_segment(&csr)?;
        let checksum = to_hex(blake3::hash(&bytes[HEADER_SIZE..]).as_bytes());
        let id = next_id;
        next_id += 1;
        let file = format!("seg_{id:08}_t{table_id}_p{partition_id}.seg");
        segments.push((
            SegmentEntry {
                id,
                file,
                table_id: *table_id,
                partition_id: *partition_id,
                txn_start: checkpoint_txn,
                txn_end: checkpoint_txn,
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
                src_id_min: Some(meta.src_id_min),
                src_id_max: Some(meta.src_id_max),
                superseded_at: None,
            },
            bytes,
        ));
    }

    Ok(CompactionPlan {
        base_head: head,
        base_next_segment_id: v.manifest.next_segment_id,
        base_checkpoint_txn: checkpoint_txn,
        system_ts,
        segments,
        rows: rows.len(),
    })
}

fn invalidation_row(inv: &InvalidationRecord, de: &DeltaEntry) -> InvalidationRow {
    InvalidationRow {
        commit_id: inv.commit_id,
        system_ts: inv.system_ts,
        object_kind: inv.object_kind.name().to_string(),
        object_id: inv.object_id,
        affected_valid_from: inv.affected_valid_from,
        affected_valid_to: inv.affected_valid_to,
        reason: inv.reason.name().to_string(),
        delta_id: de.id,
        state: match de.state {
            MaterializationState::DeltaOnly => "delta_only",
            MaterializationState::Materialized => "materialized",
        }
        .to_string(),
        file: de.file.clone(),
    }
}

/// Read the invalidation log straight from a store root (no `SegmentStore`
/// handle needed). Used by the `gtv_invalidations()` SQL surface.
/// One row of the SQL tag projection. Pure manifest read — no cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagRow {
    pub name: String,
    pub system_ts: i64,
}

/// M3: read-only projection of named snapshot references. Like
/// `read_invalidation_log`, this reads the published manifest directly so a
/// tag becomes visible exactly at the atomic manifest write.
pub fn read_tag_log(root: &Path) -> Result<Vec<TagRow>> {
    let manifest_path = root.join("manifest.json");
    if !manifest_path.exists() {
        return Ok(Vec::new());
    }
    let bytes = fs::read(&manifest_path)?;
    let m: Manifest = serde_json::from_slice(&bytes)
        .map_err(|e| err(format!("cannot parse {}: {e}", manifest_path.display())))?;
    let mut out: Vec<TagRow> = m
        .tags
        .into_iter()
        .map(|(name, system_ts)| TagRow { name, system_ts })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn validate_tag_name(name: &str) -> Result<()> {
    let valid = !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'));
    if !valid {
        return Err(err(format!(
            "invalid tag name `{name}` (1..=64 ascii alphanumeric / _ / - / .)"
        )));
    }
    Ok(())
}

pub fn read_invalidation_log(root: &Path, since_commit: u64) -> Result<Vec<InvalidationRow>> {
    let manifest_path = root.join("manifest.json");
    if !manifest_path.exists() {
        return Ok(Vec::new());
    }
    let bytes = fs::read(&manifest_path)?;
    let m: Manifest = serde_json::from_slice(&bytes)
        .map_err(|e| err(format!("cannot parse {}: {e}", manifest_path.display())))?;
    let mut out = Vec::new();
    for de in &m.deltas {
        let f = fs::read(root.join("deltas").join(&de.file)).map_err(|e| {
            err(format!("cannot read delta {} for invalidation log: {e}", de.file))
        })?;
        let d = decode_delta(&f)
            .map_err(|e| err(format!("cannot decode delta {}: {e}", de.file)))?;
        for inv in &d.invalidations {
            if inv.commit_id != de.commit_id {
                return Err(err(format!(
                    "delta {} has invalidation commit {} but manifest says {}",
                    de.id, inv.commit_id, de.commit_id
                )));
            }
            if inv.commit_id < since_commit {
                continue;
            }
            out.push(invalidation_row(inv, de));
        }
    }
    out.sort_by_key(|r| (r.commit_id, r.delta_id));
    Ok(out)
}

fn now_system_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as i64
}
