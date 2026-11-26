//! Query contract: receipts, evidence references and deterministic top-k (A1 / A3).
//!
//! # Why this module exists
//!
//! Every operator in the engine already reports truncation internally
//! ([`crate::TriState`], [`crate::Truncation`]), and every traversal can name an
//! edge by slot (`Neighbor::edge_idx`). What was missing was a value that
//! *crosses a process boundary* and survives a review six months later:
//!
//! * what was asked (`query_id`, `plan_hash`),
//! * against which state (`snapshot_id`, `as_of`) — and who enforced that state
//!   ([`PointBinding`]), because a request the engine never applied is not
//!   evidence about the time point it names,
//! * what came back (`rows`, `tri_state`),
//! * why it is partial (`incomplete`),
//! * and which physical rows back each edge (`EdgeRef`).
//!
//! [`QueryReceipt`] is that value. It is deliberately:
//!
//! * **deterministic** — no timestamps, no durations, no host names, so two
//!   runs of the same query over the same snapshot produce byte-identical
//!   receipts;
//! * **transport-neutral** — [`QueryReceipt::to_json`] for gRPC/HTTP,
//!   [`QueryReceipt::to_arrow_metadata`] for Arrow schema metadata, so the
//!   receipt can ride along with the result instead of in an out-of-band
//!   channel that clients forget to read;
//! * **honest** — [`QueryReceipt::validate`] rejects a receipt that claims
//!   `Found`/`NoMatch` while carrying an incomplete reason, rejects
//!   `Incomplete` without one, and rejects a conclusive result at a time point
//!   that was only declared and never enforced. Absence of rows must never be
//!   reported as proof of absence (the engine's R2 rule).
//!
//! # Evidence, not explanations
//!
//! [`PathEvidence`] names nodes and edges; [`EdgeRef`] resolves an edge to
//! `(segment_id, row_in_segment)` so a reviewer can re-read the exact stored
//! row instead of trusting a rendered sentence. Nothing here knows what a
//! "case", "alert" or "customer" is — that vocabulary belongs to the extension
//! repository (R1).
//!
//! # Determinism
//!
//! [`topk_dedup`] exists because parallel matchers emit the same match from
//! several threads in a schedule-dependent order. Sorting by a canonical
//! [`DedupKey`] before truncation makes the output independent of thread count
//! and of input order — the property the M2 acceptance criterion asks for.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{GtvError, Result};
use crate::traversal::{TraversalStats, TruncReason, Truncation, TriState};

/// Wire protocol version this build speaks.
///
/// Bump on any change that an older peer would misinterpret (new required
/// field, changed semantics of an existing field). Additive optional fields do
/// **not** require a bump.
pub const PROTOCOL_VERSION: u32 = 1;

/// Oldest protocol version this build still accepts.
///
/// Kept equal to [`PROTOCOL_VERSION`] today: there is exactly one released
/// version. When v2 lands, this stays at 1 for one support window so mixed
/// client/server fleets keep working during rollout.
pub const MIN_SUPPORTED_PROTOCOL: u32 = 1;

/// Version of the crate that produced a receipt.
///
/// Recorded per receipt so a reviewer can tie a result to the build that
/// computed it without consulting deployment metadata.
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Metadata key holding the individually-rendered receipt fields.
pub const RECEIPT_METADATA_PREFIX: &str = "gtv.receipt.";

/// Metadata key holding the full JSON receipt (round-trip source of truth).
pub const RECEIPT_METADATA_JSON: &str = "gtv.receipt.json";

/// `true` when `version` can be served by this build.
pub fn is_supported_version(version: u32) -> bool {
    (MIN_SUPPORTED_PROTOCOL..=PROTOCOL_VERSION).contains(&version)
}

/// Pick the version to speak with a peer that offered `theirs`.
///
/// Returns `None` when the ranges do not overlap, which callers must surface as
/// a hard error rather than silently downgrading — a mismatch means the client
/// will misread the result.
pub fn negotiate_version(theirs: u32) -> Option<u32> {
    if is_supported_version(theirs) {
        Some(theirs)
    } else if theirs > PROTOCOL_VERSION {
        // Peer is newer; we can serve at our ceiling only if it accepts it.
        Some(PROTOCOL_VERSION)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Edge / path evidence
// ---------------------------------------------------------------------------

/// Physical identity of one stored edge.
///
/// `edges_scanned` counts work; an `EdgeRef` proves *which* edge was matched.
/// `segment_id` identifies the immutable edge segment, `row_in_segment` the row
/// inside it, so `(segment_id, row_in_segment)` addresses a row that never
/// moves (segments are immutable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EdgeRef {
    pub segment_id: u64,
    pub row_in_segment: u32,
}

impl EdgeRef {
    /// Sentinel for a slot with no stable physical address (for example an
    /// unflushed WAL row folded into a materialised CSR). It can never be a
    /// real reference: real segment ids start at 0 and rows are `< u32::MAX`.
    pub const UNRESOLVED: Self = Self {
        segment_id: u64::MAX,
        row_in_segment: u32::MAX,
    };

    pub const fn new(segment_id: u64, row_in_segment: u32) -> Self {
        Self {
            segment_id,
            row_in_segment,
        }
    }

    /// Whether this reference names a real physical row.
    #[inline]
    pub const fn is_resolved(self) -> bool {
        self.segment_id != u64::MAX && self.row_in_segment != u32::MAX
    }

    /// Resolve a CSR edge slot to a segment-local reference.
    ///
    /// `segment_id` is the segment the slot belongs to; `slot` is the
    /// `Neighbor::edge_idx` value. Returns `None` for the sentinel
    /// `u32::MAX`, which means "identity unknown" (pull-direction expansion
    /// cannot supply a forward edge index).
    pub fn from_slot(segment_id: u64, slot: u32) -> Option<Self> {
        (slot != u32::MAX).then_some(Self {
            segment_id,
            row_in_segment: slot,
        })
    }
}

impl fmt::Display for EdgeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.segment_id, self.row_in_segment)
    }
}

impl FromStr for EdgeRef {
    type Err = GtvError;

    fn from_str(s: &str) -> Result<Self> {
        let (seg, row) = s.split_once(':').ok_or_else(|| {
            GtvError::InvalidArgument(format!("edge ref `{s}` is not `segment:row`"))
        })?;
        let segment_id = seg
            .parse::<u64>()
            .map_err(|_| GtvError::InvalidArgument(format!("bad segment id in `{s}`")))?;
        let row_in_segment = row
            .parse::<u32>()
            .map_err(|_| GtvError::InvalidArgument(format!("bad row in `{s}`")))?;
        Ok(Self {
            segment_id,
            row_in_segment,
        })
    }
}

/// One matched path: the nodes bound by the pattern, the edges between them and
/// each edge's half-open validity interval.
///
/// `nodes.len() == edges.len() + 1 == valid_from.len() == valid_to.len()`.
/// Intervals are half-open `[valid_from, valid_to)` to match the CSR's
/// convention; `valid_to == i64::MAX` marks an open-ended edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathEvidence {
    pub nodes: Vec<u64>,
    pub edges: Vec<EdgeRef>,
    pub valid_from: Vec<i64>,
    pub valid_to: Vec<i64>,
}

impl PathEvidence {
    /// Build and validate a path.
    pub fn new(
        nodes: Vec<u64>,
        edges: Vec<EdgeRef>,
        valid_from: Vec<i64>,
        valid_to: Vec<i64>,
    ) -> Result<Self> {
        let evidence = Self {
            nodes,
            edges,
            valid_from,
            valid_to,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    /// A single node with no edges (a "match" of a unary pattern).
    pub fn single(node: u64) -> Self {
        Self {
            nodes: vec![node],
            edges: Vec::new(),
            valid_from: Vec::new(),
            valid_to: Vec::new(),
        }
    }

    /// `true` when the path holds an open-ended edge (`valid_to == i64::MAX`).
    pub fn is_open_ended(&self) -> bool {
        self.valid_to.iter().any(|t| *t == i64::MAX)
    }

    /// Earliest start and latest end across the path's edges.
    pub fn span(&self) -> Option<(i64, i64)> {
        if self.valid_from.is_empty() {
            return None;
        }
        let from = *self.valid_from.iter().min()?;
        let to = *self.valid_to.iter().max()?;
        Some((from, to))
    }

    /// Walk the path and check both structural and temporal consistency.
    pub fn validate(&self) -> Result<()> {
        let n = self.nodes.len();
        if n == 0 {
            return Err(GtvError::InvalidArgument("path has no nodes".into()));
        }
        if self.edges.len() + 1 != n {
            return Err(GtvError::InvalidArgument(format!(
                "path with {} nodes must have {} edges, got {}",
                n,
                n - 1,
                self.edges.len()
            )));
        }
        if self.valid_from.len() != self.edges.len() || self.valid_to.len() != self.edges.len() {
            return Err(GtvError::InvalidArgument(
                "path validity arrays must match edge count".into(),
            ));
        }
        for (i, (from, to)) in self.valid_from.iter().zip(&self.valid_to).enumerate() {
            if from > to {
                return Err(GtvError::InvalidArgument(format!(
                    "edge {i} has an inverted interval [{from}, {to})"
                )));
            }
        }
        Ok(())
    }

    /// Order-independent fingerprint of the matched nodes and edges.
    ///
    /// Stable across processes and releases (FNV-1a over a fixed byte layout),
    /// unlike `std::hash::DefaultHasher`, whose output may change between Rust
    /// releases. Two equal paths always yield the same string.
    pub fn fingerprint(&self) -> String {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |bytes: &[u8]| {
            for b in bytes {
                h ^= u64::from(*b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for node in &self.nodes {
            mix(&node.to_le_bytes());
        }
        for edge in &self.edges {
            mix(&edge.segment_id.to_le_bytes());
            mix(&edge.row_in_segment.to_le_bytes());
        }
        format!("{h:016x}")
    }
}

// ---------------------------------------------------------------------------
// Deterministic dedup / top-k
// ---------------------------------------------------------------------------

/// Canonical identity of a match, used to collapse duplicates before truncation.
///
/// Parallel matchers find the same cycle from each of its rotations, and the
/// same path from each direction. The variant decides which of those are
/// considered the same match:
///
/// * [`DedupKey::Ring`] — cycles; rotation-invariant, so `(a,b,c)`, `(b,c,a)`
///   and `(c,a,b)` collapse to one;
/// * [`DedupKey::Path`] — ordered paths; direction matters, so `(a,b)` and
///   `(b,a)` stay distinct;
/// * [`DedupKey::Explicit`] — caller-supplied identity (e.g. a case id).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DedupKey {
    Ring(Vec<u64>),
    Path(Vec<u64>),
    Explicit(String),
}

impl DedupKey {
    /// Order-sensitive key.
    pub fn path(nodes: impl IntoIterator<Item = u64>) -> Self {
        Self::Path(nodes.into_iter().collect())
    }

    /// Rotation-invariant key: rotate so the smallest id (ties broken by the
    /// following sequence) comes first, then store the sequence.
    pub fn ring(nodes: impl IntoIterator<Item = u64>) -> Self {
        let nodes: Vec<u64> = nodes.into_iter().collect();
        Self::Ring(canonical_rotation(&nodes))
    }

    pub fn explicit(key: impl Into<String>) -> Self {
        Self::Explicit(key.into())
    }

    /// The underlying node ids, in canonical order.
    pub fn nodes(&self) -> &[u64] {
        match self {
            DedupKey::Ring(v) | DedupKey::Path(v) => v,
            DedupKey::Explicit(_) => &[],
        }
    }
}

/// Lexicographically smallest rotation of `nodes`; identity for empty / 1-node
/// slices. O(n²) worst case, but match arity is a handful of nodes.
fn canonical_rotation(nodes: &[u64]) -> Vec<u64> {
    if nodes.len() <= 1 {
        return nodes.to_vec();
    }
    let mut best = 0usize;
    for start in 1..nodes.len() {
        for offset in 0..nodes.len() {
            let a = nodes[(start + offset) % nodes.len()];
            let b = nodes[(best + offset) % nodes.len()];
            if a < b {
                best = start;
                break;
            }
            if a > b {
                break;
            }
        }
    }
    (0..nodes.len())
        .map(|offset| nodes[(best + offset) % nodes.len()])
        .collect()
}

/// Sort by key, drop duplicates, keep the first `k`.
///
/// The result depends only on the *set* of items, never on the order they were
/// produced in — so a 4-thread matcher and a 1-thread matcher return identical
/// rows for identical input. `k == 0` returns empty.
pub fn topk_dedup<T, F>(items: impl IntoIterator<Item = T>, k: usize, key: F) -> Vec<T>
where
    F: Fn(&T) -> DedupKey,
{
    if k == 0 {
        return Vec::new();
    }
    let mut keyed: Vec<(DedupKey, T)> = items.into_iter().map(|item| (key(&item), item)).collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    keyed.dedup_by(|a, b| a.0 == b.0);
    keyed.truncate(k);
    keyed.into_iter().map(|(_, item)| item).collect()
}

/// Deterministic top-k by descending score, ties broken by ascending key.
///
/// Use when "top" means "highest value" rather than "first in canonical order".
/// `NaN` scores are ordered last so a broken score cannot displace real
/// evidence.
pub fn topk_dedup_by_score<T, F, S>(
    items: impl IntoIterator<Item = T>,
    k: usize,
    key: F,
    score: S,
) -> Vec<T>
where
    F: Fn(&T) -> DedupKey,
    S: Fn(&T) -> f64,
{
    if k == 0 {
        return Vec::new();
    }
    let mut keyed: Vec<(DedupKey, f64, T)> = items
        .into_iter()
        .map(|item| (key(&item), score(&item), item))
        .collect();
    keyed.sort_by(|a, b| {
        let sa = if a.1.is_nan() { f64::NEG_INFINITY } else { a.1 };
        let sb = if b.1.is_nan() { f64::NEG_INFINITY } else { b.1 };
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    keyed.dedup_by(|a, b| a.0 == b.0);

    // `dedup_by` keeps the *first* of a run, which after sorting is the highest
    // score for that key — exactly what we want.
    keyed.truncate(k);
    keyed.into_iter().map(|(_, _, item)| item).collect()
}

// ---------------------------------------------------------------------------
// Receipt
// ---------------------------------------------------------------------------

/// Why a result is partial, in transport-stable terms.
///
/// Mirrors [`TruncReason`] and adds two cases the wire needs: an explicit
/// result `Limit` (not a resource cap) and `Cancelled` (client disconnect).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IncompleteReason {
    Edges,
    Nodes,
    Frontier,
    Rows,
    Hops,
    Memory,
    Deadline,
    Cancelled,
    Limit,
    /// A time point was requested but nothing enforced it, so the rows cannot
    /// be read as an answer about that point. Distinct from `Limit`: the search
    /// finished, the *question* was not the one that ran.
    TimePointNotApplied,
}

impl IncompleteReason {
    pub fn as_str(self) -> &'static str {
        match self {
            IncompleteReason::Edges => "edges",
            IncompleteReason::Nodes => "nodes",
            IncompleteReason::Frontier => "frontier",
            IncompleteReason::Rows => "rows",
            IncompleteReason::Hops => "hops",
            IncompleteReason::Memory => "memory",
            IncompleteReason::Deadline => "deadline",
            IncompleteReason::Cancelled => "cancelled",
            IncompleteReason::Limit => "limit",
            IncompleteReason::TimePointNotApplied => "time_point_not_applied",
        }
    }
}

impl fmt::Display for IncompleteReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<TruncReason> for IncompleteReason {
    fn from(reason: TruncReason) -> Self {
        match reason {
            TruncReason::Edges => IncompleteReason::Edges,
            TruncReason::Nodes => IncompleteReason::Nodes,
            TruncReason::Frontier => IncompleteReason::Frontier,
            TruncReason::Rows => IncompleteReason::Rows,
            TruncReason::Hops => IncompleteReason::Hops,
            TruncReason::Memory => IncompleteReason::Memory,
            TruncReason::Deadline => IncompleteReason::Deadline,
        }
    }
}

/// Structured explanation of a partial result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Incomplete {
    pub reason: IncompleteReason,
    /// 1-based hop during which the cap tripped; `0` when not hop-scoped.
    pub hop: usize,
    /// Value observed when the cap tripped, if the operator tracked one.
    pub observed: Option<u64>,
    /// The configured cap, if any.
    pub limit: Option<u64>,
}

impl Incomplete {
    pub fn new(reason: IncompleteReason) -> Self {
        Self {
            reason,
            hop: 0,
            observed: None,
            limit: None,
        }
    }

    pub fn at_hop(mut self, hop: usize) -> Self {
        self.hop = hop;
        self
    }

    pub fn with_observation(mut self, observed: u64, limit: u64) -> Self {
        self.observed = Some(observed);
        self.limit = Some(limit);
        self
    }
}

impl From<&Truncation> for Incomplete {
    fn from(t: &Truncation) -> Self {
        Self {
            reason: t.reason.into(),
            hop: t.hop,
            observed: Some(t.observed),
            limit: Some(t.limit),
        }
    }
}

impl From<Option<&Truncation>> for Incomplete {
    fn from(t: Option<&Truncation>) -> Self {
        match t {
            Some(t) => t.into(),
            // Callers only construct an `Incomplete` from a present truncation;
            // `None` means "no cap tripped", which is `Limit` only when the
            // caller explicitly asked for a capped answer.
            None => Self::new(IncompleteReason::Limit),
        }
    }
}

/// Work counters attached to a receipt.
///
/// Counters, not timings: a duration would make the receipt non-reproducible.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryStats {
    pub rows: u64,
    pub hops: usize,
    pub edges_scanned: u64,
    pub nodes_visited: u64,
    pub peak_frontier: usize,
}

impl From<&TraversalStats> for QueryStats {
    fn from(s: &TraversalStats) -> Self {
        Self {
            rows: s.rows,
            hops: s.hops,
            edges_scanned: s.edges_scanned,
            nodes_visited: s.nodes_visited,
            peak_frontier: s.peak_frontier,
        }
    }
}

impl From<TraversalStats> for QueryStats {
    fn from(s: TraversalStats) -> Self {
        (&s).into()
    }
}

// ---------------------------------------------------------------------------
// Freshness: which knowledge state a result reflects (G8 / G9, plan R1 / R2)
// ---------------------------------------------------------------------------

/// Which clock a [`Freshness`] watermark was read from.
///
/// The two exist because a batch-refreshed graph has two different notions of
/// "when": the commit time recorded on the ingested row, and the publish time
/// of the segment that carries it. They coincide on a single flush and diverge
/// as soon as history is backfilled, so a watermark is meaningless without
/// saying which one it is.
///
/// Core deliberately has **no default preference**: the store reports
/// [`KnowledgeClock::SegmentSystemTs`] because that is what it physically
/// knows, and a refresh pipeline that carries `recorded_at` from the ingestion
/// table stamps [`KnowledgeClock::RecordedAt`] instead. Picking one here would
/// silently decide an open question (v3 plan Q22).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeClock {
    /// Commit/knowledge time of the ingestion row (`recorded_at`).
    RecordedAt,
    /// Publish time of the newest flushed segment (`manifest.system_ts`).
    SegmentSystemTs,
}

impl KnowledgeClock {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RecordedAt => "recorded_at",
            Self::SegmentSystemTs => "segment_system_ts",
        }
    }
}

impl fmt::Display for KnowledgeClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a result can know, and what it cannot (plan R1 / R2).
///
/// Two independent limits travel with a result:
///
/// * `as_known_at` — the newest knowledge time this result includes. A report
///   built at 15:00 from a graph refreshed at 14:45 must not present the
///   14:30–15:00 gap as "no such transaction".
/// * `history_floor` — the oldest knowledge time still answerable. Below it,
///   delayed GC has reclaimed the files, and `AS KNOWN AT` becomes a hard error
///   in the store rather than a smaller answer.
///
/// Both are read from store state, never from a wall clock, so a receipt stays
/// byte-identical across replays of the same snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Freshness {
    /// Newest knowledge time included in this result.
    pub as_known_at: i64,
    /// Clock `as_known_at` was read from.
    pub clock: KnowledgeClock,
    /// Oldest knowledge time still answerable; `None` = full history retained.
    pub history_floor: Option<i64>,
    /// Newest business time present in the data, when the caller knows it.
    pub business_max: Option<i64>,
}

impl Freshness {
    pub const fn new(as_known_at: i64, clock: KnowledgeClock) -> Self {
        Self {
            as_known_at,
            clock,
            history_floor: None,
            business_max: None,
        }
    }

    pub const fn with_history_floor(mut self, floor: i64) -> Self {
        self.history_floor = Some(floor);
        self
    }

    pub const fn with_business_max(mut self, business_max: i64) -> Self {
        self.business_max = Some(business_max);
        self
    }

    /// `true` when history at business time `t` is still retained.
    ///
    /// A query below the floor is not "empty" — it is unanswerable, and the
    /// difference matters to anyone reading an absence as evidence.
    pub fn retains_business_time(&self, t: i64) -> bool {
        self.history_floor.map_or(true, |floor| t >= floor)
    }

    /// How far behind `requested` this watermark is, or `None` when current.
    pub fn lag(&self, requested: i64) -> Option<i64> {
        (self.as_known_at < requested).then(|| requested - self.as_known_at)
    }

    /// Compare against the knowledge time the caller believes it is reading.
    pub fn verdict(&self, requested: i64) -> FreshnessVerdict {
        match self.lag(requested) {
            None => FreshnessVerdict::Current,
            Some(behind_by) => FreshnessVerdict::Stale { behind_by },
        }
    }

    /// Reject a watermark that cannot come from a consistent store state.
    pub fn validate(&self) -> Result<()> {
        if let Some(floor) = self.history_floor {
            if floor > self.as_known_at {
                return Err(GtvError::InvalidArgument(format!(
                    "freshness is inconsistent: history floor {floor} is later than \
                     the watermark {}",
                    self.as_known_at
                )));
            }
        }
        Ok(())
    }
}

impl fmt::Display for Freshness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "as known at {} ({})",
            self.as_known_at,
            self.clock.as_str()
        )
    }
}

/// Outcome of comparing a receipt's watermark with a requested knowledge time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "behind_by")]
pub enum FreshnessVerdict {
    /// The watermark reaches the requested knowledge time.
    Current,
    /// The graph has not been refreshed that far yet.
    Stale { behind_by: i64 },
}

impl FreshnessVerdict {
    pub const fn is_current(self) -> bool {
        matches!(self, Self::Current)
    }
}

impl fmt::Display for FreshnessVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Current => f.write_str("current"),
            Self::Stale { behind_by } => write!(f, "stale by {behind_by}"),
        }
    }
}

/// Why a `NoMatch` is, or is not, a proof of absence (plan R1 / R3).
///
/// The engine's R2 rule says an empty result never proves absence on its own.
/// This enum names the three ways that happens, because they need different
/// answers: refresh the graph, widen the retention window, or accept the
/// answer. Collapsing them into a bare `false` reproduces exactly the failure
/// a regulator asks about ("why is the 14:30 transaction missing from the 15:00
/// report?").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Absence {
    /// No rows, complete search, watermark covers the requested time.
    Proven,
    /// No rows, but the graph was refreshed before the requested knowledge
    /// time: the row may not have arrived yet.
    NotYetRefreshed { behind_by: i64 },
    /// No rows, but history at the queried business time was reclaimed.
    HistoryCollected { floor: i64 },
    /// The question does not apply: rows came back, or the search was capped.
    NotApplicable,
}

impl Absence {
    pub const fn proves_absence(self) -> bool {
        matches!(self, Self::Proven)
    }
}

impl fmt::Display for Absence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proven => f.write_str("absence proven"),
            Self::NotYetRefreshed { behind_by } => {
                write!(f, "not yet refreshed (behind by {behind_by})")
            }
            Self::HistoryCollected { floor } => {
                write!(f, "history collected below {floor}")
            }
            Self::NotApplicable => f.write_str("absence not the question"),
        }
    }
}

/// Whether a requested time point was actually honoured, or only recorded.
///
/// A receipt names the state it was computed against (`as_of`, `snapshot_id`).
/// That claim has two very different provenances, and conflating them is how a
/// receipt becomes misleading:
///
/// * [`PointBinding::Applied`] — something enforced the point. The engine
///   filtered by it, or the requested snapshot was verified to be the one the
///   store is serving. The answer below *is* about that point.
/// * [`PointBinding::Declared`] — the caller asked for the point and the
///   receipt kept the request, but nothing made the query answer for it. The
///   rows may come from anywhere (`head`), so the receipt must not be read as
///   evidence about that point.
///
/// `Declared` therefore cannot coexist with a conclusive tri-state (see
/// [`QueryReceipt::validate`]). Without this distinction a client could ask for
/// 14:00, run a statement with no time predicate, and receive rows stamped
/// `as_of: 14:00` — a receipt that testifies to something nobody checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointBinding {
    /// The point was enforced before the query ran.
    Applied,
    /// The point was recorded because it was requested; nothing enforced it.
    Declared,
}

impl PointBinding {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Applied => "applied",
            Self::Declared => "declared",
        }
    }

    /// True when the receipt may be treated as evidence about the point.
    pub const fn is_applied(self) -> bool {
        matches!(self, Self::Applied)
    }
}

impl fmt::Display for PointBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a query did, in a form that can be re-checked later.
///
/// Build with [`QueryReceipt::new`] then chain the `with_*` setters; call
/// [`QueryReceipt::validate`] before sending. `query_id` is the caller's
/// correlation id (never generated here, so replays stay comparable).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryReceipt {
    pub protocol_version: u32,
    pub engine_version: String,
    pub query_id: String,
    pub snapshot_id: Option<u64>,
    pub as_of: Option<i64>,
    /// Provenance of `snapshot_id`. `Some` exactly when `snapshot_id` is, and
    /// required by [`Self::validate`] once a snapshot is named.
    #[serde(default)]
    pub snapshot_binding: Option<PointBinding>,
    /// Provenance of `as_of`. `Some` exactly when `as_of` is, for the same
    /// reason: naming a time point without saying who enforced it is the hole
    /// this field closes.
    #[serde(default)]
    pub as_of_binding: Option<PointBinding>,
    pub plan_hash: Option<String>,
    pub tri_state: TriState,
    pub incomplete: Option<Incomplete>,
    pub stats: QueryStats,
    /// Version of the caller's context that shaped this result (e.g. rule set).
    /// Opaque to core: core only guarantees it is carried and compared.
    pub context_version: Option<String>,
    /// Knowledge watermark this result was computed from, when the caller could
    /// supply one. Absent means "unknown", which callers must not read as
    /// "fresh": an unproven watermark is not a zero lag.
    #[serde(default)]
    pub freshness: Option<Freshness>,
}

impl QueryReceipt {
    pub fn new(query_id: impl Into<String>, tri_state: TriState) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            engine_version: ENGINE_VERSION.to_string(),
            query_id: query_id.into(),
            snapshot_id: None,
            as_of: None,
            snapshot_binding: None,
            as_of_binding: None,
            plan_hash: None,
            tri_state,
            incomplete: None,
            stats: QueryStats::default(),
            context_version: None,
            freshness: None,
        }
    }

    /// Receipt for a query that finished and found rows.
    pub fn found(query_id: impl Into<String>) -> Self {
        Self::new(query_id, TriState::Found)
    }

    /// Receipt for a query that finished and found nothing.
    pub fn no_match(query_id: impl Into<String>) -> Self {
        Self::new(query_id, TriState::NoMatch)
    }

    /// Receipt for a query that did not finish. `reason` is mandatory by
    /// construction, which is the point of having this constructor.
    pub fn incomplete(query_id: impl Into<String>, reason: Incomplete) -> Self {
        let mut receipt = Self::new(query_id, TriState::Incomplete);
        receipt.incomplete = Some(reason);
        receipt
    }

    /// Record a snapshot the caller asked for, **without** claiming it was
    /// checked. Sets the binding to [`PointBinding::Declared`], which makes a
    /// conclusive tri-state invalid: use [`Self::with_snapshot_verified`] once
    /// the id has been compared against the store's live snapshot.
    pub fn with_snapshot(mut self, snapshot_id: u64) -> Self {
        self.snapshot_id = Some(snapshot_id);
        self.snapshot_binding = Some(PointBinding::Declared);
        self
    }

    /// Record a snapshot that was verified against what the store is serving.
    pub fn with_snapshot_verified(mut self, snapshot_id: u64) -> Self {
        self.snapshot_id = Some(snapshot_id);
        self.snapshot_binding = Some(PointBinding::Applied);
        self
    }

    /// Record a business time the caller asked for, **without** claiming the
    /// query answered for it. See [`Self::with_snapshot`] for why that must not
    /// be assumed: most statements have no time predicate at all.
    pub fn with_as_of(mut self, as_of: i64) -> Self {
        self.as_of = Some(as_of);
        self.as_of_binding = Some(PointBinding::Declared);
        self
    }

    /// Record a business time the engine actually enforced.
    pub fn with_as_of_applied(mut self, as_of: i64) -> Self {
        self.as_of = Some(as_of);
        self.as_of_binding = Some(PointBinding::Applied);
        self
    }

    /// Whether the result may be read as evidence about `as_of`.
    pub fn as_of_applied(&self) -> bool {
        matches!(self.as_of_binding, Some(PointBinding::Applied))
    }

    /// Whether the result may be read as evidence about `snapshot_id`.
    pub fn snapshot_verified(&self) -> bool {
        matches!(self.snapshot_binding, Some(PointBinding::Applied))
    }

    /// True when a time point was named but nothing enforced it.
    pub fn has_declared_point(&self) -> bool {
        matches!(self.as_of_binding, Some(PointBinding::Declared))
            || matches!(self.snapshot_binding, Some(PointBinding::Declared))
    }

    pub fn with_plan_hash(mut self, plan_hash: impl Into<String>) -> Self {
        self.plan_hash = Some(plan_hash.into());
        self
    }

    pub fn with_stats(mut self, stats: QueryStats) -> Self {
        self.stats = stats;
        self
    }

    pub fn with_context_version(mut self, version: impl Into<String>) -> Self {
        self.context_version = Some(version.into());
        self
    }

    /// Stamp the knowledge watermark the result was computed from.
    ///
    /// Callers pass what the store reported (`SegmentStore::freshness()`), or a
    /// `recorded_at` high-water mark when the refresh pipeline carries one.
    pub fn with_freshness(mut self, freshness: Freshness) -> Self {
        self.freshness = Some(freshness);
        self
    }

    /// Compare this result's watermark with the knowledge time the caller
    /// believes it is reading. `None` when the receipt carries no watermark.
    pub fn freshness_verdict(&self, requested: i64) -> Option<FreshnessVerdict> {
        self.freshness.map(|f| f.verdict(requested))
    }

    /// Classify a `NoMatch`: proven, merely not refreshed yet, or unanswerable
    /// because the history was reclaimed.
    ///
    /// `requested_k` is the knowledge time the caller is asking from (its own
    /// "now", or the `AS KNOWN AT` it requested); `business_t` is the business
    /// time the query was framed at, when it has one — it decides whether the
    /// retention floor already invalidates the answer.
    ///
    /// Answers [`Absence::NotApplicable`] when there are rows or the search was
    /// capped, so callers can use this unconditionally instead of guarding it.
    pub fn absence_at(&self, requested_k: i64, business_t: Option<i64>) -> Absence {
        if self.tri_state != TriState::NoMatch || !self.is_complete() {
            return Absence::NotApplicable;
        }
        // An empty result under a declared `as_of` says nothing about that
        // time: the rows it would have matched were never filtered by it.
        if matches!(self.as_of_binding, Some(PointBinding::Declared)) {
            return Absence::NotApplicable;
        }
        let Some(freshness) = self.freshness else {
            // No watermark: an empty result is not evidence of anything.
            return Absence::NotApplicable;
        };
        if let (Some(t), Some(floor)) = (business_t, freshness.history_floor) {
            if t < floor {
                return Absence::HistoryCollected { floor };
            }
        }
        match freshness.verdict(requested_k) {
            FreshnessVerdict::Current => Absence::Proven,
            FreshnessVerdict::Stale { behind_by } => Absence::NotYetRefreshed { behind_by },
        }
    }

    /// [`Self::absence_at`] folded to a boolean: safe only when absence is
    /// positively proven.
    pub fn proves_absence_at(&self, requested_k: i64, business_t: Option<i64>) -> bool {
        self.absence_at(requested_k, business_t).proves_absence()
    }

    /// Attach truncation from an operator result, switching the tri-state to
    /// `Incomplete` when a cap tripped.
    pub fn with_truncation(mut self, truncation: Option<&Truncation>) -> Self {
        match truncation {
            Some(t) => {
                self.tri_state = TriState::Incomplete;
                self.incomplete = Some(t.into());
            }
            None => {
                if self.stats.rows > 0 {
                    self.tri_state = TriState::Found;
                }
                self.incomplete = None;
            }
        }
        self
    }

    pub fn is_complete(&self) -> bool {
        self.tri_state != TriState::Incomplete
    }

    /// Check the invariants a receipt must satisfy before it leaves the process.
    ///
    /// Rejects the three failure modes that would make the receipt misleading:
    /// a partial result presented as conclusive, an inconclusive result with no
    /// explanation, and a named time point that nothing enforced presented as
    /// if it had been.
    pub fn validate(&self) -> Result<()> {
        if !is_supported_version(self.protocol_version) {
            return Err(GtvError::InvalidArgument(format!(
                "unsupported receipt protocol version {}",
                self.protocol_version
            )));
        }
        if self.query_id.is_empty() {
            return Err(GtvError::InvalidArgument(
                "receipt has an empty query_id".into(),
            ));
        }
        match (self.tri_state, &self.incomplete) {
            (TriState::Incomplete, None) => {
                return Err(GtvError::InvalidArgument(
                    "Incomplete result must carry a reason".into(),
                ))
            }
            (TriState::Found | TriState::NoMatch, Some(_)) => {
                return Err(GtvError::InvalidArgument(
                    "conclusive result must not carry an incomplete reason".into(),
                ))
            }
            _ => {}
        }
        // Naming a time point is a claim about provenance, so it has to say who
        // enforced it. These two checks are what stop "the client asked for
        // 14:00" from being filed as "we answered for 14:00".
        self.check_binding("as_of", self.as_of.is_some(), self.as_of_binding)?;
        self.check_binding(
            "snapshot_id",
            self.snapshot_id.is_some(),
            self.snapshot_binding,
        )?;
        if let Some(freshness) = &self.freshness {
            freshness.validate()?;
            // A `NoMatch` below the retention floor is not an answer: the files
            // that would have contained the rows were reclaimed, so reporting
            // it as a conclusive empty result is the misleading case R3 exists
            // to prevent.
            if self.tri_state == TriState::NoMatch {
                if let (Some(as_of), Some(floor)) = (self.as_of, freshness.history_floor) {
                    if as_of < floor {
                        return Err(GtvError::InvalidArgument(format!(
                            "NoMatch at as_of {as_of} is below the GC floor {floor}; \
                             absence there is unanswerable, not proven"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// A time point may be named only together with its binding, and a merely
    /// declared one may not accompany a conclusive result.
    fn check_binding(
        &self,
        name: &str,
        present: bool,
        binding: Option<PointBinding>,
    ) -> Result<()> {
        match (present, binding) {
            (false, None) => Ok(()),
            (true, None) => Err(GtvError::InvalidArgument(format!(
                "{name} is recorded without a binding; say whether it was applied"
            ))),
            (false, Some(_)) => Err(GtvError::InvalidArgument(format!(
                "{name} binding is set but {name} is absent"
            ))),
            (true, Some(PointBinding::Declared)) if self.tri_state != TriState::Incomplete => {
                Err(GtvError::InvalidArgument(format!(
                    "conclusive result at a merely declared {name}; \
                     a time point nothing enforced cannot be answered conclusively"
                )))
            }
            (true, Some(PointBinding::Applied)) => Ok(()),
            (true, Some(PointBinding::Declared)) => Ok(()),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("QueryReceipt is always serializable")
    }

    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| GtvError::InvalidArgument(format!("malformed receipt: {e}")))
    }

    /// Flatten into Arrow schema metadata.
    ///
    /// The full JSON is carried under [`RECEIPT_METADATA_JSON`] so a reader can
    /// round-trip losslessly; the per-field keys exist so naive consumers (and
    /// `EXPLAIN`-style tooling) can read `tri_state` without a JSON parser.
    pub fn to_arrow_metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        meta.insert(RECEIPT_METADATA_JSON.to_string(), self.to_json());
        meta.insert(
            format!("{RECEIPT_METADATA_PREFIX}protocol_version"),
            self.protocol_version.to_string(),
        );
        meta.insert(
            format!("{RECEIPT_METADATA_PREFIX}engine_version"),
            self.engine_version.clone(),
        );
        meta.insert(
            format!("{RECEIPT_METADATA_PREFIX}query_id"),
            self.query_id.clone(),
        );
        meta.insert(
            format!("{RECEIPT_METADATA_PREFIX}tri_state"),
            tri_state_str(self.tri_state).to_string(),
        );
        if let Some(id) = self.snapshot_id {
            meta.insert(format!("{RECEIPT_METADATA_PREFIX}snapshot_id"), id.to_string());
        }
        if let Some(binding) = self.snapshot_binding {
            meta.insert(
                format!("{RECEIPT_METADATA_PREFIX}snapshot_binding"),
                binding.as_str().to_string(),
            );
        }
        if let Some(as_of) = self.as_of {
            meta.insert(format!("{RECEIPT_METADATA_PREFIX}as_of"), as_of.to_string());
        }
        if let Some(binding) = self.as_of_binding {
            meta.insert(
                format!("{RECEIPT_METADATA_PREFIX}as_of_binding"),
                binding.as_str().to_string(),
            );
        }
        if let Some(reason) = &self.incomplete {
            meta.insert(
                format!("{RECEIPT_METADATA_PREFIX}incomplete_reason"),
                reason.reason.as_str().to_string(),
            );
        }
        if let Some(freshness) = &self.freshness {
            meta.insert(
                format!("{RECEIPT_METADATA_PREFIX}as_known_at"),
                freshness.as_known_at.to_string(),
            );
            meta.insert(
                format!("{RECEIPT_METADATA_PREFIX}knowledge_clock"),
                freshness.clock.as_str().to_string(),
            );
            if let Some(floor) = freshness.history_floor {
                meta.insert(format!("{RECEIPT_METADATA_PREFIX}history_floor"), floor.to_string());
            }
        }
        meta
    }

    /// Read a receipt back from schema metadata.
    ///
    /// Returns `Ok(None)` when the metadata carries no receipt at all (a plain
    /// Arrow stream), so callers can distinguish "absent" from "malformed".
    pub fn from_arrow_metadata(meta: &HashMap<String, String>) -> Result<Option<Self>> {
        match meta.get(RECEIPT_METADATA_JSON) {
            Some(json) => Ok(Some(Self::from_json(json)?)),
            None => Ok(None),
        }
    }
}

/// Stable string form of [`TriState`] for metadata and JSON.
pub fn tri_state_str(state: TriState) -> &'static str {
    match state {
        TriState::Found => "found",
        TriState::NoMatch => "no_match",
        TriState::Incomplete => "incomplete",
    }
}

/// `true` when a partial result may be returned to the caller as rows.
///
/// Standard SQL paths answer `false` (R2: return an error, never a silent
/// prefix); extension paths that hand back `Incomplete` explicitly answer
/// `true`. Encoding the rule here keeps every transport agreeing.
pub const fn allows_partial_rows(state: TriState) -> bool {
    !matches!(state, TriState::Incomplete)
}

// ---------------------------------------------------------------------------
// Plan hashing
// ---------------------------------------------------------------------------

/// Hash a rendered plan into a short, stable identifier.
///
/// Plans are rendered deterministically by DataFusion's `display_indent`, so
/// hashing the string is enough to tell "same shape, same snapshot" from "the
/// optimizer picked another join order". Same FNV-1a as
/// [`PathEvidence::fingerprint`] to avoid pulling a hash crate into core.
pub fn plan_hash(rendered_plan: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in rendered_plan.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}
