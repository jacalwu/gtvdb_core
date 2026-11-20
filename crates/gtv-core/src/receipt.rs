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
//! * against which state (`snapshot_id`, `as_of`),
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
//!   `Found`/`NoMatch` while carrying an incomplete reason, and rejects
//!   `Incomplete` without one. Absence of rows must never be reported as proof
//!   of absence (the engine's R2 rule).
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
    pub const fn new(segment_id: u64, row_in_segment: u32) -> Self {
        Self {
            segment_id,
            row_in_segment,
        }
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
    pub plan_hash: Option<String>,
    pub tri_state: TriState,
    pub incomplete: Option<Incomplete>,
    pub stats: QueryStats,
    /// Version of the caller's context that shaped this result (e.g. rule set).
    /// Opaque to core: core only guarantees it is carried and compared.
    pub context_version: Option<String>,
}

impl QueryReceipt {
    pub fn new(query_id: impl Into<String>, tri_state: TriState) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            engine_version: ENGINE_VERSION.to_string(),
            query_id: query_id.into(),
            snapshot_id: None,
            as_of: None,
            plan_hash: None,
            tri_state,
            incomplete: None,
            stats: QueryStats::default(),
            context_version: None,
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

    pub fn with_snapshot(mut self, snapshot_id: u64) -> Self {
        self.snapshot_id = Some(snapshot_id);
        self
    }

    pub fn with_as_of(mut self, as_of: i64) -> Self {
        self.as_of = Some(as_of);
        self
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
    /// Rejects the two failure modes that would make the receipt misleading:
    /// a partial result presented as conclusive, and an inconclusive result
    /// with no explanation.
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
            (TriState::Incomplete, None) => Err(GtvError::InvalidArgument(
                "Incomplete result must carry a reason".into(),
            )),
            (TriState::Found | TriState::NoMatch, Some(_)) => Err(GtvError::InvalidArgument(
                "conclusive result must not carry an incomplete reason".into(),
            )),
            _ => Ok(()),
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
        if let Some(as_of) = self.as_of {
            meta.insert(format!("{RECEIPT_METADATA_PREFIX}as_of"), as_of.to_string());
        }
        if let Some(reason) = &self.incomplete {
            meta.insert(
                format!("{RECEIPT_METADATA_PREFIX}incomplete_reason"),
                reason.reason.as_str().to_string(),
            );
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
