//! SQL surface for `gtv-pattern` temporal motif matching (ROADMAP §3.3 / P1-4).
//!
//! # Why this exists
//!
//! `gtv-pattern` already ships a backtracking matcher with built-in AML shapes
//! (wash-trade rings, ordered temporal paths, diamonds) plus rolling-window and
//! edge-type-sequence constraints. Before this module it had **no SQL exit at
//! all**: `gtv-engine` did not reference the crate even though it was a
//! declared dependency. AML cycle detection could only be expressed in Rust.
//!
//! This module adds the thinnest possible bridge, in the same shape as the M6
//! graph operators:
//!
//! ```sql
//! -- pointwise, literal-only (baseline / compatibility)
//! SELECT * FROM motif('ring3', 100, 500);
//! SELECT * FROM motif_from('ring3', 7, 100, 500, 30);
//!
//! -- one seed batch, one planning cycle, joinable
//! SELECT * FROM gtv_motif_matches;
//! SELECT * FROM gtv_motif_edges;
//! ```
//!
//! `gtv_motif_matches` is one row per match (`match_id`, `n0..n{k-1}`,
//! `event_min`, `event_max`); `gtv_motif_edges` is one row per matched edge
//! (`match_id`, `edge_idx`, `src`, `dst`, `edge_type`, `valid_from`,
//! `valid_to`). Both are built from the **same** `Vec<Match>` in one pass, so
//! the batch tables and the pointwise UDTFs are comparable row-for-row.
//!
//! # What is deliberately *not* changed
//!
//! * No new matching algorithm: every path calls `gtv_pattern::find`,
//!   `find_from`, `find_windowed` or `find_windowed_from`.
//! * `gtv-core/src/csr.rs`, `endpoint.rs`, `segment.rs` — GTVSeg v1 column
//!   layout and the `neighbors()` hot loop are untouched.
//! * `find_bounded` (explicit `TraversalBudget` / `CancelToken`) is not exposed
//!   to SQL yet; `limit` is the bound here. See the roadmap boundary note.
//! * Only graph operators are added. No domain UDTF, no new SQL surface for
//!   non-graph workloads; those still belong in `gtvdb_ext`.

use std::fmt;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Int64Array, RecordBatch, StringArray, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::{Session, TableFunctionArgs, TableFunctionImpl, TableProvider};
use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::ExecutionPlan;
use gtv_core::{
    EdgePredicate, Incomplete, QueryReceipt, Sel, TemporalCSR, TraversalBudget, TraversalStats,
    TriState, Truncation, UNBOUND_SNAPSHOT,
};
use gtv_pattern::{EventOrder, Match, Pattern, WindowedMotif};

use crate::expr_util::{expr_to_i64, expr_to_string, expr_to_u64};

const DEFAULT_LIMIT: usize = 1_000;

fn kernel_err(e: impl fmt::Display) -> DataFusionError {
    DataFusionError::Execution(format!("temporal motif kernel: {e}"))
}

fn arg_err(signature: &str, what: &str) -> DataFusionError {
    DataFusionError::Execution(format!("{signature}: missing `{what}`"))
}

/// Built-in temporal pattern shapes. Kept as a small closed set so the match
/// table has a fixed, SQL-friendly column layout (`n0..n{k-1}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotifShape {
    /// Cycle back to the start, `v0 -> ... -> v0` (wash-trade ring for len 3).
    Ring(usize),
    /// Ordered temporal path of `n` hops (`n + 1` variables).
    Path(usize),
    /// `a -> b`, `a -> c`, `b -> d`, `c -> d`, arms ordered in time.
    Diamond,
}

impl MotifShape {
    pub fn parse(name: &str) -> Result<Self> {
        let shape = match name {
            "ring3" => Self::Ring(3),
            "ring4" => Self::Ring(4),
            "path2" => Self::Path(2),
            "path3" => Self::Path(3),
            "path4" => Self::Path(4),
            "diamond" => Self::Diamond,
            other => {
                return Err(DataFusionError::Execution(format!(
                    "unknown motif shape '{other}'; expected ring3/ring4/path2/path3/path4/diamond"
                )))
            }
        };
        Ok(shape)
    }

    pub fn name(&self) -> String {
        match *self {
            Self::Ring(n) => format!("ring{n}"),
            Self::Path(n) => format!("path{n}"),
            Self::Diamond => "diamond".to_owned(),
        }
    }

    pub fn pattern(&self) -> Pattern {
        match *self {
            Self::Ring(n) => Pattern::ring(n),
            Self::Path(n) => Pattern::temporal_path(n),
            Self::Diamond => Pattern::diamond(),
        }
    }

    pub fn num_vars(&self) -> usize {
        self.pattern().num_vars
    }
}

/// Inclusive delay bounds between two consecutive pattern edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventGap {
    pub before: usize,
    pub after: usize,
    pub min_gap: i64,
    pub max_gap: Option<i64>,
}

/// Complete specification of one motif query.
#[derive(Debug, Clone)]
pub struct MotifSpec {
    pub shape: MotifShape,
    pub valid_at: i64,
    pub limit: usize,
    /// All matched edges must fall within this many event-time units.
    pub max_event_span: Option<i64>,
    /// Matched edge types, ordered by event time, must equal this sequence.
    pub sequence: Vec<u16>,
    /// Optional per-pair time gaps, applied on top of the shape's default
    /// strict ordering. Zero means equal event times are allowed.
    pub event_gaps: Vec<EventGap>,
    /// Optional resource cap. When set, the matcher runs through
    /// `gtv_pattern::find_bounded_motif`, counts every scanned edge, and reports
    /// truncation instead of pretending the prefix is the answer.
    pub budget: Option<TraversalBudget>,
}

/// Spec equality ignores the resource budget: two queries asking the same
/// thing of the same snapshot are the same query even if one is budgeted.
impl PartialEq for MotifSpec {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.valid_at == other.valid_at
            && self.limit == other.limit
            && self.max_event_span == other.max_event_span
            && self.sequence == other.sequence
            && self.event_gaps == other.event_gaps
    }
}

impl MotifSpec {
    pub fn new(shape: MotifShape, valid_at: i64) -> Self {
        Self {
            shape,
            valid_at,
            limit: DEFAULT_LIMIT,
            max_event_span: None,
            sequence: Vec::new(),
            event_gaps: Vec::new(),
            budget: None,
        }
    }

    /// Cap the search. `budget` is ignored for windowed specs (the bounded
    /// kernel takes a plain `Pattern`), which [`find_matches_bounded`] rejects
    /// explicitly rather than silently running unbounded.
    pub fn with_budget(mut self, budget: TraversalBudget) -> Self {
        self.budget = Some(budget);
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_window(mut self, span: i64) -> Self {
        self.max_event_span = Some(span);
        self
    }

    pub fn with_sequence(mut self, sequence: Vec<u16>) -> Self {
        self.sequence = sequence;
        self
    }

    /// Require `valid_from[after] - valid_from[before]` to fall inside the
    /// inclusive bounds. Indices are pattern-edge indices, not node ids.
    pub fn with_event_gap(
        mut self,
        before: usize,
        after: usize,
        min_gap: i64,
        max_gap: Option<i64>,
    ) -> Result<Self> {
        let edges = self.shape.pattern().edges.len();
        if before >= edges || after >= edges {
            return Err(DataFusionError::Execution(format!(
                "motif: event gap indices {before}->{after} outside a {edges}-edge pattern"
            )));
        }
        if max_gap.is_some_and(|max| max < min_gap) {
            return Err(DataFusionError::Execution(format!(
                "motif: event gap {before}->{after} has upper bound below lower bound {min_gap}"
            )));
        }
        self.event_gaps.push(EventGap {
            before,
            after,
            min_gap,
            max_gap,
        });
        Ok(self)
    }

    pub fn num_vars(&self) -> usize {
        self.shape.num_vars()
    }

    /// Window / sequence constraints deliberately bypass the optimized 3-cycle
    /// fast path in `find`; this is why the two branches below are separate.
    pub fn is_windowed(&self) -> bool {
        self.max_event_span.is_some() || !self.sequence.is_empty() || !self.event_gaps.is_empty()
    }

    fn windowed_motif(&self) -> WindowedMotif {
        let mut pattern = self.shape.pattern();
        for gap in &self.event_gaps {
            let order = EventOrder::window(gap.before, gap.after, gap.min_gap, gap.max_gap);
            if let Some(existing) = pattern
                .event_orders
                .iter_mut()
                .find(|o| o.before == gap.before && o.after == gap.after)
            {
                *existing = order;
            } else {
                pattern.event_orders.push(order);
            }
        }
        let mut motif = WindowedMotif::new(pattern);
        if let Some(span) = self.max_event_span {
            motif = motif.with_window(span);
        }
        if !self.sequence.is_empty() {
            motif = motif.with_sequence(self.sequence.clone());
        }
        motif
    }
}

/// Run the existing kernel once. `seeds = None` scans all start nodes;
/// `Some(seeds)` anchors variable 0 at each seed in order. `limit` is global:
/// each seed receives the remaining budget, so the union never exceeds it.
pub fn find_matches(
    csr: &TemporalCSR,
    spec: &MotifSpec,
    seeds: Option<&[u64]>,
) -> Result<Vec<Match>> {
    if spec.limit == 0 {
        return Ok(Vec::new());
    }
    let pattern = spec.shape.pattern();
    let motif = if spec.is_windowed() {
        Some(spec.windowed_motif())
    } else {
        None
    };
    let mut out: Vec<Match> = Vec::new();

    match seeds {
        Some(seeds) => {
            for &start in seeds {
                if out.len() >= spec.limit {
                    break;
                }
                let remaining = spec.limit - out.len();
                let mut one = match &motif {
                    Some(m) => {
                        gtv_pattern::find_windowed_from(csr, m, start, spec.valid_at, remaining)
                    }
                    None => gtv_pattern::find_from(csr, &pattern, start, spec.valid_at, remaining),
                }
                .map_err(kernel_err)?;
                out.append(&mut one);
            }
        }
        None => {
            out = match &motif {
                Some(m) => gtv_pattern::find_windowed(csr, m, spec.valid_at, spec.limit),
                None => gtv_pattern::find(csr, &pattern, spec.valid_at, spec.limit),
            }
            .map_err(kernel_err)?;
        }
    }
    Ok(out)
}

/// Result of a motif run plus everything a receipt needs.
#[derive(Debug, Clone)]
pub struct MotifOutcome {
    pub matches: Vec<Match>,
    /// `Some` when a resource cap stopped the search. The matches are then a
    /// prefix and must not be reported as the answer (A4 / R2).
    pub truncation: Option<Truncation>,
    pub stats: TraversalStats,
    pub valid_at: i64,
    pub limit: usize,
}

impl MotifOutcome {
    pub fn is_complete(&self) -> bool {
        self.truncation.is_none()
    }

    pub fn tri_state(&self) -> TriState {
        match (self.matches.is_empty(), self.truncation.is_some()) {
            (_, true) => TriState::Incomplete,
            (false, false) => TriState::Found,
            (true, false) => TriState::NoMatch,
        }
    }

    /// Human-readable reason, phrased as what to change rather than what broke.
    pub fn incomplete(&self) -> Option<Incomplete> {
        self.truncation.as_ref().map(Incomplete::from)
    }

    /// Build a receipt for this run. `query_id` is the caller's correlation id.
    pub fn receipt(&self, query_id: impl Into<String>) -> QueryReceipt {
        let mut receipt = QueryReceipt::new(query_id, self.tri_state())
            .with_as_of_applied(self.valid_at)
            .with_stats(self.stats.clone().into());
        if let Some(t) = &self.truncation {
            receipt.incomplete = Some(t.into());
        }
        receipt
    }
}

/// Like [`find_matches`], but honours `spec.budget`.
///
/// With no budget and no predicate this is exactly [`find_matches`] plus
/// default stats. With a budget, window/sequence constraints, event gaps or a
/// push-down predicate it routes through `gtv_pattern::find_bounded_motif`,
/// which accounts for every scanned edge; exceeding the cap returns the prefix
/// **and** a [`Truncation`] instead of an error.
pub fn find_matches_bounded(
    csr: &TemporalCSR,
    spec: &MotifSpec,
    seeds: Option<&[u64]>,
) -> Result<MotifOutcome> {
    find_matches_bounded_with_predicate(csr, spec, seeds, None)
}

/// Bounded motif search with a push-down edge predicate.
///
/// The predicate is evaluated for every candidate edge inside the kernel, so
/// an attribute range cannot be confused with a post-filter that would let
/// rejected edges consume the result budget. When no budget is supplied an
/// effectively unlimited one is used, because that is the only path that can
/// honour the predicate.
pub fn find_matches_bounded_with_predicate(
    csr: &TemporalCSR,
    spec: &MotifSpec,
    seeds: Option<&[u64]>,
    predicate: Option<&dyn EdgePredicate>,
) -> Result<MotifOutcome> {
    if spec.budget.is_none() && predicate.is_none() {
        let matches = find_matches(csr, spec, seeds)?;
        let mut stats = TraversalStats {
            rows: matches.len() as u64,
            ..Default::default()
        };
        stats.hops = matches
            .iter()
            .map(|m| m.edges.len())
            .max()
            .unwrap_or_default();
        return Ok(MotifOutcome {
            matches,
            truncation: None,
            stats,
            valid_at: spec.valid_at,
            limit: spec.limit,
        });
    }

    if spec.limit == 0 {
        return Ok(MotifOutcome {
            matches: Vec::new(),
            truncation: None,
            stats: TraversalStats::default(),
            valid_at: spec.valid_at,
            limit: 0,
        });
    }

    let budget = spec.budget.clone().unwrap_or_default();
    let motif = if spec.is_windowed() || predicate.is_some() {
        Some(spec.windowed_motif())
    } else {
        None
    };
    let pattern = spec.shape.pattern();
    let set = match seeds {
        Some(seeds) if !seeds.is_empty() => {
            let snapshot = csr.snapshot_id().unwrap_or(UNBOUND_SNAPSHOT);
            let candidates =
                Sel::from_ids(snapshot, csr.node_count() as u64, seeds.iter().copied())
                    .map_err(kernel_err)?;
            match motif.as_ref() {
                Some(m) => gtv_pattern::find_bounded_motif(
                    csr,
                    m,
                    spec.valid_at,
                    spec.limit,
                    &candidates,
                    &budget,
                    None,
                    predicate,
                )
                .map_err(kernel_err)?,
                None => gtv_pattern::find_bounded(
                    csr,
                    &pattern,
                    spec.valid_at,
                    spec.limit,
                    &candidates,
                    &budget,
                    None,
                )
                .map_err(kernel_err)?,
            }
        }
        _ => match motif.as_ref() {
            Some(m) => {
                let all = Sel::all(snapshot_of(csr), csr.node_count() as u64);
                gtv_pattern::find_bounded_motif(
                    csr,
                    m,
                    spec.valid_at,
                    spec.limit,
                    &all,
                    &budget,
                    None,
                    predicate,
                )
                .map_err(kernel_err)?
            }
            None => gtv_pattern::find_bounded_all(
                csr,
                &pattern,
                spec.valid_at,
                spec.limit,
                &budget,
                None,
            )
            .map_err(kernel_err)?,
        },
    };

    Ok(MotifOutcome {
        matches: set.matches,
        truncation: set.truncation,
        stats: set.stats,
        valid_at: spec.valid_at,
        limit: spec.limit,
    })
}

fn snapshot_of(csr: &TemporalCSR) -> u64 {
    csr.snapshot_id().unwrap_or(UNBOUND_SNAPSHOT)
}

/// One row per match: `match_id`, `n0..n{k-1}`, `event_min`, `event_max`.
pub fn match_schema(num_vars: usize) -> SchemaRef {
    let mut fields = Vec::with_capacity(num_vars + 3);
    fields.push(Field::new("match_id", DataType::UInt64, false));
    for v in 0..num_vars {
        fields.push(Field::new(format!("n{v}"), DataType::UInt64, false));
    }
    fields.push(Field::new("event_min", DataType::Int64, false));
    fields.push(Field::new("event_max", DataType::Int64, false));
    Arc::new(Schema::new(fields))
}

/// One row per matched edge, keyed back to its match.
///
/// `edge_pos` is the edge's position within the match (0, 1, 2, …) and orders
/// the evidence path. `edge_idx` is the edge's slot in the snapshot's CSR
/// arrays — the handle a caller needs to fetch attributes or join back to the
/// source record. They are different numbers and must not be conflated:
/// `edge_pos` was previously (and wrongly) published as `edge_idx`.
pub fn edge_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("match_id", DataType::UInt64, false),
        Field::new("edge_pos", DataType::UInt32, false),
        Field::new("edge_idx", DataType::UInt32, false),
        Field::new("src", DataType::UInt64, false),
        Field::new("dst", DataType::UInt64, false),
        Field::new("edge_type", DataType::UInt16, false),
        Field::new("valid_from", DataType::Int64, false),
        Field::new("valid_to", DataType::Int64, false),
    ]))
}

pub fn build_match_batch(matches: &[Match], num_vars: usize) -> Result<RecordBatch> {
    let mut ids = Vec::with_capacity(matches.len());
    let mut nodes = vec![Vec::with_capacity(matches.len()); num_vars];
    let mut event_min = Vec::with_capacity(matches.len());
    let mut event_max = Vec::with_capacity(matches.len());

    for (i, m) in matches.iter().enumerate() {
        ids.push(i as u64);
        for (v, column) in nodes.iter_mut().enumerate() {
            column.push(m.nodes.get(v).copied().unwrap_or(0));
        }
        event_min.push(m.edges.iter().map(|e| e.valid_from).min().unwrap_or(0));
        event_max.push(m.edges.iter().map(|e| e.valid_from).max().unwrap_or(0));
    }

    let mut cols: Vec<ArrayRef> = Vec::with_capacity(num_vars + 3);
    cols.push(Arc::new(UInt64Array::from(ids)) as ArrayRef);
    for column in nodes {
        cols.push(Arc::new(UInt64Array::from(column)) as ArrayRef);
    }
    cols.push(Arc::new(Int64Array::from(event_min)) as ArrayRef);
    cols.push(Arc::new(Int64Array::from(event_max)) as ArrayRef);
    Ok(RecordBatch::try_new(match_schema(num_vars), cols)?)
}

pub fn build_edge_batch(matches: &[Match]) -> Result<RecordBatch> {
    let rows: usize = matches.iter().map(|m| m.edges.len()).sum();
    let mut ids = Vec::with_capacity(rows);
    let mut pos = Vec::with_capacity(rows);
    let mut idx = Vec::with_capacity(rows);
    let mut src = Vec::with_capacity(rows);
    let mut dst = Vec::with_capacity(rows);
    let mut et = Vec::with_capacity(rows);
    let mut vf = Vec::with_capacity(rows);
    let mut vt = Vec::with_capacity(rows);

    for (i, m) in matches.iter().enumerate() {
        for (j, e) in m.edges.iter().enumerate() {
            ids.push(i as u64);
            pos.push(j as u32);
            idx.push(e.edge_idx);
            src.push(e.src);
            dst.push(e.dst);
            et.push(e.edge_type);
            vf.push(e.valid_from);
            vt.push(e.valid_to);
        }
    }

    Ok(RecordBatch::try_new(
        edge_schema(),
        vec![
            Arc::new(UInt64Array::from(ids)) as ArrayRef,
            Arc::new(UInt32Array::from(pos)) as ArrayRef,
            Arc::new(UInt32Array::from(idx)) as ArrayRef,
            Arc::new(UInt64Array::from(src)) as ArrayRef,
            Arc::new(UInt64Array::from(dst)) as ArrayRef,
            Arc::new(UInt16Array::from(et)) as ArrayRef,
            Arc::new(Int64Array::from(vf)) as ArrayRef,
            Arc::new(Int64Array::from(vt)) as ArrayRef,
        ],
    )?)
}

/// Lightweight summary of a motif run: everything a receipt needs, without
/// holding on to the matches.
#[derive(Debug, Clone)]
pub struct MotifSummary {
    pub tri_state: TriState,
    pub incomplete: Option<Incomplete>,
    pub stats: TraversalStats,
    pub valid_at: i64,
    pub limit: usize,
}

impl MotifSummary {
    pub fn is_complete(&self) -> bool {
        self.tri_state != TriState::Incomplete
    }

    /// Build a receipt. `query_id` is the caller's correlation id.
    pub fn receipt(&self, query_id: impl Into<String>) -> QueryReceipt {
        let mut receipt = QueryReceipt::new(query_id, self.tri_state)
            .with_as_of_applied(self.valid_at)
            .with_stats(self.stats.clone().into());
        receipt.incomplete = self.incomplete.clone();
        receipt
    }
}

impl MotifOutcome {
    pub fn summary(&self) -> MotifSummary {
        MotifSummary {
            tri_state: self.tri_state(),
            incomplete: self.incomplete(),
            stats: self.stats.clone(),
            valid_at: self.valid_at,
            limit: self.limit,
        }
    }
}

/// One row: whether `gtv_motif_matches` is a complete answer, and what to do
/// about it when it is not.
///
/// This is the SQL-visible form of [`Incomplete`]. It is always exactly one
/// row, so `SELECT tri_state FROM gtv_motif_receipt` is a cheap assertion in a
/// test or a notebook, and a join against it cannot fan out.
pub fn receipt_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("tri_state", DataType::Utf8, false),
        Field::new("incomplete_reason", DataType::Utf8, true),
        Field::new("hop", DataType::UInt64, false),
        Field::new("observed", DataType::UInt64, true),
        Field::new("cap", DataType::UInt64, true),
        Field::new("rows", DataType::UInt64, false),
        Field::new("edges_scanned", DataType::UInt64, false),
        Field::new("nodes_visited", DataType::UInt64, false),
    ]))
}

pub fn build_receipt_batch(summary: &MotifSummary) -> Result<RecordBatch> {
    let (reason, hop, observed, cap) = match &summary.incomplete {
        Some(i) => (
            Some(i.reason.as_str()),
            i.hop as u64,
            i.observed,
            i.limit,
        ),
        None => (None, 0, None, None),
    };
    Ok(RecordBatch::try_new(
        receipt_schema(),
        vec![
            Arc::new(StringArray::from(vec![match summary.tri_state {
                TriState::Found => "found",
                TriState::NoMatch => "no_match",
                TriState::Incomplete => "incomplete",
            }])) as ArrayRef,
            Arc::new(StringArray::from(vec![reason])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![hop])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![observed])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![cap])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![summary.stats.rows])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![summary.stats.edges_scanned])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![summary.stats.nodes_visited])) as ArrayRef,
        ],
    )?)
}

/// Materialize the match / edge / receipt tables from a single kernel run.
pub struct MotifBatches {
    pub matches: MotifMatchTable,
    pub edges: MotifEdgeTable,
    pub receipt: MotifReceiptTable,
    pub summary: MotifSummary,
}

pub fn build_motif_tables(
    csr: &Arc<TemporalCSR>,
    spec: MotifSpec,
    seeds: &[u64],
) -> Result<MotifBatches> {
    let outcome = find_matches_bounded(csr.as_ref(), &spec, Some(seeds))?;
    let summary = outcome.summary();
    let match_batch = build_match_batch(&outcome.matches, spec.num_vars())?;
    let edge_batch = build_edge_batch(&outcome.matches)?;
    let receipt_batch = build_receipt_batch(&summary)?;
    Ok(MotifBatches {
        matches: MotifMatchTable {
            spec,
            seeds: seeds.to_vec(),
            batch: match_batch,
        },
        edges: MotifEdgeTable { batch: edge_batch },
        receipt: MotifReceiptTable {
            batch: receipt_batch,
        },
        summary,
    })
}

#[derive(Debug, Clone)]
pub struct MotifMatchTable {
    spec: MotifSpec,
    seeds: Vec<u64>,
    batch: RecordBatch,
}

impl MotifMatchTable {
    pub fn spec(&self) -> &MotifSpec {
        &self.spec
    }

    pub fn seeds(&self) -> &[u64] {
        &self.seeds
    }

    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }
}

#[derive(Debug, Clone)]
pub struct MotifEdgeTable {
    batch: RecordBatch,
}

impl MotifEdgeTable {
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }
}

/// One-row table reporting whether the match table is conclusive.
#[derive(Debug, Clone)]
pub struct MotifReceiptTable {
    batch: RecordBatch,
}

impl MotifReceiptTable {
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }
}

async fn scan_batch(
    schema: SchemaRef,
    batch: RecordBatch,
    state: &dyn Session,
    projection: Option<&Vec<usize>>,
    filters: &[Expr],
    limit: Option<usize>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mem = MemTable::try_new(schema, vec![vec![batch]])?;
    mem.scan(state, projection, filters, limit).await
}

#[async_trait::async_trait]
impl TableProvider for MotifMatchTable {
    fn schema(&self) -> SchemaRef {
        match_schema(self.spec.num_vars())
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        scan_batch(
            self.schema(),
            self.batch.clone(),
            state,
            projection,
            filters,
            limit,
        )
        .await
    }
}

#[async_trait::async_trait]
impl TableProvider for MotifEdgeTable {
    fn schema(&self) -> SchemaRef {
        edge_schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        scan_batch(
            self.schema(),
            self.batch.clone(),
            state,
            projection,
            filters,
            limit,
        )
        .await
    }
}

#[async_trait::async_trait]
impl TableProvider for MotifReceiptTable {
    fn schema(&self) -> SchemaRef {
        receipt_schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        scan_batch(
            self.schema(),
            self.batch.clone(),
            state,
            projection,
            filters,
            limit,
        )
        .await
    }
}

fn parse_shape(expr: Option<&Expr>) -> Result<MotifShape> {
    let expr = expr.ok_or_else(|| arg_err("motif(...)", "shape"))?;
    MotifShape::parse(&expr_to_string(expr)?)
}

fn parse_optional_limit(exprs: &[Expr], at: usize) -> Result<usize> {
    match exprs.get(at) {
        Some(e) => Ok(expr_to_u64(e)? as usize),
        None => Ok(DEFAULT_LIMIT),
    }
}

fn parse_optional_window(exprs: &[Expr], at: usize) -> Result<Option<i64>> {
    match exprs.get(at) {
        Some(e) => {
            let span = expr_to_i64(e)?;
            Ok(if span > 0 { Some(span) } else { None })
        }
        None => Ok(None),
    }
}

fn spec_from_args(
    exprs: &[Expr],
    shape_at: usize,
    time_at: usize,
    opts_at: usize,
    signature: &str,
) -> Result<MotifSpec> {
    let shape = parse_shape(exprs.get(shape_at))?;
    let valid_at = expr_to_i64(
        exprs
            .get(time_at)
            .ok_or_else(|| arg_err(signature, "valid_at"))?,
    )?;
    let mut spec =
        MotifSpec::new(shape, valid_at).with_limit(parse_optional_limit(exprs, opts_at)?);
    if let Some(span) = parse_optional_window(exprs, opts_at + 1)? {
        spec = spec.with_window(span);
    }
    Ok(spec)
}

fn emit(
    csr: &TemporalCSR,
    spec: &MotifSpec,
    seeds: Option<&[u64]>,
) -> Result<Arc<dyn TableProvider>> {
    let found = find_matches(csr, spec, seeds)?;
    let batch = build_match_batch(&found, spec.num_vars())?;
    let schema = batch.schema();
    Ok(Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))
}

/// `motif(shape, valid_at [, limit [, window]])` — scan all start nodes.
/// All arguments are literals, exactly like `neighbors()` / `khop()`.
#[derive(Debug)]
pub struct MotifTableFunction {
    csr: Arc<TemporalCSR>,
}

impl MotifTableFunction {
    pub fn new(csr: &TemporalCSR) -> Self {
        Self::from_arc(Arc::new(csr.clone()))
    }

    pub fn from_arc(csr: Arc<TemporalCSR>) -> Self {
        Self { csr }
    }
}

impl TableFunctionImpl for MotifTableFunction {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let exprs = args.exprs();
        let spec = spec_from_args(
            exprs,
            0,
            1,
            2,
            "motif(shape, valid_at [, limit [, window]])",
        )?;
        emit(&self.csr, &spec, None)
    }
}

/// `motif_from(shape, start, valid_at [, limit [, window]])` — anchor
/// variable 0 at one seed. This is the pointwise baseline for the batch table.
#[derive(Debug)]
pub struct MotifFromTableFunction {
    csr: Arc<TemporalCSR>,
}

impl MotifFromTableFunction {
    pub fn new(csr: &TemporalCSR) -> Self {
        Self::from_arc(Arc::new(csr.clone()))
    }

    pub fn from_arc(csr: Arc<TemporalCSR>) -> Self {
        Self { csr }
    }
}

impl TableFunctionImpl for MotifFromTableFunction {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let exprs = args.exprs();
        let signature = "motif_from(shape, start, valid_at [, limit [, window]])";
        let spec = spec_from_args(exprs, 0, 2, 3, signature)?;
        let start = expr_to_u64(exprs.get(1).ok_or_else(|| arg_err(signature, "start"))?)?;
        emit(&self.csr, &spec, Some(&[start]))
    }
}
