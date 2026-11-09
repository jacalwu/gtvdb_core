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

use arrow::array::{ArrayRef, Int64Array, RecordBatch, UInt16Array, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::{Session, TableFunctionArgs, TableFunctionImpl, TableProvider};
use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::ExecutionPlan;
use gtv_core::TemporalCSR;
use gtv_pattern::{Match, Pattern, WindowedMotif};

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

/// Complete specification of one motif query.
#[derive(Debug, Clone, PartialEq)]
pub struct MotifSpec {
    pub shape: MotifShape,
    pub valid_at: i64,
    pub limit: usize,
    /// All matched edges must fall within this many event-time units.
    pub max_event_span: Option<i64>,
    /// Matched edge types, ordered by event time, must equal this sequence.
    pub sequence: Vec<u16>,
}

impl MotifSpec {
    pub fn new(shape: MotifShape, valid_at: i64) -> Self {
        Self {
            shape,
            valid_at,
            limit: DEFAULT_LIMIT,
            max_event_span: None,
            sequence: Vec::new(),
        }
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

    pub fn num_vars(&self) -> usize {
        self.shape.num_vars()
    }

    /// Window / sequence constraints deliberately bypass the optimized 3-cycle
    /// fast path in `find`; this is why the two branches below are separate.
    pub fn is_windowed(&self) -> bool {
        self.max_event_span.is_some() || !self.sequence.is_empty()
    }

    fn windowed_motif(&self) -> WindowedMotif {
        let mut motif = WindowedMotif::new(self.shape.pattern());
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
                    Some(m) => gtv_pattern::find_windowed_from(
                        csr,
                        m,
                        start,
                        spec.valid_at,
                        remaining,
                    ),
                    None => gtv_pattern::find_from(
                        csr,
                        &pattern,
                        start,
                        spec.valid_at,
                        remaining,
                    ),
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
pub fn edge_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("match_id", DataType::UInt64, false),
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
    let mut idx = Vec::with_capacity(rows);
    let mut src = Vec::with_capacity(rows);
    let mut dst = Vec::with_capacity(rows);
    let mut et = Vec::with_capacity(rows);
    let mut vf = Vec::with_capacity(rows);
    let mut vt = Vec::with_capacity(rows);

    for (i, m) in matches.iter().enumerate() {
        for (j, e) in m.edges.iter().enumerate() {
            ids.push(i as u64);
            idx.push(j as u32);
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
            Arc::new(UInt32Array::from(idx)) as ArrayRef,
            Arc::new(UInt64Array::from(src)) as ArrayRef,
            Arc::new(UInt64Array::from(dst)) as ArrayRef,
            Arc::new(UInt16Array::from(et)) as ArrayRef,
            Arc::new(Int64Array::from(vf)) as ArrayRef,
            Arc::new(Int64Array::from(vt)) as ArrayRef,
        ],
    )?)
}

/// Materialize both batch tables from a single kernel run.
pub struct MotifBatches {
    pub matches: MotifMatchTable,
    pub edges: MotifEdgeTable,
}

pub fn build_motif_tables(
    csr: &Arc<TemporalCSR>,
    spec: MotifSpec,
    seeds: &[u64],
) -> Result<MotifBatches> {
    let found = find_matches(csr.as_ref(), &spec, Some(seeds))?;
    let match_batch = build_match_batch(&found, spec.num_vars())?;
    let edge_batch = build_edge_batch(&found)?;
    Ok(MotifBatches {
        matches: MotifMatchTable {
            spec,
            seeds: seeds.to_vec(),
            batch: match_batch,
        },
        edges: MotifEdgeTable { batch: edge_batch },
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
    let mut spec = MotifSpec::new(shape, valid_at)
        .with_limit(parse_optional_limit(exprs, opts_at)?);
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
        let spec = spec_from_args(exprs, 0, 1, 2, "motif(shape, valid_at [, limit [, window]])")?;
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
