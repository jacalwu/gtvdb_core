//! Batched temporal-graph table providers (M6 perf line).
//!
//! # Why this exists
//!
//! [`crate::graph::NeighborsTableFunction`] and `KhopTableFunction` take only
//! scalar literals (see `expr_util.rs`: non-literals are rejected with
//! "arguments must be literals"). A workload that starts from a *set* of
//! entities — a batch of AML alerts, a CRM customer list, an LE obligor group —
//! therefore pays one DataFusion parse + plan + collect cycle **per seed**,
//! while the kernel's own loop costs nanoseconds per edge.
//!
//! These two `TableProvider`s expose the identical semantics with one planning
//! cycle and one `RecordBatch`:
//!
//! ```sql
//! SELECT * FROM gtv_batch_neighbors;      -- (src, dst, edge_type, valid_from, valid_to)
//! SELECT * FROM gtv_batch_khop;          -- (seed, hop, dst)
//! ```
//!
//! Both can be joined against ordinary SQL tables, which the literal-only
//! UDTFs cannot do.
//!
//! # What is deliberately *not* changed
//!
//! * `gtv-core/src/csr.rs` — `neighbors()` stays 2 comparisons + chunk zone map.
//! * `gtv-core/src/endpoint.rs`, `gtv-storage/src/segment.rs` — GTVSeg v1 column
//!   layout and `disk == memory == cache line` are untouched.
//! * Batch results are built by calling the **same** kernel entry points in a
//!   Rust loop, so correctness is comparable row-for-row with the pointwise
//!   UDTFs (see `tests/m6_batch_graph.rs`).
//!
//! The `RecordBatch` is materialized once at construction (eager). A batch of
//! seeds is a bounded, caller-supplied set, not an unbounded stream; building
//! once avoids recomputing on every `scan()` while `RecordBatch::clone()` is a
//! cheap Arc copy.

use std::fmt;
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, UInt16Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::{Session, TableProvider};
use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::ExecutionPlan;
use gtv_core::TemporalCSR;

/// Same column shape as `neighbors(src, valid_at)` so results can be compared
/// and unioned without casts.
pub fn neighbors_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("src", DataType::UInt64, false),
        Field::new("dst", DataType::UInt64, false),
        Field::new("edge_type", DataType::UInt16, false),
        Field::new("valid_from", DataType::Int64, false),
        Field::new("valid_to", DataType::Int64, false),
    ]))
}

/// `khop()` emits `(hop, dst)`; the batch form adds the originating seed so a
/// whole alert list can be joined back to its cases.
pub fn khop_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("seed", DataType::UInt64, false),
        Field::new("hop", DataType::UInt64, false),
        Field::new("dst", DataType::UInt64, false),
    ]))
}

fn kernel_err(e: impl fmt::Display) -> DataFusionError {
    DataFusionError::Execution(format!("temporal graph kernel: {e}"))
}

/// All active neighbours at `valid_at` for a batch of source nodes.
#[derive(Debug, Clone)]
pub struct BatchNeighborsTable {
    seeds: Vec<u64>,
    valid_at: i64,
    batch: RecordBatch,
}

impl BatchNeighborsTable {
    /// Build the batch once. `seeds` may be empty (produces an empty batch).
    /// The `Arc` is borrowed only for construction: callers can hand the same
    /// graph to both providers without cloning the whole CSR twice.
    pub fn new(
        csr: &Arc<TemporalCSR>,
        seeds: impl IntoIterator<Item = u64>,
        valid_at: i64,
    ) -> Result<Self> {
        let seeds: Vec<u64> = seeds.into_iter().collect();
        let batch = build_neighbors_batch(csr.as_ref(), &seeds, valid_at)?;
        Ok(Self { seeds, valid_at, batch })
    }

    pub fn seeds(&self) -> &[u64] {
        &self.seeds
    }

    pub fn valid_at(&self) -> i64 {
        self.valid_at
    }

    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }
}

/// One `khop()` result per seed, folded into a single three-column batch.
#[derive(Debug, Clone)]
pub struct BatchKhopTable {
    seeds: Vec<u64>,
    k: usize,
    valid_at: i64,
    batch: RecordBatch,
}

impl BatchKhopTable {
    pub fn new(
        csr: &Arc<TemporalCSR>,
        seeds: impl IntoIterator<Item = u64>,
        k: usize,
        valid_at: i64,
    ) -> Result<Self> {
        let seeds: Vec<u64> = seeds.into_iter().collect();
        let batch = build_khop_batch(csr.as_ref(), &seeds, k, valid_at)?;
        Ok(Self { seeds, k, valid_at, batch })
    }

    pub fn seeds(&self) -> &[u64] {
        &self.seeds
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn valid_at(&self) -> i64 {
        self.valid_at
    }

    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }
}

/// The batch kernel: identical predicate to `NeighborsTableFunction`, but the
/// loop lives in Rust instead of being repeated by N SQL statements.
fn build_neighbors_batch(
    csr: &TemporalCSR,
    seeds: &[u64],
    valid_at: i64,
) -> Result<RecordBatch> {
    let mut src_out: Vec<u64> = Vec::new();
    let mut dst_out: Vec<u64> = Vec::new();
    let mut et_out: Vec<u16> = Vec::new();
    let mut vf_out: Vec<i64> = Vec::new();
    let mut vt_out: Vec<i64> = Vec::new();

    for &s in seeds {
        let neighbors = csr.neighbors(s, valid_at).map_err(kernel_err)?;
        for nb in neighbors {
            src_out.push(s);
            dst_out.push(nb.dst);
            et_out.push(nb.edge_type);
            vf_out.push(nb.valid_from);
            vt_out.push(nb.valid_to);
        }
    }

    let schema = neighbors_schema();
    Ok(RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(src_out)),
            Arc::new(UInt64Array::from(dst_out)),
            Arc::new(UInt16Array::from(et_out)),
            Arc::new(Int64Array::from(vf_out)),
            Arc::new(Int64Array::from(vt_out)),
        ],
    )?)
}

/// Keeps per-seed semantics: `khop(seed, k, t)` is run for each seed, so the
/// union of the batch rows grouped by `seed` equals N pointwise UDTF calls
/// exactly (no cross-seed dedup, which would change the answer).
fn build_khop_batch(
    csr: &TemporalCSR,
    seeds: &[u64],
    k: usize,
    valid_at: i64,
) -> Result<RecordBatch> {
    let mut seed_out: Vec<u64> = Vec::new();
    let mut hop_out: Vec<u64> = Vec::new();
    let mut dst_out: Vec<u64> = Vec::new();

    for &s in seeds {
        let one = UInt64Array::from(vec![s]);
        let frontiers = csr.khop(&one, k, valid_at).map_err(kernel_err)?;
        for (i, frontier) in frontiers.iter().enumerate() {
            for &d in frontier.values().as_ref() {
                seed_out.push(s);
                hop_out.push((i + 1) as u64);
                dst_out.push(d);
            }
        }
    }

    Ok(RecordBatch::try_new(
        khop_schema(),
        vec![
            Arc::new(UInt64Array::from(seed_out)),
            Arc::new(UInt64Array::from(hop_out)),
            Arc::new(UInt64Array::from(dst_out)),
        ],
    )?)
}

/// Wrap a materialized batch in a `MemTable` inside `scan()` — this avoids
/// hand-writing an `ExecutionPlan` while still forwarding projection, filters
/// and limit to DataFusion's own memory source.
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
impl TableProvider for BatchNeighborsTable {
    fn schema(&self) -> SchemaRef {
        neighbors_schema()
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
impl TableProvider for BatchKhopTable {
    fn schema(&self) -> SchemaRef {
        khop_schema()
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
