//! Graph traversal exposed to SQL as a DataFusion table function.
//!
//! `neighbors(src, valid_at)` returns the temporal neighbors of `src` that are
//! active at `valid_at` (nanoseconds), with the same column shape as the
//! registered `edges` table (temporal columns as `Int64` nanoseconds).

use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch, UInt16Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::{TableFunctionArgs, TableFunctionImpl};
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::error::{DataFusionError, Result};
use gtv_core::{Sel, TemporalCSR, TraversalBudget, UNBOUND_SNAPSHOT};

use crate::expr_util::{expr_to_i64, expr_to_u64};

/// Table function over a snapshot of the temporal CSR.
#[derive(Debug)]
pub struct NeighborsTableFunction {
    csr: Arc<TemporalCSR>,
}

impl NeighborsTableFunction {
    pub fn new(csr: &TemporalCSR) -> Self {
        Self::from_arc(Arc::new(csr.clone()))
    }

    /// Share one graph image with the other pointwise UDTF. `register_neighbors`
    /// used to clone the whole CSR twice (once per table function); this keeps
    /// a single `Arc` while leaving the public `new(&TemporalCSR)` unchanged.
    pub fn from_arc(csr: Arc<TemporalCSR>) -> Self {
        Self { csr }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::UInt64, false),
            Field::new("dst", DataType::UInt64, false),
            Field::new("edge_type", DataType::UInt16, false),
            Field::new("valid_from", DataType::Int64, false),
            Field::new("valid_to", DataType::Int64, false),
        ]))
    }
}

impl TableFunctionImpl for NeighborsTableFunction {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let exprs = args.exprs();
        let src = expr_to_u64(exprs.first().ok_or_else(|| {
            DataFusionError::Execution("neighbors(src, valid_at): missing `src`".into())
        })?)?;
        let valid_at = expr_to_i64(exprs.get(1).ok_or_else(|| {
            DataFusionError::Execution("neighbors(src, valid_at): missing `valid_at`".into())
        })?)?;

        let mut src_out = Vec::new();
        let mut dst_out = Vec::new();
        let mut et_out = Vec::new();
        let mut vf_out = Vec::new();
        let mut vt_out = Vec::new();
        let neighbors = self
            .csr
            .neighbors(src, valid_at)
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;
        for nb in neighbors {
            src_out.push(src);
            dst_out.push(nb.dst);
            et_out.push(nb.edge_type);
            vf_out.push(nb.valid_from);
            vt_out.push(nb.valid_to);
        }

        let batch = RecordBatch::try_new(
            Self::schema(),
            vec![
                Arc::new(UInt64Array::from(src_out)) as ArrayRef,
                Arc::new(UInt64Array::from(dst_out)) as ArrayRef,
                Arc::new(UInt16Array::from(et_out)) as ArrayRef,
                Arc::new(Int64Array::from(vf_out)) as ArrayRef,
                Arc::new(Int64Array::from(vt_out)) as ArrayRef,
            ],
        )?;

        Ok(Arc::new(MemTable::try_new(
            Self::schema(),
            vec![vec![batch]],
        )?))
    }
}

/// `khop(src, k, valid_at [, max_hops, max_edges])` — resource-bounded k-hop
/// BFS over the temporal CSR (B1-3). Emits one row per reached node with the
/// hop at which it was first reached.
#[derive(Debug)]
pub struct KhopTableFunction {
    csr: Arc<TemporalCSR>,
}

impl KhopTableFunction {
    pub fn new(csr: &TemporalCSR) -> Self {
        Self::from_arc(Arc::new(csr.clone()))
    }

    /// See `NeighborsTableFunction::from_arc`.
    pub fn from_arc(csr: Arc<TemporalCSR>) -> Self {
        Self { csr }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("hop", DataType::UInt64, false),
            Field::new("dst", DataType::UInt64, false),
        ]))
    }
}

impl TableFunctionImpl for KhopTableFunction {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let exprs = args.exprs();
        let src = expr_to_u64(exprs.first().ok_or_else(|| {
            DataFusionError::Execution("khop(src, k, valid_at): missing `src`".into())
        })?)?;
        let k = expr_to_i64(exprs.get(1).ok_or_else(|| {
            DataFusionError::Execution("khop(src, k, valid_at): missing `k`".into())
        })?)? as usize;
        let valid_at = expr_to_i64(exprs.get(2).ok_or_else(|| {
            DataFusionError::Execution("khop(src, k, valid_at): missing `valid_at`".into())
        })?)?;

        let mut budget = TraversalBudget::unlimited();
        if let Some(e) = exprs.get(3) {
            budget.max_hops = expr_to_i64(e)?.max(0) as usize;
        }
        if let Some(e) = exprs.get(4) {
            budget.max_edges = expr_to_i64(e)?.max(0) as u64;
        }

        // In SQL `max_hops` is an optional cap on the depth, and `k` is the
        // depth actually asked for: clamp here so a query never comes back
        // labelled "partial" for a bound it chose itself.
        let k = k.min(budget.max_hops);

        // A SQL scalar seed becomes a one-element candidate set, so the k-hop
        // entry point has exactly one input type whether the seed comes from a
        // literal or from a vector KNN.
        let seeds = Sel::from_ids(
            UNBOUND_SNAPSHOT,
            self.csr.node_count() as u64,
            std::iter::once(src),
        )
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        let result = self
            .csr
            .khop_bounded(&seeds, k, valid_at, &budget, None, None)
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        // SQL has no channel for "partial result", and an empty rowset would be
        // read as "nothing reachable" — the one conclusion a capped traversal
        // cannot support. Fail loudly instead; Rust callers use the same
        // `truncation` field to report three-state outcomes.
        if let Some(t) = &result.truncation {
            return Err(DataFusionError::Execution(format!(
                "khop stopped early by {} (limit {}, observed {}): \
                 result is partial, raise the budget or add a predicate",
                t.reason.as_str(),
                t.limit,
                t.observed
            )));
        }

        let mut hops: Vec<u64> = Vec::new();
        let mut dst: Vec<u64> = Vec::new();
        for (i, f) in result.frontiers.iter().enumerate() {
            for &d in f.values().as_ref() {
                hops.push((i + 1) as u64);
                dst.push(d);
            }
        }
        let batch = RecordBatch::try_new(
            Self::schema(),
            vec![
                Arc::new(UInt64Array::from(hops)) as ArrayRef,
                Arc::new(UInt64Array::from(dst)) as ArrayRef,
            ],
        )?;
        Ok(Arc::new(MemTable::try_new(
            Self::schema(),
            vec![vec![batch]],
        )?))
    }
}
