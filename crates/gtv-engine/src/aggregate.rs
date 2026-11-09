//! M2: SQL surface for L2 time-range aggregates.
//!
//! | call | answers |
//! | :-- | :-- |
//! | `gtv_active_edge_count(t [, system_t])` | distinct logical edges active at business instant `t` |
//! | `gtv_range_agg(t1, t2, bucket [, system_t])` | one row per half-open bucket with distinct active edges |
//!
//! Both read an [`AggregateView`]: a folded, de-duplicated projection of one
//! system snapshot, synchronised explicitly by `register_segment_store()` (or
//! `attach_aggregate_view()`). An optional `system_t` argument is a **guard**,
//! not a second query axis — a view covers exactly one snapshot, so asking for
//! another one is a plan error instead of a silently stale answer.
//!
//! De-duplication is the whole point: an interval correction splits a logical
//! edge into three physical rows, and `SELECT sum(edge_count) FROM
//! gtv_segments()` would count it three times. The view merges touching
//! intervals and `range_agg` counts each `(src,dst,edge_type)` once per bucket.

use std::sync::{Arc, RwLock};

use arrow::array::{ArrayRef, Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::{TableFunctionArgs, TableFunctionImpl};
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::error::{DataFusionError, Result};
use gtv_storage::AggregateView;

use crate::expr_util::expr_to_i64;

pub type AggregateSlot = Arc<RwLock<Option<AggregateView>>>;

pub fn empty_aggregate_slot() -> AggregateSlot {
    Arc::new(RwLock::new(None))
}

fn plan_err(msg: impl Into<String>) -> DataFusionError {
    DataFusionError::Plan(msg.into())
}

fn view_of(slot: &AggregateSlot) -> Result<AggregateView> {
    slot.read()
        .map_err(|_| plan_err("aggregate slot lock poisoned"))?
        .clone()
        .ok_or_else(|| {
            plan_err(
                "no aggregate view attached (call GtvContext::register_segment_store \
                 or attach_aggregate_view first)",
            )
        })
}

/// Check the optional `system_t` argument against the view's snapshot.
fn check_system_arg(args: &TableFunctionArgs, at: usize, view: &AggregateView) -> Result<()> {
    let exprs = args.exprs();
    if let Some(e) = exprs.get(at) {
        let requested = expr_to_i64(e)?;
        if requested != view.system_ts {
            return Err(plan_err(format!(
                "aggregate view covers system_ts {} but query asked for {requested}; \
                 re-register the store to refresh the projection",
                view.system_ts
            )));
        }
    }
    Ok(())
}

fn require(exprs_len: usize, min: usize, usage: &str) -> Result<()> {
    if exprs_len < min {
        Err(plan_err(format!("{usage}: expected {min} arguments")))
    } else {
        Ok(())
    }
}

/// `gtv_active_edge_count(t [, system_t])` — one row.
#[derive(Debug)]
pub struct ActiveEdgeCountTableFunction {
    slot: AggregateSlot,
}

impl ActiveEdgeCountTableFunction {
    pub fn new(slot: AggregateSlot) -> Self {
        Self { slot }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("business_ts", DataType::Int64, false),
            Field::new("system_ts", DataType::Int64, false),
            Field::new("active_edge_count", DataType::UInt64, false),
            Field::new("intervals", DataType::UInt64, false),
            Field::new("rows_before_dedup", DataType::UInt64, false),
            Field::new("segments_total", DataType::UInt64, false),
        ]))
    }
}

impl TableFunctionImpl for ActiveEdgeCountTableFunction {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let usage = "gtv_active_edge_count(t [, system_t])";
        require(args.exprs().len(), 1, usage)?;
        if args.exprs().len() > 2 {
            return Err(plan_err(format!("{usage}: too many arguments")));
        }
        let t = expr_to_i64(&args.exprs()[0])?;
        let view = view_of(&self.slot)?;
        check_system_arg(&args, 1, &view)?;
        let count = view.active_edge_count(t);
        let schema = Self::schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![t])) as ArrayRef,
                Arc::new(Int64Array::from(vec![view.system_ts])),
                Arc::new(UInt64Array::from(vec![count])),
                Arc::new(UInt64Array::from(vec![view.intervals.len() as u64])),
                Arc::new(UInt64Array::from(vec![view.rows_before_dedup as u64])),
                Arc::new(UInt64Array::from(vec![view.segments_total as u64])),
            ],
        )?;
        Ok(Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))
    }
}

/// `gtv_range_agg(t1, t2, bucket [, system_t])` — one row per bucket.
#[derive(Debug)]
pub struct RangeAggTableFunction {
    slot: AggregateSlot,
}

impl RangeAggTableFunction {
    pub fn new(slot: AggregateSlot) -> Self {
        Self { slot }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("bucket_start", DataType::Int64, false),
            Field::new("bucket_end", DataType::Int64, false),
            Field::new("system_ts", DataType::Int64, false),
            Field::new("active_edges", DataType::UInt64, false),
        ]))
    }
}

impl TableFunctionImpl for RangeAggTableFunction {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let usage = "gtv_range_agg(t1, t2, bucket [, system_t])";
        require(args.exprs().len(), 3, usage)?;
        if args.exprs().len() > 4 {
            return Err(plan_err(format!("{usage}: too many arguments")));
        }
        let t1 = expr_to_i64(&args.exprs()[0])?;
        let t2 = expr_to_i64(&args.exprs()[1])?;
        let bucket = expr_to_i64(&args.exprs()[2])?;
        if bucket <= 0 || t1 >= t2 {
            return Err(plan_err(format!(
                "{usage}: require bucket > 0 and t1 < t2"
            )));
        }
        let view = view_of(&self.slot)?;
        check_system_arg(&args, 3, &view)?;
        let buckets = view.range_agg(t1, t2, bucket);
        let schema = Self::schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(
                    buckets.iter().map(|b| b.bucket_start),
                )) as ArrayRef,
                Arc::new(Int64Array::from_iter_values(
                    buckets.iter().map(|b| b.bucket_end),
                )),
                Arc::new(Int64Array::from(vec![view.system_ts; buckets.len()])),
                Arc::new(UInt64Array::from_iter_values(
                    buckets.iter().map(|b| b.active_edges),
                )),
            ],
        )?;
        Ok(Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))
    }
}
