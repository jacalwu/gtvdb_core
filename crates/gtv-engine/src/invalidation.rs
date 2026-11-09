//! M1: `gtv_invalidations()` — the commit log as a read-only SQL projection.
//!
//! There is no event bus, no delivery state and no replay queue. The event
//! **is** the published `SnapshotDelta`: the UDTF reads `manifest.json` and
//! the delta files it references, so an invalidation becomes visible exactly
//! when the manifest write becomes durable (review R4 — no window in which a
//! crash can leave derived state silently stale). Replay is naturally
//! idempotent by `commit_id`.
//!
//! | call | answers |
//! | :-- | :-- |
//! | `gtv_invalidations()` | every published invalidation, oldest first |
//! | `gtv_invalidations(c)` | the same, filtered to `commit_id >= c` (cursor) |
//!
//! Deliberately **not cached**, for the same reason the L1 catalog views are
//! not: a stale metadata answer is worse than a slow one, and the delta files
//! are tiny fixed-width records.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use arrow::array::{ArrayRef, Int64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::{TableFunctionArgs, TableFunctionImpl};
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::error::{DataFusionError, Result};

use crate::expr_util::expr_to_u64;

/// Root of the `SegmentStore` whose commit log is exposed. Shared so that
/// `attach_delta_log()` on a live context is immediately visible to SQL.
pub type InvalidationSlot = Arc<RwLock<Option<PathBuf>>>;

pub fn empty_invalidation_slot() -> InvalidationSlot {
    Arc::new(RwLock::new(None))
}

fn exec_err(msg: impl Into<String>) -> DataFusionError {
    DataFusionError::Execution(msg.into())
}

fn plan_err(msg: impl Into<String>) -> DataFusionError {
    DataFusionError::Plan(msg.into())
}

/// `gtv_invalidations([cursor])` — one row per published invalidation record.
#[derive(Debug)]
pub struct InvalidationsTableFunction {
    slot: InvalidationSlot,
}

impl InvalidationsTableFunction {
    pub fn new(slot: InvalidationSlot) -> Self {
        Self { slot }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("commit_id", DataType::UInt64, false),
            Field::new("system_ts", DataType::Int64, false),
            Field::new("object_kind", DataType::Utf8, false),
            Field::new("object_id", DataType::UInt64, false),
            Field::new("affected_valid_from", DataType::Int64, false),
            Field::new("affected_valid_to", DataType::Int64, false),
            Field::new("reason", DataType::Utf8, false),
            Field::new("delta_id", DataType::UInt64, false),
            Field::new("state", DataType::Utf8, false),
            Field::new("file", DataType::Utf8, false),
        ]))
    }
}

impl TableFunctionImpl for InvalidationsTableFunction {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let exprs = args.exprs();
        if exprs.len() > 1 {
            return Err(plan_err(
                "gtv_invalidations takes at most one argument: commit_from",
            ));
        }
        let since = match exprs.first() {
            Some(e) => expr_to_u64(e)?,
            None => 0,
        };

        let root = {
            let guard = self
                .slot
                .read()
                .map_err(|_| exec_err("invalidation slot lock poisoned"))?;
            guard.clone().ok_or_else(|| {
                exec_err(
                    "gtv_invalidations: no delta log attached \
                     (call GtvContext::attach_delta_log or register_segment_store first)",
                )
            })?
        };
        let rows = gtv_storage::read_invalidation_log(&root, since)
            .map_err(|e| exec_err(format!("gtv_invalidations: {e}")))?;

        let schema = Self::schema();
        let n = rows.len();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from_iter_values(
                    rows.iter().map(|r| r.commit_id),
                )) as ArrayRef,
                Arc::new(Int64Array::from_iter_values(
                    rows.iter().map(|r| r.system_ts),
                )),
                Arc::new(StringArray::from_iter_values(
                    rows.iter().map(|r| r.object_kind.as_str()),
                )),
                Arc::new(UInt64Array::from_iter_values(
                    rows.iter().map(|r| r.object_id),
                )),
                Arc::new(Int64Array::from_iter_values(
                    rows.iter().map(|r| r.affected_valid_from),
                )),
                Arc::new(Int64Array::from_iter_values(
                    rows.iter().map(|r| r.affected_valid_to),
                )),
                Arc::new(StringArray::from_iter_values(
                    rows.iter().map(|r| r.reason.as_str()),
                )),
                Arc::new(UInt64Array::from_iter_values(
                    rows.iter().map(|r| r.delta_id),
                )),
                Arc::new(StringArray::from_iter_values(
                    rows.iter().map(|r| r.state.as_str()),
                )),
                Arc::new(StringArray::from_iter_values(
                    rows.iter().map(|r| r.file.as_str()),
                )),
            ],
        )?;
        debug_assert_eq!(batch.num_rows(), n);
        Ok(Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))
    }
}
