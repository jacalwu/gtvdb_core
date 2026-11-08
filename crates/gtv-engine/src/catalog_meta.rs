//! L1: one authoritative metadata surface, in SQL.
//!
//! Before this module the catalog existed twice: `FsCatalog` on disk (the truth)
//! and the DataFusion session catalog (whatever a caller happened to register).
//! Nothing could ask "which tables exist, what are their snapshots, which files
//! does version N consist of?" without guessing from side channels.
//!
//! This module closes that gap with three table functions that read `FsCatalog`
//! and nothing else:
//!
//! | function | answers |
//! | :-- | :-- |
//! | `gtv_tables()` | which tables exist, their schema version, partitioning, latest snapshot, row/file counts |
//! | `gtv_snapshots()` | the immutable version log of every table (parent, op, file count, summary) |
//! | `gtv_files('t' [, 'snap'])` | exactly which data files a version consists of, with event-time boxes |
//!
//! Deliberately **not** cached: every call re-reads the catalog. A stale metadata
//! cache is worse than a slow one, and the files are tiny (a few KB per table).
//!
//! Naming follows MongoDB's `CollectionCatalog` principle "one authority, many
//! read-only views" — but the storage stays JSON-on-disk, not a B-tree.

use std::sync::{Arc, RwLock};

use arrow::array::{
    ArrayRef, BooleanArray, Int64Array, StringArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::{TableFunctionArgs, TableFunctionImpl};
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::error::{DataFusionError, Result};
use gtv_catalog::{CommitOp, DataFile, FsCatalog, Snapshot, TableMeta};

use crate::expr_util::expr_to_string;

/// The attached catalog, if any. Shared by every L1 table function so
/// `attach_catalog()` on a live context is immediately visible to SQL.
pub type CatalogSlot = Arc<RwLock<Option<FsCatalog>>>;

pub fn empty_slot() -> CatalogSlot {
    Arc::new(RwLock::new(None))
}

fn exec_err(msg: impl Into<String>) -> DataFusionError {
    DataFusionError::Execution(msg.into())
}

fn plan_err(msg: impl Into<String>) -> DataFusionError {
    DataFusionError::Plan(msg.into())
}

/// Resolve the attached catalog or fail with an actionable message.
fn require(slot: &CatalogSlot, who: &str) -> Result<FsCatalog> {
    let guard = slot
        .read()
        .map_err(|_| exec_err("catalog slot lock poisoned"))?;
    guard
        .clone()
        .ok_or_else(|| exec_err(format!("{who}: no catalog attached (call GtvContext::attach_catalog first)")))
}

fn mem(schema: SchemaRef, batch: RecordBatch) -> Result<Arc<dyn TableProvider>> {
    Ok(Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))
}

fn op_name(op: CommitOp) -> &'static str {
    match op {
        CommitOp::Append => "append",
        CommitOp::Overwrite => "overwrite",
        CommitOp::Delete => "delete",
    }
}

fn partition_labels(file: &DataFile, spec: &gtv_catalog::PartitionSpec) -> String {
    if file.partition.is_empty() {
        return String::new();
    }
    file.partition
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let name = spec
                .columns
                .get(i)
                .map(|c| c.name.as_str())
                .unwrap_or("p");
            let value = match v {
                gtv_catalog::PartitionValue::Null => "__null__".to_string(),
                gtv_catalog::PartitionValue::Str(s) => s.clone(),
                gtv_catalog::PartitionValue::Int(i) => i.to_string(),
            };
            format!("{name}={value}")
        })
        .collect::<Vec<_>>()
        .join("/")
}

// -- gtv_tables() -----------------------------------------------------------

/// `gtv_tables()` — one row per catalog table, sized by its latest snapshot.
#[derive(Debug)]
pub struct TablesTableFunction {
    slot: CatalogSlot,
}

impl TablesTableFunction {
    pub fn new(slot: CatalogSlot) -> Self {
        Self { slot }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("table_id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("created_at", DataType::Int64, false),
            Field::new("schema_version", DataType::UInt32, false),
            Field::new("column_count", DataType::UInt64, false),
            Field::new("partition_spec", DataType::Utf8, false),
            Field::new("spec_version", DataType::UInt32, false),
            Field::new("event_time_column", DataType::Utf8, true),
            Field::new("rename_count", DataType::UInt64, false),
            Field::new("latest_snapshot", DataType::Utf8, true),
            Field::new("snapshot_count", DataType::UInt64, false),
            Field::new("file_count", DataType::UInt64, false),
            Field::new("row_count", DataType::UInt64, false),
            Field::new("size_bytes", DataType::UInt64, false),
            Field::new("event_time_min", DataType::Int64, true),
            Field::new("event_time_max", DataType::Int64, true),
        ]))
    }
}

/// Everything `gtv_tables()` needs about one table, gathered once.
struct TableRow {
    meta: TableMeta,
    column_count: u64,
    latest: Option<String>,
    snapshot_count: u64,
    file_count: u64,
    row_count: u64,
    size_bytes: u64,
    event_time_min: Option<i64>,
    event_time_max: Option<i64>,
}

fn gather(cat: &FsCatalog, meta: TableMeta) -> std::result::Result<TableRow, String> {
    let map = |e: gtv_catalog::CatalogError| e.to_string();
    let column_count = cat
        .schema(
            meta.table_id,
            gtv_catalog::SchemaVersion(meta.schema_version),
        )
        .map(|s| s.fields().len() as u64)
        .unwrap_or(0);
    let latest = cat.latest(meta.table_id).map_err(map)?;
    let snapshot_count = cat.snapshots(meta.table_id).map_err(map)?.len() as u64;
    let (file_count, row_count, size_bytes, lo, hi) = match latest {
        Some(id) => {
            let files = cat.files(meta.table_id, id).map_err(map)?;
            let rows: u64 = files.iter().map(|f| f.row_count).sum();
            let bytes: u64 = files.iter().map(|f| f.size_bytes).sum();
            let lo = files.iter().map(|f| f.event_time_min).min();
            let hi = files.iter().map(|f| f.event_time_max).max();
            let unbounded = |v: i64| v == i64::MIN || v == i64::MAX;
            (
                files.len() as u64,
                rows,
                bytes,
                lo.filter(|v| !unbounded(*v)),
                hi.filter(|v| !unbounded(*v)),
            )
        }
        None => (0, 0, 0, None, None),
    };
    Ok(TableRow {
        column_count,
        latest: latest.map(|s| s.to_string()),
        snapshot_count,
        file_count,
        row_count,
        size_bytes,
        event_time_min: lo,
        event_time_max: hi,
        meta,
    })
}

impl TableFunctionImpl for TablesTableFunction {
    fn call_with_args(&self, _args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let cat = require(&self.slot, "gtv_tables()")?;
        let mut rows = Vec::new();
        for meta in cat
            .list_tables()
            .map_err(|e| exec_err(format!("gtv_tables(): {e}")))?
        {
            rows.push(gather(&cat, meta).map_err(exec_err)?);
        }
        rows.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));

        let schema = Self::schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from_iter(
                    rows.iter().map(|r| Some(r.meta.table_id.to_string())),
                )) as ArrayRef,
                Arc::new(StringArray::from_iter(
                    rows.iter().map(|r| Some(r.meta.name.as_str())),
                )),
                Arc::new(Int64Array::from_iter(rows.iter().map(|r| Some(r.meta.created_at)))),
                Arc::new(UInt32Array::from_iter(
                    rows.iter().map(|r| Some(r.meta.schema_version)),
                )),
                Arc::new(UInt64Array::from_iter(rows.iter().map(|r| Some(r.column_count)))),
                Arc::new(StringArray::from_iter(rows.iter().map(|r| {
                    Some(serde_json::to_string(&r.meta.spec).unwrap_or_default())
                }))),
                Arc::new(UInt32Array::from_iter(rows.iter().map(|r| Some(r.meta.spec.version)))),
                Arc::new(StringArray::from_iter(
                    rows.iter().map(|r| r.meta.event_time_column.clone()),
                )),
                Arc::new(UInt64Array::from_iter(
                    rows.iter().map(|r| Some(r.meta.renames.len() as u64)),
                )),
                Arc::new(StringArray::from_iter(rows.iter().map(|r| r.latest.clone()))),
                Arc::new(UInt64Array::from_iter(rows.iter().map(|r| Some(r.snapshot_count)))),
                Arc::new(UInt64Array::from_iter(rows.iter().map(|r| Some(r.file_count)))),
                Arc::new(UInt64Array::from_iter(rows.iter().map(|r| Some(r.row_count)))),
                Arc::new(UInt64Array::from_iter(rows.iter().map(|r| Some(r.size_bytes)))),
                Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.event_time_min))),
                Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.event_time_max))),
            ],
        )?;
        mem(schema, batch)
    }
}

// -- gtv_snapshots() --------------------------------------------------------

/// `gtv_snapshots()` — the append-only version log across all tables.
#[derive(Debug)]
pub struct SnapshotsTableFunction {
    slot: CatalogSlot,
}

impl SnapshotsTableFunction {
    pub fn new(slot: CatalogSlot) -> Self {
        Self { slot }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("table", DataType::Utf8, false),
            Field::new("table_id", DataType::Utf8, false),
            Field::new("snapshot_id", DataType::Utf8, false),
            Field::new("parent", DataType::Utf8, true),
            Field::new("op", DataType::Utf8, false),
            Field::new("schema_version", DataType::UInt32, false),
            Field::new("spec_version", DataType::UInt32, false),
            Field::new("file_count", DataType::UInt64, false),
            Field::new("created_at", DataType::Int64, false),
            Field::new("summary", DataType::Utf8, true),
        ]))
    }
}

impl TableFunctionImpl for SnapshotsTableFunction {
    fn call_with_args(&self, _args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let cat = require(&self.slot, "gtv_snapshots()")?;
        let mut rows: Vec<(String, Snapshot)> = Vec::new();
        for meta in cat
            .list_tables()
            .map_err(|e| exec_err(format!("gtv_snapshots(): {e}")))?
        {
            for snap in cat
                .snapshots(meta.table_id)
                .map_err(|e| exec_err(format!("gtv_snapshots(): {e}")))?
            {
                rows.push((meta.name.clone(), snap));
            }
        }
        // Oldest first, per table, so `ORDER BY created_at` is cheap downstream.
        rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.created_at.cmp(&b.1.created_at)));

        let schema = Self::schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from_iter(rows.iter().map(|(n, _)| Some(n.as_str())))) as ArrayRef,
                Arc::new(StringArray::from_iter(
                    rows.iter().map(|(_, s)| Some(s.table_id.to_string())),
                )),
                Arc::new(StringArray::from_iter(
                    rows.iter().map(|(_, s)| Some(s.snapshot_id.to_string())),
                )),
                Arc::new(StringArray::from_iter(
                    rows.iter().map(|(_, s)| s.parent.map(|p| p.to_string())),
                )),
                Arc::new(StringArray::from_iter(
                    rows.iter().map(|(_, s)| Some(op_name(s.op))),
                )),
                Arc::new(UInt32Array::from_iter(
                    rows.iter().map(|(_, s)| Some(s.schema_version.0)),
                )),
                Arc::new(UInt32Array::from_iter(
                    rows.iter().map(|(_, s)| Some(s.spec_version)),
                )),
                Arc::new(UInt64Array::from_iter(
                    rows.iter().map(|(_, s)| Some(s.files.len() as u64)),
                )),
                Arc::new(Int64Array::from_iter(rows.iter().map(|(_, s)| Some(s.created_at)))),
                Arc::new(StringArray::from_iter(rows.iter().map(|(_, s)| {
                    let summary = s.summary.to_string();
                    if summary == "{}" || summary == "null" {
                        None
                    } else {
                        Some(summary)
                    }
                }))),
            ],
        )?;
        mem(schema, batch)
    }
}

// -- gtv_files('table' [, 'snapshot']) --------------------------------------

/// `gtv_files('t' [, 'snap'])` — the exact file set of one version, with the
/// event-time box each file carries (L2 at file granularity).
#[derive(Debug)]
pub struct FilesTableFunction {
    slot: CatalogSlot,
}

impl FilesTableFunction {
    pub fn new(slot: CatalogSlot) -> Self {
        Self { slot }
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("table", DataType::Utf8, false),
            Field::new("snapshot_id", DataType::Utf8, false),
            Field::new("file_id", DataType::Utf8, false),
            Field::new("path", DataType::Utf8, false),
            Field::new("format", DataType::Utf8, false),
            Field::new("managed", DataType::Boolean, false),
            Field::new("row_count", DataType::UInt64, false),
            Field::new("size_bytes", DataType::UInt64, false),
            Field::new("event_time_min", DataType::Int64, true),
            Field::new("event_time_max", DataType::Int64, true),
            Field::new("schema_version", DataType::UInt32, false),
            Field::new("partition", DataType::Utf8, false),
            Field::new("commit_id", DataType::Utf8, false),
        ]))
    }
}

impl TableFunctionImpl for FilesTableFunction {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let exprs = args.exprs();
        let table_name = expr_to_string(
            exprs
                .first()
                .ok_or_else(|| plan_err("gtv_files requires argument `table`"))?,
        )?;
        let snapshot_arg = match exprs.get(1) {
            Some(e) => Some(expr_to_string(e)?),
            None => None,
        };

        let cat = require(&self.slot, "gtv_files()")?;
        let meta = cat
            .table(&table_name)
            .map_err(|e| exec_err(format!("gtv_files('{table_name}'): {e}")))?;
        let snap_id = match snapshot_arg {
            Some(s) => s.parse().map_err(|_| {
                plan_err(format!("gtv_files: `{s}` is not a snapshot id"))
            })?,
            None => cat
                .latest(meta.table_id)
                .map_err(|e| exec_err(format!("gtv_files: {e}")))?
                .ok_or_else(|| {
                    exec_err(format!("gtv_files('{table_name}'): table has no committed snapshot"))
                })?,
        };
        let files = cat
            .files(meta.table_id, snap_id)
            .map_err(|e| exec_err(format!("gtv_files('{table_name}'): {e}")))?;

        // A file whose box is `i64::MIN..i64::MAX` is simply "no event-time
        // column": report NULL rather than a sentinel that looks like a real
        // timestamp.
        let opt_time = |v: i64| {
            if v == i64::MIN || v == i64::MAX {
                None
            } else {
                Some(v)
            }
        };

        let schema = Self::schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from_iter(
                    files.iter().map(|_| Some(table_name.as_str())),
                )) as ArrayRef,
                Arc::new(StringArray::from_iter(
                    files.iter().map(|_| Some(snap_id.to_string())),
                )),
                Arc::new(StringArray::from_iter(
                    files.iter().map(|f| Some(f.file_id.to_string())),
                )),
                Arc::new(StringArray::from_iter(files.iter().map(|f| Some(f.path.as_str())))),
                Arc::new(StringArray::from_iter(files.iter().map(|f| {
                    Some(match f.format {
                        gtv_catalog::FileFormat::Parquet => "parquet",
                        gtv_catalog::FileFormat::Csv => "csv",
                    })
                }))),
                Arc::new(BooleanArray::from_iter(files.iter().map(|f| Some(f.managed)))),
                Arc::new(UInt64Array::from_iter(files.iter().map(|f| Some(f.row_count)))),
                Arc::new(UInt64Array::from_iter(files.iter().map(|f| Some(f.size_bytes)))),
                Arc::new(Int64Array::from_iter(
                    files.iter().map(|f| opt_time(f.event_time_min)),
                )),
                Arc::new(Int64Array::from_iter(
                    files.iter().map(|f| opt_time(f.event_time_max)),
                )),
                Arc::new(UInt32Array::from_iter(
                    files.iter().map(|f| Some(f.schema_version.0)),
                )),
                Arc::new(StringArray::from_iter(
                    files.iter().map(|f| Some(partition_labels(f, &meta.spec))),
                )),
                Arc::new(StringArray::from_iter(
                    files.iter().map(|f| Some(f.commit_id.to_string())),
                )),
            ],
        )?;
        mem(schema, batch)
    }
}
