//! L1 acceptance tests: the catalog is the single authority, and SQL can see it.
//!
//! Three claims are checked:
//! 1. `gtv_tables()` / `gtv_snapshots()` / `gtv_files()` report what `FsCatalog`
//!    actually holds (table count, version count, exact file set).
//! 2. `sync_catalog()` publishes a table into SQL **from the snapshot file
//!    list**, so a table is queryable because the catalog says so.
//! 3. Without an attached catalog the functions fail loudly instead of
//!    returning an empty (and therefore easily misread) result.
//!
//! Data discipline: batches are built in memory, files live in a tempdir.

use std::sync::Arc;

use arrow::array::{Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use gtv_catalog::{
    CommitOp, CommitOptions, FileFormat, FsCatalog, NewFile, PartitionSpec, SnapshotId,
};
use gtv_engine::GtvContext;

fn tmp_root(label: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("gtv_l1_{}_{}", std::process::id(), label));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, false),
        Field::new("px", DataType::Int64, false),
        Field::new("ts", DataType::Int64, false),
    ]))
}

fn batch(syms: &[&str], px: &[i64], ts: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(syms.to_vec())),
            Arc::new(Int64Array::from(px.to_vec())),
            Arc::new(Int64Array::from(ts.to_vec())),
        ],
    )
    .unwrap()
}

/// Two append commits ⇒ two snapshots, two files.
fn seeded_catalog(root: &std::path::Path) -> (FsCatalog, SnapshotId, SnapshotId) {
    let cat = FsCatalog::open(root).unwrap();
    let table = cat
        .create_table("ticks", schema(), PartitionSpec::single())
        .unwrap();
    cat.set_event_time_column(table, Some("ts".into())).unwrap();

    let mut opts = CommitOptions::default();
    opts.event_time_column = Some("ts".into());
    cat.commit(
        table,
        CommitOp::Append,
        vec![NewFile::unpartitioned(batch(
            &["AAA", "BBB"],
            &[10, 20],
            &[1_000, 1_100],
        ))],
        &opts,
    )
    .unwrap();
    let s1 = cat.latest(table).unwrap().unwrap();

    cat.commit(
        table,
        CommitOp::Append,
        vec![NewFile::unpartitioned(batch(
            &["CCC", "DDD", "EEE"],
            &[30, 40, 50],
            &[2_000, 2_100, 2_200],
        ))],
        &opts,
    )
    .unwrap();
    let s2 = cat.latest(table).unwrap().unwrap();
    (cat, s1, s2)
}

#[tokio::test]
async fn tables_reports_the_catalog_truth() {
    let root = tmp_root("tables");
    let (_cat, s1, s2) = seeded_catalog(&root);

    let ctx = GtvContext::new();
    ctx.attach_catalog(&root).unwrap();

    let out = ctx
        .sql(
            "SELECT name, column_count, snapshot_count, file_count, row_count, \
                    event_time_min, event_time_max, latest_snapshot, event_time_column \
             FROM gtv_tables()",
        )
        .await
        .unwrap();
    assert_eq!(out.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    let b = &out[0];
    assert_eq!(b.num_columns(), 9);
    assert_eq!(
        b.column(0).as_any().downcast_ref::<StringArray>().unwrap().value(0),
        "ticks"
    );
    assert_eq!(
        b.column(1).as_any().downcast_ref::<arrow::array::UInt64Array>().unwrap().value(0),
        3,
        "schema has three columns"
    );
    assert_eq!(
        b.column(2).as_any().downcast_ref::<arrow::array::UInt64Array>().unwrap().value(0),
        2,
        "two committed snapshots"
    );
    assert_eq!(
        b.column(3).as_any().downcast_ref::<arrow::array::UInt64Array>().unwrap().value(0),
        2,
        "append keeps the previous file: 1 + 1"
    );
    assert_eq!(
        b.column(4).as_any().downcast_ref::<arrow::array::UInt64Array>().unwrap().value(0),
        5
    );
    let min = b.column(5).as_any().downcast_ref::<Int64Array>().unwrap().value(0);
    let max = b.column(6).as_any().downcast_ref::<Int64Array>().unwrap().value(0);
    assert_eq!((min, max), (1_000, 2_200), "event-time box rolls up from files");
    assert_eq!(
        b.column(7).as_any().downcast_ref::<StringArray>().unwrap().value(0),
        s2.to_string()
    );
    assert_eq!(
        b.column(8).as_any().downcast_ref::<StringArray>().unwrap().value(0),
        "ts"
    );
    let _ = std::fs::remove_dir_all(&root);
    assert_ne!(s1, s2);
}

#[tokio::test]
async fn snapshots_lists_the_version_log() {
    let root = tmp_root("snapshots");
    let (_cat, s1, s2) = seeded_catalog(&root);
    let ctx = GtvContext::new();
    ctx.attach_catalog(&root).unwrap();

    let out = ctx
        .sql("SELECT snapshot_id, op, file_count FROM gtv_snapshots() ORDER BY created_at")
        .await
        .unwrap();
    let ids: Vec<String> = (0..out[0].num_rows())
        .map(|i| {
            out[0]
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(i)
                .to_string()
        })
        .collect();
    assert_eq!(ids, vec![s1.to_string(), s2.to_string()]);

    let ops: Vec<String> = (0..out[0].num_rows())
        .map(|i| {
            out[0]
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(i)
                .to_string()
        })
        .collect();
    assert_eq!(ops, vec!["append", "append"]);
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn files_pins_an_exact_version() {
    let root = tmp_root("files");
    let (_cat, s1, s2) = seeded_catalog(&root);
    let ctx = GtvContext::new();
    ctx.attach_catalog(&root).unwrap();

    // Latest by default: the append kept snapshot 1's file, so two files.
    let out = ctx.sql("SELECT count(*) FROM gtv_files('ticks')").await.unwrap();
    assert_eq!(
        out[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        2
    );
    // Pinned to the first snapshot: exactly one file.
    let q = format!("SELECT count(*) FROM gtv_files('ticks', '{s1}')");
    let out = ctx.sql(&q).await.unwrap();
    assert_eq!(
        out[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        1
    );
    // Per-file event-time boxes are present, so file pruning has inputs.
    let out = ctx
        .sql("SELECT min(event_time_min), max(event_time_max), bool_and(managed) FROM gtv_files('ticks')")
        .await
        .unwrap();
    assert_eq!(
        out[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        1_000
    );
    assert_eq!(
        out[0].column(1).as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        2_200
    );
    let _ = s2;
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test(flavor = "multi_thread")]
async fn sync_catalog_makes_the_catalog_the_only_source_of_truth() {
    let root = tmp_root("sync");
    let (_cat, _s1, _s2) = seeded_catalog(&root);

    let ctx = GtvContext::new();
    ctx.attach_catalog(&root).unwrap();
    assert_eq!(ctx.sync_catalog().unwrap(), 1, "one table published");

    // The data came from Parquet files referenced by the snapshot, never from
    // a register_batches call.
    let out = ctx.sql("SELECT count(*) FROM ticks").await.unwrap();
    assert_eq!(
        out[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        5
    );
    let out = ctx
        .sql("SELECT min(ts), max(ts) FROM ticks")
        .await
        .unwrap();
    assert_eq!(
        out[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        1_000
    );

    // A second sync is idempotent (re-register replaces).
    assert_eq!(ctx.sync_catalog().unwrap(), 1);
    let out = ctx.sql("SELECT count(*) FROM ticks").await.unwrap();
    assert_eq!(
        out[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        5
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn missing_catalog_fails_loudly() {
    let ctx = GtvContext::new();
    let err = ctx.sql("SELECT * FROM gtv_tables()").await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("no catalog attached"),
        "unhelpful error: {msg}"
    );
    assert!(ctx.sync_catalog().is_err());
    assert!(ctx.catalog().is_none());
}

#[tokio::test]
async fn csv_registry_tables_are_reported_with_their_format() {
    let root = tmp_root("csv");
    let cat = FsCatalog::open(&root).unwrap();
    let csv = root.join("ext.csv");
    std::fs::write(&csv, "sym,px\nAAA,1\nBBB,2\n").unwrap();
    cat.register_external("ext", csv.to_str().unwrap(), FileFormat::Csv)
        .unwrap();

    let ctx = GtvContext::new();
    ctx.attach_catalog(&root).unwrap();
    let out = ctx
        .sql("SELECT format, managed, path FROM gtv_files('ext')")
        .await
        .unwrap();
    let fmt = out[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .value(0)
        .to_string();
    assert_eq!(fmt, "csv");
    let managed = out[0]
        .column(1)
        .as_any()
        .downcast_ref::<arrow::array::BooleanArray>()
        .unwrap()
        .value(0);
    assert!(!managed, "external files are references, not catalog-owned");
    let _ = std::fs::remove_dir_all(&root);
}
