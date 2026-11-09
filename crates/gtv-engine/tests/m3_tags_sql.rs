//! M3 SQL acceptance: `gtv_tags()` is a read-only projection of the published
//! manifest, sharing the same atomic publish as `gtv_invalidations()`.
//!
//! * tags survive a store restart and are visible without any event bus;
//! * `register_segment_store()` attaches the projection automatically;
//! * without an attached store the function fails loudly;
//! * tags pin history against delayed GC, but are NOT branches (review R6).
//!
//! Data discipline: edges are built in memory; all files live in a tempdir.

use std::path::PathBuf;

use arrow::array::{Int64Array, StringArray};
use gtv_engine::GtvContext;
use gtv_storage::{EdgeRecord, SegmentStore};

fn tmp_root(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("gtv_m3sql_{}_{}", std::process::id(), label));
    let _ = std::fs::remove_dir_all(&p);
    p
}

#[tokio::test]
async fn tags_projection_is_sorted_typed_and_restart_safe() {
    let root = tmp_root("projection");
    {
        let mut s = SegmentStore::open(&root, 8).unwrap();
        s.ingest(7, 3, &[EdgeRecord::new(1, 2, 0, 1_000, 7, 10)])
            .unwrap();
        s.flush_at(10).unwrap();
        s.create_tag("z-last", 10).unwrap();
        s.create_tag("a-first", 10).unwrap();
    }
    // Reopen from disk: nothing depends on an in-memory tag map.
    let store = SegmentStore::open(&root, 8).unwrap();
    let ctx = GtvContext::new();
    ctx.register_segment_store(&store, "m3").unwrap();

    let out = ctx
        .sql("SELECT name, system_ts FROM gtv_tags() ORDER BY name")
        .await
        .unwrap();
    assert_eq!(out.len(), 1);
    let batch = &out[0];
    assert_eq!(batch.num_rows(), 2);
    let names = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let points = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(names.value(0), "a-first");
    assert_eq!(names.value(1), "z-last");
    assert_eq!(points.value(0), 10);
    assert_eq!(points.value(1), 10);
}

#[tokio::test]
async fn tags_projection_without_attached_store_fails_loudly() {
    let ctx = GtvContext::new();
    let err = ctx.sql("SELECT * FROM gtv_tags()").await.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("no store attached"), "unexpected error: {msg}");
}

#[tokio::test]
async fn tags_projection_rejects_arguments() {
    let root = tmp_root("args");
    let mut s = SegmentStore::open(&root, 8).unwrap();
    s.ingest(7, 3, &[EdgeRecord::new(1, 2, 0, 1_000, 7, 10)])
        .unwrap();
    s.flush_at(10).unwrap();
    s.create_tag("v1", 10).unwrap();
    let ctx = GtvContext::new();
    ctx.attach_delta_log(&root).unwrap();
    let err = ctx.sql("SELECT * FROM gtv_tags(1)").await.unwrap_err();
    assert!(
        err.to_string().contains("takes no arguments"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn tag_visibility_follows_the_atomic_manifest_publish() {
    let root = tmp_root("publish");
    let mut s = SegmentStore::open(&root, 8).unwrap();
    s.ingest(7, 3, &[EdgeRecord::new(1, 2, 0, 1_000, 7, 10)])
        .unwrap();
    s.flush_at(10).unwrap();
    let ctx = GtvContext::new();
    ctx.attach_delta_log(&root).unwrap();

    let before = ctx.sql("SELECT count(*) AS n FROM gtv_tags()").await.unwrap();
    assert_eq!(
        before[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        0
    );

    s.create_tag("pinned", 10).unwrap();
    let after = ctx.sql("SELECT count(*) AS n FROM gtv_tags()").await.unwrap();
    assert_eq!(
        after[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        1,
        "a newly published tag must be visible immediately, with no event bus"
    );

    s.drop_tag("pinned").unwrap();
    let dropped = ctx.sql("SELECT count(*) AS n FROM gtv_tags()").await.unwrap();
    assert_eq!(
        dropped[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        0
    );
}
