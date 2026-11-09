//! M1 SQL acceptance: `gtv_invalidations()` is a read-only projection of the
//! manifest commit log, not an event bus.
//!
//! * It becomes visible exactly at the manifest publish (nothing can be read
//!   from an orphaned WAL mutation).
//! * `gtv_invalidations(cursor)` filters by `commit_id >= cursor`.
//! * `register_segment_store()` attaches the log automatically.
//! * Without an attached log the function fails loudly instead of returning
//!   an empty (easily misread) result.
//!
//! Data discipline: edges are built in memory; all files live in a tempdir.

use std::path::PathBuf;

use arrow::array::{Int64Array, StringArray, UInt64Array};
use gtv_engine::GtvContext;
use gtv_storage::delta::{Correction, EdgeKey, HolePolicy, RemoveReason};
use gtv_storage::{EdgeRecord, SegmentStore};

fn tmp_root(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("gtv_m1sql_{}_{}", std::process::id(), label));
    let _ = std::fs::remove_dir_all(&p);
    p
}

const NODE_COUNT: usize = 8;
const KEY: EdgeKey = EdgeKey {
    src: 1,
    dst: 2,
    edge_type: 7,
};

fn seeded(root: &PathBuf) -> (SegmentStore, u64, u64) {
    let mut s = SegmentStore::open(root, NODE_COUNT).unwrap();
    s.ingest(
        7,
        3,
        &[EdgeRecord::new(1, 2, 0, 1_000, 7, 10)],
    )
    .unwrap();
    s.flush_at(10).unwrap();
    let e = s
        .correct_edge(
            KEY,
            Correction {
                window: (100, 200),
                payload: None,
                reason: RemoveReason::ValidTimeCorrection,
                object_id: 17,
            },
            HolePolicy::AssertCovered,
            20,
        )
        .unwrap();
    let n = s.correct_node(3, 100, 200, 30).unwrap();
    (s, e.commit_id, n.commit_id)
}

#[tokio::test]
async fn invalidations_projection_is_cursorable_and_typed() {
    let root = tmp_root("projection");
    let (_store, edge_commit, node_commit) = seeded(&root);
    let ctx = GtvContext::new();
    ctx.attach_delta_log(&root).unwrap();

    let out = ctx
        .sql(
            "SELECT commit_id, system_ts, object_kind, object_id, reason, state, file \
             FROM gtv_invalidations() ORDER BY commit_id",
        )
        .await
        .unwrap();
    assert_eq!(out.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
    let b = &out[0];
    assert_eq!(b.num_columns(), 7);
    let commits = b
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(commits.value(0), edge_commit);
    assert_eq!(commits.value(1), node_commit);
    let kinds = b
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(kinds.value(0), "edge");
    assert_eq!(kinds.value(1), "node");
    let reasons = b
        .column(4)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(reasons.value(0), "valid_time_correction");
    assert_eq!(reasons.value(1), "lifetime_correction");
    let sys = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    assert!(sys.value(0) > 0 && sys.value(1) > sys.value(0));
    let files = b
        .column(6)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert!(files.value(0).ends_with(".gtvd"));

    // Cursor semantics: replay from the second commit returns only that row.
    let q = format!("SELECT count(*) AS n FROM gtv_invalidations({node_commit})");
    let cur = ctx.sql(&q).await.unwrap();
    assert_eq!(
        cur[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        1
    );
}

#[tokio::test]
async fn register_segment_store_attaches_the_log_and_hides_retired_segments() {
    let root = tmp_root("register");
    let (store, _, _) = seeded(&root);
    let head_segments = store
        .segments()
        .iter()
        .filter(|e| e.superseded_at.is_none())
        .count();

    let ctx = GtvContext::new();
    ctx.register_segment_store(&store, "m1").unwrap();
    assert_eq!(ctx.delta_log_root(), Some(root.clone()));

    let out = ctx
        .sql("SELECT count(*) AS n FROM gtv_invalidations()")
        .await
        .unwrap();
    assert_eq!(
        out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        2
    );
    let segs = ctx
        .sql("SELECT count(*) AS n FROM gtv_segments()")
        .await
        .unwrap();
    assert_eq!(
        segs[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        head_segments as i64,
        "retired segments must not appear twice in the head SQL view"
    );

    // The tight source-only box (M1 L2) is visible in SQL, not just Rust.
    let boxes = ctx
        .sql("SELECT src_id_min, src_id_max FROM gtv_segments()")
        .await
        .unwrap();
    let lo = boxes[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let hi = boxes[0]
        .column(1)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    for i in 0..boxes[0].num_rows() {
        assert_eq!(lo.value(i), 1);
        assert_eq!(hi.value(i), 1);
    }
}

#[tokio::test]
async fn missing_log_fails_loudly() {
    let ctx = GtvContext::new();
    let err = ctx.sql("SELECT * FROM gtv_invalidations()").await.unwrap_err();
    assert!(err.to_string().contains("no delta log attached"));
}
