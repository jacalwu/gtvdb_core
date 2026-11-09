//! M2 SQL acceptance: L2 aggregates over a folded snapshot, and persisted
//! observation versions replayed into `as_of()`.
//!
//! The point of `gtv_active_edge_count` / `gtv_range_agg` is that a corrected
//! interval splits one logical edge into several physical rows; a naive
//! `sum(edge_count)` would double count. All data is in a tempdir.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, TimestampNanosecondArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use gtv_engine::GtvContext;
use gtv_storage::delta::{Correction, EdgeKey, HolePolicy, RemoveReason};
use gtv_storage::{EdgeRecord, SegmentStore};

const NODE_COUNT: usize = 8;
const KEY: EdgeKey = EdgeKey {
    src: 1,
    dst: 2,
    edge_type: 7,
};

fn tmp_root(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("gtv_m2sql_{}_{}", std::process::id(), label));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn observation_batch(value: f64) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("bucket", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new(
            "business_valid_from",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new(
            "business_valid_to",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![0])),
            Arc::new(Float64Array::from(vec![value])),
            Arc::new(TimestampNanosecondArray::from(vec![0])),
            Arc::new(TimestampNanosecondArray::from(vec![1_000])),
        ],
    )
    .unwrap()
}

/// One edge split by a logical delete of `[100,200)` into two fragments.
fn seeded(root: &PathBuf) -> SegmentStore {
    let mut s = SegmentStore::open(root, NODE_COUNT).unwrap();
    s.ingest(7, 3, &[EdgeRecord::new(1, 2, 0, 1_000, 7, 10)])
        .unwrap();
    s.flush_at(10).unwrap();
    s.correct_edge(
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
    s
}

fn u64_col(b: &RecordBatch, i: usize) -> Vec<u64> {
    b.column(i)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .values()
        .to_vec()
}

#[tokio::test]
async fn sql_aggregates_count_logical_edges_not_fragments() {
    let root = tmp_root("aggregates");
    let store = seeded(&root);
    let ctx = GtvContext::new();
    ctx.register_segment_store(&store, "t").unwrap();
    let view = ctx.aggregate_view().unwrap();
    assert_eq!(view.system_ts, store.system_ts());
    assert_eq!(view.rows_before_dedup, 2);

    let out = ctx
        .sql(
            "SELECT active_edge_count, intervals, rows_before_dedup \
             FROM gtv_active_edge_count(150)",
        )
        .await
        .unwrap();
    assert_eq!(out[0].num_rows(), 1);
    // 150 is inside the deleted hole: 0 active edges, but two physical rows
    // and two merged intervals remain in the projection.
    assert_eq!(u64_col(&out[0], 0), vec![0]);
    assert_eq!(u64_col(&out[0], 1), vec![2]);
    assert_eq!(u64_col(&out[0], 2), vec![2]);

    for (t, expected) in [(50, 1u64), (150, 0), (250, 1), (1_000, 0)] {
        let b = ctx
            .sql(&format!(
                "SELECT active_edge_count FROM gtv_active_edge_count({t})"
            ))
            .await
            .unwrap();
        assert_eq!(u64_col(&b[0], 0), vec![expected], "business_t={t}");
    }

    // Four buckets over the split edge: every bucket sees ONE logical edge,
    // including [0,250) which touches both fragments.
    let b = ctx
        .sql("SELECT bucket_start, active_edges FROM gtv_range_agg(0, 1000, 250) ORDER BY bucket_start")
        .await
        .unwrap();
    let starts: Vec<i64> = b[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();
    assert_eq!(starts, vec![0, 250, 500, 750]);
    assert_eq!(u64_col(&b[0], 1), vec![1, 1, 1, 1]);

    let whole = ctx
        .sql("SELECT active_edges FROM gtv_range_agg(0, 1000, 1000)")
        .await
        .unwrap();
    assert_eq!(u64_col(&whole[0], 0), vec![1]);
}

#[tokio::test]
async fn aggregate_view_is_explicit_and_system_time_is_guarded() {
    let root = tmp_root("guard");
    let store = seeded(&root);
    let ctx = GtvContext::new();
    // Without an attached view the call fails loudly rather than returning 0.
    assert!(ctx
        .sql("SELECT * FROM gtv_active_edge_count(50)")
        .await
        .is_err());

    ctx.register_segment_store(&store, "t").unwrap();
    let head = store.system_ts();
    assert!(ctx
        .sql(&format!(
            "SELECT * FROM gtv_active_edge_count(50, {head})"
        ))
        .await
        .is_ok());
    // A view covers exactly one snapshot; asking for another is a plan error.
    assert!(ctx
        .sql(&format!(
            "SELECT * FROM gtv_active_edge_count(50, {})",
            head + 1
        ))
        .await
        .is_err());
    assert!(ctx
        .sql("SELECT * FROM gtv_range_agg(0, 10, 0)")
        .await
        .is_err());
}

#[tokio::test]
async fn observation_versions_replay_into_as_of_after_registration() {
    let root = tmp_root("asof");
    let mut store = SegmentStore::open(&root, NODE_COUNT).unwrap();
    store
        .publish_observation("cpi", 10, vec![observation_batch(1.5)])
        .unwrap();
    store
        .publish_observation("cpi", 20, vec![observation_batch(2.5)])
        .unwrap();

    // A brand-new session context has an empty bitemporal registry; the single
    // registration call is what rehydrates the published versions.
    let ctx = GtvContext::new();
    assert!(ctx.sql("SELECT value FROM as_of('cpi', 50, 25)").await.is_err());
    ctx.register_segment_store(&store, "t").unwrap();

    for (system_ts, expected) in [(15, 1.5), (25, 2.5)] {
        let out = ctx
            .sql(&format!("SELECT value FROM as_of('cpi', 50, {system_ts})"))
            .await
            .unwrap();
        let values = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(values.value(0), expected, "system_ts={system_ts}");
    }
}
