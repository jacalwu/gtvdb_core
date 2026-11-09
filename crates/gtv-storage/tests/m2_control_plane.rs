//! M2 acceptance tests: two-phase background compaction, delayed GC,
//! persisted observation versions, source-box narrowing and L2 de-duplication.
//!
//! Acceptance target (`asymmetric-bitemporal-review.md` §7): the same
//! `(snapshot, business_t)` answer before and after compaction/GC, all
//! §23.8 crash points harmless, and no double counting of interval-split
//! logical edges. All files live in a tempdir outside the repository.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, RecordBatch, TimestampNanosecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use gtv_storage::delta::{Correction, EdgeKey, HolePolicy, RemoveReason};
use gtv_storage::{
    AggregateView, BitemporalStore, EdgeRecord, GcPolicy, SegmentStore,
};

const NODE_COUNT: usize = 8;
const KEY: EdgeKey = EdgeKey {
    src: 1,
    dst: 2,
    edge_type: 7,
};

fn temp_root(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("gtv_m2_{}_{}", std::process::id(), label));
    let _ = fs::remove_dir_all(&p);
    p
}

fn edge(src: u64, from: i64, to: i64, edge_type: u16, system_ts: i64) -> EdgeRecord {
    EdgeRecord::new(src, KEY.dst, from, to, edge_type, system_ts)
}

fn seeded_store(root: &Path) -> SegmentStore {
    let mut s = SegmentStore::open(root, NODE_COUNT).unwrap();
    s.ingest(7, 3, &[edge(1, 0, 1_000, KEY.edge_type, 10)])
        .unwrap();
    s.flush_at(10).unwrap();
    s
}

fn split_correction(s: &mut SegmentStore, system_ts: i64) {
    s.correct_edge(
        KEY,
        Correction {
            window: (150, 250),
            payload: None,
            reason: RemoveReason::LogicalDelete,
            object_id: 17,
        },
        HolePolicy::AssertCovered,
        system_ts,
    )
    .unwrap();
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

fn value_of(batches: &[RecordBatch]) -> f64 {
    batches[0]
        .column_by_name("value")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0)
}

fn assert_no_csv(root: &Path) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in fs::read_dir(&dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|x| x.to_str()) == Some("csv") {
                panic!("test wrote a CSV into the store root: {}", p.display());
            }
        }
    }
}

// -- 1. observation versions ------------------------------------------------

#[test]
fn observation_versions_survive_restart_and_feed_as_of() {
    let root = temp_root("observations");
    let mut s = seeded_store(&root);
    s.publish_observation("cpi", 10, vec![observation_batch(1.5)])
        .unwrap();
    s.publish_observation("cpi", 20, vec![observation_batch(2.5)])
        .unwrap();
    assert_eq!(s.observations().len(), 2);

    // A fresh handle sees exactly the published versions; the session store
    // is rehydrated from them, so `as_of` keeps both axes after a restart.
    let reopened = SegmentStore::open(&root, NODE_COUNT).unwrap();
    assert_eq!(value_of(&reopened.observation_batches("cpi", 15).unwrap()), 1.5);
    assert_eq!(value_of(&reopened.observation_batches("cpi", 25).unwrap()), 2.5);
    assert!(reopened.observation_batches("cpi", 5).is_err());

    let mut bt = BitemporalStore::new();
    assert_eq!(reopened.load_observations_into(&mut bt).unwrap(), 2);
    assert_eq!(value_of(&bt.as_of("cpi", 50, 15).unwrap()), 1.5);
    assert_eq!(value_of(&bt.as_of("cpi", 50, 25).unwrap()), 2.5);
    // Idempotent replay, like a repeated register_segment_store().
    assert_eq!(reopened.load_observations_into(&mut bt).unwrap(), 2);
    assert_eq!(bt.timestamps("cpi").unwrap(), vec![10, 20]);
    assert_no_csv(&root);
}

// -- 2. two-phase compaction ------------------------------------------------

#[test]
fn worker_compaction_matches_synchronous_and_survives_restart() {
    let root = temp_root("worker");
    let mut s = seeded_store(&root);
    split_correction(&mut s, 20);
    let expected = s.visible_edges_at(20).unwrap();
    assert_eq!(expected.len(), 2);

    // Phase 1 on a worker thread: no store lock, no disk write. The main
    // thread can still answer version queries while the worker runs.
    let handle = s.spawn_compaction(30).unwrap();
    assert_eq!(s.visible_edges_at(20).unwrap(), expected);
    let plan = handle.join().unwrap().unwrap();
    assert_eq!(plan.base_head, 20);
    assert_eq!(plan.rows, 2);
    assert!(!plan.segments.is_empty());
    assert_eq!(s.active_readers(), 0, "worker scope must be released");

    let report = s.apply_compaction(plan).unwrap();
    assert_eq!(report.system_ts, 30);
    assert_eq!(report.rows, 2);
    assert_eq!(s.visible_edges_at(30).unwrap(), expected);
    assert_eq!(s.visible_edges_at(20).unwrap(), expected);
    assert_eq!(s.delta_depth(), 0);

    let reopened = SegmentStore::open(&root, NODE_COUNT).unwrap();
    assert_eq!(reopened.visible_edges_at(30).unwrap(), expected);
    assert_eq!(reopened.visible_edges_at(20).unwrap(), expected);
    assert_no_csv(&root);
}

#[test]
fn stale_compaction_plan_is_rejected_without_corruption() {
    let root = temp_root("stale");
    let mut s = seeded_store(&root);
    split_correction(&mut s, 20);
    let expected = s.visible_edges_at(20).unwrap();
    let plan = s.spawn_compaction(30).unwrap().join().unwrap().unwrap();

    // The store advanced while the worker was computing; applying the old
    // plan would rewrite a different position space.
    s.materialize().unwrap();
    let err = s.apply_compaction(plan).unwrap_err().to_string();
    assert!(err.contains("stale compaction plan"), "unexpected error: {err}");
    assert_eq!(s.visible_edges_at(30).unwrap(), expected);
    assert_no_csv(&root);
}

#[test]
fn gc_refuses_to_run_while_a_reader_scope_is_alive() {
    let root = temp_root("reader_guard");
    let mut s = seeded_store(&root);
    split_correction(&mut s, 20);
    s.materialize().unwrap();

    let scope = s.read_scope();
    assert_eq!(s.active_readers(), 1);
    assert!(s.gc(GcPolicy::collect(30)).is_err());
    drop(scope);
    assert_eq!(s.active_readers(), 0);
    assert!(s.gc(GcPolicy::collect(30)).is_ok());
}

// -- 3. source-box narrowing keeps positional retract valid ----------------

#[test]
fn narrow_source_scan_preserves_every_row_and_retract_position() {
    let root = temp_root("narrow");
    let mut s = SegmentStore::open(&root, NODE_COUNT).unwrap();
    // All edges live on src=5, so the tight L2 box is [5,5] and the sweep
    // skips src 0..4 entirely. Row ids must still start at the segment's
    // first physical row (offsets[5] = 0).
    s.ingest(
        1,
        1,
        &[
            edge(5, 0, 1_000, 3, 10),
            edge(5, 0, 1_000, 4, 10),
            edge(5, 0, 1_000, 5, 10),
        ],
    )
    .unwrap();
    s.flush_at(10).unwrap();
    let seg_id = s.segments()[0].id;
    assert_eq!(s.segments()[0].src_id_min, Some(5));
    assert_eq!(s.segments()[0].src_id_max, Some(5));

    let rows = s.live_rows_at(10).unwrap();
    assert_eq!(rows.len(), 3);
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(r.pos.segment_id, seg_id);
        assert_eq!(r.pos.row, i as u32);
    }

    // A positional retract of the middle row must hit exactly that row.
    let target = rows[1].pos;
    let draft = gtv_storage::DeltaDraft {
        system_ts: 20,
        removed: vec![target],
        ..Default::default()
    };
    let report = s.publish_delta(draft).unwrap();
    assert_eq!(report.removed_count, 1);
    let live = s.live_rows_at(20).unwrap();
    assert_eq!(live.len(), 2);
    assert!(live.iter().all(|r| r.pos != target));
    assert_no_csv(&root);
}

// -- 4. delayed GC ----------------------------------------------------------

#[test]
fn delayed_gc_preserves_head_reclaims_history_and_cleans_orphans() {
    let root = temp_root("gc");
    let mut s = seeded_store(&root);
    s.publish_observation("cpi", 10, vec![observation_batch(1.0)])
        .unwrap();
    s.publish_observation("cpi", 12, vec![observation_batch(2.0)])
        .unwrap();
    split_correction(&mut s, 20);
    s.materialize_at(30).unwrap();
    let head = s.visible_edges_at(30).unwrap();
    let retired_seg = s.segments()[0].file.clone();
    let retired_delta = s.deltas()[0].file.clone();
    assert!(root.join("segments").join(&retired_seg).exists());

    // Default policy is a no-op dry run.
    let no_op = s.gc(GcPolicy::default()).unwrap();
    assert!(no_op.dry_run);
    assert_eq!(no_op.segments_removed, 0);
    assert!(!no_op.manifest_rewritten);

    // Explicit dry run reports candidates but touches nothing.
    let dry = s
        .gc(GcPolicy {
            retain_from: Some(30),
            keep_last_n: 1,
            dry_run: true,
        })
        .unwrap();
    assert!(dry.dry_run);
    assert!(dry.segments_removed >= 1);
    assert!(dry.deltas_removed >= 1);
    assert!(dry.observations_removed >= 1);
    assert!(root.join("segments").join(&retired_seg).exists());
    assert!(!dry.manifest_rewritten);

    // An interrupted publish leaves an orphan; GC must reclaim it too.
    fs::write(root.join("segments").join("seg_orphan.seg"), b"unpublished").unwrap();

    let real = s
        .gc(GcPolicy {
            retain_from: Some(30),
            keep_last_n: 1,
            dry_run: false,
        })
        .unwrap();
    assert!(real.manifest_rewritten);
    assert!(real.segments_removed >= 1);
    assert!(real.orphan_files_removed >= 1);
    assert!(!root.join("segments").join(&retired_seg).exists());
    assert!(!root.join("deltas").join(&retired_delta).exists());
    assert!(!root.join("segments").join("seg_orphan.seg").exists());

    // Head is unchanged; history below the floor is rejected loudly.
    assert_eq!(s.visible_edges_at(30).unwrap(), head);
    assert!(s.visible_edges_at(10).is_err());
    assert!(s.graph_at(10).is_err());
    assert!(s.node_lifetime_at(10).is_err());
    // The newest observation version at/below the floor is kept for `as_of`.
    assert_eq!(value_of(&s.observation_batches("cpi", 30).unwrap()), 2.0);

    let reopened = SegmentStore::open(&root, NODE_COUNT).unwrap();
    assert_eq!(reopened.visible_edges_at(30).unwrap(), head);
    assert!(reopened.visible_edges_at(10).is_err());
    assert_eq!(reopened.verify_all().unwrap() >= 1, true);
    assert_no_csv(&root);
}

// -- 5. L2 de-duplication ---------------------------------------------------

#[test]
fn interval_split_never_double_counts_in_aggregates() {
    let root = temp_root("dedup");
    let mut s = seeded_store(&root);
    split_correction(&mut s, 20);
    let head = s.system_ts();
    let view: AggregateView = s.aggregate_view(head).unwrap();
    assert_eq!(view.rows_before_dedup, 2);
    // One logical edge, two genuinely disjoint lifetimes.
    assert_eq!(view.intervals.len(), 2);
    assert_eq!(view.active_edge_count(100), 1);
    assert_eq!(view.active_edge_count(200), 0, "the deleted hole is empty");
    assert_eq!(view.active_edge_count(300), 1);
    let whole = view.range_agg(0, 1_000, 1_000);
    assert_eq!(whole.len(), 1);
    assert_eq!(whole[0].active_edges, 1, "one logical edge, not two fragments");

    // The store-level API must agree with the SQL projection at every sample.
    for t in [0, 100, 149, 150, 200, 249, 250, 500, 999, 1_000] {
        let (count, _prune) = s.active_edge_count_at(head, t).unwrap();
        assert_eq!(count, view.active_edge_count(t), "business_t={t}");
    }
    assert_no_csv(&root);
}
