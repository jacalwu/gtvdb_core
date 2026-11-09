//! M2 control-plane regression gate (`bench_correction`).
//!
//! `bench_endpoint` guards the **read** path (CSR neighbours, 2 comparisons).
//! The asymmetric design moved all new complexity to the **control** plane, so
//! this benchmark guards the three costs that can now regress:
//!
//! 1. single-correction publish latency (WAL intent + delta file + manifest);
//! 2. forced materialization when the K=1 overlay is already occupied;
//! 3. P99 version-query latency at delta depth 0 / 1 / materialized.
//!
//! It also asserts the §23.6 invariant in bulk: the overlay answer equals the
//! materialized answer. All files live in a tempdir outside the repository;
//! no CSV or generated dataset is ever produced.
//!
//! Run with: `cargo run --release -p gtv-storage --example bench_correction`

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use gtv_storage::delta::{Correction, EdgeKey, HolePolicy, RemoveReason};
use gtv_storage::{EdgeRecord, SegmentStore};

const NODE_COUNT: usize = 512;
const EDGES: usize = 4_000;
const PUBLISH_ITERS: usize = 20;
const QUERY_REPEATS: usize = 80;

fn bench_root() -> PathBuf {
    let p = std::env::temp_dir().join(format!("gtv_bench_correction_{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    p
}

fn key_of(i: usize) -> EdgeKey {
    EdgeKey {
        src: (i % NODE_COUNT) as u64,
        dst: ((i / NODE_COUNT) % NODE_COUNT) as u64,
        edge_type: 0,
    }
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn p99(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[(v.len() * 99) / 100]
}

fn query_p99(s: &SegmentStore, t: i64) -> (Duration, usize) {
    let mut samples = Vec::with_capacity(QUERY_REPEATS);
    let mut rows = 0usize;
    for _ in 0..QUERY_REPEATS {
        let start = Instant::now();
        rows = s.visible_edges_at(t).unwrap().len();
        samples.push(start.elapsed());
    }
    (p99(samples), rows)
}

fn main() {
    let root = bench_root();
    let mut s = SegmentStore::open(&root, NODE_COUNT).unwrap();
    let edges: Vec<EdgeRecord> = (0..EDGES)
        .map(|i| {
            let k = key_of(i);
            EdgeRecord::new(k.src, k.dst, 0, 1_000, k.edge_type, 10)
        })
        .collect();
    s.ingest(1, 1, &edges).unwrap();
    s.flush_at(10).unwrap();

    let (p99_depth0, rows0) = query_p99(&s, 10);
    assert_eq!(rows0, EDGES);

    // -- 1. pure publish (depth 0 -> 1), isolated from forced compaction ----
    let mut publish = Vec::with_capacity(PUBLISH_ITERS);
    for i in 0..PUBLISH_ITERS {
        // Reset to depth 0 so the timed call measures only the publish.
        if s.delta_depth() > 0 {
            s.materialize().unwrap();
        }
        let k = key_of(i);
        let ts = s.system_ts() + 10;
        let start = Instant::now();
        s.correct_edge(
            k,
            Correction {
                window: (100, 200),
                payload: None,
                reason: RemoveReason::LogicalDelete,
                object_id: k.src,
            },
            HolePolicy::AssertCovered,
            ts,
        )
        .unwrap();
        publish.push(start.elapsed());
    }
    let publish_p50 = {
        let mut v = publish.clone();
        v.sort_unstable();
        v[v.len() / 2]
    };
    let publish_total: Duration = publish.iter().sum();

    // -- 2. forced materialization (depth 1 -> compact -> publish) ---------
    let head_overlay = s.system_ts();
    let overlay_edges = s.visible_edges_at(head_overlay).unwrap();
    let (p99_depth1, rows1) = query_p99(&s, head_overlay);
    assert_eq!(rows1, overlay_edges.len());

    let k = key_of(PUBLISH_ITERS + 1);
    let ts = s.system_ts() + 10;
    let start = Instant::now();
    let forced = s
        .correct_edge(
            k,
            Correction {
                window: (100, 200),
                payload: None,
                reason: RemoveReason::LogicalDelete,
                object_id: k.src,
            },
            HolePolicy::AssertCovered,
            ts,
        )
        .unwrap();
    let forced_us = us(start.elapsed());
    assert!(forced.forced_materialization);

    // -- 3. materialized query and the §23.6 equality gate ------------------
    let report = s.materialize().unwrap();
    let head_materialized = report.system_ts;
    let materialized_edges = s.visible_edges_at(head_materialized).unwrap();
    let (p99_materialized, rows_m) = query_p99(&s, head_materialized);
    assert_eq!(rows_m, materialized_edges.len());
    // Each logical delete splits one row into two, so the head has more
    // physical rows than the uncorrected baseline — that is expected.
    assert_eq!(
        s.visible_edges_at(forced.system_ts).unwrap(),
        materialized_edges,
        "§23.6: overlay and materialized head must be identical"
    );

    println!("bench_correction  nodes={NODE_COUNT} edges={EDGES}");
    println!("  correction publish      : {:>8.1} us/op p50  ({:.0} ops/s)",
        us(publish_p50),
        PUBLISH_ITERS as f64 / publish_total.as_secs_f64());
    println!("  forced materialization  : {forced_us:>8.1} us/op  (includes one O(E) fold)");
    println!("  query P99 depth=0       : {:>8.1} us  ({rows0} rows)", us(p99_depth0));
    println!("  query P99 depth=1       : {:>8.1} us  ({rows1} rows, {:.2}x)",
        us(p99_depth1), us(p99_depth1) / us(p99_depth0).max(1e-9));
    println!("  query P99 materialized  : {:>8.1} us  ({rows_m} rows, {:.2}x)",
        us(p99_materialized), us(p99_materialized) / us(p99_depth0).max(1e-9));
    println!("  all paths agree         : true");

    let _ = fs::remove_dir_all(&root);
}
