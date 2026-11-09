//! `graph_at` is a merge now, not a rebuild. This is what says so.
//!
//! `graph_at(t)` used to materialise `Vec<LiveRow>` -> `Vec<EdgeRecord>` ->
//! sort -> columns. It now merges the live segments' own sorted columns, which
//! is a different program that has to produce the *same graph*. Two oracles
//! check that, and they are deliberately independent:
//!
//! 1. The row path — `live_rows_at` + `MemSegment::to_csr` — which is exactly
//!    what `graph_at` used to be. Column-for-column, not just edge counts: a
//!    merge that drops one row and duplicates another has the right count and
//!    the wrong graph.
//! 2. The documented invariant, asserted directly on the output: `offsets` is
//!    monotone, `offsets[node_count]` is the edge count, and inside every
//!    source run the keys `(valid_from, valid_to, dst)` never go backwards.
//!    The oracle cannot check that one, because the row path gets its order
//!    from a sort it no longer performs.
//!
//! Data discipline: every edge is built in memory and every file lives under a
//! tempdir outside the repository. **No CSV is ever written.**

use std::fs;
use std::path::{Path, PathBuf};

use gtv_core::csr::TemporalCSR;
use gtv_storage::delta::{Correction, EdgeKey, HolePolicy, RemoveReason};
use gtv_storage::{EdgeRecord, MemSegment, SegmentStore};

const NODES: usize = 64;

fn temp_root(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("gtv_cols_{}_{}", std::process::id(), label));
    let _ = fs::remove_dir_all(&p);
    p
}

fn edges(start: u64, count: u64, system_ts: i64) -> Vec<EdgeRecord> {
    (start..start + count)
        .map(|i| {
            EdgeRecord::new(
                i % NODES as u64,
                (i * 7 + 3) % NODES as u64,
                100 + i as i64,
                100_000 + i as i64,
                (i % 3) as u16,
                system_ts,
            )
        })
        .collect()
}

/// The row path, which is what `graph_at` used to be.
fn row_path(store: &SegmentStore, t: i64) -> TemporalCSR {
    let rows = store.live_rows_at(t).unwrap();
    let edges: Vec<EdgeRecord> = rows.iter().map(|r| r.edge).collect();
    MemSegment::to_csr(&edges, store.node_count()).unwrap()
}

/// Every column, plus the invariant, for one snapshot point.
fn assert_agrees(store: &SegmentStore, t: i64) -> TemporalCSR {
    let got = store.graph_at(t).unwrap();
    let want = row_path(store, t);

    let (a, b) = (got.columns(), want.columns());
    assert_eq!(a.node_count, b.node_count, "node_count at t={t}");
    assert_eq!(a.offsets, b.offsets, "offsets at t={t}");
    assert_eq!(a.dst, b.dst, "dst at t={t}");
    assert_eq!(a.valid_from, b.valid_from, "valid_from at t={t}");
    assert_eq!(a.valid_to, b.valid_to, "valid_to at t={t}");
    assert_eq!(a.edge_type, b.edge_type, "edge_type at t={t}");
    assert_eq!(
        a.max_valid_from, b.max_valid_from,
        "max_valid_from at t={t}"
    );
    assert_eq!(a.min_valid_to, b.min_valid_to, "min_valid_to at t={t}");
    assert_eq!(a.chunk_size, b.chunk_size, "chunk_size at t={t}");
    assert_eq!(a.chunk_index, b.chunk_index, "chunk_index at t={t}");
    assert_eq!(
        a.chunk_max_valid_to, b.chunk_max_valid_to,
        "chunk zone map at t={t}"
    );

    assert_invariant(&got, t);
    got
}

/// The order `TemporalCSR` documents, checked on the output itself.
///
/// The merge never sorts, so this is the only thing standing between "fast" and
/// "wrong": a run walked in the wrong place inside a source node would show up
/// here and nowhere else.
fn assert_invariant(csr: &TemporalCSR, t: i64) {
    let c = csr.columns();
    assert_eq!(c.offsets.len(), c.node_count + 1, "offsets length at t={t}");
    assert_eq!(c.offsets[0], 0, "offsets must start at 0");
    assert_eq!(
        c.offsets[c.node_count] as usize,
        c.dst.len(),
        "offsets tail at t={t}"
    );
    for s in 0..c.node_count {
        assert!(
            c.offsets[s] <= c.offsets[s + 1],
            "offsets not monotone at node {s}, t={t}"
        );
        let a = c.offsets[s] as usize;
        let b = c.offsets[s + 1] as usize;
        for e in a..b.saturating_sub(1) {
            let left = (c.valid_from[e], c.valid_to[e], c.dst[e]);
            let right = (c.valid_from[e + 1], c.valid_to[e + 1], c.dst[e + 1]);
            assert!(
                left <= right,
                "src {s} rows out of order at t={t}: {left:?} then {right:?}"
            );
        }
    }
}

// -- the shapes the merge has to get right ---------------------------------

#[test]
fn an_empty_store_materialises_an_empty_graph() {
    let root = temp_root("empty");
    let store = SegmentStore::open(&root, 8).unwrap();
    let g = assert_agrees(&store, 1);
    assert_eq!(g.node_count(), 8);
    assert_eq!(g.edge_count(), 0);
    // A graph with no edges still has to answer for every node.
    assert_eq!(g.degree(7).unwrap(), 0);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn several_segments_merge_into_the_row_paths_columns() {
    let root = temp_root("merge");
    let mut store = SegmentStore::open(&root, NODES).unwrap();
    // Three flushes at three system times, all live at the end: the ordinary
    // uncompacted shape, and the only one that exercises the merge.
    for (i, at) in [10i64, 20, 30].iter().enumerate() {
        store.ingest(1, 1, &edges(i as u64 * 50, 50, *at)).unwrap();
        store.flush_at(*at).unwrap();
    }
    assert_eq!(store.segments().len(), 3);
    assert_eq!(assert_agrees(&store, 30).edge_count(), 150);
    // A historical point sees a prefix of them, so the merge has to handle runs
    // that are not all live at once.
    assert_eq!(assert_agrees(&store, 20).edge_count(), 100);
    assert_eq!(assert_agrees(&store, 10).edge_count(), 50);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_retract_is_folded_out_of_a_merged_segment() {
    let root = temp_root("retract");
    let mut store = SegmentStore::open(&root, NODES).unwrap();
    store.ingest(1, 1, &edges(0, 40, 10)).unwrap();
    store.flush_at(10).unwrap();
    store.ingest(1, 1, &edges(40, 40, 20)).unwrap();
    store.flush_at(20).unwrap();

    // `edges(0, 40)` puts exactly one edge at src 5: i = 5 gives
    // (5, 38, [105, 100005), type 2). A logical delete of [200, 300) retracts
    // that row and leaves two fragments in a delta namespace, so the merge has
    // to skip a retracted position inside one run *and* splice in a run from
    // the other.
    let key = EdgeKey {
        src: 5,
        dst: 38,
        edge_type: 2,
    };
    let report = store
        .correct_edge(
            key,
            Correction {
                window: (200, 300),
                payload: None,
                reason: RemoveReason::LogicalDelete,
                object_id: 1,
            },
            HolePolicy::AssertCovered,
            30,
        )
        .unwrap();
    assert_eq!(report.removed_count, 1, "expected one version to retract");
    assert_eq!(report.put_count, 2, "expected a split into two fragments");

    let g = assert_agrees(&store, 30);
    assert_eq!(g.edge_count(), 80 - 1 + 2);
    // And the old snapshot is untouched by any of it.
    assert_eq!(assert_agrees(&store, 20).edge_count(), 80);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn unflushed_rows_join_the_merge_and_stay_ordered() {
    let root = temp_root("wal");
    let mut store = SegmentStore::open(&root, NODES).unwrap();
    store.set_flush_threshold(usize::MAX);
    store.ingest(1, 1, &edges(0, 30, 10)).unwrap();
    store.flush_at(10).unwrap();
    // Still in memory: the WAL namespace, which is a row list, not columns.
    store.ingest(1, 1, &edges(30, 30, 11)).unwrap();

    let g = assert_agrees(&store, 11);
    assert_eq!(g.edge_count(), 60);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_compacted_store_copies_its_single_segment_verbatim() {
    let root = temp_root("compact");
    let mut store = SegmentStore::open(&root, NODES).unwrap();
    for (i, at) in [10i64, 20, 30].iter().enumerate() {
        store.ingest(1, 1, &edges(i as u64 * 40, 40, *at)).unwrap();
        store.flush_at(*at).unwrap();
    }
    let before = assert_agrees(&store, 30).edge_count();

    let plan = store.spawn_compaction(31).unwrap().join().unwrap().unwrap();
    store.apply_compaction(plan).unwrap();
    let live = store
        .segments()
        .iter()
        .filter(|s| s.superseded_at.is_none())
        .count();
    assert_eq!(live, 1, "compaction should leave one live run");

    // Now the answer *is* that segment's columns; the numbers must not move.
    let g = assert_agrees(&store, 31);
    assert_eq!(g.edge_count(), before);
    // The old point is still reachable, and still agrees.
    assert_agrees(&store, 30);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_retract_with_no_replacement_rows_still_removes_the_row() {
    // The nastiest near-miss for the verbatim shortcut: one segment and a delta
    // that contributes *nothing* but a position to skip. If the shortcut judged
    // "one run" by the number of cursors and not by whether the overlay is
    // empty, the retracted edge would come straight back.
    let root = temp_root("retract_only");
    let mut store = SegmentStore::open(&root, NODES).unwrap();
    store.ingest(1, 1, &edges(0, 10, 10)).unwrap();
    store.flush_at(10).unwrap();
    assert_eq!(store.segments().len(), 1);

    let key = EdgeKey {
        src: 5,
        dst: 38,
        edge_type: 2,
    };
    let report = store
        .correct_edge(
            key,
            Correction {
                window: (105, 100_005),
                payload: None,
                reason: RemoveReason::LogicalDelete,
                object_id: 1,
            },
            HolePolicy::AssertCovered,
            20,
        )
        .unwrap();
    assert_eq!(report.removed_count, 1);
    assert_eq!(
        report.put_count, 0,
        "a full-window delete puts nothing back"
    );

    let g = assert_agrees(&store, 20);
    assert_eq!(g.edge_count(), 9);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn sparse_node_ids_keep_their_offsets() {
    let root = temp_root("sparse");
    let mut store = SegmentStore::open(&root, 16).unwrap();
    // Only nodes 0, 7 and 15 carry edges; the other 13 must still have
    // `offsets[s] == offsets[s + 1]`, which a merge that only walks occupied
    // sources can easily get wrong at the tail.
    store
        .ingest(
            1,
            1,
            &[
                EdgeRecord::new(0, 1, 0, 10, 0, 5),
                EdgeRecord::new(7, 2, 0, 10, 0, 5),
                EdgeRecord::new(15, 3, 0, 10, 0, 5),
                EdgeRecord::new(15, 4, 0, 10, 0, 5),
            ],
        )
        .unwrap();
    store.flush_at(5).unwrap();

    let g = assert_agrees(&store, 5);
    let c = g.columns();
    assert_eq!(
        c.offsets,
        &[0u32, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4]
    );
    assert_eq!(g.degree(15).unwrap(), 2);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn the_same_snapshot_materialises_identically_twice() {
    // Nothing in the merge may depend on segment open order, hash iteration or
    // an allocator address: two calls on one store have to be byte-identical.
    let root = temp_root("twice");
    let mut store = SegmentStore::open(&root, NODES).unwrap();
    for (i, at) in [10i64, 20].iter().enumerate() {
        store.ingest(1, 1, &edges(i as u64 * 60, 60, *at)).unwrap();
        store.flush_at(*at).unwrap();
    }
    let first = store.graph_at(20).unwrap();
    let second = store.graph_at(20).unwrap();
    let (a, b) = (first.columns(), second.columns());
    assert_eq!(a.offsets, b.offsets);
    assert_eq!(a.dst, b.dst);
    assert_eq!(a.valid_from, b.valid_from);
    assert_eq!(a.valid_to, b.valid_to);
    assert_eq!(a.edge_type, b.edge_type);
    let _ = fs::remove_dir_all(&root);
}

/// Hard rule from P1: generated data never appears in the repository.
#[test]
fn no_test_writes_csv() {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(rd) = fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().and_then(|s| s.to_str()) == Some("csv") {
                    out.push(p);
                }
            }
        }
    }
    let root = temp_root("csv");
    let mut store = SegmentStore::open(&root, 4).unwrap();
    store.ingest(1, 1, &edges(0, 4, 1)).unwrap();
    store.flush_at(1).unwrap();
    let _ = store.graph_at(1).unwrap();
    let mut found = Vec::new();
    walk(&root, &mut found);
    assert!(found.is_empty(), "materialisation wrote CSV: {found:?}");
    let _ = fs::remove_dir_all(&root);
}
