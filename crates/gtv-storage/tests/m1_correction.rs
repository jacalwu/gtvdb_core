//! M1 acceptance tests: positional retract, bounded overlay, compaction and
//! crash recovery.
//!
//! Three claims are checked (§23.6 of the asymmetric design review):
//! 1. `graph_at()` folds the retract overlay, so a correction never leaves the
//!    old and new version both visible.
//! 2. An overlay query, a *naive replay* and a materialized query agree for a
//!    randomised sequence of corrections, at every historical snapshot point.
//! 3. The manifest write is the single publish point: an orphaned WAL mutation
//!    (crash before publish) is not replayed, and `gtv_invalidations()` cannot
//!    see it.
//!
//! Data discipline: all edges are built in memory and all files live under a
//! tempdir outside the repository. **No CSV is ever written.**

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use gtv_storage::delta::{
    Correction, DeltaContext, EdgeKey, HolePolicy, MaterializationState, RemoveReason,
    SnapshotDelta,
};
use gtv_storage::{
    read_invalidation_log, EdgeRecord, SegmentStore, Wal, WalFrame, MAX_DELTA_DEPTH,
};

const NODE_COUNT: usize = 8;
const KEY: EdgeKey = EdgeKey {
    src: 1,
    dst: 2,
    edge_type: 7,
};

fn temp_root(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("gtv_m1_{}_{}", std::process::id(), label));
    let _ = fs::remove_dir_all(&p);
    p
}

fn edge(from: i64, to: i64, edge_type: u16, system_ts: i64) -> EdgeRecord {
    EdgeRecord::new(KEY.src, KEY.dst, from, to, edge_type, system_ts)
}

fn correction(from: i64, to: i64, payload: Option<EdgeRecord>, reason: RemoveReason) -> Correction {
    Correction {
        window: (from, to),
        payload,
        reason,
        object_id: 17,
    }
}

/// Hard rule from P1: generated data must never appear in the repository.
/// Walk the whole store root to catch anything a test accidentally emits.
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

fn seeded_store(root: &Path) -> SegmentStore {
    let mut s = SegmentStore::open(root, NODE_COUNT).unwrap();
    s.ingest(7, 3, &[edge(0, 1_000, KEY.edge_type, 10)]).unwrap();
    s.flush_at(10).unwrap();
    s
}

// -- 1. retract folding -----------------------------------------------------

#[test]
fn graph_at_folds_retract_and_as_known_at_stays_stable() {
    let root = temp_root("fold");
    let mut s = seeded_store(&root);

    let base = s.visible_edges_at(10).unwrap();
    assert_eq!(base.len(), 1);
    assert_eq!(s.graph_at(10).unwrap().columns().dst.len(), 1);

    // A logical delete of [150, 250) splits the single version into two
    // fragments. Pure union would have produced three rows (old + both new).
    let r = s
        .correct_edge(
            KEY,
            correction(150, 250, None, RemoveReason::LogicalDelete),
            HolePolicy::AssertCovered,
            20,
        )
        .unwrap();
    assert_eq!(r.removed_count, 1);
    assert_eq!(r.put_count, 2);
    assert_eq!(s.delta_depth(), 1);

    let head = s.visible_edges_at(20).unwrap();
    assert_eq!(
        head,
        vec![
            (1, 2, 0, 150, 7),
            (1, 2, 250, 1_000, 7),
        ],
        "graph_at must not double-count the retracted version"
    );
    assert_eq!(s.graph_at(20).unwrap().columns().dst.len(), 2);
    assert_eq!(s.visible_edges_at(10).unwrap(), base, "AS KNOWN AT 10 is immutable");

    // Compaction is a physical rewrite: every snapshot point must survive it.
    let m = s.materialize().unwrap();
    assert_eq!(m.system_ts, 21);
    assert_eq!(m.retired_deltas, 1);
    assert_eq!(s.delta_depth(), 0);
    assert_eq!(
        s.deltas()[0].state,
        MaterializationState::Materialized
    );
    for t in [5, 10, 15, 20, 21, 30] {
        let expected = if t < 10 {
            vec![]
        } else if t < 20 {
            base.clone()
        } else {
            head.clone()
        };
        assert_eq!(s.visible_edges_at(t).unwrap(), expected, "snapshot {t}");
    }
    assert_eq!(s.graph_at(21).unwrap().columns().dst.len(), 2);
    assert_no_csv(&root);
}

#[test]
fn after_image_can_reclassify_without_resurrecting_old_type() {
    let root = temp_root("reclassify");
    let mut s = seeded_store(&root);
    s.correct_edge(
        KEY,
        correction(0, 1_000, Some(edge(0, 1_000, 9, 20)), RemoveReason::PropertyUpdate),
        HolePolicy::AssertCovered,
        20,
    )
    .unwrap();
    let head = s.visible_edges_at(20).unwrap();
    assert_eq!(head, vec![(1, 2, 0, 1_000, 9)]);
    assert_eq!(s.graph_at(20).unwrap().columns().edge_type[0], 9);
    assert_eq!(s.visible_edges_at(10).unwrap(), vec![(1, 2, 0, 1_000, 7)]);
    assert_no_csv(&root);
}

#[test]
fn tight_src_box_prunes_without_opening_a_segment() {
    let root = temp_root("srcbox");
    let mut s = SegmentStore::open(&root, 64).unwrap();
    s.ingest(7, 3, &[EdgeRecord::new(40, 2, 0, 100, 7, 10)]).unwrap();
    s.flush_at(10).unwrap();
    s.ingest(7, 3, &[EdgeRecord::new(5, 2, 0, 100, 7, 11)]).unwrap();
    s.flush_at(11).unwrap();

    assert_eq!(s.segments().len(), 2);
    assert_eq!(s.segments()[0].meta().unwrap().src_id_min, 40);
    assert_eq!(s.segments()[0].meta().unwrap().src_id_max, 40);
    // `node_id_min/max` is a union over src AND dst, so it cannot express this.
    assert!(s.segments()[0].meta().unwrap().node_id_min <= 2);
    assert_eq!(s.entries_at_src(40, 11).len(), 1);
    assert_eq!(s.entries_at_src(5, 11).len(), 1);
    assert!(s.entries_at_src(20, 11).is_empty(), "no segment has src runs near 20");
    assert_no_csv(&root);
}

#[test]
fn legacy_manifest_without_deltas_still_loads() {
    // serde(default) on every M1 field: a pre-M1 store must open unchanged.
    let root = temp_root("legacy");
    let _ = seeded_store(&root);
    let path = root.join("manifest.json");
    let mut m: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let obj = m.as_object_mut().unwrap();
    obj.remove("deltas");
    obj.remove("node_versions");
    obj.remove("next_delta_id");
    for seg in obj["segments"].as_array_mut().unwrap() {
        let seg = seg.as_object_mut().unwrap();
        seg.remove("src_id_min");
        seg.remove("src_id_max");
        seg.remove("superseded_at");
    }
    fs::write(&path, serde_json::to_vec_pretty(&m).unwrap()).unwrap();

    let reopened = SegmentStore::open(&root, NODE_COUNT).unwrap();
    assert!(reopened.deltas().is_empty());
    assert_eq!(reopened.visible_edges_at(10).unwrap().len(), 1);
    assert!(reopened.segments()[0].may_contain_src(1));
    assert!(reopened.segments()[0].may_contain_src(7), "unknown src box never prunes");
    assert_no_csv(&root);
}

// -- 2. §23.6 randomized property: overlay == naive == materialized --------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 17
    }
    fn below(&mut self, n: i64) -> i64 {
        (self.next() % n as u64) as i64
    }
}

/// Independent, position-free reimplementation of the interval sweep. It only
/// knows a list of business intervals and the corrected window — no segments,
/// no RowPos, no RetractSet. If the store's overlay walk disagrees with this,
/// the folding logic is wrong rather than merely differently ordered.
fn naive_apply(state: &mut Vec<(i64, i64)>, window: (i64, i64), payload: bool) {
    let (lo, hi) = window;
    let mut out = Vec::new();
    let mut covered = Vec::new();
    for &(f, t) in state.iter() {
        if t <= lo || f >= hi {
            out.push((f, t));
            continue;
        }
        if f < lo {
            out.push((f, lo));
        }
        if t > hi {
            out.push((hi, t));
        }
        covered.push((f.max(lo), t.min(hi)));
    }
    if payload {
        covered.sort();
        let mut merged: Vec<(i64, i64)> = Vec::new();
        for (a, b) in covered {
            if let Some(last) = merged.last_mut() {
                if a <= last.1 {
                    last.1 = last.1.max(b);
                    continue;
                }
            }
            merged.push((a, b));
        }
        out.extend(merged);
    }
    out.sort();
    *state = out;
}

fn all_snapshot_points(s: &SegmentStore) -> Vec<i64> {
    let mut pts = BTreeSet::new();
    pts.insert(0);
    for e in s.segments() {
        pts.insert(e.system_ts);
        if let Some(m) = e.superseded_at {
            pts.insert(m - 1);
            pts.insert(m);
            pts.insert(m + 1);
        }
    }
    for d in s.deltas() {
        pts.insert(d.system_ts);
        if let Some(m) = d.superseded_at {
            pts.insert(m - 1);
            pts.insert(m);
        }
    }
    pts.insert(s.system_ts());
    pts.insert(s.system_ts() + 100);
    pts.into_iter().collect()
}

#[test]
fn overlay_equals_naive_replay_and_materialized_query() {
    let mut rng = Rng(0x5eed_1234_9abc_def0);
    const CASES: usize = 120;

    for case in 0..CASES {
        // A fresh store per case: snapshot points must be reproducible from 0.
        let root = temp_root(&format!("property_{case}"));
        let mut s = seeded_store(&root);
        let mut naive = vec![(0i64, 1_000i64)];
        let mut sys = 20i64;

        for step in 0..3 {
            let lo = rng.below(900);
            let hi = lo + 1 + rng.below(1_000 - lo);
            let payload = rng.below(2) == 0;
            let c = correction(
                lo,
                hi,
                payload.then(|| edge(lo, hi, KEY.edge_type, sys)),
                if payload {
                    RemoveReason::ValidTimeCorrection
                } else {
                    RemoveReason::LogicalDelete
                },
            );
            let report = s
                .correct_edge(KEY, c, HolePolicy::PreserveHoles, sys)
                .unwrap();
            assert!(
                s.delta_depth() <= MAX_DELTA_DEPTH,
                "case {case} step {step}: overlay bound K=1 violated"
            );
            if step >= 1 {
                assert!(
                    report.forced_materialization,
                    "case {case} step {step}: second live correction must fold the first"
                );
            }
            naive_apply(&mut naive, (lo, hi), payload);
            let expected: Vec<_> = naive
                .iter()
                .map(|&(f, t)| (1, 2, f, t, 7))
                .collect();
            assert_eq!(
                s.visible_edges_at(s.system_ts()).unwrap(),
                expected,
                "case {case} step {step}: overlay disagrees with naive replay"
            );
            sys += 10;
        }

        // Every historical snapshot point, before and after a final
        // compaction, must answer identically.
        let points = all_snapshot_points(&s);
        let before: Vec<(i64, Vec<(u64, u64, i64, i64, u16)>)> = points
            .iter()
            .map(|&t| (t, s.visible_edges_at(t).unwrap()))
            .collect();
        s.materialize().unwrap();
        assert_eq!(s.delta_depth(), 0, "case {case}");
        for (t, v) in before {
            assert_eq!(
                s.visible_edges_at(t).unwrap(),
                v,
                "case {case}: snapshot {t} changed after materialization"
            );
        }
        assert_eq!(
            s.visible_edges_at(s.system_ts()).unwrap(),
            naive.iter().map(|&(f, t)| (1, 2, f, t, 7)).collect::<Vec<_>>(),
            "case {case}: head changed after materialization"
        );
        assert_no_csv(&root);
    }
}

// -- 3. crash consistency ---------------------------------------------------

#[test]
fn orphaned_wal_mutation_is_not_replayed_and_is_invisible_to_the_log() {
    let root = temp_root("orphan");
    {
        let mut s = seeded_store(&root);
        s.set_flush_threshold(usize::MAX);
    }
    // Simulate: WAL intent appended, process dies before the manifest write.
    let delta = SnapshotDelta::empty(DeltaContext {
        delta_id: 777,
        base_system_ts: 10,
        system_ts: 20,
        commit_id: 999,
        put_segment_id: 900,
    });
    {
        let (mut wal, _scan) = Wal::open(root.join("wal.log")).unwrap();
        wal.append(&WalFrame::mutation(999, delta)).unwrap();
    }

    let s = SegmentStore::open(&root, NODE_COUNT).unwrap();
    assert_eq!(s.recovery().mutations_orphaned, 1);
    assert_eq!(s.recovery().mutations_published, 0);
    assert!(s.deltas().is_empty(), "uncommitted correction must not be applied");
    assert_eq!(s.visible_edges_at(1_000).unwrap(), vec![(1, 2, 0, 1_000, 7)]);
    assert!(read_invalidation_log(&root, 0).unwrap().is_empty());
    assert_no_csv(&root);
}

#[test]
fn published_delta_survives_reopen_with_atomic_invalidation_view() {
    let root = temp_root("reopen");
    let (commit, sys) = {
        let mut s = seeded_store(&root);
        let r = s
            .correct_edge(
                KEY,
                correction(100, 200, None, RemoveReason::ValidTimeCorrection),
                HolePolicy::AssertCovered,
                20,
            )
            .unwrap();
        s.correct_node(3, 100, 200, 30).unwrap();
        (r.commit_id, r.system_ts)
    };

    let s = SegmentStore::open(&root, NODE_COUNT).unwrap();
    assert_eq!(s.deltas().len(), 2);
    assert_eq!(s.recovery().mutations_published, 2);
    assert_eq!(s.node_versions().len(), 1);

    // Old snapshots keep the original row set and the original node lifetime.
    assert_eq!(s.visible_edges_at(10).unwrap(), vec![(1, 2, 0, 1_000, 7)]);
    assert_eq!(
        s.visible_edges_at(sys).unwrap(),
        vec![(1, 2, 0, 100, 7), (1, 2, 200, 1_000, 7)]
    );
    assert!(s.node_lifetime_at(20).unwrap().active_at(3, 50));
    assert!(!s.node_lifetime_at(30).unwrap().active_at(3, 50));
    assert!(s.node_lifetime_at(30).unwrap().active_at(3, 150));
    assert_eq!(s.node_lifetime_at(30).unwrap().range(3), Some((100, 200)));

    let log = read_invalidation_log(&root, 0).unwrap();
    assert_eq!(log.len(), 2);
    assert_eq!(log[0].commit_id, commit);
    assert_eq!(log[0].object_kind, "edge");
    assert_eq!(log[0].reason, "valid_time_correction");
    assert_eq!(log[1].object_kind, "node");
    assert_eq!(log[1].object_id, 3);
    assert_eq!(log[1].reason, "lifetime_correction");
    // Cursor semantics: everything at or after the second commit.
    assert_eq!(read_invalidation_log(&root, log[1].commit_id).unwrap().len(), 1);
    assert_eq!(s.invalidations_since(0).unwrap(), log);
    assert_no_csv(&root);
}

#[test]
fn explicit_positions_cannot_be_carried_across_forced_compaction() {
    let root = temp_root("stale_positions");
    let mut s = seeded_store(&root);
    s.correct_edge(
        KEY,
        correction(100, 200, None, RemoveReason::LogicalDelete),
        HolePolicy::AssertCovered,
        20,
    )
    .unwrap();
    // A low-level caller that pre-computed positions must be told to
    // materialize first, rather than silently losing the retract.
    let stale = s.live_rows_at(20).unwrap()[0].pos;
    let err = s
        .publish_delta(gtv_storage::DeltaDraft {
            system_ts: 30,
            removed: vec![stale],
            ..Default::default()
        })
        .unwrap_err();
    assert!(err.to_string().contains("forced compaction"));
    assert_no_csv(&root);
}
