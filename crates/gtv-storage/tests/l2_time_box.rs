//! L2 time-box acceptance tests.
//!
//! The contract under test: **segment selection must never drop a segment that
//! holds an edge the brute-force scan would have returned**, and when it claims
//! `FullHit` the per-edge answer must really be "all edges".
//!
//! Data discipline: edges are built in memory, files live in a tempdir, no CSV
//! is produced (same rule as `p1_write_path.rs`).

use std::fs;
use std::path::PathBuf;

use gtv_storage::segment_meta::{PruneOutcome, SegmentMeta};
use gtv_storage::{EdgeRecord, SegmentStore};

const NODE_COUNT: usize = 64;
/// Each "month" is 1000 ticks; edges live in the first 500 ticks of a month.
const MONTH: i64 = 1000;

fn temp_root(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("gtv_l2_{}_{}", std::process::id(), label));
    let _ = fs::remove_dir_all(&p);
    p
}

/// Deterministic LCG so failures reproduce exactly.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo) as u64)) as i64
    }
}

/// Twelve monthly segments. Inside each month edges come in two staggered
/// cohorts: `[base, base+400)` and `[base+50, base+450)`. That shape gives a
/// non-empty **all-active** window `[base+50, base+400)` — exactly the case
/// where a `FullHit` may skip the per-edge test — plus a real gap between
/// months so pruning has something to skip.
fn build_store(root: &PathBuf) -> SegmentStore {
    let mut store = SegmentStore::open(root, NODE_COUNT).unwrap();
    for m in 0..12i64 {
        let base = m * MONTH;
        let mut batch = Vec::new();
        for i in 0..40u64 {
            let vf = base + (i as i64 % 2) * 50;
            let vt = vf + 400;
            batch.push(EdgeRecord::new(
                i % NODE_COUNT as u64,
                (i * 7 + 3) % NODE_COUNT as u64,
                vf,
                vt,
                (i % 3) as u16,
                m + 1,
            ));
        }
        store.ingest(0, 0, &batch).unwrap();
        store.flush_at(m + 1).unwrap();
    }
    store
}

/// Ground truth: segments that really hold an edge active in `[t1, t2)`.
fn brute_force_overlapping(store: &SegmentStore, t1: i64, t2: i64) -> Vec<u64> {
    let mut out = Vec::new();
    for entry in store.segments() {
        let seg = store.open_entry(entry).unwrap();
        let c = seg.csr().columns();
        let hit = (0..c.dst.len()).any(|e| c.valid_from[e] < t2 && c.valid_to[e] > t1);
        if hit {
            out.push(entry.id);
        }
    }
    out
}

fn brute_force_at(store: &SegmentStore, t: i64) -> Vec<u64> {
    let mut out = Vec::new();
    for entry in store.segments() {
        let seg = store.open_entry(entry).unwrap();
        let c = seg.csr().columns();
        let hit = (0..c.dst.len()).any(|e| c.valid_from[e] <= t && c.valid_to[e] > t);
        if hit {
            out.push(entry.id);
        }
    }
    out
}

#[test]
fn prune_never_loses_a_segment() {
    let root = temp_root("prune");
    let store = build_store(&root);
    assert_eq!(store.segments().len(), 12);

    let mut rng = Rng(0x5EED);
    for _ in 0..400 {
        let t1 = rng.range(-500, 13 * MONTH);
        let t2 = t1 + rng.range(1, 3 * MONTH);
        let expected = brute_force_overlapping(&store, t1, t2);
        let got: Vec<u64> = store
            .entries_overlapping(t1, t2)
            .into_iter()
            .map(|e| e.id)
            .collect();
        assert_eq!(
            got, expected,
            "range [{t1},{t2}) selected {got:?} but brute force says {expected:?}"
        );
    }

    for t in (-500..13 * MONTH).step_by(37) {
        let expected = brute_force_at(&store, t);
        let got: Vec<u64> = store.entries_at_business(t).into_iter().map(|e| e.id).collect();
        assert_eq!(got, expected, "point t={t}");
    }
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn full_hit_really_means_all_edges_active() {
    let root = temp_root("fullhit");
    let store = build_store(&root);
    let mut rng = Rng(0xC0FFEE);
    let mut full_hits = 0;
    for _ in 0..300 {
        let t1 = rng.range(0, 12 * MONTH);
        let t2 = t1 + rng.range(1, 200);
        for entry in store.entries_overlapping(t1, t2) {
            if entry.classify(t1, t2) != PruneOutcome::FullHit {
                continue;
            }
            full_hits += 1;
            let seg = store.open_entry(entry).unwrap();
            let c = seg.csr().columns();
            for e in 0..c.dst.len() {
                assert!(
                    c.valid_from[e] <= t1 && c.valid_to[e] >= t2,
                    "FullHit claimed for [{t1},{t2}) but edge {e} is [{}, {})",
                    c.valid_from[e],
                    c.valid_to[e]
                );
            }
        }
    }
    assert!(full_hits > 0, "no FullHit was produced; the short-circuit path is untested");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn prune_summary_matches_selection() {
    let root = temp_root("summary");
    let store = build_store(&root);
    for (t1, t2) in [(0, 100), (0, 12 * MONTH), (1200, 1300), (5000, 5001)] {
        let s = store.prune_summary(t1, t2);
        let selected = store.entries_overlapping(t1, t2).len();
        assert_eq!(s.selected, selected, "[{t1},{t2})");
        assert_eq!(s.full_hit + s.scan, selected);
        assert_eq!(s.skipped + selected, store.segments().len());
    }
    // A window covering everything must skip nothing.
    let all = store.prune_summary(0, 12 * MONTH);
    assert_eq!(all.skipped, 0);
    // A window in the gap between month 0 and month 1 skips both months.
    let gap = store.prune_summary(500, 990);
    assert_eq!(gap.skipped, 12);
    assert_eq!(gap.selected, 0);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn business_span_is_the_union() {
    let root = temp_root("span");
    let store = build_store(&root);
    let (lo, hi) = store.business_span().unwrap();
    assert_eq!(lo, 0);
    // Month 11: base 11000, the later cohort ends at 11050 + 400.
    assert_eq!(hi, 11 * MONTH + 450);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn bitemporal_selection_intersects_both_axes() {
    let root = temp_root("bitemporal");
    let store = build_store(&root);
    // Business time in month 5, system time pinned to month 5's flush.
    let got = store.entries_at(5 * MONTH + 10, 6);
    assert_eq!(got.len(), 1, "only month 5 is both visible and possibly active");
    assert_eq!(got[0].system_ts, 6);
    // Same business time, but the system only knows months 0..3.
    assert!(store.entries_at(5 * MONTH + 10, 4).is_empty());
    // Latest system time sees every month.
    assert_eq!(store.entries_at(5 * MONTH + 10, i64::MAX).len(), 1);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn legacy_manifest_without_l2_fields_selects_conservatively() {
    let root = temp_root("legacy");
    {
        let store = build_store(&root);
        assert!(store.segments().iter().all(|e| e.meta().is_some()));
    }

    // Simulate a manifest written before L2 existed.
    let path = root.join("manifest.json");
    let text = fs::read_to_string(&path).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
    for seg in value["segments"].as_array_mut().unwrap() {
        for k in [
            "valid_from_min",
            "valid_to_max",
            "max_valid_from",
            "min_valid_to",
            "node_id_min",
            "node_id_max",
        ] {
            seg.as_object_mut().unwrap().remove(k);
        }
    }
    fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

    let store = SegmentStore::open(&root, NODE_COUNT).unwrap();
    assert_eq!(store.segments().len(), 12);
    assert!(store.segments().iter().all(|e| e.meta().is_none()));
    // Unknown metadata must never prune.
    assert_eq!(store.entries_overlapping(0, 1).len(), 12);
    assert_eq!(store.entries_at_business(-1_000_000).len(), 12);
    assert_eq!(store.entries_overlapping(0, 12 * MONTH).len(), 12);
    assert_eq!(store.prune_summary(0, 12 * MONTH).skipped, 0);
    assert!(store.business_span().is_none());
    // Data is still fully readable.
    let g = store.graph_at(i64::MAX).unwrap();
    assert_eq!(g.edge_count(), 12 * 40);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn empty_store_has_no_span_and_prunes_everything() {
    let root = temp_root("empty");
    let store = SegmentStore::open(&root, NODE_COUNT).unwrap();
    assert!(store.business_span().is_none());
    assert!(store.entries_overlapping(i64::MIN, i64::MAX).is_empty());
    let s = store.prune_summary(0, 1);
    assert_eq!((s.skipped, s.selected), (0, 0));
    assert_eq!(s.skip_ratio(), 0.0);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn meta_survives_reopen() {
    let root = temp_root("reopen");
    let before = {
        let store = build_store(&root);
        store.segments().to_vec()
    };
    let store = SegmentStore::open(&root, NODE_COUNT).unwrap();
    assert_eq!(store.segments(), before.as_slice());
    let m: SegmentMeta = store.segments()[0].meta().unwrap();
    assert_eq!(m.edge_count, 40);
    assert_eq!(m.valid_from_min, 0);
    assert_eq!(m.valid_to_max, 450);
    // Union says "maybe", intersection says "all": both live side by side.
    assert_eq!((m.max_valid_from, m.min_valid_to), (50, 400));
    assert!(m.all_active_at(200));
    assert!(!m.all_active_at(10), "the first cohort is active but not all edges");
    assert_eq!(m.classify(100, 300), PruneOutcome::FullHit);
    assert_eq!(m.classify(0, 450), PruneOutcome::Scan);
    assert_eq!(m.classify(450, 900), PruneOutcome::Skip);
    let _ = fs::remove_dir_all(&root);
}
