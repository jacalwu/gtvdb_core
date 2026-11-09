//! M3 acceptance tests: planner-gated bitmap, resident maintenance worker,
//! named tags and their GC protection.
//!
//! Acceptance targets:
//! * review §23.4 — dropping the bitmap cache changes no answer;
//! * review §7 / §8 — a worker-prepared plan equals synchronous materialization
//!   row-for-row, and a stale plan is rejected instead of rewriting newer data;
//! * a tag pins its history across restarts and blocks delayed GC;
//! * pre-M3 manifests (no `tags` field) still load.
//!
//! All files live in a tempdir outside the repository; no CSV is written.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use gtv_core::{ActiveNodeBitmap, BitmapCache, BitmapKey, BitmapPlanner, NodeLifetime};
use gtv_storage::delta::{Correction, EdgeKey, HolePolicy, RemoveReason};
use gtv_storage::{
    read_tag_log, EdgeRecord, GcPolicy, MaintenanceConfig, MaintenanceService, SegmentStore,
};

const NODE_COUNT: usize = 8;
const KEY: EdgeKey = EdgeKey {
    src: 1,
    dst: 2,
    edge_type: 7,
};

fn temp_root(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("gtv_m3_{}_{}", std::process::id(), label));
    let _ = fs::remove_dir_all(&p);
    p
}

fn seeded_store(root: &Path) -> SegmentStore {
    let mut s = SegmentStore::open(root, NODE_COUNT).unwrap();
    s.ingest(7, 3, &[EdgeRecord::new(1, 2, 0, 1_000, 7, 10)])
        .unwrap();
    s.flush_at(10).unwrap();
    s
}

fn correction(s: &mut SegmentStore, system_ts: i64) {
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

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[test]
fn bitmap_cache_is_discardable_under_random_lifetimes() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for case in 0..200 {
        let n = 1 + rng.below(512) as usize;
        let boundary = 100 + rng.below(900) as i64;
        let mut from = Vec::with_capacity(n);
        let mut to = Vec::with_capacity(n);
        for _ in 0..n {
            let f = rng.below(80) as i64;
            let t = if rng.below(3) == 0 {
                boundary
            } else {
                f + 1 + rng.below(2_000) as i64
            };
            from.push(f);
            to.push(t.max(f));
        }
        // Guarantee at least one expired node so the O(1) all-active bounding
        // box never short-circuits the planner in this cache test.
        to[0] = from[0];
        let lifetime = NodeLifetime::new(from, to).unwrap();
        let t = rng.below(2_500) as i64;
        let key = BitmapKey::new(case as u64, t, 0);
        let bitmap = ActiveNodeBitmap::build(key.clone(), &lifetime).unwrap();
        bitmap.verify_against(&lifetime).unwrap();

        let mut cache = BitmapCache::new(BitmapPlanner {
            amortization_ratio: 1,
            min_selected_edges: 0,
            max_cache_bytes: 0,
        });
        assert!(cache.get_or_build(&key, &lifetime, n * 100).unwrap().is_some());
        let with_cache: Vec<bool> = (0..n)
            .map(|i| cache.active_at(i as u64, t, &key, &lifetime))
            .collect();
        cache.invalidate();
        let without_cache: Vec<bool> = (0..n)
            .map(|i| cache.active_at(i as u64, t, &key, &lifetime))
            .collect();
        assert_eq!(with_cache, without_cache, "case {case}");
        assert_eq!(
            without_cache,
            (0..n).map(|i| lifetime.active_at(i as u64, t)).collect::<Vec<_>>(),
            "case {case}: fallback must be the fact source"
        );
    }
}

#[test]
fn worker_plan_matches_synchronous_materialization() {
    let root = temp_root("worker_vs_sync");
    let mut worker = seeded_store(&root.join("worker"));
    let mut sync = seeded_store(&root.join("sync"));
    correction(&mut worker, 20);
    correction(&mut sync, 20);
    assert_eq!(worker.delta_depth(), 1);
    let history = worker.visible_edges_at(20).unwrap();

    let mut svc = MaintenanceService::start(MaintenanceConfig::default());
    assert!(svc.submit_store(&worker));
    let plan = svc
        .wait_for_plan(Duration::from_secs(20))
        .expect("worker produced no plan")
        .expect("worker prepare failed");
    let head = plan.system_ts;
    worker.apply_compaction(plan).unwrap();
    sync.materialize_at(head).unwrap();

    assert_eq!(
        worker.visible_edges_at(head).unwrap(),
        sync.visible_edges_at(head).unwrap(),
        "worker and synchronous paths must fold the same visible set"
    );
    assert_eq!(
        worker.visible_edges_at(20).unwrap(),
        history,
        "a physical rewrite must not change AS KNOWN AT 20"
    );
    assert_eq!(worker.delta_depth(), 0);
    let status = svc.status();
    assert_eq!(status.submitted, 1);
    assert!(status.plans_produced >= 1);
    assert!(!status.last_error.as_deref().unwrap_or("").contains("failed"));
    svc.stop();
    assert!(!svc.status().running);
    assert_no_csv(&root);
}

#[test]
fn stale_worker_plan_is_rejected_and_data_is_intact() {
    let root = temp_root("stale_plan");
    let mut s = seeded_store(&root);
    correction(&mut s, 20);
    let mut svc = MaintenanceService::start(MaintenanceConfig::default());
    assert!(svc.submit_store(&s));
    let plan = svc
        .wait_for_plan(Duration::from_secs(20))
        .unwrap()
        .unwrap();
    let before = s.visible_edges_at(s.system_ts()).unwrap();

    // A second publish advances the head (and forces materialization at K=1),
    // so the plan prepared against the old head must be refused outright.
    // Use a different window: the first correction already removed [150,250).
    s.correct_edge(
        KEY,
        Correction {
            window: (300, 400),
            payload: None,
            reason: RemoveReason::LogicalDelete,
            object_id: 18,
        },
        HolePolicy::AssertCovered,
        40,
    )
    .unwrap();
    let err = s.apply_compaction(plan).unwrap_err().to_string();
    assert!(err.contains("stale compaction plan"), "unexpected error: {err}");
    let head = s.system_ts();
    assert_eq!(s.visible_edges_at(20).unwrap(), before, "history unchanged");
    assert!(!s.visible_edges_at(head).unwrap().is_empty());
    svc.stop();
    assert_no_csv(&root);
}

#[test]
fn tag_round_trip_and_gc_protection() {
    let root = temp_root("tags_gc");
    let mut s = seeded_store(&root);
    correction(&mut s, 20);
    s.materialize().unwrap();

    s.create_tag("baseline", 10).unwrap();
    s.create_tag("after-correction", 20).unwrap();
    assert_eq!(s.resolve_tag("baseline").unwrap(), 10);
    assert_eq!(s.tags().len(), 2);
    let before = s.visible_edges_at(10).unwrap();

    // Reopen: tags are manifest state, not a session-local map.
    drop(s);
    let mut reopened = SegmentStore::open(&root, NODE_COUNT).unwrap();
    assert_eq!(reopened.resolve_tag("baseline").unwrap(), 10);
    assert_eq!(reopened.tags().get("after-correction"), Some(&20));
    let rows = read_tag_log(&root).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].name, "after-correction");
    assert_eq!(rows[1].name, "baseline");

    // An aggressive GC must be pulled back by the oldest tag.
    let report = reopened.gc(GcPolicy::collect(1_000)).unwrap();
    assert!(report.pinned_by_tags, "tag did not constrain the cutoff");
    assert!(report.cutoff <= 10, "cutoff {} exceeds the pinned tag", report.cutoff);
    assert_eq!(reopened.visible_edges_at(10).unwrap(), before);
    assert_eq!(reopened.resolve_tag("baseline").unwrap(), 10);

    // Dropping the tag releases the pin; a later GC can advance the floor.
    assert!(reopened.drop_tag("baseline").unwrap());
    assert!(!reopened.drop_tag("baseline").unwrap());
    let report2 = reopened.gc(GcPolicy::collect(1_000)).unwrap();
    assert!(!report2.pinned_by_tags);
    assert!(reopened.resolve_tag("baseline").is_err());
    assert_no_csv(&root);
}

#[test]
fn tag_names_are_validated_and_points_must_be_published() {
    let root = temp_root("tag_validation");
    let mut s = seeded_store(&root);
    for bad in ["", "has space", "a/b", "x".repeat(65).as_str()] {
        assert!(s.create_tag(bad, 10).is_err(), "accepted invalid tag `{bad}`");
    }
    assert!(s.create_tag("future", 999).is_err());
    s.create_tag("v1.0-rc_1", 10).unwrap();
    assert_eq!(s.resolve_tag("v1.0-rc_1").unwrap(), 10);
    assert!(s.resolve_tag("missing").is_err());
    assert_no_csv(&root);
}

#[test]
fn legacy_manifest_without_tags_still_loads() {
    let root = temp_root("legacy_tags");
    let mut s = seeded_store(&root);
    s.create_tag("temp", 10).unwrap();
    let path = root.join("manifest.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    value.as_object_mut().unwrap().remove("tags");
    fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

    let reopened = SegmentStore::open(&root, NODE_COUNT).unwrap();
    assert!(reopened.tags().is_empty());
    assert!(read_tag_log(&root).unwrap().is_empty());
    assert!(reopened.resolve_tag("temp").is_err());
    assert_eq!(reopened.visible_edges_at(10).unwrap().len(), 1);
    assert_no_csv(&root);
}

#[test]
fn maintenance_defaults_are_non_destructive() {
    let root = temp_root("maintenance_defaults");
    let mut s = seeded_store(&root);
    correction(&mut s, 20);
    let mut svc = MaintenanceService::start(MaintenanceConfig::default());
    let policy = svc.gc_policy(s.system_ts());
    assert_eq!(policy.retain_from, None);
    assert!(policy.dry_run);
    let report = s.gc(policy).unwrap();
    assert!(report.dry_run);
    assert!(!report.manifest_rewritten);
    assert_eq!(s.tags().len(), 0);

    let prod = MaintenanceService::start(MaintenanceConfig {
        retain_window: Some(5),
        gc_dry_run: false,
        ..MaintenanceConfig::default()
    });
    let p = prod.gc_policy(s.system_ts());
    assert_eq!(p.retain_from, Some(s.system_ts() - 5));
    assert!(!p.dry_run);
    svc.stop();
    assert_no_csv(&root);
}
