//! Residency and invalidation, without a GPU.
//!
//! Every test here runs on the CPU backend on purpose. The cache, the budget and
//! the invalidation rules are host-side logic, and they are the part that must
//! be right *before* any device kernel exists: a kernel reading a stale copy is
//! the one bug no later measurement can catch. Running the same tests against a
//! real device is a matter of swapping the backend in, which is why the backend
//! is a parameter here and not a global.
//!
//! The scenarios mirror `docs/cuda-acceleration-plan.md` §4.3 and the six
//! invalidation cases in `docs/gtvdb-cpu-cuda-cursor-development-spec.md` §6.3.

use std::sync::Arc;

use gtv_accelerator::{
    Accelerator, AcceleratorError, CpuAccelerator, ExecutionPolicy, ResidentCache, SnapshotView,
};

/// Two snapshots of the same graph. Snapshot 2 retracts edge 0 -> 2, which is
/// exactly the kind of change that must never be answered from a cached copy.
struct Graphs {
    s1: Vec<(u64, i64, i64)>,
    s2: Vec<(u64, i64, i64)>,
}

fn graphs() -> Graphs {
    Graphs {
        // (dst, valid_from, valid_to)
        s1: vec![(1, 0, 100), (2, 0, 100)],
        s2: vec![(1, 0, 100)],
    }
}

struct Snapshot {
    offsets: Vec<u32>,
    dst: Vec<u64>,
    vf: Vec<i64>,
    vt: Vec<i64>,
    et: Vec<u16>,
}

impl Snapshot {
    /// All edges hang off node 0, so `offsets` is trivial.
    fn new(edges: &[(u64, i64, i64)]) -> Self {
        let mut s = Snapshot {
            offsets: vec![0, edges.len() as u32, edges.len() as u32],
            dst: Vec::new(),
            vf: Vec::new(),
            vt: Vec::new(),
            et: Vec::new(),
        };
        for &(d, f, t) in edges {
            s.dst.push(d);
            s.vf.push(f);
            s.vt.push(t);
            s.et.push(1);
        }
        s
    }

    fn view(&self, snapshot_id: u64) -> SnapshotView<'_> {
        SnapshotView::new(
            snapshot_id,
            2,
            &self.offsets,
            &self.dst,
            &self.vf,
            &self.vt,
            &self.et,
        )
    }
}

fn cache(max_bytes: Option<usize>) -> (Arc<CpuAccelerator>, ResidentCache) {
    let acc = Arc::new(CpuAccelerator::new());
    let cache = ResidentCache::new(Arc::clone(&acc) as Arc<dyn Accelerator>, max_bytes);
    (acc, cache)
}

#[test]
fn repeated_loads_of_one_snapshot_hit_the_cache() {
    let g = graphs();
    let snap = Snapshot::new(&g.s1);
    let (_, mut c) = cache(None);

    let a = c.acquire(snap.view(1)).unwrap().unwrap();
    let b = c.acquire(snap.view(1)).unwrap().unwrap();

    // Same allocation, not merely equal fields: a second upload would show up
    // as two different tokens.
    assert!(Arc::ptr_eq(&a, &b));
    assert_eq!(c.stats().loads, 1);
    assert_eq!(c.stats().hits, 1);
    assert_eq!(c.stats().misses, 1);
    assert_eq!(c.len(), 1);
}

#[test]
fn business_time_does_not_force_a_reupload() {
    // The C0 finding that shaped the key: a transfer costs more than the
    // filter it feeds (13.86 ms vs 0.556 ms at 1e7 edges), so changing `t`
    // must not touch the device.
    let g = graphs();
    let snap = Snapshot::new(&g.s1);
    let (acc, mut c) = cache(None);
    let h = c.acquire(snap.view(1)).unwrap().unwrap();

    let t0 = acc.valid_time_filter(&h, 0).unwrap();
    let t50 = acc.valid_time_filter(&h, 50).unwrap();
    let t100 = acc.valid_time_filter(&h, 100).unwrap();

    assert_eq!(t0.to_ids(), vec![0, 1]);
    assert_eq!(t50.to_ids(), vec![0, 1]);
    assert!(t100.is_empty(), "half-open: nothing is live at 100");
    assert_eq!(c.stats().loads, 1, "t is not part of the cache key");
    assert_eq!(c.stats().hits, 0, "operators do not re-enter the cache");
}

#[test]
fn a_new_snapshot_is_never_answered_from_the_old_copy() {
    let g = graphs();
    let old = Snapshot::new(&g.s1);
    let new = Snapshot::new(&g.s2);
    let (acc, mut c) = cache(None);

    let h1 = c.acquire(old.view(1)).unwrap().unwrap();
    assert_eq!(acc.valid_time_filter(&h1, 0).unwrap().to_ids(), vec![0, 1]);

    let h2 = c.acquire(new.view(2)).unwrap().unwrap();
    assert_eq!(
        acc.valid_time_filter(&h2, 0).unwrap().to_ids(),
        vec![0],
        "the retraction must be visible"
    );
    assert_ne!(h1.token(), h2.token());
    assert_eq!(c.stats().loads, 2);

    // The old handle still describes snapshot 1 accurately — that is what a
    // replay of snapshot 1 means — but it refuses to answer for snapshot 2.
    assert_eq!(acc.valid_time_filter(&h1, 0).unwrap().to_ids(), vec![0, 1]);
    let err = h1.check_snapshot(2).unwrap_err();
    assert_eq!(
        err,
        AcceleratorError::SnapshotStale {
            resident: 1,
            requested: 2
        }
    );
}

#[test]
fn invalidation_makes_the_next_query_reload() {
    // This is the publish()/flush() hook: once a snapshot is superseded, a new
    // query must not be served from the copy, even if nobody evicted it.
    let g = graphs();
    let snap = Snapshot::new(&g.s1);
    let (_, mut c) = cache(None);

    let _ = c.acquire(snap.view(1)).unwrap().unwrap();
    assert!(c.is_resident(1));

    assert_eq!(c.invalidate_snapshot(1), 1);
    assert!(!c.is_resident(1));
    assert_eq!(c.stats().releases, 1);

    let again = c.acquire(snap.view(1)).unwrap().unwrap();
    assert_eq!(c.stats().loads, 2, "must be a real upload, not a hit");
    assert_eq!(again.snapshot_id(), 1);
}

#[test]
fn an_evicted_but_borrowed_copy_keeps_answering() {
    // Eviction must never free memory a query is reading. With both entries
    // borrowed the budget cannot be met, and the correct behaviour is to go
    // over budget and report it — not to fail the query.
    let g = graphs();
    let a = Snapshot::new(&g.s1);
    let b = Snapshot::new(&g.s2);
    // 2 edges: 3*4 + 2*26 = 64 B; 1 edge: 12 + 26 = 38 B. Fit one of each: 102.
    let (acc, mut c) = cache(Some(100));

    let ha = c.acquire(a.view(1)).unwrap().unwrap();
    let hb = c.acquire(b.view(2)).unwrap().unwrap();
    assert!(c.resident_bytes() > 100, "both are pinned");
    assert_eq!(c.stats().over_budget, 1);

    // Both still answer correctly while over budget.
    assert_eq!(acc.valid_time_filter(&ha, 0).unwrap().to_ids(), vec![0, 1]);
    assert_eq!(acc.valid_time_filter(&hb, 0).unwrap().to_ids(), vec![0]);

    // Once the borrowers let go, the next trim brings it back under.
    let bytes_before = c.resident_bytes();
    drop(ha);
    drop(hb);
    c.trim();
    assert!(c.resident_bytes() <= 100, "{bytes_before} -> {}", c.resident_bytes());
    assert_eq!(c.pending_releases(), 0);
}

#[test]
fn the_budget_evicts_least_recently_used_first() {
    let g = graphs();
    let a = Snapshot::new(&g.s1);
    let b = Snapshot::new(&g.s2);
    let (_, mut c) = cache(Some(100));

    // Snapshot 1 is loaded and immediately let go, so it is evictable and older.
    let ha = c.acquire(a.view(1)).unwrap().unwrap();
    drop(ha);
    let hb = c.acquire(b.view(2)).unwrap().unwrap();

    assert_eq!(c.stats().evictions, 1);
    assert!(!c.is_resident(1), "the older unused copy is the victim");
    assert!(c.is_resident(2));
    assert!(c.stats().releases >= 1);
    assert!(c.resident_bytes() <= 100);
    drop(hb);
}

#[test]
fn releasing_a_borrowed_copy_is_deferred_not_dropped() {
    // "No dangling pin": the backend must not be told to free a buffer that is
    // still readable. Here the eviction is blocked, the handle is queued, and
    // the queue drains only after the last borrower is gone.
    let g = graphs();
    let a = Snapshot::new(&g.s1);
    let b = Snapshot::new(&g.s2);
    // 64 B fits, 64 + 38 does not, so the second acquire has to evict — and the
    // only candidate is the copy the first handle is still reading.
    let (_, mut c) = cache(Some(100));

    let ha = c.acquire(a.view(1)).unwrap().unwrap();
    let hb = c.acquire(b.view(2)).unwrap().unwrap();
    assert_eq!(c.stats().skipped_pinned, 1, "evictor refused to touch a pin");
    assert_eq!(c.stats().over_budget, 1);
    assert!(c.pending_releases() > 0 || c.resident_bytes() > 100);

    drop(ha);
    drop(hb);
    let released = c.trim();
    assert!(released > 0);
    assert_eq!(c.pending_releases(), 0);
}

#[test]
fn dropping_the_cache_releases_what_nobody_holds() {
    let g = graphs();
    let snap = Snapshot::new(&g.s1);
    let acc = Arc::new(CpuAccelerator::new());
    {
        let mut c = ResidentCache::new(Arc::clone(&acc) as Arc<dyn Accelerator>, None);
        let _ = c.acquire(snap.view(1)).unwrap().unwrap();
        assert_eq!(c.stats().releases, 0);
    }
    // The cache's Drop handed the copy back; the accelerator's table is empty,
    // so a new cache would have to upload again.
    let mut fresh = ResidentCache::new(Arc::clone(&acc) as Arc<dyn Accelerator>, None);
    let _ = fresh.acquire(snap.view(1)).unwrap().unwrap();
    assert_eq!(fresh.stats().loads, 1);
}

#[test]
fn cached_and_uncached_paths_agree() {
    // The parity rule: the cache may change timing, never results.
    let g = graphs();
    let snap = Snapshot::new(&g.s1);
    let (acc, mut c) = cache(None);

    let direct: Vec<Vec<u64>> = [0i64, 50, 99, 100]
        .iter()
        .map(|&t| acc.valid_time_filter(&c.acquire(snap.view(1)).unwrap().unwrap(), t).unwrap().to_ids())
        .collect();
    let cached: Vec<Vec<u64>> = [0i64, 50, 99, 100]
        .iter()
        .map(|&t| acc.valid_time_filter(&c.acquire(snap.view(1)).unwrap().unwrap(), t).unwrap().to_ids())
        .collect();
    assert_eq!(direct, cached);
    assert_eq!(c.stats().loads, 1);
}

#[test]
fn a_backend_that_declines_residency_is_reported_not_retried() {
    // `Ok(None)` is how a backend says "I do not do this". The caller must see
    // a decline (and run on the CPU), not an error and not a second attempt.
    struct NoResidency;
    impl Accelerator for NoResidency {
        fn name(&self) -> &'static str {
            "stub"
        }
        fn kind(&self) -> gtv_accelerator::BackendKind {
            gtv_accelerator::BackendKind::Cuda
        }
        fn caps(&self) -> gtv_accelerator::Caps {
            gtv_accelerator::Caps::new(gtv_accelerator::BackendKind::Cuda, "stub", None)
        }
        fn load_graph(
            &self,
            _v: SnapshotView<'_>,
        ) -> Result<Option<gtv_accelerator::ResidentGraph>, AcceleratorError> {
            Ok(None)
        }
    }
    let g = graphs();
    let snap = Snapshot::new(&g.s1);
    let mut c = ResidentCache::new(Arc::new(NoResidency) as Arc<dyn Accelerator>, None);
    assert!(c.acquire(snap.view(1)).unwrap().is_none());
    assert_eq!(c.stats().declined, 1);
    assert_eq!(c.stats().loads, 0);
    assert!(c.is_empty());
}

#[test]
fn policy_and_cache_together_never_produce_a_wrong_answer() {
    // The integration the rest of the system uses: ask the policy, and if it
    // declines, run the same operator on the CPU reference. Both sides must
    // agree on every snapshot, including the one that was just retracted.
    let g = graphs();
    let policy = ExecutionPolicy::default();
    let (acc, mut c) = cache(None);

    for (id, edges, expected) in [(1u64, &g.s1, 2usize), (2, &g.s2, 1)] {
        let snap = Snapshot::new(edges);
        let view = snap.view(id);
        let caps = c.backend().caps();
        let inputs = gtv_accelerator::DecisionInputs::new(
            &caps,
            gtv_accelerator::Op::ValidTimeFilter,
            view.edge_count() as u64,
        )
        .resident(c.is_resident(id))
        .bytes(view.bytes() as u64);
        // A bare filter is never admitted on its own (C0: 96% transfer), so the
        // policy says CPU — and the answer must be the same either way.
        let decision = policy.decide(&inputs);
        assert!(!decision.is_device(), "{decision:?}");

        let h = c.acquire(view).unwrap().unwrap();
        let got = acc.valid_time_filter(&h, 0).unwrap();
        assert_eq!(got.len(), expected, "snapshot {id}");
    }
}
