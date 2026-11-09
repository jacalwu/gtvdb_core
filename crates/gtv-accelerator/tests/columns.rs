//! Column projection: what a copy contains, what it costs, and what it may answer.
//!
//! These run on the CPU backend because a projection is a *host-side* decision
//! taken before any byte moves. The device only ever sees the consequence —
//! fewer buffers — so the rules that must be right are the pricing, the
//! validation and the cache key, and all three are testable without a GPU.
//!
//! The numbers here are the reason the feature exists. On a one-interval graph
//! with out-degree 4 the full copy is 23 B/edge of which 16 is `valid_from` +
//! `valid_to`; a neighbour scan reads neither.

use std::sync::Arc;

use gtv_accelerator::{
    Accelerator, BackendKind, Caps, CpuAccelerator, GraphLoadMetrics, ResidentCache, SnapshotView,
};
use gtv_accelerator::{AcceleratorError, Column, ColumnSet, DstLayout, Fallback};

/// 8 edges over 2 nodes, all hanging off node 0, `dst` available at both widths.
struct Fixture {
    offsets: Vec<u32>,
    dst: Vec<u64>,
    dst32: Vec<u32>,
    vf: Vec<i64>,
    vt: Vec<i64>,
    et: Vec<u16>,
}

impl Fixture {
    fn new() -> Self {
        let n = 8;
        Self {
            offsets: vec![0, n as u32, n as u32],
            dst: (0..n as u64).collect(),
            dst32: (0..n as u32).collect(),
            vf: vec![0; n as usize],
            vt: vec![100; n as usize],
            et: vec![1; n as usize],
        }
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
        .with_dst32(&self.dst32)
    }
}

#[test]
fn topology_costs_a_quarter_of_the_full_copy() {
    let f = Fixture::new();
    let full = f.view(1);
    let hot = f.view(1).with_columns(ColumnSet::TOPOLOGY);

    // offsets (3 nodes x 4 B) + dst (8 edges x 4 B)
    assert_eq!(hot.bytes(), 12 + 32);
    // + valid_from, valid_to (8 B each) + edge_type (2 B)
    assert_eq!(full.bytes(), 12 + 32 + 8 * (8 + 8 + 2));
    assert_eq!(hot.deferred_bytes(), 8 * (8 + 8 + 2));
    assert_eq!(hot.bytes() + hot.deferred_bytes(), full.bytes());

    // The claim in bytes per edge, which is what decides whether a graph fits:
    // 5.5 B/edge projected against 23.5 B/edge complete on this shape.
    let edges = full.edge_count() as f64;
    assert!((hot.bytes() as f64 / edges - 5.5).abs() < 1e-9);
    assert!((full.bytes() as f64 / edges - 23.5).abs() < 1e-9);
}

#[test]
fn temporal_is_the_column_that_costs_the_most() {
    let f = Fixture::new();
    let edges = f.view(1).edge_count();

    let temporal = ColumnSet::TOPOLOGY
        .union(ColumnSet::TEMPORAL)
        .bytes(2, edges, DstLayout::U32)
        - ColumnSet::TOPOLOGY.bytes(2, edges, DstLayout::U32);
    let labels = ColumnSet::TOPOLOGY
        .union(ColumnSet::LABELS)
        .bytes(2, edges, DstLayout::U32)
        - ColumnSet::TOPOLOGY.bytes(2, edges, DstLayout::U32);

    assert_eq!(temporal, edges * 16);
    assert_eq!(labels, edges * 2);
    assert!(
        temporal > labels * 4,
        "time columns dominate an upload; the plan's 69.6% share depends on it"
    );
}

#[test]
fn a_projection_without_topology_is_refused() {
    let f = Fixture::new();
    let err = f
        .view(1)
        .with_columns(ColumnSet::TEMPORAL)
        .validate()
        .unwrap_err()
        .to_string();
    // Not just "invalid": the message must name what is missing, because the
    // caller asked for something that is not a graph.
    assert!(err.contains("offsets"), "{err}");
    assert!(err.contains("dst"), "{err}");
}

#[test]
fn a_projection_does_not_need_the_arrays_it_left_behind() {
    // The point of projecting: a caller that only wants topology should not be
    // forced to hold — or even borrow — five columns. Empty slices are legal
    // when the projection says they are not being uploaded.
    let offsets = vec![0u32, 2, 2];
    let dst32 = vec![1u32, 2];
    let view = SnapshotView::new(1, 2, &offsets, &[], &[], &[], &[])
        .with_dst32(&dst32)
        .with_columns(ColumnSet::TOPOLOGY);

    view.validate().unwrap();
    assert_eq!(view.bytes(), 12 + 8);
}

#[test]
fn a_narrow_projection_is_cheaper_than_a_wide_one() {
    let f = Fixture::new();
    // Same projection, both widths of `dst`: the width is the one column whose
    // price is a decision rather than a fact.
    let narrow = SnapshotView::new(1, 2, &f.offsets, &[], &f.vf, &f.vt, &f.et)
        .with_dst32(&f.dst32)
        .with_columns(ColumnSet::TOPOLOGY);
    let wide = SnapshotView::new(1, 2, &f.offsets, &f.dst, &f.vf, &f.vt, &f.et)
        .with_columns(ColumnSet::TOPOLOGY);

    assert_eq!(wide.bytes() - narrow.bytes(), f.dst.len() * 4);
}

#[test]
fn the_cache_keys_projections_separately() {
    let f = Fixture::new();
    let acc = Arc::new(CpuAccelerator::new());
    let mut cache = ResidentCache::new(Arc::clone(&acc) as Arc<dyn Accelerator>, None);

    let topo = cache
        .acquire(f.view(1).with_columns(ColumnSet::TOPOLOGY))
        .unwrap()
        .unwrap();
    assert_eq!(topo.columns(), ColumnSet::TOPOLOGY);

    // Same snapshot, more columns: a hit here would hand a kernel a pointer to
    // an array that was never uploaded.
    let full = cache.acquire(f.view(1)).unwrap().unwrap();
    assert_eq!(
        cache.stats().loads,
        2,
        "a wider request is a different copy"
    );
    assert_ne!(topo.token(), full.token());
    assert_eq!(full.columns(), ColumnSet::ALL);

    // And each projection is still cached on its own.
    let again = cache
        .acquire(f.view(1).with_columns(ColumnSet::TOPOLOGY))
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&topo, &again));
    assert_eq!(cache.stats().loads, 2);
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.len(), 2, "two projections coexist, they do not evict");
}

#[test]
fn a_narrow_copy_refuses_to_answer_a_wider_read() {
    let f = Fixture::new();
    let acc = CpuAccelerator::new();
    let handle = acc_handle(&acc, f.view(1).with_columns(ColumnSet::TOPOLOGY));

    handle.check_columns(ColumnSet::TOPOLOGY).unwrap();
    let err = handle
        .check_columns(ColumnSet::ALL)
        .unwrap_err()
        .to_string();
    assert!(err.contains("valid_from"), "{err}");
    // The message names both what it has and what is missing: an operator
    // falling back to the CPU needs to know which second upload to ask for.
    assert!(err.contains("offsets+dst"), "{err}");
}

#[test]
fn accounting_still_closes_on_a_projection() {
    // `priced_bytes == uploaded_bytes` was the ratchet that caught a backend
    // allocating something other than what it charged. A projection changes
    // both sides, not their equality.
    let f = Fixture::new();
    let view = f.view(1).with_columns(ColumnSet::TOPOLOGY);
    let mut m = GraphLoadMetrics::new(1, view.node_count, view.edge_count(), view.dst_layout());
    m.projection = view.columns;
    m.deferred_bytes = view.deferred_bytes();
    m.priced_bytes = view.bytes();
    m.uploaded_bytes = view.bytes();
    for c in view.columns.iter() {
        m.record_column(c, 0, 0, 0);
    }

    assert!(m.accounting_holds());
    assert!(m.is_projected());
    assert_eq!(m.column(Column::ValidFrom).bytes, 0, "never uploaded");
    assert!((m.deferred_share() - 144.0 / 188.0).abs() < 1e-9);
}

#[test]
fn column_set_names_are_stable_and_ordered() {
    assert_eq!(ColumnSet::TOPOLOGY.names(), "offsets+dst");
    assert_eq!(
        ColumnSet::ALL.names(),
        "offsets+dst+valid_from+valid_to+edge_type"
    );
    assert_eq!(ColumnSet::EMPTY.names(), "none");
    assert_eq!(
        ColumnSet::TOPOLOGY.missing(ColumnSet::ALL).names(),
        "valid_from+valid_to+edge_type"
    );
    assert!(ColumnSet::ALL.missing(ColumnSet::TOPOLOGY).is_empty());
    assert!(ColumnSet::ALL.missing(ColumnSet::ALL).is_empty());
    assert_eq!(ColumnSet::ALL.len(), 5);
}

#[test]
fn an_oversized_copy_is_refused_before_it_is_allocated() {
    // 4.8 GiB of a 8 GiB card, the measured default: one 1e9-edge copy at 23
    // B/edge is refused with both numbers, and the fallback verdict says the
    // answer is still computable on the CPU.
    let limit = 4_800_000_000usize;
    let caps = Caps::new(BackendKind::Cuda, "test", Some(limit)).with_resident(true);
    caps.admit(limit).unwrap();

    let err = caps.admit(limit + 1).unwrap_err();
    assert_eq!(
        err,
        AcceleratorError::OutOfBudget {
            needed: limit + 1,
            limit
        }
    );
    assert_eq!(err.fallback(), Fallback::Cpu);

    // The CPU reference has no budget, and saying so must not be `usize::MAX`
    // bytes of allocation: `None` means unbounded, not "the largest number".
    Caps::cpu().admit(usize::MAX).unwrap();
}

fn acc_handle(acc: &CpuAccelerator, view: SnapshotView<'_>) -> gtv_accelerator::ResidentGraph {
    acc.load_graph(view).unwrap().unwrap()
}
