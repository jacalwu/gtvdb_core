//! The CPU reference backend: always available, always the answer of record.
//!
//! This is not a fallback in the sense of "worse but present". It is the
//! definition every other backend is measured against, so it deliberately
//! reuses the existing kernels (`FlatIndex` for exact KNN, `gtv-array` for
//! rolling sums) instead of reimplementing them — a port that disagrees with
//! this backend is wrong by definition.
//!
//! It also implements residency, even though it does not need one: sharing the
//! lifecycle with a device backend is what makes the cache and invalidation
//! tests meaningful on a machine with no GPU.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use gtv_core::{Metric, Sel, VectorHit, VectorIndex};
use gtv_index::flat::FlatIndex;

use crate::api::{
    Accelerator, BackendKind, Caps, ResidentGraph, ResidentVectors, SnapshotView, VectorView,
    WindowOp,
};
use crate::error::AcceleratorError;

/// A CPU-side copy of one snapshot's columns.
struct HostGraph {
    snapshot_id: u64,
    node_count: usize,
    offsets: Vec<u32>,
    dst: Vec<u64>,
    valid_from: Vec<i64>,
    valid_to: Vec<i64>,
}

struct HostVectors {
    snapshot_id: u64,
    space: u32,
    index: FlatIndex,
}

fn new_token() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Half-open validity: `[valid_from, valid_to)`, the convention every other
/// module uses. `valid_to == OPEN` never expires.
#[inline]
pub fn live_at(valid_from: i64, valid_to: i64, t: i64) -> bool {
    valid_from <= t && t < valid_to
}

/// Trailing-window aggregate with a cumulative prefix, matching `msum` / `mavg`
/// (kdb semantics: the first `window-1` rows aggregate over what exists).
///
/// Kept `pub` because C2's device kernel has to reproduce it exactly, and the
/// cheapest way to say that is one shared definition.
pub fn window_reduce(values: &[f64], window: usize, op: WindowOp) -> Vec<f64> {
    if window == 0 {
        return Vec::new();
    }
    match op {
        WindowOp::Sum => gtv_array::window::msum(values, window),
        WindowOp::Mean => gtv_array::window::mavg(values, window),
        WindowOp::Min => sliding(values, window, |acc, v| {
            if v < acc {
                v
            } else {
                acc
            }
        }),
        WindowOp::Max => sliding(values, window, |acc, v| {
            if v > acc {
                v
            } else {
                acc
            }
        }),
        WindowOp::Count => (0..values.len())
            .map(|i| (i + 1).min(window) as f64)
            .collect(),
    }
}

/// O(n * window) reference for the two ops with no existing kernel. Min/max are
/// not idempotent under subtraction, so a running total cannot be used; when a
/// hot path needs this, it is a monotonic deque, not a SIMD tweak.
fn sliding(values: &[f64], window: usize, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let start = (i + 1).saturating_sub(window);
        let mut acc = values[start];
        for &v in &values[start + 1..=i] {
            acc = f(acc, v);
        }
        out.push(acc);
    }
    out
}

/// The CPU backend.
#[derive(Default)]
pub struct CpuAccelerator {
    graphs: Mutex<HashMap<u64, HostGraph>>,
    vectors: Mutex<HashMap<u64, HostVectors>>,
}

impl CpuAccelerator {
    pub fn new() -> Self {
        Self::default()
    }

    fn graph(&self) -> Result<std::sync::MutexGuard<'_, HashMap<u64, HostGraph>>, AcceleratorError> {
        let guard = self.graphs.lock().map_err(|_| AcceleratorError::Backend {
            backend: "cpu",
            op: "lock",
            code: 0,
            message: "graph table poisoned".into(),
        })?;
        Ok(guard)
    }
}

impl Accelerator for CpuAccelerator {
    fn name(&self) -> &'static str {
        "cpu-reference"
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn caps(&self) -> Caps {
        Caps::cpu()
    }

    fn load_graph(&self, view: SnapshotView<'_>) -> Result<Option<ResidentGraph>, AcceleratorError> {
        view.validate()?;
        let bytes = view.bytes();
        let handle = ResidentGraph::new(
            new_token(),
            view.snapshot_id,
            view.node_count,
            view.edge_count(),
            bytes,
        )
        .with_columns(view.columns);
        let host = HostGraph {
            snapshot_id: view.snapshot_id,
            node_count: view.node_count,
            offsets: view.offsets.to_vec(),
            dst: view.dst.to_vec(),
            valid_from: view.valid_from.to_vec(),
            valid_to: view.valid_to.to_vec(),
        };
        self.graphs
            .lock()
            .map_err(|_| AcceleratorError::InvalidArgument("graph table poisoned".into()))?
            .insert(handle.token(), host);
        Ok(Some(handle))
    }

    fn release_graph(&self, handle: &ResidentGraph) {
        if let Ok(mut g) = self.graphs.lock() {
            g.remove(&handle.token());
        }
    }

    fn load_vectors(
        &self,
        view: VectorView<'_>,
    ) -> Result<Option<ResidentVectors>, AcceleratorError> {
        view.validate()?;
        let index = FlatIndex::from_flat_metric(
            view.ids.to_vec(),
            view.data.to_vec(),
            view.dim,
            view.metric,
        )
        .map_err(AcceleratorError::from)?;
        let handle = ResidentVectors::new(
            new_token(),
            view.vector_space_id,
            view.snapshot_id,
            view.rows(),
            view.dim,
            view.metric,
            view.bytes(),
        );
        self.vectors
            .lock()
            .map_err(|_| AcceleratorError::InvalidArgument("vector table poisoned".into()))?
            .insert(
                handle.token(),
                HostVectors {
                    snapshot_id: view.snapshot_id,
                    space: view.vector_space_id,
                    index,
                },
            );
        Ok(Some(handle))
    }

    fn release_vectors(&self, handle: &ResidentVectors) {
        if let Ok(mut v) = self.vectors.lock() {
            v.remove(&handle.token());
        }
    }

    fn valid_time_filter(
        &self,
        handle: &ResidentGraph,
        t: i64,
    ) -> Result<Sel, AcceleratorError> {
        let guard = self.graph()?;
        let g = guard.get(&handle.token()).ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "resident graph {} is not loaded (released?)",
                handle.token()
            ))
        })?;
        // Second guard: the handle and the stored copy must agree about the
        // snapshot. A token reused after a release would otherwise answer from
        // whatever now occupies that slot.
        handle.check_snapshot(g.snapshot_id)?;
        // Edge ids *are* slot ids, so the candidate set indexes the same space
        // the CSR does and can be handed straight to k-hop or an attribute
        // join without a translation table.
        let mut ids = Vec::new();
        for v in 0..g.node_count {
            let start = g.offsets[v] as usize;
            let end = g.offsets[v + 1] as usize;
            for slot in start..end {
                if live_at(g.valid_from[slot], g.valid_to[slot], t) {
                    ids.push(slot as u64);
                }
            }
        }
        let universe = g.dst.len() as u64;
        Sel::from_ids(g.snapshot_id, universe, ids).map_err(AcceleratorError::from)
    }

    fn knn_flat(
        &self,
        handle: &ResidentVectors,
        queries: &[f32],
        k: usize,
        filter: Option<&Sel>,
    ) -> Result<Vec<Vec<VectorHit>>, AcceleratorError> {
        if handle.dim() == 0 || !queries.len().is_multiple_of(handle.dim()) {
            return Err(AcceleratorError::InvalidArgument(format!(
                "{} query floats is not a multiple of dim {}",
                queries.len(),
                handle.dim()
            )));
        }
        if let Some(f) = filter {
            // A candidate set from another snapshot would silently select the
            // wrong rows here too, so the same check as k-hop applies.
            if f.is_bound() {
                handle.check_snapshot(f.snapshot_id(), handle.vector_space_id())?;
            }
            if f.universe() != handle.rows() as u64 {
                return Err(AcceleratorError::InvalidArgument(format!(
                    "filter covers {} rows, space has {}",
                    f.universe(),
                    handle.rows()
                )));
            }
        }
        let guard = self.vectors.lock().map_err(|_| {
            AcceleratorError::InvalidArgument("vector table poisoned".into())
        })?;
        let v = guard.get(&handle.token()).ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "resident vectors {} are not loaded (released?)",
                handle.token()
            ))
        })?;
        handle.check_snapshot(v.snapshot_id, v.space)?;
        let mask = filter.map(|f| f.to_mask());
        let mut out = Vec::with_capacity(queries.len() / handle.dim());
        for q in queries.chunks_exact(handle.dim()) {
            out.push(
                v.index
                    .search(q, k, mask.as_ref())
                    .map_err(AcceleratorError::from)?,
            );
        }
        Ok(out)
    }

    fn batch_khop(
        &self,
        handle: &ResidentGraph,
        seeds: &Sel,
        hops: u8,
        at: i64,
    ) -> Result<Sel, AcceleratorError> {
        if seeds.is_bound() {
            handle.check_snapshot(seeds.snapshot_id())?;
        }
        let guard = self.graph()?;
        let g = guard.get(&handle.token()).ok_or_else(|| {
            AcceleratorError::InvalidArgument(format!(
                "resident graph {} is not loaded (released?)",
                handle.token()
            ))
        })?;
        handle.check_snapshot(g.snapshot_id)?;
        if seeds.universe() != g.node_count as u64 {
            return Err(AcceleratorError::InvalidArgument(format!(
                "seed set covers {} nodes, graph has {}",
                seeds.universe(),
                g.node_count
            )));
        }

        let mut visited = vec![false; g.node_count];
        let mut frontier: Vec<u64> = Vec::new();
        for s in seeds.iter() {
            if s < g.node_count as u64 && !visited[s as usize] {
                visited[s as usize] = true;
                frontier.push(s);
            }
        }
        for _ in 0..hops {
            if frontier.is_empty() {
                break;
            }
            let mut next = Vec::new();
            for &v in &frontier {
                let start = g.offsets[v as usize] as usize;
                let end = g.offsets[v as usize + 1] as usize;
                for slot in start..end {
                    if !live_at(g.valid_from[slot], g.valid_to[slot], at) {
                        continue;
                    }
                    let d = g.dst[slot];
                    let di = d as usize;
                    if di < visited.len() && !visited[di] {
                        visited[di] = true;
                        next.push(d);
                    }
                }
            }
            frontier = next;
        }
        // Seeds stay out of the answer, matching `TemporalCSR::khop`.
        let mut reached: Vec<u64> = Vec::new();
        for (i, seen) in visited.iter().enumerate() {
            if *seen && !seeds.contains(i as u64) {
                reached.push(i as u64);
            }
        }
        Sel::from_ids(g.snapshot_id, g.node_count as u64, reached).map_err(AcceleratorError::from)
    }

    fn window_reduce(
        &self,
        values: &[f64],
        window: usize,
        op: WindowOp,
    ) -> Result<Vec<f64>, AcceleratorError> {
        Ok(window_reduce(values, window, op))
    }
}

/// Only used by tests and the bench example to assert the two encodings agree.
pub fn metric_of(index: &FlatIndex) -> Metric {
    index.metric()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{BackendKind, Op};

    type Columns = (Vec<u32>, Vec<u64>, Vec<i64>, Vec<i64>, Vec<u16>);

    fn graph() -> Columns {
        // 0 -> 1 (@[0,100)), 0 -> 2 (@[50,200)), 1 -> 2 (@[0,100)), 2 -> 0 (@[0,100))
        (
            vec![0, 2, 3, 4],
            vec![1, 2, 2, 0],
            vec![0, 50, 0, 0],
            vec![100, 200, 100, 100],
            vec![1, 1, 1, 1],
        )
    }

    fn loaded(acc: &CpuAccelerator) -> ResidentGraph {
        let (o, d, vf, vt, et) = graph();
        let view = SnapshotView::new(9, 3, &o, &d, &vf, &vt, &et);
        acc.load_graph(view).unwrap().unwrap()
    }

    #[test]
    fn filter_returns_edge_slots_live_at_t() {
        let acc = CpuAccelerator::new();
        let h = loaded(&acc);

        // At t=0: slots 0 (0->1), 2 (1->2), 3 (2->0). Slot 1 starts at 50.
        let s = acc.valid_time_filter(&h, 0).unwrap();
        assert_eq!(s.to_ids(), vec![0, 2, 3]);
        assert_eq!(s.snapshot_id(), 9);

        // At t=50 the second parallel edge opens.
        let s = acc.valid_time_filter(&h, 50).unwrap();
        assert_eq!(s.to_ids(), vec![0, 1, 2, 3]);

        // At t=100 the three [0,100) edges have closed; only the long one
        // (0 -> 2, [50,200)) is still live. Intervals are half-open, so 100
        // belongs to the second edge and not to the first three.
        let s = acc.valid_time_filter(&h, 100).unwrap();
        assert_eq!(s.to_ids(), vec![1]);
        assert!(acc.valid_time_filter(&h, 200).unwrap().is_empty());
    }

    #[test]
    fn khop_matches_csr_convention() {
        let acc = CpuAccelerator::new();
        let h = loaded(&acc);
        let seeds = Sel::from_ids(9, 3, [0u64]).unwrap();
        // 0 -> {1} at t=0 (2 only opens at 50).
        let one = acc.batch_khop(&h, &seeds, 1, 0).unwrap();
        assert_eq!(one.to_ids(), vec![1]);
        // Two hops from 0: 1 -> 2. Seeds (0) excluded, even though 2 -> 0 closes a ring.
        let two = acc.batch_khop(&h, &seeds, 2, 0).unwrap();
        assert_eq!(two.to_ids(), vec![1, 2]);
        // Zero hops is empty, not the seed set.
        assert!(acc.batch_khop(&h, &seeds, 0, 0).unwrap().is_empty());
    }

    #[test]
    fn khop_rejects_seeds_from_another_snapshot() {
        let acc = CpuAccelerator::new();
        let h = loaded(&acc);
        let stale = Sel::from_ids(8, 3, [0u64]).unwrap();
        let err = acc.batch_khop(&h, &stale, 1, 0).unwrap_err();
        assert!(matches!(
            err,
            AcceleratorError::SnapshotStale {
                resident: 9,
                requested: 8
            }
        ));
    }

    #[test]
    fn knn_filter_is_a_candidate_list_not_a_post_filter() {
        let acc = CpuAccelerator::new();
        // Four rows in 2-D; row i sits at (i, 0).
        let data: Vec<f32> = vec![0., 0., 1., 0., 2., 0., 3., 0.];
        let ids: Vec<u64> = vec![10, 11, 12, 13];
        let v = acc
            .load_vectors(VectorView::new(1, 9, 2, Metric::L2, &data, &ids))
            .unwrap()
            .unwrap();
        let all = acc.knn_flat(&v, &[0., 0.], 2, None).unwrap();
        assert_eq!(
            all[0].iter().map(|h| h.id).collect::<Vec<_>>(),
            vec![10, 11]
        );

        // Excluding the two nearest must surface the two further ones.
        let allow = Sel::from_ids(9, 4, [2u64, 3]).unwrap();
        let got = acc.knn_flat(&v, &[0., 0.], 2, Some(&allow)).unwrap();
        assert_eq!(got[0].iter().map(|h| h.id).collect::<Vec<_>>(), vec![12, 13]);
        // A filter that empties the candidate set returns nothing, not an error.
        let none = Sel::empty(9, 4);
        assert!(acc.knn_flat(&v, &[0., 0.], 2, Some(&none)).unwrap()[0].is_empty());
    }

    #[test]
    fn knn_batches_split_queries_by_dim() {
        let acc = CpuAccelerator::new();
        let data: Vec<f32> = vec![0., 0., 5., 5.];
        let ids: Vec<u64> = vec![1, 2];
        let v = acc
            .load_vectors(VectorView::new(1, 9, 2, Metric::L2, &data, &ids))
            .unwrap()
            .unwrap();
        let out = acc.knn_flat(&v, &[0., 0., 5., 5.], 1, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0][0].id, 1);
        assert_eq!(out[1][0].id, 2);
        // A ragged batch is a caller bug, not a device fault.
        let err = acc.knn_flat(&v, &[0., 0., 5.], 1, None).unwrap_err();
        assert_eq!(err.fallback(), crate::Fallback::Surface);
    }

    #[test]
    fn window_ops_agree_with_the_kdb_prefix_rule() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(window_reduce(&v, 2, WindowOp::Sum), vec![1., 3., 5., 7.]);
        assert_eq!(
            window_reduce(&v, 2, WindowOp::Mean),
            vec![1., 1.5, 2.5, 3.5]
        );
        assert_eq!(
            window_reduce(&v, 3, WindowOp::Min),
            vec![1., 1., 1., 2.]
        );
        assert_eq!(
            window_reduce(&v, 3, WindowOp::Max),
            vec![1., 2., 3., 4.]
        );
        assert_eq!(
            window_reduce(&v, 3, WindowOp::Count),
            vec![1., 2., 3., 3.]
        );
    }

    #[test]
    fn caps_advertise_every_implemented_op() {
        let acc = CpuAccelerator::new();
        let c = acc.caps();
        assert_eq!(c.backend, BackendKind::Cpu);
        assert_eq!(c.max_device_bytes, None);
        for op in [
            Op::ValidTimeFilter,
            Op::KnnFlat,
            Op::BatchKhop,
            Op::WindowReduce,
        ] {
            assert!(c.supports(op), "{op:?}");
        }
    }

    #[test]
    fn released_handle_is_an_error_not_silence() {
        let acc = CpuAccelerator::new();
        let h = loaded(&acc);
        acc.release_graph(&h);
        let err = acc.valid_time_filter(&h, 0).unwrap_err();
        assert_eq!(err.fallback(), crate::Fallback::Surface);
        // Releasing twice is a no-op, not a panic.
        acc.release_graph(&h);
    }
}
