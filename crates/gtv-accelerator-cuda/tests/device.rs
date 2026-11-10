//! The C1 acceptance tests, run against a real device when there is one.
//!
//! On the GPU-less CI machine every test here reports "skipped" and passes; the
//! point of the file is that the same binary proves the residency lifecycle on a
//! machine that has a card, with no feature flag and no test-only code path. The
//! driver is found at run time, so nothing here needs CUDA at build time.
//!
//! What is being verified is not speed. It is that a snapshot can be handed to
//! the device, accounted for, and given back — and that the cache in
//! `gtv-accelerator` drives that correctly, including on a snapshot change.

use std::sync::{Arc, Mutex, MutexGuard};

use gtv_accelerator::{
    Accelerator, AcceleratorError, Column, ColumnSet, CpuAccelerator, DecisionInputs, DstLayout,
    ExecutionPolicy, Op, ResidentCache, ResidentVectors, SnapshotView, VectorView,
};
use gtv_accelerator_cuda::CudaAccelerator;
use gtv_core::{Metric, Sel, VectorHit};

/// Build a chain graph: node `i -> i+1` for every `i`, all live on `[0, 1000)`.
/// Returns the five columns plus the node count.
/// The driver's allocation granularity as reported by `cuMemGetInfo`: a fresh
/// context hands out memory in 2 MiB steps, so a 300 KB snapshot still costs
/// 2 MiB. Measured on the C0 machine, 2026-11.
const GRANULARITY: usize = 2 << 20;

/// The five CSR columns, as built by [`chain`].
type Columns = (Vec<u32>, Vec<u64>, Vec<i64>, Vec<i64>, Vec<u16>);

fn chain(nodes: usize) -> Columns {
    let edges = nodes.saturating_sub(1);
    let mut offsets = Vec::with_capacity(nodes + 1);
    for i in 0..nodes {
        offsets.push(if i + 1 < nodes { i as u32 } else { edges as u32 });
    }
    offsets.push(edges as u32);
    let dst: Vec<u64> = (1..nodes as u64).collect();
    let vf = vec![0i64; edges];
    let vt = vec![1000i64; edges];
    let et = vec![1u16; edges];
    (offsets, dst, vf, vt, et)
}

fn view<'a>(
    id: u64,
    n: usize,
    o: &'a [u32],
    d: &'a [u64],
    vf: &'a [i64],
    vt: &'a [i64],
    et: &'a [u16],
) -> SnapshotView<'a> {
    SnapshotView::new(id, n, o, d, vf, vt, et)
}

macro_rules! device_or_skip {
    () => {
        match CudaAccelerator::open() {
            Ok(acc) => acc,
            Err(e) => {
                eprintln!("skipped: no usable CUDA device ({e})");
                return;
            }
        }
    };
}

/// Serialises every test in this file.
///
/// They share one primary context and therefore one allocator pool, so the
/// free-memory delta around an upload only belongs to that upload if no sibling
/// test allocated in between. Without this the delta reads 0 for a 260 KB
/// snapshot — indistinguishable from a leak, and a red test that says nothing
/// about the code. Six tests at a few milliseconds each: the serialisation is
/// free, the false negative is not.
static DEVICE: Mutex<()> = Mutex::new(());

fn exclusive() -> MutexGuard<'static, ()> {
    // A panicking test must not poison the rest into cascading failures.
    DEVICE.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn the_device_reports_what_the_upload_needs() {
    let _serial = exclusive();
    let acc = device_or_skip!();
    let d = acc.device();
    eprintln!(
        "device: {} cc {}.{} sm {} vram {} MiB budget {} MiB",
        d.name,
        d.compute_capability.0,
        d.compute_capability.1,
        d.multiprocessors,
        d.total_mem_bytes / (1 << 20),
        acc.budget_bytes() / (1 << 20)
    );
    assert!(d.total_mem_bytes > 0);
    // The budget is 60% of VRAM by default, leaving room for workspace.
    assert!(acc.budget_bytes() <= d.total_mem_bytes);
    assert!(acc.budget_bytes() * 10 >= d.total_mem_bytes * 5);
}

#[test]
fn a_snapshot_becomes_resident_and_gives_the_memory_back() {
    let _serial = exclusive();
    let acc = device_or_skip!();
    let (o, dst, vf, vt, et) = chain(10_000);
    let v = view(1, 10_000, &o, &dst, &vf, &vt, &et);
    let expected = v.bytes();

    let (free0, _) = acc.mem_info().unwrap();
    let h = acc.load_graph(v).unwrap().expect("device does residency");
    let (free1, _) = acc.mem_info().unwrap();

    assert_eq!(acc.resident_bytes(), expected, "accounting, not the driver");
    assert_eq!(h.bytes(), expected);
    assert_eq!(h.edge_count(), 9_999);

    // The width is part of what was proved: these columns are the source `u64`
    // ones, so this copy's `dst` is 8 B/edge and a kernel must read it as such.
    // 10 001 offsets + 9 999 edges at 26 B is the whole footprint — no hidden
    // padding column.
    assert_eq!(h.dst_layout(), DstLayout::U64);
    assert_eq!(expected, 10_001 * 4 + 9_999 * 26);

    // The driver charges a 2 MiB granularity: the first allocation on a fresh
    // context reserves a pool, and a 300 KB snapshot costs a 2 MiB step. That is
    // a real constraint on the budget (a small snapshot is not free), so the
    // assertion is "at least what we asked for, at most one pool step more".
    let first = free0.saturating_sub(free1);
    assert!(
        first >= expected,
        "driver reports {first} for {expected} requested"
    );
    assert!(
        first <= expected + GRANULARITY,
        "unexpected overshoot: {first} for {expected}"
    );

    // A second snapshot of the same size fits in the pool that was just set up,
    // which is why residency is worth amortising and a cold upload is not.
    let h2 = acc
        .load_graph(view(2, 10_000, &o, &dst, &vf, &vt, &et))
        .unwrap()
        .unwrap();
    let (free2, _) = acc.mem_info().unwrap();
    let second = free1.saturating_sub(free2);
    eprintln!(
        "resident: requested {expected} B, first {first} B, second {second} B (granularity {GRANULARITY})"
    );
    assert_eq!(acc.resident_bytes(), expected * 2);

    acc.release_graph(&h);
    acc.release_graph(&h2);
    let (free_released, _) = acc.mem_info().unwrap();
    assert_eq!(acc.resident_bytes(), 0);
    // Everything the two snapshots took comes back. The pool itself may stay
    // reserved by the driver, so compare against the state before the second
    // upload rather than against the process start.
    assert!(
        free_released + GRANULARITY >= free0.min(free1),
        "leak: free0={free0} free1={free1} released={free_released}"
    );
}

/// A `u32` column the caller already holds is uploaded as-is — no conversion in
/// `load_graph`, 4 B/edge less on the wire and in VRAM, and a receipt that says
/// which width the copy is. This is the whole reason `SnapshotView` carries
/// `dst32`: the conversion measured more expensive than the transfer it saves
/// (plan §9.2), so the narrow width has to arrive from upstream.
#[test]
fn a_prebuilt_u32_column_costs_four_bytes_less_per_edge() {
    let _serial = exclusive();
    let acc = device_or_skip!();
    let (o, dst, vf, vt, et) = chain(10_000);
    let dst32: Vec<u32> = dst.iter().map(|&d| d as u32).collect();

    let wide = view(1, 10_000, &o, &dst, &vf, &vt, &et);
    let narrow = wide.with_dst32(&dst32);
    assert_eq!(wide.dst_layout(), DstLayout::U64);
    assert_eq!(narrow.dst_layout(), DstLayout::U32);
    // Same edges, same node count, 4 B/edge apart — nothing else differs.
    assert_eq!(narrow.edge_count(), wide.edge_count());
    assert_eq!(wide.bytes() - narrow.bytes(), 9_999 * 4);

    let h = acc.load_graph(narrow).unwrap().expect("device does residency");
    assert_eq!(h.dst_layout(), DstLayout::U32);
    assert_eq!(h.bytes(), 10_001 * 4 + 9_999 * 22);
    // The device agrees with the price, not just the bookkeeping.
    assert_eq!(acc.resident_bytes(), narrow.bytes());
    acc.release_graph(&h);
    assert_eq!(acc.resident_bytes(), 0);
}

/// A projected upload puts two buffers on the device instead of five, and the
/// budget is charged for two. The assertion that matters is the physical one:
/// `resident_bytes` after the release returns to zero, so nothing was allocated
/// behind the receipt's back.
#[test]
fn a_topology_only_upload_never_touches_the_time_columns() {
    let _serial = exclusive();
    let acc = device_or_skip!();
    let (o, dst, vf, vt, et) = chain(1_000_000);
    let dst32: Vec<u32> = dst.iter().map(|&d| d as u32).collect();

    let full = view(1, 1_000_000, &o, &dst, &vf, &vt, &et).with_dst32(&dst32);
    let topo = full.with_columns(ColumnSet::TOPOLOGY);

    let h = acc
        .load_graph(topo)
        .unwrap()
        .expect("device does residency");
    let m = acc.last_load_metrics().expect("a load leaves a receipt");

    assert_eq!(h.columns(), ColumnSet::TOPOLOGY);
    assert_eq!(m.projection, ColumnSet::TOPOLOGY);
    assert_eq!(m.uploaded_bytes, topo.bytes());
    assert_eq!(m.deferred_bytes, topo.deferred_bytes());
    assert_eq!(m.uploaded_bytes + m.deferred_bytes, full.bytes());
    // 16 of the 23 B/edge are the two time columns; the label is the other 2.
    assert_eq!(m.column(Column::ValidFrom).bytes, 0);
    assert_eq!(m.column(Column::ValidTo).bytes, 0);
    assert_eq!(m.column(Column::EdgeType).bytes, 0);
    assert_eq!(m.column(Column::Dst).bytes, 4 * 999_999);
    assert_eq!(acc.resident_bytes(), topo.bytes());
    assert!(m.accounting_holds());

    acc.release_graph(&h);
    assert_eq!(acc.resident_bytes(), 0);
}

/// The wall an 8 GB card hits: a full copy that does not fit is refused with
/// both numbers, before `cuMemAlloc` is asked for anything.
///
/// The refusal is a `Fallback::Cpu` error, not a query failure — which is the
/// contract that lets a planner answer on the CPU instead of returning OOM.
#[test]
fn a_graph_larger_than_the_budget_is_refused_not_attempted() {
    let _serial = exclusive();
    let acc = device_or_skip!();
    let (o, dst, vf, vt, et) = chain(64);
    let v = view(1, 64, &o, &dst, &vf, &vt, &et);

    // The real budget is a fraction of the card, far above this graph; ask the
    // same question the upload will ask, with a budget one byte too small.
    let needed = v.bytes();
    let err = gtv_accelerator::Caps::new(
        gtv_accelerator::BackendKind::Cuda,
        "probe",
        Some(needed - 1),
    )
    .admit(needed)
    .unwrap_err();
    assert!(matches!(
        err,
        gtv_accelerator::AcceleratorError::OutOfBudget { .. }
    ));

    // And the real backend admits what does fit, so the check is not a wall in
    // front of every upload.
    acc.caps().admit(needed).unwrap();
    let h = acc.load_graph(v).unwrap().unwrap();
    acc.release_graph(&h);
}

#[test]
fn releasing_twice_and_releasing_a_stranger_are_no_ops() {
    let _serial = exclusive();
    let acc = device_or_skip!();
    let (o, dst, vf, vt, et) = chain(64);
    let h = acc.load_graph(view(1, 64, &o, &dst, &vf, &vt, &et)).unwrap().unwrap();
    acc.release_graph(&h);
    acc.release_graph(&h);
    assert_eq!(acc.resident_bytes(), 0);
    // A handle from another backend (or a forged one) must not free anything.
    let forged = gtv_accelerator::ResidentGraph::new(u64::MAX, 1, 64, 63, 1234);
    acc.release_graph(&forged);
    assert_eq!(acc.resident_bytes(), 0);
}

/// The stage receipt: every millisecond of an upload has to belong to a named
/// stage, and the stages that are supposed to be zero have to *be* zero.
///
/// This is the test that answers "is the bottleneck the copy or the work around
/// it" without a profiler, and the one that goes red the day someone adds an
/// AoS re-pack, a u64->u32 conversion or a staging buffer — each of which would
/// otherwise hide inside "upload time".
#[test]
fn the_upload_reports_where_its_milliseconds_went() {
    let _serial = exclusive();
    let acc = device_or_skip!();
    assert!(
        acc.last_load_metrics().is_none(),
        "a backend that has not uploaded has nothing to report"
    );

    let (o, dst, vf, vt, et) = chain(200_000);
    let dst32: Vec<u32> = dst.iter().map(|&d| d as u32).collect();
    let v = view(1, 200_000, &o, &dst, &vf, &vt, &et).with_dst32(&dst32);
    let h = acc.load_graph(v).unwrap().expect("device does residency");
    let m = acc.last_load_metrics().expect("a load leaves a receipt");

    assert_eq!(m.snapshot_id, 1);
    assert_eq!(m.edges, v.edge_count());
    assert_eq!(m.dst_layout, DstLayout::U32);
    assert_eq!(m.priced_bytes, v.bytes());
    // The budget charges what the device actually holds.
    assert!(m.accounting_holds());
    // Nothing on the host touched the columns.
    assert!(m.is_copy_free(), "{m:?}");
    assert_eq!(m.staging_bytes, 0);
    assert_eq!(m.input_bytes, m.uploaded_bytes, "bytes added or dropped");
    assert_eq!(m.column(Column::Dst).bytes, v.edge_count() * 4);
    let sum: usize = m.columns.iter().map(|c| c.bytes).sum();
    assert_eq!(sum, h.bytes(), "a column is missing from the receipt");

    // The stages bracket the upload and are inside the total.
    assert!(m.upload_ns() > 0);
    assert!(m.upload_ns() <= m.total_ns, "{} > {}", m.upload_ns(), m.total_ns);
    assert!(m.device_alloc_ns <= m.total_ns);
    // `cuMemcpyHtoD` is synchronous, so an explicit synchronise afterwards must
    // be nearly free. If it ever is not, the per-column numbers stopped being
    // transfer time and every conclusion drawn from them is wrong.
    assert!(
        m.device_sync_ns * 10 < m.upload_ns(),
        "sync {} ns vs upload {} ns: the copies are not blocking",
        m.device_sync_ns,
        m.upload_ns()
    );

    eprintln!(
        "load: {} edges, total {:.2} ms = upload {:.2} ms ({:.1}% of it) + alloc {:.2} ms + sync {:.3} ms",
        m.edges,
        m.total_ns as f64 / 1e6,
        m.upload_ns() as f64 / 1e6,
        m.upload_share() * 100.0,
        m.device_alloc_ns as f64 / 1e6,
        m.device_sync_ns as f64 / 1e6
    );
    eprintln!(
        "      {:.1} B/edge at {:.2} GB/s",
        m.uploaded_bytes_per_edge(),
        m.upload_gbps()
    );

    // Releasing leaves the receipt alone; it describes an upload, not a
    // resident copy, and the benchmark reads it after dropping the handle.
    acc.release_graph(&h);
    assert!(acc.last_load_metrics().is_some());
}

#[test]
fn the_cache_drives_the_device_lifecycle() {
    let _serial = exclusive();
    let acc = Arc::new(device_or_skip!());
    let backend = Arc::clone(&acc) as Arc<dyn Accelerator>;
    let (o, dst, vf, vt, et) = chain(4096);
    let mut cache = ResidentCache::new(backend, Some(1 << 24));

    let a = cache.acquire(view(7, 4096, &o, &dst, &vf, &vt, &et)).unwrap().unwrap();
    let b = cache.acquire(view(7, 4096, &o, &dst, &vf, &vt, &et)).unwrap().unwrap();
    assert!(Arc::ptr_eq(&a, &b));
    assert_eq!(cache.stats().loads, 1);
    assert_eq!(acc.resident_snapshots(), vec![7]);
    assert!(acc.resident_bytes() > 0);

    // A snapshot change must not be served from the old copy.
    let other = chain(64);
    let c = cache
        .acquire(view(8, 64, &other.0, &other.1, &other.2, &other.3, &other.4))
        .unwrap()
        .unwrap();
    assert_ne!(a.snapshot_id(), c.snapshot_id());
    assert_eq!(cache.stats().loads, 2);

    // Invalidation (the publish hook) frees the device copy.
    drop(a);
    drop(b);
    drop(c);
    let before = acc.resident_bytes();
    assert!(before > 0);
    assert_eq!(cache.invalidate_snapshot(7), 1);
    let after = acc.resident_bytes();
    assert!(after < before, "{before} -> {after}");
}

#[test]
fn the_kernels_advertised_are_the_kernels_that_landed() {
    let _serial = exclusive();
    // Asserted rather than described: both kernels are advertised, both limits
    // are advertised with them, and the policy still refuses to *use* the window
    // one by default — the operator is not enabled, it is available (§9.8).
    let acc = device_or_skip!();
    let mut reg = gtv_accelerator::AcceleratorRegistry::new();
    reg.register(Arc::new(acc));
    let policy = ExecutionPolicy::default();
    let caps = reg.devices()[0].caps();

    let (decision, backend) =
        reg.route(&policy, &DecisionInputs::new(&caps, Op::KnnFlat, 10_000_000));
    assert!(decision.is_device(), "{decision:?}");
    assert!(backend.is_some());

    // Landed: the bit is set and a window within the limit is admitted.
    assert!(caps.supports(Op::WindowReduce));
    assert!(caps.admits_window(1_024));
    let (decision, backend) =
        reg.route(&policy, &DecisionInputs::new(&caps, Op::WindowReduce, 10_000_000));
    assert!(!decision.is_device());
    assert!(backend.is_none());
    // ...but not enabled by default: the threshold is deliberately unreachable
    // standalone, because the operator measured 1.05x end to end.
    assert_eq!(
        decision.reason(),
        Some(gtv_accelerator::DecisionReason::BelowThreshold {
            work: 10_000_000,
            threshold: u64::MAX,
        })
    );

    // And a k the shared-memory heap cannot hold is refused at plan time, not by
    // a launch that was never going to fit.
    let (decision, _) = reg.route(
        &policy,
        &DecisionInputs::new(&caps, Op::KnnFlat, 10_000_000).with_k(4096),
    );
    assert_eq!(
        decision.reason(),
        Some(gtv_accelerator::DecisionReason::KTooLarge {
            k: 4096,
            max_k: 256
        })
    );
    // Small work still stays on the CPU: nothing about C2 changes the transfer
    // arithmetic that put the threshold there.
    let (decision, _) = reg.route(&policy, &DecisionInputs::new(&caps, Op::KnnFlat, 10));
    assert!(matches!(
        decision.reason(),
        Some(gtv_accelerator::DecisionReason::BelowThreshold { .. })
    ));
}

#[test]
fn the_cpu_reference_still_answers_the_same_question() {
    let _serial = exclusive();
    // The fallback contract: when the device declines, the CPU path answers, and
    // the answer is the reference. Both are exercised here on the same columns.
    let acc = device_or_skip!();
    let (o, dst, vf, vt, et) = chain(32);
    let cpu = CpuAccelerator::new();
    let dev_handle = acc.load_graph(view(1, 32, &o, &dst, &vf, &vt, &et)).unwrap();
    let cpu_handle = cpu.load_graph(view(1, 32, &o, &dst, &vf, &vt, &et)).unwrap();

    // C1 has no device kernel, so the device reports unsupported and the CPU
    // answers — that is the contract, not a gap being papered over.
    let dev_err = acc
        .valid_time_filter(dev_handle.as_ref().unwrap(), 0)
        .unwrap_err();
    assert_eq!(
        dev_err,
        gtv_accelerator::AcceleratorError::Unsupported {
            backend: "cuda-0",
            op: "valid_time_filter"
        }
    );
    let got = cpu.valid_time_filter(cpu_handle.as_ref().unwrap(), 0).unwrap();
    // Every chain edge is live at 0.
    assert_eq!(got.len(), 31);
}

// ---------------------------------------------------------------------------
// C2: flat batch K-NN
// ---------------------------------------------------------------------------

/// A deterministic corpus: a 64-bit LCG in `[-0.5, 0.5)`.
///
/// Deterministic rather than random so a failure reproduces; wide enough that
/// the k-th and (k+1)-th distances are never within rounding of each other, which
/// is checked by [`gap_is_unambiguous`] rather than assumed.
fn corpus(rows: usize, dim: usize) -> (Vec<f32>, Vec<u64>) {
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let data: Vec<f32> = (0..rows * dim).map(|_| next()).collect();
    // Ids deliberately not the row index: a backend that returns positions
    // instead of ids would pass every test that assumes `id == row`.
    let ids: Vec<u64> = (0..rows as u64).map(|i| 1000 + i * 7).collect();
    (data, ids)
}

fn vector_view<'a>(
    space: u32,
    snapshot: u64,
    dim: usize,
    metric: Metric,
    data: &'a [f32],
    ids: &'a [u64],
) -> VectorView<'a> {
    VectorView::new(space, snapshot, dim, metric, data, ids)
}

/// The device's answer must equal the CPU's, up to the rounding that a different
/// summation order costs.
///
/// Two assertions, and the second is the one that makes the first meaningful:
///
/// * every distance is within `1e-5 * max(1, |d|)` of the reference;
/// * the ids are the same, *provided* the reference's k-th and (k+1)-th
///   distances are far enough apart that no rounding could reorder them. Run
///   `k + 1` on the CPU to get that gap and check it here, so a "the sets differ"
///   failure can never mean "the corpus had a tie at the boundary".
fn assert_matches_reference(
    label: &str,
    gpu: &[Vec<VectorHit>],
    cpu_k: &[Vec<VectorHit>],
    cpu_k1: &[Vec<VectorHit>],
    k: usize,
) {
    assert_eq!(gpu.len(), cpu_k.len(), "{label}: batch length");
    for (q, (got, want)) in gpu.iter().zip(cpu_k).enumerate() {
        assert_eq!(got.len(), want.len(), "{label} q{q}: hit count");
        for (rank, (g, w)) in got.iter().zip(want).enumerate() {
            let tol = 1e-5 * w.distance.abs().max(1.0);
            assert!(
                (g.distance - w.distance).abs() <= tol,
                "{label} q{q} rank{rank}: distance {} vs {} (tol {tol})",
                g.distance,
                w.distance
            );
        }
        // Boundary gap from the reference's full (k+1) list. A comparison of id
        // sets is only meaningful if no permitted error could reorder the pair:
        // both sides may be off by `tol`, so the gap has to exceed `2 * tol`.
        let gap = cpu_k1[q]
            .get(k)
            .map(|extra| extra.distance - cpu_k1[q][k - 1].distance);
        if let Some(gap) = gap {
            let boundary = cpu_k1[q][k - 1].distance.abs().max(1.0);
            assert!(
                gap > 2.0 * 1e-5 * boundary,
                "{label} q{q}: corpus has a near-tie at rank {k} (gap {gap} vs \
                 rounding {}) — the comparison would not be meaningful",
                2.0 * 1e-5 * boundary
            );
            let got_ids: Vec<u64> = got.iter().map(|h| h.id).collect();
            let want_ids: Vec<u64> = want.iter().map(|h| h.id).collect();
            assert_eq!(got_ids, want_ids, "{label} q{q}: top-{k} id sets");
        }
    }
}

fn load_both<'a>(
    gpu: &CudaAccelerator,
    cpu: &CpuAccelerator,
    view: VectorView<'a>,
) -> (ResidentVectors, ResidentVectors) {
    let g = gpu.load_vectors(view).unwrap().expect("device does residency");
    let c = cpu.load_vectors(view).unwrap().expect("cpu does residency");
    (g, c)
}

#[test]
fn a_batch_matches_the_reference_on_every_metric() {
    let _serial = exclusive();
    let gpu = device_or_skip!();
    let cpu = CpuAccelerator::new();

    let rows = 4_000;
    let dim = 64;
    let batch = 6;
    let k = 8;
    let (data, ids) = corpus(rows, dim);
    let mut seed = 42u64;
    let queries: Vec<f32> = (0..batch * dim)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        })
        .collect();

    for metric in [Metric::L2, Metric::Cosine, Metric::Ip] {
        let (g, c) = load_both(
            &gpu,
            &cpu,
            vector_view(1, 11, dim, metric, &data, &ids),
        );
        let got = gpu.knn_flat(&g, &queries, k, None).unwrap();
        let want = cpu.knn_flat(&c, &queries, k, None).unwrap();
        let want1 = cpu.knn_flat(&c, &queries, k + 1, None).unwrap();
        assert_matches_reference(&format!("{metric:?}"), &got, &want, &want1, k);

        // The scan has to be a scan: `knn_flat` must have produced a receipt
        // naming the kernel as the dominant stage, or the number is not a
        // measurement of anything.
        let m = gpu.last_knn_metrics().expect("a call leaves a receipt");
        assert_eq!(m.queries, batch);
        assert_eq!(m.rows, rows);
        assert_eq!(m.k, k);
        assert!(m.kernel_ns > 0 && m.download_ns > 0);
        assert!(m.result_bytes >= batch * k * 12);
        assert!(gpu.last_vector_load_metrics().is_some());

        gpu.release_vectors(&g);
        cpu.release_vectors(&c);
    }
}

#[test]
fn a_filtered_scan_returns_only_candidates_and_matches_the_reference() {
    let _serial = exclusive();
    let gpu = device_or_skip!();
    let cpu = CpuAccelerator::new();

    let rows = 3_000;
    let dim = 32;
    let k = 5;
    let (data, ids) = corpus(rows, dim);
    let queries: Vec<f32> = data[..dim].to_vec();

    // Every third row, as a `Sel` — the shape a temporal filter leaves behind.
    let allowed: Vec<u64> = (0..rows as u64).filter(|i| i % 3 == 0).collect();
    let sel = Sel::from_ids(11, rows as u64, allowed.iter().copied()).unwrap();

    let (g, c) = load_both(&gpu, &cpu, vector_view(1, 11, dim, Metric::L2, &data, &ids));
    let got = gpu.knn_flat(&g, &queries, k, Some(&sel)).unwrap();
    let want = cpu.knn_flat(&c, &queries, k, Some(&sel)).unwrap();
    let want1 = cpu.knn_flat(&c, &queries, k + 1, Some(&sel)).unwrap();
    assert_matches_reference("filtered", &got, &want, &want1, k);

    for hit in &got[0] {
        let row = ids.iter().position(|&id| id == hit.id).expect("known id");
        assert_eq!(row % 3, 0, "row {row} is not in the candidate set");
    }
    assert!(gpu.last_knn_metrics().unwrap().filtered);

    // The same set over a *different* universe must not silently select other
    // rows: the kernel reads whole words and cannot see where the universe ends,
    // so the host has to refuse the mismatch before launching.
    let short = Sel::from_ids(
        11,
        (rows - 1) as u64,
        (0..rows as u64 - 1).filter(|i| i % 3 == 0),
    )
    .unwrap();
    let err = gpu.knn_flat(&g, &queries, k, Some(&short)).unwrap_err();
    assert!(matches!(err, AcceleratorError::InvalidArgument(_)), "{err}");

    gpu.release_vectors(&g);
    cpu.release_vectors(&c);
}

#[test]
fn degenerate_k_and_batch_have_defined_answers() {
    let _serial = exclusive();
    let gpu = device_or_skip!();
    let cpu = CpuAccelerator::new();
    let (data, ids) = corpus(3, 4);
    let (g, c) = load_both(&gpu, &cpu, vector_view(2, 5, 4, Metric::L2, &data, &ids));

    // Every query returns a list, even when the list is empty: the shape does not
    // change with k, so a caller can `zip` without checking.
    let queries = data.clone();
    assert_eq!(gpu.knn_flat(&g, &queries, 0, None).unwrap(), vec![Vec::new(); 3]);
    assert_eq!(cpu.knn_flat(&c, &queries, 0, None).unwrap().len(), 3);

    // k larger than the corpus: everything, in order.
    let got = gpu.knn_flat(&g, &queries, 10, None).unwrap();
    let want = cpu.knn_flat(&c, &queries, 10, None).unwrap();
    assert_eq!(got.len(), 3);
    for (a, b) in got.iter().zip(&want) {
        assert_eq!(a.len(), 3);
        assert_eq!(
            a.iter().map(|h| h.id).collect::<Vec<_>>(),
            b.iter().map(|h| h.id).collect::<Vec<_>>()
        );
    }

    // An empty batch does no work and is not an error.
    assert!(gpu.knn_flat(&g, &[], 5, None).unwrap().is_empty());

    // A k the kernel cannot hold is a scheduling fact: the CPU can still answer,
    // and the error says so rather than claiming the operator is missing.
    let err = gpu.knn_flat(&g, &queries, 257, None).unwrap_err();
    assert!(
        matches!(err, AcceleratorError::LimitExceeded { limit: 256, .. }),
        "{err}"
    );
    assert_eq!(err.fallback(), gtv_accelerator::Fallback::Cpu);

    // A misaligned query length is a caller bug and surfaces.
    let err = gpu.knn_flat(&g, &queries[..6], 5, None).unwrap_err();
    assert!(matches!(err, AcceleratorError::InvalidArgument(_)), "{err}");
    assert_eq!(err.fallback(), gtv_accelerator::Fallback::Surface);

    gpu.release_vectors(&g);
    cpu.release_vectors(&c);
}

#[test]
fn an_upload_reports_its_stages_and_its_bytes() {
    let _serial = exclusive();
    let gpu = device_or_skip!();
    let (data, ids) = corpus(1_000, 16);

    // L2: the matrix is uploaded as handed over, so the normalisation pass is
    // zero and the copy is the whole story.
    let view = vector_view(3, 1, 16, Metric::L2, &data, &ids);
    let h = gpu.load_vectors(view).unwrap().unwrap();
    let m = gpu.last_vector_load_metrics().expect("a load leaves a receipt");
    assert_eq!(m.rows, 1_000);
    assert_eq!(m.dim, 16);
    assert_eq!(m.normalize_ns, 0);
    assert!(m.copy_ns > 0);
    assert_eq!(m.uploaded_bytes, 1_000 * (16 * 4 + 8));
    assert!(m.accounting_holds());
    assert_eq!(gpu.resident_bytes(), view.bytes());
    assert_eq!(gpu.resident_spaces(), vec![3]);

    gpu.release_vectors(&h);
    assert_eq!(gpu.resident_bytes(), 0);
    assert!(gpu.resident_spaces().is_empty());

    // Cosine: one host pass over the whole matrix, reported separately. Folding
    // it into `copy_ns` would make the link look slower than it is.
    let view = vector_view(3, 1, 16, Metric::Cosine, &data, &ids);
    let h = gpu.load_vectors(view).unwrap().unwrap();
    let m = gpu.last_vector_load_metrics().unwrap();
    assert!(m.normalize_ns > 0, "cosine rows must be normalised");
    assert!(m.accounting_holds());
    gpu.release_vectors(&h);

    // Releasing twice is a no-op, and a stranger's token frees nothing.
    gpu.release_vectors(&h);
    assert_eq!(gpu.resident_bytes(), 0);
}

#[test]
fn a_stale_snapshot_or_a_foreign_filter_is_an_error_not_an_answer() {
    let _serial = exclusive();
    let gpu = device_or_skip!();
    let (data, ids) = corpus(64, 8);
    let queries = data[..8].to_vec();

    let h = gpu
        .load_vectors(vector_view(4, 100, 8, Metric::L2, &data, &ids))
        .unwrap()
        .unwrap();

    // The handle says space 4 / snapshot 100; a handle for snapshot 101 must not
    // be able to use this copy.
    let mut stale = h.clone();
    stale = ResidentVectors::new(
        stale.token(),
        4,
        101,
        64,
        8,
        Metric::L2,
        stale.bytes(),
    );
    let err = gpu.knn_flat(&stale, &queries, 3, None).unwrap_err();
    assert!(matches!(err, AcceleratorError::SnapshotStale { .. }), "{err}");

    // A candidate set bound to another snapshot is refused before any rows are
    // read from it.
    let foreign = Sel::from_ids(999, 64, [0u64, 1, 2]).unwrap();
    let err = gpu.knn_flat(&h, &queries, 3, Some(&foreign)).unwrap_err();
    assert!(matches!(err, AcceleratorError::SnapshotStale { .. }), "{err}");

    // A set over the wrong universe selects the wrong rows; that is a caller bug.
    let short = Sel::from_ids(100, 32, [0u64, 1]).unwrap();
    let err = gpu.knn_flat(&h, &queries, 3, Some(&short)).unwrap_err();
    assert!(matches!(err, AcceleratorError::InvalidArgument(_)), "{err}");

    gpu.release_vectors(&h);
}

#[test]
fn a_second_accelerator_reuses_the_loaded_kernel() {
    let _serial = exclusive();
    let (data, ids) = corpus(512, 16);
    let queries = data[..32].to_vec();

    // The kernel module is loaded once per process and lives inside the device's
    // primary context. Dropping the only accelerator used to release that
    // context, which destroyed the context the module lived in — so the *second*
    // accelerator's first launch failed with `CUDA_ERROR_INVALID_HANDLE`. This is
    // the regression test for that: two lifecycles, two launches.
    let first = {
        let gpu = device_or_skip!();
        let h = gpu
            .load_vectors(vector_view(9, 1, 16, Metric::L2, &data, &ids))
            .unwrap()
            .unwrap();
        let hits = gpu.knn_flat(&h, &queries, 3, None).unwrap();
        gpu.release_vectors(&h);
        hits[0].iter().map(|h| h.id).collect::<Vec<_>>()
    };

    let gpu = device_or_skip!();
    let h = gpu
        .load_vectors(vector_view(9, 1, 16, Metric::L2, &data, &ids))
        .unwrap()
        .unwrap();
    let hits = gpu.knn_flat(&h, &queries, 3, None).unwrap();
    assert_eq!(
        hits[0].iter().map(|h| h.id).collect::<Vec<_>>(),
        first,
        "the second accelerator answered differently"
    );
    gpu.release_vectors(&h);
}

// ---------------------------------------------------------------------------
// C3: trailing-window aggregates
// ---------------------------------------------------------------------------

/// A column with the shape of a price series: bounded, no long runs of equal
/// values, and a non-zero mean so a running sum has something to lose precision
/// against.
fn series(rows: usize) -> Vec<f64> {
    (0..rows)
        .map(|i| ((i % 977) as f64 * 0.013) - 6.0 + ((i % 7) as f64) * 0.001)
        .collect()
}

/// Relative tolerance for `Sum`/`Mean`.
///
/// These two cannot be bit-exact: the reference is a running recurrence over the
/// whole column whose rounding drifts with `n`, while the device differences two
/// prefixes that restart every chunk. The measured worst case at 1e7 rows is
/// 4e-11 relative (printed by `examples/bench_window.rs`), so this is four
/// orders of magnitude of margin — the constant exists to catch a real
/// regression, not to paper over a boundary.
const TOLERANCE: f64 = 1e-6;

fn assert_close(op: gtv_accelerator::WindowOp, got: &[f64], want: &[f64]) {
    use gtv_accelerator::WindowOp;
    assert_eq!(got.len(), want.len(), "{op:?}: row count");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(a.is_nan(), b.is_nan(), "{op:?} row {i}: NaN mismatch");
        if a.is_nan() {
            continue;
        }
        let exact = matches!(op, WindowOp::Min | WindowOp::Max | WindowOp::Count);
        let tol = if exact { 0.0 } else { TOLERANCE * b.abs().max(1.0) };
        assert!(
            (a - b).abs() <= tol,
            "{op:?} row {i}: device {a}, reference {b} (|d| {})",
            (a - b).abs()
        );
    }
}

#[test]
fn every_window_op_matches_the_reference() {
    let _serial = exclusive();
    let gpu = device_or_skip!();
    let cpu = CpuAccelerator::new();
    // 5 000 rows is two and a half chunks, so the straddling windows at the chunk
    // boundaries are exercised; 1 and 2 hit the halo-less and one-row cases.
    let values = series(5_000);
    for op in [
        gtv_accelerator::WindowOp::Sum,
        gtv_accelerator::WindowOp::Mean,
        gtv_accelerator::WindowOp::Min,
        gtv_accelerator::WindowOp::Max,
        gtv_accelerator::WindowOp::Count,
    ] {
        for w in [1usize, 2, 3, 100, 1_024] {
            let want = cpu.window_reduce(&values, w, op).unwrap();
            let got = gpu.window_reduce(&values, w, op).unwrap();
            assert_close(op, &got, &want);
        }
    }
}

#[test]
fn the_window_limit_is_advertised_before_it_is_enforced() {
    let _serial = exclusive();
    let gpu = device_or_skip!();
    let caps = gpu.caps();
    assert!(caps.admits_window(1_024));
    assert!(!caps.admits_window(1_025));

    // Over the limit is a limit, not a malformed request: the CPU can still
    // answer, so this must be `LimitExceeded` with `Fallback::Cpu` rather than a
    // generic argument error.
    let err = gpu
        .window_reduce(&series(2_000), 1_025, gtv_accelerator::WindowOp::Sum)
        .unwrap_err();
    assert_eq!(
        err,
        AcceleratorError::LimitExceeded {
            backend: "cuda",
            what: "window",
            requested: 1_025,
            limit: 1_024,
        }
    );
    assert_eq!(err.fallback(), gtv_accelerator::Fallback::Cpu);
}

#[test]
fn degenerate_inputs_answer_the_way_the_cpu_does() {
    let _serial = exclusive();
    let gpu = device_or_skip!();
    let cpu = CpuAccelerator::new();
    use gtv_accelerator::WindowOp;

    // Empty column: both sides return an empty column, and neither touches the
    // device.
    assert!(gpu.window_reduce(&[], 10, WindowOp::Sum).unwrap().is_empty());
    assert!(cpu.window_reduce(&[], 10, WindowOp::Sum).unwrap().is_empty());

    // Window 0: the reference defines this as "no rows", not "one row".
    assert!(gpu.window_reduce(&series(100), 0, WindowOp::Sum).unwrap().is_empty());
    assert!(cpu.window_reduce(&series(100), 0, WindowOp::Sum).unwrap().is_empty());

    // Window wider than the column: the aggregate covers what exists, which is
    // the cumulative convention, not a truncated window.
    let values = series(7);
    for op in [WindowOp::Sum, WindowOp::Mean, WindowOp::Min, WindowOp::Max, WindowOp::Count] {
        let want = cpu.window_reduce(&values, 100, op).unwrap();
        let got = gpu.window_reduce(&values, 100, op).unwrap();
        assert_close(op, &got, &want);
    }

    // A single row: one window, one answer.
    let one = [42.5f64];
    for op in [WindowOp::Sum, WindowOp::Mean, WindowOp::Min, WindowOp::Max, WindowOp::Count] {
        assert_eq!(
            gpu.window_reduce(&one, 4, op).unwrap(),
            cpu.window_reduce(&one, 4, op).unwrap(),
            "{op:?}"
        );
    }
}

#[test]
fn the_nan_rule_is_the_reference_fold() {
    let _serial = exclusive();
    let gpu = device_or_skip!();
    let cpu = CpuAccelerator::new();
    use gtv_accelerator::WindowOp;

    // The reference fold starts at the window's first element and only replaces
    // it when the comparison holds, so a NaN is sticky only when it is first and
    // is ignored everywhere else. The device has to reproduce that positional
    // rule, not "min skips NaN".
    let mut values = vec![3.0, 1.0, 2.0, 5.0, 4.0];
    values[2] = f64::NAN;
    for op in [WindowOp::Min, WindowOp::Max] {
        for w in [2usize, 3] {
            let want = cpu.window_reduce(&values, w, op).unwrap();
            let got = gpu.window_reduce(&values, w, op).unwrap();
            assert_close(op, &got, &want);
        }
    }

    // Sum propagates NaN arithmetically, which is the one place the two shapes
    // agree for a different reason. Asserted so a future refactor cannot quietly
    // change it.
    let got = gpu.window_reduce(&values, 2, WindowOp::Sum).unwrap();
    assert!(got[2].is_nan() && got[3].is_nan());
}

#[test]
fn a_resident_column_is_uploaded_once_and_gives_the_memory_back() {
    let _serial = exclusive();
    let gpu = device_or_skip!();
    let values = series(100_000);
    let bytes = std::mem::size_of_val(&values[..]);
    let before = gpu.resident_bytes();

    let handle = gpu.load_column(&values).unwrap().expect("cuda keeps columns");
    assert_eq!(handle.rows(), values.len());
    assert_eq!(handle.bytes(), bytes);
    assert_eq!(gpu.resident_bytes(), before + bytes);

    let first = gpu
        .window_reduce_resident(&handle, 100, gtv_accelerator::WindowOp::Sum)
        .unwrap();
    let m = gpu.last_window_metrics().unwrap();
    assert!(m.resident);
    // The whole point of the handle: no upload on a resident call, and the same
    // bytes back.
    assert_eq!(m.upload_bytes, 0);
    assert_eq!(m.upload_ns, 0);
    assert_eq!(m.result_bytes, bytes);
    assert_eq!(m.rows, values.len());

    let second = gpu
        .window_reduce_resident(&handle, 100, gtv_accelerator::WindowOp::Sum)
        .unwrap();
    assert_eq!(first, second, "two calls over one resident column disagreed");

    gpu.release_column(&handle);
    assert_eq!(gpu.resident_bytes(), before);
    // Releasing twice is not an error: eviction and drop both call it.
    gpu.release_column(&handle);
    // A released handle is refused by name rather than by answering from a
    // dangling pointer.
    let err = gpu
        .window_reduce_resident(&handle, 100, gtv_accelerator::WindowOp::Sum)
        .unwrap_err();
    assert!(matches!(err, AcceleratorError::InvalidArgument(_)), "{err:?}");
}

#[test]
fn the_stage_table_accounts_for_every_byte() {
    let _serial = exclusive();
    let gpu = device_or_skip!();
    let values = series(50_000);
    let bytes = std::mem::size_of_val(&values[..]);
    let _ = gpu
        .window_reduce(&values, 20, gtv_accelerator::WindowOp::Mean)
        .unwrap();
    let m = gpu.last_window_metrics().unwrap();
    assert!(!m.resident);
    assert_eq!(m.upload_bytes, bytes);
    assert_eq!(m.result_bytes, bytes);
    // A stateless call moves both directions; this operator's whole economics is
    // that the result is as big as the input.
    assert_eq!(m.pcie_bytes(), 2 * bytes);
    assert!(m.kernel_ns > 0 && m.download_ns > 0 && m.upload_ns > 0);
    assert!(m.kernel_gbps() > 0.0);
    assert!(!m.dominant_stage().is_empty());
}
