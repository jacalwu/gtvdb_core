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

use std::sync::Arc;

use gtv_accelerator::{
    Accelerator, CpuAccelerator, DecisionInputs, ExecutionPolicy, Op, ResidentCache, SnapshotView,
};
use gtv_accelerator_cuda::CudaAccelerator;

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

#[test]
fn the_device_reports_what_the_upload_needs() {
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

#[test]
fn releasing_twice_and_releasing_a_stranger_are_no_ops() {
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

#[test]
fn the_cache_drives_the_device_lifecycle() {
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
fn the_device_is_not_used_until_a_kernel_exists() {
    // C1 boundary, asserted rather than described: a route through the registry
    // must say `Unsupported`, because the backend advertises no operator.
    let acc = device_or_skip!();
    let mut reg = gtv_accelerator::AcceleratorRegistry::new();
    reg.register(Arc::new(acc));
    let policy = ExecutionPolicy::default();
    let caps = reg.devices()[0].caps();

    let (decision, backend) =
        reg.route(&policy, &DecisionInputs::new(&caps, Op::KnnFlat, 10_000_000));
    assert!(!decision.is_device());
    assert!(backend.is_none());
    assert_eq!(
        decision.reason(),
        Some(gtv_accelerator::DecisionReason::Unsupported)
    );
}

#[test]
fn the_cpu_reference_still_answers_the_same_question() {
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
