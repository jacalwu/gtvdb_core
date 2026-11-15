//! P2 gate: the heat pump moves an upload off the query's critical path.
//!
//! The pump is a *scheduling* optimisation, so the gate is about routing, not
//! about arithmetic: after a pump, the next query must run on the device and
//! return exactly what the host returns. The accompanying `#[ignore]` benchmark
//! measures the latency distribution it is supposed to improve.
//!
//! Run on a machine with a CUDA device:
//!
//! ```sh
//! RUSTFLAGS= RUSTC_WRAPPER= CARGO_INCREMENTAL=0 \
//!   cargo test -p gtv-accelerator --features cuda --release \
//!   --test cuda_heat_pump -- --nocapture
//! # and the measurement:
//! cargo test -p gtv-accelerator --features cuda --release \
//!   --test cuda_heat_pump -- --ignored --nocapture bench_heat_pump
//! ```
//!
//! A missing device is a SKIP for mixed CI fleets, but a hard failure when
//! `GTV_CUDA_REQUIRE_DEVICE=1`.

#![cfg(feature = "cuda")]

use std::sync::Arc;
use std::time::{Duration, Instant};

use gtv_accelerator::{
    AccelBackend, AccelCache, AccelError, AcceleratorRegistry, HostBackend, KnnRequest, RawHit,
    ResidentKind, ResidentRegistry, VramBudget, PREFETCH_HEAT_FLOOR,
};
use gtv_core::Metric;

/// Open device 0 (same skip contract as `cuda_smoke`).
fn usable_device() -> bool {
    match gtv_accelerator::cuda::CudaBackend::probe(0) {
        Ok(_) => true,
        Err(AccelError::NoDevice(msg)) => {
            if std::env::var("GTV_CUDA_REQUIRE_DEVICE").as_deref() == Ok("1") {
                panic!("CI requires a CUDA device but none is usable: {msg}");
            }
            eprintln!("SKIP: no usable CUDA device: {msg}");
            false
        }
        Err(e) => panic!("device present but backend unusable: {e}"),
    }
}

/// A registry over a real device with the given VRAM budget and pump budget.
fn device_registry(capacity_bytes: u64, pump_blocks: usize) -> Option<AcceleratorRegistry> {
    let backend = gtv_accelerator::cuda::CudaBackend::probe(0).ok()?;
    let backend: Arc<dyn AccelBackend> = Arc::new(backend);
    let ledger = ResidentRegistry::new(
        AccelCache::On,
        VramBudget::new(capacity_bytes, capacity_bytes, 0.0),
        0,
        Duration::from_secs(5),
    );
    Some(AcceleratorRegistry::with_backend(backend, ledger).with_heat_pump(pump_blocks))
}

/// A corpus whose L2 distances strictly grow with the row index, so host and
/// device rankings cannot differ by an ULP of associativity.
struct Corpus {
    name: &'static str,
    ids: Vec<u64>,
    data: Vec<f32>,
    query: Vec<f32>,
}

impl Corpus {
    fn new(name: &'static str, n: usize, dim: usize) -> Self {
        let ids: Vec<u64> = (0..n as u64).collect();
        let mut data = vec![0f32; n * dim];
        for i in 0..n {
            data[i * dim] = (i + 1) as f32 * 0.5;
            for d in 1..dim {
                data[i * dim + d] = 0.01 * d as f32;
            }
        }
        Self {
            name,
            ids,
            data,
            query: vec![0f32; dim],
        }
    }

    fn bytes(&self) -> u64 {
        (self.ids.len() * 8 + self.data.len() * 4) as u64
    }

    fn request(&self, k: usize) -> KnnRequest<'_> {
        KnnRequest {
            name: self.name,
            version: 1,
            ids: &self.ids,
            data: &self.data,
            dim: self.query.len(),
            metric: Metric::L2,
            normalized: false,
            query: &self.query,
            k,
            mask: None,
        }
    }
}

fn ids(hits: &[RawHit]) -> Vec<u64> {
    hits.iter().map(|h| h.id).collect()
}

/// One heat-pump pass, spelled exactly as the engine's `pump_heat_prefetch`
/// does it: reclaim what the live set no longer owns, then read the shadow
/// ranking and hand each candidate to the registry.
///
/// `corpora` is the *live* set — a dataset missing from it has been dropped, and
/// the pass releases its device copy before ranking.
fn pump_once(reg: &AcceleratorRegistry, corpora: &[&Corpus]) -> usize {
    let budget = reg.heat_pump_blocks();
    if budget == 0 || !reg.is_device_backed() {
        return 0;
    }
    for (kind, name) in reg.resident_blocks() {
        if corpora.iter().any(|c| c.name == name) {
            continue;
        }
        reg.release(kind, &name);
        reg.forget_candidate(kind, &name);
    }
    let mut uploaded = 0;
    for key in reg.prefetch_candidates(PREFETCH_HEAT_FLOOR, budget) {
        if uploaded >= budget {
            break;
        }
        let Some(c) = corpora.iter().find(|c| c.name == key.name) else {
            reg.forget_candidate(key.kind, &key.name);
            continue;
        };
        if reg.prefetch_upload(
            key.kind,
            key.name.as_str(),
            &c.ids,
            &c.data,
            c.query.len(),
            1,
        ) {
            uploaded += 1;
        }
    }
    uploaded
}

/// P2 gate: a refused corpus, once pumped, is served from the device — and the
/// device answer is the host answer.
#[test]
fn a_pumped_corpus_is_served_from_the_device() {
    if !usable_device() {
        return;
    }
    const N: usize = 20_000;
    const DIM: usize = 32;
    const K: usize = 8;
    let a = Corpus::new("pump-a", N, DIM);
    let b = Corpus::new("pump-b", N, DIM);
    let corpora = [a, b];

    // Room for one corpus: once "pump-a" is hot it is protected, so "pump-b" is
    // refused and leaves shadow evidence instead of evicting it.
    let reg = device_registry(corpora[0].bytes() + corpora[0].bytes() / 2, 1)
        .expect("device registry");

    for _ in 0..5 {
        reg.exact_knn(&corpora[0].request(K));
    }
    assert_eq!(reg.telemetry().executed_on_device, 5, "the first corpus is resident");

    let mut refused = Vec::new();
    for _ in 0..5 {
        refused = reg.exact_knn(&corpora[1].request(K));
    }
    let t = reg.telemetry();
    assert_eq!(t.executed_on_host, 5, "the second corpus keeps being refused");
    assert!(t.cache_declined >= 5);
    // Correctness does not depend on the cache: the refused query still ran the
    // same kernel, on the host.
    let reference = HostBackend::new()
        .exact_knn(&corpora[1].request(K))
        .expect("host reference");
    assert_eq!(ids(&refused), ids(&reference));

    // A saturated cache gives the pump nothing to fill: it declines instead of
    // evicting the corpus a query is using.
    assert_eq!(pump_once(&reg, &[&corpora[0], &corpora[1]]), 0);
    assert!(reg.is_resident(ResidentKind::Corpus, "pump-a"), "nothing was evicted");

    // Drop the first collection: its device copy can never be a hit again, so
    // the pass reclaims those bytes and fills them with the refused corpus.
    let live = [&corpora[1]];
    assert_eq!(
        pump_once(&reg, &live),
        1,
        "the pump uploads the refused corpus into the reclaimed region"
    );
    assert!(!reg.is_resident(ResidentKind::Corpus, "pump-a"), "stale block released");
    assert!(reg.is_resident(ResidentKind::Corpus, "pump-b"));

    let before = reg.telemetry().executed_on_device;
    let pumped = reg.exact_knn(&corpora[1].request(K));
    let t = reg.telemetry();
    assert_eq!(
        t.executed_on_device,
        before + 1,
        "the pumped corpus is a device hit"
    );
    assert_eq!(ids(&pumped), ids(&reference), "device answer == host answer");
    assert!(t.accounting_holds());
}

/// One step of a benchmark plan.
#[derive(Clone, Copy)]
enum Step {
    Query(usize),
    Drop(usize),
}

/// Measurement, not a gate: the pump in the only situation it can serve — a
/// saturated cache plus a collection that gets dropped.
///
/// An evicting pump looks better on a pure rotation benchmark and is a net loss
/// in practice (it re-uploads the block a query just replaced: measured +71%
/// transferred bytes and p99 × 1.41 on a 3-corpus rotation), so the pump only
/// fills VRAM that is free right now. Slack appears in exactly two ways:
///
/// * a collection is re-registered smaller (it releases the difference);
/// * a collection is dropped (its device copy is dead weight, which the pass
///   reclaims before ranking).
///
/// Plan, with a cache that holds two of three corpora:
///
/// * `A×5, B×5` — both warm, cache full;
/// * `C×5` — refused every time: C earns shadow heat and every query pays an
///   upload on the critical path;
/// * `drop A` — the session ends; A's bytes are unreachable;
/// * `C×5` — pump off: A's bytes stay stranded and C keeps falling back to the
///   host. pump on: the pass reclaims A, fills it with C, and these become
///   device hits;
/// * `B×5` — the surviving resident corpus, same either way.
///
/// Numbers are reported, not asserted: a latency-ratio gate on one host is
/// flaky, and the conclusion belongs in the ROADMAP.
#[test]
#[ignore = "measurement, not a correctness gate"]
fn bench_heat_pump_reclaim() {
    if !usable_device() {
        return;
    }
    const N: usize = 200_000;
    const DIM: usize = 128;
    const K: usize = 10;
    let gap = Duration::from_millis(50);

    let corpora = [
        Corpus::new("rc-a", N, DIM),
        Corpus::new("rc-b", N, DIM),
        Corpus::new("rc-c", N, DIM),
    ];
    let per_corpus_mib = corpora[0].bytes() / (1024 * 1024);
    let capacity = corpora[0].bytes() * 2; // room for two of the three

    println!(
        "corpora 3 x {N} x {DIM} ({per_corpus_mib} MiB each), cache {} MiB, gap {gap:?}",
        capacity / (1024 * 1024)
    );

    let mut plan = Vec::new();
    plan.extend((0..5).map(|_| Step::Query(0)));
    plan.extend((0..5).map(|_| Step::Query(1)));
    plan.extend((0..5).map(|_| Step::Query(2)));
    plan.push(Step::Drop(0));
    plan.extend((0..5).map(|_| Step::Query(2)));
    plan.extend((0..5).map(|_| Step::Query(1)));

    let measure = |label: &str, pump_blocks: usize| {
        let reg = device_registry(capacity, pump_blocks).expect("device registry");
        let mut latencies = Vec::with_capacity(plan.len());
        let mut live: Vec<usize> = (0..corpora.len()).collect();
        let t0 = Instant::now();
        for step in &plan {
            match *step {
                Step::Query(i) => {
                    let q0 = Instant::now();
                    let hits = reg.exact_knn(&corpora[i].request(K));
                    latencies.push(q0.elapsed());
                    assert!(!hits.is_empty());
                }
                Step::Drop(i) => live.retain(|&x| x != i),
            }
            if pump_blocks > 0 {
                let view: Vec<&Corpus> = live.iter().map(|&i| &corpora[i]).collect();
                pump_once(&reg, &view);
            }
            std::thread::sleep(gap);
        }
        let total = t0.elapsed();
        latencies.sort_unstable();
        let pick = |q: f64| latencies[((latencies.len() - 1) as f64 * q) as usize];
        let t = reg.telemetry();
        println!(
            "{label:<22} total {total:>8.1?}  p50 {:>8.3?}  p99 {:>8.3?}  device {:>2} host {:>2}  uploads {:>5} MiB  pump {}/{}",
            pick(0.50),
            pick(0.99),
            t.executed_on_device,
            t.executed_on_host,
            t.resident.upload_bytes / (1024 * 1024),
            t.prefetch_uploads,
            t.prefetch_failures,
        );
        (pick(0.50), pick(0.99), total)
    };

    println!("-- drop-and-refill (slack created by a dropped collection) --");
    let (p50_off, p99_off, _) = measure("  reclaim, pump off", 0);
    let (p50_on, p99_on, _) = measure("  reclaim, pump on", 1);
    println!(
        "  delta p50 {:.2}x  p99 {:.2}x  (higher is better)",
        p50_off.as_secs_f64() / p50_on.as_secs_f64().max(f64::MIN_POSITIVE),
        p99_off.as_secs_f64() / p99_on.as_secs_f64().max(f64::MIN_POSITIVE),
    );
}
