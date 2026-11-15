//! P1c gate: the block-pipelined corpus upload (`copy(b+1) ∥ transpose(b)`)
//! must return exactly what the pre-P1c single-bulk path and the host kernel
//! return — for the flat scan and for the IVF probe.
//!
//! Run on a machine with a CUDA device:
//!
//! ```sh
//! RUSTFLAGS= RUSTC_WRAPPER= CARGO_INCREMENTAL=0 \
//!   cargo test -p gtv-accelerator --features cuda --release \
//!   --test cuda_prefetch -- --nocapture
//! ```
//!
//! A *missing device* is a SKIP for mixed CI fleets, but a hard failure when
//! `GTV_CUDA_REQUIRE_DEVICE=1`.

#![cfg(feature = "cuda")]

use gtv_accelerator::{
    cuda::CudaBackend, AccelBackend, AccelError, IvfRequest, KnnRequest, RawHit,
};
use gtv_core::Metric;

/// Open device 0 (same skip contract as `cuda_smoke`).
fn device() -> Option<CudaBackend> {
    match CudaBackend::probe(0) {
        Ok(b) => Some(b),
        Err(AccelError::NoDevice(msg)) => {
            if std::env::var("GTV_CUDA_REQUIRE_DEVICE").as_deref() == Ok("1") {
                panic!("CI requires a CUDA device but none is usable: {msg}");
            }
            eprintln!("SKIP: no usable CUDA device: {msg}");
            None
        }
        Err(e) => panic!("device present but backend unusable: {e}"),
    }
}

fn lcg(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed >> 40) as f32 / (1u64 << 24) as f32
}

/// Distances strictly grow with the row index, so a few ULPs of f32
/// associativity difference between host and device cannot change ranking.
fn separated_corpus(n: usize, dim: usize) -> (Vec<u64>, Vec<f32>, Vec<f32>) {
    let ids: Vec<u64> = (0..n as u64).collect();
    let mut data = vec![0f32; n * dim];
    for i in 0..n {
        data[i * dim] = (i + 1) as f32 * 0.5;
        for d in 1..dim {
            data[i * dim + d] = 0.01 * d as f32;
        }
    }
    // Zero query => L2 distance is the row norm squared, strictly growing.
    let query = vec![0f32; dim];
    (ids, data, query)
}

/// Ids must match rank-for-rank; distances may differ by a few ULPs.
fn assert_hits_equal(a: &[RawHit], b: &[RawHit], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: hit count differs");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.id, y.id, "{label}: id differs at rank {i}");
        let tol = 1e-4f32 * x.distance.abs().max(1.0);
        assert!(
            (x.distance - y.distance).abs() <= tol,
            "{label}: distance differs at rank {i}: {} vs {} (tol {tol})",
            x.distance,
            y.distance
        );
    }
}

/// P1c-2: a corpus uploaded as many small blocks must score identically to
/// the single-bulk path and to the host kernel — and the second query (cache
/// hit over the finished `data_t`) must repeat the first.
#[test]
fn multi_block_flat_matches_single_bulk_and_host() {
    let Some(_) = device() else { return; };
    const N: usize = 4_000;
    const DIM: usize = 16;
    const K: usize = 10;
    let (ids, data, query) = separated_corpus(N, DIM);

    // dim=16 rows are 64 B; a 4 KiB block is 64 rows -> 63 blocks.
    let block_bytes = 4 * 1024;
    let multi = CudaBackend::probe(0)
        .unwrap()
        .with_prefetch_block_bytes(block_bytes);
    let bulk = CudaBackend::probe(0)
        .unwrap()
        .with_prefetch_block_bytes(0); // pre-P1c single copy
    let host = gtv_accelerator::HostBackend::new();

    let mk = |name: &'static str| KnnRequest {
        name,
        version: 1,
        ids: &ids,
        data: &data,
        dim: DIM,
        metric: Metric::L2,
        normalized: false,
        query: &query,
        k: K,
        mask: None,
    };

    let h = host.exact_knn(&mk("h")).expect("host knn");
    let one = bulk.exact_knn(&mk("bulk")).expect("bulk knn");
    let pipelined = multi.exact_knn(&mk("multi")).expect("pipelined knn");
    assert_hits_equal(&h, &one, "host vs single-bulk");
    assert_hits_equal(&h, &pipelined, "host vs multi-block");

    // Cache hit: the same backend answers again from the finished corpus.
    let again = multi.exact_knn(&mk("multi")).expect("cached knn");
    assert_hits_equal(&pipelined, &again, "first vs cached query");
}

/// P1c-2 for the IVF surface: the pipeline builds the same cell-ordered
/// corpus the probe scans, so per-block transposes must not disturb the
/// position mapping.
#[test]
fn multi_block_ivf_probe_matches_single_bulk_and_host() {
    let Some(_) = device() else { return; };
    const N: usize = 4_000;
    const DIM: usize = 16;
    const K: usize = 8;
    let (ids, data, query) = separated_corpus(N, DIM);

    // Probe runs that deliberately straddle block boundaries (64-row blocks).
    let mut positions: Vec<u32> = Vec::new();
    positions.extend(50..130); // spans blocks 0-2
    positions.extend(2_000..2_100); // middle
    positions.extend(3_950..4_000); // tail (short final block)
    positions.sort_unstable();
    positions.dedup();

    let multi = CudaBackend::probe(0)
        .unwrap()
        .with_prefetch_block_bytes(4 * 1024);
    let bulk = CudaBackend::probe(0)
        .unwrap()
        .with_prefetch_block_bytes(0);
    let host = gtv_accelerator::HostBackend::new();

    let mk = |name: &'static str| IvfRequest {
        name,
        version: 1,
        ids: &ids,
        data: &data,
        dim: DIM,
        metric: Metric::L2,
        normalized: false,
        query: &query,
        k: K,
        positions: &positions,
    };

    let h = host.ivf_probe(&mk("h")).expect("host probe");
    let one = bulk.ivf_probe(&mk("bulk")).expect("bulk probe");
    let pipelined = multi.ivf_probe(&mk("multi")).expect("pipelined probe");
    assert_hits_equal(&h, &one, "host vs single-bulk probe");
    assert_hits_equal(&h, &pipelined, "host vs multi-block probe");
}

/// Cosine goes through the same pipeline with the normalized flag; the query
/// normalization must stay on the query side only.
#[test]
fn multi_block_cosine_matches_single_bulk() {
    let Some(_) = device() else { return; };
    const N: usize = 1_000;
    const DIM: usize = 32;
    const K: usize = 5;
    let mut seed = 0xDEAD_BEEF_CAFE_F00Du64;
    let ids: Vec<u64> = (0..N as u64).collect();
    let mut data = vec![0f32; N * DIM];
    for row in data.chunks_mut(DIM) {
        for v in row.iter_mut() {
            *v = lcg(&mut seed) + 0.01;
        }
        Metric::Cosine.normalize_in_place(row);
    }
    let mut query: Vec<f32> = (0..DIM).map(|_| lcg(&mut seed) + 0.01).collect();
    Metric::Cosine.normalize_in_place(&mut query);

    let multi = CudaBackend::probe(0)
        .unwrap()
        .with_prefetch_block_bytes(2 * 1024);
    let bulk = CudaBackend::probe(0).unwrap().with_prefetch_block_bytes(0);

    let mk = |name: &'static str| KnnRequest {
        name,
        version: 1,
        ids: &ids,
        data: &data,
        dim: DIM,
        metric: Metric::Cosine,
        normalized: true,
        query: &query,
        k: K,
        mask: None,
    };
    let one = bulk.exact_knn(&mk("bulk")).expect("bulk cosine");
    let pipelined = multi.exact_knn(&mk("multi")).expect("pipelined cosine");
    assert_hits_equal(&one, &pipelined, "single-bulk vs multi-block cosine");
}

/// P1c-3: the fused cold scan pre-applies the label mask on the host per
/// block window (kernel runs with `has_mask = 0`). Filtered results must match
/// the unmasked-style reference paths, and the cache-hit query (whole-corpus
/// path, mask uploaded to the kernel) must repeat the fused first answer.
#[test]
fn fused_cold_scan_with_label_mask_matches_reference() {
    use arrow::array::BooleanArray;

    let Some(_) = device() else { return; };
    const N: usize = 4_000;
    const DIM: usize = 16;
    const K: usize = 10;
    let (ids, data, query) = separated_corpus(N, DIM);

    // Keep only even rows: spans every 64-row block window.
    let mask = BooleanArray::from((0..N).map(|r| Some(r % 2 == 0)).collect::<Vec<_>>());

    let multi = CudaBackend::probe(0)
        .unwrap()
        .with_prefetch_block_bytes(4 * 1024); // 64-row blocks -> fused path
    let bulk = CudaBackend::probe(0)
        .unwrap()
        .with_prefetch_block_bytes(0); // single block -> pre-P1c path
    let host = gtv_accelerator::HostBackend::new();

    let mk = |name: &'static str| KnnRequest {
        name,
        version: 1,
        ids: &ids,
        data: &data,
        dim: DIM,
        metric: Metric::L2,
        normalized: false,
        query: &query,
        k: K,
        mask: Some(&mask),
    };

    let h = host.exact_knn(&mk("h")).expect("host masked knn");
    let one = bulk.exact_knn(&mk("bulk")).expect("bulk masked knn");
    let fused = multi.exact_knn(&mk("multi")).expect("fused masked knn");
    assert_hits_equal(&h, &one, "host vs single-bulk masked");
    assert_hits_equal(&h, &fused, "host vs fused masked");
    assert!(
        fused.iter().all(|hit| hit.id % 2 == 0),
        "label mask must exclude odd rows"
    );

    // Cache hit runs the whole-corpus masked path: same answer as the fused
    // first query.
    let cached = multi.exact_knn(&mk("multi")).expect("cached masked knn");
    assert_hits_equal(&fused, &cached, "fused first vs cached masked");
}

/// Measurement, not a gate: cold first-query latency of the P1c pipeline vs
/// the pre-P1c single bulk copy, and the steady-state second query.
///
/// Run with
/// `cargo test -p gtv-accelerator --features cuda --release -- --ignored --nocapture bench_cold`.
#[test]
#[ignore = "measurement, not a correctness gate"]
fn bench_cold_upload_pipeline_vs_bulk() {
    use std::time::Instant;

    let Some(_) = device() else { return; };
    const N: usize = 400_000;
    const DIM: usize = 128;
    const K: usize = 10;
    let (ids, data, query) = separated_corpus(N, DIM);
    let payload_mib = N * DIM * 4 / (1024 * 1024);

    let mk = |name: &'static str| KnnRequest {
        name,
        version: 1,
        ids: &ids,
        data: &data,
        dim: DIM,
        metric: Metric::L2,
        normalized: false,
        query: &query,
        k: K,
        mask: None,
    };

    let run = |label: &str, block_bytes: usize| {
        let backend = CudaBackend::probe(0)
            .unwrap()
            .with_prefetch_block_bytes(block_bytes);
        let t0 = Instant::now();
        let cold = backend.exact_knn(&mk("bench")).expect("cold knn");
        let cold_dt = t0.elapsed();
        let t1 = Instant::now();
        let warm = backend.exact_knn(&mk("bench")).expect("warm knn");
        let warm_dt = t1.elapsed();
        assert_eq!(cold.iter().map(|h| h.id).collect::<Vec<_>>(),
                   warm.iter().map(|h| h.id).collect::<Vec<_>>());
        println!(
            "{label:<24} cold(first query) {cold_dt:>9.3?}   warm(cache hit) {warm_dt:>9.3?}"
        );
        (cold_dt, warm_dt)
    };

    println!("corpus {N} x {DIM} ({payload_mib} MiB payload), k={K}");
    run("bulk (pre-P1c)", 0);
    run("pipeline 64 MiB", 64 * 1024 * 1024);
    run("pipeline 16 MiB", 16 * 1024 * 1024);
    run("pipeline 8 MiB", 8 * 1024 * 1024);
    run("pipeline 4 MiB", 4 * 1024 * 1024);
    run("pipeline 1 MiB", 1024 * 1024);
}
