//! C2 gate: flat batch K-NN, CPU reference against the device.
//!
//! ```text
//! cargo run --release -p gtv-accelerator-cuda --example bench_knn -- [rows] [dim] [k]
//! ```
//!
//! The correctness gate runs *first* and a mismatch aborts before any timing is
//! printed (`docs/gtvdb-cpu-cuda-cursor-development-spec.md` §10.4: a benchmark
//! that reports a speedup for a wrong answer is worse than one that reports
//! nothing).
//!
//! What it reports, and why each line exists:
//!
//! * **parity** — same ids, distances within tolerance, per metric;
//! * **cold vs resident** — the upload is paid once, so both curves are shown;
//! * **stages** — kernel / download / merge, because a total that hides the
//!   result transfer cannot explain itself when `k` grows;
//! * **batch sweep** — the device wins by having many queries in flight, so the
//!   per-query cost has to be shown as a function of batch, not at one point.

use std::time::Instant;

use gtv_accelerator::{
    Accelerator, CpuAccelerator, ExecutionPolicy, KnnMetrics, VectorView,
};
use gtv_accelerator_cuda::CudaAccelerator;
use gtv_core::Metric;

fn main() {
    let mut args = std::env::args().skip(1);
    let rows: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(100_000);
    let dim: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(128);
    let k: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(10);

    let gpu = match CudaAccelerator::open() {
        Ok(acc) => acc,
        Err(e) => {
            eprintln!("no usable CUDA device: {e}");
            return;
        }
    };
    let cpu = CpuAccelerator::new();
    let policy = ExecutionPolicy::default();

    let (data, ids) = corpus(rows, dim);
    let queries = queries(64, dim);
    println!(
        "device: {}  budget {:.2} GiB\ncorpus: {rows} rows x {dim} dim = {:.1} MB, k={k}\n\
         policy: min_knn_work={} (rows x queries)\n",
        gpu.device().name,
        gpu.budget_bytes() as f64 / (1u64 << 30) as f64,
        (rows * dim * 4) as f64 / 1e6,
        policy.min_knn_work,
    );

    // ---- 1. correctness gate --------------------------------------------
    if !parity(&gpu, &cpu, &data, &ids, dim) {
        eprintln!("parity failed — no timings reported");
        std::process::exit(1);
    }

    // ---- 2. upload, both sides ------------------------------------------
    let view = VectorView::new(1, 1, dim, Metric::L2, &data, &ids);
    let t = Instant::now();
    let g = gpu.load_vectors(view).unwrap().expect("device does residency");
    let upload = t.elapsed();
    let m = gpu.last_vector_load_metrics().expect("a load leaves a receipt");
    println!(
        "upload: {:.1} ms for {:.1} MB ({:.2} GB/s), {:.1} B/row, normalise {:.1} ms",
        upload.as_secs_f64() * 1e3,
        m.uploaded_bytes as f64 / 1e6,
        m.upload_gbps(),
        m.bytes_per_row(),
        m.normalize_ns as f64 / 1e6,
    );
    let t = Instant::now();
    let c = cpu.load_vectors(view).unwrap().expect("cpu does residency");
    println!(
        "  cpu reference load: {:.1} ms\n",
        t.elapsed().as_secs_f64() * 1e3
    );

    // ---- 3. batch sweep --------------------------------------------------
    println!(
        "{:>6} {:>12} {:>12} {:>8} {:>12} {:>10}",
        "batch", "cpu ms", "gpu ms", "speedup", "gpu us/query", "GB/s"
    );
    let mut break_even = Vec::new();
    for batch in [1usize, 2, 4, 8, 16, 32, 64] {
        let q = &queries[..batch * dim];
        // The CPU has no receipt, so wall clock is the only stage it has; the
        // device is timed the same way here and split into stages further down.
        let cpu_ms = best_of(3, || {
            cpu.knn_flat(&c, q, k, None).expect("cpu knn");
        });
        let gpu_ms = best_of(3, || {
            gpu.knn_flat(&g, q, k, None).expect("gpu knn");
        });
        let km: KnnMetrics = gpu.last_knn_metrics().expect("a call leaves a receipt");
        println!(
            "{batch:>6} {cpu_ms:>12.3} {gpu_ms:>12.3} {:>7.1}x {:>12.1} {:>10.0}",
            cpu_ms / gpu_ms,
            gpu_ms * 1e3 / batch as f64,
            km.scan_gbps()
        );
        break_even.push((batch, cpu_ms, gpu_ms));
    }

    // ---- 4. stage table for the last (largest) batch ---------------------
    let km = gpu.last_knn_metrics().expect("a call leaves a receipt");
    println!("\nstages for batch {}:", km.queries);
    for (name, ns) in [
        ("upload", km.upload_ns),
        ("kernel", km.kernel_ns),
        ("download", km.download_ns),
        ("merge", km.merge_ns),
    ] {
        println!(
            "  {name:<10} {:>9.3} ms  {:>5.1}%",
            ns as f64 / 1e6,
            ns as f64 / km.total_ns.max(1) as f64 * 100.0
        );
    }
    println!(
        "  {:<10} {:>9.3} ms   (dominant: {})\n  splits {} x {BLOCK} threads, result {:.1} KB, \
         uploaded {:.1} KB",
        "total",
        km.total_ns as f64 / 1e6,
        km.dominant_stage(),
        km.splits,
        km.result_bytes as f64 / 1e3,
        km.upload_bytes as f64 / 1e3,
    );

    // ---- 5. break-even ---------------------------------------------------
    // How many queries of this size the upload has to amortise before the device
    // is ahead. The policy's threshold is the same question asked with the
    // numbers from C0; this recomputes it with today's.
    let (_, cpu_ms, gpu_ms) = break_even[break_even.len() - 1];
    let per_query_saving = (cpu_ms - gpu_ms) / km.queries as f64;
    if per_query_saving > 0.0 {
        println!(
            "\nbreak-even: {:.0} queries of this size (upload {:.1} ms / saving {:.3} ms per query)",
            upload.as_secs_f64() * 1e3 / per_query_saving,
            upload.as_secs_f64() * 1e3,
            per_query_saving,
        );
    } else {
        println!("\nbreak-even: never — the device is slower per query on this shape");
    }

    gpu.release_vectors(&g);
    cpu.release_vectors(&c);
}

/// Threads per block, for the printout only.
const BLOCK: usize = 256;

/// Compare both backends on every metric, at `k` and `k + 1`.
fn parity(
    gpu: &CudaAccelerator,
    cpu: &CpuAccelerator,
    data: &[f32],
    ids: &[u64],
    dim: usize,
) -> bool {
    let k = 10;
    let batch = 4;
    let queries = queries(batch, dim);
    let mut ok = true;
    for metric in [Metric::L2, Metric::Cosine, Metric::Ip] {
        let view = VectorView::new(1, 1, dim, metric, data, ids);
        let g = gpu.load_vectors(view).unwrap().unwrap();
        let c = cpu.load_vectors(view).unwrap().unwrap();
        let got = gpu.knn_flat(&g, &queries, k, None).unwrap();
        let want = cpu.knn_flat(&c, &queries, k, None).unwrap();
        let want1 = cpu.knn_flat(&c, &queries, k + 1, None).unwrap();

        for (q, (a, b)) in got.iter().zip(&want).enumerate() {
            let ids_match = a.iter().map(|h| h.id).eq(b.iter().map(|h| h.id));
            // A gap smaller than twice the tolerance means the order is not
            // defined to begin with; report it instead of asserting on it.
            let gap = want1[q].get(k).map(|e| e.distance - want1[q][k - 1].distance);
            let ambiguous = gap.is_some_and(|g| {
                g <= 2.0 * 1e-5 * want1[q][k - 1].distance.abs().max(1.0)
            });
            if !ids_match && !ambiguous {
                ok = false;
                eprintln!("{metric:?} q{q}: ids differ from the reference");
            }
            for (rank, (x, y)) in a.iter().zip(b).enumerate() {
                let tol = 1e-5 * y.distance.abs().max(1.0);
                if (x.distance - y.distance).abs() > tol {
                    ok = false;
                    eprintln!(
                        "{metric:?} q{q} rank{rank}: distance {} vs {}",
                        x.distance, y.distance
                    );
                }
            }
        }
        gpu.release_vectors(&g);
        cpu.release_vectors(&c);
    }
    println!(
        "parity: {} queries x 3 metrics, top-{k} {}",
        batch,
        if ok { "ok" } else { "FAILED" }
    );
    ok
}

/// Median-free best-of: the minimum is what a latency-bound caller wants, and it
/// is the number least polluted by a stray scheduler tick.
fn best_of(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn corpus(rows: usize, dim: usize) -> (Vec<f32>, Vec<u64>) {
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let data: Vec<f32> = (0..rows * dim).map(|_| next()).collect();
    let ids: Vec<u64> = (0..rows as u64).collect();
    (data, ids)
}

fn queries(n: usize, dim: usize) -> Vec<f32> {
    let (data, _) = corpus(n, dim);
    data
}
