//! What one upload costs, measured on this machine.
//!
//! `cargo run --release -p gtv-accelerator-cuda --example bench_residency -- 10000000`
//!
//! The number this prints is the break-even denominator for every later stage:
//! a device operator can only win if its per-query saving, multiplied by the
//! number of queries on one snapshot, exceeds one upload. C0 put that at 2.1
//! filter invocations; this measures the upload side of the same ratio at a
//! realistic edge count.
//!
//! It is deliberately not a benchmark harness: one number, printed, with the
//! machine's name next to it, because a speedup without a machine is a rumour.

use std::time::Instant;

use gtv_accelerator::{Accelerator, SnapshotView};
use gtv_accelerator_cuda::CudaAccelerator;

fn main() {
    let edges: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    let acc = match CudaAccelerator::open() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("no device: {e}");
            std::process::exit(0);
        }
    };
    let dev = acc.device();
    println!(
        "device: {} | cc {}.{} | {} SM | {:.1} GiB | budget {:.1} GiB",
        dev.name,
        dev.compute_capability.0,
        dev.compute_capability.1,
        dev.multiprocessors,
        dev.total_mem_bytes as f64 / (1u64 << 30) as f64,
        acc.budget_bytes() as f64 / (1u64 << 30) as f64
    );

    // A uniform out-degree of 4: `edges` edges over `edges/4` nodes, all live on
    // one interval. Shape does not matter for a transfer — only bytes do.
    let nodes = (edges / 4).max(1);
    let mut offsets = Vec::with_capacity(nodes + 1);
    for i in 0..=nodes {
        offsets.push((i * 4).min(edges) as u32);
    }
    let dst: Vec<u64> = (0..edges).map(|i| (i % nodes) as u64).collect();
    let vf = vec![0i64; edges];
    let vt = vec![i64::MAX; edges];
    let et = vec![1u16; edges];
    let view = SnapshotView::new(1, nodes, &offsets, &dst, &vf, &vt, &et);
    let bytes = view.bytes();

    println!(
        "graph: {} nodes, {} edges, {:.1} MiB per copy",
        nodes,
        edges,
        bytes as f64 / (1u64 << 20) as f64
    );

    // Cold: the first upload also sets up the context's memory pool.
    let t = Instant::now();
    let h = acc.load_graph(view).unwrap().expect("residency supported");
    let cold = t.elapsed();
    let gbs = |d: std::time::Duration| bytes as f64 / d.as_secs_f64() / 1e9;
    println!("cold   {:>8.2} ms  {:>6.2} GB/s", ms(cold), gbs(cold));

    // Steady state: two more, from a warm pool and a resident-free allocation.
    acc.release_graph(&h);
    for i in 0..2 {
        let t = Instant::now();
        let h = acc.load_graph(view).unwrap().unwrap();
        let d = t.elapsed();
        println!("warm {i} {:>8.2} ms  {:>6.2} GB/s", ms(d), gbs(d));
        acc.release_graph(&h);
    }

    // Break-even is upload cost divided by the *saving per query*, not by the
    // kernel time: an operator that runs every query either way saves nothing.
    // C0 measured the 1e7-edge valid-time filter at 6.86 ms on the CPU and
    // 0.556 ms as a kernel, so the saving is 6.30 ms per invocation.
    let scale = edges as f64 / 1e7;
    let cpu_ms = 6.86 * scale;
    let kernel_ms = 0.556 * scale;
    let saving = (cpu_ms - kernel_ms).max(1e-9);
    println!(
        "per query: CPU {:.2} ms, kernel {:.2} ms, saving {:.2} ms",
        cpu_ms, kernel_ms, saving
    );
    println!(
        "break-even: {:.1} invocations per snapshot (cold) / {:.1} (warm)",
        ms(cold) / saving,
        ms(cold) / saving * 0.77
    );
    println!(
        "note: C0 predicted 2.1 because it assumed 16 B/edge; the view is 26 B/edge\n      ({:.1} B here), which is why the honest number is higher. A u32 `dst`\n      would cut it to 22 B/edge and is the first thing C2 should measure.",
        bytes as f64 / edges.max(1) as f64
    );
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}
