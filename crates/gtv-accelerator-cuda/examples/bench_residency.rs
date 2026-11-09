//! What one upload costs, at both widths of the `dst` column.
//!
//! `cargo run --release -p gtv-accelerator-cuda --example bench_residency -- 10000000`
//!
//! The number this prints is the break-even denominator for every later stage:
//! a device operator can only win if its per-query saving, multiplied by the
//! number of queries on one snapshot, exceeds one upload. C0 put that at 2.1
//! filter invocations; this measures the upload side of the same ratio at a
//! realistic edge count, and at both widths.
//!
//! It also answers the question that decides where the width comes from: turning
//! a `u64` column into a `u32` one costs `producer`, an upload at `u64` costs
//! `wide`, at `u32` `narrow`. If `wide - narrow < producer`, no amount of
//! cleverness in `load_graph` can pay for the conversion, and the narrow column
//! has to be built upstream. §9.2 of the plan records what this printed.
//!
//! It is deliberately not a benchmark harness: a few numbers, printed, with the
//! machine's name next to it, because a speedup without a machine is a rumour.

use std::time::Instant;

use gtv_accelerator::{Accelerator, SnapshotView};
use gtv_accelerator_cuda::CudaAccelerator;

fn main() {
    let edges: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    // `wide` or `narrow` runs one width only. Two widths in one process would be
    // dishonest about "cold": the second one inherits a pool the first already
    // paid for, which is exactly the effect residency exists to exploit.
    let mode = std::env::args().nth(2).unwrap_or_default();

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
    let wide = SnapshotView::new(1, nodes, &offsets, &dst, &vf, &vt, &et);

    // The narrow column is what a producer would build once, at CSR build time,
    // and hand to every upload thereafter. It is timed because that cost is the
    // whole question: paying it per query is a loss, paying it once is not.
    let t = Instant::now();
    let dst32: Vec<u32> = dst.iter().map(|&d| d as u32).collect();
    let producer = ms(t.elapsed());
    let narrow = wide.with_dst32(&dst32);

    println!(
        "graph: {} nodes, {} edges | wide {:>6.1} MiB ({:.1} B/edge, dst u64) | narrow {:>6.1} MiB ({:.1} B/edge, dst u32)",
        nodes,
        edges,
        wide.bytes() as f64 / (1u64 << 20) as f64,
        wide.bytes() as f64 / edges.max(1) as f64,
        narrow.bytes() as f64 / (1u64 << 20) as f64,
        narrow.bytes() as f64 / edges.max(1) as f64,
    );
    println!(
        "producer {:>7.2} ms  the one-time u32 column ({:.2} GB/s host, 12 B/edge moved)",
        producer,
        (edges * 12) as f64 / (producer / 1e3) / 1e9
    );

    let w = if mode != "narrow" {
        uploads(&acc, wide, 3)
    } else {
        Vec::new()
    };
    let n = if mode != "wide" {
        uploads(&acc, narrow, 3)
    } else {
        Vec::new()
    };
    if !w.is_empty() {
        print_row("upload u64", &w, wide.bytes());
    }
    if !n.is_empty() {
        print_row("upload u32", &n, narrow.bytes());
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
        "per query: CPU {cpu_ms:.2} ms, kernel {kernel_ms:.2} ms, saving {saving:.2} ms"
    );
    for (label, runs) in [("u64", &w), ("u32", &n)] {
        if let (Some(first), Some(last)) = (runs.first(), runs.last()) {
            println!(
                "break-even {label}: {:.1} cold / {:.1} steady invocations per snapshot",
                first / saving,
                last / saving
            );
        }
    }
    if let (Some(wm), Some(nm)) = (w.last(), n.last()) {
        println!(
            "verdict: the u32 column saves {:.2} ms per upload and costs {:.2} ms once, so it\n         pays for itself above {:.1} uploads of the same graph — and never, if the\n         conversion is repeated per upload.",
            wm - nm,
            producer,
            producer / (wm - nm).max(1e-9)
        );
    }
}

/// One upload, repeated: `[first, second, third]` in ms. The first includes
/// setting up the context's memory pool, which is why residency is worth
/// amortising and a single cold upload is not.
fn uploads(acc: &CudaAccelerator, view: SnapshotView<'_>, times: usize) -> Vec<f64> {
    (0..times)
        .map(|_| {
            let t = Instant::now();
            let h = acc.load_graph(view).unwrap().expect("residency supported");
            let d = ms(t.elapsed());
            acc.release_graph(&h);
            d
        })
        .collect()
}

fn print_row(label: &str, runs: &[f64], bytes: usize) {
    let gbs = |ms: f64| bytes as f64 / (ms / 1e3) / 1e9;
    print!("{label}:");
    for r in runs {
        print!(" {:>7.2} ms", r);
    }
    println!("  (steady {:.2} GB/s)", gbs(runs[runs.len() - 1]));
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}
