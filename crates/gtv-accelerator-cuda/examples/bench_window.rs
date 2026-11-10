//! C3 gate: rolling-window aggregates, CPU reference against the device.
//!
//! ```text
//! cargo run --release -p gtv-accelerator-cuda --example bench_window -- [rows] [window]
//! ```
//!
//! This operator has a property the KNN one does not, and it decides the whole
//! stage: **the result is the same size as the input**. A stateless call moves
//! `8n` bytes up and `8n` back down no matter how fast the kernel is, so the
//! ceiling on an end-to-end speedup is `cpu / (up + down)` — measured, not
//! assumed — and if that ceiling is below the gate, no amount of kernel work
//! reaches it. The bench therefore prints the ceiling first, then the three
//! columns that have to be compared: CPU, stateless device, resident device.
//!
//! Parity runs before any timing, per the standing rule that a benchmark which
//! reports a speedup for a wrong answer is worse than one that reports nothing.
//! The two families have different contracts and both are checked here:
//!
//! * `Count`, `Min`, `Max` — bit-exact against the reference;
//! * `Sum`, `Mean` — within [`TOLERANCE`], because the reference is a running
//!   recurrence whose rounding drifts with `n` while the device differences two
//!   prefixes that restart every chunk. The bench also reports the *maximum*
//!   deviation so the margin is visible rather than asserted.

use std::time::Instant;

use gtv_accelerator::cpu::window_reduce;
use gtv_accelerator::{Accelerator, CpuAccelerator, WindowOp};
use gtv_accelerator_cuda::CudaAccelerator;

/// Relative tolerance for the two accumulating aggregates. The observed maximum
/// is three orders of magnitude below this (printed below), and it is the
/// reference's own drift rather than a device error.
const TOLERANCE: f64 = 1e-6;

fn main() {
    let mut args = std::env::args().skip(1);
    let rows: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(10_000_000);
    let window: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(100);

    let gpu = match CudaAccelerator::open() {
        Ok(acc) => acc,
        Err(e) => {
            eprintln!("no usable CUDA device: {e}");
            return;
        }
    };
    let cpu = CpuAccelerator::new();
    let values = column(rows);
    let bytes = std::mem::size_of_val(&values[..]);
    println!(
        "device: {}\ncolumn: {rows} rows x f64 = {:.1} MB, window {window}\n\
         caps: max_window {} (CPU is unbounded)\n",
        gpu.device().name,
        bytes as f64 / 1e6,
        gpu.caps().max_window,
    );

    // ---- 1. parity gate --------------------------------------------------
    if !parity(&gpu, &cpu, &values[..rows.min(300_000)], window) {
        eprintln!("parity failed — no timings reported");
        std::process::exit(1);
    }

    // ---- 2. the three columns -------------------------------------------
    let cpu_ms = best_of(3, || {
        let t = Instant::now();
        let out = window_reduce(&values, window, WindowOp::Sum);
        let d = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(out);
        Ok(d)
    });

    let cold_ms = best_of(3, || {
        let t = Instant::now();
        let out = gpu.window_reduce(&values, window, WindowOp::Sum).unwrap();
        let d = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(out);
        Ok(d)
    });
    let cold = gpu.last_window_metrics().unwrap();

    let handle = gpu.load_column(&values).unwrap().expect("cuda keeps columns");
    let resident_ms = best_of(5, || {
        let t = Instant::now();
        let out = gpu
            .window_reduce_resident(&handle, window, WindowOp::Sum)
            .unwrap();
        let d = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(out);
        Ok(d)
    });
    let resident = gpu.last_window_metrics().unwrap();

    println!("\n{:<14}{:>10}{:>10}{:>10}", "stage", "cold ms", "resident", "share");
    println!("{:-<46}", "");
    for (name, c, r) in [
        ("upload", cold.upload_ns, resident.upload_ns),
        ("kernel", cold.kernel_ns, resident.kernel_ns),
        ("download", cold.download_ns, resident.download_ns),
        ("total", cold.total_ns, resident.total_ns),
    ] {
        println!(
            "{name:<14}{:>10.2}{:>10.2}{:>9.1}%",
            c as f64 / 1e6,
            r as f64 / 1e6,
            100.0 * r as f64 / resident.total_ns.max(1) as f64
        );
    }
    println!(
        "\nkernel bandwidth {:.1} GB/s over {} MB of chunk+halo read and result write",
        resident.kernel_gbps(),
        resident.kernel_bytes() / 1_000_000
    );

    println!("\n{:<14}{:>10}{:>10}", "path", "ms", "vs CPU");
    println!("{:-<36}", "");
    for (name, ms) in [
        ("cpu msum", cpu_ms),
        ("device cold", cold_ms),
        ("device resident", resident_ms),
    ] {
        println!("{name:<14}{ms:>10.2}{:>9.2}x", cpu_ms / ms);
    }

    // ---- 3. the ceiling, stated as an identity ---------------------------
    //
    // `up + down` is what a stateless call costs before the kernel runs, and it
    // is why the resident form exists. Reported even when the gate is met so the
    // next reader does not have to re-derive it.
    let floor_ms = (cold.upload_ns + cold.download_ns) as f64 / 1e6;
    println!(
        "\nstateless ceiling  {:.2}x  (cpu {:.2} ms / transfers {:.2} ms)\
         \nresident ceiling   {:.2}x  (cpu {:.2} ms / download {:.2} ms + kernel {:.2} ms)",
        cpu_ms / floor_ms,
        cpu_ms,
        floor_ms,
        cpu_ms / resident_ms,
        cpu_ms,
        resident.download_ns as f64 / 1e6,
        resident.kernel_ns as f64 / 1e6,
    );
    println!(
        "break-even         {:.1} calls (upload {:.2} ms amortised over the difference)",
        cold.upload_ns as f64 / 1e6 / (cold_ms - resident_ms).max(1e-6),
        cold.upload_ns as f64 / 1e6
    );

    // ---- 4. where the download time actually goes ------------------------
    //
    // A fresh result buffer is calloc'd, so its pages are mapped to a shared
    // zero page and the copy faults every one of them in. That cost is real —
    // the trait returns an owned `Vec`, so a caller pays it — but it is not
    // PCIe, and a stage table that lumps the two together sends the next
    // optimisation after the wrong number.
    if let Ok(drv) = gtv_accelerator_cuda::driver::driver() {
        if let Ok(d) = drv.alloc(bytes) {
            // SAFETY: `d` is `bytes` from `alloc(bytes)`; `values` is a live
            // slice of exactly that length.
            let _ = unsafe { drv.copy_htod(d, values.as_ptr() as *const u8, bytes) };
            let fresh_ms = best_of(3, || {
                let t = Instant::now();
                let mut fresh = vec![0f64; rows];
                // SAFETY: destination is a live `Vec` of `rows` elements.
                let rc = unsafe {
                    drv.copy_dtoh(fresh.as_mut_ptr() as *mut u8, d, bytes)
                };
                rc?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                std::hint::black_box(&fresh);
                Ok(ms)
            });
            let mut warm = vec![0f64; rows];
            for x in warm.iter_mut() {
                *x = 0.0; // touch every page; calloc alone does not
            }
            let warm_ms = best_of(3, || {
                let t = Instant::now();
                // SAFETY: as above, into a buffer that is already resident.
                let rc = unsafe {
                    drv.copy_dtoh(warm.as_mut_ptr() as *mut u8, d, bytes)
                };
                rc?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                std::hint::black_box(&warm);
                Ok(ms)
            });
            drv.free(d);
            println!(
                "\ndownload into    fresh Vec {fresh_ms:6.2} ms   pre-touched {warm_ms:6.2} ms \
                 ({:.1} vs {:.1} GB/s)",
                bytes as f64 / (fresh_ms / 1e3) / 1e9,
                bytes as f64 / (warm_ms / 1e3) / 1e9
            );
        }
    }

    // ---- 5. does the column fit? ----------------------------------------
    println!(
        "\nresident bytes     {:.1} MB of a {:.2} GiB budget",
        gpu.resident_bytes() as f64 / 1e6,
        gpu.budget_bytes() as f64 / (1u64 << 30) as f64
    );
    gpu.release_column(&handle);
}

/// Bit-exact for `Count`/`Min`/`Max`, tolerance for `Sum`/`Mean`, across a few
/// windows and both the short and the chunk-straddling shape.
fn parity(gpu: &CudaAccelerator, cpu: &CpuAccelerator, values: &[f64], window: usize) -> bool {
    let mut ok = true;
    // Worst relative deviation of the two accumulating ops. Not a pass/fail
    // number: it is how far the reference's own running recurrence has drifted
    // from the device's chunk-local prefixes, and the tolerance has to be chosen
    // above it with margin.
    let mut drifting = 0.0f64;
    let windows = dedup(&[1, 2, 3, window.min(1_000), 1_024, window]);
    for op in [
        WindowOp::Sum,
        WindowOp::Mean,
        WindowOp::Min,
        WindowOp::Max,
        WindowOp::Count,
    ] {
        for w in &windows {
            let want = match cpu.window_reduce(values, *w, op) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{op:?} w={w}: cpu: {e}");
                    return false;
                }
            };
            let got = match gpu.window_reduce(values, *w, op) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{op:?} w={w}: {e}");
                    ok = false;
                    continue;
                }
            };
            if got.len() != want.len() {
                eprintln!("{op:?} w={w}: {} rows back for {}", got.len(), want.len());
                ok = false;
                continue;
            }
            let accumulating = matches!(op, WindowOp::Sum | WindowOp::Mean);
            let mut worst = 0.0f64;
            let mut worst_at = 0;
            let mut bad = false;
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                let d = (a - b).abs();
                if accumulating {
                    drifting = drifting.max(d / b.abs().max(1.0));
                }
                let tol = if accumulating {
                    TOLERANCE * b.abs().max(1.0)
                } else {
                    0.0
                };
                if a.is_nan() != b.is_nan() || (!a.is_nan() && d > tol) {
                    bad = true;
                    if d >= worst {
                        worst = d;
                        worst_at = i;
                    }
                }
            }
            if bad {
                ok = false;
                eprintln!(
                    "{op:?} w={w}: |diff| {worst:e} at row {worst_at} \
                     (device {}, reference {})",
                    got[worst_at], want[worst_at]
                );
            }
        }
    }
    if ok {
        println!(
            "parity ok: 5 ops x {} windows; count/min/max bit-exact, \
             sum/mean worst relative deviation {drifting:.2e} (tolerance {TOLERANCE:.0e})\n",
            windows.len()
        );
    }
    ok
}

fn dedup(xs: &[usize]) -> Vec<usize> {
    let mut v: Vec<usize> = xs.iter().copied().filter(|x| *x > 0).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// A column with the shape of a price series: bounded range, no long runs of
/// equal values, and a non-zero mean so the running sum has something to lose
/// precision against.
fn column(rows: usize) -> Vec<f64> {
    (0..rows)
        .map(|i| ((i % 977) as f64 * 0.013) - 6.0 + ((i % 7) as f64) * 0.001)
        .collect()
}

fn best_of<F>(n: usize, mut f: F) -> f64
where
    F: FnMut() -> Result<f64, String>,
{
    let mut best = f64::MAX;
    for _ in 0..n {
        match f() {
            Ok(ms) => best = best.min(ms),
            Err(reason) => {
                eprintln!("failed: {reason}");
                return f64::INFINITY;
            }
        }
    }
    best
}
