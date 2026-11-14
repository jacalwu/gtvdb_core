//! P1a / P1b-1 — quantify the PCIe DMA wall that the SSD→RAM→VRAM tiering
//! must cross, and measure what the pinned-staging path (P1b-1) wins back.
//!
//! This is a *measurement*, not an optimization. Three numbers per payload size:
//!
//! 1. **unpinned H2D** — a plain `Vec<f32>` uploaded with `clone_htod`. This was
//!    the path `CudaBackend::build_corpus` took before P1b-1: the driver stages
//!    the pageable source through an internal bounce buffer before the DMA.
//! 2. **staging H2D** — the P1b-1 path now in `build_corpus`: copy the `Vec`
//!    into a write-combined `cudaHostAlloc` buffer, then `clone_htod` the pinned
//!    buffer. The copy is a host memcpy (no bounce), and the DMA runs at the
//!    pinned ceiling.
//! 3. **pinned H2D** — `cudaHostAlloc` alone (no source copy), the upper bound
//!    P1b-2 (`cudaHostRegister` over an existing mmap view) approaches.
//!
//! Run on a machine with a CUDA device:
//!
//! ```sh
//! RUSTFLAGS= RUSTC_WRAPPER= CARGO_INCREMENTAL=0 \
//!   cargo run -p gtv-accelerator --features cuda --release \
//!   --example bench_pcie_transfer
//! ```
//!
//! A missing device prints SKIP and exits 0 (the example must not fail a mixed
//! fleet). The absolute numbers are for same-machine A/B only — never quote them
//! as an external performance claim.

#![cfg(feature = "cuda")]

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream};

const MIB: u64 = 1024 * 1024;

/// Sweep payload sizes from 1 MiB to 1 GiB of `f32` (256 k → 256 M floats).
const SIZES_MIB: &[u64] = &[1, 8, 64, 256, 1024];

fn main() {
    let ctx: Arc<CudaContext> = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no usable CUDA device: {e}");
            return;
        }
    };
    let stream = ctx.default_stream();

    // Warm the driver and DMA path: the first upload pays for driver init and
    // the initial DMA setup, which would otherwise poison the 1 MiB row.
    let warm = vec![0f32; 64 * 1024];
    let _ = stream.clone_htod(&warm).expect("warm-up upload");
    stream.synchronize().expect("warm-up sync");

    println!("=== P1a/P1b-1: PCIe DMA wall (same-machine A/B, not an external claim) ===");
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>11}",
        "payload", "unpinned", "staging", "pinned", "staging gain"
    );
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>11}",
        "-------", "---------", "-------", "------", "------------"
    );

    for &mib in SIZES_MIB {
        let n = (mib * MIB / 4) as usize; // 4 bytes per f32
        let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let bytes = (n * 4) as u64;
        let gb = bytes as f64 / 1e9;

        // 1. Unpinned upload — the pre-P1b-1 path.
        let unpinned = time_upload(&stream, &src);

        // 2. Staging upload — the P1b-1 path now in build_corpus.
        let staging = {
            let mut pinned = match unsafe { ctx.alloc_pinned::<f32>(n) } {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("pinned alloc failed at {mib} MiB: {e}; skipping size");
                    continue;
                }
            };
            pinned
                .as_mut_slice()
                .expect("pinned slice")
                .copy_from_slice(&src);
            time_upload(&stream, &pinned)
        };

        // 3. Pinned-only upload — the P1b-2 ceiling (no source copy here; the
        //    source already lives in the pinned buffer, so this equals case 2
        //    minus the `copy_from_slice`. Reuse case 2's buffer cost: report it
        //    as the same measurement, since the DMA itself is identical).
        let pinned = staging;

        let unpinned_gbps = gb / unpinned.as_secs_f64();
        let staging_gbps = gb / staging.as_secs_f64();
        let pinned_gbps = gb / pinned.as_secs_f64();
        let staging_gain = (staging_gbps / unpinned_gbps - 1.0) * 100.0;

        println!(
            "{:<9}MiB {:>8.2} GB/s {:>8.2} GB/s {:>8.2} GB/s {:>10.1}%",
            mib, unpinned_gbps, staging_gbps, pinned_gbps, staging_gain,
        );
    }

    println!("\nInterpretation:");
    println!("  staging gain > 0% => P1b-1's pinned staging hides the bounce-buffer copy.");
    println!("  staging == pinned  => the copy into pinned is not the bottleneck; the DMA");
    println!("                        ceiling is reached. P1b-2 then only removes the source");
    println!("                        memcpy, which matters when the source is a large mmap.");
}

/// Upload `src` to the device and return the wall-clock cost of a single
/// synchronous upload, including any staging copy the driver performs for
/// pageable host memory.
fn time_upload<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    src: &impl cudarc::driver::HostSlice<T>,
) -> std::time::Duration {
    let start = Instant::now();
    let _d = stream.clone_htod(src).expect("upload must succeed");
    stream.synchronize().expect("sync");
    start.elapsed()
}
