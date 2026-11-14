//! P1a / P1b-1 / P1b-2 — quantify the PCIe DMA wall that the SSD→RAM→VRAM
//! tiering must cross, and measure what the pinned-staging (P1b-1) and
//! `cudaHostRegister` (P1b-2 / V-3) paths win back.
//!
//! This is a *measurement*, not an optimization. Four numbers per payload size:
//!
//! 1. **unpinned H2D** — a plain `Vec<f32>` uploaded with `clone_htod`. This was
//!    the path `CudaBackend::build_corpus` took before P1b-1: the driver stages
//!    the pageable source through an internal bounce buffer before the DMA.
//! 2. **staging H2D** — the P1b-1 path now in `build_corpus`: copy the `Vec`
//!    into a write-combined `cudaHostAlloc` buffer, then `clone_htod` the pinned
//!    buffer. The copy is a host memcpy (no bounce), and the DMA runs at the
//!    pinned ceiling.
//! 3. **pinned H2D** — `cudaHostAlloc` alone (no source copy), the upper bound.
//! 4. **register H2D** — `cuMemHostRegister` over a file `mmap` (the `Col::Mapped`
//!    source V-3 unlocks), then `clone_htod`. This is the P1b-2 zero-copy path:
//!    no source copy *and* no bounce buffer.
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
use memmap2::Mmap;

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

    println!("=== P1a/P1b-1/P1b-2: PCIe DMA wall (same-machine A/B, not an external claim) ===");
    println!(
        "{:<10} {:>11} {:>11} {:>11} {:>11} {:>12}",
        "payload", "unpinned", "staging", "pinned", "register", "register gain"
    );
    println!(
        "{:<10} {:>11} {:>11} {:>11} {:>11} {:>12}",
        "-------", "---------", "-------", "------", "--------", "-------------"
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
        //    minus the `copy_from_slice`).
        let pinned = staging;

        // 4. Register upload — `cudaHostRegister` over a file `mmap`, then
        //    `clone_htod` (the V-3 zero-copy path). The mmap mirrors a
        //    `Col::Mapped` GTVVEC column: pageable, immutable, file-backed.
        //    Windows WDDM cannot pin file-mapped pages (`CUDA_ERROR_OUT_OF_MEMORY`),
        //    so this path reports N/A there and a real number on Linux/TCC.
        let register = {
            let mapped = scratch_mmap(n, &src);
            let view: &[f32] = unsafe {
                std::slice::from_raw_parts(mapped.as_ptr() as *const f32, n)
            };
            time_upload_register(&stream, view)
        };

        let unpinned_gbps = gb / unpinned.as_secs_f64();
        let staging_gbps = gb / staging.as_secs_f64();
        let pinned_gbps = gb / pinned.as_secs_f64();
        let (register_str, register_gain_str) = match register {
            Some(d) => {
                let gbps = gb / d.as_secs_f64();
                let gain = (gbps / unpinned_gbps - 1.0) * 100.0;
                (format!("{gbps:>7.2} GB/s"), format!("{gain:>11.1}%"))
            }
            None => ("     N/A".to_string(), "        N/A".to_string()),
        };

        println!(
            "{:<9}MiB {:>7.2} GB/s {:>7.2} GB/s {:>7.2} GB/s {:>12} {:>12}",
            mib, unpinned_gbps, staging_gbps, pinned_gbps, register_str, register_gain_str,
        );
    }

    println!("\nInterpretation:");
    println!("  staging gain > 0%     => P1b-1's pinned staging hides the bounce-buffer copy.");
    println!("  register ~= pinned    => P1b-2's cudaHostRegister reaches the DMA ceiling with");
    println!("                           zero source copy (the mmap page is pinned in place).");
    println!("  register = N/A        => this driver/OS cannot pin file-mapped pages (Windows");
    println!("                           WDDM). The register path falls back to staging; the");
    println!("                           residual value of register is Linux/TCC only.");
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

/// Register `mapped` (a file mmap) and upload it, timing register + DMA + sync.
///
/// Delegates to `gtv_accelerator::cuda::register_and_upload`, the same wrapper
/// the production upload path uses: `cuMemHostRegister` pins the mapped pages in
/// place, then `clone_htod` reads them directly.
///
/// Returns `None` when the driver refuses to pin file-mapped pages (Windows
/// WDDM returns a misleading `CUDA_ERROR_OUT_OF_MEMORY`); the caller prints N/A.
fn time_upload_register<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    mapped: &[T],
) -> Option<std::time::Duration> {
    let start = Instant::now();
    gtv_accelerator::cuda::register_and_upload(stream, mapped).ok()?;
    Some(start.elapsed())
}

/// Create a file-backed `mmap` of `n` `f32`s seeded from `src`, to emulate a
/// `Col::Mapped` GTVVEC column (pageable, immutable, file-backed).
fn scratch_mmap(n: usize, src: &[f32]) -> Mmap {
    use std::io::Write;
    let path = std::env::temp_dir().join(format!(
        "bench_pcie_transfer_{}_{}.bin",
        std::process::id(),
        n
    ));
    let mut f = std::fs::File::create(&path).expect("create scratch file");
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, std::mem::size_of_val(src))
    };
    f.write_all(bytes).expect("write scratch file");
    f.sync_all().expect("flush scratch file");
    let f = std::fs::File::open(&path).expect("reopen scratch file");
    // SAFETY: the file is a private scratch file written above and not mutated
    // for the lifetime of this mapping; the map is dropped (unmapped) when the
    // returned `Mmap` is dropped, which is after the measurement completes.
    let map = unsafe { memmap2::Mmap::map(&f) }.expect("mmap scratch file");
    let _ = std::fs::remove_file(&path); // unlink immediately; the map keeps it alive
    map
}
