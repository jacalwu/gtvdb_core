//! Cold-open / steady-traversal baseline for GTVSeg (`bench_segment`).
//!
//! Run with:
//! ```text
//! cargo run -p gtv-storage --release --example bench_segment -- [edge_count]
//! ```
//!
//! This is the acceptance baseline demanded by
//! `docs/storage-mmap-design.md` §5. TC1–TC15 explicitly exclude index build
//! and I/O, so before this example there was **no** cold-load number at all.
//!
//! Two gates:
//! * `open_segment` must be orders of magnitude cheaper than `from_arrays`
//!   (the latter sorts `O(E log E)` and allocates five vectors).
//! * steady traversal on the mapped backend must **not regress** versus the
//!   owned backend — that is what validates the `Col<T>` + `Deref` design.
//!
//! Debug builds are dominated by bounds checks and are not meaningful; use
//! `--release`. Numbers are indicative: OS page cache state, huge pages and
//! storage medium all affect them.

use std::hint::black_box;
use std::time::Instant;

use arrow::array::{TimestampNanosecondArray, UInt16Array, UInt64Array};
use gtv_core::TemporalCSR;
use gtv_storage::segment::{open_segment, write_segment};

const NODE_COUNT: usize = 100_000;

fn build_arrays(n: usize) -> (UInt64Array, UInt64Array, TimestampNanosecondArray, TimestampNanosecondArray, UInt16Array) {
    // Deterministic, already scattered across sources so the sort has real
    // work to do (this is what from_arrays pays for on every cold start).
    let mut src = Vec::with_capacity(n);
    let mut dst = Vec::with_capacity(n);
    let mut vf = Vec::with_capacity(n);
    let mut vt = Vec::with_capacity(n);
    let mut et = Vec::with_capacity(n);
    for i in 0..n {
        src.push(((i as u64 * 7919) % NODE_COUNT as u64) as u64);
        dst.push(((i as u64 * 104729 + 13) % NODE_COUNT as u64) as u64);
        let t = (i % 1000) as i64;
        vf.push(t);
        vt.push(t + 500 + (i % 97) as i64);
        et.push((i % 7) as u16);
    }
    (
        UInt64Array::from(src),
        UInt64Array::from(dst),
        TimestampNanosecondArray::from(vf),
        TimestampNanosecondArray::from(vt),
        UInt16Array::from(et),
    )
}

/// Sum every active neighbor's dst over a set of time points. Uses
/// `neighbors()` (binary search + zone maps) on both backends identically.
fn steady_traversal(csr: &TemporalCSR, rounds: usize) -> u64 {
    let mut acc = 0u64;
    for _ in 0..rounds {
        for t in [0i64, 250, 500, 750, 1000] {
            for s in 0..NODE_COUNT as u64 {
                for nb in csr.neighbors(s, t).unwrap() {
                    acc = acc.wrapping_add(nb.dst);
                }
            }
        }
    }
    black_box(acc)
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    let (src, dst, vf, vt, et) = build_arrays(n);

    let t0 = Instant::now();
    let owned = TemporalCSR::from_arrays(&src, &dst, &vf, &vt, &et, NODE_COUNT).unwrap();
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let path = std::env::temp_dir().join(format!("gtv_bench_{}.seg", std::process::id()));
    let t1 = Instant::now();
    write_segment(&path, &owned).unwrap();
    let write_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let file_bytes = std::fs::metadata(&path).unwrap().len();

    // Cold open: open + mmap + header validation + CSR structural checks.
    // Nothing is decoded, copied or sorted.
    let t2 = Instant::now();
    let seg = open_segment(&path).unwrap();
    let open_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let t3 = Instant::now();
    seg.verify_data().unwrap();
    let verify_ms = t3.elapsed().as_secs_f64() * 1000.0;

    // ---- First traversal -------------------------------------------------
    // An owned Vec is already hot on the heap; the first traversal over a fresh
    // mmap pays page faults. Comparing these two directly (as an earlier draft
    // of this bench did) produces a spurious ~1.9x "regression" that is really
    // I/O, not the `Col<T>` indirection. Measure it separately — this is the
    // "First traversal" row in docs/storage-mmap-design.md §5.
    let t4 = Instant::now();
    let first_owned = black_box(steady_traversal(&owned, 1));
    let first_owned_ms = t4.elapsed().as_secs_f64() * 1000.0;

    let t5 = Instant::now();
    let first_mapped = black_box(steady_traversal(seg.csr(), 1));
    let first_mapped_ms = t5.elapsed().as_secs_f64() * 1000.0;

    // ---- Steady traversal: warm both, interleave, take the min -----------
    // The acceptance gate is on *steady* state. Warm-up removes page faults;
    // interleaving + min-of-3 removes scheduler / frequency-scaling noise.
    let _ = black_box(steady_traversal(&owned, 1));
    let _ = black_box(steady_traversal(seg.csr(), 1));
    let mut owned_best = f64::INFINITY;
    let mut mapped_best = f64::INFINITY;
    let mut checksums = (0u64, 0u64);
    for _ in 0..3 {
        let t = Instant::now();
        let a = steady_traversal(&owned, 1);
        owned_best = owned_best.min(t.elapsed().as_secs_f64() * 1000.0);
        let t = Instant::now();
        let b = steady_traversal(seg.csr(), 1);
        mapped_best = mapped_best.min(t.elapsed().as_secs_f64() * 1000.0);
        checksums = (a, b);
    }
    let (a, b) = checksums;

    println!("GTVSeg baseline  (edges={n}, nodes={NODE_COUNT})");
    println!("  segment file          : {:.2} MB", file_bytes as f64 / 1_048_576.0);
    println!("  from_arrays (build)   : {build_ms:10.3} ms");
    println!("  write_segment         : {write_ms:10.3} ms");
    println!("  open_segment (cold)   : {open_ms:10.3} ms   <-- gate 1");
    println!("  verify_data (full)    : {verify_ms:10.3} ms   (not on open path)");
    println!("  cold-open speedup     : {:10.1} x", build_ms / open_ms.max(0.001));
    println!("  first traversal owned : {first_owned_ms:10.3} ms");
    println!("  first traversal mapped: {first_mapped_ms:10.3} ms   (page faults)");
    println!("  steady owned  (min/3) : {owned_best:10.3} ms");
    println!("  steady mapped (min/3) : {mapped_best:10.3} ms   <-- gate 2");
    println!("  mapped / owned ratio  : {:10.3} x", mapped_best / owned_best.max(0.001));
    println!("  checksums owned/mapped: {a} / {b}  {}", if a == b { "(equal)" } else { "(MISMATCH!)" });
    println!("  first-traversal sum   : {first_owned} / {first_mapped}  {}", if first_owned == first_mapped { "(equal)" } else { "(MISMATCH!)" });

    assert_eq!(a, b, "mapped backend changed steady traversal results");
    assert_eq!(first_owned, first_mapped, "mapped backend changed first traversal results");
    drop(seg);
    std::fs::remove_file(&path).ok();
}
