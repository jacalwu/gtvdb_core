//! V-4: end-to-end cold-open benchmark for the vector corpus.
//!
//! The question this answers is narrow and concrete: **what does moving the
//! corpus onto an mmap actually buy?** The `.gtvidx` container used to be
//! loaded by reading the whole file into a `Vec<u8>`, splitting it, and
//! decoding the payload into a second and third copy of the corpus. The
//! mmap path instead maps the container and reinterprets the corpus columns
//! in place.
//!
//! For each corpus size this measures three things per read path:
//!
//! | column | what it times |
//! |---|---|
//! | open | from "I want the index" to "I hold a queryable index" |
//! | first query | the first search — this is where an mmap pays its page faults |
//! | steady query | the search cost once the pages are resident |
//!
//! and it reports the process's private working set after the load, which is
//! where the "three copies vs one mapping" claim becomes a number rather than
//! a slogan.
//!
//! # Honesty notes
//!
//! * These are **same-machine A/B numbers**, not an external performance
//!   claim. Run them on the deployment target and compare against your own
//!   baseline.
//! * The page cache is **not** dropped between runs (there is no portable
//!   way to do that). The mmap rows therefore report two cases: *cold pages*
//!   (the first query after the mapping) and *warm pages* (the same mapping
//!   queried again). The OS cache state is whatever the machine had, so treat
//!   the cold row as "cache miss for this corpus", not "cache dropped".
//! * The legacy path is simulated faithfully — it is the exact code the store
//!   used before the mmap dispatch: `fs::read` → `decode` → `from_bytes`.
//!
//! ```sh
//! RUSTFLAGS= RUSTC_WRAPPER= CARGO_INCREMENTAL=0 \
//!   cargo run -p gtv-index-store --release --example bench_cold_open
//! ```

use std::time::{Duration, Instant};

use gtv_core::{Metric, VectorIndex};
use gtv_index::{AnyIndex, BuildOptions, IvfIndex};
use gtv_index_store::container::decode;
use gtv_index_store::{IndexMeta, IndexStore};

/// A corpus shape to measure.
struct Shape {
    rows: usize,
    dim: usize,
}

impl Shape {
    fn label(&self) -> String {
        format!("{} x {}", self.rows, self.dim)
    }
    fn corpus_bytes(&self) -> u64 {
        (self.rows * self.dim * 4) as u64
    }
}

/// Deterministic pseudo-random corpus (no external RNG dependency).
fn corpus(rows: usize, dim: usize) -> (Vec<u64>, Vec<Vec<f32>>) {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let ids: Vec<u64> = (0..rows as u64).collect();
    let vectors: Vec<Vec<f32>> = (0..rows)
        .map(|_| (0..dim).map(|_| next()).collect())
        .collect();
    (ids, vectors)
}

fn meta() -> IndexMeta {
    IndexMeta {
        model_id: "bench".into(),
        model_version: "1".into(),
        embedding_model: "bench".into(),
        ..Default::default()
    }
}

/// Time one query round-trip.
fn time_query(index: &AnyIndex, query: &[f32], k: usize) -> Duration {
    let start = Instant::now();
    let hits = index.search(query, k, None).expect("search must succeed");
    std::hint::black_box(hits);
    start.elapsed()
}

fn fmt_ms(d: Duration) -> String {
    format!("{:>9.3} ms", d.as_secs_f64() * 1000.0)
}

fn fmt_mib(bytes: u64) -> String {
    format!("{:>9.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

/// The resident working set of this process, in bytes.
///
/// This is the number that separates "the corpus was materialized into the
/// process" from "the corpus is a mapping": the legacy path's file bytes,
/// payload bytes and decoded corpus all become resident private pages,
/// while the mmap path's corpus pages are file-backed (they belong to the OS
/// page cache and are evictable under pressure).
fn private_bytes() -> u64 {
    platform_memory::working_set_bytes()
}

/// Run one corpus shape through both read paths.
fn measure_flat(shape: &Shape, root: &std::path::Path) {
    let (ids, vectors) = corpus(shape.rows, shape.dim);
    let query = vectors[shape.rows / 2].clone();

    let store = IndexStore::open(root.to_path_buf()).expect("store");
    let index = AnyIndex::build(ids, vectors, Metric::L2, &BuildOptions::Flat).expect("build");
    store
        .save("bench", &index, &BuildOptions::Flat, &meta())
        .expect("save");
    let id = store.resolve("bench").expect("resolve");
    let version_dir = store.root().join(id.to_string()).join("v1");
    let mmap_path = version_dir.join("index.gtvidx");
    let manifest_json = std::fs::read(version_dir.join("manifest.json")).expect("manifest");

    // The legacy container: the same manifest, but the pre-V-2 self-contained
    // `GFLATv1` payload. This is exactly what the store used to write.
    let legacy_path = root.join("legacy.gtvidx");
    std::fs::write(
        &legacy_path,
        gtv_index_store::container::encode(&manifest_json, &index.to_bytes()),
    )
    .expect("write legacy container");

    println!(
        "\n--- Flat: {} ({}) ---",
        shape.label(),
        fmt_mib(shape.corpus_bytes())
    );

    // --- legacy path: fs::read + decode + materialize ---
    let before = private_bytes();
    let t0 = Instant::now();
    let bytes = std::fs::read(&legacy_path).expect("read");
    let (manifest_json, payload) = decode(&bytes).expect("decode");
    let manifest: gtv_index_store::IndexManifest =
        serde_json::from_slice(&manifest_json).expect("manifest");
    let legacy = AnyIndex::from_bytes(manifest.index_type, &payload).expect("from_bytes");
    let legacy_open = t0.elapsed();
    let legacy_after = private_bytes();
    let legacy_first = time_query(&legacy, &query, 10);
    let legacy_steady = {
        let start = Instant::now();
        for _ in 0..5 {
            time_query(&legacy, &query, 10);
        }
        start.elapsed() / 5
    };
    drop(legacy);
    drop(bytes);

    // --- mmap path (first touch: pages may be cold) ---
    let t1 = Instant::now();
    let mapped = gtv_index_store::open_mapped_payload(&mmap_path)
        .expect("open mapped")
        .expect("mmap payload");
    let mmap_open = t1.elapsed();
    let mmap_after = private_bytes();
    let mmap_first = time_query(&mapped, &query, 10);
    let mmap_steady = {
        let start = Instant::now();
        for _ in 0..5 {
            time_query(&mapped, &query, 10);
        }
        start.elapsed() / 5
    };

    println!(
        "  {:<22} {:>12} {:>13} {:>13}",
        "read path", "open", "first query", "steady query"
    );
    println!(
        "  {:<22} {:>12} {:>13} {:>13}",
        "legacy (read+decode)",
        fmt_ms(legacy_open),
        fmt_ms(legacy_first),
        fmt_ms(legacy_steady)
    );
    println!(
        "  {:<22} {:>12} {:>13} {:>13}",
        "mmap (page cache)",
        fmt_ms(mmap_open),
        fmt_ms(mmap_first),
        fmt_ms(mmap_steady)
    );
    println!(
        "  resident working set after load: legacy {:} (delta {}), mmap {:} (delta {})",
        fmt_mib(legacy_after),
        fmt_mib(legacy_after.saturating_sub(before)),
        fmt_mib(mmap_after),
        fmt_mib(mmap_after.saturating_sub(before)),
    );
}

/// Run one corpus shape through the IVF paths.
fn measure_ivf(shape: &Shape, root: &std::path::Path) {
    let (ids, vectors) = corpus(shape.rows, shape.dim);
    let query = vectors[shape.rows / 2].clone();

    // Uniform assignment keeps the build cheap (no k-means training), so the
    // benchmark measures the *read* path rather than the trainer.
    let mut data = Vec::with_capacity(shape.rows * shape.dim);
    for v in &vectors {
        data.extend_from_slice(v);
    }
    let nlist = 64.min(shape.rows);
    let ivf = IvfIndex::with_uniform(ids, data, shape.dim, nlist, 8, Metric::L2).expect("ivf");
    let index = AnyIndex::Ivf(ivf);

    let store = IndexStore::open(root.to_path_buf()).expect("store");
    let opts = BuildOptions::Ivf {
        nlist,
        nprobe: 8,
    };
    store.save("bench", &index, &opts, &meta()).expect("save");
    let id = store.resolve("bench").expect("resolve");
    let version_dir = store.root().join(id.to_string()).join("v1");
    let mmap_path = version_dir.join("index.gtvidx");
    let manifest_json = std::fs::read(version_dir.join("manifest.json")).expect("manifest");

    // Legacy container: the pre-V-4 self-contained `GIVFv2` payload.
    let legacy_path = root.join("legacy.gtvidx");
    std::fs::write(
        &legacy_path,
        gtv_index_store::container::encode(&manifest_json, &index.to_bytes()),
    )
    .expect("write legacy container");

    println!(
        "\n--- IVF: {} ({}), nlist={} ---",
        shape.label(),
        fmt_mib(shape.corpus_bytes()),
        nlist
    );

    // --- legacy path ---
    let before = private_bytes();
    let t0 = Instant::now();
    let bytes = std::fs::read(&legacy_path).expect("read");
    let (manifest_json, payload) = decode(&bytes).expect("decode");
    let manifest: gtv_index_store::IndexManifest =
        serde_json::from_slice(&manifest_json).expect("manifest");
    let legacy = AnyIndex::from_bytes(manifest.index_type, &payload).expect("from_bytes");
    let legacy_open = t0.elapsed();
    let legacy_after = private_bytes();
    let legacy_first = time_query(&legacy, &query, 10);
    drop(legacy);
    drop(bytes);

    // --- mmap path ---
    let t1 = Instant::now();
    let mapped = gtv_index_store::open_mapped_payload(&mmap_path)
        .expect("open mapped")
        .expect("mmap payload");
    let mmap_open = t1.elapsed();
    let mmap_after = private_bytes();
    let mmap_first = time_query(&mapped, &query, 10);
    let mmap_steady = {
        let start = Instant::now();
        for _ in 0..5 {
            time_query(&mapped, &query, 10);
        }
        start.elapsed() / 5
    };

    println!(
        "  {:<22} {:>12} {:>13} {:>13}",
        "read path", "open", "first query", "steady query"
    );
    println!(
        "  {:<22} {:>12} {:>13} {:>13}",
        "legacy (read+decode)",
        fmt_ms(legacy_open),
        fmt_ms(legacy_first),
        "-"
    );
    println!(
        "  {:<22} {:>12} {:>13} {:>13}",
        "mmap (page cache)",
        fmt_ms(mmap_open),
        fmt_ms(mmap_first),
        fmt_ms(mmap_steady)
    );
    println!(
        "  resident working set after load: legacy {:} (delta {}), mmap {:} (delta {})",
        fmt_mib(legacy_after),
        fmt_mib(legacy_after.saturating_sub(before)),
        fmt_mib(mmap_after),
        fmt_mib(mmap_after.saturating_sub(before)),
    );
}

fn main() {
    println!("=== V-4: cold-open benchmark for the vector corpus ===");
    println!("(same-machine A/B only — never quote these as an external claim)");

    let root = std::env::temp_dir().join(format!("gtv_cold_open_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);

    // Flat scales to a realistic corpus size; IVF stops earlier because its
    // build (uniform assignment over the whole corpus) is O(rows × nlist).
    let flat_shapes = [
        Shape {
            rows: 10_000,
            dim: 64,
        },
        Shape {
            rows: 100_000,
            dim: 128,
        },
        Shape {
            rows: 1_000_000,
            dim: 128,
        },
    ];
    for shape in &flat_shapes {
        let dir = root.join(format!("flat_{}_{}", shape.rows, shape.dim));
        measure_flat(shape, &dir);
    }

    let ivf_shapes = [
        Shape {
            rows: 10_000,
            dim: 64,
        },
        Shape {
            rows: 100_000,
            dim: 128,
        },
    ];
    for shape in &ivf_shapes {
        let dir = root.join(format!("ivf_{}_{}", shape.rows, shape.dim));
        measure_ivf(shape, &dir);
    }

    let _ = std::fs::remove_dir_all(&root);

    println!("\nInterpretation:");
    println!("  open latency      : the mmap path validates a header instead of");
    println!("                      decoding the corpus — the gap grows with rows x dim.");
    println!("  first query       : the mmap path pays page faults here; the legacy");
    println!("                      path already paid them while decoding.");
    println!("  steady query      : both paths read the same bytes from the same page");
    println!("                      cache, so they converge.");
    println!("  resident bytes   : the legacy path holds the file bytes, the payload");
    println!("                      bytes and the materialized corpus; the mmap path");
    println!("                      holds a mapping (its pages are file-backed and");
    println!("                      evictable, so the OS can reclaim them).");
}

/// Platform-specific process-memory measurement.
mod platform_memory {
    /// Resident working set of this process, in bytes.
    pub fn working_set_bytes() -> u64 {
        #[cfg(windows)]
        {
            windows_working_set_bytes()
        }
        #[cfg(not(windows))]
        {
            unix_working_set_bytes()
        }
    }

    #[cfg(windows)]
    fn windows_working_set_bytes() -> u64 {
        // `PROCESS_MEMORY_COUNTERS` from psapi.h: `cb` and `PageFaultCount`
        // are DWORDs here (the EX variant widens them to SIZE_T).
        #[repr(C)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }

        #[link(name = "psapi")]
        extern "system" {
            fn GetProcessMemoryInfo(
                process: *mut core::ffi::c_void,
                counters: *mut ProcessMemoryCounters,
                size: u32,
            ) -> i32;
            fn GetCurrentProcess() -> *mut core::ffi::c_void;
        }

        let mut counters = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            page_fault_count: 0,
            peak_working_set_size: 0,
            working_set_size: 0,
            quota_peak_paged_pool_usage: 0,
            quota_paged_pool_usage: 0,
            quota_peak_non_paged_pool_usage: 0,
            quota_non_paged_pool_usage: 0,
            pagefile_usage: 0,
            peak_pagefile_usage: 0,
        };
        // SAFETY: `counters` is a correctly sized, correctly initialized
        // PROCESS_MEMORY_COUNTERS; the call only writes into it.
        let ok = unsafe {
            GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut counters,
                std::mem::size_of::<ProcessMemoryCounters>() as u32,
            )
        };
        if ok == 0 {
            return 0;
        }
        counters.working_set_size as u64
    }

    #[cfg(unix)]
    fn unix_working_set_bytes() -> u64 {
        // `/proc/self/statm`: size resident shared text lib data dt (in pages).
        let Ok(statm) = std::fs::read_to_string("/proc/self/statm") else {
            return 0;
        };
        let mut fields = statm.split_whitespace();
        let _size: u64 = fields.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let resident: u64 = fields.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        resident.saturating_mul(4096)
    }
}
