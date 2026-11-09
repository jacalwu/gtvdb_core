//! Stage A–D of the graph-load pipeline, measured where each stage lives.
//!
//! `cargo run --release -p gtv-engine --example bench_graph_load -- 1000000`
//! `cargo run --release -p gtv-engine --example bench_graph_load --features cuda -- 1000000`
//!
//! # What this exists to answer
//!
//! "The upload is 30 ms" is only actionable if the upload is the largest thing
//! in the cold path. This walks the four stages that precede a kernel and prices
//! each one:
//!
//! ```text
//! A  live_rows_at(t)      mmap the segments, walk the system-time versions
//! B  rows -> TemporalCSR  per-row records -> five column arrays
//! C  CSR -> SnapshotView  a borrow; must be O(1)
//! D  upload               per column, measured by the backend's own receipt
//! ```
//!
//! Stage B is the one worth watching. [`SegmentStore::graph_at`] materialises a
//! `Vec<EdgeRecord>` — one array-of-structs copy of every live edge — and then
//! folds it into the CSR's struct-of-arrays. That is the "SoA -> AoS -> SoA"
//! round trip a loader must not have, and it is *not* in the upload: it is one
//! snapshot-publish cost, paid once and amortised over every query on that
//! snapshot. This example prices it rather than assuming it is small, because if
//! it is not small, no amount of CUDA work changes the cold path.
//!
//! The store is real (segments on disk, a manifest, a WAL) because the mmap read
//! is part of stage A. The graph is synthetic: out-degree 4, one interval, all
//! live at `t = 0`. Shape does not change a copy's cost — only bytes do.

use std::mem::size_of;
use std::time::Instant;

use gtv_accelerator::SnapshotView;
use gtv_engine::accelerator::registry;
use gtv_storage::{EdgeRecord, SegmentStore};

fn main() {
    let edges: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let nodes = (edges / 4).max(4);

    // A fresh directory every run: the pid is reused across invocations, and a
    // stale manifest from an earlier run (a `remove_dir_all` that lost to an
    // open mmap on Windows) makes `flush_at` fail with "goes backwards" — a
    // confusing way to learn that the fixture was not clean.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let root = std::env::temp_dir().join(format!("gtv_bench_load_{stamp}"));
    let _ = std::fs::remove_dir_all(&root);

    // -- ingest ------------------------------------------------------------
    // Batched: the ingest path is not what is being measured, and one 1e7-record
    // `Vec<EdgeRecord>` would be half a gigabyte of test fixture.
    let t = Instant::now();
    let mut store = SegmentStore::open(&root, nodes).unwrap();
    for chunk in 0..edges.div_ceil(100_000) {
        let lo = chunk * 100_000;
        let hi = (lo + 100_000).min(edges);
        let batch: Vec<EdgeRecord> = (lo..hi)
            .map(|i| EdgeRecord::new((i % nodes) as u64, ((i * 3) % nodes) as u64, 0, 1000, 1, 1))
            .collect();
        store.ingest(1, 1, &batch).unwrap();
    }
    // The head, not a constant: `ingest` flushes on its own once the in-memory
    // buffer fills, so a large graph is already published at a wall-clock point
    // before this line runs, and `flush_at(1)` would be rejected as going
    // backwards. Whatever the store says its head is, that is the point the
    // materialisation has to be measured at.
    let head = store.system_ts().max(1);
    store.flush_at(head).unwrap();
    println!(
        "store: {} nodes, {} edges ingested + flushed in {:.1} ms",
        nodes,
        edges,
        ms(t.elapsed())
    );

    // -- A: rows at t ------------------------------------------------------
    let t = Instant::now();
    let rows = store.live_rows_at(head).unwrap();
    let rows_ms = ms(t.elapsed());
    println!(
        "A live_rows_at         {:>8.2} ms  {} rows, {} MiB of LiveRow ({:.1} B/row)",
        rows_ms,
        rows.len(),
        rows.len() * size_of::<gtv_storage::LiveRow>() / (1 << 20),
        size_of::<gtv_storage::LiveRow>() as f64
    );
    let row_bytes = rows.len() * size_of::<gtv_storage::LiveRow>();
    drop(rows);

    // -- A+B: the materialisation as the engine calls it -------------------
    // Twice, because the first call also faults the segments in from disk. The
    // second is what a resident snapshot would have cost had it been published
    // while the process was warm — and the gap between them is the difference
    // between "this is the mmap" and "this is the rebuild".
    let t = Instant::now();
    let csr = store.graph_at(head).unwrap();
    let graph_cold_ms = ms(t.elapsed());
    let t = Instant::now();
    drop(store.graph_at(head).unwrap());
    let graph_warm_ms = ms(t.elapsed());
    let graph_ms = graph_warm_ms;
    let csr_bytes = csr.columns().offsets.len() * 4
        + csr.columns().dst.len() * 8
        + csr.columns().valid_from.len() * 8
        + csr.columns().valid_to.len() * 8
        + csr.columns().edge_type.len() * 2;
    println!(
        "A+B graph_at           {:>8.2} ms  {} edges -> {} MiB SoA ({} MiB of it was an AoS detour)",
        graph_warm_ms,
        csr.edge_count(),
        csr_bytes / (1 << 20),
        row_bytes / (1 << 20)
    );
    println!(
        "    cold {:.2} ms / warm {:.2} ms  = {:.1} ns per edge warm; AoS detour cost {:.1} ms of the {:.1}",
        graph_cold_ms,
        graph_warm_ms,
        graph_warm_ms * 1e6 / csr.edge_count().max(1) as f64,
        graph_warm_ms - rows_ms,
        graph_warm_ms
    );

    // -- C: the upload view -------------------------------------------------
    // Must be O(1): a borrow of the CSR's own columns, no per-edge loop. If this
    // ever grows with `edges`, the loader is copying and the copy is the bug.
    let csr = csr.with_snapshot_id(7);
    let t = Instant::now();
    let view = SnapshotView::from_csr(&csr).unwrap();
    let view_us = t.elapsed().as_secs_f64() * 1e6;
    println!(
        "C  view                {:>8.2} us  {} B priced, no allocation (O(1) by construction)",
        view_us,
        view.bytes()
    );
    assert!(
        view_us < 1_000.0,
        "building the upload view took {view_us} us: it is copying"
    );

    // -- D: upload, from the backend's own receipt -------------------------
    let reg = registry();
    let Some(dev) = reg.devices().first() else {
        println!(
            "D  upload                    n/a  no device registered (build with --features cuda)"
        );
        println!(
            "\ncold path here: {:.2} ms of CPU materialisation and no transfer at all.",
            graph_ms
        );
        cleanup(&root);
        return;
    };
    let t = Instant::now();
    let handle = dev.load_graph(view).unwrap().expect("residency supported");
    let load_ms = ms(t.elapsed());
    let m = dev
        .last_load_metrics()
        .expect("a device upload leaves a receipt")
        .with_materialize_ns((graph_ms * 1e6) as u64);
    dev.release_graph(&handle);

    println!(
        "D  upload              {:>8.2} ms  {:.1} B/edge at {:.2} GB/s ({:.1}% of the load is PCIe)",
        load_ms,
        m.uploaded_bytes_per_edge(),
        m.upload_gbps(),
        m.upload_share() * 100.0
    );
    println!(
        "   stages: bind {:.3} + alloc {:.2} + copy {:.2} + sync {:.3} = {:.2} ms; \
         pack/transform/staging {} / {} / {}",
        m.bind_ns as f64 / 1e6,
        m.device_alloc_ns as f64 / 1e6,
        m.upload_ns() as f64 / 1e6,
        m.device_sync_ns as f64 / 1e6,
        m.total_ns as f64 / 1e6,
        m.host_pack_ns,
        m.layout_transform_ns,
        m.staging_bytes
    );

    let cold = graph_ms + load_ms;
    let scale = edges as f64 / 1e7;
    let saving = (6.86 * scale - 0.556 * scale).max(1e-9);
    println!(
        "\ncold path: {:.2} ms = {:.2} materialise + {:.2} upload ({:.0}% materialise)",
        cold,
        graph_ms,
        load_ms,
        graph_ms / cold * 100.0
    );
    println!(
        "per query after that: kernel only. Break-even at {:.1} invocations per snapshot \
         (CPU saving {:.2} ms/query).",
        cold / saving,
        saving
    );

    cleanup(&root);
}

fn cleanup(root: &std::path::Path) {
    // The store holds open mmaps; deleting on Windows is best-effort and a
    // leftover temp directory is not a failure worth reporting.
    let _ = std::fs::remove_dir_all(root);
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}
