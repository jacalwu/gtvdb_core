//! Stage A–D of the graph-load pipeline, measured where each stage lives.
//!
//! `cargo run --release -p gtv-engine --example bench_graph_load -- 10000000`
//! `cargo run --release -p gtv-engine --example bench_graph_load --features cuda -- 10000000`
//!
//! # What this exists to answer
//!
//! "The upload is 30 ms" is only actionable if the upload is the largest thing
//! in the cold path. This walks the stages that precede a kernel and prices each
//! one:
//!
//! ```text
//! A  live_rows_at(t)      mmap the segments, walk the system-time versions
//! B  graph_at(t)          the same answer as columns, not rows
//! C  CSR -> SnapshotView  a borrow; must be O(1)
//! D  upload               per column, measured by the backend's own receipt
//! ```
//!
//! Stage B is measured **twice on purpose**: materialisation cost is not a
//! property of the graph, it is a property of how many segments the graph is
//! spread over.
//!
//! * Before compaction there are several baseline segments, so the answer is a
//!   k-way merge of their sorted runs, and the merge's per-edge comparison is
//!   the largest single item in it.
//! * After compaction there is one baseline segment that no retract touches, so
//!   the answer *is* that segment's columns and `graph_at` copies them.
//!
//! The gap between those two lines is the honest price of an uncompacted store,
//! and it is why `graph_at` is not a fixed cost to plan around.
//!
//! The store is real (segments on disk, a manifest, a WAL) because the mmap read
//! is part of stage A. The graph is synthetic: out-degree 4, one interval, all
//! live at the flush point. Shape does not change a copy's cost — only bytes do.

use std::time::Instant;

use gtv_accelerator::SnapshotView;
use gtv_core::csr::TemporalCSR;
use gtv_engine::accelerator::registry;
use gtv_storage::{EdgeRecord, MemSegment, SegmentStore};

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
    let runs = store.segments().len();
    println!(
        "store: {} nodes, {} edges ingested + flushed in {:.1} ms, {runs} live segments",
        nodes,
        edges,
        ms(t.elapsed())
    );

    // -- A: rows at t ------------------------------------------------------
    let t = Instant::now();
    let rows = store.live_rows_at(head).unwrap();
    let rows_ms = ms(t.elapsed());
    let row_bytes = rows.len() * std::mem::size_of::<gtv_storage::LiveRow>();
    println!(
        "A live_rows_at         {:>8.2} ms  {} rows, {} MiB of LiveRow ({:.1} B/row)",
        rows_ms,
        rows.len(),
        row_bytes >> 20,
        row_bytes as f64 / rows.len().max(1) as f64
    );

    // -- B: columns at t, as an uncompacted store answers it ---------------
    let mut merge_runs = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        drop(store.graph_at(head).unwrap());
        merge_runs.push(ms(t.elapsed()));
    }
    let merge_ms = median(&merge_runs);

    // The oracle is the row path — which is what `graph_at` used to be. It is
    // also the equivalence check: the merge has to reproduce it column for
    // column, not merely count the same edges.
    let t = Instant::now();
    let built = store.graph_at(head).unwrap();
    let oracle =
        MemSegment::to_csr(&rows.iter().map(|r| r.edge).collect::<Vec<_>>(), nodes).unwrap();
    let oracle_ms = ms(t.elapsed());
    assert_same(&built, &oracle);
    println!(
        "B graph_at (merge)     {:>8.2} ms  {} edges -> {} MiB SoA over {runs} runs \
         [{:.1} ns/edge]",
        merge_ms,
        built.edge_count(),
        column_bytes(&built) >> 20,
        merge_ms * 1e6 / built.edge_count().max(1) as f64
    );
    println!(
        "    equals the row path column-for-column; that path takes {:.2} ms ({:.1}x) and \
         {} MiB of LiveRow",
        oracle_ms,
        oracle_ms / merge_ms,
        row_bytes >> 20
    );
    drop(rows);
    drop(built);

    // -- B again, after compaction: one run, so nothing to merge -----------
    //
    // Small graphs fit in one flush and there is nothing to compact; then the
    // merge line above already *was* the copy line, and compacting again would
    // only add a rewrite. Say that instead of printing a 1.0x speedup.
    let plain_ms = if runs == 1 {
        println!("B graph_at (compacted)      n/a  one segment from the start: [merge] above is the copy path");
        merge_ms
    } else {
        let t = Instant::now();
        let plan = store
            .spawn_compaction(head + 1)
            .unwrap()
            .join()
            .unwrap()
            .unwrap();
        let report = store.apply_compaction(plan).unwrap();
        let compact_ms = ms(t.elapsed());
        let head = store.system_ts();
        // Retired segments stay in the manifest (marked superseded) so an older
        // snapshot can still be read; what matters is how many are *live*.
        let live = store
            .segments()
            .iter()
            .filter(|s| s.superseded_at.is_none())
            .count();
        assert_eq!(live, 1, "compaction left more than one live run");

        let mut plain_runs = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            drop(store.graph_at(head).unwrap());
            plain_runs.push(ms(t.elapsed()));
        }
        let plain_ms = median(&plain_runs);
        println!(
            "B graph_at (compacted) {:>8.2} ms  {} segments -> {}, {} rows [{:.1} ns/edge]",
            plain_ms,
            report.retired_segments,
            report.new_segments,
            report.rows,
            plain_ms * 1e6 / report.rows.max(1) as f64
        );
        println!(
            "    {:.1}x faster than merging {runs} runs; compaction itself {:.0} ms",
            merge_ms / plain_ms,
            compact_ms
        );
        plain_ms
    };

    // -- C: the upload view -------------------------------------------------
    // Must be O(1): a borrow of the CSR's own columns, no per-edge loop. If this
    // ever grows with `edges`, the loader is copying and the copy is the bug.
    let csr = store.graph_at(head).unwrap().with_snapshot_id(7);
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
            "\ncold path here: {:.2} ms of CPU materialisation (compacted) and no transfer at all.",
            plain_ms
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
        .with_materialize_ns((plain_ms * 1e6) as u64);
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

    let cold = plain_ms + load_ms;
    let scale = edges as f64 / 1e7;
    let saving = (6.86 * scale - 0.556 * scale).max(1e-9);
    println!(
        "\ncold path: {:.2} ms = {:.2} materialise + {:.2} upload ({:.0}% materialise); \
         break-even {:.1} queries",
        cold,
        plain_ms,
        load_ms,
        plain_ms / cold * 100.0,
        cold / saving
    );

    cleanup(&root);
}

/// The median of a handful of runs. Not the minimum: the first run after a
/// compaction shares the page cache with the compaction's own writes, which is
/// real but is not what the next query will see, and the fastest run is the one
/// least disturbed by whatever else the machine is doing.
fn median(runs: &[f64]) -> f64 {
    let mut v = runs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn column_bytes(csr: &TemporalCSR) -> usize {
    let c = csr.columns();
    c.offsets.len() * 4
        + c.dst.len() * 8
        + c.valid_from.len() * 8
        + c.valid_to.len() * 8
        + c.edge_type.len() * 2
}

/// Every column, not just the edge count: a merge that drops one row and
/// duplicates another has the right count and the wrong graph.
fn assert_same(a: &TemporalCSR, b: &TemporalCSR) {
    let (x, y) = (a.columns(), b.columns());
    assert_eq!(x.node_count, y.node_count, "node_count");
    assert_eq!(x.offsets, y.offsets, "offsets");
    assert_eq!(x.dst, y.dst, "dst");
    assert_eq!(x.valid_from, y.valid_from, "valid_from");
    assert_eq!(x.valid_to, y.valid_to, "valid_to");
    assert_eq!(x.edge_type, y.edge_type, "edge_type");
    assert_eq!(x.max_valid_from, y.max_valid_from, "max_valid_from");
    assert_eq!(x.min_valid_to, y.min_valid_to, "min_valid_to");
    assert_eq!(x.chunk_index, y.chunk_index, "chunk_index");
    assert_eq!(x.chunk_max_valid_to, y.chunk_max_valid_to, "chunk zone map");
}

fn cleanup(root: &std::path::Path) {
    // The store holds open mmaps; deleting on Windows is best-effort and a
    // leftover temp directory is not a failure worth reporting.
    let _ = std::fs::remove_dir_all(root);
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}
