//! Where an upload's milliseconds go, column by column.
//!
//! `cargo run --release -p gtv-accelerator-cuda --example bench_residency -- 10000000 stage`
//! `cargo run --release -p gtv-accelerator-cuda --example bench_residency -- 10000000 pinned`
//!
//! # Why a stage table and not "upload takes 31 ms"
//!
//! A single number cannot be acted on. Thirty-one milliseconds could be PCIe, a
//! re-pack on the host, an allocation, or a driver bounce copy, and each of those
//! has a different fix — or none. So every stage gets a row, and the row that
//! *should* be zero is printed even when it is zero, because that is the row that
//! goes non-zero the day someone adds an AoS pass.
//!
//! `stage` (the default) reads the receipt the backend keeps:
//! [`gtv_accelerator::GraphLoadMetrics`]. `pinned` runs the A/B that decides
//! whether a page-locked staging pool is worth building: pageable `cuMemcpyHtoD`
//! against `fill + cuMemcpyHtoDAsync + sync`, end to end, because a pinned path
//! that skips the host staging copy in its own timing is measuring the wrong
//! thing.
//!
//! It is deliberately not a benchmark harness: a few numbers, printed, with the
//! machine's name next to them, because a speedup without a machine is a rumour.

use std::time::Instant;

use gtv_accelerator::{Accelerator, Column, ColumnSet, GraphLoadMetrics, SnapshotView};
use gtv_accelerator_cuda::{driver, CudaAccelerator};

fn main() {
    let edges: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    // `wide` / `narrow` / `pinned`. Two widths in one process would be dishonest
    // about "cold": the second one inherits a pool the first already paid for,
    // which is exactly the effect residency exists to exploit. The stage table
    // prints both anyway, labelled warm, because its job is the breakdown rather
    // than the cold number.
    let mode = std::env::args().nth(2).unwrap_or_else(|| "stage".into());

    let acc = match CudaAccelerator::open() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("no device: {e}");
            return;
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
        "graph: {} nodes, {} edges | wide {:.1} MiB ({:.1} B/edge, dst u64) | narrow {:.1} MiB ({:.1} B/edge, dst u32)",
        nodes,
        edges,
        wide.bytes() as f64 / (1u64 << 20) as f64,
        wide.bytes() as f64 / edges.max(1) as f64,
        narrow.bytes() as f64 / (1u64 << 20) as f64,
        narrow.bytes() as f64 / edges.max(1) as f64,
    );
    println!(
        "producer {:>6.2} ms  the one-time u32 column ({:.2} GB/s host, 12 B/edge moved)\n",
        producer,
        (edges * 12) as f64 / (producer / 1e3) / 1e9
    );

    if mode == "pinned" {
        pinned_ab(&acc, narrow);
        return;
    }

    if mode == "topology" {
        projection_table(&acc, narrow);
        return;
    }

    if mode != "narrow" {
        stage_table(&acc, wide, "wide, dst u64");
    }
    if mode != "wide" {
        stage_table(&acc, narrow, "narrow, dst u32");
    }
}

/// What a projection buys, and what it costs to keep five columns instead.
///
/// Three copies of the same graph — complete, complete-minus-labels, and
/// topology only — each uploaded three times so the printed number is the steady
/// one and not the context's first-touch. The last column of the table is the one
/// that decides a design: how many edges of each projection fit in this device's
/// budget, which is the wall an 8 GB card hits long before PCIe does.
fn projection_table(acc: &CudaAccelerator, narrow: SnapshotView<'_>) {
    let with_time = ColumnSet::ALL.without(ColumnSet::LABELS);
    let rows = [
        ("all 5", narrow),
        ("no labels", narrow.with_columns(with_time)),
        ("topology", narrow.with_columns(ColumnSet::TOPOLOGY)),
    ];

    println!("== projection ==");
    println!(
        "  {:<10} {:>10} {:>13} {:>13} {:>7} {:>7} {:>9} {:>12}",
        "columns", "cols", "bytes", "deferred", "B/edge", "ms", "GB/s", "edges @budget"
    );
    let budget = acc.budget_bytes();
    for (label, view) in rows {
        let mut steady = 0.0;
        let mut m = None;
        for _ in 0..3 {
            let t = Instant::now();
            let h = acc.load_graph(view).unwrap().expect("residency supported");
            steady = ms(t.elapsed());
            acc.release_graph(&h);
            m = acc.last_load_metrics();
        }
        let Some(m) = m else { continue };
        let per_edge = view.bytes() as f64 / view.edge_count().max(1) as f64;
        println!(
            "  {:<10} {:>10} {:>13} {:>13} {:>7.2} {:>7.2} {:>9.2} {:>12}",
            label,
            view.columns.len(),
            view.bytes(),
            view.deferred_bytes(),
            per_edge,
            steady,
            m.upload_gbps(),
            fmt_edges((budget as f64 / per_edge) as u64)
        );
    }

    // The same arithmetic stated once, because it is the argument for the whole
    // feature: the five-column copy is what overflows the card, and dropping the
    // two arrays a walk never reads is a bigger capacity win than any narrowing.
    let per_edge_all = narrow.bytes() as f64 / narrow.edge_count().max(1) as f64;
    let per_edge_topo = {
        let v = narrow.with_columns(ColumnSet::TOPOLOGY);
        v.bytes() as f64 / v.edge_count().max(1) as f64
    };
    println!(
        "\n  budget {:.2} GiB | full copy {:.2} B/edge -> {} edges | topology {:.2} B/edge -> {} edges ({:.1}x)",
        budget as f64 / (1u64 << 30) as f64,
        per_edge_all,
        fmt_edges((budget as f64 / per_edge_all) as u64),
        per_edge_topo,
        fmt_edges((budget as f64 / per_edge_topo) as u64),
        per_edge_all / per_edge_topo
    );
    let temporal_bytes =
        ColumnSet::TEMPORAL.bytes(narrow.node_count, narrow.edge_count(), narrow.dst_layout());
    let temporal_share = temporal_bytes as f64 / narrow.bytes() as f64 * 100.0;
    println!("  temporal columns alone: {temporal_share:.1}% of the full copy\n");
}

fn fmt_edges(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.0}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.0}k", n as f64 / 1e3)
    } else {
        format!("{n}")
    }
}

/// The stage table for one width, from the backend's own receipt.
///
/// Three uploads: the first is cold (the context's pool is unpopulated) and the
/// third is steady. The breakdown is printed for the steady one, because that is
/// the number a query will actually pay; the cold/steady ratio says how much of
/// the first upload was the driver setting up rather than moving data.
fn stage_table(acc: &CudaAccelerator, view: SnapshotView<'_>, label: &str) {
    let mut totals = Vec::new();
    let mut receipt: Option<GraphLoadMetrics> = None;
    for _ in 0..3 {
        let t = Instant::now();
        let h = acc.load_graph(view).unwrap().expect("residency supported");
        totals.push(ms(t.elapsed()));
        acc.release_graph(&h);
        receipt = acc.last_load_metrics();
    }
    let Some(m) = receipt else { return };

    println!("== {label} ==");
    println!(
        "  {:<12} {:>13} {:>9} {:>7} {:>9}",
        "stage", "bytes", "MiB", "ms", "share"
    );
    let total_ms = m.upload_ns().max(1) as f64 / 1e6;
    for col in Column::ALL {
        let c = m.column(col);
        println!(
            "  {:<12} {:>13} {:>9.1} {:>7.2} {:>8.1}%  {:>5.2} GB/s",
            col.name(),
            c.bytes,
            c.bytes as f64 / (1u64 << 20) as f64,
            c.h2d_ns as f64 / 1e6,
            c.h2d_ns as f64 / 1e6 / total_ms * 100.0,
            bytes_per_s(c.bytes, c.h2d_ns)
        );
    }
    // Rows that exist to be zero. Printed rather than omitted: an AoS re-pack, a
    // width conversion or a staging buffer would land here and nowhere else.
    let share = |ns: u64| ns as f64 / m.upload_ns().max(1) as f64 * 100.0;
    for (label, ns) in [
        ("host pack", m.host_pack_ns),
        ("transform", m.layout_transform_ns),
    ] {
        println!(
            "  {:<12} {:>13} {:>9} {:>7.2} {:>8.1}%",
            label,
            0,
            "—",
            ns as f64 / 1e6,
            share(ns)
        );
    }
    println!(
        "  {:<12} {:>13} {:>9} {:>7.2}",
        "staging", m.staging_bytes, "—", m.host_alloc_ns as f64 / 1e6
    );
    println!(
        "  {:<12} {:>13} {:>9} {:>7.3}",
        "device sync", "—", "—", m.device_sync_ns as f64 / 1e6
    );
    println!(
        "  {:<12} {:>13} {:>9.1} {:>7.2}",
        "upload total",
        m.uploaded_bytes,
        m.uploaded_bytes as f64 / (1u64 << 20) as f64,
        m.upload_ns() as f64 / 1e6
    );
    println!(
        "  {:<12} {:>13} {:>9} {:>7.2}",
        "device alloc", "—", "—", m.device_alloc_ns as f64 / 1e6
    );
    println!(
        "  {:<12} {:>13} {:>9} {:>7.2}   ({:.1}% of the load is PCIe)",
        "load total",
        m.priced_bytes,
        "—",
        m.total_ns as f64 / 1e6,
        m.upload_share() * 100.0
    );
    println!(
        "  wall clock: cold {:.2} ms, steady {:.2} ms | {:.2} B/edge, {:.2} GB/s | copy-free {} | accounting {}",
        totals[0],
        totals[2],
        m.uploaded_bytes_per_edge(),
        m.upload_gbps(),
        m.is_copy_free(),
        m.accounting_holds()
    );

    // The projection the next decision needs: the two time columns are the bulk
    // of the bytes, and item 9 of the review is exactly "narrow them". Priced
    // here from the measured bandwidth of the same upload rather than guessed.
    let time_bytes = m.column(Column::ValidFrom).bytes + m.column(Column::ValidTo).bytes;
    let gbs = m.upload_gbps().max(1e-9);
    println!(
        "  time columns are {:.1}% of the wire ({} MiB); as u32 deltas they would be {:.1} MiB\n  and save {:.1} ms per upload at this bandwidth — a projection, not a measurement.\n",
        time_bytes as f64 / m.uploaded_bytes.max(1) as f64 * 100.0,
        time_bytes / (1 << 20),
        time_bytes as f64 / 2.0 / (1u64 << 20) as f64,
        time_bytes as f64 / 2.0 / gbs / 1e9 * 1e3
    );

    let scale = view.edge_count() as f64 / 1e7;
    let saving = (6.86 * scale - 0.556 * scale).max(1e-9);
    println!(
        "  break-even: {:.1} cold / {:.1} steady filter invocations per snapshot (CPU saving {:.2} ms/query)\n",
        totals[0] / saving,
        totals[2] / saving,
        saving
    );
}

/// Pageable `cuMemcpyHtoD` against `fill + cuMemcpyHtoDAsync + sync`.
///
/// Both sides exclude the device allocation and the context bind, so the
/// comparison is transfer against transfer. The pinned side includes the host
/// staging copy *and* the synchronisation, because those are what the pinned
/// path costs and what a comparison of the copy calls alone would hide — the
/// trap the review names in item 5.
fn pinned_ab(acc: &CudaAccelerator, view: SnapshotView<'_>) {
    let Some(driver) = driver::driver().ok() else {
        eprintln!("no driver");
        return;
    };
    let cols: [(Column, *const u8, usize); 5] = [
        (
            Column::Offsets,
            view.offsets.as_ptr() as *const u8,
            std::mem::size_of_val(view.offsets),
        ),
        (
            Column::Dst,
            view.dst32.unwrap().as_ptr() as *const u8,
            std::mem::size_of_val(view.dst32.unwrap()),
        ),
        (
            Column::ValidFrom,
            view.valid_from.as_ptr() as *const u8,
            std::mem::size_of_val(view.valid_from),
        ),
        (
            Column::ValidTo,
            view.valid_to.as_ptr() as *const u8,
            std::mem::size_of_val(view.valid_to),
        ),
        (
            Column::EdgeType,
            view.edge_type.as_ptr() as *const u8,
            std::mem::size_of_val(view.edge_type),
        ),
    ];

    // Pageable: what `load_graph` does today.
    let mut pageable = Vec::new();
    for _ in 0..3 {
        let h = acc.load_graph(view).unwrap().unwrap();
        // Upload only: the same stages the pinned side measures, so the A/B is
        // transfer against transfer rather than transfer against allocation.
        pageable.push(acc.last_load_metrics().unwrap().upload_ns() as f64 / 1e6);
        acc.release_graph(&h);
    }

    // Pinned: one buffer per column, allocated once. Reuse is the whole premise
    // of a pool — allocating pinned memory per upload would cost more than the
    // transfer it saves, which is why the pool is measured before it is built.
    let t = Instant::now();
    let mut pinned: Vec<driver::PinnedBuffer> = cols
        .iter()
        .map(|(_, _, len)| driver.host_alloc(*len).expect("pinned alloc"))
        .collect();
    let pinned_alloc = ms(t.elapsed());
    // Device side: allocate directly, so both sides can be compared without the
    // backend's own table in the way.
    let device: Vec<u64> = cols
        .iter()
        .map(|(_, _, len)| driver.alloc(*len).expect("device alloc"))
        .collect();

    let mut staged = Vec::new();
    let mut copied = Vec::new();
    let mut synced = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        for (i, buf) in pinned.iter_mut().enumerate() {
            // SAFETY: `cols[i].2` bytes are readable in the view for as long as
            // the view lives, and the buffer was allocated for exactly that.
            unsafe { buf.fill(cols[i].1, cols[i].2).unwrap() };
        }
        staged.push(ms(t.elapsed()));

        let t = Instant::now();
        for (i, buf) in pinned.iter().enumerate() {
            // SAFETY: `device[i]` came from `alloc(cols[i].2)` above.
            unsafe { driver.copy_htod_async(device[i], buf).unwrap() };
        }
        let t2 = Instant::now();
        driver.sync().unwrap();
        copied.push(ms(t2.elapsed()));
        synced.push(ms(t.elapsed()));
    }
    for ptr in &device {
        driver.free(*ptr);
    }

    let bytes: usize = cols.iter().map(|c| c.2).sum();
    println!("== pageable vs pinned, narrow view, {:.1} MiB ==", bytes as f64 / (1u64 << 20) as f64);
    print_runs("pageable  cuMemcpyHtoD", &pageable, bytes);
    print_runs("pinned    fill", &staged, bytes);
    print_runs("pinned    async copy", &copied, bytes);
    let total: Vec<f64> = staged.iter().zip(&copied).map(|(a, b)| a + b).collect();
    print_runs("pinned    fill + copy", &total, bytes);
    println!(
        "  (pinned alloc for all five columns: {:.2} ms, one-time; sync per round: {:.3} ms)",
        pinned_alloc,
        synced[2]
    );
    let p = pageable[2];
    let n = total[2];
    println!(
        "  verdict: pinned is {:.0}% {} than pageable ({:.2} vs {:.2} ms) — {}",
        ((n / p - 1.0) * 100.0).abs(),
        if n < p { "faster" } else { "slower" },
        n,
        p,
        if n < p {
            "a staging pool would pay for itself"
        } else {
            "the staging copy costs more than the DMA buys, so stay pageable"
        }
    );
}

fn print_runs(label: &str, runs: &[f64], bytes: usize) {
    print!("  {label:<22}");
    for r in runs {
        print!(" {:>7.2} ms", r);
    }
    println!("  ({:.2} GB/s steady)", bytes as f64 / (runs[runs.len() - 1] / 1e3) / 1e9);
}

fn bytes_per_s(bytes: usize, ns: u64) -> f64 {
    if ns == 0 {
        return 0.0;
    }
    bytes as f64 / (ns as f64 / 1e9) / 1e9
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}
