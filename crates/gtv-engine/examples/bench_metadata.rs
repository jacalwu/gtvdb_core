//! L1 + L2 metadata benchmark on **real MongoDB sample data**.
//!
//! ```text
//! cargo run -p gtv-engine --release --example bench_metadata -- <data_dir>
//! ```
//!
//! `<data_dir>` holds `.gtvedge` files produced by `_benchdata/export_edges.py`
//! (mongodump BSON from `github.com/erelbi/mongodb_sample_data`, projected onto
//! an edge list). Nothing from that directory is stored in this repository.
//!
//! # What is measured, and why it is the right thing to measure
//!
//! | question | metric |
//! | :-- | :-- |
//! | does L2 make cold start worse? | `SegmentStore::open` over S segments |
//! | is a prune decision free? | `entries_overlapping` vs opening every segment |
//! | how much work does pruning remove? | skip ratio + selected/actual edges |
//! | is pruning *correct*? | pruned active-edge count == full-scan count, asserted |
//! | does L1 answer catalog questions? | `gtv_tables()` / `gtv_snapshots()` / `gtv_files()` latency |
//!
//! The correctness assertion is not decoration: a benchmark that only measures
//! speed would happily reward a pruner that drops real rows.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use arrow::array::Int64Array;
use gtv_engine::GtvContext;
use gtv_storage::{EdgeRecord, SegmentStore};

const DAY_NS: i64 = 86_400 * 1_000_000_000;
/// Bucket width used to decide flush boundaries (30 days).
const BUCKET_NS: i64 = 30 * DAY_NS;

// -- edge file ---------------------------------------------------------------

struct EdgeFile {
    edges: Vec<(u64, u64, i64, i64, u16)>,
    node_count: usize,
}

fn read_edges(path: &Path) -> EdgeFile {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(&bytes[0..8], b"GTVEDGE1", "bad magic in {}", path.display());
    let edge_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let node_count = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    let mut edges = Vec::with_capacity(edge_count);
    let mut off = 24;
    for _ in 0..edge_count {
        let src = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        let dst = u64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap());
        let vf = i64::from_le_bytes(bytes[off + 16..off + 24].try_into().unwrap());
        let vt = i64::from_le_bytes(bytes[off + 24..off + 32].try_into().unwrap());
        let et = u16::from_le_bytes(bytes[off + 32..off + 34].try_into().unwrap());
        edges.push((src, dst, vf, vt, et));
        off += 34;
    }
    EdgeFile { edges, node_count }
}

// -- store -------------------------------------------------------------------

/// Ingest every edge, flushing one segment per bucket.
///
/// `bucket_of` maps an edge to its flush group. Interactive (30-day) bucketing
/// mirrors how a real streaming loader would flush: one immutable segment per
/// time window, which is exactly the shape L2 pruning is designed for.
fn build_store(
    root: &Path,
    file: &EdgeFile,
    bucket_of: impl Fn(&(u64, u64, i64, i64, u16)) -> i64,
) -> (SegmentStore, f64, f64) {
    let _ = std::fs::remove_dir_all(root);
    let mut store = SegmentStore::open(root, file.node_count.max(1)).unwrap();

    // Group without sorting the whole edge list: bucket -> indices.
    let mut buckets: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, e) in file.edges.iter().enumerate() {
        buckets.entry(bucket_of(e)).or_default().push(i);
    }

    let t_ingest = Instant::now();
    let mut ingest_ms = 0.0;
    let mut flush_ms = 0.0;
    for (b, idxs) in &buckets {
        let sys = *b + 1;
        let batch: Vec<EdgeRecord> = idxs
            .iter()
            .map(|&i| {
                let (s, d, vf, vt, et) = file.edges[i];
                EdgeRecord::new(s, d, vf, vt, et, sys)
            })
            .collect();
        let t = Instant::now();
        store.ingest(1, 0, &batch).unwrap();
        ingest_ms += t.elapsed().as_secs_f64() * 1000.0;
        // Flush per bucket: `flush_at` publishes everything currently in the
        // delta, so ingest and flush must be interleaved (not batched at the
        // end) for "one segment per window" to actually hold.
        let t = Instant::now();
        store.flush_at(sys).unwrap();
        flush_ms += t.elapsed().as_secs_f64() * 1000.0;
    }
    let _ = t_ingest;
    (store, ingest_ms, flush_ms)
}

/// Open every segment — what a reader without manifest metadata must pay to
/// learn the same time bounds.
fn open_all_ms(store: &SegmentStore) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        for e in store.segments() {
            let seg = store.open_entry(e).unwrap();
            std::hint::black_box(seg.header().edge_count);
        }
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    best
}

/// Ground truth: number of edges whose `[vf, vt)` intersects `[t1, t2)`.
/// When `only` is `Some`, restrict to those segment ids — that is how we prove
/// the pruned set is sufficient.
fn active_edges(store: &SegmentStore, t1: i64, t2: i64, only: Option<&[u64]>) -> u64 {
    let mut n = 0u64;
    for e in store.segments() {
        if let Some(ids) = only {
            if !ids.contains(&e.id) {
                continue;
            }
        }
        let seg = store.open_entry(e).unwrap();
        let c = seg.csr().columns();
        for i in 0..c.dst.len() {
            if c.valid_from[i] < t2 && c.valid_to[i] > t1 {
                n += 1;
            }
        }
    }
    n
}

fn names(store: &SegmentStore, t1: i64, t2: i64) -> (Vec<u64>, usize) {
    let selected = store.entries_overlapping(t1, t2);
    let ids: Vec<u64> = selected.iter().map(|e| e.id).collect();
    let bound: usize = selected.iter().map(|e| e.edge_count as usize).sum();
    (ids, bound)
}

struct Report {
    dataset: String,
    edges: usize,
    nodes: usize,
    segments: usize,
    file_mb: f64,
    manifest_kb: f64,
    ingest_ms: f64,
    flush_ms: f64,
    open_ms: f64,
    open_all_ms: f64,
    prune_us: f64,
}

fn bench_dataset(name: &str, path: &Path, bucket: impl Fn(&(u64, u64, i64, i64, u16)) -> i64) -> Option<Report> {
    if !path.exists() {
        println!("  (skipped: {} missing)", path.display());
        return None;
    }
    let file = read_edges(path);
    let root = std::env::temp_dir().join(format!(
        "gtv_benchmeta_{}_{}",
        std::process::id(),
        name
    ));
    let (store, ingest_ms, flush_ms) = build_store(&root, &file, bucket);

    let segs = store.segments().len();
    let file_mb: f64 = store
        .segments()
        .iter()
        .map(|e| store.segment_path(e).metadata().map(|m| m.len() as f64).unwrap_or(0.0))
        .sum::<f64>()
        / 1_048_576.0;
    let manifest_kb = store
        .manifest_path()
        .metadata()
        .map(|m| m.len() as f64 / 1024.0)
        .unwrap_or(0.0);

    // Cold open with S segments in the manifest (no .seg file is touched).
    let mut open_ms = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        let s = SegmentStore::open(&root, file.node_count.max(1)).unwrap();
        open_ms = open_ms.min(t.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(s.segments().len(), segs);
    }

    let open_all = open_all_ms(&store);

    // Data axis for window generation.
    //
    // `business_span()` is the span of *possible* activity, so an open-ended
    // edge makes it `[lo, i64::MAX)`. That is the right pruning bound but a
    // useless window generator — every window would land past the data. For
    // queries we want the span between the earliest start and the latest start.
    let lo = store
        .segments()
        .iter()
        .filter_map(|e| e.valid_from_min)
        .min()
        .unwrap_or(0);
    let hi_start = store
        .segments()
        .iter()
        .filter_map(|e| e.max_valid_from)
        .max()
        .unwrap_or(lo);
    let span = (hi_start - lo).max(BUCKET_NS);
    let _ = store.business_span();

    // Window sweep: one bucket, a quarter, a year, everything.
    let windows = [
        ("1x30d", lo + span / 4, lo + span / 4 + BUCKET_NS),
        ("3x30d", lo + span / 2, lo + span / 2 + 3 * BUCKET_NS),
        ("12x30d", lo + span / 10, lo + span / 10 + 12 * BUCKET_NS),
        ("full", i64::MIN, i64::MAX),
    ];

    println!("\n=== {name} ===");
    println!(
        "  edges={}  nodes={}  segments={}  data={file_mb:.2} MB ({:.0} B/edge)  manifest={manifest_kb:.1} KB",
        file.edges.len(),
        file.node_count,
        segs,
        file_mb * 1_048_576.0 / file.edges.len().max(1) as f64
    );
    println!("  ingest (WAL+mem)      : {ingest_ms:9.1} ms");
    println!("  flush (encode+publish): {flush_ms:9.1} ms");
    println!("  cold open (manifest)  : {open_ms:9.3} ms   <-- L2 asks nothing of open");
    println!("  open every .seg       : {open_all:9.3} ms   <-- the pre-L2 alternative");

    let mut prune_us = 0.0;
    for (label, t1, t2) in windows {
        let s = store.prune_summary(t1, t2);
        let (ids, bound) = names(&store, t1, t2);

        // Cost of the decision itself (min of 20) — no file is opened here.
        let mut best = f64::INFINITY;
        for _ in 0..20 {
            let t = Instant::now();
            let v = store.entries_overlapping(t1, t2);
            best = best.min(t.elapsed().as_secs_f64() * 1_000_000.0);
            std::hint::black_box(v.len());
        }

        // Correctness: the pruned set must reproduce the full-scan answer.
        let full = active_edges(&store, t1, t2, None);
        let pruned = active_edges(&store, t1, t2, Some(&ids));
        assert_eq!(
            pruned, full,
            "{label}: pruning lost rows ({pruned} of {full})"
        );

        // Full-scan cost for comparison: every segment, every edge.
        let t = Instant::now();
        let full_again = active_edges(&store, t1, t2, None);
        let full_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let pruned_again = active_edges(&store, t1, t2, Some(&ids));
        let pruned_ms = t.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(full_again, pruned_again);

        if label == "3x30d" {
            prune_us = best;
        }
        println!(
            "  [{label:6}] selected {:4}/{:4} segs (skip {:5.1}%  full_hit {:3}  scan {:4})  \
             edge-bound {:8}  active {:8}  decide {:7.2} us  scan {:7.2} ms (all) -> {:7.2} ms",
            s.selected,
            s.selected + s.skipped,
            s.skip_ratio() * 100.0,
            s.full_hit,
            s.scan,
            bound,
            full,
            best,
            full_ms,
            pruned_ms
        );
    }

    let report = Report {
        dataset: name.to_string(),
        edges: file.edges.len(),
        nodes: file.node_count,
        segments: segs,
        file_mb,
        manifest_kb,
        ingest_ms,
        flush_ms,
        open_ms,
        open_all_ms: open_all,
        prune_us,
    };
    let _ = std::fs::remove_dir_all(&root);
    Some(report)
}

// -- L1 ----------------------------------------------------------------------

fn bench_catalog(tables: usize, snapshots: usize) -> f64 {
    use arrow::array::{RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use gtv_catalog::{CommitOp, CommitOptions, FsCatalog, NewFile, PartitionSpec};
    use std::sync::Arc;

    let root = std::env::temp_dir().join(format!("gtv_benchcat_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let cat = FsCatalog::open(&root).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, false),
        Field::new("px", DataType::Int64, false),
    ]));

    for t in 0..tables {
        let id = cat
            .create_table(&format!("t{t:04}"), schema.clone(), PartitionSpec::single())
            .unwrap();
        for s in 0..snapshots {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec!["AAA"; 10])),
                    Arc::new(Int64Array::from(vec![s as i64; 10])),
                ],
            )
            .unwrap();
            cat.commit(
                id,
                CommitOp::Append,
                vec![NewFile::unpartitioned(batch)],
                &CommitOptions::default(),
            )
            .unwrap();
        }
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let ctx = GtvContext::new();
    ctx.attach_catalog(&root).unwrap();

    let run = |sql: &'static str| -> (usize, i64) {
        rt.block_on(async {
            let out = ctx.sql(sql).await.unwrap();
            let rows = out.iter().map(|b| b.num_rows()).sum();
            let first = out
                .first()
                .and_then(|b| b.column(0).as_any().downcast_ref::<Int64Array>())
                .filter(|a| !a.is_empty())
                .map(|a| a.value(0))
                .unwrap_or(-1);
            (rows, first)
        })
    };

    println!("\n=== L1 catalog ({tables} tables x {snapshots} snapshots) ===");
    let t = Instant::now();
    let listed = cat.list_tables().unwrap().len();
    println!(
        "  FsCatalog::list_tables : {:8.2} ms  ({listed} tables)",
        t.elapsed().as_secs_f64() * 1000.0
    );

    // Fixed cost of the measurement itself: the same query against an empty
    // catalog. Everything above this line is metadata work, not SQL overhead.
    let empty_root = std::env::temp_dir().join(format!("gtv_benchcat_empty_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&empty_root);
    let _ = FsCatalog::open(&empty_root).unwrap();
    let ectx = GtvContext::new();
    ectx.attach_catalog(&empty_root).unwrap();
    let mut fixed_ms = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        let _ = rt.block_on(async { ectx.sql("SELECT count(*) FROM gtv_tables()").await.unwrap() });
        fixed_ms = fixed_ms.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    println!("  empty catalog baseline : {fixed_ms:8.2} ms  (UDTF + planner overhead)");
    let _ = std::fs::remove_dir_all(&empty_root);

    for (label, sql) in [
        ("gtv_tables()", "SELECT count(*) FROM gtv_tables()"),
        ("gtv_snapshots()", "SELECT count(*) FROM gtv_snapshots()"),
        (
            "gtv_files('t0000')",
            "SELECT count(*) FROM gtv_files('t0000')",
        ),
    ] {
        let mut best = f64::INFINITY;
        let mut value = -1;
        for _ in 0..5 {
            let t = Instant::now();
            let (_rows, v) = run(sql);
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
            value = v;
        }
        println!("  {label:22} : {best:8.2} ms  (count={value})");
    }

    let t = Instant::now();
    let n = ctx.sync_catalog().unwrap();
    let sync_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("  sync_catalog()         : {sync_ms:8.2} ms  ({n} tables published)");
    let (rows, _) = run("SELECT count(*) FROM t0000");
    println!("  SELECT count(*) FROM t0000 : {rows} rows via snapshot files");

    let _ = std::fs::remove_dir_all(&root);
    sync_ms
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "_benchdata".to_string());
    let dir = PathBuf::from(dir);
    println!("gtvdb L1/L2 metadata benchmark\n  data dir: {}", dir.display());

    let mut reports = Vec::new();

    // 1M edges, 100 sequential 30-day windows. The main L2 workload: one
    // segment per window, so a single-window query should touch 1 of 100.
    if let Some(r) = bench_dataset(
        "trips_x100 (intervals, 1M edges)",
        &dir.join("trips_x100.gtvedge"),
        |e| e.2.div_euclid(BUCKET_NS),
    ) {
        reports.push(r);
    }

    // Real interactions, point events: `[t, t+1)` — the case where a time box
    // is maximally selective.
    if let Some(r) = bench_dataset(
        "comments_point (50k point events)",
        &dir.join("comments_point.gtvedge"),
        |e| e.2.div_euclid(BUCKET_NS),
    ) {
        reports.push(r);
    }

    // Same data, yearly buckets. A GTVSeg is a *dense* CSR: every segment
    // carries `node_count` offsets no matter how few edges it holds, so
    // segment granularity has to be chosen against node cardinality, not only
    // against edge volume. This pair makes that cost visible.
    if let Some(r) = bench_dataset(
        "comments_point (yearly buckets)",
        &dir.join("comments_point.gtvedge"),
        |e| e.2.div_euclid(365 * DAY_NS),
    ) {
        reports.push(r);
    }

    // Same data as a cumulative as-of graph: `[t, MAX)`. Pruning can only cut
    // the tail, which is the honest counter-example.
    if let Some(r) = bench_dataset(
        "comments_cumulative (50k as-of edges)",
        &dir.join("comments_cumulative.gtvedge"),
        |e| e.2.div_euclid(BUCKET_NS),
    ) {
        reports.push(r);
    }

    // Static graph control: every segment overlaps every window, so L2 must
    // prune nothing and must not claim otherwise.
    if let Some(r) = bench_dataset(
        "routes (static control, 67k edges)",
        &dir.join("routes.gtvedge"),
        |e| (e.0 % 12) as i64,
    ) {
        reports.push(r);
    }

    let sync_ms = bench_catalog(50, 20);

    println!("\n=== summary ===");
    println!(
        "{:<38} {:>9} {:>8} {:>9} {:>11} {:>10}",
        "dataset", "edges", "segs", "open ms", "open-all ms", "decide us"
    );
    for r in &reports {
        println!(
            "{:<38} {:>9} {:>8} {:>9.3} {:>11.2} {:>10.2}",
            r.dataset, r.edges, r.segments, r.open_ms, r.open_all_ms, r.prune_us
        );
    }
    println!("{:<38} {:>9} {:>8} {:>9} {:>11} {:>10}", "catalog sync", "-", "-", "-", "-", format!("{sync_ms:.1} ms"));

    // Machine-readable copy for the write-up.
    let json: Vec<String> = reports
        .iter()
        .map(|r| {
            format!(
                "{{\"dataset\":\"{}\",\"edges\":{},\"nodes\":{},\"segments\":{},\"data_mb\":{:.2},\
                 \"manifest_kb\":{:.1},\"ingest_ms\":{:.1},\"flush_ms\":{:.1},\"open_ms\":{:.3},\
                 \"open_all_ms\":{:.2},\"prune_us\":{:.2}}}",
                r.dataset, r.edges, r.nodes, r.segments, r.file_mb, r.manifest_kb,
                r.ingest_ms, r.flush_ms, r.open_ms, r.open_all_ms, r.prune_us
            )
        })
        .collect();
    println!("\nJSON {}", json.join("\nJSON "));
}
