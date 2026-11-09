//! M6 admission gate: make "kdb-class speed" falsifiable before touching the
//! kernel.
//!
//! `docs/perf-first-roadmap-review.md` argues that M6 should be driven by three
//! measured questions instead of the ~20 remaining ROADMAP items. This example
//! establishes that baseline. The only production change it exercises is the
//! batch graph operator (the P1 item this gate admits).
//!
//! | section | question | decision rule |
//! | :-- | :-- | :-- |
//! | A (Q1) | does one batched graph query beat N literal-only UDTF queries? | GO only if batch is faster at N >= 32 AND answers agree row-for-row |
//! | B (Q2) | how much does dense-CSR segment granularity cost? | reports the B/edge crossover that justifies M6-b node-range narrowing |
//! | C (Q3) | ingest + publish throughput | numbers only — Windows fsync dominates, compare trends |
//!
//! Workloads are shaped like the three target use cases (AML flows, CRM
//! guarantee chains, LE exposure propagation) but are **synthetic** and built
//! in memory. The small store in section C lives in a tempdir outside the
//! repository; no CSV / .seg / .gtvd / .gtvobs is ever written into the repo.
//!
//! Run with:
//! `cargo run --release -p gtv-engine --example bench_m6_gate`

use std::fs;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{TimestampNanosecondArray, UInt16Array, UInt64Array};
use gtv_core::TemporalCSR;
use gtv_engine::GtvContext;
use gtv_storage::{EdgeRecord, SegmentStore};

const V: usize = 20_000;
const DEGREE: usize = 10;
const E: usize = V * DEGREE;
const VALID_AT: i64 = 500_000;
const KHOP: usize = 2;

/// Above this many bytes per edge, dense `offsets[node_count+1]` dominates the
/// payload and segment narrowing (M6-b) deserves its own benchmark.
const B_PER_EDGE_TRIGGER: f64 = 256.0;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

/// A financial account graph: a deterministic ring (so AML cycles exist),
/// random cross-links, three edge types (transfer / guarantee / ownership)
/// and a mix of finite intervals and open-ended relationships.
fn financial_graph() -> (TemporalCSR, Vec<EdgeRecord>) {
    let mut rng = Lcg(0xC0FF_EE01);
    let mut src = Vec::with_capacity(E);
    let mut dst = Vec::with_capacity(E);
    let mut vf = Vec::with_capacity(E);
    let mut vt = Vec::with_capacity(E);
    let mut et = Vec::with_capacity(E);

    for s in 0..V as u64 {
        for d in 0..DEGREE {
            // 1 in 10 edges follows the ring; the rest are random counterparties.
            let to = if d == 0 {
                (s + 1) % V as u64
            } else {
                rng.next() % V as u64
            };
            let from = (rng.next() % 900_000) as i64;
            // 20% are enduring relationships (ownership / guarantee).
            let to_t = if d % 5 == 0 {
                i64::MAX
            } else {
                from + 50_000 + (rng.next() % 300_000) as i64
            };
            src.push(s);
            dst.push(to);
            vf.push(from);
            vt.push(to_t);
            et.push((d % 3) as u16);
        }
    }

    let csr = TemporalCSR::from_arrays(
        &UInt64Array::from(src.clone()),
        &UInt64Array::from(dst.clone()),
        &TimestampNanosecondArray::from(vf.clone()),
        &TimestampNanosecondArray::from(vt.clone()),
        &UInt16Array::from(et.clone()),
        V,
    )
    .expect("financial graph must build");

    let records = (0..E)
        .map(|i| EdgeRecord::new(src[i], dst[i], vf[i], vt[i], et[i], 1_000 + i as i64))
        .collect();
    (csr, records)
}

fn u64_column(batch: &arrow::array::RecordBatch, i: usize) -> Vec<u64> {
    batch
        .column(i)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("expected UInt64 column")
        .values()
        .to_vec()
}

/// Pointwise path: one full SQL statement per seed, exactly what a caller must
/// do today because `khop()` only accepts literals.
async fn pointwise_khop(ctx: &GtvContext, seeds: &[u64]) -> (Vec<(u64, u64, u64)>, Vec<f64>) {
    let mut rows = Vec::new();
    let mut latencies = Vec::with_capacity(seeds.len());
    for &s in seeds {
        let sql = format!("SELECT hop, dst FROM khop({s}, {KHOP}, {VALID_AT})");
        let t = Instant::now();
        let batches = ctx.sql(&sql).await.expect("pointwise khop failed");
        latencies.push(us(t.elapsed()));
        for b in &batches {
            let hop = u64_column(b, 0);
            let dst = u64_column(b, 1);
            for (h, d) in hop.into_iter().zip(dst) {
                rows.push((s, h, d));
            }
        }
    }
    rows.sort_unstable();
    (rows, latencies)
}

/// Batch path: one SQL statement over a pre-registered table.
async fn batch_khop(ctx: &GtvContext) -> Vec<(u64, u64, u64)> {
    let mut rows = Vec::new();
    let batches = ctx
        .sql("SELECT seed, hop, dst FROM gtv_batch_khop")
        .await
        .expect("batch khop failed");
    for b in &batches {
        let seed = u64_column(b, 0);
        let hop = u64_column(b, 1);
        let dst = u64_column(b, 2);
        for ((s, h), d) in seed.into_iter().zip(hop).zip(dst) {
            rows.push((s, h, d));
        }
    }
    rows.sort_unstable();
    rows
}

/// Both costs are charged: handing the seed set to the engine (registration)
/// and running the SQL statement. No nested Tokio runtime is created.
async fn batch_once_owned(ctx: &GtvContext, csr: &TemporalCSR, seeds: &[u64]) -> Duration {
    let t = Instant::now();
    ctx.register_batch_graph(black_box(csr), black_box(seeds), VALID_AT, KHOP)
        .expect("register batch graph");
    let rows = batch_khop(black_box(ctx)).await;
    black_box(rows);
    t.elapsed()
}

/// Same path with a reused `Arc<TemporalCSR>`: isolates the hidden whole-graph
/// clone that `register_batch_graph(&TemporalCSR)` pays per seed batch.
async fn batch_once_shared(
    ctx: &GtvContext,
    shared: &Arc<TemporalCSR>,
    seeds: &[u64],
) -> Duration {
    let t = Instant::now();
    ctx.register_batch_graph_shared(black_box(shared), black_box(seeds), VALID_AT, KHOP)
        .expect("register batch graph (shared)");
    let rows = batch_khop(black_box(ctx)).await;
    black_box(rows);
    t.elapsed()
}

async fn section_a(ctx: &GtvContext, csr: &TemporalCSR) -> bool {
    println!("\n=== A | Q1 batch graph operator (V={V}, E={E}, k={KHOP}) ===");
    println!(
        "{:>7} {:>10} {:>10} {:>11} {:>11} {:>9} {:>9} {:>6}",
        "seeds", "pt p50", "pt p99", "owned us", "shared us", "spd own", "spd shr", "agree"
    );

    let shared = Arc::new(csr.clone());
    let mut go = true;
    for &n in &[1usize, 8, 32, 128] {
        let seeds: Vec<u64> = (0..n).map(|i| (i * 7919) as u64 % V as u64).collect();

        // Warm both paths so parser/caches do not charge only the first run.
        let _ = pointwise_khop(ctx, &seeds[..1]).await;
        ctx.register_batch_graph(csr, &seeds, VALID_AT, KHOP)
            .expect("register batch graph");
        let _ = batch_khop(ctx).await;

        let (point_rows, mut lat) = pointwise_khop(ctx, &seeds).await;
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&lat, 0.50);
        let p99 = percentile(&lat, 0.99);
        let point_total: f64 = lat.iter().sum();

        let mut best_owned = Duration::MAX;
        let mut best_shared = Duration::MAX;
        for _ in 0..5 {
            best_owned = best_owned.min(batch_once_owned(ctx, csr, &seeds).await);
            best_shared = best_shared.min(batch_once_shared(ctx, &shared, &seeds).await);
        }
        let owned_us = us(best_owned);
        let shared_us = us(best_shared);
        let batch_rows = batch_khop(ctx).await;

        // Hard assertion: never relaxed by timing noise.
        assert_eq!(
            point_rows, batch_rows,
            "batch khop disagrees with pointwise khop at seeds={n}"
        );
        let speedup = point_total / owned_us;
        let speedup_shared = point_total / shared_us;
        // Conservative gate: even the owned (whole-graph-clone) path must win.
        if n >= 32 && speedup <= 1.0 {
            go = false;
        }
        println!(
            "{n:>7} {p50:>10.1} {p99:>10.1} {owned_us:>11.1} {shared_us:>11.1} \
             {speedup:>8.2}x {speedup_shared:>8.2}x {:>6}",
            "yes"
        );
    }
    println!(
        "A verdict: {} — {}",
        if go { "GO" } else { "NO-GO" },
        if go {
            "one planning cycle beats N pointwise UDTF calls; answers agree row-for-row"
        } else {
            "batch path is not consistently faster at N>=32"
        }
    );
    go
}

/// Section B reproduces `metadata-benchmark.md` 5.5 in memory: every segment
/// carries `node_count+1` dense offsets regardless of how few edges it holds.
/// The *same* edge set is split differently, so active counts must not change.
fn section_b(records: &[EdgeRecord], full: &TemporalCSR) {
    println!("\n=== B | Q2 segment granularity (dense CSR O(V) overhead) ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12}",
        "segs", "edges/seg", "build us", "B/edge", "edges kept"
    );

    let truth: u64 = (0..V as u64)
        .map(|s| full.neighbors(s, VALID_AT).unwrap().count() as u64)
        .sum();

    for &segs in &[1usize, 10, 100, 500] {
        let mut buckets: Vec<Vec<EdgeRecord>> = (0..segs).map(|_| Vec::new()).collect();
        for (i, e) in records.iter().enumerate() {
            buckets[i % segs].push(*e);
        }

        let t = Instant::now();
        let mut built = Vec::with_capacity(segs);
        for b in &buckets {
            let mut src = Vec::with_capacity(b.len());
            let mut dst = Vec::with_capacity(b.len());
            let mut vf = Vec::with_capacity(b.len());
            let mut vt = Vec::with_capacity(b.len());
            let mut et = Vec::with_capacity(b.len());
            for e in b {
                src.push(e.src);
                dst.push(e.dst);
                vf.push(e.valid_from);
                vt.push(e.valid_to);
                et.push(e.edge_type);
            }
            built.push(
                TemporalCSR::from_arrays(
                    &UInt64Array::from(src),
                    &UInt64Array::from(dst),
                    &TimestampNanosecondArray::from(vf),
                    &TimestampNanosecondArray::from(vt),
                    &UInt16Array::from(et),
                    V,
                )
                .unwrap(),
            );
        }
        let build_us = us(t.elapsed());
        black_box(&built);

        let kept: usize = built.iter().map(|c| c.edge_count()).sum();
        assert_eq!(kept, E, "splitting into {segs} segments lost edges");
        if segs == 1 {
            let active: u64 = (0..V as u64)
                .map(|s| built[0].neighbors(s, VALID_AT).unwrap().count() as u64)
                .sum();
            assert_eq!(active, truth, "rebuilt CSR changed the active edge set");
        }

        // offsets u32 + chunk_index u32, both node_count+1 per segment.
        let dense_bytes = segs * (V + 1) * 2 * std::mem::size_of::<u32>();
        let b_per_edge = dense_bytes as f64 / E as f64;
        println!(
            "{segs:>8} {:>12} {build_us:>12.1} {b_per_edge:>12.1} {kept:>12}",
            E / segs
        );
    }
    println!(
        "B verdict: trigger for M6-b node-range narrowing is B/edge > {B_PER_EDGE_TRIGGER:.0}; \
         the dense layout pays O(V) offsets per segment regardless of edges/seg."
    );
}

fn section_c(records: &[EdgeRecord]) {
    println!("\n=== C | Q3 ingest + publish (tempdir only; fsync is platform-bound) ===");
    let root = std::env::temp_dir().join(format!("gtv_bench_m6_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);

    let probe = &records[..20_000];
    let mut store = SegmentStore::open(&root, V).expect("open store");
    store.set_flush_threshold(usize::MAX); // measure ingest, not an auto-flush

    let t = Instant::now();
    store.ingest(1, 0, probe).expect("ingest");
    let ingest_us = us(t.elapsed());
    let edges_per_s = probe.len() as f64 / (ingest_us / 1e6);

    let t = Instant::now();
    let report = store.flush_at(2_000_000).expect("flush");
    let publish_us = us(t.elapsed());

    assert_eq!(report.edges_flushed, probe.len(), "publish lost edges");
    let visible = store.visible_edges_at(report.system_ts).unwrap().len();
    assert_eq!(visible, probe.len(), "visible edges lost across publish");

    println!(
        "ingest {:.1} us ({:.2} M edges/s) | publish {:.1} us | segments {} | visible {}",
        ingest_us,
        edges_per_s / 1e6,
        publish_us,
        report.new_segments,
        visible
    );
    println!(
        "C verdict: numbers only. M2 already measured a ~10x Windows/Linux fsync gap; \
         compare trends, never absolute milliseconds."
    );
    let _ = fs::remove_dir_all(&root);
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    println!("M6 admission gate — target: kdb-class latency on temporal graph queries");
    println!("data: synthetic financial graph, V={V}, E={E}, valid_at={VALID_AT}");
    println!("all data is in memory or a tempdir outside the repository");

    let (csr, records) = financial_graph();
    let ctx = GtvContext::new();
    ctx.register_neighbors(&csr);

    let a = section_a(&ctx, &csr).await;
    section_b(&records, &csr);
    section_c(&records);

    println!("\n=== gate summary ===");
    println!("A batch graph operator : {}", if a { "GO" } else { "NO-GO" });
    println!("B node-range narrowing : DEFER (M6-b, needs the B/edge crossover above)");
    println!("C hot-loop SIMD/narrow : DEFER (M6-b, needs perf stat on Linux)");
    println!("hard assertions        : all paths agree");
}
