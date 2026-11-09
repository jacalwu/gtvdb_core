//! P1-4 admission gate: does the motif SQL exit buy anything measurable?
//!
//! `ROADMAP.md` §3.3 recorded that `gtv-pattern` had no SQL surface at all,
//! even though the crate was already a dependency of `gtv-engine`. This gate
//! answers the only question the perf-first rule allows:
//!
//! > does one batched motif query beat N literal-only `motif_from()` calls on
//! > Q1 (AML wash-trade ring detection), and do the answers agree row-for-row?
//!
//! | section | question | decision rule |
//! | :-- | :-- | :-- |
//! | A (Q1) | batched vs N pointwise UDTF calls | GO only if the conservative owned path wins at N >= 32 |
//! | K | kernel parity | hard assertion: SQL == `gtv_pattern::find_from` |
//!
//! The graph is **synthetic** and built entirely in memory: groups of four
//! accounts each contain one ordered 3-cycle (transfer / guarantee / ownership
//! at increasing event times) plus sparse cross-group background edges. No
//! file is written; no CSV / .seg / .gtvd / .gtvobs enters the repository.
//!
//! Run with:
//! `cargo run --release -p gtv-engine --example bench_motif_gate`

use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{Int64Array, TimestampNanosecondArray, UInt16Array, UInt64Array};
use gtv_core::TemporalCSR;
use gtv_engine::{find_matches, GtvContext, MotifShape, MotifSpec};
use gtv_pattern::Match;

const V: usize = 20_000;
const GROUP: usize = 4;
const VALID_AT: i64 = 100;
const LIMIT: usize = 500;

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

/// One ordered ring per group + one sparse cross-group edge per node.
/// Event times 10 < 20 < 30 satisfy `Pattern::ring(3)` event ordering; the
/// background edge (t=70) is active at `VALID_AT` but never closes a ring
/// with increasing times by construction of the group-local edges.
fn motif_graph() -> TemporalCSR {
    let groups = V / GROUP;
    let mut rng = Lcg(0xA11CE_0001);
    let mut src = Vec::with_capacity(V * 2);
    let mut dst = Vec::with_capacity(V * 2);
    let mut vf = Vec::with_capacity(V * 2);
    let mut vt = Vec::with_capacity(V * 2);
    let mut et = Vec::with_capacity(V * 2);

    for g in 0..groups {
        let b = (g * GROUP) as u64;
        for (s, d, t, ty) in [
            (b, b + 1, 10, 1),
            (b + 1, b + 2, 20, 2),
            (b + 2, b, 30, 3),
            (b + 3, b + 1, 40, 0),
        ] {
            src.push(s);
            dst.push(d);
            vf.push(t);
            vt.push(1_000);
            et.push(ty);
        }
        for k in 0..GROUP {
            let s = b + k as u64;
            let to = (rng.next() % V as u64 / GROUP as u64) * GROUP as u64;
            src.push(s);
            dst.push(to);
            vf.push(70);
            vt.push(1_000);
            et.push(0);
        }
    }

    TemporalCSR::from_arrays(
        &UInt64Array::from(src),
        &UInt64Array::from(dst),
        &TimestampNanosecondArray::from(vf),
        &TimestampNanosecondArray::from(vt),
        &UInt16Array::from(et),
        V,
    )
    .expect("motif graph must build")
}

fn u64_column(b: &arrow::array::RecordBatch, i: usize) -> Vec<u64> {
    b.column(i)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .values()
        .to_vec()
}

fn i64_column(b: &arrow::array::RecordBatch, i: usize) -> Vec<i64> {
    b.column(i)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec()
}

type Row = (u64, u64, u64, i64, i64);

fn rows(batches: &[arrow::array::RecordBatch]) -> Vec<Row> {
    let mut out = Vec::new();
    for b in batches {
        let (n0, n1, n2) = (u64_column(b, 0), u64_column(b, 1), u64_column(b, 2));
        let (emin, emax) = (i64_column(b, 3), i64_column(b, 4));
        for i in 0..b.num_rows() {
            out.push((n0[i], n1[i], n2[i], emin[i], emax[i]));
        }
    }
    out.sort_unstable();
    out
}

fn kernel_rows(matches: &[Match]) -> Vec<Row> {
    let mut out: Vec<Row> = matches
        .iter()
        .map(|m| {
            let emin = m.edges.iter().map(|e| e.valid_from).min().unwrap_or(0);
            let emax = m.edges.iter().map(|e| e.valid_from).max().unwrap_or(0);
            (m.nodes[0], m.nodes[1], m.nodes[2], emin, emax)
        })
        .collect();
    out.sort_unstable();
    out
}

const COLS: &str = "n0, n1, n2, event_min, event_max";

/// One full SQL statement per seed — the only option before the batch table.
async fn pointwise(ctx: &GtvContext, seeds: &[u64]) -> (Vec<Row>, Vec<f64>) {
    let mut out = Vec::new();
    let mut latencies = Vec::with_capacity(seeds.len());
    for &s in seeds {
        let q = format!("SELECT {COLS} FROM motif_from('ring3', {s}, {VALID_AT}, {LIMIT})");
        let t = Instant::now();
        let batches = ctx.sql(&q).await.expect("pointwise motif_from failed");
        latencies.push(us(t.elapsed()));
        out.extend(rows(&batches));
    }
    out.sort_unstable();
    (out, latencies)
}

async fn batch_rows(ctx: &GtvContext) -> Vec<Row> {
    rows(
        &ctx.sql(&format!("SELECT {COLS} FROM gtv_motif_matches"))
            .await
            .expect("batch motif query failed"),
    )
}

/// Both costs are charged: registration (including an owned whole-graph clone)
/// and the SQL statement itself.
async fn batch_once_owned(ctx: &GtvContext, csr: &TemporalCSR, spec: &MotifSpec, seeds: &[u64]) -> Duration {
    let t = Instant::now();
    ctx.register_motif_batch(black_box(csr), black_box(spec), black_box(seeds))
        .expect("register motif batch");
    let r = batch_rows(black_box(ctx)).await;
    black_box(r);
    t.elapsed()
}

async fn batch_once_shared(
    ctx: &GtvContext,
    shared: &Arc<TemporalCSR>,
    spec: &MotifSpec,
    seeds: &[u64],
) -> Duration {
    let t = Instant::now();
    ctx.register_motif_batch_shared(black_box(shared), black_box(spec), black_box(seeds))
        .expect("register motif batch (shared)");
    let r = batch_rows(black_box(ctx)).await;
    black_box(r);
    t.elapsed()
}

async fn section_a(ctx: &GtvContext, csr: &TemporalCSR, spec: &MotifSpec) -> bool {
    println!("\n=== A | Q1 batched wash-trade ring detection (V={V}, ring3) ===");
    println!(
        "{:>7} {:>10} {:>10} {:>11} {:>11} {:>9} {:>9} {:>6}",
        "seeds", "pt p50", "pt p99", "owned us", "shared us", "spd own", "spd shr", "agree"
    );
    let shared = Arc::new(csr.clone());
    let groups = (V / GROUP) as u64;
    let mut go = true;

    for &n in &[1usize, 8, 32, 128] {
        let seeds: Vec<u64> = (0..n)
            .map(|i| ((i as u64 * 7919) % groups) * GROUP as u64)
            .collect();

        // Warm both paths so parser / plan caches do not charge only the first.
        let _ = pointwise(ctx, &seeds[..1]).await;
        ctx.register_motif_batch(csr, spec, &seeds).unwrap();
        let _ = batch_rows(ctx).await;

        let (point_rows, mut lat) = pointwise(ctx, &seeds).await;
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&lat, 0.50);
        let p99 = percentile(&lat, 0.99);
        let point_total: f64 = lat.iter().sum();

        let mut best_owned = Duration::MAX;
        let mut best_shared = Duration::MAX;
        for _ in 0..5 {
            best_owned = best_owned.min(batch_once_owned(ctx, csr, spec, &seeds).await);
            best_shared = best_shared.min(batch_once_shared(ctx, &shared, spec, &seeds).await);
        }
        let owned_us = us(best_owned);
        let shared_us = us(best_shared);
        let batched = batch_rows(ctx).await;

        // Hard assertions, never relaxed by timing noise.
        assert_eq!(
            point_rows, batched,
            "batch motif disagrees with N pointwise motif_from calls at seeds={n}"
        );
        assert!(!batched.is_empty(), "fixture must produce ring3 matches at seeds={n}");
        let kernel = find_matches(csr, spec, Some(&seeds)).unwrap();
        assert_eq!(batched, kernel_rows(&kernel), "SQL must equal the Rust kernel");

        let speedup = point_total / owned_us;
        let speedup_shared = point_total / shared_us;
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

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    println!("P1-4 motif admission gate — gtv-pattern finally has a SQL exit");
    println!("data: synthetic financial account graph, V={V}, ring3, valid_at={VALID_AT}");
    println!("all data is in memory; no generated files are written");

    let csr = motif_graph();
    let ctx = GtvContext::new();
    ctx.register_motif(&csr);
    let spec = MotifSpec::new(MotifShape::Ring(3), VALID_AT).with_limit(LIMIT);

    let go = section_a(&ctx, &csr, &spec).await;

    println!("\n=== gate summary ===");
    println!("A motif batch operator : {}", if go { "GO" } else { "NO-GO" });
    println!("K kernel parity        : all SQL rows equal gtv_pattern::find_from");
    println!("hard assertions        : pointwise == batch == Rust kernel");
    assert!(go, "P1-4 gate NO-GO: batch motif is not faster at N>=32");
}
