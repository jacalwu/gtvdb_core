//! Micro-benchmark for [`gtv_array::crm::crm_alloc_greedy`].
//!
//! Run with:
//! ```text
//! cargo run --release -p gtv-array --example bench_crm
//! ```
//!
//! Builds deterministic multi-to-multi CRM graphs at several scales (mirroring
//! `testcase/crm_data.py` fan-out defaults) and reports **min** wall time over
//! `TRIALS` after one warmup — same Instant convention as `bench_asof` / HFT.

use std::env;
use std::time::{Duration, Instant};

use gtv_array::crm::{
    crm_alloc_greedy, AllocationMode, Collateral, CollateralEdge, GuaranteeEdge, Guarantor, Loan,
    RepartitionMode,
};

/// SplitMix64 — tiny deterministic PRNG (same class of seed as HFT benches).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Sample `k` distinct ids from `1..=n`.
    fn sample_ids(&mut self, n: u64, k: usize) -> Vec<u64> {
        use std::collections::HashSet;
        let k = k.min(n as usize);
        let mut out = Vec::with_capacity(k);
        let mut seen = HashSet::with_capacity(k * 2);
        while out.len() < k {
            let id = (self.next_u64() % n) + 1;
            if seen.insert(id) {
                out.push(id);
            }
        }
        out
    }
}

struct Graph {
    loans: Vec<Loan>,
    collaterals: Vec<Collateral>,
    guarantors: Vec<Guarantor>,
    coll_edges: Vec<CollateralEdge>,
    guar_edges: Vec<GuaranteeEdge>,
}

fn build_graph(n_loans: u64, n_cols: u64, n_guars: u64, coll_fanout: usize, guar_fanout: usize) -> Graph {
    let mut rng = Rng(0xC0FFEE_u64.wrapping_mul(n_loans).wrapping_add(17));
    let loans: Vec<Loan> = (1..=n_loans)
        .map(|id| Loan::new(id, rng.uniform(5e6, 20e6), rng.uniform(0.01, 0.05)))
        .collect();
    let collaterals: Vec<Collateral> = (1..=n_cols)
        .map(|id| {
            let value = rng.uniform(1e6, 6e6);
            let hc = rng.uniform(0.05, 0.15);
            let hfx = rng.uniform(0.0, 0.05);
            Collateral::new(id, value * (1.0 - hc - hfx).max(0.0), rng.uniform(0.0, 1.0))
        })
        .collect();
    let guarantors: Vec<Guarantor> = (1..=n_guars)
        .map(|id| Guarantor::new(id, rng.uniform(20e6, 80e6), rng.uniform(0.0, 1.0)))
        .collect();

    let mut coll_edges = Vec::with_capacity(n_cols as usize * coll_fanout);
    for c in 1..=n_cols {
        for loan in rng.sample_ids(n_loans, coll_fanout) {
            let mode = if rng.next_f64() < 0.3 {
                AllocationMode::Specified
            } else {
                AllocationMode::Optimizable
            };
            coll_edges.push(CollateralEdge {
                col_id: c,
                loan_id: loan,
                ratio: rng.uniform(0.05, 0.3),
                mode,
                        repartition: RepartitionMode::Sequential,
        });
        }
    }
    let mut guar_edges = Vec::with_capacity(n_guars as usize * guar_fanout);
    for g in 1..=n_guars {
        for loan in rng.sample_ids(n_loans, guar_fanout) {
            let mode = if rng.next_f64() < 0.3 {
                AllocationMode::Specified
            } else {
                AllocationMode::Optimizable
            };
            guar_edges.push(GuaranteeEdge {
                guarantor_id: g,
                loan_id: loan,
                amount: rng.uniform(1e6, 10e6),
                mode,
                        repartition: RepartitionMode::Sequential,
        });
        }
    }

    Graph {
        loans,
        collaterals,
        guarantors,
        coll_edges,
        guar_edges,
    }
}

fn time_min(trials: usize, mut f: impl FnMut()) -> Duration {
    f(); // warmup (excluded)
    let mut best = Duration::from_secs(u64::MAX);
    for _ in 0..trials {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed();
        if dt < best {
            best = dt;
        }
    }
    best
}

fn run_scale(label: &str, n_loans: u64, n_cols: u64, n_guars: u64, trials: usize) {
    let g = build_graph(n_loans, n_cols, n_guars, 10, 15);
    let n_ce = g.coll_edges.len();
    let n_ge = g.guar_edges.len();
    let dt = time_min(trials, || {
        let r = crm_alloc_greedy(
            &g.loans,
            &g.collaterals,
            &g.guarantors,
            &g.coll_edges,
            &g.guar_edges,
        )
        .expect("crm_alloc_greedy");
        // keep compiler from eliding the work
        std::hint::black_box(r.net_exposure.len());
        std::hint::black_box(r.allocations.len());
    });
    let r = crm_alloc_greedy(
        &g.loans,
        &g.collaterals,
        &g.guarantors,
        &g.coll_edges,
        &g.guar_edges,
    )
    .unwrap();
    let cover: f64 = r.collateral_cover.iter().sum::<f64>() + r.guarantee_cover.iter().sum::<f64>();
    let ead: f64 = r.exposure.iter().sum();
    println!(
        "{label:18} loans={n_loans:<6} coll={n_cols:<6} guar={n_guars:<5} \
         c_edges={n_ce:<7} g_edges={n_ge:<7}  min={:>8.2} ms  \
         audit_rows={}  cover/ead={:.3}",
        dt.as_secs_f64() * 1e3,
        r.allocations.len(),
        if ead > 0.0 { cover / ead } else { 0.0 },
    );
}

fn main() {
    let trials: usize = env::var("GTV_CRM_BENCH_TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    println!("gtv-array crm_alloc_greedy micro-bench (min of {trials} after warmup)");
    println!("env: GTV_CRM_BENCH_TRIALS={trials}");
    println!();

    // Small → medium → scale (defaults align with crm_data.py proportions).
    run_scale("S", 1_000, 2_000, 100, trials);
    run_scale("M", 5_000, 10_000, 500, trials);
    run_scale("L", 25_000, 50_000, 2_500, trials);

    // Optional full scale (50k/100k) — skip unless requested (slow to build edges).
    if env::var("GTV_CRM_BENCH_FULL").ok().as_deref() == Some("1") {
        run_scale("XL", 50_000, 100_000, 5_000, trials);
    } else {
        println!("(set GTV_CRM_BENCH_FULL=1 to also run 50k/100k)");
    }
}
