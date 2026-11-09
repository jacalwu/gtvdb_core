//! Benchmark: cost of ACTIVE GRAPH endpoint intersection vs the original CSR
//! neighbour iterator. This is the P2/P4 performance contract in
//! `docs/minimal-temporal-update-impl.md` §7.
//!
//! Run with: `cargo run --release -p gtv-core --example bench_endpoint`

use std::time::Instant;

use arrow::array::{TimestampNanosecondArray, UInt16Array, UInt64Array};
use gtv_core::{
    EndpointPolicy, EndpointView, NodeLifetime, TemporalCSR, OPEN_FROM, OPEN_TO,
};

const NODES: usize = 20_000;
const DEGREE: usize = 20;
const CUTOFF: i64 = 1_500;

fn build_csr() -> TemporalCSR {
    let edges = NODES * DEGREE;
    let mut src = Vec::with_capacity(edges);
    let mut dst = Vec::with_capacity(edges);
    let mut et = Vec::with_capacity(edges);
    let mut vf = Vec::with_capacity(edges);
    let mut vt = Vec::with_capacity(edges);
    for s in 0..NODES as u64 {
        for d in 0..DEGREE {
            let e = (s as usize * DEGREE + d) as i64;
            src.push(s);
            dst.push((s + (d as u64 * 37 + 11) % NODES as u64) % NODES as u64);
            // Half the edges are ClampToEndpoints, half are historical.
            et.push(if d % 2 == 0 { 1u16 } else { 2u16 });
            let from = e % 900;
            vf.push(from);
            vt.push(from + 500 + e % 700);
        }
    }
    TemporalCSR::from_arrays(
        &UInt64Array::from(src),
        &UInt64Array::from(dst),
        &TimestampNanosecondArray::from(vf),
        &TimestampNanosecondArray::from(vt),
        &UInt16Array::from(et),
        NODES,
    )
    .unwrap()
}

fn expiring_lifetime() -> NodeLifetime {
    NodeLifetime::new(
        vec![OPEN_FROM; NODES],
        (0..NODES)
            .map(|i| if i % 20 == 0 { CUTOFF } else { OPEN_TO })
            .collect(),
    )
    .unwrap()
}

fn sweep<F: Fn(u64) -> usize>(f: F) -> usize {
    let mut total = 0usize;
    for s in 0..NODES as u64 {
        total = total.wrapping_add(f(s));
    }
    total
}

/// Min-of-rounds: a single sweep is only ~1 ms on this workload, so wall-clock
/// jitter easily dominates a 10% difference. Report the fastest of 3 rounds.
fn measure<F: Fn(u64) -> usize>(name: &str, iters: usize, f: F) -> (f64, usize) {
    let warm = sweep(&f);
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        let mut acc = 0usize;
        for _ in 0..iters {
            acc = acc.wrapping_add(std::hint::black_box(sweep(&f)));
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        best = best.min(us);
        std::hint::black_box(acc);
    }
    let ns_per_edge = best * 1e3 / (NODES * DEGREE) as f64;
    println!("{name:42} {best:10.1} us/sweep  {ns_per_edge:7.2} ns/edge  edges={warm}");
    (best, warm)
}

fn main() {
    let csr = build_csr();
    let forever = NodeLifetime::forever(NODES);
    let expiring = expiring_lifetime();
    let none = EndpointPolicy::none();
    let clamp = EndpointPolicy::clamp(&[1]).unwrap();

    let hist = EndpointView::new(&csr, &forever, none).unwrap();
    let no_policy = EndpointView::new(&csr, &forever, none).unwrap();
    let clamp_view = EndpointView::new(&csr, &expiring, clamp).unwrap();

    println!(
        "nodes={NODES} degree={DEGREE} edges={} | clamped edges={}",
        csr.edge_count(),
        clamp_view.clamp_edges()
    );

    for &(label, t) in &[
        ("t=1000 (no node expired yet)", 1000i64),
        ("t=1600 (5% of nodes expired at 1500)", 1600i64),
    ] {
        let box_hit = clamp_view.all_endpoints_active_at(t);
        println!("\n=== {label} | segment box hit = {box_hit} ===");
        let (base_us, n0) = measure("historical: original neighbors()", 100, |s| {
            hist.historical_neighbors(s, t).unwrap().count()
        });
        let (_, n1) = measure("active: no policy", 100, |s| {
            no_policy.active_neighbors(s, t).unwrap().count()
        });
        let (active_us, n2) = measure("active: clamp policy (auto fast path)", 100, |s| {
            clamp_view.active_neighbors(s, t).unwrap().count()
        });
        // Naive per-edge intersection: what §8.3's edge_visible() would cost
        // without the policy bitmap / slice-level src check / segment box.
        let (reference_us, n3) = measure("reference: explicit per-edge filtering", 100, |s| {
            let mut c = 0usize;
            for n in hist.historical_neighbors(s, t).unwrap() {
                if clamp.is_clamped(n.edge_type)
                    && (!expiring.active_at(s, t) || !expiring.active_at(n.dst, t))
                {
                    continue;
                }
                c += 1;
            }
            c
        });

        // Faithful simulation of the row-level bitemporal model in §3/§8.3:
        // `version_id: u128` must be resolved through an index before the four
        // inequalities can be evaluated, for *both* endpoints of every edge.
        // A HashMap<u64, (valid_from, valid_to, system_from, system_to)> is
        // the cheapest possible stand-in for that version index.
        let system_t = std::hint::black_box(10_000i64);
        let versions: std::collections::HashMap<u64, (i64, i64, i64, i64)> = (0..NODES as u64)
            .map(|i| (i, (OPEN_FROM, expiring.valid_to(i), 0, OPEN_TO)))
            .collect();
        let (row_level_us, n4) = measure("document design: row-level bitemporal", 100, |s| {
            let mut c = 0usize;
            for n in hist.historical_neighbors(s, t).unwrap() {
                if clamp.is_clamped(n.edge_type) {
                    let (sf, st, ssf, sst) = versions[&s];
                    let (df, dt, dsf, dst) = versions[&n.dst];
                    let src_ok = sf <= t && t < st && ssf <= system_t && system_t < sst;
                    let dst_ok = df <= t && t < dt && dsf <= system_t && system_t < dst;
                    if !(src_ok && dst_ok) {
                        continue;
                    }
                }
                c += 1;
            }
            c
        });

        // Order control: re-run the very first case at the end. If this drifts
        // by the same amount as the cheap cases, the differences above are
        // measurement/warm-up artefacts, not real per-edge cost.
        let (recheck_us, _) = measure("historical: recheck (order control)", 100, |s| {
            hist.historical_neighbors(s, t).unwrap().count()
        });

        assert_eq!(n0, n1, "empty policy changed the result");
        assert_eq!(n2, n3, "clamp policy changed the result");
        assert_eq!(n3, n4, "row-level reference disagrees");
        println!(
            "  all paths agree ({n3} edges) | active/baseline = {:.2}x | plain filtering = {:.2}x | row-level doc design = {:.2}x | baseline drift = {:.2}x",
            active_us / base_us,
            reference_us / base_us,
            row_level_us / base_us,
            recheck_us / base_us,
        );
    }
}
