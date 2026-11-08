//! Quantum state reconstruction over a temporal provenance graph.
//!
//! Run with:
//!
//! ```text
//! cargo run -p gtv-core --example quantum_state_demo
//! ```
//!
//! The log below is a 3-qubit circuit with a mid-circuit measurement, i.e. a
//! state tree rather than a chain: after `measure q[0]` both branches continue
//! independently, and each later state can be traced back to the exact gate
//! prefix that produced it.

use gtv_core::quantum_state::{ProbabilitySemantics, QuantumStateLog, QuantumStateRecord};

fn rec(
    state_id: u64,
    parent_state: Option<u64>,
    gate: &str,
    timestamp: i64,
    probability: f64,
) -> QuantumStateRecord {
    QuantumStateRecord {
        state_id,
        parent_state,
        gate: gate.to_string(),
        timestamp,
        probability,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Conditional weights: the branch weight *given* the parent state.
    let log = QuantumStateLog::from_records(vec![
        rec(0, None, "|000>", 0, 1.0),
        rec(1, Some(0), "h q[0]", 1, 1.0),
        rec(2, Some(1), "cx q[0],q[1]", 2, 1.0),
        rec(3, Some(2), "cx q[0],q[2]", 3, 1.0),
        // measure q[0]: two classical branches, 50/50
        rec(4, Some(3), "measure q[0]", 4, 0.5),
        rec(5, Some(3), "measure q[0]", 4, 0.5),
        // branch |000>: rotate q[1], then measure it
        rec(6, Some(4), "rx(0.785398) q[1]", 5, 1.0),
        rec(7, Some(6), "measure q[1]", 6, 0.8),
        rec(8, Some(6), "measure q[1]", 6, 0.2),
        // branch |111>: apply x, no further measurement
        rec(9, Some(5), "x q[2]", 5, 1.0),
    ])?;

    let stats = log.graph().stats();
    println!("== log ==");
    println!(
        "semantics={:?}  states={}  roots={}  leaves={}  max_depth={}  mean_branching={:.2}",
        log.semantics(),
        stats.nodes,
        stats.roots,
        stats.leaves,
        stats.max_depth,
        stats.mean_branching
    );
    // Growth rate is the point: one row per transition, so storage follows the
    // gate count. A statevector follows 2^n and is re-materialised per snapshot.
    println!(
        "{} transitions stored; each extra qubit would double an amplitude table but add \
         at most a few rows here\n",
        stats.edges
    );

    println!("== reconstruct leaf 8 (state_id=8) ==");
    let r = log.reconstruct(8)?;
    println!("circuit     : {}", r.circuit());
    println!("depth       : {} gates", r.depth);
    println!("path        : {:?}", r.path);
    println!("timestamps  : {:?}", r.timestamps);
    println!("probability : {:.6}", r.probability);
    println!("post-measure: {}\n", r.measured());

    println!("== frontier at t=4 (two GHZ branches) ==");
    for b in log.frontier(4) {
        println!(
            "  state {:>2}  t={}  depth={}  p={:.4}  gate={}",
            b.state_id, b.timestamp, b.depth, b.probability, b.gate
        );
    }

    println!("\n== frontier at t=6 (all leaves) ==");
    for b in log.frontier(6) {
        println!(
            "  state {:>2}  p={:.4}  gate={}",
            b.state_id, b.probability, b.gate
        );
    }

    println!("\n== dominant branches at t=6, cut at 0.10 ==");
    let set = log.dominant(6, 0.10);
    println!("  explained mass before renormalisation: {:.4}", set.explained);
    for (id, p) in set.renormalized() {
        println!("  state {id:>2}  p*={p:.4}");
    }

    println!("\n== coverage (does every parent's children add up?) ==");
    for c in log.coverage() {
        println!(
            "  parent {:>2} @t={}  child_mass={:.4}  ratio={:.4}",
            c.parent, c.at, c.child_mass, c.ratio
        );
    }
    let bad = log.violations(1e-9);
    println!("  over-attributed parents: {}\n", bad.len());

    println!("== JSONL (first 2 lines of the persisted log) ==");
    for line in log.to_jsonl()?.lines().take(2) {
        println!("  {line}");
    }

    // The same file read back with absolute probabilities behaves identically:
    // 0.5 absolute for both branches of state 3 is still 0.5 conditional.
    let absolute = QuantumStateLog::from_records_absolute(vec![
        rec(0, None, "|00>", 0, 1.0),
        rec(1, Some(0), "h q[0]", 1, 1.0),
        rec(2, Some(1), "measure q[0]", 2, 0.5),
        rec(3, Some(1), "measure q[0]", 2, 0.5),
    ])?;
    println!(
        "\n== absolute semantics check == {:.4} / {:.4}",
        absolute.reconstruct(2)?.probability,
        absolute.reconstruct(3)?.probability
    );
    assert_eq!(absolute.semantics(), ProbabilitySemantics::Absolute);

    Ok(())
}
