//! `cargo run -p gtv-engine --example accel_state [--features cuda]`
//!
//! Prints which backends this binary has and what each one would answer for each
//! operator. This is the "why is my query still on the CPU?" tool: it separates
//! `feature_disabled` (rebuild with the feature) from `device_unavailable`
//! (install a driver) from `unsupported` (that kernel has not been ported yet),
//! which is the distinction the C1 acceptance criteria ask for.

use gtv_accelerator::{DecisionInputs, ExecutionPolicy, Op};
use gtv_engine::accelerator::{availability, registry};

fn main() {
    let avail = availability();
    println!(
        "cuda: compiled={} available={} ({})",
        avail.cuda_compiled,
        avail.cuda_available,
        avail.describe()
    );

    let reg = registry();
    println!("cpu: {}", reg.cpu_caps().name);
    for d in reg.devices() {
        let caps = d.caps();
        println!(
            "device: {} resident={} budget={:?}",
            caps.name,
            caps.resident,
            caps.max_device_bytes
        );
    }
    if reg.devices().is_empty() {
        println!("device: (none registered)");
    }

    let policy = ExecutionPolicy::default();
    let caps = reg
        .devices()
        .first()
        .map(|d| d.caps())
        .unwrap_or_else(|| reg.cpu_caps());

    println!("\n{:<18} {:>12}  {:<12} decision", "operator", "work", "reason");
    for (op, work) in [
        (Op::ValidTimeFilter, 10_000_000u64),
        (Op::KnnFlat, 10_000_000),
        (Op::BatchKhop, 1_000_000),
        (Op::WindowReduce, 1_000_000),
    ] {
        let inputs = DecisionInputs::new(&caps, op, work).bytes(257 << 20);
        let (decision, backend) = reg.route(&policy, &inputs);
        println!(
            "{:<18} {:>12}  {:<12} {:?}{}",
            op.as_str(),
            work,
            decision.reason().map(|r| r.as_str()).unwrap_or("device"),
            decision,
            backend
                .map(|b| format!(" -> {}", b.name()))
                .unwrap_or_default()
        );
    }
}
