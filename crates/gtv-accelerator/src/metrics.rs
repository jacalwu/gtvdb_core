//! Per-operator records, so a benchmark can say which side answered what.
//!
//! A speedup number that does not name its backend, its input size and its
//! decision reason is not reproducible. Each call appends one [`OpRecord`]; the
//! bench example prints them as a table, and a test can assert that the CPU and
//! device sides of a parity run were both actually exercised.

use crate::api::{BackendKind, Op};
use crate::policy::BackendDecision;

/// One operator invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpRecord {
    pub op: Op,
    pub backend: BackendKind,
    pub decision: BackendDecision,
    /// Wall clock for the operator call only.
    pub micros: u64,
    /// Bytes the backend moved or touched for this call.
    pub bytes: u64,
    /// Operator-specific size (rows, seeds, ...).
    pub work: u64,
    /// Whether the result came from a resident copy.
    pub resident: bool,
}

/// Append-only log for one process or one benchmark iteration.
#[derive(Debug, Default)]
pub struct AcceleratorMetrics {
    records: Vec<OpRecord>,
}

impl AcceleratorMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, record: OpRecord) {
        self.records.push(record);
    }

    pub fn records(&self) -> &[OpRecord] {
        &self.records
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Total wall clock for one operator on one backend.
    pub fn micros(&self, op: Op, backend: BackendKind) -> u64 {
        self.records
            .iter()
            .filter(|r| r.op == op && r.backend == backend)
            .map(|r| r.micros)
            .sum()
    }

    pub fn calls(&self, op: Op, backend: BackendKind) -> usize {
        self.records
            .iter()
            .filter(|r| r.op == op && r.backend == backend)
            .count()
    }

    /// Why the CPU answered, counted by reason. This is the line that explains a
    /// disappointing speedup without re-running anything.
    pub fn fallback_reasons(&self) -> Vec<(&'static str, usize)> {
        let mut out: Vec<(&'static str, usize)> = Vec::new();
        for r in &self.records {
            let Some(reason) = r.decision.reason() else {
                continue;
            };
            let name = reason.as_str();
            match out.iter_mut().find(|(n, _)| *n == name) {
                Some((_, c)) => *c += 1,
                None => out.push((name, 1)),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::DecisionReason;

    fn rec(op: Op, backend: BackendKind, micros: u64, decision: BackendDecision) -> OpRecord {
        OpRecord {
            op,
            backend,
            decision,
            micros,
            bytes: 0,
            work: 0,
            resident: true,
        }
    }

    #[test]
    fn totals_are_per_operator_and_per_backend() {
        let mut m = AcceleratorMetrics::new();
        m.record(rec(Op::KnnFlat, BackendKind::Cuda, 10, BackendDecision::CudaResident));
        m.record(rec(Op::KnnFlat, BackendKind::Cuda, 30, BackendDecision::CudaResident));
        m.record(rec(
            Op::KnnFlat,
            BackendKind::Cpu,
            900,
            BackendDecision::Cpu {
                reason: DecisionReason::BelowThreshold {
                    work: 1,
                    threshold: 2,
                },
            },
        ));
        assert_eq!(m.micros(Op::KnnFlat, BackendKind::Cuda), 40);
        assert_eq!(m.calls(Op::KnnFlat, BackendKind::Cuda), 2);
        assert_eq!(m.micros(Op::KnnFlat, BackendKind::Cpu), 900);
        assert_eq!(m.micros(Op::BatchKhop, BackendKind::Cuda), 0);
        assert_eq!(m.fallback_reasons(), vec![("below_threshold", 1)]);
    }
}
