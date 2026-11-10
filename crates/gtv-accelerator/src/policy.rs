//! Where the choice "CPU or device" is made — in exactly one function.
//!
//! Scattering `if use_gpu { … }` across call sites is how a system ends up with
//! one query path that silently loses precision and another that silently uses
//! the wrong snapshot. [`ExecutionPolicy::decide`] is a pure function: no
//! backend is called, nothing is loaded, and every branch names its reason, so
//! the same inputs always produce the same scheduling and a benchmark can
//! explain its own configuration.
//!
//! The policy is conservative by construction. Every uncertain case resolves to
//! the CPU, because the CPU is the reference and a wrong fast answer is worse
//! than a right slow one.

use crate::api::{BackendKind, Caps, Op};
use crate::error::AcceleratorError;

/// Operator-specific admission thresholds.
///
/// The defaults are the C0 measurements, not guesses: the valid-time filter
/// broke even at 2.1 invocations and the dot-product kernel was bus-bound, so
/// small work is not worth a transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPolicy {
    /// Hard off switch. Also the switch a test flips to prove parity.
    pub prefer_device: bool,
    /// `rows * queries` below which a flat scan stays on the CPU.
    pub min_knn_work: u64,
    /// Seed count below which a batch expansion stays on the CPU.
    pub min_khop_seeds: u64,
    /// Row count below which a rolling reduce stays on the CPU.
    pub min_window_rows: u64,
    /// Hard ceiling; `None` means "whatever the backend reports".
    pub max_device_bytes: Option<usize>,
    /// Allow an upload on a miss. When false only an already-resident copy is
    /// used, which is what a latency-sensitive path wants.
    pub allow_upload: bool,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            prefer_device: true,
            // 1e5 rows x 1 query: below this the upload dominates (C0 §3.4).
            min_knn_work: 100_000,
            // N=32..128 seeds measured at 268–316 µs on the CPU; a kernel plus
            // transfer cannot beat that (cuda-acceleration-plan §6.2).
            min_khop_seeds: 10_000,
            // Off by default, and the threshold is what switches it off.
            //
            // C3's kernel is 15x the CPU's arithmetic (1.35 ms against 19-20 ms
            // at 1e7 rows), but this operator's result is the same size as its
            // input: a stateless call moves 8n up and 8n back, and even the
            // resident form measures 1.05x end to end — the read back plus the
            // materialisation of a fresh 80 MB result costs as much as the whole
            // CPU reference (§9.8). So it is not a default, it is a building
            // block: a chained plan that never ships the column back gets the
            // 15x, and that plan has to opt in explicitly. This is exactly the
            // condition spec Phase E sets — "no end-to-end win on its own means
            // internal operator only, never enabled by default".
            min_window_rows: u64::MAX,
            max_device_bytes: None,
            allow_upload: true,
        }
    }
}

impl ExecutionPolicy {
    /// Never use a device. The parity harness runs both sides this way.
    pub fn cpu_only() -> Self {
        Self {
            prefer_device: false,
            ..Self::default()
        }
    }

    pub fn with_max_device_bytes(mut self, bytes: Option<usize>) -> Self {
        self.max_device_bytes = bytes;
        self
    }

    pub fn with_upload(mut self, allow: bool) -> Self {
        self.allow_upload = allow;
        self
    }

    /// Threshold for one operator, in that operator's own unit.
    pub fn threshold(&self, op: Op) -> u64 {
        match op {
            Op::KnnFlat => self.min_knn_work,
            Op::BatchKhop => self.min_khop_seeds,
            Op::WindowReduce => self.min_window_rows,
            // A filter is only ever run to feed the next device operator; on
            // its own it is never worth a transfer (C0 §3.2: 96% transfer).
            Op::ValidTimeFilter => u64::MAX,
        }
    }

    /// Decide where `op` runs.
    pub fn decide(&self, inputs: &DecisionInputs<'_>) -> BackendDecision {
        let cpu = |reason| BackendDecision::Cpu { reason };

        if !inputs.caps.resident || inputs.caps.backend == BackendKind::Cpu {
            return cpu(DecisionReason::NoDevice);
        }
        if !self.prefer_device {
            return cpu(DecisionReason::ForcedCpu);
        }
        if let Some(err) = inputs.last_error {
            // A backend that just failed is not retried on the next call: one
            // bad pointer or one OOM is enough to poison a query batch.
            return cpu(DecisionReason::AfterError(err.fallback_reason()));
        }
        if !inputs.caps.supports(inputs.op) {
            return cpu(DecisionReason::Unsupported);
        }
        if let Some(k) = inputs.k {
            if !inputs.caps.admits_k(k) {
                return cpu(DecisionReason::KTooLarge {
                    k,
                    max_k: inputs.caps.max_k,
                });
            }
        }
        let limit = self
            .max_device_bytes
            .or(inputs.caps.max_device_bytes)
            .unwrap_or(usize::MAX);
        if inputs.required_bytes > limit as u64 {
            return cpu(DecisionReason::OverBudget {
                needed: inputs.required_bytes,
                limit: limit as u64,
            });
        }
        if inputs.resident {
            return BackendDecision::CudaResident;
        }
        if !self.allow_upload {
            return cpu(DecisionReason::UploadNotAllowed);
        }
        // Thresholds are per operator and only apply when paying for an upload.
        // A resident copy is already paid for, so even small work is free.
        let threshold = self.threshold(inputs.op);
        if inputs.work < threshold {
            return cpu(DecisionReason::BelowThreshold {
                work: inputs.work,
                threshold,
            });
        }
        BackendDecision::CudaLoadThenRun
    }
}

/// Everything the decision depends on, gathered by the caller.
#[derive(Debug, Clone)]
pub struct DecisionInputs<'a> {
    pub caps: &'a Caps,
    pub op: Op,
    /// Operator-specific size: `rows * queries`, seed count, row count.
    pub work: u64,
    /// Whether a handle for the current snapshot is already resident.
    pub resident: bool,
    /// Worst-case bytes the operator would upload.
    pub required_bytes: u64,
    /// The backend's last failure in this query, if any.
    pub last_error: Option<&'a AcceleratorError>,
    /// `k` for a KNN; `None` when the operator has no such parameter.
    pub k: Option<usize>,
}

impl<'a> DecisionInputs<'a> {
    pub fn new(caps: &'a Caps, op: Op, work: u64) -> Self {
        Self {
            caps,
            op,
            work,
            resident: false,
            required_bytes: 0,
            last_error: None,
            k: None,
        }
    }

    /// The KNN parameter, so the policy can refuse a `k` the kernel cannot hold.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    pub fn resident(mut self, resident: bool) -> Self {
        self.resident = resident;
        self
    }

    pub fn bytes(mut self, required_bytes: u64) -> Self {
        self.required_bytes = required_bytes;
        self
    }

    pub fn after(mut self, err: Option<&'a AcceleratorError>) -> Self {
        self.last_error = err;
        self
    }
}

/// Where an operator will run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendDecision {
    Cpu { reason: DecisionReason },
    /// A handle is resident: run with no upload.
    CudaResident,
    /// Upload the snapshot, then run.
    CudaLoadThenRun,
}

impl BackendDecision {
    pub fn is_device(&self) -> bool {
        matches!(
            self,
            BackendDecision::CudaResident | BackendDecision::CudaLoadThenRun
        )
    }

    pub fn reason(&self) -> Option<DecisionReason> {
        match self {
            BackendDecision::Cpu { reason } => Some(*reason),
            _ => None,
        }
    }
}

/// Why the CPU was chosen. Logged with the query, so a benchmark can say
/// "4 of 5 operators ran on the device, the fifth was unsupported".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReason {
    /// No accelerator registered, or the registered one is the CPU itself.
    NoDevice,
    /// `prefer_device == false`.
    ForcedCpu,
    /// Backend did not implement the operator.
    Unsupported,
    /// Work is below the operator's admission threshold.
    BelowThreshold { work: u64, threshold: u64 },
    /// Would not fit the device budget even after eviction.
    OverBudget { needed: u64, limit: u64 },
    /// Not resident, and the policy forbids an upload on a miss.
    UploadNotAllowed,
    /// `k` is larger than the backend's kernel can hold.
    KTooLarge { k: usize, max_k: usize },
    /// The backend failed earlier in this query.
    AfterError(&'static str),
}

impl DecisionReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            DecisionReason::NoDevice => "no_device",
            DecisionReason::ForcedCpu => "forced_cpu",
            DecisionReason::Unsupported => "unsupported",
            DecisionReason::BelowThreshold { .. } => "below_threshold",
            DecisionReason::OverBudget { .. } => "over_budget",
            DecisionReason::UploadNotAllowed => "upload_not_allowed",
            DecisionReason::KTooLarge { .. } => "k_too_large",
            DecisionReason::AfterError(_) => "after_error",
        }
    }
}

impl AcceleratorError {
    /// Short stable label for [`DecisionReason::AfterError`].
    pub fn fallback_reason(&self) -> &'static str {
        match self {
            AcceleratorError::FeatureDisabled { .. } => "feature_disabled",
            AcceleratorError::DeviceUnavailable { .. } => "device_unavailable",
            AcceleratorError::Unsupported { .. } => "unsupported",
            AcceleratorError::SnapshotStale { .. } => "snapshot_stale",
            AcceleratorError::OutOfBudget { .. } => "out_of_budget",
            AcceleratorError::LimitExceeded { .. } => "limit_exceeded",
            AcceleratorError::Backend { .. } => "backend_error",
            AcceleratorError::InvalidArgument(_) => "invalid_argument",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Caps;

    fn cuda(resident: bool, max_bytes: Option<usize>) -> Caps {
        let mut c = Caps::new(BackendKind::Cuda, "cuda-0", max_bytes).with_resident(resident);
        for op in [Op::ValidTimeFilter, Op::KnnFlat, Op::BatchKhop] {
            c = c.with_op(op);
        }
        c
    }

    #[test]
    fn cpu_is_the_answer_when_nothing_is_registered() {
        let policy = ExecutionPolicy::default();
        let caps = Caps::cpu();
        let d = policy.decide(&DecisionInputs::new(&caps, Op::KnnFlat, 10_000_000));
        assert_eq!(
            d,
            BackendDecision::Cpu {
                reason: DecisionReason::NoDevice
            }
        );
    }

    #[test]
    fn a_backend_without_residency_is_never_used() {
        // The C0 finding: without residency every call pays the transfer, and
        // the transfer was 96% of the cold path.
        let policy = ExecutionPolicy::default();
        let caps = cuda(false, None);
        let d = policy.decide(&DecisionInputs::new(&caps, Op::KnnFlat, 10_000_000));
        assert_eq!(d.reason(), Some(DecisionReason::NoDevice));
    }

    #[test]
    fn unsupported_operator_falls_back_per_operator() {
        let policy = ExecutionPolicy::default();
        let caps = cuda(true, None); // window_reduce not registered
        let d = policy.decide(&DecisionInputs::new(&caps, Op::WindowReduce, 10_000_000));
        assert_eq!(d.reason(), Some(DecisionReason::Unsupported));
        // ... while the operators it does have still go to the device.
        let d = policy.decide(&DecisionInputs::new(&caps, Op::KnnFlat, 10_000_000));
        assert_eq!(d, BackendDecision::CudaLoadThenRun);
    }

    #[test]
    fn small_work_stays_on_the_cpu_but_residency_overrides_the_threshold() {
        let policy = ExecutionPolicy::default();
        let caps = cuda(true, None);
        let d = policy.decide(&DecisionInputs::new(&caps, Op::KnnFlat, 10));
        assert!(matches!(
            d.reason(),
            Some(DecisionReason::BelowThreshold { .. })
        ));
        // Already uploaded: the transfer is sunk, so size no longer matters.
        let d = policy.decide(&DecisionInputs::new(&caps, Op::KnnFlat, 10).resident(true));
        assert_eq!(d, BackendDecision::CudaResident);
    }

    #[test]
    fn budget_and_upload_policy_are_separate_refusals() {
        let caps = cuda(true, Some(1 << 20));
        let d = ExecutionPolicy::default().decide(
            &DecisionInputs::new(&caps, Op::KnnFlat, 1_000_000).bytes(2 << 20),
        );
        assert_eq!(
            d.reason(),
            Some(DecisionReason::OverBudget {
                needed: 2 << 20,
                limit: 1 << 20
            })
        );

        let d = ExecutionPolicy::default()
            .with_upload(false)
            .decide(&DecisionInputs::new(&caps, Op::KnnFlat, 1_000_000));
        assert_eq!(d.reason(), Some(DecisionReason::UploadNotAllowed));
    }

    #[test]
    fn a_failed_backend_is_not_retried_in_the_same_query() {
        let policy = ExecutionPolicy::default();
        let caps = cuda(true, None);
        let err = AcceleratorError::DeviceUnavailable {
            backend: "cuda",
            reason: "driver reset".into(),
        };
        let d = policy.decide(&DecisionInputs::new(&caps, Op::KnnFlat, 1_000_000).after(Some(&err)));
        assert_eq!(d.reason(), Some(DecisionReason::AfterError("device_unavailable")));
    }

    #[test]
    fn a_k_the_kernel_cannot_hold_is_routed_at_plan_time() {
        // 8 warps x k x 12 B has to fit a block's shared memory; the CPU has no
        // such bound, so an oversized k is a scheduling fact, not a failure.
        let caps = cuda(true, None).with_max_k(256);
        let ok = ExecutionPolicy::default()
            .decide(&DecisionInputs::new(&caps, Op::KnnFlat, 10_000_000).with_k(256));
        assert_eq!(ok, BackendDecision::CudaLoadThenRun);

        let d = ExecutionPolicy::default()
            .decide(&DecisionInputs::new(&caps, Op::KnnFlat, 10_000_000).with_k(257));
        assert_eq!(
            d.reason(),
            Some(DecisionReason::KTooLarge {
                k: 257,
                max_k: 256
            })
        );
        assert_eq!(DecisionReason::KTooLarge { k: 1, max_k: 0 }.as_str(), "k_too_large");
    }

    #[test]
    fn forced_cpu_beats_every_other_reason() {
        let caps = cuda(true, None);
        let d = ExecutionPolicy::cpu_only().decide(
            &DecisionInputs::new(&caps, Op::KnnFlat, 10_000_000).resident(true),
        );
        assert_eq!(d.reason(), Some(DecisionReason::ForcedCpu));
    }

    #[test]
    fn a_bare_filter_is_never_admitted_on_its_own() {
        // C0 §3.2: 1e7-edge filter was 0.556 ms on the device and 13.86 ms end
        // to end, i.e. 96% transfer — only worth it chained behind another op.
        let caps = cuda(true, None);
        let d = ExecutionPolicy::default()
            .decide(&DecisionInputs::new(&caps, Op::ValidTimeFilter, 10_000_000));
        assert!(matches!(
            d.reason(),
            Some(DecisionReason::BelowThreshold { .. })
        ));
    }
}
