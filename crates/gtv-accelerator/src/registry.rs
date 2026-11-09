//! Backend registry: which implementations exist, and which one should answer.
//!
//! Registration is explicit and ordered: the CPU reference is always present
//! and always last, so "no device answered" is a normal outcome rather than a
//! branch. The CUDA backend registers itself only when the `cuda` feature is on
//! *and* a driver answered, so a registry built on a GPU-less machine and one
//! built on a GPU machine differ by content, not by code path.

use std::sync::Arc;

use crate::api::{Accelerator, BackendKind, Caps, Op};
use crate::cpu::CpuAccelerator;
use crate::policy::{BackendDecision, DecisionInputs, DecisionReason, ExecutionPolicy};

/// Ordered list of usable backends.
pub struct AcceleratorRegistry {
    /// Device backends in preference order.
    devices: Vec<Arc<dyn Accelerator>>,
    /// Always present, never selected for acceleration.
    cpu: Arc<CpuAccelerator>,
}

impl Default for AcceleratorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AcceleratorRegistry {
    /// A registry with only the CPU reference.
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
            cpu: Arc::new(CpuAccelerator::new()),
        }
    }

    /// Add a device backend. Backends that report `BackendKind::Cpu` are
    /// rejected: two references would make parity checks meaningless.
    pub fn register(&mut self, backend: Arc<dyn Accelerator>) -> &mut Self {
        if backend.kind() != BackendKind::Cpu {
            self.devices.push(backend);
        }
        self
    }

    /// The CPU reference. Never absent, so a caller always has a way to run.
    pub fn cpu(&self) -> Arc<CpuAccelerator> {
        Arc::clone(&self.cpu)
    }

    pub fn cpu_caps(&self) -> Caps {
        self.cpu.caps()
    }

    /// Device backends, in preference order.
    pub fn devices(&self) -> &[Arc<dyn Accelerator>] {
        &self.devices
    }

    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    /// First device that advertises `op` and does residency.
    ///
    /// "Advertises" is not "will win": the policy still applies thresholds and
    /// the budget. A backend that has not ported an operator is skipped for that
    /// operator only.
    pub fn best_for(&self, op: Op) -> Option<&Arc<dyn Accelerator>> {
        self.devices
            .iter()
            .find(|b| b.caps().resident && b.caps().supports(op))
    }

    /// Decide and, when the device wins, name it.
    ///
    /// Returning both keeps the reason and the choice from drifting apart: a
    /// caller cannot log one and act on the other.
    pub fn route<'a>(
        &'a self,
        policy: &ExecutionPolicy,
        inputs: &DecisionInputs<'_>,
    ) -> (BackendDecision, Option<&'a Arc<dyn Accelerator>>) {
        if let Some(backend) = self.best_for(inputs.op) {
            let decision = policy.decide(inputs);
            if decision.is_device() {
                return (decision, Some(backend));
            }
            return (decision, None);
        }
        // Nothing here can take the operator, and the reason differs:
        //
        // * a device exists and knows the operator but does not do residency —
        //   unusable by construction (C0: transfer dominates), report NoDevice;
        // * a device exists but has not ported this operator — report
        //   Unsupported, which is the normal state of a partially ported backend
        //   and must not read as "no GPU";
        // * nothing is registered — NoDevice.
        //
        // An operator owner asking "why is this still on the CPU?" needs those
        // three cases to be distinguishable in a log line.
        let known = self.devices.iter().any(|b| b.caps().supports(inputs.op));
        if known {
            return (
                BackendDecision::Cpu {
                    reason: DecisionReason::NoDevice,
                },
                None,
            );
        }
        if !self.devices.is_empty() {
            return (
                BackendDecision::Cpu {
                    reason: DecisionReason::Unsupported,
                },
                None,
            );
        }
        let caps = self.cpu_caps();
        let probe = DecisionInputs {
            caps: &caps,
            op: inputs.op,
            work: inputs.work,
            resident: inputs.resident,
            required_bytes: inputs.required_bytes,
            last_error: inputs.last_error,
            k: inputs.k,
        };
        (policy.decide(&probe), None)
    }
}

impl std::fmt::Debug for AcceleratorRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcceleratorRegistry")
            .field("devices", &self.devices.len())
            .field("cpu", &self.cpu.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ResidentGraph;
    use crate::error::AcceleratorError;
    use crate::policy::DecisionReason;

    /// A device backend that claims residency and one operator, and nothing
    /// else. Stands in for a real backend in tests that must not depend on a
    /// GPU being present.
    struct FakeDevice {
        ops: Vec<Op>,
        fail: bool,
    }

    impl Accelerator for FakeDevice {
        fn name(&self) -> &'static str {
            "fake-cuda"
        }
        fn kind(&self) -> BackendKind {
            BackendKind::Cuda
        }
        fn caps(&self) -> Caps {
            let mut c = Caps::new(BackendKind::Cuda, "fake-cuda", Some(1 << 30)).with_resident(true);
            for op in &self.ops {
                c = c.with_op(*op);
            }
            c
        }
        fn load_graph(
            &self,
            view: crate::api::SnapshotView<'_>,
        ) -> Result<Option<ResidentGraph>, AcceleratorError> {
            if self.fail {
                return Err(AcceleratorError::DeviceUnavailable {
                    backend: "fake-cuda",
                    reason: "simulated reset".into(),
                });
            }
            Ok(Some(ResidentGraph::new(
                1,
                view.snapshot_id,
                view.node_count,
                view.edge_count(),
                view.bytes(),
            )))
        }
    }

    #[test]
    fn cpu_only_registry_routes_everything_to_the_cpu() {
        let reg = AcceleratorRegistry::new();
        let policy = ExecutionPolicy::default();
        let caps = reg.cpu_caps();
        let inputs = DecisionInputs::new(&caps, Op::KnnFlat, 10_000_000);
        let (decision, backend) = reg.route(&policy, &inputs);
        assert_eq!(decision.reason(), Some(DecisionReason::NoDevice));
        assert!(backend.is_none());
    }

    #[test]
    fn a_device_is_picked_per_operator() {
        let mut reg = AcceleratorRegistry::new();
        reg.register(Arc::new(FakeDevice {
            ops: vec![Op::KnnFlat],
            fail: false,
        }));
        let policy = ExecutionPolicy::default();

        let caps = reg.best_for(Op::KnnFlat).unwrap().caps();
        let (d, b) = reg.route(&policy, &DecisionInputs::new(&caps, Op::KnnFlat, 10_000_000));
        assert_eq!(d, BackendDecision::CudaLoadThenRun);
        assert_eq!(b.unwrap().name(), "fake-cuda");

        // The operator it did not port falls through to the CPU, and says why:
        // `Unsupported` (a device exists, just not for this op), not `NoDevice`.
        assert!(reg.best_for(Op::BatchKhop).is_none());
        let caps = reg.devices()[0].caps();
        let (d, b) = reg.route(&policy, &DecisionInputs::new(&caps, Op::BatchKhop, 10_000_000));
        assert_eq!(d.reason(), Some(DecisionReason::Unsupported));
        assert!(b.is_none());
    }

    #[test]
    fn a_second_cpu_backend_cannot_be_registered() {
        // Two references would let a "parity" test compare the reference with
        // itself and always pass.
        struct AlsoCpu;
        impl Accelerator for AlsoCpu {
            fn name(&self) -> &'static str {
                "cpu-2"
            }
            fn kind(&self) -> BackendKind {
                BackendKind::Cpu
            }
            fn caps(&self) -> Caps {
                Caps::cpu()
            }
            fn load_graph(
                &self,
                _v: crate::api::SnapshotView<'_>,
            ) -> Result<Option<ResidentGraph>, AcceleratorError> {
                Ok(None)
            }
        }
        let mut reg = AcceleratorRegistry::new();
        reg.register(Arc::new(AlsoCpu));
        assert!(reg.is_empty());
    }
}
