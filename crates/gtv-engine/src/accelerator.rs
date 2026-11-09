//! Where the engine registers its backends — and the only place that knows
//! whether a CUDA backend was compiled in.
//!
//! The default build contains no device code at all: `gtv-accelerator` is a pure
//! CPU crate, and `gtv-accelerator-cuda` is an optional dependency behind the
//! `cuda` feature. Nothing in the query path consults this module today, which is
//! deliberate — C1 added the plumbing and changed no behaviour, and C2 added a
//! kernel behind the trait without wiring it into SQL, so `cargo test` results
//! are still identical with and without the feature.
//!
//! Registration is conditional at *run* time as well as build time. A build with
//! `--features cuda` on a machine without a driver produces a CPU-only registry:
//! the failure is a missing device, not a failed query.

use gtv_accelerator::{AcceleratorRegistry, ExecutionPolicy};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Build the process-wide backend list.
///
/// `Ok` on every machine: a device that cannot open is reported through
/// [`registry_state`] and simply not registered.
pub fn registry() -> AcceleratorRegistry {
    #[allow(unused_mut)]
    let mut reg = AcceleratorRegistry::new();
    #[cfg(feature = "cuda")]
    {
        match gtv_accelerator_cuda::CudaAccelerator::open() {
            Ok(acc) => {
                reg.register(Arc::new(acc));
            }
            Err(_) => {
                // No device, no driver, or a driver that failed to initialise.
                // The CPU reference is already registered; this is the answer
                // `AcceleratorRegistry::route` will report as `NoDevice`.
            }
        }
    }
    reg
}

/// What the build can do, independent of what the machine can do.
///
/// Kept separate from [`registry`] because the two answers differ in the way
/// that matters when diagnosing a slow benchmark: "not compiled in" and "no
/// device found" produce the same routing but not the same fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendAvailability {
    /// The crate was compiled with `--features cuda`.
    pub cuda_compiled: bool,
    /// A device answered at registration time.
    pub cuda_available: bool,
}

impl BackendAvailability {
    /// One-line explanation for a log or a benchmark header.
    pub fn describe(&self) -> &'static str {
        match (self.cuda_compiled, self.cuda_available) {
            (false, _) => "feature_disabled",
            (true, false) => "device_unavailable",
            (true, true) => "device_ready",
        }
    }
}

/// Probe availability without building a registry.
pub fn availability() -> BackendAvailability {
    #[cfg(feature = "cuda")]
    {
        BackendAvailability {
            cuda_compiled: true,
            cuda_available: gtv_accelerator_cuda::CudaAccelerator::probe().is_ok(),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        BackendAvailability {
            cuda_compiled: false,
            cuda_available: false,
        }
    }
}

/// Policy defaults for this engine.
///
/// A separate function rather than a `Default` impl so the engine can change
/// thresholds without changing the crate's public default, and so a test can
/// assert that the engine and the crate agree today.
pub fn policy() -> ExecutionPolicy {
    ExecutionPolicy::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gtv_accelerator::{Accelerator, BackendKind};

    #[test]
    fn the_cpu_reference_is_always_registered() {
        let reg = registry();
        assert_eq!(reg.cpu().kind(), BackendKind::Cpu);
        assert!(reg.cpu_caps().resident);
    }

    #[test]
    fn the_default_build_registers_no_device() {
        // This assertion is the feature boundary. If it ever fails without
        // `--features cuda`, an indirect dependency has pulled the CUDA crate
        // into the default build, which the spec forbids.
        if !cfg!(feature = "cuda") {
            assert!(registry().is_empty());
            assert!(!availability().cuda_compiled);
            assert_eq!(availability().describe(), "feature_disabled");
        }
    }

    #[test]
    fn availability_explains_missing_devices() {
        let a = availability();
        if a.cuda_compiled && !a.cuda_available {
            assert_eq!(a.describe(), "device_unavailable");
            // Still a usable registry: the CPU reference is the answer.
            assert!(registry().cpu_caps().supports(gtv_accelerator::Op::KnnFlat));
        }
    }
}
