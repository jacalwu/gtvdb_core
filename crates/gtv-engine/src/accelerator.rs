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
use gtv_params::ResolvedParams;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Build the process-wide backend list.
///
/// `Ok` on every machine: a device that cannot open is reported through
/// [`registry_state`] and simply not registered. This form consults the parameter
/// file via `resolved`; callers that predate it (or tests) pass `None` and get
/// the historical behaviour — open the device if one is present.
pub fn registry() -> AcceleratorRegistry {
    registry_with(None)
}

/// Build the backend list honouring a resolved parameter file.
///
/// When `resolved` says CUDA is disabled (`CUDA_MODE=FALSE`), the device is never
/// opened. When enabled, the residency budget comes from `CUDA_VRAM` (typically
/// 80% of device VRAM) rather than the crate's 60% fallback. A device that fails
/// to open is simply not registered — the CPU reference remains.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub fn registry_with(resolved: Option<&ResolvedParams>) -> AcceleratorRegistry {
    let reg = AcceleratorRegistry::new();
    #[cfg(feature = "cuda")]
    {
        let enabled = match resolved {
            Some(r) => r.cuda_enabled,
            // Legacy callers: open the device if one is present.
            None => true,
        };
        if enabled {
            let budget = resolved
                .filter(|r| r.cuda_vram_bytes > 0)
                .map(|r| r.cuda_vram_bytes);
            let device = resolved.map(|r| r.cuda_device).unwrap_or(0);
            let opts = gtv_accelerator_cuda::OpenOptions { device, budget };
            match gtv_accelerator_cuda::CudaAccelerator::open_with(opts) {
                Ok(acc) => {
                    reg.register(Arc::new(acc));
                }
                Err(_) => {
                    // No device, no driver, or a driver that failed to
                    // initialise. The CPU reference is already registered.
                }
            }
        }
    }
    reg
}

/// Detected device VRAM in bytes, or `None` when the `cuda` feature is off or no
/// device is present. Used to resolve `CUDA_VRAM`'s fraction default.
pub fn cuda_device_total() -> Option<u64> {
    #[cfg(feature = "cuda")]
    {
        gtv_accelerator_cuda::CudaAccelerator::probe()
            .ok()
            .map(|d| d.total_mem_bytes)
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
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
