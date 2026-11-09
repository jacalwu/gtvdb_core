//! One error type for every backend, and one classification of it.
//!
//! The classification matters more than the variants: a device error is a
//! *scheduling* fact, not a query failure. [`AcceleratorError::fallback`] says
//! whether the caller may retry on the CPU and still return a correct answer,
//! and the only case that says "no" is a malformed argument — retrying that on
//! another backend would just hide a caller bug.

use thiserror::Error;

/// What the caller should do with an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallback {
    /// The answer is still computable elsewhere; rerun on the CPU reference.
    Cpu,
    /// Not a backend problem. Surface it.
    Surface,
}

/// Failure of a backend, never of the storage layer.
///
/// No variant represents "the data is wrong": device memory is a cache, so the
/// authoritative copy is always the CPU-side segment and every variant here is
/// recoverable by not using the device.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AcceleratorError {
    /// The backend was compiled out (`--features cuda` absent).
    #[error("accelerator {backend} not compiled in")]
    FeatureDisabled { backend: &'static str },

    /// Present but unusable: no driver, no device, unsupported compute
    /// capability, init failure. Carries the reason for the log line.
    #[error("accelerator {backend} unavailable: {reason}")]
    DeviceUnavailable {
        backend: &'static str,
        reason: String,
    },

    /// The backend is alive and healthy but does not implement this operator
    /// yet. Distinct from `DeviceUnavailable` on purpose: this is the normal
    /// state of a partially ported backend and must not disable it.
    #[error("backend {backend} does not implement {op}")]
    Unsupported {
        backend: &'static str,
        op: &'static str,
    },

    /// A resident handle outlived the snapshot it was built from.
    #[error("resident handle is for snapshot {resident}, query wants {requested}")]
    SnapshotStale { resident: u64, requested: u64 },

    /// A request that cannot fit in the device budget even after eviction.
    ///
    /// The caller downgrades to the CPU; this is never a query failure.
    #[error("device budget {limit} bytes cannot hold {needed} bytes")]
    OutOfBudget { needed: usize, limit: usize },

    /// A well-formed request outside this backend's parameter limits, where
    /// another backend can still serve it.
    ///
    /// Distinct from [`AcceleratorError::InvalidArgument`], which means the
    /// caller is wrong and retrying elsewhere would hide a bug. A device kernel
    /// that keeps its candidate heap in shared memory cannot answer `k =
    /// 10_000`; that is a fact about the kernel, not about the query, and the
    /// CPU reference has no such bound.
    #[error("backend {backend} cannot serve {what}={requested} (limit is {limit})")]
    LimitExceeded {
        backend: &'static str,
        what: &'static str,
        requested: usize,
        limit: usize,
    },

    /// The driver rejected a call.
    #[error("backend {backend} failed in {op}: {message} (code {code})")]
    Backend {
        backend: &'static str,
        op: &'static str,
        code: i32,
        message: String,
    },

    /// Malformed input: wrong length, wrong dimension, stale candidate set.
    #[error("{0}")]
    InvalidArgument(String),
}

impl AcceleratorError {
    /// Whether the caller can rerun on the CPU and still return the right
    /// answer.
    pub fn fallback(&self) -> Fallback {
        match self {
            AcceleratorError::InvalidArgument(_) => Fallback::Surface,
            _ => Fallback::Cpu,
        }
    }

    /// True when the backend itself is not at fault, so a registry should keep
    /// it in the candidate list.
    pub fn is_backend_fault(&self) -> bool {
        matches!(
            self,
            AcceleratorError::DeviceUnavailable { .. }
                | AcceleratorError::Backend { .. }
                | AcceleratorError::OutOfBudget { .. }
                | AcceleratorError::LimitExceeded { .. }
        )
    }
}

impl From<gtv_core::GtvError> for AcceleratorError {
    fn from(e: gtv_core::GtvError) -> Self {
        AcceleratorError::InvalidArgument(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_malformed_input_surfaces() {
        assert_eq!(
            AcceleratorError::InvalidArgument("bad".into()).fallback(),
            Fallback::Surface
        );
        for e in [
            AcceleratorError::FeatureDisabled { backend: "cuda" },
            AcceleratorError::DeviceUnavailable {
                backend: "cuda",
                reason: "no driver".into(),
            },
            AcceleratorError::Unsupported {
                backend: "cuda",
                op: "knn_flat",
            },
            AcceleratorError::SnapshotStale {
                resident: 1,
                requested: 2,
            },
            AcceleratorError::OutOfBudget {
                needed: 2,
                limit: 1,
            },
            AcceleratorError::LimitExceeded {
                backend: "cuda",
                what: "k",
                requested: 10_000,
                limit: 256,
            },
        ] {
            assert_eq!(e.fallback(), Fallback::Cpu, "{e}");
        }
    }

    #[test]
    fn unsupported_is_not_a_backend_fault() {
        // A backend that has not ported an operator yet still works for the
        // operators it has; disabling it would be wrong.
        assert!(!AcceleratorError::Unsupported {
            backend: "cuda",
            op: "window_reduce"
        }
        .is_backend_fault());
        assert!(AcceleratorError::DeviceUnavailable {
            backend: "cuda",
            reason: "no device".into()
        }
        .is_backend_fault());
    }
}
