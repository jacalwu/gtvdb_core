//! Error and warning types for parameter parsing.

use std::path::PathBuf;

/// A fatal problem in a parameter file that prevents gtVdB from starting with a
/// coherent configuration.
///
/// Parse failures are fatal on purpose: a parameter file is the single source of
/// truth for deployment behaviour, so a typo must stop the process rather than
/// silently fall back to defaults that look right but are not what was written.
#[derive(Debug, thiserror::Error)]
pub enum ParamError {
    #[error("parameter file not found: {0}")]
    FileNotFound(PathBuf),

    #[error("could not read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("line {line}: {message}")]
    Parse { line: usize, message: String },

    #[error("invalid value for {key}: `{value}` ({reason})")]
    InvalidValue {
        key: String,
        value: String,
        reason: String,
    },

    #[error("CUDA_MODE=ON but no CUDA device is available; set CUDA_MODE=FALSE or run on a machine with a GPU")]
    CudaNoDevice,

    #[error(
        "CUDA_VRAM is a fraction but the device VRAM is unknown (no device); set CUDA_VRAM to an explicit size"
    )]
    CudaVramUnknown,

    #[error("MEMORY_OBJ_AREA ({area} B) exceeds MEMORY_TARGET ({target} B)")]
    ObjAreaExceedsTarget { area: u64, target: u64 },
}

/// A non-fatal observation while loading parameters.
///
/// Unknown keys and unreadable-but-optional files produce warnings, not errors,
/// so an operator can extend the file over time without breaking older builds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Warning {
    pub line: usize,
    pub key: String,
    pub message: String,
}

impl std::fmt::Display for Warning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "params: line {}: [{}] {}", self.line, self.key, self.message)
    }
}

/// Convenience alias used by [`crate::GtvParams::load`].
pub type Warnings = Vec<Warning>;
