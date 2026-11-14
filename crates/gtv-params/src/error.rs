use std::path::PathBuf;

/// Errors that abort parameter loading.
///
/// Unknown keys and missing *optional* files are *warnings*, not errors (see
/// [`crate::Warnings`]); these variants are the hard failures that should stop
/// boot.
#[derive(Debug, thiserror::Error)]
pub enum ParamError {
    /// The parameters file named by `GTV_PARAMS_FILE` / a discovered path does
    /// not exist.
    #[error("parameters file not found: {0}")]
    FileNotFound(PathBuf),

    /// I/O error while reading a located parameters file.
    #[error("io error reading {path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },

    /// A line could not be interpreted as `KEY=VALUE` / `KEY VALUE`.
    #[error("parse error on line {line}: {message}")]
    Parse { line: usize, message: String },

    /// A recognized key had a value that could not be parsed / validated.
    #[error("invalid value for {key}: `{value}` ({reason})")]
    InvalidValue {
        key: String,
        value: String,
        reason: String,
    },

    /// `CUDA_MODE=ON` was requested but no CUDA device is available.
    #[error("CUDA_MODE=ON but no CUDA device is available (set CUDA_MODE=AUTO or OFF)")]
    CudaNoDevice,

    /// `CUDA_VRAM` could not be resolved because no device total was supplied.
    #[error("CUDA_VRAM could not be resolved (no device total available)")]
    CudaVramUnknown,

    /// `MEMORY_OBJ_AREA` is larger than `MEMORY_TARGET`.
    #[error(
        "MEMORY_OBJ_AREA ({area} MiB) exceeds MEMORY_TARGET ({target} MiB)"
    )]
    ObjAreaExceedsTarget { area: u64, target: u64 },
}

/// A non-fatal issue encountered while loading parameters.
///
/// Unknown keys and a missing-but-optional file produce warnings so that a
/// slightly-off config still boots instead of crashing.
#[derive(Debug, Clone)]
pub struct Warning {
    /// Source line number, when the warning came from a file.
    pub line: Option<usize>,
    /// The parameter key involved, when known.
    pub key: Option<String>,
    /// Human-readable explanation.
    pub message: String,
}

impl std::fmt::Display for Warning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.line, &self.key) {
            (Some(l), Some(k)) => write!(f, "line {}: key `{}`: {}", l, k, self.message),
            (Some(l), None) => write!(f, "line {}: {}", l, self.message),
            (None, Some(k)) => write!(f, "key `{}`: {}", k, self.message),
            (None, None) => write!(f, "{}", self.message),
        }
    }
}

/// Collection of non-fatal warnings produced during a load.
pub type Warnings = Vec<Warning>;
