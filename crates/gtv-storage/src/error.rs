//! Storage-layer error type (kept separate so gtv-core stays free of parquet).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    /// A second process (or a second handle) wants a mode the current holder
    /// forbids. Distinct from `Msg` because callers — and the server's startup
    /// path — need to tell "someone else has it" from "this directory is
    /// broken", and because the message has a fixed shape: it names the holder.
    #[error("data directory `{path}` is already open by {holder} — cannot open as {mode}")]
    Locked {
        path: String,
        holder: String,
        mode: crate::lock::OpenMode,
    },

    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, StorageError>;
