//! Error type shared across the gtv engine crates.

use thiserror::Error;

use crate::metric::Metric;

#[derive(Debug, Error)]
pub enum GtvError {
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("schema error: {0}")]
    Schema(String),

    #[error("node id out of range: {0}")]
    NodeOutOfRange(u64),

    #[error("metric mismatch: index={index}, query={query}")]
    MetricMismatch { index: Metric, query: Metric },

    #[error("dimension mismatch: index={index}, query={query}")]
    DimensionMismatch { index: usize, query: usize },

    /// A graph traversal exceeded one of its resource budgets.
    ///
    /// Since M1 budget exhaustion is *reported* on the result (see
    /// [`crate::traversal::Truncation`]) rather than raised; this variant stays
    /// for callers that want to convert a truncation into an error at their own
    /// boundary.
    #[error("budget exceeded during {stage}: limit={limit}, observed={observed}")]
    BudgetExceeded {
        stage: &'static str,
        limit: u64,
        observed: u64,
    },

    /// A candidate set built for one snapshot was handed to another.
    #[error("snapshot mismatch: selection was built for {expected}, operand is {got}")]
    SnapshotMismatch { expected: u64, got: u64 },

    /// A candidate set/index covers a different number of rows than expected.
    #[error("universe mismatch: expected {expected} rows, operand has {got}")]
    UniverseMismatch { expected: u64, got: u64 },

    /// A high-degree node was hit without a predicate to narrow the traversal.
    #[error(
        "high-degree node {node} (degree {degree}); add an edge predicate or raise the budget"
    )]
    HighDegreeNode {
        node: u64,
        degree: u64,
        hint: &'static str,
    },

    /// The traversal was cancelled by the caller.
    #[error("traversal cancelled")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, GtvError>;
