//! gtv-core: in-memory data structures and extension traits for the
//! Temporal-Columnar Graph-Vector engine.
//!
//! Phase 1 provides the immutable in-memory Temporal-CSR index, Arrow-backed
//! node/edge tables, and the three core extension traits.

pub mod bitemporal;
pub mod chunk;
pub mod column;
pub mod csr;
pub mod endpoint;
pub mod error;
pub mod graph;
pub mod metric;
pub mod provenance;
pub mod quantum_state;
pub mod table;
pub mod temporal;
pub mod traits;
pub mod traversal;

pub use bitemporal::{
    bitemporal_edge_schema, business_date, find_overlaps, migrate_legacy_edges, BitemporalRange,
    Overlap, OPEN_ENDED,
};
pub use chunk::TemporalEdgeChunk;
pub use column::{Col, ColumnError, MappedRegion, Pod};
pub use csr::{
    CsrColumns, DirectionMode, Neighbor, NeighborStrategy, Neighbors, TemporalCSR, TemporalCsrStats,
};
pub use endpoint::{
    ActiveNeighbors, EndpointPolicy, EndpointView, NodeLifetime, OPEN_FROM, OPEN_TO,
};
pub use error::{GtvError, Result};
pub use graph::TemporalGraph;
pub use metric::{DistanceMetric, Metric};
pub use provenance::{
    BranchSet, ProvenanceEdge, ProvenanceError, ProvenanceGraph, ProvenanceNode, ProvenanceStats,
    StateProbability,
};
pub use quantum_state::{
    Gate, ProbabilitySemantics, QuantumStateLog, QuantumStateRecord, Reconstruction, StateError,
};
pub use table::{EdgeTable, NodeTable};
pub use traits::{CustomTemporalOperator, TemporalGraphIndex, VectorHit, VectorIndex};
pub use traversal::{
    BudgetTracker, CancelToken, EdgePredicate, KhopResult, TraversalBudget, TraversalStats,
    VisitedSet,
};
