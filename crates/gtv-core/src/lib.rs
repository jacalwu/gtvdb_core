//! gtv-core: in-memory data structures and extension traits for the
//! Temporal-Columnar Graph-Vector engine.
//!
//! Phase 1 provides the immutable in-memory Temporal-CSR index, Arrow-backed
//! node/edge tables, and the three core extension traits.

pub mod active_bitmap;
pub mod bitemporal;
pub mod chunk;
pub mod column;
pub mod csr;
pub mod edge_attr;
pub mod endpoint;
pub mod error;
pub mod graph;
pub mod metric;
pub mod provenance;
pub mod quantum_state;
pub mod receipt;
pub mod selection;
pub mod table;
pub mod temporal;
pub mod traits;
pub mod traversal;

pub use active_bitmap::{ActiveNodeBitmap, BitmapCache, BitmapDecision, BitmapKey, BitmapPlanner};
pub use bitemporal::{
    bitemporal_edge_schema, business_date, find_overlaps, migrate_legacy_edges, BitemporalRange,
    Overlap, OPEN_ENDED,
};
pub use chunk::TemporalEdgeChunk;
pub use column::{Col, ColumnError, MappedRegion, Pod};
pub use csr::{
    CsrColumns, DirectionMode, Neighbor, NeighborStrategy, Neighbors, TemporalCSR, TemporalCsrStats,
};
pub use edge_attr::{AttrCol, AttrPred, EdgeAttrSchema, EdgeAttrs, RangeOp};
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
pub use receipt::{
    allows_partial_rows, is_supported_version, negotiate_version, plan_hash, topk_dedup,
    topk_dedup_by_score, tri_state_str, Absence, DedupKey, EdgeRef, Freshness, FreshnessVerdict,
    Incomplete, IncompleteReason, KnowledgeClock, PathEvidence, PointBinding, QueryReceipt,
    QueryStats, ENGINE_VERSION, MIN_SUPPORTED_PROTOCOL, PROTOCOL_VERSION, RECEIPT_METADATA_JSON,
    RECEIPT_METADATA_PREFIX,
};
pub use selection::{Sel, SelIter, SelRepr, UNBOUND_SNAPSHOT};
pub use table::{EdgeTable, NodeTable};
pub use traits::{CustomTemporalOperator, TemporalGraphIndex, VectorHit, VectorIndex};
pub use traversal::{
    BudgetTracker, CancelToken, EdgePredicate, KhopResult, TraversalBudget, TraversalOutcome,
    TraversalStats, TriState, TruncReason, Truncation, VisitedSet,
};
