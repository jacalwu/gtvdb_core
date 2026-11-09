//! gtv-engine: the SQL surface of the temporal graph platform.
//!
//! This crate is the *core* DataFusion integration: temporal / bitemporal graph
//! access, vector search, catalog-aware planning, lineage, data-quality gates and
//! quantum **state metadata** (storage, not simulation).
//!
//! Everything domain-specific — banking analytics, HFT/quant trading, quantum
//! compute, crypto/PQC, the q frontend — lives in `gtv_ext_lib/` and reaches SQL
//! only through [`ext::GtvExtension`]. Nothing in this crate names a domain, so a
//! core build stays small and an extension can be dropped without touching it.
//!
//! ```text
//! gtv-engine (core)                 gtv_ext_lib/ (workloads)
//!   as_of / ann / knn / graph  <──   banking: crm_alloc, cr_ead_rw, le_*
//!   dq_gate / cbo_explain      <──   hft:     tick_to_trade, backtest, var
//!   quantum_state_*            <──   quantum: quantum_knn, quantum_walk
//!   gtv_extensions()                  crypto/q: token signing, q parser
//! ```

pub mod accelerator;
pub mod aggregate;
pub mod ann;
pub mod asof;
pub mod batch_graph;
pub mod bitemporal;
pub mod catalog_meta;
pub mod cbo;
pub mod context;
pub mod csv;
pub mod datetime;
pub mod dq;
pub mod embedding;
pub mod ext;
pub mod graph;
pub mod invalidation;
pub mod knn;
pub mod lineage;
pub mod monitor;
pub mod motif;
pub mod quantum_state;
pub mod registry;
pub mod segment;
pub mod udf;
pub mod workload;

pub mod expr_util;

pub use aggregate::{
    empty_aggregate_slot, ActiveEdgeCountTableFunction, AggregateSlot, RangeAggTableFunction,
};
pub use batch_graph::{BatchKhopTable, BatchNeighborsTable, khop_schema, neighbors_schema};
pub use context::{ExecutionOptions, GtvContext};
pub use catalog_meta::{CatalogSlot, FilesTableFunction, SnapshotsTableFunction, TablesTableFunction};
pub use dq::{
    blocking_failures, decision_from_outcomes, evaluate, evaluate_rule, evaluate_rules,
    DqGateTableFunction, RuleOutcome,
};
pub use embedding::{EmbeddingCollection, EmbeddingHit, EmbeddingRegistry, EmbeddingSearchTableFunction};
pub use ext::{ExtensionInfo, ExtensionRegistry, GtvExtension};
pub use invalidation::{
    empty_invalidation_slot, InvalidationsTableFunction, InvalidationSlot, TagsTableFunction,
};
pub use lineage::{extract_udfs, ReplayError, ENGINE_VERSION, LINEAGE_TABLE};
pub use motif::{
    build_motif_tables, edge_schema, find_matches, match_schema, MotifBatches, MotifEdgeTable,
    MotifFromTableFunction, MotifMatchTable, MotifShape, MotifSpec, MotifTableFunction,
};
pub use registry::{AsofResource, HftRegistry, PitResource, ResourceRegistry};
