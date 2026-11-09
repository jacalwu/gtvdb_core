//! gtv-storage: Arrow ↔ Parquet persistence plus a multi-versioned in-memory
//! time-travel store.

pub mod aggregate;
pub mod atomic;
pub mod bitemporal;
pub mod cache;
pub mod csv;
pub mod delta;
pub mod error;
pub mod gc;
pub mod hdb;
pub mod mem_segment;
pub mod node_lifetime;
pub mod observation;
pub mod parquet;
pub mod segment;
pub mod segment_meta;
pub mod snapshot;
pub mod store;
pub mod wal;

pub use aggregate::{AggregateView, EdgeInterval, LogicalEdge, RangeBucket};
pub use atomic::{fsync_dir, tmp_path, write_atomic, write_batch_atomic};
pub use bitemporal::BitemporalStore;
pub use gc::{GcPolicy, GcReport, ReadScope};

pub use cache::StaticCache;
pub use csv::read_csv;
pub use error::{Result, StorageError};
pub use hdb::{read_parquet_mmap, HdbStore};
pub use parquet::{
    default_writer_properties, infer_csv_schema, parquet_schema, read_batches, write_batch,
    DEFAULT_MAX_ROW_GROUP_ROWS,
};
pub use snapshot::{Snapshot, SnapshotStore};
pub use segment::{
    encode_segment, open_segment, write_segment, ColumnDir, Segment, SegmentHeader,
};
pub use segment_meta::{PruneOutcome, SegmentMeta};
pub use mem_segment::{IngestedEdge, MemSegment};
pub use observation::{
    decode_observation, encode_observation, read_observation, ObservationEntry,
};
pub use store::{
    read_invalidation_log, CompactionInput, CompactionPlan, DeltaDraft, DeltaEntry,
    EdgeCorrectionPlan, FlushReport, InvalidationRow, LiveRow, Manifest, MaterializeReport,
    NodePublish, NodeVersionEntry, PruneSummary, PublishReport, RecoveryReport, SegmentEntry,
    SegmentStore, MAX_DELTA_DEPTH,
};
pub use wal::{EdgeRecord, Wal, WalFrame, WalScan};
