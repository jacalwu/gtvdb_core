//! gtv-storage: Arrow ↔ Parquet persistence plus a multi-versioned in-memory
//! time-travel store.

pub mod atomic;
pub mod bitemporal;
pub mod cache;
pub mod csv;
pub mod error;
pub mod hdb;
pub mod mem_segment;
pub mod parquet;
pub mod segment;
pub mod segment_meta;
pub mod snapshot;
pub mod store;
pub mod wal;

pub use atomic::{fsync_dir, tmp_path, write_atomic, write_batch_atomic};
pub use bitemporal::BitemporalStore;

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
pub use store::{FlushReport, Manifest, PruneSummary, RecoveryReport, SegmentEntry, SegmentStore};
pub use wal::{EdgeRecord, Wal, WalFrame, WalScan};
