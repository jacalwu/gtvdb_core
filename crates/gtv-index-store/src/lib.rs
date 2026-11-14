//! gtv-index-store — persistence, versioning and lifecycle for vector indexes.
//!
//! Builds on the per-index serialization in [`gtv_index`] (B1-4 / B2-2):
//!
//! * [`IndexManifest`] records the corpus snapshot, embedding model/version,
//!   dimension, metric, build parameters and payload checksum of an index.
//! * the `.gtvidx` container wraps a manifest and an index payload with a
//!   versioned header and a trailing blake3 checksum.
//! * [`IndexStore`] keeps every build in `v<n>/`, with a `CURRENT` pointer that
//!   is switched atomically — enabling shadow builds, atomic swaps and rollback.

pub mod container;
pub mod embedding;
pub mod manifest;
pub mod payload;
pub mod store;
pub mod vector_segment_io;

pub use container::{decode, encode, split_container_header, CONTAINER_MAGIC, CONTAINER_VERSION};
pub use embedding::{build_and_save, build_index_from_embeddings, EmbeddingIndexError, EmbeddingIndexSpec};
pub use manifest::IndexManifest;
pub use payload::{container_payload_kind, open_mapped_payload, PayloadKind};
pub use store::{IndexMeta, IndexStore, IndexVersion, LoadedIndex};
pub use vector_segment_io::{
    container_payload_is_gtvvec, corpus_file_name, corpus_path, open_vector_segment_at,
    open_vector_segment_from_container, verify_vector_segment, write_vector_segment,
};

/// Errors from the index store.
#[derive(Debug, thiserror::Error)]
pub enum IndexStoreError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("catalog error: {0}")]
    Catalog(#[from] gtv_catalog::CatalogError),

    #[error("storage error: {0}")]
    Storage(#[from] gtv_storage::StorageError),

    #[error("index error: {0}")]
    Index(#[from] gtv_core::GtvError),

    #[error("index not found: {0}")]
    NotFound(String),

    #[error("index version not found: {name} v{version}")]
    VersionNotFound { name: String, version: u32 },

    #[error("corrupt index: {0}")]
    Corrupt(String),

    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, IndexStoreError>;
