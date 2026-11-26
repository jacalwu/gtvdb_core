//! Payload-kind dispatch for `.gtvidx` containers.
//!
//! A container's payload is self-describing: its first eight bytes are a
//! magic. The store uses that magic to choose between two very different
//! read paths:
//!
//! | kind | magic | read path |
//! |---|---|---|
//! | [`PayloadKind::GtvVec`] | `GTVVEC\0\0` | `mmap` the container, reinterpret the corpus columns in place (O(1) cold open) |
//! | [`PayloadKind::IvfCompound`] | `GIVFCMP\0` | `mmap` the container, decode the IVF cell layout, map the corpus (O(rows) cold open) |
//! | [`PayloadKind::Legacy`] | anything else | read the file, verify the container checksum, decode the payload |
//!
//! The legacy path stays because old containers (`GFLATv1`, `GIVFv1/v2`,
//! `GIVFH1`) are still valid and must keep loading unchanged.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use gtv_core::column::{subregion, MappedRegion};
use gtv_index::{open_ivf_payload, open_vector_segment, AnyIndex, FlatIndex};
use memmap2::Mmap;

use crate::container::{decode, split_container_header, CONTAINER_MAGIC, HEADER_LEN};
use crate::manifest::IndexManifest;
use crate::{IndexStoreError, Result};

/// What a container's payload is, decided by its leading magic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadKind {
    /// `GTVVEC\0\0` — a flat corpus.
    GtvVec,
    /// `GIVFCMP\0` — a compound IVF payload (cell layout + GTVVEC corpus).
    IvfCompound,
    /// Any other payload: the legacy self-contained encodings.
    Legacy,
}

impl PayloadKind {
    /// Classify a payload by its leading magic.
    pub fn from_magic(magic: &[u8]) -> Self {
        if magic == &gtv_index::vector_segment::MAGIC[..] {
            PayloadKind::GtvVec
        } else if magic == &gtv_index::ivf_segment::MAGIC[..] {
            PayloadKind::IvfCompound
        } else {
            PayloadKind::Legacy
        }
    }
}

/// An `mmap` over an immutable container file, as a [`MappedRegion`].
struct MappedContainer(Mmap);

impl fmt::Debug for MappedContainer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedContainer")
            .field("len", &self.0.len())
            .finish()
    }
}

impl MappedRegion for MappedContainer {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        &self.0[..]
    }
}

/// Map a container file read-only.
///
/// # Safety contract
///
/// The container is immutable: it is published by atomic rename and never
/// modified in place. The mapping's address is stable for as long as the
/// returned `Arc` lives, which is what lets the payload columns be viewed
/// in place.
fn map_container(path: &Path) -> Result<Arc<dyn MappedRegion>> {
    let file = std::fs::File::open(path).map_err(IndexStoreError::Io)?;
    let file_len = file.metadata().map_err(IndexStoreError::Io)?.len() as usize;
    if file_len < HEADER_LEN {
        return Err(IndexStoreError::Corrupt(format!(
            "index container too short: {file_len} bytes"
        )));
    }
    // SAFETY: the container is immutable and opened read-only; the mapping
    // cannot be mutated while a payload column points into it.
    let map = unsafe { Mmap::map(&file) }.map_err(IndexStoreError::Io)?;
    Ok(Arc::new(MappedContainer(map)))
}

/// Classify a container's payload without materializing it (O(1)).
pub fn container_payload_kind(path: impl AsRef<Path>) -> Result<PayloadKind> {
    let path = path.as_ref();
    let mut header = [0u8; HEADER_LEN];
    let mut f = std::fs::File::open(path).map_err(IndexStoreError::Io)?;
    use std::io::Read;
    if f.read_exact(&mut header).is_err() {
        return Ok(PayloadKind::Legacy); // too short to be a valid container
    }
    if header[0..8] != *CONTAINER_MAGIC {
        return Ok(PayloadKind::Legacy);
    }
    let manifest_len = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    let payload_offset = HEADER_LEN + manifest_len;

    use std::io::{Seek, SeekFrom};
    f.seek(SeekFrom::Start(payload_offset as u64))
        .map_err(IndexStoreError::Io)?;
    let mut magic = [0u8; 8];
    if f.read_exact(&mut magic).is_err() {
        return Ok(PayloadKind::Legacy);
    }
    Ok(PayloadKind::from_magic(&magic))
}

/// Open a container's payload with the mmap cold-open path.
///
/// Returns `Ok(None)` when the container holds a legacy payload — the caller
/// then falls back to the materializing decode path.
pub fn open_mapped_payload(path: impl AsRef<Path>) -> Result<Option<AnyIndex>> {
    let path = path.as_ref();
    let region = map_container(path)?;
    let (payload_offset, payload_end) = split_container_header(region.as_bytes())?;
    // Bound the payload view: the container's trailing checksum must not be
    // visible to the payload's own checksum (which covers exactly its bytes).
    let payload = subregion(region, payload_offset, Some(payload_end - payload_offset));

    let kind = match payload.as_bytes().get(..8) {
        Some(magic) => PayloadKind::from_magic(magic),
        None => PayloadKind::Legacy,
    };
    match kind {
        PayloadKind::GtvVec => {
            let segment = open_vector_segment(payload).map_err(IndexStoreError::Index)?;
            Ok(Some(AnyIndex::Flat(FlatIndex::from_segment(segment))))
        }
        PayloadKind::IvfCompound => {
            let index = open_ivf_payload(payload).map_err(IndexStoreError::Index)?;
            Ok(Some(AnyIndex::Ivf(index)))
        }
        PayloadKind::Legacy => Ok(None),
    }
}

/// Load an `AnyIndex` from a single `.gtvidx` container file, transparently
/// choosing the cold-open (mmap) path when the payload is mmap-friendly
/// (`GTVVEC` / `GIVFCMP`) and falling back to the materializing decode path
/// for legacy payloads.
///
/// This is the bare-file counterpart of [`IndexStore::load`]: it needs no
/// store registry or version directory, which is exactly what a REPL
/// `.knn-index` / `.ann-index` cold-open command wants.
pub fn load_index(path: impl AsRef<Path>) -> Result<AnyIndex> {
    load_index_with_manifest(path).map(|(index, _)| index)
}

/// Like [`load_index`], but also returns the index manifest when one is
/// available.
///
/// The manifest is what carries the `vector_space_id`, so a cold-open caller
/// needs it to bind a `knn` collection to its embedding space. It is looked up
/// in two places, in order: a sibling `manifest.json` (the `IndexStore`
/// version directory layout), then the manifest embedded in a legacy container.
/// A cold-open `.gtvidx` payload with neither yields `None` — the caller then
/// registers an unbound collection rather than guessing a space.
pub fn load_index_with_manifest(
    path: impl AsRef<Path>,
) -> Result<(AnyIndex, Option<IndexManifest>)> {
    let path = path.as_ref();
    let sibling_manifest = path
        .parent()
        .map(|dir| dir.join("manifest.json"))
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|bytes| serde_json::from_slice::<IndexManifest>(&bytes).ok());

    // Cold-open path first: zero-copy mmap view for GTVVEC / GIVFCMP payloads.
    if let Some(any) = open_mapped_payload(path)? {
        return Ok((any, sibling_manifest));
    }
    // Legacy payload: read the file, split the container, and hand the payload
    // to `AnyIndex` with the manifest-recorded index type.
    let bytes = std::fs::read(path).map_err(IndexStoreError::Io)?;
    let (manifest_json, payload) = decode(&bytes)?;
    let manifest: IndexManifest = serde_json::from_slice(&manifest_json)?;
    let index = AnyIndex::from_bytes(manifest.index_type, &payload).map_err(IndexStoreError::Index)?;
    // The embedded manifest is authoritative for a legacy container.
    Ok((index, Some(manifest)))
}

/// Run the O(file size) data-integrity check over a container's mapped payload.
///
/// Legacy payloads are verified by the container's trailing checksum during
/// the decode path, so there is nothing to do here for them.
pub fn verify_mapped_payload(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let region = map_container(path)?;
    let (payload_offset, payload_end) = split_container_header(region.as_bytes())?;
    let payload = subregion(region, payload_offset, Some(payload_end - payload_offset));

    let kind = match payload.as_bytes().get(..8) {
        Some(magic) => PayloadKind::from_magic(magic),
        None => PayloadKind::Legacy,
    };
    match kind {
        PayloadKind::GtvVec => {
            let segment = open_vector_segment(payload).map_err(IndexStoreError::Index)?;
            segment.verify_data().map_err(IndexStoreError::Index)
        }
        PayloadKind::IvfCompound => {
            gtv_index::ivf_segment::verify_ivf_payload_data(payload)
                .map_err(IndexStoreError::Index)
        }
        PayloadKind::Legacy => Ok(()),
    }
}
