//! V-1: on-disk I/O for [`gtv_index::VectorSegment`] — write-atomically + mmap
//! back, mirroring how `gtv-storage` publishes GTVSeg segments.
//!
//! This is the storage half of the `GTVVEC v1` charter
//! (`docs/vector-corpus-gtvseg.md`). It deliberately lives *beside* the existing
//! [`crate::store::IndexStore`] path rather than inside it: the `AnyIndex`
//! payloads are still `GFLATv1`-serialized `Vec`-backed indexes (V-2 swaps
//! `FlatIndex` onto `Col`). Until then, a corpus can be published as an
//! immutable GTVVEC file and read back zero-copy, without disturbing the
//! versioned `.gtvidx` container machinery.
//!
//! # mmap adapter
//!
//! `gtv-storage` keeps its `Mmap` adapter private, so this module owns a small
//! [`MappedRegion`] implementation over `memmap2::Mmap`. The safety contract is
//! identical: read-only, immutable file, published by atomic rename, address
//! stable for the lifetime of the `Arc`.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use gtv_core::column::MappedRegion;
use gtv_index::{encode_vector_segment, open_vector_segment, VectorSegment};
use gtv_storage::write_atomic;
use memmap2::Mmap;

use crate::{IndexStoreError, Result};

/// An `mmap` over an immutable GTVVEC file, as a [`MappedRegion`].
struct MappedFile(Mmap);

impl fmt::Debug for MappedFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedFile").field("len", &self.0.len()).finish()
    }
}

impl MappedRegion for MappedFile {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        &self.0[..]
    }
}

/// A [`MappedRegion`] view over a byte sub-range of an enclosing region.
///
/// Used to hand the payload slice of a `.gtvidx` container to
/// [`open_vector_segment`] as if it were a standalone GTVVEC file: the container
/// header (magic + version + flags + manifest) precedes the payload, so the
/// GTVVEC magic sits at `payload_offset`, not at byte 0.
struct SubRegion {
    inner: Arc<dyn MappedRegion>,
    offset: usize,
}

impl fmt::Debug for SubRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubRegion").field("offset", &self.offset).finish()
    }
}

impl MappedRegion for SubRegion {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        &self.inner.as_bytes()[self.offset..]
    }
}

/// Serialize `(ids, data, dim, metric)` and publish it at `path` atomically.
///
/// The caller owns normalization: for Cosine pass already-normalized rows and
/// `normalized = true`.
#[allow(clippy::too_many_arguments)]
pub fn write_vector_segment(
    path: impl AsRef<Path>,
    ids: &[u64],
    data: &[f32],
    dim: usize,
    metric: gtv_core::Metric,
    normalized: bool,
) -> Result<()> {
    let bytes = encode_vector_segment(ids, data, dim, metric, normalized)
        .map_err(IndexStoreError::Index)?;
    write_atomic(path.as_ref(), &bytes).map_err(IndexStoreError::Storage)
}

/// mmap a GTVVEC file and open it read-only, validating the header (O(1)).
///
/// Column bytes are zero-copy views over the mapping; the returned segment
/// keeps the mapping alive. Call [`VectorSegment::verify_data`] explicitly for
/// a full O(file size) integrity check.
pub fn open_vector_segment_at(path: impl AsRef<Path>) -> Result<VectorSegment> {
    let path = path.as_ref().to_path_buf();
    let file = std::fs::File::open(&path).map_err(IndexStoreError::Io)?;
    let file_len = file.metadata().map_err(IndexStoreError::Io)?.len() as usize;
    if file_len < gtv_index::vector_segment::HEADER_SIZE {
        return Err(IndexStoreError::Corrupt(format!(
            "GTVVEC file too short: {file_len} bytes"
        )));
    }
    // SAFETY: GTVVEC is immutable; we open read-only and publish via atomic
    // rename only. The mapping cannot be mutated while a `VectorSegment` holds
    // a pointer into it. Callers must not truncate/replace the path in place.
    let map = unsafe { Mmap::map(&file) }.map_err(IndexStoreError::Io)?;
    let region: Arc<dyn MappedRegion> = Arc::new(MappedFile(map));
    open_vector_segment(region).map_err(IndexStoreError::Index)
}

/// mmap a `.gtvidx` container file and open the GTVVEC payload it wraps.
///
/// The container layout is `magic(8) | version(2) | flags(2) | manifest_len(4)
/// | manifest | payload | checksum(32)`; the GTVVEC payload therefore begins at
/// `16 + manifest_len`. This is the cold-open path [`crate::IndexStore::load`]
/// takes when it detects a GTVVEC payload: one `mmap`, O(1) header validation,
/// and the corpus columns are zero-copy views into the mapping (no
/// `rows × dim` materialization).
pub fn open_vector_segment_from_container(
    path: impl AsRef<Path>,
) -> Result<gtv_index::VectorSegment> {
    use crate::container::split_container_header;

    let path = path.as_ref().to_path_buf();
    let file = std::fs::File::open(&path).map_err(IndexStoreError::Io)?;
    let file_len = file.metadata().map_err(IndexStoreError::Io)?.len() as usize;
    if file_len < crate::container::HEADER_LEN {
        return Err(IndexStoreError::Corrupt(format!(
            "index container too short: {file_len} bytes"
        )));
    }
    // SAFETY: same immutable/atomic-publish reasoning as `open_vector_segment_at`.
    let map = unsafe { Mmap::map(&file) }.map_err(IndexStoreError::Io)?;
    let region: Arc<dyn MappedRegion> = Arc::new(MappedFile(map));

    let (payload_offset, _payload_end) = split_container_header(region.as_bytes())?;
    let sub: Arc<dyn MappedRegion> = Arc::new(SubRegion {
        inner: region,
        offset: payload_offset,
    });
    open_vector_segment(sub).map_err(IndexStoreError::Index)
}

/// Read only the container header and report whether its payload begins with the
/// GTVVEC magic — a cheap O(1) probe that lets [`crate::IndexStore::load`]
/// dispatch between the mmap cold-open path and the legacy decode path without
/// materializing the payload.
pub fn container_payload_is_gtvvec(path: impl AsRef<Path>) -> Result<bool> {
    use std::io::{Read, Seek, SeekFrom};

    let path = path.as_ref().to_path_buf();
    let mut f = std::fs::File::open(&path).map_err(IndexStoreError::Io)?;

    let mut header = [0u8; crate::container::HEADER_LEN];
    if f.read_exact(&mut header).is_err() {
        return Ok(false); // too short to be a valid container
    }
    if header[0..8] != *crate::container::CONTAINER_MAGIC {
        return Ok(false);
    }
    let manifest_len = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    let payload_offset = crate::container::HEADER_LEN + manifest_len;

    f.seek(SeekFrom::Start(payload_offset as u64))
        .map_err(IndexStoreError::Io)?;
    let mut magic = [0u8; 8];
    if f.read_exact(&mut magic).is_err() {
        return Ok(false);
    }
    Ok(magic == gtv_index::vector_segment::MAGIC)
}

/// Open a GTVVEC file and run a full data-integrity check in one step.
pub fn verify_vector_segment(path: impl AsRef<Path>) -> Result<()> {
    let seg = open_vector_segment_at(path)?;
    seg.verify_data().map_err(IndexStoreError::Index)
}

/// The path an engine would store a corpus under (`DATA_DIR`-relative), kept
/// here as the single place the `.gtvvec` suffix is spelled.
pub fn corpus_file_name(name: &str) -> String {
    format!("{name}.gtvvec")
}

/// Convenience: derive a full path from a root dir and a corpus name.
pub fn corpus_path(root: &Path, name: &str) -> PathBuf {
    root.join(corpus_file_name(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gtv_core::Metric;

    #[test]
    fn write_open_round_trip() {
        let dir = std::env::temp_dir().join("gtvvec_v1_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("demo.gtvvec");

        let ids = vec![10u64, 20, 30];
        let data = vec![
            1.0f32, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ];
        write_vector_segment(&path, &ids, &data, 4, Metric::L2, false).unwrap();

        let seg = open_vector_segment_at(&path).unwrap();
        assert_eq!(seg.row_count(), 3);
        assert_eq!(seg.dim(), 4);
        assert_eq!(seg.metric(), Metric::L2);
        assert_eq!(seg.ids(), &ids[..]);
        assert_eq!(seg.data(), &data[..]);
        assert!(seg.verify_data().is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_missing_file_is_io_error() {
        let dir = std::env::temp_dir().join("gtvvec_v1_missing");
        std::fs::create_dir_all(&dir).unwrap();
        let err = open_vector_segment_at(dir.join("nope.gtvvec")).unwrap_err();
        assert!(matches!(err, IndexStoreError::Io(_)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn verify_detects_corruption() {
        let dir = std::env::temp_dir().join("gtvvec_v1_corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.gtvvec");
        let ids = vec![1u64, 2];
        let data = vec![1.0f32, 0.0, 0.0, 1.0];
        write_vector_segment(&path, &ids, &data, 2, Metric::L2, false).unwrap();

        // Corrupt a byte in the vectors column.
        let bytes = std::fs::read(&path).unwrap();
        let mut corrupted = bytes.clone();
        let last = corrupted.len() - 1;
        corrupted[last] ^= 0xff;
        std::fs::write(&path, &corrupted).unwrap();

        assert!(verify_vector_segment(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
