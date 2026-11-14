//! GTVVEC v1 — an immutable, fixed-width, cache-line-aligned vector segment.
//!
//! The vector-corpus sibling of `gtv-storage`'s GTVSeg. Where GTVSeg stores a
//! temporal graph CSR, GTVVEC stores a K-NN corpus — a row-major `f32` matrix
//! plus its `u64` ids — laid out so that
//!
//! ```text
//! disk byte layout == in-memory FlatIndex layout == CPU scan access unit
//! ```
//!
//! Opening one is `mmap + header validation`: no decode, no allocation
//! proportional to `rows × dim`, no copy. The OS page cache *is* the corpus,
//! and the stable mapped page address is what later lets P1b-2
//! (`cudaHostRegister`) DMA straight from the mapping. See
//! `docs/vector-corpus-gtvseg.md` for the charter.
//!
//! # Format (v1)
//!
//! All integers are little-endian. The header is a fixed 4096 bytes so every
//! column starts 64-byte aligned (one x86-64 cache line).
//!
//! ```text
//! 0    magic[8]            "GTVVEC\0\0"
//! 8    version   u16       1
//! 10   flags     u16       see FLAG_*
//! 12   row_count  u64
//! 20   dim        u32
//! 24   metric     u8       L2=0 / Cosine=1 / Ip=2 (gtv-index bytes.rs)
//! 25   reserved[7]
//! 32   reserved[32]
//! 64   col_dir[4] { offset u64, len_bytes u64, elem_size u8, kind u8,
//!                    reserved[6] } = 24 B each
//! 160  reserved[96]
//! 256  header_checksum[32]  blake3 over bytes [0, 256)
//! 288  data_checksum[32]    blake3 over bytes [4096, EOF)
//! 320  reserved[3776]
//! 4096 column 0..1, each padded up to the next 64 B boundary
//! ```
//!
//! Columns, in a fixed order (slots 2 and 3 are reserved):
//!
//! | # | kind | element | count |
//! |---|------|---------|-------|
//! | 0 | ids      | `u64` | row_count |
//! | 1 | vectors  | `f32` | row_count × dim |
//!
//! # Checksums and cold-open cost
//!
//! `header_checksum` is **always** verified on open and covers every field that
//! could make a mapped column out of bounds or misaligned. It is a constant
//! 256-byte hash, so validation stays O(1) even for a 100 GB corpus.
//!
//! `data_checksum` covers the column bytes. The writer computes it, but the
//! reader does **not** by default — hashing the whole file on open would turn
//! an O(1) mmap open into an O(file size) read, exactly the cost this format
//! exists to remove. Call [`VectorSegment::verify_data`] explicitly for
//! offline integrity checks.
//!
//! # Immutability
//!
//! Segments are written once (the caller uses `write_atomic`) and never
//! modified in place. An update must produce a *new* segment (copy-on-write).
//!
//! # Cosine normalization
//!
//! Normalization happens at **write** time (the segment is immutable, so the
//! reader cannot normalize in place the way `FlatIndex::from_flat_metric`
//! does). When [`Metric::Cosine`] is requested the writer normalizes the rows
//! and sets [`FLAG_NORMALIZED`]; the reader trusts that flag.

use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use gtv_core::column::{Col, MappedCol, MappedRegion, Pod};
use gtv_core::{GtvError, Metric, Result};

pub const MAGIC: [u8; 8] = *b"GTVVEC\0\0";
pub const VERSION: u16 = 1;
/// Fixed header size. Every column offset is a multiple of this and of 64.
pub const HEADER_SIZE: usize = 4096;
/// Cache-line alignment for every column start.
pub const ALIGN: usize = 64;
/// Total column-directory slots (ids, vectors, and two reserved).
pub const COL_SLOTS: usize = 4;
/// Slots actually used by v1.
const USED_SLOTS: usize = 2;

/// Rows were unit-normalized at write time (Cosine).
pub const FLAG_NORMALIZED: u16 = 1 << 0;
/// File was produced on a big-endian host (v1 reader rejects it).
pub const FLAG_BIG_ENDIAN: u16 = 1 << 1;
const FLAG_KNOWN: u16 = FLAG_NORMALIZED | FLAG_BIG_ENDIAN;

const OFF_HEADER_CHECKSUM: usize = 256;
const OFF_DATA_CHECKSUM: usize = 288;
const OFF_COL_DIR: usize = 64;
const COL_DIR_SIZE: usize = 24;

/// One column's location inside the segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnDir {
    pub offset: usize,
    pub len_bytes: usize,
    pub elem_size: u8,
    pub kind: u8,
}

impl ColumnDir {
    const EMPTY: Self = Self {
        offset: 0,
        len_bytes: 0,
        elem_size: 0,
        kind: 0,
    };
}

/// Parsed and validated GTVVEC header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorHeader {
    pub version: u16,
    pub flags: u16,
    pub row_count: usize,
    pub dim: usize,
    pub metric: Metric,
    pub columns: [ColumnDir; COL_SLOTS],
    pub header_checksum: [u8; 32],
    pub data_checksum: [u8; 32],
}

impl VectorHeader {
    /// Whether the stored rows are unit-normalized (Cosine).
    pub fn is_normalized(&self) -> bool {
        self.flags & FLAG_NORMALIZED != 0
    }
}

/// An opened, memory-mapped vector segment.
///
/// Derefs to its row-major `f32` matrix; ids via [`VectorSegment::ids`].
pub struct VectorSegment {
    region: Arc<dyn MappedRegion>,
    header: VectorHeader,
    ids: Col<u64>,
    data: Col<f32>,
}

impl fmt::Debug for VectorSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorSegment")
            .field("row_count", &self.header.row_count)
            .field("dim", &self.header.dim)
            .field("metric", &self.header.metric)
            .finish()
    }
}

impl Deref for VectorSegment {
    type Target = [f32];

    #[inline]
    fn deref(&self) -> &[f32] {
        &self.data
    }
}

impl VectorSegment {
    pub fn header(&self) -> &VectorHeader {
        &self.header
    }

    pub fn row_count(&self) -> usize {
        self.header.row_count
    }

    pub fn dim(&self) -> usize {
        self.header.dim
    }

    pub fn metric(&self) -> Metric {
        self.header.metric
    }

    pub fn is_normalized(&self) -> bool {
        self.header.is_normalized()
    }

    /// Zero-copy row-major vector matrix (vector `i` = `data[i*dim..(i+1)*dim]`).
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Zero-copy ids, aligned with [`VectorSegment::data`].
    pub fn ids(&self) -> &[u64] {
        &self.ids
    }

    /// Full integrity check over the column bytes (O(file size)).
    pub fn verify_data(&self) -> Result<()> {
        let bytes = self.region.as_bytes();
        if bytes.len() < HEADER_SIZE {
            return Err(GtvError::InvalidArgument(
                "GTVVEC shorter than 4096-byte header".into(),
            ));
        }
        let got = blake3::hash(&bytes[HEADER_SIZE..]);
        if got.as_bytes() != &self.header.data_checksum {
            return Err(GtvError::InvalidArgument(
                "GTVVEC data checksum mismatch (column bytes corrupted)".into(),
            ));
        }
        Ok(())
    }
}

/// Map `region` (a whole GTVVEC file already mmap'd by the caller) read-only.
///
/// Returns a segment whose columns are zero-copy views into `region`. The
/// caller supplies the mapping as an [`Arc<dyn MappedRegion>`] so it can reuse
/// whatever mapping type it already has (e.g. `gtv-storage`'s private
/// `MappedFile`, or a plain `memmap2::Mmap` adapter in the index store).
pub fn open_vector_segment(region: Arc<dyn MappedRegion>) -> Result<VectorSegment> {
    if cfg!(target_endian = "big") {
        return Err(GtvError::InvalidArgument(
            "GTVVEC v1 requires a little-endian host".into(),
        ));
    }
    let header = parse_header(region.as_bytes())?;
    build(&header, region)
}

fn u64le(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}
fn u32le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn u16le(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap())
}

/// Parse and fully validate a GTVVEC header from mapped bytes.
fn parse_header(b: &[u8]) -> Result<VectorHeader> {
    if b.len() < HEADER_SIZE {
        return Err(GtvError::InvalidArgument(
            "GTVVEC shorter than 4096-byte header".into(),
        ));
    }
    if b[0..8] != MAGIC {
        return Err(GtvError::InvalidArgument(
            "GTVVEC bad magic (not a GTVVEC file, or truncated)".into(),
        ));
    }
    let version = u16le(b, 8);
    if version != VERSION {
        return Err(GtvError::InvalidArgument(format!(
            "unsupported GTVVEC version {version} (reader supports {VERSION})"
        )));
    }
    let flags = u16le(b, 10);
    if flags & !FLAG_KNOWN != 0 {
        return Err(GtvError::InvalidArgument(format!(
            "GTVVEC unknown flags bits {:#x}",
            flags & !FLAG_KNOWN
        )));
    }
    if flags & FLAG_BIG_ENDIAN != 0 {
        return Err(GtvError::InvalidArgument(
            "GTVVEC was produced on a big-endian host".into(),
        ));
    }

    // Header checksum covers every field that determines column placement.
    let mut stored_header = [0u8; 32];
    stored_header.copy_from_slice(&b[OFF_HEADER_CHECKSUM..OFF_HEADER_CHECKSUM + 32]);
    if blake3::hash(&b[0..OFF_HEADER_CHECKSUM]).as_bytes() != &stored_header {
        return Err(GtvError::InvalidArgument(
            "GTVVEC header checksum mismatch (metadata corrupted)".into(),
        ));
    }

    let mut stored_data = [0u8; 32];
    stored_data.copy_from_slice(&b[OFF_DATA_CHECKSUM..OFF_DATA_CHECKSUM + 32]);

    let row_count = u64le(b, 12) as usize;
    let dim = u32le(b, 20) as usize;
    let metric = match b[24] {
        0 => Metric::L2,
        1 => Metric::Cosine,
        2 => Metric::Ip,
        other => {
            return Err(GtvError::InvalidArgument(format!(
                "GTVVEC unknown metric code {other}"
            )))
        }
    };

    if dim == 0 {
        return Err(GtvError::InvalidArgument("GTVVEC dim is zero".into()));
    }

    let vector_count = row_count
        .checked_mul(dim)
        .ok_or_else(|| GtvError::InvalidArgument("GTVVEC row_count × dim overflow".into()))?;
    let expected_counts = [row_count, vector_count, 0, 0];
    let expected_elem = [8u8, 4, 0, 0];

    let mut columns = [ColumnDir::EMPTY; COL_SLOTS];
    for i in 0..COL_SLOTS {
        let base = OFF_COL_DIR + i * COL_DIR_SIZE;
        let offset = u64le(b, base) as usize;
        let len_bytes = u64le(b, base + 8) as usize;
        let elem_size = b[base + 16];
        let kind = b[base + 17];
        let dir = ColumnDir {
            offset,
            len_bytes,
            elem_size,
            kind,
        };

        if i < USED_SLOTS {
            if kind != i as u8 {
                return Err(GtvError::InvalidArgument(format!(
                    "GTVVEC column {i} has unexpected kind {kind}"
                )));
            }
            if elem_size != expected_elem[i] {
                return Err(GtvError::InvalidArgument(format!(
                    "GTVVEC column {i} element size {elem_size}, expected {}",
                    expected_elem[i]
                )));
            }
            if offset < HEADER_SIZE || offset % ALIGN != 0 {
                return Err(GtvError::InvalidArgument(format!(
                    "GTVVEC column {i} offset {offset} is not 64 B aligned / past header"
                )));
            }
            if len_bytes % elem_size as usize != 0 {
                return Err(GtvError::InvalidArgument(format!(
                    "GTVVEC column {i} length {len_bytes} is not a multiple of {elem_size}"
                )));
            }
            if len_bytes / elem_size as usize != expected_counts[i] {
                return Err(GtvError::InvalidArgument(format!(
                    "GTVVEC column {i} has {} elements, expected {}",
                    len_bytes / elem_size as usize,
                    expected_counts[i]
                )));
            }
            let end = offset
                .checked_add(len_bytes)
                .ok_or_else(|| GtvError::InvalidArgument("GTVVEC column offset overflow".into()))?;
            if end > b.len() {
                return Err(GtvError::InvalidArgument(format!(
                    "GTVVEC column {i} ends at {end}, file has {} bytes",
                    b.len()
                )));
            }
        } else if offset != 0 || len_bytes != 0 {
            return Err(GtvError::InvalidArgument(format!(
                "GTVVEC reserved column slot {i} is not empty"
            )));
        }
        columns[i] = dir;
    }

    Ok(VectorHeader {
        version,
        flags,
        row_count,
        dim,
        metric,
        columns,
        header_checksum: stored_header,
        data_checksum: stored_data,
    })
}

fn mapped_col<T: Pod>(
    region: &Arc<dyn MappedRegion>,
    dir: &ColumnDir,
    slot: usize,
) -> Result<Col<T>> {
    let size = std::mem::size_of::<T>();
    if dir.elem_size as usize != size {
        return Err(GtvError::InvalidArgument(format!(
            "GTVVEC column {slot} element size mismatch for mapped type"
        )));
    }
    let n = dir.len_bytes / size;
    MappedCol::new(Arc::clone(region), dir.offset, n).map_err(|e| {
        GtvError::InvalidArgument(format!("GTVVEC column {slot}: {e}"))
    })
}

fn build(header: &VectorHeader, region: Arc<dyn MappedRegion>) -> Result<VectorSegment> {
    let c = &header.columns;
    let ids = mapped_col::<u64>(&region, &c[0], 0)?;
    let data = mapped_col::<f32>(&region, &c[1], 1)?;
    Ok(VectorSegment {
        region,
        header: header.clone(),
        ids,
        data,
    })
}

fn align_up(v: usize, a: usize) -> Result<usize> {
    v.checked_add(a - 1)
        .map(|x| x & !(a - 1))
        .ok_or_else(|| GtvError::InvalidArgument("segment offset overflow".into()))
}

fn write_u64_le(dst: &mut [u8], src: &[u64]) {
    for (chunk, v) in dst.chunks_exact_mut(8).zip(src) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
}
fn write_f32_le(dst: &mut [u8], src: &[f32]) {
    for (chunk, v) in dst.chunks_exact_mut(4).zip(src) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
}

/// Serialize `(ids, data, dim, metric)` into GTVVEC v1 bytes.
///
/// The caller owns normalization: for Cosine it must pass already-normalized
/// rows and set `normalized = true` so the reader can trust `FLAG_NORMALIZED`.
/// For L2 / Ip pass the raw rows with `normalized = false`.
pub fn encode_vector_segment(
    ids: &[u64],
    data: &[f32],
    dim: usize,
    metric: Metric,
    normalized: bool,
) -> Result<Vec<u8>> {
    if cfg!(target_endian = "big") {
        return Err(GtvError::InvalidArgument(
            "GTVVEC writer requires a little-endian host".into(),
        ));
    }
    if dim == 0 {
        return Err(GtvError::InvalidArgument("GTVVEC dim is zero".into()));
    }
    let row_count = ids.len();
    if data.len() != row_count * dim {
        return Err(GtvError::InvalidArgument(
            "data length != ids.len() * dim".into(),
        ));
    }
    // A normalized flag must not be set unless the metric actually requires it.
    if normalized && !metric.requires_normalization() {
        return Err(GtvError::InvalidArgument(
            "normalized flag set for a non-normalizing metric".into(),
        ));
    }

    let specs: [(usize, usize); USED_SLOTS] = [(ids.len(), 8), (data.len(), 4)];

    let mut cursor = HEADER_SIZE;
    let mut dirs = [ColumnDir::EMPTY; COL_SLOTS];
    for (i, (n, elem)) in specs.iter().enumerate() {
        cursor = align_up(cursor, ALIGN)?;
        let len_bytes = n
            .checked_mul(*elem)
            .ok_or_else(|| GtvError::InvalidArgument("column byte length overflow".into()))?;
        dirs[i] = ColumnDir {
            offset: cursor,
            len_bytes,
            elem_size: *elem as u8,
            kind: i as u8,
        };
        cursor = cursor
            .checked_add(len_bytes)
            .ok_or_else(|| GtvError::InvalidArgument("segment size overflow".into()))?;
    }
    let total = align_up(cursor, ALIGN)?;
    let mut buf = vec![0u8; total];

    buf[0..8].copy_from_slice(&MAGIC);
    buf[8..10].copy_from_slice(&VERSION.to_le_bytes());
    let mut flags = 0u16;
    if normalized {
        flags |= FLAG_NORMALIZED;
    }
    buf[10..12].copy_from_slice(&flags.to_le_bytes());
    buf[12..20].copy_from_slice(&(row_count as u64).to_le_bytes());
    buf[20..24].copy_from_slice(&(dim as u32).to_le_bytes());
    buf[24] = match metric {
        Metric::L2 => 0,
        Metric::Cosine => 1,
        Metric::Ip => 2,
    };

    for (i, d) in dirs.iter().enumerate() {
        let base = OFF_COL_DIR + i * COL_DIR_SIZE;
        buf[base..base + 8].copy_from_slice(&(d.offset as u64).to_le_bytes());
        buf[base + 8..base + 16].copy_from_slice(&(d.len_bytes as u64).to_le_bytes());
        buf[base + 16] = d.elem_size;
        buf[base + 17] = d.kind;
    }

    write_u64_le(&mut buf[dirs[0].offset..dirs[0].offset + dirs[0].len_bytes], ids);
    write_f32_le(&mut buf[dirs[1].offset..dirs[1].offset + dirs[1].len_bytes], data);

    let header_hash = blake3::hash(&buf[0..OFF_HEADER_CHECKSUM]);
    buf[OFF_HEADER_CHECKSUM..OFF_HEADER_CHECKSUM + 32].copy_from_slice(header_hash.as_bytes());
    let data_hash = blake3::hash(&buf[HEADER_SIZE..]);
    buf[OFF_DATA_CHECKSUM..OFF_DATA_CHECKSUM + 32].copy_from_slice(data_hash.as_bytes());

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct FakeRegion(Vec<u8>);
    impl MappedRegion for FakeRegion {
        fn as_bytes(&self) -> &[u8] {
            &self.0
        }
    }

    fn l2_corpus() -> (Vec<u64>, Vec<f32>, usize) {
        let ids = vec![10, 20, 30];
        // 3 rows × 4 dims
        let data = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ];
        (ids, data, 4)
    }

    #[test]
    fn round_trip_l2() {
        let (ids, data, dim) = l2_corpus();
        let bytes = encode_vector_segment(&ids, &data, dim, Metric::L2, false).unwrap();
        assert_eq!(&bytes[0..8], &MAGIC);
        let region: Arc<dyn MappedRegion> = Arc::new(FakeRegion(bytes));
        let seg = open_vector_segment(region).unwrap();
        assert_eq!(seg.row_count(), 3);
        assert_eq!(seg.dim(), 4);
        assert_eq!(seg.metric(), Metric::L2);
        assert!(!seg.is_normalized());
        assert_eq!(seg.ids(), &[10, 20, 30]);
        assert_eq!(seg.data(), &data);
        assert!(seg.verify_data().is_ok());
    }

    #[test]
    fn cosine_is_flagged_normalized() {
        let ids = vec![1u64, 2];
        // Two unit vectors (already normalized).
        let data = vec![1.0, 0.0, 0.0, 1.0];
        let bytes = encode_vector_segment(&ids, &data, 2, Metric::Cosine, true).unwrap();
        let region: Arc<dyn MappedRegion> = Arc::new(FakeRegion(bytes));
        let seg = open_vector_segment(region).unwrap();
        assert_eq!(seg.metric(), Metric::Cosine);
        assert!(seg.is_normalized());
    }

    #[test]
    fn normalized_flag_rejected_for_l2() {
        let (ids, data, dim) = l2_corpus();
        assert!(encode_vector_segment(&ids, &data, dim, Metric::L2, true).is_err());
    }

    #[test]
    fn data_length_mismatch_rejected() {
        let ids = vec![1u64, 2];
        let data = vec![0.0f32; 6]; // 2 rows × 3 dims expected, but dim=4
        assert!(encode_vector_segment(&ids, &data, 4, Metric::L2, false).is_err());
    }

    #[test]
    fn tampered_header_is_rejected() {
        let (ids, data, dim) = l2_corpus();
        let mut bytes = encode_vector_segment(&ids, &data, dim, Metric::L2, false).unwrap();
        bytes[12] ^= 0xff; // corrupt row_count before checksum
        let region: Arc<dyn MappedRegion> = Arc::new(FakeRegion(bytes));
        assert!(open_vector_segment(region).is_err());
    }

    #[test]
    fn data_corruption_detected_by_verify() {
        let (ids, data, dim) = l2_corpus();
        let mut bytes = encode_vector_segment(&ids, &data, dim, Metric::L2, false).unwrap();
        // Flip a byte in the vectors column (past the header).
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let region: Arc<dyn MappedRegion> = Arc::new(FakeRegion(bytes));
        let seg = open_vector_segment(region).unwrap(); // open still succeeds (header ok)
        assert!(seg.verify_data().is_err());
    }
}
