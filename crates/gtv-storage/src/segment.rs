//! GTVSeg v1 — an immutable, fixed-width, cache-line-aligned CSR segment.
//!
//! # Why this format exists
//!
//! Parquet is a *compressed and encoded interchange format*: reading it always
//! decompresses and materializes Arrow `RecordBatch`es, then `from_arrays`
//! copies the columns into `Vec`s and sorts them `O(E log E)`. A GTVSeg file is
//! instead laid out so that
//!
//! ```text
//! disk byte layout == in-memory CSR layout == CPU cache access unit
//! ```
//!
//! Opening one is `open + mmap + header validation`: no decode, no allocation
//! proportional to E, no sort. The OS page cache *is* the dataset.
//!
//! # Format (v1)
//!
//! All integers are little-endian. The header is a fixed 4096 bytes so every
//! column starts 64-byte aligned (one x86-64 cache line).
//!
//! ```text
//! 0    magic[8]            "GTVSEG\0\0"
//! 8    version   u16       1
//! 10   flags     u16       see FLAG_*
//! 12   node_count    u64
//! 20   edge_count    u64
//! 28   chunk_size    u32
//! 32   chunk_count   u32
//! 36   max_valid_from i64
//! 44   min_valid_to   i64
//! 52   reserved[12]
//! 64   col_dir[8]     { offset u64, len_bytes u64, elem_size u8, kind u8,
//!                       reserved[6] }  = 24 B each
//! 256  header_checksum[32]  blake3 over bytes [0, 256)
//! 288  data_checksum[32]    blake3 over bytes [4096, EOF)
//! 320  reserved[3776]
//! 4096 column 0..6, each padded up to the next 64 B boundary
//! ```
//!
//! Columns, in a fixed order (the 8th directory slot is reserved for the P2
//! skip-index columns):
//!
//! | # | kind | element | count |
//! |---|------|---------|-------|
//! | 0 | offsets | `u32` | node_count + 1 |
//! | 1 | chunk_index | `u32` | node_count + 1 |
//! | 2 | chunk_max_valid_to | `i64` | chunk_count |
//! | 3 | dst | `u64` | edge_count |
//! | 4 | valid_from | `i64` | edge_count |
//! | 5 | valid_to | `i64` | edge_count |
//! | 6 | edge_type | `u16` | edge_count |
//!
//! # Checksums and cold-open cost
//!
//! `header_checksum` is **always** verified on open and covers every field that
//! could make a mapped column out of bounds or misaligned. It is a constant
//! 256-byte hash, so validation stays O(1) even for a 100 GB segment.
//!
//! `data_checksum` covers the column bytes. The writer computes it, but the
//! reader does **not** by default — hashing the whole file on open would turn
//! an O(1) mmap open into an O(file size) read, exactly the cost this format
//! exists to remove. Call [`Segment::verify_data`] explicitly for offline
//! integrity checks / repair.
//!
//! # Immutability
//!
//! Segments are written once via [`write_atomic`] and never modified in place.
//! An mmap that is being read while its backing file is rewritten would observe
//! torn data, and `MappedCol`'s cached pointer assumes a stable address. Any
//! update must produce a *new* segment (copy-on-write) — see P1 of
//! `docs/storage-mmap-design.md`.

use std::fmt;
use std::fs::File;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use gtv_core::column::{Col, MappedCol, MappedRegion, Pod};
use gtv_core::csr::TemporalCSR;
use memmap2::Mmap;

use crate::atomic::write_atomic;
use crate::error::{Result, StorageError};

pub const MAGIC: [u8; 8] = *b"GTVSEG\0\0";
pub const VERSION: u16 = 1;
/// Fixed header size. Every column offset is a multiple of this and of 64.
pub const HEADER_SIZE: usize = 4096;
/// Cache-line alignment for every column start.
pub const ALIGN: usize = 64;
pub const COL_SLOTS: usize = 8;
const USED_SLOTS: usize = 7;

// flags — all currently reserved for P2 narrow/delta/skip-index formats.
/// `dst` stored as `u32` instead of `u64` (P2).
pub const FLAG_NARROW_DST: u16 = 1 << 0;
/// timestamps stored as `u32` deltas (P2).
pub const FLAG_DELTA_TIME: u16 = 1 << 1;
/// layered skip-index columns present (P2).
pub const FLAG_SKIP_INDEX: u16 = 1 << 2;
/// file was produced on a big-endian host (v1 reader rejects it).
pub const FLAG_BIG_ENDIAN: u16 = 1 << 3;
const FLAG_KNOWN: u16 =
    FLAG_NARROW_DST | FLAG_DELTA_TIME | FLAG_SKIP_INDEX | FLAG_BIG_ENDIAN;
/// The v1 reader understands none of the P2 encodings yet.
const FLAG_UNSUPPORTED: u16 = FLAG_NARROW_DST | FLAG_DELTA_TIME | FLAG_SKIP_INDEX;

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

/// Parsed and validated GTVSeg header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentHeader {
    pub version: u16,
    pub flags: u16,
    pub node_count: usize,
    pub edge_count: usize,
    pub chunk_size: usize,
    pub chunk_count: usize,
    pub max_valid_from: i64,
    pub min_valid_to: i64,
    pub columns: [ColumnDir; COL_SLOTS],
    pub header_checksum: [u8; 32],
    pub data_checksum: [u8; 32],
}

/// An opened, memory-mapped segment together with its ready-to-query CSR.
///
/// `Segment` derefs to `TemporalCSR`, so every existing graph operator works
/// unchanged. Cloning a `TemporalCSR` obtained from here is cheap: mapped
/// columns are `Arc` handles, not copies of the edge arrays.
pub struct Segment {
    path: PathBuf,
    region: Arc<MappedFile>,
    header: SegmentHeader,
    csr: TemporalCSR,
}

/// Adapter turning `memmap2::Mmap` into a [`MappedRegion`].
///
/// `Debug` is handwritten because `memmap2::Mmap`'s own `Debug` is not part of
/// its stable API surface, and `MappedRegion` requires it.
struct MappedFile(Mmap);

impl fmt::Debug for MappedFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedFile")
            .field("len", &self.0.len())
            .finish()
    }
}

impl MappedRegion for MappedFile {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        &self.0[..]
    }
}

impl fmt::Debug for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Segment")
            .field("path", &self.path)
            .field("node_count", &self.header.node_count)
            .field("edge_count", &self.header.edge_count)
            .finish()
    }
}

impl Deref for Segment {
    type Target = TemporalCSR;

    #[inline]
    fn deref(&self) -> &TemporalCSR {
        &self.csr
    }
}

impl Segment {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn header(&self) -> &SegmentHeader {
        &self.header
    }

    pub fn csr(&self) -> &TemporalCSR {
        &self.csr
    }

    pub fn file_len(&self) -> usize {
        self.region.as_bytes().len()
    }

    /// Full integrity check over the column bytes.
    ///
    /// Deliberately **not** called by `open`: it is O(file size). Use it in
    /// offline verification, repair tooling and tests.
    pub fn verify_data(&self) -> Result<()> {
        let bytes = self.region.as_bytes();
        if bytes.len() < HEADER_SIZE {
            return Err(StorageError::Msg("segment shorter than header".into()));
        }
        let got = blake3::hash(&bytes[HEADER_SIZE..]);
        if got.as_bytes() != &self.header.data_checksum {
            return Err(StorageError::Msg(
                "GTVSeg data checksum mismatch (column bytes corrupted)".into(),
            ));
        }
        Ok(())
    }
}

/// Open a GTVSeg file and return a CSR served directly from the mapping.
pub fn open_segment(path: impl AsRef<Path>) -> Result<Segment> {
    Segment::open(path)
}

impl Segment {
    /// Map `path` read-only and validate it. O(1) in the number of edges.
    pub fn open(path: impl AsRef<Path>) -> Result<Segment> {
        // v1 stores native-width little-endian integers and reinterprets them
        // in place. A big-endian host would silently read byte-swapped values,
        // so refuse it loudly instead of returning a plausible wrong graph.
        if cfg!(target_endian = "big") {
            return Err(StorageError::Msg(
                "GTVSeg v1 requires a little-endian host (no byte-swap reader yet)".into(),
            ));
        }

        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let file_len = file.metadata()?.len() as usize;
        if file_len < HEADER_SIZE {
            return Err(StorageError::Msg(format!(
                "GTVSeg file too short: {file_len} bytes (need >= {HEADER_SIZE})"
            )));
        }

        // SAFETY: the segment format is immutable; we open read-only, publish
        // via atomic rename only, and never write into a mapped file. The
        // mapping therefore cannot be mutated while `MappedCol` holds a
        // pointer into it. Callers must not truncate/replace the path in place.
        let map = unsafe { Mmap::map(&file) }?;
        let region = Arc::new(MappedFile(map));

        let header = parse_header(region.as_bytes())?;
        let region_trait: Arc<dyn MappedRegion> = region.clone();
        let csr = build_csr(&header, &region_trait)?;

        Ok(Segment {
            path,
            region,
            header,
            csr,
        })
    }
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
fn i64le(b: &[u8], at: usize) -> i64 {
    i64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// Parse and fully validate a segment header from mapped bytes.
fn parse_header(b: &[u8]) -> Result<SegmentHeader> {
    if b.len() < HEADER_SIZE {
        return Err(StorageError::Msg("GTVSeg shorter than 4096-byte header".into()));
    }
    if b[0..8] != MAGIC {
        return Err(StorageError::Msg(
            "GTVSeg bad magic (not a GTVSeg file, or truncated)".into(),
        ));
    }
    let version = u16le(b, 8);
    if version != VERSION {
        return Err(StorageError::Msg(format!(
            "unsupported GTVSeg version {version} (reader supports {VERSION})"
        )));
    }
    let flags = u16le(b, 10);
    if flags & !FLAG_KNOWN != 0 {
        return Err(StorageError::Msg(format!(
            "GTVSeg unknown flags bits {:#x}",
            flags & !FLAG_KNOWN
        )));
    }
    if flags & FLAG_UNSUPPORTED != 0 || flags & FLAG_BIG_ENDIAN != 0 {
        return Err(StorageError::Msg(format!(
            "GTVSeg flags {flags:#x} use a P2 encoding this v1 reader cannot decode"
        )));
    }

    // Header checksum covers every field that determines column placement.
    // Verify it *before* trusting any parsed offset or length.
    let mut stored_header = [0u8; 32];
    stored_header.copy_from_slice(&b[OFF_HEADER_CHECKSUM..OFF_HEADER_CHECKSUM + 32]);
    if blake3::hash(&b[0..OFF_HEADER_CHECKSUM]).as_bytes() != &stored_header {
        return Err(StorageError::Msg(
            "GTVSeg header checksum mismatch (metadata corrupted)".into(),
        ));
    }

    let mut stored_data = [0u8; 32];
    stored_data.copy_from_slice(&b[OFF_DATA_CHECKSUM..OFF_DATA_CHECKSUM + 32]);

    let node_count = u64le(b, 12) as usize;
    let edge_count = u64le(b, 20) as usize;
    let chunk_size = u32le(b, 28) as usize;
    let chunk_count = u32le(b, 32) as usize;
    let max_valid_from = i64le(b, 36);
    let min_valid_to = i64le(b, 44);

    let expected_counts = [
        node_count.saturating_add(1), // offsets
        node_count.saturating_add(1), // chunk_index
        chunk_count,                 // chunk_max_valid_to
        edge_count,                  // dst
        edge_count,                  // valid_from
        edge_count,                  // valid_to
        edge_count,                  // edge_type
    ];
    let expected_elem = [4u8, 4, 8, 8, 8, 8, 2];

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
                return Err(StorageError::Msg(format!(
                    "GTVSeg column {i} has unexpected kind {kind}"
                )));
            }
            if elem_size != expected_elem[i] {
                return Err(StorageError::Msg(format!(
                    "GTVSeg column {i} element size {elem_size}, expected {}",
                    expected_elem[i]
                )));
            }
            if offset < HEADER_SIZE || offset % ALIGN != 0 {
                return Err(StorageError::Msg(format!(
                    "GTVSeg column {i} offset {offset} is not 64 B aligned / past header"
                )));
            }
            if len_bytes % elem_size as usize != 0 {
                return Err(StorageError::Msg(format!(
                    "GTVSeg column {i} length {len_bytes} is not a multiple of {elem_size}"
                )));
            }
            if len_bytes / elem_size as usize != expected_counts[i] {
                return Err(StorageError::Msg(format!(
                    "GTVSeg column {i} has {} elements, expected {}",
                    len_bytes / elem_size as usize,
                    expected_counts[i]
                )));
            }
            let end = offset
                .checked_add(len_bytes)
                .ok_or_else(|| StorageError::Msg(format!("GTVSeg column {i} offsets overflow")))?;
            if end > b.len() {
                return Err(StorageError::Msg(format!(
                    "GTVSeg column {i} ends at {end}, file has {} bytes",
                    b.len()
                )));
            }
        } else if offset != 0 || len_bytes != 0 {
            return Err(StorageError::Msg(format!(
                "GTVSeg reserved column slot {i} is not empty"
            )));
        }
        columns[i] = dir;
    }

    if chunk_size == 0 {
        return Err(StorageError::Msg("GTVSeg chunk_size is zero".into()));
    }

    Ok(SegmentHeader {
        version,
        flags,
        node_count,
        edge_count,
        chunk_size,
        chunk_count,
        max_valid_from,
        min_valid_to,
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
        return Err(StorageError::Msg(format!(
            "GTVSeg column {slot} element size mismatch for mapped type"
        )));
    }
    let n = dir.len_bytes / size;
    MappedCol::new(Arc::clone(region), dir.offset, n)
        .map_err(|e| StorageError::Msg(format!("GTVSeg column {slot}: {e}")))
}

fn build_csr(h: &SegmentHeader, region: &Arc<dyn MappedRegion>) -> Result<TemporalCSR> {
    let c = &h.columns;
    let offsets = mapped_col::<u32>(region, &c[0], 0)?;
    let chunk_index = mapped_col::<u32>(region, &c[1], 1)?;
    let chunk_max_valid_to = mapped_col::<i64>(region, &c[2], 2)?;
    let dst = mapped_col::<u64>(region, &c[3], 3)?;
    let valid_from = mapped_col::<i64>(region, &c[4], 4)?;
    let valid_to = mapped_col::<i64>(region, &c[5], 5)?;
    let edge_type = mapped_col::<u16>(region, &c[6], 6)?;

    TemporalCSR::from_columns(
        h.node_count,
        offsets,
        dst,
        valid_from,
        valid_to,
        edge_type,
        h.max_valid_from,
        h.min_valid_to,
        h.chunk_size,
        chunk_index,
        chunk_max_valid_to,
    )
    .map_err(|e| StorageError::Msg(format!("GTVSeg describes an invalid CSR: {e}")))
}

fn align_up(v: usize, a: usize) -> Result<usize> {
    v.checked_add(a - 1)
        .map(|x| x & !(a - 1))
        .ok_or_else(|| StorageError::Msg("segment offset overflow".into()))
}

fn write_u16_le(dst: &mut [u8], src: &[u16]) {
    for (chunk, v) in dst.chunks_exact_mut(2).zip(src) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
}
fn write_u32_le(dst: &mut [u8], src: &[u32]) {
    for (chunk, v) in dst.chunks_exact_mut(4).zip(src) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
}
fn write_u64_le(dst: &mut [u8], src: &[u64]) {
    for (chunk, v) in dst.chunks_exact_mut(8).zip(src) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
}
fn write_i64_le(dst: &mut [u8], src: &[i64]) {
    for (chunk, v) in dst.chunks_exact_mut(8).zip(src) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
}

/// Serialize any `TemporalCSR` (owned **or** mapped) into GTVSeg v1 bytes.
///
/// Works on the mapped backend too, which makes `open → write → open` a usable
/// copy-on-write / format-upgrade path without materializing an owned CSR.
pub fn encode_segment(csr: &TemporalCSR) -> Result<Vec<u8>> {
    if cfg!(target_endian = "big") {
        return Err(StorageError::Msg(
            "GTVSeg v1 writer requires a little-endian host".into(),
        ));
    }
    let c = csr.columns();

    let specs: [(usize, usize); USED_SLOTS] = [
        (c.offsets.len(), 4),
        (c.chunk_index.len(), 4),
        (c.chunk_max_valid_to.len(), 8),
        (c.dst.len(), 8),
        (c.valid_from.len(), 8),
        (c.valid_to.len(), 8),
        (c.edge_type.len(), 2),
    ];

    let mut cursor = HEADER_SIZE;
    let mut dirs = [ColumnDir::EMPTY; COL_SLOTS];
    for (i, (n, elem)) in specs.iter().enumerate() {
        cursor = align_up(cursor, ALIGN)?;
        let len_bytes = n
            .checked_mul(*elem)
            .ok_or_else(|| StorageError::Msg("column byte length overflow".into()))?;
        dirs[i] = ColumnDir {
            offset: cursor,
            len_bytes,
            elem_size: *elem as u8,
            kind: i as u8,
        };
        cursor = cursor
            .checked_add(len_bytes)
            .ok_or_else(|| StorageError::Msg("segment size overflow".into()))?;
    }
    let total = align_up(cursor, ALIGN)?;
    let mut buf = vec![0u8; total];

    buf[0..8].copy_from_slice(&MAGIC);
    buf[8..10].copy_from_slice(&VERSION.to_le_bytes());
    buf[10..12].copy_from_slice(&0u16.to_le_bytes()); // flags: plain v1
    buf[12..20].copy_from_slice(&(c.node_count as u64).to_le_bytes());
    buf[20..28].copy_from_slice(&(c.dst.len() as u64).to_le_bytes());
    buf[28..32].copy_from_slice(&(c.chunk_size as u32).to_le_bytes());
    buf[32..36].copy_from_slice(&(c.chunk_max_valid_to.len() as u32).to_le_bytes());
    buf[36..44].copy_from_slice(&c.max_valid_from.to_le_bytes());
    buf[44..52].copy_from_slice(&c.min_valid_to.to_le_bytes());

    for (i, d) in dirs.iter().enumerate() {
        let base = OFF_COL_DIR + i * COL_DIR_SIZE;
        buf[base..base + 8].copy_from_slice(&(d.offset as u64).to_le_bytes());
        buf[base + 8..base + 16].copy_from_slice(&(d.len_bytes as u64).to_le_bytes());
        buf[base + 16] = d.elem_size;
        buf[base + 17] = d.kind;
    }

    macro_rules! put {
        ($i:expr, $src:expr, $f:ident) => {{
            let d = &dirs[$i];
            $f(&mut buf[d.offset..d.offset + d.len_bytes], $src);
        }};
    }
    put!(0, c.offsets, write_u32_le);
    put!(1, c.chunk_index, write_u32_le);
    put!(2, c.chunk_max_valid_to, write_i64_le);
    put!(3, c.dst, write_u64_le);
    put!(4, c.valid_from, write_i64_le);
    put!(5, c.valid_to, write_i64_le);
    put!(6, c.edge_type, write_u16_le);

    let header_hash = blake3::hash(&buf[0..OFF_HEADER_CHECKSUM]);
    buf[OFF_HEADER_CHECKSUM..OFF_HEADER_CHECKSUM + 32].copy_from_slice(header_hash.as_bytes());
    let data_hash = blake3::hash(&buf[HEADER_SIZE..]);
    buf[OFF_DATA_CHECKSUM..OFF_DATA_CHECKSUM + 32].copy_from_slice(data_hash.as_bytes());

    Ok(buf)
}

/// Write `csr` to `path` as an immutable GTVSeg, published atomically.
pub fn write_segment(path: impl AsRef<Path>, csr: &TemporalCSR) -> Result<()> {
    let bytes = encode_segment(csr)?;
    write_atomic(path.as_ref(), &bytes)
}
