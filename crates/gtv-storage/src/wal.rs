//! Append-only write-ahead log for the P1 ingest path.
//!
//! # Why a WAL at all
//!
//! GTVSeg files are immutable: an mmap that is being read while its backing
//! file changes would observe torn data. So ingest cannot append to a segment.
//! It goes to a durable WAL first and to an in-memory `MemSegment` second; a
//! flush publishes a **new** segment via `write_atomic` and only then advances
//! the manifest checkpoint and truncates the WAL.
//!
//! # Layout (all little-endian)
//!
//! ```text
//! 0        WAL header (64 B): magic / version / record sizes / blake3
//! 64       frame 0, frame 1, ...                 (append-only)
//!
//! frame = header[24] + payload[n * 42] + trailer[32]
//!   header:  txn_id u64 | op u8 | table_id u32 | partition_id u32
//!            | payload_len u32 | reserved[3]
//!   payload: n fixed-width EdgeRecords (no varint, no dictionary, no zstd)
//!   trailer: blake3(header || payload) — detects a torn tail after kill -9
//!
//! EdgeRecord[42] = src u64 | dst u64 | valid_from i64 | valid_to i64
//!                | edge_type u16 | system_ts i64
//! ```
//!
//! **Zero encoding / zero compression on the write path**: fixed-width native
//! integers only. The one `to_le_bytes()` per field is a byte-order conversion,
//! not the RLE/dictionary/compression pipeline OceanBase keeps out of its
//! ingest path too. Compressed Parquet belongs to the L2 archive only.
//!
//! # Crash semantics
//!
//! Frames are checksummed. On open, scanning stops at the first invalid or
//! incomplete frame and the file is truncated at that exact offset — a kill -9
//! during `write_all` can only damage the *tail*, because the log is
//! append-only. Everything before the tail is replayed. Frames whose
//! `txn_id <= manifest.checkpoint_txn` are skipped by the store: they are
//! already covered by a published GTVSeg, so replay is idempotent even if the
//! WAL was not truncated before the crash.

use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::atomic::write_atomic;
use crate::error::{Result, StorageError};

pub const WAL_MAGIC: [u8; 8] = *b"GTVWAL\0\0";
pub const WAL_VERSION: u16 = 1;
pub const WAL_HEADER_SIZE: usize = 64;
pub const FRAME_HEADER_SIZE: usize = 24;
pub const FRAME_TRAILER_SIZE: usize = 32;
pub const EDGE_RECORD_SIZE: usize = 42;
pub const OP_INSERT: u8 = 1;
/// Guard against a corrupt length field asking for a huge allocation.
const MAX_FRAME_PAYLOAD: usize = 64 * 1024 * 1024;

fn err(msg: impl Into<String>) -> StorageError {
    StorageError::Msg(msg.into())
}

fn read_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap())
}
fn read_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn read_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}
fn read_i64(b: &[u8], at: usize) -> i64 {
    i64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// One ingested temporal edge plus its bitemporal system-time stamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeRecord {
    pub src: u64,
    pub dst: u64,
    pub valid_from: i64,
    pub valid_to: i64,
    pub edge_type: u16,
    /// Transaction/system time at which this edge became known.
    pub system_ts: i64,
}

impl EdgeRecord {
    pub fn new(
        src: u64,
        dst: u64,
        valid_from: i64,
        valid_to: i64,
        edge_type: u16,
        system_ts: i64,
    ) -> Self {
        Self {
            src,
            dst,
            valid_from,
            valid_to,
            edge_type,
            system_ts,
        }
    }

    fn write_le(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.src.to_le_bytes());
        out.extend_from_slice(&self.dst.to_le_bytes());
        out.extend_from_slice(&self.valid_from.to_le_bytes());
        out.extend_from_slice(&self.valid_to.to_le_bytes());
        out.extend_from_slice(&self.edge_type.to_le_bytes());
        out.extend_from_slice(&self.system_ts.to_le_bytes());
    }

    /// Caller guarantees `b.len() >= EDGE_RECORD_SIZE`.
    fn read_le(b: &[u8]) -> Self {
        Self {
            src: read_u64(b, 0),
            dst: read_u64(b, 8),
            valid_from: read_i64(b, 16),
            valid_to: read_i64(b, 24),
            edge_type: read_u16(b, 32),
            system_ts: read_i64(b, 34),
        }
    }
}

/// A single durable ingest transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalFrame {
    pub txn_id: u64,
    pub op: u8,
    pub table_id: u32,
    pub partition_id: u32,
    pub edges: Vec<EdgeRecord>,
}

impl WalFrame {
    pub fn insert(
        txn_id: u64,
        table_id: u32,
        partition_id: u32,
        edges: Vec<EdgeRecord>,
    ) -> Self {
        Self {
            txn_id,
            op: OP_INSERT,
            table_id,
            partition_id,
            edges,
        }
    }

    fn encode(&self) -> Result<Vec<u8>> {
        if self.op != OP_INSERT {
            return Err(err(format!("WAL op {} is not supported", self.op)));
        }
        let payload_len = self
            .edges
            .len()
            .checked_mul(EDGE_RECORD_SIZE)
            .ok_or_else(|| err("WAL frame payload length overflow"))?;
        if payload_len > MAX_FRAME_PAYLOAD {
            return Err(err(format!(
                "WAL frame payload {payload_len} B exceeds the {MAX_FRAME_PAYLOAD} B limit; split the batch"
            )));
        }

        let mut buf = Vec::with_capacity(FRAME_HEADER_SIZE + payload_len + FRAME_TRAILER_SIZE);
        buf.extend_from_slice(&self.txn_id.to_le_bytes());
        buf.push(self.op);
        buf.extend_from_slice(&self.table_id.to_le_bytes());
        buf.extend_from_slice(&self.partition_id.to_le_bytes());
        buf.extend_from_slice(&(payload_len as u32).to_le_bytes());
        buf.extend_from_slice(&[0u8; 3]); // reserved, header is 24 B
        debug_assert_eq!(buf.len(), FRAME_HEADER_SIZE);
        for e in &self.edges {
            e.write_le(&mut buf);
        }
        let checksum = blake3::hash(&buf);
        buf.extend_from_slice(checksum.as_bytes());
        Ok(buf)
    }
}

/// Result of scanning a WAL on (re)open.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WalScan {
    /// Every intact frame, in file order.
    pub frames: Vec<WalFrame>,
    /// Byte offset of the first damaged/incomplete frame (also the new EOF).
    pub valid_len: usize,
    /// Bytes discarded from a torn tail (0 on a clean shutdown).
    pub torn_bytes: usize,
    /// Highest `txn_id` seen.
    pub last_txn: u64,
}

fn fresh_header() -> Vec<u8> {
    let mut h = vec![0u8; WAL_HEADER_SIZE];
    h[0..8].copy_from_slice(&WAL_MAGIC);
    h[8..10].copy_from_slice(&WAL_VERSION.to_le_bytes());
    h[10..12].copy_from_slice(&(EDGE_RECORD_SIZE as u16).to_le_bytes());
    h[12..14].copy_from_slice(&(FRAME_HEADER_SIZE as u16).to_le_bytes());
    h[14..16].copy_from_slice(&(FRAME_TRAILER_SIZE as u16).to_le_bytes());
    // [16..32) reserved, [32..64) checksum over the first 32 bytes.
    let checksum = blake3::hash(&h[0..32]);
    h[32..64].copy_from_slice(checksum.as_bytes());
    h
}

fn validate_header(b: &[u8]) -> Result<()> {
    if b.len() < WAL_HEADER_SIZE {
        return Err(err("WAL shorter than its 64-byte header"));
    }
    if b[0..8] != WAL_MAGIC {
        return Err(err("WAL bad magic (not a GTV WAL, or truncated)"));
    }
    let version = read_u16(b, 8);
    if version != WAL_VERSION {
        return Err(err(format!(
            "unsupported WAL version {version} (reader supports {WAL_VERSION})"
        )));
    }
    if read_u16(b, 10) as usize != EDGE_RECORD_SIZE
        || read_u16(b, 12) as usize != FRAME_HEADER_SIZE
        || read_u16(b, 14) as usize != FRAME_TRAILER_SIZE
    {
        return Err(err("WAL header record sizes do not match this reader"));
    }
    if blake3::hash(&b[0..32]).as_bytes() != &b[32..64] {
        return Err(err("WAL header checksum mismatch"));
    }
    Ok(())
}

/// Scan frames from `b`. The first invalid frame marks a torn tail; everything
/// after it is discarded (the log is append-only, so no valid frame can follow
/// a damaged one).
fn scan_frames(b: &[u8]) -> Result<WalScan> {
    let mut at = WAL_HEADER_SIZE;
    let mut frames = Vec::new();
    let mut last_txn = 0u64;

    while at < b.len() {
        let remaining = b.len() - at;
        if remaining < FRAME_HEADER_SIZE + FRAME_TRAILER_SIZE {
            break;
        }
        let payload_len = read_u32(b, at + 17) as usize;
        if payload_len > MAX_FRAME_PAYLOAD || payload_len % EDGE_RECORD_SIZE != 0 {
            break;
        }
        let frame_len = match FRAME_HEADER_SIZE
            .checked_add(payload_len)
            .and_then(|x| x.checked_add(FRAME_TRAILER_SIZE))
        {
            Some(v) => v,
            None => break,
        };
        if frame_len > remaining {
            break;
        }

        let body_end = at + FRAME_HEADER_SIZE + payload_len;
        let expected = &b[body_end..at + frame_len];
        if blake3::hash(&b[at..body_end]).as_bytes() != expected {
            break;
        }
        let op = b[at + 8];
        if op != OP_INSERT {
            return Err(err(format!(
                "WAL frame at {at} has valid checksum but unknown op {op}"
            )));
        }

        let n = payload_len / EDGE_RECORD_SIZE;
        let mut edges = Vec::with_capacity(n);
        for i in 0..n {
            let off = at + FRAME_HEADER_SIZE + i * EDGE_RECORD_SIZE;
            edges.push(EdgeRecord::read_le(&b[off..off + EDGE_RECORD_SIZE]));
        }
        let txn_id = read_u64(b, at);
        last_txn = last_txn.max(txn_id);
        frames.push(WalFrame {
            txn_id,
            op,
            table_id: read_u32(b, at + 9),
            partition_id: read_u32(b, at + 13),
            edges,
        });
        at += frame_len;
    }

    Ok(WalScan {
        frames,
        valid_len: at,
        torn_bytes: b.len() - at,
        last_txn,
    })
}

/// An opened WAL, positioned at EOF for appends.
#[derive(Debug)]
pub struct Wal {
    #[allow(dead_code)]
    path: PathBuf,
    file: File,
    valid_len: u64,
}

impl Wal {
    /// Open (creating if needed), validate the header, scan frames and truncate
    /// a torn tail so subsequent appends land at a clean frame boundary.
    pub fn open(path: impl AsRef<Path>) -> Result<(Self, WalScan)> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let need_header = match fs::metadata(&path) {
            Ok(m) => m.len() < WAL_HEADER_SIZE as u64,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
            Err(e) => return Err(e.into()),
        };
        if need_header {
            write_atomic(&path, &fresh_header())?;
        }

        let bytes = fs::read(&path)?;
        validate_header(&bytes)?;
        let scan = scan_frames(&bytes)?;

        // `write(true)` (not `append(true)`) is required on Windows: an
        // append-only handle lacks the generic write access that `set_len`
        // needs for torn-tail truncation and checkpointing. We always seek to
        // EOF before a write, so the semantics are still append-only.
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
        if scan.torn_bytes > 0 {
            file.set_len(scan.valid_len as u64)?;
            file.sync_all()?;
        }
        file.seek(SeekFrom::End(0))?;

        Ok((
            Self {
                path,
                file,
                valid_len: scan.valid_len as u64,
            },
            scan,
        ))
    }

    /// Append one frame and fsync it. Returns only after the bytes are durable.
    pub fn append(&mut self, frame: &WalFrame) -> Result<()> {
        let buf = frame.encode()?;
        self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(&buf)?;
        self.file.sync_all()?;
        self.valid_len += buf.len() as u64;
        Ok(())
    }

    /// Drop all frames after a successful flush. Called **after** the manifest
    /// checkpoint has been durably published, so a crash here cannot lose data.
    pub fn checkpoint(&mut self) -> Result<usize> {
        let reclaimed = self.valid_len.saturating_sub(WAL_HEADER_SIZE as u64) as usize;
        self.file.set_len(WAL_HEADER_SIZE as u64)?;
        self.file.seek(SeekFrom::End(0))?;
        self.file.sync_all()?;
        self.valid_len = WAL_HEADER_SIZE as u64;
        Ok(reclaimed)
    }

    pub fn len(&self) -> u64 {
        self.valid_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_record_roundtrip_is_fixed_width() {
        let e = EdgeRecord::new(7, 9, 100, 200, 3, 42);
        let frame = WalFrame::insert(1, 2, 3, vec![e]);
        let bytes = frame.encode().unwrap();
        assert_eq!(
            bytes.len(),
            FRAME_HEADER_SIZE + EDGE_RECORD_SIZE + FRAME_TRAILER_SIZE
        );
        let mut with_header = fresh_header();
        with_header.extend_from_slice(&bytes);
        let scan = scan_frames(&with_header).unwrap();
        assert_eq!(scan.frames.len(), 1);
        assert_eq!(scan.frames[0], frame);
        assert_eq!(scan.torn_bytes, 0);
    }

    #[test]
    fn torn_tail_stops_scan_at_last_intact_frame() {
        let e = EdgeRecord::new(1, 2, 10, 20, 1, 1);
        let good = WalFrame::insert(1, 0, 0, vec![e]).encode().unwrap();
        let mut buf = fresh_header();
        buf.extend_from_slice(&good);
        let intact = buf.len();
        buf.extend_from_slice(b"half-written-frame");
        let scan = scan_frames(&buf).unwrap();
        assert_eq!(scan.frames.len(), 1);
        assert_eq!(scan.valid_len, intact);
        assert_eq!(scan.torn_bytes, buf.len() - intact);
    }
}
