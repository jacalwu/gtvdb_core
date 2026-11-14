//! `GIVFCMP` — the compound IVF index payload: IVF metadata + a GTVVEC corpus.
//!
//! An [`IvfIndex`] is two very different things glued together:
//!
//! * the **corpus** — the bulk of the payload (`rows × dim` `f32` plus the
//!   `u64` ids), which is exactly what GTVVEC already stores as an immutable,
//!   fixed-width, 64-byte-aligned, mmap-able segment;
//! * the **metadata** — the coarse quantizer (`nlist × dim` centroids), the
//!   cell offsets, the original-position permutation and the training
//!   provenance, which are small, `Vec`-shaped and cheap to decode.
//!
//! `GIVFCMP` keeps the two apart and reuses GTVVEC for the part that matters:
//!
//! ```text
//! 0    magic[8]            "GIVFCMP\0"
//! 8    meta_len   u32      length of the metadata blob (padded to 8 B)
//! 12   version    u16      1
//! 14   reserved  u16
//! 16   meta[meta_len]      IVF metadata (see below)
//! ...  padding to an 8-byte boundary
//! ...  gtvvec[...]         a GTVVEC v1 segment holding ids + vectors
//! ```
//!
//! The GTVVEC segment starts on an 8-byte boundary, so its `u64` id column
//! stays element-aligned in the mapping and can be viewed as a `Col<u64>`
//! without copying a single vector.
//!
//! # Metadata blob
//!
//! Little-endian, in order:
//!
//! ```text
//! dim          u32
//! metric       u8        L2=0 / Cosine=1 / Ip=2
//! nlist        u32
//! nprobe       u32
//! row_count    u64
//! orig_pos     u32[row_count]
//! centroids    f32[nlist * dim]
//! list_offsets u32[nlist + 1]
//! training     u8        0 = absent, 1 = present
//!   (if present) seed u64, restarts u32, iters u32, sample u64, inertia u64
//! ```
//!
//! # Cold-open cost
//!
//! The metadata blob is `O(rows)` bytes (`4·rows` for the permutation) versus
//! the corpus's `O(rows · dim)`. At `dim = 128` that is a 128× smaller
//! materialization, which is what turns an IVF cold open from "decode the
//! whole corpus" into "decode the cell layout, map the corpus".
//!
//! # Compatibility
//!
//! The legacy `GIVFv1` / `GIVFv2` payloads stay readable and are unaffected;
//! `GIVFCMP` is a *new* payload kind that the store dispatches on.

use std::sync::Arc;

use gtv_core::column::{subrange, Col, MappedRegion};
use gtv_core::{GtvError, Metric, Result};

use crate::bytes::{metric_code, metric_from_code, Reader, Writer};
use crate::ivf::{IvfIndex, IvfTrainingMeta};
use crate::vector_segment::{encode_vector_segment, open_vector_segment};

/// Compound IVF payload magic.
pub const MAGIC: [u8; 8] = *b"GIVFCMP\0";
/// Compound IVF payload version.
pub const VERSION: u16 = 1;
/// Fixed compound header length: magic(8) + meta_len(4) + version(2) + reserved(2).
pub const HEADER_SIZE: usize = 16;
/// The GTVVEC segment starts on this boundary (element alignment for `u64`).
pub const ALIGN: usize = 8;

/// The IVF metadata half of a compound payload.
#[derive(Debug, Clone)]
pub struct IvfMeta {
    pub dim: usize,
    pub metric: Metric,
    pub nlist: usize,
    pub nprobe: usize,
    pub row_count: u64,
    pub orig_pos: Vec<u32>,
    pub centroids: Vec<f32>,
    pub list_offsets: Vec<u32>,
    pub training: Option<IvfTrainingMeta>,
}

/// Serialize the IVF metadata half.
fn encode_meta(index: &IvfIndex) -> Vec<u8> {
    let mut w = Writer::with_capacity(
        32 + index.orig_pos().len() * 4
            + index.centroids().len() * 4
            + index.list_offsets().len() * 4,
    );
    w.u32(index.dim() as u32)
        .u8(metric_code(index.metric()))
        .u32(index.nlist() as u32)
        .u32(index.nprobe() as u32)
        .u64(index.len() as u64);
    for p in index.orig_pos() {
        w.u32(*p);
    }
    for c in index.centroids() {
        w.f32(*c);
    }
    for o in index.list_offsets() {
        w.u32(*o);
    }
    match index.training() {
        Some(t) => {
            w.u8(1)
                .u64(t.seed)
                .u32(t.restarts)
                .u32(t.iters)
                .u64(t.sample)
                .u64(t.inertia.to_bits());
        }
        None => {
            w.u8(0);
        }
    }
    w.into_vec()
}

/// Parse the IVF metadata half.
fn decode_meta(bytes: &[u8]) -> Result<IvfMeta> {
    let mut r = Reader::new(bytes);
    let dim = r.u32()? as usize;
    let metric = metric_from_code(r.u8()?)?;
    let nlist = r.u32()? as usize;
    let nprobe = r.u32()? as usize;
    let row_count = r.u64()?;
    let n = row_count as usize;

    let mut orig_pos = Vec::with_capacity(n);
    for _ in 0..n {
        orig_pos.push(r.u32()?);
    }
    let mut centroids = Vec::with_capacity(nlist * dim);
    for _ in 0..nlist * dim {
        centroids.push(r.f32()?);
    }
    let mut list_offsets = Vec::with_capacity(nlist + 1);
    for _ in 0..nlist + 1 {
        list_offsets.push(r.u32()?);
    }
    let training = if r.u8()? != 0 {
        Some(IvfTrainingMeta {
            kmeans: true,
            seed: r.u64()?,
            restarts: r.u32()?,
            iters: r.u32()?,
            sample: r.u64()?,
            inertia: f64::from_bits(r.u64()?),
        })
    } else {
        None
    };
    Ok(IvfMeta {
        dim,
        metric,
        nlist,
        nprobe,
        row_count,
        orig_pos,
        centroids,
        list_offsets,
        training,
    })
}

/// Round `n` up to the next multiple of [`ALIGN`].
fn align_up(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

/// Encode an IVF index as a compound payload: metadata + a GTVVEC corpus.
///
/// The corpus is the *cell-ordered* corpus (the order the index scans in), so
/// reopening the payload reproduces the index's probe layout exactly.
pub fn encode_ivf_payload(index: &IvfIndex) -> Result<Vec<u8>> {
    let meta = encode_meta(index);
    let meta_len = align_up(meta.len());
    let gtvvec = encode_vector_segment(
        index.ids(),
        index.data(),
        index.dim(),
        index.metric(),
        index.is_normalized(),
    )?;

    let mut out = Vec::with_capacity(HEADER_SIZE + meta_len + gtvvec.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&(meta_len as u32).to_le_bytes());
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&meta);
    out.resize(HEADER_SIZE + meta_len, 0);
    out.extend_from_slice(&gtvvec);
    Ok(out)
}

/// Decode a compound IVF payload from bytes (the materializing path).
///
/// This exists for symmetry with [`IvfIndex::from_bytes`] and for tests; the
/// store's cold-open path uses [`open_ivf_payload`] instead, which keeps the
/// corpus mapped instead of copying it out.
pub fn decode_ivf_payload(bytes: &[u8]) -> Result<IvfIndex> {
    let (meta, gtvvec) = split_payload(bytes)?;
    let meta = decode_meta(meta)?;
    let (ids, data) = read_corpus(gtvvec)?;
    build_index(meta, ids, data)
}

/// Open a compound IVF payload from a mapped region (the O(1) cold-open path).
///
/// The corpus columns are `Col::Mapped` views into the mapping — no vector is
/// copied out of the page cache.
pub fn open_ivf_payload(region: Arc<dyn MappedRegion>) -> Result<IvfIndex> {
    let bytes = region.as_bytes();
    let (meta_bytes, gtvvec_offset) = split_payload_at(bytes)?;
    let meta = decode_meta(meta_bytes)?;
    let sub = subrange(region, gtvvec_offset);
    let segment = open_vector_segment(sub)?;
    let (ids, data) = segment.into_columns();
    build_index(meta, ids, data)
}

/// Run the O(file size) data-integrity check over a compound payload's corpus.
///
/// The compound header is validated by the open path; this checks the GTVVEC
/// segment's `data_checksum`, which the cold open deliberately skips.
pub fn verify_ivf_payload_data(region: Arc<dyn MappedRegion>) -> Result<()> {
    let bytes = region.as_bytes();
    let (_meta_bytes, gtvvec_offset) = split_payload_at(bytes)?;
    let sub = subrange(region, gtvvec_offset);
    let segment = open_vector_segment(sub)?;
    segment.verify_data()
}

/// Read the corpus half of a payload into owned columns.
fn read_corpus(gtvvec: &[u8]) -> Result<(Col<u64>, Col<f32>)> {
    let segment = open_vector_segment(Arc::new(BytesRegion(gtvvec.to_vec())))?;
    Ok(segment.into_columns())
}

/// Assemble the index from decoded metadata and corpus columns.
fn build_index(meta: IvfMeta, ids: Col<u64>, data: Col<f32>) -> Result<IvfIndex> {
    IvfIndex::from_columns(
        data,
        ids,
        meta.orig_pos,
        meta.dim,
        meta.metric,
        meta.nlist,
        meta.nprobe,
        meta.centroids,
        meta.list_offsets,
        meta.training,
    )
}

/// Split a payload into its metadata slice and its GTVVEC slice.
fn split_payload(bytes: &[u8]) -> Result<(&[u8], &[u8])> {
    let (meta, offset) = split_payload_at(bytes)?;
    Ok((meta, &bytes[offset..]))
}

/// Parse the compound header and return `(metadata, gtvvec_offset)`.
fn split_payload_at(bytes: &[u8]) -> Result<(&[u8], usize)> {
    if bytes.len() < HEADER_SIZE {
        return Err(GtvError::InvalidArgument(
            "GIVFCMP payload shorter than its header".into(),
        ));
    }
    if &bytes[0..8] != &MAGIC[..] {
        return Err(GtvError::InvalidArgument(
            "GIVFCMP bad magic (not a compound IVF payload)".into(),
        ));
    }
    let version = u16::from_le_bytes(bytes[12..14].try_into().unwrap());
    if version != VERSION {
        return Err(GtvError::InvalidArgument(format!(
            "unsupported GIVFCMP version {version} (reader supports {VERSION})"
        )));
    }
    let meta_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if meta_len % ALIGN != 0 {
        return Err(GtvError::InvalidArgument(
            "GIVFCMP metadata length is not 8-byte aligned".into(),
        ));
    }
    let meta_end = HEADER_SIZE
        .checked_add(meta_len)
        .ok_or_else(|| GtvError::InvalidArgument("GIVFCMP metadata length overflow".into()))?;
    if meta_end > bytes.len() {
        return Err(GtvError::InvalidArgument(
            "GIVFCMP metadata exceeds payload".into(),
        ));
    }
    Ok((&bytes[HEADER_SIZE..meta_end], meta_end))
}

/// An owned byte region, used only by the materializing decode path.
#[derive(Debug)]
struct BytesRegion(Vec<u8>);

impl MappedRegion for BytesRegion {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ivf::IvfIndex;

    fn corpus(n: usize, dim: usize) -> (Vec<u64>, Vec<f32>) {
        let ids: Vec<u64> = (0..n as u64).collect();
        let data: Vec<f32> = (0..n * dim).map(|i| (i % 97) as f32 * 0.01).collect();
        (ids, data)
    }

    fn ivf(n: usize, dim: usize) -> IvfIndex {
        let (ids, data) = corpus(n, dim);
        IvfIndex::new(ids, data, dim, 4, 2).unwrap()
    }

    #[test]
    fn compound_payload_round_trips_through_bytes() {
        let index = ivf(256, 4);
        let payload = encode_ivf_payload(&index).unwrap();
        assert_eq!(&payload[0..8], &MAGIC[..]);

        let back = decode_ivf_payload(&payload).unwrap();
        assert_eq!(back.len(), index.len());
        assert_eq!(back.dim(), index.dim());
        assert_eq!(back.nlist(), index.nlist());
        assert_eq!(back.nprobe(), index.nprobe());
        assert_eq!(back.ids(), index.ids());
        assert_eq!(back.data(), index.data());
        assert_eq!(back.centroids(), index.centroids());
        assert_eq!(back.list_offsets(), index.list_offsets());
        assert_eq!(back.orig_pos(), index.orig_pos());
    }

    #[test]
    fn compound_payload_round_trips_through_a_mapped_region() {
        let index = ivf(512, 8);
        let payload = encode_ivf_payload(&index).unwrap();
        let region: Arc<dyn MappedRegion> = Arc::new(BytesRegion(payload));
        let back = open_ivf_payload(region).unwrap();
        assert_eq!(back.ids(), index.ids());
        assert_eq!(back.data(), index.data());
        assert_eq!(back.list_offsets(), index.list_offsets());
    }

    #[test]
    fn compound_payload_preserves_training_metadata() {
        let (ids, data) = corpus(300, 4);
        let index = IvfIndex::new(ids, data, 4, 4, 2).unwrap();
        let payload = encode_ivf_payload(&index).unwrap();
        let back = decode_ivf_payload(&payload).unwrap();
        assert_eq!(back.training(), index.training());
    }

    #[test]
    fn compound_payload_rejects_a_bad_magic() {
        let mut payload = encode_ivf_payload(&ivf(32, 2)).unwrap();
        payload[0] = b'X';
        let err = decode_ivf_payload(&payload).unwrap_err();
        assert!(matches!(err, GtvError::InvalidArgument(_)));
    }

    #[test]
    fn compound_payload_rejects_a_truncated_body() {
        let payload = encode_ivf_payload(&ivf(32, 2)).unwrap();
        let err = decode_ivf_payload(&payload[..payload.len() / 2]).unwrap_err();
        assert!(matches!(err, GtvError::InvalidArgument(_)));
    }

    #[test]
    fn mapped_cold_open_does_not_copy_the_corpus() {
        // The mapped path must hand back columns that point into the mapping,
        // not fresh allocations: a copy would change the address of the data.
        let index = ivf(128, 4);
        let payload = encode_ivf_payload(&index).unwrap();
        let region: Arc<dyn MappedRegion> = Arc::new(BytesRegion(payload));
        let base = region.as_bytes().as_ptr() as usize;
        let back = open_ivf_payload(region).unwrap();
        let data_ptr = back.data().as_ptr() as usize;
        assert!(
            data_ptr > base && data_ptr < base + 4096 * 4,
            "mapped corpus column must point into the mapping"
        );
    }
}
