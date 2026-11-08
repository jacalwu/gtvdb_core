//! GTVSeg round-trip, backend-equivalence and corruption tests.
//!
//! The central guarantee under test is not "open works" but "an mmap-backed
//! TemporalCSR is indistinguishable from an owned one across every operator".
//! That is the precondition for claiming zero-copy loading does not change
//! query semantics.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use arrow::array::{TimestampNanosecondArray, UInt16Array, UInt64Array};
use gtv_core::csr::TemporalCSR;
use gtv_storage::segment::{encode_segment, open_segment, write_segment, Segment};

static SEQ: AtomicU64 = AtomicU64::new(0);

fn temp_path(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "gtvseg_{}_{}_{}.seg",
        tag,
        std::process::id(),
        n
    ))
}

/// A graph with multiple time windows, expired edges, a zero-degree node and
/// a high-degree source, so both the binary-search and zone-map paths run.
fn sample_csr() -> TemporalCSR {
    TemporalCSR::from_arrays(
        &UInt64Array::from(vec![0u64, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5]),
        &UInt64Array::from(vec![1u64, 2, 3, 2, 3, 3, 4, 5, 4, 5, 0, 0]),
        &TimestampNanosecondArray::from(vec![
            0i64, 50, 10, 0, 100, 5, 20, 200, 0, 30, 60, 0,
        ]),
        &TimestampNanosecondArray::from(vec![
            100i64, 200, 80, 100, 300, 90, 150, 400, 300, 120, 180, 50,
        ]),
        &UInt16Array::from(vec![1u16, 1, 2, 2, 1, 3, 1, 2, 3, 1, 2, 1]),
        6,
    )
    .unwrap()
}

fn assert_columns_equal(a: &TemporalCSR, b: &TemporalCSR) {
    let x = a.columns();
    let y = b.columns();
    assert_eq!(x.node_count, y.node_count);
    assert_eq!(x.offsets, y.offsets, "offsets differ");
    assert_eq!(x.dst, y.dst, "dst differ");
    assert_eq!(x.valid_from, y.valid_from, "valid_from differ");
    assert_eq!(x.valid_to, y.valid_to, "valid_to differ");
    assert_eq!(x.edge_type, y.edge_type, "edge_type differ");
    assert_eq!(x.max_valid_from, y.max_valid_from);
    assert_eq!(x.min_valid_to, y.min_valid_to);
    assert_eq!(x.chunk_size, y.chunk_size);
    assert_eq!(x.chunk_index, y.chunk_index, "chunk_index differ");
    assert_eq!(
        x.chunk_max_valid_to, y.chunk_max_valid_to,
        "chunk_max_valid_to differ"
    );
}

fn reseal_header(bytes: &mut [u8]) {
    let h = blake3::hash(&bytes[0..256]);
    bytes[256..288].copy_from_slice(h.as_bytes());
}

fn write_raw(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
}

#[test]
fn roundtrip_preserves_every_column_byte_for_byte() {
    let owned = sample_csr();
    let path = temp_path("roundtrip");
    write_segment(&path, &owned).unwrap();

    let seg = Segment::open(&path).unwrap();
    assert_columns_equal(&owned, seg.csr());
    seg.verify_data().unwrap();

    assert_eq!(seg.header().node_count, 6);
    assert_eq!(seg.header().edge_count, owned.edge_count());
    assert_eq!(seg.header().chunk_count, owned.chunk_count());
    assert_eq!(seg.header().version, 1);
    assert_eq!(seg.header().flags, 0);
    // Every column starts on a cache-line boundary.
    for d in &seg.header().columns[..7] {
        assert_eq!(d.offset % 64, 0, "column {:?} is not 64 B aligned", d);
    }

    drop(seg);
    fs::remove_file(&path).ok();
}

#[test]
fn mmap_backend_matches_owned_on_all_graph_operators() {
    let owned = sample_csr();
    let path = temp_path("operators");
    write_segment(&path, &owned).unwrap();
    let seg = open_segment(&path).unwrap();
    let mapped = seg.csr();

    // Scalar metadata
    assert_eq!(mapped.node_count(), owned.node_count());
    assert_eq!(mapped.edge_count(), owned.edge_count());
    assert_eq!(mapped.chunk_count(), owned.chunk_count());
    assert_eq!(mapped.max_degree(), owned.max_degree());
    assert_eq!(mapped.degree_histogram(), owned.degree_histogram());
    let ms = mapped.stats();
    let os = owned.stats();
    assert_eq!(ms.node_count, os.node_count);
    assert_eq!(ms.edge_count, os.edge_count);
    assert_eq!(ms.chunk_count, os.chunk_count);
    assert_eq!(ms.max_degree, os.max_degree);
    assert_eq!(ms.degree_histogram, os.degree_histogram);

    for t in [0i64, 10, 49, 50, 60, 79, 100, 149, 200, 299, 399, 500] {
        assert_eq!(mapped.all_active_at(t), owned.all_active_at(t), "t={t}");
        for src in 0..6u64 {
            assert_eq!(mapped.degree(src).unwrap(), owned.degree(src).unwrap());
            let (d0, f0, t0, e0) = mapped.edge_slices(src).unwrap();
            let (d1, f1, t1, e1) = owned.edge_slices(src).unwrap();
            assert_eq!(d0, d1, "edge_slices(dst) src={src}");
            assert_eq!(f0, f1, "edge_slices(valid_from) src={src}");
            assert_eq!(t0, t1, "edge_slices(valid_to) src={src}");
            assert_eq!(e0, e1, "edge_slices(edge_type) src={src}");

            let a: Vec<_> = mapped.neighbors(src, t).unwrap().collect();
            let b: Vec<_> = owned.neighbors(src, t).unwrap().collect();
            assert_eq!(a, b, "neighbors differ: src={src} t={t}");
        }
    }

    // transpose() rebuilds an owned CSR from whatever backend it is called on;
    // both must produce identical arrays.
    assert_columns_equal(&owned.transpose().unwrap(), &mapped.transpose().unwrap());

    drop(seg);
    fs::remove_file(&path).ok();
}

#[test]
fn mapped_csr_can_be_rewritten_as_a_new_segment() {
    // open -> encode -> open is the copy-on-write / format-upgrade path; it
    // must work without ever materialising an owned CSR.
    let owned = sample_csr();
    let p1 = temp_path("cow1");
    let p2 = temp_path("cow2");
    write_segment(&p1, &owned).unwrap();

    let seg1 = open_segment(&p1).unwrap();
    let bytes = encode_segment(seg1.csr()).unwrap();
    write_raw(&p2, &bytes);
    let seg2 = open_segment(&p2).unwrap();

    assert_columns_equal(&owned, seg2.csr());
    assert_columns_equal(seg1.csr(), seg2.csr());
    seg2.verify_data().unwrap();

    drop(seg1);
    drop(seg2);
    fs::remove_file(&p1).ok();
    fs::remove_file(&p2).ok();
}

#[test]
fn data_corruption_is_not_checked_on_open_but_is_detected_by_verify_data() {
    // Deliberate: hashing the whole file on open would make cold-open O(E).
    let owned = sample_csr();
    let path = temp_path("datacorrupt");
    let mut bytes = encode_segment(&owned).unwrap();
    let at = 4096 + 8; // inside the offsets column
    bytes[at] ^= 0x01;
    write_raw(&path, &bytes);

    let seg = open_segment(&path).unwrap(); // header checksum still valid
    assert!(matches!(seg.verify_data(), Err(_)), "verify_data must fail");
    let err = seg.verify_data().unwrap_err().to_string();
    assert!(err.contains("checksum mismatch"), "unexpected error: {err}");

    drop(seg);
    fs::remove_file(&path).ok();
}

#[test]
fn header_corruption_is_rejected_on_open() {
    let owned = sample_csr();
    let path = temp_path("headercorrupt");
    let mut bytes = encode_segment(&owned).unwrap();
    bytes[12] ^= 0x01; // node_count
    write_raw(&path, &bytes);

    let err = open_segment(&path).unwrap_err().to_string();
    assert!(
        err.contains("header checksum mismatch"),
        "unexpected error: {err}"
    );
    fs::remove_file(&path).ok();
}

#[test]
fn bad_magic_and_unsupported_version_are_rejected() {
    let owned = sample_csr();
    let bytes = encode_segment(&owned).unwrap();

    let p = temp_path("magic");
    let mut b = bytes.clone();
    b[0] = b'X';
    write_raw(&p, &b);
    assert!(open_segment(&p).unwrap_err().to_string().contains("bad magic"));
    fs::remove_file(&p).ok();

    let p = temp_path("version");
    let mut b = bytes.clone();
    b[8..10].copy_from_slice(&99u16.to_le_bytes());
    reseal_header(&mut b); // isolate the version check from the checksum check
    write_raw(&p, &b);
    assert!(open_segment(&p)
        .unwrap_err()
        .to_string()
        .contains("unsupported GTVSeg version"));
    fs::remove_file(&p).ok();
}

#[test]
fn truncated_and_misaligned_files_are_rejected() {
    let owned = sample_csr();
    let bytes = encode_segment(&owned).unwrap();

    let p = temp_path("short");
    write_raw(&p, &bytes[..1024]);
    assert!(open_segment(&p)
        .unwrap_err()
        .to_string()
        .contains("too short"));
    fs::remove_file(&p).ok();

    // Move column 0 one byte off its 64 B boundary, then re-seal the header
    // so the semantic alignment check (not the checksum) is what fires.
    let p = temp_path("misaligned");
    let mut b = bytes.clone();
    let off_at = 64usize; // col_dir[0].offset
    let mut cur = u64::from_le_bytes(b[off_at..off_at + 8].try_into().unwrap());
    cur += 1;
    b[off_at..off_at + 8].copy_from_slice(&cur.to_le_bytes());
    reseal_header(&mut b);
    write_raw(&p, &b);
    let err = open_segment(&p).unwrap_err().to_string();
    assert!(err.contains("64 B aligned"), "unexpected error: {err}");
    fs::remove_file(&p).ok();
}

#[test]
fn column_length_mismatch_is_rejected() {
    let owned = sample_csr();
    let path = temp_path("badlen");
    let mut b = encode_segment(&owned).unwrap();
    let len_at = 64usize + 8; // col_dir[0].len_bytes
    let mut cur = u64::from_le_bytes(b[len_at..len_at + 8].try_into().unwrap());
    cur += 4;
    b[len_at..len_at + 8].copy_from_slice(&cur.to_le_bytes());
    reseal_header(&mut b);
    write_raw(&path, &b);

    let err = open_segment(&path).unwrap_err().to_string();
    assert!(
        err.contains("expected") || err.contains("multiple"),
        "unexpected error: {err}"
    );
    fs::remove_file(&path).ok();
}

#[test]
fn p2_flags_are_refused_by_the_v1_reader() {
    let owned = sample_csr();
    let path = temp_path("p2flags");
    let mut b = encode_segment(&owned).unwrap();
    b[10..12].copy_from_slice(&1u16.to_le_bytes()); // FLAG_NARROW_DST
    reseal_header(&mut b);
    write_raw(&path, &b);

    let err = open_segment(&path).unwrap_err().to_string();
    assert!(err.contains("P2 encoding"), "unexpected error: {err}");
    fs::remove_file(&path).ok();
}

#[test]
fn empty_graph_roundtrips() {
    // Zero edges / zero chunks exercises the `len == 0` dangling-pointer path
    // in MappedCol::new.
    let empty = TemporalCSR::from_arrays(
        &UInt64Array::from(Vec::<u64>::new()),
        &UInt64Array::from(Vec::<u64>::new()),
        &TimestampNanosecondArray::from(Vec::<i64>::new()),
        &TimestampNanosecondArray::from(Vec::<i64>::new()),
        &UInt16Array::from(Vec::<u16>::new()),
        3,
    )
    .unwrap();
    let path = temp_path("empty");
    write_segment(&path, &empty).unwrap();
    let seg = open_segment(&path).unwrap();
    assert_columns_equal(&empty, seg.csr());
    assert_eq!(seg.csr().edge_count(), 0);
    seg.verify_data().unwrap();

    drop(seg);
    fs::remove_file(&path).ok();
}
