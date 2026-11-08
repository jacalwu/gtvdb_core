//! End-to-end SQL tests for GTVSeg registration.

use std::fs;
use std::path::PathBuf;

use arrow::array::{Int64Array, StringArray, UInt16Array, UInt64Array};
use arrow::array::{TimestampNanosecondArray, UInt64Array as U64};
use gtv_engine::GtvContext;
use gtv_storage::segment::write_segment;

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "gtvseg_sql_{}_{}_{}.seg",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn sample_csr() -> gtv_core::TemporalCSR {
    gtv_core::TemporalCSR::from_arrays(
        &U64::from(vec![0u64, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5]),
        &U64::from(vec![1u64, 2, 3, 2, 3, 3, 4, 5, 4, 5, 0, 0]),
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

#[tokio::test]
async fn register_segment_exposes_it_in_sql_without_materialising() {
    let owned = sample_csr();
    let path = temp_path("register");
    write_segment(&path, &owned).unwrap();

    let ctx = GtvContext::new();
    ctx.register_segment(path.to_str().unwrap(), "g").unwrap();

    // The registry view: one row, mmap backend, correct counts.
    let batches = ctx
        .sql("SELECT name, backend, node_count, edge_count, chunk_count FROM gtv_segments()")
        .await
        .unwrap();
    let b = &batches[0];
    assert_eq!(b.num_rows(), 1);
    assert_eq!(
        b.column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "g"
    );
    assert_eq!(
        b.column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "mmap"
    );
    assert_eq!(
        b.column(2)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0),
        6
    );
    assert_eq!(
        b.column(3)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0),
        owned.edge_count() as u64
    );

    // Rust-side access still goes through the zero-copy CSR.
    let seg = ctx.segment("g").expect("segment registered");
    assert_eq!(seg.csr().edge_count(), owned.edge_count());
    let (d0, _, _, _) = seg.csr().edge_slices(0).unwrap();
    let (d1, _, _, _) = owned.edge_slices(0).unwrap();
    assert_eq!(d0, d1);

    fs::remove_file(&path).ok();
}

#[tokio::test]
async fn segment_edges_udtf_returns_the_sorted_edge_columns() {
    let owned = sample_csr();
    let path = temp_path("edges");
    write_segment(&path, &owned).unwrap();

    let ctx = GtvContext::new();
    ctx.register_segment(path.to_str().unwrap(), "g").unwrap();

    let batches = ctx
        .sql("SELECT src, dst, valid_from, valid_to, edge_type FROM gtv_segment_edges('g')")
        .await
        .unwrap();
    let b = &batches[0];
    let c = owned.columns();
    assert_eq!(b.num_rows(), c.dst.len());

    let dst = b
        .column(1)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let vf = b.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
    let vt = b.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
    let et = b
        .column(4)
        .as_any()
        .downcast_ref::<UInt16Array>()
        .unwrap();
    for i in 0..c.dst.len() {
        assert_eq!(dst.value(i), c.dst[i], "dst at {i}");
        assert_eq!(vf.value(i), c.valid_from[i], "valid_from at {i}");
        assert_eq!(vt.value(i), c.valid_to[i], "valid_to at {i}");
        assert_eq!(et.value(i), c.edge_type[i], "edge_type at {i}");
    }

    // A plain SQL predicate still works on the materialised view.
    let filtered = ctx
        .sql("SELECT count(*) FROM gtv_segment_edges('g') WHERE valid_to > 200")
        .await
        .unwrap();
    let cnt = filtered[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(cnt, c.valid_to.iter().filter(|&&v| v > 200).count() as i64);

    fs::remove_file(&path).ok();
}

#[tokio::test]
async fn unknown_segment_and_bad_path_fail_cleanly() {
    let ctx = GtvContext::new();

    let empty = ctx.sql("SELECT * FROM gtv_segments()").await.unwrap();
    assert_eq!(empty[0].num_rows(), 0);

    let err = ctx
        .sql("SELECT * FROM gtv_segment_edges('missing')")
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("no GTVSeg registered"),
        "unexpected error: {err}"
    );

    let err = ctx
        .register_segment("C:/definitely/not/a/gtvseg/file.seg", "bad")
        .unwrap_err();
    assert!(
        err.to_string().contains("cannot open GTVSeg"),
        "unexpected error: {err}"
    );
}
