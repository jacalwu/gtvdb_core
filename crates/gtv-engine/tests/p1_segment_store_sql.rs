//! P1 ingest → GTVSeg → SQL registration end-to-end.
//!
//! The data is built in memory and written only to a tempdir. No CSV exists at
//! any point: `assert_no_csv` walks the store root before the test finishes.

use std::fs;
use std::path::{Path, PathBuf};

use arrow::array::{Int64Array, StringArray, UInt64Array};
use gtv_engine::GtvContext;
use gtv_storage::{EdgeRecord, SegmentStore};

fn temp_root(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("gtv_p1_sql_{}_{}", std::process::id(), label));
    let _ = fs::remove_dir_all(&p);
    p
}

fn assert_no_csv(root: &Path) {
    fn walk(dir: &Path, found: &mut Vec<PathBuf>) {
        if let Ok(rd) = fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, found);
                } else if p.extension().and_then(|s| s.to_str()) == Some("csv") {
                    found.push(p);
                }
            }
        }
    }
    let mut found = Vec::new();
    walk(root, &mut found);
    assert!(found.is_empty(), "P1 wrote CSV data: {found:?}");
}

#[tokio::test]
async fn ingested_edges_are_queryable_as_mmap_segments() {
    let root = temp_root("register");
    let mut store = SegmentStore::open(&root, 16).unwrap();
    store.set_flush_threshold(usize::MAX);

    let edges: Vec<EdgeRecord> = (0..24u64)
        .map(|i| EdgeRecord::new(i % 16, (i * 5 + 1) % 16, 10, 1000, 1, 7))
        .collect();
    store.ingest(3, 0, &edges).unwrap();
    let report = store.flush_at(10).unwrap();
    assert_eq!(report.new_segments, 1);

    let ctx = GtvContext::new();
    ctx.register_segment_store(&store, "p1").unwrap();

    let batches = ctx
        .sql("SELECT name, backend, node_count, edge_count FROM gtv_segments()")
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
        "p1_1"
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
        16
    );
    assert_eq!(
        b.column(3)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0),
        24
    );

    // The materialising UDTF sees exactly the ingested edges.
    let batches = ctx
        .sql("SELECT count(*) AS n FROM gtv_segment_edges('p1_1')")
        .await
        .unwrap();
    let b = &batches[0];
    assert_eq!(
        b.column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        24
    );

    assert_no_csv(&root);
    drop(store);
    let _ = fs::remove_dir_all(&root);
}
