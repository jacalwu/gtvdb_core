//! L2 in SQL: `gtv_segments()` must expose both halves of the time box, and be
//! honest about which half it actually knows.
//!
//! * `all_active_from` / `all_active_to` come from the GTVSeg header, so they are
//!   available for **any** opened segment (intersection window).
//! * `valid_from_min` / `valid_to_max` come from the manifest, so they are NULL
//!   for a segment registered by bare path (union box).
//!
//! Getting this backwards is the classic pruning bug: the intersection is not a
//! bounding box, and the union is not an all-active window.

use arrow::array::{Int64Array, StringArray};
use gtv_engine::GtvContext;
use gtv_storage::{EdgeRecord, SegmentStore};

fn tmp_root(label: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("gtv_l2sql_{}_{}", std::process::id(), label));
    let _ = std::fs::remove_dir_all(&p);
    p
}

const NODES: usize = 16;

/// Three monthly segments; each covers `[m*1000, m*1000+450)`.
fn store(root: &std::path::Path) -> SegmentStore {
    let mut s = SegmentStore::open(root, NODES).unwrap();
    for m in 0..3i64 {
        let base = m * 1000;
        let batch: Vec<EdgeRecord> = (0..8u64)
            .map(|i| {
                let vf = base + (i as i64 % 2) * 50;
                EdgeRecord::new(i % 16, (i * 3) % 16, vf, vf + 400, 0, m + 1)
            })
            .collect();
        s.ingest(1, 0, &batch).unwrap();
        s.flush_at(m + 1).unwrap();
    }
    s
}

#[tokio::test]
async fn gtv_segments_exposes_both_time_boxes() {
    let root = tmp_root("boxes");
    let s = store(&root);

    let ctx = GtvContext::new();
    ctx.register_segment_store(&s, "m").unwrap();

    let out = ctx
        .sql(
            "SELECT name, all_active_from, all_active_to, valid_from_min, valid_to_max, \
                    node_id_min, node_id_max, prunable \
             FROM gtv_segments() ORDER BY name",
        )
        .await
        .unwrap();
    let b = &out[0];
    assert_eq!(b.num_rows(), 3);
    assert_eq!(b.num_columns(), 8);

    let col = |i: usize| b.column(i).as_any().downcast_ref::<Int64Array>().unwrap();
    let colu = |i: usize| {
        b.column(i)
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap()
    };
    let name0 = b.column(0).as_any().downcast_ref::<StringArray>().unwrap().value(0);
    assert!(name0.starts_with("m_"), "unexpected name {name0}");

    // Header half (intersection): always present.
    assert_eq!(col(1).value(0), 50, "all_active_from");
    assert_eq!(col(2).value(0), 400, "all_active_to");
    // Manifest half (union): also present because we came from a manifest.
    assert_eq!(col(3).value(0), 0, "valid_from_min");
    assert_eq!(col(4).value(0), 450, "valid_to_max");
    assert_eq!(colu(5).value(0), 0, "node_id_min");
    assert!(colu(6).value(0) > 0, "node_id_max");

    // The last segment is the third month.
    assert_eq!(col(3).value(2), 2000);
    assert_eq!(col(4).value(2), 2450);
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn bare_segment_path_is_not_prunable() {
    let root = tmp_root("bare");
    let s = store(&root);
    let path = s.segment_path(&s.segments()[0]).to_string_lossy().into_owned();

    let ctx = GtvContext::new();
    ctx.register_segment(&path, "one").unwrap();

    let out = ctx
        .sql("SELECT prunable, valid_from_min, valid_to_max, all_active_from FROM gtv_segments()")
        .await
        .unwrap();
    let b = &out[0];
    let prunable = b
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::BooleanArray>()
        .unwrap()
        .value(0);
    assert!(!prunable, "a bare .seg has header bounds but no union box");
    assert!(b.column(1).is_null(0), "valid_from_min must be NULL, not 0");
    assert!(b.column(2).is_null(0), "valid_to_max must be NULL, not 0");
    assert_eq!(
        b.column(3).as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        50,
        "the intersection window is still known from the header"
    );

    // Explicitly attaching the manifest box flips it to prunable.
    let meta = s.segments()[0].meta().unwrap();
    ctx.register_segment_meta("one", meta).unwrap();
    let out = ctx.sql("SELECT prunable FROM gtv_segments()").await.unwrap();
    assert!(out[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::BooleanArray>()
        .unwrap()
        .value(0));
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn full_hit_is_answerable_in_sql() {
    let root = tmp_root("fullhit");
    let s = store(&root);
    let ctx = GtvContext::new();
    ctx.register_segment_store(&s, "m").unwrap();

    // "Which segments are wholly active at t=200?" needs no edge access at all.
    let out = ctx
        .sql(
            "SELECT count(*) FROM gtv_segments() \
             WHERE all_active_from <= 200 AND 200 < all_active_to",
        )
        .await
        .unwrap();
    assert_eq!(
        out[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        1,
        "only month 0 is wholly active at 200"
    );

    // "Which segments could touch [2200, 2300)?" — union box, no file opened.
    let out = ctx
        .sql(
            "SELECT count(*) FROM gtv_segments() \
             WHERE valid_from_min < 2300 AND valid_to_max > 2200",
        )
        .await
        .unwrap();
    assert_eq!(
        out[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        1
    );
    let _ = std::fs::remove_dir_all(&root);
}
