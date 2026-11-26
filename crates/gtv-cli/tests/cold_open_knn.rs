//! End-to-end test for the REPL cold-open commands (`.knn-index` / `.ann-index`).
//!
//! It exercises the exact same code path the REPL uses:
//! `IndexStore::build` → `load_index(path)` → `register_knn_from_index` /
//! `register_any_index`, then runs the SQL `knn()` / `ann()` surfaces and
//! asserts the returned neighbors are correct. This proves the mmap cold-open
//! gap (V-5) is closed end-to-end, not just at the engine-API layer.

use gtv_core::Metric;
use gtv_engine::GtvContext;
use gtv_index::{AnyIndex, BuildOptions, FlatIndex};
use gtv_index_store::{load_index, IndexMeta, IndexStore};

/// Unique scratch directory per test.
///
/// `SystemTime` alone is not enough on Windows: its granularity can be several
/// milliseconds, so tests started in the same tick would share one store root
/// and clobber each other's `indexes.json` (surfacing as `NotFound("<name>")`).
/// The process-wide counter makes the name unique regardless of clock
/// resolution.
fn temp_store_root() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);

    let base = std::env::temp_dir().join(format!(
        "gtv_cli_cold_open_{}_{}_{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&base).unwrap();
    base
}

/// Build a tiny, hand-checked corpus and persist it through `IndexStore` so the
/// container exercises the real GTVVEC mmap payload (not the legacy path).
fn build_fixture(root: &std::path::Path) -> std::path::PathBuf {
    // dim = 4. id0 at origin; id1/id2 one axis out; id3 far away.
    let ids = vec![0u64, 1, 2, 3];
    let vectors = vec![
        vec![0.0f32, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![10.0, 10.0, 10.0, 10.0],
    ];
    let flat = FlatIndex::with_metric(ids, vectors, Metric::L2).unwrap();
    let store = IndexStore::open(root).unwrap();
    let v = store
        .build("songs", &AnyIndex::Flat(flat), &BuildOptions::Flat, &IndexMeta::default())
        .unwrap();
    // Mirror the path layout `IndexStore` writes: <root>/<id>/v<version>/index.gtvidx.
    root.join(v.index_id.to_string())
        .join(format!("v{}", v.version))
        .join("index.gtvidx")
}

fn ids_of(batches: &[arrow::array::RecordBatch]) -> Vec<u64> {
    use arrow::array::AsArray;
    let b = &batches[0];
    let col = b.column(0).as_primitive::<arrow::datatypes::UInt64Type>();
    (0..col.len()).map(|i| col.value(i)).collect()
}

#[tokio::test]
async fn knn_index_cold_open_serves_mmap_corpus() {
    let root = temp_store_root();
    let container = build_fixture(&root);

    // The exact sequence the `.knn-index` REPL command performs:
    let index = load_index(&container).unwrap();
    let ctx = GtvContext::new();
    ctx.register_knn_from_index("songs", index, None).unwrap();

    // Query at the origin → nearest is id0, then id1/id2 (tie → ascending id).
    let batches = ctx
        .sql("SELECT id, distance FROM knn('songs', '0,0,0,0', 3)")
        .await
        .unwrap();
    assert_eq!(ids_of(&batches), vec![0, 1, 2]);

    // Query at id3's location → nearest is id3 (distance 0).
    let batches = ctx
        .sql("SELECT id FROM knn('songs', '10,10,10,10', 1)")
        .await
        .unwrap();
    assert_eq!(ids_of(&batches), vec![3]);

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn ann_index_cold_open_serves_mmap_corpus() {
    let root = temp_store_root();
    let container = build_fixture(&root);

    // The exact sequence the `.ann-index` REPL command performs:
    let index = load_index(&container).unwrap();
    let ctx = GtvContext::new();
    ctx.register_any_index("idx", index);

    let batches = ctx
        .sql("SELECT id FROM ann('idx', '0,0,0,0', 3, 'l2')")
        .await
        .unwrap();
    assert_eq!(ids_of(&batches), vec![0, 1, 2]);

    let _ = std::fs::remove_dir_all(&root);
}

/// T3c: the save direction. Register an in-memory (Flat/L2) collection, persist
/// it with the same `knn_to_any_index` + `IndexStore::save` sequence the
/// `.knn-save` command uses, then cold-open the produced `.gtvidx` in a fresh
/// context and assert both answer identically.
#[tokio::test]
async fn knn_save_then_cold_open_round_trips() {
    let root = temp_store_root();
    let ctx = GtvContext::new();

    let ids = vec![0u64, 1, 2, 3];
    let vectors = vec![
        vec![0.0f32, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![10.0, 10.0, 10.0, 10.0],
    ];
    ctx.register_knn("songs", ids, vectors, None).unwrap();

    // Persist exactly as `.knn-save` does.
    let (index, options) = ctx.knn_to_any_index("songs").unwrap();
    let store = IndexStore::open(&root).unwrap();
    let v = store
        .save("songs", &index, &options, &IndexMeta::default())
        .unwrap();
    let container = root
        .join(v.index_id.to_string())
        .join(format!("v{}", v.version))
        .join("index.gtvidx");

    // Cold-open through the `.knn-index` path in a brand-new context.
    let ctx2 = GtvContext::new();
    let loaded = load_index(&container).unwrap();
    ctx2.register_knn_from_index("songs2", loaded, None).unwrap();

    let before = ctx
        .sql("SELECT id FROM knn('songs', '10,10,10,10', 2)")
        .await
        .unwrap();
    let after = ctx2
        .sql("SELECT id FROM knn('songs2', '10,10,10,10', 2)")
        .await
        .unwrap();
    // Nearest is the far-corner vector (id 3), then id 1.
    assert_eq!(ids_of(&before), vec![3, 1]);
    // The mmap round-trip is lossless: identical answers before vs after.
    assert_eq!(ids_of(&before), ids_of(&after));

    let _ = std::fs::remove_dir_all(&root);
}

/// T3c (IVF): the same save→cold-open loop over the compound `GIVFCMP` payload.
#[tokio::test]
async fn knn_ivf_save_then_cold_open_round_trips() {
    use gtv_engine::knn::IvfOptions;

    let root = temp_store_root();
    let ctx = GtvContext::new();

    let ids: Vec<u64> = (0..64).collect();
    let vectors: Vec<Vec<f32>> = (0..64)
        .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i % 7) as f32])
        .collect();
    ctx.register_knn_ivf("ivf", ids, vectors, None, IvfOptions::new(Metric::L2, 8, 8))
        .unwrap();

    let (index, options) = ctx.knn_to_any_index("ivf").unwrap();
    assert!(matches!(options, gtv_index::BuildOptions::Ivf { .. }));

    let store = IndexStore::open(&root).unwrap();
    let v = store
        .save("ivf", &index, &options, &IndexMeta::default())
        .unwrap();
    let container = root
        .join(v.index_id.to_string())
        .join(format!("v{}", v.version))
        .join("index.gtvidx");

    let ctx2 = GtvContext::new();
    let loaded = load_index(&container).unwrap();
    ctx2.register_knn_from_index("ivf2", loaded, None).unwrap();

    // Full-probe IVF equals the flat scan; the round-trip must be bit-identical.
    let before = ctx
        .sql("SELECT id FROM knn('ivf', '0,0,0,0', 5)")
        .await
        .unwrap();
    let after = ctx2
        .sql("SELECT id FROM knn('ivf2', '0,0,0,0', 5)")
        .await
        .unwrap();
    assert_eq!(ids_of(&before)[0], 0);
    assert_eq!(ids_of(&before), ids_of(&after));

    let _ = std::fs::remove_dir_all(&root);
}

/// V-5 (labels): metadata filtering must survive save → cold-open.
///
/// IVF stores the corpus cell-ordered but keeps `orig_pos`, and the `GIVFCMP`
/// payload persists that column, so a label sidecar in **original input order**
/// still lands on the right rows after a cold open. This drives the exact
/// `.knn-index <name> <file> <labels.txt>` path.
#[tokio::test]
async fn knn_ivf_labels_survive_cold_open() {
    use gtv_engine::knn::IvfOptions;

    let root = temp_store_root();
    let ctx = GtvContext::new();

    // A cluster per parity so an unfiltered probe would happily mix the two
    // labels — a wrong mask domain therefore fails loudly, not by luck.
    let ids: Vec<u64> = (0..64).collect();
    let vectors: Vec<Vec<f32>> = (0..64)
        .map(|i| {
            let base = if i % 2 == 0 { 0.0f32 } else { 50.0 };
            vec![base + i as f32 * 0.01, base, base, base]
        })
        .collect();
    let labels: Vec<String> = (0..64)
        .map(|i| if i % 2 == 0 { "even" } else { "odd" }.to_string())
        .collect();

    ctx.register_knn_ivf(
        "ivf",
        ids,
        vectors,
        Some(labels.clone()),
        IvfOptions::new(Metric::L2, 8, 8),
    )
    .unwrap();

    let (index, options) = ctx.knn_to_any_index("ivf").unwrap();
    let store = IndexStore::open(&root).unwrap();
    let v = store
        .save("ivf", &index, &options, &IndexMeta::default())
        .unwrap();
    let container = root
        .join(v.index_id.to_string())
        .join(format!("v{}", v.version))
        .join("index.gtvidx");

    // Write the sidecar the way a user would, then cold-open with it.
    let sidecar = root.join("labels.txt");
    std::fs::write(&sidecar, labels.join("\n")).unwrap();

    let ctx2 = GtvContext::new();
    let loaded = load_index(&container).unwrap();
    ctx2.register_knn_from_index("ivf2", loaded, Some(labels)).unwrap();

    for label in ["even", "odd"] {
        let want = ctx
            .sql(&format!("SELECT id FROM knn('ivf', '0,0,0,0', 6, '{label}')"))
            .await
            .unwrap();
        let got = ctx2
            .sql(&format!("SELECT id FROM knn('ivf2', '0,0,0,0', 6, '{label}')"))
            .await
            .unwrap();
        assert_eq!(ids_of(&got), ids_of(&want), "label `{label}` must match");
        for id in ids_of(&got) {
            let expect_even = label == "even";
            assert_eq!(id % 2 == 0, expect_even, "id {id} leaked into `{label}`");
        }
    }

    // No filter → the two clusters are distinct, so both parities appear.
    let all = ctx2
        .sql("SELECT id FROM knn('ivf2', '0,0,0,0', 6)")
        .await
        .unwrap();
    assert!(ids_of(&all).iter().any(|id| id % 2 == 0));

    let _ = std::fs::remove_dir_all(&root);
}

/// The REPL cold-open path must bind the collection to the embedding space the
/// manifest recorded, so a later query naming a different model is refused.
#[tokio::test]
async fn knn_cold_open_binds_the_recorded_vector_space() {
    use gtv_core::{VectorSpace, VectorSpaceBinding};

    let root = temp_store_root();
    let container = build_fixture(&root);

    let (index, manifest) = gtv_index_store::load_index_with_manifest(&container).unwrap();
    // build_fixture uses default meta: empty model/version/feature, L2, dim 4.
    let space = VectorSpace::new("", "", "", Metric::L2, 4);
    let recorded = manifest
        .and_then(|m| m.vector_space_id)
        .expect("cold-open manifest records a vector_space_id");
    assert_eq!(recorded, space.id());

    let ctx = GtvContext::new();
    ctx.register_knn_bound(
        "songs",
        index,
        None,
        Some(VectorSpaceBinding::new(None, recorded)),
    )
    .unwrap();

    // A query naming a *different* space is refused, not answered.
    let other = VectorSpace::new("other", "", "", Metric::L2, 4);
    let sql = format!(
        "SELECT id FROM knn('songs', '0,0,0,0', 1, '*', 'l2', '{}')",
        other.id()
    );
    let err = ctx.sql(&sql).await.unwrap_err();
    assert!(
        format!("{err}").contains("vector space mismatch"),
        "unexpected error: {err}"
    );

    // The matching space answers normally.
    let ok_sql = format!(
        "SELECT id FROM knn('songs', '0,0,0,0', 1, '*', 'l2', '{}')",
        recorded
    );
    ctx.sql(&ok_sql).await.unwrap();
}
