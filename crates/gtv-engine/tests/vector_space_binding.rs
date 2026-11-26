//! Strong vector-space binding at the SQL surface.
//!
//! A collection registered with a `VectorSpaceBinding` must refuse a `knn()`
//! query that names a different embedding space, instead of returning plausible
//! but meaningless neighbours. An unbound collection must refuse a query that
//! names *any* space, because "cannot verify" is not "compatible".

use gtv_core::{Metric, VectorSpace, VectorSpaceBinding};
use gtv_engine::GtvContext;
use gtv_index::{AnyIndex, FlatIndex};

fn two_dim_vectors() -> Vec<Vec<f32>> {
    vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![5.0, 5.0]]
}

#[tokio::test]
async fn knn_refuses_a_query_for_a_different_space() {
    let ctx = GtvContext::new();
    let space_a = VectorSpace::new("model-a", "v1", "feat-v1", Metric::L2, 2);
    let space_b = VectorSpace::new("model-b", "v1", "feat-v1", Metric::L2, 2);
    assert_ne!(space_a.id(), space_b.id(), "same dim, different model");

    let index = AnyIndex::Flat(
        FlatIndex::with_metric(vec![0, 1, 2], two_dim_vectors(), Metric::L2).unwrap(),
    );
    ctx.register_knn_bound(
        "vecs",
        index,
        None,
        Some(VectorSpaceBinding::new(Some(7), space_a.id())),
    )
    .unwrap();

    // No space named: answers normally.
    let ok = ctx
        .sql("SELECT id FROM knn('vecs', '0,0', 1)")
        .await
        .unwrap();
    assert_eq!(ok.len(), 1);

    // The right space id: answers normally.
    let sql_ok = format!(
        "SELECT id FROM knn('vecs', '0,0', 1, '*', 'l2', '{}')",
        space_a.id()
    );
    ctx.sql(&sql_ok)
        .await
        .expect("matching space must be accepted");

    // A different model at the same dimension: refused.
    let sql_bad = format!(
        "SELECT id FROM knn('vecs', '0,0', 1, '*', 'l2', '{}')",
        space_b.id()
    );
    let err = ctx.sql(&sql_bad).await.unwrap_err();
    assert!(
        format!("{err}").contains("vector space mismatch"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn unbound_collection_refuses_a_named_space() {
    let ctx = GtvContext::new();
    ctx.register_knn_metric("loose", vec![0, 1, 2], two_dim_vectors(), None, Metric::L2)
        .unwrap();

    // Unqualified query works.
    ctx.sql("SELECT id FROM knn('loose', '0,0', 1)")
        .await
        .unwrap();

    // Naming a space against an unbound collection is refused.
    let space = VectorSpace::new("model-a", "v1", "feat-v1", Metric::L2, 2);
    let sql = format!(
        "SELECT id FROM knn('loose', '0,0', 1, '*', 'l2', '{}')",
        space.id()
    );
    let err = ctx.sql(&sql).await.unwrap_err();
    assert!(
        format!("{err}").contains("unbound"),
        "unexpected error: {err}"
    );
}
