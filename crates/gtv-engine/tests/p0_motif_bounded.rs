//! A4 / C4 acceptance: bounded motif search over SQL, and the strict-vs-lenient
//! split that keeps a truncated prefix from being read as "no match".
//!
//! The rule under test (R2): a standard SQL path must never return partial rows;
//! an extension path may, but only if it also exposes the receipt.
//!
//! Data discipline: graphs are built in memory, nothing is written to disk.

use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray, TimestampNanosecondArray, UInt16Array, UInt64Array};
use gtv_core::{IncompleteReason, TemporalCSR, TraversalBudget, TriState};
use gtv_engine::{receipt_from_batches, GtvContext, MotifShape, MotifSpec};

const T: i64 = 100;

/// Two disjoint rings (0→1→2→0, 3→4→5→3) plus a chain. 6 nodes.
fn graph() -> TemporalCSR {
    let edges: Vec<(u64, u64, i64, i64, u16)> = vec![
        (0, 1, 10, 200, 1),
        (1, 2, 20, 200, 1),
        (2, 0, 30, 200, 1),
        (3, 4, 40, 200, 2),
        (4, 5, 50, 200, 2),
        (5, 3, 60, 200, 2),
        (0, 5, 70, 200, 3),
    ];
    let src: Vec<u64> = edges.iter().map(|e| e.0).collect();
    let dst: Vec<u64> = edges.iter().map(|e| e.1).collect();
    let vf: Vec<i64> = edges.iter().map(|e| e.2).collect();
    let vt: Vec<i64> = edges.iter().map(|e| e.3).collect();
    let et: Vec<u16> = edges.iter().map(|e| e.4).collect();
    TemporalCSR::from_arrays(
        &UInt64Array::from(src),
        &UInt64Array::from(dst),
        &TimestampNanosecondArray::from(vf),
        &TimestampNanosecondArray::from(vt),
        &UInt16Array::from(et),
        6,
    )
    .unwrap()
}

async fn scalar_string(ctx: &GtvContext, sql: &str) -> String {
    let batches = ctx.sql(sql).await.unwrap();
    assert_eq!(batches.len(), 1, "expected a single batch from `{sql}`");
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .value(0)
        .to_string()
}

#[tokio::test]
async fn unbounded_run_reports_found_and_its_row_count() {
    let ctx = GtvContext::new();
    let csr = Arc::new(graph());
    let spec = MotifSpec::new(MotifShape::Ring(3), T).with_limit(100);
    let summary = ctx
        .register_motif_batch_shared(&csr, &spec, &[0, 3])
        .unwrap();
    assert_eq!(summary.tri_state, TriState::Found);
    assert!(summary.is_complete());

    assert_eq!(scalar_string(&ctx, "SELECT tri_state FROM gtv_motif_receipt").await, "found");
    let rows = ctx.sql("SELECT * FROM gtv_motif_matches").await.unwrap();
    let n: usize = rows.iter().map(|b| b.num_rows()).sum();
    assert_eq!(n as u64, summary.stats.rows);
    // Two rings, each found from its own seed.
    assert!(n >= 2, "expected both rings, got {n}");
}

#[tokio::test]
async fn tight_budget_is_reported_not_silently_truncated() {
    let ctx = GtvContext::new();
    let csr = Arc::new(graph());
    let spec = MotifSpec::new(MotifShape::Ring(3), T)
        .with_limit(100)
        .with_budget(TraversalBudget::unlimited().with_max_edges(1));

    let summary = ctx
        .register_motif_batch_shared(&csr, &spec, &[0, 3])
        .unwrap();
    assert_eq!(summary.tri_state, TriState::Incomplete);
    let incomplete = summary.incomplete.expect("reason carried");
    assert_eq!(incomplete.reason, IncompleteReason::Edges);

    // The partial rows are still readable ...
    let tri = scalar_string(&ctx, "SELECT tri_state FROM gtv_motif_receipt").await;
    assert_eq!(tri, "incomplete");
    let reason = scalar_string(&ctx, "SELECT incomplete_reason FROM gtv_motif_receipt").await;
    assert_eq!(reason, "edges");
}

#[tokio::test]
async fn strict_path_refuses_partial_rows() {
    let ctx = GtvContext::new();
    let csr = Arc::new(graph());
    let spec = MotifSpec::new(MotifShape::Ring(3), T)
        .with_limit(100)
        .with_budget(TraversalBudget::unlimited().with_max_edges(1));

    let err = ctx
        .register_motif_batch_strict(&csr, &spec, &[0, 3])
        .expect_err("strict path must not return a prefix");
    let msg = err.to_string();
    assert!(msg.contains("incomplete"), "unhelpful error: {msg}");
    assert!(msg.contains("edges"), "error should name the cap: {msg}");

    // Nothing was registered: a partial table must not linger from a failed call.
    assert!(ctx.sql("SELECT * FROM gtv_motif_matches").await.is_err());
}

#[tokio::test]
async fn strict_path_registers_when_conclusive() {
    let ctx = GtvContext::new();
    let csr = Arc::new(graph());
    let spec = MotifSpec::new(MotifShape::Ring(3), T).with_limit(100);
    let summary = ctx
        .register_motif_batch_strict(&csr, &spec, &[0, 3])
        .unwrap();
    assert!(summary.is_complete());
    assert_eq!(scalar_string(&ctx, "SELECT tri_state FROM gtv_motif_receipt").await, "found");
}

#[tokio::test]
async fn statement_receipt_inherits_dependency_truncation() {
    let ctx = GtvContext::new();
    let csr = Arc::new(graph());
    let spec = MotifSpec::new(MotifShape::Ring(3), T)
        .with_limit(100)
        .with_budget(TraversalBudget::unlimited().with_max_edges(1));
    let summary = ctx
        .register_motif_batch_shared(&csr, &spec, &[0, 3])
        .unwrap();

    let outcome = ctx
        .sql_with_attached_receipt(
            "q-bounded-1",
            "SELECT n0, n1 FROM gtv_motif_matches",
            Some(T),
            Some(11),
            &[("gtv_motif_matches".to_string(), summary)],
        )
        .await
        .unwrap();

    assert_eq!(outcome.receipt.tri_state, TriState::Incomplete);
    assert_eq!(
        outcome.receipt.incomplete.as_ref().unwrap().reason,
        IncompleteReason::Edges
    );
    assert_eq!(outcome.receipt.snapshot_id, Some(11));
    assert_eq!(outcome.receipt.as_of, Some(T));
    assert_eq!(outcome.receipt.query_id, "q-bounded-1");
    // A dependency-truncated statement names the dependency in its plan hash.
    assert!(outcome
        .receipt
        .plan_hash
        .as_ref()
        .unwrap()
        .ends_with("@gtv_motif_matches"));

    // And the receipt travelled with the rows.
    let back = receipt_from_batches(&outcome.batches)
        .expect("receipt present")
        .unwrap();
    assert_eq!(back, outcome.receipt);
}

#[tokio::test]
async fn statement_receipt_is_conclusive_when_dependencies_are() {
    let ctx = GtvContext::new();
    let csr = Arc::new(graph());
    let spec = MotifSpec::new(MotifShape::Ring(3), T).with_limit(100);
    let summary = ctx
        .register_motif_batch_shared(&csr, &spec, &[0, 3])
        .unwrap();

    let outcome = ctx
        .sql_with_receipt(
            "q-ok-1",
            "SELECT n0, n1 FROM gtv_motif_matches",
            Some(T),
            None,
            &[("gtv_motif_matches".to_string(), summary)],
        )
        .await
        .unwrap();
    assert_eq!(outcome.receipt.tri_state, TriState::Found);
    assert!(outcome.receipt.incomplete.is_none());
    assert_eq!(outcome.receipt.stats.rows, outcome.rows() as u64);

    // A statement that finds nothing is NoMatch, not Found.
    let empty = ctx
        .sql_with_receipt("q-ok-2", "SELECT 1 WHERE false", None, None, &[])
        .await
        .unwrap();
    assert_eq!(empty.receipt.tri_state, TriState::NoMatch);
    assert!(empty.receipt.validate().is_ok());
}

#[tokio::test]
async fn budgeted_and_unbudgeted_agree_on_the_prefix_they_share() {
    // A generous budget must not change the answer, only the accounting.
    let ctx = GtvContext::new();
    let csr = Arc::new(graph());
    let plain = MotifSpec::new(MotifShape::Ring(3), T).with_limit(100);
    let budgeted = plain
        .clone()
        .with_budget(TraversalBudget::unlimited().with_max_edges(1_000_000));

    let a = ctx.register_motif_batch_shared(&csr, &plain, &[0, 3]).unwrap();
    let batch_a: Vec<RecordBatch> = ctx.sql("SELECT n0, n1, n2 FROM gtv_motif_matches").await.unwrap();
    let b = ctx.register_motif_batch_shared(&csr, &budgeted, &[0, 3]).unwrap();
    let batch_b: Vec<RecordBatch> = ctx.sql("SELECT n0, n1, n2 FROM gtv_motif_matches").await.unwrap();

    assert_eq!(a.stats.rows, b.stats.rows);
    assert_eq!(batch_a, batch_b);
}
