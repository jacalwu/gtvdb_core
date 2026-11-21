//! A requested time point is a claim about provenance, so it has to be earned.
//!
//! The hole this file guards is narrow and easy to reopen: a client sends
//! `as_of = 14:00`, the server runs `SELECT * FROM transfers`, and the rows come
//! back stamped `as_of: 14:00`. Nothing filtered by 14:00, so the receipt
//! testifies to something nobody checked — an audit finding waiting to happen.
//!
//! Three rules, in the order they matter:
//!
//! 1. a point nothing enforced caps the result at `Incomplete`
//!    (`a_requested_as_of_nothing_applied_is_not_conclusive`);
//! 2. naming a point without saying who enforced it is invalid, not "assumed
//!    applied" (`a_point_without_a_binding_never_ships`);
//! 3. a pinned snapshot is either verified against the head or refused — never
//!    quietly swapped for the one the server happens to hold
//!    (`a_pinned_snapshot_that_is_not_the_head_is_refused`).
//!
//! Data discipline: in-memory tables, nothing written to disk.

use std::sync::Arc;

use arrow::array::{Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use gtv_core::{IncompleteReason, PointBinding, TriState};
use gtv_engine::{receipt_from_batches, GtvContext, QueryBinding};

fn seed(ctx: &GtvContext) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::UInt64, false),
        Field::new("event_time", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![1u64, 2])),
            Arc::new(Int64Array::from(vec![100i64, 140])),
        ],
    )
    .unwrap();
    ctx.register_batches("transfers", schema, vec![batch])
        .unwrap();
}

#[tokio::test]
async fn a_requested_as_of_nothing_applied_is_not_conclusive() {
    let ctx = GtvContext::new();
    seed(&ctx);

    // The old behaviour: 2 rows, `tri_state = found`, `as_of = 150`, and no
    // statement in sight that mentions time.
    let outcome = ctx
        .sql_with_receipt_binding(
            "q-1",
            "SELECT * FROM transfers",
            QueryBinding::at(150),
            &[],
        )
        .await
        .unwrap();

    assert_eq!(outcome.rows(), 2, "the rows are real and still returned");
    assert_eq!(outcome.receipt.tri_state, TriState::Incomplete);
    assert_eq!(
        outcome.receipt.incomplete.as_ref().unwrap().reason,
        IncompleteReason::TimePointNotApplied
    );
    assert_eq!(outcome.receipt.as_of, Some(150));
    assert_eq!(outcome.receipt.as_of_binding, Some(PointBinding::Declared));
    assert!(!outcome.receipt.as_of_applied());
    assert!(outcome.receipt.validate().is_ok());
}

#[tokio::test]
async fn an_enforced_as_of_stays_conclusive() {
    let ctx = GtvContext::new();
    seed(&ctx);

    // `event_time <= 100` is the statement enforcing the point, so claiming it
    // is honest — and the receipt says so instead of hiding behind `Incomplete`.
    let outcome = ctx
        .sql_with_receipt_binding(
            "q-2",
            "SELECT * FROM transfers WHERE event_time <= 100",
            QueryBinding::at(100).applied(),
            &[],
        )
        .await
        .unwrap();

    assert_eq!(outcome.receipt.tri_state, TriState::Found);
    assert!(outcome.receipt.as_of_applied());
    assert!(outcome.receipt.incomplete.is_none());
    assert_eq!(outcome.rows(), 1);
}

#[tokio::test]
async fn the_binding_travels_with_the_rows() {
    let ctx = GtvContext::new();
    seed(&ctx);

    let outcome = ctx
        .sql_with_attached_receipt_binding(
            "q-3",
            "SELECT count(*) FROM transfers",
            QueryBinding::at(150),
            &[],
        )
        .await
        .unwrap();

    let meta = outcome.batches[0].schema().metadata().clone();
    assert_eq!(
        meta.get("gtv.receipt.as_of_binding").map(String::as_str),
        Some("declared"),
        "an Arrow-only reader must see the same caveat as a JSON reader"
    );
    assert_eq!(
        meta.get("gtv.receipt.incomplete_reason").map(String::as_str),
        Some("time_point_not_applied")
    );
    let back = receipt_from_batches(&outcome.batches)
        .expect("present")
        .expect("readable");
    assert_eq!(back, outcome.receipt);
}

#[tokio::test]
async fn a_pinned_snapshot_that_is_the_head_is_verified() {
    let ctx = GtvContext::new();
    seed(&ctx);
    ctx.set_head_snapshot(7);

    let outcome = ctx
        .sql_with_receipt_binding(
            "q-4",
            "SELECT count(*) FROM transfers",
            QueryBinding::snapshot(7),
            &[],
        )
        .await
        .unwrap();

    assert_eq!(outcome.receipt.snapshot_id, Some(7));
    assert_eq!(outcome.receipt.snapshot_binding, Some(PointBinding::Applied));
    assert_eq!(
        outcome.receipt.tri_state,
        TriState::Found,
        "a verified snapshot needs no caveat"
    );
}

#[tokio::test]
async fn a_pinned_snapshot_that_is_not_the_head_is_refused() {
    let ctx = GtvContext::new();
    seed(&ctx);
    ctx.set_head_snapshot(7);

    // Refusing beats re-reading the head and hoping the client does not notice:
    // the client pinned 9 on purpose, and an answer from 7 is a different fact.
    let err = ctx
        .sql_with_receipt_binding(
            "q-5",
            "SELECT count(*) FROM transfers",
            QueryBinding::snapshot(9),
            &[],
        )
        .await
        .expect_err("a pinned snapshot must not be silently replaced");
    let text = err.to_string();
    assert!(text.contains("snapshot mismatch"), "unexpected error: {text}");
    assert!(text.contains("expected 9"), "the message must name both: {text}");
}

#[tokio::test]
async fn an_unverifiable_snapshot_is_declared_not_assumed() {
    let ctx = GtvContext::new();
    seed(&ctx);
    // No head declared: the server cannot say which state it is serving, so it
    // must not pretend the client's guess was right.
    assert!(ctx.head_snapshot().is_none());

    let outcome = ctx
        .sql_with_receipt_binding(
            "q-6",
            "SELECT count(*) FROM transfers",
            QueryBinding::snapshot(7),
            &[],
        )
        .await
        .unwrap();

    assert_eq!(outcome.receipt.snapshot_id, Some(7));
    assert_eq!(outcome.receipt.snapshot_binding, Some(PointBinding::Declared));
    assert_eq!(outcome.receipt.tri_state, TriState::Incomplete);

    // And once the host declares a head, the same request becomes verifiable.
    ctx.set_head_snapshot(7);
    let verified = ctx
        .sql_with_receipt_binding(
            "q-7",
            "SELECT count(*) FROM transfers",
            QueryBinding::snapshot(7),
            &[],
        )
        .await
        .unwrap();
    assert_eq!(verified.receipt.tri_state, TriState::Found);
}

#[tokio::test]
async fn clearing_the_head_snapshot_returns_to_declared() {
    let ctx = GtvContext::new();
    seed(&ctx);
    ctx.set_head_snapshot(7);
    ctx.clear_head_snapshot();
    assert!(ctx.head_snapshot().is_none());

    let outcome = ctx
        .sql_with_receipt_binding(
            "q-8",
            "SELECT count(*) FROM transfers",
            QueryBinding::snapshot(7),
            &[],
        )
        .await
        .unwrap();
    assert_eq!(outcome.receipt.snapshot_binding, Some(PointBinding::Declared));
}
