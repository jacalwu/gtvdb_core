//! G8 / G9 acceptance: a result says which knowledge state it reflects.
//!
//! The v3 bank-scenario plan asks one question that a graph engine usually
//! cannot answer: "why is the 14:30 transaction missing from the 15:00 report?"
//! The answer must be a readable watermark, not a shrug — and an empty result
//! must never be the answer when the graph simply has not been refreshed yet.
//!
//! These tests pin the three cases apart:
//!
//! * a stamped watermark is carried by the receipt and by the Arrow metadata;
//! * a `NoMatch` behind the caller's "now" is `NotYetRefreshed`, not proof;
//! * an unstamped context says "freshness unknown" instead of "lag zero".
//!
//! Data discipline: in-memory tables, nothing written to disk.

use std::sync::Arc;

use arrow::array::{Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use gtv_core::{Absence, Freshness, KnowledgeClock};
use gtv_engine::{receipt_from_batches, GtvContext, QueryBinding};

fn seed(ctx: &GtvContext) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::UInt64, false),
        Field::new("event_time", DataType::Int64, false),
    ]));
    let batch = arrow::array::RecordBatch::try_new(
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

fn freshness(known_at: i64) -> Freshness {
    Freshness::new(known_at, KnowledgeClock::RecordedAt)
}

#[tokio::test]
async fn receipt_carries_the_watermark_it_was_told() {
    let ctx = GtvContext::new();
    seed(&ctx);
    ctx.set_freshness(freshness(90).with_history_floor(10));

    let outcome = ctx
        .sql_with_attached_receipt("q-1", "SELECT count(*) FROM transfers", None, None, &[])
        .await
        .unwrap();

    let stamped = outcome.receipt.freshness.expect("watermark stamped");
    assert_eq!(stamped.as_known_at, 90);
    assert_eq!(stamped.clock, KnowledgeClock::RecordedAt);
    assert_eq!(stamped.history_floor, Some(10));

    // It also rides on the Arrow metadata, so an Arrow-only reader sees it.
    let read_back = receipt_from_batches(&outcome.batches)
        .expect("present")
        .expect("readable");
    assert_eq!(read_back, outcome.receipt);
    assert_eq!(
        outcome.batches[0]
            .schema()
            .metadata()
            .get("gtv.receipt.as_known_at")
            .map(String::as_str),
        Some("90")
    );
}

#[tokio::test]
async fn empty_result_behind_the_watermark_is_not_a_proof() {
    let ctx = GtvContext::new();
    seed(&ctx);
    // The graph was refreshed at 90; the report is being asked from 150.
    ctx.set_freshness(freshness(90));

    // `.applied()` because this test is about the watermark, not about whether
    // anybody enforced the business time: a declared point would cap the result
    // at `Incomplete` and there would be no absence to classify.
    let outcome = ctx
        .sql_with_receipt_binding(
            "q-2",
            "SELECT * FROM transfers WHERE account_id = 999",
            QueryBinding::at(150).applied(),
            &[],
        )
        .await
        .unwrap();

    assert_eq!(outcome.receipt.tri_state, gtv_core::TriState::NoMatch);
    assert_eq!(
        outcome.receipt.absence_at(150, Some(150)),
        Absence::NotYetRefreshed { behind_by: 60 }
    );
    assert!(!outcome.receipt.proves_absence_at(150, Some(150)));
    // At its own watermark the same answer is conclusive.
    assert!(outcome.receipt.proves_absence_at(90, Some(150)));
}

#[tokio::test]
async fn unstamped_context_reports_unknown_not_zero_lag() {
    let ctx = GtvContext::new();
    seed(&ctx);

    let outcome = ctx
        .sql_with_receipt("q-3", "SELECT * FROM transfers WHERE account_id = 999", None, None, &[])
        .await
        .unwrap();

    assert!(outcome.receipt.freshness.is_none());
    assert_eq!(outcome.receipt.freshness_verdict(150), None);
    assert_eq!(outcome.receipt.absence_at(150, None), Absence::NotApplicable);
}

#[tokio::test]
async fn clearing_the_watermark_stops_stamping_it() {
    let ctx = GtvContext::new();
    seed(&ctx);
    ctx.set_freshness(freshness(90));
    ctx.clear_freshness();
    assert!(ctx.freshness().is_none());

    let outcome = ctx
        .sql_with_receipt("q-4", "SELECT count(*) FROM transfers", None, None, &[])
        .await
        .unwrap();
    assert!(outcome.receipt.freshness.is_none());
    assert!(outcome.receipt.validate().is_ok());
}

#[tokio::test]
async fn absence_below_the_retention_floor_fails_instead_of_shipping() {
    let ctx = GtvContext::new();
    seed(&ctx);
    ctx.set_freshness(freshness(200).with_history_floor(120));

    // A `NoMatch` below the retention floor would be a misleading receipt — the
    // files that might have held the rows are gone — so the statement fails
    // rather than returning an empty result that reads as "no such transfer".
    let err = ctx
        .sql_with_receipt_binding(
            "q-5",
            "SELECT * FROM transfers WHERE account_id = 999",
            QueryBinding::at(50).applied(),
            &[],
        )
        .await
        .expect_err("absence below the GC floor must not ship");
    let text = err.to_string();
    assert!(text.contains("GC floor"), "unexpected error: {text}");
    assert!(text.contains("unanswerable"), "unexpected error: {text}");
}
