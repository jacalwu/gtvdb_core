//! Wire contract tests for protocol v1 (A1 / B5).
//!
//! Two layers:
//!
//! * pure tests over `decode_response` / `encode_response` — the failure modes
//!   (partial result, version mismatch, missing receipt) must be provable
//!   without a network;
//! * one live-loopback test per behaviour we promise a client: receipt travels,
//!   v0 callers still work, an unsupported version is refused with a code.
//!
//! No file is written; the server binds `127.0.0.1:0`.

use std::sync::Arc;

use arrow::array::{RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use gtv_core::{
    Freshness, Incomplete, IncompleteReason, KnowledgeClock, QueryReceipt, QueryStats, TriState,
};
use gtv_proto::{
    accept_version, bind, decode_response, encode_response, error_code_from_i32, query_remote_full,
    query_remote_with, receipt_from_ipc, tri_state_from_wire, ErrorCode, QueryRequest,
    QueryResponse, QueryServiceClient, RemoteQuery, WireError,
};

fn batch(values: &[u64]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::UInt64, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(UInt64Array::from(values.to_vec()))]).unwrap()
}

#[test]
fn legacy_request_is_understood() {
    // A v0 client sends only `sql`; every v1 field keeps its default.
    let q = RemoteQuery::new("SELECT 1", "req-1").with_as_of(500);
    let legacy = q.to_legacy_request();
    assert_eq!(legacy.sql, "SELECT 1");
    assert_eq!(legacy.protocol_version, 0);
    assert_eq!(legacy.as_of, None, "v0 has no as-of field");

    let v1 = q.to_request();
    assert_eq!(v1.protocol_version, 1);
    assert_eq!(v1.as_of, Some(500));
}

#[test]
fn version_gate_refuses_legacy_and_too_old() {
    assert_eq!(accept_version(1), Ok(1));
    // A newer client is served at our ceiling rather than refused.
    assert_eq!(accept_version(9), Ok(1));
    assert!(accept_version(0).is_err(), "v0 must use Execute");
    assert!(matches!(
        accept_version(0).unwrap_err().code,
        ErrorCode::UnsupportedVersion
    ));
}

#[test]
fn unknown_wire_codes_degrade_to_internal() {
    assert_eq!(error_code_from_i32(42), ErrorCode::Internal);
    assert_eq!(error_code_from_i32(-1), ErrorCode::Internal);
    assert_eq!(error_code_from_i32(1), ErrorCode::InvalidArgument);
    assert_eq!(tri_state_from_wire(3), None, "UNSPECIFIED is not a tri-state");
    assert_eq!(tri_state_from_wire(2), Some(TriState::Incomplete));
}

#[test]
fn receipt_roundtrips_through_a_wire_response() {
    let receipt = QueryReceipt::found("req-2")
        .with_as_of(700)
        .with_snapshot(3)
        .with_stats(QueryStats {
            rows: 2,
            ..Default::default()
        });
    let wire = encode_response(Ok((vec![batch(&[1, 2])], receipt.clone())));
    assert_eq!(wire.code, ErrorCode::Ok as i32);
    assert_eq!(wire.tri_state, gtv_proto::WireTriState::Found as i32);

    let outcome = decode_response(wire).unwrap();
    assert_eq!(outcome.rows(), 2);
    assert!(outcome.is_conclusive());
    assert_eq!(outcome.receipt.unwrap(), receipt);
    assert_eq!(outcome.protocol_version, 1);
}

#[test]
fn partial_result_keeps_its_rows_and_its_reason() {
    let receipt = QueryReceipt::incomplete(
        "req-3",
        Incomplete::new(IncompleteReason::Edges).at_hop(2),
    );
    let wire = encode_response(Ok((vec![batch(&[7])], receipt)));
    let outcome = decode_response(wire).unwrap();

    assert_eq!(outcome.rows(), 1, "partial rows must not be thrown away");
    assert!(!outcome.is_conclusive(), "an incomplete result proves nothing");
    assert_eq!(outcome.tri_state, Some(TriState::Incomplete));
    assert_eq!(
        outcome.receipt.unwrap().incomplete.unwrap().reason,
        IncompleteReason::Edges
    );
}

#[test]
fn error_response_carries_a_stable_code() {
    let wire = encode_response(Err(WireError::snapshot_mismatch("expected 4, got 5")));
    let err = decode_response(wire).unwrap_err();
    assert_eq!(err.code, ErrorCode::SnapshotMismatch);
    assert!(err.message.contains("expected 4"));

    let wire = encode_response(Err(WireError::incomplete("budget")));
    let outcome = decode_response(wire).unwrap();
    assert_eq!(outcome.rows(), 0);
    assert!(!outcome.is_conclusive());
}

#[test]
fn receipt_is_stamped_into_ipc_metadata() {
    // A consumer that only understands Arrow still gets the receipt.
    let receipt = QueryReceipt::found("req-4").with_as_of(9);
    let wire = encode_response(Ok((vec![batch(&[5])], receipt.clone())));
    let parsed = receipt_from_ipc(&wire.arrow_ipc)
        .expect("receipt present")
        .unwrap();
    assert_eq!(parsed, receipt);
}

#[test]
fn malformed_receipt_is_an_error_not_silence() {
    let wire = QueryResponse {
        arrow_ipc: Vec::new(),
        error: String::new(),
        protocol_version: 1,
        code: ErrorCode::Ok as i32,
        tri_state: 0,
        receipt_json: "{".to_string(),
    };
    match decode_response(wire) {
        Err(e) => assert_eq!(e.code, ErrorCode::Internal),
        Ok(_) => panic!("malformed receipt must not be accepted"),
    }
}

// ---------------------------------------------------------------------------
// Loopback tests
// ---------------------------------------------------------------------------

async fn start_server() -> (String, tokio::task::JoinHandle<()>) {
    let (addr, handle) = bind(
        "127.0.0.1:0".parse().unwrap(),
        gtv_proto::QueryServiceImpl::from_fn(|q| async move {
            let rows = if q.sql.contains("empty") { 0 } else { 2 };
            let mut receipt = QueryReceipt::new(
                q.request_id.clone(),
                if rows > 0 {
                    TriState::Found
                } else {
                    TriState::NoMatch
                },
            )
            .with_stats(QueryStats {
                rows,
                ..Default::default()
            });
            if let Some(as_of) = q.as_of {
                receipt = receipt.with_as_of(as_of);
            }
            Ok((vec![batch(&[1, 2])], receipt))
        })
        .into_handler(),
    )
    .await
    .unwrap();
    (addr.to_string(), handle)
}

#[tokio::test]
async fn v1_client_receives_a_receipt_over_the_wire() {
    let (addr, handle) = start_server().await;

    let outcome = query_remote_full(
        &addr,
        &RemoteQuery::new("SELECT n", "req-live-1").with_as_of(1234),
        None,
        None,
    )
    .await
    .expect("query succeeds");

    assert_eq!(outcome.rows(), 2);
    assert_eq!(outcome.protocol_version, 1);
    let receipt = outcome.receipt.expect("v1 server returns a receipt");
    assert_eq!(receipt.query_id, "req-live-1");
    assert_eq!(receipt.as_of, Some(1234));
    assert_eq!(receipt.tri_state, TriState::Found);
    assert!(receipt.validate().is_ok());

    handle.abort();
}

#[tokio::test]
async fn v0_client_still_works_against_a_v1_server() {
    let (addr, handle) = start_server().await;
    let batches = query_remote_with(&addr, "SELECT n", None, None)
        .await
        .expect("legacy path must keep working");
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    handle.abort();
}

/// The v0 wire has no `request_id`, and a receipt without one is invalid — so a
/// handler that validates (the real engine does) would reject every v0 call.
/// The service must derive an id rather than pass the empty string through.
#[tokio::test]
async fn a_v0_request_without_an_id_still_gets_a_traceable_receipt() {
    let (addr, handle) = bind(
        "127.0.0.1:0".parse().unwrap(),
        gtv_proto::QueryServiceImpl::from_fn(|q| async move {
            assert!(
                !q.request_id.is_empty(),
                "the service must derive an id for a v0 request"
            );
            let receipt = QueryReceipt::found(q.request_id.clone());
            receipt.validate().expect("derived id must validate");
            Ok((vec![batch(&[1])], receipt))
        })
        .into_handler(),
    )
    .await
    .unwrap();
    let addr = addr.to_string();

    // `to_legacy_request` deliberately drops everything but `sql`.
    let mut client = QueryServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let resp = client
        .execute(RemoteQuery::new("SELECT n", "").to_legacy_request())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(resp.error, "", "v0 must not fail for lack of a request id");

    let outcome = decode_response(resp).expect("decode");
    let receipt = outcome.receipt.expect("receipt travels in Arrow metadata");
    assert!(
        receipt.query_id.starts_with("v0-"),
        "derived ids are recognisable: {}",
        receipt.query_id
    );
    handle.abort();
}

#[tokio::test]
async fn a_derived_v0_id_is_stable_across_calls() {
    let ids = [
        gtv_proto::QueryServiceImpl::from_fn(|q| async move {
            Ok::<_, WireError>((vec![batch(&[1])], QueryReceipt::found(q.request_id)))
        }),
        gtv_proto::QueryServiceImpl::from_fn(|q| async move {
            Ok::<_, WireError>((vec![batch(&[1])], QueryReceipt::found(q.request_id)))
        }),
    ];
    // Same SQL through two independent services must yield the same id, so a
    // replay comparison is not fooled by a random suffix.
    let mut seen = Vec::new();
    for svc in ids {
        let (addr, handle) = bind("127.0.0.1:0".parse().unwrap(), svc.into_handler())
            .await
            .unwrap();
        let mut client = QueryServiceClient::connect(format!("http://{addr}"))
            .await
            .unwrap();
        let resp = client
            .execute(RemoteQuery::new("SELECT n", "").to_legacy_request())
            .await
            .unwrap()
            .into_inner();
        let outcome = decode_response(resp).unwrap();
        seen.push(outcome.receipt.unwrap().query_id);
        handle.abort();
    }
    assert_eq!(seen[0], seen[1]);
}

#[tokio::test]
async fn execute_query_refuses_a_legacy_version() {
    let (addr, handle) = start_server().await;
    let mut client = QueryServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let resp = client
        .execute_query(QueryRequest {
            sql: "SELECT n".into(),
            protocol_version: 0,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(resp.code, ErrorCode::UnsupportedVersion as i32);
    assert!(!resp.error.is_empty());
    handle.abort();
}

#[tokio::test]
async fn empty_result_is_no_match_not_found() {
    let (addr, handle) = start_server().await;
    let outcome = query_remote_full(&addr, &RemoteQuery::new("SELECT empty", "req-live-2"), None, None)
        .await
        .unwrap();
    assert!(outcome.is_conclusive());
    assert_eq!(outcome.receipt.unwrap().tri_state, TriState::NoMatch);
    handle.abort();
}

#[test]
fn freshness_survives_the_wire_and_the_arrow_payload() {
    // The watermark must reach a client that only reads Arrow, and a client
    // that reads `receipt_json`. Both paths are asserted here because the plan
    // (R1) requires either one to answer "as of when".
    let receipt = QueryReceipt::no_match("req-fresh")
        .with_as_of(95)
        .with_freshness(
            Freshness::new(90, KnowledgeClock::RecordedAt).with_history_floor(10),
        );
    let wire = encode_response(Ok((vec![batch(&[1, 2])], receipt.clone())));

    let outcome = decode_response(wire).unwrap();
    assert_eq!(outcome.rows(), 2);
    let back = outcome.receipt.expect("receipt travels");
    assert_eq!(back, receipt);
    assert_eq!(back.freshness.expect("watermark").as_known_at, 90);

    // Arrow-only reader: the envelope fields plus schema metadata.
    let ipc = gtv_proto::encode_response(Ok((vec![batch(&[1, 2])], receipt.clone()))).arrow_ipc;
    let from_arrow = receipt_from_ipc(&ipc).expect("readable").expect("present");
    assert_eq!(from_arrow, receipt);
}
