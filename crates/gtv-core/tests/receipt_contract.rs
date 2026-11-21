//! Contract tests for the query receipt (A1) and deterministic evidence (A3).
//!
//! These tests are the executable form of the acceptance criteria:
//!
//! * same input ⇒ same output, regardless of production order or thread count
//!   (`topk_dedup_is_thread_count_invariant`);
//! * a receipt never claims completeness it cannot prove
//!   (`incomplete_requires_reason`, `conclusive_rejects_reason`);
//! * evidence resolves back to a physical row (`edge_ref_roundtrip`);
//! * the receipt survives every transport we ship (JSON, Arrow metadata).

use std::collections::HashMap;

use gtv_core::receipt::{
    is_supported_version, negotiate_version, plan_hash, topk_dedup, topk_dedup_by_score, Absence,
    DedupKey, EdgeRef, Freshness, FreshnessVerdict, Incomplete, IncompleteReason, KnowledgeClock,
    PathEvidence, QueryReceipt, QueryStats, MIN_SUPPORTED_PROTOCOL, PROTOCOL_VERSION,
    RECEIPT_METADATA_JSON, RECEIPT_METADATA_PREFIX,
};
use gtv_core::traversal::{TruncReason, Truncation, TriState};

#[test]
fn ring_key_is_rotation_invariant() {
    let a = DedupKey::ring([7, 3, 9]);
    let b = DedupKey::ring([3, 9, 7]);
    let c = DedupKey::ring([9, 7, 3]);
    assert_eq!(a, b);
    assert_eq!(b, c);
    // Canonical form starts at the smallest id.
    assert_eq!(a.nodes(), &[3, 9, 7]);
}

#[test]
fn path_key_keeps_direction() {
    assert_ne!(DedupKey::path([1, 2]), DedupKey::path([2, 1]));
    assert_eq!(DedupKey::path([1, 2]), DedupKey::path([1, 2]));
}

#[test]
fn topk_dedup_is_thread_count_invariant() {
    // 4,000 raw matches over 400 distinct rings: each ring found from every
    // rotation, plus noise, exactly what a parallel matcher emits.
    let mut raw: Vec<Vec<u64>> = Vec::new();
    for i in 0..400u64 {
        let base = [i, i + 1_000, i + 2_000];
        for rot in 0..3 {
            let mut v = Vec::with_capacity(3);
            for k in 0..3 {
                v.push(base[(rot + k) % 3]);
            }
            raw.push(v);
        }
        for dup in 0..7 {
            raw.push(vec![i, i + 1_000 + dup, i + 2_000]);
        }
    }

    let key = |m: &Vec<u64>| DedupKey::ring(m.iter().copied());

    // Single-threaded baseline.
    let baseline = topk_dedup(raw.clone(), 100, key);
    assert_eq!(baseline.len(), 100);

    // Four "threads" producing disjoint slices, merged in a scrambled order,
    // as a parallel matcher would.
    let chunk = raw.len() / 4;
    let mut partials: Vec<Vec<Vec<u64>>> = (0..4)
        .map(|t| {
            let lo = t * chunk;
            let hi = if t == 3 { raw.len() } else { lo + chunk };
            topk_dedup(raw[lo..hi].to_vec(), 100, key)
        })
        .collect();
    partials.swap(0, 2);
    partials.swap(1, 3);
    let merged = topk_dedup(partials.into_iter().flatten(), 100, key);

    assert_eq!(baseline, merged, "thread count changed the answer");
}

#[test]
fn topk_dedup_by_score_is_stable_and_dedups_to_best() {
    let items = vec![
        ("ring:1", 0.4),
        ("ring:1", 0.9),
        ("ring:2", 0.9),
        ("ring:2", 0.1),
        ("ring:3", 0.5),
    ];
    let out = topk_dedup_by_score(
        items,
        2,
        |(k, _)| DedupKey::explicit(*k),
        |(_, s)| *s,
    );
    assert_eq!(out.len(), 2);
    // Highest score first, ties by key.
    assert_eq!(out[0].0, "ring:1");
    assert_eq!(out[1].0, "ring:2");
}

#[test]
fn topk_zero_is_empty() {
    let out = topk_dedup(vec![vec![1u64, 2]], 0, |v: &Vec<u64>| {
        DedupKey::path(v.iter().copied())
    });
    assert!(out.is_empty());
}

#[test]
fn edge_ref_roundtrip() {
    let e = EdgeRef::new(42, 7);
    assert_eq!(e.to_string(), "42:7");
    assert_eq!("42:7".parse::<EdgeRef>().unwrap(), e);
    assert!("42".parse::<EdgeRef>().is_err());
    assert!("x:7".parse::<EdgeRef>().is_err());
    assert!(EdgeRef::from_slot(1, u32::MAX).is_none());
}

#[test]
fn path_evidence_validates_shape() {
    let ok = PathEvidence::new(
        vec![1, 2, 3],
        vec![EdgeRef::new(0, 10), EdgeRef::new(0, 11)],
        vec![100, 200],
        vec![150, 300],
    )
    .expect("well-formed path");
    assert_eq!(ok.span(), Some((100, 300)));
    assert!(!ok.is_open_ended());

    // Two nodes, one edge: fine. Two edges: rejected.
    assert!(PathEvidence::new(vec![1, 2], vec![EdgeRef::new(0, 1), EdgeRef::new(0, 2)], vec![0, 0], vec![1, 1]).is_err());
    // Inverted interval rejected.
    assert!(PathEvidence::new(vec![1], Vec::new(), Vec::new(), Vec::new()).is_ok());
    assert!(PathEvidence::new(
        vec![1, 2],
        vec![EdgeRef::new(0, 1)],
        vec![500],
        vec![100]
    )
    .is_err());
    // Mismatched validity arrays rejected.
    assert!(PathEvidence::new(vec![1, 2], vec![EdgeRef::new(0, 1)], vec![0, 0], vec![1]).is_err());
}

#[test]
fn path_fingerprint_is_stable_and_order_sensitive() {
    let a = PathEvidence::new(vec![1, 2], vec![EdgeRef::new(0, 1)], vec![0], vec![10]).unwrap();
    let b = a.clone();
    assert_eq!(a.fingerprint(), b.fingerprint());
    // Same node sequence, different edge: different fingerprint.
    let c = PathEvidence::new(vec![1, 2], vec![EdgeRef::new(0, 2)], vec![0], vec![10]).unwrap();
    assert_ne!(a.fingerprint(), c.fingerprint());
    // Pin the byte layout: a change here invalidates stored evidence ids.
    assert_eq!(a.fingerprint().len(), 16);
}

#[test]
fn incomplete_requires_reason() {
    let mut r = QueryReceipt::found("q-1");
    r.tri_state = TriState::Incomplete;
    r.incomplete = None;
    assert!(r.validate().is_err(), "Incomplete without reason must be rejected");
}

#[test]
fn conclusive_rejects_reason() {
    let mut r = QueryReceipt::found("q-2");
    r.incomplete = Some(Incomplete::new(IncompleteReason::Rows));
    assert!(r.validate().is_err());
}

#[test]
fn incomplete_constructor_is_valid_by_construction() {
    let r = QueryReceipt::incomplete(
        "q-3",
        Incomplete::new(IncompleteReason::Edges).at_hop(2).with_observation(10, 10),
    )
    .with_snapshot(7)
    .with_as_of(1_700_000_000);
    assert!(r.validate().is_ok());
    assert!(!r.is_complete());
}

#[test]
fn truncation_drives_tri_state() {
    let t = Truncation {
        reason: TruncReason::Deadline,
        hop: 3,
        frontier_at_stop: 55,
        limit: 60_000,
        observed: 60_001,
    };
    let r = QueryReceipt::found("q-4").with_truncation(Some(&t));
    assert_eq!(r.tri_state, TriState::Incomplete);
    let reason = r.incomplete.clone().expect("reason carried");
    assert_eq!(reason.reason, IncompleteReason::Deadline);
    assert_eq!(reason.hop, 3);
    assert_eq!(reason.limit, Some(60_000));
    assert!(r.validate().is_ok());

    // No truncation + rows ⇒ Found; no truncation + no rows ⇒ NoMatch.
    let found = QueryReceipt::no_match("q-5")
        .with_stats(QueryStats {
            rows: 4,
            ..Default::default()
        })
        .with_truncation(None);
    assert_eq!(found.tri_state, TriState::Found);
    assert!(found.validate().is_ok());
}

#[test]
fn receipt_roundtrips_through_json() {
    let r = QueryReceipt::incomplete("q-6", Incomplete::new(IncompleteReason::Limit))
        .with_snapshot(9)
        .with_as_of(123)
        .with_plan_hash(plan_hash("Projection\n  TableScan gtv_batch_khop"))
        .with_context_version("rules-2026.11");
    let back = QueryReceipt::from_json(&r.to_json()).expect("roundtrip");
    assert_eq!(r, back);
}

#[test]
fn receipt_roundtrips_through_arrow_metadata() {
    let r = QueryReceipt::found("q-7")
        .with_snapshot(3)
        .with_as_of(456)
        .with_stats(QueryStats {
            rows: 12,
            hops: 2,
            edges_scanned: 900,
            nodes_visited: 40,
            peak_frontier: 9,
        });
    let meta: HashMap<String, String> = r.to_arrow_metadata();
    assert_eq!(
        meta.get(&format!("gtv.receipt.{}", "tri_state")).map(String::as_str),
        Some("found")
    );
    let back = QueryReceipt::from_arrow_metadata(&meta)
        .expect("well-formed metadata")
        .expect("receipt present");
    assert_eq!(r, back);

    // A plain Arrow stream carries no receipt: absent, not an error.
    assert!(QueryReceipt::from_arrow_metadata(&HashMap::new())
        .expect("absent is not an error")
        .is_none());

    // Malformed JSON is an error, not silence.
    let mut broken = HashMap::new();
    broken.insert(RECEIPT_METADATA_JSON.to_string(), "{".to_string());
    assert!(QueryReceipt::from_arrow_metadata(&broken).is_err());
}

#[test]
fn protocol_version_is_negotiable_and_self_consistent() {
    assert!(is_supported_version(PROTOCOL_VERSION));
    assert!(is_supported_version(MIN_SUPPORTED_PROTOCOL));
    assert_eq!(negotiate_version(PROTOCOL_VERSION), Some(PROTOCOL_VERSION));
    // A newer peer can be served at our ceiling.
    assert_eq!(negotiate_version(PROTOCOL_VERSION + 5), Some(PROTOCOL_VERSION));
    // Too old: refuse rather than silently misread.
    assert_eq!(negotiate_version(0), None);

    let mut r = QueryReceipt::found("q-8");
    r.protocol_version = PROTOCOL_VERSION + 5;
    assert!(r.validate().is_err());
}

#[test]
fn partial_rows_never_leave_a_standard_path() {
    assert!(gtv_core::receipt::allows_partial_rows(TriState::Found));
    assert!(gtv_core::receipt::allows_partial_rows(TriState::NoMatch));
    assert!(!gtv_core::receipt::allows_partial_rows(TriState::Incomplete));
}

#[test]
fn plan_hash_is_deterministic() {
    let a = plan_hash("HashJoin Executor\n  TableScan gtv_batch_khop");
    let b = plan_hash("HashJoin Executor\n  TableScan gtv_batch_khop");
    assert_eq!(a, b);
    assert_ne!(a, plan_hash("HashJoin Executor\n  TableScan gtv_motif_matches"));
}

// ---------------------------------------------------------------------------
// Freshness: "no such row" versus "not refreshed yet" (plan R1 / R2 / R3)
// ---------------------------------------------------------------------------

fn freshness(known_at: i64) -> Freshness {
    Freshness::new(known_at, KnowledgeClock::SegmentSystemTs)
}

#[test]
fn watermark_roundtrips_through_every_transport() {
    let receipt = QueryReceipt::no_match("q-f1")
        .with_as_of(100)
        .with_freshness(freshness(90).with_history_floor(10).with_business_max(120));

    let json = QueryReceipt::from_json(&receipt.to_json()).expect("json roundtrip");
    assert_eq!(json, receipt);

    let meta = receipt.to_arrow_metadata();
    assert_eq!(
        meta.get(&format!("{RECEIPT_METADATA_PREFIX}as_known_at")).map(String::as_str),
        Some("90")
    );
    assert_eq!(
        meta.get(&format!("{RECEIPT_METADATA_PREFIX}knowledge_clock")).map(String::as_str),
        Some("segment_system_ts")
    );
    assert_eq!(
        meta.get(&format!("{RECEIPT_METADATA_PREFIX}history_floor")).map(String::as_str),
        Some("10")
    );
    let back = QueryReceipt::from_arrow_metadata(&meta).expect("readable").expect("present");
    assert_eq!(back, receipt);
}

#[test]
fn stale_watermark_turns_no_match_into_not_yet_refreshed() {
    // Report asked at knowledge time 100; the graph was refreshed at 90.
    let receipt = QueryReceipt::no_match("q-f2")
        .with_as_of(95)
        .with_freshness(freshness(90));

    assert_eq!(
        receipt.absence_at(100, Some(95)),
        Absence::NotYetRefreshed { behind_by: 10 }
    );
    assert!(!receipt.proves_absence_at(100, Some(95)));
    // Read from its own watermark, the same result is conclusive.
    assert_eq!(receipt.absence_at(90, Some(95)), Absence::Proven);
}

#[test]
fn a_missing_watermark_is_never_a_proof_of_absence() {
    let receipt = QueryReceipt::no_match("q-f3").with_as_of(95);
    assert_eq!(receipt.freshness_verdict(100), None);
    assert_eq!(receipt.absence_at(100, Some(95)), Absence::NotApplicable);
    assert!(!receipt.proves_absence_at(100, Some(95)));
}

#[test]
fn history_below_the_gc_floor_is_unanswerable_not_empty() {
    let receipt = QueryReceipt::no_match("q-f4")
        .with_as_of(5)
        .with_freshness(freshness(90).with_history_floor(50));
    assert_eq!(
        receipt.absence_at(90, Some(5)),
        Absence::HistoryCollected { floor: 50 }
    );
    assert!(!receipt.proves_absence_at(90, Some(5)));
}

#[test]
fn rows_or_truncation_make_the_absence_question_moot() {
    let found = QueryReceipt::found("q-f5")
        .with_as_of(95)
        .with_freshness(freshness(90));
    assert_eq!(found.absence_at(100, Some(95)), Absence::NotApplicable);

    let truncated = QueryReceipt::incomplete(
        "q-f6",
        Incomplete::new(IncompleteReason::Edges).at_hop(2),
    )
    .with_as_of(95)
    .with_freshness(freshness(90));
    // Even at a current watermark, a capped search cannot prove absence.
    assert_eq!(truncated.absence_at(90, Some(95)), Absence::NotApplicable);
}

#[test]
fn validate_rejects_absence_below_the_retention_floor() {
    let mut r = QueryReceipt::no_match("q-f7")
        .with_as_of(5)
        .with_freshness(freshness(90).with_history_floor(50));
    assert!(r.validate().is_err());

    // The same receipt with rows is consistent: the rows exist.
    r.tri_state = TriState::Found;
    assert!(r.validate().is_ok());
}

#[test]
fn validate_rejects_a_floor_after_the_watermark() {
    let r = QueryReceipt::found("q-f8").with_freshness(freshness(10).with_history_floor(50));
    assert!(r.validate().is_err());
}

#[test]
fn freshness_is_deterministic_across_replays() {
    let build = || {
        QueryReceipt::found("q-f9")
            .with_as_of(95)
            .with_freshness(freshness(90).with_history_floor(10))
    };
    assert_eq!(build().to_json(), build().to_json());
}

#[test]
fn verdict_reports_the_lag() {
    let f = freshness(90);
    assert!(f.verdict(80).is_current());
    assert_eq!(f.lag(80), None);
    assert_eq!(f.verdict(95), FreshnessVerdict::Stale { behind_by: 5 });
    assert!(f.retains_business_time(5));
    assert!(!f.with_history_floor(50).retains_business_time(5));
}
