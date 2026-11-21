//! Query-level receipt plumbing (A1).
//!
//! [`crate::motif`] attaches a receipt to graph operators; this module attaches
//! one to **any** SQL statement and makes it ride along with the result.
//!
//! Two rules shape the code here:
//!
//! * the receipt is metadata on the Arrow schema, not a side channel — a client
//!   that only knows how to read Arrow still carries the receipt home;
//! * a statement whose result was truncated must not look conclusive, so the
//!   tri-state is derived from the tables the statement read, not from
//!   `rows > 0` alone (see [`crate::GtvContext::sql_with_receipt`]).

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use gtv_core::{plan_hash, Incomplete, QueryReceipt, QueryStats, TriState};

use crate::motif::MotifSummary;

/// The time points a statement was requested at, and who enforced them.
///
/// This exists because a handler can receive an `as_of` and have no way to make
/// the statement answer for it: `SELECT * FROM prices` has no time predicate,
/// and rewriting it is not something a transport layer may do silently. Passing
/// the request straight into a receipt produces rows stamped with a time
/// nobody checked — see [`PointBinding`].
///
/// `Default` is "nothing requested", and it is the only state a handler gets
/// for free: both applied-flags must be set deliberately, by code that actually
/// enforced the point.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueryBinding {
    pub as_of: Option<i64>,
    /// True only when the engine filtered by `as_of` (or a session default
    /// reached every operator the statement touched).
    pub as_of_applied: bool,
    pub snapshot_id: Option<u64>,
    /// True only when `snapshot_id` was compared against the store's live
    /// snapshot before the query ran.
    pub snapshot_verified: bool,
}

impl QueryBinding {
    /// Nothing requested: the receipt will name no time point, which is honest
    /// and conclusive.
    pub fn none() -> Self {
        Self::default()
    }

    /// A business time the caller asked for, not (yet) enforced.
    pub fn at(as_of: i64) -> Self {
        Self {
            as_of: Some(as_of),
            ..Self::default()
        }
    }

    /// A snapshot the caller pinned, not (yet) verified.
    pub fn snapshot(snapshot_id: u64) -> Self {
        Self {
            snapshot_id: Some(snapshot_id),
            ..Self::default()
        }
    }

    /// Mark the business time as enforced. Call this only from code that
    /// actually applied it.
    pub fn applied(mut self) -> Self {
        self.as_of_applied = self.as_of.is_some();
        self
    }

    /// Pin a snapshot the caller asked for, not (yet) verified.
    pub fn with_snapshot(mut self, snapshot_id: u64) -> Self {
        self.snapshot_id = Some(snapshot_id);
        self
    }

    /// Mark the snapshot as verified against the store head.
    pub fn verified(mut self) -> Self {
        self.snapshot_verified = self.snapshot_id.is_some();
        self
    }

    /// True when a point is named that nothing enforced, which caps the
    /// tri-state at `Incomplete`.
    pub fn declares_an_unenforced_point(&self) -> bool {
        (self.as_of.is_some() && !self.as_of_applied)
            || (self.snapshot_id.is_some() && !self.snapshot_verified)
    }
}

/// A statement's rows plus the receipt that explains them.
#[derive(Debug, Clone)]
pub struct QueryOutcome {
    pub batches: Vec<RecordBatch>,
    pub receipt: QueryReceipt,
}

impl QueryOutcome {
    pub fn rows(&self) -> usize {
        self.batches.iter().map(|b| b.num_rows()).sum()
    }
}

/// Number of result rows, saturating at `u64::MAX`.
pub fn count_rows(batches: &[RecordBatch]) -> u64 {
    batches
        .iter()
        .fold(0u64, |acc, b| acc.saturating_add(b.num_rows() as u64))
}

/// Hash a rendered logical plan so a receipt can be compared across runs.
pub fn hash_plan(rendered: &str) -> String {
    plan_hash(rendered)
}

/// Rewrite every batch's schema to carry `receipt` as metadata.
///
/// Existing metadata is preserved and the receipt keys are merged on top, so a
/// caller that already stamped its own provenance does not lose it. Returns the
/// batches unchanged when there are none.
pub fn attach_receipt(
    batches: Vec<RecordBatch>,
    receipt: &QueryReceipt,
) -> arrow::error::Result<Vec<RecordBatch>> {
    let meta = receipt.to_arrow_metadata();
    batches
        .into_iter()
        .map(|batch| {
            let schema = batch.schema();
            let mut merged: HashMap<String, String> = schema.metadata().clone();
            merged.extend(meta.clone());
            let new_schema = Arc::new(Schema::new_with_metadata(
                schema.fields().clone(),
                merged,
            ));
            RecordBatch::try_new(new_schema, batch.columns().to_vec())
        })
        .collect()
}

/// Read a receipt back from the first batch's schema metadata.
///
/// `None` means the stream carries no receipt (a plain Arrow result); an `Err`
/// means a receipt is present but malformed, which callers must not treat as
/// "no receipt".
pub fn receipt_from_batches(batches: &[RecordBatch]) -> Option<gtv_core::Result<QueryReceipt>> {
    let first = batches.first()?;
    match QueryReceipt::from_arrow_metadata(first.schema().metadata()) {
        Ok(Some(receipt)) => Some(Ok(receipt)),
        Ok(None) => None,
        Err(e) => Some(Err(e)),
    }
}

/// Derive a receipt for a statement from its rows and the motif summaries it
/// depended on.
///
/// If any dependency was truncated the whole statement is `Incomplete`: a join
/// against a partial table produces a partial answer even when the joined
/// result has rows. The first (deterministic, by name) truncated dependency
/// supplies the reason, so re-running the same statement yields the same
/// receipt.
///
/// A time point in `binding` that nothing enforced also caps the result at
/// `Incomplete`: the rows are real, but they are not an answer about the point
/// that was named. Truncation keeps the reason when both apply, because it says
/// something about the data rather than about the question.
pub fn receipt_for_statement(
    query_id: impl Into<String>,
    batches: &[RecordBatch],
    binding: QueryBinding,
    rendered_plan: &str,
    summaries: &[(String, MotifSummary)],
) -> QueryReceipt {
    let mut sorted: Vec<&(String, MotifSummary)> = summaries.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let truncated = sorted.iter().find(|(_, s)| !s.is_complete());

    let rows = count_rows(batches);
    let stats = QueryStats {
        rows,
        ..sorted
            .iter()
            .fold(QueryStats::default(), |acc, (_, s)| QueryStats {
                hops: acc.hops.max(s.stats.hops),
                edges_scanned: acc.edges_scanned.saturating_add(s.stats.edges_scanned),
                nodes_visited: acc.nodes_visited.max(s.stats.nodes_visited),
                peak_frontier: acc.peak_frontier.max(s.stats.peak_frontier),
                rows,
            })
    };

    let state = if truncated.is_some() {
        TriState::Incomplete
    } else if rows > 0 {
        TriState::Found
    } else {
        TriState::NoMatch
    };

    let mut receipt = QueryReceipt::new(query_id, state)
        .with_stats(stats)
        .with_plan_hash(hash_plan(rendered_plan));
    if let Some(as_of) = binding.as_of {
        receipt = if binding.as_of_applied {
            receipt.with_as_of_applied(as_of)
        } else {
            receipt.with_as_of(as_of)
        };
    }
    if let Some(id) = binding.snapshot_id {
        receipt = if binding.snapshot_verified {
            receipt.with_snapshot_verified(id)
        } else {
            receipt.with_snapshot(id)
        };
    }
    if let Some((name, summary)) = truncated {
        let mut incomplete = summary
            .incomplete
            .clone()
            .unwrap_or_else(|| Incomplete::new(gtv_core::IncompleteReason::Limit));
        // Record which operator produced the partial input; the receipt's
        // `context_version` is reserved for caller policy, so the operator name
        // is folded into the plan hash instead of overloading another field.
        receipt.plan_hash = Some(format!(
            "{}@{}",
            receipt.plan_hash.unwrap_or_default(),
            name
        ));
        incomplete.hop = incomplete.hop.max(1);
        receipt.incomplete = Some(incomplete);
    }
    // A point nothing enforced cannot leave the answer conclusive: the rows may
    // come from any state, so `Found`/`NoMatch` would be a claim about a state
    // nobody read. Truncation keeps the reason when both apply.
    if receipt.tri_state != TriState::Incomplete && binding.declares_an_unenforced_point() {
        receipt.tri_state = TriState::Incomplete;
        receipt.incomplete = Some(Incomplete::new(
            gtv_core::IncompleteReason::TimePointNotApplied,
        ));
    }
    receipt
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::UInt64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use gtv_core::{IncompleteReason, TraversalStats};

    fn batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::UInt64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(UInt64Array::from(vec![1u64, 2]))]).unwrap()
    }

    #[test]
    fn receipt_rides_on_schema_metadata() {
        let receipt = QueryReceipt::found("q-1").with_snapshot(4);
        let out = attach_receipt(vec![batch()], &receipt).unwrap();
        assert_eq!(out[0].num_rows(), 2);
        let back = receipt_from_batches(&out).expect("present").unwrap();
        assert_eq!(back, receipt);
    }

    #[test]
    fn existing_metadata_survives() {
        let schema = Arc::new(Schema::new_with_metadata(
            vec![Field::new("n", DataType::UInt64, false)],
            HashMap::from([("owner".to_string(), "risk".to_string())]),
        ));
        let b = RecordBatch::try_new(schema, vec![Arc::new(UInt64Array::from(vec![1u64]))]).unwrap();
        let out = attach_receipt(vec![b], &QueryReceipt::found("q-2")).unwrap();
        assert_eq!(out[0].schema().metadata().get("owner").map(String::as_str), Some("risk"));
        assert!(out[0].schema().metadata().contains_key("gtv.receipt.json"));
    }

    #[test]
    fn plain_stream_has_no_receipt() {
        assert!(receipt_from_batches(&[batch()]).is_none());
    }

    fn summary(tri: TriState, edges: u64) -> MotifSummary {
        MotifSummary {
            tri_state: tri,
            incomplete: (tri == TriState::Incomplete)
                .then(|| Incomplete::new(IncompleteReason::Edges).at_hop(2)),
            stats: TraversalStats {
                rows: 3,
                hops: 2,
                edges_scanned: edges,
                ..Default::default()
            },
            valid_at: 100,
            limit: 10,
        }
    }

    #[test]
    fn truncated_dependency_makes_the_statement_incomplete() {
        let r = receipt_for_statement(
            "q-3",
            &[batch()],
            QueryBinding::at(100).with_snapshot(7),
            "Join",
            &[("gtv_motif_matches".to_string(), summary(TriState::Incomplete, 50))],
        );
        assert_eq!(r.tri_state, TriState::Incomplete);
        assert_eq!(r.incomplete.as_ref().unwrap().reason, IncompleteReason::Edges);
        assert!(r.plan_hash.as_ref().unwrap().ends_with("@gtv_motif_matches"));
        assert!(r.validate().is_ok());
    }

    #[test]
    fn a_declared_point_caps_the_tri_state() {
        // Rows came back, but they were not filtered by `as_of`, so the answer
        // is not an answer about 100.
        let declared = receipt_for_statement(
            "q-6",
            &[batch()],
            QueryBinding::at(100),
            "Scan",
            &[],
        );
        assert_eq!(declared.tri_state, TriState::Incomplete);
        assert_eq!(
            declared.incomplete.as_ref().unwrap().reason,
            IncompleteReason::TimePointNotApplied
        );
        assert!(declared.validate().is_ok(), "an honest receipt is a valid one");

        // The same rows, with the point actually enforced, are conclusive.
        let applied =
            receipt_for_statement("q-7", &[batch()], QueryBinding::at(100).applied(), "Scan", &[]);
        assert_eq!(applied.tri_state, TriState::Found);
        assert!(applied.as_of_applied());
        assert!(applied.incomplete.is_none());
    }

    #[test]
    fn truncation_explains_a_declared_point_better_than_the_binding_can() {
        let r = receipt_for_statement(
            "q-8",
            &[batch()],
            QueryBinding::at(100),
            "Join",
            &[("gtv_motif_matches".to_string(), summary(TriState::Incomplete, 50))],
        );
        assert_eq!(r.incomplete.as_ref().unwrap().reason, IncompleteReason::Edges);
        // The binding still records the unenforced point, so a reader can see
        // both facts.
        assert!(!r.as_of_applied());
    }

    #[test]
    fn conclusive_dependency_leaves_rows_decisive() {
        let r = receipt_for_statement(
            "q-4",
            &[batch()],
            QueryBinding::none(),
            "Join",
            &[("gtv_motif_matches".to_string(), summary(TriState::Found, 50))],
        );
        assert_eq!(r.tri_state, TriState::Found);
        assert!(r.incomplete.is_none());
        assert_eq!(r.stats.rows, 2);

        let empty = receipt_for_statement("q-5", &[], QueryBinding::none(), "Empty", &[]);
        assert_eq!(empty.tri_state, TriState::NoMatch);
        assert!(empty.validate().is_ok());
    }
}
