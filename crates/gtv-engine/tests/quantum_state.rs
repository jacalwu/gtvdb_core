//! `QUANTUM_STATE_*` SQL table functions, end to end.
//!
//! The claim this test protects is narrow and important: a state log written by a
//! backend can be interrogated **from SQL**, with the same numbers the Rust API
//! reports. No Python, no pickle, no bespoke viewer.

use arrow::array::{Array, Float64Array, Int64Array, StringArray, UInt64Array};
use arrow::record_batch::RecordBatch;
use gtv_engine::GtvContext;

/// GHZ-3 with a mid-circuit measurement, i.e. a tree with two roots-to-leaf paths.
const LOG: &str = "\
# ghz-3, job 4f2a
{\"state_id\":0,\"parent_state\":null,\"gate\":\"|000>\",\"timestamp\":0,\"probability\":1.0}
{\"state_id\":1,\"parent_state\":0,\"gate\":\"h q[0]\",\"timestamp\":1,\"probability\":1.0}
{\"state_id\":2,\"parent_state\":1,\"gate\":\"cx q[0],q[1]\",\"timestamp\":2,\"probability\":1.0}
{\"state_id\":3,\"parent_state\":2,\"gate\":\"cx q[0],q[2]\",\"timestamp\":3,\"probability\":1.0}
{\"state_id\":4,\"parent_state\":3,\"gate\":\"measure q[0]\",\"timestamp\":4,\"probability\":0.5}
{\"state_id\":5,\"parent_state\":3,\"gate\":\"measure q[0]\",\"timestamp\":4,\"probability\":0.5}
{\"state_id\":6,\"parent_state\":4,\"gate\":\"rx(0.785398) q[1]\",\"timestamp\":5,\"probability\":1.0}
{\"state_id\":7,\"parent_state\":6,\"gate\":\"measure q[1]\",\"timestamp\":6,\"probability\":0.8}
{\"state_id\":8,\"parent_state\":6,\"gate\":\"measure q[1]\",\"timestamp\":6,\"probability\":0.2}
";

fn log_path(name: &str) -> String {
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, LOG).expect("write log");
    path.to_string_lossy().into_owned()
}

fn rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

fn u64_col(batches: &[RecordBatch], name: &str) -> Vec<u64> {
    let mut out = Vec::new();
    for b in batches {
        let a = b
            .column_by_name(name)
            .unwrap_or_else(|| panic!("column {name}"))
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("UInt64");
        out.extend((0..b.num_rows()).map(|i| a.value(i)));
    }
    out
}

fn f64_col(batches: &[RecordBatch], name: &str) -> Vec<f64> {
    let mut out = Vec::new();
    for b in batches {
        let a = b
            .column_by_name(name)
            .unwrap_or_else(|| panic!("column {name}"))
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Float64");
        out.extend((0..b.num_rows()).map(|i| a.value(i)));
    }
    out
}

fn str_col(batches: &[RecordBatch], name: &str) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        let a = b
            .column_by_name(name)
            .unwrap_or_else(|| panic!("column {name}"))
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Utf8");
        out.extend((0..b.num_rows()).map(|i| a.value(i).to_string()));
    }
    out
}

fn i64_col(batches: &[RecordBatch], name: &str) -> Vec<i64> {
    let mut out = Vec::new();
    for b in batches {
        let a = b
            .column_by_name(name)
            .unwrap_or_else(|| panic!("column {name}"))
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64");
        out.extend((0..b.num_rows()).map(|i| a.value(i)));
    }
    out
}

async fn q(ctx: &GtvContext, sql: &str) -> Vec<RecordBatch> {
    ctx.sql(sql).await.expect(sql)
}

#[tokio::test]
async fn summary_describes_the_forest_from_sql() {
    let ctx = GtvContext::new();
    let path = log_path("gtv_e2e_summary.jsonl");
    let batches = q(
        &ctx,
        &format!("SELECT * FROM quantum_state_summary('{path}')"),
    )
    .await;

    assert_eq!(rows(&batches), 1);
    assert_eq!(u64_col(&batches, "states"), vec![9]);
    assert_eq!(u64_col(&batches, "transitions"), vec![8]);
    assert_eq!(u64_col(&batches, "leaves"), vec![3]);
    assert_eq!(u64_col(&batches, "max_depth"), vec![6]);
    assert_eq!(str_col(&batches, "semantics"), vec!["Conditional"]);
}

#[tokio::test]
async fn frontier_can_be_filtered_and_sorted_in_sql() {
    let ctx = GtvContext::new();
    let path = log_path("gtv_e2e_frontier.jsonl");

    // At t=4 there are two GHZ branches, 50/50.
    let batches = q(
        &ctx,
        &format!("SELECT state_id FROM quantum_state_frontier('{path}', 4)"),
    )
    .await;
    assert_eq!(u64_col(&batches, "state_id"), vec![4, 5]);

    // At t=6, all three leaves; the SQL ORDER BY must survive the table function.
    let batches = q(
        &ctx,
        &format!(
            "SELECT state_id, probability FROM quantum_state_frontier('{path}', 6) \
             ORDER BY probability DESC"
        ),
    )
    .await;
    assert_eq!(u64_col(&batches, "state_id"), vec![5, 7, 8]);
    let p = f64_col(&batches, "probability");
    assert!((p[0] - 0.5).abs() < 1e-12, "got {p:?}");
    assert!((p[1] - 0.4).abs() < 1e-12);
    assert!((p[2] - 0.1).abs() < 1e-12);

    // A 0.25 cut keeps the two dominant leaves and reports the retained mass.
    let batches = q(
        &ctx,
        &format!("SELECT * FROM quantum_state_frontier('{path}', 6, 0.25)"),
    )
    .await;
    assert_eq!(u64_col(&batches, "state_id"), vec![5, 7]);
    let explained = f64_col(&batches, "explained_mass");
    assert!(explained.iter().all(|m| (m - 0.9).abs() < 1e-12), "got {explained:?}");
    // Renormalised over the retained 0.9: 0.5/0.9 and 0.4/0.9.
    let renorm = f64_col(&batches, "probability_renormalized");
    assert!((renorm[0] - 0.5 / 0.9).abs() < 1e-12);
    assert!((renorm[1] - 0.4 / 0.9).abs() < 1e-12);
}

#[tokio::test]
async fn reconstruct_returns_the_gate_prefix_from_sql() {
    let ctx = GtvContext::new();
    let path = log_path("gtv_e2e_reconstruct.jsonl");
    let batches = q(
        &ctx,
        &format!("SELECT * FROM quantum_state_reconstruct('{path}', 8)"),
    )
    .await;

    assert_eq!(rows(&batches), 7, "root + 6 transitions");
    assert_eq!(
        str_col(&batches, "gate"),
        vec![
            "|000>",
            "h q[0]",
            "cx q[0],q[1]",
            "cx q[0],q[2]",
            "measure q[0]",
            "rx(0.785398) q[1]",
            "measure q[1]",
        ]
    );
    assert_eq!(i64_col(&batches, "timestamp"), vec![0, 1, 2, 3, 4, 5, 6]);

    // Prefix probabilities are non-increasing along a conditional path, and the
    // last one is the state's own probability.
    let p = f64_col(&batches, "prefix_probability");
    assert!((p[0] - 1.0).abs() < 1e-12);
    assert!(p.windows(2).all(|w| w[1] <= w[0] + 1e-12), "got {p:?}");
    assert!((p[p.len() - 1] - 0.1).abs() < 1e-12, "got {p:?}");
}

#[tokio::test]
async fn unknown_state_and_missing_file_are_errors_not_panics() {
    let ctx = GtvContext::new();
    let path = log_path("gtv_e2e_errors.jsonl");
    assert!(ctx
        .sql(&format!(
            "SELECT * FROM quantum_state_reconstruct('{path}', 999)"
        ))
        .await
        .is_err());
    assert!(ctx
        .sql("SELECT * FROM quantum_state_frontier('/no/such/log.jsonl', 1)")
        .await
        .is_err());
}

#[tokio::test]
async fn column_types_are_what_a_client_expects() {
    let ctx = GtvContext::new();
    let path = log_path("gtv_e2e_types.jsonl");
    let batches = q(
        &ctx,
        &format!("SELECT * FROM quantum_state_frontier('{path}', 6)"),
    )
    .await;
    let batch = &batches[0];
    assert_eq!(batch.num_columns(), 7);
    for b in batches.iter() {
        assert!(b.column(0).len() == b.num_rows());
    }
}
