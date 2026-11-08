//! Quantum state graph — the `QUANTUM_STATE_*` SQL surface.
//!
//! A state log (see [`gtv_core::quantum_state`]) is a temporal provenance forest:
//! one row per state, one parent each, monotone timestamps, a branch weight.
//! Exposing it to SQL turns "why is the run in this state" into a query instead
//! of a Python session over a pickle file.
//!
//! | Function | Signature | Rows |
//! |:--|:--|:--|
//! | `quantum_state_summary` | `(log_path)` | shape of the forest |
//! | `quantum_state_frontier` | `(log_path, at [, min_prob])` | live branches at `at` |
//! | `quantum_state_reconstruct` | `(log_path, state_id)` | root-path gate sequence |
//!
//! `log_path` is a JSONL file in the record format documented on
//! [`QuantumStateRecord`]; `#` comments are allowed so the file can carry its
//! own provenance.
//!
//! ```sql
//! SELECT * FROM quantum_state_summary('runs/ghz3.jsonl');
//!
//! -- which branches survive step 6, and with how much mass?
//! SELECT state_id, gate, probability
//! FROM quantum_state_frontier('runs/ghz3.jsonl', 6, 0.05)
//! ORDER BY probability DESC;
//!
//! -- the exact gate prefix behind the unlikely branch
//! SELECT step, gate, prefix_probability
//! FROM quantum_state_reconstruct('runs/ghz3.jsonl', 8);
//! ```

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::{TableFunctionArgs, TableFunctionImpl};
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::error::{DataFusionError, Result};

use gtv_core::quantum_state::QuantumStateLog;

use crate::expr_util::{expr_to_f64, expr_to_i64, expr_to_string, expr_to_u64};

/// Load a state log from a JSONL path, mapping every failure to a SQL error.
fn load(path: &str) -> Result<QuantumStateLog> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| DataFusionError::Execution(format!("state log `{path}`: {e}")))?;
    QuantumStateLog::from_jsonl(&text)
        .map_err(|e| DataFusionError::Execution(format!("state log `{path}`: {e}")))
}

fn table(schema: SchemaRef, batch: RecordBatch) -> Result<Arc<dyn TableProvider>> {
    Ok(Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))
}

fn missing(fn_name: &str, arg: &str) -> DataFusionError {
    DataFusionError::Execution(format!("{fn_name}: missing argument `{arg}`"))
}

// ---------------------------------------------------------------------------
// quantum_state_summary
// ---------------------------------------------------------------------------

/// `quantum_state_summary(log_path)` — one row describing the forest.
#[derive(Debug)]
pub struct QuantumStateSummaryTableFunction;

impl TableFunctionImpl for QuantumStateSummaryTableFunction {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let exprs = args.exprs();
        let path = expr_to_string(
            exprs
                .first()
                .ok_or_else(|| missing("quantum_state_summary", "log_path"))?,
        )?;
        let log = load(&path)?;
        let stats = log.graph().stats();

        let schema = Arc::new(Schema::new(vec![
            Field::new("states", DataType::UInt64, false),
            Field::new("transitions", DataType::UInt64, false),
            Field::new("roots", DataType::UInt64, false),
            Field::new("leaves", DataType::UInt64, false),
            Field::new("max_depth", DataType::UInt64, false),
            Field::new("mean_branching", DataType::Float64, false),
            Field::new("semantics", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![stats.nodes as u64])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![stats.edges as u64])),
                Arc::new(UInt64Array::from(vec![stats.roots as u64])),
                Arc::new(UInt64Array::from(vec![stats.leaves as u64])),
                Arc::new(UInt64Array::from(vec![stats.max_depth as u64])),
                Arc::new(Float64Array::from(vec![stats.mean_branching])),
                Arc::new(StringArray::from(vec![format!("{:?}", log.semantics())])),
            ],
        )?;
        table(schema, batch)
    }
}

// ---------------------------------------------------------------------------
// quantum_state_frontier
// ---------------------------------------------------------------------------

/// `quantum_state_frontier(log_path, at [, min_prob])` — live branches at `at`.
///
/// `probability_renormalized` is relative to the mass retained by the cut, so a
/// caller can see both "how likely in absolute terms" and "share of what is
/// left". Always read `probability` first; the normalised column is a view, not
/// a fact.
#[derive(Debug)]
pub struct QuantumStateFrontierTableFunction;

impl TableFunctionImpl for QuantumStateFrontierTableFunction {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let exprs = args.exprs();
        let path = expr_to_string(
            exprs
                .first()
                .ok_or_else(|| missing("quantum_state_frontier", "log_path"))?,
        )?;
        let at = expr_to_i64(
            exprs
                .get(1)
                .ok_or_else(|| missing("quantum_state_frontier", "at"))?,
        )?;
        let min_prob = match exprs.get(2) {
            Some(e) => expr_to_f64(e)?,
            None => 0.0,
        };

        let log = load(&path)?;
        let set = log.dominant(at, min_prob);
        let renorm: std::collections::HashMap<u64, f64> = set.renormalized().into_iter().collect();

        let schema = Arc::new(Schema::new(vec![
            Field::new("state_id", DataType::UInt64, false),
            Field::new("gate", DataType::Utf8, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("depth", DataType::UInt64, false),
            Field::new("probability", DataType::Float64, false),
            Field::new("probability_renormalized", DataType::Float64, false),
            Field::new("explained_mass", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(
                    set.branches.iter().map(|b| b.state_id).collect::<Vec<_>>(),
                )) as ArrayRef,
                Arc::new(StringArray::from(
                    set.branches
                        .iter()
                        .map(|b| b.gate.to_string())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(
                    set.branches.iter().map(|b| b.timestamp).collect::<Vec<_>>(),
                )),
                Arc::new(UInt64Array::from(
                    set.branches.iter().map(|b| b.depth as u64).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    set.branches
                        .iter()
                        .map(|b| b.probability)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    set.branches
                        .iter()
                        .map(|b| *renorm.get(&b.state_id).unwrap_or(&0.0))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(vec![set.explained; set.branches.len()])),
            ],
        )?;
        table(schema, batch)
    }
}

// ---------------------------------------------------------------------------
// quantum_state_reconstruct
// ---------------------------------------------------------------------------

/// `quantum_state_reconstruct(log_path, state_id)` — root path to a state.
///
/// One row per gate, root first. `prefix_probability` is the probability of the
/// prefix ending at that row, so the last row equals the state's probability;
/// reading it column-wise shows where a branch started to become unlikely.
#[derive(Debug)]
pub struct QuantumStateReconstructTableFunction;

impl TableFunctionImpl for QuantumStateReconstructTableFunction {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let exprs = args.exprs();
        let path = expr_to_string(
            exprs
                .first()
                .ok_or_else(|| missing("quantum_state_reconstruct", "log_path"))?,
        )?;
        let state_id = expr_to_u64(
            exprs
                .get(1)
                .ok_or_else(|| missing("quantum_state_reconstruct", "state_id"))?,
        )?;

        let log = load(&path)?;
        let r = log
            .reconstruct(state_id)
            .map_err(|e| DataFusionError::Execution(format!("quantum_state_reconstruct: {e}")))?;

        let prefixes: Vec<f64> = r
            .path
            .iter()
            .map(|&id| log.graph().probability(id).unwrap_or(0.0))
            .collect();

        let schema = Arc::new(Schema::new(vec![
            Field::new("step", DataType::UInt64, false),
            Field::new("state_id", DataType::UInt64, false),
            Field::new("gate", DataType::Utf8, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("prefix_probability", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(
                    (0..r.path.len() as u64).collect::<Vec<_>>(),
                )) as ArrayRef,
                Arc::new(UInt64Array::from(r.path.clone())),
                Arc::new(StringArray::from(
                    r.gates.iter().map(|g| g.to_string()).collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(r.timestamps.clone())),
                Arc::new(Float64Array::from(prefixes)),
            ],
        )?;
        table(schema, batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG: &str = "# ghz-3\n\
{\"state_id\":0,\"parent_state\":null,\"gate\":\"|000>\",\"timestamp\":0,\"probability\":1.0}\n\
{\"state_id\":1,\"parent_state\":0,\"gate\":\"h q[0]\",\"timestamp\":1,\"probability\":1.0}\n\
{\"state_id\":2,\"parent_state\":1,\"gate\":\"measure q[0]\",\"timestamp\":2,\"probability\":0.25}\n\
{\"state_id\":3,\"parent_state\":1,\"gate\":\"measure q[0]\",\"timestamp\":2,\"probability\":0.75}\n";

    fn write_log(name: &str) -> String {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, LOG).expect("write log");
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn summary_reports_shape() {
        let path = write_log("gtv_state_summary.jsonl");
        let log = load(&path).unwrap();
        let stats = log.graph().stats();
        assert_eq!((stats.nodes, stats.edges, stats.roots, stats.leaves), (4, 3, 1, 2));
        assert_eq!(stats.max_depth, 2);
    }

    #[test]
    fn reconstruct_walks_prefix_probabilities() {
        let path = write_log("gtv_state_reconstruct.jsonl");
        let log = load(&path).unwrap();
        let r = log.reconstruct(3).unwrap();
        assert_eq!(r.circuit(), "|000>; h q[0]; measure q[0]");
        let prefixes: Vec<f64> = r
            .path
            .iter()
            .map(|&id| log.graph().probability(id).unwrap())
            .collect();
        assert!((prefixes[0] - 1.0).abs() < 1e-12);
        assert!((prefixes[2] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn frontier_and_pruning_agree_with_the_kernel() {
        let path = write_log("gtv_state_frontier.jsonl");
        let log = load(&path).unwrap();
        assert_eq!(log.dominant(2, 0.0).branches.len(), 2);
        let pruned = log.dominant(2, 0.5);
        assert_eq!(pruned.branches.len(), 1);
        assert_eq!(pruned.branches[0].state_id, 3);
        assert!((pruned.explained - 0.75).abs() < 1e-12);
    }

    #[test]
    fn missing_file_is_a_sql_error_not_a_panic() {
        assert!(load("/definitely/not/here.jsonl").is_err());
    }
}
