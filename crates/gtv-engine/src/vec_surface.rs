//! Shared plumbing for the vector SQL surfaces: `knn` / `ann` / `embedding_search`.
//!
//! **T5 decision (2026-11-15, Option C)**: keep the facades, unify the kernel.
//! The three surfaces intentionally differ in semantics —
//!
//! | surface | guarantee | filter | hardware | planning |
//! |:--|:--|:--|:--|:--|
//! | `knn` / `vector_search` | exact (flat scan; IVF probes its registered `nprobe`) | label string | AcceleratorRegistry (VRAM / IVF probe offload) | — |
//! | `ann` / `ann_search` | approximate + rerank (B3-3 strategy routing) | id allow-list | — | `ann_explain` / `cbo_explain` |
//! | `embedding_search` | exact (governed set) | tenant + `as_of` | — | — |
//!
//! but the mechanical work around a query — parsing a CSV query vector,
//! reporting a distance in SQL units, the `(id, distance)` schema, dim guards,
//! metric guards and the hits → `RecordBatch` conversion — is identical and
//! therefore lives here, so the facades cannot drift apart. The CPU search
//! kernels themselves are shared at the `gtv-index` layer (`exact_topk*`,
//! `execute_ann`); only `knn` carries the engine-level GPU overlay.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::error::{DataFusionError, Result};
use gtv_core::Metric;

/// Parse a comma-separated query vector (`"0.1,0.2"` → `[0.1, 0.2]`).
pub(crate) fn parse_query(surface: &str, query: &str) -> Result<Vec<f32>> {
    query
        .split(',')
        .map(|t| {
            t.trim().parse::<f32>().map_err(|_| {
                DataFusionError::Execution(format!("{surface}: bad query component `{t}`"))
            })
        })
        .collect()
}

/// Report a metric distance in SQL units: L2 keeps its historical Euclidean
/// (sqrt) report, cosine / dot report as-is.
pub(crate) fn report_distance(metric: Metric, raw: f32) -> f64 {
    match metric {
        Metric::L2 => (raw.max(0.0) as f64).sqrt(),
        _ => raw as f64,
    }
}

/// The `(id, distance)` schema every vector surface returns.
pub(crate) fn vec_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("distance", DataType::Float64, false),
    ]))
}

/// Reject a query whose dimension disagrees with the corpus.
pub(crate) fn check_dim(surface: &str, query: &[f32], dim: usize) -> Result<()> {
    if query.len() != dim {
        return Err(DataFusionError::Execution(format!(
            "{surface}: query dim {} != index dim {}",
            query.len(),
            dim
        )));
    }
    Ok(())
}

/// Parse a caller-requested metric and reject it if it disagrees with the
/// corpus metric. `None` means "use the collection's metric".
pub(crate) fn check_requested_metric(
    surface: &str,
    index_metric: Metric,
    requested: Option<&str>,
) -> Result<()> {
    if let Some(m) = requested {
        let metric = Metric::parse(m).ok_or_else(|| {
            DataFusionError::Execution(format!(
                "{surface}: unknown metric `{m}` (l2|cosine|dot)"
            ))
        })?;
        if metric != index_metric {
            return Err(DataFusionError::Execution(format!(
                "{surface}: metric mismatch — index is {}, query asked for {}",
                index_metric.as_str(),
                metric.as_str()
            )));
        }
    }
    Ok(())
}

/// Materialize `(id, distance)` hits as the single-batch table provider the
/// UDTFs return.
pub(crate) fn hits_table(hits: &[(u64, f64)]) -> Result<Arc<dyn TableProvider>> {
    let ids: Vec<u64> = hits.iter().map(|(id, _)| *id).collect();
    let dists: Vec<f64> = hits.iter().map(|(_, d)| *d).collect();
    let schema = vec_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(ids)) as ArrayRef,
            Arc::new(Float64Array::from(dists)) as ArrayRef,
        ],
    )?;
    Ok(Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_query_splits_and_trims() {
        assert_eq!(parse_query("knn", " 0.1 , 2 ,3.5").unwrap(), vec![0.1, 2.0, 3.5]);
        assert!(parse_query("knn", "0.1,oops").is_err());
    }

    #[test]
    fn report_distance_sqrt_only_for_l2() {
        assert!((report_distance(Metric::L2, 4.0) - 2.0).abs() < 1e-12);
        assert!((report_distance(Metric::Cosine, 0.25) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn metric_guard_accepts_match_and_rejects_mismatch() {
        assert!(check_requested_metric("knn", Metric::L2, None).is_ok());
        assert!(check_requested_metric("knn", Metric::L2, Some("l2")).is_ok());
        assert!(check_requested_metric("knn", Metric::L2, Some("cosine")).is_err());
        assert!(check_requested_metric("knn", Metric::L2, Some("nope")).is_err());
    }

    #[test]
    fn dim_guard_and_hits_table_round_trip() {
        assert!(check_dim("knn", &[1.0, 2.0], 2).is_ok());
        assert!(check_dim("knn", &[1.0], 2).is_err());

        let provider = hits_table(&[(7, 1.5), (9, 2.5)]).unwrap();
        assert_eq!(provider.schema().fields().len(), 2);
    }
}
