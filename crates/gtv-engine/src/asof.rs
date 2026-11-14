//! kdb+-style `asof` join exposed to SQL as a DataFusion table function.
//!
//! `asof_join(t0, t1, ...)` matches each left time against a right-side series
//! captured at registration time. Two shapes are supported:
//!   * single-column: `(t, value)`
//!   * multi-column + tolerance: `(t, price, spread)` where a match is dropped
//!     when `left_t - matched_right_t > tolerance`.
//!
//! Matching is one monotonic sweep ([`gtv_accelerator::asof_match`]) rather than
//! a binary search per left row: left rows are visited in timestamp order and a
//! single right cursor only ever advances, so the whole join is `O(L + R)`. When
//! the caller's left column is unsorted the rows are ordered through an index
//! permutation and scattered back, which keeps the output aligned with the input
//! for one `O(L log L)` sort instead of `L` independent probes.
//!
//! The right-side series is also eligible for the device cache: it is uploaded
//! once per version and every later query downloads only one matched index per
//! left row, never the carried price/spread payload.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::{TableFunctionArgs, TableFunctionImpl};
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::error::Result;
use gtv_accelerator::{AcceleratorRegistry, AsofRequest};

use crate::expr_util::expr_to_i64;

/// Table function over a snapshot of a right-side time series.
#[derive(Debug)]
pub struct AsofJoinTableFunction {
    right_times: Vec<i64>,
    right_price: Vec<f64>,
    /// When `Some`, the table function returns `(t, price, spread)` instead of
    /// `(t, value)` and applies the tolerance window.
    right_spread: Option<Vec<f64>>,
    tolerance_ns: Option<i64>,
    /// Shared device cache; always present, host-only when no GPU is engaged.
    accelerators: Arc<AcceleratorRegistry>,
    /// Cache version for this snapshot. A re-registration gets a new one, which
    /// invalidates the device copy.
    version: u64,
}

impl AsofJoinTableFunction {
    /// Single-column as-of join (backward compatible): `(t, value)`.
    pub fn new(
        right_times: Vec<i64>,
        right_values: Vec<f64>,
        accelerators: Arc<AcceleratorRegistry>,
    ) -> Self {
        debug_assert_eq!(right_times.len(), right_values.len());
        let version = accelerators.next_version();
        Self {
            right_times,
            right_price: right_values,
            right_spread: None,
            tolerance_ns: None,
            accelerators,
            version,
        }
    }

    /// Multi-column as-of join with tolerance: `(t, price, spread)`.
    pub fn new_multi(
        right_times: Vec<i64>,
        right_price: Vec<f64>,
        right_spread: Vec<f64>,
        tolerance_ns: i64,
        accelerators: Arc<AcceleratorRegistry>,
    ) -> Self {
        debug_assert_eq!(right_times.len(), right_price.len());
        debug_assert_eq!(right_times.len(), right_spread.len());
        let version = accelerators.next_version();
        Self {
            right_times,
            right_price,
            right_spread: Some(right_spread),
            tolerance_ns: Some(tolerance_ns),
            accelerators,
            version,
        }
    }

    /// Bytes the carried payload (price + spread) occupies on the device.
    fn payload_bytes(&self) -> u64 {
        let price = self.right_price.len() as u64 * 8;
        let spread = self.right_spread.as_ref().map_or(0, |s| s.len() as u64 * 8);
        price + spread
    }

    fn schema(&self) -> SchemaRef {
        let mut fields = vec![
            Field::new("t", DataType::Int64, false),
            Field::new(
                if self.right_spread.is_some() { "price" } else { "value" },
                DataType::Float64,
                true,
            ),
        ];
        if self.right_spread.is_some() {
            fields.push(Field::new("spread", DataType::Float64, true));
        }
        Arc::new(Schema::new(fields))
    }
}

impl TableFunctionImpl for AsofJoinTableFunction {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let left: Vec<i64> = args
            .exprs()
            .iter()
            .map(expr_to_i64)
            .collect::<Result<_>>()?;

        let matched = self.accelerators.asof_join(&AsofRequest {
            name: "asof_join",
            version: self.version,
            left: &left,
            right_times: &self.right_times,
            payload_bytes: self.payload_bytes(),
            tolerance_ns: self.tolerance_ns,
        });

        let schema = self.schema();
        let n = left.len();
        let mut price: Vec<Option<f64>> = Vec::with_capacity(n);
        let mut spread: Vec<Option<f64>> = Vec::with_capacity(n);
        for hit in matched.iter().take(n) {
            match *hit {
                Some(j) => {
                    let j = j as usize;
                    price.push(self.right_price.get(j).copied());
                    if let Some(s) = &self.right_spread {
                        spread.push(s.get(j).copied());
                    }
                }
                None => {
                    price.push(None);
                    if self.right_spread.is_some() {
                        spread.push(None);
                    }
                }
            }
        }
        // Defensive: a backend returning a short vector must not shift rows.
        while price.len() < n {
            price.push(None);
            if self.right_spread.is_some() {
                spread.push(None);
            }
        }

        let mut cols: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(left)) as ArrayRef,
            Arc::new(Float64Array::from(price)) as ArrayRef,
        ];
        if self.right_spread.is_some() {
            cols.push(Arc::new(Float64Array::from(spread)) as ArrayRef);
        }

        let batch = RecordBatch::try_new(schema.clone(), cols)?;
        Ok(Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn function() -> AsofJoinTableFunction {
        AsofJoinTableFunction::new(
            vec![0, 10, 20, 30],
            vec![100.0, 110.0, 120.0, 130.0],
            Arc::new(AcceleratorRegistry::host_only()),
        )
    }

    #[test]
    fn singles_report_the_preceding_value() {
        let f = function();
        let got = gtv_accelerator::asof_match(&[5, 10, 25], &f.right_times, f.tolerance_ns);
        assert_eq!(got, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn payload_bytes_covers_price_and_spread() {
        let f = function();
        assert_eq!(f.payload_bytes(), 4 * 8);
        let m = AsofJoinTableFunction::new_multi(
            vec![0, 100],
            vec![10.0, 20.0],
            vec![1.0, 2.0],
            5,
            Arc::new(AcceleratorRegistry::host_only()),
        );
        assert_eq!(m.payload_bytes(), 2 * 8 * 2);
    }

    #[test]
    fn multi_column_applies_the_tolerance() {
        let f = AsofJoinTableFunction::new_multi(
            vec![0, 100],
            vec![10.0, 20.0],
            vec![1.0, 2.0],
            5,
            Arc::new(AcceleratorRegistry::host_only()),
        );
        let got = gtv_accelerator::asof_match(&[110], &f.right_times, f.tolerance_ns);
        assert_eq!(got, vec![None], "10 ns stale exceeds a 5 ns tolerance");
    }

    #[test]
    fn left_rows_before_the_series_are_null() {
        let f = function();
        let got = gtv_accelerator::asof_match(&[-5], &f.right_times, f.tolerance_ns);
        assert_eq!(got, vec![None]);
    }
}
