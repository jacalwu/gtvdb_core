//! Fixed-width attribute sidecars for graph edges.
//!
//! The CSR hot arrays carry only identity and time. Amount, currency code,
//! channel and similar values belong in a sidecar indexed by
//! [`Neighbor::edge_idx`], so a predicate can be pushed into traversal without
//! widening the hot row or embedding domain semantics in the index.
//!
//! Values are `f64` because that is the only floating-point representation
//! already admissible as a mapped [`Col`]. Enumeration-like attributes (e.g.
//! currency) are encoded by the caller as stable numeric codes; the names and
//! decoding tables are a domain concern and stay outside core.

use std::sync::Arc;

use crate::column::Col;
use crate::csr::Neighbor;
use crate::error::{GtvError, Result};
use crate::traversal::EdgePredicate;

/// One named numeric attribute column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttrCol {
    pub name: String,
}

impl From<&str> for AttrCol {
    fn from(name: &str) -> Self {
        Self {
            name: name.to_owned(),
        }
    }
}

/// Names and stable ids of the attribute columns in one sidecar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeAttrSchema {
    pub cols: Vec<AttrCol>,
}

impl EdgeAttrSchema {
    /// Build a schema, rejecting duplicates and more than `u16::MAX` columns.
    pub fn new(names: impl IntoIterator<Item = impl Into<String>>) -> Result<Self> {
        let cols: Vec<AttrCol> = names
            .into_iter()
            .map(|name| AttrCol { name: name.into() })
            .collect();
        if cols.len() > u16::MAX as usize {
            return Err(GtvError::InvalidArgument(format!(
                "edge attribute schema has {} columns, limit is {}",
                cols.len(),
                u16::MAX
            )));
        }
        for (i, col) in cols.iter().enumerate() {
            if cols[..i].iter().any(|other| other.name == col.name) {
                return Err(GtvError::InvalidArgument(format!(
                    "duplicate edge attribute column {:?}",
                    col.name
                )));
            }
        }
        Ok(Self { cols })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.cols.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cols.is_empty()
    }

    /// Column id by name.
    pub fn column(&self, name: &str) -> Option<u16> {
        self.cols
            .iter()
            .position(|c| c.name == name)
            .map(|i| i as u16)
    }

    #[inline]
    pub fn name(&self, col: u16) -> Option<&str> {
        self.cols.get(col as usize).map(|c| c.name.as_str())
    }
}

/// Numeric edge attributes, indexed by the CSR's global edge slot.
///
/// All columns must have the same length. `validity[c][e] == 0` means the value
/// for column `c` on edge `e` is null; NaN is a value, not a null, but predicates
/// reject non-finite values explicitly.
#[derive(Debug, Clone)]
pub struct EdgeAttrs {
    schema: EdgeAttrSchema,
    cols: Vec<Col<f64>>,
    validity: Vec<Vec<u8>>,
}

impl EdgeAttrs {
    /// Build a sidecar where every value is valid.
    pub fn new(schema: EdgeAttrSchema, cols: Vec<Col<f64>>) -> Result<Self> {
        if cols.len() != schema.len() {
            return Err(GtvError::Schema(format!(
                "edge attribute schema has {} columns but {} were supplied",
                schema.len(),
                cols.len()
            )));
        }
        let first = cols.first().map(Col::len);
        if let Some(expected) = first {
            for (i, col) in cols.iter().enumerate() {
                if col.len() != expected {
                    return Err(GtvError::Schema(format!(
                        "edge attribute column {} has {} rows, expected {}",
                        i,
                        col.len(),
                        expected
                    )));
                }
            }
        }
        let validity = cols.iter().map(|c| vec![1u8; c.len()]).collect::<Vec<_>>();
        Ok(Self {
            schema,
            cols,
            validity,
        })
    }

    /// Build a sidecar with explicit null bitmaps (`1` = valid).
    pub fn with_validity(
        schema: EdgeAttrSchema,
        cols: Vec<Col<f64>>,
        validity: Vec<Vec<u8>>,
    ) -> Result<Self> {
        let mut attrs = Self::new(schema, cols)?;
        if validity.len() != attrs.cols.len() {
            return Err(GtvError::Schema(format!(
                "edge attribute schema has {} columns but {} validity bitmaps were supplied",
                attrs.cols.len(),
                validity.len()
            )));
        }
        for (i, (bits, col)) in validity.iter().zip(attrs.cols.iter()).enumerate() {
            if bits.len() != col.len() {
                return Err(GtvError::Schema(format!(
                    "edge attribute validity bitmap {} has {} rows, expected {}",
                    i,
                    bits.len(),
                    col.len()
                )));
            }
        }
        attrs.validity = validity;
        Ok(attrs)
    }

    /// Convenience constructor for in-memory columns.
    pub fn from_vecs(names: &[&str], cols: Vec<Vec<f64>>) -> Result<Self> {
        let schema = EdgeAttrSchema::new(names.iter().copied())?;
        Self::new(schema, cols.into_iter().map(Col::Owned).collect::<Vec<_>>())
    }

    #[inline]
    pub fn schema(&self) -> &EdgeAttrSchema {
        &self.schema
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.cols.first().map_or(0, Col::len)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Value of `col` for the global edge slot `edge_idx`.
    ///
    /// Out-of-range slots, unknown columns and null values all return `None`;
    /// a caller that needs to distinguish them should check the slot against
    /// [`Self::len`] first.
    #[inline]
    pub fn get(&self, col: u16, edge_idx: u32) -> Option<f64> {
        let row = edge_idx as usize;
        let value = *self.cols.get(col as usize)?.get(row)?;
        if self.validity.get(col as usize)?.get(row) == Some(&0) {
            return None;
        }
        Some(value)
    }
}

/// Numeric comparison used by [`AttrPred`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeOp {
    Eq,
    Gt,
    Gte,
    Lt,
    Lte,
    /// Inclusive lower and upper bounds.
    Between,
}

impl RangeOp {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Gt => "gt",
            Self::Gte => "gte",
            Self::Lt => "lt",
            Self::Lte => "lte",
            Self::Between => "between",
        }
    }
}

/// Push a fixed-width attribute test into edge expansion.
///
/// A predicate that cannot see an edge slot must not guess: it returns `false`
/// and [`EdgePredicate::needs_edge_identity`] forces the traversal to use a
/// direction that can supply one.
#[derive(Debug, Clone)]
pub struct AttrPred {
    attrs: Arc<EdgeAttrs>,
    col: u16,
    op: RangeOp,
    lo: f64,
    hi: f64,
}

impl AttrPred {
    pub fn new(attrs: Arc<EdgeAttrs>, col: u16, op: RangeOp, lo: f64, hi: f64) -> Result<Self> {
        if col as usize >= attrs.schema().len() {
            return Err(GtvError::InvalidArgument(format!(
                "edge attribute column {col} is out of range (schema has {} columns)",
                attrs.schema().len()
            )));
        }
        if op == RangeOp::Between && lo > hi {
            return Err(GtvError::InvalidArgument(format!(
                "between predicate has lower bound {lo} above upper bound {hi}"
            )));
        }
        Ok(Self {
            attrs,
            col,
            op,
            lo,
            hi,
        })
    }

    pub fn eq(attrs: Arc<EdgeAttrs>, col: u16, value: f64) -> Result<Self> {
        Self::new(attrs, col, RangeOp::Eq, value, value)
    }
    pub fn gt(attrs: Arc<EdgeAttrs>, col: u16, value: f64) -> Result<Self> {
        Self::new(attrs, col, RangeOp::Gt, value, value)
    }
    pub fn gte(attrs: Arc<EdgeAttrs>, col: u16, value: f64) -> Result<Self> {
        Self::new(attrs, col, RangeOp::Gte, value, value)
    }
    pub fn lt(attrs: Arc<EdgeAttrs>, col: u16, value: f64) -> Result<Self> {
        Self::new(attrs, col, RangeOp::Lt, value, value)
    }
    pub fn lte(attrs: Arc<EdgeAttrs>, col: u16, value: f64) -> Result<Self> {
        Self::new(attrs, col, RangeOp::Lte, value, value)
    }
    pub fn between(attrs: Arc<EdgeAttrs>, col: u16, lo: f64, hi: f64) -> Result<Self> {
        Self::new(attrs, col, RangeOp::Between, lo, hi)
    }

    #[inline]
    pub fn attrs(&self) -> &EdgeAttrs {
        &self.attrs
    }

    #[inline]
    pub fn col(&self) -> u16 {
        self.col
    }

    #[inline]
    pub fn op(&self) -> RangeOp {
        self.op
    }

    #[inline]
    fn compare(&self, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        match self.op {
            RangeOp::Eq => value == self.lo,
            RangeOp::Gt => value > self.lo,
            RangeOp::Gte => value >= self.lo,
            RangeOp::Lt => value < self.lo,
            RangeOp::Lte => value <= self.lo,
            RangeOp::Between => value >= self.lo && value <= self.hi,
        }
    }
}

impl EdgePredicate for AttrPred {
    #[inline]
    fn keep(&self, edge: &Neighbor) -> bool {
        let Some(slot) = edge.edge_idx() else {
            return false;
        };
        self.attrs
            .get(self.col, slot)
            .is_some_and(|value| self.compare(value))
    }

    #[inline]
    fn needs_edge_identity(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neighbor(slot: u32) -> Neighbor {
        Neighbor {
            dst: 1,
            edge_type: 0,
            valid_from: 10,
            valid_to: 20,
            edge_idx: slot,
        }
    }

    #[test]
    fn schema_rejects_duplicates_and_out_of_range_ids() {
        let schema = EdgeAttrSchema::new(["amount", "channel"]).unwrap();
        assert_eq!(schema.column("channel"), Some(1));
        assert_eq!(schema.name(0), Some("amount"));
        assert_eq!(schema.column("missing"), None);
        assert!(EdgeAttrSchema::new(["a", "a"]).is_err());
    }

    #[test]
    fn get_is_keyed_by_the_global_edge_slot() {
        let attrs =
            EdgeAttrs::from_vecs(&["amount", "code"], vec![vec![10.0, 20.0], vec![1.0, 2.0]])
                .unwrap();
        assert_eq!(attrs.len(), 2);
        assert_eq!(attrs.get(0, 0), Some(10.0));
        assert_eq!(attrs.get(0, 1), Some(20.0));
        assert_eq!(attrs.get(1, 1), Some(2.0));
        assert_eq!(attrs.get(0, 2), None);
        assert_eq!(attrs.get(2, 0), None);
    }

    #[test]
    fn attribute_predicate_needs_edge_identity() {
        let attrs = Arc::new(EdgeAttrs::from_vecs(&["amount"], vec![vec![5.0, 50.0]]).unwrap());
        let pred = AttrPred::gt(attrs.clone(), 0, 10.0).unwrap();
        assert!(pred.needs_edge_identity());
        assert!(!pred.keep(&neighbor(0)));
        assert!(pred.keep(&neighbor(1)));
        assert!(!pred.keep(&neighbor(u32::MAX)));
    }

    #[test]
    fn between_is_inclusive_and_null_or_nan_never_matches() {
        let attrs = Arc::new(
            EdgeAttrs::with_validity(
                EdgeAttrSchema::new(["amount"]).unwrap(),
                vec![Col::Owned(vec![10.0, 20.0, f64::NAN])],
                vec![vec![1, 0, 1]],
            )
            .unwrap(),
        );
        let pred = AttrPred::between(attrs, 0, 10.0, 20.0).unwrap();
        assert!(pred.keep(&neighbor(0)));
        assert!(!pred.keep(&neighbor(1)));
        assert!(!pred.keep(&neighbor(2)));
        assert!(AttrPred::between(
            Arc::new(EdgeAttrs::from_vecs(&["v"], vec![vec![]]).unwrap()),
            0,
            2.0,
            1.0
        )
        .is_err());
    }
}
