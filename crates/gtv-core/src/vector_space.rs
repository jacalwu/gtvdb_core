//! Vector-space identity: a stable name for "which embedding space these
//! vectors live in".
//!
//! An index built from model A must never silently answer a query meant for
//! model B, even when both models emit the same dimension. [`VectorSpace`]
//! captures the five inputs that actually determine comparability —
//! `(model_id, model_version, feature_schema_id, metric, dim)` — and reduces
//! them to one [`VectorSpaceId`]. Two spaces with a different model (or a
//! different schema/metric/dim) therefore get different ids, and a binding
//! check can refuse a cross-space query instead of returning plausible but
//! meaningless neighbours.
//!
//! # Why a stable hand-rolled hash and not `DefaultHasher`
//!
//! The id is persisted in the index manifest and re-verified on cold open, so
//! it must be identical across processes and compiler versions.
//! [`std::collections::hash_map::DefaultHasher`] explicitly does **not**
//! promise that. This module uses FNV-1a 64 over length-prefixed fields: fully
//! specified, dependency-free, and deterministic. It is a 64-bit identity, so
//! a collision is possible in principle (~2⁻⁶⁴); that is the same guarantee a
//! `u64` id can offer at all, and the manifest also keeps the human-readable
//! descriptor next to the id for exactly this reason.

use crate::metric::Metric;

/// Stable identifier of one embedding space.
pub type VectorSpaceId = u64;

/// The five inputs that decide whether two vectors are comparable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VectorSpace {
    pub model_id: String,
    pub model_version: String,
    /// Identifier of the feature pipeline / schema that produced the vectors.
    /// A changed feature set is a different space even under the same model.
    pub feature_schema_id: String,
    pub metric: Metric,
    pub dim: usize,
}

impl VectorSpace {
    pub fn new(
        model_id: impl Into<String>,
        model_version: impl Into<String>,
        feature_schema_id: impl Into<String>,
        metric: Metric,
        dim: usize,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            model_version: model_version.into(),
            feature_schema_id: feature_schema_id.into(),
            metric,
            dim,
        }
    }

    /// Stable id of this space.
    ///
    /// Deterministic across processes and builds: same descriptor ⇒ same id,
    /// any of the five components differing ⇒ (with overwhelming probability) a
    /// different id.
    pub fn id(&self) -> VectorSpaceId {
        let mut h = FNV_OFFSET;
        feed_str(&mut h, &self.model_id);
        feed_str(&mut h, &self.model_version);
        feed_str(&mut h, &self.feature_schema_id);
        feed_str(&mut h, self.metric.as_str());
        feed_u64(&mut h, self.dim as u64);
        h
    }
}

/// A claim that "this index answers for this space, built from this snapshot".
///
/// `snapshot_id` is the **corpus** snapshot (the vectors' source rows), not the
/// graph snapshot: a vector index and the temporal graph it annotates are two
/// different snapshots, so mixing their ids would produce false mismatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorSpaceBinding {
    pub snapshot_id: Option<u64>,
    pub vector_space_id: VectorSpaceId,
}

impl VectorSpaceBinding {
    pub const fn new(snapshot_id: Option<u64>, vector_space_id: VectorSpaceId) -> Self {
        Self {
            snapshot_id,
            vector_space_id,
        }
    }

    /// Refuse a request that contradicts this binding.
    ///
    /// `requested_*` are what the caller asks for; `None` means "not specified"
    /// and imposes no constraint. A specified value must equal the bound one —
    /// a mismatch is an error, never a silent fallback to the bound space,
    /// because answering a model-B query from a model-A index is worse than
    /// refusing it.
    pub fn check(
        &self,
        requested_snapshot: Option<u64>,
        requested_space: Option<VectorSpaceId>,
    ) -> Result<(), crate::error::GtvError> {
        if let Some(got) = requested_space {
            if got != self.vector_space_id {
                return Err(crate::error::GtvError::VectorSpaceMismatch {
                    expected: self.vector_space_id,
                    got,
                });
            }
        }
        if let (Some(expected), Some(got)) = (self.snapshot_id, requested_snapshot) {
            if expected != got {
                return Err(crate::error::GtvError::SnapshotMismatch { expected, got });
            }
        }
        Ok(())
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[inline]
fn feed_byte(h: &mut u64, b: u8) {
    *h ^= b as u64;
    *h = h.wrapping_mul(FNV_PRIME);
}

#[inline]
fn feed_u64(h: &mut u64, v: u64) {
    for b in v.to_le_bytes() {
        feed_byte(h, b);
    }
}

/// Length-prefix every string so `("ab", "c")` and `("a", "bc")` cannot hash
/// to the same digest — the classic concatenation ambiguity.
#[inline]
fn feed_str(h: &mut u64, s: &str) {
    feed_u64(h, s.len() as u64);
    for &b in s.as_bytes() {
        feed_byte(h, b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space(model: &str, dim: usize) -> VectorSpace {
        VectorSpace::new(model, "v1", "feat-v1", Metric::Cosine, dim)
    }

    #[test]
    fn id_is_stable_for_the_same_descriptor() {
        let a = space("m1", 128);
        let b = space("m1", 128);
        assert_eq!(a.id(), b.id());
    }

    #[test]
    fn any_component_change_yields_a_different_id() {
        let base = space("m1", 128).id();
        assert_ne!(base, space("m2", 128).id(), "different model");
        assert_ne!(
            base,
            VectorSpace::new("m1", "v2", "feat-v1", Metric::Cosine, 128).id(),
            "different model version"
        );
        assert_ne!(
            base,
            VectorSpace::new("m1", "v1", "feat-v2", Metric::Cosine, 128).id(),
            "different feature schema"
        );
        assert_ne!(
            base,
            VectorSpace::new("m1", "v1", "feat-v1", Metric::L2, 128).id(),
            "different metric"
        );
        assert_ne!(base, space("m1", 256).id(), "different dimension");
    }

    #[test]
    fn length_prefixing_removes_concatenation_ambiguity() {
        // "ab"+"c" must not collide with "a"+"bc".
        let a = VectorSpace::new("ab", "c", "f", Metric::L2, 1).id();
        let b = VectorSpace::new("a", "bc", "f", Metric::L2, 1).id();
        assert_ne!(a, b);
    }

    #[test]
    fn binding_refuses_a_contradicting_request() {
        let b = VectorSpaceBinding::new(Some(7), 42);
        assert!(b.check(None, None).is_ok());
        assert!(b.check(Some(7), Some(42)).is_ok());
        assert!(matches!(
            b.check(None, Some(43)),
            Err(crate::error::GtvError::VectorSpaceMismatch {
                expected: 42,
                got: 43
            })
        ));
        assert!(matches!(
            b.check(Some(8), Some(42)),
            Err(crate::error::GtvError::SnapshotMismatch {
                expected: 7,
                got: 8
            })
        ));
        // An unspecified snapshot is not a contradiction.
        assert!(b.check(None, Some(42)).is_ok());
    }
}
