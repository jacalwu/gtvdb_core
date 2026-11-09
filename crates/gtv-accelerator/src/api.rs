//! The contract every backend implements, and the two borrowed views it is fed.
//!
//! Two rules shape this module:
//!
//! 1. **A backend never owns the truth.** [`SnapshotView`] and [`VectorView`]
//!    borrow the CPU-side arrays; [`ResidentGraph`] / [`ResidentVectors`] are
//!    opaque receipts for a *copy*. Losing a copy costs time, never data.
//! 2. **`business_t` is not part of a handle's identity.** Graph shape is a
//!    function of the snapshot; the time predicate is a separate argument
//!    ([`Accelerator::valid_time_filter`]). Mixing the two would re-upload the
//!    whole graph for every `t`, which is the one thing the CPU path does for
//!    free.
//!
//! Both views are `Copy`-free borrowed structs of slices, so building one costs
//! nothing and needs no allocation.

use gtv_core::{CsrColumns, Metric, Sel, TemporalCSR, VectorHit};

use crate::error::AcceleratorError;

/// Identifies one immutable version of the whole database.
///
/// Equal ids mean byte-identical graph and vector contents. Every cache key in
/// this crate is built from this value alone; forgetting it is how a system
/// silently answers from a stale graph.
pub type SnapshotId = u64;

/// Backend-private handle token. `0` means "no external resource".
pub type Token = u64;

/// Which implementation answered a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Cuda,
}

impl BackendKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Cuda => "cuda",
        }
    }
}

/// An operator a backend may or may not implement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    ValidTimeFilter,
    KnnFlat,
    BatchKhop,
    WindowReduce,
}

impl Op {
    pub fn as_str(&self) -> &'static str {
        match self {
            Op::ValidTimeFilter => "valid_time_filter",
            Op::KnnFlat => "knn_flat",
            Op::BatchKhop => "batch_khop",
            Op::WindowReduce => "window_reduce",
        }
    }

    fn bit(&self) -> u8 {
        match self {
            Op::ValidTimeFilter => 1,
            Op::KnnFlat => 2,
            Op::BatchKhop => 4,
            Op::WindowReduce => 8,
        }
    }
}

/// What a backend can do, reported once at registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Caps {
    pub backend: BackendKind,
    /// Stable name for logs and benchmarks (`"cpu-reference"`, `"cuda-0"`).
    pub name: String,
    /// Whether [`Accelerator::load_graph`] can actually keep a copy.
    pub resident: bool,
    /// `None` = unbounded (the CPU reference).
    pub max_device_bytes: Option<usize>,
    supported: u8,
}

impl Caps {
    pub fn cpu() -> Self {
        Self {
            backend: BackendKind::Cpu,
            name: "cpu-reference".into(),
            resident: true,
            max_device_bytes: None,
            supported: Op::ValidTimeFilter.bit()
                | Op::KnnFlat.bit()
                | Op::BatchKhop.bit()
                | Op::WindowReduce.bit(),
        }
    }

    pub fn new(backend: BackendKind, name: impl Into<String>, max_device_bytes: Option<usize>) -> Self {
        Self {
            backend,
            name: name.into(),
            resident: false,
            max_device_bytes,
            supported: 0,
        }
    }

    pub fn with_resident(mut self, resident: bool) -> Self {
        self.resident = resident;
        self
    }

    pub fn with_op(mut self, op: Op) -> Self {
        self.supported |= op.bit();
        self
    }

    pub fn supports(&self, op: Op) -> bool {
        self.supported & op.bit() != 0
    }
}

/// A borrowed, immutable view of one snapshot's CSR arrays.
///
/// Field-for-field the same arrays the CPU reads, so uploading is a straight
/// `memcpy` per column and nothing has to be re-derived. `chunk_index` /
/// `chunk_max_valid_to` are deliberately absent: they are a CPU-side scan
/// accelerator that a device backend rebuilds from `valid_from` / `valid_to` if
/// it wants them.
#[derive(Debug, Clone, Copy)]
pub struct SnapshotView<'a> {
    pub snapshot_id: SnapshotId,
    pub node_count: usize,
    pub offsets: &'a [u32],
    pub dst: &'a [u64],
    pub valid_from: &'a [i64],
    pub valid_to: &'a [i64],
    pub edge_type: &'a [u16],
}

impl<'a> SnapshotView<'a> {
    /// From raw columns.
    pub fn new(
        snapshot_id: SnapshotId,
        node_count: usize,
        offsets: &'a [u32],
        dst: &'a [u64],
        valid_from: &'a [i64],
        valid_to: &'a [i64],
        edge_type: &'a [u16],
    ) -> Self {
        Self {
            snapshot_id,
            node_count,
            offsets,
            dst,
            valid_from,
            valid_to,
            edge_type,
        }
    }

    /// From an already-materialised [`CsrColumns`].
    pub fn from_columns(snapshot_id: SnapshotId, c: &CsrColumns<'a>) -> Self {
        Self {
            snapshot_id,
            node_count: c.node_count,
            offsets: c.offsets,
            dst: c.dst,
            valid_from: c.valid_from,
            valid_to: c.valid_to,
            edge_type: c.edge_type,
        }
    }

    /// From a CSR that declares its snapshot.
    ///
    /// A CSR without a `snapshot_id` is rejected rather than uploaded under a
    /// guessed id: the id is the only thing that later proves a resident copy
    /// is still the current one.
    pub fn from_csr(csr: &'a TemporalCSR) -> Result<Self, AcceleratorError> {
        let snapshot_id = csr.snapshot_id().ok_or_else(|| {
            AcceleratorError::InvalidArgument(
                "graph has no snapshot_id; call TemporalCSR::with_snapshot_id before uploading"
                    .into(),
            )
        })?;
        Ok(Self::from_columns(snapshot_id, &csr.columns()))
    }

    pub fn edge_count(&self) -> usize {
        self.dst.len()
    }

    /// Bytes of a device copy of this snapshot.
    ///
    /// `dst` is `u64` because that is the source width — a `u32` narrowing is a
    /// separate decision for whoever writes the first kernel, and doing it here
    /// would mean a conversion on every upload with no kernel to pay it back.
    pub fn bytes(&self) -> usize {
        let offsets = (self.node_count + 1) * std::mem::size_of::<u32>();
        let per_edge = std::mem::size_of::<u64>()          // dst
            + 2 * std::mem::size_of::<i64>()              // valid_from, valid_to
            + std::mem::size_of::<u16>(); // edge_type
        offsets + self.edge_count() * per_edge
    }

    /// Shape checks shared by every backend.
    pub fn validate(&self) -> Result<(), AcceleratorError> {
        let e = self.edge_count();
        if self.offsets.len() != self.node_count + 1 {
            return Err(AcceleratorError::InvalidArgument(format!(
                "offsets has {} entries for {} nodes",
                self.offsets.len(),
                self.node_count
            )));
        }
        for (name, len) in [
            ("valid_from", self.valid_from.len()),
            ("valid_to", self.valid_to.len()),
            ("edge_type", self.edge_type.len()),
        ] {
            if len != e {
                return Err(AcceleratorError::InvalidArgument(format!(
                    "{name} has {len} entries for {e} edges"
                )));
            }
        }
        if self.offsets.last().copied().unwrap_or(0) as usize != e {
            return Err(AcceleratorError::InvalidArgument(format!(
                "offsets end at {} but there are {e} edges",
                self.offsets.last().copied().unwrap_or(0)
            )));
        }
        Ok(())
    }
}

/// A borrowed, immutable view of one vector space at one snapshot.
#[derive(Debug, Clone, Copy)]
pub struct VectorView<'a> {
    /// Caller-assigned identity of the space (model + feature schema). Two
    /// spaces with the same dimension but different ids must never be mixed.
    pub vector_space_id: u32,
    pub snapshot_id: SnapshotId,
    pub dim: usize,
    pub metric: Metric,
    /// Row-major, `rows * dim` long.
    pub data: &'a [f32],
    /// Object id per row.
    pub ids: &'a [u64],
}

impl<'a> VectorView<'a> {
    pub fn new(
        vector_space_id: u32,
        snapshot_id: SnapshotId,
        dim: usize,
        metric: Metric,
        data: &'a [f32],
        ids: &'a [u64],
    ) -> Self {
        Self {
            vector_space_id,
            snapshot_id,
            dim,
            metric,
            data,
            ids,
        }
    }

    pub fn rows(&self) -> usize {
        self.ids.len()
    }

    pub fn bytes(&self) -> usize {
        self.rows() * self.dim * std::mem::size_of::<f32>() + self.rows() * 8
    }

    pub fn validate(&self) -> Result<(), AcceleratorError> {
        if self.dim == 0 {
            return Err(AcceleratorError::InvalidArgument("dim is 0".into()));
        }
        if self.data.len() != self.rows() * self.dim {
            return Err(AcceleratorError::InvalidArgument(format!(
                "data holds {} floats but {} rows x {} dim needs {}",
                self.data.len(),
                self.rows(),
                self.dim,
                self.rows() * self.dim
            )));
        }
        Ok(())
    }
}

/// Receipt for a resident copy of one snapshot's graph.
///
/// Carries no pointer and no lifetime: it can be cloned into a query plan,
/// stored in a cache, and compared by `snapshot_id` anywhere. The backend owns
/// the actual buffers and looks them up by [`ResidentGraph::token`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentGraph {
    token: Token,
    snapshot_id: SnapshotId,
    node_count: usize,
    edge_count: usize,
    bytes: usize,
}

impl ResidentGraph {
    pub fn new(
        token: Token,
        snapshot_id: SnapshotId,
        node_count: usize,
        edge_count: usize,
        bytes: usize,
    ) -> Self {
        Self {
            token,
            snapshot_id,
            node_count,
            edge_count,
            bytes,
        }
    }

    pub fn token(&self) -> Token {
        self.token
    }

    pub fn snapshot_id(&self) -> SnapshotId {
        self.snapshot_id
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Fail unless this copy belongs to `snapshot_id`.
    ///
    /// Every operator that consumes a handle calls this first: a resident copy
    /// that outlived its snapshot must produce an error, never an answer.
    pub fn check_snapshot(&self, snapshot_id: SnapshotId) -> Result<(), AcceleratorError> {
        if self.snapshot_id == snapshot_id {
            Ok(())
        } else {
            Err(AcceleratorError::SnapshotStale {
                resident: self.snapshot_id,
                requested: snapshot_id,
            })
        }
    }
}

/// Receipt for a resident copy of one vector space at one snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentVectors {
    token: Token,
    vector_space_id: u32,
    snapshot_id: SnapshotId,
    rows: usize,
    dim: usize,
    metric: Metric,
    bytes: usize,
}

impl ResidentVectors {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        token: Token,
        vector_space_id: u32,
        snapshot_id: SnapshotId,
        rows: usize,
        dim: usize,
        metric: Metric,
        bytes: usize,
    ) -> Self {
        Self {
            token,
            vector_space_id,
            snapshot_id,
            rows,
            dim,
            metric,
            bytes,
        }
    }

    pub fn token(&self) -> Token {
        self.token
    }

    pub fn vector_space_id(&self) -> u32 {
        self.vector_space_id
    }

    pub fn snapshot_id(&self) -> SnapshotId {
        self.snapshot_id
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Fail unless this copy belongs to the request's snapshot and space.
    ///
    /// Vectors are keyed by both because the same object id has a different
    /// vector per space: checking only the snapshot would let a model upgrade
    /// mix two spaces inside one query.
    pub fn check_snapshot(
        &self,
        snapshot_id: SnapshotId,
        vector_space_id: u32,
    ) -> Result<(), AcceleratorError> {
        if self.snapshot_id != snapshot_id {
            return Err(AcceleratorError::SnapshotStale {
                resident: self.snapshot_id,
                requested: snapshot_id,
            });
        }
        if self.vector_space_id != vector_space_id {
            return Err(AcceleratorError::InvalidArgument(format!(
                "resident vectors are for space {}, request is for {}",
                self.vector_space_id, vector_space_id
            )));
        }
        Ok(())
    }
}

/// Aggregation performed by [`Accelerator::window_reduce`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowOp {
    Sum,
    Mean,
    Min,
    Max,
    Count,
}

/// A backend: load a snapshot into its own memory, then answer operator calls
/// against that copy.
///
/// Every method has a default that reports
/// [`AcceleratorError::Unsupported`], so a new backend implements exactly the
/// operators it has ported and the policy routes the rest to the CPU. Nothing
/// here may panic, abort, or return a degraded answer: the only two acceptable
/// outcomes are a result or an error the caller can fall back from.
pub trait Accelerator: Send + Sync {
    fn name(&self) -> &'static str;

    fn kind(&self) -> BackendKind;

    fn caps(&self) -> Caps;

    /// Upload a snapshot. `Ok(None)` = this backend does not do residency at
    /// all and should not be asked again.
    fn load_graph(&self, view: SnapshotView<'_>) -> Result<Option<ResidentGraph>, AcceleratorError>;

    /// Drop a resident copy. Called by the cache on eviction, invalidation and
    /// drop; idempotent for unknown tokens.
    fn release_graph(&self, _handle: &ResidentGraph) {}

    fn load_vectors(&self, _view: VectorView<'_>) -> Result<Option<ResidentVectors>, AcceleratorError> {
        Ok(None)
    }

    fn release_vectors(&self, _handle: &ResidentVectors) {}

    /// Edge ids live at `t`, as a candidate set bound to the handle's snapshot.
    fn valid_time_filter(
        &self,
        _handle: &ResidentGraph,
        _t: i64,
    ) -> Result<Sel, AcceleratorError> {
        Err(self.unsupported(Op::ValidTimeFilter))
    }

    /// Exact flat scan for a batch of queries. `filter` restricts the scan to a
    /// candidate set (already `Sel`, so no third selection type appears here).
    fn knn_flat(
        &self,
        _handle: &ResidentVectors,
        _queries: &[f32],
        _k: usize,
        _filter: Option<&Sel>,
    ) -> Result<Vec<Vec<VectorHit>>, AcceleratorError> {
        Err(self.unsupported(Op::KnnFlat))
    }

    /// Breadth-first expansion from a seed set, seeds excluded from the output
    /// (same convention as [`gtv_core::TemporalCSR::khop`]).
    fn batch_khop(
        &self,
        _handle: &ResidentGraph,
        _seeds: &Sel,
        _hops: u8,
        _at: i64,
    ) -> Result<Sel, AcceleratorError> {
        Err(self.unsupported(Op::BatchKhop))
    }

    /// Trailing-window aggregate over one column. Stateless: no handle, because
    /// the value column is not part of the graph.
    fn window_reduce(
        &self,
        _values: &[f64],
        _window: usize,
        _op: WindowOp,
    ) -> Result<Vec<f64>, AcceleratorError> {
        Err(self.unsupported(Op::WindowReduce))
    }

    fn unsupported(&self, op: Op) -> AcceleratorError {
        AcceleratorError::Unsupported {
            backend: self.name(),
            op: op.as_str(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view<'a>(
        offsets: &'a [u32],
        dst: &'a [u64],
        vf: &'a [i64],
        vt: &'a [i64],
        et: &'a [u16],
    ) -> SnapshotView<'a> {
        SnapshotView::new(7, offsets.len() - 1, offsets, dst, vf, vt, et)
    }

    #[test]
    fn view_rejects_ragged_columns() {
        let offsets = [0u32, 1, 2];
        let dst = [1u64, 2];
        let vf = [0i64, 0];
        let vt = [10i64, 10];
        let et = [1u16, 1];
        assert!(view(&offsets, &dst, &vf, &vt, &et).validate().is_ok());

        // One column short is the mistake a backend would otherwise read past.
        let short = [1u16];
        let err = view(&offsets, &dst, &vf, &vt, &short).validate().unwrap_err();
        assert!(err.to_string().contains("edge_type"), "{err}");

        // offsets disagreeing with the edge count is the other one.
        let wrong = [0u32, 1, 3];
        let err = view(&wrong, &dst, &vf, &vt, &et).validate().unwrap_err();
        assert!(err.to_string().contains("offsets end at 3"), "{err}");
    }

    #[test]
    fn graph_bytes_follow_the_columns() {
        let offsets = [0u32, 2, 2];
        let dst = [1u64, 2];
        let vf = [0i64, 0];
        let vt = [10i64, 10];
        let et = [1u16, 1];
        let v = view(&offsets, &dst, &vf, &vt, &et);
        // 3 offsets * 4 B + 2 edges * (8 + 8 + 8 + 2) = 12 + 52
        assert_eq!(v.bytes(), 64);
    }

    #[test]
    fn stale_handle_is_an_error_not_an_answer() {
        let h = ResidentGraph::new(1, 41, 2, 2, 64);
        assert!(h.check_snapshot(41).is_ok());
        let err = h.check_snapshot(42).unwrap_err();
        assert!(matches!(
            err,
            AcceleratorError::SnapshotStale {
                resident: 41,
                requested: 42
            }
        ));
        assert_eq!(err.fallback(), crate::Fallback::Cpu);
    }

    #[test]
    fn caps_are_additive() {
        let c = Caps::new(BackendKind::Cuda, "cuda-0", Some(1024)).with_op(Op::KnnFlat);
        assert!(c.supports(Op::KnnFlat));
        assert!(!c.supports(Op::BatchKhop));
        assert!(Caps::cpu().supports(Op::BatchKhop));
    }
}
