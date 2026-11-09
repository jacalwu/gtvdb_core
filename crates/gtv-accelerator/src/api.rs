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
use crate::metrics::{Column, ColumnSet, GraphLoadMetrics};

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

    /// Refuse a copy that cannot fit, before a byte is allocated.
    ///
    /// The budget is the *only* number that says "this graph is too big", and it
    /// has to be consulted first: `cuMemAlloc` failing halfway through an upload
    /// leaves the pool fragmented, reports a driver code instead of a size, and
    /// tells the caller nothing about which columns were too expensive. A
    /// pre-flight turns "OOM, somewhere" into "23 B/edge x 1.2e9 does not fit
    /// 4.8 GiB", which is the sentence a query planner needs to choose the CPU
    /// — or, later, a smaller chunk.
    ///
    /// `Fallback::Cpu` is what makes this safe to return: nothing was allocated,
    /// nothing was lost, and the answer is still computable.
    pub fn admit(&self, needed: usize) -> Result<(), AcceleratorError> {
        match self.max_device_bytes {
            Some(limit) if needed > limit => Err(AcceleratorError::OutOfBudget { needed, limit }),
            _ => Ok(()),
        }
    }
}

/// Device width of the `dst` column.
///
/// The source CSR stores a node id as `u64`, which is 8 B per edge on the wire
/// and in VRAM for every edge in the graph. No database this project targets
/// comes close to `u32::MAX` nodes, so the narrow form is worth having — but it
/// has to arrive *already narrow*, because converting it at upload loses: the
/// pass reads 8 B/edge and writes 4 to save 4 on a link that is slower than
/// either, and 1e7 edges measured 14 ms of host work against 5.4 ms of transfer.
/// See `docs/cuda-acceleration-plan.md` §9.2 for the numbers.
///
/// So a view never converts. It either carries a `u32` column a producer already
/// built ([`SnapshotView::with_dst32`]) or it carries the source `u64` one, and
/// [`SnapshotView::bytes`] prices whichever it has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DstLayout {
    /// 4 B per edge, from a column the caller already holds.
    U32,
    /// 8 B per edge, the source width.
    U64,
}

impl DstLayout {
    pub fn bytes_per_edge(self) -> usize {
        match self {
            DstLayout::U32 => 4,
            DstLayout::U64 => 8,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DstLayout::U32 => "u32",
            DstLayout::U64 => "u64",
        }
    }
}

/// A borrowed, immutable view of one snapshot's CSR arrays.
///
/// Four columns are uploaded as-is, a straight `memcpy` each. `dst` is uploaded
/// at whatever width the caller has it in — `u64` from the CSR, or `u32` if a
/// producer built one — and never converted here, because a conversion pass
/// measured more expensive than the bytes it saves (see [`DstLayout`]).
/// `chunk_index` / `chunk_max_valid_to` are deliberately absent: they are a
/// CPU-side scan accelerator that a device backend rebuilds from `valid_from` /
/// `valid_to` if it wants them.
#[derive(Debug, Clone, Copy)]
pub struct SnapshotView<'a> {
    pub snapshot_id: SnapshotId,
    pub node_count: usize,
    pub offsets: &'a [u32],
    pub dst: &'a [u64],
    pub valid_from: &'a [i64],
    pub valid_to: &'a [i64],
    pub edge_type: &'a [u16],
    /// The same edges as `dst`, already at 4 B/edge.
    ///
    /// `Some` means `dst` may be left empty and the backend uploads this column
    /// instead. Built by whichever producer can afford the pass once — at build
    /// time, where it is free, not at upload time, where it is not.
    pub dst32: Option<&'a [u32]>,
    /// Which columns to upload. [`ColumnSet::ALL`] unless the caller projects.
    ///
    /// The arrays above are always borrowed; this says which of them the device
    /// copy must contain. A walk that only follows edges asks for
    /// [`ColumnSet::TOPOLOGY`] and leaves 16 of 23 bytes per edge on the host.
    pub columns: ColumnSet,
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
            dst32: None,
            columns: ColumnSet::ALL,
        }
    }

    /// Carry the `dst` column at 4 B/edge instead of 8.
    ///
    /// For a producer that can build the narrow column once — at CSR build time,
    /// where the pass is amortised over every query, not at upload time, where
    /// it is not. The ids are not range-checked here: whatever built a `u32`
    /// column already had to prove they fit, and re-scanning 40 MB per upload to
    /// re-prove it would reintroduce the cost this method exists to avoid.
    pub fn with_dst32(mut self, dst32: &'a [u32]) -> Self {
        self.dst32 = Some(dst32);
        self
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
            dst32: None,
            columns: ColumnSet::ALL,
        }
    }

    /// Upload only these columns.
    ///
    /// The choice is a property of the *operator*, not of the graph: a neighbour
    /// scan never reads `valid_from`, a temporal as-of filter reads nothing else
    /// but it and `valid_to`. Two operators therefore want two copies, and the
    /// cache keys them separately — see [`crate::ResidentCache`].
    ///
    /// Topology (`offsets` + `dst`) is not optional: a copy without it is not a
    /// graph, and [`SnapshotView::validate`] rejects it rather than letting a
    /// kernel find out.
    pub fn with_columns(mut self, columns: ColumnSet) -> Self {
        self.columns = columns;
        self
    }

    /// Columns this request needs from a resident copy.
    ///
    /// Kept as a method rather than a field read so the name states the
    /// invariant: the projection *is* the requirement.
    pub fn required_columns(&self) -> ColumnSet {
        self.columns
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
        // Exactly one of the two carries the edges; `validate` enforces it.
        match self.dst32 {
            Some(d) => d.len(),
            None => self.dst.len(),
        }
    }

    /// The width `dst` will be uploaded at — whatever the caller holds.
    ///
    /// O(1): it is a property of which column the view carries, not of the
    /// graph's contents, so pricing an upload never costs a pass over it.
    pub fn dst_layout(&self) -> DstLayout {
        match self.dst32 {
            Some(_) => DstLayout::U32,
            None => DstLayout::U64,
        }
    }

    /// Bytes of a device copy of this snapshot.
    ///
    /// This is the number the resident cache charges against
    /// `Caps::max_device_bytes`, so it has to be the number the backend really
    /// allocates — 22 B/edge with a `u32` column, 26 B/edge with the source
    /// `u64` one, plus the offsets. It prices the *projection*, not the CSR: an
    /// upload asked for topology only is charged for topology only. A backend
    /// that allocates differently must fail rather than disagree; see
    /// `CudaAccelerator::load_graph`, which compares its column total with this
    /// figure before returning a handle.
    pub fn bytes(&self) -> usize {
        self.columns
            .bytes(self.node_count, self.edge_count(), self.dst_layout())
    }

    /// Bytes this projection leaves on the host.
    ///
    /// The part of the graph the device cannot answer from, which is also the
    /// size of the second upload if a later operator asks for it.
    pub fn deferred_bytes(&self) -> usize {
        let deferred = self.columns.missing(ColumnSet::ALL);
        deferred.bytes(self.node_count, self.edge_count(), self.dst_layout())
    }

    /// Shape checks shared by every backend.
    pub fn validate(&self) -> Result<(), AcceleratorError> {
        // Topology is the definition of the copy, not a column in it: `edge_count`
        // is read from `dst`/`dst32`, and every other check is phrased in edges.
        // A projection that drops it would price itself at zero edges and let a
        // kernel walk an array of offsets with nothing behind it.
        let topology = self.columns.missing(ColumnSet::TOPOLOGY);
        if !topology.is_empty() {
            return Err(AcceleratorError::InvalidArgument(format!(
                "a graph copy needs {}, the projection omits them",
                topology.names()
            )));
        }
        let e = self.edge_count();
        // Two widths of the same column may both be present, but not disagree:
        // a backend uploads one of them and prices the edge count from it.
        if let Some(d) = self.dst32 {
            if !self.dst.is_empty() && d.len() != self.dst.len() {
                return Err(AcceleratorError::InvalidArgument(format!(
                    "dst holds {} ids and dst32 holds {}, both describe the same edges",
                    self.dst.len(),
                    d.len()
                )));
            }
        }
        if self.offsets.len() != self.node_count + 1 {
            return Err(AcceleratorError::InvalidArgument(format!(
                "offsets has {} entries for {} nodes",
                self.offsets.len(),
                self.node_count
            )));
        }
        // Only the columns this upload will carry are checked. A projected view
        // is not required to hold arrays it will not transfer — that is the
        // point of the projection, and demanding them would force the caller to
        // borrow data it deliberately left behind.
        for (column, len) in [
            (Column::Dst, e),
            (Column::ValidFrom, self.valid_from.len()),
            (Column::ValidTo, self.valid_to.len()),
            (Column::EdgeType, self.edge_type.len()),
        ] {
            if !self.columns.contains(column) {
                continue;
            }
            if len != e {
                return Err(AcceleratorError::InvalidArgument(format!(
                    "{} has {len} entries for {e} edges",
                    column.name()
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
    dst_layout: DstLayout,
    columns: ColumnSet,
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
            // The source width, so a receipt built by hand describes the graph
            // it was built from. A backend that narrowed its copy must say so.
            dst_layout: DstLayout::U64,
            // A receipt built by hand is assumed to describe a complete copy;
            // a backend that projected must say so, for the same reason.
            columns: ColumnSet::ALL,
        }
    }

    /// Record which columns the copy actually carries.
    ///
    /// Not decoration: [`ResidentGraph::check_columns`] is how an operator that
    /// reads `valid_from` refuses a topology-only copy instead of reading past
    /// the end of five device arrays.
    pub fn with_columns(mut self, columns: ColumnSet) -> Self {
        self.columns = columns;
        self
    }

    /// The columns this copy is guaranteed to carry.
    pub fn columns(&self) -> ColumnSet {
        self.columns
    }

    /// Fail unless the copy carries everything the operator reads.
    pub fn check_columns(&self, required: ColumnSet) -> Result<(), AcceleratorError> {
        let missing = self.columns.missing(required);
        if missing.is_empty() {
            Ok(())
        } else {
            Err(AcceleratorError::InvalidArgument(format!(
                "resident copy of snapshot {} carries {}, but this operator reads {}",
                self.snapshot_id,
                self.columns.names(),
                missing.names()
            )))
        }
    }

    /// Record the width `dst` was copied at. A kernel reads this instead of
    /// assuming, because reading a `u32` column as `u64` yields two garbage
    /// node ids per real one and no error anywhere.
    pub fn with_dst_layout(mut self, layout: DstLayout) -> Self {
        self.dst_layout = layout;
        self
    }

    pub fn dst_layout(&self) -> DstLayout {
        self.dst_layout
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

    /// What this backend's last [`Accelerator::load_graph`] cost, stage by
    /// stage.
    ///
    /// Out of band rather than in the return value: `load_graph` answers "is it
    /// resident", and every caller that does not care about timing should not
    /// have to destructure a second field to find out. `None` means the backend
    /// did not upload anything (CPU) or did not measure.
    fn last_load_metrics(&self) -> Option<GraphLoadMetrics> {
        None
    }

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
        // No `dst32`, so the source width: 3 offsets * 4 B + 2 edges * (8 + 8 + 8 + 2).
        assert_eq!(v.dst_layout(), DstLayout::U64);
        assert_eq!(v.bytes(), 12 + 2 * 26);

        // The same graph with the narrow column the caller already holds.
        let dst32 = [1u32, 2];
        let v = v.with_dst32(&dst32);
        assert_eq!(v.dst_layout(), DstLayout::U32);
        assert_eq!(v.bytes(), 12 + 2 * 22);
        assert_eq!(v.edge_count(), 2);
    }

    #[test]
    fn a_ten_million_edge_graph_costs_22_bytes_per_edge_narrow() {
        // The number the break-even is computed from, asserted rather than
        // quoted: 1e7 edges at 22 B is 209.8 MiB, against 247.9 MiB wide.
        let edges = 10_000_000usize;
        let nodes = edges / 4;
        let offsets = vec![0u32; nodes + 1];
        let dst = vec![0u64; edges];
        let dst32 = vec![0u32; edges];
        let ts = vec![0i64; edges];
        let et = vec![0u16; edges];
        let offsets_bytes = (nodes + 1) * 4;

        let wide = SnapshotView::new(1, nodes, &offsets, &dst, &ts, &ts, &et);
        let narrow = wide.with_dst32(&dst32);
        assert_eq!(wide.dst_layout(), DstLayout::U64);
        assert_eq!(narrow.dst_layout(), DstLayout::U32);
        assert_eq!(narrow.bytes() - offsets_bytes, edges * 22);
        // 40 MB — the 15% of VRAM that the narrow column buys, and the reason
        // it is worth having a producer build one.
        assert_eq!(wide.bytes() - narrow.bytes(), edges * 4);
    }

    #[test]
    fn the_width_comes_from_the_column_not_from_the_node_count() {
        // A `u64` column is uploaded as `u64` even though every id would fit:
        // the view never converts, so the width it prices is the width it has.
        // This is the whole finding of §9.2 — converting costs more than the
        // transfer it saves, so the narrow width must arrive from upstream.
        let dst = [1u64, 2, 3];
        let ts = [0i64; 3];
        let et = [0u16; 3];
        let v = SnapshotView::new(1, 4, &[0, 3, 3, 3, 3], &dst, &ts, &ts, &et);
        assert_eq!(v.dst_layout(), DstLayout::U64);
        assert_eq!(v.bytes() - 5 * 4, 3 * 26);
    }

    #[test]
    fn two_widths_of_one_column_must_agree_on_the_edge_count() {
        let dst = [1u64, 2, 3];
        let ts = [0i64; 3];
        let et = [0u16; 3];
        let offsets = [0u32, 3, 3, 3, 3];
        let short = [1u32, 2];
        let err = SnapshotView::new(1, 4, &offsets, &dst, &ts, &ts, &et)
            .with_dst32(&short)
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("dst32 holds 2"), "{err}");

        // Empty `dst` plus a `dst32` is the intended shape, and it validates.
        let dst32 = [1u32, 2, 3];
        let v = SnapshotView::new(1, 4, &offsets, &[], &ts, &ts, &et).with_dst32(&dst32);
        assert!(v.validate().is_ok());
        assert_eq!(v.edge_count(), 3);
    }

    #[test]
    fn a_receipt_carries_the_width_of_its_own_copy() {
        let h = ResidentGraph::new(1, 41, 2, 2, 56);
        // Built by hand = the source width; a narrowing backend must say so.
        assert_eq!(h.dst_layout(), DstLayout::U64);
        let narrowed = h.clone().with_dst_layout(DstLayout::U32);
        assert_eq!(narrowed.dst_layout(), DstLayout::U32);
        // Two copies of one snapshot at different widths are not the same fact,
        // so the receipts differ and a cache cannot substitute one for the other.
        assert_ne!(h, narrowed);
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
