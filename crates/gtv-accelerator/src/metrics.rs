//! Per-operator records, so a benchmark can say which side answered what.
//!
//! A speedup number that does not name its backend, its input size and its
//! decision reason is not reproducible. Each call appends one [`OpRecord`]; the
//! bench example prints them as a table, and a test can assert that the CPU and
//! device sides of a parity run were both actually exercised.
//!
//! [`GraphLoadMetrics`] is the same idea applied to the other half of the
//! problem: an upload is not one number. "The graph takes 40 ms to load" hides
//! whether the milliseconds went into materialising a snapshot, re-packing it
//! into another layout, allocating, or moving bytes across PCIe — and those four
//! have four different fixes. Every stage that can be timed is timed, every
//! byte that can be counted is counted, and the stages that are *supposed* to be
//! zero are fields rather than absences so that a regression is visible instead
//! of invisible.

use crate::api::{BackendKind, DstLayout, Op, SnapshotId};
use crate::policy::BackendDecision;

/// One column of the CSR, as an upload sees it.
///
/// A named enum rather than five `h2d_*_ns` fields: the benchmark prints the
/// table by iterating [`Column::ALL`], and a sixth column would otherwise mean
/// editing every print site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Column {
    Offsets,
    Dst,
    ValidFrom,
    ValidTo,
    EdgeType,
}

impl Column {
    pub const ALL: [Column; 5] = [
        Column::Offsets,
        Column::Dst,
        Column::ValidFrom,
        Column::ValidTo,
        Column::EdgeType,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Column::Offsets => "offsets",
            Column::Dst => "dst",
            Column::ValidFrom => "valid_from",
            Column::ValidTo => "valid_to",
            Column::EdgeType => "edge_type",
        }
    }

    /// Bytes per edge this column contributes, which is where the 22-vs-26
    /// B/edge figure comes from. `offsets` is per node, so it reports 0.
    pub fn bytes_per_edge(self, layout: DstLayout) -> usize {
        match self {
            Column::Offsets => 0,
            // The one column whose width is a decision rather than a fact.
            Column::Dst => layout.bytes_per_edge(),
            Column::ValidFrom | Column::ValidTo => 8,
            Column::EdgeType => 2,
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }
}

/// A projection: which of the five columns a copy actually carries.
///
/// A device copy is not one object, it is five arrays that happen to share a
/// snapshot id. Uploading all five when a walk only reads topology costs 23
/// B/edge for 5 — the two time columns alone are 16 of those bytes, and a
/// neighbour scan never touches them. The projection is how a caller says
/// "topology only" without a second view type, and it is also the second half
/// of a cache key: a topology-only copy must never answer a request that needs
/// `valid_from`.
///
/// Ordering and hashing come from the bit pattern, so a set is usable as a map
/// key next to the snapshot id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ColumnSet(u8);

impl ColumnSet {
    pub const EMPTY: Self = Self(0);
    /// What a neighbour scan reads: where a node's edges begin, and where they
    /// point. Nothing else is on the walk's critical path.
    pub const TOPOLOGY: Self = Self(0b00011);
    /// When each edge is live. Read only by a time-filtered walk, and it is the
    /// largest part of an upload: 16 of the 23 B/edge.
    pub const TEMPORAL: Self = Self(0b01100);
    /// The edge label, for a typed walk.
    pub const LABELS: Self = Self(0b10000);
    pub const ALL: Self = Self(0b11111);

    pub fn single(column: Column) -> Self {
        Self(1 << column.index())
    }

    pub fn contains(self, column: Column) -> bool {
        self.0 & (1 << column.index()) != 0
    }

    pub fn insert(&mut self, column: Column) {
        self.0 |= 1 << column.index();
    }

    pub fn remove(&mut self, column: Column) {
        self.0 &= !(1 << column.index());
    }

    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// This set minus `other`.
    pub fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Columns in `required` that this set does not carry.
    ///
    /// This is the one predicate that stands between a projection and a wrong
    /// answer: an operator asks for what it reads, and a copy that lacks it
    /// must be re-uploaded rather than read out of bounds.
    pub fn missing(self, required: Self) -> Self {
        Self(required.0 & !self.0)
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Present columns, in [`Column::ALL`] order so a print site is stable.
    pub fn iter(self) -> impl Iterator<Item = Column> {
        Column::ALL.into_iter().filter(move |c| self.contains(*c))
    }

    /// A stable label for logs and error messages (`"offsets+dst"`).
    pub fn names(self) -> String {
        let mut out = String::new();
        for c in self.iter() {
            if !out.is_empty() {
                out.push('+');
            }
            out.push_str(c.name());
        }
        if out.is_empty() {
            out.push_str("none");
        }
        out
    }

    /// Bytes these columns occupy on the device.
    ///
    /// The one place the byte model lives. [`crate::SnapshotView::bytes`] calls
    /// it, the capacity check calls it, and a benchmark prints it — three
    /// callers that must never disagree about what 4 GB of VRAM holds.
    pub fn bytes(self, nodes: usize, edges: usize, dst: DstLayout) -> usize {
        let mut total = 0;
        for c in self.iter() {
            total += match c {
                // One entry per node boundary, not per edge.
                Column::Offsets => (nodes + 1) * std::mem::size_of::<u32>(),
                _ => edges * c.bytes_per_edge(dst),
            };
        }
        total
    }
}

impl From<Column> for ColumnSet {
    fn from(column: Column) -> Self {
        Self::single(column)
    }
}

impl FromIterator<Column> for ColumnSet {
    fn from_iter<T: IntoIterator<Item = Column>>(iter: T) -> Self {
        let mut set = Self::EMPTY;
        for c in iter {
            set.insert(c);
        }
        set
    }
}

/// What one column cost to place on the device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ColumnLoad {
    /// Bytes on the wire — for `dst` this is the width actually uploaded.
    pub bytes: usize,
    /// `cuMemAlloc` for this column, including the driver's own granularity
    /// rounding (2 MiB on the device measured in §9.2).
    pub alloc_ns: u64,
    /// The host-to-device copy for this column.
    pub h2d_ns: u64,
}

/// The receipt for one [`crate::Accelerator::load_graph`].
///
/// # Why some fields exist to be zero
///
/// `host_pack_ns`, `layout_transform_ns` and `staging_bytes` are all zero in the
/// current path, and that is the point: they are the three ways a loader
/// silently becomes slower than its own PCIe link. An AoS re-pack, a `u64`→`u32`
/// conversion, or a copy into a staging buffer would each show up here as a
/// non-zero stage *and* as `staging_bytes > 0`, instead of hiding inside
/// "upload time". A benchmark that reports only `cudaMemcpy` cannot tell those
/// apart, which is why it reports the wrong bottleneck with confidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphLoadMetrics {
    pub snapshot_id: SnapshotId,
    pub nodes: usize,
    pub edges: usize,
    pub dst_layout: DstLayout,
    /// The projection this upload was asked for.
    pub projection: ColumnSet,

    /// Caller-side cost of producing the CSR this upload consumed. `None` when
    /// the caller did not measure it — a backend cannot know how long its own
    /// input took to exist, and a zero there would be a lie rather than a
    /// measurement.
    pub snapshot_materialize_ns: Option<u64>,
    /// Making the context current on this thread.
    pub bind_ns: u64,
    /// Host-side allocation, if any. Zero on the pageable path; a staging pool
    /// would report its first-touch cost here.
    pub host_alloc_ns: u64,
    /// Host-side re-pack of the columns into another layout.
    pub host_pack_ns: u64,
    /// Any per-edge conversion between the CPU layout and the uploaded one.
    pub layout_transform_ns: u64,

    /// Per column, indexed by [`Column::index`].
    pub columns: [ColumnLoad; 5],
    /// Sum of `columns[..].alloc_ns`.
    pub device_alloc_ns: u64,
    /// An explicit `cuCtxSynchronize` after the last copy. Proves the per-column
    /// numbers above are transfer time and not queued work.
    pub device_sync_ns: u64,
    /// Everything from `bind` to the last synchronise.
    pub total_ns: u64,

    /// Bytes the view handed over.
    pub input_bytes: usize,
    /// Bytes that went through a host staging buffer.
    pub staging_bytes: usize,
    /// Bytes that crossed PCIe.
    pub uploaded_bytes: usize,
    /// Bytes the projection left on the host. Not an error and not a saving to
    /// celebrate: it is the part of the graph a kernel cannot read yet, so it is
    /// the size of the eventual second upload if a later operator needs it.
    pub deferred_bytes: usize,
    /// What [`crate::SnapshotView::bytes`] priced for this upload. Must equal
    /// `uploaded_bytes`, or the budget is being charged for something other than
    /// what is resident.
    pub priced_bytes: usize,
}

impl GraphLoadMetrics {
    pub fn new(snapshot_id: SnapshotId, nodes: usize, edges: usize, dst_layout: DstLayout) -> Self {
        Self {
            snapshot_id,
            nodes,
            edges,
            dst_layout,
            projection: ColumnSet::ALL,
            snapshot_materialize_ns: None,
            bind_ns: 0,
            host_alloc_ns: 0,
            host_pack_ns: 0,
            layout_transform_ns: 0,
            columns: [ColumnLoad::default(); 5],
            device_alloc_ns: 0,
            device_sync_ns: 0,
            total_ns: 0,
            input_bytes: 0,
            staging_bytes: 0,
            uploaded_bytes: 0,
            deferred_bytes: 0,
            priced_bytes: 0,
        }
    }

    /// Attach a caller-measured materialisation cost.
    pub fn with_materialize_ns(mut self, ns: u64) -> Self {
        self.snapshot_materialize_ns = Some(ns);
        self
    }

    pub fn record_column(&mut self, column: Column, bytes: usize, alloc_ns: u64, h2d_ns: u64) {
        self.columns[column.index()] = ColumnLoad {
            bytes,
            alloc_ns,
            h2d_ns,
        };
        self.device_alloc_ns += alloc_ns;
    }

    pub fn column(&self, column: Column) -> ColumnLoad {
        self.columns[column.index()]
    }

    /// Time spent moving this graph across PCIe.
    pub fn upload_ns(&self) -> u64 {
        self.columns.iter().map(|c| c.h2d_ns).sum()
    }

    /// Bytes per edge as actually uploaded — offsets amortised over the edges,
    /// so it lands on 23.0 rather than 22.0 for a graph with out-degree 4.
    pub fn uploaded_bytes_per_edge(&self) -> f64 {
        self.uploaded_bytes as f64 / self.edges.max(1) as f64
    }

    /// Effective PCIe throughput of this upload.
    pub fn upload_gbps(&self) -> f64 {
        let s = self.upload_ns() as f64 / 1e9;
        if s > 0.0 {
            self.uploaded_bytes as f64 / s / 1e9
        } else {
            0.0
        }
    }

    /// True when nothing on the host touched the columns — the loader is a pure
    /// series of `memcpy`s out of the caller's own arrays.
    pub fn is_copy_free(&self) -> bool {
        self.staging_bytes == 0
            && self.host_pack_ns == 0
            && self.layout_transform_ns == 0
            && self.host_alloc_ns == 0
    }

    /// The budget is charged for `priced_bytes` and the device holds
    /// `uploaded_bytes`; they must be the same number or residency silently
    /// over- or under-commits.
    pub fn accounting_holds(&self) -> bool {
        self.priced_bytes == self.uploaded_bytes
    }

    /// True when the upload carried fewer than five columns.
    ///
    /// A projected copy is not a partial load waiting to be completed: it is the
    /// answer to "what does this operator read?" — and it is the number that
    /// decides whether the graph fits in the device at all.
    pub fn is_projected(&self) -> bool {
        self.deferred_bytes > 0
    }

    /// Share of the graph left on the host, 0.0–1.0.
    pub fn deferred_share(&self) -> f64 {
        let total = self.uploaded_bytes + self.deferred_bytes;
        if total == 0 {
            return 0.0;
        }
        self.deferred_bytes as f64 / total as f64
    }

    /// Share of the load that was PCIe transfer, 0.0–1.0. The number that
    /// decides whether optimising the loader is worth anything at all.
    pub fn upload_share(&self) -> f64 {
        if self.total_ns == 0 {
            return 0.0;
        }
        self.upload_ns() as f64 / self.total_ns as f64
    }
}

/// One operator invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpRecord {
    pub op: Op,
    pub backend: BackendKind,
    pub decision: BackendDecision,
    /// Wall clock for the operator call only.
    pub micros: u64,
    /// Bytes the backend moved or touched for this call.
    pub bytes: u64,
    /// Operator-specific size (rows, seeds, ...).
    pub work: u64,
    /// Whether the result came from a resident copy.
    pub resident: bool,
}

/// Append-only log for one process or one benchmark iteration.
#[derive(Debug, Default)]
pub struct AcceleratorMetrics {
    records: Vec<OpRecord>,
}

impl AcceleratorMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, record: OpRecord) {
        self.records.push(record);
    }

    pub fn records(&self) -> &[OpRecord] {
        &self.records
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Total wall clock for one operator on one backend.
    pub fn micros(&self, op: Op, backend: BackendKind) -> u64 {
        self.records
            .iter()
            .filter(|r| r.op == op && r.backend == backend)
            .map(|r| r.micros)
            .sum()
    }

    pub fn calls(&self, op: Op, backend: BackendKind) -> usize {
        self.records
            .iter()
            .filter(|r| r.op == op && r.backend == backend)
            .count()
    }

    /// Why the CPU answered, counted by reason. This is the line that explains a
    /// disappointing speedup without re-running anything.
    pub fn fallback_reasons(&self) -> Vec<(&'static str, usize)> {
        let mut out: Vec<(&'static str, usize)> = Vec::new();
        for r in &self.records {
            let Some(reason) = r.decision.reason() else {
                continue;
            };
            let name = reason.as_str();
            match out.iter_mut().find(|(n, _)| *n == name) {
                Some((_, c)) => *c += 1,
                None => out.push((name, 1)),
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Vectors: the upload and the scan
// ---------------------------------------------------------------------------

/// What one `load_vectors` cost.
///
/// Shorter than [`GraphLoadMetrics`] on purpose — a vector matrix is one array,
/// not five columns, so there is no per-column table to keep. What it does keep
/// is the one stage a graph upload does not have: **normalisation**. Cosine
/// space normalises every row, and that is a host pass over the whole matrix
/// (51 MB at 1e5 x 128) which would otherwise hide inside "upload took 30 ms"
/// and be mistaken for PCIe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorLoadMetrics {
    pub vector_space_id: u32,
    pub snapshot_id: SnapshotId,
    pub rows: usize,
    pub dim: usize,
    /// What the view priced; must equal `uploaded_bytes`.
    pub priced_bytes: usize,
    /// What the backend really allocated.
    pub uploaded_bytes: usize,
    /// Host pass that unit-normalises rows. Zero for `L2` / `Ip`, where the
    /// matrix is uploaded as it was handed over.
    pub normalize_ns: u64,
    pub alloc_ns: u64,
    pub copy_ns: u64,
    pub total_ns: u64,
}

impl VectorLoadMetrics {
    pub fn new(vector_space_id: u32, snapshot_id: SnapshotId, rows: usize, dim: usize) -> Self {
        Self {
            vector_space_id,
            snapshot_id,
            rows,
            dim,
            priced_bytes: 0,
            uploaded_bytes: 0,
            normalize_ns: 0,
            alloc_ns: 0,
            copy_ns: 0,
            total_ns: 0,
        }
    }

    /// Bytes per row, `dim` floats plus the id.
    pub fn bytes_per_row(&self) -> f64 {
        if self.rows == 0 {
            0.0
        } else {
            self.uploaded_bytes as f64 / self.rows as f64
        }
    }

    pub fn upload_gbps(&self) -> f64 {
        if self.copy_ns == 0 {
            0.0
        } else {
            self.uploaded_bytes as f64 / (self.copy_ns as f64 / 1e9) / 1e9
        }
    }

    /// What fraction of the load was the normalisation pass rather than the
    /// link. Above ~0.5 the upload is CPU-bound and PCIe is not the lever.
    pub fn normalize_share(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.normalize_ns as f64 / self.total_ns as f64
        }
    }

    /// The budget must be charged for exactly what was allocated, or residency
    /// silently over- or under-commits.
    pub fn accounting_holds(&self) -> bool {
        self.priced_bytes == self.uploaded_bytes
    }
}

/// What one `knn_flat` call cost, split the way a pipeline can act on it.
///
/// Four stages with four different fixes: `upload_ns` is query bytes over PCIe,
/// `kernel_ns` is the scan, `download_ns` is the result over PCIe, and
/// `merge_ns` is the host-side reduction of the per-split partial lists. A
/// benchmark that reports only the total cannot tell "the kernel is slow" from
/// "the results are too big to bring back" — and at small `k` the latter wins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnnMetrics {
    pub queries: usize,
    pub rows: usize,
    pub dim: usize,
    pub k: usize,
    /// Blocks per query in the row dimension: the grid's `y`.
    pub splits: usize,
    pub filtered: bool,
    /// Query rows + candidate bitmap, host to device.
    pub upload_bytes: usize,
    /// `k * (4 + 8)` bytes per query, device to host, before trimming.
    pub result_bytes: usize,
    pub upload_ns: u64,
    pub kernel_ns: u64,
    pub download_ns: u64,
    /// Host merge of `splits` sorted lists per query, into the final top-k.
    pub merge_ns: u64,
    pub total_ns: u64,
}

impl KnnMetrics {
    pub fn new(queries: usize, rows: usize, dim: usize, k: usize) -> Self {
        Self {
            queries,
            rows,
            dim,
            k,
            splits: 1,
            filtered: false,
            upload_bytes: 0,
            result_bytes: 0,
            upload_ns: 0,
            kernel_ns: 0,
            download_ns: 0,
            merge_ns: 0,
            total_ns: 0,
        }
    }

    /// Corpus bytes one query has to read. Multiply by `queries` for the work,
    /// divide by `kernel_ns` for the achieved bandwidth — which is the number
    /// that says whether the scan is memory-bound (it should be) or has found
    /// something to do per row that it should not.
    pub fn bytes_per_pass(&self) -> usize {
        self.rows * self.dim * std::mem::size_of::<f32>()
    }

    /// Effective read bandwidth, counting one corpus pass per query. Above the
    /// card's HBM bandwidth when the batch reuses L2, which is expected and is
    /// not an error — `bench_knn` compares it against `cuMemGetInfo`-sized
    /// reality rather than against the datasheet.
    pub fn scan_gbps(&self) -> f64 {
        if self.kernel_ns == 0 {
            0.0
        } else {
            self.bytes_per_pass() as f64 * self.queries as f64
                / (self.kernel_ns as f64 / 1e9)
                / 1e9
        }
    }

    /// Per-query latency, the number a single-query caller cares about.
    pub fn per_query_us(&self) -> f64 {
        if self.queries == 0 {
            0.0
        } else {
            self.total_ns as f64 / self.queries as f64 / 1e3
        }
    }

    /// The stage that dominated. Printed by the bench so a regression names
    /// itself.
    pub fn dominant_stage(&self) -> &'static str {
        let mut best = ("none", 0u64);
        for (name, ns) in [
            ("upload", self.upload_ns),
            ("kernel", self.kernel_ns),
            ("download", self.download_ns),
            ("merge", self.merge_ns),
        ] {
            if ns > best.1 {
                best = (name, ns);
            }
        }
        best.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::DecisionReason;

    fn rec(op: Op, backend: BackendKind, micros: u64, decision: BackendDecision) -> OpRecord {
        OpRecord {
            op,
            backend,
            decision,
            micros,
            bytes: 0,
            work: 0,
            resident: true,
        }
    }

    #[test]
    fn totals_are_per_operator_and_per_backend() {
        let mut m = AcceleratorMetrics::new();
        m.record(rec(Op::KnnFlat, BackendKind::Cuda, 10, BackendDecision::CudaResident));
        m.record(rec(Op::KnnFlat, BackendKind::Cuda, 30, BackendDecision::CudaResident));
        m.record(rec(
            Op::KnnFlat,
            BackendKind::Cpu,
            900,
            BackendDecision::Cpu {
                reason: DecisionReason::BelowThreshold {
                    work: 1,
                    threshold: 2,
                },
            },
        ));
        assert_eq!(m.micros(Op::KnnFlat, BackendKind::Cuda), 40);
        assert_eq!(m.calls(Op::KnnFlat, BackendKind::Cuda), 2);
        assert_eq!(m.micros(Op::KnnFlat, BackendKind::Cpu), 900);
        assert_eq!(m.micros(Op::BatchKhop, BackendKind::Cuda), 0);
        assert_eq!(m.fallback_reasons(), vec![("below_threshold", 1)]);
    }

    fn loaded(edges: usize, layout: DstLayout) -> GraphLoadMetrics {
        let mut m = GraphLoadMetrics::new(1, edges / 4, edges, layout);
        // Only the accounting matters in these tests; the columns are counted,
        // not moved. Distinct nans per column so a mixed-up index shows up.
        m.record_column(Column::Offsets, (edges / 4 + 1) * 4, 1, 2);
        m.record_column(Column::Dst, edges * layout.bytes_per_edge(), 3, 4);
        m.record_column(Column::ValidFrom, edges * 8, 5, 6);
        m.record_column(Column::ValidTo, edges * 8, 7, 8);
        m.record_column(Column::EdgeType, edges * 2, 9, 10);
        m.uploaded_bytes = m.columns.iter().map(|c| c.bytes).sum();
        m.priced_bytes = m.uploaded_bytes;
        m.total_ns = 100;
        m
    }

    #[test]
    fn a_load_reports_every_stage_and_adds_up() {
        let m = loaded(10_000_000, DstLayout::U32);
        assert_eq!(m.upload_ns(), 2 + 4 + 6 + 8 + 10);
        assert_eq!(m.device_alloc_ns, 1 + 3 + 5 + 7 + 9);
        assert_eq!(m.column(Column::Dst).bytes, 40_000_000);
        assert!(m.accounting_holds());
        assert_eq!(m.column(Column::Offsets).bytes, (2_500_001) * 4);
    }

    #[test]
    fn the_per_edge_figure_is_the_one_the_break_even_uses() {
        // 22 B/edge plus the offsets column, which is per node: out-degree 4
        // means +1 B/edge (plus one terminal offset), so 23.0 — the number §9.2
        // measured, not the 22.0 the layout alone would suggest.
        let narrow = loaded(10_000_000, DstLayout::U32);
        assert!((narrow.uploaded_bytes_per_edge() - 23.0).abs() < 1e-5);
        let wide = loaded(10_000_000, DstLayout::U64);
        assert!((wide.uploaded_bytes_per_edge() - 27.0).abs() < 1e-5);
        // The 4 B/edge difference is the whole point of the narrow column.
        assert_eq!(wide.uploaded_bytes - narrow.uploaded_bytes, 40_000_000);
    }

    #[test]
    fn a_clean_load_is_copy_free_and_says_so() {
        let m = loaded(1_000, DstLayout::U32);
        assert!(m.is_copy_free());
        assert_eq!(m.snapshot_materialize_ns, None, "unmeasured, not zero");

        // Each of the three hidden-copy risks, one at a time: any of them makes
        // the loader no longer a pure memcpy, and `is_copy_free` is what a test
        // asserts instead of a comment claiming there is no AoS pass.
        let mut staged = m.clone();
        staged.staging_bytes = 1;
        assert!(!staged.is_copy_free());

        let mut packed = m.clone();
        packed.host_pack_ns = 1;
        assert!(!packed.is_copy_free());

        let mut converted = m.clone();
        converted.layout_transform_ns = 1;
        assert!(!converted.is_copy_free());

        let mut allocated = m.clone();
        allocated.host_alloc_ns = 1;
        assert!(!allocated.is_copy_free());
    }

    #[test]
    fn a_mispriced_upload_fails_the_accounting_check() {
        let mut m = loaded(1_000, DstLayout::U64);
        assert!(m.accounting_holds());
        // A backend that allocates a different total than the view priced would
        // be charging the budget for bytes it does not hold.
        m.uploaded_bytes += 4;
        assert!(!m.accounting_holds());
    }

    #[test]
    fn upload_share_separates_the_copy_from_the_rest() {
        let mut m = loaded(1_000, DstLayout::U32);
        // 30 ns of a 100 ns load: a loader that is 70% something else is a
        // loader where the PCIe link is not the thing to optimise.
        assert!((m.upload_share() - 0.30).abs() < 1e-9);
        m.total_ns = 0;
        assert_eq!(m.upload_share(), 0.0);
    }

    #[test]
    fn a_vector_load_names_the_normalisation_pass() {
        let mut m = VectorLoadMetrics::new(7, 3, 100_000, 128);
        m.priced_bytes = 100_000 * (128 * 4 + 8);
        m.uploaded_bytes = m.priced_bytes;
        m.normalize_ns = 15_000_000;
        m.copy_ns = 30_000_000;
        m.total_ns = 45_000_000;

        assert!(m.accounting_holds());
        assert!((m.bytes_per_row() - 520.0).abs() < 1e-9);
        // 51.2 MB in 30 ms is 1.7 GB/s of *copy* — the number to compare with
        // the link, not with the 45 ms total.
        assert!((m.upload_gbps() - 1.7).abs() < 0.1);
        assert!((m.normalize_share() - 1.0 / 3.0).abs() < 1e-9);

        // An L2 space does no pass, and the zero has to read as "not done"
        // rather than as "done instantly".
        let mut l2 = m.clone();
        l2.normalize_ns = 0;
        l2.total_ns = 30_000_000;
        assert_eq!(l2.normalize_share(), 0.0);

        // Mispriced upload is the same failure as in the graph path.
        l2.uploaded_bytes += 1;
        assert!(!l2.accounting_holds());
    }

    #[test]
    fn a_knn_call_names_the_stage_that_dominated() {
        let mut m = KnnMetrics::new(8, 100_000, 128, 10);
        m.splits = 8;
        m.result_bytes = 8 * 10 * 12;
        m.upload_bytes = 8 * 128 * 4;
        m.upload_ns = 10_000;
        m.kernel_ns = 970_000;
        m.download_ns = 20_000;
        m.merge_ns = 1_000;
        m.total_ns = 1_001_000;

        assert_eq!(m.dominant_stage(), "kernel");
        assert_eq!(m.bytes_per_pass(), 100_000 * 128 * 4);
        // 8 passes of 51.2 MB in 0.97 ms: 422 GB/s, i.e. well above HBM and
        // therefore L2 reuse across the batch. Real, and not a data error.
        assert!(m.scan_gbps() > 400.0);
        assert!((m.per_query_us() - 125.1).abs() < 1.0);

        // With no kernel measured the derived numbers are zero, not a division
        // by zero and not an infinity in a log line.
        let empty = KnnMetrics::new(0, 0, 0, 0);
        assert_eq!(empty.scan_gbps(), 0.0);
        assert_eq!(empty.per_query_us(), 0.0);
        assert_eq!(empty.dominant_stage(), "none");
    }
}
