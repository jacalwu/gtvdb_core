//! The backend contract: what an accelerator must be able to do.
//!
//! Three operations are in scope because these are the ones the engine can hand
//! off without moving a result that is as large as its input:
//!
//! * [`AccelOp::ExactKnn`] — returns `k` hits per query from an `N`-row corpus.
//! * [`AccelOp::IvfProbe`] — returns `k` hits from the `nprobe` nearest cells
//!   of a resident, cell-ordered corpus (typically ~64 MB scanned, not 2 GB).
//! * [`AccelOp::AsofJoin`] — returns one matched index per left row.
//!
//! Everything else (window functions, `k`-hop traversal, pattern matching) was
//! deliberately left out: their output is the same size as their input or their
//! access pattern is pointer-chasing, so the PCIe round-trip eats the kernel
//! gain. Putting them behind this trait would only advertise a loss.

use arrow::array::BooleanArray;
use gtv_core::Metric;

/// Which kind of device a backend executes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Cuda,
}

impl DeviceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Cuda => "cuda",
        }
    }
}

/// Static description of a backend's device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    pub kind: DeviceKind,
    /// Reported device memory (`0` for the host backend).
    pub total_bytes: u64,
}

impl DeviceInfo {
    /// The always-available host backend.
    pub fn cpu() -> Self {
        Self {
            name: "cpu".to_string(),
            kind: DeviceKind::Cpu,
            total_bytes: 0,
        }
    }
}

/// An operation that can be offloaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelOp {
    ExactKnn,
    IvfProbe,
    AsofJoin,
}

impl AccelOp {
    pub fn as_str(self) -> &'static str {
        match self {
            AccelOp::ExactKnn => "exact_knn",
            AccelOp::IvfProbe => "ivf_probe",
            AccelOp::AsofJoin => "asof_join",
        }
    }
}

/// One `(distance, id)` candidate. Distances are in the metric's canonical form
/// (squared L2, `1 - cos`, negated inner product), exactly what
/// [`gtv_index::exact_topk`] returns — callers decide how to present them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawHit {
    pub distance: f32,
    pub id: u64,
}

/// A zero-copy exact K-NN request over a borrowed corpus.
#[derive(Debug, Clone)]
pub struct KnnRequest<'a> {
    /// Cache name (the collection/index name).
    pub name: &'a str,
    /// Content version. A new version invalidates the device copy.
    pub version: u64,
    pub ids: &'a [u64],
    /// Row-major, `ids.len() * dim` floats.
    pub data: &'a [f32],
    pub dim: usize,
    pub metric: Metric,
    /// Whether rows (and therefore the query) are already unit-length.
    pub normalized: bool,
    pub query: &'a [f32],
    pub k: usize,
    pub mask: Option<&'a BooleanArray>,
}

impl KnnRequest<'_> {
    /// Bytes a full device copy of this corpus would occupy.
    pub fn resident_bytes(&self) -> u64 {
        (self.ids.len() * 8 + self.data.len() * 4) as u64
    }

    pub fn rows(&self) -> u64 {
        self.ids.len() as u64
    }
}

/// An IVF probe request over a *cell-ordered* corpus.
///
/// `positions` comes from [`gtv_index::IvfIndex::probe_positions`]: ascending,
/// unique row indices into `ids` / `data`, with the filter mask already applied.
/// Both the host and device backends consume that same list, so cell selection
/// is not duplicated and the device's answer is exactly the host's answer.
/// The corpus stays wholly resident — only the (small) position list moves per
/// query.
#[derive(Debug, Clone)]
pub struct IvfRequest<'a> {
    /// Cache name (the collection/index name).
    pub name: &'a str,
    /// Content version. A new version invalidates the device copy.
    pub version: u64,
    /// Ids in cell order. A returned position `p` maps to `ids[p]`.
    pub ids: &'a [u64],
    /// Row-major, cell-ordered `f32` corpus: `ids.len() * dim` floats.
    pub data: &'a [f32],
    pub dim: usize,
    pub metric: Metric,
    /// Whether rows (and therefore the query) are already unit-length.
    pub normalized: bool,
    pub query: &'a [f32],
    pub k: usize,
    /// Corpus rows to scan (probed cells, mask applied).
    pub positions: &'a [u32],
}

impl IvfRequest<'_> {
    /// Bytes a full device copy of the cell-ordered corpus would occupy.
    pub fn resident_bytes(&self) -> u64 {
        (self.ids.len() * 8 + self.data.len() * 4) as u64
    }

    pub fn rows(&self) -> u64 {
        self.ids.len() as u64
    }

    /// Rows the query actually scans (the IVF cost proxy).
    pub fn scanned_rows(&self) -> usize {
        self.positions.len()
    }
}

/// An as-of join request: match every `left` timestamp against `right_times`.
#[derive(Debug, Clone)]
pub struct AsofRequest<'a> {
    pub name: &'a str,
    pub version: u64,
    pub left: &'a [i64],
    /// Ascending.
    pub right_times: &'a [i64],
    /// Bytes of the carried payload (price/spread columns) that must also be
    /// resident for the device path to be worthwhile.
    pub payload_bytes: u64,
    pub tolerance_ns: Option<i64>,
}

impl AsofRequest<'_> {
    pub fn resident_bytes(&self) -> u64 {
        self.right_times.len() as u64 * 8 + self.payload_bytes
    }

    pub fn rows(&self) -> u64 {
        self.right_times.len() as u64
    }
}

/// Why a backend could not execute. Every variant is recoverable: the registry
/// records it and runs the operation on the host.
#[derive(Debug, thiserror::Error)]
pub enum AccelError {
    #[error("no device available: {0}")]
    NoDevice(String),
    #[error("device memory exhausted: need {need} B, have {free} B")]
    OutOfMemory { need: u64, free: u64 },
    #[error("kernel launch failed: {0}")]
    Launch(String),
    #[error("`{op}` is not supported on {device}")]
    Unsupported { device: String, op: &'static str },
    #[error("invalid request: {0}")]
    Invalid(String),
}

/// A device (or host) execution engine.
///
/// Implementations must be safe to call from several threads: the registry hands
/// out `&self` and relies on the backend to serialize its own stream access.
pub trait AccelBackend: Send + Sync + std::fmt::Debug {
    /// Static device description.
    fn device(&self) -> &DeviceInfo;

    /// Whether `op` can run here.
    fn supports(&self, op: AccelOp) -> bool;

    /// Exact K-NN. Must rank nearest-first, ties broken by ascending id.
    fn exact_knn(&self, req: &KnnRequest<'_>) -> Result<Vec<RawHit>, AccelError>;

    /// IVF probe: exact top-K over `req.positions`.
    ///
    /// A backend that cannot do it returns [`AccelError::Unsupported`] and the
    /// registry runs the identical host scan. Ranking must match
    /// `gtv_index::exact_topk_positions`.
    fn ivf_probe(&self, req: &IvfRequest<'_>) -> Result<Vec<RawHit>, AccelError>;

    /// As-of join returning the matched right-row index per left row.
    fn asof_join(&self, req: &AsofRequest<'_>) -> Result<Vec<Option<u32>>, AccelError>;

    /// Live `(free, total)` device memory in bytes, when reportable.
    ///
    /// The registry feeds this into the VRAM budget so `CUDA_VRAM` stays a
    /// ceiling rather than a claim about what the driver actually has free.
    /// Host backends return `None`, which leaves the budget rigid.
    fn mem_info(&self) -> Option<(u64, u64)> {
        None
    }

    /// Drop any device copy this backend keeps for `(kind, name)`.
    ///
    /// The registry calls this when it evicts a block, so the ledger and the
    /// backend's own buffers cannot drift apart into an over-commit. The host
    /// backend holds nothing, hence the default no-op.
    fn release(&self, kind: crate::resident::ResidentKind, name: &str) {
        let _ = (kind, name);
    }
}
