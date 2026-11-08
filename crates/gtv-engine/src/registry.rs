//! Registered resources that compiled kernel plans and table functions bind to.
//!
//! A `ResourceRegistry` is just four name → snapshot maps: point-in-time
//! interval columns, as-of series, temporal graphs for wash-trade detection, and
//! raw batches. It lives in the core (not in an extension) because three
//! unrelated consumers read it:
//!
//! * the HFT kernel-plan compiler (extension),
//! * the data-quality gate (core),
//! * the observability / health diagnostics (core).
//!
//! Keeping it domain-neutral is what lets the HFT extension be removed from a
//! build without taking `dq_gate` and `dq_report` with it.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;

use gtv_core::TemporalCSR;

/// Interval columns for `pit(T)` / `point_in_time(T)`.
#[derive(Debug)]
pub struct PitResource {
    pub valid_from: Arc<Vec<i64>>,
    pub valid_to: Arc<Vec<i64>>,
}

/// Series for the as-of join family.
#[derive(Debug)]
pub struct AsofResource {
    pub times: Arc<Vec<i64>>,
    pub price: Arc<Vec<f64>>,
    pub spread: Arc<Vec<f64>>,
    pub tolerance: i64,
}

/// Named snapshots that kernel plans and table functions resolve by name.
#[derive(Debug, Default)]
pub struct ResourceRegistry {
    pub pit: HashMap<String, Arc<PitResource>>,
    pub asof: HashMap<String, Arc<AsofResource>>,
    pub wash: HashMap<String, Arc<TemporalCSR>>,
    /// Registered tables, for column-extraction operators.
    pub tables: HashMap<String, Arc<Vec<RecordBatch>>>,
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn table(&self, name: &str) -> Option<Arc<Vec<RecordBatch>>> {
        self.tables.get(name).cloned()
    }

    pub fn source(&self, name: &str) -> Option<Arc<Vec<RecordBatch>>> {
        self.tables.get(name).cloned()
    }
}

/// Historical name, kept so the HFT extension and existing call sites keep
/// compiling. New code should say [`ResourceRegistry`].
pub type HftRegistry = ResourceRegistry;
