//! Backend abstraction for the optional CUDA path — and the residency rules
//! that decide whether a device copy is worth having at all.
//!
//! # Scope (C1)
//!
//! This crate is the *contract*: one trait, one cache, one policy, one error
//! type. It contains no kernel. That ordering is deliberate and measured: a
//! transfer costs more than the kernel it feeds (a 1e7-edge snapshot is 30 ms of
//! PCIe against a 0.556 ms filter kernel), so until a snapshot stays resident,
//! any kernel timing is a measurement of PCIe. Measured since: the transfer is
//! 97% of the *load* but 2.5% of the *cold path*, because materialising the CSR
//! costs 1.6 s. Both numbers are in `docs/cuda-acceleration-plan.md` §9.3, and
//! [`GraphLoadMetrics`] is how a caller tells them apart.
//!
//! So C1 delivers the part that every later stage depends on:
//!
//! | piece | module | what it guarantees |
//! |---|---|---|
//! | [`Accelerator`] | [`api`] | one trait, `Send + Sync`, no panics, every failure falls back |
//! | [`CpuAccelerator`] | [`cpu`] | the reference answer, built on the existing kernels |
//! | [`ResidentCache`] / [`VectorCache`] | [`resident`] | upload once per snapshot, evict by LRU, never serve a stale copy |
//! | [`ExecutionPolicy`] | [`policy`] | one place decides CPU vs device, and says why |
//! | [`AcceleratorRegistry`] | [`registry`] | which backends exist, per operator |
//!
//! # Invariants
//!
//! * **A snapshot id is the cache key; `business_t` is not.** Graph shape is a
//!   function of the snapshot. Keying on time would re-upload for every query.
//! * **Device memory is a cache.** Every error is recoverable by rerunning on
//!   the CPU; [`AcceleratorError::fallback`] is the machine-checkable form of
//!   that promise.
//! * **A resident handle cannot outlive its snapshot unnoticed.**
//!   [`ResidentGraph::check_snapshot`] is called by every operator, and
//!   [`ResidentCache`] drops superseded entries from its index immediately.
//! * **Default builds contain no CUDA.** Only `gtv-engine --features cuda`
//!   links the device backend.
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//! use gtv_accelerator::{Accelerator, CpuAccelerator, ResidentCache, SnapshotView};
//!
//! // 0 -> 1, live on [0, 100)
//! let offsets = [0u32, 1, 1];
//! let dst = [1u64];
//! let vf = [0i64];
//! let vt = [100i64];
//! let et = [1u16];
//!
//! let acc = Arc::new(CpuAccelerator::new());
//! let mut cache = ResidentCache::new(Arc::clone(&acc) as Arc<dyn Accelerator>, None);
//! let view = SnapshotView::new(3, 2, &offsets, &dst, &vf, &vt, &et);
//!
//! // Same snapshot twice: one load, one hit.
//! let a = cache.acquire(view).unwrap().unwrap();
//! let b = cache.acquire(view).unwrap().unwrap();
//! assert!(Arc::ptr_eq(&a, &b));
//! assert_eq!(cache.stats().loads, 1);
//!
//! // The handle answers, and its result carries the snapshot it came from.
//! let live = acc.valid_time_filter(&a, 0).unwrap();
//! assert_eq!(live.to_ids(), vec![0]);
//! assert_eq!(live.snapshot_id(), 3);
//! ```

pub mod api;
pub mod cpu;
pub mod error;
pub mod metrics;
pub mod policy;
pub mod registry;
pub mod resident;

pub use api::{
    Accelerator, BackendKind, Caps, DstLayout, Op, ResidentGraph, ResidentVectors, SnapshotId,
    SnapshotView, VectorView, WindowOp,
};
pub use cpu::{live_at, CpuAccelerator};
pub use error::{AcceleratorError, Fallback};
pub use metrics::{AcceleratorMetrics, Column, ColumnLoad, GraphLoadMetrics, OpRecord};
pub use policy::{BackendDecision, DecisionInputs, DecisionReason, ExecutionPolicy};
pub use registry::AcceleratorRegistry;
pub use resident::{CacheStats, Resident, ResidentCache, VectorCache};
