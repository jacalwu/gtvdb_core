//! M3: optional resident maintenance worker with automatic retention.
//!
//! Design constraint from the review (§6 / §7): **no resident lock on the
//! store**. The worker never owns the `SegmentStore`; it only receives owned
//! [`CompactionInput`] snapshots after a publish, runs the very same phase-1
//! `prepare()` as the synchronous path, and leaves the resulting plan in a
//! single slot. The foreground takes the plan and calls
//! `SegmentStore::apply_compaction()` in its own short `&mut` critical section.
//!
//! Consequences:
//!
//! * the worker writes no file, advances no manifest and holds no store lock;
//! * a stale plan is still rejected by `apply_compaction()`'s three-cursor
//!   check, so a concurrent publish can never rewrite newer data;
//! * stopping the service cannot corrupt anything — an unused plan is simply
//!   dropped, and the next explicit `materialize()` recomputes it;
//! * GC stays a foreground, explicit decision. `gc_policy()` only *derives* a
//!   policy from the configured retention window; it never deletes a file.
//!
//! The service is opt-in: nothing runs until [`MaintenanceService::start`] is
//! called, and the default config keeps all history (`retain_window = None`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::error::Result;
use crate::gc::{GcPolicy, ReadScope};
use crate::store::{CompactionInput, CompactionPlan, SegmentStore, MAX_DELTA_DEPTH};

/// How often the worker wakes up to notice the stop flag while idle.
const IDLE_POLL: Duration = Duration::from_millis(20);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaintenanceConfig {
    /// Prepare a plan once the live overlay reaches this depth. `0` disables
    /// automatic compaction (explicit `materialize()` still works).
    pub compact_when_depth: usize,
    /// Keep the newest `window` system-time units of history. `None` = keep
    /// everything, i.e. `GcPolicy::never()`.
    pub retain_window: Option<i64>,
    /// Never collect the newest N published snapshot points.
    pub keep_last_n: usize,
    /// Development default: report GC candidates without deleting them.
    pub gc_dry_run: bool,
}

impl Default for MaintenanceConfig {
    fn default() -> Self {
        Self {
            compact_when_depth: MAX_DELTA_DEPTH,
            retain_window: None,
            keep_last_n: 2,
            gc_dry_run: true,
        }
    }
}

/// A prepared plan or the reason preparation failed. Stored as a plain enum
/// so the slot never needs to move a `StorageError` across threads.
#[derive(Debug)]
enum PlanSlot {
    Ready(CompactionPlan),
    Failed(String),
}

#[derive(Debug)]
struct Shared {
    plan: Mutex<Option<PlanSlot>>,
    stop: AtomicBool,
    submitted: AtomicUsize,
    dropped: AtomicUsize,
    plans_produced: AtomicUsize,
    plans_taken: AtomicUsize,
    last_error: Mutex<Option<String>>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MaintenanceStatus {
    pub running: bool,
    pub submitted: usize,
    pub dropped: usize,
    pub plans_produced: usize,
    pub plans_taken: usize,
    pub pending_plan: bool,
    pub last_error: Option<String>,
}

/// Resident maintenance worker. `start()` spawns exactly one OS thread;
/// `stop()` joins it and is idempotent (also called on `Drop`).
#[derive(Debug)]
pub struct MaintenanceService {
    config: MaintenanceConfig,
    shared: Arc<Shared>,
    tx: Option<mpsc::SyncSender<CompactionInput>>,
    handle: Option<JoinHandle<()>>,
}

impl MaintenanceService {
    /// Start the worker. The foreground keeps full ownership of the store.
    pub fn start(config: MaintenanceConfig) -> Self {
        let shared = Arc::new(Shared {
            plan: Mutex::new(None),
            stop: AtomicBool::new(false),
            submitted: AtomicUsize::new(0),
            dropped: AtomicUsize::new(0),
            plans_produced: AtomicUsize::new(0),
            plans_taken: AtomicUsize::new(0),
            last_error: Mutex::new(None),
        });
        // A bounded channel is the back-pressure policy: if the foreground is
        // publishing faster than it applies plans, the newest unprocessed
        // snapshot is simply skipped. Nothing is lost — the head is still
        // readable, and the next submit after apply will prepare again.
        let (tx, rx) = mpsc::sync_channel::<CompactionInput>(1);
        let worker_shared = shared.clone();
        let compact_when_depth = config.compact_when_depth;
        let handle = thread::spawn(move || loop {
            if worker_shared.stop.load(Ordering::SeqCst) {
                break;
            }
            match rx.recv_timeout(IDLE_POLL) {
                Ok(input) => {
                    if compact_when_depth == 0 || input.delta_depth() < compact_when_depth {
                        continue;
                    }
                    // Keep GC from unlinking the worker's input segments.
                    let _scope = ReadScope::new(&input.readers);
                    let result = input.prepare(input.head() + 1);
                    match result {
                        Ok(plan) => {
                            *worker_shared.plan.lock().unwrap() = Some(PlanSlot::Ready(plan));
                            worker_shared.plans_produced.fetch_add(1, Ordering::SeqCst);
                        }
                        Err(e) => {
                            let msg = e.to_string();
                            *worker_shared.last_error.lock().unwrap() = Some(msg.clone());
                            *worker_shared.plan.lock().unwrap() = Some(PlanSlot::Failed(msg));
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        });

        Self {
            config,
            shared,
            tx: Some(tx),
            handle: Some(handle),
        }
    }

    pub fn config(&self) -> &MaintenanceConfig {
        &self.config
    }

    /// Hand a read-only snapshot to the worker. Non-blocking; returns `false`
    /// when the queue already holds a newer snapshot or the service is stopped.
    pub fn submit(&self, input: CompactionInput) -> bool {
        let Some(tx) = &self.tx else {
            return false;
        };
        match tx.try_send(input) {
            Ok(()) => {
                self.shared.submitted.fetch_add(1, Ordering::SeqCst);
                true
            }
            Err(_) => {
                self.shared.dropped.fetch_add(1, Ordering::SeqCst);
                false
            }
        }
    }

    /// Convenience wrapper: submit the store's current read-only state.
    pub fn submit_store(&self, store: &SegmentStore) -> bool {
        self.submit(store.compaction_input())
    }

    /// Take the prepared plan, if any. The caller applies it with
    /// `store.apply_compaction(plan)`; a stale plan is rejected there.
    pub fn take_plan(&self) -> Option<Result<CompactionPlan>> {
        let slot = self.shared.plan.lock().unwrap().take()?;
        self.shared.plans_taken.fetch_add(1, Ordering::SeqCst);
        Some(match slot {
            PlanSlot::Ready(p) => Ok(p),
            // A phase-1 failure is a control-plane error, never a data change.
            // Surface it as a regular storage error on the foreground thread.
            PlanSlot::Failed(msg) => Err(crate::error::StorageError::Msg(format!(
                "background compaction prepare failed: {msg}"
            ))),
        })
    }

    /// Derive a GC policy from the configured retention window. A rolling
    /// window is measured from the current head so a long-running store does
    /// not need a hard-coded absolute floor.
    pub fn gc_policy(&self, head: i64) -> GcPolicy {
        match self.config.retain_window {
            None => GcPolicy::never(),
            Some(window) => GcPolicy {
                retain_from: Some(head.saturating_sub(window.max(0))),
                keep_last_n: self.config.keep_last_n,
                dry_run: self.config.gc_dry_run,
            },
        }
    }

    pub fn status(&self) -> MaintenanceStatus {
        MaintenanceStatus {
            running: self.handle.is_some(),
            submitted: self.shared.submitted.load(Ordering::SeqCst),
            dropped: self.shared.dropped.load(Ordering::SeqCst),
            plans_produced: self.shared.plans_produced.load(Ordering::SeqCst),
            plans_taken: self.shared.plans_taken.load(Ordering::SeqCst),
            pending_plan: self.shared.plan.lock().unwrap().is_some(),
            last_error: self.shared.last_error.lock().unwrap().clone(),
        }
    }

    /// Block until a plan is ready (or the deadline expires). Test helper and
    /// production convenience; the foreground is never forced to wait.
    pub fn wait_for_plan(&self, timeout: Duration) -> Option<Result<CompactionPlan>> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(p) = self.take_plan() {
                return Some(p);
            }
            if Instant::now() >= deadline {
                return None;
            }
            thread::sleep(Duration::from_millis(2));
        }
    }

    /// Stop and join. Idempotent; safe to call before `Drop`.
    pub fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        // Dropping the sender makes recv_timeout return Disconnected.
        self.tx = None;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for MaintenanceService {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_keeps_all_history() {
        let svc = MaintenanceService::start(MaintenanceConfig::default());
        let p = svc.gc_policy(100);
        assert_eq!(p.retain_from, None);
        assert!(p.dry_run);
    }

    #[test]
    fn rolling_retention_is_measured_from_the_head() {
        let svc = MaintenanceService::start(MaintenanceConfig {
            retain_window: Some(10),
            gc_dry_run: false,
            ..MaintenanceConfig::default()
        });
        let p = svc.gc_policy(100);
        assert_eq!(p.retain_from, Some(90));
        assert!(!p.dry_run);
        assert_eq!(p.keep_last_n, 2);
        // A negative floor simply means "nothing is old enough to collect".
        assert_eq!(svc.gc_policy(5).retain_from, Some(-5));
    }

    #[test]
    fn disabled_compaction_and_idempotent_stop() {
        let mut svc = MaintenanceService::start(MaintenanceConfig {
            compact_when_depth: 0,
            ..MaintenanceConfig::default()
        });
        assert_eq!(svc.status().plans_produced, 0);
        assert!(svc.status().running);
        svc.stop();
        svc.stop();
        assert!(!svc.status().running);
    }
}
