//! M2: delayed garbage collection primitives.
//!
//! Two rules make GC safe without a reference-counting filesystem:
//!
//! 1. **Explicit retention floor.** `AS KNOWN AT S` for *any* older S needs the
//!    retired segments, deltas, node lifetimes and observation versions. So
//!    nothing is collected until a caller declares `retain_from = F`: queries
//!    below F are then rejected loudly instead of silently returning a
//!    different answer. Default policy collects nothing.
//! 2. **Publish before unlink.** The manifest is rewritten to drop every
//!    reference *first*; only then are the files removed. A crash between the
//!    two leaves unreferenced files (invisible, reclaimable on the next run),
//!    never a manifest pointing at a deleted file.
//!
//! `ReadScope` is the mmap-reader guard: `gc()` refuses to run while any
//! version query, aggregate projection or background compaction is in flight.

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::error::Result;

/// Retention decision. `GcPolicy::default()` is a no-op dry run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcPolicy {
    /// Queries pinned below this system time become unsupported after GC.
    /// `None` = keep all history (never collect).
    pub retain_from: Option<i64>,
    /// Never collect the newest N published snapshot points, even when they
    /// fall below `retain_from`.
    pub keep_last_n: usize,
    /// Report candidates without touching the manifest or any file.
    pub dry_run: bool,
}

impl GcPolicy {
    /// Development default: full history, no deletion.
    pub fn never() -> Self {
        Self {
            retain_from: None,
            keep_last_n: 2,
            dry_run: true,
        }
    }

    /// Production-style policy: actually reclaim everything below `floor`.
    pub fn collect(floor: i64) -> Self {
        Self {
            retain_from: Some(floor),
            keep_last_n: 2,
            dry_run: false,
        }
    }
}

impl Default for GcPolicy {
    fn default() -> Self {
        Self::never()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GcReport {
    pub cutoff: i64,
    pub dry_run: bool,
    pub active_readers: usize,
    pub segments_removed: usize,
    pub deltas_removed: usize,
    pub node_versions_removed: usize,
    pub observations_removed: usize,
    pub orphan_files_removed: usize,
    pub bytes_reclaimed: u64,
    pub manifest_rewritten: bool,
}

impl GcReport {
    pub fn no_op(active_readers: usize) -> Self {
        Self {
            dry_run: true,
            active_readers,
            ..Self::default()
        }
    }
}

/// RAII ticket for an in-flight reader. Cheap (one atomic increment) and only
/// taken on control-plane queries, never inside `TemporalCSR::neighbors()`.
#[derive(Debug)]
pub struct ReadScope {
    counter: Arc<AtomicUsize>,
}

impl ReadScope {
    pub fn new(counter: &Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Self {
            counter: counter.clone(),
        }
    }
}

impl Drop for ReadScope {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::SeqCst);
    }
}

pub fn reader_count(counter: &Arc<AtomicUsize>) -> usize {
    counter.load(Ordering::SeqCst)
}

/// Delete one file, returning its reclaimed size. A missing file is not an
/// error: a previous GC may already have removed it after publishing.
pub fn remove_file(path: &Path) -> Result<u64> {
    match fs::metadata(path) {
        Ok(m) => {
            fs::remove_file(path)?;
            Ok(m.len())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_never_collects() {
        let p = GcPolicy::default();
        assert_eq!(p.retain_from, None);
        assert!(p.dry_run);
        assert_eq!(GcPolicy::collect(42).dry_run, false);
    }

    #[test]
    fn read_scope_blocks_only_while_alive() {
        let c = Arc::new(AtomicUsize::new(0));
        assert_eq!(reader_count(&c), 0);
        {
            let _a = ReadScope::new(&c);
            let _b = ReadScope::new(&c);
            assert_eq!(reader_count(&c), 2);
        }
        assert_eq!(reader_count(&c), 0);
    }
}
