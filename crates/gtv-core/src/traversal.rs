//! Resource-bounded graph traversal primitives (B1-3 / M1).
//!
//! [`crate::TemporalCSR::khop_bounded`] and the `gtv-pattern` matcher share
//! these types so that a single budget / cancellation contract applies to every
//! graph query:
//!
//! * [`VisitedSet`] — O(1)-reset, generation-stamped bitmap (4 bytes/node).
//! * [`TraversalBudget`] — hard caps on hops, nodes, edges, frontier, rows,
//!   memory and wall-clock time.
//! * [`CancelToken`] — cooperative cancellation from another thread.
//! * [`EdgePredicate`] — push edge filters into the expansion loop.
//! * [`Truncation`] / [`TraversalOutcome`] — *why* a traversal stopped early.
//!
//! # Budget exhaustion is data, not an error
//!
//! Until M1 an over-budget traversal returned `Err(BudgetExceeded)` and dropped
//! everything it had already found. A caller could not distinguish "no pattern"
//! from "did not finish", which is a regulatory distinction, not a performance
//! one. Now:
//!
//! * budget caps **record** a [`Truncation`] and stop expansion, and the
//!   partial result is returned normally;
//! * only cancellation ([`GtvError::Cancelled`]) and a rejected query
//!   ([`GtvError::HighDegreeNode`], [`GtvError::InvalidArgument`]) are errors.
//!
//! The reason for that split: "cancel" means *abandon*, "deadline" means *time
//! is up, give me what you have*.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::csr::Neighbor;
use crate::error::{GtvError, Result};

/// Cooperative cancellation flag shared with a traversal.
pub type CancelToken = Arc<AtomicBool>;

/// Hard limits for a graph traversal. `Default` is effectively unlimited.
#[derive(Debug, Clone)]
pub struct TraversalBudget {
    /// Maximum number of hops (frontiers) to expand.
    pub max_hops: usize,
    /// Maximum cumulative edges scanned across all hops.
    pub max_edges: u64,
    /// Maximum cumulative distinct nodes visited.
    pub max_nodes: u64,
    /// Maximum size of a single frontier.
    pub max_frontier: usize,
    /// Maximum cumulative output nodes emitted.
    pub max_rows: u64,
    /// Maximum estimated frontier + visited memory (bytes).
    pub max_memory_bytes: u64,
    /// When set, a node with a higher degree requires an [`EdgePredicate`] (or
    /// the query is rejected) to avoid one-node blowups.
    pub max_degree: Option<u32>,
    /// Wall-clock deadline.
    pub deadline: Option<Instant>,
}

impl Default for TraversalBudget {
    fn default() -> Self {
        Self {
            max_hops: usize::MAX,
            max_edges: u64::MAX,
            max_nodes: u64::MAX,
            max_frontier: usize::MAX,
            max_rows: u64::MAX,
            max_memory_bytes: u64::MAX,
            max_degree: None,
            deadline: None,
        }
    }
}

impl TraversalBudget {
    /// No limits (equivalent to [`Default`]).
    pub fn unlimited() -> Self {
        Self::default()
    }

    pub fn with_max_hops(mut self, n: usize) -> Self {
        self.max_hops = n;
        self
    }
    pub fn with_max_edges(mut self, n: u64) -> Self {
        self.max_edges = n;
        self
    }
    pub fn with_max_nodes(mut self, n: u64) -> Self {
        self.max_nodes = n;
        self
    }
    pub fn with_max_frontier(mut self, n: usize) -> Self {
        self.max_frontier = n;
        self
    }
    pub fn with_max_rows(mut self, n: u64) -> Self {
        self.max_rows = n;
        self
    }
    pub fn with_max_memory_bytes(mut self, n: u64) -> Self {
        self.max_memory_bytes = n;
        self
    }
    pub fn with_max_degree(mut self, n: u32) -> Self {
        self.max_degree = Some(n);
        self
    }
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }
}

/// Which cap stopped the traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncReason {
    /// `max_edges` — cumulative scanned edges.
    Edges,
    /// `max_nodes` — cumulative distinct nodes.
    Nodes,
    /// `max_frontier` — one hop's frontier.
    Frontier,
    /// `max_rows` — cumulative emitted rows.
    Rows,
    /// `max_hops` — the requested depth exceeded the cap.
    Hops,
    /// `max_memory_bytes` — estimated frontier + visited footprint.
    Memory,
    /// `deadline` — wall clock.
    Deadline,
}

impl TruncReason {
    pub fn as_str(self) -> &'static str {
        match self {
            TruncReason::Edges => "edges",
            TruncReason::Nodes => "nodes",
            TruncReason::Frontier => "frontier",
            TruncReason::Rows => "rows",
            TruncReason::Hops => "hops",
            TruncReason::Memory => "memory",
            TruncReason::Deadline => "deadline",
        }
    }
}

/// Why a traversal stopped before exhausting the graph.
///
/// `hop` is 1-based and identifies the hop during which the cap tripped, so a
/// caller can say "computed 2 of 4 hops" instead of "failed".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Truncation {
    pub reason: TruncReason,
    pub hop: usize,
    pub frontier_at_stop: usize,
    pub limit: u64,
    pub observed: u64,
}

/// A traversal result plus the reason it may be incomplete.
///
/// Generic so that every operator reports the same way:
/// `TraversalOutcome<Vec<UInt64Array>>`, `TraversalOutcome<Vec<Match>>`, …
#[derive(Debug, Clone)]
pub struct TraversalOutcome<T> {
    pub rows: T,
    pub stats: TraversalStats,
    /// `None` means the traversal finished; `Some` means the rows are partial
    /// **and must not be reported as "no match"**.
    pub truncation: Option<Truncation>,
}

impl<T> TraversalOutcome<T> {
    pub fn is_complete(&self) -> bool {
        self.truncation.is_none()
    }

    /// `true` when the query provably found everything there was to find.
    pub fn is_conclusive(&self) -> bool {
        self.truncation.is_none()
    }

    /// Three-state classification for callers that must report an answer.
    pub fn tri_state(&self, non_empty: bool) -> TriState {
        match (non_empty, self.truncation.is_some()) {
            (_, true) => TriState::Incomplete,
            (true, false) => TriState::Found,
            (false, false) => TriState::NoMatch,
        }
    }
}

/// The three outcomes a detection query can have.
///
/// [`TriState::NoMatch`] and [`TriState::Incomplete`] must never be collapsed
/// into one another: the first proves absence, the second proves nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriState {
    /// Finished, and there were rows.
    Found,
    /// Finished, and there was nothing.
    NoMatch,
    /// Did not finish; absence of rows proves nothing.
    Incomplete,
}

/// What a traversal actually did (for telemetry / tests).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TraversalStats {
    pub hops: usize,
    pub edges_scanned: u64,
    pub rows: u64,
    pub peak_frontier: usize,
    /// Distinct nodes visited, when the traversal tracks them.
    pub nodes_visited: u64,
}

impl TraversalStats {
    /// Attach a visited-node count.
    pub fn with_nodes(mut self, n: u64) -> Self {
        self.nodes_visited = n;
        self
    }
}

/// Push an edge filter into the expansion loop so an over-broad traversal can be
/// narrowed *before* the frontier grows.
pub trait EdgePredicate: Send + Sync {
    /// Return `false` to drop this edge from the traversal.
    fn keep(&self, edge: &Neighbor) -> bool;

    /// Whether `keep` reads [`Neighbor::edge_idx`].
    ///
    /// Pull-direction expansion reconstructs a *forward* edge from the
    /// transposed graph and therefore cannot supply a forward edge index. A
    /// predicate that reads it must force push; the traversal does that
    /// automatically, so this is a performance hint that also prevents silent
    /// wrong answers. Default `false`.
    fn needs_edge_identity(&self) -> bool {
        false
    }
}

impl<F> EdgePredicate for F
where
    F: Fn(&Neighbor) -> bool + Send + Sync,
{
    #[inline]
    fn keep(&self, edge: &Neighbor) -> bool {
        self(edge)
    }
}

/// O(1)-reset, generation-stamped visited bitmap over dense node ids.
///
/// `mark` returns `true` the first time a node is seen in the current
/// generation; `reset` just bumps the generation counter, so clearing is O(1)
/// instead of O(n).
#[derive(Debug, Clone)]
pub struct VisitedSet {
    stamp: Vec<u32>,
    gen: u32,
}

impl VisitedSet {
    pub fn new(node_count: usize) -> Self {
        Self {
            stamp: vec![0u32; node_count],
            gen: 1,
        }
    }

    /// Mark `id`; returns `true` if it was not already visited this generation.
    #[inline]
    pub fn mark(&mut self, id: u64) -> bool {
        let i = id as usize;
        if self.stamp[i] == self.gen {
            false
        } else {
            self.stamp[i] = self.gen;
            true
        }
    }

    #[inline]
    pub fn seen(&self, id: u64) -> bool {
        self.stamp[id as usize] == self.gen
    }

    /// Clear in O(1) by advancing the generation.
    pub fn reset(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.gen = 1;
        }
    }

    /// Estimated resident bytes.
    pub fn memory_bytes(&self) -> u64 {
        (self.stamp.len() * std::mem::size_of::<u32>()) as u64
    }
}

/// Thread-safe budget accountant, usable from the rayon-parallel pattern
/// matcher as well as the sequential BFS.
///
/// Every `add_*` / `check_*` returns `false` once a cap has tripped; the first
/// trip is recorded and reported through [`BudgetTracker::truncation`]. Callers
/// stop when they see `false` — they do **not** propagate an error.
pub struct BudgetTracker {
    budget: TraversalBudget,
    edges: AtomicU64,
    rows: AtomicU64,
    nodes: AtomicU64,
    peak_frontier: AtomicU64,
    hop: AtomicUsize,
    truncated: AtomicBool,
    trunc: Mutex<Option<Truncation>>,
    cancel: Option<CancelToken>,
    start: Instant,
}

impl BudgetTracker {
    pub fn new(budget: TraversalBudget, cancel: Option<CancelToken>) -> Self {
        Self {
            budget,
            edges: AtomicU64::new(0),
            rows: AtomicU64::new(0),
            nodes: AtomicU64::new(0),
            peak_frontier: AtomicU64::new(0),
            hop: AtomicUsize::new(0),
            truncated: AtomicBool::new(false),
            trunc: Mutex::new(None),
            cancel,
            start: Instant::now(),
        }
    }

    pub fn budget(&self) -> &TraversalBudget {
        &self.budget
    }

    #[inline]
    pub fn cancel(&self) {
        if let Some(c) = &self.cancel {
            c.store(true, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(false)
    }

    /// Record the hop currently being expanded (1-based) for the truncation
    /// report.
    #[inline]
    pub fn set_hop(&self, hop: usize) {
        self.hop.store(hop, Ordering::Relaxed);
    }

    #[inline]
    pub fn is_truncated(&self) -> bool {
        self.truncated.load(Ordering::Relaxed)
    }

    /// The recorded truncation, if a cap tripped.
    pub fn truncation(&self) -> Option<Truncation> {
        self.trunc.lock().ok().and_then(|g| g.clone())
    }

    /// Record the first cap that trips; later trips are ignored so the reported
    /// reason is the one that actually changed the outcome.
    fn trip(&self, reason: TruncReason, limit: u64, observed: u64) {
        if self.truncated.swap(true, Ordering::Relaxed) {
            return;
        }
        let mut g = match self.trunc.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if g.is_none() {
            *g = Some(Truncation {
                reason,
                hop: self.hop.load(Ordering::Relaxed),
                frontier_at_stop: self.peak_frontier.load(Ordering::Relaxed) as usize,
                limit,
                observed,
            });
        }
    }

    #[inline]
    fn ok(&self) -> bool {
        !self.is_truncated()
    }

    /// Check cancellation (hard error) and the deadline (soft truncation).
    ///
    /// `Ok(false)` means "stop, a cap tripped"; `Err(Cancelled)` means the
    /// caller abandoned the query and partial rows are discarded.
    #[inline]
    pub fn check_time(&self) -> Result<bool> {
        if self.is_cancelled() {
            return Err(GtvError::Cancelled);
        }
        if let Some(deadline) = self.budget.deadline {
            if Instant::now() >= deadline {
                self.trip(
                    TruncReason::Deadline,
                    0,
                    self.start.elapsed().as_millis() as u64,
                );
                return Ok(false);
            }
        }
        Ok(self.ok())
    }

    /// Account `n` scanned edges and enforce `max_edges`.
    #[inline]
    pub fn add_edges(&self, n: u64) -> bool {
        if !self.ok() {
            return false;
        }
        let total = self.edges.fetch_add(n, Ordering::Relaxed) + n;
        if total > self.budget.max_edges {
            self.trip(TruncReason::Edges, self.budget.max_edges, total);
            return false;
        }
        true
    }

    /// Account `n` emitted rows and enforce `max_rows`.
    #[inline]
    pub fn add_rows(&self, n: u64) -> bool {
        if !self.ok() {
            return false;
        }
        let total = self.rows.fetch_add(n, Ordering::Relaxed) + n;
        if total > self.budget.max_rows {
            self.trip(TruncReason::Rows, self.budget.max_rows, total);
            return false;
        }
        true
    }

    /// Account `n` newly visited nodes and enforce `max_nodes`.
    #[inline]
    pub fn add_nodes(&self, n: u64) -> bool {
        if !self.ok() {
            return false;
        }
        let total = self.nodes.fetch_add(n, Ordering::Relaxed) + n;
        if total > self.budget.max_nodes {
            self.trip(TruncReason::Nodes, self.budget.max_nodes, total);
            return false;
        }
        true
    }

    /// Enforce `max_frontier` and record the peak frontier size.
    #[inline]
    pub fn record_frontier(&self, n: usize) -> bool {
        self.peak_frontier.fetch_max(n as u64, Ordering::Relaxed);
        if n as u64 > self.budget.max_frontier as u64 {
            self.trip(
                TruncReason::Frontier,
                self.budget.max_frontier as u64,
                n as u64,
            );
            return false;
        }
        self.ok()
    }

    /// Enforce an estimated memory figure against `max_memory_bytes`.
    #[inline]
    pub fn check_memory(&self, bytes: u64) -> bool {
        if bytes > self.budget.max_memory_bytes {
            self.trip(TruncReason::Memory, self.budget.max_memory_bytes, bytes);
            return false;
        }
        self.ok()
    }

    pub fn stats(&self, hops: usize) -> TraversalStats {
        TraversalStats {
            hops,
            edges_scanned: self.edges.load(Ordering::Relaxed),
            rows: self.rows.load(Ordering::Relaxed),
            peak_frontier: self.peak_frontier.load(Ordering::Relaxed) as usize,
            nodes_visited: self.nodes.load(Ordering::Relaxed),
        }
    }

    /// Total scanned edges so far.
    pub fn edges_scanned(&self) -> u64 {
        self.edges.load(Ordering::Relaxed)
    }
}

/// Result of a bounded k-hop traversal.
///
/// `frontiers[i]` holds the nodes first reached at hop `i + 1` (sorted, unique).
#[derive(Debug, Clone)]
pub struct KhopResult {
    pub frontiers: Vec<arrow::array::UInt64Array>,
    pub stats: TraversalStats,
    /// `Some` when a cap stopped expansion before the graph was exhausted.
    pub truncation: Option<Truncation>,
}

impl KhopResult {
    pub fn is_complete(&self) -> bool {
        self.truncation.is_none()
    }

    /// Total nodes reached across every hop.
    pub fn reached(&self) -> usize {
        self.frontiers.iter().map(|f| f.len()).sum()
    }

    /// Three-state view: rows reached *and* whether the search finished.
    pub fn tri_state(&self) -> TriState {
        let non_empty = self.reached() > 0;
        match (non_empty, self.truncation.is_some()) {
            (_, true) => TriState::Incomplete,
            (true, false) => TriState::Found,
            (false, false) => TriState::NoMatch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visited_set_generation_reset() {
        let mut v = VisitedSet::new(4);
        assert!(v.mark(2));
        assert!(!v.mark(2));
        assert!(v.seen(2));
        v.reset();
        assert!(!v.seen(2));
        assert!(v.mark(2));
    }

    #[test]
    fn edges_cap_truncates_without_error() {
        let t = BudgetTracker::new(TraversalBudget::unlimited().with_max_edges(10), None);
        assert!(t.add_edges(8));
        assert!(!t.add_edges(5));
        let tr = t.truncation().expect("truncation recorded");
        assert_eq!(tr.reason, TruncReason::Edges);
        assert_eq!(tr.limit, 10);
        assert_eq!(tr.observed, 13);
        // Sticky: further accounting stays blocked and keeps the first reason.
        assert!(!t.add_edges(1));
        assert_eq!(t.truncation().unwrap().observed, 13);
    }

    #[test]
    fn cancel_token_trips() {
        let token: CancelToken = Arc::new(AtomicBool::new(false));
        let t = BudgetTracker::new(TraversalBudget::unlimited(), Some(token.clone()));
        assert_eq!(t.check_time().unwrap(), true);
        token.store(true, Ordering::Relaxed);
        assert!(matches!(t.check_time(), Err(GtvError::Cancelled)));
        // Cancellation is a hard error, not a truncation.
        assert!(!t.is_truncated());
    }

    #[test]
    fn deadline_is_a_truncation_not_an_error() {
        let past = Instant::now() - std::time::Duration::from_millis(1);
        let t = BudgetTracker::new(TraversalBudget::unlimited().with_deadline(past), None);
        assert_eq!(t.check_time().unwrap(), false);
        assert_eq!(t.truncation().unwrap().reason, TruncReason::Deadline);
    }

    #[test]
    fn frontier_memory_and_nodes_enforced() {
        let t = BudgetTracker::new(
            TraversalBudget::unlimited()
                .with_max_frontier(3)
                .with_max_memory_bytes(100),
            None,
        );
        assert!(t.record_frontier(3));
        assert!(!t.record_frontier(4));
        assert_eq!(t.truncation().unwrap().reason, TruncReason::Frontier);
        // Memory no longer trips because the first reason is sticky.
        assert!(!t.check_memory(200));
        assert_eq!(t.truncation().unwrap().reason, TruncReason::Frontier);

        let t2 = BudgetTracker::new(TraversalBudget::unlimited().with_max_nodes(5), None);
        assert!(t2.add_nodes(4));
        assert!(!t2.add_nodes(2));
        assert_eq!(t2.truncation().unwrap().reason, TruncReason::Nodes);
    }

    #[test]
    fn tri_state_distinguishes_empty_from_incomplete() {
        let complete: TraversalOutcome<Vec<u64>> = TraversalOutcome {
            rows: vec![1, 2],
            stats: TraversalStats::default(),
            truncation: None,
        };
        assert_eq!(complete.tri_state(true), TriState::Found);

        let none: TraversalOutcome<Vec<u64>> = TraversalOutcome {
            rows: vec![],
            stats: TraversalStats::default(),
            truncation: None,
        };
        assert_eq!(none.tri_state(false), TriState::NoMatch);

        let partial: TraversalOutcome<Vec<u64>> = TraversalOutcome {
            rows: vec![1],
            stats: TraversalStats::default(),
            truncation: Some(Truncation {
                reason: TruncReason::Edges,
                hop: 2,
                frontier_at_stop: 10,
                limit: 100,
                observed: 101,
            }),
        };
        assert_eq!(partial.tri_state(true), TriState::Incomplete);
        // The critical case: partial but empty must not read as "no match".
        assert_eq!(partial.tri_state(false), TriState::Incomplete);
        assert!(!partial.is_complete());
    }
}
