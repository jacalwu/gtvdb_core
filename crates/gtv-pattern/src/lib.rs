//! gtv-pattern: temporal graph pattern matching (GQL/Cypher-lite) over a
//! [`TemporalCSR`].
//!
//! A [`Pattern`] is a set of typed edges between *variables* plus temporal
//! ordering constraints between edges. Matching is a backtracking DFS that
//! binds each variable to a node, requires every matched edge to be active at
//! the reference time `valid_at`, and enforces the event-time (edge
//! `valid_from`) ordering constraints.

use gtv_core::{
    BudgetTracker, CancelToken, DedupKey, EdgePredicate, EdgeRef, Neighbor, PathEvidence, Result,
    Sel, TemporalCSR, TraversalBudget, TraversalStats, TriState, Truncation, UNBOUND_SNAPSHOT,
};

/// A pattern edge from variable `from` to variable `to`.
///
/// The matcher processes edges in declaration order and assumes each edge's
/// `from` variable is bound before the edge is reached (all built-in
/// constructors guarantee this "forward" ordering).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatternEdge {
    pub from: usize,
    pub to: usize,
    /// If `Some`, only edges of this type match.
    pub edge_type: Option<u16>,
}

/// Temporal ordering between two matched edges.
///
/// Defaults to strict ordering (`after - before >= 1`) so built-in patterns keep
/// their original semantics. Setting `min_gap = 0` allows equal event times;
/// `max_gap` bounds the delay. Both bounds are inclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventOrder {
    pub before: usize,
    pub after: usize,
    pub min_gap: i64,
    pub max_gap: Option<i64>,
}

impl EventOrder {
    /// Strict ordering, equivalent to `before < after`.
    pub fn strict(before: usize, after: usize) -> Self {
        Self {
            before,
            after,
            min_gap: 1,
            max_gap: None,
        }
    }

    /// Inclusive gap bounds. `min_gap > max_gap` is rejected at match time by
    /// treating the constraint as unsatisfiable rather than panicking.
    pub fn window(before: usize, after: usize, min_gap: i64, max_gap: Option<i64>) -> Self {
        Self {
            before,
            after,
            min_gap,
            max_gap,
        }
    }
}

/// A temporal graph pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    pub num_vars: usize,
    pub edges: Vec<PatternEdge>,
    pub event_orders: Vec<EventOrder>,
}

impl Pattern {
    /// A temporal path of `len` hops: `v0 -> v1 -> ... -> v_{len}`, with
    /// strictly increasing event times along the chain (`T1 < T2 < ...`).
    pub fn temporal_path(len: usize) -> Self {
        assert!(len >= 1, "path length must be >= 1");
        let edges: Vec<PatternEdge> = (0..len)
            .map(|i| PatternEdge {
                from: i,
                to: i + 1,
                edge_type: None,
            })
            .collect();
        let event_orders = (0..len.saturating_sub(1))
            .map(|i| EventOrder::strict(i, i + 1))
            .collect();
        Self {
            num_vars: len + 1,
            edges,
            event_orders,
        }
    }

    /// A diamond: `a -> b`, `a -> c`, `b -> d`, `c -> d`, with each arm ordered
    /// in time (`T(a->b) < T(b->d)` and `T(a->c) < T(c->d)`).
    pub fn diamond() -> Self {
        Self {
            num_vars: 4,
            edges: vec![
                PatternEdge {
                    from: 0,
                    to: 1,
                    edge_type: None,
                },
                PatternEdge {
                    from: 0,
                    to: 2,
                    edge_type: None,
                },
                PatternEdge {
                    from: 1,
                    to: 3,
                    edge_type: None,
                },
                PatternEdge {
                    from: 2,
                    to: 3,
                    edge_type: None,
                },
            ],
            event_orders: vec![EventOrder::strict(0, 2), EventOrder::strict(1, 3)],
        }
    }

    /// A temporal ring (cycle) of `len` hops back to the start:
    /// `v0 -> v1 -> ... -> v_{len-1} -> v0`, with increasing event times.
    pub fn ring(len: usize) -> Self {
        assert!(len >= 2, "ring length must be >= 2");
        let edges: Vec<PatternEdge> = (0..len)
            .map(|i| PatternEdge {
                from: i,
                to: (i + 1) % len,
                edge_type: None,
            })
            .collect();
        let event_orders = (0..len.saturating_sub(1))
            .map(|i| EventOrder::strict(i, i + 1))
            .collect();
        Self {
            num_vars: len,
            edges,
            event_orders,
        }
    }
}

/// A [`Pattern`] plus the AML motif constraints added by prod_p4 D3:
///
/// * **rolling window** — every matched edge's event time must fall inside a
///   window of at most `max_event_span` units
///   (`max(valid_from) - min(valid_from) <= span`);
/// * **sequence** — when `sequence` is non-empty it must have exactly one edge
///   type per pattern edge, and the matched edge types, ordered by event time
///   (ties broken by edge declaration order), must equal it.
///
/// This is a purely additive wrapper: [`Pattern`], [`find`], [`find_from`],
/// [`find_bounded`] and the built-in ring / path / diamond constructors are
/// unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowedMotif {
    pub pattern: Pattern,
    pub max_event_span: Option<i64>,
    pub sequence: Vec<u16>,
}

impl WindowedMotif {
    pub fn new(pattern: Pattern) -> Self {
        Self {
            pattern,
            max_event_span: None,
            sequence: Vec::new(),
        }
    }

    /// Require all matched edges to occur within `span` event-time units.
    pub fn with_window(mut self, span: i64) -> Self {
        self.max_event_span = Some(span);
        self
    }

    /// Require the matched edges, in event-time order, to follow this edge-type
    /// sequence (one entry per pattern edge).
    pub fn with_sequence(mut self, sequence: Vec<u16>) -> Self {
        self.sequence = sequence;
        self
    }

    /// True when a fully bound DFS state satisfies the motif constraints.
    fn accepts(&self, state: &DfsState) -> bool {
        if let Some(span) = self.max_event_span {
            let min = state.valid_from.iter().copied().min().unwrap_or(0);
            let max = state.valid_from.iter().copied().max().unwrap_or(0);
            if max - min > span {
                return false;
            }
        }
        if !self.sequence.is_empty() {
            if self.sequence.len() != state.edge_type.len() {
                return false;
            }
            let mut order: Vec<usize> = (0..state.valid_from.len()).collect();
            order.sort_by(|a, b| {
                state.valid_from[*a]
                    .cmp(&state.valid_from[*b])
                    .then(a.cmp(b))
            });
            if order
                .iter()
                .enumerate()
                .any(|(rank, &i)| state.edge_type[i] != self.sequence[rank])
            {
                return false;
            }
        }
        true
    }
}

/// One matched edge within a [`Match`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchedEdge {
    pub src: u64,
    pub dst: u64,
    pub edge_type: u16,
    pub valid_from: i64,
    pub valid_to: i64,
    /// Slot of this edge in the CSR's edge arrays — the handle back to the
    /// storage row (attributes, provenance, lineage).
    ///
    /// Without it the result only carries `(src, dst, type, time)`, so two
    /// equal transfers on the same day are indistinguishable and no row can be
    /// cited as evidence. `u32::MAX` means "not recoverable here" (only the
    /// pull direction of a k-hop loses it; pattern matching always has it).
    pub edge_idx: u32,
}

/// A complete pattern match: node bindings per variable plus the matched edges.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    /// `nodes[var]` is the bound node id for variable `var`.
    pub nodes: Vec<u64>,
    pub edges: Vec<MatchedEdge>,
}

impl Match {
    /// Convert this match into transportable evidence (A3).
    ///
    /// `segment_id` identifies the segment the CSR was built from: the CSR
    /// stores edge *slots*, and a slot only becomes a citable row once paired
    /// with the segment it belongs to. Callers that do not track segments can
    /// pass `0` and keep the slot as the row offset.
    ///
    /// Edges whose slot was not recoverable (`u32::MAX`, which cannot happen
    /// for pattern matching but is defended against anyway) are reported as an
    /// error rather than silently mapped to row 0 — a wrong row reference is
    /// worse than a refused one.
    pub fn evidence(&self, segment_id: u64) -> Result<PathEvidence> {
        let mut edges = Vec::with_capacity(self.edges.len());
        for e in &self.edges {
            let edge = EdgeRef::from_slot(segment_id, e.edge_idx).ok_or_else(|| {
                gtv_core::GtvError::InvalidArgument(format!(
                    "match edge {}->{} has no recoverable slot",
                    e.src, e.dst
                ))
            })?;
            edges.push(edge);
        }
        PathEvidence::new(
            self.nodes.clone(),
            edges,
            self.edges.iter().map(|e| e.valid_from).collect(),
            self.edges.iter().map(|e| e.valid_to).collect(),
        )
    }

    /// Build evidence using the CSR's physical provenance when available.
    ///
    /// For a single-segment CSR this agrees with [`Self::evidence`] (slot is
    /// the physical row). For a materialised graph whose slots were re-ordered
    /// by a k-way merge, `origin_of` is the only way to name the *original*
    /// row; without it, a slot would cite the wrong segment/row. An edge whose
    /// source has no stable address (unflushed WAL row, or a pull-direction
    /// synthetic edge) is refused rather than cited at a plausible-but-wrong
    /// row.
    pub fn evidence_from(&self, csr: &TemporalCSR) -> Result<PathEvidence> {
        let mut edges = Vec::with_capacity(self.edges.len());
        for e in &self.edges {
            let edge = csr.origin_of(e.edge_idx).ok_or_else(|| {
                gtv_core::GtvError::InvalidArgument(format!(
                    "match edge {}->{} (slot {}) has no physical origin",
                    e.src, e.dst, e.edge_idx
                ))
            })?;
            edges.push(edge);
        }
        PathEvidence::new(
            self.nodes.clone(),
            edges,
            self.edges.iter().map(|e| e.valid_from).collect(),
            self.edges.iter().map(|e| e.valid_to).collect(),
        )
    }

    /// Canonical dedup identity: `ring = true` for cycles (rotation-invariant),
    /// `false` for ordered paths (direction-sensitive).
    pub fn dedup_key(&self, ring: bool) -> DedupKey {
        if ring {
            DedupKey::ring(self.nodes.iter().copied())
        } else {
            DedupKey::path(self.nodes.iter().copied())
        }
    }
}

/// Find up to `limit` matches anchoring variable 0 at `start`, at time `valid_at`.
pub fn find_from(
    csr: &TemporalCSR,
    pattern: &Pattern,
    start: u64,
    valid_at: i64,
    limit: usize,
) -> Result<Vec<Match>> {
    if limit == 0 || pattern.num_vars == 0 {
        return Ok(Vec::new());
    }
    let mut state = DfsState::new(pattern);
    state.nodes[0] = start;
    let mut matches = Vec::new();
    dfs(
        csr,
        pattern,
        valid_at,
        0,
        &mut state,
        &mut matches,
        limit,
        None,
        None,
    )?;
    Ok(matches)
}

/// Find up to `limit` windowed-motif matches anchoring variable 0 at `start`.
pub fn find_windowed_from(
    csr: &TemporalCSR,
    motif: &WindowedMotif,
    start: u64,
    valid_at: i64,
    limit: usize,
) -> Result<Vec<Match>> {
    let pattern = &motif.pattern;
    if limit == 0 || pattern.num_vars == 0 {
        return Ok(Vec::new());
    }
    let mut state = DfsState::new(pattern);
    state.nodes[0] = start;
    let mut matches = Vec::new();
    dfs(
        csr,
        pattern,
        valid_at,
        0,
        &mut state,
        &mut matches,
        limit,
        None,
        Some(motif),
    )?;
    Ok(matches)
}

/// True when `pattern` is the 3-cycle `v0 → v1 → v2 → v0` (the wash-trade ring).
fn is_ring3(pattern: &Pattern) -> bool {
    pattern.num_vars == 3
        && pattern.edges.len() == 3
        && pattern.edges[0]
            == PatternEdge {
                from: 0,
                to: 1,
                edge_type: None,
            }
        && pattern.edges[1]
            == PatternEdge {
                from: 1,
                to: 2,
                edge_type: None,
            }
        && pattern.edges[2]
            == PatternEdge {
                from: 2,
                to: 0,
                edge_type: None,
            }
}

/// Recursion-free 3-cycle matcher (`A→B→C→A` with strictly increasing event
/// times).
///
/// The generic DFS pays a function call and a `Result` unwind per level; across
/// a 500k-node scan that overhead dominates. This runs three tight loops,
/// parallelizes the independent start-node scan with rayon, and — when every
/// edge is active at `valid_at` — drops the per-edge validity test entirely.
fn find_ring3(csr: &TemporalCSR, valid_at: i64, limit: usize) -> Result<Vec<Match>> {
    use rayon::prelude::*;

    let n = csr.node_count();
    if limit == 0 {
        return Ok(Vec::new());
    }
    let all_active = csr.all_active_at(valid_at);

    // Start nodes are scanned independently, so split them across the pool. Each
    // chunk collects its own matches; the results are concatenated and truncated
    // to `limit` (the ordering is unspecified, matching the generic `find`).
    const CHUNK: usize = 8192;
    let starts: Vec<u64> = (0..n as u64).collect();
    let parts: Result<Vec<Vec<Match>>> = starts
        .par_chunks(CHUNK)
        .map(|chunk| {
            let mut local = Vec::new();
            for &a in chunk {
                if local.len() >= limit {
                    break;
                }
                ring3_from(csr, a, valid_at, all_active, limit, &mut local)?;
            }
            Ok(local)
        })
        .collect();

    let mut out = Vec::new();
    for mut local in parts? {
        out.append(&mut local);
        if out.len() >= limit {
            out.truncate(limit);
            break;
        }
    }
    Ok(out)
}

/// Scan one start node `a` for a 3-cycle rooted there, pushing matches into `out`.
#[inline]
fn ring3_from(
    csr: &TemporalCSR,
    a: u64,
    valid_at: i64,
    all_active: bool,
    limit: usize,
    out: &mut Vec<Match>,
) -> Result<()> {
    let (base_a, da, vfa, vta, eta) = csr.edge_slices_with_start(a)?;
    for i in 0..da.len() {
        let b = da[i];
        let t0 = vfa[i];
        if b == a || (!all_active && (t0 > valid_at || valid_at >= vta[i])) {
            continue;
        }
        let (base_b, db, vfb, vtb, etb) = csr.edge_slices_with_start(b)?;
        for j in 0..db.len() {
            let c = db[j];
            if c == a || c == b {
                continue;
            }
            let t1 = vfb[j];
            if t1 <= t0 || (!all_active && (t1 > valid_at || valid_at >= vtb[j])) {
                continue;
            }
            let (base_c, dc, vfc, vtc, etc_) = csr.edge_slices_with_start(c)?;
            for k in 0..dc.len() {
                if dc[k] != a {
                    continue;
                }
                let t2 = vfc[k];
                if t2 <= t1 || (!all_active && (t2 > valid_at || valid_at >= vtc[k])) {
                    continue;
                }
                out.push(Match {
                    nodes: vec![a, b, c],
                    edges: vec![
                        MatchedEdge {
                            src: a,
                            dst: b,
                            edge_type: eta[i],
                            valid_from: t0,
                            valid_to: vta[i],
                            edge_idx: (base_a + i) as u32,
                        },
                        MatchedEdge {
                            src: b,
                            dst: c,
                            edge_type: etb[j],
                            valid_from: t1,
                            valid_to: vtb[j],
                            edge_idx: (base_b + j) as u32,
                        },
                        MatchedEdge {
                            src: c,
                            dst: a,
                            edge_type: etc_[k],
                            valid_from: t2,
                            valid_to: vtc[k],
                            edge_idx: (base_c + k) as u32,
                        },
                    ],
                });
                if out.len() >= limit {
                    return Ok(());
                }
            }
        }
    }
    Ok(())
}

/// Find up to `limit` matches over *all* possible start nodes, at time `valid_at`.
///
/// Allocation-free hot path: a single [`DfsState`] is reused across every start
/// node, and the 3-cycle (wash-trade) shape is dispatched to a dedicated
/// recursion-free matcher. [`find_from`] allocates a fresh state (three `Vec`s)
/// per start, which turns a 500k-node scan into ~1.5M heap allocations.
pub fn find(
    csr: &TemporalCSR,
    pattern: &Pattern,
    valid_at: i64,
    limit: usize,
) -> Result<Vec<Match>> {
    if limit == 0 || pattern.num_vars == 0 {
        return Ok(Vec::new());
    }
    if is_ring3(pattern) {
        return find_ring3(csr, valid_at, limit);
    }
    let mut out = Vec::new();
    let mut state = DfsState::new(pattern);
    for start in 0..csr.node_count() as u64 {
        if out.len() >= limit {
            break;
        }
        state.nodes[0] = start;
        dfs(
            csr, pattern, valid_at, 0, &mut state, &mut out, limit, None, None,
        )?;
    }
    Ok(out)
}

/// Find up to `limit` windowed-motif matches over all start nodes.
///
/// The optimized 3-cycle fast path is intentionally bypassed so the window /
/// sequence constraints are always enforced.
pub fn find_windowed(
    csr: &TemporalCSR,
    motif: &WindowedMotif,
    valid_at: i64,
    limit: usize,
) -> Result<Vec<Match>> {
    let pattern = &motif.pattern;
    if limit == 0 || pattern.num_vars == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut state = DfsState::new(pattern);
    for start in 0..csr.node_count() as u64 {
        if out.len() >= limit {
            break;
        }
        state.nodes[0] = start;
        dfs(
            csr,
            pattern,
            valid_at,
            0,
            &mut state,
            &mut out,
            limit,
            None,
            Some(motif),
        )?;
    }
    Ok(out)
}

/// A bounded pattern search: the matches that were found, what the search cost,
/// and whether it finished.
#[derive(Debug, Clone)]
pub struct MatchSet {
    pub matches: Vec<Match>,
    pub stats: TraversalStats,
    /// `Some` when a budget cap stopped the search early. The matches are then
    /// a *prefix*, not an answer: absence of a match proves nothing.
    pub truncation: Option<Truncation>,
}

impl MatchSet {
    pub fn is_complete(&self) -> bool {
        self.truncation.is_none()
    }

    pub fn tri_state(&self) -> TriState {
        match (self.matches.is_empty(), self.truncation.is_some()) {
            (_, true) => TriState::Incomplete,
            (false, false) => TriState::Found,
            (true, false) => TriState::NoMatch,
        }
    }
}

/// Resource-bounded variant of [`find`] (B1-3 / M1).
///
/// Enforces the traversal [`TraversalBudget`] (edges / nodes / rows / deadline)
/// and cooperatively honours a [`CancelToken`]. The optimized parallel 3-cycle
/// matcher is intentionally bypassed here so that every scanned edge is
/// accounted for; use [`find`] when no budget is needed.
///
/// `candidates` restricts which nodes may bind variable 0. It is validated
/// against the graph's declared snapshot, so a set built for one snapshot
/// cannot silently seed a query on another.
///
/// Exceeding the budget is **not** an error: the matches found so far come back
/// with [`MatchSet::truncation`] set, so a caller can distinguish "no pattern"
/// from "did not finish".
pub fn find_bounded(
    csr: &TemporalCSR,
    pattern: &Pattern,
    valid_at: i64,
    limit: usize,
    candidates: &Sel,
    budget: &TraversalBudget,
    cancel: Option<&CancelToken>,
) -> Result<MatchSet> {
    find_bounded_impl(
        csr, pattern, None, valid_at, limit, candidates, budget, cancel, None,
    )
}

/// Bounded search with window/sequence constraints and a push-down edge
/// predicate (for example a fixed-width attribute range).
///
/// This is the general form used by domain extensions that need all three at
/// once: a snapshot-bound candidate set, a resource budget that reports
/// truncation, and an attribute filter that must run before a match is emitted.
/// The plain [`find_bounded`] remains the zero-predicate, non-windowed case.
#[allow(clippy::too_many_arguments)]
pub fn find_bounded_motif(
    csr: &TemporalCSR,
    motif: &WindowedMotif,
    valid_at: i64,
    limit: usize,
    candidates: &Sel,
    budget: &TraversalBudget,
    cancel: Option<&CancelToken>,
    predicate: Option<&dyn EdgePredicate>,
) -> Result<MatchSet> {
    find_bounded_impl(
        csr,
        &motif.pattern,
        Some(motif),
        valid_at,
        limit,
        candidates,
        budget,
        cancel,
        predicate,
    )
}

#[allow(clippy::too_many_arguments)]
fn find_bounded_impl(
    csr: &TemporalCSR,
    pattern: &Pattern,
    motif: Option<&WindowedMotif>,
    valid_at: i64,
    limit: usize,
    candidates: &Sel,
    budget: &TraversalBudget,
    cancel: Option<&CancelToken>,
    predicate: Option<&dyn EdgePredicate>,
) -> Result<MatchSet> {
    if let Some(id) = csr.snapshot_id() {
        candidates.check_against(id)?;
    }
    let tracker = BudgetTracker::new(budget.clone(), cancel.cloned());
    let mut out = Vec::new();
    if limit == 0 || pattern.num_vars == 0 {
        return Ok(MatchSet {
            matches: out,
            stats: tracker.stats(0),
            truncation: None,
        });
    }
    // Ask for one more match than the caller wants.
    //
    // `limit` alone cannot say whether the answer was cut: filling it exactly is
    // also what an exhausted search does. Collecting the (limit+1)-th match and
    // throwing it away is the cheapest way to tell the two apart, and it stops
    // "this is a prefix" and "this is everything" from being the same value.
    let probe = limit.saturating_add(1);
    let mut state = DfsState::new(pattern);
    for start in candidates.iter() {
        if !tracker.check_time()? || out.len() >= probe {
            break;
        }
        tracker.set_hop(1);
        state.nodes[0] = start;
        let keep = dfs_inner(
            csr,
            pattern,
            valid_at,
            0,
            &mut state,
            &mut out,
            probe,
            Some(&tracker),
            motif,
            predicate,
        )?;
        if !keep {
            break;
        }
    }
    // Row accounting is done once, at the end: counting `out.len()` inside the
    // loop counted the same rows again on every start node.
    let cut_by_limit = out.len() >= probe;
    if cut_by_limit {
        out.truncate(limit);
    }
    tracker.add_rows(out.len() as u64);
    if cut_by_limit {
        tracker.trip_row_limit(limit, probe as u64);
    }
    let stats = tracker.stats(0);
    Ok(MatchSet {
        matches: out,
        stats,
        truncation: tracker.truncation(),
    })
}

/// [`find_bounded`] over every node, for callers with no candidate set.
pub fn find_bounded_all(
    csr: &TemporalCSR,
    pattern: &Pattern,
    valid_at: i64,
    limit: usize,
    budget: &TraversalBudget,
    cancel: Option<&CancelToken>,
) -> Result<MatchSet> {
    let all = Sel::all(UNBOUND_SNAPSHOT, csr.node_count() as u64);
    find_bounded(csr, pattern, valid_at, limit, &all, budget, cancel)
}

struct DfsState {
    nodes: Vec<u64>,
    valid_from: Vec<i64>,
    valid_to: Vec<i64>,
    edge_type: Vec<u16>,
    edge_idx: Vec<u32>,
}

impl DfsState {
    fn new(pattern: &Pattern) -> Self {
        Self {
            nodes: vec![u64::MAX; pattern.num_vars],
            valid_from: vec![0; pattern.edges.len()],
            valid_to: vec![0; pattern.edges.len()],
            edge_type: vec![0; pattern.edges.len()],
            edge_idx: vec![u32::MAX; pattern.edges.len()],
        }
    }

    fn to_match(&self, pattern: &Pattern) -> Match {
        let edges = pattern
            .edges
            .iter()
            .enumerate()
            .map(|(i, e)| MatchedEdge {
                src: self.nodes[e.from],
                dst: self.nodes[e.to],
                edge_type: self.edge_type[i],
                valid_from: self.valid_from[i],
                valid_to: self.valid_to[i],
                edge_idx: self.edge_idx[i],
            })
            .collect();
        Match {
            nodes: self.nodes.clone(),
            edges,
        }
    }

    fn event_orders_satisfied(&self, pattern: &Pattern) -> bool {
        pattern.event_orders.iter().all(|o| {
            let Some(delta) = self.valid_from[o.after].checked_sub(self.valid_from[o.before])
            else {
                return false;
            };
            if delta < o.min_gap {
                return false;
            }
            o.max_gap.is_none_or(|max| delta <= max)
        })
    }
}

/// Returns `Ok(false)` when a budget cap tripped and the walk was abandoned.
///
/// The partial `matches` stay in the caller's buffer: a truncated search is a
/// partial answer, not an error.
#[allow(clippy::too_many_arguments)]
fn dfs_inner(
    csr: &TemporalCSR,
    pattern: &Pattern,
    valid_at: i64,
    ei: usize,
    state: &mut DfsState,
    matches: &mut Vec<Match>,
    limit: usize,
    tracker: Option<&BudgetTracker>,
    motif: Option<&WindowedMotif>,
    predicate: Option<&dyn EdgePredicate>,
) -> Result<bool> {
    if matches.len() >= limit {
        return Ok(true);
    }
    if ei == pattern.edges.len() {
        if state.event_orders_satisfied(pattern) && motif.is_none_or(|m| m.accepts(state)) {
            matches.push(state.to_match(pattern));
        }
        return Ok(true);
    }

    let e = pattern.edges[ei];
    let from_node = state.nodes[e.from];
    if from_node == u64::MAX {
        return Ok(true);
    }

    let to_bound = state.nodes[e.to] != u64::MAX;

    // Direct slice access — skips the `neighbors` iterator's per-edge `Neighbor`
    // struct and closure overhead, which dominates a multi-million-node scan.
    let (base, dst, vf, vt, et) = csr.edge_slices_with_start(from_node)?;
    for idx in 0..dst.len() {
        if let Some(t) = tracker {
            // A tripped cap unwinds one frame at a time; `false` propagates all
            // the way up so the caller sees a truncated (not empty) result.
            if !t.add_edges(1) {
                return Ok(false);
            }
            if idx & 0x3ff == 0 && !t.check_time()? {
                return Ok(false);
            }
        }
        // Active at `valid_at`: `valid_from <= valid_at < valid_to`.
        let vf_i = vf[idx];
        if vf_i > valid_at || valid_at >= vt[idx] {
            continue;
        }
        if let Some(t) = e.edge_type {
            if et[idx] != t {
                continue;
            }
        }
        if let Some(pred) = predicate {
            let slot = (base + idx) as u32;
            let edge = Neighbor {
                dst: dst[idx],
                edge_type: et[idx],
                valid_from: vf_i,
                valid_to: vt[idx],
                edge_idx: slot,
            };
            if !pred.keep(&edge) {
                continue;
            }
        }
        let dst_i = dst[idx];
        if to_bound {
            // The destination is already fixed: only edges into it count.
            if dst_i != state.nodes[e.to] {
                continue;
            }
            state.valid_from[ei] = vf_i;
            state.valid_to[ei] = vt[idx];
            state.edge_type[ei] = et[idx];
            state.edge_idx[ei] = (base + idx) as u32;
            let keep = dfs_inner(
                csr,
                pattern,
                valid_at,
                ei + 1,
                state,
                matches,
                limit,
                tracker,
                motif,
                predicate,
            )?;
            if !keep {
                return Ok(false);
            }
        } else {
            // Bind a fresh variable: keep node assignments distinct.
            let mut dup = false;
            for k in 0..pattern.num_vars {
                if state.nodes[k] == dst_i {
                    dup = true;
                    break;
                }
            }
            if dup {
                continue;
            }
            state.nodes[e.to] = dst_i;
            state.valid_from[ei] = vf_i;
            state.valid_to[ei] = vt[idx];
            state.edge_type[ei] = et[idx];
            state.edge_idx[ei] = (base + idx) as u32;
            let keep = dfs_inner(
                csr,
                pattern,
                valid_at,
                ei + 1,
                state,
                matches,
                limit,
                tracker,
                motif,
                predicate,
            )?;
            state.nodes[e.to] = u64::MAX;
            if !keep {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn dfs(
    csr: &TemporalCSR,
    pattern: &Pattern,
    valid_at: i64,
    ei: usize,
    state: &mut DfsState,
    matches: &mut Vec<Match>,
    limit: usize,
    tracker: Option<&BudgetTracker>,
    motif: Option<&WindowedMotif>,
) -> Result<bool> {
    dfs_inner(
        csr, pattern, valid_at, ei, state, matches, limit, tracker, motif, None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use gtv_core::{EdgeTable, NodeTable, TemporalGraph, TruncReason};
    use std::sync::Arc;

    /// Transfer graph with distinct event times:
    ///   0 -> 1 @10, 1 -> 2 @20, 2 -> 3 @30, 3 -> 0 @40  (a temporal ring)
    ///   0 -> 4 @15, 4 -> 3 @25 (a diamond with the ring's left half)
    fn transfer_graph() -> TemporalGraph {
        let nodes = NodeTable::new(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("id", DataType::UInt64, false),
                    Field::new("value", DataType::Float64, false),
                ])),
                vec![
                    Arc::new(UInt64Array::from(vec![0u64, 1, 2, 3, 4])) as _,
                    Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])) as _,
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let edges = EdgeTable::from_vecs(
            vec![0, 1, 2, 3, 0, 4],
            vec![1, 2, 3, 0, 4, 3],
            vec![1u16, 1, 1, 1, 1, 1],
            vec![10, 20, 30, 40, 15, 25],
            vec![1000, 1000, 1000, 1000, 1000, 1000],
        )
        .unwrap();
        TemporalGraph::new(nodes, edges).unwrap()
    }

    #[test]
    fn temporal_path_finds_increasing_chain() {
        let g = transfer_graph();
        let pat = Pattern::temporal_path(3);
        let m = find_from(g.csr(), &pat, 0, 500, 10).unwrap();
        // Only 0 -> 1 -> 2 -> 3 has strictly increasing times (10 < 20 < 30).
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].nodes, vec![0, 1, 2, 3]);
    }

    #[test]
    fn ring_finds_cycle_back_to_start() {
        let g = transfer_graph();
        let pat = Pattern::ring(4);
        let m = find_from(g.csr(), &pat, 0, 500, 10).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].nodes, vec![0, 1, 2, 3]);
        assert_eq!(m[0].edges[3].dst, 0);
    }

    #[test]
    fn diamond_finds_two_arm_structure() {
        let g = transfer_graph();
        let pat = Pattern::diamond();
        let m = find_from(g.csr(), &pat, 0, 500, 10).unwrap();
        // Diamond: 0->1 (10), 0->4 (15), 1->3 (20 via? no 1->3 is not an edge).
        // Our graph has 0->1,0->4 and 4->3, but no 1->3, so no diamond from 0.
        assert!(m.is_empty());
    }

    #[test]
    fn event_order_prunes_decreasing_times() {
        // Build a chain whose times DECREASE: 0->1 @50, 1->2 @40.
        let g = {
            let nodes = NodeTable::new(
                RecordBatch::try_new(
                    Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)])),
                    vec![Arc::new(UInt64Array::from(vec![0u64, 1, 2])) as _],
                )
                .unwrap(),
            )
            .unwrap();
            let edges = EdgeTable::from_vecs(
                vec![0, 1],
                vec![1, 2],
                vec![1u16, 1],
                vec![50, 40],
                vec![2000, 2000],
            )
            .unwrap();
            TemporalGraph::new(nodes, edges).unwrap()
        };
        let pat = Pattern::temporal_path(2);
        // Edges are active at T=1500, but their event times DECREASE (50 > 40),
        // so the temporal path (which requires increasing times) must not match.
        let m = find_from(g.csr(), &pat, 0, 1500, 10).unwrap();
        assert!(m.is_empty());
    }

    /// Two transfers with the same counterparty, type and interval are
    /// indistinguishable through `(src, dst, type, time)`. `edge_idx` is the
    /// only thing that separates them, and it must resolve back to the CSR.
    #[test]
    fn parallel_edges_are_distinguishable_by_edge_idx() {
        let nodes = NodeTable::new(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)])),
                vec![Arc::new(UInt64Array::from(vec![0u64, 1])) as _],
            )
            .unwrap(),
        )
        .unwrap();
        let edges = EdgeTable::from_vecs(
            vec![0, 0],
            vec![1, 1],
            vec![5u16, 5],
            vec![10, 10],
            vec![1000, 1000],
        )
        .unwrap();
        let g = TemporalGraph::new(nodes, edges).unwrap();

        let m = find_from(g.csr(), &Pattern::temporal_path(1), 0, 500, 10).unwrap();
        assert_eq!(m.len(), 2, "both parallel edges must be reported");

        let slots: Vec<u32> = m.iter().map(|x| x.edges[0].edge_idx).collect();
        assert_eq!(slots, vec![0, 1], "each match must carry its own slot");
        // And the slot still resolves to the edge the match describes.
        let (dst, vf, _vt, _et) = g.csr().edge_slices(0).unwrap();
        for (i, edge) in m.iter().map(|x| x.edges[0]).enumerate() {
            let slot = edge.edge_idx;
            assert_eq!(dst[slot as usize], edge.dst);
            assert_eq!(vf[slot as usize], edge.valid_from);
            assert_eq!(slot, slots[i]);
        }
    }

    #[test]
    fn find_bounded_matches_find_and_reports_truncation() {
        let g = transfer_graph();
        let pat = Pattern::temporal_path(3);
        let all = Sel::all(UNBOUND_SNAPSHOT, g.csr().node_count() as u64);
        let unbounded = find(g.csr(), &pat, 500, 10).unwrap();
        let bounded = find_bounded(
            g.csr(),
            &pat,
            500,
            10,
            &all,
            &TraversalBudget::unlimited(),
            None,
        )
        .unwrap();
        assert_eq!(unbounded, bounded.matches);
        assert!(bounded.is_complete());
        assert_eq!(bounded.tri_state(), TriState::Found);

        // An edge cap of 1 trips before the chain is found. The old contract
        // returned `Err` and threw the partial result away; the search must now
        // say "did not finish" so the caller cannot report it as "no match".
        let t = find_bounded(
            g.csr(),
            &pat,
            500,
            10,
            &all,
            &TraversalBudget::unlimited().with_max_edges(1),
            None,
        )
        .unwrap();
        let tr = t
            .truncation
            .as_ref()
            .expect("edge cap must be reported")
            .clone();
        assert_eq!(tr.reason, TruncReason::Edges);
        assert_eq!(t.tri_state(), TriState::Incomplete);
    }

    #[test]
    fn find_bounded_honours_cancel() {
        let g = transfer_graph();
        let pat = Pattern::temporal_path(3);
        let all = Sel::all(UNBOUND_SNAPSHOT, g.csr().node_count() as u64);
        let token: CancelToken = Arc::new(std::sync::atomic::AtomicBool::new(true));
        // Cancellation is a hard stop (the caller abandoned the query), unlike
        // a budget cap which returns partial rows.
        let got = find_bounded(
            g.csr(),
            &pat,
            500,
            10,
            &all,
            &TraversalBudget::unlimited(),
            Some(&token),
        );
        assert!(matches!(got, Err(gtv_core::GtvError::Cancelled)));
    }

    /// Two chains with distinct edge types:
    ///   0 ->(7) 1 ->(8) 2 ->(9) 3   @10, 20, 30
    ///   0 ->(9) 4 ->(8) 5 ->(7) 6   @10, 20, 30
    fn sequenced_graph() -> TemporalGraph {
        let nodes = NodeTable::new(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)])),
                vec![Arc::new(UInt64Array::from(vec![0u64, 1, 2, 3, 4, 5, 6])) as _],
            )
            .unwrap(),
        )
        .unwrap();
        let edges = EdgeTable::from_vecs(
            vec![0, 1, 2, 0, 4, 5],
            vec![1, 2, 3, 4, 5, 6],
            vec![7u16, 8, 9, 9, 8, 7],
            vec![10, 20, 30, 10, 20, 30],
            vec![1000, 1000, 1000, 1000, 1000, 1000],
        )
        .unwrap();
        TemporalGraph::new(nodes, edges).unwrap()
    }

    #[test]
    fn sequence_constraint_selects_matching_order() {
        let g = sequenced_graph();
        let base = Pattern::temporal_path(3);

        // Both chains match without a sequence constraint.
        assert_eq!(find_from(g.csr(), &base, 0, 500, 10).unwrap().len(), 2);

        let motif = WindowedMotif::new(base.clone()).with_sequence(vec![7, 8, 9]);
        let m = find_windowed_from(g.csr(), &motif, 0, 500, 10).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].nodes, vec![0, 1, 2, 3]);

        let reversed = WindowedMotif::new(base).with_sequence(vec![9, 8, 7]);
        let m = find_windowed_from(g.csr(), &reversed, 0, 500, 10).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].nodes, vec![0, 4, 5, 6]);
    }

    #[test]
    fn rolling_window_filters_by_event_span() {
        let g = sequenced_graph();
        let base = Pattern::temporal_path(3);
        // span = 30 - 10 = 20
        let wide = WindowedMotif::new(base.clone()).with_window(20);
        assert_eq!(
            find_windowed_from(g.csr(), &wide, 0, 500, 10)
                .unwrap()
                .len(),
            2
        );
        let tight = WindowedMotif::new(base).with_window(15);
        assert!(find_windowed_from(g.csr(), &tight, 0, 500, 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn windowed_motif_combines_window_and_sequence() {
        let g = sequenced_graph();
        let motif = WindowedMotif::new(Pattern::temporal_path(3))
            .with_window(20)
            .with_sequence(vec![7, 8, 9]);
        let m = find_windowed(g.csr(), &motif, 500, 10).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].nodes, vec![0, 1, 2, 3]);
    }

    #[test]
    fn windowed_without_constraints_matches_plain_find() {
        let g = sequenced_graph();
        let plain = find_from(g.csr(), &Pattern::temporal_path(3), 0, 500, 10).unwrap();
        let motif = WindowedMotif::new(Pattern::temporal_path(3));
        let windowed = find_windowed_from(g.csr(), &motif, 0, 500, 10).unwrap();
        assert_eq!(plain, windowed);
    }

    #[test]
    fn wrong_sequence_length_matches_nothing() {
        let g = sequenced_graph();
        let motif = WindowedMotif::new(Pattern::temporal_path(3)).with_sequence(vec![7, 8]);
        assert!(find_windowed_from(g.csr(), &motif, 0, 500, 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn event_order_gap_bounds_are_inclusive() {
        let nodes = NodeTable::new(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)])),
                vec![Arc::new(UInt64Array::from(vec![0u64, 1, 2])) as _],
            )
            .unwrap(),
        )
        .unwrap();
        let edges = EdgeTable::from_vecs(
            vec![0, 1],
            vec![1, 2],
            vec![1u16, 1],
            vec![10, 70],
            vec![1000, 1000],
        )
        .unwrap();
        let g = TemporalGraph::new(nodes, edges).unwrap();

        let mut pattern = Pattern::temporal_path(2);
        pattern.event_orders = vec![EventOrder::window(0, 1, 50, Some(65))];
        assert_eq!(find_from(g.csr(), &pattern, 0, 500, 10).unwrap().len(), 1);

        pattern.event_orders = vec![EventOrder::window(0, 1, 50, Some(55))];
        assert!(find_from(g.csr(), &pattern, 0, 500, 10).unwrap().is_empty());

        pattern.event_orders = vec![EventOrder::window(0, 1, 61, None)];
        assert!(find_from(g.csr(), &pattern, 0, 500, 10).unwrap().is_empty());

        let mut equal = Pattern::temporal_path(2);
        equal.event_orders = vec![EventOrder::window(0, 1, 0, None)];
        assert_eq!(find_from(g.csr(), &equal, 0, 500, 10).unwrap().len(), 1);
    }

    #[test]
    fn matched_edge_idx_is_the_global_csr_slot() {
        let nodes = NodeTable::new(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)])),
                vec![Arc::new(UInt64Array::from(vec![0u64, 1, 2])) as _],
            )
            .unwrap(),
        )
        .unwrap();
        let edges = EdgeTable::from_vecs(
            vec![0, 1],
            vec![1, 2],
            vec![1u16, 1],
            vec![10, 20],
            vec![1000, 1000],
        )
        .unwrap();
        let g = TemporalGraph::new(nodes, edges).unwrap();

        let matches = find_from(g.csr(), &Pattern::temporal_path(1), 1, 500, 10).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].edges[0].dst, 2);
        assert_eq!(matches[0].edges[0].edge_idx, 1);
    }

    #[test]
    fn bounded_motif_pushes_attribute_predicate_into_matching() {
        use gtv_core::{AttrPred, EdgeAttrs};

        let nodes = NodeTable::new(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)])),
                vec![Arc::new(UInt64Array::from(vec![0u64, 1])) as _],
            )
            .unwrap(),
        )
        .unwrap();
        let edges = EdgeTable::from_vecs(
            vec![0, 0],
            vec![1, 1],
            vec![1u16, 1],
            vec![10, 20],
            vec![1000, 1000],
        )
        .unwrap();
        let g = TemporalGraph::new(nodes, edges).unwrap();

        let attrs = Arc::new(EdgeAttrs::from_vecs(&["amount"], vec![vec![5.0, 50.0]]).unwrap());
        let pred = AttrPred::gt(attrs, 0, 10.0).unwrap();
        let motif = WindowedMotif::new(Pattern::temporal_path(1));
        let candidates = Sel::all(UNBOUND_SNAPSHOT, g.csr().node_count() as u64);
        let set = find_bounded_motif(
            g.csr(),
            &motif,
            500,
            10,
            &candidates,
            &TraversalBudget::unlimited(),
            None,
            Some(&pred),
        )
        .unwrap();

        assert_eq!(set.matches.len(), 1);
        assert_eq!(set.matches[0].edges[0].edge_idx, 1);
        assert!(set.truncation.is_none());
    }

    #[test]
    fn a_limit_cut_is_reported_and_an_exact_fit_is_not() {
        // Three independent edges, so a one-hop path matches exactly three times.
        let csr = TemporalCSR::from_edge_columns(
            &[0, 1, 2],
            &[1, 2, 3],
            &[0, 0, 0],
            &[10, 10, 10],
            &[0, 0, 0],
            4,
        )
        .unwrap();
        let pattern = Pattern::temporal_path(1);
        let all = Sel::from_ids(UNBOUND_SNAPSHOT, 4, [0u64, 1, 2]).unwrap();
        let budget = TraversalBudget::unlimited();

        // More matches exist than were asked for: the answer is a prefix, and
        // says so instead of reading as the whole result.
        let cut = find_bounded(&csr, &pattern, 0, 2, &all, &budget, None).unwrap();
        assert_eq!(cut.matches.len(), 2);
        let t = cut
            .truncation
            .as_ref()
            .expect("a limit-cut answer must be reported");
        assert!(matches!(t.reason, TruncReason::Rows));
        assert_eq!(t.limit, 2);
        assert_eq!(t.observed, 3);
        assert!(!cut.is_complete());

        // Exactly as many matches as asked for: nothing was cut, and claiming
        // otherwise would make every full page look incomplete.
        let exact = find_bounded(&csr, &pattern, 0, 3, &all, &budget, None).unwrap();
        assert_eq!(exact.matches.len(), 3);
        assert!(exact.is_complete(), "an exact fit is not a truncation");
    }
}
