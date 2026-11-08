//! Temporal provenance DAG: the generic "how did this state come to be" kernel.
//!
//! A provenance graph is a forest of states where every non-root state has
//! exactly **one** incoming transition (`parent -> child`) carrying a branch
//! weight in `(0, 1]`. Nodes carry the logical time at which the state came
//! into existence; edges carry no timestamp of their own — a transition is
//! dated by the birth of its child, which removes the possibility of an edge
//! and its endpoint disagreeing about when the transition happened.
//!
//! This is the same shape as the temporal edge model in [`crate::temporal`],
//! specialised to the case where the interesting question is not "who is
//! adjacent at `T`" but "which chain of events produced this state, and how
//! likely was that chain".
//!
//! Three invariants are enforced at construction time, because every later
//! query assumes them:
//!
//! 1. **Unique ids.** A `state_id` identifies exactly one node.
//! 2. **Forest, not DAG-with-joins.** At most one parent per state. A merge
//!    (two parents) is rejected rather than silently resolved, because
//!    "which parent explains this state" has no correct default answer.
//! 3. **Monotone time.** `parent.at <= child.at`. Equal timestamps are allowed
//!    (several gates in one logical step); going backwards is not, because it
//!    would let a reconstruction walk into the future.
//!
//! Cycle detection is folded into the same pass as the time check: the graph is
//! topologically ordered from the roots with Kahn's algorithm, so a surviving
//! unvisited node means a cycle — and the traversal also yields the depth of
//! every state for free.
//!
//! ```rust
//! use gtv_core::provenance::{ProvenanceEdge, ProvenanceGraph, ProvenanceNode};
//!
//! let nodes = vec![
//!     ProvenanceNode::new(0, 0, "init"),
//!     ProvenanceNode::new(1, 1, "h"),
//!     ProvenanceNode::new(2, 2, "measure"),
//! ];
//! let edges = vec![
//!     ProvenanceEdge::new(0, 1, 1.0),
//!     ProvenanceEdge::new(1, 2, 0.25),
//! ];
//! let g = ProvenanceGraph::new(nodes, edges)?;
//! assert_eq!(g.ancestry(2)?, vec![0, 1, 2]);
//! assert!((g.probability(2)? - 0.25).abs() < 1e-12);
//! # Ok::<(), gtv_core::provenance::ProvenanceError>(())
//! ```

use std::collections::{HashMap, VecDeque};

/// A state (node) in the provenance forest.
///
/// `at` is the logical time the state came into existence — a gate index, a
/// nanosecond timestamp, a version number. `label` is free-form and is what a
/// human reads back during reconstruction (`"h q[0]"`, `"aml_flag_set"`, …).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceNode {
    pub id: u64,
    pub at: i64,
    pub label: String,
}

impl ProvenanceNode {
    pub fn new(id: u64, at: i64, label: impl Into<String>) -> Self {
        Self {
            id,
            at,
            label: label.into(),
        }
    }
}

/// A directed `parent -> child` transition with a branch weight in `(0, 1]`.
///
/// The weight is the *conditional* probability of taking this branch given that
/// the parent was reached, so the probability of any state is the product along
/// its root path (see [`ProvenanceGraph::probability`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProvenanceEdge {
    pub parent: u64,
    pub child: u64,
    pub weight: f64,
}

impl ProvenanceEdge {
    pub fn new(parent: u64, child: u64, weight: f64) -> Self {
        Self {
            parent,
            child,
            weight,
        }
    }
}

/// Construction and query failures.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ProvenanceError {
    #[error("duplicate state id {0}")]
    DuplicateNode(u64),
    #[error("transition references unknown state {0}")]
    UnknownState(u64),
    #[error("state {child} has more than one parent ({first} and {second}); provenance must be a forest")]
    MultipleParents { child: u64, first: u64, second: u64 },
    #[error("transition weight for state {child} must lie in (0, 1], got {weight}")]
    WeightOutOfRange { child: u64, weight: f64 },
    #[error("parent {parent} (t={parent_at}) is newer than child {child} (t={child_at})")]
    TimeInversion {
        parent: u64,
        child: u64,
        parent_at: i64,
        child_at: i64,
    },
    #[error("cycle detected: state {0} is reachable but never rooted")]
    Cycle(u64),
    #[error("unknown state {0}")]
    UnknownQuery(u64),
}

/// One state of a time slice, with the probability of having reached it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateProbability {
    pub id: u64,
    pub probability: f64,
    pub depth: usize,
}

/// A pruned time slice plus the probability mass it retains.
///
/// `explained` is the mass *before* renormalisation. Reporting it separately is
/// what stops pruning from becoming a silent lie: a branch set covering 0.02 of
/// the mass means the remaining branches are curiosities, not the answer.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchSet {
    pub branches: Vec<StateProbability>,
    pub explained: f64,
}

impl BranchSet {
    /// The retained branches renormalised to sum to 1.
    ///
    /// Only meaningful when `explained > 0`; returns an empty vector otherwise.
    pub fn renormalized(&self) -> Vec<(u64, f64)> {
        if self.explained <= 0.0 {
            return Vec::new();
        }
        self.branches
            .iter()
            .map(|b| (b.id, b.probability / self.explained))
            .collect()
    }
}

/// Shape summary of a provenance forest.
#[derive(Debug, Clone, PartialEq)]
pub struct ProvenanceStats {
    pub nodes: usize,
    pub edges: usize,
    pub roots: usize,
    pub leaves: usize,
    pub max_depth: usize,
    /// Mean number of children over non-leaf states; 0 when there are none.
    pub mean_branching: f64,
}

/// An immutable provenance forest with O(1) parent/child lookups.
#[derive(Debug, Clone)]
pub struct ProvenanceGraph {
    nodes: Vec<ProvenanceNode>,
    edges: Vec<ProvenanceEdge>,
    index: HashMap<u64, usize>,
    children: Vec<Vec<u64>>,
    /// Edge index of the single incoming transition, `None` for roots.
    parent_edge: Vec<Option<usize>>,
    depth: Vec<usize>,
}

impl ProvenanceGraph {
    /// Build a forest, validating every invariant documented on the module.
    ///
    /// Edges may arrive in any order; nodes may too. The cost is O(nodes+edges)
    /// time and memory.
    pub fn new(
        nodes: Vec<ProvenanceNode>,
        edges: Vec<ProvenanceEdge>,
    ) -> Result<Self, ProvenanceError> {
        let n = nodes.len();
        let mut index = HashMap::with_capacity(n);
        for (i, node) in nodes.iter().enumerate() {
            if index.insert(node.id, i).is_some() {
                return Err(ProvenanceError::DuplicateNode(node.id));
            }
        }

        let mut children: Vec<Vec<u64>> = vec![Vec::new(); n];
        let mut parent_edge: Vec<Option<usize>> = vec![None; n];

        for (ei, edge) in edges.iter().enumerate() {
            // `!(w > 0 && w <= 1)` also rejects NaN, which a plain range check
            // would let through.
            if !(edge.weight > 0.0 && edge.weight <= 1.0) {
                return Err(ProvenanceError::WeightOutOfRange {
                    child: edge.child,
                    weight: edge.weight,
                });
            }
            let p = *index
                .get(&edge.parent)
                .ok_or(ProvenanceError::UnknownState(edge.parent))?;
            let c = *index
                .get(&edge.child)
                .ok_or(ProvenanceError::UnknownState(edge.child))?;
            if let Some(prev) = parent_edge[c] {
                return Err(ProvenanceError::MultipleParents {
                    child: edge.child,
                    first: edges[prev].parent,
                    second: edge.parent,
                });
            }
            parent_edge[c] = Some(ei);
            children[p].push(edge.child);
        }

        // Kahn from the roots: verifies acyclicity, and computes depth on the
        // way. A node that is never dequeued sits on a cycle.
        let mut in_degree: Vec<usize> = parent_edge.iter().map(|e| usize::from(e.is_some())).collect();
        let mut depth = vec![0usize; n];
        let mut queue: VecDeque<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, &d)| d == 0)
            .map(|(i, _)| i)
            .collect();
        let mut visited = 0usize;

        while let Some(u) = queue.pop_front() {
            visited += 1;
            for &child in &children[u] {
                let ci = index[&child];
                if nodes[ci].at < nodes[u].at {
                    return Err(ProvenanceError::TimeInversion {
                        parent: nodes[u].id,
                        child,
                        parent_at: nodes[u].at,
                        child_at: nodes[ci].at,
                    });
                }
                depth[ci] = depth[u] + 1;
                in_degree[ci] -= 1;
                if in_degree[ci] == 0 {
                    queue.push_back(ci);
                }
            }
        }

        if visited != n {
            let stuck = (0..n)
                .find(|&i| in_degree[i] > 0)
                .map(|i| nodes[i].id)
                .unwrap_or_default();
            return Err(ProvenanceError::Cycle(stuck));
        }

        Ok(Self {
            nodes,
            edges,
            index,
            children,
            parent_edge,
            depth,
        })
    }

    fn pos(&self, id: u64) -> Result<usize, ProvenanceError> {
        self.index
            .get(&id)
            .copied()
            .ok_or(ProvenanceError::UnknownQuery(id))
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn nodes(&self) -> &[ProvenanceNode] {
        &self.nodes
    }

    pub fn node(&self, id: u64) -> Result<&ProvenanceNode, ProvenanceError> {
        self.pos(id).map(|i| &self.nodes[i])
    }

    /// States that came into existence at or before `at`, root-first order.
    pub fn as_of(&self, at: i64) -> Vec<u64> {
        self.nodes
            .iter()
            .filter(|n| n.at <= at)
            .map(|n| n.id)
            .collect()
    }

    /// Children of `id` in insertion order.
    pub fn children(&self, id: u64) -> Result<&[u64], ProvenanceError> {
        self.pos(id).map(|i| self.children[i].as_slice())
    }

    /// Incoming transitions, most recent first is *not* guaranteed — order is
    /// the order edges were supplied.
    pub fn incoming(&self, id: u64) -> Result<&[ProvenanceEdge], ProvenanceError> {
        let i = self.pos(id)?;
        Ok(match self.parent_edge[i] {
            Some(ei) => std::slice::from_ref(&self.edges[ei]),
            None => &[],
        })
    }

    /// Root-first chain of state ids ending at `id`.
    pub fn ancestry(&self, id: u64) -> Result<Vec<u64>, ProvenanceError> {
        let mut chain = vec![id];
        let mut i = self.pos(id)?;
        while let Some(ei) = self.parent_edge[i] {
            let parent = self.edges[ei].parent;
            chain.push(parent);
            i = self.pos(parent)?;
        }
        chain.reverse();
        Ok(chain)
    }

    /// Root-first transitions along the path to `id`.
    pub fn path(&self, id: u64) -> Result<Vec<ProvenanceEdge>, ProvenanceError> {
        let mut edges = Vec::new();
        let mut i = self.pos(id)?;
        while let Some(ei) = self.parent_edge[i] {
            edges.push(self.edges[ei]);
            i = self.pos(self.edges[ei].parent)?;
        }
        edges.reverse();
        Ok(edges)
    }

    /// Number of transitions from the root: 0 for a root.
    pub fn depth(&self, id: u64) -> Result<usize, ProvenanceError> {
        self.pos(id).map(|i| self.depth[i])
    }

    /// Product of the branch weights along the root path; 1.0 for a root.
    ///
    /// Underflow is deliberate: at ~1e-320 the branch is not worth reporting,
    /// and clamping to zero there is honest.
    pub fn probability(&self, id: u64) -> Result<f64, ProvenanceError> {
        let mut p = 1.0f64;
        for edge in self.path(id)? {
            p *= edge.weight;
        }
        Ok(p)
    }

    /// The time slice at `at`: states that exist by `at` and have no child that
    /// also exists by `at`, i.e. the leading edge of the frontier.
    ///
    /// Sorted by probability descending, then id ascending, so the dominant
    /// explanation is the first element.
    pub fn frontier(&self, at: i64) -> Vec<StateProbability> {
        let mut out = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if node.at > at {
                continue;
            }
            let extended = self.children[i]
                .iter()
                .any(|c| self.nodes[self.index[c]].at <= at);
            if extended {
                continue;
            }
            out.push(StateProbability {
                id: node.id,
                probability: self.probability(node.id).unwrap_or(0.0),
                depth: self.depth[i],
            });
        }
        sort_by_probability(&mut out);
        out
    }

    /// [`Self::frontier`] restricted to branches at or above `min_prob`.
    ///
    /// `explained` is the retained mass, so a caller can decide whether the cut
    /// is acceptable before trusting the renormalised view.
    pub fn frontier_pruned(&self, at: i64, min_prob: f64) -> BranchSet {
        let branches: Vec<StateProbability> = self
            .frontier(at)
            .into_iter()
            .filter(|b| b.probability >= min_prob)
            .collect();
        let explained = branches.iter().map(|b| b.probability).sum();
        BranchSet {
            branches,
            explained,
        }
    }

    /// Probability mass per logical time step.
    ///
    /// For a properly normalised per-step branching (quantum measurement, Monte
    /// Carlo paths) this is <= 1 at every step; a value above 1 means sibling
    /// weights were recorded as absolute probabilities while the edges were
    /// interpreted as conditional ones.
    pub fn mass_by_step(&self) -> Vec<(i64, f64)> {
        let mut totals: HashMap<i64, f64> = HashMap::new();
        for node in &self.nodes {
            if self.parent_edge[self.index[&node.id]].is_none() {
                continue; // roots carry the whole mass; not a branch
            }
            *totals.entry(node.at).or_insert(0.0) += self.edges
                [self.parent_edge[self.index[&node.id]].expect("checked above")]
            .weight;
        }
        let mut out: Vec<(i64, f64)> = totals.into_iter().collect();
        out.sort_by_key(|(t, _)| *t);
        out
    }

    pub fn stats(&self) -> ProvenanceStats {
        let roots = self.parent_edge.iter().filter(|e| e.is_none()).count();
        let leaves = self.children.iter().filter(|c| c.is_empty()).count();
        let internal = self.nodes.len() - leaves;
        let mean_branching = if internal == 0 {
            0.0
        } else {
            self.edges.len() as f64 / internal as f64
        };
        ProvenanceStats {
            nodes: self.nodes.len(),
            edges: self.edges.len(),
            roots,
            leaves,
            max_depth: self.depth.iter().copied().max().unwrap_or(0),
            mean_branching,
        }
    }
}

/// Probability-descending, id-ascending. Kept separate so both query paths (and
/// their tests) share exactly one definition of "best first".
pub fn sort_by_probability(states: &mut [StateProbability]) {
    states.sort_by(|a, b| {
        b.probability
            .partial_cmp(&a.probability)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64, at: i64) -> ProvenanceNode {
        ProvenanceNode::new(id, at, format!("s{id}"))
    }

    fn chain(n: u64) -> ProvenanceGraph {
        let nodes: Vec<_> = (0..n).map(|i| node(i, i as i64)).collect();
        let edges: Vec<_> = (0..n - 1)
            .map(|i| ProvenanceEdge::new(i, i + 1, 0.5))
            .collect();
        ProvenanceGraph::new(nodes, edges).expect("valid chain")
    }

    #[test]
    fn ancestry_and_probability_follow_the_path() {
        let g = chain(4);
        assert_eq!(g.ancestry(3).unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(g.depth(3).unwrap(), 3);
        assert!((g.probability(3).unwrap() - 0.125).abs() < 1e-12);
        assert_eq!(g.probability(0).unwrap(), 1.0);
    }

    #[test]
    fn frontier_is_the_leading_edge_not_everything() {
        let g = chain(4);
        assert_eq!(g.frontier(1).iter().map(|b| b.id).collect::<Vec<_>>(), vec![1]);
        assert_eq!(g.frontier(99).len(), 1);
        let at_root = g.frontier(0);
        assert_eq!(at_root.len(), 1);
        assert_eq!(at_root[0].id, 0);
        assert_eq!(g.frontier(-1).len(), 0);
    }

    #[test]
    fn branching_frontier_reports_all_leaves_sorted() {
        let nodes = vec![node(0, 0), node(1, 1), node(2, 1), node(3, 1)];
        let edges = vec![
            ProvenanceEdge::new(0, 1, 0.1),
            ProvenanceEdge::new(0, 2, 0.6),
            ProvenanceEdge::new(0, 3, 0.3),
        ];
        let g = ProvenanceGraph::new(nodes, edges).unwrap();
        let f = g.frontier(1);
        assert_eq!(f.iter().map(|b| b.id).collect::<Vec<_>>(), vec![2, 3, 1]);
        let sum: f64 = f.iter().map(|b| b.probability).sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pruning_reports_explained_mass_and_renormalises() {
        let nodes = vec![node(0, 0), node(1, 1), node(2, 1), node(3, 1)];
        let edges = vec![
            ProvenanceEdge::new(0, 1, 0.1),
            ProvenanceEdge::new(0, 2, 0.6),
            ProvenanceEdge::new(0, 3, 0.3),
        ];
        let g = ProvenanceGraph::new(nodes, edges).unwrap();
        let set = g.frontier_pruned(1, 0.25);
        assert_eq!(set.branches.len(), 2);
        assert!((set.explained - 0.9).abs() < 1e-12);
        let renorm = set.renormalized();
        assert!((renorm.iter().map(|(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((renorm[0].1 - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn deeper_branch_probability_is_a_product() {
        let nodes = vec![node(0, 0), node(1, 1), node(2, 2), node(3, 2)];
        let edges = vec![
            ProvenanceEdge::new(0, 1, 0.5),
            ProvenanceEdge::new(1, 2, 0.5),
            ProvenanceEdge::new(1, 3, 0.5),
        ];
        let g = ProvenanceGraph::new(nodes, edges).unwrap();
        assert!((g.probability(2).unwrap() - 0.25).abs() < 1e-12);
        let mass = g.mass_by_step();
        // Step 1 has one edge (0.5), step 2 has two (0.25 each = 0.5).
        assert_eq!(mass.len(), 2);
        assert!((mass[0].1 - 0.5).abs() < 1e-12);
        assert!((mass[1].1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn second_parent_is_rejected_not_merged() {
        let nodes = vec![node(0, 0), node(1, 0), node(2, 1)];
        let edges = vec![
            ProvenanceEdge::new(0, 2, 0.5),
            ProvenanceEdge::new(1, 2, 0.5),
        ];
        match ProvenanceGraph::new(nodes, edges) {
            Err(ProvenanceError::MultipleParents { child, .. }) => assert_eq!(child, 2),
            other => panic!("expected MultipleParents, got {other:?}"),
        }
    }

    #[test]
    fn cycle_is_rejected() {
        let nodes = vec![node(0, 0), node(1, 0)];
        let edges = vec![
            ProvenanceEdge::new(0, 1, 1.0),
            ProvenanceEdge::new(1, 0, 1.0),
        ];
        assert!(matches!(
            ProvenanceGraph::new(nodes, edges),
            Err(ProvenanceError::Cycle(_))
        ));
    }

    #[test]
    fn time_inversion_is_rejected() {
        let nodes = vec![node(0, 5), node(1, 4)];
        assert!(matches!(
            ProvenanceGraph::new(nodes, vec![ProvenanceEdge::new(0, 1, 1.0)]),
            Err(ProvenanceError::TimeInversion { .. })
        ));
    }

    #[test]
    fn equal_timestamps_are_allowed() {
        let nodes = vec![node(0, 3), node(1, 3)];
        assert!(ProvenanceGraph::new(nodes, vec![ProvenanceEdge::new(0, 1, 1.0)]).is_ok());
    }

    #[test]
    fn weight_bounds_include_one_and_reject_nan() {
        for bad in [0.0, -0.1, 1.5, f64::NAN] {
            let nodes = vec![node(0, 0), node(1, 1)];
            assert!(
                ProvenanceGraph::new(nodes, vec![ProvenanceEdge::new(0, 1, bad)]).is_err(),
                "weight {bad} should be rejected"
            );
        }
        let nodes = vec![node(0, 0), node(1, 1)];
        assert!(ProvenanceGraph::new(nodes, vec![ProvenanceEdge::new(0, 1, 1.0)]).is_ok());
    }

    #[test]
    fn duplicate_and_unknown_ids_are_reported() {
        assert!(matches!(
            ProvenanceGraph::new(vec![node(1, 0), node(1, 1)], vec![]),
            Err(ProvenanceError::DuplicateNode(1))
        ));
        assert!(matches!(
            ProvenanceGraph::new(vec![node(0, 0)], vec![ProvenanceEdge::new(0, 7, 1.0)]),
            Err(ProvenanceError::UnknownState(7))
        ));
        let g = chain(2);
        assert!(matches!(
            g.ancestry(42),
            Err(ProvenanceError::UnknownQuery(42))
        ));
    }

    #[test]
    fn stats_describe_shape() {
        let nodes = vec![node(0, 0), node(1, 1), node(2, 1)];
        let edges = vec![
            ProvenanceEdge::new(0, 1, 0.5),
            ProvenanceEdge::new(0, 2, 0.5),
        ];
        let s = ProvenanceGraph::new(nodes, edges).unwrap().stats();
        assert_eq!((s.nodes, s.edges, s.roots, s.leaves), (3, 2, 1, 2));
        assert_eq!(s.max_depth, 1);
        assert!((s.mean_branching - 2.0).abs() < 1e-12);
    }

    #[test]
    fn mass_by_step_flags_over_normalised_siblings() {
        let nodes = vec![node(0, 0), node(1, 1), node(2, 1)];
        // Two absolute probabilities recorded as conditional weights.
        let edges = vec![
            ProvenanceEdge::new(0, 1, 0.8),
            ProvenanceEdge::new(0, 2, 0.7),
        ];
        let g = ProvenanceGraph::new(nodes, edges).unwrap();
        let mass = g.mass_by_step();
        assert!(mass[0].1 > 1.0, "1.5 expected, got {}", mass[0].1);
    }
}
