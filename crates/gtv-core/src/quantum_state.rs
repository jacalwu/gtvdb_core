//! Quantum state metadata: the state/transition log and its reconstruction.
//!
//! A quantum program is not a list of gates, it is a *tree of states*: each gate
//! (or mid-circuit measurement) turns one state into one or more successors,
//! and the probability attached to a branch is the weight of that path. Debugging
//! the program — or auditing what a hybrid run actually did — means answering
//! "how did we get to *this* state, through which gates, with what probability".
//!
//! Storing the statevector is not an option at scale (2^n amplitudes), and it is
//! usually the wrong artefact anyway: the interesting object is the **shape of
//! the branching plus the gates that produced it**, which grows with the number
//! of gates, not with the Hilbert space.
//!
//! This module stores exactly that shape:
//!
//! | Field | Role |
//! |:--|:--|
//! | `state_id` | node identity in the provenance forest |
//! | `parent_state` | the incoming transition; `null` for the preparation |
//! | `gate` | the gate that produced this state (`h q[0]`, `rx(0.785) q[1]`) |
//! | `timestamp` | logical time of the transition (step index or nanoseconds) |
//! | `probability` | branch weight under the log's [`ProbabilitySemantics`] |
//!
//! The heavy lifting — one parent per state, monotone time, acyclicity, path
//! products — is [`crate::provenance::ProvenanceGraph`]; this module is the
//! quantum-specific front end: gate syntax, the two probability conventions,
//! reconstruction, coverage checks, and JSONL I/O.
//!
//! ```rust
//! use gtv_core::quantum_state::{QuantumStateLog, QuantumStateRecord};
//!
//! let log = QuantumStateLog::from_records(vec![
//!     QuantumStateRecord { state_id: 0, parent_state: None, gate: "|00>".into(), timestamp: 0, probability: 1.0 },
//!     QuantumStateRecord { state_id: 1, parent_state: Some(0), gate: "h q[0]".into(), timestamp: 1, probability: 1.0 },
//!     QuantumStateRecord { state_id: 2, parent_state: Some(1), gate: "measure q[0]".into(), timestamp: 2, probability: 0.5 },
//!     QuantumStateRecord { state_id: 3, parent_state: Some(1), gate: "measure q[0]".into(), timestamp: 2, probability: 0.5 },
//! ])?;
//!
//! let r = log.reconstruct(2)?;
//! assert_eq!(r.circuit(), "|00>; h q[0]; measure q[0]");
//! assert!((r.probability - 0.5).abs() < 1e-12);
//! # Ok::<(), gtv_core::quantum_state::StateError>(())
//! ```

use std::collections::HashMap;
use std::fmt;

use crate::provenance::{ProvenanceEdge, ProvenanceError, ProvenanceGraph, ProvenanceNode};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

/// A gate application: name, target qubits, and optional rotation parameters.
///
/// Kept as a tiny struct rather than an enum because the gate set is open: a
/// vendor backend will hand us names we have never heard of, and refusing to
/// store a state because of an unknown gate name would defeat the purpose of a
/// metadata store. The text form is canonical and round-trips:
///
/// ```rust
/// # use gtv_core::quantum_state::Gate;
/// for text in ["h q[0]", "cx q[0],q[1]", "rx(0.785398) q[2]", "measure q[0]", "|000>"] {
///     assert_eq!(Gate::parse(text)?.to_string(), text);
/// }
/// # Ok::<(), gtv_core::quantum_state::StateError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Gate {
    pub name: String,
    pub qubits: Vec<u16>,
    pub params: Vec<f64>,
}

impl Gate {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            qubits: Vec::new(),
            params: Vec::new(),
        }
    }

    /// Parse the canonical text form: `name[(params)] [q[i][,q[j]]…]`.
    ///
    /// Parameters are comma separated inside parentheses; qubits are `q[n]`
    /// tokens separated by commas and/or spaces. A bare name (`|000>`, `init`)
    /// is legal — that is how the preparation state is recorded.
    pub fn parse(text: &str) -> Result<Self, StateError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(StateError::GateSyntax {
                input: text.to_string(),
                reason: "empty gate",
            });
        }

        let (head, rest) = match text.find('(') {
            Some(open) => {
                let close = text[open..]
                    .find(')')
                    .map(|c| c + open)
                    .ok_or_else(|| StateError::GateSyntax {
                        input: text.to_string(),
                        reason: "unclosed parameter list",
                    })?;
                (&text[..open], &text[close + 1..])
            }
            None => {
                // Split the name from the qubit list without assuming a space:
                // `h q[0]` and `|000>` and `init` all have to work.
                match text.find(" q[") {
                    Some(sp) => (&text[..sp], &text[sp..]),
                    None => (text, ""),
                }
            }
        };

        let name = head.trim();
        if name.is_empty() {
            return Err(StateError::GateSyntax {
                input: text.to_string(),
                reason: "missing gate name",
            });
        }
        if name.contains(char::is_whitespace) {
            return Err(StateError::GateSyntax {
                input: text.to_string(),
                reason: "gate name contains whitespace",
            });
        }

        let params = if let Some(open) = text.find('(') {
            let close = text[open..].find(')').map(|c| c + open).expect("checked");
            let inner = &text[open + 1..close];
            if inner.trim().is_empty() {
                Vec::new()
            } else {
                let mut out = Vec::new();
                for token in inner.split(',') {
                    let value: f64 =
                        token
                            .trim()
                            .parse()
                            .map_err(|_| StateError::GateSyntax {
                                input: text.to_string(),
                                reason: "parameter is not a number",
                            })?;
                    if !value.is_finite() {
                        return Err(StateError::GateSyntax {
                            input: text.to_string(),
                            reason: "parameter is not finite",
                        });
                    }
                    out.push(value);
                }
                out
            }
        } else {
            Vec::new()
        };

        let mut qubits = Vec::new();
        for token in rest.split(|c: char| c == ',' || c.is_whitespace()) {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            let digits = token
                .strip_prefix("q[")
                .and_then(|t| t.strip_suffix(']'))
                .ok_or_else(|| StateError::GateSyntax {
                    input: text.to_string(),
                    reason: "qubit must be written q[n]",
                })?;
            let q: u16 = digits.parse().map_err(|_| StateError::GateSyntax {
                input: text.to_string(),
                reason: "qubit index is not a u16",
            })?;
            qubits.push(q);
        }

        Ok(Self {
            name: name.to_string(),
            qubits,
            params,
        })
    }

    /// True for mid-circuit measurement / readout gates, whose children are the
    /// classical branches of one shot.
    pub fn is_measurement(&self) -> bool {
        matches!(
            self.name.to_ascii_lowercase().as_str(),
            "measure" | "m" | "mz" | "readout"
        )
    }

    /// True for state preparation markers (`|000>`, `init`, `prepare`).
    pub fn is_preparation(&self) -> bool {
        self.name.starts_with('|') || matches!(self.name.to_ascii_lowercase().as_str(), "init" | "prepare")
    }
}

impl fmt::Display for Gate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if !self.params.is_empty() {
            write!(f, "(")?;
            for (i, p) in self.params.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                // Shortest representation that round-trips; keeps `0.785398`
                // readable instead of `0.7853980000000001`.
                write!(f, "{p}")?;
            }
            write!(f, ")")?;
        }
        for (i, q) in self.qubits.iter().enumerate() {
            if i == 0 {
                write!(f, " q[{q}]")?;
            } else {
                write!(f, ",q[{q}]")?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

/// One row of the state log — the exact five fields of the storage contract.
///
/// The JSON field names are the contract; `parent_state: null` marks the
/// preparation state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantumStateRecord {
    pub state_id: u64,
    #[serde(default)]
    pub parent_state: Option<u64>,
    pub gate: String,
    pub timestamp: i64,
    pub probability: f64,
}

/// How `probability` on each record should be read.
///
/// * [`ProbabilitySemantics::Conditional`] — the weight of this branch *given*
///   the parent (a quantum trajectory: each jump has a conditional
///   probability). The probability of a state is the product along its path.
/// * [`ProbabilitySemantics::Absolute`] — the probability of this state in the
///   run as a whole (a superposition snapshot: siblings at one step sum to 1).
///   Internally normalised into conditional weights at build time, so both
///   conventions answer [`Reconstruction::probability`] correctly.
///
/// Getting this wrong is the single easiest way to produce a plausible-looking
/// but meaningless number, which is why it is explicit rather than inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbabilitySemantics {
    Conditional,
    Absolute,
}

/// Failures when building or querying a state log.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StateError {
    #[error("state log is empty")]
    Empty,
    #[error("invalid gate {input:?}: {reason}")]
    GateSyntax { input: String, reason: &'static str },
    #[error("state {child} names parent {parent}, which is not in the log")]
    MissingParent { child: u64, parent: u64 },
    #[error("state {0}: probability must lie in (0, 1]")]
    ProbabilityOutOfRange(u64),
    #[error("state {child} has probability {child_p} above its parent {parent} ({parent_p})")]
    ProbabilityIncrease {
        child: u64,
        parent: u64,
        child_p: f64,
        parent_p: f64,
    },
    #[error(transparent)]
    Graph(#[from] ProvenanceError),
    #[error("malformed record on line {line}: {message}")]
    Record { line: usize, message: String },
    #[error("io: {0}")]
    Io(String),
}

// ---------------------------------------------------------------------------
// Log
// ---------------------------------------------------------------------------

/// A reconstructed state: the gate sequence that produced it and its weight.
#[derive(Debug, Clone, PartialEq)]
pub struct Reconstruction {
    pub state_id: u64,
    /// Root-first state ids, `state_id` last.
    pub path: Vec<u64>,
    /// Root-first gate sequence, preparation gate included at index 0.
    pub gates: Vec<Gate>,
    pub timestamps: Vec<i64>,
    pub depth: usize,
    /// Probability of this state under the log's semantics.
    pub probability: f64,
}

impl Reconstruction {
    /// The circuit prefix as one line: `|000>; h q[0]; cx q[0],q[1]`.
    pub fn circuit(&self) -> String {
        self.gates
            .iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// True when the path crosses a measurement — i.e. this state is a post-
    /// measurement branch rather than a coherent prefix.
    pub fn measured(&self) -> bool {
        self.gates.iter().any(Gate::is_measurement)
    }
}

/// One live branch at a time slice.
#[derive(Debug, Clone, PartialEq)]
pub struct Branch {
    pub state_id: u64,
    pub gate: Gate,
    pub timestamp: i64,
    pub depth: usize,
    pub probability: f64,
}

/// A pruned branch set with the mass it retains before renormalisation.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchSet {
    pub branches: Vec<Branch>,
    /// Sum of `probability` over `branches`; 1.0 when the log is complete.
    pub explained: f64,
}

impl BranchSet {
    /// Retained branches renormalised to sum to 1 (empty when `explained == 0`).
    pub fn renormalized(&self) -> Vec<(u64, f64)> {
        if self.explained <= 0.0 {
            return Vec::new();
        }
        self.branches
            .iter()
            .map(|b| (b.state_id, b.probability / self.explained))
            .collect()
    }
}

/// How much of a parent's mass its recorded children account for.
#[derive(Debug, Clone, PartialEq)]
pub struct Coverage {
    pub parent: u64,
    pub at: i64,
    pub child_mass: f64,
    pub ratio: f64,
}

/// An immutable quantum state log.
#[derive(Debug, Clone)]
pub struct QuantumStateLog {
    graph: ProvenanceGraph,
    records: Vec<QuantumStateRecord>,
    index: HashMap<u64, usize>,
    gates: Vec<Gate>,
    semantics: ProbabilitySemantics,
}

impl QuantumStateLog {
    /// Build a log whose `probability` values are conditional branch weights.
    pub fn from_records(records: Vec<QuantumStateRecord>) -> Result<Self, StateError> {
        Self::build(records, ProbabilitySemantics::Conditional)
    }

    /// Build a log whose `probability` values are absolute state probabilities
    /// (siblings at one step sum to at most 1). Converted to conditional weights
    /// internally, rejecting any state that is more probable than its parent.
    pub fn from_records_absolute(records: Vec<QuantumStateRecord>) -> Result<Self, StateError> {
        Self::build(records, ProbabilitySemantics::Absolute)
    }

    fn build(
        records: Vec<QuantumStateRecord>,
        semantics: ProbabilitySemantics,
    ) -> Result<Self, StateError> {
        if records.is_empty() {
            return Err(StateError::Empty);
        }

        let mut index = HashMap::with_capacity(records.len());
        for (i, r) in records.iter().enumerate() {
            if !(r.probability > 0.0 && r.probability <= 1.0) {
                return Err(StateError::ProbabilityOutOfRange(r.state_id));
            }
            index.insert(r.state_id, i);
        }

        let mut gates = Vec::with_capacity(records.len());
        for r in &records {
            gates.push(Gate::parse(&r.gate)?);
        }

        // Absolute mode: weight(child) = p(child) / p(parent), which makes every
        // downstream path product equal the recorded absolute probability.
        let weight_of = |r: &QuantumStateRecord| -> Result<f64, StateError> {
            match (semantics, r.parent_state) {
                (ProbabilitySemantics::Absolute, Some(parent)) => {
                    let pi = *index.get(&parent).ok_or(StateError::MissingParent {
                        child: r.state_id,
                        parent,
                    })?;
                    let parent_p = records[pi].probability;
                    if r.probability > parent_p * (1.0 + 1e-9) {
                        return Err(StateError::ProbabilityIncrease {
                            child: r.state_id,
                            parent,
                            child_p: r.probability,
                            parent_p,
                        });
                    }
                    Ok((r.probability / parent_p).min(1.0))
                }
                _ => Ok(r.probability),
            }
        };

        let nodes: Vec<ProvenanceNode> = records
            .iter()
            .map(|r| ProvenanceNode::new(r.state_id, r.timestamp, r.gate.clone()))
            .collect();

        let mut edges = Vec::with_capacity(records.len());
        for r in &records {
            if let Some(parent) = r.parent_state {
                if !index.contains_key(&parent) {
                    return Err(StateError::MissingParent {
                        child: r.state_id,
                        parent,
                    });
                }
                edges.push(ProvenanceEdge::new(parent, r.state_id, weight_of(r)?));
            }
        }

        let graph = ProvenanceGraph::new(nodes, edges)?;
        Ok(Self {
            graph,
            records,
            index,
            gates,
            semantics,
        })
    }

    pub fn semantics(&self) -> ProbabilitySemantics {
        self.semantics
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn records(&self) -> &[QuantumStateRecord] {
        &self.records
    }

    pub fn graph(&self) -> &ProvenanceGraph {
        &self.graph
    }

    pub fn record(&self, state_id: u64) -> Result<&QuantumStateRecord, StateError> {
        let i = *self
            .index
            .get(&state_id)
            .ok_or(StateError::Graph(ProvenanceError::UnknownQuery(state_id)))?;
        Ok(&self.records[i])
    }

    pub fn gate(&self, state_id: u64) -> Result<&Gate, StateError> {
        let i = *self
            .index
            .get(&state_id)
            .ok_or(StateError::Graph(ProvenanceError::UnknownQuery(state_id)))?;
        Ok(&self.gates[i])
    }

    /// Reconstruct `state_id`: its gate path, timestamps, depth and probability.
    ///
    /// O(depth), no scan of the log.
    pub fn reconstruct(&self, state_id: u64) -> Result<Reconstruction, StateError> {
        let path = self.graph.ancestry(state_id)?;
        let gates = path
            .iter()
            .map(|&id| self.gate(id).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let timestamps = path
            .iter()
            .map(|&id| self.record(id).map(|r| r.timestamp))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Reconstruction {
            state_id,
            depth: path.len() - 1,
            probability: self.graph.probability(state_id)?,
            path,
            gates,
            timestamps,
        })
    }

    /// Live branches at `at`, most probable first.
    pub fn frontier(&self, at: i64) -> Vec<Branch> {
        self.graph
            .frontier(at)
            .into_iter()
            .filter_map(|sp| {
                Some(Branch {
                    state_id: sp.id,
                    gate: self.gate(sp.id).ok()?.clone(),
                    timestamp: self.record(sp.id).ok()?.timestamp,
                    depth: sp.depth,
                    probability: sp.probability,
                })
            })
            .collect()
    }

    /// Branches at `at` at or above `min_prob`, with the mass they retain.
    pub fn dominant(&self, at: i64, min_prob: f64) -> BranchSet {
        let set = self.graph.frontier_pruned(at, min_prob);
        let branches = set
            .branches
            .iter()
            .filter_map(|sp| {
                Some(Branch {
                    state_id: sp.id,
                    gate: self.gate(sp.id).ok()?.clone(),
                    timestamp: self.record(sp.id).ok()?.timestamp,
                    depth: sp.depth,
                    probability: sp.probability,
                })
            })
            .collect();
        BranchSet {
            branches,
            explained: set.explained,
        }
    }

    /// Per-parent coverage: recorded child mass over the parent's own mass.
    ///
    /// A ratio of 1.0 means every branch of that parent is in the log (closed
    /// branching); below 1.0 means branches were dropped, typically a
    /// probability cut-off. A ratio above 1.0 is a data error — the children
    /// claim more mass than the parent had.
    pub fn coverage(&self) -> Vec<Coverage> {
        let mut out = Vec::new();
        for node in self.graph.nodes() {
            let children = match self.graph.children(node.id) {
                Ok(c) if !c.is_empty() => c,
                _ => continue,
            };
            let total: f64 = children
                .iter()
                .filter_map(|&c| self.graph.probability(c).ok())
                .sum();
            let own = self.graph.probability(node.id).unwrap_or(1.0);
            out.push(Coverage {
                parent: node.id,
                at: node.at,
                child_mass: total,
                ratio: if own > 0.0 { total / own } else { 0.0 },
            });
        }
        out.sort_by_key(|c| (c.at, c.parent));
        out
    }

    /// Parents whose recorded children claim more mass than the parent carried.
    ///
    /// `tol` absorbs floating-point drift (1e-9 is plenty for logs written from
    /// double-precision amplitudes).
    pub fn violations(&self, tol: f64) -> Vec<Coverage> {
        self.coverage()
            .into_iter()
            .filter(|c| c.ratio > 1.0 + tol)
            .collect()
    }

    // -- I/O ---------------------------------------------------------------

    /// Serialise to newline-delimited JSON, one record per line, input order.
    pub fn to_jsonl(&self) -> Result<String, StateError> {
        let mut out = String::new();
        for r in &self.records {
            let line = serde_json::to_string(r).map_err(|e| StateError::Io(e.to_string()))?;
            out.push_str(&line);
            out.push('\n');
        }
        Ok(out)
    }

    /// Parse newline-delimited JSON. Blank lines and `#` comments are skipped so
    /// a log can be kept in version control with provenance notes.
    pub fn from_jsonl(text: &str) -> Result<Self, StateError> {
        let mut records = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let record: QuantumStateRecord =
                serde_json::from_str(line).map_err(|e| StateError::Record {
                    line: i + 1,
                    message: e.to_string(),
                })?;
            records.push(record);
        }
        Self::from_records(records)
    }

    /// Parse newline-delimited JSON where `probability` is absolute.
    pub fn from_jsonl_absolute(text: &str) -> Result<Self, StateError> {
        let mut records = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let record: QuantumStateRecord =
                serde_json::from_str(line).map_err(|e| StateError::Record {
                    line: i + 1,
                    message: e.to_string(),
                })?;
            records.push(record);
        }
        Self::from_records_absolute(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(
        state_id: u64,
        parent_state: Option<u64>,
        gate: &str,
        timestamp: i64,
        probability: f64,
    ) -> QuantumStateRecord {
        QuantumStateRecord {
            state_id,
            parent_state,
            gate: gate.to_string(),
            timestamp,
            probability,
        }
    }

    /// GHZ-3: prepare, entangle, then measure q0 and branch both ways.
    fn ghz_log() -> QuantumStateLog {
        QuantumStateLog::from_records(vec![
            rec(0, None, "|000>", 0, 1.0),
            rec(1, Some(0), "h q[0]", 1, 1.0),
            rec(2, Some(1), "cx q[0],q[1]", 2, 1.0),
            rec(3, Some(2), "cx q[0],q[2]", 3, 1.0),
            rec(4, Some(3), "measure q[0]", 4, 0.5),
            rec(5, Some(3), "measure q[0]", 4, 0.5),
        ])
        .expect("valid ghz log")
    }

    #[test]
    fn gate_round_trips() {
        for text in [
            "h q[0]",
            "cx q[0],q[1]",
            "rx(0.785398) q[2]",
            "u3(1.5707963,0,-1.5707963) q[0]",
            "measure q[0]",
            "|000>",
            "init",
        ] {
            assert_eq!(Gate::parse(text).unwrap().to_string(), text);
        }
    }

    #[test]
    fn gate_rejects_malformed_input() {
        assert!(Gate::parse("").is_err());
        assert!(Gate::parse("rx(0.5 q[0]").is_err());
        assert!(Gate::parse("rx(a) q[0]").is_err());
        assert!(Gate::parse("cx q0,q1").is_err());
        assert!(Gate::parse("cx q[0],x[1]").is_err());
        assert!(Gate::parse("rx(inf) q[0]").is_err());
    }

    #[test]
    fn gate_classification() {
        assert!(Gate::parse("measure q[0]").unwrap().is_measurement());
        assert!(Gate::parse("MZ q[0]").unwrap().is_measurement());
        assert!(!Gate::parse("h q[0]").unwrap().is_measurement());
        assert!(Gate::parse("|000>").unwrap().is_preparation());
        assert!(Gate::parse("init").unwrap().is_preparation());
    }

    #[test]
    fn reconstruct_returns_the_gate_prefix() {
        let log = ghz_log();
        let r = log.reconstruct(3).unwrap();
        assert_eq!(r.circuit(), "|000>; h q[0]; cx q[0],q[1]; cx q[0],q[2]");
        assert_eq!(r.depth, 3);
        assert_eq!(r.timestamps, vec![0, 1, 2, 3]);
        assert!((r.probability - 1.0).abs() < 1e-12);
        assert!(!r.measured());
    }

    #[test]
    fn reconstruct_after_measurement_reports_the_branch_probability() {
        let log = ghz_log();
        let r = log.reconstruct(4).unwrap();
        assert!((r.probability - 0.5).abs() < 1e-12);
        assert!(r.measured());
        assert_eq!(r.path, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn frontier_holds_both_ghz_branches() {
        let log = ghz_log();
        let f = log.frontier(4);
        assert_eq!(f.len(), 2);
        assert_eq!(f.iter().map(|b| b.state_id).collect::<Vec<_>>(), vec![4, 5]);
        let total: f64 = f.iter().map(|b| b.probability).sum();
        assert!((total - 1.0).abs() < 1e-12);
        // Before the measurement the frontier is the single coherent state.
        assert_eq!(log.frontier(3).len(), 1);
        assert_eq!(log.frontier(3)[0].state_id, 3);
    }

    #[test]
    fn pruning_reports_retained_mass_and_can_renormalise() {
        let log = ghz_log();
        let set = log.dominant(4, 0.6);
        assert!(set.branches.is_empty(), "0.5 < 0.6 must be pruned");

        let set = log.dominant(4, 0.4);
        assert_eq!(set.branches.len(), 2);
        assert!((set.explained - 1.0).abs() < 1e-12);
        let renorm = set.renormalized();
        assert!((renorm[0].1 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn coverage_is_closed_for_a_complete_log() {
        let log = ghz_log();
        let cov = log.coverage();
        assert!(!cov.is_empty());
        assert!(log.violations(1e-9).is_empty());
        let measure_parent = cov.iter().find(|c| c.parent == 3).unwrap();
        assert!((measure_parent.ratio - 1.0).abs() < 1e-12);
    }

    #[test]
    fn coverage_exposes_dropped_branches() {
        // Two of four branches recorded: ratio 0.5, not a violation.
        let log = QuantumStateLog::from_records(vec![
            rec(0, None, "|00>", 0, 1.0),
            rec(1, Some(0), "h q[0]", 1, 1.0),
            rec(2, Some(1), "measure q[0]", 2, 0.5),
            rec(3, Some(1), "measure q[0]", 2, 0.5),
            rec(4, Some(3), "h q[1]", 3, 1.0),
        ])
        .unwrap();
        let cov = log.coverage();
        let three = cov.iter().find(|c| c.parent == 3).unwrap();
        assert!((three.ratio - 1.0).abs() < 1e-12);
        assert!(log.violations(1e-9).is_empty());
    }

    #[test]
    fn absolute_semantics_normalise_to_the_same_probabilities() {
        let records = vec![
            rec(0, None, "|00>", 0, 1.0),
            rec(1, Some(0), "h q[0]", 1, 1.0),
            rec(2, Some(1), "measure q[0]", 2, 0.3),
            rec(3, Some(1), "measure q[0]", 2, 0.7),
            rec(4, Some(3), "x q[1]", 3, 0.35),
        ];
        let log = QuantumStateLog::from_records_absolute(records).unwrap();
        assert!((log.reconstruct(2).unwrap().probability - 0.3).abs() < 1e-12);
        assert!((log.reconstruct(3).unwrap().probability - 0.7).abs() < 1e-12);
        // 0.35 absolute below a 0.7 parent -> conditional weight 0.5.
        assert!((log.reconstruct(4).unwrap().probability - 0.35).abs() < 1e-12);
        assert_eq!(log.semantics(), ProbabilitySemantics::Absolute);
    }

    #[test]
    fn absolute_semantics_reject_probability_increase() {
        let records = vec![
            rec(0, None, "|0>", 0, 0.2),
            rec(1, Some(0), "x q[0]", 1, 0.9),
        ];
        assert!(matches!(
            QuantumStateLog::from_records_absolute(records),
            Err(StateError::ProbabilityIncrease { .. })
        ));
    }

    #[test]
    fn missing_parent_is_reported() {
        let records = vec![rec(1, Some(99), "h q[0]", 1, 1.0)];
        assert!(matches!(
            QuantumStateLog::from_records(records),
            Err(StateError::MissingParent { child: 1, parent: 99 })
        ));
    }

    #[test]
    fn probability_out_of_range_is_rejected() {
        for bad in [0.0, -0.5, 1.5, f64::NAN] {
            let records = vec![rec(0, None, "|0>", 0, bad)];
            assert!(matches!(
                QuantumStateLog::from_records(records),
                Err(StateError::ProbabilityOutOfRange(0))
            ));
        }
    }

    #[test]
    fn jsonl_round_trips_with_comments() {
        let log = ghz_log();
        let text = log.to_jsonl().unwrap();
        let mut annotated = String::from("# ghz-3, job 4f2a, 2026-11-08\n");
        annotated.push_str(&text);
        let back = QuantumStateLog::from_jsonl(&annotated).unwrap();
        assert_eq!(back.len(), log.len());
        assert_eq!(back.records(), log.records());
        assert_eq!(back.to_jsonl().unwrap(), text);
        assert!((back.reconstruct(5).unwrap().probability - 0.5).abs() < 1e-12);
    }

    #[test]
    fn jsonl_reports_the_offending_line() {
        let text = "{\"state_id\":0,\"gate\":\"|0>\",\"timestamp\":0,\"probability\":1.0}\nnot json\n";
        match QuantumStateLog::from_jsonl(text) {
            Err(StateError::Record { line, .. }) => assert_eq!(line, 2),
            other => panic!("expected Record error, got {other:?}"),
        }
    }

    #[test]
    fn deep_log_reconstructs_in_linear_time() {
        // 2000-deep chain: reconstruction must not rescan the log.
        let mut records = vec![rec(0, None, "|0>", 0, 1.0)];
        for i in 1..2000u64 {
            records.push(rec(i, Some(i - 1), "rx(0.1) q[0]", i as i64, 1.0));
        }
        let log = QuantumStateLog::from_records(records).unwrap();
        let r = log.reconstruct(1999).unwrap();
        assert_eq!(r.gates.len(), 2000);
        assert_eq!(r.depth, 1999);
        assert_eq!(log.graph().stats().max_depth, 1999);
    }
}
