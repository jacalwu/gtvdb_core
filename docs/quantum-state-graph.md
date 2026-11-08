# Quantum state metadata as a temporal graph

Status: **alpha** (API may change) ｜ Crates: `gtv-core::provenance`, `gtv-core::quantum_state`, `gtv-engine::quantum_state`

---

## 1. What problem this solves

A circuit is easy to describe and hard to explain. When a hybrid or variational run
produces an unexpected result, the questions are:

- Which gate prefix produced this state?
- How many branches were there at step *t*, and how much probability did each carry?
- Did the run lose probability mass (a cut-off we forgot, a mis-recorded branch)?

The usual artefact is the statevector: `2^n` complex amplitudes. It answers "what is
the state" but not "how did we get here", it cannot be diffed between runs, and at
`n = 30` it is 16 GiB.

What is actually needed is the **branch structure**: one row per state, one parent,
one gate, one timestamp, one probability. That object grows with the number of gates,
not with the Hilbert space, and it is a temporal graph — which is exactly what this
engine stores natively.

---

## 2. The record

```json
{"state_id": 0, "parent_state": null, "gate": "|000>",      "timestamp": 0, "probability": 1.0}
{"state_id": 1, "parent_state": 0,    "gate": "h q[0]",     "timestamp": 1, "probability": 1.0}
{"state_id": 2, "parent_state": 1,    "gate": "cx q[0],q[1]","timestamp": 2, "probability": 1.0}
{"state_id": 3, "parent_state": 2,    "gate": "measure q[0]","timestamp": 3, "probability": 0.5}
{"state_id": 4, "parent_state": 2,    "gate": "measure q[0]","timestamp": 3, "probability": 0.5}
{"state_id": 5, "parent_state": 3,    "gate": "rx(0.785398) q[1]", "timestamp": 4, "probability": 1.0}
```

| Field | Type | Meaning |
|:--|:--|:--|
| `state_id` | `u64` | node identity; unique within a run |
| `parent_state` | `u64?` | the incoming transition; `null` marks preparation |
| `gate` | `string` | canonical gate text: `name[(params)] [q[i][,q[j]]…]` |
| `timestamp` | `i64` | logical time: gate index, depth, or nanoseconds |
| `probability` | `f64` in `(0, 1]` | branch weight; meaning depends on semantics (§3) |

The file is JSONL: one run per file, `#` comments allowed, so a log can live in
version control with its own provenance note. This is deliberately the simplest
format that survives a crash — a Parquet writer is on the roadmap, not a prerequisite.

---

## 3. Two probability conventions (read this before trusting a number)

There are two legitimate things to record in that column, and mixing them silently
is the easiest way to produce a plausible but meaningless answer. Both are supported,
explicitly:

**Conditional** — `QuantumStateLog::from_records`. The weight is `P(this branch |
parent was reached)`. This is the quantum-trajectory convention: every jump has a
conditional probability, and the probability of a state is the product along its root
path. Siblings of one parent may sum to 1 (complete branching) or less (cut-off).

**Absolute** — `QuantumStateLog::from_records_absolute`. The weight is `P(this state)`
in the run as a whole, as read off a superposition snapshot; siblings at one step sum
to at most 1. Internally converted to conditional weights (`p_child / p_parent`) at
build time, so a state that is *more* probable than its parent is rejected as a data
error rather than silently producing a path product above 1.

In both cases `Reconstruction::probability` returns the probability of the state, and
`Coverage::ratio = child_mass / parent_mass` tells you how much of a parent's mass its
recorded children account for. A ratio above `1 + 1e-9` is reported by
`QuantumStateLog::violations` — that is a data bug, not a rounding artefact.

---

## 4. Guarantees

Enforced at construction (`ProvenanceGraph::new`), because every query assumes them:

| Invariant | Failure mode if violated | Behaviour |
|:--|:--|:--|
| Unique `state_id` | two states alias | `DuplicateNode` |
| At most one parent | "which parent explains this" has no correct answer | `MultipleParents` (merge is *rejected*, not guessed) |
| `parent.timestamp <= child.timestamp` | reconstruction walks into the future | `TimeInversion` |
| Acyclic | infinite reconstruction | `Cycle` (found by Kahn's traversal) |
| `probability ∈ (0, 1]`, finite | division by zero, NaN poisoning every product | `ProbabilityOutOfRange` / `WeightOutOfRange` |

Equal timestamps are allowed (several gates in one logical step). Timestamps may skip.

---

## 5. Query surface

### Rust

```rust
use gtv_core::quantum_state::QuantumStateLog;

let log = QuantumStateLog::from_jsonl(&text)?;

let r = log.reconstruct(5)?;          // O(depth)
r.circuit();                          // "|000>; h q[0]; cx q[0],q[1]; measure q[0]; rx(0.785398) q[1]"
r.probability;                        // 0.5
r.measured();                         // true — the path crosses a measurement

log.frontier(4);                      // live branches at t=4, most probable first
log.dominant(4, 0.05);               // pruned, with `explained` mass before renormalisation
log.coverage();                       // per-parent child mass ratio
log.violations(1e-9);                 // over-attributed parents
```

### SQL

```sql
SELECT * FROM quantum_state_summary('examples/quantum-state/ghz3.jsonl');
-- states | transitions | roots | leaves | max_depth | mean_branching | semantics

SELECT state_id, gate, probability, probability_renormalized
FROM quantum_state_frontier('examples/quantum-state/ghz3.jsonl', 6, 0.05)
ORDER BY probability DESC;

SELECT step, gate, prefix_probability
FROM quantum_state_reconstruct('examples/quantum-state/ghz3.jsonl', 8);
```

`prefix_probability` is the probability of the prefix ending at that row, so the last
row equals the state's probability and the column shows *where* a branch became
unlikely. Read `probability` before `probability_renormalized`: the normalised column
is a view of what survived the cut, not a fact about the run.

---

## 6. Cost

| Operation | Cost |
|:--|:--|
| Build / validate | O(states + transitions) |
| `reconstruct(state)` | O(depth) — path product, no log scan |
| `frontier(at)` | O(states + transitions) per call; incremental snapshots are roadmap work |
| Storage | one row per transition ≈ 60–80 bytes of JSON, ~24 bytes packed |

Compare: a full statevector for the same circuit is `2^n × 16` bytes. For a 20-qubit,
10 000-gate run the log is ~1 MB while the statevector is 16 MiB *per snapshot*.

---

## 7. Hooking a backend up

The log is backend-agnostic; anything that can emit `(parent, gate, step, p)` can feed
it. For a qoqo/PennyLane run, the natural insertion points are:

1. **Circuit level** — walk the circuit definition and emit one record per operation
   with `p = 1.0` (a coherent prefix has no branching). This gives free
   "which gate produced this state" for every intermediate.
2. **Measurement level** — for each mid-circuit measurement, emit one child per
   outcome with the conditional probability from the backend's marginal.
3. **Shot level** — a full trajectory is a root-to-leaf path; renormalise the leaf
   weights across shots to get empirical branch probabilities, then compare with the
   ideal ones (`violations` catches the mismatch).

Nothing in this module calls a simulator, and nothing requires one: logs can be
produced by a real backend, a simulator, or a classical model of the same branching.

---

## 8. Limits (stated up front, on purpose)

- **No amplitudes.** This is metadata. You cannot compute interference, fidelity, or
  entanglement entropy from it, and you cannot rebuild the statevector. Storing
  amplitude blobs *by reference* is future work; the graph is the index, not the state.
- **Probability is only as good as the producer.** The log validates internal
  consistency (`ratio <= 1`), not physical correctness.
- **Single-parent forest.** Genuine DAGs with merges (e.g. circuit rewriting where two
  paths converge) are rejected by design. If that becomes a real workload, the model
  needs an explicit merge node with a declared combination rule — not a silent default.
- **No de-duplication of identical sub-states.** Two branches that reach the same
  basis state stay distinct; that is usually what you want for provenance, but it means
  the log is not a compact representation of the wavefunction.

---

## 9. Roadmap for this surface

1. **Parquet landing + partition by run** — reuse `gtv-storage`, keep JSONL as the
   ingest format.
2. **Incremental frontier** — maintain the live branch set across appends instead of
   recomputing per query.
3. **Run comparison** — `diff(run_a, run_b)` over the same circuit: which step first
   diverged, and by how much.
4. **Amplitude side-table** — `state_id -> blob_ref` with the vector index over
   amplitude-derived features, so "find states similar to this one" becomes an ANN query
   on the same engine.
