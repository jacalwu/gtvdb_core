# Time-point contract: a receipt may only name what was enforced

**Status** implemented (protocol v1, additive) ｜ **Applies to** `QueryReceipt`,
`GtvContext::{sql_with_receipt*,set_head_snapshot}`, the gRPC `ExecuteQuery` path.

---

## 1. The failure this closes

A client sent `as_of = 14:00`. The handler forwarded it to
`sql_with_receipt`, which wrote it into the receipt and ran the statement as
written. `SELECT * FROM holdings` has no time predicate, so the rows came from
whatever the engine currently held — and the receipt came back saying:

```json
{ "as_of": 1732178400000, "tri_state": "found", "stats": { "rows": 42 } }
```

That is not a missing feature. It is a receipt that testifies to something
nobody checked: it can be filed, quoted to a regulator, and compared against a
statement produced at a different time. `snapshot_id` had the same shape — it
was echoed from the request and never compared against anything. The wire even
had an `ErrorCode::SNAPSHOT_MISMATCH` that no code path ever produced.

A receipt that cannot distinguish "we answered for 14:00" from "the caller asked
for 14:00" is worse than one with no time field at all, because the first can be
mistaken for evidence.

## 2. The rule

Every named time point carries a **binding** (`PointBinding`):

| Binding | Meaning | Conclusive tri-state allowed |
|:--|:--|:--|
| `applied` | Something enforced the point before the query ran: the statement filtered by it, a session default reached every operator, or the snapshot was verified to be the head. | yes |
| `declared` | The point is recorded because it was requested. Nothing enforced it. | **no** |

Enforced by `QueryReceipt::validate`, which rejects three shapes:

1. `as_of` / `snapshot_id` present **without** a binding — the field would be an
   unattributed claim;
2. a binding present **without** its point — a claim about nothing;
3. a `Declared` binding together with `Found` / `NoMatch` — the case above.

A `Declared` point therefore always arrives as `Incomplete`, with
`IncompleteReason::TimePointNotApplied` (unless a truncation already explains the
result better; truncation says something about the *data*, so it keeps the
reason, and the binding still records the unenforced point).

`absence_at` refuses to prove absence under a `Declared` `as_of`: an unfiltered
empty result is not an answer about the requested business time.

## 3. What this does *not* do

It does not make `as_of` work. That is deliberate, and it is the whole point of
landing this first:

* **No statement rewriting.** Turning `SELECT * FROM t` into an as-of query
  requires knowing which column is the time axis. Guessing silently changes the
  answer, which is worse than declining to.
* **No session default.** The time-aware operators (`as_of`, `neighbors`,
  `embedding_search`) take the time as an argument, so a default cannot reach
  them without per-request session state — a larger change than this one.
* **No snapshot identity in storage.** `SegmentStore` defines no snapshot id, and
  `manifest.system_ts` cannot serve as one: `flush_at` accepts a repeated
  timestamp, so two different states can share it. Until storage defines an
  identity, a client-supplied id is unverifiable, and "unverifiable" is reported
  as `declared`, not assumed to match.

So today, a `snapshot_id` sent to a server that did not call `set_head_snapshot`
comes back `declared`. That is the honest answer to "is this the state I pinned?"
— which is *"I cannot tell you"*.

## 4. Making a point `applied`

**Business time** — the handler knows whether the statement answers for it:

```rust
// The motif table was materialised at `t`, so this statement really does
// answer for that business time.
ctx.sql_with_receipt_binding(
    query_id,
    "SELECT * FROM gtv_motif_matches",
    QueryBinding::at(t).applied(),
    &[("gtv_motif_matches".into(), summary)],
)
.await
```

Without `.applied()` the same query is reported `Incomplete` — the safe
direction to be wrong in, because a correct answer labelled inconclusive is a
nuisance while an inconclusive answer labelled correct is an audit finding.

**Snapshot** — the host declares what it serves, and the engine verifies against
it:

```rust
ctx.set_head_snapshot(published_manifest_id);   // or a replica seqno, a catalog generation
```

`sql_with_receipt_binding` then checks a requested `snapshot_id` **before** the
statement runs:

| Head declared | Requested | Result |
|:--|:--|:--|
| none | any | `declared` → `Incomplete` |
| `7` | `7` | `applied`, verified |
| `7` | `9` | `Err(SnapshotMismatch { expected: 9, got: 7 })` |

Refusing beats re-reading the head and hoping nobody notices: the client pinned
`9` on purpose, and rows from `7` are a different fact.

## 5. Wire surface

No protocol change — the fields are additive inside the receipt, which already
travels twice (JSON + Arrow schema metadata).

| Key | Value |
|:--|:--|
| `as_of_binding` | `applied` / `declared` |
| `snapshot_binding` | `applied` / `declared` |
| `incomplete_reason` | `time_point_not_applied` when a declared point capped the result |

`ErrorCode::SNAPSHOT_MISMATCH` (4) is now actually emitted, by handlers that
call `GtvContext::verify_snapshot` before running the query — `serve_demo` and
`gtv-server` both do.

Receipts written before this change still parse: the new fields are
`#[serde(default)]`, so an archived receipt reads back with no binding rather
than failing to load. It cannot be *validated* if it names a point without one —
which is correct, since that is exactly the claim it could not support.

## 6. Client guidance

* A result with `tri_state != incomplete` and no `as_of_binding` answers for
  "now", not for any requested time. That is normal and conclusive.
* A result with `as_of_binding = declared` means the pin was recorded only. Do
  not use it as evidence about that time; re-run with a statement that filters
  by it.
* `SNAPSHOT_MISMATCH` means the state you pinned is not being served. Retry
  against the head (or refresh your pin); do not treat it as a missing row.

## 7. Non-goals

* Applying `as_of` automatically (see §3) — separate decision, needs a time-axis
  metadata story.
* Defining the snapshot identity in storage — needs a monotone, unique generation
  in the manifest; `system_ts` is not it.
* Deciding session-level pinning defaults (`pin on connect` vs `latest each
  time`) — a server policy question, not a receipt one.
