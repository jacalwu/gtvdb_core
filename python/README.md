# gtvdb (Python)

DB-API 2.0 driver for the gtv temporal-graph engine. Pure Python over gRPC —
nothing to compile, no platform wheel, no `protoc` on the user's machine.

```bash
pip install gtvdb            # or: pip install gtvdb[pandas]
```

```python
import gtvdb

con = gtvdb.connect("gtv://127.0.0.1:50051")
cur = con.cursor()
cur.execute("SELECT * FROM numbers WHERE n <= ?", (3,))
print(cur.fetchall())          # [(1, 1.0), (2, 4.0), (3, 9.0)]
print(cur.description[0][0])   # 'n'
con.close()
```

Works with pandas out of the box:

```python
import pandas as pd
pd.read_sql("SELECT * FROM prices ORDER BY t", con)
```

## Pinning business time

`as_of` is the engine's "answer as of T", and it belongs to the connection, not
to each query — the equivalent of opening a kdb+ session at a timestamp:

```python
con = gtvdb.connect("gtv://host:50051", as_of=1_700_000_000_000_000_000)
con.cursor().execute("SELECT count(*) FROM positions").fetchall()

at_open = con.pin(as_of=1_700_000_000_000_000_000)   # same channel, new pin
```

## Evidence, for workloads that need it

DB-API cannot express "this empty result proves nothing", so the cursor keeps
the engine's answer alongside the rows:

```python
cur.execute("SELECT * FROM edges WHERE src = ?", (99,))
cur.fetchall()        # []
cur.tri_state         # 'no_match'  -> absence is proven
cur.receipt           # full QueryReceipt: snapshot, as_of, freshness
```

`tri_state == "incomplete"` (budget or deadline hit) means the empty result
proves nothing. `cur.is_conclusive` is `False` for an old server that cannot
compute a tri-state at all.

## What it is not

* **Read-only.** `commit()` / `rollback()` are no-ops; DML is refused by the
  engine, not faked by the driver. Writes belong to an RDBMS or Redis that
  refreshes into the store.
* **Not a wire format of its own.** It speaks `gtvquery.QueryService` v1
  (`crates/gtv-proto/proto/gtvquery.proto`), the same contract the Rust client
  uses. `?` parameters are rendered client-side into typed literals, because
  protocol v1 has no bind slots.

## Development

```bash
pip install -e ./python                 # from the gtvdb_core repo root
python python/gen_stubs.py              # regenerate gtvdb/_stubs from the .proto
python python/gen_stubs.py --check      # CI: fail if stubs drift
python -m pytest python/tests           # unit tests, no server needed
```
