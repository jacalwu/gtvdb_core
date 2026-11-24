# gtvdb (Python)

DB-API 2.0 driver for the gtv temporal-graph engine. Pure Python over gRPC —
nothing to compile, no platform wheel, no `protoc` on the user's machine.

```bash
pip install pygtvdb            # or: pip install pygtvdb[pandas]
```

> The PyPI distribution is named **`pygtvdb`** (the shorter `gtvdb` name was
> unavailable at publish time), but you still **`import gtvdb`** in Python —
> distribution name and import name are deliberately decoupled, so the code
> below is unchanged from earlier previews.

Requires Python 3.9+ and a reachable engine. The driver is a client only: it
does not embed the engine, and it cannot create one. The quickest engine is the
published container image (no Rust toolchain, no git clone):

```bash
docker run --rm -p 50051:50051 -e GTV_ADDR=0.0.0.0:50051 \
  ghcr.io/jacalwu/gtvdb-core:0.1.0 gtv-demo-server
# in another shell: pip install pygtvdb && python your_client.py
```

The image also ships `gtv`, the zero-extension SQL REPL
(`docker run --rm -it ghcr.io/jacalwu/gtvdb-core:0.1.0`). Linux binaries are on
the GitHub release tagged `v*`. The `gtv-demo-server` serves one immutable
snapshot for protocol development; the production store-backed engine
(`gtv-server`) lives in the companion `gtvdb_ext` repository. From a source
checkout, `cargo run -p gtv-engine --example serve_demo` works too.

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

## Releasing

The version lives in exactly one place, `gtvdb/_version.py`; `pyproject.toml`
reads it back rather than repeating it. Before an upload:

```bash
python python/check_release.py          # licence copy, stub freshness, tests,
                                        # metadata, artefact contents, name on PyPI
```

Pushing a `py-v*` tag publishes through GitHub Actions with PyPI trusted
publishing, so no token is stored anywhere. The full procedure — including the
first-release bootstrap, TestPyPI rehearsal, and why a version number can never
be reused — is in [PUBLISHING.md](https://github.com/jacalwu/gtvdb_core/blob/main/python/PUBLISHING.md).
