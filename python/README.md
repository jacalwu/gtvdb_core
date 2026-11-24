# gtvdb (Python)

DB-API 2.0 driver for the gtv temporal-graph engine — **two modes behind one
`connect()`**:

| mode | DSN | engine runs | needs |
|:--|:--|:--|:--|
| remote | `gtv://host:port` | in `gtv-server`, over gRPC | a reachable server |
| local | `gtv://local` | **inside this process** | nothing else |

```bash
pip install pygtvdb            # or: pip install pygtvdb[pandas]
```

The wheel for your platform carries the compiled engine, so **local mode needs
no Rust toolchain, no Docker and no separate server**. (A wheel built without
the engine still serves remote mode, and says so at `connect()` rather than
failing somewhere deeper.)

> The PyPI distribution is named **`pygtvdb`**, but you **`import gtvdb`** —
> distribution name and import name are deliberately decoupled.

## Remote mode

Needs a running engine. The quickest one is the published container image (no
Rust toolchain, no git clone):

```bash
docker run --rm -p 50051:50051 -e GTV_ADDR=0.0.0.0:50051 \
  ghcr.io/jacalwu/gtvdb-core:0.1.0 gtv-demo-server
# in another shell: pip install pygtvdb && python your_client.py
```

```python
import gtvdb

con = gtvdb.connect("gtv://127.0.0.1:50051")
cur = con.cursor()
cur.execute("SELECT * FROM numbers WHERE n <= ?", (3,))
print(cur.fetchall())          # [(1, 1.0), (2, 4.0), (3, 9.0)]
print(cur.description[0][0])   # 'n'
print(cur.tri_state)           # remote only: 'found' / 'no_match' / 'incomplete'
con.close()
```

The image also ships `gtv`, the zero-extension SQL REPL
(`docker run --rm -it ghcr.io/jacalwu/gtvdb-core:0.1.0`). Linux binaries are on
the GitHub release tagged `v*`. `gtv-demo-server` serves one immutable snapshot
for protocol development; the production store-backed engine (`gtv-server`)
lives in the companion `gtvdb_ext` repository. From a source checkout,
`cargo run -p gtv-engine --example serve_demo` works too.

Works with pandas out of the box:

```python
import pandas as pd
pd.read_sql("SELECT * FROM prices ORDER BY t", con)
```

## Local mode

The same engine, compiled into the Python process. No server, no port, no
Docker — which is what makes notebooks and single-machine analysis practical:

```python
import gtvdb

con = gtvdb.connect("gtv://local")                              # in-memory
con = gtvdb.connect("gtv://local", local_store="C:/data/gtv")   # on disk

cur = con.cursor()
cur.execute("SELECT n, n*n AS sq FROM range(1, 4) t(n)")
print(cur.fetchall())          # [(1, 1), (2, 4), (3, 9)]
print([d[1].__name__ for d in cur.description])   # ['int', 'int']
```

`local_store` is a directory; it may also be passed in the DSN
(`gtv://local?local_store=C:/data/gtv`). Opening a store publishes its committed
tables immediately, so existing data is queryable without a second call.

```python
import pyarrow as pa, pyarrow.parquet as pq

pq.write_table(pa.table({"n": [1, 2, 3]}), "nums.parquet")

con = gtvdb.connect("gtv://local")
con.register_parquet("nums.parquet", "nums")          # a local file as a table
con.cursor().execute("SELECT sum(n) FROM nums").fetchall()   # [(6,)]
```

Local mode has no wire protocol, so it does not produce `tri_state` or
`receipt` — there is no evidence chain without a server. A detection workload
("an empty result must be *proved* empty") belongs on remote mode.

Options that only mean something for a server — `as_of`, `snapshot_id`, `token`,
`ca_cert`, `timeout_ms` — raise `TypeError` in local mode rather than being
silently ignored.

## Pinning business time

`as_of` is the engine's "answer as of T", and it belongs to the connection, not
to each query — the equivalent of opening a kdb+ session at a timestamp. Remote
mode only:

```python
con = gtvdb.connect("gtv://host:50051", as_of=1_700_000_000_000_000_000)
con.cursor().execute("SELECT count(*) FROM positions").fetchall()

at_open = con.pin(as_of=1_700_000_000_000_000_000)   # same channel, new pin
```

## Evidence, for workloads that need it

DB-API cannot express "this empty result proves nothing", so the remote cursor
keeps the engine's answer alongside the rows:

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
* **Not a wire format of its own.** Remote mode speaks `gtvquery.QueryService`
  v1 (`crates/gtv-proto/proto/gtvquery.proto`), the same contract the Rust
  client uses. `?` parameters are rendered client-side into typed literals,
  because protocol v1 has no bind slots.

## Development

```bash
pip install -e ./python                 # from the gtvdb_core repo root

# Build the in-process engine into the package, then the platform wheel.
python python/build_native.py --check   # report toolchain, build nothing
python python/build_native.py --wheel   # extension + wheel
python python/check_wheel.py            # install the wheel and query through it

python python/gen_stubs.py --check      # CI: fail if stubs drift
python -m unittest discover -s python/tests -t python/tests
```

`build_native.py` exists because the three ways this build fails are all
environmental and all reported badly: Git Bash's `link.exe` shadowing MSVC's, a
global `RUSTFLAGS=-C linker=lld` without `lld` installed, and `sccache`
rejecting an exported `CARGO_INCREMENTAL`. It normalises all three, builds via
cargo, and drops the artefact in `gtvdb/_local/`.

Run it from **PowerShell on Windows**, not from a restricted shell: a sandbox
that blocks `reg.exe` makes `link.exe` hang for tens of minutes with no output.

## Releasing

The version lives in exactly one place, `gtvdb/_version.py`; `pyproject.toml`
reads it back rather than repeating it. Before an upload:

```bash
python python/check_release.py --keep-dist   # licence copy, stub freshness,
                                             # tests, metadata, wheel contents,
                                             # engine presence, name on PyPI
```

The distribution is a **platform wheel** (`cp39-abi3`), because it carries the
engine. One artefact per platform covers Python 3.9–3.13.

Pushing a `py-v*` tag builds all three platforms and publishes through GitHub
Actions with PyPI trusted publishing, so no token is stored anywhere. No sdist
is uploaded: one without the engine would make pip try to compile Rust on a
machine that has none. The full procedure — first-release bootstrap, TestPyPI
rehearsal, the platform matrix, and why a version number can never be reused —
is in [PUBLISHING.md](https://github.com/jacalwu/gtvdb_core/blob/main/python/PUBLISHING.md).
