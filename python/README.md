# gtvdb — Python driver for the gtv engine

> 中文说明（本地与远程两种用法）：[README.zh-CN.md](README.zh-CN.md)

One package, one `connect()`, two ways to run the engine:

| mode | DSN | engine runs in | needs |
|:--|:--|:--|:--|
| **remote** | `gtv://host:50051` | a `gtv-server` process, spoken to over gRPC | a reachable server |
| **local** | `gtv://local` | **this Python process** (compiled extension) | nothing else |

```python
import gtvdb

con = gtvdb.connect("gtv://127.0.0.1:50051")                   # remote
con = gtvdb.connect("gtv://local")                             # local, in memory
con = gtvdb.connect("gtv://local", local_store="C:/data/gtv")  # local, on disk
```

Both return the same PEP 249 connection: same cursor, same `description` type
codes, same exception hierarchy, same `?` parameters. What changes is *where*
the engine runs — and therefore whether you get the evidence chain
(`tri_state` / `receipt`), which only a wire protocol can carry.

## Install

```bash
pip install pygtvdb              # ~27 MB: the wheel carries the compiled engine
pip install "pygtvdb[pandas]"    # + DataFrame helpers
pip install "pygtvdb[polars]"    # + to_polars()
```

> The distribution is named **`pygtvdb`**; you **`import gtvdb`**. The names are
> deliberately decoupled.

The wheel is platform-specific (`cp39-abi3`, so one wheel per OS covers Python
3.9–3.13) because the engine is inside it. That is what makes local mode work
without a Rust toolchain or a container. A wheel built without the engine still
serves remote mode and says so at `connect("gtv://local")` instead of failing
somewhere deeper.

## Local mode — nothing to start

```python
import gtvdb

con = gtvdb.connect("gtv://local")
cur = con.cursor()
cur.execute("SELECT n, n * n AS sq FROM range(1, 4) t(n)")
print(cur.fetchall())                              # [(1, 1), (2, 4), (3, 9)]
print([d[1].__name__ for d in cur.description])    # ['int', 'int']
```

`local_store` is a directory, and it may also ride in the DSN
(`gtv://local?local_store=C:/data/gtv`). Opening an existing store publishes its
committed tables immediately, so "open my database" means the data is queryable
without a second call:

```python
con = gtvdb.connect("gtv://local", local_store="C:/data/gtv")
con.sync_catalog()          # publish any tables committed since the last open
```

Files can be queried in place, without ingesting them:

```python
import pyarrow as pa, pyarrow.parquet as pq

pq.write_table(pa.table({"n": [1, 2, 3]}), "nums.parquet")

con = gtvdb.connect("gtv://local")
con.register_parquet("nums.parquet", "nums")                 # local file as a table
print(con.cursor().execute("SELECT sum(n) FROM nums").fetchall())   # [(6,)]
```

Use it for notebooks, single-machine analysis, tests and CI: there is no port to
bind, no container to start, and each `Connection` owns a private engine, so two
connections cannot see each other's registered tables.

## Remote mode — one server, many clients

```bash
# The engine, zero dependencies. `gtv-demo-server` serves one immutable snapshot
# for protocol work; a store-backed server is what you run in production.
docker run --rm -p 50051:50051 -e GTV_ADDR=0.0.0.0:50051 \
  ghcr.io/jacalwu/gtvdb-core:0.1.0 gtv-demo-server
```

```python
import gtvdb

con = gtvdb.connect("gtv://127.0.0.1:50051")
cur = con.cursor()
cur.execute("SELECT n, square FROM numbers WHERE n <= ?", (3,))
print(cur.fetchall())      # [(1, 1.0), (2, 4.0), (3, 9.0)]
print(cur.tri_state)       # 'found'
con.close()
```

Remote mode is what you want when several processes or machines read one store,
and it is the only mode with a receipt. The same image also ships `gtv`, the
zero-extension SQL REPL (`docker run --rm -it ghcr.io/jacalwu/gtvdb-core:0.1.0`).

Connection options, all remote-only:

```python
gtvdb.connect(
    "gtv://engine.internal:50061?as_of=1700000000000000000&timeout_ms=5000",
    token="bearer-token",     # authentication
    ca_cert="/etc/gtv/ca.pem",  # passing this switches the channel to TLS
)
```

`as_of` is `ns` since the epoch and belongs to the *connection*, not to a query —
the equivalent of opening a kdb+ session at T:

```python
con = gtvdb.connect("gtv://host:50051", as_of=1_700_000_000_000_000_000)
con.cursor().execute("SELECT count(*) FROM positions").fetchall()

same_channel = con.pin(as_of=1_700_000_000_000_000_000)
```

Passing a remote-only option in local mode raises `TypeError` rather than
silently ignoring it — a pin that quietly does nothing is worse than a failure.

## Which mode

| | local | remote |
|:--|:--|:--|
| start-up | process-local, no server | needs a reachable `gtv-server` |
| data sharing | this process only | many clients, one store |
| `tri_state` / `receipt` | **absent** | present |
| `as_of` / `snapshot_id` pinning | **absent** | present |
| persistence | `local_store=` directory | the server's store |
| best for | notebooks, CI, one-machine analysis | production, shared data, detection |

Local mode has no wire protocol, so it produces no evidence chain. A workload
where "an empty result must be *proved* empty" (fraud, risk, anti-detection)
belongs on remote mode; reporting workloads can ignore the difference.

## Results without row objects

`fetchall()` converts every row to Python tuples. For anything large, take the
Arrow table instead — either mode supports both spellings:

```python
cur.to_arrow()                    # pyarrow.Table, types preserved
cur.to_pandas()                   # DataFrame
cur.to_polars()                   # polars DataFrame
con.read_table("SELECT * FROM big")   # Arrow, skipping the cursor entirely
```

`pandas.read_sql("SELECT ...", con)` works too.

## Parameters and type codes

`paramstyle` is `qmark`; named parameters also work:

```python
cur.execute("SELECT * FROM t WHERE n = ?", (3,))
cur.execute("SELECT * FROM t WHERE n = :n", {"n": 3})
```

Scalars accepted: `None` → `NULL`, `bool`, `int`, `float`, `Decimal`, `str`,
`bytes` → `X'..'`, `datetime`/`date`/`time` → `TIMESTAMP '..'`.
Non-finite floats raise `DataError`.

`cursor.description[i][1]` is a real type object, so `is` comparisons work:

```python
cur.description[0][1] is gtvdb.INTEGER    # not NUMBER — ints stay ints
```

`STRING`, `BINARY`, `NUMBER`, `INTEGER`, `BOOLEAN`, `ROWID`, `DATETIME`, `DATE`,
`TIME`. `INTEGER` and `BOOLEAN` are extensions: DBAPI20 folds every number into
`NUMBER`, which reports an `int` column as `float` and silently corrupts anyone
building an INSERT or a JSON payload from `description`.

## The evidence chain (remote only)

DB-API cannot express "this empty result proves nothing", so the remote cursor
keeps the engine's own answer next to the rows:

```python
cur.execute("SELECT * FROM edges WHERE src = ?", (99,))
cur.fetchall()        # []
cur.tri_state         # 'no_match'  -> absence was proven
cur.receipt           # snapshot, as_of, freshness, stats
```

| `tri_state` | meaning |
|:--|:--|
| `found` | rows matched |
| `no_match` | nothing matched, **and that is proven** |
| `incomplete` | budget or deadline hit — the empty result proves nothing |

`cur.is_conclusive` is `False` against a server too old to compute a tri-state at
all. `as_of_applied` and `snapshot_verified` report the same way: whether the
answer really is for the time point you asked for.

## What it is not

* **Read-only.** `commit()` / `rollback()` are no-ops, and `autocommit` is
  `True`. DML is refused by the engine, not faked by the driver: writes belong to
  an RDBMS or Redis that refreshes into the store.
* **Not its own wire format.** Remote mode speaks `gtvquery.QueryService` v1
  (`crates/gtv-proto/proto/gtvquery.proto`), the same contract the Rust client
  uses. Protocol v1 has no bind slots, so `?` parameters are rendered
  client-side into typed literals — safe against injection, but the engine
  re-plans on every call, and a parameter cannot stand where a table or column
  name is expected.
* **Not thread-local.** `threadsafety` is 2: the module and connections may be
  shared across threads; cursors may not (each owns its own result).

## Development

```bash
pip install -e ./python                  # from the repository root

python python/build_native.py --check    # report the toolchain, build nothing
python python/build_native.py --wheel    # engine + platform wheel
python python/check_wheel.py             # install that wheel and query through it

python python/gen_stubs.py --check       # fail if the gRPC stubs drifted

# The suite runs from this directory, not from an installed package.
cd python && python -m unittest discover -s tests -t tests
```

`build_native.py` exists because the three ways this build fails are all
environmental and all reported badly: Git Bash's `link.exe` shadowing MSVC's, a
global `RUSTFLAGS=-C linker=lld` with no `lld` installed, and `sccache` refusing
an exported `CARGO_INCREMENTAL`. It normalises all three, then drops the artefact
into `gtvdb/_local/`. Run it from **PowerShell on Windows** — a sandboxed shell
that blocks `reg.exe` makes `link.exe` hang for tens of minutes with no output.

## Releasing

The version lives in exactly one place, `gtvdb/_version.py`; `pyproject.toml`
reads it back instead of repeating it.

```bash
python python/check_release.py --keep-dist
# licence copy, stub freshness, tests, metadata, wheel contents, engine
# presence, and whether the name is still free on PyPI
```

Pushing a `py-v*` tag builds all three platforms and uploads through GitHub
Actions with PyPI trusted publishing, so no token is stored anywhere. No sdist
is published: one without the engine would make pip try to compile Rust on a
machine that has none. The full procedure — first-release bootstrap, TestPyPI
rehearsal, the platform matrix, and why a version number can never be reused —
is in [PUBLISHING.md](https://github.com/jacalwu/gtvdb_core/blob/main/python/PUBLISHING.md).
