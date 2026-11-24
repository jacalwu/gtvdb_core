# pygtvdb-local

DuckDB-style, **in-process** Python driver for `gtvdb_core`. `pip install`
gives you the whole engine — no server, no Docker, no protoc, no Rust toolchain.

```bash
pip install pygtvdb-local
```

```python
import pygtvdb_local as gtv

con = gtv.connect()                       # engine runs inside this process
cur = con.cursor()
cur.execute("SELECT n, n*n AS sq FROM range(1, 4) t(n)")
print(cur.fetchall())                     # [(1, 1), (2, 4), (3, 9)]
print(cur.description[0][0])              # 'n'
cur.close()
con.close()
```

## Relation to `pygtvdb`

| | `pygtvdb` | `pygtvdb-local` |
|:--|:--|:--|
| Engine | remote `gtv-server` (gRPC) | embedded in-process |
| Import | `import gtvdb` | `import pygtvdb_local` |
| Needs a server | yes | **no** |
| Wheel | 30 KB, pure Python | platform wheel, ~28 MB compressed |
| Use for | production, multi-user, shared store | analysis, notebooks, prototypes, CI, single-machine recompute |

Both expose PEP 249 (`apilevel`, `connect`, `cursor`, `execute`, `fetch*`,
`description`, `rowcount`, context managers) and the same type codes, so moving
between them is a one-line import change. They install side by side.

## Local persistence (no server)

```python
import pyarrow as pa, pyarrow.parquet as pq
pq.write_table(pa.table({"n": [1, 2, 3]}), "/tmp/nums.parquet")

con = gtv.connect()
con.register_parquet("/tmp/nums.parquet", "nums")
cur = con.cursor()
cur.execute("SELECT sum(n) FROM nums")
assert cur.fetchall() == [(6,)]

# Or attach a directory catalog and publish its committed tables:
con.attach_catalog("/path/to/catalog")
n = con.sync_catalog()
```

## Engine-specific extras (beyond PEP 249)

- `cur.schema` — Arrow schema of the last result;
- `cur.arrow_table()` — result as `pyarrow.Table`;
- `gtv.connect(catalog="...")` — attach a catalog on open.

The core SQL surface (`as_of`, `ann`/`knn`, graph table functions, lineage, DQ
gates, `gtv_segments()`, `gtv_extensions()`) is available exactly as in the
server build. Domain workloads (banking / HFT / quantum) load through
`GtvExtension` in the companion `gtvdb_ext` repository.

## Requirements

- Python 3.9+ (single `abi3` wheel per platform covers all supported versions);
- `pyarrow>=14` (installed automatically);
- x86_64 Linux (glibc ≥ 2.34), x64 Windows, or Apple Silicon macOS. Wheels for
  all three are built by `.github/workflows/publish-python-local.yml`; until a
  release has shipped them, build locally with the script below.

## Building from source

One script builds the extension and drops it where Python imports it. Use it
rather than the two cargo commands — it normalises the three environment traps
that make a manual build fail in ways the error messages do not explain (Git
Bash's `link.exe` shadowing MSVC's, a global `RUSTFLAGS=-C linker=lld` without
`lld` installed, and `sccache` rejecting an exported `CARGO_INCREMENTAL`):

```bash
python python_local/build_native.py            # build, install into the package
python python_local/build_native.py --check    # report the toolchain, build nothing
python python_local/build_native.py --wheel    # also build the wheel
python python_local/build_native.py --clean    # cargo clean -p gtvpy-local first
python python_local/check_wheel.py             # install the wheel and query it
```

The wheel is `cp39-abi3`, so one artefact per platform covers Python 3.9–3.13.
Set `CARGO_NET_OFFLINE=1` if the workspace lock must not be re-resolved (adding
new deps online can restart the DataFusion 55 dependency graph).

Releases — the PyPI trusted-publisher setup, the platform matrix, and why a
partial matrix must never be published — are in [PUBLISHING.md](PUBLISHING.md).
