"""gtvdb — a DB-API 2.0 driver for the gtv temporal-graph engine.

    import gtvdb

    with gtvdb.connect("gtv://127.0.0.1:50051") as con:
        with con.cursor() as cur:
            cur.execute("SELECT * FROM numbers WHERE n <= ?", (3,))
            print(cur.fetchall())            # [(1, 1.0), (2, 4.0), (3, 9.0)]

    import pandas as pd
    print(pd.read_sql("SELECT * FROM prices", con))

Install with `pip install gtvdb`; there is nothing to compile — the driver is
pure Python over gRPC, and Arrow comes from `pyarrow`.

# The three layers, in case you need to drop below DB-API

| layer | object | what it adds |
|---|---|---|
| DB-API 2.0 | `Connection` / `Cursor` | `execute` / `fetchall` / `description`, pandas, SQLAlchemy |
| session | `Session` / `Result` | `as_of` pinning, `QueryReceipt`, `tri_state` |
| wire | `gtvdb._stubs` | `gtvquery.QueryService`, protocol v1 |

DB-API has no vocabulary for "this empty result proves nothing", so the cursor
keeps the engine's own answer: `cur.tri_state == "no_match"` means absence was
proven, `"incomplete"` means it was not. Detection workloads must read it;
reporting workloads can ignore it.
"""

from __future__ import annotations

from typing import Any, Optional

from . import _version
from .client import DEFAULT_ADDR, GtvError, Result, Session
from .connection import Connection
from .cursor import (
    BINARY,
    BOOLEAN,
    DATETIME,
    INTEGER,
    NUMBER,
    ROWID,
    STRING,
    Cursor,
)
from .errors import (
    DataError,
    DatabaseError,
    Error,
    IntegrityError,
    InterfaceError,
    InternalError,
    NotSupportedError,
    OperationalError,
    ProgrammingError,
    Warning,
)

#: Single-sourced from `_version.py` so `pyproject.toml` and the runtime can
#: never disagree about what was published.
__version__ = _version.__version__

# --- PEP 249 module attributes ---------------------------------------------
apilevel = "2.0"
#: Threads may share the module and connections (the gRPC channel is thread
#: safe, and each cursor owns its result). Cursors themselves are not shared.
threadsafety = 2
#: `?` placeholders, rendered client-side (see `_params.py` for the exact rules).
paramstyle = "qmark"


def connect(
    dsn: Optional[str] = None,
    *,
    host: str = "127.0.0.1",
    port: int = 50051,
    as_of: Optional[int] = None,
    snapshot_id: Optional[int] = None,
    token: Optional[str] = None,
    ca_cert: Optional[str] = None,
    timeout_ms: Optional[int] = None,
    **kwargs: Any,
) -> Connection:
    """Open a connection. `dsn` is `gtv://host:port`, or pass `host`/`port`.

        gtvdb.connect()                                  # 127.0.0.1:50051
        gtvdb.connect("gtv://engine.internal:50061")
        gtvdb.connect("gtv://host:50051?as_of=1700000000000000000")
        gtvdb.connect(host="10.0.0.5", port=50051, token="...")

    `as_of` pins business time for every query on this connection — the
    equivalent of opening a kdb+ session at T. `token` is sent as a bearer
    token; `ca_cert` is a PEM path and switches the channel to TLS.
    """
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"connect() got unexpected keyword argument(s): {unknown}")
    return Connection.open(
        dsn,
        host=host,
        port=port,
        as_of=as_of,
        snapshot_id=snapshot_id,
        token=token,
        ca_cert=ca_cert,
        timeout_ms=timeout_ms,
    )


__all__ = [
    # PEP 249
    "apilevel",
    "threadsafety",
    "paramstyle",
    "connect",
    "Connection",
    "Cursor",
    "Warning",
    "Error",
    "InterfaceError",
    "DatabaseError",
    "DataError",
    "OperationalError",
    "IntegrityError",
    "InternalError",
    "ProgrammingError",
    "NotSupportedError",
    # type objects for `description`
    "STRING",
    "BINARY",
    "NUMBER",
    "INTEGER",
    "DATETIME",
    "ROWID",
    "BOOLEAN",
    # lower layers
    "Session",
    "Result",
    "GtvError",
    "DEFAULT_ADDR",
    "__version__",
]
