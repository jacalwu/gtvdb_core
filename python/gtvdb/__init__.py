"""gtvdb — a DB-API 2.0 driver for the gtv temporal-graph engine.

One `connect()` covers both ways to run the engine:

    import gtvdb

    # Remote: a gtv-server reachable over gRPC.
    con = gtvdb.connect("gtv://127.0.0.1:50051")

    # In-process: the engine lives in this Python process. No server, no Docker.
    con = gtvdb.connect("gtv://local", local_store="C:/data/gtv")

    with con:
        with con.cursor() as cur:
            cur.execute("SELECT * FROM numbers WHERE n <= ?", (3,))
            print(cur.fetchall())
            print(cur.tri_state)      # remote only: 'found' / 'no_match' / 'incomplete'

    import pandas as pd
    print(pd.read_sql("SELECT * FROM prices", con))

Install with `pip install pygtvdb`. The wheel for your platform carries the
compiled engine, so local mode works without a Rust toolchain; a wheel built
without it still supports remote mode and says so at `connect()`.

# The three layers, in case you need to drop below DB-API

| layer | object | what it adds |
|---|---|---|
| DB-API 2.0 | `Connection` / `Cursor` | `execute` / `fetchall` / `description`, pandas, SQLAlchemy |
| session | `Session` / `Result` | `as_of` pinning, `QueryReceipt`, `tri_state` |
| wire | `gtvdb._stubs` | `gtvquery.QueryService`, protocol v1 |

DB-API has no vocabulary for "this empty result proves nothing", so the cursor
keeps the engine's own answer: `cur.tri_state == "no_match"` means absence was
proven, `"incomplete"` means it was not. Detection workloads must read it;
reporting workloads can ignore it. The in-process engine has no wire protocol,
so it does not produce a tri-state; run detection workloads against a server.
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from . import _version
from ._types import (
    BINARY,
    BOOLEAN,
    DATE,
    DATETIME,
    INTEGER,
    NUMBER,
    ROWID,
    STRING,
    TIME,
)
from .client import DEFAULT_ADDR, GtvError, Result, Session
from .connection import Connection
from .cursor import Cursor
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


#: Host name that selects the in-process engine instead of a network address.
_LOCAL_HOST = "local"
#: Sentinel separating "not a local DSN" from "local DSN without a store".
_NETWORK = object()


def _local_store_from_dsn(dsn: Optional[str]) -> Any:
    """Return the store path when `dsn` selects the in-process engine.

    Args:
        dsn: Connection string, or None for the default remote address.

    Returns:
        `_NETWORK` for a remote address; `None` for local mode without a store;
        otherwise the store path.

    Notes:
        Two sentinels are required because "no store" (`None`) is a valid local
        configuration — folding it into "not local" would silently open a gRPC
        channel to a host literally named `local`.
    """
    if not dsn:
        return _NETWORK
    parsed = urlparse(dsn if "://" in dsn else f"gtv://{dsn}")
    if parsed.scheme not in ("gtv", "gtvdb", "grpc", "grpcs"):
        return _NETWORK
    if (parsed.hostname or "").lower() != _LOCAL_HOST:
        return _NETWORK
    params = parse_qs(parsed.query)
    for key in ("local_store", "catalog"):
        values = params.get(key)
        if values:
            return values[-1]
    return None


def _connect_local(store: Optional[str]) -> Connection:
    """Open the in-process engine.

    Imported lazily: a wheel built without the native artifact must still
    import the package (version, type codes, exception hierarchy), and a
    missing engine has to read as an interface problem at `connect()` time
    rather than as an ImportError escaping an unrelated line.
    """
    try:
        from ._local import connect as _local_connect
    except ImportError as exc:  # pragma: no cover - depends on the artefact
        raise InterfaceError(
            "local mode runs the engine inside this process, and this "
            "installation has no native engine: install pygtvdb from a wheel "
            "built for your platform"
        ) from exc
    return _local_connect(local_store=store)


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
    local_store: Optional[str] = None,
    **kwargs: Any,
) -> Connection:
    """Open a connection — remote over gRPC, or in-process against a local store.

        gtvdb.connect()                                       # 127.0.0.1:50051
        gtvdb.connect("gtv://engine.internal:50061")          # remote
        gtvdb.connect("gtv://host:50051?as_of=1700000000000") # remote, pinned
        gtvdb.connect("gtv://local")                          # in-process, memory
        gtvdb.connect("gtv://local", local_store="C:/data")   # in-process, on disk

    Args:
        dsn: `gtv://host:port`, or `gtv://local` for the in-process engine.
            `None` means the default remote address.
        host: Remote host; unused in local mode.
        port: Remote port; unused in local mode.
        as_of: Pin business time (ns since epoch) for every query on this
            connection — the equivalent of opening a kdb+ session at T.
        snapshot_id: Pin a specific snapshot instead of the newest.
        token: Bearer token for the remote server.
        ca_cert: PEM path; passing it switches the channel to TLS.
        timeout_ms: Per-query deadline on the remote server.
        local_store: Directory holding the in-process engine's data. Selecting
            it implies local mode even without a DSN; it may also be passed as
            `gtv://local?local_store=...`.

    Returns:
        A PEP 249 connection. In local mode the concrete class is
        `gtvdb._local.Connection`, which adds `register_parquet`,
        `attach_catalog` and `sync_catalog` for on-disk work.

    Raises:
        TypeError: An unknown keyword, or a remote-only option in local mode.
        InterfaceError: Unknown DSN scheme, or local mode requested on an
            installation built without the native engine.
    """
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"connect() got unexpected keyword argument(s): {unknown}")

    store = _local_store_from_dsn(dsn)
    if local_store is not None:
        # An explicit store selects local mode even without a DSN, so
        # `connect(local_store=...)` is the shortest honest spelling.
        store = local_store

    if store is not _NETWORK:
        remote_only = {
            "as_of": as_of,
            "snapshot_id": snapshot_id,
            "token": token,
            "ca_cert": ca_cert,
            "timeout_ms": timeout_ms,
        }
        given = [name for name, value in remote_only.items() if value is not None]
        if given:
            raise TypeError(
                f"local mode does not accept {', '.join(given)}; "
                "those apply to a remote server"
            )
        return _connect_local(store)

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
    "BOOLEAN",
    "ROWID",
    "DATETIME",
    "DATE",
    "TIME",
    # lower layers
    "Session",
    "Result",
    "GtvError",
    "DEFAULT_ADDR",
    "__version__",
]
