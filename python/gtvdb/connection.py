"""`Connection` — the object `gtvdb.connect()` hands back.

    import gtvdb
    con = gtvdb.connect("gtv://127.0.0.1:50051")
    cur = con.cursor()
    cur.execute("SELECT * FROM numbers WHERE n <= ?", (3,))
    print(cur.fetchall())

What the connection *is*: one gRPC channel plus the pins that make a series of
answers comparable (`as_of`, `snapshot_id`). There is no server-side session,
no transaction, and no write buffer — the engine serves reads, and writes are
someone else's job (an RDBMS or Redis that refreshes into the store, per the
bank-scenario design). `commit()` and `rollback()` therefore exist only to
satisfy PEP 249 and do nothing; DML is refused by the engine, not faked here.

Pins can be changed without dropping the channel:

    con = gtvdb.connect(...)
    at_open = con.pin(as_of=1_700_000_000_000_000_000)
    at_open.cursor().execute("SELECT count(*) FROM positions")
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import pyarrow as pa

from . import errors
from ._params import render
from .client import DEFAULT_ADDR, Session
from .cursor import Cursor


class Connection:
    """A read-only, DB-API 2.0 connection to one gtv engine."""

    #: The engine never holds a transaction, so every statement is its own.
    autocommit = True

    def __init__(self, session: Session, *, owns_session: bool = True) -> None:
        self._session = session
        self._owns_session = owns_session
        self._closed = False

    # -- construction ------------------------------------------------------

    @classmethod
    def open(
        cls,
        dsn: Optional[str] = None,
        *,
        host: str = "127.0.0.1",
        port: int = 50051,
        as_of: Optional[int] = None,
        snapshot_id: Optional[int] = None,
        token: Optional[str] = None,
        ca_cert: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> "Connection":
        """Connect from a DSN or from keyword arguments.

        DSN forms accepted: `gtv://host:port`, `gtvdb://user@host:port`, or bare
        `host:port`. Query parameters override the keyword arguments:
        `?as_of=...&snapshot_id=...&token=...&ca_cert=...&timeout_ms=...`.
        """
        if dsn:
            parsed = _parse_dsn(dsn)
            host = parsed.get("host", host)
            port = int(parsed.get("port", port))
            as_of = _opt_int(parsed.get("as_of"), as_of)
            snapshot_id = _opt_int(parsed.get("snapshot_id"), snapshot_id)
            token = parsed.get("token", token)
            ca_cert = parsed.get("ca_cert", ca_cert)
            timeout_ms = _opt_int(parsed.get("timeout_ms"), timeout_ms)

        session = Session.connect(
            f"{host}:{port}",
            token=token,
            ca_cert=ca_cert,
            as_of=as_of,
            snapshot_id=snapshot_id,
            timeout_ms=timeout_ms,
        )
        return cls(session)

    # -- pins --------------------------------------------------------------

    def pin(self, *, as_of: Optional[int] = None, snapshot_id: Optional[int] = None) -> "Connection":
        """Same channel, different pinned business time / snapshot."""
        return Connection(self._session.at(as_of=as_of, snapshot_id=snapshot_id), owns_session=False)

    @property
    def as_of(self) -> Optional[int]:
        return self._session._as_of

    @property
    def snapshot_id(self) -> Optional[int]:
        return self._session._snapshot_id

    # -- PEP 249 -----------------------------------------------------------

    def cursor(self) -> Cursor:
        if self._closed:
            raise errors.InterfaceError("connection is closed")
        return Cursor(self)

    def execute(self, operation: str, parameters: Any = None) -> Cursor:
        """Shortcut: `con.execute(...)` is `con.cursor().execute(...)`.

        Not in PEP 249, but pandas' `read_sql` looks for it and sqlite3 has it.
        """
        return self.cursor().execute(operation, parameters)

    def commit(self) -> None:
        """No-op: the engine has no transaction to commit.

        Present because PEP 249 requires it, and because pandas calls
        `rollback()` on some code paths.
        """

    def rollback(self) -> None:
        """No-op, for the same reason as `commit()`."""

    def close(self) -> None:
        if self._owns_session:
            self._session.close()
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- engine-specific ---------------------------------------------------

    def read_table(self, operation: str, parameters: Any = None) -> pa.Table:
        """Skip the row conversion: get the Arrow table and its receipt.

        `pandas.read_sql` works too (it drives the cursor), but this path keeps
        Arrow types intact and avoids materialising Python tuples.
        """
        sql = render(operation, parameters)
        try:
            return self._session.query(sql).table
        except Exception as exc:  # GtvError -> DB-API exception
            raise _as_dbapi(exc) from exc

    @property
    def session(self) -> Session:
        """The protocol-v1 session underneath (receipts, handshake, version)."""
        return self._session

    @property
    def protocol_version(self) -> int:
        return self._session.protocol_version

    @property
    def handshake(self) -> Optional[Any]:
        return self._session.handshake

    def __repr__(self) -> str:
        return f"<gtvdb.Connection protocol={self._session.protocol_version} as_of={self.as_of}>"

    # -- internals ---------------------------------------------------------

    def _begin_query(self) -> None:
        if self._closed:
            raise errors.InterfaceError("connection is closed")


def _as_dbapi(exc: Exception) -> Exception:
    from .client import GtvError

    if isinstance(exc, GtvError):
        return errors.error_for(exc.code, exc.message)
    return exc


def _parse_dsn(dsn: str) -> dict:
    """`gtv://host:port?a=b` -> {"host": ..., "port": ..., "a": "b"}."""
    target = dsn if "://" in dsn else f"gtv://{dsn}"
    parsed = urlparse(target)
    if parsed.scheme not in ("gtv", "gtvdb", "grpc", "grpcs"):
        raise errors.InterfaceError(
            f"unsupported DSN scheme {parsed.scheme!r}; use gtv://host:port"
        )
    out: dict = {}
    if parsed.hostname:
        out["host"] = parsed.hostname
    if parsed.port:
        out["port"] = parsed.port
    for key, values in parse_qs(parsed.query).items():
        if values:
            out[key] = values[-1]
    return out


def _opt_int(raw: Optional[str], fallback: Optional[int]) -> Optional[int]:
    if raw is None:
        return fallback
    try:
        return int(raw)
    except ValueError as exc:
        raise errors.InterfaceError(f"DSN value {raw!r} is not an integer") from exc


__all__ = ["Connection", "DEFAULT_ADDR"]
