"""`Cursor` — PEP 249 surface over one engine result set.

A cursor is a *result handle*, not a server-side resource: the engine returns
the whole answer in one round trip (there is no portal to page through, no
fetch-size hint that reaches the server). `fetchmany` therefore just slices a
list already in memory, and `arraysize` is a client-side size.

The one place this cursor goes beyond PEP 249 is the evidence it keeps:
`cursor.receipt` is the engine's `QueryReceipt`, and `cursor.tri_state` says
whether an empty `fetchall()` proved absence (`no_match`) or proved nothing
(`incomplete`). A DB-API consumer can ignore both; a detection workload must
not.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Iterator, Optional, Sequence

import pyarrow as pa

from . import errors
from ._params import render
from .client import GtvError, Result, Session

# PEP 249 requires these type objects so callers can compare
# `cursor.description[i][1] is gtvdb.STRING`. They are deliberately plain
# builtins, which is what sqlite3 and psycopg2 do. `INTEGER` and `BOOLEAN` are
# extensions: DBAPI20 folds every number into NUMBER, but an int column
# reported as `float` misleads anyone building an insert or a JSON payload.
STRING = str
BINARY = bytes
NUMBER = float
INTEGER = int
DATETIME = _dt.datetime
ROWID = int
BOOLEAN = bool

_INTEGERS = {
    pa.int8(),
    pa.int16(),
    pa.int32(),
    pa.int64(),
    pa.uint8(),
    pa.uint16(),
    pa.uint32(),
    pa.uint64(),
}
_FLOATS = {pa.float16(), pa.float32(), pa.float64()}
_STRINGS = {pa.string(), pa.large_string()}
_BINARY_TYPES = {pa.binary(), pa.large_binary()}
_TEMPORALS = {
    pa.timestamp("s"),
    pa.timestamp("ms"),
    pa.timestamp("us"),
    pa.timestamp("ns"),
    pa.date32(),
    pa.date64(),
}


def _type_code(arrow_type: pa.DataType) -> Any:
    """Map an Arrow type to a PEP 249 type object (or its name if unmapped)."""
    if arrow_type in _STRINGS:
        return STRING
    if arrow_type in _BINARY_TYPES:
        return BINARY
    if arrow_type in _INTEGERS:
        return INTEGER
    if arrow_type in _FLOATS or pa.types.is_decimal(arrow_type):
        return NUMBER
    if arrow_type == pa.bool_():
        return BOOLEAN
    if arrow_type in _TEMPORALS or isinstance(arrow_type, pa.TimestampType):
        return DATETIME
    return str(arrow_type)


class Cursor:
    """One statement at a time. Create with `connection.cursor()`."""

    #: DB-API: rows to fetch per `fetchmany()` call.
    arraysize: int = 1
    #: DB-API: `None` for a SELECT (the engine assigns no row ids).
    lastrowid: Optional[int] = None

    def __init__(self, connection: "Connection") -> None:
        self._connection = connection
        self._result: Optional[Result] = None
        self._rows: list[tuple] = []
        self._position = 0
        self._closed = False

    # -- execution ---------------------------------------------------------

    def execute(self, operation: str, parameters: Any = None) -> "Cursor":
        """Run one statement. Returns `self`, so `cur.execute(...).fetchall()` works."""
        self._ensure_open()
        sql = render(operation, parameters)
        self._connection._begin_query()
        try:
            result = self._connection.session.query(sql)
        except GtvError as exc:
            raise errors.error_for(exc.code, exc.message) from exc
        self._load(result)
        return self

    def executemany(self, operation: str, seq_of_parameters: Sequence[Any]) -> "Cursor":
        """Run the same statement once per parameter set.

        The engine has no batch DML, so this is a loop. `rowcount` reports the
        rows of the *last* statement, as PEP 249 allows.
        """
        self._ensure_open()
        for params in seq_of_parameters:
            self.execute(operation, params)
        return self

    # -- fetching ----------------------------------------------------------

    def fetchone(self) -> Optional[tuple]:
        self._ensure_result()
        if self._position >= len(self._rows):
            return None
        row = self._rows[self._position]
        self._position += 1
        return row

    def fetchmany(self, size: Optional[int] = None) -> list[tuple]:
        self._ensure_result()
        size = self.arraysize if size is None else size
        start = self._position
        self._position = min(start + size, len(self._rows))
        return self._rows[start : self._position]

    def fetchall(self) -> list[tuple]:
        self._ensure_result()
        rows = self._rows[self._position :]
        self._position = len(self._rows)
        return rows

    # -- PEP 249 attributes ------------------------------------------------

    @property
    def description(self) -> Optional[list[tuple]]:
        """`(name, type_code, display_size, internal_size, precision, scale, null_ok)`."""
        if self._result is None:
            return None
        schema = self._result.table.schema
        return [
            (field.name, _type_code(field.type), None, None, None, None, field.nullable)
            for field in schema
        ]

    @property
    def rowcount(self) -> int:
        """Rows in the result of the last statement, or `-1` before any."""
        return -1 if self._result is None else len(self._rows)

    # -- engine-specific: the evidence -------------------------------------

    @property
    def schema(self) -> Optional[pa.Schema]:
        """The Arrow schema of the last result (types `description` flattens)."""
        return None if self._result is None else self._result.table.schema

    @property
    def receipt(self) -> Optional[dict]:
        """The `QueryReceipt` for the last statement, if the server sent one."""
        return None if self._result is None else self._result.receipt

    @property
    def tri_state(self) -> Optional[str]:
        """`"found"` / `"no_match"` / `"incomplete"`, when the server computed one."""
        return None if self._result is None else self._result.tri_state

    @property
    def is_conclusive(self) -> bool:
        """Whether an empty result proves absence. `False` for an old server."""
        return False if self._result is None else self._result.is_conclusive

    @property
    def as_of_applied(self) -> bool:
        """Whether these rows really answer for the requested `as_of`."""
        return False if self._result is None else self._result.as_of_applied

    @property
    def snapshot_verified(self) -> bool:
        """Whether the server confirmed it serves the pinned snapshot."""
        return False if self._result is None else self._result.snapshot_verified

    def to_arrow(self) -> pa.Table:
        """The result as an Arrow table — no row conversion, keeps the receipt."""
        self._ensure_result()
        return self._result.table

    def to_pandas(self):
        """Result as a DataFrame (requires `pip install gtvdb[pandas]`)."""
        return self.to_arrow().to_pandas()

    def to_polars(self):
        """Result as a polars DataFrame (requires `pip install polars`)."""
        import polars as pl  # local import: polars is optional

        return pl.from_arrow(self.to_arrow())

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._closed = True
        self._result = None
        self._rows = []
        self._position = 0

    @property
    def closed(self) -> bool:
        return self._closed

    # DB-API optional no-ops: there are no server-side buffers to size.
    def setinputsizes(self, sizes: Any) -> None:
        pass

    def setoutputsize(self, size: Any, column: Any = None) -> None:
        pass

    def __iter__(self) -> Iterator[tuple]:
        self._ensure_result()
        while True:
            row = self.fetchone()
            if row is None:
                return
            yield row

    def __enter__(self) -> "Cursor":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"rows={len(self._rows)}"
        return f"<gtvdb.Cursor {state}>"

    # -- internals ---------------------------------------------------------

    def _load(self, result: Result) -> None:
        self._result = result
        table = result.table
        if table.num_rows == 0:
            self._rows = []
        else:
            columns = [column.to_pylist() for column in table.columns]
            self._rows = list(zip(*columns)) if columns else [()] * table.num_rows
        self._position = 0

    def _ensure_open(self) -> None:
        if self._closed:
            raise errors.InterfaceError("cursor is closed")
        if self._connection.closed:
            raise errors.InterfaceError("connection is closed")

    def _ensure_result(self) -> None:
        self._ensure_open()
        if self._result is None:
            raise errors.InterfaceError("no result set; call execute() first")
