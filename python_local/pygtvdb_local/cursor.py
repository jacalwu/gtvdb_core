"""PEP 249 Cursor over an in-process gtv engine (Arrow IPC results)."""

from __future__ import annotations

import io
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.ipc as ipc

from ._params import render
from .errors import InterfaceError, ProgrammingError

# PEP 249 type codes are plain Python types, matching the network client so
# callers can switch between pygtvdb and pygtvdb-local without changes.
STRING = str
NUMBER = float
BINARY = bytes
INTEGER = int
BOOLEAN = bool
ROWID = int
DATETIME = __import__("datetime").datetime
DATE = __import__("datetime").date
TIME = __import__("datetime").time

_STRINGS = {pa.string(), pa.large_string()}
_BINARIES = {pa.binary(), pa.large_binary(), *[pa.binary(n) for n in (0,)]}


def _type_code(arrow_type: pa.DataType) -> Any:
    """Map an Arrow type to the same PEP 249 type object `gtvdb` reports.

    Every number folding into `NUMBER`/`float` is what DBAPI20 does, but it
    makes an `int64` column indistinguishable from a `double` — a caller
    building a JSON payload or an INSERT then coerces integers to floats. Keep
    `INTEGER` and `BOOLEAN` distinct so switching between the network client
    and this one does not change the reported schema.
    """
    if arrow_type in _STRINGS:
        return STRING
    if arrow_type in _BINARIES:
        return BINARY
    if pa.types.is_boolean(arrow_type):
        return BOOLEAN
    if pa.types.is_integer(arrow_type):
        return INTEGER
    if pa.types.is_timestamp(arrow_type):
        return DATETIME
    if pa.types.is_date(arrow_type):
        return DATE
    if pa.types.is_time(arrow_type):
        return TIME
    if pa.types.is_floating(arrow_type) or pa.types.is_decimal(arrow_type):
        return NUMBER
    return STRING


class Cursor:
    """One statement at a time. Create with ``connection.cursor()``."""

    arraysize: int = 1
    lastrowid: Optional[int] = None

    def __init__(self, connection: "Connection") -> None:
        self._connection = connection
        self._table: Optional[pa.Table] = None
        self._rows: List[Tuple] = []
        self._position = 0
        self._closed = False

    # -- execution ---------------------------------------------------------

    def execute(self, operation: str, parameters: Any = None) -> "Cursor":
        self._ensure_open()
        sql = render(operation, parameters)
        try:
            buf = self._connection._native.sql(sql)
        except Exception as exc:
            raise ProgrammingError(str(exc)) from exc
        self._load(buf)
        return self

    def executemany(self, operation: str, seq_of_parameters: Sequence[Any]) -> "Cursor":
        for params in seq_of_parameters:
            self.execute(operation, params)
        return self

    def _load(self, ipc_bytes: bytes) -> None:
        reader = ipc.open_stream(io.BytesIO(ipc_bytes))
        table = reader.read_all()
        self._table = table
        # Convert once to Python tuples in schema column order.
        cols = table.column_names
        self._rows = [tuple(row[c] for c in cols) for row in table.to_pylist()]
        self._position = 0

    # -- fetching ----------------------------------------------------------

    def fetchone(self) -> Optional[Tuple]:
        self._ensure_result()
        if self._position >= len(self._rows):
            return None
        row = self._rows[self._position]
        self._position += 1
        return row

    def fetchmany(self, size: Optional[int] = None) -> List[Tuple]:
        self._ensure_result()
        size = self.arraysize if size is None else size
        start = self._position
        self._position = min(start + size, len(self._rows))
        return self._rows[start : self._position]

    def fetchall(self) -> List[Tuple]:
        self._ensure_result()
        rows = self._rows[self._position :]
        self._position = len(self._rows)
        return rows

    # -- PEP 249 attributes ------------------------------------------------

    @property
    def description(self) -> Optional[List[Tuple]]:
        """``(name, type_code, display_size, internal_size, precision, scale, null_ok)``."""
        if self._table is None:
            return None
        return [
            (field.name, _type_code(field.type), None, None, None, None, field.nullable)
            for field in self._table.schema
        ]

    @property
    def rowcount(self) -> int:
        return -1 if self._table is None else len(self._rows)

    # -- engine-specific extras -------------------------------------------

    @property
    def schema(self) -> Optional[pa.Schema]:
        """The Arrow schema of the last result."""
        return None if self._table is None else self._table.schema

    def arrow_table(self) -> Optional[pa.Table]:
        """The full last result as a ``pyarrow.Table`` (engine-native)."""
        return self._table

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._force_close()

    def _force_close(self) -> None:
        self._closed = True
        self._table = None
        self._rows = []

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise InterfaceError("cursor is closed")

    def _ensure_result(self) -> None:
        self._ensure_open()
        if self._table is None:
            # InterfaceError, not ProgrammingError: nothing was queried yet, so
            # this is interface misuse. `gtvdb` raises the same type for the
            # same state, and the two drivers must be swappable.
            raise InterfaceError("no result set; call execute() first")

    def __iter__(self) -> Iterator[Tuple]:
        """Iterate the remaining rows as tuples (as `gtvdb` and sqlite3 do)."""
        self._ensure_result()
        while True:
            row = self.fetchone()
            if row is None:
                return
            yield row

    def __enter__(self) -> "Cursor":
        self._ensure_open()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
