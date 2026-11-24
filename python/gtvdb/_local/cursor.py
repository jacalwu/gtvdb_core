"""PEP 249 Cursor over an in-process gtv engine (Arrow IPC results)."""

from __future__ import annotations

import io
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.ipc as ipc

from .._params import render
from .._types import (
    BINARY,  # noqa: F401 - re-exported for parity with the remote cursor
    BOOLEAN,
    DATE,
    DATETIME,
    INTEGER,
    NUMBER,
    ROWID,
    STRING,
    TIME,
    type_code,
)
from ..errors import InterfaceError, ProgrammingError


class Cursor:
    """One statement at a time. Create with ``connection.cursor()``."""

    arraysize: int = 1
    lastrowid: Optional[int] = None

    def __init__(self, connection: "Any") -> None:
        self._connection = connection
        self._table: Optional[pa.Table] = None
        self._rows: List[Tuple] = []
        self._position = 0
        self._closed = False

    # -- execution ---------------------------------------------------------

    def execute(self, operation: str, parameters: Any = None) -> "Cursor":
        """Run one statement. Returns `self`, so `cur.execute(...).fetchall()` works."""
        self._ensure_open()
        sql = render(operation, parameters)
        try:
            buf = self._connection._native.sql(sql)
        except Exception as exc:
            raise ProgrammingError(str(exc)) from exc
        self._load(buf)
        return self

    def executemany(self, operation: str, seq_of_parameters: Sequence[Any]) -> "Cursor":
        """Run the same statement once per parameter set."""
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
            (field.name, type_code(field.type), None, None, None, None, field.nullable)
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

    def to_arrow(self) -> pa.Table:
        """The result as an Arrow table — no row conversion.

        Same name and shape as the remote cursor's, so a result can be handed to
        `pandas`/`polars` without the caller knowing which mode produced it.
        """
        self._ensure_result()
        assert self._table is not None  # narrowed by _ensure_result
        return self._table

    def to_pandas(self):
        """Result as a DataFrame (requires `pip install pygtvdb[pandas]`)."""
        return self.to_arrow().to_pandas()

    def to_polars(self):
        """Result as a polars DataFrame (requires `pip install polars`)."""
        import polars as pl  # local import: polars is optional

        return pl.from_arrow(self.to_arrow())

    def arrow_table(self) -> Optional[pa.Table]:
        """Deprecated alias for `to_arrow`; kept for the retired `pygtvdb-local`."""
        return self._table

    # DB-API optional no-ops: there are no buffers to size.
    def setinputsizes(self, sizes: Any) -> None:
        pass

    def setoutputsize(self, size: Any, column: Any = None) -> None:
        pass

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
            # this is interface misuse. The remote cursor raises the same type
            # for the same state, and the two must be swappable.
            raise InterfaceError("no result set; call execute() first")

    def __iter__(self) -> Iterator[Tuple]:
        """Iterate the remaining rows as tuples (as the remote cursor does)."""
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
