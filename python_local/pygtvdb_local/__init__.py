"""pygtvdb-local — in-process (DuckDB-style) Python driver for gtvdb_core.

Unlike :mod:`gtvdb` (a thin client that needs a running ``gtv-server``), this
package embeds the engine in the Python process via a compiled extension. There
is no server, no Docker and no protoc to install — just ``pip install`` and
query.

    >>> import pygtvdb_local as gtv
    >>> con = gtv.connect()                       # local engine, in-process
    >>> cur = con.cursor()
    >>> cur.execute("SELECT 1 AS n")
    >>> cur.fetchall()
    [(1,)]

The distribution is named ``pygtvdb-local`` and imported as ``pygtvdb_local``.
It can be installed alongside ``pygtvdb`` (the network client): import names
are distinct, and both expose the same PEP 249 surface.

Engine-specific extras, on top of PEP 249:

* ``cursor.schema`` — the Arrow schema of the last result;
* ``cursor.arrow_table()`` — the result as a ``pyarrow.Table`` (zero extra copy
  beyond the IPC decode);
* ``connection.register_parquet(path, name)`` / ``attach_catalog(root)`` /
  ``sync_catalog()`` — local persistence, no server involved.
"""

from __future__ import annotations

from typing import Any, Optional

from . import _version
from .connection import Connection

# Re-export the PEP 249 type objects, same names as `gtvdb`, so a caller can
# compare `cursor.description[i][1] is gtv.INTEGER` against either driver.
from .cursor import (  # noqa: E402
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

__version__ = _version.__version__

# --- PEP 249 module attributes ---------------------------------------------
apilevel = "2.0"
#: Connections may be shared between threads; cursors may not. Queries run on
#: each connection's private runtime with the GIL released.
threadsafety = 1
#: `?` placeholders, rendered client-side by `_params.render`.
paramstyle = "qmark"

# Re-export the PEP 249 exception hierarchy (``gtv.Warning``, ``gtv.Error``, …)
# so callers can catch errors off the module, as with sqlite3/psycopg.
from .errors import (  # noqa: E402
    DatabaseError,
    DataError,
    Error,
    IntegrityError,
    InterfaceError,
    InternalError,
    NotSupportedError,
    OperationalError,
    ProgrammingError,
    Warning,
)

__all__ = [
    "connect",
    "Connection",
    "apilevel",
    "threadsafety",
    "paramstyle",
    # type objects for `description`, mirrored from `gtvdb`
    "STRING",
    "BINARY",
    "NUMBER",
    "INTEGER",
    "BOOLEAN",
    "ROWID",
    "DATETIME",
    "DATE",
    "TIME",
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
]


def connect(
    dsn: Optional[str] = None,
    *,
    catalog: Optional[str] = None,
    **kwargs: Any,
) -> Connection:
    """Open an in-process engine connection.

    ``dsn`` is optional and accepted for symmetry with :mod:`gtvdb`; the local
    engine has no network address. Pass ``catalog="/path/to/dir"`` to attach a
    filesystem catalog on open (its committed tables are published via
    ``sync_catalog()``).

        gtv.connect()                              # fresh in-memory engine
        gtv.connect(catalog="./gtvdata")           # attach a local catalog
    """
    if dsn not in (None, "", "gtv://local", "local:"):
        raise ValueError(
            f"pygtvdb-local has no network address; got dsn={dsn!r}. "
            "Use connect() or connect(catalog=...)."
        )
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"connect() got unexpected keyword argument(s): {unknown}")
    return Connection.open(catalog=catalog)
