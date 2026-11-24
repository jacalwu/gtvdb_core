"""The in-process engine, behind the same PEP 249 surface as remote mode.

Reached through `gtvdb.connect("gtv://local")`. Importing this package directly
is not the supported path — the dispatcher in `gtvdb` owns the choice between
local and remote, so that callers never have to know which one they got.
"""

from __future__ import annotations

from typing import Optional

from .connection import Connection

__all__ = ["connect", "Connection"]


def connect(
    local_store: Optional[str] = None,
    *,
    catalog: Optional[str] = None,
) -> Connection:
    """Open the in-process engine.

    Args:
        local_store: Directory holding the engine's data. Its committed tables
            are published immediately, so opening an existing store makes its
            tables queryable without a separate call — that is what "open my
            database" is expected to mean.
        catalog: Alias for `local_store`, kept because the standalone
            `pygtvdb-local` package used that name.

    Returns:
        A `Connection` whose cursor speaks the same dialect as the remote one.

    Raises:
        InterfaceError: The native extension is missing, or the engine failed
            to start (message carries the underlying reason).
    """
    store = local_store if local_store is not None else catalog
    con = Connection.open()
    if store is not None:
        con.attach_catalog(store)
        con.sync_catalog()
    return con
