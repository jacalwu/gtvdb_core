"""Compatibility shim: the client now lives in the installable package.

The protocol client moved to `python/gtvdb/client.py` so it ships as
`pip install gtvdb` instead of being copy-pasted into example scripts. This
file stays so `from gtv_client import connect` keeps working from the examples
directory, with or without an install.

`connect()` here is the **session**-level connect (returns `Session`), which is
what it always meant. The DB-API `connect()` — the one that returns a
`Connection` with `cursor()` — is `gtvdb.connect`.

    import gtvdb
    con = gtvdb.connect("gtv://127.0.0.1:50051")
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "python"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from gtvdb import (  # noqa: E402  (path set above)
    DEFAULT_ADDR,
    Connection,
    Cursor,
    GtvError,
    Result,
    Session,
)


def connect(addr: str = DEFAULT_ADDR, **kwargs):
    """Session-level connect: `connect(addr)` -> `Session`."""
    return Session.connect(addr, **kwargs)


__all__ = [
    "DEFAULT_ADDR",
    "Connection",
    "Cursor",
    "GtvError",
    "Result",
    "Session",
    "connect",
]
