"""PEP 249 Connection backed by an in-process gtv engine."""

from __future__ import annotations

from typing import Any, List, Optional

try:
    from . import _gtv_native  # normal: extension sits inside the package
except ImportError:  # building/developing with the .so next to the package dir
    import _gtv_native  # type: ignore[no-redef]

from .cursor import Cursor
from .errors import InterfaceError

_Native = _gtv_native.LocalConnection


class Connection:
    """An open, in-process gtv engine. Create with :func:`pygtvdb_local.connect`."""

    def __init__(self, native: Any, catalog: Optional[str] = None) -> None:
        self._native = native
        self._cursors: List[Cursor] = []
        self._closed = False
        if catalog is not None:
            self._native.attach_catalog(catalog)

    @classmethod
    def open(cls, *, catalog: Optional[str] = None) -> "Connection":
        try:
            native = _Native()
        except Exception as exc:  # engine/runtime init failure
            raise InterfaceError(f"cannot open local engine: {exc}") from exc
        return cls(native, catalog=catalog)

    # -- cursor factory ----------------------------------------------------

    def cursor(self) -> Cursor:
        self._ensure_open()
        cur = Cursor(self)
        self._cursors.append(cur)
        return cur

    # -- engine-specific: local persistence -------------------------------

    def register_parquet(self, path: str, name: str) -> None:
        """Register a Parquet file as a SQL table in this engine session."""
        self._ensure_open()
        self._native.register_parquet(path, name)

    def attach_catalog(self, root: str) -> None:
        """Attach the filesystem catalog rooted at ``root``."""
        self._ensure_open()
        self._native.attach_catalog(root)

    def sync_catalog(self) -> int:
        """Publish committed catalog tables into SQL; return their count."""
        self._ensure_open()
        return self._native.sync_catalog()

    # -- PEP 249 transaction surface --------------------------------------

    def commit(self) -> None:
        """No-op: reads are consistent snapshots; writes go through catalog publish."""
        self._ensure_open()

    def rollback(self) -> None:
        """No-op: the engine has no in-session transaction to roll back."""
        self._ensure_open()

    def close(self) -> None:
        self._closed = True
        for cur in self._cursors:
            cur._force_close()
        self._cursors.clear()

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise InterfaceError("connection is closed")

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "Connection":
        self._ensure_open()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
