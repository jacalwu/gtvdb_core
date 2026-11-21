"""PEP 249 exception hierarchy, mapped from the engine's stable error codes.

The wire carries a numeric `ErrorCode` (see `gtvquery.proto`) precisely so
clients never branch on message text. This module is the one place that turns
those codes into the exceptions a DB-API consumer already knows how to catch.

    try:
        cursor.execute(...)
    except gtvdb.ProgrammingError:      # bad SQL / bad argument
        ...
    except gtvdb.OperationalError:      # engine refused, timed out, went away
        ...
"""

from __future__ import annotations

from typing import Optional


class Warning(Exception):  # noqa: A001 - PEP 249 mandates this name
    """Raised for important warnings. Unused today; kept for API completeness."""


class Error(Exception):
    """Base of every error this driver raises."""

    #: Engine code, when the failure came back over the wire ("INVALID_ARGUMENT").
    code: Optional[str] = None

    def __init__(self, message: str = "", *, code: Optional[str] = None):
        super().__init__(f"{code}: {message}" if code and message else (message or code or ""))
        self.code = code
        self.message = message


class InterfaceError(Error):
    """The driver was used wrongly (closed cursor, missing result set)."""


class DatabaseError(Error):
    """Anything that went wrong on the engine side."""


class DataError(DatabaseError):
    """A value could not be processed (bad parameter type, overflow)."""


class OperationalError(DatabaseError):
    """The engine refused the query, ran out of budget, or the link broke."""


class IntegrityError(DatabaseError):
    """A constraint was violated. The engine is read-mostly, so this is rare."""


class InternalError(DatabaseError):
    """The engine reported an internal inconsistency."""


class ProgrammingError(DatabaseError):
    """Bad SQL, unknown table, wrong number of parameters."""


class NotSupportedError(DatabaseError):
    """A DB-API feature this engine cannot honour (writes, transactions)."""


#: Wire code -> PEP 249 class. Anything unmapped is an `OperationalError`,
#: because "the engine said no" is the safest reading of an unknown refusal.
_CODE_MAP = {
    "INVALID_ARGUMENT": ProgrammingError,
    "UNSUPPORTED_VERSION": NotSupportedError,
    "SNAPSHOT_MISMATCH": OperationalError,
    "RESULT_INCOMPLETE": OperationalError,
    "CANCELLED": OperationalError,
    "DEADLINE_EXCEEDED": OperationalError,
    "INTERNAL": InternalError,
}


def error_for(code: Optional[str], message: str) -> DatabaseError:
    """Build the right PEP 249 exception for a wire code."""
    return _CODE_MAP.get(code or "", OperationalError)(message, code=code)
