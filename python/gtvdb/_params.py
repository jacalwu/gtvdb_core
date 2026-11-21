"""Client-side parameter rendering (`paramstyle = "qmark"`).

Why client-side: protocol v1 carries SQL as a string with no bind slots
(`QueryRequest` has no parameter array). Rather than pretend, this module
renders values into literals *carefully* and says so:

* `?` placeholders are replaced only outside string literals, quoted
  identifiers and comments, so `WHERE s = 'a?b'` is untouched;
* values are rendered as literals of the right type (quoted text with `''`
  doubled, `X'..'` for bytes, `TIMESTAMP '..'` for datetimes) — never by
  `str()`-ing them into the SQL;
* `None` becomes `NULL`, which is the only rendering that can never be wrong.

This is the same compromise `sqlite3` makes when it interpolates, and it is
safe against injection because values never reach the SQL text unquoted. It is
*not* as strong as a real bind protocol: a parameter cannot be used in places
where a literal is illegal (an identifier, a `LIMIT` clause in some planners),
and the engine re-plans on every call. Bound parameters are a protocol v2 item;
until then, `paramstyle` tells the truth about what happens.
"""

from __future__ import annotations

import datetime as _dt
import decimal
from typing import Any, Mapping, Sequence

from .errors import ProgrammingError, DataError

_NAMED_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_NAMED_BODY = _NAMED_START | set("0123456789")


def render(sql: str, params: Any) -> str:
    """Substitute `params` into `sql`. `None`/empty means "nothing to do"."""
    if params is None:
        return sql
    if isinstance(params, Mapping):
        return _render_named(sql, params)
    if isinstance(params, (str, bytes)):
        raise ProgrammingError("parameters must be a sequence or a mapping, not a string")
    return _render_qmark(sql, list(params))


def _render_qmark(sql: str, values: Sequence[Any]) -> str:
    out: list[str] = []
    used = 0
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in ("'", '"'):
            chunk, i = _quoted(sql, i)
            out.append(chunk)
            continue
        if sql.startswith("--", i):
            chunk, i = _to_eol(sql, i)
            out.append(chunk)
            continue
        if sql.startswith("/*", i):
            chunk, i = _block_comment(sql, i)
            out.append(chunk)
            continue
        if ch == "?":
            if used >= len(values):
                raise ProgrammingError(
                    f"not enough parameters: query has more than {len(values)} placeholder(s)"
                )
            out.append(literal(values[used]))
            used += 1
            i += 1
            continue
        out.append(ch)
        i += 1

    if used != len(values):
        raise ProgrammingError(
            f"parameter count mismatch: {len(values)} given, {used} placeholder(s)"
        )
    return "".join(out)


def _render_named(sql: str, values: Mapping[str, Any]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in ("'", '"'):
            chunk, i = _quoted(sql, i)
            out.append(chunk)
            continue
        if sql.startswith("--", i):
            chunk, i = _to_eol(sql, i)
            out.append(chunk)
            continue
        if sql.startswith("/*", i):
            chunk, i = _block_comment(sql, i)
            out.append(chunk)
            continue
        # `::` is a cast, not two placeholders: consume it whole so the second
        # colon cannot be re-read as `:bigint`.
        if sql.startswith("::", i):
            out.append("::")
            i += 2
            continue
        if ch == ":" and i + 1 < n and sql[i + 1] in _NAMED_START:
            j = i + 1
            while j < n and sql[j] in _NAMED_BODY:
                j += 1
            name = sql[i + 1 : j]
            if name not in values:
                raise ProgrammingError(f"missing value for parameter :{name}")
            out.append(literal(values[name]))
            seen.add(name)
            i = j
            continue
        out.append(ch)
        i += 1

    unused = set(values) - seen
    if unused:
        raise ProgrammingError(
            "unused parameter(s): " + ", ".join(sorted(unused))
        )
    return "".join(out)


# ---------------------------------------------------------------------------
# Literal rendering
# ---------------------------------------------------------------------------


def literal(value: Any) -> str:
    """Render one Python value as a SQL literal of the matching type."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):  # before int: bool is an int subclass
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise DataError(f"non-finite float cannot be a literal: {value!r}")
        return repr(value)
    if isinstance(value, decimal.Decimal):
        if not value.is_finite():
            raise DataError(f"non-finite decimal cannot be a literal: {value!r}")
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "X'" + bytes(value).hex() + "'"
    if isinstance(value, _dt.datetime):
        return "TIMESTAMP '" + value.isoformat(sep=" ") + "'"
    if isinstance(value, _dt.date):
        return "DATE '" + value.isoformat() + "'"
    if isinstance(value, _dt.time):
        return "TIME '" + value.isoformat() + "'"
    raise DataError(
        f"unsupported parameter type {type(value).__name__}; "
        "pass a scalar, str, bytes, date, datetime, Decimal, or None"
    )


# ---------------------------------------------------------------------------
# Scanner helpers: skip regions where `?` / `:name` are just text
# ---------------------------------------------------------------------------


def _quoted(sql: str, i: int) -> tuple[str, int]:
    """Read a '...' or "..." region starting at `i`, honouring doubled quotes."""
    quote = sql[i]
    j = i + 1
    n = len(sql)
    while j < n:
        if sql[j] == quote:
            if j + 1 < n and sql[j + 1] == quote:
                j += 2
                continue
            j += 1
            break
        j += 1
    return sql[i:j], j


def _to_eol(sql: str, i: int) -> tuple[str, int]:
    j = sql.find("\n", i)
    return (sql[i:], len(sql)) if j < 0 else (sql[i:j], j)


def _block_comment(sql: str, i: int) -> tuple[str, int]:
    j = sql.find("*/", i + 2)
    return (sql[i:], len(sql)) if j < 0 else (sql[i : j + 2], j + 2)
