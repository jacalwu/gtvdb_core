"""PEP 249 type objects, and the Arrow type → type-code mapping.

Both engines report `description[i][1]` through this one table. They used to
carry private copies, and the copies drifted: the in-process cursor folded every
integer into `NUMBER`/float, so `description[i][1] is gtvdb.INTEGER` held on a
remote connection and silently failed on a local one. One table, one answer.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pyarrow as pa

#: PEP 249 requires these so a caller can compare
#: `cursor.description[i][1] is gtvdb.STRING`. Plain builtins, as sqlite3 and
#: psycopg2 do. `INTEGER` and `BOOLEAN` are extensions: DBAPI20 folds every
#: number into NUMBER, but an int column reported as `float` misleads anyone
#: building an INSERT or a JSON payload out of `description`.
STRING = str
BINARY = bytes
NUMBER = float
INTEGER = int
BOOLEAN = bool
ROWID = int
DATETIME = _dt.datetime
DATE = _dt.date
TIME = _dt.time


def type_code(arrow_type: pa.DataType) -> Any:
    """Map an Arrow column type to its PEP 249 type object.

    Args:
        arrow_type: A field type taken from the result's Arrow schema.

    Returns:
        One of the module-level type objects; the Arrow type's string form when
        nothing matches, because a slightly vague description beats an
        exception raised while describing a successful query.

    Notes:
        Order matters: booleans and integers must be tested before the generic
        floating/decimal branch, and temporal types before the string fallback,
        or an int column would come back as `float`.
    """
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return STRING
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
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
    return str(arrow_type)
