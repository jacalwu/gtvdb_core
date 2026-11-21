from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OK: _ClassVar[ErrorCode]
    INVALID_ARGUMENT: _ClassVar[ErrorCode]
    UNSUPPORTED_VERSION: _ClassVar[ErrorCode]
    RESULT_INCOMPLETE: _ClassVar[ErrorCode]
    SNAPSHOT_MISMATCH: _ClassVar[ErrorCode]
    CANCELLED: _ClassVar[ErrorCode]
    DEADLINE_EXCEEDED: _ClassVar[ErrorCode]
    INTERNAL: _ClassVar[ErrorCode]

class TriState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FOUND: _ClassVar[TriState]
    NO_MATCH: _ClassVar[TriState]
    INCOMPLETE: _ClassVar[TriState]
    UNSPECIFIED: _ClassVar[TriState]
OK: ErrorCode
INVALID_ARGUMENT: ErrorCode
UNSUPPORTED_VERSION: ErrorCode
RESULT_INCOMPLETE: ErrorCode
SNAPSHOT_MISMATCH: ErrorCode
CANCELLED: ErrorCode
DEADLINE_EXCEEDED: ErrorCode
INTERNAL: ErrorCode
FOUND: TriState
NO_MATCH: TriState
INCOMPLETE: TriState
UNSPECIFIED: TriState

class VersionRequest(_message.Message):
    __slots__ = ("protocol_version", "client")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    CLIENT_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    client: str
    def __init__(self, protocol_version: _Optional[int] = ..., client: _Optional[str] = ...) -> None: ...

class VersionResponse(_message.Message):
    __slots__ = ("protocol_version", "min_supported", "max_supported", "engine_version", "error")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    MIN_SUPPORTED_FIELD_NUMBER: _ClassVar[int]
    MAX_SUPPORTED_FIELD_NUMBER: _ClassVar[int]
    ENGINE_VERSION_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    min_supported: int
    max_supported: int
    engine_version: str
    error: str
    def __init__(self, protocol_version: _Optional[int] = ..., min_supported: _Optional[int] = ..., max_supported: _Optional[int] = ..., engine_version: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class QueryRequest(_message.Message):
    __slots__ = ("sql", "protocol_version", "as_of", "snapshot_id", "request_id", "timeout_ms")
    SQL_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    AS_OF_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    sql: str
    protocol_version: int
    as_of: int
    snapshot_id: int
    request_id: str
    timeout_ms: int
    def __init__(self, sql: _Optional[str] = ..., protocol_version: _Optional[int] = ..., as_of: _Optional[int] = ..., snapshot_id: _Optional[int] = ..., request_id: _Optional[str] = ..., timeout_ms: _Optional[int] = ...) -> None: ...

class QueryResponse(_message.Message):
    __slots__ = ("arrow_ipc", "error", "protocol_version", "code", "tri_state", "receipt_json")
    ARROW_IPC_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    TRI_STATE_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_JSON_FIELD_NUMBER: _ClassVar[int]
    arrow_ipc: bytes
    error: str
    protocol_version: int
    code: ErrorCode
    tri_state: TriState
    receipt_json: str
    def __init__(self, arrow_ipc: _Optional[bytes] = ..., error: _Optional[str] = ..., protocol_version: _Optional[int] = ..., code: _Optional[_Union[ErrorCode, str]] = ..., tri_state: _Optional[_Union[TriState, str]] = ..., receipt_json: _Optional[str] = ...) -> None: ...
