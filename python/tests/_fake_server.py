"""In-process fake gtv servers, so the driver's tests need no Rust process.

Two flavours, because the driver has to behave differently against each:

* `v0` — only the deprecated `Execute` RPC exists (this is what the extension
  repo's `gtv-server` still is: no version, no `as_of`, no `receipt_json`,
  receipt only in Arrow metadata);
* `v1` — `Negotiate` + `ExecuteQuery`, with a real receipt echoed back.

The fakes record every request, so parameter rendering can be asserted on the
SQL text that actually reached the "server".
"""

from __future__ import annotations

import json
from concurrent import futures
from typing import Any, Optional

import grpc
import pyarrow as pa

from gtvdb._stubs import (
    ErrorCode,
    QueryRequest,
    QueryResponse,
    QueryServiceStub,
    TriState,
    VersionRequest,
    VersionResponse,
)
from gtvdb._stubs import gtvquery_pb2_grpc as pb_grpc

__all__ = ["FakeServer", "ipc", "RECEIPT"]


def ipc(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _optional(request: QueryRequest, field: str) -> Optional[int]:
    """proto3 `optional` reads as 0 when unset; the real server sends null."""
    return getattr(request, field) if request.HasField(field) else None


def _receipt(request: QueryRequest, tri_state: str) -> dict:
    return {
        "protocol_version": 1,
        "engine_version": "fake",
        "query_id": request.request_id or "anon",
        "snapshot_id": _optional(request, "snapshot_id"),
        "as_of": _optional(request, "as_of"),
        "plan_hash": "0" * 16,
        "tri_state": tri_state,
        "incomplete": None,
        "stats": {"rows": 2},
        "freshness": {
            "as_known_at": 1_700_000_000_000_000_000,
            "clock": "segment_system_ts",
            "history_floor": None,
            "business_max": None,
        },
    }


RECEIPT = {"tri_state": "no_match", "snapshot_id": 7, "as_of": None}


class _Base(pb_grpc.QueryServiceServicer):
    def __init__(self) -> None:
        self.requests: list[QueryRequest] = []

    def _table(self, sql: str) -> pa.Table:
        # "empty" selects nothing: lets a test assert tri-state handling.
        rows = [] if "empty" in sql else [1, 2]
        return pa.table({"n": rows, "label": ["a" * len(rows) for _ in rows]})

    def _fail(self, sql: str) -> bool:
        return "boom" in sql


class V0Server(_Base):
    """A server from before protocol v1: one unary RPC, no receipt field."""

    def Execute(self, request: QueryRequest, context):  # noqa: N802 (gRPC name)
        self.requests.append(request)
        if self._fail(request.sql):
            return QueryResponse(arrow_ipc=b"", error="no such table: boom")
        table = self._table(request.sql)
        if "bare" not in request.sql:
            table = table.replace_schema_metadata(
                {b"gtv.receipt.json": json.dumps(RECEIPT).encode()}
            )
        return QueryResponse(arrow_ipc=ipc(table))


class V1Server(_Base):
    """Protocol v1: version handshake, as-of echo, receipt on the wire."""

    def Negotiate(self, request: VersionRequest, context):  # noqa: N802
        return VersionResponse(
            protocol_version=1,
            min_supported=1,
            max_supported=1,
            engine_version="fake",
            error="",
        )

    def ExecuteQuery(self, request: QueryRequest, context):  # noqa: N802
        self.requests.append(request)
        if self._fail(request.sql):
            return QueryResponse(
                arrow_ipc=b"",
                error="no such table: boom",
                protocol_version=1,
                code=ErrorCode.INVALID_ARGUMENT,
                tri_state=TriState.UNSPECIFIED,
                receipt_json="",
            )
        table = self._table(request.sql)
        empty = table.num_rows == 0
        receipt = _receipt(request, "no_match" if empty else "found")
        table = table.replace_schema_metadata(
            {b"gtv.receipt.json": json.dumps(receipt).encode()}
        )
        return QueryResponse(
            arrow_ipc=ipc(table),
            error="",
            protocol_version=1,
            code=ErrorCode.OK,
            tri_state=TriState.NO_MATCH if empty else TriState.FOUND,
            receipt_json=json.dumps(receipt),
        )

    # `Execute` stays unimplemented on purpose: a v1 client must not need it.


class FakeServer:
    """`with FakeServer("v1") as srv: gtvdb.connect(srv.dsn)`."""

    def __init__(self, kind: str = "v1"):
        self.kind = kind
        self.servicer: _Base = V0Server() if kind == "v0" else V1Server()
        self._server: Optional[grpc.Server] = None
        self.addr = ""

    def __enter__(self) -> "FakeServer":
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        pb_grpc.add_QueryServiceServicer_to_server(self.servicer, self._server)
        port = self._server.add_insecure_port("127.0.0.1:0")
        self._server.start()
        self.addr = f"127.0.0.1:{port}"
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._server is not None:
            self._server.stop(grace=None)

    @property
    def dsn(self) -> str:
        return f"gtv://{self.addr}"

    @property
    def requests(self) -> list[QueryRequest]:
        return self.servicer.requests

    def stub(self) -> QueryServiceStub:
        return QueryServiceStub(grpc.insecure_channel(self.addr))
