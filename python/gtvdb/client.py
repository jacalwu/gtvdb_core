"""Connect to a gtv engine from Python and run SQL against it.

This is the *session* layer: one gRPC channel, a negotiated protocol version,
pinned business time, and the engine's receipts. Most users want the DB-API
surface above it (`gtvdb.connect(...).cursor()`); drop to this layer when you
need the receipt, the tri-state, or the handshake.

gtvdb ships no PyO3 module: Python talks to the engine over the same gRPC
protocol the Rust clients use (`crates/gtv-proto/proto/gtvquery.proto`,
protocol v1). That is deliberate — one wire contract, one set of semantics,
no second ABI to keep in sync.

    from gtvdb import Session

    with Session.connect("127.0.0.1:50051") as session:
        result = session.query("SELECT * FROM numbers")
        print(result.table.to_pydict())
        print(result.receipt["tri_state"], result.receipt["freshness"])

# What "opening a session" means here

The server is stateless: there is no `CREATE SESSION`, no transaction to begin
and no handle to leak. A session is the client-side bundle of everything that
must stay fixed for a series of answers to be comparable:

    the channel            one TCP connection, reused by every query
    the negotiated version v1 (or v0 against an old server)
    the bearer token       sent as `authorization: Bearer <token>`
    the pinned as-of       business time every query is answered at
    the pinned snapshot    the snapshot the caller believes it is reading

`session.at(as_of=...)` returns a child session with the business time pinned,
so "open a session at T, ask five questions" reads the same as it does in kdb+.
Pinning matters: without `as_of` each query answers at the engine's current
snapshot, and two questions in a row can see different data.

# What a result carries

`Result.table` is a `pyarrow.Table`. `Result.receipt` is the engine's
`QueryReceipt` as a dict, and it is the reason to prefer this path over
shelling out: it tells you *whether the answer is complete* (`tri_state`),
which snapshot it came from, and how fresh the engine was
(`freshness.as_known_at`, `freshness.history_floor`). An empty table with
`tri_state == "no_match"` proves absence; an empty table with
`tri_state == "incomplete"` proves nothing. Do not treat them alike.

A v0 server sends no `receipt_json`; the receipt is still read out of the
Arrow schema metadata (`gtv.receipt.json`), so `Result.receipt` is populated
whenever the server stamped one. `Result.protocol_version == 0` means the peer
is old — there is no as-of pinning and no completeness signal beyond rows.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import grpc
import pyarrow as pa

from ._stubs import (
    ErrorCode,
    QueryRequest,
    QueryServiceStub,
    TriState,
    VersionRequest,
)

CLIENT_VERSION = "gtvdb 0.1.0"
PROTOCOL_VERSION = 1
DEFAULT_ADDR = "127.0.0.1:50051"
RECEIPT_METADATA_KEY = "gtv.receipt.json"

#: Sentinels so `as_of=None` ("no pin") differs from "not passed".
_UNSET: Any = object()


class GtvError(RuntimeError):
    """A query failed, the version is unsupported, or the transport broke."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}" if code else message)
        self.code = code
        self.message = message


@dataclass
class Result:
    """Rows plus everything the server could prove about them."""

    table: pa.Table
    receipt: Optional[dict]
    tri_state: Optional[str]
    protocol_version: int

    @property
    def rows(self) -> int:
        return self.table.num_rows

    @property
    def is_conclusive(self) -> bool:
        """`True` only when the engine finished: found or provably nothing."""
        return self.tri_state in ("found", "no_match")

    @property
    def freshness(self) -> Optional[dict]:
        return (self.receipt or {}).get("freshness")

    @property
    def as_known_at(self) -> Optional[int]:
        """Newest knowledge time reflected in these rows, if the server said."""
        freshness = self.freshness
        return None if freshness is None else freshness.get("as_known_at")

    @property
    def incomplete_reason(self) -> Optional[str]:
        incomplete = (self.receipt or {}).get("incomplete")
        return None if incomplete is None else incomplete.get("reason")

    @property
    def as_of_applied(self) -> bool:
        """Whether the rows really answer for the `as_of` that was requested.

        `False` covers both "no as_of requested" and "requested but nothing
        enforced it" — the latter is what `incomplete_reason ==
        'time_point_not_applied'` marks. A pin is a request, not a directive.
        """
        return (self.receipt or {}).get("as_of_binding") == "applied"

    @property
    def snapshot_verified(self) -> bool:
        """Whether the server confirmed it serves the snapshot that was pinned."""
        return (self.receipt or {}).get("snapshot_binding") == "applied"

    def __len__(self) -> int:
        return self.rows

    def __repr__(self) -> str:
        return (
            f"Result(rows={self.rows}, tri_state={self.tri_state!r}, "
            f"protocol_version={self.protocol_version}, "
            f"as_known_at={self.as_known_at})"
        )


class Session:
    """A connected client. Reuse it: one channel, many queries."""

    def __init__(
        self,
        channel: grpc.Channel,
        stub: QueryServiceStub,
        *,
        protocol_version: int,
        handshake: Optional[Any] = None,
        token: Optional[str] = None,
        as_of: Optional[int] = None,
        snapshot_id: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        owns_channel: bool = True,
        request_prefix: str = "py",
    ):
        self._channel = channel
        self._stub = stub
        self._token = token
        self._as_of = as_of
        self._snapshot_id = snapshot_id
        self._timeout_ms = timeout_ms
        self._owns_channel = owns_channel
        self._counter = itertools.count(1)
        self._prefix = request_prefix
        self.protocol_version = protocol_version
        self.handshake = handshake

    # -- construction ------------------------------------------------------

    @classmethod
    def connect(
        cls,
        addr: str = DEFAULT_ADDR,
        *,
        token: Optional[str] = None,
        ca_cert: Optional[str] = None,
        as_of: Optional[int] = None,
        snapshot_id: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        negotiate: bool = True,
    ) -> "Session":
        """Open a channel, negotiate the protocol version, return a session.

        `addr` is `host:port`; a `http://` / `https://` prefix is accepted and
        stripped, since gRPC chooses the scheme from the credentials.
        `token` is sent as a bearer token on every call. `ca_cert` is a path to
        a PEM file; without it the channel is plaintext (loopback / sidecar).

        Negotiation happens here on purpose: a version mismatch should fail at
        `connect()`, not on the first query.
        """
        token = token or os.environ.get("GTV_AUTH_TOKEN")
        ca_cert = ca_cert or os.environ.get("GTV_TLS_CA")
        host = addr.split("://", 1)[-1]
        pem = Path(ca_cert).read_bytes() if ca_cert else None

        options = [
            ("grpc.max_receive_message_length", -1),
            ("grpc.max_send_message_length", -1),
        ]
        if pem is not None:
            channel = grpc.secure_channel(
                host, grpc.ssl_channel_credentials(root_certificates=pem), options=options
            )
        else:
            channel = grpc.insecure_channel(host, options=options)

        stub = QueryServiceStub(channel)
        session = cls(
            channel,
            stub,
            protocol_version=PROTOCOL_VERSION,
            token=token,
            as_of=as_of,
            snapshot_id=snapshot_id,
            timeout_ms=timeout_ms,
        )
        if negotiate:
            session._negotiate()
        return session

    def _negotiate(self) -> None:
        """v1 handshake with a v0 fallback, so an old server is still usable."""
        try:
            reply = self._stub.Negotiate(
                VersionRequest(
                    protocol_version=PROTOCOL_VERSION,
                    client=CLIENT_VERSION,
                ),
                timeout=_seconds(self._timeout_ms),
            )
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                # Server predates v1: no handshake, no as-of, no receipt.
                self.protocol_version = 0
                self.handshake = None
                return
            raise GtvError(exc.code().name, exc.details() or "negotiate failed") from exc

        self.handshake = reply
        if reply.protocol_version == 0:
            raise GtvError(
                "UNSUPPORTED_VERSION",
                reply.error or f"server supports v{reply.min_supported}..v{reply.max_supported}",
            )
        self.protocol_version = reply.protocol_version

    # -- pinning -----------------------------------------------------------

    def at(self, as_of: Optional[int] = None, snapshot_id: Optional[int] = None) -> "Session":
        """Child session with business time and/or snapshot pinned.

        Shares the channel (no extra connection) and the auth token.
        """
        child = Session(
            self._channel,
            self._stub,
            protocol_version=self.protocol_version,
            handshake=self.handshake,
            token=self._token,
            as_of=self._as_of if as_of is None else as_of,
            snapshot_id=self._snapshot_id if snapshot_id is None else snapshot_id,
            timeout_ms=self._timeout_ms,
            owns_channel=False,
            request_prefix=self._prefix,
        )
        return child

    # -- queries -----------------------------------------------------------

    def query(
        self,
        sql: str,
        *,
        as_of: Any = _UNSET,
        snapshot_id: Any = _UNSET,
        timeout_ms: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Result:
        """Run one statement. Raises `GtvError`, except for a partial answer.

        A partial result comes back as a `Result` with
        `tri_state == "incomplete"` — the rows are real, the coverage is not.
        """
        as_of = self._as_of if as_of is _UNSET else as_of
        snapshot_id = self._snapshot_id if snapshot_id is _UNSET else snapshot_id
        request_id = request_id or f"{self._prefix}-{next(self._counter)}"
        timeout = _seconds(timeout_ms or self._timeout_ms)

        if self.protocol_version == 0:
            return self._query_legacy(sql, timeout)

        request = QueryRequest(
            sql=sql,
            protocol_version=PROTOCOL_VERSION,
            request_id=request_id,
            timeout_ms=0 if timeout_ms is None else int(timeout_ms),
        )
        if as_of is not None:
            request.as_of = int(as_of)
        if snapshot_id is not None:
            request.snapshot_id = int(snapshot_id)

        try:
            response = self._stub.ExecuteQuery(
                request, timeout=timeout, metadata=self._metadata()
            )
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                # Old server: drop to v0 for the rest of this session.
                self.protocol_version = 0
                return self._query_legacy(sql, timeout)
            raise GtvError(exc.code().name, exc.details() or "query failed") from exc

        return self._decode(response)

    def _query_legacy(self, sql: str, timeout: Optional[float]) -> Result:
        """v0: SQL in, Arrow IPC out. No receipt on the wire, maybe in metadata."""
        try:
            response = self._stub.Execute(
                QueryRequest(sql=sql), timeout=timeout, metadata=self._metadata()
            )
        except grpc.RpcError as exc:
            raise GtvError(exc.code().name, exc.details() or "query failed") from exc
        if response.error:
            raise GtvError("INTERNAL", response.error)
        return self._decode(response, legacy=True)

    def ping(self) -> dict:
        """Round-trip a trivial query; returns what the server said about itself."""
        self.query("SELECT 1")
        reply = self.handshake
        return {
            "protocol_version": self.protocol_version,
            "engine_version": getattr(reply, "engine_version", None),
            "min_supported": getattr(reply, "min_supported", None),
            "max_supported": getattr(reply, "max_supported", None),
        }

    # -- internals ---------------------------------------------------------

    def _metadata(self) -> Optional[list]:
        if not self._token:
            return None
        return [("authorization", f"Bearer {self._token}")]

    def _decode(self, response: Any, *, legacy: bool = False) -> Result:
        code = _error_code_name(response.code)
        table = _decode_arrow(response.arrow_ipc)
        receipt = _receipt(response.receipt_json, table)
        # A v0 server never computes a tri-state, so the field arrives as its
        # proto3 default (`FOUND`). Trusting it would turn "I don't know" into
        # "there were rows". Only the receipt can speak for a v0 peer.
        is_v0 = legacy or response.protocol_version == 0
        tri_state = (None if is_v0 else _tri_state_name(response.tri_state)) or _receipt_tri_state(
            receipt
        )

        if code not in (None, "OK", "RESULT_INCOMPLETE"):
            raise GtvError(code, response.error or "query failed")
        if code is None and response.error and table.num_rows == 0:
            raise GtvError("INTERNAL", response.error)

        return Result(
            table=table,
            receipt=receipt,
            tri_state=tri_state,
            protocol_version=0 if legacy else response.protocol_version,
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._owns_channel:
            self._channel.close()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Session(protocol_version={self.protocol_version}, "
            f"as_of={self._as_of}, snapshot_id={self._snapshot_id})"
        )


def connect(addr: str = DEFAULT_ADDR, **kwargs: Any) -> Session:
    """Shorthand for `Session.connect(addr, **kwargs)`."""
    return Session.connect(addr, **kwargs)


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------


def _seconds(timeout_ms: Optional[int]) -> Optional[float]:
    return None if not timeout_ms else timeout_ms / 1000.0


def _decode_arrow(payload: bytes) -> pa.Table:
    if not payload:
        return pa.table({})
    with pa.ipc.open_stream(pa.BufferReader(payload)) as reader:
        return reader.read_all()


def _receipt(receipt_json: str, table: pa.Table) -> Optional[dict]:
    """Prefer the v1 field; fall back to the Arrow schema metadata."""
    if receipt_json:
        return json.loads(receipt_json)
    metadata = table.schema.metadata or {}
    raw = metadata.get(RECEIPT_METADATA_KEY.encode())
    if raw is None:
        return None
    return json.loads(raw.decode())


def _receipt_tri_state(receipt: Optional[dict]) -> Optional[str]:
    return None if receipt is None else receipt.get("tri_state")


def _tri_state_name(value: int) -> Optional[str]:
    try:
        name = TriState.Name(value).lower()
    except ValueError:
        return None
    return None if name == "unspecified" else name


def _error_code_name(value: int) -> Optional[str]:
    try:
        return ErrorCode.Name(value)
    except ValueError:
        return str(value)
