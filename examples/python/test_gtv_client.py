"""Client tests that need no Rust process: a fake server speaks v0 on purpose.

    python examples/python/test_gtv_client.py

The interesting case is the *old* server: no `Negotiate`, no `ExecuteQuery`, no
`receipt_json`. The client must fall back to `Execute` and still recover the
receipt, because a v1 server also stamps it into the Arrow schema metadata.
"""

from __future__ import annotations

import json
import threading
import unittest
from concurrent import futures

import grpc
import pyarrow as pa

from gtv_client import GtvError, connect
from gtvproto import QueryRequest, QueryResponse
from gtvproto import gtvquery_pb2_grpc as pb_grpc

RECEIPT = {"tri_state": "no_match", "as_of": None, "snapshot_id": 7}


def _ipc(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


class V0Server(pb_grpc.QueryServiceServicer):
    """Only the deprecated RPC exists; the v1 ones are unimplemented."""

    def Execute(self, request: QueryRequest, context):  # noqa: N802 (gRPC name)
        if "boom" in request.sql:
            return QueryResponse(arrow_ipc=b"", error="no such table: boom")
        table = pa.table({"n": [1, 2]})
        if "bare" not in request.sql:
            table = table.replace_schema_metadata(
                {b"gtv.receipt.json": json.dumps(RECEIPT).encode()}
            )
        return QueryResponse(arrow_ipc=_ipc(table))


class ClientAgainstV0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        pb_grpc.add_QueryServiceServicer_to_server(V0Server(), cls.server)
        port = cls.server.add_insecure_port("127.0.0.1:0")
        cls.server.start()
        cls.addr = f"127.0.0.1:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.stop(grace=None)

    def test_falls_back_and_recovers_the_receipt(self):
        with connect(self.addr) as session:
            self.assertEqual(session.protocol_version, 0, "v0 server detected at connect()")
            result = session.query("SELECT * FROM numbers")
            self.assertEqual(result.protocol_version, 0)
            self.assertEqual(result.rows, 2)
            self.assertEqual(result.receipt["snapshot_id"], 7)
            # v0 sends no tri-state field, so the default `FOUND` must not leak:
            # the receipt is the only thing that can speak for a v0 peer.
            self.assertEqual(result.tri_state, "no_match")
            self.assertTrue(result.is_conclusive)

    def test_rows_without_a_receipt_prove_nothing(self):
        with connect(self.addr) as session:
            result = session.query("SELECT bare FROM numbers")
            self.assertEqual(result.rows, 2)
            self.assertIsNone(result.receipt)
            self.assertIsNone(result.tri_state)
            self.assertFalse(result.is_conclusive)

    def test_as_of_is_not_sent_to_a_v0_server(self):
        # v0 has no as-of field; pinning must not silently change the request.
        with connect(self.addr) as session:
            result = session.at(as_of=42).query("SELECT * FROM numbers")
            self.assertIsNone(result.receipt["as_of"])

    def test_server_error_raises_with_a_code(self):
        with connect(self.addr) as session:
            with self.assertRaises(GtvError) as caught:
                session.query("SELECT * FROM boom")
            self.assertEqual(caught.exception.code, "INTERNAL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
