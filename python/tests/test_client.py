"""The protocol client against a *v0* server (no version, no receipt field).

This is the compatibility case that matters in production today: the extension
repo's `gtv-server` is pinned to a core revision that predates protocol v1, so
every real deployment a user can reach right now answers v0.
"""

from __future__ import annotations

import unittest

from _fake_server import FakeServer

from gtvdb import GtvError, Session


class ClientAgainstV0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = FakeServer("v0").__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.server.__exit__(None, None, None)

    def test_falls_back_and_recovers_the_receipt(self):
        with Session.connect(self.server.addr) as session:
            self.assertEqual(session.protocol_version, 0, "v0 server detected at connect()")
            result = session.query("SELECT * FROM numbers")
            self.assertEqual(result.protocol_version, 0)
            self.assertEqual(result.rows, 2)
            self.assertEqual(result.receipt["snapshot_id"], 7)
            # v0 sends no tri-state field, so the proto3 default `FOUND` must not
            # leak: only the receipt can speak for a v0 peer.
            self.assertEqual(result.tri_state, "no_match")
            self.assertTrue(result.is_conclusive)

    def test_as_of_is_not_sent_to_a_v0_server(self):
        # v0 has no as-of field; pinning must not silently change the request.
        with Session.connect(self.server.addr) as session:
            result = session.at(as_of=42).query("SELECT * FROM numbers")
            self.assertIsNone(result.receipt["as_of"])

    def test_rows_without_a_receipt_prove_nothing(self):
        with Session.connect(self.server.addr) as session:
            result = session.query("SELECT bare FROM numbers")
            self.assertEqual(result.rows, 2)
            self.assertIsNone(result.receipt)
            self.assertIsNone(result.tri_state)
            self.assertFalse(result.is_conclusive)

    def test_server_error_raises_with_a_code(self):
        with Session.connect(self.server.addr) as session:
            with self.assertRaises(GtvError) as caught:
                session.query("SELECT * FROM boom")
            self.assertEqual(caught.exception.code, "INTERNAL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
