"""Mode dispatch — pure logic, no engine required.

`gtvdb.connect()` decides from the DSN alone whether to open a gRPC channel or
the in-process engine. That decision is testable on any machine, including one
where the wheel carries no native artefact, so the contract stays verified in CI
jobs that never build the extension.
"""

from __future__ import annotations

import unittest

import gtvdb


class LocalDsnRecognition(unittest.TestCase):
    def test_local_dsn_means_in_process_without_a_store(self):
        self.assertIsNone(gtvdb._local_store_from_dsn("gtv://local"))

    def test_local_dsn_can_carry_a_store(self):
        self.assertEqual(
            gtvdb._local_store_from_dsn("gtv://local?local_store=C:/data"),
            "C:/data",
        )

    def test_catalog_is_accepted_as_an_alias(self):
        self.assertEqual(
            gtvdb._local_store_from_dsn("gtv://local?catalog=/tmp/x"), "/tmp/x"
        )

    def test_bare_local_is_accepted(self):
        # `connect("local")` reads naturally, and the DSN parser already
        # tolerates a missing scheme for remote addresses.
        self.assertIsNone(gtvdb._local_store_from_dsn("local"))

    def test_remote_dsn_is_not_local(self):
        for dsn in (
            "gtv://127.0.0.1:50051",
            "gtv://engine.internal:50061",
            "10.0.0.5:50051",
            "gtv://host:50051?as_of=1",
        ):
            self.assertIs(gtvdb._local_store_from_dsn(dsn), gtvdb._NETWORK, dsn)

    def test_default_is_remote(self):
        self.assertIs(gtvdb._local_store_from_dsn(None), gtvdb._NETWORK)
        self.assertIs(gtvdb._local_store_from_dsn(""), gtvdb._NETWORK)

    def test_localhost_is_a_network_address_not_the_engine(self):
        # The tempting shortcut here is a bug: `localhost` is a host, and
        # treating it as local mode would hide a misconfigured server.
        self.assertIs(
            gtvdb._local_store_from_dsn("gtv://localhost:50051"), gtvdb._NETWORK
        )


class LocalModeRejectsRemoteOptions(unittest.TestCase):
    def test_remote_only_options_raise(self):
        for kwargs in (
            {"as_of": 1},
            {"snapshot_id": 7},
            {"token": "t"},
            {"ca_cert": "x.pem"},
            {"timeout_ms": 10},
        ):
            with self.assertRaises(TypeError) as caught:
                gtvdb.connect("gtv://local", **kwargs)
            self.assertIn("local mode does not accept", str(caught.exception))
            self.assertIn(next(iter(kwargs)), str(caught.exception))

    def test_unknown_keyword_raises(self):
        with self.assertRaises(TypeError):
            gtvdb.connect("gtv://local", databse="oops")

    def test_explicit_store_selects_local_mode_without_a_dsn(self):
        # `connect(local_store=...)` must not fall through to the remote
        # default, so this has to raise before any engine work happens.
        with self.assertRaises(TypeError):
            gtvdb.connect(local_store="C:/data", token="t")


if __name__ == "__main__":
    unittest.main(verbosity=2)
