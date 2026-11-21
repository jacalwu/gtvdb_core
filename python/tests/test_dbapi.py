"""DB-API 2.0 conformance tests against an in-process v1 server.

Covers the parts of the PEP 249 contract this driver claims: module attributes,
connection lifecycle, cursor `description` / `rowcount` / fetch family, the
`qmark` and named parameter styles, error mapping, and the two extras that make
this driver worth using (`as_of` pinning and tri-state).

    python -m unittest discover -s python/tests -v
"""

from __future__ import annotations

import datetime as dt
import decimal
import unittest

from _fake_server import FakeServer

import gtvdb


class ModuleAttributes(unittest.TestCase):
    def test_pep249_module_surface(self):
        self.assertEqual(gtvdb.apilevel, "2.0")
        self.assertEqual(gtvdb.paramstyle, "qmark")
        self.assertIn(gtvdb.threadsafety, (0, 1, 2))
        for name in ("Warning", "Error", "InterfaceError", "DatabaseError"):
            self.assertTrue(issubclass(getattr(gtvdb, name), Exception))

    def test_exception_hierarchy(self):
        self.assertTrue(issubclass(gtvdb.ProgrammingError, gtvdb.DatabaseError))
        self.assertTrue(issubclass(gtvdb.DatabaseError, gtvdb.Error))
        self.assertTrue(issubclass(gtvdb.DataError, gtvdb.DatabaseError))

    def test_type_objects_exist_for_description(self):
        self.assertIs(gtvdb.STRING, str)
        self.assertIs(gtvdb.NUMBER, float)
        self.assertIs(gtvdb.INTEGER, int)
        self.assertIs(gtvdb.DATETIME, dt.datetime)


class ConnectionLifecycle(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer("v1").__enter__()
        self.add = self.server.addr

    def tearDown(self):
        self.server.__exit__(None, None, None)

    def test_connect_by_dsn_and_kwargs(self):
        for con in (gtvdb.connect(self.server.dsn), gtvdb.connect(host="127.0.0.1", port=int(self.add.split(":")[1]))):
            self.assertEqual(con.protocol_version, 1)
            self.assertEqual(con.handshake.engine_version, "fake")
            con.close()

    def test_commit_and_rollback_are_no_ops(self):
        with gtvdb.connect(self.server.dsn) as con:
            con.commit()
            con.rollback()
            self.assertTrue(con.autocommit)

    def test_close_is_idempotent_and_blocks_new_cursors(self):
        con = gtvdb.connect(self.server.dsn)
        con.close()
        con.close()
        self.assertTrue(con.closed)
        with self.assertRaises(gtvdb.InterfaceError):
            con.cursor()

    def test_bad_dsn_scheme_is_rejected(self):
        with self.assertRaises(gtvdb.InterfaceError):
            gtvdb.connect("postgres://localhost:5432/db")

    def test_unknown_keyword_is_a_type_error(self):
        with self.assertRaises(TypeError):
            gtvdb.connect(self.server.dsn, databse="oops")


class CursorQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = FakeServer("v1").__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.server.__exit__(None, None, None)

    def con(self):
        return gtvdb.connect(self.server.dsn)

    def test_execute_description_and_fetchall(self):
        with self.con() as con:
            cur = con.cursor()
            self.assertEqual(cur.execute("SELECT * FROM numbers"), cur, "execute returns self")
            self.assertEqual([d[0] for d in cur.description], ["n", "label"])
            self.assertIs(cur.description[0][1], gtvdb.INTEGER)
            self.assertIs(cur.description[1][1], gtvdb.STRING)
            self.assertEqual(cur.rowcount, 2)
            self.assertEqual(cur.fetchall(), [(1, "aa"), (2, "aa")])
            self.assertEqual(cur.fetchall(), [], "a drained cursor stays drained")
            self.assertEqual(cur.fetchone(), None)

    def test_fetchmany_and_iteration(self):
        with self.con() as con:
            cur = con.cursor().execute("SELECT * FROM numbers")
            self.assertEqual(len(cur.fetchmany()), 1, "arraysize defaults to 1")
            cur.arraysize = 5
            self.assertEqual([row[0] for row in cur], [2])

    def test_cursor_after_close_raises(self):
        with self.con() as con:
            cur = con.cursor()
            cur.close()
            with self.assertRaises(gtvdb.InterfaceError):
                cur.execute("SELECT 1")
            with self.assertRaises(gtvdb.InterfaceError):
                con.cursor().fetchall()

    def test_read_table_keeps_arrow(self):
        import pyarrow as pa

        with self.con() as con:
            table = con.read_table("SELECT * FROM numbers")
            self.assertIsInstance(table, pa.Table)
            self.assertEqual(table.num_rows, 2)

    def test_empty_result_proves_absence(self):
        with self.con() as con:
            cur = con.cursor().execute("SELECT * FROM empty")
            self.assertEqual(cur.fetchall(), [])
            self.assertEqual(cur.tri_state, "no_match")
            self.assertTrue(cur.is_conclusive)

    def test_receipt_carries_snapshot_and_freshness(self):
        with self.con() as con:
            cur = con.cursor().execute("SELECT * FROM numbers")
            self.assertEqual(cur.receipt["snapshot_id"], None)
            self.assertEqual(cur.receipt["freshness"]["clock"], "segment_system_ts")

    def test_error_maps_to_programming_error(self):
        with self.con() as con:
            with self.assertRaises(gtvdb.ProgrammingError) as caught:
                con.cursor().execute("SELECT * FROM boom")
            self.assertEqual(caught.exception.code, "INVALID_ARGUMENT")


class Parameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = FakeServer("v1").__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.server.__exit__(None, None, None)

    def last_sql(self) -> str:
        return self.server.requests[-1].sql

    def test_qmark_renders_typed_literals(self):
        with gtvdb.connect(self.server.dsn) as con:
            con.cursor().execute(
                "SELECT * FROM t WHERE a = ? AND b = ? AND c = ? AND d = ?",
                (7, "O'Brien", None, dt.datetime(2026, 11, 21, 12, 0, tzinfo=dt.timezone.utc)),
            )
        self.assertIn("a = 7", self.last_sql())
        self.assertIn("b = 'O''Brien'", self.last_sql(), "single quotes are doubled, not escaped away")
        self.assertIn("c = NULL", self.last_sql())
        self.assertIn("d = TIMESTAMP '2026-11-21 12:00:00+00:00'", self.last_sql())

    def test_qmark_ignores_placeholders_inside_literals_and_comments(self):
        with gtvdb.connect(self.server.dsn) as con:
            con.cursor().execute("SELECT '?', ? -- and ? here\n/* ? */", (1,))
        self.assertIn("SELECT '?', 1", self.last_sql())
        self.assertIn("-- and ? here", self.last_sql())

    def test_named_style_and_cast_is_not_a_parameter(self):
        with gtvdb.connect(self.server.dsn) as con:
            con.cursor().execute("SELECT :n::bigint, :s", {"n": 3, "s": "x"})
        self.assertIn("SELECT 3::bigint, 'x'", self.last_sql())

    def test_counts_are_checked(self):
        with gtvdb.connect(self.server.dsn) as con:
            with self.assertRaises(gtvdb.ProgrammingError):
                con.cursor().execute("SELECT ?, ?", (1,))
            with self.assertRaises(gtvdb.ProgrammingError):
                con.cursor().execute("SELECT ?", (1, 2))
            with self.assertRaises(gtvdb.ProgrammingError):
                con.cursor().execute("SELECT :a", {"b": 1})

    def test_unsupported_type_fails_before_the_engine(self):
        with gtvdb.connect(self.server.dsn) as con:
            with self.assertRaises(gtvdb.DataError):
                con.cursor().execute("SELECT ?", (object(),))

    def test_decimal_and_bool_and_bytes(self):
        with gtvdb.connect(self.server.dsn) as con:
            con.cursor().execute(
                "SELECT ?, ?, ?", (decimal.Decimal("1.25"), True, b"\x01\x02")
            )
        self.assertIn("SELECT 1.25, TRUE, X'0102'", self.last_sql())

    def test_executemany_runs_once_per_row(self):
        with gtvdb.connect(self.server.dsn) as con:
            before = len(self.server.requests)
            con.cursor().executemany("SELECT * FROM t WHERE n = ?", [(1,), (2,), (3,)])
        self.assertEqual(len(self.server.requests) - before, 3)


class Pinning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = FakeServer("v1").__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.server.__exit__(None, None, None)

    def test_as_of_travels_on_the_wire(self):
        with gtvdb.connect(self.server.dsn, as_of=1_700_000_000_000_000_000) as con:
            cur = con.cursor().execute("SELECT * FROM numbers")
            self.assertEqual(self.server.requests[-1].as_of, 1_700_000_000_000_000_000)
            self.assertEqual(cur.receipt["as_of"], 1_700_000_000_000_000_000)

    def test_pin_reuses_the_connection(self):
        with gtvdb.connect(self.server.dsn) as con:
            pinned = con.pin(as_of=123, snapshot_id=9)
            pinned.cursor().execute("SELECT * FROM numbers")
            self.assertEqual(self.server.requests[-1].as_of, 123)
            self.assertEqual(self.server.requests[-1].snapshot_id, 9)
            self.assertIsNone(con.as_of, "the parent keeps its own pins")

    def test_dsn_query_parameters_are_honoured(self):
        con = gtvdb.connect(f"gtv://{self.server.addr}?as_of=42&snapshot_id=7")
        con.cursor().execute("SELECT * FROM numbers")
        self.assertEqual(self.server.requests[-1].as_of, 42)
        self.assertEqual(self.server.requests[-1].snapshot_id, 7)
        con.close()


class PandasIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = FakeServer("v1").__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.server.__exit__(None, None, None)

    def test_read_sql(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - optional dependency
            self.skipTest("pandas not installed")

        with gtvdb.connect(self.server.dsn) as con:
            frame = pd.read_sql("SELECT * FROM numbers", con)
        self.assertEqual(list(frame.columns), ["n", "label"])
        self.assertEqual(len(frame), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
