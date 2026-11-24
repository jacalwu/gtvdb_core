"""PEP 249 conformance tests for `gtv://local` — the in-process engine.

These run against the real embedded engine (no server, no fake): the extension
is the thing being tested, so stubbing it would test nothing. When the compiled
`_gtv_native` module is absent the whole module skips rather than fails, so the
suite is safe to run from a bare source checkout and in CI before the build
step. Dispatch logic — which needs no engine — lives in `test_dispatch.py`.

    python -m unittest discover -s python/tests -t python/tests -v

Covers what the driver claims in its README: the DB-API module surface, type
codes identical to remote mode, connection lifecycle, the fetch family, both
parameter styles, local persistence (Parquet / store), and connection isolation.
"""

from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

# The tests sit one level below the package root and must import without an
# editable install, so put `python_local/` on the path explicitly.
_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import gtvdb as gtv  # noqa: E402


def _native_available() -> bool:
    try:
        from gtvdb._local import _gtv_native  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


NATIVE = _native_available()


def setUpModule() -> None:
    if not NATIVE:
        raise unittest.SkipTest(
            "_gtv_native is not built; run `python python/build_native.py`"
        )


class ModuleSurface(unittest.TestCase):
    def test_pep249_attributes(self):
        self.assertEqual(gtv.apilevel, "2.0")
        self.assertEqual(gtv.paramstyle, "qmark")
        self.assertIn(gtv.threadsafety, (0, 1, 2))

    def test_exception_hierarchy(self):
        self.assertTrue(issubclass(gtv.ProgrammingError, gtv.DatabaseError))
        self.assertTrue(issubclass(gtv.DatabaseError, gtv.Error))
        self.assertTrue(issubclass(gtv.DataError, gtv.DatabaseError))
        self.assertTrue(issubclass(gtv.OperationalError, gtv.DatabaseError))

    def test_type_objects_are_exported_and_match_the_network_client(self):
        """Regression: `_type_code` used to fold every number into NUMBER, so a
        wheel promised `int` but reported `float` in `description`."""
        self.assertIs(gtv.STRING, str)
        self.assertIs(gtv.NUMBER, float)
        self.assertIs(gtv.INTEGER, int)
        self.assertIs(gtv.BOOLEAN, bool)
        self.assertIs(gtv.BINARY, bytes)
        self.assertIs(gtv.ROWID, int)
        self.assertIs(gtv.DATETIME, dt.datetime)
        self.assertIs(gtv.DATE, dt.date)
        self.assertIs(gtv.TIME, dt.time)


class ConnectionLifecycle(unittest.TestCase):
    def test_connect_opens_an_in_memory_engine(self):
        with gtv.connect("gtv://local") as con:
            self.assertFalse(con.closed)
            self.assertIsNotNone(con.cursor())

    def test_explicit_local_store_selects_local_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            with gtv.connect("gtv://local", local_store=tmp) as con:
                # An empty store must open and publish nothing, not fail.
                self.assertEqual(con.sync_catalog(), 0)

    def test_store_from_dsn_query_is_honoured(self):
        with tempfile.TemporaryDirectory() as tmp:
            with gtv.connect(f"gtv://local?local_store={tmp}") as con:
                self.assertIsNotNone(con.cursor())

    def test_local_store_alone_selects_local_mode(self):
        # No DSN at all: an explicit store must not fall through to 127.0.0.1.
        with tempfile.TemporaryDirectory() as tmp:
            with gtv.connect(local_store=tmp) as con:
                self.assertIsNotNone(con.cursor())

    def test_close_is_idempotent_and_blocks_new_cursors(self):
        con = gtv.connect("gtv://local")
        cur = con.cursor()
        con.close()
        con.close()
        self.assertTrue(con.closed)
        self.assertTrue(cur.closed, "cursors die with their connection")
        with self.assertRaises(gtv.InterfaceError):
            con.cursor()

    def test_commit_and_rollback_are_no_ops(self):
        with gtv.connect("gtv://local") as con:
            con.commit()
            con.rollback()
            self.assertTrue(con.cursor() is not None)


class CursorQueries(unittest.TestCase):
    def con(self):
        return gtv.connect("gtv://local")

    def test_execute_description_and_fetchall(self):
        with self.con() as con:
            cur = con.cursor()
            self.assertIs(cur.execute("SELECT n FROM range(1, 4) t(n)"), cur)
            self.assertEqual([d[0] for d in cur.description], ["n"])
            self.assertEqual(cur.rowcount, 3)
            self.assertEqual(cur.fetchall(), [(1,), (2,), (3,)])
            self.assertEqual(cur.fetchall(), [], "a drained cursor stays drained")
            self.assertIsNone(cur.fetchone())

    def test_fetchone_and_fetchmany_walk_the_same_result(self):
        with self.con() as con:
            cur = con.cursor().execute("SELECT n FROM range(1, 4) t(n)")
            self.assertEqual(cur.fetchone(), (1,))
            self.assertEqual(cur.fetchmany(1), [(2,)])
            cur.arraysize = 5
            self.assertEqual(cur.fetchmany(), [(3,)])

    def test_iteration(self):
        with self.con() as con:
            cur = con.cursor().execute("SELECT n FROM range(1, 4) t(n)")
            self.assertEqual([row[0] for row in cur], [1, 2, 3])

    def test_empty_result_has_zero_rows_but_a_schema(self):
        with self.con() as con:
            cur = con.cursor().execute("SELECT n FROM range(1, 4) t(n) WHERE n > 99")
            self.assertEqual(cur.fetchall(), [])
            self.assertEqual(cur.rowcount, 0)
            self.assertEqual([d[0] for d in cur.description], ["n"],
                             "an empty result still knows its columns")

    def test_fetch_before_execute_is_an_interface_error(self):
        with self.con() as con:
            with self.assertRaises(gtv.InterfaceError):
                con.cursor().fetchall()

    def test_closed_cursor_raises(self):
        with self.con() as con:
            cur = con.cursor()
            cur.close()
            self.assertTrue(cur.closed)
            with self.assertRaises(gtv.InterfaceError):
                cur.execute("SELECT 1")

    def test_arrow_table_and_schema_preserve_types(self):
        with self.con() as con:
            cur = con.cursor().execute("SELECT n, n * 2 AS twice FROM range(1, 3) t(n)")
            table = cur.arrow_table()
            self.assertIsInstance(table, pa.Table)
            self.assertEqual(table.num_rows, 2)
            # pyarrow DataType instances are compared structurally, not by
            # identity — `pa.int64() is pa.int64()` is False.
            self.assertEqual(cur.schema.field(0).type, pa.int64())
            self.assertEqual(table.column("twice").to_pylist(), [2, 4])

    def test_executemany_runs_once_per_row(self):
        with self.con() as con:
            cur = con.cursor()
            cur.executemany("SELECT n FROM range(1, 10) t(n) WHERE n = ?", [(1,), (2,), (3,)])
            self.assertEqual(cur.rowcount, 1, "rowcount reflects the last statement")
            self.assertEqual(cur.fetchall(), [(3,)])


class TypeCodes(unittest.TestCase):
    """Every column of a result must report the same type object `gtvdb` does."""

    def test_arrow_types_map_to_pep249_objects(self):
        with gtv.connect("gtv://local") as con:
            cur = con.cursor().execute(
                "SELECT 7 AS i, 1.5 AS f, true AS b, 'x' AS s, "
                "TIMESTAMP '2026-01-02 03:04:05' AS ts, DATE '2026-01-02' AS d"
            )
            codes = {d[0]: d[1] for d in cur.description}

        self.assertIs(codes["i"], gtv.INTEGER)
        self.assertIs(codes["f"], gtv.NUMBER)
        self.assertIs(codes["b"], gtv.BOOLEAN)
        self.assertIs(codes["s"], gtv.STRING)
        self.assertIs(codes["ts"], gtv.DATETIME)
        self.assertIs(codes["d"], gtv.DATE)

    def test_values_come_back_as_python_objects(self):
        with gtv.connect("gtv://local") as con:
            row = con.cursor().execute(
                "SELECT 7 AS i, 1.5 AS f, true AS b, 'x' AS s, DATE '2026-01-02' AS d"
            ).fetchone()

        self.assertIs(type(row[0]), int)
        self.assertIs(type(row[1]), float)
        self.assertIs(type(row[2]), bool)
        self.assertIs(type(row[3]), str)
        self.assertEqual(row[4], dt.date(2026, 1, 2))


class Parameters(unittest.TestCase):
    def test_qmark_renders_typed_literals(self):
        with gtv.connect("gtv://local") as con:
            row = con.cursor().execute(
                "SELECT ? AS i, ? AS s, ? AS nul, ? AS b", (7, "O'Brien", None, True)
            ).fetchone()
        self.assertEqual(row, (7, "O'Brien", None, True))

    def test_named_style(self):
        with gtv.connect("gtv://local") as con:
            row = con.cursor().execute("SELECT :n AS v", {"n": 5}).fetchone()
        self.assertEqual(row, (5,))

    def test_placeholder_count_is_checked(self):
        with gtv.connect("gtv://local") as con:
            cur = con.cursor()
            with self.assertRaises(gtv.ProgrammingError):
                cur.execute("SELECT ?, ?", (1,))
            with self.assertRaises(gtv.ProgrammingError):
                cur.execute("SELECT ?", (1, 2))
            with self.assertRaises(gtv.ProgrammingError):
                cur.execute("SELECT :a", {"b": 1})

    def test_unsupported_type_fails_before_the_engine(self):
        with gtv.connect("gtv://local") as con:
            with self.assertRaises(gtv.DataError):
                con.cursor().execute("SELECT ?", (object(),))


class LocalPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_register_parquet_makes_a_queryable_table(self):
        path = self.root / "nums.parquet"
        pq.write_table(pa.table({"n": [1, 2, 3], "s": ["a", "b", "c"]}), path)

        with gtv.connect("gtv://local") as con:
            con.register_parquet(str(path), "nums")
            cur = con.cursor()
            cur.execute("SELECT sum(n) AS total, count(*) AS cnt FROM nums")
            self.assertEqual(cur.fetchone(), (6, 3))
            cur.execute("SELECT s FROM nums WHERE n = ?", (2,))
            self.assertEqual(cur.fetchall(), [("b",)])

    def test_attach_empty_catalog_publishes_nothing(self):
        with gtv.connect("gtv://local", local_store=str(self.root)) as con:
            self.assertEqual(con.sync_catalog(), 0)

    def test_catalog_can_be_attached_after_connect(self):
        with gtv.connect("gtv://local") as con:
            con.attach_catalog(str(self.root))
            self.assertEqual(con.sync_catalog(), 0)


class Isolation(unittest.TestCase):
    def test_connections_do_not_share_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nums.parquet"
            pq.write_table(pa.table({"n": [1]}), path)

            with gtv.connect("gtv://local") as first, gtv.connect("gtv://local") as second:
                first.register_parquet(str(path), "nums")
                self.assertEqual(first.cursor().execute("SELECT count(*) FROM nums").fetchone(), (1,))
                with self.assertRaises(gtv.ProgrammingError):
                    second.cursor().execute("SELECT * FROM nums")


class EngineSurface(unittest.TestCase):
    def test_engine_table_functions_exist(self):
        with gtv.connect("gtv://local") as con:
            for probe in ("SELECT * FROM gtv_segments()", "SELECT * FROM gtv_extensions()"):
                cur = con.cursor().execute(probe)
                self.assertEqual(cur.fetchall(), [], f"{probe} should answer with no rows")


class Errors(unittest.TestCase):
    def test_unknown_table_maps_to_programming_error(self):
        with gtv.connect("gtv://local") as con:
            with self.assertRaises(gtv.ProgrammingError) as caught:
                con.cursor().execute("SELECT * FROM no_such_table")
        self.assertIn("not found", str(caught.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
