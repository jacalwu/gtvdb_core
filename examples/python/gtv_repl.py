"""Drive the gtv REPL from Python — a thin client, not a binding.

gtvdb_core ships no PyO3 module (see ROADMAP: M8/M9 are shelved), so there are
two supported ways to reach the engine from Python: this file, which drives a
local `gtv` subprocess over the REPL's stdin/stdout, and `gtv_client.py`, which
speaks protocol v1 over gRPC to a running server (as-of pinning, receipts,
remote host). Use this one for local files and REPL dot-commands; use
`gtv_client.py` when the engine is already running somewhere else.

The subprocess route runs every SQL statement and dot command the REPL itself
can run, and nothing more — graph traversal (`neighbors` / `khop` / `motif`)
needs a `TemporalCSR` registered from Rust, so it is out of reach here exactly
as it is out of reach in the REPL.

How a reply is delimited: the REPL has no prompt when stdin is a pipe, and no
end-of-result marker, so after each statement we append

    SELECT '<uuid>' AS __gtv_eof;

and read until that uuid shows up. A per-call uuid means a statement whose own
output contains the string can never be mistaken for the terminator.

Usage:
    from gtv_repl import Gtv

    with Gtv() as db:
        print(db.sql("SELECT * FROM gtv_segments();"))
        rows = db.table(db.sql("SELECT * FROM quantum_state_summary('...')"))

Run `python examples/python/hello_python.py` for a worked example.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import unicodedata
import uuid
from pathlib import Path
from typing import Iterable, Optional

REPO = Path(__file__).resolve().parents[2]
SENTINEL_COLUMN = "__gtv_eof"


class GtvError(RuntimeError):
    """A statement failed, or the process died, or the reply timed out."""


def find_binary() -> Path:
    """Locate the gtv executable: $GTV_BIN, then target/release, then debug."""
    override = os.environ.get("GTV_BIN")
    if override:
        path = Path(override)
        if not path.exists():
            raise GtvError(f"GTV_BIN points at a missing file: {path}")
        return path
    for rel in ("target/release/gtv", "target/debug/gtv"):
        base = REPO / rel
        for candidate in (base, base.with_suffix(".exe")):
            if candidate.exists():
                return candidate
    raise GtvError(
        "gtv executable not found. Build it first:  cargo build -p gtv-cli\n"
        "(or point GTV_BIN at an existing binary)"
    )


class Gtv:
    """One gtv subprocess; state (registered tables, segments, params) persists.

    Constructing an instance spawns nothing — `start()` (or the `with` block)
    does. `close()` sends `.quit` and waits; it is safe to call twice.
    """

    def __init__(
        self,
        binary: Optional[Path] = None,
        cwd: Optional[Path] = None,
        timeout: float = 120.0,
        env: Optional[dict] = None,
    ) -> None:
        self.binary = Path(binary) if binary else find_binary()
        self.cwd = Path(cwd) if cwd else REPO
        self.timeout = timeout
        self._env = env
        self._proc: Optional[subprocess.Popen] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._lock = threading.Lock()
        self.banner = ""

    # ---- lifecycle -----------------------------------------------------

    def __enter__(self) -> "Gtv":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> "Gtv":
        if self._proc is not None:
            return self
        env = dict(os.environ if self._env is None else self._env)
        # The REPL only sets RUST_LOG when the caller did not; keep output quiet
        # by default so `error:` lines mean a real error.
        env.setdefault("RUST_LOG", "error")
        self._proc = subprocess.Popen(
            [str(self.binary)],
            cwd=str(self.cwd),
            stdin=subprocess.PIPE,
            # stderr is merged: the REPL prints `error: ...` there, and a merged
            # stream keeps it in order with the failing statement's output.
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._pump, daemon=True).start()
        # The banner has no terminator either; drain it with a sentinel call.
        self.banner = self._command("").strip()
        return self

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.stdin.write(".quit\n")
                proc.stdin.flush()
                proc.wait(timeout=10)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            pass
        finally:
            for stream in (proc.stdin, proc.stdout):
                try:
                    stream.close()
                except Exception:
                    pass
            if proc.poll() is None:
                proc.kill()

    def _pump(self) -> None:
        """Read the child's stdout on one thread and hand lines to a queue."""
        assert self._proc is not None and self._proc.stdout is not None
        stream = self._proc.stdout
        while True:
            line = stream.readline()
            if line == "":
                self._lines.put(None)
                return
            self._lines.put(line.rstrip("\r\n"))

    # ---- the protocol --------------------------------------------------

    def _command(self, text: str) -> str:
        with self._lock:
            if self._proc is None:
                raise GtvError("call start() first (or use the context manager)")
            assert self._proc.stdin is not None
            token = "eof_" + uuid.uuid4().hex
            payload = f"{text}\nSELECT '{token}' AS {SENTINEL_COLUMN};\n"
            try:
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise GtvError(f"gtv exited before accepting the statement: {exc}") from exc

            lines: list[str] = []
            while True:
                try:
                    line = self._lines.get(timeout=self.timeout)
                except queue.Empty:
                    raise GtvError(
                        f"no reply within {self.timeout}s; the statement may still be running"
                    ) from None
                if line is None:
                    raise GtvError(
                        f"gtv exited mid-statement (is the SQL malformed?)\n{snippet(lines)}"
                    )
                if token in line:
                    break
                lines.append(line)

            # The sentinel table is 5 lines (border, header, border, row,
            # border); we stopped on the row, so consume the closing border too
            # or it would open the *next* call's output.
            try:
                closing = self._lines.get(timeout=self.timeout)
            except queue.Empty:
                closing = None
            if closing is not None and not closing.lstrip().startswith("+"):
                # Not a border after all — keep it for the next read.
                self._lines.put(closing)

            # Drop the sentinel's own chrome: border, header, border.
            if len(lines) >= 3 and lines[-1].lstrip().startswith("+") and SENTINEL_COLUMN in lines[-2]:
                del lines[-3:]
            while lines and not lines[-1].strip():
                lines.pop()
            return "\n".join(lines)

    # ---- public surface ------------------------------------------------

    def sql(self, statement: str, *, check: bool = True) -> str:
        """Run SQL (or a dot command) and return its output as text.

        `check=True` raises [`GtvError`] if the reply contains an `error:` line —
        which is how the REPL reports a failed statement. Pass `check=False` to
        inspect the raw text instead (used by the error-handling example).
        """
        text = self._command(statement.strip())
        if check:
            errors = [ln for ln in text.splitlines() if ln.startswith("error: ")]
            if errors:
                raise GtvError("\n".join(errors))
        return text

    def try_sql(self, statement: str) -> tuple[bool, str]:
        """`sql` without the exception: returns (ok, text)."""
        text = self._command(statement.strip())
        ok = not any(ln.startswith("error: ") for ln in text.splitlines())
        return ok, text

    def table(self, text: str) -> list[dict[str, str]]:
        """Parse the last ASCII table in `text` into a list of row dicts.

        The REPL renders results with arrow's pretty printer, so a table is a
        header line plus rows, both wrapped in `+---+` borders. Cells keep their
        printed text (no type inference); a NULL shows up as an empty string.

        Columns are located from the border's `+` positions rather than by
        splitting on `|`: a value may contain a pipe (the quantum gate `|000>`
        does), and a CJK value occupies two terminal cells per character, so the
        slice has to be measured in display width, not in Python `len`.
        """
        lines = text.splitlines()
        blocks: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("+") or stripped.startswith("|"):
                current.append(stripped)
            elif current:
                blocks.append(current)
                current = []
        if current:
            blocks.append(current)
        if not blocks:
            raise GtvError(f"no table in the reply:\n{text}")

        block = blocks[-1]
        rows = [ln for ln in block if ln.startswith("|")]
        if len(rows) < 2:
            raise GtvError(f"table has no data rows:\n{text}")
        border = next((ln for ln in block if ln.startswith("+")), None)
        if border is None:
            raise GtvError(f"table has no border to measure:\n{text}")
        cuts = [i for i, ch in enumerate(border) if ch == "+"]

        def cells(line: str) -> list[str]:
            out: list[str] = []
            for start, end in zip(cuts, cuts[1:]):
                out.append(_slice_cells(line, start + 1, end).strip())
            return out

        header = cells(rows[0])
        out = []
        for row in rows[1:]:
            out.append(dict(zip(header, cells(row))))
        return out

    def query(self, statement: str) -> list[dict[str, str]]:
        """`sql` + `table` in one call: run SQL, parse the last table."""
        return self.table(self.sql(statement))

    def dot(self, command: str) -> str:
        """Run a dot command (`.tables`, `.csv`, `.segment`, `.accel`, ...)."""
        if not command.startswith("."):
            raise GtvError(f"not a dot command: {command!r}")
        return self.sql(command)


def _display_width(ch: str) -> int:
    """Terminal cells a character occupies: 0 combining, 2 wide, else 1."""
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _slice_cells(line: str, start: int, end: int) -> str:
    """Take display columns [start, end) out of an aligned table line."""
    if line.isascii():
        return line[start:end]
    out: list[str] = []
    col = 0
    for ch in line:
        width = _display_width(ch)
        if col >= start and col + width <= end:
            out.append(ch)
        col += width
        if col >= end:
            break
    return "".join(out)


def snippet(lines: Iterable[str], limit: int = 20) -> str:
    lines = list(lines)
    return "\n".join(lines[-limit:])
