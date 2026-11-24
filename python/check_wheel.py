#!/usr/bin/env python
"""Install the built wheel into a throwaway directory and query through it.

A wheel can build cleanly and still be unusable: the native extension is a file
that `build_native.py` copies into the package, and a wheel produced before
that copy (or with the wrong suffix) installs into a module that fails the
moment `connect()` touches the engine.

Importing from inside the repository hides the problem completely — the source
package shadows the installed one — so this installs to a temporary directory
and runs a subprocess from a different working directory with only that
directory on `PYTHONPATH`.

    python python/check_wheel.py                    # newest wheel in dist/
    python python/check_wheel.py --wheel path.whl
    python python/check_wheel.py --fresh-venv       # release gate (see below)

`--fresh-venv` installs into a brand-new virtual environment instead of a
directory, so only the dependencies the wheel *declares* are present. The fast
path deliberately skips that: it runs with this interpreter's site-packages, so
a package missing from `Requires-Dist` still imports here — which is how
`import gtvdb` once shipped broken, with `protobuf` used by the committed stubs
but absent from the metadata. That check costs a `pip install`, so a release
pays it and a local loop does not.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"

#: Runs in a subprocess: proves the installed package can open an engine and
#: answer a query, not merely that its `__init__` is importable. Local mode is
#: the interesting half — a wheel without the native artefact imports fine and
#: fails exactly here.
PROBE = """
import sys
import gtvdb

assert gtvdb.__file__.startswith(sys.argv[1]), f"imported {gtvdb.__file__}, not the wheel"

con = gtvdb.connect("gtv://local")
cur = con.cursor()
cur.execute("SELECT n FROM range(1, 4) t(n)")
assert cur.fetchall() == [(1,), (2,), (3,)], "query through the wheel failed"
assert [d[1].__name__ for d in cur.description] == ["int"], "type codes lost"
con.close()

# A network DSN must still dispatch to the remote path without an engine probe.
assert gtvdb._local_store_from_dsn("gtv://127.0.0.1:50051") is gtvdb._NETWORK
print(f"wheel ok: {gtvdb.__version__} from {gtvdb.__file__}")
"""


def _extension_suffixes() -> tuple[str, ...]:
    if sys.platform == "win32":
        return (".pyd",)
    return (".so", ".dylib")


def find_wheel(explicit: str | None) -> Path:
    """The wheel to test: explicit path, or the only one in ``dist/``."""
    if explicit:
        wheel = Path(explicit).resolve()
        if not wheel.exists():
            raise SystemExit(f"error: no such wheel: {wheel}")
        return wheel

    wheels = sorted(DIST.glob("*.whl"))
    if not wheels:
        raise SystemExit(
            f"error: no wheel in {DIST}; run "
            "`python python/build_native.py --wheel` first"
        )
    if len(wheels) > 1:
        names = ", ".join(w.name for w in wheels)
        raise SystemExit(
            f"error: more than one wheel ({names}); pass --wheel to choose one"
        )
    return wheels[0]


def check_contents(wheel: Path) -> None:
    """The wheel must carry the compiled extension, or it cannot work."""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()

    extension = [
        name for name in names
        if name.startswith("gtvdb/_local/") and name.endswith(_extension_suffixes())
    ]
    if not extension:
        raise SystemExit(
            f"error: {wheel.name} contains no compiled extension under "
            f"gtvdb/_local/ (expected one of {_extension_suffixes()}); "
            "did you build before running `build_native.py`?"
        )

    if any("__pycache__" in name for name in names):
        raise SystemExit(f"error: {wheel.name} carries __pycache__")

    # A wheel tagged for the exact interpreter defeats the abi3 build: pyo3 is
    # configured for cp39-abi3, so the file name must say so.
    if "cp39-abi3" not in wheel.name:
        print(
            f"warning: {wheel.name} is not tagged cp39-abi3; "
            "pip will only install it on one Python version",
            file=sys.stderr,
        )

    # A `linux_x86_64` wheel installs fine on Linux and PyPI still refuses it.
    # This is not hypothetical: `check_release.py --keep-dist` used to rebuild
    # into python/dist and silently replace the manylinux wheel with one tagged
    # `linux_x86_64`, so the artifact looked fine until the upload step.
    if sys.platform.startswith("linux") and "manylinux_" not in wheel.name:
        raise SystemExit(
            f"error: {wheel.name} has no manylinux platform tag; PyPI rejects "
            "plain linux_x86_64 wheels. Check that nothing rebuilt the wheel "
            "after `build_native.py --wheel` (a release gate must not pass "
            "--keep-dist)."
        )

    print(f"contents ok: {extension[0]} present, {len(names)} files")


def install_and_probe(wheel: Path, *, fresh_env: bool = False) -> None:
    if fresh_env:
        _probe_in_fresh_venv(wheel)
        return

    target = Path(tempfile.mkdtemp(prefix="gtvpy-wheel-"))
    print(f"installing into {target}")

    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--force-reinstall",
            "--target",
            str(target),
            str(wheel),
        ],
        check=False,
    )
    if install.returncode != 0:
        raise SystemExit("error: pip could not install the wheel")

    env = dict(os.environ)
    # Prepend, do not replace: pyarrow lives in the interpreter's own
    # site-packages and is deliberately not bundled into the wheel.
    env["PYTHONPATH"] = str(target) + os.pathsep + env.get("PYTHONPATH", "")

    probe = subprocess.run(
        [sys.executable, "-c", PROBE, str(target)],
        cwd=tempfile.gettempdir(),
        env=env,
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(
            "error: the installed wheel could not answer a query "
            "(see the traceback above)"
        )


def _probe_in_fresh_venv(wheel: Path) -> None:
    """Install the wheel into a new venv, resolving only its declared deps.

    A wheel's metadata is a promise about what the machine needs. This is the
    only check that can break that promise: everything else runs next to
    packages that happen to be installed already.
    """
    root = Path(tempfile.mkdtemp(prefix="gtvpy-venv-")) / "v"
    print(f"creating a clean venv at {root}")
    subprocess.run([sys.executable, "-m", "venv", str(root)], check=True)
    python = (
        root / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else root / "bin" / "python"
    )

    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "--no-cache-dir", str(wheel)],
        check=False,
    )
    if install.returncode != 0:
        raise SystemExit(
            "error: pip could not install the wheel with its declared dependencies"
        )

    probe = subprocess.run(
        [str(python), "-c", PROBE, str(root)],
        cwd=tempfile.gettempdir(),
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(
            "error: the wheel does not work in a clean environment — its "
            "Requires-Dist is incomplete (see the traceback above)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that a built pygtvdb_local wheel installs and works.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--wheel", help="wheel path (default: the one in dist/)")
    parser.add_argument(
        "--fresh-venv",
        action="store_true",
        help="install into a new venv so only declared dependencies are present "
        "(slower; this is the check a release should pass)",
    )
    args = parser.parse_args()

    wheel = find_wheel(args.wheel)
    print(f"checking {wheel.name} ({wheel.stat().st_size / (1024 * 1024):.1f} MiB)")
    check_contents(wheel)
    install_and_probe(wheel, fresh_env=args.fresh_venv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
