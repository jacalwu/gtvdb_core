#!/usr/bin/env python
"""Pre-flight gate for publishing the `pygtvdb` driver to PyPI.

    python python/check_release.py                # all checks, builds into a temp dir
    python python/check_release.py --keep-dist    # leave the artefacts in python/dist
    python python/check_release.py --offline      # skip the PyPI name/version query

Every check here is one that has actually broken a release somewhere: a licence
copy that drifted from the repository's, a committed stub older than the
`.proto`, a version that PyPI already has, a wheel missing `py.typed`.

Uploading is deliberately *not* part of this script — a check you can run on a
laptop should never be able to push a release. See `python/PUBLISHING.md`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LICENSE_SRC = REPO / "LICENSE.md"
LICENSE_COPY = HERE / "LICENSE.md"
STUBS_SCRIPT = HERE / "gen_stubs.py"
NAME = "pygtvdb"  # PyPI distribution name; the Python import stays `gtvdb`

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}PASS{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")


def skip(msg: str) -> None:
    print(f"  {YELLOW}SKIP{RESET}  {msg}")


def version() -> str:
    """Read the version without importing the package (grpcio may be absent)."""
    ns: dict[str, object] = {}
    exec((HERE / "gtvdb" / "_version.py").read_text(encoding="utf-8"), ns)  # noqa: S102
    return str(ns["__version__"])


def check_license() -> bool:
    """`python/LICENSE.md` must be a copy of the repository's, byte for byte.

    It exists only because PEP 639 forbids `license-files` globs containing
    `..`, so the build root cannot reference the file one level up.
    """
    if not LICENSE_COPY.exists():
        fail(f"{LICENSE_COPY.name} is missing (copy {LICENSE_SRC})")
        return False
    if LICENSE_SRC.read_bytes() != LICENSE_COPY.read_bytes():
        fail(f"python/{LICENSE_COPY.name} differs from the repository's LICENSE.md")
        return False
    ok("licence copy matches the repository")
    return True


def check_stubs() -> bool:
    r = subprocess.run(
        [sys.executable, str(STUBS_SCRIPT), "--check"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        fail("committed gRPC stubs are stale; run `python python/gen_stubs.py`")
        print(DIM + (r.stdout + r.stderr).strip() + RESET)
        return False
    ok("gRPC stubs match crates/gtv-proto/proto/gtvquery.proto")
    return True


def check_version() -> bool:
    try:
        from packaging.version import Version
    except ModuleNotFoundError:  # packaging ships with setuptools, but be explicit
        skip("packaging not installed; cannot validate the version string")
        return True
    try:
        Version(version())
    except Exception as e:  # noqa: BLE001
        fail(f"version {version()!r} is not PEP 440: {e}")
        return False
    ok(f"version {version()} is PEP 440")
    return True


def check_runtime_version() -> bool:
    try:
        import gtvdb  # noqa: PLC0415
    except ModuleNotFoundError as e:
        skip(f"gtvdb not importable here ({e.name}); runtime version unchecked")
        return True
    if gtvdb.__version__ != version():
        fail(f"gtvdb.__version__={gtvdb.__version__} != _version.py={version()}")
        return False
    ok("runtime __version__ matches the build metadata")
    return True


def check_tests() -> bool:
    r = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "tests"],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or [""]
    if r.returncode != 0:
        fail(f"test suite failed: {tail[0]}")
        print(DIM + (r.stderr or r.stdout)[-2000:] + RESET)
        return False
    ok(f"test suite green ({tail[0]})")
    return True


def build(dist: Path) -> tuple[Path, Path] | None:
    for tool in ("build", "twine"):
        try:
            __import__(tool)
        except ModuleNotFoundError:
            fail(f"`{tool}` is not installed: pip install build twine")
            return None
    if dist.exists():
        shutil.rmtree(dist)
    r = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(dist)],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        fail("build failed")
        print(DIM + (r.stdout + r.stderr)[-3000:] + RESET)
        return None
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if not wheels or not sdists:
        fail(f"expected one wheel and one sdist, got {[p.name for p in dist.iterdir()]}")
        return None
    r = subprocess.run(
        [sys.executable, "-m", "twine", "check", "--strict", *map(str, wheels + sdists)],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        fail("twine check failed (metadata PyPI would reject)")
        print(DIM + (r.stdout + r.stderr)[-2000:] + RESET)
        return None
    ok(f"built and validated {wheels[0].name} + {sdists[0].name}")
    return wheels[0], sdists[0]


def check_artefacts(wheel: Path, sdist: Path, *, require_engine: bool = True) -> bool:
    """The wheel must carry what a reader of the source expects to be there."""
    required = {
        "gtvdb/py.typed",
        "gtvdb/_version.py",
        "gtvdb/_stubs/gtvquery_pb2.py",
        "gtvdb/_stubs/gtvquery_pb2_grpc.py",
        "gtvdb/_stubs/gtvquery_pb2.pyi",
        f"{NAME}-{version()}.dist-info/licenses/LICENSE.md",
    }
    names = set(zipfile.ZipFile(wheel).namelist())
    missing = sorted(required - names)
    if missing:
        fail(f"wheel is missing: {', '.join(missing)}")
        return False

    # The in-process engine is the reason this distribution is a platform wheel
    # at all. Shipping one without it would advertise `connect("gtv://local")`
    # and then fail at that call, so a release must say so loudly rather than
    # quietly publish a remote-only build under a two-mode description.
    engine = [
        n for n in names
        if n.startswith("gtvdb/_local/") and n.endswith((".pyd", ".so", ".dylib"))
    ]
    if engine:
        ok(f"wheel carries the in-process engine ({engine[0].rsplit('/', 1)[-1]})")
    elif require_engine:
        fail(
            "wheel has no in-process engine: run `python python/build_native.py` "
            "before building, or pass --allow-missing-engine for a remote-only build"
        )
        return False
    else:
        skip("wheel is remote-only (no in-process engine)")

    if any("__pycache__" in n for n in names):
        fail("wheel contains __pycache__")
        return False
    ok("wheel carries py.typed, stubs, licence, no bytecode")

    sdist_names = tarfile.open(sdist).getnames()
    wanted = ("tests/_fake_server.py", "tests/test_dbapi.py", "MANIFEST.in")
    absent = [w for w in wanted if not any(n.endswith(w) for n in sdist_names)]
    if absent:
        fail(f"sdist is missing: {', '.join(absent)}")
        return False
    ok("sdist is self-sufficient (tests included, runnable)")
    return True


def check_pypi() -> bool:
    """Refuse to build a release PyPI will reject: name taken by someone else,
    or this exact version already uploaded (PyPI never allows a re-upload)."""
    url = f"https://pypi.org/pypi/{NAME}/json"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:  # noqa: S310
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            ok(f"name `{NAME}` is unclaimed on PyPI (first release)")
            return True
        skip(f"PyPI query returned HTTP {e.code}")
        return True
    except (urllib.error.URLError, TimeoutError) as e:
        skip(f"PyPI unreachable ({e.__class__.__name__}); cannot verify the name")
        return True

    published = sorted(data.get("releases", {}))
    if version() in published:
        fail(f"{NAME} {version()} is already on PyPI; bump gtvdb/_version.py")
        return False
    ok(f"name `{NAME}` is ours; {version()} not yet published (have: {', '.join(published) or 'none'})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true", help="skip the PyPI name/version query")
    ap.add_argument("--keep-dist", action="store_true", help="keep artefacts in python/dist")
    ap.add_argument("--skip-tests", action="store_true", help="skip the unit-test gate")
    ap.add_argument(
        "--allow-missing-engine",
        action="store_true",
        help="accept a remote-only wheel (no in-process engine). Use this for a "
        "metadata-only gate; a release should not need it",
    )
    args = ap.parse_args()

    print(f"gtvdb {version()} — release pre-flight\n")
    results = [
        ("licence", check_license()),
        ("stubs", check_stubs()),
        ("version", check_version()),
        ("runtime version", check_runtime_version()),
    ]
    if not args.skip_tests:
        results.append(("tests", check_tests()))
    if not args.offline:
        results.append(("pypi", check_pypi()))

    dist = HERE / "dist" if args.keep_dist else Path(tempfile_dir())
    print(f"\n{DIM}building into {dist}{RESET}")
    built = build(dist)
    if built is not None:
        results.append(
            (
                "artefacts",
                check_artefacts(*built, require_engine=not args.allow_missing_engine),
            )
        )
    else:
        results.append(("artefacts", False))

    print()
    bad = [name for name, good in results if not good]
    if bad:
        print(f"{RED}release blocked{RESET}: {', '.join(bad)}")
        return 1
    print(f"{GREEN}ready to upload{RESET} — see python/PUBLISHING.md")
    if args.keep_dist:
        print(f"artefacts: {dist}")
    else:
        shutil.rmtree(dist, ignore_errors=True)
        print(DIM + "artefacts discarded; re-run with --keep-dist to upload them" + RESET)
    return 0


def tempfile_dir() -> str:
    import tempfile

    return tempfile.mkdtemp(prefix="gtvdb-release-")


if __name__ == "__main__":
    sys.exit(main())
