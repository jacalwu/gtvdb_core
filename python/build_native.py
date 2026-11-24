#!/usr/bin/env python
"""Build the in-process engine extension and install it into the package.

Why a script rather than the three README commands: this build fails for
reasons that are purely environmental and that every new machine hits —

* Git Bash's ``/usr/bin/link.exe`` shadows MSVC's linker, so ``rustc`` links
  with a GNU ``link`` that does not understand MSVC object files;
* a global ``RUSTFLAGS=-C linker=lld`` breaks the link when ``lld`` is not
  installed (and says so only at the very end of a multi-minute build);
* ``RUSTC_WRAPPER=sccache`` refuses to run when ``CARGO_INCREMENTAL=1``,
  which some shells export globally.

This normalises all three, runs the build, then copies the artefact under the
exact name CPython imports. CI calls the same entry point, so a green pipeline
means the documented local command works.

    python python/build_native.py            # build, install into package
    python python/build_native.py --wheel    # also build the wheel
    python python/build_native.py --check    # report toolchain, build nothing
    python python/build_native.py --clean    # cargo clean, then build

The MSVC environment is assembled directly instead of by shelling out to
``vcvars64.bat``: ``cmd /c`` mangles the quoted batch path, and the batch file
reaches for ``reg.exe`` to probe SDK versions, which locked-down sandboxes
block. Setting ``PATH``/``INCLUDE``/``LIB`` is all that batch file does anyway.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # python/
REPO = HERE.parent  # repository root
PACKAGE = HERE / "gtvdb" / "_local"  # where the package imports the artefact
CRATE = "gtvpy-local"
LIB_NAME = "_gtv_native"  # [lib] name in crates/gtvpy-local/Cargo.toml
HOST = "Hostx64"  # the interpreter's own architecture; x64 is the only wheel we ship


def _is_windows() -> bool:
    return sys.platform == "win32"


def _artefact_name() -> str:
    """Cargo's output file name for a cdylib on this platform."""
    if _is_windows():
        return f"{LIB_NAME}.dll"
    if sys.platform == "darwin":
        return f"lib{LIB_NAME}.dylib"
    return f"lib{LIB_NAME}.so"


def _target_name() -> str:
    """The name CPython's extension loader will find inside the package."""
    if _is_windows():
        return f"{LIB_NAME}.pyd"
    # `.abi3.so` is one of the suffixes importlib probes, and it records that
    # the artefact really is the limited-API build pyo3 was asked to produce.
    return f"{LIB_NAME}.abi3.so"


# ---------------------------------------------------------------------------
# MSVC discovery (no vcvars, no reg.exe)
# ---------------------------------------------------------------------------


def _newest_dir(root: Path) -> Path | None:
    """The lexicographically greatest subdirectory, or None."""
    if not root.is_dir():
        return None
    entries = [entry for entry in root.iterdir() if entry.is_dir()]
    return max(entries, key=lambda entry: entry.name) if entries else None


def _find_vs_install() -> Path | None:
    """Locate the Visual Studio install root (the directory holding ``VC/``)."""
    candidates: list[Path] = []

    vswhere = (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    if vswhere.exists():
        result = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        candidates.extend(
            Path(line.strip()) for line in result.stdout.splitlines() if line.strip()
        )

    program_files = os.environ.get("ProgramFiles")
    if program_files:
        for edition in ("Enterprise", "Professional", "Community", "BuildTools"):
            candidates.append(
                Path(program_files) / "Microsoft Visual Studio" / "2022" / edition
            )

    return next((c for c in candidates if (c / "VC" / "Tools" / "MSVC").is_dir()), None)


def _msvc_environment(vs_root: Path) -> tuple[dict[str, str], list[str]]:
    """Return ``(env, notes)`` reproducing what ``vcvars64.bat`` sets."""
    notes: list[str] = []
    msvc_root = _newest_dir(vs_root / "VC" / "Tools" / "MSVC")
    if msvc_root is None:
        raise FileNotFoundError(f"no MSVC toolset under {vs_root / 'VC' / 'Tools' / 'MSVC'}")

    env = dict(os.environ)

    bin_dirs = [msvc_root / "bin" / HOST / "x64"]
    include_dirs = [msvc_root / "include"]
    lib_dirs = [msvc_root / "lib" / "x64"]

    sdk_root = (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Windows Kits"
        / "10"
    )
    sdk_version = _newest_dir(sdk_root / "Include")
    if sdk_version is None:
        notes.append(f"no Windows SDK under {sdk_root}; linking may fail")
    else:
        bin_dirs.append(sdk_root / "bin" / sdk_version.name / "x64")
        include_dirs.extend(
            sdk_root / "Include" / sdk_version.name / part
            for part in ("ucrt", "shared", "um", "winrt")
        )
        lib_dirs.extend(
            sdk_root / "Lib" / sdk_version.name / part / "x64"
            for part in ("ucrt", "um")
        )
        notes.append(f"MSVC {msvc_root.name}, Windows SDK {sdk_version.name}")

    env["PATH"] = os.pathsep.join(str(p) for p in bin_dirs) + os.pathsep + env.get("PATH", "")
    env["INCLUDE"] = os.pathsep.join(str(p) for p in include_dirs)
    env["LIB"] = os.pathsep.join(str(p) for p in lib_dirs)
    return env, notes


# ---------------------------------------------------------------------------
# Environment normalisation
# ---------------------------------------------------------------------------


def _strip_lld_flag(rustflags: str) -> str:
    """Remove ``-C linker=lld`` (either spelling) and keep everything else.

    Token-wise rather than by ``str.replace``: ``RUSTFLAGS`` usually carries
    several ``-C`` pairs, and dropping only the value would leave a dangling
    ``-C`` that rustc reports as a much less obvious error.
    """
    tokens = rustflags.split()
    kept: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-C" and index + 1 < len(tokens):
            if tokens[index + 1].strip() == "linker=lld":
                index += 2
                continue
        if token == "-Clinker=lld":
            index += 1
            continue
        kept.append(token)
        index += 1
    return " ".join(kept)


def build_env() -> tuple[dict[str, str], list[str]]:
    """Return ``(env, notes)`` with the known build blockers removed."""
    notes: list[str] = []

    # MSVC first: it rebuilds the environment from scratch, so any cleanup done
    # before this point would be thrown away.
    if _is_windows():
        vs_root = _find_vs_install()
        if vs_root is None:
            raise FileNotFoundError(
                "no Visual Studio C++ toolchain found; install Visual Studio Build "
                "Tools with the 'Desktop development with C++' workload"
            )
        env, msvc_notes = _msvc_environment(vs_root)
        notes.extend(msvc_notes)
    else:
        env = dict(os.environ)

    # 1. lld: if RUSTFLAGS asks for it but it is not on PATH, drop just that
    #    flag and fall back to the platform linker. Keeping it turns a missing
    #    tool into a multi-minute build that fails at the last step.
    rustflags = env.get("RUSTFLAGS", "")
    if "linker=lld" in rustflags and shutil.which("lld") is None:
        env["RUSTFLAGS"] = _strip_lld_flag(rustflags)
        notes.append("dropped `linker=lld` from RUSTFLAGS (lld is not installed)")

    # 2. sccache refuses incremental compilation. It wants the variable *gone*,
    #    not set to "0" — setting it to 0 still aborts with "Unset
    #    CARGO_INCREMENTAL to continue".
    if env.get("RUSTC_WRAPPER", "").endswith("sccache") and "CARGO_INCREMENTAL" in env:
        env.pop("CARGO_INCREMENTAL", None)
        notes.append("unset CARGO_INCREMENTAL (sccache forbids incremental builds)")

    # Keep warnings-colour tokens out of the artefact comparison in CI logs.
    env.setdefault("CARGO_TERM_COLOR", "always")
    return env, notes


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cargo_command() -> list[str]:
    return ["cargo", "build", "-p", CRATE, "--release"]


def build(*, clean: bool) -> int:
    try:
        env, notes = build_env()
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for note in notes:
        print(f"note: {note}")

    if clean:
        print(f"cargo clean -p {CRATE}")
        subprocess.run(["cargo", "clean", "-p", CRATE], cwd=REPO, env=env, check=False)

    command = _cargo_command()
    print("building: " + subprocess.list2cmdline(command))
    result = subprocess.run(command, cwd=REPO, env=env, check=False)
    if result.returncode != 0:
        print("error: cargo build failed", file=sys.stderr)
        return result.returncode

    produced = REPO / "target" / "release" / _artefact_name()
    if not produced.exists():
        print(f"error: expected artefact missing: {produced}", file=sys.stderr)
        return 3

    destination = PACKAGE / _target_name()
    shutil.copy2(produced, destination)
    size_mib = destination.stat().st_size / (1024 * 1024)
    print(f"installed {destination.relative_to(REPO)} ({size_mib:.1f} MiB)")
    return 0


def build_wheel() -> int:
    """Build the platform wheel for the current interpreter."""
    # The Linux branch invokes setup.py directly (see below), so it runs in this
    # interpreter's environment rather than in a build-isolated one. setup.py
    # subclasses `bdist_wheel`, which needs either setuptools >= 70.1 or the
    # `wheel` package — and on a bare runner neither is guaranteed. Checking up
    # front turns a traceback into an install instruction.
    missing = []
    if importlib.util.find_spec("wheel") is None:
        try:
            from setuptools.command.bdist_wheel import (  # noqa: F401
                bdist_wheel as _probe,
            )
        except ImportError:
            missing.append("wheel")
    if missing:
        print(
            f"error: the wheel step needs {', '.join(missing)} installed in this "
            "interpreter; run `python -m pip install wheel` and try again",
            file=sys.stderr,
        )
        return 2

    if sys.platform.startswith("linux"):
        # PyPI rejects `linux_x86_64`; the tag must name the glibc floor the
        # wheel was built against. `python -m build` cannot pass this through.
        command = [
            sys.executable,
            "setup.py",
            "bdist_wheel",
            "--plat-name",
            "manylinux_2_34_x86_64",
        ]
    else:
        command = [sys.executable, "-m", "build", "--wheel"]

    print("building wheel: " + subprocess.list2cmdline(command))
    result = subprocess.run(command, cwd=HERE, check=False)
    if result.returncode != 0:
        # Name the stage that failed: the cargo build above succeeded, so a
        # non-zero exit here is packaging, not compilation, and saying so saves
        # reading a 20 000-line log to find out.
        print(
            f"error: the extension built, but the wheel step failed "
            f"(exit {result.returncode})",
            file=sys.stderr,
        )
        return result.returncode

    built = sorted(HERE.glob("dist/*.whl"))
    if not built:
        print("error: the wheel step reported success but produced no file", file=sys.stderr)
        return 4
    for path in built:
        print(f"wheel: {path.relative_to(REPO)}")
    return 0


def check() -> int:
    """Report the toolchain state without building anything."""
    print(f"platform          : {sys.platform}")
    print(f"python            : {sys.version.split()[0]} ({sys.executable})")
    print(f"artefact          : {_artefact_name()} -> {_target_name()}")

    cargo = shutil.which("cargo")
    print(f"cargo             : {cargo or 'MISSING'}")
    if cargo:
        version = subprocess.run(
            ["cargo", "--version"], capture_output=True, text=True, check=False
        )
        print(f"cargo version     : {version.stdout.strip() or version.stderr.strip()}")

    try:
        env, notes = build_env()
    except FileNotFoundError as error:
        print(f"toolchain         : MISSING ({error})")
        return 1

    print(f"RUSTC_WRAPPER     : {env.get('RUSTC_WRAPPER') or '(unset)'}")
    print(f"RUSTFLAGS         : {env.get('RUSTFLAGS') or '(unset)'}")
    print(f"CARGO_INCREMENTAL : {env.get('CARGO_INCREMENTAL') or '(unset)'}")
    if _is_windows():
        print(f"INCLUDE           : {env.get('INCLUDE', '')[:80]}...")
    for note in notes:
        print(f"adjustment        : {note}")

    installed = PACKAGE / _target_name()
    print(f"in package        : {'present' if installed.exists() else 'not built'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the pygtvdb_local native extension.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--wheel", action="store_true", help="also build the wheel")
    parser.add_argument("--check", action="store_true", help="report toolchain only")
    parser.add_argument("--clean", action="store_true", help="cargo clean first")
    args = parser.parse_args()

    if args.check:
        return check()

    code = build(clean=args.clean)
    if code != 0:
        return code
    if args.wheel:
        return build_wheel()
    return 0


if __name__ == "__main__":
    sys.exit(main())
