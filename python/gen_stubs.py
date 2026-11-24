"""Regenerate the gRPC stubs in `gtvdb/_stubs/` from the core `.proto`.

    python python/gen_stubs.py            # write
    python python/gen_stubs.py --check    # fail if out of date (CI)

The stubs are committed and shipped in the package, so a user can
`pip install pygtvdb` without owning a protobuf toolchain. This script exists so
the shipped copy cannot silently drift from
`crates/gtv-proto/proto/gtvquery.proto`.

Only one post-processing step is applied: protoc emits an absolute
`import gtvquery_pb2`, which breaks when the generated files live in a package.
It is rewritten to a relative import. Everything else is protoc's output.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROTO_DIR = Path(__file__).resolve().parents[1] / "crates" / "gtv-proto" / "proto"
PROTO = "gtvquery.proto"
OUT_DIR = Path(__file__).resolve().parent / "gtvdb" / "_stubs"
GENERATED = ("gtvquery_pb2.py", "gtvquery_pb2.pyi", "gtvquery_pb2_grpc.py")


def generate(target: Path) -> None:
    from grpc_tools import protoc  # imported late so --help works without it

    target.mkdir(parents=True, exist_ok=True)
    rc = protoc.main(
        [
            "protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={target}",
            f"--grpc_python_out={target}",
            f"--pyi_out={target}",
            PROTO,
        ]
    )
    if rc != 0:
        raise SystemExit(f"protoc failed with exit code {rc}")

    grpc_file = target / "gtvquery_pb2_grpc.py"
    source = grpc_file.read_text(encoding="utf-8")
    patched = source.replace(
        "import gtvquery_pb2 as gtvquery__pb2",
        "from . import gtvquery_pb2 as gtvquery__pb2",
    )
    grpc_file.write_text(patched, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed stubs instead of writing them",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp)
        generate(fresh)
        if not args.check:
            for name in GENERATED:
                (OUT_DIR / name).write_text(
                    (fresh / name).read_text(encoding="utf-8"), encoding="utf-8"
                )
            print(f"wrote {len(GENERATED)} files to {OUT_DIR}")
            return 0

        stale = [
            name
            for name in GENERATED
            if not (OUT_DIR / name).exists()
            or (OUT_DIR / name).read_text(encoding="utf-8")
            != (fresh / name).read_text(encoding="utf-8")
        ]
        if stale:
            print("stubs are out of date: " + ", ".join(stale), file=sys.stderr)
            print("run: python examples/python/gen_stubs.py", file=sys.stderr)
            return 1
        print("stubs match the .proto")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
