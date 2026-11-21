"""Worked example: connect to gtv from Python, pin business time, read receipts.

    # terminal 1
    cargo run -p gtv-engine --example serve_demo

    # terminal 2
    python examples/python/hello_grpc.py
    python examples/python/hello_grpc.py 127.0.0.1:50051

Requires `pip install grpcio pyarrow` (see examples/python/README.md).
"""

from __future__ import annotations

import sys

from gtv_client import DEFAULT_ADDR, GtvError, connect


def main() -> int:
    addr = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ADDR

    with connect(addr) as session:
        # 1. The handshake already happened inside connect(); this is what the
        #    server said about itself.
        print("server:", session.ping())

        # 2. A plain query. The rows are an Arrow table; the receipt says
        #    whether the answer is complete and how fresh the engine was.
        result = session.query("SELECT * FROM numbers WHERE n <= 3")
        print(f"\nquery -> {result.rows} rows, tri_state={result.tri_state}")
        print(result.table.to_pydict())
        print("snapshot:", result.receipt["snapshot_id"], "as_of:", result.receipt["as_of"])
        print("freshness:", result.freshness)

        # 3. Pin a business time. The pin travels on every following query — but
        #    note what comes back: `as_of_binding = declared`. Nothing in the
        #    server filtered by that time, so the receipt refuses to claim the
        #    rows answer for it, and tri_state drops to `incomplete`. A pin only
        #    becomes evidence when the statement itself enforces the time (or a
        #    session default reaches every operator).
        pinned = session.at(as_of=1_700_000_000_000_000_000)
        later = pinned.query("SELECT count(*) AS c FROM prices")
        print(f"\npinned at {pinned._as_of} -> {later.table.to_pydict()}")
        print(
            "  as_of:", later.receipt["as_of"],
            "| binding:", later.receipt.get("as_of_binding"),
            "| tri_state:", later.tri_state,
            "| reason:", later.receipt["incomplete"].get("reason")
            if later.receipt.get("incomplete") else None,
        )

        # 3b. Pin the *snapshot* instead. The demo serves exactly one, and it
        #     says which, so the receipt can verify the pin rather than echo it.
        exact = session.at(snapshot_id=1)
        ok = exact.query("SELECT count(*) AS c FROM prices")
        print(
            "  snapshot 1 ->",
            ok.receipt.get("snapshot_binding"),
            "| tri_state:", ok.tri_state,
        )
        try:
            session.at(snapshot_id=99).query("SELECT count(*) FROM prices")
        except GtvError as exc:
            print("  snapshot 99 ->", exc.code)

        # 4. Absence, proven vs. not proven. An empty result is only evidence
        #    when tri_state == "no_match".
        empty = session.query("SELECT * FROM numbers WHERE n > 100")
        print(f"\nempty -> rows={empty.rows}, tri_state={empty.tri_state}, "
              f"conclusive={empty.is_conclusive}")

        # 5. Failures raise, with a stable code rather than prose to parse.
        try:
            session.query("SELECT * FROM no_such_table")
        except GtvError as exc:
            print(f"\nerror -> code={exc.code}")

        # 6. The receipt travels inside the Arrow stream too, so a result that
        #    was written to a file still carries its provenance.
        metadata = result.table.schema.metadata or {}
        print("\nreceipt in Arrow metadata:", b"gtv.receipt.json" in metadata)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
