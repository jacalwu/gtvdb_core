"""The user guide's REPL session, driven from Python.

    python examples/python/hello_python.py

Everything here also works by hand in the REPL; the point is that Python can
own the session: build the inputs, run the SQL, and get rows back as dicts.
Requires a built binary (`cargo build -p gtv-cli`).
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gtv_repl import Gtv, GtvError  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> int:
    t0 = time.perf_counter()
    work = Path(tempfile.mkdtemp(prefix="gtv-py-"))
    tx_csv = work / "tx.csv"
    edge_csv = work / "edges.csv"
    tx_csv.write_text(
        "ts,amount,party\n"
        "1000,50.0,A\n"
        "2000,70.0,B\n"
        "3000,90.0,C\n",
        encoding="utf-8",
    )
    edge_csv.write_text(
        "src,dst,edge_type,valid_from,valid_to\n"
        "0,1,1,10000000000,20000000000\n"
        "1,2,1,11000000000,21000000000\n"
        "2,0,1,12000000000,22000000000\n",
        encoding="utf-8",
    )

    with Gtv() as db:
        rule("① 启动：一个 gtv 子进程，状态在整个会话里保持")
        print(f"binary   {db.binary}")
        print(f"startup  {(time.perf_counter() - t0) * 1000:.0f} ms (含构建产物定位)")
        print(db.banner)

        rule("② 注册表是空的——REPL 起来时不认识任何表")
        print(db.dot(".tables") or "(空)")

        rule("③ 用 Python 写 CSV，再交给 gtv 注册")
        print(f"wrote {tx_csv.as_posix()}")
        # 注意：路径要用 as_posix()，REPL 不认 MSYS 的 /c/... 写法
        print(db.dot(f".csv tx {tx_csv.as_posix()}"))
        print(db.dot(f".csv edges {edge_csv.as_posix()}"))
        print(db.dot(".tables"))

        rule("④ 数据质量：先把报表取回来，再在 Python 里判断")
        for row in db.query("SELECT * FROM dq_report('tx');"):
            print(f"dq_report  rows={row['rows']} dup_ts_sym={row['dup_ts_sym']} "
                  f"non_monotonic={row['non_monotonic']}")
        checks = db.query("SELECT * FROM health_check('tx');")
        warns = [c for c in checks if c["status"] == "warn"]
        for c in checks:
            print(f"  {c['check']:<14} {c['status']:<5} {c['detail']}")
        print(f"→ {len(warns)} 项 warn，{len(checks) - len(warns)} 项 ok"
              "（ts=1000 是 1970 年，所以 freshness 报 warn 是正常的）")

        rule("⑤ 量子状态图：森林形状 / 存活分支 / 稀有分支的门序列")
        summary = db.query(
            "SELECT * FROM quantum_state_summary('examples/quantum-state/ghz3.jsonl');"
        )[0]
        print(f"states={summary['states']} transitions={summary['transitions']} "
              f"depth={summary['max_depth']} semantics={summary['semantics']}")

        frontier = db.query(
            "SELECT state_id, gate, probability "
            "FROM quantum_state_frontier('examples/quantum-state/ghz3.jsonl', 6, 0.05) "
            "ORDER BY probability DESC;"
        )
        for row in frontier:
            print(f"  state {row['state_id']:<3} {row['gate']:<14} p={row['probability']}")
        rarest = min(frontier, key=lambda r: float(r["probability"]))
        print(f"→ 最稀有分支 state {rarest['state_id']} (p={rarest['probability']})，追溯它的门序列：")

        for row in db.query(
            "SELECT step, gate, prefix_probability FROM "
            f"quantum_state_reconstruct('examples/quantum-state/ghz3.jsonl', {rarest['state_id']});"
        ):
            print(f"  [{row['step']}] {row['gate']:<20} p={row['prefix_probability']}")

        rule("⑥ 参数与加速层：REPL 的原生输出，Python 原样拿到")
        params = db.dot(".params").splitlines()
        print("\n".join(params[:6]))
        print("...")
        print("\n".join(db.dot(".accel").splitlines()[:4]))

        rule("⑦ 冷开一份 Rust 侧建好的向量索引（如果 hello_vector 跑过）")
        indexes = sorted(
            Path(tempfile.gettempdir()).glob("gtv-hello-vector/*/v2/index.gtvidx")
        )
        if not indexes:
            print("没找到索引文件。先跑一次：")
            print("  cargo run -q -p gtv-cli --example hello_vector")
            print("（建索引走 Rust API，Python 侧只能读——见手册 §12.5 的边界）")
        else:
            path = indexes[-1].as_posix()
            print(f"index {path}")
            print(db.dot(f".knn-index songs {path}"))
            hits = db.query("SELECT id, distance FROM knn('songs', '0.1,0.9', 2);")
            for row in hits:
                print(f"  id={row['id']:<3} distance={row['distance']}")

        rule("⑧ 出错时 Python 拿到的是异常，不是被吞掉的输出")
        for statement in (
            f".csv nope {work / 'does-not-exist.csv'}",
            "SELECT * FROM neighbors(0, 16000000000);",
        ):
            ok, text = db.try_sql(statement)
            print(f"ok={ok}  {statement.split()[0]} → {text.splitlines()[-1]}")
        try:
            db.sql("SELECT * FROM neighbors(0, 16000000000);")
        except GtvError as exc:
            print(f"raised GtvError: {exc}")

    print(f"\n总耗时 {(time.perf_counter() - t0) * 1000:.0f} ms（含一次进程启动）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
