#!/usr/bin/env bash
#
# Regenerate the large benchmark CSVs under testcase/data.
#
# These three files are deterministic (pinned RNG seed, LF forced by the
# generators), so regeneration reproduces the exact sha256 recorded in
# ./SHA256SUMS. They are NOT committed: together they are ~167 MB of
# reproducible output, only the generators and the checksum list are.
# P1 hard rule: no CSV is ever added to git (CI rejects any tracked *.csv).
#
# Usage:
#   ./testcase/data/gen_datasets.sh          # generate + verify
#   PYTHON=python3 ./testcase/data/gen_datasets.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3 python py; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "gen_datasets.sh: no python interpreter found (set PYTHON=...)" >&2
  exit 1
fi

echo "==> python: $($PY --version 2>&1)"
echo "==> generating edges_1m.csv + vectors_1m.csv"
"$PY" "$here/gen_graph_vector.py" "$here"

echo "==> generating pt_test_tab.csv (1M rows)"
"$PY" "$here/gen_pt_test_tab.py" 1000000 "$here/pt_test_tab.csv"

echo "==> verifying sha256 against SHA256SUMS"
if command -v sha256sum >/dev/null 2>&1; then
  ( cd "$here" && sha256sum -c SHA256SUMS )
elif command -v shasum >/dev/null 2>&1; then
  ( cd "$here" && shasum -a 256 -c SHA256SUMS )
else
  echo "    no sha256sum/shasum available; skipping verification" >&2
fi

echo "==> done"
