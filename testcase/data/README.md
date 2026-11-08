# testcase/data — benchmark datasets

The performance scripts (`testcase/run_tc_duration.sh`,
`run_tc_duration1.sh`) run against **fixed bytes**, so latency numbers are
comparable across machines and across code versions.

To keep that property while not storing 167 MB of reproducible output in git:

* the **generators** and the **checksum list** are committed;
* the three large CSVs are git-ignored and rebuilt on demand;
* **no CSV is ever committed** — `*.csv` is ignored repo-wide and the
  `generated-data` CI job fails if any `.csv` / `.seg` file is tracked
  (`git ls-files`). Two >50 MB CSV blobs are already stuck in the history from
  earlier commits; this rule stops the problem from growing.

This also constrains the P1 write path (WAL / MemSegment / flush): its tests
build their data in memory or under a tempdir and may not write a `.csv` into
the tree. An explicit `git add -f` does not help — the guard checks the tracked
set, not the ignore rules.

| File | Size | Produced by | Committed |
|:--|--:|:--|:--|
| `edges_1m.csv` | 33 MB | `gen_graph_vector.py` (`gen_edges`, no RNG) | no — regenerate |
| `vectors_1m.csv` | 77 MB | `gen_graph_vector.py` (`gen_vectors`, seed 20240904) | no — regenerate |
| `pt_test_tab.csv` | 58 MB | `gen_pt_test_tab.py` (seed 20240904) | no — regenerate |
| `stocks_{MCO,NVDA,TSLA}_tick.parquet` | 41 MB | downloaded from the market data API | **yes** (no offline generator) |
| `gen_data.py` | — | misc fixture generator, unrelated to the above | yes |

## Rebuild

```bash
./testcase/data/gen_datasets.sh
```

It generates the three CSVs and then verifies them against `SHA256SUMS`. The
script fails if the bytes differ, so a change in a generator is caught instead
of silently moving the benchmark baseline.

## Why the bytes really are identical

* Both generators use `random.Random(20240904)` with a pinned seed;
  `gen_edges` uses no randomness at all.
* They open output with `newline="\n"`, and `.gitattributes` pins
  `/testcase/data/*.csv text eol=lf`. Both halves are required. Plain text mode
  would emit CRLF on Windows (this machine has `core.autocrlf=true`) and LF on
  Linux, so the same seed produced two different sha256 values and the
  "identical bytes" guarantee only held on the machine that generated the file.
  This was a real defect; it is fixed.
* `SHA256SUMS` is the authority and is computed over that LF form. Regenerating
  and running `sha256sum -c SHA256SUMS` is a positive proof that your bytes are
  the benchmark bytes — it does not rely on trusting the generator.

> Note: before this change the committed blobs were already LF (git normalises on
> commit) while the Windows working tree was CRLF, so the two disagreeing forms
> coexisted unnoticed. `SHA256SUMS` now records the single canonical form.

## Known caveat on the numbers

`tc_history.md` / `hft_tc2-tc4_results.md` (in `docs/archive/`) recorded TC4 at
`< 8 ms` with IVF while `recall@10` was 0%/40%. Latency is only meaningful
paired with a recall threshold — do not read these datasets' timings as
correctness results.
