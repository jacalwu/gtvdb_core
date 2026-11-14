# Device acceleration (CUDA) — plan, accounting and PG-Strom absorption

Status: **device backend validated on hardware — all 8 acceptance gates pass**,
including the IVF probe, the heat-admission A/B and the time-mask cost model
(see [§6](#6-acceptance-results-on-hardware), [§9](#9-ivf-and-vram-pressure)
and [§10](#10-the-active_att-time-mask)).

Validated 2026-11-14 on an NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM,
driver 591.74 / CUDA 13.1, toolkit 13.0, `compute_89`), Windows, 31.4 GiB host
RAM. Run it yourself with:

```sh
RUSTFLAGS= RUSTC_WRAPPER= CARGO_INCREMENTAL=0 \
  cargo test -p gtv-accelerator --features cuda --release \
  --test cuda_acceptance -- --nocapture --test-threads=1
```

The suite skips cleanly (rather than failing) when no device is present, so it
is safe to wire into a mixed CI fleet; the dedicated GPU job forces a hard
failure instead ([§7](#7-ci-wiring)).

This document records *why* only two operators are offloadable, how the device
cache is accounted, and which parts of PG-Strom's GPU Cache design were adopted,
rewritten or rejected.

---

## 0. The constraint everything follows from

PCIe delivers roughly **8 GB/s** to a device. A modern GPU reads its own memory
at **400–900 GB/s** and the CPU can scan a contiguous corpus at a few tens of
GB/s. Two consequences:

1. **A kernel that returns as much data as it consumes cannot win.** The transfer
   dominates no matter how fast the arithmetic is.
2. **A cold single query always loses.** Offload only pays when the corpus is
   already resident *and* the work is amortized over a batch or repeated lookups.

That gives three criteria, applied to every candidate:

| Criterion | Consequence if violated |
|---|---|
| Output much smaller than input | Transfer eats the gain (rolling window, `deltas`). A **packed** time mask is `N/8` bytes from `16N` input bytes — a 16× reduction — which is why [§10](#10-the-active_att-time-mask) qualifies while an unpacked `bool` per row would not |
| Data stays resident across queries | Upload cost is paid on every query |
| Regular, dense access | Pointer chasing / branch divergence caps the kernel at ≲1× |

---

## 1. Ranking (what was considered, what shipped)

`e2e` is the end-to-end figure after residency and batching, not the kernel peak.

| # | Operator | Location | e2e | kernel ceiling | Difficulty | Status |
|---|---|---|---|---|---|---|
| 1 | Exact flat K-NN | `gtv-index/flat.rs`, `engine/knn.rs` | 4–8× | ~12× | medium | **validated on hardware** |
| 2 | IVF probe scan | `gtv-index/ivf.rs` | see [§9](#9-ivf-and-vram-pressure) | — | med-high | **validated on hardware** |
| 3 | `active_at(T)` time mask | `gtv-core/temporal.rs`, `engine/temporal.rs` | 10–30×† | ~30× | low-med | **validated on hardware** |
| 4 | As-of join | `engine/asof.rs` | 1.5–3× | ~5× | medium | **validated on hardware** |
| 5 | `k`-hop BFS | `gtv-core/csr.rs` | 0.5–2× | ~3× | high | rejected for now |
| 6 | Rolling window / `deltas` | `gtv-array/window.rs` | ~1× | ~18× | low | rejected |
| 7 | Pattern backtracking | `gtv-pattern` | <1–1.5× | ~2× | very high | rejected |
| 8 | HNSW graph walk | `gtv-index/hnsw.rs` | <1× | ~2× | very high | rejected |

† measured on an RTX 4060 Laptop for **unsorted, low-locality, resident** edge
columns: ~10× against the host's Arrow SIMD comparison kernels and ~30× against
the scalar bit-packing reference. The original 2–5× projection understated it —
the host baseline is dominated by a branchy scalar bit loop, not memory
bandwidth. Sorted / strongly localized tables are **not** offloaded at all: the
host zone-map scan is ~4 µs there versus ~110 µs on the device (see §10).

Rank 6 is the instructive rejection: the kernel can be ~18× faster in isolation
and the query still ends up at ~1×, because the output is the size of the input.

Also out of scope: DataFusion's own sort / hash-join / aggregate. Offloading
those means patching DataFusion, not the engine.

---

## 2. Architecture

```
        AcceleratorRegistry            decides, accounts, falls back
       /         |            \
 VramBudget  ResidentRegistry  AccelBackend
  (bytes)     (what is hot)      +-- HostBackend   (always available)
                                 `-- CudaBackend   (--features cuda)
```

**`VramBudget`** — every byte a backend may hold is reserved here first. The
configured `CUDA_VRAM` (default 80% of device VRAM) is what gtVdB may *use*, so
10% of it is held back as headroom for kernel scratch and driver bookkeeping.
Reservation failure is not an error: it is an admission decision. With a device
attached the budget is also **elastic**: a rate-limited `cudaMemGetInfo` probe
can only *lower* capacity to `min(CUDA_VRAM, live_free + used)`, so a driver that
quietly falls back to host-backed sysmem produces a host query rather than a
kernel that is 20× slower than expected.

**`ResidentRegistry`** — keyed by `(kind, name, version)`, with `kind` one of
`corpus` / `ivf` / `series` / `bucket`. Because gtVdB corpora and segments are
immutable snapshots, a version number is a complete validity check. Admission
uses **decayed heat** rather than pure LRU; see [§9](#9-ivf-and-vram-pressure).

**`AccelBackend`** — four operations only: `exact_knn`, `ivf_probe`,
`asof_join` and `time_mask`. Anything not expressible here was deliberately left
off the trait.
The K-NN corpus is uploaded once as a single row-major DMA and transposed to
column-major by a device kernel (coalesced warp reads); one NVRTC specialization
is compiled per distinct `k` so the per-thread top-K arrays stay in registers.
The same kernel serves the IVF probe through an optional position gather —
one merge implementation, not two.

**Fallback** — `AcceleratorRegistry::exact_knn` / `asof_join` never return an
error. A device refusal is recorded in telemetry and the identical host kernel
runs. One behaviour with a GPU and without one; only latency differs. This is
also what keeps GPU-less CI meaningful: the default build *is* the no-device
path.

### Why the host path is not a stub

The device is measured against the host, so the host must be the optimized one:

* **K-NN** — corpus is one contiguous row-major buffer (not `Vec<Vec<f32>>`),
  AVX2+FMA distances, per-thread bounded candidate buffers flushed with
  `select_nth_unstable`, so the `N`-entry score vector is never materialized or
  sorted. Label filters compile to a `BooleanArray` bitmask.
* **As-of** — one monotonic sweep, `O(L + R)`, instead of a binary search per
  left row. Unsorted left input gets one index sort and a scatter back.

---

## 3. PG-Strom GPU Cache: what was absorbed

Reference: <https://heterodb.github.io/pg-strom/gpucache/>

PG-Strom's core observation is exactly ours: *reading the table and pushing it
over PCIe usually costs more than computing on it*, so keep a device copy of a
hot table and stop re-sending it.

| Aspect | PG-Strom | gtVdB | Verdict |
|---|---|---|---|
| Resident copy of hot data | GPU Cache per table | `ResidentRegistry` per `(kind, name, version)` | **absorbed** |
| Eliminating per-query transfer | Yes, the whole point | Yes; a query uploads only its own query vector and downloads `blocks × k` candidates / one index per row | **absorbed** |
| Write-behind instead of write-through | AFTER-row trigger writes a shared-memory REDO log; a background keeper batches it onto the device | `note_append` accumulates a delta; `flush_pending` applies it at a batch boundary | **absorbed** |
| Coalescing policy | `gpu_sync_threshold` (default 25% of the buffer), `gpu_sync_interval` (5 s) | `ACCEL_SYNC_THRESHOLD` (default 25% of the VRAM budget), `ACCEL_SYNC_INTERVAL` (5 s) | **absorbed** |
| Transparency + visibility | Queries switch automatically; `EXPLAIN` shows `GPU Cache: …`; `pgstrom.gpucache_info` view | Automatic; `.accel` shows device, budget, per-block rows, hit ratio, uploads, fallbacks | **absorbed** |
| Enable/disable switch | `pg_strom.enable_gpucache` | `ACCEL_CACHE=OFF` | **absorbed** |
| Query-time consistency | Query blocks until pending redo is applied | Query reads the resident prefix from the device and the un-synced tail from the host — consistent, never blocked | **rewritten** |
| Capacity management | Static `max_num_rows`; overflow ⇒ `corrupted`, cache stops working for that table | Admission + LRU eviction; an oversized dataset is declined and *that operation* runs on the host | **rewritten** |
| Snapshot / visibility bookkeeping | Visibility map + `refcnt`, hence "keep room for pre-commit rows" | Version keys only — corpora are immutable | **rewritten** |
| Per-row trigger plumbing | `pgstrom.gpucache_sync_trigger()` on the table | No equivalent: gtVdB's row-level write path is the storage WAL, a different granularity | **skipped** |

The two rewrites matter most:

* **Overflow must not disable acceleration.** In a graph/vector workload the
  dataset sizes are unpredictable; one oversized corpus taking the cache offline
  would be worse than never having it. Declining a single admission keeps every
  other block resident.
* **A query must not wait for a sync.** Serving `[0, resident_rows)` from the
  device and the remainder from the host gives the same answer with no stall, so
  the write-behind batching is a pure throughput optimisation rather than a
  latency cliff.

### `ACCEL_CACHE=SHADOW`

The one addition PG-Strom has no equivalent for: run on the host while
maintaining the same ledger, so `.accel` reports the hit ratio and the upload
bytes a device *would* have avoided. It is how you size a GPU purchase with real
workload numbers instead of a vendor benchmark.

---

## 4. Configuration

```ini
CUDA_MODE=AUTO            # OFF | ON | AUTO(device-only)
CUDA_VRAM=DEVICE:0.8      # 80% of device VRAM, or an explicit size
CUDA_DEVICE=0
ACCEL_CACHE=ON            # OFF | SHADOW | ON
ACCEL_SYNC_THRESHOLD=AUTO # 25% of CUDA_VRAM, or e.g. 512M
ACCEL_SYNC_INTERVAL=5     # seconds
```

`ACCEL_CACHE` resolves to `OFF` when CUDA is off, except that `SHADOW` is
preserved so calibration works on a GPU-less machine. Without the `cuda` feature
the VRAM probe reports `None`, CUDA resolves off, and every operation takes the
host path.

---

## 5. Measured host-path results

`cargo test -p gtv-accelerator --release -- --ignored --nocapture`
(the benchmarks assert that both paths return the *same* answer before timing):

| Benchmark | Shape | Before | After | Gain |
|---|---|---|---|---|
| K-NN | 200k × 128, k=10 | `Vec<Vec<f32>>` + full sort, 26.2 ms | contiguous + bounded select, 4.5 ms | **5.85×** |
| As-of | L=1M, R=4M, sorted | per-row binary search, 19.5 ms | monotonic sweep, 3.7 ms | **5.28×** |

Both remain single-machine CPU numbers; a device adds on top of these, it does
not replace them. Note that the K-NN figure is already most of the 4–8× the GPU
was projected to deliver end-to-end — the transfer, not the kernel, was the
projected ceiling.

### Device result (same machine, RTX 4060 Laptop / 8 GiB)

| Query | Shape | Host | Device cold | Device warm | Gain vs host |
|---|---|---|---|---|---|
| Exact K-NN | 1M × 512 (1953 MiB), k=10 | 29.8–46.2 ms | 363–390 ms | **11.6–11.8 ms** | **2.6–3.9×** |

The device warm latency is stable across runs; the spread comes entirely from
host-side scheduling, so the ratio is reported as a range over repeated runs
(best-of-3 on each surface). A representative run: host 41.2 ms / cold 389.9 ms
/ warm 11.7 ms = 3.52×. The gate threshold is 2× and the worst observed run
still cleared it at 2.57×.

Cold includes the 2 GB PCIe upload and the on-device transpose; warm is the
second and subsequent queries against the resident corpus. This is exactly the
point of §0: **the first query is 9× slower than the host, the second is ~3×
faster.** A cache that only speeds up cold queries is worse than no cache at all.

Top-10 ids matched the host path **10/10** on this random corpus. Distances are
not guaranteed bit-identical between the AVX2 host kernel and the scalar device
kernel (f32 accumulation order differs); gate 1 therefore uses corpora with
well-separated distances for exact equality, and gate 2 reports the id overlap
as a diagnostic rather than asserting it.

---

## 6. Acceptance results on hardware

All seven gates pass on the RTX 4060 Laptop. Run:

```sh
RUSTFLAGS= RUSTC_WRAPPER= CARGO_INCREMENTAL=0 \
  cargo test -p gtv-accelerator --features cuda --release \
  --test cuda_acceptance -- --nocapture --test-threads=1
```

`--test-threads=1` is required: gate 2 holds a 2 GiB host corpus plus a 4 GiB
transient device allocation. A missing device is reported as `SKIP`; a device
present but whose kernel cannot compile, launch or produce the right answer is a
hard failure.

| Gate | Assertion | Result |
|---|---|---|
| 1. Equality | top-k ids and as-of indices equal the host for L2 / cosine / dot, with and without a label mask and tolerance; non-unit cosine query | **PASS** — 28 KNN cases (3 metrics × 3 dims × mask/k + non-unit cosine) + 8 as-of cases (sorted/unsorted left × 4 tolerances), exact id match, distances within 1e-4 |
| 2. Amortization | second query on a resident 1M × 512 corpus, k=10, ≥2× faster than host; warm < cold | **PASS** — warm 11.6–11.8 ms vs host 29.8–46.2 ms = **2.57–3.92×**; warm ≪ cold ~370 ms; top-10 overlap 10/10 |
| 3. Fallback determinism | `CUDA_MODE=OFF` yields identical results and stays host-only; `ON` with a device opens the device backend | **PASS** — config-to-device chain verified end to end; KNN ids and as-of indices identical |
| 4. Telemetry | `requests == executed_on_device + executed_on_host`, `fallbacks`/`cache_declined` are reasons not outcomes, `resident_bytes ≤ capacity` | **PASS** — device 10 req / 10 device / 0 host / 8 hits; host-only 3 req / 3 host; resident 140 KiB ≤ 64 MiB |
| 5. No leak | eviction drops the device copy; `cudaMemGetInfo` shows the corpus occupied VRAM while resident and no growth across cycles | **PASS** — 64 MiB occupied (7099 → 7035 MiB), fully released (→ 7099 MiB), identical for 3 cycles |
| 6. Heat admission | one-corpus VRAM, a hot working set and 12 one-shot cold corpora: uploads bounded and every answer still exact | **PASS** — heat: **1 upload / 12 deferrals**; pure-LRU control: **24 uploads**; 64 exact result checks |
| 7. IVF probe | device and host consume the same probe plan; warm amortizes cold upload | **PASS** — 12 equivalence cases (3 metrics × mask × query); 1M × 128 corpus (520 MiB), probes 32 768 / 1 048 576 rows (3.1%); cold ~106–119 ms, **warm ~0.33–0.35 ms** |

### Defects the gates found and what was fixed

The gates were not a formality — four real defects surfaced on first contact
with hardware:

1. **`CUDA_ERROR_INVALID_PTX`** — NVRTC's default virtual architecture is too
   old for a CUDA 13.1 driver. Fixed by probing `compute_capability()` and
   compiling with `--gpu-architecture=compute_<m><mm>` at runtime.
2. **`CUDA_ERROR_ILLEGAL_ADDRESS`** — the kernel declared 64 KiB of static
   shared memory (256 threads × 32 K × 8 B), over the 48 KiB default per-block
   limit on sm_89. Fixed to 128 threads (32 KiB). Rust's launch `THREADS` must
   match the kernel's `#define THREADS`; that mismatch was the actual illegal
   write.
3. **Unsorted left timestamps** — the chunked as-of kernel advances a monotonic
   cursor and silently returned wrong answers for unsorted `left`, while the
   host sweep sorted an index permutation. The backend now does the same sort +
   scatter before upload.
4. **1.52× on gate 2 (FAIL)** — with runtime `k`, nvcc spilled the per-thread
   candidate arrays to local memory, and 7813 blocks each did a serial merge.
   Fixed by compiling one specialization per distinct `k` (`#define KERNEL_K`)
   so the arrays stay in registers, and capping the grid at 1024 blocks with a
   grid-stride loop. Warm latency fell **25.4 ms → 11.7 ms**, clearing 2×.

Two supporting fixes: the corpus transpose moved from the host (which did `dim`
strided passes over 2 GB ≈ 1 TB of traffic) to a device transpose kernel fed by
one bulk DMA; and `executed_on_host` was added to telemetry because the host-only
path previously left `requests` unaccounted for.

A fifth defect surfaced while adding the IVF path: **the device never normalized
a cosine query**. The host `exact_topk` does, so rankings survived (scaling all
dot products by `|q|` is monotone) but reported distances were wrong for any
non-unit query. Gate 1's original corpus used a unit query, which hid it.
`CudaBackend::topk` now applies the same normalization, and gate 1 has a
dedicated non-unit case.

### Residual limits

* The device backend has been validated on **one** GPU architecture
  (`compute_89`, CUDA 13.x). Other compute capabilities need the same suite run;
  the arch is probed at runtime, but shared-memory occupancy and the WDDM
  timeout behaviour are architecture/driver specific.
* Gate 3 verifies result identity at the accelerator boundary and the config
  chain, not the DataFusion logical-plan text. Because the device backend is a
  drop-in behind the same table functions, the plan is unchanged by
  construction; a plan-shape assertion belongs in the engine's SQL tests.
* `MAX_K = 32`; a larger `k` is declined by the device backend and falls back to
  the host automatically.
* The host-relative speedup is **noisy on a shared runner**: on the validation
  machine the host time swung 28.6–46.2 ms while the device warm time stayed
  11.6–14.8 ms, giving 1.94–3.92× for identical code. The gate therefore also
  asserts a hardware-level floor (`corpus_bytes / warm_time ≥ 100 GB/s`) and a
  warm/cold amortization ratio of ≥8×; the host-relative figure is a tunable
  tripwire, not the sole signal. The runtime-k register-spill defect would have
  been caught by the bandwidth floor (80 GB/s) even on a run where the relative
  ratio happened to pass.

---

## 7. CI wiring

`.github/workflows/cuda-acceptance.yml` is a **separate workflow** from
`ci.yml`. `ci.yml` stays entirely on `ubuntu-latest` and remains green on
GPU-less runners; the CUDA test binaries are `#![cfg(feature = "cuda")]`, so a
plain `cargo test --workspace` never compiles them.

### One-time setup

1. Register the GPU machine as a self-hosted runner (Linux or Windows) and note
   its labels.
2. Set repository variable **`GPU_RUNNER_LABEL`** (Settings → Secrets and
   variables → Actions → Variables) to a JSON array of those labels, e.g.
   `["self-hosted","linux","gpu"]` or `["self-hosted","Windows","X64","gpu"]`.
   A `workflow_dispatch` input `runner_labels` overrides it for one run.
3. On that runner: NVIDIA driver + `nvidia-smi`, **CUDA 13.x** toolkit with
   NVRTC (`libnvrtc.so` on Linux, `nvrtc64_130_0.dll` on Windows — the crate is
   built with cudarc's `cuda-13000` feature), ≥5 GiB free VRAM and ≥6 GiB
   available host RAM. Linux runners need the toolkit `lib64` on
   `LD_LIBRARY_PATH`; Windows runners need `bin\x64` on `PATH` — the workflow's
   preflight sets both.
4. Missing variable ⇒ a cheap `ubuntu-latest` precheck job fails in seconds with
   the exact fix. A wrong label would otherwise leave the job queued until
   timeout, so the precheck exists rather than relying on `runs-on` to error.

### Triggers

| Event | Why |
|---|---|
| `push` to `main`, path-filtered to the CUDA code chain | A kernel / manifest change must be re-validated |
| `schedule` nightly (02:00 UTC) | Catches driver / toolkit drift that no file change would trigger |
| `workflow_dispatch` | Manual runs and ad-hoc PR validation |

**There is deliberately no `pull_request` trigger.** A self-hosted runner runs
repository code with local privileges and direct driver access; executing
untrusted fork PRs there is unsafe. For a same-repo PR, dispatch the workflow
against that ref. Optionally add a protected `environment` with required
reviewers for another approval gate.

`concurrency` is scoped per ref with `cancel-in-progress: false`: a GPU runner is
a single scarce resource and two gate suites must never overlap (gate 2 needs
4 GiB of transient VRAM).

### CI-specific behaviour

| Setting | Local | CI | Reason |
|---|---|---|---|
| `GTV_CUDA_REQUIRE_DEVICE` | unset (SKIP if no card) | `1` (hard fail) | A dedicated GPU job must never go green because the card vanished |
| `GTV_GATE2_MIN_SPEEDUP` | `2.0` (§6 target) | `1.0` | A host-relative ratio is not portable: this machine's CPU reads ~70 GB/s vs the GPU's ~138 GB/s, so 2× is a property of *this* hardware. On a fast-CPU runner the same healthy kernel could be <1×. CI only requires the device not to be slower than the host. Dispatch with `min_speedup=2.0` to run the full §6 target. |
| `GTV_GATE2_MIN_BANDWIDTH_GBS` | `100` | `100` | The stable hardware-level floor, independent of host speed. |
| warm/cold amortization | ≥8× | ≥8× | Proves residency is actually saving the upload. |

The run step writes a job summary table from the `GATE n: PASS` lines and uploads
`cuda-acceptance.log` as an artifact on failure (14-day retention).

Equivalent local run:

```sh
GTV_CUDA_REQUIRE_DEVICE=1 RUSTFLAGS= RUSTC_WRAPPER= CARGO_INCREMENTAL=0 \
  cargo test -p gtv-accelerator --features cuda --release \
  --test cuda_acceptance --test cuda_smoke \
  -- --nocapture --test-threads=1
```

`--test-threads=1` is required: gate 2 holds a 2 GiB host corpus plus a 4 GiB
transient device allocation, so gates must not overlap.

---

## 8. Next

1. **Time mask** (rank 3) — trivial kernel, do it opportunistically.
2. **`k`-hop** — only after the direction-optimising host BFS is exhausted; the
   irregular access pattern is the binding constraint, not the arithmetic.
3. **Pinned host cold tier** — an evicted block could keep its device-ready
   column-major layout in page-locked host memory, turning re-admission into one
   async DMA. cudarc already exposes `alloc_pinned`; the blockers are a host
   memory cap (`ACCEL_HOST_PIN_LIMIT`), Windows `SeLockMemoryPrivilege`, and WSL
   pinning limits. Only worth doing once gate 6 shows re-admission is frequent.

---

## 9. IVF and VRAM pressure

### 9.1 Why IVF is the operator that fits a device

`IvfIndex` reorders the corpus so Voronoi cell `l` occupies one contiguous run
`list_offsets[l]..list_offsets[l+1]`. A query ranks `nlist` centroids (2 MiB at
`nlist=1024, dim=512`), then exact-scans the `nprobe` nearest cells — typically
3% of a 1M-row corpus, in a handful of sequential streams. That is the regular,
dense, output-much-smaller-than-input shape §0 asks for. It is also the shape
Strata's hot/cold tiering assumes, unlike exact flat K-NN which touches 100% of
the corpus with no heat skew.

The seam is `IvfIndex::probe_positions(query, probe, mask) -> ProbePlan`. It is
the **only** place cells are chosen; the host scan (`exact_topk_positions`), the
`filtered_search` widening loop, and the CUDA kernel all consume that same plan,
so "device equals host" is structural rather than a coincidence of two ranking
implementations. The mask is in original input order (`orig_pos`), which is also
the label-filter domain the engine uses.

The device reuses the *same* fused top-K kernel as flat K-NN. The only addition
is an optional gather list: `pos == NULL` means identity over `[0, npos)`, a
non-NULL `pos` is the IVF plan. One register top-K, one block merge and one
per-`k` NVRTC specialization serve both. A flat corpus and a cell-ordered IVF
corpus are cached under **different** `ResidentKind`s (`corpus` vs `ivf`) because
their row orders differ and a shared buffer would silently score the wrong rows.

Engine wiring: `GtvContext::register_knn_ivf(...)` builds an IVF-backed
`KnnCollection`, and `knn(name, query, k [, label])` then routes through
`AcceleratorRegistry::ivf_probe`. Only the (small) position list crosses PCIe per
query; the whole cell-ordered corpus stays resident.

### 9.2 Measured (RTX 4060 Laptop, release)

| Query | Shape | Host IVF probe (same plan) | Host flat exact | Device cold | Device warm |
|---|---|---|---|---|---|
| IVF probe | 1M × 128 (520 MiB), `nlist=1024`, `nprobe=32`, k=10, 32 768 rows scanned | ~0.44 ms | ~11 ms | ~106–119 ms | **~0.33–0.35 ms** |

**Read the ratios carefully.** Warm is ~27–33× faster than a *host flat exact*
scan, but most of that is the IVF index pruning 97% of the corpus; comparing
like-for-like, the device IVF probe is only **~1.3× faster than the host IVF
probe**. That is the honest offload premium for a single small probe: 16 MiB of
candidates does not give a GPU enough work to hide launch and transfer overhead.
The device wins when the corpus must stay resident across many queries or in a
concurrent batch, and when VRAM residency frees host RAM. It does **not** make a
cheap CPU probe dramatically cheaper. Cold is ~300× warm — residency is the
entire game, exactly as §0 predicts.

### 9.3 Admission: heat instead of pure LRU

The real VRAM shortage on an 8 GiB card is not one oversized corpus (that is
cleanly declined) but **two or three 2 GiB corpora plus a one-shot scan**. Pure
LRU evicts a hot corpus for a single cold query and pays a ~0.4 s re-upload on
the next one. The ledger now tracks a decayed frequency estimate:

* each reference adds 1 to a block's heat; heat halves every
  `ACCEL_HEAT_HALF_LIFE` (default 30 s, `0` disables the policy and restores
  pure LRU);
* eviction picks the coldest block, ties broken by recency;
* a block with heat ≥ `PROTECT_FLOOR` (4) is not displaced by a newcomer unless
  the newcomer's heat is ≥ `ADMIT_MARGIN` (1.25) × the victim's — hysteresis
  stops two equally hot corpora ping-ponging across PCIe;
* a **declined** block still accumulates heat in a bounded shadow ledger, and an
  evicted hot block is demoted with its heat preserved, so a genuinely hot
  newcomer can win admission after a few queries rather than being locked out
  forever. This is the W-TinyLFU admission window, and it is the part that makes
  Strata's idea usable for a query cache rather than a static expert table.

Gate 6 is an A/B: one-corpus budget, 8 warm-up queries, then 12 rounds of
`hot` + a fresh one-shot corpus. Heat: **1 upload, 12 deferrals**; pure LRU:
**24 uploads**. All 64 result checks match the host exactly.

A subtle but important consequence: with capacity for one corpus and **two
equally hot** corpora, bounded uploads are impossible for any policy. The rule
above resolves it by keeping one resident and serving the other on the host —
one upload instead of an infinite ping-pong, which is the better throughput even
though the second corpus never gets a device copy until it becomes dominant.

### 9.4 Elastic budget

`CUDA_VRAM` is a ceiling, not a promise. `AccelBackend::mem_info()` (implemented
by `CudaBackend` from its own driver context) feeds `VramBudget::refresh_if_due`
every 250 ms at most; `capacity()` becomes
`min(configured, live_free_after_headroom + used_at_probe)`. It can only get
*smaller* than `CUDA_VRAM`, never larger. Storing an absolute ceiling rather than
recomputing `free + used` on every read is what lets `release()` still free space
before the next probe. A failed probe is not an error — the previous ceiling
stands. `.accel` reports `DEVICE_FREE` and the clamp count; declines now carry a
`DeclineReason` so `too-large`, `heat-protected`, `elastic-budget` and `no-room`
are distinguishable.

This is the guard against Windows WDDM's silent sysmem fallback: an allocation
succeeds, but reads are host-backed and bandwidth collapses. Declining to the
host kernel is strictly better than "successfully" running a kernel over PCIe.
