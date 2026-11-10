// gtv-window-scan -- trailing-window aggregate over one f64 column.
//
// One block per chunk of GTV_CHUNK output rows. A block loads its own chunk
// *plus* the `window-1` values before it, because the first few windows of a
// chunk reach backwards; that halo is why the whole operator needs no global
// prefix array and no cross-block communication at all.
//
// Two shapes inside the block, because the five aggregates split into two
// families that want different things:
//
//   * sum / mean / count -- a block-wide inclusive scan of the loaded values,
//     then `out[k] = P[end] - P[start-1]`. Fully parallel, bandwidth bound.
//   * min / max -- a monotonic deque walked by one thread, because a windowed
//     extremum cannot be recovered by subtracting two prefixes (it is not
//     invertible). One thread per block, but 70-odd blocks in flight, which is
//     what makes it fast: the parallelism is across chunks, not inside one.
//
// Numerics, stated plainly because this is where the parity contract lives:
//
//   * The CPU reference (`gtv_array::window::msum`) is a *running* recurrence
//     over the whole column: `s += v[i]; if (i >= w) s -= v[i-w]`. Its rounding
//     error drifts with n. The device computes each window as a difference of
//     two prefixes that restart every chunk, so it is closer to the exact sum
//     than the reference is. The two can therefore never be bit-identical, and
//     the contract is a tolerance (`WINDOW_TOLERANCE` on the host side), not
//     equality. For min/max/count the device *is* exact.
//   * min/max and NaN: the CPU fold starts at the window's first element and
//     then only replaces it when `v < acc`, so a NaN is sticky only when it is
//     the first element and is ignored everywhere else. The deque reproduces
//     that exactly: NaN never enters the deque, and an output whose window
//     starts on a NaN returns that NaN.
//   * sum/mean and NaN: ordinary IEEE arithmetic, so NaN propagates. That
//     differs from the fold's positional rule, and it is the one place where
//     the two shapes do not agree with the reference for the same op family.
//     Documented rather than papered over: a column with NaN in it should be
//     aggregated with Sum if the caller needs a defined answer.
//
// Built to PTX by `cargo xtask cuda-ptx` and embedded; nothing here runs at
// build time on a machine without a toolkit. Numbers: see
// docs/cuda-acceleration-plan.md.

typedef long long i64;

#define GTV_BLOCK 256
#define GTV_CHUNK 2048
#define GTV_MAX_WINDOW 1024

// Window ops, matching `gtv_accelerator::WindowOp` order.
#define GTV_SUM 0
#define GTV_MEAN 1
#define GTV_MIN 2
#define GTV_MAX 3
#define GTV_COUNT 4

__device__ __forceinline__ bool gtv_is_nan(double x) { return x != x; }

__device__ __forceinline__ i64 gtv_min_i64(i64 a, i64 b) { return a < b ? a : b; }

extern "C" __global__ void gtv_window_scan(const double* __restrict__ v,
                                           i64 n,
                                           int window,
                                           int op,
                                           double* __restrict__ out) {
    extern __shared__ char raw[];
    double* s = reinterpret_cast<double*>(raw);

    const i64 base = (i64)blockIdx.x * (i64)GTV_CHUNK;
    if (base >= n) {
        return;
    }
    const i64 len = gtv_min_i64((i64)GTV_CHUNK, n - base);
    const i64 w = (i64)window;
    // How many values before `base` this block needs: `window-1`, or fewer when
    // the column starts inside the window (the trailing aggregate then covers
    // only what exists, which is the kdb convention the CPU reference encodes).
    const i64 pre = gtv_min_i64(w - 1, base);
    const i64 m = pre + len;

    if (op == GTV_COUNT) {
        // No memory, no scan: the count is `min(i+1, w)` by definition. Handled
        // before the load because this is the one op launched with no shared
        // memory reserved, and writing `s[j]` there is an illegal access.
        for (i64 j = threadIdx.x; j < len; j += blockDim.x) {
            const i64 i = base + j;
            out[base + j] = (double)gtv_min_i64(i + 1, w);
        }
        return;
    }

    for (i64 j = threadIdx.x; j < m; j += blockDim.x) {
        s[j] = v[base - pre + j];
    }
    __syncthreads();

    if (op == GTV_MIN || op == GTV_MAX) {
        int* dq = reinterpret_cast<int*>(raw + ((size_t)m * sizeof(double) + 7) / 8 * 8);
        if (threadIdx.x == 0) {
            int head = 0;
            int tail = 0;
            for (i64 j = 0; j < m; ++j) {
                const double x = s[j];
                if (!gtv_is_nan(x)) {
                    if (op == GTV_MIN) {
                        // Pop the back while it can no longer be the extremum:
                        // a later value that is <= it dominates it forever.
                        while (tail > head && !(s[dq[tail - 1]] < x)) {
                            --tail;
                        }
                    } else {
                        while (tail > head && !(s[dq[tail - 1]] > x)) {
                            --tail;
                        }
                    }
                    dq[tail++] = (int)j;
                }
                if (j < pre) {
                    continue;  // halo row: loaded, not an output
                }
                const i64 i = base + (j - pre);
                const i64 start = (i + 1 >= w) ? (i + 1 - w) : 0;
                const i64 sr = start - (base - pre);
                while (head < tail && (i64)dq[head] < sr) {
                    ++head;
                }
                const double first = s[sr];
                if (gtv_is_nan(first)) {
                    // Sticky NaN at the window's first position, matching the
                    // reference fold. `first` itself is the answer.
                    out[i] = first;
                } else {
                    out[i] = tail > head ? s[dq[head]] : first;
                }
            }
        }
        return;
    }

    // Sum / mean: inclusive scan in place, then a difference of two prefixes.
    //
    // The scan is the ordinary two-level one: each thread folds its own
    // contiguous slice, thread 0 folds the 256 slice totals in order, and each
    // thread re-walks its slice adding its exclusive offset. Folding the totals
    // in order (rather than a Hillis-Steele tree) costs 256 serial adds per
    // block and buys the same left-to-right association as a sequential pass,
    // which is what keeps the mean's rounding sane.
    double* totals = s + m;
    const i64 per = (m + GTV_BLOCK - 1) / GTV_BLOCK;
    const i64 lo = (i64)threadIdx.x * per;
    const i64 hi = gtv_min_i64(lo + per, m);

    double acc = 0.0;
    for (i64 j = lo; j < hi; ++j) {
        acc += s[j];
        s[j] = acc;
    }
    totals[threadIdx.x] = acc;
    __syncthreads();

    if (threadIdx.x == 0) {
        double run = 0.0;
        for (int t = 0; t < GTV_BLOCK; ++t) {
            const double keep = totals[t];
            totals[t] = run;
            run += keep;
        }
    }
    __syncthreads();

    const double off = totals[threadIdx.x];
    for (i64 j = lo; j < hi; ++j) {
        s[j] += off;
    }
    __syncthreads();

    for (i64 k = threadIdx.x; k < len; k += blockDim.x) {
        const i64 i = base + k;
        const i64 start = (i + 1 >= w) ? (i + 1 - w) : 0;
        const i64 sr = start - (base - pre);
        const double sum = s[pre + k] - (sr > 0 ? s[sr - 1] : 0.0);
        out[i] = (op == GTV_MEAN) ? sum / (double)gtv_min_i64(i + 1, w) : sum;
    }
}
