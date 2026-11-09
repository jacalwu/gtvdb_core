// gtv-knn-flat — exact batched K-NN over a resident vector matrix.
//
// One block per (query, split); each of the eight warps in a block keeps its
// own sorted top-k in shared memory, and the block merges them once at the end.
// The hierarchy matters: a per-*thread* heap would need k registers per thread
// and a per-*block* heap would serialise every insertion behind one lock. Eight
// warps is the middle — eight independent heaps, each serialised inside its own
// warp by `__ballot_sync` + leader election, no lock at all.
//
// Why the split dimension: a single block scanning a million rows uses one SM.
// The host picks `gridDim.y` from the SM count and the batch size, so a batch of
// one still fills the card; the partial lists it leaves behind are merged on the
// host, where eight sorted lists of k cost nothing next to a 51 MB scan.
//
// Determinism: ties are broken by id ascending, exactly like the CPU reference
// (`cmp_scored` in gtv-index). Without that, two rows at the same distance could
// come back in either order and "top-k parity" would be a flaky test rather than
// an invariant.
//
// Numbers: see docs/cuda-acceleration-plan.md §9.7. This file is compiled to
// PTX by `cargo xtask cuda-ptx` and embedded; nothing here runs at build time on
// a machine without a toolkit.

typedef unsigned int u32;
typedef unsigned long long u64;

#define GTV_WARPS 8
#define GTV_BLOCK (GTV_WARPS * 32)

// Metric codes, matching `gtv_accelerator::Metric` order.
#define GTV_L2 0
#define GTV_COSINE 1
#define GTV_IP 2

// Same value the host enforces; a block's heaps are `GTV_WARPS * k` entries of
// 12 bytes, so k is bounded by shared memory rather than by arithmetic.
#define GTV_MAX_K 256

// Squared L2.
//
// The float4 path is gated on `dim % 4 == 0`, not on `i + 4 <= dim`: a row starts
// at `row * dim * 4` bytes, so a vector load at the head of a row is only
// 16-byte aligned when `dim` is a multiple of four. Loading a float4 from an
// address that is merely 4-byte aligned is undefined behaviour, and it would
// fault on the *tall* rows only — the worst kind of bug to find later.
__device__ __forceinline__ float gtv_l2(const float* __restrict__ a,
                                        const float* __restrict__ b, int dim) {
    float s = 0.0f;
    if ((dim & 3) == 0) {
        for (int i = 0; i < dim; i += 4) {
            const float4 x = *reinterpret_cast<const float4*>(a + i);
            const float4 y = *reinterpret_cast<const float4*>(b + i);
            const float d0 = x.x - y.x;
            const float d1 = x.y - y.y;
            const float d2 = x.z - y.z;
            const float d3 = x.w - y.w;
            s += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }
    } else {
        for (int i = 0; i < dim; ++i) {
            const float d = a[i] - b[i];
            s += d * d;
        }
    }
    return s;
}

__device__ __forceinline__ float gtv_dot(const float* __restrict__ a,
                                         const float* __restrict__ b, int dim) {
    float s = 0.0f;
    if ((dim & 3) == 0) {
        for (int i = 0; i < dim; i += 4) {
            const float4 x = *reinterpret_cast<const float4*>(a + i);
            const float4 y = *reinterpret_cast<const float4*>(b + i);
            s += x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
        }
    } else {
        for (int i = 0; i < dim; ++i) {
            s += a[i] * b[i];
        }
    }
    return s;
}

// The CPU's `metric_distance`: L2 is already the distance, cosine rows are
// unit-normalised so `1 - dot` is the distance, inner product is negated to keep
// "lower is closer" for every metric.
__device__ __forceinline__ float gtv_dist(const float* __restrict__ q,
                                          const float* __restrict__ row, int dim,
                                          int metric) {
    if (metric == GTV_L2) {
        return gtv_l2(q, row, dim);
    }
    const float d = gtv_dot(q, row, dim);
    return (metric == GTV_COSINE) ? 1.0f - d : -d;
}

// Insert into an ascending-by-(distance, id) list of at most `k` entries,
// dropping the last. The caller has already established that the pair beats the
// current worst, so this only has to find its place.
__device__ __forceinline__ void gtv_insert(float* dist, u64* id, int* count, int k,
                                           float d, u64 oid) {
    int n = *count;
    if (n < k) {
        *count = n + 1;
    } else {
        n = k - 1;
    }
    int i = n;
    while (i > 0 && (dist[i - 1] > d || (dist[i - 1] == d && id[i - 1] > oid))) {
        dist[i] = dist[i - 1];
        id[i] = id[i - 1];
        --i;
    }
    dist[i] = d;
    id[i] = oid;
}

extern "C" __global__ void gtv_knn_flat(
    const float* __restrict__ corpus,   // [rows][dim]
    const u64* __restrict__ ids,        // [rows]
    long long rows,
    const float* __restrict__ queries,  // [batch][dim]
    int dim,
    int k,
    int metric,
    const u32* __restrict__ filter,     // ceil(rows/32) words, or null
    float* __restrict__ out_dist,       // [batch][split][k]
    u64* __restrict__ out_id,           // [batch][split][k]
    int* __restrict__ out_count)        // [batch][split]
{
    extern __shared__ char smem[];
    // Ids first: the dynamic-shared base is 16-byte aligned, so the u64 view is
    // aligned for every k, while the float view's offset stays 8-byte aligned.
    u64* s_id = reinterpret_cast<u64*>(smem);
    float* s_dist = reinterpret_cast<float*>(smem + (size_t)GTV_WARPS * k * sizeof(u64));
    __shared__ int s_count[GTV_WARPS];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    u64* w_id = s_id + (size_t)warp * k;
    float* w_dist = s_dist + (size_t)warp * k;

    if (threadIdx.x < GTV_WARPS) {
        s_count[threadIdx.x] = 0;
    }
    __syncthreads();

    const int query = blockIdx.x;
    const int split = blockIdx.y;
    const float* q = queries + (size_t)query * dim;

    // Every thread runs the same number of iterations, with `active` folded into
    // the candidate test: `__ballot_sync` needs the whole warp present, so a
    // `continue` on the ragged last tile would be undefined behaviour.
    const long long stride = (long long)GTV_BLOCK * gridDim.y;
    for (long long base = (long long)split * GTV_BLOCK; base < rows; base += stride) {
        const long long r = base + threadIdx.x;
        const bool active = r < rows;
        bool cand = active;
        float d = 0.0f;
        u64 oid = 0;
        if (active) {
            // A filtered-out row is not a candidate, but it still has to take
            // part in the ballot — hence the flag rather than a branch out.
            if (filter != 0) {
                cand = ((filter[r >> 5] >> (r & 31)) & 1u) != 0u;
            }
            d = gtv_dist(q, corpus + r * (long long)dim, dim, metric);
            oid = ids[r];
        }

        int n = s_count[warp];
        if (cand) {
            cand = (n < k) || (d < w_dist[k - 1]) ||
                   (d == w_dist[k - 1] && oid < w_id[k - 1]);
        }

        u32 m = __ballot_sync(0xffffffffu, cand);
        while (m != 0u) {
            const int leader = __ffs((int)m) - 1;
            if (lane == leader) {
                // Re-test after the barrier: an insertion this round lowers the
                // worst entry, and the pre-ballot test may have used the old
                // one. Dropping the candidate here is correct — it lost to a
                // pair inserted ahead of it in the same round.
                const int n2 = s_count[warp];
                if ((n2 < k) || (d < w_dist[k - 1]) ||
                    (d == w_dist[k - 1] && oid < w_id[k - 1])) {
                    gtv_insert(w_dist, w_id, &s_count[warp], k, d, oid);
                }
            }
            __syncwarp();
            m &= m - 1;
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        int cur[GTV_WARPS];
#pragma unroll
        for (int w = 0; w < GTV_WARPS; ++w) {
            cur[w] = 0;
        }
        const size_t slot = ((size_t)query * gridDim.y + split) * (size_t)k;
        int outn = 0;
        for (int j = 0; j < k; ++j) {
            int best = -1;
            float bd = 0.0f;
            u64 bi = 0;
#pragma unroll
            for (int w = 0; w < GTV_WARPS; ++w) {
                if (cur[w] >= s_count[w]) {
                    continue;
                }
                const float d = s_dist[(size_t)w * k + cur[w]];
                const u64 oid = s_id[(size_t)w * k + cur[w]];
                if (best < 0 || d < bd || (d == bd && oid < bi)) {
                    best = w;
                    bd = d;
                    bi = oid;
                }
            }
            if (best < 0) {
                break;
            }
            cur[best]++;
            out_dist[slot + outn] = bd;
            out_id[slot + outn] = bi;
            outn++;
        }
        out_count[(size_t)query * gridDim.y + split] = outn;
    }
}
