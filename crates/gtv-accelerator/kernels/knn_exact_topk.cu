// knn_exact_topk.cu — exact top-K over a *device-resident* f32 corpus.
//
// Layout: the corpus is uploaded **column-major** (`data_t[d * n + i]`) via the
// transpose kernel below. For a fixed dimension the 32 lanes of a warp then
// read 32 consecutive floats — one coalesced 128 B transaction — instead of
// addresses `dim * 4` bytes apart, which on a row-major layout would spend a
// whole transaction per lane. The transpose is paid once at upload.
//
// Fusion: squared distance and the top-K reduction live in the same kernel, so
// the N intermediate distances never reach memory. Each thread keeps a
// register-resident candidate array; a block merges its threads' candidates
// through shared memory; only `num_blocks * k` candidates leave the device.
//
// KERNEL_K IS COMPILE-TIME. The host prepends `#define KERNEL_K <k>` and
// compiles one specialization per distinct k (cached). With a runtime `k`, the
// `ld[K]` / `li[K]` arrays cannot stay in registers — nvcc spills them to local
// memory, which on a 1M-thread scan costs more than the distance computation.
// Measured on an RTX 4060 Laptop: runtime-k was 25.4 ms for 1M x 512, k=10.
//
// Metric: 0 = squared L2, 1 = 1 - dot (cosine on normalized rows), 2 = -dot.
// The host normalizes rows for cosine at index build time, so `1 - dot` ordering
// is identical to `-dot`; carrying the constant keeps the reported distance
// consistent with the host path.

#ifndef KERNEL_K
#define KERNEL_K 10
#endif

#define MAX_K 32
// 128 threads keeps the block-merge staging area at 128 x KERNEL_K x 2 x 4 B.
// At the largest KERNEL_K=32 that is 32 KiB of *static* shared memory, inside
// the 48 KiB default per-block limit on sm_75..sm_89. 256 threads would need
// 64 KiB and fail to launch without a driver opt-in.
#define THREADS 128
#define TILE 32
#define F_INF 3.402823466e38F

#if KERNEL_K > MAX_K
#error "KERNEL_K exceeds MAX_K"
#endif

// Insert into an ascending register array (0 = best). KERNEL_K is a compile-time
// constant, so the shift loop unrolls and the array stays in registers.
__device__ __forceinline__ void topk_insert(float d, int id, float* dist, int* ids) {
    if (d >= dist[KERNEL_K - 1]) return;
    int j = KERNEL_K - 1;
    #pragma unroll
    for (int s = 0; s < KERNEL_K - 1; ++s) {
        if (j <= 0 || dist[j - 1] <= d) break;
        dist[j] = dist[j - 1];
        ids[j] = ids[j - 1];
        --j;
    }
    dist[j] = d;
    ids[j] = id;
}

// Row-major -> column-major transpose, executed ON the device.
//
// Uploading the corpus as one bulk DMA (row-major) and transposing here costs
// one H2D transfer plus one device read/write pass. Doing the transpose on the
// host instead would need either a second n*dim staging copy or `dim` strided
// passes over the whole corpus — at 1M x 512 that is ~1 TB of host traffic for
// a 2 GB corpus. The TILE+1 padding avoids shared-memory bank conflicts.
//
// `in[v * dim + d]` becomes `out[d * n + v]`. Grid is flattened in x:
// tile_x indexes dimensions, tile_y indexes vectors.
extern "C" __global__ void transpose_rowmajor_to_colmajor(
    const float* __restrict__ in,   // n * dim, row-major
    float* __restrict__ out,        // dim * n, column-major
    int n,
    int dim,
    int tiles_x)
{
    __shared__ float tile[TILE][TILE + 1];

    const int tile_x = blockIdx.x % tiles_x;
    const int tile_y = blockIdx.x / tiles_x;

    const int v = tile_y * TILE + threadIdx.y;
    const int d = tile_x * TILE + threadIdx.x;
    if (v < n && d < dim) {
        tile[threadIdx.y][threadIdx.x] = in[v * dim + d];
    }
    __syncthreads();

    const int d2 = tile_x * TILE + threadIdx.y;
    const int v2 = tile_y * TILE + threadIdx.x;
    if (d2 < dim && v2 < n) {
        out[d2 * n + v2] = tile[threadIdx.x][threadIdx.y];
    }
}

// Windowed variant for the P1c block pipeline: transposes one row-block of a
// corpus into its final position of the *full* column-major matrix, so blocks
// can be uploaded and transposed one at a time (copy of block N+1 overlapping
// this kernel on block N) without a second full-size pass.
//
// `in` holds `n_in * dim` row-major floats (the block); `out` is the full
// `dim * n_total` column-major corpus and only rows `[row0, row0 + n_in)` are
// written: `in[v * dim + d]` becomes `out[d * n_total + row0 + v]`.
// `tiles_x` counts dimension tiles over `dim`; tiles over rows use `n_in`.
extern "C" __global__ void transpose_rowmajor_to_colmajor_window(
    const float* __restrict__ in,   // n_in * dim, row-major
    float* __restrict__ out,        // dim * n_total, column-major
    int n_total,
    int dim,
    int tiles_x,
    int row0,
    int n_in)
{
    __shared__ float tile[TILE][TILE + 1];

    const int tile_x = blockIdx.x % tiles_x;
    const int tile_y = blockIdx.x / tiles_x;

    const int v = tile_y * TILE + threadIdx.y;
    const int d = tile_x * TILE + threadIdx.x;
    if (v < n_in && d < dim) {
        tile[threadIdx.y][threadIdx.x] = in[v * dim + d];
    }
    __syncthreads();

    const int d2 = tile_x * TILE + threadIdx.y;
    const int v2 = tile_y * TILE + threadIdx.x;
    if (d2 < dim && v2 < n_in) {
        out[d2 * n_total + row0 + v2] = tile[threadIdx.x][threadIdx.y];
    }
}

// One kernel serves BOTH the flat scan and the IVF probe. `pos` is an optional
// gather list of corpus row indices: NULL means "identity over [0, npos)",
// which is the flat scan; non-NULL is an IVF probe plan (ascending contiguous
// runs, so coalescing inside a cell is preserved). Sharing the kernel matters:
// the register top-K, the block merge and the distance loop then have exactly
// one implementation, and the per-k NVRTC specialization benefits both.
extern "C" __global__ void knn_exact_topk_kernel(
    const float* __restrict__ data_t,        // column-major, n * dim floats
    const float* __restrict__ query,         // dim floats
    const unsigned char* __restrict__ mask,  // read only when has_mask != 0
    int has_mask,
    int n,                                   // column stride: total corpus rows
    int npos,                                // number of items to iterate
    const int* __restrict__ pos,             // NULL => flat; else IVF positions
    int dim,
    int metric,
    int* __restrict__ out_id,                // num_blocks * KERNEL_K
    float* __restrict__ out_dist)            // num_blocks * KERNEL_K
{
    __shared__ float s_d[THREADS * KERNEL_K];
    __shared__ int s_i[THREADS * KERNEL_K];

    float ld[KERNEL_K];
    int li[KERNEL_K];
    #pragma unroll
    for (int j = 0; j < KERNEL_K; ++j) {
        ld[j] = F_INF;
        li[j] = -1;
    }

    const int stride = gridDim.x * blockDim.x;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < npos; t += stride) {
        const int idx = (pos != 0) ? pos[t] : t;
        // A malformed plan is skipped, never read out of bounds. For the IVF
        // path the mask was already applied on the host (has_mask == 0).
        if (idx < 0 || idx >= n) continue;
        if (has_mask != 0 && !mask[idx]) continue;

        float acc = 0.0f;
        const float* col = data_t + idx;
        if (metric == 0) {
            #pragma unroll 4
            for (int d = 0; d < dim; ++d) {
                const float diff = *col - query[d];
                acc += diff * diff;
                col += n;  // strength-reduced stride to the next dimension
            }
        } else {
            float dot = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < dim; ++d) {
                dot += *col * query[d];
                col += n;
            }
            acc = (metric == 1) ? (1.0f - dot) : -dot;
        }
        topk_insert(acc, idx, ld, li);
    }

    #pragma unroll
    for (int j = 0; j < KERNEL_K; ++j) {
        s_d[threadIdx.x * KERNEL_K + j] = ld[j];
        s_i[threadIdx.x * KERNEL_K + j] = li[j];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float bd[KERNEL_K];
        int bi[KERNEL_K];
        #pragma unroll
        for (int j = 0; j < KERNEL_K; ++j) {
            bd[j] = F_INF;
            bi[j] = -1;
        }
        for (int t = 0; t < THREADS; ++t) {
            #pragma unroll
            for (int j = 0; j < KERNEL_K; ++j) {
                topk_insert(s_d[t * KERNEL_K + j], s_i[t * KERNEL_K + j], bd, bi);
            }
        }
        #pragma unroll
        for (int j = 0; j < KERNEL_K; ++j) {
            out_id[blockIdx.x * KERNEL_K + j] = bi[j];
            out_dist[blockIdx.x * KERNEL_K + j] = bd[j];
        }
    }
}
