// CUDA admission probe for gtvdb (C0 gate).
//
// Measures, on the local device:
//   1. H2D / D2H bandwidth, pageable vs pinned, for a 1e7-edge graph payload
//   2. valid-time filter: CPU scan vs GPU kernel, cold and resident
//   3. 512-d dot product (the arithmetic core of TC4 KNN), coalesced tiling
//
// No project headers, no CUB: this is a hardware cost probe, not a kernel.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

static double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// ---- valid-time filter -----------------------------------------------------
__global__ void k_vfilter(const long long* vf, const long long* vt,
                          long long t, unsigned int n, unsigned long long* out) {
    unsigned long long c = 0;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        c += (vf[i] <= t && t < vt[i]) ? 1ULL : 0ULL;
    }
    if (c) atomicAdd(out, c);
}

static unsigned long long cpu_vfilter(const long long* vf, const long long* vt,
                                      long long t, unsigned int n) {
    unsigned long long c = 0;
    for (unsigned int i = 0; i < n; ++i) c += (vf[i] <= t && t < vt[i]) ? 1ULL : 0ULL;
    return c;
}

// ---- 512-d dot product, coalesced -----------------------------------------
// One block walks vectors with a grid stride; threads cover the 512 dims in two
// coalesced 256-float segments. The naive "one thread per vector" mapping reads
// 2 KiB-strided, which is what makes a KNN kernel look 100x slower than it is.
__global__ void k_dot(const float* __restrict__ vecs, const float* __restrict__ q,
                      unsigned int nvec, int d, float* __restrict__ out) {
    float acc = 0.f;
    for (unsigned int v = blockIdx.x; v < nvec; v += gridDim.x) {
        const float* row = vecs + (size_t)v * d;
        for (int k = threadIdx.x; k < d; k += blockDim.x) acc = fmaf(row[k], q[k], acc);
    }
    __shared__ float sh[256];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}

// AVX2-friendly: four independent accumulators over a contiguous query row.
static volatile float g_cpu_sink = 0.f;

static double cpu_dot(const float* vecs, const float* q, unsigned int nvec, int d,
                      unsigned int nq, double* out_gflops) {
    double t0 = now_ms();
    float sink = 0.f;
    for (unsigned int qi = 0; qi < nq; ++qi) {
        const float* qv = q + (size_t)qi * d;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (unsigned int i = 0; i < nvec; ++i) {
            const float* v = vecs + (size_t)i * d;
            int k = 0;
            for (; k + 4 <= d; k += 4) {
                a0 = fmaf(v[k + 0], qv[k + 0], a0);
                a1 = fmaf(v[k + 1], qv[k + 1], a1);
                a2 = fmaf(v[k + 2], qv[k + 2], a2);
                a3 = fmaf(v[k + 3], qv[k + 3], a3);
            }
            sink += (a0 + a1) + (a2 + a3);
        }
    }
    g_cpu_sink = sink;   // volatile store: without a consumer the loop is dead code
    double ms = now_ms() - t0;
    *out_gflops = 2.0 * (double)nq * nvec * d / (ms * 1e6);
    return ms;
}

int main() {
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, 0));
    std::printf("device: %s  SMs=%d  VRAM=%.1f GiB  cc=%d.%d  bus=%d-bit  l2=%.1f MiB\n\n",
                p.name, p.multiProcessorCount, p.totalGlobalMem / 1073741824.0,
                p.major, p.minor, p.memoryBusWidth, p.l2CacheSize / 1048576.0);

    const unsigned int E = 10'000'000;              // 1e7 edges
    const size_t bytes = (size_t)E * 8;             // one int64 column

    // ---- 1. transfer -------------------------------------------------------
    std::printf("[1] transfer, 1e7 edges x 2 int64 columns = %.0f MiB\n", 2 * bytes / 1048576.0);
    long long* h_vf = (long long*)malloc(bytes);
    long long* h_vt = (long long*)malloc(bytes);
    for (unsigned int i = 0; i < E; ++i) {
        h_vf[i] = (long long)i * 7;
        h_vt[i] = h_vf[i] + 1000;
    }
    long long *d_vf, *d_vt;
    CK(cudaMalloc(&d_vf, bytes));
    CK(cudaMalloc(&d_vt, bytes));

    CK(cudaMemcpy(d_vf, h_vf, bytes, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    double t0 = now_ms();
    for (int r = 0; r < 3; ++r) {
        CK(cudaMemcpy(d_vf, h_vf, bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_vt, h_vt, bytes, cudaMemcpyHostToDevice));
    }
    CK(cudaDeviceSynchronize());
    double page_ms = (now_ms() - t0) / 3.0;

    long long* h_pin_vf; long long* h_pin_vt;
    CK(cudaHostAlloc(&h_pin_vf, bytes, cudaHostAllocDefault));
    CK(cudaHostAlloc(&h_pin_vt, bytes, cudaHostAllocDefault));
    memcpy(h_pin_vf, h_vf, bytes);
    memcpy(h_pin_vt, h_vt, bytes);
    CK(cudaMemcpy(d_vf, h_pin_vf, bytes, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    t0 = now_ms();
    for (int r = 0; r < 3; ++r) {
        CK(cudaMemcpy(d_vf, h_pin_vf, bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_vt, h_pin_vt, bytes, cudaMemcpyHostToDevice));
    }
    CK(cudaDeviceSynchronize());
    double pin_ms = (now_ms() - t0) / 3.0;
    // single-shot, this is what a cold query actually pays
    double pin_one = pin_ms / 2.0;

    t0 = now_ms();
    for (int r = 0; r < 3; ++r) {
        CK(cudaMemcpy(h_vf, d_vf, bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_vt, d_vt, bytes, cudaMemcpyDeviceToHost));
    }
    CK(cudaDeviceSynchronize());
    double d2h_ms = (now_ms() - t0) / 3.0;

    std::printf("  H2D pageable : %8.3f ms  %6.1f GB/s\n", page_ms, 2 * bytes / (page_ms * 1e6));
    std::printf("  H2D pinned   : %8.3f ms  %6.1f GB/s  (one column pair)\n",
                pin_ms, 2 * bytes / (pin_ms * 1e6));
    std::printf("  D2H          : %8.3f ms  %6.1f GB/s\n\n", d2h_ms, 2 * bytes / (d2h_ms * 1e6));

    // ---- 2. valid-time filter ---------------------------------------------
    std::printf("[2] valid-time filter over %u edges\n", E);
    const long long t = 35'000'000LL;   // inside the range: each edge is active for 1000 of every 7000 ticks
    double cpu_best = 1e9;
    unsigned long long cpu_n = 0, gpu_n = 0;
    for (int r = 0; r < 5; ++r) {
        double c0 = now_ms();
        cpu_n = cpu_vfilter(h_vf, h_vt, t, E);
        cpu_best = std::min(cpu_best, now_ms() - c0);
    }
    std::printf("  cpu scan     : %8.3f ms  %6.2f Gedges/s  active=%llu\n",
                cpu_best, E / (cpu_best * 1e6), cpu_n);

    unsigned long long* d_cnt;
    CK(cudaMalloc(&d_cnt, sizeof(*d_cnt)));
    int threads = 256, blocks = std::min(4096u, (E + threads - 1) / threads);
    double gpu_best = 1e9;
    cudaEvent_t ev0, ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
    for (int r = 0; r < 5; ++r) {
        CK(cudaMemset(d_cnt, 0, sizeof(*d_cnt)));
        CK(cudaEventRecord(ev0));
        k_vfilter<<<blocks, threads>>>(d_vf, d_vt, t, E, d_cnt);
        CK(cudaEventRecord(ev1));
        CK(cudaEventSynchronize(ev1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, ev0, ev1));
        gpu_best = std::min(gpu_best, (double)ms);
    }
    CK(cudaMemcpy(&gpu_n, d_cnt, sizeof(gpu_n), cudaMemcpyDeviceToHost));
    std::printf("  gpu kernel   : %8.3f ms  %6.2f Gedges/s  active=%llu  parity=%s\n",
                gpu_best, E / (gpu_best * 1e6), gpu_n, (gpu_n == cpu_n ? "ok" : "MISMATCH"));
    std::printf("  cold (H2D + kernel) : %8.3f ms  (transfer %.0f%%)\n",
                pin_ms + gpu_best, 100.0 * pin_ms / (pin_ms + gpu_best));
    std::printf("  resident break-even : %.1f repeated queries on the same snapshot\n\n",
                pin_ms / std::max(1e-9, cpu_best - gpu_best));

    // ---- 3. 512-d dot product ---------------------------------------------
    const unsigned int NV = 100'000, D = 512, NQ = 1000;
    size_t vbytes = (size_t)NV * D * sizeof(float);
    float* h_vecs = (float*)malloc(vbytes);
    float* h_q = (float*)malloc((size_t)NQ * D * sizeof(float));
    for (size_t i = 0; i < (size_t)NV * D; ++i)
        h_vecs[i] = (float)((i * 2654435761u) % 1000) / 1000.0f;
    for (size_t i = 0; i < (size_t)NQ * D; ++i)
        h_q[i] = (float)((i * 40503u) % 997) / 997.0f;

    std::printf("[3] 512-d dot product: %u vectors x %u queries = %.1f GFLOP (%.0f MiB corpus)\n",
                NV, NQ, 2.0 * NV * NQ * D / 1e9, vbytes / 1048576.0);
    double cpu_gf = 0;
    double cpu_ms = cpu_dot(h_vecs, h_q, NV, D, NQ, &cpu_gf);
    std::printf("  cpu          : %8.1f ms  %6.1f GFLOP/s\n", cpu_ms, cpu_gf);

    float *d_vecs, *d_q, *d_sum;
    CK(cudaMalloc(&d_vecs, vbytes));
    CK(cudaMalloc(&d_q, (size_t)NQ * D * sizeof(float)));
    CK(cudaMalloc(&d_sum, sizeof(float)));
    CK(cudaMemcpy(d_vecs, h_vecs, vbytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_q, h_q, (size_t)NQ * D * sizeof(float), cudaMemcpyHostToDevice));

    // one launch per query, 1024 blocks of 256 threads, grid-stride over vectors
    const int gblocks = 1024;
    auto run_all_queries = [&]() {
        for (unsigned int qi = 0; qi < NQ; ++qi) {
            k_dot<<<gblocks, threads>>>(d_vecs, d_q + (size_t)qi * D, NV, D, d_sum);
        }
    };
    CK(cudaMemset(d_sum, 0, sizeof(float)));
    run_all_queries();
    CK(cudaDeviceSynchronize());

    double gpu_ms = 1e9;
    for (int r = 0; r < 3; ++r) {
        CK(cudaDeviceSynchronize());
        double c0 = now_ms();
        run_all_queries();
        CK(cudaDeviceSynchronize());
        gpu_ms = std::min(gpu_ms, now_ms() - c0);
    }
    double gpu_gf = 2.0 * NV * NQ * D / (gpu_ms * 1e6);
    double h2d_ms = pin_ms * (double)vbytes / (double)(2 * bytes);   // scale the measured 2-column transfer
    std::printf("  gpu          : %8.1f ms  %6.1f GFLOP/s\n", gpu_ms, gpu_gf);
    std::printf("  resident speedup           : %5.1fx\n", cpu_ms / gpu_ms);
    std::printf("  speedup incl. one 195 MiB  : %5.1fx  (H2D %.1f ms)\n",
                cpu_ms / (gpu_ms + h2d_ms), h2d_ms);
    std::printf("  per-query traffic          : %.0f MiB -> %.0f GB total (%.0f GB/s effective)\n",
                vbytes / 1048576.0, (double)vbytes * NQ / 1e9, (double)vbytes * NQ / (gpu_ms * 1e6));
    return 0;
}
