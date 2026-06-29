// Correctness harness for the standalone FP32 SGEMM kernels in blas/.
// Each kernel is launched against a device-side fp32 triple-loop reference;
// results are compared with abs/err mean/max metrics and a relative-error gate.

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// Config types matching the core-template kernel (must match the linked object).
// ---------------------------------------------------------------------------
#ifndef MYCUBLAS_LAYOUT_ENUM
#define MYCUBLAS_LAYOUT_ENUM
enum class SgemmLayout { NT, TN, NN };
#endif

template <int BM_, int BN_, int BK_, int STAGES_, int THREADS_>
struct SgemmTileConfigSM89 {
    static constexpr int BM      = BM_;
    static constexpr int BN      = BN_;
    static constexpr int BK      = BK_;
    static constexpr int STAGES  = STAGES_;
    static constexpr int THREADS = THREADS_;

    static constexpr int AS_SIZE    = BM * BK;
    static constexpr int BS_SIZE    = BK * BN;
    static constexpr int STAGE_SIZE = AS_SIZE + BS_SIZE;

    static constexpr int SMEM_BYTES = STAGES * (BM + BN) * BK * sizeof(float);
    static constexpr int MAX_OCC    = (BM * BN >= 256 * 128) ? 1 : 2;

    static constexpr int WARPS_TOTAL = THREADS / 32;
    static constexpr int WARPS_N     = (BN >= 64 && WARPS_TOTAL > 1)
                                         ? ((BN / 64 < WARPS_TOTAL / 2) ? BN / 64 : WARPS_TOTAL / 2)
                                         : 1;
    static constexpr int WARPS_M     = (WARPS_TOTAL > 1) ? (WARPS_TOTAL / WARPS_N) : 1;

    static constexpr int WARP_TILE_M = BM / WARPS_M;
    static constexpr int WARP_TILE_N = BN / WARPS_N;

    static constexpr int MMA_M = WARP_TILE_M / 16;
    static constexpr int MMA_N = WARP_TILE_N / 8;
};

// ---------------------------------------------------------------------------
// Forward declarations of the kernels under test (defined in ../sgemm_*.cu).
// ---------------------------------------------------------------------------
template <typename Cfg, bool IsAligned, bool IsSplitK, SgemmLayout Layout>
__global__ void sgemm_sm89_kernel(
    int M, int N, int K, float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta,
    float* C, int ldc, long long strideC,
    const float* bias, long long bias_stride,
    int batchCount, int splitK);

template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned>
__global__ void sgemm_addmm_sm89_kernel(
    int M, int N, int K, float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta,
    const float* bias, int64_t bias_numel,
    float* C, int ldc, long long strideC,
    int batchCount);


template <bool IsSplitK>
__global__ void sgemm_nn_256x128_sm89_2_opt_k(
    int M, int N, int K, float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta, float* C, int ldc, long long strideC, int batchCount, int splitK);

__global__ void sgemm_nn_256x64_sm89_2_k(
    int M, int N, int K, float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta, float* C, int ldc, long long strideC, int batchCount);


__global__ void sgemm_nt_128x128_sm89_2_k(
    int M, int N, int K, float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta, float* C, int ldc, long long strideC, int batchCount);

template <bool IsSplitK>
__global__ void sgemm_nt_256x128_sm89_2_k(
    int M, int N, int K, float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta, float* C, int ldc, long long strideC, int batchCount, int splitK);

__global__ void sgemm_nt_64x128_sm89_2_k(
    int M, int N, int K, float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta, float* C, int ldc, long long strideC, int batchCount);

__global__ void sgemm_tn_128x128_sm89_2_k(
    int M, int N, int K, float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta, float* C, int ldc, long long strideC, int batchCount, int splitK);

__global__ void sgemm_tn_256x128_sm89_optimized(
    int M, int N, int K, float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta, float* C, int ldc, long long strideC, int batchCount, int splitK);

__global__ void sgemm_tn_256x64_sm89_2_k(
    int M, int N, int K, float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta, float* C, int ldc, long long strideC, int batchCount);


// ---------------------------------------------------------------------------
// Helpers: error check, random fill, reference GEMM, metrics.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while(0)

// Dynamic shared-memory byte count for a multi-stage tile, matching the library:
// SMEM = STAGES * (BM*BK + BN*BK) * sizeof(float).
static constexpr int smem_bytes(int BM, int BN, int BK, int STAGES) {
    return STAGES * (BM * BK + BN * BK) * (int)sizeof(float);
}

// These kernels request more than the 48 KB static cap, so the launcher must opt in
// with cudaFuncAttributeMaxDynamicSharedMemorySize before the launch (the library
// dispatcher does this; the standalone harness must do the same). Sets the attribute
// once per kernel and returns the byte count to pass as the launch's smem argument.
template <typename Fn>
static int opt_in_smem(Fn kernel, int bytes) {
    CUDA_CHECK(cudaFuncSetAttribute((const void*)kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    return bytes;
}

// Splitmix64 device fill to [min_val, max_val].
__global__ void fill_random_kernel(float* data, size_t count, uint64_t seed,
                                   float min_val, float max_val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint64_t x = seed + idx;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        float u = (float)(x >> 40) * (1.0f / 16777216.0f);
        data[idx] = u * (max_val - min_val) + min_val;
    }
}
static void fill_random(float* d, size_t count, uint64_t seed, float lo, float hi) {
    fill_random_kernel<<<(count + 255) / 256, 256>>>(d, count, seed, lo, hi);
    CUDA_CHECK(cudaGetLastError());
}

// Reference: C = alpha * op(A) * op(B) + beta * C0 (+ bias). All buffers row-major.
// transpose_a: A is [K,M] (TN). transpose_b: B is [N,K] (NT).
template <bool transpose_a, bool transpose_b, bool has_bias>
__global__ void reference_gemm_kernel(
    float* C, const float* C0, const float* A, const float* B, const float* bias,
    int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = transpose_a ? A[(long long)k * M + row] : A[(long long)row * K + k];
            float b = transpose_b ? B[(long long)col * K + k] : B[(long long)k * N + col];
            acc += a * b;
        }
        float out = alpha * acc + beta * C0[(long long)row * N + col];
        if (has_bias) out += bias[col];
        C[(long long)row * N + col] = out;
    }
}

struct Metrics { double abs_mean, abs_max, err_mean, err_max, rel; };

static Metrics check(const float* d_out, const float* d_ref, size_t count) {
    std::vector<float> ho(count), hr(count);
    CUDA_CHECK(cudaMemcpy(ho.data(), d_out, count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hr.data(), d_ref, count * sizeof(float), cudaMemcpyDeviceToHost));
    double abs_sum = 0, abs_max = 0, err_sum = 0, err_max = 0;
    for (size_t i = 0; i < count; ++i) {
        double v = ho[i], r = hr[i], e = std::abs(v - r);
        abs_sum += std::abs(r); abs_max = std::max(abs_max, std::abs(r));
        err_sum += e; err_max = std::max(err_max, e);
    }
    Metrics m;
    m.abs_mean = abs_sum / count; m.abs_max = abs_max;
    m.err_mean = err_sum / count; m.err_max = err_max;
    m.rel = err_max / std::max(abs_max, 1e-6);
    return m;
}

// ---------------------------------------------------------------------------
// Per-kernel test driver.
// ---------------------------------------------------------------------------
static const float kRelTol = 2e-2f;   // TF32 / MMA tolerance, ThunderKittens style
static int g_fail = 0;

enum Layout { NN, NT, TN };

// Launches the reference for the given layout into d_ref, using d_C0 as the pre-existing C.
template <bool has_bias>
static void run_reference(Layout lay, float* d_ref, const float* d_C0,
                          const float* dA, const float* dB, const float* d_bias,
                          int M, int N, int K, float alpha, float beta) {
    dim3 blk(16, 16), grd((N + 15) / 16, (M + 15) / 16);
    if (lay == NN)
        reference_gemm_kernel<false, false, has_bias><<<grd, blk>>>(d_ref, d_C0, dA, dB, d_bias, M, N, K, alpha, beta);
    else if (lay == NT)
        reference_gemm_kernel<false, true, has_bias><<<grd, blk>>>(d_ref, d_C0, dA, dB, d_bias, M, N, K, alpha, beta);
    else
        reference_gemm_kernel<true, false, has_bias><<<grd, blk>>>(d_ref, d_C0, dA, dB, d_bias, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}

// A is [M,K] (NN/NT) or [K,M] (TN); B is [K,N] (NN/TN) or [N,K] (NT). C is [M,N].
static void report(const std::string& name, int M, int N, int K, const Metrics& m, const char* err) {
    bool pass = (err == nullptr) && (m.rel <= kRelTol);
    if (!pass) g_fail++;
    printf("%-44s M=%4d N=%4d K=%4d  rel=%.3e abs_max=%.3e err_max=%.3e  %s\n",
           name.c_str(), M, N, K, m.rel, m.abs_max, m.err_max, pass ? "PASS" : "FAIL");
    if (err) printf("    ^^^ CUDA error: %s\n", err);
}

// Sizes per layout: A dims, B dims depend on layout.
// The vectorized kernels issue 16-byte (4-float) cp.async loads, so M/N/K and the
// leading dimensions must be multiples of 4 (the library dispatcher enforces the same;
// it falls back to a scalar path for ragged sizes, which these standalone kernels omit).
struct Size { int M, N, K; };
static const Size kSizes[] = { {256, 256, 256}, {512, 384, 128}, {256, 128, 96} };

// Allocate inputs sized for a layout, fill random, return device pointers.
struct Buffers {
    float *dA, *dB, *dC0, *dC, *dRef, *dBias;
    int M, N, K;
};

static Buffers make_buffers(Layout lay, int M, int N, int K, bool with_bias, uint64_t seed) {
    Buffers b; b.M = M; b.N = N; b.K = K;
    size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
    CUDA_CHECK(cudaMalloc(&b.dA, aN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.dB, bN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.dC0, cN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.dC, cN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.dRef, cN * sizeof(float)));
    b.dBias = nullptr;
    if (with_bias) CUDA_CHECK(cudaMalloc(&b.dBias, (size_t)N * sizeof(float)));
    fill_random(b.dA, aN, seed + 1, -1.f, 1.f);
    fill_random(b.dB, bN, seed + 2, -1.f, 1.f);
    fill_random(b.dC0, cN, seed + 3, -1.f, 1.f);
    if (with_bias) fill_random(b.dBias, N, seed + 4, -1.f, 1.f);
    CUDA_CHECK(cudaMemcpy(b.dC, b.dC0, cN * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    return b;
}
static void free_buffers(Buffers& b) {
    cudaFree(b.dA); cudaFree(b.dB); cudaFree(b.dC0); cudaFree(b.dC); cudaFree(b.dRef);
    if (b.dBias) cudaFree(b.dBias);
}

// Grid helper.
static dim3 grid(int N, int N_tile, int M, int M_tile, int batch) {
    return dim3((N + N_tile - 1) / N_tile, (M + M_tile - 1) / M_tile, batch);
}

// Run one launch, sync, capture error, run reference, compare, report.
// The lambda Launch issues the kernel into b.dC.
template <typename Launch>
static void run_case(const std::string& name, Layout lay, Buffers& b,
                     float alpha, float beta, bool with_bias, Launch&& launch) {
    // Each kernel computes C = alpha*A*B + beta*C in place, reading the existing C.
    // Buffers are reused across kernels, so restore C to the pristine C0 before every
    // launch; otherwise kernel N reads kernel N-1's output as its beta*C term while
    // the reference still uses C0, producing a graded error across the batch.
    CUDA_CHECK(cudaMemcpy(b.dC, b.dC0, (size_t)b.M * b.N * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    launch();
    cudaError_t le = cudaGetLastError();
    cudaError_t se = cudaDeviceSynchronize();
    const char* err = (le != cudaSuccess) ? cudaGetErrorString(le)
                    : (se != cudaSuccess) ? cudaGetErrorString(se) : nullptr;
    if (err) { Metrics z{0,0,0,0,1e9}; report(name, b.M, b.N, b.K, z, err); return; }
    if (with_bias) run_reference<true>(lay, b.dRef, b.dC0, b.dA, b.dB, b.dBias, b.M, b.N, b.K, alpha, beta);
    else           run_reference<false>(lay, b.dRef, b.dC0, b.dA, b.dB, nullptr, b.M, b.N, b.K, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());
    Metrics m = check(b.dC, b.dRef, (size_t)b.M * b.N);
    report(name, b.M, b.N, b.K, m, nullptr);
}

int main() {
    int dev = 0; cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    const float alpha = 1.0f, beta = 0.5f;
    const long long s = 0;  // batchCount = 1 -> strides unused (single matrix)

    for (const Size& sz : kSizes) {
        int M = sz.M, N = sz.N, K = sz.K;

        // ---- NN kernels (A:[M,K], B:[K,N], lda=K, ldb=N, ldc=N) ----
        {
            Buffers b = make_buffers(NN, M, N, K, false, 100 + M);
            int lda = K, ldb = N, ldc = N;


            run_case("sgemm_nn_256x128", NN, b, alpha, beta, false, [&]{
                int sm = opt_in_smem(sgemm_nn_256x128_sm89_2_opt_k<false>, smem_bytes(256,128,16,4));
                sgemm_nn_256x128_sm89_2_opt_k<false><<<grid(N,128,M,256,1), 256, sm>>>(
                    M, N, K, alpha, b.dA, lda, s, b.dB, ldb, s, beta, b.dC, ldc, s, 1, 1); });

            run_case("sgemm_nn_256x64", NN, b, alpha, beta, false, [&]{
                int sm = opt_in_smem(sgemm_nn_256x64_sm89_2_k, smem_bytes(256,64,16,4));
                sgemm_nn_256x64_sm89_2_k<<<grid(N,64,M,256,1), 256, sm>>>(
                    M, N, K, alpha, b.dA, lda, s, b.dB, ldb, s, beta, b.dC, ldc, s, 1); });


            // Core template, NN, config <128,128,16,6,128>.
            {
                using Cfg = SgemmTileConfigSM89<128,128,16,6,128>;
                run_case("sgemm_core_template_NN", NN, b, alpha, beta, false, [&]{
                    int sm = opt_in_smem(sgemm_sm89_kernel<Cfg, true, false, SgemmLayout::NN>, Cfg::SMEM_BYTES);
                    sgemm_sm89_kernel<Cfg, true, false, SgemmLayout::NN><<<grid(N,128,M,128,1), 128, sm>>>(
                        M, N, K, alpha, b.dA, lda, s, b.dB, ldb, s, beta, b.dC, ldc, s, nullptr, 0, 1, 1); });
            }
            free_buffers(b);
        }

        // ---- addmm (NN + bias), config <128,128,16,6,128,true> ----
        {
            Buffers b = make_buffers(NN, M, N, K, true, 200 + M);
            int lda = K, ldb = N, ldc = N;
            run_case("sgemm_addmm (NN+bias)", NN, b, alpha, beta, true, [&]{
                int sm = opt_in_smem(sgemm_addmm_sm89_kernel<128,128,16,6,128,true>, smem_bytes(128,128,16,6));
                sgemm_addmm_sm89_kernel<128,128,16,6,128,true><<<grid(N,128,M,128,1), 128, sm>>>(
                    M, N, K, alpha, b.dA, lda, s, b.dB, ldb, s, beta, b.dBias, N, b.dC, ldc, s, 1); });
            free_buffers(b);
        }

        // ---- NT kernels (A:[M,K], B:[N,K], lda=K, ldb=K, ldc=N) ----
        {
            // B is [N,K] row-major so its leading dim is K.
            Buffers b; b.M=M; b.N=N; b.K=K;
            size_t aN=(size_t)M*K, bN=(size_t)N*K, cN=(size_t)M*N;
            CUDA_CHECK(cudaMalloc(&b.dA, aN*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&b.dB, bN*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&b.dC0, cN*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&b.dC, cN*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&b.dRef, cN*sizeof(float)));
            b.dBias=nullptr;
            fill_random(b.dA, aN, 301+M, -1.f, 1.f);
            fill_random(b.dB, bN, 302+M, -1.f, 1.f);
            fill_random(b.dC0, cN, 303+M, -1.f, 1.f);
            CUDA_CHECK(cudaMemcpy(b.dC, b.dC0, cN*sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            int lda=K, ldb=K, ldc=N;

            run_case("sgemm_nt_128x128", NT, b, alpha, beta, false, [&]{
                int sm = opt_in_smem(sgemm_nt_128x128_sm89_2_k, smem_bytes(128,128,16,4));
                sgemm_nt_128x128_sm89_2_k<<<grid(N,128,M,128,1), 256, sm>>>(
                    M, N, K, alpha, b.dA, lda, s, b.dB, ldb, s, beta, b.dC, ldc, s, 1); });

            run_case("sgemm_nt_256x128", NT, b, alpha, beta, false, [&]{
                int sm = opt_in_smem(sgemm_nt_256x128_sm89_2_k<false>, smem_bytes(256,128,16,4));
                sgemm_nt_256x128_sm89_2_k<false><<<grid(N,128,M,256,1), 256, sm>>>(
                    M, N, K, alpha, b.dA, lda, s, b.dB, ldb, s, beta, b.dC, ldc, s, 1, 1); });

            run_case("sgemm_nt_64x128", NT, b, alpha, beta, false, [&]{
                int sm = opt_in_smem(sgemm_nt_64x128_sm89_2_k, smem_bytes(64,128,16,3));
                sgemm_nt_64x128_sm89_2_k<<<grid(N,128,M,64,1), 128, sm>>>(
                    M, N, K, alpha, b.dA, lda, s, b.dB, ldb, s, beta, b.dC, ldc, s, 1); });

            {
                using Cfg = SgemmTileConfigSM89<128,128,16,6,128>;
                run_case("sgemm_core_template_NT", NT, b, alpha, beta, false, [&]{
                    int sm = opt_in_smem(sgemm_sm89_kernel<Cfg, true, false, SgemmLayout::NT>, Cfg::SMEM_BYTES);
                    sgemm_sm89_kernel<Cfg, true, false, SgemmLayout::NT><<<grid(N,128,M,128,1), 128, sm>>>(
                        M, N, K, alpha, b.dA, lda, s, b.dB, ldb, s, beta, b.dC, ldc, s, nullptr, 0, 1, 1); });
            }
            cudaFree(b.dA); cudaFree(b.dB); cudaFree(b.dC0); cudaFree(b.dC); cudaFree(b.dRef);
        }

        // ---- TN kernels (A:[K,M], B:[K,N], lda=M, ldb=N, ldc=N) ----
        {
            Buffers b; b.M=M; b.N=N; b.K=K;
            size_t aN=(size_t)K*M, bN=(size_t)K*N, cN=(size_t)M*N;
            CUDA_CHECK(cudaMalloc(&b.dA, aN*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&b.dB, bN*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&b.dC0, cN*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&b.dC, cN*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&b.dRef, cN*sizeof(float)));
            b.dBias=nullptr;
            fill_random(b.dA, aN, 401+M, -1.f, 1.f);
            fill_random(b.dB, bN, 402+M, -1.f, 1.f);
            fill_random(b.dC0, cN, 403+M, -1.f, 1.f);
            CUDA_CHECK(cudaMemcpy(b.dC, b.dC0, cN*sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            int lda=M, ldb=N, ldc=N;

            run_case("sgemm_tn_128x128", TN, b, alpha, beta, false, [&]{
                int sm = opt_in_smem(sgemm_tn_128x128_sm89_2_k, smem_bytes(128,128,16,4));
                sgemm_tn_128x128_sm89_2_k<<<grid(N,128,M,128,1), 256, sm>>>(
                    M, N, K, alpha, b.dA, lda, s, b.dB, ldb, s, beta, b.dC, ldc, s, 1, 1); });

            run_case("sgemm_tn_256x128", TN, b, alpha, beta, false, [&]{
                int sm = opt_in_smem(sgemm_tn_256x128_sm89_optimized, smem_bytes(256,128,16,4));
                sgemm_tn_256x128_sm89_optimized<<<grid(N,128,M,256,1), 256, sm>>>(
                    M, N, K, alpha, b.dA, lda, s, b.dB, ldb, s, beta, b.dC, ldc, s, 1, 1); });

            run_case("sgemm_tn_256x64", TN, b, alpha, beta, false, [&]{
                int sm = opt_in_smem(sgemm_tn_256x64_sm89_2_k, smem_bytes(256,64,16,4));
                sgemm_tn_256x64_sm89_2_k<<<grid(N,64,M,256,1), 256, sm>>>(
                    M, N, K, alpha, b.dA, lda, s, b.dB, ldb, s, beta, b.dC, ldc, s, 1); });


            {
                using Cfg = SgemmTileConfigSM89<128,128,16,6,128>;
                run_case("sgemm_core_template_TN", TN, b, alpha, beta, false, [&]{
                    int sm = opt_in_smem(sgemm_sm89_kernel<Cfg, true, false, SgemmLayout::TN>, Cfg::SMEM_BYTES);
                    sgemm_sm89_kernel<Cfg, true, false, SgemmLayout::TN><<<grid(N,128,M,128,1), 128, sm>>>(
                        M, N, K, alpha, b.dA, lda, s, b.dB, ldb, s, beta, b.dC, ldc, s, nullptr, 0, 1, 1); });
            }
            cudaFree(b.dA); cudaFree(b.dB); cudaFree(b.dC0); cudaFree(b.dC); cudaFree(b.dRef);
        }
        printf("\n");
    }

    printf("%s  (%d failures, tol rel<=%.0e)\n", g_fail == 0 ? "ALL PASSED" : "SOME FAILED",
           g_fail, kRelTol);
    return g_fail == 0 ? 0 : 1;
}
