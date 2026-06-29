// test_sgemm_addmm_sm89.cu
// Benchmark and accuracy test: SM89 FP32 (TF32) batched AddMM (NN) vs cuBLAS.
// Math: C = alpha * (A * B) + beta * bias

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#define CHECK_CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(EXIT_FAILURE); } } while(0)
#define CHECK_CUBLAS(call) do { cublasStatus_t e = (call); if (e != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)e, __FILE__, __LINE__); exit(EXIT_FAILURE); } } while(0)

// ─── kernel declaration ───────────────────────────────────────────────────────
template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned>
__global__ void sgemm_addmm_sm89_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda, long long strideA,
    const float* __restrict__ B, int ldb, long long strideB,
    float beta,
    const float* __restrict__ bias, int64_t bias_numel,
    float* __restrict__ C, int ldc, long long strideC,
    int batchCount);

static void fill_rand(std::vector<float>& v, unsigned seed) {
    srand(seed);
    for (auto& x : v) x = (float(rand()) / RAND_MAX) * 2.f - 1.f;
}

static double max_rel_err(const std::vector<float>& ref, const std::vector<float>& out) {
    double max_e = 0.0;
    size_t bad_idx = 0;
    double worst_r = 0, worst_o = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double r = ref[i], o = out[i];
        if (std::abs(r) < 1e-2 && std::abs(o) < 1e-2) continue; // ignore zero-cancellation noise
        double e = std::abs(r - o) / std::max(1e-5, (double)std::abs(r));
        if (e > max_e) {
            max_e = e;
            bad_idx = i; worst_r = r; worst_o = o;
        }
    }
    if (max_e > 1e-2) {
        printf("\n    -> Worst Mismatch at %zu: ref=%f, custom=%f, rel=%f\n", bad_idx, worst_r, worst_o, max_e);
    }
    return max_e;
}

// Helper kernel to initialize C_ref with bias for cuBLAS
__global__ void init_c_with_bias(float* C, const float* bias, int M, int N) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < M && c < N) C[r * N + c] = bias[c];
}

struct Result { double custom_tflops; double cublas_tflops; double rel_err; bool pass; };

static Result run_test(int M, int N, int K, float alpha, float beta, cublasHandle_t cublas, cudaStream_t stream) {
    const int batchCount = 1;
    const long long elA = M * K, elB = K * N, elC = M * N;

    std::vector<float> hA(elA), hB(elB), hBias(N);
    fill_rand(hA, 42); fill_rand(hB, 99); fill_rand(hBias, 13);

    float *dA, *dB, *dBias, *dC_ref, *dC_custom;
    CHECK_CUDA(cudaMalloc(&dA, elA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, elB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dBias, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_ref, elC * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_custom, elC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), elA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), elB * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBias, hBias.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_custom, 0, elC * sizeof(float)));

    dim3 blk(16, 16), grd((N+15)/16, (M+15)/16);
    init_c_with_bias<<<grd, blk, 0, stream>>>(dC_ref, dBias, M, N);
    
    cublasMath_t mode; cublasGetMathMode(cublas, &mode);
    cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH);

    CHECK_CUBLAS(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &alpha, dB, CUDA_R_32F, N, dA, CUDA_R_32F, K, &beta, dC_ref, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // One-time setup for all tile configs
    static bool smem_set = false;
    if (!smem_set) {
        cudaFuncSetAttribute(sgemm_addmm_sm89_kernel<256, 128, 32, 2, 256, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cudaFuncSetAttribute(sgemm_addmm_sm89_kernel<128, 128, 32, 3, 128, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cudaFuncSetAttribute(sgemm_addmm_sm89_kernel<128, 128, 16, 6, 128, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        smem_set = true;
    }

    auto launch = [&]() {
        // Always reset C_custom to zero so beta doesn't corrupt from previous runs
        CHECK_CUDA(cudaMemsetAsync(dC_custom, 0, elC * sizeof(float), stream));
        if (M >= 2048 && N >= 2048) {
            // Large tile: 256x128, 2 stages, 256 threads.
            // At 2048x2048 → 128 blocks = 1 block/SM (sweet spot for 256x128 on SM89).
            int smem = 2 * (256*32 + 128*32) * (int)sizeof(float);
            dim3 g((N+127)/128, (M+255)/256, batchCount);
            sgemm_addmm_sm89_kernel<256, 128, 32, 2, 256, true><<<g, 256, smem, stream>>>(
                M, N, K, alpha, dA, K, (long long)K*M, dB, N, (long long)K*N,
                beta, dBias, N, dC_custom, N, (long long)M*N, batchCount);
        } else if (K >= 256) {
            // Mid tile: 128x128x32, 3 stages, 128 threads → 2 blocks/SM on SM89 (48KB fits 2×24KB)
            int smem = 3 * (128*32 + 128*32) * (int)sizeof(float);
            dim3 g((N+127)/128, (M+127)/128, batchCount);
            sgemm_addmm_sm89_kernel<128, 128, 32, 3, 128, true><<<g, 128, smem, stream>>>(
                M, N, K, alpha, dA, K, (long long)K*M, dB, N, (long long)K*N,
                beta, dBias, N, dC_custom, N, (long long)M*N, batchCount);
        } else {
            // Short K: 128x128x16, 6 stages — deeper pipeline hides short-K SMEM latency
            int smem = 6 * (128*16 + 128*16) * (int)sizeof(float);
            dim3 g((N+127)/128, (M+127)/128, batchCount);
            sgemm_addmm_sm89_kernel<128, 128, 16, 6, 128, true><<<g, 128, smem, stream>>>(
                M, N, K, alpha, dA, K, (long long)K*M, dB, N, (long long)K*N,
                beta, dBias, N, dC_custom, N, (long long)M*N, batchCount);
        }
        CHECK_CUDA(cudaGetLastError());
    };

    for (int w = 0; w < 5; ++w) launch();
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    CHECK_CUDA(cudaEventRecord(t0, stream));
    for (int i = 0; i < 20; ++i) launch();
    CHECK_CUDA(cudaEventRecord(t1, stream));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float custom_ms = 0.f; cudaEventElapsedTime(&custom_ms, t0, t1);

    CHECK_CUDA(cudaEventRecord(t0, stream));
    for (int i = 0; i < 20; ++i) {
        init_c_with_bias<<<grd, blk, 0, stream>>>(dC_ref, dBias, M, N);
        cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_32F, N, dA, CUDA_R_32F, K, &beta, dC_ref, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    CHECK_CUDA(cudaEventRecord(t1, stream));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float cub_ms = 0.f; cudaEventElapsedTime(&cub_ms, t0, t1);

    std::vector<float> hC_ref_h(elC), hC_custom_h(elC);
    CHECK_CUDA(cudaMemcpy(hC_ref_h.data(), dC_ref, elC*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_custom_h.data(), dC_custom, elC*sizeof(float), cudaMemcpyDeviceToHost));
    double rel_err = max_rel_err(hC_ref_h, hC_custom_h);

    double tfl = 2.0 * M * N * K * batchCount * 20.0 * 1e-12;
    Result res = {tfl / (custom_ms * 1e-3), tfl / (cub_ms * 1e-3), rel_err, rel_err < 5e-2};

    cudaFree(dA); cudaFree(dB); cudaFree(dBias); cudaFree(dC_ref); cudaFree(dC_custom);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cublasSetMathMode(cublas, mode);
    return res;
}

int main() {
    cudaStream_t stream; cudaStreamCreate(&stream);
    cublasHandle_t cublas; cublasCreate(&cublas); cublasSetStream(cublas, stream);

    printf("\n=== Testing FP32 AddMM Kernel (SM89) ===\n");
    printf("%-10s %5s %5s %5s  %-8s  %-8s  %-10s  %s\n", "Shape", "M", "N", "K", "Custom", "cuBLAS", "MaxRelErr", "Status");
    printf("%s\n", std::string(75, '-').c_str());

    std::vector<std::vector<int>> shapes = {{256,256,256}, {512,512,512}, {1024,1024,1024}, {2048,2048,2048}, {4096,4096,4096}};
    for (auto& s : shapes) {
        Result r = run_test(s[0], s[1], s[2], 1.0f, 0.5f, cublas, stream);
        printf("%-10s %5d %5d %5d  %7.2f T  %7.2f T  %9.2e  %s\n",
               "square", s[0], s[1], s[2], r.custom_tflops, r.cublas_tflops, r.rel_err, r.pass ? "PASS" : "FAIL");
    }

    cublasDestroy(cublas); cudaStreamDestroy(stream);
    return 0;
}
