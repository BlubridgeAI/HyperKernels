// test_sgemm_sm89_core_template.cu
// Benchmark and accuracy test: SM89 FP32 (TF32) core template vs cuBLAS.
// Math: C = alpha * A * B + beta * C + bias

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

template <int BM_, int BN_, int BK_, int STAGES_, int THREADS_>
struct SgemmTileConfigSM89 {
  static constexpr int BM = BM_;
  static constexpr int BN = BN_;
  static constexpr int BK = BK_;
  static constexpr int STAGES = STAGES_;
  static constexpr int THREADS = THREADS_;
};

enum class SgemmLayout { NT, TN, NN };

template <typename Config, bool IsAligned, bool IsSplitK, SgemmLayout Layout>
__global__ void sgemm_sm89_kernel(
    int M, int N, int K, float alpha,
    const float* __restrict__ A, int lda, long long strideA,
    const float* __restrict__ B, int ldb, long long strideB,
    float beta,
    float* __restrict__ C, int ldc, long long strideC,
    const float* __restrict__ bias, long long bias_stride,
    int batchCount, int splitK);

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

__global__ void add_bias_to_c(float* C, const float* bias, int M, int N) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < M && c < N) C[r * N + c] += bias[c];
}

struct Result { double custom_tflops; double cublas_tflops; double rel_err; bool pass; };

static Result run_test(SgemmLayout lay, int M, int N, int K, float alpha, float beta, cublasHandle_t cublas, cudaStream_t stream) {
    const int batchCount = 1;
    long long elA = M * K, elB = K * N, elC = M * N;
    int lda = (lay == SgemmLayout::TN) ? M : K;
    int ldb = (lay == SgemmLayout::NT) ? K : N;
    
    std::vector<float> hA(elA), hB(elB), hC0(elC), hBias(N);
    fill_rand(hA, 42); fill_rand(hB, 99); fill_rand(hC0, 24); fill_rand(hBias, 13);

    float *dA, *dB, *dBias, *dC0, *dC_ref, *dC_custom;
    CHECK_CUDA(cudaMalloc(&dA, elA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, elB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dBias, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC0, elC * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_ref, elC * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_custom, elC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), elA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), elB * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBias, hBias.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC0, hC0.data(), elC * sizeof(float), cudaMemcpyHostToDevice));

    cublasMath_t mode; cublasGetMathMode(cublas, &mode);
    cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH);

    cublasOperation_t trans_for_B = (lay == SgemmLayout::NN || lay == SgemmLayout::TN) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t trans_for_A = (lay == SgemmLayout::NN || lay == SgemmLayout::NT) ? CUBLAS_OP_N : CUBLAS_OP_T;

    auto run_cublas = [&]() {
        cudaMemcpyAsync(dC_ref, dC0, elC * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cublasGemmEx(cublas, trans_for_B, trans_for_A, N, M, K, &alpha, dB, CUDA_R_32F, ldb, dA, CUDA_R_32F, lda, &beta, dC_ref, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        dim3 blk(16, 16), grd((N+15)/16, (M+15)/16);
        add_bias_to_c<<<grd, blk, 0, stream>>>(dC_ref, dBias, M, N);
    };
    run_cublas(); CHECK_CUDA(cudaStreamSynchronize(stream));

    dim3 block(256);
    dim3 grid((N + 127) / 128, (M + 255) / 256, batchCount);
    
    using Cfg = SgemmTileConfigSM89<256, 128, 16, 4, 256>;
    int smem_bytes = 4 * (256 + 128) * 16 * sizeof(float);
    
    auto run_custom = [&]() {
        cudaMemcpyAsync(dC_custom, dC0, elC * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        if (lay == SgemmLayout::NN) {
            cudaFuncSetAttribute(sgemm_sm89_kernel<Cfg, true, false, SgemmLayout::NN>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            sgemm_sm89_kernel<Cfg, true, false, SgemmLayout::NN><<<grid, block, smem_bytes, stream>>>(M, N, K, alpha, dA, lda, elA, dB, ldb, elB, beta, dC_custom, N, elC, dBias, N, batchCount, 1);
        } else if (lay == SgemmLayout::NT) {
            cudaFuncSetAttribute(sgemm_sm89_kernel<Cfg, true, false, SgemmLayout::NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            sgemm_sm89_kernel<Cfg, true, false, SgemmLayout::NT><<<grid, block, smem_bytes, stream>>>(M, N, K, alpha, dA, lda, elA, dB, ldb, elB, beta, dC_custom, N, elC, dBias, N, batchCount, 1);
        } else {
            cudaFuncSetAttribute(sgemm_sm89_kernel<Cfg, true, false, SgemmLayout::TN>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            sgemm_sm89_kernel<Cfg, true, false, SgemmLayout::TN><<<grid, block, smem_bytes, stream>>>(M, N, K, alpha, dA, lda, elA, dB, ldb, elB, beta, dC_custom, N, elC, dBias, N, batchCount, 1);
        }
    };

    for (int w = 0; w < 5; ++w) run_custom();
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    CHECK_CUDA(cudaEventRecord(t0, stream));
    for (int i = 0; i < 20; ++i) run_custom();
    CHECK_CUDA(cudaEventRecord(t1, stream));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float custom_ms = 0.f; cudaEventElapsedTime(&custom_ms, t0, t1);

    CHECK_CUDA(cudaEventRecord(t0, stream));
    for (int i = 0; i < 20; ++i) run_cublas();
    CHECK_CUDA(cudaEventRecord(t1, stream));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float cub_ms = 0.f; cudaEventElapsedTime(&cub_ms, t0, t1);

    std::vector<float> hC_ref(elC), hC_custom(elC);
    CHECK_CUDA(cudaMemcpy(hC_ref.data(), dC_ref, elC*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_custom.data(), dC_custom, elC*sizeof(float), cudaMemcpyDeviceToHost));
    
    if (lay == SgemmLayout::NN) {
        printf("DEBUG NN: ref[0]=%.3f custom[0]=%.3f | ref[1]=%.3f custom[1]=%.3f\n",
            hC_ref[0], hC_custom[0], hC_ref[1], hC_custom[1]);
    }
    
    double rel_err = max_rel_err(hC_ref, hC_custom);
    double tfl = 2.0 * M * N * K * batchCount * 20.0 * 1e-12;
    Result res = {tfl / (custom_ms * 1e-3), tfl / (cub_ms * 1e-3), rel_err, rel_err < 1e-2};

    cudaFree(dA); cudaFree(dB); cudaFree(dBias); cudaFree(dC0); cudaFree(dC_ref); cudaFree(dC_custom);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cublasSetMathMode(cublas, mode);
    return res;
}

int main() {
    cudaStream_t stream; cudaStreamCreate(&stream);
    cublasHandle_t cublas; cublasCreate(&cublas); cublasSetStream(cublas, stream);

    printf("\n=== Testing FP32 Core Template (SM89) ===\n");
    printf("%-10s %5s %5s %5s  %-8s  %-8s  %-10s  %s\n", "Layout", "M", "N", "K", "Custom", "cuBLAS", "MaxRelErr", "Status");
    printf("%s\n", std::string(75, '-').c_str());

    std::vector<SgemmLayout> layouts = {SgemmLayout::NN, SgemmLayout::NT, SgemmLayout::TN};
    const char* lay_names[] = {"NN", "NT", "TN"};
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096, 8192, 16384};
    
    for (int size : sizes) {
        for (int i = 0; i < 3; i++) {
            Result r = run_test(layouts[i], size, size, size, 1.0f, 0.5f, cublas, stream);
            printf("%-10s %5d %5d %5d  %7.2f T  %7.2f T  %9.2e  %s\n",
                   lay_names[i], size, size, size, r.custom_tflops, r.cublas_tflops, r.rel_err, r.pass ? "PASS" : "FAIL");
        }
        printf("%s\n", std::string(75, '-').c_str());
    }

    cublasDestroy(cublas); cudaStreamDestroy(stream);
    return 0;
}
