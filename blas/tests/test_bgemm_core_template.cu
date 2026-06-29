// test_bgemm_core_template.cu
// Benchmark and accuracy test: SM89 BF16 Core Template vs cuBLAS.
// Math: C = alpha * A * B + beta * C

#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
struct BgemmTileConfig {
  static constexpr int BM = BM_;
  static constexpr int BN = BN_;
  static constexpr int BK = BK_;
  static constexpr int STAGES = STAGES_;
  static constexpr int THREADS = THREADS_;
};

enum class BgemmLayout { NT, TN, NN };

template <typename Config, bool IsAligned, int SplitK, BgemmLayout Layout>
__global__ void bgemm_backward_template_kernel(
    int M, int N, int K, __nv_bfloat16 alpha,
    const __nv_bfloat16* __restrict__ A, int lda, long long strideA,
    const __nv_bfloat16* __restrict__ B, int ldb, long long strideB,
    __nv_bfloat16 beta, __nv_bfloat16* __restrict__ C, int ldc, long long strideC, int batchCount);

static void fill_rand(std::vector<__nv_bfloat16>& v, unsigned seed) {
    srand(seed);
    for (auto& x : v) x = __float2bfloat16((float(rand()) / RAND_MAX) * 2.f - 1.f);
}

static double max_rel_err(const std::vector<__nv_bfloat16>& ref, const std::vector<__nv_bfloat16>& out) {
    double max_e = 0.0;
    size_t bad_idx = 0;
    double worst_r = 0, worst_o = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double r = __bfloat162float(ref[i]), o = __bfloat162float(out[i]);
        if (std::abs(r) < 2e-2 && std::abs(o) < 2e-2) continue; // ignore zero-cancellation noise (scaled for large K)
        double e = std::abs(r - o) / std::max(1e-5, (double)std::abs(r));
        if (e > max_e) {
            max_e = e;
            bad_idx = i; worst_r = r; worst_o = o;
        }
    }
    if (max_e > 1.5e-1) {
        printf("\n    -> Worst Mismatch at %zu: ref=%f, custom=%f, rel=%f\n", bad_idx, worst_r, worst_o, max_e);
    }
    return max_e;
}

__global__ void cast_f32_to_bf16(__nv_bfloat16* out, const float* in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = __float2bfloat16(in[idx]);
}
__global__ void cast_bf16_to_f32(float* out, const __nv_bfloat16* in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = __bfloat162float(in[idx]);
}

struct Result { double custom_tflops; double cublas_tflops; double rel_err; bool pass; };

static Result run_test(BgemmLayout lay, int M, int N, int K, float alpha_f, float beta_f, cublasHandle_t cublas, cudaStream_t stream) {
    const int batchCount = 1;
    long long elA = M * K, elB = K * N, elC = M * N;
    int lda = (lay == BgemmLayout::TN) ? M : K;
    int ldb = (lay == BgemmLayout::NT) ? K : N;
    
    std::vector<__nv_bfloat16> hA(elA), hB(elB), hC0(elC);
    fill_rand(hA, 42); fill_rand(hB, 99); fill_rand(hC0, 24);

    __nv_bfloat16 *dA, *dB, *dC0, *dC_ref_bf16, *dC_custom;
    float *dC_ref_fp32;
    CHECK_CUDA(cudaMalloc(&dA, elA * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dB, elB * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC0, elC * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC_ref_bf16, elC * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC_custom, elC * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC_ref_fp32, elC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), elA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), elB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC0, hC0.data(), elC * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    cublasOperation_t trans_for_B = (lay == BgemmLayout::NN || lay == BgemmLayout::TN) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t trans_for_A = (lay == BgemmLayout::NN || lay == BgemmLayout::NT) ? CUBLAS_OP_N : CUBLAS_OP_T;

    auto run_cublas = [&]() {
        cast_bf16_to_f32<<<(elC + 255) / 256, 256, 0, stream>>>(dC_ref_fp32, dC0, elC);
        cublasGemmEx(cublas, trans_for_B, trans_for_A, N, M, K, &alpha_f, dB, CUDA_R_16BF, ldb, dA, CUDA_R_16BF, lda, &beta_f, dC_ref_fp32, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cast_f32_to_bf16<<<(elC + 255) / 256, 256, 0, stream>>>(dC_ref_bf16, dC_ref_fp32, elC);
    };
    run_cublas(); CHECK_CUDA(cudaStreamSynchronize(stream));

    using Cfg = BgemmTileConfig<256, 128, 32, 3, 256>;
    dim3 block(256);
    dim3 grid((N + 127) / 128, (M + 255) / 256, batchCount);
    int smem_bytes = 3 * (256*32 + 128*32) * 2;
    
    auto run_custom = [&]() {
        cudaMemcpyAsync(dC_custom, dC0, elC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
        if (lay == BgemmLayout::NN) {
            cudaFuncSetAttribute(bgemm_backward_template_kernel<Cfg, true, 1, BgemmLayout::NN>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            bgemm_backward_template_kernel<Cfg, true, 1, BgemmLayout::NN><<<grid, block, smem_bytes, stream>>>(M, N, K, __float2bfloat16(alpha_f), dA, lda, elA, dB, ldb, elB, __float2bfloat16(beta_f), dC_custom, N, elC, batchCount);
        } else if (lay == BgemmLayout::NT) {
            cudaFuncSetAttribute(bgemm_backward_template_kernel<Cfg, true, 1, BgemmLayout::NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            bgemm_backward_template_kernel<Cfg, true, 1, BgemmLayout::NT><<<grid, block, smem_bytes, stream>>>(M, N, K, __float2bfloat16(alpha_f), dA, lda, elA, dB, ldb, elB, __float2bfloat16(beta_f), dC_custom, N, elC, batchCount);
        } else {
            cudaFuncSetAttribute(bgemm_backward_template_kernel<Cfg, true, 1, BgemmLayout::TN>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            bgemm_backward_template_kernel<Cfg, true, 1, BgemmLayout::TN><<<grid, block, smem_bytes, stream>>>(M, N, K, __float2bfloat16(alpha_f), dA, lda, elA, dB, ldb, elB, __float2bfloat16(beta_f), dC_custom, N, elC, batchCount);
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

    std::vector<__nv_bfloat16> hC_ref(elC), hC_custom(elC);
    CHECK_CUDA(cudaMemcpy(hC_ref.data(), dC_ref_bf16, elC*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_custom.data(), dC_custom, elC*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    
    if (lay == BgemmLayout::NN) {
        printf("DEBUG NN: ref[0]=%.3f custom[0]=%.3f | ref[1]=%.3f custom[1]=%.3f\n",
            __bfloat162float(hC_ref[0]), __bfloat162float(hC_custom[0]),
            __bfloat162float(hC_ref[1]), __bfloat162float(hC_custom[1]));
    }
    
    double rel_err = max_rel_err(hC_ref, hC_custom);

    double tfl = 2.0 * M * N * K * batchCount * 20.0 * 1e-12;
    Result res = {tfl / (custom_ms * 1e-3), tfl / (cub_ms * 1e-3), rel_err, rel_err < 1.5e-1};

    cudaFree(dA); cudaFree(dB); cudaFree(dC0); cudaFree(dC_ref_bf16); cudaFree(dC_ref_fp32); cudaFree(dC_custom);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return res;
}

int main() {
    cudaStream_t stream; cudaStreamCreate(&stream);
    cublasHandle_t cublas; cublasCreate(&cublas); cublasSetStream(cublas, stream);

    printf("\n=== Testing BF16 Core Template (SM89) ===\n");
    printf("%-10s %5s %5s %5s  %-8s  %-8s  %-10s  %s\n", "Layout", "M", "N", "K", "Custom", "cuBLAS", "MaxRelErr", "Status");
    printf("%s\n", std::string(75, '-').c_str());

    std::vector<BgemmLayout> layouts = {BgemmLayout::NN, BgemmLayout::NT, BgemmLayout::TN};
    const char* lay_names[] = {"NN", "NT", "TN"};
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096, 8192};
    
    for (int size : sizes) {
        for (int i = 0; i < 3; i++) {
            Result r = run_test(layouts[i], size, size, size, 1.0f, 0.5f, cublas, stream);
            printf("%-10s %5d %5d %5d  %7.2f T  %7.2f T  %9.2e  %s\n",
                   lay_names[i], size, size, size, r.custom_tflops, r.cublas_tflops, r.rel_err, r.pass ? "PASS" : "FAIL");
        }
        printf("%s\n", std::string(75, '-').c_str());
    }

    cublasDestroy(cublas); cudaStreamDestroy(stream); return 0;
}
