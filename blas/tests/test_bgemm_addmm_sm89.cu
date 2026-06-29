// test_bgemm_addmm_sm89.cu
// Benchmark and accuracy test: SM89 BF16 batched AddMM (NN + bias) vs cuBLAS.
// Math: C = alpha * (A * B) + beta * C + beta * bias

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

template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned>
__global__ void bgemm_addmm_kernel(
    int M, int N, int K, __nv_bfloat16 alpha,
    const __nv_bfloat16* __restrict__ A, int lda, long long strideA,
    const __nv_bfloat16* __restrict__ B, int ldb, long long strideB,
    __nv_bfloat16 beta, const __nv_bfloat16* __restrict__ bias, int64_t bias_numel,
    __nv_bfloat16* __restrict__ C, int ldc, long long strideC, int batchCount);

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
        if (std::abs(r) < 1e-2 && std::abs(o) < 1e-2) continue; // ignore zero-cancellation noise
        double e = std::abs(r - o) / std::max(1e-5, (double)std::abs(r));
        if (e > max_e) {
            max_e = e;
            bad_idx = i; worst_r = r; worst_o = o;
        }
    }
    if (max_e > 5e-2) {
        printf("\n    -> Worst Mismatch at %zu: ref=%f, custom=%f, rel=%f\n", bad_idx, worst_r, worst_o, max_e);
    }
    return max_e;
}

__global__ void init_beta_bias(float* dC_float, const __nv_bfloat16* bias, float beta, int M, int N) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < M && c < N) dC_float[r * N + c] = beta * __bfloat162float(bias[c]);
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

static Result run_test(int M, int N, int K, float alpha_f, float beta_f, cublasHandle_t cublas, cudaStream_t stream) {
    const int batchCount = 1;
    long long elA = M * K, elB = K * N, elC = M * N;

    std::vector<__nv_bfloat16> hA(elA), hB(elB), hC0(elC), hBias(N);
    fill_rand(hA, 42); fill_rand(hB, 99); fill_rand(hC0, 24); fill_rand(hBias, 13);

    __nv_bfloat16 *dA, *dB, *dBias, *dC0, *dC_ref_bf16, *dC_custom;
    float *dC_ref_fp32;
    CHECK_CUDA(cudaMalloc(&dA, elA * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dB, elB * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dBias, N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC0, elC * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC_ref_bf16, elC * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC_custom, elC * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC_ref_fp32, elC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), elA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), elB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBias, hBias.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC0, hC0.data(), elC * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    auto run_cublas = [&]() {
        dim3 blk(16, 16), grd((N+15)/16, (M+15)/16);
        init_beta_bias<<<grd, blk, 0, stream>>>(dC_ref_fp32, dBias, 1.0f, M, N);
        CHECK_CUBLAS(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_f,
                     dB, CUDA_R_16BF, N, dA, CUDA_R_16BF, K, &beta_f,
                     dC_ref_fp32, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        cast_f32_to_bf16<<<(elC + 255) / 256, 256, 0, stream>>>(dC_ref_bf16, dC_ref_fp32, elC);
    };
    run_cublas(); CHECK_CUDA(cudaStreamSynchronize(stream));

    dim3 block(256);
    auto run_custom = [&]() {
        cudaMemcpyAsync(dC_custom, dC0, elC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
        if (M >= 4096 && N >= 256) {
            int smem_bytes_l = 3 * (256*32 + 32*128) * 2;
            cudaFuncSetAttribute(bgemm_addmm_kernel<256, 128, 32, 3, 256, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes_l);
            dim3 grid_l((N + 127) / 128, (M + 255) / 256, batchCount);
            bgemm_addmm_kernel<256, 128, 32, 3, 256, true><<<grid_l, block, smem_bytes_l, stream>>>(
                M, N, K, __float2bfloat16(alpha_f), dA, K, K*M, dB, N, K*N,
                __float2bfloat16(beta_f), dBias, N, dC_custom, N, M*N, batchCount);
        } else {
            int smem_bytes_m = 4 * (128*32 + 32*128) * 2;
            cudaFuncSetAttribute(bgemm_addmm_kernel<128, 128, 32, 4, 256, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes_m);
            dim3 grid_m((N + 127) / 128, (M + 127) / 128, batchCount);
            bgemm_addmm_kernel<128, 128, 32, 4, 256, true><<<grid_m, block, smem_bytes_m, stream>>>(
                M, N, K, __float2bfloat16(alpha_f), dA, K, K*M, dB, N, K*N,
                __float2bfloat16(beta_f), dBias, N, dC_custom, N, M*N, batchCount);
        }
        CHECK_CUDA(cudaGetLastError());
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
    
    double rel_err = max_rel_err(hC_ref, hC_custom);

    double tfl = 2.0 * M * N * K * batchCount * 20.0 * 1e-12;
    Result res = {tfl / (custom_ms * 1e-3), tfl / (cub_ms * 1e-3), rel_err, rel_err < 5e-2};

    cudaFree(dA); cudaFree(dB); cudaFree(dBias); cudaFree(dC0); cudaFree(dC_ref_bf16); cudaFree(dC_ref_fp32); cudaFree(dC_custom);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return res;
}

int main() {
    cudaStream_t stream; cudaStreamCreate(&stream);
    cublasHandle_t cublas; cublasCreate(&cublas); cublasSetStream(cublas, stream);

    printf("\n=== Testing BF16 AddMM Kernel (SM89) ===\n");
    printf("%-10s %5s %5s %5s  %-8s  %-8s  %-10s  %s\n", "Shape", "M", "N", "K", "Custom", "cuBLAS", "MaxRelErr", "Status");
    printf("%s\n", std::string(75, '-').c_str());

    std::vector<std::vector<int>> shapes = {{256,256,256}, {512,512,512}, {1024,1024,1024}, {2048,2048,2048}, {4096,4096,4096}, {8192,8192,8192}};
    for (auto& s : shapes) {
        Result r = run_test(s[0], s[1], s[2], 1.0f, 0.5f, cublas, stream);
        printf("%-10s %5d %5d %5d  %7.2f T  %7.2f T  %9.2e  %s\n", "square", s[0], s[1], s[2], r.custom_tflops, r.cublas_tflops, r.rel_err, r.pass ? "PASS" : "FAIL");
    }

    cublasDestroy(cublas); cudaStreamDestroy(stream); return 0;
}
