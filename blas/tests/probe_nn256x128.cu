// Focused probe: launch sgemm_nn_256x128 standalone on a tiny known input and
// dump the top-left 4x4 of kernel output vs a CPU reference. The error SHAPE
// localizes the cause:
//   - transposed corner      -> layout mismatch
//   - uniform ~2x / ~0.5x     -> alpha/beta double-apply
//   - half the block zero      -> grid / tile coverage
//   - garbage / NaN            -> uninitialized accumulator or bad smem
// Build (server, sm_89):
//   nvcc -arch=sm_89 -O2 -std=c++17 -static-global-template-stub=false \
//     probe_nn256x128.cu ../fp32/sgemm_nn_256x128_sm89_kernel.cu -o probe_nn256x128
//   CUDA_VISIBLE_DEVICES=7 ./probe_nn256x128

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
  printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

template <bool IsSplitK>
__global__ void sgemm_nn_256x128_sm89_2_opt_k(
    int M, int N, int K, float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta, float* C, int ldc, long long strideC, int batchCount, int splitK);

static constexpr int smem_bytes(int BM,int BN,int BK,int ST){return ST*(BM*BK+BN*BK)*(int)sizeof(float);}

int main(){
    // Match the harness: M=256,N=256,K=256, inputs in [-1,1], beta=0.5.
    const int M=256, N=256, K=256;
    const float alpha=1.0f, beta=0.5f;

    std::vector<float> hA((size_t)M*K), hB((size_t)K*N), hC0((size_t)M*N), hRef((size_t)M*N);
    // Bounded pseudo-random in [-1,1], deterministic.
    auto rnd=[&](size_t i){ uint64_t x=i*2654435761u+12345u; x^=x>>13; x*=0x9E3779B1u; x^=x>>15;
                            return (float)((x&0xFFFFFF)/(double)0xFFFFFF)*2.0f-1.0f; };
    for(size_t i=0;i<hA.size();++i) hA[i]=rnd(i+1);
    for(size_t i=0;i<hB.size();++i) hB[i]=rnd(i+1000003);
    for(size_t i=0;i<hC0.size();++i) hC0[i]=rnd(i+5000011);

    // CPU reference: NN, row-major. C = alpha*A*B + beta*C0.
    for(int r=0;r<M;++r) for(int c=0;c<N;++c){
        double acc=0; for(int k=0;k<K;++k) acc += (double)hA[(size_t)r*K+k]*hB[(size_t)k*N+c];
        hRef[(size_t)r*N+c] = alpha*(float)acc + beta*hC0[(size_t)r*N+c];
    }

    float *dA,*dB,*dC; size_t aB=hA.size()*4,bB=hB.size()*4,cB=hC0.size()*4;
    CK(cudaMalloc(&dA,aB)); CK(cudaMalloc(&dB,bB)); CK(cudaMalloc(&dC,cB));
    CK(cudaMemcpy(dA,hA.data(),aB,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,hB.data(),bB,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dC,hC0.data(),cB,cudaMemcpyHostToDevice));   // C starts at C0

    int sm = smem_bytes(256,128,16,4);
    CK(cudaFuncSetAttribute(sgemm_nn_256x128_sm89_2_opt_k<false>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
    dim3 grid((N+127)/128, (M+255)/256, 1);
    sgemm_nn_256x128_sm89_2_opt_k<false><<<grid, 256, sm>>>(
        M,N,K,alpha, dA,K,0, dB,N,0, beta, dC,N,0, 1, 1);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    std::vector<float> hOut((size_t)M*N);
    CK(cudaMemcpy(hOut.data(),dC,cB,cudaMemcpyDeviceToHost));

    printf("Top-left 4x4  (kernel | reference):\n");
    for(int r=0;r<4;++r){
        printf("  ");
        for(int c=0;c<4;++c) printf("%8.3f ", hOut[(size_t)r*N+c]);
        printf("   |  ");
        for(int c=0;c<4;++c) printf("%8.3f ", hRef[(size_t)r*N+c]);
        printf("\n");
    }
    // Also bottom-right corner (catches partial-tile coverage).
    printf("Bottom-right 4x4 (rows %d-%d, cols %d-%d):\n", M-4,M-1,N-4,N-1);
    for(int r=M-4;r<M;++r){
        printf("  ");
        for(int c=N-4;c<N;++c) printf("%8.3f ", hOut[(size_t)r*N+c]);
        printf("   |  ");
        for(int c=N-4;c<N;++c) printf("%8.3f ", hRef[(size_t)r*N+c]);
        printf("\n");
    }
    double emax=0; for(size_t i=0;i<hOut.size();++i) emax=fmax(emax,fabs(hOut[i]-hRef[i]));
    printf("err_max = %.4e\n", emax);
    return 0;
}
