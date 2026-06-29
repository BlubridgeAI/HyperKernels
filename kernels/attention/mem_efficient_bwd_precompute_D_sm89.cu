// mem_efficient_bwd_precompute_D_sm89.cu
// Precomputes D = rowsum(dO * O) per (batch, head, row) for the attention backward pass.

#include <cstdint>
#include <cuda_runtime.h>

struct MemEfficientBwdParams {
    const float* __restrict__ Q;
    const float* __restrict__ K;
    const float* __restrict__ V;
    const float* __restrict__ O;   // saved attention output
    const float* __restrict__ dO;
    const float* __restrict__ LSE;
    float* __restrict__ D;   // precompute_D writes, main kernels read
    float* __restrict__ dQ;
    float* __restrict__ dK;
    float* __restrict__ dV;
    int B;
    int nh;
    int T;
    float scale;
    bool is_causal;
    int64_t q_strideB, q_strideM, q_strideH;
    int64_t k_strideB, k_strideM, k_strideH;
    int64_t v_strideB, v_strideM, v_strideH;
    int64_t o_strideB, o_strideM, o_strideH;
    int64_t do_strideB, do_strideM, do_strideH;
    int64_t dq_strideB, dq_strideM, dq_strideH;
    int64_t dk_strideB, dk_strideM, dk_strideH;
    int64_t dv_strideB, dv_strideM, dv_strideH;
    int64_t lse_strideB, lse_strideH;
    int64_t d_strideB,   d_strideH;
};

static constexpr int SM89_BWD_WARP_SZ     = 32;
static constexpr int SM89_BWD_BLOCK_M_D   = 8;

__inline__ __device__ float sm89_bwd_warp_sum(float val) {
    #pragma unroll
    for (int offset = SM89_BWD_WARP_SZ / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

template<int HeadDim>
__global__ void mem_efficient_bwd_precompute_D_sm89(MemEfficientBwdParams params)
{
    const float* __restrict__ dO = params.dO;
    const float* __restrict__ O  = params.O;
    float*       __restrict__ D  = params.D;
    const int64_t T              = params.T;
    const int nh                 = params.nh;

    constexpr int LocalN = (HeadDim + SM89_BWD_WARP_SZ - 1) / SM89_BWD_WARP_SZ;

    const int bh      = blockIdx.y;
    const int warp_id = threadIdx.x / SM89_BWD_WARP_SZ;
    const int lane_id = threadIdx.x % SM89_BWD_WARP_SZ;
    const int b       = bh / nh;
    const int h       = bh - b * nh;

    const int base_row = blockIdx.x * (SM89_BWD_BLOCK_M_D * 2) + warp_id * 2;
    const int row0     = base_row;
    const int row1     = base_row + 1;
    const bool v0      = (row0 < T);
    const bool v1      = (row1 < T);

    const float* dO_bh = dO + b * params.do_strideB + h * params.do_strideH;
    const float* O_bh  = O  + b * params.o_strideB  + h * params.o_strideH;
    const long long dO_off0 = (long long)row0 * params.do_strideM;
    const long long dO_off1 = (long long)row1 * params.do_strideM;
    const long long O_off0  = (long long)row0 * params.o_strideM;
    const long long O_off1  = (long long)row1 * params.o_strideM;

    float sum0 = 0.f, sum1 = 0.f;
    #pragma unroll
    for (int i = 0; i < LocalN; ++i) {
        const int k = lane_id + i * SM89_BWD_WARP_SZ;
        if (k < HeadDim) {
            if (v0) sum0 += __ldg(&dO_bh[dO_off0 + k]) * __ldg(&O_bh[O_off0 + k]);
            if (v1) sum1 += __ldg(&dO_bh[dO_off1 + k]) * __ldg(&O_bh[O_off1 + k]);
        }
    }
    sum0 = sm89_bwd_warp_sum(sum0);
    sum1 = sm89_bwd_warp_sum(sum1);
    if (lane_id == 0) {
        float* D_bh = D + b * params.d_strideB + h * params.d_strideH;
        if (v0) D_bh[row0] = sum0;
        if (v1) D_bh[row1] = sum1;
    }
}

template __global__ void mem_efficient_bwd_precompute_D_sm89< 16>(MemEfficientBwdParams);
template __global__ void mem_efficient_bwd_precompute_D_sm89< 32>(MemEfficientBwdParams);
template __global__ void mem_efficient_bwd_precompute_D_sm89< 48>(MemEfficientBwdParams);
template __global__ void mem_efficient_bwd_precompute_D_sm89< 64>(MemEfficientBwdParams);
template __global__ void mem_efficient_bwd_precompute_D_sm89< 80>(MemEfficientBwdParams);
template __global__ void mem_efficient_bwd_precompute_D_sm89< 96>(MemEfficientBwdParams);
template __global__ void mem_efficient_bwd_precompute_D_sm89<128>(MemEfficientBwdParams);
template __global__ void mem_efficient_bwd_precompute_D_sm89<160>(MemEfficientBwdParams);
template __global__ void mem_efficient_bwd_precompute_D_sm89<192>(MemEfficientBwdParams);
template __global__ void mem_efficient_bwd_precompute_D_sm89<256>(MemEfficientBwdParams);

