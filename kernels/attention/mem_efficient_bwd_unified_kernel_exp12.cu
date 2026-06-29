// mem_efficient_bwd_unified_kernel_exp12.cu
// Fused attention backward: computes dQ (atomicAdd), dK, dV per KV tile using TF32 tensor-core MMA.

#include <cuda_runtime.h>
#include <cstdint>

namespace OwnTensor {

static constexpr int SM89_BWD_WARP_SZ = 32;

struct MemEfficientBwdParams {
    const float* __restrict__ Q;
    const float* __restrict__ K;
    const float* __restrict__ V;
    const float* __restrict__ O;
    const float* __restrict__ dO;
    const float* __restrict__ LSE;
    float* __restrict__ D;
    float* __restrict__ dQ;
    float* __restrict__ dK;
    float* __restrict__ dV;
    int B;
    int nh;
    int T;
    float scale;
    bool is_causal;
    // Strides in elements; last-dim stride is 1. Q/K/V/O/dO/dQ/dK/dV are [B,nh,T,HeadDim], LSE/D are [B,nh,T].
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

// m16n8k8 TF32 MMA, f32 accumulate.
__device__ __forceinline__
void bwd_mma_tf32(float& d0, float& d1, float& d2, float& d3,
                  uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                  uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// KV-tile-centric backward. dK/dV in persistent register accumulators; dQ via atomicAdd.
template <int HeadDim, bool Causal>
__launch_bounds__(256, 2)
__global__ void mem_efficient_bwd_unified_kernel_exp12(MemEfficientBwdParams params)
{
    constexpr int BlockN    = 16;
    constexpr int BM_WMMA   = 32;
    constexpr int BM_TILES  = BM_WMMA / 16;
    constexpr int HD_CHUNKS = HeadDim / 16;
    constexpr int HD_PAD    = HeadDim + 4;
    constexpr int BKN_PAD   = BlockN  + 4;
    constexpr int BM_PAD    = BM_WMMA + 4;
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    extern __shared__ float smem_f[];
    float* Q_sm    = smem_f;
    float* dO_sm   = Q_sm    + BM_WMMA * HD_PAD;
    float* Ks      = dO_sm   + BM_WMMA * HD_PAD;
    float* Vs      = Ks      + BlockN   * HD_PAD;
    float* ds_qd   = Vs      + BlockN   * HD_PAD;
    float* DPV_sm  = ds_qd   + BM_WMMA  * BKN_PAD;
    float* ds_kd   = DPV_sm  + BM_WMMA  * BKN_PAD;
    float* p_kd    = ds_kd   + BlockN   * BM_PAD;
    float* tile_st = p_kd    + BlockN   * BM_PAD;
    float* LSE_sm  = tile_st + BM_WMMA  * HD_PAD;
    float* D_sm    = LSE_sm  + BM_WMMA;

    const int bh           = blockIdx.y;
    const int kv_tile      = blockIdx.x;
    const int kv_base      = kv_tile * BlockN;
    const int kv_tile_size = min(BlockN, params.T - kv_base);
    if (kv_base >= params.T) return;

    const int warp_id = threadIdx.x / SM89_BWD_WARP_SZ;
    const int lane    = threadIdx.x % SM89_BWD_WARP_SZ;
    const int chunk   = warp_id % HD_CHUNKS;
    const int b       = bh / params.nh;
    const int h       = bh - b * params.nh;

    const float* Q_bh   = params.Q   + b * params.q_strideB   + h * params.q_strideH;
    const float* K_bh   = params.K   + b * params.k_strideB   + h * params.k_strideH;
    const float* V_bh   = params.V   + b * params.v_strideB   + h * params.v_strideH;
    const float* dO_bh  = params.dO  + b * params.do_strideB  + h * params.do_strideH;
    const float* LSE_bh = params.LSE + b * params.lse_strideB + h * params.lse_strideH;
    const float* D_bh   = params.D   + b * params.d_strideB   + h * params.d_strideH;
    float*       dQ_bh  = params.dQ  + b * params.dq_strideB  + h * params.dq_strideH;
    float*       dK_bh  = params.dK  + b * params.dk_strideB  + h * params.dk_strideH;
    float*       dV_bh  = params.dV  + b * params.dv_strideB  + h * params.dv_strideH;

    const int64_t q_sM  = params.q_strideM;
    const int64_t k_sM  = params.k_strideM;
    const int64_t v_sM  = params.v_strideM;
    const int64_t do_sM = params.do_strideM;
    const int64_t dq_sM = params.dq_strideM;
    const int64_t dk_sM = params.dk_strideM;
    const int64_t dv_sM = params.dv_strideM;

    // MMA helpers: loadA = ldmatrix.x4; loadB_nt/loadB_nn = B operand loads; scatter = write 16x8 tile.
    auto loadA = [&](uint32_t r[4], int m_base, int k8,
                     const float* As, int stride) __attribute__((always_inline)) {
        const int row  = m_base + (lane & 15);
        const int col  = k8 + (lane >> 4) * 4;
        const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&As[row * stride + col]);
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
        r[0] += 0x1000u; r[1] += 0x1000u; r[2] += 0x1000u; r[3] += 0x1000u;
    };

    auto loadB_nt = [&](uint32_t r[2], int n_base, int k8,
                        const float* Bs, int stride) __attribute__((always_inline)) {
        const int n = n_base + (lane >> 2);
        const int k = k8 + (lane & 3);
        r[0] = *(const uint32_t*)(&Bs[n * stride + k]);
        r[1] = *(const uint32_t*)(&Bs[n * stride + k + 4]);
        r[0] += 0x1000u; r[1] += 0x1000u;
    };

    auto loadB_nn = [&](uint32_t r[2], int n_base, int k8,
                        const float* Bs, int stride) __attribute__((always_inline)) {
        const int n = n_base + (lane >> 2);
        const int k = k8 + (lane & 3);
        r[0] = *(const uint32_t*)(&Bs[k       * stride + n]);
        r[1] = *(const uint32_t*)(&Bs[(k + 4) * stride + n]);
        r[0] += 0x1000u; r[1] += 0x1000u;
    };

    auto scatter = [&](const float d[4], float* dst, int r_base, int c_base,
                       int stride) __attribute__((always_inline)) {
        const int r0 = r_base + (lane >> 2), r1 = r0 + 8;
        const int c  = c_base + (lane & 3) * 2;
        dst[r0 * stride + c]     = d[0];
        dst[r0 * stride + c + 1] = d[1];
        dst[r1 * stride + c]     = d[2];
        dst[r1 * stride + c + 1] = d[3];
    };

    // Load K and V tiles into smem (persistent across the Q loop).
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        const int g = kv_base + r;
        Ks[r * HD_PAD + k] = (g < params.T) ? K_bh[g * k_sM + k] : 0.f;
        Vs[r * HD_PAD + k] = (g < params.T) ? V_bh[g * v_sM + k] : 0.f;
    }
    __syncthreads();

    // Persistent dK/dV accumulators (used only by warps HD_CHUNKS..2*HD_CHUNKS-1).
    float dk_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float dv_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int q_loop_start = Causal ? kv_base : 0;

    for (int q_base = q_loop_start; q_base < params.T; q_base += BM_WMMA) {
        __syncthreads();

        const int q_tile_size = min(BM_WMMA, params.T - q_base);

        // Load Q, dO, LSE, D for this Q tile.
        for (int idx = threadIdx.x; idx < BM_WMMA * HeadDim; idx += blockDim.x) {
            const int r  = idx / HeadDim, k = idx % HeadDim;
            const int qi = q_base + r;
            const bool vq = (qi < params.T);
            Q_sm [r * HD_PAD + k] = vq ? Q_bh [qi * q_sM  + k] : 0.f;
            dO_sm[r * HD_PAD + k] = vq ? dO_bh[qi * do_sM + k] : 0.f;
        }
        if (threadIdx.x < BM_WMMA) {
            const int qi  = q_base + threadIdx.x;
            const bool vq = (qi < params.T);
            LSE_sm[threadIdx.x] = vq ? LSE_bh[qi] : 0.f;
            D_sm  [threadIdx.x] = vq ? D_bh  [qi] : 0.f;
        }
        __syncthreads();

        // Phase A: warps 0-1 compute S = Q @ K^T, warps 2-3 compute DPV = dO @ V^T.
        if (warp_id < 2 * BM_TILES) {
            const int  rg       = warp_id % BM_TILES;
            const bool is_qk    = (warp_id < BM_TILES);
            const float* src_sm = (is_qk ? Q_sm  : dO_sm) + rg * 16 * HD_PAD;
            const float* kv_sm  =  is_qk ? Ks    : Vs;
            float*       dst_sm = (is_qk ? ds_qd : DPV_sm) + rg * 16 * BKN_PAD;

            float acc_l[4] = {0.f, 0.f, 0.f, 0.f};
            float acc_r[4] = {0.f, 0.f, 0.f, 0.f};
            uint32_t frA[4], frB_l[2], frB_r[2];

            constexpr int KS_TOTAL = 2 * HD_CHUNKS;
            #pragma unroll
            for (int ks = 0; ks < KS_TOTAL; ks++) {
                const int k8 = ks * 8;
                loadA   (frA,   0, k8, src_sm, HD_PAD);
                loadB_nt(frB_l, 0,     k8, kv_sm, HD_PAD);
                loadB_nt(frB_r, 8,     k8, kv_sm, HD_PAD);
                bwd_mma_tf32(acc_l[0], acc_l[1], acc_l[2], acc_l[3],
                             frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
                bwd_mma_tf32(acc_r[0], acc_r[1], acc_r[2], acc_r[3],
                             frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
            }
            scatter(acc_l, dst_sm, 0, 0, BKN_PAD);
            scatter(acc_r, dst_sm, 0, 8, BKN_PAD);
        }
        __syncthreads();

        // Compute p and ds; fill ds_qd, ds_kd, p_kd.
        for (int elem = threadIdx.x; elem < BM_WMMA * BlockN; elem += blockDim.x) {
            const int qi_local = elem / BlockN;
            const int j_local  = elem % BlockN;
            const float raw_s  = ds_qd [qi_local * BKN_PAD + j_local];
            const float dpv    = DPV_sm[qi_local * BKN_PAD + j_local];
            const float L      = LSE_sm[qi_local];
            const float D_val  = D_sm  [qi_local];
            const bool qi_ok   = ((q_base + qi_local) < params.T);
            const bool j_ok    = (j_local < kv_tile_size);
            const bool cok     = !Causal || ((kv_base + j_local) <= (q_base + qi_local));
            float p = 0.f;
            if (qi_ok && j_ok && cok)
                p = exp2f(BWD_LOG2E * (raw_s * params.scale - L));
            const float ds = p * (dpv - D_val) * params.scale;
            ds_qd[qi_local * BKN_PAD + j_local] = ds;
            ds_kd[j_local  * BM_PAD  + qi_local] = ds;
            p_kd [j_local  * BM_PAD  + qi_local] = p;
        }
        __syncthreads();

        // Phase B1: warps 0-3 -> dQ for this tile, warps 4-7 -> dK (persistent accumulate).
        {
            uint32_t frA[4], frB_l[2], frB_r[2];

            if (warp_id < HD_CHUNKS) {
                const int n_base = chunk * 16;
                float dq_l[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
                float dq_r[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

                #pragma unroll
                for (int rg = 0; rg < BM_TILES; rg++) {
                    const float* a_base = ds_qd + rg * 16 * BKN_PAD;
                    #pragma unroll
                    for (int ks = 0; ks < 2; ks++) {
                        const int k8 = ks * 8;
                        loadA   (frA,   0, k8, a_base, BKN_PAD);
                        loadB_nn(frB_l, n_base,     k8, Ks, HD_PAD);
                        loadB_nn(frB_r, n_base + 8, k8, Ks, HD_PAD);
                        bwd_mma_tf32(dq_l[rg][0], dq_l[rg][1], dq_l[rg][2], dq_l[rg][3],
                                     frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
                        bwd_mma_tf32(dq_r[rg][0], dq_r[rg][1], dq_r[rg][2], dq_r[rg][3],
                                     frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
                    }
                    scatter(dq_l[rg], tile_st, rg * 16, n_base,     HD_PAD);
                    scatter(dq_r[rg], tile_st, rg * 16, n_base + 8, HD_PAD);
                }
            } else {
                const int n_base = chunk * 16;
                #pragma unroll
                for (int ks = 0; ks < BM_TILES * 2; ks++) {
                    const int k8 = ks * 8;
                    loadA   (frA,   0, k8, ds_kd, BM_PAD);
                    loadB_nn(frB_l, n_base,     k8, Q_sm, HD_PAD);
                    loadB_nn(frB_r, n_base + 8, k8, Q_sm, HD_PAD);
                    bwd_mma_tf32(dk_acc[0], dk_acc[1], dk_acc[2], dk_acc[3],
                                 frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
                    bwd_mma_tf32(dk_acc[4], dk_acc[5], dk_acc[6], dk_acc[7],
                                 frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
                }
            }
        }
        __syncthreads();

        // atomicAdd tile_st (dQ for this Q-tile) into global dQ.
        for (int idx = threadIdx.x; idx < q_tile_size * HeadDim; idx += blockDim.x) {
            const int r = idx / HeadDim, k = idx % HeadDim;
            atomicAdd(&dQ_bh[(q_base + r) * dq_sM + k], tile_st[r * HD_PAD + k]);
        }

        // Phase B2: warps 4-7 -> dV (persistent accumulate).
        if (warp_id >= HD_CHUNKS) {
            uint32_t frA[4], frB_l[2], frB_r[2];
            const int n_base = chunk * 16;
            #pragma unroll
            for (int ks = 0; ks < BM_TILES * 2; ks++) {
                const int k8 = ks * 8;
                loadA   (frA,   0, k8, p_kd, BM_PAD);
                loadB_nn(frB_l, n_base,     k8, dO_sm, HD_PAD);
                loadB_nn(frB_r, n_base + 8, k8, dO_sm, HD_PAD);
                bwd_mma_tf32(dv_acc[0], dv_acc[1], dv_acc[2], dv_acc[3],
                             frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
                bwd_mma_tf32(dv_acc[4], dv_acc[5], dv_acc[6], dv_acc[7],
                             frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
            }
        }
    }

    // Scatter dk_acc/dv_acc through tile_st to global dK/dV.
    __syncthreads();

    if (warp_id >= HD_CHUNKS) {
        const int n_base = chunk * 16;
        scatter(dk_acc,     tile_st, 0, n_base,     HD_PAD);
        scatter(dk_acc + 4, tile_st, 0, n_base + 8, HD_PAD);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        dK_bh[(kv_base + r) * dk_sM + k] = tile_st[r * HD_PAD + k];
    }
    __syncthreads();

    if (warp_id >= HD_CHUNKS) {
        const int n_base = chunk * 16;
        scatter(dv_acc,     tile_st, 0, n_base,     HD_PAD);
        scatter(dv_acc + 4, tile_st, 0, n_base + 8, HD_PAD);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        dV_bh[(kv_base + r) * dv_sM + k] = tile_st[r * HD_PAD + k];
    }
}

template __global__ void mem_efficient_bwd_unified_kernel_exp12<64, false>(MemEfficientBwdParams);
template __global__ void mem_efficient_bwd_unified_kernel_exp12<64, true >(MemEfficientBwdParams);

} // namespace OwnTensor

