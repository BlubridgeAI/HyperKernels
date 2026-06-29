// fused_attn_forward_kernel_tc_sm89.cu
// FlashAttention-style fused attention forward (Q@K^T, online softmax, P@V) using TF32 tensor-core MMA.

#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>

struct MemEfficientFwdParams {
    const float* Q;
    const float* K;
    const float* V;
    float*       O;
    float*       LSE;

    int     B;
    int     nh;
    int64_t T;

    float   scale;
    bool    is_causal;
    float   dropout_p;
    const float* dropout_mask;

    int64_t q_strideB, q_strideM, q_strideH;
    int64_t k_strideB, k_strideM, k_strideH;
    int64_t v_strideB, v_strideM, v_strideH;
    int64_t o_strideB, o_strideM, o_strideH;
    int64_t lse_strideB, lse_strideH;
};

// cp.async helpers
__device__ __forceinline__ void sm89_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template<int N>
__device__ __forceinline__ void sm89_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}
__device__ __forceinline__ void sm89_cp_async_l16(
    void* smem_ptr, const void* global_ptr, bool pred)
{
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "  @!p st.shared.v4.u32 [%0], {0,0,0,0};\n"
        "}\n"
        : : "r"(smem_addr), "l"(global_ptr), "r"((int)pred) : "memory");
}

static constexpr int SM89_NUM_THREADS = 256;  // 8 warps
static constexpr int SM89_WMMA_M      = 16;
static constexpr int SM89_WMMA_N      = 16;   // 2 x MMA n=8
static constexpr int SM89_WMMA_K      = 8;
static constexpr int SM89_SMEM_PAD    = 4;
static constexpr int SM89_BK_PAD      = 8;

__device__ __forceinline__
void mma_tf32_m16n8k8(float& d0, float& d1, float& d2, float& d3,
                      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                      uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template<int HeadDim, int BQ_TILE, int BK_TILE, int MaxBlocksPerSM>
__global__ void __launch_bounds__(SM89_NUM_THREADS, MaxBlocksPerSM)
fused_attn_forward_kernel_tc_sm89(MemEfficientFwdParams params)
{
    const float* __restrict__ Q            = params.Q;
    const float* __restrict__ K            = params.K;
    const float* __restrict__ V            = params.V;
    float*       __restrict__ O            = params.O;
    float*       __restrict__ LSE          = params.LSE;
    const int64_t T                        = params.T;
    const int     nh                       = params.nh;
    const float   scale                    = params.scale;
    const bool    is_causal                = params.is_causal;
    const float   dropout_p                = params.dropout_p;
    const float* __restrict__ dropout_mask = params.dropout_mask;

    static_assert(HeadDim % SM89_WMMA_N == 0, "HeadDim must be divisible by 16");
    static_assert(BQ_TILE % SM89_WMMA_M == 0, "BQ_TILE must be divisible by 16");
    static_assert(BK_TILE % 32          == 0, "BK_TILE must be a multiple of 32");

    constexpr int HD_PAD    = HeadDim + SM89_SMEM_PAD;
    constexpr int NUM_WARPS = SM89_NUM_THREADS / 32;

    // Score GEMM: Q[BQxHD] @ K[BKxHD]^T -> s_scores[BQxBK]
    constexpr int SCORE_M_TILES = BQ_TILE / SM89_WMMA_M;
    constexpr int SCORE_N_TILES = BK_TILE / SM89_WMMA_N;
    constexpr int SCORE_K_TILES = HeadDim / SM89_WMMA_K;
    constexpr int SCORE_TOTAL   = SCORE_M_TILES * SCORE_N_TILES;
    constexpr int BK_STRIDE     = BK_TILE + SM89_BK_PAD;

    // P@V GEMM: P[BQxBK] @ V[BKxHD] -> s_out[BQxHD]
    constexpr int PV_M_TILES = BQ_TILE / SM89_WMMA_M;
    constexpr int PV_N_TILES = HeadDim / SM89_WMMA_N;
    constexpr int PV_TOTAL   = PV_M_TILES * PV_N_TILES;
    constexpr int PV_K_TILES = BK_TILE   / SM89_WMMA_K;
    constexpr int PV_PASSES  = (PV_TOTAL + NUM_WARPS - 1) / NUM_WARPS;

    constexpr int ROWS_PER_WARP   = BQ_TILE / NUM_WARPS;
    constexpr int COLS_PER_THREAD = BK_TILE / 32;

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int tid     = threadIdx.x;

    const int64_t qi_block = (int64_t)blockIdx.x * BQ_TILE;
    const int     bnh      = blockIdx.y;
    const int     b        = bnh / nh;
    const int     h        = bnh - b * nh;

    const float* Q_bnh   = Q   + b * params.q_strideB   + h * params.q_strideH;
    const float* K_bnh   = K   + b * params.k_strideB   + h * params.k_strideH;
    const float* V_bnh   = V   + b * params.v_strideB   + h * params.v_strideH;
    float*       O_bnh   = O   + b * params.o_strideB   + h * params.o_strideH;
    float*       LSE_bnh = LSE + b * params.lse_strideB + h * params.lse_strideH;

    const int64_t q_sM = params.q_strideM;
    const int64_t k_sM = params.k_strideM;
    const int64_t v_sM = params.v_strideM;
    const int64_t o_sM = params.o_strideM;

    extern __shared__ float smem[];
    float* s_q       = smem;
    float* s_kv_base = s_q      + BQ_TILE * HD_PAD;
    float* s_kv[2]   = { s_kv_base, s_kv_base + BK_TILE * HD_PAD };
    float* s_scores  = s_kv_base + 2 * BK_TILE * HD_PAD;
    float* s_m       = s_scores  + BQ_TILE * BK_STRIDE;
    float* s_l       = s_m       + BQ_TILE;
    float* s_alpha   = s_l       + BQ_TILE;

    for (int i = tid; i < BQ_TILE; i += SM89_NUM_THREADS) {
        s_m[i] = -INFINITY;
        s_l[i] =  0.0f;
    }

    // Per-warp register output accumulators, persistent across KV iterations.
    float O_acc_l[PV_PASSES][4];
    float O_acc_r[PV_PASSES][4];
    #pragma unroll
    for (int p = 0; p < PV_PASSES; ++p) {
        O_acc_l[p][0] = 0.f; O_acc_l[p][1] = 0.f;
        O_acc_l[p][2] = 0.f; O_acc_l[p][3] = 0.f;
        O_acc_r[p][0] = 0.f; O_acc_r[p][1] = 0.f;
        O_acc_r[p][2] = 0.f; O_acc_r[p][3] = 0.f;
    }

    // Async load Q -> s_q
    {
        const int vt = (BQ_TILE * HeadDim) / 4;
        for (int i = tid; i < vt; i += SM89_NUM_THREADS) {
            const int q = (i * 4) / HeadDim, d = (i * 4) % HeadDim;
            sm89_cp_async_l16(&s_q[q * HD_PAD + d],
                              &Q_bnh[(qi_block + q) * q_sM + d],
                              (qi_block + q < T));
        }
        sm89_cp_async_commit();
    }

    const int actual_q = (int)(
        ((int64_t)BQ_TILE < (T - qi_block)) ? (int64_t)BQ_TILE : (T - qi_block));
    if (actual_q <= 0) return;

    const int64_t max_kj = is_causal
        ? (((qi_block + (int64_t)actual_q) < T) ? (qi_block + (int64_t)actual_q) : T)
        : T;

    // Pre-fetch first K tile
    {
        const int vt = (BK_TILE * HeadDim) / 4;
        for (int i = tid; i < vt; i += SM89_NUM_THREADS) {
            const int k = (i * 4) / HeadDim, d = (i * 4) % HeadDim;
            sm89_cp_async_l16(&s_kv[0][k * HD_PAD + d],
                              &K_bnh[(int64_t)k * k_sM + d], (int64_t)k < T);
        }
        sm89_cp_async_commit();
    }
    sm89_cp_async_wait_group<0>();
    __syncthreads();

    auto loadA = [&](uint32_t r[4], int m_tile, int k8,
                     const float* As, int stride) __attribute__((always_inline)) {
        const int row  = m_tile * 16 + (lane & 15);
        const int col  = k8 + (lane >> 4) * 4;
        const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&As[row * stride + col]);
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
        r[0] += 0x1000u; r[1] += 0x1000u; r[2] += 0x1000u; r[3] += 0x1000u;
    };

    // Loads both left and right B-operand halves for the score GEMM (K stored n-major).
    auto loadB_nt_x4 = [&](uint32_t l[2], uint32_t r[2], int n_base, int k8,
                           const float* Bs, int stride) __attribute__((always_inline)) {
        const int row = n_base + (lane & 15);
        const int col = k8 + (lane >> 4) * 4;
        const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&Bs[row * stride + col]);
        uint32_t d0, d1, d2, d3;
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"
            : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
        l[0] = d0 + 0x1000u; l[1] = d2 + 0x1000u;
        r[0] = d1 + 0x1000u; r[1] = d3 + 0x1000u;
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

    auto gather = [&](float d[4], const float* src, int r_base, int c_base,
                      int stride) __attribute__((always_inline)) {
        const int r0 = r_base + (lane >> 2), r1 = r0 + 8;
        const int c  = c_base + (lane & 3) * 2;
        d[0] = src[r0 * stride + c];
        d[1] = src[r0 * stride + c + 1];
        d[2] = src[r1 * stride + c];
        d[3] = src[r1 * stride + c + 1];
    };
    (void)gather;  // unused

    // Main KV-tile loop
    for (int64_t kj_block = 0; kj_block < max_kj; kj_block += BK_TILE) {
        const int block_len = (int)(
            ((int64_t)BK_TILE < (T - kj_block)) ? (int64_t)BK_TILE : (T - kj_block));
        const int64_t next_kj_block = kj_block + BK_TILE;
        const bool    has_next      = (next_kj_block < max_kj);

        // 2a: async load V -> s_kv[1]
        {
            const int vt = (BK_TILE * HeadDim) / 4;
            for (int i = tid; i < vt; i += SM89_NUM_THREADS) {
                const int v = (i * 4) / HeadDim, d = (i * 4) % HeadDim;
                const int64_t g = kj_block + v;
                sm89_cp_async_l16(&s_kv[1][v * HD_PAD + d], &V_bnh[g * v_sM + d], g < T);
            }
            sm89_cp_async_commit();
        }

        // 2b: score GEMM Q@K^T -> s_scores
        for (int tile_idx = warp_id; tile_idx < SCORE_TOTAL; tile_idx += NUM_WARPS) {
            const int m_tile = tile_idx / SCORE_N_TILES;
            const int n_tile = tile_idx % SCORE_N_TILES;

            float acc_l[4] = {0.f, 0.f, 0.f, 0.f};
            float acc_r[4] = {0.f, 0.f, 0.f, 0.f};
            uint32_t frA[4], frB_l[2], frB_r[2];

            #pragma unroll
            for (int k = 0; k < SCORE_K_TILES; k++) {
                const int k8 = k * SM89_WMMA_K;
                loadA       (frA,   m_tile,      k8, s_q,     HD_PAD);
                loadB_nt_x4 (frB_l, frB_r, n_tile * 16, k8, s_kv[0], HD_PAD);
                mma_tf32_m16n8k8(acc_l[0], acc_l[1], acc_l[2], acc_l[3],
                                 frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
                mma_tf32_m16n8k8(acc_r[0], acc_r[1], acc_r[2], acc_r[3],
                                 frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
            }
            scatter(acc_l, s_scores, m_tile * 16, n_tile * 16,     BK_STRIDE);
            scatter(acc_r, s_scores, m_tile * 16, n_tile * 16 + 8, BK_STRIDE);
        }
        __syncthreads();

        // 2c: online softmax
        {
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const int     row       = warp_id * ROWS_PER_WARP + r;
                const int64_t qi_global = qi_block + row;
                const bool    qi_valid  = (qi_global < T);

                float cached_score[COLS_PER_THREAD];
                float row_max = -INFINITY;
                #pragma unroll
                for (int j = 0; j < COLS_PER_THREAD; ++j) {
                    const int col = j * 32 + lane;
                    float v = (col < block_len && qi_valid)
                              ? s_scores[row * BK_STRIDE + col] * scale : -INFINITY;
                    if (is_causal && (kj_block + col) > qi_global) v = -INFINITY;
                    cached_score[j] = v;
                    row_max = fmaxf(row_max, v);
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, off));

                const float m_old = s_m[row];
                const float m_new = qi_valid ? fmaxf(m_old, row_max) : m_old;
                const float alpha = (m_old == -INFINITY) ? 0.0f
                                  : (m_old == m_new)     ? 1.0f
                                  :                        expf(m_old - m_new);

                // O rescale deferred to PV stage; publish alpha for step 2e
                float row_sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < COLS_PER_THREAD; ++j) {
                    const int col = j * 32 + lane;
                    float exp_s = 0.0f;
                    if (cached_score[j] > -INFINITY && qi_valid && m_new > -INFINITY) {
                        exp_s = expf(cached_score[j] - m_new);
                        if (dropout_p > 0.0f && dropout_mask != nullptr) {
                            exp_s *= dropout_mask[
                                (bnh * T + qi_global) * T + (kj_block + col)];
                        }
                    }
                    s_scores[row * BK_STRIDE + col] = exp_s;
                    row_sum += exp_s;
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    row_sum += __shfl_xor_sync(0xffffffff, row_sum, off);

                if (lane == 0) {
                    s_l[row]     = alpha * s_l[row] + row_sum;
                    s_m[row]     = m_new;
                    s_alpha[row] = alpha;
                }
            }
        }
        __syncthreads();

        // 2d: prefetch next K tile
        if (has_next) {
            const int vt = (BK_TILE * HeadDim) / 4;
            for (int i = tid; i < vt; i += SM89_NUM_THREADS) {
                const int k = (i * 4) / HeadDim, d = (i * 4) % HeadDim;
                const int64_t g = next_kj_block + k;
                sm89_cp_async_l16(&s_kv[0][k * HD_PAD + d], &K_bnh[g * k_sM + d], g < T);
            }
            sm89_cp_async_commit();
        }

        // 2e: rescale O accumulators by alpha, then accumulate P@V
        sm89_cp_async_wait_group<1>();
        __syncthreads();

        #pragma unroll
        for (int pass = 0; pass < PV_PASSES; ++pass) {
            const int tile_id = warp_id + pass * NUM_WARPS;
            if (tile_id < PV_TOTAL) {
                const int m_tile = tile_id / PV_N_TILES;
                const int n_tile = tile_id % PV_N_TILES;

                // online softmax rescale of the two rows this lane owns
                const float a_lo = s_alpha[m_tile * 16 + (lane >> 2)];
                const float a_hi = s_alpha[m_tile * 16 + (lane >> 2) + 8];
                O_acc_l[pass][0] *= a_lo; O_acc_l[pass][1] *= a_lo;
                O_acc_l[pass][2] *= a_hi; O_acc_l[pass][3] *= a_hi;
                O_acc_r[pass][0] *= a_lo; O_acc_r[pass][1] *= a_lo;
                O_acc_r[pass][2] *= a_hi; O_acc_r[pass][3] *= a_hi;

                uint32_t frA[4], frB_l[2], frB_r[2];
                #pragma unroll
                for (int k = 0; k < PV_K_TILES; k++) {
                    const int k8 = k * SM89_WMMA_K;
                    loadA   (frA,   m_tile,          k8, s_scores, BK_STRIDE);
                    loadB_nn(frB_l, n_tile * 16,     k8, s_kv[1],  HD_PAD);
                    loadB_nn(frB_r, n_tile * 16 + 8, k8, s_kv[1],  HD_PAD);
                    mma_tf32_m16n8k8(O_acc_l[pass][0], O_acc_l[pass][1],
                                     O_acc_l[pass][2], O_acc_l[pass][3],
                                     frA[0], frA[1], frA[2], frA[3],
                                     frB_l[0], frB_l[1]);
                    mma_tf32_m16n8k8(O_acc_r[pass][0], O_acc_r[pass][1],
                                     O_acc_r[pass][2], O_acc_r[pass][3],
                                     frA[0], frA[1], frA[2], frA[3],
                                     frB_r[0], frB_r[1]);
                }
            }
        }
        __syncthreads();
    }

    // Scatter O accumulators -> smem, normalise, write O (reuse dead s_q region)
    float* s_out = s_q;

    #pragma unroll
    for (int pass = 0; pass < PV_PASSES; ++pass) {
        const int tile_id = warp_id + pass * NUM_WARPS;
        if (tile_id < PV_TOTAL) {
            const int m_tile = tile_id / PV_N_TILES;
            const int n_tile = tile_id % PV_N_TILES;
            scatter(O_acc_l[pass], s_out, m_tile * 16, n_tile * 16,     HD_PAD);
            scatter(O_acc_r[pass], s_out, m_tile * 16, n_tile * 16 + 8, HD_PAD);
        }
    }
    __syncthreads();

    {
        const int vt = (actual_q * HeadDim) / 4;
        for (int i = tid; i < vt; i += SM89_NUM_THREADS) {
            const int q = (i * 4) / HeadDim, d = (i * 4) % HeadDim;
            const float inv_l = (s_l[q] > 0.0f) ? (1.0f / s_l[q]) : 0.0f;
            float4* ptr = (float4*)&s_out[q * HD_PAD + d];
            float4 v = *ptr;
            v.x *= inv_l; v.y *= inv_l; v.z *= inv_l; v.w *= inv_l;
            *(float4*)&O_bnh[(qi_block + q) * o_sM + d] = v;
        }
    }
    __syncthreads();

    // Write LSE
    for (int i = tid; i < actual_q; i += SM89_NUM_THREADS) {
        const float m = s_m[i], l = s_l[i];
        LSE_bnh[qi_block + i] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
    }
}

template __global__ void
fused_attn_forward_kernel_tc_sm89<64, 64, 64, 1>(MemEfficientFwdParams);

