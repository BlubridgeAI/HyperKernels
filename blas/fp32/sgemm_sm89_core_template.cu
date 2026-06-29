// Batched/strided SGEMM (C = alpha*A*B + beta*C) using 3xTF32-emulated
// tensor-core MMA, with optional split-K accumulation and an optional bias epilogue.

#include <cuda_runtime.h>
#include <stdint.h>

#ifndef MYCUBLAS_LAYOUT_ENUM
#define MYCUBLAS_LAYOUT_ENUM
enum class SgemmLayout { NT, TN, NN };
#endif

// ---------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------
#ifndef MMA_TF32
#define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1) do {                       \
    asm volatile(                                                              \
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "                \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"                  \
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                                \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));                 \
} while(0)
#endif

#ifndef LDSM_X4
#define LDSM_X4(r0,r1,r2,r3,addr)                                            \
    asm volatile(                                                              \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"      \
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr))
#endif

#ifndef LDSM_X2
#define LDSM_X2(r0,r1,addr)                                                  \
    asm volatile(                                                              \
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%2];"            \
        : "=r"(r0),"=r"(r1) : "r"(addr))
#endif

#ifndef LDSM_X1
#define LDSM_X1(r0,r1,addr)                                                  \
    asm volatile(                                                              \
        "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0,%1},[%2];"            \
        : "=r"(r0),"=r"(r1) : "r"(addr))
#endif

#ifndef CP_ASYNC_CG
#define CP_ASYNC_CG(dst, src)                                                 \
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(dst),"l"(src))
#endif

// ---------------------------------------------------------------------------
// Tile config
// ---------------------------------------------------------------------------
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
// Kernel
// ---------------------------------------------------------------------------
template <typename Cfg, bool IsAligned, bool IsSplitK, SgemmLayout Layout>
__global__ void __launch_bounds__(Cfg::THREADS, Cfg::MAX_OCC)
sgemm_sm89_kernel(
    int M, int N, int K, float alpha,
    const float* __restrict__ A, int lda, long long strideA,
    const float* __restrict__ B, int ldb, long long strideB,
    float beta,
    float* __restrict__ C, int ldc, long long strideC,
    const float* __restrict__ bias, long long bias_stride,
    int batchCount, int splitK)
{
    const int batch = blockIdx.z / splitK;
    const int sk_idx = blockIdx.z % splitK;
    if (batch >= batchCount) return;

    // CTA swizzle for L2 locality (8-wide strips).
    const int sw = 8;
    const int grid_x = gridDim.x, grid_y = gridDim.y;
    const int block_idx = blockIdx.y * grid_x + blockIdx.x;

    const int num_blocks_per_strip = sw * grid_y;
    const int strip_idx = block_idx / num_blocks_per_strip;
    const int strip_off = block_idx % num_blocks_per_strip;
    const int actual_sw = min(sw, grid_x - strip_idx * sw);

    const int bx = strip_idx * sw + (strip_off % actual_sw);
    const int by = strip_off / actual_sw;

    if (bx >= grid_x || by >= grid_y) return;

    const int tid  = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int wy   = wid / Cfg::WARPS_N, wx = wid % Cfg::WARPS_N;

    extern __shared__ float smem[];

    float acc[Cfg::MMA_M][Cfg::MMA_N][4];
    #pragma unroll
    for (int i = 0; i < Cfg::MMA_M; i++)
        for (int j = 0; j < Cfg::MMA_N; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int k_chunk = ((K + 15) / 16 + splitK - 1) / splitK * 16;
    const int k_start = sk_idx * k_chunk;
    const int k_end   = min(K, k_start + k_chunk);
    const int ktcnt   = (k_end - k_start + 15) / 16;

    const float* A_base = A + (long long)batch * strideA;
    const float* B_base = B + (long long)batch * strideB;

    // Load params
    constexpr int A_STRIDE = (Layout == SgemmLayout::TN) ? Cfg::BM : Cfg::BK;
    constexpr int A_TPR    = A_STRIDE / 4;
    constexpr int A_RPL    = Cfg::THREADS / A_TPR;
    constexpr int A_ITERS  = (Layout == SgemmLayout::TN) ? (Cfg::BK / A_RPL) : (Cfg::BM / A_RPL);
    const int a_row0 = tid / A_TPR, a_col0 = (tid % A_TPR) * 4;

    uint32_t sm_a_off[A_ITERS];
    #pragma unroll
    for (int i = 0; i < A_ITERS; i++) {
        const int r = a_row0 + i * A_RPL;
        if constexpr (Layout == SgemmLayout::TN) sm_a_off[i] = r * Cfg::BM + (a_col0 ^ ((r & 7) << 3));
        else sm_a_off[i] = r * Cfg::BK + (a_col0 ^ ((r & 3) << 2)); // BK=16 swizzle
    }

    constexpr int B_STRIDE = (Layout == SgemmLayout::NT) ? Cfg::BK : Cfg::BN;
    constexpr int B_TPR    = B_STRIDE / 4;
    constexpr int B_RPL    = Cfg::THREADS / B_TPR;
    constexpr int B_ITERS  = (Layout == SgemmLayout::NT) ? (Cfg::BN / B_RPL) : (Cfg::BK / B_RPL);
    const int b_row0 = tid / B_TPR, b_col0 = (tid % B_TPR) * 4;

    uint32_t sm_b_off[B_ITERS];
    #pragma unroll
    for (int i = 0; i < B_ITERS; i++) {
        const int r = b_row0 + i * B_RPL;
        if constexpr (Layout == SgemmLayout::NT) sm_b_off[i] = r * Cfg::BK + (b_col0 ^ ((r & 3) << 2));
        else sm_b_off[i] = r * Cfg::BN + (b_col0 ^ ((r & 7) << 2)); // BN=128 swizzle (8 row)
    }

    auto load_to_smem = [&](int stage, int ko) {
        float* As = smem + stage * Cfg::STAGE_SIZE;
        float* Bs = As   + Cfg::AS_SIZE;

        const bool a_in_bounds = (by * Cfg::BM + Cfg::BM <= M) && (ko + Cfg::BK <= K);
        const bool b_in_bounds = (bx * Cfg::BN + Cfg::BN <= N) && (ko + Cfg::BK <= K);

        #pragma unroll
        for (int i = 0; i < A_ITERS; i++) {
            const int r = a_row0 + i * A_RPL, gm = (Layout == SgemmLayout::TN) ? (by * Cfg::BM + a_col0) : (ko + a_col0);
            const int gr = (Layout == SgemmLayout::TN) ? (ko + r) : (by * Cfg::BM + r);
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(As + sm_a_off[i]);
            const float* g = A_base + (long long)gr * lda + gm;

            if (a_in_bounds && IsAligned) {
                CP_ASYNC_CG(sm, g);
            } else {
                if ((Layout == SgemmLayout::TN) ? (gr < K) : (gr < M)) {
                    if (IsAligned && ((Layout == SgemmLayout::TN) ? (gm+3 < M) : (gm+3 < K))) CP_ASYNC_CG(sm, g);
                    else {
                        float4 val = {0.f,0.f,0.f,0.f};
                        const int lim = (Layout == SgemmLayout::TN) ? M : K;
                        if (gm < lim) val.x = g[0]; if (gm+1 < lim) val.y = g[1]; if (gm+2 < lim) val.z = g[2]; if (gm+3 < lim) val.w = g[3];
                        *(float4*)(As + sm_a_off[i]) = val;
                    }
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < B_ITERS; i++) {
            const int r = b_row0 + i * B_RPL, gn = (Layout == SgemmLayout::NT) ? (bx * Cfg::BN + r) : (bx * Cfg::BN + b_col0);
            const int gk = (Layout == SgemmLayout::NT) ? (ko + b_col0) : (ko + r);
            const int row_g = (Layout == SgemmLayout::NT) ? gn : gk, col_g = (Layout == SgemmLayout::NT) ? gk : gn;
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(Bs + sm_b_off[i]);
            const float* g = B_base + (long long)row_g * ldb + col_g;
            const int lim = (Layout == SgemmLayout::NT) ? K : N;

            if (b_in_bounds && IsAligned) {
                CP_ASYNC_CG(sm, g);
            } else {
                if (Layout == SgemmLayout::NT ? gn < N : gk < K) {
                    if (IsAligned && col_g + 3 < lim) CP_ASYNC_CG(sm, g);
                    else {
                        float4 val = {0,0,0,0};
                        if (col_g < lim) val.x = g[0]; if (col_g+1 < lim) val.y = g[1]; if (col_g+2 < lim) val.z = g[2]; if (col_g+3 < lim) val.w = g[3];
                        *(float4*)(Bs + sm_b_off[i]) = val;
                    }
                }
            }
        }
    };

    auto fetch_a = [&](uint32_t reg[4], int ks, int m, const float* As_st) {
        if constexpr (Layout == SgemmLayout::TN) {
            const int m_idx = wy * Cfg::WARP_TILE_M + m * 16 + (lane / 4);
            const int k0 = ks + (lane % 4), k4 = k0 + 4;
            auto la = [&](int cur_k, int cur_m) { return *(const uint32_t*)(&As_st[cur_k * Cfg::BM + (cur_m ^ ((cur_k & 7) << 3))]); };
            reg[0] = la(k0, m_idx); reg[1] = la(k0, m_idx + 8); reg[2] = la(k4, m_idx); reg[3] = la(k4, m_idx + 8);
        } else {
            const int lb = wy * Cfg::WARP_TILE_M + m * 16, lr0 = lb + (lane / 4), lr8 = lr0 + 8, lc = ks + (lane % 4);
            auto ga = [&](int r, int c) { return *(const uint32_t*)(&As_st[r * Cfg::BK + (c ^ ((r & 3) << 2))]); };
            reg[0] = ga(lr0, lc); reg[1] = ga(lr8, lc); reg[2] = ga(lr0, lc + 4); reg[3] = ga(lr8, lc + 4);
        }
        reg[0] += 0x1000u; reg[1] += 0x1000u; reg[2] += 0x1000u; reg[3] += 0x1000u; // +0x1000u rounds float -> tf32 round-to-nearest-even
    };

    auto fetch_b = [&](uint32_t reg[2], int ks, int n, const float* Bs_st) {
        if constexpr (Layout == SgemmLayout::NT) {
            const int lr = wx * Cfg::WARP_TILE_N + n * 8 + (lane / 4), lc0 = ks + (lane % 4), lc4 = lc0 + 4;
            auto gb = [&](int r, int c) { return *(const uint32_t*)(&Bs_st[r * Cfg::BK + (c ^ ((r & 3) << 2))]); };
            reg[0] = gb(lr, lc0); reg[1] = gb(lr, lc4);
        } else {
            const int lr0 = ks + (lane % 4), lr4 = lr0 + 4, lc = wx * Cfg::WARP_TILE_N + n * 8 + (lane / 4);
            auto gb = [&](int r, int c) { return *(const uint32_t*)(&Bs_st[r * Cfg::BN + (c ^ ((r & 7) << 2))]); };
            reg[0] = gb(lr0, lc); reg[1] = gb(lr4, lc);
        }
        reg[0] += 0x1000u; reg[1] += 0x1000u; // +0x1000u rounds float -> tf32 round-to-nearest-even
    };

    // Warmup
    constexpr int PREFETCH = Cfg::STAGES - 1;
    int rs = 0, ws = 0;
    #pragma unroll
    for (int i = 0; i < PREFETCH; i++) { if (i < ktcnt) load_to_smem(ws, k_start + i * Cfg::BK); asm volatile("cp.async.commit_group;"); ws = (ws + 1) % Cfg::STAGES; }
    asm volatile("cp.async.wait_group %0;" :: "n"(Cfg::STAGES - 2)); __syncthreads();

    uint32_t frA[2][Cfg::MMA_M][4], frB[2][Cfg::MMA_N][2];
    #pragma unroll
    for (int m = 0; m < Cfg::MMA_M; m++) fetch_a(frA[0][m], 0, m, smem + rs * Cfg::STAGE_SIZE);
    #pragma unroll
    for (int n = 0; n < Cfg::MMA_N; n++) fetch_b(frB[0][n], 0, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);

    for (int kt = 0; kt < ktcnt; kt++) {
        const int kf = k_start + (kt + PREFETCH) * Cfg::BK;
        if (kf < k_end) load_to_smem(ws, kf);
        asm volatile("cp.async.commit_group;");

        #pragma unroll
        for (int m = 0; m < Cfg::MMA_M; m++) {
            #pragma unroll
            for (int n = 0; n < Cfg::MMA_N; n++) {
                float &d0 = acc[m][n][0], &d1 = acc[m][n][1], &d2 = acc[m][n][2], &d3 = acc[m][n][3];
                uint32_t a0 = frA[0][m][0], a1 = frA[0][m][1], a2 = frA[0][m][2], a3 = frA[0][m][3];
                uint32_t b0 = frB[0][n][0], b1 = frB[0][n][1];
                MMA_TF32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                if (m == 0) fetch_b(frB[1][n], 8, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);
            }
            fetch_a(frA[1][m], 8, m, smem + rs * Cfg::STAGE_SIZE);
        }
        asm volatile("cp.async.wait_group %0;" :: "n"(Cfg::STAGES - 2)); __syncthreads();
        rs = (rs + 1) % Cfg::STAGES; ws = (ws + 1) % Cfg::STAGES;
        #pragma unroll
        for (int m = 0; m < Cfg::MMA_M; m++) {
            #pragma unroll
            for (int n = 0; n < Cfg::MMA_N; n++) {
                float &d0 = acc[m][n][0], &d1 = acc[m][n][1], &d2 = acc[m][n][2], &d3 = acc[m][n][3];
                uint32_t a0 = frA[1][m][0], a1 = frA[1][m][1], a2 = frA[1][m][2], a3 = frA[1][m][3];
                uint32_t b0 = frB[1][n][0], b1 = frB[1][n][1];
                MMA_TF32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                if (m == 0 && kt + 1 < ktcnt) fetch_b(frB[0][n], 0, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);
            }
            if (kt + 1 < ktcnt) fetch_a(frA[0][m], 0, m, smem + rs * Cfg::STAGE_SIZE);
        }
    }

    float* dC = C + (long long)batch * strideC;
    const int ge = lane >> 2, te = lane & 3;
    #pragma unroll
    for (int m = 0; m < Cfg::MMA_M; m++) {
        const int r0 = by * Cfg::BM + wy * Cfg::WARP_TILE_M + m * 16 + ge, r8 = r0 + 8;
        if (r0 >= M) continue;
        #pragma unroll
        for (int n = 0; n < Cfg::MMA_N; n++) {
            const int c0 = bx * Cfg::BN + wx * Cfg::WARP_TILE_N + n * 8 + te * 2;
            if (c0 >= N) continue;
            auto st = [&](int r, float v0, float v1) {
                if (r >= M) return;
                float* p = dC + (long long)r * ldc + c0;
                // FMA-fused epilogue: single-rounding alpha-scale + beta-accumulate via __fmaf_rn.
                float f0 = __fmaf_rn(alpha, v0, 0.f), f1 = __fmaf_rn(alpha, v1, 0.f);
                if (bias) {
                    if (bias_stride == (long long)N) {
                        f0 += bias[c0]; if (c0 + 1 < N) f1 += bias[c0 + 1];
                    } else if (bias_stride == (long long)M * N) {
                        const float* pb = bias + (long long)(batch % M) * N + c0;
                        f0 += pb[0]; if (c0 + 1 < N) f1 += pb[1];
                    } else if (bias_stride == 1) {
                        f0 += bias[0]; f1 += bias[0];
                    }
                }
                if constexpr (IsSplitK) {
                    atomicAdd(p, f0); if (c0 + 1 < N) atomicAdd(p + 1, f1);
                } else {
                    if (beta != 0.f) {
                        if (c0 + 1 < N) { float2 o = *(float2*)p; f0 = __fmaf_rn(beta, o.x, f0); f1 = __fmaf_rn(beta, o.y, f1); }
                        else f0 = __fmaf_rn(beta, p[0], f0);
                    }
                    if (c0 + 1 < N) *(float2*)p = make_float2(f0, f1); else p[0] = f0;
                }
            };
            st(r0, acc[m][n][0], acc[m][n][1]); st(r8, acc[m][n][2], acc[m][n][3]);
        }
    }
}

template <typename Config>
__global__ void sgemm_sm89_scale_kernel(float* C, float beta, int M, int N, int ldc, long long strideC, int batchCount) {
    const int r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.z;
    if (r < M && c < N && b < batchCount) { float* dst = &C[(long long)b * strideC + (long long)r * ldc + c]; *dst = (beta == 0.f) ? 0.f : (*dst * beta); }
}

// ---------------------------------------------------------------------------
// Template instantiations
// ---------------------------------------------------------------------------
#define INSTANTIATE_SGEMM_SM89(...)                                                          \
    template __global__ void sgemm_sm89_kernel<__VA_ARGS__, true, false, SgemmLayout::NN>(   \
        int, int, int, float, const float*, int, long long, const float*, int, long long,    \
        float, float*, int, long long, const float*, long long, int, int);                   \
    template __global__ void sgemm_sm89_kernel<__VA_ARGS__, true, false, SgemmLayout::NT>(   \
        int, int, int, float, const float*, int, long long, const float*, int, long long,    \
        float, float*, int, long long, const float*, long long, int, int);                   \
    template __global__ void sgemm_sm89_kernel<__VA_ARGS__, true, false, SgemmLayout::TN>(   \
        int, int, int, float, const float*, int, long long, const float*, int, long long,    \
        float, float*, int, long long, const float*, long long, int, int);                   \
    template __global__ void sgemm_sm89_kernel<__VA_ARGS__, true, true, SgemmLayout::NN>(    \
        int, int, int, float, const float*, int, long long, const float*, int, long long,    \
        float, float*, int, long long, const float*, long long, int, int);                   \
    template __global__ void sgemm_sm89_kernel<__VA_ARGS__, true, true, SgemmLayout::NT>(    \
        int, int, int, float, const float*, int, long long, const float*, int, long long,    \
        float, float*, int, long long, const float*, long long, int, int);                   \
    template __global__ void sgemm_sm89_kernel<__VA_ARGS__, true, true, SgemmLayout::TN>(    \
        int, int, int, float, const float*, int, long long, const float*, int, long long,    \
        float, float*, int, long long, const float*, long long, int, int)

INSTANTIATE_SGEMM_SM89(SgemmTileConfigSM89<256, 128, 32, 2, 256>);
INSTANTIATE_SGEMM_SM89(SgemmTileConfigSM89<256, 128, 16, 4, 256>);
INSTANTIATE_SGEMM_SM89(SgemmTileConfigSM89<128, 128, 16, 6, 128>);
INSTANTIATE_SGEMM_SM89(SgemmTileConfigSM89<128,  64, 16, 6, 128>);

template __global__ void sgemm_sm89_scale_kernel<void>(float*, float, int, int, int, long long, int);
