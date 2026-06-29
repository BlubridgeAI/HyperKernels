// Batched fused SGEMM + bias: C = alpha * (A * B) + beta * bias, A:[M,K] row-major, B:[K,N] row-major (NN).
// 3xTF32 emulation (+0x1000u operand bias) with an FMA-fused epilogue.

#include <cuda_runtime.h>
#include <stdint.h>

#ifndef MMA_TF32
#define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1) do {            \
    asm volatile(                                                   \
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "     \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"       \
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                     \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));      \
} while(0)
#endif

// Tile configuration that deduces load patterns and warp layout from tile dims.
template <int BM_, int BN_, int BK_, int STAGES_, int THREADS_>
struct SgemmTileConfig {
    static constexpr int BM = BM_;
    static constexpr int BN = BN_;
    static constexpr int BK = BK_;
    static constexpr int STAGES = STAGES_;
    static constexpr int THREADS = THREADS_;

    static constexpr int AS_SIZE = BM * BK;
    static constexpr int BS_SIZE = BN * BK;
    static constexpr int STAGE_SIZE = AS_SIZE + BS_SIZE;

    // NT load config: A:[M,K] and B:[N,K] contiguous along K.
    static constexpr int NT_VEC_A = (BK % 4 == 0) ? 4 : 1;
    static constexpr int NT_THREADS_PER_ROW_A = BK / NT_VEC_A;
    static constexpr int NT_ROWS_PER_ITER_A = THREADS / NT_THREADS_PER_ROW_A;
    static constexpr int NT_LOAD_ITERS_A = BM / NT_ROWS_PER_ITER_A;

    static constexpr int NT_VEC_B = (BK % 4 == 0) ? 4 : 1;
    static constexpr int NT_THREADS_PER_ROW_B = BK / NT_VEC_B;
    static constexpr int NT_ROWS_PER_ITER_B = THREADS / NT_THREADS_PER_ROW_B;
    static constexpr int NT_LOAD_ITERS_B = BN / NT_ROWS_PER_ITER_B;

    // TN load config: A:[K,M] and B:[K,N] contiguous along M/N.
    static constexpr int TN_VEC_A = (BM % 4 == 0) ? 4 : 1;
    static constexpr int TN_THREADS_PER_ROW_A = BM / TN_VEC_A;
    static constexpr int TN_ROWS_PER_ITER_A = THREADS / TN_THREADS_PER_ROW_A;
    static constexpr int TN_LOAD_ITERS_A = BK / TN_ROWS_PER_ITER_A;

    static constexpr int TN_VEC_B = (BN % 4 == 0) ? 4 : 1;
    static constexpr int TN_THREADS_PER_ROW_B = BN / TN_VEC_B;
    static constexpr int TN_ROWS_PER_ITER_B = THREADS / TN_THREADS_PER_ROW_B;
    static constexpr int TN_LOAD_ITERS_B = BK / TN_ROWS_PER_ITER_B;

    // Warp layout: most square-like arrangement that fits the BM/BN ratio.
    static constexpr int WARP_COUNT = THREADS / 32;
    static constexpr int WARPS_M = (BM >= BN * 2) ? (WARP_COUNT >= 4 ? 4 : 2) :
                                   (BM == BN && WARP_COUNT >= 4 ? 2 :
                                   (WARP_COUNT >= 2 ? 1 : 1));
    static constexpr int WARPS_N = WARP_COUNT / WARPS_M;

    static constexpr int WARP_TILE_M = BM / WARPS_M;
    static constexpr int WARP_TILE_N = BN / WARPS_N;

    static constexpr int MMA_M = WARP_TILE_M / 16;
    static constexpr int MMA_N = WARP_TILE_N / 8;
};

template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned>
__global__ void __launch_bounds__(THREADS, 1)
sgemm_addmm_sm89_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda, long long strideA,
    const float* __restrict__ B, int ldb, long long strideB,
    float beta,
    const float* __restrict__ bias, int64_t bias_numel,
    float* __restrict__ C, int ldc, long long strideC,
    int batchCount)
{
    using Config = SgemmTileConfig<BM, BN, BK, STAGES, THREADS>;

    const int batch = blockIdx.z;
    if (batch >= batchCount) return;

    const int sw = 8;
    const int grid_x = (N + BN - 1) / BN, grid_y = (M + BM - 1) / BM;
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
    const int wy   = wid / Config::WARPS_N;
    const int wx   = wid % Config::WARPS_N;

    extern __shared__ float smem[];

    float acc[Config::MMA_M][Config::MMA_N][4];
    #pragma unroll
    for (int i = 0; i < Config::MMA_M; i++)
        #pragma unroll
        for (int j = 0; j < Config::MMA_N; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const float* gA_ptr[Config::NT_LOAD_ITERS_A];
    const float* gB_ptr[Config::TN_LOAD_ITERS_B];

    #pragma unroll
    for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
        const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
        const int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
        gA_ptr[i] = A + (long long)batch * strideA
                      + (long long)(by * BM + r) * lda + c;
    }
    #pragma unroll
    for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
        const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B;
        const int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
        gB_ptr[i] = B + (long long)batch * strideB
                      + (long long)r * ldb + (bx * BN + c);
    }

    uint32_t sm_a_off[Config::NT_LOAD_ITERS_A];
    #pragma unroll
    for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
        const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
        const int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
        const int sc = c ^ ((r & 3) << 2);
        sm_a_off[i] = r * Config::BK + sc;
    }

    uint32_t sm_b_off[Config::TN_LOAD_ITERS_B];
    #pragma unroll
    for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
        const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B;
        const int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
        const int sc = c ^ ((r & 7) << 2);
        sm_b_off[i] = r * Config::BN + sc;
    }

    auto load_to_stage = [&](int stage, int ko) {
        float* As = smem + stage * Config::STAGE_SIZE;
        float* Bs = As   + Config::AS_SIZE;

        const bool a_in_bounds = (by * BM + BM <= M) && (ko + BK <= K);
        const bool b_in_bounds = (bx * BN + BN <= N) && (ko + BK <= K);

        #pragma unroll
        for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
            const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A, c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
            const int gr = by * BM + r, gc = ko + c;
            uint32_t sm = __cvta_generic_to_shared(As + sm_a_off[i]);
            if (a_in_bounds) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sm), "l"(gA_ptr[i]));
            } else {
                if (gr < M && r < BM) {
                    int bytes = max(0, min(16, (K - gc) * 4));
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm), "l"(gA_ptr[i]), "r"(bytes));
                } else if (r < BM) { *(float4*)(As + sm_a_off[i]) = {0,0,0,0}; }
            }
            gA_ptr[i] += BK;
        }
        #pragma unroll
        for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
            const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B, c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
            const int gk = ko + r, gn = bx * BN + c;
            uint32_t sm = __cvta_generic_to_shared(Bs + sm_b_off[i]);
            if (b_in_bounds) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sm), "l"(gB_ptr[i]));
            } else {
                if (gk < K && r < BK) {
                    int bytes = max(0, min(16, (N - gn) * 4));
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm), "l"(gB_ptr[i]), "r"(bytes));
                } else if (r < BK) { *(float4*)(Bs + sm_b_off[i]) = {0,0,0,0}; }
            }
            gB_ptr[i] += (long long)BK * ldb;
        }
    };

    const int g_sh = lane / 4, t_sh = lane % 4;

    auto load_frA = [&](uint32_t reg[4], int ks, int mi, int st) {
        float* As = smem + st * Config::STAGE_SIZE;
        const int lb = wy * Config::WARP_TILE_M + mi * 16, lr0 = lb + (lane / 4), lr8 = lr0 + 8, lc = ks + (lane % 4);
        auto ga = [&](int r, int c) { return *(const uint32_t*)(&As[r * Config::BK + (c ^ ((r & 3) << 2))]); };
        reg[0] = ga(lr0, lc); reg[1] = ga(lr8, lc); reg[2] = ga(lr0, lc+4); reg[3] = ga(lr8, lc+4);
        reg[0] += 0x1000u; reg[1] += 0x1000u; reg[2] += 0x1000u; reg[3] += 0x1000u;
    };

    auto load_frB = [&](uint32_t reg[2], int ks, int ni, int st) {
        float* Bs = smem + st * Config::STAGE_SIZE + Config::AS_SIZE;
        const int lr0 = ks + (lane % 4), lr4 = lr0 + 4, lc = wx * Config::WARP_TILE_N + ni * 8 + (lane / 4);
        auto gb = [&](int r, int c) { return *(const uint32_t*)(&Bs[r * Config::BN + (c ^ ((r & 7) << 2))]); };
        reg[0] = gb(lr0, lc); reg[1] = gb(lr4, lc);
        reg[0] += 0x1000u; reg[1] += 0x1000u;
    };

    load_to_stage(0, 0);
    asm volatile("cp.async.commit_group;\n");
    #pragma unroll
    for (int s = 1; s < Config::STAGES - 1; s++) {
        if (s * Config::BK < K) load_to_stage(s, s * Config::BK);
        asm volatile("cp.async.commit_group;\n");
    }

    int ws = Config::STAGES - 1, rs = 0;
    uint32_t frA[2][Config::MMA_M][4], frB[2][Config::MMA_N][2];

    asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2));
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < Config::MMA_M; i++) load_frA(frA[0][i], 0, i, rs);
    #pragma unroll
    for (int j = 0; j < Config::MMA_N; j++) load_frB(frB[0][j], 0, j, rs);

    for (int k = 0; k < K; k += Config::BK) {
        if (k + (Config::STAGES - 1) * Config::BK < K)
            load_to_stage(ws, k + (Config::STAGES - 1) * Config::BK);
        asm volatile("cp.async.commit_group;\n");

        #pragma unroll
        for (int ks = 0; ks < Config::BK; ks += 16) {
            #pragma unroll
            for (int i = 0; i < Config::MMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < Config::MMA_N; j++) {
                    MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                             frA[0][i][0], frA[0][i][1], frA[0][i][2], frA[0][i][3],
                             frB[0][j][0], frB[0][j][1]);
                    if (i == 0) load_frB(frB[1][j], ks + 8, j, rs);
                }
                load_frA(frA[1][i], ks + 8, i, rs);
            }

            #pragma unroll
            for (int i = 0; i < Config::MMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < Config::MMA_N; j++) {
                    MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                             frA[1][i][0], frA[1][i][1], frA[1][i][2], frA[1][i][3],
                             frB[1][j][0], frB[1][j][1]);
                }
            }

            if (ks + 16 < Config::BK) {
                #pragma unroll
                for (int i = 0; i < Config::MMA_M; i++) {
                    #pragma unroll
                    for (int j = 0; j < Config::MMA_N; j++) if (i == 0) load_frB(frB[0][j], ks + 16, j, rs);
                    load_frA(frA[0][i], ks + 16, i, rs);
                }
            } else if (k + Config::BK < K) {
                asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2));
                __syncthreads();
                rs = (rs + 1) % Config::STAGES;
                ws = (ws + 1) % Config::STAGES;
                #pragma unroll
                for (int i = 0; i < Config::MMA_M; i++) {
                    #pragma unroll
                    for (int j = 0; j < Config::MMA_N; j++) if (i == 0) load_frB(frB[0][j], 0, j, rs);
                    load_frA(frA[0][i], 0, i, rs);
                }
            }
        }
    }

    const int g_epi = lane / 4, t_epi = lane % 4;
    float* dC = C + (long long)batch * strideC;

    #pragma unroll
    for (int i = 0; i < Config::MMA_M; i++) {
        #pragma unroll
        for (int j = 0; j < Config::MMA_N; j++) {
            const int r0 = by * BM + wy * Config::WARP_TILE_M + i * 16 + g_epi;
            const int r8 = r0 + 8;
            const int c0 = bx * BN + wx * Config::WARP_TILE_N + j * 8 + t_epi * 2;
            const int c1 = c0 + 1;

            float b0 = 0.f, b1 = 0.f;
            if (bias) {
                if (bias_numel == 1) {
                    b0 = b1 = bias[0];
                } else if (bias_numel == (int64_t)N) {
                    if (c0 < N) b0 = bias[c0];
                    if (c1 < N) b1 = bias[c1];
                } else if (bias_numel == (int64_t)M * N) {
                    const float* pb = bias + (long long)(batch % M) * N + c0;
                    if (c0 < N) b0 = pb[0];
                    if (c1 < N) b1 = pb[1];
                }
            }

            // FMA-fused epilogue with float2 vectorized stores.
            auto store_pair = [&](int r, int c, float f0, float f1, float b0, float b1) __attribute__((always_inline)) {
                if (r >= M || c >= N) return;
                float* dst = &dC[(long long)r * ldc + c];

                float s0 = __fmaf_rn(alpha, f0, 0.f);
                float s1 = __fmaf_rn(alpha, f1, 0.f);

                float res0, res1;
                if (beta == 0.f) {
                    res0 = s0;
                    res1 = s1;
                } else {
                    if (c + 1 < N) {
                        float2 o = *(const float2*)dst;
                        res0 = __fmaf_rn(beta, o.x, s0);
                        res1 = __fmaf_rn(beta, o.y, s1);
                    } else {
                        res0 = __fmaf_rn(beta, dst[0], s0);
                        res1 = s1;
                    }
                }
                if (bias) {
                    res0 = __fmaf_rn(beta, b0, res0);
                    res1 = __fmaf_rn(beta, b1, res1);
                }

                if (c + 1 < N) {
                    *(float2*)dst = make_float2(res0, res1);
                } else {
                    dst[0] = res0;
                }
            };

            store_pair(r0, c0, acc[i][j][0], acc[i][j][1], b0, b1);
            store_pair(r8, c0, acc[i][j][2], acc[i][j][3], b0, b1);
        }
    }
}

// Tile configs matching the launch heuristic: 256x128, 128x128 (BK gated by K), 64x64.
template __global__ void sgemm_addmm_sm89_kernel<256, 128, 32, 2, 256, true>(
    int, int, int, float, const float*, int, long long, const float*, int, long long,
    float, const float*, int64_t, float*, int, long long, int);
template __global__ void sgemm_addmm_sm89_kernel<256, 128, 32, 2, 256, false>(
    int, int, int, float, const float*, int, long long, const float*, int, long long,
    float, const float*, int64_t, float*, int, long long, int);
template __global__ void sgemm_addmm_sm89_kernel<128, 128, 32, 3, 128, true>(
    int, int, int, float, const float*, int, long long, const float*, int, long long,
    float, const float*, int64_t, float*, int, long long, int);
template __global__ void sgemm_addmm_sm89_kernel<128, 128, 32, 3, 128, false>(
    int, int, int, float, const float*, int, long long, const float*, int, long long,
    float, const float*, int64_t, float*, int, long long, int);
template __global__ void sgemm_addmm_sm89_kernel<128, 128, 16, 6, 128, true>(
    int, int, int, float, const float*, int, long long, const float*, int, long long,
    float, const float*, int64_t, float*, int, long long, int);
template __global__ void sgemm_addmm_sm89_kernel<128, 128, 16, 6, 128, false>(
    int, int, int, float, const float*, int, long long, const float*, int, long long,
    float, const float*, int64_t, float*, int, long long, int);
template __global__ void sgemm_addmm_sm89_kernel<64, 64, 16, 6, 64, true>(
    int, int, int, float, const float*, int, long long, const float*, int, long long,
    float, const float*, int64_t, float*, int, long long, int);
template __global__ void sgemm_addmm_sm89_kernel<64, 64, 16, 6, 64, false>(
    int, int, int, float, const float*, int, long long, const float*, int, long long,
    float, const float*, int64_t, float*, int, long long, int);
