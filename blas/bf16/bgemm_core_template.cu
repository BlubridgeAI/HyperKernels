#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>

#ifndef MMA_M16N8K16_F32_BF16
#define MMA_M16N8K16_F32_BF16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1) \
    asm volatile( \
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};" \
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
#endif

template <int BM_, int BN_, int BK_, int STAGES_, int THREADS_>
struct BgemmTileConfig {
    static constexpr int BM = BM_;
    static constexpr int BN = BN_;
    static constexpr int BK = BK_;
    static constexpr int STAGES = STAGES_;
    static constexpr int THREADS = THREADS_;

    static constexpr int AS_SIZE = BM * BK;
    static constexpr int BS_SIZE = BN * BK;
    static constexpr int STAGE_SIZE = AS_SIZE + BS_SIZE;

    static constexpr int VEC_SIZE = 8;
    static constexpr int NT_VEC_A = 8;
    static constexpr int NT_THREADS_PER_ROW_A = BK / NT_VEC_A;
    static constexpr int NT_ROWS_PER_ITER_A   = THREADS / NT_THREADS_PER_ROW_A;
    static constexpr int NT_LOAD_ITERS_A       = BM / NT_ROWS_PER_ITER_A;

    static constexpr int NT_VEC_B = 8;
    static constexpr int NT_THREADS_PER_ROW_B = BK / NT_VEC_B;
    static constexpr int NT_ROWS_PER_ITER_B   = THREADS / NT_THREADS_PER_ROW_B;
    static constexpr int NT_LOAD_ITERS_B       = BN / NT_ROWS_PER_ITER_B;

    static constexpr int TN_VEC_A = 8;
    static constexpr int TN_THREADS_PER_ROW_A = BM / TN_VEC_A;
    static constexpr int TN_ROWS_PER_ITER_A   = THREADS / TN_THREADS_PER_ROW_A;
    static constexpr int TN_LOAD_ITERS_A       = BK / TN_ROWS_PER_ITER_A;

    static constexpr int TN_VEC_B = 8;
    static constexpr int TN_THREADS_PER_ROW_B = BN / TN_VEC_B;
    static constexpr int TN_ROWS_PER_ITER_B   = THREADS / TN_THREADS_PER_ROW_B;
    static constexpr int TN_LOAD_ITERS_B       = BK / TN_ROWS_PER_ITER_B;

    static constexpr int WARP_COUNT = THREADS / 32;
    static constexpr int WARPS_M = (BM >= BN * 2) ? (WARP_COUNT >= 4 ? 4 : 2) : (BM == BN ? (WARP_COUNT>=4?2:1) : 1);
    static constexpr int WARPS_N = WARP_COUNT / WARPS_M;
    static constexpr int WARP_TILE_M = BM / WARPS_M;
    static constexpr int WARP_TILE_N = BN / WARPS_N;
    static constexpr int MMA_M = WARP_TILE_M / 16;
    static constexpr int MMA_N = WARP_TILE_N / 8;
};

enum class BgemmLayout { NT, TN, NN };

template <typename Config, bool IsAligned, int SplitK, BgemmLayout Layout>
__global__ void __launch_bounds__(Config::THREADS, 1)
bgemm_backward_template_kernel(
    int M, int N, int K,
    __nv_bfloat16 alpha,
    const __nv_bfloat16* __restrict__ A, int lda, long long strideA,
    const __nv_bfloat16* __restrict__ B, int ldb, long long strideB,
    __nv_bfloat16 beta,
    __nv_bfloat16* __restrict__ C, int ldc, long long strideC,
    int batchCount)
{
    const int batch = blockIdx.z / SplitK;
    const int sk_idx = blockIdx.z % SplitK;
    if (batch >= batchCount) return;

    const int grid_x = (N + Config::BN - 1) / Config::BN, grid_y = (M + Config::BM - 1) / Config::BM;
    const int block_id = blockIdx.y * grid_x + blockIdx.x;
    const int sw = max(1, min(grid_y, 8));
    const int bx = (block_id / sw) % grid_x, by = (block_id % sw) + (block_id / (grid_x * sw)) * sw;
    if (by * Config::BM >= M || bx * Config::BN >= N) return;

    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int wy = wid / Config::WARPS_N, wx = wid % Config::WARPS_N;
    const int k_tiles = (K + Config::BK - 1) / Config::BK, tiles_per_sk = (k_tiles + SplitK - 1) / SplitK;
    const int k_start = sk_idx * tiles_per_sk * Config::BK, k_end = min(K, (sk_idx + 1) * tiles_per_sk * Config::BK);

    extern __shared__ __align__(16) __nv_bfloat16 b_smem[];
    float acc[Config::MMA_M][Config::MMA_N][4];
    #pragma unroll
    for (int i=0; i<Config::MMA_M; i++) for (int j=0; j<Config::MMA_N; j++) for (int d=0; d<4; d++) acc[i][j][d] = 0.f;

    const __nv_bfloat16* gA_ptr[Layout == BgemmLayout::TN ? Config::TN_LOAD_ITERS_A : Config::NT_LOAD_ITERS_A];
    const __nv_bfloat16* gB_ptr[Layout == BgemmLayout::NT ? Config::NT_LOAD_ITERS_B : Config::TN_LOAD_ITERS_B];

    if constexpr (Layout == BgemmLayout::TN) {
        #pragma unroll
        for (int i = 0; i < Config::TN_LOAD_ITERS_A; i++) {
            int r = (tid % Config::TN_THREADS_PER_ROW_A) * Config::TN_VEC_A, c = (tid / Config::TN_THREADS_PER_ROW_A) + i * Config::TN_ROWS_PER_ITER_A;
            gA_ptr[i] = A + (long long)batch * strideA + (long long)(k_start + c) * lda + (by * Config::BM + r);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
            int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A, c = (tid % Config::NT_THREADS_PER_ROW_A) * Config::NT_VEC_A;
            gA_ptr[i] = A + (long long)batch * strideA + (long long)(by * Config::BM + r) * lda + (k_start + c);
        }
    }

    if constexpr (Layout == BgemmLayout::NT) {
        #pragma unroll
        for (int i = 0; i < Config::NT_LOAD_ITERS_B; i++) {
            int lr = (tid % Config::NT_THREADS_PER_ROW_B) * Config::NT_VEC_B, lc = (tid / Config::NT_THREADS_PER_ROW_B) + i * Config::NT_ROWS_PER_ITER_B;
            gB_ptr[i] = B + (long long)batch * strideB + (long long)(bx * Config::BN + lc) * ldb + (k_start + lr);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
            int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B, c = (tid % Config::TN_THREADS_PER_ROW_B) * Config::TN_VEC_B;
            gB_ptr[i] = B + (long long)batch * strideB + (long long)(k_start + r) * ldb + (bx * Config::BN + c);
        }
    }

    auto load_to_stage = [&](int stage, int ko) {
        __nv_bfloat16* As = b_smem + stage * Config::STAGE_SIZE; __nv_bfloat16* Bs = As + Config::AS_SIZE;
        if constexpr (Layout == BgemmLayout::TN) {
            #pragma unroll
            for (int i = 0; i < Config::TN_LOAD_ITERS_A; i++) {
                int lc_s = (tid % Config::TN_THREADS_PER_ROW_A) * Config::TN_VEC_A, lr_s = (tid / Config::TN_THREADS_PER_ROW_A) + i * Config::TN_ROWS_PER_ITER_A;
                int sc = lc_s ^ ((lr_s & 7) << 3); uint32_t sm_a = __cvta_generic_to_shared(&As[lr_s * Config::BM + sc]);
                int gk = ko + lr_s, gm = by * Config::BM + lc_s; int bytes = (gk < K && gm < M) ? max(0, min(16, (M - gm) * 2)) : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_a), "l"((const void*)gA_ptr[i]), "r"(bytes));
                gA_ptr[i] += Config::BK * lda;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
                int r_sm = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A, c_sm = (tid % Config::NT_THREADS_PER_ROW_A) * Config::NT_VEC_A;
                int sc = c_sm ^ ((r_sm & 3) << 3); uint32_t sm_a = __cvta_generic_to_shared(&As[r_sm * Config::BK + sc]);
                int gr = by * Config::BM + r_sm, gc = ko + c_sm; int bytes = (gr < M && gc < K) ? max(0, min(16, (K - gc) * 2)) : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_a), "l"((const void*)gA_ptr[i]), "r"(bytes));
                gA_ptr[i] += Config::BK;
            }
        }
        if constexpr (Layout == BgemmLayout::NT) {
            #pragma unroll
            for (int i = 0; i < Config::NT_LOAD_ITERS_B; i++) {
                int lr = (tid % Config::NT_THREADS_PER_ROW_B) * Config::NT_VEC_B, lc = (tid / Config::NT_THREADS_PER_ROW_B) + i * Config::NT_ROWS_PER_ITER_B;
                int sc = lr ^ ((lc & 3) << 3); uint32_t sm_b = __cvta_generic_to_shared(&Bs[lc * Config::BK + sc]);
                int gr = bx * Config::BN + lc, gc = ko + lr; int bytes = (gr < N && gc < K) ? max(0, min(16, (K - gc) * 2)) : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_b), "l"((const void*)gB_ptr[i]), "r"(bytes));
                gB_ptr[i] += Config::BK;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
                int r_sm = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B, c_sm = (tid % Config::TN_THREADS_PER_ROW_B) * Config::TN_VEC_B;
                int sc = c_sm ^ ((r_sm & 7) << 3); uint32_t sm_b = __cvta_generic_to_shared(&Bs[r_sm * Config::BN + sc]);
                int gk = ko + r_sm, gn = bx * Config::BN + c_sm; int bytes = (gk < K && gn < N) ? max(0, min(16, (N - gn) * 2)) : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_b), "l"((const void*)gB_ptr[i]), "r"(bytes));
                gB_ptr[i] += Config::BK * ldb;
            }
        }
    };

    auto load_frA = [&](uint32_t reg[4], int ks, int mi, int st) {
        __nv_bfloat16* As = b_smem + st * Config::STAGE_SIZE;
        if constexpr (Layout == BgemmLayout::TN) {
            int k = ks + (lane % 8) + (lane / 16) * 8;
            int m = wy * Config::WARP_TILE_M + mi * 16 + (lane / 8 % 2) * 8;
            int sc = m ^ ((k & 7) << 3); uint32_t addr = __cvta_generic_to_shared(&As[k * Config::BM + sc]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3]) : "r"(addr));
        } else {
            int r = wy * Config::WARP_TILE_M + mi * 16 + (lane % 16);
            int c = ks + (lane / 16) * 8;
            int sc = c ^ ((r & 3) << 3); uint32_t addr = __cvta_generic_to_shared(&As[r * Config::BK + sc]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3]) : "r"(addr));
        }
    };

    auto load_frB = [&](uint32_t reg[2], int ks, int ni, int st) {
        __nv_bfloat16* Bs = b_smem + st * Config::STAGE_SIZE + Config::AS_SIZE;
        if constexpr (Layout == BgemmLayout::NT) {
            int r = wx * Config::WARP_TILE_N + ni * 8 + (lane % 8);
            int k = ks + (lane / 8 % 2) * 8;
            int sc = k ^ ((r & 3) << 3); uint32_t addr = __cvta_generic_to_shared(&Bs[r * Config::BK + sc]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(reg[0]), "=r"(reg[1]) : "r"(addr));
        } else {
            int k = ks + (lane % 8) + (lane / 8 % 2) * 8;
            int n = wx * Config::WARP_TILE_N + ni * 8;
            int sc = n ^ ((k & 7) << 3); uint32_t addr = __cvta_generic_to_shared(&Bs[k * Config::BN + sc]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(reg[0]), "=r"(reg[1]) : "r"(addr));
        }
    };

    if (k_start < k_end) {
        load_to_stage(0, k_start); asm volatile("cp.async.commit_group;\n");
        #pragma unroll
        for (int s = 1; s < Config::STAGES - 1; s++) { if (k_start + s * Config::BK < k_end) load_to_stage(s, k_start + s * Config::BK); asm volatile("cp.async.commit_group;\n"); }
    }
    int ws = Config::STAGES - 1, rs = 0; uint32_t frA[2][Config::MMA_M][4], frB[2][Config::MMA_N][2];
    asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2)); __syncthreads();
    #pragma unroll
    for (int i = 0; i < Config::MMA_M; i++) load_frA(frA[0][i], 0, i, rs);
    #pragma unroll
    for (int j = 0; j < Config::MMA_N; j++) load_frB(frB[0][j], 0, j, rs);


    for (int k = k_start; k < k_end; k += Config::BK) {
        if (k + (Config::STAGES - 1) * Config::BK < k_end) load_to_stage(ws, k + (Config::STAGES - 1) * Config::BK);
        asm volatile("cp.async.commit_group;\n");
        #pragma unroll
        for (int ks = 0; ks < Config::BK; ks += 16) {
            const int p = (ks >> 4) & 1, q = p ^ 1;
            #pragma unroll
            for (int i = 0; i < Config::MMA_M; i++) {
                if (ks + 16 < Config::BK) { load_frA(frA[q][i], ks + 16, i, rs); if (i == 0) for (int j = 0; j < Config::MMA_N; j++) load_frB(frB[q][j], ks + 16, j, rs); }
                else if (k + Config::BK < k_end) { if (i == 0) { asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2)); __syncthreads(); rs = (rs + 1) % Config::STAGES; ws = (ws + 1) % Config::STAGES; } load_frA(frA[q][i], 0, i, rs); if (i == 0) for (int j = 0; j < Config::MMA_N; j++) load_frB(frB[q][j], 0, j, rs); }
                #pragma unroll
                for (int j = 0; j < Config::MMA_N; j++) MMA_M16N8K16_F32_BF16(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3], frA[p][i][0], frA[p][i][1], frA[p][i][2], frA[p][i][3], frB[p][j][0], frB[p][j][1]);
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n"); __syncthreads();
    __nv_bfloat16* s_out = b_smem; const int rt = lane / 4, ct = (lane % 4) * 2; const float alpha_f = __bfloat162float(alpha);
    #pragma unroll
    for (int i = 0; i < Config::MMA_M; i++) {
        #pragma unroll
        for (int j = 0; j < Config::MMA_N; j++) {
            int r = wy * Config::WARP_TILE_M + i * 16 + rt, c = wx * Config::WARP_TILE_N + j * 8 + ct;
            // Swizzle row index with column group to avoid 8-way bank conflicts
            int swizzled_r = r ^ ((c / 8) & 7);
            *(__nv_bfloat162*)&s_out[swizzled_r * Config::BN + c] = {__float2bfloat16(acc[i][j][0] * alpha_f), __float2bfloat16(acc[i][j][1] * alpha_f)};
            *(__nv_bfloat162*)&s_out[(swizzled_r ^ 8) * Config::BN + c] = {__float2bfloat16(acc[i][j][2] * alpha_f), __float2bfloat16(acc[i][j][3] * alpha_f)};
        }
    }
    __syncthreads();
    const __nv_bfloat162 h2beta = {beta, beta};
    #pragma unroll
    for (int i = 0; i < (Config::BM * Config::BN) / (Config::THREADS * 8); i++) {
        int r_base = (tid / (Config::BN / 8)) + i * (Config::THREADS / (Config::BN / 8)), c_base = (tid % (Config::BN / 8)) * 8;
        int gr = by * Config::BM + r_base, gc = bx * Config::BN + c_base;
        if (gr < M && gc < N) {
            // Un-swizzle row index
            int swizzled_r = r_base ^ ((c_base / 8) & 7);
            int4 vals = *(int4*)&s_out[swizzled_r * Config::BN + c_base];
            if constexpr (SplitK > 1) {
                __nv_bfloat162* h2v = (__nv_bfloat162*)&vals;
                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    if (gc + l * 2 + 1 < N) {
                        // atomicAdd for __nv_bfloat162 is available on SM80+
                        atomicAdd((__nv_bfloat162*)((__nv_bfloat16*)C + (long long)batch * strideC + (long long)gr * ldc + gc + l * 2), h2v[l]);
                    } else {
                        atomicAdd((__nv_bfloat16*)C + (long long)batch * strideC + (long long)gr * ldc + gc + l * 2, h2v[l].x);
                    }
                }
            } else {
                if (beta != __float2bfloat16(0.0f)) {
                    int4 old = *(int4*)&C[(long long)batch * strideC + (long long)gr * ldc + gc];
                    __nv_bfloat162* h2v = (__nv_bfloat162*)&vals, * h2o = (__nv_bfloat162*)&old;
                    #pragma unroll
                    for (int l = 0; l < 4; l++) h2v[l] = __hadd2(h2v[l], __hmul2(h2o[l], h2beta));
                }
                *(int4*)&C[(long long)batch * strideC + (long long)gr * ldc + gc] = vals;
            }
        }
    }
}

template <typename Config>
__global__ void bgemm_scale_template_kernel(__nv_bfloat16* C, __nv_bfloat16 beta, int M, int N, int ldc, long long strideC, int batchCount, const __nv_bfloat16* bias = nullptr, int64_t bias_numel = 0) {
    int r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.z;
    if (r < M && c < N && b < batchCount) {
        __nv_bfloat16* dst = &C[b * strideC + (long long)r * ldc + c];
        const float betaf = __bfloat162float(beta);
        float val;
        if (bias) {
            float bv = 0.f;
            if (bias_numel == N) bv = __bfloat162float(bias[c]);
            else if (bias_numel == M) bv = __bfloat162float(bias[r]);
            else if (bias_numel == 1) bv = __bfloat162float(bias[0]);
            else if (bias_numel == (int64_t)M*N) bv = __bfloat162float(bias[(long long)r * N + c]);
            val = betaf * bv;
        } else {
            val = (beta == __float2bfloat16(0.f)) ? 0.f : (__bfloat162float(*dst) * betaf);
        }
        *dst = __float2bfloat16(val);
    }
}

// Explicit template instantiations (CFG via __VA_ARGS__ so its commas survive).
#define INSTANTIATE_BGEMM_CORE(SK, ...)                                              \
    template __global__ void bgemm_backward_template_kernel<__VA_ARGS__, true, SK, BgemmLayout::NN>(\
        int,int,int,__nv_bfloat16,const __nv_bfloat16*,int,long long,const __nv_bfloat16*,int,long long,__nv_bfloat16,__nv_bfloat16*,int,long long,int);                                                                 \
    template __global__ void bgemm_backward_template_kernel<__VA_ARGS__, true, SK, BgemmLayout::NT>(\
        int,int,int,__nv_bfloat16,const __nv_bfloat16*,int,long long,const __nv_bfloat16*,int,long long,__nv_bfloat16,__nv_bfloat16*,int,long long,int);                                                                 \
    template __global__ void bgemm_backward_template_kernel<__VA_ARGS__, true, SK, BgemmLayout::TN>(\
        int,int,int,__nv_bfloat16,const __nv_bfloat16*,int,long long,const __nv_bfloat16*,int,long long,__nv_bfloat16,__nv_bfloat16*,int,long long,int)

INSTANTIATE_BGEMM_CORE(1, BgemmTileConfig<128, 128, 32, 6, 128>);
INSTANTIATE_BGEMM_CORE(4, BgemmTileConfig<128, 128, 32, 6, 128>);
INSTANTIATE_BGEMM_CORE(1, BgemmTileConfig<256, 128, 32, 3, 256>);

template __global__ void bgemm_scale_template_kernel<BgemmTileConfig<128, 128, 32, 6, 128>>(
    __nv_bfloat16*, __nv_bfloat16, int, int, int, long long, int, const __nv_bfloat16*, int64_t);
