// Batched single-precision GEMM, NN layout: C = alpha*A*B + beta*C per batch.
// 256x128x16 tile, 4-stage cp.async pipeline, TF32 tensor-core MMA, 256 threads.

#include <cuda_runtime.h>
#include <stdint.h>

// 3xTF32-style MMA: m16n8k8 row/col, f32 accumulate.
#ifndef MMA_TF32
#define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                \
    asm volatile(                                                   \
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "     \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"       \
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                     \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
#endif

// Async copy global->smem, 16-byte, .cg bypasses L1. src_size handles zero-fill.
__device__ __forceinline__
void cp_async_bypass(uint32_t smem_addr, const void* gmem_ptr, int src_size) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_size));
}

__device__ __forceinline__ void bar_sync_defer(int /*bar_id*/, int /*threads*/) {
    __syncthreads();
}

// Constants for the 256x128x16 tile.
static constexpr int BM = 256;
static constexpr int BN = 128;
static constexpr int BK = 16;
static constexpr int STAGES = 4;
static constexpr int THREADS = 256;
static constexpr int WARPS_M = 4;
static constexpr int WARPS_N = 2;
static constexpr int WARP_TILE_M = BM / WARPS_M;  // 64
static constexpr int WARP_TILE_N = BN / WARPS_N;  // 64
static constexpr int MMA_M = WARP_TILE_M / 16;    // 4
static constexpr int MMA_N = WARP_TILE_N / 8;     // 8

static constexpr int AS_SIZE   = BM * BK;      // 4096
static constexpr int BS_SIZE   = BK * BN;      // 2048
static constexpr int STAGE_SIZE = AS_SIZE + BS_SIZE; // 6144

static constexpr int LOAD_VEC   = 4;
static constexpr int TPR_A      = BK / LOAD_VEC;   // threads per row for A = 4
static constexpr int RPL_A      = THREADS / TPR_A;  // rows per load iteration = 64
static constexpr int ITERS_A    = BM / RPL_A;       // load iterations for A = 4

static constexpr int TPR_B      = BN / LOAD_VEC;   // threads per row for B = 32
static constexpr int RPL_B      = THREADS / TPR_B;  // rows per load iteration = 8
static constexpr int ITERS_B    = BK / RPL_B;       // load iterations for B = 2

__global__ void __launch_bounds__(THREADS, 1)
sgemm_nn_256x128_bypass_sm89_batched_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda, long long strideA,
    const float* __restrict__ B, int ldb, long long strideB,
    float beta,
    float* __restrict__ C, int ldc, long long strideC,
    int batchCount)
{
    const int batch = blockIdx.z;
    if (batch >= batchCount) return;

    // Robust block mapping for non-square grids
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    if (bx >= (N + BN - 1) / BN || by >= (M + BM - 1) / BM) return;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int wy   = wid / WARPS_N;
    const int wx   = wid % WARPS_N;

    extern __shared__ float smem[];

    const int a_row = tid / TPR_A;
    const int a_col = (tid % TPR_A) * LOAD_VEC;
    const int b_row = tid / TPR_B;
    const int b_col = (tid % TPR_B) * LOAD_VEC;

    const float* A_batch = A + (long long)batch * strideA;
    const float* B_batch = B + (long long)batch * strideB;
    float*       C_batch = C + (long long)batch * strideC;

    const float* gA[ITERS_A];
    const float* gB[ITERS_B];
    #pragma unroll
    for (int i = 0; i < ITERS_A; i++) {
        const int r = a_row + i * RPL_A;
        gA[i] = A_batch + (long long)(by * BM + r) * lda + a_col;
    }
    #pragma unroll
    for (int i = 0; i < ITERS_B; i++) {
        const int r = b_row + i * RPL_B;
        gB[i] = B_batch + (long long)r * ldb + (bx * BN + b_col);
    }

    auto smem_addr_f = [](const float* ptr) -> uint32_t {
        return __cvta_generic_to_shared(ptr);
    };

    uint32_t sm_a_off[ITERS_A], sm_b_off[ITERS_B];
    #pragma unroll
    for (int i = 0; i < ITERS_A; i++) {
        const int r  = a_row + i * RPL_A;
        const int sc = a_col ^ ((r & 3) << 2); // BK=16, must stay in [0,15]
        sm_a_off[i] = r * BK + sc;
    }
    #pragma unroll
    for (int i = 0; i < ITERS_B; i++) {
        const int r  = b_row + i * RPL_B;
        const int sc = b_col ^ ((r & 7) << 2); // BN=128, can use 8-row depth
        sm_b_off[i] = r * BN + sc;
    }

    // Fragments

    auto load_frA = [&](uint32_t reg[4], int ks, int mi, const float* As_st) {
        const int row_base = wy * WARP_TILE_M + mi * 16;
        const int r0 = (lane / 4);
        const int r8 = r0 + 8;
        const int c0 = ks + (lane % 4);
        const int c4 = c0 + 4;

        auto get_ptr = [&](int r, int c) {
            const int mask = (r & 3) << 2;
            return (const uint32_t*)&As_st[r * BK + (c ^ mask)];
        };

        // m16n8k8 tf32 A fragment per thread: A[r0,c0],A[r0,c4],A[r8,c0],A[r8,c4].
        reg[0] = *get_ptr(row_base + r0, c0);
        reg[1] = *get_ptr(row_base + r0, c4);
        reg[2] = *get_ptr(row_base + r8, c0);
        reg[3] = *get_ptr(row_base + r8, c4);
        reg[0] += 0x1000u; reg[1] += 0x1000u; reg[2] += 0x1000u; reg[3] += 0x1000u;
    };

    auto load_frB = [&](uint32_t reg[2], int ks, int ni, const float* Bs_st) {
        const int n  = wx * WARP_TILE_N + ni * 8 + (lane / 4);
        const int k0 = ks + (lane % 4);
        const int k4 = k0 + 4;

        auto get_val = [&](int r, int c) {
            const int mask = (r & 7) << 2;
            return *(const uint32_t*)(&Bs_st[r * BN + (c ^ mask)]);
        };
        reg[0] = get_val(k0, n);
        reg[1] = get_val(k4, n);
        reg[0] += 0x1000u; reg[1] += 0x1000u;
    };

    auto issue_stage = [&](int stage, int ko) {
        float* As = smem + stage * STAGE_SIZE;
        float* Bs = As + AS_SIZE;
        const bool k_aligned = (K % BK == 0);

        #pragma unroll
        for (int i = 0; i < ITERS_A; i++) {
            const int rA  = a_row + i * RPL_A;
            const int gcA = ko + a_col;
            uint32_t smA = smem_addr_f(As + sm_a_off[i]);
            const float* gA_ptr = gA[i];

            // ptr must be 16-byte aligned for cp.async 16
            bool can_fast_A = (by * BM + rA < M) && (((size_t)gA_ptr & 15) == 0);

            if (can_fast_A && (gcA + 4 <= K)) {
                cp_async_bypass(smA, gA_ptr, 16);
            } else if (by * BM + rA < M) {
                int bytes_A = max(0, min(16, (K - gcA) * 4));
                cp_async_bypass(smA, gA_ptr, bytes_A);
            } else {
                *(float4*)(As + sm_a_off[i]) = {0,0,0,0};
            }
            gA[i] += BK;

            if (i % 2 == 1) {
                const int b_idx = i / 2;
                const int rB  = b_row + b_idx * RPL_B;
                const int gnB = bx * BN + b_col;
                uint32_t smB = smem_addr_f(Bs + sm_b_off[b_idx]);
                const float* gB_ptr = gB[b_idx];

                bool can_fast_B = (ko + rB < K) && (((size_t)gB_ptr & 15) == 0);

                if (can_fast_B && (gnB + 4 <= N)) {
                    cp_async_bypass(smB, gB_ptr, 16);
                } else if (ko + rB < K) {
                    int bytes_B = max(0, min(16, (N - gnB) * 4));
                    cp_async_bypass(smB, gB_ptr, bytes_B);
                } else {
                    *(float4*)(Bs + sm_b_off[b_idx]) = {0,0,0,0};
                }
                gB[b_idx] += BK * ldb;
            }
        }
    };

    float acc[MMA_M][MMA_N][4] = {0};
    #pragma unroll
    for (int i = 0; i < MMA_M; i++)
        #pragma unroll
        for (int j = 0; j < MMA_N; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int k_tiles = (K + BK - 1) / BK;

    for (int s = 0; s < min(k_tiles, STAGES - 1); s++) {
        issue_stage(s, s * BK);
        asm volatile("cp.async.commit_group;\n");
    }
    asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2));
    bar_sync_defer(0, THREADS);

    int rs = 0, ws = STAGES - 1;
    uint32_t frA[2][MMA_M][4], frB[2][MMA_N][2];

    // Initial preload for Stage 0 (ks=0)
    const float* As_rs = smem + rs * STAGE_SIZE;
    const float* Bs_rs = As_rs + AS_SIZE;
    #pragma unroll
    for (int mi = 0; mi < MMA_M; mi++) load_frA(frA[0][mi], 0, mi, As_rs);
    #pragma unroll
    for (int ni = 0; ni < MMA_N; ni++) load_frB(frB[0][ni], 0, ni, Bs_rs);

    // Main K-tile loop: interleaved pipelining
    for (int k_tile = 0; k_tile < k_tiles; k_tile++) {
        const int k_fetch = (k_tile + STAGES - 1) * BK;
        float* As_ws = smem + (ws % STAGES) * STAGE_SIZE;
        float* Bs_ws = As_ws + AS_SIZE;

        // Compute ks=0, load ks=8, interleave fetches
        #pragma unroll
        for (int mi = 0; mi < MMA_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < MMA_N; ni++) {
                MMA_TF32(acc[mi][ni][0], acc[mi][ni][1],
                         acc[mi][ni][2], acc[mi][ni][3],
                         frA[0][mi][0], frA[0][mi][1],
                         frA[0][mi][2], frA[0][mi][3],
                         frB[0][ni][0], frB[0][ni][1]);
                if (ni % 4 == 0) {
                    const int b_idx = mi * 2 + ni / 4;
                    load_frB(frB[1][b_idx], 8, b_idx, Bs_rs);
                }
            }
            load_frA(frA[1][mi], 8, mi, As_rs);

            if (k_fetch < K) {
                const int rA  = a_row + mi * RPL_A;
                const int gcA = k_fetch + a_col;
                uint32_t smA = smem_addr_f(As_ws + sm_a_off[mi]);
                const float* ptrA = gA[mi];
                if ((by * BM + rA < M) && (((size_t)ptrA & 15) == 0) && (gcA + 4 <= K)) cp_async_bypass(smA, ptrA, 16);
                else if (by * BM + rA < M) cp_async_bypass(smA, ptrA, max(0, min(16, (K - gcA) * 4)));
                else *(float4*)(As_ws + sm_a_off[mi]) = {0,0,0,0};
                gA[mi] += BK;

                if (mi == 1 || mi == 3) {
                    const int b_idx = mi / 2;
                    const int rB  = b_row + b_idx * RPL_B;
                    const int gnB = bx * BN + b_col;
                    uint32_t smB = smem_addr_f(Bs_ws + sm_b_off[b_idx]);
                    const float* ptrB = gB[b_idx];
                    if ((k_fetch + rB < K) && (((size_t)ptrB & 15) == 0) && (gnB + 4 <= N)) cp_async_bypass(smB, ptrB, 16);
                    else if (k_fetch + rB < K) cp_async_bypass(smB, ptrB, max(0, min(16, (N - gnB) * 4)));
                    else *(float4*)(Bs_ws + sm_b_off[b_idx]) = {0,0,0,0};
                    gB[b_idx] += BK * ldb;
                }
            }
        }
        asm volatile("cp.async.commit_group;\n");

        asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2));
        bar_sync_defer(0, THREADS);

        rs = (rs + 1) % STAGES;
        ws = (ws + 1) % STAGES;
        As_rs = smem + rs * STAGE_SIZE;
        Bs_rs = As_rs + AS_SIZE;

        // Compute ks=8, load ks=0
        #pragma unroll
        for (int mi = 0; mi < MMA_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < MMA_N; ni++) {
                MMA_TF32(acc[mi][ni][0], acc[mi][ni][1],
                         acc[mi][ni][2], acc[mi][ni][3],
                         frA[1][mi][0], frA[1][mi][1],
                         frA[1][mi][2], frA[1][mi][3],
                         frB[1][ni][0], frB[1][ni][1]);
                if (ni % 4 == 0) {
                    const int b_idx = mi * 2 + ni / 4;
                    load_frB(frB[0][b_idx], 0, b_idx, Bs_rs);
                }
            }
            load_frA(frA[0][mi], 0, mi, As_rs);
        }
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");
    asm volatile("bar.warp.sync 0xffffffff;\n");

    const int g_epi = lane >> 2;
    const int t_epi = lane & 3;

    #pragma unroll
    for (int mi = 0; mi < MMA_M; mi++) {
        const int r0 = by * BM + wy * WARP_TILE_M + mi * 16 + g_epi;
        const int r8 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < MMA_N; ni++) {
            const int c0 = bx * BN + wx * WARP_TILE_N + ni * 8 + t_epi * 2;
            if (c0 >= N) continue;
            const bool col_ok = (c0 + 1 < N);
            if (r0 < M) {
                float* dst = C_batch + (long long)r0 * ldc + c0;
                float f0 = __fmaf_rn(alpha, acc[mi][ni][0], 0.f);
                float f1 = __fmaf_rn(alpha, acc[mi][ni][1], 0.f);
                if (beta != 0.f) {
                    f0 = __fmaf_rn(beta, dst[0], f0);
                    if(col_ok) f1 = __fmaf_rn(beta, dst[1], f1);
                }
                dst[0] = f0; if (col_ok) dst[1] = f1;
            }
            if (r8 < M) {
                float* dst = C_batch + (long long)r8 * ldc + c0;
                float f0 = __fmaf_rn(alpha, acc[mi][ni][2], 0.f);
                float f1 = __fmaf_rn(alpha, acc[mi][ni][3], 0.f);
                if (beta != 0.f) {
                    f0 = __fmaf_rn(beta, dst[0], f0);
                    if(col_ok) f1 = __fmaf_rn(beta, dst[1], f1);
                }
                dst[0] = f0; if (col_ok) dst[1] = f1;
            }
        }
    }
}
