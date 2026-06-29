// Batched NT-layout SGEMM, 256x128 tile, TF32 tensor-core MMA with multi-stage cp.async pipeline.

#include <cuda_runtime.h>
#include <stdint.h>

// ldmatrix x4: load four 8x8 b16 fragments from shared memory into registers.
#ifndef LDSM_X4
#define LDSM_X4(r0,r1,r2,r3,addr)                                            \
    asm volatile(                                                              \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"      \
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr))
#endif

// Tile, stage, and warp-mapping constants for the 256x128 NT configuration.
namespace nt256x128 {
    static constexpr int BM = 256, BN = 128, BK = 16, STAGES = 4, THREADS = 256;
    static constexpr int AS_SIZE = BM * BK, BS_SIZE = BN * BK, STAGE_SIZE = AS_SIZE + BS_SIZE;
    static constexpr int MMA_M = 4, MMA_N = 8;
    static constexpr int WARPS_N = 2; // 256 threads / 32 = 8 warps. 8/4 = 2.
    static constexpr int WARP_TILE_M = 64, WARP_TILE_N = 64;
    static constexpr int TPR_A = BK / 4, RPL_A = THREADS / TPR_A;
    static constexpr int TPR_B = BN / 4, RPL_B = THREADS / TPR_B;
}

// m16n8k8 TF32 tensor-core matrix multiply-accumulate.
static __device__ __forceinline__ void mma_tf32(float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void bar_sync_defer(int /*bar_id*/, int /*threads*/) {
    __syncthreads();
}

template <bool IsSplitK>
__global__ void __launch_bounds__(nt256x128::THREADS, 1)
sgemm_nt_256x128_sm89_2_k(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda, long long strideA,
    const float* __restrict__ B, int ldb, long long strideB,
    float beta,
    float* __restrict__ C, int ldc, long long strideC,
    int batchCount, int splitK)
{
    using namespace nt256x128;
    // SplitK and Batch indexing
    const int batch = blockIdx.z / splitK;
    const int sk_idx = blockIdx.z % splitK;
    if (batch >= batchCount) return;

    // Robust CTA swizzling (8-wide strips) for L2 locality
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
    const int wy   = wid / WARPS_N, wx = wid % WARPS_N;

    extern __shared__ float smem[];

    // K-chunking for SplitK
    const int k_chunk = ((K + BK - 1) / BK + splitK - 1) / splitK * BK;
    const int k_start = sk_idx * k_chunk;
    const int k_end   = min(K, k_start + k_chunk);
    const int ktcnt   = (k_end - k_start + BK - 1) / BK;

    const float* A_base = A + (long long)batch * strideA + (long long)k_start;
    const float* B_base = B + (long long)batch * strideB + (long long)k_start;
    float*       C_ptr  = C + (long long)batch * strideC;

    const int a_row = tid / (BK / 4), a_col = (tid % (BK / 4)) * 4;
    const int b_n   = tid / (BK / 4), b_k   = (tid % (BK / 4)) * 4;

    uint32_t sm_a_off[4], sm_b_off[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int r = a_row + i * (THREADS / (BK / 4));
        sm_a_off[i] = r * BK + (a_col ^ ((r & 3) << 2));
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) { // B is [N, K]: 128 rows x 16 cols, 256 threads x float4, 2 passes per stage
        int br = (tid / (BK / 4)) + i * (THREADS / (BK / 4));
        int bc = (tid % (BK / 4)) * 4;
        sm_b_off[i] = br * BK + (bc ^ ((br & 3) << 2));
    }

    auto load_to_smem = [&](int stage, int ko) {
        float* As = smem + stage * STAGE_SIZE;
        float* Bs = As + AS_SIZE;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = a_row + i * (THREADS / (BK / 4)), c = ko + a_col;
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(As + sm_a_off[i]);
            int src_size = (by * BM + r < M && k_start + c < k_end) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(sm), "l"(A_base + (long long)(by * BM + r) * lda + c), "r"(src_size));
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int r = (tid / (BK / 4)) + i * (THREADS / (BK / 4)), c = ko + (tid % (BK / 4)) * 4;
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(Bs + sm_b_off[i]);
            int src_size = (bx * BN + r < N && k_start + c < k_end) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(sm), "l"(B_base + (long long)(bx * BN + r) * ldb + c), "r"(src_size));
        }
    };

    float acc[MMA_M][MMA_N][4] = {0};
    #pragma unroll
    for (int s = 0; s < min(ktcnt, STAGES - 1); s++) { load_to_smem(s, s * BK); asm volatile("cp.async.commit_group;"); }
    asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2));
    bar_sync_defer(0, THREADS);

    uint32_t frA[2][MMA_M][4], frB[2][MMA_N][2];
    int rs = 0, ws = STAGES - 1;

    auto fetch_a = [&](uint32_t r[4], int ks, int mi, const float* As) {
        const int m_idx = wy * WARP_TILE_M + mi * 16 + (lane % 16);
        const int k0 = ks + (lane / 16) * 4;
        LDSM_X4(r[0], r[1], r[2], r[3], (uint32_t)__cvta_generic_to_shared(&As[m_idx * BK + (k0 ^ ((m_idx & 3) << 2))]));
        r[0] += 0x1000u; r[1] += 0x1000u; r[2] += 0x1000u; r[3] += 0x1000u;
    };
    auto fetch_b = [&](uint32_t r[2], int ks, int ni, const float* Bs) {
        const int n_idx = wx * WARP_TILE_N + ni * 8 + (lane >> 2), k0 = ks + (lane & 3), k4 = k0 + 4;
        const int mask = (n_idx & 3) << 2;
        r[0] = *(const uint32_t*)(&Bs[n_idx * BK + (k0 ^ mask)]);
        r[1] = *(const uint32_t*)(&Bs[n_idx * BK + (k4 ^ mask)]);
        r[0] += 0x1000u; r[1] += 0x1000u;
    };

    #pragma unroll
    for (int mi = 0; mi < MMA_M; mi++) fetch_a(frA[0][mi], 0, mi, smem + rs * STAGE_SIZE);
    #pragma unroll
    for (int ni = 0; ni < MMA_N; ni++) fetch_b(frB[0][ni], 0, ni, smem + rs * STAGE_SIZE + AS_SIZE);

    for (int kt = 0; kt < ktcnt; kt++) {
        const int kf = (kt + STAGES - 1) * BK;
        if (kf < (k_end - k_start)) load_to_smem(ws % STAGES, kf);
        asm volatile("cp.async.commit_group;");

        #pragma unroll
        for (int mi = 0; mi < MMA_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < MMA_N; ni++) {
                mma_tf32(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3], frA[0][mi][0], frA[0][mi][1], frA[0][mi][2], frA[0][mi][3], frB[0][ni][0], frB[0][ni][1]);
                if (ni % 4 == 0) { int b_idx = mi * 2 + ni / 4; fetch_b(frB[1][b_idx], 8, b_idx, smem + rs * STAGE_SIZE + AS_SIZE); }
            }
            fetch_a(frA[1][mi], 8, mi, smem + rs * STAGE_SIZE);
        }

        asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2));
        bar_sync_defer(0, THREADS);
        rs = (rs + 1) % STAGES; ws = (ws + 1) % STAGES;

        #pragma unroll
        for (int mi = 0; mi < MMA_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < MMA_N; ni++) {
                mma_tf32(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3], frA[1][mi][0], frA[1][mi][1], frA[1][mi][2], frA[1][mi][3], frB[1][ni][0], frB[1][ni][1]);
                if (ni % 4 == 0) { int b_idx = mi * 2 + ni / 4; fetch_b(frB[0][b_idx], 0, b_idx, smem + rs * STAGE_SIZE + AS_SIZE); }
            }
            if (kt + 1 < ktcnt) fetch_a(frA[0][mi], 0, mi, smem + rs * STAGE_SIZE);
        }
    }

    asm volatile("cp.async.wait_all;");
    const int ge = lane >> 2, te = lane & 3;
    #pragma unroll
    for (int mi = 0; mi < MMA_M; mi++) {
        const int r0 = by * BM + wy * WARP_TILE_M + mi * 16 + ge, r8 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < MMA_N; ni++) {
            const int c0 = bx * BN + wx * WARP_TILE_N + ni * 8 + te * 2;
            if (c0 >= N) continue;
            auto st = [&](int r, float v0, float v1) {
                if (r >= M) return;
                float* p = C_ptr + (long long)r * ldc + c0;
                float f0 = alpha * v0, f1 = alpha * v1;
                if constexpr (IsSplitK) {
                    atomicAdd(p, f0); if (c0 + 1 < N) atomicAdd(p + 1, f1);
                } else {
                    if (beta != 0.f) { f0 += beta * p[0]; if (c0 + 1 < N) f1 += beta * p[1]; }
                    p[0] = f0; if (c0 + 1 < N) p[1] = f1;
                }
            };
            st(r0, acc[mi][ni][0], acc[mi][ni][1]); st(r8, acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

// Non-splitK and splitK paths used by the launcher.
template __global__ void sgemm_nt_256x128_sm89_2_k<false>(int, int, int, float, const float*, int, long long, const float*, int, long long, float, float*, int, long long, int, int);
template __global__ void sgemm_nt_256x128_sm89_2_k<true>(int, int, int, float, const float*, int, long long, const float*, int, long long, float, float*, int, long long, int, int);
