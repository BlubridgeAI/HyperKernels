// Batched SGEMM, TN layout, 256x64 tile. TF32 MMA on shared-memory staged tiles.

#include <cuda_runtime.h>
#include <stdint.h>

namespace tn256x64 {
    static constexpr int BM = 256, BN = 64, BK = 16, STAGES = 4, THREADS = 256;
    static constexpr int AS_SIZE = BM * BK, BS_SIZE = BN * BK, STAGE_SIZE = AS_SIZE + BS_SIZE;
}

// m16n8k8 TF32 tensor-core MMA, row.col, fp32 accumulate.
static __device__ __forceinline__ void mma_tf32(float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(256, 1)
sgemm_tn_256x64_sm89_2_k(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount) {
    using namespace tn256x64;
    const int batch = blockIdx.z; if (batch >= batchCount) return;
    const int bx = blockIdx.x, by = blockIdx.y;
    if (by * BM >= M || bx * BN >= N) return;
    const int tid = threadIdx.x, lane = tid % 32, wid = tid / 32;
    const int wy = wid % 4, wx = wid / 4;
    extern __shared__ float smem[];
    const float *A_ptr = A + (long long)batch * strideA, *B_ptr = B + (long long)batch * strideB;
    float* C_ptr = C + (long long)batch * strideC;

    const int ar_k = tid / 64, ar_m = (tid % 64) * 4;
    const int br_k = tid / 16, br_n = (tid % 16) * 4;

    // Stage a BMxBK A tile and BNxBK B tile into shared memory via cp.async.
    auto issue = [&](int stage, int ko) {
        float *As = smem + stage * STAGE_SIZE, *Bs = As + AS_SIZE;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = ar_k + i * 4, m = ar_m;
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(&As[r * BM + (m ^ ((r & 3) << 2))]);
            int src_size = (ko + r < K && by * BM + m < M) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(sm), "l"(A_ptr + (long long)(ko + r) * lda + by * BM + m), "r"(src_size));
        }
        #pragma unroll
        for (int i = 0; i < 1; i++) {
            int r = br_k + i * 16, n = br_n;
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(&Bs[r * BN + (n ^ ((r & 3) << 2))]);
            int src_size = (ko + r < K && bx * BN + n < N) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(sm), "l"(B_ptr + (long long)(ko + r) * ldb + bx * BN + n), "r"(src_size));
        }
    };

    float acc[4][4][4] = {0};
    const int k_tiles = (K+BK-1)/BK;
    for (int s=0; s<min(k_tiles, STAGES-1); s++) { issue(s, s*BK); asm volatile("cp.async.commit_group;"); }
    asm volatile("cp.async.wait_group %0;" :: "n"(STAGES-2)); __syncthreads();

    uint32_t frA[2][4][4], frB[2][4][2];
    int rs=0, ws=STAGES-1;

    // Load A fragment for one MMA tile from shared memory (+0x1000 TF32 rounding bias).
    auto loadA = [&](uint32_t r[4], int ks, int mi, const float* As) {
        const int m = wy * 64 + mi * 16 + (lane >> 2);
        const int k = ks + (lane % 4);
        auto la = [&](int _m, int _k) { return *(const uint32_t*)(&As[_k * BM + (_m ^ ((_k & 3) << 2))]); };
        r[0] = la(m, k); r[1] = la(m + 8, k); r[2] = la(m, k + 4); r[3] = la(m + 8, k + 4);
        r[0] += 0x1000u; r[1] += 0x1000u; r[2] += 0x1000u; r[3] += 0x1000u;
    };

    // Load B fragment for one MMA tile from shared memory (+0x1000 TF32 rounding bias).
    auto loadB = [&](uint32_t r[2], int ks, int ni, const float* Bs) {
        const int n = wx * 32 + ni * 8 + (lane >> 2);
        const int k = ks + (lane % 4);
        auto lb = [&](int _n, int _k) { return *(const uint32_t*)(&Bs[_k * BN + (_n ^ ((_k & 3) << 2))]); };
        r[0] = lb(n, k); r[1] = lb(n, k + 4);
        r[0] += 0x1000u; r[1] += 0x1000u;
    };

    for (int mi=0; mi<4; mi++) loadA(frA[0][mi], 0, mi, smem+rs*STAGE_SIZE);
    for (int ni=0; ni<4; ni++) loadB(frB[0][ni], 0, ni, smem+rs*STAGE_SIZE+AS_SIZE);

    for (int kt=0; kt<k_tiles; kt++) {
        int nk = (kt + STAGES - 1) * BK; if (nk < K) issue(ws % STAGES, nk); asm volatile("cp.async.commit_group;");
        #pragma unroll
        for (int mi=0; mi<4; mi++) {
            #pragma unroll
            for (int ni=0; ni<4; ni++) {
                mma_tf32(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                         frA[0][mi][0], frA[0][mi][1], frA[0][mi][2], frA[0][mi][3],
                         frB[0][ni][0], frB[0][ni][1]);
                if (mi == 0) loadB(frB[1][ni], 8, ni, smem + rs*STAGE_SIZE + AS_SIZE);
            }
            loadA(frA[1][mi], 8, mi, smem + rs*STAGE_SIZE);
        }
        asm volatile("cp.async.wait_group %0;" :: "n"(STAGES-2)); __syncthreads();
        rs = (rs+1)%STAGES; ws = (ws+1)%STAGES;
        #pragma unroll
        for (int mi=0; mi<4; mi++) {
            #pragma unroll
            for (int ni=0; ni<4; ni++) {
                mma_tf32(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                         frA[1][mi][0], frA[1][mi][1], frA[1][mi][2], frA[1][mi][3],
                         frB[1][ni][0], frB[1][ni][1]);
                if (mi == 0 && kt + 1 < k_tiles) loadB(frB[0][ni], 0, ni, smem + rs*STAGE_SIZE + AS_SIZE);
            }
            if (kt + 1 < k_tiles) loadA(frA[0][mi], 0, mi, smem + rs*STAGE_SIZE);
        }
    }

    const int ge = lane >> 2, te = lane & 3;
    #pragma unroll
    for (int mi=0; mi<4; mi++) {
        const int r0 = by*BM+wy*64+mi*16+ge, r8 = r0+8;
        if (r0 >= M) continue;
        #pragma unroll
        for (int ni=0; ni<4; ni++) {
            const int c0 = bx*BN+wx*32+ni*8+te*2;
            if (c0 >= N) continue;
            float* d = C_ptr+(long long)r0*ldc+c0;
            d[0] = alpha*acc[mi][ni][0]+(beta==0.f?0.f:beta*d[0]);
            if (c0+1<N) d[1] = alpha*acc[mi][ni][1]+(beta==0.f?0.f:beta*d[1]);
            if (r8 < M) {
                float* d8 = C_ptr+(long long)r8*ldc+c0;
                d8[0] = alpha*acc[mi][ni][2]+(beta==0.f?0.f:beta*d8[0]);
                if (c0+1<N) d8[1] = alpha*acc[mi][ni][3]+(beta==0.f?0.f:beta*d8[1]);
            }
        }
    }
}
