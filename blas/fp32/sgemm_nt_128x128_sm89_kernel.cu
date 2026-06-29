// NT-layout batched SGEMM, 128x128 tile, TF32 tensor-core path.

#include <cuda_runtime.h>
#include <stdint.h>

#ifndef LDSM_X4
#define LDSM_X4(r0,r1,r2,r3,addr)                                            \
    asm volatile(                                                              \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"      \
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr))
#endif

namespace nt128x128 {
    static constexpr int BM = 128, BN = 128, BK = 16, STAGES = 4, THREADS = 256;
    static constexpr int AS_SIZE = BM * BK, BS_SIZE = BN * BK, STAGE_SIZE = AS_SIZE + BS_SIZE;
}

// One m16n8k8 TF32 tensor-core MMA, accumulating into d0..d3.
static __device__ __forceinline__ void mma_tf32(float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(256, 1)
sgemm_nt_128x128_sm89_2_k(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount) {
    using namespace nt128x128;
    const int batch = blockIdx.z; if (batch >= batchCount) return;
    const int bx = blockIdx.x, by = blockIdx.y;
    if (by * BM >= M || bx * BN >= N) return;
    const int tid = threadIdx.x, lane = tid % 32, wid = tid / 32;
    const int wy = wid / 2, wx = wid % 2;
    extern __shared__ float smem[];
    const float *A_ptr = A + (long long)batch * strideA, *B_ptr = B + (long long)batch * strideB;
    float* C_ptr = C + (long long)batch * strideC;

    const int a_row = tid / 4, a_col = (tid % 4) * 4;
    const int b_n = tid / 4, b_k = (tid % 4) * 4;

    auto issue = [&](int stage, int ko) {
        float *As = smem + stage * STAGE_SIZE, *Bs = As + AS_SIZE;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int r = a_row + i * 64, c = ko + a_col;
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(&As[r * BK + (a_col ^ ((r & 3) << 2))]);
            int src_size = (by * BM + r < M && c < K) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(sm), "l"(A_ptr + (long long)(by * BM + r) * lda + c), "r"(src_size));
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int n_s = b_n + i * 64, c = ko + b_k;
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(&Bs[n_s * BK + (b_k ^ ((n_s & 3) << 2))]);
            int src_size = (bx * BN + n_s < N && c < K) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(sm), "l"(B_ptr + (long long)(bx * BN + n_s) * ldb + c), "r"(src_size));
        }
    };

    float acc[2][8][4] = {0};
    const int k_tiles = (K+BK-1)/BK;
    for (int s=0; s<min(k_tiles, STAGES-1); s++) { issue(s, s*BK); asm volatile("cp.async.commit_group;"); }
    asm volatile("cp.async.wait_group %0;" :: "n"(STAGES-2)); __syncthreads();

    uint32_t frA[2][2][4], frB[2][8][2];
    int rs=0, ws=STAGES-1;

    auto loadA = [&](uint32_t r[4], int ks, int mi, const float* As) {
        const int m_idx = wy * 32 + mi * 16 + (lane % 16);
        const int k0 = ks + (lane / 16) * 4;
        const int mask = (m_idx & 3) << 2;
        LDSM_X4(r[0], r[1], r[2], r[3], (uint32_t)__cvta_generic_to_shared(&As[m_idx * BK + (k0 ^ mask)]));
        r[0] += 0x1000u; r[1] += 0x1000u; r[2] += 0x1000u; r[3] += 0x1000u;
    };

    auto loadB = [&](uint32_t r[2], int ks, int ni, const float* Bs) {
        const int n_idx = wx * 64 + ni * 8 + (lane >> 2);
        const int k0 = ks + (lane & 3);
        const int k4 = k0 + 4;
        const int mask = (n_idx & 3) << 2;
        r[0] = *(const uint32_t*)(&Bs[n_idx * BK + (k0 ^ mask)]);
        r[1] = *(const uint32_t*)(&Bs[n_idx * BK + (k4 ^ mask)]);
        r[0] += 0x1000u; r[1] += 0x1000u;
    };

    for (int kt=0; kt<k_tiles; kt++) {
        #pragma unroll
        for (int k_step = 0; k_step < 2; k_step++) {
            const int ks = k_step * 8;
            for (int mi=0; mi<2; mi++) loadA(frA[0][mi], ks, mi, smem+rs*STAGE_SIZE);
            for (int ni=0; ni<8; ni++) loadB(frB[0][ni], ks, ni, smem+rs*STAGE_SIZE+AS_SIZE);

            #pragma unroll
            for (int mi=0; mi<2; mi++) {
                for (int ni=0; ni<8; ni++) {
                    mma_tf32(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                             frA[0][mi][0], frA[0][mi][1], frA[0][mi][2], frA[0][mi][3],
                             frB[0][ni][0], frB[0][ni][1]);
                }
            }
        }
        int nk = (kt + STAGES - 1) * BK;
        issue(ws % STAGES, nk);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group %0;" :: "n"(STAGES-2));
        __syncthreads();
        rs = (rs+1)%STAGES; ws = (ws+1)%STAGES;
    }

    const int g_epi = lane >> 2, t_epi = lane & 3;
    #pragma unroll
    for (int mi=0; mi<2; mi++) {
        const int r0 = by*BM+wy*32+mi*16+g_epi;
        if (r0 >= M) continue;
        #pragma unroll
        for (int ni=0; ni<8; ni++) {
            const int c0 = bx*BN+wx*64+ni*8+t_epi*2;
            if (c0 >= N) continue;
            float* d = C_ptr+(long long)r0*ldc+c0;
            float f0 = __fmaf_rn(alpha, acc[mi][ni][0], 0.f);
            float f1 = __fmaf_rn(alpha, acc[mi][ni][1], 0.f);
            if (beta == 0.f) {
                d[0] = f0;
                if (c0+1<N) d[1] = f1;
            } else {
                if (c0+1<N) {
                    float2 o = *(float2*)d;
                    d[0] = __fmaf_rn(beta, o.x, f0);
                    d[1] = __fmaf_rn(beta, o.y, f1);
                } else {
                    d[0] = __fmaf_rn(beta, d[0], f0);
                }
            }
            if (r0+8 < M) {
                float* d8 = C_ptr+(long long)(r0+8)*ldc+c0;
                float f8_0 = __fmaf_rn(alpha, acc[mi][ni][2], 0.f);
                float f8_1 = __fmaf_rn(alpha, acc[mi][ni][3], 0.f);
                if (beta == 0.f) {
                    d8[0] = f8_0;
                    if (c0+1<N) d8[1] = f8_1;
                } else {
                    if (c0+1<N) {
                        float2 o = *(float2*)d8;
                        d8[0] = __fmaf_rn(beta, o.x, f8_0);
                        d8[1] = __fmaf_rn(beta, o.y, f8_1);
                    } else {
                        d8[0] = __fmaf_rn(beta, d8[0], f8_0);
                    }
                }
            }
        }
    }
}
