// TN-layout batched SGEMM, 128x128 tile, TF32 MMA with split-K accumulation.

#include <cuda_runtime.h>
#include <stdint.h>

namespace tn128x128 {
    static constexpr int BM = 128, BN = 128, BK = 16, STAGES = 4, THREADS = 256;
    static constexpr int AS_SIZE = BM * BK, BS_SIZE = BK * BN, STAGE_SIZE = AS_SIZE + BS_SIZE;
    static constexpr int MMA_M = 4, MMA_N = 4;
}

// m16n8k8 TF32 tensor-core MMA, accumulating into d0..d3.
static __device__ __forceinline__ void mma_tf32(float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(256, 1)
sgemm_tn_128x128_sm89_2_k(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK) {
    using namespace tn128x128;
    const int batch = blockIdx.z / splitK; if (batch >= batchCount) return;
    const int sk_idx = blockIdx.z % splitK;
    const int bx = blockIdx.x, by = blockIdx.y;
    if (by * BM >= M || bx * BN >= N) return;

    const int tiles_per_sk = ((K + 15) / 16 + splitK - 1) / splitK;
    const int kt_start = sk_idx * tiles_per_sk;
    const int kt_end = min((K + 15) / 16, (sk_idx + 1) * tiles_per_sk);
    if (kt_start >= kt_end) return;

    const int tid = threadIdx.x, lane = tid % 32, wid = tid / 32;
    const int wy = wid / 4, wx = wid % 4;
    extern __shared__ float smem[];
    const float *A_ptr = A + (long long)batch * strideA, *B_ptr = B + (long long)batch * strideB;
    float* C_ptr = C + (long long)batch * strideC;
    const int ar_m = (tid % 32) * 4, ar_k = tid / 32;
    const int br_n = (tid % 32) * 4, br_k = tid / 32;

    // Async copy of one K-slab of A and B tiles into a shared-memory stage.
    auto issue = [&](int stage, int ko) {
        float *As = smem + stage * STAGE_SIZE, *Bs = As + AS_SIZE;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int r = ar_k + i * 8, c = ar_m;
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(&As[r * BM + (c ^ ((r & 3) << 2))]);
            int bytes = ((ko + r) < K && (by * BM + c) < M) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(sm), "l"(A_ptr + (long long)(ko + r) * lda + by * BM + c), "r"(bytes));
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int r = br_k + i * 8, c = br_n;
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(&Bs[r * BN + (c ^ ((r & 3) << 2))]);
            int bytes = ((ko + r) < K && (bx * BN + c) < N) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(sm), "l"(B_ptr + (long long)(ko + r) * ldb + bx * BN + c), "r"(bytes));
        }
    };

    float acc[MMA_M][MMA_N][4] = {0};
    int stages_to_load = min(kt_end - kt_start, STAGES - 1);
    for (int s = 0; s < stages_to_load; s++) { issue(s, (kt_start + s) * BK); asm volatile("cp.async.commit_group;"); }
    asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2)); __syncthreads();

    uint32_t frA[2][MMA_M][4], frB[2][MMA_N][2];
    int rs = 0, ws = stages_to_load % STAGES;

    // Load A fragment for one MMA tile from a swizzled shared stage.
    auto loadA = [&](uint32_t r[4], int ks, int mi, const float* As) {
        const int m_base = wy * 64 + mi * 16;
        const int row = m_base + (lane / 4), c0 = ks + (lane % 4);
        r[0] = *(const uint32_t*)&As[c0 * BM + (row ^ ((c0 & 3) << 2))];
        r[1] = *(const uint32_t*)&As[c0 * BM + ((row + 8) ^ ((c0 & 3) << 2))];
        const int c4 = c0 + 4;
        r[2] = *(const uint32_t*)&As[c4 * BM + (row ^ ((c4 & 3) << 2))];
        r[3] = *(const uint32_t*)&As[c4 * BM + ((row + 8) ^ ((c4 & 3) << 2))];
        r[0] += 0x1000u; r[1] += 0x1000u; r[2] += 0x1000u; r[3] += 0x1000u;
    };
    // Load B fragment for one MMA tile from a swizzled shared stage.
    auto loadB = [&](uint32_t r[2], int ks, int ni, const float* Bs) {
        int c_s = wx * 32 + ni * 8 + (lane / 4), k0 = ks + (lane % 4), k4 = k0 + 4;
        r[0] = *(const uint32_t*)(&Bs[k0 * BN + (c_s ^ ((k0 & 3) << 2))]);
        r[1] = *(const uint32_t*)(&Bs[k4 * BN + (c_s ^ ((k4 & 3) << 2))]);
        r[0] += 0x1000u; r[1] += 0x1000u;
    };

    const float* As_rs = smem + rs * STAGE_SIZE;
    const float* Bs_rs = As_rs + AS_SIZE;
    #pragma unroll
    for (int mi = 0; mi < MMA_M; mi++) loadA(frA[0][mi], 0, mi, As_rs);
    #pragma unroll
    for (int ni = 0; ni < MMA_N; ni++) loadB(frB[0][ni], 0, ni, Bs_rs);

    for (int kt = kt_start; kt < kt_end; kt++) {
        int nk_t = kt + (STAGES - 1);
        if (nk_t < kt_end) issue(ws, nk_t * BK);
        asm volatile("cp.async.commit_group;");

        #pragma unroll
        for (int mi = 0; mi < MMA_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < MMA_N; ni++) {
                mma_tf32(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                         frA[0][mi][0], frA[0][mi][1], frA[0][mi][2], frA[0][mi][3],
                         frB[0][ni][0], frB[0][ni][1]);
                if (ni == 0) loadB(frB[1][mi], 8, mi, Bs_rs);
            }
            loadA(frA[1][mi], 8, mi, As_rs);
        }

        asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2)); __syncthreads();
        rs = (rs + 1) % STAGES; ws = (ws + 1) % STAGES;
        As_rs = smem + rs * STAGE_SIZE;
        Bs_rs = As_rs + AS_SIZE;

        #pragma unroll
        for (int mi = 0; mi < MMA_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < MMA_N; ni++) {
                mma_tf32(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                         frA[1][mi][0], frA[1][mi][1], frA[1][mi][2], frA[1][mi][3],
                         frB[1][ni][0], frB[1][ni][1]);
                if (ni == 0) loadB(frB[0][mi], 0, mi, Bs_rs);
            }
            loadA(frA[0][mi], 0, mi, As_rs);
        }
    }

    const int g_epi = lane >> 2, t_epi = lane & 3;
    #pragma unroll
    for (int mi = 0; mi < MMA_M; mi++) {
        const int r0 = by * BM + wy * 64 + mi * 16 + g_epi, r8 = r0 + 8;
        if (r0 >= M) continue;
        #pragma unroll
        for (int ni = 0; ni < MMA_N; ni++) {
            const int c0 = bx * BN + wx * 32 + ni * 8 + t_epi * 2;
            if (c0 >= N) continue;
            float* d = C_ptr + (long long)r0 * ldc + c0;
            float f0 = __fmaf_rn(alpha, acc[mi][ni][0], 0.f);
            float f1 = __fmaf_rn(alpha, acc[mi][ni][1], 0.f);
            if (splitK > 1) {
                atomicAdd(d, f0);
                if (c0 + 1 < N) atomicAdd(d + 1, f1);
                if (r8 < M) {
                    float* d8 = C_ptr + (long long)r8 * ldc + c0;
                    atomicAdd(d8, __fmaf_rn(alpha, acc[mi][ni][2], 0.f));
                    if (c0 + 1 < N) atomicAdd(d8 + 1, __fmaf_rn(alpha, acc[mi][ni][3], 0.f));
                }
            } else {
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
                if (r8 < M) {
                    float* d8 = C_ptr + (long long)r8 * ldc + c0;
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
}
