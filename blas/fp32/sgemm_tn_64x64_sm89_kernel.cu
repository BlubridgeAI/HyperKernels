// TN-layout batched SGEMM, 64x64 tile, TF32 MMA with optional splitK.
// SM89 (Ada): 128 threads/CTA, 2-stage cp.async pipeline.

#include <cuda_runtime.h>
#include <stdint.h>

namespace tn64x64 {
    static constexpr int BM = 64, BN = 64, BK = 16, STAGES = 2, THREADS = 128;
    static constexpr int AS_SIZE = BM * BK, BS_SIZE = BK * BN, STAGE_SIZE = AS_SIZE + BS_SIZE;
    static constexpr int MMA_M = 2, MMA_N = 4; // Per Warp: 2x4 MMA calls (32x32 output)
}

static __device__ __forceinline__ void mma_tf32(float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(128, 1)
sgemm_tn_64x64_sm89_k(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK) {
    using namespace tn64x64;
    const int batch = blockIdx.z / splitK; if (batch >= batchCount) return;
    const int sk_idx = blockIdx.z % splitK;
    const int bx = blockIdx.x, by = blockIdx.y;
    if (by * BM >= M || bx * BN >= N) return;

    const int tiles_per_sk = ((K + 15) / 16 + splitK - 1) / splitK;
    const int kt_start = sk_idx * tiles_per_sk, kt_end = min((K + 15) / 16, (sk_idx + 1) * tiles_per_sk);
    if (kt_start >= kt_end) return;

    const int tid = threadIdx.x, lane = tid % 32, wid = tid / 32;
    const int wy = wid / 2, wx = wid % 2;
    extern __shared__ float smem[];
    const float *A_ptr = A + (long long)batch * strideA, *B_ptr = B + (long long)batch * strideB;

    auto issue = [&](int stage, int ko) {
        float *As = smem + stage * STAGE_SIZE, *Bs = As + AS_SIZE;
        // A tile is 64x16; 128 threads load 4 floats each, so 2 passes cover it.
        for (int i = 0; i < 2; i++) {
            int r = (tid / 16) + i * 8, c = (tid % 16) * 4;
            uint32_t sm_a = (uint32_t)__cvta_generic_to_shared(&As[r * BM + (c ^ ((r & 7) << 2))]);
            int b_a = ((ko + r) < K && (by * BM + c) < M) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(sm_a), "l"(A_ptr + (long long)(ko + r) * lda + by * BM + c), "r"(b_a));
            uint32_t sm_b = (uint32_t)__cvta_generic_to_shared(&Bs[r * BN + (c ^ ((r & 7) << 2))]);
            int b_b = ((ko + r) < K && (bx * BN + c) < N) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(sm_b), "l"(B_ptr + (long long)(ko + r) * ldb + bx * BN + c), "r"(b_b));
        }
    };

    float acc[2][4][4] = {0};
    for (int s = 0; s < min(kt_end - kt_start, STAGES - 1); s++) { issue(s, (kt_start + s) * BK); asm volatile("cp.async.commit_group;"); }
    asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2)); __syncthreads();

    uint32_t frA[2][2][4], frB[2][4][2];
    int rs = 0, ws = (STAGES - 1) % STAGES;

    auto loadA = [&](uint32_t r[2][4], int ks, const float* As) {
        const int row = wy * 32 + (lane % 16), c0 = ks + (lane / 16) * 4, c4 = c0 + 4;
        auto la = [&](int k_idx, int m_idx) { return *(uint32_t*)&As[k_idx * BM + (m_idx ^ ((k_idx & 7) << 2))]; };
        r[0][0] = la(c0, row); r[0][1] = la(c0, row + 8); r[0][2] = la(c4, row); r[0][3] = la(c4, row + 8);
        r[1][0] = la(c0, row + 16); r[1][1] = la(c0, row + 24); r[1][2] = la(c4, row + 16); r[1][3] = la(c4, row + 24);
        r[0][0] += 0x1000u; r[0][1] += 0x1000u; r[0][2] += 0x1000u; r[0][3] += 0x1000u;
        r[1][0] += 0x1000u; r[1][1] += 0x1000u; r[1][2] += 0x1000u; r[1][3] += 0x1000u;
    };
    auto loadB = [&](uint32_t r[4][2], int ks, const float* Bs) {
        const int col = wx * 32 + (lane / 4), k0 = ks + (lane % 4), k4 = k0 + 4;
        auto lb = [&](int k_idx, int n_idx) { return *(uint32_t*)&Bs[k_idx * BN + (n_idx ^ ((k_idx & 7) << 2))]; };
        r[0][0] = lb(k0, col); r[0][1] = lb(k4, col);
        r[1][0] = lb(k0, col + 8); r[1][1] = lb(k4, col + 8);
        r[2][0] = lb(k0, col + 16); r[2][1] = lb(k4, col + 16);
        r[3][0] = lb(k0, col + 24); r[3][1] = lb(k4, col + 24);
        r[0][0] += 0x1000u; r[0][1] += 0x1000u;
        r[1][0] += 0x1000u; r[1][1] += 0x1000u;
        r[2][0] += 0x1000u; r[2][1] += 0x1000u;
        r[3][0] += 0x1000u; r[3][1] += 0x1000u;
    };

    loadA(frA[0], 0, smem + rs * STAGE_SIZE);
    loadB(frB[0], 0, smem + rs * STAGE_SIZE + AS_SIZE);

    for (int kt = kt_start; kt < kt_end; kt++) {
        if (kt + (STAGES - 1) < kt_end) issue(ws, (kt + STAGES - 1) * BK);
        asm volatile("cp.async.commit_group;");

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) mma_tf32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3], frA[0][i][0], frA[0][i][1], frA[0][i][2], frA[0][i][3], frB[0][j][0], frB[0][j][1]);
        }
        loadA(frA[1], 8, smem + rs * STAGE_SIZE);
        loadB(frB[1], 8, smem + rs * STAGE_SIZE + AS_SIZE);

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) mma_tf32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3], frA[1][i][0], frA[1][i][1], frA[1][i][2], frA[1][i][3], frB[1][j][0], frB[1][j][1]);
        }

        asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2)); __syncthreads();
        rs = (rs + 1) % STAGES; ws = (ws + 1) % STAGES;
        loadA(frA[0], 0, smem + rs * STAGE_SIZE);
        loadB(frB[0], 0, smem + rs * STAGE_SIZE + AS_SIZE);
    }

    float* C_batch = C + (long long)batch * strideC;
    const int ge = lane >> 2, te = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int r0 = by * BM + wy * 32 + i * 16 + ge, r8 = r0 + 8;
        if (r0 >= M) continue;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int c0 = bx * BN + wx * 32 + j * 8 + te * 2;
            if (c0 >= N) continue;
            float* d = C_batch + (long long)r0 * ldc + c0;
            if (splitK > 1) {
                atomicAdd(d, alpha * acc[i][j][0]); if (c0 + 1 < N) atomicAdd(d + 1, alpha * acc[i][j][1]);
                if (r8 < M) { float* d8 = C_batch + (long long)r8 * ldc + c0; atomicAdd(d8, alpha * acc[i][j][2]); if (c0 + 1 < N) atomicAdd(d8 + 1, alpha * acc[i][j][3]); }
            } else {
                d[0] = alpha * acc[i][j][0] + (beta == 0.f ? 0.f : beta * d[0]); if (c0 + 1 < N) d[1] = alpha * acc[i][j][1] + (beta == 0.f ? 0.f : beta * d[1]);
                if (r8 < M) { float* d8 = C_batch + (long long)r8 * ldc + c0; d8[0] = alpha * acc[i][j][2] + (beta == 0.f ? 0.f : beta * d8[0]); if (c0 + 1 < N) d8[1] = alpha * acc[i][j][3] + (beta == 0.f ? 0.f : beta * d8[1]); }
            }
        }
    }
}
