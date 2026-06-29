// NN-layout batched SGEMM, 64x64 tile, TF32 MMA with 4-stage cp.async pipeline.

#include <cuda_runtime.h>
#include <stdint.h>

namespace nn64x64 {
    static constexpr int BM = 64, BN = 64, BK = 16, STAGES = 4, THREADS = 128;
    static constexpr int MMA_M = 2, MMA_N = 4;
    static constexpr int AS_SIZE = BM * BK, BS_SIZE = BK * BN, STAGE_SIZE = AS_SIZE + BS_SIZE;

    // 16-byte async copy from global to shared.
    __device__ __forceinline__ void cp_async(uint32_t smem_addr, const void* gmem_ptr) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr) : "memory");
    }

    // Load four 8x8 b16 matrices from shared into registers.
    #define LDMATRIX_X4(r0, r1, r2, r3, addr) \
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr))

    // m16n8k8 TF32 MMA accumulating in place.
    #define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1) \
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
            : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
}

__global__ void __launch_bounds__(128, 1)
sgemm_nn_64x64_sm89_2_k(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount) {
    using namespace nn64x64;
    const int batch = blockIdx.z; if (batch >= batchCount) return;
    const int bx = blockIdx.x, by = blockIdx.y;
    if (by * BM >= M || bx * BN >= N) return;
    const int tid = threadIdx.x, lane = tid % 32, wid = tid / 32;
    const int wy = wid / 2, wx = wid % 2;
    extern __shared__ float smem[];
    const float *A_ptr = A + (long long)batch * strideA, *B_ptr = B + (long long)batch * strideB;
    float* C_ptr = C + (long long)batch * strideC;
    const int a_row = tid / 4, a_col = (tid % 4) * 4;
    const int b_row = tid / 16, b_col = (tid % 16) * 4;
    auto issue = [&](int stage, int ko) {
        float *As = smem + stage * STAGE_SIZE, *Bs = As + AS_SIZE;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int r = a_row + i * 32, c = ko + a_col;
            uint32_t sm_addr = (uint32_t)__cvta_generic_to_shared(&As[r * BK + (a_col ^ ((r & 3) << 2))]);
            if (by * BM + r < M && c < K) cp_async(sm_addr, A_ptr + (long long)(by * BM + r) * lda + c);
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int r = ko + b_row + i * 8, c = bx * BN + b_col;
            uint32_t sm_addr = (uint32_t)__cvta_generic_to_shared(&Bs[(b_row + i * 8) * BN + (b_col ^ (( (b_row + i * 8) & 3) << 2))]);
            if (r < K && c < N) cp_async(sm_addr, B_ptr + (long long)r * ldb + c);
        }
    };
    float acc[2][4][4] = {0};
    const int k_tiles = (K + BK - 1) / BK;
    for (int s = 0; s < min(k_tiles, STAGES-1); s++) { issue(s, s * BK); asm volatile("cp.async.commit_group;\n"); }
    asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2)); __syncthreads();
    uint32_t frA[2][2][4], frB[2][4][2]; int rs = 0, ws = STAGES - 1;
    auto load_frA = [&](uint32_t r[4], int ks, int mi, const float* As) {
        int row = wy * 32 + mi * 16 + (lane % 16), col = ks + (lane / 16) * 4;
        LDMATRIX_X4(r[0], r[1], r[2], r[3], (uint32_t)__cvta_generic_to_shared(&As[row * BK + (col ^ ((row & 3) << 2))]));
        r[0] += 0x1000u; r[1] += 0x1000u; r[2] += 0x1000u; r[3] += 0x1000u;
    };
    auto load_frB = [&](uint32_t r[2], int ks, int ni, const float* Bs) {
        int col = wx * 32 + ni * 8 + (lane / 4), k0 = ks + (lane % 4), k4 = k0 + 4;
        r[0] = *(uint32_t*)(&Bs[k0 * BN + (col ^ ((k0 & 3) << 2))]);
        r[1] = *(uint32_t*)(&Bs[k4 * BN + (col ^ ((k4 & 3) << 2))]);
        r[0] += 0x1000u; r[1] += 0x1000u;
    };
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) load_frA(frA[0][mi], 0, mi, smem + rs * STAGE_SIZE);
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) load_frB(frB[0][ni], 0, ni, smem + rs * STAGE_SIZE + AS_SIZE);
    for (int kt = 0; kt < k_tiles; kt++) {
        int nk = (kt + STAGES - 1) * BK; if (nk < K) issue(ws % STAGES, nk);
        asm volatile("cp.async.commit_group;\n");
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                MMA_TF32(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3], frA[0][mi][0], frA[0][mi][1], frA[0][mi][2], frA[0][mi][3], frB[0][ni][0], frB[0][ni][1]);
                if (mi == 0) load_frB(frB[1][ni], 8, ni, smem + rs * STAGE_SIZE + AS_SIZE);
            }
            load_frA(frA[1][mi], 8, mi, smem + rs * STAGE_SIZE);
        }
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2)); __syncthreads();
        rs = (rs + 1) % STAGES; ws = (ws + 1) % STAGES;
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                MMA_TF32(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3], frA[1][mi][0], frA[1][mi][1], frA[1][mi][2], frA[1][mi][3], frB[1][ni][0], frB[1][ni][1]);
                if (mi == 0) load_frB(frB[0][ni], 0, ni, smem + rs * STAGE_SIZE + AS_SIZE);
            }
            load_frA(frA[0][mi], 0, mi, smem + rs * STAGE_SIZE);
        }
    }
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int r0 = by * BM + wy * 32 + mi * 16 + (lane / 4), c0 = bx * BN + wx * 32 + ni * 8 + (lane % 4) * 2;
            if (r0 < M && c0 < N) {
                float* dst = C_ptr + (long long)r0 * ldc + c0;
                dst[0] = alpha * acc[mi][ni][0] + beta * dst[0];
                if (c0 + 1 < N) dst[1] = alpha * acc[mi][ni][1] + beta * dst[1];
            }
            int r8 = r0 + 8;
            if (r8 < M && c0 < N) {
                float* dst = C_ptr + (long long)r8 * ldc + c0;
                dst[0] = alpha * acc[mi][ni][2] + beta * dst[0];
                if (c0 + 1 < N) dst[1] = alpha * acc[mi][ni][3] + beta * dst[1];
            }
        }
    }
}
