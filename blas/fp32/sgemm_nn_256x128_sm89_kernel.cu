// Batched single-precision GEMM (NN layout): C = alpha * A * B + beta * C.
// 256x128 tile, TF32 tensor-core MMA, multi-stage cp.async pipeline, optional split-K.

#include <cuda_runtime.h>
#include <stdint.h>

namespace nn256x128_opt {
    static constexpr int BM = 256, BN = 128, BK = 16, STAGES = 4, THREADS = 256;
    static constexpr int AS_SIZE = BM * BK, BS_SIZE = BK * BN, STAGE_SIZE = AS_SIZE + BS_SIZE;
    static constexpr int MMA_M = 4, MMA_N = 8;
}

template <bool IsSplitK>
__global__ void __launch_bounds__(256, 1)
sgemm_nn_256x128_sm89_2_opt_k(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK) {
    using namespace nn256x128_opt;

    // SplitK and Batch indexing
    const int batch = blockIdx.z / splitK;
    const int sk_idx = blockIdx.z % splitK;
    if (batch >= batchCount) return;

    // Robust CTA swizzling (8-wide strips) for L2 locality
    const int sw = 8;
    const int grid_x = gridDim.x, grid_y = gridDim.y;
    const int block_idx = blockIdx.y * grid_x + blockIdx.x;
    const int num_blocks_per_strip = sw * grid_y;
    const int strip_idx = block_idx / num_blocks_per_strip;
    const int strip_off = block_idx % num_blocks_per_strip;
    const int actual_sw = min(sw, grid_x - strip_idx * sw);
    const int bx = strip_idx * sw + (strip_off % actual_sw);
    const int by = strip_off / actual_sw;

    if (by * BM >= M || bx * BN >= N) return;

    const int tid = threadIdx.x, lane = tid % 32, wid = tid / 32;
    const int wy = wid / 2, wx = wid % 2;
    extern __shared__ float smem[];

    // K-chunking for SplitK
    const int k_chunk = ((K + 15) / 16 + splitK - 1) / splitK * 16;
    const int k_start = sk_idx * k_chunk;
    const int k_end   = min(K, k_start + k_chunk);
    const int k_tiles = (k_end - k_start + 15) / 16;

    const float* A_base = A + (long long)batch * strideA + (long long)k_start;
    const float* B_base = B + (long long)batch * strideB + (long long)k_start * ldb;
    float* C_ptr = C + (long long)batch * strideC;

    const int ar_m = tid / 4, ar_k = (tid % 4) * 4;
    const int br_k = tid / 32, br_n = (tid % 32) * 4;

    auto issue = [&](int stage, int ko) {
        float *As = smem + stage * STAGE_SIZE, *Bs = As + AS_SIZE;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = ar_m + i * 64, c = ko + ar_k;
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(&As[r * BK + (ar_k ^ ((r & 3) << 2))]);
            int bytes = ((by * BM + r) < M && (k_start + c) < k_end) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(sm), "l"(A_base + (long long)(by * BM + r) * lda + c), "r"(bytes));
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int r = br_k + i * 8, c = bx * BN + br_n;
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(&Bs[r * BN + (br_n ^ ((r & 3) << 2))]);
            int bytes = ((k_start + ko + r) < k_end && c < N) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(sm), "l"(B_base + (long long)(ko + r) * ldb + c), "r"(bytes));
        }
    };

    float acc[MMA_M][MMA_N][4] = {0};
    for (int s = 0; s < min(k_tiles, STAGES - 1); s++) { issue(s, s * BK); asm volatile("cp.async.commit_group;"); }
    asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2)); __syncthreads();

    uint32_t frA[2][MMA_M][4], frB[2][MMA_N][2];
    int rs = 0, ws = STAGES - 1;

    auto loadA = [&](uint32_t r[4], int ks, int mi, const float* As) {
        int r_s = wy * 64 + mi * 16 + (lane % 16), c_s = ks + (lane / 16) * 4;
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
            : "r"((uint32_t)__cvta_generic_to_shared(&As[r_s * BK + (c_s ^ ((r_s & 3) << 2))])));
        r[0] += 0x1000u; r[1] += 0x1000u; r[2] += 0x1000u; r[3] += 0x1000u;
    };
    auto loadB = [&](uint32_t r[2], int ks, int ni, const float* Bs) {
        int c_s = wx * 64 + ni * 8 + (lane / 4), k0 = ks + (lane % 4), k4 = k0 + 4;
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

    for (int kt = 0; kt < k_tiles; kt++) {
        int nk = (kt + STAGES - 1) * BK;
        if (kt + STAGES - 1 < k_tiles) issue(ws % STAGES, nk);
        asm volatile("cp.async.commit_group;");

        #pragma unroll
        for (int mi = 0; mi < MMA_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < MMA_N; ni++) {
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                      "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                    : "r"(frA[0][mi][0]), "r"(frA[0][mi][1]),
                      "r"(frA[0][mi][2]), "r"(frA[0][mi][3]),
                      "r"(frB[0][ni][0]), "r"(frB[0][ni][1]));
                if (ni % 4 == 0) {
                    int b_idx = mi * 2 + ni / 4;
                    loadB(frB[1][b_idx], 8, b_idx, Bs_rs);
                }
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
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                      "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                    : "r"(frA[1][mi][0]), "r"(frA[1][mi][1]),
                      "r"(frA[1][mi][2]), "r"(frA[1][mi][3]),
                      "r"(frB[1][ni][0]), "r"(frB[1][ni][1]));
                if (ni % 4 == 0) {
                    int b_idx = mi * 2 + ni / 4;
                    loadB(frB[0][b_idx], 0, b_idx, Bs_rs);
                }
            }
            if (kt + 1 < k_tiles) loadA(frA[0][mi], 0, mi, As_rs);
        }
    }

    const int g_epi = lane >> 2, t_epi = lane & 3;
    #pragma unroll
    for (int mi = 0; mi < MMA_M; mi++) {
        const int r0 = by * BM + wy * 64 + mi * 16 + g_epi, r8 = r0 + 8;
        if (r0 >= M) continue;
        #pragma unroll
        for (int j = 0; j < MMA_N; j++) {
            const int c0 = bx * BN + wx * 64 + j * 8 + t_epi * 2;
            if (c0 >= N) continue;
            float* dst = C_ptr + (long long)r0 * ldc + c0;
            float f0 = alpha * acc[mi][j][0], f1 = alpha * acc[mi][j][1];
            float f8_0 = alpha * acc[mi][j][2], f8_1 = alpha * acc[mi][j][3];

            if constexpr (IsSplitK) {
                atomicAdd(dst, f0); if (c0 + 1 < N) atomicAdd(dst + 1, f1);
                if (r8 < M) {
                    float* dst8 = C_ptr + (long long)r8 * ldc + c0;
                    atomicAdd(dst8, f8_0); if (c0 + 1 < N) atomicAdd(dst8 + 1, f8_1);
                }
            } else {
                if (beta != 0.f) { f0 += beta * dst[0]; if (c0 + 1 < N) f1 += beta * dst[1]; }
                dst[0] = f0; if (c0 + 1 < N) dst[1] = f1;
                if (r8 < M) {
                    float* dst8 = C_ptr + (long long)r8 * ldc + c0;
                    if (beta != 0.f) { f8_0 += beta * dst8[0]; if (c0 + 1 < N) f8_1 += beta * dst8[1]; }
                    dst8[0] = f8_0; if (c0 + 1 < N) dst8[1] = f8_1;
                }
            }
        }
    }
}

template __global__ void sgemm_nn_256x128_sm89_2_opt_k<false>(int, int, int, float, const float*, int, long long, const float*, int, long long, float, float*, int, long long, int, int);
template __global__ void sgemm_nn_256x128_sm89_2_opt_k<true>(int, int, int, float, const float*, int, long long, const float*, int, long long, float, float*, int, long long, int, int);
