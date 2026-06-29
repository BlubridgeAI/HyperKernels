// Bgemm backward NT: C = alpha * A * B^T + beta * C (BF16, dA = dY * dX^T).
// 128x128x32 tile, 3-stage cp.async pipeline, bf16 MMA; templated on alignment.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define BM      128
#define BN      128
#define BK      32
#define STAGES  3
#define THREADS 128
#define AS_SIZE (BM * BK)
#define BS_SIZE (BN * BK)
#define STG_SZ  (AS_SIZE + BS_SIZE)

#define MMA_M16N8K16_F32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1) \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};" \
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

template <bool Aligned>
__global__ void __launch_bounds__(THREADS, 1)
bgemm_nt_backward_kernel(
    int M, int N, int K,
    __nv_bfloat16 alpha,
    const __nv_bfloat16* __restrict__ A, int lda, long long sA,
    const __nv_bfloat16* __restrict__ B, int ldb, long long sB,
    __nv_bfloat16 beta,
    __nv_bfloat16* __restrict__ C, int ldc, long long sC,
    int batchCount)
{
    const int batch = blockIdx.z;
    if (batch >= batchCount) return;

    int bx = blockIdx.x, by = blockIdx.y;
    const int sw = 8;
    if (gridDim.y % sw == 0) {
        const int bid = blockIdx.y * gridDim.x + blockIdx.x;
        by = (bid % sw) + (bid / (gridDim.x * sw)) * sw;
        bx = (bid / sw) % gridDim.x;
    }
    if (by * BM >= M || bx * BN >= N) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int wy = wid >> 1, wx = wid & 1;

    const __nv_bfloat16* A_batch = A + (long long)batch * sA;
    const __nv_bfloat16* B_batch = B + (long long)batch * sB;
    __nv_bfloat16*       C_batch = C + (long long)batch * sC;

    extern __shared__ __nv_bfloat16 s_mem[];
    
    float acc[4][8][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) for (int d = 0; d < 4; d++) acc[i][j][d] = 0.f;

    auto load_tile = [&](int stage, int ko) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = tid; int c = i * 8;
            const __nv_bfloat16* g_ptr = A_batch + (long long)(by * BM + r) * lda + (ko + c);
            int swizzled_c = c ^ ((r & 3) << 3); 
            uint32_t sm_addr = __cvta_generic_to_shared(&s_mem[stage * STG_SZ + r * BK + swizzled_c]);
            int bytes = (by * BM + r < M && ko + c < K) ? max(0, min(16, (K - (ko + c)) * 2)) : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_addr), "l"(g_ptr), "r"(bytes));
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = tid; int c = i * 8;
            const __nv_bfloat16* g_ptr = B_batch + (long long)(bx * BN + r) * ldb + (ko + c); 
            int swizzled_c = c ^ ((r & 3) << 3); 
            uint32_t sm_addr = __cvta_generic_to_shared(&s_mem[stage * STG_SZ + AS_SIZE + r * BK + swizzled_c]);
            int bytes = (bx * BN + r < N && ko + c < K) ? max(0, min(16, (K - (ko + c)) * 2)) : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_addr), "l"(g_ptr), "r"(bytes));
        }
    };

    load_tile(0, 0); asm volatile("cp.async.commit_group;\n");
    if (BK < K) { load_tile(1, BK); asm volatile("cp.async.commit_group;\n"); }
    
    int ws = 2, rs = 0;
    uint32_t frA[2][4][4], frB[2][8][2];

    asm volatile("cp.async.wait_group 1;\n"); __syncthreads();

    auto load_regA = [&](uint32_t reg[4], int ki, int mi, int st) {
        int r = wy * 64 + mi * 16 + ((lane / 8) % 2) * 8 + (lane % 8);
        int c = ki + (lane / 16) * 8;
        int swizzled_c = c ^ ((r & 3) << 3);
        uint32_t addr = __cvta_generic_to_shared(&s_mem[st * STG_SZ + r * BK + swizzled_c]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" 
                     : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3]) : "r"(addr));
    };

    auto load_regB = [&](uint32_t reg[2], int ki, int ni, int st) {
        int r = wx * 64 + ni * 8 + (lane % 8);
        int c = ki + (lane / 8 % 2) * 8;
        int swizzled_c = c ^ ((r & 3) << 3);
        uint32_t addr = __cvta_generic_to_shared(&s_mem[st * STG_SZ + AS_SIZE + r * BK + swizzled_c]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" 
                     : "=r"(reg[0]), "=r"(reg[1]) : "r"(addr));
    };

    #pragma unroll
    for (int i = 0; i < 4; i++) load_regA(frA[0][i], 0, i, rs);
    #pragma unroll
    for (int j = 0; j < 8; j++) load_regB(frB[0][j], 0, j, rs);

    for (int k = 0; k < K; k += BK) {
        if (k + 2 * BK < K) load_tile(ws, k + 2 * BK); 
        asm volatile("cp.async.commit_group;\n");

        #pragma unroll
        for (int ks = 0; ks < 32; ks += 16) {
            int p = (ks >> 4) & 1, q = p ^ 1;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                if (ks + 16 < 32) {
                    load_regA(frA[q][i], ks + 16, i, rs);
                    load_regB(frB[q][i*2], ks + 16, i*2, rs); load_regB(frB[q][i*2+1], ks + 16, i*2+1, rs);
                } else if (k + BK < K) {
                    if (i == 0) {
                        asm volatile("cp.async.wait_group 1;\n"); __syncthreads();
                        rs = (rs + 1) % STAGES; ws = (ws + 1) % STAGES;
                    }
                    load_regA(frA[q][i], 0, i, rs);
                    load_regB(frB[q][i*2], 0, i*2, rs); load_regB(frB[q][i*2+1], 0, i*2+1, rs);
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) MMA_M16N8K16_F32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3], frA[p][i][0], frA[p][i][1], frA[p][i][2], frA[p][i][3], frB[p][j][0], frB[p][j][1]);
            }
        }
    }

    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    const float alpha_f = __bfloat162float(alpha);
    const int rt = lane / 4;
    const int ct = (lane % 4) * 2;

    // Scatter accumulators into SMEM with XOR swizzle to avoid bank conflicts
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int r = wy * 64 + i * 16 + rt;
            int c = wx * 64 + j * 8 + ct;
            int sr = r ^ ((c / 8) & 7);
            *(__nv_bfloat162*)&s_mem[sr * BN + c]       = {__float2bfloat16(acc[i][j][0] * alpha_f), __float2bfloat16(acc[i][j][1] * alpha_f)};
            *(__nv_bfloat162*)&s_mem[(sr ^ 8) * BN + c] = {__float2bfloat16(acc[i][j][2] * alpha_f), __float2bfloat16(acc[i][j][3] * alpha_f)};
        }
    }
    __syncthreads();

    // Gather from SMEM with vectorized int4 (128-bit) global stores
    const __nv_bfloat162 h2beta = {beta, beta};
    #pragma unroll
    for (int i = 0; i < (BM * BN) / (THREADS * 8); i++) {
        int r = (tid / (BN / 8)) + i * (THREADS / (BN / 8));
        int c = (tid % (BN / 8)) * 8;
        int gr = by * BM + r;
        int gc = bx * BN + c;
        if (gr < M && gc < N) {
            int sr = r ^ ((c / 8) & 7);
            int4 vals = *(int4*)&s_mem[sr * BN + c];
            if (beta != __float2bfloat16(0.0f)) {
                int4 old = *(int4*)&C_batch[(long long)gr * ldc + gc];
                __nv_bfloat162* h2v = (__nv_bfloat162*)&vals;
                const __nv_bfloat162* h2o = (const __nv_bfloat162*)&old;
                #pragma unroll
                for (int l = 0; l < 4; l++) h2v[l] = __hadd2(h2v[l], __hmul2(h2o[l], h2beta));
            }
            *(int4*)&C_batch[(long long)gr * ldc + gc] = vals;
        }
    }
}

template __global__ void bgemm_nt_backward_kernel<true>(int,int,int,__nv_bfloat16,const __nv_bfloat16*,int,long long int,const __nv_bfloat16*,int,long long int,__nv_bfloat16,__nv_bfloat16*,int,long long int,int);
template __global__ void bgemm_nt_backward_kernel<false>(int,int,int,__nv_bfloat16,const __nv_bfloat16*,int,long long int,const __nv_bfloat16*,int,long long int,__nv_bfloat16,__nv_bfloat16*,int,long long int,int);
