// Batched BF16 GEMM, 128x128x32 tile, 3-stage cp.async pipeline, bf16 MMA.
// Templated on the transpose state of A and B (NN / NT / TN / TT).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Transpose state of A / B, matching the library's mycublasOperation_t encoding.
enum mycublasOperation_t { MYCUBLAS_OP_N = 0, MYCUBLAS_OP_T = 1, MYCUBLAS_OP_C = 2 };

#define BM 128
#define BN 128
#define BK 32
#define STAGES 3
#define THREADS_PER_BLOCK 128

#define MMA_M16N8K16_F32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1) \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};" \
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

template <mycublasOperation_t transA, mycublasOperation_t transB>
__global__ void __launch_bounds__(128, 2)
bgemm_NN_kernel(
    int M, int N, int K,
    __nv_bfloat16 alpha,
    const __nv_bfloat16* __restrict__ A, int lda, long long int strideA,
    const __nv_bfloat16* __restrict__ B, int ldb, long long int strideB,
    __nv_bfloat16 beta,
    __nv_bfloat16* __restrict__ C_ptr, int ldc, long long int strideC,
    int batchCount)
{
    const int batch_idx = (int)blockIdx.z;
    if (batch_idx >= batchCount) return;

    const int bx = blockIdx.x, by = blockIdx.y;
    if (by * BM >= M || bx * BN >= N) return;

    const __nv_bfloat16* A_batch = A + (long long)batch_idx * strideA;
    const __nv_bfloat16* B_batch = B + (long long)batch_idx * strideB;
    __nv_bfloat16*       C_batch = C_ptr + (long long)batch_idx * strideC;

    extern __shared__ __nv_bfloat16 s_mem[];
    const int s_step_a = BM * BK; 
    const int s_step_b = BN * BK;
    const int s_stage_size = s_step_a + s_step_b;
    
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    float acc[4][8][4]; 
    #pragma unroll
    for(int i=0; i<4; i++) for(int j=0; j<8; j++) for(int l=0; l<4; l++) acc[i][j][l] = 0.0f;

    auto load_to_stage = [&](int stage, int k_off) {
        // Load A: 128 (M) x 32 (K). 128 threads.
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            // Coalesced loading: 4 threads per row.
            int r = (tid / 4) + i * 32;
            int c = (tid % 4) * 8;
            const __nv_bfloat16* g_ptr;
            int pred;
            if constexpr (transA == MYCUBLAS_OP_N) {
                g_ptr = A_batch + (by * BM + r) * lda + (k_off + c);
                pred = (by * BM + r < M && k_off + c < K) ? 16 : 0;
            } else {
                g_ptr = A_batch + (k_off + c) * lda + (by * BM + r);
                pred = (k_off + c < K && by * BM + r < M) ? 16 : 0;
            }
            int swizzled_c = c ^ ((r & 3) << 3); 
            uint32_t sm_addr = __cvta_generic_to_shared(&s_mem[stage * s_stage_size + r * BK + swizzled_c]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_addr), "l"(g_ptr), "r"(pred));
        }
        // Load B: 32 (K) x 128 (N). 128 threads.
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r_tile = (tid / 16) + i * 8;
            int c_tile = (tid % 16) * 8;
            const __nv_bfloat16* g_ptr;
            int pred;
            if constexpr (transB == MYCUBLAS_OP_N) {
                g_ptr = B_batch + (k_off + r_tile) * ldb + (bx * BN + c_tile);
                pred = (k_off + r_tile < K && bx * BN + c_tile < N) ? 16 : 0;
            } else {
                g_ptr = B_batch + (bx * BN + c_tile) * ldb + (k_off + r_tile);
                pred = (bx * BN + c_tile < N && k_off + r_tile < K) ? 16 : 0;
            }
            int swizzled_c = c_tile ^ ((r_tile & 7) << 3);
            uint32_t sm_addr = __cvta_generic_to_shared(&s_mem[stage * s_stage_size + s_step_a + r_tile * BN + swizzled_c]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_addr), "l"(g_ptr), "r"(pred));
        }
    };

    int write_stage = 0;
    load_to_stage(0, 0);
    asm volatile("cp.async.commit_group;\n");
    load_to_stage(1, BK);
    asm volatile("cp.async.commit_group;\n");
    write_stage = 2;

    int read_stage = 0;
    const int wy = wid / 2;
    const int wx = wid % 2;
    
    uint32_t frA[2][4][4]; 
    uint32_t frB[2][8][2]; 

    asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
    __syncthreads();

    auto load_frA = [&](uint32_t reg[4], int ki, int mi, int st) {
        int r = wy * 64 + mi * 16 + ((lane / 8) % 2) * 8 + (lane % 8);
        int c = ki + (lane / 16) * 8;
        int swizzled_c = c ^ ((r & 3) << 3);
        uint32_t addr = __cvta_generic_to_shared(&s_mem[st * s_stage_size + r * BK + swizzled_c]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" 
                     : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3]) : "r"(addr));
    };

    auto load_frB = [&](uint32_t reg[2], int ki, int ni, int st) {
        int r = ki + ((lane / 8) % 2) * 8 + (lane % 8);
        int c = wx * 64 + ni * 8;
        int swizzled_c = c ^ ((r & 7) << 3);
        uint32_t addr = __cvta_generic_to_shared(&s_mem[st * s_stage_size + s_step_a + r * BN + swizzled_c]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" 
                     : "=r"(reg[0]), "=r"(reg[1]) : "r"(addr));
    };

    #pragma unroll
    for (int i = 0; i < 4; i++) load_frA(frA[0][i], 0, i, read_stage);
    #pragma unroll
    for (int j = 0; j < 8; j++) load_frB(frB[0][j], 0, j, read_stage);

    for (int k = 0; k < K; k += BK) {
        if (k + (STAGES - 1) * BK < K) {
            load_to_stage(write_stage, k + (STAGES - 1) * BK);
        }
        asm volatile("cp.async.commit_group;\n");

        #pragma unroll
        for (int ks = 0; ks < 32; ks += 16) {
            int p = (ks >> 4) & 1; 
            int q = p ^ 1;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                if (ks + 16 < 32) {
                    load_frA(frA[q][i], ks + 16, i, read_stage);
                    load_frB(frB[q][i*2], ks + 16, i*2, read_stage);
                    load_frB(frB[q][i*2+1], ks + 16, i*2+1, read_stage);
                } else if (k + BK < K) {
                    if (i == 0) {
                        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
                        __syncthreads();
                        read_stage = (read_stage + 1) % STAGES;
                        write_stage = (write_stage + 1) % STAGES;
                    }
                    load_frA(frA[q][i], 0, i, read_stage);
                    load_frB(frB[q][i*2], 0, i*2, read_stage);
                    load_frB(frB[q][i*2+1], 0, i*2+1, read_stage);
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    MMA_M16N8K16_F32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                                     frA[p][i][0], frA[p][i][1], frA[p][i][2], frA[p][i][3], 
                                     frB[p][j][0], frB[p][j][1]);
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    
    __nv_bfloat16* s_out = s_mem;
    const int rt = lane / 4; 
    const int ct = (lane % 4) * 2; 

    float alpha_f = __bfloat162float(alpha);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int r = wy * 64 + i * 16 + rt;
            int c = wx * 64 + j * 8 + ct;
            __nv_bfloat162 h2_01 = {__float2bfloat16(acc[i][j][0] * alpha_f), __float2bfloat16(acc[i][j][1] * alpha_f)};
            __nv_bfloat162 h2_23 = {__float2bfloat16(acc[i][j][2] * alpha_f), __float2bfloat16(acc[i][j][3] * alpha_f)};
            *(__nv_bfloat162*)&s_out[r * BN + c] = h2_01;
            *(__nv_bfloat162*)&s_out[(r+8) * BN + c] = h2_23;
        }
    }
    __syncthreads();

    const __nv_bfloat162 h2beta = {beta, beta};
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int r = (tid >> 4) + i * 8; 
        int c = (tid & 15) << 3;
        int gr = by * BM + r;
        int gc = bx * BN + c;
        if (gr < M && gc < N) {
            int4 vals = *(int4*)&s_out[r * BN + c];
            if (beta != __float2bfloat16(0.0f)) {
                int4 old = *(int4*)&C_batch[(long long)gr * ldc + gc];
                __nv_bfloat162* h2v = (__nv_bfloat162*)&vals;
                __nv_bfloat162* h2o = (__nv_bfloat162*)&old;
                #pragma unroll
                for(int l = 0; l < 4; l++) h2v[l] = __hadd2(h2v[l], __hmul2(h2o[l], h2beta));
            }
            *(int4*)&C_batch[(long long)gr * ldc + gc] = vals;
        }
    }
}

template __global__ void bgemm_NN_kernel<MYCUBLAS_OP_N, MYCUBLAS_OP_N>(int,int,int,__nv_bfloat16,const __nv_bfloat16*,int,long long int,const __nv_bfloat16*,int,long long int,__nv_bfloat16,__nv_bfloat16*,int,long long int,int);
template __global__ void bgemm_NN_kernel<MYCUBLAS_OP_N, MYCUBLAS_OP_T>(int,int,int,__nv_bfloat16,const __nv_bfloat16*,int,long long int,const __nv_bfloat16*,int,long long int,__nv_bfloat16,__nv_bfloat16*,int,long long int,int);
template __global__ void bgemm_NN_kernel<MYCUBLAS_OP_T, MYCUBLAS_OP_N>(int,int,int,__nv_bfloat16,const __nv_bfloat16*,int,long long int,const __nv_bfloat16*,int,long long int,__nv_bfloat16,__nv_bfloat16*,int,long long int,int);
template __global__ void bgemm_NN_kernel<MYCUBLAS_OP_T, MYCUBLAS_OP_T>(int,int,int,__nv_bfloat16,const __nv_bfloat16*,int,long long int,const __nv_bfloat16*,int,long long int,__nv_bfloat16,__nv_bfloat16*,int,long long int,int);
