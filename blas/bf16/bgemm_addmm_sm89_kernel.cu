// Bgemm Addmm - fused batched BF16 matmul + bias: C = alpha*A*B + beta*bias (NN).
// Multi-stage cp.async pipeline, bf16 MMA; templated on tile config and alignment.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define MMA_M16N8K16_F32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)               \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "                   \
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"        \
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)                                 \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned>
__global__ void __launch_bounds__(THREADS, 1)
    bgemm_addmm_kernel(int M, int N, int K, __nv_bfloat16 alpha,
                       const __nv_bfloat16 *__restrict__ A, int lda,
                       long long strideA, const __nv_bfloat16 *__restrict__ B,
                       int ldb, long long strideB, __nv_bfloat16 beta,
                       const __nv_bfloat16 *__restrict__ bias,
                       int64_t bias_numel, __nv_bfloat16 *__restrict__ C,
                       int ldc, long long strideC, int batchCount) {
  constexpr int AS_SIZE = BM * BK;
  constexpr int BS_SIZE = BK * BN;
  constexpr int STAGE_SIZE = AS_SIZE + BS_SIZE;

  // Warp layout (WARPS_N always 2 for our BN=128 tiles)
  constexpr int WARP_COUNT = THREADS / 32;
  constexpr int WARPS_N_C = 2;
  constexpr int WARPS_M_C = WARP_COUNT / WARPS_N_C;
  constexpr int WARP_TILE_M = BM / WARPS_M_C; // 64 for both tiles
  constexpr int WARP_TILE_N = BN / WARPS_N_C; // 64 for BN=128
  constexpr int MMA_M = WARP_TILE_M / 16;     // 4
  constexpr int MMA_N = WARP_TILE_N / 8;      // 8

  // Global-to-shared load parameters
  constexpr int A_LOAD_ITERS = BK / 8;
  constexpr int B_THREADS_PER_ROW = THREADS / BK;
  constexpr int B_LOAD_ITERS = BN / (B_THREADS_PER_ROW * 8);

  const int batch = (int)blockIdx.z;
  if (batch >= batchCount)
    return;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  const int swizzle_factor = 8;
  if (gridDim.y % swizzle_factor == 0) {
    const int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
    by = (block_idx % swizzle_factor) +
         (block_idx / (gridDim.x * swizzle_factor)) * swizzle_factor;
    bx = (block_idx / swizzle_factor) % gridDim.x;
  }

  if (by * BM >= M || bx * BN >= N)
    return;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  const int wy = wid >> 1; // wid / WARPS_N_C  (WARPS_N_C==2)
  const int wx = wid & 1;  // wid % WARPS_N_C

  const __nv_bfloat16 *A_batch = A + (long long)batch * strideA;
  const __nv_bfloat16 *B_batch = B + (long long)batch * strideB;
  __nv_bfloat16 *C_batch = C + (long long)batch * strideC;

  extern __shared__ __nv_bfloat16 s_mem[];

  float acc[MMA_M][MMA_N][4];
#pragma unroll
  for (int i = 0; i < MMA_M; i++)
#pragma unroll
    for (int j = 0; j < MMA_N; j++)
      acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  // Global-to-shared async pipeline
  auto load_to_stage = [&](int stage, int k_curr) {
    // A tile (BM x BK): A_THREADS_PER_ROW threads cover one BM row
    constexpr int A_THREADS_PER_ROW = THREADS / BM > 0 ? THREADS / BM : 1;
    constexpr int A_ROWS_PER_ITER = THREADS / BM > 0 ? 1 : BM / THREADS;
    constexpr int A_LOAD_ITERS = BK / (A_THREADS_PER_ROW * 8);

#pragma unroll
    for (int i = 0; i < A_LOAD_ITERS; i++) {
#pragma unroll
      for (int j = 0; j < A_ROWS_PER_ITER; j++) {
        int r = (tid / A_THREADS_PER_ROW) + j * THREADS;
        int c = (tid % A_THREADS_PER_ROW) * 8 + i * A_THREADS_PER_ROW * 8;
        const __nv_bfloat16 *g_ptr =
            A_batch + (long long)(by * BM + r) * lda + (k_curr + c);
        int swizzled_c = c ^ ((r & 3) << 3);
        uint32_t sm_addr = __cvta_generic_to_shared(
            &s_mem[stage * STAGE_SIZE + r * BK + swizzled_c]);
        int pred = (by * BM + r < M && k_curr + c < K) ? 16 : 0;
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sm_addr),
            "l"(g_ptr), "r"(pred));
      }
    }
// B tile (BK x BN): B_LOAD_ITERS iterations cover all BN columns
#pragma unroll
    for (int i = 0; i < B_LOAD_ITERS; i++) {
      int r = tid / B_THREADS_PER_ROW;
      int c = (tid % B_THREADS_PER_ROW) * 8 + i * B_THREADS_PER_ROW * 8;
      const __nv_bfloat16 *g_ptr =
          B_batch + (long long)(k_curr + r) * ldb + (bx * BN + c);
      int swizzled_c = c ^ ((r & 7) << 3);
      uint32_t sm_addr = __cvta_generic_to_shared(
          &s_mem[stage * STAGE_SIZE + AS_SIZE + r * BN + swizzled_c]);
      int pred = (k_curr + r < K && bx * BN + c < N) ? 16 : 0;
      asm volatile(
          "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sm_addr),
          "l"(g_ptr), "r"(pred));
    }
  };

  // Prime 2 stages ahead
  load_to_stage(0, 0);
  asm volatile("cp.async.commit_group;\n");
  if (BK < K)
    load_to_stage(1, BK);
  asm volatile("cp.async.commit_group;\n");

  int write_stage = 2, read_stage = 0;
  uint32_t frA[2][MMA_M][4], frB[2][MMA_N][2];

  asm volatile("cp.async.wait_group 1;\n");
  __syncthreads();

  // ldmatrix wrappers
  auto load_frA = [&](uint32_t reg[4], int ki, int mi, int st) {
    int r = wy * WARP_TILE_M + mi * 16 + ((lane / 8) % 2) * 8 + (lane % 8);
    int c = ki + (lane / 16) * 8;
    int swizzled_c = c ^ ((r & 3) << 3);
    uint32_t addr =
        __cvta_generic_to_shared(&s_mem[st * STAGE_SIZE + r * BK + swizzled_c]);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(addr));
  };

  auto load_frB = [&](uint32_t reg[2], int ki, int ni, int st) {
    int r = ki + ((lane / 8) % 2) * 8 + (lane % 8);
    int c = wx * WARP_TILE_N + ni * 8;
    int swizzled_c = c ^ ((r & 7) << 3);
    uint32_t addr = __cvta_generic_to_shared(
        &s_mem[st * STAGE_SIZE + AS_SIZE + r * BN + swizzled_c]);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(reg[0]), "=r"(reg[1])
        : "r"(addr));
  };

// Initial register prefetch from stage 0
#pragma unroll
  for (int i = 0; i < MMA_M; i++)
    load_frA(frA[0][i], 0, i, read_stage);
#pragma unroll
  for (int j = 0; j < MMA_N; j++)
    load_frB(frB[0][j], 0, j, read_stage);

  // Main K loop
  for (int k = 0; k < K; k += BK) {
    if (k + 2 * BK < K)
      load_to_stage(write_stage, k + 2 * BK);
    asm volatile("cp.async.commit_group;\n");

#pragma unroll
    for (int ks = 0; ks < BK; ks += 16) {
      int p = (ks >> 4) & 1, q = p ^ 1;
#pragma unroll
      for (int i = 0; i < MMA_M; i++) {
#pragma unroll
        for (int j = 0; j < MMA_N; j++) {
          MMA_M16N8K16_F32(acc[i][j][0], acc[i][j][1], acc[i][j][2],
                           acc[i][j][3], frA[p][i][0], frA[p][i][1],
                           frA[p][i][2], frA[p][i][3], frB[p][j][0],
                           frB[p][j][1]);
        }

        constexpr int b_loads_per_a = (MMA_N + MMA_M - 1) / MMA_M;
        if (ks + 16 < BK) {
          load_frA(frA[q][i], ks + 16, i, read_stage);
#pragma unroll
          for (int b_idx = 0; b_idx < b_loads_per_a; b_idx++) {
            int j_actual = i * b_loads_per_a + b_idx;
            if (j_actual < MMA_N)
              load_frB(frB[q][j_actual], ks + 16, j_actual, read_stage);
          }
        } else if (k + BK < K) {
          if (i == 0) {
            asm volatile("cp.async.wait_group 1;\n");
            __syncthreads();
            read_stage = (read_stage + 1) % STAGES;
            write_stage = (write_stage + 1) % STAGES;
          }
          load_frA(frA[q][i], 0, i, read_stage);
#pragma unroll
          for (int b_idx = 0; b_idx < b_loads_per_a; b_idx++) {
            int j_actual = i * b_loads_per_a + b_idx;
            if (j_actual < MMA_N)
              load_frB(frB[q][j_actual], 0, j_actual, read_stage);
          }
        }
      }
    }
  }

  asm volatile("cp.async.wait_group 0;\n");
  __syncthreads();

  // Vectorized epilogue: C = alpha * acc + beta * bias
  const int g = lane / 4, t = lane % 4;
  float alpha_f = __bfloat162float(alpha);
  float beta_f = __bfloat162float(beta);

#pragma unroll
  for (int j = 0; j < MMA_N; j++) {
#pragma unroll
    for (int i = 0; i < MMA_M; i++) {
      float a0 = acc[i][j][0] * alpha_f, a1 = acc[i][j][1] * alpha_f;
      float a2 = acc[i][j][2] * alpha_f, a3 = acc[i][j][3] * alpha_f;

      float n0 = __shfl_xor_sync(0xffffffff, a0, 1);
      float n1 = __shfl_xor_sync(0xffffffff, a1, 1);
      float n2 = __shfl_xor_sync(0xffffffff, a2, 1);
      float n3 = __shfl_xor_sync(0xffffffff, a3, 1);

      float4 f4_r0, f4_r8;
      if (t % 2 == 0) {
        f4_r0 = {a0, a1, n0, n1};
        f4_r8 = {a2, a3, n2, n3};
      } else {
        f4_r0 = {n0, n1, a0, a1};
        f4_r8 = {n2, n3, a2, a3};
      }

      if (t % 2 == 0) {
        const int r0 = by * BM + wy * WARP_TILE_M + i * 16 + g;
        const int r8 = r0 + 8;
        const int c = bx * BN + wx * WARP_TILE_N + j * 8 + (t / 2) * 4;

        float4 bv0 = {0, 0, 0, 0}, bv8 = {0, 0, 0, 0};
        if (bias) {
          if (bias_numel == 1) {
            float b = __bfloat162float(bias[0]);
            bv0 = bv8 = {b, b, b, b};
          } else if (bias_numel == N) {
            if (c < N) {
              if (c + 3 < N && (((size_t)(&bias[c]) & 7) == 0)) {
                uint64_t b_vec = *(uint64_t *)&bias[c];
                __nv_bfloat16 *bh = (__nv_bfloat16 *)&b_vec;
                bv0 = {__bfloat162float(bh[0]), __bfloat162float(bh[1]),
                       __bfloat162float(bh[2]), __bfloat162float(bh[3])};
                bv8 = bv0;
              } else {
                bv0.x = __bfloat162float(bias[c]);
                if (c + 1 < N)
                  bv0.y = __bfloat162float(bias[c + 1]);
                if (c + 2 < N)
                  bv0.z = __bfloat162float(bias[c + 2]);
                if (c + 3 < N)
                  bv0.w = __bfloat162float(bias[c + 3]);
                bv8 = bv0;
              }
            }
          } else if (bias_numel == (int64_t)M * N) {
            const int64_t idx0 = (int64_t)r0 * N + c;
            const int64_t idx8 = (int64_t)r8 * N + c;
            if (r0 < M && c < N) {
              bv0.x = __bfloat162float(bias[idx0]);
              if (c + 1 < N)
                bv0.y = __bfloat162float(bias[idx0 + 1]);
              if (c + 2 < N)
                bv0.z = __bfloat162float(bias[idx0 + 2]);
              if (c + 3 < N)
                bv0.w = __bfloat162float(bias[idx0 + 3]);
            }
            if (r8 < M && c < N) {
              bv8.x = __bfloat162float(bias[idx8]);
              if (c + 1 < N)
                bv8.y = __bfloat162float(bias[idx8 + 1]);
              if (c + 2 < N)
                bv8.z = __bfloat162float(bias[idx8 + 2]);
              if (c + 3 < N)
                bv8.w = __bfloat162float(bias[idx8 + 3]);
            }
          }
        }

        f4_r0.x += beta_f * bv0.x;
        f4_r0.y += beta_f * bv0.y;
        f4_r0.z += beta_f * bv0.z;
        f4_r0.w += beta_f * bv0.w;
        f4_r8.x += beta_f * bv8.x;
        f4_r8.y += beta_f * bv8.y;
        f4_r8.z += beta_f * bv8.z;
        f4_r8.w += beta_f * bv8.w;

        auto sb4 = [&](int r, int cl, float4 v) {
          if (r < M && cl < N) {
            __nv_bfloat16 *p = C_batch + (long long)r * ldc + cl;
            if (cl + 3 < N && (((size_t)p & 7) == 0)) {
              __nv_bfloat162 b01 = __float22bfloat162_rn({v.x, v.y});
              __nv_bfloat162 b23 = __float22bfloat162_rn({v.z, v.w});
              uint32_t i01 = reinterpret_cast<const uint32_t &>(b01);
              uint32_t i23 = reinterpret_cast<const uint32_t &>(b23);
              *reinterpret_cast<uint64_t *>(p) =
                  (static_cast<uint64_t>(i23) << 32) | i01;
            } else {
              p[0] = __float2bfloat16(v.x);
              if (cl + 1 < N)
                p[1] = __float2bfloat16(v.y);
              if (cl + 2 < N)
                p[2] = __float2bfloat16(v.z);
              if (cl + 3 < N)
                p[3] = __float2bfloat16(v.w);
            }
          }
        };
        sb4(r0, c, f4_r0);
        sb4(r8, c, f4_r8);
      }
    }
  }
}

template __global__ void bgemm_addmm_kernel<256, 128, 32, 4, 256, true>(int,int,int,__nv_bfloat16,const __nv_bfloat16*,int,long long,const __nv_bfloat16*,int,long long,__nv_bfloat16,const __nv_bfloat16*,int64_t,__nv_bfloat16*,int,long long,int);
template __global__ void bgemm_addmm_kernel<128, 128, 32, 4, 256, true>(int,int,int,__nv_bfloat16,const __nv_bfloat16*,int,long long,const __nv_bfloat16*,int,long long,__nv_bfloat16,const __nv_bfloat16*,int64_t,__nv_bfloat16*,int,long long,int);
template __global__ void bgemm_addmm_kernel<256, 128, 32, 3, 256, true>(int,int,int,__nv_bfloat16,const __nv_bfloat16*,int,long long,const __nv_bfloat16*,int,long long,__nv_bfloat16,const __nv_bfloat16*,int64_t,__nv_bfloat16*,int,long long,int);
