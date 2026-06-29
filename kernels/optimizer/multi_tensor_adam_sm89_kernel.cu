// multi_tensor_adam_sm89_kernel.cu
// Fused float4-vectorized Adam/AdamW step over many tensors in one launch via by-value chunk metadata.

#include <cuda_runtime.h>
#include <cstdint>

// Float elements processed per block.
static const int CHUNK_SIZE = 32768;
static const int MAX_BLOCKS_PER_LAUNCH = 320;
static const int MAX_TENSORS_PER_LAUNCH = 48;

// Per-launch tensor pointers, sizes, and block-to-(tensor,chunk) mapping passed by value.
struct AdamLaunchMetadata {
    float* params[MAX_TENSORS_PER_LAUNCH];
    float* grads[MAX_TENSORS_PER_LAUNCH];
    float* ms[MAX_TENSORS_PER_LAUNCH];
    float* vs[MAX_TENSORS_PER_LAUNCH];
    int64_t sizes[MAX_TENSORS_PER_LAUNCH];
    unsigned char block_to_tensor[MAX_BLOCKS_PER_LAUNCH];
    int block_to_chunk[MAX_BLOCKS_PER_LAUNCH];
};

__global__ void __launch_bounds__(256, 2) multi_tensor_adam_sm89_kernel(
    AdamLaunchMetadata meta,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2
) {
    int loc_block_idx = blockIdx.x;
    if (loc_block_idx >= MAX_BLOCKS_PER_LAUNCH) return;

    int tensor_idx = meta.block_to_tensor[loc_block_idx];
    int chunk_idx = meta.block_to_chunk[loc_block_idx];
    int64_t numel = meta.sizes[tensor_idx];

    int64_t start = (int64_t)chunk_idx * CHUNK_SIZE;
    int64_t end = start + CHUNK_SIZE;
    if (end > numel) end = numel;

    float* p = meta.params[tensor_idx];
    float* g = meta.grads[tensor_idx];
    float* m = meta.ms[tensor_idx];
    float* v = meta.vs[tensor_idx];

    // float4-aligned vector boundaries
    int64_t vec_start = (start + 3) / 4 * 4;
    int64_t vec_end   = end / 4 * 4;
    if (vec_start > vec_end) vec_start = vec_end;

    // scalar head
    for (int64_t i = start + threadIdx.x; i < vec_start; i += blockDim.x) {
        float gi = g[i];
        float pi = p[i];
        float mi = m[i];
        float vi = v[i];

        float m_new = fmaf(beta1, mi, (1.0f - beta1) * gi);
        float v_new = fmaf(beta2, vi, (1.0f - beta2) * gi * gi);
        m[i] = m_new;
        v[i] = v_new;
        // eps must stay inside the denominator: when v_hat=0, denom=eps gives
        // update=0 instead of the NaN from 0*Inf.
        float denom = sqrtf(v_new / bias_correction2) + eps;
        p[i] = pi - lr * ((m_new / bias_correction1) / denom + weight_decay * pi);
    }

    // Vectorized main loop
    for (int64_t i = vec_start + threadIdx.x * 4; i < vec_end; i += (int64_t)blockDim.x * 4) {
        float4* g4 = (float4*)(&g[i]);
        float4* p4 = (float4*)(&p[i]);
        float4* m4 = (float4*)(&m[i]);
        float4* v4 = (float4*)(&v[i]);

        float4 g_vec = *g4;
        float4 p_vec = *p4;
        float4 m_vec = *m4;
        float4 v_vec = *v4;

        float4 m_out, v_out, p_out;

#define ADAM_UPDATE(gj, pj, mj, vj, m_out_j, v_out_j, p_out_j)          \
        m_out_j = fmaf(beta1, mj, (1.0f - beta1) * gj);                 \
        v_out_j = fmaf(beta2, vj, (1.0f - beta2) * gj * gj);            \
        p_out_j = pj - lr * ((m_out_j / bias_correction1) /             \
                  (sqrtf(v_out_j / bias_correction2) + eps)             \
                  + weight_decay * pj);

        ADAM_UPDATE(g_vec.x, p_vec.x, m_vec.x, v_vec.x, m_out.x, v_out.x, p_out.x)
        ADAM_UPDATE(g_vec.y, p_vec.y, m_vec.y, v_vec.y, m_out.y, v_out.y, p_out.y)
        ADAM_UPDATE(g_vec.z, p_vec.z, m_vec.z, v_vec.z, m_out.z, v_out.z, p_out.z)
        ADAM_UPDATE(g_vec.w, p_vec.w, m_vec.w, v_vec.w, m_out.w, v_out.w, p_out.w)
#undef ADAM_UPDATE

        *m4 = m_out;
        *v4 = v_out;
        *p4 = p_out;
    }

    // Scalar tail
    int64_t tail_start = max(start, vec_end);
    for (int64_t i = tail_start + threadIdx.x; i < end; i += blockDim.x) {
        float gi = g[i];
        float pi = p[i];
        float mi = m[i];
        float vi = v[i];

        float m_new = fmaf(beta1, mi, (1.0f - beta1) * gi);
        float v_new = fmaf(beta2, vi, (1.0f - beta2) * gi * gi);
        m[i] = m_new;
        v[i] = v_new;
        float denom = sqrtf(v_new / bias_correction2) + eps;
        p[i] = pi - lr * ((m_new / bias_correction1) / denom + weight_decay * pi);
    }
}
