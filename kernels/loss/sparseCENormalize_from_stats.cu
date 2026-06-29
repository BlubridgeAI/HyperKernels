// sparseCENormalize_from_stats.cu
// Sparse cross-entropy backward: grad = (softmax(logits) - target) * scale, from saved per-row (max,sum) stats. One block per row.

#include <cuda_runtime.h>
#include <cstdint>

template<typename T, typename T_idx>
__global__ void sparseCENormalize_from_stats(
    const T* __restrict__ logits,
    const T_idx* __restrict__ targets,
    T* __restrict__ grad,
    const float* __restrict__ saved_max,
    const float* __restrict__ saved_sum,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale)
{
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const T* row_logits = logits + row * vocab_size;
    T* row_grad = grad + row * vocab_size;
    int64_t target_idx = static_cast<int64_t>(targets[row]);

    const float final_max = saved_max[row];
    const float inv_sum   = 1.0f / saved_sum[row];
    const float f_scale   = static_cast<float>(*grad_output) * host_scale;

    // coalesced float4 loads, one 128B sector per warp per stride
    const int64_t vocab_vec4 = vocab_size / 4;  // assumes vocab_size % 4 == 0
    const float4* logits4 = reinterpret_cast<const float4*>(row_logits);
    float4*       grad4   = reinterpret_cast<float4*>(row_grad);

    for (int64_t v = tid; v < vocab_vec4; v += bdim) {
        float4 vec = logits4[v];
        int64_t col_base = v * 4;

        float4 out_grad;
        out_grad.x = (expf(vec.x - final_max) * inv_sum - (col_base + 0 == target_idx ? 1.0f : 0.0f)) * f_scale;
        out_grad.y = (expf(vec.y - final_max) * inv_sum - (col_base + 1 == target_idx ? 1.0f : 0.0f)) * f_scale;
        out_grad.z = (expf(vec.z - final_max) * inv_sum - (col_base + 2 == target_idx ? 1.0f : 0.0f)) * f_scale;
        out_grad.w = (expf(vec.w - final_max) * inv_sum - (col_base + 3 == target_idx ? 1.0f : 0.0f)) * f_scale;

        grad4[v] = out_grad;
    }
}

// T must be a 4-byte type (float) for the float4 reinterpret; T_idx is the target index dtype.
template __global__ void sparseCENormalize_from_stats<float, int32_t>(
    const float*, const int32_t*, float*, const float*, const float*,
    int64_t, int64_t, const float*, float);
template __global__ void sparseCENormalize_from_stats<float, int64_t>(
    const float*, const int64_t*, float*, const float*, const float*,
    int64_t, int64_t, const float*, float);
template __global__ void sparseCENormalize_from_stats<float, uint16_t>(
    const float*, const uint16_t*, float*, const float*, const float*,
    int64_t, int64_t, const float*, float);
