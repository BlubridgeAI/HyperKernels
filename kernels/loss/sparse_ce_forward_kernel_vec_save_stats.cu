// sparse_ce_forward_kernel_vec_save_stats.cu
// Sparse cross-entropy forward (one block per row): online softmax, saves per-row (max,sum) stats for backward reuse.

#include <cuda_runtime.h>
#include <cstdint>

// 128-bit cp.async global->shared, .cg bypasses L1
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* glob_ptr) {
    unsigned int smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 : : "r"(smem_addr), "l"(glob_ptr));
}

template<typename T, typename T_idx>
__global__ void sparse_ce_forward_kernel_vec_save_stats(
    const T* logits,
    const T_idx* targets,
    T* losses,
    float* saved_max,
    float* saved_sum,
    int64_t batch_size,
    int64_t vocab_size
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    extern __shared__ float s_data[];
    int64_t row = blockIdx.x;
    if (row >= batch_size) return;

    const T* row_logits = logits + row * vocab_size;
    float local_max = -1e38f;
    float local_sum = 0.0f;

    const int64_t vec_size = 4;
    const int64_t vec_count = vocab_size / vec_size;
    if ((reinterpret_cast<uintptr_t>(row_logits) & 0xF) == 0) {
        for (int64_t j = threadIdx.x * 4; j < vocab_size; j += blockDim.x * 4) {
            cp_async_16(&s_data[tid * 4], &row_logits[j]);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            // No __syncthreads() here: each thread reads only its own slot; a barrier
            // in this variable-iteration loop would deadlock when vocab_size % (bdim*4) != 0.
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float val = s_data[tid * 4 + k];
                if (val > local_max) {
                    local_sum = local_sum * expf(local_max - val) + 1.0f;
                    local_max = val;
                } else {
                    local_sum += expf(val - local_max);
                }
            }
        }
    } else {
        for (int64_t j = threadIdx.x; j < vec_count * vec_size; j += blockDim.x) {
            float val = static_cast<float>(row_logits[j]);
            if (val > local_max) {
                local_sum = local_sum * expf(local_max - val) + 1.0f;
                local_max = val;
            } else {
                local_sum += expf(val - local_max);
            }
        }
    }
    for (int64_t j = vec_count * vec_size + threadIdx.x; j < vocab_size; j += blockDim.x) {
        float val = static_cast<float>(row_logits[j]);
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }

    // smax/ssum placed after the bdim*4 staging buffer to avoid aliasing the in-flight cp.async region.
    float* smax = s_data + bdim * 4;
    float* ssum = s_data + bdim * 4 + bdim;
    smax[tid] = local_max;
    ssum[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other_max = smax[tid + s];
            float other_sum = ssum[tid + s];
            if (other_max > smax[tid]) {
                ssum[tid] = ssum[tid] * expf(smax[tid] - other_max) + other_sum;
                smax[tid] = other_max;
            } else {
                ssum[tid] += other_sum * expf(other_max - smax[tid]);
            }
        }
        __syncthreads();
    }
    float final_max = smax[0];
    float final_sum = ssum[0];

    if (tid == 0) {
        saved_max[row] = final_max;
        saved_sum[row] = final_sum;
        int64_t target_idx = static_cast<int64_t>(targets[row]);
        float target_logit = static_cast<float>(row_logits[target_idx]);
        float loss = logf(final_sum) + final_max - target_logit;
        losses[row] = static_cast<T>(loss);
    }
}

template __global__ void sparse_ce_forward_kernel_vec_save_stats<float, int32_t>(
    const float*, const int32_t*, float*, float*, float*, int64_t, int64_t);
template __global__ void sparse_ce_forward_kernel_vec_save_stats<float, int64_t>(
    const float*, const int64_t*, float*, float*, float*, int64_t, int64_t);
template __global__ void sparse_ce_forward_kernel_vec_save_stats<float, uint16_t>(
    const float*, const uint16_t*, float*, float*, float*, int64_t, int64_t);

