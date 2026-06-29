// ln_backward_gamma_beta_sm89_kernel.cu
// LayerNorm backward for gamma/beta grads: grad_gamma = sum(grad_y * norm_x), grad_beta = sum(grad_y), via atomicAdd.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// dtype-to-float conversion helper
template<typename T> __device__ __forceinline__ float to_float_sm89(T val);
template<> __device__ __forceinline__ float to_float_sm89(float val) { return val; }
template<> __device__ __forceinline__ float to_float_sm89(__half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float_sm89(__nv_bfloat16 val) { return __bfloat162float(val); }

// Gamma/beta gradient kernel, templated over dtype.
template<typename T>
__global__ __launch_bounds__(256)
void ln_backward_gamma_beta_sm89_kernel(
    const T* __restrict__ grad_y, const T* __restrict__ x,
    const float* __restrict__ mean, const float* __restrict__ rstd,
    float* __restrict__ grad_gamma, float* __restrict__ grad_beta,
    int rows, int cols)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    __shared__ float s_dg[8][32], s_db[8][32];

    #pragma unroll 4
    for (int cb = blockIdx.x * 32; cb < cols; cb += gridDim.x * 32) {
        int col = cb + tx;
        float dg = 0, db = 0;
        if (col < cols) {
            for (int r = blockIdx.y * 8 + ty; r < rows; r += gridDim.y * 8) {
                float gy = to_float_sm89(grad_y[r*cols+col]);
                float v  = to_float_sm89(x[r*cols+col]);
                float nx = (v - mean[r]) * rstd[r];
                db += gy; dg += gy * nx;
            }
        }
        s_dg[ty][tx] = dg; s_db[ty][tx] = db;
        __syncthreads();
        if (ty == 0 && col < cols) {
            float fg = 0, fb = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) { fg += s_dg[i][tx]; fb += s_db[i][tx]; }
            atomicAdd(&grad_gamma[col], fg);
            atomicAdd(&grad_beta[col], fb);
        }
        __syncthreads();
    }
}

template __global__ void ln_backward_gamma_beta_sm89_kernel<float>(
    const float*, const float*, const float*, const float*, float*, float*, int, int);
template __global__ void ln_backward_gamma_beta_sm89_kernel<__half>(
    const __half*, const __half*, const float*, const float*, float*, float*, int, int);
template __global__ void ln_backward_gamma_beta_sm89_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, float*, float*, int, int);

