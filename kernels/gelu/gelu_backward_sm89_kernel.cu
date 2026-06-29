// gelu_backward_sm89_kernel.cu
// GELU backward (tanh approximation): grad_in = grad * dGELU/dx. float4 for fp32, half2/bf162 for 16-bit.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <cstdint>

namespace OwnTensor {
namespace cuda {

// Hardware tanh approximation (PTX).
__device__ __forceinline__ float fast_tanh_sm89(float x) {
    float res;
    asm("tanh.approx.f32 %0, %1;" : "=f"(res) : "f"(x));
    return res;
}

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeLUCoef    = 0.044715f;

// dGELU/dx (tanh approximation).
__device__ __forceinline__ float gelu_bwd_f32(float x) {
    float x2 = x * x;
    float u  = kSqrt2OverPi * (x + kGeLUCoef * x2 * x);
    float du = kSqrt2OverPi * (1.0f + 3.0f * kGeLUCoef * x2);
    float th = fast_tanh_sm89(u);
    return 0.5f * (1.0f + th) + 0.5f * x * (1.0f - th * th) * du;
}

// Type helpers: convert dtype to/from fp32.
template<typename T> __device__ __forceinline__ float to_f(T v);
template<> __device__ __forceinline__ float to_f(float v) { return v; }
template<> __device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
template<> __device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ __forceinline__ T from_f(float v);
template<> __device__ __forceinline__ float from_f(float v) { return v; }
template<> __device__ __forceinline__ __half from_f(float v) { return __float2half(v); }
template<> __device__ __forceinline__ __nv_bfloat16 from_f(float v) { return __float2bfloat16(v); }

// Backward: float4 for fp32, half2/bf162 for 16-bit.
template<typename T>
__global__ __launch_bounds__(512)
void gelu_backward_sm89_kernel(const T* __restrict__ grad, const T* __restrict__ in,
                                T* __restrict__ grad_in, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    if constexpr (std::is_same_v<T, float>) {
        int64_t numel4 = numel / 4;
        #pragma unroll 8
        for (int64_t i = idx; i < numel4; i += stride) {
            float4 g = reinterpret_cast<const float4*>(grad)[i];
            float4 x = reinterpret_cast<const float4*>(in)[i];
            reinterpret_cast<float4*>(grad_in)[i] = {
                g.x * gelu_bwd_f32(x.x), g.y * gelu_bwd_f32(x.y),
                g.z * gelu_bwd_f32(x.z), g.w * gelu_bwd_f32(x.w)};
        }
        for (int64_t i = numel4*4+idx; i < numel; i += stride)
            grad_in[i] = grad[i] * gelu_bwd_f32(in[i]);
    } else if constexpr (std::is_same_v<T, __half>) {
        int64_t numel2 = numel / 2;
        #pragma unroll 8
        for (int64_t i = idx; i < numel2; i += stride) {
            float2 gf = __half22float2(reinterpret_cast<const __half2*>(grad)[i]);
            float2 xf = __half22float2(reinterpret_cast<const __half2*>(in)[i]);
            float2 res = {gf.x * gelu_bwd_f32(xf.x), gf.y * gelu_bwd_f32(xf.y)};
            reinterpret_cast<__half2*>(grad_in)[i] = __float22half2_rn(res);
        }
        for (int64_t i = numel2*2+idx; i < numel; i += stride)
            grad_in[i] = from_f<T>(to_f(grad[i]) * gelu_bwd_f32(to_f(in[i])));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        int64_t numel2 = numel / 2;
        #pragma unroll 8
        for (int64_t i = idx; i < numel2; i += stride) {
            float2 gf = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(grad)[i]);
            float2 xf = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(in)[i]);
            float2 res = {gf.x * gelu_bwd_f32(xf.x), gf.y * gelu_bwd_f32(xf.y)};
            reinterpret_cast<__nv_bfloat162*>(grad_in)[i] = __float22bfloat162_rn(res);
        }
        for (int64_t i = numel2*2+idx; i < numel; i += stride)
            grad_in[i] = from_f<T>(to_f(grad[i]) * gelu_bwd_f32(to_f(in[i])));
    }
}

template __global__ void gelu_backward_sm89_kernel<float>(const float*, const float*, float*, int64_t);
template __global__ void gelu_backward_sm89_kernel<__half>(const __half*, const __half*, __half*, int64_t);
template __global__ void gelu_backward_sm89_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int64_t);

} // namespace cuda
} // namespace OwnTensor
