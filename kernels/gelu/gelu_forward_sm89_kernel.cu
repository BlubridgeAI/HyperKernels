// gelu_forward_sm89_kernel.cu
// GELU forward (tanh approximation), float4 vectorized for fp32 and half2/bf162 for 16-bit; math in fp32.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <cstdint>

__device__ __forceinline__ float fast_tanh_sm89(float x) {
    float res;
    asm("tanh.approx.f32 %0, %1;" : "=f"(res) : "f"(x));
    return res;
}

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeLUCoef    = 0.044715f;

__device__ __forceinline__ float gelu_fwd_f32(float x) {
    float inner = kSqrt2OverPi * (x + kGeLUCoef * x * x * x);
    return 0.5f * x * (1.0f + fast_tanh_sm89(inner));
}

// Scalar fp32 conversions for the tail loop.
template<typename T> __device__ __forceinline__ float to_f(T v);
template<> __device__ __forceinline__ float to_f(float v) { return v; }
template<> __device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
template<> __device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ __forceinline__ T from_f(float v);
template<> __device__ __forceinline__ float from_f(float v) { return v; }
template<> __device__ __forceinline__ __half from_f(float v) { return __float2half(v); }
template<> __device__ __forceinline__ __nv_bfloat16 from_f(float v) { return __float2bfloat16(v); }

template<typename T>
__global__ __launch_bounds__(512)
void gelu_forward_sm89_kernel(const T* __restrict__ in, T* __restrict__ out, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    if constexpr (std::is_same_v<T, float>) {
        int64_t numel4 = numel / 4;
        #pragma unroll 8
        for (int64_t i = idx; i < numel4; i += stride) {
            float4 x = reinterpret_cast<const float4*>(in)[i];
            reinterpret_cast<float4*>(out)[i] = {
                gelu_fwd_f32(x.x), gelu_fwd_f32(x.y),
                gelu_fwd_f32(x.z), gelu_fwd_f32(x.w)};
        }
        for (int64_t i = numel4*4+idx; i < numel; i += stride)
            out[i] = gelu_fwd_f32(in[i]);
    } else if constexpr (std::is_same_v<T, __half>) {
        // half2 vectorized: load 2, compute in fp32, store 2.
        int64_t numel2 = numel / 2;
        #pragma unroll 8
        for (int64_t i = idx; i < numel2; i += stride) {
            __half2 h = reinterpret_cast<const __half2*>(in)[i];
            float2 f = __half22float2(h);
            f.x = gelu_fwd_f32(f.x); f.y = gelu_fwd_f32(f.y);
            reinterpret_cast<__half2*>(out)[i] = __float22half2_rn(f);
        }
        for (int64_t i = numel2*2+idx; i < numel; i += stride)
            out[i] = from_f<T>(gelu_fwd_f32(to_f(in[i])));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        int64_t numel2 = numel / 2;
        #pragma unroll 8
        for (int64_t i = idx; i < numel2; i += stride) {
            __nv_bfloat162 h = reinterpret_cast<const __nv_bfloat162*>(in)[i];
            float2 f = __bfloat1622float2(h);
            f.x = gelu_fwd_f32(f.x); f.y = gelu_fwd_f32(f.y);
            reinterpret_cast<__nv_bfloat162*>(out)[i] = __float22bfloat162_rn(f);
        }
        for (int64_t i = numel2*2+idx; i < numel; i += stride)
            out[i] = from_f<T>(gelu_fwd_f32(to_f(in[i])));
    }
}

template __global__ void gelu_forward_sm89_kernel<float>(const float*, float*, int64_t);
template __global__ void gelu_forward_sm89_kernel<__half>(const __half*, __half*, int64_t);
template __global__ void gelu_forward_sm89_kernel<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t);

