// ln_backward_input_sm89_kernel.cu
// LayerNorm backward w.r.t input: computes grad_x from grad_y, x, per-row mean/rstd, optional gamma. fp32/fp16/bf16.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace OwnTensor {
namespace cuda {

// load: scalar dtype to float
template<typename T> __device__ __forceinline__ float to_float_sm89(T val);
template<> __device__ __forceinline__ float to_float_sm89(float val) { return val; }
template<> __device__ __forceinline__ float to_float_sm89(__half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float_sm89(__nv_bfloat16 val) { return __bfloat162float(val); }

// store: float to scalar dtype
template<typename T> __device__ __forceinline__ T from_float_sm89(float val);
template<> __device__ __forceinline__ float from_float_sm89(float val) { return val; }
template<> __device__ __forceinline__ __half from_float_sm89(float val) { return __float2half(val); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float_sm89(float val) { return __float2bfloat16(val); }

template<typename T>
__inline__ __device__ T warpReduceSum_sm89(T val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Input grad: float4 vectorized, 512 threads, all dtypes
template<typename T>
__global__ __launch_bounds__(512)
void ln_backward_input_sm89_kernel(
    const T* __restrict__ grad_y, const T* __restrict__ x,
    const float* __restrict__ mean, const float* __restrict__ rstd,
    const T* __restrict__ gamma, T* __restrict__ grad_x, int cols)
{
    int row = blockIdx.x, tid = threadIdx.x;
    const T* dy_row = grad_y + row*cols;
    const T* x_row = x + row*cols;
    T* dx_row = grad_x + row*cols;
    float m = mean[row], rs = rstd[row];

    float sum1 = 0, sum2 = 0;
    if constexpr (std::is_same_v<T, float>) {
        const float4* dyv = reinterpret_cast<const float4*>(dy_row);
        const float4* xv  = reinterpret_cast<const float4*>(x_row);
        const float4* gv  = gamma ? reinterpret_cast<const float4*>(gamma) : nullptr;
        const int vc = cols/4;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 d = dyv[i], xi = xv[i], g = gv ? gv[i] : make_float4(1,1,1,1);
            float nx=(xi.x-m)*rs, ny=(xi.y-m)*rs, nz=(xi.z-m)*rs, nw=(xi.w-m)*rs;
            float a=d.x*g.x, b=d.y*g.y, c=d.z*g.z, e=d.w*g.w;
            sum1 += a+b+c+e; sum2 += a*nx+b*ny+c*nz+e*nw;
        }
        for (int i = vc*4+tid; i < cols; i += blockDim.x) {
            float g = gamma ? gamma[i] : 1.0f, dy = dy_row[i], nx = (x_row[i]-m)*rs;
            sum1 += dy*g; sum2 += dy*g*nx;
        }
    } else {
        #pragma unroll 4
        for (int i = tid; i < cols; i += blockDim.x) {
            float g = gamma ? to_float_sm89(gamma[i]) : 1.0f;
            float dy = to_float_sm89(dy_row[i]), nx = (to_float_sm89(x_row[i])-m)*rs;
            sum1 += dy*g; sum2 += dy*g*nx;
        }
    }

    sum1 = warpReduceSum_sm89(sum1); sum2 = warpReduceSum_sm89(sum2);
    __shared__ float s1, s2;
    if (tid == 0) { s1 = 0; s2 = 0; }
    __syncthreads();
    if (tid % warpSize == 0) { atomicAdd(&s1, sum1); atomicAdd(&s2, sum2); }
    __syncthreads();
    float t1 = s1, t2 = s2, ic = 1.0f/cols;

    if constexpr (std::is_same_v<T, float>) {
        float4* dxv = reinterpret_cast<float4*>(dx_row);
        const float4* dyv = reinterpret_cast<const float4*>(dy_row);
        const float4* xv  = reinterpret_cast<const float4*>(x_row);
        const float4* gv  = gamma ? reinterpret_cast<const float4*>(gamma) : nullptr;
        const int vc = cols/4;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 d = dyv[i], xi = xv[i], g = gv ? gv[i] : make_float4(1,1,1,1);
            float nx=(xi.x-m)*rs, ny=(xi.y-m)*rs, nz=(xi.z-m)*rs, nw=(xi.w-m)*rs;
            dxv[i] = {rs*(d.x*g.x-(t1+nx*t2)*ic), rs*(d.y*g.y-(t1+ny*t2)*ic),
                      rs*(d.z*g.z-(t1+nz*t2)*ic), rs*(d.w*g.w-(t1+nw*t2)*ic)};
        }
        for (int i = vc*4+tid; i < cols; i += blockDim.x) {
            float g = gamma ? gamma[i] : 1.0f, nx = (x_row[i]-m)*rs;
            dx_row[i] = rs*(dy_row[i]*g-(t1+nx*t2)*ic);
        }
    } else {
        #pragma unroll 4
        for (int i = tid; i < cols; i += blockDim.x) {
            float g = gamma ? to_float_sm89(gamma[i]) : 1.0f;
            float dy = to_float_sm89(dy_row[i]), nx = (to_float_sm89(x_row[i])-m)*rs;
            dx_row[i] = from_float_sm89<T>(rs*(dy*g-(t1+nx*t2)*ic));
        }
    }
}

template __global__ void ln_backward_input_sm89_kernel<float>(
    const float*, const float*, const float*, const float*, const float*, float*, int);
template __global__ void ln_backward_input_sm89_kernel<__half>(
    const __half*, const __half*, const float*, const float*, const __half*, __half*, int);
template __global__ void ln_backward_input_sm89_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*,
    const __nv_bfloat16*, __nv_bfloat16*, int);

} // namespace cuda
} // namespace OwnTensor
