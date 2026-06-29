// layer_norm_forward_sm89_kernel.cu
// LayerNorm forward: Welford one-pass mean/var + float4 vectorized normalize. fp32/fp16/bf16.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>

// Welford state: count, mean, sum of squared deviations.
template<typename AccT>
struct WelfordData_sm89 { AccT n, mu, m2; };

// Stable merge of two Welford partials.
template<typename AccT>
__device__ __inline__ WelfordData_sm89<AccT> welford_merge_sm89(
    WelfordData_sm89<AccT> a, WelfordData_sm89<AccT> b) {
    if (a.n == 0) return b;
    if (b.n == 0) return a;
    WelfordData_sm89<AccT> res;
    res.n  = a.n + b.n;
    AccT delta = b.mu - a.mu;
    res.mu = a.mu + delta * (b.n / res.n);
    res.m2 = a.m2 + b.m2 + delta * delta * (a.n * b.n / res.n);
    return res;
}

template<typename T, typename AccT>
__global__ __launch_bounds__(512)
void layer_norm_forward_sm89_kernel(
    const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta,
    T* __restrict__ y, AccT* __restrict__ mean_out, AccT* __restrict__ rstd_out,
    int cols, AccT eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const T* row_x = x + row * cols;
    T* row_y = y + row * cols;

    AccT local_sum = 0, local_sq_sum = 0;
    int local_count = 0;

    if constexpr (std::is_same_v<T, float>) {
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vec_cols = cols / 4;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 v = x_vec[i];
            local_sum += v.x + v.y + v.z + v.w;
            local_sq_sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
            local_count += 4;
        }
        for (int i = vec_cols * 4 + tid; i < cols; i += blockDim.x) {
            AccT val = row_x[i];
            local_sum += val; local_sq_sum += val * val; local_count++;
        }
    } else if constexpr (std::is_same_v<T, __half>) {
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vec_cols = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 raw = x_vec[i];
            const __half2* h = reinterpret_cast<const __half2*>(&raw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 f = __half22float2(h[k]);
                local_sum += (AccT)f.x + (AccT)f.y;
                local_sq_sum += (AccT)f.x*(AccT)f.x + (AccT)f.y*(AccT)f.y;
            }
            local_count += 8;
        }
        for (int i = vec_cols * 8 + tid; i < cols; i += blockDim.x) {
            AccT val = (AccT)row_x[i];
            local_sum += val; local_sq_sum += val * val; local_count++;
        }
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vec_cols = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 raw = x_vec[i];
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 f = __bfloat1622float2(h[k]);
                local_sum += (AccT)f.x + (AccT)f.y;
                local_sq_sum += (AccT)f.x*(AccT)f.x + (AccT)f.y*(AccT)f.y;
            }
            local_count += 8;
        }
        for (int i = vec_cols * 8 + tid; i < cols; i += blockDim.x) {
            AccT val = (AccT)row_x[i];
            local_sum += val; local_sq_sum += val * val; local_count++;
        }
    }

    // Welford state
    AccT n = (AccT)local_count;
    WelfordData_sm89<AccT> state = {
        n,
        (n > 0) ? local_sum / n : (AccT)0,
        (n > 0) ? (local_sq_sum - local_sum * local_sum / n) : (AccT)0
    };

    // warp reduction
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        WelfordData_sm89<AccT> other;
        other.n  = __shfl_down_sync(0xffffffff, state.n,  offset);
        other.mu = __shfl_down_sync(0xffffffff, state.mu, offset);
        other.m2 = __shfl_down_sync(0xffffffff, state.m2, offset);
        state = welford_merge_sm89(state, other);
    }

    // block reduction
    __shared__ WelfordData_sm89<AccT> s_welford[16];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    if (lane_id == 0) s_welford[warp_id] = state;
    __syncthreads();

    if (tid == 0) {
        WelfordData_sm89<AccT> fs = s_welford[0];
        for (int i = 1; i < 16; ++i) fs = welford_merge_sm89(fs, s_welford[i]);
        s_welford[0] = fs;
    }
    __syncthreads();

    const AccT mu   = s_welford[0].mu;
    const AccT rstd = rsqrtf(s_welford[0].m2 / cols + eps);
    if (tid == 0) { if (mean_out) mean_out[row] = mu; if (rstd_out) rstd_out[row] = rstd; }

    // vectorized normalize
    if constexpr (std::is_same_v<T, float>) {
        float4* y_vec = reinterpret_cast<float4*>(row_y);
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vc = cols / 4;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 xv = x_vec[i];
            float4 gv = gamma ? reinterpret_cast<const float4*>(gamma)[i] : make_float4(1,1,1,1);
            float4 bv = beta  ? reinterpret_cast<const float4*>(beta)[i]  : make_float4(0,0,0,0);
            y_vec[i] = {(xv.x-mu)*rstd*gv.x+bv.x, (xv.y-mu)*rstd*gv.y+bv.y,
                        (xv.z-mu)*rstd*gv.z+bv.z, (xv.w-mu)*rstd*gv.w+bv.w};
        }
        for (int i = vc*4+tid; i < cols; i += blockDim.x) {
            AccT g = gamma ? (AccT)gamma[i] : (AccT)1;
            AccT b = beta  ? (AccT)beta[i]  : (AccT)0;
            row_y[i] = (T)(((AccT)row_x[i]-mu)*rstd*g+b);
        }
    } else if constexpr (std::is_same_v<T, __half>) {
        float4* y_vec = reinterpret_cast<float4*>(row_y);
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vc = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 xraw = x_vec[i];
            const __half2* xh = reinterpret_cast<const __half2*>(&xraw);
            float4 graw, braw;
            if (gamma) graw = reinterpret_cast<const float4*>(gamma)[i];
            if (beta)  braw = reinterpret_cast<const float4*>(beta)[i];
            const __half2* gh = gamma ? reinterpret_cast<const __half2*>(&graw) : nullptr;
            const __half2* bh = beta  ? reinterpret_cast<const __half2*>(&braw) : nullptr;
            float4 yraw; __half2* yh = reinterpret_cast<__half2*>(&yraw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 xf = __half22float2(xh[k]);
                xf.x = (xf.x-mu)*rstd; xf.y = (xf.y-mu)*rstd;
                if (gh) { float2 gf = __half22float2(gh[k]); xf.x *= gf.x; xf.y *= gf.y; }
                if (bh) { float2 bf = __half22float2(bh[k]); xf.x += bf.x; xf.y += bf.y; }
                yh[k] = __float22half2_rn(xf);
            }
            y_vec[i] = yraw;
        }
        for (int i = vc*8+tid; i < cols; i += blockDim.x) {
            AccT g = gamma ? (AccT)gamma[i] : (AccT)1;
            AccT b = beta  ? (AccT)beta[i]  : (AccT)0;
            row_y[i] = (T)(((AccT)row_x[i]-mu)*rstd*g+b);
        }
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        float4* y_vec = reinterpret_cast<float4*>(row_y);
        const float4* x_vec = reinterpret_cast<const float4*>(row_x);
        const int vc = cols / 8;
        #pragma unroll 4
        for (int i = tid; i < vc; i += blockDim.x) {
            float4 xraw = x_vec[i];
            const __nv_bfloat162* xh = reinterpret_cast<const __nv_bfloat162*>(&xraw);
            float4 graw, braw;
            if (gamma) graw = reinterpret_cast<const float4*>(gamma)[i];
            if (beta)  braw = reinterpret_cast<const float4*>(beta)[i];
            const __nv_bfloat162* gh = gamma ? reinterpret_cast<const __nv_bfloat162*>(&graw) : nullptr;
            const __nv_bfloat162* bh = beta  ? reinterpret_cast<const __nv_bfloat162*>(&braw) : nullptr;
            float4 yraw; __nv_bfloat162* yh = reinterpret_cast<__nv_bfloat162*>(&yraw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 xf = __bfloat1622float2(xh[k]);
                xf.x = (xf.x-mu)*rstd; xf.y = (xf.y-mu)*rstd;
                if (gh) { float2 gf = __bfloat1622float2(gh[k]); xf.x *= gf.x; xf.y *= gf.y; }
                if (bh) { float2 bf = __bfloat1622float2(bh[k]); xf.x += bf.x; xf.y += bf.y; }
                yh[k] = __float22bfloat162_rn(xf);
            }
            y_vec[i] = yraw;
        }
        for (int i = vc*8+tid; i < cols; i += blockDim.x) {
            AccT g = gamma ? (AccT)gamma[i] : (AccT)1;
            AccT b = beta  ? (AccT)beta[i]  : (AccT)0;
            row_y[i] = (T)(((AccT)row_x[i]-mu)*rstd*g+b);
        }
    }
}

template __global__ void layer_norm_forward_sm89_kernel<float, float>(
    const float*, const float*, const float*, float*, float*, float*, int, float);
template __global__ void layer_norm_forward_sm89_kernel<__half, float>(
    const __half*, const __half*, const __half*, __half*, float*, float*, int, float);
template __global__ void layer_norm_forward_sm89_kernel<__nv_bfloat16, float>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*, float*, int, float);
