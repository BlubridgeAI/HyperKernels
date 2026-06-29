// unified_reduce_standalone.cu
// Unified single-CTA GPU reduction with three layout paths (inner-contiguous,
// outer-contiguous, generic) plus a self-test against a CPU reference.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <type_traits>

// occupancy macros
#if defined(__CUDACC__)
#if __CUDA_ARCH__ == 750
  #define CUDA_MAX_THREADS_PER_SM 1024
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1200
  #define CUDA_MAX_THREADS_PER_SM 1536
#else
  #define CUDA_MAX_THREADS_PER_SM 2048
#endif
#define GAU_MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm)        \
  ((((threads_per_block) * (blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM) \
        ? (blocks_per_sm)                                              \
        : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block) - 1) /       \
           (threads_per_block))))
#endif

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                             \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(_err),        \
                   __FILE__, __LINE__);                                         \
            return 1;                                                           \
        }                                                                       \
    } while (0)

namespace OwnTensor {
namespace detail {

// reduction layout descriptor
struct ReductionLayout {
    enum class Path { InnerContiguous, OuterContiguous, Generic };
    Path    path = Path::InnerContiguous;
    int64_t num_outputs  = 1;   // independent outputs
    int64_t reduced_count = 1;  // elements reduced per output
    int64_t inner_count   = 1;  // fast dim size (used by OuterContiguous)
};

// reduce config
template<typename T>
struct MaxThreads { static constexpr int VALUE = 512; };

inline int next_pow2(int x) {
    if (x <= 0) return 1;
    --x;
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}
inline int last_pow2(int x)   { return next_pow2(x + 1) / 2; }
inline int div_up(int a, int b){ return (a + b - 1) / b; }

struct GpuReduceConfig {
    static constexpr int BLOCK_X = 0;
    static constexpr int BLOCK_Y = 1;
    static constexpr int CTA     = 2;

    int element_size_bytes;
    int num_inputs;           // reduced_count: elements per output
    int num_outputs;          // independent outputs
    int step_input  = 1;
    int step_output = 1;
    int ctas_per_output = 1;
    int input_mult[3]  = {0, 0, 0};
    int output_mult[2] = {0, 0};
    int block_width  = 1;
    int block_height = 1;
    int num_threads  = 1;
    bool vectorize_input = false;
    int  output_vec_size = 1;

    GpuReduceConfig() = default;
    GpuReduceConfig(int esz, int nout, int nin)
        : element_size_bytes(esz), num_inputs(nin), num_outputs(nout) {}

    template<typename T>
    void set_block_dimension(int64_t dim0, int64_t dim1) {
        const int max_t = MaxThreads<T>::VALUE / output_vec_size;
        int d0 = dim0 < max_t ? static_cast<int>(last_pow2(dim0)) : max_t;
        int d1 = dim1 < max_t ? static_cast<int>(last_pow2(dim1)) : max_t;
        block_width  = std::min(d0, 32);
        block_height = std::min(d1, max_t / block_width);
        block_width  = std::min(d0, max_t / block_height);
        num_threads  = block_width * block_height;
    }

    int split_input (int p) { int s = step_input;  step_input  *= p; return s; }
    int split_output(int p) { int s = step_output; step_output *= p; return s; }

    dim3 block() const { return dim3(block_width, block_height); }
    dim3 grid()  const { return dim3(div_up(num_outputs / output_vec_size, step_output), ctas_per_output); }

    __host__ __device__ bool should_block_x_reduce() const { return input_mult[BLOCK_X] != 0; }
    __host__ __device__ bool should_block_y_reduce() const { return input_mult[BLOCK_Y] != 0; }
    __host__ __device__ bool should_global_reduce()  const { return input_mult[CTA]     != 0; }

    int shared_memory_size() const {
        if (!should_block_y_reduce() && (!should_block_x_reduce() || block_width <= 32))
            return 0;
        return element_size_bytes * num_threads * output_vec_size;
    }
    int values_per_thread() const { return div_up(num_inputs, step_input); }
};

// host-side solver (single-CTA: ctas_per_output stays 1)
template<typename acc_t>
GpuReduceConfig build_reduce_config(const ReductionLayout& layout) {
    int num_outputs       = static_cast<int>(layout.num_outputs);
    int inputs_per_output = static_cast<int>(layout.reduced_count);

    if (layout.path == ReductionLayout::Path::OuterContiguous) {
        num_outputs       = static_cast<int>(layout.inner_count);
        inputs_per_output = static_cast<int>(layout.reduced_count);
    } else if (layout.path == ReductionLayout::Path::Generic) {
        if (num_outputs       == 0) num_outputs       = 1;
        if (inputs_per_output == 0) inputs_per_output = 1;
    }

    auto config = GpuReduceConfig(sizeof(acc_t), num_outputs, inputs_per_output);
    bool inner = (layout.path != ReductionLayout::Path::OuterContiguous);

    int64_t dim0 = inner ? inputs_per_output : num_outputs;
    int64_t dim1 = inner ? num_outputs       : inputs_per_output;
    config.set_block_dimension<acc_t>(dim0, dim1);

    int bw = config.block_width, bh = config.block_height;

    if (inner) config.input_mult[0]  = config.split_input(bw);
    else       config.output_mult[0] = config.split_output(bw);

    constexpr int min_vpt = 16, max_vpt = 256;
    (void)min_vpt; (void)max_vpt;
    bool split_warps = config.values_per_thread() >= std::min<int>(bh * 16, max_vpt);
    if (split_warps) config.input_mult[1]  = config.split_input(bh);
    else             config.output_mult[1] = config.split_output(bh);

    return config;
}

// strided offset calculator
template<int NARGS = 1, typename index_t = uint32_t>
struct OffsetCalculator {
    static constexpr int MAX_DIMS = 10;
    int     dims;
    index_t sizes[MAX_DIMS];
    index_t strides[NARGS][MAX_DIMS];

    __host__ __device__ OffsetCalculator() : dims(0) {}

    __host__ __device__ OffsetCalculator(int dims, const int64_t* shape,
                                          const int64_t* const* strides_arr) : dims(dims) {
        for (int i = 0; i < dims && i < MAX_DIMS; i++) {
            this->sizes[i] = static_cast<index_t>(shape[i]);
            for (int j = 0; j < NARGS; j++)
                this->strides[j][i] = static_cast<index_t>(strides_arr[j][i]);
        }
    }

    __host__ __device__ index_t get(index_t linear_idx, int arg = 0) const {
        index_t offset = 0;
        for (int d = dims - 1; d >= 0; d--) {
            index_t coord  = linear_idx % sizes[d];
            linear_idx    /= sizes[d];
            offset        += coord * strides[arg][d];
        }
        return offset;
    }
};

// packed reduce op
template<typename scalar_t, typename out_scalar_t, typename ops_t, typename index_t = uint32_t>
struct ReduceOp {
    ops_t ops;
    GpuReduceConfig config;
    OffsetCalculator<1, index_t> input_calc;
    OffsetCalculator<1, index_t> output_calc;
    const scalar_t*  __restrict__ src;
    out_scalar_t*    __restrict__ dst;
    int64_t step_stride;
};

} // namespace detail

// sum functor (accumulator type == float)
struct SumOp {
    using arg_t = float;
    __host__ __device__ arg_t identity() const { return 0.0f; }
    // fold one input element into acc
    __device__ arg_t reduce(arg_t acc, float val, int64_t /*idx*/) const { return acc + val; }
    __device__ arg_t combine(arg_t a, arg_t b) const { return a + b; }
    __device__ arg_t project(arg_t v) const { return v; }
    __device__ arg_t warp_shfl_down(arg_t v, int offset) const {
        return __shfl_down_sync(0xffffffff, v, offset);
    }
};

namespace cuda {

// warp + block reduce stages
template<typename arg_t, typename ops_t>
__device__ arg_t block_x_reduce(arg_t value, const ops_t& ops, char* shared_memory) {
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        value = ops.combine(value, ops.warp_shfl_down(value, offset));

    arg_t* smem = reinterpret_cast<arg_t*>(shared_memory);
    if (lane == 0) smem[wid] = value;
    __syncthreads();

    if (wid == 0) {
        value = (threadIdx.x < blockDim.x / 32) ? smem[lane] : ops.identity();
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            value = ops.combine(value, ops.warp_shfl_down(value, offset));
    }
    return value;
}

template<typename arg_t, typename ops_t>
__device__ arg_t block_y_reduce(arg_t value, const ops_t& ops, char* shared_memory) {
    arg_t* smem = reinterpret_cast<arg_t*>(shared_memory);
    smem[threadIdx.x + threadIdx.y * blockDim.x] = value;
    __syncthreads();

    if (threadIdx.y == 0) {
        value = smem[threadIdx.x];
        for (int i = 1; i < blockDim.y; i++)
            value = ops.combine(value, smem[threadIdx.x + i * blockDim.x]);
    }
    return value;
}

// unified reduction kernel (single-CTA write path)
template<int NT>
struct MinBlocksPerSM {
    static constexpr int requested = (NT >= 256) ? (2048 / NT) : 4;
    static constexpr int VALUE = GAU_MIN_BLOCKS_PER_SM(NT, requested);
};

template<typename scalar_t, typename out_scalar_t, typename ops_t, typename index_t,
         detail::ReductionLayout::Path PATH, int NT, int VT0 = 4>
__launch_bounds__(NT, MinBlocksPerSM<NT>::VALUE)
__global__ void unified_reduce_kernel(detail::ReduceOp<scalar_t, out_scalar_t, ops_t, index_t> op) {
    extern __shared__ char shared_memory[];

    using arg_t = decltype(op.ops.identity());
    const auto& config = op.config;

    // map thread to output index
    index_t output_idx =
        (index_t)threadIdx.x * config.output_mult[detail::GpuReduceConfig::BLOCK_X] +
        (index_t)threadIdx.y * config.output_mult[detail::GpuReduceConfig::BLOCK_Y] +
        (index_t)blockIdx.x  * config.step_output;

    if (output_idx >= (index_t)config.num_outputs) return;

    // map thread to starting input index
    index_t idx =
        (index_t)threadIdx.x * config.input_mult[detail::GpuReduceConfig::BLOCK_X] +
        (index_t)threadIdx.y * config.input_mult[detail::GpuReduceConfig::BLOCK_Y] +
        (index_t)blockIdx.y  * config.input_mult[detail::GpuReduceConfig::CTA];

    const index_t end    = (index_t)config.num_inputs;
    const index_t stride = (index_t)config.step_input;

    // thread-local reduction with VT0 independent accumulators
    arg_t acc[VT0];
    #pragma unroll
    for (int i = 0; i < VT0; i++) acc[i] = op.ops.identity();

    if constexpr (PATH == detail::ReductionLayout::Path::InnerContiguous) {
        const scalar_t* __restrict__ base_ptr = op.src + output_idx * (index_t)config.num_inputs;
        while (idx + (VT0 - 1) * stride < end) {
            #pragma unroll
            for (int i = 0; i < VT0; i++) {
                acc[i] = op.ops.reduce(acc[i], base_ptr[idx + i * stride], (int64_t)(idx + i * stride));
            }
            idx += stride * VT0;
        }
        while (idx < end) {
            acc[0] = op.ops.reduce(acc[0], base_ptr[idx], (int64_t)idx);
            idx += stride;
        }
    }
    else if constexpr (PATH == detail::ReductionLayout::Path::OuterContiguous) {
        const scalar_t* __restrict__ base_ptr = op.src + output_idx;
        const index_t row_stride = (index_t)op.step_stride;
        while (idx + (VT0 - 1) * stride < end) {
            #pragma unroll
            for (int i = 0; i < VT0; i++) {
                acc[i] = op.ops.reduce(acc[i], base_ptr[(idx + i * stride) * row_stride], (int64_t)(idx + i * stride));
            }
            idx += stride * VT0;
        }
        while (idx < end) {
            acc[0] = op.ops.reduce(acc[0], base_ptr[idx * row_stride], (int64_t)idx);
            idx += stride;
        }
    }
    else {
        index_t out_base = (op.output_calc.dims > 0)
            ? op.output_calc.get(output_idx)
            : output_idx;
        while (idx + (VT0 - 1) * stride < end) {
            #pragma unroll
            for (int i = 0; i < VT0; i++) {
                index_t cur = idx + i * stride;
                index_t in_off = (op.input_calc.dims > 0) ? op.input_calc.get(cur) : cur;
                acc[i] = op.ops.reduce(acc[i], op.src[out_base + in_off], (int64_t)cur);
            }
            idx += stride * VT0;
        }
        while (idx < end) {
            index_t in_off = (op.input_calc.dims > 0) ? op.input_calc.get(idx) : idx;
            acc[0] = op.ops.reduce(acc[0], op.src[out_base + in_off], (int64_t)idx);
            idx += stride;
        }
    }

    // combine accumulators into acc[0]
    #pragma unroll
    for (int i = 1; i < VT0; i++) {
        acc[0] = op.ops.combine(acc[0], acc[i]);
    }

    arg_t value = acc[0];

    // block-level reduce stages
    if (config.should_block_x_reduce())
        value = block_x_reduce(value, op.ops, shared_memory);
    if (config.should_block_y_reduce())
        value = block_y_reduce(value, op.ops, shared_memory);

    // write output
    bool is_leader = (!config.should_block_x_reduce() || threadIdx.x == 0) &&
                     (!config.should_block_y_reduce() || threadIdx.y == 0);
    if (is_leader) {
        auto final_val = op.ops.project(value);
        op.dst[output_idx] = static_cast<out_scalar_t>(final_val);
    }
}

// host-side kernel launcher
template<typename scalar_t, typename out_scalar_t, typename ops_t, typename index_t = uint32_t, int VT0 = 4>
inline void launch_reduce_kernel(
    const ops_t& ops,
    const detail::GpuReduceConfig& config,
    const detail::OffsetCalculator<1, index_t>& input_calc,
    const detail::OffsetCalculator<1, index_t>& output_calc,
    const scalar_t* src,
    out_scalar_t* dst,
    detail::ReductionLayout::Path path,
    int64_t step_stride,
    int smem,
    cudaStream_t stream)
{
    detail::ReduceOp<scalar_t, out_scalar_t, ops_t, index_t> op;
    op.ops = ops;
    op.config = config;
    op.input_calc = input_calc;
    op.output_calc = output_calc;
    op.src = src;
    op.dst = dst;
    op.step_stride = step_stride;

    const int nt = config.num_threads;

    #define LAUNCH_KERNEL(PATH_ENUM, NT_VAL) \
        unified_reduce_kernel<scalar_t, out_scalar_t, ops_t, index_t, PATH_ENUM, NT_VAL, VT0> \
            <<<config.grid(), config.block(), smem, stream>>>(op)

    if (path == detail::ReductionLayout::Path::InnerContiguous) {
        if      (nt <= 32)  { LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 32);  }
        else if (nt <= 64)  { LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 64);  }
        else if (nt <= 128) { LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 128); }
        else if (nt <= 256) { LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 256); }
        else                { LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 512); }
    }
    else if (path == detail::ReductionLayout::Path::OuterContiguous) {
        if      (nt <= 32)  { LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 32);  }
        else if (nt <= 64)  { LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 64);  }
        else if (nt <= 128) { LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 128); }
        else if (nt <= 256) { LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 256); }
        else                { LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 512); }
    }
    else {
        if      (nt <= 32)  { LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 32);  }
        else if (nt <= 64)  { LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 64);  }
        else if (nt <= 128) { LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 128); }
        else if (nt <= 256) { LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 256); }
        else                { LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 512); }
    }
    #undef LAUNCH_KERNEL
}

} // namespace cuda
} // namespace OwnTensor

using namespace OwnTensor;

// Run one case, return max abs error vs CPU. rows = outputs, cols = reduced length.
// Memory layout per path:
//   InnerContiguous : src[r*cols + k]  row-major, reduce over k     -> out[r]
//   OuterContiguous : src[k*rows + r]  col is fast/output, reduce k -> out[r]
//   Generic         : row-major as Inner, via OffsetCalculator      -> out[r]
static double run_case(const char* name,
                       detail::ReductionLayout::Path path,
                       int rows, int cols)
{
    const int n = rows * cols;
    std::vector<float> h_src(n);
    for (int i = 0; i < n; i++) h_src[i] = (float)((i % 17) - 8) * 0.25f; // small bounded values

    // CPU reference: out[r] = sum over the reduced axis
    std::vector<float> h_ref(rows, 0.0f);
    if (path == detail::ReductionLayout::Path::OuterContiguous) {
        // src[k*rows + r], reduce over k
        for (int r = 0; r < rows; r++)
            for (int k = 0; k < cols; k++)
                h_ref[r] += h_src[k * rows + r];
    } else {
        // src[r*cols + k], reduce over k
        for (int r = 0; r < rows; r++)
            for (int k = 0; k < cols; k++)
                h_ref[r] += h_src[r * cols + k];
    }

    detail::ReductionLayout layout;
    layout.path = path;
    if (path == detail::ReductionLayout::Path::OuterContiguous) {
        layout.inner_count   = rows;   // outputs == fast dim
        layout.reduced_count = cols;
        layout.num_outputs   = rows;
    } else {
        layout.num_outputs   = rows;
        layout.reduced_count = cols;
        layout.inner_count   = 1;
    }
    auto config = detail::build_reduce_config<float>(layout);

    // Device buffers
    float *d_src = nullptr, *d_dst = nullptr;
    cudaMalloc(&d_src, sizeof(float) * n);
    cudaMalloc(&d_dst, sizeof(float) * rows);
    cudaMemcpy(d_src, h_src.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0, sizeof(float) * rows);

    // offset calculators: only the Generic path consumes them
    detail::OffsetCalculator<1, uint32_t> in_calc;   // dims==0 -> identity
    detail::OffsetCalculator<1, uint32_t> out_calc;  // dims==0 -> identity
    if (path == detail::ReductionLayout::Path::Generic) {
        int64_t shape[1]      = { cols };
        int64_t in_strides[1] = { 1 };
        const int64_t* in_ptrs[1] = { in_strides };
        in_calc = detail::OffsetCalculator<1, uint32_t>(1, shape, in_ptrs);

        int64_t out_shape[1]   = { rows };
        int64_t out_strides[1] = { cols };
        const int64_t* out_ptrs[1] = { out_strides };
        out_calc = detail::OffsetCalculator<1, uint32_t>(1, out_shape, out_ptrs);
    }

    // step_stride only matters for OuterContiguous (row stride == rows)
    int64_t step_stride = (path == detail::ReductionLayout::Path::OuterContiguous) ? rows : 1;

    int smem = config.shared_memory_size();

    cuda::launch_reduce_kernel<float, float, SumOp, uint32_t, 4>(
        SumOp{}, config, in_calc, out_calc, d_src, d_dst, path, step_stride, smem, 0);

    cudaError_t kerr = cudaDeviceSynchronize();
    if (kerr != cudaSuccess) {
        printf("[%-16s] LAUNCH/SYNC ERROR: %s\n", name, cudaGetErrorString(kerr));
        cudaFree(d_src); cudaFree(d_dst);
        return 1e9;
    }

    std::vector<float> h_out(rows, 0.0f);
    cudaMemcpy(h_out.data(), d_dst, sizeof(float) * rows, cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for (int r = 0; r < rows; r++)
        max_err = std::max(max_err, (double)std::fabs(h_out[r] - h_ref[r]));

    printf("[%-16s] rows=%d cols=%d  block=(%d,%d) grid=(%d,%d) smem=%d  "
           "out[0]=%.4f ref[0]=%.4f  max_err=%.3e\n",
           name, rows, cols, config.block_width, config.block_height,
           config.grid().x, config.grid().y, smem,
           h_out[0], h_ref[0], max_err);

    cudaFree(d_src); cudaFree(d_dst);
    return max_err;
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    const double TOL = 1e-3;  // float32 accumulation tolerance for bounded inputs
    int failures = 0;

    // All 3 paths, a few shapes each (small + large reduced axis to hit both
    // the single-warp and multi-warp / block-y configs of the solver).
    struct Case { const char* name; detail::ReductionLayout::Path path; int rows, cols; };
    Case cases[] = {
        { "Inner small",   detail::ReductionLayout::Path::InnerContiguous, 64,   100   },
        { "Inner large",   detail::ReductionLayout::Path::InnerContiguous, 1024, 4096  },
        { "Outer small",   detail::ReductionLayout::Path::OuterContiguous, 64,   100   },
        { "Outer large",   detail::ReductionLayout::Path::OuterContiguous, 512,  2048  },
        { "Generic small", detail::ReductionLayout::Path::Generic,         64,   100   },
        { "Generic large", detail::ReductionLayout::Path::Generic,         256,  3000  },
    };

    for (const auto& c : cases) {
        double err = run_case(c.name, c.path, c.rows, c.cols);
        if (!(err <= TOL)) {
            printf("    ^^^ FAIL (err %.3e > tol %.3e)\n", err, TOL);
            failures++;
        }
    }

    printf("\n%s  (%d/%zu passed, tol=%.1e)\n",
           failures == 0 ? "ALL PASS" : "SOME FAILED",
           (int)(sizeof(cases)/sizeof(cases[0])) - failures,
           sizeof(cases)/sizeof(cases[0]), TOL);
    return failures == 0 ? 0 : 1;
}
