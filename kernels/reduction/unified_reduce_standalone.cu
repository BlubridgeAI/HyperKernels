// unified_reduce_standalone.cu
// Unified GPU reduction with three layout paths (inner-contiguous,
// outer-contiguous, generic), an output-vectorised (float4) outer-contiguous
// kernel, and a multi-CTA global-reduce path, plus a self-test against a CPU
// reference.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <type_traits>
#include <vector>

// occupancy macros
#if defined(__CUDACC__)
#if __CUDA_ARCH__ == 750
#define CUDA_MAX_THREADS_PER_SM 1024
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890 ||  \
    __CUDA_ARCH__ == 1200
#define CUDA_MAX_THREADS_PER_SM 1536
#else
#define CUDA_MAX_THREADS_PER_SM 2048
#endif
#define GAU_MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm)                \
  ((((threads_per_block) * (blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM)         \
        ? (blocks_per_sm)                                                      \
        : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block) - 1) /               \
           (threads_per_block))))
#endif

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(_err), __FILE__,   \
             __LINE__);                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

namespace OwnTensor {
namespace detail {

// reduction layout descriptor
struct ReductionLayout {
  enum class Path { InnerContiguous, OuterContiguous, Generic };
  Path path = Path::InnerContiguous;
  int64_t num_outputs = 1;   // independent outputs
  int64_t reduced_count = 1; // elements reduced per output
  int64_t input_outer_stride =
      1;                   // InnerContiguous: stride between output slices
  int64_t inner_count = 1; // fast dim size (= num_outputs for OuterContiguous)
  int64_t input_row_stride =
      1; // OuterContiguous: stride between reduction rows
};

// reduce config
template <typename T> struct MaxThreads {
  static constexpr int VALUE = 512;
};

inline int next_pow2(int x) {
  if (x <= 0)
    return 1;
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}
inline int last_pow2(int x) { return next_pow2(x + 1) / 2; }
inline int div_up(int a, int b) { return (a + b - 1) / b; }

struct GpuReduceConfig {
  static constexpr int BLOCK_X = 0;
  static constexpr int BLOCK_Y = 1;
  static constexpr int CTA = 2;

  int element_size_bytes;
  int num_inputs;  // reduced_count: elements per output
  int num_outputs; // independent outputs
  int step_input = 1;
  int step_output = 1;
  int ctas_per_output = 1;
  int input_mult[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};
  int block_width = 1;
  int block_height = 1;
  int num_threads = 1;
  bool vectorize_input = false;
  int output_vec_size = 1;

  GpuReduceConfig() = default;
  GpuReduceConfig(int esz, int nout, int nin)
      : element_size_bytes(esz), num_inputs(nin), num_outputs(nout) {}

  template <typename T> void set_block_dimension(int64_t dim0, int64_t dim1) {
    const int max_t = MaxThreads<T>::VALUE / output_vec_size;
    int d0 = dim0 < max_t ? static_cast<int>(last_pow2(dim0)) : max_t;
    int d1 = dim1 < max_t ? static_cast<int>(last_pow2(dim1)) : max_t;
    block_width = std::min(d0, 32);
    block_height = std::min(d1, max_t / block_width);
    block_width = std::min(d0, max_t / block_height);
    num_threads = block_width * block_height;
  }

  int split_input(int p) {
    int s = step_input;
    step_input *= p;
    return s;
  }
  int split_output(int p) {
    int s = step_output;
    step_output *= p;
    return s;
  }

  dim3 block() const { return dim3(block_width, block_height); }
  dim3 grid() const {
    return dim3(div_up(num_outputs / output_vec_size, step_output),
                ctas_per_output);
  }

  __host__ __device__ bool should_block_x_reduce() const {
    return input_mult[BLOCK_X] != 0;
  }
  __host__ __device__ bool should_block_y_reduce() const {
    return input_mult[BLOCK_Y] != 0;
  }
  __host__ __device__ bool should_global_reduce() const {
    return input_mult[CTA] != 0;
  }

  int shared_memory_size() const {
    if (!should_block_y_reduce() &&
        (!should_block_x_reduce() || block_width <= 32))
      return 0;
    return element_size_bytes * num_threads * output_vec_size;
  }
  int values_per_thread() const { return div_up(num_inputs, step_input); }
  int semaphore_size() const {
    return should_global_reduce() ? (int)(sizeof(int) * grid().x) : 0;
  }
};

// host-side solver. num_mp / max_threads_per_mp come from the device and drive
// the multi-CTA decision (how many blocks the GPU can run concurrently).
template <typename acc_t>
GpuReduceConfig build_reduce_config(const ReductionLayout &layout, int num_mp,
                                    int max_threads_per_mp) {
  int num_outputs = static_cast<int>(layout.num_outputs);
  int inputs_per_output = static_cast<int>(layout.reduced_count);

  if (layout.path == ReductionLayout::Path::OuterContiguous) {
    num_outputs = static_cast<int>(layout.num_outputs);
    inputs_per_output = static_cast<int>(layout.reduced_count);
  } else if (layout.path == ReductionLayout::Path::Generic) {
    if (num_outputs == 0)
      num_outputs = 1;
    if (inputs_per_output == 0)
      inputs_per_output = 1;
  }

  auto config = GpuReduceConfig(sizeof(acc_t), num_outputs, inputs_per_output);
  bool inner = (layout.path != ReductionLayout::Path::OuterContiguous);

  int64_t dim0 = inner ? inputs_per_output : num_outputs;
  int64_t dim1 = inner ? num_outputs : inputs_per_output;

  // Output vectorisation (float4 stores). Eligible when the output dim is the
  // contiguous (fast) axis and a leading-axis reduction: fp32, leading-axis
  // (input_outer_stride == input_row_stride), num_outputs % 4 == 0, dim0 >=
  // 128. Halves max threads in set_block_dimension, so the block shrinks and
  // more blocks fit per SM.
  constexpr bool kEnableOutputVec = true;
  if (kEnableOutputVec && !inner && sizeof(acc_t) == sizeof(float) &&
      layout.input_outer_stride == layout.input_row_stride &&
      (num_outputs % 4) == 0 && dim0 >= 128) {
    config.output_vec_size = 4;
    dim0 /= 4;
  }

  config.set_block_dimension<acc_t>(dim0, dim1);

  int bw = config.block_width, bh = config.block_height;

  if (inner)
    config.input_mult[0] = config.split_input(bw);
  else
    config.output_mult[0] = config.split_output(bw);

  constexpr int min_vpt = 16, max_vpt = 256;
  bool split_warps =
      config.values_per_thread() >= std::min<int>(bh * 16, max_vpt);
  if (split_warps)
    config.input_mult[1] = config.split_input(bh);
  else
    config.output_mult[1] = config.split_output(bh);

  // Multi-CTA global reduce: split each output across several CTAs when the
  // grid is too small to fill the GPU and there is enough per-thread work to
  // subdivide. The per-CTA partials are combined by the kernel's global-reduce
  // path.
  const int blocks_per_sm = max_threads_per_mp / config.num_threads;
  const int target_grid = num_mp * blocks_per_sm;
  int gx = config.grid().x;
  if (config.input_mult[1] != 0 && config.values_per_thread() >= max_vpt &&
      gx <= target_grid) {
    int cpo =
        std::max(std::min<int>(div_up(target_grid, gx),
                               div_up(config.values_per_thread(), min_vpt)),
                 div_up(config.values_per_thread(), max_vpt));
    config.ctas_per_output = cpo;
    if (cpo > 1)
      config.input_mult[2] = config.split_input(cpo);
  }
  return config;
}

// strided offset calculator (Generic path only)
template <int NARGS = 1, typename index_t = uint32_t> struct OffsetCalculator {
  static constexpr int MAX_DIMS = 10;
  int dims;
  index_t sizes[MAX_DIMS];
  index_t strides[NARGS][MAX_DIMS];

  __host__ __device__ OffsetCalculator() : dims(0) {}

  __host__ __device__ OffsetCalculator(int dims, const int64_t *shape,
                                       const int64_t *const *strides_arr)
      : dims(dims) {
    for (int i = 0; i < dims && i < MAX_DIMS; i++) {
      this->sizes[i] = static_cast<index_t>(shape[i]);
      for (int j = 0; j < NARGS; j++)
        this->strides[j][i] = static_cast<index_t>(strides_arr[j][i]);
    }
  }

  __host__ __device__ index_t get(index_t linear_idx, int arg = 0) const {
    index_t offset = 0;
    for (int d = dims - 1; d >= 0; d--) {
      index_t coord = linear_idx % sizes[d];
      linear_idx /= sizes[d];
      offset += coord * strides[arg][d];
    }
    return offset;
  }
};

// packed reduce op
template <typename scalar_t, typename out_scalar_t, typename ops_t,
          typename index_t = uint32_t>
struct ReduceOp {
  ops_t ops;
  GpuReduceConfig config;
  OffsetCalculator<1, index_t> input_calc;
  OffsetCalculator<1, index_t> output_calc;
  const scalar_t *__restrict__ src;
  out_scalar_t *__restrict__ dst;
  int64_t step_stride;  // OuterContiguous: stride between reduction rows (=
                        // inner_count)
  int64_t outer_stride; // OuterContiguous middle-axis slab stride; 0 for a
                        // leading-axis reduction
  // multi-CTA staging, populated only when should_global_reduce(). semaphores
  // are zero-initialised by the host before launch.
  void *cta_buf = nullptr;
  int *semaphores = nullptr;
};

} // namespace detail

// sum functor (accumulator type == float)
struct SumOp {
  using arg_t = float;
  __host__ __device__ arg_t identity() const { return 0.0f; }
  // fold one input element into acc
  __device__ arg_t reduce(arg_t acc, float val, int64_t /*idx*/) const {
    return acc + val;
  }
  __device__ arg_t combine(arg_t a, arg_t b) const { return a + b; }
  __device__ arg_t project(arg_t v) const { return v; }
  __device__ arg_t warp_shfl_down(arg_t v, int offset) const {
    return __shfl_down_sync(0xffffffff, v, offset);
  }
};

namespace cuda {

// block-x reduce: threads in a row (threadIdx.x) reduce the input; threadIdx.y
// indexes independent outputs, so the reduction stays within a row. The
// cross-warp stage runs only when a row spans more than one warp.
template <typename arg_t, typename ops_t>
__device__ arg_t block_x_reduce(arg_t value, const ops_t &ops,
                                char *shared_memory) {
  int dim_x = blockDim.x;
  arg_t *smem = reinterpret_cast<arg_t *>(shared_memory);

  if (dim_x > 32) {
    int addr = threadIdx.x + threadIdx.y * blockDim.x;
    smem[addr] = value;
    for (int offset = dim_x / 2; offset >= 32; offset >>= 1) {
      __syncthreads();
      if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
        value = ops.combine(value, smem[addr + offset]);
        smem[addr] = value;
      }
    }
    dim_x = 32;
  }

  __syncthreads();
  // intra-warp shuffle bounded by the row width so offsets never cross into a
  // neighbouring output; result lands in threadIdx.x == 0.
  for (int offset = dim_x >> 1; offset > 0; offset >>= 1)
    value = ops.combine(value, ops.warp_shfl_down(value, offset));
  return value;
}

// block-y reduce: threads in the same column (same threadIdx.x) reduce across
// all y rows via shared memory.
template <typename arg_t, typename ops_t>
__device__ arg_t block_y_reduce(arg_t value, const ops_t &ops,
                                char *shared_memory) {
  arg_t *smem = reinterpret_cast<arg_t *>(shared_memory);
  smem[threadIdx.x + threadIdx.y * blockDim.x] = value;
  __syncthreads();

  if (threadIdx.y == 0) {
    value = smem[threadIdx.x];
    for (int i = 1; i < blockDim.y; i++)
      value = ops.combine(value, smem[threadIdx.x + i * blockDim.x]);
  }
  // trailing barrier: the vec4 kernel calls this once per output lane reusing
  // the same shared buffer, so ty!=0 threads must not race ahead and overwrite
  // smem while ty==0 is still reading the current lane.
  __syncthreads();
  return value;
}

template <int NT> struct MinBlocksPerSM {
  static constexpr int requested = (NT >= 256) ? (2048 / NT) : 4;
  static constexpr int VALUE = GAU_MIN_BLOCKS_PER_SM(NT, requested);
};

// scalar unified reduction kernel (3 paths + multi-CTA global reduce)
template <typename scalar_t, typename out_scalar_t, typename ops_t,
          typename index_t, detail::ReductionLayout::Path PATH, int NT,
          int VT0 = 4>
__launch_bounds__(NT, MinBlocksPerSM<NT>::VALUE) __global__
    void unified_reduce_kernel(
        detail::ReduceOp<scalar_t, out_scalar_t, ops_t, index_t> op) {
  extern __shared__ char shared_memory[];

  using arg_t = decltype(op.ops.identity());
  const auto &config = op.config;

  // map thread to output index
  index_t output_idx =
      (index_t)threadIdx.x *
          config.output_mult[detail::GpuReduceConfig::BLOCK_X] +
      (index_t)threadIdx.y *
          config.output_mult[detail::GpuReduceConfig::BLOCK_Y] +
      (index_t)blockIdx.x * config.step_output;

  if (output_idx >= (index_t)config.num_outputs)
    return;

  // map thread to its starting input index. The CTA term gives each block
  // along grid.y a disjoint slice of the reduced axis (multi-CTA).
  index_t idx =
      (index_t)threadIdx.x *
          config.input_mult[detail::GpuReduceConfig::BLOCK_X] +
      (index_t)threadIdx.y *
          config.input_mult[detail::GpuReduceConfig::BLOCK_Y] +
      (index_t)blockIdx.y * config.input_mult[detail::GpuReduceConfig::CTA];

  const index_t end = (index_t)config.num_inputs;
  const index_t stride = (index_t)config.step_input;

  // thread-local reduction with VT0 independent accumulators
  arg_t acc[VT0];
#pragma unroll
  for (int i = 0; i < VT0; i++)
    acc[i] = op.ops.identity();

  if constexpr (PATH == detail::ReductionLayout::Path::InnerContiguous) {
    const scalar_t *__restrict__ base_ptr =
        op.src + output_idx * (index_t)config.num_inputs;
    while (idx + (VT0 - 1) * stride < end) {
#pragma unroll
      for (int i = 0; i < VT0; i++) {
        acc[i] = op.ops.reduce(acc[i], base_ptr[idx + i * stride],
                               (int64_t)(idx + i * stride));
      }
      idx += stride * VT0;
    }
    while (idx < end) {
      acc[0] = op.ops.reduce(acc[0], base_ptr[idx], (int64_t)idx);
      idx += stride;
    }
  } else if constexpr (PATH == detail::ReductionLayout::Path::OuterContiguous) {
    // remap the flat output index to its input slab base: a leading-axis
    // reduction has o_outer == 0 so base == output_idx; a middle-axis
    // reduction lives at o_outer*outer_stride + o_inner.
    const index_t inner_count = (index_t)op.step_stride;
    const index_t o_outer = output_idx / inner_count;
    const index_t o_inner = output_idx - o_outer * inner_count;
    const scalar_t *__restrict__ base_ptr =
        op.src + o_outer * (index_t)op.outer_stride + o_inner;
    const index_t row_stride = (index_t)op.step_stride;
    while (idx + (VT0 - 1) * stride < end) {
#pragma unroll
      for (int i = 0; i < VT0; i++) {
        acc[i] =
            op.ops.reduce(acc[i], base_ptr[(idx + i * stride) * row_stride],
                          (int64_t)(idx + i * stride));
      }
      idx += stride * VT0;
    }
    while (idx < end) {
      acc[0] = op.ops.reduce(acc[0], base_ptr[idx * row_stride], (int64_t)idx);
      idx += stride;
    }
  } else {
    index_t out_base =
        (op.output_calc.dims > 0) ? op.output_calc.get(output_idx) : output_idx;
    while (idx + (VT0 - 1) * stride < end) {
#pragma unroll
      for (int i = 0; i < VT0; i++) {
        index_t cur = idx + i * stride;
        index_t in_off =
            (op.input_calc.dims > 0) ? op.input_calc.get(cur) : cur;
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
  for (int i = 1; i < VT0; i++)
    acc[0] = op.ops.combine(acc[0], acc[i]);

  arg_t value = acc[0];

  // block-level reduce stages
  if (config.should_block_x_reduce())
    value = block_x_reduce(value, op.ops, shared_memory);
  if (config.should_block_y_reduce())
    value = block_y_reduce(value, op.ops, shared_memory);

  bool is_leader = (!config.should_block_x_reduce() || threadIdx.x == 0) &&
                   (!config.should_block_y_reduce() || threadIdx.y == 0);

  // multi-CTA: stage each CTA's partial, the last CTA combines them
  if (config.should_global_reduce()) {
    arg_t *cta_buf = reinterpret_cast<arg_t *>(op.cta_buf);

    auto staging_offset = [&](index_t cta2) -> index_t {
      index_t off = cta2 + (index_t)blockIdx.x * (index_t)gridDim.y;
      if (!config.should_block_x_reduce())
        off = (index_t)threadIdx.x + off * (index_t)blockDim.x;
      return off;
    };

    if (is_leader)
      cta_buf[staging_offset((index_t)blockIdx.y)] = value;
    __threadfence(); // store must be visible before the atomicAdd
    __syncthreads();

    __shared__ bool s_is_last;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int prev = atomicAdd(&op.semaphores[blockIdx.x], 1);
      s_is_last = (prev == config.ctas_per_output - 1);
    }
    __syncthreads();
    if (!s_is_last)
      return;

    __threadfence(); // pair with the producers' fence above

    value = op.ops.identity();
    if (config.should_block_x_reduce()) {
      index_t input_offset =
          (index_t)threadIdx.x + (index_t)threadIdx.y * (index_t)blockDim.x;
      index_t step = (index_t)blockDim.x * (index_t)blockDim.y;
      for (; input_offset < (index_t)config.ctas_per_output;
           input_offset += step)
        value = op.ops.combine(value, cta_buf[staging_offset(input_offset)]);
    } else {
      index_t input_offset = (index_t)threadIdx.y;
      index_t step = (index_t)blockDim.y;
      for (; input_offset < (index_t)config.ctas_per_output;
           input_offset += step)
        value = op.ops.combine(value, cta_buf[staging_offset(input_offset)]);
    }

    value = block_y_reduce(value, op.ops, shared_memory);
    if (config.should_block_x_reduce())
      value = block_x_reduce(value, op.ops, shared_memory);

    if (is_leader && output_idx < (index_t)config.num_outputs) {
      auto final_val = op.ops.project(value);
      op.dst[output_idx] = static_cast<out_scalar_t>(final_val);
    }
    return;
  }

  // single-CTA: leader writes dst directly
  if (is_leader) {
    auto final_val = op.ops.project(value);
    op.dst[output_idx] = static_cast<out_scalar_t>(final_val);
  }
}

// vectorised (float4) outer-contiguous kernel. Each thread handles 4
// consecutive output columns: one float4 load per input row, 4 accumulators,
// one float4 store per output. Used when output_vec_size == 4 and path ==
// OuterContiguous.
template <typename scalar_t, typename out_scalar_t, typename ops_t,
          typename index_t, int NT, int VT0 = 4>
__launch_bounds__(NT, MinBlocksPerSM<NT>::VALUE) __global__
    void unified_reduce_kernel_outer_vec4(
        detail::ReduceOp<scalar_t, out_scalar_t, ops_t, index_t> op) {
  extern __shared__ char shared_memory[];
  using arg_t = decltype(op.ops.identity());
  const auto &config = op.config;

  constexpr int OUTPUT_VEC = 4;

  index_t output_idx_base =
      ((index_t)threadIdx.x *
           config.output_mult[detail::GpuReduceConfig::BLOCK_X] +
       (index_t)threadIdx.y *
           config.output_mult[detail::GpuReduceConfig::BLOCK_Y] +
       (index_t)blockIdx.x * config.step_output) *
      OUTPUT_VEC;

  if (output_idx_base >= (index_t)config.num_outputs)
    return;

  index_t idx =
      (index_t)threadIdx.x *
          config.input_mult[detail::GpuReduceConfig::BLOCK_X] +
      (index_t)threadIdx.y *
          config.input_mult[detail::GpuReduceConfig::BLOCK_Y] +
      (index_t)blockIdx.y * config.input_mult[detail::GpuReduceConfig::CTA];

  const index_t end = (index_t)config.num_inputs;
  const index_t stride = (index_t)config.step_input;
  const index_t row_stride = (index_t)op.step_stride;

  arg_t acc[OUTPUT_VEC];
#pragma unroll
  for (int v = 0; v < OUTPUT_VEC; v++)
    acc[v] = op.ops.identity();

  const scalar_t *__restrict__ base_ptr = op.src + output_idx_base;
  while (idx < end) {
    const scalar_t *row_ptr = base_ptr + (index_t)idx * row_stride;
    if constexpr (std::is_same_v<scalar_t, float>) {
      float4 v4 = *reinterpret_cast<const float4 *>(row_ptr);
      acc[0] = op.ops.reduce(acc[0], v4.x, (int64_t)idx);
      acc[1] = op.ops.reduce(acc[1], v4.y, (int64_t)idx);
      acc[2] = op.ops.reduce(acc[2], v4.z, (int64_t)idx);
      acc[3] = op.ops.reduce(acc[3], v4.w, (int64_t)idx);
    } else {
#pragma unroll
      for (int v = 0; v < OUTPUT_VEC; v++)
        acc[v] = op.ops.reduce(acc[v], row_ptr[v], (int64_t)idx);
    }
    idx += stride;
  }

  if (config.should_block_x_reduce()) {
#pragma unroll
    for (int v = 0; v < OUTPUT_VEC; v++)
      acc[v] = block_x_reduce(acc[v], op.ops, shared_memory);
  }
  if (config.should_block_y_reduce()) {
#pragma unroll
    for (int v = 0; v < OUTPUT_VEC; v++)
      acc[v] = block_y_reduce(acc[v], op.ops, shared_memory);
  }

  bool is_leader = (!config.should_block_x_reduce() || threadIdx.x == 0) &&
                   (!config.should_block_y_reduce() || threadIdx.y == 0);

  // multi-CTA staging: same protocol as the scalar kernel, OUTPUT_VEC values
  // per slot
  if (config.should_global_reduce()) {
    arg_t *cta_buf = reinterpret_cast<arg_t *>(op.cta_buf);

    auto staging_offset = [&](index_t cta2) -> index_t {
      index_t off = cta2 + (index_t)blockIdx.x * (index_t)gridDim.y;
      if (!config.should_block_x_reduce())
        off = (index_t)threadIdx.x + off * (index_t)blockDim.x;
      return off;
    };

    if (is_leader) {
      index_t base_off = staging_offset((index_t)blockIdx.y) * OUTPUT_VEC;
#pragma unroll
      for (int v = 0; v < OUTPUT_VEC; v++)
        cta_buf[base_off + v] = acc[v];
    }
    __threadfence();
    __syncthreads();

    __shared__ bool s_is_last;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int prev = atomicAdd(&op.semaphores[blockIdx.x], 1);
      s_is_last = (prev == config.ctas_per_output - 1);
    }
    __syncthreads();
    if (!s_is_last)
      return;

    __threadfence();

#pragma unroll
    for (int v = 0; v < OUTPUT_VEC; v++)
      acc[v] = op.ops.identity();

    if (config.should_block_x_reduce()) {
      index_t input_offset =
          (index_t)threadIdx.x + (index_t)threadIdx.y * (index_t)blockDim.x;
      index_t step = (index_t)blockDim.x * (index_t)blockDim.y;
      for (; input_offset < (index_t)config.ctas_per_output;
           input_offset += step) {
        index_t base_off = staging_offset(input_offset) * OUTPUT_VEC;
#pragma unroll
        for (int v = 0; v < OUTPUT_VEC; v++)
          acc[v] = op.ops.combine(acc[v], cta_buf[base_off + v]);
      }
    } else {
      index_t input_offset = (index_t)threadIdx.y;
      index_t step = (index_t)blockDim.y;
      for (; input_offset < (index_t)config.ctas_per_output;
           input_offset += step) {
        index_t base_off = staging_offset(input_offset) * OUTPUT_VEC;
#pragma unroll
        for (int v = 0; v < OUTPUT_VEC; v++)
          acc[v] = op.ops.combine(acc[v], cta_buf[base_off + v]);
      }
    }

#pragma unroll
    for (int v = 0; v < OUTPUT_VEC; v++)
      acc[v] = block_y_reduce(acc[v], op.ops, shared_memory);
    if (config.should_block_x_reduce()) {
#pragma unroll
      for (int v = 0; v < OUTPUT_VEC; v++)
        acc[v] = block_x_reduce(acc[v], op.ops, shared_memory);
    }

    if (is_leader && output_idx_base < (index_t)config.num_outputs) {
      if constexpr (std::is_same_v<out_scalar_t, float>) {
        float4 r;
        r.x = static_cast<float>(op.ops.project(acc[0]));
        r.y = static_cast<float>(op.ops.project(acc[1]));
        r.z = static_cast<float>(op.ops.project(acc[2]));
        r.w = static_cast<float>(op.ops.project(acc[3]));
        *reinterpret_cast<float4 *>(op.dst + output_idx_base) = r;
      } else {
#pragma unroll
        for (int v = 0; v < OUTPUT_VEC; v++)
          op.dst[output_idx_base + v] =
              static_cast<out_scalar_t>(op.ops.project(acc[v]));
      }
    }
    return;
  }

  // single-CTA: leader does the vector store directly
  if (is_leader) {
    if constexpr (std::is_same_v<out_scalar_t, float>) {
      float4 r;
      r.x = static_cast<float>(op.ops.project(acc[0]));
      r.y = static_cast<float>(op.ops.project(acc[1]));
      r.z = static_cast<float>(op.ops.project(acc[2]));
      r.w = static_cast<float>(op.ops.project(acc[3]));
      *reinterpret_cast<float4 *>(op.dst + output_idx_base) = r;
    } else {
#pragma unroll
      for (int v = 0; v < OUTPUT_VEC; v++)
        op.dst[output_idx_base + v] =
            static_cast<out_scalar_t>(op.ops.project(acc[v]));
    }
  }
}

// host-side kernel launcher
template <typename scalar_t, typename out_scalar_t, typename ops_t,
          typename index_t = uint32_t, int VT0 = 4>
inline void
launch_reduce_kernel(const ops_t &ops, const detail::GpuReduceConfig &config,
                     const detail::OffsetCalculator<1, index_t> &input_calc,
                     const detail::OffsetCalculator<1, index_t> &output_calc,
                     const scalar_t *src, out_scalar_t *dst,
                     detail::ReductionLayout::Path path, int64_t step_stride,
                     int smem, cudaStream_t stream, int64_t outer_stride = 0) {
  detail::ReduceOp<scalar_t, out_scalar_t, ops_t, index_t> op;
  op.ops = ops;
  op.config = config;
  op.input_calc = input_calc;
  op.output_calc = output_calc;
  op.src = src;
  op.dst = dst;
  op.step_stride = step_stride;
  op.outer_stride = outer_stride;

  // multi-CTA staging: cta_buf (uninitialised) plus zero-initialised semaphores
  using arg_t = decltype(ops.identity());
  if (config.should_global_reduce()) {
    const dim3 g = config.grid();
    size_t slots = static_cast<size_t>(g.x) * static_cast<size_t>(g.y);
    if (!config.should_block_x_reduce())
      slots *= static_cast<size_t>(config.block_width);
    size_t cta_bytes =
        sizeof(arg_t) * slots * static_cast<size_t>(config.output_vec_size);
    size_t sem_bytes = sizeof(int) * static_cast<size_t>(g.x);
    cudaMalloc(&op.cta_buf, cta_bytes);
    cudaMalloc(reinterpret_cast<void **>(&op.semaphores), sem_bytes);
    cudaMemsetAsync(op.semaphores, 0, sem_bytes, stream);
  }

  const int nt = config.num_threads;

#define LAUNCH_KERNEL(PATH_ENUM, NT_VAL)                                       \
  unified_reduce_kernel<scalar_t, out_scalar_t, ops_t, index_t, PATH_ENUM,     \
                        NT_VAL, VT0>                                           \
      <<<config.grid(), config.block(), smem, stream>>>(op)
#define LAUNCH_VEC4(NT_VAL)                                                    \
  unified_reduce_kernel_outer_vec4<scalar_t, out_scalar_t, ops_t, index_t,     \
                                   NT_VAL, VT0>                                \
      <<<config.grid(), config.block(), smem, stream>>>(op)

  if (config.output_vec_size == 4 &&
      path == detail::ReductionLayout::Path::OuterContiguous) {
    if (nt <= 32) {
      LAUNCH_VEC4(32);
    } else if (nt <= 64) {
      LAUNCH_VEC4(64);
    } else if (nt <= 128) {
      LAUNCH_VEC4(128);
    } else if (nt <= 256) {
      LAUNCH_VEC4(256);
    } else {
      LAUNCH_VEC4(512);
    }
  } else if (path == detail::ReductionLayout::Path::InnerContiguous) {
    if (nt <= 32) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 32);
    } else if (nt <= 64) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 64);
    } else if (nt <= 128) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 128);
    } else if (nt <= 256) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 256);
    } else {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 512);
    }
  } else if (path == detail::ReductionLayout::Path::OuterContiguous) {
    if (nt <= 32) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 32);
    } else if (nt <= 64) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 64);
    } else if (nt <= 128) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 128);
    } else if (nt <= 256) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 256);
    } else {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 512);
    }
  } else {
    if (nt <= 32) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 32);
    } else if (nt <= 64) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 64);
    } else if (nt <= 128) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 128);
    } else if (nt <= 256) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 256);
    } else {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 512);
    }
  }
#undef LAUNCH_KERNEL
#undef LAUNCH_VEC4

  if (op.cta_buf)
    cudaFree(op.cta_buf);
  if (op.semaphores)
    cudaFree(op.semaphores);
}

} // namespace cuda
} // namespace OwnTensor

using namespace OwnTensor;

// Run one case, return max abs error vs CPU. rows = outputs, cols = reduced
// length. Memory layout per path:
//   InnerContiguous : src[r*cols + k]  row-major, reduce over k     -> out[r]
//   OuterContiguous : src[k*rows + r]  col is fast/output, reduce k -> out[r]
//   Generic         : row-major as Inner, via OffsetCalculator      -> out[r]
static double run_case(const char *name, detail::ReductionLayout::Path path,
                       int rows, int cols, int num_mp, int max_tpm) {
  const long long n = (long long)rows * cols;
  std::vector<float> h_src(n);
  for (long long i = 0; i < n; i++)
    h_src[i] = (float)((i % 17) - 8) * 0.25f; // small bounded values

  // CPU reference: out[r] = sum over the reduced axis
  std::vector<float> h_ref(rows, 0.0f);
  if (path == detail::ReductionLayout::Path::OuterContiguous) {
    for (int r = 0; r < rows; r++)
      for (int k = 0; k < cols; k++)
        h_ref[r] += h_src[(long long)k * rows + r];
  } else {
    for (int r = 0; r < rows; r++)
      for (int k = 0; k < cols; k++)
        h_ref[r] += h_src[(long long)r * cols + k];
  }

  detail::ReductionLayout layout;
  layout.path = path;
  if (path == detail::ReductionLayout::Path::OuterContiguous) {
    layout.num_outputs = rows; // outputs == fast (contiguous) dim
    layout.inner_count = rows;
    layout.reduced_count = cols;
    // leading-axis reduction: row stride == outer stride, which enables vec4
    layout.input_row_stride = rows;
    layout.input_outer_stride = rows;
  } else {
    layout.num_outputs = rows;
    layout.reduced_count = cols;
    layout.input_outer_stride = cols;
    layout.inner_count = 1;
    layout.input_row_stride = 1;
  }
  auto config = detail::build_reduce_config<float>(layout, num_mp, max_tpm);

  float *d_src = nullptr, *d_dst = nullptr;
  cudaMalloc(&d_src, sizeof(float) * n);
  cudaMalloc(&d_dst, sizeof(float) * rows);
  cudaMemcpy(d_src, h_src.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemset(d_dst, 0, sizeof(float) * rows);

  // offset calculators: only the Generic path consumes them
  detail::OffsetCalculator<1, uint32_t> in_calc;
  detail::OffsetCalculator<1, uint32_t> out_calc;
  if (path == detail::ReductionLayout::Path::Generic) {
    int64_t shape[1] = {cols};
    int64_t in_strides[1] = {1};
    const int64_t *in_ptrs[1] = {in_strides};
    in_calc = detail::OffsetCalculator<1, uint32_t>(1, shape, in_ptrs);

    int64_t out_shape[1] = {rows};
    int64_t out_strides[1] = {cols};
    const int64_t *out_ptrs[1] = {out_strides};
    out_calc = detail::OffsetCalculator<1, uint32_t>(1, out_shape, out_ptrs);
  }

  // OuterContiguous: row stride == rows (= inner_count); single slab, so
  // outer_stride == 0
  int64_t step_stride =
      (path == detail::ReductionLayout::Path::OuterContiguous) ? rows : 1;
  int64_t outer_stride = 0;

  int smem = config.shared_memory_size();

  cuda::launch_reduce_kernel<float, float, SumOp, uint32_t, 4>(
      SumOp{}, config, in_calc, out_calc, d_src, d_dst, path, step_stride, smem,
      0, outer_stride);

  cudaError_t kerr = cudaDeviceSynchronize();
  if (kerr != cudaSuccess) {
    printf("[%-16s] LAUNCH/SYNC ERROR: %s\n", name, cudaGetErrorString(kerr));
    cudaFree(d_src);
    cudaFree(d_dst);
    return 1e9;
  }

  std::vector<float> h_out(rows, 0.0f);
  cudaMemcpy(h_out.data(), d_dst, sizeof(float) * rows, cudaMemcpyDeviceToHost);

  double max_err = 0.0;
  for (int r = 0; r < rows; r++)
    max_err = std::max(max_err, (double)std::fabs(h_out[r] - h_ref[r]));

  const char *kind = (config.output_vec_size == 4 &&
                      path == detail::ReductionLayout::Path::OuterContiguous)
                         ? "vec4"
                         : "scalar";
  printf("[%-18s] rows=%-5d cols=%-6d block=(%d,%d) grid=(%d,%d) cpo=%-4d %-6s "
         "%s  "
         "out[0]=%.4f ref[0]=%.4f  max_err=%.3e\n",
         name, rows, cols, config.block_width, config.block_height,
         config.grid().x, config.grid().y, config.ctas_per_output, kind,
         config.should_global_reduce() ? "multi-cta" : "single   ", h_out[0],
         h_ref[0], max_err);

  cudaFree(d_src);
  cudaFree(d_dst);
  return max_err;
}

int main() {
  int dev = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  const int num_mp = prop.multiProcessorCount;
  const int max_tpm = prop.maxThreadsPerMultiProcessor;
  printf("Device: %s  (sm_%d%d)  SMs=%d  maxThreads/SM=%d\n\n", prop.name,
         prop.major, prop.minor, num_mp, max_tpm);

  const double TOL =
      2e-2; // float32 accumulation tolerance over large reduced axes
  int failures = 0;

  struct Case {
    const char *name;
    detail::ReductionLayout::Path path;
    int rows, cols;
  };
  Case cases[] = {
      {"Inner small", detail::ReductionLayout::Path::InnerContiguous, 64, 100},
      {"Inner large", detail::ReductionLayout::Path::InnerContiguous, 1024,
       4096},
      {"Outer small", detail::ReductionLayout::Path::OuterContiguous, 64, 100},
      {"Outer large", detail::ReductionLayout::Path::OuterContiguous, 512,
       2048},
      {"Generic small", detail::ReductionLayout::Path::Generic, 64, 100},
      {"Generic large", detail::ReductionLayout::Path::Generic, 256, 3000},
      // OuterContiguous bias-gradient sums (feature dim x B*T=16384). Large
      // enough to take the vec4 + multi-cta path; the per-case line below
      // reports which path each shape actually used.
      {"Outer F=2304", detail::ReductionLayout::Path::OuterContiguous, 2304,
       16384},
      {"Outer F=768", detail::ReductionLayout::Path::OuterContiguous, 768,
       16384},
  };

  for (const auto &c : cases) {
    double err = run_case(c.name, c.path, c.rows, c.cols, num_mp, max_tpm);
    if (!(err <= TOL)) {
      printf("    ^^^ FAIL (err %.3e > tol %.3e)\n", err, TOL);
      failures++;
    }
  }

  printf("\n%s  (%d/%zu passed, tol=%.1e)\n",
         failures == 0 ? "ALL PASS" : "SOME FAILED",
         (int)(sizeof(cases) / sizeof(cases[0])) - failures,
         sizeof(cases) / sizeof(cases[0]), TOL);
  return failures == 0 ? 0 : 1;
}
