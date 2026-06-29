NVCC    ?= nvcc
ARCH    ?= sm_89
NVFLAGS := -arch=$(ARCH) -O2 -std=c++17 -static-global-template-stub=false

# Kernels live in category subfolders one level up. vpath lets make resolve each
# source by basename regardless of which folder it sits in.
vpath %.cu ../attention ../gelu ../layernorm ../loss ../optimizer

# Only the kernels referenced by kernels_test.cu. ../reduction/unified_reduce_standalone.cu
# has its own main() and is built/run separately.
KERNEL_SRCS := \
  gelu_forward_sm89_kernel.cu \
  gelu_backward_sm89_kernel.cu \
  layer_norm_forward_sm89_kernel.cu \
  ln_backward_input_sm89_kernel.cu \
  ln_backward_gamma_beta_sm89_kernel.cu \
  sparse_ce_forward_kernel_vec_save_stats.cu \
  sparseCENormalize_from_stats.cu \
  multi_tensor_adam_sm89_kernel.cu \
  mem_efficient_bwd_precompute_D_sm89.cu \
  fused_attn_forward_kernel_tc_sm89.cu \
  mem_efficient_bwd_unified_kernel_exp12.cu

KERNEL_OBJS := $(patsubst %.cu,%.o,$(KERNEL_SRCS))

all: kernels_test

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

kernels_test.o: kernels_test.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

kernels_test: kernels_test.o $(KERNEL_OBJS)
	$(NVCC) $(NVFLAGS) $^ -o $@

run: kernels_test
	./kernels_test

clean:
	rm -f *.o kernels_test

.PHONY: all run clean
