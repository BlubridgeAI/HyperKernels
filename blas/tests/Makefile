NVCC    ?= nvcc
ARCH    ?= sm_89
# -static-global-template-stub=false: CUDA 13 defaults this to true, which would
# require each launched __global__ template instantiation to be defined in the
# launching TU. The kernels live in their own objects, so we opt out.
NVFLAGS := -arch=$(ARCH) -O2 -std=c++17 -static-global-template-stub=false

KERNEL_DIR := ../fp32
KERNEL_SRCS := \
  sgemm_sm89_core_template.cu \
  sgemm_addmm_sm89_kernel.cu \
  sgemm_nn_256x128_sm89_kernel.cu \
  sgemm_nn_256x64_sm89_kernel.cu \
  sgemm_nt_128x128_sm89_kernel.cu \
  sgemm_nt_256x128_sm89_kernel.cu \
  sgemm_nt_64x128_sm89_kernel.cu \
  sgemm_tn_128x128_sm89_kernel.cu \
  sgemm_tn_256x128_sm89_kernel.cu \
  sgemm_tn_256x64_sm89_kernel.cu

KERNEL_OBJS := $(patsubst %.cu,%.o,$(KERNEL_SRCS))

BF16_DIR  := ../bf16
BF16_SRCS := \
  bgemm_core_template.cu \
  bgemm_addmm_sm89_kernel.cu \
  bgemm_nn_128x128_sm89_kernel.cu \
  bgemm_nn_128x128_64x3_sm89_kernel.cu \
  bgemm_wmma_128x128_sm89_kernel.cu \
  bgemm_backward_nt_sm89_kernel.cu \
  bgemm_backward_tn_sm89_kernel.cu \
  bgemv_sm89_kernel.cu

BF16_OBJS := $(patsubst %.cu,bf16_%.o,$(BF16_SRCS))

all: sgemm_test bgemm_test

# ---- FP32 SGEMM test ----
%.o: $(KERNEL_DIR)/%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

sgemm_test.o: sgemm_test.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

sgemm_test: sgemm_test.o $(KERNEL_OBJS)
	$(NVCC) $(NVFLAGS) $^ -o $@

# ---- BF16 BGEMM test ----
bf16_%.o: $(BF16_DIR)/%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

bgemm_test.o: bgemm_test.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

bgemm_test: bgemm_test.o $(BF16_OBJS)
	$(NVCC) $(NVFLAGS) $^ -o $@

run: sgemm_test bgemm_test
	./sgemm_test
	./bgemm_test

run-bf16: bgemm_test
	./bgemm_test

clean:
	@echo "--- Cleaning build artifacts ---"
	rm -f *.o sgemm_test bgemm_test
	find . -maxdepth 1 -type f -executable -name 'test_*' -delete

# ---- Standalone cuBLAS Benchmark Tests ----
test_sgemm_%.o: test_sgemm_%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

test_sgemm_%: test_sgemm_%.o $(KERNEL_OBJS)
	$(NVCC) $(NVFLAGS) $^ -lcublas -o $@

test_bgemm_%.o: test_bgemm_%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

test_bgemm_%: test_bgemm_%.o $(BF16_OBJS)
	$(NVCC) $(NVFLAGS) $^ -lcublas -o $@

# Build a single test — Usage: make test TEST=test_bgemm_addmm_sm89
test:
	@if [ -z "$(TEST)" ]; then \
		echo "ERROR: Please specify TEST name"; \
		echo "Usage: make test TEST=test_bgemm_addmm_sm89"; \
		exit 1; \
	fi
	$(eval TEST_NAME := $(basename $(notdir $(TEST))))
	@echo "--- Building/Updating test: $(TEST_NAME) ---"
	@$(MAKE) $(TEST_NAME)
	@echo "✅  Built: $(TEST_NAME)"

# Build and run a test — Usage: make run-test TEST=test_bgemm_addmm_sm89
run-test:
	@if [ -z "$(TEST)" ]; then \
		echo "ERROR: Please specify TEST name"; \
		echo "Usage: make run-test TEST=test_bgemm_addmm_sm89"; \
		exit 1; \
	fi
	$(eval TEST_NAME := $(basename $(notdir $(TEST))))
	@$(MAKE) test TEST=$(TEST_NAME)
	@echo "--- Running $(TEST_NAME) ---"
	./$(TEST_NAME)

.PHONY: all run run-bf16 clean test run-test
