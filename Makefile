NVCC ?= nvcc
ARCH ?= sm_89

all: kernels blas

kernels:
	$(MAKE) -C kernels/tests all ARCH=$(ARCH) NVCC=$(NVCC)

blas:
	$(MAKE) -C blas/tests all ARCH=$(ARCH) NVCC=$(NVCC)

test:
	$(MAKE) -C kernels/tests run ARCH=$(ARCH) NVCC=$(NVCC)
	$(MAKE) -C blas/tests run ARCH=$(ARCH) NVCC=$(NVCC)

test-kernels:
	$(MAKE) -C kernels/tests run ARCH=$(ARCH) NVCC=$(NVCC)

test-blas:
	$(MAKE) -C blas/tests run ARCH=$(ARCH) NVCC=$(NVCC)

clean:
	$(MAKE) -C kernels/tests clean
	$(MAKE) -C blas/tests clean

.PHONY: all kernels blas test test-kernels test-blas clean
