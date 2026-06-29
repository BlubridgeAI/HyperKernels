# BLAS kernels (sm_89)

Standalone batched-GEMM kernels for Ada Lovelace (`sm_89`), built with CUDA 13.0.
Each `.cu` is one kernel with its device-side dependency closure inlined, so it
compiles on its own to an object file (no `main()`).

| Precision | Path | MMA path | Count |
|-----------|------|----------|-------|
| FP32 (TF32 tensor core) | [`fp32/`](fp32/) | `mma.sync.aligned.m16n8k8` | 14 |
| BF16 | [`bf16/`](bf16/) | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | 8 |

All kernels use a multi-stage `cp.async` shared-memory pipeline. Several configs
exceed the 48 KB static shared-memory limit; for those the caller must raise the cap
with `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`
and pass the matching smem size in the launch config.

## FP32 SGEMM ([`fp32/`](fp32/))

Naming is `sgemm_<layout>_<tileM>x<tileN>_sm89_kernel.cu`, where layout is one of
NN, NT, TN (the transpose state of A and B). `sgemm_sm89_core_template.cu` is the
templated core covering all three layouts and the tile configs in one kernel,
including split-K and bias.

```bash
# Run from inside fp32/. Every file is kernel-only (no main()), built with -c.

# Templated core (NN/NT/TN, multiple tile configs, split-K, bias)
nvcc -arch=sm_89 -O2 -std=c++17 -c sgemm_sm89_core_template.cu            -o sgemm_sm89_core_template.o

# Fused GEMM + bias
nvcc -arch=sm_89 -O2 -std=c++17 -c sgemm_addmm_sm89_kernel.cu             -o sgemm_addmm_sm89_kernel.o

# NN layout
nvcc -arch=sm_89 -O2 -std=c++17 -c sgemm_nn_256x128_bypass_sm89_kernel.cu -o sgemm_nn_256x128_bypass_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c sgemm_nn_256x128_sm89_kernel.cu        -o sgemm_nn_256x128_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c sgemm_nn_256x64_sm89_kernel.cu         -o sgemm_nn_256x64_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c sgemm_nn_64x64_sm89_kernel.cu          -o sgemm_nn_64x64_sm89_kernel.o

# NT layout
nvcc -arch=sm_89 -O2 -std=c++17 -c sgemm_nt_256x128_sm89_kernel.cu        -o sgemm_nt_256x128_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c sgemm_nt_128x128_sm89_kernel.cu        -o sgemm_nt_128x128_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c sgemm_nt_64x128_sm89_kernel.cu         -o sgemm_nt_64x128_sm89_kernel.o

# TN layout
nvcc -arch=sm_89 -O2 -std=c++17 -c sgemm_tn_256x128_sm89_kernel.cu        -o sgemm_tn_256x128_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c sgemm_tn_256x64_sm89_kernel.cu         -o sgemm_tn_256x64_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c sgemm_tn_128x128_sm89_kernel.cu        -o sgemm_tn_128x128_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c sgemm_tn_64x64_sm89_kernel.cu          -o sgemm_tn_64x64_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c sgemm_tn_32x32_sm89_kernel.cu          -o sgemm_tn_32x32_sm89_kernel.o
```

## BF16 BGEMM ([`bf16/`](bf16/))

The bfloat16 counterpart, including forward, backward (dA / dB), a WMMA-API variant,
and GEMV. See [`bf16/README.md`](bf16/README.md) for the kernel table and compile
commands.

## Tests

[`tests/`](tests/) holds correctness harnesses for both precisions: `sgemm_test.cu`
and `bgemm_test.cu`, sharing one `Makefile`.

```bash
cd tests
make run        # builds and runs both sgemm_test and bgemm_test
make run-bf16   # bf16 only
```

Each kernel is validated against a float reference over identical inputs (bf16 uses
`rel <= 5e-2`, sized for its 8-bit mantissa). The vectorized kernels require
8-element-aligned leading dimensions (16-byte `cp.async`/`int4` access), so the tests
use M/N/K that are multiples of 8 — the same constraint the library dispatcher
enforces.
