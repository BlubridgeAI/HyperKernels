# BF16 BGEMM kernels (sm_89)

Each `.cu` here is one standalone bfloat16 batched-GEMM kernel (the BF16
counterpart of the FP32 kernels in the parent `blas/` directory) with its
device-side dependency closure inlined, so it compiles on its own. All use the
bf16 tensor-core MMA path (`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`)
with a multi-stage `cp.async` shared-memory pipeline. Build with CUDA 13.0.

These are extracted from `BluBridge-BLAS/src/level3/batched_gemm/BF16/`. The
forward kernels are templated on the transpose state of A and B
(`mycublasOperation_t`, inlined here as a plain enum); the kernel-only files
end with explicit template instantiations so the symbols are emitted under `-c`.
The library's dispatcher and the `*_variants` instantiation files are *not*
extracted — only the kernels themselves.

| File | Kernel | Notes |
|------|--------|-------|
| `bgemm_core_template.cu`             | templated core (NN/NT/TN, split-K) | + split-K scale kernel |
| `bgemm_addmm_sm89_kernel.cu`         | fused GEMM + bias (`C = alpha*A*B + beta*bias`) | NN |
| `bgemm_nn_128x128_sm89_kernel.cu`    | 128x128x32 forward | NN/NT/TN/TT |
| `bgemm_nn_128x128_64x3_sm89_kernel.cu` | 128x128x64 forward | NN/NT/TN/TT |
| `bgemm_wmma_128x128_sm89_kernel.cu`  | 128x128x32 forward via WMMA API | NN |
| `bgemm_backward_nt_sm89_kernel.cu`   | `C = alpha*A*B^T + beta*C` | NT (dA) |
| `bgemm_backward_tn_sm89_kernel.cu`   | `C = alpha*A^T*B + beta*C` | TN (dB) |
| `bgemv_sm89_kernel.cu`               | GEMV (M==1 / N==1) | NN/NT/TN |

## Compile commands

Run from inside `bf16/`. Every file is kernel-only (no `main()`), so each
compiles to an object with `-c`.

```bash
nvcc -arch=sm_89 -O2 -std=c++17 -c bgemm_core_template.cu             -o bgemm_core_template.o
nvcc -arch=sm_89 -O2 -std=c++17 -c bgemm_addmm_sm89_kernel.cu         -o bgemm_addmm_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c bgemm_nn_128x128_sm89_kernel.cu    -o bgemm_nn_128x128_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c bgemm_nn_128x128_64x3_sm89_kernel.cu -o bgemm_nn_128x128_64x3_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c bgemm_wmma_128x128_sm89_kernel.cu  -o bgemm_wmma_128x128_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c bgemm_backward_nt_sm89_kernel.cu   -o bgemm_backward_nt_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c bgemm_backward_tn_sm89_kernel.cu   -o bgemm_backward_tn_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c bgemv_sm89_kernel.cu               -o bgemv_sm89_kernel.o
```

Most tiles use exactly 48 KB of shared memory; the 64x3, 256x128 and the
6-stage core-template configs exceed it, so the caller must raise the dynamic
shared-memory cap with
`cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`
and pass the matching smem size in the launch config.

## Tests

`../tests/bgemm_test.cu` is a correctness harness covering these kernels. From
`../tests/`:

```bash
make run-bf16
```

It validates each kernel against a float reference over identical bf16 inputs
(`rel <= 5e-2`, sized for bf16's 8-bit mantissa). Note:

- The vectorized kernels require 8-element-aligned leading dimensions (16-byte
  `cp.async`/`int4` access), so the test uses M/N/K that are multiples of 8 —
  the same constraint the library dispatcher enforces.
- `bgemm_backward_nt` / `bgemm_backward_tn` are exercised only as **non-gated
  diagnostics**: in a plain single-tile launch their `load_regB` reads past the
  B tile, so correct results need the library dispatcher's launch configuration.
  They are extracted verbatim and compile/link like the rest.
