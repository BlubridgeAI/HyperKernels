# Compute kernels (sm_89)

Standalone FP32 CUDA kernels extracted from a larger codebase, each with its real
dependency closure inlined so it compiles on its own. All target `sm_89` and build
with CUDA 13.0.

The kernel-only files contain no `main()` and compile to object files with `-c`.
`reduction/unified_reduce_standalone.cu` is a full program with a built-in
correctness test and compiles to an executable.

## Layout

| Folder | Kernels |
|--------|---------|
| [`attention/`](attention/) | fused attention forward (tensor-core), memory-efficient backward (precompute D + unified) |
| [`layernorm/`](layernorm/) | LayerNorm forward, backward (input), backward (gamma/beta) |
| [`gelu/`](gelu/) | GELU forward, backward |
| [`loss/`](loss/) | sparse cross-entropy forward (saves stats), normalize-from-stats |
| [`optimizer/`](optimizer/) | multi-tensor Adam |
| [`reduction/`](reduction/) | unified reduce (standalone, with built-in test) |
| [`tests/`](tests/) | `kernels_test.cu` correctness harness + `Makefile` |

## Compile commands

Each kernel compiles independently. Paths below are relative to this `kernels/`
directory.

```bash
# Attention
nvcc -arch=sm_89 -O2 -std=c++17 -c attention/fused_attn_forward_kernel_tc_sm89.cu        -o fused_attn_forward_kernel_tc_sm89.o
nvcc -arch=sm_89 -O2 -std=c++17 -c attention/mem_efficient_bwd_precompute_D_sm89.cu       -o mem_efficient_bwd_precompute_D_sm89.o
nvcc -arch=sm_89 -O2 -std=c++17 -c attention/mem_efficient_bwd_unified_kernel_exp12.cu    -o mem_efficient_bwd_unified_kernel_exp12.o

# LayerNorm
nvcc -arch=sm_89 -O2 -std=c++17 -c layernorm/layer_norm_forward_sm89_kernel.cu            -o layer_norm_forward_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c layernorm/ln_backward_gamma_beta_sm89_kernel.cu        -o ln_backward_gamma_beta_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c layernorm/ln_backward_input_sm89_kernel.cu             -o ln_backward_input_sm89_kernel.o

# GELU
nvcc -arch=sm_89 -O2 -std=c++17 -c gelu/gelu_forward_sm89_kernel.cu                       -o gelu_forward_sm89_kernel.o
nvcc -arch=sm_89 -O2 -std=c++17 -c gelu/gelu_backward_sm89_kernel.cu                      -o gelu_backward_sm89_kernel.o

# Loss (sparse cross-entropy)
nvcc -arch=sm_89 -O2 -std=c++17 -c loss/sparse_ce_forward_kernel_vec_save_stats.cu        -o sparse_ce_forward_kernel_vec_save_stats.o
nvcc -arch=sm_89 -O2 -std=c++17 -c loss/sparseCENormalize_from_stats.cu                   -o sparseCENormalize_from_stats.o

# Optimizer
nvcc -arch=sm_89 -O2 -std=c++17 -c optimizer/multi_tensor_adam_sm89_kernel.cu             -o multi_tensor_adam_sm89_kernel.o

# Reduction (full program with a built-in correctness test, builds an executable)
nvcc -arch=sm_89 -O2 -std=c++17 reduction/unified_reduce_standalone.cu                    -o unified_reduce_standalone
./unified_reduce_standalone
```

## Tests

`tests/kernels_test.cu` is a correctness harness that links the kernel objects and
validates each against a host reference. From `tests/`:

```bash
make run
```

The harness covers GELU, LayerNorm, sparse CE, Adam, and the memory-efficient
backward precompute. The two FlashAttention kernels and the standalone reduce are
built and run separately (see above).
