# Measurements

Every measured number in the book comes from `results.json` in this folder, produced by the scripts below
on one machine. When you quote a number, quote the machine with it.

## The machine

| | |
|---|---|
| **CPU** | Intel Xeon, Emerald Rapids (family 6, model 207), cloud VM, 4 vCPUs |
| **Caches** | 48 KB L1d and 2 MB L2 per core, 260 MB shared L3 |
| **ISA** | AVX2, AVX-512 (F, BW, VL, VNNI, BF16, FP16), AMX (tile, bf16, int8) |
| **Compilers** | gcc 13.3, clang 18.1 |
| **Python** | 3.13, NumPy 2.5 (single-threaded BLAS under `taskset`) |
| **Method** | one core (`taskset -c 1`); each C number is the best of 3 processes × best of 5 timed trials of ≥ 0.15 s; Python numbers are the best of 3–5 batches; checksums printed so work cannot be eliminated |

Before pinning, repeated runs on this VM differed by up to 1.6×. Pinning and best-of-N brought repeated
cells within a few percent.

## Scripts

| Script | Produces | Book |
|---|---|---|
| `run_bench.py` | `results.json`: E1 width ladder in L1, E2 saxpy by memory level, E3 stride sweep, E4 dot accumulators, E5 matmul loop order, E6 pairwise distances in C, E7 FMA peak per width, E8 AMX peak | Ch 0, 3, 7, 9, 14, 16, 20 |
| `run_bench.py --only amx` | re-runs only the AMX experiment | Ch 16 |
| `python_bench.py [--only …]` | merges `pdist_python`, `numpy_idioms`, `numpy_call_overhead`, `np_vectorize`, `reduction_orders` into `results.json` | Ch 2, 6, 11, 20 |
| `capture_remarks.sh` | `remarks.txt` (gcc and clang vectorization remarks, the strict-IEEE `dot` assembly) and `mca/*.txt` (llvm-mca timelines) | Ch 4, 6, 13, 20 |
| `count_vector_ops.sh` | `vector_op_counts.txt`: xmm/ymm/zmm and scalar-op counts per function per target | Ch 21 |
| `vmap_jaxpr.py` | `vmap_jaxpr.txt`: jaxprs before and after `jax.vmap`, loop vs vmap timing (needs `jax[cpu]`) | Ch 12 |
| `../../code/simd/check.sh` | `simd_asm.txt`: SVE, RISC-V V and NEON disassembly | Ch 14 |
| `../make_figures.py` | every SVG in `assets/images/` | all |

```bash
python tools/measurements/run_bench.py        # ~2 minutes
python tools/measurements/python_bench.py     # ~30 seconds
bash tools/measurements/capture_remarks.sh
bash tools/measurements/count_vector_ops.sh
python tools/make_figures.py
```

## Experiments and their units

| Key in `results.json` | Unit | Notes |
|---|---|---|
| `width_ladder_L1` | ns/element, GFLOP/s, GB/s | n = 4096 fp32, data in L1 |
| `saxpy_memory_levels` | GB/s | n = 4 K, 64 K, 1 M, 64 M (16 KB to 256 MB per array) |
| `stride_sweep` | ns per useful element | 1 M outputs, stride 1–32 |
| `dot_accumulators_L1` | GFLOP/s | strict, fast-math, intrinsics with 1/2/4/8 accumulators |
| `matmul_loop_order` | GFLOP/s | N = 512 |
| `pdist_c`, `pdist_python` | ns per (pair × dimension) | n = m = 1024, d = 64 (Python loops at n = 96) |
| `peak_fma` | GFLOP/s | register-only, 12 chains, 1/4/8/16 fp32 lanes |
| `amx_peak` | GFLOP/s | bf16 in, fp32 accumulate, 4 tile accumulators |
| `numpy_idioms`, `numpy_call_overhead`, `np_vectorize` | seconds | sizes stated per row |
| `reduction_orders` | ulps of the exact sum | 10⁶ float32 values ~ N(10⁴, 10³²) |
