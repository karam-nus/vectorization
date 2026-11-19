# The book's companion code

Everything here backs a number, a figure or a lab in the book. Three parts:

| Folder | What it is | Run |
|---|---|---|
| `vzlab/` | small, tested NumPy implementations of the book's ideas; each interactive lab mirrors one function | `pytest` (67 tests) |
| `bench/` | C kernels compiled five ways (scalar, SSE2, AVX2, AVX-512, AVX-512 + fast-math), AVX-512 intrinsics, and an Intel AMX microbenchmark | via `tools/measurements/run_bench.py` |
| `simd/` | the same saxpy for Arm SVE, RISC-V V and NEON, cross-compiled and disassembled without the hardware | `bash simd/check.sh` |

```bash
cd code
pip install -r requirements.txt
pytest -q
```

Requirements: Python 3.10+ and NumPy for `vzlab`; gcc ≥ 13 (or clang ≥ 17) on x86-64 for `bench/`; clang
with the AArch64 and RISC-V targets for `simd/` (any recent LLVM build has them).

## `vzlab` module ↔ chapter map

| Module | Contents | Chapters | Lab |
|---|---|:---:|---|
| `patterns.py` | loop vs vectorized: pairwise distances (four rungs), softmax and online softmax, LayerNorm, IoU matrix, one-hot (broadcast and scatter), moving average via cumsum, histogram, leaky ReLU, linear recurrence as a parallel scan | 2, 10, 11, 18, 20 | — |
| `reduce.py` | float32 summation in sequential, lane-split, pairwise and Kahan orders; ulp error | 6 | `reduce` |
| `deps.py` | dependence distance, the d ≤ 0 or d ≥ V rule, brute-force scalar-vs-chunked simulation, GCD test | 6 | `deps` |
| `broadcast.py` | NumPy broadcasting rules with per-axis explanations | 11 | `broadcast` |
| `perfmodel.py` | roofline, ridge point, Amdahl, tail strategies, branch utilisation, accumulators needed | 8, 9 | `roofline`, `amdahl`, `lanes`, `mask` |
| `memory.py` | cache lines and GPU sectors touched by strided / AoS access | 7, 15 | `stride` |
| `systolic.py` | cycle-level output-stationary systolic array | 16 | `systolic` |

## Tests

| File | What it checks |
|---|---|
| `tests/test_patterns.py` | every vectorized idiom equals its loop; the GEMM distance trick's float32 cancellation; online softmax; the `h[idx] += 1` duplicate trap; the parallel scan |
| `tests/test_reduce_deps.py` | reordered sums differ and the tree is more accurate; the legality rule equals simulation for 36 (R, W, V) cases; GCD test |
| `tests/test_models.py` | broadcasting vs `np.broadcast_shapes`; roofline, Amdahl, tails, branch utilisation; cache-line and sector counts; the systolic array equals `A @ B` with K + R + C − 2 cycles |

## `bench/`

| File | Contents |
|---|---|
| `bench.c` | saxpy, leaky ReLU, strided read, dot, prefix sum, AoS/SoA particle step, matmul in ijk/ikj order, pairwise distances (naive and loop-reordered), register-only FMA peak at 1/4/8/16 lanes, and AVX-512 intrinsics (dot with 1/2/4/8 accumulators, masked leaky ReLU, gather) |
| `amx_peak.c` | four independent `TDPBF16PS` tile multiplies; needs Linux ≥ 5.16 and a CPU with AMX |

Both print one JSON object per run. Do not run them by hand to quote numbers: `tools/measurements/run_bench.py`
pins the core, repeats, and records the machine alongside the results.
