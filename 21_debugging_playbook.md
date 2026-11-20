---
title: "Chapter 21 — The Debugging & Profiling Playbook"
---

[← Back to Table of Contents](./README.md)

# Chapter 21 — The Debugging & Profiling Playbook

> *"Vectorization fails silently in three directions: it doesn't happen, it happens and doesn't help, or it helps and changes the answer."*

## Overview

This chapter is organised by symptom, not by tool. Each of the four ways vectorization goes wrong gets
a triage table (cause, diagnostic, fix) that points back to the chapter explaining the mechanism. A
tool atlas follows, covering compiler remarks, binary inspection, pipeline models, hardware counters,
GPU profilers and framework logs, with real output from this book's kernels wherever the container
allowed it.

<div class="crt">
<div class="crt-bar">tty8 — bash tools/measurements/count_vector_ops.sh  (gcc 13.3; per function in bench.c)</div>
<div class="crt-screen"><pre>
target   function instrs   xmm   ymm   zmm  scalar ss
scalar   saxpy        14     4     0     0          2
avx2     saxpy        66    14     4     0          3     <span class="dim">vectorized, 256-bit</span>
avx2     leaky        17     6     0     0          1     <span class="bad">no ymm at all: not vectorized</span>
avx2     dot          83    33     3     0         15     <span class="warn">15 scalar adds: in-order "vectorization"</span>
avx2     step_aos    267   102   112     0          4     <span class="warn">267 instrs: shuffle storm (2x slower)</span>
avx512   leaky       122    37     6     7          7     <span class="dim">zmm present: masked version exists</span>
avx512   dot         127    53    15     3         31     <span class="bad">31 scalar adds around 3 zmm ops</span>
avx512fm dot          97    30     7     4          7     <span class="dim">-ffast-math: real vector reduction</span>
avx512   prefix       14     3     0     0          1     <span class="bad">recurrence: scalar everywhere</span>
</pre></div>
</div>
<div class="genre-caption">The cheapest diagnostic there is: disassemble and count register widths per function. No profiler or hardware counters needed. Each red line matches a slowdown measured in Chapters 7–9.</div>

---

## Symptom A — "It didn't vectorize"

| Likely cause | Diagnostic | Fix | Chapter |
|---|---|---|:---:|
| floating-point reduction | remark: *cannot prove it is safe to reorder floating-point operations* | `omp simd reduction`, partial sums, narrow FP flags | [6](./06_dependences.md) |
| true recurrence | remark: *value that could not be identified as reduction*; *unsupported use* | parallel scan or accept scalar | [10](./10_technique_catalogue.md) |
| possible aliasing | remark mentions runtime checks or aliasing | `restrict`, local copies of pointers, fewer pointer arguments | [6](./06_dependences.md) |
| branch the compiler will not speculate | *control flow in loop* | masks (AVX-512/SVE target), `-fno-trapping-math`, explicit selects | [8](./08_control_flow.md) |
| math call with `errno` | *statement clobbers memory* | `-fno-math-errno`, vector math library | [6](./06_dependences.md) |
| opaque function call | *call instruction cannot be vectorized* | inline, `declare simd` | [13](./13_auto_vectorization.md) |
| non-unit stride / 2-D indexing | *not suitable for strided load*, *complicated access pattern* | interchange, re-layout, transpose | [7](./07_memory_layout.md) |
| uncountable loop | *loop control flow is not understood* | compute the trip count first; remove `break` | [8](./08_control_flow.md) |
| wrong target | binary has only `xmm` | `-march=…`, multi-versioning | [13](./13_auto_vectorization.md) |
| Python-level loop | profiler shows time in the interpreter | array idioms, `vmap`, compile | [11](./11_array_programming.md), [12](./12_vmap.md) |

---

## Symptom B — "It vectorized, but it isn't faster"

| Likely cause | Diagnostic | Fix | Chapter |
|---|---|---|:---:|
| memory-bound | roofline: dot on a slanted line; time scales with bytes, not instructions | fuse passes, smaller types, tiling, better layout | [9](./09_profitability.md) |
| strided / gathered access | memory strip; cache-line count per element | layout change (SoA), interchange, shuffle-based lookup | [7](./07_memory_layout.md) |
| shuffle storm | instruction count jumps (`step_aos` AVX2: 23 → 267) | SoA / AoSoA, different loop order | [7](./07_memory_layout.md) |
| in-order reduction | many scalar `ss` ops next to vector ops (`dot`) | allow reassociation; multiple accumulators | [6](./06_dependences.md), [14](./14_explicit_simd.md) |
| latency chain | piano roll / llvm-mca: `===` staircase; far below both roofs | more independent accumulators (latency × ports) | [4](./04_seeing_vectors.md), [9](./09_profitability.md) |
| tails dominate | short trip counts; per-call time flat in $n$ | masked tails, padding, batching calls | [8](./08_control_flow.md) |
| divergence | mask utilisation; GPU branch efficiency | sort/bin by branch, compaction | [8](./08_control_flow.md) |
| Amdahl | the loop is a small share of total time | profile the whole program | [9](./09_profitability.md) |
| dispatch overhead | NumPy calls on tiny arrays; many small GPU kernels | bigger calls, fusion, CUDA Graphs | [11](./11_array_programming.md), [12](./12_vmap.md) |
| frequency reduction | lower clocks under 512-bit load (older parts) | measure; try 256-bit preference | [9](./09_profitability.md) |

---

## Symptom C — "It's faster, but the results changed"

| Likely cause | Diagnostic | Fix | Chapter |
|---|---|---|:---:|
| reassociated reduction | differences of a few ulps relative to partial-sum magnitude | expected; compare against float64 / `math.fsum`, set tolerance accordingly | [6](./06_dependences.md) |
| `-ffast-math` removed a NaN/Inf check | `isnan` tests silently pass; NaNs propagate differently | narrow flags; keep fast-math out of files with special-value logic | [6](./06_dependences.md) |
| denormals flushed to zero | tiny values become exactly 0 | accept it, or disable FTZ/DAZ for that code | [14](./14_explicit_simd.md) |
| array semantics changed a recurrence | NumPy rewrite differs from the loop after the first element | the loop had a flow dependence: use a scan | [6](./06_dependences.md) |
| tail bug | wrong values only for $n \bmod V \ne 0$ | test lengths 1, V−1, V, V+1, 2V+1 | [8](./08_control_flow.md) |
| lane permutation bug | wrong values in a regular pattern (every 4th, 8th) | draw the register strip; check in-lane vs cross-lane shuffles | [3](./03_lanes_and_registers.md) |
| scatter conflicts dropped updates | totals too small when indices repeat | `np.add.at`, `index_add_`, conflict detection, privatization | [10](./10_technique_catalogue.md) |
| algebraic reformulation cancels | small distances come out negative or zero | clamp, centre data, higher precision | [20](./20_handson_ladder.md) |
| non-determinism | different results run to run (GPU atomics, split-K) | deterministic algorithms flag; fixed reduction order | [15](./15_gpus_simt.md) |

<div class="callout field"><span class="callout-title">Field note</span>
<p>Decide the tolerance from the computation, not from what the test currently gets. For a sum of n terms
with largest partial sum S, a reordered float32 result can differ by roughly n·ε·S (ε ≈ 6 × 10⁻⁸) in the
worst case, and by about √n·ε·S typically. A test that compares with <code>rtol=1e-7</code> against the
final value will flake the day someone changes the vector width.</p>
</div>

---

## Symptom D — "Fast in my benchmark, slow in production"

| Likely cause | Diagnostic | Fix |
|---|---|---|
| **different CPU or a VM that hides features** | `lscpu` / `/proc/cpuinfo` flags in production; `numpy.show_runtime()`; `torch.backends.cpu.get_cpu_capability()` | runtime dispatch; build for the oldest target, or ship multiple builds |
| **different sizes** | production $n$ is small or not a multiple of $V$ | benchmark with production size distributions |
| **cold caches** | benchmark reran on hot data | measure end-to-end, include first-touch |
| **unaligned or non-contiguous inputs** | `x.stride()`, `x.is_contiguous()`, alignment of buffers from other libraries | copy once to an aligned contiguous buffer |
| **thread oversubscription** | many processes each using all cores (OpenMP + data loaders) | set `OMP_NUM_THREADS`, `torch.set_num_threads` |
| **dtype drift** | float64 sneaking in (NumPy defaults) | assert dtypes at the boundary |
| **graph breaks / recompiles** | `torch.compile` logs; JAX retraces on new shapes | stable shapes, bucketing, marking dynamic dims |
| **the benchmark lied** | dead-code elimination, no warm-up, no pinning | print a checksum; warm up; pin; report best and median (Chapter 9) |

<div class="callout warn"><span class="callout-title">Trap</span>
<p>Container and VM platforms can hide CPU features from the guest. A library that dispatches on CPUID
will then silently take its AVX2 or SSE path on hardware that has AVX-512. Log the dispatch level at startup,
as you would log the library version.</p>
</div>

---

## The tool atlas

| Question | CPU tool | GPU tool | Framework-level |
|---|---|---|---|
| Did it vectorize, and why not? | `-Rpass*=loop-vectorize` (clang), `-fopt-info-vec*` (gcc), `-qopt-report` (icx), Compiler Explorer | `nvcc -lineinfo` + SASS in Nsight Compute; `cuobjdump -sass` | `TORCH_LOGS=output_code`, XLA dumps, `jax.make_jaxpr` |
| What instructions ran? | `objdump -d` / `llvm-objdump` (count `zmm`/`ymm`), `perf annotate` | Nsight Compute source/SASS view | profiler kernel names (e.g. which GEMM backend) |
| How many vector FLOPs? | `perf stat -e fp_arith_inst_retired.{scalar_single,128b_packed_single,256b_packed_single,512b_packed_single}` (Intel) | Nsight Compute "SOL", tensor-pipe utilisation | PyTorch profiler FLOP estimates |
| Latency or throughput bound? | `llvm-mca -timeline`, uiCA, VTune microarchitecture analysis | Nsight Compute warp-state stall reasons | — |
| Memory-bound? | VTune memory access, `perf` cache events, measured roofline | Nsight Compute memory workload analysis (sectors/request, DRAM throughput) | profiler memory views |
| Which library path was taken? | `ONEDNN_VERBOSE=1`, `MKL_VERBOSE=1`, `numpy.show_runtime()` | `CUBLASLT_LOG_LEVEL`, kernel names | `torch.backends.*` flags |

The `perf` counters in the third row are the most direct "did it vectorize?" measure on Intel: they
count retired floating-point instructions by width. `perf` was not available in this book's container
(hardware counters are commonly hidden inside cloud VMs), which is why the binary-inspection script
above exists. Use the counters when you can.

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>Every diagnostic in this chapter answers one of the six questions of Chapter 1. Which question does your
team have <em>no</em> tool for today? For many ML teams it is question 3, "how does the data arrive?": nobody
looks at strides, layouts or sectors per request until a kernel is mysteriously slow.</p>
</div>

---

## A ten-minute triage routine

1. **Profile the whole program.** Is the hot spot your code, a library call, or the interpreter? (Amdahl first.)
2. **Classify the hot loop's species** (Chapter 2) and **compute its intensity** (Chapter 9). Place it on the roofline.
3. **Ask the compiler** for remarks on that loop. If it did not vectorize, the remark names the chapter to read.
4. **Inspect the binary** for vector registers and scalar ops in the loop.
5. **Model the pipeline** (`llvm-mca` / uiCA) if it is compute-bound but slow.
6. **Fix the cheapest cause**: code shape and layout before flags, flags before intrinsics.
7. **Differential-test** against the scalar reference with tail lengths and a justified tolerance.
8. **Re-measure** with pinning, warm-up and production sizes; record the machine.

---

## Key Takeaways

- Organise debugging by symptom: not vectorized, vectorized but not faster, faster but different, fast only in the benchmark.
- Compiler remarks are the first tool for symptom A; each remark maps to a chapter's mechanism.
- Counting register widths per function in the disassembly is a profiler-free check that caught every failure measured in this book.
- Symptom B is usually memory, shuffles, latency chains or tails; the roofline and the piano roll separate them.
- Result changes after vectorization are usually expected reassociation; set tolerances from the computation, and test tails explicitly.
- Production differs from benchmarks in CPU features, sizes, cache state, layouts, dtypes and threads; log the dispatch level.

## Check Yourself

<details class="check"><summary>A function's disassembly has many <code>zmm</code> instructions and also ~30 scalar <code>vaddss</code>. What do you suspect?</summary>
An in-order floating-point reduction: the multiplies were vectorized but the additions were kept sequential to preserve IEEE semantics (like this book's AVX-512 <code>dot</code>). Allow reassociation or write partial sums.
</details>

<details class="check"><summary>Your vectorized kernel's error versus a float64 reference is 3 × 10⁻⁵ relative to the result, for a sum of 10⁶ terms whose partial sums are 100× larger than the result. Bug or expected?</summary>
Expected. Relative to the partial-sum magnitude the error is 3 × 10⁻⁷, well inside the typical √n·ε bound for float32 with n = 10⁶ (√10⁶ × 6 × 10⁻⁸ = 6 × 10⁻⁵). Cancellation (partial sums 100× the result) is what amplifies it to 3 × 10⁻⁵ relative to the final value.
</details>

<details class="check"><summary>The same Python service is 3× slower on a new cloud instance type with the same core count. Two things to check first?</summary>
The CPU features visible inside the VM (and which SIMD path NumPy/PyTorch/oneDNN dispatched to), and thread settings (oversubscription or a lower default thread count). Then memory bandwidth per core.
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Intel 64 and IA-32 Optimization Reference Manual; Intel PMU event reference (FP_ARITH_INST_RETIRED) | Intel | intel.com; perfmon-events.intel.com | Floating-point width counters |
| Intel VTune Profiler User Guide — Vectorization and Code Insights | Intel | intel.com | Vectorization advisor |
| LLVM `-Rpass` remarks; llvm-mca documentation | LLVM Project | llvm.org/docs | Remarks and pipeline modelling |
| uiCA | Abel & Reineke, ICS 2022 | uica.uops.info | Throughput prediction |
| NVIDIA Nsight Compute Kernel Profiling Guide | NVIDIA | docs.nvidia.com/nsight-compute | Memory workload and warp-state analysis |
| Accuracy and Stability of Numerical Algorithms, ch. 4 | Higham, SIAM 2002 | — | Summation error bounds for tolerances |
| PyTorch reproducibility notes (`torch.use_deterministic_algorithms`) | PyTorch | pytorch.org/docs | Non-determinism from atomics and algorithm choice |
| This book's tools | `tools/measurements/count_vector_ops.sh`, `remarks.txt`, `mca/` | [repo](https://github.com/karam-nus/vectorization/tree/main/tools/measurements) | Real diagnostic output |

---

**Next:** [Chapter 22 — Questions Worth Pondering & The Decision Guide](./22_questions_worth_pondering.md) — the questions to carry into your own designs.
