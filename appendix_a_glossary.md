---
title: "Appendix A — Glossary & Symbols"
---

[← Back to Table of Contents](./README.md)

# Appendix A — Glossary & Symbols

> *One term, one definition, one chapter where it lives.*

## Symbols

| Symbol | Meaning | Defined in |
|---|---|:---:|
| $n$ | trip count: number of elements or iterations | [3](./03_lanes_and_registers.md) |
| $w$ | vector register width in bits | [3](./03_lanes_and_registers.md) |
| $e$ | element size in bits | [3](./03_lanes_and_registers.md) |
| $V = w/e$ | lanes per instruction | [3](./03_lanes_and_registers.md) |
| $d = W - R$ | dependence distance in iterations | [6](./06_dependences.md) |
| $f$ | vectorizable fraction of runtime (Amdahl) | [9](./09_profitability.md) |
| $I$ | arithmetic intensity, FLOP per byte from a given memory level | [9](./09_profitability.md) |
| $P_{\text{peak}}$, $B$ | peak compute (FLOP/s) and bandwidth (byte/s) | [9](./09_profitability.md) |
| $I^\ast = P_{\text{peak}}/B$ | ridge point of the roofline | [9](./09_profitability.md) |
| $S$ | speedup over a named baseline | [9](./09_profitability.md) |
| $L$, $T$ | instruction latency, reciprocal throughput (cycles) | [3](./03_lanes_and_registers.md), [9](./09_profitability.md) |
| $M, N, K$ | GEMM dimensions, $C_{M\times N} \mathrel{+}= A_{M\times K} B_{K\times N}$ | [16](./16_matrix_engines.md) |
| $M_R, N_R$ | microkernel register block | [18](./18_ml_kernels.md) |
| $\eta$ | padding efficiency of a batch | [19](./19_batching_systems.md) |

## Terms

| Term | Definition | Chapter |
|---|---|:---:|
| **Accumulator** | a register holding a running partial result; several independent ones hide latency | [14](./14_explicit_simd.md) |
| **Aliasing** | two pointers or views referring to overlapping memory | [6](./06_dependences.md) |
| **AMX** | Intel Advanced Matrix Extensions: 8 tile registers and a tile multiply unit (TMUL) per core | [16](./16_matrix_engines.md) |
| **Anti dependence** | write-after-read between iterations; safe under chunked vector execution | [6](./06_dependences.md) |
| **AoS / SoA / AoSoA** | array of structs, struct of arrays, blocked hybrid | [7](./07_memory_layout.md) |
| **Arithmetic intensity** | FLOPs per byte moved from a memory level | [9](./09_profitability.md) |
| **Bank conflict** | GPU shared-memory accesses to the same bank at different addresses, serialised | [15](./15_gpus_simt.md) |
| **Batching rule** | how a primitive is rewritten when vmap adds a batch axis | [12](./12_vmap.md) |
| **Blend / select** | per-lane choice between two vectors under a mask | [8](./08_control_flow.md) |
| **Broadcast** | one scalar copied to every lane; in array libraries, stretching size-1 axes as stride-0 views | [3](./03_lanes_and_registers.md), [11](./11_array_programming.md) |
| **Cache line** | unit of transfer between memory levels, 64 bytes on most CPUs | [7](./07_memory_layout.md) |
| **Coalescing** | merging a warp's 32 addresses into few 32-byte sectors | [15](./15_gpus_simt.md) |
| **Compaction / compress** | packing active lanes contiguously | [8](./08_control_flow.md) |
| **Continuous batching** | re-forming the serving batch every decode step | [19](./19_batching_systems.md) |
| **Dependence distance** | iterations between a write and a dependent read | [6](./06_dependences.md) |
| **Dispatch (runtime)** | choosing among builds for different ISAs at run time | [14](./14_explicit_simd.md) |
| **Divergence** | lanes or threads of one vector taking different branches | [8](./08_control_flow.md) |
| **Epilogue** | (loop) scalar code after the vector loop for the tail; (GEMM) fused post-processing of the output | [8](./08_control_flow.md), [18](./18_ml_kernels.md) |
| **First-faulting load** | SVE load that stops at the first invalid address and reports progress | [8](./08_control_flow.md) |
| **Flow dependence** | read-after-write between iterations; illegal for $0 < d < V$ | [6](./06_dependences.md) |
| **Fusion** | combining several operations into one pass over memory | [12](./12_vmap.md) |
| **Gather / scatter** | vector load / store with one address per lane | [7](./07_memory_layout.md) |
| **Horizontal operation** | combining lanes of one register (reductions) | [3](./03_lanes_and_registers.md) |
| **If-conversion** | replacing a branch with computation of both sides plus a select | [8](./08_control_flow.md) |
| **Interleave count** | compiler unrolling of the vector body into independent copies | [13](./13_auto_vectorization.md) |
| **Intrinsic** | a C function that maps to one machine instruction | [14](./14_explicit_simd.md) |
| **Lane** | one element position within a vector register | [3](./03_lanes_and_registers.md) |
| **Loop interchange** | swapping the nesting order of loops | [7](./07_memory_layout.md), [10](./10_technique_catalogue.md) |
| **Loop vectorizer** | compiler pass that widens loop iterations into lanes | [13](./13_auto_vectorization.md) |
| **Mask / predicate** | one bit per lane controlling which lanes an instruction affects | [3](./03_lanes_and_registers.md), [8](./08_control_flow.md) |
| **Matrix engine** | hardware performing a small matrix multiply per instruction (tensor cores, AMX, SME, MXU) | [16](./16_matrix_engines.md) |
| **Microkernel** | innermost register-blocked GEMM kernel | [18](./18_ml_kernels.md) |
| **Online algorithm** | one-pass computation carrying rescalable statistics (softmax, variance) | [18](./18_ml_kernels.md) |
| **Operand reuse** | MACs per delivered input operand; $N/2$ for an $N\times N$ tile | [16](./16_matrix_engines.md) |
| **Outer-loop vectorization** | putting an outer, independent loop on the lanes | [10](./10_technique_catalogue.md), [20](./20_handson_ladder.md) |
| **Packing** | copying GEMM panels into the order the microkernel reads them | [18](./18_ml_kernels.md) |
| **Padding** | extending data to a multiple of the lane or tile size with harmless values | [8](./08_control_flow.md), [10](./10_technique_catalogue.md) |
| **Parallel scan** | prefix computation in $O(\log n)$ steps for an associative operator | [10](./10_technique_catalogue.md) |
| **Reassociation** | changing the grouping of additions; changes floating-point results | [6](./06_dependences.md) |
| **Reduction** | combining many values into one with an associative operator | [6](./06_dependences.md) |
| **Ridge point** | intensity where the roofline's slope meets its flat roof | [9](./09_profitability.md) |
| **Roofline** | attainable performance $\min(P_{\text{peak}}, I B)$ | [9](./09_profitability.md) |
| **Sector** | 32-byte unit of GPU memory transactions | [15](./15_gpus_simt.md) |
| **SIMD** | single instruction, multiple data | [1](./01_grand_overview.md) |
| **SIMT** | single instruction, multiple threads: GPU warps | [15](./15_gpus_simt.md) |
| **SLP vectorizer** | compiler pass packing isomorphic straight-line statements | [13](./13_auto_vectorization.md) |
| **SME** | Arm Scalable Matrix Extension: outer-product engine with the ZA array | [16](./16_matrix_engines.md) |
| **SPMD** | single program, multiple data: scalar-looking code mapped to lanes (CUDA, ISPC) | [14](./14_explicit_simd.md) |
| **Strip-mining** | splitting a loop into chunks of $V$ plus a tail | [10](./10_technique_catalogue.md) |
| **Stride** | step between consecutive accessed elements | [7](./07_memory_layout.md) |
| **SVE / RVV** | Arm Scalable Vector Extension / RISC-V Vector extension: vector-length agnostic ISAs | [14](./14_explicit_simd.md), [17](./17_architects_view.md) |
| **SWAR** | SIMD within a register: bitwise lanes in an ordinary integer | [10](./10_technique_catalogue.md) |
| **Systolic array** | grid of MAC units passing operands to neighbours each clock | [16](./16_matrix_engines.md) |
| **Tail** | the last $n \bmod V$ elements | [8](./08_control_flow.md) |
| **Tensor core** | NVIDIA matrix engine issued by a warp, warpgroup or single thread | [15](./15_gpus_simt.md), [16](./16_matrix_engines.md) |
| **Tiling / blocking** | restructuring loops to work on sub-blocks that fit in fast memory | [10](./10_technique_catalogue.md), [16](./16_matrix_engines.md) |
| **VLA** | vector-length agnostic: code that does not encode $V$ | [5](./05_history.md), [17](./17_architects_view.md) |
| **vmap** | automatic batching transform | [12](./12_vmap.md) |
| **VNNI** | x86 int8/int16 dot-product-accumulate instructions (`vpdpbusd`) | [3](./03_lanes_and_registers.md), [18](./18_ml_kernels.md) |
| **Warp / wavefront** | 32 (NVIDIA) or 64/32 (AMD) threads executing one instruction | [15](./15_gpus_simt.md) |
| **Wave quantization** | idle SMs in the last wave of a GPU grid | [15](./15_gpus_simt.md) |

---

**Next:** [Appendix B — Math Reference](./appendix_b_math.md) — every formula in the book on one page.
