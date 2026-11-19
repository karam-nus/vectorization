---
title: "Appendix C — Technique Index"
---

[← Back to Table of Contents](./README.md)

# Appendix C — Technique Index

> *The master table: every element of the periodic table in Chapter 10, one row each. If a chapter and this table disagree, this table is the bug.*

## Overview

Columns follow the technique fact block of the [authoring prompt](./PROMPT.md): family, who applies it,
when it is legal, when it pays, how it breaks, and where the book covers it. *Applied by*: **Y** you
(source), **F** framework or library, **C** compiler, **I** you in intrinsics, **H** hardware.

---

## Loop transformations

| # | Sym | Technique | By | Legal when | Pays off when | Breaks when | Ch |
|:---:|:---:|---|:---:|---|---|---|:---:|
| 1 | Sm | **Strip-mining** | C, I | no dependence with $0<d<V$ | $n \gg V$ | tail forgotten; tiny $n$ | [8](./08_control_flow.md), [10](./10_technique_catalogue.md) |
| 2 | Lv | **Inner-loop vectorization** | C | as Sm; reductions recognised | contiguous, compute-bound inner loop | aliasing, calls, strides | [13](./13_auto_vectorization.md) |
| 3 | Ov | **Outer-loop vectorization** | Y, C | outer iterations independent | inner loop is a reduction or short | outer axis strided (re-layout first) | [10](./10_technique_catalogue.md), [20](./20_handson_ladder.md) |
| 4 | Ma | **Multiple accumulators** | C, I | reassociation allowed | latency-bound reduction | register spills; changed rounding | [9](./09_profitability.md), [14](./14_explicit_simd.md) |
| 5 | Ix | **Loop interchange** | Y, C | no dependence direction reversed | makes inner loop contiguous (22× measured) | breaks another loop's locality | [7](./07_memory_layout.md) |
| 6 | Fs | **Loop fusion** | F, C, Y | no backward dependence between bodies | memory-bound chains: passes → 1 | register pressure; mismatched trip counts | [12](./12_vmap.md), [18](./18_ml_kernels.md) |
| 7 | Fi | **Loop fission** | C, Y | statements separable | isolates a recurrence so the rest vectorizes | extra passes over memory | [6](./06_dependences.md) |
| 8 | Tl | **Tiling / blocking** | F, Y | reordering respects dependences | data reuse; raises intensity | wrong tile size; complexity | [16](./16_matrix_engines.md), [18](./18_ml_kernels.md) |
| 9 | Sk | **Skewing (wavefront)** | Y, C | uniform dependence distances | 2-D recurrences | non-uniform dependences | [4](./04_seeing_vectors.md), [6](./06_dependences.md) |
| 10 | Pe | **Peeling** | C, I | always | alignment of the main loop | short loops | [7](./07_memory_layout.md) |
| 11 | Uj | **Unroll-and-jam** | C, Y | as interchange | register reuse in nested loops (GEMM) | register spills | [18](./18_ml_kernels.md) |

## Data layout

| # | Sym | Technique | By | Legal when | Pays off when | Breaks when | Ch |
|:---:|:---:|---|:---:|---|---|---|:---:|
| 12 | Sa | **AoS → SoA** | Y | always (re-layout) | bulk passes over few fields (9.5× vs AVX2 AoS) | whole-record random access | [7](./07_memory_layout.md) |
| 13 | Ab | **AoSoA / blocked layout** | Y, F | always | both bulk passes and record locality | block ≠ vector width | [7](./07_memory_layout.md) |
| 14 | Pd | **Padding** | Y, F | padded values cannot change results | removes tails, bank conflicts, wave tails | wrong pad value (0 in a max, 0 logits) | [8](./08_control_flow.md), [15](./15_gpus_simt.md) |
| 15 | Al | **Alignment** | Y, C | always | wide loads that would split lines | false alignment hints (UB) | [7](./07_memory_layout.md) |
| 16 | Tr | **In-register transpose** | I | always | small AoS ↔ SoA conversions | shuffle-port bound | [7](./07_memory_layout.md) |
| 17 | Pk | **Panel packing** | F | always | GEMM: contiguous, aligned microkernel reads | small matrices (packing dominates) | [18](./18_ml_kernels.md) |

## Control flow

| # | Sym | Technique | By | Legal when | Pays off when | Breaks when | Ch |
|:---:|:---:|---|:---:|---|---|---|:---:|
| 18 | If | **If-conversion** | C, Y | both sides safe to compute | cheap sides, unpredictable branch (15.6× measured) | expensive rare side; FP-exception rules | [8](./08_control_flow.md) |
| 19 | Mk | **Masking / predication** | H, C, I | masked loads suppress faults | tails, conditional stores | no mask registers (SSE/AVX2/NEON) | [3](./03_lanes_and_registers.md), [8](./08_control_flow.md) |
| 20 | Cp | **Compaction** | H, F, Y | always | heavy work after a selective filter | nearly everything survives | [8](./08_control_flow.md) |
| 21 | Bn | **Binning by branch** | Y | order of work is free | divergent branches, MoE, ray tracing | sort cost exceeds savings | [8](./08_control_flow.md) |
| 22 | Ff | **First-faulting loads** | H, C | ISA support (SVE) | early-exit loops (`strlen`) | ISA without it (use aligned blocks) | [8](./08_control_flow.md) |

## Reductions and scans

| # | Sym | Technique | By | Legal when | Pays off when | Breaks when | Ch |
|:---:|:---:|---|:---:|---|---|---|:---:|
| 23 | Rd | **Partial-sum reduction** | C, I, F | associative operator (floats need permission) | any long reduction | bit-exact requirements | [6](./06_dependences.md) |
| 24 | Ht | **Horizontal tree** | I, C | as Rd | once, after the loop | done every iteration | [3](./03_lanes_and_registers.md) |
| 25 | Sc | **Parallel scan** | F, Y | associative step operator | long recurrences, wide machines | extra work dominates on narrow machines | [10](./10_technique_catalogue.md) |
| 26 | Hg | **Histogram privatization** | Y | commutative updates | bins fit in cache | many bins × many lanes | [10](./10_technique_catalogue.md) |
| 27 | Cd | **Conflict detection** | I, H | ISA support (AVX-512 CD) | scatter-adds with few conflicts | heavy conflicts serialise | [10](./10_technique_catalogue.md) |

## Data movement

| # | Sym | Technique | By | Legal when | Pays off when | Breaks when | Ch |
|:---:|:---:|---|:---:|---|---|---|:---:|
| 28 | Bc | **Broadcast** | H, C, F | always | scalars and shared operands in vector code | — | [3](./03_lanes_and_registers.md), [11](./11_array_programming.md) |
| 29 | Sh | **Shuffle / permute** | I, C | always | rearranging lanes | shuffle-port pressure | [3](./03_lanes_and_registers.md) |
| 30 | Ga | **Gather / scatter** | H | gathers always; scatters without conflicts | heavy arithmetic around indirect access | the gather is the whole loop | [7](./07_memory_layout.md) |
| 31 | Lu | **Shuffle lookup table** | I | table ≤ 16/64 entries | nibble dequant, UTF-8, small LUTs | larger tables | [3](./03_lanes_and_registers.md), [18](./18_ml_kernels.md) |
| 32 | Il | **(De)interleave** | H, I | always | fixed-stride fields (RGB, complex) | prefer SoA when possible | [7](./07_memory_layout.md) |

## Batching

| # | Sym | Technique | By | Legal when | Pays off when | Breaks when | Ch |
|:---:|:---:|---|:---:|---|---|---|:---:|
| 33 | Bt | **Batching** | F, Y | instances independent | shared operand dominates bytes ($I \approx B$) | latency limits; padding | [19](./19_batching_systems.md) |
| 34 | Vm | **vmap** | F | function traceable, shapes static | per-example computations | data-dependent shapes; branches → both sides | [12](./12_vmap.md) |
| 35 | Xi | **Vectorize across instances** | Y | instances independent | inner work hostile (trees, pointer chases) | instances diverge heavily | [2](./02_domains.md) |
| 36 | Sw | **SWAR / bit-slicing** | Y | bitwise or carry-free ops | 1–8-bit data, logic simulation | transposition cost | [10](./10_technique_catalogue.md) |

## Hardware primitives

| # | Sym | Technique | By | Legal when | Pays off when | Breaks when | Ch |
|:---:|:---:|---|:---:|---|---|---|:---:|
| 37 | Mm | **Matrix-engine tiles** | H, F | GEMM-shaped work | large, aligned matrices (AMX 13× AVX-512 measured) | small/odd shapes; non-GEMM work | [16](./16_matrix_engines.md) |
| 38 | Sy | **Systolic dataflow** | H | GEMM / convolution | long streams of tiles | fill/drain on small tiles | [16](./16_matrix_engines.md) |
| 39 | Wd | **Widening dot product** | H, I | accumulator cannot overflow | int8/int4 inference | VNNI sign conventions | [3](./03_lanes_and_registers.md), [18](./18_ml_kernels.md) |
| 40 | Lp | **Lower precision** | Y, F | accuracy survives | more lanes and fewer bytes | outliers, accumulation error | [3](./03_lanes_and_registers.md), [17](./17_architects_view.md) |

## Algorithmic

| # | Sym | Technique | By | Legal when | Pays off when | Breaks when | Ch |
|:---:|:---:|---|:---:|---|---|---|:---:|
| 41 | Rf | **Reformulate as GEMM** | Y | mathematically equivalent | moves work onto BLAS / tensor cores (488× over Python) | cancellation, larger intermediates | [10](./10_technique_catalogue.md), [20](./20_handson_ladder.md) |
| 42 | On | **Online algorithm** | Y, F | state is exactly rescalable | multi-pass memory-bound code | extra transcendental work | [18](./18_ml_kernels.md) |
| 43 | Px | **Polynomial math** | F | accuracy target met | branch-free exp/log/tanh in SIMD | needs range reduction | [18](./18_ml_kernels.md) |
| 44 | Sn | **Sorting network** | F, I | fixed-size sorts | top-k, small sorts in registers | large n (use merge/radix) | [2](./02_domains.md) |

---

**Next:** [Appendix D — Hardware Matrix](./appendix_d_hardware_matrix.md) — every ISA and matrix engine in one table.
