---
title: "Chapter 0 — Why Think About Vectorization?"
---

[← Back to Table of Contents](./README.md)

# Chapter 0 — Why Think About Vectorization?

> *"A modern core can finish sixteen multiply-adds in the time it takes to decide what to do next. Most code asks it to do one."*

## Overview

Every processor that matters for machine learning, from a phone NPU to a GPU to a TPU pod, is built
around one bargain: **pay for one decision, apply it to many numbers.** Vectorization is the act of
writing (or transforming) a computation so it can accept that bargain. This chapter makes the case
in three measured gaps: arithmetic you leave unused, energy you spend on bookkeeping, and
interpreter overhead you pay per element. Then it says when *not* to care.

<div class="receipts">
<div class="receipt">
<div class="r-head">SCALAR CORE &amp; CO.</div>
<div class="r-sub">1 x fp32 fused multiply-add</div>
<div class="r-rule"></div>
<div class="r-line"><span>instruction fetch + decode</span><span>~70 pJ</span></div>
<div class="r-line muted"><span>&nbsp;&nbsp;sched., rename, regs</span><span>(incl.)</span></div>
<div class="r-line hl"><span>fp32 multiply</span><span>3.7 pJ</span></div>
<div class="r-line hl"><span>fp32 add</span><span>0.9 pJ</span></div>
<div class="r-rule"></div>
<div class="r-total"><span>TOTAL / element</span><span>~75 pJ</span></div>
<div class="r-line muted"><span>of which arithmetic</span><span>6 %</span></div>
<div class="r-foot">order-of-magnitude, 45 nm-era figures<br>Horowitz 2014 · Dally et al. 2008</div>
<div class="r-barcode"></div>
</div>
<div class="receipt">
<div class="r-head">VECTOR CORE &amp; CO.</div>
<div class="r-sub">1 x 16-lane fp32 FMA (one instruction)</div>
<div class="r-rule"></div>
<div class="r-line"><span>instruction fetch + decode</span><span>~70 pJ</span></div>
<div class="r-line muted"><span>&nbsp;&nbsp;shared by 16 lanes</span><span>4.4 pJ ea.</span></div>
<div class="r-line hl"><span>16 x fp32 multiply</span><span>59 pJ</span></div>
<div class="r-line hl"><span>16 x fp32 add</span><span>14 pJ</span></div>
<div class="r-line muted"><span>wider reg. file + wiring</span><span>extra*</span></div>
<div class="r-rule"></div>
<div class="r-total"><span>TOTAL / element</span><span>~9 pJ*</span></div>
<div class="r-line muted"><span>of which arithmetic</span><span>~50 %</span></div>
<div class="r-foot">*wide datapaths cost more per instruction;<br>the ratio, not the digits, is the point</div>
<div class="r-barcode"></div>
</div>
</div>
<div class="genre-caption">Two receipts for the same work. The overhead line item is paid once per instruction, so a vector instruction splits it sixteen ways. This is why every accelerator is a vector or matrix machine.</div>

---

## Gap 1 — The arithmetic you leave on the table

A single core of the Xeon this book measures on (Emerald Rapids, one core pinned, `gcc 13.3`) can
sustain these floating-point rates when nothing but registers are involved
([`tools/measurements`](https://github.com/karam-nus/vectorization/blob/main/tools/measurements/README.md),
experiment `peak_fma`):

| Code shape | Lanes per instruction | Measured fp32 peak | Share of the core |
|---|:---:|:---:|:---:|
| **Scalar** FMA chains | 1 | 12.9 GFLOP/s | 7 % |
| **SSE**-width vectors | 4 | 51.7 GFLOP/s | 28 % |
| **AVX2**-width vectors | 8 | 103.1 GFLOP/s | 55 % |
| **AVX-512** vectors | 16 | 185.8 GFLOP/s | 100 % |

The ratio 185.8 / 12.9 = **14.4×** is the width gap: the core has the same two FMA units either way,
and each one is sixteen lanes wide. Scalar code drives one lane of each. Nothing else about the
program changed: same clock, same core, same compiler.

<div class="gauges">
<div class="gauge bad" style="--v:0.0004"><div class="gauge-dial"></div><div class="gauge-val">0.04 %</div><div class="gauge-lbl">Python loops<br>(pairwise distance)</div></div>
<div class="gauge bad" style="--v:0.069"><div class="gauge-dial"></div><div class="gauge-val">7 %</div><div class="gauge-lbl">best possible<br>scalar C</div></div>
<div class="gauge warn" style="--v:0.25"><div class="gauge-dial"></div><div class="gauge-val">25 %</div><div class="gauge-lbl">matmul, auto-vectorized<br>(no blocking)</div></div>
<div class="gauge good" style="--v:0.376"><div class="gauge-dial"></div><div class="gauge-val">38 %</div><div class="gauge-lbl">dot product, AVX-512,<br>4 accumulators (load-bound)</div></div>
<div class="gauge good" style="--v:1"><div class="gauge-dial"></div><div class="gauge-val">100 %</div><div class="gauge-lbl">register-only<br>AVX-512 FMAs</div></div>
</div>
<div class="genre-caption">Share of one core's fp32 peak (185.8 GFLOP/s) actually used, measured. The dot product is capped by loads, not lanes; the matmul by missing cache blocking (Chapter 18).</div>

On a GPU the same gap reappears one level up. An H100's tensor cores deliver about 7× the dense
FP16 throughput of its ordinary CUDA cores (about 15× their FP32 rate); code that never reaches a
tensor core leaves that on the table however many threads it launches (NVIDIA H100 whitepaper, 2022).

---

## Gap 2 — The energy you spend deciding

Arithmetic is cheap; *deciding which arithmetic to do* is not. Horowitz's ISSCC 2014 keynote put
numbers on the parts of a 45 nm processor. A 32-bit floating-point add costs about 0.9 pJ and a
multiply about 3.7 pJ. Fetching a 64-bit word from an 8 KB cache costs about 10 pJ, and from DRAM
1.3 to 2.6 nJ. Dally and colleagues estimated that supplying an instruction to a conventional
processor (fetch, decode, schedule, register access) costs on the order of 70 pJ, many times
the arithmetic it controls.

Those two facts produce the receipts above, and they explain the shape of modern hardware better
than any other argument:

- **SIMD** shares one instruction's overhead across 4–64 lanes.
- **SIMT** (GPUs) shares one instruction across a 32-thread warp.
- **Tensor cores and systolic arrays** share one instruction across a whole matrix tile: thousands of
  multiply-adds per decision (Chapter 16).
- **Data movement** is the next bill: an operand fetched from DRAM costs hundreds of times the
  multiply that uses it, which is why Chapter 7 says *memory is the real vector*.

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>If the instruction is the expensive part and the multiply is nearly free, what is the cheapest
possible machine for a workload that is <em>only</em> matrix multiplication? Hold that thought until
Chapter 16, where the answer turns out to be a grid of multipliers with no instruction stream at all.</p>
</div>

---

## Gap 3 — The interpreter in the loop

Python adds a third bill that has nothing to do with hardware: every `a + b` on two Python floats
is a dynamic type check, a method lookup, an object allocation and a reference-count update. The
companion code measures the same computation, pairwise squared distances between 1024 points in 64
dimensions, at eight levels of the software stack ([Chapter 20](./20_handson_ladder.md)):

| Implementation | ns per (pair × dim) | Relative to Python loops |
|---|:---:|:---:|
| **Python lists**, three nested loops (n = 96) | 42.1 | 1× |
| **NumPy**, loop over rows | 1.18 | 36× |
| **C**, `-O2`, scalar | 0.488 | 86× |
| **NumPy**, one GEMM via $\lVert x\rVert^2+\lVert y\rVert^2-2XY^\top$ | 0.0863 | 488× |
| **C**, loops reordered, AVX-512 | 0.0505 | **834×** |

Read the table as two separate wins. Leaving the interpreter (Python → C) buys 86×. Filling the
lanes (scalar C → vectorized C) buys another 9.7×. A NumPy call does both for you, but only for the
work *inside* the call: the row loop still pays Python once per row. Leiserson et al. (2020) told the
same story for 4096 × 4096 matrix multiplication: going from Python to an optimised, vectorized,
parallel C version was worth about 60,000×. In their table, vectorization alone was one of the
largest single steps.

<div class="callout myth"><span class="callout-title">Myth</span>
<p>"Vectorized" means "uses NumPy". In NumPy the word means <em>no Python loop per element</em>, an
interpreter fact. In a compiler it means <em>SIMD instructions</em>, a hardware fact. A NumPy call can be
vectorized in the first sense and still run scalar machine code inside. A C loop can avoid the
interpreter entirely and still be scalar. This book uses the word for both and always says which.</p>
</div>

---

## Vectorization is a way of thinking, not an instruction set

The same move, *one decision applied to many data*, shows up at every layer an ML engineer touches.
Each later chapter takes one of them apart:

| Layer | What "one decision, many data" looks like | Chapter |
|---|---|:---:|
| **Your Python** | whole-array expressions, broadcasting, `einsum` | [11](./11_array_programming.md) |
| **Your framework** | `vmap`, `torch.compile`, XLA fusion | [12](./12_vmap.md) |
| **The compiler** | loop and SLP vectorizers turning loops into SIMD | [13](./13_auto_vectorization.md) |
| **The ISA** | SSE/AVX/AVX-512, NEON/SVE, RISC-V V, masks, gathers | [3](./03_lanes_and_registers.md), [14](./14_explicit_simd.md) |
| **The GPU** | 32-thread warps executing in lockstep | [15](./15_gpus_simt.md) |
| **The matrix engine** | tensor cores, systolic arrays, AMX, SME | [16](./16_matrix_engines.md) |
| **The serving system** | batching requests so one weight read serves many users | [19](./19_batching_systems.md) |

That last row is why this matters even if you never write a kernel. Continuous batching in an LLM
server is vectorization at the scale of a datacenter. The batch dimension plays the part of the
lane, and the weight read is the instruction overhead it amortises.

---

## Why ML engineers in particular should care

1. **Your shapes are hardware decisions.** Head dimensions of 64 or 128, hidden sizes that are
   multiples of 128, vocabulary sizes padded to a multiple of 64: these are lane and tile
   constraints wearing a modelling hat (Chapter 17).
2. **The library stops where your idea starts.** `torch.matmul` is vectorized by experts. Your custom
   loss, your sampling routine, your data augmentation and your quantization kernel are not, until you
   make them so.
3. **Numerics change when you vectorize.** Reordering a floating-point sum changes its value
   ([Chapter 6](./06_dependences.md)): 128 ulps for the in-order sum of a million float32 values in
   our measurement, 11–84 ulps for SIMD-style orders. That matters for reproducibility, for
   gradient checks and for bit-exact comparisons between a reference and a kernel.
4. **Chips are designed around it.** If you work anywhere near model–hardware co-design, the
   questions "how wide?", "how are masks done?" and "how is data fed?" *are* the architecture
   (Chapter 17).

---

## When not to care

Vectorization is a ceiling-raiser, and it only matters when that ceiling is the one you hit.

| Situation | Why vectorizing won't help much | What to do instead |
|---|---|---|
| **Memory-bound** loop over data in DRAM | measured saxpy from DRAM: AVX-512 is 1.8× scalar but only 1.14× SSE2 | reduce bytes: fuse, tile, quantize ([Ch 9](./09_profitability.md)) |
| **Tiny n**, called rarely | setup, tail and dispatch costs dominate | leave it scalar, or batch the calls |
| **Pointer chasing**, trees, hash maps | no contiguous lanes to fill | change the data structure first ([Ch 7](./07_memory_layout.md)) |
| **I/O or network bound** | the CPU is waiting, not computing | profile the wait |
| **Already a library call** | `@`, `conv2d`, `softmax` are vectorized | check you are not wrapping them in a Python loop |

<div class="callout field"><span class="callout-title">Field note</span>
<p>The first profiler run decides which table you are in. If the hot spot is a library call, your job is
to call it fewer times with bigger inputs (batching). If it is your own loop, the rest of this book applies.
If it is waiting on memory, start at Chapter 7 and Chapter 9 before touching a single intrinsic.</p>
</div>

---

## Key Takeaways

- One core here has a 14.4× gap between scalar and AVX-512 peak with the same clock and units; scalar code uses about 7 % of it.
- Instruction supply costs an order of magnitude more energy than the arithmetic it controls; vector, SIMT and matrix hardware all exist to amortise it.
- In Python the first win is leaving the interpreter (86× here); filling the lanes is a second, separate win (another 9.7×).
- "Vectorized" means two different things — no Python loop, or SIMD instructions — and code can be one without the other.
- Vectorization raises the compute ceiling only; memory-bound code gains little (1.8× for saxpy from DRAM).
- Reordering arithmetic to fill lanes changes floating-point results, so vectorization is a numerical decision as well as a performance one.

## Check Yourself

<details class="check"><summary>A colleague's NumPy code has no Python loops. Is it necessarily using SIMD instructions?</summary>
No. It avoids the interpreter per element, which is usually the bigger win, but whether the machine code inside each NumPy call uses SIMD depends on the operation, the dtype, the memory layout and NumPy's CPU dispatch. Strided views, object dtypes and some reductions fall back to scalar loops.
</details>

<details class="check"><summary>Why does AVX-512 give 9× on saxpy from L1 but only 1.8× from DRAM?</summary>
From DRAM the loop is limited by bytes per second (about 21 GB/s on one core here), not by instructions. Wider registers issue fewer instructions for the same bytes, which helps only until the memory system saturates. SSE2 already gets within 12 % of the AVX-512 rate.
</details>

<details class="check"><summary>Using the receipts, why is a systolic array more energy-efficient than a SIMD unit for matrix multiplication?</summary>
A SIMD instruction amortises the ~70 pJ instruction overhead across 16 lanes; a systolic array performs thousands of multiply-adds per control decision and passes operands between neighbours instead of re-reading a register file, so both instruction and data-movement overheads per FLOP shrink further.
</details>

<details class="check"><summary>Name two reasons an ML researcher who never writes C should still care.</summary>
Model shapes are constrained by lane and tile widths (head dimension, hidden size padding); and batching in training and serving is vectorization at system scale — it decides throughput and cost per token. Custom losses and data pipelines written as Python loops are a third.
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Computing's Energy Problem (and what we can do about it) | Horowitz, ISSCC 2014 | doi:10.1109/ISSCC.2014.6757323 | Energy per add, multiply, cache and DRAM access at 45 nm |
| Efficient Embedded Computing | Dally et al., IEEE Computer 2008 | doi:10.1109/MC.2008.224 | ~70 pJ instruction-supply overhead versus arithmetic |
| There's plenty of room at the Top | Leiserson et al., Science 2020 | doi:10.1126/science.aam9744 | Python → optimised C matrix multiply speedup chain |
| NVIDIA H100 Tensor Core GPU Architecture | NVIDIA, 2022 | nvidia.com whitepaper | Tensor-core vs CUDA-core throughput gap |
| This book's measurements | `tools/measurements/results.json` | [repo](https://github.com/karam-nus/vectorization/tree/main/tools/measurements) | `peak_fma`, `saxpy_memory_levels`, `pdist_*`, `reduction_orders` |

---

**Next:** [Chapter 1 — The Whole Field on One Page](./01_grand_overview.md) — the six questions every vectorization decision answers, and the map for the rest of the book.
