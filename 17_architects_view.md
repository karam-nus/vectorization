---
title: "Chapter 17 — The Architect's View: Designing Vector Hardware"
---

[← Back to Table of Contents](./README.md)

# Chapter 17 — The Architect's View: Designing Vector Hardware

> *"Every lane you add must be fed, wired, powered and, above all, used. The last requirement is the one that is not on the floorplan."*

## Overview

The previous chapters took hardware as given. This one asks the architect's question: given a
workload, how wide should the vectors be, how should width be exposed, what data types should lanes
support, and how much of the die should go to a matrix engine instead? The answers are trade-offs
between arithmetic density and four costs that grow with width: feeding, wiring, power, and
utilisation in real code. The chapter is for ML engineers who sit in hardware co-design discussions
and for anyone who wants to know why the chips in the [Atlas](./atlas.md) look the way they do.

<div class="comic" style="--panels:4">
<div class="panel"><span class="cap">Design review, room 4B</span><span class="who">ARCHITECT</span><div class="bubble">Next gen: double the vector width to 1024 bits. Twice the FLOPs, same frequency.</div></div>
<div class="panel blue"><span class="who">COMPILER LEAD</span><div class="bubble">Our median vectorized loop has 37 iterations. At 32 lanes most of that is tail.</div><span class="who">ML RESEARCHER</span><div class="bubble r">And our attention kernels are bandwidth-bound. Will L1 feed 2× more?</div></div>
<div class="panel pink"><span class="who">ARCHITECT</span><div class="bubble">…Two 128-byte loads per cycle, plus a bigger shuffle crossbar.</div><span class="sfx">AREA!</span></div>
<div class="panel green"><span class="who">EVERYONE</span><div class="bubble r">What if the width were a run-time property, and the big FLOPs went into a matrix unit?</div><span class="cap">Approximately how SVE + SME happened.</span></div>
</div>
<div class="genre-caption">A dramatisation, not a transcript: the trade-offs in this chapter, argued by the three people who each pay for one of them. The "37 iterations" is illustrative.</div>

---

## What grows when you widen the vector

Doubling $w$ doubles the peak arithmetic per instruction. Everything below must grow too, or the extra
lanes sit idle:

| Component | How its cost scales with $V$ | Why it must grow | If it does not |
|---|---|---|---|
| **Datapath** (multipliers, adders) | ~linear | one unit per lane | n/a: this is the point |
| **Register file** | linear in width × ports | each vector register is $w$ bits; FMA needs 3 reads + 1 write | register-file energy dominates per op |
| **L1 load/store bandwidth** | linear | an FMA consuming two loaded operands needs $2w$ bits per cycle | memory-bound below the ridge (Chapter 9) |
| **Cross-lane permute network** | $O(V \log V)$ to $O(V^2)$ wires | shuffles, reductions, gathers | slow or restricted shuffles (in-lane only) |
| **Power and current delivery** | linear in active lanes | all lanes switch together | frequency reduction under heavy use (Chapter 9) |
| **Context-switch state** | linear | the OS saves every register | larger save area; AMX needed OS opt-in (Chapter 16) |
| **Lane utilisation in real code** | *falls* with $V$ | tails, divergence, short loops | the extra lanes are mostly masked |

The last row is the one that does not appear on a floorplan. From Chapters 8 and 9:

- **Tails**: with masked tails a loop of $n$ iterations takes $\lceil n/V \rceil$ instructions. For $n = 37$:
  $V = 16$ needs 3 (12.3× fewer instructions, 77 % lane utilisation); $V = 32$ needs 2 (18.5×, 58 %).
  Doubling the width bought 1.5×, not 2×.
- **Divergence**: the probability that a chunk needs both sides of a branch, $1 - p^V - (1-p)^V$, rises toward 1 as $V$ grows.
- **Amdahl**: the scalar fraction of the program does not shrink at all.

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>GPUs have 32-wide warps but thousands of them; CPUs have 16-wide vectors and one instruction stream
per core. Both are about the same total lanes per mm². What does each design assume about how much
<em>independent</em> work your program has? Which assumption does an LLM serving workload satisfy, and
which does a compiler or a database satisfy?</p>
</div>

---

## The width ladder, and why 512 bits is a natural stopping point for CPUs

Three facts made 512 bits a sweet spot for general-purpose cores:

1. **One cache line per load.** A 64-byte cache line is exactly one 512-bit register. A wider register
   would need two lines per load, and an unaligned one would often touch three.
2. **Measured scaling held to 512.** On this book's core the register-only FMA roof scaled from
   51.7 (128-bit) to 103.1 (256-bit) to 185.8 GFLOP/s (512-bit), within 10 % of linear.
3. **But in-cache kernels stopped scaling first.** saxpy from L2 gained nothing from 256 → 512 bits
   (136.6 → 122.8 GB/s), because the L2 cannot deliver lines faster. Wider registers move the
   bottleneck to the memory system unless bandwidth grows with them.

Beyond 512 bits, designers have gone in two directions instead of simply widening registers:
**vector-length-agnostic** ISAs that let each implementation pick its width, and **matrix engines** that
grow reuse rather than width.

---

## Fixed width versus vector-length agnostic

<div class="versus">
<div class="fighter p1"><div class="f-name">FIXED WIDTH</div><div class="f-title">SSE · AVX · AVX-512 · NEON</div>
<div class="f-stat"><span>Compiler simplicity</span><span class="f-bar" style="--v:.9"></span></div>
<div class="f-stat"><span>Binary portability</span><span class="f-bar" style="--v:.2"></span></div>
<div class="f-stat"><span>Tail handling</span><span class="f-bar" style="--v:.4"></span></div>
<div class="f-stat"><span>Width-specific tricks</span><span class="f-bar" style="--v:.95"></span></div>
<div class="f-stat"><span>Future-proof</span><span class="f-bar" style="--v:.15"></span></div>
<div class="f-quote">"I know V at compile time, so I can unroll, tile and shuffle exactly."</div></div>
<div class="vs">VS</div>
<div class="fighter p2"><div class="f-name">LENGTH-AGNOSTIC</div><div class="f-title">Cray · SVE · RISC-V V</div>
<div class="f-stat"><span>Compiler simplicity</span><span class="f-bar" style="--v:.5"></span></div>
<div class="f-stat"><span>Binary portability</span><span class="f-bar" style="--v:.95"></span></div>
<div class="f-stat"><span>Tail handling</span><span class="f-bar" style="--v:1"></span></div>
<div class="f-stat"><span>Width-specific tricks</span><span class="f-bar" style="--v:.35"></span></div>
<div class="f-stat"><span>Future-proof</span><span class="f-bar" style="--v:.9"></span></div>
<div class="f-quote">"Ship one binary. The chip decides how wide today."</div></div>
</div>
<div class="genre-caption">Bars are qualitative judgements. The SVE loop in Chapter 14 was eight instructions with no tail and runs on 128- to 2048-bit hardware; an AVX-512 binary does not run on an AVX2-only CPU at all.</div>

The x86 world's answer to the same problem is **versioning**. AVX10 numbers its feature levels so that
software can target "AVX10.2" rather than a list of AVX-512 subsets, and after a 2024 specification
revision it requires 512-bit support on all cores (Intel AVX10.2 specification; first parts announced,
timing unconfirmed).

---

## Data types are a hardware decision

A multiplier's area grows roughly with the square of the mantissa width; an adder's grows roughly
linearly. That is why lower precision is not just "the same unit with fewer bits", but *more units per mm²*:

| Format | Mantissa bits (incl. implicit) | Relative multiplier area (∝ m²) | Energy per multiply, 45 nm (Horowitz 2014) |
|---|:---:|:---:|:---:|
| **fp32** | 24 | 1 | 3.7 pJ |
| **fp16** | 11 | ~0.21 | 1.1 pJ |
| **bf16** | 8 | ~0.11 | — |
| **fp8 E4M3** | 4 | ~0.03 | — |
| **int8** | 8 (integer) | ~0.11 | 0.2 pJ |

The area column is a first-order estimate, not a measured layout. Combined with Chapter 3's lane
arithmetic (halving bits doubles lanes per register) and Chapter 9's bandwidth arithmetic (halving bits
halves bytes per operand), this is why every accelerator generation adds a narrower format: fp16 → bf16 →
fp8 → fp4 with block scaling (NVFP4, MXFP4). Each step is a vectorization win three ways, if the model
tolerates it. The quantization book in this knowledge base covers that last condition.

---

## Sizing a matrix engine

For a systolic or tile engine of side $N$:

| Decision | Bigger $N$ gives | Bigger $N$ costs |
|---|---|---|
| **Array size** | reuse per operand $N/2$; fewer instructions | fill/drain $2N - 2$ clocks; utilisation $K/(K + 2N - 2)$ per tile; padding waste on small matrices |
| **Operand bandwidth** | — | $2N$ operands per clock must arrive at the edges |
| **Accumulator precision** | fp32 accumulation keeps long sums accurate | area; fp32 accumulators are much larger than bf16 multipliers |
| **Supported shapes** | flexibility for small or odd matrices | control complexity |

Google's TPU went from a 256 × 256 int8 array (v1) to 128 × 128 bf16 arrays (v2–v5), more of them per
chip, and back to 256 × 256 (v6e) as models and batch sizes grew. NVIDIA's tensor-core tiles grew per
generation as the issuing unit grew from a warp to a CTA. The right size tracks the matrices the
workload actually has.

---

## Co-design: models shaped by hardware, hardware shaped by models

| Model-side choice | The hardware fact behind it |
|---|---|
| hidden sizes and FFN widths in multiples of 64, 128 or 256 | tile shapes, wave quantization (Chapter 15), systolic array sizes |
| attention head dimension 64 or 128 | tensor-core $k$ dimension, shared-memory tile size |
| vocabulary padded to a multiple of 64 or 128 | the final GEMM's $N$ dimension |
| grouped-query and multi-query attention | decode reads the KV cache at ~1 FLOP/B; fewer KV heads means fewer bytes |
| mixture-of-experts with token permutation | GEMMs per expert need contiguous token blocks (Chapter 8) |
| block-scaled 4- and 8-bit formats | lanes per register, bytes per operand, multiplier area |

| Hardware-side choice | The workload fact behind it |
|---|---|
| ever-larger matrix engines | transformer FLOPs are ~all GEMM |
| faster special-function units for `exp` | softmax became the bottleneck between GEMMs (Chapter 18) |
| asynchronous bulk copies (TMA), tensor memory | operands must stream to the matrix unit without occupying threads |
| huge HBM bandwidth per chip | decoding is bandwidth-bound |
| vector units beside matrix units (TPU VPU, AVX-512 next to AMX) | everything that is not GEMM |

---

## An architect's checklist for a workload

Before choosing lanes, tiles or data types, characterise the workload with the same six questions this
book asks of a loop:

1. **Vector length distribution.** What are the trip counts of hot loops? (Short loops penalise wide $V$.)
2. **Dependence structure.** What fraction is maps, reductions, scans, gathers? (Chapter 2's species census.)
3. **Arithmetic intensity per memory level.** Where on the roofline do the kernels sit? (Chapter 9.)
4. **Control-flow coherence.** How often do neighbouring elements take different branches? (Chapter 8.)
5. **Matrix fraction.** What fraction of FLOPs is GEMM-shaped, and with what $M$, $N$, $K$?
6. **Software reach.** Which compilers and libraries will target the new units on day one? A unit no
   compiler targets is dark silicon; AVX-512's slow adoption across fragmented subsets is the cautionary tale.

<div class="callout field"><span class="callout-title">Field note</span>
<p>The cheapest place to make a hardware decision is a model that runs your traces: a roofline per kernel,
a lane-utilisation estimate from real trip counts, a fill/drain model for the matrix unit (the systolic lab's
arithmetic). Every number this book measured on one core is a data point of that kind, and the
methodology in <code>tools/measurements</code> transfers to any simulator.</p>
</div>

---

## Key Takeaways

- Widening vectors scales the datapath linearly but also the register file, load bandwidth, permute network, power and saved state, while utilisation in real code *falls*.
- 512 bits matches one 64-byte cache line per load and scaled near-linearly for register-only work here; in-cache kernels had already stopped scaling at 256 bits from L2.
- Vector-length-agnostic ISAs trade compile-time knowledge of $V$ for binary portability and free tails; x86 answered with versioned feature levels (AVX10).
- Narrower data types are a triple vectorization win: more lanes per register, fewer bytes per operand, smaller multipliers.
- Matrix-engine size trades reuse ($N/2$) against fill/drain, operand bandwidth ($2N$ per clock) and padding waste.
- Model shapes and hardware shapes co-evolve; most "magic numbers" in model configs are tile and lane sizes.

## Check Yourself

<details class="check"><summary>Why does a wider vector need a disproportionately larger permute network?</summary>
An arbitrary permutation lets any output lane take any input lane: a full crossbar has V² connections, and even multi-stage networks need O(V log V) switches and long wires across the whole datapath, while the arithmetic grows only linearly.
</details>

<details class="check"><summary>A workload's hot loops have trip counts uniformly between 8 and 40. Would you rather have V = 16 or V = 64 (masked tails), measured by lane utilisation?</summary>
With V = 64 every loop is a single partially masked instruction: utilisation averages 24/64 ≈ 38 %. With V = 16, a 24-iteration loop takes 2 instructions at 75 % and a 40-iteration loop 3 at 83 %: average utilisation around 70 %. V = 64 still finishes each loop in one instruction, so it is faster in instructions but wastes most of its datapath: the decision depends on whether area or latency is the constraint.
</details>

<details class="check"><summary>Why does a TPU-style chip still need a vector unit next to its systolic arrays?</summary>
Softmax, normalisation, activations, element-wise residuals, reductions and data reshuffling are not matrix multiplies; they need lane-wise and cross-lane vector operations. Without a capable vector unit those parts would bottleneck the matrix engine.
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Computer Architecture: A Quantitative Approach, 6th ed., ch. 4 and 7 | Hennessy & Patterson, 2017 | Morgan Kaufmann | Vector width trade-offs; domain-specific architectures |
| A New Golden Age for Computer Architecture | Hennessy & Patterson, CACM 2019 | doi:10.1145/3282307 | Domain-specific accelerators and co-design |
| Computing's Energy Problem | Horowitz, ISSCC 2014 | doi:10.1109/ISSCC.2014.6757323 | Energy per multiply by precision |
| The ARM Scalable Vector Extension | Stephens et al., IEEE Micro 2017 | doi:10.1109/MM.2017.35 | VLA design rationale |
| Intel AVX10.2 Architecture Specification | Intel, 2024 | intel.com | Versioned feature levels, 512-bit on all cores |
| Ten Lessons From Three Generations Shaped Google's TPUv4i | Jouppi et al., ISCA 2021 | doi:10.1109/ISCA52012.2021.00010 | Matrix-unit sizing, co-design lessons |
| OCP Microscaling Formats (MX) Specification v1.0 | Open Compute Project, 2023 | opencompute.org | Block-scaled low-precision formats |
| This book's measurements | `tools/measurements/results.json` | [repo](https://github.com/karam-nus/vectorization/tree/main/tools/measurements) | `peak_fma`, `saxpy_memory_levels`, `amx_peak` |

---

**Next:** [Chapter 18 — Vectorizing ML Kernels](./18_ml_kernels.md) — softmax, LayerNorm, GEMM, attention and quantized dot products, with everything so far applied.
