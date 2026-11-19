---
title: "Chapter 1 — The Whole Field on One Page"
---

[← Back to Table of Contents](./README.md)

# Chapter 1 — The Whole Field on One Page

> *"Every vectorization decision answers the same six questions. Most bugs come from answering only the last one."*

## Overview

Vectorization looks like a dozen unrelated topics: NumPy broadcasting, compiler flags, AVX-512
intrinsics, CUDA warps, tensor cores, continuous batching. They are one idea seen from different
floors of the same building. This chapter draws the building, names the six questions that every
vectorization decision answers on every floor, and tells you which five chapters to read first.

<div class="blueprint">
<svg viewBox="0 0 760 440" width="760" role="img" aria-label="Cross-section of the vectorization stack as a building">
<g fill="none" stroke="#e3f1ff" stroke-width="1.6">
<path d="M70 70 L380 22 L690 70"/>
<rect x="70" y="70" width="620" height="330"/>
<line x1="70" y1="125" x2="690" y2="125"/><line x1="70" y1="180" x2="690" y2="180"/>
<line x1="70" y1="235" x2="690" y2="235"/><line x1="70" y1="290" x2="690" y2="290"/>
<line x1="70" y1="345" x2="690" y2="345"/>
<rect x="600" y="70" width="60" height="330" stroke-dasharray="5 4"/>
<line x1="20" y1="400" x2="740" y2="400" stroke-width="2.4"/>
</g>
<g fill="#bcdcff" stroke="none"><rect x="610" y="90" width="40" height="12" opacity="0.35"/><rect x="610" y="370" width="40" height="12" opacity="0.35"/></g>
<g font-size="11" fill="#e3f1ff" letter-spacing="0.06em">
<text x="380" y="60" text-anchor="middle">ROOF · SERVING — batching (CH 19)</text>
<text x="86" y="95">L5 · YOUR PYTHON — whole-array expressions, broadcasting, einsum</text><text x="86" y="112" fill="#8fc3f5">CH 11 · "no loop per element"</text>
<text x="86" y="150">L4 · FRAMEWORK — vmap, jit, XLA / Inductor fusion</text><text x="86" y="167" fill="#8fc3f5">CH 12 · "add a batch axis for me"</text>
<text x="86" y="205">L3 · COMPILER — loop + SLP vectorizers, cost models</text><text x="86" y="222" fill="#8fc3f5">CH 13 · "prove it is legal, then widen"</text>
<text x="86" y="260">L2 · ISA — SIMD, vector-length agnostic, SIMT, tiles</text><text x="86" y="277" fill="#8fc3f5">CH 3, 14, 15 · "what one instruction can say"</text>
<text x="86" y="315">L1 · MICROARCHITECTURE — ports, latency, register files</text><text x="86" y="332" fill="#8fc3f5">CH 9, 14 · "how many per cycle, really"</text>
<text x="86" y="370">B · SILICON — lanes, matrix engines, systolic arrays</text><text x="86" y="387" fill="#8fc3f5">CH 16, 17 · "what the transistors were built for"</text>
</g>
<g font-size="10" fill="#e3f1ff" letter-spacing="0.08em"><text x="630" y="240" text-anchor="middle" transform="rotate(-90 630 240)">MEMORY SHAFT · DATA MOVEMENT · CH 7</text></g>
<g stroke="#e3f1ff" stroke-width="1" fill="none"><line x1="40" y1="70" x2="40" y2="400"/><line x1="34" y1="70" x2="46" y2="70"/><line x1="34" y1="400" x2="46" y2="400"/></g>
<g font-size="10" fill="#bcdcff"><text x="30" y="235" text-anchor="middle" transform="rotate(-90 30 235)">ABSTRACTION ↑ · CONTROL ↓</text></g>
</svg>
<div class="bp-notes">NOTE 1: The same idea — one decision, many data — appears on every floor. NOTE 2: The memory shaft serves all floors; most "vectorization didn't help" reports are a shaft problem, not a floor problem.</div>
<div class="bp-block"><span>DWG NO.</span><span>VZ-01</span><span>TITLE</span><span>THE VECTORIZATION STACK</span><span>SCALE</span><span>NTS</span><span>SHEET</span><span>1 OF 1</span></div>
</div>

---

## The six questions

Whatever floor you work on, a vectorization decision is complete only when all six are answered.
The rest of the book is organised around them.

| # | Question | What it really asks | Fails as | Home |
|:---:|---|---|---|:---:|
| 1 | **What is the vector?** | Which axis of the problem becomes the lanes: elements, batch, heads, channels, pixels, requests? | the wrong axis is short, strided or dependent | [3](./03_lanes_and_registers.md), [10](./10_technique_catalogue.md) |
| 2 | **Is it legal?** | May these operations run in a different order, or at the same time? Dependences, aliasing, floating-point rules, exceptions | wrong answers, sometimes only on some inputs | [6](./06_dependences.md) |
| 3 | **How does the data arrive?** | Contiguous, strided, gathered? Which layout, which alignment, from which memory level? | vectorized but no faster | [7](./07_memory_layout.md) |
| 4 | **What about control?** | Branches, early exits, ragged lengths, tails | half the lanes idle, or the compiler refuses | [8](./08_control_flow.md) |
| 5 | **Is it worth it?** | Compute- or memory-bound? What fraction of runtime? What overheads appear? | effort with no speedup | [9](./09_profitability.md) |
| 6 | **Who does it?** | You in Python, the framework, the compiler, you in intrinsics, or the hardware? | brittle code, or performance that vanishes on the next refactor | [11](./11_array_programming.md)–[16](./16_matrix_engines.md) |

<div class="callout warn"><span class="callout-title">Trap</span>
<p>Engineers usually start at question 6 ("should I use intrinsics?") and skip 2, 3 and 5. The compiler
already answered question 2 for you, conservatively, which is why it refused. The roofline already answered
question 5, which is why the intrinsics didn't help.</p>
</div>

---

## Five shapes of parallel hardware

The ISA floor is where the six questions meet silicon. There are five recurring shapes. Each is a
different answer to "how many data does one decision control, and how does the hardware know how many?"

| Shape | One decision controls | Width is | Masking | Examples | Chapter |
|---|---|---|---|---|:---:|
| **Fixed-width SIMD** | one register of lanes | baked into the opcode (128/256/512 bit) | blend (SSE/AVX2), mask registers (AVX-512) | SSE, AVX2, AVX-512, NEON, WASM SIMD | [3](./03_lanes_and_registers.md), [14](./14_explicit_simd.md) |
| **Vector-length agnostic** | up to `VL` elements, chosen at run time | a hardware property the binary queries | predicate registers | Cray-1, Arm SVE/SVE2, RISC-V V | [5](./05_history.md), [17](./17_architects_view.md) |
| **SIMT** | a 32- or 64-thread warp/wavefront | fixed, but programmed as scalar threads | per-thread predicates, divergence hardware | NVIDIA, AMD GPUs | [15](./15_gpus_simt.md) |
| **Tile / matrix engine** | a small matrix multiply | a tile shape (16×16, 64×N, …) | tile configuration | Tensor Cores, AMX, SME, MFMA | [16](./16_matrix_engines.md) |
| **Systolic array** | a wavefront of MACs across a grid | the array size (128×128, 256×256) | n/a — dataflow | TPU MXU, many NPUs | [16](./16_matrix_engines.md) |

Flynn's 1966 taxonomy named the first category, *single instruction, multiple data*, and
the label stuck. The four later shapes are all SIMD in Flynn's sense. They differ in *who knows
the width* (the opcode, the runtime, the programmer, or nobody, because data just flows) and in
*how big the decision is* (a lane-wise add, or a 256×256 multiply-accumulate wave).

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>Each step down that table makes one instruction control more arithmetic and makes the hardware less
general. Where does your workload sit? A model that is 95 % GEMM by FLOPs wants the bottom of the table.
A tokenizer, a beam search or a sparse feature lookup wants the top. Most real systems need both, in the
same program.</p>
</div>

---

## Three mental moves you will use everywhere

**1. Find the independent axis.** Every vectorization starts by finding a dimension along which
the work is independent. For an elementwise op it is obvious. For a reduction it is hidden: you
create independence by splitting into partial sums. For a recurrence it may not exist, or it exists
only after a transformation such as the wavefront skew in [Chapter 6](./06_dependences.md) or the
parallel scan in [Chapter 10](./10_technique_catalogue.md).

**2. Make the data match the lanes.** The independent axis should be the *contiguous* axis in
memory. When it is not, you transpose, re-layout (AoS → SoA, NCHW → NHWC), or pay for gathers
([Chapter 7](./07_memory_layout.md)).

**3. Remove the decisions inside the loop.** Branches become masks or selects, ragged lengths become
padding or predication, early exits become "compute everything, discard later"
([Chapter 8](./08_control_flow.md)). Each conversion trades wasted lanes for removed control flow.

<div class="manual">
<div class="manual-head"><b>VEKTÖR</b><span>assembly instructions · 3 steps · tools: a pencil</span></div>
<div class="manual-steps">
<div class="mstep"><div class="mstep-n">1</div><svg viewBox="0 0 200 90"><g stroke="#111" fill="none" stroke-width="1.5"><rect x="10" y="20" width="180" height="20"/><line x1="55" y1="20" x2="55" y2="40"/><line x1="100" y1="20" x2="100" y2="40"/><line x1="145" y1="20" x2="145" y2="40"/></g><g stroke="#c00" stroke-width="2" fill="none"><path d="M10 60 H190"/><path d="M180 54 L190 60 L180 66"/></g><text x="100" y="82" font-size="10" text-anchor="middle">find the independent axis</text></svg><div class="mstep-cap">Mark the dimension with no arrows between its points.</div></div>
<div class="mstep"><div class="mstep-n">2</div><svg viewBox="0 0 200 90"><g stroke="#111" fill="none" stroke-width="1.5"><rect x="10" y="10" width="40" height="40"/><rect x="60" y="10" width="40" height="40"/><path d="M110 30 H140"/><path d="M134 24 L140 30 L134 36"/><rect x="150" y="10" width="15" height="40"/><rect x="167" y="10" width="15" height="40"/></g><text x="100" y="80" font-size="10" text-anchor="middle">lay it out contiguously</text></svg><div class="mstep-cap">Make that axis the fastest-moving one in memory.</div></div>
<div class="mstep warn"><div class="mstep-n">3</div><svg viewBox="0 0 200 90"><g stroke="#111" fill="none" stroke-width="1.5"><path d="M20 20 L50 45 L20 70"/><path d="M50 45 H90"/><rect x="110" y="25" width="70" height="40"/><path d="M120 45 L135 55 L170 33" stroke="#c00" stroke-width="2.2"/></g><text x="100" y="86" font-size="10" text-anchor="middle">turn branches into masks</text></svg><div class="mstep-cap">Compute both sides; select per lane. Count the wasted lanes.</div></div>
</div>
</div>

---

## How to read this book

<div class="diagram">
<div class="diagram-grid cols-3">
<div class="diagram-card accent"><div class="card-title">Part I — Mental model (Ch 2–5)</div><div class="card-desc">Where vectorization lives, what a register really holds, how to draw it, and how we got here.</div></div>
<div class="diagram-card purple"><div class="card-title">Part II — Legality &amp; profitability (Ch 6–9)</div><div class="card-desc">Dependences, memory, control flow, roofline. The chapters that stop you wasting a week.</div></div>
<div class="diagram-card green"><div class="card-title">Part III — The toolbox (Ch 10–14)</div><div class="card-desc">Every technique, then each layer that applies them: NumPy, vmap, compilers, intrinsics.</div></div>
<div class="diagram-card orange"><div class="card-title">Part IV — Accelerators (Ch 15–17)</div><div class="card-desc">GPUs, matrix engines, systolic arrays, and the architect's trade-offs.</div></div>
<div class="diagram-card pink"><div class="card-title">Part V — ML systems (Ch 18–19)</div><div class="card-desc">Softmax, LayerNorm, GEMM, attention, quantized dot products; batching at system scale.</div></div>
<div class="diagram-card cyan"><div class="card-title">Part VI — Practice (Ch 20–23)</div><div class="card-desc">A measured ladder, a debugging playbook, questions worth pondering, and where to go next.</div></div>
</div>
</div>

**If you read only five chapters**, read these, in this order:

1. [3 — Lanes & Registers](./03_lanes_and_registers.md): the vocabulary.
2. [6 — Dependences & Aliasing](./06_dependences.md): when it is legal.
3. [7 — Memory Is the Real Vector](./07_memory_layout.md): why legal is not enough.
4. [9 — Is It Worth It?](./09_profitability.md): the roofline, measured.
5. [18 — Vectorizing ML Kernels](./18_ml_kernels.md): all of it applied to softmax, GEMM and attention.

---

## Key Takeaways

- Vectorization is one idea, one decision applied to many data, appearing on every layer from Python to silicon.
- A decision is complete only when it answers six questions: the vector axis, legality, data arrival, control, profitability and who does it.
- Legality (may I reorder?) and profitability (will it be faster?) are separate questions; most confusion comes from mixing them.
- The five hardware shapes (fixed SIMD, VLA, SIMT, tile engines, systolic arrays) differ in who knows the width and how much work one decision controls.
- The three recurring moves are: find the independent axis, make it contiguous, and turn decisions into masks.

## Check Yourself

<details class="check"><summary>For a batched softmax over a [B, T, V] tensor along V, what is "the vector" on a CPU, and what on a GPU?</summary>
On a CPU, usually V: it is contiguous and long, and SIMD lanes run along it, with reductions (max, sum) across lanes. On a GPU, threads are typically spread over both B·T (one row per warp or block) and V (lanes within the row). The choice depends on V's length relative to the warp and register budget.
</details>

<details class="check"><summary>Which of the six questions does "the compiler said it couldn't prove no aliasing" belong to?</summary>
Question 2, legality. The compiler could not prove that writes through one pointer do not change data read through another, so reordering might change results.
</details>

<details class="check"><summary>What distinguishes SVE or RISC-V V from AVX-512, given that all three have predicate masks?</summary>
Vector-length agnosticism: the binary does not encode the register width. It asks the hardware (or <code>vsetvl</code>) how many elements to process per iteration, so the same code runs on 128-bit and 2048-bit implementations.
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Very High-Speed Computing Systems | Flynn, Proc. IEEE 1966 | doi:10.1109/PROC.1966.5273 | The SISD/SIMD/MISD/MIMD taxonomy |
| Computer Architecture: A Quantitative Approach, 6th ed., ch. 4 | Hennessy & Patterson, 2017 | Morgan Kaufmann | Vector, SIMD-extension and GPU architectures as one family |
| The ARM Scalable Vector Extension | Stephens et al., IEEE Micro 2017 | doi:10.1109/MM.2017.35 | Vector-length agnostic programming model |
| Optimizing Compilers for Modern Architectures | Allen & Kennedy, 2001 | Morgan Kaufmann | Dependence-based framing of legality |

---

**Next:** [Chapter 2 — A Field Guide to Vectorizable Workloads](./02_domains.md) — where in the wild these loops live, and how to recognise them.
