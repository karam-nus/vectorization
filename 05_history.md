---
title: "Chapter 5 — From Cray-1 to Tensor Cores"
---

[← Back to Table of Contents](./README.md)

# Chapter 5 — From Cray-1 to Tensor Cores

> *"Every generation rediscovers that one instruction should do more work, and then argues about how much more."*

## Overview

The history of vectorization is a pendulum. Supercomputers started with long vectors. Desktop CPUs
started again with short, fixed ones and grew them slowly, while GPUs hid the vector inside the
thread. The newest designs swing back to long vector-length-agnostic registers on one side and
whole-matrix instructions on the other. Knowing the swings explains today's zoo of ISAs and predicts
the next one. The front pages below are the events; the sections after them are the three arguments
the field keeps having.

<div class="newspaper">
<div class="np-mast"><b>The Lane Gazette</b></div>
<div class="np-dateline"><span>Est. 1966</span><span>"All the lanes that fit"</span><span>Price: one instruction</span></div>
<div class="np-cols">
<div class="np-item lead"><div class="np-year">1976 · CHIPPEWA FALLS</div><div class="np-hl">CRAY-1 SHIPS: EIGHT VECTOR REGISTERS OF 64 WORDS</div><p>Seymour Cray's machine reaches Los Alamos with registers that hold 64 numbers each and a vector-length register that says how many to use. "Chaining" lets the result of one pipeline feed the next without waiting. It is the first commercially successful vector computer and remains the reference design for half a century.</p></div>
<div class="np-item"><div class="np-year">1966 · STANFORD</div><div class="np-hl">FLYNN SORTS COMPUTERS INTO FOUR BOXES</div><p>Michael Flynn classifies machines by instruction and data streams. "SIMD" enters the vocabulary and never leaves.</p></div>
<div class="np-item"><div class="np-year">1974 · MINNEAPOLIS</div><div class="np-hl">STAR-100 STREAMS VECTORS STRAIGHT FROM MEMORY</div><p>CDC's machine has no vector registers. Every vector operation starts from memory, and the start-up cost is so long that short vectors run slower than scalar code. Designers take note.</p></div>
<div class="np-item"><div class="np-year">1975 · URBANA / AMES</div><div class="np-hl">ILLIAC IV: 64 PROCESSORS, ONE INSTRUCTION</div><p>A lockstep array of 64 processing elements finally runs, years late and far over budget. The idea, many simple units obeying one controller, outlives the machine.</p></div>
<div class="np-item"><div class="np-year">1985 · CAMBRIDGE, MASS.</div><div class="np-hl">CONNECTION MACHINE WIRES 65,536 ONE-BIT PROCESSORS</div><p>Thinking Machines bets on massive data parallelism; its programming model of one operation over a whole data set foreshadows array languages and GPUs.</p></div>
<div class="np-item"><div class="np-year">1997 · SANTA CLARA</div><div class="np-hl">MMX PUTS SIMD ON THE DESKTOP</div><p>Intel's Pentium MMX adds 64-bit integer vectors for multimedia, borrowing the floating-point registers. Short-vector SIMD becomes a consumer feature.</p></div>
<div class="np-item"><div class="np-year">1999 · SANTA CLARA &amp; AUSTIN</div><div class="np-hl">SSE AND ALTIVEC GO TO 128 BITS</div><p>Pentium III's SSE brings four-wide floats; Motorola's AltiVec arrives with 32 registers and a general permute. SSE2 (2001) later becomes mandatory for every x86-64 chip.</p></div>
<div class="np-item"><div class="np-year">2005 · CAMBRIDGE, UK</div><div class="np-hl">NEON GIVES PHONES A VECTOR UNIT</div><p>Arm's Advanced SIMD ships with Cortex-A8. A decade later every smartphone CPU can do 128-bit SIMD.</p></div>
<div class="np-item lead"><div class="np-year">2006 · SANTA CLARA</div><div class="np-hl">G80 AND CUDA: THE VECTOR HIDES INSIDE THE THREAD</div><p>NVIDIA's unified architecture runs 32 threads per instruction ("SIMT"), but programmers write ordinary scalar code per thread. The hardware, not the programmer, manages lanes and divergence. Within ten years it is how most of the world's floating-point arithmetic for AI is done.</p></div>
<div class="np-item"><div class="np-year">2011–2013 · SANTA CLARA</div><div class="np-hl">AVX DOUBLES TO 256; AVX2 ADDS FMA AND GATHER</div><p>Sandy Bridge widens floating point; Haswell brings integers, fused multiply-add and the first x86 gather.</p></div>
<div class="np-item"><div class="np-year">2015–2017 · MOUNTAIN VIEW</div><div class="np-hl">GOOGLE'S TPU: A 256 × 256 GRID OF MULTIPLIERS</div><p>Deployed in 2015 and revealed at ISCA 2017, the TPU's matrix unit performs 65,536 int8 multiply-adds per cycle as a systolic array, with no instruction per multiply at all.</p></div>
<div class="np-item"><div class="np-year">2016 · HOT CHIPS</div><div class="np-hl">AVX-512 SHIPS; ARM UNVEILS "SCALABLE" VECTORS</div><p>Intel's Knights Landing brings 512-bit vectors with mask registers. The same summer Arm announces SVE: code that does not know its own vector length.</p></div>
<div class="np-item"><div class="np-year">2017 · SAN JOSE</div><div class="np-hl">VOLTA'S TENSOR CORES: THE INSTRUCTION BECOMES A MATRIX</div><p>Each tensor core performs a 4 × 4 × 4 matrix multiply-accumulate per clock. The unit of work is no longer a lane but a tile.</p></div>
<div class="np-item"><div class="np-year">2020 · KOBE</div><div class="np-hl">FUGAKU TOPS THE WORLD ON 512-BIT SVE</div><p>Fujitsu's A64FX, the first SVE implementation, powers the fastest supercomputer of June 2020, with no GPUs.</p></div>
<div class="np-item"><div class="np-year">2021 · ZURICH</div><div class="np-hl">RISC-V V 1.0 RATIFIED</div><p>An open vector ISA in the Cray tradition: <code>vsetvl</code> asks the hardware how many elements it may process, and VLEN is left to each chip.</p></div>
<div class="np-item"><div class="np-year">2022–2024 · EVERYWHERE</div><div class="np-hl">MATRIX ENGINES IN EVERY CLASS OF CHIP</div><p>Hopper adds asynchronous warpgroup MMA and FP8; Intel's AMX puts tile multiplies in Xeon cores (2023); Apple's M4 exposes Arm SME (2024); Blackwell moves accumulators into dedicated tensor memory and adds FP4.</p></div>
<div class="np-item"><div class="np-year">2024–2026 · SANTA CLARA</div><div class="np-hl">AVX10 PROMISES ONE ISA, 512 BITS ON EVERY CORE</div><p>After a decade of AVX-512 fragmentation, Intel's AVX10.2 specification converges the features and, after a 2024 revision, requires 512-bit support on all cores. First silicon is announced for future parts (timing unconfirmed).</p></div>
</div>
</div>

Every row of this timeline is also in the [Vector ISA Atlas](./atlas.md), with width, masking,
gather support, ML data types and a source per row.

---

## Argument 1 — Registers or memory?

The STAR-100 and the Cray-1 settled the first argument within two years. Operating directly on memory
means every vector instruction pays memory latency up front. Hennessy and Patterson's account puts
the STAR-100's start-up at well over a hundred cycles, so vectors had to be very long before the
machine beat a scalar processor. The Cray-1 loaded vectors into **registers** once and chained
operations between them, which made short and medium vectors profitable.

The lesson has been relearned at every scale since. A GPU kernel that reads every operand from global
memory is a STAR-100. One that stages tiles in shared memory and registers is a Cray-1. A matrix
engine is the extreme case: Blackwell gives its tensor cores a private memory (TMEM) just to hold
accumulators near the arithmetic.

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>"Start-up cost versus vector length" is the same trade-off as "kernel launch overhead versus tensor
size" on a GPU, and "Python dispatch overhead versus array size" in NumPy. Where in your stack is the
fixed cost per operation, and how long must your vectors be before it disappears?</p>
</div>

---

## Argument 2 — Who decides the width?

| Era | Width decided by | Consequence |
|---|---|---|
| **Cray-1**, vector supercomputers | a vector-length register set at run time, up to 64 | one binary handles any $n$; strip-mining loops are trivial |
| **MMX → AVX-512** | the opcode (64 → 128 → 256 → 512 bit) | every width change means new instructions, recompilation and runtime dispatch; ~5 ISA generations of intrinsics |
| **GPUs (SIMT)** | the hardware warp size, hidden from per-thread code | portable across generations; width appears only in performance tuning |
| **SVE, RISC-V V** | the hardware, queried at run time | one binary for 128- to 2048-bit implementations; tail handling built in |
| **Matrix engines** | a tile shape per instruction family | hardware-specific kernels again (CUTLASS has a code path per architecture) |

Fixed-width SIMD won the desktop because it was cheap to add to an out-of-order core and easy for
compilers to target one register size. Its cost is visible in any performance library: NumPy, oneDNN,
OpenCV and PyTorch all ship multiple compiled versions of their hot loops and pick one at run time
(**function multi-versioning** or **CPU dispatch**). Vector-length-agnostic designs remove that cost,
at the price of compilers and programmers not knowing $V$ at compile time ([Chapter 17](./17_architects_view.md)).

---

## Argument 3 — General or specialised?

The most consequential swing is the newest: from lane-wise vectors, which can express anything, to
tile and systolic engines, which express only matrix multiplication but do it at a fraction of the
energy per operation. The TPU paper made the case quantitatively: for the inference workloads it
measured, a domain-specific matrix unit delivered far higher performance per watt than contemporary
CPUs and GPUs (Jouppi et al., 2017).

The cost is generality. Everything that is *not* a matrix multiply (softmax, normalisation, routing,
sampling, quantization scales) still runs on ordinary vector units next to the matrix engine. As
matrix throughput grows faster than vector throughput, those parts become the bottleneck. This is
why Blackwell and Rubin-generation GPUs advertised faster special-function units for softmax
(secondary press coverage; unconfirmed in vendor documentation at the time of writing), and why
[Chapter 18](./18_ml_kernels.md) spends as long on softmax as on GEMM.

<div class="callout field"><span class="callout-title">Field note</span>
<p>When a new accelerator is announced, ask three questions from this chapter. Where do vector operands
live (memory, registers, private tile memory)? Who decides the width (opcode, runtime, hardware)? What
runs on the general vector units, and how fast are they relative to the matrix units? The answers
predict which of your kernels will port easily and which will need rewriting.</p>
</div>

---

## What the pendulum predicts

1. **Matrix engines will keep growing faster than vector units**, so non-GEMM work (attention softmax,
   normalisation, activation, dequantization) will keep becoming a larger share of runtime. Expect
   more fusion and more special-function hardware.
2. **Vector-length agnosticism will spread**, because it is the only design that lets one binary
   survive width changes. AVX10's version numbers are a partial answer for x86.
3. **Low-precision lanes will keep multiplying**: every halving of the element size doubles $V$ for
   free (Chapter 3), so FP8, FP6, FP4 and block-scaled formats are a vectorization story as much as a
   quantization one.

---

## Key Takeaways

- Vectors began long (Cray-1, 64 elements), shrank to 64–128 bits on desktops, and are growing again in two directions: length-agnostic vectors and matrix tiles.
- Register-based vectors beat memory-to-memory vectors because start-up cost must be amortised; the same trade-off reappears as kernel-launch and Python-dispatch overhead.
- Fixed-width ISAs forced libraries into runtime dispatch across ISA generations; VLA designs remove that at the cost of an unknown $V$ at compile time.
- Matrix engines win on energy per operation but are specialised; non-matrix work increasingly sets the pace.
- New hardware can be read with three questions: where operands live, who decides the width, and how fast the general vector units are.

## Check Yourself

<details class="check"><summary>Why did the STAR-100 run short vectors slower than scalar code?</summary>
Each vector instruction streamed operands from memory and paid a long start-up latency before the first result. For short vectors that fixed cost was not amortised; the Cray-1 avoided it by keeping vectors in registers and chaining pipelines.
</details>

<details class="check"><summary>What does "function multi-versioning" cost a library, and which ISA design removes the need?</summary>
Multiple compiled copies of each hot function (SSE2, AVX2, AVX-512, …), larger binaries, a dispatch layer and a test matrix per ISA. Vector-length-agnostic ISAs (SVE, RISC-V V) let one binary run at any implemented width, though different feature levels can still require dispatch.
</details>

<details class="check"><summary>Why does faster matrix hardware make softmax more important?</summary>
Attention's GEMMs speed up with each tensor-core generation, but the softmax between them runs on general vector units and special-function units (exp). If those grow more slowly, softmax's share of the runtime rises until it is a bottleneck; hence fused kernels (FlashAttention) and faster SFUs.
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| The CRAY-1 Computer System | Russell, CACM 1978 | doi:10.1145/359327.359336 | Vector registers, vector length, chaining |
| Computer Architecture: A Quantitative Approach, 6th ed., ch. 4 and App. G | Hennessy & Patterson, 2017 | Morgan Kaufmann | STAR-100 vs Cray-1 start-up, vector processor history |
| Very High-Speed Computing Systems | Flynn, Proc. IEEE 1966 | doi:10.1109/PROC.1966.5273 | SIMD taxonomy |
| The Connection Machine | Hillis, MIT Press 1985 | — | Massive data parallelism |
| MMX Technology Extension to the Intel Architecture | Peleg & Weiser, IEEE Micro 1996 | doi:10.1109/40.526921 | First desktop SIMD |
| NVIDIA Tesla: A Unified Graphics and Computing Architecture | Lindholm et al., IEEE Micro 2008 | doi:10.1109/MM.2008.31 | SIMT |
| In-Datacenter Performance Analysis of a Tensor Processing Unit | Jouppi et al., ISCA 2017 | arXiv:1704.04760 | Systolic matrix unit, performance per watt |
| The ARM Scalable Vector Extension | Stephens et al., IEEE Micro 2017 | doi:10.1109/MM.2017.35 | VLA design |
| NVIDIA Tesla V100 GPU Architecture | NVIDIA, 2017 | whitepaper | First tensor cores |
| Intel AVX10.2 Architecture Specification; Nova Lake press coverage | Intel 2024; secondary 2025 | intel.com; press | AVX10 convergence, 512-bit on all cores (timing unconfirmed) |
| `assets/data/isas.json` | this book | [atlas](./atlas.md) | Dates and widths for every row |

---

**Next:** [Chapter 6 — Dependences & Aliasing: The Legality Question](./06_dependences.md) — before "is it faster?", "is it still correct?"
