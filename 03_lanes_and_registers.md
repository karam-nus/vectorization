---
title: "Chapter 3 — Lanes, Registers & the SIMD Contract"
---

[← Back to Table of Contents](./README.md)

# Chapter 3 — Lanes, Registers & the SIMD Contract

> *"SIMD is synchronized swimming: everyone does the same stroke at the same time, and nobody may swim in a neighbour's lane without the coach's permission."*

## Overview

A vector register is $w$ bits with no opinion about what they mean. An instruction decides: the same
512 bits are 64 int8 lanes, 32 fp16 lanes, 16 fp32 lanes or 8 fp64 lanes. This chapter states the
**SIMD contract**, the four promises every vector instruction set makes. It then walks the eight
families of vector instructions with register strips that show exactly what moves where. By the end
you should be able to read any intrinsic name and draw its effect.

<div class="pool anim">
<div class="pool-title">lane 0–7 · one instruction · one stroke</div>
<div class="lane"><span class="lane-tag">lane 0</span><span class="swimmer" style="--sw:#22d3ee">a0</span></div>
<div class="lane"><span class="lane-tag">lane 1</span><span class="swimmer" style="--sw:#a78bfa">a1</span></div>
<div class="lane"><span class="lane-tag">lane 2</span><span class="swimmer" style="--sw:#34d399">a2</span></div>
<div class="lane sit"><span class="lane-tag">lane 3</span><span class="swimmer">a3</span></div>
<div class="lane"><span class="lane-tag">lane 4</span><span class="swimmer" style="--sw:#f472b6">a4</span></div>
<div class="lane"><span class="lane-tag">lane 5</span><span class="swimmer" style="--sw:#60a5fa">a5</span></div>
<div class="lane"><span class="lane-tag">lane 6</span><span class="swimmer" style="--sw:#fb923c">a6</span></div>
<div class="lane"><span class="lane-tag">lane 7</span><span class="swimmer" style="--sw:#a3e635">a7</span></div>
<div class="pool-foot"><span>start wall = instruction issue</span><span>far wall = result written, all lanes together</span></div>
</div>
<div class="genre-caption">Eight lanes execute one instruction in lockstep. Lane 3 is masked off: it still occupies its lane for the whole length of the pool, but it does not swim. A masked lane costs the same time as an active one.</div>

---

## The SIMD contract

Every vector instruction set, from MMX to AVX-512, NEON, SVE and RISC-V V, makes the same four promises.
Each one has a cost that later chapters trace back here.

| # | Promise | What it buys | What it costs | Chapter |
|:---:|---|---|---|:---:|
| 1 | **Lockstep**: one operation, all lanes, one instruction | decode and schedule paid once per $V$ elements | every lane waits for the slowest; masked lanes still occupy time | [8](./08_control_flow.md) |
| 2 | **Lane independence**: lane $i$ of the result depends only on lane $i$ of the inputs, unless the instruction says otherwise | simple, wide datapaths with no wiring between lanes | moving data *across* lanes needs separate, slower shuffle instructions | this chapter |
| 3 | **Contiguity**: a plain vector load reads $V$ consecutive elements | one cache-line access feeds all lanes | anything else is a gather, costing a load per element | [7](./07_memory_layout.md) |
| 4 | **Uniform width**: every lane is the same type and size | cheap hardware | mixed precision needs explicit widen/narrow instructions | this chapter |

The lane count follows from the register width $w$ and the element size $e$:

$$V = \frac{w}{e}$$

Numerical check: an AVX-512 register ($w = 512$) holding bfloat16 ($e = 16$) has $512 / 16 = 32$
lanes; the same register holding int8 has 64.

<div class="pack">
<div class="pack-title">What fits in one 512-bit register</div>
<div class="pack-row"><span class="pack-name">int8 / fp8</span><span class="pack-bar" style="--n:64"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></span><span class="pack-count">64 lanes</span></div>
<div class="pack-row"><span class="pack-name">fp16 / bf16</span><span class="pack-bar" style="--n:32;--c:#a78bfa"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></span><span class="pack-count">32 lanes</span></div>
<div class="pack-row"><span class="pack-name">fp32 / int32</span><span class="pack-bar" style="--n:16;--c:#34d399"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></span><span class="pack-count">16 lanes</span></div>
<div class="pack-row"><span class="pack-name">fp64 / int64</span><span class="pack-bar" style="--n:8;--c:#fbbf24"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></span><span class="pack-count">8 lanes</span></div>
<div class="pack-row"><span class="pack-name">fp32 in AVX2</span><span class="pack-bar" style="--n:16;--c:#60a5fa"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i></span><span class="pack-count">8 lanes</span></div>
</div>
<div class="genre-caption">Halving the element size doubles the lanes for the same instruction: the arithmetic reason low precision is fast, separate from the memory reason (Chapter 18). int4 is not a native lane type on most ISAs; it is unpacked into int8 by shuffles first.</div>

---

## The eight families of vector instructions

### 1. Lane-wise (vertical) arithmetic

Add, multiply, fused multiply-add, min, max, compare, convert. Lane $i$ of the result uses only lane $i$
of each input. These are the cheap ones: on Intel's recent big cores a 512-bit FMA has a 4-cycle
latency and two can start every cycle (uops.info; Intel optimization manual).

<div class="regs">
<div class="regs-title">vfmadd231ps zmm0, zmm1, zmm2 — c = a · b + c, lane by lane</div>
<div class="reg-row"><span class="reg-name">a (zmm1)</span><span class="reg-lanes"><span class="ln">1</span><span class="ln">2</span><span class="ln">3</span><span class="ln">4</span><span class="ln">5</span><span class="ln">6</span><span class="ln">7</span><span class="ln">8</span></span></div>
<div class="reg-row"><span class="reg-name">b (zmm2)</span><span class="reg-lanes"><span class="ln l1">2</span><span class="ln l1">2</span><span class="ln l1">2</span><span class="ln l1">2</span><span class="ln l1">0</span><span class="ln l1">0</span><span class="ln l1">1</span><span class="ln l1">1</span></span></div>
<div class="reg-row"><span class="reg-name">c (zmm0) in</span><span class="reg-lanes"><span class="ln l2">1</span><span class="ln l2">1</span><span class="ln l2">1</span><span class="ln l2">1</span><span class="ln l2">1</span><span class="ln l2">1</span><span class="ln l2">1</span><span class="ln l2">1</span></span></div>
<div class="reg-row"><span class="reg-name">c (zmm0) out</span><span class="reg-lanes"><span class="ln hot">3</span><span class="ln hot">5</span><span class="ln hot">7</span><span class="ln hot">9</span><span class="ln hot">1</span><span class="ln hot">1</span><span class="ln hot">8</span><span class="ln hot">9</span></span><span class="reg-note">8 of 16 lanes shown</span></div>
</div>

### 2. Broadcast

Copy one scalar into every lane (`_mm512_set1_ps`, `vbroadcastss`, NEON `dup`). This is how a
constant like the `a` in `a*x + y` or a GEMM's $A_{mk}$ enters a vector computation. A broadcast from
memory is often free, folded into the arithmetic instruction's memory operand.

<div class="regs">
<div class="regs-title">vbroadcastss zmm3, [a] — one value, every lane</div>
<div class="reg-row"><span class="reg-name">a (scalar)</span><span class="reg-lanes"><span class="ln l3">2.5</span></span></div>
<div class="reg-row"><span class="reg-name">zmm3</span><span class="reg-lanes"><span class="ln l3">2.5</span><span class="ln l3">2.5</span><span class="ln l3">2.5</span><span class="ln l3">2.5</span><span class="ln l3">2.5</span><span class="ln l3">2.5</span><span class="ln l3">2.5</span><span class="ln l3">2.5</span></span></div>
</div>

### 3. Horizontal operations and reductions

Anything that combines lanes *of the same register*: sum all lanes, max of all lanes. There is
usually no single instruction for it. A reduction is $\log_2 V$ rounds of "shuffle half the lanes
next to the other half, then add". `_mm512_reduce_add_ps` is a compiler macro that expands to exactly that.

<div class="regs">
<div class="regs-title">Horizontal sum of 8 lanes in 3 shuffle + add rounds</div>
<div class="reg-row"><span class="reg-name">v</span><span class="reg-lanes"><span class="ln">1</span><span class="ln">2</span><span class="ln">3</span><span class="ln">4</span><span class="ln l1">5</span><span class="ln l1">6</span><span class="ln l1">7</span><span class="ln l1">8</span></span></div>
<div class="reg-row"><span class="reg-name">+ upper half</span><span class="reg-lanes"><span class="ln hot">6</span><span class="ln hot">8</span><span class="ln hot">10</span><span class="ln hot">12</span><span class="ln dim">·</span><span class="ln dim">·</span><span class="ln dim">·</span><span class="ln dim">·</span></span><span class="reg-note">round 1</span></div>
<div class="reg-row"><span class="reg-name">+ upper quarter</span><span class="reg-lanes"><span class="ln hot">16</span><span class="ln hot">20</span><span class="ln dim">·</span><span class="ln dim">·</span><span class="ln dim">·</span><span class="ln dim">·</span><span class="ln dim">·</span><span class="ln dim">·</span></span><span class="reg-note">round 2</span></div>
<div class="reg-row"><span class="reg-name">+ neighbour</span><span class="reg-lanes"><span class="ln hot">36</span><span class="ln dim">·</span><span class="ln dim">·</span><span class="ln dim">·</span><span class="ln dim">·</span><span class="ln dim">·</span><span class="ln dim">·</span><span class="ln dim">·</span></span><span class="reg-note">round 3 = log₂ 8</span></div>
</div>

A reduction therefore costs roughly $\log_2 V$ dependent shuffle-add pairs *once*, after the main loop.
Doing one per iteration, as gcc did for the strict-IEEE dot product in Chapter 13, is how a vectorized
loop ends up slower than the scalar one.

### 4. Permutes and shuffles

Rearrange lanes: reverse, interleave, rotate, arbitrary table-driven permutes (`vpermps`, `vpermt2ps`,
NEON `tbl`, SVE `tbl`, RISC-V `vrgather`). Two things to know:

- **In-lane vs cross-lane.** Many x86 shuffles (`vshufps`, `vpshufb`) only move data within each
  128-bit chunk, a legacy of SSE; crossing 128-bit boundaries needs a different, usually slower
  (3-cycle) instruction.
- **Shuffles compete for one port** on many cores. A kernel that is "vectorized" but does three
  shuffles per FMA is shuffle-bound, not FMA-bound.

`pshufb` is also a **16-entry lookup table** per 128-bit chunk, indexed by the low 4 bits of each
byte. That one fact powers SIMD UTF-8 validation, base64, hex decoding and 4-bit weight dequantization
in llama.cpp-style kernels (Chapter 18).

### 5. Loads and stores

| Kind | What it does | Cost relative to a plain load |
|---|---|---|
| **Aligned / unaligned** | $V$ consecutive elements | ~1×; unaligned costs more only when it crosses a cache line |
| **Masked** (`vmaskmov`, AVX-512 `{k}`, SVE predicated) | load/store only active lanes; inactive lanes do not fault | ~1× |
| **Broadcast load** | one element to all lanes | ~1× |
| **Gather** (`vgatherdps`, SVE `ld1w` with vector index, RVV `vluxei`) | $V$ elements from $V$ addresses | one memory access per element; often no faster than scalar code (measured, Chapter 7) |
| **Scatter** (AVX-512, SVE, RVV) | $V$ stores to $V$ addresses | one access per element, and lanes may conflict |
| **First-faulting** (SVE `ldff1`) | load until the first bad address, report how far it got | lets `strlen`-style loops vectorize safely |

### 6. Masks, predication and blends

A compare produces a **mask**: one bit per lane. Later instructions use it to merge or zero results
lane by lane. AVX-512 has eight mask registers `k0–k7` and accepts a mask on almost every instruction;
SVE has sixteen predicate registers `p0–p15`; RISC-V V uses register `v0`. SSE and AVX2 have no mask
registers and emulate them with a vector of all-ones/all-zeros lanes and `blendv`.

<div class="regs">
<div class="regs-title">Leaky ReLU: compare → mask → blend (AVX-512)</div>
<div class="reg-row"><span class="reg-name">x</span><span class="reg-lanes"><span class="ln">-2</span><span class="ln">3</span><span class="ln">-1</span><span class="ln">4</span><span class="ln">0.5</span><span class="ln">-8</span><span class="ln">6</span><span class="ln">-3</span></span></div>
<div class="reg-row"><span class="reg-name">k1 = x &gt; 0</span><span class="reg-lanes"><span class="ln mask0">0</span><span class="ln mask1">1</span><span class="ln mask0">0</span><span class="ln mask1">1</span><span class="ln mask1">1</span><span class="ln mask0">0</span><span class="ln mask1">1</span><span class="ln mask0">0</span></span><span class="reg-note">vcmpps k1, x, 0, GT</span></div>
<div class="reg-row"><span class="reg-name">0.01 · x</span><span class="reg-lanes"><span class="ln l4">-.02</span><span class="ln l4">.03</span><span class="ln l4">-.01</span><span class="ln l4">.04</span><span class="ln l4">.005</span><span class="ln l4">-.08</span><span class="ln l4">.06</span><span class="ln l4">-.03</span></span><span class="reg-note">computed for every lane</span></div>
<div class="reg-row"><span class="reg-name">y = blend(k1)</span><span class="reg-lanes"><span class="ln hot">-.02</span><span class="ln hot">3</span><span class="ln hot">-.01</span><span class="ln hot">4</span><span class="ln hot">0.5</span><span class="ln hot">-.08</span><span class="ln hot">6</span><span class="ln hot">-.03</span></span><span class="reg-note">vblendmps y{k1}, slope·x, x</span></div>
</div>

Both sides of the branch are computed for all lanes; the mask picks per lane. Measured on our Xeon,
this loop runs 15.6× faster than scalar with AVX-512. With AVX2, gcc refused to vectorize it at all.
[Chapter 8](./08_control_flow.md) explains why: the multiply on the "wrong" side might raise a
floating-point exception, and only masked execution lets the compiler compute it safely.

### 7. Widening, narrowing and mixed-precision dot products

Uniform width (promise 4) collides with ML's love of mixed precision. Accumulating int8 products in
int8 would overflow immediately, so ISAs provide instructions that *change lane width*:

- **Widen**: `vpmovsxbd` turns 16 int8 lanes into 16 int32 lanes (and needs a 4× wider register).
- **Narrow with saturation**: `vpackssdw`, `vpmovsdb` clamp int32 down to int16 or int8.
- **Dot-product-accumulate**: AVX-512 VNNI `vpdpbusd` multiplies 4 adjacent uint8×int8 pairs and adds
  them into one int32 lane, so 64 bytes of each input become 16 int32 partial sums in one instruction.
  Arm's `sdot`/`udot` and NVIDIA's `dp4a` do the same for 4-element groups.

<div class="regs">
<div class="regs-title">vpdpbusd — four int8 products summed into each int32 lane</div>
<div class="reg-row"><span class="reg-name">a (u8)</span><span class="reg-lanes"><span class="ln">1</span><span class="ln">2</span><span class="ln">3</span><span class="ln">4</span><span class="ln l1">5</span><span class="ln l1">6</span><span class="ln l1">7</span><span class="ln l1">8</span></span><span class="reg-note">8 bytes = 2 groups</span></div>
<div class="reg-row"><span class="reg-name">b (s8)</span><span class="reg-lanes"><span class="ln">1</span><span class="ln">1</span><span class="ln">1</span><span class="ln">1</span><span class="ln l1">-1</span><span class="ln l1">2</span><span class="ln l1">0</span><span class="ln l1">1</span></span></div>
<div class="reg-row"><span class="reg-name">acc (s32) +=</span><span class="reg-lanes"><span class="ln hot wide4">1+2+3+4 = 10</span><span class="ln hot wide4">−5+12+0+8 = 15</span></span><span class="reg-note">one 32-bit lane per 4 bytes</span></div>
</div>

### 8. Compress and expand

AVX-512's `vcompressps` packs the active lanes of a vector contiguously (the **filter** species from
Chapter 2); `vexpandps` does the inverse. SVE has `compact`, RISC-V V has `vcompress`. Without them,
stream compaction needs a prefix sum over the mask and a permute.

---

## The register file you are actually programming

| ISA | Vector registers | Width | Masks / predicates | Total architectural vector state |
|---|:---:|---|---|:---:|
| **SSE** (x86-64) | 16 `xmm` | 128 bit | none (blend with vector masks) | 256 B |
| **AVX2** | 16 `ymm` | 256 bit | none | 512 B |
| **AVX-512** | 32 `zmm` | 512 bit | 8 `k` registers | 2 KB + masks |
| **NEON** (AArch64) | 32 `v` | 128 bit | none (bit-select) | 512 B |
| **SVE / SVE2** | 32 `z` | 128–2048 bit, implementation-defined | 16 `p` registers | 512 B – 8 KB |
| **RISC-V V** | 32 `v` | VLEN, implementation-defined; groups of up to 8 via LMUL | mask in `v0` | VLEN × 4 B |
| **AMX** | 8 tiles | 16 rows × 64 B each | tile configuration | 8 KB |

The number of registers matters as much as their width. A matrix-multiply microkernel keeps a block of
$C$ in registers while streaming $A$ and $B$ past it; 32 `zmm` registers allow a 6 × 64 fp32 block of
$C$ (24 registers) plus operands, which 16 `ymm` registers cannot (Chapter 18).

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>The Cray-1 had 64-element vector registers in 1976. Why did desktop CPUs spend 25 years growing
from 64 to 512 bits instead of jumping straight to 4096? Consider what happens to the register file,
the wiring between lanes for shuffles, the cache line feeding a load, the clock frequency, and the
fraction of real loops longer than 64. Chapter 17 takes it up.</p>
</div>

---

## Interactive: who processes which element

<div class="lab" data-lab="lanes"></div>

Set $n = 27$ and $V = 16$. One full vector covers elements 0–15; the remaining 11 are the **tail**.
A scalar epilogue needs 11 more instructions, and lane utilisation becomes 27/27 but the speedup
falls to 27/12 = 2.25× of a possible 16×. A masked tail does it in one instruction at 84 %
utilisation. Short loops are tail-dominated, which is why $n$ appears in every profitability argument.

---

## Key Takeaways

- A register has no type; the instruction picks the lane count $V = w/e$, so halving precision doubles lanes.
- The SIMD contract is lockstep, lane independence, contiguity and uniform width; each promise has a cost that shows up as masked lanes, shuffles, gathers or widening.
- Lane-wise arithmetic is cheap; crossing lanes (reductions, permutes) is the expensive part, so do it once after the loop, not every iteration.
- Masks turn branches into data; both sides are computed and a mask selects per lane.
- Mixed-precision dot products (VNNI, `sdot`, `dp4a`) exist because uniform width and int8 accumulation do not mix.
- Register count is as important as width: it decides how big a block of output a kernel can keep resident.

## Check Yourself

<details class="check"><summary>How many fp16 lanes does a 256-bit AVX2 register have, and how many int8?</summary>
256 / 16 = 16 fp16 lanes; 256 / 8 = 32 int8 lanes.
</details>

<details class="check"><summary>Why should a vectorized sum keep a vector accumulator through the loop and reduce horizontally only at the end?</summary>
Horizontal reduction costs log₂V dependent shuffle-add rounds. Done once at the end it is negligible; done per iteration it adds a long dependency chain and shuffle-port pressure to every iteration, and can make the vector loop slower than scalar.
</details>

<details class="check"><summary>A loop computes <code>y[i] = x[i] &gt; 0 ? log(x[i]) : 0</code>. Under masking, what happens to <code>log</code> for negative lanes?</summary>
With masked execution the log is not computed (or its result discarded without raising exceptions) for inactive lanes. With an unmasked compute-both-sides-and-blend approach, log of a negative number produces NaN in those lanes and might raise an invalid-operation flag. The blend discards the value, but the flag is a side effect that strict IEEE semantics require the compiler to preserve.
</details>

<details class="check"><summary>What single property of <code>pshufb</code> makes it useful for dequantizing 4-bit weights?</summary>
It is a 16-entry byte lookup table applied in every 128-bit chunk, indexed by the low 4 bits of each byte, so a 4-bit code can be mapped to an int8 value (or codebook entry) for 16–64 lanes in one instruction.
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Intel 64 and IA-32 Architectures Software Developer's Manual, Vol. 1 ch. 10–15 | Intel, 2024 | intel.com/sdm | SSE/AVX/AVX-512 register files, masks, instruction semantics |
| Intel Intrinsics Guide | Intel | intel.com/content/www/us/en/docs/intrinsics-guide | Intrinsic names, operations, latency/throughput tables |
| uops.info: Characterizing Latency, Throughput, and Port Usage of Instructions on Intel Microarchitectures | Abel & Reineke, ASPLOS 2019 | uops.info | FMA latency/throughput, shuffle port pressure |
| Arm Architecture Reference Manual (A-profile), SVE/SVE2 | Arm, 2024 | developer.arm.com | Predicates, first-faulting loads, VLA registers |
| RISC-V V Vector Extension 1.0 | RISC-V International, 2021 | github.com/riscv/riscv-v-spec | `vsetvl`, LMUL, mask register |
| This book's measurements | `tools/measurements/results.json` | [repo](https://github.com/karam-nus/vectorization/tree/main/tools/measurements) | leaky ReLU 15.6× with AVX-512 masking |

---

**Next:** [Chapter 4 — Seeing Vectors: A Visual Grammar](./04_seeing_vectors.md) — the seven drawings to make before writing a vectorized kernel.
