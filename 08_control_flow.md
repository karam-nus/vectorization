---
title: "Chapter 8 — Masks, Predication, Divergence & Tails"
---

[← Back to Table of Contents](./README.md)

# Chapter 8 — Masks, Predication, Divergence & Tails

> *"A vector unit cannot take a branch. It can only take both, and then forget one."*

## Overview

Lockstep execution (promise 1 of the SIMD contract, Chapter 3) means all lanes run the same instruction.
Real code is full of decisions that differ per element: `if` statements, early exits, loops whose
trip counts vary, and arrays whose length is not a multiple of $V$. This chapter covers the four
mechanisms that reconcile the two (if-conversion, masking, compaction and tail handling), what each
costs in idle lanes, and the safety rules that decide whether the compiler may use them at all.

<div class="pool">
<div class="pool-title">lap 1 · the "if" side (x &gt; 0)</div>
<div class="lane"><span class="lane-tag">lane 0</span><span class="swimmer" style="--sw:#22d3ee;--x:62%">3</span></div>
<div class="lane sit"><span class="lane-tag">lane 1</span><span class="swimmer">-2</span></div>
<div class="lane"><span class="lane-tag">lane 2</span><span class="swimmer" style="--sw:#22d3ee;--x:62%">4</span></div>
<div class="lane sit"><span class="lane-tag">lane 3</span><span class="swimmer">-1</span></div>
<div class="lane sit"><span class="lane-tag">lane 4</span><span class="swimmer">-8</span></div>
<div class="lane"><span class="lane-tag">lane 5</span><span class="swimmer" style="--sw:#22d3ee;--x:62%">6</span></div>
<div class="lane sit"><span class="lane-tag">lane 6</span><span class="swimmer">-3</span></div>
<div class="lane"><span class="lane-tag">lane 7</span><span class="swimmer" style="--sw:#22d3ee;--x:62%">.5</span></div>
<div class="pool-foot"><span>4 of 8 lanes swim</span><span>time: one full lap</span></div>
</div>
<div class="pool">
<div class="pool-title">lap 2 · the "else" side</div>
<div class="lane sit"><span class="lane-tag">lane 0</span><span class="swimmer">3</span></div>
<div class="lane"><span class="lane-tag">lane 1</span><span class="swimmer" style="--sw:#f472b6;--x:62%">-2</span></div>
<div class="lane sit"><span class="lane-tag">lane 2</span><span class="swimmer">4</span></div>
<div class="lane"><span class="lane-tag">lane 3</span><span class="swimmer" style="--sw:#f472b6;--x:62%">-1</span></div>
<div class="lane"><span class="lane-tag">lane 4</span><span class="swimmer" style="--sw:#f472b6;--x:62%">-8</span></div>
<div class="lane sit"><span class="lane-tag">lane 5</span><span class="swimmer">6</span></div>
<div class="lane"><span class="lane-tag">lane 6</span><span class="swimmer" style="--sw:#f472b6;--x:62%">-3</span></div>
<div class="lane sit"><span class="lane-tag">lane 7</span><span class="swimmer">.5</span></div>
<div class="pool-foot"><span>the other 4 lanes swim</span><span>total: two laps for one lap of work</span></div>
</div>
<div class="genre-caption">A branch with a 50/50 split costs two passes at half occupancy. Lane utilisation is 50 %. The same picture describes AVX-512 masking, SVE predication and a diverged GPU warp.</div>

---

## If-conversion: turning control flow into data flow

The compiler (or you) replaces

```c
if (x[i] > 0) y[i] = x[i]; else y[i] = 0.01f * x[i];
```

with straight-line code that computes both candidate values and selects per lane:

```c
float a = x[i], b = 0.01f * x[i];
y[i] = (x[i] > 0) ? a : b;      // a select, not a branch: vblendmps / vbsl / vmerge
```

The cost model is simple: **cost of A + cost of B + one select**, regardless of the data, versus a
scalar branch whose cost depends on how predictable the data is. For leaky ReLU on random data that
trade is excellent, as measured on our Xeon (n = 4096, in L1):

| Version | ns / element | vs scalar |
|---|:---:|:---:|
| **scalar** `-O2` (a real branch, mispredicted about half the time on random data) | 0.893 | 1.0× |
| **SSE2 / AVX2** `-O3` (gcc declined: "control flow in loop") | 0.841 / 0.885 | ≈ 1× |
| **AVX-512** `-O3` (masked) | 0.057 | 15.6× |
| **AVX-512** `-O3 -ffast-math` | 0.046 | 19.5× |
| **AVX-512 intrinsics**, compare + `mask_blend`, masked tail | 0.050 | 17.9× |

When the two sides are expensive, for example a cheap common path and a costly rare one, if-conversion
can be a loss: every vector pays for the rare path. The lab below shows that the fix is often to
*skip* a side when no lane in the chunk needs it, which only pays if lanes that need it are clustered.

---

## Why the compiler may refuse: speculation must be safe

Computing the "else" value in a lane that took the "if" side is **speculation**. It is legal only if
it has no observable effect. Three things can make it observable:

1. **Memory faults.** `if (i < len) y[i] = table[idx[i]];`: computing `table[idx[i]]` for lanes past
   `len` may read an unmapped page. Masked loads (`vmaskmovps`, AVX-512 `{k}` loads, SVE predicated
   loads) suppress faults on inactive lanes, so masking makes this legal; a plain load plus blend does not.
2. **Floating-point exceptions.** `0.01f * x` cannot fault, but under the default `-ftrapping-math` gcc
   treats a possible exception flag (overflow, invalid operation) raised by an inactive lane as
   observable. With AVX2's blend, both sides execute in all lanes; with AVX-512's masked multiply,
   inactive lanes do not execute at all. That is exactly why our leaky ReLU vectorized for AVX-512 and
   not for AVX2, and why `-ffast-math` (which implies `-fno-trapping-math`) made it a little faster still.
3. **Side effects**: stores, calls, `errno` ([Chapter 6](./06_dependences.md)). Conditional stores become
   masked stores; conditional calls usually block vectorization unless a vector variant of the function exists.

<div class="callout warn"><span class="callout-title">Trap</span>
<p>Hand-written "branchless" code that computes both sides with plain loads (<code>v = cond ? a[i] : b[j]</code>
rewritten as two loads and a select) can read out of bounds on lanes where <code>j</code> is garbage. It
works until the array ends exactly at a page boundary. Use masked loads or clamp the index.</p>
</div>

---

## The masking mechanisms, compared

| ISA | Mechanism | Merge or zero inactive lanes | Faults suppressed on inactive lanes | Cost of a masked op |
|---|---|---|---|---|
| **SSE4.1 / AVX2** | vector of 0/−1 lanes + `blendv` / `and`/`or` | blend chooses | only `vmaskmov` loads/stores | an extra blend instruction |
| **AVX-512** | `k0–k7` on almost every instruction | both (`{k}` merges, `{k}{z}` zeroes) | yes, for masked loads/stores | usually free |
| **NEON** | `bsl`/`bif` bit select | blend chooses | no masked loads (scalar tail needed) | an extra select |
| **SVE / SVE2** | `p0–p15` governing predicates; `whilelt` builds tail predicates | both (`/m`, `/z`) | yes, plus first-faulting loads | usually free |
| **RISC-V V** | mask in `v0`; `vl` limits the active length | policy bits choose (`ma`/`mu`) | yes | usually free |
| **GPU SIMT** | per-thread active mask managed by hardware | inactive threads simply do not write | yes | the full instruction slot, per path |

---

## Divergence: the cost depends on the data

For an if/else executed with masks, each chunk of $V$ lanes runs side A if *any* lane needs A, and side B if
*any* lane needs B. With probability $p$ that a lane takes A, independently, a chunk needs both sides
with probability

$$P(\text{both}) = 1 - p^{V} - (1-p)^{V}$$

**Numerical check.** $p = 0.5$, $V = 16$: $1 - 2 \cdot 0.5^{16} \approx 0.99997$. Essentially every
chunk runs both sides. Utilisation is then 50 % whatever the code does. $p = 0.95$, $V = 16$:
$1 - 0.95^{16} - 0.05^{16} \approx 0.56$, so 44 % of chunks skip the rare side entirely. Wider vectors
make divergence *worse*, because more lanes must agree. That holds for AVX-512 versus AVX2 just as for
64-wide AMD wavefronts versus 32-wide NVIDIA warps.

<div class="lab" data-lab="mask"></div>

Set P = 0.5 with random order, then switch to *sorted by condition*: utilisation jumps from about 50 %
to almost 100 %, because nearly every chunk is uniform. That is the whole idea behind **sorting or
binning work by branch** before processing it:

| Domain | The divergent decision | The coherence trick |
|---|---|---|
| Ray tracing | which material each ray hit | sort rays by material before shading (wavefront path tracing) |
| Mixture-of-experts | which expert each token is routed to | permute tokens so each expert sees a contiguous block, then one GEMM per expert |
| Variable-length sequences | where each sequence ends | bucket by length, or pack sequences end to end ([Chapter 19](./19_batching_systems.md)) |
| Object detection post-processing | which boxes pass the score threshold | compact survivors first, then run NMS on the dense list |
| Sparse activations | which neurons are non-zero | gather the non-zero indices, compute densely on them |

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>Mixture-of-experts routing is a branch with as many sides as there are experts. With 64 experts and
tokens routed independently, how often would a 16-wide chunk of tokens all want the same expert? Now you
know why MoE layers sort and permute tokens before computing anything, and why that permutation, a pure
data-movement step, is a measurable part of their runtime.</p>
</div>

---

## Compaction: removing idle lanes instead of masking them

Masking keeps idle lanes in place. **Compaction** (stream compaction, filtering) packs the active
elements together so later work runs at full occupancy:

<div class="regs">
<div class="regs-title">vcompressps — keep the lanes where the mask is 1, packed to the front</div>
<div class="reg-row"><span class="reg-name">x</span><span class="reg-lanes"><span class="ln">3</span><span class="ln l4">-2</span><span class="ln">4</span><span class="ln l4">-1</span><span class="ln l4">-8</span><span class="ln">6</span><span class="ln l4">-3</span><span class="ln">.5</span></span></div>
<div class="reg-row"><span class="reg-name">k = x &gt; 0</span><span class="reg-lanes"><span class="ln mask1">1</span><span class="ln mask0">0</span><span class="ln mask1">1</span><span class="ln mask0">0</span><span class="ln mask0">0</span><span class="ln mask1">1</span><span class="ln mask0">0</span><span class="ln mask1">1</span></span></div>
<div class="reg-row"><span class="reg-name">compress(x, k)</span><span class="reg-lanes"><span class="ln hot">3</span><span class="ln hot">4</span><span class="ln hot">6</span><span class="ln hot">.5</span><span class="ln dim">·</span><span class="ln dim">·</span><span class="ln dim">·</span><span class="ln dim">·</span></span><span class="reg-note">popcount(k) = 4 → advance output by 4</span></div>
</div>

Without a compress instruction, the same thing is a **prefix sum of the mask** (each active element's
output position) followed by a scatter or permute. That is how GPUs and NumPy do it:
`x[x > 0]` is a compare, a count and a compacting copy. Compaction pays when the work after the filter
is much heavier than the filter itself.

---

## Loops that end at different times

`while (|f(x)| > tol) x = x - f(x)/f'(x);` converges in a different number of iterations per element. A
vectorized version keeps iterating until **all** lanes have converged, masking out the finished ones.
The vector does as many iterations as its *slowest* lane. Typical fixes:

- **Bound the iterations** and accept the tail cost (most vectorized math libraries do this, using a
  fixed polynomial instead of iterating to convergence).
- **Refill finished lanes** with new work items from a queue (persistent-thread style on GPUs; "lane refill" in SIMD ray tracers).
- **Early-exit tests per chunk**: `if (!any(active)) break;`, a single `kortest` / `ptest` instruction.

Search loops (`find`, `strlen`, `memchr`) are the special case "exit when any lane finds it". They
vectorize by comparing $V$ elements at once and testing the mask for any set bit. The hazard is reading
past the end of the buffer. Implementations read in aligned blocks, which never cross a page boundary
(the trick inside glibc's `strlen`), or use SVE's first-faulting loads, which stop at the first bad
page and report how far they got.

---

## Tails: the last partial vector

When $n$ is not a multiple of $V$, the last $n \bmod V$ elements need special handling. The lab in
[Chapter 3](./03_lanes_and_registers.md#interactive-who-processes-which-element) animates these:

| Strategy | Instructions for the tail | Correctness needs | Seen in |
|---|---|---|---|
| **Scalar epilogue** | up to $V - 1$ scalar iterations | nothing | gcc/clang default for SSE/AVX2 |
| **Masked last vector** | 1 masked instruction | masked loads/stores (AVX-512, SVE, RVV) | AVX-512 code, SVE `whilelt` loops, our `leaky_avx512` |
| **Padding** | 1 full instruction on padded data | buffer padded to a multiple of $V$; padding values must be harmless | many ML kernels, image libraries |
| **Overlapping last vector** | 1 full vector ending exactly at $n$, recomputing some elements | the operation is idempotent (pure maps, not `+=`) | hand-written string and memcpy routines |
| **Vector-length agnostic** | none: `vsetvl` shrinks the last iteration | ISA support (RISC-V V) | RVV code |

How much the tail costs depends on $n / V$. With the scalar-epilogue strategy, the instruction-count
speedup is

$$S_{\text{instr}} = \frac{n}{\lfloor n/V \rfloor + (n \bmod V)}$$

Numerical check: $n = 100$, $V = 16$: $\lfloor 100/16 \rfloor = 6$ vectors plus $100 \bmod 16 = 4$ scalars,
so $S = 100/10 = 10\times$ rather than 16×. At $n = 27$ it is $27 / 12 = 2.25\times$. Short rows (a
head dimension of 80, a vocabulary chunk of 1000) are tail-sensitive. Padding inner dimensions to a
multiple of 16, 64 or 128 is often a pure performance decision that looks like a modelling one
([Chapter 17](./17_architects_view.md)).

---

## Branchless idioms worth knowing

| Instead of | Write | Maps to |
|---|---|---|
| `if (x < lo) x = lo; if (x > hi) x = hi;` | `x = min(max(x, lo), hi)` | `vmaxps` + `vminps` |
| `y = x < 0 ? -x : x` | `fabsf(x)` | an AND that clears the sign bit |
| `if (c) s += v;` | `s += c ? v : 0` | masked add / select + add |
| `switch` over a few small cases | a lookup table indexed by the case | `pshufb`, `vpermps`, `tbl` |
| `if (x > t) count++;` | `count += (x > t)` | compare → mask → popcount or subtract the −1 mask |
| boolean flags in an array of structs | a bitmask per chunk | `k` registers, `movemask` |

In NumPy and PyTorch the same idioms are `np.where`, `np.clip`, `np.abs`, boolean masks multiplied in,
and `torch.where`. The rule is identical: both sides are computed for every element.

---

## Key Takeaways

- SIMD hardware handles a branch by computing both sides and selecting per lane; the cost is the sum of both paths, independent of the data.
- Speculation must be safe: memory faults need masked loads, and floating-point exception semantics are why gcc vectorized leaky ReLU (15.6×) for AVX-512 masking but not for AVX2.
- Divergence grows with width: with a 50/50 branch nearly every 16-lane chunk runs both sides; sorting or binning work by branch restores utilisation.
- Compaction removes idle lanes for later work; without a compress instruction it is a prefix sum of the mask plus a permute.
- Tails cost up to $V-1$ extra instructions; masked tails, padding, overlapping vectors and VLA `vsetvl` remove most of it.
- Many dimensions padded to multiples of 16, 64 or 128 in ML models are tail-handling decisions in disguise.

## Check Yourself

<details class="check"><summary>Why might a branchy scalar loop beat its if-converted vector version?</summary>
If one side is very expensive and rarely taken, and the branch is predictable, the scalar loop almost never pays for the expensive side, while every vector chunk that contains even one lane needing it pays for it in full. Skipping a side when its mask is empty helps only if the rare lanes are clustered.
</details>

<details class="check"><summary>With p = 0.9 and V = 8, what fraction of chunks must execute both sides of a branch?</summary>
1 − 0.9⁸ − 0.1⁸ ≈ 1 − 0.430 − 0.00000001 ≈ 0.57, so about 57 % of chunks run both sides.
</details>

<details class="check"><summary>Why can AVX-512 or SVE vectorize <code>if (i &lt; len) y[i] = a[idx[i]]</code> safely while AVX2 code that loads first and blends later cannot?</summary>
Masked loads in AVX-512 and SVE suppress memory faults for inactive lanes, so lanes beyond <code>len</code> never touch an invalid address. An unmasked load followed by a blend performs the load for every lane, which can fault on garbage indices.
</details>

<details class="check"><summary>Which tail strategy is wrong for <code>y[i] += x[i]</code>, and why?</summary>
The overlapping-last-vector strategy: it re-processes some elements, and an accumulating update is not idempotent, so those elements would be added twice.
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Conversion of control dependence to data dependence | Allen, Kennedy, Porterfield & Warren, POPL 1983 | doi:10.1145/567067.567085 | If-conversion |
| Intel AVX-512 instruction set reference (masking, `vcompressps`) | Intel SDM Vol. 2, 2024 | intel.com/sdm | Opmask merge/zero semantics, compress, fault suppression |
| The ARM Scalable Vector Extension | Stephens et al., IEEE Micro 2017 | doi:10.1109/MM.2017.35 | Governing predicates, `whilelt`, first-faulting loads |
| Megakernels Considered Harmful: Wavefront Path Tracing on GPUs | Laine, Karras & Aila, HPG 2013 | doi:10.1145/2492045.2492060 | Sorting work by branch to reduce divergence |
| GCC Optimize Options (`-ftrapping-math`) | GNU | gcc.gnu.org/onlinedocs | Why speculation of FP operations is restricted |
| This book's measurements | `tools/measurements/results.json`, `remarks.txt` | [repo](https://github.com/karam-nus/vectorization/tree/main/tools/measurements) | leaky ReLU across targets; gcc "control flow in loop" remark |

---

**Next:** [Chapter 9 — Is It Worth It? Roofline, Amdahl & Real Ceilings](./09_profitability.md) — legal and possible; now, will it pay?
