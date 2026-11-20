---
title: "Chapter 18 — Vectorizing ML Kernels"
---

[← Back to Table of Contents](./README.md)

# Chapter 18 — Vectorizing ML Kernels

> *"In a transformer the matrix multiplies are the easy part. Everything between them decides whether the matrix engine waits."*

## Overview

This chapter applies the whole book to the kernels that make up a modern model. It covers elementwise
activations and the vectorized math behind `exp`, and row reductions (softmax, LayerNorm, RMSNorm) and
the online algorithms that cut their memory passes. It derives the GEMM microkernel's register blocking
from first principles and explains FlashAttention as five techniques from Chapter 10. It ends with
quantized dot products, convolution, embeddings and sampling. Each section names the species
(Chapter 2), the limiting roof (Chapter 9) and the techniques (Chapter 10) that apply.

<div class="roll" style="--cols:16">
<div class="roll-title">Memory passes for softmax over one row of scores (time →)</div>
<div class="roll-row"><span class="roll-name">3-pass</span><span class="roll-track"><span class="note n5" style="grid-column:1 / span 4">read: max</span><span class="note n5" style="grid-column:5 / span 4">read: Σ exp</span><span class="note n5" style="grid-column:9 / span 4">read: exp/Σ</span><span class="note n4" style="grid-column:13 / span 4">write</span></span></div>
<div class="roll-row"><span class="roll-name">online</span><span class="roll-track"><span class="note n2" style="grid-column:1 / span 4">read: max + Σ</span><span class="note n2" style="grid-column:5 / span 4">read: exp/Σ</span><span class="note n4" style="grid-column:9 / span 4">write</span></span></div>
<div class="roll-row"><span class="roll-name">fused in GEMM</span><span class="roll-track"><span class="note n0" style="grid-column:1 / span 3">in registers</span><span class="note n0" style="grid-column:4 / span 2">P·V</span></span></div>
<div class="roll-axis"><span></span><span class="ticks"><span>0</span><span></span><span></span><span></span><span>1×</span><span></span><span></span><span></span><span>2×</span><span></span><span></span><span></span><span>3×</span><span></span><span></span><span>4×</span></span></div>
<div class="roll-legend">Each bar is one pass over the row in memory. Softmax is memory-bound (≈5 FLOP per element), so passes are runtime. Online softmax saves one; FlashAttention never writes the scores at all.</div>
</div>

---

## Elementwise: activations and the vectorized `exp`

Activations are **maps**: docile, memory-bound in isolation, and best fused into the kernel that
produced their input (Chapter 12). The interesting part is the math function inside. A scalar libm
`expf` has branches for special cases. A vectorized `exp` has to be branch-free over all lanes, and
the standard construction is:

1. **Range reduction.** Write $x = k \ln 2 + r$ with $k = \operatorname{round}(x / \ln 2)$ and $|r| \le \ln 2 / 2$.
2. **Polynomial.** Approximate $e^r$ by a degree-5 to degree-7 polynomial evaluated with FMAs (Horner's
   scheme), with no table lookups and no branches.
3. **Reconstruction.** $e^x = 2^k e^r$: add $k$ to the exponent bits of the float, a single integer add.
4. **Special cases** (overflow, underflow, NaN) are handled with compares and blends at the end.

**Numerical check.** $x = 3$: $k = \operatorname{round}(3 / 0.6931) = 4$, $r = 3 - 4 \cdot 0.6931 = 0.2274$,
$e^r = 1.2553$, $2^4 \cdot 1.2553 = 20.085$; $e^3 = 20.0855$. ✓

This is what SLEEF, glibc's `libmvec`, Intel's SVML and every GPU `__expf` do. It is also why `-ffast-math`
or `-fno-math-errno` lets compilers call these vector variants (Chapter 6). GELU's `tanh` and `erf` are
built the same way, or replaced by cheaper approximations. Hardware special-function units (GPU SFUs)
evaluate a few functions such as `exp2` at reduced precision, and newer GPUs added SFU throughput
specifically because attention's softmax had become the bottleneck (Chapter 5).

---

## Row reductions: softmax, LayerNorm, RMSNorm

All three reduce along the hidden or vocabulary axis and then apply a map along the same axis.
Vectorizing them means **two-level reductions**: per-lane partial results across the row (vertical,
cheap), then one horizontal reduction (Chapter 3), then the map.

### Softmax: three passes, then two, then zero

The textbook stable softmax reads the row three times:

$$m = \max_j x_j, \qquad \ell = \sum_j e^{x_j - m}, \qquad y_j = e^{x_j - m} / \ell$$

The **online softmax** (Milakov & Gimelshein, 2018) computes $m$ and $\ell$ in a single pass by
rescaling the running sum whenever the running maximum grows:

$$m' = \max(m, x_j), \qquad \ell' = \ell \, e^{m - m'} + e^{x_j - m'}$$

**Numerical check.** Row $(1, 3, 2)$. Start $m = -\infty$, $\ell = 0$. After 1: $m = 1$, $\ell = 1$. After 3:
$m' = 3$, $\ell' = 1 \cdot e^{-2} + 1 = 1.1353$. After 2: $m' = 3$, $\ell' = 1.1353 + e^{-1} = 1.5032$.
Direct: $e^{-2} + e^{0} + e^{-1} = 0.1353 + 1 + 0.3679 = 1.5032$. ✓
`vzlab.patterns.online_softmax_stats` implements it vectorized over rows and is tested against the
direct formula.

### LayerNorm and RMSNorm

LayerNorm needs a mean and a variance. Computing $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ in one pass is fast and
**numerically dangerous**: when the mean is large relative to the spread, the subtraction cancels
catastrophically in float32. Welford's online algorithm (and its parallel merge form by Chan et al.) gives
a stable one-pass variance and combines across lanes and threads with an associative update. RMSNorm
drops the mean entirely ($y = x / \sqrt{\overline{x^2} + \epsilon}$), one reduction instead of two. That
is part of why it is cheaper and fuses more easily.

| Kernel | Reductions per row | Passes (naive → best) | Limit | Techniques |
|---|:---:|:---:|---|---|
| **Softmax** | max, sum | 3 → 2 (online) → 0 (fused into attention) | memory | On, Rd, Fs |
| **LayerNorm** | mean, variance | 3 → 2 (Welford) → fused with residual add | memory | On, Rd, Fs |
| **RMSNorm** | mean of squares | 2 → 1 fused with residual and quantization | memory | Rd, Fs |
| **Cross-entropy** | logsumexp | 2 → 1 (online) → chunked over vocabulary | memory | On, Rd, Tl |

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>The final cross-entropy over a 128 000-token vocabulary for a batch of 8 000 tokens produces a logits
matrix of a billion floats. Chunked or "fused linear cross-entropy" kernels never materialise it: they
compute logits tile by tile, keep an online logsumexp, and discard each tile. Which other place in your
training loop materialises a huge tensor only to reduce it immediately?</p>
</div>

---

## GEMM: register blocking from first principles

High-performance GEMM on a CPU (GotoBLAS, BLIS, OpenBLAS, oneDNN) is five loops around a tiny
**microkernel** that keeps an $M_R \times N_R$ block of $C$ in vector registers. Per step of $k$, it
broadcasts $M_R$ values of $A$, loads $N_R / V$ vectors of $B$, and issues $M_R N_R / V$ FMAs:

<div class="regs">
<div class="regs-title">One k-step of a 6 × 32 fp32 microkernel (AVX-512, V = 16): 6 broadcasts, 2 loads, 12 FMAs</div>
<div class="reg-row"><span class="reg-name">B[k, 0:32]</span><span class="reg-lanes"><span class="ln l2 wide4">b0 … b15 (zmm)</span><span class="ln l2 wide4">b16 … b31 (zmm)</span></span><span class="reg-note">2 vector loads</span></div>
<div class="reg-row"><span class="reg-name">A[0..5, k]</span><span class="reg-lanes"><span class="ln l3">a0</span><span class="ln l3">a1</span><span class="ln l3">a2</span><span class="ln l3">a3</span><span class="ln l3">a4</span><span class="ln l3">a5</span></span><span class="reg-note">6 broadcasts</span></div>
<div class="reg-row"><span class="reg-name">C[0..5, 0:32]</span><span class="reg-lanes"><span class="ln wide4">c00…c0,15</span><span class="ln wide4">c0,16…c0,31</span></span><span class="reg-note">× 6 rows = 12 zmm accumulators</span></div>
</div>

The registers constrain the block: accumulators plus the B vectors plus a broadcast register must fit
in the register file:

$$\frac{M_R N_R}{V} + \frac{N_R}{V} + 1 \le 32 \quad\text{(AVX-512)}$$

and the ratio of FMAs to loads decides whether the kernel is FMA-bound or load-bound:

$$\frac{\text{FMAs}}{\text{loads}} = \frac{M_R N_R / V}{M_R + N_R / V}$$

**Numerical check.** $M_R = 6$, $N_R = 32$, $V = 16$: registers $12 + 2 + 1 = 15 \le 32$ ✓;
FMA:load $= 12 / 8 = 1.5$. With two FMA ports and two load ports per cycle, 1.5 > 1 means the FMA
ports, not the loads, are the bottleneck: the kernel can approach peak. Contrast the dot product, at
FMA:load $= 1/2$, which was load-bound at 70 GFLOP/s (Chapter 14). Bigger blocks (e.g. $M_R = 14$, $N_R = 32$:
$28 + 2 + 1 = 31$ registers, ratio 1.75) push further. Twelve or more independent accumulators also
satisfy Chapter 9's latency rule of 8 chains.

The outer loops block for L1, L2 and L3 (tiles of $A$ and $B$ sized to each cache) and **pack** the
panels into contiguous, aligned buffers in exactly the order the microkernel reads them. That is
layout transformation (Chapter 7) paid once per panel and amortised over many microkernel calls. This
book's unblocked `mm_ikj` reached 46.6 GFLOP/s, 25 % of the core's peak. Register blocking, cache
blocking and packing are the remaining distance, and on this core AMX raises the ceiling a further
13× (Chapter 16).

---

## Attention: FlashAttention as five techniques

Standard attention computes $S = QK^\top$ (a $T \times T$ matrix per head), writes it to memory,
runs softmax over it, writes $P$, and multiplies $PV$. For long $T$ the $T^2$ intermediate dominates
memory traffic. FlashAttention (Dao et al., 2022) is a composition of Chapter 10's elements:

| Technique | What it does in FlashAttention |
|---|---|
| **Tiling** (Tl) | blocks of $Q$ rows and $K, V$ columns sized to on-chip SRAM |
| **Fusion** (Fs) | $QK^\top$, softmax and $PV$ in one kernel; $S$ and $P$ never reach HBM |
| **Online algorithm** (On) | running max and sum per row, rescaling the partial output when the max changes |
| **Partial-sum reduction** (Rd) | output accumulates across $K, V$ blocks; split-K variants (Flash-Decoding) for long contexts |
| **Matrix engine + low precision** (Mm, Lp) | both GEMMs on tensor cores in fp16/bf16 (fp8 in FlashAttention-3), accumulation in fp32 |

The paper shows HBM accesses drop from $\Theta(Td + T^2)$ for standard attention to
$\Theta(T^2 d^2 / M)$ for an SRAM of size $M$. For typical $d$ = 64–128 and $M$ of about 100 KB that is
many times fewer bytes. The FLOPs are unchanged, so the kernel moves from the memory-bound part of the
roofline toward the compute-bound part.

**Decoding is different.** With one new query token per sequence, $QK^\top$ is a matrix-vector product
against the whole KV cache: ~1 FLOP per byte of cache read. No tiling fixes that intensity; only
reading fewer bytes does. Hence grouped-query attention (fewer KV heads), KV-cache quantization, and
batching many sequences so that the *weights*, at least, are shared (Chapter 19). Paged attention stores
the cache in fixed-size blocks and gathers them by a block table, a **gather** whose rows are long and
contiguous, which is the friendly case from Chapter 7.

---

## Quantized dot products

| Kernel | Vector trick | Where it bites |
|---|---|---|
| **int8 × int8 → int32** (CPU) | `vpdpbusd` (VNNI) multiplies u8 × s8 in groups of 4 into int32 lanes | VNNI wants *unsigned* activations: add 128 to activations, then subtract $128 \cdot \sum_k w_k$ (precomputed per column) |
| **int8 on GPU** | `dp4a`, int8 tensor cores | requantization (int32 → int8 with a scale) in the epilogue |
| **4-bit weights** | unpack nibbles with shift/mask or a `pshufb` lookup table into int8 (or fp16), then dot | at batch 1 dequantization hides under memory time; at large batch it competes with the GEMM |
| **fp8 / fp4 block-scaled** | hardware block scaling (NVFP4: one fp8 scale per 16 values; MXFP4: one E8M0 scale per 32) | scales must be laid out where the tensor core expects them |

**Numerical check of the VNNI offset trick.** Activations $a = (-3, 5)$, weights $w = (2, 1)$:
$a \cdot w = -6 + 5 = -1$. Shift $a$ by 128: $(125, 133)$, unsigned. $(125, 133) \cdot (2, 1) = 250 + 133 = 383$.
Subtract $128 \cdot (2 + 1) = 384$: $383 - 384 = -1$. ✓

The [quantization book](https://karam-nus.github.io/quantization) in this knowledge base covers formats
and accuracy; this chapter's point is that every quantized kernel is first a vectorization problem of
unpacking, widening and dot-product-accumulating.

---

## Convolution, embeddings and sampling

**Convolution.** *im2col* copies each receptive field into a column so the convolution becomes one
GEMM (Chapter 10, Rf). It is simple and uses the best GEMM, but multiplies memory traffic by the kernel
area. Direct convolution in channels-last or blocked layouts (NHWC, nChw16c) vectorizes along channels
without the copy. **Depthwise** convolution has almost no reuse (each filter touches one channel). It is a
vector-unit kernel, not a matrix-engine kernel, which is why depthwise layers run far below their
FLOP-proportional share on accelerators. Winograd transforms trade multiplies for adds for 3×3 kernels.

**Embeddings.** Forward is a gather of whole rows (vectorize along the row). `EmbeddingBag` fuses the
gather with a sum or mean pool. The backward pass is a scatter-add with conflicts (Chapter 10, Hg),
usually solved by sorting indices and segment-reducing.

**Sampling.** Top-k and top-p need a partial sort and a prefix sum over probabilities (sort + scan).
The **Gumbel-max trick** turns sampling into an argmax: $\arg\max_i (\log p_i + g_i)$ with
$g_i \sim \text{Gumbel}(0, 1)$ samples from $p$. That is a map plus a reduction, both docile, with no
sort and no scan.

---

## A fused-kernel checklist

| Ask | Because |
|---|---|
| How many times does this sequence of ops read and write the tensor? | memory-bound kernels cost passes (Chapter 9) |
| Can the elementwise tail live in the previous GEMM's epilogue? | bias, activation, residual and quantization are free there |
| Is any reduction recomputing an intermediate that could be carried online? | softmax, logsumexp, variance |
| Does the reduction axis have a tail relative to the lane or warp width? | odd hidden sizes waste lanes (Chapter 8) |
| Are accumulators wide enough? | fp16/bf16 sums need fp32 accumulators; int8 dots need int32 |
| Is the result deterministic across batch sizes and hardware? | reduction trees differ by $V$ and by split (Chapter 6) |

---

## Key Takeaways

- Vectorized math functions are branch-free range reduction + polynomial + exponent reconstruction; that is what `-fno-math-errno` and fast-math let compilers call.
- Row reductions (softmax, LayerNorm, RMSNorm, cross-entropy) are memory-bound; online algorithms cut passes, and fusion removes them.
- A GEMM microkernel keeps an $M_R \times N_R$ block of $C$ in registers; the register budget and the FMA:load ratio ($1.5$ for 6 × 32 on AVX-512) decide whether it can reach peak.
- FlashAttention is tiling + fusion + online softmax + partial sums + tensor cores; it changes memory traffic, not FLOPs.
- Decode attention is a GEMV over the KV cache, about 1 FLOP/B; only fewer bytes (GQA, KV quantization) or more reuse (batching) help.
- Quantized kernels are vectorization problems first: unpacking, widening and dot-product instructions with sign and scale bookkeeping.

## Check Yourself

<details class="check"><summary>Why can't you vectorize the online softmax recurrence across the columns of one row the same way you vectorize a sum?</summary>
You can: the (m, ℓ) pair has an associative merge, (m₁, ℓ₁) ⊕ (m₂, ℓ₂) = (max, ℓ₁e^{m₁−max} + ℓ₂e^{m₂−max}). Each lane keeps its own (m, ℓ) over a strided subset of columns, and the lanes are merged with a tree at the end, exactly like partial sums.
</details>

<details class="check"><summary>For an AVX-512 fp32 microkernel with M_R = 8 and N_R = 48, check the register budget and the FMA:load ratio.</summary>
Accumulators 8·48/16 = 24, B vectors 3, broadcast 1: 28 ≤ 32 ✓. FMA:load = 24 / (8 + 3) ≈ 2.2, FMA-bound.
</details>

<details class="check"><summary>Why does a 4-bit weight-only GEMM get relatively slower as the batch size grows?</summary>
At batch 1 it is a GEMV, memory-bound on weight bytes, so 4-bit weights cut time and the unpacking hides under memory stalls. As the batch grows the same weights are reused across more rows, the kernel becomes compute-bound, and the unpacking/dequantization work competes directly with the multiply-adds.
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Online normalizer calculation for softmax | Milakov & Gimelshein, 2018 | arXiv:1805.02867 | Online softmax recurrence |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao et al., NeurIPS 2022 | arXiv:2205.14135 | Tiling, fusion, HBM-access bound |
| FlashAttention-3 | Shah et al., 2024 | arXiv:2407.08608 | Hopper wgmma/TMA, fp8 |
| Anatomy of High-Performance Matrix Multiplication | Goto & van de Geijn, ACM TOMS 2008 | doi:10.1145/1356052.1356053 | Layered blocking, packing, microkernel |
| BLIS: A Framework for Rapidly Instantiating BLAS Functionality | Van Zee & van de Geijn, ACM TOMS 2015 | doi:10.1145/2764454 | Microkernel register blocking |
| SLEEF: A Portable Vectorized Library of C Standard Mathematical Functions | Shibata & Petrogalli, IEEE TPDS 2020 | arXiv:2001.09258 | Vectorized exp/log/trig construction |
| Updating Formulae and a Pairwise Algorithm for Computing Sample Variances | Chan, Golub & LeVeque, 1979 | Stanford CS tech report | Parallel Welford merge |
| Efficient Memory Management for Large Language Model Serving with PagedAttention | Kwon et al., SOSP 2023 | arXiv:2309.06180 | Block-table KV cache (gather of pages) |
| A* Sampling | Maddison, Tarlow & Minka, NeurIPS 2014 | arXiv:1411.0030 | Gumbel-max trick: sampling as an argmax |
| This book's code and measurements | `code/vzlab/patterns.py`; `results.json` | [repo](https://github.com/karam-nus/vectorization) | Online softmax test; `mm_ikj`, `dot_accumulators_L1`, `amx_peak` |

---

**Next:** [Chapter 19 — Batching: Vectorization at System Scale](./19_batching_systems.md) — the batch is the widest vector you will ever control.
