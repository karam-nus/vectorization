---
title: "Chapter 15 — GPUs: SIMT, Warps & Tiles"
---

[← Back to Table of Contents](./README.md)

# Chapter 15 — GPUs: SIMT, Warps & Tiles

> *"A GPU is a vector machine that lets you pretend it is not, and charges you for every time you forget."*

## Overview

A CUDA kernel is written as scalar code for one thread. The hardware groups 32 threads into a **warp**
(AMD: 64 or 32 into a *wavefront*) and issues each instruction once for the whole group. The warp is
the vector, and the threads are its lanes. Every idea from Chapters 3–9 reappears with a new name:
masks become the active mask, contiguity becomes coalescing, shuffles become warp shuffles, the tail
becomes wave quantization, and the widest vector instruction of all is the tensor core's matrix multiply.
This chapter maps the vocabulary, then shows how Triton lets you program the block, not the thread.

<div class="versus">
<div class="fighter p1"><div class="f-name">CPU SIMD</div><div class="f-title">AVX-512 core · Emerald Rapids</div>
<div class="f-stat"><span>Lanes / instr.</span><span class="f-bar" style="--v:.5"></span></div>
<div class="f-stat"><span>Lanes in flight</span><span class="f-bar" style="--v:.08"></span></div>
<div class="f-stat"><span>Latency hiding</span><span class="f-bar" style="--v:.35"></span></div>
<div class="f-stat"><span>Branchy code</span><span class="f-bar" style="--v:.8"></span></div>
<div class="f-stat"><span>Programmer sees width</span><span class="f-bar" style="--v:1"></span></div>
<div class="f-quote">"16 lanes, and a huge out-of-order engine to keep them busy."</div></div>
<div class="vs">VS</div>
<div class="fighter p2"><div class="f-name">GPU SIMT</div><div class="f-title">NVIDIA SM · warps of 32</div>
<div class="f-stat"><span>Lanes / instr.</span><span class="f-bar" style="--v:1"></span></div>
<div class="f-stat"><span>Lanes in flight</span><span class="f-bar" style="--v:1"></span></div>
<div class="f-stat"><span>Latency hiding</span><span class="f-bar" style="--v:.95"></span></div>
<div class="f-stat"><span>Branchy code</span><span class="f-bar" style="--v:.4"></span></div>
<div class="f-stat"><span>Programmer sees width</span><span class="f-bar" style="--v:.25"></span></div>
<div class="f-quote">"Thousands of lanes; when one warp waits, another runs."</div></div>
</div>
<div class="genre-caption">Same bargain, opposite strategy for waiting. A CPU core hides memory latency with caches, prefetchers and out-of-order execution of one instruction stream. An SM hides it by switching among dozens of resident warps. Bars are qualitative.</div>

---

## The vocabulary map

| CPU SIMD idea (chapter) | GPU SIMT name | What changes |
|---|---|---|
| lane (3) | thread | you write code for one lane |
| vector register / instruction (3) | warp of 32 threads, one instruction per warp | width fixed at 32 on NVIDIA; 64 or 32 on AMD |
| mask, predication (8) | active mask; divergence | hardware manages it; both paths of a divergent branch run serially |
| contiguous vector load (7) | **coalesced** load | 32 threads' addresses merged into 32-byte sectors |
| gather (7) | every load | each thread supplies its own address; coalescing makes neighbours cheap |
| shuffle / permute (3) | `__shfl_sync`, `__shfl_xor_sync` | cross-lane movement within a warp, no shared memory needed |
| horizontal reduction (3) | warp reduction: 5 shuffle-add steps | $\log_2 32 = 5$ |
| register blocking (18) | registers per thread, shared memory per block | occupancy trade-off |
| tail (8) | partial warps, partial blocks, **wave quantization** | the last wave of blocks may leave SMs idle |
| matrix instruction (16) | `mma.sync`, `wgmma`, `tcgen05.mma` | issued by a warp, a warpgroup, or a single thread |

<div class="callout myth"><span class="callout-title">Myth</span>
<p>"GPU threads are independent, so divergence doesn't matter." Since Volta, NVIDIA GPUs track a program
counter per thread (independent thread scheduling), which makes divergent code <em>correct</em> in more
cases, for example locks within a warp. The two sides of a branch still execute one after the other with
partial masks. Divergence costs exactly what Chapter 8 said it costs.</p>
</div>

---

## Coalescing: the memory strip with 32 lanes

A warp's load instruction presents 32 addresses at once. The memory system serves them in 32-byte
sectors, so the cost is the number of distinct sectors touched (`vzlab.memory.warp_sectors`):

<div class="mem">
<div class="mem-row"><span class="mem-name">x[tid]</span><span class="cl touched"><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i></span><span class="cl touched"><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i></span><span class="cl touched"><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i></span><span class="cl touched"><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i><i class="use"></i></span><span class="mem-note">4 sectors · 128 B · 100 %</span></div>
<div class="mem-row"><span class="mem-name">x[2*tid]</span><span class="cl touched"><i class="use"></i><i class="load"></i><i class="use"></i><i class="load"></i><i class="use"></i><i class="load"></i><i class="use"></i><i class="load"></i></span><span class="cl touched"><i class="use"></i><i class="load"></i><i class="use"></i><i class="load"></i><i class="use"></i><i class="load"></i><i class="use"></i><i class="load"></i></span><span class="mem-note">… 8 sectors · 50 %</span></div>
<div class="mem-row"><span class="mem-name">x[32*tid]</span><span class="cl touched"><i class="use"></i><i class="load"></i><i class="load"></i><i class="load"></i><i class="load"></i><i class="load"></i><i class="load"></i><i class="load"></i></span><span class="cl"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></span><span class="cl"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></span><span class="cl"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></span><span class="cl touched"><i class="use"></i><i class="load"></i><i class="load"></i><i class="load"></i><i class="load"></i><i class="load"></i><i class="load"></i><i class="load"></i></span><span class="mem-note">… 32 sectors · 12.5 %</span></div>
</div>
<div class="genre-caption">fp32, 32-byte sectors. Thread tid reads the cyan element. Reading a column of a row-major matrix (stride = row length) is the third row: every thread pays for its own sector.</div>

The standard cure for column access is the shared-memory transpose: load a tile row-wise (coalesced)
into shared memory, synchronise, and read it column-wise from shared memory. Shared memory has its own
rule: 32 **banks**, 4 bytes wide. Threads that hit the same bank at different addresses serialise.
A 32 × 32 fp32 tile read by column puts every element of a column in the same bank, and padding each
row to 33 elements shifts each row by one bank and removes the conflict.

```cuda
__shared__ float tile[32][33];             // +1 column: column reads hit 32 different banks
tile[threadIdx.y][threadIdx.x] = in[(by + threadIdx.y) * n + bx + threadIdx.x];   // coalesced
__syncthreads();
out[(bx + threadIdx.y) * n + by + threadIdx.x] = tile[threadIdx.x][threadIdx.y];  // coalesced
```

---

## Vectorization *inside* a thread still matters

SIMT does not make per-thread vectorization obsolete. A thread that loads one `float` per instruction
issues four times as many load instructions as one that loads a `float4` (a 128-bit `LDG.128`). For
bandwidth-bound kernels, NVIDIA's own guidance and most hand-written kernels use vector loads:

```cuda
// 4 elements per thread per load; requires 16-byte alignment and n % 4 == 0 (or a tail)
float4 v = reinterpret_cast<const float4*>(x)[i];
v.x = fmaxf(v.x, 0.f); v.y = fmaxf(v.y, 0.f); v.z = fmaxf(v.z, 0.f); v.w = fmaxf(v.w, 0.f);
reinterpret_cast<float4*>(y)[i] = v;
```

Packed half-precision math (`half2`, `__hfma2`) does the same for arithmetic: two fp16 lanes in one
32-bit register, one instruction.

---

## Warp-level reductions: shuffles instead of memory

A sum across a warp's 32 lanes is the horizontal reduction of Chapter 3, done with `__shfl_xor_sync`:
each step, every lane adds the value of the lane whose index differs in one bit.

```cuda
__device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)      // 16, 8, 4, 2, 1: log2(32) = 5 steps
        v += __shfl_xor_sync(0xffffffff, v, offset);       // butterfly: all lanes end with the sum
    return v;
}
```

Five shuffle-add steps, no shared memory, no `__syncthreads()`. Softmax, LayerNorm and every other
row reduction in an LLM kernel are built on this (Chapter 18). The summation order is a butterfly, so
the result differs from a sequential sum in the last bits, just as on a CPU (Chapter 6).

---

## Tensor cores: the warp issues a matrix

The largest vector instruction on a GPU is a matrix multiply-accumulate issued cooperatively:

| Generation | Instruction | Issued by | Operands live in | Shape per instruction (fp16/bf16) |
|---|---|---|---|---|
| **Volta** (2017) | `mma.sync` / `wmma` | one warp | registers | 16×16×16 (wmma API) built from 4×4×4 per-core steps |
| **Ampere** (2020) | `mma.sync` | one warp | registers; `cp.async` stages tiles in shared memory | m16n8k16 |
| **Hopper** (2022) | `wgmma.mma_async` | a **warpgroup** (4 warps, 128 threads) | A in registers or shared memory, B in shared memory | m64nNk16, N up to 256 |
| **Blackwell** (2024) | `tcgen05.mma` (UMMA) | **one thread** for the CTA | shared memory and **Tensor Memory (TMEM)**; accumulators in TMEM | up to m128/m256 across a CTA pair |

The trend is the history of Chapter 5 compressed into seven years. The unit of decision grows from a
warp to a warpgroup to a whole thread block, and operands move out of the register file into memory
dedicated to the matrix unit. The warp's 32 lanes are no longer the vector; the tile is. Chapter 16
draws what happens inside.

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>On Blackwell a single thread issues a matrix multiply whose accumulators never pass through any
thread's registers. In what sense is that still "SIMT"? Programming models lag hardware. CUTLASS's CuTe
layouts and Triton's block programs exist because "one thread, one element" stopped describing the
machine several generations ago.</p>
</div>

---

## Triton: program the block, not the thread

Triton (Tillet, Kung & Cox, 2019) moves the programmer one level up. A Triton program describes what
one **block** of the output does, using whole-tile operations. The compiler decides how threads,
warps, vector loads and tensor-core instructions implement it:

```python
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(out_ptr, in_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)                               # one program per row
    cols = tl.arange(0, BLOCK)                           # a vector of BLOCK column indices
    mask = cols < n_cols                                 # the tail, as a mask (Chapter 8)
    x = tl.load(in_ptr + row * stride + cols, mask=mask, other=-float("inf"))  # padding = -inf
    x = x - tl.max(x, axis=0)                            # block reduction -> warp shuffles
    e = tl.exp(x)
    tl.store(out_ptr + row * stride + cols, e / tl.sum(e, axis=0), mask=mask)
```

Every line is a vector operation over `BLOCK` elements. The mask handles the tail. The padding value
`-inf` is chosen so the padded lanes cannot win the max or add to the sum, which is the padding-legality
rule of Chapter 10. `tl.max` and `tl.sum` compile to the warp shuffles above. `tl.dot` on two tiles compiles
to `mma`/`wgmma`. TorchInductor generates kernels in this style for most fused PyTorch operations
(Chapter 12).

---

## Wave quantization: the GPU's tail

A GPU runs blocks in **waves**: with $S$ streaming multiprocessors and one block per SM at a time, a
grid of $G$ blocks takes $\lceil G/S \rceil$ waves. The last wave may be mostly empty:

$$\text{efficiency} = \frac{G}{S \cdot \lceil G/S \rceil}$$

**Numerical check.** An A100 has 108 SMs. A GEMM tiled into 108 blocks runs in one wave at 100 %
efficiency; 109 blocks need two waves, so efficiency is $109 / 216 = 50.5\,\%$. One extra row of the
output matrix can nearly double the runtime of a small GEMM. That is Chapter 8's tail, scaled up from lanes
to SMs, and it is why NVIDIA's performance guide discusses tile counts relative to SM counts and why
matrix dimensions in models are padded to multiples of 64 or 128.

| GPU-level tail | Cause | Mitigation |
|---|---|---|
| **Partial warp** | block size not a multiple of 32 | choose block sizes in multiples of 32 |
| **Partial tile** | matrix dims not multiples of the tile (e.g. 128) | pad dims; masked loads (Triton `mask=`) |
| **Wave quantization** | grid size not a multiple of the SM count | different tile size, split-K, stream-K or persistent kernels |
| **Tensor-core shape alignment** | dims not multiples of 8 (fp16) or 16 (int8) | pad; NVIDIA's guide recommends these multiples, larger ones for best efficiency |

---

## Key Takeaways

- SIMT is SIMD with the vector hidden: a warp of 32 threads is one vector instruction, and all of Chapters 3–9 apply with new names.
- Coalescing is the memory strip for warps: contiguous fp32 per warp is 4 sectors; stride 32 is 32 sectors at 12.5 % efficiency.
- Per-thread vectorization (`float4`, `half2`) still pays for bandwidth-bound kernels.
- Warp reductions are 5 shuffle-add steps; their order differs from sequential summation.
- Tensor cores grew from warp-issued (Volta) to warpgroup (Hopper) to single-thread-issued with dedicated tensor memory (Blackwell); the tile, not the warp, is now the vector.
- Wave quantization is the GPU's tail: 109 blocks on 108 SMs runs at about 50 % efficiency.

## Check Yourself

<details class="check"><summary>A kernel reads <code>A[tid * 33]</code> (fp32) for a 32-thread warp. How many 32-byte sectors?</summary>
32: consecutive threads are 132 bytes apart, more than a sector, so each thread touches its own sector (12.5 % efficiency).
</details>

<details class="check"><summary>Why does padding a shared-memory tile from 32 to 33 columns remove bank conflicts on column reads?</summary>
Bank = (address / 4) mod 32. With 32 columns, element (r, c) is in bank c for every row r, so a column read hits one bank 32 times. With 33 columns it is in bank (33r + c) mod 32 = (r + c) mod 32, so a column spans all 32 banks.
</details>

<details class="check"><summary>In the Triton softmax, why <code>other=-float("inf")</code> rather than 0?</summary>
Masked-off lanes must not affect the result. −∞ never wins the max, and exp(−∞ − max) = 0 contributes nothing to the sum; 0 could exceed all real (negative) logits and would add exp(0 − max) to the denominator.
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| CUDA C++ Programming Guide — SIMT architecture, independent thread scheduling, warp shuffle | NVIDIA | docs.nvidia.com/cuda | Warps, divergence, `__shfl_xor_sync` |
| CUDA C++ Best Practices Guide — coalescing, shared memory bank conflicts | NVIDIA | docs.nvidia.com/cuda | Sectors, transpose with padding |
| PTX ISA — `mma`, `wgmma`, `tcgen05` | NVIDIA | docs.nvidia.com/cuda/parallel-thread-execution | Tensor-core instruction shapes and issue granularity |
| NVIDIA A100 / H100 architecture whitepapers; Blackwell technical brief | NVIDIA, 2020–2024 | nvidia.com | SM counts, warpgroups, TMEM |
| CUTLASS tutorials: WGMMA and Blackwell UMMA/TMEM | Colfax Research, 2024–2025 | research.colfax-intl.com | Operand placement and single-thread MMA issue |
| Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations | Tillet, Kung & Cox, MAPL 2019 | doi:10.1145/3315508.3329973 | Block-level programming model |
| NVIDIA Deep Learning Performance Guide — matrix multiplication background, wave quantization | NVIDIA | docs.nvidia.com/deeplearning/performance | Tile and wave quantization, alignment multiples |
| AMD CDNA ISA reference guides | AMD | amd.com | 64-wide wavefronts, EXEC mask, MFMA |

---

**Next:** [Chapter 16 — Matrix Engines: Systolic Arrays, Tensor Cores, AMX & SME](./16_matrix_engines.md) — inside the box where one instruction does thousands of multiply-adds.
