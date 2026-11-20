---
title: "Appendix F — Question Bank"
---

[← Back to Table of Contents](./README.md)

# Appendix F — Question Bank

> *Sixty questions for self-study, interviews and reading groups, keyed to chapters. Answers are one click away; try first.*

## Foundations (Chapters 0–5)

<details class="check"><summary>[Ch 0] Why does a vector instruction save energy even if its arithmetic costs the same per lane?</summary>
Instruction fetch, decode, scheduling and register access (~70 pJ class at 45 nm) are paid once per instruction and shared across V lanes, while the arithmetic per lane is a few pJ.
</details>
<details class="check"><summary>[Ch 0] Name the two separate wins hidden in "I vectorized my Python".</summary>
Leaving the interpreter (no per-element dispatch) and filling SIMD lanes in the compiled code. Measured here as 86× and 9.7× on the pairwise-distance problem.
</details>
<details class="check"><summary>[Ch 1] List the six questions of a vectorization decision.</summary>
What is the vector? Is it legal? How does the data arrive? What about control? Is it worth it? Who does it?
</details>
<details class="check"><summary>[Ch 1] What distinguishes SIMT from SIMD from the programmer's point of view?</summary>
SIMT exposes scalar threads and lets hardware group them into warps and manage divergence; SIMD exposes the vector width and masks explicitly.
</details>
<details class="check"><summary>[Ch 2] Which loop species is an embedding backward pass, and what goes wrong if vectorized naively?</summary>
A scatter-add; duplicate indices in a vector lose updates.
</details>
<details class="check"><summary>[Ch 2] How do you vectorize decision-tree inference?</summary>
Across rows (or trees): one lane per row, node indices computed arithmetically, thresholds gathered per level.
</details>
<details class="check"><summary>[Ch 3] How many int8 lanes in a 256-bit register? How many fp64?</summary>
32 and 4.
</details>
<details class="check"><summary>[Ch 3] What does VNNI's vpdpbusd compute per 32-bit lane?</summary>
The sum of four u8 × s8 products added into an int32 accumulator.
</details>
<details class="check"><summary>[Ch 3] Why should horizontal reductions happen after the loop?</summary>
They cost log₂V dependent shuffle-add steps; per iteration they add a long chain and shuffle pressure.
</details>
<details class="check"><summary>[Ch 4] Which drawing answers "is it worth it?" before any code is written?</summary>
The roofline.
</details>
<details class="check"><summary>[Ch 4] In an llvm-mca timeline, what does a staircase of '=' indicate?</summary>
Instructions waiting for operands from earlier instructions: a latency (dependency) chain.
</details>
<details class="check"><summary>[Ch 5] Why did register-based vectors beat memory-to-memory vectors?</summary>
They amortise memory start-up latency by loading once and chaining operations, making short vectors profitable.
</details>
<details class="check"><summary>[Ch 5] What is function multi-versioning, and why do fixed-width ISAs need it?</summary>
Compiling several versions of a function for different ISA levels and choosing at run time; because a binary using AVX-512 will not run on CPUs without it.
</details>

## Legality and profitability (Chapters 6–9)

<details class="check"><summary>[Ch 6] State the legality rule for dependence distance d and V lanes.</summary>
Chunked vector execution is legal iff d ≤ 0 or d ≥ V.
</details>
<details class="check"><summary>[Ch 6] Is <code>a[i] = a[i-4] * 2</code> legal with V = 4? With V = 8?</summary>
d = 4: legal at V = 4 (d ≥ V), illegal at V = 8.
</details>
<details class="check"><summary>[Ch 6] Why does NumPy's <code>a[1:] = a[:-1] * 2</code> differ from the equivalent loop?</summary>
Array semantics read all inputs before writing (NumPy buffers overlapping inputs); the loop has a flow dependence and sees updated values.
</details>
<details class="check"><summary>[Ch 6] Which gcc flag alone let a <code>sqrtf</code> loop vectorize, and why?</summary>
-fno-math-errno: sqrtf may set errno, a global side effect that blocks vectorization.
</details>
<details class="check"><summary>[Ch 6] Is a vectorized float sum more or less accurate than the sequential one?</summary>
Usually more (smaller running partial sums), but never bit-identical; measured 128 vs 11–84 ulps here.
</details>
<details class="check"><summary>[Ch 7] Why did AVX-512 saxpy gain only 1.14× over SSE2 from DRAM?</summary>
DRAM bandwidth (~21 GB/s per core) is the limit; SSE2 already nearly saturates it.
</details>
<details class="check"><summary>[Ch 7] What efficiency does stride 16 give for fp32 on 64-byte lines?</summary>
6.25 %: one useful 4-byte element per 64-byte line.
</details>
<details class="check"><summary>[Ch 7] Why was vectorized AoS code with AVX2 slower than scalar?</summary>
De-interleaving fields required many shuffles and extra loads (267 instructions vs 23), saturating the shuffle port.
</details>
<details class="check"><summary>[Ch 7] What does <code>model.to(memory_format=torch.channels_last)</code> change?</summary>
The strides (making channels contiguous, NHWC order), not the logical shape.
</details>
<details class="check"><summary>[Ch 8] What does if-conversion cost, independent of the data?</summary>
The cost of both branches plus a select, for every chunk.
</details>
<details class="check"><summary>[Ch 8] Why do wider vectors suffer more from divergence?</summary>
The probability that all V lanes agree, pᵛ + (1−p)ᵛ, falls with V.
</details>
<details class="check"><summary>[Ch 8] Name four tail strategies.</summary>
Scalar epilogue, masked last vector, padding, overlapping last vector (plus VLA vsetvl).
</details>
<details class="check"><summary>[Ch 9] Define the ridge point and say why it matters for vectorization.</summary>
I* = P_peak / B. Vectorization only raises P_peak, so it helps only kernels whose intensity exceeds the scalar ridge for their memory level.
</details>
<details class="check"><summary>[Ch 9] How many accumulators saturate two FMA ports with 4-cycle latency?</summary>
8.
</details>
<details class="check"><summary>[Ch 9] A loop is 40 % of runtime and you make it 20× faster. Program speedup?</summary>
1 / (0.6 + 0.02) ≈ 1.61×.
</details>

## The toolbox (Chapters 10–14)

<details class="check"><summary>[Ch 10] Which five techniques make up FlashAttention?</summary>
Tiling, fusion, online softmax, partial-sum reduction, matrix-engine tiles (with lower precision).
</details>
<details class="check"><summary>[Ch 10] Show that the affine-pair operator for linear recurrences is associative.</summary>
((a₃,b₃)•(a₂,b₂))•(a₁,b₁) = (a₃a₂a₁, a₃a₂b₁ + a₃b₂ + b₃) = (a₃,b₃)•((a₂,b₂)•(a₁,b₁)).
</details>
<details class="check"><summary>[Ch 10] What does NumPy's <code>h[idx] += 1</code> do with duplicate indices?</summary>
Counts each index once (gather, add, scatter); use np.add.at or bincount.
</details>
<details class="check"><summary>[Ch 11] What is NumPy's per-call overhead here, and the crossover size?</summary>
About 0.23 µs per call; roughly 800 elements per call before the per-element cost dominates.
</details>
<details class="check"><summary>[Ch 11] Is <code>np.vectorize</code> vectorized?</summary>
No: it is a Python loop with broadcasting semantics (13.1 ms vs 12.8 ms for a list comprehension here).
</details>
<details class="check"><summary>[Ch 11] What is the stride of a broadcast (stretched) axis?</summary>
Zero.
</details>
<details class="check"><summary>[Ch 12] What does vmap turn <code>w @ x</code> into when x is batched and w is shared?</summary>
A matrix-matrix product: the weights are read once for the whole batch.
</details>
<details class="check"><summary>[Ch 12] What happens to <code>lax.cond</code> under vmap with a batched predicate?</summary>
Both branches are computed and a select picks per example.
</details>
<details class="check"><summary>[Ch 13] What does "interleaved count: 4" mean in a clang remark?</summary>
Four independent copies of the vector body per iteration (four accumulators / unroll).
</details>
<details class="check"><summary>[Ch 13] Why can "loop vectorized" be slower than scalar?</summary>
E.g. an in-order float reduction: the multiplies are vectorized but the adds stay serial with extra shuffles.
</details>
<details class="check"><summary>[Ch 13] Name three code-shape changes that help vectorizers.</summary>
Counted loops with local accumulators, restrict-qualified pointers, no calls in the body, SoA layout, hoisting invariant branches.
</details>
<details class="check"><summary>[Ch 14] Decode <code>svmla_n_f32_m</code>.</summary>
SVE multiply-add with a scalar operand, fp32, merging inactive lanes.
</details>
<details class="check"><summary>[Ch 14] Why does the SVE saxpy need no tail loop?</summary>
whilelt builds a predicate each iteration; the last iteration has fewer active lanes and masked memory operations.
</details>

## Accelerators and ML systems (Chapters 15–19)

<details class="check"><summary>[Ch 15] How many 32-byte sectors does a fully coalesced fp32 warp load touch?</summary>
4 (128 bytes).
</details>
<details class="check"><summary>[Ch 15] How many shuffle steps does a warp sum take?</summary>
5 (log₂ 32).
</details>
<details class="check"><summary>[Ch 15] What is wave quantization?</summary>
Idle SMs in the last wave when the number of blocks is not a multiple of the SM count.
</details>
<details class="check"><summary>[Ch 15] Who issues a tensor-core MMA on Volta, Hopper and Blackwell?</summary>
A warp; a warpgroup of four warps; a single thread (with accumulators in tensor memory).
</details>
<details class="check"><summary>[Ch 16] Operand reuse of an N × N tile operation?</summary>
N/2 MACs per input operand.
</details>
<details class="check"><summary>[Ch 16] Utilisation of an isolated tile on an R × C systolic array?</summary>
K / (K + R + C − 2).
</details>
<details class="check"><summary>[Ch 16] What did AMX achieve on one core here, relative to AVX-512 fp32?</summary>
2.41 TFLOP/s bf16, about 13×.
</details>
<details class="check"><summary>[Ch 17] Why does 512 bits fit general-purpose CPUs well?</summary>
One register equals one 64-byte cache line per load; beyond it, bandwidth, permute networks and utilisation scale badly.
</details>
<details class="check"><summary>[Ch 17] Why is lower precision a triple win?</summary>
More lanes per register, fewer bytes per operand, smaller multipliers per lane.
</details>
<details class="check"><summary>[Ch 18] How many memory passes does softmax take naively, online, and fused into attention?</summary>
3, 2, 0 (scores never written).
</details>
<details class="check"><summary>[Ch 18] Why is E[x²] − E[x]² risky for LayerNorm in float32?</summary>
Catastrophic cancellation when the mean is large relative to the spread; Welford's update is stable.
</details>
<details class="check"><summary>[Ch 18] For a 6 × 32 AVX-512 microkernel, what is the FMA:load ratio?</summary>
12 FMAs / (6 broadcasts + 2 loads) = 1.5.
</details>
<details class="check"><summary>[Ch 18] How does the Gumbel-max trick help vectorization?</summary>
Sampling becomes an argmax over log-probabilities plus Gumbel noise: a map plus a reduction, no sort or scan.
</details>
<details class="check"><summary>[Ch 19] Why is batch size approximately arithmetic intensity for a linear layer?</summary>
FLOPs scale with B × weights while bytes are dominated by the weights, read once per batch.
</details>
<details class="check"><summary>[Ch 19] Why is packing without a document mask wrong?</summary>
Tokens would attend across unrelated sequences in the same row.
</details>
<details class="check"><summary>[Ch 19] How is speculative decoding a form of vectorization?</summary>
It verifies k drafted tokens in one forward pass, turning k GEMVs into one small GEMM along the time axis.
</details>

## Practice (Chapters 20–23)

<details class="check"><summary>[Ch 20] Why did full broadcasting lose to a NumPy row loop?</summary>
It materialised a 256 MB temporary, making the computation DRAM-bound; the row loop's temporaries fit in L2.
</details>
<details class="check"><summary>[Ch 20] What made rung 8 faster than the -ffast-math version without changing semantics?</summary>
Vectorizing over the j axis (outer-loop vectorization) with transposed Y: every lane owns its own accumulator, so no reduction is reordered.
</details>
<details class="check"><summary>[Ch 21] A profiler-free way to check whether a function vectorized?</summary>
Disassemble and count ymm/zmm operands and scalar ss operations in the function.
</details>
<details class="check"><summary>[Ch 21] How should a tolerance for a reordered float sum be chosen?</summary>
Relative to the magnitude of the partial sums (≈ √n·ε typical, n·ε worst), not the final value.
</details>
<details class="check"><summary>[Ch 22] Which question would have prevented the one-hot slowdown?</summary>
"Am I building a dense array to express a sparse idea?"
</details>
<details class="check"><summary>[Ch 23] What is the distributed analogue of a horizontal reduction?</summary>
The gradient all-reduce.
</details>

---

**Back to:** [Table of Contents](./README.md)
