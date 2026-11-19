---
title: "Chapter 22 — Questions Worth Pondering & The Decision Guide"
---

[← Back to Table of Contents](./README.md)

# Chapter 22 — Questions Worth Pondering & The Decision Guide

> *"A good optimisation starts as a good question. Most bad ones start as an answer."*

## Overview

This chapter is the book compressed into questions, the ones experienced performance engineers ask
before writing any vector code, pinned to cork boards by theme. Each note says why the question matters
and links to the chapter with the answer. A decision guide by starting point follows, so the questions
turn into a first move. Print the boards, or better, take them into your next design review.

---

## Board 1 — Before you start

<div class="corkboard">
<div class="sticky">Is the hot spot actually in my code?<small>If it is a library call, the job is calling it fewer times with bigger inputs. <a href="./09_profitability.html">Ch 9</a></small></div>
<div class="sticky b">What is the vector here?<small>Elements, rows, heads, requests, environments? Pick the axis that is independent <em>and</em> long. <a href="./01_grand_overview.html">Ch 1</a></small></div>
<div class="sticky g">Which species of loop is this?<small>Map, reduction, stencil, scan, gather, scatter, filter, contraction, sort, pointer chase. <a href="./02_domains.html">Ch 2</a></small></div>
<div class="sticky o">Bytes or FLOPs: which do I run out of first?<small>Below the scalar ridge, wider registers buy nothing. <a href="./09_profitability.html">Ch 9</a></small></div>
<div class="sticky p">What is the best I could possibly get?<small>Roofline for the loop, Amdahl for the program. Under ~1.3×, spend the effort elsewhere. <a href="./09_profitability.html">Ch 9</a></small></div>
<div class="sticky w">Have I drawn it?<small>Register strip, iteration space, memory strip, piano roll, roofline, tile box, shape strip. <a href="./04_seeing_vectors.html">Ch 4</a></small></div>
</div>

## Board 2 — Legality and numerics

<div class="corkboard">
<div class="sticky p">Does any iteration read what an earlier one wrote?<small>Distance d with 0 &lt; d &lt; V forbids V lanes. <a href="./06_dependences.html">Ch 6</a></small></div>
<div class="sticky">Is this "reduction" actually a recurrence?<small>A sum reorders; a scan needs an associative reformulation. <a href="./10_technique_catalogue.html">Ch 10</a></small></div>
<div class="sticky b">Am I allowed to change the order of float additions?<small>Who decided, and is it documented next to the code? <a href="./06_dependences.html">Ch 6</a></small></div>
<div class="sticky g">What tolerance does reordering justify?<small>Relative to the partial sums, not the final value. <a href="./21_debugging_playbook.html">Ch 21</a></small></div>
<div class="sticky o">Do I need bit-exact reproducibility?<small>Across vector widths, thread counts and GPUs? Then fix the reduction tree explicitly. <a href="./06_dependences.html">Ch 6</a></small></div>
<div class="sticky w">Could two lanes write the same address?<small>Scatter-adds, histograms, embedding gradients. <a href="./10_technique_catalogue.html">Ch 10</a></small></div>
</div>

## Board 3 — Data and memory

<div class="corkboard">
<div class="sticky b">Is the vector axis the contiguous axis?<small>Check <code>x.stride()</code> before anything else. <a href="./07_memory_layout.html">Ch 7</a></small></div>
<div class="sticky">How many bytes do I move per useful byte?<small>Stride 16 on fp32 uses 4 of every 64 bytes. <a href="./07_memory_layout.html">Ch 7</a></small></div>
<div class="sticky g">How many times does this data cross DRAM?<small>Every temporary and every pass is a crossing. Fuse. <a href="./18_ml_kernels.html">Ch 18</a></small></div>
<div class="sticky p">Am I building a dense array to express a sparse idea?<small>One-hot, broadcast differences, masks the size of a matrix. <a href="./11_array_programming.html">Ch 11</a></small></div>
<div class="sticky o">Would a smaller type do?<small>Half the bits: twice the lanes, half the bytes, smaller multipliers. <a href="./17_architects_view.html">Ch 17</a></small></div>
<div class="sticky w">Is the data reused, and does the reuse fit in cache?<small>That is what tiling is for. <a href="./16_matrix_engines.html">Ch 16</a></small></div>
</div>

## Board 4 — Control and shape

<div class="corkboard">
<div class="sticky o">What fraction of lanes does useful work?<small>Branches, tails and ragged lengths all cost lanes. <a href="./08_control_flow.html">Ch 8</a></small></div>
<div class="sticky">Can I sort the work so neighbours agree?<small>Bin by branch, bucket by length, permute tokens by expert. <a href="./08_control_flow.html">Ch 8</a></small></div>
<div class="sticky g">Is n large relative to V?<small>A 27-element loop on 16 lanes gets 2.25×, not 16×. <a href="./03_lanes_and_registers.html">Ch 3</a></small></div>
<div class="sticky b">Are my dimensions friendly to the hardware?<small>Multiples of 16, 64, 128; 109 blocks on 108 SMs is two waves. <a href="./15_gpus_simt.html">Ch 15</a></small></div>
<div class="sticky p">If I can't vectorize inside one instance, how many instances are there?<small>Requests, beams, environments, test patterns. <a href="./02_domains.html">Ch 2</a></small></div>
</div>

## Board 5 — Hardware and co-design

<div class="corkboard">
<div class="sticky w">Which unit should run this: vector, matrix, or SFU?<small>Matrix engines idle on elementwise work; vector units starve on GEMM. <a href="./16_matrix_engines.html">Ch 16</a></small></div>
<div class="sticky">Where do operands live, and who decides the width?<small>The two questions that predict how code ports to a new chip. <a href="./05_history.html">Ch 5</a></small></div>
<div class="sticky b">What happens to this kernel on the next generation?<small>Matrix FLOPs grow faster than vector FLOPs and bandwidth. <a href="./05_history.html">Ch 5</a></small></div>
<div class="sticky g">Does a compiler target the unit I'm counting on?<small>Silicon no software reaches is dark silicon. <a href="./17_architects_view.html">Ch 17</a></small></div>
<div class="sticky o">Is the model shape a hardware decision in disguise?<small>Head dim, hidden size, vocab padding, KV heads. <a href="./17_architects_view.html">Ch 17</a></small></div>
</div>

## Board 6 — Systems and teams

<div class="corkboard">
<div class="sticky g">What is my batch, and what does it amortise?<small>Batch size is arithmetic intensity for shared weights. <a href="./19_batching_systems.html">Ch 19</a></small></div>
<div class="sticky p">What latency am I trading for throughput?<small>Every lane waits for its whole vector; every request for its batch. <a href="./19_batching_systems.html">Ch 19</a></small></div>
<div class="sticky">Will this still be vectorized after the next refactor?<small>Guard the remark or the benchmark in CI. <a href="./13_auto_vectorization.html">Ch 13</a></small></div>
<div class="sticky b">Which CPU path does production actually take?<small>Log the dispatch level next to the library version. <a href="./21_debugging_playbook.html">Ch 21</a></small></div>
<div class="sticky w">Who maintains the intrinsics when I leave?<small>Prefer code shape, then portable SIMD, then intrinsics. <a href="./14_explicit_simd.html">Ch 14</a></small></div>
</div>

## Board 7 — Questions without settled answers

<div class="corkboard">
<div class="sticky o">Will vector-length-agnostic code win on servers and phones?<small>SVE and RVV say yes; most libraries still dispatch fixed widths.</small></div>
<div class="sticky">Can compilers learn to reorder loops as well as people?<small>Rung 8 of the ladder was a loop interchange no flag would perform.</small></div>
<div class="sticky b">What replaces "one thread, one element" as the GPU programming model?<small>Triton, CuTe and tile languages are candidates.</small></div>
<div class="sticky p">How much of attention's softmax can hardware absorb?<small>SFU throughput keeps rising; fused kernels keep moving.</small></div>
<div class="sticky g">Will sparse and irregular workloads get their own vector units?<small>Gathers are legal everywhere and fast almost nowhere.</small></div>
</div>

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>Pick one note from each board and answer it for the system you work on this week. If two answers
contradict each other (for example, "the batch is the vector" and "latency matters most"), you have found the
real design decision.</p>
</div>

---

## The decision guide

Find your starting point; the first three moves are in order.

| You are starting with… | Move 1 | Move 2 | Move 3 |
|---|---|---|---|
| **A Python loop over elements, rows or pairs** | rewrite with array idioms; check temporaries vs cache ([11](./11_array_programming.md)) | look for a matmul reformulation ([10](./10_technique_catalogue.md)) | `torch.compile` / `jax.jit` / numba ([12](./12_vmap.md)) |
| **A function written for one example** | `vmap` it ([12](./12_vmap.md)) | check the batching rules that became selects or loops | `jit` for fusion |
| **A C/C++ loop the compiler won't vectorize** | read the remark ([13](./13_auto_vectorization.md)) | fix the cause by code shape: `restrict`, layout, loop order ([6](./06_dependences.md), [7](./07_memory_layout.md)) | narrow pragma or flag; intrinsics last ([14](./14_explicit_simd.md)) |
| **A vectorized loop that isn't faster** | place it on the roofline ([9](./09_profitability.md)) | memory strip: stride, layout, passes ([7](./07_memory_layout.md)) | piano roll: chains, shuffles ([4](./04_seeing_vectors.md)) |
| **A memory-bound elementwise chain** | fuse ([12](./12_vmap.md), [18](./18_ml_kernels.md)) | smaller dtype ([17](./17_architects_view.md)) | online algorithms for any reductions ([18](./18_ml_kernels.md)) |
| **A GEMM-shaped workload** | use the vendor library / tensor cores / AMX ([16](./16_matrix_engines.md)) | align dimensions to tiles ([15](./15_gpus_simt.md)) | fuse the epilogue ([18](./18_ml_kernels.md)) |
| **A branchy, irregular workload** | find the independent instances to vectorize across ([2](./02_domains.md)) | sort/bin by branch, compact ([8](./08_control_flow.md)) | SPMD (ISPC, GPU) if branches are inherent ([14](./14_explicit_simd.md)) |
| **A serving or training system** | measure tokens per weight read ([19](./19_batching_systems.md)) | continuous batching, packing with masks | KV-cache bytes: GQA, quantization, paging ([18](./18_ml_kernels.md)) |
| **A chip or accelerator design question** | characterise the workload by the six questions ([17](./17_architects_view.md)) | roofline and utilisation models from real traces | confirm a compiler and library will target the units |
| **Results changed after vectorizing** | compare against float64 / `fsum` ([6](./06_dependences.md)) | test tails and permutations ([21](./21_debugging_playbook.md)) | document the order and tolerance |

---

## Key Takeaways

- The six questions of Chapter 1 expand into about forty concrete questions; asking them before coding prevents most wasted vectorization work.
- Legality and numerics questions decide correctness; data and memory questions decide whether it is faster.
- Control-flow and shape questions decide lane utilisation; hardware questions decide which unit should run the work.
- System questions (batching, latency, dispatch, maintenance) decide whether the speedup survives in production.
- The decision guide turns a starting point into an ordered first three moves, each pointing at the chapter that explains it.

## Check Yourself

<details class="check"><summary>A colleague proposes rewriting a C++ preprocessing step with AVX-512 intrinsics. Which three questions from the boards do you ask first?</summary>
Is it the hot spot (profile, Amdahl)? Is it memory-bound (roofline: then intrinsics will not help)? Did the compiler already vectorize it or say why not (remarks)? Then: who maintains intrinsics, and does production have AVX-512?
</details>

<details class="check"><summary>Which board's questions would have prevented the one-hot slowdown in Chapter 2?</summary>
Board 3 (data and memory): "Am I building a dense array to express a sparse idea?" and "How many times does this data cross DRAM?"
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| This book, Chapters 0–21 | — | [Table of contents](./README.md) | Every question links to its answer |
| Computer Architecture: A Quantitative Approach, 6th ed. | Hennessy & Patterson, 2017 | Morgan Kaufmann | The quantitative habit behind the questions |
| Roofline | Williams, Waterman & Patterson, CACM 2009 | doi:10.1145/1498765.1498785 | "Bytes or FLOPs first?" |

---

**Next:** [Chapter 23 — Where to Go Next](./23_where_next.md) — the map of what to learn after vectorization.
