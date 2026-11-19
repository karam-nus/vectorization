---
title: "Chapter 2 — A Field Guide to Vectorizable Workloads"
---

[← Back to Table of Contents](./README.md)

# Chapter 2 — A Field Guide to Vectorizable Workloads

> *"You will not find 'vectorization' in a profile. You will find ten kinds of loop, over and over, in every domain."*

## Overview

Vectorization opportunities are recognised by their *shape*, not by their domain. A recommender
system's embedding lookup, a graph neural network's message passing and a database's join probe are
the same species of loop, the **gather**, in different habitats. This chapter is a naturalist's
guide: ten loop species with their field marks and temperament. It then gives an atlas of domains
and the species that dominate each, and ends with the trick that rescues the unvectorizable ones:
**vectorize across independent instances instead of within one.**

<div class="field-guide">
<div class="fg-head"><b>A Field Guide to the Loops</b><small>with notes on habitat, field marks and temperament toward SIMD</small></div>
<div class="fg-grid">
<div class="species"><div class="sp-name">The Map</div><div class="sp-latin">Mappus elementaria</div><div class="sp-sketch">for i: y[i] = f(x[i])</div><div class="sp-field"><b>Field marks:</b> each output reads only its own inputs.</div><div class="sp-field"><b>Habitat:</b> activations, scaling, (de)quantization, pixel ops, augmentation.</div><div class="sp-field"><b>Temperament:</b> <span class="temper docile">docile</span> limited only by memory bandwidth.</div></div>
<div class="species"><div class="sp-name">The Reduction</div><div class="sp-latin">Summa reordinata</div><div class="sp-sketch">for i: s = s + x[i]</div><div class="sp-field"><b>Field marks:</b> one accumulator, an associative operator.</div><div class="sp-field"><b>Habitat:</b> losses, norms, softmax denominators, dot products, statistics.</div><div class="sp-field"><b>Temperament:</b> <span class="temper wary">wary</span> needs permission to reorder floats.</div></div>
<div class="species"><div class="sp-name">The Stencil</div><div class="sp-latin">Vicinia convoluta</div><div class="sp-sketch">y[i] = w0*x[i-1] + w1*x[i] + w2*x[i+1]</div><div class="sp-field"><b>Field marks:</b> fixed neighbourhood, reads overlap between outputs.</div><div class="sp-field"><b>Habitat:</b> convolution, image filters, PDE solvers, FIR audio filters.</div><div class="sp-field"><b>Temperament:</b> <span class="temper docile">docile</span> if outputs never feed inputs.</div></div>
<div class="species"><div class="sp-name">The Scan</div><div class="sp-latin">Prefixus sequentialis</div><div class="sp-sketch">for i: y[i] = y[i-1] + x[i]</div><div class="sp-field"><b>Field marks:</b> each output needs the previous output.</div><div class="sp-field"><b>Habitat:</b> cumsum, RNNs, state-space models, IIR filters, top-p sampling.</div><div class="sp-field"><b>Temperament:</b> <span class="temper fierce">fierce</span> needs an associative reformulation (Ch 10).</div></div>
<div class="species"><div class="sp-name">The Gather</div><div class="sp-latin">Indexus dispersus</div><div class="sp-sketch">for i: y[i] = table[idx[i]]</div><div class="sp-field"><b>Field marks:</b> indirect read through an index array.</div><div class="sp-field"><b>Habitat:</b> embedding lookups, paged KV cache, sparse features, MoE routing.</div><div class="sp-field"><b>Temperament:</b> <span class="temper wary">wary</span> legal, but costs a cache line per element.</div></div>
<div class="species"><div class="sp-name">The Scatter-Add</div><div class="sp-latin">Conflictus additivus</div><div class="sp-sketch">for i: h[idx[i]] += w[i]</div><div class="sp-field"><b>Field marks:</b> indirect write; two lanes may hit one slot.</div><div class="sp-field"><b>Habitat:</b> histograms, index_add, GNN aggregation, embedding gradients.</div><div class="sp-field"><b>Temperament:</b> <span class="temper fierce">fierce</span> lane conflicts lose updates silently.</div></div>
<div class="species"><div class="sp-name">The Filter</div><div class="sp-latin">Filtrum compressum</div><div class="sp-sketch">for i: if p(x[i]): out[k++] = x[i]</div><div class="sp-field"><b>Field marks:</b> output position depends on earlier decisions.</div><div class="sp-field"><b>Habitat:</b> NMS candidates, WHERE clauses, parsers, top-k prefilters.</div><div class="sp-field"><b>Temperament:</b> <span class="temper wary">wary</span> needs compress or a prefix sum.</div></div>
<div class="species"><div class="sp-name">The Contraction</div><div class="sp-latin">Matrix multiplicata</div><div class="sp-sketch">C[m,n] += A[m,k] * B[k,n]</div><div class="sp-field"><b>Field marks:</b> three loops, every operand reused many times.</div><div class="sp-field"><b>Habitat:</b> every linear layer, attention, convolution via im2col.</div><div class="sp-field"><b>Temperament:</b> <span class="temper docile">docile</span> and prized: the reason accelerators exist.</div></div>
<div class="species"><div class="sp-name">The Sort</div><div class="sp-latin">Ordo bitonicus</div><div class="sp-sketch">compare-exchange(a[i], a[j])</div><div class="sp-field"><b>Field marks:</b> data-dependent order.</div><div class="sp-field"><b>Habitat:</b> top-k, sampling, database operators, deduplication.</div><div class="sp-field"><b>Temperament:</b> <span class="temper wary">wary</span> sorting networks vectorize; quicksort's branches do not.</div></div>
<div class="species"><div class="sp-name">The Pointer Chase</div><div class="sp-latin">Catena indirecta</div><div class="sp-sketch">while n: n = n.next</div><div class="sp-field"><b>Field marks:</b> the next address comes from the current load.</div><div class="sp-field"><b>Habitat:</b> trees, tries, linked lists, decision-tree inference.</div><div class="sp-field"><b>Temperament:</b> <span class="temper fierce">fierce</span> within one chain; tame it across many chains.</div></div>
</div>
</div>

---

## Reading the temperaments

**Docile** species vectorize under any compiler and any framework. Their only enemy is memory
bandwidth (Chapter 9). **Wary** species are legal to vectorize but need something extra: permission to
reassociate (reductions), a gather unit and tolerance for cache misses (gathers), a compress
instruction (filters), or a sorting network (sorts). **Fierce** species are not vectorizable as written. The
scan has a true loop-carried dependence, the scatter-add has write conflicts, and the pointer chase
has an address dependence. Each has a known reformulation; none of them is automatic.

The companion code's loop-versus-NumPy measurements show the temperaments directly
(`numpy_idioms` in `results.json`; Xeon, 1 core, NumPy 2.5):

| Idiom | Species | Size | Python loop | NumPy | Speedup |
|---|---|---|:---:|:---:|:---:|
| **moving average** via `cumsum` | scan → reformulated | 200 000 samples, w = 50 | 627 ms | 0.95 ms | 661× |
| **histogram** via `bincount` | scatter-add, done right | 20 000 indices | 3.8 ms | 0.022 ms | 171× |
| **layernorm** | map + reduction | 256 × 512 | 62 ms | 0.56 ms | 112× |
| **softmax** | map + reduction | 256 × 512 | 44 ms | 0.54 ms | 81× |
| **IoU matrix** | broadcast map | 200 × 200 boxes | 56 ms | 1.2 ms | 47× |
| **leaky ReLU** | map with a branch | 200 000 values | 23 ms | 1.1 ms | 21× |
| **one-hot** via broadcast compare | map over a huge output | 20 000 × 1000 | 25 ms | 47 ms | **0.53×** |

The last row is a warning, not a typo: the "vectorized" one-hot writes 20 million float64s (160 MB)
and allocates a 20-million-element boolean temporary on the way. The loop writes 20 000 ones into a
zeroed array of the same size, so both are dominated by memory traffic, and the vectorized version
moves more of it. [Chapter 11](./11_array_programming.md) explains why, and why the right fix is not
to build the one-hot at all.

---

## The habitat atlas

| Domain | Dominant species | What vectorization buys there | Representative systems |
|---|---|---|---|
| **Deep-learning training & inference** | contraction, map, reduction | tensor cores for GEMM; fused maps and reductions for everything else | cuBLAS, CUTLASS, oneDNN, XLA, Inductor, Triton |
| **LLM serving** | contraction, gather, scan, sort | batched decode turns GEMV into GEMM; paged KV is a gather; sampling is sort + scan | vLLM, SGLang, TensorRT-LLM, llama.cpp |
| **Recommender systems** | gather, scatter-add, reduction | embedding bags are gathers plus pooling; their gradients are scatter-adds | FBGEMM, TorchRec |
| **Graph neural networks** | gather, scatter-add | message passing = gather from neighbours, scatter-add into nodes | PyG, DGL (segment reductions) |
| **Computer vision pre-processing** | map, stencil, gather | resize and colour conversion are maps and gathers over pixels | OpenCV (universal intrinsics), DALI, torchvision v2 |
| **Scientific computing / HPC** | stencil, contraction, scan | the original home of vector machines (Cray-1, 1976) | BLAS, LAPACK, FFTW, PETSc |
| **Signal & audio** | stencil, scan | FIR filters vectorize; IIR filters are scans | FFTW, IPP, CMSIS-DSP |
| **Databases & analytics** | filter, reduction, gather | column-at-a-time execution processes a vector of rows per operator call | MonetDB/X100, DuckDB, Velox, ClickHouse |
| **Text & parsing** | filter, map | classify 32–64 bytes per instruction with shuffles and bit masks | simdjson, simdutf, Hyperscan |
| **Genomics** | stencil, scan (dynamic programming) | striped SIMD Smith–Waterman fills alignment matrices 8–16 cells at a time | Farrar's SSW, minimap2, parasail |
| **Cryptography & compression** | map, filter, gather | AES rounds, CRC and literal copies in single instructions | AES-NI, zstd, ISA-L |
| **Reinforcement learning** | map (over environments) | thousands of simulated environments stepped as one batch | EnvPool, Brax, Gymnax |
| **Quantized inference** | contraction, map | int8/int4 dot products (VNNI, `sdot`, `dp4a`), nibble unpacking by shuffle | llama.cpp, FBGEMM, TFLite XNNPACK |
| **Chip design & verification** | map over test vectors | bit-parallel logic simulation runs 64 test patterns per machine word (SWAR) | gate-level simulators, fault simulators |

<div class="callout field"><span class="callout-title">Field note</span>
<p>Notice how often the "vectorized" win in a domain came from <em>changing what the vector runs over</em>.
Databases vectorize over rows of a column, not over the operations of one row. RL vectorizes over
environments, not over the physics of one. Logic simulators vectorize over test patterns, not over
gates. When the work inside one instance is hostile, look for many instances.</p>
</div>

---

## Taming the fierce species: vectorize across, not within

A single pointer chase cannot be vectorized: each load's address comes from the previous load.
But 16 *independent* pointer chases can be, one per lane, with a gather per step. The same trick
turns many "unvectorizable" workloads into very regular ones:

| Fierce inside one instance | Independent axis to vectorize across | Known example |
|---|---|---|
| Traversing one decision tree | many input rows (or many trees) at once | QuickScorer / RapidScorer for tree ensembles |
| One RNN or SSM time step after another | the batch, and the hidden dimension within a step | cuDNN RNN kernels; Mamba's parallel scan |
| Beam search on one sequence | the beams and the batch | every LLM serving stack |
| Tokenizing one string | many strings, or many byte positions with a parallel lexer | simdjson stage 1 (structural indexing) |
| One environment step with branches | thousands of environments | EnvPool, Brax |
| One Monte Carlo path | many paths | option pricing, particle filters |

<div class="regs">
<div class="regs-title">8 independent tree traversals (one per lane), one gather per level</div>
<div class="reg-row"><span class="reg-name">node (depth 0)</span><span class="reg-lanes"><span class="ln">0</span><span class="ln">0</span><span class="ln">0</span><span class="ln">0</span><span class="ln">0</span><span class="ln">0</span><span class="ln">0</span><span class="ln">0</span></span><span class="reg-note">every row starts at the root</span></div>
<div class="reg-row"><span class="reg-name">x[feat] &lt; thr</span><span class="reg-lanes"><span class="ln mask1">1</span><span class="ln mask0">0</span><span class="ln mask1">1</span><span class="ln mask1">1</span><span class="ln mask0">0</span><span class="ln mask1">1</span><span class="ln mask0">0</span><span class="ln mask1">1</span></span><span class="reg-note">one compare, 8 rows</span></div>
<div class="reg-row"><span class="reg-name">node (depth 1)</span><span class="reg-lanes"><span class="ln l1">1</span><span class="ln l2">2</span><span class="ln l1">1</span><span class="ln l1">1</span><span class="ln l2">2</span><span class="ln l1">1</span><span class="ln l2">2</span><span class="ln l1">1</span></span><span class="reg-note">node = 2·node + 1 + (1 − mask)</span></div>
<div class="reg-row"><span class="reg-name">node (depth 2)</span><span class="reg-lanes"><span class="ln l3">3</span><span class="ln l5">6</span><span class="ln l4">4</span><span class="ln l3">3</span><span class="ln l6">5</span><span class="ln l4">4</span><span class="ln l5">6</span><span class="ln l3">3</span></span><span class="reg-note">gather thresholds for 8 different nodes</span></div>
</div>
<div class="genre-caption">Each lane walks its own path; the "pointer" becomes an index computed arithmetically, and the loads become a gather. Complete trees of fixed depth make the loop trip count uniform: no lane exits early.</div>

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>Pick the most expensive loop in your current project and name its species. If it is fierce, what is the
largest set of <em>independent instances</em> of it that exist at the same time? Requests in a server,
samples in a batch, beams, environments, test cases? That set is your vector.</p>
</div>

---

## Field marks: is this loop a candidate?

A loop is worth a closer look when most of these are true:

1. **The trip count is known when the loop starts**, even if not at compile time. Loops that exit on a
   data-dependent condition (`while x != 0`) are harder ([Chapter 8](./08_control_flow.md)).
2. **The body applies the same operations to each element.** Function pointers, virtual calls and
   type switches inside the loop body are not that.
3. **No iteration writes what a later iteration reads**, or the only such write is an associative
   accumulation ([Chapter 6](./06_dependences.md)).
4. **Access is contiguous or regularly strided** along the loop axis ([Chapter 7](./07_memory_layout.md)).
5. **Branches are short and can be turned into selects** ([Chapter 8](./08_control_flow.md)).
6. **n is large enough** that the vector body dominates setup and the tail: 
   at least several times the lane count.
7. **The loop matters**: a profiler, not intuition, says it holds a real fraction of runtime ([Chapter 9](./09_profitability.md)).

---

## Key Takeaways

- Vectorization opportunities are identified by loop shape; ten species cover almost everything you will meet.
- Maps, stencils and contractions are docile; reductions, gathers, filters and sorts are wary; scans, scatter-adds and pointer chases are fierce as written.
- Vectorized NumPy beat Python loops by 21–661× on six idioms here, and lost (0.53×) on one-hot, where the output size, not the arithmetic, sets the cost.
- Most domains got their big vectorization win by choosing a different axis — rows of a column, environments, test patterns, requests — not by vectorizing inside one instance.
- A fierce loop becomes docile when you run many independent copies of it in lockstep, one per lane.

## Check Yourself

<details class="check"><summary>Which species is the backward pass of an embedding lookup, and why is it harder than the forward pass?</summary>
A scatter-add. The forward pass gathers rows (reads only, always legal). The backward pass adds gradients into the rows that were read; when two examples in the batch use the same token, two lanes write the same row, so a naive vectorized scatter loses updates. It needs conflict detection, sorting by index, or atomics.
</details>

<details class="check"><summary>Why is a FIR filter docile but an IIR filter fierce?</summary>
A FIR filter's output depends only on inputs (a stencil). An IIR filter's output depends on previous outputs (a recurrence/scan), creating a loop-carried dependence. It can be parallelised with a linear-recurrence scan, at extra arithmetic cost.
</details>

<details class="check"><summary>A database evaluates <code>WHERE price &gt; 100</code> over 10 million rows. Which species, and what instruction family helps?</summary>
A filter. Comparisons produce a mask per vector; AVX-512's <code>vcompressps</code> (or a prefix sum of the mask in other ISAs) packs the surviving rows contiguously.
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| MonetDB/X100: Hyper-Pipelining Query Execution | Boncz, Zukowski & Nes, CIDR 2005 | cidrdb.org | Vectorized (column-at-a-time) query execution |
| Parsing Gigabytes of JSON per Second | Langdale & Lemire, VLDB Journal 2019 | arXiv:1902.08318 | SIMD parsing via structural bitmasks |
| Striped Smith–Waterman speeds database searches six times over other SIMD implementations | Farrar, Bioinformatics 2007 | doi:10.1093/bioinformatics/btl582 | SIMD dynamic programming |
| QuickScorer: a fast algorithm to rank documents with additive ensembles of regression trees | Lucchese et al., SIGIR 2015 | doi:10.1145/2766462.2767733 | Vectorizing tree ensembles across documents |
| EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine | Weng et al., NeurIPS 2022 | arXiv:2206.10558 | Batched environment stepping |
| This book's measurements | `tools/measurements/results.json` | [repo](https://github.com/karam-nus/vectorization/tree/main/tools/measurements) | `numpy_idioms` |

---

**Next:** [Chapter 3 — Lanes, Registers & the SIMD Contract](./03_lanes_and_registers.md) — what one vector instruction actually promises, lane by lane.
