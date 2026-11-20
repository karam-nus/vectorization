---
title: "Chapter 11 — Array Programming: NumPy, PyTorch & einsum"
---

[← Back to Table of Contents](./README.md)

# Chapter 11 — Array Programming: NumPy, PyTorch & einsum

> *"An array library is a vector machine whose instruction is a function call. Learn its instruction set and its overhead, and you can predict every speedup."*

## Overview

NumPy, PyTorch, JAX and their relatives expose a vector instruction set at the level of whole arrays:
one call, one decision, applied to millions of elements in compiled code. This chapter treats them
exactly like hardware. It covers their cost model (a fixed overhead per call plus a cost per element,
both measured), their broadcasting rule, drawn as a spreadsheet, and `einsum` as the most expressive
vector instruction they have. It also covers the three ways array code goes wrong: temporaries that
eat bandwidth, Python loops hidden behind a friendly name, and outputs too large to be worth building.

<div class="sheets">
<table class="sheet"><caption>x: shape (4, 1) — a column</caption><tr><th></th><th>A</th><th>B</th><th>C</th></tr><tr><th>1</th><td class="s">10</td><td class="g">10</td><td class="g">10</td></tr><tr><th>2</th><td class="s">20</td><td class="g">20</td><td class="g">20</td></tr><tr><th>3</th><td class="s">30</td><td class="g">30</td><td class="g">30</td></tr><tr><th>4</th><td class="s">40</td><td class="g">40</td><td class="g">40</td></tr></table>
<span class="op">+</span>
<table class="sheet"><caption>y: shape (3,) — a row</caption><tr><th></th><th>A</th><th>B</th><th>C</th></tr><tr><th>1</th><td class="s">1</td><td class="s">2</td><td class="s">3</td></tr><tr><th>2</th><td class="g">1</td><td class="g">2</td><td class="g">3</td></tr><tr><th>3</th><td class="g">1</td><td class="g">2</td><td class="g">3</td></tr><tr><th>4</th><td class="g">1</td><td class="g">2</td><td class="g">3</td></tr></table>
<span class="op">=</span>
<table class="sheet"><caption>x + y: shape (4, 3)</caption><tr><th></th><th>A</th><th>B</th><th>C</th></tr><tr><th>1</th><td>11</td><td>12</td><td>13</td></tr><tr><th>2</th><td>21</td><td>22</td><td>23</td></tr><tr><th>3</th><td>31</td><td>32</td><td>33</td></tr><tr><th>4</th><td>41</td><td>42</td><td>43</td></tr></table>
</div>
<div class="genre-caption">Broadcasting is "fill right" and "fill down" in a spreadsheet, except nothing is filled: the ghosted cells are a stride-0 view of the green ones. Only the result is materialised.</div>

---

## The cost model: overhead per call, cost per element

Time for one array operation on $n$ elements is, to a good approximation,

$$t(n) \approx t_{\text{call}} + n \cdot t_{\text{elem}}$$

Measured for `np.add(a, b, out=c)` on float32, one core, NumPy 2.5 (`numpy_call_overhead`):

| n | Time per call | ns per element | Regime |
|---:|:---:|:---:|---|
| 1 | 0.233 µs | 233 | pure overhead |
| 100 | 0.248 µs | 2.48 | overhead-dominated |
| 1 000 | 0.549 µs | 0.55 | crossover |
| 10 000 | 3.19 µs | 0.32 | vectorized, in L1/L2 |
| 100 000 | 28.4 µs | 0.28 | vectorized, in L2 |
| 1 000 000 | 490 µs | 0.49 | L3 (three 4 MB arrays) |
| 10 000 000 | 7.79 ms | 0.78 | DRAM-bound |

The call overhead $t_{\text{call}} \approx 0.23$ µs is argument parsing, dtype resolution, choosing an inner
loop, and allocating or checking the output. The element cost bottoms out near 0.3 ns, then rises
as the arrays fall out of cache (Chapter 7). The crossover size is

$$n^\ast = \frac{t_{\text{call}}}{t_{\text{elem}}} \approx \frac{0.23\ \mu\text{s}}{0.3\ \text{ns}} \approx 800$$

Below a few hundred elements per call you are mostly paying for the call. Above a few thousand you get
the vectorized inner loop. Large calls are not automatically fast, though. The pairwise-distance
"row loop" rung makes calls of 65 536 elements, far above $n^\ast$, yet costs 1.18 ns per element,
four times `np.add`'s rate. Each row does three full passes (subtract, square, sum), each with its own
temporary, and the sum reduces along a short axis of 64. Overhead per call and passes per element are
separate costs; Chapter 9's roofline counts the second.

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>A GPU kernel launch costs a few microseconds, roughly ten times NumPy's call overhead, and a GPU's
per-element cost is far lower. So the crossover $n^\ast$ on a GPU is in the tens or hundreds of thousands
of elements. How many of the operations in your model's forward pass are smaller than that? Those are
the ones CUDA Graphs and kernel fusion (Chapter 12) exist to rescue.</p>
</div>

---

## `np.vectorize` is not vectorization

The most misleading name in the ecosystem. `np.vectorize(f)` gives a Python function NumPy's
broadcasting *interface*, but it still calls `f` once per element from Python. Measured on 100 000
values of leaky ReLU (`np_vectorize`):

| Implementation | Time | vs list comprehension |
|---|:---:|:---:|
| **list comprehension** `[f(v) for v in x]` | 12.8 ms | 1.0× |
| **`np.vectorize(f)(x)`** | 13.1 ms | 0.98× |
| **`np.where(x > 0, x, 0.01 * x)`** | 0.51 ms | **25×** |

The NumPy documentation says so plainly: `vectorize` is provided "primarily for convenience, not for
performance. The implementation is essentially a for loop." The same is true of `pandas.DataFrame.apply`
with a Python function, and of `map` over a tensor's elements.

<div class="callout myth"><span class="callout-title">Myth</span>
<p>"I vectorized it with <code>np.vectorize</code> / <code>apply</code> / <code>map</code>." These remove the
loop from your source, not from the execution. Vectorization in the NumPy sense means the per-element
work runs inside one compiled call, which requires expressing it with array primitives (<code>where</code>,
arithmetic, ufuncs, reductions, indexing).</p>
</div>

---

## Broadcasting: the rule and its cost

**The rule.** Align shapes on the right. For each axis, the sizes must be equal, or one of them must
be 1 (a missing axis counts as 1). The result takes the larger size; size-1 axes are *stretched*.

| A | B | Result | Stretched |
|---|---|---|---|
| (4, 1) | (3,) | (4, 3) | A along axis −1, B along axis −2 |
| (8, 1, 6, 1) | (7, 1, 5) | (8, 7, 6, 5) | both, on different axes |
| (B, T, D) | (D,) | (B, T, D) | the bias over B and T |
| (n, 1, d) | (1, m, d) | (n, m, d) | each operand along the other's axis |
| (3, 4) | (3,) | **error** | axis −1: 4 vs 3 |

The last row is the classic error, `operands could not be broadcast together with shapes (3,4) (3,)`.
The fix is to say which axis you mean: `y[:, None]` gives shape (3, 1).

**How stretching is free.** A stretched axis is a view with **stride 0**: every step along it lands on
the same memory. You can see it:

```python
import numpy as np
x = np.arange(4.0)[:, None]                 # (4, 1)
print(np.broadcast_to(x, (4, 3)).strides)   # (8, 0): moving along axis 1 does not move in memory
```

Inside the ufunc's inner loop, a stride-0 operand is a broadcast register (Chapter 3): one load, every
lane.

<div class="lab" data-lab="broadcast"></div>

**Where broadcasting costs memory.** The *output* of a broadcast operation is materialised in full. In
the pairwise-distance problem, `X[:, None, :] - Y[None, :, :]` for $n = m = 1024$, $d = 64$ creates a
(1024, 1024, 64) float32 temporary of 256 MB before the sum reduces it. The broadcast rung ran at
1.67 ns per pair-dimension, *slower* than the row loop's 1.18, because it streams 256 MB through
DRAM twice. The GEMM reformulation (0.086 ns, Chapter 10's Rf card) never builds it.

<div class="callout warn"><span class="callout-title">Trap</span>
<p>Before broadcasting two operands of shapes (n, 1, d) and (1, m, d), multiply n·m·d by the element size.
If the answer is bigger than your L2 cache, look for a formulation that reduces over d inside a matmul,
or process blocks of n at a time.</p>
</div>

---

## `einsum`: the shape strip as code

`einsum` names every axis with a letter. Letters that appear in the output are kept; letters that appear
only in the inputs are summed over. It is the shape strip of Chapter 4 written as a string, and it is the
most general vector instruction an array library offers:

| Operation | einsum | Equivalent |
|---|---|---|
| **Dot product** | `"i,i->"` | `x @ y` |
| **Outer product** | `"i,j->ij"` | `x[:, None] * y[None, :]` |
| **Matrix multiply** | `"ik,kj->ij"` | `A @ B` |
| **Batched matmul** | `"bik,bkj->bij"` | `torch.bmm` |
| **Attention scores** | `"bhtd,bhsd->bhts"` | `Q @ K.transpose(-1, -2)` |
| **Row-wise dot** | `"ij,ij->i"` | `(X * Y).sum(1)` without the temporary |
| **Bilinear form** | `"i,ij,j->"` | `x @ A @ y` |

Two practical facts. First, `np.einsum(..., optimize=True)` and `torch.einsum` find a contraction order
and dispatch to BLAS matmuls where they can, so a well-chosen einsum *is* a tensor-core or BLAS call.
Second, `"ij,ij->i"` computes a row-wise dot product without materialising `X * Y`. The companion
`pdist_broadcast` uses exactly that to save one of its two passes.

---

## Temporaries: the hidden memory traffic

Every intermediate in an array expression is a full array written to memory and read back:

```python
y = a * x + b        # tmp = a*x (write n), y = tmp + b (read n, write n): 3 passes of n floats
```

For memory-bound elementwise chains (Chapter 9), passes *are* the runtime. The remedies, in order of effort:

| Remedy | How | Effect |
|---|---|---|
| **In-place / `out=`** | `np.multiply(x, a, out=y); y += b` | no new allocation; still two passes |
| **One fused primitive** | `np.clip`, `np.where`, `torch.addcmul`, `torch.nn.functional.*` | one pass inside one kernel |
| **Expression compiler** | `numexpr.evaluate("a*x+b")` | blocks the expression into cache-sized chunks |
| **Graph compiler** | `torch.compile`, `jax.jit` | fuses whole chains into one generated loop ([Chapter 12](./12_vmap.md)) |

---

## The idiom phrasebook

Each row turns a loop pattern from the field guide (Chapter 2) into array primitives:

| Loop does… | Species | Array idiom | Watch out |
|---|---|---|---|
| `y[i] = f(x[i])` with a branch | map | `np.where(c, a, b)`, `np.clip` | both sides computed |
| running total | scan | `np.cumsum`, `ufunc.accumulate` | float error grows with n |
| sliding window sum / mean | stencil via scan | difference of `cumsum`s (measured 661×) | cancellation for long series |
| general sliding window | stencil | `np.lib.stride_tricks.sliding_window_view` | a view, not a copy, until you reduce |
| count / sum into bins | scatter-add | `np.bincount`, `np.add.at`, `torch.index_add_`, `scatter_add_` | **`h[idx] += 1` drops duplicates** |
| pick per-row element | gather | `np.take_along_axis`, `torch.gather` | index dtype must be integer |
| filter | filter | boolean mask `x[x > t]` | output size unknown in advance: a sync on GPU |
| find positions in a sorted list | search | `np.searchsorted` | inputs must be sorted |
| group by key | sort + segment | `np.unique(..., return_inverse=True)` + `bincount` | sorting is O(n log n) |
| top-k | sort | `np.argpartition`, `torch.topk` | partition is not sorted |
| first index where | search | `np.argmax(mask)` | returns 0 when nothing matches: check `mask.any()` |

The measured speedups for seven of these are in [Chapter 2](./02_domains.md#reading-the-temperaments).

---

## When the output is the problem: one-hot

One-hot encoding is the clearest case of vectorized code that should not exist. The broadcast-compare
version measured 0.53× the speed of a Python loop, and the scatter version only 1.13×. Both are bound by
writing a 20 000 × 1000 float64 array, 160 MB of mostly zeros. The array-level fix is to **not build the
one-hot at all**:

| You wanted | Do instead | Why it is faster |
|---|---|---|
| `one_hot(y) @ W` | `W[y]` (an embedding lookup / gather) | reads k rows instead of multiplying by n × k zeros |
| cross-entropy with one-hot targets | `F.cross_entropy(logits, y)` with integer `y` | gathers one logit per row |
| class counts | `np.bincount(y)` | n increments instead of n × k writes |

<div class="callout field"><span class="callout-title">Field note</span>
<p>When a vectorized rewrite is slower than the loop, compare the size of the arrays it allocates with
the size of the data it reads. If they differ by a factor of the number of classes, positions or
pairs, the vectorization replaced a sparse computation with a dense one. Find the gather, scatter or matmul
that keeps it sparse.</p>
</div>

---

## PyTorch specifics

Everything above transfers, with four additions:

1. **`expand` versus `repeat`.** `x.expand(4, 3)` is a stride-0 view, as in broadcasting; `x.repeat(4, 3)`
   copies. Prefer `expand` unless you must write into the result.
2. **Contiguity is visible.** `x.is_contiguous()`, `x.stride()`, `x.contiguous()`. Some kernels silently
   copy non-contiguous inputs; `transpose` followed by a reduction along the wrong axis is a common
   hidden cost.
3. **Device synchronisation.** Shape-dependent operations such as boolean-mask indexing, `nonzero` and
   `.item()` force the CPU to wait for the GPU. In a loop they serialise everything. Vectorize them away
   with fixed-shape alternatives like `torch.where` and padding.
4. **Default dtypes.** NumPy defaults to float64, PyTorch to float32. A NumPy pipeline that silently
   promotes to float64 moves twice the bytes and gets half the SIMD lanes (Chapter 3).

---

## Key Takeaways

- An array call costs a fixed overhead (0.23 µs here) plus a per-element cost (0.3 ns in cache); below roughly 800 elements per call you mostly pay overhead.
- `np.vectorize` is a Python loop (13.1 ms versus 12.8 ms for a list comprehension); `np.where` was 25× faster.
- Broadcasting aligns shapes on the right and stretches size-1 axes as stride-0 views; only the output is materialised, and that output can be enormous.
- `einsum` is the shape strip as code and dispatches to BLAS where possible; it can also avoid temporaries such as `X * Y` in a row-wise dot.
- Every temporary in an expression is an extra pass over memory; fusing passes matters more than lanes for elementwise code.
- When vectorized code is slower than the loop, it has usually replaced a sparse computation (gather, scatter, count) with a dense one.

## Check Yourself

<details class="check"><summary>What is the shape of <code>a[:, None, :] - b[None, :, :]</code> for a of shape (100, 3) and b of shape (50, 3), and how much memory does it allocate in float64?</summary>
(100, 50, 3), 15 000 elements × 8 bytes = 120 KB. Fine here; at (10 000, 10 000, 3) it would be 2.4 GB.
</details>

<details class="check"><summary>Why can a NumPy loop over 10 000 rows of length 32 be slower per element than a loop over 100 rows of length 3200?</summary>
Each call pays ~0.23 µs of overhead. With 32 elements per call that overhead is ~7 ns per element, far above the 0.3 ns vectorized cost; with 3200 elements it is under 0.1 ns per element.
</details>

<details class="check"><summary>Rewrite <code>for i in range(n): out[i] = x[i, idx[i]]</code> without a loop.</summary>
<code>out = x[np.arange(n), idx]</code> or <code>np.take_along_axis(x, idx[:, None], axis=1)[:, 0]</code>; in PyTorch <code>x.gather(1, idx[:, None]).squeeze(1)</code>.
</details>

<details class="check"><summary>Is <code>torch.zeros(n).index_add_(0, idx, w)</code> safe when <code>idx</code> has duplicates?</summary>
Yes: <code>index_add_</code> accumulates every entry (using atomics on GPU, so floating-point results may vary run to run). It is <code>h[idx] += w</code> that loses duplicates.
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| The NumPy array: a structure for efficient numerical computation | van der Walt, Colbert & Varoquaux, CiSE 2011 | doi:10.1109/MCSE.2011.37 | Strides, views, broadcasting as stride 0 |
| Array programming with NumPy | Harris et al., Nature 2020 | doi:10.1038/s41586-020-2649-2 | Vectorization and broadcasting semantics |
| NumPy documentation: `numpy.vectorize`, broadcasting, `einsum` | NumPy | numpy.org/doc | "essentially a for loop"; broadcasting rules; einsum optimisation |
| opt_einsum — A Python package for optimizing contraction order for einsum-like expressions | Smith & Gray, JOSS 2018 | doi:10.21105/joss.00753 | Contraction-path optimisation |
| PyTorch documentation: `Tensor.expand`, `index_add_`, `torch.gather` | PyTorch | pytorch.org/docs | Stride-0 views, accumulation semantics |
| This book's measurements | `tools/measurements/results.json` | [repo](https://github.com/karam-nus/vectorization/tree/main/tools/measurements) | `numpy_call_overhead`, `np_vectorize`, `numpy_idioms`, `pdist_python` |

---

**Next:** [Chapter 12 — Automatic Batching: vmap, jit & Friends](./12_vmap.md) — write the function for one example; let the framework add the vector axis.
