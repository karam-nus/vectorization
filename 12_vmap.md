---
title: "Chapter 12 — Automatic Batching: vmap, jit & Friends"
---

[← Back to Table of Contents](./README.md)

# Chapter 12 — Automatic Batching: vmap, jit & Friends

> *"Write the function for one example. Let the framework find the vector."*

## Overview

Chapter 11's array style asks you to think about the batch axis all the time. `vmap`
(JAX's `jax.vmap`, PyTorch's `torch.func.vmap`) inverts that. You write a function for a single
example, and the framework rewrites the *program*, not the data, so that every operation carries an
extra batch axis. A matrix-vector product becomes a matrix-matrix product, a `where` becomes a batched
`select`, and a `grad` becomes per-example gradients. This chapter shows the rewrite on real
compiler output, says what vmap can and cannot batch, and covers its partner `jit`: fusion and code
generation that vectorize the result at the instruction level.

<div class="annot">
<pre><code>def predict(w, x):          # x: one example, shape (d,)  <span class="pin">1</span>
    h = jnp.tanh(w @ x)     # (k,d) @ (d,) -> (k,)        <span class="pin">2</span>
    return jnp.where(h &gt; 0, h, 0.01 * h).sum()            <span class="pin">3</span>

batched = jax.vmap(predict, in_axes=(None, 0))           <span class="pin">4</span>
batched(w, X)               # X: (B, d) -> (B,)           <span class="pin">5</span>

per_example_grads = jax.vmap(jax.grad(loss),
                             in_axes=(None, 0, 0))       <span class="pin">6</span></code></pre>
<ol class="annot-notes">
<li><span class="pin">1</span><b>Written for one example.</b> No batch axis anywhere in the source; shapes are the ones you would draw for a single input.</li>
<li><span class="pin">2</span><b>A GEMV becomes a GEMM.</b> Under vmap, <code>w @ x</code> with a batched <code>x</code> is rewritten into one matrix-matrix product: the weights are read once for all B examples.</li>
<li><span class="pin">3</span><b>Branches become selects.</b> <code>where</code> was already a select; a Python <code>if</code> on a traced value is not allowed, and <code>lax.cond</code> under vmap computes both sides.</li>
<li><span class="pin">4</span><b>in_axes says which inputs carry the batch.</b> <code>None</code> = shared by all examples (the weights), <code>0</code> = batched along axis 0.</li>
<li><span class="pin">5</span><b>Not a loop.</b> The batched program has the same number of operations as the original, each on bigger arrays.</li>
<li><span class="pin">6</span><b>Transforms compose.</b> vmap of grad gives one gradient per example, the building block of differentially private SGD, influence functions and per-sample clipping.</li>
</ol>
</div>

---

## What vmap actually does: batching rules

vmap traces the function and replaces every primitive with its **batched version**. Each primitive has a
batching rule that says where the batch axis goes. Here is the real JAX intermediate representation
(`jaxpr`) of `predict` before and after vmap (`tools/measurements/vmap_jaxpr.py`, JAX 0.12.1):

<div class="crt">
<div class="crt-bar">tty3 — jax.make_jaxpr(predict) vs jax.make_jaxpr(jax.vmap(predict, …)) · trimmed: dtype hints and the where() wrapper removed</div>
<div class="crt-screen"><pre>
<span class="dim">## one example</span>
{ lambda ; a:f32[8,16] b:<span class="warn">f32[16]</span>. let
    c:f32[8] = dot_general[dimension_numbers=(([1], [0]), ([], []))] a b
    d:f32[8] = tanh c
    e:bool[8] = gt d 0.0
    f:f32[8] = mul 0.01 d
    g:f32[8] = select_n e f d
    h:f32[] = reduce_sum[axes=(0,)] g
  in (h,) }

<span class="dim">## vmap over a batch of 32</span>
{ lambda ; a:f32[8,16] b:<span class="warn">f32[32,16]</span>. let
    c:<span class="warn">f32[8,32]</span> = dot_general[dimension_numbers=(([1], [<span class="warn">1</span>]), ([], []))] a b
    d:f32[8,32] = tanh c
    e:bool[8,32] = gt d 0.0
    f:f32[8,32] = mul 0.01 d
    g:f32[8,32] = select_n e f d
    h:f32[32] = reduce_sum[axes=(0,)] g
  in (h,) }
</pre></div>
</div>

Read it line by line. The program has exactly the same six operations. The `dot_general` that was a
matrix-vector product is now a matrix-matrix product contracting `w`'s axis 1 against the batch's
axis 1. The elementwise operations just gained an axis. The reduction kept its axis 0, which now
means "over k" because vmap chose to put the batch axis *last* in the intermediates (`f32[8,32]`)
and adjusted the reduction to match. Batching rules are free to move the batch axis wherever the next
operation prefers it.

Measured on one CPU core with B = 4096: a Python loop calling the jitted single-example function took
348.5 ms; one call of the jitted vmapped function took 0.077 ms, **4526× faster**. Most of that gap is
per-call dispatch overhead (Chapter 11's cost model, at about 85 µs per jitted call), and the rest is
GEMV becoming GEMM.

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>vmap with <code>in_axes=(None, 0)</code> turned a weight read per example into one weight read per
batch. What happens with <code>in_axes=(0, 0)</code>, a different weight matrix per example, as in
ensembles or per-example LoRA adapters? The GEMM becomes a batched GEMM with no weight reuse. Which of
the roofline's lines does that move you onto?</p>
</div>

---

## Batching rules for the primitives that matter

| Primitive (one example) | Under vmap | Vectorization consequence |
|---|---|---|
| elementwise `f(x)` | same op on a bigger array | free; one more axis of lanes |
| `w @ x` with shared `w` | GEMM | weights reused across the batch: the big win |
| `w @ x` with batched `w` | batched GEMM (`bmm`) | no reuse of weights; bandwidth-bound for small matrices |
| `x.sum()` | sum over the non-batch axes | reduction axis renumbered |
| `x[i]` with batched `i` | gather | one index per example |
| `x.at[i].set(v)` / `index_put_` | scatter | conflicts are per example, so none across the batch |
| `lax.cond(p, f, g)` with batched `p` | `select(p, f(x), g(x))` | **both branches computed** (Chapter 8) |
| `lax.while_loop` with batched predicate | loop until *all* examples are done, masking finished ones | trip count = slowest example |
| `lax.scan` | scan over time, batch inside each step | vectorized across the batch, sequential in time |

The `cond` and `while_loop` rows are Chapter 8 again, discovered by a compiler instead of by hand. vmap
cannot make divergence disappear; it can only make it uniform.

---

## What vmap is good for

| Use | Recipe | Why it vectorizes well |
|---|---|---|
| **Batching a single-example model** | `vmap(model, in_axes=(None, 0))` | turns GEMV into GEMM |
| **Per-example gradients** | `vmap(grad(loss), in_axes=(None, 0, 0))` | one backward pass for the whole batch instead of B passes |
| **Jacobians** | `jax.jacrev` = vmap of a VJP over basis vectors | one batched VJP instead of a loop over outputs |
| **Ensembles** | `vmap(model, in_axes=(0, None))` over stacked parameters | many small models in one kernel |
| **Many environments / simulations** | `vmap(step)` over states | Brax, Gymnax: thousands of environments per call |
| **Hyperparameter sweeps** | vmap over learning rates or seeds | many small training runs in parallel |
| **Pairwise functions** | `vmap(vmap(f, (None, 0)), (0, None))` | an all-pairs kernel from a two-argument function |

---

## What vmap cannot batch

| Construct | Why not | Workaround |
|---|---|---|
| **Data-dependent output shapes** (`x[x > 0]`, `nonzero`, `unique`) | every example would need a different shape | fixed-size versions with a fill value (`jnp.nonzero(size=…, fill_value=…)`), masks |
| **Python control flow on traced values** (`if x.sum() > 0:`) | the value is unknown while tracing | `jnp.where`, `lax.cond`, `lax.select` |
| **`.item()` / converting to Python numbers** | needs a concrete value per example | keep it in the array world |
| **In-place mutation of captured tensors** (PyTorch) | the batched version has a different shape | functional updates |
| **Ragged batches** | vmap needs a rectangular batch axis | pad and mask, or bucket by length ([Chapter 19](./19_batching_systems.md)) |
| **Randomness** | one key per example must be split explicitly | `jax.random.split(key, B)` passed with `in_axes=0`; `torch.func.vmap(randomness="different")` |

<div class="callout warn"><span class="callout-title">Trap</span>
<p>PyTorch's <code>torch.func.vmap</code> falls back to a slow per-example loop, with a performance
warning, for operators without a batching rule. If a vmapped function is mysteriously slow, look for that
warning before profiling anything else.</p>
</div>

---

## `jit`: from batched program to vector instructions

vmap produces a batched *program*. Something still has to turn it into vector *instructions*, and that
is the compiler behind `jax.jit` (XLA) and `torch.compile` (TorchInductor):

1. **Fusion.** Consecutive elementwise and reduction operations, such as `tanh`, the compare, the
   multiply, the select and the sum above, are fused into one loop. That removes the temporaries of
   Chapter 11. For memory-bound chains this is the main speedup.
2. **Library calls for the heavy ops.** `dot_general` becomes a call to oneDNN, Eigen or cuBLAS, or a
   generated Triton GEMM.
3. **Code generation for the rest.** On GPU, Inductor writes Triton kernels (Chapter 15). On CPU it writes
   C++ loops over `at::vec::Vectorized<float>`, a portable SIMD wrapper that compiles to AVX2/AVX-512/NEON,
   and adds OpenMP across threads. XLA's CPU backend emits LLVM IR that LLVM's own vectorizer finishes (Chapter 13).

You can read what was generated:

```bash
TORCH_LOGS="output_code" python train.py             # Inductor's generated Triton / C++ source
XLA_FLAGS="--xla_dump_to=/tmp/xla" python train.py   # XLA HLO before and after fusion
```

```python
print(jax.jit(f).lower(x).as_text())                 # StableHLO for one function
```

<div class="callout field"><span class="callout-title">Field note</span>
<p>When <code>torch.compile</code> makes a model slower, the generated code usually shows why in a few
minutes: a reduction it could not fuse, a graph break that sends half the model back to eager mode, or a
dynamic shape that forces recompilation. Read the output code before reaching for
<code>mode="max-autotune"</code>.</p>
</div>

---

## Choosing between vmap, manual batching and a loop

| Situation | Choose | Why |
|---|---|---|
| Model naturally takes a batch axis already | **manual batching** | nothing to gain; vmap would produce the same program |
| Per-example quantity (gradients, Jacobian rows, losses with per-example structure) | **vmap** | the manual version is error-prone index bookkeeping |
| Variable-length or data-dependent shapes per example | **pad + mask + vmap**, or **a loop over buckets** | vmap needs rectangular batches |
| Heavy per-example control flow (search, parsing) | **a loop**, or vectorize *across* examples by hand (Chapter 2) | vmap would compute every branch for every example |
| Many independent tiny models | **vmap over stacked params** | one batched GEMM instead of many launches |

---

## Key Takeaways

- vmap rewrites the program so every primitive carries a batch axis; it does not loop. Each primitive has a batching rule.
- The headline rewrite is GEMV → GEMM: shared weights are read once per batch. Measured: 4526× over a loop of jitted single-example calls, mostly dispatch overhead.
- Batched `cond` computes both branches and batched `while_loop` runs until the slowest example finishes; divergence becomes uniform, not free.
- vmap cannot batch data-dependent shapes, Python control flow on traced values, or ragged batches without padding.
- `jit` and `torch.compile` turn the batched program into vector code: fusion removes temporaries, heavy ops go to libraries, the rest becomes generated SIMD or Triton code.
- Transforms compose: vmap of grad gives per-example gradients, and vmap of vmap gives all-pairs kernels.

## Check Yourself

<details class="check"><summary>What does <code>jax.vmap(jnp.dot, in_axes=(0, None))</code> compute for a (B, n) first argument and an (n,) second?</summary>
A dot product of each of the B rows with the shared vector: a matrix-vector product of shape (B,). The batching rule turns B dot products into one GEMV.
</details>

<details class="check"><summary>Why is <code>vmap(grad(loss))</code> much faster than looping <code>grad(loss)</code> over examples, but not faster than <code>grad</code> of the mean loss?</summary>
It runs one batched backward pass (GEMMs over the batch) instead of B separate passes (GEMVs plus dispatch overhead). <code>grad</code> of the mean loss also runs one batched pass but sums the per-example gradients inside the GEMMs; per-example gradients keep B separate results, which costs memory (B × parameters) and some reuse.
</details>

<details class="check"><summary>Inside a vmapped function you wrote <code>if x.sum() &gt; 0: return f(x) else: return g(x)</code>. What happens, and what is the fix?</summary>
Tracing fails (a concrete boolean is needed from a traced value). Use <code>jnp.where(x.sum() &gt; 0, f(x), g(x))</code> or <code>lax.cond</code>; under vmap both compute f and g for every example and select per example.
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| JAX: composable transformations of Python+NumPy programs | Bradbury et al., Google, 2018– | github.com/jax-ml/jax | vmap, jit, batching rules, jaxprs |
| Compiling machine learning programs via high-level tracing | Frostig, Johnson & Leary, SysML 2018 | mlsys.org | JAX tracing and XLA compilation |
| torch.func (functorch) documentation: vmap, UX limitations | PyTorch | pytorch.org/docs/stable/func.html | Batching rules, fallback, randomness |
| PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation | Ansel et al., ASPLOS 2024 | doi:10.1145/3620665.3640366 | TorchInductor: fusion, Triton and C++/`Vectorized` codegen |
| Deep Learning with Differential Privacy | Abadi et al., CCS 2016 | arXiv:1607.00133 | Per-example gradients as a use case |
| This book's measurement | `tools/measurements/vmap_jaxpr.py`, `vmap_jaxpr.txt` | [repo](https://github.com/karam-nus/vectorization/tree/main/tools/measurements) | Jaxprs before/after vmap; loop vs vmap timing |

---

**Next:** [Chapter 13 — Compiler Auto-Vectorization](./13_auto_vectorization.md) — the layer below: how a compiler turns your loop into SIMD, and why it so often says no.
