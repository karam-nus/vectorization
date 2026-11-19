---
title: "Chapter 19 — Batching: Vectorization at System Scale"
---

[← Back to Table of Contents](./README.md)

# Chapter 19 — Batching: Vectorization at System Scale

> *"The batch dimension is a vector register the size of a datacenter. The weight read is the instruction overhead it amortises."*

## Overview

Everything in this book has a system-level twin. A batch of requests is a vector of lanes. A shared
weight read is the instruction fetch that the lanes share. Padding is the masked tail, and requests
of different lengths are divergence. This chapter derives why batch size *is* arithmetic intensity
for LLM inference, then compares the batching strategies of training and serving systems by the
lanes they waste, from static padding to continuous batching and sequence packing. It ends with batching
outside LLMs and with speculative decoding, which batches over the future.

<div class="pack">
<div class="pack-title">One batch of 6 sequences, 16 token slots each — padded vs packed</div>
<div class="pack-row"><span class="pack-name">seq A (11)</span><span class="pack-bar" style="--n:16"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i></span><span class="pack-count">69 %</span></div>
<div class="pack-row"><span class="pack-name">seq B (4)</span><span class="pack-bar" style="--n:16;--c:#a78bfa"><i></i><i></i><i></i><i></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i></span><span class="pack-count">25 %</span></div>
<div class="pack-row"><span class="pack-name">seq C (16)</span><span class="pack-bar" style="--n:16;--c:#34d399"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></span><span class="pack-count">100 %</span></div>
<div class="pack-row"><span class="pack-name">seq D (7)</span><span class="pack-bar" style="--n:16;--c:#fbbf24"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i></span><span class="pack-count">44 %</span></div>
<div class="pack-row"><span class="pack-name">seq E (9)</span><span class="pack-bar" style="--n:16;--c:#f472b6"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i></span><span class="pack-count">56 %</span></div>
<div class="pack-row"><span class="pack-name">seq F (5)</span><span class="pack-bar" style="--n:16;--c:#60a5fa"><i></i><i></i><i></i><i></i><i></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i></span><span class="pack-count">31 %</span></div>
<div class="pack-row"><span class="pack-name">packed row 1</span><span class="pack-bar" style="--n:16"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i style="--c:#a78bfa"></i><i style="--c:#a78bfa"></i><i style="--c:#a78bfa"></i><i style="--c:#a78bfa"></i><i class="e"></i></span><span class="pack-count">94 %</span></div>
<div class="pack-row"><span class="pack-name">packed row 2</span><span class="pack-bar" style="--n:16;--c:#34d399"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></span><span class="pack-count">100 %</span></div>
<div class="pack-row"><span class="pack-name">packed row 3</span><span class="pack-bar" style="--n:16;--c:#fbbf24"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i style="--c:#f472b6"></i><i style="--c:#f472b6"></i><i style="--c:#f472b6"></i><i style="--c:#f472b6"></i><i style="--c:#f472b6"></i><i style="--c:#f472b6"></i><i style="--c:#f472b6"></i><i style="--c:#f472b6"></i><i style="--c:#f472b6"></i></span><span class="pack-count">100 %</span></div>
<div class="pack-row"><span class="pack-name">packed row 4</span><span class="pack-bar" style="--n:16;--c:#60a5fa"><i></i><i></i><i></i><i></i><i></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i><i class="e"></i></span><span class="pack-count">31 %</span></div>
</div>
<div class="genre-caption">Padded: 52 real tokens in 96 slots (54 %). Packed into rows of 16 (sequences never split; an attention mask keeps them apart): 52 in 64 (81 %). Every empty block is a masked lane in a GEMM.</div>

---

## Batch size is arithmetic intensity

A linear layer with a $d_{\text{in}} \times d_{\text{out}}$ weight matrix in bf16 (2 bytes per weight),
applied to $B$ tokens:

$$\text{FLOP} = 2 B\, d_{\text{in}} d_{\text{out}}, \qquad \text{bytes} \approx 2\, d_{\text{in}} d_{\text{out}} + 2B(d_{\text{in}} + d_{\text{out}})$$

When the weights dominate the bytes (any realistic $d$ and moderate $B$),

$$I \approx \frac{2 B\, d_{\text{in}} d_{\text{out}}}{2\, d_{\text{in}} d_{\text{out}}} = B \ \text{FLOP/byte}$$

**The batch size is the arithmetic intensity.** Each extra token in the batch reuses every weight byte
once more, exactly as each extra lane reuses one instruction fetch (Chapter 0's receipts).

**Numerical check.** An H100 SXM offers about 989 TFLOP/s of dense bf16 tensor-core compute and
3.35 TB/s of HBM bandwidth, a ridge point near $989 / 3.35 \approx 295$ FLOP/B (NVIDIA H100 datasheet).
A batch of 1 token per step sits at $I \approx 1$, below 0.5 % of the ridge. That is why single-stream
decoding is memory-bound and the tensor cores are mostly idle. Around 300 tokens processed together per
weight read, the layer can become compute-bound. On this book's single Xeon core with AMX (2.41 TFLOP/s
bf16, 21 GB/s from DRAM) the same arithmetic gives a ridge near 115.

| Phase | Tokens per weight read | Intensity | Limit |
|---|---|:---:|---|
| **Prefill** of a 2000-token prompt | 2000 | ~2000 FLOP/B | compute (GEMM) |
| **Decode**, 1 sequence | 1 | ~1 | weight bandwidth (GEMV) |
| **Decode**, 64 sequences batched | 64 | ~64 | still bandwidth, 64× better throughput |
| **Decode**, 256 sequences | 256 | ~256 | near the ridge |
| **Speculative decode**, verify 4 draft tokens | 4 per sequence | ~4× | fewer weight reads per accepted token |

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>Attention's KV-cache reads do <em>not</em> amortise across a batch: each sequence has its own cache,
so batching multiplies KV bytes by B along with FLOPs. As batches and contexts grow, the KV cache, not the
weights, becomes the dominant byte stream. Which techniques from earlier chapters target those bytes? (GQA,
KV quantization, paged gathers, FlashAttention-style tiling for prefill.)</p>
</div>

---

## Serving: from static batches to continuous batching

| Strategy | How a batch is formed | Wasted lanes come from | Systems |
|---|---|---|---|
| **No batching** | one request at a time | the whole GEMM is a GEMV | naive deployments |
| **Static batching** | wait for B requests, run them to completion together | padding to the longest prompt; finished sequences idle until the longest finishes | early TF Serving, simple scripts |
| **Dynamic batching** | collect requests for up to $t$ ms or $B$ requests | same as static, plus queueing latency | Triton Inference Server for fixed-shape models |
| **Continuous (iteration-level) batching** | re-form the batch every decode step: finished sequences leave, new ones join | almost none in the decode GEMMs | Orca (OSDI 2022), vLLM, SGLang, TensorRT-LLM |
| **Chunked prefill** | split long prompts into chunks and mix them into decode steps | balances compute-bound prefill with bandwidth-bound decode | vLLM, SGLang, Sarathi-Serve |
| **Prefill/decode disaggregation** | different GPUs for the two phases | transfers the KV cache between them | DistServe, Mooncake, production stacks |

Continuous batching is exactly Chapter 8's **lane refill**: when a lane (sequence) finishes its
"loop", it is immediately replaced with a new work item instead of sitting masked until the slowest lane
finishes. Static batching is the masked-until-done strategy, and its waste is the divergence of output
lengths.

---

## Training: padding, bucketing and packing

For variable-length training data, the padding efficiency of a batch is

$$\eta = \frac{\sum_b \ell_b}{B \cdot \max_b \ell_b}$$

**Numerical check** (the Tetris figure): lengths 11, 4, 16, 7, 9, 5 sum to 52; $B \cdot \max = 6 \cdot 16 = 96$;
$\eta = 52/96 = 54\%$. Nearly half of every GEMM in that batch multiplies padding.

| Technique | Efficiency | Cost | Use when |
|---|---|---|---|
| **Pad to the batch max** | as above (often 40–70 % on natural-length data) | none | lengths are similar |
| **Length bucketing** | high within a bucket | batches are no longer i.i.d. in length; shuffle buckets | variable lengths, fixed-shape kernels |
| **Sequence packing** | near 100 % | attention must not cross sequence boundaries: block-diagonal (document) masks or variable-length kernels (`cu_seqlens` in FlashAttention's varlen API) | pre-training, fine-tuning on short examples |
| **Nested / jagged tensors** | 100 % | operator coverage; kernels must handle ragged shapes | ragged features (TorchRec jagged tensors, `torch.nested`) |

<div class="callout warn"><span class="callout-title">Trap</span>
<p>Packing without a document mask lets tokens attend across unrelated examples that happen to share a
row. It trains fine on the loss curve and quietly changes what the model learns. Packing is legal only with the
mask, which is the padding-legality rule of Chapter 10 at the system level.</p>
</div>

---

## Batching beyond LLMs

| Workload | The lane axis | The shared "instruction" amortised |
|---|---|---|
| **Data loading and augmentation** | images in a batch, on GPU | one kernel launch and one decode pipeline per batch (DALI, torchvision v2 batched transforms) |
| **Recommender inference** | requests × candidate items | embedding-table reads and the dense tower's weights |
| **RL environments** | environments | one simulator step for thousands of states (EnvPool, Brax) |
| **Evaluation** | examples and metrics | vectorized metric computation instead of per-example Python |
| **Hyperparameter sweeps / ensembles** | models | one batched GEMM for many small models (vmap over parameters, Chapter 12) |
| **Embedding search** | queries | one GEMM of queries × database (the pairwise-distance problem, Chapter 20) |

---

## Speculative decoding: batching over the future

Decode is one token per weight read. Speculative decoding (Leviathan et al., 2023; Chen et al., 2023)
lets a cheap draft model propose $k$ tokens, then runs the large model *once* over all $k$ positions to
verify them. That single forward pass is a small GEMM with $k$ rows instead of $k$ separate GEMVs, which
raises the intensity of the large model's weight reads by about $k$ for the accepted tokens. It is
vectorization along the time axis, made legal by verification: the draft may be wrong, so the output
distribution is preserved only because rejected tokens are resampled.

---

## The latency price

Batching trades latency for throughput. By Little's law, the number of requests in the system equals
throughput × latency, so a server that runs larger batches at the same throughput holds each request
longer. Every serving system therefore has a latency target (time to first token, time per output
token) that caps the batch. The vector analogy holds here too: a wider register does more per
instruction, but each element waits for its whole vector.

---

## Key Takeaways

- For a linear layer with shared weights, arithmetic intensity is approximately the number of tokens processed per weight read: batch size is intensity.
- On an H100-class GPU the ridge is near 300 FLOP/B, so batch-1 decoding uses well under 1 % of tensor-core capability; batching fixes throughput, not latency.
- Continuous batching is lane refill; static batching is masking finished lanes until the slowest finishes.
- Padding efficiency is $\sum \ell / (B \max \ell)$; the example batch was 54 % padded and 81 % packed. Packing is legal only with document masks.
- KV-cache reads do not amortise across a batch; they become the dominant byte stream at large batch and long context.
- Speculative decoding vectorizes along time: verifying k drafted tokens in one pass turns k GEMVs into one small GEMM.

## Check Yourself

<details class="check"><summary>A 7B-parameter model in bf16 decodes at batch 1 on a GPU with 3.35 TB/s. Ignoring the KV cache, what is the best possible tokens per second?</summary>
Each token reads all weights once: 7 × 10⁹ × 2 B = 14 GB. 3.35 TB/s / 14 GB ≈ 240 tokens/s. Batching 64 sequences gives up to ~64× the aggregate throughput while the per-token weight traffic stays the same.
</details>

<details class="check"><summary>Lengths in a batch are 100, 100, 100 and 1000 tokens. What is the padding efficiency, and what would you do?</summary>
η = 1300 / (4 × 1000) = 32.5 %. Bucket by length (put the 1000-token sequence with other long ones) or pack the three short sequences into one row with a document mask.
</details>

<details class="check"><summary>Why does continuous batching help decode throughput more than prefill throughput?</summary>
Prefill is already compute-bound (thousands of tokens per weight read). Decode is bandwidth-bound and its sequences finish at different times; continuous batching keeps the decode batch full every step, raising tokens per weight read from "however many remain" to the target batch size.
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Orca: A Distributed Serving System for Transformer-Based Generative Models | Yu et al., OSDI 2022 | usenix.org | Iteration-level (continuous) batching |
| Efficient Memory Management for Large Language Model Serving with PagedAttention | Kwon et al., SOSP 2023 | arXiv:2309.06180 | vLLM, paged KV cache |
| Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve | Agrawal et al., OSDI 2024 | arXiv:2403.02310 | Chunked prefill |
| DistServe: Disaggregating Prefill and Decoding | Zhong et al., OSDI 2024 | arXiv:2401.09670 | Prefill/decode disaggregation |
| Fast Inference from Transformers via Speculative Decoding | Leviathan, Kalman & Matias, ICML 2023 | arXiv:2211.17192 | Speculative decoding |
| Accelerating Large Language Model Decoding with Speculative Sampling | Chen et al., 2023 | arXiv:2302.01318 | Speculative sampling |
| NVIDIA H100 Tensor Core GPU datasheet | NVIDIA, 2023 | nvidia.com | bf16 dense throughput, HBM bandwidth |
| FlashAttention-2 (variable-length API) | Dao, 2023 | arXiv:2307.08691 | Packed sequences with `cu_seqlens` |
| This book's measurements | `results.json` (`amx_peak`, `saxpy_memory_levels`) | [repo](https://github.com/karam-nus/vectorization/tree/main/tools/measurements) | CPU ridge point with AMX |

---

**Next:** [Chapter 20 — Hands-On: One Kernel, Eight Rungs](./20_handson_ladder.md) — climb from Python loops to AVX-512 on one problem, measuring every step.
