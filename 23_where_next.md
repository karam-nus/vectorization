---
title: "Chapter 23 — Where to Go Next"
---

[← Back to Table of Contents](./README.md)

# Chapter 23 — Where to Go Next

> *"Vectorization is the first time you learn to think about many things at once. Everything after it is the same lesson at a different scale."*

## Overview

Vectorization sits at an interchange. From here six lines lead outward: precision (fewer bits per lane),
compilers and fusion (fewer bytes per operation), GPUs (SIMT in depth), silicon (where lanes and tiles come
from), serving systems (batching at datacenter scale), and parallelism beyond one core. The map below
links each station to a chapter in this knowledge base. The sections after it say why each line is a
natural next step and what carries over.

<div class="metro-map">
<svg viewBox="0 0 960 520" role="img" aria-label="Metro map of next topics radiating from vectorization">
<rect x="40" y="40" width="120" height="440" rx="60" fill="#ffffff" stroke="#111111" stroke-width="5"/>
<text x="100" y="260" text-anchor="middle" font-size="17" font-weight="800" fill="#111" transform="rotate(-90 100 260)">VECTORIZATION</text>
<text x="128" y="260" text-anchor="middle" font-size="11" fill="#555" transform="rotate(-90 128 260)">you are here · interchange for all lines</text>
<g stroke-width="9" fill="none" stroke-linecap="round">
<path d="M160 80 H900" stroke="#7c3aed"/>
<path d="M160 155 H560 L600 195 H900" stroke="#0d9488"/>
<path d="M160 235 H900" stroke="#16a34a"/>
<path d="M160 310 H900" stroke="#2563eb"/>
<path d="M160 385 H900" stroke="#ea580c"/>
<path d="M160 460 H900" stroke="#dc2626"/>
</g>
<g font-size="10.5" font-weight="700" fill="#ffffff">
<rect x="172" y="69" width="78" height="22" rx="11" fill="#7c3aed"/><text x="211" y="84" text-anchor="middle">PRECISION</text>
<rect x="172" y="144" width="78" height="22" rx="11" fill="#0d9488"/><text x="211" y="159" text-anchor="middle">COMPILERS</text>
<rect x="172" y="224" width="78" height="22" rx="11" fill="#16a34a"/><text x="211" y="239" text-anchor="middle">GPU</text>
<rect x="172" y="299" width="78" height="22" rx="11" fill="#2563eb"/><text x="211" y="314" text-anchor="middle">SILICON</text>
<rect x="172" y="374" width="78" height="22" rx="11" fill="#ea580c"/><text x="211" y="389" text-anchor="middle">SERVING</text>
<rect x="172" y="449" width="78" height="22" rx="11" fill="#dc2626"/><text x="211" y="464" text-anchor="middle">SCALE</text>
</g>
<g stroke-width="4" fill="#ffffff">
<a href="https://karam-nus.github.io/quantization/02_number_formats"><circle cx="320" cy="80" r="8" stroke="#7c3aed"/><text x="320" y="62" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">Number formats</text></a>
<a href="https://karam-nus.github.io/quantization/"><circle cx="490" cy="80" r="11" stroke="#7c3aed"/><text x="490" y="60" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937" stroke="none">Quantization ★</text></a>
<a href="https://karam-nus.github.io/quantization/41_quantized_kernels"><circle cx="660" cy="80" r="8" stroke="#7c3aed"/><text x="660" y="62" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">Quantized kernels</text></a>
<a href="https://github.com/karam-nus/model-optimization"><circle cx="830" cy="80" r="8" stroke="#7c3aed" stroke-dasharray="4 3"/><text x="830" y="62" text-anchor="middle" font-size="12" fill="#6b7280" stroke="none">Model optimization (planned)</text></a>
<a href="./18_ml_kernels.html"><circle cx="320" cy="155" r="8" stroke="#0d9488"/><text x="320" y="137" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">Kernel fusion (Ch 18)</text></a>
<a href="https://karam-nus.github.io/software/17_performance_and_profiling"><circle cx="490" cy="155" r="8" stroke="#0d9488"/><text x="490" y="137" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">Profiling</text></a>
<a href="https://karam-nus.github.io/inference/11_compiler_stack"><circle cx="680" cy="195" r="11" stroke="#0d9488"/><text x="680" y="177" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937" stroke="none">ML compiler stack ★</text></a>
<a href="https://karam-nus.github.io/language-modelling/25_cuda_and_kernel_development"><circle cx="840" cy="195" r="8" stroke="#0d9488"/><text x="840" y="177" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">Kernel development</text></a>
<a href="https://karam-nus.github.io/nvidia/02_gpu_microarchitecture"><circle cx="320" cy="235" r="8" stroke="#16a34a"/><text x="320" y="257" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">GPU microarchitecture</text></a>
<a href="https://karam-nus.github.io/nvidia/03_cuda_programming"><circle cx="490" cy="235" r="11" stroke="#16a34a"/><text x="490" y="259" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937" stroke="none">CUDA programming ★</text></a>
<a href="https://karam-nus.github.io/nvidia/05_optimization_stack"><circle cx="680" cy="235" r="8" stroke="#16a34a"/><text x="680" y="257" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">Optimization stack</text></a>
<a href="https://karam-nus.github.io/nvidia/08_gpu_generations"><circle cx="830" cy="235" r="8" stroke="#16a34a"/><text x="830" y="257" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">GPU generations</text></a>
<a href="https://karam-nus.github.io/chips/08_parallel_processing"><circle cx="320" cy="310" r="8" stroke="#2563eb"/><text x="320" y="332" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">Parallel processing</text></a>
<a href="https://karam-nus.github.io/chips/11_memory_wall_and_bandwidth"><circle cx="490" cy="310" r="11" stroke="#2563eb"/><text x="490" y="334" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937" stroke="none">The memory wall ★</text></a>
<a href="https://karam-nus.github.io/chips/16_tpus_and_systolic_arrays"><circle cx="660" cy="310" r="8" stroke="#2563eb"/><text x="660" y="332" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">TPUs &amp; systolic arrays</text></a>
<a href="https://karam-nus.github.io/chips/31_chip_design_flow"><circle cx="830" cy="310" r="8" stroke="#2563eb"/><text x="830" y="332" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">Chip design flow</text></a>
<a href="https://karam-nus.github.io/language-modelling/18_kv_cache_optimization"><circle cx="320" cy="385" r="8" stroke="#ea580c"/><text x="320" y="407" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">KV-cache optimization</text></a>
<a href="https://karam-nus.github.io/inference/09_serving_and_middleware"><circle cx="490" cy="385" r="11" stroke="#ea580c"/><text x="490" y="409" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937" stroke="none">Serving systems ★</text></a>
<a href="https://karam-nus.github.io/inference/10_edge_and_embedded"><circle cx="660" cy="385" r="8" stroke="#ea580c"/><text x="660" y="407" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">Edge &amp; embedded</text></a>
<a href="https://karam-nus.github.io/vision/31_efficient_inference"><circle cx="830" cy="385" r="8" stroke="#ea580c"/><text x="830" y="407" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">Efficient vision inference</text></a>
<a href="https://karam-nus.github.io/nvidia/07_parallelism"><circle cx="320" cy="460" r="8" stroke="#dc2626"/><text x="320" y="442" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">Parallelism</text></a>
<a href="https://karam-nus.github.io/language-modelling/26_distributed_training"><circle cx="490" cy="460" r="11" stroke="#dc2626"/><text x="490" y="440" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2937" stroke="none">Distributed training ★</text></a>
<a href="https://karam-nus.github.io/language-modelling/33_mixture_of_experts"><circle cx="660" cy="460" r="8" stroke="#dc2626"/><text x="660" y="442" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">Mixture of experts</text></a>
<a href="https://karam-nus.github.io/nvidia/10_multi_gpu_interconnects"><circle cx="830" cy="460" r="8" stroke="#dc2626"/><text x="830" y="442" text-anchor="middle" font-size="12" fill="#1f2937" stroke="none">Multi-GPU interconnects</text></a>
</g>
<rect x="655" y="113" width="50" height="132" rx="10" fill="none" stroke="#111" stroke-width="2.5" stroke-dasharray="2 3" opacity="0.35"/>
</svg>
<div class="metro-legend"><span style="--m:#7c3aed">Precision — fewer bits per lane</span><span style="--m:#0d9488">Compilers — fewer bytes per op</span><span style="--m:#16a34a">GPU — SIMT in depth</span><span style="--m:#2563eb">Silicon — where lanes come from</span><span style="--m:#ea580c">Serving — the batch as a vector</span><span style="--m:#dc2626">Scale — beyond one chip</span></div>
</div>
<div class="genre-caption">Click any station. ★ marks the recommended first stop on each line; dashed stations are planned books. The dotted outline groups three stations that teach the same thing from three sides: how a kernel becomes fast.</div>

---

## Why each line follows from this book

| Line | The vectorization idea it extends | What carries over | Start at |
|---|---|---|---|
| **Precision** | halving the bits doubles the lanes and halves the bytes (Chapters 3, 9, 17) | lane arithmetic, VNNI/`dp4a`, unpacking, scales as broadcasts | [Quantization](https://karam-nus.github.io/quantization/) |
| **Compilers** | fusion removes passes; code generation emits the SIMD (Chapters 12, 13, 18) | legality, cost models, schedules, reading generated code | [Inference: compiler stack](https://karam-nus.github.io/inference/11_compiler_stack) |
| **GPU** | SIMT is SIMD with the vector hidden (Chapter 15) | coalescing, divergence, warp reductions, tensor-core tiles | [NVIDIA: CUDA programming](https://karam-nus.github.io/nvidia/03_cuda_programming) |
| **Silicon** | operand reuse, systolic arrays, the cost of width (Chapters 16, 17) | the roofline, energy per operation, dataflows | [Chips: the memory wall](https://karam-nus.github.io/chips/11_memory_wall_and_bandwidth) |
| **Serving** | batch size is arithmetic intensity (Chapter 19) | GEMV vs GEMM, padding vs packing, lane refill | [Inference: serving](https://karam-nus.github.io/inference/09_serving_and_middleware) |
| **Scale** | "one decision, many data" at the grain of cores, GPUs and nodes | Amdahl, communication as the new memory wall, load balance as divergence | [Language modelling: distributed training](https://karam-nus.github.io/language-modelling/26_distributed_training) |

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>Data parallelism across GPUs is vectorization with the batch split across chips, and the all-reduce of
gradients is the horizontal reduction of Chapter 3, done over a network. Tensor parallelism is tiling a
GEMM across devices. Which of this book's ceilings (memory, tails, divergence, Amdahl) reappears first when
you scale out, and what plays the role of the cache line?</p>
</div>

---

## If you read only one more thing: quantization

The most direct continuation is **quantization**, because it pulls both of this book's levers at once.
Fewer bits per value means more lanes per instruction (Chapter 3) and fewer bytes per operand
(Chapter 9), and in the decode regime of Chapter 19 bytes are the whole story. The companion book
covers the formats, calibration, GPTQ/AWQ/SmoothQuant, and, in its kernels chapter, exactly the
unpacking and dot-product instructions this book introduced. Start with
[Hardware & the Q/DQ contract](https://karam-nus.github.io/quantization/01_hardware_and_the_qdq_contract)
and [How a quantized matmul runs](https://karam-nus.github.io/quantization/41_quantized_kernels).

---

## Outside this knowledge base

| Resource | Why it is worth your time |
|---|---|
| **Algorithms for Modern Hardware** (Slotin, en.algorithmica.org/hpc) | the best free treatment of SIMD algorithms: binary search, prefix sums, sorting, with measured code |
| **Agner Fog's optimization manuals** (agner.org/optimize) | instruction tables and microarchitecture guides for every x86 core |
| **Performance Analysis and Tuning on Modern CPUs** (Bakhvalov) | profiling methodology, top-down analysis, hardware counters |
| **Programming Massively Parallel Processors**, 4th ed. (Hwu, Kirk & El Hajj) | the standard CUDA text: tiling, reductions, scans, sparse kernels |
| **Computer Architecture: A Quantitative Approach**, ch. 4 and 7 (Hennessy & Patterson) | vector, SIMD and GPU architectures; domain-specific accelerators |
| **Daniel Lemire's blog** (lemire.me) | short, measured SIMD tricks for parsing, hashing and decoding |
| **Triton tutorials** and **CUTLASS / CuTe documentation** | block-level GPU programming and tensor-core tiling in practice |
| **Highway** and **std::simd** documentation | portable SIMD you can ship |

---

## Projects to cement it

1. **Climb your own ladder.** Take one Python loop from your work and repeat Chapter 20: measure every rung,
   record the machine, plot it with `tools/make_figures.py` as a template.
2. **Write a fused kernel.** Implement RMSNorm + residual add + int8 quantization in Triton or C++, and show
   on a roofline that it touches each byte once.
3. **Read a generated kernel.** Run a small model under `torch.compile` with `TORCH_LOGS=output_code`, find a
   reduction and a fused elementwise chain, and name the techniques from the periodic table that the compiler used.
4. **Port to vector-length agnostic.** Rewrite one intrinsics kernel with Highway or SVE intrinsics, cross-compile
   it as in `code/simd/check.sh`, and compare the instruction count of the tail.
5. **Model a matrix engine.** Extend `vzlab.systolic` to weight-stationary dataflow and back-to-back tiles, and
   find the matrix shapes at which utilisation exceeds 90 %.

---

## Key Takeaways

- Six lines lead out of vectorization: precision, compilers, GPUs, silicon, serving and scale; each extends one idea from this book.
- Quantization is the most direct next step because it multiplies lanes and divides bytes at once.
- Distributed parallelism is the same "one decision, many data" bargain at a larger grain, with communication playing the memory system's role.
- The best way to keep the knowledge is to climb a ladder on your own code and plot it.

## Check Yourself

<details class="check"><summary>Which two of this book's quantities does quantization change, and in which direction?</summary>
Lanes per instruction V = w/e goes up as e shrinks; bytes per operand go down, raising arithmetic intensity I for a fixed computation. Both move a kernel up the roofline, provided accuracy holds.
</details>

<details class="check"><summary>What is the distributed-training analogue of (a) a horizontal reduction, (b) a tail, (c) divergence?</summary>
(a) The gradient all-reduce. (b) Uneven last micro-batches or shards, idle devices in the final wave. (c) Load imbalance: some ranks finishing earlier (variable sequence lengths, MoE expert imbalance) while others still work.
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Knowledge base index | karam-nus | [karam-nus.github.io](https://karam-nus.github.io) | Companion books on the map |
| Algorithms for Modern Hardware | Slotin, 2022– | en.algorithmica.org/hpc | Recommended reading |
| Programming Massively Parallel Processors, 4th ed. | Hwu, Kirk & El Hajj, 2022 | Morgan Kaufmann | Recommended reading |
| Performance Analysis and Tuning on Modern CPUs | Bakhvalov, 2020/2024 | easyperf.net | Recommended reading |
| Software optimization resources | Fog | agner.org/optimize | Recommended reading |

---

**Next:** [Interactive Labs](./labs.md) — all ten labs from the book on one page.
