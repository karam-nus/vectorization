---
title: "Vectorization — The Complete Guide"
permalink: /
---

# 🧮 Vectorization — The Complete Guide

> **From one lane to ten thousand**: how computation is reshaped to run many-at-once, in NumPy, in compilers, in CUDA kernels and in silicon, and how to *see* it before you write it. Every speedup in this book is measured, every rule is derived, and every idea has a picture.

## Who This Is For

You are an ML scientist or AI engineer. You know Python, NumPy and PyTorch, you have heard that "vectorized
code is faster", and now you have to make something faster yourself: a data pipeline, a custom loss, a
kernel, a serving system, or a decision about hardware. This book treats vectorization as one idea seen on
every floor of the stack, *one decision applied to many data*. It is organised around six questions that
every vectorization decision must answer: what the vector is, whether reordering is legal, how the data
arrives, what to do about control flow, whether it is worth it, and who should do it.

The authoring standard is checked in at [PROMPT.md](./PROMPT.md).

## What Makes This Book Different

<div class="diagram">
<div class="diagram-grid cols-4">
  <div class="diagram-card accent"><div class="card-icon">📏</div><div class="card-title">Measured, not quoted</div><div class="card-desc">Every speedup comes from the companion benchmarks on one named machine: scalar to AVX-512 to AMX, L1 to DRAM, Python to intrinsics. The raw JSON and the scripts are in the repo.</div></div>
  <div class="diagram-card purple"><div class="card-icon">🎨</div><div class="card-title">A visual grammar</div><div class="card-desc">Each idea gets a familiar picture: receipts for energy, synchronized swimming for lanes, blueprints for hardware, CRT terminals for compiler output, a periodic table of techniques, cork boards of questions, metro maps for what's next.</div></div>
  <div class="diagram-card green"><div class="card-icon">🧪</div><div class="card-title">Ten interactive labs</div><div class="card-desc">Step through SIMD tails, move dependence arrows, watch cache lines fill, diverge a warp, slide along a measured roofline, clock a systolic array. Each lab mirrors a tested function.</div></div>
  <div class="diagram-card cyan"><div class="card-icon">✅</div><div class="card-title">Tested companion code</div><div class="card-desc"><code>code/vzlab</code> (67 tests): loop vs vectorized idioms, legality by simulation, summation orders, broadcasting, roofline and systolic models. Plus C/intrinsics kernels, SVE/RVV/NEON listings and an AMX microbenchmark.</div></div>
</div>
</div>

## 📋 Table of Contents

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| **Orientation** | | |
| 0 | [Why Think About Vectorization?](./00_why_vectorization.md) | Three measured gaps: unused lanes (14.4×), energy spent deciding, interpreter overhead (834×), and when not to care |
| 1 | [The Whole Field on One Page](./01_grand_overview.md) | The vectorization stack as a building, the six questions, five shapes of parallel hardware |
| 2 | [A Field Guide to Vectorizable Workloads](./02_domains.md) | Ten loop species across 14 domains, and vectorizing *across* instances when *within* fails |
| **Part I — The Mental Model** | | |
| 3 | [Lanes, Registers & the SIMD Contract](./03_lanes_and_registers.md) | V = w/e, the four promises of SIMD, eight instruction families drawn lane by lane |
| 4 | [Seeing Vectors: A Visual Grammar](./04_seeing_vectors.md) | **How to visualize it**: seven drawings, from register strips to rooflines, and the tools that draw them |
| 5 | [From Cray-1 to Tensor Cores](./05_history.md) | Fifty years as front pages, and the three arguments the field keeps having |
| **Part II — Legality & Profitability** | | |
| 6 | [Dependences & Aliasing](./06_dependences.md) | The d ≤ 0 or d ≥ V rule, aliasing, NumPy's copy semantics, why float reductions need permission |
| 7 | [Memory Is the Real Vector](./07_memory_layout.md) | Cache lines, stride, AoS vs SoA, tensor layouts, coalescing; why width stops mattering beyond L2 |
| 8 | [Masks, Predication, Divergence & Tails](./08_control_flow.md) | If-conversion, safe speculation, masking across ISAs, divergence probability, compaction, tails |
| 9 | [Is It Worth It? Roofline, Amdahl & Real Ceilings](./09_profitability.md) | A measured roofline, the scalar ridge point, Amdahl, latency chains, honest measurement |
| **Part III — The Toolbox** | | |
| 10 | [The Technique Catalogue](./10_technique_catalogue.md) | **44 techniques** as a periodic table and 18 trading cards; parallel scan, histograms, outer-loop vectorization |
| 11 | [Array Programming: NumPy, PyTorch & einsum](./11_array_programming.md) | The per-call cost model, `np.vectorize` exposed, broadcasting as spreadsheets, temporaries, idioms |
| 12 | [Automatic Batching: vmap, jit & Friends](./12_vmap.md) | Batching rules on real jaxprs, GEMV → GEMM, what vmap cannot batch, fusion and codegen |
| 13 | [Compiler Auto-Vectorization](./13_auto_vectorization.md) | The vectorizer's six steps, real gcc/clang remarks, the remark → cause → fix table |
| 14 | [Explicit SIMD: Intrinsics, Libraries & SPMD](./14_explicit_simd.md) | Reading intrinsics, measured accumulators, SVE and RISC-V V without tails, Highway, ISPC, dispatch |
| **Part IV — Accelerators & Silicon** | | |
| 15 | [GPUs: SIMT, Warps & Tiles](./15_gpus_simt.md) | Warps as vectors, coalescing, warp shuffles, tensor-core generations, Triton, wave quantization |
| 16 | [Matrix Engines: Systolic Arrays, Tensor Cores, AMX & SME](./16_matrix_engines.md) | Operand reuse N/2, systolic dataflows, **AMX measured at 13× AVX-512** on one core |
| 17 | [The Architect's View: Designing Vector Hardware](./17_architects_view.md) | What grows with width, fixed vs length-agnostic, data types as area, model–hardware co-design |
| **Part V — ML Systems** | | |
| 18 | [Vectorizing ML Kernels](./18_ml_kernels.md) | Vectorized exp, online softmax, Welford, the GEMM microkernel derived, FlashAttention, quantized dots |
| 19 | [Batching: Vectorization at System Scale](./19_batching_systems.md) | Batch size is arithmetic intensity, continuous batching, packing vs padding, speculative decoding |
| **Part VI — Practice & Thinking** | | |
| 20 | [Hands-On: One Kernel, Eight Rungs](./20_handson_ladder.md) | Pairwise distances from Python to AVX-512, measured, with the two rungs that go *down* |
| 21 | [The Debugging & Profiling Playbook](./21_debugging_playbook.md) | Four symptoms, triage tables, a profiler-free "did it vectorize?" check, the tool atlas |
| 22 | [Questions Worth Pondering & The Decision Guide](./22_questions_worth_pondering.md) | Forty questions on seven cork boards, and first moves by starting point |
| 23 | [Where to Go Next](./23_where_next.md) | A metro map into quantization, compilers, GPUs, silicon, serving and scale |
| **Interactive** | | |
| ⚗ | [Interactive Labs](./labs.md) | All ten labs on one page |
| 🗺 | [Vector ISA Atlas](./atlas.md) | Every vector ISA and matrix engine from the Cray-1 to Ironwood, filterable |
| **Appendices** | | |
| A | [Glossary & Symbols](./appendix_a_glossary.md) | Every term and symbol, one definition each |
| B | [Math Reference](./appendix_b_math.md) | Every formula with its numerical check |
| C | [Technique Index](./appendix_c_technique_index.md) | The master table: 44 techniques × legality × payoff × failure mode |
| D | [Hardware Matrix](./appendix_d_hardware_matrix.md) | ISA × width × masking × gather × ML types (generated from `isas.json`) |
| E | [Bibliography](./appendix_e_bibliography.md) | The reading list by theme, with a suggested order |
| F | [Question Bank](./appendix_f_question_bank.md) | Sixty self-test and interview questions keyed to chapters |

## 🗺️ Learning Path

<div class="metro-map">
<svg viewBox="0 0 960 250" role="img" aria-label="Line diagram of the book: six coloured segments for the six parts, one station per chapter">
<g stroke-width="10" fill="none" stroke-linecap="round">
<path d="M40 150 H150" stroke="#111111"/><path d="M150 150 H262" stroke="#0891b2"/><path d="M262 150 H410" stroke="#7c3aed"/>
<path d="M410 150 H595" stroke="#16a34a"/><path d="M595 150 H706" stroke="#ea580c"/><path d="M706 150 H780" stroke="#db2777"/><path d="M780 150 H930" stroke="#2563eb"/>
</g>
<g font-size="11" font-weight="700">
<text x="95" y="190" text-anchor="middle" fill="#111">ORIENTATION</text><text x="206" y="190" text-anchor="middle" fill="#0891b2">MENTAL MODEL</text>
<text x="336" y="190" text-anchor="middle" fill="#7c3aed">LEGALITY &amp; PROFIT</text><text x="502" y="190" text-anchor="middle" fill="#16a34a">TOOLBOX</text>
<text x="650" y="190" text-anchor="middle" fill="#ea580c">ACCELERATORS</text><text x="743" y="190" text-anchor="middle" fill="#db2777">ML SYSTEMS</text><text x="855" y="190" text-anchor="middle" fill="#2563eb">PRACTICE</text>
</g>
<g font-size="11" fill="#1f2937" stroke="none">
<a href="./00_why_vectorization.html"><circle cx="55" cy="150" r="9" fill="#fff" stroke="#111" stroke-width="4"/><text x="55" y="130" transform="rotate(-40 55 130)">00 Why</text></a>
<a href="./01_grand_overview.html"><circle cx="92" cy="150" r="9" fill="#fff" stroke="#111" stroke-width="4"/><text x="92" y="130" transform="rotate(-40 92 130)">01 The field</text></a>
<a href="./02_domains.html"><circle cx="129" cy="150" r="9" fill="#fff" stroke="#111" stroke-width="4"/><text x="129" y="130" transform="rotate(-40 129 130)">02 Domains</text></a>
<a href="./03_lanes_and_registers.html"><circle cx="170" cy="150" r="9" fill="#fff" stroke="#0891b2" stroke-width="4"/><text x="170" y="130" transform="rotate(-40 170 130)">03 Lanes</text></a>
<a href="./04_seeing_vectors.html"><circle cx="207" cy="150" r="12" fill="#fff" stroke="#0891b2" stroke-width="4"/><text x="207" y="128" transform="rotate(-40 207 128)" font-weight="700">04 Seeing vectors ★</text></a>
<a href="./05_history.html"><circle cx="244" cy="150" r="9" fill="#fff" stroke="#0891b2" stroke-width="4"/><text x="244" y="130" transform="rotate(-40 244 130)">05 History</text></a>
<a href="./06_dependences.html"><circle cx="282" cy="150" r="12" fill="#fff" stroke="#7c3aed" stroke-width="4"/><text x="282" y="128" transform="rotate(-40 282 128)" font-weight="700">06 Legality ★</text></a>
<a href="./07_memory_layout.html"><circle cx="319" cy="150" r="12" fill="#fff" stroke="#7c3aed" stroke-width="4"/><text x="319" y="128" transform="rotate(-40 319 128)" font-weight="700">07 Memory ★</text></a>
<a href="./08_control_flow.html"><circle cx="356" cy="150" r="9" fill="#fff" stroke="#7c3aed" stroke-width="4"/><text x="356" y="130" transform="rotate(-40 356 130)">08 Masks &amp; tails</text></a>
<a href="./09_profitability.html"><circle cx="393" cy="150" r="12" fill="#fff" stroke="#7c3aed" stroke-width="4"/><text x="393" y="128" transform="rotate(-40 393 128)" font-weight="700">09 Roofline ★</text></a>
<a href="./10_technique_catalogue.html"><circle cx="430" cy="150" r="9" fill="#fff" stroke="#16a34a" stroke-width="4"/><text x="430" y="130" transform="rotate(-40 430 130)">10 Techniques</text></a>
<a href="./11_array_programming.html"><circle cx="467" cy="150" r="9" fill="#fff" stroke="#16a34a" stroke-width="4"/><text x="467" y="130" transform="rotate(-40 467 130)">11 NumPy</text></a>
<a href="./12_vmap.html"><circle cx="504" cy="150" r="9" fill="#fff" stroke="#16a34a" stroke-width="4"/><text x="504" y="130" transform="rotate(-40 504 130)">12 vmap</text></a>
<a href="./13_auto_vectorization.html"><circle cx="541" cy="150" r="9" fill="#fff" stroke="#16a34a" stroke-width="4"/><text x="541" y="130" transform="rotate(-40 541 130)">13 Compilers</text></a>
<a href="./14_explicit_simd.html"><circle cx="578" cy="150" r="9" fill="#fff" stroke="#16a34a" stroke-width="4"/><text x="578" y="130" transform="rotate(-40 578 130)">14 Intrinsics</text></a>
<a href="./15_gpus_simt.html"><circle cx="615" cy="150" r="9" fill="#fff" stroke="#ea580c" stroke-width="4"/><text x="615" y="130" transform="rotate(-40 615 130)">15 GPUs</text></a>
<a href="./16_matrix_engines.html"><circle cx="652" cy="150" r="9" fill="#fff" stroke="#ea580c" stroke-width="4"/><text x="652" y="130" transform="rotate(-40 652 130)">16 Matrix engines</text></a>
<a href="./17_architects_view.html"><circle cx="689" cy="150" r="9" fill="#fff" stroke="#ea580c" stroke-width="4"/><text x="689" y="130" transform="rotate(-40 689 130)">17 Architecture</text></a>
<a href="./18_ml_kernels.html"><circle cx="726" cy="150" r="12" fill="#fff" stroke="#db2777" stroke-width="4"/><text x="726" y="128" transform="rotate(-40 726 128)" font-weight="700">18 ML kernels ★</text></a>
<a href="./19_batching_systems.html"><circle cx="763" cy="150" r="9" fill="#fff" stroke="#db2777" stroke-width="4"/><text x="763" y="130" transform="rotate(-40 763 130)">19 Batching</text></a>
<a href="./20_handson_ladder.html"><circle cx="800" cy="150" r="9" fill="#fff" stroke="#2563eb" stroke-width="4"/><text x="800" y="130" transform="rotate(-40 800 130)">20 The ladder</text></a>
<a href="./21_debugging_playbook.html"><circle cx="837" cy="150" r="9" fill="#fff" stroke="#2563eb" stroke-width="4"/><text x="837" y="130" transform="rotate(-40 837 130)">21 Debugging</text></a>
<a href="./22_questions_worth_pondering.html"><circle cx="874" cy="150" r="9" fill="#fff" stroke="#2563eb" stroke-width="4"/><text x="874" y="130" transform="rotate(-40 874 130)">22 Questions</text></a>
<a href="./23_where_next.html"><circle cx="911" cy="150" r="9" fill="#fff" stroke="#2563eb" stroke-width="4"/><text x="911" y="130" transform="rotate(-40 911 130)">23 Next</text></a>
</g>
<text x="40" y="232" font-size="11" fill="#555">★ = the five chapters to read first · every station is a link</text>
</svg>
</div>

## ⚡ Quick Start Paths

### Path A: "My NumPy / PyTorch code is slow"

1. [11 — Array Programming](./11_array_programming.md): the cost model and the idioms
2. [12 — vmap & jit](./12_vmap.md): batching and fusion done for you
3. [07 — Memory Is the Real Vector](./07_memory_layout.md): strides, layouts, temporaries
4. [20 — Hands-On: The Ladder](./20_handson_ladder.md): one problem, eight measured rungs

### Path B: "I write C++ or CUDA kernels"

1. [06 — Dependences](./06_dependences.md) → [13 — Auto-Vectorization](./13_auto_vectorization.md): why the compiler said no
2. [09 — Roofline](./09_profitability.md): whether it can ever be faster
3. [14 — Explicit SIMD](./14_explicit_simd.md) → [15 — GPUs](./15_gpus_simt.md)
4. [18 — ML Kernels](./18_ml_kernels.md) → [21 — Debugging Playbook](./21_debugging_playbook.md)

### Path C: "I design or evaluate hardware"

1. [00 — Why](./00_why_vectorization.md) → [05 — History](./05_history.md): the energy argument and the pendulum
2. [16 — Matrix Engines](./16_matrix_engines.md) → [17 — The Architect's View](./17_architects_view.md)
3. [09 — Roofline](./09_profitability.md) and the [Vector ISA Atlas](./atlas.md)

### Path D: "I train or serve models at scale"

1. [19 — Batching at System Scale](./19_batching_systems.md): batch size is arithmetic intensity
2. [18 — ML Kernels](./18_ml_kernels.md): softmax, attention, quantized dots
3. [15 — GPUs](./15_gpus_simt.md): tiles, waves, alignment
4. [23 — Where Next](./23_where_next.md): quantization, inference, distributed training

### Path E: "I want to see it before I read it"

1. [04 — Seeing Vectors](./04_seeing_vectors.md): the seven drawings
2. [Interactive Labs](./labs.md): all ten labs
3. [10 — The Technique Catalogue](./10_technique_catalogue.md): the periodic table
4. [22 — Questions Worth Pondering](./22_questions_worth_pondering.md): the cork boards

## 🧰 Companion Code

```bash
cd code && pip install -r requirements.txt && pytest          # 67 tests: vzlab vs loops, simulation, numpy
python tools/measurements/run_bench.py                         # C kernels x 5 compile targets (+ AMX if present)
python tools/measurements/python_bench.py                      # Python / NumPy rungs and idioms
python tools/make_figures.py                                   # every chart in assets/images from results.json
```

See [code/README.md](https://github.com/karam-nus/vectorization/blob/main/code/README.md) for the module ↔ chapter map
and [tools/measurements/README.md](https://github.com/karam-nus/vectorization/blob/main/tools/measurements/README.md)
for the machine, the method, and every experiment behind the numbers.

---

*Part of a knowledge base published at [karam-nus.github.io](https://karam-nus.github.io). Companion repos: [quantization](https://karam-nus.github.io/quantization), [inference](https://karam-nus.github.io/inference), [nvidia](https://karam-nus.github.io/nvidia), [chips](https://karam-nus.github.io/chips), [language-modelling](https://karam-nus.github.io/language-modelling), [object-detection](https://karam-nus.github.io/object-detection), [model-optimization](https://karam-nus.github.io/model-optimization).*
