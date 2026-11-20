---
title: "Interactive Labs"
---

[← Back to Table of Contents](./README.md)

# Interactive Labs

> *"Move the slider until the number surprises you. That is where the understanding is."*

Ten small browser labs, one per idea that is easier to feel than to read. They run entirely in your
browser (`assets/js/vz.js`, no dependencies), and each mirrors a tested function in
[`code/vzlab`](https://github.com/karam-nus/vectorization/tree/main/code/vzlab), so the numbers on this page
and in the tests agree. On the raw GitHub view the labs do not run; open the published site instead.

---

## 1. Scalar vs SIMD — who processes which element

Step through an array of $n$ elements with $V$ lanes and three tail strategies: scalar epilogue, masked
last vector, or padding. The readout counts instructions and lane utilisation.
**Try:** $n = 27$, $V = 16$: the tail is most of the work. Switch to "masked" and watch 11 scalar instructions become one.

<div class="lab" data-lab="lanes"></div>

Chapters: [3 — Lanes & Registers](./03_lanes_and_registers.md), [8 — Masks, Divergence & Tails](./08_control_flow.md). Mirrors `vzlab.perfmodel.tail_cost`.

---

## 2. Dependence distance — may these iterations share a vector?

Move the read offset $R$ and the write offset $W$ of `a[i+W] = 0.5*a[i+R] + b[i]`. Red arrows are flow
dependences that land inside one vector chunk: those break. The lab checks the rule $d \le 0$ or $d \ge V$
against a brute-force scalar-versus-chunked simulation.
**Try:** $R = 0$, $W = 8$ with $V = 8$ (legal), then $V = 16$ (illegal).

<div class="lab" data-lab="deps"></div>

Chapter: [6 — Dependences & Aliasing](./06_dependences.md). Mirrors `vzlab.deps`.

---

## 3. Stride, layout and the bytes you actually move

Choose CPU cache lines or a GPU warp's 32-byte sectors, a stride or an AoS/SoA layout, and an element size.
The lab colours the bytes you use and the bytes that came along anyway.
**Try:** stride 16 with 4-byte elements on the CPU (6.25 % useful), then the GPU warp at stride 1 (4 sectors).

<div class="lab" data-lab="stride"></div>

Chapters: [7 — Memory Is the Real Vector](./07_memory_layout.md), [15 — GPUs](./15_gpus_simt.md). Mirrors `vzlab.memory`.

---

## 4. Masks and divergence

128 elements, a condition true with probability $p$, $V$ lanes per chunk. Each chunk runs the "if" side if any
lane needs it and the "else" side if any lane needs it.
**Try:** $p = 0.5$, random order (≈ 50 % utilisation), then *sorted by condition* (≈ 100 %).

<div class="lab" data-lab="mask"></div>

Chapter: [8 — Masks, Divergence & Tails](./08_control_flow.md). Mirrors `vzlab.perfmodel.branch_utilization`.

---

## 5. Roofline with vector roofs

The measured roofs of one Xeon core (scalar 12.9 → AVX-512 185.8 GFLOP/s) and its measured bandwidth per
memory level. Pick a kernel or drag its arithmetic intensity, and pick where its data lives.
**Try:** saxpy from DRAM (AVX-512 gains nothing over scalar at the roof), then from L1 (4×).

<div class="lab" data-lab="roofline"></div>

Chapter: [9 — Is It Worth It?](./09_profitability.md). Mirrors `vzlab.perfmodel.roofline`.

---

## 6. Amdahl and the vector fraction

The vectorizable fraction $f$, the lane count $V$ and a lane efficiency that folds in tails, masks and stalls.
**Try:** $f = 0.9$: no width beats 10×. Then lower the lane efficiency to 0.3.

<div class="lab" data-lab="amdahl"></div>

Chapter: [9 — Is It Worth It?](./09_profitability.md). Mirrors `vzlab.perfmodel.amdahl`.

---

## 7. Same numbers, different order, different answer

Sum the same float32 numbers sequentially, with $V$ lane accumulators plus a tree, pairwise, and with Kahan
compensation. Bars show the error in ulps of the exact sum.
**Try:** the alternating ±10⁸ data: order decides whether the small values survive at all.

<div class="lab" data-lab="reduce"></div>

Chapter: [6 — Dependences & Aliasing](./06_dependences.md). Mirrors `vzlab.reduce`.

---

## 8. Broadcasting

Type two shapes. The lab right-aligns them, marks which operand is stretched along which axis (a stride-0
view), and draws small 2-D cases as spreadsheets with ghost cells.
**Try:** `(4, 1)` and `(3,)`; then `(3, 4)` and `(3,)` for the classic error.

<div class="lab" data-lab="broadcast"></div>

Chapter: [11 — Array Programming](./11_array_programming.md). Mirrors `vzlab.broadcast`.

---

## 9. A systolic array, one clock at a time

An output-stationary $R \times R$ array multiplies small integer matrices. Watch the skewed operands meet,
count fill and drain clocks, and confirm the result equals $A \cdot B$.
**Try:** $4 \times 4$ with $K = 3$ (utilisation under 40 %), then $K = 6$.

<div class="lab" data-lab="systolic"></div>

Chapter: [16 — Matrix Engines](./16_matrix_engines.md). Mirrors `vzlab.systolic`.

---

## 10. The Vector ISA Atlas

Every instruction set and matrix engine in the book on one chart, with a filterable table. It has its own page:
[Vector ISA Atlas](./atlas.md).

<div class="lab" data-lab="atlas"></div>

Chapters: [5 — Cray-1 to Tensor Cores](./05_history.md), [17 — The Architect's View](./17_architects_view.md), [Appendix D](./appendix_d_hardware_matrix.md).
