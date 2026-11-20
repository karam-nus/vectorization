---
title: "Chapter 10 — The Technique Catalogue"
---

[← Back to Table of Contents](./README.md)

# Chapter 10 — The Technique Catalogue

> *"There are about forty tricks. Every fast kernel you will ever read is a handful of them, composed."*

## Overview

Every vectorization technique fits one of eight families: loop transformations, data layout, control
flow, reductions and scans, data movement, batching, hardware primitives and algorithmic reformulation.
This chapter lays all of them out as a periodic table, so you can see the whole space at once. It then
deals the eighteen that matter most as trading cards, each carrying the book's fact block: when the
technique is legal, when it pays, what it costs, where you have seen it and how it breaks. The master
table with every row is [Appendix C](./appendix_c_technique_index.md).

<div class="ptable" style="--cols:9">
<a class="el fam-loop" href="#card-sm"><span class="el-num">1</span><span class="el-sym">Sm</span><span class="el-name">Strip-mining</span></a>
<a class="el fam-loop" href="./appendix_c_technique_index.html"><span class="el-num">2</span><span class="el-sym">Lv</span><span class="el-name">Inner-loop vectorization</span></a>
<a class="el fam-loop" href="./appendix_c_technique_index.html"><span class="el-num">3</span><span class="el-sym">Ov</span><span class="el-name">Outer-loop vectorization</span></a>
<a class="el fam-loop" href="#card-ma"><span class="el-num">4</span><span class="el-sym">Ma</span><span class="el-name">Multiple accumulators</span></a>
<a class="el fam-loop" href="#card-ix"><span class="el-num">5</span><span class="el-sym">Ix</span><span class="el-name">Loop interchange</span></a>
<a class="el fam-loop" href="#card-fs"><span class="el-num">6</span><span class="el-sym">Fs</span><span class="el-name">Loop fusion</span></a>
<a class="el fam-loop" href="./appendix_c_technique_index.html"><span class="el-num">7</span><span class="el-sym">Fi</span><span class="el-name">Loop fission</span></a>
<a class="el fam-loop" href="#card-tl"><span class="el-num">8</span><span class="el-sym">Tl</span><span class="el-name">Tiling / blocking</span></a>
<a class="el fam-loop" href="./appendix_c_technique_index.html"><span class="el-num">9</span><span class="el-sym">Sk</span><span class="el-name">Skewing (wavefront)</span></a>
<a class="el fam-loop" href="./appendix_c_technique_index.html"><span class="el-num">10</span><span class="el-sym">Pe</span><span class="el-name">Peeling</span></a>
<a class="el fam-loop" href="./appendix_c_technique_index.html"><span class="el-num">11</span><span class="el-sym">Uj</span><span class="el-name">Unroll-and-jam</span></a>
<a class="el fam-data" href="#card-sa"><span class="el-num">12</span><span class="el-sym">Sa</span><span class="el-name">AoS → SoA</span></a>
<a class="el fam-data" href="./appendix_c_technique_index.html"><span class="el-num">13</span><span class="el-sym">Ab</span><span class="el-name">AoSoA / blocked layout</span></a>
<a class="el fam-data" href="#card-pd"><span class="el-num">14</span><span class="el-sym">Pd</span><span class="el-name">Padding</span></a>
<a class="el fam-data" href="./appendix_c_technique_index.html"><span class="el-num">15</span><span class="el-sym">Al</span><span class="el-name">Alignment</span></a>
<a class="el fam-data" href="./appendix_c_technique_index.html"><span class="el-num">16</span><span class="el-sym">Tr</span><span class="el-name">In-register transpose</span></a>
<a class="el fam-data" href="./appendix_c_technique_index.html"><span class="el-num">17</span><span class="el-sym">Pk</span><span class="el-name">Panel packing</span></a>
<a class="el fam-ctrl" href="#card-if"><span class="el-num">18</span><span class="el-sym">If</span><span class="el-name">If-conversion</span></a>
<a class="el fam-ctrl" href="./appendix_c_technique_index.html"><span class="el-num">19</span><span class="el-sym">Mk</span><span class="el-name">Masking / predication</span></a>
<a class="el fam-ctrl" href="#card-cp"><span class="el-num">20</span><span class="el-sym">Cp</span><span class="el-name">Compaction</span></a>
<a class="el fam-ctrl" href="./appendix_c_technique_index.html"><span class="el-num">21</span><span class="el-sym">Bn</span><span class="el-name">Binning by branch</span></a>
<a class="el fam-ctrl" href="./appendix_c_technique_index.html"><span class="el-num">22</span><span class="el-sym">Ff</span><span class="el-name">First-faulting loads</span></a>
<a class="el fam-red" href="#card-rd"><span class="el-num">23</span><span class="el-sym">Rd</span><span class="el-name">Partial-sum reduction</span></a>
<a class="el fam-red" href="./appendix_c_technique_index.html"><span class="el-num">24</span><span class="el-sym">Ht</span><span class="el-name">Horizontal tree</span></a>
<a class="el fam-red" href="#card-sc"><span class="el-num">25</span><span class="el-sym">Sc</span><span class="el-name">Parallel scan</span></a>
<a class="el fam-red" href="#card-hg"><span class="el-num">26</span><span class="el-sym">Hg</span><span class="el-name">Histogram privatization</span></a>
<a class="el fam-red" href="./appendix_c_technique_index.html"><span class="el-num">27</span><span class="el-sym">Cd</span><span class="el-name">Conflict detection</span></a>
<a class="el fam-perm" href="./appendix_c_technique_index.html"><span class="el-num">28</span><span class="el-sym">Bc</span><span class="el-name">Broadcast</span></a>
<a class="el fam-perm" href="./appendix_c_technique_index.html"><span class="el-num">29</span><span class="el-sym">Sh</span><span class="el-name">Shuffle / permute</span></a>
<a class="el fam-perm" href="#card-ga"><span class="el-num">30</span><span class="el-sym">Ga</span><span class="el-name">Gather / scatter</span></a>
<a class="el fam-perm" href="#card-lu"><span class="el-num">31</span><span class="el-sym">Lu</span><span class="el-name">Shuffle lookup table</span></a>
<a class="el fam-perm" href="./appendix_c_technique_index.html"><span class="el-num">32</span><span class="el-sym">Il</span><span class="el-name">(De)interleave</span></a>
<a class="el fam-batch" href="#card-bt"><span class="el-num">33</span><span class="el-sym">Bt</span><span class="el-name">Batching</span></a>
<a class="el fam-batch" href="./appendix_c_technique_index.html"><span class="el-num">34</span><span class="el-sym">Vm</span><span class="el-name">vmap</span></a>
<a class="el fam-batch" href="./appendix_c_technique_index.html"><span class="el-num">35</span><span class="el-sym">Xi</span><span class="el-name">Across instances</span></a>
<a class="el fam-batch" href="#card-sw"><span class="el-num">36</span><span class="el-sym">Sw</span><span class="el-name">SWAR / bit-slicing</span></a>
<a class="el fam-hw" href="./appendix_c_technique_index.html"><span class="el-num">37</span><span class="el-sym">Mm</span><span class="el-name">Matrix-engine tiles</span></a>
<a class="el fam-hw" href="./appendix_c_technique_index.html"><span class="el-num">38</span><span class="el-sym">Sy</span><span class="el-name">Systolic dataflow</span></a>
<a class="el fam-hw" href="#card-wd"><span class="el-num">39</span><span class="el-sym">Wd</span><span class="el-name">Widening dot product</span></a>
<a class="el fam-hw" href="#card-lp"><span class="el-num">40</span><span class="el-sym">Lp</span><span class="el-name">Lower precision</span></a>
<a class="el fam-algo" href="#card-rf"><span class="el-num">41</span><span class="el-sym">Rf</span><span class="el-name">Reformulate as GEMM</span></a>
<a class="el fam-algo" href="#card-on"><span class="el-num">42</span><span class="el-sym">On</span><span class="el-name">Online algorithm</span></a>
<a class="el fam-algo" href="./appendix_c_technique_index.html"><span class="el-num">43</span><span class="el-sym">Px</span><span class="el-name">Polynomial math</span></a>
<a class="el fam-algo" href="./appendix_c_technique_index.html"><span class="el-num">44</span><span class="el-sym">Sn</span><span class="el-name">Sorting network</span></a>
</div>
<div class="ptable-legend"><span class="fam-loop">loop transformation</span><span class="fam-data">data layout</span><span class="fam-ctrl">control flow</span><span class="fam-red">reduction &amp; scan</span><span class="fam-perm">data movement</span><span class="fam-batch">batching</span><span class="fam-hw">hardware primitive</span><span class="fam-algo">algorithmic</span></div>

Read the table by columns of intent. **Loop** and **layout** techniques make the independent axis
contiguous (question 1 and 3 of Chapter 1). **Control** techniques remove decisions (question 4).
**Reduction and scan** techniques break dependences that look fatal (question 2). **Data movement**
techniques pay for lanes that need data from elsewhere. **Batching** finds a new axis. **Hardware**
primitives do more per instruction. **Algorithmic** techniques change the computation into one of the
docile species. Most kernels combine three to six; FlashAttention, for instance, is Tl + Fs + On + Rd + Mm + Lp.

---

## The cards

Each card's stat bars rate **payoff** (typical gain when it applies), **risk** (chance of a wrong
result or slowdown if misapplied) and **effort** (work to apply by hand). The cost line in the top
right is the family and the card's number in the table.

<div class="tcards">
<div class="tcard fam-loop" id="card-sm"><div class="tc-in"><div class="tc-head"><span class="tc-name">Strip-mining</span><span class="tc-cost">Sm · 01</span></div><div class="tc-art">for (i = 0; i + V &lt;= n; i += V)
  y[i:i+V] = f(x[i:i+V]);
tail(i .. n);</div><div class="tc-type">Loop · applied by every vectorizer</div><div class="tc-text"><b>Legal when:</b> the loop has no dependence with 0 &lt; d &lt; V. <b>Pays when:</b> n ≫ V. <b>Costs:</b> a tail. <b>Breaks when:</b> n is tiny, or the tail is forgotten.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.8"></span><span>Risk</span><span class="tc-bar" style="--v:.15"></span><span>Effort</span><span class="tc-bar" style="--v:.1"></span></div><div class="tc-flavor">"First, cut the loop into vector-sized pieces."</div><div class="tc-foot"><span>Ch 8, 13</span><span>seen in: every compiler</span></div></div></div>
<div class="tcard fam-loop" id="card-ma"><div class="tc-in"><div class="tc-head"><span class="tc-name">Multiple Accumulators</span><span class="tc-cost">Ma · 04</span></div><div class="tc-art">acc0 += x[i+ 0:16]*y[..]
acc1 += x[i+16:32]*y[..]
acc2 += ...   acc3 += ...
s = (acc0+acc1)+(acc2+acc3)</div><div class="tc-type">Loop · compiler ("interleave count") or you</div><div class="tc-text"><b>Legal when:</b> reassociation is allowed. <b>Pays when:</b> a reduction is latency-bound. <b>Costs:</b> registers, changed rounding. Measured <b>2.5×</b> from 1 → 4 accumulators.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.7"></span><span>Risk</span><span class="tc-bar" style="--v:.3"></span><span>Effort</span><span class="tc-bar" style="--v:.2"></span></div><div class="tc-flavor">"Latency × issue rate = chains you need."</div><div class="tc-foot"><span>Ch 4, 9, 14</span><span>seen in: BLAS, clang -O3</span></div></div></div>
<div class="tcard fam-loop" id="card-ix"><div class="tc-in"><div class="tc-head"><span class="tc-name">Loop Interchange</span><span class="tc-cost">Ix · 05</span></div><div class="tc-art">for i: for j: for k:   // B[k][j]: stride N
→ for i: for k: for j: // B[k][j]: stride 1</div><div class="tc-type">Loop · compiler (sometimes) or you</div><div class="tc-text"><b>Legal when:</b> no dependence direction is reversed. <b>Pays when:</b> it makes the inner loop contiguous or removes a reduction. Measured <b>22×</b> on a 512² matmul.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.9"></span><span>Risk</span><span class="tc-bar" style="--v:.2"></span><span>Effort</span><span class="tc-bar" style="--v:.15"></span></div><div class="tc-flavor">"Same arithmetic, different order, twenty times faster."</div><div class="tc-foot"><span>Ch 7, 20</span><span>seen in: Polly, Graphite, you</span></div></div></div>
<div class="tcard fam-loop" id="card-fs"><div class="tc-in"><div class="tc-head"><span class="tc-name">Loop Fusion</span><span class="tc-cost">Fs · 06</span></div><div class="tc-art">t = a*x;  y = t+b;  z = relu(y)
→ z = relu(a*x + b)   // one pass</div><div class="tc-type">Loop · XLA, Inductor, Triton, you</div><div class="tc-text"><b>Legal when:</b> no consumer needs a producer value from a later iteration. <b>Pays when:</b> the loops are memory-bound: k passes become 1. <b>Costs:</b> register pressure, bigger kernels.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.85"></span><span>Risk</span><span class="tc-bar" style="--v:.15"></span><span>Effort</span><span class="tc-bar" style="--v:.3"></span></div><div class="tc-flavor">"The cheapest byte is the one you never write back."</div><div class="tc-foot"><span>Ch 12, 18</span><span>seen in: torch.compile, XLA</span></div></div></div>
<div class="tcard fam-loop" id="card-tl"><div class="tc-in"><div class="tc-head"><span class="tc-name">Tiling</span><span class="tc-cost">Tl · 08</span></div><div class="tc-art">for i0, j0, k0 in tiles:
  for i in i0..i0+Ti:   # fits in
    for k in k0..k0+Tk: # L1 / smem /
      for j in ...:     # registers</div><div class="tc-type">Loop · libraries, Triton, CUTLASS, you</div><div class="tc-text"><b>Legal when:</b> the reordered iterations respect dependences (always for GEMM). <b>Pays when:</b> data is reused: raises arithmetic intensity. <b>Costs:</b> complexity, tile-size tuning.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.95"></span><span>Risk</span><span class="tc-bar" style="--v:.2"></span><span>Effort</span><span class="tc-bar" style="--v:.7"></span></div><div class="tc-flavor">"Climb the roofline's slope by reusing what you loaded."</div><div class="tc-foot"><span>Ch 4, 16, 18</span><span>seen in: every GEMM</span></div></div></div>
<div class="tcard fam-data" id="card-sa"><div class="tc-in"><div class="tc-head"><span class="tc-name">AoS → SoA</span><span class="tc-cost">Sa · 12</span></div><div class="tc-art">struct {x,y,z,vx,..} p[n]
→ struct {x[n], y[n], z[n], ..}</div><div class="tc-type">Layout · you (compilers will not change your structs)</div><div class="tc-text"><b>Legal when:</b> always (it is a re-layout). <b>Pays when:</b> bulk passes over a few fields. Measured <b>9.5×</b> over vectorized AoS with AVX2. <b>Breaks:</b> random whole-record access.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.8"></span><span>Risk</span><span class="tc-bar" style="--v:.1"></span><span>Effort</span><span class="tc-bar" style="--v:.6"></span></div><div class="tc-flavor">"Store the spreadsheet by column."</div><div class="tc-foot"><span>Ch 7</span><span>seen in: Arrow, ECS engines</span></div></div></div>
<div class="tcard fam-data" id="card-pd"><div class="tc-in"><div class="tc-head"><span class="tc-name">Padding</span><span class="tc-cost">Pd · 14</span></div><div class="tc-art">d_model = 4096      ✓
vocab   = 50257 → 50304 (×64)
tile    = 32 → 33 cols (banks)</div><div class="tc-type">Layout · you, frameworks</div><div class="tc-text"><b>Legal when:</b> padded values cannot affect results (zeros in sums, −∞ in max). <b>Pays when:</b> it removes tails or bank conflicts. <b>Costs:</b> wasted memory and FLOPs.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.5"></span><span>Risk</span><span class="tc-bar" style="--v:.35"></span><span>Effort</span><span class="tc-bar" style="--v:.15"></span></div><div class="tc-flavor">"Make n a multiple of the hardware's favourite number."</div><div class="tc-foot"><span>Ch 8, 15, 17</span><span>seen in: nanoGPT vocab</span></div></div></div>
<div class="tcard fam-ctrl" id="card-if"><div class="tc-in"><div class="tc-head"><span class="tc-name">If-Conversion</span><span class="tc-cost">If · 18</span></div><div class="tc-art">if (c) y=a; else y=b;
→ y = select(c, a, b)  // both computed</div><div class="tc-type">Control · compiler, np.where, you</div><div class="tc-text"><b>Legal when:</b> computing both sides has no side effects (faults, FP flags, errno). <b>Pays when:</b> sides are cheap, branch unpredictable. Measured <b>15.6×</b> (leaky ReLU, AVX-512).</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.75"></span><span>Risk</span><span class="tc-bar" style="--v:.3"></span><span>Effort</span><span class="tc-bar" style="--v:.1"></span></div><div class="tc-flavor">"Take both roads. Keep one."</div><div class="tc-foot"><span>Ch 8</span><span>seen in: every vectorizer</span></div></div></div>
<div class="tcard fam-ctrl" id="card-cp"><div class="tc-in"><div class="tc-head"><span class="tc-name">Compaction</span><span class="tc-cost">Cp · 20</span></div><div class="tc-art">keep = score &gt; t
pos  = exclusive_scan(keep)
out[pos[keep]] = x[keep]</div><div class="tc-type">Control · vcompress, x[mask], you</div><div class="tc-text"><b>Legal when:</b> always. <b>Pays when:</b> later work is heavy and few elements survive. <b>Costs:</b> a scan or compress pass. <b>Breaks:</b> when nearly everything survives.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.6"></span><span>Risk</span><span class="tc-bar" style="--v:.1"></span><span>Effort</span><span class="tc-bar" style="--v:.35"></span></div><div class="tc-flavor">"Do not mask what you can delete."</div><div class="tc-foot"><span>Ch 2, 8</span><span>seen in: NMS, databases</span></div></div></div>
<div class="tcard fam-red" id="card-rd"><div class="tc-in"><div class="tc-head"><span class="tc-name">Partial-Sum Reduction</span><span class="tc-cost">Rd · 23</span></div><div class="tc-art">acc[0..V] += x[i:i+V]   // per lane
s = tree_sum(acc)        // once</div><div class="tc-type">Reduction · compiler with permission, you</div><div class="tc-text"><b>Legal when:</b> the operator is associative (ints, min/max; floats with permission). <b>Pays when:</b> always for long n. <b>Costs:</b> different rounding (usually better).</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.85"></span><span>Risk</span><span class="tc-bar" style="--v:.35"></span><span>Effort</span><span class="tc-bar" style="--v:.15"></span></div><div class="tc-flavor">"128 ulps off in order; 11 at sixteen lanes."</div><div class="tc-foot"><span>Ch 3, 6</span><span>seen in: np.sum, BLAS dot</span></div></div></div>
<div class="tcard fam-red" id="card-sc"><div class="tc-in"><div class="tc-head"><span class="tc-name">Parallel Scan</span><span class="tc-cost">Sc · 25</span></div><div class="tc-art">y[i] = a[i]*y[i-1] + b[i]
(a,b)∘(a',b') = (a·a', a·b' + b)
→ log₂ n combine steps</div><div class="tc-type">Reduction · CUB, jax.lax.associative_scan, you</div><div class="tc-text"><b>Legal when:</b> the step is an associative operator, e.g. affine maps. <b>Pays when:</b> n is long and parallel width large. <b>Costs:</b> 2–log n × the work.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.7"></span><span>Risk</span><span class="tc-bar" style="--v:.4"></span><span>Effort</span><span class="tc-bar" style="--v:.75"></span></div><div class="tc-flavor">"A recurrence is a reduction you have not recognised yet."</div><div class="tc-foot"><span>Ch 6, 10</span><span>seen in: Mamba, CUB</span></div></div></div>
<div class="tcard fam-red" id="card-hg"><div class="tc-in"><div class="tc-head"><span class="tc-name">Histogram Privatization</span><span class="tc-cost">Hg · 26</span></div><div class="tc-art">h[lane][idx] += 1   // V private copies
h = sum over lanes  // merge once</div><div class="tc-type">Reduction · you, GPU shared-memory atomics</div><div class="tc-text"><b>Legal when:</b> updates commute (counts, sums). <b>Pays when:</b> bins fit in cache. <b>Costs:</b> V× memory for bins, a merge. Alternative: conflict detection (<code>vpconflictd</code>).</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.6"></span><span>Risk</span><span class="tc-bar" style="--v:.4"></span><span>Effort</span><span class="tc-bar" style="--v:.5"></span></div><div class="tc-flavor">"Give every lane its own scoreboard."</div><div class="tc-foot"><span>Ch 2, 11</span><span>seen in: bincount, GNNs</span></div></div></div>
<div class="tcard fam-perm" id="card-ga"><div class="tc-in"><div class="tc-head"><span class="tc-name">Gather / Scatter</span><span class="tc-cost">Ga · 30</span></div><div class="tc-art">v = gather(table, idx[i:i+V])
scatter(out, idx2[i:i+V], v)</div><div class="tc-type">Data movement · hardware (AVX2+, SVE, RVV, GPUs)</div><div class="tc-text"><b>Legal when:</b> gathers always; scatters only without conflicts. <b>Pays when:</b> it lets heavy arithmetic around it vectorize. Measured: no faster than scalar for strided reads.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.35"></span><span>Risk</span><span class="tc-bar" style="--v:.4"></span><span>Effort</span><span class="tc-bar" style="--v:.2"></span></div><div class="tc-flavor">"Legal everywhere. Fast nowhere."</div><div class="tc-foot"><span>Ch 3, 7</span><span>seen in: embeddings, KV pages</span></div></div></div>
<div class="tcard fam-perm" id="card-lu"><div class="tc-in"><div class="tc-head"><span class="tc-name">Shuffle Lookup Table</span><span class="tc-cost">Lu · 31</span></div><div class="tc-art">lut = [-8,-7,...,7]   // 16 entries
w8  = pshufb(lut, nibbles)  // 64 at once</div><div class="tc-type">Data movement · you, llama.cpp, simdjson</div><div class="tc-text"><b>Legal when:</b> the table has ≤ 16 entries per 128 bits (or 64 with <code>vpermb</code>). <b>Pays when:</b> a gather or branch would otherwise be needed per element.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.75"></span><span>Risk</span><span class="tc-bar" style="--v:.2"></span><span>Effort</span><span class="tc-bar" style="--v:.6"></span></div><div class="tc-flavor">"The fastest table is the one that lives in a register."</div><div class="tc-foot"><span>Ch 3, 18</span><span>seen in: 4-bit dequant, UTF-8</span></div></div></div>
<div class="tcard fam-batch" id="card-bt"><div class="tc-in"><div class="tc-head"><span class="tc-name">Batching</span><span class="tc-cost">Bt · 33</span></div><div class="tc-art">for req in requests: y = W @ x   # GEMV
→ Y = W @ X                       # GEMM</div><div class="tc-type">Batching · serving systems, data loaders, you</div><div class="tc-text"><b>Legal when:</b> instances are independent. <b>Pays when:</b> a shared operand (weights) dominates traffic: each read is reused B times. <b>Costs:</b> latency, padding.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.95"></span><span>Risk</span><span class="tc-bar" style="--v:.15"></span><span>Effort</span><span class="tc-bar" style="--v:.4"></span></div><div class="tc-flavor">"The batch is a lane the size of a datacenter."</div><div class="tc-foot"><span>Ch 12, 19</span><span>seen in: vLLM, every trainer</span></div></div></div>
<div class="tcard fam-batch" id="card-sw"><div class="tc-in"><div class="tc-head"><span class="tc-name">SWAR / Bit-Slicing</span><span class="tc-cost">Sw · 36</span></div><div class="tc-art">uint64 word = 64 test vectors
out = (a &amp; b) | c   // 64 gates sim'd</div><div class="tc-type">Batching · you</div><div class="tc-text"><b>Legal when:</b> the per-lane ops are bitwise or carry-free. <b>Pays when:</b> elements are 1–8 bits: logic simulation, bitsets, Bloom filters, binary nets. <b>Costs:</b> transposing data into bit planes.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.7"></span><span>Risk</span><span class="tc-bar" style="--v:.3"></span><span>Effort</span><span class="tc-bar" style="--v:.6"></span></div><div class="tc-flavor">"Every integer register is already a vector of bits."</div><div class="tc-foot"><span>Ch 2, 17</span><span>seen in: EDA simulators</span></div></div></div>
<div class="tcard fam-hw" id="card-wd"><div class="tc-in"><div class="tc-head"><span class="tc-name">Widening Dot Product</span><span class="tc-cost">Wd · 39</span></div><div class="tc-art">acc_i32 += Σ₄ a_u8 · b_s8
vpdpbusd / sdot / dp4a</div><div class="tc-type">Hardware · intrinsics, quantized kernels</div><div class="tc-text"><b>Legal when:</b> int32 accumulators cannot overflow over the reduction length. <b>Pays when:</b> int8/int4 inference. <b>Breaks:</b> u8×s8 sign conventions (VNNI) need an offset trick.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.8"></span><span>Risk</span><span class="tc-bar" style="--v:.35"></span><span>Effort</span><span class="tc-bar" style="--v:.6"></span></div><div class="tc-flavor">"Four products, one lane, no overflow."</div><div class="tc-foot"><span>Ch 3, 18</span><span>seen in: FBGEMM, oneDNN</span></div></div></div>
<div class="tcard fam-hw" id="card-lp"><div class="tc-in"><div class="tc-head"><span class="tc-name">Lower Precision</span><span class="tc-cost">Lp · 40</span></div><div class="tc-art">fp32 → bf16 → fp8 → fp4
16  →  32  →  64 → 128 lanes / 512 b</div><div class="tc-type">Hardware · quantization, mixed precision</div><div class="tc-text"><b>Legal when:</b> accuracy survives (see the quantization book). <b>Pays when:</b> compute- or memory-bound alike: more lanes <i>and</i> fewer bytes. <b>Costs:</b> scaling, conversion, accuracy.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.9"></span><span>Risk</span><span class="tc-bar" style="--v:.55"></span><span>Effort</span><span class="tc-bar" style="--v:.5"></span></div><div class="tc-flavor">"Halve the bits, double the lanes."</div><div class="tc-foot"><span>Ch 3, 18</span><span>seen in: every LLM server</span></div></div></div>
<div class="tcard fam-algo" id="card-rf"><div class="tc-in"><div class="tc-head"><span class="tc-name">Reformulate as GEMM</span><span class="tc-cost">Rf · 41</span></div><div class="tc-art">‖x−y‖² = ‖x‖² + ‖y‖² − 2·x·y
conv2d = im2col + GEMM</div><div class="tc-type">Algorithmic · you</div><div class="tc-text"><b>Legal when:</b> mathematically equivalent; watch cancellation (clamp ≥ 0). <b>Pays when:</b> it moves the work onto BLAS / tensor cores. Measured: fastest NumPy rung, <b>488×</b> over loops.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.95"></span><span>Risk</span><span class="tc-bar" style="--v:.4"></span><span>Effort</span><span class="tc-bar" style="--v:.4"></span></div><div class="tc-flavor">"If you can't vectorize it, multiply a matrix instead."</div><div class="tc-foot"><span>Ch 2, 20</span><span>seen in: FAISS, cdist</span></div></div></div>
<div class="tcard fam-algo" id="card-on"><div class="tc-in"><div class="tc-head"><span class="tc-name">Online Algorithm</span><span class="tc-cost">On · 42</span></div><div class="tc-art">m' = max(m, x)
l' = l·e^(m−m') + e^(x−m')
// softmax stats in ONE pass</div><div class="tc-type">Algorithmic · FlashAttention, you</div><div class="tc-text"><b>Legal when:</b> the state can be rescaled exactly (softmax, Welford variance). <b>Pays when:</b> memory-bound multi-pass code: 3 passes → 1. <b>Costs:</b> extra exp/multiply per step.</div><div class="tc-stats"><span>Payoff</span><span class="tc-bar" style="--v:.85"></span><span>Risk</span><span class="tc-bar" style="--v:.3"></span><span>Effort</span><span class="tc-bar" style="--v:.55"></span></div><div class="tc-flavor">"Carry the statistics, not the data."</div><div class="tc-foot"><span>Ch 18</span><span>seen in: FlashAttention</span></div></div></div>
</div>

---

## Three techniques worth a derivation

### Parallel scan: vectorizing the unvectorizable recurrence

A first-order linear recurrence $y_i = a_i\, y_{i-1} + b_i$ has a distance-1 flow dependence, so
Chapter 6 forbids vectorizing it directly. Treat each step as a function $f_i(y) = a_i y + b_i$.
Composition of two such affine maps is again affine:

$$f_j \circ f_i\,(y) = a_j(a_i y + b_i) + b_j = (a_j a_i)\,y + (a_j b_i + b_j)$$

so the pair operator $(a_j, b_j) \bullet (a_i, b_i) = (a_j a_i,\; a_j b_i + b_j)$ is **associative**,
and any associative operator can be scanned in $\lceil \log_2 n \rceil$ parallel steps (Hillis–Steele)
or with $O(n)$ total work in $2 \log_2 n$ steps (Blelloch).

**Numerical check.** $a = (2, 3)$, $b = (1, 1)$, $y_0 = 0$: sequentially $y_1 = 2\cdot0+1 = 1$,
$y_2 = 3\cdot1 + 1 = 4$. Combined: $(3, 1) \bullet (2, 1) = (6, 3\cdot1 + 1) = (6, 4)$, and
$6 \cdot y_0 + 4 = 4$. ✓

```python
import numpy as np

def linear_scan(a, b):
    """y[i] = a[i]*y[i-1] + b[i], y[-1] = 0, in log2(n) vectorized steps (Hillis–Steele)."""
    A, B = a.astype(np.float64).copy(), b.astype(np.float64).copy()
    shift = 1
    while shift < len(a):
        A_prev = np.concatenate([np.ones(shift), A[:-shift]])   # identity map for the first lanes
        B_prev = np.concatenate([np.zeros(shift), B[:-shift]])
        A, B = A * A_prev, A * B_prev + B                         # (A,B) • (A_prev,B_prev)
        shift *= 2
    return B
```

Every line inside the `while` is a whole-array operation: the recurrence has become $\log_2 n$ docile
maps. This is the structure behind `jax.lax.associative_scan`, CUB's device scan, and the "selective
scan" of state-space models such as Mamba, which run long recurrences on GPUs this way. The price is
extra work: Hillis–Steele does $n \log_2 n$ combines instead of $n$.

### Histogram: two ways around write conflicts

`h[idx[i]] += 1` vectorized naively loses updates when two lanes share an index. Two fixes:

1. **Privatization.** Each lane (or thread) owns a private histogram; merge at the end. Exact, needs $V$
   copies of the bins.
2. **Conflict detection.** AVX-512 CD's `vpconflictd` reports, for each lane, which earlier lanes hold the
   same index, so a loop can combine duplicates before the scatter.

In NumPy the trap is concrete: `h[idx] += 1` performs a gather, an add and a scatter, and **silently
counts duplicates once**. `np.add.at(h, idx, 1)` or `np.bincount` give the right answer; the companion test
`test_fancy_index_increment_drops_duplicates` pins the behaviour down.

### Outer-loop vectorization: when the inner loop is the wrong axis

Pairwise distances have an inner loop over the feature dimension $d$, which is a reduction (needs
reassociation) and short (64). Vectorizing the **middle** loop over points $j$ instead turns the inner
work into a map with $V$ independent accumulators. The companion `pdist_outer` kernel does exactly this.
It ran at 0.0505 ns per pair-dimension versus 0.502 for the naive loop under the same flags, with no
`-ffast-math` needed, because no float reduction is reordered.

<div class="callout ponder"><span class="callout-title">Ponder</span>
<p>The periodic table's columns suggest combinations that are rarely tried. Online algorithms (On) plus
scans (Sc) gave linear-time attention variants. Batching (Bt) plus binning (Bn) gave continuous batching
with length bucketing. Which two elements, far apart in the table, have <em>not</em> been combined in your
area yet?</p>
</div>

---

## Key Takeaways

- About 44 techniques in 8 families cover practical vectorization; real kernels compose 3–6 of them.
- Loop and layout techniques make the right axis contiguous; control techniques remove decisions; reduction and scan techniques break apparent dependences.
- Measured on one core: interchange 22×, AoS→SoA 9.5× (AVX2), if-conversion with masks 15.6×, 4 accumulators 2.5×, reformulation as GEMM 488× over Python loops.
- Any recurrence whose step is an associative operator, including first-order linear ones, becomes $\log_2 n$ vectorized steps via a scan, at extra work.
- Write conflicts in scatter-adds need privatization or conflict detection; NumPy's `h[idx] += 1` silently drops duplicates.
- Choosing a different axis (outer-loop vectorization) can remove the need for floating-point reassociation entirely.

## Check Yourself

<details class="check"><summary>Which techniques from the table does FlashAttention combine?</summary>
Tiling (Tl) of Q, K, V into SRAM-sized blocks; fusion (Fs) of QKᵀ, softmax and PV into one kernel; an online softmax (On) carrying max and sum across blocks; partial-sum reductions (Rd); matrix-engine tiles (Mm) for the two GEMMs; and lower precision (Lp, fp16/bf16 inputs with fp32 accumulation).
</details>

<details class="check"><summary>Why does the parallel scan cost more arithmetic than the sequential loop, and why is that acceptable?</summary>
Hillis–Steele performs n·log₂n combines and Blelloch about 2n, versus n for the loop. It is acceptable when the parallel or vector width is large and the sequential loop is latency-bound: on a GPU, log₂n steps over thousands of lanes finish far sooner than n dependent steps on one lane.
</details>

<details class="check"><summary>When is padding illegal?</summary>
When the padded elements can change the result: padding a max-reduction with 0 when all values are negative, padding a softmax with 0 logits (they get probability mass; use −∞), or padding an average without adjusting the count.
</details>

## References

| Source | Authors / Org, Year | Link | What this chapter takes from it |
|--------|--------------------|------|--------------------------------|
| Optimizing Compilers for Modern Architectures | Allen & Kennedy, 2001 | Morgan Kaufmann | Interchange, fusion, fission, skewing, strip-mining |
| Prefix Sums and Their Applications | Blelloch, CMU-CS-90-190, 1990 | cs.cmu.edu | Work-efficient scan; linear recurrences as scans |
| Data Parallel Algorithms | Hillis & Steele, CACM 1986 | doi:10.1145/7902.7903 | Log-step scan |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Gu & Dao, 2023 | arXiv:2312.00752 | Hardware-aware parallel scan for SSMs |
| FlashAttention | Dao et al., NeurIPS 2022 | arXiv:2205.14135 | Tiling + fusion + online softmax |
| Online normalizer calculation for softmax | Milakov & Gimelshein, 2018 | arXiv:1805.02867 | The online softmax recurrence |
| Exploiting Superword Level Parallelism with Multimedia Instruction Sets | Larsen & Amarasinghe, PLDI 2000 | doi:10.1145/349299.349320 | SLP vectorization (basis of Lv vs straight-line packing) |
| Intel AVX-512 Conflict Detection (`vpconflictd`) | Intel SDM Vol. 2 | intel.com/sdm | Conflict-detecting scatters |
| This book's code and measurements | `code/vzlab`, `results.json` | [repo](https://github.com/karam-nus/vectorization) | Numbers on the cards; histogram trap test |

---

**Next:** [Chapter 11 — Array Programming: NumPy, PyTorch & einsum](./11_array_programming.md) — the first layer where you apply these techniques without writing a loop.
