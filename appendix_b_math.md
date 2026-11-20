---
title: "Appendix B — Math Reference"
---

[← Back to Table of Contents](./README.md)

# Appendix B — Math Reference

> *Every formula in the book, with the numerical check that goes with it.*

## Lanes and tails

$$V = \frac{w}{e} \qquad\text{e.g. } \frac{512}{16} = 32 \text{ bf16 lanes}$$

Instruction count with a scalar epilogue and with a masked tail:

$$N_{\text{epi}} = \left\lfloor \frac{n}{V} \right\rfloor + (n \bmod V), \qquad N_{\text{mask}} = \left\lceil \frac{n}{V} \right\rceil$$

Check: $n = 100$, $V = 16$: $N_{\text{epi}} = 6 + 4 = 10$, $N_{\text{mask}} = 7$; utilisation with masking $100/112 = 89\%$. ([Ch 8](./08_control_flow.md))

## Legality

For `a[i+W] = f(a[i+R])`, dependence distance $d = W - R$; chunked vector execution is legal iff
$d \le 0$ or $d \ge V$. GCD test for `a[c₁i + k₁]` written and `a[c₂j + k₂]` read: a dependence is
possible only if $\gcd(c_1, c_2) \mid (k_2 - k_1)$. Check: $c_1 = c_2 = 2$, $k_2 - k_1 = 1$: $2 \nmid 1$, independent. ([Ch 6](./06_dependences.md))

## Divergence

Probability that a chunk of $V$ independent lanes needs both sides of a branch taken with probability $p$:

$$P(\text{both}) = 1 - p^V - (1-p)^V$$

Check: $p = 0.9$, $V = 8$: $1 - 0.430 - 10^{-8} \approx 0.57$. ([Ch 8](./08_control_flow.md))

## Roofline and Amdahl

$$P(I) = \min(P_{\text{peak}},\, I B), \qquad I^\ast = \frac{P_{\text{peak}}}{B}, \qquad S_{\max}(I) = \frac{\min(P_{\text{vec}}, IB)}{\min(P_{\text{scalar}}, IB)}$$

$$S_{\text{Amdahl}} = \frac{1}{(1-f) + f/s} \le \frac{1}{1-f}$$

Checks (measured core): DRAM ridge for scalar $12.9 / 21 = 0.61$ FLOP/B; Amdahl $f = 0.7$, $s = 9$:
$1 / (0.3 + 0.078) = 2.65$. ([Ch 9](./09_profitability.md))

## Latency hiding

Independent chains needed to saturate a pipelined unit (Little's law):

$$\text{chains} = L \times (\text{issues per cycle})$$

Check: FMA latency 4, two ports: 8 accumulators. ([Ch 9](./09_profitability.md), [Ch 14](./14_explicit_simd.md))

## Memory

Cache lines per useful element at stride $s$ (elements of $e$ bytes, 64-byte lines):
$\min(1, s e / 64)$. Check: $s = 16$, $e = 4$ B: 1 line per element. GPU warp sectors (32 B) for thread $t$
reading element $t \cdot s$: 4 at $s = 1$, 32 at $s \ge 8$ (fp32). Shared-memory bank of fp32 element at
address $a$: $(a/4) \bmod 32$. ([Ch 7](./07_memory_layout.md), [Ch 15](./15_gpus_simt.md))

## Reductions and floating point

Error bounds for summing $n$ terms in precision $\varepsilon$ (unit roundoff $2^{-24}$ for float32), $S_{\max}$
the largest partial-sum magnitude: sequential $\lesssim (n-1)\,\varepsilon\, S_{\max}$ worst case; pairwise
$\lesssim \lceil \log_2 n \rceil\, \varepsilon\, S_{\max}$; Kahan $\lesssim 2\varepsilon\,\sum |x_i|$ (Higham 2002).
Measured: $10^6$ values of $\mathcal{N}(10^4, 10^3)$: sequential 128 ulps, pairwise 0.1. ([Ch 6](./06_dependences.md))

## Parallel scan

Linear recurrence $y_i = a_i y_{i-1} + b_i$ composes as $(a_j, b_j) \bullet (a_i, b_i) = (a_j a_i,\, a_j b_i + b_j)$,
associative; Hillis–Steele takes $\lceil \log_2 n \rceil$ steps and $n \log_2 n$ work; Blelloch $2\log_2 n$ steps and $O(n)$ work.
Check: $(3,1) \bullet (2,1) = (6, 4)$, matching $y_2 = 3(2 \cdot 0 + 1) + 1 = 4$. ([Ch 10](./10_technique_catalogue.md))

## Online softmax and variance

$$m' = \max(m, x), \quad \ell' = \ell\, e^{m - m'} + e^{x - m'}; \qquad (m_1,\ell_1) \oplus (m_2, \ell_2) = (m,\; \ell_1 e^{m_1 - m} + \ell_2 e^{m_2 - m}),\; m = \max(m_1, m_2)$$

Welford/Chan merge of $(n_a, \mu_a, M_a)$ and $(n_b, \mu_b, M_b)$ with $\delta = \mu_b - \mu_a$:
$n = n_a + n_b$, $\mu = \mu_a + \delta n_b / n$, $M = M_a + M_b + \delta^2 n_a n_b / n$, variance $M/n$. ([Ch 18](./18_ml_kernels.md))

## Vectorized exp

$x = k \ln 2 + r$, $k = \operatorname{round}(x/\ln 2)$, $e^x = 2^k p(r)$. Check: $x = 3$: $k = 4$, $r = 0.2274$, $16 \cdot 1.2553 = 20.085$. ([Ch 18](./18_ml_kernels.md))

## GEMM microkernel

Register budget $M_R N_R / V + N_R / V + 1 \le R_{\text{vec}}$; FMA:load ratio $\dfrac{M_R N_R/V}{M_R + N_R/V}$.
Check: $6 \times 32$, $V = 16$: 15 registers, ratio 1.5. ([Ch 18](./18_ml_kernels.md))

## Matrix engines

Operand reuse of an $N \times N \times N$ tile: $N^3 / 2N^2 = N/2$. Systolic utilisation for one isolated
tile: $K / (K + R + C - 2)$. Check: $R = C = K = 256$: 33 %. ([Ch 16](./16_matrix_engines.md))

## GPUs

Wave efficiency $G / (S \lceil G/S \rceil)$. Check: 109 blocks on 108 SMs: 50.5 %. ([Ch 15](./15_gpus_simt.md))

## Batching

Linear layer with shared weights: $I \approx B$ FLOP/byte (tokens per weight read). Padding efficiency
$\eta = \sum_b \ell_b / (B \max_b \ell_b)$. Check: lengths 11, 4, 16, 7, 9, 5: $52/96 = 54\%$. ([Ch 19](./19_batching_systems.md))

## Quantized dot products (VNNI offset)

With activations $a$ shifted to unsigned $a + 128$: $a \cdot w = (a + 128)\cdot w - 128 \sum_k w_k$.
Check: $a = (-3, 5)$, $w = (2, 1)$: $383 - 384 = -1$. ([Ch 18](./18_ml_kernels.md))

---

**Next:** [Appendix C — Technique Index](./appendix_c_technique_index.md) — every technique in one table.
