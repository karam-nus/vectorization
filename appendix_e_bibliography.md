---
title: "Appendix E — Bibliography"
---

[← Back to Table of Contents](./README.md)

# Appendix E — Bibliography

> *The reading list by theme, with a suggested order. Each chapter's References table says exactly what it took from each source.*

## Suggested order

1. Williams, Waterman & Patterson — *Roofline* (2009): the single most useful mental model.
2. Hennessy & Patterson — *Computer Architecture: A Quantitative Approach*, ch. 4: vector, SIMD and GPU in one place.
3. LLVM — *Auto-Vectorization in LLVM*: what your compiler is doing.
4. Slotin — *Algorithms for Modern Hardware*: SIMD algorithms with measured code.
5. Dao et al. — *FlashAttention* (2022): every technique in this book composed in one kernel.
6. Jouppi et al. — *TPU* (2017) and Kung — *Why Systolic Architectures?* (1982): the case for matrix engines.

---

## Foundations and history

| Work | Authors, year | Link |
|---|---|---|
| Very High-Speed Computing Systems | Flynn, Proc. IEEE 1966 | doi:10.1109/PROC.1966.5273 |
| The Parallel Execution of DO Loops | Lamport, CACM 1974 | doi:10.1145/360827.360844 |
| The CRAY-1 Computer System | Russell, CACM 1978 | doi:10.1145/359327.359336 |
| Data Parallel Algorithms | Hillis & Steele, CACM 1986 | doi:10.1145/7902.7903 |
| MMX Technology Extension to the Intel Architecture | Peleg & Weiser, IEEE Micro 1996 | doi:10.1109/40.526921 |
| NVIDIA Tesla: A Unified Graphics and Computing Architecture | Lindholm et al., IEEE Micro 2008 | doi:10.1109/MM.2008.31 |
| The ARM Scalable Vector Extension | Stephens et al., IEEE Micro 2017 | doi:10.1109/MM.2017.35 |
| Computer Architecture: A Quantitative Approach, 6th ed. | Hennessy & Patterson, 2017 | Morgan Kaufmann |
| A New Golden Age for Computer Architecture | Hennessy & Patterson, CACM 2019 | doi:10.1145/3282307 |

## Performance models and energy

| Work | Authors, year | Link |
|---|---|---|
| Validity of the single processor approach… | Amdahl, AFIPS 1967 | doi:10.1145/1465482.1465560 |
| Reevaluating Amdahl's Law | Gustafson, CACM 1988 | doi:10.1145/42411.42415 |
| Roofline: An Insightful Visual Performance Model | Williams, Waterman & Patterson, CACM 2009 | doi:10.1145/1498765.1498785 |
| Efficient Embedded Computing | Dally et al., IEEE Computer 2008 | doi:10.1109/MC.2008.224 |
| Computing's Energy Problem (and what we can do about it) | Horowitz, ISSCC 2014 | doi:10.1109/ISSCC.2014.6757323 |
| There's plenty of room at the Top | Leiserson et al., Science 2020 | doi:10.1126/science.aam9744 |
| What Every Programmer Should Know About Memory | Drepper, 2007 | akkadia.org/drepper/cpumemory.pdf |

## Compilers and dependence analysis

| Work | Authors, year | Link |
|---|---|---|
| Conversion of control dependence to data dependence | Allen, Kennedy, Porterfield & Warren, POPL 1983 | doi:10.1145/567067.567085 |
| Exploiting Superword Level Parallelism with Multimedia Instruction Sets | Larsen & Amarasinghe, PLDI 2000 | doi:10.1145/349299.349320 |
| Optimizing Compilers for Modern Architectures | Allen & Kennedy, 2001 | Morgan Kaufmann |
| Halide | Ragan-Kelley et al., PLDI 2013 | doi:10.1145/2491956.2462176 |
| Auto-Vectorization in LLVM | LLVM Project | llvm.org/docs/Vectorizers.html |
| Auto-vectorization in GCC | GCC Project | gcc.gnu.org |
| OpenMP 5.2 — SIMD directives | OpenMP ARB, 2021 | openmp.org |

## Numerics

| Work | Authors, year | Link |
|---|---|---|
| Updating Formulae and a Pairwise Algorithm for Computing Sample Variances | Chan, Golub & LeVeque, 1979 | Stanford tech report |
| What Every Computer Scientist Should Know About Floating-Point Arithmetic | Goldberg, ACM Computing Surveys 1991 | doi:10.1145/103162.103163 |
| Accuracy and Stability of Numerical Algorithms, 2nd ed. | Higham, SIAM 2002 | — |
| SLEEF: A Portable Vectorized Library of C Standard Mathematical Functions | Shibata & Petrogalli, 2020 | arXiv:2001.09258 |

## Explicit SIMD and instruction costs

| Work | Authors, year | Link |
|---|---|---|
| Intel 64 and IA-32 Software Developer's Manual; Intrinsics Guide; Optimization Manual | Intel | intel.com |
| Arm ACLE for SVE/SME; Arm ARM | Arm | developer.arm.com |
| RISC-V V Vector Extension 1.0; RVV intrinsics | RISC-V International, 2021 | github.com/riscv |
| ispc: A SPMD Compiler for High-Performance CPU Programming | Pharr & Mark, InPar 2012 | doi:10.1109/InPar.2012.6339601 |
| uops.info | Abel & Reineke, ASPLOS 2019 | uops.info |
| uiCA | Abel & Reineke, ICS 2022 | uica.uops.info |
| Software optimization resources | Fog | agner.org/optimize |
| std::simd (P1928) | Kretz, WG21 | open-std.org |
| Highway | Google | github.com/google/highway |

## Array programming and frameworks

| Work | Authors, year | Link |
|---|---|---|
| The NumPy array | van der Walt, Colbert & Varoquaux, CiSE 2011 | doi:10.1109/MCSE.2011.37 |
| Array programming with NumPy | Harris et al., Nature 2020 | doi:10.1038/s41586-020-2649-2 |
| opt_einsum | Smith & Gray, JOSS 2018 | doi:10.21105/joss.00753 |
| Compiling machine learning programs via high-level tracing (JAX) | Frostig, Johnson & Leary, SysML 2018 | mlsys.org |
| PyTorch 2 (TorchDynamo, TorchInductor) | Ansel et al., ASPLOS 2024 | doi:10.1145/3620665.3640366 |
| Triton | Tillet, Kung & Cox, MAPL 2019 | doi:10.1145/3315508.3329973 |

## GPUs and matrix engines

| Work | Authors, year | Link |
|---|---|---|
| Why Systolic Architectures? | Kung, IEEE Computer 1982 | doi:10.1109/MC.1982.1653825 |
| Eyeriss | Chen, Emer & Sze, ISCA 2016 | doi:10.1109/ISCA.2016.40 |
| In-Datacenter Performance Analysis of a TPU | Jouppi et al., ISCA 2017 | arXiv:1704.04760 |
| A Domain-Specific Supercomputer for Training DNNs | Jouppi et al., CACM 2020 | doi:10.1145/3360307 |
| Ten Lessons From Three Generations Shaped Google's TPUv4i | Jouppi et al., ISCA 2021 | doi:10.1109/ISCA52012.2021.00010 |
| NVIDIA V100, A100, H100 whitepapers; Blackwell brief; PTX ISA | NVIDIA, 2017–2025 | nvidia.com, docs.nvidia.com |
| CUDA C++ Programming Guide and Best Practices Guide | NVIDIA | docs.nvidia.com/cuda |
| CUTLASS tutorials (WGMMA, Blackwell UMMA/TMEM) | Colfax Research | research.colfax-intl.com |
| Hello SME! | Remke & Breuer, 2024 | arXiv:2409.18779 |
| Megakernels Considered Harmful | Laine, Karras & Aila, HPG 2013 | doi:10.1145/2492045.2492060 |

## ML kernels and systems

| Work | Authors, year | Link |
|---|---|---|
| Anatomy of High-Performance Matrix Multiplication | Goto & van de Geijn, ACM TOMS 2008 | doi:10.1145/1356052.1356053 |
| BLIS | Van Zee & van de Geijn, ACM TOMS 2015 | doi:10.1145/2764454 |
| Online normalizer calculation for softmax | Milakov & Gimelshein, 2018 | arXiv:1805.02867 |
| FlashAttention; FlashAttention-2; FlashAttention-3 | Dao et al. 2022; Dao 2023; Shah et al. 2024 | arXiv:2205.14135; 2307.08691; 2407.08608 |
| Mamba | Gu & Dao, 2023 | arXiv:2312.00752 |
| Prefix Sums and Their Applications | Blelloch, 1990 | CMU-CS-90-190 |
| Orca | Yu et al., OSDI 2022 | usenix.org |
| PagedAttention (vLLM) | Kwon et al., SOSP 2023 | arXiv:2309.06180 |
| Sarathi-Serve; DistServe | Agrawal et al. 2024; Zhong et al. 2024 | arXiv:2403.02310; 2401.09670 |
| Speculative decoding; speculative sampling | Leviathan et al. 2023; Chen et al. 2023 | arXiv:2211.17192; 2302.01318 |
| Deep Learning with Differential Privacy | Abadi et al., CCS 2016 | arXiv:1607.00133 |
| FAISS | Johnson, Douze & Jégou, 2019 | arXiv:1702.08734 |
| OCP Microscaling (MX) Formats v1.0 | OCP, 2023 | opencompute.org |

## Domains

| Work | Authors, year | Link |
|---|---|---|
| MonetDB/X100 | Boncz, Zukowski & Nes, CIDR 2005 | cidrdb.org |
| Parsing Gigabytes of JSON per Second | Langdale & Lemire, 2019 | arXiv:1902.08318 |
| Striped Smith–Waterman | Farrar, Bioinformatics 2007 | doi:10.1093/bioinformatics/btl582 |
| QuickScorer | Lucchese et al., SIGIR 2015 | doi:10.1145/2766462.2767733 |
| EnvPool | Weng et al., NeurIPS 2022 | arXiv:2206.10558 |

## Books for after this one

| Book | Authors | Why |
|---|---|---|
| Algorithms for Modern Hardware | Slotin | SIMD algorithms, measured |
| Performance Analysis and Tuning on Modern CPUs | Bakhvalov | profiling methodology |
| Programming Massively Parallel Processors, 4th ed. | Hwu, Kirk & El Hajj | CUDA in depth |

---

**Next:** [Appendix F — Question Bank](./appendix_f_question_bank.md) — test yourself, chapter by chapter.
