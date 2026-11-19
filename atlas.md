---
title: "Vector ISA Atlas"
---

[← Back to Table of Contents](./README.md)

# Vector ISA Atlas

> *"Fifty years of the same bargain, on one chart."*

The Atlas plots and tabulates every vector instruction set and matrix engine discussed in this book from a
single data file, [`assets/data/isas.json`](https://github.com/karam-nus/vectorization/blob/main/assets/data/isas.json).
The vertical axis is the number of bits one instruction (or one matrix operation) processes, on a log scale;
diamonds are vector-length-agnostic designs. Filter by kind, search any field, hover a point for its notes and
source, and click a column to sort.

<div class="lab" data-lab="atlas"></div>

---

## How to read it

- **Bits per instruction** means different things per kind, and the `note` field says which. For fixed
  SIMD it is the register width. For vector-length-agnostic ISAs (SVE, RISC-V V) it is the
  architectural *maximum*; real chips implement less (A64FX: 512, Neoverse V1: 256, SpacemiT X60: 256).
  For matrix engines it is the operand footprint of one operation (an AMX tile, a TPU array's weights).
  Compare within a kind, not across kinds.
- **GPU SIMT** rows count a 32-thread warp × 32 bits = 1024 bits; Hopper's warpgroup rows count four warps.
- **Masking** names the mechanism: blends on vector masks (SSE/AVX2), mask registers (AVX-512), predicate
  registers (SVE), a mask in `v0` (RISC-V V), hardware active masks (GPUs). Chapter 8 compares them.
- **Unconfirmed** in a note means the figure comes from secondary sources (press coverage, third-party
  analyses), not vendor documentation, at the time of writing.

## The data file

Each row has: `name, vendor, kind, year, width_bits, vla, predication, gather, scatter, ml_types, matrix,
first_silicon, note, source`. `source` names the manual, paper or document the row was taken from. The same
file generates [Appendix D — Hardware Matrix](./appendix_d_hardware_matrix.md) (`tools/build_hardware_matrix.py`).

To add a row: append it with a source, keep `kind` to one of the existing values, and regenerate the appendix.

If the interactive view does not load (for example on the raw GitHub view), use
[Appendix D](./appendix_d_hardware_matrix.md), which contains the same rows as a static table.
