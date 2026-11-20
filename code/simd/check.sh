#!/usr/bin/env bash
# Cross-compile the three saxpy variants and show the vector instructions each produced.
# Needs only clang (no Arm or RISC-V machine, no sysroot): -ffreestanding avoids libc headers.
#   bash code/simd/check.sh   -> tools/measurements/simd_asm.txt
set -euo pipefail
cd "$(dirname "$0")"
OUT=../../tools/measurements/simd_asm.txt
TMP=$(mktemp -d)
{
  echo "### $(clang --version | head -1)"
  echo; echo "## Arm SVE (vector-length agnostic): clang --target=aarch64-linux-gnu -march=armv8-a+sve"
  clang --target=aarch64-linux-gnu -march=armv8-a+sve -ffreestanding -O2 -c saxpy_sve.c -o "$TMP/sve.o"
  llvm-objdump -d --no-show-raw-insn "$TMP/sve.o" | grep -E "^\s+[0-9a-f]+:" | sed 's/^ *//'
  echo; echo "## RISC-V V 1.0 (vector-length agnostic): clang --target=riscv64-linux-gnu -march=rv64gcv"
  clang --target=riscv64-linux-gnu -march=rv64gcv -ffreestanding -O2 -c saxpy_rvv.c -o "$TMP/rvv.o"
  llvm-objdump -d --mattr=+v --no-show-raw-insn "$TMP/rvv.o" | grep -E "^\s+[0-9a-f]+:" | sed 's/^ *//'
  echo; echo "## Arm NEON (fixed 128-bit): clang --target=aarch64-linux-gnu"
  clang --target=aarch64-linux-gnu -ffreestanding -O2 -c saxpy_neon.c -o "$TMP/neon.o"
  llvm-objdump -d --no-show-raw-insn "$TMP/neon.o" | grep -E "^\s+[0-9a-f]+:" | sed 's/^ *//'
} > "$OUT"
rm -rf "$TMP"
echo "wrote $OUT"
