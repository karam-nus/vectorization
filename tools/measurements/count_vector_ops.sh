#!/usr/bin/env bash
# "Did it vectorize?" without a profiler: count register widths used in each function (Chapter 21).
#   bash tools/measurements/count_vector_ops.sh  -> tools/measurements/vector_op_counts.txt
# xmm = 128-bit (also used by scalar SSE math), ymm = 256-bit, zmm = 512-bit;
# "scalar ss" counts scalar single-precision add/mul/fma: many of them inside a hot loop is a red flag.
set -euo pipefail
cd "$(dirname "$0")/../.."
TMP=$(mktemp -d)
OUT=tools/measurements/vector_op_counts.txt
{
  printf "%-8s %-8s %6s %5s %5s %5s %10s\n" target function instrs xmm ymm zmm "scalar ss"
  for v in "scalar:-O2 -fno-tree-vectorize -fno-tree-slp-vectorize -march=x86-64" \
           "avx2:-O3 -march=x86-64-v3" "avx512:-O3 -march=x86-64-v4 -mprefer-vector-width=512" \
           "avx512fm:-O3 -march=x86-64-v4 -mprefer-vector-width=512 -ffast-math"; do
    name=${v%%:*}; flags=${v#*:}
    # shellcheck disable=SC2086
    gcc $flags -c code/bench/bench.c -o "$TMP/$name.o"
    for fn in saxpy leaky dot prefix step_aos; do
      objdump -d --no-show-raw-insn "$TMP/$name.o" | awk -v f="<$fn>:" '$2==f{p=1;next} /^$/{p=0} p' > "$TMP/f.txt"
      printf "%-8s %-8s %6d %5d %5d %5d %10d\n" "$name" "$fn" "$(wc -l < "$TMP/f.txt")" \
        "$(grep -c xmm "$TMP/f.txt" || true)" "$(grep -c ymm "$TMP/f.txt" || true)" \
        "$(grep -c zmm "$TMP/f.txt" || true)" "$(grep -cE '\sv?(add|mul|sub|fmadd...|fnmadd...)ss\s' "$TMP/f.txt" || true)"
    done
  done
} > "$OUT"
rm -rf "$TMP"
cat "$OUT"
