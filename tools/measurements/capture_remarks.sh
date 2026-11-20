#!/usr/bin/env bash
# Capture the compiler's own explanation of what it vectorized, for Chapters 13 and 21.
#   bash tools/measurements/capture_remarks.sh   -> tools/measurements/remarks.txt
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC=code/bench/bench.c
OUT=tools/measurements/remarks.txt
{
  echo "### $(gcc --version | head -1)"
  for target in "-march=x86-64" "-march=x86-64-v3" "-march=x86-64-v4 -mprefer-vector-width=512" \
                "-march=x86-64-v4 -mprefer-vector-width=512 -ffast-math"; do
    echo; echo "## gcc -O3 $target -fopt-info-vec-optimized -fopt-info-vec-missed"
    # shellcheck disable=SC2086
    gcc -O3 $target -fopt-info-vec-optimized -fopt-info-vec-missed -c "$SRC" -o /dev/null 2>&1 \
      | grep -E "bench\.c:([0-9]{1,2}|1[0-5][0-9]):[0-9]+: (optimized: loop vectorized|missed: not vectorized|missed: couldn't vectorize)" \
      | sort -t: -k2,2n -k3,3n -u || true
  done
  echo; echo "### $(clang --version | head -1)"
  echo; echo "## clang -O3 -march=x86-64-v4 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize"
  clang -O3 -march=x86-64-v4 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize \
        -Rpass-analysis=loop-vectorize -c "$SRC" -o /dev/null 2>&1 | grep -E "bench.c:([0-9]{1,2}|1[0-5][0-9]):[0-9]+: remark:" | sort -t: -k2,2n -k3,3n -u || true
  echo; echo "## gcc -O3 -march=x86-64-v4 -mprefer-vector-width=512 -S : the body of dot()"
  gcc -O3 -march=x86-64-v4 -mprefer-vector-width=512 -S -o - "$SRC" \
    | awk '/^dot:/,/jne/' | grep -vE "^\s*\.(cfi|p2align|size|type)|endbr64"
} > "$OUT"
echo "wrote $OUT"

# llvm-mca pipeline model of a dot-product body with 1 vs 4 accumulators (Chapter 4 and 14)
MCA=tools/measurements/mca
for f in dot1 dot4; do
  {
    echo "### llvm-mca -mcpu=sapphirerapids -timeline : $f.s"
    llvm-mca -mtriple=x86_64 -mcpu=sapphirerapids -iterations=2 -timeline -timeline-max-cycles=40 "$MCA/$f.s" 2>&1 \
      | sed -n '/Timeline view/,/^$/p;/Timeline view/,/Truncated/p' | awk '!seen[$0]++'
    echo "### -iterations=100 summary"
    llvm-mca -mtriple=x86_64 -mcpu=sapphirerapids -iterations=100 "$MCA/$f.s" 2>&1 | grep -E "Total Cycles|IPC|Block RThroughput"
  } > "$MCA/$f.txt"
done
echo "wrote $MCA/dot1.txt $MCA/dot4.txt"
