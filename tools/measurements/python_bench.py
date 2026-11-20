#!/usr/bin/env python3
"""Python / NumPy measurements for Chapters 2, 6, 11 and 20. Merges into results.json.

    python tools/measurements/python_bench.py                       # every experiment
    python tools/measurements/python_bench.py --only np_vectorize   # just one (others untouched)

Run run_bench.py first (it creates results.json). Experiments added here:
  pdist_python         the Python and NumPy rungs of the Chapter 20 ladder
  numpy_idioms         loop vs vectorized for the Chapter 2 / 11 idioms
  numpy_call_overhead  np.add cost per call and per element across sizes (Chapter 11)
  np_vectorize         np.vectorize vs a list comprehension vs np.where (Chapter 11)
  reduction_orders     float32 sum of the same 1 M numbers in seven orders (Chapter 6)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
from vzlab import patterns as P  # noqa: E402
from vzlab import reduce as R  # noqa: E402

OUT = os.path.join(HERE, "results.json")


def best(f, *args, repeats=5, min_time=0.2):
    """Best per-call time over `repeats` batches, each long enough to be measurable."""
    reps = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            f(*args)
        if time.perf_counter() - t0 > min_time / 4 or reps > 1 << 20:
            break
        reps *= 2
    out = 1e30
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(reps):
            f(*args)
        out = min(out, (time.perf_counter() - t0) / reps)
    return out


def exp_pdist_python(rng):
    d, rows = 64, []
    n_small = 96  # pure Python at n = 1024 would take minutes; cost is normalised per pair x dim
    Xs = rng.standard_normal((n_small, d)).astype(np.float32)
    t = best(P.pdist_loop, Xs.tolist(), Xs.tolist(), repeats=3)
    rows.append({"rung": "python_loops", "n": n_small, "sec": t, "ns_per_pair_dim": t / (n_small ** 2 * d) * 1e9})
    n = 1024
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = rng.standard_normal((n, d)).astype(np.float32)
    for name, f in (("numpy_rowloop", P.pdist_rowloop), ("numpy_broadcast", P.pdist_broadcast),
                    ("numpy_gemm", P.pdist_gemm)):
        t = best(f, X, Y, repeats=5)
        rows.append({"rung": name, "n": n, "sec": t, "ns_per_pair_dim": t / (n * n * d) * 1e9})
    return rows


def exp_numpy_idioms(rng):
    x2 = rng.standard_normal((256, 512)).astype(np.float64)

    def boxes(k):
        xy = rng.uniform(0, 500, (k, 2))
        return np.concatenate([xy, xy + rng.uniform(5, 80, (k, 2))], 1)
    a, b = boxes(200), boxes(200)
    lab = rng.integers(0, 1000, 20000)
    ser = rng.standard_normal(200_000)
    cases = [
        ("softmax", "256 x 512", P.softmax_loop, P.softmax, (x2,)),
        ("layernorm", "256 x 512", P.layernorm_loop, P.layernorm, (x2,)),
        ("iou_matrix", "200 x 200 boxes", P.iou_matrix_loop, P.iou_matrix, (a, b)),
        ("one_hot (broadcast compare)", "20 000 labels, 1000 classes", P.one_hot_loop, P.one_hot, (lab, 1000)),
        ("one_hot (scatter)", "20 000 labels, 1000 classes", P.one_hot_loop, P.one_hot_scatter, (lab, 1000)),
        ("moving_average", "200 000 samples, window 50", P.moving_average_loop, P.moving_average, (ser, 50)),
        ("histogram", "20 000 indices, 1000 bins", P.histogram_loop, P.histogram, (lab, 1000)),
        ("leaky_relu", "200 000 values", P.leaky_relu_loop, P.leaky_relu, (ser,)),
    ]
    rows = []
    for name, size, loop, vec, args in cases:
        tl = best(loop, *args, repeats=3, min_time=0.3)
        tv = best(vec, *args, repeats=5)
        rows.append({"idiom": name, "size": size, "loop_sec": tl, "vector_sec": tv, "speedup": tl / tv})
    return rows


def exp_numpy_call_overhead(rng):
    rows = []
    for n in (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000):
        a = rng.standard_normal(n).astype(np.float32)
        b = rng.standard_normal(n).astype(np.float32)
        c = np.empty_like(a)
        t = best(np.add, a, b, c, repeats=5)
        rows.append({"n": n, "sec": t, "ns_per_elem": t / n * 1e9})
    return rows


def exp_np_vectorize(rng):
    x = rng.standard_normal(100_000)

    def f_py(v):
        return v if v > 0 else 0.01 * v
    vf = np.vectorize(f_py)
    return {"n": len(x),
            "list_comprehension_sec": best(lambda: [f_py(v) for v in x], repeats=3),
            "np_vectorize_sec": best(vf, x, repeats=3),
            "np_where_sec": best(P.leaky_relu, x, repeats=5)}


def exp_reduction_orders(rng):
    x = (rng.standard_normal(1_000_000) * 1e3 + 1e4).astype(np.float32)
    r = R.compare(x, widths=(4, 8, 16))
    ref = r["exact"]
    return {"n": len(x), "distribution": "N(1e4, 1e3^2), float32", "exact": ref,
            "results": {k: {"value": v, "ulp_error": R.ulp_error(v, ref)} for k, v in r.items() if k != "exact"}}


EXPERIMENTS = {
    "pdist_python": exp_pdist_python,
    "numpy_idioms": exp_numpy_idioms,
    "numpy_call_overhead": exp_numpy_call_overhead,
    "np_vectorize": exp_np_vectorize,
    "reduction_orders": exp_reduction_orders,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(EXPERIMENTS), default=None)
    args = ap.parse_args()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"experiments": {}}
    for name, fn in EXPERIMENTS.items():
        if args.only is not None and name not in args.only:
            continue
        res["experiments"][name] = fn(np.random.default_rng(0))
        print(f"{name} done", file=sys.stderr)
    res.setdefault("machine", {})["numpy"] = np.__version__
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"updated {OUT}")


if __name__ == "__main__":
    main()
