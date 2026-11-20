"""Summation order and float32 error (Chapter 6 legality, Chapter 18 kernels; lab "reduce").

A vectorized reduction with V lanes does not add the numbers in the order you wrote. These
functions reproduce the orders a compiler or a kernel actually uses, in float32, so you can see
how far apart the answers are.
"""
import math

import numpy as np

f32 = np.float32


def sum_sequential(x):
    """Left to right, one float32 accumulator: what the C source literally says."""
    return f32(np.cumsum(np.asarray(x, dtype=f32), dtype=f32)[-1]) if len(x) else f32(0)


def sum_lanes(x, V):
    """V independent accumulators (lane l sums x[l], x[l+V], ...), then a tree across lanes.

    This is the order of a V-wide SIMD loop followed by a horizontal reduction.
    """
    x = np.asarray(x, dtype=f32)
    pad = (-len(x)) % V
    lanes = np.concatenate([x, np.zeros(pad, dtype=f32)]).reshape(-1, V)
    acc = np.cumsum(lanes, axis=0, dtype=f32)[-1] if len(lanes) else np.zeros(V, dtype=f32)
    return _tree(list(acc))


def sum_pairwise(x):
    """Recursive halving: error grows like log2(n) instead of n."""
    return _tree([f32(v) for v in np.asarray(x, dtype=f32)])


def _tree(vals):
    vals = [f32(v) for v in vals]
    if not vals:
        return f32(0)
    while len(vals) > 1:
        nxt = [f32(vals[i] + vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
        if len(vals) % 2:
            nxt.append(vals[-1])
        vals = nxt
    return vals[0]


def sum_kahan(x):
    """Compensated summation: carries the rounding error of each add in a second variable."""
    s = f32(0)
    c = f32(0)
    for v in np.asarray(x, dtype=f32):
        y = f32(v - c)
        t = f32(s + y)
        c = f32(f32(t - s) - y)
        s = t
    return s


def exact(x):
    """Correctly rounded sum of the float32 inputs, computed in arbitrary precision."""
    return math.fsum(float(v) for v in np.asarray(x, dtype=f32))


def ulp_error(value, reference):
    """Error of a float32 result in units in the last place of the reference."""
    ref = f32(reference)
    ulp = np.spacing(abs(ref)) if ref != 0 else np.spacing(f32(1))
    return float(abs(float(value) - reference) / float(ulp))


def compare(x, widths=(1, 4, 8, 16)):
    ref = exact(x)
    out = {"exact": ref, "sequential": float(sum_sequential(x)), "pairwise": float(sum_pairwise(x)),
           "kahan": float(sum_kahan(x)), "numpy_sum": float(np.sum(np.asarray(x, dtype=f32)))}
    for V in widths:
        out[f"lanes_{V}"] = float(sum_lanes(x, V))
    return out
