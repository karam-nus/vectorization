"""Dependence distance and vector legality (Chapter 6; lab "deps").

The loop studied is   for i in range(n):  a[i + W] = c * a[i + R] + b[i]

Vector execution with V lanes is modelled the way hardware does it: for each chunk of V
iterations, *all* loads happen, then the arithmetic, then *all* stores. The vector version is
legal exactly when it produces the same array as the scalar loop.
"""
from math import gcd


def distance(R, W):
    """Dependence distance d = W - R.  d > 0: flow (read-after-write) from iteration i to i + d.
    d < 0: anti (write-after-read). d == 0: read and write of the same element in one iteration."""
    return W - R


def legal_analytic(R, W, V):
    """Chunked vector execution is legal iff d <= 0 or d >= V."""
    d = distance(R, W)
    return d <= 0 or d >= V


def max_safe_vl(R, W, cap=1 << 16):
    """Largest V for which the loop may be vectorized (cap if unlimited)."""
    d = distance(R, W)
    return cap if d <= 0 else d


def run_scalar(a, b, R, W, n, c=0.5):
    a = list(a)
    for i in range(n):
        a[i + W] = c * a[i + R] + b[i]
    return a


def run_vector(a, b, R, W, n, V, c=0.5):
    a = list(a)
    for base in range(0, n, V):
        idx = range(base, min(base + V, n))
        loads = [a[i + R] for i in idx]           # one vector load
        vals = [c * x + b[i] for x, i in zip(loads, idx)]  # one vector FMA
        for i, v in zip(idx, vals):               # one vector store
            a[i + W] = v
    return a


def legal_by_simulation(R, W, V, n=64):
    off = max(0, -min(R, W))
    size = n + off + max(R, W, 0) + 1
    a = [float(k % 7) - 3.0 for k in range(size)]
    b = [0.25 * (k % 5) for k in range(n)]
    return run_scalar(a, b, R + off, W + off, n) == run_vector(a, b, R + off, W + off, n, V)


def gcd_test(c_write, k_write, c_read, k_read):
    """Can a[c_w*i + k_w] and a[c_r*j + k_r] ever name the same element for integers i, j?

    Only if gcd(c_w, c_r) divides (k_r - k_w). False means *provably independent*; True means
    "maybe" — the test is conservative, which is why compilers give up on loops that are fine.
    """
    g = gcd(c_write, c_read)
    if g == 0:
        return k_write == k_read
    return (k_read - k_write) % g == 0
