"""A cycle-level output-stationary systolic array (Chapter 16; lab "systolic").

PE(i, j) holds C[i, j]. A rows enter from the left, B columns from the top, each skewed by one
cycle per row/column so that A[i, k] and B[k, j] meet in PE(i, j) at cycle t = i + j + k.
"""
import numpy as np


def simulate(A, B):
    R, K = A.shape
    K2, Cc = B.shape
    assert K == K2
    C = np.zeros((R, Cc), dtype=np.result_type(A, B))
    a_reg = np.zeros((R, Cc), dtype=C.dtype)  # value each PE passes right
    b_reg = np.zeros((R, Cc), dtype=C.dtype)  # value each PE passes down
    busy = 0
    cycles = K + R + Cc - 2
    for t in range(cycles):
        new_a = np.zeros_like(a_reg)
        new_b = np.zeros_like(b_reg)
        for i in range(R):
            for j in range(Cc):
                k = t - i - j
                a_in = (A[i, k] if 0 <= k < K else 0) if j == 0 else a_reg[i, j - 1]
                b_in = (B[k, j] if 0 <= k < K else 0) if i == 0 else b_reg[i - 1, j]
                if 0 <= k < K:
                    C[i, j] += a_in * b_in
                    busy += 1
                new_a[i, j], new_b[i, j] = a_in, b_in
        a_reg, b_reg = new_a, new_b
    return C, {"cycles": cycles, "utilization": busy / (R * Cc * cycles)}
