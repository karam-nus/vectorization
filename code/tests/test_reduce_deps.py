import numpy as np
import pytest

from vzlab import deps, reduce


def test_reduction_orders_differ_but_tree_is_more_accurate():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(100_000) * 1e3 + 1e4).astype(np.float32)
    r = reduce.compare(x)
    ref = r["exact"]
    seq = reduce.ulp_error(r["sequential"], ref)
    tree = reduce.ulp_error(r["pairwise"], ref)
    assert r["sequential"] != r["lanes_16"]            # reordering changes the bits
    assert tree < seq                                  # and the tree is closer to the truth
    assert reduce.ulp_error(r["kahan"], ref) <= 1.0


def test_lane_sum_matches_manual_simd_order():
    x = np.arange(1, 21, dtype=np.float32)
    assert float(reduce.sum_lanes(x, 4)) == 210.0


@pytest.mark.parametrize("R,W", [(0, 1), (0, 3), (0, 8), (0, 16), (1, 0), (4, 0), (0, 0), (-2, 5), (3, 3)])
@pytest.mark.parametrize("V", [2, 4, 8, 16])
def test_analytic_legality_matches_simulation(R, W, V):
    assert deps.legal_analytic(R, W, V) == deps.legal_by_simulation(R, W, V)


def test_distance_classification():
    assert deps.distance(0, 1) == 1          # a[i+1] = f(a[i])   flow, distance 1 -> scalar only
    assert deps.max_safe_vl(0, 8) == 8       # distance 8: safe up to 8 lanes (one AVX2 fp32 register)
    assert deps.max_safe_vl(4, 0) == 1 << 16  # anti-dependence: always safe in chunked execution


def test_gcd_test():
    assert deps.gcd_test(2, 0, 2, 1) is False   # a[2i] vs a[2j+1]: evens never meet odds
    assert deps.gcd_test(2, 0, 4, 2) is True    # may alias
