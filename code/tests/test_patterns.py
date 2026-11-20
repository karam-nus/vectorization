import numpy as np
import pytest

from vzlab import patterns as P

rng = np.random.default_rng(0)


def test_pdist_all_rungs_agree():
    X = rng.standard_normal((17, 8))
    Y = rng.standard_normal((13, 8))
    ref = P.pdist_loop(X.tolist(), Y.tolist())
    for f in (P.pdist_rowloop, P.pdist_broadcast, P.pdist_gemm):
        np.testing.assert_allclose(f(X, Y), ref, rtol=1e-10, atol=1e-10)


def test_pdist_gemm_cancellation_needs_clamp():
    # Nearly identical, far-from-origin points: ||x||^2 + ||y||^2 - 2x.y cancels catastrophically.
    X = (1000.0 + rng.standard_normal((4, 16)) * 1e-3).astype(np.float32)
    D = X[:, None, :] - X[None, :, :]
    exact = np.einsum("ijk,ijk->ij", D.astype(np.float64), D.astype(np.float64))
    xx = np.einsum("ij,ij->i", X, X)[:, None]
    raw = xx + xx.T - 2.0 * (X @ X.T)
    assert np.abs(raw - exact).max() > 1e-3          # the trick loses the answer in float32 ...
    assert (P.pdist_gemm(X, X) >= 0).all()            # ... and the clamp at least keeps it >= 0


def test_softmax_and_online_softmax():
    x = rng.standard_normal((5, 33)).astype(np.float64) * 10
    np.testing.assert_allclose(P.softmax(x), P.softmax_loop(x), rtol=1e-12)
    m, l = P.online_softmax_stats(x)
    np.testing.assert_allclose(np.exp(x - m[:, None]) / l[:, None], P.softmax(x), rtol=1e-12)


def test_softmax_survives_large_logits():
    x = np.array([[1000.0, 1001.0, 1002.0]])
    assert np.isfinite(P.softmax(x)).all()


def test_layernorm():
    x = rng.standard_normal((4, 64))
    np.testing.assert_allclose(P.layernorm(x), P.layernorm_loop(x), rtol=1e-10, atol=1e-12)


def test_iou_matrix():
    def boxes(n):
        xy = rng.uniform(0, 50, (n, 2))
        wh = rng.uniform(1, 30, (n, 2))
        return np.concatenate([xy, xy + wh], axis=1)
    a, b = boxes(9), boxes(7)
    np.testing.assert_allclose(P.iou_matrix(a, b), P.iou_matrix_loop(a, b), rtol=1e-12)


def test_small_idioms():
    labels = rng.integers(0, 6, 40)
    np.testing.assert_array_equal(P.one_hot(labels, 6), P.one_hot_loop(labels, 6))
    np.testing.assert_array_equal(P.one_hot_scatter(labels, 6), P.one_hot_loop(labels, 6))
    x = rng.standard_normal(100)
    np.testing.assert_allclose(P.moving_average(x, 7), P.moving_average_loop(x, 7), atol=1e-12)
    np.testing.assert_array_equal(P.histogram(labels, 6), P.histogram_loop(labels, 6))
    np.testing.assert_allclose(P.leaky_relu(x), P.leaky_relu_loop(x))


def test_fancy_index_increment_drops_duplicates():
    # The trap Chapter 11 warns about: h[idx] += 1 is a gather, an add and a scatter.
    idx = np.array([1, 1, 1, 2])
    h = np.zeros(3, dtype=int)
    h[idx] += 1
    assert h.tolist() == [0, 1, 1]                    # not [0, 3, 1]
    h2 = np.zeros(3, dtype=int)
    np.add.at(h2, idx, 1)
    assert h2.tolist() == [0, 3, 1]


@pytest.mark.parametrize("n", [1, 2, 3, 16, 37, 1000])
def test_linear_recurrence_scan_matches_loop(n):
    a = rng.uniform(0.5, 1.0, n)
    b = rng.standard_normal(n)
    np.testing.assert_allclose(P.linear_recurrence_scan(a, b), P.linear_recurrence_loop(a, b), rtol=1e-10, atol=1e-12)
