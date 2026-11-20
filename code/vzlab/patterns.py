"""Loop versus vectorized implementations of common ML idioms.

Every ``*_loop`` function is the literal Python you would write first; the function without the
suffix is the vectorized NumPy version. tests/test_patterns.py checks they agree.
"""
import numpy as np


# ---------------------------------------------------------------- pairwise squared distances (Ch 20)
def pdist_loop(X, Y):
    """Pure Python lists, three nested loops: the rung everyone starts on."""
    n, d, m = len(X), len(X[0]), len(Y)
    D = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            acc = 0.0
            for k in range(d):
                t = X[i][k] - Y[j][k]
                acc += t * t
            D[i][j] = acc
    return np.array(D, dtype=np.float64)


def pdist_rowloop(X, Y):
    """Half-vectorized: a Python loop over rows, NumPy inside."""
    return np.stack([((x[None, :] - Y) ** 2).sum(axis=1) for x in X])


def pdist_broadcast(X, Y):
    """Fully vectorized by broadcasting. Materialises an (n, m, d) temporary."""
    diff = X[:, None, :] - Y[None, :, :]
    return np.einsum("ijk,ijk->ij", diff, diff)


def pdist_gemm(X, Y):
    """Algebraic reformulation: ||x||^2 + ||y||^2 - 2 x.y — the work becomes one matmul.

    Faster and memory-light, but it subtracts two large numbers: for near-identical points the
    result can come out slightly negative in float32, so clamp at zero.
    """
    xx = np.einsum("ij,ij->i", X, X)[:, None]
    yy = np.einsum("ij,ij->i", Y, Y)[None, :]
    return np.maximum(xx + yy - 2.0 * (X @ Y.T), 0.0)


# ---------------------------------------------------------------- softmax / layernorm (Ch 18)
def softmax_loop(x):
    out = np.empty_like(x)
    for r in range(x.shape[0]):
        m = max(x[r])
        e = [np.exp(v - m) for v in x[r]]
        s = sum(e)
        out[r] = [v / s for v in e]
    return out


def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)  # subtracting the max keeps exp() from overflowing
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def online_softmax_stats(x):
    """One pass over each row: running max m and running sum l rescaled when m grows.

    This is the recurrence FlashAttention uses to fuse softmax with the matmul (Ch 18).
    Returns (m, l) such that softmax(x) = exp(x - m) / l.
    """
    m = np.full(x.shape[0], -np.inf, dtype=x.dtype)
    l = np.zeros(x.shape[0], dtype=x.dtype)
    for j in range(x.shape[1]):  # stream over columns, vectorized over rows
        m_new = np.maximum(m, x[:, j])
        l = l * np.exp(m - m_new) + np.exp(x[:, j] - m_new)
        m = m_new
    return m, l


def layernorm_loop(x, eps=1e-5):
    out = np.empty_like(x)
    for r in range(x.shape[0]):
        mu = sum(x[r]) / x.shape[1]
        var = sum((v - mu) ** 2 for v in x[r]) / x.shape[1]
        out[r] = [(v - mu) / np.sqrt(var + eps) for v in x[r]]
    return out


def layernorm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


# ---------------------------------------------------------------- boxes (Ch 11)
def iou_matrix_loop(a, b):
    out = np.zeros((len(a), len(b)))
    for i, p in enumerate(a):
        for j, q in enumerate(b):
            iw = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
            ih = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
            inter = iw * ih
            union = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
            out[i, j] = inter / union
    return out


def iou_matrix(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])  # (n, m, 2) by broadcasting
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


# ---------------------------------------------------------------- small idioms (Ch 11)
def one_hot_loop(labels, k):
    out = np.zeros((len(labels), k))
    for i, c in enumerate(labels):
        out[i, c] = 1.0
    return out


def one_hot(labels, k):
    """Broadcast-compare: n*k comparisons and an n*k boolean temporary. Vectorized, not cheap."""
    return (labels[:, None] == np.arange(k)[None, :]).astype(np.float64)


def one_hot_scatter(labels, k):
    """Scatter: allocate zeros once, write n ones. Same answer, n instead of n*k work."""
    out = np.zeros((len(labels), k))
    out[np.arange(len(labels)), labels] = 1.0
    return out


def moving_average_loop(x, w):
    return np.array([sum(x[i:i + w]) / w for i in range(len(x) - w + 1)])


def moving_average(x, w):
    """A sliding window turned into two prefix sums: O(n) work, no Python loop."""
    c = np.cumsum(np.concatenate([[0.0], x]))
    return (c[w:] - c[:-w]) / w


def histogram_loop(idx, k):
    h = np.zeros(k, dtype=np.int64)
    for i in idx:
        h[i] += 1
    return h


def histogram(idx, k):
    """Scatter-add with conflicts: bincount does it; ``h[idx] += 1`` silently does not."""
    return np.bincount(idx, minlength=k)


def leaky_relu_loop(x, slope=0.01):
    return np.array([v if v > 0 else slope * v for v in x])


def leaky_relu(x, slope=0.01):
    return np.where(x > 0, x, slope * x)  # both sides computed, then a per-lane select


# ---------------------------------------------------------------- linear recurrence as a parallel scan (Ch 10)
def linear_recurrence_loop(a, b):
    """y[i] = a[i] * y[i-1] + b[i], y[-1] = 0: a distance-1 flow dependence."""
    y = np.empty(len(a))
    prev = 0.0
    for i in range(len(a)):
        prev = a[i] * prev + b[i]
        y[i] = prev
    return y


def linear_recurrence_scan(a, b):
    """The same recurrence in ceil(log2 n) whole-array steps (Hillis–Steele scan).

    Each step composes affine maps: (A, B) • (A_prev, B_prev) = (A * A_prev, A * B_prev + B).
    """
    A, B = np.asarray(a, dtype=np.float64).copy(), np.asarray(b, dtype=np.float64).copy()
    shift = 1
    while shift < len(A):
        A_prev = np.concatenate([np.ones(shift), A[:-shift]])
        B_prev = np.concatenate([np.zeros(shift), B[:-shift]])
        A, B = A * A_prev, A * B_prev + B
        shift *= 2
    return B
