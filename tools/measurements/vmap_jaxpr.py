#!/usr/bin/env python3
"""Show what jax.vmap does to a program (Chapter 12). Needs `pip install "jax[cpu]"`.

    python tools/measurements/vmap_jaxpr.py > tools/measurements/vmap_jaxpr.txt
"""
import time

import jax
import jax.numpy as jnp


def predict(w, x):            # written for ONE example: x has shape (d,)
    h = jnp.tanh(w @ x)       # (k, d) @ (d,) -> (k,)
    return jnp.where(h > 0, h, 0.01 * h).sum()


def loss(w, x, y):
    return (predict(w, x) - y) ** 2


k, d, B = 8, 16, 32
w = jnp.ones((k, d)) * 0.01
x1, y1 = jnp.ones((d,)), 1.0
xb, yb = jnp.ones((B, d)), jnp.ones((B,))

print(f"# jax {jax.__version__}\n")
print("## predict(w, x) for one example")
print(jax.make_jaxpr(predict)(w, x1))
print("\n## jax.vmap(predict, in_axes=(None, 0)) for a batch of", B)
print(jax.make_jaxpr(jax.vmap(predict, in_axes=(None, 0)))(w, xb))
print("\n## per-example gradients: jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))")
pe = jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))
print("output shape:", pe(w, xb, yb).shape, "(one gradient of w per example)")

# timing: Python loop of single-example calls vs one vmapped call (both jitted)
f1 = jax.jit(predict)
fb = jax.jit(jax.vmap(predict, in_axes=(None, 0)))
B = 4096
xb = jnp.ones((B, d))
f1(w, x1).block_until_ready(); fb(w, xb).block_until_ready()


def timeit(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


t_loop = timeit(lambda: [f1(w, xb[i]) for i in range(B)][-1].block_until_ready(), 3)
t_vmap = timeit(lambda: fb(w, xb).block_until_ready(), 200)
print(f"\n## timing, B = {B}, one CPU core: loop of jitted single calls {t_loop * 1e3:.1f} ms, "
      f"one vmapped call {t_vmap * 1e3:.3f} ms, ratio {t_loop / t_vmap:.0f}x")
