"""Time one network call: SHRX (NumPy) vs QLKNN (JAX, jit).   TORAX env.

    [OMP_NUM_THREADS=1] python bench_latency.py <out.json>

Median of 11 x 400 calls on the same machine in the same process; the range is the min/max of the 11.
"network" = inputs -> fluxes only; "evaluate_full" = the whole chain a transport step needs (state -> inputs ->
network -> chi_e, chi_i) with the same NumPy bridge in front of both networks (QLKNN's is the jitted JAX model).  Idle machine, please: other processes distort the numbers.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torax  # noqa: F401

sys.path.append(str(Path(__file__).resolve().parents[2]))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from tokamak import shrx, transport as tr  # noqa: E402
from tokamak.machines import get_machine  # noqa: E402
from tokamak.shrx import reference  # noqa: E402

net = shrx.load()
model, _ = reference._load()                       # the real QLKNN, jitted below in f32 and f64
rng = np.random.default_rng(0)


def inputs(N):
    lo, hi = net.lo.copy(), net.hi.copy()
    lo[:2], hi[:2] = 0.0, 12.0
    lo[2:4], hi[2:4] = -1.0, 6.0
    hi[4], hi[5], lo[8], hi[8] = 6.0, 3.5, -4.0, 0.4
    return rng.uniform(lo + 0.2 * (hi - lo), lo + 0.8 * (hi - lo), (N, 10))


def bench(fn, sync=None, reps=400, trials=11):
    fn()
    out = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(reps):
            r = fn()
        if sync:
            sync(r)
        out.append((time.perf_counter() - t0) / reps * 1e6)
    return {"median_us": float(np.median(out)), "range_us": [float(min(out)), float(max(out))]}


res = {"threads_env": os.environ.get("OMP_NUM_THREADS", "default"), "jax": jax.__version__}
m = get_machine("iter")
for N in (26, 65):
    X = inputs(N)
    X32 = X.astype(np.float32)
    f32, f64 = jax.jit(model.predict), jax.jit(model.predict)
    a32, a64 = jnp.asarray(X32), jnp.asarray(X, dtype=jnp.float64)
    jax.block_until_ready(f32(a32))
    jax.block_until_ready(f64(a64))
    res[f"N{N}"] = {
        "shrx_network": bench(lambda: net.predict(X32)),
        "qlknn_jit_float32": bench(lambda: f32(a32), sync=jax.block_until_ready),
        "qlknn_jit_float64": bench(lambda: f64(a64), sync=jax.block_until_ready),
    }
    rho = np.linspace(0, 1, N)
    Te = 12 * (1 - 0.9 * rho ** 2) + 0.1
    q = 1 + 2.1 * rho ** 2
    res[f"N{N}"]["shrx_evaluate_full"] = bench(
        lambda: shrx.evaluate(rho, Te, Te * 0.9, 1.0 - 0.6 * rho ** 2, q, m, 1.6, 0.9))
    res[f"N{N}"]["qlknn_evaluate_full"] = bench(
        lambda: reference.evaluate(rho, Te, Te * 0.9, 1.0 - 0.6 * rho ** 2, q, m, 1.6, 0.9))
    g = tr.Grid(n=N, a=m.a, V_total=m.V, kappa_a=m.kappa_a)
    s = tr.initial_state(g, ne0=1.0, Te0=10.0, Ti0=9.0)
    res[f"N{N}"]["cgm_formula"] = bench(lambda: tr.chi_cgm(s, m.a, True, R0=m.R0))
    print(N, {k: round(v["median_us"], 1) for k, v in res[f"N{N}"].items()})
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
