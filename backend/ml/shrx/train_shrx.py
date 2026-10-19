"""Train SHRX: distil QLKNN into a small network that runs on NumPy alone.   TORAX env (JAX + optax).

    python train_shrx.py <labeled.npz> <out_dir> [--width 128 --depth 3 --epochs 60 --seed 0]
                         [--holdout nstx,demo] [--name shrx_w128_d3]

Inputs are QLKNN's ten dimensionless numbers, outputs the gyro-Bohm fluxes (q_i, q_e, Gamma_e).
Targets are compressed with asinh so the loss sees both the weak flux near the critical gradient
and the strong flux far above it.  The held-out machines never appear in training or validation.
The result is a plain .npz (weights, scalers, input box, metrics) that tokamak.shrx loads.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torax  # noqa: F401  (before sys.path edits)

sys.path.append(str(Path(__file__).resolve().parents[2]))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from tokamak.shrx import bridge  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("out_dir")
ap.add_argument("--width", type=int, default=128)
ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--holdout", default="nstx,demo")
ap.add_argument("--name", default=None)
ap.add_argument("--w-real", type=float, default=1.0)
ap.add_argument("--w-aug", type=float, default=0.7)
ap.add_argument("--w-box", type=float, default=0.4)
ap.add_argument("--w-thresh", type=float, default=1.0, help="weight of the threshold-targeted samples (tag 3)")
ap.add_argument("--bag", type=float, default=1.0, help="fraction of the training rows each member sees (bagging)")
ap.add_argument("--scale-mult", type=float, default=0.25,
                help="asinh scale = scale-mult x typical flux; smaller = loss closer to relative error")
ap.add_argument("--w-source", default="", help="e.g. qlknnref=2.5,shrxr1=1.5: extra weight for states from those runs")
ap.add_argument("--w-machine", default="", help="e.g. iter=2: extra weight for one machine's states")
args = ap.parse_args()
name = args.name or f"shrx_w{args.width}_d{args.depth}_s{args.seed}"

d = np.load(args.data, allow_pickle=False)
X, Y, tag, mach = d["X"], d["Y"], d["tag"], d["machine"]
lo, hi = d["lo"], d["hi"]
hold = np.isin(mach, args.holdout.split(","))
rng = np.random.default_rng(args.seed)
val = (~hold) & (rng.random(len(X)) < 0.04)
train = (~hold) & (~val)
if args.bag < 1.0:
    train &= rng.random(len(X)) < args.bag
test = hold & (tag <= 1)
print(f"samples {len(X)}: train {train.sum()}  val {val.sum()}  test (held-out machines {args.holdout}) {test.sum()}")

mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
# per-output asinh scale: typical size of a non-negligible flux
scale = args.scale_mult * np.array([np.median(np.abs(Y[train, k])[np.abs(Y[train, k]) > 1e-3]) for k in range(3)], np.float32)
print("asinh scales (qi, qe, pfe):", scale)
w = np.where(tag == 0, args.w_real, np.where(tag == 1, args.w_aug, np.where(tag == 3, args.w_thresh, args.w_box))).astype(np.float32)
if "source" in d.files:
    for item in filter(None, args.w_source.split(",")):
        k, v = item.split("=")
        w[d["source"] == k] *= float(v)
for item in filter(None, args.w_machine.split(",")):
    k, v = item.split("=")
    w[mach == k] *= float(v)

f32 = lambda a: jnp.asarray(a, jnp.float32)
Xs = f32((X - mu) / sd)
Ys = f32(np.arcsinh(Y / scale))
Wt = f32(w)


def init(key, sizes):
    ps = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a), jnp.zeros(b)))
    return ps


def forward(ps, x):
    for W, b in ps[:-1]:
        x = jax.nn.relu(x @ W + b)
    return x @ ps[-1][0] + ps[-1][1]


def loss_fn(ps, x, y, wt):
    return jnp.mean(wt[:, None] * (forward(ps, x) - y) ** 2)


ps = init(jax.random.PRNGKey(args.seed), [10] + [args.width] * args.depth + [3])
n_train = int(train.sum())
steps_per_epoch = n_train // args.batch
sched = optax.cosine_decay_schedule(args.lr, args.epochs * steps_per_epoch, alpha=0.005)
opt = optax.adam(sched)
st = opt.init(ps)


@jax.jit
def step(ps, st, x, y, wt):
    l, g = jax.value_and_grad(loss_fn)(ps, x, y, wt)
    up, st = opt.update(g, st)
    return optax.apply_updates(ps, up), st, l


tr_idx = np.where(train)[0]
va_idx = np.where(val)[0]
t0 = time.time()
for ep in range(args.epochs):
    rng.shuffle(tr_idx)
    tot = 0.0
    for i in range(steps_per_epoch):
        b = tr_idx[i * args.batch:(i + 1) * args.batch]
        ps, st, l = step(ps, st, Xs[b], Ys[b], Wt[b])
        tot += float(l)
    if ep % 10 == 0 or ep == args.epochs - 1:
        vl = float(loss_fn(ps, Xs[va_idx], Ys[va_idx], Wt[va_idx]))
        print(f"epoch {ep:3d}  train {tot / steps_per_epoch:.5f}  val {vl:.5f}  ({time.time() - t0:.0f}s)")

# ----------------------------------------------------------------------------- export
Wn = [np.asarray(W, np.float32) for W, _ in ps]
bn = [np.asarray(b, np.float32) for _, b in ps]


def predict_np(Xin):
    h = (np.clip(Xin, lo, hi).astype(np.float32) - mu.astype(np.float32)) / sd.astype(np.float32)
    for W, b in zip(Wn[:-1], bn[:-1]):
        h = np.maximum(h @ W + b, 0.0)
    return np.sinh(h @ Wn[-1] + bn[-1]) * scale


# ----------------------------------------------------------------------------- metrics
g_ref = bridge.chi_gb(8.0, 5.3, 2.0)            # ITER-like gyro-Bohm unit, to express chi in m^2/s
c_min, c_max = 0.05 / g_ref, 100.0 / g_ref


def chi_hat(q, A):
    return np.clip(q / np.maximum(A, bridge.EPS), c_min, c_max)


def report(sel, label):
    P, T = predict_np(X[sel]), Y[sel]
    out = {}
    for k, nm in enumerate(("qi", "qe", "pfe")):
        out[f"r2_{nm}"] = float(1 - np.sum((T[:, k] - P[:, k]) ** 2) / np.sum((T[:, k] - T[:, k].mean()) ** 2))
    for k, nm, col in ((0, "chi_i", 0), (1, "chi_e", 1)):
        A = X[sel][:, col if k == 0 else 1]
        # chi_i uses R/L_Ti (col 0), chi_e uses R/L_Te (col 1)
        ct, cp = chi_hat(T[:, k], X[sel][:, 0 if k == 0 else 1]), chi_hat(P[:, k], X[sel][:, 0 if k == 0 else 1])
        # (no R^2 of chi itself: chi = q/A explodes where the gradient A vanishes and that swamps the sum)
        act = ct > 2 * c_min
        out[f"{nm}_median_rel_err_active"] = float(np.median(np.abs(cp[act] - ct[act]) / ct[act])) if act.any() else None
        out[f"{nm}_within20pct_active"] = float(np.mean(np.abs(cp[act] - ct[act]) / ct[act] < 0.2)) if act.any() else None
    print(label, {k: (round(v, 4) if v is not None else None) for k, v in out.items()})
    return out


metrics = {"val_all": report(val, "val  "),
           "test_heldout_machines": report(test, "test "),
           "test_real_only": report(test & (tag == 0), "test(real)")}

# latency, plain NumPy float32
lat = {}
for N in (26, 65):
    xb = X[test][:N]
    predict_np(xb)
    ts = []
    for _ in range(9):
        t1 = time.perf_counter()
        for _ in range(300):
            predict_np(xb)
        ts.append((time.perf_counter() - t1) / 300 * 1e6)
    lat[f"N{N}_us_median"] = float(np.median(ts))
print("latency (us, numpy fp32):", lat)

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
arrays = {"mu": mu, "sd": sd, "scale": scale, "lo": lo, "hi": hi, "n_layers": np.array(len(Wn))}
for i, (W, b) in enumerate(zip(Wn, bn)):
    arrays[f"W{i}"], arrays[f"b{i}"] = W, b
np.savez_compressed(out_dir / f"{name}.npz", **arrays)
meta = {"name": name, "width": args.width, "depth": args.depth, "epochs": args.epochs, "seed": args.seed,
        "n_parameters": int(sum(W.size + b.size for W, b in zip(Wn, bn))),
        "holdout_machines": args.holdout.split(","), "n_train": int(train.sum()),
        "metrics": metrics, "latency_us": lat, "teacher": "QLKNN qlknn_7_11_v1 (fusion_surrogates, Apache-2.0) via TORAX"}
(out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1))
print("saved", out_dir / f"{name}.npz", meta["n_parameters"], "parameters")
