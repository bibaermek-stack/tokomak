"""Train SHRX-ST: distil TGLFNN-STEP into a small NumPy-only network.   TORAX env (JAX + optax).

    python train_shrx_st.py <dataset.npz> <out.npz> [--width 128 --depth 4 --epochs 60 --holdout mast --seed 0]

Same recipe as SHRX (asinh-compressed gyro-Bohm fluxes, ReLU MLP, input box = the teacher's), with the 15 TGLF inputs and the
outputs (efe_gb, efi_gb, pfi_gb).  Rows of the held-out group never appear in training or validation; the metrics printed are
on those rows and on the training machines, restricted to points where the teacher's turbulence is on (|flux| > 1 GB).
The file is written in the format tokamak.shrx.Shrx reads.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torax  # noqa: F401  (before sys.path edits)

sys.path.append(str(Path(__file__).resolve().parents[3]))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--width", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--holdout", default="mast")
ap.add_argument("--scale-mult", type=float, default=0.25)
ap.add_argument("--w-onpolicy", type=float, default=2.0)
ap.add_argument("--w-thresh", type=float, default=1.0)
args = ap.parse_args()

d = np.load(args.data, allow_pickle=False)
X, Y, tag, mach, lo, hi = d["X"], d["Y"], d["tag"], d["machine"], d["lo"], d["hi"]
hold = np.isin(mach, args.holdout.split(",")) & (tag <= 1)
hold_all = np.isin(mach, args.holdout.split(","))
rng = np.random.default_rng(args.seed)
val = (~hold_all) & (rng.random(len(X)) < 0.04)
train = (~hold_all) & (~val)
print(f"samples {len(X)}: train {train.sum()} val {val.sum()} held-out ({args.holdout}) {hold.sum()}")
mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
scale = args.scale_mult * np.array([np.median(np.abs(Y[train, k])[np.abs(Y[train, k]) > 1e-2]) for k in range(3)], np.float32)
print("asinh scales:", scale)
w = np.where(tag == 0, args.w_onpolicy, np.where(tag == 1, 1.0, np.where(tag == 3, args.w_thresh, 0.4))).astype(np.float32)
Xs, Ys, Wt = jnp.asarray((X - mu) / sd, jnp.float32), jnp.asarray(np.arcsinh(Y / scale), jnp.float32), jnp.asarray(w)


def forward(ps, x):
    for W, b in ps[:-1]:
        x = jax.nn.relu(x @ W + b)
    return x @ ps[-1][0] + ps[-1][1]


key = jax.random.PRNGKey(args.seed)
sizes = [X.shape[1]] + [args.width] * args.depth + [3]
ps = []
for a, b in zip(sizes[:-1], sizes[1:]):
    key, k = jax.random.split(key)
    ps.append((jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a), jnp.zeros(b)))
n_train = int(train.sum())
spe = n_train // args.batch
opt = optax.adam(optax.cosine_decay_schedule(args.lr, args.epochs * spe, alpha=0.005))
st = opt.init(ps)


@jax.jit
def step(ps, st, x, y, wt):
    l, g = jax.value_and_grad(lambda p: jnp.mean(wt[:, None] * (forward(p, x) - y) ** 2))(ps)
    up, st = opt.update(g, st)
    return optax.apply_updates(ps, up), st, l


tr_idx, va_idx = np.where(train)[0], np.where(val)[0]
t0 = time.time()
for ep in range(args.epochs):
    rng.shuffle(tr_idx)
    tot = 0.0
    for i in range(spe):
        b = tr_idx[i * args.batch:(i + 1) * args.batch]
        ps, st, l = step(ps, st, Xs[b], Ys[b], Wt[b])
        tot += float(l)
    if ep % 10 == 9 or ep == args.epochs - 1:
        vl = float(jnp.mean((forward(ps, Xs[va_idx[:50000]]) - Ys[va_idx[:50000]]) ** 2))
        print(f"epoch {ep + 1:3d}  train {tot / spe:.5f}  val {vl:.5f}  ({time.time() - t0:.0f}s)", flush=True)

W_np = [(np.asarray(W, np.float32), np.asarray(b, np.float32)) for W, b in ps]


def predict(Xin):
    x = (np.clip(Xin, lo, hi) - mu) / sd
    for W, b in W_np[:-1]:
        x = np.maximum(x @ W + b, 0.0)
    return np.sinh(x @ W_np[-1][0] + W_np[-1][1]) * scale


def metrics(sel):
    if sel.sum() == 0:
        return None
    p, t = predict(X[sel].astype(np.float32)), Y[sel]
    out = {"n": int(sel.sum())}
    for k, name in enumerate(("efe", "efi")):
        on = np.abs(t[:, k]) > 1.0
        rel = np.abs(p[on, k] - t[on, k]) / np.abs(t[on, k])
        out[name] = {"n_on": int(on.sum()), "median_rel_err": float(np.median(rel)), "within25": float(np.mean(rel < 0.25)),
                     "r2": float(1 - ((p[:, k] - t[:, k]) ** 2).sum() / ((t[:, k] - t[:, k].mean()) ** 2).sum())}
    return out


res = {"heldout_onpolicy": metrics(hold & (tag == 0)), "heldout_perturbed": metrics(hold & (tag == 1)),
       "train_onpolicy": metrics(train & (tag == 0)), "train_perturbed": metrics(train & (tag == 1)),
       "box": metrics(tag == 2)}
for k, v in res.items():
    print(k, json.dumps(v))
arrays = {"n_layers": np.array(len(W_np)), "mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "scale": scale,
          "lo": lo.astype(np.float32), "hi": hi.astype(np.float32)}
for i, (W, b) in enumerate(W_np):
    arrays[f"W{i}"], arrays[f"b{i}"] = W, b
np.savez_compressed(args.out, **arrays)
n_par = sum(W.size + b.size for W, b in W_np)
Path(args.out).with_suffix(".json").write_text(json.dumps({"width": args.width, "depth": args.depth, "epochs": args.epochs,
                                                           "seed": args.seed, "holdout": args.holdout, "n_parameters": int(n_par),
                                                           "metrics": res}, indent=1))
print("saved", args.out, n_par, "parameters")
