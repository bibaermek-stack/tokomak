"""Training set for SHRX-ST: on-policy states of spherical tokamaks, perturbed copies, and a box, all labelled by TGLFNN-STEP.

    TGLFNN_STEP_PKL=.../step.pkl python build_dataset.py <out.npz> <states.npz> [states2.npz ...] [--n-aug 1500000] [--n-box 400000] [--n-thr 600000]

  tag 0  the states the simulator visited with the teacher driving its transport (the data the model will meet)
  tag 1  the same states with the gradients, shear, collisionality ... scaled by random factors (log-normal, sigma 0.3-0.4)
  tag 2  uniform draws inside the teacher's input range (mean +- 2.5 std, gradients >= 0): keeps the far field sane
  tag 3  threshold-targeted: on-policy states with both temperature gradients scaled together by a factor in [0.3, 2.5], so
         the onset of the turbulence (where a stiff plasma sits) is resolved

``machine`` groups the rows for the held-out test: records of MAST / MAST-U and the scripted MAST-U discharge are one group.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tokamak.shrx import tglf  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = lambda name, default: int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
n_aug, n_box, n_thr = opt("--n-aug", 1_500_000), opt("--n-box", 400_000), opt("--n-thr", 600_000)
out_path, state_files = args[0], args[1:]
net = tglf.load()
rng = np.random.default_rng(0)
J = {n: i for i, n in enumerate(tglf.FEATURES)}


def label(X, batch=20000):
    Y = np.empty((len(X), 3), np.float32)
    S = np.empty((len(X), 2), np.float32)
    for i in range(0, len(X), batch):
        f = net.predict(X[i:i + batch])
        Y[i:i + batch] = np.stack([f["efe_gb"], f["efi_gb"], f["pfi_gb"]], 1)
        S[i:i + batch] = np.stack([f["efe_gb_std"], f["efi_gb_std"]], 1)
    return Y, S


def group(name):
    n = str(name).lower()
    if n.startswith("mast"):
        return "mast"
    for k in ("nstxu", "nstx", "start", "ktm"):
        if n.startswith(k):
            return k
    return n


Xs, Ys, Ss, srcs = [], [], [], []
for f in state_files:
    d = np.load(f)
    Xs.append(d["X"]); Ys.append(d["Y"]); Ss.append(d["std"]); srcs.append(np.array([group(s) for s in d["source"]]))
X0, Y0, S0, G0 = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ss), np.concatenate(srcs)
X0 = np.clip(X0, net.lo, net.hi)
print("on-policy rows", len(X0), {g: int((G0 == g).sum()) for g in np.unique(G0)})

rows = [(X0, Y0, S0, np.zeros(len(X0), np.int8), G0)]
grad = [J["RLNS_1"], J["RLTS_1"], J["RLTS_2"]]
other = [J[k] for k in ("TAUS_2", "DRMAJDX_LOC", "Q_LOC", "SHAT", "XNUE", "S_KAPPA_LOC", "S_DELTA_LOC", "BETAE")]


def pick(n):
    idx = rng.integers(0, len(X0), n)
    return X0[idx].astype(np.float64), G0[idx]


# tag 1: perturbed copies
Xa, Ga = pick(n_aug)
for c in grad:
    Xa[:, c] = Xa[:, c] * np.exp(rng.normal(0, 0.4, len(Xa))) + rng.normal(0, 0.15, len(Xa)) * (c == J["RLNS_1"])
for c in other:
    Xa[:, c] = Xa[:, c] + np.abs(Xa[:, c]) * rng.normal(0, 0.3, len(Xa))
Xa[:, J["RLTS_1"]] = np.maximum(Xa[:, J["RLTS_1"]], 0.0)
Xa[:, J["RLTS_2"]] = np.maximum(Xa[:, J["RLTS_2"]], 0.0)
Xa = np.clip(Xa, net.lo, net.hi).astype(np.float32)
Ya, Sa = label(Xa)
rows.append((Xa, Ya, Sa, np.ones(len(Xa), np.int8), Ga))

# tag 2: uniform box
lo = np.maximum(net.mu - 2.5 * net.sd, [-3, 0, 0, 0.02, 0.1, -1.5, 0.3, -1, 1e-5, 1.0, 0.0, 0.0, 0.0, 1e-5, 1.0])
hi = net.mu + 2.5 * net.sd
Xb = rng.uniform(lo, hi, (n_box, 15)).astype(np.float32)
Yb, Sb = label(Xb)
rows.append((Xb, Yb, Sb, np.full(len(Xb), 2, np.int8), np.full(len(Xb), "box")))

# tag 3: threshold-targeted
Xt, Gt = pick(n_thr)
f = np.exp(rng.uniform(np.log(0.3), np.log(2.5), len(Xt)))
Xt[:, J["RLTS_1"]] *= f
Xt[:, J["RLTS_2"]] *= f
Xt = np.clip(Xt, net.lo, net.hi).astype(np.float32)
Yt, St = label(Xt)
rows.append((Xt, Yt, St, np.full(len(Xt), 3, np.int8), Gt))

X = np.concatenate([r[0] for r in rows]); Y = np.concatenate([r[1] for r in rows]); S = np.concatenate([r[2] for r in rows])
tag = np.concatenate([r[3] for r in rows]); mach = np.concatenate([r[4] for r in rows])
np.savez_compressed(out_path, X=X, Y=Y, std=S, tag=tag, machine=mach, lo=net.lo, hi=net.hi, mu_teacher=net.mu, sd_teacher=net.sd)
print(out_path, X.shape, {t: int((tag == t).sum()) for t in range(4)})
