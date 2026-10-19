"""Targeted training data around QLKNN's turbulence threshold.   TORAX environment.

    python threshold_samples.py <labeled_in.npz> <labeled_out.npz> [--states 160000] [--per-state 6]

For many solver states (tag 0) the critical factor f_c is found with QLKNN (threshold.py).  Each such state is then
re-sampled at f = f_c * U[0.8, 1.5] along the common scaling of R/L_Ti and R/L_Te, half of the copies with a
small (3 %) jitter on the other inputs, and labelled by QLKNN.  The output is the input file with these rows
appended, tag 3.  Machines keep their labels, so the held-out machines stay held out.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml.shrx import qlknn_common as qc  # noqa: E402
from ml.shrx.threshold import ATE, ATI, critical_factor  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
n_states = int(sys.argv[sys.argv.index("--states") + 1]) if "--states" in sys.argv else 160_000
k = int(sys.argv[sys.argv.index("--per-state") + 1]) if "--per-state" in sys.argv else 6
rng = np.random.default_rng(404)

d = np.load(src, allow_pickle=False)
X, Y, tag, mach = d["X"], d["Y"], d["tag"], d["machine"]
real = np.where((tag == 0) & (mach != "box"))[0]
idx = rng.choice(real, min(2 * n_states, len(real)), replace=False)
Xs = X[idx].astype(np.float64)
fc = critical_factor(qc.label, Xs)
keep = np.isfinite(fc)
idx, Xs, fc = idx[keep][:n_states], Xs[keep][:n_states], fc[keep][:n_states]
print(f"{len(idx)} states with a bracketed threshold (of {keep.size} tried)")

new_X, new_m = [], []
for j in range(k):
    f = fc * np.exp(rng.uniform(np.log(0.8), np.log(1.5), len(fc)))
    Z = Xs.copy()
    Z[:, ATI] *= f
    Z[:, ATE] *= f
    if j % 2:
        for col in (2, 3, 4, 5, 6, 7, 8):                    # Ane, Ani, q, smag, x, Ti/Te, log nu*
            Z[:, col] *= 1.0 + 0.03 * rng.standard_normal(len(fc))
    new_X.append(qc.clip_box(Z))
    new_m.append(mach[idx])
Xn = np.concatenate(new_X)
Yn = qc.label(Xn)
mn = np.concatenate(new_m)
print("labelled", len(Xn), "threshold samples; flux range", float(Yn[:, 1].min()), float(Yn[:, 1].max()))
np.savez_compressed(
    dst, X=np.concatenate([X, Xn.astype(np.float32)]), Y=np.concatenate([Y, Yn]),
    tag=np.concatenate([tag, np.full(len(Xn), 3, np.int8)]),
    machine=np.concatenate([mach, mn]),
    source=np.concatenate([d["source"], np.full(len(Xn), "thresh")]) if "source" in d.files else np.full(len(X) + len(Xn), "x"),
    lo=d["lo"], hi=d["hi"])
print("saved", dst)
