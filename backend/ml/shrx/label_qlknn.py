"""Label SHRX training inputs with the real QLKNN (TORAX's turbulent-transport network).

    python label_qlknn.py <states_dir> <out.npz> [--extra other_labeled.npz ...]

Run in the TORAX environment (JAX).  Reads ``states_<machine>_<model>.npz`` written by
collect_states.py and builds three kinds of inputs:

  tag 0  states the solver really visited (clipped to QLKNN's valid box)
  tag 1  the same states with the gradients, q, shear, collisionality and Ti/Te perturbed
  tag 2  uniform samples of the box, so the network is not blind between the visited states

and records QLKNN's gyro-Bohm fluxes for each, combined exactly as TORAX combines them
(q_i = ITG + TEM;  q_e = ITG + TEM + ETG/3;  Gamma_e = ITG + TEM).
"""
import sys
from pathlib import Path

import numpy as np
import torax  # noqa: F401  (before sys.path edits: backend/imas shadows the real imas)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from tokamak.shrx import bridge  # noqa: E402

from ml.shrx.qlknn_common import J, LO, HI, NAMES, MODEL, clip_box, label  # noqa: E402

rng = np.random.default_rng(2026)
names = NAMES


def perturb(X, k):
    """k perturbed copies of every row (labels come from QLKNN, so any input is valid)."""
    j = {n: i for i, n in enumerate(names)}
    copies = []
    for _ in range(k):
        Y = X.copy()
        common = np.exp(rng.uniform(np.log(0.5), np.log(1.8), (len(X), 1)))
        for n in ("Ati", "Ate"):
            Y[:, j[n]] *= common[:, 0] * np.exp(rng.normal(0, 0.25, len(X)))
        for n in ("Ane", "Ani"):
            Y[:, j[n]] = Y[:, j[n]] * np.exp(rng.normal(0, 0.3, len(X))) + rng.normal(0, 0.25, len(X))
        Y[:, j["q"]] *= np.exp(rng.normal(0, 0.12, len(X)))
        Y[:, j["smag"]] += rng.normal(0, 0.3, len(X))
        Y[:, j["LogNuStar"]] += rng.normal(0, 0.3, len(X))
        Y[:, j["Ti_Te"]] *= np.exp(rng.normal(0, 0.1, len(X)))
        Y[:, j["x"]] += rng.normal(0, 0.02, len(X))
        copies.append(Y)
    return np.concatenate(copies)


def box_samples(n):
    u = lambda lo, hi: rng.uniform(lo, hi, n)
    Ane = u(-1.0, 6.0)
    return np.stack([
        u(0, 1) ** 1.5 * 16.0,                 # Ati: denser near the threshold
        u(0, 1) ** 1.5 * 16.0,                 # Ate
        Ane,
        Ane * rng.uniform(0.8, 1.2, n) + rng.normal(0, 0.2, n),
        np.exp(u(np.log(0.7), np.log(8.0))),   # q
        u(-0.5, 3.5),                          # smag
        u(0.1, 0.95),                          # x
        np.exp(u(np.log(0.4), np.log(2.0))),   # Ti/Te
        u(-4.0, 0.4),                          # log10 nu*
        u(0.75, 1.0),                          # n_i / n_e
    ], axis=1)


files = sorted(Path(sys.argv[1]).glob("states_*_*.npz"))
Xr, mach, model_of = [], [], []
for f in files:
    d = np.load(f)
    Xr.append(d["X"])
    parts = f.stem.split("_", 2)             # states_<machine>_<model>
    mach += [parts[1]] * len(d["X"])
    model_of += [parts[2]] * len(d["X"])
Xr = clip_box(np.concatenate(Xr).astype(np.float64))
mach, model_of = np.array(mach), np.array(model_of)
print(f"{len(files)} state files, {len(Xr)} real radial samples")

Xa = perturb(Xr, 3)
Xb = box_samples(400_000)
X = np.concatenate([Xr, Xa, clip_box(Xb)])
tag = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xa)), np.full(len(Xb), 2)]).astype(np.int8)
mach_all = np.concatenate([mach, np.tile(mach, 3), np.full(len(Xb), "box")])
src_all = np.concatenate([model_of, np.tile(model_of, 3), np.full(len(Xb), "box")])
Xa = clip_box(Xa)
X = np.concatenate([Xr, Xa, clip_box(Xb)])
Y = label(X)
print("labelled", len(X), "samples; flux ranges  qi %.3g..%.3g  qe %.3g..%.3g  pfe %.3g..%.3g" % (
    Y[:, 0].min(), Y[:, 0].max(), Y[:, 1].min(), Y[:, 1].max(), Y[:, 2].min(), Y[:, 2].max()))
np.savez_compressed(sys.argv[2], X=X.astype(np.float32), Y=Y, tag=tag, machine=mach_all, source=src_all,
                    lo=LO.astype(np.float32), hi=HI.astype(np.float32))
print("saved", sys.argv[2])
