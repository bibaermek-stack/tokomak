"""Accuracy of the shipped SHRX weights against QLKNN on the labelled data (NumPy only).

    python accuracy.py <labeled.npz> <out.json> [--n-scatter 4000]

Per machine and per kind of sample: R^2 of the gyro-Bohm fluxes and of chi (gyro-Bohm units, where
R/L_T >= 2), median relative error of chi where turbulence is on (QLKNN chi above twice the floor)
and the share of those points within 25 %.  nstx and demo were held out of training.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokamak import shrx  # noqa: E402
from tokamak.shrx import bridge  # noqa: E402

HELD_OUT = ("nstx", "demo")
data = np.load(sys.argv[1])
out_path = Path(sys.argv[2])
n_scatter = int(sys.argv[sys.argv.index("--n-scatter") + 1]) if "--n-scatter" in sys.argv else 4000
net = shrx.load()
X, Y, tag, mach = data["X"], data["Y"], data["tag"], data["machine"]
P = net.predict(X)
floor = 2 * 0.05 / bridge.chi_gb(8.0, 5.3, 2.0)       # "turbulence is on": above twice TORAX's chi_min (ITER-like unit)
cmax = 100.0 / bridge.chi_gb(8.0, 5.3, 2.0)


def stats(sel):
    res = {"n": int(sel.sum())}
    for k, ch in ((0, "i"), (1, "e")):
        A = X[sel][:, k]
        t, p = Y[sel][:, k], P[sel][:, k]
        res[f"r2_flux_{ch}"] = float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))
        a = A >= 2.0
        ct, cp = t[a] / A[a], p[a] / A[a]
        res[f"r2_chi_{ch}"] = float(1 - np.sum((ct - cp) ** 2) / np.sum((ct - ct.mean()) ** 2))
        ct = np.clip(t / np.maximum(A, bridge.EPS), 0.05 / bridge.chi_gb(8.0, 5.3, 2.0), cmax)
        cp = np.clip(p / np.maximum(A, bridge.EPS), 0.05 / bridge.chi_gb(8.0, 5.3, 2.0), cmax)
        on = ct > floor
        rel = np.abs(cp[on] - ct[on]) / ct[on]
        res[f"median_rel_err_{ch}"] = float(np.median(rel)) if on.any() else None
        res[f"within25_{ch}"] = float(np.mean(rel < 0.25)) if on.any() else None
        res["turbulent_share"] = float(on.mean())
    return res


out = {"groups": {}, "per_machine": {}}
out["groups"]["held_out_solver_states"] = stats(np.isin(mach, HELD_OUT) & (tag == 0))
out["groups"]["held_out_perturbed"] = stats(np.isin(mach, HELD_OUT) & (tag == 1))
out["groups"]["train_machines_solver_states"] = stats(~np.isin(mach, HELD_OUT) & (tag == 0) & (mach != "box"))
out["groups"]["all_solver_states"] = stats(tag == 0)
out["groups"]["box_uniform"] = stats(tag == 2)
for m in sorted(set(mach[tag == 0])):
    out["per_machine"][m] = {**stats((mach == m) & (tag == 0)), "held_out": m in HELD_OUT}

# a scatter sample (held-out machines, solver states, turbulence on): the electron heat flux q_e the network outputs
rng = np.random.default_rng(1)
sel = np.where(np.isin(mach, HELD_OUT) & (tag == 0) & (Y[:, 1] > 0.15))[0]
pick = rng.choice(sel, min(n_scatter, len(sel)), replace=False)
out["scatter_flux_e"] = {"qlknn": Y[pick][:, 1].tolist(), "shrx": P[pick][:, 1].tolist(), "machine": mach[pick].tolist()}
out_path.write_text(json.dumps(out))
for k, v in out["groups"].items():
    print(f"{k:32s} n={v['n']:7d}  R2 flux i/e {v['r2_flux_i']:.4f}/{v['r2_flux_e']:.4f}   R2 chi i/e {v['r2_chi_i']:.4f}/{v['r2_chi_e']:.4f}   "
          f"med err i/e {v['median_rel_err_i']}/{v['median_rel_err_e']}")
