"""How far outside its training range does the TGLFNN-STEP teacher find itself on SINTEZ's spherical-tokamak states?   NumPy only.

    python tools/collect_st.py <data_dir>      (states_machines.npz, states_records.npz from backend/ml/shrx/st/collect_tglf_states.py)

For every group of states (the scripted KTM / NSTX / NSTX-U / MAST-U discharges and the NSTX/MAST/START database records), restricted
to radial points inside the teacher's radial range (0.1 < r/a < 0.95): the share of points with any input beyond mean +- 3.5 std of the
teacher's training inputs, the same per input, the median z-score, and a histogram of log10 of the collisionality input.
"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[1] / "data" / "st_teacher.json"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from tokamak.shrx import tglf  # noqa: E402

# training mean / std of the 15 inputs, from the weights file's stats (read once with the restricted loader, no network needed)
net = tglf.load()
mu, sd = net.mu.astype(float), net.sd.astype(float)
names = list(tglf.FEATURES)
J = {n: i for i, n in enumerate(names)}
bins = np.linspace(-4.0, 4.0, 81)
res = {"features": names, "mu": mu.tolist(), "sd": sd.tolist(), "sigma_box": tglf.SIGMA_BOX, "log10_bins": bins.tolist(),
       "xnue_train_mean": float(mu[J["XNUE"]]), "xnue_train_std": float(sd[J["XNUE"]]), "groups": {}}


def group(name):
    n = str(name).lower()
    if n.startswith("mast-") or n.startswith("mast"):
        return "records MAST" if n != "mastu" else "mastu"
    for k in ("nstxu", "nstx", "start", "ktm"):
        if n.startswith(k):
            return k if k in ("nstxu", "ktm") else f"records {k.upper()}" if "-" in n else k
    return n


for f in ("states_machines.npz", "states_records.npz"):
    d = np.load(D / "tglf" / f)
    X, src = d["X"].astype(float), d["source"]
    inside = (X[:, J["RMIN_LOC"]] > 0.1) & (X[:, J["RMIN_LOC"]] < 0.95)
    z = (X - mu) / sd
    labels = np.array([("scripted " + s) if f.startswith("states_m") else ("records " + str(s).split("-")[0]) for s in src])
    for g in sorted(set(labels)):
        m = (labels == g) & inside
        if m.sum() == 0:
            continue
        per = {n: {"outside": float((np.abs(z[m, j]) > tglf.SIGMA_BOX).mean()), "median_z": float(np.median(z[m, j]))} for j, n in enumerate(names)}
        xn = np.log10(np.maximum(X[m, J["XNUE"]], 1e-6))
        h, _ = np.histogram(np.clip(xn, bins[0], bins[-1] - 1e-9), bins=bins)
        res["groups"][g] = {"rows": int(m.sum()), "any_outside": float((np.abs(z[m]) > tglf.SIGMA_BOX).any(axis=1).mean()), "per_feature": per,
                            "xnue_hist": (h / h.sum()).tolist(), "xnue_median": float(np.median(X[m, J["XNUE"]]))}
OUT.write_text(json.dumps(res))
for g, v in res["groups"].items():
    print(g, v["rows"], round(v["any_outside"], 2), "XNUE median %.3g (train %.4f ± %.4f)" % (v["xnue_median"], res["xnue_train_mean"], res["xnue_train_std"]))
