"""Spherical-tokamak states from the database runs, read by SHRX and by the real QLKNN.   TORAX environment.

    python st_frozen.py <records.csv> <sintez_out.json> <out.json>

Takes the end state of each SINTEZ run (profiles saved by sintez_runner) and asks the two networks for chi_e, chi_i on it.
Two things come out: how much of the plasma lies inside the box QLKNN was trained in (T_i/T_e, q, shear, x), and whether
SHRX and QLKNN agree there.  Agreement means SHRX inherits QLKNN's blind spot; it does not mean either is right.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torax  # noqa: F401  (before the sys.path edit: backend/imas shadows the real imas)

sys.path.append(str(Path(__file__).resolve().parents[3]))
from tokamak import shrx  # noqa: E402
from tokamak.shrx import bridge, reference  # noqa: E402

csv, sintez_path, out_path = sys.argv[1:4]
df = pd.read_csv(csv)
runs = json.load(open(sintez_path))
net = shrx.load()
FLOOR = 0.05 * 1.0001
rows = []
for rec, run in zip(df.itertuples(index=False), runs):
    if not run.get("ok"):
        continue
    p = run["profile"]
    m = SimpleNamespace(R0=float(rec.RGEO), a=float(rec.AMIN), B0=abs(float(rec.BT)), kappa_x=float(rec.KAPPA),
                        a_mass=float(rec.MEFF))
    rho = np.array(p["rho"])
    args = (rho, np.array(p["Te"]), np.array(p["Ti"]), np.array(p["ne"]), np.array(p["q"]), m, 2.0, 0.9)
    s, r = shrx.evaluate(*args), reference.evaluate(*args)
    X = s["features"]
    j = {n: i for i, n in enumerate(bridge.FEATURES)}
    inside = (X[:, j["x"]] >= net.lo[j["x"]]) & (X[:, j["x"]] <= net.hi[j["x"]])
    d = {"TOK": rec.TOK, "SHOT": int(rec.SHOT), "eps": m.a / m.R0, "kappa": m.kappa_x, "n_inside_x": int(inside.sum()),
         "ood_core": float(s["ood_core"]), "Te0": run["Te0"], "Ti0": run["Ti0"], "W_sim_MJ": run["Wth_MJ"], "W_exp_MJ": rec.WTH / 1e6}
    for name in ("Ti_Te", "q", "smag"):
        v = X[inside, j[name]]
        d[f"{name}_min"], d[f"{name}_max"] = (float(v.min()), float(v.max())) if v.size else (None, None)
    d["frac_TiTe_below_box"] = float((X[inside, j["Ti_Te"]] < net.lo[j["Ti_Te"]]).mean()) if inside.any() else None
    for ch in ("e", "i"):
        a, b = s[f"chi_{ch}"][inside], r[f"chi_{ch}"][inside]
        live = (a > FLOOR) | (b > FLOOR)
        d[f"chi_{ch}_floor_share_shrx"] = float((a <= FLOOR).mean()) if a.size else None
        d[f"chi_{ch}_floor_share_qlknn"] = float((b <= FLOOR).mean()) if b.size else None
        d[f"chi_{ch}_median_ratio"] = float(np.median(a[live] / b[live])) if live.any() else None
        d[f"chi_{ch}_median_qlknn"] = float(np.median(b)) if b.size else None
        d[f"chi_{ch}_median_shrx"] = float(np.median(a)) if a.size else None
    rows.append(d)
Path(out_path).write_text(json.dumps(rows))
print(f"{len(rows)} states -> {out_path}")
