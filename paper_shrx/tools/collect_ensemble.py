"""Ensemble data for the article: error distributions, spread-vs-error and threshold arrays.   NumPy; the threshold part needs the TORAX env.

    python tools/collect_ensemble.py <labeled.npz> <ensemble.npz> <member0.npz> <out.json> [--threshold] [--n 6000]

Held-out machines only (NSTX, DEMO).  ``err`` is |chi_e(network) - chi_e(QLKNN)| / chi_e(QLKNN) in gyro-Bohm units on states where QLKNN's
chi_e is above the floor.  ``--threshold`` adds f_c(QLKNN), f_c(member0), f_c(ensemble) for a random sample (backend/ml/shrx/threshold.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

BACKEND = Path(__file__).resolve().parents[2] / "backend"
labeled, ens_path, mem_path, out_path = sys.argv[1:5]
n_thr = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 6000
if "--threshold" in sys.argv:
    import torax  # noqa: F401  (before the sys.path edit: backend/imas shadows the real imas)
sys.path.insert(0, str(BACKEND))
from tokamak.shrx import Shrx  # noqa: E402
from tokamak.shrx.ensemble import ShrxEnsemble  # noqa: E402

ens, single = ShrxEnsemble(Path(ens_path)), Shrx(Path(mem_path))
d = np.load(labeled)
X, Y, tag, mach = d["X"], d["Y"], d["tag"], d["machine"]
FLOOR = 0.1
out = {"floor_gb": FLOOR, "n_members": ens.K}
q = np.linspace(0, 1, 201)


def gb(flux, A):
    return flux / np.maximum(A, 1e-7)


for name in ("nstx", "demo"):
    sel = (mach == name) & (tag == 0)
    Xs, Ys = X[sel], Y[sel]
    Ate = Xs[:, 1]
    on = (gb(Ys[:, 1], Ate) > FLOOR) & (Ate > 2.0)
    Xs, Ys, Ate = Xs[on], Ys[on], Ate[on]
    truth = gb(Ys[:, 1], Ate)
    mem = ens.predict_members(Xs)
    c_mem = gb(mem[:, :, 1], Ate[None])
    c_ens = gb(mem.mean(0)[:, 1], Ate)
    c_one = gb(single.predict(Xs.astype(np.float32))[:, 1], Ate)
    e_ens, e_one = np.abs(c_ens - truth) / truth, np.abs(c_one - truth) / truth
    spread = c_mem.std(0) / np.maximum(c_mem.mean(0), 1e-9)
    dec = np.quantile(spread, np.linspace(0, 1, 11))
    per_dec = [float(np.median(e_ens[(spread >= lo) & (spread <= hi)])) for lo, hi in zip(dec[:-1], dec[1:])]
    big = e_ens > 0.25
    r = rankdata(spread)
    auroc = float((r[big].sum() - big.sum() * (big.sum() + 1) / 2) / (big.sum() * (~big).sum()))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(spread), min(3000, len(spread)), replace=False)
    out[name] = {"n_on": int(on.sum()), "err_quantiles_ensemble": np.quantile(e_ens, q).tolist(), "err_quantiles_single": np.quantile(e_one, q).tolist(),
                 "median_err_ensemble": float(np.median(e_ens)), "median_err_single": float(np.median(e_one)),
                 "within25_ensemble": float(np.mean(e_ens < 0.25)), "within25_single": float(np.mean(e_one < 0.25)),
                 "spread_deciles": dec.tolist(), "median_err_per_decile": per_dec, "auroc": auroc,
                 "spearman": float(np.corrcoef(rankdata(spread), rankdata(e_ens))[0, 1]),
                 "scatter_spread": spread[idx].tolist(), "scatter_err": e_ens[idx].tolist()}
    print(name, out[name]["median_err_ensemble"], out[name]["median_err_single"], auroc)

if "--threshold" in sys.argv:
    from ml.shrx import qlknn_common as qc  # noqa: E402
    from ml.shrx.threshold import critical_factor  # noqa: E402
    rng = np.random.default_rng(1)
    for name, sel in (("held_out", np.isin(mach, ["nstx", "demo"]) & (tag == 0)),
                      ("train", ~np.isin(mach, ["nstx", "demo"]) & (tag == 0) & (mach != "box"))):
        idx = rng.choice(np.where(sel)[0], min(n_thr, int(sel.sum())), replace=False)
        Xs = X[idx].astype(np.float64)
        fq = critical_factor(qc.label, Xs)
        fs = critical_factor(lambda a: single.predict(a.astype(np.float32)), Xs)
        fe = critical_factor(lambda a: ens.predict(a.astype(np.float32)), Xs)
        ok = np.isfinite(fq) & np.isfinite(fs) & np.isfinite(fe)
        out[f"threshold_{name}"] = {"n": int(len(idx)), "n_both": int(ok.sum()), "err_single": (fs[ok] / fq[ok] - 1).tolist(),
                                    "err_ensemble": (fe[ok] / fq[ok] - 1).tolist(), "fc_qlknn": fq[ok].tolist()}
        print(name, "threshold median |err| single/ensemble", np.median(np.abs(fs[ok] / fq[ok] - 1)), np.median(np.abs(fe[ok] / fq[ok] - 1)))
Path(out_path).write_text(json.dumps(out))
