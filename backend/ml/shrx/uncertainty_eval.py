"""Does the ensemble spread tell where SHRX is wrong?   NumPy only.

    python uncertainty_eval.py <labeled.npz> <ensemble.npz> <out.json> [--single member.npz]

Held-out machines only (nstx, demo; never seen in training).  For every state where QLKNN's chi_e is above twice the floor
("turbulence on") compare, in gyro-Bohm units,

  * the error of the ensemble mean and of each single member against QLKNN,
  * the relative spread of chi_e across members (std / mean),

and report how well the spread ranks the errors (Spearman rho, AUROC for "error above 25 %"), what share of the large
errors falls in the top 20 % of spread, and the median error per spread decile.  The spread threshold used to flag a point
"uncertain" is calibrated on one held-out machine and tested on the other, and vice versa.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokamak.shrx.ensemble import ShrxEnsemble  # noqa: E402

labeled, ens_path, out_path = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
ens = ShrxEnsemble(ens_path)
d = np.load(labeled)
X, Y, tag, mach = d["X"], d["Y"], d["tag"], d["machine"]
FLOOR = 0.1                      # gyro-Bohm chi_e below this is "off" (about twice TORAX's floor in an ITER-like unit)


def chi_gb(q, A):
    return q / np.maximum(A, 1e-7)


def analyse(sel):
    Xs, Ys = X[sel], Y[sel]
    mem = ens.predict_members(Xs)                         # (K, N, 3)
    Ate = Xs[:, 1]
    c_true = chi_gb(Ys[:, 1], Ate)
    c_mem = chi_gb(mem[:, :, 1], Ate[None])
    c_mean = chi_gb(mem.mean(0)[:, 1], Ate)
    on = (c_true > FLOOR) & (Ate > 2.0)
    ct, cm, cmean = c_true[on], c_mem[:, on], c_mean[on]
    err_mean = np.abs(cmean - ct) / ct
    err_mem = np.abs(cm - ct[None]) / ct[None]
    spread = cm.std(0) / np.maximum(cm.mean(0), 1e-9)
    return {"on": int(on.sum()), "err_mean": err_mean, "err_mem": err_mem, "spread": spread}


def auroc(score, positive):
    r = rankdata(score)
    n1, n0 = positive.sum(), (~positive).sum()
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


res = {"n_members": ens.K, "floor_gb": FLOOR}
groups = {"nstx": (mach == "nstx") & (tag == 0), "demo": (mach == "demo") & (tag == 0),
          "heldout_perturbed": np.isin(mach, ["nstx", "demo"]) & (tag == 1)}
A = {k: analyse(v) for k, v in groups.items()}
for name, a in A.items():
    e, s = a["err_mean"], a["spread"]
    big = e > 0.25
    top = s >= np.quantile(s, 0.8)
    deciles = np.quantile(s, np.linspace(0, 1, 11))
    per_dec = [float(np.median(e[(s >= lo) & (s <= hi)])) if ((s >= lo) & (s <= hi)).any() else None
               for lo, hi in zip(deciles[:-1], deciles[1:])]
    res[name] = {"n_on": a["on"],
                 "median_err_mean": float(np.median(e)), "median_err_single": float(np.median(a["err_mem"])),
                 "within25_mean": float(np.mean(e < 0.25)), "within25_single": float(np.mean(a["err_mem"] < 0.25)),
                 "spearman_spread_vs_err": spearman(s, e), "auroc_err_gt_25": auroc(s, big) if 0 < big.sum() < len(big) else None,
                 "share_of_large_errors_in_top20pct_spread": float(top[big].mean() * 0 + (big & top).sum() / max(big.sum(), 1)),
                 "share_large": float(big.mean()), "median_err_per_spread_decile": per_dec}

# two-fold threshold calibration across the two held-out machines
def best_threshold(s, e):
    big = e > 0.25
    best, arg = -1, 0.25
    for t in np.quantile(s, np.linspace(0.3, 0.97, 60)):
        flagged = s > t
        tp = (flagged & big).sum()
        prec, rec = tp / max(flagged.sum(), 1), tp / max(big.sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best:
            best, arg = f1, float(t)
    return arg


cal = {}
for fit, test in (("nstx", "demo"), ("demo", "nstx")):
    t = best_threshold(A[fit]["spread"], A[fit]["err_mean"])
    s, e = A[test]["spread"], A[test]["err_mean"]
    flagged, big = s > t, e > 0.25
    cal[f"calibrated_on_{fit}_tested_on_{test}"] = {
        "threshold": t, "flagged_share": float(flagged.mean()),
        "precision": float((flagged & big).sum() / max(flagged.sum(), 1)),
        "recall": float((flagged & big).sum() / max(big.sum(), 1)),
        "median_err_flagged": float(np.median(e[flagged])) if flagged.any() else None,
        "median_err_unflagged": float(np.median(e[~flagged]))}
res["calibration"] = cal
res["unc_threshold_suggested"] = float(np.mean([v["threshold"] for v in cal.values()]))
Path(out_path).write_text(json.dumps(res, indent=1))
for k in ("nstx", "demo", "heldout_perturbed"):
    r = res[k]
    print(f"{k:18s} n={r['n_on']:6d} err(mean ens)={r['median_err_mean']:.3f} err(single)={r['median_err_single']:.3f} "
          f"within25 {r['within25_mean']:.2f}/{r['within25_single']:.2f}  spearman={r['spearman_spread_vs_err']:.2f} "
          f"AUROC={r['auroc_err_gt_25']} top20%-share={r['share_of_large_errors_in_top20pct_spread']:.2f}")
print(json.dumps(res["calibration"], indent=1))
