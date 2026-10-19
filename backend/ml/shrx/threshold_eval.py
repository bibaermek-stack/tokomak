"""Threshold error of an SHRX weight file against QLKNN.   TORAX environment.

    python threshold_eval.py <labeled.npz> <weights.npz|weights_dir> [--n 40000] [--out out.json]

Takes solver states (tag 0) from the held-out machines (nstx, demo) and from the training machines, finds f_c with
QLKNN and with SHRX (see threshold.py) and reports the distribution of (f_c SHRX / f_c QLKNN - 1).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml.shrx import qlknn_common as qc  # noqa: E402
from ml.shrx.threshold import critical_factor  # noqa: E402
from tokamak.shrx import Shrx  # noqa: E402


def evaluate(labeled, weights, n=40000, seed=0):
    d = np.load(labeled)
    X, tag, mach = d["X"], d["tag"], d["machine"]
    net = Shrx(Path(weights))
    rng = np.random.default_rng(seed)
    out = {}
    for name, sel in (("held_out_machines", np.isin(mach, ["nstx", "demo"]) & (tag == 0)),
                      ("training_machines", ~np.isin(mach, ["nstx", "demo"]) & (tag == 0) & (mach != "box"))):
        idx = rng.choice(np.where(sel)[0], min(n, int(sel.sum())), replace=False)
        Xs = X[idx].astype(np.float64)
        f_q = critical_factor(qc.label, Xs)
        f_s = critical_factor(lambda a: net.predict(a.astype(np.float32)), Xs)
        ok = np.isfinite(f_q) & np.isfinite(f_s)
        err = f_s[ok] / f_q[ok] - 1.0
        out[name] = {"n_states": len(idx), "n_bracketed_both": int(ok.sum()),
                     "bracketed_qlknn_not_shrx": int((np.isfinite(f_q) & ~np.isfinite(f_s)).sum()),
                     "median_abs_err": float(np.median(np.abs(err))), "mean_err": float(err.mean()),
                     "p90_abs_err": float(np.percentile(np.abs(err), 90)),
                     "share_within_3pct": float(np.mean(np.abs(err) < 0.03)),
                     "share_within_10pct": float(np.mean(np.abs(err) < 0.10))}
        print(f"{name:20s}", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()})
    return out


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 40000
    res = evaluate(sys.argv[1], sys.argv[2], n)
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(res, indent=1))
