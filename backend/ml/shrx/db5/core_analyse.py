"""Central temperatures with the pedestal calibrated to the measured stored energy ("core only").   pandas/NumPy.

    python core_analyse.py <records.csv> <out.json> --runs name=core_sintez.json [name2=core_torax.json ...]

W_th is matched by construction (the pedestal-top temperature is adjusted), so the only free outcome is how the core
profile is shaped: central T_e0 and T_i0 against the measured TE0 / TI0.  Ratio = predicted / measured; reported as median,
scatter of ln(ratio) about its mean, and the share within +-30 %.  Records whose calibration did not converge (|ln W_pred/W_exp|
> 5 %) or whose pedestal hit the search bound are listed apart and excluded.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality import valid  # noqa: E402

csv, out_path = sys.argv[1], sys.argv[2]
runs = {}
i = sys.argv.index("--runs") + 1
while i < len(sys.argv) and "=" in sys.argv[i]:
    k, v = sys.argv[i].split("=", 1)
    runs[k] = {(r["TOK"], r["SHOT"], round(r["TIME"], 4)): r for r in json.load(open(v))}
    i += 1
df = pd.read_csv(csv)
keys = list(zip(df.TOK, df.SHOT, df.TIME.round(4)))
exp = {k: row for k, row in zip(keys, df.to_dict("records"))}
for row in exp.values():                 # the database gives central temperatures in eV
    row["TE0"], row["TI0"] = row["TE0"] / 1e3, row["TI0"] / 1e3


def stats(ratio):
    r = np.asarray([x for x in ratio if np.isfinite(x) and x > 0])
    if len(r) == 0:
        return None
    ln = np.log(r)
    return {"n": int(len(r)), "median_ratio": float(np.median(r)), "rms_ln_about_mean": float(ln.std()),
            "within_30pct": float(np.mean(np.abs(r - 1) < 0.3)), "p10": float(np.percentile(r, 10)), "p90": float(np.percentile(r, 90))}


def converged(r):
    """Stored energy matched, pedestal inside the search range, and the run itself valid (density, power, no runaway)."""
    k = (r["TOK"], r["SHOT"], round(r["TIME"], 4))
    return (bool(r.get("ok")) and abs(r.get("calibration_residual", 0.0)) < 0.05 and 0.04 < r.get("T_ped_calibrated", 0) < 39.0
            and valid(r, exp[k]))


res = {"n_records": len(df), "models": {}}
for name, rr in runs.items():
    good = {k: r for k, r in rr.items() if converged(r)}
    m = {"n_ok": sum(1 for r in rr.values() if r.get("ok")), "n_calibrated": len(good), "stats": {}, "by_machine": {}}
    for field, col, label in (("Te0", "TE0", "Te0"), ("Ti0", "TI0", "Ti0")):
        ratio = [r[field] / exp[k][col] for k, r in good.items() if np.isfinite(exp[k][col]) and exp[k][col] > 0 and k in exp]
        m["stats"][label] = stats(ratio)
    # temperature ratio Te0/Ti0: measured vs predicted, where both exist
    both = [k for k in good if np.isfinite(exp[k]["TE0"]) and np.isfinite(exp[k]["TI0"])]
    if both:
        m["stats"]["Te0_over_Ti0_ratio_of_ratios"] = stats([(good[k]["Te0"] / good[k]["Ti0"]) / (exp[k]["TE0"] / exp[k]["TI0"]) for k in both])
    for tok in sorted(set(k[0] for k in good)):
        ks = [k for k in good if k[0] == tok]
        e = [good[k]["Te0"] / exp[k]["TE0"] for k in ks if np.isfinite(exp[k]["TE0"])]
        t = [good[k]["Ti0"] / exp[k]["TI0"] for k in ks if np.isfinite(exp[k]["TI0"])]
        m["by_machine"][tok] = {"n": len(ks), "Te0_median_ratio": float(np.median(e)) if e else None,
                                "Ti0_median_ratio": float(np.median(t)) if t else None,
                                "T_ped_median_keV": float(np.median([good[k]["T_ped_calibrated"] for k in ks]))}
    m["not_calibrated"] = [f"{k[0]}-{k[1]}@{k[2]}" for k, r in rr.items() if not converged(r)]
    res["models"][name] = m
if len(runs) >= 2:
    names = list(runs)
    a, b = names[0], names[1]
    common = [k for k in runs[a] if k in runs[b] and converged(runs[a][k]) and converged(runs[b][k])]
    cmp_ = {"n_common": len(common)}
    for field in ("Te0", "Ti0", "Wth_MJ"):
        cmp_[f"{a}/{b} {field}"] = stats([runs[a][k][field] / runs[b][k][field] for k in common])
    res["comparison"] = cmp_
Path(out_path).write_text(json.dumps(res, indent=1))
for name, m in res["models"].items():
    print(f"{name}: {m['n_calibrated']}/{m['n_ok']} calibrated")
    for lab, s in m["stats"].items():
        if s:
            print(f"   {lab:32s} n={s['n']:3d} median {s['median_ratio']:.2f}  scatter(ln) {s['rms_ln_about_mean']:.2f}  within30% {s['within_30pct']:.2f}")
    for tok, v in m["by_machine"].items():
        print("     ", tok, {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items()})
if "comparison" in res:
    print(json.dumps(res["comparison"], indent=1))
