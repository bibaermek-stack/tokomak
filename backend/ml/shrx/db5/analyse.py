"""Predicted vs measured thermal stored energy on the ITPA H-mode database records.   NumPy/pandas/matplotlib only.

    python analyse.py <records.csv> <out.json> --runs name=sintez.json [name=sintez2.json ...] [--torax name=torax.json ...]

For every code/model: the ratio W_pred / W_exp per record, summarised as the median ratio, the scatter of ln(ratio)
(RMS about its mean) and the share within +-30 %.  The IPB98(y,2) scaling -- a regression *fitted to this very data* -- is the
yardstick: it is what a model that knows the database achieves.  Per-machine rows are medians.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality import valid  # noqa: E402

csv, out_path = sys.argv[1], sys.argv[2]


def parse(flag):
    if flag not in sys.argv:
        return {}
    i = sys.argv.index(flag) + 1
    d = {}
    while i < len(sys.argv) and "=" in sys.argv[i] and not sys.argv[i].startswith("--"):
        k, v = sys.argv[i].split("=", 1)
        d[k] = json.load(open(v))
        i += 1
    return d


sintez, torax = parse("--runs"), parse("--torax")
df = pd.read_csv(csv)
df["key"] = list(zip(df.TOK, df.SHOT, df.TIME.round(4)))
assert df.key.is_unique, "records must be unique per (TOK, SHOT, TIME)"
WTH = dict(zip(df.key, df.WTH))
PLTH = dict(zip(df.key, df.PLTH))
KEYS = list(df.key)


REC = {k: row for k, row in zip(df.key, df.to_dict("records"))}


def table(runs, field, only_valid=False):
    rows = {}
    for r in runs:
        k = (r["TOK"], r["SHOT"], round(r["TIME"], 4))
        if r.get("ok") and (not only_valid or valid(r, REC[k])):
            rows[k] = r[field]
    return rows


VALID_KEYS = {}


def stats(ratio):
    ratio = np.asarray([x for x in ratio if np.isfinite(x) and x > 0])
    if len(ratio) == 0:
        return None
    ln = np.log(ratio)
    return {"n": int(len(ratio)), "median_ratio": float(np.median(ratio)), "mean_ln": float(ln.mean()),
            "rms_ln_about_mean": float(ln.std()), "rms_ln_about_zero": float(np.sqrt((ln ** 2).mean())),
            "within_30pct": float(np.mean(np.abs(ratio - 1) < 0.30)), "p10": float(np.percentile(ratio, 10)),
            "p90": float(np.percentile(ratio, 90))}


res = {"n_records": int(len(df)), "records": {}, "by_machine": {}, "stats": {}}
# IPB98(y,2): measured tau / scaling = HIPB98Y2 (database column); the scaling's own ratio is 1/H
ipb = 1.0 / df.HIPB98Y2.replace(0, np.nan)
res["stats"]["IPB98y2_fit"] = stats(ipb.values)

preds = {}
for name, runs in sintez.items():
    t = table(runs, "Wth_MJ")
    preds[f"SINTEZ+{name}"] = {k: v * 1e6 for k, v in t.items()}
    VALID_KEYS[f"SINTEZ+{name}"] = set(table(runs, "Wth_MJ", only_valid=True))
for name, runs in torax.items():
    t = table(runs, "Wth_MJ")
    preds[f"TORAX({name})"] = {k: v * 1e6 for k, v in t.items()}
for name, p in preds.items():
    ratio = [p[k] / WTH[k] for k in KEYS if k in p]
    res["stats"][name] = stats(ratio)
    if name in VALID_KEYS:
        res["stats"][name + " [valid runs only]"] = stats([p[k] / WTH[k] for k in KEYS if k in VALID_KEYS[name]])
        # the yardstick on the same records
        res["stats"]["IPB98y2_fit [same records]"] = stats([ipb[i] for i, k in enumerate(KEYS) if k in VALID_KEYS[name]])

# per record table and the agreement between codes
for k in KEYS:
    rec = {"WTH_MJ": float(WTH[k] / 1e6), "PLTH_MW": float(PLTH[k] / 1e6)}
    for name, p in preds.items():
        if k in p:
            rec[name] = p[k] / 1e6
    res["records"][f"{k[0]}-{k[1]}@{k[2]}"] = rec
names = list(preds)
if len(names) >= 2:
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            common = [k for k in KEYS if k in preds[names[i]] and k in preds[names[j]]]
            r = np.array([preds[names[i]][k] / preds[names[j]][k] for k in common])
            if len(r):
                res["stats"][f"{names[i]} / {names[j]}"] = {"n": len(r), "median_ratio": float(np.median(r)),
                                                           "rms_ln_about_mean": float(np.log(r).std())}
for tok, g in df.groupby("TOK"):
    res["by_machine"][tok] = {"n": int(len(g)), "median_H98_measured": float(np.nanmedian(g.HIPB98Y2))}
    for name, p in preds.items():
        rr = [p[k] / WTH[k] for k in g.key if k in p]
        if rr:
            res["by_machine"][tok][name] = float(np.median(rr))
Path(out_path).write_text(json.dumps(res, indent=1))
print(f"{'':36s} {'n':>4s} {'median':>7s} {'RMS(ln)':>8s} {'+-30%':>6s}")
for k, v in res["stats"].items():
    if v and "within_30pct" in v:
        print(f"{k:36s} {v['n']:4d} {v['median_ratio']:7.2f} {v['rms_ln_about_mean']:8.2f} {v['within_30pct']:6.2f}")
