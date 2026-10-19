"""Freeze the experimental-database results (SINTEZ + SHRX, TORAX, IPB98(y,2), spherical tokamaks) into data/db5.json.

    python tools/collect_db5.py <data_dir>

<data_dir> holds what backend/ml/shrx/db5 wrote: sel_*.csv, conv_sintez_shrx.json, valid_torax_*.json, core_sintez_shrx.json, core_torax_*.json,
st_sintez_shrx.json, st_frozen.json, st_sintez_tglf.json (the last is optional).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[1] / "data" / "db5.json"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend" / "ml" / "shrx" / "db5"))
from quality import flags, valid  # noqa: E402

key = lambda r: (r["TOK"], int(r["SHOT"]), round(float(r["TIME"]), 4))


def load_many(pattern):
    rows = {}
    for p in sorted(D.glob(pattern)):
        for r in json.loads(p.read_text()):
            if r.get("ok"):
                rows[key(r)] = r
    return rows


def stats(ratio):
    r = np.asarray([x for x in ratio if np.isfinite(x) and x > 0])
    if len(r) == 0:
        return None
    ln = np.log(r)
    return {"n": int(len(r)), "median": float(np.median(r)), "sigma_ln": float(ln.std()), "mean_ln": float(ln.mean()),
            "f30": float(np.mean(np.abs(r - 1) < 0.30)), "p10": float(np.percentile(r, 10)), "p90": float(np.percentile(r, 90))}


out = {}
# --------------------------------------------------------------------------------------------------- conventional aspect ratio
df = pd.read_csv(D / "sel_conv.csv")
runs = json.loads((D / "conv_sintez_shrx.json").read_text())
torax = load_many("valid_torax_*.json")
recs = []
for rec, run in zip(df.to_dict("records"), runs):
    k = key({"TOK": rec["TOK"], "SHOT": rec["SHOT"], "TIME": rec["TIME"]})
    d = {"tok": rec["TOK"], "shot": int(rec["SHOT"]), "time": float(rec["TIME"]), "W_exp": rec["WTH"] / 1e6, "P_lth": rec["PLTH"] / 1e6,
         "H98": None if not np.isfinite(rec["HIPB98Y2"]) or rec["HIPB98Y2"] <= 0 else float(rec["HIPB98Y2"]),
         "eps": rec["AMIN"] / rec["RGEO"], "ok": bool(run.get("ok"))}
    if run.get("ok"):
        f = flags(run, rec)
        d.update(W_shrx=run["Wth_MJ"], betaN=run["betaN"], f_bs=run["f_bs"], n_ratio=f["n_ratio"], p_ratio=f["p_ratio"],
                 valid=bool(valid(run, rec)), T_ped=run["T_ped"], Te0=run["Te0"], Ti0=run["Ti0"], dens_ok=bool(f["density_ok"]),
                 pow_ok=bool(f["power_ok"]), no_runaway=bool(f["no_runaway"]))
        if k in torax:
            d["W_torax"] = torax[k]["Wth_MJ"]
            d["Te0_torax"], d["Ti0_torax"] = torax[k]["Te0"], torax[k]["Ti0"]
    recs.append(d)
out["records"] = recs
out["n_conv"] = len(recs)
out["n_st"] = int(len(pd.read_csv(D / "sel_st.csv")))
R = lambda sel, f: [f(d) for d in recs if sel(d)]
ok = lambda d: d.get("ok")
vl = lambda d: d.get("valid")
out["stats"] = {
    "shrx_all": stats(R(ok, lambda d: d["W_shrx"] / d["W_exp"])),
    "shrx_valid": stats(R(vl, lambda d: d["W_shrx"] / d["W_exp"])),
    "ipb_all": stats(R(lambda d: d["H98"], lambda d: 1.0 / d["H98"])),
    "ipb_on_valid": stats(R(lambda d: vl(d) and d["H98"], lambda d: 1.0 / d["H98"])),
    "torax_valid": stats(R(lambda d: "W_torax" in d, lambda d: d["W_torax"] / d["W_exp"])),
    "shrx_on_torax_set": stats(R(lambda d: "W_torax" in d, lambda d: d["W_shrx"] / d["W_exp"])),
    "ipb_on_torax_set": stats(R(lambda d: "W_torax" in d and d["H98"], lambda d: 1.0 / d["H98"])),
}
both = [d for d in recs if "W_torax" in d]
if len(both) > 2:
    a, b = np.log([d["W_shrx"] for d in both]), np.log([d["W_torax"] for d in both])
    out["shrx_vs_torax"] = {"n": len(both), "median_ratio": float(np.exp(np.median(a - b))), "sigma_ln": float((a - b).std()),
                            "corr_ln": float(np.corrcoef(a, b)[0, 1]), "f30": float(np.mean(np.abs(np.exp(a - b) - 1) < 0.30))}
mach = {}
for tok in sorted(set(d["tok"] for d in recs)):
    sub = [d for d in recs if d["tok"] == tok]
    m = {"n": len(sub), "eps": float(np.median([d["eps"] for d in sub])),
         "n_valid": sum(1 for d in sub if d.get("valid")), "dens_ok": sum(1 for d in sub if d.get("dens_ok")),
         "pow_ok": sum(1 for d in sub if d.get("pow_ok")), "no_runaway": sum(1 for d in sub if d.get("no_runaway")),
         "H98_median": float(np.nanmedian([d["H98"] if d["H98"] else np.nan for d in sub])) if any(d["H98"] for d in sub) else None}
    allr = [d["W_shrx"] / d["W_exp"] for d in sub if ok(d)]
    m["shrx_all_median"] = float(np.median(allr)) if allr else None
    vr = [d["W_shrx"] / d["W_exp"] for d in sub if vl(d)]
    m["shrx_valid_median"] = float(np.median(vr)) if vr else None
    tr = [d["W_torax"] / d["W_exp"] for d in sub if "W_torax" in d]
    m["torax_median"] = float(np.median(tr)) if tr else None
    m["n_torax"] = len(tr)
    mach[tok] = m
out["by_machine"] = mach

# --------------------------------------------------------------------------------------------------- core only
dfc = pd.read_csv(D / "sel_core.csv")
core_runs = json.loads((D / "core_sintez_shrx.json").read_text())
core_torax = load_many("core_torax_*.json")
cores = []
for rec, run in zip(dfc.to_dict("records"), core_runs):
    if not run.get("ok"):
        continue
    good = abs(run["calibration_residual"]) < 0.05 and 0.04 < run["T_ped_calibrated"] < 39.0 and valid(run, rec)
    te, ti = [v / 1e3 if np.isfinite(v) and v > 200.0 else None for v in (rec["TE0"], rec["TI0"])]
    k = key({"TOK": rec["TOK"], "SHOT": rec["SHOT"], "TIME": rec["TIME"]})
    d = {"tok": rec["TOK"], "shot": int(rec["SHOT"]), "good": bool(good), "Te0_exp": te, "Ti0_exp": ti, "Te0": run["Te0"], "Ti0": run["Ti0"],
         "W_exp": rec["WTH"] / 1e6, "W_shrx": run["Wth_MJ"], "T_ped": run["T_ped_calibrated"], "resid": run["calibration_residual"],
         "split_nbi": run["split_nbi"]}
    if good and k in core_torax:
        d["Te0_torax"], d["Ti0_torax"], d["W_torax"] = core_torax[k]["Te0"], core_torax[k]["Ti0"], core_torax[k]["Wth_MJ"]
    cores.append(d)
out["core"] = cores
out["core_counts"] = {"n_runs": len(core_runs), "n_good": sum(1 for d in cores if d["good"]), "n_torax": sum(1 for d in cores if "W_torax" in d),
                      "dens_ok": sum(1 for rec, run in zip(dfc.to_dict("records"), core_runs) if run.get("ok") and flags(run, rec)["density_ok"]),
                      "pow_ok": sum(1 for rec, run in zip(dfc.to_dict("records"), core_runs) if run.get("ok") and flags(run, rec)["power_ok"]),
                      "no_runaway": sum(1 for rec, run in zip(dfc.to_dict("records"), core_runs) if run.get("ok") and flags(run, rec)["no_runaway"])}
def cstats(field_pred, field_exp, sel=lambda d: True):
    return stats([d[field_pred] / d[field_exp] for d in cores if d["good"] and sel(d) and d.get(field_exp) and d.get(field_pred) is not None])
out["core_stats"] = {"Te0_shrx": cstats("Te0", "Te0_exp"), "Ti0_shrx": cstats("Ti0", "Ti0_exp"),
                     "Te0_torax": cstats("Te0_torax", "Te0_exp"), "Ti0_torax": cstats("Ti0_torax", "Ti0_exp")}
pair = [d for d in cores if "Te0_torax" in d]
out["core_pair"] = {"n": len(pair),
                    "Te0_shrx_over_torax": stats([d["Te0"] / d["Te0_torax"] for d in pair]), "Ti0_shrx_over_torax": stats([d["Ti0"] / d["Ti0_torax"] for d in pair]),
                    "W_torax_over_exp": stats([d["W_torax"] / d["W_exp"] for d in pair])}
ti_te = [(d["Ti0"] / d["Te0"], d["Ti0_exp"] / d["Te0_exp"]) for d in cores if d["good"] and d["Ti0_exp"] and d["Te0_exp"]]
out["core_ti_te"] = {"n": len(ti_te), "pred_median": float(np.median([a for a, _ in ti_te])) if ti_te else None,
                     "exp_median": float(np.median([b for _, b in ti_te])) if ti_te else None}

# --------------------------------------------------------------------------------------------------- spherical tokamaks
dfs = pd.read_csv(D / "sel_st.csv")
st_runs = json.loads((D / "st_sintez_shrx.json").read_text())
st = []
for rec, run in zip(dfs.to_dict("records"), st_runs):
    if run.get("ok"):
        st.append({"tok": rec["TOK"], "shot": int(rec["SHOT"]), "eps": rec["AMIN"] / rec["RGEO"], "kappa": rec["KAPPA"], "W_exp": rec["WTH"] / 1e6,
                   "W_shrx": run["Wth_MJ"], "betaN": run["betaN"], "Te0": run["Te0"], "Ti0": run["Ti0"], "ood": run["shrx_ood"], "T_ped": run["T_ped"]})
out["st"] = st
out["st_stats"] = {"shrx": stats([d["W_shrx"] / d["W_exp"] for d in st]),
                   **{tok: stats([d["W_shrx"] / d["W_exp"] for d in st if d["tok"] == tok]) for tok in sorted(set(d["tok"] for d in st))}}
frozen = json.loads((D / "st_frozen.json").read_text())
fz = {}
for tok in sorted(set(r["TOK"] for r in frozen)):
    sub = [r for r in frozen if r["TOK"] == tok]
    med = lambda k: float(np.nanmedian([r[k] if r[k] is not None else np.nan for r in sub]))
    fz[tok] = {"n": len(sub), "eps": med("eps"), "ood_core": med("ood_core"), "tite_below": med("frac_TiTe_below_box"),
               "chi_e_shrx": med("chi_e_median_shrx"), "chi_e_qlknn": med("chi_e_median_qlknn"),
               "chi_i_shrx": med("chi_i_median_shrx"), "chi_i_qlknn": med("chi_i_median_qlknn"),
               "floor_e_shrx": med("chi_e_floor_share_shrx"), "floor_e_qlknn": med("chi_e_floor_share_qlknn"), "Te0": med("Te0"), "Ti0": med("Ti0")}
out["st_frozen"] = fz
tg = D / "st_sintez_tglf.json"
if tg.is_file():
    tr = json.loads(tg.read_text())
    out["st_tglf"] = []
    for rec, run in zip(dfs.to_dict("records"), tr):
        if run.get("ok"):
            out["st_tglf"].append({"tok": rec["TOK"], "shot": int(rec["SHOT"]), "W_exp": rec["WTH"] / 1e6, "W_tglf": run["Wth_MJ"], "betaN": run["betaN"],
                                   "Te0": run["Te0"], "Ti0": run["Ti0"], "ood": run["shrx_ood"], "T_ped": run["T_ped"]})
    out["st_tglf_stats"] = {"all": stats([d["W_tglf"] / d["W_exp"] for d in out["st_tglf"]]),
                            **{tok: stats([d["W_tglf"] / d["W_exp"] for d in out["st_tglf"] if d["tok"] == tok]) for tok in sorted(set(d["tok"] for d in out["st_tglf"]))}}
OUT.write_text(json.dumps(out))
print(json.dumps({k: out[k] for k in ("n_conv", "n_st", "stats", "shrx_vs_torax", "core_counts", "core_stats", "core_pair") if k in out}, indent=1)[:3500])
