"""Derived numbers for the text.  Every figure quoted in the article comes from data/*.json through here or through a {{= }} macro."""
import numpy as np


def make(D):
    V = {}
    S, T, E, I, C = (D.get(k) for k in ("shrx_results", "torax_results", "ensemble", "inloop", "chaos"))
    if S and T:
        runs = S["sintez_runs"]
        f = {k: runs[k]["final"] for k in runs}
        tx = T["torax"]["main"]["final"]
        V["iter"] = {"shrx": f["shrx"], "qlknn": f["qlknn_ref"], "v0": f["shrx_v0"], "anch": f["scaling_anchored"], "cgm": f["cgm"], "torax": tx,
                     "torax_nosaw": T["torax"]["no_sawteeth"]["final"], "torax_rad": T["torax"]["more_radiation"]["final"],
                     "net_err": f["shrx"]["Pfus"] / f["qlknn_ref"]["Pfus"] - 1,
                     "code_gap": f["qlknn_ref"]["Pfus"] / tx["Pfus"] - 1,
                     "vs_torax": f["shrx"]["Pfus"] / tx["Pfus"] - 1,
                     "design": S["iter_ref"]}
        M = S["machines"]
        keys = [k for k in M if "qlknn_ref" in M[k] and "shrx" in M[k]]
        V["machines"] = {"keys": keys}
        for fld in ("Wth", "Te", "tauE"):
            r = np.array([M[k]["shrx"][fld] / max(M[k]["qlknn_ref"][fld], 1e-9) for k in keys])
            V["machines"][fld] = {"median_abs_dev": float(np.median(np.abs(r - 1))), "within30": int(np.sum(np.abs(r - 1) < 0.30)),
                                  "n": len(keys), "ratios": dict(zip(keys, r.tolist()))}
        conv = [k for k in keys if k in ("iter", "demo", "sparc", "jet", "diiid", "east", "kstar", "t15md")]
        V["machines"]["conv"] = conv
        r = np.array([M[k]["shrx"]["Wth"] / max(M[k]["qlknn_ref"]["Wth"], 1e-9) for k in conv])
        V["machines"]["conv_Wth_within30"] = int(np.sum(np.abs(r - 1) < 0.30))
        lat = S["latency"]["one_thread"]
        V["lat"] = {n: {k: lat[f"N{n}"][k]["median_us"] for k in lat[f"N{n}"]} for n in (26, 65)}
        for n in (26, 65):
            V["lat"][n]["speed32"] = lat[f"N{n}"]["qlknn_jit_float32"]["median_us"] / lat[f"N{n}"]["shrx_network"]["median_us"]
            V["lat"][n]["speed64"] = lat[f"N{n}"]["qlknn_jit_float64"]["median_us"] / lat[f"N{n}"]["shrx_network"]["median_us"]
            V["lat"][n]["speed_full"] = lat[f"N{n}"]["qlknn_evaluate_full"]["median_us"] / lat[f"N{n}"]["shrx_evaluate_full"]["median_us"]
    if I:
        names = [f"member{i}" for i in range(5)]
        P = [I["iter"][n]["Pfus"] for n in names]
        V["spread"] = {"iter_min": min(P), "iter_max": max(P), "iter_ens": I["iter"]["ensemble"]["Pfus"], "iter_ship": I["iter"]["shipped"]["Pfus"],
                       "iter_rel": (max(P) - min(P)) / np.mean(P) / 2}
        for m, d in I["spread"].items():
            te = [x["Te"] for x in d["members"].values()]
            V["spread"][m] = {"min": min(te), "max": max(te), "ens": d["ensemble"]["Te"], "ratio": max(te) / min(te)}
        V["floor"] = {m: {"ratio": max(x["Te"] for x in d.values()) / min(x["Te"] for x in d.values())} for m, d in I["floor"].items()}
    import math

    def xnue(te_kev, ne_m3, a):
        te_ev = te_kev * 1e3
        ln_l = 24.0 - math.log(math.sqrt(ne_m3 * 1e-6) / te_ev)
        nu = 2.91e-6 * ne_m3 * 1e-6 * ln_l * te_ev ** -1.5
        cs = math.sqrt(te_kev * 1e3 * 1.602176634e-19 / (2.01410178 * 1.6605390666e-27))
        return nu / (cs / a)
    V["xnue"] = {"ktm": xnue(2.0, 5e19, 0.43), "step": xnue(10.0, 8e19, 2.0), "nstx": xnue(1.0, 5e19, 0.65), "jet": xnue(3.0, 5e19, 1.25),
                 "ktm_curve": {te: xnue(te, 5e19, 0.43) for te in (0.5, 1.0, 2.0, 3.0)}}
    upper = 0.0087218 + 3.5 * 0.0082602            # the teacher's training mean + 3.5 std of XNUE (tokamak/shrx/tglf.py SIGMA_BOX)
    lo_, hi_ = 0.05, 50.0
    for _ in range(60):
        mid = math.sqrt(lo_ * hi_)
        lo_, hi_ = (mid, hi_) if xnue(mid, 5e19, 0.43) > upper else (lo_, mid)
    V["xnue"]["upper"] = upper
    V["xnue"]["te_min_ktm_5e19"] = hi_
    if C:
        V["chaos_t2"] = {m: {k: float(np.interp(1.8, d["t"], v)) for k, v in d["deviation"].items()} for m, d in C.items()}
        V["chaos_t10"] = {m: {k: float(np.interp(10.3, d["t"], v)) for k, v in d["deviation"].items()} for m, d in C.items()}
        V["chaos_t15"] = {m: {k: float(np.interp(15.3, d["t"], v)) for k, v in d["deviation"].items()} for m, d in C.items()}
        V["chaos_plateau"] = {m: {k: float(np.median(np.array(v)[np.array(d["t"]) >= 25.0])) for k, v in d["deviation"].items()} for m, d in C.items()}
    return V
