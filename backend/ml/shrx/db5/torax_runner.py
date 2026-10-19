"""The same database records in TORAX (QLKNN).   TORAX environment.

    python torax_runner.py <records.csv> <sintez_out.json> <out.json> [--limit N] [--shard i/n]

``--shard i/n`` runs every n-th record starting at i, so n processes can share one CPU box (JAX uses one core each here).

Equal footing with the SINTEZ runs: same machine parameters, same density profile (SINTEZ's, prescribed), same pedestal
top (SINTEZ's EPED-like prediction -- TORAX has no predictive pedestal), same electron/ion split of the heating, same Z_eff,
same conducted power.  What differs is the core transport model and the rest of the code.  Circular-section geometry with
the record's elongation (TORAX's analytic geometry), fixed q profile (j ~ (1 - r^2)^2 normalised to Ip), no current diffusion.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torax

csv, sintez_path, out_path = sys.argv[1:4]
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
df = pd.read_csv(csv)
sintez = {(r["TOK"], r["SHOT"], round(r["TIME"], 4)): r for r in json.load(open(sintez_path))}
T_FINAL = float(sys.argv[sys.argv.index("--t-final") + 1]) if "--t-final" in sys.argv else 6.0


def ion_mix(meff: float) -> dict:
    """Main-ion mixture with mean mass meff: H/D, D, or D/T."""
    if meff <= 1.0:
        return {"H": 1.0}
    if meff < 2.0:
        return {"H": 2.0 - meff, "D": meff - 1.0}
    if meff <= 3.0:
        return {"D": 3.0 - meff, "T": meff - 2.0}
    return {"T": 1.0}


def config(r, s):
    zeff = float(r.ZEFF) if np.isfinite(r.ZEFF) and r.ZEFF > 1.0 else 2.0
    prof = s["profile"]
    rho = np.array(prof["rho"])
    ne = np.array(prof["ne"]) * 1e20
    idx = np.arange(0, len(rho), 2)
    f_e = max(0.05, min(0.95, 1.0 - float(s["split_nbi"])))           # electron share of the beam power, SINTEZ's slowing-down split
    ipk = abs(float(r.IP))
    return {
        "plasma_composition": {"main_ion": ion_mix(float(r.MEFF)), "impurity": "C", "Z_eff": zeff},
        "profile_conditions": {
            "Ip": ipk,
            "T_i": {0.0: {0.0: 6.0, 1.0: 0.1}}, "T_i_right_bc": 0.1,
            "T_e": {0.0: {0.0: 6.0, 1.0: 0.1}}, "T_e_right_bc": 0.1,
            "n_e": {0: {float(rho[i]): float(ne[i]) for i in idx}}, "n_e_right_bc": float(ne[-1]),
            "n_e_nbar_is_fGW": False, "normalize_n_e_to_nbar": False,
            "initial_psi_mode": "j", "initial_j_is_total_current": True, "current_profile_nu": 2.0,
        },
        "numerics": {"t_final": T_FINAL, "evolve_ion_heat": True, "evolve_electron_heat": True, "evolve_current": False,
                     "evolve_density": False, "max_dt": 0.2, "chi_timestep_prefactor": 50, "dt_reduction_factor": 3},
        "geometry": {"geometry_type": "circular", "R_major": float(r.RGEO), "a_minor": float(r.AMIN),
                     "B_0": abs(float(r.BT)), "elongation_LCFS": float(r.KAPPA), "n_rho": 25},
        "neoclassical": {"bootstrap_current": {}},
        "sources": {
            "generic_heat": {"gaussian_location": 0.15, "gaussian_width": 0.30, "P_total": float(r.PLTH),
                             "electron_heat_fraction": f_e},
            "ei_exchange": {"Qei_multiplier": 1.0},
            "generic_current": {"fraction_of_total_current": 0.0, "gaussian_width": 0.1, "gaussian_location": 0.3},
        },
        "pedestal": {"model_name": "set_T_ped_n_ped", "set_pedestal": True, "T_i_ped": max(float(s["T_ped"]), 0.05),
                     "T_e_ped": max(float(s["T_ped"]), 0.05), "n_e_ped": float(s["n_ped"]) * 1e20, "rho_norm_ped_top": 0.93},
        "transport": {"model_name": "qlknn", "apply_inner_patch": True, "D_e_inner": 0.25, "V_e_inner": 0.0, "chi_i_inner": 1.0,
                      "chi_e_inner": 1.0, "rho_inner": 0.2, "apply_outer_patch": True, "D_e_outer": 0.1, "V_e_outer": 0.0,
                      "chi_i_outer": 2.0, "chi_e_outer": 2.0, "rho_outer": 0.9, "chi_min": 0.05, "chi_max": 100,
                      "D_e_min": 0.05, "DV_effective": True, "include_ITG": True, "include_TEM": True, "include_ETG": True,
                      "avoid_big_negative_s": True, "An_min": 0.05, "ITG_flux_ratio_correction": 1},
        "solver": {"solver_type": "linear", "use_predictor_corrector": True, "n_corrector_steps": 1, "chi_pereverzev": 30,
                   "D_pereverzev": 15, "use_pereverzev": True},
        "time_step_calculator": {"calculator_type": "chi"},
    }


results = []
rows = list(df.itertuples(index=False))
if "--shard" in sys.argv:
    i, n = (int(v) for v in sys.argv[sys.argv.index("--shard") + 1].split("/"))
    rows = rows[i::n]
if limit:
    rows = rows[:limit]
for r in rows:
    key = (r.TOK, int(r.SHOT), round(float(r.TIME), 4))
    out = {"TOK": r.TOK, "SHOT": int(r.SHOT), "TIME": float(r.TIME), "ok": False}
    s = sintez.get(key)
    if not s or not s.get("ok"):
        out["error"] = "no SINTEZ reference run"
        results.append(out)
        continue
    t0 = time.time()
    try:
        tc = torax.ToraxConfig.from_dict(config(r, s))
        data_tree, _ = torax.run_simulation(tc, progress_bar=False)
        sc = data_tree["/scalars"].to_dataset()
        pr = data_tree["/profiles"].to_dataset().isel(time=-1)
        out.update(ok=True, Wth_MJ=float(sc["W_thermal_total"].values[-1]) / 1e6, Te0=float(pr["T_e"].values[0]),
                   Ti0=float(pr["T_i"].values[0]), Te_avg=float(sc["T_e_volume_avg"].values[-1]),
                   Ti_avg=float(sc["T_i_volume_avg"].values[-1]), tau_E_torax=float(sc["tau_E"].values[-1]),
                   P_SOL=float(sc["P_SOL_total"].values[-1]) / 1e6, q95=float(sc["q95"].values[-1]),
                   dWdt_MW=float(sc["dW_thermal_dt_smoothed"].values[-1]) / 1e6, beta_N=float(sc["beta_N"].values[-1]))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    out["wall_s"] = time.time() - t0
    results.append(out)
    print(r.TOK, int(r.SHOT), "ok" if out["ok"] else out.get("error"), f"{out['wall_s']:.0f}s",
          f"W {out.get('Wth_MJ', float('nan')):.2f} MJ", flush=True)
    Path(out_path).write_text(json.dumps(results))
Path(out_path).write_text(json.dumps(results))
