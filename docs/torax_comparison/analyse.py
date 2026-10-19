"""Combine the SINTEZ runs, the TORAX runs and the surrogate-vs-QLKNN checks into results.json.

    python analyse.py <workdir> [out.json]

Run with the project's own Python (numpy/scipy/scikit-learn) after

    run_sintez.py        -> <workdir>/sintez_<model>.json   (5 models)
    run_torax.py         -> <workdir>/torax_{main,nosaw,rad}.json/.nc
    torax_frozen.py      -> <workdir>/frozen_{neural_surrogate,scaling_anchored}.json
    export_torax.py      -> <workdir>/torax_export.json
"""
import json
import sys
import time
from pathlib import Path

WORK = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("results.json")
BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402

from tokamak import transport as tr  # noqa: E402
from tokamak.geometry import attach_geometry  # noqa: E402
from tokamak.machines import get_machine  # noqa: E402
from tokamak.scalings import ConfinementInputs, tau_energy  # noqa: E402

MODELS = ["scaling_anchored", "cgm", "neural_surrogate", "bohm_gyrobohm", "gyrobohm"]
ITER = attach_geometry(get_machine("iter"))
R0, A_MIN, B0, V_TOT, KAPPA_A = ITER.R0, ITER.a, ITER.B0, ITER.V, ITER.kappa_a

# ITER Q = 10 design point as used by tokamak.validation.check_integrated
ITER_REF = {"Pfus": 500.0, "Q": 10.0, "Tped": 4.5, "Te": 8.9, "Ti": 8.1, "Te0": 25.0,
            "Wth": 325.0, "tauE": 3.70, "H_factor": 1.00, "betaN": 1.77, "q95": 3.00}

res = {"iter_ref": ITER_REF}

# --------------------------------------------------------------------------- SINTEZ
sintez = {}
for tm in MODELS:
    d = json.load(open(WORK / f"sintez_{tm}.json"))
    L = d["log"][-1]
    P = d["profiles"]
    sintez[tm] = {
        "final": {k: L[k] for k in ["t", "Pfus", "Q", "Palpha", "Paux", "Prad", "Wth", "tauE",
                                    "H_factor", "betaN", "Te", "Ti", "Te0", "Ti0", "ne", "ne0",
                                    "Tped", "q0", "q95", "fBS", "Zeff", "fHePc", "li"]},
        "wall_s": d["wall_s"],
        "disrupted": d["disrupted"],
        "trace": {"t": [r["t"] for r in d["log"]], "Pfus": [r["Pfus"] for r in d["log"]],
                  "Q": [r["Q"] for r in d["log"]]},
        "profiles": {k: P[k] for k in ["rho", "ne", "Te", "Ti", "q", "chi"]},
        "chi_e": d["chi_e"], "chi_i": d["chi_i"],
    }
res["sintez"] = sintez

# --------------------------------------------------------------------------- TORAX
tx = json.load(open(WORK / "torax_export.json"))
torax = {}
for tag, label in [("torax_main", "main"), ("torax_nosaw", "no_sawteeth"), ("torax_rad", "more_radiation")]:
    e = tx[tag]
    s = {k: v[-1] for k, v in e["scalars_t"].items()}
    Te, Ti = np.array(e["T_e"]), np.array(e["T_i"])
    P_alpha, P_aux = s["P_alpha_total"] / 1e6, s["P_aux_total"] / 1e6
    P_rad = -(s["P_radiation_e"] + s["P_bremsstrahlung_e"] + s["P_cyclotron_e"]) / 1e6
    W_th = s["W_thermal_total"] / 1e6
    # SINTEZ's own definitions, so that the table compares like with like:
    #   tau_E = W_th / (P_aux + P_alpha - P_rad),  H98 = tau_E / tau_IPB98(y,2)(P_loss)
    P_loss = P_aux + P_alpha - P_rad
    tau_E = W_th / P_loss
    tau_98 = tau_energy("ipb98y2", ConfinementInputs(
        Ip=15.0, B0=B0, P_loss=P_loss, n_bar20=s["n_e_line_avg"] / 1e20,
        R0=R0, a=A_MIN, kappa_a=KAPPA_A, a_mass=ITER.a_mass))
    torax[label] = {
        "final": {"Pfus": s["P_fusion"] / 1e6, "Q": s["Q_fusion"], "Palpha": P_alpha, "Paux": P_aux,
                  "Prad": P_rad, "Wth": W_th, "tauE": tau_E, "H_factor": tau_E / tau_98,
                  "tauE_torax": s["tau_E"], "H98_torax": s["H98"],
                  "betaN": s["beta_N"], "Te": s["T_e_volume_avg"], "Ti": s["T_i_volume_avg"],
                  "Te0": float(Te[0]), "Ti0": float(Ti[0]), "q0": float(e["q"][0]), "q95": s["q95"],
                  "fBS": s["f_bootstrap"], "li": s["li3"]},
        "sawtooth_crashes": e["sawtooth_crashes"], "wall_s": e["wall_s"], "config": e["config"],
        "trace_t": e["time"],
        "trace_Pfus": [x / 1e6 for x in e["scalars_t"]["P_fusion"]],
        "trace_Wth": [x / 1e6 for x in e["scalars_t"]["W_thermal_total"]],
        "profiles": {k: e[k] for k in ["rho_cell", "rho_face", "T_e", "T_i", "n_e", "q", "smag",
                                       "chi_turb_e", "chi_turb_i"]},
    }
res["torax"] = torax

# ------------------------------------------------- surrogate vs QLKNN on identical states
grid = tr.Grid(n=26, a=A_MIN, V_total=V_TOT, kappa_a=KAPPA_A)   # TORAX's 26 faces are uniform in rho
rho_f = grid.rho


def state_from(Te_c, Ti_c, ne_c, rho_c):
    """Cell/node profiles -> face grid, in SINTEZ units (keV, 1e20 m^-3)."""
    f = lambda y: np.interp(rho_f, rho_c, y)
    return tr.PlasmaState(grid, f(ne_c) / 1e20, f(Te_c), f(Ti_c), np.zeros_like(rho_f))


def nn_and_teacher(state, q):
    chi_nn = tr.chi_neural_surrogate(state, R0, A_MIN, q, True, B0)
    # the teacher the network was trained to imitate: a 30-70 % blend of Bohm/gyro-Bohm
    # and critical-gradient chi (train_transport_surrogate.py); 50 % shown here
    bg = tr.chi_bohm_gyrobohm(state, B0, ITER.a_mass, q, True)
    cg = tr.chi_cgm(state, A_MIN, True, R0=R0)
    return chi_nn, np.clip(0.5 * bg + 0.5 * cg, 0.05, 15.0)


MASK = (rho_f >= 0.25) & (rho_f <= 0.85)      # outside TORAX's inner / outer patches and the pedestal


def metrics(pred, ref):
    p, r = pred[MASK], ref[MASK]
    return {"pearson_r": float(np.corrcoef(p, r)[0, 1]),
            "median_ratio": float(np.median(p / r)),
            "mean_ratio": float(np.mean(p) / np.mean(r)),
            "within_x2": float(np.mean((p / r > 0.5) & (p / r < 2.0)))}


def pack(label, st, q, chi_e, chi_i):
    nn, te = nn_and_teacher(st, q)
    chi_e, chi_i = np.array(chi_e), np.array(chi_i)
    return {"label": label, "rho": rho_f.tolist(), "chi_nn": nn.tolist(), "chi_teacher": te.tolist(),
            "chi_qlknn_e": chi_e.tolist(), "chi_qlknn_i": chi_i.tolist(),
            "Te": st.Te.tolist(), "Ti": st.Ti.tolist(), "q": list(map(float, q)),
            "metrics_vs_e": metrics(nn, chi_e), "metrics_vs_i": metrics(nn, chi_i),
            "teacher_vs_e": metrics(te, chi_e)}


e = tx["torax_main"]
stA = state_from(np.array(e["T_e"]), np.array(e["T_i"]), np.array(e["n_e"]), np.array(e["rho_cell"]))
cross = {"torax_state": pack("A", stA, np.array(e["q"]), e["chi_turb_e"], e["chi_turb_i"])}
for tm, key in [("scaling_anchored", "B2"), ("neural_surrogate", "B1")]:
    f = json.load(open(WORK / f"frozen_{tm}.json"))
    st = state_from(np.array(f["T_e"]), np.array(f["T_i"]), np.array(f["n_e"]), np.array(f["rho_cell"]))
    cross["sintez_" + tm] = pack(key, st, np.array(f["q_torax"]), f["chi_turb_e"], f["chi_turb_i"])
res["cross"] = cross
res["cross_mask"] = [0.25, 0.85]

# ------------------------------------------------------ fidelity to its own teacher
from ml.train_transport_surrogate import generate_transport_dataset  # noqa: E402

Xh, yh = generate_transport_dataset(n_samples=26_000, seed=2026)   # a seed the training never saw
w = tr._load_surrogate_weights()
h1 = np.maximum(0.0, ((Xh - w["scaler_mean"]) / w["scaler_scale"]) @ w["w1"] + w["b1"])
h2 = np.maximum(0.0, h1 @ w["w2"] + w["b2"])
yp = (h2 @ w["w3"] + w["b3"]).ravel()
res["fidelity_to_teacher"] = {
    "n_points": int(len(yh)),
    "r2": float(1 - np.sum((yh - yp) ** 2) / np.sum((yh - yh.mean()) ** 2)),
    "mae": float(np.mean(np.abs(yh - yp))),
    "report_json": json.load(open(BACKEND / "ml/models/neural_transport_surrogate_report.json")),
    "n_parameters": int(sum(w[k].size for k in ["w1", "b1", "w2", "b2", "w3", "b3"])),
}


# ---------------------------------------------------------------- latency (this machine)
def bench(fn, reps=400, trials=9):
    fn()
    out = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        out.append((time.perf_counter() - t0) / reps * 1e6)
    return float(np.median(out)), [float(min(out)), float(max(out))]


g65 = tr.Grid(n=65, a=A_MIN, V_total=V_TOT, kappa_a=KAPPA_A)
s65 = tr.initial_state(g65, ne0=1.0, Te0=10.0, Ti0=9.0)
q65 = 1.0 + 2.1 * g65.rho ** 2
s26 = tr.initial_state(grid, ne0=1.0, Te0=10.0, Ti0=9.0)
q26 = 1.0 + 2.1 * rho_f ** 2
lat = {}
for name, fn in [("our_nn_N65", lambda: tr.chi_neural_surrogate(s65, R0, A_MIN, q65, True, B0)),
                 ("our_nn_N26", lambda: tr.chi_neural_surrogate(s26, R0, A_MIN, q26, True, B0)),
                 ("our_cgm_N65", lambda: tr.chi_cgm(s65, A_MIN, True, R0=R0)),
                 ("our_bgb_N65", lambda: tr.chi_bohm_gyrobohm(s65, B0, ITER.a_mass, q65, True))]:
    lat[name + "_us"], lat[name + "_range"] = bench(fn)
for N in (26, 65):
    lat[f"qlknn_N{N}_us"] = tx["latency"][f"qlknn_jit_us_N{N}"]
    lat[f"qlknn_N{N}_range"] = tx["latency"][f"qlknn_jit_us_N{N}_range"]
res["latency"] = lat
res["versions"] = {"jax": tx["latency"]["jax_version"], "jax_backend": tx["latency"]["jax_backend"],
                   "torax": torax["main"]["config"].get("torax_version"),
                   "numpy": np.__version__, "python": sys.version.split()[0]}

OUT.write_text(json.dumps(res))
print("written", OUT, round(OUT.stat().st_size / 1024), "kB")
