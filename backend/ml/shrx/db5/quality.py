"""Is a SINTEZ database run a fair test of the transport model?   NumPy only.

A run is *valid* when it actually reproduced the discharge it is compared with: the line-averaged density within 20 % of the
record (the density feedback can only add particles), the power crossing the separatrix within 20 % of PLTH (the code's own
ohmic power can exceed what a small machine really has), and a plasma that is not in thermal runaway (beta_N < 3.5, bootstrap
fraction < 0.8 -- no H-mode in the database is beyond either).  The gate looks at the run's own sanity, not at the stored
energy it is about to be compared with.
"""
TOL = 0.20
BETA_N_MAX = 3.5
F_BS_MAX = 0.8


def flags(run: dict, rec: dict) -> dict:
    n_ratio = run["n_bar"] / (rec["NEL"] / 1e20)
    p_ratio = run["P_loss"] / (rec["PLTH"] / 1e6)
    return {"density_ok": abs(n_ratio - 1) < TOL, "power_ok": abs(p_ratio - 1) < TOL,
            "no_runaway": run["betaN"] < BETA_N_MAX and run["f_bs"] < F_BS_MAX, "n_ratio": n_ratio, "p_ratio": p_ratio}


def valid(run: dict, rec: dict) -> bool:
    if not run.get("ok"):
        return False
    f = flags(run, rec)
    return f["density_ok"] and f["power_ok"] and f["no_runaway"]
