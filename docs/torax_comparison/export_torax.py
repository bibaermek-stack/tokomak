"""Dump the TORAX runs to plain JSON and benchmark QLKNN latency.

    python export_torax.py <workdir>

Run inside the TORAX environment.  Reads torax_{main,nosaw,rad}.nc from <workdir>,
writes <workdir>/torax_export.json.
"""
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

WORK = Path(sys.argv[1])
SCALARS = ["P_fusion", "Q_fusion", "W_thermal_total", "tau_E", "H98", "beta_N", "q95",
           "T_e_volume_avg", "T_i_volume_avg", "P_alpha_total", "P_radiation_e",
           "P_bremsstrahlung_e", "P_cyclotron_e", "P_SOL_total", "P_aux_total", "P_heat_total",
           "f_bootstrap", "li3", "n_e_volume_avg", "n_e_line_avg", "fgw_n_e_line_avg", "q_min"]

out = {}
for tag in ["torax_main", "torax_nosaw", "torax_rad"]:
    dt = xr.open_datatree(WORK / f"{tag}.nc")
    pr = dt["/profiles"].to_dataset()
    sc = dt["/scalars"].to_dataset()
    last = pr.isel(time=-1)
    meta = json.load(open(WORK / f"{tag}.json"))
    out[tag] = {
        "wall_s": meta["wall_s"], "config": meta["config_summary"],
        "time": pr["time"].values.tolist(),
        "scalars_t": {k: sc[k].values.tolist() for k in SCALARS},
        "rho_cell": last["rho_norm"].values.tolist(),
        "rho_face": last["rho_face_norm"].values.tolist(),
        "T_e": last["T_e"].values.tolist(), "T_i": last["T_i"].values.tolist(),
        "n_e": last["n_e"].values.tolist(), "q": last["q"].values.tolist(),
        "smag": last["magnetic_shear"].values.tolist(),
        "chi_turb_e": last["chi_turb_e"].values.tolist(),
        "chi_turb_i": last["chi_turb_i"].values.tolist(),
        "sawtooth_crashes": int(np.sum(dt["/numerics"]["sawtooth_crash"].values)),
    }

# --- QLKNN latency: jitted forward pass of the model TORAX loads by default -----------
from torax._src.transport_model import qlknn_transport_model as qt  # noqa: E402

model = qt.get_model("", "qlknn_7_11_v1")
names = list(model.inputs_and_ranges.keys())
lat = {"input_names": names, "jax_backend": jax.default_backend(), "jax_version": jax.__version__}
for N in (26, 65):
    rng = np.random.default_rng(0)
    cols = []
    for k in names:
        b = model.inputs_and_ranges[k]
        lo, hi = b.get("min", 0.0), b.get("max", 1.0)
        lo = lo if np.isfinite(lo) else -1.0
        hi = hi if np.isfinite(hi) else 1.0
        cols.append(rng.uniform(lo + 0.3 * (hi - lo), lo + 0.7 * (hi - lo), N))
    X = jnp.asarray(np.stack(cols, axis=1), dtype=jnp.float32)
    f = jax.jit(model.predict)
    jax.block_until_ready(f(X))
    trials = []
    for _ in range(9):                       # median of 9 x 500 calls: CPU timing is noisy
        t0 = time.perf_counter()
        for _ in range(500):
            r = f(X)
        jax.block_until_ready(r)
        trials.append((time.perf_counter() - t0) / 500 * 1e6)
    lat[f"qlknn_jit_us_N{N}"] = float(np.median(trials))
    lat[f"qlknn_jit_us_N{N}_range"] = [float(min(trials)), float(max(trials))]
out["latency"] = lat
(WORK / "torax_export.json").write_text(json.dumps(out))
print(json.dumps(lat, indent=1))
