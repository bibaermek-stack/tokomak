"""Freeze reference values from the real QLKNN / TORAX so the unit tests need no JAX.   TORAX env.

    python make_golden.py <labeled.npz> <sintez_scaling_anchored.json> <out_dir>

Writes ``shrx_flux_golden.npz`` (QLKNN inputs and fluxes) and ``shrx_bridge_golden.npz`` (one TORAX
state: its profiles and the ten QLKNN inputs TORAX itself computed from them).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torax  # noqa: F401

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml.shrx import torax_tools as tt  # noqa: E402

labeled, sintez_json, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

d = np.load(labeled)
rng = np.random.default_rng(7)
real = np.where(d["tag"] == 0)[0]
box = np.where(d["tag"] == 2)[0]
idx = np.concatenate([rng.choice(real, 700, replace=False), rng.choice(box, 100, replace=False)])
np.savez_compressed(out / "shrx_flux_golden.npz", X=d["X"][idx], Y=d["Y"][idx], tag=d["tag"][idx],
                    machine=d["machine"][idx])

src = json.load(open(sintez_json))
tc, cp, geo = tt.run_state(tt.frozen_config(src))
X_t, qi, model = tt.torax_inputs(tc, cp, geo)
np.savez_compressed(
    out / "shrx_bridge_golden.npz",
    rho=np.asarray(geo.rho_face_norm), Te=np.asarray(cp.T_e.face_value()), Ti=np.asarray(cp.T_i.face_value()),
    ne20=np.asarray(cp.n_e.face_value()) / 1e20, q=np.asarray(cp.q_face),
    zeff=float(np.mean(np.asarray(cp.Z_eff_face))),
    normni=float(np.mean(np.asarray(cp.n_i.face_value() / cp.n_e.face_value()))),
    kappa_edge=1.85, R0=tt.R0, a=tt.A_MIN, B0=tt.B0, X_torax=X_t)
print("wrote golden files to", out)
