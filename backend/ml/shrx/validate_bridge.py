"""Check the NumPy bridge (tokamak.shrx.bridge) against TORAX itself.   Run in the TORAX env.

    python validate_bridge.py <dir with sintez_<model>.json> [out.json]

For two SINTEZ end states (the neural run, T_e0 ~ 5.6 keV, and the Q ~ 10 default run) it
holds the profile fixed in TORAX and compares, on TORAX's 26 faces:

  1. the ten QLKNN inputs: TORAX's own vs ``bridge.features`` computed from the same profiles
  2. chi_i, chi_e: TORAX's output  vs  real QLKNN run on *our* inputs and read by ``bridge.to_si``
"""
import json
import sys
from pathlib import Path

import numpy as np
import torax  # noqa: F401  (import before touching sys.path: backend/imas shadows the real imas)

BACKEND = Path(__file__).resolve().parents[2]
sys.path.append(str(BACKEND))

import jax.numpy as jnp  # noqa: E402
from ml.shrx import torax_tools as tt  # noqa: E402
from tokamak.shrx import bridge  # noqa: E402
from torax._src.transport_model import qlknn_transport_model as qt  # noqa: E402

work = Path(sys.argv[1])
out = {}
for name in ("scaling_anchored", "neural_surrogate"):
    src = json.load(open(work / f"sintez_{name}.json"))
    tc, cp, geo = tt.run_state(tt.frozen_config(src))
    X_t, qi_t, model = tt.torax_inputs(tc, cp, geo)

    tm = tc.transport.build_transport_model()
    tp = tc.transport.build_runtime_params(0.0)
    cfg_in = qt.QLKNNRuntimeConfigInputs(transport=tp, Ped_top=0.93, set_pedestal=False)
    turb = tm._combined(cfg_in, geo, cp, 1.0)
    chi_i_ref, chi_e_ref = np.asarray(turb.chi_face_ion), np.asarray(turb.chi_face_el)

    # profiles exactly as TORAX holds them on the faces
    rho = np.asarray(geo.rho_face_norm)
    Te = np.asarray(cp.T_e.face_value()); Ti = np.asarray(cp.T_i.face_value())
    ne20 = np.asarray(cp.n_e.face_value()) / 1e20
    q = np.asarray(cp.q_face)
    zeff = float(np.mean(np.asarray(cp.Z_eff_face)))
    normni = float(np.mean(np.asarray(cp.n_i.face_value() / cp.n_e.face_value())))
    X_m, aux = bridge.features(rho, Te, Ti, ne20, q, tt.R0, tt.A_MIN, tt.B0, zeff, normni, kappa_edge=1.85)

    # (1) inputs
    rel = {}
    sl = slice(2, 24)                      # skip the axis and the very last faces (finite-difference ends)
    for j, key in enumerate(bridge.FEATURES):
        a_, b_ = X_t[sl, j], X_m[sl, j]
        scale = np.maximum(np.abs(a_), 0.05)
        rel[key] = {"median_abs_err": float(np.median(np.abs(a_ - b_) / scale)),
                    "p90_abs_err": float(np.percentile(np.abs(a_ - b_) / scale, 90))}

    # (2) fluxes -> chi.  real QLKNN on both input sets
    def run(X):
        o = model.predict(jnp.asarray(X, dtype=jnp.float64))
        qi = np.asarray(o["qi_itg"]).ravel() + np.asarray(o["qi_tem"]).ravel()
        qe = (np.asarray(o["qe_itg"]).ravel() + np.asarray(o["qe_tem"]).ravel()
              + bridge.ETG_FACTOR * np.asarray(o["qe_etg"]).ravel())
        return qi, qe

    qi_m, qe_m = run(X_m)
    chi_i_m, chi_e_m = bridge.to_si(qi_m, qe_m, aux, a_mass=float(np.asarray(cp.A_i)))
    # TORAX inputs through *our* conversion: isolates the flux -> chi step
    qi_t_, qe_t_ = run(X_t)
    aux_t = dict(aux, Ati=np.asarray(qi_t.lref_over_lti), Ate=np.asarray(qi_t.lref_over_lte))
    chi_i_c, chi_e_c = bridge.to_si(qi_t_, qe_t_, aux_t, a_mass=float(np.asarray(cp.A_i)))

    m = (rho >= 0.25) & (rho <= 0.85)
    ratio = lambda a_, b_: float(np.median(a_[m] / b_[m]))
    spread = lambda a_, b_: float(np.median(np.abs(a_[m] / b_[m] - 1.0)))
    clip = lambda a_: np.clip(a_, 0.05, 100.0)
    out[name] = {
        "inputs": rel,
        "chi_conversion_only": {   # TORAX inputs, our flux->chi
            "chi_e_median_ratio": ratio(chi_e_c, clip(chi_e_ref)), "chi_e_median_abs_dev": spread(chi_e_c, clip(chi_e_ref)),
            "chi_i_median_ratio": ratio(chi_i_c, clip(chi_i_ref)), "chi_i_median_abs_dev": spread(chi_i_c, clip(chi_i_ref))},
        "chi_full_bridge": {       # our inputs + our flux->chi
            "chi_e_median_ratio": ratio(chi_e_m, clip(chi_e_ref)), "chi_e_median_abs_dev": spread(chi_e_m, clip(chi_e_ref)),
            "chi_i_median_ratio": ratio(chi_i_m, clip(chi_i_ref)), "chi_i_median_abs_dev": spread(chi_i_m, clip(chi_i_ref))},
        "chi_ref_range": [float(clip(chi_e_ref)[m].min()), float(clip(chi_e_ref)[m].max())],
    }
    print(name)
    for k, v in rel.items():
        print(f"  {k:9s} median |dX|/max(|X|,.05): {v['median_abs_err']:.3f}   p90: {v['p90_abs_err']:.3f}")
    print("  conversion only :", {k: round(v, 3) for k, v in out[name]["chi_conversion_only"].items()})
    print("  full bridge     :", {k: round(v, 3) for k, v in out[name]["chi_full_bridge"].items()})
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
