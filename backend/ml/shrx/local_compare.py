"""SHRX vs the real QLKNN on identical plasma states: local chi_e(rho), chi_i(rho).   TORAX env.

    python local_compare.py <workdir> <out.json>

<workdir> holds what docs/torax_comparison produced: sintez_<model>.json, torax_export.json.
Three states, as in that comparison --
  A  the converged TORAX ITER state (consistent with QLKNN),
  B1 the end state of SINTEZ's first neural run (flat, T_e0 ~ 5.6 keV),
  B2 the end state of SINTEZ's default run (Q ~ 10, T_e0 ~ 24 keV)
-- and three answers on each: TORAX's own chi (native), the real QLKNN read through our bridge, and SHRX.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torax  # noqa: F401

BACKEND = Path(__file__).resolve().parents[2]
sys.path.append(str(BACKEND))
from ml.shrx import torax_tools as tt  # noqa: E402
from tokamak import shrx  # noqa: E402
from tokamak.machines import get_machine  # noqa: E402
from tokamak.shrx import reference  # noqa: E402
from torax._src.transport_model import qlknn_transport_model as qt  # noqa: E402

work, out = Path(sys.argv[1]), Path(sys.argv[2])
iter_m = get_machine("iter")
clip = lambda a: np.clip(np.asarray(a), 0.05, 100.0)
MASK = lambda rho: (rho >= 0.25) & (rho <= 0.85)


def compare(label, rho, Te, Ti, ne20, q, zeff, normni, native_e, native_i):
    s = shrx.evaluate(rho, Te, Ti, ne20, q, iter_m, zeff, normni)
    r = reference.evaluate(rho, Te, Ti, ne20, q, iter_m, zeff, normni)
    m = MASK(rho)
    d = {"label": label, "rho": rho.tolist(), "Te": Te.tolist(), "Ti": Ti.tolist(), "q": q.tolist()}
    for ch in ("e", "i"):
        sh, rf = clip(s[f"chi_{ch}"]), clip(r[f"chi_{ch}"])
        native = native_e if ch == "e" else native_i
        nt = None if native is None else clip(native)
        d[f"chi_{ch}"] = {"shrx": sh.tolist(), "qlknn_bridge": rf.tolist(),
                          "torax_native": None if nt is None else nt.tolist()}
        for name, ref_ in (("vs_qlknn_bridge", rf), ("vs_torax_native", nt)):
            if ref_ is None:
                continue
            ratio = sh[m] / ref_[m]
            d[f"chi_{ch}"][name] = {
                "median_ratio": float(np.median(ratio)),
                "median_abs_dev": float(np.median(np.abs(ratio - 1))),
                "within_25pct": float(np.mean(np.abs(ratio - 1) < 0.25)),
                "within_x2": float(np.mean((ratio > 0.5) & (ratio < 2.0))),
                "pearson_r": float(np.corrcoef(sh[m], ref_[m])[0, 1]) if np.std(ref_[m]) > 0 else None}
    print(label, {k: round(v, 3) for k, v in d["chi_e"]["vs_qlknn_bridge"].items() if v is not None},
          "| vs TORAX native:", {k: round(v, 3) for k, v in d["chi_e"].get("vs_torax_native", {}).items() if v is not None})
    return d


res = {}
# A: TORAX converged flat-top (compared with QLKNN through the same bridge only)
tx = json.load(open(work / "torax_export.json"))["torax_main"]
rc, rf_ = np.array(tx["rho_cell"]), np.array(tx["rho_face"])
f = lambda y: np.interp(rf_, rc, np.array(y))
res["A"] = compare("A  TORAX flat-top", rf_, f(tx["T_e"]), f(tx["T_i"]), f(tx["n_e"]) / 1e20, np.array(tx["q"]),
                   tx["config"]["Zeff"], 0.925, None, None)   # TORAX's stored chi is from the step before this profile
# (a sawtoothing plasma changes abruptly between the two, and near the critical gradient chi is extremely sensitive)
# B1, B2: frozen SINTEZ end states held in TORAX
for key, name in (("B1", "neural_surrogate"), ("B2", "scaling_anchored")):
    src = json.load(open(work / f"sintez_{name}.json"))
    tc, cp, geo = tt.run_state(tt.frozen_config(src))
    tm = tc.transport.build_transport_model()
    tp = tc.transport.build_runtime_params(0.0)
    turb = tm._combined(qt.QLKNNRuntimeConfigInputs(transport=tp, Ped_top=0.93, set_pedestal=False), geo, cp, 1.0)
    rho = np.asarray(geo.rho_face_norm)
    res[key] = compare(f"{key} SINTEZ end state ({name})", rho, np.asarray(cp.T_e.face_value()),
                       np.asarray(cp.T_i.face_value()), np.asarray(cp.n_e.face_value()) / 1e20,
                       np.asarray(cp.q_face), float(np.mean(np.asarray(cp.Z_eff_face))),
                       float(np.mean(np.asarray(cp.n_i.face_value() / cp.n_e.face_value()))),
                       np.asarray(turb.chi_face_el), np.asarray(turb.chi_face_ion))
out.write_text(json.dumps(res))
