"""The real QLKNN as a labeller, shared by the SHRX data scripts.   TORAX environment (JAX).

``label(X)`` runs QLKNN on (N, 10) inputs in the order of :data:`tokamak.shrx.bridge.FEATURES` and returns
the gyro-Bohm fluxes (q_i, q_e, Gamma_e) combined exactly as TORAX combines them.  ``clip_box`` clips inputs to
the range QLKNN was trained on.
"""
import sys
from pathlib import Path

import numpy as np
import torax  # noqa: F401  (before sys.path edits: backend/imas shadows the real imas)

sys.path.append(str(Path(__file__).resolve().parents[2]))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from tokamak.shrx import bridge  # noqa: E402
from torax._src.transport_model import qlknn_transport_model as qt  # noqa: E402

MODEL = qt.get_model("", "qlknn_7_11_v1")
NAMES = list(MODEL.inputs_and_ranges)
assert tuple(NAMES) == bridge.FEATURES, NAMES
LO = np.array([max(MODEL.inputs_and_ranges[k].get("min", -1e9), -1e9) for k in NAMES])
HI = np.array([MODEL.inputs_and_ranges[k].get("max", 1e9) for k in NAMES])
LO[NAMES.index("Ati")] = LO[NAMES.index("Ate")] = 0.0       # QLKNN's 1e-14 floor, as zero
J = {n: i for i, n in enumerate(NAMES)}
_PREDICT = jax.jit(MODEL.predict)


def clip_box(X):
    return np.clip(X, LO, HI)


def label(X, batch=200_000):
    out = []
    for i in range(0, len(X), batch):
        o = _PREDICT(jnp.asarray(X[i:i + batch], dtype=jnp.float64))
        qi = np.asarray(o["qi_itg"]).ravel() + np.asarray(o["qi_tem"]).ravel()
        qe = (np.asarray(o["qe_itg"]).ravel() + np.asarray(o["qe_tem"]).ravel()
              + bridge.ETG_FACTOR * np.asarray(o["qe_etg"]).ravel())
        pf = np.asarray(o["pfe_itg"]).ravel() + np.asarray(o["pfe_tem"]).ravel()
        out.append(np.stack([qi, qe, pf], axis=1))
    return np.concatenate(out).astype(np.float32)
