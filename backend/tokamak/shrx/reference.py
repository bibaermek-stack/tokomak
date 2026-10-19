"""The teacher inside SINTEZ: DeepMind's QLKNN driving SINTEZ's transport through the same bridge as SHRX.

Evaluation only.  It needs JAX and ``fusion_surrogates`` (``pip install torax``), so it is not part of
the normal install and is *not* registered in ``tokamak.transport.TRANSPORT_MODELS``; the solver accepts
``transport_model="qlknn_ref"`` when the libraries are present.  Comparing SHRX with this, in the same
simulator, separates the error of the network from every other difference between SINTEZ and TORAX.
"""

from __future__ import annotations

import numpy as np

from . import bridge

_MODEL = None
_PREDICT = None
_NAME_MAP = {"efiITG": "qi_itg", "efiTEM": "qi_tem", "efeITG": "qe_itg", "efeTEM": "qe_tem",
             "efeETG": "qe_etg", "pfeITG": "pfe_itg", "pfeTEM": "pfe_tem"}


def _load():
    global _MODEL, _PREDICT
    if _MODEL is None:
        import jax
        jax.config.update("jax_enable_x64", True)
        from fusion_surrogates.qlknn import qlknn_model
        _MODEL = qlknn_model.QLKNNModel.load_default_model()
        assert tuple(_MODEL.inputs_and_ranges) == bridge.FEATURES
        _PREDICT = jax.jit(_MODEL.predict)
    return _MODEL, _PREDICT


def fluxes(X: np.ndarray) -> np.ndarray:
    """Real QLKNN on (N, 10) inputs -> (N, 3) gyro-Bohm fluxes, combined as TORAX does."""
    import jax.numpy as jnp
    _, predict = _load()
    o = {_NAME_MAP.get(k, k): np.asarray(v).ravel() for k, v in predict(jnp.asarray(X, jnp.float64)).items()}
    qi = o["qi_itg"] + o["qi_tem"]
    qe = o["qe_itg"] + o["qe_tem"] + bridge.ETG_FACTOR * o["qe_etg"]
    return np.stack([qi, qe, o["pfe_itg"] + o["pfe_tem"]], axis=1)


def evaluate(rho, Te, Ti, ne20, q, machine, zeff: float, normni: float = 0.9) -> dict:
    """Same signature and result as :func:`tokamak.shrx.evaluate`, from the real QLKNN."""
    model, _ = _load()
    X, aux = bridge.features(rho, Te, Ti, ne20, q, machine.R0, machine.a, machine.B0, zeff, normni,
                             kappa_edge=machine.kappa_x)
    lo = np.array([max(model.inputs_and_ranges[k].get("min", -1e9), -1e9) for k in bridge.FEATURES])
    hi = np.array([model.inputs_and_ranges[k].get("max", 1e9) for k in bridge.FEATURES])
    lo[0] = lo[1] = 0.0
    Xc = np.clip(X, lo, hi)
    flux = fluxes(Xc)
    chi_i, chi_e = bridge.to_si(flux[:, 0], flux[:, 1], aux, a_mass=machine.a_mass)
    ood = np.any((X < lo) | (X > hi), axis=1)
    j = bridge.FEATURES.index("x")
    inside = (X[:, j] >= lo[j]) & (X[:, j] <= hi[j])
    return {"chi_e": chi_e, "chi_i": chi_i, "gamma_e_gb": flux[:, 2], "ood_fraction": float(ood.mean()),
            "ood_core": float(ood[inside].mean()) if inside.any() else 1.0, "ood": ood, "features": X}
