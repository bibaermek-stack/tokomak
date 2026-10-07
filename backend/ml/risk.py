"""Runtime access to the trained disruption-risk model (lazy, optional)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

MODELS = Path(__file__).resolve().parent / "models"
CAVEAT_SYNTHETIC = ("Модель СИНТЕТИКАЛЫҚ деректерде оқытылған (Kaggle: "
                    "synthetic-fusion-reactor-plasma); нақты токамак разрядтарында "
                    "валидацияланбаған. Бағыттаушы көрсеткіш қана.")
CAVEAT_JTEXT = ("Модель НАҚТЫ J-TEXT токамагының 1 кГц разрядтарында оқытылған "
                "(Kaggle: multi-machine-disruption-prediction-challenge). "
                "Хронологиялық бөлініспен бағаланған.")

CAVEAT_FEC = ("Модель НАҚТЫ DIII-D токамагының магниттік диагностикаларынан (FEC Challenge, "
               "Kaggle: anulum/fec-tables-holdout) плазма тепе-теңдігін (R_axis, Z_axis, "
               "q95, beta_N, kappa, triangularity) қалпына келтіретін терең нейрондық желі (MLP).")

_cache: dict = {}


def _load_synthetic() -> Optional[dict]:
    if "bundle_syn" in _cache:
        return _cache["bundle_syn"]
    path = MODELS / "disruption.joblib"
    bundle = None
    if path.is_file():
        import joblib
        bundle = joblib.load(path)
    _cache["bundle_syn"] = bundle
    return bundle


def _load_jtext() -> Optional[dict]:
    if "bundle_jtext" in _cache:
        return _cache["bundle_jtext"]
    path = MODELS / "jtext_disruption.joblib"
    bundle = None
    if path.is_file():
        import joblib
        bundle = joblib.load(path)
    _cache["bundle_jtext"] = bundle
    return bundle


def _load_jtext_neural() -> Optional[dict]:
    if "bundle_jtext_neural" in _cache:
        return _cache["bundle_jtext_neural"]
    path = MODELS / "jtext_neural_disruption.joblib"
    bundle = None
    if path.is_file():
        import joblib
        bundle = joblib.load(path)
    _cache["bundle_jtext_neural"] = bundle
    return bundle


def _load_equilibrium_neural() -> Optional[dict]:
    if "bundle_eq_neural" in _cache:
        return _cache["bundle_eq_neural"]
    path = MODELS / "neural_equilibrium.joblib"
    bundle = None
    if path.is_file():
        import joblib
        bundle = joblib.load(path)
    _cache["bundle_eq_neural"] = bundle
    return bundle


def _load_transport_surrogate() -> Optional[dict]:
    if "bundle_trans_surrogate" in _cache:
        return _cache["bundle_trans_surrogate"]
    path = MODELS / "neural_transport_surrogate.joblib"
    bundle = None
    if path.is_file():
        import joblib
        bundle = joblib.load(path)
    _cache["bundle_trans_surrogate"] = bundle
    return bundle


CAVEAT_TORAX = ("DeepMind TORAX стиліндегі 1D ядролық турбулентті тасымал суррогаты (MLP). "
                "Жергілікті градиенттерден (R/L_T, q, s) sub-millisecond ішінде chi(rho) есептейді.")


def status() -> dict:
    try:
        bsyn = _load_synthetic()
        bjtext = _load_jtext()
        b_jtext_nn = _load_jtext_neural()
        b_eq_nn = _load_equilibrium_neural()
        b_trans = _load_transport_surrogate()
        err = None
    except ImportError as exc:
        bsyn, bjtext, b_jtext_nn, b_eq_nn, b_trans, err = None, None, None, None, None, f"missing dependency: {exc.name}"

    rep_syn = MODELS / "disruption_report.json"
    rep_jtext = MODELS / "jtext_report.json"
    rep_eq = MODELS / "neural_equilibrium_report.json"
    rep_trans = MODELS / "neural_transport_surrogate_report.json"

    return {
        "synthetic_model_available": bsyn is not None,
        "jtext_real_model_available": bjtext is not None,
        "jtext_neural_network_available": b_jtext_nn is not None,
        "neural_equilibrium_model_available": b_eq_nn is not None,
        "neural_transport_surrogate_available": b_trans is not None,
        "error": err,
        "synthetic_report": json.loads(rep_syn.read_text(encoding="utf-8"))
        if rep_syn.is_file() else None,
        "jtext_report": json.loads(rep_jtext.read_text(encoding="utf-8"))
        if rep_jtext.is_file() else None,
        "neural_equilibrium_report": json.loads(rep_eq.read_text(encoding="utf-8"))
        if rep_eq.is_file() else None,
        "neural_transport_surrogate_report": json.loads(rep_trans.read_text(encoding="utf-8"))
        if rep_trans.is_file() else None,
        "caveats": {
            "synthetic": CAVEAT_SYNTHETIC,
            "jtext": CAVEAT_JTEXT,
            "fec": CAVEAT_FEC,
            "torax": CAVEAT_TORAX,
        },
    }


def disruption_risk(B: float, Ip: float, P_heat: float, ne19: float,
                    Ti_kev: float, h_mode: bool) -> float:
    """Probability of a disruption in [0, 1] using synthetic-trained model."""
    bundle = _load_synthetic()
    if bundle is None:
        raise LookupError("no trained model; run python -m ml.train_disruption")
    values = {"magnetic_field_tesla": B, "plasma_current_ma": Ip,
              "heating_power_mw": P_heat, "electron_density_10_19_m3": ne19,
              "ion_temperature_kev": Ti_kev, "h_mode": float(h_mode)}
    x = [[values[f] for f in bundle["features"]]]
    return float(bundle["model"].predict_proba(x)[0, 1])


def jtext_risk_from_features(feat_dict: dict, use_neural: bool = False) -> float:
    """Risk from raw J-TEXT 1kHz feature mapping or precomputed vector."""
    bundle = _load_jtext_neural() if use_neural else _load_jtext()
    if bundle is None:
        bundle = _load_jtext()
    if bundle is None:
        raise LookupError("no trained J-TEXT model; run python -m ml.train_jtext")
    clf = bundle["model"]
    features = bundle["features"]
    row = [feat_dict.get(f, 0.0) for f in features]
    return float(clf.predict_proba([row])[0, 1])


def predict_equilibrium_neural(features_367) -> dict:
    """Reconstruct equilibrium parameters from 367 DIII-D magnetic diagnostics."""
    import numpy as np
    bundle = _load_equilibrium_neural()
    if bundle is None:
        raise LookupError("no trained equilibrium neural network; run python -m ml.train_equilibrium_neural")
    model = bundle["model"]
    names = bundle["target_names"]
    arr = np.asarray(features_367, float)
    pred = model.predict(np.atleast_2d(arr))[0]
    return {name: float(val) for name, val in zip(names, pred)}


def predict_transport_surrogate(rho: float, r_lt: float, r_ln: float, q: float,
                                Te: float, ne: float, Ti_over_Te: float = 1.0,
                                B0: float = 5.3, R0: float = 6.2, a: float = 2.0,
                                h_mode: bool = True) -> float:
    """Evaluates turbulent thermal diffusivity chi [m^2/s] with DeepMind TORAX style neural surrogate."""
    import numpy as np
    bundle = _load_transport_surrogate()
    if bundle is None:
        raise LookupError("no trained transport surrogate model; run python -m ml.train_transport_surrogate")
    model = bundle["model"]
    scaler = bundle["scaler"]
    x = np.array([[rho, r_lt, r_ln, q, Te, ne, Ti_over_Te, B0, R0, a, 1.0 if h_mode else 0.0]], dtype=float)
    x_scaled = scaler.transform(x)
    pred = float(model.predict(x_scaled)[0])
    return float(np.clip(pred, 0.05, 15.0))

