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


def status() -> dict:
    try:
        bsyn = _load_synthetic()
        bjtext = _load_jtext()
        err = None
    except ImportError as exc:
        bsyn, bjtext, err = None, None, f"missing dependency: {exc.name}"

    rep_syn = MODELS / "disruption_report.json"
    rep_jtext = MODELS / "jtext_report.json"

    return {
        "synthetic_model_available": bsyn is not None,
        "jtext_real_model_available": bjtext is not None,
        "error": err,
        "synthetic_report": json.loads(rep_syn.read_text(encoding="utf-8"))
        if rep_syn.is_file() else None,
        "jtext_report": json.loads(rep_jtext.read_text(encoding="utf-8"))
        if rep_jtext.is_file() else None,
        "caveats": {
            "synthetic": CAVEAT_SYNTHETIC,
            "jtext": CAVEAT_JTEXT,
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


def jtext_risk_from_features(feat_dict: dict) -> float:
    """Risk from raw J-TEXT 1kHz feature mapping or precomputed vector."""
    bundle = _load_jtext()
    if bundle is None:
        raise LookupError("no trained J-TEXT model; run python -m ml.train_jtext")
    clf = bundle["model"]
    features = bundle["features"]
    row = [feat_dict.get(f, 0.0) for f in features]
    return float(clf.predict_proba([row])[0, 1])
