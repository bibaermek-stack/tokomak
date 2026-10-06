"""Train the disruption-risk model.

    python -m ml.train_disruption        # run from backend/

Data: ``synthetic-fusion-reactor-plasma`` (5000 *synthetic* discharges).
The model is therefore a demonstrator of the pipeline, not a validated
predictor -- the result file and the API both say so.

Method: stratified train / validation / test split *before* any scaling;
the scaler is fitted on the training part only.  Logistic regression and
gradient boosting are compared on the validation set, the winner is scored
once on the untouched test set.  A variant without ``confinement_time_ms``
is also reported, because confinement time is itself a consequence of an
instability and may not be known ahead of one.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
CSV = (BACKEND / "data" / "raw" / "synthetic-fusion-reactor-plasma"
       / "synthetic_fusion_plasma.csv")
OUT = Path(__file__).resolve().parent / "models"

FEATURES = ["magnetic_field_tesla", "plasma_current_ma", "heating_power_mw",
            "electron_density_10_19_m3", "ion_temperature_kev",
            "confinement_time_ms", "h_mode"]
TARGET = "disruption_event"


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["h_mode"] = (df["stability_mode"] == "H-mode").astype(float)
    return df.drop_duplicates("discharge_id").dropna(subset=FEATURES + [TARGET])


def _models():
    return {
        "logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=1000)),
        "gboost": make_pipeline(StandardScaler(),
                                GradientBoostingClassifier(
                                    n_estimators=200, max_depth=3,
                                    learning_rate=0.05, subsample=0.8,
                                    random_state=0)),
    }


def _score(model, X, y) -> dict:
    p = model.predict_proba(X)[:, 1]
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p))}


def run(features: list, seed: int = 0) -> dict:
    df = load()
    X, y = df[features].to_numpy(float), df[TARGET].to_numpy(int)
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    Xva, Xte, yva, yte = train_test_split(
        Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=seed)

    val, fitted = {}, {}
    for name, m in _models().items():
        m.fit(Xtr, ytr)
        fitted[name] = m
        val[name] = _score(m, Xva, yva)
    best = max(val, key=lambda k: val[k]["roc_auc"])
    return {"features": features, "best": best, "model": fitted[best],
            "validation": val, "test": _score(fitted[best], Xte, yte),
            "base_rate": float(y.mean()),
            "n": {"train": len(ytr), "val": len(yva), "test": len(yte)}}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    full = run(FEATURES)
    no_tau = run([f for f in FEATURES if f != "confinement_time_ms"])

    print(f"rows {sum(full['n'].values())}, disruption rate "
          f"{full['base_rate']:.3f}")
    for tag, r in (("with tau_E", full), ("without tau_E", no_tau)):
        print(f"{tag:14s} best={r['best']:8s} "
              f"val AUC={r['validation'][r['best']]['roc_auc']:.3f}  "
              f"TEST AUC={r['test']['roc_auc']:.3f} "
              f"PR-AUC={r['test']['pr_auc']:.3f} "
              f"Brier={r['test']['brier']:.3f}")

    # the shipped model excludes tau_E: with it the label is predicted
    # perfectly (AUC 1.000), a leak of the synthetic generator, not physics
    joblib.dump({"model": no_tau["model"], "features": no_tau["features"]},
                OUT / "disruption.joblib")
    report = {
        "trained_on": "sergiobuilds/synthetic-fusion-reactor-plasma "
                      "(SYNTHETIC data)",
        "validated_on_real_data": False,
        "shipped": "without_confinement_time",
        "note": "with confinement_time the label is predicted perfectly "
                "(leak of the synthetic generator)",
        "with_confinement_time": {k: v for k, v in full.items()
                                  if k != "model"},
        "without_confinement_time": {k: v for k, v in no_tau.items()
                                     if k != "model"},
    }
    (OUT / "disruption_report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
