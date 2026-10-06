"""Train a Neural Network (MLP) for plasma equilibrium and shape reconstruction.

Dataset: Fusion Equilibrium Challenge (DIII-D holdout release from Kaggle: anulum/fec-tables-holdout).
Task: Predict real plasma position (r_axis, z_axis), elongation (kappa),
triangularity (tri_top, tri_bot), safety factor (q95), beta_n, and internal inductance (li)
directly from 367 magnetic probe & flux loop sensor features.

Evaluation follows strict ML best practices:
- Group/shot-based split: rows from the same discharge never leak between train, validation, and test.
- Preprocessing pipelines fitted strictly on training data only.
- Evaluation metrics: R^2, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
DATA_PATH = BACKEND / "data" / "raw" / "fec" / "table_holdout.npz"
OUT_DIR = BACKEND / "ml" / "models"

TARGET_NAMES = [
    "r_axis",     # major radius of magnetic axis [m]
    "z_axis",     # vertical position of magnetic axis [m]
    "kappa",      # elongation
    "tri_top",    # upper triangularity
    "tri_bot",    # lower triangularity
    "q95",        # safety factor at 95% flux
    "beta_n",     # normalised beta
    "li",         # internal inductance
]


def load_dataset():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run kaggle download first.")
    npz = np.load(DATA_PATH)
    features = npz["features"]            # (78875, 367)
    labels = npz["labels"]                # (78875, 78)
    label_names = list(npz["label_names"])
    shot_ids = npz["shot_of_row"]          # (78875,)

    target_indices = [label_names.index(name) for name in TARGET_NAMES]
    y = labels[:, target_indices]

    # Clean NaNs or Infs if any
    valid_mask = np.isfinite(features).all(axis=1) & np.isfinite(y).all(axis=1)
    return features[valid_mask], y[valid_mask], shot_ids[valid_mask]


def split_by_shot(X, y, shots, train_ratio=0.7, val_ratio=0.15, seed=42):
    unique_shots = np.unique(shots)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_shots)

    n_shots = len(unique_shots)
    n_tr = int(train_ratio * n_shots)
    n_va = int(val_ratio * n_shots)

    tr_shots = set(perm[:n_tr])
    va_shots = set(perm[n_tr:n_tr + n_va])
    te_shots = set(perm[n_tr + n_va:])

    tr_idx = np.isin(shots, list(tr_shots))
    va_idx = np.isin(shots, list(va_shots))
    te_idx = np.isin(shots, list(te_shots))

    return (X[tr_idx], y[tr_idx]), (X[va_idx], y[va_idx]), (X[te_idx], y[te_idx])


def train_neural_equilibrium(max_iter=40, hidden=(128, 64)):
    print("Loading DIII-D FEC dataset...")
    X, y, shots = load_dataset()
    print(f"Loaded {len(X):,} samples from {len(np.unique(shots))} real tokamak shots.")
    print(f"Input features: {X.shape[1]} magnetics diagnostics; Targets: {len(TARGET_NAMES)}")

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_by_shot(X, y, shots)
    print(f"Train samples: {len(X_tr):,}, Val: {len(X_va):,}, Test: {len(X_te):,}")

    print(f"Training Deep MLP Neural Network {hidden}...")
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="tanh",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            verbose=True,
        )
    )
    model.fit(X_tr, y_tr)

    print("Evaluating Neural Network on unseen test discharges...")
    y_pred_te = model.predict(X_te)

    report = {"targets": {}, "overall": {}}
    for i, name in enumerate(TARGET_NAMES):
        true_col = y_te[:, i]
        pred_col = y_pred_te[:, i]
        mae = float(np.mean(np.abs(true_col - pred_col)))
        rmse = float(np.sqrt(np.mean((true_col - pred_col) ** 2)))
        ss_tot = np.sum((true_col - np.mean(true_col)) ** 2)
        r2 = float(1.0 - (np.sum((true_col - pred_col) ** 2) / max(ss_tot, 1e-9)))

        report["targets"][name] = {"R2": r2, "MAE": mae, "RMSE": rmse}
        print(f"  {name:10s} -> R^2: {r2:6.3f} | MAE: {mae:7.4f} | RMSE: {rmse:7.4f}")

    r2_mean = float(np.mean([m["R2"] for m in report["targets"].values()]))
    report["overall"]["mean_R2"] = r2_mean
    print(f"\nMean Test R^2: {r2_mean:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "target_names": TARGET_NAMES, "input_dim": X.shape[1]},
                OUT_DIR / "neural_equilibrium.joblib")
    (OUT_DIR / "neural_equilibrium_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Model and report successfully saved in {OUT_DIR}!")
    return report


if __name__ == "__main__":
    train_neural_equilibrium()
