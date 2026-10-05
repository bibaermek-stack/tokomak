"""Train a disruption predictor on real J-TEXT shots.

    python -m ml.kaggle_fetch --jtext     # once, from backend/
    python -m ml.train_jtext

Protocol
* Split **by shot, chronologically**: earliest 60 % train, next 20 %
  validation, latest 20 % test.  Rows of one shot never straddle splits, and
  the test shots are the ones a deployed model would see *after* training.
* Label: a sample is positive if it lies within ``WARN_WINDOW`` before the
  disruption of a disruptive shot; everything else is negative.
* Alarm: risk >= threshold for ``HOLD`` consecutive samples.  The threshold
  is chosen on validation shots as the most sensitive one whose false-alarm
  rate on non-disruptive shots is <= ``FAR_TARGET``; the test set is scored
  once with it.
* A disruption counts as predicted if the first alarm comes at least
  ``MIN_WARNING`` before it (shorter is useless to a control system).
* Baseline: the same procedure on one number, the Greenwald fraction
  ``ne_nG`` -- the model has to beat a one-variable rule to be worth using.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from . import jtext
from .kaggle_fetch import JTEXT_DIR

OUT = Path(__file__).resolve().parent / "models"
WARN_WINDOW = 0.030      # s before DownTime labelled positive
MIN_WARNING = 0.010      # s of lead time that counts as a success
HOLD = 3                 # consecutive samples above threshold
FAR_TARGET = 0.05


def labels(s: jtext.Shot) -> np.ndarray:
    if not s.disrupt:
        return np.zeros(len(s.t), int)
    return (s.t >= s.t_down - WARN_WINDOW).astype(int)


def stack(shots: List[jtext.Shot]):
    X = np.concatenate([s.X for s in shots])
    y = np.concatenate([labels(s) for s in shots])
    return X, y


def first_alarm(risk: np.ndarray, thr: float):
    above = risk >= thr
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= HOLD:
            return i
    return None


def shot_metrics(shots, scores: List[np.ndarray], thr: float) -> dict:
    tp = fn = fp = tn = 0
    lead = []
    for s, r in zip(shots, scores):
        i = first_alarm(r, thr)
        if s.disrupt:
            if i is not None and s.t_down - s.t[i] >= MIN_WARNING:
                tp += 1
                lead.append(s.t_down - s.t[i])
            else:
                fn += 1
        else:
            fp, tn = (fp + 1, tn) if i is not None else (fp, tn + 1)
    return {"disruptive": tp + fn, "predicted": tp,
            "tpr": tp / max(tp + fn, 1),
            "non_disruptive": fp + tn, "false_alarms": fp,
            "far": fp / max(fp + tn, 1),
            "median_warning_ms": float(np.median(lead) * 1e3) if lead else None}


def pick_threshold(shots, scores) -> float:
    best, best_tpr = None, -1.0
    for thr in np.unique(np.quantile(np.concatenate(scores),
                                     np.linspace(0.5, 0.9995, 120))):
        m = shot_metrics(shots, scores, thr)
        if m["far"] <= FAR_TARGET and m["tpr"] > best_tpr:
            best, best_tpr = float(thr), m["tpr"]
    return best if best is not None else float("inf")


def evaluate(name: str, score_fn: Callable, val, test) -> dict:
    sv = [score_fn(s.X) for s in val]
    st = [score_fn(s.X) for s in test]
    thr = pick_threshold(val, sv)
    yt = np.concatenate([labels(s) for s in test])
    pt = np.concatenate(st)
    return {"name": name, "threshold": thr,
            "validation": shot_metrics(val, sv, thr),
            "test": shot_metrics(test, st, thr),
            "test_sample_roc_auc": float(roc_auc_score(yt, pt)),
            "test_sample_pr_auc": float(average_precision_score(yt, pt))}


def _show(r: dict) -> None:
    for part in ("validation", "test"):
        m = r[part]
        w = m["median_warning_ms"]
        print(f"  {part:10s} TPR {m['tpr']:5.1%} ({m['predicted']}/"
              f"{m['disruptive']})  FAR {m['far']:5.1%} "
              f"({m['false_alarms']}/{m['non_disruptive']})  "
              f"median warning {'-' if w is None else f'{w:.0f} ms'}")
    print(f"  test sample-level ROC-AUC {r['test_sample_roc_auc']:.3f}  "
          f"PR-AUC {r['test_sample_pr_auc']:.3f}")


def main(limit: int = 0) -> None:
    shots = jtext.load_all(JTEXT_DIR, limit)
    if len(shots) < 50:
        raise SystemExit(f"only {len(shots)} shots in {JTEXT_DIR}; run "
                         "python -m ml.kaggle_fetch --jtext")
    n = len(shots)
    tr, va, te = shots[:int(.6 * n)], shots[int(.6 * n):int(.8 * n)], \
        shots[int(.8 * n):]
    for tag, part in (("train", tr), ("val", va), ("test", te)):
        d = sum(s.disrupt for s in part)
        print(f"{tag:5s} shots {len(part):5d}  disruptive {d:4d} "
              f"({d / len(part):.1%})  shots {part[0].shot}..{part[-1].shot}")

    Xtr, ytr = stack(tr)
    print(f"train samples {len(ytr)}, positive {ytr.mean():.2%}")
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, class_weight="balanced",
        early_stopping=True, validation_fraction=0.15, random_state=0)
    clf.fit(Xtr, ytr)
    print(f"boosting rounds used: {clf.n_iter_}")

    names = jtext.feature_names()
    gk = names.index("ne_nG:value")
    results = [
        evaluate("gradient boosting", lambda X: clf.predict_proba(X)[:, 1],
                 va, te),
        evaluate("baseline: Greenwald fraction only", lambda X: X[:, gk],
                 va, te),
    ]
    for r in results:
        print(f"\n{r['name']}  (threshold {r['threshold']:.4g})")
        _show(r)

    OUT.mkdir(exist_ok=True)
    joblib.dump({"model": clf, "features": names,
                 "signals": jtext.SIGNALS,
                 "threshold": results[0]["threshold"], "hold": HOLD,
                 "min_warning_s": MIN_WARNING}, OUT / "jtext_disruption.joblib")
    (OUT / "jtext_report.json").write_text(json.dumps({
        "data": "hark99/multi-machine-disruption-prediction-challenge "
                "(J-TEXT, real shots)",
        "split": "chronological by shot 60/20/20",
        "warn_window_s": WARN_WINDOW, "min_warning_s": MIN_WARNING,
        "far_target": FAR_TARGET, "hold": HOLD, "shots": n,
        "results": results}, indent=1), encoding="utf-8")
    print("\nsaved", OUT)


if __name__ == "__main__":
    import sys
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
