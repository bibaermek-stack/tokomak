"""J-TEXT shot loading and feature extraction for disruption prediction.

One row per millisecond of a shot's flat-top (the files are already cut to
``StartTime .. DownTime``).  Thirteen 1 kHz diagnostics are used -- the
ones the J-TEXT readme lists as extracted features plus the basic plasma
signals -- and each contributes four numbers: the value, its change over
the last 5 ms, its standard deviation over the last 10 ms and its maximum
over the last 20 ms.  Only past samples enter a row, so a prediction at time
t never sees t+1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

SIGNALS = ["ip", "bt", "ne0", "ne_nG", "qa_proxy", "radiation_proxy",
           "rotating_mode_proxy", "n=1 amplitude", "dx", "dy", "ip_error",
           "P_in", "P_rad"]
LOG_SIGNALS = {"n=1 amplitude"}        # spans decades
DT = 1e-3                              # 1 kHz


def feature_names() -> List[str]:
    return [f"{s}:{k}" for s in SIGNALS
            for k in ("value", "d5", "std10", "max20")]


@dataclass
class Shot:
    shot: int
    disrupt: bool
    t: np.ndarray                      # seconds
    t_down: float
    X: np.ndarray                      # (n_samples, n_features)


def _rolling(x: np.ndarray, w: int, fn) -> np.ndarray:
    pad = np.concatenate([np.full(w - 1, x[0]), x])
    win = np.lib.stride_tricks.sliding_window_view(pad, w)
    return fn(win, axis=1)


def features(raw: dict) -> np.ndarray:
    """raw: signal name -> 1-D array on a common 1 kHz axis."""
    cols = []
    for s in SIGNALS:
        x = np.asarray(raw[s], float)
        if s in LOG_SIGNALS:
            x = np.log10(np.maximum(np.abs(x), 1e-12))
        d5 = x - np.concatenate([np.full(5, x[0]), x[:-5]])
        cols += [x, d5, _rolling(x, 10, np.std), _rolling(x, 20, np.max)]
    return np.stack(cols, axis=1)


def load_shot(path: Path) -> Optional[Shot]:
    try:
        with h5py.File(path, "r") as f:
            d, m = f["data"], f["meta"]
            if any(s not in d for s in SIGNALS):
                return None
            n = min(len(d[s]) for s in SIGNALS)
            if n < 40:
                return None
            raw = {s: d[s][:n] for s in SIGNALS}
            start = float(m["StartTime"][()])
            t_down = float(m["DownTime"][()])
            disrupt = bool(m["IsDisrupt"][()])
    except (OSError, KeyError):
        return None
    X = features(raw)
    if not np.isfinite(X).all():
        return None
    t = start + DT * np.arange(n)
    return Shot(int(path.stem), disrupt, t, t_down, X)


def load_all(folder: Path, limit: int = 0) -> List[Shot]:
    files = sorted(folder.glob("*.hdf5"), key=lambda p: int(p.stem))
    shots = [s for s in (load_shot(p) for p in (files[:limit] if limit
                                                 else files)) if s]
    return shots
