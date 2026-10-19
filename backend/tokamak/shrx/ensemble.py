"""SHRX as an ensemble: the mean is a better predictor, the spread says where to distrust it.

The members are independently trained networks (different seeds and different 85 % samples of the training
rows).  Where they agree they are all close to QLKNN; where they disagree -- almost always right at the turbulence
threshold, or outside what the training data covered -- at least some of them are wrong.  The relative spread of
chi across members is therefore an error estimate that QLKNN itself does not provide.

Weights are stacked along a leading member axis, so the whole ensemble is one batched matrix product per layer
(a handful of NumPy calls whatever the number of members).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ENSEMBLE_PATH = Path(__file__).with_name("ensemble.npz")


class ShrxEnsemble:
    def __init__(self, path: Path = ENSEMBLE_PATH):
        z = np.load(path)
        n = int(z["n_layers"])
        self.K = int(z["W0"].shape[0])
        mu, sd = z["mu"].astype(np.float32), z["sd"].astype(np.float32)              # (K, 10)
        W = [z[f"W{i}"].astype(np.float32) for i in range(n)]                        # each (K, in, out)
        b = [z[f"b{i}"].astype(np.float32) for i in range(n)]                        # each (K, out)
        # fold the input standardisation into the first layer of every member
        self._W0 = np.ascontiguousarray(W[0] / sd[:, :, None])
        self._b0 = (b[0] - np.einsum("ki,kio->ko", mu / sd, W[0])).astype(np.float32)
        self._W = W[1:]
        self._b = [x[:, None, :] for x in b[1:]]
        self._b0 = self._b0[:, None, :]
        self.scale = z["scale"].astype(np.float32)[:, None, :]                        # (K, 1, 3)
        self.lo, self.hi = z["lo"].astype(np.float32), z["hi"].astype(np.float32)
        meta = path.with_suffix(".json")
        self.meta = json.loads(meta.read_text()) if meta.is_file() else {}
        #: relative spread of chi above which a point is flagged "uncertain" (calibrated on held-out machines)
        self.unc_threshold = float(self.meta.get("unc_threshold", 0.25))

    def out_of_range(self, X: np.ndarray) -> np.ndarray:
        return np.any((X < self.lo) | (X > self.hi), axis=1)

    def predict_members(self, X: np.ndarray) -> np.ndarray:
        """(N, 10) inputs -> (K, N, 3) gyro-Bohm fluxes, one slab per member."""
        x = np.clip(X.astype(np.float32, copy=False), self.lo, self.hi)
        h = np.matmul(x[None], self._W0)
        h += self._b0
        np.maximum(h, 0.0, out=h)
        for W, b in zip(self._W[:-1], self._b[:-1]):
            h = np.matmul(h, W)
            h += b
            np.maximum(h, 0.0, out=h)
        o = np.matmul(h, self._W[-1])
        o += self._b[-1]
        np.sinh(o, out=o)
        o *= self.scale
        return o

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble-mean fluxes, (N, 3)."""
        return self.predict_members(X).mean(axis=0)
