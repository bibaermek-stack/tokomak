"""SHRX -- SINTEZ's neural turbulent-transport model.

SHRX is a small network (NumPy inference, no JAX) that returns the turbulent heat diffusivities
chi_e and chi_i of a tokamak core from local, dimensionless plasma parameters.  It is distilled
from QLKNN -- the network DeepMind's TORAX uses, itself trained on QuaLiKiz gyrokinetic runs --
on the states SINTEZ's own simulations visit, so on those it reproduces QLKNN while evaluating
faster and without a JAX dependency.  What it cannot do is be *more* accurate than QLKNN about
gyrokinetic physics: its teacher is QLKNN.

    from tokamak.shrx import evaluate
    out = evaluate(rho, Te, Ti, ne20, q, machine, zeff, normni)   # out["chi_e"], out["chi_i"] [m^2/s]

The network input/output convention and the conversions to and from SINTEZ's state live in
:mod:`tokamak.shrx.bridge`; training scripts are in ``backend/ml/shrx``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from . import bridge

#: the shipped weights; SHRX_WEIGHTS=/path/to.npz overrides them (used to try candidates during training)
CAVEAT = ("SHRX: SINTEZ-тің нейрожелілік тасымал моделі, QLKNN (DeepMind TORAX) желісінен дистилляцияланған. "
          "QLKNN-нің өзі QuaLiKiz гирокинетикасының суррогаты, сондықтан SHRX гирокинетикалық дәлдікте QLKNN-нен "
          "асып түспейді. Кіріс ауқымынан тыс нүктелер қиылады (out_of_range). Қараңыз: docs/shrx/.")
WEIGHTS_PATH = Path(os.environ.get("SHRX_WEIGHTS") or Path(__file__).with_name("weights.npz"))


class Shrx:
    """The trained network.  ``predict`` maps QLKNN-style inputs to gyro-Bohm fluxes."""

    def __init__(self, path: Path = WEIGHTS_PATH):
        z = np.load(path)
        n = int(z["n_layers"])
        self.W = [z[f"W{i}"].astype(np.float32) for i in range(n)]
        self.b = [z[f"b{i}"].astype(np.float32) for i in range(n)]
        self.mu, self.sd = z["mu"].astype(np.float32), z["sd"].astype(np.float32)
        self.scale = z["scale"].astype(np.float32)
        self.lo, self.hi = z["lo"].astype(np.float32), z["hi"].astype(np.float32)
        self.meta = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").is_file() else {}
        # fold the input standardisation into the first layer: one matmul less per call
        self._W0 = np.ascontiguousarray(self.W[0] / self.sd[:, None])
        self._b0 = (self.b[0] - (self.mu / self.sd) @ self.W[0]).astype(np.float32)

    @property
    def n_parameters(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    def out_of_range(self, X: np.ndarray) -> np.ndarray:
        """True where an input lies outside the box QLKNN (and so SHRX) was built for."""
        return np.any((X < self.lo) | (X > self.hi), axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(N, 10) inputs -> (N, 3) gyro-Bohm fluxes (q_i, q_e, Gamma_e).  Inputs are clipped to the box."""
        h = np.clip(X.astype(np.float32, copy=False), self.lo, self.hi) @ self._W0
        h += self._b0
        np.maximum(h, 0.0, out=h)
        for W, b in zip(self.W[1:-1], self.b[1:-1]):
            h = h @ W
            h += b
            np.maximum(h, 0.0, out=h)
        o = h @ self.W[-1]
        o += self.b[-1]
        np.sinh(o, out=o)
        o *= self.scale
        return o


_MODEL: Optional[Shrx] = None


def available() -> bool:
    return WEIGHTS_PATH.is_file()


def load() -> Shrx:
    """The shared network (loaded once)."""
    global _MODEL
    if _MODEL is None:
        if not available():
            raise FileNotFoundError(f"SHRX weights not found: {WEIGHTS_PATH}")
        _MODEL = Shrx()
    return _MODEL


def evaluate(rho, Te, Ti, ne20, q, machine, zeff: float, normni: float = 0.9) -> dict:
    """chi_e, chi_i [m^2/s] on the given radial grid for one machine.

    ``machine`` needs ``R0``, ``a``, ``B0``, ``kappa_x`` and ``a_mass`` (a :class:`tokamak.machines.Machine`).
    Also returns the gyro-Bohm particle flux and the fraction of radial points outside the
    training box (their inputs were clipped).
    """
    X, aux = bridge.features(rho, Te, Ti, ne20, q, machine.R0, machine.a, machine.B0, zeff, normni,
                             kappa_edge=machine.kappa_x)
    net = load()
    flux = net.predict(X)
    chi_i, chi_e = bridge.to_si(flux[:, 0], flux[:, 1], aux, a_mass=machine.a_mass)
    ood = net.out_of_range(X)
    return {"chi_e": chi_e, "chi_i": chi_i, "gamma_e_gb": flux[:, 2],
            "ood_fraction": float(ood.mean()), "ood": ood, "features": X}
