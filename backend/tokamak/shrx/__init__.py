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

#: QLKNN (and so SHRX) was trained on gyrokinetic runs at one inverse aspect ratio, epsilon = 1/3, with circular flux
#: surfaces.  Trapped-particle physics -- hence the ITG/TEM thresholds -- moves with epsilon, and spherical tokamaks add
#: electromagnetic and micro-tearing channels QuaLiKiz-based networks do not contain.  The limits below are where the
#: database test (docs/shrx/) stopped trusting the model: conventional tokamaks to epsilon ~ 0.45, spherical ones fail.
EPS_OK = (0.2, 0.45)
EPS_MARGINAL = (0.15, 0.6)
KAPPA_OK = 2.2


def machine_validity(machine) -> dict:
    """Is this machine inside the regime SHRX's teacher (QLKNN) was built for?  ``status``: ok / marginal / outside."""
    eps = float(machine.a / machine.R0)
    kappa = float(getattr(machine, "kappa_x", 1.0))
    reasons = []
    if eps < EPS_MARGINAL[0] or eps > EPS_MARGINAL[1]:
        status = "outside"
        reasons.append(f"a/R0 = {eps:.2f}: QLKNN was trained at 1/3 and fails far from it (spherical tokamaks)")
    elif eps < EPS_OK[0] or eps > EPS_OK[1]:
        status = "marginal"
        reasons.append(f"a/R0 = {eps:.2f}: outside the {EPS_OK[0]}-{EPS_OK[1]} range the model was checked on")
    else:
        status = "ok"
    if kappa > KAPPA_OK:
        status = "outside" if status == "outside" else "marginal"
        reasons.append(f"kappa = {kappa:.2f}: QLKNN's flux surfaces are close to circular")
    return {"status": status, "epsilon": eps, "kappa": kappa, "reasons": reasons}


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
_ENSEMBLE = None


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


def ensemble_available() -> bool:
    from .ensemble import ENSEMBLE_PATH
    return ENSEMBLE_PATH.is_file()


def load_ensemble():
    """The shared ensemble (loaded once)."""
    global _ENSEMBLE
    if _ENSEMBLE is None:
        from .ensemble import ShrxEnsemble
        if not ensemble_available():
            raise FileNotFoundError("SHRX ensemble weights not found (backend/tokamak/shrx/ensemble.npz)")
        _ENSEMBLE = ShrxEnsemble()
    return _ENSEMBLE


def _ood_core(net, X: np.ndarray, ood: np.ndarray) -> float:
    """Share of the radial points that QLKNN's radial range covers (x in its box) whose inputs still leave the box.

    ``ood_fraction`` also counts the axis and the edge, which lie outside the box in every discharge; this one is ~0 in a
    healthy run and large when the state itself (T_i/T_e, q, shear, ...) is outside what the network knows."""
    j = bridge.FEATURES.index("x")
    inside = (X[:, j] >= net.lo[j]) & (X[:, j] <= net.hi[j])
    return float(ood[inside].mean()) if inside.any() else 1.0


def evaluate(rho, Te, Ti, ne20, q, machine, zeff: float, normni: float = 0.9, uncertainty: bool = False) -> dict:
    """chi_e, chi_i [m^2/s] on the given radial grid for one machine.

    ``machine`` needs ``R0``, ``a``, ``B0``, ``kappa_x`` and ``a_mass`` (a :class:`tokamak.machines.Machine`).
    Also returns the gyro-Bohm particle flux and the fraction of radial points outside the
    training box (their inputs were clipped).

    With ``uncertainty=True`` the ensemble is used: ``chi_e``/``chi_i`` come from the ensemble-mean flux, and
    ``chi_e_spread``/``chi_i_spread`` are the relative standard deviation of chi across members (0 where the turbulence is
    off in every member), ``uncertain`` flags points whose spread exceeds the calibrated threshold.
    """
    X, aux = bridge.features(rho, Te, Ti, ne20, q, machine.R0, machine.a, machine.B0, zeff, normni,
                             kappa_edge=machine.kappa_x)
    if not uncertainty:
        net = load()
        flux = net.predict(X)
        chi_i, chi_e = bridge.to_si(flux[:, 0], flux[:, 1], aux, a_mass=machine.a_mass)
        ood = net.out_of_range(X)
        return {"chi_e": chi_e, "chi_i": chi_i, "gamma_e_gb": flux[:, 2],
                "ood_fraction": float(ood.mean()), "ood_core": _ood_core(net, X, ood), "ood": ood, "features": X}
    net = load_ensemble()
    members = net.predict_members(X)                                  # (K, N, 3)
    flux = members.mean(axis=0)
    chi_i, chi_e = bridge.to_si(flux[:, 0], flux[:, 1], aux, a_mass=machine.a_mass)
    ci = np.stack([bridge.to_si(m[:, 0], m[:, 1], aux, a_mass=machine.a_mass) for m in members])   # (K, 2, N)
    spread = {}
    for j, name in ((0, "chi_i"), (1, "chi_e")):
        c = ci[:, j, :]
        spread[name] = c.std(axis=0) / np.maximum(c.mean(axis=0), 1e-9)
    # points where every member sits on the chi floor are "off", not "uncertain"
    live = (ci > 0.05 * 1.0001).any(axis=(0, 1))
    uncertain = live & ((spread["chi_e"] > net.unc_threshold) | (spread["chi_i"] > net.unc_threshold))
    ood = net.out_of_range(X)
    return {"chi_e": chi_e, "chi_i": chi_i, "gamma_e_gb": flux[:, 2], "ood_fraction": float(ood.mean()), "ood_core": _ood_core(net, X, ood),
            "ood": ood, "features": X, "chi_e_spread": spread["chi_e"], "chi_i_spread": spread["chi_i"], "uncertain": uncertain,
            "uncertain_fraction": float(uncertain.mean())}
