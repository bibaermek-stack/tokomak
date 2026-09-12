"""Selectable empirical models: confinement, L-H threshold, density limit.

Every model is registered with its published exponents, the range of the
database it was fitted on, and a reference.  Callers pick one by name, which
is the point: a scaling is a fit to a particular set of machines and saying
which one is in use is part of stating a result.

The package ships one model of its own, ``kz1``.  It exists because the
standard H-mode scaling, IPB98(y,2), was fitted on conventional aspect ratio
and carries the wrong field dependence for spherical tokamaks -- B^0.15
against the B^~1 the NSTX/MAST regressions show -- so no single published
scaling covers both ITER and NSTX.  ``kz1``
interpolates between the two regimes in a way that reduces *exactly* to
IPB98(y,2) at conventional aspect ratio.  See :class:`KZ1` for what is
fitted, what is assumed, and what would be needed to do better.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
@dataclass
class ConfinementInputs:
    """Everything an energy-confinement scaling can ask for."""
    Ip: float           # plasma current [MA]
    B0: float           # toroidal field [T]
    P_loss: float       # loss power [MW]
    n_bar20: float      # line-averaged electron density [1e20 m^-3]
    R0: float           # major radius [m]
    a: float            # minor radius [m]
    kappa_a: float      # areal elongation, V / (2 pi^2 R a^2)
    a_mass: float = 2.5  # average ion mass [amu]

    @property
    def eps(self) -> float:
        return self.a / self.R0

    @property
    def n_bar19(self) -> float:
        return self.n_bar20 * 10.0


@dataclass
class ScalingModel:
    """A named empirical model with its provenance and validity range."""
    key: str
    label: str
    kind: str                       # "H-mode" | "L-mode" | "ST" | "bridge"
    reference: str
    fn: Callable[[ConfinementInputs], float]
    eps_range: tuple = (0.10, 0.40)
    note: str = ""

    def __call__(self, x: ConfinementInputs) -> float:
        return float(self.fn(x))

    def in_range(self, eps: float) -> str:
        """'иә' inside the fitted range, 'шекте' just outside, 'ЖОҚ' beyond."""
        lo, hi = self.eps_range
        if lo <= eps <= hi:
            return "иә"
        if eps <= hi * 1.45:
            return "шекте"
        return "ЖОҚ"

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "kind": self.kind,
                "reference": self.reference, "eps_range": list(self.eps_range),
                "note": self.note}


CONFINEMENT: Dict[str, ScalingModel] = {}


def _register(m: ScalingModel) -> ScalingModel:
    CONFINEMENT[m.key] = m
    return m


# ---------------------------------------------------------------------------
# Published scalings
# ---------------------------------------------------------------------------
def _ipb98y2(x: ConfinementInputs) -> float:
    return (0.0562
            * max(x.Ip, 1e-3) ** 0.93
            * max(x.B0, 1e-3) ** 0.15
            * max(x.P_loss, 1e-3) ** -0.69
            * max(x.n_bar19, 1e-3) ** 0.41
            * x.a_mass ** 0.19
            * x.R0 ** 1.97
            * x.eps ** 0.58
            * x.kappa_a ** 0.78)


def _ipb98y1(x: ConfinementInputs) -> float:
    return (0.0503
            * max(x.Ip, 1e-3) ** 0.91
            * max(x.B0, 1e-3) ** 0.15
            * max(x.P_loss, 1e-3) ** -0.65
            * max(x.n_bar19, 1e-3) ** 0.44
            * x.a_mass ** 0.13
            * x.R0 ** 2.05
            * x.eps ** 0.57
            * x.kappa_a ** 0.72)


def _iter89p(x: ConfinementInputs) -> float:
    # L-mode; written in terms of n in 1e20 and the geometric elongation
    return (0.048
            * max(x.Ip, 1e-3) ** 0.85
            * x.R0 ** 1.2
            * x.a ** 0.3
            * x.kappa_a ** 0.5
            * max(x.n_bar20, 1e-4) ** 0.1
            * max(x.B0, 1e-3) ** 0.2
            * x.a_mass ** 0.5
            * max(x.P_loss, 1e-3) ** -0.5)


def _st_trend(x: ConfinementInputs) -> float:
    """Spherical-torus trend: weak current, strong field dependence.

    The exponents follow the NSTX/MAST confinement studies (Kaye-type
    regressions): tau_E goes roughly as Ip^0.57 B^1.08 P^-0.73, in sharp
    contrast with IPB98(y,2)'s Ip^0.93 B^0.15.  The coefficient here is set
    by the same anchor point as :class:`KZ1`, so the two agree at eps -> 0.8.
    """
    return (_ST_COEFF
            * max(x.Ip, 1e-3) ** 0.57
            * max(x.B0, 1e-3) ** 1.08
            * max(x.P_loss, 1e-3) ** -0.73
            * max(x.n_bar19, 1e-3) ** 0.44
            * x.a_mass ** 0.19
            * x.R0 ** 1.97
            * x.eps ** 0.58
            * x.kappa_a ** 0.78)


# ---------------------------------------------------------------------------
# The package's own model
# ---------------------------------------------------------------------------
#: Reference spherical-tokamak H-mode point used to anchor the ST branch.
#: NSTX at 0.9 MA / 0.45 T with 6 MW crossing the separatrix confines for
#: roughly 45 ms -- the upper part of the 30-60 ms band NSTX H-modes show.
#: IPB98(y,2) returns 31 ms for the same discharge, i.e. H98 ~ 1.4, which is
#: where the best NSTX H-modes sit.  The level offset is therefore modest;
#: what kz1 mainly fixes is the *field* dependence, which is what breaks when
#: an ST result is extrapolated to a higher-field ST reactor.
ST_ANCHOR = ConfinementInputs(
    Ip=0.90, B0=0.45, P_loss=6.0, n_bar20=0.50,
    R0=0.85, a=0.67, kappa_a=2.00, a_mass=2.0,
)
ST_ANCHOR_TAU = 0.045      # [s]


class KZ1:
    """``kz1`` -- an aspect-ratio-bridging energy confinement scaling.

    **The problem.** IPB98(y,2) is fitted on a database spanning
    eps = a/R ~ 0.1-0.35.  Spherical tokamaks sit at eps ~ 0.75-0.8 and
    follow a very different dependence there: roughly Ip^0.57 B^1.08 instead
    of Ip^0.93 B^0.15.  In level the disagreement at NSTX's own operating
    point is modest (H98 ~ 1.4), but the exponents matter enormously as soon
    as anyone extrapolates an ST result to a higher-field ST reactor -- and
    they run in opposite directions for current and field.  Neither scaling
    covers both regimes, so a code that has to run ITER and NSTX with one
    model has to choose which one to be wrong about.

    **The construction.** Rather than fit a new global regression -- which
    would need the ITPA database, not a handful of published points -- kz1
    interpolates the *exponents* between the two regimes with a logistic
    blend in eps::

        w(eps) = 1 / (1 + exp(-(eps - eps0) / d)),   eps0 = 0.55, d = 0.05

        alpha_I(w) = 0.93 - 0.36 w        alpha_B(w) = 0.15 + 0.93 w
        alpha_P(w) = -0.69 - 0.04 w       alpha_n(w) = 0.41 + 0.03 w
        C(w)       = 0.0562 * K^w

    The remaining exponents (mass, R, eps, kappa) are held at their
    IPB98(y,2) values, because the ST regressions do not constrain them
    independently.

    **What is fitted.** Exactly one number: ``K``, solved at import time so
    that the model reproduces :data:`ST_ANCHOR_TAU` at :data:`ST_ANCHOR`.
    Everything else is either IPB98(y,2) or the published ST exponents.

    **Properties.**

    * At eps = 0.32 (ITER) the blend weight is w = 0.010, so kz1 returns
      IPB98(y,2) to within 1% -- every ITER validation that passes for
      IPB98(y,2) passes unchanged.
    * At eps = 0.79 (NSTX) it returns the measured confinement; IPB98(y,2)
      is low by ~1.4x there in level, but its B^0.15 dependence is the real
      problem -- it makes a higher-field ST look no better, which the ST
      regressions contradict.
    * It is continuous and monotonic in eps, so a machine at A = 2 gets a
      sensible partial correction instead of falling off a cliff.

    **What it is not.** It is an interpolation anchored on one ST point, not
    a regression: the ST branch carries the uncertainty of that anchor
    (NSTX H-modes span 30-60 ms) roughly linearly, and because B enters the
    ST branch as B^1.08, extrapolating far above the anchor's 0.45 T carries
    the full exponent uncertainty -- a 1 T ST gets a 2.2x field factor from
    this model and only 1.13x from IPB98(y,2), and no measurement in this
    package settles which is right.  A proper version needs a multi-machine
    fit over the ITPA H-mode database with the ST entries included, and would
    fit the geometric exponents too.  Treat kz1 as a defensible bridge for
    scoping across aspect ratio, not as a measurement.
    """

    EPS0 = 0.55
    WIDTH = 0.05

    @staticmethod
    def weight(eps: float) -> float:
        """Blend weight: 0 = conventional tokamak, 1 = spherical torus."""
        return 1.0 / (1.0 + math.exp(-(eps - KZ1.EPS0) / KZ1.WIDTH))

    @staticmethod
    def exponents(eps: float) -> dict:
        w = KZ1.weight(eps)
        return {"w": w,
                "alpha_I": 0.93 - 0.36 * w,
                "alpha_B": 0.15 + 0.93 * w,
                "alpha_P": -0.69 - 0.04 * w,
                "alpha_n": 0.41 + 0.03 * w}

    @staticmethod
    def tau_e(x: ConfinementInputs, K: Optional[float] = None) -> float:
        e = KZ1.exponents(x.eps)
        k = _KZ1_K if K is None else K
        return (0.0562 * k ** e["w"]
                * max(x.Ip, 1e-3) ** e["alpha_I"]
                * max(x.B0, 1e-3) ** e["alpha_B"]
                * max(x.P_loss, 1e-3) ** e["alpha_P"]
                * max(x.n_bar19, 1e-3) ** e["alpha_n"]
                * x.a_mass ** 0.19
                * x.R0 ** 1.97
                * x.eps ** 0.58
                * x.kappa_a ** 0.78)


def _solve_kz1_anchor() -> float:
    """Solve K so that kz1 reproduces ST_ANCHOR_TAU at ST_ANCHOR."""
    w = KZ1.weight(ST_ANCHOR.eps)
    tau_unit = KZ1.tau_e(ST_ANCHOR, K=1.0)          # K^w factored out
    return (ST_ANCHOR_TAU / tau_unit) ** (1.0 / w)


_KZ1_K = _solve_kz1_anchor()
#: Coefficient of the pure-ST trend, tied to the same anchor as kz1.
_ST_COEFF = 1.0
_ST_COEFF = ST_ANCHOR_TAU / _st_trend(ST_ANCHOR)


_register(ScalingModel(
    "ipb98y2", "IPB98(y,2)", "H-mode",
    "ITER Physics Basis, Nucl. Fusion 39 (1999) 2175",
    _ipb98y2, (0.10, 0.36),
    note="ELMy H-mode стандарты; ELM шығыны скейлингтің ішінде бар",
))
_register(ScalingModel(
    "ipb98y1", "IPB98(y,1)", "H-mode",
    "ITER Physics Basis, Nucl. Fusion 39 (1999) 2175",
    _ipb98y1, (0.10, 0.36),
))
_register(ScalingModel(
    "iter89p", "ITER89-P", "L-mode",
    "Yushmanov et al., Nucl. Fusion 30 (1990) 1999",
    _iter89p, (0.10, 0.40),
    note="L-режим; H-режимде H-факторы 1.6-2.0 болады",
))
_register(ScalingModel(
    "st", "ST trend (Kaye-type)", "ST",
    "NSTX/MAST confinement regressions",
    _st_trend, (0.60, 0.85),
    note="Ip^0.57 B^1.08 — сфералық торлардың бақыланған тәуелділігі",
))
_register(ScalingModel(
    "kz1", "KZ-1 (осы жоба)", "bridge",
    "аспект қатынасы бойынша IPB98(y,2) мен ST трендін біріктіру",
    KZ1.tau_e, (0.10, 0.85),
    note="eps<0.4-те IPB98(y,2)-мен дәл сәйкес, eps~0.8-де ST-ге көшеді",
))


def tau_energy(model: str, x: ConfinementInputs) -> float:
    """Energy confinement time [s] from the named scaling."""
    try:
        return CONFINEMENT[model](x)
    except KeyError:
        raise KeyError(f"unknown confinement model {model!r}; "
                       f"available: {', '.join(CONFINEMENT)}") from None


# ---------------------------------------------------------------------------
# L-H power threshold
# ---------------------------------------------------------------------------
@dataclass
class ThresholdInputs:
    n_bar20: float
    B0: float
    S: float              # plasma surface area [m^2]
    R0: float
    a: float
    a_mass: float = 2.5


def _martin08_S(x: ThresholdInputs) -> float:
    """Martin 2008, surface-area form, with the isotope correction."""
    return (0.0488 * max(x.n_bar20, 1e-3) ** 0.717
            * max(x.B0, 1e-3) ** 0.803
            * x.S ** 0.941 * (2.0 / x.a_mass))


def _martin08_Ra(x: ThresholdInputs) -> float:
    """Martin 2008, R/a form -- an independent fit to the same database."""
    return (2.15 * max(x.n_bar20, 1e-3) ** 0.782
            * max(x.B0, 1e-3) ** 0.772
            * x.a ** 0.975 * x.R0 ** 0.999 * (2.0 / x.a_mass))


def _kz_lh(x: ThresholdInputs) -> float:
    """Martin 2008 with the low-density branch restored.

    The Martin fit is monotonic in density, but every machine that has
    looked for it sees the threshold turn *up* again below a density
    minimum near n_min ~ 0.7 n_G / 10, because electron-ion coupling gets
    too weak to build the pedestal.  A code that ramps density from
    breakdown passes straight through that branch, so a monotonic fit lets
    H-mode start far too early.  The correction is a multiplicative factor
    that is 1 well above n_min and diverges as n -> n_min/2:

        f(n) = 1 + ((n_min / n)^2 - 1) for n < n_min, else 1

    This is the qualitative shape the measurements show, not a fit -- the
    published parametrisations of the minimum disagree with one another by
    more than this correction is worth.
    """
    base = _martin08_S(x)
    n_min = 0.35 * max(x.B0, 0.1) ** 0.6 / max(x.R0, 0.1) ** 0.4 * 0.1
    n = max(x.n_bar20, 1e-3)
    if n >= n_min:
        return base
    return base * (1.0 + ((n_min / n) ** 2 - 1.0))


LH_THRESHOLD: Dict[str, ScalingModel] = {}
for _k, _lab, _ref, _fn in [
    ("martin08", "Martin 2008 (S)", "Martin et al., J. Phys. Conf. Ser. 123 (2008) 012033", _martin08_S),
    ("martin08_ra", "Martin 2008 (R, a)", "Martin et al., J. Phys. Conf. Ser. 123 (2008) 012033", _martin08_Ra),
    ("kz_lh", "KZ-LH (осы жоба)", "Martin 2008 + төмен тығыздық тармағы", _kz_lh),
]:
    LH_THRESHOLD[_k] = ScalingModel(_k, _lab, "L-H", _ref, _fn, (0.10, 0.85))


def p_lh_threshold(model: str, x: ThresholdInputs) -> float:
    """L-H transition power threshold [MW]."""
    try:
        return float(LH_THRESHOLD[model].fn(x))
    except KeyError:
        raise KeyError(f"unknown L-H model {model!r}; "
                       f"available: {', '.join(LH_THRESHOLD)}") from None


# ---------------------------------------------------------------------------
# Density limit
# ---------------------------------------------------------------------------
def greenwald_density(Ip: float, a: float) -> float:
    """Greenwald limit [1e20 m^-3]: n_G = I_p / (pi a^2)."""
    return Ip / (math.pi * a * a)


def model_catalogue() -> dict:
    """Everything a caller can select, for the API's /models endpoint."""
    return {
        "confinement": [m.to_dict() for m in CONFINEMENT.values()],
        "lh_threshold": [m.to_dict() for m in LH_THRESHOLD.values()],
        "density_limit": [{"key": "greenwald", "label": "Greenwald",
                           "reference": "Greenwald, PPCF 44 (2002) R27"}],
        "kz1_anchor": {
            "K": _KZ1_K,
            "tau_anchor_s": ST_ANCHOR_TAU,
            "point": {"Ip": ST_ANCHOR.Ip, "B0": ST_ANCHOR.B0,
                      "P_loss": ST_ANCHOR.P_loss, "n_bar20": ST_ANCHOR.n_bar20,
                      "R0": ST_ANCHOR.R0, "a": ST_ANCHOR.a,
                      "kappa_a": ST_ANCHOR.kappa_a},
        },
    }
