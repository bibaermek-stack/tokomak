"""Pedestal height from peeling-ballooning physics (EPED-like).

The pedestal decides the fusion power.  A stiff core carries whatever
gradient the turbulence allows and no more, so the temperature profile is
essentially the pedestal value multiplied by a fixed factor -- which means
the pedestal height, not the core, sets the stored energy and therefore the
burn.  Any model that leaves it as a free parameter has left the answer as
a free parameter.

EPED's insight is that two constraints close on each other:

* **Kinetic-ballooning-mode width.**  The pedestal cannot be narrower than
  the KBM limit allows, and that width goes as the square root of the
  pedestal poloidal beta::

      Delta_N = C_w sqrt(beta_p,ped)

* **Peeling-ballooning pressure limit.**  Across that width the pressure
  gradient cannot exceed the ideal-MHD ballooning threshold::

      alpha = -(2 mu0 R q^2 / B^2) dp/dr  <=  alpha_crit

Solving the pair self-consistently gives both the height and the width, and
the well-known consequence falls out rather than being assumed:
**p_ped scales as Ip^2**.  That scaling is the check this module is
validated against, not just the ITER number it is calibrated to.

``C_w = 0.076`` is EPED1's published width coefficient.  ``ALPHA_CRIT`` is
calibrated once, in the full 1.5-D chain on ITER's Q = 10 point, because
the ballooning threshold depends on the local magnetic shear and
flux-surface shaping, which a 0-D reduction of the pedestal cannot
resolve.  What that calibration absorbs is written out in full at the
constant itself -- it is one named, measured residual, not a free
parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .constants import MU0, P_UNIT

#: EPED1 kinetic-ballooning width coefficient, Delta_N = C_W sqrt(beta_p,ped).
C_W = 0.076

#: Ballooning threshold in THIS normalisation, calibrated so that ITER at
#: 15 MA with n_ped = 0.75e20 returns T_ped = 4.5 keV -- the value its
#: pedestal predictions cluster around.
#:
#: An earlier comment here claimed the value was large because this
#: normalisation differs from the textbook s-alpha one.  It does not: this
#: is the same alpha = -(2 mu0 R q^2 / B^2) dp/dr.  The real reason is
#: physical and more interesting.  The circular first-stability boundary
#: sits near alpha ~ 1-2, and ITER's pedestal is at 13 -- because strong
#: triangularity connects the first and second stable regions, so a shaped
#: pedestal climbs straight past the first-stability boundary into the
#: second stable region.  That is why shaping buys pedestal height, and it
#: is why ``mhd.ballooning_alpha_crit`` is documented as marking
#: "not first-stable" rather than "unstable".
#:
#: What is NOT calibrated is the scaling: with the width constraint closing
#: on the gradient limit, p_ped ~ Ip^2 falls out, and the validation suite
#: measures that exponent rather than trusting it.
#:
#: CALIBRATED IN-CHAIN, AND CARRYING ONE KNOWN RESIDUAL.
#: The value is set by running the full 1.5-D solve to a stationary burn
#: and asking for ITER's Q = 10 point, not by a standalone pedestal
#: solve -- so it necessarily absorbs whatever error the rest of the chain
#: brings to the pedestal.  There is exactly one such error and it is not
#: hidden here:
#:
#:   * The fixed-boundary Grad-Shafranov solve returns q95 = 3.126 where
#:     ITER's free-boundary equilibrium gives 3.00 (+4.2%), because there
#:     is no X-point in a fixed-boundary solve.
#:   * This module has p_ped ~ q95^-4 and T_ped ~ alpha_crit^2 (both
#:     measured, see ``scaling_exponent``), so the q95 residual costs
#:     T_ped a factor (3.126/3.00)^-4 = 0.848 and recovering it costs
#:     alpha_crit a factor (3.126/3.00)^2 = 1.086.
#:   * 13.40 x 1.086 = 14.55, against the 14.45 the in-chain calibration
#:     actually lands on.  So the recalibration is the q95 residual and
#:     nothing else -- the 0.7% left over is the chain's own feedback.
#:
#: The distinction from a fudge factor matters, and this package has
#: already been bitten by the difference: ``Q95_XPOINT`` in
#: :mod:`tokamak.equilibrium` was once set to 1.109 to close a q95 gap
#: that turned out to be an over-smoothed spline, and the constant made
#: the bug invisible for as long as it stood.  A calibration is legitimate
#: when the quantity it absorbs is measured, named, and still reported --
#: q95 stays a separate validation check at its true +4.2%, and it is not
#: quietly corrected anywhere.  A calibration is a fudge when it makes the
#: residual disappear from view.
ALPHA_CRIT = 14.45


@dataclass
class Pedestal:
    p_ped: float          # total pedestal-top pressure [Pa]
    T_ped: float          # pedestal-top temperature, T_e = T_i [keV]
    n_ped20: float        # pedestal-top electron density [1e20 m^-3]
    width_psi: float      # pedestal width in normalised poloidal flux
    width_m: float        # radial width [m]
    beta_p_ped: float
    alpha: float          # ballooning parameter achieved
    converged: bool

    def to_dict(self) -> dict:
        return {k: (float(v) if not isinstance(v, bool) else v)
                for k, v in self.__dict__.items()}


def solve(*, R0: float, a: float, kappa_a: float, B0: float, Ip: float,
          q95: float, n_ped20: float, f_ion: float = 0.87,
          L_pol: Optional[float] = None,
          alpha_crit: Optional[float] = None,
          c_w: Optional[float] = None, max_iter: int = 80) -> Pedestal:
    """Solve the coupled width / ballooning-limit pair for the pedestal.

    Parameters
    ----------
    n_ped20 : pedestal-top electron density [1e20 m^-3]
    f_ion : n_i / n_e, so the total pressure is p = (1 + f_ion) n_e T
    q95 : safety factor at the 95% surface -- the pedestal sits there, so
        this is what enters the ballooning parameter

    Returns the pedestal temperature with T_e = T_i, which is what the
    pedestal measurements show: the two species are collisional enough at
    the pedestal top to equilibrate.
    """
    # Resolve the calibration constants at call time, not at import time:
    # binding them as default arguments freezes them, so a caller that
    # rebinds the module constant to explore the calibration silently gets
    # the original value back.
    alpha_crit = ALPHA_CRIT if alpha_crit is None else alpha_crit
    c_w = C_W if c_w is None else c_w
    if L_pol is None:
        L_pol = 2.0 * np.pi * a * np.sqrt((1.0 + kappa_a ** 2) / 2.0)
    B_pol = MU0 * Ip * 1e6 / L_pol

    p = 5.0e4                       # first guess [Pa]
    converged = False
    width_psi = 0.04
    for _ in range(max_iter):
        beta_p = 2.0 * MU0 * p / B_pol ** 2
        width_psi = c_w * np.sqrt(max(beta_p, 1e-6))
        # psi_N ~ rho^2 near the edge, so d(psi_N) = 2 rho d(rho) ~ 2 d(rho)
        width_m = 0.5 * width_psi * a
        # alpha = -(2 mu0 R q^2 / B^2) dp/dr, with dp/dr ~ p / width
        p_new = (alpha_crit * B0 ** 2 * width_m
                 / (2.0 * MU0 * R0 * max(q95, 0.5) ** 2))
        if abs(p_new - p) < 1e-4 * max(p, 1.0):
            p = p_new
            converged = True
            break
        p = 0.5 * p + 0.5 * p_new

    beta_p = 2.0 * MU0 * p / B_pol ** 2
    width_psi = c_w * np.sqrt(max(beta_p, 1e-6))
    width_m = 0.5 * width_psi * a
    alpha = (2.0 * MU0 * R0 * max(q95, 0.5) ** 2 * p
             / (B0 ** 2 * max(width_m, 1e-4)))

    # p = (n_e T_e + n_i T_i) with T_e = T_i at the pedestal top
    T_ped = p / (P_UNIT * max(n_ped20, 1e-3) * (1.0 + f_ion))
    return Pedestal(p_ped=float(p), T_ped=float(T_ped),
                    n_ped20=float(n_ped20), width_psi=float(width_psi),
                    width_m=float(width_m), beta_p_ped=float(beta_p),
                    alpha=float(alpha), converged=converged)


def scaling_exponent(quantity: str = "Ip", **base) -> float:
    """Numerically measure how p_ped scales with one input.

    Used by the validation suite: EPED predicts p_ped ~ Ip^2, and a model
    that reproduces the ITER point but not that exponent has been fitted
    rather than derived.
    """
    lo = dict(base)
    hi = dict(base)
    lo[quantity] = base[quantity] * 0.8
    hi[quantity] = base[quantity] * 1.25
    # q95 goes inversely with Ip at fixed field and shape
    if quantity == "Ip":
        lo["q95"] = base["q95"] / 0.8
        hi["q95"] = base["q95"] / 1.25
    p_lo = solve(**lo).p_ped
    p_hi = solve(**hi).p_ped
    return float(np.log(p_hi / p_lo) / np.log(hi[quantity] / lo[quantity]))
