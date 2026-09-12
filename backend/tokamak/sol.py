"""Scrape-off layer and divertor: two-point model with detachment.

The 0-D stand-in for divertor load -- ``P_sep (1 - f_rad) / A_wetted`` with a
constant wetted area -- gets the right number for ITER only because the area
was chosen to make it so.  It cannot say whether the target detaches, how
much impurity seeding that takes, or what happens on a machine whose
poloidal field is different.

This module implements the conduction-limited two-point model:

    T_u^{7/2} = T_t^{7/2} + (7/2) q_par L_par / kappa_0e      (conduction)
    2 n_t T_t = n_u T_u                                       (pressure)
    q_t       = gamma n_t c_s T_t                             (sheath)

with the power channel width from the Eich regression, and momentum and
power loss factors that turn on as the target cools -- which is what
detachment is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .constants import E_CHARGE, M_P, MU0

#: Parallel electron heat conductivity coefficient, Spitzer-Harm [W eV^-7/2 m^-1]
KAPPA_0E = 2000.0
#: Sheath heat transmission coefficient for electrons + ions
GAMMA_SHEATH = 7.0


@dataclass
class DivertorState:
    lambda_q_mm: float        # upstream power channel width
    lambda_int_mm: float      # integral width the target sees, lam_q + 1.64 S
    q_par_MWm2: float         # parallel heat flux entering the SOL
    T_upstream_eV: float
    T_target_eV: float
    n_target_20: float
    q_target_MWm2: float      # perpendicular load on the target
    f_rad_div: float          # fraction of P_sep radiated in the divertor
    detached: bool
    no_attached_root: bool    # the sheath cannot carry q_par at any T_t
    regime: str               # "attached" | "partially detached" | "detached"
                              # | "шешім жоқ — себу міндетті"
    A_wetted: float           # [m^2]
    P_target: float           # [MW] reaching the target plates

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in self.__dict__.items()}


def eich_lambda_q(B_pol: float) -> float:
    """Midplane power channel width [mm], Eich regression #14.

    ``lambda_q = 0.63 B_pol^-1.19`` for H-mode.  For ITER's B_pol ~ 1 T this
    gives ~0.6 mm, the result that made divertor detachment mandatory rather
    than optional.
    """
    return 0.63 * max(B_pol, 0.05) ** -1.19


def connection_length(R0: float, q95: float) -> float:
    """Parallel connection length midplane -> target [m]."""
    return float(np.pi * R0 * q95)


def two_point(P_sep: float, *, R0: float, a: float, B0: float, B_pol: float,
              q95: float, n_sep20: float, A_mass: float = 2.5,
              f_rad_seed: float = 0.0, flux_expansion: float = 5.0,
              tilt_deg: float = 2.5, n_div_targets: int = 2,
              s_spread_mm: float = 1.0) -> DivertorState:
    """Solve the two-point model for the target conditions.

    Parameters
    ----------
    P_sep : power crossing the separatrix [MW]
    n_sep20 : upstream separatrix density [1e20 m^-3]
    f_rad_seed : fraction of P_sep removed by seeded impurity radiation in
        the SOL before the power reaches the target
    flux_expansion : poloidal flux expansion between midplane and target
    tilt_deg : angle between the field line and the target surface
    s_spread_mm : divertor broadening S.  Eich's fit has two widths: the
        upstream channel lambda_q and a Gaussian spreading S that widens the
        footprint inside the divertor.  The integral width the target
        actually sees is lambda_int = lambda_q + 1.64 S, and for ITER S ~ 1 mm
        against lambda_q ~ 0.6 mm -- so leaving it out overstates the target
        load by a factor of three and makes the seeding requirement
        unreachable.
    """
    lam_q = eich_lambda_q(B_pol)                      # [mm]
    lam_int = lam_q + 1.64 * max(s_spread_mm, 0.0)    # [mm]
    lam_m = lam_int * 1e-3

    # wetted area: the channel is spread by flux expansion and the tilt
    A_wet = (2.0 * np.pi * R0 * lam_m * flux_expansion
             * n_div_targets / max(np.sin(np.radians(tilt_deg)) * 0 + 1.0, 1.0))
    A_wet = 2.0 * np.pi * R0 * lam_m * flux_expansion * n_div_targets

    P_sol = max(P_sep * (1.0 - f_rad_seed), 1e-3)
    # parallel flux at the midplane entrance to the SOL
    B_ratio = np.hypot(B0, B_pol) / max(B_pol, 1e-3)
    q_par = P_sol / (2.0 * np.pi * R0 * lam_m) * B_ratio / n_div_targets  # MW/m^2

    L_par = connection_length(R0, q95)

    # --- solve the two-point system for the target temperature -----------
    # Residual: sheath-transmitted flux minus the flux that survives the
    # volumetric losses.  Both sides are monotone in T_t with opposite
    # slopes, so the root is bracketed and bisection is unconditionally
    # robust -- the fixed-point relaxation this replaces could sit on the
    # clamp instead of converging.
    def residual(T_t: float) -> float:
        T_u = (T_t ** 3.5 + 3.5 * q_par * 1e6 * L_par / KAPPA_0E) ** (2.0 / 7.0)
        f_mom = 1.0 / (1.0 + (5.0 / max(T_t, 0.3)) ** 2)
        f_pow = 1.0 / (1.0 + (3.0 / max(T_t, 0.3)) ** 2)
        c_s = 9.79e3 * np.sqrt(2.0 * T_t / A_mass)
        n_t = max(f_mom * n_u * T_u / (2.0 * T_t), 1e14)
        q_sheath = GAMMA_SHEATH * n_t * c_s * T_t * E_CHARGE / 1e6
        return q_sheath - q_par * f_pow

    n_u = n_sep20 * 1e20
    # Both loss factors vanish as the target cools, so the residual has a
    # spurious cold root as well as the physical one.  Walk in from the hot
    # end and take the FIRST sign change: that is the attached branch, and
    # the plasma only detaches when no hot root exists at all.
    # up to 5 keV: at ITER's unseeded parallel flux the sheath only
    # balances at a target temperature of order a keV, which is the
    # model saying the plate cannot survive without seeding
    grid = np.geomspace(5000.0, 0.2, 300)
    vals = np.array([residual(T) for T in grid])
    idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    no_root = len(idx) == 0
    if no_root:
        T_t = float(grid[int(np.argmin(np.abs(vals)))])
    else:
        lo, hi = grid[idx[0] + 1], grid[idx[0]]
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            if residual(hi) * residual(mid) <= 0:
                lo = mid
            else:
                hi = mid
        T_t = 0.5 * (lo + hi)
    T_u = (T_t ** 3.5 + 3.5 * q_par * 1e6 * L_par / KAPPA_0E) ** (2.0 / 7.0)

    c_s = 9.79e3 * np.sqrt(2.0 * max(T_t, 0.2) / A_mass)
    f_mom = 1.0 / (1.0 + (5.0 / max(T_t, 0.5)) ** 2)
    f_pow = 1.0 / (1.0 + (3.0 / max(T_t, 0.5)) ** 2)
    n_t = max(f_mom * n_u * T_u / (2.0 * max(T_t, 0.2)), 1e15)

    P_target = P_sol * f_pow
    q_target = P_target / max(A_wet, 1e-6)

    if no_root:
        # The sheath cannot transmit q_par at any target temperature: there
        # is no steady attached solution and the exhaust has to be radiated
        # before it arrives.  Reporting a cold target here instead would
        # look like a comfortable detached divertor, which is the opposite
        # of what the model just found.
        regime, detached = "шешім жоқ — себу міндетті", False
    elif T_t < 2.0:
        regime, detached = "detached", True
    elif T_t < 8.0:
        regime, detached = "partially detached", True
    else:
        regime, detached = "attached", False

    f_rad_div = 1.0 - P_target / max(P_sep, 1e-6)

    return DivertorState(
        lambda_q_mm=float(lam_q), lambda_int_mm=float(lam_int), q_par_MWm2=float(q_par),
        T_upstream_eV=float(T_u), T_target_eV=float(T_t),
        n_target_20=float(n_t / 1e20), q_target_MWm2=float(q_target),
        f_rad_div=float(np.clip(f_rad_div, 0.0, 0.999)), detached=detached,
        no_attached_root=bool(no_root), regime=regime,
        A_wetted=float(A_wet), P_target=float(P_target),
    )


def seeding_for_target(P_sep: float, q_limit: float = 10.0, **kw) -> float:
    """Impurity seeding fraction needed to keep the target below ``q_limit``.

    Answers the question a divertor designer actually asks: given this
    exhaust power, how much of it has to be radiated away before the plate?
    """
    lo, hi = 0.0, 0.98
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        st = two_point(P_sep, f_rad_seed=mid, **kw)
        # more seeding is needed while either the load is too high OR the
        # sheath still has no solution to carry the flux at all
        if st.no_attached_root or st.q_target_MWm2 > q_limit:
            lo = mid
        else:
            hi = mid
    return hi
