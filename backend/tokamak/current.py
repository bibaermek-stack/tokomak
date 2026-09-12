"""Plasma current: resistivity, bootstrap, drive, diffusion and flux budget.

Three things live here that a 0-D burn model cannot express:

* **Neoclassical resistivity** -- Sauter's trapped-particle correction, which
  is a factor of two to three above Spitzer at reactor aspect ratio.
* **Bootstrap current** -- the self-driven current the pressure gradient
  produces, 15% of Ip in ITER's baseline and the whole point of a steady-
  state scenario.  Coefficients follow Sauter et al.
* **Current diffusion and the flux budget** -- the current penetrates on the
  resistive timescale, and the central solenoid has a finite volt-second
  store, which is what actually sets the pulse length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import solve_banded

from .constants import MU0, E_CHARGE, LN_LAMBDA, P_UNIT
from .geometry import trapped_fraction


# ---------------------------------------------------------------------------
# Resistivity
# ---------------------------------------------------------------------------
def spitzer_resistivity(Te_kev, z_eff: float):
    """Parallel Spitzer resistivity [Ohm m]."""
    Te = np.maximum(np.asarray(Te_kev, dtype=float), 0.02)
    return 2.8e-8 * z_eff / Te ** 1.5


def collisionality(rho, ne20, Te_kev, z_eff: float, R0: float, eps: float,
                   q_prof):
    """Electron collisionality nu*_e on each flux surface."""
    Te = np.maximum(np.asarray(Te_kev, dtype=float), 0.02)
    n = np.maximum(np.asarray(ne20, dtype=float), 1e-4) * 1e20
    eps_loc = np.maximum(eps * np.asarray(rho, dtype=float), 1e-3)
    # nu*_e = 6.921e-18 R q n_e Z_eff lnL / (T_e^2 eps^1.5)   (Sauter)
    return (6.921e-18 * R0 * np.maximum(q_prof, 0.3) * n * z_eff * LN_LAMBDA
            / ((Te * 1e3) ** 2 * eps_loc ** 1.5))


def neoclassical_resistivity(rho, ne20, Te_kev, z_eff: float,
                             R0: float, eps: float, q_prof):
    """Neoclassical parallel resistivity [Ohm m], Sauter form.

    sigma_neo / sigma_Sp = 1 - f_teff (1 + 0.36/Z) + f_teff^2 (0.59/Z)
                             - 0.23 f_teff^3 / Z
    with a collisionality-dependent effective trapped fraction.
    """
    Z = max(z_eff, 1.0)
    ft = np.array([trapped_fraction(eps * max(r, 1e-3))
                   for r in np.atleast_1d(rho)])
    nue = collisionality(rho, ne20, Te_kev, z_eff, R0, eps, q_prof)
    ft33 = ft / (1.0 + (0.55 - 0.1 * ft) * np.sqrt(nue)
                 + 0.45 * (1.0 - ft) * nue / Z ** 1.5)
    ratio = (1.0 - ft33 * (1.0 + 0.36 / Z)
             + ft33 ** 2 * 0.59 / Z - 0.23 * ft33 ** 3 / Z)
    ratio = np.clip(ratio, 0.05, 1.0)
    return spitzer_resistivity(Te_kev, z_eff) / ratio


# ---------------------------------------------------------------------------
# Bootstrap current (Sauter coefficients)
# ---------------------------------------------------------------------------
def _F31(X, Z):
    return ((1.0 + 1.4 / (Z + 1.0)) * X - 1.9 / (Z + 1.0) * X ** 2
            + 0.3 / (Z + 1.0) * X ** 3 + 0.2 / (Z + 1.0) * X ** 4)


def _F32ee(X, Z):
    return ((0.05 + 0.62 * Z) / (Z * (1.0 + 0.44 * Z)) * (X - X ** 4)
            + 1.0 / (1.0 + 0.22 * Z) * (X ** 2 - X ** 4 - 1.2 * (X ** 3 - X ** 4))
            + 1.2 / (1.0 + 0.5 * Z) * X ** 4)


def _F32ei(Y, Z):
    return (-(0.56 + 1.93 * Z) / (Z * (1.0 + 0.44 * Z)) * (Y - Y ** 4)
            + 4.95 / (1.0 + 2.48 * Z) * (Y ** 2 - Y ** 4 - 0.55 * (Y ** 3 - Y ** 4))
            - 1.2 / (1.0 + 0.5 * Z) * Y ** 4)


@dataclass
class BootstrapResult:
    j_bs: np.ndarray        # <j_bs . B> / B0, i.e. a current density [A/m^2]
    I_bs: float             # [MA]
    f_bs: float             # I_bs / Ip
    L31: np.ndarray
    L32: np.ndarray
    L34: np.ndarray
    alpha: np.ndarray


def bootstrap(rho, ne20, Te_kev, Ti_kev, q_prof, *, R0: float, a: float,
              B0: float, Ip: float, z_eff: float, f_ion: float,
              A_cs: float) -> BootstrapResult:
    """Bootstrap current density and fraction, Sauter formulation.

    The radial coordinate is rho = r/a; gradients are taken with respect to
    the poloidal flux through dpsi/drho = a B_pol(rho), with
    B_pol(rho) = mu0 I(rho) / (2 pi a rho).

    Validated on ITER's Q=10 point, where it returns f_bs close to the
    published 0.15 -- see ``tests/test_current.py``.
    """
    rho = np.asarray(rho, dtype=float)
    Z = max(z_eff, 1.0)
    eps = a / R0
    ne = np.maximum(np.asarray(ne20, dtype=float), 1e-4)
    Te = np.maximum(np.asarray(Te_kev, dtype=float), 0.02)
    Ti = np.maximum(np.asarray(Ti_kev, dtype=float), 0.02)
    ni = ne * f_ion

    ft = np.array([trapped_fraction(eps * max(r, 1e-3)) for r in rho])
    nue = collisionality(rho, ne, Te, z_eff, R0, eps, q_prof)
    nui = nue * (Ti / Te) ** 2 * np.sqrt(2.0)      # rough ion scaling

    X31 = ft / (1.0 + (1.0 - 0.1 * ft) * np.sqrt(nue)
                + 0.5 * (1.0 - ft) * nue / Z)
    L31 = _F31(X31, Z)

    X32e = ft / (1.0 + 0.26 * (1.0 - ft) * np.sqrt(nue)
                 + 0.18 * (1.0 - 0.37 * ft) * nue / np.sqrt(Z))
    Y32ei = ft / (1.0 + (1.0 + 0.6 * ft) * np.sqrt(nue)
                  + 0.85 * (1.0 - 0.37 * ft) * nue * (1.0 + Z))
    L32 = _F32ee(X32e, Z) + _F32ei(Y32ei, Z)

    X34 = ft / (1.0 + (1.0 - 0.1 * ft) * np.sqrt(nue)
                + 0.5 * (1.0 - 0.5 * ft) * nue / Z)
    L34 = _F31(X34, Z)

    a0 = -1.17 * (1.0 - ft) / (1.0 - 0.22 * ft - 0.19 * ft ** 2)
    alpha = (((a0 + 0.25 * (1.0 - ft ** 2) * np.sqrt(nui))
              / (1.0 + 0.5 * np.sqrt(nui)))
             + 0.315 * nui ** 2 * ft ** 6) / (1.0 + 0.15 * nui ** 2 * ft ** 6)

    # --- pressures and gradients -----------------------------------------
    pe = P_UNIT * ne * Te
    pi = P_UNIT * ni * Ti
    p = pe + pi
    R_pe = pe / np.maximum(p, 1e-9)

    # Enclosed current: the cylindrical relation gives the SHAPE from q(rho),
    # but its absolute level is the cylindrical q, which for ITER is 1.95
    # against the real q95 = 3.  Renormalise so that I(1) = Ip; otherwise
    # B_pol comes out 1.5x low and every gradient with respect to psi -- and
    # so the bootstrap current -- is overestimated by the same factor.
    kappa_a = A_cs / (np.pi * a * a)
    I_shape = (2.0 * np.pi * a ** 2 * kappa_a * B0 * rho ** 2
               / (MU0 * R0 * np.maximum(q_prof, 0.3)))
    I_encl = I_shape * (Ip * 1e6 / max(I_shape[-1], 1e-9))
    B_pol = MU0 * I_encl / (2.0 * np.pi * a * np.maximum(rho, 1e-3) *
                            np.sqrt(max(kappa_a, 0.5)))
    # dpsi/drho = R B_pol dr/drho = R0 B_pol a
    dpsi_drho = R0 * np.maximum(B_pol, 1e-4) * a          # [Wb/rad per rho]

    def dln(f):
        return np.gradient(np.log(np.maximum(f, 1e-30)), rho) / dpsi_drho

    I_flux = B0 * R0                                     # F = R B_phi
    j_bs_B = -I_flux * pe * (
        L31 * (p / np.maximum(pe, 1e-9)) * dln(p)
        + L32 * dln(Te)
        + L34 * alpha * (1.0 - R_pe) / np.maximum(R_pe, 1e-6) * dln(Ti)
    )
    j_bs = j_bs_B / B0                                   # <j.B>/B0 [A/m^2]
    j_bs[0] = 0.0                                        # no gradient on axis

    # integrate over the cross-section: dA = 2 pi a^2 kappa_a rho drho
    dA = 2.0 * np.pi * a ** 2 * kappa_a * rho
    I_bs = float(np.trapezoid(j_bs * dA, rho)) / 1e6      # [MA]
    return BootstrapResult(j_bs=j_bs, I_bs=I_bs, f_bs=I_bs / max(Ip, 1e-6),
                           L31=L31, L32=L32, L34=L34, alpha=alpha)


# ---------------------------------------------------------------------------
# Non-inductive drive
# ---------------------------------------------------------------------------
def nb_current_drive(P_nbi: float, E_beam_kev: float, ne20_avg: float,
                     Te_kev_avg: float, R0: float, z_eff: float) -> float:
    """Neutral-beam driven current [MA].

    Uses the current-drive figure of merit
    ``gamma_CD = I R n / P`` [1e20 A/W/m^2], which for negative-ion beams at
    ITER-like temperatures is about 0.3 and rises roughly linearly with T_e.
    """
    if P_nbi <= 0:
        return 0.0
    gamma = 0.024 * max(Te_kev_avg, 0.1) * (5.0 / (5.0 + z_eff)) * \
        min(1.0 + E_beam_kev / 1000.0, 2.0)
    return gamma * P_nbi / (R0 * max(ne20_avg, 1e-3)) / 1.0


def ec_current_drive(P_ecrf: float, ne20_avg: float, Te_kev_avg: float,
                     R0: float, z_eff: float) -> float:
    """Electron-cyclotron driven current [MA]; gamma_CD ~ 0.2 at ITER."""
    if P_ecrf <= 0:
        return 0.0
    gamma = 0.015 * max(Te_kev_avg, 0.1) * (5.0 / (5.0 + z_eff))
    return gamma * P_ecrf / (R0 * max(ne20_avg, 1e-3))


# ---------------------------------------------------------------------------
# Current diffusion and the volt-second budget
# ---------------------------------------------------------------------------
@dataclass
class FluxBudget:
    """Central-solenoid volt-second accounting.

    The available flux is what limits an inductive pulse.  ``consumed``
    accumulates ``V_loop dt``; when it reaches ``available`` the discharge
    has to end, which is the physical origin of ITER's 400 s flat-top and
    the thing a 0-D model with a hard-coded shot length cannot express.
    """
    available: float           # [Wb]
    consumed: float = 0.0      # [Wb]

    def step(self, v_loop: float, dt: float) -> None:
        self.consumed += max(v_loop, 0.0) * dt

    @property
    def remaining(self) -> float:
        return self.available - self.consumed

    @property
    def fraction(self) -> float:
        return self.consumed / max(self.available, 1e-9)

    @property
    def exhausted(self) -> bool:
        return self.consumed >= self.available


def cs_flux_capacity(R0: float, a: float, Ip: float, li: float = 0.9,
                     c_ejima: float = 0.45) -> float:
    """Volt-seconds a machine must supply for one inductive pulse [Wb].

    Internal + external inductive flux plus the Ejima resistive allowance,
    which is what the solenoid has to beat to reach flat-top at all.
    """
    eps = a / R0
    L_p = MU0 * R0 * (np.log(8.0 / eps) + li / 2.0 - 2.0)
    psi_ind = L_p * Ip * 1e6
    psi_res = c_ejima * MU0 * R0 * Ip * 1e6
    return float(psi_ind + psi_res)


def diffuse_current(rho: np.ndarray, psi: np.ndarray, eta: np.ndarray,
                    j_ni: np.ndarray, a: float, R0: float, B0: float,
                    dt: float, v_surf: float) -> np.ndarray:
    """One implicit step of the cylindrical current-diffusion equation.

        dpsi/dt = (eta / mu0 a^2) (1/rho) d/drho (rho dpsi/drho)
                  - eta R0 j_ni

    with ``v_surf`` the applied surface loop voltage as the edge condition.
    This is the reduced form used in transport codes when full flux
    coordinates are not carried; it reproduces the resistive penetration
    time, which is what matters for how q0 and the sawtooth onset evolve.
    """
    n = len(rho)
    dr = rho[1] - rho[0]
    coef = eta / (MU0 * a ** 2)

    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)
    rhs = np.zeros(n)

    for i in range(1, n - 1):
        r = rho[i]
        aw = coef[i] * (r - 0.5 * dr) / (r * dr ** 2)
        ae = coef[i] * (r + 0.5 * dr) / (r * dr ** 2)
        lower[i] = -dt * aw
        upper[i] = -dt * ae
        diag[i] = 1.0 + dt * (aw + ae)
        rhs[i] = psi[i] - dt * eta[i] * R0 * j_ni[i]

    ae0 = coef[0] * 2.0 / dr ** 2
    diag[0] = 1.0 + dt * ae0
    upper[0] = -dt * ae0
    rhs[0] = psi[0] - dt * eta[0] * R0 * j_ni[0]

    diag[-1] = 1.0
    rhs[-1] = psi[-1] + dt * v_surf / (2.0 * np.pi)

    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)
