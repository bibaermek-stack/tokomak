"""Plasma boundary geometry.

The boundary is the standard shaped-tokamak parametrisation

    R(t) = R0 + a cos(t + arcsin(delta) sin t)
    Z(t) = kappa a sin t

Volume and surface come from Pappus/Green applied to that curve rather than
from an elliptical approximation, which overstates ITER's volume by 8%.  A
diverted plasma loses the private-flux region below the X-point that the
smooth curve still encloses; ``X_POINT_TRIM`` removes it and reproduces
ITER's published 840 m^3 / 680 m^2 to better than 0.5%.
"""

from __future__ import annotations

import numpy as np

from .machines import Machine

#: Fraction of the smooth-boundary volume that survives the X-point notch.
#: Calibrated on ITER (840 m^3 published against 849.5 m^3 smooth).
X_POINT_TRIM = 0.988


def boundary(R0: float, a: float, kappa: float, delta: float,
             n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Sample the plasma boundary. Returns (R, Z), closed (last == first)."""
    t = np.linspace(0.0, 2.0 * np.pi, n + 1)
    ds = np.arcsin(np.clip(delta, -0.99, 0.99))
    R = R0 + a * np.cos(t + ds * np.sin(t))
    Z = kappa * a * np.sin(t)
    return R, Z


def shape_integrals(R0: float, a: float, kappa: float, delta: float,
                    diverted: bool = True, n: int = 4096) -> dict:
    """Volume, surface, poloidal perimeter and cross-section area.

    Volume uses ``V = |int pi R^2 dZ|`` (Pappus via Green's theorem, exact
    for a body of revolution); surface revolves the perimeter about the axis.
    """
    R, Z = boundary(R0, a, kappa, delta, n)
    dR, dZ = np.diff(R), np.diff(Z)
    Rm = 0.5 * (R[:-1] + R[1:])
    seg = np.hypot(dR, dZ)

    V = abs(np.sum(np.pi * (R[:-1] ** 2 + R[1:] ** 2) * 0.5 * dZ))
    S = float(np.sum(2.0 * np.pi * Rm * seg))
    L_pol = float(np.sum(seg))
    # cross-sectional area, shoelace
    A_cs = abs(0.5 * np.sum(R[:-1] * Z[1:] - R[1:] * Z[:-1]))

    trim = X_POINT_TRIM if diverted else 1.0
    V *= trim
    S *= trim
    A_cs *= trim
    return {
        "V": float(V),
        "S": float(S),
        "L_pol": L_pol,
        "A_cs": float(A_cs),
        # Areal elongation AS IPB98(y,2) DEFINES IT, kappa_a = V / (2 pi^2 R a^2).
        # This is not the same as the cross-section ratio A_cs / (pi a^2): a
        # D-shape's centroid sits outboard of R0, so the two differ by ~4%.
        # ITER's quoted kappa_a = 1.7 is the volume definition.
        "kappa_a": float(V / (2.0 * np.pi ** 2 * R0 * a * a)),
        "kappa_areal": float(A_cs / (np.pi * a * a)),
    }


def attach_geometry(m: Machine) -> Machine:
    """Fill a machine's V / S / L_pol / A_cs / kappa_a in place."""
    g = shape_integrals(m.R0, m.a, m.kappa_x, m.delta_x, m.diverted)
    m.V = g["V"]
    m.S = g["S"]
    m.L_pol = g["L_pol"]
    m.A_cs = g["A_cs"]
    m.kappa_a = g["kappa_a"]
    return m


def trapped_fraction(eps: float) -> float:
    """Trapped-particle fraction on a surface of inverse aspect ratio eps.

    The standard large-aspect-ratio result f_t = 1.46 sqrt(eps) is only good
    to eps ~ 0.2; this is the Lin-Liu / Miller form that stays bounded and is
    what the Sauter bootstrap coefficients expect.
    """
    e = float(np.clip(eps, 0.0, 0.95))
    return float(1.0 - (1.0 - e) ** 2 / (np.sqrt(1.0 - e * e) * (1.0 + 1.46 * np.sqrt(e))))


def b_poloidal(Ip_MA: float, L_pol: float) -> float:
    """Average poloidal field from the current and the poloidal perimeter."""
    from .constants import MU0
    return MU0 * Ip_MA * 1e6 / L_pol
