"""Green's functions of axisymmetric current loops.

Every conductor in the co-simulation -- PF and central-solenoid windings,
vacuum-vessel segments and the plasma itself -- is represented by coaxial
circular filaments.  Everything the circuit model, the force balance and the
synthetic diagnostics need follows from two closed forms:

    M(R1,Z1; R2,Z2) = mu0 sqrt(R1 R2) [ (2/k - k) K(k) - (2/k) E(k) ]
    k^2 = 4 R1 R2 / ((R1 + R2)^2 + (Z1 - Z2)^2)

the mutual inductance [H] of two loops (= total poloidal flux [Wb] through
loop 1 per ampere in loop 2), and the field [T per A] of a loop at (Rc, Zc)
evaluated at (R, Z):

    B_R = mu0/(2 pi) dz / (R s) [ -K + (Rc^2 + R^2 + dz^2) / d^2 E ]
    B_Z = mu0/(2 pi)  1 / s     [  K + (Rc^2 - R^2 - dz^2) / d^2 E ]
    s^2 = (R + Rc)^2 + dz^2,   d^2 = (R - Rc)^2 + dz^2,   dz = Z - Zc

They satisfy B_R = -(1/2 pi R) dPsi/dZ and B_Z = (1/2 pi R) dPsi/dR with
Psi = M I, which the tests check by finite differences.

scipy's ``ellipk`` / ``ellipe`` take the parameter m = k^2, not k.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipk

from ..constants import MU0

#: m = k^2 is clipped below 1 so coincident filaments stay finite; callers
#: that need a self-inductance use :func:`self_inductance` instead.
_M_MAX = 1.0 - 1e-12


def mutual(R1, Z1, R2, Z2) -> np.ndarray:
    """Mutual inductance [H] between loops, broadcasting over the inputs."""
    R1, Z1, R2, Z2 = np.broadcast_arrays(*(np.asarray(x, float)
                                           for x in (R1, Z1, R2, Z2)))
    m = 4.0 * R1 * R2 / ((R1 + R2) ** 2 + (Z1 - Z2) ** 2)
    m = np.clip(m, 1e-14, _M_MAX)
    k = np.sqrt(m)
    return MU0 * np.sqrt(R1 * R2) * ((2.0 / k - k) * ellipk(m)
                                     - 2.0 / k * ellipe(m))


def field(Rc, Zc, R, Z) -> tuple[np.ndarray, np.ndarray]:
    """(B_R, B_Z) [T/A] at (R, Z) from a unit-current loop at (Rc, Zc)."""
    Rc, Zc, R, Z = np.broadcast_arrays(*(np.asarray(x, float)
                                         for x in (Rc, Zc, R, Z)))
    dz = Z - Zc
    s2 = (R + Rc) ** 2 + dz ** 2
    d2 = np.maximum((R - Rc) ** 2 + dz ** 2, 1e-12)
    m = np.clip(4.0 * R * Rc / s2, 0.0, _M_MAX)
    K, E = ellipk(m), ellipe(m)
    s = np.sqrt(s2)
    c = MU0 / (2.0 * np.pi)
    BR = c * dz / (R * s) * (-K + (Rc ** 2 + R ** 2 + dz ** 2) / d2 * E)
    BZ = c / s * (K + (Rc ** 2 - R ** 2 - dz ** 2) / d2 * E)
    return BR, BZ


def self_inductance(R: float, r_wire: float) -> float:
    """Self-inductance [H] of a single circular loop of wire radius r_wire."""
    return float(MU0 * R * (np.log(8.0 * R / r_wire) - 1.75))


def mutual_and_field(Rs, Zs, R, Z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M, B_R, B_Z of source loops (Rs, Zs) at observation loops (R, Z).

    One elliptic-integral evaluation serves all three, which is what the
    simulator's inner Newton loop spends its time on.  Broadcasting as in
    :func:`mutual`; the field is per ampere in the source loop.
    """
    Rs, Zs, R, Z = np.broadcast_arrays(*(np.asarray(x, float)
                                         for x in (Rs, Zs, R, Z)))
    dz = Z - Zs
    s2 = (R + Rs) ** 2 + dz ** 2
    d2 = np.maximum((R - Rs) ** 2 + dz ** 2, 1e-12)
    m = np.clip(4.0 * R * Rs / s2, 1e-14, _M_MAX)
    K, E = ellipk(m), ellipe(m)
    k = np.sqrt(m)
    s = np.sqrt(s2)
    M = MU0 * np.sqrt(R * Rs) * ((2.0 / k - k) * K - 2.0 / k * E)
    c = MU0 / (2.0 * np.pi)
    BR = c * dz / (R * s) * (-K + (Rs ** 2 + R ** 2 + dz ** 2) / d2 * E)
    BZ = c / s * (K + (Rs ** 2 - R ** 2 - dz ** 2) / d2 * E)
    return M, BR, BZ
