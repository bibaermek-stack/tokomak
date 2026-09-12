"""Physical constants and unit conventions used across the package.

Unit convention throughout the package, unless a docstring says otherwise:

    lengths        m
    magnetic field T
    plasma current MA
    density        1e20 m^-3   (written n20)
    temperature    keV
    power          MW
    energy         MJ
    time           s

The two averages that appear everywhere are distinguished explicitly:

    <X>_V   volume average,           (1/V) int X dV
    <X>_n   density-weighted average, int n X dV / int n dV

Stored energy is exact in the second convention:
``W = (3/2) (<n_e>_V <T_e>_n + <n_i>_V <T_i>_n) V``.
"""

from __future__ import annotations

import math

# --- fundamental ----------------------------------------------------------
E_CHARGE = 1.602176634e-19          # C
KEV_J = 1.602176634e-16             # J per keV
MU0 = 4.0e-7 * math.pi              # H/m
EPS0 = 8.8541878128e-12             # F/m
M_E = 9.1093837015e-31              # kg
M_P = 1.67262192369e-27             # kg
C_LIGHT = 2.99792458e8              # m/s

# --- D-T fusion -----------------------------------------------------------
E_FUSION_J = 2.8183e-12             # 17.59 MeV per D-T reaction
E_ALPHA_KEV = 3520.0                # alpha birth energy
E_NEUTRON_KEV = 14070.0             # neutron energy
F_ALPHA = E_ALPHA_KEV / (E_ALPHA_KEV + E_NEUTRON_KEV)   # 0.20014

# --- derived scale factors ------------------------------------------------
MW = 1.0e6
#: (3/2) k_B expressed in MJ per (1e20 m^-3 * keV * m^3)
W_UNIT = 1.5 * KEV_J * 1e20 / MW    # 2.40327e-2
#: pressure in Pa from n20 [1e20 m^-3] and T [keV]
P_UNIT = KEV_J * 1e20               # 1.60218e4

#: Coulomb logarithm used where a constant is adequate for a 0-D/1-D model.
LN_LAMBDA = 17.0

__all__ = [
    "E_CHARGE", "KEV_J", "MU0", "EPS0", "M_E", "M_P", "C_LIGHT",
    "E_FUSION_J", "E_ALPHA_KEV", "E_NEUTRON_KEV", "F_ALPHA",
    "MW", "W_UNIT", "P_UNIT", "LN_LAMBDA",
]
