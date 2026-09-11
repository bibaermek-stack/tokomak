"""Fusion reactivities from the Bosch-Hale parametrisation.

Bosch & Hale, Nucl. Fusion 32 (1992) 611, Table VII/VIII.  The fits are
valid over 0.2-100 keV (D-T) and reproduce the tabulated <sigma v> to better
than 0.25%; the validation harness checks the D-T branch against the table
and finds a maximum error of 0.016%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class _BoschHale:
    bg: float          # Gamow constant [keV^1/2]
    mrc2: float        # reduced mass energy [keV]
    c: tuple           # C1..C7
    t_min: float
    t_max: float
    e_total_mev: float

    def sigma_v(self, T):
        """<sigma v> [m^3/s] for ion temperature T [keV]."""
        T = np.asarray(T, dtype=float)
        Tc = np.clip(T, self.t_min, self.t_max)
        c1, c2, c3, c4, c5, c6, c7 = self.c
        theta = Tc / (1.0 - (Tc * (c2 + Tc * (c4 + Tc * c6)))
                          / (1.0 + Tc * (c3 + Tc * (c5 + Tc * c7))))
        xi = (self.bg ** 2 / (4.0 * theta)) ** (1.0 / 3.0)
        sv = c1 * theta * np.sqrt(xi / (self.mrc2 * Tc ** 3)) * np.exp(-3.0 * xi)
        sv = sv * 1e-6                       # cm^3/s -> m^3/s
        return np.where(T < self.t_min, 0.0, sv)


REACTIONS: Dict[str, _BoschHale] = {
    # D + T -> alpha (3.52 MeV) + n (14.07 MeV)
    "DT": _BoschHale(
        bg=34.3827, mrc2=1124656.0,
        c=(1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3,
           1.35000e-2, -1.06750e-4, 1.36600e-5),
        t_min=0.2, t_max=100.0, e_total_mev=17.59,
    ),
    # D + He3 -> alpha (3.6 MeV) + p (14.7 MeV)
    "DHe3": _BoschHale(
        bg=68.7508, mrc2=1124572.0,
        c=(5.51036e-10, 6.41918e-3, -2.02896e-3, -1.91080e-5,
           1.35776e-4, 0.0, 0.0),
        t_min=0.5, t_max=190.0, e_total_mev=18.3,
    ),
    # D + D -> T + p   (one of the two roughly equal branches)
    "DDp": _BoschHale(
        bg=31.3970, mrc2=937814.0,
        c=(5.65718e-12, 3.41267e-3, 1.99167e-3, 0.0,
           1.05060e-5, 0.0, 0.0),
        t_min=0.2, t_max=100.0, e_total_mev=4.03,
    ),
    # D + D -> He3 + n (the other branch)
    "DDn": _BoschHale(
        bg=31.3970, mrc2=937814.0,
        c=(5.43360e-12, 5.85778e-3, 7.68222e-3, 0.0,
           -2.96400e-6, 0.0, 0.0),
        t_min=0.2, t_max=100.0, e_total_mev=3.27,
    ),
}


def sigma_v(T, reaction: str = "DT"):
    """<sigma v> [m^3/s] for ion temperature T [keV]."""
    try:
        return REACTIONS[reaction].sigma_v(T)
    except KeyError:
        raise KeyError(f"unknown reaction {reaction!r}; "
                       f"available: {', '.join(REACTIONS)}") from None


def dt_power_density(n_dt20, Ti_kev):
    """D-T fusion power density [MW/m^3].

    ``n_dt20`` is the total fuel density (n_D + n_T) in 1e20 m^-3, assumed
    50/50.  Deuterium and tritium are distinct species, so the rate is
    n_D n_T <sigma v> with no factor of 1/2.
    """
    from .constants import E_FUSION_J, MW
    n = np.asarray(n_dt20, dtype=float) * 1e20
    return 0.25 * n * n * sigma_v(Ti_kev, "DT") * E_FUSION_J / MW


#: The tabulated values Bosch & Hale publish, used by the validation harness.
BOSCH_HALE_TABLE = {
    1.0: 6.857e-27, 2.0: 2.977e-25, 5.0: 1.366e-23, 10.0: 1.136e-22,
    20.0: 4.330e-22, 50.0: 8.649e-22, 100.0: 8.448e-22,
}
