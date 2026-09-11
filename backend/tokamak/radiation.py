"""Radiated power and plasma composition.

Three channels:

* **Bremsstrahlung** -- the standard 5.35e-3 Z_eff n20^2 sqrt(T_e) MW/m^3,
  with the relativistic enhancement that reaches ~10% at 25 keV.
* **Synchrotron** -- Trubnikov's optically-thick scaling shape reduced by
  wall reflection.  The absolute level is anchored to the Albajar (2001)
  ITER result, since a full radiation-transport treatment is out of scope.
* **Line radiation** -- per-species cooling rates.  Light impurities are
  fully stripped above a few keV and then contribute only bremsstrahlung;
  tungsten never is, so it dominates core line radiation in a reactor.

The impurity inventory also sets the fuel dilution and Z_eff, which is why
composition lives here rather than in the transport solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Impurity species
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Species:
    """An impurity species.

    ``z_mean`` is the mean charge at core temperatures, not the nuclear
    charge: tungsten sits near Z = 60 at 8-25 keV, not 74.
    """
    name: str
    z_mean: float
    a_mass: float
    #: cooling-rate plateau [W m^3] and the temperature below which it climbs
    l_plateau: float
    t_knee: float
    knee_slope: float = 0.55
    fully_stripped_above: float = 0.0

    def cooling_rate(self, Te_kev):
        """Radiative cooling rate L_z(T_e) [W m^3], non-coronal-free.

        Above ``fully_stripped_above`` the ion has no bound electrons left
        and radiates only through bremsstrahlung, which is accounted for
        separately -- so L_z goes to zero there rather than to a plateau.
        """
        Te = np.asarray(Te_kev, dtype=float)
        L = np.where(Te >= self.t_knee, self.l_plateau,
                     self.l_plateau * np.power(
                         np.maximum(self.t_knee / np.maximum(Te, 1e-3), 1.0),
                         self.knee_slope))
        if self.fully_stripped_above > 0:
            L = np.where(Te > self.fully_stripped_above, 0.0, L)
        return np.where(Te <= 0.02, 0.0, L)


#: Cooling-rate plateaus follow the ADAS curves collected by Puetterich et al.,
#: Nucl. Fusion 50 (2010) 025012.  Beryllium and neon are fully stripped a
#: little above 1-2 keV, so their core line radiation vanishes.
SPECIES: Dict[str, Species] = {
    "Be": Species("Be", z_mean=4.0, a_mass=9.0, l_plateau=1.0e-34,
                  t_knee=0.1, fully_stripped_above=2.0),
    "C": Species("C", z_mean=6.0, a_mass=12.0, l_plateau=2.0e-34,
                 t_knee=0.2, fully_stripped_above=3.0),
    "Ne": Species("Ne", z_mean=10.0, a_mass=20.0, l_plateau=6.0e-34,
                  t_knee=0.5, fully_stripped_above=4.0),
    "Ar": Species("Ar", z_mean=17.5, a_mass=40.0, l_plateau=3.0e-33,
                  t_knee=2.0, fully_stripped_above=25.0),
    "Kr": Species("Kr", z_mean=32.0, a_mass=84.0, l_plateau=1.2e-32,
                  t_knee=3.0),
    "W": Species("W", z_mean=60.0, a_mass=184.0, l_plateau=4.3e-32,
                 t_knee=3.0),
    "He": Species("He", z_mean=2.0, a_mass=4.0, l_plateau=0.0,
                  t_knee=0.1, fully_stripped_above=0.5),
}


@dataclass
class Composition:
    """Impurity inventory, as fractions of the electron density.

    The default is ITER's assumed mix: beryllium eroded off the first wall,
    a trace of tungsten sputtered off the divertor targets, and neon seeded
    to radiate in the divertor.  Helium ash is tracked separately by the
    transport solver and passed in, since it evolves with the burn.
    """
    fractions: Dict[str, float] = field(default_factory=lambda: {
        "Be": 0.0200, "Ne": 0.00214, "W": 3.80e-5,
    })

    def resolve(self, f_he: float = 0.0) -> dict:
        """Fuel dilution, Z_eff and the ion inventory for a given ash level.

        Charge neutrality fixes the fuel fraction:
        ``f_DT = 1 - sum_j Z_j f_j``.
        """
        f_he = float(np.clip(f_he, 0.0, 0.4))
        sum_z = 2.0 * f_he
        sum_z2 = 4.0 * f_he
        sum_ion = f_he
        z2a = f_he * 4.0 / 4.0
        for name, frac in self.fractions.items():
            sp = SPECIES[name]
            sum_z += sp.z_mean * frac
            sum_z2 += sp.z_mean ** 2 * frac
            sum_ion += frac
            z2a += frac * sp.z_mean ** 2 / sp.a_mass
        f_dt = max(0.0, 1.0 - sum_z)
        return {
            "f_dt": f_dt,
            "f_he": f_he,
            "f_ion": f_dt + sum_ion,           # n_i / n_e
            "z_eff": float(np.clip(f_dt + sum_z2, 1.0, 8.0)),
            # sum_j (n_j/n_e) Z_j^2 / A_j -- sets equipartition and E_crit
            "z2a": z2a,
        }

    def with_seeding(self, species: str, fraction: float) -> "Composition":
        """A copy with one species' fraction replaced."""
        f = dict(self.fractions)
        f[species] = fraction
        return Composition(f)


# ---------------------------------------------------------------------------
# Radiated power densities  [MW/m^3]
# ---------------------------------------------------------------------------
def bremsstrahlung(n20, Te_kev, z_eff, relativistic: bool = True):
    """Bremsstrahlung power density [MW/m^3]."""
    n = np.asarray(n20, dtype=float)
    Te = np.maximum(np.asarray(Te_kev, dtype=float), 1e-3)
    p = 5.35e-3 * z_eff * n * n * np.sqrt(Te)
    if relativistic:
        # first-order relativistic correction; ~+10% at 25 keV
        p = p * (1.0 + 2.0 * Te / 511.0)
    return p


#: Level anchor for the synchrotron scaling, set so that the ITER Q=10 point
#: reproduces the ~20 MW that Albajar et al. (2001) compute with full
#: radiation transport at a wall reflectivity of 0.7.
SYNC_COEFF = 2.50e-6


def synchrotron(n20, Te_kev, B_t, a_minor, reflectivity: float = 0.7):
    """Synchrotron power density [MW/m^3].

    Trubnikov's optically-thick result scales as n^1/2 T^5/2 B^5/2 a^-1/2,
    reduced by sqrt(1 - r) for wall reflection.  The coefficient is an
    anchor, not a first-principles number -- see ``SYNC_COEFF``.
    """
    n = np.maximum(np.asarray(n20, dtype=float), 1e-6)
    Te = np.maximum(np.asarray(Te_kev, dtype=float), 1e-3)
    return (SYNC_COEFF * np.sqrt(n) * Te ** 2.5 * B_t ** 2.5
            * np.sqrt(max(1.0 - reflectivity, 1e-3) / a_minor))


def line_radiation(n20, Te_kev, comp: Composition):
    """Impurity line radiation [MW/m^3]."""
    from .constants import MW
    n = np.asarray(n20, dtype=float) * 1e20
    total = np.zeros_like(np.asarray(Te_kev, dtype=float))
    for name, frac in comp.fractions.items():
        total = total + n * (n * frac) * SPECIES[name].cooling_rate(Te_kev)
    return total / MW


def total_radiation(n20, Te_kev, B_t, a_minor, comp: Composition,
                    f_he: float = 0.0, reflectivity: float = 0.7) -> dict:
    """All three channels plus the composition they were computed with."""
    r = comp.resolve(f_he)
    brem = bremsstrahlung(n20, Te_kev, r["z_eff"])
    sync = synchrotron(n20, Te_kev, B_t, a_minor, reflectivity)
    line = line_radiation(n20, Te_kev, comp)
    return {"brem": brem, "sync": sync, "line": line,
            "total": brem + sync + line, **r}
