"""Disruption physics: quench timescales, runaway electrons, halo currents.

A 0-D model can say *that* a limit was crossed.  What makes a disruption
the thing that constrains a reactor is what happens afterwards, and that is
three separate problems:

* **Thermal quench** -- the stored energy lands on the wall in ~1 ms.
* **Current quench** -- L/R decay at the post-quench temperature, which
  fixes the induced loop voltage and hence everything below.
* **Runaway electrons** -- that loop voltage exceeds the critical field by
  orders of magnitude, and the avalanche gain goes as exp(2.5 Ip[MA]).  For
  ITER that is e^37, which is why runaways, not the thermal load, are the
  disruption problem in a reactor.
* **Halo currents** -- current that flows through the wall during a
  vertical displacement, and the forces it produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (E_CHARGE, EPS0, M_E, C_LIGHT, MU0, LN_LAMBDA)


@dataclass
class DisruptionResult:
    cause: str
    W_thermal: float          # [MJ] dumped in the thermal quench
    W_magnetic: float         # [MJ] poloidal field energy to dissipate
    tau_tq_ms: float
    tau_cq_ms: float
    Te_post_eV: float
    E_critical: float         # [V/m] runaway threshold
    E_induced: float          # [V/m] during the current quench
    avalanche_gain_exp: float  # natural-log gain of the avalanche
    I_runaway: float          # [MA]
    halo_fraction: float
    tpf: float                # toroidal peaking factor
    vertical_force_MN: float
    q_tq_MJm2: float          # thermal quench load on the first wall
    mitigated: bool

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in self.__dict__.items()}


def critical_field(ne20: float) -> float:
    """Connor-Hastie critical electric field for runaway generation [V/m].

    E_c = n_e e^3 lnLambda / (4 pi eps0^2 m_e c^2)
    """
    n = max(ne20, 1e-4) * 1e20
    return (n * E_CHARGE ** 3 * LN_LAMBDA
            / (4.0 * np.pi * EPS0 ** 2 * M_E * C_LIGHT ** 2))


def avalanche_growth_rate(E: float, E_c: float, z_eff: float,
                          ne20: float) -> float:
    """Rosenbluth-Putvinski secondary (avalanche) growth rate [1/s].

    gamma = (E/E_c - 1) / (tau_c lnLambda) sqrt(pi phi / (3 (Z + 5)))
    with tau_c the relativistic collision time.
    """
    if E <= E_c:
        return 0.0
    n = max(ne20, 1e-4) * 1e20
    tau_c = (4.0 * np.pi * EPS0 ** 2 * M_E ** 2 * C_LIGHT ** 3
             / (n * E_CHARGE ** 4 * LN_LAMBDA))
    phi = 1.0                       # no toroidicity correction at this level
    return ((E / E_c - 1.0) / (tau_c * LN_LAMBDA)
            * np.sqrt(np.pi * phi / (3.0 * (z_eff + 5.0))))


def current_quench_time(Ip: float, R0: float, a: float, kappa: float,
                        Te_post_eV: float, z_eff: float,
                        li: float = 0.9) -> float:
    """L/R current quench time [s] at the post-thermal-quench temperature.

    Uses the plasma's own self-inductance, mu0 R (ln(8R/a) + li/2 - 2),
    rather than a local estimate -- for ITER that is ~12 uH against the
    ~7 uH a minor-radius approximation gives, and the quench time is
    linear in it.
    """
    eta = 2.8e-8 * z_eff / max(Te_post_eV / 1e3, 1e-3) ** 1.5
    area = np.pi * a * a * kappa
    eps = a / R0
    L = MU0 * R0 * (np.log(8.0 / eps) + li / 2.0 - 2.0)
    R = eta * 2.0 * np.pi * R0 / area
    return float(np.clip(L / max(R, 1e-12), 1e-3, 2.0))


def simulate(*, cause: str, Ip: float, R0: float, a: float, kappa: float,
             B0: float, W_thermal: float, li: float, ne20: float,
             z_eff: float, S_wall: float, mitigated: bool = False,
             seed_runaway_A: float = 1.0) -> DisruptionResult:
    """Run the post-disruption chain for one event.

    ``mitigated`` stands for massive gas or shattered-pellet injection: it
    raises the density by two orders of magnitude, which both radiates the
    thermal energy isotropically and lifts the critical field above the
    induced one -- the reason a mitigation system exists.
    """
    eps = a / R0
    # shattered-pellet injection delivers ~1e24 atoms, two to three orders
    # of magnitude above the plasma inventory
    ne_eff = ne20 * (200.0 if mitigated else 1.0)
    Te_post = 15.0 if mitigated else 5.0      # eV

    # --- magnetic energy stored in the poloidal field ---------------------
    L_p = MU0 * R0 * (np.log(8.0 / eps) + li / 2.0 - 2.0)
    W_mag = 0.5 * L_p * (Ip * 1e6) ** 2 / 1e6                 # [MJ]

    tau_tq = 1.0e-3 if not mitigated else 3.0e-3              # [s]
    tau_cq = current_quench_time(Ip, R0, a, kappa, Te_post, z_eff, li)

    # --- induced field during the current quench --------------------------
    v_loop = L_p * (Ip * 1e6) / tau_cq                        # [V]
    E_ind = v_loop / (2.0 * np.pi * R0)                       # [V/m]
    E_c = critical_field(ne_eff)

    gamma = avalanche_growth_rate(E_ind, E_c, z_eff, ne_eff)
    gain_exp = float(np.clip(gamma * tau_cq, 0.0, 80.0))

    # runaway current: the seed is amplified, but cannot exceed the pre-
    # disruption current -- the beam carries what the plasma was carrying
    I_re = min(seed_runaway_A * np.exp(gain_exp) / 1e6, Ip * 0.85)
    if E_ind <= E_c:
        I_re = 0.0

    # --- halo current and vertical force ----------------------------------
    # ITER design basis: halo fraction up to 0.4 with a toroidal peaking
    # factor of 2, and the product f * TPF is what the structure must hold.
    halo_f = 0.20 if mitigated else 0.35
    tpf = 1.6 if mitigated else 2.0
    F_vert = halo_f * Ip * 1e6 * B0 * 2.0 * np.pi * R0 * 0.15 / 1e6   # [MN]

    # --- thermal quench wall load -----------------------------------------
    # mitigation spreads the load over the whole wall; unmitigated it lands
    # on a fraction of it
    wetted = S_wall * (1.0 if mitigated else 0.25)
    q_tq = W_thermal / max(wetted, 1e-3)                      # [MJ/m^2]

    return DisruptionResult(
        cause=cause, W_thermal=W_thermal, W_magnetic=W_mag,
        tau_tq_ms=tau_tq * 1e3, tau_cq_ms=tau_cq * 1e3, Te_post_eV=Te_post,
        E_critical=E_c, E_induced=E_ind, avalanche_gain_exp=gain_exp,
        I_runaway=I_re, halo_fraction=halo_f, tpf=tpf,
        vertical_force_MN=F_vert, q_tq_MJm2=q_tq, mitigated=mitigated,
    )


# ---------------------------------------------------------------------------
def check_limits(*, f_greenwald: float, q95: float, beta_n: float,
                 p_rad: float, p_heat: float, Te_avg: float,
                 li: float = 0.9) -> str | None:
    """Return the name of the first operational limit that is violated."""
    if f_greenwald > 1.25:
        return "ТЫҒЫЗДЫҚ ШЕГІ — ГРИНВАЛЬД АСЫП КЕТТІ"
    if q95 < 2.0:
        return "q95 ТӨМЕН — КИНК ТҰРАҚСЫЗДЫҒЫ"
    if beta_n > 4.0 * li:
        return "БЕТА ШЕГІ — ИДЕАЛ МГД (Troyon)"
    if p_heat > 1.0 and p_rad > 1.35 * p_heat and Te_avg > 1.0:
        return "СӘУЛЕЛЕНУ КОЛЛАПСЫ"
    return None


def vertical_stability_margin(kappa: float, li: float,
                              wall_gap: float = 0.15) -> float:
    """Stability margin m_s against a vertical displacement event.

    An elongated plasma is vertically unstable; the passive structure and
    the control coils hold it.  The margin falls with elongation and rises
    with the internal inductance -- which is why a high-kappa, low-li
    plasma is the one that goes vertical.
    """
    k_crit = 1.4 + 1.2 * li + 2.4 * wall_gap
    return float(np.clip((k_crit - kappa) / max(k_crit - 1.0, 1e-3), -1.0, 1.0))
