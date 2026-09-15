"""MHD stability: limits, growth rates, and their feedback on transport.

Three things a stability module has to do in an integrated code, in
increasing order of how much they matter:

1. **Limits** -- where the operating point has to stop.  Troyon, the kink
   limit on q95, the no-wall and ideal-wall beta limits.  Cheap, and every
   systems code has them.
2. **Growth** -- how fast an instability develops once it is unstable.  The
   modified Rutherford equation for a neoclassical tearing mode; the
   internal kink for a sawtooth.
3. **Feedback** -- what the instability does to the solution.  A magnetic
   island flattens the temperature across its width and costs confinement;
   a sawtooth crash reconnects the core and flattens everything inside the
   mixing radius; an ELM empties part of the pedestal.

Only the third makes this a stability *module* rather than a set of
warnings printed next to the answer, and it is what the solver consumes.

What is here
------------
* Ideal: Troyon / no-wall / ideal-wall beta limits, kink margin on q95,
  the s-alpha ballooning boundary, vertical stability.
* Resistive: neoclassical tearing modes through the modified Rutherford
  equation, with the bootstrap drive, the small-island (GGJ + polarisation)
  stabilisation that creates the seed-island threshold, and ECCD.
* Sawteeth: Kadomtsev reconnection -- the mixing radius follows from
  helical flux conservation, not from a fitted number.
* ELMs: frequency from the power balance that refills the pedestal.

What is NOT here
----------------
Mercier and the resistive interchange are deliberately absent rather than
guessed: their reduced forms differ between sources by factors that matter,
and a criterion written down wrong is worse than one left out.  Nor is this
a stability *code* -- there is no eigenvalue solve, so a genuinely 2-D mode
structure (external kinks against a shaped wall, the full peeling-ballooning
spectrum) is outside what these reduced criteria can see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import MU0, P_UNIT


# ---------------------------------------------------------------------------
# Ideal limits
# ---------------------------------------------------------------------------
def magnetic_shear(rho: np.ndarray, q: np.ndarray) -> np.ndarray:
    """s = (r/q) dq/dr on the given radial grid."""
    r = np.maximum(np.asarray(rho, dtype=float), 1e-6)
    qq = np.maximum(np.asarray(q, dtype=float), 1e-6)
    return r / qq * np.gradient(qq, r)


def ballooning_alpha(rho: np.ndarray, p: np.ndarray, q: np.ndarray,
                     R0: float, a: float, B0: float) -> np.ndarray:
    """alpha = -(2 mu0 R q^2 / B^2) dp/dr, the ballooning drive."""
    dpdr = np.gradient(np.asarray(p, dtype=float), np.asarray(rho)) / a
    return -(2.0 * MU0 * R0 * np.asarray(q) ** 2 / B0 ** 2) * dpdr


def ballooning_alpha_crit(s: np.ndarray, kappa: float = 1.0,
                          delta: float = 0.0) -> np.ndarray:
    """First-stability boundary in the s-alpha plane.

    For a circular, large-aspect-ratio plasma the marginal boundary is
    close to ``alpha_crit = 0.8 s``.  Elongation and triangularity raise it
    substantially, and above a certain shaping the first and second stable
    regions connect, so a strongly shaped pedestal can sit at alpha far
    above this line without being unstable -- which is exactly what ITER's
    pedestal does (see :mod:`tokamak.pedestal`).  A point above this
    boundary is therefore "not first-stable", not "unstable".
    """
    shape = 1.0 + 1.4 * (kappa - 1.0) + 3.0 * delta ** 2
    return 0.8 * np.maximum(np.asarray(s, dtype=float), 0.0) * shape


def troyon_limit(li: float, coefficient: float = 4.0) -> float:
    """No-wall beta_N limit.

    ``beta_N <= 4 li`` is the standard practical form; Troyon's original
    coefficient for the global kink limit with typical profiles is 2.8.  The
    li dependence is what makes a peaked current profile more stable.
    """
    return float(coefficient * max(li, 0.1))


def wall_stabilised_limit(no_wall: float, wall_gap_ratio: float = 1.3) -> float:
    """Ideal-wall limit: a conducting wall buys roughly 30-50% more beta,
    but only for as long as the wall's resistive time -- beyond the no-wall
    limit the mode reappears as a resistive wall mode."""
    return float(no_wall * wall_gap_ratio)


def rwm_margin(beta_n: float, no_wall: float, ideal_wall: float,
               rotation_stabilised: bool = True) -> float:
    """Where the operating point sits between the two beta limits.

    Returns ``C_beta = (beta_N - no_wall) / (ideal_wall - no_wall)``: below
    0 the plasma is ideally stable without a wall, above 1 no wall can help.
    Between them the resistive wall mode grows on the wall time unless
    rotation or feedback holds it.
    """
    span = max(ideal_wall - no_wall, 1e-6)
    c = (beta_n - no_wall) / span
    if rotation_stabilised:
        c -= 0.3          # rotation and feedback push the effective limit up
    return float(c)


def kink_margin(q95: float, q_crit: float = 2.0) -> float:
    """Margin against the external kink: (q95 - 2) / 2, negative is unstable."""
    return float((q95 - q_crit) / q_crit)


def vertical_margin(kappa: float, li: float, wall_gap: float = 0.15) -> float:
    """Stability margin against a vertical displacement event."""
    k_crit = 1.4 + 1.2 * li + 2.4 * wall_gap
    return float(np.clip((k_crit - kappa) / max(k_crit - 1.0, 1e-3), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Neoclassical tearing modes
# ---------------------------------------------------------------------------
@dataclass
class NTM:
    """One rational surface, evolving through the modified Rutherford equation.

        tau_R / r_s * dw/dt =
              r_s Delta'(w)                                  classical
            + a_bs beta_p sqrt(eps) (L_q/L_p) w/(w^2 + w_d^2)  bootstrap drive
            - a_gg beta_p sqrt(eps) (L_q/L_p) w_d^2/w^2 ... /  small-island
            - a_cd (j_cd/j_bs) ...                             ECCD

    The bootstrap term is the whole story: a seed island removes the
    pressure gradient across itself, the bootstrap current that gradient was
    driving disappears, and the missing current makes the island grow.  It
    is the ``w / (w^2 + w_d^2)`` shape that makes this a *threshold*
    problem -- below the marginal width ``w_d``, set by the competition
    between parallel and perpendicular heat transport across the island, the
    drive falls off faster than the classical stabilisation and the seed
    heals.  That is why NTMs need a trigger (a sawtooth, an ELM) and why
    they are avoided by avoiding the trigger, not only by lowering beta.
    """

    m: int
    n: int
    r_s: float                 # rational surface radius [m]
    rho_s: float               # ... normalised
    w: float = 0.0             # island width [m]

    # Coefficients of the modified Rutherford equation, written in the
    # dimensionless island width w_hat = w / r_s so the balance is legible:
    # the mode saturates near w_hat ~ C_bs / |Delta'_hat|, so with
    # |Delta'_hat| = 2 a saturated 2/1 island of 10% of r_s needs C_bs ~ 0.2.
    a_bs: float = 0.185        # bootstrap drive
    a_gg: float = 0.150        # small-island (GGJ + polarisation) stabilisation
    delta_prime_0: float = -2.0     # r_s Delta'(0), classically stable
    saturation: float = 0.8         # how fast Delta' falls with w_hat

    beta_p: float = 0.0
    eps: float = 0.3
    lq_over_lp: float = 2.0
    w_d: float = 0.02          # marginal island width [m]
    eccd_drive: float = 0.0    # j_cd / j_bs at the surface

    growth: float = 0.0        # dw/dt [m/s]
    tau_r: float = 100.0       # resistive time [s]

    @property
    def key(self) -> str:
        return f"{self.m}/{self.n}"

    @property
    def c_bs(self) -> float:
        return self.a_bs * self.beta_p * np.sqrt(max(self.eps, 1e-3)) \
            * self.lq_over_lp

    @property
    def c_gg(self) -> float:
        return self.a_gg * self.beta_p * np.sqrt(max(self.eps, 1e-3)) \
            * self.lq_over_lp

    def dwdt(self, w: Optional[float] = None) -> float:
        """Island growth rate [m/s].

            tau_R d(w_hat)/dt = Delta'_hat(w_hat)
                              + C_bs w_hat / (w_hat^2 + w_d_hat^2)
                              - C_gg w_d_hat^2 / (w_hat (w_hat^2 + w_d_hat^2))
                              - C_cd

        At w_hat = 0 the last term diverges, which is the model saying an
        infinitesimal island heals -- not that it collapses infinitely fast.
        Below w_d the growth rate is therefore not a physical rate, only a
        sign; ``seed_threshold`` is the quantity that means something there.
        """
        w = self.w if w is None else w
        wh = max(w, 1e-6) / max(self.r_s, 1e-6)
        wd = max(self.w_d, 1e-6) / max(self.r_s, 1e-6)
        denom = wh * wh + wd * wd
        classical = self.delta_prime_0 - self.saturation * wh
        drive = self.c_bs * wh / denom
        small = self.c_gg * wd * wd / (wh * denom)
        eccd = self.eccd_drive * 4.0 * wd / max(wh, wd)
        return float(self.r_s / max(self.tau_r, 1e-6)
                     * (classical + drive - small - eccd))

    def step(self, dt: float) -> None:
        """Advance the island, but only once it has been seeded."""
        if self.w <= 0.0:
            self.growth = 0.0
            return
        self.growth = self.dwdt()
        self.w = float(np.clip(self.w + self.growth * dt, 0.0, 0.5 * self.r_s))

    def seed(self, width: float) -> None:
        """Deposit a seed island, e.g. from a sawtooth crash or an ELM."""
        self.w = max(self.w, float(width))

    def _roots(self) -> List[float]:
        """Where dw/dt changes sign, scanning outward from w_d / 20.

        The growth rate is NOT monotonic in w: it is negative for a tiny
        island (the small-island terms win), positive in between (the
        bootstrap drive wins), and negative again once Delta' has saturated.
        Bisecting between the ends therefore finds nothing.  The inner root
        is the seed threshold, the outer one the saturated width.
        """
        w = np.geomspace(self.w_d / 20.0, 0.5 * self.r_s, 240)
        f = np.array([self.dwdt(x) for x in w])
        out: List[float] = []
        for i in np.where(np.diff(np.sign(f)) != 0)[0]:
            lo, hi = w[i], w[i + 1]
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if np.sign(self.dwdt(lo)) == np.sign(self.dwdt(mid)):
                    lo = mid
                else:
                    hi = mid
            out.append(0.5 * (lo + hi))
        return out

    def saturated_width(self) -> float:
        """Outermost zero of dw/dt: where a grown island stops growing."""
        r = self._roots()
        if not r:
            # no sign change: either stable everywhere or unstable everywhere
            return 0.5 * self.r_s if self.dwdt(0.25 * self.r_s) > 0 else 0.0
        return float(r[-1]) if self.dwdt(0.5 * (r[-1] + 0.5 * self.r_s)) < 0 \
            else 0.5 * self.r_s

    def seed_threshold(self) -> float:
        """Innermost zero of dw/dt: the smallest island that grows."""
        r = self._roots()
        if len(r) < 2:
            return float("inf") if self.saturated_width() <= 0 else 0.0
        return float(r[0])

    def confinement_penalty(self, a: float) -> float:
        """Fractional loss of stored energy from a saturated island.

        The island flattens the temperature across its width, so the plasma
        loses the pressure the removed gradient was holding.  The usual
        "belt" estimate makes that loss go as the island width times the
        radius at which it sits.
        """
        if self.w <= 0:
            return 0.0
        return float(np.clip(2.0 * self.w / a * self.rho_s, 0.0, 0.6))

    def to_dict(self) -> dict:
        return {"mode": self.key, "r_s": self.r_s, "rho_s": self.rho_s,
                "w": self.w, "w_sat": self.saturated_width(),
                "w_seed": self.seed_threshold(), "w_d": self.w_d,
                "dwdt": self.growth, "beta_p": self.beta_p,
                "eccd_drive": self.eccd_drive}


def find_rational_surface(rho: np.ndarray, q: np.ndarray,
                          m: int, n: int) -> Optional[float]:
    """Normalised radius where q = m/n, or None if the surface is absent."""
    target = m / n
    qq = np.asarray(q, dtype=float)
    rr = np.asarray(rho, dtype=float)
    sign = np.sign(qq - target)
    cross = np.where(np.diff(sign) != 0)[0]
    if len(cross) == 0:
        return None
    i = int(cross[-1])            # outermost crossing
    q0, q1 = qq[i], qq[i + 1]
    if abs(q1 - q0) < 1e-9:
        return float(rr[i])
    f = (target - q0) / (q1 - q0)
    return float(rr[i] + f * (rr[i + 1] - rr[i]))


def marginal_island_width(r_s: float, L_q: float,
                          chi_ratio: float = 1e-8) -> float:
    """Marginal island width w_d [m].

    Below it the parallel transport can no longer flatten the temperature
    across the island, the bootstrap drive collapses, and the seed heals.
    ``w_d ~ 1.8 (chi_perp/chi_par)^{1/4} sqrt(r_s L_q)`` -- for ITER that is
    a couple of centimetres, which is why a sawtooth or an ELM is needed to
    seed the mode at all.
    """
    return float(1.8 * chi_ratio ** 0.25 * np.sqrt(max(r_s * L_q, 1e-6)))


# ---------------------------------------------------------------------------
# Sawteeth: Kadomtsev reconnection
# ---------------------------------------------------------------------------
@dataclass
class Sawtooth:
    r_q1: float                # q = 1 surface [normalised radius]
    r_mix: float               # mixing radius after reconnection
    unstable: bool
    shear_q1: float
    period_estimate: float     # [s]

    def to_dict(self) -> dict:
        return {k: (float(v) if not isinstance(v, bool) else v)
                for k, v in self.__dict__.items()}


def kadomtsev_mixing_radius(rho: np.ndarray, q: np.ndarray) -> Tuple[float, float]:
    """q = 1 radius and the Kadomtsev mixing radius.

    A full sawtooth crash reconnects the helical flux inside q = 1.  Helical
    flux conservation requires

        int_0^{r_mix} (1/q(r) - 1) r dr = 0,

    which fixes the mixing radius: everything inside it is flattened and q
    is reset to 1 there.  No fitted number enters -- the extent of the crash
    follows from the q profile the equilibrium produced.
    """
    rr = np.asarray(rho, dtype=float)
    qq = np.asarray(q, dtype=float)
    # A sawtoothing discharge sits with q0 a few percent below 1 between
    # crashes, and rises back through 1 at each crash.  An equilibrium
    # solved with q0 clamped AT 1 -- which is what a reduced code does to
    # represent exactly this cycle -- has a degenerate q = 1 surface, so
    # taking q0 > 1 literally would report "no sawteeth" for the one case
    # that certainly has them.  Anything within a couple of percent of 1 is
    # therefore treated as the marginal state and evaluated at the q0 = 0.95
    # a real discharge sits at between crashes.
    if qq[0] > 1.02:
        return 0.0, 0.0
    if qq[0] > 0.95:
        qq = qq - (qq[0] - 0.95)
    r1 = find_rational_surface(rr, qq, 1, 1)
    if r1 is None:
        return 0.0, 0.0
    integrand = (1.0 / np.maximum(qq, 1e-6) - 1.0) * rr
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(rr))])
    outside = np.where((rr > r1) & (cum <= 0.0))[0]
    if len(outside) == 0:
        return float(r1), float(min(1.4 * r1, rr[-1]))
    i = int(outside[0])
    return float(r1), float(rr[i])


#: Fraction of the local resistive time that a sawtooth cycle takes.
C_SAWTOOTH = 0.30


def sawtooth_state(rho: np.ndarray, q: np.ndarray, tau_e: float,
                   te_avg: float, a: float = 2.0,
                   eta_q1: Optional[float] = None,
                   fast_ion_fraction: float = 0.0) -> Sawtooth:
    """Whether the internal kink is unstable, and how often it should crash.

    The period is set by current diffusion, not by the confinement time: a
    crash flattens the current inside q = 1, and the next one cannot happen
    until resistive diffusion has re-steepened the shear there.  So

        tau_saw ~ C mu0 r_1^2 / eta(r_1) * (1 + c_f f_fast)

    which for ITER is tens of seconds, not the ~3 s a confinement-time
    estimate gives.  The fast-ion term matters: energetic alphas stabilise
    the internal kink, so a burning plasma holds off the crash for longer
    and then releases more when it goes -- the "monster sawtooth" that is a
    concern for ITER precisely because a large crash is an excellent seed
    for a neoclassical tearing mode.
    """
    r1, r_mix = kadomtsev_mixing_radius(rho, q)
    if r1 <= 0:
        return Sawtooth(0.0, 0.0, False, 0.0, float("inf"))
    sh = magnetic_shear(rho, q)
    s1 = float(np.interp(r1, rho, sh))
    unstable = s1 > 0.05 and te_avg > 1.0
    if eta_q1 is None or eta_q1 <= 0:
        period = 0.35 + 1.9 * min(tau_e, 1.6)
    else:
        tau_res = MU0 * (r1 * a) ** 2 / eta_q1
        period = C_SAWTOOTH * tau_res * (1.0 + 3.0 * fast_ion_fraction)
    return Sawtooth(r1, r_mix, unstable, s1, float(np.clip(period, 0.05, 400.0)))


def apply_sawtooth_crash(rho: np.ndarray, prof: np.ndarray,
                         r_mix: float, keep: float = 1.0) -> np.ndarray:
    """Flatten a profile inside the mixing radius, conserving its content.

    The crash redistributes; it does not by itself expel energy.  The
    flattened value is the volume-weighted mean of what was there, so the
    integral over the mixed region is preserved to within ``keep``.
    """
    if r_mix <= 0:
        return prof
    rr = np.asarray(rho, dtype=float)
    inside = rr <= r_mix
    if inside.sum() < 2:
        return prof
    w = 2.0 * rr[inside]
    mean = float(np.trapezoid(prof[inside] * w, rr[inside])
                 / max(np.trapezoid(w, rr[inside]), 1e-12))
    out = prof.copy()
    out[inside] = mean * keep
    return out


# ---------------------------------------------------------------------------
# ELMs
# ---------------------------------------------------------------------------
@dataclass
class ELMState:
    frequency: float          # [Hz]
    energy_loss: float        # [MJ] per ELM
    loss_fraction: float      # of the total stored energy
    power_to_target: float    # [MW] time-averaged
    mitigated: bool

    def to_dict(self) -> dict:
        return {k: (float(v) if not isinstance(v, bool) else v)
                for k, v in self.__dict__.items()}


def elm_state(W_th: float, P_sep: float, ped_width_psi: float,
              mitigated: bool = True, f_ped: float = 0.30) -> ELMState:
    """ELM frequency from the power balance that refills the pedestal.

    An ELM expels a fixed fraction of the stored energy; the next one cannot
    happen until the power crossing the separatrix has put it back.  So the
    frequency is not a free parameter once the loss per ELM is set::

        dW_ELM = f_loss W_th        f_ELM = f_ped P_sep / dW_ELM

    Type-I ELMs lose 5-8% of the stored energy.  For ITER that is ~20 MJ at
    ~1 Hz, which no divertor material survives -- which is why ELM control
    is not optional.  Mitigation (pellet pacing, resonant magnetic
    perturbations) trades size for frequency at constant time-averaged
    power: the same exhaust arrives in many small bites instead of few
    large ones.
    """
    frac = 0.010 if mitigated else 0.060
    dW = max(W_th * frac, 1e-3)
    f = max(f_ped * P_sep, 1e-3) / dW
    return ELMState(frequency=float(f), energy_loss=float(dW),
                    loss_fraction=float(frac),
                    power_to_target=float(f * dW), mitigated=mitigated)


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------
@dataclass
class StabilityReport:
    beta_n: float
    beta_n_no_wall: float
    beta_n_ideal_wall: float
    troyon_fraction: float
    c_beta: float                       # RWM position between the two limits
    kink_margin: float
    vertical_margin: float
    shear: np.ndarray = field(default_factory=lambda: np.zeros(0))
    alpha: np.ndarray = field(default_factory=lambda: np.zeros(0))
    alpha_crit: np.ndarray = field(default_factory=lambda: np.zeros(0))
    first_stable_fraction: float = 1.0
    ntms: List[NTM] = field(default_factory=list)
    sawtooth: Optional[Sawtooth] = None
    elms: Optional[ELMState] = None
    limiting_mode: str = "тұрақты"
    margin: float = 1.0
    confinement_penalty: float = 0.0

    def to_dict(self) -> dict:
        return {
            "beta_n": self.beta_n,
            "beta_n_no_wall": self.beta_n_no_wall,
            "beta_n_ideal_wall": self.beta_n_ideal_wall,
            "troyon_fraction": self.troyon_fraction,
            "c_beta": self.c_beta,
            "kink_margin": self.kink_margin,
            "vertical_margin": self.vertical_margin,
            "first_stable_fraction": self.first_stable_fraction,
            "ntms": [n.to_dict() for n in self.ntms],
            "sawtooth": self.sawtooth.to_dict() if self.sawtooth else None,
            "elms": self.elms.to_dict() if self.elms else None,
            "limiting_mode": self.limiting_mode,
            "margin": self.margin,
            "confinement_penalty": self.confinement_penalty,
        }


#: Rational surfaces worth tracking, most dangerous first.  The 2/1 is the
#: one that locks and disrupts; the 3/2 costs confinement but is survivable.
NTM_MODES = ((2, 1), (3, 2))


def analyse(*, rho: np.ndarray, q: np.ndarray, p: np.ndarray,
            beta_n: float, beta_p: float, li: float, q95: float,
            R0: float, a: float, B0: float, kappa: float, delta: float,
            tau_e: float, te_avg: float, W_th: float, P_sep: float,
            ped_width_psi: float = 0.04,
            eta_q1: Optional[float] = None, fast_ion_fraction: float = 0.0,
            existing: Optional[Dict[str, NTM]] = None,
            eccd: Optional[Dict[str, float]] = None,
            mitigated_elms: bool = True) -> StabilityReport:
    """Full stability assessment of one equilibrium + profile set."""
    rho = np.asarray(rho, dtype=float)
    q = np.asarray(q, dtype=float)
    s = magnetic_shear(rho, q)
    alpha = ballooning_alpha(rho, p, q, R0, a, B0)
    a_crit = ballooning_alpha_crit(s, kappa, delta)
    # Skip the axis: there both alpha and the shear go to zero, so the
    # comparison is between two vanishing numbers and says nothing.  Weight
    # the rest by volume, since dV goes as rho.
    band = rho > 0.15
    wgt = rho[band]
    first_stable = float(np.sum(wgt * (alpha[band] <= a_crit[band]))
                         / max(np.sum(wgt), 1e-9))

    no_wall = troyon_limit(li)
    ideal_wall = wall_stabilised_limit(no_wall)
    rep = StabilityReport(
        beta_n=beta_n, beta_n_no_wall=no_wall, beta_n_ideal_wall=ideal_wall,
        troyon_fraction=beta_n / max(no_wall, 1e-6),
        c_beta=rwm_margin(beta_n, no_wall, ideal_wall),
        kink_margin=kink_margin(q95),
        vertical_margin=vertical_margin(kappa, li),
        shear=s, alpha=alpha, alpha_crit=a_crit,
        first_stable_fraction=first_stable,
    )

    # --- tearing modes -----------------------------------------------------
    eccd = eccd or {}
    penalty = 0.0
    for m, n in NTM_MODES:
        rho_s = find_rational_surface(rho, q, m, n)
        if rho_s is None or rho_s <= 0.05:
            continue
        r_s = rho_s * a
        s_loc = max(float(np.interp(rho_s, rho, s)), 0.1)
        L_q = r_s / s_loc
        p_loc = float(np.interp(rho_s, rho, p))
        dpdr = float(np.interp(rho_s, rho, np.gradient(p, rho))) / a
        L_p = abs(p_loc / dpdr) if abs(dpdr) > 1e-9 else 10.0
        key = f"{m}/{n}"
        prev = (existing or {}).get(key)
        mode = NTM(m=m, n=n, r_s=r_s, rho_s=rho_s,
                   w=prev.w if prev else 0.0,
                   beta_p=beta_p, eps=rho_s * a / R0,
                   lq_over_lp=float(np.clip(L_q / max(L_p, 1e-3), 0.2, 10.0)),
                   w_d=marginal_island_width(r_s, L_q),
                   eccd_drive=eccd.get(key, 0.0),
                   tau_r=max(50.0 * tau_e, 1.0))
        mode.growth = mode.dwdt()
        rep.ntms.append(mode)
        penalty += mode.confinement_penalty(a)

    rep.confinement_penalty = float(np.clip(penalty, 0.0, 0.6))
    rep.sawtooth = sawtooth_state(rho, q, tau_e, te_avg, a=a,
                                  eta_q1=eta_q1,
                                  fast_ion_fraction=fast_ion_fraction)
    rep.elms = elm_state(W_th, P_sep, ped_width_psi, mitigated_elms)

    # --- what is closest to biting ----------------------------------------
    margins = {
        "кink (q₉₅)": rep.kink_margin,
        "тік тұрақсыздық (VDE)": rep.vertical_margin,
        "идеал бета (no-wall)": 1.0 - rep.troyon_fraction,
    }
    for mode in rep.ntms:
        if mode.saturated_width() > mode.w_d:
            margins[f"NTM {mode.key}"] = -mode.saturated_width() / max(a, 1e-3)
    worst = min(margins.items(), key=lambda kv: kv[1])
    rep.limiting_mode = worst[0] if worst[1] < 0.35 else "тұрақты"
    rep.margin = float(worst[1])
    return rep
