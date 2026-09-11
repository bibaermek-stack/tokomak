"""Integrated 1.5-D burn simulator.

Couples, on every step:

    composition -> radiation -> fusion -> fast ions -> equipartition
        -> 1-D transport (T_e, T_i, n_e, n_He)
        -> current (bootstrap, drive, diffusion, volt-second budget)
        -> equilibrium (q profile, li, beta) at intervals
        -> SOL / divertor two-point model
        -> operational limits and, if crossed, the disruption chain

"1.5-D" in the usual sense: 1-D transport on flux surfaces whose geometry
comes from a 2-D Grad-Shafranov solve.

Everything selectable -- machine, confinement scaling, L-H threshold,
transport model -- is chosen in :class:`SolverConfig`, so a run states which
models produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

import numpy as np

from . import current as cur
from . import disruption as disr
from . import equilibrium as eqm
from . import fuelcycle as fc
from . import pedestal as ped
from . import radiation as rad
from . import sol as sol_mod
from . import transport as tr
from .constants import (E_ALPHA_KEV, E_FUSION_J, F_ALPHA, LN_LAMBDA, MU0,
                        MW, P_UNIT, W_UNIT)
from .geometry import attach_geometry, b_poloidal
from .machines import Machine, get_machine
from .reactivity import dt_power_density
from .scalings import (ConfinementInputs, ThresholdInputs, greenwald_density,
                       p_lh_threshold, tau_energy)


# ---------------------------------------------------------------------------
@dataclass
class SolverConfig:
    machine: str = "iter"
    confinement_model: str = "ipb98y2"
    lh_model: str = "martin08"
    transport_model: str = "scaling_anchored"
    #: "eped"  -- pedestal height from peeling-ballooning physics, so the
    #:            global confinement is PREDICTED and H98 is an output
    #: "anchor" -- pedestal driven to whatever reproduces the chosen 0-D
    #:            scaling, so H98 is an input by construction
    pedestal_model: str = "eped"
    n_rho: int = 65
    dt: float = 0.02
    equilibrium: bool = True
    eq_interval: float = 4.0          # plasma seconds between GS solves
    eq_grid: tuple = (73, 109)
    eq_relax: float = 3.0             # s to glide onto a new equilibrium
    chi_relax: float = 0.5            # s to relax the stiff diffusivity
    h_factor: float = 1.0             # multiplier on the chosen scaling
    reflectivity: float = 0.7
    tau_he_ratio: float = 5.85        # tau_He* / tau_E
    tau_p_ratio: float = 2.0          # particle / energy confinement
    seeding: str = "Ne"
    seeding_fraction: Optional[float] = None
    disruption_enabled: bool = True
    mitigation: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["eq_grid"] = list(self.eq_grid)
        return d


@dataclass
class Actuators:
    p_nbi: float = 0.0
    p_icrf: float = 0.0
    p_ecrf: float = 0.0
    gas: float = 0.0            # 0..1 valve position
    Ip_request: float = 0.0     # [MA]
    v_surface: float = 0.0      # applied loop voltage [V]


# ---------------------------------------------------------------------------
def _slowing_integral(x: float) -> float:
    """I(x) = int_0^x du / (1 + u^{3/2}); I(inf) = 4 pi / (3 sqrt 3)."""
    I_INF = 4.0 * np.pi / (3.0 * np.sqrt(3.0))
    if x <= 0:
        return 0.0
    if x > 4.0:
        return I_INF - 2.0 / np.sqrt(x)
    u = np.linspace(0.0, x, 65)
    return float(np.trapezoid(1.0 / (1.0 + u ** 1.5), u))


def _slowing_G(x: float) -> float:
    """Stored energy per unit birth power, in units of tau_s."""
    return 0.0 if x <= 0 else (x - _slowing_integral(x)) / (2.0 * x)


def _ion_fraction(x: float) -> float:
    """Fraction of a fast population's energy delivered to the bulk ions."""
    return 1.0 if x <= 0 else _slowing_integral(x) / x


# ---------------------------------------------------------------------------
class Simulator:
    """Time-dependent integrated simulation of one discharge."""

    def __init__(self, config: Optional[SolverConfig] = None,
                 machine: Optional[Machine] = None):
        self.cfg = config or SolverConfig()
        self.m = attach_geometry(machine or get_machine(self.cfg.machine))
        self.grid = tr.Grid(n=self.cfg.n_rho, a=self.m.a,
                            V_total=self.m.V, kappa_a=self.m.kappa_a)
        self.comp = rad.Composition()
        if self.cfg.seeding_fraction is not None:
            self.comp = self.comp.with_seeding(self.cfg.seeding,
                                               self.cfg.seeding_fraction)
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        g = self.grid
        self.t = 0.0
        self.state = tr.initial_state(g, ne0=0.10, Te0=0.30, Ti0=0.25)
        self.Ip = 0.2
        self.act = Actuators(Ip_request=0.2)
        self.h_mode = False
        self.disrupted = False
        self.disruption: Optional[disr.DisruptionResult] = None
        self.chi0 = 1.0
        self.q_prof = np.full(g.n, 3.0)
        self.q95 = eqm.analytic_q95(self.m.R0, self.m.a, self.m.kappa_95,
                                    self.m.delta_95, self.m.B0, self.m.Ip)
        self.q0 = 2.0
        self.li = 0.9
        self.eq: Optional[eqm.Equilibrium] = None
        self._t_last_eq = -1e9
        self._eq_target = None
        self._eq_seen = False
        self._chi_e = None
        self._chi_i = None
        self.flux = cur.FluxBudget(
            available=cur.cs_flux_capacity(self.m.R0, self.m.a, self.m.Ip) * 1.35)
        self.psi_prof = np.zeros(g.n)
        self.n_sep = 0.02
        self.n_he_sep = 0.0
        self.T_ped = 0.20
        self.ped = None
        self.f_bs = 0.0
        self.I_bs = 0.0
        self.I_cd = 0.0
        self.log: List[dict] = []
        self.alarms: List[str] = []
        self._derive()

    # -- the physics chain -------------------------------------------------
    def _derive(self) -> None:
        """Everything that follows algebraically from the current profiles."""
        g, m, s = self.grid, self.m, self.state
        cfg = self.cfg

        # --- composition ---------------------------------------------------
        f_he = s.n_he_avg / max(s.ne_avg, 1e-6)
        c = self.comp.resolve(f_he, m.a_mass)
        self.c = c

        # --- fusion --------------------------------------------------------
        n_dt = s.ne * c["f_dt"]
        self.p_fus_prof = dt_power_density(n_dt, s.Ti)              # MW/m^3
        self.P_fus = g.integrate(self.p_fus_prof)
        self.P_alpha = self.P_fus * F_ALPHA
        self.p_alpha_prof = self.p_fus_prof * F_ALPHA
        self.neutron_rate = self.P_fus * MW / E_FUSION_J

        # --- radiation -----------------------------------------------------
        r = rad.total_radiation(s.ne, s.Te, m.B0, m.a, self.comp, f_he,
                                cfg.reflectivity, m.a_mass)
        self.P_brem = g.integrate(r["brem"])
        self.P_sync = g.integrate(r["sync"])
        self.P_line = g.integrate(r["line"])
        self.p_rad_prof = r["total"]
        self.P_rad = self.P_brem + self.P_sync + self.P_line

        # --- fast ions ------------------------------------------------------
        self._fast_ions(c)

        # --- heating --------------------------------------------------------
        self.P_aux = self.act.p_nbi + self.act.p_icrf + self.act.p_ecrf
        self.P_heat = self.P_aux + self.P_alpha + getattr(self, "P_ohm", 0.0)
        self.P_loss = max(self.P_heat - self.P_rad, 0.5)
        self.P_sep = self.P_loss

        # --- global scalars --------------------------------------------------
        self.W_th = s.stored_energy(c["f_ion"])
        self.W = self.W_th + self.W_fast
        self.n_G = greenwald_density(self.Ip, m.a)
        self.f_G = s.ne_avg / max(self.n_G, 1e-6)
        p_avg = P_UNIT * g.volume_average(s.ne * (s.Te + c["f_ion"] * s.Ti))
        self.p_avg = p_avg
        p_tot = p_avg + (2.0 / 3.0) * self.W_fast * MW / max(m.V, 1e-6)
        self.beta_t = 2.0 * MU0 * p_tot / m.B0 ** 2
        self.beta_n = 100.0 * self.beta_t * m.a * m.B0 / max(self.Ip, 1e-6)
        B_pa = b_poloidal(max(self.Ip, 1e-6), m.L_pol)
        self.beta_p = 2.0 * MU0 * p_tot / B_pa ** 2

        self.tau_E = self.W_th / self.P_loss
        x = ConfinementInputs(Ip=self.Ip, B0=m.B0, P_loss=self.P_loss,
                              n_bar20=s.n_bar, R0=m.R0, a=m.a,
                              kappa_a=m.kappa_a, a_mass=m.a_mass)
        self.tau_scaling = tau_energy(cfg.confinement_model, x)
        self.H_factor = self.tau_E / max(self.tau_scaling, 1e-9)
        self.Q = self.P_fus / self.P_aux if self.P_aux > 0.3 else (
            999.0 if self.P_fus > 1 else 0.0)

        # --- L-H threshold ---------------------------------------------------
        th = ThresholdInputs(n_bar20=s.n_bar, B0=m.B0, S=m.S, R0=m.R0,
                             a=m.a, a_mass=m.a_mass)
        self.P_lh = p_lh_threshold(cfg.lh_model, th)

    def _fast_ions(self, c: dict) -> None:
        """Slowing-down stored energy and the electron/ion heating split."""
        g, s, m = self.grid, self.state, self.m
        z2a = max(c["z2a"], 0.05)
        lnL = LN_LAMBDA

        def tau_s(A, Z, Te, n20):
            return 0.19826 * A / (Z * Z * lnL) * np.maximum(Te, 0.1) ** 1.5 \
                / np.maximum(n20, 0.02)

        def e_crit(A, Te):
            return 14.8 * A * np.maximum(Te, 0.1) * z2a ** (2.0 / 3.0)

        # alphas: born in the hot core, where tau_s ~ T^3/2 is much longer
        src = self.p_alpha_prof * g.Vp
        w_tot = float(np.trapezoid(src, g.rho))
        if w_tot > 1e-9:
            ts = tau_s(4.0, 2.0, s.Te, s.ne)
            xs = E_ALPHA_KEV / e_crit(4.0, s.Te)
            Gs = np.array([_slowing_G(v) for v in xs])
            Fi = np.array([_ion_fraction(v) for v in xs])
            W_a = float(np.trapezoid(src * ts * Gs, g.rho))
            f_alpha_ion = float(np.trapezoid(src * Fi, g.rho)) / w_tot
        else:
            W_a, f_alpha_ion = 0.0, 0.15

        Te_a = max(s.Te_avg, 0.1)
        n_a = max(s.ne_avg, 0.02)
        x_nb = self.m.e_nbi_kev / e_crit(2.0, Te_a)
        x_ic = 500.0 / e_crit(1.0, Te_a)
        W_nb = self.act.p_nbi * tau_s(2.0, 1.0, Te_a, n_a) * _slowing_G(x_nb)
        W_ic = self.act.p_icrf * tau_s(1.0, 1.0, Te_a, n_a) * _slowing_G(x_ic)

        self.W_fast = float(W_a + W_nb + W_ic)
        self.split = {"alpha": f_alpha_ion,
                      "nbi": _ion_fraction(x_nb),
                      "icrf": _ion_fraction(x_ic)}

    # -- transport ---------------------------------------------------------
    def _chi(self) -> np.ndarray:
        cfg, g, s, m = self.cfg, self.grid, self.state, self.m
        if cfg.transport_model == "gyrobohm":
            return tr.chi_gyrobohm(s, m.B0, m.a_mass, self.h_mode)
        if cfg.transport_model == "bohm_gyrobohm":
            return tr.chi_bohm_gyrobohm(s, m.B0, m.a_mass, self.q_prof,
                                        self.h_mode)
        if cfg.transport_model == "cgm":
            return tr.chi_cgm(s, m.a, self.h_mode, m.R0)
        # scaling-anchored: stiff shape, amplitude set by the chosen scaling
        return self.chi0 * tr.chi_shape(g.rho, self.h_mode, s.Te, m.R0, m.a)

    def _update_pedestal(self, dt: float) -> None:
        """Set the pedestal-top temperature for this step.

        With ``pedestal_model = "eped"`` the height comes from the coupled
        KBM-width / ballooning-limit pair, so the stored energy -- and
        therefore H98 -- is an OUTPUT of the model.  That is the difference
        between a predictive code and one that reproduces a scaling because
        it was told to.

        With ``"anchor"`` the old behaviour is kept: the pedestal is driven
        to whatever makes the global confinement match the chosen 0-D
        scaling, which is useful for comparing against that scaling but
        cannot predict anything it was not given.
        """
        g = self.grid
        if not self.h_mode:
            target = 0.35 * max(self.state.Te_avg, 0.05)
            self.ped = None
        elif self.cfg.pedestal_model == "eped":
            i_ped = int(round(0.93 * (g.n - 1)))
            n_ped = float(self.state.ne[i_ped])
            self.ped = ped.solve(
                R0=self.m.R0, a=self.m.a, kappa_a=self.m.kappa_a,
                B0=self.m.B0, Ip=max(self.Ip, 0.05), q95=max(self.q95, 1.2),
                n_ped20=max(n_ped, 0.02), f_ion=self.c["f_ion"],
                L_pol=self.m.L_pol)
            target = self.ped.T_ped * self.cfg.h_factor
        else:
            tau_target = max(self.cfg.h_factor * self.tau_scaling, 1e-3)
            ratio = float(np.clip(tau_target * self.P_loss
                                  / max(self.W_th, 1e-3), 0.25, 4.0))
            target = self.T_ped * ratio ** 0.06
            self.ped = None
        # first-order response: the pedestal rebuilds on the energy
        # confinement time, not instantly
        tau_ped = max(0.3 * self.tau_E, 0.05)
        self.T_ped += (float(np.clip(target, 0.02, 30.0)) - self.T_ped) \
            * min(dt / tau_ped, 1.0)

    # -- one step ----------------------------------------------------------
    def step(self, dt: Optional[float] = None) -> None:
        dt = float(dt if dt is not None else self.cfg.dt)
        if dt <= 0:
            return
        g, m, s, cfg = self.grid, self.m, self.state, self.cfg
        c = self.c

        if self.disrupted:
            self._step_disrupted(dt)
            return

        # q95 from the analytic form whenever no equilibrium is available;
        # the Grad-Shafranov solve overwrites it when it runs
        if not cfg.equilibrium or self.eq is None:
            self.q95 = eqm.analytic_q95(m.R0, m.a, m.kappa_95, m.delta_95,
                                        m.B0, max(self.Ip, 0.05))
            self.q0 = max(1.0, self.q95 / (2.6 + 1.4 * self.li))
            self.q_prof = self.q0 + (self.q95 - self.q0) * g.rho ** 2

        # --- current ramp and non-inductive drive ---------------------------
        rate = 0.55 * m.Ip / 15.0 * 4.0
        self.Ip = float(np.clip(self.act.Ip_request,
                                self.Ip - rate * dt, self.Ip + rate * dt))
        self.Ip = max(self.Ip, 0.01)

        eta = cur.neoclassical_resistivity(g.rho, s.ne, s.Te, c["z_eff"],
                                           m.R0, m.eps, self.q_prof)
        bs = cur.bootstrap(g.rho, s.ne, s.Te, s.Ti, self.q_prof,
                           R0=m.R0, a=m.a, B0=m.B0, Ip=self.Ip,
                           z_eff=c["z_eff"], f_ion=c["f_ion"], A_cs=m.A_cs)
        self.f_bs = float(np.clip(bs.f_bs, 0.0, 0.95))
        self.I_bs = self.f_bs * self.Ip
        self.I_cd = (cur.nb_current_drive(self.act.p_nbi, m.e_nbi_kev,
                                          s.ne_avg, s.Te_avg, m.R0, c["z_eff"])
                     + cur.ec_current_drive(self.act.p_ecrf, s.ne_avg,
                                            s.Te_avg, m.R0, c["z_eff"]))
        I_ind = max(self.Ip - self.I_bs - self.I_cd, 0.0)

        eta_ax = float(np.interp(0.35, g.rho, eta))
        R_p = eta_ax * 2.0 * np.pi * m.R0 / max(m.A_cs, 1e-3)
        self.P_ohm = min(R_p * (I_ind * 1e6) ** 2 / MW, 80.0)
        self.V_loop = R_p * I_ind * 1e6
        self.flux.step(self.V_loop, dt)

        # --- L-H transition --------------------------------------------------
        p_heat = self.P_aux + self.P_alpha + self.P_ohm
        if not self.h_mode and p_heat > 1.05 * self.P_lh and \
                self.Ip > 0.4 * m.Ip and self.t > 4.0:
            self.h_mode = True
            self.alarms.append(f"t={self.t:.1f}s L-H АУЫСУЫ")
        elif self.h_mode and p_heat < 0.75 * self.P_lh:
            self.h_mode = False
            self.alarms.append(f"t={self.t:.1f}s H-L КЕРІ АУЫСУЫ")

        # --- sources ----------------------------------------------------------
        sp = self.split
        rho = g.rho
        dep_nb = tr.normalise_source(g, tr.deposition(rho, "broad"), self.act.p_nbi)
        dep_ic = tr.normalise_source(g, tr.deposition(rho, "core"), self.act.p_icrf)
        dep_ec = tr.normalise_source(g, tr.deposition(rho, "core"), self.act.p_ecrf)
        dep_oh = tr.normalise_source(g, tr.deposition(rho, "broad"), self.P_ohm)

        pei = tr.equipartition(s, c["z2a"])          # ions -> electrons

        s_e = (self.p_alpha_prof * (1.0 - sp["alpha"])
               + dep_nb * (1.0 - sp["nbi"]) + dep_ic * (1.0 - sp["icrf"])
               + dep_ec + dep_oh + pei - self.p_rad_prof)
        s_i = (self.p_alpha_prof * sp["alpha"]
               + dep_nb * sp["nbi"] + dep_ic * sp["icrf"] - pei)

        # --- transport step ----------------------------------------------------
        # Stiff transport evaluated on the previous step's gradient is an
        # explicit treatment of an implicit problem, and it produces a
        # relaxation oscillation: chi overshoots, the profile over-flattens,
        # chi collapses, the profile rebuilds.  Relaxing chi in time turns
        # that limit cycle into the steady state it is oscillating about.
        w = min(dt / max(cfg.chi_relax, 1e-3), 1.0)
        chi_new = self._chi()
        chi_i_new = 1.8 * self.chi0 * tr.chi_shape(g.rho, self.h_mode, s.Ti,
                                                   m.R0, m.a)
        if getattr(self, "_chi_e", None) is None:
            self._chi_e, self._chi_i = chi_new, chi_i_new
        self._chi_e = self._chi_e + (chi_new - self._chi_e) * w
        self._chi_i = self._chi_i + (chi_i_new - self._chi_i) * w
        chi, chi_i = self._chi_e, self._chi_i
        self.chi = chi
        cap_e = W_UNIT * s.ne
        cap_i = W_UNIT * s.ne * c["f_ion"]
        zero = np.zeros(g.n)

        T_sep = 0.12 if self.h_mode else 0.05
        self._update_pedestal(dt)
        i_ped = int(round(0.93 * (g.n - 1))) if self.h_mode else None
        Te_new = tr._diffuse(g, s.Te, cap_e, chi, s_e, zero, dt,
                             self.T_ped if self.h_mode else T_sep,
                             i_edge=i_ped, y_sep=T_sep)
        Ti_new = tr._diffuse(g, s.Ti, cap_i, chi_i, s_i, zero, dt,
                             self.T_ped if self.h_mode else T_sep,
                             i_edge=i_ped, y_sep=T_sep)
        s.Te = np.maximum(Te_new, 0.02)
        s.Ti = np.maximum(Ti_new, 0.02)

        # --- particles ----------------------------------------------------------
        # particle diffusivity: D/chi ~ 0.25 is the usual experimental ratio
        D = 0.25 * chi
        # source rates are volume-averaged [1e20 m^-3 s^-1]; normalise_source
        # wants the volume-integrated total, hence the factor of V
        gas_rate = self.act.gas * 0.02 * m.V
        s_gas = tr.normalise_source(g, tr.deposition(rho, "edge"), gas_rate)
        # beam fuelling: one ion per beam particle, P / E_beam of them
        nbi_rate = self.act.p_nbi * 62.4 / max(m.e_nbi_kev, 1.0)   # [1e20 /s]
        s_nbi = tr.normalise_source(g, tr.deposition(rho, "broad"), nbi_rate)
        # Density control.  Fuelling acts on the separatrix density through
        # recycling, and the core profile then follows from D and the pinch --
        # which is how density feedback actually works on a machine.  A
        # volumetric "gas source" with no sink cannot hold a set point at all.
        pinch = tr.pinch_profile(rho)
        ne_new = tr._diffuse(g, s.ne, np.ones(g.n), D, s_gas + s_nbi,
                             zero, dt, self.n_sep, pinch=pinch)
        s.ne = np.maximum(ne_new, 0.005)

        # Helium ash: born where the fusion is, transported like the fuel,
        # and removed only by the divertor pumps.  Holding the separatrix
        # helium at zero AND applying a 1/tau_He sink double-counts the
        # removal and pins the ash two orders of magnitude below where it
        # belongs.  What recycling actually does is hold a finite helium
        # concentration at the edge, so that is what is controlled here,
        # towards the assumed tau_He* / tau_E.
        src_he = self.p_fus_prof * MW / E_FUSION_J / 1e20
        he_new = tr._diffuse(g, s.n_he, np.ones(g.n), D, src_he,
                             zero, dt, self.n_he_sep, pinch=pinch)
        s.n_he = np.maximum(he_new, 0.0)
        src_total = g.integrate(src_he)
        n_he_want = src_total * cfg.tau_he_ratio * self.tau_E / max(m.V, 1e-6)
        self.n_he_sep = float(np.clip(
            self.n_he_sep + 0.8 * (n_he_want - s.n_he_avg) * dt
            / max(self.tau_E, 0.05), 0.0, 0.4))

        self.t += dt
        self._derive()

        # --- equilibrium ---------------------------------------------------------
        if cfg.equilibrium:
            if (self.t - self._t_last_eq) >= cfg.eq_interval \
                    and self.Ip > 0.25 * m.Ip:
                self._solve_equilibrium()
            # glide towards the latest solution on the resistive timescale
            self._apply_equilibrium(dt / max(cfg.eq_relax, 1e-3))

        # --- divertor -------------------------------------------------------------
        self._solve_divertor()

        # --- limits ---------------------------------------------------------------
        if cfg.disruption_enabled and self.t > 4.0 and self.Ip > 0.15 * m.Ip:
            cause = disr.check_limits(
                f_greenwald=self.f_G, q95=self.q95, beta_n=self.beta_n,
                p_rad=self.P_rad, p_heat=p_heat, Te_avg=s.Te_avg, li=self.li)
            if cause:
                self.trigger_disruption(cause)

    def _step_disrupted(self, dt: float) -> None:
        s = self.state
        self.Ip = max(0.0, self.Ip - dt * self.m.Ip / max(
            (self.disruption.tau_cq_ms / 1e3), 1e-3))
        s.Te = np.maximum(s.Te * np.exp(-dt / 2e-3), 0.005)
        s.Ti = np.maximum(s.Ti * np.exp(-dt / 2e-3), 0.005)
        self.t += dt
        self._derive()

    # -- sub-models ---------------------------------------------------------
    def _solve_equilibrium(self) -> None:
        m = self.m
        # A sawtoothing plasma clamps q on axis at 1: every time q0 drops
        # below it the internal kink reconnects the core and pushes it back.
        # Without that constraint the current profile shape is free, and
        # since the pedestal goes as q95^-4 an unconstrained q95 propagates
        # into a 30% error in the fusion power.
        q0_target = 1.0 if (self.state.Te_avg > 2.0
                            and self.Ip > 0.7 * m.Ip) else None
        try:
            nR, nZ = self.cfg.eq_grid
            eq = eqm.solve_matched(m.R0, m.a, m.kappa_x, m.delta_x,
                                   self.Ip, m.B0, p_avg=self.p_avg,
                                   q0=q0_target, nR=nR, nZ=nZ, max_outer=6)
        except Exception:
            return
        self.eq = eq
        self._t_last_eq = self.t
        # The equilibrium is re-solved at intervals, but q95 must not step:
        # the pedestal goes as q95^-4, so a discrete jump in q95 drives a
        # 20% swing in fusion power and the burn never settles.  Relax the
        # equilibrium quantities towards the new solution instead -- which
        # is also physically right, since the current profile evolves on
        # the resistive timescale, not instantaneously.
        self._eq_target = {"q95": eq.q95, "q0": eq.q0, "li": eq.li3,
                           "q_prof": np.interp(self.grid.rho, eq.rho, eq.q)}
        if not self._eq_seen:
            self._apply_equilibrium(1.0)
            self._eq_seen = True

    def _apply_equilibrium(self, frac: float) -> None:
        t = getattr(self, "_eq_target", None)
        if not t:
            return
        f = float(np.clip(frac, 0.0, 1.0))
        self.q95 += (t["q95"] - self.q95) * f
        self.q0 += (t["q0"] - self.q0) * f
        self.li += (t["li"] - self.li) * f
        self.q_prof = self.q_prof + (t["q_prof"] - self.q_prof) * f

    def _solve_divertor(self) -> None:
        m = self.m
        B_pol = b_poloidal(max(self.Ip, 1e-3), m.L_pol)
        n_sep = 0.25 * max(self.state.ne_avg, 0.02)
        try:
            self.div = sol_mod.two_point(
                self.P_sep, R0=m.R0, a=m.a, B0=m.B0, B_pol=B_pol,
                q95=max(self.q95, 1.5), n_sep20=n_sep, A_mass=m.a_mass,
                f_rad_seed=0.60)
        except Exception:
            self.div = None

    def trigger_disruption(self, cause: str) -> None:
        if self.disrupted:
            return
        m = self.m
        self.disrupted = True
        self.disruption = disr.simulate(
            cause=cause, Ip=self.Ip, R0=m.R0, a=m.a, kappa=m.kappa_x,
            B0=m.B0, W_thermal=self.W_th, li=self.li,
            ne20=self.state.ne_avg, z_eff=self.c["z_eff"], S_wall=m.S,
            mitigated=self.cfg.mitigation)
        self.alarms.append(f"t={self.t:.1f}s ДИЗРУПЦИЯ: {cause}")

    # -- reporting -----------------------------------------------------------
    def scalars(self) -> dict:
        s, m, c = self.state, self.m, self.c
        d = {
            "t": self.t, "Ip": self.Ip, "B0": m.B0,
            "ne": s.ne_avg, "n_bar": s.n_bar, "ne0": float(s.ne[0]),
            "Te": s.Te_avg, "Ti": s.Ti_avg,
            "Te0": float(s.Te[0]), "Ti0": float(s.Ti[0]),
            "Pfus": self.P_fus, "Palpha": self.P_alpha, "Paux": self.P_aux,
            "Pohm": getattr(self, "P_ohm", 0.0), "Prad": self.P_rad,
            "Pbrem": self.P_brem, "Psync": self.P_sync, "Pline": self.P_line,
            "Psep": self.P_sep, "Ploss": self.P_loss, "Plh": self.P_lh,
            "Q": self.Q, "Wth": self.W_th, "Wfast": self.W_fast, "W": self.W,
            "tauE": self.tau_E, "tau_scaling": self.tau_scaling,
            "H_factor": self.H_factor,
            "q0": self.q0, "q95": self.q95, "li": self.li,
            "betaN": self.beta_n, "betaT": self.beta_t * 100.0,
            "betaP": self.beta_p, "nG": self.n_G, "fG": self.f_G,
            "Zeff": c["z_eff"], "fDT": c["f_dt"],
            "fHePc": 100.0 * s.n_he_avg / max(s.ne_avg, 1e-9),
            "fBS": self.f_bs, "Ibs": self.I_bs, "Icd": self.I_cd,
            "Vloop": getattr(self, "V_loop", 0.0),
            "flux_used": self.flux.consumed,
            "flux_fraction": self.flux.fraction,
            "neutronRate": self.neutron_rate,
            "hMode": self.h_mode, "disrupted": self.disrupted,
            "Tped": self.T_ped,
            "ped_width_psi": self.ped.width_psi if self.ped else 0.0,
            "ped_p": self.ped.p_ped if self.ped else 0.0,
            "V": m.V, "S": m.S, "kappaA": m.kappa_a,
        }
        if getattr(self, "div", None) is not None:
            d.update({"qDiv": self.div.q_target_MWm2,
                      "T_target_eV": self.div.T_target_eV,
                      "lambda_q_mm": self.div.lambda_q_mm,
                      "div_regime": self.div.regime,
                      "f_rad_div": self.div.f_rad_div})
        return d

    def profiles(self) -> dict:
        g, s = self.grid, self.state
        return {"rho": g.rho.tolist(), "ne": s.ne.tolist(),
                "Te": s.Te.tolist(), "Ti": s.Ti.tolist(),
                "nHe": s.n_he.tolist(), "q": self.q_prof.tolist(),
                "p_fus": self.p_fus_prof.tolist(),
                "p_rad": self.p_rad_prof.tolist(),
                "chi": getattr(self, "chi", np.zeros(g.n)).tolist()}

    def plant(self) -> dict:
        b = fc.blanket(self.P_fus, self.m.S)
        p = fc.plant_balance(self.P_fus, max(self.P_aux, 1e-6),
                             P_ohmic=getattr(self, "P_ohm", 0.0),
                             superconducting=self.m.superconducting,
                             R0=self.m.R0)
        return {"blanket": b.to_dict(), "plant": p.to_dict(),
                "damage": fc.neutron_damage(b.wall_load)}

    # -- scripted discharge --------------------------------------------------
    def run_scenario(self, t_end: float = 120.0,
                     record_every: float = 1.0) -> List[dict]:
        """Run the reference inductive discharge for this machine."""
        m = self.m
        next_rec = 0.0
        while self.t < t_end and not (self.disrupted and self.Ip < 0.05):
            self._scenario_actuators()
            self.step()
            if self.t >= next_rec:
                self.log.append(self.scalars())
                next_rec += record_every
        return self.log

    def _scenario_actuators(self) -> None:
        m, t = self.m, self.t
        scale = m.Ip / 15.0

        def ramp(t0, t1, v0, v1):
            if t <= t0:
                return v0
            if t >= t1:
                return v1
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)

        self.act.Ip_request = ramp(1, 26, 0.4 * scale, m.Ip)
        # ECRH carries the ramp-up and is handed over to NBI/ICRF; it must
        # not ramp out before they arrive or the plasma radiates away in the gap
        self.act.p_ecrf = (12.0 * scale if 1.5 < t < 10 else
                           (ramp(10, 24, 12.0 * scale, 0.0) if t >= 10 else 0.0))
        self.act.p_nbi = 0.0 if t < 14 else ramp(14, 22, 0.0, 33.0 * scale)
        self.act.p_icrf = 0.0 if t < 18 else ramp(18, 24, 0.0, 17.0 * scale)

        f_target = ramp(2, 20, 0.30, 0.46)
        if self.h_mode:
            self._f_ramp = min(0.857, getattr(self, "_f_ramp", 0.46) + 0.028 * self.cfg.dt)
            f_target = max(f_target, self._f_ramp)

        n_target = f_target * max(self.n_G, 0.005)
        self.n_target = n_target
        # proportional feedback on the separatrix density
        err = n_target - self.state.ne_avg
        self.n_sep = float(np.clip(self.n_sep + 1.2 * err * self.cfg.dt,
                                   0.005, 0.85 * max(self.n_G, 0.01)))
        self.act.gas = float(np.clip(4.0 * err, 0.0, 1.0))
