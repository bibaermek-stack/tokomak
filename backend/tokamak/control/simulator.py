"""Block 1 -- the physical simulator.

A free-boundary *rigid-current-displacement* model (the "RZIp" class used
for vertical-stability design on DIII-D, TCV and ITER) coupled to the
project's fixed-boundary Grad-Shafranov solver:

* **Equilibrium.**  At t = 0 the Grad-Shafranov solver
  (:mod:`tokamak.equilibrium`) produces j_phi(R, Z), l_i and beta_p for the
  machine.  The current density is coarse-grained into toroidal filaments
  that keep their relative weights and move together with the current
  centroid (R_c, Z_c).  Coil currents that hold this plasma -- radial and
  vertical force balance plus constant flux on the target boundary
  (isoflux) -- come from a regularised least-squares solve.

* **Circuits.**  PF coils, the central solenoid, an in-vessel vertical
  stabilisation pair, the vacuum-vessel segments and the plasma obey

      d/dt [ M(x) I ] + R I = V,       I = [I_coils, I_vessel, I_p]
                                       V = [V_coils, 0,        0  ]

  where only the plasma row and column of M depend on the plasma position
  x = (R_c, Z_c).  Writing it for M I rather than M dI/dt keeps the motional
  EMF of a moving plasma, which is what makes the vessel currents push back
  on a vertical displacement.

* **Force balance.**  The plasma is massless (its Alfven time is ~1 us,
  three orders below the control step), so x is not a dynamical variable
  but the root of

      F_R = 2 pi Ip sum_f w_f R_f B_Z,ext(f) + mu0 Ip^2 / 2 * Lambda = 0
      F_Z = -2 pi Ip sum_f w_f R_f B_R,ext(f)                      = 0
      Lambda = ln(8 R_c / a sqrt(kappa)) + beta_p + l_i / 2 - 3/2

  (Shafranov's hoop + tyre-tube force against the external field).  An
  elongated plasma sits at an *unstable* root in Z; what slows the escape
  from it to the vessel L/R time is the induced vessel current, and that is
  the physics a vertical controller has to beat.

Integration is implicit Euler on the coupled DAE, with a chord-Newton
iteration on x at every sub-step: for a trial x the circuit equation is
linear in the new currents, so only a 2-D root has to be found.  Implicit
Euler damps an unstable mode slightly (amplification 1/(1 - gamma h)
against e^(gamma h)); the default sub-step keeps gamma h below 0.1.

What it does NOT model: plasma shape changes beyond rigid translation,
profile evolution (l_i and beta_p change only through imposed
disturbances), eddy currents in anything but the vessel, coil-power-supply
dynamics beyond a voltage limit.  The coil set is a generic layout scaled
to the machine, not an as-built engineering geometry.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..constants import MU0
from ..equilibrium import solve as gs_solve
from ..geometry import boundary
from ..machines import Machine, get_machine
from .greens import mutual, mutual_and_field, field as loop_field, \
    self_inductance

#: Resistivities [Ohm m]: water-cooled copper coils, stainless steel vessel.
RHO_CU = 1.75e-8
RHO_SS = 7.4e-7


class LossOfEquilibrium(RuntimeError):
    """No force-balance root near the previous position: the plasma is lost."""


# ---------------------------------------------------------------------------
@dataclass
class DeviceConfig:
    """Generic conductor layout scaled to a registered machine.

    Lengths are in units of the minor radius unless stated otherwise.
    """
    machine: str = "ktm"
    Ip_MA: Optional[float] = None       # None -> machine flat-top value

    # vacuum vessel: a D-shaped shell, n_vessel segments
    wall_gap: float = 0.30              # plasma-to-wall gap on the midplane
    wall_thickness_m: float = 0.004
    n_vessel: int = 24

    # PF coils on an ellipse outside the vessel
    pf_radius: float = 2.2              # semi-axis / a (scaled by kappa in Z)
    pf_angles_deg: tuple = (-140.0, -85.0, -30.0, 30.0, 85.0, 140.0)
    pf_turns: int = 40
    pf_area_m2: float = 0.02

    # central solenoid, one circuit of n_cs_filaments stacked filaments
    cs_turns: int = 480
    flux_budget_s: float = 1.0          # flat-top the CS swing must cover
    cs_filaments: int = 6
    cs_area_m2: float = 0.03

    # in-vessel vertical-stabilisation pair, anti-series in one circuit
    vs_turns: int = 4
    vs_area_m2: float = 0.002
    vs_angle_deg: float = 55.0
    vs_radius: float = 1.6

    # plasma
    Te_keV: float = 1.0                 # sets the Spitzer resistance
    z_eff: float = 2.0
    gs_grid: tuple = (65, 97)
    filament_block: int = 9             # GS cells per filament, each way

    # isoflux control points on the target boundary
    n_boundary: int = 16
    coil_regularisation: float = 1e-2


@dataclass
class Conductors:
    """Every filament, grouped into circuit elements."""
    R: np.ndarray                 # filament major radius [m]
    Z: np.ndarray
    turns: np.ndarray             # signed turns per filament
    element: np.ndarray           # circuit index of each filament
    names: list                   # one per circuit element
    resistance: np.ndarray        # per circuit [Ohm]
    n_active: int                 # circuits 0..n_active-1 are driven


@dataclass
class PlasmaShape:
    """The rigid current distribution, relative to its own centroid."""
    dR: np.ndarray                # filament offsets from the centroid
    dZ: np.ndarray
    w: np.ndarray                 # current fractions, sum = 1
    R0: float                     # centroid at t = 0
    Z0: float
    a: float
    kappa: float
    li: float
    beta_p: float
    R_axis_offset: float          # magnetic axis minus centroid (GS)
    boundary_dR: np.ndarray       # LCFS relative to the centroid
    boundary_dZ: np.ndarray
    resistance: float             # [Ohm]
    gs_summary: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
def _ellipse_points(R0, a, kappa, delta, n):
    R, Z = boundary(R0, a, kappa, delta, n)
    return R[:-1], Z[:-1]


def build_device(cfg: DeviceConfig, m: Optional[Machine] = None):
    """Conductors, plasma filaments and wall contour for a machine."""
    m = m or get_machine(cfg.machine)
    Ip = (cfg.Ip_MA if cfg.Ip_MA is not None else m.Ip)
    a, R0, kap, dlt = m.a, m.R0, m.kappa_x, m.delta_x

    # --- equilibrium -------------------------------------------------------
    eq = gs_solve(R0, a, kap, dlt, Ip, m.B0, nR=cfg.gs_grid[0],
                  nZ=cfg.gs_grid[1], diverted=m.diverted)
    b = cfg.filament_block
    nZ, nR = eq.j_phi.shape
    nZb, nRb = nZ // b, nR // b
    Rg, Zg = np.meshgrid(eq.R, eq.Z)
    J = eq.j_phi[:nZb * b, :nRb * b] * eq.mask[:nZb * b, :nRb * b]
    Rw = Rg[:nZb * b, :nRb * b]
    Zw = Zg[:nZb * b, :nRb * b]
    blk = lambda A: A.reshape(nZb, b, nRb, b).sum(axis=(1, 3))
    Ib = blk(J)
    keep = Ib > 1e-4 * Ib.sum()
    w = Ib[keep] / Ib[keep].sum()
    Rf = blk(J * Rw)[keep] / Ib[keep]
    Zf = blk(J * Zw)[keep] / Ib[keep]
    Rc, Zc = float(np.sum(w * Rf)), float(np.sum(w * Zf))

    Rb, Zb = boundary(R0, a, kap, dlt, 64)
    # Spitzer parallel resistivity with a neoclassical factor of 2
    eta = 2.0 * 1.65e-9 * cfg.z_eff * 17.0 / cfg.Te_keV ** 1.5
    A_cs = np.pi * a * a * kap
    Rp = eta * 2.0 * np.pi * R0 / A_cs

    plasma = PlasmaShape(
        dR=Rf - Rc, dZ=Zf - Zc, w=w, R0=Rc, Z0=Zc, a=a, kappa=kap,
        li=float(eq.li3), beta_p=float(eq.beta_p),
        R_axis_offset=float(eq.R_axis - Rc),
        boundary_dR=Rb[:-1] - Rc, boundary_dZ=Zb[:-1] - Zc,
        resistance=float(Rp), gs_summary=eq.summary())

    # --- conductors --------------------------------------------------------
    Rs, Zs, turns, elem, names, res = [], [], [], [], [], []

    def add(Rl, Zl, tl, name, R_ohm):
        k = len(names)
        Rs.extend(Rl); Zs.extend(Zl); turns.extend(tl)
        elem.extend([k] * len(Rl)); names.append(name); res.append(R_ohm)

    # central solenoid: inboard of the vessel, as tall as the PF ellipse
    gap = cfg.wall_gap * a
    if R0 - a - gap < 0.1 * a:
        raise ValueError(
            f"{m.label}: no room inboard for the vessel and a solenoid "
            f"(R0 - a - gap = {R0 - a - gap:.2f} m).  The generic layout is "
            "for conventional aspect ratio; a spherical torus needs its own "
            "conductor geometry.")
    R_cs = max(0.65 * (R0 - a - gap), 0.08)
    h_cs = 0.9 * cfg.pf_radius * kap * a
    zc = np.linspace(-h_cs, h_cs, cfg.cs_filaments)
    n_per = cfg.cs_turns / cfg.cs_filaments
    r_cs = RHO_CU * 2.0 * np.pi * R_cs * cfg.cs_turns ** 2 \
        / (0.6 * cfg.cs_area_m2 * cfg.cs_filaments)
    add([R_cs] * cfg.cs_filaments, list(zc), [n_per] * cfg.cs_filaments,
        "CS", r_cs)

    for k, th in enumerate(np.deg2rad(cfg.pf_angles_deg)):
        # a spherical torus has no room inboard: keep clear of the CS
        Rk = max(R0 + cfg.pf_radius * a * np.cos(th), R_cs + 0.1 * a)
        Zk = cfg.pf_radius * kap * a * np.sin(th)
        rk = RHO_CU * 2.0 * np.pi * Rk * cfg.pf_turns ** 2 \
            / (0.6 * cfg.pf_area_m2)
        add([Rk], [Zk], [cfg.pf_turns], f"PF{k + 1}", rk)

    th = np.deg2rad(cfg.vs_angle_deg)
    Rv = max(R0 + cfg.vs_radius * a * np.cos(th), R_cs + 0.1 * a)
    Zv = cfg.vs_radius * kap * a * np.sin(th)
    r_vs = 2 * RHO_CU * 2.0 * np.pi * Rv * cfg.vs_turns ** 2 \
        / (0.6 * cfg.vs_area_m2)
    add([Rv, Rv], [Zv, -Zv], [cfg.vs_turns, -cfg.vs_turns], "VS", r_vs)
    n_active = len(names)

    # vessel: a D-shell with half the plasma triangularity
    a_v = a + gap
    kap_v = (kap * a + gap) / a_v
    Rw_, Zw_ = boundary(R0, a_v, kap_v, 0.5 * dlt, cfg.n_vessel)
    Rmid = 0.5 * (Rw_[:-1] + Rw_[1:])
    Zmid = 0.5 * (Zw_[:-1] + Zw_[1:])
    seg = np.hypot(np.diff(Rw_), np.diff(Zw_))
    for k in range(cfg.n_vessel):
        rk = RHO_SS * 2.0 * np.pi * Rmid[k] / (cfg.wall_thickness_m * seg[k])
        add([Rmid[k]], [Zmid[k]], [1.0], f"VV{k + 1}", rk)

    cond = Conductors(R=np.array(Rs), Z=np.array(Zs), turns=np.array(turns),
                      element=np.array(elem), names=names,
                      resistance=np.array(res), n_active=n_active)

    # inner surface of the vessel, finely sampled, for the gap check
    wall_R, wall_Z = boundary(R0, a_v - 0.5 * cfg.wall_thickness_m,
                              (kap * a + gap - 0.5 * cfg.wall_thickness_m)
                              / (a_v - 0.5 * cfg.wall_thickness_m),
                              0.5 * dlt, 144)
    # radial size of each element, for self-inductances
    sizes = np.zeros(len(names))
    sizes[0] = 0.2235 * 2 * np.sqrt(cfg.cs_area_m2)
    sizes[1:1 + len(cfg.pf_angles_deg)] = 0.2235 * 2 * np.sqrt(cfg.pf_area_m2)
    sizes[n_active - 1] = 0.2235 * 2 * np.sqrt(cfg.vs_area_m2)
    sizes[n_active:] = 0.2235 * (seg + cfg.wall_thickness_m)
    return m, Ip, eq, plasma, cond, (wall_R, wall_Z), sizes


# ---------------------------------------------------------------------------
@dataclass
class PlasmaState:
    """What the simulator exposes each step (block 1 -> block 2)."""
    t: float
    R_c: float
    Z_c: float
    Ip: float                     # [A]
    I_coils: np.ndarray           # [A per turn]
    I_vessel: np.ndarray
    psi_loops: np.ndarray         # [Wb]
    B_probes: np.ndarray          # [T] tangential to the wall
    dIp_dt: float
    d_min: float                  # plasma-wall clearance [m]
    gaps: np.ndarray              # isoflux boundary error [m]
    R_axis: float
    Z_axis: float


class TokamakSimulator:
    """Coupled circuit / force-balance simulator (block 1)."""

    def __init__(self, cfg: Optional[DeviceConfig] = None,
                 dt: float = 1e-3, n_sub: int = 4,
                 n_flux_loops: int = 12, n_probes: int = 16):
        self.cfg = cfg or DeviceConfig()
        (self.machine, Ip_MA, self.eq, self.plasma, self.cond,
         self.wall, sizes) = build_device(self.cfg)
        self.dt = dt
        self.n_sub = n_sub
        self.Ip0 = Ip_MA * 1e6
        c, p = self.cond, self.plasma
        self.n_el = len(c.names)
        self.n_act = c.n_active
        self.coil_names = c.names[:c.n_active]

        # --- constant external-external inductance matrix -----------------
        n = self.n_el
        T = np.zeros((n, len(c.R)))
        T[c.element, np.arange(len(c.R))] = c.turns
        self._T = T                               # element <- filament map
        G = mutual(c.R[:, None], c.Z[:, None], c.R[None, :], c.Z[None, :])
        np.fill_diagonal(G, 0.0)
        M_ee = T @ G @ T.T
        for k in range(n):
            fil = c.element == k
            # self-inductance of each filament, then the series sum handles
            # mutual coupling between filaments of the same element
            for j in np.flatnonzero(fil):
                M_ee[k, k] += c.turns[j] ** 2 * self_inductance(c.R[j], sizes[k])
        self.M_ee = M_ee
        self.R_nom = np.concatenate([c.resistance, [p.resistance]])
        self.R_el = self.R_nom.copy()           # what the plant actually has
        self.plant = {"wall_res": 1.0, "plasma_res": 1.0}

        # --- diagnostics -------------------------------------------------
        wR, wZ = self.wall
        idx = np.linspace(0, len(wR) - 1, n_flux_loops, endpoint=False).astype(int)
        self.loop_R, self.loop_Z = wR[idx] * 1.0, wZ[idx] * 1.0
        idx = np.linspace(0, len(wR) - 1, n_probes, endpoint=False).astype(int)
        self.probe_R, self.probe_Z = wR[idx], wZ[idx]
        tR = np.gradient(wR)[idx]
        tZ = np.gradient(wZ)[idx]
        tn = np.hypot(tR, tZ)
        self.probe_t = np.stack([tR / tn, tZ / tn])
        self.M_loop_e = mutual(self.loop_R[:, None], self.loop_Z[:, None],
                               c.R[None, :], c.Z[None, :]) @ T.T
        BR, BZ = loop_field(c.R[None, :], c.Z[None, :],
                            self.probe_R[:, None], self.probe_Z[:, None])
        self.Bt_probe_e = (self.probe_t[0][:, None] * BR
                           + self.probe_t[1][:, None] * BZ) @ T.T

        # isoflux control points on the target boundary
        kb = np.linspace(0, len(p.boundary_dR), self.cfg.n_boundary,
                         endpoint=False).astype(int)
        self.ctrl_R = p.R0 + p.boundary_dR[kb]
        self.ctrl_Z = p.Z0 + p.boundary_dZ[kb]
        self.M_ctrl_e = mutual(self.ctrl_R[:, None], self.ctrl_Z[:, None],
                               c.R[None, :], c.Z[None, :]) @ T.T

        self.ref_R, self.ref_Z = p.R0, p.Z0
        self.li, self.beta_p = p.li, p.beta_p
        self._F_scale = MU0 * self.Ip0 ** 2 * p.R0 / p.a
        self._init_wall_distance()
        self._init_coils()
        self.reset()

    # ------------------------------------------------------------------
    def _plasma_couplings(self, Rc: float, Zc: float):
        """Plasma-external mutuals and the external field on the plasma.

        Returns m_pe (n_el,) [H], and the weighted field moments
        gR, gZ (n_el,) such that F_R(ext) = 2 pi Ip gZ . I and
        F_Z = -2 pi Ip gR . I.
        """
        c, p = self.cond, self.plasma
        Rf = Rc + p.dR
        Zf = Zc + p.dZ
        M, BR, BZ = mutual_and_field(c.R[None, :], c.Z[None, :],
                                     Rf[:, None], Zf[:, None])
        wf = p.w[:, None]
        m_pe = (wf * M).sum(axis=0) @ self._T.T
        gR = (wf * Rf[:, None] * BR).sum(axis=0) @ self._T.T
        gZ = (wf * Rf[:, None] * BZ).sum(axis=0) @ self._T.T
        return m_pe, gR, gZ

    def _Lp(self, Rc: float) -> float:
        p = self.plasma
        a_eff = p.a * np.sqrt(p.kappa)
        return float(MU0 * Rc * (np.log(8.0 * Rc / a_eff) + self.li / 2 - 2.0))

    def _hoop(self, Rc: float, Ip: float) -> float:
        p = self.plasma
        a_eff = p.a * np.sqrt(p.kappa)
        lam = np.log(8.0 * Rc / a_eff) + self.beta_p + self.li / 2 - 1.5
        return float(0.5 * MU0 * Ip * Ip * lam)

    def _M(self, Rc, Zc):
        m_pe, gR, gZ = self._plasma_couplings(Rc, Zc)
        n = self.n_el
        M = np.empty((n + 1, n + 1))
        M[:n, :n] = self.M_ee
        M[:n, n] = m_pe
        M[n, :n] = m_pe
        M[n, n] = self._Lp(Rc)
        return M, gR, gZ

    def _forces(self, Rc, Ip, I_ext, gR, gZ):
        FR = 2.0 * np.pi * Ip * (gZ @ I_ext) + self._hoop(Rc, Ip)
        FZ = -2.0 * np.pi * Ip * (gR @ I_ext)
        return np.array([FR, FZ])

    # ------------------------------------------------------------------
    def _init_coils(self):
        """Coil currents that hold the GS plasma at the target position.

        Weighted least squares over the active coils plus the boundary
        flux psi_b: force balance (two heavily weighted rows), constant flux
        on the isoflux control points, Tikhonov regularisation, and the VS
        pair held at zero.
        """
        p = self.plasma
        na = self.n_act
        Ip = self.Ip0
        _, gR, gZ = self._plasma_couplings(p.R0, p.Z0)
        # plasma flux on the control points
        Mcp = mutual(self.ctrl_R[:, None], self.ctrl_Z[:, None],
                     (p.R0 + p.dR)[None, :], (p.Z0 + p.dZ)[None, :])
        psi_p = (Mcp * p.w[None, :]).sum(axis=1) * Ip

        B_scale = MU0 * Ip / (2 * np.pi * p.a)
        F_scale = 2 * np.pi * p.R0 * Ip * B_scale
        psi_scale = 2 * np.pi * p.R0 * B_scale * p.a
        nb = len(self.ctrl_R)
        rows, rhs = [], []
        # force balance
        rows.append(np.concatenate([2 * np.pi * Ip * gZ[:na], [0.0]]) / F_scale * 1e3)
        rhs.append(-self._hoop(p.R0, Ip) / F_scale * 1e3)
        rows.append(np.concatenate([-2 * np.pi * Ip * gR[:na], [0.0]]) / F_scale * 1e3)
        rhs.append(0.0)
        # isoflux: psi(ctrl) - psi_b = 0
        A_iso = np.hstack([self.M_ctrl_e[:, :na], -np.ones((nb, 1))]) / psi_scale
        rows.extend(A_iso)
        rhs.extend(-psi_p / psi_scale)
        # VS pair carries no equilibrium current
        e = np.zeros(na + 1); e[na - 1] = 1e3
        rows.append(e); rhs.append(0.0)
        # the CS swing that supplies the loop voltage for flux_budget_s; no
        # pre-magnetisation -- its stray field would cost shape accuracy
        # and a simulated window of a second does not need it
        m_pe, _, _ = self._plasma_couplings(p.R0, p.Z0)
        self.cs_swing = abs(self.R_el[-1] * Ip * self.cfg.flux_budget_s
                            / m_pe[0])
        # regularisation, in ampere-turns relative to Ip
        turns = np.array([np.abs(self.cond.turns[self.cond.element == k]).sum()
                          for k in range(na)])
        lam = self.cfg.coil_regularisation
        for k in range(na):
            r = np.zeros(na + 1); r[k] = lam * turns[k] / Ip
            rows.append(r); rhs.append(0.0)
        A = np.array(rows)
        y = np.array(rhs)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.I_coils0 = sol[:na]
        self.psi_b0 = float(sol[na])
        self.turns_active = turns

        # limits: generous on the equilibrium currents, floors for the
        # coils that happen to carry little, voltages from a slew time
        Imax = np.maximum(2.0 * np.abs(self.I_coils0),
                          0.25 * np.max(np.abs(self.I_coils0 * turns)) / turns)
        Imax[na - 1] = 0.02 * Ip / turns[na - 1]
        Imax[0] = max(Imax[0], self.cs_swing)
        self.I_max = Imax
        L = np.diag(self.M_ee)[:na]
        tau = np.full(na, 0.15)
        tau[0] = 0.5
        tau[na - 1] = 0.004
        self.V_max = np.maximum(3.0 * self.R_el[:na] * Imax, L * Imax / tau)
        self.V_max = np.maximum(self.V_max, 3.0 * np.abs(self.feedforward_at(
            self.I_coils0)))

    # ------------------------------------------------------------------
    def reset(self, vs_kick: float = 0.0,
              li: Optional[float] = None, beta_p: Optional[float] = None):
        """Return to the t = 0 equilibrium.

        A massless plasma cannot simply be put somewhere else: its position
        is whatever the currents make it.  ``vs_kick`` [A per turn] adds a
        current to the VS pair instead, which moves the force-balance root
        off the midplane and seeds the vertical mode.
        """
        p = self.plasma
        self.t = 0.0
        self.li = p.li if li is None else li
        self.beta_p = p.beta_p if beta_p is None else beta_p
        n = self.n_el
        self.I = np.zeros(n + 1)
        self.I[:self.n_act] = self.I_coils0
        self.I[self.n_act - 1] += vs_kick
        self.I[n] = self.Ip0
        self.x = np.array([p.R0, p.Z0])
        self._J = None
        self._v = np.zeros(2)
        self._last_M = None
        if vs_kick or li is not None or beta_p is not None:
            self._settle()
        self.dIp_dt = 0.0
        self.V_last = self.feedforward()
        return self.state()

    def _settle(self):
        """Move x to the force-balance root for the present currents."""
        for _ in range(30):
            _, gR, gZ = self._plasma_couplings(*self.x)
            F = self._forces(self.x[0], self.I[-1], self.I[:-1], gR, gZ)
            J = np.empty((2, 2))
            for k in range(2):
                xp = self.x.copy(); xp[k] += 1e-5
                _, gR_, gZ_ = self._plasma_couplings(*xp)
                J[:, k] = (self._forces(xp[0], self.I[-1], self.I[:-1],
                                        gR_, gZ_) - F) / 1e-5
            dx = -np.linalg.solve(J, F)
            self.x = self.x + np.clip(dx, -0.02, 0.02)
            if np.max(np.abs(dx)) < 1e-9:
                break

    def set_plant(self, wall_res: float = 1.0, plasma_res: float = 1.0) -> None:
        """Change the plant's resistances relative to the nominal model.

        ``wall_res`` scales every vacuum-vessel segment (steel resistivity,
        temperature, bolted joints), ``plasma_res`` the plasma (Spitzer
        resistance, Z_eff, electron temperature).  Feedforward voltages,
        the safety filter's prediction and the baseline's design keep using
        the nominal values (:attr:`R_nom`): a controller does not know the
        plant it is actually connected to.
        """
        self.R_el = self.R_nom.copy()
        self.R_el[self.n_act:self.n_el] *= wall_res
        self.R_el[-1] *= plasma_res
        self.plant = {"wall_res": float(wall_res),
                      "plasma_res": float(plasma_res)}
        self._J = None

    @contextlib.contextmanager
    def nominal_plant(self):
        """Temporarily put the nominal plant back (for designing against)."""
        saved = dict(self.plant)
        self.set_plant(1.0, 1.0)
        try:
            yield self
        finally:
            self.set_plant(**saved)

    def feedforward_at(self, I_coils: np.ndarray) -> np.ndarray:
        """Feedforward for given coil currents at the reference plasma."""
        na = self.n_act
        V = self.R_nom[:na] * I_coils
        m_pe, _, _ = self._plasma_couplings(self.plasma.R0, self.plasma.Z0)
        dIcs = -self.R_nom[-1] * self.Ip0 / m_pe[0]
        return V + self.M_ee[:na, 0] * dIcs

    def feedforward(self) -> np.ndarray:
        """Nominal coil voltages: resistive hold plus the CS ramp for Ip.

        V_ff = R I_coils + M_coils,cs dI_cs/dt, with the CS ramp rate that
        supplies the plasma's resistive loop voltage (M_p,cs dI_cs/dt =
        -R_p Ip) and the mutual term cancelling what that ramp would induce
        in the other coils.
        """
        na = self.n_act
        V = self.R_nom[:na] * self.I[:na]
        m_pe = self._M_now()[:self.n_el, self.n_el]
        dIcs = -self.R_nom[-1] * self.I[-1] / m_pe[0]
        # the other coils see the CS ramp through their mutuals; cancel it
        # so that only the CS current moves
        V += self.M_ee[:na, 0] * dIcs
        return V

    # ------------------------------------------------------------------
    def step(self, V_coils: np.ndarray) -> "PlasmaState":
        """Advance dt with the given coil voltages [V]."""
        V_coils = np.asarray(V_coils, float)
        n = self.n_el
        h = self.dt / self.n_sub
        Vfull = np.zeros(n + 1)
        Vfull[:self.n_act] = V_coils
        Ip_start = self.I[n]
        for _ in range(self.n_sub):
            self._substep(Vfull, h)
            self.t += h
        self.dIp_dt = (self.I[n] - Ip_start) / self.dt
        self.V_last = V_coils.copy()
        return self.state()

    def _residual(self, x, rhs0, h):
        M, gR, gZ = self._M(*x)
        A = M + h * np.diag(self.R_el)
        I = np.linalg.solve(A, rhs0)
        F = self._forces(x[0], I[-1], I[:-1], gR, gZ)
        self._last_M = M
        return F, I

    def _M_now(self):
        """M at the current position, reused from the last residual."""
        M = getattr(self, "_last_M", None)
        if M is None or not np.array_equal(self._last_x, self.x):
            M, _, _ = self._M(*self.x)
            self._last_M = M
        self._last_x = self.x.copy()
        return M

    def _substep(self, Vfull, h):
        M0 = self._M_now()
        rhs0 = M0 @ self.I + h * Vfull
        tol = 2e-7 * self._F_scale
        # predictor: carry the last sub-step's velocity forward
        v = getattr(self, "_v", np.zeros(2))
        x = self.x + v * h
        F, I = self._residual(x, rhs0, h)
        if self._J is None:
            self._J = self._jacobian(x, rhs0, h, F)
        it = 0
        while np.max(np.abs(F)) > tol:
            dx = -np.linalg.solve(self._J, F)
            step = np.max(np.abs(dx))
            if step > 0.05:
                dx *= 0.05 / step
            x = x + dx
            F, I = self._residual(x, rhs0, h)
            it += 1
            if it in (4, 12):            # slow: refresh the chord Jacobian
                self._J = self._jacobian(x, rhs0, h, F)
            if it >= 25 or not np.all(np.isfinite(x)):
                raise LossOfEquilibrium(
                    f"force balance diverged at t={self.t:.4f}s")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(I))):
            raise LossOfEquilibrium("non-finite state")
        self._v = (x - self.x) / h
        self.x, self.I = x, I
        self._last_x = x.copy()

    def _jacobian(self, x, rhs0, h, F0):
        J = np.empty((2, 2))
        eps = 1e-5
        for k in range(2):
            xp = x.copy(); xp[k] += eps
            Fp, _ = self._residual(xp, rhs0, h)
            J[:, k] = (Fp - F0) / eps
        return J

    # ------------------------------------------------------------------
    def perturb(self, li: Optional[float] = None,
                beta_p: Optional[float] = None) -> None:
        """Change the plasma's internal parameters (an ELM, a minor
        disruption, a sawtooth crash) -- the force balance moves with them."""
        if li is not None:
            self.li = li
        if beta_p is not None:
            self.beta_p = beta_p
        self._J = None
        self._v = np.zeros(2)

    # ------------------------------------------------------------------
    def plasma_boundary(self):
        p = self.plasma
        return self.x[0] + p.boundary_dR, self.x[1] + p.boundary_dZ

    def _init_wall_distance(self, res: float = 0.004):
        """Distance-to-wall field on a grid, for a fast clearance check."""
        wR, wZ = self.wall
        R = np.arange(wR.min() - 0.02, wR.max() + 0.02, res)
        Z = np.arange(wZ.min() - 0.02, wZ.max() + 0.02, res)
        Rg, Zg = np.meshgrid(R, Z)
        P = np.stack([Rg.ravel(), Zg.ravel()], axis=1)
        A = np.stack([wR[:-1], wZ[:-1]], axis=1)
        AB = np.stack([wR[1:], wZ[1:]], axis=1) - A
        d = np.full(len(P), np.inf)
        for k in range(len(A)):           # one wall segment at a time
            t = np.clip(((P - A[k]) @ AB[k]) / (AB[k] @ AB[k]), 0.0, 1.0)
            d = np.minimum(d, np.linalg.norm(P - (A[k] + t[:, None] * AB[k]),
                                             axis=1))
        from ..equilibrium import _inside
        sign = np.where(_inside(Rg, Zg, wR, wZ), 1.0, -1.0)
        self._wd = (R, Z, sign * d.reshape(Rg.shape))

    def wall_clearance(self) -> float:
        """Shortest distance from the plasma boundary to the first wall.

        Bilinear interpolation of a precomputed signed distance field (4 mm
        grid, negative outside the vessel; a distance function is
        1-Lipschitz, so the error stays under a millimetre).  A boundary
        point past the wall or off the grid gives zero.
        """
        bR, bZ = self.plasma_boundary()
        R, Z, D = self._wd
        fi = (bR - R[0]) / (R[1] - R[0])
        fj = (bZ - Z[0]) / (Z[1] - Z[0])
        if (fi.min() < 0 or fj.min() < 0 or fi.max() >= len(R) - 1
                or fj.max() >= len(Z) - 1):
            return 0.0
        i = fi.astype(int); j = fj.astype(int)
        u = fi - i; w = fj - j
        d = ((1 - u) * (1 - w) * D[j, i] + u * (1 - w) * D[j, i + 1]
             + (1 - u) * w * D[j + 1, i] + u * w * D[j + 1, i + 1])
        return float(max(d.min(), 0.0))

    def _plasma_flux_at(self, R, Z):
        p = self.plasma
        Mf = mutual(R[:, None], Z[:, None], (self.x[0] + p.dR)[None, :],
                    (self.x[1] + p.dZ)[None, :])
        return (Mf * p.w[None, :]).sum(axis=1) * self.I[-1]

    def boundary_gaps(self) -> np.ndarray:
        """Isoflux shape error on the target boundary [m].

        On the true LCFS psi is constant.  The flux at each target point
        minus their mean, divided by 2 pi R |B_p| there, is how far the
        actual boundary sits from the target along the normal.
        """
        psi = self.M_ctrl_e @ self.I[:-1] + self._plasma_flux_at(self.ctrl_R,
                                                                 self.ctrl_Z)
        if not hasattr(self, "_ctrl_gradpsi"):
            self._ctrl_gradpsi = self._grad_psi_ctrl()
        return (psi - psi.mean()) / self._ctrl_gradpsi

    def _grad_psi_ctrl(self):
        """2 pi R |B_p| on the control points at the reference state."""
        h = 1e-4
        I_ext = self.I[:-1]

        def psi_at(R, Z):
            M = mutual(R[:, None], Z[:, None], self.cond.R[None, :],
                       self.cond.Z[None, :]) @ self._T.T
            return M @ I_ext + self._plasma_flux_at(R, Z)
        gR = (psi_at(self.ctrl_R + h, self.ctrl_Z)
              - psi_at(self.ctrl_R - h, self.ctrl_Z)) / (2 * h)
        gZ = (psi_at(self.ctrl_R, self.ctrl_Z + h)
              - psi_at(self.ctrl_R, self.ctrl_Z - h)) / (2 * h)
        return np.maximum(np.hypot(gR, gZ), 1e-9)

    def state(self) -> PlasmaState:
        n = self.n_el
        I_ext = self.I[:n]
        p = self.plasma
        Rf = self.x[0] + p.dR
        Zf = self.x[1] + p.dZ
        Ip = self.I[n]
        psi = self.M_loop_e @ I_ext + (mutual(
            self.loop_R[:, None], self.loop_Z[:, None], Rf[None, :],
            Zf[None, :]) * p.w[None, :]).sum(axis=1) * Ip
        BR, BZ = loop_field(Rf[None, :], Zf[None, :],
                            self.probe_R[:, None], self.probe_Z[:, None])
        Bt = self.Bt_probe_e @ I_ext + ((self.probe_t[0][:, None] * BR
                                         + self.probe_t[1][:, None] * BZ)
                                        * p.w[None, :]).sum(axis=1) * Ip
        return PlasmaState(
            t=self.t, R_c=float(self.x[0]), Z_c=float(self.x[1]), Ip=float(Ip),
            I_coils=self.I[:self.n_act].copy(),
            I_vessel=self.I[self.n_act:n].copy(),
            psi_loops=psi, B_probes=Bt, dIp_dt=float(self.dIp_dt),
            d_min=self.wall_clearance(), gaps=self.boundary_gaps(),
            R_axis=float(self.x[0] + p.R_axis_offset), Z_axis=float(self.x[1]))

    # ------------------------------------------------------------------
    def vertical_growth_rate(self) -> float:
        """Open-loop growth rate [1/s] of the vertical mode at this state.

        Linearises the coupled circuit / force-balance DAE with the coil
        voltages frozen: eliminate x through dF = 0 and take the largest
        real eigenvalue of the resulting current dynamics.
        """
        n = self.n_el
        x0, I0 = self.x.copy(), self.I.copy()
        h = 1e-6
        M, gR, gZ = self._M(*x0)
        F0 = self._forces(x0[0], I0[-1], I0[:-1], gR, gZ)

        def F_of(x, I):
            _, gR_, gZ_ = self._plasma_couplings(*x)
            return self._forces(x[0], I[-1], I[:-1], gR_, gZ_)
        Fx = np.empty((2, 2))
        for k in range(2):
            xp = x0.copy(); xp[k] += h * 10
            Fx[:, k] = (F_of(xp, I0) - F0) / (h * 10)
        FI = np.empty((2, n + 1))
        for k in range(n + 1):
            Ip_ = I0.copy(); dI = 1e-6 * max(abs(I0[k]), 1.0)
            Ip_[k] += dI
            FI[:, k] = (F_of(x0, Ip_) - F0) / dI
        # dx = -Fx^-1 FI dI
        dxdI = -np.linalg.solve(Fx, FI)
        # d(M I)/dt = M dI/dt + (dM/dx I) dx/dt
        dMIdx = np.empty((n + 1, 2))
        for k in range(2):
            xp = x0.copy(); xp[k] += h * 10
            Mp, _, _ = self._M(*xp)
            dMIdx[:, k] = (Mp - M) @ I0 / (h * 10)
        Aeff = M + dMIdx @ dxdI
        ev = np.linalg.eigvals(-np.linalg.solve(Aeff, np.diag(self.R_el)))
        return float(np.max(ev.real))


# ---------------------------------------------------------------------------
@dataclass
class LinearModel:
    """One control step of the plant, linearised about an operating point.

        I_{n+1} = A I_n + B V_n,      x_n = x0 + C (I_n - I0)

    I is every circuit current, V the coil voltages, x = (R_c, Z_c).  The
    position is algebraic (force balance), hence the static output map C.
    """
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    I0: np.ndarray
    x0: np.ndarray
    V0: np.ndarray
    dt: float

    @property
    def growth_rate(self) -> float:
        """Growth rate [1/s] of the most unstable discrete mode."""
        rho = float(np.max(np.abs(np.linalg.eigvals(self.A))))
        return float(np.log(rho) / self.dt)


def linearise(sim: TokamakSimulator) -> LinearModel:
    """Finite-difference linearisation of ``sim.step`` about its state.

    The simulator is restored afterwards.
    """
    saved = (sim.I.copy(), sim.x.copy(), sim.t, sim.li, sim.beta_p,
             sim.V_last.copy())
    I0, x0 = sim.I.copy(), sim.x.copy()
    V0 = sim.feedforward()
    n, m = len(I0), sim.n_act

    def settle(I):
        sim.I = I.copy()
        sim.x = x0.copy()
        sim._J = None
        sim._v = np.zeros(2)
        sim._last_M = None
        sim._settle()

    def f(I, V):
        settle(I)
        sim.step(V)
        return sim.I.copy()

    Ib = f(I0, V0)
    A = np.empty((n, n))
    C = np.empty((2, n))
    for k in range(n):
        d = 1e-3 * max(abs(I0[k]), 1.0)
        Ik = I0.copy(); Ik[k] += d
        A[:, k] = (f(Ik, V0) - Ib) / d
        settle(Ik)
        C[:, k] = (sim.x - x0) / d
    B = np.empty((n, m))
    for k in range(m):
        d = 1e-2 * max(sim.V_max[k], 1.0)
        Vk = V0.copy(); Vk[k] += d
        B[:, k] = (f(I0, Vk) - Ib) / d
    sim.I, sim.x, sim.t, sim.li, sim.beta_p, sim.V_last = saved
    sim._J = None
    sim._v = np.zeros(2)
    sim._last_M = None
    return LinearModel(A=A, B=B, C=C, I0=I0, x0=x0, V0=V0, dt=sim.dt)
