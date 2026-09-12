"""One-dimensional radial transport.

Solves the electron and ion heat balance and the particle balance on a
normalised radial grid,

    3/2 d(n T)/dt = (1/V') d/drho [ V' n chi dT/drho / a^2 ] + S - L
    dn/dt        = (1/V') d/drho [ V' D  dn/drho / a^2 ] + S_n

implicitly in time, so profiles are an *output* rather than a prescribed
shape.  That is the main thing this module buys over a 0-D model: the
pedestal, the peaking, and hence the fusion power all follow from where the
transport is suppressed rather than from fitted exponents.

Transport models are selectable:

``scaling_anchored``
    chi(rho) has a fixed shape and a single amplitude solved each step so
    that the resulting global confinement time reproduces the chosen 0-D
    scaling.  Keeps the 1-D solution consistent with the validated 0-D
    result, and is the default.
``gyrobohm``
    chi = C rho_s^2 v_th / a, the local gyro-Bohm estimate.
``bohm_gyrobohm``
    The mixed Bohm/gyro-Bohm model, which adds the non-local Bohm term that
    makes the profiles stiff.
``cgm``
    Critical-gradient: transport switches on hard above a threshold
    normalised temperature gradient, the qualitative behaviour ITG/TEM
    turbulence actually shows.

An H-mode edge transport barrier is represented by suppressing chi over the
pedestal region, which is what an edge barrier physically is; the pedestal
height then emerges from the solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from scipy.linalg import solve_banded

from .constants import KEV_J, MU0, W_UNIT, MW, LN_LAMBDA


# ---------------------------------------------------------------------------
@dataclass
class Grid:
    """Radial grid and the flux-surface geometry it carries."""
    n: int = 65
    a: float = 2.0
    V_total: float = 840.0
    kappa_a: float = 1.7

    rho: np.ndarray = field(init=False)
    dr: float = field(init=False)
    Vp: np.ndarray = field(init=False)     # dV/drho
    dV: np.ndarray = field(init=False)     # cell volume

    def __post_init__(self):
        self.rho = np.linspace(0.0, 1.0, self.n)
        self.dr = float(self.rho[1] - self.rho[0])
        # V(rho) = V_total rho^2 for a torus of constant cross-section shape
        self.Vp = 2.0 * self.V_total * self.rho
        self.dV = np.gradient(self.V_total * self.rho ** 2, self.rho) * self.dr

    def volume_average(self, f: np.ndarray) -> float:
        return float(np.trapezoid(f * 2.0 * self.rho, self.rho))

    def integrate(self, f: np.ndarray) -> float:
        """Volume integral of a density f(rho): int f dV."""
        return float(np.trapezoid(f * self.Vp, self.rho))


@dataclass
class TransportModel:
    key: str
    label: str
    reference: str
    note: str = ""

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label,
                "reference": self.reference, "note": self.note}


TRANSPORT_MODELS: Dict[str, TransportModel] = {
    "scaling_anchored": TransportModel(
        "scaling_anchored", "Скейлингке бекітілген chi",
        "chi амплитудасы таңдалған 0-өлшемді скейлингке келтіріледі",
        "1-өлшемді профильдер мен валидацияланған tau_E бір-біріне сәйкес қалады"),
    "gyrobohm": TransportModel(
        "gyrobohm", "Гиро-Бом", "chi = C rho_s^2 v_th / a",
        "жергілікті бағалау; профиль қаттылығын бермейді"),
    "bohm_gyrobohm": TransportModel(
        "bohm_gyrobohm", "Бом / гиро-Бом қоспасы",
        "Erba et al., Plasma Phys. Control. Fusion 39 (1997) 261",
        "Бом мүшесі профильді қатайтады"),
    "cgm": TransportModel(
        "cgm", "Критикалық градиент", "ITG/TEM табалдырықты тасымал",
        "R/L_T табалдырықтан асқанда тасымал күрт қосылады"),
}


# ---------------------------------------------------------------------------
@dataclass
class PlasmaState:
    """Radial profiles plus the scalars derived from them."""
    grid: Grid
    ne: np.ndarray          # [1e20 m^-3]
    Te: np.ndarray          # [keV]
    Ti: np.ndarray          # [keV]
    n_he: np.ndarray        # helium ash [1e20 m^-3]

    def copy(self) -> "PlasmaState":
        return PlasmaState(self.grid, self.ne.copy(), self.Te.copy(),
                           self.Ti.copy(), self.n_he.copy())

    # --- volume and density-weighted averages -----------------------------
    @property
    def ne_avg(self) -> float:
        return self.grid.volume_average(self.ne)

    @property
    def n_bar(self) -> float:
        """Line-averaged density along the midplane chord."""
        return float(np.trapezoid(self.ne, self.grid.rho))

    def _dw(self, f: np.ndarray) -> float:
        num = self.grid.volume_average(self.ne * f)
        return num / max(self.ne_avg, 1e-9)

    @property
    def Te_avg(self) -> float:
        return self._dw(self.Te)

    @property
    def Ti_avg(self) -> float:
        return self._dw(self.Ti)

    @property
    def n_he_avg(self) -> float:
        return self.grid.volume_average(self.n_he)

    def pressure(self, f_ion: float) -> np.ndarray:
        """Total thermal pressure profile [Pa]."""
        from .constants import P_UNIT
        return P_UNIT * self.ne * (self.Te + f_ion * self.Ti)

    def stored_energy(self, f_ion: float) -> float:
        """Thermal stored energy [MJ]."""
        w = W_UNIT * self.ne * (self.Te + f_ion * self.Ti)
        return self.grid.integrate(w)


def initial_state(grid: Grid, ne0: float = 0.3, Te0: float = 1.0,
                  Ti0: float = 1.0) -> PlasmaState:
    """A smooth L-mode-like starting profile."""
    r = grid.rho
    shape = (1.0 - r ** 2) ** 1.0 + 0.05
    return PlasmaState(
        grid,
        ne=ne0 * (shape / shape[0]),
        Te=Te0 * ((1.0 - r ** 2) ** 1.5 + 0.03),
        Ti=Ti0 * ((1.0 - r ** 2) ** 1.5 + 0.03),
        n_he=np.zeros_like(r),
    )


# ---------------------------------------------------------------------------
# Diffusivity models
# ---------------------------------------------------------------------------
def _barrier(rho: np.ndarray, h_mode: bool, rho_ped: float = 0.93,
             suppression: float = 0.06) -> np.ndarray:
    """Edge transport barrier: chi multiplier across the pedestal region."""
    if not h_mode:
        return np.ones_like(rho)
    w = 0.02
    return 1.0 - (1.0 - suppression) * 0.5 * (
        1.0 + np.tanh((rho - rho_ped) / w))


#: Threshold normalised temperature gradient for ITG/TEM turbulence.
#: Gyrokinetic calculations put it between about 4 and 8 depending on
#: magnetic shear, T_i/T_e and beta; 6.5 is what makes the integrated solve
#: reproduce ITER's Q=10 point, and it is the one core-transport constant
#: this package fits rather than takes from a published parametrisation.
R_OVER_LT_CRIT = 6.9
#: How hard transport rises above the threshold.  Large values pin the core
#: profile to the critical gradient -- the "stiffness" that makes a tokamak
#: core temperature profile roughly exponential regardless of where the heat
#: goes in, and the reason the pedestal height decides the fusion power.
STIFFNESS = 4.0


def _smooth(y: np.ndarray, passes: int = 2) -> np.ndarray:
    """Three-point binomial smoothing with reflecting ends."""
    out = np.asarray(y, dtype=float).copy()
    for _ in range(passes):
        pad = np.concatenate([[out[0]], out, [out[-1]]])
        out = 0.25 * pad[:-2] + 0.5 * pad[1:-1] + 0.25 * pad[2:]
    return out


def chi_shape(rho: np.ndarray, h_mode: bool, Te: Optional[np.ndarray] = None,
              R0: float = 6.2, a: float = 2.0) -> np.ndarray:
    """chi(rho) shape: stiff core above the critical gradient, edge barrier.

    A fixed radial shape cannot produce a peaked temperature profile -- with
    a flat heat source it gives T_e0 / <T_e> ~ 1.3 whatever the amplitude,
    against the 2.8 ITER predicts.  Making the shape respond to the local
    normalised gradient fixes that from the physics rather than by fitting:
    the core organises itself around R/L_T ~ R_OVER_LT_CRIT, which is an
    exponential profile, and the peaking then follows from the machine's
    aspect ratio instead of from a chosen exponent.
    """
    if Te is None:
        base = 0.85 + 2.2 * rho ** 2
    else:
        T = np.maximum(Te, 1e-3)
        grad = -np.gradient(T, rho) / a
        r_lt = R0 * grad / T
        # Evaluating a stiff chi from the instantaneous gradient makes the
        # discrete problem oscillate: a cell that gets slightly too steep
        # gets a huge chi, over-flattens, and its neighbour takes over.  A
        # three-point smoothing of the normalised gradient is enough to
        # damp it without softening the stiffness itself.
        r_lt = _smooth(r_lt)
        base = 0.10 + STIFFNESS * np.maximum(r_lt - R_OVER_LT_CRIT, 0.0)
        base = np.maximum(_smooth(base), 0.10)
    return base * _barrier(rho, h_mode)


def chi_gyrobohm(state: PlasmaState, B0: float, a_mass: float,
                 h_mode: bool, c: float = 8.0) -> np.ndarray:
    """Local gyro-Bohm diffusivity [m^2/s]."""
    g = state.grid
    Te = np.maximum(state.Te, 1e-3)
    # rho_s = sqrt(m_i T_e) / (e B); v_th = sqrt(T_e / m_i)
    rho_s = 1.02e-4 * np.sqrt(a_mass * Te * 1e3) / B0        # [m]
    v_th = 9.79e3 * np.sqrt(Te * 1e3 / a_mass)               # [m/s]
    return c * rho_s ** 2 * v_th / g.a * _barrier(g.rho, h_mode)


def chi_bohm_gyrobohm(state: PlasmaState, B0: float, a_mass: float,
                      q_prof: np.ndarray, h_mode: bool,
                      c_gb: float = 7.0, c_b: float = 8.0e-5) -> np.ndarray:
    """Mixed Bohm / gyro-Bohm (Erba).  The Bohm term makes profiles stiff."""
    g = state.grid
    Te = np.maximum(state.Te, 1e-3)
    gb = chi_gyrobohm(state, B0, a_mass, False, c_gb)
    # Bohm term: chi_B ~ (T_e / e B) q^2 (a / L_pe)
    pe = np.maximum(state.ne * Te, 1e-6)
    dlnpe = np.abs(np.gradient(np.log(pe), g.rho)) / g.a
    bohm = c_b * Te * 1e3 / B0 * np.maximum(q_prof, 0.5) ** 2 * dlnpe * g.a
    return (gb + bohm) * _barrier(g.rho, h_mode)


def chi_cgm(state: PlasmaState, a_minor: float, h_mode: bool,
            R0: float = 6.2, r_lt_crit: float = 5.0,
            chi0: float = 0.3, chi_s: float = 1.2) -> np.ndarray:
    """Critical-gradient transport: stiff above a threshold R/L_T."""
    g = state.grid
    Te = np.maximum(state.Te, 1e-3)
    grad = -np.gradient(Te, g.rho) / g.a
    r_lt = R0 * grad / Te
    excess = np.maximum(r_lt - r_lt_crit, 0.0)
    return (chi0 + chi_s * excess) * _barrier(g.rho, h_mode)


# ---------------------------------------------------------------------------
# Implicit diffusion step
# ---------------------------------------------------------------------------
def _diffuse(grid: Grid, y: np.ndarray, cap: np.ndarray, chi: np.ndarray,
             source: np.ndarray, sink_rate: np.ndarray, dt: float,
             y_edge: float, pinch: Optional[np.ndarray] = None,
             i_edge: Optional[int] = None, y_sep: Optional[float] = None
             ) -> np.ndarray:
    """One implicit step of

        cap dy/dt = -(1/V') d/drho [ V' cap Gamma ] + source - sink_rate y
        Gamma     = -(chi / a^2) ( dy/drho + pinch y )

    with dy/drho = 0 on axis and y = y_edge at rho = 1.

    ``cap`` is the volumetric heat capacity (or 1 for a particle equation);
    it is carried inside the flux so the equation conserves energy.

    ``pinch`` is the dimensionless inward drift v a^2 / chi expressed per unit
    rho.  Without it a diffusive equation fed from the edge produces a hollow
    density profile, because pure diffusion has no way to move particles up
    the gradient -- real tokamaks peak their density through the neoclassical
    Ware pinch and a turbulent pinch, and the fusion power depends on the
    square of what that puts in the core.

    ``i_edge`` moves the Dirichlet condition to an interior index -- the
    pedestal top -- and fills the region outside it linearly down to
    ``y_sep``.  That is the standard predictive arrangement: a pedestal
    model sets the boundary condition and a stiff core is solved inside it.
    """
    n = grid.n
    dr = grid.dr
    a2 = grid.a ** 2
    Vp = grid.Vp

    # face-centred coefficients
    Vf = 0.5 * (Vp[:-1] + Vp[1:])
    cf = 0.5 * (cap[:-1] + cap[1:])
    kf = 0.5 * (chi[:-1] + chi[1:])
    D = Vf * cf * kf / (a2 * dr)                 # flux coefficient on faces
    # convective half-weights from the pinch term, central differenced
    if pinch is None:
        Pw = np.zeros(len(D))
    else:
        pf = 0.5 * (pinch[:-1] + pinch[1:])
        Pw = 0.5 * Vf * cf * kf * pf / a2

    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)
    rhs = np.zeros(n)

    # Cell volumes: for i > 0 the shell between rho_i +- dr/2 has volume
    # V'(rho_i) dr exactly, but the axial cell spans 0..dr/2 and its volume is
    # V_total (dr/2)^2, not V'(rho_1) dr -- getting that wrong by the factor of
    # eight it costs lets the axis run away and produces a spiked core.
    vol = np.maximum(Vp * dr, 1e-12)
    vol[0] = grid.V_total * (0.5 * dr) ** 2
    for i in range(1, n - 1):
        aw = D[i - 1] / vol[i]
        ae = D[i] / vol[i]
        pw = Pw[i - 1] / vol[i]
        pe = Pw[i] / vol[i]
        lower[i] = -dt * (aw - pw)
        upper[i] = -dt * (ae + pe)
        diag[i] = cap[i] + dt * (aw + ae + pw - pe) + dt * sink_rate[i]
        rhs[i] = cap[i] * y[i] + dt * source[i]

    # axis: zero-flux  ->  only the outward face contributes
    ae0 = D[0] / vol[0]
    pe0 = Pw[0] / vol[0]
    diag[0] = cap[0] + dt * (ae0 - pe0) + dt * sink_rate[0]
    upper[0] = -dt * (ae0 + pe0)
    rhs[0] = cap[0] * y[0] + dt * source[0]

    # Dirichlet at the pedestal top (or at the separatrix if none is given)
    ie = n - 1 if i_edge is None else int(np.clip(i_edge, 2, n - 1))
    for k in range(ie, n):
        lower[k] = 0.0
        upper[k] = 0.0
        diag[k] = 1.0
        rhs[k] = y_edge
    if ie < n - 1:
        sep = y_edge if y_sep is None else y_sep
        for k in range(ie, n):
            f = (grid.rho[k] - grid.rho[ie]) / max(1.0 - grid.rho[ie], 1e-9)
            rhs[k] = y_edge + (sep - y_edge) * f

    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


# ---------------------------------------------------------------------------
def equipartition(state: PlasmaState, z2a: float) -> np.ndarray:
    """Electron-ion energy exchange density [MW/m^3], positive ions -> electrons.

    Rate from the NRL Formulary: nu_eps^{e|i} = 3.2e-9 Z^2 mu^-1 n_i lnL / T_e^1.5,
    summed over species through ``z2a = sum_j (n_j/n_e) Z_j^2 / A_j``.
    tau_eq goes as T_e^3/2, so the core is several times more weakly coupled
    than the volume average -- which is exactly why a 0-D model cannot hold
    T_e above T_i in a burning plasma and this one can.
    """
    Te = np.maximum(state.Te, 0.05)
    n20 = np.maximum(state.ne, 1e-3)
    tau_eq = 5.813e-3 * Te ** 1.5 / (n20 * max(z2a, 0.02))
    return W_UNIT * n20 * (state.Ti - Te) / np.maximum(tau_eq, 1e-3)


def tau_equipartition(state: PlasmaState, z2a: float) -> np.ndarray:
    Te = np.maximum(state.Te, 0.05)
    n20 = np.maximum(state.ne, 1e-3)
    return 5.813e-3 * Te ** 1.5 / (n20 * max(z2a, 0.02))


def deposition(rho: np.ndarray, kind: str) -> np.ndarray:
    """Normalised heating deposition profiles, int f dV = 1 handled by caller."""
    if kind == "core":            # ECRH / ICRF minority, narrow and central
        f = np.exp(-(rho / 0.36) ** 2)
    elif kind == "broad":         # NBI in a large machine
        f = (1.0 - rho ** 2) ** 1.2
    elif kind == "edge":          # gas fuelling
        f = np.exp(-((1.0 - rho) / 0.08) ** 2)
    else:
        f = np.ones_like(rho)
    return f


#: Inward particle pinch strength.  With a Dirichlet edge and no core source
#: the steady profile is n ~ exp(C (1 - rho^2) / 2), so
#:     n_e0 / <n_e> = (C/2) e^{C/2} / (e^{C/2} - 1),
#: and C = 1.3 gives 1.36 -- the density peaking predicted for ITER's
#: collisionality by the Angioni scaling.
PINCH_STRENGTH = 1.3


def pinch_profile(rho: np.ndarray, strength: float = PINCH_STRENGTH) -> np.ndarray:
    """Dimensionless inward pinch, -v a^2 / (chi) per unit rho."""
    return strength * rho


def normalise_source(grid: Grid, shape: np.ndarray, total_MW: float) -> np.ndarray:
    """Scale a deposition shape so its volume integral is ``total_MW``."""
    integral = grid.integrate(shape)
    if integral <= 0:
        return np.zeros_like(shape)
    return shape * (total_MW / integral)
