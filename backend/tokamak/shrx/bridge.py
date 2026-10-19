"""Bridge between a SINTEZ plasma state and QLKNN's dimensionless input space.

QLKNN (the neural net DeepMind's TORAX uses for turbulent transport) is a function of
ten *local, dimensionless* numbers -- the normalised gradients R/L_Ti, R/L_Te, R/L_ne,
R/L_ni, the safety factor q, magnetic shear s, normalised radius x, T_i/T_e, the
collisionality log10(nu*) and n_i/n_e -- and returns *gyro-Bohm-normalised* fluxes.
It therefore knows nothing about a particular machine.  SHRX keeps that interface, so one
network serves every machine in :mod:`tokamak.machines`.

This module does the two conversions around it, in plain NumPy, following TORAX 1.4
(``qualikiz_based_transport_model.py``, ``quasilinear_transport_model.py``,
``qlknn_transport_model.py``) so SHRX and QLKNN see the same inputs and their outputs
are read the same way:

    state  --features()-->  X (10 inputs)  --network-->  q_i, q_e, Gamma_e  --to_si()-->  chi_i, chi_e

Geometry is the midplane-radius one QLKNN was trained in (circular, r_mid), with the
flux-surface elongation entering only through the r_mid(rho) map.  validate_bridge.py compares
this bridge with TORAX's own inputs and outputs on real states.
"""

from __future__ import annotations

import numpy as np

# SI constants, identical to torax._src.constants
KEV_J = 1e3 * 1.602176634e-19
M_AMU = 1.6605390666e-27
Q_E = 1.602176634e-19
M_E = 9.1093837e-31
MU0 = 4.0e-7 * np.pi
EPS0 = 8.85418782e-12
EPS = 1e-7                      # TORAX's numerical epsilon
EPSILON_NN = 1.0 / 3.0          # inverse aspect ratio of the QLKNN training set

#: order of the network inputs (the order QLKNN itself uses)
FEATURES = ("Ati", "Ate", "Ane", "Ani", "q", "smag", "x", "Ti_Te", "LogNuStar", "normni")
#: what the network returns (gyro-Bohm units): ion / electron heat flux, electron particle flux
OUTPUTS = ("qi", "qe", "pfe")
#: TORAX's default correction on the ETG electron flux
ETG_FACTOR = 1.0 / 3.0

#: Elongation profile of the flux surfaces, kappa(x) = kappa_edge (K0 + (1 - K0) x^P), x = r_mid / a.
#: Fitted (max error 0.003 in rho) to the CHEASE equilibrium TORAX ships for ITER.  It matters:
#: rho (sqrt of toroidal flux) is not r_mid / a in a shaped plasma -- near the axis r_mid is 15 % larger --
#: and every normalised gradient is taken with respect to r_mid.
KAPPA_AXIS_FRACTION = 0.74
KAPPA_POWER = 4.93

#: impurity charge assumed when turning n_i/n_e into an impurity density (Ne, as in the comparison)
Z_IMP = 10.0


def midplane_x(rho, kappa_edge: float):
    """x = r_mid / a for flux surfaces labelled by rho = sqrt(toroidal flux fraction).

    With elliptical surfaces rho = x sqrt(kappa(x) / kappa_edge)."""
    rho = np.asarray(rho, float)
    if kappa_edge <= 1.0 + 1e-9:
        return rho.copy()
    xs = np.linspace(0.0, 1.0, 2001)
    k = KAPPA_AXIS_FRACTION + (1.0 - KAPPA_AXIS_FRACTION) * xs ** KAPPA_POWER
    return np.interp(rho, xs * np.sqrt(k), xs)


_LN1E3 = float(np.log(1e3))
_LN_KEV_J = float(np.log(KEV_J))
_LN_M_E = float(np.log(M_E))
_INV_LN10 = 1.0 / float(np.log(10.0))
#: -4 ln(e) + 0.5 ln(m_e/2) + 2 ln(eps0): the constant part of ln(tau_e) in Wesson's formula
_LOG_TAU_CONST = float(-4.0 * np.log(Q_E) + 0.5 * np.log(M_E / 2.0) + 2.0 * np.log(EPS0))

_GEOM_CACHE: dict = {}


def _geometry(rho: np.ndarray, kappa_edge: float):
    """x = r_mid/a, dr_mid/drho on this grid (cached: the solver calls with the same grid every step)."""
    key = (rho.tobytes(), round(float(kappa_edge), 6))
    hit = _GEOM_CACHE.get(key)
    if hit is None:
        x = midplane_x(rho, kappa_edge)
        dxdrho = np.gradient(x, rho)
        uniform = bool(np.allclose(np.diff(rho), rho[1] - rho[0], rtol=1e-6))
        hit = (x, dxdrho, uniform)
        if len(_GEOM_CACHE) > 64:
            _GEOM_CACHE.clear()
        _GEOM_CACHE[key] = hit
    return hit


def _ddr(Y: np.ndarray, rho: np.ndarray, dr_drho: np.ndarray, uniform: bool) -> np.ndarray:
    """d/dr_mid of the rows of Y (k, N): finite differences in rho, then the chain rule.

    Same stencil as numpy.gradient (central inside, one-sided at the ends)."""
    if uniform:
        d = rho[1] - rho[0]
        G = np.empty_like(Y)
        G[:, 1:-1] = (Y[:, 2:] - Y[:, :-2]) / (2.0 * d)
        G[:, 0] = (Y[:, 1] - Y[:, 0]) / d
        G[:, -1] = (Y[:, -1] - Y[:, -2]) / d
    else:
        G = np.gradient(Y, rho, axis=1)
    return G / dr_drho


def _guard(g: np.ndarray) -> np.ndarray:
    """TORAX's guard: a vanishing normalised gradient is replaced by +eps."""
    return np.where(np.abs(g) < EPS, EPS, g)


def log_gradient(y: np.ndarray, r: np.ndarray, R0: float) -> np.ndarray:
    """R / L_y = -R (dy/dr) / y, with TORAX's guards against zero."""
    safe = np.where(np.abs(y) < EPS, EPS, y)
    return _guard(-R0 * np.gradient(y, r) / safe)


def features(rho, Te, Ti, ne20, q, R0, a, B0, zeff, normni=0.9, kappa_edge=1.0):
    """QLKNN inputs from a profile on a radial grid.

    Parameters: ``rho`` normalised radius (0..1); ``Te``, ``Ti`` [keV]; ``ne20`` electron
    density [1e20 m^-3]; ``q`` safety factor; machine ``R0``, ``a`` [m], ``B0`` [T];
    ``zeff``; ``normni`` = n_i/n_e (main-ion dilution); ``kappa_edge`` separatrix elongation
    (1 = circular; use ``machine.kappa_x``).

    Returns ``(X, aux)``: X has shape (N, 10) in :data:`FEATURES` order, ``aux`` holds the
    pieces :func:`to_si` needs.
    """
    rho = np.ascontiguousarray(rho, dtype=float)
    Te = np.maximum(np.asarray(Te, float), 1e-3)
    Ti = np.maximum(np.asarray(Ti, float), 1e-3)
    ne = np.maximum(np.asarray(ne20, float), 1e-4) * 1e20
    q_raw = np.maximum(np.asarray(q, float), 0.05)
    x_mid, dxdrho, uniform = _geometry(rho, kappa_edge)
    r = x_mid * a
    dr_drho = np.maximum(a * dxdrho, 1e-12)

    ni = normni * ne
    n_imp = max(1.0 - normni, 0.0) * ne / Z_IMP

    # all four radial derivatives in one pass: T_e, T_i, n_e and iota = 1/q
    iota = 1.0 / q_raw
    Y = np.stack([Te, Ti, ne, iota])
    G = _ddr(Y, rho, dr_drho, uniform)
    safe = np.where(np.abs(Y[:3]) < EPS, EPS, Y[:3])
    A = _guard(-R0 * G[:3] / safe)
    Ate, Ati, Ane = A[0], A[1], A[2]
    Ani = Ane                               # n_i ~ n_e: same gradient length
    Ani1 = Ane

    # magnetic shear s = (r/q) dq/dr  (= -r d(iota)/dr / iota)
    smag = -r * G[3] / iota
    smag[0] = 0.0

    # Shafranov-shift correction (van Mulders NF 2021), exactly as in TORAX
    f0 = 2.0 * KEV_J / B0 ** 2 * MU0 * q_raw ** 2
    alpha = f0 * (Te * ne * (Ate + Ane) + ni * Ti * (Ati + Ani) + n_imp * Ti * (Ati + Ani1))
    smag = smag - 0.5 * alpha
    # crude sawtooth proxy: inside q = 1 behave as if q ~ 1 with almost no shear
    low_q = q_raw < 1.0
    smag = np.where(low_q, 0.1, smag)
    q_in = np.where(low_q, 1.0, q_raw)
    smag = np.where(smag - alpha < -0.2, alpha - 0.2, smag)

    # normalised radius, rescaled to the aspect ratio QLKNN was trained at
    x = np.maximum(x_mid, EPS) * (a / R0) / EPSILON_NN

    # electron-ion collisionality: nu* = nu_ei * tau_bounce  (Wesson), kept in logs end to end
    lnTe = np.log(Te)
    log_lambda = 31.3 - 0.5 * np.log(ne) + lnTe + _LN1E3
    log_tau = (np.log(12.0 * np.pi ** 1.5 / (ne * log_lambda)) + _LOG_TAU_CONST + 1.5 * (lnTe + _LN_KEV_J))
    eps_loc = np.clip(r / R0, EPS, None)
    log_tau_b = np.log(q_raw * R0) - 1.5 * np.log(eps_loc) - 0.5 * (lnTe + _LN_KEV_J - _LN_M_E)
    log_tau_b[0] = log_tau_b[1]
    log_nustar = (np.log(zeff) - log_tau + log_tau_b) * _INV_LN10

    X = np.empty((len(rho), 10))
    X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4] = Ati, Ate, Ane, Ani, q_in
    X[:, 5], X[:, 6], X[:, 7], X[:, 8], X[:, 9] = smag, x, Ti / Te, log_nustar, normni
    return X, {"Ate": Ate, "Ati": Ati, "Ane": Ane, "Ti": Ti, "ne": ne, "R0": R0, "a": a,
               "B0": B0}


def chi_gb(Ti, B0, a, a_mass=2.5):
    """Gyro-Bohm diffusivity [m^2/s] with QLKNN's normalisation (T_i, reference length a)."""
    return (np.sqrt(a_mass * M_AMU) / (B0 * Q_E) ** 2 * (Ti * KEV_J) ** 1.5 / a)


def to_si(qi, qe, aux, a_mass=2.5, chi_min=0.05, chi_max=100.0):
    """Gyro-Bohm fluxes -> chi_i, chi_e [m^2/s]:  chi = (R/a) q / (R/L_T) * chi_GB, clipped like TORAX."""
    gb = chi_gb(aux["Ti"], aux["B0"], aux["a"], a_mass)
    scale = aux["R0"] / aux["a"]
    chi_i = scale * qi / aux["Ati"] * gb
    chi_e = scale * qe / aux["Ate"] * gb
    return (np.clip(chi_i, chi_min, chi_max), np.clip(chi_e, chi_min, chi_max))
