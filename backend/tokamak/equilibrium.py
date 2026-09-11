"""Fixed-boundary Grad-Shafranov equilibrium solver.

Solves

    Delta* psi = -mu0 R^2 dp/dpsi - F dF/dpsi = -mu0 R j_phi

on a rectangular (R, Z) grid with psi = 0 imposed on the prescribed plasma
boundary, using the standard parametrised current profile

    j_phi(R, psi_n) = lambda [ beta0 R/R0 + (1 - beta0) R0/R ]
                              (1 - psi_n^alpha_m)^alpha_n

and Picard iteration: solve the linear system, renormalise lambda so that
the enclosed current is Ip, update psi_n, repeat.

This replaces the analytic q95 formula with a q profile that comes out of an
actual equilibrium, which is what makes q0, the internal inductance, the
Shafranov shift and the flux-surface geometry available at all.

Conventions
    psi          poloidal flux per radian [Wb/rad], psi = psi_axis on axis
                 and 0 on the boundary
    B_R = -(1/R) dpsi/dZ,   B_Z = (1/R) dpsi/dR
    q    = (1/2pi) |dPhi_tor / dpsi|
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import factorized
from scipy.interpolate import UnivariateSpline

from .constants import MU0, P_UNIT
from .geometry import boundary


@dataclass
class GSProfiles:
    """Shape parameters of the parametrised current profile.

    ``beta0`` splits the current between the pressure-driven term (R/R0) and
    the poloidal-current term (R0/R); it maps monotonically onto beta_p.
    ``alpha_m`` and ``alpha_n`` set how peaked the current is, and therefore
    q0 and the internal inductance.
    """
    beta0: float = 0.55
    alpha_m: float = 2.0
    alpha_n: float = 1.6


@dataclass
class Equilibrium:
    """Result of a Grad-Shafranov solve."""
    R: np.ndarray                 # grid, 1-D [m]
    Z: np.ndarray                 # grid, 1-D [m]
    psi: np.ndarray               # (nZ, nR) [Wb/rad]
    psi_n: np.ndarray             # normalised, 0 on axis -> 1 on boundary
    mask: np.ndarray              # True inside the plasma
    psi_axis: float
    R_axis: float
    Z_axis: float
    j_phi: np.ndarray             # (nZ, nR) [A/m^2]

    rho: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    V_of_rho: np.ndarray = field(default_factory=lambda: np.zeros(0))
    psi_n_grid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    p_avg: float = 0.0

    q0: float = 0.0
    q95: float = 0.0
    li3: float = 0.0
    beta_p: float = 0.0
    beta_t: float = 0.0
    beta_n: float = 0.0
    shafranov: float = 0.0
    V: float = 0.0
    Ip: float = 0.0
    converged: bool = False
    iterations: int = 0
    residual: float = 0.0
    profiles: Optional[GSProfiles] = None

    def summary(self) -> dict:
        return {"q0": self.q0, "q95": self.q95, "li3": self.li3,
                "beta_p": self.beta_p, "beta_t": self.beta_t,
                "beta_n": self.beta_n, "shafranov": self.shafranov,
                "V": self.V, "R_axis": self.R_axis, "psi_axis": self.psi_axis,
                "converged": self.converged, "iterations": self.iterations,
                "residual": self.residual}


def _inside(Rg: np.ndarray, Zg: np.ndarray,
            Rb: np.ndarray, Zb: np.ndarray) -> np.ndarray:
    """Even-odd ray casting; the boundary curve is closed and convex enough."""
    inside = np.zeros(Rg.shape, dtype=bool)
    n = len(Rb) - 1
    for k in range(n):
        r1, z1 = Rb[k], Zb[k]
        r2, z2 = Rb[k + 1], Zb[k + 1]
        if z1 == z2:
            continue
        cond = ((z1 > Zg) != (z2 > Zg))
        with np.errstate(divide="ignore", invalid="ignore"):
            r_cross = r1 + (Zg - z1) * (r2 - r1) / (z2 - z1)
        inside ^= cond & (Rg < r_cross)
    return inside


def _build_operator(R: np.ndarray, Z: np.ndarray, mask: np.ndarray):
    """Sparse Delta* with Dirichlet rows outside the plasma.

    Delta* = d2/dR2 - (1/R) d/dR + d2/dZ2
    """
    nZ, nR = mask.shape
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    N = nR * nZ
    A = lil_matrix((N, N))
    idx = lambda i, j: i * nR + j          # i over Z, j over R

    for i in range(nZ):
        for j in range(nR):
            k = idx(i, j)
            if not mask[i, j] or i == 0 or i == nZ - 1 or j == 0 or j == nR - 1:
                A[k, k] = 1.0              # psi = 0 outside / on the frame
                continue
            r = R[j]
            A[k, idx(i, j - 1)] = 1.0 / dR ** 2 + 1.0 / (2.0 * r * dR)
            A[k, idx(i, j + 1)] = 1.0 / dR ** 2 - 1.0 / (2.0 * r * dR)
            A[k, idx(i - 1, j)] = 1.0 / dZ ** 2
            A[k, idx(i + 1, j)] = 1.0 / dZ ** 2
            A[k, k] = -2.0 / dR ** 2 - 2.0 / dZ ** 2
    return csr_matrix(A)


def solve(R0: float, a: float, kappa: float, delta: float,
          Ip: float, B0: float,
          profiles: Optional[GSProfiles] = None,
          nR: int = 97, nZ: int = 145,
          tol: float = 1e-5, max_iter: int = 60) -> Equilibrium:
    """Solve the fixed-boundary equilibrium.

    Parameters
    ----------
    Ip : plasma current [MA]
    B0 : vacuum toroidal field at R0 [T]
    """
    prof = profiles or GSProfiles()
    Ip_A = Ip * 1e6

    Rb, Zb = boundary(R0, a, kappa, delta, 512)
    pad = 1.12
    R = np.linspace(R0 - pad * a, R0 + pad * a, nR)
    Z = np.linspace(-pad * kappa * a, pad * kappa * a, nZ)
    Rg, Zg = np.meshgrid(R, Z)
    mask = _inside(Rg, Zg, Rb, Zb)
    if mask.sum() < 50:
        raise RuntimeError("plasma mask is empty -- check the grid extent")

    A = _build_operator(R, Z, mask)
    lu = factorized(A.tocsc())

    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    dA = dR * dZ

    # geometric part of the current profile, fixed through the iteration
    geom = prof.beta0 * Rg / R0 + (1.0 - prof.beta0) * R0 / Rg

    # initial guess: a paraboloid peaked at the geometric centre
    rho2 = ((Rg - R0) / a) ** 2 + (Zg / (kappa * a)) ** 2
    psi = np.where(mask, np.maximum(1.0 - rho2, 0.0), 0.0)
    psi_axis = max(psi.max(), 1e-9)

    converged = False
    residual = np.inf
    it = 0
    j_phi = np.zeros_like(psi)

    for it in range(1, max_iter + 1):
        psi_n = np.clip(1.0 - psi / psi_axis, 0.0, 1.0)
        g = np.where(mask, (1.0 - psi_n ** prof.alpha_m) ** prof.alpha_n, 0.0)

        shape = geom * g
        norm = float(np.sum(shape[mask]) * dA)
        lam = Ip_A / max(norm, 1e-30)
        j_phi = lam * shape

        rhs = np.where(mask, -MU0 * Rg * j_phi, 0.0)
        psi_new = lu(rhs.ravel()).reshape(psi.shape)
        psi_new = np.where(mask, psi_new, 0.0)

        new_axis = float(psi_new.max())
        residual = float(np.max(np.abs(psi_new - psi)) / max(new_axis, 1e-12))
        # under-relax: the source depends on psi through psi_n
        psi = 0.5 * psi + 0.5 * psi_new
        psi_axis = max(float(psi.max()), 1e-12)
        if residual < tol:
            converged = True
            break

    psi_n = np.clip(1.0 - psi / psi_axis, 0.0, 1.0)
    k_ax = int(np.argmax(psi))
    i_ax, j_ax = divmod(k_ax, len(R))
    R_axis, Z_axis = float(R[j_ax]), float(Z[i_ax])

    eq = Equilibrium(R=R, Z=Z, psi=psi, psi_n=psi_n, mask=mask,
                     psi_axis=psi_axis, R_axis=R_axis, Z_axis=Z_axis,
                     j_phi=j_phi, Ip=Ip, converged=converged,
                     iterations=it, residual=residual)
    _derive(eq, R0, a, B0, Ip, prof, lam)
    return eq


def _derive(eq: Equilibrium, R0: float, a: float, B0: float, Ip: float,
            prof: GSProfiles, lam: float) -> None:
    """Flux-surface quantities, q profile, beta and internal inductance."""
    R, Z, psi, mask = eq.R, eq.Z, eq.psi, eq.mask
    Rg, _ = np.meshgrid(R, Z)
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    dA = dR * dZ
    psi_a = eq.psi_axis

    # --- p(psi) and F(psi) from the same ansatz that produced j_phi --------
    # dp/dpsi   = lam beta0 g / R0
    # d(F^2/2)/dpsi = mu0 lam (1 - beta0) R0 g
    s = np.linspace(0.0, 1.0, 257)                     # psi_n grid
    g_s = (1.0 - s ** prof.alpha_m) ** prof.alpha_n
    # integral of g from s to 1, times psi_a (since dpsi = -psi_a dpsi_n)
    tail = np.concatenate([[0.0], np.cumsum(0.5 * (g_s[1:] + g_s[:-1]) * np.diff(s))])
    int_g = (tail[-1] - tail) * psi_a                  # int_{s}^{1} g dpsi_n * psi_a

    p_s = lam * prof.beta0 / R0 * int_g                # [Pa], zero at the edge
    F_vac = B0 * R0
    F2_s = F_vac ** 2 + 2.0 * MU0 * lam * (1.0 - prof.beta0) * R0 * int_g
    F_s = np.sqrt(np.maximum(F2_s, 1e-12))

    # --- bin grid cells into psi_n shells ---------------------------------
    psin_flat = eq.psi_n[mask]
    Rflat = Rg[mask]
    F_cell = np.interp(psin_flat, s, F_s)

    # Cumulative toroidal flux and volume as functions of psi_n.  These are
    # monotone, so they are far better conditioned than per-shell bins; q is
    # then the derivative of a smoothing spline through Phi(psi_n) rather
    # than a finite difference of a staircase, which is what made q95 jump
    # around with grid resolution.
    nb = 200
    edges = np.linspace(0.0, 1.0, nb + 1)
    which = np.clip(np.digitize(psin_flat, edges) - 1, 0, nb - 1)

    dV_cell = 2.0 * np.pi * Rflat * dA
    dPhi_cell = (F_cell / Rflat) * dA                  # B_phi dA_pol

    dV = np.bincount(which, weights=dV_cell, minlength=nb)
    dPhi = np.bincount(which, weights=dPhi_cell, minlength=nb)

    V_cum = np.concatenate([[0.0], np.cumsum(dV)])
    Phi_cum = np.concatenate([[0.0], np.cumsum(dPhi)])

    spline = UnivariateSpline(edges, Phi_cum, k=3,
                              s=max(1e-6, 1e-4 * Phi_cum[-1] ** 2 * nb))
    # q = (1/2pi) |dPhi/dpsi|,  dpsi = -psi_a d(psi_n)
    centres = np.linspace(0.005, 0.995, 200)
    q_shell = spline.derivative()(centres) / (2.0 * np.pi * psi_a)
    q_shell = np.abs(q_shell)

    # q on axis: the innermost shells enclose only a handful of cells, so
    # extrapolate the well-resolved 0.05-0.30 range inwards rather than
    # trusting the spline at psi_n -> 0.
    fit = (centres > 0.05) & (centres < 0.30)
    q0 = float(np.polyval(np.polyfit(centres[fit] ** 2, q_shell[fit], 1), 0.0))
    rho = np.sqrt(np.clip(spline(centres) / max(spline(1.0), 1e-30), 0.0, 1.0))
    V_prof = np.interp(centres, edges, V_cum)

    eq.rho = rho
    eq.q = q_shell
    eq.V_of_rho = V_prof
    eq.V = float(V_cum[-1])
    eq.q0 = q0
    eq.q95 = float(np.interp(0.95, centres, q_shell))
    eq.psi_n_grid = centres

    # --- poloidal field, internal inductance ------------------------------
    dpsi_dZ, dpsi_dR = np.gradient(psi, Z, R)
    B_R = -dpsi_dZ / Rg
    B_Z = dpsi_dR / Rg
    Bp2 = B_R ** 2 + B_Z ** 2
    W_pol = float(np.sum(Bp2[mask] * 2.0 * np.pi * Rg[mask] * dA))
    eq.li3 = float(2.0 * W_pol / (MU0 ** 2 * (Ip * 1e6) ** 2 * R0))

    # --- beta -------------------------------------------------------------
    p_cell = np.interp(psin_flat, s, p_s)
    p_avg = float(np.sum(p_cell * dV_cell) / max(eq.V, 1e-30))
    B_pa = MU0 * Ip * 1e6 / _poloidal_perimeter(eq)
    eq.beta_p = float(2.0 * MU0 * p_avg / B_pa ** 2)
    eq.beta_t = float(2.0 * MU0 * p_avg / B0 ** 2)
    eq.beta_n = float(100.0 * eq.beta_t * a * B0 / Ip)
    eq.shafranov = float(eq.R_axis - R0)
    eq.p_avg = p_avg


def _poloidal_perimeter(eq: Equilibrium) -> float:
    """Perimeter of the outermost closed flux surface, from the mask."""
    # perimeter ~ boundary of the mask; use the marching-free estimate
    # L = sum over boundary cells of the cell edge length facing outside
    m = eq.mask
    dR = eq.R[1] - eq.R[0]
    dZ = eq.Z[1] - eq.Z[0]
    edge_r = np.sum(m[:, :-1] != m[:, 1:]) * dZ
    edge_z = np.sum(m[:-1, :] != m[1:, :]) * dR
    # a staircase boundary overestimates a smooth curve by ~4/pi
    return float((edge_r + edge_z) * np.pi / 4.0)


def solve_for_machine(m, Ip: Optional[float] = None,
                      profiles: Optional[GSProfiles] = None,
                      **kw) -> Equilibrium:
    """Convenience wrapper: solve the equilibrium for a registered machine."""
    return solve(m.R0, m.a, m.kappa_x, m.delta_x,
                 Ip if Ip is not None else m.Ip, m.B0,
                 profiles=profiles, **kw)


def solve_matched(R0: float, a: float, kappa: float, delta: float,
                  Ip: float, B0: float,
                  p_avg: Optional[float] = None,
                  q0: Optional[float] = None,
                  max_outer: int = 12, tol: float = 2e-3,
                  **kw) -> Equilibrium:
    """Solve for the profile parameters that match a given plasma state.

    ``beta0`` maps monotonically onto the volume-averaged pressure and
    ``alpha_n`` onto q on axis, and the two are close to orthogonal, so a
    pair of secant iterations converges in a handful of equilibrium solves.
    This is what couples the equilibrium to the transport solution: the
    pressure comes from the energy balance, not from a guessed profile.

    Parameters
    ----------
    p_avg : target volume-averaged total pressure [Pa]
    q0    : target safety factor on axis (1.0 for a sawtoothing discharge)
    """
    prof = GSProfiles(**{k: v for k, v in kw.pop("profiles", {}).items()}) \
        if isinstance(kw.get("profiles"), dict) else GSProfiles()
    eq = solve(R0, a, kappa, delta, Ip, B0, profiles=prof, **kw)

    for _ in range(max_outer):
        done = True
        if p_avg is not None and eq.p_avg > 0:
            err = p_avg / eq.p_avg
            if abs(err - 1.0) > tol:
                done = False
                prof.beta0 = float(np.clip(prof.beta0 * err, 0.02, 0.98))
        if q0 is not None and eq.q0 > 0:
            err = eq.q0 / q0
            if abs(err - 1.0) > tol:
                done = False
                # q0 falls as the current peaks up, so alpha_n moves with err
                prof.alpha_n = float(np.clip(prof.alpha_n * err ** 1.6, 0.3, 8.0))
        if done:
            break
        eq = solve(R0, a, kappa, delta, Ip, B0, profiles=prof, **kw)

    eq.profiles = prof
    return eq
