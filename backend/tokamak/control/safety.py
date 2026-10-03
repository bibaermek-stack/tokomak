"""Block 4 -- deterministic safety filter (QP projection).

The network's voltages never reach the coils directly.  They are projected
onto the set of voltages the hardware and the coils can take:

    V* = argmin_V  1/2 || W (V - V_target) ||^2

    subject to   max(-V_max, V_prev - dV_max) <= V <= min(V_max, V_prev + dV_max)
                 I_min <= a + B V <= I_max

with W = diag(1/V_max), so the correction is measured in units of each
supply's rating rather than letting the CS's kilovolts swamp the VS's
hundred volts.  The first row is the supply limit together with its slew
rate (an inductive load cannot follow a voltage step without one).  The
second is the coil-current limit at the end of the prediction horizon,
linear in V because the circuit is:

    (M + h R) I_{n+1} = M I_n + h [V, 0, 0]
    =>  I_coils(n+H) = a + B V      (V held for H steps, plasma frozen)

The QP is solved by ADMM (the OSQP iteration) on the scaled problem, then
*polished*: the active set the ADMM identifies is solved exactly as an
equality-constrained least-squares problem and kept if it is primal and
dual feasible.  A feasible target returns unchanged after one check, which
is the common case and costs nothing.

The voltage box is a hard limit: the output is clipped to it whatever
happens.  The current limit can be infeasible -- a coil already past its
limit cannot be brought back inside in one step -- and then each row's
bounds are first intersected with what the box can reach, so the filter
pushes back as hard as the supply allows instead of failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FilterResult:
    V: np.ndarray
    intervened: bool
    feasible: bool
    iterations: int
    active_box: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    active_current: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    I_pred: np.ndarray = field(default_factory=lambda: np.zeros(0))


def solve_box_qp(x_t: np.ndarray, C: np.ndarray, l: np.ndarray,
                 u: np.ndarray, y0: Optional[np.ndarray] = None,
                 rho: float = 1.0, sigma: float = 1e-6, alpha: float = 1.6,
                 max_iter: int = 400, eps: float = 1e-7):
    """min 1/2 ||x - x_t||^2  s.t.  l <= C x <= u.

    Returns (x, y, iterations, polished).  ``y`` are the constraint
    multipliers (positive on an active upper bound), usable as a warm start.
    """
    n = len(x_t)
    m = C.shape[0]
    CtC = C.T @ C
    Kinv = np.linalg.inv((1.0 + sigma) * np.eye(n) + rho * CtC)
    x = x_t.copy()
    z = np.clip(C @ x, l, u)
    y = np.zeros(m) if y0 is None or len(y0) != m else y0.copy()
    it = 0
    for it in range(1, max_iter + 1):
        x = Kinv @ (sigma * x + x_t + C.T @ (rho * z - y))
        zt = C @ x
        zr = alpha * zt + (1.0 - alpha) * z
        z_new = np.clip(zr + y / rho, l, u)
        y = y + rho * (zr - z_new)
        r_pri = np.max(np.abs(zt - z_new))
        r_dua = rho * np.max(np.abs(C.T @ (z_new - z)))
        z = z_new
        if r_pri < eps and r_dua < eps:
            break
        if it % 25 == 0:
            # OSQP's adaptive step: balance the primal and dual residuals
            ratio = np.sqrt(r_pri / max(r_dua, 1e-30))
            if ratio > 5.0 or ratio < 0.2:
                new_rho = float(np.clip(rho * ratio, 1e-4, 1e4))
                rho = new_rho
                Kinv = np.linalg.inv((1.0 + sigma) * np.eye(n) + rho * CtC)
    # the multipliers name the active set; so do the bounds z sits on
    for guess in (y, _bound_sign(z, l, u)):
        xp = _polish(x_t, C, l, u, guess)
        if xp is not None:
            return xp, y, it, True
    return x, y, it, False


def _bound_sign(z, l, u):
    """+1 / -1 where z sits on its upper / lower bound."""
    scale = np.maximum(np.abs(u - l), 1.0)
    return np.where(z >= u - 1e-6 * scale, 1.0,
                    np.where(z <= l + 1e-6 * scale, -1.0, 0.0))


def _polish(x_t, C, l, u, y, tol=1e-9):
    """Exact solve on the active set guessed from the multipliers."""
    lo = y < -tol
    hi = y > tol
    act = lo | hi
    if not act.any():
        x = x_t.copy()
    else:
        CA = C[act]
        b = np.where(hi[act], u[act], l[act])
        try:
            lam = np.linalg.solve(CA @ CA.T, CA @ x_t - b)
        except np.linalg.LinAlgError:
            return None
        x = x_t - CA.T @ lam
        # KKT dual feasibility: an upper bound pushes down, a lower one up
        signs = np.where(hi[act], 1.0, -1.0)
        if np.any(signs * lam < -1e-9):
            return None
    Cx = C @ x
    scale = np.maximum(np.abs(u - l), 1.0)
    if np.any(Cx > u + 1e-7 * scale) or np.any(Cx < l - 1e-7 * scale):
        return None
    return x


class SafetyFilter:
    """QP projection of the requested coil voltages (block 4)."""

    def __init__(self, V_max: np.ndarray, I_max: np.ndarray,
                 dV_max: Optional[np.ndarray] = None, horizon: int = 1):
        self.V_max = np.asarray(V_max, float)
        self.I_max = np.asarray(I_max, float)
        # default slew: full range in 5 ms
        self.dV_max = (np.asarray(dV_max, float) if dV_max is not None
                       else 2.0 * self.V_max / 5.0)
        self.horizon = horizon
        self._y = None

    def reset(self):
        self._y = None

    def current_model(self, M: np.ndarray, R: np.ndarray, I: np.ndarray,
                      h: float, n_coils: int):
        """(a, B) with I_coils(n+H) = a + B V for a held voltage V."""
        n = len(I)
        A = M + h * np.diag(R)
        Phi = np.linalg.solve(A, M)
        Gam = h * np.linalg.solve(A, np.eye(n)[:, :n_coils])
        x = I.copy()
        Bfull = np.zeros((n, n_coils))
        for _ in range(self.horizon):
            x = Phi @ x
            Bfull = Phi @ Bfull + Gam
        return x[:n_coils], Bfull[:n_coils]

    def box(self, V_prev: Optional[np.ndarray]):
        lo, hi = -self.V_max.copy(), self.V_max.copy()
        if V_prev is not None:
            lo = np.maximum(lo, V_prev - self.dV_max)
            hi = np.minimum(hi, V_prev + self.dV_max)
            # a previous voltage outside the box (never produced here, but
            # possible from an external caller) must not empty it
            bad = lo > hi
            lo[bad], hi[bad] = -self.V_max[bad], self.V_max[bad]
        return lo, hi

    def __call__(self, V_target: np.ndarray, a: np.ndarray, B: np.ndarray,
                 V_prev: Optional[np.ndarray] = None) -> FilterResult:
        V_target = np.asarray(V_target, float)
        lo, hi = self.box(V_prev)
        I_lo = -self.I_max - a
        I_hi = self.I_max - a

        # cheap path: the request is already safe
        Vc = np.clip(V_target, lo, hi)
        Ip = a + B @ Vc
        if np.array_equal(Vc, V_target) and np.all(np.abs(Ip) <= self.I_max):
            return FilterResult(V=V_target.copy(), intervened=False,
                                feasible=True, iterations=0,
                                active_box=np.zeros(len(lo), bool),
                                active_current=np.zeros(len(lo), bool),
                                I_pred=Ip)

        # scaled variables x = V / V_max
        D = self.V_max
        Bs = B * D[None, :]
        # what the box can reach on each current row
        xlo, xhi = lo / D, hi / D
        reach_lo = np.sum(np.minimum(Bs * xlo, Bs * xhi), axis=1)
        reach_hi = np.sum(np.maximum(Bs * xlo, Bs * xhi), axis=1)
        feasible = bool(np.all(I_lo <= reach_hi) and np.all(I_hi >= reach_lo))
        I_lo_e = np.clip(I_lo, reach_lo, reach_hi)
        I_hi_e = np.clip(I_hi, reach_lo, reach_hi)
        I_lo_e = np.minimum(I_lo_e, I_hi_e)

        # unit-norm rows for conditioning
        rn = np.maximum(np.linalg.norm(Bs, axis=1), 1e-30)
        n = len(D)
        C = np.vstack([np.eye(n), Bs / rn[:, None]])
        l = np.concatenate([xlo, I_lo_e / rn])
        u = np.concatenate([xhi, I_hi_e / rn])
        x, self._y, it, polished = solve_box_qp(V_target / D, C, l, u,
                                                y0=self._y)
        V = np.clip(x * D, lo, hi)
        Ip = a + B @ V
        tol = 1e-6 * self.V_max
        return FilterResult(
            V=V, intervened=True,
            feasible=feasible and bool(np.all(np.abs(Ip) <= self.I_max * (1 + 1e-6))),
            iterations=it,
            active_box=(V <= lo + tol) | (V >= hi - tol),
            active_current=np.abs(Ip) >= self.I_max * (1 - 1e-6),
            I_pred=Ip)
