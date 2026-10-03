"""A classical model-based controller, for comparison and as a teacher.

The architecture every operating tokamak uses, reduced to its bones:

* **vertical**: PD on Z_c driving the fast VS supply's voltage directly
  -- the unstable mode is too quick for anything slower.  The two gains are
  *designed*, not tuned: the plant is linearised (:func:`linearise`), the
  measurement delay appended as a shift register, and the pair that
  minimises the worst closed-loop spectral radius over every delay the
  sensors can produce is taken from a grid;
* **radial and Ip**: plasma-level errors are turned into coil-current
  requests through the plant's static sensitivity dx/dI (force balance at
  fixed flux), and the CS ramp is trimmed to hold Ip;
* **VS offload**: the PF set slowly moves the vertical equilibrium point
  onto the reference, so the VS current returns to zero and a vertical
  reference step does not park the VS supply at its current limit;
* **coil currents**: each supply tracks its current request with
  V = V_ff + (L / tau_c) (I_req - I).

It only sees what the network sees -- the delayed, noisy, normalised
measurement -- and inverts the normalisation itself.  Its gains come from
the device model, not from tuning against the reward, which is what makes
it a fair baseline and a reasonable demonstration to clone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .env import TokamakControlEnv
from .simulator import LinearModel, linearise


@dataclass
class PIDGains:
    kp_z: Optional[float] = None     # VS volts per metre; None -> designed
    kd_z: Optional[float] = None     # VS volts per (m/s)
    kp_r: Optional[float] = None     # radial pattern, [1/s]; None -> designed
    ki_r: Optional[float] = None     # [1/s^2]
    z_filter: float = 0.5            # derivative low-pass, per step
    ki_z: float = 2.0                # VS-current offload onto the PF set [1/s]
    k_ip: float = 1.0
    tau_ip: float = 0.05
    tau_coil: float = 0.3            # PF current hold, slow on purpose


@dataclass
class Loop:
    """One SISO loop of the linear design: y = c . I, actuated along u."""
    c: np.ndarray                    # output row (n,)
    u: np.ndarray                    # voltage pattern (m,)
    kp: float = 0.0
    kd: float = 0.0
    ki: float = 0.0


def closed_loop_radius(lin: LinearModel, loops, delay: int) -> float:
    """Spectral radius of the plant with every loop closed through a delay.

    Each loop adds a shift register [y_n, ..., y_{n-d-1}] and an integral
    state, and applies V = -u (kp y_{n-d} + kd (y_{n-d} - y_{n-d-1})/dt
    + ki sum y_{n-d} dt).
    """
    A, B, dt = lin.A, lin.B, lin.dt
    n = A.shape[0]
    nb = delay + 2
    N = n + len(loops) * (nb + 1)
    Acl = np.zeros((N, N))
    Acl[:n, :n] = A
    K = np.zeros((B.shape[1], N))            # V = K state
    for j, lp in enumerate(loops):
        o = n + j * (nb + 1)
        Acl[o, :n] = lp.c @ A
        for q in range(1, nb):
            Acl[o + q, o + q - 1] = 1.0
        Acl[o + nb, o + nb] = 1.0            # integral accumulates y_{n-d}
        Acl[o + nb, o + delay] = dt
        g = np.zeros(N)
        g[o + delay] = -(lp.kp + lp.kd / dt + lp.ki * dt)
        g[o + delay + 1] = lp.kd / dt
        g[o + nb] = -lp.ki
        K += np.outer(lp.u, g)
    BK = B @ K
    Acl[:n] += BK
    for j, lp in enumerate(loops):
        o = n + j * (nb + 1)
        Acl[o] += lp.c @ BK
    return float(np.max(np.abs(np.linalg.eigvals(Acl))))


def deepest_stable(R: np.ndarray) -> tuple:
    """Grid index deepest inside the set where R is at its minimum.

    The slow circuit modes no loop touches sit at |z| ~ 1 for every
    stabilising gain, so "smallest spectral radius" picks an arbitrary
    point of the stable set, often on its edge.  The point farthest (in
    grid steps) from any failing gain is the robust choice.
    """
    from scipy.ndimage import distance_transform_edt
    ok = R < R.min() + 1e-5
    depth = distance_transform_edt(np.pad(ok, 1))[1:-1, 1:-1]
    idx = np.unravel_index(int(np.argmax(depth)), depth.shape)
    return idx, float(depth[idx])


class PIDController:
    def __init__(self, env: TokamakControlEnv, gains: Optional[PIDGains] = None):
        self.env = env
        self.g = gains or PIDGains()
        s = env.sim
        self.m = s.n_act
        # static sensitivity of the plasma position to each current
        x0, I0 = s.x.copy(), s.I.copy()
        n = s.n_el

        def F(x, I):
            _, gR, gZ = s._plasma_couplings(*x)
            return s._forces(x[0], I[-1], I[:-1], gR, gZ)
        F0 = F(x0, I0)
        h = 1e-5
        Fx = np.column_stack([(F(x0 + e, I0) - F0) / h for e in np.eye(2) * h])
        FI = np.empty((2, n + 1))
        for k in range(n + 1):
            dI = 1e-4 * max(abs(I0[k]), 1.0)
            Ik = I0.copy(); Ik[k] += dI
            FI[:, k] = (F(x0, Ik) - F0) / dI
        self.dxdI = -np.linalg.solve(Fx, FI)[:, :self.m]
        self.L = np.diag(s.M_ee)[:self.m]
        m_pe, _, _ = s._plasma_couplings(*x0)
        self.M_p_cs = m_pe[0]
        self.Lp = s._Lp(x0[0])
        # PF current patterns (CS and VS excluded) that move the plasma's
        # equilibrium point by one metre radially / vertically, least-norm
        # in ampere-turns
        pf = np.arange(1, self.m - 1)
        J = self.dxdI[:, pf] / s.turns_active[pf]
        Jp = np.linalg.pinv(J)
        self.pattern_R = np.zeros(self.m)
        self.pattern_R[pf] = (Jp @ np.array([1.0, 0.0])) / s.turns_active[pf]
        self.pattern_Z = np.zeros(self.m)
        self.pattern_Z[pf] = (Jp @ np.array([0.0, 1.0])) / s.turns_active[pf]
        # the radial loop acts in volts: L * pattern is the voltage that
        # ramps the pattern currents at one "metre per second"
        self.volt_R = self.L * self.pattern_R
        self.design_info = {}
        if None in (self.g.kp_z, self.g.kd_z, self.g.kp_r, self.g.ki_r):
            self.design()
        self.reset()

    # ------------------------------------------------------------------
    def operating_points(self):
        """Linear models across the window the controller has to cover.

        The vertical growth rate is not a constant: the CS ramp curves the
        field and a profile event moves l_i and beta_p, and a gain pair
        chosen at t = 0 alone loses the plasma later in the shot.  The
        design takes the gains that are deepest inside the stable set of
        *every* model here.
        """
        s = self.env.sim
        d = self.env.cfg.disturbance
        saved = (s.I.copy(), s.x.copy(), s.li, s.beta_p)
        # how far the CS ramps in one episode, in the direction it ramps
        frac = min(1.0, self.env.cfg.episode_steps * self.env.cfg.dt
                   / s.cfg.flux_budget_s)
        dcs = np.sign(s.feedforward()[0]) * s.cs_swing * frac
        fb = 1.0 - d.beta_p_drop[1]
        cases = [(0.0, 1.0, 1.0), (dcs, 1.0, 1.0),
                 (0.0, fb, 1.0 + d.li_change[1]),
                 (dcs, fb, 1.0 + d.li_change[0])]
        models = []
        for dI, fbp, fli in cases:
            s.I, s.x = saved[0].copy(), saved[1].copy()
            s.I[0] += dI
            s.perturb(li=saved[2] * fli, beta_p=saved[3] * fbp)
            s._settle()
            # the radial loop would bring R_c back; so does the model
            for _ in range(6):
                s.I[:self.m] -= (s.x[0] - saved[1][0]) * self.pattern_R
                s._settle()
            models.append(linearise(s))
        s.I, s.x = saved[0].copy(), saved[1].copy()
        s.perturb(li=saved[2], beta_p=saved[3])
        return models

    def design(self):
        """Gains from the linearised plant, robust over delays and models."""
        env, g = self.env, self.g
        lo, hi = env.cfg.sensors.delay_steps
        delays = range(lo, hi + 1)
        models = self.operating_points()
        self.lin = models[0]
        e_vs = np.zeros(self.m); e_vs[-1] = 1.0
        self.design_info["growth_rates"] = [lin.growth_rate for lin in models]

        def worst(loops_of):
            return max(closed_loop_radius(lin, loops_of(lin), d)
                       for lin in models for d in delays)

        # vertical first, everything else open
        if g.kp_z is None or g.kd_z is None:
            lin0 = models[0]
            sc = 1.0 / max(abs(lin0.C[1] @ lin0.B[:, -1]), 1e-30) / 1e3
            kps = sc * np.logspace(-2, 1, 19)
            kds = np.concatenate([[0.0], sc * lin0.dt * np.logspace(-1, 2, 19)])
            best = None
            for sgn in (1.0, -1.0):
                R = np.array([[worst(lambda lin: [Loop(lin.C[1], e_vs,
                                                       sgn * kp, sgn * kd)])
                               for kd in kds] for kp in kps])
                (i, j), depth = deepest_stable(R)
                if best is None or depth > best[0]:
                    best = (depth, sgn * kps[i], sgn * kds[j], R[i, j])
            g.kp_z, g.kd_z = float(best[1]), float(best[2])
            self.design_info["vertical_rho"] = float(best[3])
        # then radial, with the vertical loop closed
        if g.kp_r is None or g.ki_r is None:
            kps = np.logspace(0, 3, 13)
            kis = np.concatenate([[0.0], np.logspace(1, 5, 17)])
            R = np.array([[worst(lambda lin: [
                Loop(lin.C[1], e_vs, g.kp_z, g.kd_z),
                Loop(lin.C[0], self.volt_R, kp, 0.0, ki)])
                for ki in kis] for kp in kps])
            (i, j), _ = deepest_stable(R)
            g.kp_r, g.ki_r = float(kps[i]), float(kis[j])
            self.design_info["radial_rho"] = float(R[i, j])
        self.design_rho = self.design_info.get("vertical_rho", np.nan)

    def reset(self):
        self.z_prev = None
        self.zdot = 0.0
        self.er_int = 0.0
        self.ezeq_int = 0.0

    def measurement(self, obs):
        sens = self.env.sensors
        return obs[:sens.size] * sens.scale + sens.mu

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        env, g, s = self.env, self.g, self.env.sim
        y = self.measurement(obs)
        Rc, Zc, Ip = y[0], y[1], y[2]
        I = y[3:3 + self.m]
        dt = env.cfg.dt
        ez = Zc - env.Z_ref
        er = Rc - env.R_ref
        eip = Ip - env.Ip_ref
        if self.z_prev is not None:
            self.zdot += g.z_filter * ((Zc - self.z_prev) / dt - self.zdot)
        self.z_prev = Zc

        V_ff = s.feedforward() if env.cfg.feedforward else s.R_el[:self.m] * I
        V = V_ff.copy()
        # slow: hold the PF currents, walk the vertical equilibrium point
        # onto the reference so the VS current returns to zero
        z_eq = Zc - self.dxdI[1, -1] * I[-1]
        self.ezeq_int += (z_eq - env.Z_ref) * dt
        I_req = s.I_coils0 - g.ki_z * self.ezeq_int * self.pattern_Z
        dI = I_req - I
        dI[0] = 0.0               # the CS ramps; it is not held
        dI[-1] = 0.0              # the VS has its own loop
        # the radial loop owns the radial pattern; do not fight it
        pr = self.pattern_R / max(np.linalg.norm(self.pattern_R), 1e-30)
        dI -= pr * (pr @ dI)
        V += self.L * dI / g.tau_coil
        # radial: voltage-mode PI along the radial pattern
        self.er_int += er * dt
        V -= (g.kp_r * er + g.ki_r * self.er_int) * self.volt_R
        # Ip: extra CS ramp worth the loop voltage that closes the error,
        # Lp dIp/dt = -M_p,cs dI_cs/dt  =>  dI_cs/dt = k eip Lp / (tau M)
        V[0] += self.L[0] * (g.k_ip * eip * self.Lp / g.tau_ip) / self.M_p_cs
        # vertical: voltage-mode PD on the VS supply
        V[-1] = V_ff[-1] - (g.kp_z * ez + g.kd_z * self.zdot)
        a = (V - (s.feedforward() if env.cfg.feedforward else 0.0)) / env.V_max
        return np.clip(a, -1.0, 1.0)
