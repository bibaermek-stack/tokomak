"""Closed-loop environment: blocks 1, 2 and 4 behind a step() interface.

One call of :meth:`TokamakControlEnv.step` is one tick of the lock-step
protocol at f = 1 kHz:

    A_t (block 3)
      -> V_target = V_ff + V_max * A_t
      -> V*_t     = QP projection (block 4)
      -> plasma advanced by dt = 1 ms with V*_t (block 1)
      -> reward, early-termination check
      -> measurement through delay / noise / drift / normalisation (block 2)
      -> S_{t+1}

V_ff is the nominal voltage that holds the equilibrium currents and
supplies the plasma's loop voltage (:meth:`TokamakSimulator.feedforward`):
the network outputs corrections around it in units of each supply's
rating.  ``feedforward=False`` drops it, and the network then has to learn
the resistive hold as well.

Reward (all terms dimensionless, weights in :class:`RewardConfig`)

    r_t = w_pos  exp(-(e_pos / s_pos)^2)
        + w_shp  exp(-(e_shp / s_shp)^2)
        + w_ip   exp(-((Ip - Ip_ref) / (s_ip Ip_ref))^2)
        - w_smo  ||A_t - A_{t-1}||^2 / m
        - w_qp   ||W (V*_t - V_target)||^2 / m
        - P_term [episode terminated]

    e_pos = || (R_c, Z_c) - (R_ref, Z_ref) ||
    e_shp = RMS isoflux boundary error after removing rigid translation

The last two terms keep the network away from what the safety filter has
to fix (and from chattering voltages that heat the coils and excite the
vessel); the terminal penalty prices the events listed below.

Early termination (``terminated``, with the penalty):

* wall contact:   d_min <= 2 cm
* current quench: |dIp/dt| > 1e7 A/s
* VDE:            |Z_c| > 0.30 m, or the force balance has no root left
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .safety import SafetyFilter
from .sensors import SensorConfig, SensorEmulator, raw_vector
from .simulator import DeviceConfig, LossOfEquilibrium, TokamakSimulator


@dataclass
class RewardConfig:
    w_pos: float = 0.4
    w_shape: float = 0.3
    w_ip: float = 0.3
    w_smooth: float = 0.1
    w_qp: float = 0.1
    s_pos_m: float = 0.01
    s_shape_m: float = 0.01
    s_ip_rel: float = 0.01
    terminal_penalty: float = 50.0


@dataclass
class TerminationConfig:
    d_min_m: float = 0.02
    dIp_dt_max: float = 1e7
    z_max_m: float = 0.30


@dataclass
class DisturbanceConfig:
    """Per-episode randomisation, so the policy is not trained on one shot."""
    vs_kick_rel: float = 0.02            # of the VS current limit, 1 sigma
    # one profile event per episode: an ELM / sawtooth crash / minor
    # disruption moves beta_p and l_i, and with them the force balance
    event_prob: float = 0.7
    beta_p_drop: tuple = (0.0, 0.10)     # fractional drop
    li_change: tuple = (-0.03, 0.03)     # fractional change
    event_ramp_steps: int = 10           # spread over 10 ms
    # reference steps the controller has to follow
    ref_step_prob: float = 0.5
    dR_ref_m: float = 0.01
    dZ_ref_m: float = 0.02
    enabled: bool = True


@dataclass
class EnvConfig:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    dt: float = 1e-3
    n_sub: int = 2
    episode_steps: int = 500
    feedforward: bool = True
    safety_horizon: int = 5


class TokamakControlEnv:
    """The plant as the controller sees it."""

    def __init__(self, cfg: Optional[EnvConfig] = None, seed: int = 0,
                 sim: Optional[TokamakSimulator] = None):
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(seed)
        c = self.cfg
        self.sim = sim or TokamakSimulator(c.device, dt=c.dt, n_sub=c.n_sub)
        s = self.sim
        self.m = s.n_act
        self.V_max = s.V_max
        self.ref0 = raw_vector(s.state())
        self.sensors = SensorEmulator(
            s.n_act, len(s.loop_R), len(s.probe_R), self.ref0, s.I_max,
            s.Ip0, c.sensors, rng=self.rng)
        self.safety = SafetyFilter(s.V_max, s.I_max, horizon=c.safety_horizon)
        self.gaps_ref = s.state().gaps.copy()
        # outward unit normals of the target boundary at the control points
        p = s.plasma
        nR = s.ctrl_R - p.R0
        nZ = (s.ctrl_Z - p.Z0) / max(p.kappa, 1e-3) ** 2
        nn = np.hypot(nR, nZ)
        self._normals = np.stack([nR / nn, nZ / nn], axis=1)
        self.obs_dim = self.sensors.size + 3 + self.m
        self.act_dim = self.m
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, delay: Optional[int] = None,
              disturb: Optional[bool] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.sensors.rng = self.rng
        d = self.cfg.disturbance
        disturb = d.enabled if disturb is None else disturb
        kick = (self.rng.normal() * d.vs_kick_rel * self.sim.I_max[-1]
                if disturb else 0.0)
        self.sim.reset(vs_kick=kick)
        self.safety.reset()
        self.sensors.reset(delay=delay)
        self.k = 0
        self.R_ref, self.Z_ref = self.sim.ref_R, self.sim.ref_Z
        self.Ip_ref = self.sim.Ip0
        n = self.cfg.episode_steps
        self.event = None
        self.ref_step = None
        if disturb and self.rng.random() < d.event_prob:
            self.event = (int(self.rng.integers(n // 5, 4 * n // 5)),
                          1.0 - self.rng.uniform(*d.beta_p_drop),
                          1.0 + self.rng.uniform(*d.li_change))
        if disturb and self.rng.random() < d.ref_step_prob:
            self.ref_step = (int(self.rng.integers(n // 5, 4 * n // 5)),
                             self.rng.uniform(-1, 1) * d.dR_ref_m,
                             self.rng.uniform(-1, 1) * d.dZ_ref_m)
        self.a_prev = np.zeros(self.m)
        self.V_prev = self.sim.V_last.copy()
        self.state = self.sim.state()
        self.sensors.push(raw_vector(self.state))
        return self._observe()

    def _observe(self) -> np.ndarray:
        y = self.sensors.measure(self.sim.t)
        s = self.sensors.normalise(y)
        sc = self.sensors.scale
        err = np.array([(y[0] - self.R_ref) / sc[0], (y[1] - self.Z_ref) / sc[1],
                        (y[2] - self.Ip_ref) / sc[2]])
        obs = np.concatenate([s, err, self.a_prev])
        self.last_measurement = y
        return np.clip(obs, -10.0, 10.0)

    # ------------------------------------------------------------------
    def target_voltage(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        V = self.V_max * a
        if self.cfg.feedforward:
            V = V + self.sim.feedforward()
        return V

    def shape_error(self, gaps: np.ndarray) -> float:
        """RMS boundary error after removing the best-fit translation."""
        e = gaps - self.gaps_ref
        d, *_ = np.linalg.lstsq(self._normals, e, rcond=None)
        r = e - self._normals @ d
        return float(np.sqrt(np.mean(r * r)))

    def step(self, action: np.ndarray):
        c = self.cfg
        s = self.sim
        a = np.clip(np.asarray(action, float), -1.0, 1.0)

        # scheduled disturbances: a profile event ramps beta_p and l_i to
        # their new values (an instant step makes the massless plasma jump,
        # and flux conservation turns the jump into a current spike)
        if self.event is not None:
            k0, fb, fl = self.event
            nr = max(self.cfg.disturbance.event_ramp_steps, 1)
            if k0 <= self.k < k0 + nr:
                frac = (self.k - k0 + 1) / nr
                p = s.plasma
                s.perturb(beta_p=p.beta_p * (1 + (fb - 1) * frac),
                          li=p.li * (1 + (fl - 1) * frac))
        if self.ref_step is not None and self.k == self.ref_step[0]:
            self.R_ref = s.ref_R + self.ref_step[1]
            self.Z_ref = s.ref_Z + self.ref_step[2]

        # block 4: QP projection of the requested voltages.  The filter
        # predicts coil currents from the supplies' own current readings,
        # which are fast and local and taken as exact here -- unlike the
        # plasma diagnostics the network sees through block 2
        V_t = self.target_voltage(a)
        M = s._M_now()
        a_lin, B_lin = self.safety.current_model(M, s.R_el, s.I, c.dt, self.m)
        filt = self.safety(V_t, a_lin, B_lin, V_prev=self.V_prev)
        V = filt.V

        # block 1: advance the plant
        reason = None
        try:
            st = s.step(V)
        except LossOfEquilibrium:
            st = None
            reason = "VDE: тік тепе-теңдік жоғалды"
        self.k += 1

        t = c.termination
        if st is not None:
            if st.d_min <= t.d_min_m:
                reason = "қабырғаға соғылу"
            elif abs(st.dIp_dt) > t.dIp_dt_max:
                reason = "ток өшуі (current quench)"
            elif abs(st.Z_c) > t.z_max_m:
                reason = "VDE: |Z_c| > шек"
            self.state = st
        terminated = reason is not None
        truncated = (not terminated) and self.k >= c.episode_steps

        # reward
        rc = c.reward
        st = self.state
        e_pos = float(np.hypot(st.R_c - self.R_ref, st.Z_c - self.Z_ref))
        e_shp = self.shape_error(st.gaps)
        e_ip = (st.Ip - self.Ip_ref) / (rc.s_ip_rel * self.Ip_ref)
        dqp = float(np.mean(((V - V_t) / self.V_max) ** 2))
        dsm = float(np.mean((a - self.a_prev) ** 2))
        terms = {
            "pos": rc.w_pos * np.exp(-(e_pos / rc.s_pos_m) ** 2),
            "shape": rc.w_shape * np.exp(-(e_shp / rc.s_shape_m) ** 2),
            "ip": rc.w_ip * np.exp(-e_ip ** 2),
            "smooth": -rc.w_smooth * dsm,
            "qp": -rc.w_qp * dqp,
            "terminal": -rc.terminal_penalty if terminated else 0.0,
        }
        reward = float(sum(terms.values()))

        self.a_prev = a
        self.V_prev = V
        if st is not None and not terminated:
            self.sensors.push(raw_vector(st))
        obs = self._observe()
        info = {
            "t": s.t, "reason": reason, "terms": terms,
            "e_pos": e_pos, "e_shape": e_shp, "V": V, "V_target": V_t,
            "qp_intervened": filt.intervened, "qp_feasible": filt.feasible,
            "R_c": st.R_c, "Z_c": st.Z_c, "Ip": st.Ip, "d_min": st.d_min,
            "R_ref": self.R_ref, "Z_ref": self.Z_ref,
        }
        return obs, reward, terminated, truncated, info
