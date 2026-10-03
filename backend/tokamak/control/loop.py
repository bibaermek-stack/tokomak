"""Lock-step co-simulation: the four blocks run as one closed loop.

Step lock-step protocol (f = 1 kHz, dt = 1 ms for both the integrator and
the network's inference, strictly synchronised):

1. **Initialisation** -- the simulator computes the t = 0 equilibrium
   (Grad-Shafranov + coil currents that hold it); its measurement becomes
   S_0.
2. **Request-response cycle**, every tick:
   * the interface reads S_t from the simulator,
   * the delay buffer returns the sample from t - tau and noise / drift are
     added,
   * the network infers A_t = tanh(W_L h + b_L),
   * the safety filter projects V_target onto the admissible set and writes
     V*_t into the simulator's input buffer,
   * the simulator integrates the circuit and force-balance equations over
     dt (t <- t + dt).

:class:`CoSimulation` runs that cycle for any controller -- the trained
network, the classical baseline, or a callable of your own -- and records
the traces.  Only the I/O contract crosses the boundary:

    simulator -> controller : [R_c, Z_c, I_p, I_coils, Psi, B]   float64
    controller -> simulator : [V_1, ..., V_m]  [V]               float64
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .env import EnvConfig, TokamakControlEnv
from .policy import ActorCritic
from .sensors import raw_vector

Controller = Callable[[np.ndarray], np.ndarray]


def contract_spec(env: TokamakControlEnv) -> dict:
    """The I/O data contract, with names, units and sizes."""
    s = env.sim
    m, nl, npb = s.n_act, len(s.loop_R), len(s.probe_R)
    return {
        "dt_s": env.cfg.dt,
        "sim_to_controller": {
            "layout": ["R_c", "Z_c", "I_p"]
                      + [f"I_{n}" for n in s.coil_names]
                      + [f"Psi_{k + 1}" for k in range(nl)]
                      + [f"B_{k + 1}" for k in range(npb)],
            "units": ["m", "m", "A"] + ["A"] * m + ["Wb"] * nl + ["T"] * npb,
            "dtype": "float64",
        },
        "controller_to_sim": {
            "layout": [f"V_{n}" for n in s.coil_names],
            "units": ["V"] * m,
            "dtype": "float64",
            "V_max": [float(v) for v in s.V_max],
        },
        "observation_dim": env.obs_dim,
        "action_dim": env.act_dim,
    }


@dataclass
class Trace:
    t: list = field(default_factory=list)
    R_c: list = field(default_factory=list)
    Z_c: list = field(default_factory=list)
    Ip: list = field(default_factory=list)
    d_min: list = field(default_factory=list)
    e_pos: list = field(default_factory=list)
    e_shape: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    V: list = field(default_factory=list)
    V_target: list = field(default_factory=list)
    qp: list = field(default_factory=list)
    R_ref: list = field(default_factory=list)
    Z_ref: list = field(default_factory=list)
    reason: Optional[str] = None
    wall_time_s: float = 0.0

    def summary(self) -> dict:
        n = len(self.t)
        if n == 0:
            return {"steps": 0}
        ep = np.array(self.e_pos)
        return {
            "steps": n,
            "duration_s": float(self.t[-1]),
            "terminated": self.reason is not None,
            "reason": self.reason,
            "return": float(np.sum(self.reward)),
            "reward_per_step": float(np.mean(self.reward)),
            "rms_position_error_mm": float(np.sqrt(np.mean(ep ** 2)) * 1e3),
            "max_abs_Z_mm": float(np.max(np.abs(self.Z_c)) * 1e3),
            "rms_shape_error_mm": float(np.sqrt(np.mean(np.square(self.e_shape))) * 1e3),
            "min_wall_gap_mm": float(np.min(self.d_min) * 1e3),
            "qp_intervention_frac": float(np.mean(self.qp)),
            "ms_per_step": float(self.wall_time_s / n * 1e3),
        }

    def to_dict(self, every: int = 1) -> dict:
        out = {}
        for k in ("t", "R_c", "Z_c", "Ip", "d_min", "e_pos", "e_shape",
                  "reward", "R_ref", "Z_ref"):
            out[k] = [float(v) for v in getattr(self, k)[::every]]
        out["V"] = [list(map(float, v)) for v in self.V[::every]]
        out["qp"] = [bool(v) for v in self.qp[::every]]
        out["summary"] = self.summary()
        return out


class CoSimulation:
    """Runs the closed loop for one controller."""

    def __init__(self, env: Optional[TokamakControlEnv] = None,
                 cfg: Optional[EnvConfig] = None, seed: int = 0):
        self.env = env or TokamakControlEnv(cfg, seed=seed)

    @property
    def contract(self) -> dict:
        return contract_spec(self.env)

    def run(self, controller: Controller, steps: Optional[int] = None,
            seed: Optional[int] = None, delay: Optional[int] = None,
            disturb: Optional[bool] = None) -> Trace:
        env = self.env
        obs = env.reset(seed=seed, delay=delay, disturb=disturb)
        if hasattr(controller, "reset"):
            controller.reset()
        steps = steps or env.cfg.episode_steps
        tr = Trace()
        t0 = time.perf_counter()
        for _ in range(steps):
            a = controller(obs)
            obs, r, term, trunc, info = env.step(a)
            tr.t.append(info["t"]); tr.R_c.append(info["R_c"])
            tr.Z_c.append(info["Z_c"]); tr.Ip.append(info["Ip"])
            tr.d_min.append(info["d_min"]); tr.e_pos.append(info["e_pos"])
            tr.e_shape.append(info["e_shape"]); tr.reward.append(r)
            tr.V.append(info["V"]); tr.V_target.append(info["V_target"])
            tr.qp.append(info["qp_intervened"])
            tr.R_ref.append(info["R_ref"]); tr.Z_ref.append(info["Z_ref"])
            if term:
                tr.reason = info["reason"]
                break
        tr.wall_time_s = time.perf_counter() - t0
        return tr


class NeuralController:
    """Block 3 at deployment: the deterministic actor mean."""

    def __init__(self, ac: ActorCritic):
        self.ac = ac

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.ac.mean_action(obs)


class ZeroController:
    """Feedforward only -- the open loop, for comparison."""

    def __init__(self, m: int):
        self.m = m

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.m)


def measurement_vector(env: TokamakControlEnv) -> np.ndarray:
    """The current contract vector (true, undelayed) -- for external
    co-simulation partners that bring their own sensor model."""
    return raw_vector(env.sim.state())
