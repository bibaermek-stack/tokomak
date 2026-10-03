"""Block 2 -- sensors and delay emulator.

Between the plant and the controller sits everything that makes a real
measurement worse than the simulator's state:

* a **FIFO delay** of tau = 1-3 ms (acquisition, reconstruction, network),
  an integer number of control steps;
* **Gaussian noise** N(0, sigma^2) on every channel;
* **integrator drift** on the magnetic channels: flux loops and pick-up
  coils measure dPsi/dt and dB/dt, and the analogue integrators that turn
  them into Psi and B accumulate an offset linear in time,

      y_meas(t) = y(t - tau) + d * t + n(t),     n ~ N(0, sigma^2)

* **normalisation** into dimensionless units the network can digest,

      s = (y_meas - mu) / sigma_s

  with mu the reference (t = 0) equilibrium and sigma_s a fixed physical
  scale per channel class -- not running statistics, so the same
  measurement always maps to the same input.  Flux loops enter as
  differences from their mean: the CS ramp's common-mode flux would
  otherwise push them off scale within a few hundred milliseconds.

The raw contract vector (simulator -> network) is

    [R_c, Z_c, I_p, I_coils(m), Psi(n_loops), B(n_probes)]   [m, m, A, A, Wb, T]

as float64.  :meth:`SensorEmulator.observe` returns the normalised vector
S_t, extended with the reference errors and the previous action so the
policy sees what it is asked to track and what it last did.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .simulator import PlasmaState


def raw_vector(s: PlasmaState) -> np.ndarray:
    """The I/O contract vector, simulator -> controller (float64)."""
    return np.concatenate([[s.R_c, s.Z_c, s.Ip], s.I_coils,
                           s.psi_loops, s.B_probes]).astype(np.float64)


@dataclass
class SensorConfig:
    delay_steps: tuple = (1, 3)          # inclusive range, drawn per episode
    noise_pos_m: float = 1e-3            # R_c, Z_c reconstruction noise
    noise_ip_rel: float = 2e-3
    noise_coil_rel: float = 2e-3         # of each coil's current limit
    noise_psi_rel: float = 1e-3          # of the flux scale
    noise_b_rel: float = 5e-3            # of the field scale
    drift_rel_per_s: float = 2e-3        # integrator drift, per second
    enabled: bool = True


class SensorEmulator:
    """Delay buffer, noise, drift and normalisation (block 2)."""

    def __init__(self, n_coils: int, n_loops: int, n_probes: int,
                 ref: np.ndarray, I_max: np.ndarray, Ip0: float,
                 cfg: Optional[SensorConfig] = None,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg or SensorConfig()
        self.rng = rng or np.random.default_rng()
        self.nc, self.nl, self.npb = n_coils, n_loops, n_probes
        self.mu = np.asarray(ref, float).copy()

        psi_ref = ref[3 + n_coils:3 + n_coils + n_loops]
        b_ref = ref[3 + n_coils + n_loops:]
        self._psi = slice(3 + n_coils, 3 + n_coils + n_loops)
        psi_s = max(float(np.max(np.abs(psi_ref))), 1e-6)
        dpsi_s = max(float(np.max(np.abs(psi_ref - psi_ref.mean()))), 1e-6)
        b_s = max(float(np.max(np.abs(b_ref))), 1e-6)
        self.scale = np.concatenate([
            [0.05, 0.05, 0.05 * Ip0],              # 5 cm, 5 cm, 5 % of Ip
            np.asarray(I_max, float),
            np.full(n_loops, 0.05 * dpsi_s),
            np.full(n_probes, 0.1 * b_s)])
        c = self.cfg
        self.sigma = np.concatenate([
            [c.noise_pos_m, c.noise_pos_m, c.noise_ip_rel * Ip0],
            c.noise_coil_rel * np.asarray(I_max, float),
            np.full(n_loops, c.noise_psi_rel * psi_s),
            np.full(n_probes, c.noise_b_rel * b_s)])
        self._drift_scale = np.concatenate([
            np.zeros(3 + n_coils),
            np.full(n_loops, c.drift_rel_per_s * psi_s),
            np.full(n_probes, c.drift_rel_per_s * b_s)])
        self.reset()

    @property
    def size(self) -> int:
        return len(self.mu)

    def reset(self, delay: Optional[int] = None) -> None:
        lo, hi = self.cfg.delay_steps
        self.delay = int(delay if delay is not None
                         else self.rng.integers(lo, hi + 1))
        self.drift = self.rng.normal(0.0, 1.0, self.size) * self._drift_scale
        self.buf: deque = deque(maxlen=self.delay + 1)

    def push(self, y: np.ndarray) -> None:
        """Feed the newest true measurement into the FIFO."""
        if not self.buf:
            # an empty line holds the first sample: no garbage before t = tau
            for _ in range(self.delay):
                self.buf.append(np.asarray(y, float).copy())
        self.buf.append(np.asarray(y, float).copy())

    def delayed(self) -> np.ndarray:
        """The measurement from tau steps ago, before noise."""
        return self.buf[0]

    def measure(self, t: float) -> np.ndarray:
        """Delayed, drifting, noisy raw vector in physical units."""
        y = self.delayed().copy()
        if self.cfg.enabled:
            y += self.drift * t
            y += self.rng.normal(0.0, 1.0, self.size) * self.sigma
        return y

    def normalise(self, y: np.ndarray) -> np.ndarray:
        """S = (y - mu) / sigma_s, flux loops as differences from their mean.

        The CS ramp adds a nearly uniform flux to every loop that grows
        through the shot and carries no information about the plasma's
        position or shape; the loop-to-loop differences do.
        """
        d = y - self.mu
        d[self._psi] -= d[self._psi].mean()
        return d / self.scale
