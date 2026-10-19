"""Record the plasma states a SINTEZ discharge actually visits, as QLKNN inputs.

    python collect_states.py <machine> <transport_model> <out.npz> [t_end] [--no-eq]

Every second of plasma time one snapshot of the radial profiles is turned into the ten QLKNN
inputs by ``tokamak.shrx.bridge.features``.  SHRX is trained where the solver really goes, not on
an invented distribution.  Project environment (NumPy only).
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokamak.shrx import bridge  # noqa: E402
from tokamak.solver import Simulator, SolverConfig  # noqa: E402

machine, tmodel, out = sys.argv[1:4]
t_end = float(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else 90.0
use_eq = "--no-eq" not in sys.argv

sim = Simulator(SolverConfig(machine=machine, transport_model=tmodel, equilibrium=use_eq,
                             eq_interval=8.0, disruption_enabled=False, n_rho=65))
m = sim.m
X, T0, meta = [], [], []
next_rec, t0 = 1.0, time.time()
while sim.t < t_end:
    sim._scenario_actuators()
    sim.step()
    if sim.t >= next_rec:
        next_rec += 1.0
        s, g = sim.state, sim.grid
        x, aux = bridge.features(g.rho, s.Te, s.Ti, s.ne, sim.q_prof, m.R0, m.a, m.B0,
                                 sim.c["z_eff"], sim.c["f_dt"], kappa_edge=m.kappa_x)
        X.append(x)
        meta.append([sim.t, float(sim.h_mode), m.R0, m.a, m.B0, m.kappa_x, sim.c["z_eff"], sim.c["f_dt"]])
np.savez_compressed(out, X=np.concatenate(X).astype(np.float32),
                    meta=np.repeat(np.array(meta, np.float32), len(sim.grid.rho), axis=0),
                    rho=np.tile(sim.grid.rho, len(X)).astype(np.float32))
print(f"{machine:8s} {tmodel:17s} {len(X)} snapshots, {time.time() - t0:.0f} s wall, "
      f"Te_avg end {sim.state.Te_avg:.2f} keV, P_fus {sim.P_fus:.0f} MW")
