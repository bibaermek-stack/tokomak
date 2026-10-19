"""Sensitivity of the integrated simulation to a tiny change of the initial state.   (project Python; backend on the path)

    python tools/collect_chaos.py <out.json> <t_end> machine [machine ...]

For each machine the scripted discharge is run once as is and once per relative perturbation of the initial electron temperature
(1e-12, 1e-9, 1e-6); the maximum over the radius of |T_e - T_e^ref| / max_rho T_e^ref (the central-temperature scale: the cold edge would otherwise dominate a
pointwise ratio) is stored against time.  Round-off is ~1e-16, so an
optimisation that changes the order of floating-point operations is such a perturbation.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from tokamak.solver import Simulator, SolverConfig  # noqa: E402

out_path, t_end, machines = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
res = {}
for mach in machines:
    runs = {}
    for delta in (0.0, 1e-12, 1e-9, 1e-6):
        sim = Simulator(SolverConfig(machine=mach, transport_model="shrx", disruption_enabled=False))
        sim.state.Te = sim.state.Te * (1.0 + delta)
        t, Te = [], []
        while sim.t < t_end:
            sim._scenario_actuators()
            sim.step()
            if round(sim.t / 0.5) != round((sim.t - sim.cfg.dt) / 0.5):
                t.append(float(sim.t))
                Te.append(sim.state.Te.copy())
        runs[delta] = (np.array(t), np.array(Te))
        print(mach, delta, flush=True)
    t, ref = runs[0.0]
    res[mach] = {"t": t.tolist(), "deviation": {f"{d:g}": (np.abs(Te - ref).max(axis=1) / np.maximum(ref.max(axis=1), 1e-3)).tolist()
                                                for d, (_, Te) in runs.items() if d > 0}}
Path(out_path).write_text(json.dumps(res))
