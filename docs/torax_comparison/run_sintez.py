"""Run the SINTEZ reference ITER discharge with one transport model.

    python run_sintez.py <transport_model> <out.json>

transport_model: scaling_anchored | cgm | neural_surrogate | bohm_gyrobohm | gyrobohm
Needs the project's own Python environment (numpy, scipy; scikit-learn only to retrain).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import numpy as np  # noqa: E402
from tokamak.solver import Simulator, SolverConfig  # noqa: E402

tm, out = sys.argv[1], sys.argv[2]
# this report was made with the first neural surrogate, now called shrx_v0; the old name "neural_surrogate"
# nowadays resolves to the new SHRX, so keep reproducing the report with the old model
run_model = "shrx_v0" if tm == "neural_surrogate" else tm
t0 = time.time()
sim = Simulator(SolverConfig(machine="iter", transport_model=run_model))
log = sim.run_scenario(t_end=120.0, record_every=2.0)
wall = time.time() - t0
res = {
    "transport_model": tm,
    "wall_s": wall,
    "log": log,
    "profiles": sim.profiles(),
    "alarms": sim.alarms,
    "disrupted": sim.disrupted,
    "split": sim.split,
    "chi_i": [float(x) for x in sim._chi_i],
    "chi_e": [float(x) for x in sim._chi_e],
    "composition": {k: float(v) for k, v in sim.c.items() if np.isscalar(v)},
    "q95": float(sim.q95), "q0": float(sim.q0), "li": float(sim.li),
}
Path(out).write_text(json.dumps(res, default=float))
last = log[-1]
print(tm, f"wall={wall:.1f}s",
      {k: round(last[k], 3) for k in ["t", "Pfus", "Q", "Te", "Ti", "tauE", "H_factor", "betaN"]})
