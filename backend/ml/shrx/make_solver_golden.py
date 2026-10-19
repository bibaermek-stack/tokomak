"""Freeze the solver's numbers so that speed-ups cannot silently change the physics.

    python make_solver_golden.py <out.json>

Runs a few short discharges (30 s of plasma time) and stores selected scalars at t = 10, 20, 30 s.  The numbers
in tests/data/solver_golden.json were produced by the solver *before* the speed work; tests/test_solver_regression.py
checks that the current solver still reproduces them.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokamak.solver import Simulator, SolverConfig  # noqa: E402

CASES = [("iter", "scaling_anchored"), ("iter", "cgm"), ("iter", "shrx"), ("nstx", "shrx"),
         ("jet", "bohm_gyrobohm"), ("demo", "shrx")]
KEYS = ["Pfus", "Q", "Te", "Ti", "Te0", "ne", "Wth", "tauE", "betaN", "q95", "fBS", "Prad", "Tped", "qDiv"]
out = {}
for machine, tm in CASES:
    t0 = time.time()
    sim = Simulator(SolverConfig(machine=machine, transport_model=tm, disruption_enabled=False))
    log = sim.run_scenario(t_end=30.0, record_every=10.0)
    rows = [r for r in log if abs(r["t"] - 10) < 0.5 or abs(r["t"] - 20) < 0.5 or r["t"] > 29.5]
    out[f"{machine}:{tm}"] = [{"t": round(r["t"], 3), **{k: r.get(k) for k in KEYS}} for r in rows]
    print(machine, tm, f"{time.time() - t0:.1f} s", len(rows), "records")
import platform  # noqa: E402

import numpy  # noqa: E402
import scipy  # noqa: E402

# round-off differs between numpy/BLAS builds and a stiff discharge amplifies it, so bit-level goldens are
# only meaningful on the platform that produced them
out["_platform"] = {"numpy": numpy.__version__, "scipy": scipy.__version__, "python": platform.python_version(),
                    "machine": platform.machine()}
# the shrx cases depend on the shipped weights: record which ones, so the test knows when they are stale
out["_weights_sha256"] = hashlib.sha256((Path(__file__).resolve().parents[2] / "tokamak" / "shrx" / "weights.npz").read_bytes()).hexdigest()
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
