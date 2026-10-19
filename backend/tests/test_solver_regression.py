"""The solver must keep producing the numbers frozen in tests/data/solver_golden.json.

The golden values come from the solver as it was before the speed-up work (ml/shrx/make_solver_golden.py), and the
optimised solver reproduces them bit for bit.  If a change is *meant* to alter the physics, regenerate the file in the same
commit and say so; a pure optimisation must not move a single bit.  The primitives that were rewritten for speed are
also checked directly, on any platform, in test_speed_equivalence.py.
The cases that use SHRX are skipped when the shipped weights are not the ones the goldens were made with.
"""
import hashlib
import json
import platform
from pathlib import Path

import numpy
import pytest
import scipy

from tokamak.solver import Simulator, SolverConfig

DATA = Path(__file__).parent / "data" / "solver_golden.json"
GOLDEN = json.loads(DATA.read_text())
WEIGHTS = Path(__file__).resolve().parents[1] / "tokamak" / "shrx" / "weights.npz"
SHRX_SAME = WEIGHTS.is_file() and hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == GOLDEN["_weights_sha256"]
CASES = sorted(k for k in GOLDEN if not k.startswith("_"))
HERE = {"numpy": numpy.__version__, "scipy": scipy.__version__, "python": platform.python_version(),
        "machine": platform.machine()}
SAME_PLATFORM = HERE == GOLDEN["_platform"]


@pytest.mark.skipif(not SAME_PLATFORM, reason="bit-level goldens: only meaningful on the platform that made them "
                    "(round-off differs between numpy/BLAS builds and a stiff discharge amplifies it)")
@pytest.mark.parametrize("case", CASES)
def test_solver_matches_golden(case):
    machine, tm = case.split(":")
    if tm == "shrx" and not SHRX_SAME:
        pytest.skip("shipped SHRX weights differ from the ones the golden values were made with")
    sim = Simulator(SolverConfig(machine=machine, transport_model=tm, disruption_enabled=False))
    log = sim.run_scenario(t_end=30.0, record_every=10.0)
    got = [r for r in log if abs(r["t"] - 10) < 0.5 or abs(r["t"] - 20) < 0.5 or r["t"] > 29.5]
    ref = GOLDEN[case]
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert abs(g["t"] - r["t"]) < 1e-6
        for k, v in r.items():
            if k == "t" or v is None:
                continue
            tol = 1e-12 * max(abs(v), 1.0)       # the speed-ups are bit-for-bit: any real deviation is a bug
            assert abs(g[k] - v) <= tol, f"{case} t={r['t']}: {k} = {g[k]!r}, golden {v!r}"
