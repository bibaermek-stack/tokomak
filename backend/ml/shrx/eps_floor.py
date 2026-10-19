"""The simulator's own sensitivity: how far does the flat-top move when the *start* is nudged by a tiny amount?

    python eps_floor.py <machine> <t_end> [transport_model] [delta ...]

Runs the scripted discharge once as is and once per ``delta`` with the initial electron temperature scaled by (1 + delta), and
prints the flat-top <T_e>, W_th and P_fus.  Where this alone changes the answer by tens of per cent, a difference of that size
between two transport models (or two networks of equal accuracy) is not evidence that one is better.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokamak.solver import Simulator, SolverConfig  # noqa: E402

machine, t_end = sys.argv[1], float(sys.argv[2])
tm = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3][0].isdigit() else "shrx"
deltas = [float(x) for x in sys.argv[3:] if x[0].isdigit() or x[0] == "-"] or [1e-6, 1e-3]
out = {}
for delta in [0.0] + deltas:
    sim = Simulator(SolverConfig(machine=machine, transport_model=tm, disruption_enabled=False))
    sim.state.Te = sim.state.Te * (1.0 + delta)
    log = sim.run_scenario(t_end=t_end, record_every=2.0)
    tail = [r for r in log if r["t"] >= t_end - 20] or log[-1:]
    avg = lambda k: sum(r[k] for r in tail) / len(tail)
    out[f"{delta:g}"] = {k: round(avg(k), 3) for k in ("Te", "Ti", "Wth", "Pfus", "betaN")}
    print(machine, tm, f"delta={delta:g}", out[f"{delta:g}"], flush=True)
print(json.dumps(out))
