"""Run SINTEZ's ITER discharge with a candidate SHRX weight file and print the flat-top numbers.

    SHRX_WEIGHTS=/path/shrx_w128_d3_s0.npz python eval_inloop.py [machine] [t_end] [transport_model]

``transport_model`` defaults to shrx; ``qlknn_ref`` (run in the TORAX env) is the real QLKNN in the same loop.

This is the test that matters for SHRX: the network is only good if the *integrated* plasma it
produces matches the one QLKNN produces (SINTEZ + QLKNN at the same settings).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokamak.solver import Simulator, SolverConfig  # noqa: E402

machine = sys.argv[1] if len(sys.argv) > 1 else "iter"
t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
t0 = time.time()
tmodel = sys.argv[3] if len(sys.argv) > 3 else "shrx"
# disruptions off: this compares transport models, not the (unvalidated) disruption triggers of small machines
sim = Simulator(SolverConfig(machine=machine, transport_model=tmodel, disruption_enabled=False))
log = sim.run_scenario(t_end=t_end, record_every=2.0)
tail = [r for r in log if r["t"] >= t_end - 20] or log[-1:]    # average the last 20 s: the burn breathes a little
avg = lambda k: sum(r[k] for r in tail) / len(tail)
print(json.dumps({"machine": machine, "transport": tmodel, "wall_s": round(time.time() - t0, 1),
                  **{k: round(avg(k), 2) for k in ("Pfus", "Q", "Te", "Ti", "Wth", "tauE", "H_factor", "betaN")},
                  "ood": round(avg("shrx_ood"), 3), "disrupted": bool(sim.disrupted)}))
