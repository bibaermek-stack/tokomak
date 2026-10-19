"""Run spherical-tokamak discharges with the TGLFNN teacher in the loop and record every (input, flux) it saw.

    TGLFNN_STEP_PKL=.../step.pkl python collect_tglf_states.py <out.npz> machine:<key>:<t_end> [machine:...] [record:<csv>:<index>:<t_end> ...]

The states are *on-policy*: whatever profiles the simulator produces when the teacher drives its transport.  Every call stores
the 15 TGLF inputs, the ensemble-mean fluxes (efe, efi, pfi) and the member spreads of efe, efi.  ``machine:`` runs the built-in
scripted discharge; ``record:`` runs one database record (``db5/sintez_runner.ShotSimulator``) at its measured parameters.
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "db5"))
from tokamak.solver import Simulator, SolverConfig  # noqa: E402

out_path, jobs = sys.argv[1], sys.argv[2:]
rows, tags = [], []
for job in jobs:
    kind, *rest = job.split(":")
    t0 = time.time()
    if kind == "machine":
        key, t_end = rest[0], float(rest[1])
        sim = Simulator(SolverConfig(machine=key, transport_model="tglfnn_ref", disruption_enabled=False))
        sim.tglf_log = []
        sim.run_scenario(t_end=t_end, record_every=5.0)
        name = key
    else:
        import sintez_runner as sr
        csv, idx, t_end = rest[0], int(rest[1]), float(rest[2])
        rec = SimpleNamespace(**pd.read_csv(csv).iloc[idx].to_dict())
        sim = sr.ShotSimulator(rec, "tglfnn_ref")
        sim.tglf_log = []
        while sim.t < sr.T_START + t_end:
            sim._scenario_actuators()
            sim.step()
        name = f"{rec.TOK}-{int(rec.SHOT)}"
    log = sim.tglf_log[::2]
    rows.append(np.concatenate(log))
    tags += [name] * sum(len(a) for a in log)
    print(f"{job}: {sum(len(a) for a in log)} rows, {time.time() - t0:.0f} s, Te0 {sim.state.Te[0]:.2f} Ti0 {sim.state.Ti[0]:.2f} keV", flush=True)
A = np.concatenate(rows)
np.savez_compressed(out_path, X=A[:, :15], Y=A[:, 15:18], std=A[:, 18:20], source=np.array(tags))
print(out_path, A.shape)
