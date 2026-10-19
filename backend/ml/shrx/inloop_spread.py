"""How much does the integrated plasma depend on *which* equally accurate SHRX network is used, and does the ensemble help?

    python inloop_spread.py <out.json> <t_end> <machine,machine,...> <member1.npz> [member2.npz ...] [--ensemble]

For every machine the scripted discharge is run once with each member network (SHRX_WEIGHTS) and once with the ensemble mean
(``shrx_ens``) via ``eval_inloop.py``; the flat-top <T_e>, W_th and P_fus are collected.  The spread of the members is the
noise a single network brings; the ensemble run should sit inside it, near its centre.  Compare with the intrinsic noise floor
of the simulator (``eps_floor.py``): differences below that cannot be told apart.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
out_path, t_end, machines = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
rest = sys.argv[4:]
with_ens = "--ensemble" in rest
members = [m for m in rest if m != "--ensemble"]
res = {}
for mach in machines:
    res[mach] = {"members": {}, "ensemble": None}
    jobs = [(Path(m).stem, {"SHRX_WEIGHTS": m}, "shrx") for m in members]
    if with_ens:
        jobs.append(("ensemble", {}, "shrx_ens"))
    for name, env, tm in jobs:
        p = subprocess.run([sys.executable, str(HERE / "eval_inloop.py"), mach, t_end, tm], capture_output=True, text=True,
                           env={**os.environ, "OMP_NUM_THREADS": "1", **env})
        try:
            row = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            row = {"error": (p.stderr or p.stdout)[-300:]}
        if name == "ensemble":
            res[mach]["ensemble"] = row
        else:
            res[mach]["members"][name] = row
        print(mach, name, {k: row.get(k) for k in ("Te", "Wth", "Pfus", "ood")}, flush=True)
        Path(out_path).write_text(json.dumps(res, indent=1))
