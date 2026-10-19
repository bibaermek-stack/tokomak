"""SHRX vs the real QLKNN, in the same SINTEZ discharge, on every machine.

    python compare_machines.py <out.json> [t_end]

Runs the scripted discharge of each machine twice -- transport_model "qlknn_ref" (the teacher) and
"shrx" -- and records the end-of-pulse numbers (mean of the last 20 s).  Run in the TORAX env (needs
JAX for the teacher); set OMP_NUM_THREADS=1.  Two machines (nstx, demo) were held out of training.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
out = Path(sys.argv[1])
t_end = sys.argv[2] if len(sys.argv) > 2 else "100"
machines = ["iter", "demo", "sparc", "jet", "diiid", "east", "kstar", "t15md", "ktm", "nstx", "nstxu", "mastu"]


def run(job):
    m, tm = job
    r = subprocess.run([sys.executable, str(HERE / "eval_inloop.py"), m, t_end, tm], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    return (m, tm, json.loads(line[-1]) if line else {"error": r.stderr[-300:]})


jobs = [(m, tm) for m in machines for tm in ("qlknn_ref", "shrx")]
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(run, jobs))
data = {}
for m, tm, d in res:
    data.setdefault(m, {})[tm] = d
out.write_text(json.dumps(data, indent=1))
for m in machines:
    a, b = data[m].get("qlknn_ref", {}), data[m].get("shrx", {})
    print(f"{m:7s} QLKNN Pfus {a.get('Pfus')!s:>7} Te {a.get('Te')!s:>6} | SHRX Pfus {b.get('Pfus')!s:>7} Te {b.get('Te')!s:>6}")
