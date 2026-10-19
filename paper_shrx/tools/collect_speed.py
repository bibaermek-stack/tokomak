"""Wall-clock time of the same discharge before and after the speed-up, on an idle machine.   (project Python)

    python tools/collect_speed.py <old_backend_dir> <new_backend_dir> <out.json> [repeats]

``old_backend_dir`` is a checkout of the commit before the speed-up work (git worktree of 15635f6), ``new_backend_dir`` the current backend.
Each case is run ``repeats`` times alternately with one thread (OMP_NUM_THREADS=1) through backend/ml/shrx/eval_inloop.py; the medians and the printed
flat-top numbers are stored so that identical results can be checked as well.
"""
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

old, new, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 3
cases = [("iter", "120", "shrx"), ("jet", "120", "shrx"), ("demo", "120", "shrx"), ("iter", "120", "scaling_anchored")]
res = {"cases": {}}
for mach, t_end, tm in cases:
    key = f"{mach}:{tm}"
    res["cases"][key] = {"old": [], "new": [], "numbers": {}}
    for rep in range(repeats):
        for tag, root in (("old", old), ("new", new)):
            p = subprocess.run([sys.executable, str(root / "ml/shrx/eval_inloop.py"), mach, t_end, tm], capture_output=True, text=True,
                               env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, cwd=str(root))
            row = json.loads(p.stdout.strip().splitlines()[-1])
            res["cases"][key][tag].append(row["wall_s"])
            res["cases"][key]["numbers"][tag] = {k: row[k] for k in ("Pfus", "Q", "Te", "Wth", "betaN") if k in row}
        print(key, rep, res["cases"][key]["old"][-1], res["cases"][key]["new"][-1], flush=True)
    c = res["cases"][key]
    c["median_old"], c["median_new"] = statistics.median(c["old"]), statistics.median(c["new"])
    c["speedup"] = c["median_old"] / c["median_new"]
    c["identical"] = c["numbers"]["old"] == c["numbers"]["new"]
Path(out).write_text(json.dumps(res, indent=1))
