"""Merges the experiment outputs and training histories into results.json.

    python collect_results.py out.json res_1.json res_2.json ... \
        --train PPO-0=A.json --train PPO-DR=C.json --train PPO-DR-AC=B.json

Later files override earlier ones on a key clash.
"""
import json
import sys

import numpy as np

args = sys.argv[1:]
out = args[0]
files, train = [], {}
i = 1
while i < len(args):
    if args[i] == "--train":
        n, p = args[i + 1].split("=", 1)
        train[n] = p
        i += 2
    else:
        files.append(args[i]); i += 1
res = {}
for f in files:
    part = json.load(open(f))
    sc = part.pop("seeds_check", None)
    res.update(part)
    if sc:                      # merge per policy and condition, never replace
        for pol, conds in sc.items():
            res.setdefault("seeds_check", {}).setdefault(pol, {}).update(conds)
tr = {}
for name, path in train.items():
    h = json.load(open(path))
    ppo = h["ppo"]
    best = max(ppo, key=lambda r: r["return"])
    late = [r["explained_var"] for r in ppo if r["steps"] >= 20000]
    tr[name] = {
        "best_return": best["return"], "best_step": int(best["steps"]),
        "best_pos_mm": best["rms_position_error_mm"],
        "final_return": ppo[-1]["return"], "final_pos_mm": ppo[-1]["rms_position_error_mm"],
        "ev_min_late": float(min(late)), "ev_mean_late": float(np.mean(late)), "ev_max": float(max(r["explained_var"] for r in ppo)),
        "expert_return": h["expert"]["return"], "bc_return": h["bc"]["return"],
        "n_updates": len(ppo),
    }
res["training"] = tr
json.dump(res, open(out, "w"), default=float, indent=0)
print(sorted(res))
