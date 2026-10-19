"""Stack independently trained SHRX members into one ensemble file (NumPy only).

    python make_ensemble.py <out.npz> <member1.npz> <member2.npz> ...  [--meta key=value ...]

Writes <out.npz> (weights stacked along a leading member axis, see tokamak/shrx/ensemble.py) and <out>.json with the
member names and any extra metadata (e.g. the calibrated ``unc_threshold``).
"""
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
rest = sys.argv[2:]
meta_kv = {}
if "--meta" in rest:
    i = rest.index("--meta")
    for kv in rest[i + 1:]:
        k, v = kv.split("=")
        meta_kv[k] = float(v)
    rest = rest[:i]
members = [Path(p) for p in rest]
zs = [np.load(p) for p in members]
n = int(zs[0]["n_layers"])
assert all(int(z["n_layers"]) == n for z in zs) and all(z["W0"].shape == zs[0]["W0"].shape for z in zs), \
    "members must share one architecture"
arrays = {"n_layers": np.array(n)}
for key in ("mu", "sd", "scale", "lo", "hi"):
    arrays[key] = np.stack([z[key] for z in zs])
arrays["lo"], arrays["hi"] = zs[0]["lo"], zs[0]["hi"]            # the QLKNN box is shared
for i in range(n):
    arrays[f"W{i}"] = np.stack([z[f"W{i}"] for z in zs])
    arrays[f"b{i}"] = np.stack([z[f"b{i}"] for z in zs])
np.savez_compressed(out, **arrays)
meta = {"members": [p.stem for p in members], "n_members": len(members), **meta_kv}
for p in members:
    jp = p.with_suffix(".json")
    if jp.is_file():
        m = json.loads(jp.read_text())
        meta.setdefault("member_info", []).append({k: m.get(k) for k in ("name", "seed", "epochs", "n_parameters", "n_train")})
out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
print("wrote", out, f"{out.stat().st_size / 1024:.0f} kB,", len(members), "members")
