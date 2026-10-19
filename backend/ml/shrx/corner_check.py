"""SHRX vs the real QLKNN in the cold-ion, weak-gradient corner of QLKNN's box.   TORAX environment.

    python corner_check.py [out.json]

Electron-heated and low-density plasmas (and spherical tokamaks) live where T_i/T_e is at or below the box floor (0.25) and the
gradients are weak: QLKNN says "turbulence off" there.  The check draws random points from that corner and reports how
often SHRX disagrees, so the shipped weights can be compared on a region the main training sets under-represent.
"""
import json
import sys
from pathlib import Path

import numpy as np

from qlknn_common import clip_box, label  # noqa: E402  (imports torax first)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from tokamak import shrx  # noqa: E402

net = shrx.load()
rng = np.random.default_rng(0)
N = 20000
X = np.stack([rng.uniform(0, 2, N), rng.uniform(0, 3, N), rng.uniform(0, 3, N), rng.uniform(0, 3, N),
              rng.uniform(1, 3, N), rng.uniform(0, 4, N), rng.uniform(0.15, 0.9, N), rng.uniform(0.25, 0.4, N),
              rng.uniform(-3.5, -1.5, N), np.full(N, 0.9)], axis=1)
Xc = clip_box(X)
f = label(Xc)
g = net.predict(Xc.astype(np.float32))
out = {"n": N}
off = (np.abs(f[:, 0]) < 1e-3) & (np.abs(f[:, 1]) < 1e-3)           # QLKNN: turbulence off
out["qlknn_off_share"] = float(off.mean())
for j, n in enumerate(("q_i", "q_e")):
    out[f"shrx_{n}_on_where_qlknn_off"] = float((np.abs(g[off, j]) > 0.1).mean()) if off.any() else None
    out[f"shrx_{n}_p99_where_qlknn_off"] = float(np.percentile(np.abs(g[off, j]), 99)) if off.any() else None
    out[f"qlknn_{n}_median"] = float(np.median(np.abs(f[:, j])))
    out[f"shrx_{n}_median"] = float(np.median(np.abs(g[:, j])))
print(json.dumps(out, indent=1))
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
