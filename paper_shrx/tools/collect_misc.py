"""Freeze small experiment outputs into data/*.json: in-loop network spread, simulator noise floor, TGLFNN teacher runs, speed-up timings.

    python tools/collect_misc.py <data_dir>      (the scratchpad folder the experiments wrote to)
"""
import ast
import json
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[1] / "data"
res = {"iter": {}, "spread": json.loads((D / "inloop/spread.json").read_text()), "floor": {}}
for name in ["shipped", "member0", "member1", "member2", "member3", "member4", "ensemble"]:
    res["iter"][name] = json.loads((D / f"inloop/{name}.json").read_text())
for line in (D / "inloop/floor.log").read_text().splitlines():
    m = re.match(r"(\w+) shrx delta=([\d.e+-]+) (\{.*\})", line)
    if m:
        res["floor"].setdefault(m.group(1), {})[m.group(2)] = ast.literal_eval(m.group(3))
res["teacher_runs"] = {k: json.loads((D / f"tglf/ref_{k}.json").read_text()) for k in ("ktm", "nstx", "mastu") if (D / f"tglf/ref_{k}.json").stat().st_size}
(OUT / "inloop.json").write_text(json.dumps(res, indent=1))
print({k: list(v) if isinstance(v, dict) else v for k, v in res.items()})
