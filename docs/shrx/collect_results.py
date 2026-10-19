"""Merge everything the SHRX evaluation produced into one compact results.json.

    python collect_results.py <workdir> [out.json]

<workdir> holds: runs/sintez_{shrx,qlknn_ref,shrx_v0}.json (run_sintez.py), machines.json (compare_machines.py),
accuracy.json (accuracy.py), local_compare.json (local_compare.py), latency_{default,1thread}.json
(bench_latency.py), bridge_validation.json (validate_bridge.py), and sintez_scaling_anchored.json +
torax_export.json from docs/torax_comparison.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "results.json"
OLD = json.load(open(HERE.parent / "torax_comparison" / "results.json"))
load = lambda name: json.load(open(W / name))

res = {"iter_ref": OLD["iter_ref"], "versions": OLD["versions"]}
KEYS = ["t", "Pfus", "Q", "Palpha", "Paux", "Prad", "Wth", "tauE", "H_factor", "betaN", "Te", "Ti", "Te0",
        "Tped", "q0", "q95"]

runs = {}
for name in ("shrx", "qlknn_ref", "shrx_v0"):
    d = load(f"runs/sintez_{name}.json")
    L = d["log"][-1]
    runs[name] = {"final": {k: L[k] for k in KEYS}, "wall_s": d["wall_s"],
                  "trace": {"t": [r["t"] for r in d["log"]], "Pfus": [r["Pfus"] for r in d["log"]]},
                  "profiles": {k: d["profiles"][k] for k in ("rho", "Te", "Ti", "q", "chi")}}
for name in ("scaling_anchored", "cgm"):
    o = OLD["sintez"][name]
    runs[name] = {"final": {k: o["final"][k] for k in KEYS}, "wall_s": o["wall_s"],
                  "trace": {"t": o["trace"]["t"], "Pfus": o["trace"]["Pfus"]},
                  "profiles": o["profiles"]}
res["sintez_runs"] = runs
res["torax_standalone"] = {"final": OLD["torax"]["main"]["final"], "profiles": OLD["torax"]["main"]["profiles"],
                           "trace_t": OLD["torax"]["main"]["trace_t"], "trace_Pfus": OLD["torax"]["main"]["trace_Pfus"],
                           "config": OLD["torax"]["main"]["config"]}

res["machines"] = load("machines.json")
res["weights_sensitivity"] = load("weights_sensitivity.json")
res["accuracy"] = load("accuracy.json")
lc = load("local_compare.json")
for key, src in (("A", "torax_state"), ("B1", "sintez_neural_surrogate"), ("B2", "sintez_scaling_anchored")):
    lc[key]["chi_v0"] = OLD["cross"][src]["chi_nn"]            # the old network, same states and grid
res["local"] = lc
res["latency"] = {"default": load("latency_default.json"), "one_thread": load("latency_1thread.json")}
res["bridge_validation"] = load("bridge_validation.json")
meta = json.load(open(HERE.parents[1] / "backend" / "tokamak" / "shrx" / "weights.json"))
res["model"] = {k: meta[k] for k in ("name", "width", "depth", "epochs", "seed", "n_parameters", "holdout_machines",
                                     "n_train", "teacher")}
res["qlknn_parameters"] = 73823
res["training_rounds"] = [
    {"round": 1, "data": "12 machines x 4 transport models (SINTEZ default, CGM, Bohm/gyro-Bohm, v0), 100 s each",
     "samples": 1648000, "result": "ITER 267 MW (Q 5.3): visible gap on states the first model itself visits"},
    {"round": 2, "data": "+ states visited when the real QLKNN (and round-1 SHRX) drives SINTEZ", "samples": 2272000,
     "result": "ITER 466 MW (128x3)"},
    {"round": 3, "data": "+ states visited by round-2 SHRX; 96x4 network, 100 epochs", "samples": 2584000,
     "result": "ITER 492 MW (shipped)"}]
OUT.write_text(json.dumps(res))
print("written", OUT, round(OUT.stat().st_size / 1024), "kB")
