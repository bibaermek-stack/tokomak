"""Experiments for the paper: one entry point, one JSON file of raw numbers.

    python -m tokamak.control.experiments --out ../paper/results.json \
        --policy A=tokamak/control/policy.npz --policy B=... --policy C=...

Every experiment evaluates controllers on the same seeds, so episodes are
paired: seed s fixes the disturbance sequence, the sensor noise, the delay
and (when the plant is drawn) the plant, whichever controller runs.  Paired
tests therefore compare controllers, not luck.

Experiments
    main      nominal plant and the training distribution of plants
    ood       plants outside the training distribution (wall, plasma,
              supply gain, and a combination)
    delay     sensor delay 0..5 ms (training covers 1..3)
    safety    the same policies with the current-limit row of the safety
              filter switched off
    growth    how fast an instability a PD loop can hold at a given sample
              rate and delay: the linear limit, no learning involved
    cost      inference time and size of the networks
    traces    one disturbed episode per controller, for the figures
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, Optional

import numpy as np

from .baseline import PIDController, design_vertical
from .env import EnvConfig, PlantRandomisation, TokamakControlEnv
from .loop import CoSimulation, NeuralController, ZeroController
from .policy import ActorCritic
from .simulator import DeviceConfig, TokamakSimulator, linearise

SEEDS = list(range(5000, 5030))


def _episode(env, ctrl, seed, plant=None, delay=None) -> dict:
    tr = CoSimulation(env).run(ctrl, seed=seed, plant=plant, delay=delay)
    s = tr.summary()
    return {k: s[k] for k in (
        "steps", "terminated", "reason", "return", "rms_position_error_mm",
        "rms_shape_error_mm", "max_abs_Z_mm", "min_wall_gap_mm",
        "qp_intervention_frac", "max_I_over", "frac_steps_over_limit")}


def run_set(env, ctrl, seeds=SEEDS, plant=None, delay=None) -> dict:
    """Episodes plus the aggregates the paper quotes."""
    eps = [_episode(env, ctrl, sd, plant, delay) for sd in seeds]
    pos = np.array([e["rms_position_error_mm"] for e in eps])
    ret = np.array([e["return"] for e in eps])
    surv = np.array([not e["terminated"] for e in eps])
    return {
        "survival": float(surv.mean()),
        "n": len(eps),
        "return_mean": float(ret.mean()),
        "return_se": float(ret.std(ddof=1) / np.sqrt(len(ret))),
        # error statistics over the surviving episodes only: a lost plasma
        # has no meaningful RMS error, and survival is reported on its own
        "pos_mm_mean": float(pos[surv].mean()) if surv.any() else None,
        "pos_mm_se": (float(pos[surv].std(ddof=1) / np.sqrt(surv.sum()))
                      if surv.sum() > 1 else None),
        "shape_mm_mean": float(np.mean([e["rms_shape_error_mm"]
                                        for e, s in zip(eps, surv) if s]))
        if surv.any() else None,
        "qp_frac": float(np.mean([e["qp_intervention_frac"] for e in eps])),
        "over_limit_frac": float(np.mean([e["frac_steps_over_limit"]
                                          for e in eps])),
        "max_I_over": float(np.max([e["max_I_over"] for e in eps])),
        "episodes": eps,
    }


def paired(a: dict, b: dict, key="return") -> dict:
    """Wilcoxon signed-rank test of controller a against b, same seeds."""
    from scipy.stats import wilcoxon
    x = np.array([e[key] for e in a["episodes"]])
    y = np.array([e[key] for e in b["episodes"]])
    d = x - y
    if np.allclose(d, 0):
        return {"median_diff": 0.0, "p": 1.0}
    return {"median_diff": float(np.median(d)),
            "p": float(wilcoxon(x, y).pvalue)}


def load_controllers(env_nom, policies: Dict[str, str]) -> Dict[str, object]:
    ctrls = {"open_loop": ZeroController(env_nom.act_dim),
             "PID": PIDController(env_nom)}
    for name, path in policies.items():
        ctrls[name] = NeuralController(ActorCritic.load(path))
    return ctrls


# ---------------------------------------------------------------------------
OOD_PLANTS = {
    "wall x0.5": {"wall_res": 0.5},
    "wall x2": {"wall_res": 2.0},
    "plasma x0.25": {"plasma_res": 0.25},
    "plasma x4": {"plasma_res": 4.0},
    "gain 0.7": {"gain": 0.7},
    "gain 1.3": {"gain": 1.3},
    "all adverse": {"wall_res": 0.5, "plasma_res": 4.0, "gain": 1.3},
}


def exp_main(envs, ctrls, seeds):
    out = {}
    for name, c in ctrls.items():
        out[name] = {
            "nominal": run_set(envs["nom"], c, seeds, plant={}),
            "randomised": run_set(envs["dr"], c, seeds),
        }
    return out


def exp_ood(envs, ctrls, seeds):
    return {name: {cond: run_set(envs["nom"], c, seeds, plant=p)
                   for cond, p in OOD_PLANTS.items()}
            for name, c in ctrls.items() if name != "open_loop"}


def exp_delay(envs, ctrls, seeds, delays=range(0, 6)):
    return {name: {str(d): run_set(envs["nom"], c, seeds, plant={}, delay=d)
                   for d in delays}
            for name, c in ctrls.items() if name != "open_loop"}


def exp_safety(envs, ctrls, seeds):
    out = {}
    for name, c in ctrls.items():
        if name in ("open_loop", "PID"):
            continue
        out[name] = {}
        for cond, plant in (("nominal", {}), ("gain 1.3", {"gain": 1.3}),
                            ("all adverse", OOD_PLANTS["all adverse"])):
            out[name][cond] = {
                "filter_on": run_set(envs["nom"], c, seeds, plant=plant),
                "filter_off": run_set(envs["nofilter"], c, seeds, plant=plant),
            }
    return out


def exp_growth(thickness_mm=(8, 4, 3, 2, 1.5, 1, 0.75, 0.5, 0.35, 0.25),
               configs=((1e-3, 2), (2.5e-4, 2), (1e-4, 2)),
               delay_ms=2.0, stable_below=1.0005):
    """Linear controllability limit of the vertical PD loop.

    For each wall thickness (which sets the open-loop growth rate gamma) and
    each (dt, n_sub), linearise the plant and ask :func:`design_vertical`
    for the best PD pair at a fixed *physical* delay.  "Stabilisable" means
    the best closed-loop spectral radius is below ``stable_below`` (the slow
    circuit modes sit at exactly 1, hence the small margin).
    """
    rows = []
    for th in thickness_mm:
        dev = DeviceConfig(wall_thickness_m=th * 1e-3)
        for dt, nsub in configs:
            sim = TokamakSimulator(dev, dt=dt, n_sub=nsub)
            lin = linearise(sim)
            d = int(round(delay_ms * 1e-3 / dt))
            kp, kd, rho = design_vertical([lin], [d], n_kp=25, n_kd=25,
                                       refine=True)
            rows.append({"wall_mm": th, "gamma": sim.vertical_growth_rate(),
                         "dt_ms": dt * 1e3, "delay_steps": d,
                         "rho": rho, "stabilisable": bool(rho < stable_below)})
    return rows


def exp_device(envs):
    """The plant itself: what the controller is asked to hold."""
    env = envs["nom"]
    s = env.sim
    m = s.machine
    gs = s.plasma.gs_summary
    return {
        "machine": m.label, "R0": m.R0, "a": m.a, "kappa": m.kappa_x,
        "delta": m.delta_x, "B0": m.B0, "Ip_MA": m.Ip,
        "growth_rate": s.vertical_growth_rate(),
        "linear_growth_rate": linearise(s).growth_rate,
        "coils": s.coil_names, "n_vessel": len(s.cond.names) - s.n_act,
        "n_filaments": int(len(s.plasma.w)), "n_loops": int(len(s.loop_R)),
        "n_probes": int(len(s.probe_R)), "obs_dim": env.obs_dim,
        "act_dim": env.act_dim, "priv_dim": TokamakControlEnv(EnvConfig(
            privileged=True)).priv_dim,
        "V_max": [float(v) for v in s.V_max],
        "I_max": [float(v) for v in s.I_max],
        "li": float(s.plasma.li), "beta_p": float(s.plasma.beta_p),
        "q95": float(gs["q95"]), "q0": float(gs["q0"]),
        "wall_gap_min_mm": float(s.wall_clearance() * 1e3),
        "dt_ms": env.cfg.dt * 1e3,
    }


def exp_seeds(envs, ctrls, seeds, only=None):
    """Does the ranking survive retraining?  Every variant, every training
    seed (names ``VARIANT#seed``), on five conditions (``only`` filters)."""
    conds = {
        "nominal": (envs["nom"], {}, None),
        "randomised": (envs["dr"], None, None),
        "gain 1.3": (envs["nom"], {"gain": 1.3}, None),
        "wall x2": (envs["nom"], {"wall_res": 2.0}, None),
        "delay 4 ms": (envs["nom"], {}, 4),
    }
    if only:
        conds = {k: v for k, v in conds.items() if k in only}
    out = {}
    for name, c in ctrls.items():
        if "#" not in name:
            continue
        out[name] = {cond: run_set(env, c, seeds, plant=plant, delay=dl)
                     for cond, (env, plant, dl) in conds.items()}
    return out


def exp_draws(envs, seeds):
    """The plant, delay and open-loop growth rate each seed draws.

    Lets a failure be tied to the physics: an episode whose draw is beyond
    what any controller in the comparison can hold says nothing about the
    differences between controllers.
    """
    env = envs["dr"]
    out = []
    for sd in seeds:
        env.reset(seed=sd)
        d = env.plant_draw
        out.append({"seed": sd, "wall_res": d["wall_res"],
                    "plasma_res": d["plasma_res"], "gain": d["gain"],
                    "delay": env.sensors.delay,
                    "gamma": env.sim.vertical_growth_rate()})
    return out


def exp_cost(policies: Dict[str, str], n=2000):
    out = {}
    for name, path in policies.items():
        ac = ActorCritic.load(path)
        o = np.random.default_rng(0).normal(size=ac.obs_dim)
        for _ in range(50):
            ac.mean_action(o)
        t = time.perf_counter()
        for _ in range(n):
            ac.mean_action(o)
        us = (time.perf_counter() - t) / n * 1e6
        sizes = [ac.obs_dim, *ac.hidden, ac.act_dim]
        params = sum(i * o_ + o_ for i, o_ in zip(sizes[:-1], sizes[1:]))
        flops = sum(2 * i * o_ for i, o_ in zip(sizes[:-1], sizes[1:]))
        out[name] = {"actor_params": int(params), "actor_flops": int(flops),
                     "inference_us": float(us), "actor_in": ac.obs_dim,
                     "critic_in": ac.obs_dim + ac.priv_dim}
    return out


def exp_traces(envs, ctrls, seed=5003, plant=None):
    out = {}
    for name, c in ctrls.items():
        tr = CoSimulation(envs["nom"]).run(c, seed=seed, plant=plant or {})
        out[name] = tr.to_dict(every=2)
        out[name]["delay_ms"] = envs["nom"].sensors.delay
    return out


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--policy", action="append", default=[],
                    help="NAME=path.npz (repeatable)")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--seed-conds", default="",
                    help="comma list of conditions for seeds_check")
    ap.add_argument("--only", default="device,draws,main,ood,delay,safety,growth,cost,traces")
    a = ap.parse_args(argv)
    policies = dict(p.split("=", 1) for p in a.policy)
    seeds = SEEDS[:a.seeds]
    todo = a.only.split(",")

    def env(**kw):
        return TokamakControlEnv(EnvConfig(episode_steps=500, **kw), seed=0)
    envs = {
        "nom": env(plant=PlantRandomisation(enabled=False)),
        "dr": env(plant=PlantRandomisation(enabled=True)),
        "nofilter": env(plant=PlantRandomisation(enabled=False),
                        safety_enabled=False),
    }
    ctrls = load_controllers(envs["nom"], policies)
    res = {"seeds": seeds, "policies": policies}
    for name in todo:
        t0 = time.time()
        if name == "main":
            res[name] = exp_main(envs, ctrls, seeds)
        elif name == "ood":
            res[name] = exp_ood(envs, ctrls, seeds)
        elif name == "delay":
            res[name] = exp_delay(envs, ctrls, seeds)
        elif name == "safety":
            res[name] = exp_safety(envs, ctrls, seeds)
        elif name == "growth":
            res[name] = exp_growth()
        elif name == "seeds_check":
            res[name] = exp_seeds(envs, ctrls, seeds,
                                  [c for c in a.seed_conds.split(',') if c])
        elif name == "draws":
            res[name] = exp_draws(envs, seeds)
        elif name == "device":
            res[name] = exp_device(envs)
        elif name == "cost":
            res[name] = exp_cost(policies)
        elif name == "traces":
            res[name] = exp_traces(envs, ctrls)
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
        with open(a.out, "w") as f:
            json.dump(res, f, default=float)


if __name__ == "__main__":
    main()
