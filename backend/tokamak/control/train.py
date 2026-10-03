"""Training: behaviour cloning from the classical controller, then PPO.

    python -m tokamak.control.train --ppo-steps 200000 --out policy.npz

1. **Demonstrations.**  The model-based controller (:mod:`.baseline`) runs
   the closed loop with exploration noise added to what it *executes*,
   while the label is always what it *asked for* -- so the data covers the
   off-nominal states a cloned policy drifts into (DAgger in spirit).
2. **Cloning.**  The actor mean is fitted to those labels; the critic is
   fitted to the discounted returns of the same episodes, so PPO's first
   advantages are not noise.
3. **PPO.**  Clipped-objective fine-tuning against the full reward with
   randomised delays, noise, drift, profile events and reference steps.
   The deterministic policy is evaluated on fixed seeds after every update
   and the best one is kept.

Everything is numpy; a few hundred thousand steps take tens of minutes on
one core, dominated by the simulator, not the network.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import numpy as np

from .baseline import PIDController
from .env import EnvConfig, TokamakControlEnv
from .loop import CoSimulation, NeuralController
from .policy import (ActorCritic, PPO, PPOConfig, RolloutBuffer, Adam,
                     behaviour_clone, clip_grads)


def collect_demonstrations(env: TokamakControlEnv, expert: PIDController,
                           episodes: int, noise: float = 0.05,
                           seed: int = 0, gamma: float = 0.99):
    """(obs, expert_action, discounted_return) arrays."""
    rng = np.random.default_rng(seed)
    O, A, G = [], [], []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        expert.reset()
        rews = []
        while True:
            a_exp = expert(obs)
            a_run = np.clip(a_exp + noise * rng.normal(size=a_exp.shape), -1, 1)
            O.append(obs); A.append(a_exp)
            obs, r, term, trunc, _ = env.step(a_run)
            rews.append(r)
            if term or trunc:
                break
        # a truncated episode would have gone on: close its tail with the
        # steady-state value of its last rewards, or the targets depend on
        # the time left -- which the observation does not contain
        g = 0.0 if term else float(np.mean(rews[-50:])) / (1.0 - gamma)
        ret = []
        for r in reversed(rews):
            g = r + gamma * g
            ret.append(g)
        G.extend(reversed(ret))
    return np.array(O), np.array(A), np.array(G)


def fit_critic(ac: ActorCritic, obs, ret, epochs: int = 60, lr: float = 1e-3,
               batch: int = 256, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    opt = Adam(ac.critic.params, lr=lr)
    n = len(obs)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            _, grads = ac.value_and_grad_fn(obs[idx], ret[idx])
            grads, _ = clip_grads(grads, 5.0)
            opt.step(grads)
    return float(np.mean((ac.value(obs) - ret) ** 2))


def evaluate(env: TokamakControlEnv, controller, seeds) -> dict:
    sim = CoSimulation(env)
    rows = [sim.run(controller, seed=int(s)).summary() for s in seeds]
    return {
        "return": float(np.mean([r["return"] for r in rows])),
        "survival": float(np.mean([not r["terminated"] for r in rows])),
        "rms_position_error_mm": float(np.mean(
            [r["rms_position_error_mm"] for r in rows])),
        "rms_shape_error_mm": float(np.mean(
            [r["rms_shape_error_mm"] for r in rows])),
        "qp_intervention_frac": float(np.mean(
            [r["qp_intervention_frac"] for r in rows])),
        "episodes": rows,
    }


def train(ppo_steps: int = 200_000, bc_episodes: int = 20,
          rollout: int = 2048, episode_steps: int = 500,
          hidden=(128, 128), log_std: float = -2.5, seed: int = 0,
          eval_seeds=range(1000, 1006), out: Optional[str] = None,
          ppo_cfg: Optional[PPOConfig] = None, log=print) -> dict:
    env = TokamakControlEnv(EnvConfig(episode_steps=episode_steps), seed=seed)
    expert = PIDController(env)
    log(f"device: {env.sim.coil_names}, vertical growth "
        f"{env.sim.vertical_growth_rate():.0f}/s, "
        f"PID gains kp_z={expert.g.kp_z:.0f} kd_z={expert.g.kd_z:.2f}")

    ac = ActorCritic(env.obs_dim, env.act_dim, hidden=hidden,
                     log_std_init=log_std, seed=seed)
    history = {"expert": None, "bc": None, "ppo": []}
    if bc_episodes > 0:
        t0 = time.time()
        O, A, G = collect_demonstrations(env, expert, bc_episodes, seed=seed)
        mse = behaviour_clone(ac, O, A, seed=seed)
        vmse = fit_critic(ac, O, G, seed=seed)
        log(f"cloned {len(O)} samples: actor mse {mse:.2e}, critic mse "
            f"{vmse:.2f} ({time.time() - t0:.0f}s)")
    history["expert"] = _strip(evaluate(env, expert, eval_seeds))
    history["bc"] = _strip(evaluate(env, NeuralController(ac), eval_seeds))
    log(f"expert: {_fmt(history['expert'])}")
    log(f"cloned: {_fmt(history['bc'])}")

    ppo = PPO(ac, ppo_cfg or PPOConfig())
    best = history["bc"]["return"]
    best_state = {k: v.copy() for k, v in ac.state_dict().items()}
    obs = env.reset()
    steps = 0
    t0 = time.time()
    while steps < ppo_steps:
        buf = RolloutBuffer()
        for _ in range(rollout):
            a, logp, v = ac.act(obs)
            obs2, r, term, trunc, _ = env.step(a)
            nv = ac.value(obs2)[0] if trunc else 0.0
            buf.add(obs, a, logp, r, v, term, trunc, nv)
            obs = env.reset() if (term or trunc) else obs2
            steps += 1
        last_v = float(ac.value(obs)[0])
        stats = ppo.update(buf, last_v)
        ev = evaluate(env, NeuralController(ac), eval_seeds)
        row = {"steps": steps, **_strip(ev), **stats,
               "std": float(np.exp(ac.log_std).mean()),
               "elapsed_s": time.time() - t0}
        history["ppo"].append(row)
        log(f"[{steps:7d}] {_fmt(ev)}  kl={stats.get('approx_kl', 0):.4f} "
            f"ev={stats['explained_var']:.2f}")
        if ev["return"] > best:
            best = ev["return"]
            best_state = {k: v.copy() for k, v in ac.state_dict().items()}
        obs = env.reset()

    # restore the best evaluated policy
    for k in range(len(ac.actor.params)):
        ac.actor.params[k][...] = best_state[f"actor_{k}"]
    for k in range(len(ac.critic.params)):
        ac.critic.params[k][...] = best_state[f"critic_{k}"]
    ac.log_std[...] = best_state["log_std"]
    history["final"] = _strip(evaluate(env, NeuralController(ac), eval_seeds))
    if out:
        ac.save(out, episode_steps=episode_steps)
        with open(out.rsplit(".", 1)[0] + ".json", "w") as f:
            json.dump(history, f, indent=1, default=float)
    return {"ac": ac, "env": env, "history": history}


def _strip(ev: dict) -> dict:
    return {k: v for k, v in ev.items() if k != "episodes"}


def _fmt(ev: dict) -> str:
    return (f"return {ev['return']:7.1f}  survival {ev['survival']:.2f}  "
            f"pos {ev['rms_position_error_mm']:5.1f} mm  "
            f"shape {ev['rms_shape_error_mm']:5.1f} mm  "
            f"qp {ev['qp_intervention_frac']:.2f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ppo-steps", type=int, default=200_000)
    p.add_argument("--bc-episodes", type=int, default=20)
    p.add_argument("--rollout", type=int, default=2048)
    p.add_argument("--episode-steps", type=int, default=500)
    p.add_argument("--log-std", type=float, default=-2.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="policy.npz")
    a = p.parse_args(argv)
    train(ppo_steps=a.ppo_steps, bc_episodes=a.bc_episodes,
          rollout=a.rollout, episode_steps=a.episode_steps,
          log_std=a.log_std, seed=a.seed, out=a.out)


if __name__ == "__main__":
    main()
