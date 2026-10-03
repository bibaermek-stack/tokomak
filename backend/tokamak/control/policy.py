"""Block 3 -- the neural controller: actor-critic networks and PPO.

Plain numpy, so the controller runs wherever the simulator does and its
inference is exactly the matrix products the deployment would execute:

    h_0 = S_t,   h_{l+1} = tanh(W_l h_l + b_l),   A_t = tanh(W_L h + b_L)

* **Actor** pi_theta(A | S): a Gaussian with that mean and a
  state-independent log standard deviation.  Training samples from it;
  deployment uses the mean.  A sample outside [-1, 1] is clipped by the
  environment, while its log-probability is taken before the clip, as the
  PPO derivation assumes.
* **Critic** V_phi(S): the same trunk shape, linear output.

PPO objective (Schulman et al. 2017) maximised over theta:

    L^CLIP = E_t[ min( r_t A_t , clip(r_t, 1 - eps, 1 + eps) A_t ) ]
    r_t    = pi_theta(A_t | S_t) / pi_theta_old(A_t | S_t)

plus a value loss c_v (V_phi - R_t)^2 and an entropy bonus c_H H[pi].
Advantages from Generalized Advantage Estimation:

    delta_t = r_t + gamma V(S_{t+1}) (1 - done_t) - V(S_t)
    A_t     = sum_l (gamma lambda)^l delta_{t+l}

A time-limit truncation bootstraps from V(S_T); only a real termination
(VDE, wall contact, current quench) zeroes the future.

Gradients are written out by hand (no autograd here) and checked against
finite differences in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
class MLP:
    """tanh MLP with optional tanh output; manual backprop."""

    def __init__(self, sizes, out_tanh: bool, rng: np.random.Generator,
                 out_gain: float = 0.01):
        self.sizes = list(sizes)
        self.out_tanh = out_tanh
        self.params = []
        for k, (i, o) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = out_gain if k == len(sizes) - 2 else np.sqrt(2.0)
            W = _orthogonal(o, i, rng) * gain
            self.params += [W, np.zeros(o)]

    def forward(self, x: np.ndarray):
        """x: (N, in).  Returns (out, cache)."""
        acts = [x]
        h = x
        L = len(self.params) // 2
        for k in range(L):
            W, b = self.params[2 * k], self.params[2 * k + 1]
            z = h @ W.T + b
            h = np.tanh(z) if (k < L - 1 or self.out_tanh) else z
            acts.append(h)
        return h, acts

    def backward(self, acts, g_out: np.ndarray):
        """Gradients of sum(g_out * out) w.r.t. every parameter."""
        grads = [None] * len(self.params)
        L = len(self.params) // 2
        g = g_out
        for k in reversed(range(L)):
            h_out = acts[k + 1]
            if k < L - 1 or self.out_tanh:
                g = g * (1.0 - h_out ** 2)
            W = self.params[2 * k]
            grads[2 * k] = g.T @ acts[k]
            grads[2 * k + 1] = g.sum(axis=0)
            g = g @ W
        return grads


def _orthogonal(o, i, rng):
    a = rng.normal(size=(max(o, i), min(o, i)))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    return q.T[:o, :i] if o <= i else q[:o, :i]


class Adam:
    def __init__(self, params, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def clip_grads(grads, max_norm):
    norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
    if norm > max_norm:
        grads = [g * (max_norm / norm) for g in grads]
    return grads, norm


# ---------------------------------------------------------------------------
class ActorCritic:
    """pi_theta(A|S) and V_phi(S)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 128),
                 log_std_init: float = -1.2, seed: int = 0,
                 value_scale: float = 100.0):
        self.rng = np.random.default_rng(seed)
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.hidden = tuple(hidden)
        self.actor = MLP([obs_dim, *hidden, act_dim], out_tanh=True,
                         rng=self.rng, out_gain=0.01)
        self.critic = MLP([obs_dim, *hidden, 1], out_tanh=False,
                          rng=self.rng, out_gain=1.0)
        self.log_std = np.full(act_dim, float(log_std_init))
        # The critic's raw output is V / value_scale.  Returns here are
        # O(1 / (1 - gamma)) ~ 100, and Adam moves a weight by ~lr per step
        # whatever the gradient, so an unscaled output layer would need tens
        # of thousands of steps just to reach the mean return.
        self.value_scale = float(value_scale)

    # --- inference -----------------------------------------------------
    def act(self, obs: np.ndarray, deterministic: bool = False):
        """Returns (action, log_prob, value) for one observation."""
        o = np.asarray(obs, float)[None]
        mu, _ = self.actor.forward(o)
        v, _ = self.critic.forward(o)
        v = v * self.value_scale
        mu = mu[0]
        if deterministic:
            a = mu.copy()
        else:
            a = mu + np.exp(self.log_std) * self.rng.normal(size=self.act_dim)
        return a, self.log_prob(mu[None], a[None])[0], float(v[0, 0])

    def mean_action(self, obs: np.ndarray) -> np.ndarray:
        """A_t = tanh(W_L h + b_L): the deployed controller."""
        mu, _ = self.actor.forward(np.asarray(obs, float)[None])
        return mu[0]

    def value(self, obs: np.ndarray) -> np.ndarray:
        v, _ = self.critic.forward(np.atleast_2d(obs))
        return v[:, 0] * self.value_scale

    def value_and_grad_fn(self, obs: np.ndarray, ret: np.ndarray,
                          weight: float = 1.0):
        """weight * mean((V - ret)^2) and its gradient w.r.t. the critic."""
        o, acts = self.critic.forward(obs)
        v = o[:, 0] * self.value_scale
        g = (weight * 2.0 * (v - ret) / len(obs) * self.value_scale)[:, None]
        return v, self.critic.backward(acts, g)

    def log_prob(self, mu, a):
        std = np.exp(self.log_std)
        return np.sum(-0.5 * ((a - mu) / std) ** 2 - self.log_std
                      - 0.5 * LOG_2PI, axis=1)

    def entropy(self) -> float:
        return float(np.sum(self.log_std + 0.5 * (LOG_2PI + 1.0)))

    # --- persistence -------------------------------------------------
    def state_dict(self) -> dict:
        d = {f"actor_{k}": p for k, p in enumerate(self.actor.params)}
        d.update({f"critic_{k}": p for k, p in enumerate(self.critic.params)})
        d["log_std"] = self.log_std
        d["value_scale"] = np.array(self.value_scale)
        d["sizes"] = np.array([self.obs_dim, self.act_dim, *self.hidden])
        return d

    def save(self, path: str, **meta) -> None:
        extra = {f"meta_{k}": np.asarray(v) for k, v in meta.items()}
        np.savez(path, **self.state_dict(), **extra)

    @classmethod
    def load(cls, path: str) -> "ActorCritic":
        d = np.load(path)
        sizes = [int(x) for x in d["sizes"]]
        ac = cls(sizes[0], sizes[1], hidden=tuple(sizes[2:]),
                 value_scale=float(d["value_scale"]) if "value_scale" in d
                 else 100.0)
        for k in range(len(ac.actor.params)):
            ac.actor.params[k][...] = d[f"actor_{k}"]
        for k in range(len(ac.critic.params)):
            ac.critic.params[k][...] = d[f"critic_{k}"]
        ac.log_std[...] = d["log_std"]
        return ac


# ---------------------------------------------------------------------------
@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 8
    minibatch: int = 512
    lr: float = 1e-4
    c_value: float = 0.5
    c_entropy: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.02
    log_std_min: float = -4.0
    log_std_max: float = 0.0


class RolloutBuffer:
    def __init__(self):
        self.obs, self.act, self.logp, self.rew, self.val = [], [], [], [], []
        self.done, self.next_val = [], []

    def add(self, obs, act, logp, rew, val, terminated, truncated, next_val):
        """``next_val`` is V(S_{t+1}) -- used only on a truncation, where
        the episode ends without the plasma having been lost."""
        self.obs.append(np.asarray(obs, float))
        self.act.append(np.asarray(act, float))
        self.logp.append(float(logp))
        self.rew.append(float(rew))
        self.val.append(float(val))
        self.done.append((bool(terminated), bool(truncated)))
        self.next_val.append(float(next_val))

    def __len__(self):
        return len(self.rew)

    def compute(self, last_value: float, gamma: float, lam: float):
        """GAE advantages and returns.  ``last_value`` bootstraps a
        rollout that stops mid-episode."""
        n = len(self.rew)
        adv = np.zeros(n)
        g = 0.0
        for t in reversed(range(n)):
            term, trunc = self.done[t]
            if term:
                nv, cont = 0.0, 0.0
            elif trunc:
                nv, cont = self.next_val[t], 0.0
            else:
                nv = self.val[t + 1] if t + 1 < n else last_value
                cont = 1.0
            delta = self.rew[t] + gamma * nv - self.val[t]
            g = delta + gamma * lam * cont * g
            adv[t] = g
        ret = adv + np.array(self.val)
        return (np.array(self.obs), np.array(self.act), np.array(self.logp),
                adv, ret)


class PPO:
    def __init__(self, ac: ActorCritic, cfg: Optional[PPOConfig] = None):
        self.ac = ac
        self.cfg = cfg or PPOConfig()
        self.opt_pi = Adam(ac.actor.params + [ac.log_std], lr=self.cfg.lr)
        self.opt_v = Adam(ac.critic.params, lr=self.cfg.lr)

    def loss_and_grads(self, obs, act, logp_old, adv, ret):
        """Value and gradients of the PPO loss on one minibatch.

        Loss = -L^CLIP + c_v * mean((V - R)^2) - c_H * H.  Returned as
        separate actor and critic gradient lists plus diagnostics.
        """
        c = self.cfg
        ac = self.ac
        N = len(obs)
        mu, acts_pi = ac.actor.forward(obs)
        std = np.exp(ac.log_std)
        logp = ac.log_prob(mu, act)
        ratio = np.exp(logp - logp_old)
        clipped = np.clip(ratio, 1 - c.clip, 1 + c.clip)
        s1, s2 = ratio * adv, clipped * adv
        surr = np.minimum(s1, s2)
        # d min(s1, s2)/d ratio = adv where the unclipped branch is the min
        use = (s1 <= s2)
        dL_dlogp = -(adv * ratio * use) / N            # d(-surr)/dlogp
        z = (act - mu) / std
        g_mu = dL_dlogp[:, None] * (z / std)
        g_logstd = np.sum(dL_dlogp[:, None] * (z * z - 1.0), axis=0)
        g_logstd -= c.c_entropy                         # -c_H dH/dlogstd
        g_actor = ac.actor.backward(acts_pi, g_mu)

        v, g_critic = ac.value_and_grad_fn(obs, ret, weight=c.c_value)

        loss = (-np.mean(surr) + c.c_value * np.mean((v - ret) ** 2)
                - c.c_entropy * ac.entropy())
        info = {
            "loss": float(loss),
            "policy_loss": float(-np.mean(surr)),
            "value_loss": float(np.mean((v - ret) ** 2)),
            "approx_kl": float(np.mean((ratio - 1.0) - np.log(ratio))),
            "clip_frac": float(np.mean(np.abs(ratio - 1.0) > c.clip)),
        }
        return loss, g_actor + [g_logstd], g_critic, info

    def update(self, buf: RolloutBuffer, last_value: float) -> dict:
        c = self.cfg
        obs, act, logp_old, adv, ret = buf.compute(last_value, c.gamma, c.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(obs)
        rng = self.ac.rng
        hist = []
        stop = False
        for _ in range(c.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, c.minibatch):
                idx = perm[s:s + c.minibatch]
                _, gpi, gv, info = self.loss_and_grads(
                    obs[idx], act[idx], logp_old[idx], adv[idx], ret[idx])
                if c.target_kl is not None and info["approx_kl"] > 1.5 * c.target_kl:
                    stop = True
                    break
                gpi, _ = clip_grads(gpi, c.max_grad_norm)
                gv, _ = clip_grads(gv, c.max_grad_norm)
                self.opt_pi.step(gpi)
                self.opt_v.step(gv)
                np.clip(self.ac.log_std, c.log_std_min, c.log_std_max,
                        out=self.ac.log_std)
                hist.append(info)
            if stop:
                break
        out = {k: float(np.mean([h[k] for h in hist])) for k in hist[0]} \
            if hist else {}
        out["early_stop"] = stop
        out["explained_var"] = float(
            1.0 - np.var(ret - self.ac.value(obs)) / max(np.var(ret), 1e-12))
        return out


# ---------------------------------------------------------------------------
def behaviour_clone(ac: ActorCritic, obs: np.ndarray, act: np.ndarray,
                    epochs: int = 30, lr: float = 1e-3, batch: int = 256,
                    seed: int = 0) -> float:
    """Fit the actor mean to an expert's actions (mean-squared error).

    Starting PPO from a policy that already holds the plasma turns the
    problem from "discover vertical stabilisation from random voltages" --
    which almost always ends in a VDE within 100 ms -- into improving a
    working controller, which is how learned controllers are commissioned
    in practice.  Returns the final training MSE.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(ac.actor.params, lr=lr)
    target = np.clip(act, -0.999, 0.999)
    n = len(obs)
    mse = np.inf
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            mu, acts = ac.actor.forward(obs[idx])
            g = 2.0 * (mu - target[idx]) / len(idx)
            grads = ac.actor.backward(acts, g)
            grads, _ = clip_grads(grads, 5.0)
            opt.step(grads)
        mu, _ = ac.actor.forward(obs)
        mse = float(np.mean((mu - target) ** 2))
    return mse
