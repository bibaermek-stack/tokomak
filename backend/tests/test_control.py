"""Tests for the closed-loop control package (tokamak.control).

Physics first (Green's functions, force balance, the vertical instability),
then each block of the loop on its own, then the loop closed.
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from tokamak.control import greens
from tokamak.control.baseline import PIDController
from tokamak.control.env import EnvConfig, TokamakControlEnv
from tokamak.control.loop import CoSimulation, ZeroController, contract_spec
from tokamak.control.policy import (MLP, PPO, ActorCritic, PPOConfig,
                                    RolloutBuffer, behaviour_clone)
from tokamak.control.safety import SafetyFilter, solve_box_qp
from tokamak.control.sensors import SensorConfig, SensorEmulator, raw_vector
from tokamak.control.simulator import TokamakSimulator, linearise


@pytest.fixture(scope="module")
def sim():
    return TokamakSimulator()


@pytest.fixture(scope="module")
def env():
    return TokamakControlEnv(EnvConfig(episode_steps=300), seed=0)


@pytest.fixture(scope="module")
def pid(env):
    return PIDController(env)


# --- Green's functions -----------------------------------------------------
@pytest.mark.parametrize("Rs,Zs,R,Z", [(1.3, 0.4, 0.9, -0.1),
                                       (0.5, -0.7, 1.4, 0.2),
                                       (2.0, 0.0, 2.1, 0.05)])
def test_field_is_curl_of_flux(Rs, Zs, R, Z):
    """B_R = -(1/2piR) dPsi/dZ, B_Z = (1/2piR) dPsi/dR."""
    h = 1e-6
    M = lambda r, z: greens.mutual(Rs, Zs, r, z)
    BR, BZ = greens.field(Rs, Zs, R, Z)
    assert BR == pytest.approx(-(M(R, Z + h) - M(R, Z - h)) / (2 * h)
                               / (2 * np.pi * R), rel=1e-6)
    assert BZ == pytest.approx((M(R + h, Z) - M(R - h, Z)) / (2 * h)
                               / (2 * np.pi * R), rel=1e-6)
    M2, BR2, BZ2 = greens.mutual_and_field(Rs, Zs, R, Z)
    assert (M2, BR2, BZ2) == pytest.approx((M(R, Z), BR, BZ), rel=1e-12)


def test_mutual_symmetric_and_on_axis_field():
    assert greens.mutual(1.0, 0.2, 2.0, -0.3) == pytest.approx(
        greens.mutual(2.0, -0.3, 1.0, 0.2), rel=1e-12)
    # on the axis of a loop: B_Z = mu0 R^2 / (2 (R^2 + z^2)^1.5)
    R, z = 1.5, 0.7
    _, BZ = greens.field(R, 0.0, 1e-6, z)
    assert BZ == pytest.approx(4e-7 * np.pi * R ** 2
                               / (2 * (R ** 2 + z ** 2) ** 1.5), rel=1e-4)


# --- simulator -------------------------------------------------------------
def test_initial_state_is_in_force_balance(sim):
    sim.reset()
    _, gR, gZ = sim._plasma_couplings(*sim.x)
    F = sim._forces(sim.x[0], sim.I[-1], sim.I[:-1], gR, gZ)
    hoop = sim._hoop(sim.x[0], sim.I[-1])
    assert np.all(np.abs(F) < 1e-4 * hoop)


def test_equilibrium_comes_from_grad_shafranov(sim):
    gs = sim.plasma.gs_summary
    assert gs["converged"]
    assert sim.plasma.li == pytest.approx(gs["li3"])
    assert sim.plasma.w.sum() == pytest.approx(1.0)
    assert sim.wall_clearance() > 0.05


def test_inductance_matrix_is_positive_definite(sim):
    M, _, _ = sim._M(*sim.x)
    assert np.allclose(M, M.T, rtol=1e-6, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(0.5 * (M + M.T)) > 0)


def test_vertical_mode_is_unstable_on_the_wall_time(sim):
    sim.reset()
    gamma = sim.vertical_growth_rate()
    tau_wall = sim.M_ee[sim.n_act, sim.n_act] / sim.R_el[sim.n_act]
    assert 20.0 < gamma < 500.0
    # resistive-wall mode: slower than a single segment's L/R would allow
    assert gamma < 1.0 / tau_wall
    # the time-stepper sees the same instability
    assert linearise(sim).growth_rate == pytest.approx(gamma, rel=0.2)


def test_open_loop_plasma_is_lost(sim):
    sim.reset(vs_kick=20.0)
    gamma = sim.vertical_growth_rate()
    z = []
    lost = False
    for _ in range(400):
        try:
            st = sim.step(sim.feedforward())
        except Exception:
            lost = True
            break
        z.append(st.Z_c)
        if st.d_min <= 0.02:
            lost = True
            break
    assert lost
    # exponential growth at roughly the linear rate while still small
    z = np.abs(np.array(z))
    t = np.arange(len(z)) * sim.dt
    w = (z > 1e-3) & (z < 1e-2)
    rate = np.polyfit(t[w], np.log(z[w]), 1)[0]
    assert rate == pytest.approx(gamma, rel=0.4)


def test_feedforward_holds_plasma_current(sim):
    sim.reset()
    for _ in range(50):
        sim.step(sim.feedforward())
    # the CS ramp supplies the loop voltage: Ip stays within 1 %
    assert sim.I[-1] == pytest.approx(sim.Ip0, rel=0.01)


# --- sensors ---------------------------------------------------------------
def _emulator(**kw):
    ref = np.zeros(3 + 2 + 2 + 2)
    return SensorEmulator(2, 2, 2, ref, np.ones(2), 1e6,
                          SensorConfig(**kw), rng=np.random.default_rng(1))


@pytest.mark.parametrize("tau", [1, 2, 3])
def test_delay_buffer_is_exact(tau):
    em = _emulator(enabled=False)
    em.reset(delay=tau)
    seen = []
    for k in range(10):
        em.push(np.full(em.size, float(k)))
        seen.append(em.measure(0.0)[0])
    # before tau samples exist the first one is held
    assert seen == [max(k - tau, 0) for k in range(10)]


def test_noise_and_drift_statistics():
    em = _emulator(noise_pos_m=1e-3, drift_rel_per_s=0.0)
    em.reset(delay=1)
    em.push(np.zeros(em.size))
    x = np.array([em.measure(0.0)[0] for _ in range(4000)])
    assert np.std(x) == pytest.approx(1e-3, rel=0.05)
    em = _emulator(noise_pos_m=0.0, noise_psi_rel=0.0, noise_b_rel=0.0,
                   noise_ip_rel=0.0, noise_coil_rel=0.0, drift_rel_per_s=1.0)
    em.reset(delay=1)
    em.push(np.ones(em.size))
    d1, d2 = em.measure(1.0) - 1.0, em.measure(2.0) - 1.0
    assert np.allclose(d2, 2 * d1)          # linear in time
    assert np.allclose(d1[:5], 0.0)         # only the magnetic channels drift
    assert np.all(d1[5:] != 0.0)


def test_contract_layout_matches_vector(env):
    spec = contract_spec(env)
    v = raw_vector(env.sim.state())
    assert len(spec["sim_to_controller"]["layout"]) == len(v)
    assert len(spec["sim_to_controller"]["units"]) == len(v)
    assert len(spec["controller_to_sim"]["layout"]) == env.act_dim
    assert v.dtype == np.float64


# --- safety filter ---------------------------------------------------------
def test_qp_matches_reference_solver():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = 8
        xt = rng.normal(0, 1.5, n)
        B = rng.normal(0, 1, (n, n)) * 0.3 + np.eye(n)
        C = np.vstack([np.eye(n), B / np.linalg.norm(B, axis=1)[:, None]])
        l = np.concatenate([-np.ones(n), -rng.uniform(0.2, 1, n)])
        u = -l
        x, _, _, _ = solve_box_qp(xt, C, l, u)
        ref = minimize(lambda v: 0.5 * np.sum((v - xt) ** 2), np.zeros(n),
                       jac=lambda v: v - xt, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": lambda v: C @ v - l},
                                    {"type": "ineq", "fun": lambda v: u - C @ v}],
                       options={"ftol": 1e-12, "maxiter": 500})
        assert np.allclose(x, ref.x, atol=1e-6)


def test_filter_passes_safe_requests_and_projects_unsafe_ones(sim):
    sim.reset()
    f = SafetyFilter(sim.V_max, sim.I_max, horizon=5)
    M, _, _ = sim._M(*sim.x)
    a, B = f.current_model(M, sim.R_el, sim.I, sim.dt, sim.n_act)
    V0 = sim.feedforward()
    r = f(V0, a, B, V_prev=V0)
    assert not r.intervened and np.array_equal(r.V, V0)

    V_bad = V0 + 5.0 * sim.V_max * np.sign(np.random.default_rng(2).normal(size=sim.n_act))
    r = f(V_bad, a, B, V_prev=V0)
    lo, hi = f.box(V0)
    assert r.intervened
    assert np.all(r.V >= lo - 1e-9) and np.all(r.V <= hi + 1e-9)


def test_filter_enforces_current_limit():
    # one coil, I_next = I + 0.5 V: at I = 0.9 a +1 V request would exceed 1
    f = SafetyFilter(np.array([1.0]), np.array([1.0]), horizon=1)
    r = f(np.array([1.0]), a=np.array([0.9]), B=np.array([[0.5]]))
    assert r.V[0] == pytest.approx(0.2, abs=1e-6)
    assert abs(r.I_pred[0]) <= 1.0 + 1e-9


# --- networks and PPO ------------------------------------------------------
def test_mlp_backprop_matches_finite_differences():
    rng = np.random.default_rng(0)
    net = MLP([5, 7, 3], out_tanh=True, rng=rng, out_gain=1.0)
    x = rng.normal(size=(4, 5))
    w = rng.normal(size=(4, 3))
    out, acts = net.forward(x)
    grads = net.backward(acts, w)
    for k, p in enumerate(net.params):
        idx = tuple(rng.integers(0, s) for s in p.shape)
        old = p[idx]
        p[idx] = old + 1e-6
        fp = np.sum(net.forward(x)[0] * w)
        p[idx] = old - 1e-6
        fm = np.sum(net.forward(x)[0] * w)
        p[idx] = old
        assert grads[k][idx] == pytest.approx((fp - fm) / 2e-6, rel=1e-4, abs=1e-8)


def test_ppo_gradient_matches_finite_differences():
    ac = ActorCritic(4, 2, hidden=(8,), seed=3)
    ppo = PPO(ac, PPOConfig(c_entropy=0.01))
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(16, 4))
    act = rng.normal(size=(16, 2)) * 0.3
    mu, _ = ac.actor.forward(obs)
    logp_old = ac.log_prob(mu, act) + rng.normal(size=16) * 0.05
    adv, ret = rng.normal(size=16), rng.normal(size=16)
    _, gpi, gv, _ = ppo.loss_and_grads(obs, act, logp_old, adv, ret)
    params = ac.actor.params + [ac.log_std] + ac.critic.params
    grads = gpi + gv
    for p, g in zip(params, grads):
        idx = tuple(rng.integers(0, s) for s in p.shape)
        old = p[idx]
        p[idx] = old + 1e-6
        lp = ppo.loss_and_grads(obs, act, logp_old, adv, ret)[0]
        p[idx] = old - 1e-6
        lm = ppo.loss_and_grads(obs, act, logp_old, adv, ret)[0]
        p[idx] = old
        assert g[idx] == pytest.approx((lp - lm) / 2e-6, rel=1e-3, abs=1e-7)


def test_gae_bootstraps_truncation_but_not_termination():
    b = RolloutBuffer()
    b.add(np.zeros(1), np.zeros(1), 0.0, 1.0, 0.5, False, True, next_val=2.0)
    b.add(np.zeros(1), np.zeros(1), 0.0, 1.0, 0.5, True, False, next_val=9.0)
    _, _, _, adv, ret = b.compute(last_value=0.0, gamma=0.9, lam=1.0)
    assert adv[0] == pytest.approx(1.0 + 0.9 * 2.0 - 0.5)   # truncated
    assert adv[1] == pytest.approx(1.0 - 0.5)               # terminated


def test_ppo_learns_a_contextual_bandit():
    """r = -(a - 0.5 s)^2: the optimum is a linear policy."""
    rng = np.random.default_rng(0)
    ac = ActorCritic(1, 1, hidden=(16,), log_std_init=-1.0, seed=0)
    ppo = PPO(ac, PPOConfig(epochs=10, minibatch=128, lr=3e-3, target_kl=None))
    test = np.linspace(-1, 1, 21)[:, None]
    err0 = np.mean((ac.actor.forward(test)[0][:, 0] - 0.5 * test[:, 0]) ** 2)
    for _ in range(25):
        buf = RolloutBuffer()
        for _ in range(512):
            s = rng.uniform(-1, 1, 1)
            a, lp, v = ac.act(s)
            buf.add(s, a, lp, -float((a[0] - 0.5 * s[0]) ** 2), v,
                    True, False, 0.0)
        ppo.update(buf, 0.0)
    err = np.mean((ac.actor.forward(test)[0][:, 0] - 0.5 * test[:, 0]) ** 2)
    assert err < 0.1 * err0


def test_policy_round_trips_through_disk(tmp_path):
    ac = ActorCritic(6, 3, hidden=(10, 10), seed=4)
    path = tmp_path / "p.npz"
    ac.save(str(path))
    bc = ActorCritic.load(str(path))
    o = np.random.default_rng(0).normal(size=6)
    assert np.allclose(ac.mean_action(o), bc.mean_action(o))
    assert np.allclose(ac.value(o), bc.value(o))


def test_behaviour_cloning_fits_a_linear_expert():
    rng = np.random.default_rng(0)
    O = rng.normal(size=(2000, 5))
    W = rng.normal(size=(5, 2)) * 0.2
    ac = ActorCritic(5, 2, hidden=(32,), seed=0)
    Y = np.tanh(O @ W)
    # within 1 % of the target variance
    assert behaviour_clone(ac, O, Y, epochs=40) < 0.01 * np.var(Y)


# --- closed loop -----------------------------------------------------------
def test_baseline_gains_stabilise_the_linear_plant(pid):
    assert pid.design_info["vertical_rho"] < 1.001
    assert pid.g.kp_z != 0 and pid.g.kd_z != 0


def test_baseline_holds_the_plasma(env, pid):
    tr = CoSimulation(env).run(pid, steps=300, seed=0, delay=2, disturb=False)
    s = tr.summary()
    assert not s["terminated"], s["reason"]
    assert s["max_abs_Z_mm"] < 10.0
    assert s["rms_position_error_mm"] < 15.0
    assert abs(tr.Ip[-1] / env.sim.Ip0 - 1.0) < 0.02


def test_open_loop_terminates_with_a_named_reason(env):
    env.cfg.disturbance.vs_kick_rel = 0.02
    tr = CoSimulation(env).run(ZeroController(env.act_dim), steps=300, seed=3,
                               disturb=True)
    assert tr.reason is not None
    assert any(k in tr.reason for k in ("VDE", "қабырға", "ток"))


def test_reward_terms_and_termination_penalty(env):
    env.reset(seed=0, disturb=False)
    _, r, term, trunc, info = env.step(np.zeros(env.act_dim))
    assert not term
    assert r <= 1.0 + 1e-9
    assert set(info["terms"]) == {"pos", "shape", "ip", "smooth", "qp",
                                  "terminal"}
    # full VS voltage drives the plasma vertically into the wall
    env.reset(seed=0, disturb=False)
    a = np.zeros(env.act_dim)
    a[-1] = 1.0
    for _ in range(300):
        _, r, term, trunc, info = env.step(a)
        if term:
            break
    assert term and r < -env.cfg.reward.terminal_penalty + 1.0
