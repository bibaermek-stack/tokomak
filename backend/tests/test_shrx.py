"""SHRX: the bridge to QLKNN's variables, the network, and its use inside the solver.

The reference numbers in tests/data were produced once by the real QLKNN / TORAX
(ml/shrx/make_golden.py) so these tests need neither JAX nor TORAX.
"""
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from tokamak import shrx
from tokamak import transport as tr
from tokamak.machines import list_machines
from tokamak.shrx import bridge
from tokamak.solver import Simulator, SolverConfig

DATA = Path(__file__).parent / "data"
client = TestClient(app)
needs_weights = pytest.mark.skipif(not shrx.available(), reason="SHRX weights not present")


def test_bridge_reproduces_torax_inputs():
    """Our NumPy inputs vs the ten numbers TORAX itself feeds QLKNN, on one real state (ITER, Q ~ 10)."""
    g = np.load(DATA / "shrx_bridge_golden.npz")
    X, _ = bridge.features(g["rho"], g["Te"], g["Ti"], g["ne20"], g["q"], float(g["R0"]), float(g["a"]),
                           float(g["B0"]), float(g["zeff"]), float(g["normni"]), kappa_edge=float(g["kappa_edge"]))
    ref = g["X_torax"]
    sl = slice(2, 24)                       # the axis and the last faces are finite-difference ends
    for j, name in enumerate(bridge.FEATURES):
        err = np.median(np.abs(X[sl, j] - ref[sl, j]) / np.maximum(np.abs(ref[sl, j]), 0.05))
        assert err < 0.05, f"{name}: median relative error {err:.3f}"


def test_flux_to_si_conversion():
    # chi = (R/a) q / (R/L_T) chi_GB, clipped to TORAX's [0.05, 100]
    aux = {"Ati": np.array([5.0, 5.0]), "Ate": np.array([4.0, 4.0]), "Ti": np.array([8.0, 8.0]),
           "B0": 5.3, "R0": 6.2, "a": 2.0}
    chi_i, chi_e = bridge.to_si(np.array([2.0, -1.0]), np.array([3.0, 1e9]), aux)
    gb = bridge.chi_gb(8.0, 5.3, 2.0)
    assert chi_i[0] == pytest.approx(3.1 * 2.0 / 5.0 * gb)
    assert chi_e[0] == pytest.approx(3.1 * 3.0 / 4.0 * gb)
    assert chi_i[1] == 0.05 and chi_e[1] == 100.0


@needs_weights
def test_shrx_matches_qlknn_on_golden_samples():
    g = np.load(DATA / "shrx_flux_golden.npz")
    pred, ref = shrx.load().predict(g["X"]), g["Y"]
    real = g["tag"] == 0
    for k, name in enumerate(("q_i", "q_e")):
        r2 = 1 - np.sum((ref[real, k] - pred[real, k]) ** 2) / np.sum((ref[real, k] - ref[real, k].mean()) ** 2)
        assert r2 > 0.9, f"{name}: R2 = {r2:.3f} on states the solver visits"
        active = np.abs(ref[real, k]) > 1.0
        rel = np.median(np.abs(pred[real, k][active] - ref[real, k][active]) / np.abs(ref[real, k][active]))
        assert rel < 0.25, f"{name}: median relative error {rel:.3f}"


@needs_weights
def test_shrx_clips_inputs_and_flags_them():
    net = shrx.load()
    X = np.array([[7, 7, 2, 2, 1.5, 1.0, 0.5, 1.0, -1.5, 0.9],
                  [1e4, 1e4, 1e3, 1e3, 1e3, 1e3, 5.0, 50.0, 50.0, 5.0]], dtype=np.float32)
    flux = net.predict(X)
    assert np.all(np.isfinite(flux))
    assert list(net.out_of_range(X)) == [False, True]


@needs_weights
@pytest.mark.parametrize("machine", sorted(list_machines()))
def test_shrx_runs_every_machine(machine):
    sim = Simulator(SolverConfig(machine=machine, transport_model="shrx", equilibrium=False,
                                 disruption_enabled=False, n_rho=33))
    for _ in range(15):
        sim._scenario_actuators()
        sim.step()
    assert sim._chi_i_model is not None            # chi_i comes from SHRX, not the default shape
    assert np.all(np.isfinite(sim.state.Te)) and sim.state.Te.min() > 0
    assert np.all(np.isfinite(sim.chi)) and sim.chi.min() > 0.0


@needs_weights
def test_old_name_still_works_and_is_shrx():
    assert tr.canonical_transport("neural_surrogate") == "shrx"
    sim = Simulator(SolverConfig(machine="iter", transport_model="neural_surrogate", equilibrium=False,
                                 disruption_enabled=False, n_rho=33))
    sim._scenario_actuators()
    sim.step()
    assert sim._chi_i_model is not None


@needs_weights
def test_shrx_iter_flat_top_is_burning():
    """The integrated check that matters: SHRX in the ITER discharge ends near the design point.

    Reference values: ITER design 500 MW; SINTEZ driven by the real QLKNN gives 470-490 MW.
    The bounds are wide because this runs without the Grad-Shafranov solve to stay fast.
    """
    sim = Simulator(SolverConfig(machine="iter", transport_model="shrx", equilibrium=False,
                                 disruption_enabled=False, n_rho=65))
    sim.run_scenario(t_end=100.0, record_every=10.0)
    assert 250.0 < sim.P_fus < 800.0, sim.P_fus
    assert sim.state.Te_avg > 6.0


@needs_weights
def test_shrx_endpoint():
    r = client.post("/ml/shrx", json={"Ati": 9.0, "Ate": 9.0, "Ane": 2.0, "q": 2.0, "smag": 1.5, "x": 0.6,
                                      "Ti_Te": 1.0, "LogNuStar": -1.5})
    assert r.status_code == 200
    d = r.json()
    assert 0.05 <= d["chi_e_m2_s"] <= 100.0 and 0.05 <= d["chi_i_m2_s"] <= 100.0
    assert d["model"] == "SHRX" and d["out_of_range"] is False
    # far below the critical gradient QLKNN (and SHRX) switch the turbulence off
    low = client.post("/ml/shrx", json={"Ati": 1.0, "Ate": 1.0, "Ane": 1.0, "q": 2.0, "smag": 1.5, "x": 0.6}).json()
    assert low["chi_i_m2_s"] < d["chi_i_m2_s"]


def test_machine_validity_flags_spherical_tokamaks():
    machines = list_machines()
    assert shrx.machine_validity(machines["iter"])["status"] == "ok"
    assert shrx.machine_validity(machines["diiid"])["status"] == "ok"
    assert shrx.machine_validity(machines["ktm"])["status"] == "marginal"
    for key in ("nstx", "nstxu", "mastu"):
        v = shrx.machine_validity(machines[key])
        assert v["status"] == "outside" and v["reasons"], key


@needs_weights
def test_ood_core_ignores_the_structural_edge_points():
    """ood_fraction counts the axis and edge, outside QLKNN's radial range in every discharge; ood_core does not."""
    m = list_machines()["jet"]
    rho = np.linspace(0, 1, 65)
    Te = 1.0 + 20.0 * (1 - rho ** 2) ** 1.5
    out = shrx.evaluate(rho, Te, 0.9 * Te, 0.1 + 0.9 * (1 - 0.5 * rho ** 2) ** 0.5, 1.0 + 2.5 * rho ** 2, m, 1.8)
    assert out["ood_fraction"] > 0.2 and out["ood_core"] == 0.0
    # cold ions (T_i/T_e below the box) are flagged as soon as they sit inside the radial range
    out = shrx.evaluate(rho, Te, 0.1 * Te, 0.1 + 0.9 * (1 - 0.5 * rho ** 2) ** 0.5, 1.0 + 2.5 * rho ** 2, m, 1.8)
    assert out["ood_core"] > 0.5
