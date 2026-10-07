"""Tests for ML endpoints and predictors."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_ml_status():
    resp = client.get("/ml/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "synthetic_model_available" in data
    assert "jtext_real_model_available" in data
    assert "caveats" in data


def test_disruption_risk_prediction():
    payload = {
        "B": 3.0,
        "Ip": 8.0,
        "P_heat": 20.0,
        "ne19": 15.0,
        "Ti_kev": 8.0,
        "h_mode": True,
    }
    resp = client.post("/ml/disruption-risk", json=payload)
    if resp.status_code == 503:
        pytest.skip("model not trained (artefacts are gitignored)")
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["probability"] <= 1.0
    assert "caveat" in data
    assert data["inputs"]["B"] == 3.0


def test_disruption_risk_invalid_inputs():
    # Negative B
    resp = client.post("/ml/disruption-risk", json={
        "B": -1.0, "Ip": 8.0, "P_heat": 20.0, "ne19": 15.0, "Ti_kev": 8.0
    })
    assert resp.status_code == 422


def test_equilibrium_neural_prediction():
    # 367 dummy magnetic features
    features = [0.1] * 367
    resp = client.post("/ml/equilibrium-neural", json={"features": features})
    if resp.status_code == 503:
        pytest.skip("model not trained (artefacts are gitignored)")
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    pred = data["prediction"]
    assert "r_axis" in pred
    assert "z_axis" in pred
    assert "q95" in pred
    assert "beta_n" in pred
    assert "caveat" in data


def test_equilibrium_neural_wrong_dimension():
    # Wrong number of features (e.g. 10 instead of 367)
    resp = client.post("/ml/equilibrium-neural", json={"features": [0.1] * 10})
    assert resp.status_code == 400


def test_transport_surrogate_prediction():
    payload = {
        "rho": 0.4,
        "r_lt": 7.5,
        "r_ln": 2.2,
        "q": 1.5,
        "Te": 6.0,
        "ne": 0.9,
        "Ti_over_Te": 1.1,
        "B0": 5.3,
        "R0": 6.2,
        "a": 2.0,
        "h_mode": True,
    }
    resp = client.post("/ml/transport-surrogate", json=payload)
    if resp.status_code == 503:
        pytest.skip("model not trained (artefacts are gitignored)")
    assert resp.status_code == 200
    data = resp.json()
    assert "chi_thermal_diffusivity_m2_s" in data
    assert 0.05 <= data["chi_thermal_diffusivity_m2_s"] <= 15.0
    assert "DeepMind TORAX" in data["model"]


def test_simulator_runs_neural_surrogate_transport():
    from tokamak.solver import Simulator, SolverConfig
    sim = Simulator(SolverConfig(machine="iter", transport_model="neural_surrogate",
                                 equilibrium=False, disruption_enabled=False, n_rho=33))
    for _ in range(10):
        sim._scenario_actuators()
        sim.step()
    assert sim.t > 0.0
    scalars = sim.scalars()
    assert scalars["Te"] > 0.0


