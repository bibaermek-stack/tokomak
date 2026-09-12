"""Test suite for the tokamak package.

The validation module carries the checks against published physics; these
tests cover the things that should hold regardless of calibration -- units,
conservation, monotonicity, and that every selectable model actually runs.
"""

import numpy as np
import pytest

from tokamak import current as cur
from tokamak import disruption as disr
from tokamak import equilibrium as eqm
from tokamak import fuelcycle as fc
from tokamak import geometry as geo
from tokamak import radiation as rad
from tokamak import sol as sol_mod
from tokamak import transport as tr
from tokamak import validation as val
from tokamak.machines import get_machine, list_machines, custom_machine
from tokamak.reactivity import BOSCH_HALE_TABLE, sigma_v, dt_power_density
from tokamak.scalings import (CONFINEMENT, ConfinementInputs, LH_THRESHOLD,
                              KZ1, ThresholdInputs, tau_energy,
                              p_lh_threshold)
from tokamak.solver import Simulator, SolverConfig


# --- reactivity ------------------------------------------------------------
@pytest.mark.parametrize("T,ref", sorted(BOSCH_HALE_TABLE.items()))
def test_bosch_hale_table(T, ref):
    assert sigma_v(T) == pytest.approx(ref, rel=5e-3)


def test_reactivity_peaks_near_64_kev():
    T = np.linspace(5, 150, 2000)
    assert 55 < T[int(np.argmax(sigma_v(T)))] < 75


def test_reactivity_zero_below_threshold():
    assert sigma_v(0.05) == 0.0


# --- geometry --------------------------------------------------------------
def test_geometry_matches_iter():
    m = geo.attach_geometry(get_machine("iter"))
    assert m.V == pytest.approx(840.0, rel=0.01)
    assert m.S == pytest.approx(680.0, rel=0.01)
    assert m.kappa_a == pytest.approx(1.70, rel=0.02)


def test_circular_limit_reduces_to_torus():
    g = geo.shape_integrals(10.0, 1.0, 1.0, 0.0, diverted=False)
    assert g["V"] == pytest.approx(2 * np.pi ** 2 * 10.0, rel=1e-3)
    assert g["kappa_a"] == pytest.approx(1.0, rel=1e-3)


def test_trapped_fraction_monotonic_and_bounded():
    e = np.linspace(0.01, 0.9, 50)
    ft = np.array([geo.trapped_fraction(x) for x in e])
    assert np.all(np.diff(ft) > 0)
    assert ft.min() > 0 and ft.max() < 1


# --- machines --------------------------------------------------------------
def test_every_machine_has_sane_geometry():
    for key in list_machines():
        m = geo.attach_geometry(get_machine(key))
        assert m.V > 0 and m.S > 0
        assert 1.0 < m.aspect < 6.0
        assert m.n_greenwald > 0


def test_custom_machine_overrides_only_what_is_given():
    m = custom_machine(base="iter", B0=7.0, label="ITER-B7")
    assert m.B0 == 7.0
    assert m.R0 == get_machine("iter").R0
    assert m.label == "ITER-B7"


def test_unknown_machine_raises():
    with pytest.raises(KeyError):
        get_machine("not-a-tokamak")


# --- composition and radiation ---------------------------------------------
def test_charge_neutrality_and_zeff():
    c = rad.Composition().resolve(f_he=0.041)
    assert 0.75 < c["f_dt"] < 0.90
    assert 1.4 < c["z_eff"] < 1.9
    assert c["f_ion"] < 1.0


def test_more_impurity_means_more_dilution():
    base = rad.Composition().resolve(0.04)
    seeded = rad.Composition().with_seeding("Ne", 0.01).resolve(0.04)
    assert seeded["f_dt"] < base["f_dt"]
    assert seeded["z_eff"] > base["z_eff"]


def test_bremsstrahlung_scales_as_n_squared():
    a = rad.bremsstrahlung(1.0, 10.0, 1.6)
    b = rad.bremsstrahlung(2.0, 10.0, 1.6)
    assert b / a == pytest.approx(4.0, rel=1e-6)


def test_light_impurities_do_not_line_radiate_in_the_core():
    be = rad.SPECIES["Be"].cooling_rate(15.0)
    w = rad.SPECIES["W"].cooling_rate(15.0)
    assert be == 0.0
    assert w > 0.0


# --- scalings --------------------------------------------------------------
def test_every_confinement_model_runs_and_is_positive():
    x = ConfinementInputs(Ip=15, B0=5.3, P_loss=90, n_bar20=1.0,
                          R0=6.2, a=2.0, kappa_a=1.7)
    for key in CONFINEMENT:
        assert tau_energy(key, x) > 0


def test_every_lh_model_runs():
    th = ThresholdInputs(n_bar20=1.0, B0=5.3, S=680.0, R0=6.2, a=2.0)
    for key in LH_THRESHOLD:
        assert p_lh_threshold(key, th) > 0


def test_confinement_falls_with_power():
    kw = dict(Ip=15, B0=5.3, n_bar20=1.0, R0=6.2, a=2.0, kappa_a=1.7)
    lo = tau_energy("ipb98y2", ConfinementInputs(P_loss=50, **kw))
    hi = tau_energy("ipb98y2", ConfinementInputs(P_loss=150, **kw))
    assert hi < lo


def test_kz1_reduces_to_ipb98_at_conventional_aspect_ratio():
    x = ConfinementInputs(Ip=15, B0=5.3, P_loss=90, n_bar20=1.06,
                          R0=6.2, a=2.0, kappa_a=1.715)
    assert tau_energy("kz1", x) == pytest.approx(
        tau_energy("ipb98y2", x), rel=0.02)


def test_kz1_weight_is_monotonic_in_aspect_ratio():
    eps = np.linspace(0.1, 0.85, 40)
    w = np.array([KZ1.weight(e) for e in eps])
    assert np.all(np.diff(w) > 0)
    assert w[0] < 0.01 and w[-1] > 0.98


def test_kz1_field_dependence_is_stronger_for_a_spherical_torus():
    """The whole point of kz1: B matters far more at low aspect ratio."""
    def ratio(R0, a):
        kw = dict(Ip=1.0, P_loss=6.0, n_bar20=0.5, R0=R0, a=a,
                  kappa_a=2.0, a_mass=2.0)
        lo = tau_energy("kz1", ConfinementInputs(B0=0.5, **kw))
        hi = tau_energy("kz1", ConfinementInputs(B0=1.0, **kw))
        return hi / lo
    st = ratio(0.85, 0.67)          # eps = 0.79
    conv = ratio(6.2, 2.0)          # eps = 0.32
    assert st > 1.8
    assert conv < 1.2


# --- equilibrium -----------------------------------------------------------
def test_equilibrium_converges_and_is_grid_independent():
    a = eqm.solve(6.2, 2.0, 1.85, 0.45, 15.0, 5.3, nR=73, nZ=109)
    b = eqm.solve(6.2, 2.0, 1.85, 0.45, 15.0, 5.3, nR=113, nZ=169)
    assert a.converged and b.converged
    assert a.q95 == pytest.approx(b.q95, rel=0.03)
    assert a.li3 == pytest.approx(b.li3, rel=0.05)


def test_equilibrium_volume_matches_the_shape_integral():
    eq = eqm.solve(6.2, 2.0, 1.85, 0.45, 15.0, 5.3, nR=97, nZ=145)
    smooth = geo.shape_integrals(6.2, 2.0, 1.85, 0.45, diverted=False)["V"]
    assert eq.V == pytest.approx(smooth, rel=0.01)


def test_equilibrium_matches_requested_pressure_and_q0():
    eq = eqm.solve_matched(6.2, 2.0, 1.85, 0.45, 15.0, 5.3,
                           p_avg=2.59e5, q0=1.0, nR=73, nZ=109)
    assert eq.p_avg == pytest.approx(2.59e5, rel=0.03)
    assert eq.q0 == pytest.approx(1.0, rel=0.03)


def test_shafranov_shift_is_outboard_and_grows_with_beta():
    lo = eqm.solve_matched(6.2, 2.0, 1.85, 0.45, 15.0, 5.3, p_avg=1.0e5,
                           nR=73, nZ=109)
    hi = eqm.solve_matched(6.2, 2.0, 1.85, 0.45, 15.0, 5.3, p_avg=4.0e5,
                           nR=73, nZ=109)
    assert lo.shafranov >= 0
    assert hi.shafranov >= lo.shafranov


# --- transport -------------------------------------------------------------
def test_diffusion_conserves_energy_without_sources():
    g = tr.Grid(n=41, a=2.0, V_total=840.0)
    y = np.ones(g.n) * 5.0
    cap = np.ones(g.n)
    chi = np.ones(g.n) * 0.5
    zero = np.zeros(g.n)
    out = tr._diffuse(g, y, cap, chi, zero, zero, 0.1, 5.0)
    assert out == pytest.approx(y, rel=1e-8)


def test_pinch_produces_a_peaked_density():
    g = tr.Grid(n=81, a=2.0, V_total=840.0)
    n = np.ones(g.n) * 0.3
    zero = np.zeros(g.n)
    D = np.ones(g.n) * 0.4
    pin = tr.pinch_profile(g.rho)
    for _ in range(400):
        n = tr._diffuse(g, n, np.ones(g.n), D, zero, zero, 0.5, 0.3,
                        pinch=pin)
    assert n[0] / n[-1] == pytest.approx(np.exp(tr.PINCH_STRENGTH / 2.0),
                                         rel=0.05)


def test_density_weighted_average_is_exact_for_flat_profiles():
    g = tr.Grid(n=41)
    s = tr.PlasmaState(g, np.ones(g.n), np.ones(g.n) * 7.0,
                       np.ones(g.n) * 6.0, np.zeros(g.n))
    assert s.Te_avg == pytest.approx(7.0, rel=1e-9)
    assert s.ne_avg == pytest.approx(1.0, rel=1e-9)


def test_stored_energy_formula():
    g = tr.Grid(n=61, V_total=840.0)
    s = tr.PlasmaState(g, np.ones(g.n) * 1.0, np.ones(g.n) * 10.0,
                       np.ones(g.n) * 10.0, np.zeros(g.n))
    # W = 1.5 k (n Te + n_i Ti) V, with f_ion = 1 both channels are equal
    expected = 2.40327e-2 * 840.0 * (10.0 + 10.0)
    assert s.stored_energy(1.0) == pytest.approx(expected, rel=1e-3)


# --- current ---------------------------------------------------------------
def test_neoclassical_resistivity_exceeds_spitzer():
    rho = np.array([0.5])
    neo = cur.neoclassical_resistivity(rho, np.array([1.0]), np.array([10.0]),
                                       1.65, 6.2, 0.32, np.array([2.0]))[0]
    sp = cur.spitzer_resistivity(10.0, 1.65)
    assert 1.3 < neo / sp < 4.0


def test_bootstrap_vanishes_without_a_gradient():
    rho = np.linspace(0, 1, 41)
    flat = np.ones_like(rho)
    bs = cur.bootstrap(rho, flat, flat * 10, flat * 10, 1 + 2 * rho ** 2,
                       R0=6.2, a=2.0, B0=5.3, Ip=15.0, z_eff=1.65,
                       f_ion=0.87, A_cs=22.0)
    assert abs(bs.f_bs) < 1e-3


def test_bootstrap_grows_with_beta_poloidal():
    rho = np.linspace(0, 1, 61)
    n = 1.0 * (1 - 0.8 * rho ** 2)
    q = 1 + 2 * rho ** 2
    kw = dict(R0=6.2, a=2.0, B0=5.3, Ip=15.0, z_eff=1.65, f_ion=0.87,
              A_cs=22.0)
    lo = cur.bootstrap(rho, n, 5 * (1 - rho ** 2) + 0.5,
                       5 * (1 - rho ** 2) + 0.5, q, **kw).f_bs
    hi = cur.bootstrap(rho, n, 15 * (1 - rho ** 2) + 1.0,
                       15 * (1 - rho ** 2) + 1.0, q, **kw).f_bs
    assert hi > lo > 0


def test_flux_budget_exhausts():
    b = cur.FluxBudget(available=10.0)
    b.step(1.0, 5.0)
    assert not b.exhausted
    b.step(1.0, 6.0)
    assert b.exhausted and b.remaining < 0


# --- SOL -------------------------------------------------------------------
def test_lambda_q_falls_with_poloidal_field():
    assert sol_mod.eich_lambda_q(2.0) < sol_mod.eich_lambda_q(0.5)


def test_seeding_monotonically_reduces_target_load():
    kw = dict(R0=6.2, a=2.0, B0=5.3, B_pol=1.0, q95=3.0, n_sep20=0.3)
    loads = [sol_mod.two_point(90.0, f_rad_seed=f, **kw).q_target_MWm2
             for f in (0.0, 0.3, 0.6, 0.9)]
    assert all(b < a for a, b in zip(loads, loads[1:]))


def test_seeding_solver_meets_the_target():
    kw = dict(R0=6.2, a=2.0, B0=5.3, B_pol=1.0, q95=3.0, n_sep20=0.3)
    f = sol_mod.seeding_for_target(90.0, q_limit=10.0, **kw)
    st = sol_mod.two_point(90.0, f_rad_seed=f, **kw)
    assert st.q_target_MWm2 <= 10.5


# --- disruption ------------------------------------------------------------
def test_mitigation_suppresses_runaways():
    kw = dict(cause="t", Ip=15.0, R0=6.2, a=2.0, kappa=1.85, B0=5.3,
              W_thermal=325.0, li=0.9, ne20=1.0, z_eff=1.65, S_wall=680.0)
    un = disr.simulate(mitigated=False, **kw)
    mi = disr.simulate(mitigated=True, **kw)
    assert un.I_runaway > mi.I_runaway
    assert mi.q_tq_MJm2 < un.q_tq_MJm2


def test_avalanche_gain_grows_with_current():
    kw = dict(cause="t", R0=6.2, a=2.0, kappa=1.85, B0=5.3, W_thermal=325.0,
              li=0.9, ne20=1.0, z_eff=1.65, S_wall=680.0)
    small = disr.simulate(Ip=5.0, **kw).avalanche_gain_exp
    large = disr.simulate(Ip=15.0, **kw).avalanche_gain_exp
    assert large > small


def test_limit_checks_fire():
    assert disr.check_limits(f_greenwald=1.4, q95=3, beta_n=2, p_rad=1,
                             p_heat=10, Te_avg=5) is not None
    assert disr.check_limits(f_greenwald=0.8, q95=3, beta_n=2, p_rad=1,
                             p_heat=10, Te_avg=5) is None


# --- fuel cycle ------------------------------------------------------------
def test_tritium_burn_rate_scales_with_power():
    a = fc.blanket(500.0, 680.0).tritium_burn_rate
    b = fc.blanket(1000.0, 680.0).tritium_burn_rate
    assert b / a == pytest.approx(2.0, rel=1e-6)


def test_breeding_below_unity_never_doubles():
    b = fc.blanket(500.0, 680.0, tbr_design=1.0, coverage=0.8)
    assert b.TBR < 1.0
    assert b.doubling_time_days > 1e5


def test_recirculating_power_reduces_net_output():
    p = fc.plant_balance(500.0, 50.0, R0=6.2)
    assert p.P_net_electric < p.P_gross_electric
    assert p.P_recirculating > 0
    assert sum(p.breakdown.values()) == pytest.approx(p.P_recirculating,
                                                      rel=1e-9)


# --- solver ----------------------------------------------------------------
def test_simulator_runs_every_transport_model():
    for key in tr.TRANSPORT_MODELS:
        sim = Simulator(SolverConfig(machine="iter", transport_model=key,
                                     equilibrium=False,
                                     disruption_enabled=False, n_rho=33))
        for _ in range(60):
            sim._scenario_actuators()
            sim.step()
        d = sim.scalars()
        assert np.isfinite(d["Wth"]) and d["Wth"] >= 0
        assert np.isfinite(d["Te"]) and d["Te"] > 0


def test_simulator_runs_on_a_spherical_torus():
    sim = Simulator(SolverConfig(machine="nstx", confinement_model="kz1",
                                 equilibrium=False, n_rho=33,
                                 disruption_enabled=False))
    for _ in range(80):
        sim._scenario_actuators()
        sim.step()
    assert np.isfinite(sim.scalars()["Wth"])


def test_density_control_tracks_its_set_point():
    sim = Simulator(SolverConfig(machine="iter", equilibrium=False,
                                 disruption_enabled=False, n_rho=49))
    while sim.t < 60:
        sim._scenario_actuators()
        sim.step()
    d = sim.scalars()
    assert 0.7 < d["fG"] < 1.0


# --- validation module -----------------------------------------------------
def test_validation_report_passes():
    r = val.report()
    failures = [c["name"] for c in r["checks"] if not c["passed"]]
    assert failures == [], f"failing checks: {failures}"


def test_validation_states_its_limitations():
    assert len(val.NOT_VALIDATED) >= 5
