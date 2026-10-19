"""Helpers that need the TORAX environment (JAX): frozen states and TORAX's own QLKNN inputs.

Used only to *validate* and *train* SHRX; SHRX itself runs on NumPy alone.
"""
import numpy as np
import torax
from torax._src.transport_model import qlknn_transport_model as qt

R0, A_MIN, B0 = 6.2, 2.0, 5.3


def frozen_config(src: dict, nu: float = 1.2) -> dict:
    """TORAX config that holds a SINTEZ profile (src = sintez_*.json) fixed, patches off."""
    P = src["profiles"]
    rho = np.array(P["rho"])
    last = src["log"][-1]
    idx = np.arange(0, len(rho), 2)
    d = lambda y: {float(rho[i]): float(y[i]) for i in idx}
    Te, Ti, ne = np.array(P["Te"]), np.array(P["Ti"]), np.array(P["ne"]) * 1e20
    return {
        "plasma_composition": {"main_ion": {"D": 0.5, "T": 0.5}, "impurity": "Ne", "Z_eff": float(last["Zeff"])},
        "profile_conditions": {
            "Ip": float(last["Ip"]) * 1e6,
            "T_i": {0.0: d(Ti)}, "T_i_right_bc": float(Ti[-1]),
            "T_e": {0.0: d(Te)}, "T_e_right_bc": float(Te[-1]),
            "n_e": {0: d(ne)}, "n_e_right_bc": float(ne[-1]),
            "n_e_nbar_is_fGW": False, "normalize_n_e_to_nbar": False,
            "initial_psi_mode": "j", "initial_j_is_total_current": True, "current_profile_nu": nu,
        },
        "numerics": {"t_final": 0.02, "fixed_dt": 0.01, "evolve_ion_heat": False,
                     "evolve_electron_heat": False, "evolve_current": False, "evolve_density": False},
        "geometry": {"geometry_type": "chease", "geometry_file": "iterhybrid.mat2cols",
                     "Ip_from_parameters": True, "R_major": R0, "a_minor": A_MIN, "B_0": B0},
        "sources": {},
        "pedestal": {"model_name": "set_T_ped_n_ped", "set_pedestal": False},
        "transport": {"model_name": "qlknn", "apply_inner_patch": False, "apply_outer_patch": False,
                      "chi_min": 0.05, "chi_max": 100, "D_e_min": 0.05, "DV_effective": True,
                      "include_ITG": True, "include_TEM": True, "include_ETG": True,
                      "avoid_big_negative_s": True, "An_min": 0.05, "ITG_flux_ratio_correction": 1},
        "solver": {"solver_type": "linear"},
        "time_step_calculator": {"calculator_type": "fixed"},
    }


def run_state(cfg: dict):
    """Run a (short) TORAX config; return (config, last core_profiles, last geometry)."""
    tc = torax.ToraxConfig.from_dict(cfg)
    _, hist = torax.run_simulation(tc, progress_bar=False)
    return tc, hist._core_profiles[-1], hist._geometries[-1]


def torax_inputs(tc, cp, geo):
    """TORAX's own QLKNN input matrix (N, 10), the x-rescale applied, plus the raw QualikizInputs."""
    import dataclasses
    tm = tc.transport.build_transport_model()
    tp = tc.transport.build_runtime_params(0.0)
    qi = tm._prepare_qualikiz_inputs(transport=tp, geo=geo, core_profiles=cp, poloidal_velocity_multiplier=1.0)
    qi = dataclasses.replace(qi, x=qi.x * qi.epsilon[-1] / qt._EPSILON_NN)
    model = qt.get_model("", "qlknn_7_11_v1")
    X = model.get_model_inputs_from_qualikiz_inputs(qi)
    return np.asarray(X), qi, model
