"""Run Google DeepMind TORAX (QLKNN) on the same ITER flat-top SINTEZ simulates.

Everything that is *not* the transport model is pinned to what SINTEZ ended up with
at t = 120 s in its neural-surrogate run:

  * machine: R0 = 6.2 m, a = 2.0 m, B0 = 5.3 T, Ip = 15 MA (ITER shape, CHEASE file shipped with TORAX)
  * density profile n_e(rho): prescribed, not evolved
  * pedestal top T_ped and n_ped at rho = 0.93
  * 50 MW auxiliary heating, Z_eff of the SINTEZ run (Ne impurity), D-T 50:50

    python run_torax.py <sintez_neural_json> <out.json> [--t-final 30] [--no-sawteeth]
                        [--line-radiation-fraction F]

Needs a separate environment with `pip install torax==1.4.3` (see README.md).
"""
import argparse
import json
import time

import numpy as np
import torax

ap = argparse.ArgumentParser()
ap.add_argument("sintez_json")
ap.add_argument("out_json")
ap.add_argument("--t-final", type=float, default=30.0)
ap.add_argument("--no-sawteeth", action="store_true",
                help="leave out TORAX's sawtooth model (q0 then falls to ~0.4: sensitivity only)")
ap.add_argument("--line-radiation-fraction", type=float, default=None,
                help="replace the Mavrin Ne radiation by a flat sink taking this fraction of the heating power")
args = ap.parse_args()

sintez = json.load(open(args.sintez_json))
rho_s = np.array(sintez["profiles"]["rho"])
ne_s = np.array(sintez["profiles"]["ne"]) * 1e20          # m^-3
last = sintez["log"][-1]
T_ped = float(last["Tped"])
n_ped = float(ne_s[int(round(0.93 * (len(rho_s) - 1)))])
Zeff = float(last["Zeff"])
Ip = float(last["Ip"]) * 1e6
P_aux = float(last["Paux"]) * 1e6
split = sintez["split"]
# electron share of the auxiliary power: NBI 33 MW + ICRF 17 MW, with SINTEZ's slowing-down split
f_e = (33.0 * (1 - split["nbi"]) + 17.0 * (1 - split["icrf"])) / 50.0

idx = np.arange(0, len(rho_s), 2)
n_e_profile = {float(rho_s[i]): float(ne_s[i]) for i in idx}

if args.line_radiation_fraction is None:
    impurity_radiation = {"model_name": "mavrin_fit"}
else:
    impurity_radiation = {"model_name": "P_in_scaled_flat_profile",
                          "fraction_P_heating": args.line_radiation_fraction}

CONFIG = {
    "plasma_composition": {"main_ion": {"D": 0.5, "T": 0.5}, "impurity": "Ne", "Z_eff": Zeff},
    "profile_conditions": {
        "Ip": Ip,
        "T_i": {0.0: {0.0: 12.0, 1.0: 0.12}}, "T_i_right_bc": 0.12,
        "T_e": {0.0: {0.0: 12.0, 1.0: 0.12}}, "T_e_right_bc": 0.12,
        "n_e_right_bc": float(ne_s[-1]),
        "n_e_nbar_is_fGW": False,
        "normalize_n_e_to_nbar": False,
        "n_e": {0: n_e_profile},
    },
    "numerics": {
        "t_final": args.t_final,
        # speeds the current diffusion up; it also multiplies ohmic power, so no ohmic source below
        "resistivity_multiplier": 200,
        "evolve_ion_heat": True, "evolve_electron_heat": True,
        "evolve_current": True,
        "evolve_density": False,          # density is prescribed from SINTEZ
        "max_dt": 0.5, "chi_timestep_prefactor": 50, "dt_reduction_factor": 3,
    },
    "geometry": {"geometry_type": "chease", "geometry_file": "iterhybrid.mat2cols",
                 "Ip_from_parameters": True, "R_major": 6.2, "a_minor": 2.0, "B_0": 5.3},
    "neoclassical": {"bootstrap_current": {"bootstrap_multiplier": 1.0}},
    "sources": {
        "generic_current": {"fraction_of_total_current": 0.0,
                            "gaussian_width": 0.1, "gaussian_location": 0.3},
        # NBI (33 MW, broad) + ICRF (17 MW, central) lumped into one broad source
        "generic_heat": {"gaussian_location": 0.15, "gaussian_width": 0.30,
                         "P_total": P_aux, "electron_heat_fraction": f_e},
        "fusion": {},
        "ei_exchange": {"Qei_multiplier": 1.0},
        "bremsstrahlung": {},
        "cyclotron_radiation": {},
        "impurity_radiation": impurity_radiation,
    },
    "pedestal": {"model_name": "set_T_ped_n_ped", "set_pedestal": True,
                 "T_i_ped": T_ped, "T_e_ped": T_ped, "n_e_ped": n_ped,
                 "rho_norm_ped_top": 0.93},
    "transport": {
        "model_name": "qlknn",
        # TORAX's own ad-hoc core / edge patches, as in its iterhybrid example
        "apply_inner_patch": True, "D_e_inner": 0.25, "V_e_inner": 0.0,
        "chi_i_inner": 1.0, "chi_e_inner": 1.0, "rho_inner": 0.2,
        "apply_outer_patch": True, "D_e_outer": 0.1, "V_e_outer": 0.0,
        "chi_i_outer": 2.0, "chi_e_outer": 2.0, "rho_outer": 0.9,
        "chi_min": 0.05, "chi_max": 100, "D_e_min": 0.05,
        "DV_effective": True,
        "include_ITG": True, "include_TEM": True, "include_ETG": True,
        "avoid_big_negative_s": True, "An_min": 0.05,
        "ITG_flux_ratio_correction": 1,
    },
    "solver": {"solver_type": "linear", "use_predictor_corrector": True,
               "n_corrector_steps": 1, "chi_pereverzev": 30, "D_pereverzev": 15,
               "use_pereverzev": True},
    "time_step_calculator": {"calculator_type": "chi"},
}
if not args.no_sawteeth:
    CONFIG["mhd"] = {"sawtooth": {
        "trigger_model": {"model_name": "simple", "minimum_radius": 0.05, "s_critical": 0.2},
        "redistribution_model": {"model_name": "simple", "flattening_factor": 1.01,
                                 "mixing_radius_multiplier": 1.5},
        "crash_step_duration": 1e-3,
    }}

t0 = time.time()
tc = torax.ToraxConfig.from_dict(CONFIG)
data_tree, hist = torax.run_simulation(tc)
wall = time.time() - t0
data_tree.to_netcdf(args.out_json.replace(".json", ".nc"))

json.dump({"wall_s": wall,
           "config_summary": {"Ip_A": Ip, "P_aux_W": P_aux, "T_ped_keV": T_ped, "n_ped_m3": n_ped,
                              "Zeff": Zeff, "electron_heat_fraction": f_e,
                              "sawteeth": not args.no_sawteeth,
                              "line_radiation_fraction": args.line_radiation_fraction,
                              "torax_version": torax.__version__}},
          open(args.out_json, "w"))
print("done, wall", round(wall, 1), "s")
