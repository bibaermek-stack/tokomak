"""Evaluate TORAX's QLKNN on a *frozen* plasma state taken from a SINTEZ run.

Nothing is evolved: T_e, T_i, n_e are SINTEZ's profiles at t = 120 s and the
current profile peaking (env NU, default 1.2) puts q0 ~ 1 and q95 ~ 3.3.  We read back
the turbulent chi_e / chi_i QLKNN assigns to exactly that state.

Usage: NU=1.2 python torax_frozen.py <sintez_run.json> <out.json>
"""
import json
import os
import sys

import numpy as np
import torax

src = json.load(open(sys.argv[1]))
out_path = sys.argv[2]

P = src["profiles"]
rho = np.array(P["rho"])
Te, Ti = np.array(P["Te"]), np.array(P["Ti"])
ne = np.array(P["ne"]) * 1e20
q_s = np.array(P["q"])
last = src["log"][-1]
Zeff = float(last["Zeff"])
Ip = float(last["Ip"]) * 1e6

R0, a, B0 = 6.2, 2.0, 5.3
MU0 = 4e-7 * np.pi
# circular-cross-section current density consistent with the SINTEZ q profile:
#   q = r B0 / (R0 B_theta),  I(r) = 2 pi r B_theta / mu0  ->  j = (1/2 pi r) dI/dr
r = rho * a
I_enc = 2 * np.pi * r ** 2 * B0 / (MU0 * R0 * np.maximum(q_s, 0.2))
j = np.gradient(I_enc, r) / (2 * np.pi * np.maximum(r, 1e-3))
j[0] = j[1]
j = np.maximum(j, 0.0)

idx = np.arange(0, len(rho), 2)
as_dict = lambda y: {float(rho[i]): float(y[i]) for i in idx}

CONFIG = {
    'plasma_composition': {'main_ion': {'D': 0.5, 'T': 0.5}, 'impurity': 'Ne', 'Z_eff': Zeff},
    'profile_conditions': {
        'Ip': Ip,
        'T_i': {0.0: as_dict(Ti)}, 'T_i_right_bc': float(Ti[-1]),
        'T_e': {0.0: as_dict(Te)}, 'T_e_right_bc': float(Te[-1]),
        'n_e': {0: as_dict(ne)}, 'n_e_right_bc': float(ne[-1]),
        'n_e_nbar_is_fGW': False, 'normalize_n_e_to_nbar': False,
        'initial_psi_mode': 'j',
        'initial_j_is_total_current': True,
        'current_profile_nu': float(os.environ.get('NU', '1.5')),
    },
    'numerics': {
        't_final': 0.02, 'fixed_dt': 0.01,
        'evolve_ion_heat': False, 'evolve_electron_heat': False,
        'evolve_current': False, 'evolve_density': False,
    },
    'geometry': {'geometry_type': 'chease', 'geometry_file': 'iterhybrid.mat2cols',
                 'Ip_from_parameters': True, 'R_major': R0, 'a_minor': a, 'B_0': B0},
    'sources': {},
    'pedestal': {'model_name': 'set_T_ped_n_ped', 'set_pedestal': False},
    'transport': {
        'model_name': 'qlknn',
        'apply_inner_patch': False, 'apply_outer_patch': False,
        'chi_min': 0.05, 'chi_max': 100, 'D_e_min': 0.05,
        'DV_effective': True,
        'include_ITG': True, 'include_TEM': True, 'include_ETG': True,
        'avoid_big_negative_s': True, 'An_min': 0.05,
        'ITG_flux_ratio_correction': 1,
    },
    'solver': {'solver_type': 'linear'},
    'time_step_calculator': {'calculator_type': 'fixed'},
}

tc = torax.ToraxConfig.from_dict(CONFIG)
data_tree, hist = torax.run_simulation(tc)
pr = data_tree['/profiles'].to_dataset().isel(time=0)
res = {
    'rho_face': pr['rho_face_norm'].values.tolist(),
    'rho_cell': pr['rho_norm'].values.tolist(),
    'chi_turb_e': pr['chi_turb_e'].values.tolist(),
    'chi_turb_i': pr['chi_turb_i'].values.tolist(),
    'q_torax': pr['q'].values.tolist(),
    'smag_torax': pr['magnetic_shear'].values.tolist(),
    'T_e': pr['T_e'].values.tolist(), 'T_i': pr['T_i'].values.tolist(),
    'n_e': pr['n_e'].values.tolist(),
}
json.dump(res, open(out_path, 'w'))
print('ok', 'q0_torax', float(pr['q'].values[0]), 'q95_torax', float(data_tree['/scalars']['q95'].values[0]), 'q95_sintez', last['q95'])
