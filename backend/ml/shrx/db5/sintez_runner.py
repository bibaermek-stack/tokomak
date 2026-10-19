"""One database record -> one SINTEZ discharge to steady state.

The record fixes the machine (R, a, kappa, delta, B, Ip, isotope mass), the line-averaged density, the Z_eff and the
power crossing the separatrix (the database's thermal loss power PLTH).  Everything else -- the pedestal, the profiles, the
stored energy -- is SINTEZ's own prediction with the chosen transport model.  H-mode is imposed (the records are H-modes;
the question here is what confinement the model predicts *given* H-mode, not whether it reaches it).

Importable (``run_record``) and runnable:

    python sintez_runner.py <records.csv> <out.json> <transport_model> [--workers 4] [--t-end 12] [--calibrate-pedestal]
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tokamak import radiation as rad  # noqa: E402
from tokamak import transport as tr  # noqa: E402
from tokamak.machines import custom_machine  # noqa: E402
from tokamak.scalings import greenwald_density  # noqa: E402
from tokamak.solver import Simulator, SolverConfig  # noqa: E402

Z_IMP = 4.0                      # Be-like impurity carries the Z_eff the record gives (f = (Zeff - 1) / (Z (Z - 1)))
T_START, T_AVG = 4.5, 1.5        # the solver allows H-mode from t > 4 s; average the last T_AVG seconds


def machine_from_record(r, shot_id: str):
    ratio = 1.0 / 1.09
    return custom_machine(
        key=shot_id, label=shot_id, R0=float(r.RGEO), a=float(r.AMIN), kappa_x=float(r.KAPPA), delta_x=float(r.DELTA),
        kappa_95=float(r.KAPPA) * ratio, delta_95=float(r.DELTA) * 0.73, B0=abs(float(r.BT)), Ip=abs(float(r.IP)) / 1e6,
        a_mass=float(r.MEFF), fuel="D", superconducting=False, diverted=True, p_nbi=float(r.PLTH) / 1e6, p_icrf=0.0, p_ecrf=0.0,
        e_nbi_kev=float(r.ENBI) if np.isfinite(getattr(r, "ENBI", np.nan)) and r.ENBI > 10 else 80.0)


class ShotSimulator(Simulator):
    """The solver's scripted ITER discharge replaced by a flat-top at the record's parameters."""

    def __init__(self, record, transport_model, n_rho=65, t_ped_fixed=None):
        self.rec = record
        self.t_ped_fixed = t_ped_fixed
        m = machine_from_record(record, f"{record.TOK}-{int(record.SHOT)}")
        cfg = SolverConfig(machine="iter", transport_model=transport_model, disruption_enabled=False, mhd_enabled=True,
                           eq_interval=3.0, n_rho=n_rho)
        super().__init__(cfg, machine=m)
        zeff = float(record.ZEFF) if np.isfinite(getattr(record, "ZEFF", np.nan)) and record.ZEFF > 1.0 else 2.0
        f_imp = min((zeff - 1.0) / (Z_IMP * (Z_IMP - 1.0)), 0.08)
        self.comp = rad.Composition({"Be": f_imp})
        self.P_target = float(record.PLTH) / 1e6
        self.n_target20 = float(record.NEL) / 1e20
        # start the flat-top already at full current, density and a warm plasma
        self.t = T_START
        self.Ip = m.Ip
        self.act.Ip_request = m.Ip
        self.state = tr.initial_state(self.grid, ne0=1.15 * self.n_target20, Te0=1.5, Ti0=1.5)
        self.n_sep = 0.25 * self.n_target20
        self._derive()

    def _derive(self):
        super()._derive()
        self.P_lh = 0.0                      # H-mode imposed

    def _update_pedestal(self, dt):
        """The EPED-like prediction, unless a pedestal-top temperature is imposed (the 'core only' test)."""
        if self.t_ped_fixed is None:
            return super()._update_pedestal(dt)
        self.ped = None
        tau_ped = max(0.3 * self.tau_E, 0.05)
        self.T_ped += (self.t_ped_fixed - self.T_ped) * min(dt / tau_ped, 1.0)

    def _scenario_actuators(self):
        m, dt = self.m, self.cfg.dt
        self.act.Ip_request = m.Ip
        self.act.p_nbi, self.act.p_icrf, self.act.p_ecrf = self.P_target, 0.0, 0.0
        err = self.n_target20 - self.state.n_bar          # density feedback on the line average (what the record gives)
        n_G = greenwald_density(self.Ip, m.a)
        self.n_sep = float(np.clip(self.n_sep + 1.2 * err * dt, 0.005, 0.85 * max(n_G, 0.01)))
        self.act.gas = float(np.clip(4.0 * err, 0.0, 1.0))
        self.n_target = self.n_target20


def _burn(record, transport_model, t_end, p_target, t_ped_fixed=None):
    sim = ShotSimulator(record, transport_model, t_ped_fixed=t_ped_fixed)
    if t_ped_fixed is not None:
        sim.T_ped = t_ped_fixed
    sim.P_target = p_target
    acc, n = {}, 0
    while sim.t < T_START + t_end:
        sim._scenario_actuators()
        sim.step()
        if sim.t > T_START + t_end - T_AVG:
            for k, v in sim.scalars().items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    acc[k] = acc.get(k, 0.0) + v
            n += 1
    return sim, {k: v / max(n, 1) for k, v in acc.items()}


def _summary(sim, d, out):
    s = sim.state
    out.update(ok=True, Wth_MJ=d["Wth"], tauE=d["tauE"], Te0=d["Te0"], Ti0=d["Ti0"], Te_avg=d["Te"], Ti_avg=d["Ti"],
               n_bar=d["n_bar"], P_loss=d["Ploss"], P_rad=d["Prad"], P_aux=d["Paux"], T_ped=d["Tped"], q95=d["q95"],
               betaN=d["betaN"], f_bs=d["fBS"], H98_model=d["H_factor"], t_sim=float(sim.t),
               shrx_ood=d.get("shrx_ood", 0.0), h_mode=bool(sim.h_mode),
               split_nbi=float(sim.split["nbi"]), kappa_a=float(sim.m.kappa_a), V=float(sim.m.V),
               profile={"rho": sim.grid.rho.tolist(), "ne": s.ne.tolist(), "Te": s.Te.tolist(), "Ti": s.Ti.tolist(),
                        "q": sim.q_prof.tolist()}, n_ped=float(s.ne[int(round(0.93 * (sim.grid.n - 1)))]))


def run_record_calibrated(args):
    """'Core only': impose the pedestal-top temperature that reproduces the measured thermal stored energy.

    A secant search on log T_ped (stored energy rises monotonically with the pedestal).  What is then left to compare with
    the measurement is the shape of the core profile -- the central temperatures -- which is what the core transport
    model alone decides."""
    record, transport_model, t_end = args
    record = SimpleNamespace(**record)
    t0 = time.time()
    out = {"TOK": record.TOK, "SHOT": int(record.SHOT), "ok": False}
    try:
        target_w = float(record.WTH) / 1e6
        p_target = float(record.PLTH) / 1e6
        # radiation correction as in run_record, from a first predictive pass
        _, d0 = _burn(record, transport_model, t_end, p_target)
        p_aux = p_target + d0["Prad"]
        x_lo, x_hi, f_lo, f_hi = None, None, None, None
        x = np.log(max(d0["Tped"], 0.05))
        for it in range(7):
            sim, d = _burn(record, transport_model, t_end, p_aux, t_ped_fixed=float(np.exp(x)))
            f = np.log(d["Wth"] / target_w)
            if abs(f) < 0.02:
                break
            if f > 0:
                x_hi, f_hi = x, f
            else:
                x_lo, f_lo = x, f
            if x_lo is not None and x_hi is not None:
                x = x_lo + (x_hi - x_lo) * (-f_lo) / (f_hi - f_lo)
            else:
                x = x + (-f) * 1.6                       # stored energy ~ T_ped^0.6 near the stiff edge: overshoot a little
            x = float(np.clip(x, np.log(0.03), np.log(40.0)))
        _summary(sim, d, out)
        out.update(calibrated=True, calibration_iterations=it + 1, calibration_residual=float(f), T_ped_calibrated=float(np.exp(x)))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["wall_s"] = time.time() - t0
    return out


def run_record(args):
    record, transport_model, t_end = args
    record = SimpleNamespace(**record)             # plain dicts cross the process boundary, namespaces read nicely
    t0 = time.time()
    out = {"TOK": record.TOK, "SHOT": int(record.SHOT), "TIME": float(record.TIME), "ok": False}
    try:
        # The record gives the power that crosses the separatrix.  SINTEZ radiates some of the heating on its way out, so
        # a first pass measures that and a second pass adds it back: the conducted power then equals the record's PLTH.
        target = float(record.PLTH) / 1e6
        sim, d = _burn(record, transport_model, t_end, target)
        out["P_aux_first_pass"] = target
        sim, d = _burn(record, transport_model, t_end, target + d["Prad"])
        _summary(sim, d, out)
    except Exception as exc:                                    # one bad record must not kill the batch
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["wall_s"] = time.time() - t0
    return out


if __name__ == "__main__":
    csv, out_path, tm = sys.argv[1:4]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    t_end = float(sys.argv[sys.argv.index("--t-end") + 1]) if "--t-end" in sys.argv else 12.0
    fn = run_record_calibrated if "--calibrate-pedestal" in sys.argv else run_record
    df = pd.read_csv(csv)
    jobs = [(r, tm, t_end) for r in df.to_dict("records")]
    with Pool(workers) as pool:
        res = pool.map(fn, jobs, chunksize=1)
    Path(out_path).write_text(json.dumps(res))
    ok = [r for r in res if r["ok"]]
    print(f"{tm}: {len(ok)}/{len(res)} runs ok, mean wall {np.mean([r['wall_s'] for r in res]):.1f} s")
    for r in res:
        if not r["ok"]:
            print("  failed:", r["TOK"], r["SHOT"], r.get("error"))
