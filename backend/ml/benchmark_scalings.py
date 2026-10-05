"""Check the project's confinement scalings against real multi-machine data.

    python -m ml.benchmark_scalings      # run from backend/

For each row of the Zenodo multi-machine table that has geometry, B, Ip,
loss power, line-averaged density and stored energy, the measured
tau_E = W / P_loss is compared with every scaling in
``tokamak.scalings``.  Rows are *not* used to fit anything: with this few
usable rows (and many duplicates) a fit would be noise, so this is an
external validation set.

Caveats stated up front: the table's elongation is the shape elongation,
used here for kappa_a; P_loss is P_sol where given, else P_in (so
radiative losses are not removed); rows with density missing or zero are
skipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tokamak.scalings import CONFINEMENT, ConfinementInputs, tau_energy

from .kaggle_fetch import RAW

CSV = (RAW / "multi-machine-fusion-detachment-and-fueling-database"
       / "puff_data_v3_Zenodo.csv")


def load_rows() -> pd.DataFrame:
    d = pd.read_csv(CSV).iloc[:, :58]
    d.columns = [c.split(" [")[0] for c in d.columns]
    d["Ploss"] = d["P_sol"].fillna(d["P_in"])
    need = ["W_E", "Ploss", "Bt", "Ip", "ne_avg", "Major radius",
            "Minor radius", "Elongation k"]
    d = d.dropna(subset=need)
    d = d[(d.Ploss > 0) & (d.Ip > 0) & (d.Bt > 0) & (d.ne_avg > 0)
          & (d.W_E > 0)]
    d = d.drop_duplicates(subset=["Machine", "Bt", "Ip", "Ploss", "ne_avg",
                                  "W_E"])
    d["tau_meas"] = d["W_E"] / d["Ploss"] / 1000.0   # kJ / MW = ms -> s
    return d.reset_index(drop=True)


def evaluate(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in d.iterrows():
        x = ConfinementInputs(Ip=r.Ip, B0=r.Bt, P_loss=r.Ploss,
                              n_bar20=r.ne_avg / 1e20, R0=r["Major radius"],
                              a=r["Minor radius"], kappa_a=r["Elongation k"])
        row = {"machine": r.Machine, "tau_meas_s": r.tau_meas}
        for k in CONFINEMENT:
            row[k] = tau_energy(k, x) / r.tau_meas        # predicted / measured
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    d = load_rows()
    res = evaluate(d)
    keys = list(CONFINEMENT)
    print(f"usable independent rows: {len(res)} "
          f"({res.machine.nunique()} machines: "
          f"{', '.join(sorted(res.machine.unique()))})")
    print("\npredicted / measured tau_E   (1.00 = perfect)")
    print(f"{'model':10s} {'median':>7s} {'log-RMS':>8s} {'within 30%':>11s}")
    for k in keys:
        ratio = res[k].to_numpy()
        lrms = float(np.sqrt(np.mean(np.log(ratio) ** 2)))
        print(f"{k:10s} {np.median(ratio):7.2f} {lrms:8.2f} "
              f"{np.mean(np.abs(ratio - 1) < 0.3):11.0%}")
    print("\nby machine (median ratio):")
    print(res.groupby("machine")[keys].median().round(2).to_string())


if __name__ == "__main__":
    main()
