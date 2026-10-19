"""Pick the ITPA H-mode confinement database records used for the validation.

    python select_records.py <hdb5.csv(.gz)> <out.csv> [--per-machine 14] [--seed 7]

Source of the data: the public HDB5 sample shipped with FUSE.jl (ProjectTorreyPines/FUSE.jl, sample/HDB5_compressed.csv,
Apache-2.0), 14 153 records of the ITPA global H-mode confinement database.  Selection (all must hold):

  * an ELMy / ELM-free H-mode phase (PHASE starts with H), no internal transport barrier, no hybrid flag
  * quasi-steady: |dW/dt| <= 15 % of the loss power
  * every input the simulation needs is present: R, a, kappa, delta, B, Ip, line-averaged density, thermal loss power
    PLTH, thermal stored energy WTH, isotope mass
  * conventional aspect ratio (R/a >= 1.5): spherical tokamaks are treated separately (validate_st)
  * not more than --per-machine records per device (a random draw, fixed seed), so JET's 6 000 records do not drown the rest

The loss power PLTH is the one the database's own H98(y,2) uses (our IPB98(y,2) reproduces HIPB98Y2 with it).
"""
import sys

import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
per = int(sys.argv[sys.argv.index("--per-machine") + 1]) if "--per-machine" in sys.argv else 14
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 7
spherical = "--spherical" in sys.argv

df = pd.read_csv(src, low_memory=False)
need = ["RGEO", "AMIN", "KAPPA", "DELTA", "BT", "IP", "NEL", "PLTH", "WTH", "MEFF", "TAUTH"]
ok = df.dropna(subset=need)
ok = ok[ok.PHASE.astype(str).str.startswith("H")]
ok = ok[~ok.ITB.astype(str).str.upper().isin(["YES", "Y", "TRUE"])]
ok = ok[ok.HYBRID.astype(str).str.upper().isin(["UNKNOWN", "NO", "N", "NA", "H"])]
steady = np.abs(ok.DWDIA.fillna(0.0)) <= 0.15 * ok.PLTH
ok = ok[steady & (ok.PLTH > 1e5) & (ok.WTH > 0)]
aspect = ok.RGEO / ok.AMIN
ok = ok[aspect <= 1.5] if spherical else ok[aspect >= 1.5]
if not spherical:
    ok = ok[~ok.TOK.isin(["TEXTOR", "MAST", "NSTX", "START"])]     # limiter / spherical devices are not part of this set
rng = np.random.default_rng(seed)
parts = []
for tok, g in ok.groupby("TOK"):
    take = g.sample(min(per, len(g)), random_state=int(rng.integers(1 << 30)))
    parts.append(take)
sel = pd.concat(parts).sort_values(["TOK", "SHOT"])
cols = ["TOK", "SHOT", "TIME", "PHASE", "ELMTYPE", "STANDARD", "RGEO", "AMIN", "KAPPA", "KAPPAA", "DELTA", "BT", "IP", "NEL",
        "MEFF", "ZEFF", "PLTH", "PL", "PINJ", "PRAD", "POHM", "WTH", "TAUTH", "DWDIA", "TE0", "TI0", "ENBI", "HIPB98Y2", "AUXHEAT",
        "PNBI", "PICRH", "PECRH"]
sel[[c for c in cols if c in sel]].to_csv(out, index=False)
print(len(sel), "records:", sel.TOK.value_counts().to_dict())
