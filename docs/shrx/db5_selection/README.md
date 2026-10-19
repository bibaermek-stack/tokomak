# Records of the ITPA H-mode confinement database used for the validation

Source: the public HDB5 sample shipped with [FUSE.jl](https://github.com/ProjectTorreyPines/FUSE.jl)
(`sample/HDB5_compressed.csv`, Apache-2.0) — 14 153 records of the ITPA global H-mode confinement database (DB5).
The data belong to the ITPA Confinement Database and Modelling Topical Group; cite the database and its source above when you
use these numbers. The IPB98(y,2) scaling used as a yardstick is from the ITER Physics Basis, *Nucl. Fusion* 39 (1999) 2175 (chapter 2).

| file | records | selection |
|---|---|---|
| `records_conventional.csv` | 206 | ELMy / ELM-free H-mode, quasi-steady (\|dW/dt\| ≤ 15 % of P_loss), R/a ≥ 1.5, ≤ 16 random records per device |
| `records_with_central_temperatures.csv` | 94 | the subset of those with a measured T_e0 or T_i0 (eV in the database) |
| `records_spherical.csv` | 41 | same rules, R/a < 1.5 (NSTX, MAST, START) |

Only the rows are a selection; the columns are the database's own (units as in the database: SI, temperatures in eV).
The script that makes the selection is `backend/ml/shrx/db5/select_records.py`.
