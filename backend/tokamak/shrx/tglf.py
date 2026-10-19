"""TGLFNN-UKAEA (STEP version) as a second teacher for spherical tokamaks.   NumPy only.

QLKNN, SHRX's first teacher, was trained at one aspect ratio (R/a = 3) and says nothing reliable about spherical tokamaks.
UKAEA publishes (github.com/ukaea/tglfnn-ukaea, LGPL) a neural surrogate of TGLF-EM trained on the input space of JETTO
simulations of the STEP ramp-up (SPR45: R/a ~ 1.8, kappa ~ 2.3, finite beta).  Its weights come as a pickle of NumPy arrays:
15 local inputs, a 5-member deep ensemble per flux, six linear layers of width 512 with ReLU, and a (mean, softplus-variance)
output pair in normalised units.  The forward pass below was checked against the onnxruntime export of the multimachine model
of the same family (identical weights; mean agrees to 1e-5 relative).

The weights are not part of this repository.  Point ``TGLFNN_STEP_PKL`` at ``tglfnn_ukaea/weights/step.pkl`` of a clone.
The pickle is read with a restricted unpickler that only accepts NumPy array reconstruction: loading a pickle with the stock
``pickle.load`` would execute whatever code the file carries.

    from tokamak.shrx import tglf
    out = tglf.evaluate(rho, Te, Ti, ne20, q, machine, zeff, li=0.9, beta_p=1.0)    # chi_e, chi_i [m^2/s], spreads

Conventions (TGLF "GYRO units"): lengths in the minor radius a, fluxes in n_e T_e c_s (rho_s/a)^2 with deuterium c_s and
rho_s built from B_unit.  See :func:`features` for how SINTEZ's state is mapped; the mapping of the geometry inputs
(Shafranov shift, shape profiles) is a model assumption, not a validated bridge.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from . import bridge

DEFAULT_PATH = Path.home() / "ukaea" / "tglfnn-ukaea" / "tglfnn_ukaea" / "weights" / "step.pkl"
FEATURES = ("RLNS_1", "RLTS_1", "RLTS_2", "TAUS_2", "RMIN_LOC", "DRMAJDX_LOC", "Q_LOC", "SHAT", "XNUE", "KAPPA_LOC",
            "S_KAPPA_LOC", "DELTA_LOC", "S_DELTA_LOC", "BETAE", "ZEFF")
FLUXES = ("efe_gb", "efi_gb", "pfi_gb")
M_D = 2.01410178 * bridge.M_AMU           # TGLF's reference mass: deuterium
#: the training box is not published: the inputs are limited to the training mean +- this many standard deviations
SIGMA_BOX = 3.5


class _SafeUnpickler(pickle.Unpickler):
    _OK = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
           ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self._OK:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name} in a weights file")


def path() -> Path:
    return Path(os.environ.get("TGLFNN_STEP_PKL") or DEFAULT_PATH)


def available() -> bool:
    return path().is_file()


#: the distilled student (SHRX-ST): shipped, NumPy only, needs no teacher weights
ST_PATH = Path(__file__).with_name("weights_st.npz")


def st_available() -> bool:
    return ST_PATH.is_file()


class TglfNN:
    def __init__(self, file: Optional[Path] = None):
        with open(file or path(), "rb") as f:
            d = _SafeUnpickler(f).load()
        assert tuple(d["input_labels"]) == FEATURES, d["input_labels"]
        st = d["stats"]
        self.mu = np.array([st[k]["mean"] for k in FEATURES], np.float32)
        self.sd = np.array([st[k]["std"] for k in FEATURES], np.float32)
        self.out_mu = {k: np.float32(st[k]["mean"]) for k in FLUXES}
        self.out_sd = {k: np.float32(st[k]["std"]) for k in FLUXES}
        self.W, self.b = {}, {}
        for flux in FLUXES:
            members = [d["params"][flux][f"MLP_{m}"] for m in range(5)]
            n = len(members[0])
            self.W[flux] = [np.stack([m[f"FullyConnectedLayer_{i}"]["weight"].T for m in members]).astype(np.float32)
                            for i in range(n)]
            self.b[flux] = [np.stack([m[f"FullyConnectedLayer_{i}"]["bias"] for m in members])[:, None, :].astype(np.float32)
                            for i in range(n)]
        self.lo, self.hi = self.mu - SIGMA_BOX * self.sd, self.mu + SIGMA_BOX * self.sd
        self.meta = {"gkmodel": d["config"].get("gkmodel"), "loss": d["config"].get("loss_function"),
                     "members": 5, "width": d["config"].get("hidden_size"), "layers": d["config"].get("model_size")}

    def out_of_range(self, X: np.ndarray) -> np.ndarray:
        """True where an input lies outside mean +- SIGMA_BOX std of the training inputs (clipped before the network)."""
        return np.any((X < self.lo) | (X > self.hi), axis=1)

    def predict_members(self, X: np.ndarray, flux: str) -> np.ndarray:
        """(N, 15) inputs -> (K, N) flux of every ensemble member, in gyro-Bohm units."""
        x = (np.clip(X.astype(np.float32, copy=False), self.lo, self.hi) - self.mu) / self.sd
        h = np.matmul(x[None], self.W[flux][0]) + self.b[flux][0]
        np.maximum(h, 0.0, out=h)
        for W, b in zip(self.W[flux][1:-1], self.b[flux][1:-1]):
            h = np.matmul(h, W)
            h += b
            np.maximum(h, 0.0, out=h)
        o = np.matmul(h, self.W[flux][-1]) + self.b[flux][-1]          # (K, N, 2): normalised mean, raw variance
        return o[..., 0] * self.out_sd[flux] + self.out_mu[flux]

    def predict(self, X: np.ndarray) -> dict:
        """Ensemble-mean fluxes and the relative spread of the members, per flux."""
        out = {}
        for flux in FLUXES:
            m = self.predict_members(X, flux)
            out[flux] = m.mean(axis=0)
            out[flux + "_std"] = m.std(axis=0)
        return out


_MODEL: Optional[TglfNN] = None
_ST = None


def load_st():
    """SHRX-ST: the small network distilled from the teacher (same file format and class as SHRX)."""
    global _ST
    if _ST is None:
        if not st_available():
            raise FileNotFoundError(f"SHRX-ST weights not found: {ST_PATH}")
        from . import Shrx
        _ST = Shrx(ST_PATH)
    return _ST


def load() -> TglfNN:
    global _MODEL
    if _MODEL is None:
        if not available():
            raise FileNotFoundError(f"TGLFNN STEP weights not found: {path()} (set TGLFNN_STEP_PKL)")
        _MODEL = TglfNN()
    return _MODEL


# --------------------------------------------------------------------------------------------------- bridge
def _kappa_profile(x: np.ndarray, kappa_edge: float):
    """kappa(x) and s_kappa = x kappa'/kappa for the elongation profile used by the QLKNN bridge."""
    k0, p = bridge.KAPPA_AXIS_FRACTION, bridge.KAPPA_POWER
    xp = np.power(np.maximum(x, 1e-9), p)
    kappa = kappa_edge * (k0 + (1.0 - k0) * xp)
    s = p * (1.0 - k0) * xp / (k0 + (1.0 - k0) * xp)
    return kappa, s


def features(rho, Te, Ti, ne20, q, machine, zeff: float, normni: float = 0.9, li: float = 0.9, beta_p: float = 0.5):
    """SINTEZ state -> the 15 TGLFNN inputs on the radial grid, plus what :func:`to_si` needs.

    Gradients are taken with respect to the mid-plane half-width r and normalised to a (TGLF's GYRO units).  The shape inputs
    follow the elongation profile of the QLKNN bridge and a triangularity delta_edge x^2; the Shafranov-shift gradient is
    dR0/dr = -(r/R0)(beta_p,local + li/2) with beta_p,local from the local pressure and the circular-limit poloidal field.
    """
    rho = np.ascontiguousarray(rho, dtype=float)
    R0, a, B0 = float(machine.R0), float(machine.a), float(machine.B0)
    kappa_e = float(machine.kappa_x)
    delta_e = float(getattr(machine, "delta_x", 0.0))
    Te = np.maximum(np.asarray(Te, float), 1e-3)
    Ti = np.maximum(np.asarray(Ti, float), 1e-3)
    ne = np.maximum(np.asarray(ne20, float), 1e-4) * 1e20
    q_raw = np.maximum(np.asarray(q, float), 0.05)
    x_mid, dxdrho, uniform = bridge._geometry(rho, kappa_e)
    r = x_mid * a
    dr_drho = np.maximum(a * dxdrho, 1e-12)

    iota = 1.0 / q_raw
    Y = np.stack([Te, Ti, ne, iota])
    G = bridge._ddr(Y, rho, dr_drho, uniform)
    safe = np.where(np.abs(Y[:3]) < bridge.EPS, bridge.EPS, Y[:3])
    A = bridge._guard(-a * G[:3] / safe)                    # a / L_y
    shat = -r * G[3] / iota
    shat[0] = 0.0

    kappa, s_kappa = _kappa_profile(x_mid, kappa_e)
    delta = delta_e * x_mid ** 2
    s_delta = 2.0 * delta

    # B_unit = B0 (rho_eff / r)(d rho_eff / d r), rho_eff = rho a sqrt(kappa_a)  (toroidal-flux radius)
    kappa_a = float(getattr(machine, "kappa_a", kappa_e))
    with np.errstate(divide="ignore", invalid="ignore"):
        b_unit = B0 * kappa_a * np.where(x_mid > 1e-6, rho / x_mid, 1.0 / np.maximum(dxdrho[0], 1e-6)) / np.maximum(dxdrho, 1e-6)
    b_unit = np.clip(b_unit, 0.5 * B0, 3.0 * kappa_e * B0)

    # Shafranov shift gradient
    p = ne * Te * bridge.KEV_J + 0.9 * ne * Ti * bridge.KEV_J
    rr = np.maximum(r, 1e-6)
    p_avg = np.concatenate([[p[0]], 2.0 * np.cumsum(0.5 * (p[1:] * rr[1:] + p[:-1] * rr[:-1]) * np.diff(rr)) / rr[1:] ** 2])
    b_theta = np.maximum(rr * B0 / (q_raw * R0), 1e-4)
    beta_p_loc = np.clip(2.0 * bridge.MU0 * (p_avg - p) / b_theta ** 2, 0.0, 3.0)
    drmajdx = -(r / R0) * (beta_p_loc + 0.5 * li)

    # electron-electron collision frequency in units of c_s / a
    te_ev = Te * 1e3
    ln_lambda = 24.0 - np.log(np.sqrt(ne * 1e-6) / te_ev)
    nu_e = 2.91e-6 * ne * 1e-6 * ln_lambda * te_ev ** -1.5
    cs = np.sqrt(Te * bridge.KEV_J / M_D)
    xnue = nu_e / (cs / a)

    betae = 2.0 * bridge.MU0 * ne * Te * bridge.KEV_J / b_unit ** 2

    X = np.empty((len(rho), len(FEATURES)))
    X[:, 0], X[:, 1], X[:, 2], X[:, 3] = A[2], A[0], A[1], Ti / Te
    X[:, 4], X[:, 5], X[:, 6], X[:, 7], X[:, 8] = x_mid, drmajdx, q_raw, shat, xnue
    X[:, 9], X[:, 10], X[:, 11], X[:, 12], X[:, 13], X[:, 14] = kappa, s_kappa, delta, s_delta, betae, zeff
    return X, {"Ate": A[0], "Ati": A[1], "Te": Te, "Ti": Ti, "b_unit": b_unit, "a": a, "normni": normni}


def chi_gb(Te, b_unit, a):
    """c_s rho_s^2 / a [m^2/s] with deuterium c_s, rho_s from B_unit (TGLF's GYRO units)."""
    T = Te * bridge.KEV_J
    return np.sqrt(M_D) * T ** 1.5 / (bridge.Q_E ** 2 * b_unit ** 2 * a)


def to_si(efe, efi, X, aux, chi_min=0.05, chi_max=100.0):
    """Gyro-Bohm heat fluxes -> chi_e, chi_i [m^2/s]:  chi = Q_GB-flux * c_s rho_s^2 / (a * a/L_T) (ions: / (n_i T_i / n_e T_e))."""
    gb = chi_gb(aux["Te"], aux["b_unit"], aux["a"])
    chi_e = efe * gb / np.maximum(aux["Ate"], bridge.EPS)
    chi_i = efi * gb / (aux["normni"] * np.maximum(X[:, 3], 1e-3) * np.maximum(aux["Ati"], bridge.EPS))
    return np.clip(chi_e, chi_min, chi_max), np.clip(chi_i, chi_min, chi_max)


def evaluate(rho, Te, Ti, ne20, q, machine, zeff: float, normni: float = 0.9, li: float = 0.9, beta_p: float = 0.5,
             student: bool = False) -> dict:
    """chi_e, chi_i [m^2/s] from the TGLFNN STEP surrogate (or, with ``student=True``, from the distilled SHRX-ST network).

    Same keys as :func:`tokamak.shrx.evaluate`; the teacher also returns its member spreads."""
    X, aux = features(rho, Te, Ti, ne20, q, machine, zeff, normni, li, beta_p)
    if student:
        net = load_st()
        flux = net.predict(X.astype(np.float32))
        chi_e, chi_i = to_si(flux[:, 0], flux[:, 1], X, aux)
        ood = net.out_of_range(X)
        j = FEATURES.index("RMIN_LOC")
        inside = (X[:, j] > 0.1) & (X[:, j] < 0.95)
        return {"chi_e": chi_e, "chi_i": chi_i, "gamma_e_gb": flux[:, 2], "ood_fraction": float(ood.mean()),
                "ood_core": float(ood[inside].mean()) if inside.any() else 1.0, "ood": ood, "features": X}
    net = load()
    f = net.predict(X)
    chi_e, chi_i = to_si(f["efe_gb"], f["efi_gb"], X, aux)
    ood = net.out_of_range(X)
    j = FEATURES.index("RMIN_LOC")
    inside = (X[:, j] > 0.1) & (X[:, j] < 0.95)
    return {"chi_e": chi_e, "chi_i": chi_i, "gamma_e_gb": f["pfi_gb"], "ood_fraction": float(ood.mean()),
            "ood_core": float(ood[inside].mean()) if inside.any() else 1.0, "ood": ood, "features": X,
            "efe_gb": f["efe_gb"], "efi_gb": f["efi_gb"], "efe_gb_std": f["efe_gb_std"], "efi_gb_std": f["efi_gb_std"]}
