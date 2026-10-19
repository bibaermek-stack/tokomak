"""Where QLKNN switches the turbulence on, and how well SHRX knows it.

QLKNN's heat flux is zero below a critical gradient and rises steeply above it: scaling both R/L_Ti and R/L_Te
by a factor f, the flux of a typical state goes from 0 to O(10) GB within f = 1.0 ... 1.25.  In a stiff plasma the
gradient sits just above that threshold, so an error delta in the threshold position (as a fraction of f) becomes
an error of about delta in the temperature gradient, i.e. in the stored energy.  That makes the *threshold error*,
not the flux error far above it, the number that predicts how SHRX behaves in the integrated simulation.

    critical_factor(predict, X)   ->  f_c per row  (NaN where the onset is not bracketed by f in [0.3, 3])

``predict`` maps (N, 10) inputs to (N, 3) gyro-Bohm fluxes (q_i, q_e, Gamma_e): ``qlknn_common.label`` for the teacher,
``Shrx.predict`` for the student.
"""
import numpy as np

Q_ON = 1.0            # GB units: "turbulence is on" when q_i + q_e exceeds this
F_LO, F_HI = 0.3, 3.0
# columns of the input matrix that carry the temperature gradients (bridge.FEATURES order)
ATI, ATE = 0, 1


def _scaled(X, f):
    Y = X.copy()
    Y[:, ATI] *= f
    Y[:, ATE] *= f
    return Y


def _flux(predict, X, f):
    y = predict(_scaled(X, f))
    return y[:, 0] + y[:, 1]


def critical_factor(predict, X, q_on=Q_ON, iters=22):
    n = len(X)
    lo = np.full(n, F_LO)
    hi = np.full(n, F_HI)
    ok = (_flux(predict, X, lo) < q_on) & (_flux(predict, X, hi) > q_on)       # onset bracketed
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        on = _flux(predict, X, mid) > q_on
        hi = np.where(on, mid, hi)
        lo = np.where(on, lo, mid)
    fc = np.sqrt(lo * hi)
    return np.where(ok, fc, np.nan)
