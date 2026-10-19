"""The primitives that were rewritten for speed must equal the straightforward versions they replaced.

Each test carries the original (slow, obviously-correct) implementation and compares.  Unlike the golden-discharge
test these hold on any platform: they compare like with like, so round-off does not enter.
"""
import numpy as np
import pytest
from scipy.linalg import solve_banded
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import factorized

from tokamak import equilibrium as eq
from tokamak import solver as slv
from tokamak import sol
from tokamak import transport as tr
from tokamak.geometry import boundary, trapped_fraction


# --------------------------------------------------------------------------- _diffuse
def _diffuse_reference(grid, y, cap, chi, source, sink_rate, dt, y_edge, pinch=None, i_edge=None, y_sep=None):
    """The element-by-element assembly _diffuse used before it was vectorised."""
    n, dr, a2, Vp = grid.n, grid.dr, grid.a ** 2, grid.Vp
    Vf = 0.5 * (Vp[:-1] + Vp[1:])
    cf = 0.5 * (cap[:-1] + cap[1:])
    kf = 0.5 * (chi[:-1] + chi[1:])
    D = Vf * cf * kf / (a2 * dr)
    if pinch is None:
        Pw = np.zeros(len(D))
    else:
        pf = 0.5 * (pinch[:-1] + pinch[1:])
        Pw = 0.5 * Vf * cf * kf * pf / a2
    lower, diag, upper, rhs = (np.zeros(n) for _ in range(4))
    vol = np.maximum(Vp * dr, 1e-12)
    vol[0] = grid.V_total * (0.5 * dr) ** 2
    for i in range(1, n - 1):
        aw, ae = D[i - 1] / vol[i], D[i] / vol[i]
        pw, pe = Pw[i - 1] / vol[i], Pw[i] / vol[i]
        lower[i] = -dt * (aw - pw)
        upper[i] = -dt * (ae + pe)
        diag[i] = cap[i] + dt * (aw + ae + pw - pe) + dt * sink_rate[i]
        rhs[i] = cap[i] * y[i] + dt * source[i]
    ae0, pe0 = D[0] / vol[0], Pw[0] / vol[0]
    diag[0] = cap[0] + dt * (ae0 - pe0) + dt * sink_rate[0]
    upper[0] = -dt * (ae0 + pe0)
    rhs[0] = cap[0] * y[0] + dt * source[0]
    ie = n - 1 if i_edge is None else int(np.clip(i_edge, 2, n - 1))
    for k in range(ie, n):
        lower[k] = 0.0
        upper[k] = 0.0
        diag[k] = 1.0
        rhs[k] = y_edge
    if ie < n - 1:
        sep = y_edge if y_sep is None else y_sep
        for k in range(ie, n):
            f = (grid.rho[k] - grid.rho[ie]) / max(1.0 - grid.rho[ie], 1e-9)
            rhs[k] = y_edge + (sep - y_edge) * f
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


@pytest.mark.parametrize("n", [33, 65])
@pytest.mark.parametrize("kw", [{}, {"pinch": True}, {"i_edge": 30, "y_sep": 0.1}, {"pinch": True, "i_edge": 28}])
def test_diffuse_is_bit_identical(n, kw):
    rng = np.random.default_rng(n)
    g = tr.Grid(n=n, a=2.0, V_total=840.0)
    y, cap = rng.uniform(0.1, 10, n), rng.uniform(0.5, 3, n)
    chi, src, sink = rng.uniform(0.05, 20, n), rng.uniform(0, 5, n), rng.uniform(0, 0.2, n)
    kw = dict(kw)
    if kw.pop("pinch", False):
        kw["pinch"] = tr.pinch_profile(g.rho)
    new = tr._diffuse(g, y, cap, chi, src, sink, 0.02, 0.3, **kw)
    ref = _diffuse_reference(g, y, cap, chi, src, sink, 0.02, 0.3, **kw)
    assert np.array_equal(new, ref)


# --------------------------------------------------------------------------- grid integrals
def test_grid_integrals_equal_numpy_trapezoid_bit_for_bit():
    rng = np.random.default_rng(3)
    g = tr.Grid(n=65, a=2.0, V_total=840.0)
    for _ in range(20):
        f = rng.uniform(-3, 3, g.n)
        assert g.volume_average(f) == float(np.trapezoid(f * 2.0 * g.rho, g.rho))
        assert g.integrate(f) == float(np.trapezoid(f * g.Vp, g.rho))


# --------------------------------------------------------------------------- trapped fraction
def test_trapped_fraction_array_matches_scalar():
    eps = np.concatenate([np.linspace(0.0, 1.2, 50), [-0.1, 0.95, 0.2]])
    arr = trapped_fraction(eps)
    assert arr.shape == eps.shape
    for e, a in zip(eps, arr):
        s = trapped_fraction(float(e))
        assert isinstance(s, float) and s == a
    # and against the closed form
    e = np.clip(eps, 0, 0.95)
    assert np.allclose(arr, 1 - (1 - e) ** 2 / (np.sqrt(1 - e * e) * (1 + 1.46 * np.sqrt(e))), rtol=1e-15, atol=1e-15)


# --------------------------------------------------------------------------- slowing-down integrals
def test_slowing_down_array_versions_match_scalar_versions():
    x = np.concatenate([[-1.0, 0.0, 1e-9], np.linspace(0.01, 9.0, 60), [4.0, 4.0000001]])
    for arr_fn, sc_fn in ((slv._slowing_G_arr, slv._slowing_G), (slv._ion_fraction_arr, slv._ion_fraction),
                          (slv._slowing_integral_arr, slv._slowing_integral)):
        got = arr_fn(x)
        want = np.array([sc_fn(float(v)) for v in x])
        assert np.array_equal(got, want), arr_fn.__name__


# --------------------------------------------------------------------------- divertor root
def _target_temperature_reference(P_sep, *, R0, a, B0, B_pol, q95, n_sep20, A_mass=2.5, f_rad_seed=0.0,
                                  flux_expansion=5.0, n_div_targets=2, s_spread_mm=1.0):
    """The target temperature as the original code found it: 300-point scan, then 70 bisection steps."""
    lam_m = (sol.eich_lambda_q(B_pol) + 1.64 * max(s_spread_mm, 0.0)) * 1e-3
    P_sol = max(P_sep * (1.0 - f_rad_seed), 1e-3)
    q_par = P_sol / (2.0 * np.pi * R0 * lam_m) * np.hypot(B0, B_pol) / max(B_pol, 1e-3) / n_div_targets
    L_par = sol.connection_length(R0, q95)
    n_u = n_sep20 * 1e20

    def residual(T_t):
        T_u = (T_t ** 3.5 + 3.5 * q_par * 1e6 * L_par / sol.KAPPA_0E) ** (2.0 / 7.0)
        f_mom = 1.0 / (1.0 + (5.0 / max(T_t, 0.3)) ** 2)
        f_pow = 1.0 / (1.0 + (3.0 / max(T_t, 0.3)) ** 2)
        c_s = 9.79e3 * np.sqrt(2.0 * T_t / A_mass)
        n_t = max(f_mom * n_u * T_u / (2.0 * T_t), 1e14)
        return sol.GAMMA_SHEATH * n_t * c_s * T_t * sol.E_CHARGE / 1e6 - q_par * f_pow

    grid = np.geomspace(5000.0, 0.2, 300)
    vals = np.array([residual(T) for T in grid])
    idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(idx) == 0:
        return float(grid[int(np.argmin(np.abs(vals)))]), False
    lo, hi = grid[idx[0] + 1], grid[idx[0]]
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if residual(hi) * residual(mid) <= 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), True


def test_two_point_root_matches_bisection_reference():
    rng = np.random.default_rng(11)
    n_root = n_noroot = 0
    for _ in range(120):
        kw = dict(R0=rng.uniform(0.8, 9.0), a=rng.uniform(0.4, 3.0), B0=rng.uniform(0.5, 12.0),
                  B_pol=rng.uniform(0.2, 1.5), q95=rng.uniform(2.5, 6.0), n_sep20=rng.uniform(0.02, 0.6),
                  f_rad_seed=float(rng.choice([0.0, 0.3, 0.6, 0.85])))
        P = rng.uniform(1.0, 300.0)
        st = sol.two_point(P, **kw)
        T_ref, has_root = _target_temperature_reference(P, **kw)
        assert (not st.no_attached_root) == has_root
        assert st.T_target_eV == pytest.approx(T_ref, rel=1e-9)
        n_root += has_root
        n_noroot += not has_root
    assert n_root > 20 and n_noroot >= 1          # the sample exercised both branches


# --------------------------------------------------------------------------- Grad-Shafranov operator
def _build_operator_reference(R, Z, mask):
    nZ, nR = mask.shape
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    A = lil_matrix((nR * nZ, nR * nZ))
    idx = lambda i, j: i * nR + j
    for i in range(nZ):
        for j in range(nR):
            k = idx(i, j)
            if not mask[i, j] or i == 0 or i == nZ - 1 or j == 0 or j == nR - 1:
                A[k, k] = 1.0
                continue
            r = R[j]
            A[k, idx(i, j - 1)] = 1.0 / dR ** 2 + 1.0 / (2.0 * r * dR)
            A[k, idx(i, j + 1)] = 1.0 / dR ** 2 - 1.0 / (2.0 * r * dR)
            A[k, idx(i - 1, j)] = 1.0 / dZ ** 2
            A[k, idx(i + 1, j)] = 1.0 / dZ ** 2
            A[k, k] = -2.0 / dR ** 2 - 2.0 / dZ ** 2
    return csr_matrix(A)


@pytest.mark.parametrize("shape", [(6.2, 2.0, 1.85, 0.45, 41, 61), (0.85, 0.65, 2.5, 0.5, 33, 65)])
def test_gs_operator_equals_reference_and_lu_cache_is_transparent(shape):
    R0, a, kappa, delta, nR, nZ = shape
    R, Z, Rg, Zg, mask = eq._grid_and_mask(R0, a, kappa, delta, nR, nZ)
    A_new, A_ref = eq._build_operator(R, Z, mask), _build_operator_reference(R, Z, mask)
    assert A_new.shape == A_ref.shape and (A_new != A_ref).nnz == 0
    rhs = np.random.default_rng(0).normal(size=A_ref.shape[0])
    cached = eq._factorised_operator(R, Z, mask)(rhs)
    assert np.array_equal(cached, factorized(A_ref.tocsc())(rhs))
    assert eq._factorised_operator(R, Z, mask) is eq._factorised_operator(R, Z, mask)      # second call: cache hit
    assert eq._grid_and_mask(R0, a, kappa, delta, nR, nZ)[4] is mask                        # and the mask is shared
