"""g*_rho(T), g*_s(T) and the resulting w(T), cs^2(T), from the Saikawa &
Shirai fitting functions (arXiv:1803.01038, App. C). Two branches matched
at T=120 MeV: a rational-polynomial fit above, an explicit sum over
photon/lepton/hadron-resonance contributions below.

w(T), cs2(T): above 120 MeV all species share one temperature, so
w = 4 g*_s/(3 g*_rho) - 1 [Almeida & Torrado 2307.01653 Eq.(1)]. Below
120 MeV neutrinos have decoupled (own T_nu(T) != T), so we split into a
neutrino sector and a single-temperature "everything else" sector and
combine by energy density -- this keeps w,cs2 <= 1/3 everywhere, unlike
the naive single-temperature formula applied below its regime of
validity. Derivatives via JAX autodiff.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# --- Branch 1: 120 MeV <= T <= 1e16 GeV [Table 1] ---
_A = jnp.array([1.0, 1.11724e+00, 3.12672e-01, -4.68049e-02, -2.65004e-02, -1.19760e-03,
                1.82812e-04, 1.36436e-04, 8.55051e-05, 1.22840e-05, 3.82259e-07, -6.87035e-09])
_B = jnp.array([1.43382e-02, 1.37559e-02, 2.92108e-03, -5.38533e-04, -1.62496e-04, -2.87906e-05,
                -3.84278e-06, 2.78776e-06, 7.40342e-07, 1.17210e-07, 3.72499e-09, -6.74107e-11])
_C = jnp.array([1.0, 6.07869e-01, -1.54485e-01, -2.24034e-01, -2.82147e-02, 2.90620e-02,
                6.86778e-03, -1.00005e-03, -1.69104e-04, 1.06301e-05, 1.69528e-06, -9.33311e-08])
_D = jnp.array([7.07388e+01, 9.18011e+01, 3.31892e+01, -1.39779e+00, -1.52558e+00, -1.97857e-02,
                -1.60146e-01, 8.22615e-05, 2.02651e-02, -1.82134e-05, 7.83943e-05, 7.13518e-05])
_POWERS = jnp.arange(12)


def _poly(t, coeffs):
    return jnp.sum(t[..., None] ** _POWERS * coeffs, axis=-1)


def _gstar_high(T_GeV):
    t = jnp.log(T_GeV)
    grho = _poly(t, _A) / _poly(t, _B)
    ratio = 1.0 + _poly(t, _C) / _poly(t, _D)  # g*_rho/g*_s
    return grho, grho / ratio


# --- Branch 2: T < 120 MeV ---
_M_E, _M_MU = 511e-6, 0.1056
_M_PI0, _M_PIC = 0.135, 0.140
_M_1, _M_2, _M_3, _M_4 = 0.5, 0.77, 1.2, 2.0


def _f_rho(x): return jnp.exp(-1.04855 * x) * (1 + 1.03757 * x + 0.508630 * x**2 + 0.0893988 * x**3)
def _b_rho(x): return jnp.exp(-1.03149 * x) * (1 + 1.03317 * x + 0.398264 * x**2 + 0.0648056 * x**3)
def _f_s(x): return jnp.exp(-1.04190 * x) * (1 + 1.03400 * x + 0.456426 * x**2 + 0.0595248 * x**3)
def _b_s(x): return jnp.exp(-1.03365 * x) * (1 + 1.03397 * x + 0.342548 * x**2 + 0.0506182 * x**3)
def _S_fit(x): return 1 + 1.75 * jnp.exp(-1.0419 * x) * (1 + 1.034 * x + 0.456426 * x**2 + 0.0595249 * x**3)


def _gstar_low_nu(T_GeV):
    """Neutrinos: own T_nu/T_gamma ratio, interpolated via S_fit(m_e/T)."""
    xe = _M_E / T_GeV
    return 1.353 * _S_fit(xe)**(4.0 / 3.0), 1.923 * _S_fit(xe)


def _gstar_low_nonnu(T_GeV):
    """Photons/e+-/muons/hadron resonances: one common temperature T."""
    xe, xmu = _M_E / T_GeV, _M_MU / T_GeV
    xpi0, xpic = _M_PI0 / T_GeV, _M_PIC / T_GeV
    x1, x2, x3, x4 = _M_1 / T_GeV, _M_2 / T_GeV, _M_3 / T_GeV, _M_4 / T_GeV
    grho = (2.030 + 3.495 * _f_rho(xe) + 3.446 * _f_rho(xmu)
            + 1.05 * _b_rho(xpi0) + 2.08 * _b_rho(xpic) + 4.165 * _b_rho(x1)
            + 30.55 * _b_rho(x2) + 89.4 * _b_rho(x3) + 8209 * _b_rho(x4))
    gs = (2.008 + 3.442 * _f_s(xe) + 3.468 * _f_s(xmu)
          + 1.034 * _b_s(xpi0) + 2.068 * _b_s(xpic) + 4.16 * _b_s(x1)
          + 30.55 * _b_s(x2) + 90 * _b_s(x3) + 6209 * _b_s(x4))
    return grho, gs


def _gstar_low(T_GeV):
    grho_nu, gs_nu = _gstar_low_nu(T_GeV)
    grho_x, gs_x = _gstar_low_nonnu(T_GeV)
    return grho_x + grho_nu, gs_x + gs_nu


_T_MATCH = 0.120  # GeV


def gstar_gstars(T_GeV):
    """g*_rho(T), g*_s(T) for scalar or array T_GeV [GeV]."""
    T_GeV = jnp.asarray(T_GeV, dtype=jnp.float64)
    grho_hi, gs_hi = _gstar_high(T_GeV)
    grho_lo, gs_lo = _gstar_low(T_GeV)
    hi = T_GeV >= _T_MATCH
    return jnp.where(hi, grho_hi, grho_lo), jnp.where(hi, gs_hi, gs_lo)


def _gstar_gstars_scalar(T_GeV):
    hi = T_GeV >= _T_MATCH
    grho_hi, gs_hi = _gstar_high(T_GeV)
    grho_lo, gs_lo = _gstar_low(T_GeV)
    return jnp.where(hi, grho_hi, grho_lo), jnp.where(hi, gs_hi, gs_lo)


def _grho_scalar(T_GeV): return _gstar_gstars_scalar(T_GeV)[0]
def _gs_scalar(T_GeV): return _gstar_gstars_scalar(T_GeV)[1]


_dgrho_dT_scalar = jax.grad(_grho_scalar)
_dgs_dT_scalar = jax.grad(_gs_scalar)
_dgrho_dT_vec = jax.vmap(_dgrho_dT_scalar)
_dgs_dT_vec = jax.vmap(_dgs_dT_scalar)


def gstar_derivs(T_GeV):
    """d g*_rho/dT, d g*_s/dT [per GeV] for scalar or 1d-array T_GeV."""
    T_GeV = jnp.asarray(T_GeV, dtype=jnp.float64)
    if T_GeV.ndim == 0:
        return _dgrho_dT_scalar(T_GeV), _dgs_dT_scalar(T_GeV)
    return _dgrho_dT_vec(T_GeV), _dgs_dT_vec(T_GeV)


def _w_scalar_high(T):
    """T >= 120 MeV: single-temperature formula, clipped to <=1/3 (fit noise)."""
    grho, gs = _gstar_high(T)
    return jnp.minimum(4.0 * gs / (3.0 * grho) - 1.0, 1.0 / 3.0)


def _Tnu(T):
    """Neutrino temperature T_nu(T_gamma)."""
    return (4.0 / 11.0)**(1.0 / 3.0) * _S_fit(_M_E / T)**(1.0 / 3.0) * T


def _P_low(T):
    """p_total(T) / [(pi^2/30) T^4], two-sector split below 120 MeV."""
    grho_nu, gs_nu = _gstar_low_nu(T)
    grho_x, gs_x = _gstar_low_nonnu(T)
    p_x = (4.0 / 3.0) * gs_x - grho_x
    p_nu = (4.0 / 3.0) * (_Tnu(T) / T) * gs_nu - grho_nu
    return p_x + p_nu


def _grho_low_total(T):
    grho_nu, _ = _gstar_low_nu(T)
    grho_x, _ = _gstar_low_nonnu(T)
    return grho_x + grho_nu


_dP_low_dT_scalar = jax.grad(_P_low)
_dgrho_low_dT_scalar = jax.grad(_grho_low_total)


def _w_cs2_scalar_low(T):
    """T < 120 MeV: neutrino sector + single-T sector, combined by energy density."""
    grho_nu, gs_nu = _gstar_low_nu(T)
    grho_x, gs_x = _gstar_low_nonnu(T)
    grho_tot = grho_x + grho_nu
    P = _P_low(T)
    w = P / grho_tot
    dP_dT = _dP_low_dT_scalar(T)
    dgrho_dT = _dgrho_low_dT_scalar(T)
    cs2 = (4.0 * P + T * dP_dT) / (4.0 * grho_tot + T * dgrho_dT)
    return w, cs2


_w_cs2_scalar_low_vec = jax.vmap(_w_cs2_scalar_low)

_dgrho_high_dT_scalar = jax.grad(lambda t: _gstar_high(t)[0])
_dgs_high_dT_scalar = jax.grad(lambda t: _gstar_high(t)[1])


def _cs2_scalar_high(T):
    grho, gs = _gstar_high(T)
    dgrho_dT, dgs_dT = _dgrho_high_dT_scalar(T), _dgs_high_dT_scalar(T)
    cs2 = (4.0 * (dgs_dT * T + 4.0 * gs)) / (3.0 * (dgrho_dT * T + 4.0 * grho)) - 1.0
    return jnp.minimum(cs2, 1.0 / 3.0)


# Below this T, g*_x is numerically frozen but at 0.332376 not exactly 1/3
# (fit-normalization residual): blend to exactly 1/3 to avoid a secular
# drift in a(eta) when integrating over many e-folds past this point.
_T_ASYMPTOTIC_RD = 1.5e-5  # GeV
_ASYMPTOTIC_BLEND_WIDTH = 0.3


def w_cs2_of_T(T_GeV):
    """Physical w(T), cs2(T): two-sector below 120 MeV, single-T above."""
    T_GeV = jnp.asarray(T_GeV, dtype=jnp.float64)
    scalar_in = T_GeV.ndim == 0
    T_flat = T_GeV.reshape(-1)
    w_hi = jax.vmap(_w_scalar_high)(T_flat)
    cs2_hi = jax.vmap(_cs2_scalar_high)(T_flat)
    w_lo, cs2_lo = _w_cs2_scalar_low_vec(T_flat)
    hi = T_flat >= _T_MATCH
    w = jnp.where(hi, w_hi, w_lo)
    cs2 = jnp.where(hi, cs2_hi, cs2_lo)
    blend = 0.5 * (1.0 + jnp.tanh((jnp.log(T_flat) - jnp.log(_T_ASYMPTOTIC_RD)) / _ASYMPTOTIC_BLEND_WIDTH))
    w = blend * w + (1.0 - blend) * (1.0 / 3.0)
    cs2 = blend * cs2 + (1.0 - blend) * (1.0 / 3.0)
    if scalar_in:
        return w[0], cs2[0]
    return w.reshape(T_GeV.shape), cs2.reshape(T_GeV.shape)
