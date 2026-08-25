"""QCD-era background: T(eta), a(eta) solved from lattice_gstar.py's
g_rho(T), g_s(T) via

    drho/deta = -3(1+w) H rho ,   3 Mpl^2 H^2 = a^2 rho ,   rho(T) = (pi^2/30) g_rho(T) T^4
    w(T) = 4 g_s(T) / (3 g_rho(T)) - 1

Deep-RD initial condition at T_i=1e13 GeV via entropy conservation,
rescaled post-hoc so a=1 today (T=T0=2.725K).
"""
import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx

import lattice_gstar as lgs

jax.config.update("jax_enable_x64", True)

M_PL_GEV = 2.435e18
HBARC_GEV_CM = 1.973269804e-14
CM_PER_MPC = 3.0856775814913673e24
GEV_TO_INV_MPC = CM_PER_MPC / HBARC_GEV_CM

T0_KELVIN = 2.725
KELVIN_TO_GEV = 8.617333262e-14
T0_GEV = T0_KELVIN * KELVIN_TO_GEV

_TAB_T = np.geomspace(1e-15, 1e15, 80000)
_TAB_GRHO, _TAB_GS = lgs.gstar_gstars(_TAB_T)
_TAB_W = lgs.w_of_T(_TAB_T)

_LOG_TAB_T_J = jnp.asarray(np.log(_TAB_T))
_LOG_TAB_GRHO_J = jnp.asarray(np.log(_TAB_GRHO))
_LOG_TAB_GS_J = jnp.asarray(np.log(_TAB_GS))
_LOG_TAB_W_J = jnp.asarray(np.log(_TAB_W))


def _grho_of_T_jax(T):
    return jnp.exp(jnp.interp(jnp.log(T), _LOG_TAB_T_J, _LOG_TAB_GRHO_J))


def _gs_of_T_jax(T):
    return jnp.exp(jnp.interp(jnp.log(T), _LOG_TAB_T_J, _LOG_TAB_GS_J))


def _w_of_T_jax(T):
    return jnp.exp(jnp.interp(jnp.log(T), _LOG_TAB_T_J, _LOG_TAB_W_J))


def _rho_of_T(T):
    return (jnp.pi**2 / 30.0) * _grho_of_T_jax(T) * T**4


def _calH(a, T):
    """Conformal Hubble rate [Mpc^-1]."""
    return a * jnp.sqrt(_rho_of_T(T) / 3.0 / M_PL_GEV**2) * GEV_TO_INV_MPC


_TI_GEV = 1e13
_GS0 = float(_gs_of_T_jax(jnp.asarray(T0_GEV)))
_gs_Ti = float(_gs_of_T_jax(jnp.asarray(_TI_GEV)))
_AI = (_GS0 / _gs_Ti)**(1.0 / 3.0) * (T0_GEV / _TI_GEV)
_ETAI = float(1.0 / _calH(jnp.asarray(_AI), jnp.asarray(_TI_GEV)))
_ETAF = 1e4  # g* is fully frozen well before this


def _rhs(s, y, args):
    """s = ln(eta); y = [ln T, ln a]."""
    lnT, lnA = y
    T, a = jnp.exp(lnT), jnp.exp(lnA)
    eta = jnp.exp(s)
    grho, grhop = _grho_of_T_jax(T), jax.grad(_grho_of_T_jax)(T)
    w = _w_of_T_jax(T)
    H = _calH(a, T)
    dT_deta = -3.0 * (1.0 + w) * H * grho / (grhop + 4.0 * grho / T)
    return jnp.array([eta * dT_deta / T, eta * H])


_N_DENSE = 20000
_S_I, _S_F = np.log(_ETAI), np.log(_ETAF)
_S_DENSE = np.linspace(_S_I, _S_F, _N_DENSE)

print("background_qcd: solving coupled continuity-equation ODE for T(eta), a(eta) "
      "(g_rho, g_s, w from lattice_gstar.py)...")
_sol = dfx.diffeqsolve(
    dfx.ODETerm(_rhs), dfx.Dopri8(), t0=_S_I, t1=_S_F, dt0=(_S_F - _S_I) / 2000.0,
    y0=jnp.array([jnp.log(_TI_GEV), jnp.log(_AI)]),
    stepsize_controller=dfx.PIDController(rtol=1e-10, atol=1e-12),
    saveat=dfx.SaveAt(ts=jnp.asarray(_S_DENSE)), max_steps=200000,
)
_lnT_raw = np.asarray(_sol.ys[:, 0])
_lnA_raw = np.asarray(_sol.ys[:, 1])

# post-hoc rescaling to a=1 today
_NORM = (np.exp(_lnT_raw[-1]) / T0_GEV) * np.exp(_lnA_raw[-1])
print(f"  norm = {_NORM:.10f}")

_ETA_DENSE = _NORM * np.exp(_S_DENSE)
_T_DENSE = np.exp(_lnT_raw)
_A_DENSE = np.exp(_lnA_raw) / _NORM

_ETA_MIN_TAB = float(_ETA_DENSE[0])
_ETA_MAX_TAB = float(_ETA_DENSE[-1])

_w_dense = lgs.w_of_T(_T_DENSE)
_cs2_dense = lgs.cs2_of_T(_T_DENSE)
_H_dense = np.asarray(_calH(jnp.asarray(_A_DENSE), jnp.asarray(_T_DENSE)))

_LOG_ETA_DENSE_J = jnp.asarray(np.log(_ETA_DENSE))
_LNT_DENSE_J = jnp.asarray(np.log(_T_DENSE))
_W_DENSE_J = jnp.asarray(_w_dense)
_CS2_DENSE_J = jnp.asarray(_cs2_dense)
_H_DENSE_J = jnp.asarray(_H_dense)
_LNA_DENSE_J = jnp.asarray(np.log(_A_DENSE))


def _lookup(eta, table):
    eta = jnp.asarray(eta, dtype=jnp.float64)
    log_eta = jnp.log(jnp.clip(eta, _ETA_MIN_TAB, _ETA_MAX_TAB))
    return jnp.interp(log_eta, _LOG_ETA_DENSE_J, table)


def T_of_eta(eta):
    """Temperature [GeV] at conformal time eta [Mpc]."""
    return jnp.exp(_lookup(eta, _LNT_DENSE_J))


def w_of_eta(eta): return _lookup(eta, _W_DENSE_J)
def cs2_of_eta(eta): return _lookup(eta, _CS2_DENSE_J)
def H_of_eta(eta): return _lookup(eta, _H_DENSE_J)
def a_of_eta(eta): return jnp.exp(_lookup(eta, _LNA_DENSE_J))


# Generic-form coefficients, X'' + f1 X' + (k^2 f3 + f2) X = 0
def f1_Phi(eta): return 3.0 * H_of_eta(eta) * (1.0 + cs2_of_eta(eta))
def f2_Phi(eta): return 3.0 * H_of_eta(eta)**2 * (cs2_of_eta(eta) - w_of_eta(eta))
def f3_Phi(eta): return cs2_of_eta(eta)


def f1_g(eta):
    eta = jnp.asarray(eta, dtype=jnp.float64)
    return jnp.zeros_like(eta)


def f2_g(eta): return -0.5 * (1.0 - 3.0 * w_of_eta(eta)) * H_of_eta(eta)**2


def f3_g(eta):
    eta = jnp.asarray(eta, dtype=jnp.float64)
    return jnp.ones_like(eta)
