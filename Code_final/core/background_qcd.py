"""QCD-era background, self-consistently solved from the Saikawa-Shirai
g*(T) fits, following Almeida & Torrado (2307.01653) Eq.(2):

    drho/deta = -3(1+w) H rho ,   3 Mpl^2 H^2 = a^2 rho ,   rho(T) = (pi^2/30) g*_rho(T) T^4

with w(T), cs2(T) from thermo.w_cs2_of_T. Solved as a coupled ODE for
T(eta), a(eta) directly (not via the entropy-conservation shortcut, which
is not exactly equivalent since g*_rho, g*_s are independently fitted
rational functions, not both derived from one potential p(T): the two
routes differ at the percent level).

Deep-RD initial condition at T_i=1e13 GeV (via entropy conservation,
exact to <0.1% there), then rescaled post-hoc so a=1 today (T=T0=2.725K).
"""
import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx

import thermo as ss

jax.config.update("jax_enable_x64", True)

M_PL_GEV = 2.435e18
HBARC_GEV_CM = 1.973269804e-14
CM_PER_MPC = 3.0856775814913673e24
GEV_TO_INV_MPC = CM_PER_MPC / HBARC_GEV_CM

T0_KELVIN = 2.725
KELVIN_TO_GEV = 8.617333262e-14
T0_GEV = T0_KELVIN * KELVIN_TO_GEV

_GS0 = float(ss.gstar_gstars(jnp.asarray(T0_GEV))[1])


def _rho_of_T(T):
    grho, _ = ss.gstar_gstars(T)
    return (jnp.pi**2 / 30.0) * grho * T**4


def _calH(a, T):
    """Conformal Hubble rate [Mpc^-1]."""
    return a * jnp.sqrt(_rho_of_T(T) / 3.0 / M_PL_GEV**2) * GEV_TO_INV_MPC


# deep-RD anchor: deep enough that eta_in(k) never needs T,H below the
# table's lower boundary for any k of interest here
_TI_GEV = 1e13
_gs_Ti = float(ss.gstar_gstars(jnp.asarray(_TI_GEV))[1])
_AI = (_GS0 / _gs_Ti)**(1.0 / 3.0) * (T0_GEV / _TI_GEV)
_ETAI = float(1.0 / _calH(jnp.asarray(_AI), jnp.asarray(_TI_GEV)))
_ETAF = 1e4  # g* is fully frozen well before this


def _rhs(s, y, args):
    """s = ln(eta); y = [ln T, ln a]."""
    lnT, lnA = y
    T, a = jnp.exp(lnT), jnp.exp(lnA)
    eta = jnp.exp(s)
    grho, grhop = ss.gstar_gstars(T)[0], ss.gstar_derivs(T)[0]
    w, _ = ss.w_cs2_of_T(T)
    H = _calH(a, T)
    dT_deta = -3.0 * (1.0 + w) * H * grho / (grhop + 4.0 * grho / T)
    return jnp.array([eta * dT_deta / T, eta * H])


_N_DENSE = 20000
_S_I, _S_F = np.log(_ETAI), np.log(_ETAF)
_S_DENSE = np.linspace(_S_I, _S_F, _N_DENSE)

print("background_qcd: solving coupled continuity-equation ODE for T(eta), a(eta)...")
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

_w_dense, _cs2_dense = np.asarray(ss.w_cs2_of_T(jnp.asarray(_T_DENSE)))
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
