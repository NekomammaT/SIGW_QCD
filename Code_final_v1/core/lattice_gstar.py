"""g*_rho(T), g*_s(T) from the lattice-QCD equation-of-state table
(lattice_gstar_table.txt, T in [1 MeV, 282 GeV]), cubic-spline
interpolated in log10(T) and smoothly cross-faded into the Saikawa-Shirai
fit (thermo.py) outside that range.

w(T), cs2(T) follow from the single-bath identity w = 4 g_s/(3 g_rho) - 1,
valid only while every species shares one temperature; outside the table
this hands off to thermo.w_cs2_of_T directly (the two-sector-aware
treatment) rather than re-deriving w from a single-bath blend of g_rho,
g_s, which would not stay bounded by 1/3 as T->0.
"""
import os
import numpy as np
from scipy.interpolate import CubicSpline

import thermo as ss

_TABLE = os.path.join(os.path.dirname(__file__), "lattice_gstar_table.txt")
_T, _GRHO, _GS = np.loadtxt(_TABLE, unpack=True)
_T_MIN, _T_MAX = _T[0], _T[-1]

_LOG10T = np.log10(_T)
_grho_spline = CubicSpline(_LOG10T, _GRHO, bc_type="not-a-knot")
_gs_spline = CubicSpline(_LOG10T, _GS, bc_type="not-a-knot")
_dgrho_spline = _grho_spline.derivative()
_dgs_spline = _gs_spline.derivative()

_BLEND_WIDTH = 0.5  # in ln(T)
_T_BLEND_LO = _T_MIN
_T_BLEND_HI = _T_MAX


def _weight_and_deriv(T):
    """weight(T): ~1 deep inside the table, ~0 far outside, 0.5 at the
    two edges; smooth tanh cross-fade. Returns (weight, dweight/dT)."""
    lnT = np.log(T)
    x_lo = (lnT - np.log(_T_BLEND_LO)) / _BLEND_WIDTH
    x_hi = (lnT - np.log(_T_BLEND_HI)) / _BLEND_WIDTH
    w_lo = 0.5 * (1.0 + np.tanh(x_lo))
    w_hi = 0.5 * (1.0 - np.tanh(x_hi))
    weight = w_lo * w_hi
    dw_lo_dT = 0.5 * (1.0 - np.tanh(x_lo)**2) / (_BLEND_WIDTH * T)
    dw_hi_dT = -0.5 * (1.0 - np.tanh(x_hi)**2) / (_BLEND_WIDTH * T)
    dweight_dT = dw_lo_dT * w_hi + w_lo * dw_hi_dT
    return weight, dweight_dT


def gstar_gstars_table(T):
    """(g_rho, g_s) from the lattice-table spline, frozen at the edge
    value outside [T_min, T_max] (never extrapolated)."""
    Tc = np.clip(T, _T_MIN, _T_MAX)
    l10 = np.log10(Tc)
    return _grho_spline(l10), _gs_spline(l10)


def _dtable_dT(T):
    inside = (T > _T_MIN) & (T < _T_MAX)
    Tc = np.clip(T, _T_MIN, _T_MAX)
    l10 = np.log10(Tc)
    dgrho = _dgrho_spline(l10) / (Tc * np.log(10.0))
    dgs = _dgs_spline(l10) / (Tc * np.log(10.0))
    return np.where(inside, dgrho, 0.0), np.where(inside, dgs, 0.0)


def _w_cs2_table(T):
    """Single-bath w, cs2 from the table's own g_rho, g_s."""
    grho, gs = gstar_gstars_table(T)
    dgrho, dgs = _dtable_dT(T)
    w = 4.0 * gs / (3.0 * grho) - 1.0
    cs2 = (4.0 * (dgs * T + 4.0 * gs)) / (3.0 * (dgrho * T + 4.0 * grho)) - 1.0
    return np.minimum(w, 1.0 / 3.0), np.minimum(cs2, 1.0 / 3.0)


def _w_cs2_ss(T):
    """Saikawa-Shirai w, cs2, via thermo.w_cs2_of_T."""
    import jax.numpy as jnp
    w, cs2 = ss.w_cs2_of_T(jnp.asarray(T))
    return np.asarray(w), np.asarray(cs2)


def w_cs2_of_T(T_GeV):
    """w(T), cs2(T): table-based single-bath values inside [T_min, T_max],
    smoothly cross-faded into Saikawa-Shirai (thermo.w_cs2_of_T) outside."""
    T = np.atleast_1d(np.asarray(T_GeV, dtype=float))
    weight, _ = _weight_and_deriv(T)
    w_tab, cs2_tab = _w_cs2_table(T)
    w_ss, cs2_ss = _w_cs2_ss(T)
    w = weight * w_tab + (1.0 - weight) * w_ss
    cs2 = weight * cs2_tab + (1.0 - weight) * cs2_ss
    shape = np.asarray(T_GeV).shape
    if shape:
        return w.reshape(shape), cs2.reshape(shape)
    return float(w[0]), float(cs2[0])


def w_of_T(T_GeV):
    return w_cs2_of_T(T_GeV)[0]


def cs2_of_T(T_GeV):
    return w_cs2_of_T(T_GeV)[1]


def gstar_gstars(T_GeV):
    """(g_rho, g_s): lattice table inside [T_min, T_max], smoothly
    cross-faded into Saikawa-Shirai outside."""
    import jax.numpy as jnp
    T = np.atleast_1d(np.asarray(T_GeV, dtype=float))
    weight, _ = _weight_and_deriv(T)
    grho_tab, gs_tab = gstar_gstars_table(T)
    grho_ss, gs_ss = ss.gstar_gstars(jnp.asarray(T))
    grho_ss, gs_ss = np.asarray(grho_ss), np.asarray(gs_ss)
    grho = weight * grho_tab + (1.0 - weight) * grho_ss
    gs = weight * gs_tab + (1.0 - weight) * gs_ss
    shape = np.asarray(T_GeV).shape
    if shape:
        return grho.reshape(shape), gs.reshape(shape)
    return float(grho[0]), float(gs[0])


def gstar_derivs(T_GeV):
    """d(g_rho)/dT, d(g_s)/dT of the blended definition above."""
    import jax.numpy as jnp
    T = np.atleast_1d(np.asarray(T_GeV, dtype=float))
    weight, dweight = _weight_and_deriv(T)
    grho_tab, gs_tab = gstar_gstars_table(T)
    dgrho_tab, dgs_tab = _dtable_dT(T)
    grho_ss, gs_ss = ss.gstar_gstars(jnp.asarray(T))
    grho_ss, gs_ss = np.asarray(grho_ss), np.asarray(gs_ss)
    dgrho_ss, dgs_ss = ss.gstar_derivs(jnp.asarray(T))
    dgrho_ss, dgs_ss = np.asarray(dgrho_ss), np.asarray(dgs_ss)
    dgrho = dweight * (grho_tab - grho_ss) + weight * dgrho_tab + (1.0 - weight) * dgrho_ss
    dgs = dweight * (gs_tab - gs_ss) + weight * dgs_tab + (1.0 - weight) * dgs_ss
    shape = np.asarray(T_GeV).shape
    if shape:
        return dgrho.reshape(shape), dgs.reshape(shape)
    return float(dgrho[0]), float(dgs[0])
