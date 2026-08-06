"""Second-order induced-GW kernel I(t,s,eta), in the (t,s) loop-momentum
variables of Kohri & Terada (KT, arXiv:1804.08577).

Loop momenta k1=u k, k2=v k, with t=u+v-1 in [0,inf), s=u-v in [-1,1],
u=(t+s+1)/2, v=(t-s+1)/2. The physical (triangle) region is exactly the
half-strip t>=0, |s|<=1 -- k-independent, unlike a (u,v) grid.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid

import modes

ETA_IN_FACTOR = 0.01   # eta_in(k) = ETA_IN_FACTOR / k
ETA_C_FACTOR = 400.0   # eta_c(k) = ETA_C_FACTOR / k
T_RESONANCE = np.sqrt(3.0) - 1.0  # kernel resonance line, u+v = sqrt(3)


def u_of(t, s): return (t + s + 1.0) / 2.0
def v_of(t, s): return (t - s + 1.0) / 2.0


def weight(t, s):
    """Geometric prefactor squared in the (t,s) form of P_h [KT eq.(17)]."""
    return (t * (2.0 + t) * (s**2 - 1.0) / ((1.0 - s + t) * (1.0 + s + t)))**2


def source_f(Phi1, dPhi1, Phi2, dPhi2, w, H):
    """Source term f(t,s,eta) [KT eq.(16), general w(eta)]."""
    c1 = 6.0 * (w + 1.0) / (3.0 * w + 5.0)
    c2 = 6.0 * (1.0 + 3.0 * w) * (w + 1.0) / (3.0 * w + 5.0)**2
    c3 = 3.0 * (1.0 + 3.0 * w)**2 * (1.0 + w) / (3.0 * w + 5.0)**2
    iH = 1.0 / H
    return c1 * Phi1 * Phi2 + c2 * iH * (dPhi1 * Phi2 + dPhi2 * Phi1) + c3 * iH**2 * dPhi1 * dPhi2


def match_tensor(k, bg, eps=1e-3, safety=2.0):
    """Tensor Green's functions g1 (g=0,g'=1), g2 (g=1,g'=0) at external k."""
    eta_in, eta_c = ETA_IN_FACTOR / k, ETA_C_FACTOR / k
    g1 = modes.match_mode(k, bg.f1_g, bg.f2_g, bg.f3_g, eta_in, 0.0, 1.0, eta_c, eps=eps, safety=safety)
    g2 = modes.match_mode(k, bg.f1_g, bg.f2_g, bg.f3_g, eta_in, 1.0, 0.0, eta_c, eps=eps, safety=safety)
    return g1, g2


def match_scalar(k1, eta_c, bg, eps=1e-3, safety=2.0):
    """Scalar transfer function Phi at wavenumber k1, evolved to eta_c."""
    return modes.match_mode(k1, bg.f1_Phi, bg.f2_Phi, bg.f3_Phi, ETA_IN_FACTOR / k1, 1.0, 0.0, eta_c,
                            eps=eps, safety=safety)


def source_grid(k, t, s, eta_c, pts_per_period=25.0, n_min=800, n_max=60000):
    """Log-spaced eta grid resolving the fastest phase present at eta_c."""
    q_fast = max(u_of(t, s), v_of(t, s), 1.0)
    eta_lo = ETA_IN_FACTOR / (q_fast * k)
    n = int(np.clip(pts_per_period * q_fast * k * eta_c / (2.0 * np.pi), n_min, n_max))
    return np.geomspace(eta_lo, eta_c, n)


def I2bar_numeric(k, t, s, bg, g1m, g2m, phi_u=None, phi_v=None,
                  n_periods_avg=8, n_per_period=20, pts_per_period=25.0):
    """x_c^2 <I^2>(t,s) at external k, by direct mode-function solve + source
    quadrature + oscillation average over the last n_periods_avg periods."""
    eta_c = ETA_C_FACTOR / k
    u, v = u_of(t, s), v_of(t, s)
    if phi_u is None:
        phi_u = match_scalar(u * k, eta_c, bg)
    if phi_v is None:
        phi_v = match_scalar(v * k, eta_c, bg)

    eta = source_grid(k, t, s, eta_c, pts_per_period=pts_per_period)
    Phi1, dPhi1 = modes.eval_mode(phi_u, eta)
    Phi2, dPhi2 = modes.eval_mode(phi_v, eta)
    g1, _ = modes.eval_mode(g1m, eta)
    g2, _ = modes.eval_mode(g2m, eta)

    w = np.asarray(bg.w_of_eta(eta))
    H = np.asarray(bg.H_of_eta(eta))
    a = np.asarray(bg.a_of_eta(eta))
    f = source_f(Phi1, dPhi1, Phi2, dPhi2, w, H)

    J1 = cumulative_trapezoid(g1 * a * f, eta, initial=0.0)[-1]
    J2 = cumulative_trapezoid(g2 * a * f, eta, initial=0.0)[-1]

    period = 2.0 * np.pi / k
    win = np.linspace(eta_c - n_periods_avg * period, eta_c, n_periods_avg * n_per_period)
    win = win[win > ETA_IN_FACTOR / k]
    g1w, _ = modes.eval_mode(g1m, win)
    g2w, _ = modes.eval_mode(g2m, win)
    aw = np.asarray(bg.a_of_eta(win))
    I = (k**2 / aw) * (g1w * J2 - g2w * J1)
    return float(np.trapezoid((k * win)**2 * I**2, win) / (win[-1] - win[0]))


def I2bar_rd_exact(t, s):
    """x^2 <I^2>(t,s) in exact RD, KT eq.(27) -- closed form, no expansion."""
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    A = -5.0 + s**2 + t * (2.0 + t)
    pref = 288.0 * A**2 / ((1.0 - s + t)**6 * (1.0 + s + t)**6)
    resonant = (np.pi**2 / 4.0) * A**2 * np.heaviside(t - T_RESONANCE, 0.5)
    log = (-(1.0 - s + t) * (1.0 + s + t)
           + 0.5 * A * np.log(np.abs((-2.0 + t * (2.0 + t)) / (3.0 - s**2))))**2
    return pref * (resonant + log)
