"""P_h(k) and Omega_GW(k) by direct (t,s) integration.

(t,s) instead of (u,v): the physical region is exactly the half-strip
t>=0, |s|<=1 -- an O(1) range at every external k, unlike (u,v) where the
momentum-conservation window sits at u~v~k*/k and is under-resolved by a
shared grid once k << k*. The t-range is chosen per-k from the primordial
spectrum's own support (`t_segments`), since the loop momenta scale as
u,v ~ (t+1)/2 and a spectrum peaked at k* contributes only near
t+1 = 2k*/k.
"""
import numpy as np
from scipy.integrate import simpson

import kernel as K

OMEGA_R0_H2 = 4.2e-5
MPC_INV_TO_HZ = 1.0 / 6.313e14


def k_to_f_Hz(k): return np.asarray(k) * MPC_INV_TO_HZ
def f_Hz_to_k(f): return np.asarray(f) / MPC_INV_TO_HZ


def flat(k):
    return np.ones_like(np.asarray(k, dtype=float))


def lognormal(k, kstar, sigma):
    k = np.asarray(k, dtype=float)
    return (1.0 / np.sqrt(2.0 * np.pi * sigma**2)) * np.exp(-(np.log10(k / kstar))**2 / (2.0 * sigma**2))


def make_pzeta(spec):
    """spec = ('flat',) or ('lognormal', kstar, sigma)."""
    if spec[0] == "flat":
        return flat
    if spec[0] == "lognormal":
        _, kstar, sigma = spec
        return lambda k: lognormal(k, kstar, sigma)
    raise ValueError(spec)


def _seg_log(lo, hi, n):
    """Uniform in ln t."""
    x = np.linspace(np.log(lo), np.log(hi), n)
    t = np.exp(x)
    return t, x, t


def _seg_toward(t0, width, n, side, eps=1e-6):
    """Uniform in ln|t-t0|: clusters onto the resonance so the integrable
    ln^2|t-t_res| feature converges fast."""
    y = np.linspace(np.log(eps), np.log(width), n)
    d = np.exp(y)
    return (t0 + d if side > 0 else t0 - d), y, d


def t_segments(k, spec, n_seg=200, n_sigma=6.0, t_max_flat=300.0, t_min=1e-4):
    """t-segments covering the integrand's support at external k, uniform
    in their own variable so Simpson stays stable across a resonance."""
    if spec[0] == "lognormal":
        _, kstar, sigma = spec
        centre = 2.0 * kstar / k
        lo = max(centre * 10.0**(-n_sigma * sigma) - 1.0, t_min)
        hi = centre * 10.0**(+n_sigma * sigma) - 1.0
    else:
        lo, hi = t_min, t_max_flat

    tr = K.T_RESONANCE
    if not (lo < tr < hi):
        return [_seg_log(lo, hi, 4 * n_seg)]

    w = min(0.2, 0.5 * (tr - lo), 0.5 * (hi - tr))
    segs = []
    if lo < tr - w:
        segs.append(_seg_log(lo, tr - w, n_seg))
    segs.append(_seg_toward(tr, w, n_seg, side=-1))
    segs.append(_seg_toward(tr, w, n_seg, side=+1))
    if tr + w < hi:
        segs.append(_seg_log(tr + w, hi, n_seg))
    return segs


def s_grid(n_s=101):
    """s in [0,1); the integrand vanishes as (s^2-1)^2 at s=1."""
    return np.linspace(0.0, 1.0 - 1e-9, n_s)


def Ph(k, spec, bg=None, i2bar=None, n_s=101, verbose=False, **seg_kw):
    """x_c^2 <P_h(eta_c,k)> / A_zeta^2 = 4 int dt int ds W(t,s) <I^2>(t,s) Pz(ku) Pz(kv).

    i2bar=None -> full numeric mode-function solve on bg
        =callable(t,s) -> use that instead (e.g. kernel.I2bar_rd_exact)."""
    pz = make_pzeta(spec)
    ss = s_grid(n_s)
    total = 0.0
    if i2bar is None:
        eta_c = K.ETA_C_FACTOR / k
        g1m, g2m = K.match_tensor(k, bg)

    for iseg, (ts, xs, jac) in enumerate(t_segments(k, spec, **seg_kw)):
        T, S = np.meshgrid(ts, ss, indexing="ij")
        W = K.weight(T, S) * pz(k * K.u_of(T, S)) * pz(k * K.v_of(T, S))
        if i2bar is not None:
            I2 = i2bar(T, S)
        else:
            I2 = np.zeros_like(T)
            wmax = W.max()
            for i, t in enumerate(ts):
                for j, s in enumerate(ss):
                    if W[i, j] <= 1e-12 * wmax:
                        continue
                    u, v = K.u_of(t, s), K.v_of(t, s)
                    phi_u = K.match_scalar(u * k, eta_c, bg)
                    phi_v = K.match_scalar(v * k, eta_c, bg)
                    I2[i, j] = K.I2bar_numeric(k, t, s, bg, g1m, g2m, phi_u=phi_u, phi_v=phi_v)
                    del phi_u, phi_v
                if verbose and (i % 25 == 0):
                    print(f"    seg {iseg} t {i+1}/{len(ts)}", flush=True)
        total += simpson(simpson(W * I2, x=ss, axis=1) * jac, x=xs)
    return 4.0 * total


def omega_gw(k, Ph_value, bg, eta_f, eta_c=None):
    """Omega_GW h^2 / (Omega_r0 h^2 A_zeta^2) from Ph_value = x_c^2<P_h>/A^2:

        Omega_GW = c_g (1/24) (k/H_c)^2 <P_h>,   c_g = (a_c H_c / a_f H_f)^2.

    eta_c defaults to 400/k (the time Ph was evaluated at); pass eta_c=1/k
    for an exact-RD kernel, whose meaningful time is horizon crossing."""
    if eta_c is None:
        eta_c = K.ETA_C_FACTOR / k
    Hc = float(bg.H_of_eta(eta_c))
    ac = float(bg.a_of_eta(eta_c))
    af = float(bg.a_of_eta(eta_f))
    Hf = float(bg.H_of_eta(eta_f))
    cg = (ac * Hc / (af * Hf))**2
    return cg * (1.0 / 24.0) * Ph_value / (Hc * eta_c)**2
