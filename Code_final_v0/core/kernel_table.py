"""Fast tabulation of Ibar2(t,s;k) (and I1,I2 separately), for export to
kernels_data/. Speedup over the naive per-(t,s) approach: the WKB
continuation's non-oscillatory integrals y(eta)=int f1, z(eta)=int
sqrt(f3) are background-only (not k-dependent), so they are solved ONCE,
globally (`BackgroundTables`), and reused for every scalar leg
(`match_scalar_fast`+`eval_fast`) instead of resolved per leg.
"""
import numpy as np
import jax.numpy as jnp
from scipy.integrate import cumulative_trapezoid

import modes
import kernel as K


class BackgroundTables:
    """Global y(eta), z(eta), f1, f3, f3' on one fixed log grid spanning
    every eta that will ever be needed."""

    def __init__(self, f1, f3, eta_lo, eta_hi, n=8000):
        eta = np.geomspace(eta_lo, eta_hi, n)
        y, z = modes._solve_yz_at(f1, f3, eta_lo, eta_hi, jnp.asarray(eta))
        self.eta = eta
        self.Y, self.Z = np.asarray(y), np.asarray(z)
        self.f1 = np.asarray(f1(jnp.asarray(eta)))
        self.f3 = np.asarray(f3(jnp.asarray(eta)))
        self.f3p = np.asarray(modes._deriv(f3)(jnp.asarray(eta)))

    def Yz(self, eta):
        eta = np.asarray(eta, dtype=float)
        return np.interp(eta, self.eta, self.Y), np.interp(eta, self.eta, self.Z)

    def f_of(self, eta):
        eta = np.asarray(eta, dtype=float)
        return (np.interp(eta, self.eta, self.f1),
                np.interp(eta, self.eta, self.f3),
                np.interp(eta, self.eta, self.f3p))


class FastMode:
    __slots__ = ("k", "eta_in", "eta_wkb", "Cplus", "Cminus", "brute_eta", "brute_X", "brute_Xp")


def match_scalar_fast(k1, eta_c, bg, bgtab, X_in=1.0, dX_in=0.0, eps=1e-3, safety=2.0,
                      pts_per_period=25.0, n_brute_min=1000, n_brute_max=20000):
    """Same result as kernel.match_scalar, without re-solving the shared
    (k-independent) y,z integrals; the brute-force sub-horizon segment is
    still solved exactly, per k1."""
    eta_in = K.ETA_IN_FACTOR / k1
    eta_wkb_j, eta_star_j = modes.find_eta_wkb(k1, bg.f1_Phi, bg.f2_Phi, bg.f3_Phi, eta_in, eta_c, eps=eps, safety=safety)
    eta_wkb = min(float(eta_wkb_j), eta_c)

    m = FastMode()
    m.k = k1
    m.eta_in = eta_in
    if eta_wkb > eta_in * (1.0 + 1e-12):
        x_span = k1 * max(eta_wkb - eta_in, 0.0)
        n_brute = modes._bucket_size(int(np.clip(pts_per_period * x_span / (2.0 * np.pi), n_brute_min, n_brute_max)))
        brute_eta = np.geomspace(eta_in, eta_wkb, n_brute)
        Xb, Pib = modes._solve_ivp_at(k1, bg.f1_Phi, bg.f2_Phi, bg.f3_Phi, eta_in, eta_wkb, X_in, dX_in, jnp.asarray(brute_eta))
        m.brute_eta, m.brute_X, m.brute_Xp = np.asarray(brute_eta), np.asarray(Xb), np.asarray(Pib)
        X0, dX0 = float(m.brute_X[-1]), float(m.brute_Xp[-1])
    else:
        eta_wkb = eta_in
        m.brute_eta, m.brute_X, m.brute_Xp = np.array([eta_in]), np.array([X_in]), np.array([dX_in])
        X0, dX0 = X_in, dX_in

    m.eta_wkb = eta_wkb
    f3_wkb = float(bg.f3_Phi(jnp.asarray(eta_wkb)))
    m.Cplus, m.Cminus = modes.match_C(X0, dX0, k1, f3_wkb)
    return m


def eval_fast(m: FastMode, eta_array, bgtab):
    """eval_mode-equivalent for a FastMode built by match_scalar_fast."""
    eta_array = np.asarray(eta_array, dtype=float)
    X, Xp = np.empty_like(eta_array), np.empty_like(eta_array)
    pre = eta_array <= m.eta_wkb
    post = ~pre
    if np.any(pre):
        X[pre] = np.interp(eta_array[pre], m.brute_eta, m.brute_X)
        Xp[pre] = np.interp(eta_array[pre], m.brute_eta, m.brute_Xp)
    if np.any(post):
        Y0, Z0 = bgtab.Yz(m.eta_wkb)
        Y, Z = bgtab.Yz(eta_array[post])
        f1e, f3e, f3pe = bgtab.f_of(eta_array[post])
        Xw, Xpw = modes._wkb_values_np(m.k, Y - Y0, Z - Z0, f3e, f1e, f3pe, m.Cplus, m.Cminus)
        X[post], Xp[post] = Xw, Xpw
    return X, Xp


def _source_pieces(k, t, s, bg, g1m, g2m, bgtab, n_max, n_min, pts_per_period):
    """Shared expensive part of I2bar_table/I1I2_table: mode solves + source
    quadrature, returning J1, J2 = int g_i a f deta up to eta_c."""
    eta_c = K.ETA_C_FACTOR / k
    u, v = K.u_of(t, s), K.v_of(t, s)
    eta = K.source_grid(k, t, s, eta_c, pts_per_period=pts_per_period, n_max=n_max, n_min=n_min)

    mu = match_scalar_fast(u * k, eta_c, bg, bgtab)
    mv = match_scalar_fast(v * k, eta_c, bg, bgtab)
    Phi1, dPhi1 = eval_fast(mu, eta, bgtab)
    Phi2, dPhi2 = eval_fast(mv, eta, bgtab)

    g1, _ = modes.eval_mode(g1m, eta)
    g2, _ = modes.eval_mode(g2m, eta)

    w = np.asarray(bg.w_of_eta(eta))
    H = np.asarray(bg.H_of_eta(eta))
    a = np.asarray(bg.a_of_eta(eta))
    f = K.source_f(Phi1, dPhi1, Phi2, dPhi2, w, H)

    J1 = cumulative_trapezoid(g1 * a * f, eta, initial=0.0)[-1]
    J2 = cumulative_trapezoid(g2 * a * f, eta, initial=0.0)[-1]
    return eta_c, J1, J2


def I2bar_table(k, t, s, bg, g1m, g2m, bgtab, n_periods_avg=8, n_per_period=20,
                pts_per_period=25.0, n_max=60000, n_min=800):
    """Ibar2(t,s) at external k: oscillation-averaged x_c^2 <I^2>."""
    eta_c, J1, J2 = _source_pieces(k, t, s, bg, g1m, g2m, bgtab, n_max, n_min, pts_per_period)

    period = 2.0 * np.pi / k
    win = np.linspace(eta_c - n_periods_avg * period, eta_c, n_periods_avg * n_per_period)
    win = win[win > K.ETA_IN_FACTOR / k]
    g1w, _ = modes.eval_mode(g1m, win)
    g2w, _ = modes.eval_mode(g2m, win)
    aw = np.asarray(bg.a_of_eta(win))
    I = (k**2 / aw) * (g1w * J2 - g2w * J1)
    return float(np.trapezoid((k * win)**2 * I**2, win) / (win[-1] - win[0]))


def I1I2_table(k, t, s, bg, g1m, g2m, bgtab, n_max=60000, n_min=800, pts_per_period=25.0):
    """I1(t,s), I2(t,s) at external k and eta_c: the two un-squared pieces
    of I = g1k*I2 - g2k*I1, I_i = (k^2/a(eta_c)) J_i, evaluated at eta_c
    (not oscillation-averaged). Needed, together with g_bilinear_averages,
    to reconstruct the cross-oscillation-average at two different (t,s)
    points (see notebook 03 and the Draft's Appendix B)."""
    eta_c, J1, J2 = _source_pieces(k, t, s, bg, g1m, g2m, bgtab, n_max, n_min, pts_per_period)
    a_c = float(bg.a_of_eta(eta_c))
    return float((k**2 / a_c) * J1), float((k**2 / a_c) * J2)


def g_bilinear_averages(k, bg, g1m, g2m, n_periods_avg=8, n_per_period=20):
    """Gab = <(k eta)^2 (a_c/a(eta))^2 ga(eta) gb(eta)>, oscillation-averaged
    over the same window as I2bar_table. k-only (not t,s-dependent).
    Combined with I1,I2 from I1I2_table:
        Ibar(1,2) = I2(1)I2(2) G11 - [I1(1)I2(2)+I2(1)I1(2)] G12 + I1(1)I1(2) G22,
    which reduces exactly to I2bar_table's own output at (t1,s1)=(t2,s2)."""
    eta_c = K.ETA_C_FACTOR / k
    period = 2.0 * np.pi / k
    win = np.linspace(eta_c - n_periods_avg * period, eta_c, n_periods_avg * n_per_period)
    win = win[win > K.ETA_IN_FACTOR / k]
    g1w, _ = modes.eval_mode(g1m, win)
    g2w, _ = modes.eval_mode(g2m, win)
    aw = np.asarray(bg.a_of_eta(win))
    a_c = float(bg.a_of_eta(eta_c))
    wsq = (k * win)**2 * (a_c / aw)**2
    norm = win[-1] - win[0]
    G11 = float(np.trapezoid(wsq * g1w * g1w, win) / norm)
    G22 = float(np.trapezoid(wsq * g2w * g2w, win) / norm)
    G12 = float(np.trapezoid(wsq * g1w * g2w, win) / norm)
    return G11, G22, G12


def t_grid_universal(t_min=1e-3, t_max=3000.0, n_per_decade=6, n_seg_res=20, eps=1e-6):
    """Pzeta-independent (t,s) grid clustered around the resonance."""
    from spectra import _seg_log, _seg_toward
    tr = K.T_RESONANCE
    w = min(0.2, 0.5 * (tr - t_min), 0.5 * (t_max - tr))
    segs = []
    if t_min < tr - w:
        n = max(int(np.log10((tr - w) / t_min) * n_per_decade), 4)
        segs.append(_seg_log(t_min, tr - w, n))
    segs.append(_seg_toward(tr, w, n_seg_res, side=-1, eps=eps))
    segs.append(_seg_toward(tr, w, n_seg_res, side=+1, eps=eps))
    if tr + w < t_max:
        n = max(int(np.log10(t_max / (tr + w)) * n_per_decade), 4)
        segs.append(_seg_log(tr + w, t_max, n))
    t_all = np.concatenate([seg[0] for seg in segs])
    order = np.argsort(t_all)
    return t_all[order]
