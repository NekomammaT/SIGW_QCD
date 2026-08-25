"""Exact non-Gaussian (F_NL^2) correction to the SIGW spectrum on the QCD
background (Draft Appendix, "hybrid", "t", "u" diagrams; the "s" diagram
vanishes identically by angular integration).

Uses kernels_data/ (I2bar, for "hybrid") and kernels_data_I1I2/ (I1, I2,
G11=<g1k^2>, G22=<g2k^2>, G12=<g1k g2k>, for the "t"/"u" cross term),
evaluated exactly at the tabulated external-k nodes. The
cross-oscillation-average at two different (t,s) points is

    x^2 <I(t1,s1) I(t2,s2)> = I2(1)I2(2) G11 - [I1(1)I2(2)+I2(1)I1(2)] G12 + I1(1)I1(2) G22 .

I1, I2 are even in s, matching the tabulated s>=0 convention.

For an exactly scale-invariant (flat) P_zeta, two loop-momentum
configurations in the bulk of the integration domain are not integrable:
the corner u,v->0 in "hybrid", and the collinear limit q1->q2 (or
k-q2) in "t"/"u". An explicit infrared floor IR_REG regulates both.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import kernel as K
from spectra import _seg_log, _seg_toward

IR_REG = 0.1


class NodeKernels:
    """I2bar(t,s), I1(t,s), I2(t,s), G11, G22, G12 at one external-k node."""

    def __init__(self, tab_i2bar, tab_i1i2):
        t, s = tab_i2bar["t"], tab_i2bar["s"]
        assert np.allclose(t, tab_i1i2["t"]) and np.allclose(s, tab_i1i2["s"]), \
            "kernels_data and kernels_data_I1I2 must share the same (t,s) grid"
        keep = np.concatenate([[True], np.diff(t) > 0])
        t = t[keep]
        I2bar_g, I1_g, I2_g = tab_i2bar["I2bar"][keep], tab_i1i2["I1"][keep], tab_i1i2["I2"][keep]
        self._t_min, self._t_max = float(t[0]), float(t[-1])
        self._s_max = float(s[-1])
        self._i2bar = RegularGridInterpolator((t, s), I2bar_g,
                                              bounds_error=False, fill_value=None)
        self._i1 = RegularGridInterpolator((t, s), I1_g,
                                           bounds_error=False, fill_value=None)
        self._i2 = RegularGridInterpolator((t, s), I2_g,
                                           bounds_error=False, fill_value=None)
        self.G11 = float(tab_i1i2["G11"])
        self.G22 = float(tab_i1i2["G22"])
        self.G12 = float(tab_i1i2["G12"])

    def _pts(self, t, s):
        t = np.clip(np.asarray(t, dtype=float), self._t_min, self._t_max)
        s = np.clip(np.abs(np.asarray(s, dtype=float)), 0.0, self._s_max)
        return np.stack([t.ravel(), s.ravel()], axis=-1)

    def i2bar(self, t, s):
        t, s = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(s, dtype=float))
        return self._i2bar(self._pts(t, s)).reshape(t.shape)

    def i1i2(self, t, s):
        t, s = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(s, dtype=float))
        pts = self._pts(t, s)
        return self._i1(pts).reshape(t.shape), self._i2(pts).reshape(t.shape)

    def cross(self, t1, s1, t2, s2):
        """x^2 <I(t1,s1) I(t2,s2)>."""
        I1_1, I2_1 = self.i1i2(t1, s1)
        I1_2, I2_2 = self.i1i2(t2, s2)
        return (I2_1 * I2_2 * self.G11 - (I1_1 * I2_2 + I2_1 * I1_2) * self.G12
                + I1_1 * I1_2 * self.G22)


def Jfun(t, s):
    """J(t,s) = t(t+2)(1-s^2) I(t,s)."""
    return t * (2.0 + t) * (1.0 - s**2)


def J2bar(nk: NodeKernels, t, s):
    return Jfun(t, s)**2 * nk.i2bar(t, s)


def y_pieces(t1, s1, t2, s2):
    """Phi-independent building blocks of y_of."""
    A1 = t1 * (t1 + 2.0) * (s1**2 - 1.0)
    A2 = t2 * (t2 + 2.0) * (s2**2 - 1.0)
    sqrt_A1A2 = np.sqrt(np.clip(A1 * A2, 0.0, None))
    cross_const = 0.25 * (1.0 - s1 * (t1 + 1.0)) * (1.0 - s2 * (t2 + 1.0))
    return sqrt_A1A2, cross_const


def y_of(t1, s1, t2, s2, phi):
    sqrt_A1A2, cross_const = y_pieces(t1, s1, t2, s2)
    return 0.25 * np.cos(phi) * sqrt_A1A2 + cross_const


def w_of(t1, s1, t2, s2, phi, reg=IR_REG):
    """|q1-q2|/k, floored at `reg`."""
    v1, v2 = K.v_of(t1, s1), K.v_of(t2, s2)
    y = y_of(t1, s1, t2, s2, phi)
    return np.sqrt(np.clip(v1**2 + v2**2 - 2.0 * y, reg**2, None))


def wbar_of(t1, s1, t2, s2, phi, reg=IR_REG):
    """|k-q1-q2|/k, floored at `reg`."""
    v1, v2 = K.v_of(t1, s1), K.v_of(t2, s2)
    z1 = 0.5 * (1.0 - s1 * (t1 + 1.0))
    z2 = 0.5 * (1.0 - s2 * (t2 + 1.0))
    y = y_of(t1, s1, t2, s2, phi)
    return np.sqrt(np.clip(1.0 + v1**2 + v2**2 - 2.0 * z1 - 2.0 * z2 + 2.0 * y, reg**2, None))


def _w_wbar_of_y(t1, s1, t2, s2, y, reg=IR_REG):
    """w_of, wbar_of from an already-computed y."""
    v1, v2 = K.v_of(t1, s1), K.v_of(t2, s2)
    w = np.sqrt(np.clip(v1**2 + v2**2 - 2.0 * y, reg**2, None))
    z1 = 0.5 * (1.0 - s1 * (t1 + 1.0))
    z2 = 0.5 * (1.0 - s2 * (t2 + 1.0))
    wbar = np.sqrt(np.clip(1.0 + v1**2 + v2**2 - 2.0 * z1 - 2.0 * z2 + 2.0 * y, reg**2, None))
    return w, wbar


def _ts_grid_flat(n_t_seg, n_s, t_min, t_max, eps=1e-6):
    """Flattened (t,s) grid, 2D trapezoidal weights, t-axis clustered
    around the kernel resonance. `n_t_seg` is points per segment; the
    concatenated t-axis ends up with roughly 3.8x that many points."""
    tr = K.T_RESONANCE
    w = min(0.2, 0.5 * (tr - t_min), 0.5 * (t_max - tr))
    segs = []
    if t_min < tr - w:
        n = max(int(np.log10((tr - w) / t_min) * n_t_seg), 4)
        segs.append(_seg_log(t_min, tr - w, n)[0])
    segs.append(_seg_toward(tr, w, n_t_seg, side=-1, eps=eps)[0])
    segs.append(_seg_toward(tr, w, n_t_seg, side=+1, eps=eps)[0])
    if tr + w < t_max:
        n = max(int(np.log10(t_max / (tr + w)) * n_t_seg), 4)
        segs.append(_seg_log(tr + w, t_max, n)[0])
    t = np.unique(np.concatenate(segs))

    s = np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, n_s)
    wt, ws = np.gradient(t), np.gradient(s)
    T, S = np.meshgrid(t, s, indexing="ij")
    W = np.outer(wt, ws)
    return T.ravel(), S.ravel(), W.ravel()


def P_h_FNL2_unit(nk: NodeKernels, k, pz, n_t_seg=30, n_s=41, n_phi=28, t_min=2.0 * IR_REG,
                  t_max=30.0, reg=IR_REG, **_ignored):
    """P_h^{F_NL^2}(k) / F_NL^2 (hybrid + t + u), on a fixed,
    resonance-clustered (t,s) grid per loop and a fixed phi grid."""
    t1, s1, w1 = _ts_grid_flat(n_t_seg, n_s, t_min, t_max)
    u1, v1 = K.u_of(t1, s1), K.v_of(t1, s1)

    t1c, s1c, w1c = t1[:, None], s1[:, None], w1[:, None]
    u1c, v1c = u1[:, None], v1[:, None]
    t2r, s2r, w2r = t1[None, :], s1[None, :], w1[None, :]
    u2r, v2r = u1[None, :], v1[None, :]
    W2 = w1c * w2r

    J2_1 = J2bar(nk, t1, s1)[:, None]
    hybrid = np.sum(
        J2_1 / np.clip(u1c * v1c * u2r * v2r, reg**2, None)**2
        * pz(k * v1c * v2r) * pz(k * u1c) * pz(k * v1c * u2r)
        * W2
    )

    JJ = Jfun(t1, s1)[:, None] * Jfun(t1, s1)[None, :] * nk.cross(t1c, s1c, t2r, s2r)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi

    sqrt_A1A2, y_const = y_pieces(t1c, s1c, t2r, s2r)
    pz_v2r, pz_u2r, pz_v1c = pz(k * v2r), pz(k * u2r), pz(k * v1c)
    JJ_t = JJ * pz_v2r / v2r**3 * pz_u2r / u2r**3
    JJ_u = JJ * pz_v1c / v1c**3 * pz_v2r / v2r**3

    term_t = 0.0
    term_u = 0.0
    for p in phi:
        cp = np.cos(2.0 * p)
        y = 0.25 * np.cos(p) * sqrt_A1A2 + y_const
        w12, wbar12 = _w_wbar_of_y(t1c, s1c, t2r, s2r, y, reg=reg)

        it = (cp / np.pi) * JJ_t * pz(k * w12) / w12**3
        term_t += dphi * np.sum(it * W2)

        iu = (cp / np.pi) * JJ_u * pz(k * wbar12) / wbar12**3
        term_u += dphi * np.sum(iu * W2)

    return 0.125 * (hybrid + term_t + term_u)


def omega_gw_FNL2_unit(nk: NodeKernels, k, pz, bg, eta_f, eta_c=None, **kw):
    """Omega_GW h^2 / F_NL^2 from the F_NL^2 term at wavenumber k [Mpc^-1].
    eta_c defaults to K.ETA_C_FACTOR/k, matching the Gaussian piece."""
    import spectra as sp
    Ph_unit = P_h_FNL2_unit(nk, k, pz, **kw)
    if eta_c is None:
        eta_c = K.ETA_C_FACTOR / k
    return sp.omega_gw(k, Ph_unit, bg, eta_f, eta_c=eta_c)
