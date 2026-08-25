"""Generic WKB mode-function engine.

Solves X'' + f1(eta) X' + [k^2 f3(eta) + f2(eta)] X = 0 by brute-force ODE
integration from eta_in to eta_wkb, then continues analytically using the
WKB ansatz

    X(eta) = exp(-y/2) f3^{-1/4} [C+ exp(i k z) + C- exp(-i k z)]

with y' = f1, z' = sqrt(f3) (both zero at eta_wkb), and C+/C- fixed by
matching X, X' at eta_wkb. eta_wkb is the point beyond which
k^2 f3 >> |f2| + |f1'|/2 + f1^2/4 (with a safety margin), so the WKB
frequency is guaranteed positive. Phi (f1!=0) and the tensor Green's
functions g1,g2 (f1=0) are both special cases of this same equation.
"""
from dataclasses import dataclass
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx

jax.config.update("jax_enable_x64", True)

_deriv_cache = {}


def _deriv(f):
    """Pointwise derivative of f(eta), memoized+JIT-compiled by identity of f."""
    cached = _deriv_cache.get(f)
    if cached is not None:
        return cached
    g = jax.grad(f)

    @jax.jit
    def df(eta):
        eta = jnp.asarray(eta, dtype=jnp.float64)
        return g(eta) if eta.ndim == 0 else jax.vmap(g)(eta)

    _deriv_cache[f] = df
    return df


@partial(jax.jit, static_argnames=("f1", "f2", "f3", "n_scan"))
def find_eta_wkb(k, f1, f2, f3, eta_lo, eta_hi, n_scan=8000, eps=1e-3, safety=20.0):
    """Smallest eta beyond which the WKB frequency is safely positive,
    pushed out by `safety` for margin."""
    f1p = _deriv(f1)
    etas = jnp.geomspace(eta_lo, eta_hi, n_scan)
    dominant = k**2 * f3(etas)
    sub = jnp.abs(f2(etas)) + 0.5 * jnp.abs(f1p(etas)) + 0.25 * f1(etas)**2
    ok = sub / dominant < eps
    rev_ok = jnp.flip(ok.astype(jnp.int32))
    stays_true = jnp.flip(jax.lax.cummin(rev_ok)).astype(bool)
    idx = jnp.argmax(stays_true)
    eta_star = jnp.where(jnp.any(stays_true), etas[idx], eta_hi)
    return jnp.clip(safety * eta_star, eta_lo, eta_hi), eta_star


@partial(jax.jit, static_argnames=("f1", "f2", "f3", "max_steps"))
def _solve_ivp_at_jit(k, f1, f2, f3, eta0, eta1, X0, dX0, ts, max_steps):
    def vf(eta, y, args):
        X, Pi = y
        return jnp.array([Pi, -f1(eta) * Pi - (k**2 * f3(eta) + f2(eta)) * X])
    sol = dfx.diffeqsolve(
        dfx.ODETerm(vf), dfx.Dopri8(), t0=eta0, t1=eta1, dt0=(eta1 - eta0) / 1000.0,
        y0=jnp.array([X0, dX0]), stepsize_controller=dfx.PIDController(rtol=1e-9, atol=1e-11),
        saveat=dfx.SaveAt(ts=ts), max_steps=max_steps,
    )
    return sol.ys[:, 0], sol.ys[:, 1]


def _solve_ivp_at(k, f1, f2, f3, eta0, eta1, X0, dX0, ts, max_steps=2**20):
    """Brute-force solve, X,X' sampled exactly at `ts` (ts[-1] = eta1)."""
    return _solve_ivp_at_jit(k, f1, f2, f3, eta0, eta1, X0, dX0, ts, max_steps)


@partial(jax.jit, static_argnames=("f1", "f3", "max_steps"))
def _solve_yz_at_jit(f1, f3, eta0, eta1, ts, max_steps):
    def vf(eta, yz, args):
        return jnp.array([f1(eta), jnp.sqrt(f3(eta))])
    sol = dfx.diffeqsolve(
        dfx.ODETerm(vf), dfx.Dopri8(), t0=eta0, t1=eta1, dt0=(eta1 - eta0) / 1000.0,
        y0=jnp.array([0.0, 0.0]), stepsize_controller=dfx.PIDController(rtol=1e-10, atol=1e-12),
        saveat=dfx.SaveAt(ts=ts), max_steps=max_steps,
    )
    return sol.ys[:, 0], sol.ys[:, 1]


def _solve_yz_at(f1, f3, eta0, eta1, ts, max_steps=2**20):
    """Cumulative integrals y'=f1, z'=sqrt(f3), both zero at eta0."""
    return _solve_yz_at_jit(f1, f3, eta0, eta1, ts, max_steps)


_SIZE_BUCKETS = np.unique(np.round(np.geomspace(500, 100000, 40)).astype(int))


def _bucket_size(n):
    """Round n up to a fixed bucket, so nearby calls reuse one JIT compile."""
    idx = np.searchsorted(_SIZE_BUCKETS, n)
    return int(_SIZE_BUCKETS[min(idx, len(_SIZE_BUCKETS) - 1)])


def match_C(X0, dX0, k, f3_wkb):
    """WKB matching coefficients C+, C- at eta_wkb (plain Python complex)."""
    X0, dX0, f3_wkb = float(X0), float(dX0), float(f3_wkb)
    A0 = f3_wkb**(-0.25)
    S1 = f3_wkb**0.5
    Cplus = 0.5 * (X0 / A0 + dX0 / (1j * k * S1 * A0))
    Cminus = 0.5 * (X0 / A0 - dX0 / (1j * k * S1 * A0))
    return complex(Cplus), complex(Cminus)


def _wkb_values_np(k, y, z, f3e, f1e, f3pe, Cplus, Cminus):
    """Plain-numpy WKB evaluation, no JAX tracing."""
    A = np.exp(-0.5 * y) * f3e**(-0.25)
    phase = k * z
    eplus, eminus = np.exp(1j * phase), np.exp(-1j * phase)
    X = A * (Cplus * eplus + Cminus * eminus)
    Ap = A * (-0.5 * f1e - 0.25 * f3pe / f3e)
    Xp = Ap * (Cplus * eplus + Cminus * eminus) \
         + A * (1j * k * np.sqrt(f3e)) * (Cplus * eplus - Cminus * eminus)
    return np.real(X), np.real(Xp)


@dataclass
class ModeMatch:
    """Result of the (expensive) brute-force + WKB-matching phase, reusable
    via `eval_mode` at arbitrary target grids without repeating any solve."""
    k: float
    eta_wkb: float
    eta_star: float
    Cplus: complex
    Cminus: complex
    brute_eta: np.ndarray
    brute_X: np.ndarray
    brute_Xp: np.ndarray
    wkb_eta: np.ndarray
    wkb_y: np.ndarray
    wkb_z: np.ndarray
    wkb_f3: np.ndarray
    wkb_f1: np.ndarray
    wkb_f3p: np.ndarray


def match_mode(k, f1, f2, f3, eta_in, X_in, dX_in, eta_hi_scan, eps=1e-3, safety=2.0,
               full_brute=False, pts_per_period=25.0, n_brute_min=1000, n_brute_max=20000,
               n_wkb_per_decade=2000, n_wkb_min=500, n_wkb_max=40000):
    """Find eta_wkb, solve the brute-force segment once, solve the smooth
    WKB integrals y,z once from eta_wkb to eta_hi_scan."""
    k = float(k)
    eta_wkb_j, eta_star_j = find_eta_wkb(k, f1, f2, f3, eta_in, eta_hi_scan, eps=eps, safety=safety)
    eta_wkb = min(float(eta_wkb_j), eta_hi_scan)
    eta_star = float(eta_star_j)
    if full_brute:
        eta_wkb = eta_hi_scan

    if eta_wkb > eta_in * (1.0 + 1e-12):
        x_span = k * max(eta_wkb - eta_in, 0.0)
        n_brute = _bucket_size(int(np.clip(pts_per_period * x_span / (2.0 * np.pi), n_brute_min, n_brute_max)))
        brute_eta = np.geomspace(eta_in, eta_wkb, n_brute)
        Xb, Pib = _solve_ivp_at(k, f1, f2, f3, eta_in, eta_wkb, X_in, dX_in, jnp.asarray(brute_eta))
        brute_X, brute_Xp = np.asarray(Xb), np.asarray(Pib)
        X0, dX0 = brute_X[-1], brute_Xp[-1]
    else:
        eta_wkb = eta_in
        brute_eta = np.array([eta_in])
        brute_X, brute_Xp = np.array([X_in]), np.array([dX_in])
        X0, dX0 = X_in, dX_in

    f3_wkb = float(f3(jnp.asarray(eta_wkb)))
    Cplus, Cminus = match_C(X0, dX0, k, f3_wkb)

    if eta_hi_scan > eta_wkb * (1.0 + 1e-12):
        n_decades = np.log10(eta_hi_scan / eta_wkb)
        n_wkb = _bucket_size(int(np.clip(n_wkb_per_decade * max(n_decades, 0.0), n_wkb_min, n_wkb_max)))
        wkb_eta = np.geomspace(eta_wkb, eta_hi_scan, n_wkb)
        y_g, z_g = _solve_yz_at(f1, f3, eta_wkb, eta_hi_scan, jnp.asarray(wkb_eta))
        wkb_y, wkb_z = np.asarray(y_g), np.asarray(z_g)
        wkb_f3 = np.asarray(f3(jnp.asarray(wkb_eta)))
        wkb_f1 = np.asarray(f1(jnp.asarray(wkb_eta)))
        wkb_f3p = np.asarray(_deriv(f3)(jnp.asarray(wkb_eta)))
    else:
        wkb_eta = np.array([eta_wkb])
        wkb_y, wkb_z = np.array([0.0]), np.array([0.0])
        wkb_f3 = np.array([f3_wkb])
        wkb_f1 = np.array([float(f1(jnp.asarray(eta_wkb)))])
        wkb_f3p = np.array([float(_deriv(f3)(jnp.asarray(eta_wkb)))])

    return ModeMatch(k=k, eta_wkb=eta_wkb, eta_star=eta_star,
                      Cplus=Cplus, Cminus=Cminus,
                      brute_eta=brute_eta, brute_X=brute_X, brute_Xp=brute_Xp,
                      wkb_eta=wkb_eta, wkb_y=wkb_y, wkb_z=wkb_z,
                      wkb_f3=wkb_f3, wkb_f1=wkb_f1, wkb_f3p=wkb_f3p)


def eval_mode(match: ModeMatch, target_eta):
    """Evaluate an already-matched mode at a new target grid: pure numpy
    interpolation, no further ODE solves."""
    target_eta = np.asarray(target_eta, dtype=np.float64)
    X_out = np.empty_like(target_eta)
    Xp_out = np.empty_like(target_eta)
    pre = target_eta <= match.eta_wkb
    post = ~pre

    if np.any(pre):
        X_out[pre] = np.interp(target_eta[pre], match.brute_eta, match.brute_X)
        Xp_out[pre] = np.interp(target_eta[pre], match.brute_eta, match.brute_Xp)

    if np.any(post):
        pts = target_eta[post]
        y = np.interp(pts, match.wkb_eta, match.wkb_y)
        z = np.interp(pts, match.wkb_eta, match.wkb_z)
        f3e = np.interp(pts, match.wkb_eta, match.wkb_f3)
        f1e = np.interp(pts, match.wkb_eta, match.wkb_f1)
        f3pe = np.interp(pts, match.wkb_eta, match.wkb_f3p)
        Xw, Xpw = _wkb_values_np(match.k, y, z, f3e, f1e, f3pe, match.Cplus, match.Cminus)
        X_out[post], Xp_out[post] = Xw, Xpw

    return X_out, Xp_out
