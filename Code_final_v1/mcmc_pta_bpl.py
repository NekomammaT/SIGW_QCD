"""Fit the NANOGrav 15yr free-spectrum likelihood with a scalar-induced GW
background from a broken-power-law curvature spectrum,

    P_zeta(k) = A (alpha+beta)^gamma
                / [ beta (k/k_*)^(-alpha/gamma) + alpha (k/k_*)^(beta/gamma) ]^gamma ,

on either the exact-RD or the QCD-crossover kernel.

Usage:  python3 mcmc_pta_bpl.py qcd
Output: data/chain_pta_bpl_qcd.npz
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
import runtime

import numpy as np
import h5py
import emcee

import kernel as K
import kernel_table as KT
import spectra as sp
import background_qcd as qcd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DATA_STATIC = DATA
ETA_F = 0.5

# NG15 binned likelihood, at fixed nuisance parameters (pbh, fref, alpha)
# of the accompanying SMBH-binary model. pbh=0 switches the SMBH channel
# off; the tabulated grid's pbh axis starts at 0.01, so pbh=0 is reached
# by linear extrapolation in ln L from the two lowest nodes, skipped
# wherever either node sits on the tabulation's floor.
PBH_FIX, FREF_FIX, ALPHA_FIX = 0.0, 4.0, 0.8
LFLOOR = 1e-99          # tabulation's floor
MAX_EXTRAP = 5.0        # cap on the |ln L| correction

# Which quantity the tabulated "A" axis holds: the reference likelihood is
# evaluated at log10(Omega_GW), not log10(Omega_GW h^2), a rigid
# 2 log10(h) shift. Set by the environment variable OMEGA_CONVENTION;
# "no_h2" (default) reproduces the reference implementation.
H_LITTLE = 0.674
CONVENTION = os.environ.get("OMEGA_CONVENTION", "no_h2")
LOG_SHIFT = 0.0 if CONVENTION == "h2" else -2.0 * np.log10(H_LITTLE)


def load_likelihood():
    with h5py.File(os.path.join(DATA_STATIC, "likelihood_SMBHmodel_params_v2.h5"), "r") as f:
        pbh = np.array(f["pbh"]); fref = np.array(f["fref"])
        alpha = np.array(f["alpha"]); Agrid = np.array(f["A"]); fNG = np.array(f["fNG15"])
    jf = int(np.argmin(abs(fref - FREF_FIX)))
    ka = int(np.argmin(abs(alpha - ALPHA_FIX)))
    assert abs(fref[jf]-FREF_FIX) < 1e-9 and abs(alpha[ka]-ALPHA_FIX) < 1e-9, \
        "fref and alpha must land on grid nodes"
    with h5py.File(os.path.join(DATA_STATIC, "likelihood_SMBHmodel_v2.h5"), "r") as f:
        L0 = np.stack([np.array(f[f"f{i}"])[0, jf, ka, :] for i in range(1, 15)])
        L1 = np.stack([np.array(f[f"f{i}"])[1, jf, ka, :] for i in range(1, 15)])
    lg0, lg1 = np.log(1e-300 + L0), np.log(1e-300 + L1)
    if abs(PBH_FIX - pbh[0]) < 1e-12:
        return fNG, Agrid, lg0                            # exactly the lowest node
    # linear extrapolation in ln L along the pbh axis, only where both nodes
    # carry real information (neither sits on the tabulation's floor)
    c = (PBH_FIX - pbh[0]) / (pbh[1] - pbh[0])
    corr = np.clip(c * (lg1 - lg0), -MAX_EXTRAP, MAX_EXTRAP)
    ok = (L0 > LFLOOR) & (L1 > LFLOOR)
    return fNG, Agrid, np.where(ok, lg0 + corr, lg0)


# ----------------------------------------------------------------------
# quadrature: Omega_i = sum_{t,s} QW_i(t,s) Pz(k_i u) Pz(k_i v)
# ----------------------------------------------------------------------
def _trapz_w(x):
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2]); w[0] = 0.5 * (x[1] - x[0]); w[-1] = 0.5 * (x[-1] - x[-2])
    return w


def build_quadrature(model):
    """Returns QW (14,N) float32, LNKU/LNKV (14,N) float32, f_Hz (14,)."""
    if model == "qcd":
        d = np.load(os.path.join(DATA, "kernels_ng15.npz"))
        t_g, s_g, ks, f_Hz = d["t"], d["s"], d["k"], d["f_Hz"]
        I2 = d["I2bar"]                                    # (14, n_t, n_s)
        assert not np.isnan(I2).any(), "kernel table has NaNs"
        eta_c_of_k = lambda k: K.ETA_C_FACTOR / k          # full solve: eta_c = 400/k
    else:
        T_OBS_S = 16.03 * 3.15e7
        f_Hz = np.arange(1, 15) / T_OBS_S
        ks = sp.f_Hz_to_k(f_Hz)
        t_g = KT.t_grid_universal(1e-3, 3000.0, 25, 100)
        s_g = sp.s_grid(14)
        T, S = np.meshgrid(t_g, s_g, indexing="ij")
        I2 = np.repeat(K.I2bar_rd_exact(T, S)[None], 14, axis=0)   # k-independent
        eta_c_of_k = lambda k: 1.0 / k                     # exact-RD kernel: horizon crossing

    T, S = np.meshgrid(t_g, s_g, indexing="ij")
    W = K.weight(T, S)
    U, V = K.u_of(T, S), K.v_of(T, S)
    wt = _trapz_w(t_g)[:, None] * _trapz_w(s_g)[None, :]

    QW = np.empty((14, T.size)); LNKU = np.empty_like(QW); LNKV = np.empty_like(QW)
    af = float(qcd.a_of_eta(np.asarray(ETA_F))); Hf = float(qcd.H_of_eta(np.asarray(ETA_F)))
    for i, k in enumerate(ks):
        k = float(k)
        if model == "qcd":
            norm = sp.OMEGA_R0_H2 * sp.omega_gw(k, 1.0, qcd, ETA_F, eta_c=eta_c_of_k(k))
        else:
            # exact-RD kernel: only the dilution factor comes from the real
            # background, evaluated at horizon crossing eta=1/k
            ec = 1.0 / k
            ac = float(qcd.a_of_eta(np.asarray(ec))); Hc = float(qcd.H_of_eta(np.asarray(ec)))
            norm = sp.OMEGA_R0_H2 * (ac * Hc / (af * Hf)) ** 2 / 24.0
        QW[i] = (4.0 * norm * wt * W * I2[i]).ravel()
        LNKU[i] = np.log(k * U).ravel()
        LNKV[i] = np.log(k * V).ravel()
    return (QW.astype(np.float32), LNKU.astype(np.float32),
            LNKV.astype(np.float32), f_Hz)


# ----------------------------------------------------------------------
# model + posterior
# ----------------------------------------------------------------------
GAMMA = 1.0            # BPL smoothing exponent, held fixed

# log10A, log10k*, alpha, beta
PRIOR_LO = np.array([-3.0, 6.0, 0.1, 0.1])
PRIOR_HI = np.array([0.0, 9.0, 8.0, 8.0])


def omega_of_theta(theta, QW, LNKU, LNKV):
    """theta (nw,4) -> Omega_GW h^2 (nw,14). P_zeta = A (a+b)^g /
    [b e^{-aL/g} + a e^{bL/g}]^g with L=ln(k/k_*), evaluated in log space
    via logaddexp to avoid float32 overflow."""
    th = np.atleast_2d(theta).astype(np.float32)
    lnA = (th[:, 0] * np.float32(np.log(10.0)))[:, None, None]
    lnks = (th[:, 1] * np.float32(np.log(10.0)))[:, None, None]
    a = th[:, 2][:, None, None]; b = th[:, 3][:, None, None]
    g = np.float32(GAMMA)
    C = lnA + g * np.log(a + b)

    def ln_pz(LNK):
        L = LNK[None, :, :] - lnks
        return C - g * np.logaddexp(np.log(b) - (a / g) * L, np.log(a) + (b / g) * L)

    return np.einsum("kn,ikn->ik", QW, np.exp(ln_pz(LNKU) + ln_pz(LNKV)))


def make_log_prob(QW, LNKU, LNKV, Agrid, logL):
    nA = len(Agrid); A0, dA = Agrid[0], Agrid[1] - Agrid[0]

    def log_prob(theta):
        th = np.atleast_2d(theta)
        out = np.full(len(th), -np.inf)
        ok = np.all((th > PRIOR_LO) & (th < PRIOR_HI), axis=1)
        if not ok.any():
            return out
        om = omega_of_theta(th[ok], QW, LNKU, LNKV)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.log10(np.where(om > 0, om, 1e-300)) + LOG_SHIFT
        idx = np.clip((np.clip(x, Agrid[0], Agrid[-1]) - A0) / dA, 0, nA - 1 - 1e-9)
        i0 = idx.astype(np.int64); w = (idx - i0).astype(np.float64)
        bins = np.arange(14)[None, :]
        ll = ((1 - w) * logL[bins, i0] + w * logL[bins, i0 + 1]).sum(axis=1)
        out[ok] = np.where(np.isfinite(ll), ll, -np.inf)
        return out

    return log_prob


def main():
    model = (sys.argv[1] if len(sys.argv) > 1 else "qcd").lower()
    assert model in ("rd", "qcd")
    fNG_lik, Agrid, logL = load_likelihood()
    QW, LNKU, LNKV, f_Hz = build_quadrature(model)
    assert np.allclose(f_Hz, fNG_lik, rtol=2e-3), "model and likelihood frequencies differ"
    log_prob = make_log_prob(QW, LNKU, LNKV, Agrid, logL)

    nwalkers = 32
    nsteps = int(os.environ.get("NSTEPS", 60000))
    thin_by = int(os.environ.get("THIN_BY", 5))
    burn = int(os.environ.get("BURN", 10000))
    rng = np.random.default_rng(1234)
    p0 = np.column_stack([rng.uniform(-2.0, -0.5, nwalkers), rng.uniform(7.0, 8.5, nwalkers),
                          rng.uniform(1.0, 5.0, nwalkers), rng.uniform(1.0, 5.0, nwalkers)])
    t0 = time.time()
    sampler = emcee.EnsembleSampler(
        nwalkers, 4, log_prob, vectorize=True,
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
    sampler.run_mcmc(p0, nsteps, thin_by=thin_by, progress=False)
    dt = time.time() - t0

    chain = sampler.get_chain(discard=burn, flat=True)
    lnp = sampler.get_log_prob(discard=burn, flat=True)
    try:
        tau = sampler.get_autocorr_time(discard=burn, quiet=True)
    except Exception:
        tau = np.full(4, np.nan)
    print(f"[{model}] {nsteps*thin_by} iters ({nsteps} stored), {dt/60:.1f} min, acc={np.mean(sampler.acceptance_fraction):.3f}, "
          f"tau={np.round(tau,1)}, Neff={len(chain)/np.nanmax(tau):.0f}")
    lab = ["log10A", "log10k*", "alpha", "beta"]
    for j, l in enumerate(lab):
        q = np.percentile(chain[:, j], [16, 50, 84])
        print(f"   {l:8s} = {q[1]:7.3f} +{q[2]-q[1]:.3f} -{q[1]-q[0]:.3f}")
    best = chain[np.argmax(lnp)]
    print(f"   max-post: {dict(zip(lab, np.round(best,3)))}  lnL={lnp.max():.2f}")
    np.savez(os.path.join(DATA, f"chain_pta_bpl_{model}{os.environ.get('CHAIN_SUFFIX','')}.npz"), convention=CONVENTION,
             chain=chain, lnp=lnp, tau=tau, labels=lab, f_Hz=f_Hz,
             nsteps=nsteps, thin_by=thin_by, burn=burn,
             omega_best=omega_of_theta(best[None], QW, LNKU, LNKV)[0],
             acc=np.mean(sampler.acceptance_fraction), runtime_min=dt/60)
    print(f"   saved data/chain_pta_bpl_{model}{os.environ.get('CHAIN_SUFFIX','')}.npz")


if __name__ == "__main__":
    main()
