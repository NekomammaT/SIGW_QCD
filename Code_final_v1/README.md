# SIGW QCD kernels

Scalar-induced gravitational wave (SIGW) transfer functions across the
QCD crossover, and the code used to produce every figure of the
accompanying Draft.

## Layout

```
core/                 physics modules (background, kernel, spectra, ...)
notebooks/            one notebook per figure (or figure group)
export_kernels.py     exports kernels_data/, kernels_data_I1I2/
export_kernels_ng15.py  kernel at the 14 NANOGrav 15yr frequencies
mcmc_pta_bpl.py        broken-power-law fit to the NANOGrav 15yr likelihood
kernels_data/          I2bar(t,s;k), 110 external-k nodes
kernels_data_I1I2/      I1, I2, G11, G22, G12 at the same nodes
data/                   MCMC chains, NG15 likelihood tables, NG caches
```

## core/

| module | contents |
|---|---|
| `thermo.py` | Saikawa-Shirai `g*_rho(T)`, `g*_s(T)` fit |
| `lattice_gstar.py` | lattice-QCD `g*_rho(T)`, `g*_s(T)`, stitched to `thermo.py` |
| `background_qcd.py` | `T(eta)`, `a(eta)`, `H(eta)`, `w(eta)`, `cs2(eta)` |
| `modes.py` | generic WKB mode-function solver |
| `kernel.py` | second-order kernel `I(t,s,eta)` in Kohri-Terada `(t,s)` variables |
| `kernel_table.py` | fast tabulation of `I2bar`, `I1`, `I2` for export |
| `ng_kernels.py` | exact non-Gaussian (`F_NL^2`) correction on the QCD kernel |
| `spectra.py` | `P_h(k)`, `Omega_GW(k)` by direct `(t,s)` quadrature |
| `plotstyle.py`, `runtime.py` | plotting/process-safety utilities |

## notebooks/

Each notebook reads `core/` and the stored tables directly and writes its
figure(s) into `../../Draft/Figures/`.

| notebook | figures | runtime |
|---|---|---|
| `01_background_and_eos.ipynb` | 1, 2 | seconds |
| `02_gaussian_spectra.ipynb` | 3, 4, 5 | seconds |
| `03_pta_fit.ipynb` | 6, 7 | seconds (reads pre-computed chains) |
| `04_non_gaussian_spectra.ipynb` | 8, 9 | seconds (reads cache) / ~15-30 min per panel without it |

## Reproducing the tables and chains from scratch

```bash
python3 export_kernels.py             # ~2 days, checkpointed, --resume-able
python3 export_kernels_ng15.py        # ~1 hour
python3 mcmc_pta_bpl.py rd             # ~70 min each
python3 mcmc_pta_bpl.py qcd
```

`export_kernels.py --benchmark` estimates the runtime on the current
machine before committing to a full run.

## Data

`data/likelihood_SMBHmodel_v2.h5` is 225 MB, over GitHub's 100 MB
per-file limit -- use Git LFS (or host it externally) before pushing
this package to GitHub.
