# SIGW QCD kernels -- minimal package

QCD-crossover transfer functions for scalar-induced gravitational waves.
Trimmed, documented reference implementation; see `Code_develop/` in the
parent repo for the full research codebase (MCMC, figure scripts, etc).

## Layout

- `core/` -- the physics modules (import with `sys.path.insert(0, "core")`)
  - `thermo.py` -- g*_rho(T), g*_s(T), w(T), cs2(T) (Saikawa-Shirai fit)
  - `background_qcd.py` -- self-consistent QCD-era background T(eta), a(eta), H(eta)
  - `background_rd.py` -- exact RD background + closed-form mode functions (validation)
  - `modes.py` -- generic brute-force + WKB mode-function solver
  - `kernel.py` -- the SIGW kernel I(t,s,eta), direct (slow) evaluation
  - `kernel_table.py` -- fast per-k tabulation machinery (shared WKB integrals)
  - `kernel_table_io.py` -- load kernels_data/, fold P_zeta through it
  - `spectra.py` -- P_h(k), Omega_GW(k), (t,s) integration grids
  - `plotstyle.py`, `runtime.py` -- plot style; thread/memory caps for parallel export
- `export_kernels.py` -- rebuild `kernels_data/` (run once, ~10-15h, parallel + memory-capped)
- `kernels_data/` -- exported Ibar2(t,s;k) tables (60 external-k nodes)
- `notebooks/`
  - `01_background.ipynb` -- solve and plot the QCD background
  - `02_wkb_matching.ipynb` -- WKB matching, validated against exact RD
  - `03_kernel_at_k.ipynb` -- the kernel Ibar2(t,s) at one external k
  - `04_omega_gw_from_tables.ipynb` -- Omega_GW(f) from the tabulated kernels
  - `05_draft_figures_4_5.ipynb` -- reproduces the paper's Figures 4 and 5
    (flat spectrum vs. linear reference; lognormal spectra + peak ratio),
    using only this folder -- the RD reference is computed live from
    `kernel.I2bar_rd_exact`, not loaded from a stored data file

## Quick start

```bash
cd notebooks && jupyter lab
```

Run `01`, `02`, `03` for the physics from scratch (no data needed beyond
this repo). `04` and `05` need `../kernels_data/` (included; regenerate
with `export_kernels.py` if needed).
