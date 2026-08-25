"""Precompute the QCD-background kernel Ibar2(t,s;k) at the 14 NANOGrav
15yr frequencies, f_i = i / T_obs with T_obs = 16.03 yr. These do not
coincide with the nodes of kernels_data/ and cannot be interpolated
across k in this band (near the resonance the kernel varies by O(1)
between adjacent nodes), so it is solved directly at each frequency.

Output: data/kernels_ng15.npz
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
import runtime  # thread/memory limits -- must precede jax

import numpy as np

import kernel as K
import kernel_table as KT
import spectra as sp

OUT = os.path.join(os.path.dirname(__file__), "data", "kernels_ng15.npz")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

T_OBS_S = 16.03 * 3.15e7            # NG15 observing baseline
F_NG15 = np.arange(1, 15) / T_OBS_S  # 14 frequencies, Hz

T_MIN, T_MAX = 1e-3, 3000.0
N_PER_DECADE, N_SEG_RES = 25, 100
N_S = 14
N_MAX = 90000
N_WORKERS = 6
ROWS_PER_CHUNK = 8                  # t-rows per task (x N_S points)
RSS_HARD_GB = 1.4


def _worker(args):
    i_f, f_Hz, i_lo, i_hi = args
    import runtime as rt
    import importlib
    qcd = importlib.import_module("background_qcd")
    k = float(sp.f_Hz_to_k(np.array([f_Hz]))[0])

    g1m, g2m = K.match_tensor(k, qcd)
    bgtab = KT.BackgroundTables(qcd.f1_Phi, qcd.f3_Phi, 1e-20, 2.0, n=8000)
    t_grid = KT.t_grid_universal(T_MIN, T_MAX, N_PER_DECADE, N_SEG_RES)
    s_grid = sp.s_grid(N_S)

    out = np.empty((i_hi - i_lo, N_S))
    n = 0
    for a, i in enumerate(range(i_lo, i_hi)):
        for j in range(N_S):
            out[a, j] = KT.I2bar_table(k, float(t_grid[i]), float(s_grid[j]),
                                       qcd, g1m, g2m, bgtab, n_max=N_MAX)
            n += 1
            if n % 25 == 0 and rt.rss_gb() > RSS_HARD_GB:
                raise MemoryError(f"f#{i_f} rows[{i_lo},{i_hi}) RSS={rt.rss_gb():.2f}GB")
    return i_f, i_lo, i_hi, out


def main():
    t_grid = KT.t_grid_universal(T_MIN, T_MAX, N_PER_DECADE, N_SEG_RES)
    s_grid = sp.s_grid(N_S)
    n_t = len(t_grid)
    ks = sp.f_Hz_to_k(F_NG15)

    tasks = []
    for i_f, f in enumerate(F_NG15):
        for lo in range(0, n_t, ROWS_PER_CHUNK):
            tasks.append((i_f, float(f), lo, min(lo + ROWS_PER_CHUNK, n_t)))
    print(f"{len(F_NG15)} NG15 frequencies, n_t={n_t}, n_s={N_S} "
          f"({n_t*N_S} pts/freq, {len(F_NG15)*n_t*N_S} total), "
          f"{len(tasks)} tasks, {N_WORKERS} workers", flush=True)

    I2 = np.full((len(F_NG15), n_t, N_S), np.nan)
    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
    t0 = time.time()
    done_n = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS, max_tasks_per_child=1) as ex:
        pending = {ex.submit(_worker, t): t for t in tasks}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                task = pending.pop(fut)
                try:
                    i_f, lo, hi, blk = fut.result()
                except Exception as e:
                    i_f, f, lo, hi = task
                    if hi - lo <= 1:
                        print(f"  !!! giving up f#{i_f} rows[{lo},{hi}): {e}", flush=True)
                        done_n += 1
                        continue
                    mid = lo + (hi - lo) // 2
                    print(f"  split f#{i_f} rows[{lo},{hi}) ({e})", flush=True)
                    for t2 in [(i_f, f, lo, mid), (i_f, f, mid, hi)]:
                        pending[ex.submit(_worker, t2)] = t2
                    continue
                I2[i_f, lo:hi, :] = blk
                done_n += 1
                el = time.time() - t0
                print(f"  {done_n}/{len(tasks)}  f#{i_f} rows[{lo},{hi})  "
                      f"elapsed={el/60:.1f}min  ETA={(len(tasks)-done_n)*el/max(done_n,1)/60:.1f}min",
                      flush=True)

    np.savez(OUT, f_Hz=F_NG15, k=ks, t=t_grid, s=s_grid, I2bar=I2,
             n_per_decade=N_PER_DECADE, n_seg_res=N_SEG_RES, n_max=N_MAX)
    print(f"saved {OUT}  ({np.isnan(I2).sum()} NaNs)  total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
