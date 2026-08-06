"""Export Ibar2(t,s;k) tables to kernels_data/, on a Pzeta-independent
(t,s) grid, for a scan of external k. Run: python export_kernels.py
(takes ~10-15h on a modern multicore machine; parallel + memory-capped).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
import runtime  # thread/memory limits -- must precede jax

import numpy as np

import kernel as K
import kernel_table as KT
import spectra as sp

OUT = os.path.join(os.path.dirname(__file__), "..", "kernels_data")
os.makedirs(OUT, exist_ok=True)

N_K = 60
F_LO_NHZ, F_HI_NHZ = 1e-3, 1e4
T_MIN, T_MAX = 1e-3, 3000.0
N_PER_DECADE, N_SEG_RES = 50, 400
N_S = 18
N_MAX = 90000
N_WORKERS = 6
CHUNK_SIZE = 100   # small chunks: RSS grows a few MB/point in some (t,s) regions
CHUNK_MIN = 20     # floor for split-and-retry on a chunk MemoryError
RSS_HARD_GB = 1.4  # checked every 25 points within a chunk


def _worker_chunk(args):
    f_nHz, i_lo, i_hi, n_s = args
    import runtime as rt
    rt.cap_memory(2.5)
    import importlib
    qcd = importlib.import_module("background_qcd")
    k = float(sp.f_Hz_to_k(np.array([f_nHz * 1e-9]))[0])

    g1m, g2m = K.match_tensor(k, qcd)
    bgtab = KT.BackgroundTables(qcd.f1_Phi, qcd.f3_Phi, 1e-20, 2.0, n=8000)
    t_grid = KT.t_grid_universal(T_MIN, T_MAX, N_PER_DECADE, N_SEG_RES)
    s_grid = sp.s_grid(n_s)

    out = np.empty(i_hi - i_lo)
    for k_idx, idx in enumerate(range(i_lo, i_hi)):
        i, j = divmod(idx, n_s)
        out[k_idx] = KT.I2bar_table(k, float(t_grid[i]), float(s_grid[j]), qcd, g1m, g2m, bgtab, n_max=N_MAX)
        if (k_idx + 1) % 25 == 0 and rt.rss_gb() > RSS_HARD_GB:
            raise MemoryError(f"f={f_nHz:.4g}nHz chunk [{i_lo},{i_hi}): RSS>{RSS_HARD_GB}GB")
    return f_nHz, i_lo, i_hi, out


def main():
    f_nHz_arr = np.geomspace(F_LO_NHZ, F_HI_NHZ, N_K)
    t_grid = KT.t_grid_universal(T_MIN, T_MAX, N_PER_DECADE, N_SEG_RES)
    s_grid = sp.s_grid(N_S)
    n_t, n_s = len(t_grid), len(s_grid)
    n_per_node = n_t * n_s

    tasks0 = []
    for f_nHz in f_nHz_arr:
        n_chunks = (n_per_node + CHUNK_SIZE - 1) // CHUNK_SIZE
        for c in range(n_chunks):
            i_lo, i_hi = c * CHUNK_SIZE, min((c + 1) * CHUNK_SIZE, n_per_node)
            tasks0.append((float(f_nHz), i_lo, i_hi, N_S))
    print(f"exporting {N_K} external k, t-grid {n_t}, s-grid {n_s} "
          f"({n_per_node} pts/node), {len(tasks0)} chunks, {N_WORKERS} workers", flush=True)

    partial = {float(f): np.full(n_per_node, np.nan) for f in f_nHz_arr}
    done_count = {float(f): 0 for f in f_nHz_arr}

    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
    t0 = time.time()
    n_finished, n_submitted = 0, len(tasks0)

    with ProcessPoolExecutor(max_workers=N_WORKERS, max_tasks_per_child=1) as ex:
        pending = {ex.submit(_worker_chunk, task): task for task in tasks0}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                f_nHz, i_lo, i_hi, n_s_ = pending.pop(fut)
                try:
                    _, _, _, out = fut.result()
                except Exception as e:
                    size = i_hi - i_lo
                    if size <= CHUNK_MIN:
                        print(f"  !!! giving up on f={f_nHz:.4g}nHz [{i_lo}:{i_hi}]: {e}", flush=True)
                        partial[f_nHz][i_lo:i_hi] = -1.0  # marks permanently failed
                        done_count[f_nHz] += size
                        n_finished += 1
                    else:
                        mid = i_lo + size // 2
                        t1, t2 = (f_nHz, i_lo, mid, n_s_), (f_nHz, mid, i_hi, n_s_)
                        pending[ex.submit(_worker_chunk, t1)] = t1
                        pending[ex.submit(_worker_chunk, t2)] = t2
                        n_submitted += 1
                    continue

                partial[f_nHz][i_lo:i_hi] = out
                done_count[f_nHz] += (i_hi - i_lo)
                n_finished += 1
                elapsed = time.time() - t0
                eta_h = (n_submitted - n_finished) / (n_finished / elapsed) / 3600 if n_finished else float("nan")
                print(f"  chunk {n_finished}/~{n_submitted} (f={f_nHz:.4g}nHz) "
                      f"elapsed={elapsed/3600:.2f}h ETA~{eta_h:.2f}h", flush=True)

                if done_count[f_nHz] == n_per_node:
                    k = float(sp.f_Hz_to_k(np.array([f_nHz * 1e-9]))[0])
                    I2 = partial.pop(f_nHz).reshape(n_t, n_s)
                    fname = os.path.join(OUT, f"kernel_f{f_nHz:.6g}nHz.npz")
                    np.savez(fname, f_nHz=f_nHz, k=k, t=t_grid, s=s_grid, I2bar=I2)
                    print(f"  *** node f={f_nHz:.4g}nHz complete -> {os.path.basename(fname)}", flush=True)

    print(f"total: {time.time()-t0:.0f}s", flush=True)
    np.savez(os.path.join(OUT, "index.npz"), f_nHz=f_nHz_arr, t_min=T_MIN, t_max=T_MAX,
             n_per_decade=N_PER_DECADE, n_seg_res=N_SEG_RES, n_s=N_S, n_max=N_MAX)
    print("saved kernels_data/index.npz")


if __name__ == "__main__":
    main()
