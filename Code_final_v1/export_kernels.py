"""Export kernel tables on the lattice-QCD background to two folders:

  kernels_data/       I2bar(t,s;k)
  kernels_data_I1I2/  I1(t,s;k), I2(t,s;k), G11(k), G22(k), G12(k)

I1, I2 are the only expensive quantity computed per (t,s) point
(kernel_table.I1I2_table); I2bar is derived algebraically,

    I2bar(t,s) = I2(t,s)^2 G11 - 2 I1(t,s) I2(t,s) G12 + I1(t,s)^2 G22 ,

so both folders come from one pass. G11, G22, G12
(kernel_table.g_bilinear_averages) are k-only, computed once per node.

Chunked, memory-capped worker processes; checkpointed per external-k
node, safe to interrupt and resume.

Run:  python3 export_kernels.py             (fresh run)
      python3 export_kernels.py --resume    (skip completed nodes)
      python3 export_kernels.py --benchmark (estimate total runtime,
                                              write nothing)
"""
import sys, os, time, shutil, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
import runtime  # thread/memory limits -- must precede jax

import numpy as np

import kernel as K
import kernel_table as KT
import spectra as sp

HERE = os.path.dirname(__file__)
OUT_I2BAR = os.path.join(HERE, "kernels_data")
OUT_I1I2 = os.path.join(HERE, "kernels_data_I1I2")

N_K = 110
F_LO_NHZ, F_HI_NHZ = 1e-3, 1e4
T_MIN, T_MAX = 1e-3, 3000.0
N_PER_DECADE, N_SEG_RES = 65, 520
N_S = 24
N_MAX = 200000
PTS_PER_PERIOD = 45.0
N_WORKERS = 4  # tune to available cores/RAM
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

    out1 = np.empty(i_hi - i_lo)
    out2 = np.empty(i_hi - i_lo)
    for k_idx, idx in enumerate(range(i_lo, i_hi)):
        i, j = divmod(idx, n_s)
        I1, I2 = KT.I1I2_table(k, float(t_grid[i]), float(s_grid[j]), qcd, g1m, g2m, bgtab,
                               n_max=N_MAX, pts_per_period=PTS_PER_PERIOD)
        out1[k_idx] = I1
        out2[k_idx] = I2
        if (k_idx + 1) % 25 == 0 and rt.rss_gb() > RSS_HARD_GB:
            raise MemoryError(f"f={f_nHz:.4g}nHz chunk [{i_lo},{i_hi}): RSS>{RSS_HARD_GB}GB")
    return f_nHz, i_lo, i_hi, out1, out2


def _worker_gbilinear(f_nHz):
    """G11, G22, G12 for one external-k node."""
    import runtime as rt
    rt.cap_memory(2.5)
    import importlib
    qcd = importlib.import_module("background_qcd")
    k = float(sp.f_Hz_to_k(np.array([f_nHz * 1e-9]))[0])
    g1m, g2m = K.match_tensor(k, qcd)
    G11, G22, G12 = KT.g_bilinear_averages(k, qcd, g1m, g2m)
    return f_nHz, G11, G22, G12


def _node_filenames(f_nHz):
    fi2 = os.path.join(OUT_I2BAR, f"kernel_f{f_nHz:.6g}nHz.npz")
    fi12 = os.path.join(OUT_I1I2, f"kernel_I1I2_f{f_nHz:.6g}nHz.npz")
    return fi2, fi12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="skip nodes whose output files already exist in both folders")
    ap.add_argument("--benchmark", action="store_true",
                    help="time a handful of real chunks, print a runtime estimate, exit")
    args = ap.parse_args()

    f_nHz_arr = np.geomspace(F_LO_NHZ, F_HI_NHZ, N_K)
    t_grid = KT.t_grid_universal(T_MIN, T_MAX, N_PER_DECADE, N_SEG_RES)
    s_grid = sp.s_grid(N_S)
    n_t, n_s = len(t_grid), len(s_grid)
    n_per_node = n_t * n_s
    n_total = n_per_node * N_K

    if args.benchmark:
        from concurrent.futures import ProcessPoolExecutor
        print(f"benchmark: {N_K} external k, t-grid {n_t}, s-grid {n_s} "
              f"({n_per_node} pts/node, {n_total} pts total), {N_WORKERS} workers")
        sample_tasks = [(float(f_nHz_arr[i]), 0, CHUNK_SIZE, N_S)
                        for i in np.linspace(0, N_K - 1, min(N_WORKERS, N_K)).astype(int)]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=N_WORKERS, max_tasks_per_child=1) as ex:
            list(ex.map(_worker_chunk, sample_tasks))
        dt = time.time() - t0
        rate = (len(sample_tasks) * CHUNK_SIZE) / dt  # points/s, across N_WORKERS parallel
        eta_h = n_total / rate / 3600.0
        print(f"  {len(sample_tasks)} chunks x {CHUNK_SIZE} pts in {dt:.1f}s "
              f"-> {rate:.2f} pts/s (at {N_WORKERS} workers)")
        print(f"  ESTIMATED total runtime: {eta_h:.1f} h ({eta_h/24:.2f} days)")
        return

    print(f"exporting {N_K} external k, t-grid {n_t}, s-grid {n_s} "
          f"({n_per_node} pts/node, {n_total} pts total), {N_WORKERS} workers", flush=True)

    if args.resume:
        remaining = []
        for f_nHz in f_nHz_arr:
            fi2, fi12 = _node_filenames(float(f_nHz))
            if os.path.exists(fi2) and os.path.exists(fi12):
                continue
            remaining.append(f_nHz)
        f_nHz_arr = np.array(remaining)
        print(f"  --resume: {N_K - len(f_nHz_arr)} nodes already done, "
              f"{len(f_nHz_arr)} remaining", flush=True)
    else:
        for d in (OUT_I2BAR, OUT_I1I2):
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)
        print(f"  fresh run: wiped {OUT_I2BAR} and {OUT_I1I2}", flush=True)

    if len(f_nHz_arr) == 0:
        print("nothing to do")
        return

    tasks0 = []
    for f_nHz in f_nHz_arr:
        n_chunks = (n_per_node + CHUNK_SIZE - 1) // CHUNK_SIZE
        for c in range(n_chunks):
            i_lo, i_hi = c * CHUNK_SIZE, min((c + 1) * CHUNK_SIZE, n_per_node)
            tasks0.append((float(f_nHz), i_lo, i_hi, N_S))

    partial1 = {float(f): np.full(n_per_node, np.nan) for f in f_nHz_arr}
    partial2 = {float(f): np.full(n_per_node, np.nan) for f in f_nHz_arr}
    done_count = {float(f): 0 for f in f_nHz_arr}

    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
    t0 = time.time()
    n_finished, n_submitted = 0, len(tasks0)

    with ProcessPoolExecutor(max_workers=N_WORKERS, max_tasks_per_child=1) as ex:
        gbilinear = {}
        for f_nHz, G11, G22, G12 in ex.map(_worker_gbilinear, [float(f) for f in f_nHz_arr]):
            gbilinear[f_nHz] = (G11, G22, G12)
        print(f"  G11,G22,G12 done for all {len(f_nHz_arr)} nodes "
              f"({time.time()-t0:.0f}s)", flush=True)

        pending = {ex.submit(_worker_chunk, task): task for task in tasks0}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                f_nHz, i_lo, i_hi, n_s_ = pending.pop(fut)
                try:
                    _, _, _, out1, out2 = fut.result()
                except Exception as e:
                    size = i_hi - i_lo
                    if size <= CHUNK_MIN:
                        print(f"  !!! giving up on f={f_nHz:.4g}nHz [{i_lo}:{i_hi}]: {e}", flush=True)
                        partial1[f_nHz][i_lo:i_hi] = -1.0
                        partial2[f_nHz][i_lo:i_hi] = -1.0
                        done_count[f_nHz] += size
                        n_finished += 1
                    else:
                        mid = i_lo + size // 2
                        t1, t2 = (f_nHz, i_lo, mid, n_s_), (f_nHz, mid, i_hi, n_s_)
                        pending[ex.submit(_worker_chunk, t1)] = t1
                        pending[ex.submit(_worker_chunk, t2)] = t2
                        n_submitted += 1
                    continue

                partial1[f_nHz][i_lo:i_hi] = out1
                partial2[f_nHz][i_lo:i_hi] = out2
                done_count[f_nHz] += (i_hi - i_lo)
                n_finished += 1
                elapsed = time.time() - t0
                eta_h = (n_submitted - n_finished) / (n_finished / elapsed) / 3600 if n_finished else float("nan")
                print(f"  chunk {n_finished}/~{n_submitted} (f={f_nHz:.4g}nHz) "
                      f"elapsed={elapsed/3600:.2f}h ETA~{eta_h:.2f}h", flush=True)

                if done_count[f_nHz] == n_per_node:
                    k = float(sp.f_Hz_to_k(np.array([f_nHz * 1e-9]))[0])
                    I1 = partial1.pop(f_nHz).reshape(n_t, n_s)
                    I2 = partial2.pop(f_nHz).reshape(n_t, n_s)
                    G11, G22, G12 = gbilinear[f_nHz]

                    I2bar = I2**2 * G11 - 2.0 * I1 * I2 * G12 + I1**2 * G22

                    fi2, fi12 = _node_filenames(f_nHz)
                    np.savez(fi2, f_nHz=f_nHz, k=k, t=t_grid, s=s_grid, I2bar=I2bar)
                    np.savez(fi12, f_nHz=f_nHz, k=k, t=t_grid, s=s_grid, I1=I1, I2=I2,
                             G11=G11, G22=G22, G12=G12)
                    print(f"  *** node f={f_nHz:.4g}nHz complete -> "
                          f"{os.path.basename(fi2)}, {os.path.basename(fi12)}", flush=True)

    print(f"total: {time.time()-t0:.0f}s", flush=True)
    idx_kw = dict(f_nHz=f_nHz_arr, t_min=T_MIN, t_max=T_MAX, n_per_decade=N_PER_DECADE,
                  n_seg_res=N_SEG_RES, n_s=N_S, n_max=N_MAX, pts_per_period=PTS_PER_PERIOD)
    np.savez(os.path.join(OUT_I2BAR, "index.npz"), **idx_kw)
    np.savez(os.path.join(OUT_I1I2, "index.npz"), **idx_kw)
    print("saved index.npz in both output folders")


if __name__ == "__main__":
    main()
