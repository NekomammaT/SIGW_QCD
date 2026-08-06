"""Load kernels_data/ and provide Ibar2(t,s;k) at any external k via
log-log interpolation across the k-node axis (Ibar2 varies smoothly with
k away from the resonance, which each node's (t,s) grid already resolves)."""
import os
import glob
import numpy as np
from scipy.integrate import trapezoid

import kernel as K

KDATA = os.path.join(os.path.dirname(__file__), "..", "kernels_data")


class KernelTable:
    def __init__(self, kdata=KDATA):
        files = sorted(glob.glob(os.path.join(kdata, "kernel_f*.npz")))
        tabs = sorted([np.load(fn) for fn in files], key=lambda t: float(t["k"]))
        self.k_nodes = np.array([float(t["k"]) for t in tabs])
        self.f_nHz_nodes = np.array([float(t["f_nHz"]) for t in tabs])
        self.t, self.s = tabs[0]["t"], tabs[0]["s"]
        for t in tabs:
            assert np.allclose(t["t"], self.t) and np.allclose(t["s"], self.s), \
                "all tables must share the same (t,s) grid"
        self.log_I2 = np.stack([np.log(np.clip(t["I2bar"], 1e-300, None)) for t in tabs], axis=0)
        self.log_k_nodes = np.log(self.k_nodes)
        self.k_min, self.k_max = self.k_nodes.min(), self.k_nodes.max()

    def I2bar(self, k):
        """log-log interpolation of Ibar2(t,s) at external k (clipped to the tabulated range)."""
        lk = np.log(np.clip(k, self.k_min, self.k_max))
        flat = self.log_I2.reshape(len(self.k_nodes), -1)
        interp_flat = np.apply_along_axis(lambda col: np.interp(lk, self.log_k_nodes, col), 0, flat)
        return np.exp(interp_flat.reshape(self.log_I2[0].shape))

    def Ph(self, k, pzeta):
        T, S = np.meshgrid(self.t, self.s, indexing="ij")
        I2 = self.I2bar(k)
        W = K.weight(T, S) * pzeta(k * K.u_of(T, S)) * pzeta(k * K.v_of(T, S))
        inner = trapezoid(W * I2, x=self.s, axis=1)
        return 4.0 * trapezoid(inner, x=self.t)
