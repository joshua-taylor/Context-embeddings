"""Nonlinear components.

Selective reservoir : h_u <- a_u(token) * h_u + (1 - a_u(token)) * drive_u(token).
                      a_u(token) = lo_u + (hi_u - lo_u) * sigmoid(gamma * E[token] . g_u + b_u), a fixed random table
                      (vocab x units). Tokens decide what each unit keeps or forgets (the Mamba / GLA idea, but random
                      and fixed). Same cost as the linear reservoir: one table lookup + a diagonal scan.
TensorSketch        : compact sketch of the outer product u (x) v (Pham & Pagh 2013): count-sketch both, multiply in
                      the Fourier domain. Inner products of sketches approximate (u.u')(v.v'), so a linear readout on
                      them can use products like "not" x "hot". Cost O(d + D log D) per token.
RandomTanh          : tanh(z W + b), a fixed random nonlinear expansion (approximates a smooth kernel).
"""
import numpy as np, numba as nb
from scipy import sparse

@nb.njit(parallel=True, cache=True, fastmath=True)
def scan_gated(D, A, h, nblk):
    m, U = D.shape; H = np.empty((m, U), np.float32); bs = (U + nblk - 1) // nblk
    for b in nb.prange(nblk):
        lo = b * bs; hi = min(U, lo + bs)
        for t in range(m):
            for u in range(lo, hi):
                a = A[t, u]; h[u] = a * h[u] + (1.0 - a) * D[t, u]; H[t, u] = h[u]
    return H

class Selective:
    def __init__(self, E, U=256, seed=11, gamma=1.5):
        r = np.random.default_rng(seed); d = E.shape[1]
        self.T = (E @ (r.standard_normal((d, U)) / np.sqrt(d))).astype(np.float32)
        g = r.standard_normal((d, U)); g /= np.linalg.norm(g, axis=0)
        lo = np.r_[np.full(U // 2, 0.05), np.full(U - U // 2, 0.6)]; hi = np.r_[np.full(U // 2, 0.9), np.full(U - U // 2, 0.995)]
        z = E @ g; z = (z - z.mean(0)) / z.std(0); z = gamma * z + r.normal(0, 0.5, U)      # each unit's gate varies strongly across tokens
        self.A = (lo + (hi - lo) / (1 + np.exp(-z))).astype(np.float32); self.U = U
    def init(self): return np.zeros(self.U, np.float32)
    def run(self, tok, h): return scan_gated(self.T[tok], self.A[tok], h, 2)

class TensorSketch:
    def __init__(self, d1, d2, D, seed):
        r = np.random.default_rng(seed); self.D = D
        mk = lambda d: sparse.csr_matrix((r.choice([-1.0, 1.0], d), (np.arange(d), r.integers(0, D, d))), shape=(d, D), dtype=np.float32)
        self.C1, self.C2 = mk(d1), mk(d2)
    def __call__(self, U, V, chunk=20_000):
        out = np.empty((len(U), self.D), np.float32)
        for c in range(0, len(U), chunk):
            a = np.asarray((self.C1.T @ U[c:c + chunk].T).T); b = np.asarray((self.C2.T @ V[c:c + chunk].T).T)
            out[c:c + chunk] = np.fft.irfft(np.fft.rfft(a, axis=1) * np.fft.rfft(b, axis=1), n=self.D, axis=1)
        return out

class RandomTanh:
    def __init__(self, d, D, seed, gain=1.5):
        r = np.random.default_rng(seed); self.W = (r.standard_normal((d, D)) / np.sqrt(d) * gain).astype(np.float32); self.b = r.uniform(-1, 1, D).astype(np.float32)
    def __call__(self, Z): return np.tanh(Z @ self.W + self.b)

class Standardise:
    def __init__(self, X): self.m = X.mean(0); self.s = X.std(0) + 1e-6
    def __call__(self, X): return (X - self.m) / self.s

class RandomBilinear:
    """phi = (U Wa) * (V Wb): random multiplicative interactions between two inputs (e.g. current word x context).
    Inner products approximate (u.u')(v.v'); two matrix multiplies + an elementwise product (BLAS/GPU friendly)."""
    def __init__(self, d1, d2, D, seed):
        r = np.random.default_rng(seed); self.Wa = (r.standard_normal((d1, D)) / np.sqrt(d1)).astype(np.float32); self.Wb = (r.standard_normal((d2, D)) / np.sqrt(d2)).astype(np.float32)
    def __call__(self, U, V): return (U @ self.Wa) * (V @ self.Wb)
