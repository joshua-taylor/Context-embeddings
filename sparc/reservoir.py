"""Fused diagonal memory: per direction, U fixed-decay units + U token-gated ('selective') units.

h_u <- a_u h_u + (1 - a_u) * drive_u(token)       a_u fixed (half fast .1-.7, half slow .8-.98)  or  a_u = A[token, u]
Each unit is independent (diagonal), so the recurrence is a parallel scan. States reset at every sentence and are
bias-corrected (divided by 1 - prod(a)), so short sentences are not under-weighted. The input is the CURRENT token's
meaning only (via a fixed random projection); exact local word order comes from the pattern tables instead.

mode 0 (cloze) : [fixed fwd BEFORE t, fixed bwd AFTER t, gated fwd BEFORE t, gated bwd AFTER t]   (t excluded)
mode 1 (fwd)   : [fixed fwd INCLUDING t, gated fwd INCLUDING t]
mode 2 (bwd)   : [fixed bwd INCLUDING t, gated bwd INCLUDING t]
"""
import numpy as np
import numba as nb


@nb.njit(parallel=True, cache=True, fastmath=True)
def _scan(tok, starts, T, a, Tg, A, mode, out):
    U = T.shape[1]; ns = len(starts) - 1
    for k in nb.prange(ns):
        s0 = starts[k]; s1 = starts[k + 1]
        for d_ in range(2):
            if (mode == 1 and d_ == 1) or (mode == 2 and d_ == 0): continue
            h = np.zeros(U, np.float32); c = np.zeros(U, np.float32); hg = np.zeros(U, np.float32); cg = np.zeros(U, np.float32)
            if mode == 0: c1 = d_ * U; c2 = 2 * U + d_ * U
            else: c1 = 0; c2 = U
            for step in range(s1 - s0):
                i = s0 + step if d_ == 0 else s1 - 1 - step
                x = tok[i]
                if mode == 0:
                    for u in range(U):
                        out[i, c1 + u] = h[u] / c[u] if c[u] > 0 else 0.0
                        out[i, c2 + u] = hg[u] / cg[u] if cg[u] > 0 else 0.0
                for u in range(U):
                    h[u] = a[u] * h[u] + (1 - a[u]) * T[x, u]; c[u] = a[u] * c[u] + (1 - a[u])
                    ag = A[x, u]; hg[u] = ag * hg[u] + (1 - ag) * Tg[x, u]; cg[u] = ag * cg[u] + (1 - ag)
                if mode != 0:
                    for u in range(U):
                        out[i, c1 + u] = h[u] / c[u]; out[i, c2 + u] = hg[u] / cg[u]


class Memory:
    def __init__(self, E, U=256, seed=3):
        r = np.random.default_rng(seed); d = E.shape[1]; self.U = U
        self.T = (E @ (r.standard_normal((d, U)) / np.sqrt(d))).astype(np.float32)
        self.a = np.r_[r.uniform(0.1, 0.7, U // 2), r.uniform(0.8, 0.98, U - U // 2)].astype(np.float32)
        g = r.standard_normal((d, U)); g /= np.linalg.norm(g, axis=0); z = E @ g
        z = (z - z.mean(0)) / z.std(0); z = 1.5 * z + r.normal(0, 0.5, U)
        lo = np.r_[np.full(U // 2, 0.05), np.full(U - U // 2, 0.6)]; hi = np.r_[np.full(U // 2, 0.9), np.full(U - U // 2, 0.995)]
        self.A = (lo + (hi - lo) / (1 + np.exp(-z))).astype(np.float32)
        self.Tg = (E @ (r.standard_normal((d, U)) / np.sqrt(d))).astype(np.float32)

    def width(self, mode):
        return 4 * self.U if mode == 0 else 2 * self.U

    def run(self, tok, starts, mode):
        out = np.empty((len(tok), self.width(mode)), np.float32)
        _scan(tok, starts, self.T, self.a, self.Tg, self.A, mode, out)
        return out
