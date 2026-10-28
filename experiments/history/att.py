"""Closed-form attention: query/key tables from corpus PMI, values = token meanings.

score(t attends to s) = PMI(s precedes t) = A[s].B[t]   if s < t
                      = PMI(t precedes s) = A[t].B[s]   if s > t
self score = `sink` (a learned-free 'stay as myself' option); optional distance penalty -lam*log|t-s|.
alpha = softmax(score / tau);  output_t = sum_s alpha_ts V[s].
"""
import numpy as np, numba as nb

@nb.njit(parallel=True, cache=True, fastmath=True)
def pmi_attend(tok, starts, QA, QB, Vv, tau, sink, lam, kb, out, wself):
    ns = len(starts) - 1; d = Vv.shape[1]; k = QA.shape[1]
    for i in nb.prange(ns):
        s0 = starts[i]; s1 = starts[i + 1]; L = s1 - s0
        sc = np.empty(L, np.float32)
        for t in range(L):
            a = tok[s0 + t]; mx = -1e30
            for s in range(L):
                if s == t: v = sink
                else:
                    b = tok[s0 + s]; v = 0.0
                    if s < t:
                        for j in range(k): v += QA[b, j] * QB[a, j]
                    else:
                        for j in range(k): v += QA[a, j] * QB[b, j]
                    if lam > 0: v -= lam * np.log(abs(t - s))
                    v += kb[b]
                v /= tau; sc[s] = v
                if v > mx: mx = v
            z = 0.0
            for s in range(L): sc[s] = np.exp(sc[s] - mx); z += sc[s]
            for j in range(d): out[s0 + t, j] = 0.0
            for s in range(L):
                w = sc[s] / z; b = tok[s0 + s]
                for j in range(d): out[s0 + t, j] += w * Vv[b, j]
            wself[s0 + t] = sc[t] / z

class PMIAttn:
    def __init__(self, A, B, Vv, rank=256, tau=1.0, sink=0.0, lam=0.0, kb=None):
        self.kb = np.zeros(len(A), np.float32) if kb is None else kb.astype(np.float32)
        self.A = np.ascontiguousarray(A[:, :rank]); self.B = np.ascontiguousarray(B[:, :rank]); self.V = Vv.astype(np.float32)
        self.tau, self.sink, self.lam = np.float32(tau), np.float32(sink), np.float32(lam)
    def __call__(self, t, st):
        out = np.empty((len(t), self.V.shape[1]), np.float32); ws = np.empty(len(t), np.float32)
        pmi_attend(t.astype(np.int64), st.astype(np.int64), self.A, self.B, self.V, self.tau, self.sink, self.lam, self.kb, out, ws)
        self.wself = ws; return out

@nb.njit(parallel=True, cache=True, fastmath=True)
def query_attend(tok, starts, QAq, QBq, KA, KB, Vv, tau, sink, out):
    """Attention for a HIDDEN slot: the slot's identity is unknown, so its PMI query vectors (QAq, QBq: one row per
    position) come from the first-stage guess. score(i, s) = KA[s].QBq[i] (s before i) or QAq[i].KB[s] (s after i)."""
    ns = len(starts) - 1; d = Vv.shape[1]; k = KA.shape[1]
    for si in nb.prange(ns):
        s0 = starts[si]; s1 = starts[si + 1]; L = s1 - s0; sc = np.empty(L + 1, np.float32)
        for i in range(s0, s1):
            mx = sink / tau; sc[L] = mx
            for s in range(s0, s1):
                if s == i: sc[s - s0] = -1e30; continue
                b = tok[s]; v = 0.0
                if s < i:
                    for j in range(k): v += KA[b, j] * QBq[i, j]
                else:
                    for j in range(k): v += QAq[i, j] * KB[b, j]
                v /= tau; sc[s - s0] = v
                if v > mx: mx = v
            z = 0.0
            for s in range(L + 1): sc[s] = np.exp(sc[s] - mx) if sc[s] > -1e29 else 0.0; z += sc[s]
            for j in range(d): out[i, j] = 0.0
            for s in range(s0, s1):
                w = sc[s - s0] / z
                if w > 0:
                    for j in range(d): out[i, j] += w * Vv[tok[s], j]
