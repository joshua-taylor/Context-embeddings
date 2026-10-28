"""Counting attention: relevance scores are corpus PMI statistics; the query is the current layer's guess.

score(key s, target) = PMI(token_s before target) = KA[token_s] . QB      (key left of the target)
                     = PMI(target before token_s) = QA . KB[token_s]      (key right of the target)
The target's identity is unknown (it is what we are guessing), so its PMI vectors QA, QB are predicted from the
layer's meaning guess by a fixed frequency-weighted ridge map (meanings -> PMI factor space).
Softmax with temperature tau over the allowed keys plus a 'nothing relevant' option (score 0).
Allowed keys: cloze = every other token in the sentence; fwd = tokens <= t (target t+1 is after them all);
bwd = tokens >= t. Output = attention-weighted average of the attended tokens' meanings.
"""
import numpy as np
import numba as nb
from .utils import nz


@nb.njit(parallel=True, cache=True, fastmath=True)
def _attend(tok, qpos, qs0, qs1, QA, QB, KA, KB, V, mode, tau, out):
    nq = len(qpos); k = KA.shape[1]; d = V.shape[1]
    for qi in nb.prange(nq):
        i = qpos[qi]; s0 = qs0[qi]; s1 = qs1[qi]; L = s1 - s0
        sc = np.empty(L + 1, np.float32); mx = 0.0; sc[L] = 0.0
        for s in range(s0, s1):
            ok = (s != i) if mode == 0 else ((s <= i) if mode == 1 else (s >= i))
            if not ok:
                sc[s - s0] = -1e30; continue
            b = tok[s]; v = 0.0
            if (mode == 0 and s < i) or mode == 1:
                for j in range(k): v += KA[b, j] * QB[qi, j]
            else:
                for j in range(k): v += QA[qi, j] * KB[b, j]
            v /= tau; sc[s - s0] = v
            if v > mx: mx = v
        z = 0.0
        for s in range(L + 1):
            sc[s] = np.exp(sc[s] - mx) if sc[s] > -1e29 else 0.0; z += sc[s]
        for j in range(d): out[qi, j] = 0.0
        for s in range(s0, s1):
            w = sc[s - s0] / z
            if w > 0:
                for j in range(d): out[qi, j] += w * V[tok[s], j]


class CountingAttention:
    def __init__(self, KA, KB, E, freq, tau=0.2):
        self.KA = np.ascontiguousarray(KA, np.float32); self.KB = np.ascontiguousarray(KB, np.float32); self.E = E; self.tau = np.float32(tau)
        w = np.sqrt(freq + 1e-9)[:, None]; Ew = E * w; G = Ew.T @ Ew + 1e-3 * np.eye(E.shape[1])
        self.MA = np.linalg.solve(G, Ew.T @ (self.KA * w)).astype(np.float32)
        self.MB = np.linalg.solve(G, Ew.T @ (self.KB * w)).astype(np.float32)

    def __call__(self, tok, starts, q, guess, mode):
        sid = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
        out = np.empty((len(q), self.E.shape[1]), np.float32)
        _attend(tok, q.astype(np.int64), starts[sid[q]].astype(np.int64), starts[sid[q] + 1].astype(np.int64),
                (guess @ self.MA).astype(np.float32), (guess @ self.MB).astype(np.float32), self.KA, self.KB, self.E, mode, self.tau, out)
        return nz(out)
