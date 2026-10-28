"""Sparse products: exact interaction features for a masked slot, learned by counting.

A dense random product (x_{t-1} Pa)*(x_{t+1} Pb) approximates the interaction between the left and right
neighbours. The exact version of that product, for one-hot tokens, is the indicator of the pair itself.
For each context pattern g (e.g. left+right neighbour, two-left+right), we store for every pattern value seen
>= MINC times the AVERAGE MEANING of the tokens that filled the slot (a closed-form conditional expectation).
At a new slot we look the pattern up: 'in contexts "the ___ of", the fillers mean ...'.

Training uses leave-one-out (the slot's own token is removed from its pattern's average), so the readout never
sees its own answer. A ridge readout learns how to weigh the patterns (+ reservoir memories for long range).
Storage: ~1M pattern values per group, 128 float16 each. Lookups are O(1) per token -> scales by counting.
"""
import numpy as np, numba as nb, gc
from ctx import Res, corpus_flags

GROUPS = {"L": (-1,), "R": (1,), "LR": (-1, 1), "LL": (-2, -1), "RR": (1, 2), "LLR": (-2, -1, 1), "LRR": (-1, 1, 2)}

@nb.njit(parallel=True, cache=True)
def _accum(inv, tok, E, S, nblk):
    d = E.shape[1]; n = len(inv); bs = (d + nblk - 1) // nblk
    for b in nb.prange(nblk):
        for i in range(n):
            k = inv[i]
            if k >= 0:
                for j in range(b * bs, min(d, b * bs + bs)): S[k, j] += E[tok[i], j]

def codes(tok, sid, pos, offs, V):
    n = len(tok); c = np.zeros(len(pos), np.int64); ok = np.ones(len(pos), np.bool_)
    for o in offs:
        q = np.clip(pos + o, 0, n - 1); c = c * V + tok[q].astype(np.int64); ok &= (sid[q] == sid[pos]) & (pos + o >= 0) & (pos + o < n)
    c[~ok] = -1; return c

class PatternTable:
    def __init__(self, ids, sid, E, offs, hi, minc=3, V=16000):
        self.offs = offs; self.V = V; pos = np.arange(2, hi - 2)
        c = codes(ids, sid, pos, offs, V); m = c >= 0; c = c[m]; p = pos[m]
        u, inv, cnt = np.unique(c, return_inverse=True, return_counts=True); del c
        keep = cnt >= minc; remap = np.full(len(u), -1, np.int64); remap[keep] = np.arange(keep.sum())
        inv = remap[inv]; self.keys = u[keep]; self.cnt = cnt[keep].astype(np.float32); del u, cnt, remap
        S = np.zeros((len(self.keys), E.shape[1]), np.float32); _accum(inv, ids[p], E, S, 2)
        self.mean = (S / self.cnt[:, None]).astype(np.float16); del S, inv, p; gc.collect()
    def lookup(self, tok, sid, pos, target=None, E=None):
        """mean filler meaning + log count for each slot; leave-one-out when target tokens are given."""
        c = codes(tok, sid, pos, self.offs, self.V); j = np.searchsorted(self.keys, c); j = np.clip(j, 0, len(self.keys) - 1)
        hit = (self.keys[j] == c) & (c >= 0); cnt = np.where(hit, self.cnt[j], 0.0).astype(np.float32)
        M = np.where(hit[:, None], self.mean[j].astype(np.float32), 0.0)
        if target is not None:
            loo = cnt - 1; good = loo > 0
            M = np.where(good[:, None], (M * cnt[:, None] - E[target]) / np.maximum(loo, 1)[:, None], 0.0); cnt = np.where(good, loo, 0.0)
        return M.astype(np.float32), np.log1p(cnt)[:, None].astype(np.float32)

class SparseCloze:
    """Masked-slot features = pattern lookups (sparse products) + bias-corrected reservoir memories."""
    def __init__(self, E, ids, dot, groups=("L", "R", "LR", "LL", "RR", "LLR", "LRR"), hi=None, minc=3, res=True):
        self.E = E; n = len(ids); hi = hi or int(0.9 * n); st, en, sid = corpus_flags(ids, dot)
        self.tables = {}
        for g in groups: self.tables[g] = PatternTable(ids, sid, E, GROUPS[g], hi, minc); gc.collect()
        self.res = res
        if res: self.fw = Res(E, 256, 3); self.bw = Res(E, 256, 4)
    def feats(self, tok, st, en, sid, pos, target=None):
        out = []
        for g, T in self.tables.items():
            M, lc = T.lookup(tok, sid, pos, target, self.E); out += [M, lc]
        if self.res: out += [self.fw.before(tok, st, pos), self.bw.after(tok, en, pos)]
        return np.concatenate(out, 1)
    def bank_feats(self, t, starts):
        n = len(t); st = np.zeros(n, np.bool_); st[starts[:-1]] = True; en = np.zeros(n, np.bool_); en[starts[1:] - 1] = True
        sid = np.repeat(np.arange(len(starts) - 1), np.diff(starts)); return self.feats(t.astype(np.int64), st, en, sid, np.arange(n))
