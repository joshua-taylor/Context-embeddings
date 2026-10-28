"""Closed-form masked-token model (the BERT objective, solved by ridge regression).

For a slot t (the token itself hidden) build features from both sides:
  window   : x_{t-2}, x_{t-1}, x_{t+1}, x_{t+2}                       (exact neighbours, zero outside the sentence)
  memory   : forward reservoir state before t, backward state after t (sentence-reset, bias-corrected)
  products : (x_{t-1} Pa)*(x_{t+1} Pb)    the 'sandwich' interaction (e.g. "the ___ of")
             (h_fwd Qa)*(h_bwd Qb)        left-context x right-context interaction
Target: the hidden token's meaning E[tok_t]. Frequent tokens are subsampled when picking training slots.
The prediction ('what fits here') is a substitute-style contextual vector, like BERT's output at a masked slot.
"""
import numpy as np
from ctx import Res, corpus_flags
from nl import RandomBilinear
from hx import RidgeFit, nz, tick

class MLM:
    def __init__(self, E, parts=("win", "res", "prod"), dp=1024, U=256, seed=0):
        self.E = E; d = E.shape[1]; self.parts = parts
        self.fw = Res(E, U, seed + 3); self.bw = Res(E, U, seed + 4)
        self.p1 = RandomBilinear(d, d, dp, seed + 5); self.p2 = RandomBilinear(U, U, dp, seed + 6)
    def feats(self, tok, st, en, sid, pos):
        E = self.E; n = len(tok)
        def nb(o):
            q = np.clip(pos + o, 0, n - 1); ok = sid[q] == sid[pos]; return E[tok[q]] * ok[:, None]
        out = []
        xm1, xp1 = nb(-1), nb(1)
        if "win" in self.parts: out += [nb(-2), xm1, xp1, nb(2)]
        hf = self.fw.before(tok, st, pos); hb = self.bw.after(tok, en, pos)
        if "res" in self.parts: out += [hf, hb]
        if "prod" in self.parts: out += [self.p1(xm1, xp1), self.p2(hf, hb)]
        return np.concatenate(out, 1).astype(np.float32)
    def fit(self, ids, dot, freq, ntr=600_000, seed=0, split=None):
        n = len(ids); split = split or int(0.9 * n); st, en, sid = corpus_flags(ids, dot); r = np.random.default_rng(seed)
        keep_p = np.minimum(1.0, np.sqrt(1e-4 / (freq + 1e-12)) + 1e-4 / (freq + 1e-12))
        def sample(lo, hi, m):
            c = r.integers(lo, hi, int(m * 4)); c = c[r.random(len(c)) < keep_p[ids[c]]]; return np.sort(np.unique(c)[:m])
        ptr = sample(2, split - 2, ntr); pva = sample(split + 2, n - 2, 20000)
        self.rf = RidgeFit(); ch = 100_000
        for c in range(0, len(ptr), ch):
            p = ptr[c:c + ch]; self.rf.add(self.feats(ids, st, en, sid, p), self.E[ids[p]])
        Fv = self.feats(ids, st, en, sid, pva); cos = self.rf.solve(Fv, self.E[ids[pva]])
        G = nz(self.rf(Fv)) @ self.E.T; rank = (G > G[np.arange(len(pva)), ids[pva]][:, None]).sum(1)
        self.val = dict(cos=cos, mrr=float(np.mean(1 / (rank + 1))), top1=float(np.mean(rank == 0)), top10=float(np.mean(rank < 10)))
        return self.val
    def bank_feats(self, t, starts):
        n = len(t); st = np.zeros(n, np.bool_); st[starts[:-1]] = True; en = np.zeros(n, np.bool_); en[starts[1:] - 1] = True
        sid = np.repeat(np.arange(len(starts) - 1), np.diff(starts)); return self.feats(t, st, en, sid, np.arange(n))
    def guess(self, t, starts): return self.rf(self.bank_feats(t, starts))
