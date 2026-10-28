"""Closed-form masked-token model with a FULL-RANK vocabulary readout.

Previous readouts predicted a 128-d meaning vector and compared it with every token's meaning (cosine). That
forces the model to output one 'average' meaning, which blurs multi-modal slots ("the ___ of" -> many options).
Here ridge regression predicts a score for every one of the 16k tokens directly (one-hot targets, linear
probability model): each token gets its own weight vector over the context features, exactly like the output
layer of BERT, still solved in closed form (X'X shared, X'Y is a scatter-add of features into token rows).

Contextual vector for a token = substitute vector: the meanings of the tokens the model would put in the slot,
weighted by their (positive) scores. (This is how substitute-based sense induction works with real BERT.)
"""
import numpy as np, gc
from ctx import corpus_flags
from hx import nz, tick
from mlm import MLM

class VocabRidge:
    def __init__(self, V): self.V = V; self.G = None
    def add(self, X, y, sub=20000):
        for c in range(0, len(X), sub):
            Xc = X[c:c + sub].astype(np.float32); yc = y[c:c + sub]
            if self.G is None:
                d = Xc.shape[1]; self.mu = Xc.mean(0); self.sd = Xc.std(0) + 1e-6
                self.G = np.zeros((d + 1, d + 1)); self.XYt = np.zeros((self.V, d + 1)); self.n = 0
            Xs = np.c_[(Xc - self.mu) / self.sd, np.ones(len(Xc), np.float32)].astype(np.float32)
            self.G += (Xs.T @ Xs).astype(np.float64); o = np.argsort(yc, kind="stable"); ys = yc[o]
            u, st = np.unique(ys, return_index=True); self.XYt[u] += np.add.reduceat(Xs[o], st, axis=0); self.n += len(Xc)
    def std(self, X): return np.c_[(X - self.mu) / self.sd, np.ones(len(X), np.float32)].astype(np.float32)
    def solve(self, Xv, yv, lams=(1e-3, 1e-2, 1e-1)):
        d = len(self.G); Xs = self.std(Xv); best = None
        for lam in lams:
            W = np.linalg.solve(self.G + lam * np.trace(self.G) / d * np.eye(d), self.XYt.T).astype(np.float32)
            r = self.ranks(Xs, yv, W); mrr = float(np.mean(1 / (r + 1)))
            if best is None or mrr > best[0]: best = (mrr, W, lam, r)
        self.mrr, self.W, self.lam, r = best
        return dict(mrr=self.mrr, top1=float(np.mean(r == 0)), top10=float(np.mean(r < 10)), lam=self.lam)
    @staticmethod
    def ranks(Xs, y, W, ch=2000):
        out = []
        for c in range(0, len(Xs), ch):
            S = Xs[c:c + ch] @ W; out.append((S > S[np.arange(len(S)), y[c:c + ch]][:, None]).sum(1))
        return np.concatenate(out)
    def scores(self, X): return self.std(X) @ self.W

class MLM2(MLM):
    def fit(self, ids, dot, freq, ntr=500_000, seed=0, split=None, chunk=50_000):
        n = len(ids); split = split or int(0.9 * n); st, en, sid = corpus_flags(ids, dot); r = np.random.default_rng(seed)
        keep_p = np.minimum(1.0, np.sqrt(1e-4 / (freq + 1e-12)) + 1e-4 / (freq + 1e-12))
        def sample(lo, hi, m):
            c = r.integers(lo, hi, int(m * 4)); c = c[r.random(len(c)) < keep_p[ids[c]]]; return np.sort(np.unique(c)[:m])
        ptr = sample(2, split - 2, ntr); pva = sample(split + 2, n - 2, 10000)
        self.vr = VocabRidge(len(self.E))
        for c in range(0, len(ptr), chunk):
            p = ptr[c:c + chunk]; self.vr.add(self.feats(ids, st, en, sid, p), ids[p]); gc.collect()
        self.val = self.vr.solve(self.feats(ids, st, en, sid, pva), ids[pva]); return self.val
    def substitute(self, t, starts, k=50, ch=4000):
        F = self.bank_feats(t, starts); out = np.empty((len(t), self.E.shape[1]), np.float32)
        for c in range(0, len(t), ch):
            S = self.vr.scores(F[c:c + ch]); idx = np.argpartition(-S, k, axis=1)[:, :k]; w = np.maximum(np.take_along_axis(S, idx, 1), 0)
            out[c:c + ch] = np.einsum("nk,nkd->nd", w, self.E[idx])
        return out

class Hidden:
    """BERT-like hidden state for a slot: the context features projected onto the readout's top-r directions
    (the part of feature space the model actually uses to predict tokens). Also the model's own estimate of how
    predictable the actual token was (score of the true token), used for contextual-SIF pooling."""
    def __init__(self, m, r=256):
        W = m.vr.W[:-1]                                   # (d, V) on standardised features
        U, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)
        self.P = (U[:, :r] * S[:r]).astype(np.float32); self.m = m; self.Wt = m.vr.W   # keep full W for true-token scores
    def __call__(self, t, starts, with_score=False):
        F = self.m.vr.std(self.m.bank_feats(t, starts)); Z = F[:, :-1] @ self.P
        if not with_score: return Z
        sc = np.einsum("nd,dn->n", F, self.Wt[:, t]) if len(t) < 50000 else np.concatenate([np.einsum("nd,dn->n", F[c:c+20000], self.Wt[:, t[c:c+20000]]) for c in range(0, len(t), 20000)])
        return Z, sc
