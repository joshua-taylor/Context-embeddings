"""Closed-form deep gap-filler.

Layer 1 input  : BASE = [5 pattern groups (sparse products) ; linear + selective reservoir memories]
Layer k output : two ridge readouts on the layer's features -> meaning guess g_k (128) and full vocabulary scores;
                 hidden state h_k = features projected on the top-r directions of the vocabulary readout
Layer k+1 input: [BASE ; layer-k output (g or h) ; counting attention queried by g_k ;
                  NONLINEAR expansion = RandomBilinear(layer-k output, [attention ; random view of the memories])]
Everything at a gap t is computed without token t (leave-one-out pattern statistics in training)."""
import numpy as np, pickle, gc
from scipy.linalg import cho_factor, cho_solve
from hx import nz, V, Bank, tick, sentence_bounds, load_corpus, token_freq, BIG
from zoo import Zoo2
from att import query_attend
from nl import RandomBilinear

class Deep:
    def __init__(self, E, sc, statsfile="roundQ_stats.npz", res=("LIN", "SEL"), freq=None, r_hidden=128, dprod=2048):
        self.E = E; self.sc = sc; self.z = Zoo2(E); self.res = list(res); d = E.shape[1]
        Z = np.load(statsfile); names = list(Z["names"]); dims = Z["dims"]; off = np.cumsum(np.r_[0, dims])
        idx = {k: np.arange(off[i], off[i + 1]) for i, k in enumerate(names)}
        self.cols = np.r_[idx["P"], np.concatenate([idx[k] for k in self.res])]; self.mu0, self.sd0 = Z["mu"][self.cols], Z["sd"][self.cols]; self.np_ = len(idx["P"])
        Zp = np.load(f"{BIG}/pmi_W8.npz"); self.KA, self.KB = np.ascontiguousarray(Zp["A"]), np.ascontiguousarray(Zp["B"]); w = np.sqrt(freq + 1e-9)[:, None]
        self.MA = np.linalg.solve((E * w).T @ (E * w) + 1e-3 * np.eye(d), (E * w).T @ (self.KA * w)).astype(np.float32)
        self.MB = np.linalg.solve((E * w).T @ (E * w) + 1e-3 * np.eye(d), (E * w).T @ (self.KB * w)).astype(np.float32)
        nres = len(self.cols) - self.np_; self.R = (np.random.default_rng(42).standard_normal((nres, 128)) / np.sqrt(nres)).astype(np.float32)
        self.rh = r_hidden; self.dprod = dprod; self.layers = []
    def base(self, t, st, loo):
        sid = np.repeat(np.arange(len(st) - 1), np.diff(st)); allp = np.arange(len(t)); Pm = []
        for g, T in self.sc.tables.items(): M, lc = T.lookup(t, sid, allp, t if loo else None, self.E); Pm += [M, lc]
        X = np.concatenate([np.concatenate(Pm, 1)] + [self.z.run(k, t, st) for k in self.res], 1)
        return ((X - self.mu0) / self.sd0).astype(np.float32)
    def attend(self, t, st, g):
        o = np.empty((len(t), self.E.shape[1]), np.float32)
        query_attend(t, st, (g @ self.MA).astype(np.float32), (g @ self.MB).astype(np.float32), self.KA, self.KB, self.E, np.float32(0.2), np.float32(0.0), o); return nz(o)
    def layer_inputs(self, k, t, st, B, outs):
        """features of layer k (1-based) given BASE and outputs of earlier layers"""
        if k == 1: return B
        L = self.layers[k - 2]; o = outs[k - 2]; carry = o["g"] if L["carry"] == "g" else o["h"]
        A = self.attend(t, st, o["g"]); ctx = np.c_[A, nz(B[:, self.np_:] @ self.R)]
        prev = o["X"] if L.get("residual") else B                       # residual: keep everything the previous layer saw
        return np.c_[prev, carry, A, L["xp"](carry, ctx)] if L["prod"] else np.c_[prev, carry, A]
    def outputs(self, k, X):
        L = self.layers[k - 1]; Xq = (X - L["mu"]) / L["sd"]
        g = nz(Xq @ L["Wm"][:-1] + L["Wm"][-1]); h = nz(Xq @ L["P"]); return dict(g=g, h=h, Xq=Xq, X=X)
    def forward(self, t, st, loo, upto):
        B = self.base(t, st, loo); outs = []; X = None
        for k in range(1, upto + 1):
            X = self.layer_inputs(k, t, st, B, outs)
            if k <= len(self.layers): outs.append(self.outputs(k, X))
        return B, X, outs
    def add_layer(self, bank, keep_p, carry="g", prod=True, ntr=200_000, seed=1, lam_m=1e-3, lam_v=1e-4, residual=False, chunk=400):
        """fit layer k = len(layers)+1 on training gaps (closed form)"""
        k = len(self.layers) + 1; rng = np.random.default_rng(seed); G = None; n = 0
        if k > 1:
            cd = self.E.shape[1] if carry == "g" else self.rh
            self.layers[-1]["carry"] = carry; self.layers[-1]["prod"] = prod; self.layers[-1]["residual"] = residual; self.layers[-1]["xp"] = RandomBilinear(cd, 256, self.dprod, 50 + k)
            self.layers[-1]["Wv"] = None; gc.collect()                       # only the newest layer keeps its 16k-token readout
        for c in range(0, bank.n, chunk):
            s0, s1 = c, min(bank.n, c + chunk); lo, hi_ = bank.starts[s0], bank.starts[s1]
            t = bank.tok[lo:hi_].astype(np.int64); st = (bank.starts[s0:s1 + 1] - lo).astype(np.int64)
            pos = np.flatnonzero(rng.random(len(t)) < keep_p[t]); _, X, _ = self.forward(t, st, True, k); X = X[pos]; y = t[pos]
            if G is None: D = X.shape[1]; mu = X.mean(0); sd = X.std(0) + 1e-6; G = np.zeros((D + 1, D + 1)); XYm = np.zeros((D + 1, self.E.shape[1])); XYv = np.zeros((V, D + 1), np.float32)
            Xq = np.c_[(X - mu) / sd, np.ones(len(X), np.float32)].astype(np.float32); del X
            G += (Xq.T @ Xq).astype(np.float64); XYm += (Xq.T @ self.E[y]).astype(np.float64)
            o = np.argsort(y, kind="stable"); u, si = np.unique(y[o], return_index=True); XYv[u] += np.add.reduceat(Xq[o], si, 0); del Xq; n += len(y)
            if n >= ntr: break
        tr = np.trace(G) / len(G)
        Wm = cho_solve(cho_factor(G + lam_m * tr * np.eye(len(G))), XYm).astype(np.float32)
        cf = cho_factor(G + lam_v * tr * np.eye(len(G))); del G; Wv = np.empty((XYv.shape[1], V), np.float32)
        for v0 in range(0, V, 2000): Wv[:, v0:v0 + 2000] = cho_solve(cf, XYv[v0:v0 + 2000].T.astype(np.float64)).astype(np.float32)   # vocabulary slices: low memory
        del XYv, cf; gc.collect()
        from sklearn.utils.extmath import randomized_svd
        U, S, _ = randomized_svd(Wv[:-1], self.rh, n_iter=4, random_state=0); P = (U * S).astype(np.float32)
        self.layers.append(dict(mu=mu, sd=sd, Wm=Wm, Wv=Wv, P=P, carry=None, prod=None, xp=None)); return n
    def evaluate(self, bank, keep_p, seed=2):
        k = len(self.layers); rng = np.random.default_rng(seed); vr, mr = [], []
        for c in range(0, bank.n, 500):
            s0, s1 = c, min(bank.n, c + 500); lo, hi_ = bank.starts[s0], bank.starts[s1]
            t = bank.tok[lo:hi_].astype(np.int64); st = (bank.starts[s0:s1 + 1] - lo).astype(np.int64)
            pos = np.flatnonzero(rng.random(len(t)) < keep_p[t]); _, X, outs = self.forward(t, st, False, k); o = outs[-1]; y = t[pos]
            Xq = o["Xq"][pos]; S = Xq @ self.layers[-1]["Wv"][:-1] + self.layers[-1]["Wv"][-1]; vr.append((S > S[np.arange(len(S)), y][:, None]).sum(1))
            S = o["g"][pos] @ self.E.T; mr.append((S > S[np.arange(len(S)), y][:, None]).sum(1))
        vr = np.concatenate(vr); mr = np.concatenate(mr)
        return dict(vocab_mrr=float(np.mean(1 / (vr + 1))), top1=float(np.mean(vr == 0)), meaning_mrr=float(np.mean(1 / (mr + 1))))
    def token_features(self, t, st):
        """per-token outputs of every layer (for probes / sentence vectors); test-time (no leave-one-out)"""
        B, X, outs = self.forward(t, st, False, len(self.layers)); return B, outs
