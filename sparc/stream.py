"""One SPARC stream: a stack of closed-form layers predicting a token from a restricted view of the sentence.

  cloze : target = token t,   sees both sides, never t          ('what fits here')
  fwd   : target = token t+1, sees tokens <= t                  (word t read in its left context)
  bwd   : target = token t-1, sees tokens >= t                  (word t read in its right context)

Layer 1 features at a query position: pattern-table lookups for the target gap + memory states.
Layer k+1 features = [everything layer k saw (residual stream) ; layer-k meaning guess g ;
                      counting attention queried by g ; products(g, [attention ; random view of the memory])].
Each layer: ridge readouts to the target's meaning (guess g) and to all V tokens (scores; the top `rh` directions of
that readout define the layer's hidden state h). Standardisation is folded into the weights.

Training engineering: the same training rows are used for every layer and each layer's feature matrix is cached on
disk (float16), so layer k+1 never recomputes lookups, memories or earlier layers; X'X statistics are incremental
(layer k's X'X is the top-left block of layer k+1's) and centring removes the bias row; the V-token readout is solved
in vocabulary slices. Inference: every feature block is multiplied once by the stacked readout rows of all the layers
that read it, so the residual stream is never concatenated.
"""
import os, gc, time, shutil
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.utils.extmath import randomized_svd
from .utils import nz
from .products import make_products

GROUPS = {"cloze": ("L", "R", "LR", "LL", "RR"), "fwd": ("L", "LL"), "bwd": ("R", "RR")}
MODE_ID = {"cloze": 0, "fwd": 1, "bwd": 2}
SHIFT = {"cloze": 0, "fwd": 1, "bwd": -1}


class Stream:
    def __init__(self, mode, shared, n_layers=3, width=2048, products="random", rh=128, seed=0):
        self.mode = mode; self.n_layers = n_layers; self.width = width; self.kind = products; self.rh = rh; self.seed = seed
        self.layers = []; self.attach(shared)
        d = self.E.shape[1]; mw = self.mem.width(MODE_ID[mode])
        self.np_ = len(GROUPS[mode]) * (d + 1); self.nb = self.np_ + mw
        self.R = (np.random.default_rng(42 + seed).standard_normal((mw, 128)) / np.sqrt(mw)).astype(np.float32)

    # shared components are not pickled with the stream
    def attach(self, shared):
        self.E, self.pat, self.mem, self.att, self.V = shared["E"], shared["patterns"], shared["memory"], shared["attention"], shared["V"]

    def detach(self):
        self.E = self.pat = self.mem = self.att = None

    # ------------------------------------------------------------------ features
    def targets(self, n, starts):
        """index of the token this stream predicts at each position (-1 if outside the sentence)."""
        sid = np.repeat(np.arange(len(starts) - 1), np.diff(starts)); i = np.arange(n); j = i + SHIFT[self.mode]
        ok = (j >= 0) & (j < n); ok[ok] &= sid[j[ok]] == sid[i[ok]]
        return np.where(ok, j, -1)

    def base(self, tok, starts, q, loo):
        sid = np.repeat(np.arange(len(starts) - 1), np.diff(starts)).astype(np.int32)
        X = np.empty((len(q), self.nb), np.float32)
        gaps = (q + SHIFT[self.mode]).astype(np.int64)
        self.pat.lookup(GROUPS[self.mode], tok, sid, gaps, q.astype(np.int64), loo, self.E, X, 0)
        X[:, self.np_:] = self.mem.run(tok, starts, MODE_ID[self.mode])[q]
        return X

    def _next_block(self, L, tok, starts, q, X):
        """the three new blocks a layer adds on top of its input X: guess, attention, products"""
        g = nz(X @ L["Wm"] + L["bm"]); A = self.att(tok, starts, q, g, MODE_ID[self.mode])
        c = np.c_[A, nz(X[:, self.np_:self.nb] @ self.R)]
        return g, A, c

    def features(self, tok, starts, q, loo, upto):
        X = self.base(tok, starts, q, loo)
        for k in range(1, upto):
            L = self.layers[k - 1]; g, A, c = self._next_block(L, tok, starts, q, X)
            X = np.c_[X, g, A, L["prod"](g, c)]
        return X

    # ------------------------------------------------------------------ training
    def fit(self, bank, keep_p, ntr=200_000, chunk=600, cache_dir="/tmp/sparc_cache", lam_m=1e-3, lam_v=1e-4, val=None, log=print):
        cdir = f"{cache_dir}_{self.mode}"; shutil.rmtree(cdir, ignore_errors=True); os.makedirs(cdir)
        rng = np.random.default_rng(self.seed); self.chunks = []; n = 0; t0 = time.time()
        for tok, starts in bank.chunks(chunk):
            tg = self.targets(len(tok), starts); cand = np.flatnonzero(tg >= 0)
            q = cand[rng.random(len(cand)) < keep_p[tok[tg[cand]]]]
            p = f"{cdir}/{len(self.chunks)}.npy"; np.save(p, self.base(tok, starts, q, True).astype(np.float16))
            self.chunks.append(dict(tok=tok, starts=starts, q=q, y=tok[tg[q]], p=p)); n += len(q)
            if n >= ntr: break
        self.S = None; report = []; log(f"{self.mode:5s} layer-1 inputs for {n:,} training targets: {time.time() - t0:.1f}s")
        for k in range(1, self.n_layers + 1):
            info = self._add_layer(k, lam_m, lam_v)
            if val is not None: info.update(self.evaluate(val, keep_p))
            report.append(info); log(f"{self.mode:5s} layer {k}: " + ", ".join(f"{a} {b:.3f}" if isinstance(b, float) else f"{a} {b}" for a, b in info.items()))
        for ch in self.chunks: os.remove(ch["p"])
        shutil.rmtree(cdir, ignore_errors=True); self.chunks = None; self.S = None
        for L in self.layers[:-1]: L["Wv"] = L["bv"] = None                     # keep the last V-token readout (for fill-in demos)
        self.layers[-1]["Wv"] = self.layers[-1]["Wv"].astype(np.float16)
        self.compile(); return report

    def _add_layer(self, k, lam_m, lam_v):
        t0 = time.time(); d = self.E.shape[1]
        if k > 1:
            L = self.layers[-1]
            if L["prod"] is None:                                              # fit / draw the products feeding layer k
                L["prod"] = make_products(self.kind, d, 2 * d, self.width, 100 * self.seed + k)
                G_, C_ = [], []
                for ch in self.chunks[:4]:
                    X = np.load(ch["p"]).astype(np.float32); g, A, c = self._next_block(L, ch["tok"], ch["starts"], ch["q"], X); G_.append(g); C_.append(c)
                L["prod"].fit(np.concatenate(G_), np.concatenate(C_))
            for ch in self.chunks:
                X = np.load(ch["p"]).astype(np.float32); g, A, c = self._next_block(L, ch["tok"], ch["starts"], ch["q"], X)
                np.save(ch["p"], np.c_[X, g, A, L["prod"](g, c)].astype(np.float16))
        Do = 0 if self.S is None else len(self.S["mu"]); t_feat = time.time() - t0
        s1 = s2 = None; m = 0
        for ch in self.chunks:
            N = np.load(ch["p"])[:, Do:].astype(np.float64)
            s1 = N.sum(0) if s1 is None else s1 + N.sum(0); s2 = (N ** 2).sum(0) if s2 is None else s2 + (N ** 2).sum(0); m += len(N)
        mu_n = (s1 / m).astype(np.float32); sd_n = (np.sqrt(np.maximum(s2 / m - (s1 / m) ** 2, 0)) + 1e-6).astype(np.float32); Dn = len(mu_n)
        C = np.zeros((Do, Dn)); NN = np.zeros((Dn, Dn)); XYm_n = np.zeros((Dn, d)); XYv_n = np.zeros((self.V, Dn), np.float32)
        ysum = np.zeros(d); ycnt = np.zeros(self.V)
        for ch in self.chunks:
            X = np.load(ch["p"]).astype(np.float32); y = ch["y"]; N = (X[:, Do:] - mu_n) / sd_n
            if Do: C += (((X[:, :Do] - self.S["mu"]) / self.S["sd"]).T @ N).astype(np.float64)
            NN += (N.T @ N).astype(np.float64); XYm_n += (N.T @ self.E[y]).astype(np.float64); ysum += self.E[y].sum(0); np.add.at(ycnt, y, 1)
            o = np.argsort(y, kind="stable"); u, si = np.unique(y[o], return_index=True); XYv_n[u] += np.add.reduceat(N[o], si, 0)
        if Do:
            G = np.block([[self.S["G"], C], [C.T, NN]]); XYm = np.r_[self.S["XYm"], XYm_n]; XYv = np.c_[self.S["XYv"], XYv_n]
            mu = np.r_[self.S["mu"], mu_n]; sd = np.r_[self.S["sd"], sd_n]
        else:
            G, XYm, XYv, mu, sd = NN, XYm_n, XYv_n, mu_n, sd_n
        del C, NN, XYm_n, XYv_n; self.S = dict(G=G, XYm=XYm, XYv=XYv, mu=mu, sd=sd); D = len(mu); t_stats = time.time() - t0 - t_feat
        tr = np.trace(G) / D
        Wm = cho_solve(cho_factor(G + lam_m * tr * np.eye(D)), XYm); bm = ysum / m
        cf = cho_factor(G + lam_v * tr * np.eye(D)); Wv = np.empty((D, self.V), np.float32)
        for v0 in range(0, self.V, 2000): Wv[:, v0:v0 + 2000] = cho_solve(cf, XYv[v0:v0 + 2000].T.astype(np.float64)).astype(np.float32)
        del cf; gc.collect()
        U_, S_, _ = randomized_svd(Wv, self.rh, n_iter=4, random_state=0); P = U_ * S_
        fold = lambda W, b: ((W / sd[:, None]).astype(np.float32), (b - (mu / sd) @ W).astype(np.float32))
        Wm_, bm_ = fold(Wm, bm); P_, bp_ = fold(P, np.zeros(self.rh)); Wv_, bv_ = fold(Wv, ycnt / m)
        self.layers.append(dict(Wm=Wm_, bm=bm_, P=P_, bp=bp_, Wv=Wv_, bv=bv_, prod=None))
        return dict(rows=m, features=D, t_features=round(t_feat, 1), t_stats=round(t_stats, 1), t_total=round(time.time() - t0, 1))

    # ------------------------------------------------------------------ evaluation / inference
    def evaluate(self, bank, keep_p, seed=7, chunk=500):
        L = self.layers[-1]; rng = np.random.default_rng(seed); vr = []
        Wv = L["Wv"].astype(np.float32)
        for tok, starts in bank.chunks(chunk):
            tg = self.targets(len(tok), starts); cand = np.flatnonzero(tg >= 0); q = cand[rng.random(len(cand)) < keep_p[tok[tg[cand]]]]
            X = self.features(tok, starts, q, False, len(self.layers)); y = tok[tg[q]]
            S = X @ Wv + L["bv"]; vr.append((S > S[np.arange(len(S)), y][:, None]).sum(1))
        vr = np.concatenate(vr); return dict(mrr=float(np.mean(1 / (vr + 1))), top1=float(np.mean(vr == 0)))

    def compile(self):
        """stack, per feature block, the [guess | hidden] readout rows of every layer that reads the block"""
        d = self.E.shape[1]; K = len(self.layers); sizes = [self.nb] + sum([[d, d, self.layers[k]["prod"].D] for k in range(K - 1)], [])
        off = np.cumsum([0] + sizes); self.blocks = []
        for bi in range(len(sizes)):
            first = 1 if bi == 0 else (bi - 1) // 3 + 2
            W = np.concatenate([np.c_[self.layers[k - 1]["Wm"][off[bi]:off[bi + 1]], self.layers[k - 1]["P"][off[bi]:off[bi + 1]]] for k in range(first, K + 1)], 1)
            self.blocks.append((first, np.ascontiguousarray(W, np.float32)))
        self.bias = [np.r_[L["bm"], L["bp"]].astype(np.float32) for L in self.layers]

    def encode(self, tok, starts):
        """per layer: guess g (what this stream expects for its target) and hidden state h, for every token."""
        K = len(self.layers); d = self.E.shape[1]; w = d + self.rh; q = np.arange(len(tok))
        acc = [np.tile(b, (len(tok), 1)) for b in self.bias]
        def push(bi, X):
            first, W = self.blocks[bi]; Y = X @ W
            for j, k in enumerate(range(first, K + 1)): acc[k - 1] += Y[:, j * w:(j + 1) * w]
        B = self.base(tok, starts, q, False); push(0, B); Rv = nz(B[:, self.np_:self.nb] @ self.R); del B
        outs = []
        for k in range(1, K + 1):
            o = acc[k - 1]; g = nz(o[:, :d]); outs.append(dict(g=g, h=nz(o[:, d:])))
            if k < K:
                A = self.att(tok, starts, q, g, MODE_ID[self.mode]); bi = 1 + 3 * (k - 1)
                push(bi, g); push(bi + 1, A); push(bi + 2, self.layers[k - 1]["prod"](g, np.c_[A, Rv]))
        return outs

    def scores(self, tok, starts, q):
        """V-token scores at positions q from the last layer (what this stream predicts for its target there)."""
        X = self.features(tok, starts, np.asarray(q, np.int64), False, len(self.layers)); L = self.layers[-1]
        return X @ L["Wv"].astype(np.float32) + L["bv"]
