"""SPARC-3 engineering pass (same maths as sparc3.Stream, organised for speed).

Training: the same training rows are used for every layer and each layer's feature matrix is cached (float16, on disk),
          so fitting layer k+1 never recomputes pattern lookups, reservoirs or earlier layers.
Inference: no residual concatenation. Each feature block, the moment it exists, is multiplied ONCE by the stacked
           readout rows of every later layer that will read it ([meaning | hidden] for layers k..K in one BLAS call).
           Layer k's outputs are then just accumulated partial sums.
"""
import numpy as np, os, gc, time, shutil
from scipy.linalg import cho_factor, cho_solve
from sklearn.utils.extmath import randomized_svd
from nl import RandomBilinear
from sparc3 import Stream, nz

class FastStream(Stream):
    def add_layer(self, bank, keep_p, ntr=200_000, chunk=600, lam_m=1e-3, lam_v=1e-4, cache="/home/claude/big/cache3"):
        k = len(self.layers) + 1; t0 = time.time(); d = self.E.shape[1]; cdir = f"{cache}_{self.mode}"
        if k == 1:
            shutil.rmtree(cdir, ignore_errors=True); os.makedirs(cdir); rng = np.random.default_rng(self.seed); self.chunks = []; n_ = 0
            for c in range(0, bank.n, chunk):
                s0, s1 = c, min(bank.n, c + chunk); lo, hi_ = bank.starts[s0], bank.starts[s1]
                t = bank.tok[lo:hi_].astype(np.int64); st = (bank.starts[s0:s1 + 1] - lo).astype(np.int64)
                tg = self.target_pos(len(t), st); cand = np.flatnonzero(tg >= 0); q = cand[rng.random(len(cand)) < keep_p[t[tg[cand]]]]
                X = self.base(t, st, q, True); p = f"{cdir}/{len(self.chunks)}.npy"; np.save(p, X.astype(np.float16))
                self.chunks.append(dict(t=t, st=st, q=q, y=t[tg[q]], p=p)); n_ += len(q)
                if n_ >= ntr: break
            self.nb = X.shape[1]
        else:
            L = self.layers[-1]; L["xp"] = RandomBilinear(d, 256, self.dprod, 100 * self.seed + k); L["Wv"] = None; L["bv"] = None
            for ch in self.chunks:                                   # extend cached features by one layer
                X = np.load(ch["p"]).astype(np.float32); g = nz(X @ L["Wm"] + L["bm"]); A = self.attend(ch["t"], ch["st"], ch["q"], g)
                X = np.c_[X, g, A, L["xp"](g, np.c_[A, nz(X[:, self.nb - self.nres:self.nb] @ self.R)])]; np.save(ch["p"], X.astype(np.float16))
        t_feat = time.time() - t0
        G = None
        for ch in self.chunks:
            X = np.load(ch["p"]).astype(np.float32); y = ch["y"]
            if G is None: D = X.shape[1]; s1 = np.zeros(D); s2 = np.zeros(D); m = 0
            s1 += X.sum(0); s2 += (X.astype(np.float64) ** 2).sum(0); m += len(X)
        mu = (s1 / m).astype(np.float32); sd = (np.sqrt(np.maximum(s2 / m - (s1 / m) ** 2, 0)) + 1e-6).astype(np.float32)
        G = np.zeros((D + 1, D + 1)); XYm = np.zeros((D + 1, d)); XYv = np.zeros((16000, D + 1), np.float32)
        for ch in self.chunks:
            Xq = np.c_[(np.load(ch["p"]).astype(np.float32) - mu) / sd, np.ones(len(ch["y"]), np.float32)].astype(np.float32); y = ch["y"]
            G += (Xq.T @ Xq).astype(np.float64); XYm += (Xq.T @ self.E[y]).astype(np.float64)
            o = np.argsort(y, kind="stable"); u, si = np.unique(y[o], return_index=True); XYv[u] += np.add.reduceat(Xq[o], si, 0); del Xq
        tr = np.trace(G) / len(G); Wm = cho_solve(cho_factor(G + lam_m * tr * np.eye(len(G))), XYm); cf = cho_factor(G + lam_v * tr * np.eye(len(G))); del G
        Wv = np.empty((D + 1, 16000), np.float32)
        for v0 in range(0, 16000, 2000): Wv[:, v0:v0 + 2000] = cho_solve(cf, XYv[v0:v0 + 2000].T.astype(np.float64)).astype(np.float32)
        del XYv, cf; gc.collect()
        U_, S_, _ = randomized_svd(Wv[:-1], self.rh, n_iter=4, random_state=0); P = U_ * S_
        f = lambda W: ((W[:-1] / sd[:, None]).astype(np.float32), (W[-1] - (mu / sd) @ W[:-1]).astype(np.float32))
        Wm_, bm = f(Wm); P_, bp = f(np.r_[P, np.zeros((1, P.shape[1]))]); Wv_, bv = f(Wv)
        self.layers.append(dict(Wm=Wm_, bm=bm, P=P_, bp=bp, Wv=Wv_, bv=bv, xp=None))
        return dict(n=sum(len(c["y"]) for c in self.chunks), t_feat=t_feat, t_total=time.time() - t0, dims=D)
    def compile(self):
        """stack, for every feature block, the readout rows of all layers that read it -> one matmul per block at inference"""
        K = len(self.layers); d = self.E.shape[1]; self.blocks = []          # (start, end) row ranges of each block in the residual stream
        sizes = [self.nb] + sum([[d, d, self.dprod] for _ in range(K - 1)], [])
        off = np.cumsum([0] + sizes); self.blocks = [(off[i], off[i + 1]) for i in range(len(sizes))]
        self.RW = []
        for bi, (a, b) in enumerate(self.blocks):
            first = 1 if bi == 0 else (bi - 1) // 3 + 2                   # first layer whose input contains this block
            self.RW.append((first, np.concatenate([np.c_[self.layers[k - 1]["Wm"][a:b], self.layers[k - 1]["P"][a:b]] for k in range(first, K + 1)], 1).astype(np.float32)))
        self.RB = [np.r_[L["bm"], L["bp"]] for L in self.layers]
    def encode_fast(self, t, starts):
        K = len(self.layers); d = self.E.shape[1]; q = np.arange(len(t)); acc = [np.tile(self.RB[k], (len(t), 1)) for k in range(K)]
        def push(bi, X):
            first, W = self.RW[bi]; Y = X @ W
            for j, k in enumerate(range(first, K + 1)): acc[k - 1] += Y[:, j * 2 * d:(j + 1) * 2 * d] if self.rh == d else Y[:, j * (d + self.rh):(j + 1) * (d + self.rh)]
        B = self.base(t, starts, q, False); push(0, B); Rv = nz(B[:, self.nb - self.nres:self.nb] @ self.R); del B
        outs = []
        for k in range(1, K + 1):
            o = acc[k - 1]; g = nz(o[:, :d]); h = nz(o[:, d:]); outs.append(dict(g=g, h=h))
            if k < K:
                A = self.attend(t, starts, q, g); bi = 1 + 3 * (k - 1)
                push(bi, g); push(bi + 1, A); push(bi + 2, self.layers[k - 1]["xp"](g, np.c_[A, Rv]))
        return outs
    def clear_cache(self):
        for ch in getattr(self, "chunks", []):
            try: os.remove(ch["p"])
            except OSError: pass
        self.chunks = []

class FastStream2(FastStream):
    """+ incremental statistics: layer k+1 reuses layer k's X'X (its features are a prefix of layer k+1's residual stream),
    so only the cross terms of the NEW blocks are computed. Features are centred on the training rows, which decouples
    the bias, so X'X needs no bias row."""
    def add_layer(self, bank, keep_p, ntr=200_000, chunk=600, lam_m=1e-3, lam_v=1e-4, cache="/home/claude/big/cache3"):
        k = len(self.layers) + 1; t0 = time.time(); d = self.E.shape[1]; cdir = f"{cache}_{self.mode}"
        if k == 1:
            shutil.rmtree(cdir, ignore_errors=True); os.makedirs(cdir); rng = np.random.default_rng(self.seed); self.chunks = []; n_ = 0
            for c in range(0, bank.n, chunk):
                s0, s1 = c, min(bank.n, c + chunk); lo, hi_ = bank.starts[s0], bank.starts[s1]
                t = bank.tok[lo:hi_].astype(np.int64); st = (bank.starts[s0:s1 + 1] - lo).astype(np.int64)
                tg = self.target_pos(len(t), st); cand = np.flatnonzero(tg >= 0); q = cand[rng.random(len(cand)) < keep_p[t[tg[cand]]]]
                X = self.base(t, st, q, True); p = f"{cdir}/{len(self.chunks)}.npy"; np.save(p, X.astype(np.float16))
                self.chunks.append(dict(t=t, st=st, q=q, y=t[tg[q]], p=p)); n_ += len(q)
                if n_ >= ntr: break
            self.nb = X.shape[1]; Do = 0; self.S = None
        else:
            L = self.layers[-1]; L["xp"] = RandomBilinear(d, 256, self.dprod, 100 * self.seed + k); L["Wv"] = None; L["bv"] = None
            Do = self.S["mu"].shape[0]
            for ch in self.chunks:
                X = np.load(ch["p"]).astype(np.float32); g = nz(X @ L["Wm"] + L["bm"]); A = self.attend(ch["t"], ch["st"], ch["q"], g)
                X = np.c_[X, g, A, L["xp"](g, np.c_[A, nz(X[:, self.nb - self.nres:self.nb] @ self.R)])]; np.save(ch["p"], X.astype(np.float16))
        t_feat = time.time() - t0
        # pass 1: mean / sd of the NEW columns
        s1 = s2 = None; m = 0
        for ch in self.chunks:
            N = np.load(ch["p"])[:, Do:].astype(np.float64)
            s1 = N.sum(0) if s1 is None else s1 + N.sum(0); s2 = (N ** 2).sum(0) if s2 is None else s2 + (N ** 2).sum(0); m += len(N)
        mu_n = (s1 / m).astype(np.float32); sd_n = (np.sqrt(np.maximum(s2 / m - (s1 / m) ** 2, 0)) + 1e-6).astype(np.float32); Dn = len(mu_n)
        # pass 2: only the blocks that involve new columns
        C = np.zeros((Do, Dn)); NN = np.zeros((Dn, Dn)); XYm_n = np.zeros((Dn, d)); XYv_n = np.zeros((16000, Dn), np.float32); ysum_m = np.zeros(d); ycnt = np.zeros(16000)
        for ch in self.chunks:
            X = np.load(ch["p"]).astype(np.float32); y = ch["y"]; N = (X[:, Do:] - mu_n) / sd_n
            if Do: Xo = (X[:, :Do] - self.S["mu"]) / self.S["sd"]; C += (Xo.T @ N).astype(np.float64); del Xo
            NN += (N.T @ N).astype(np.float64); XYm_n += (N.T @ self.E[y]).astype(np.float64); ysum_m += self.E[y].sum(0); np.add.at(ycnt, y, 1)
            o = np.argsort(y, kind="stable"); u, si = np.unique(y[o], return_index=True); XYv_n[u] += np.add.reduceat(N[o], si, 0); del X, N
        if Do:
            G = np.block([[self.S["G"], C], [C.T, NN]]); XYm = np.r_[self.S["XYm"], XYm_n]; XYv = np.c_[self.S["XYv"], XYv_n]
            mu = np.r_[self.S["mu"], mu_n]; sd = np.r_[self.S["sd"], sd_n]
        else: G, XYm, XYv, mu, sd = NN, XYm_n, XYv_n, mu_n, sd_n
        del C, NN, XYm_n, XYv_n; self.S = dict(G=G, XYm=XYm, XYv=XYv, mu=mu, sd=sd); D = len(mu); t_stats = time.time() - t0 - t_feat
        tr = np.trace(G) / D
        Wm = cho_solve(cho_factor(G + lam_m * tr * np.eye(D)), XYm); bm = ysum_m / m
        cf = cho_factor(G + lam_v * tr * np.eye(D)); Wv = np.empty((D, 16000), np.float32)
        for v0 in range(0, 16000, 2000): Wv[:, v0:v0 + 2000] = cho_solve(cf, XYv[v0:v0 + 2000].T.astype(np.float64)).astype(np.float32)
        bvv = (ycnt / m).astype(np.float32); del cf; gc.collect()
        U_, S_, _ = randomized_svd(Wv, self.rh, n_iter=4, random_state=0); P = U_ * S_
        fold = lambda W, b: ((W / sd[:, None]).astype(np.float32), (b - (mu / sd) @ W).astype(np.float32))
        Wm_, bm_ = fold(Wm, bm); P_, bp_ = fold(P, np.zeros(P.shape[1])); Wv_, bv_ = fold(Wv, bvv)
        self.layers.append(dict(Wm=Wm_, bm=bm_, P=P_, bp=bp_, Wv=Wv_, bv=bv_, xp=None))
        return dict(n=m, t_feat=t_feat, t_stats=t_stats, t_total=time.time() - t0, dims=D)
    def clear_cache(self):
        super().clear_cache(); self.S = None
