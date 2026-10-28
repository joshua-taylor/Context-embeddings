"""SPARC-3: deep closed-form encoder with three streams that share one design.

  cloze : guess token t from both sides (t hidden)                    -> 'what fits here'
  fwd   : guess token t+1 from tokens <= t (t visible)                -> word-in-left-context representation
  bwd   : guess token t-1 from tokens >= t (t visible)                -> word-in-right-context representation
The fwd/bwd streams are the visible-token pathway: each word's own representation, built from its identity and its
context, trained to predict its neighbour without ever seeing it (leak-free, ELMo-style).

Per stream, layer 1 features ('base'):
  sparse products = pattern-table lookups for the target gap (fwd: keys x_t and (x_{t-1}, x_t); bwd: x_t and (x_t, x_{t+1});
                    cloze: L, R, LR, LL, RR around t)
  memories        = linear + selective diagonal reservoir states (cloze: exclusive both sides; fwd/bwd: inclusive, one side)
Layer k+1 features = residual stream (everything layer k saw) + layer-k guess + counting attention queried by the guess
                     (causal for fwd/bwd) + random products of the guess with [attention ; memories].
Readouts per layer: meaning guess (128-d), full 16k-token scores (training metric + hidden directions).

Engineering:
  - training touches only the sampled target positions (lookups, attention, products computed for those rows only);
    reservoirs scan whole sentences but are cheap
  - standardisation is folded into the readout weights (no standardised copies at inference)
  - the 16k-token readout is solved in vocabulary slices (low memory) and only kept for the newest layer
"""
import numpy as np, numba as nb, gc, time
from scipy.linalg import cho_factor, cho_solve
from sklearn.utils.extmath import randomized_svd
from nl import RandomBilinear

def nz(X): return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)

@nb.njit(parallel=True, cache=True, fastmath=True)
def k_res(tok, starts, T, a, A, gated, incl, out):
    """diagonal leaky reservoir, bias-corrected. out[:, :U] forward, out[:, U:] backward.
    incl=True: state includes token i; False: forward state BEFORE i, backward state AFTER i."""
    U = T.shape[1]; ns = len(starts) - 1
    for k in nb.prange(ns):
        s0 = starts[k]; s1 = starts[k + 1]
        for d_ in range(2):
            h = np.zeros(U, np.float32); c = np.zeros(U, np.float32)
            rng = range(s0, s1) if d_ == 0 else range(s1 - 1, s0 - 1, -1)
            for i in rng:
                x = tok[i]
                if not incl:
                    for u in range(U): out[i, d_ * U + u] = h[u] / c[u] if c[u] > 0 else 0.0
                for u in range(U):
                    aa = A[x, u] if gated else a[u]
                    h[u] = aa * h[u] + (1 - aa) * T[x, u]; c[u] = aa * c[u] + (1 - aa)
                if incl:
                    for u in range(U): out[i, d_ * U + u] = h[u] / c[u]

@nb.njit(parallel=True, cache=True, fastmath=True)
def k_attend(tok, qpos, qs0, qs1, QA, QB, KA, KB, V, mode, tau, out):
    """counting attention for query rows. mode 0 (cloze): keys s != i, PMI direction by side; mode 1 (fwd, target after
    all keys): keys s <= i, score KA[s].QB ; mode 2 (bwd, target before all keys): keys s >= i, score QA.KB[s]."""
    nq = len(qpos); k = KA.shape[1]; d = V.shape[1]
    for qi in nb.prange(nq):
        i = qpos[qi]; s0 = qs0[qi]; s1 = qs1[qi]; L = s1 - s0; sc = np.empty(L + 1, np.float32); mx = 0.0; sc[L] = 0.0
        for s in range(s0, s1):
            ok = (s != i) if mode == 0 else ((s <= i) if mode == 1 else (s >= i))
            if not ok: sc[s - s0] = -1e30; continue
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

GROUPS = {"cloze": ("L", "R", "LR", "LL", "RR"), "fwd": ("L", "LL"), "bwd": ("R", "RR")}
OFFS = {"L": (-1,), "R": (1,), "LR": (-1, 1), "LL": (-2, -1), "RR": (1, 2)}

class Stream:
    def __init__(self, mode, E, tables, res, freq, pmi, dprod=2048, rh=128, seed=0):
        self.mode = mode; self.E = E; self.res = res; self.rh = rh; self.dprod = dprod; d = E.shape[1]
        self.tabs = {g: tables[g] for g in GROUPS[mode]}; self.layers = []; self.seed = seed
        self.KA, self.KB = pmi; w = np.sqrt(freq + 1e-9)[:, None]; G_ = (E * w).T @ (E * w) + 1e-3 * np.eye(d)
        self.MA = np.linalg.solve(G_, (E * w).T @ (self.KA * w)).astype(np.float32); self.MB = np.linalg.solve(G_, (E * w).T @ (self.KB * w)).astype(np.float32)
        self.nres = 4 * res["U"] if mode == "cloze" else 2 * res["U"]
        self.R = (np.random.default_rng(42).standard_normal((self.nres, 128)) / np.sqrt(self.nres)).astype(np.float32)
    # ---- targets
    def target_pos(self, n, starts):
        """for each position: index of the token this stream predicts (or -1 if outside the sentence)"""
        sid = np.repeat(np.arange(len(starts) - 1), np.diff(starts)); i = np.arange(n)
        if self.mode == "cloze": return i
        j = i + (1 if self.mode == "fwd" else -1); ok = (j >= 0) & (j < n); ok[ok] &= sid[j[ok]] == sid[i[ok]]; return np.where(ok, j, -1)
    # ---- layer-1 features for query rows
    def base(self, t, starts, q, loo):
        n = len(t); sid = np.repeat(np.arange(len(starts) - 1), np.diff(starts)); tgt = self.target_pos(n, starts)
        gap = tgt[q] if self.mode != "cloze" else q
        parts = []
        for g, T in self.tabs.items():
            code = np.zeros(len(q), np.int64); ok = np.ones(len(q), bool)
            for o in OFFS[g]:
                p = gap + o; okp = (p >= 0) & (p < n); pc = np.clip(p, 0, n - 1); ok &= okp & (sid[pc] == sid[q]); code = code * 16000 + t[pc]
            j = np.clip(np.searchsorted(T.keys, code), 0, len(T.keys) - 1); hit = ok & (T.keys[j] == code)
            cnt = np.where(hit, T.cnt[j], 0).astype(np.float32); M = np.where(hit[:, None], T.mean[j].astype(np.float32), 0)
            if loo:
                y = t[gap]; l1 = cnt - 1; good = l1 > 0
                M = np.where(good[:, None], (M * cnt[:, None] - self.E[y]) / np.maximum(l1, 1)[:, None], 0); cnt = np.where(good, l1, 0)
            parts += [M.astype(np.float32), np.log1p(cnt)[:, None].astype(np.float32)]
        U = self.res["U"]; H = []
        for kind in ("LIN", "SEL"):
            out = np.empty((n, 2 * U), np.float32); r = self.res[kind]
            k_res(t, starts, r["T"], r["a"], r["A"], kind == "SEL", self.mode != "cloze", out)
            H.append(out if self.mode == "cloze" else (out[:, :U] if self.mode == "fwd" else out[:, U:]))
        parts.append(np.concatenate(H, 1)[q]); return np.concatenate(parts, 1)
    def attend(self, t, starts, q, g):
        sid = np.repeat(np.arange(len(starts) - 1), np.diff(starts)); out = np.empty((len(q), self.E.shape[1]), np.float32)
        k_attend(t, q.astype(np.int64), starts[sid[q]].astype(np.int64), starts[sid[q] + 1].astype(np.int64), (g @ self.MA).astype(np.float32),
                 (g @ self.MB).astype(np.float32), self.KA, self.KB, self.E, {"cloze": 0, "fwd": 1, "bwd": 2}[self.mode], np.float32(0.2), out)
        return nz(out)
    # ---- layer stack
    def features(self, t, starts, q, loo, upto):
        """features of layer `upto` at query rows, plus outputs of layers < upto"""
        X = self.base(t, starts, q, loo); nb_ = X.shape[1]; outs = []
        for k in range(1, upto):
            L = self.layers[k - 1]; g = nz(X @ L["Wm"] + L["bm"]); h = nz(X @ L["P"] + L["bp"]); outs.append(dict(g=g, h=h))
            A = self.attend(t, starts, q, g); ctx = np.c_[A, nz(X[:, nb_ - self.nres:nb_] @ self.R)]
            X = np.c_[X, g, A, L["xp"](g, ctx)]
        return X, outs
    def add_layer(self, bank, keep_p, ntr=200_000, chunk=600, lam_m=1e-3, lam_v=1e-4):
        k = len(self.layers) + 1; rng = np.random.default_rng(self.seed + k); G = None; n_ = 0; t0 = time.time()
        if k > 1: self.layers[-1]["xp"] = RandomBilinear(self.E.shape[1], 256, self.dprod, 100 * self.seed + k); self.layers[-1]["Wv"] = None
        for c in range(0, bank.n, chunk):
            s0, s1 = c, min(bank.n, c + chunk); lo, hi_ = bank.starts[s0], bank.starts[s1]
            t = bank.tok[lo:hi_].astype(np.int64); st = (bank.starts[s0:s1 + 1] - lo).astype(np.int64)
            tg = self.target_pos(len(t), st); cand = np.flatnonzero(tg >= 0); q = cand[rng.random(len(cand)) < keep_p[t[tg[cand]]]]
            X, _ = self.features(t, st, q, True, k); y = t[tg[q]]
            if G is None: D = X.shape[1]; mu = X.mean(0); sd = X.std(0) + 1e-6; G = np.zeros((D + 1, D + 1)); XYm = np.zeros((D + 1, self.E.shape[1])); XYv = np.zeros((16000, D + 1), np.float32)
            Xq = np.c_[(X - mu) / sd, np.ones(len(X), np.float32)].astype(np.float32); del X
            G += (Xq.T @ Xq).astype(np.float64); XYm += (Xq.T @ self.E[y]).astype(np.float64)
            o = np.argsort(y, kind="stable"); u, si = np.unique(y[o], return_index=True); XYv[u] += np.add.reduceat(Xq[o], si, 0); del Xq; n_ += len(y)
            if n_ >= ntr: break
        t_feat = time.time() - t0; tr = np.trace(G) / len(G)
        Wm = cho_solve(cho_factor(G + lam_m * tr * np.eye(len(G))), XYm); cf = cho_factor(G + lam_v * tr * np.eye(len(G))); del G
        Wv = np.empty((XYv.shape[1], 16000), np.float32)
        for v0 in range(0, 16000, 2000): Wv[:, v0:v0 + 2000] = cho_solve(cf, XYv[v0:v0 + 2000].T.astype(np.float64)).astype(np.float32)
        del XYv, cf; gc.collect()
        U_, S_, _ = randomized_svd(Wv[:-1], self.rh, n_iter=4, random_state=0); P = U_ * S_
        # fold standardisation: outputs computed directly on raw features
        f = lambda W: ((W[:-1] / sd[:, None]).astype(np.float32), (W[-1] - (mu / sd) @ W[:-1]).astype(np.float32))
        Wm_, bm = f(Wm); P_, bp = f(np.r_[P, np.zeros((1, P.shape[1]))]); Wv_, bv = f(Wv)
        self.layers.append(dict(Wm=Wm_, bm=bm, P=P_, bp=bp, Wv=Wv_, bv=bv, xp=None))
        return dict(n=n_, t_feat=t_feat, t_total=time.time() - t0, dims=D)
    def evaluate(self, bank, keep_p, seed=7, chunk=500):
        k = len(self.layers); L = self.layers[-1]; rng = np.random.default_rng(seed); vr = []; mr = []
        for c in range(0, bank.n, chunk):
            s0, s1 = c, min(bank.n, c + chunk); lo, hi_ = bank.starts[s0], bank.starts[s1]
            t = bank.tok[lo:hi_].astype(np.int64); st = (bank.starts[s0:s1 + 1] - lo).astype(np.int64)
            tg = self.target_pos(len(t), st); cand = np.flatnonzero(tg >= 0); q = cand[rng.random(len(cand)) < keep_p[t[tg[cand]]]]
            X, _ = self.features(t, st, q, False, k); y = t[tg[q]]
            S = X @ L["Wv"] + L["bv"]; vr.append((S > S[np.arange(len(S)), y][:, None]).sum(1))
            S = nz(X @ L["Wm"] + L["bm"]) @ self.E.T; mr.append((S > S[np.arange(len(S)), y][:, None]).sum(1))
        vr = np.concatenate(vr); mr = np.concatenate(mr)
        return dict(vocab_mrr=float(np.mean(1 / (vr + 1))), top1=float(np.mean(vr == 0)), meaning_mrr=float(np.mean(1 / (mr + 1))))
    def encode(self, t, starts):
        """all positions, all layers: returns list of dict(g, h) per layer (rows = tokens; fwd/bwd last-token rows have no target but are still encoded)"""
        q = np.arange(len(t)); X = self.base(t, starts, q, False); nb_ = X.shape[1]; outs = []
        for k, L in enumerate(self.layers, 1):
            g = nz(X @ L["Wm"] + L["bm"]); h = nz(X @ L["P"] + L["bp"]); outs.append(dict(g=g, h=h))
            if k < len(self.layers):
                A = self.attend(t, starts, q, g); X = np.c_[X, g, A, L["xp"](g, np.c_[A, nz(X[:, nb_ - self.nres:nb_] @ self.R)])]
        return outs
    def strip(self):
        for L in self.layers: L["Wv"] = None; L["bv"] = None

def make_reservoirs(E, U=256, seed=3):
    """the fused linear + selective reservoir (same parameters as the round-Q 'LIN' and 'SEL')"""
    from zoo import Zoo
    z = Zoo(E, U=U, seed=seed)
    return dict(U=U, LIN=dict(T=z.T, a=z.a, A=np.zeros((1, 1), np.float32)), SEL=dict(T=z.Tsel, a=np.zeros(U, np.float32), A=z.Asel))
