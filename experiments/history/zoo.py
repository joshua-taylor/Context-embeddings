"""Reservoir zoo. Every reservoir reads token meanings, resets at each sentence start, and returns for every token
the forward state BEFORE it and the backward state AFTER it (so a slot never sees its own token).

LIN  : diagonal leaky linear units, half fast (a in .1-.7) half slow (.8-.98), bias-corrected (divide by 1-a^k)
HIP  : HiPPO-LegT: per input channel, N Legendre coefficients of the last theta tokens (theta = 4 and 16);
       an optimal linear memory of the recent window: it keeps WHERE things were, not just a decayed sum
ESN  : classic nonlinear echo-state network: h <- (1-l) h + l tanh(W h + Win x + b), spectral radius .9
SEL  : selective (token-gated) diagonal units: each token decides how much each unit keeps/forgets
LINX : LIN followed by a fixed random tanh expansion of [fwd ; bwd] (static nonlinearity on a linear memory)
"""
import numpy as np, numba as nb

@nb.njit(parallel=True, cache=True, fastmath=True)
def k_lin(tok, starts, T, a, H):
    U = T.shape[1]; ns = len(starts) - 1
    for k in nb.prange(ns):
        s0 = starts[k]; s1 = starts[k + 1]; h = np.zeros(U, np.float32); c = np.zeros(U, np.float32)
        for i in range(s0, s1):
            for u in range(U): H[i, u] = h[u] / c[u] if c[u] > 0 else 0.0
            for u in range(U): h[u] = a[u] * h[u] + (1 - a[u]) * T[tok[i], u]; c[u] = a[u] * c[u] + (1 - a[u])
        h[:] = 0.0; c[:] = 0.0
        for i in range(s1 - 1, s0 - 1, -1):
            for u in range(U): H[i, U + u] = h[u] / c[u] if c[u] > 0 else 0.0
            for u in range(U): h[u] = a[u] * h[u] + (1 - a[u]) * T[tok[i], u]; c[u] = a[u] * c[u] + (1 - a[u])

@nb.njit(parallel=True, cache=True, fastmath=True)
def k_sel(tok, starts, T, A, H):
    U = T.shape[1]; ns = len(starts) - 1
    for k in nb.prange(ns):
        s0 = starts[k]; s1 = starts[k + 1]; h = np.zeros(U, np.float32); c = np.zeros(U, np.float32)
        for i in range(s0, s1):
            for u in range(U): H[i, u] = h[u] / c[u] if c[u] > 0 else 0.0
            x = tok[i]
            for u in range(U): a = A[x, u]; h[u] = a * h[u] + (1 - a) * T[x, u]; c[u] = a * c[u] + (1 - a)
        h[:] = 0.0; c[:] = 0.0
        for i in range(s1 - 1, s0 - 1, -1):
            for u in range(U): H[i, U + u] = h[u] / c[u] if c[u] > 0 else 0.0
            x = tok[i]
            for u in range(U): a = A[x, u]; h[u] = a * h[u] + (1 - a) * T[x, u]; c[u] = a * c[u] + (1 - a)

@nb.njit(parallel=True, cache=True, fastmath=True)
def k_hippo(tok, starts, Uin, Ad, Bd, H):
    """Uin: vocab x C channel inputs; Ad: (W, N, N), Bd: (W, N) discretised LegT for W windows. state C*N*W per dir."""
    C = Uin.shape[1]; Wn, N = Bd.shape; D = C * N * Wn; ns = len(starts) - 1
    for k in nb.prange(ns):
        s0 = starts[k]; s1 = starts[k + 1]
        for d_ in range(2):
            h = np.zeros(D, np.float32); tmp = np.zeros(N, np.float32)
            rng = range(s0, s1) if d_ == 0 else range(s1 - 1, s0 - 1, -1)
            for i in rng:
                for j in range(D): H[i, d_ * D + j] = h[j]
                x = tok[i]
                for w in range(Wn):
                    for c in range(C):
                        base = (w * C + c) * N; u = Uin[x, c]
                        for n in range(N):
                            acc = Bd[w, n] * u
                            for m in range(N): acc += Ad[w, n, m] * h[base + m]
                            tmp[n] = acc
                        for n in range(N): h[base + n] = tmp[n]

@nb.njit(parallel=True, cache=True, fastmath=True)
def k_esn(tok, starts, Win, W, b, leak, H):
    U = W.shape[0]; ns = len(starts) - 1
    for k in nb.prange(ns):
        s0 = starts[k]; s1 = starts[k + 1]
        for d_ in range(2):
            h = np.zeros(U, np.float32); z = np.zeros(U, np.float32)
            rng = range(s0, s1) if d_ == 0 else range(s1 - 1, s0 - 1, -1)
            for i in rng:
                for u in range(U): H[i, d_ * U + u] = h[u]
                x = tok[i]
                for u in range(U):
                    acc = Win[x, u] + b[u]
                    for v in range(U): acc += W[u, v] * h[v]
                    z[u] = np.tanh(acc)
                for u in range(U): h[u] = (1 - leak[u]) * h[u] + leak[u] * z[u]

def legt(N, theta):
    n = np.arange(N); M = np.zeros((N, N))
    for i in range(N):
        for j in range(N): M[i, j] = (2 * i + 1) * ((-1.0) ** (i - j) if j <= i else 1.0)
    A = -M / theta; B = ((2 * n + 1) * (-1.0) ** n) / theta
    I = np.eye(N); inv = np.linalg.inv(I - A / 2); return inv @ (I + A / 2), inv @ B

class Zoo:
    def __init__(self, E, U=256, seed=3):
        r = np.random.default_rng(seed); d = E.shape[1]; self.U = U
        self.T = (E @ (r.standard_normal((d, U)) / np.sqrt(d))).astype(np.float32)
        self.a = np.r_[r.uniform(0.1, 0.7, U // 2), r.uniform(0.8, 0.98, U - U // 2)].astype(np.float32)
        # selective gates
        g = r.standard_normal((d, U)); g /= np.linalg.norm(g, axis=0); z = E @ g; z = (z - z.mean(0)) / z.std(0); z = 1.5 * z + r.normal(0, 0.5, U)
        lo = np.r_[np.full(U // 2, 0.05), np.full(U - U // 2, 0.6)]; hi = np.r_[np.full(U // 2, 0.9), np.full(U - U // 2, 0.995)]
        self.Asel = (lo + (hi - lo) / (1 + np.exp(-z))).astype(np.float32)
        self.Tsel = (E @ (r.standard_normal((d, U)) / np.sqrt(d))).astype(np.float32)
        # HiPPO-LegT: C channels x N orders x 2 windows = U per direction
        self.N = 8; self.thetas = (4.0, 16.0); C = U // (self.N * len(self.thetas))
        self.Uin = (E @ (r.standard_normal((d, C)) / np.sqrt(d)) * 3).astype(np.float32)
        AB = [legt(self.N, th) for th in self.thetas]; self.Ad = np.stack([x[0] for x in AB]).astype(np.float32); self.Bd = np.stack([x[1] for x in AB]).astype(np.float32)
        # ESN
        W = r.standard_normal((U, U)) * (r.random((U, U)) < 0.1); W *= 0.9 / np.max(np.abs(np.linalg.eigvals(W))); self.W = W.astype(np.float32)
        self.Win = (E @ (r.standard_normal((d, U)) / np.sqrt(d)) * 1.5).astype(np.float32); self.b = r.uniform(-0.2, 0.2, U).astype(np.float32)
        self.leak = np.r_[r.uniform(0.6, 1.0, U // 2), r.uniform(0.1, 0.4, U - U // 2)].astype(np.float32)
        # LINX
        self.Wx = (r.standard_normal((2 * U, 2 * U)) / np.sqrt(2 * U) * 1.5).astype(np.float32); self.bx = r.uniform(-1, 1, 2 * U).astype(np.float32)
    def run(self, kind, t, starts, Hlin=None):
        t = t.astype(np.int64); starts = starts.astype(np.int64); n = len(t); H = np.empty((n, 2 * self.U), np.float32)
        if kind == "LIN": k_lin(t, starts, self.T, self.a, H)
        elif kind == "SEL": k_sel(t, starts, self.Tsel, self.Asel, H)
        elif kind == "HIP": k_hippo(t, starts, self.Uin, self.Ad, self.Bd, H)
        elif kind == "ESN": k_esn(t, starts, self.Win, self.W, self.b, self.leak, H)
        elif kind == "LINX":
            if Hlin is None: Hlin = self.run("LIN", t, starts)
            if not hasattr(self, "lx_mu"): self.lx_mu = Hlin.mean(0); self.lx_sd = Hlin.std(0) + 1e-6   # fixed on first (training) batch
            S = (Hlin - self.lx_mu) / self.lx_sd; H = np.tanh(S @ self.Wx + self.bx).astype(np.float32)
        return H

# ---------------- consolidation round: cheaper / more diverse diagonal memories ----------------
@nb.njit(parallel=True, cache=True, fastmath=True)
def k_osc(tok, starts, T, r, cs, sn, H):
    """complex diagonal units: h <- r e^{i theta} h + (1 - r) x ; output (Re, Im) / bias. Phase encodes 'how long ago'."""
    U = T.shape[1]; ns = len(starts) - 1
    for k in nb.prange(ns):
        s0 = starts[k]; s1 = starts[k + 1]
        for d_ in range(2):
            hr = np.zeros(U, np.float32); hi = np.zeros(U, np.float32); c = np.zeros(U, np.float32)
            rng = range(s0, s1) if d_ == 0 else range(s1 - 1, s0 - 1, -1)
            base = d_ * 2 * U
            for i in rng:
                for u in range(U):
                    if c[u] > 0: H[i, base + u] = hr[u] / c[u]; H[i, base + U + u] = hi[u] / c[u]
                    else: H[i, base + u] = 0.0; H[i, base + U + u] = 0.0
                x = tok[i]
                for u in range(U):
                    a = r[u] * (cs[u] * hr[u] - sn[u] * hi[u]) + (1 - r[u]) * T[x, u]
                    b = r[u] * (sn[u] * hr[u] + cs[u] * hi[u])
                    hr[u] = a; hi[u] = b; c[u] = r[u] * c[u] + (1 - r[u])

class Zoo2:
    def __init__(self, E, seed=7):
        r = np.random.default_rng(seed); d = E.shape[1]; self.z = Zoo(E)
        # wide linear: 512 units/direction, log-spaced timescales
        U = 512; tau = np.exp(r.uniform(np.log(2), np.log(100), U - U // 4)); a = np.r_[r.uniform(0.1, 0.7, U // 4), 1 - 1 / tau]
        self.TW = (E @ (r.standard_normal((d, U)) / np.sqrt(d))).astype(np.float32); self.aW = a.astype(np.float32)
        # oscillating: 128 complex units/direction (512 numbers total)
        Uo = 128; self.TO = (E @ (r.standard_normal((d, Uo)) / np.sqrt(d))).astype(np.float32)
        self.rO = np.r_[r.uniform(0.5, 0.8, Uo // 2), r.uniform(0.85, 0.98, Uo - Uo // 2)].astype(np.float32)
        th = 2 * np.pi / np.exp(r.uniform(np.log(2), np.log(40), Uo)); self.cO = np.cos(th).astype(np.float32); self.sO = np.sin(th).astype(np.float32)
        # HiPPO retry: 64 channels (was 16), 4 Legendre orders, windows 4 and 16 -> 512 numbers/direction
        self.N = 4; C = 64; self.Uin2 = (E @ (r.standard_normal((d, C)) / np.sqrt(d)) * 3).astype(np.float32)
        AB = [legt(self.N, th_) for th_ in (4.0, 16.0)]; self.Ad2 = np.stack([x[0] for x in AB]).astype(np.float32); self.Bd2 = np.stack([x[1] for x in AB]).astype(np.float32)
    def run(self, kind, t, starts):
        t = t.astype(np.int64); starts = starts.astype(np.int64); n = len(t)
        if kind in ("LIN", "SEL"): return self.z.run(kind, t, starts)
        if kind == "LINW": H = np.empty((n, 1024), np.float32); k_lin(t, starts, self.TW, self.aW, H)
        elif kind == "OSC": H = np.empty((n, 512), np.float32); k_osc(t, starts, self.TO, self.rO, self.cO, self.sO, H)
        elif kind == "HIP2": H = np.empty((n, 1024), np.float32); k_hippo(t, starts, self.Uin2, self.Ad2, self.Bd2, H)
        return H
