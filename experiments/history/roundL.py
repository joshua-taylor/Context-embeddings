"""Round L: two-stage 'counting' attention inside the gap-filler (a closed-form 2-layer transformer).
Stage 1: gap guess g1 from patterns + memories (round-J readout).
Stage 2: g1 is mapped into PMI query space (closed-form ridge from meanings to the PMI factor tables), attends over
the sentence with PMI scores (directional), and the attended meanings feed a second readout.
Controls: uniform attention over the sentence (leave-one-out sentence mean = 'topic')."""
from hx import *
import pickle, gc
from scipy.linalg import cho_factor, cho_solve
from zoo import Zoo
from att import query_attend
LOG = "roundL.log"; open(LOG, "w")
E = np.load(f"{BIG}/E.npy"); d = E.shape[1]; sc = pickle.load(open("tables128.pkl", "rb")); z = Zoo(E)
Zs = np.load("roundJ_stats.npz"); ORDER = ["P", "LIN", "SEL", "HIP", "ESN", "LINX"]; dims = Zs["dims"]; off = np.cumsum(np.r_[0, dims])
idx = {k: np.arange(off[i], off[i + 1]) for i, k in enumerate(ORDER)}; mu0, sd0 = Zs["mu"], Zs["sd"]; D0 = Zs["G"].shape[0] - 1
S1 = ["P", "LIN", "SEL", "ESN"]
cols1 = np.r_[np.concatenate([idx[k] for k in S1]), D0]; Gs = Zs["G"][np.ix_(cols1, cols1)]
W1 = cho_solve(cho_factor(Gs + 1e-3 * np.trace(Gs) / len(cols1) * np.eye(len(cols1))), Zs["XYm"][cols1]).astype(np.float32); del Zs, Gs; gc.collect()
ids = load_corpus().astype(np.int64); n = len(ids); hi = int(0.9 * n); freq = token_freq(ids)
Zp = np.load(f"{BIG}/pmi_W8.npz"); KA, KB = np.ascontiguousarray(Zp["A"]), np.ascontiguousarray(Zp["B"])
w = np.sqrt(freq + 1e-9)[:, None]                                   # frequency-weighted ridge: meanings -> PMI query space
MA = np.linalg.solve((E * w).T @ (E * w) + 1e-3 * np.eye(d), (E * w).T @ (KA * w)).astype(np.float32)
MB = np.linalg.solve((E * w).T @ (E * w) + 1e-3 * np.eye(d), (E * w).T @ (KB * w)).astype(np.float32)
r = np.random.default_rng(0); s_, e_ = sentence_bounds(ids, 0, hi); ktr = np.sort(r.choice(len(s_), 100_000, replace=False))
sv, ev = sentence_bounds(ids, hi, n); kva = np.sort(r.choice(len(sv), 3000, replace=False))
train = Bank(toks=[ids[s_[i]:e_[i]] for i in ktr]); val = Bank(toks=[ids[sv[i]:ev[i]] for i in kva]); del ids; gc.collect()
keep_p = np.minimum(1.0, np.sqrt(1e-4 / (freq + 1e-12)) + 1e-4 / (freq + 1e-12)).astype(np.float32)
TAUS = (0.2, 0.5)
def blocks(bank, s0, s1, loo, rng):
    lo, hi_ = bank.starts[s0], bank.starts[s1]; t = bank.tok[lo:hi_].astype(np.int64); st = (bank.starts[s0:s1 + 1] - lo).astype(np.int64)
    sid = np.repeat(np.arange(s1 - s0), np.diff(st)); pos = np.flatnonzero(rng.random(len(t)) < keep_p[t]); allp = np.arange(len(t))
    Pm = []
    for g, T in sc.tables.items(): M, lc = T.lookup(t, sid, allp, t if loo else None, E); Pm += [M, lc]
    B = {"P": np.concatenate(Pm, 1)}
    for k in ["LIN", "SEL", "ESN"]: B[k] = z.run(k, t, st)
    c1 = cols1[:-1]; X = np.concatenate([B[k] for k in S1], 1)
    g1 = nz(((X - mu0[c1]) / sd0[c1]) @ W1[:-1] + W1[-1]); del X
    qA, qB = (g1 @ MA).astype(np.float32), (g1 @ MB).astype(np.float32)
    for tau in TAUS:
        o = np.empty((len(t), d), np.float32); query_attend(t, st, qA, qB, KA, KB, E, np.float32(tau), np.float32(0.0), o); B[f"ATT{tau}"] = o
    S = np.add.reduceat(E[t], st[:-1], 0); L = np.diff(st); B["TOPIC"] = nz((S[sid] - E[t]) / np.maximum(L[sid] - 1, 1)[:, None])
    B["G1"] = g1
    return {k: v[pos] for k, v in B.items()}, t[pos]
ORD = ["P", "LIN", "SEL", "ESN", "G1", "TOPIC"] + [f"ATT{t}" for t in TAUS]
G = None; NTR = 0; rng = np.random.default_rng(1)
for c in range(0, train.n, 800):
    B, y = blocks(train, c, min(train.n, c + 800), True, rng); X = np.concatenate([B[k] for k in ORD], 1).astype(np.float32)
    if G is None:
        dm = {k: B[k].shape[1] for k in ORD}; D = X.shape[1]; mu = X.mean(0); sd = X.std(0) + 1e-6
        G = np.zeros((D + 1, D + 1)); XYm = np.zeros((D + 1, d)); XYv = np.zeros((V, D + 1))
    Xs = np.c_[(X - mu) / sd, np.ones(len(X), np.float32)].astype(np.float32); del X
    G += (Xs.T @ Xs).astype(np.float64); XYm += (Xs.T @ E[y]).astype(np.float64)
    o = np.argsort(y, kind="stable"); u, st_ = np.unique(y[o], return_index=True); Xo = Xs[o]; del Xs; XYv[u] += np.add.reduceat(Xo, st_, axis=0); del Xo, B; gc.collect()
    NTR += len(y)
    if NTR >= 250_000: break
tick(f"accumulated {NTR:,} gaps, {D} features")
rv = np.random.default_rng(2); Xv, yv = [], []
for c in range(0, val.n, 500):
    Bv, y_ = blocks(val, c, min(val.n, c + 500), False, rv); X_ = np.concatenate([Bv[k] for k in ORD], 1); Xv.append(np.c_[(X_ - mu) / sd, np.ones(len(X_), np.float32)].astype(np.float32)); yv.append(y_); del Bv, X_
Xv = np.concatenate(Xv); yv = np.concatenate(yv)
np.savez("roundL_stats.npz", G=G, XYm=XYm, XYv=XYv, Xv=Xv, yv=yv, dims=np.array([dm[k] for k in ORD]), names=np.array(ORD))
tick("saved stats")
