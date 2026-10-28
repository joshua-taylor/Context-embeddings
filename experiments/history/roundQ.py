"""Round Q: consolidated reservoirs (wide linear, oscillating, HiPPO retry) with 5 pattern groups. Based on round J: inside the gap-filler (sparse patterns + full-vocabulary readout).
One pass computes every feature block for the same training gaps and accumulates ONE big X'X and X'Y;
every combination of blocks is then just a sub-matrix solve (no re-running)."""
from hx import *
import pickle, gc, os
from scipy.linalg import cho_factor, cho_solve
from sparse import SparseCloze
from zoo import Zoo2
LOG = "roundQ.log"; open(LOG, "w")
E = np.load(f"{BIG}/E.npy"); d = E.shape[1]
ids = load_corpus().astype(np.int64); n = len(ids); hi = int(0.9 * n); freq = token_freq(ids)
t0 = time.time()
if os.path.exists("tables128.pkl"): sc = pickle.load(open("tables128.pkl", "rb"))
else: sc = SparseCloze(E, ids, DOT, hi=hi, res=False); pickle.dump(sc, open("tables128.pkl", "wb"))
tick(f"pattern tables ready {time.time()-t0:.0f}s")
r = np.random.default_rng(0)
s_, e_ = sentence_bounds(ids, 0, hi); ktr = np.sort(r.choice(len(s_), 100_000, replace=False))
sv, ev = sentence_bounds(ids, hi, n); kva = np.sort(r.choice(len(sv), 3000, replace=False))
train = Bank(toks=[ids[s_[i]:e_[i]] for i in ktr]); val = Bank(toks=[ids[sv[i]:ev[i]] for i in kva]); del ids; gc.collect()
keep_p = np.minimum(1.0, np.sqrt(1e-4 / (freq + 1e-12)) + 1e-4 / (freq + 1e-12)).astype(np.float32)
z = Zoo2(E); KINDS = ["LIN", "SEL", "LINW", "OSC", "HIP2"]
for g in ("LLR", "LRR"): del sc.tables[g]
def blocks(bank, s0, s1, loo, rng):
    lo, hi_ = bank.starts[s0], bank.starts[s1]; t = bank.tok[lo:hi_].astype(np.int64); st = (bank.starts[s0:s1 + 1] - lo).astype(np.int64)
    sid = np.repeat(np.arange(s1 - s0), np.diff(st)); pos = np.flatnonzero(rng.random(len(t)) < keep_p[t])
    out = {}; Pm = []
    for g, T in sc.tables.items():
        M, lc = T.lookup(t, sid, pos, t[pos] if loo else None, E); Pm += [M, lc]
    out["P"] = np.concatenate(Pm, 1)
    for k in KINDS: out[k] = z.run(k, t, st)[pos]
    return out, t[pos]
ORDER = ["P"] + KINDS
# ---- one pass over the training gaps
G = None; NTR = 0; tm = {k: 0.0 for k in ORDER}
rng = np.random.default_rng(1)
for c in range(0, train.n, 800):
    B, y = blocks(train, c, min(train.n, c + 800), True, rng)
    X = np.concatenate([B[k] for k in ORDER], 1).astype(np.float32)
    if G is None:
        dims = {k: B[k].shape[1] for k in ORDER}; D = X.shape[1]; mu = X.mean(0); sd = X.std(0) + 1e-6
        G = np.zeros((D + 1, D + 1)); XYm = np.zeros((D + 1, d)); XYv = np.zeros((V, D + 1))
    Xs = np.c_[(X - mu) / sd, np.ones(len(X), np.float32)].astype(np.float32); del X
    G += (Xs.T @ Xs).astype(np.float64); XYm += (Xs.T @ E[y]).astype(np.float64)
    o = np.argsort(y, kind="stable"); u, st_ = np.unique(y[o], return_index=True); Xo = Xs[o]; del Xs; XYv[u] += np.add.reduceat(Xo, st_, axis=0); del Xo
    NTR += len(y); del B; gc.collect()
    if NTR >= 250_000: break
tick(f"accumulated {NTR:,} training gaps, {D} features")
rv = np.random.default_rng(2); Xv, yv = [], []
for c in range(0, val.n, 500):
    Bv, y_ = blocks(val, c, min(val.n, c + 500), False, rv); X_ = np.concatenate([Bv[k] for k in ORDER], 1); Xv.append(np.c_[(X_ - mu) / sd, np.ones(len(X_), np.float32)].astype(np.float32)); yv.append(y_); del Bv, X_
Xv = np.concatenate(Xv); yv = np.concatenate(yv); del sc; gc.collect()
np.savez("roundQ_stats.npz", G=G, XYm=XYm, XYv=XYv, Xv=Xv, yv=yv, mu=mu, sd=sd, dims=np.array([dims[k] for k in ORDER]), names=np.array(ORDER)); tick("saved"); sys.exit(0)
off = np.cumsum([0] + [dims[k] for k in ORDER]); idx = {k: np.arange(off[i], off[i + 1]) for i, k in enumerate(ORDER)}
def solve(names, lams=(1e-4, 1e-3)):
    cols = np.r_[np.concatenate([idx[k] for k in names]), D]; Gs = G[np.ix_(cols, cols)]; Xs = Xv[:, cols]; best = None; dd = len(cols)
    for lam in lams:
        cf = cho_factor(Gs + lam * np.trace(Gs) / dd * np.eye(dd))
        W = cho_solve(cf, XYv[:, cols].T).astype(np.float32); rk = []
        for c in range(0, len(Xs), 2000):
            S = Xs[c:c + 2000] @ W; rk.append((S > S[np.arange(len(S)), yv[c:c + 2000]][:, None]).sum(1))
        rk = np.concatenate(rk); mrr = float(np.mean(1 / (rk + 1)))
        if best is None or mrr > best[0]: best = (mrr, float(np.mean(rk == 0)), float(np.mean(rk < 10)), lam)
        del W
    Wm = cho_solve(cho_factor(Gs + best[3] * np.trace(Gs) / dd * np.eye(dd)), XYm[cols]).astype(np.float32)
    Gm = nz(Xs @ Wm) @ E.T; rk = (Gm > Gm[np.arange(len(Gm)), yv][:, None]).sum(1)
    line = f"  {'+'.join(names):22s} ({dd-1:5d} feats)  vocab MRR {best[0]:.3f} top1 {best[1]*100:5.1f}% top10 {best[2]*100:5.1f}% | meaning-space MRR {np.mean(1/(rk+1)):.3f}"
    tick(line); open(LOG, "a").write(line + "\n")
COMBOS = [["LIN"], ["SEL"], ["HIP"], ["ESN"], ["LINX"], ["LIN", "SEL", "HIP", "ESN", "LINX"],
          ["P"], ["P", "LIN"], ["P", "SEL"], ["P", "HIP"], ["P", "ESN"], ["P", "LINX"],
          ["P", "LIN", "HIP"], ["P", "LIN", "ESN"], ["P", "LIN", "SEL"], ["P", "HIP", "ESN"], ["P", "LIN", "SEL", "HIP", "ESN", "LINX"]]
for cmb in COMBOS: solve(cmb)
tick("done")
