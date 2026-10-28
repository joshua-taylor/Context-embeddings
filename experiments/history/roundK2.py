"""Round K2: which contextual blocks carry the most linearly-readable context? (POS / supersense on ambiguous words)
Every block is computed once per probe word; each representation is a column subset of one big X'X (sub-matrix solves).
Identity is given 768 dims in every representation so the probe is not starved of 'which word is this'."""
from hx import *
import pickle, gc
from scipy.linalg import cho_factor, cho_solve
from zoo import Zoo
from att import PMIAttn, query_attend
from nl import RandomBilinear
import probe as PR
LOG = "roundK2.log"; open(LOG, "w"); open(LOG, "a").write(PR.baseline_line() + "\n"); print(PR.baseline_line())
E = np.load(f"{BIG}/E.npy"); E768 = np.load(f"{BIG}/E768.npy"); d = 128; sc = pickle.load(open("tables128.pkl", "rb")); z = Zoo(E)
Zs = np.load("roundJ_stats.npz"); ORDER = ["P", "LIN", "SEL", "HIP", "ESN", "LINX"]; dims = Zs["dims"]; off = np.cumsum(np.r_[0, dims])
idx = {k: np.arange(off[i], off[i + 1]) for i, k in enumerate(ORDER)}; mu0, sd0 = Zs["mu"], Zs["sd"]; D0 = Zs["G"].shape[0] - 1
S1 = ["P", "LIN", "SEL", "ESN"]; cols1 = np.r_[np.concatenate([idx[k] for k in S1]), D0]; Gs = Zs["G"][np.ix_(cols1, cols1)]
W1 = cho_solve(cho_factor(Gs + 1e-3 * np.trace(Gs) / len(cols1) * np.eye(len(cols1))), Zs["XYm"][cols1]).astype(np.float32); del Zs, Gs; gc.collect()
freq = token_freq(load_corpus()); Zp = np.load(f"{BIG}/pmi_W8.npz"); KA, KB = np.ascontiguousarray(Zp["A"]), np.ascontiguousarray(Zp["B"])
w = np.sqrt(freq + 1e-9)[:, None]
MA = np.linalg.solve((E * w).T @ (E * w) + 1e-3 * np.eye(d), (E * w).T @ (KA * w)).astype(np.float32)
MB = np.linalg.solve((E * w).T @ (E * w) + 1e-3 * np.eye(d), (E * w).T @ (KB * w)).astype(np.float32)
at = PMIAttn(KA, KB, E, tau=0.4, sink=1.0); xp = RandomBilinear(d, 2 * d, 1024, 31)
BLK = ["E768", "LIN", "SEL", "ESN", "HIP", "G", "ATT", "ATT2", "XP"]
def chunk_blocks(t, st):
    sid = np.repeat(np.arange(len(st) - 1), np.diff(st)); pos = np.arange(len(t)); Pm = []
    for g, T in sc.tables.items(): M, lc = T.lookup(t, sid, pos, None, E); Pm += [M, lc]
    B = {"P": np.concatenate(Pm, 1)}
    for k in ["LIN", "SEL", "ESN", "HIP"]: B[k] = z.run(k, t, st)
    c1 = cols1[:-1]; X = np.concatenate([B[k] for k in S1], 1); g = nz(((X - mu0[c1]) / sd0[c1]) @ W1[:-1] + W1[-1]); del X
    B["G"] = g; B["ATT"] = nz(at(t, st)); o = np.empty((len(t), d), np.float32)
    query_attend(t, st, (g @ MA).astype(np.float32), (g @ MB).astype(np.float32), KA, KB, E, np.float32(0.2), np.float32(0.0), o); B["ATT2"] = nz(o)
    B["XP"] = xp(E[t], np.c_[B["G"], B["ATT"]]); B["E768"] = E768[t]; del B["P"]
    return B
# ---- per-word block matrices (float16) for the probe bank
dimsB = None; WM = None; wi = 0; t0 = time.time()
for c in range(0, PR.BANK.n, 600):
    s0, s1 = c, min(PR.BANK.n, c + 600); a, b = PR.BANK.starts[s0], PR.BANK.starts[s1]
    B = chunk_blocks(PR.BANK.tok[a:b].astype(np.int64), (PR.BANK.starts[s0:s1 + 1] - a).astype(np.int64))
    if WM is None:
        dimsB = {k: B[k].shape[1] for k in BLK}; WM = np.zeros((len(PR.W_IDX), sum(dimsB.values())), np.float16)
        offB = np.cumsum([0] + [dimsB[k] for k in BLK])
    j0 = wi
    while wi < len(PR.W_IDX) and PR.W_IDX[wi][0] < s1: wi += 1
    Fc = np.concatenate([B[k] for k in BLK], 1); del B
    for j in range(j0, wi): WM[j] = Fc[PR.W_IDX[j][2] - a].mean(0)
    del Fc
del sc; gc.collect(); tick(f"probe blocks computed {time.time()-t0:.0f}s: " + ", ".join(f"{k} {v}" for k, v in dimsB.items()))
X = WM; del WM
offB = np.cumsum([0] + [dimsB[k] for k in BLK]); ixB = {k: np.arange(offB[i], offB[i + 1]) for i, k in enumerate(BLK)}; DD = X.shape[1] + 1
RES = {}
for task in ("pos", "ss"):
    T = PR.TASKS[task]; m_tr = PR.IS_TR & T["keep"]; itr = np.flatnonzero(m_tr); cut = int(0.9 * len(itr)); fit, hold = itr[:cut], itr[cut:]
    te = np.flatnonzero((~PR.IS_TR) & T["keep"]); s1_ = np.zeros(X.shape[1]); s2_ = np.zeros(X.shape[1])
    for c in range(0, len(itr), 10000): x_ = X[itr[c:c + 10000]].astype(np.float64); s1_ += x_.sum(0); s2_ += (x_ ** 2).sum(0)
    mu = (s1_ / len(itr)).astype(np.float32); sd = (np.sqrt(np.maximum(s2_ / len(itr) - mu.astype(np.float64) ** 2, 0)) + 1e-6).astype(np.float32); nc = T["nc"]
    def std(rows): return np.c_[(X[rows].astype(np.float32) - mu) / sd, np.ones(len(rows), np.float32)].astype(np.float32)
    def stats(rows):
        G = np.zeros((DD, DD)); XY = np.zeros((DD, nc))
        for c in range(0, len(rows), 10000):
            Xs = std(rows[c:c + 10000]); G += (Xs.T @ Xs).astype(np.float64); Y = np.zeros((len(Xs), nc), np.float32); Y[np.arange(len(Xs)), T["y"][rows[c:c + 10000]]] = 1; XY += (Xs.T @ Y).astype(np.float64)
        return G, XY
    Gf, XYf = stats(fit); Gh, XYh = stats(hold); Xh = std(hold).astype(np.float16); Xt = np.concatenate([std(te[c:c + 10000]).astype(np.float16) for c in range(0, len(te), 10000)])
    RES[task] = (Gf, XYf, Gh, XYh, Xh, Xt, hold, te)
def pred(Xq, cols, W): return np.concatenate([(Xq[c:c + 10000][:, cols].astype(np.float32) @ W).argmax(1) for c in range(0, len(Xq), 10000)])
def run(spec):
    cols = np.r_[np.concatenate([ixB[k] for k in spec]), DD - 1]; out = {}
    for task, (Gf, XYf, Gh, XYh, Xh, Xt, hold, te) in RES.items():
        T = PR.TASKS[task]; dd = len(cols); g = Gf[np.ix_(cols, cols)]; xy = XYf[cols]; best = None
        for lam in (1e-3, 1e-2, 1e-1):
            W = np.linalg.solve(g + lam * np.trace(g) / dd * np.eye(dd), xy).astype(np.float32); a_ = np.mean(pred(Xh, cols, W) == T["y"][hold])
            if best is None or a_ > best[0]: best = (a_, lam)
        g2 = g + Gh[np.ix_(cols, cols)]; W = np.linalg.solve(g2 + best[1] * np.trace(g2) / dd * np.eye(dd), xy + XYh[cols]).astype(np.float32)
        ok = pred(Xt, cols, W) == T["y"][te]; amb = T["amb"][te]; out[task] = (ok.mean(), ok[amb].mean())
    line = f"  {' + '.join(spec):46s} dims {len(cols)-1:5d} | POS all {out['pos'][0]*100:5.1f} ambiguous {out['pos'][1]*100:5.1f} | supersense all {out['ss'][0]*100:5.1f} ambiguous {out['ss'][1]*100:5.1f}"
    tick(line); open(LOG, "a").write(line + "\n")
SPECS = [["E768"], ["E768", "LIN"], ["E768", "SEL"], ["E768", "ESN"], ["E768", "HIP"], ["E768", "LIN", "SEL", "ESN", "HIP"],
         ["E768", "G"], ["E768", "ATT"], ["E768", "ATT2"], ["E768", "G", "ATT", "ATT2"], ["E768", "XP"], ["E768", "G", "ATT", "ATT2", "XP"],
         ["E768", "LIN", "SEL", "ESN", "HIP", "G", "ATT", "ATT2", "XP"], ["LIN", "SEL", "ESN", "HIP", "G", "ATT", "ATT2", "XP"]]
for s in SPECS: run(s)
tick("done")
