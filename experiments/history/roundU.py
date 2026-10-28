"""Round U: do deeper layers give richer token vectors? Probe (POS / supersense on ambiguous words) + sentences."""
from hx import *
import pickle, gc
from nl import RandomBilinear
import probe as PR
LOG = "roundU.log"; open(LOG, "w"); open(LOG, "a").write(PR.baseline_line() + "\n"); print(PR.baseline_line())
E = np.load(f"{BIG}/E.npy"); E768 = np.load(f"{BIG}/E768.npy")
m = pickle.load(open("deep_model_T.pkl", "rb")); sc = pickle.load(open("tables128.pkl", "rb"))
for g in ("LLR", "LRR"): del sc.tables[g]
m.sc = sc
for L in m.layers: L["Wv"] = None
gc.collect(); K = len(m.layers)
xp1 = RandomBilinear(128, 256, 2048, 61); xpd = RandomBilinear(128, 256, 2048, 62)
def blocks(t, st):
    B, outs = m.token_features(t.astype(np.int64), st.astype(np.int64)); out = {"E768": E768[t]}
    for k, o in enumerate(outs, 1): out[f"g{k}"] = o["g"]; out[f"h{k}"] = o["h"]
    A1 = m.attend(t, st, outs[0]["g"]); AK = m.attend(t, st, outs[-1]["g"])
    out["XP1"] = xp1(E[t], np.c_[outs[0]["g"], A1]); out["XPdeep"] = xpd(E[t], np.c_[outs[-1]["g"], AK])
    return out
BLK = ["E768"] + [f"g{k}" for k in range(1, K + 1)] + [f"h{k}" for k in range(1, K + 1)] + ["XP1", "XPdeep"]
WM = None; wi = 0; t0 = time.time()
for c in range(0, PR.BANK.n, 500):
    s0, s1 = c, min(PR.BANK.n, c + 500); a, b = PR.BANK.starts[s0], PR.BANK.starts[s1]
    Bk = blocks(PR.BANK.tok[a:b].astype(np.int64), (PR.BANK.starts[s0:s1 + 1] - a).astype(np.int64))
    if WM is None: dimsB = {k: Bk[k].shape[1] for k in BLK}; WM = np.zeros((len(PR.W_IDX), sum(dimsB.values())), np.float16)
    j0 = wi
    while wi < len(PR.W_IDX) and PR.W_IDX[wi][0] < s1: wi += 1
    Fc = np.concatenate([Bk[k] for k in BLK], 1); del Bk
    for j in range(j0, wi): WM[j] = Fc[PR.W_IDX[j][2] - a].mean(0)
    del Fc
tick(f"probe blocks {time.time()-t0:.0f}s"); X = WM; del WM
offB = np.cumsum([0] + [dimsB[k] for k in BLK]); ixB = {k: np.arange(offB[i], offB[i + 1]) for i, k in enumerate(BLK)}; DD = X.shape[1] + 1
SPECS = [["E768"], ["E768", "g1"], ["E768", "h1"]] + [["E768", f"h{k}"] for k in range(2, K + 1)] + [["E768"] + [f"h{k}" for k in range(1, K + 1)],
         ["E768", "g1", "XP1"], ["E768", f"g{K}", "XPdeep"], ["E768"] + [f"h{k}" for k in range(1, K + 1)] + ["g1", "XP1", "XPdeep"]]
OUT = {tuple(s): {} for s in SPECS}
def pred(Xq, cols, W): return np.concatenate([(Xq[c:c + 5000][:, cols].astype(np.float32) @ W).argmax(1) for c in range(0, len(Xq), 5000)])
for task in ("pos", "ss"):
    T = PR.TASKS[task]; itr = np.flatnonzero(PR.IS_TR & T["keep"]); cut = int(0.9 * len(itr)); fit, hold = itr[:cut], itr[cut:]
    te = np.flatnonzero((~PR.IS_TR) & T["keep"]); nc = T["nc"]; s1_ = np.zeros(X.shape[1]); s2_ = np.zeros(X.shape[1])
    for c in range(0, len(itr), 10000): x_ = X[itr[c:c + 10000]].astype(np.float64); s1_ += x_.sum(0); s2_ += (x_ ** 2).sum(0)
    mu = (s1_ / len(itr)).astype(np.float32); sd = (np.sqrt(np.maximum(s2_ / len(itr) - mu.astype(np.float64) ** 2, 0)) + 1e-6).astype(np.float32)
    std = lambda rows: np.c_[(X[rows].astype(np.float32) - mu) / sd, np.ones(len(rows), np.float32)].astype(np.float32)
    def stats(rows):
        G = np.zeros((DD, DD)); XY = np.zeros((DD, nc))
        for c in range(0, len(rows), 5000):
            Xs = std(rows[c:c + 5000]); G += (Xs.T @ Xs).astype(np.float64); Y = np.zeros((len(Xs), nc), np.float32); Y[np.arange(len(Xs)), T["y"][rows[c:c + 5000]]] = 1; XY += (Xs.T @ Y).astype(np.float64)
        return G, XY
    Gf, XYf = stats(fit); Gh, XYh = stats(hold); Xh = std(hold).astype(np.float16); Xt = np.concatenate([std(te[c:c + 5000]).astype(np.float16) for c in range(0, len(te), 5000)])
    for spec in SPECS:
        cols = np.r_[np.concatenate([ixB[k] for k in spec]), DD - 1]; dd = len(cols); g = Gf[np.ix_(cols, cols)]; xy = XYf[cols]; best = None
        for lam in (1e-3, 1e-2, 1e-1):
            W = np.linalg.solve(g + lam * np.trace(g) / dd * np.eye(dd), xy).astype(np.float32); a_ = np.mean(pred(Xh, cols, W) == T["y"][hold])
            if best is None or a_ > best[0]: best = (a_, lam)
        g2 = g + Gh[np.ix_(cols, cols)]; W = np.linalg.solve(g2 + best[1] * np.trace(g2) / dd * np.eye(dd), xy + XYh[cols]).astype(np.float32)
        ok = pred(Xt, cols, W) == T["y"][te]; amb = T["amb"][te]; OUT[tuple(spec)][task] = (ok.mean(), ok[amb].mean(), dd - 1)
    del Gf, Gh, XYf, XYh, Xh, Xt; gc.collect()
for spec in SPECS:
    o = OUT[tuple(spec)]
    line = f"  {' + '.join(spec):56s} dims {o['pos'][2]:5d} | POS all {o['pos'][0]*100:5.1f} ambiguous {o['pos'][1]*100:5.1f} | supersense all {o['ss'][0]*100:5.1f} ambiguous {o['ss'][1]*100:5.1f}"
    tick(line); open(LOG, "a").write(line + "\n")
tick("done")
