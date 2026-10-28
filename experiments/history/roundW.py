"""Round W: SPARC-3 token vectors (cloze + forward + backward streams): probe, sentences, word-in-context, speed."""
from hx import *
from hx import _sem_keys
import pickle, gc
from nl import RandomBilinear
from sparc3 import GROUPS
import probe as PR
LOG = "roundW.log"; open(LOG, "w"); open(LOG, "a").write(PR.baseline_line() + "\n"); print(PR.baseline_line())
E = np.load(f"{BIG}/E.npy"); E768 = np.load(f"{BIG}/E768.npy")
sc = pickle.load(open("tables128.pkl", "rb")); tables = sc.tables; del sc
S = {}
for m in ("cloze", "fwd", "bwd"):
    s = pickle.load(open(f"stream_{m}.pkl", "rb")); s.tabs = {g: tables[g] for g in GROUPS[m]}; S[m] = s
for g in ("LLR", "LRR"): tables.pop(g, None)
gc.collect(); K = 3; xp = RandomBilinear(128, 256, 2048, 77)
def blocks(t, st):
    t = t.astype(np.int64); st = st.astype(np.int64); out = {"E768": E768[t]}
    for m, s in S.items():
        for k, o in enumerate(s.encode(t, st), 1): out[f"{m}_g{k}"] = o["g"]; out[f"{m}_h{k}"] = o["h"]
    out["XPfb"] = xp(E[t], np.c_[out[f"fwd_g{K}"], out[f"bwd_g{K}"]]); return out
# ---------- speed
st_ = TEST.starts[:np.searchsorted(TEST.starts, 30000, "right")].astype(np.int64); t_ = TEST.tok[:st_[-1]].astype(np.int64)
for s in S.values(): s.encode(t_[:st_[5]], st_[:6])
for m, s in S.items():
    t0 = time.time(); s.encode(t_, st_); line = f"  speed: {m} stream, {K} layers: {len(t_)/(time.time()-t0):,.0f} tokens/s"; tick(line); open(LOG, "a").write(line + "\n")
# ---------- probe
BLK = ["E768"] + [f"{m}_{v}{k}" for m in ("cloze", "fwd", "bwd") for v in ("g", "h") for k in range(1, K + 1)] + ["XPfb"]
WM = None; wi = 0
for c in range(0, PR.BANK.n, 500):
    s0, s1 = c, min(PR.BANK.n, c + 500); a, b = PR.BANK.starts[s0], PR.BANK.starts[s1]
    Bk = blocks(PR.BANK.tok[a:b], PR.BANK.starts[s0:s1 + 1] - a)
    if WM is None: dimsB = {k: Bk[k].shape[1] for k in BLK}; WM = np.zeros((len(PR.W_IDX), sum(dimsB.values())), np.float16)
    j0 = wi
    while wi < len(PR.W_IDX) and PR.W_IDX[wi][0] < s1: wi += 1
    Fc = np.concatenate([Bk[k] for k in BLK], 1); del Bk
    for j in range(j0, wi): WM[j] = Fc[PR.W_IDX[j][2] - a].mean(0)
    del Fc
tick("probe features"); X = WM; del WM
offB = np.cumsum([0] + [dimsB[k] for k in BLK]); ixB = {k: np.arange(offB[i], offB[i + 1]) for i, k in enumerate(BLK)}; DD = X.shape[1] + 1
L3 = lambda m: [f"{m}_g3", f"{m}_h3"]; ALL = lambda m: [f"{m}_{v}{k}" for v in ("g", "h") for k in range(1, K + 1)]
SPECS = [["E768"], ["E768"] + L3("cloze"), ["E768"] + ["fwd_g1", "fwd_h1", "bwd_g1", "bwd_h1"], ["E768"] + L3("fwd") + L3("bwd"),
         ["E768"] + ALL("fwd") + ALL("bwd"), ["E768", "XPfb"], ["E768"] + ALL("fwd") + ALL("bwd") + ALL("cloze") + ["XPfb"], ALL("fwd") + ALL("bwd")]
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
NAMES = {0: "meaning (768)", 1: "+ cloze layer 3", 2: "+ fwd/bwd layer 1", 3: "+ fwd/bwd layer 3", 4: "+ fwd/bwd all layers", 5: "+ word x (fwd,bwd) products",
         6: "+ everything (3 streams, all layers, products)", 7: "fwd/bwd all layers WITHOUT the meaning"}
for i, spec in enumerate(SPECS):
    o = OUT[tuple(spec)]
    line = f"  {NAMES[i]:48s} dims {o['pos'][2]:5d} | POS all {o['pos'][0]*100:5.1f} ambiguous {o['pos'][1]*100:5.1f} | supersense all {o['ss'][0]*100:5.1f} ambiguous {o['ss'][1]*100:5.1f}"
    tick(line); open(LOG, "a").write(line + "\n")
del X; gc.collect()
# ---------- word in context (coarse + fine, per lemma) and sentences
def wic_vec(cols_names):
    def f(t, st):
        B = blocks(t, st); return np.concatenate([nz(B[k]) for k in cols_names], 1)
    return f
for nm, spec in [("fwd/bwd layer 3 (no meaning)", L3("fwd") + L3("bwd")), ("meaning + fwd/bwd all layers", ["E768"] + ALL("fwd") + ALL("bwd"))]:
    wic_eval(wic_vec(spec), nm, LOG)
ref = make_ref(load_corpus(), n=10000)
def sent(t, st):
    B = blocks(t, st); return np.c_[B["E768"], B["cloze_g1"], B["fwd_h3"], B["bwd_h3"], B["fwd_g3"], B["bwd_g3"]].astype(np.float32)
Wt = {"mean": lambda t, s, F: np.ones(len(t), np.float32)}
te_, _ = run_bank(TEST, sent, Wt, need=NEED_SENT); rf_, _ = run_bank(ref, sent, Wt)
for nm, sl in {"meaning + cloze guess (SPARC)": slice(0, 896), "+ fwd/bwd layer-3 hidden": slice(0, 1152), "+ fwd/bwd layer-3 hidden & guess": slice(0, 1408)}.items():
    f = Post(rf_["mean"][:, sl], white=True); r = metrics(f(te_["mean"][:, sl])); line = f"  {nm:40s} mean+white {fmt(r)}"; tick(line); open(LOG, "a").write(line + "\n")
tick("done")
