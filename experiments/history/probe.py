"""Linear probes for contextual richness (SemCor, split by document).
Word vector = mean over the word's BPE pieces of a model's per-token features.
Probe = closed-form ridge classifier (one-hot targets), lambda picked on a slice of the training words.
Reported: accuracy on ALL words and on AMBIGUOUS words (types seen with >= 2 labels in training), plus the
'most frequent tag of this word' baseline, which needs no context. Context-free models cannot beat it on ambiguous words."""
import json, numpy as np
from collections import Counter, defaultdict
from hx import Bank, nz, tick
P = json.load(open("/home/claude/bench/data/probe.json"))
r = np.random.default_rng(0)
tr = [s for s in P if s["split"] == "train"]; te = [s for s in P if s["split"] == "test"]
import os
NTR_PROBE = int(os.environ.get("NTR_PROBE", 8000)); tr = [tr[i] for i in r.choice(len(tr), NTR_PROBE, replace=False)]; te = [te[i] for i in r.choice(len(te), 2500, replace=False)]
SENTS = tr + te; BANK = Bank([s["w"] for s in SENTS]); NTR_S = len(tr)
# word -> token pieces
W_IDX = []          # (sentence, word index, token positions)
lo = BANK.starts
for si in range(BANK.n):
    wm = BANK.wmap[lo[si]:lo[si + 1]]
    for wi in range(len(SENTS[si]["w"])):
        m = np.flatnonzero(wm == wi) + lo[si]
        if len(m): W_IDX.append((si, wi, m))
def labels(task):
    key = "pos" if task == "pos" else "ss"; out = []
    for si, wi, m in W_IDX: out.append(SENTS[si][key][wi])
    return out
WORD = [SENTS[si]["w"][wi] for si, wi, _ in W_IDX]; IS_TR = np.array([si < NTR_S for si, _, _ in W_IDX])
TASKS = {}
for task in ("pos", "ss"):
    lab = labels(task); keep = np.array([l is not None for l in lab]); classes = sorted({l for l in lab if l is not None}); ci = {c: i for i, c in enumerate(classes)}
    y = np.array([ci[l] if l is not None else -1 for l in lab])
    by = defaultdict(Counter)
    for w, l, t in zip(WORD, y, IS_TR):
        if t and l >= 0: by[w][l] += 1
    amb = np.array([len(by[w]) >= 2 for w in WORD]); glob = Counter(y[IS_TR & keep]).most_common(1)[0][0]
    mfs = np.array([by[w].most_common(1)[0][0] if by[w] else glob for w in WORD])
    TASKS[task] = dict(y=y, keep=keep, amb=amb, mfs=mfs, nc=len(classes))
def word_matrix(feats, chunk=600):
    """per-word feature matrix (mean over pieces), preallocated"""
    out = None; wi = 0
    for c in range(0, BANK.n, chunk):
        s0, s1 = c, min(BANK.n, c + chunk); a, b = BANK.starts[s0], BANK.starts[s1]
        F = feats(BANK.tok[a:b].astype(np.int64), (BANK.starts[s0:s1 + 1] - a).astype(np.int64))
        if out is None: out = np.zeros((len(W_IDX), F.shape[1]), np.float32)
        while wi < len(W_IDX) and W_IDX[wi][0] < s1:
            out[wi] = F[W_IDX[wi][2] - a].mean(0); wi += 1
        del F
    return out
def ridge_probe(X, task, lams=(1e-3, 1e-2, 1e-1, 1.0)):
    T = TASKS[task]; m_tr = IS_TR & T["keep"]; m_te = (~IS_TR) & T["keep"]
    mu = X[m_tr].mean(0); sd = X[m_tr].std(0) + 1e-6
    itr = np.flatnonzero(m_tr); cut = int(0.9 * len(itr)); fit, hold = itr[:cut], itr[cut:]; nc = T["nc"]; dd = X.shape[1] + 1
    def stats(rows):
        G = np.zeros((dd, dd)); XY = np.zeros((dd, nc))
        for c in range(0, len(rows), 20000):
            rr = rows[c:c + 20000]; Xs = np.c_[(X[rr] - mu) / sd, np.ones(len(rr), np.float32)].astype(np.float32)
            G += (Xs.T @ Xs).astype(np.float64); np.add.at(XY.T, T["y"][rr], Xs)
        return G, XY
    def predict(rows, W):
        return np.concatenate([(np.c_[(X[rows[c:c + 20000]] - mu) / sd, np.ones(len(rows[c:c + 20000]), np.float32)] @ W).argmax(1) for c in range(0, len(rows), 20000)])
    Gf, XYf = stats(fit); Gh, XYh = stats(hold)
    sol = lambda G, XY, lam: np.linalg.solve(G + lam * np.trace(G) / dd * np.eye(dd), XY).astype(np.float32)
    best = max(lams, key=lambda l: np.mean(predict(hold, sol(Gf, XYf, l)) == T["y"][hold]))
    W = sol(Gf + Gh, XYf + XYh, best); te = np.flatnonzero(m_te); ok = predict(te, W) == T["y"][te]; amb = T["amb"][te]
    return dict(all=float(ok.mean()), amb=float(ok[amb].mean()), n_amb=int(amb.sum()))
def probe(feats, name, log=None, X=None):
    X = word_matrix(feats) if X is None else X; res = {t: ridge_probe(X, t) for t in TASKS}
    line = (f"  {name:40s} dims {X.shape[1]:5d} | POS all {res['pos']['all']*100:5.1f} ambiguous {res['pos']['amb']*100:5.1f} | "
            f"supersense all {res['ss']['all']*100:5.1f} ambiguous {res['ss']['amb']*100:5.1f}")
    tick(line)
    if log: open(log, "a").write(line + "\n")
    return res, X
def baseline_line():
    p, s = TASKS["pos"], TASKS["ss"]
    def acc(T, amb): m = (~IS_TR) & T["keep"] & (T["amb"] if amb else True); return float((T["mfs"] == T["y"])[m].mean())
    return (f"  {'most-frequent-tag-per-word (no context)':40s}            | POS all {acc(p,0)*100:5.1f} ambiguous {acc(p,1)*100:5.1f} | "
            f"supersense all {acc(s,0)*100:5.1f} ambiguous {acc(s,1)*100:5.1f}   (n ambiguous: POS {int(((~IS_TR)&p['keep']&p['amb']).sum())}, SS {int(((~IS_TR)&s['keep']&s['amb']).sum())})")
