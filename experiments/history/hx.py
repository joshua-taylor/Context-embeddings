"""Experiment harness: one scorecard for every model variant.

A model supplies  feats(tok_ids, starts) -> per-token vectors  (sentences are encoded independently: every
sentence starts from a reset state, in training and at test time, so the two always match).
The harness pools token vectors into sentence vectors in several ways and scores:
  STS-B, SICK-R (Spearman), SICK entail>contradict (AUC), negation triplets (22), SemCor word-in-context (AUC/acc).
Pooling weights / post-processing are fitted on 20k ordinary corpus sentences (never on test data).
"""
import sys, json, time, numpy as np, numba as nb
sys.path.insert(0, "/home/claude/bench"); sys.path.insert(0, "/home/claude/big")
from tokenizers import Tokenizer
from scipy.stats import spearmanr
from evals import PAIRS, auc
from nlexp_neg import NEG
BIG = "/home/claude/big"
T0 = time.time()
def tick(m): print(f"[{time.time()-T0:6.0f}s] {m}  ({int(open('/proc/self/statm').read().split()[1])*4096/1e9:.1f} GB)", flush=True)
def nz(X): return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)

tok = Tokenizer.from_file(f"{BIG}/bpe.json"); V = tok.get_vocab_size(); DOT = tok.token_to_id(".")

def load_corpus():
    ids = np.load(f"{BIG}/tokens.npz")["ids"]; return ids

def sentence_bounds(ids, lo=0, hi=None, minlen=4, maxlen=64):
    """Corpus sentences = runs of tokens ending in '.', returned as (start, end) arrays."""
    hi = len(ids) if hi is None else hi; d = np.flatnonzero(ids[lo:hi] == DOT) + lo
    s = np.r_[lo, d[:-1] + 1]; e = d + 1; L = e - s; k = (L >= minlen) & (L <= maxlen); return s[k], e[k]

class Bank:
    """A set of sentences stored as one concatenated token array with sentence starts and word maps."""
    def __init__(self, sents=None, toks=None):
        if sents is not None:
            enc = tok.encode_batch([" ".join(s) for s in sents])
            self.tok = np.concatenate([np.array(e.ids, np.int32) for e in enc])
            self.wmap = np.concatenate([np.array([w if w is not None else -1 for w in e.word_ids], np.int32) for e in enc])
            L = np.array([len(e.ids) for e in enc])
        else:
            self.tok = np.concatenate(toks).astype(np.int32); L = np.array([len(t) for t in toks]); self.wmap = None
        self.starts = np.r_[0, np.cumsum(L)].astype(np.int64); self.n = len(L)
        self.sid = np.repeat(np.arange(self.n), L)

# ---------------- test sets ----------------
T = json.load(open(f"{BIG}/tests.json"))
_S = {}
def _add(ws): k = tuple(ws); return _S.setdefault(k, len(_S))
STS = [(_add(a), _add(b), s) for a, b, s in T["sts"]]
SICK = [(_add(a), _add(b), s, e) for a, b, s, e in T["sick"]]
NEGI = [tuple(_add(s.split()) for s in tr) for tr in NEG]
SEM = [((_add(p["a"][0]), p["a"][1]), (_add(p["b"][0]), p["b"][1])) for p in PAIRS]
CPAIRS = json.load(open("/home/claude/bench/data/coarse_pairs.json"))
CSEM = [((_add(p["a"][0]), p["a"][1]), (_add(p["b"][0]), p["b"][1])) for p in CPAIRS]
y_c = np.array([p["label"] for p in CPAIRS]); dev_c = np.array([p["split"] == "dev" for p in CPAIRS])
SENTS = [list(k) for k in _S]
TEST = Bank(SENTS)
NEED_SENT = np.zeros(TEST.n, bool)
for a, b, *_ in STS + SICK: NEED_SENT[a] = NEED_SENT[b] = True
for tr in NEGI: NEED_SENT[list(tr)] = True
y_sem = np.array([p["label"] for p in PAIRS]); dev_sem = np.array([p["split"] == "dev" for p in PAIRS])
# token positions of the SemCor target words (mean over the word's pieces)
_sem_keys = sorted(set([x for pr in SEM + CSEM for x in pr])); SEM_IX = {k: i for i, k in enumerate(_sem_keys)}
_tp = []
for (si, wi) in _sem_keys:
    lo, hi = TEST.starts[si], TEST.starts[si + 1]; m = np.flatnonzero(TEST.wmap[lo:hi] == wi) + lo
    if not len(m): m = np.array([hi - 1])
    _tp.append(m)
SEM_TOK = _tp

def token_freq(ids):
    c = np.bincount(ids, minlength=V).astype(np.float64); return c / c.sum()

def make_ref(ids, n=20000, seed=0, hi=None):
    s, e = sentence_bounds(ids, 0, hi or int(0.9 * len(ids))); r = np.random.default_rng(seed); k = np.sort(r.choice(len(s), n, replace=False))
    return Bank(toks=[ids[s[i]:e[i]] for i in k])

# ---------------- pooling ----------------
def pool(Vt, w, starts):
    W = Vt * w[:, None]; S = np.add.reduceat(W, starts[:-1], axis=0); Z = np.add.reduceat(w, starts[:-1]); return S / (Z[:, None] + 1e-9)

def run_bank(bank, feats, weightfn, chunk=800, need=None, extra_tok=None):
    """Encode a bank in sentence chunks; return pooled sentence vectors per weighting and SemCor word vectors."""
    out = {k: [] for k in weightfn}; semv = {}; idx = []
    for c in range(0, bank.n, chunk):
        s0, s1 = c, min(bank.n, c + chunk)
        if need is not None and not need[s0:s1].any() and extra_tok is None:
            for k in out: out[k].append(None)
            idx.append((s0, s1)); continue
        lo, hi = bank.starts[s0], bank.starts[s1]; t = bank.tok[lo:hi]; st = bank.starts[s0:s1 + 1] - lo
        F = feats(t, st)
        for k, wf in weightfn.items(): out[k].append(pool(F, wf(t, st, F), st).astype(np.float32))
        if extra_tok is not None:
            for j, m in enumerate(extra_tok):
                if lo <= m[0] < hi: semv[j] = F[m - lo].mean(0)
        idx.append((s0, s1)); del F
    d = next(x for x in out[next(iter(out))] if x is not None).shape[1]
    res = {}
    for k in out:
        A = np.zeros((bank.n, d), np.float32)
        for (s0, s1), x in zip(idx, out[k]):
            if x is not None: A[s0:s1] = x
        res[k] = A
    return res, semv

class Post:
    """Post-processing fitted on corpus sentence vectors: center, remove top-k common directions, optional whitening."""
    def __init__(self, R, k=1, white=False, center=True, wdim=None):
        self.m = R.mean(0) if center else np.zeros(R.shape[1], np.float32); X = R - self.m
        U, S, Vt = np.linalg.svd(X[:20000].astype(np.float64), full_matrices=False); self.white = white
        if white:
            dims = wdim or len(S); self.W = (Vt[:dims].T / (S[:dims] / np.sqrt(len(X)) + 1e-6 * S[0])).astype(np.float32)
        else:
            self.P = Vt[:k].astype(np.float32)
    def __call__(self, X):
        X = X - self.m
        if self.white: return X @ self.W
        return X - (X @ self.P.T) @ self.P if len(self.P) else X

def metrics(S, semv=None):
    S = nz(S); r = {}
    r["STS-B"] = float(spearmanr([S[a] @ S[b] for a, b, _ in STS], [s for *_, s in STS]).correlation)
    sk = np.array([S[a] @ S[b] for a, b, *_ in SICK]); r["SICK"] = float(spearmanr(sk, [x[2] for x in SICK]).correlation)
    lab = np.array([x[3] for x in SICK]); m = np.isin(lab, ["ENTAILMENT", "CONTRADICTION"]); r["ent>con"] = auc(sk[m], (lab[m] == "ENTAILMENT").astype(int))
    r["neg"] = float(np.mean([S[a] @ S[b] > S[a] @ S[c] for a, b, c in NEGI]))
    if semv is not None:
        Wv = nz(np.array([semv[SEM_IX[k]] for k in _sem_keys])); s = np.array([Wv[SEM_IX[a]] @ Wv[SEM_IX[b]] for a, b in SEM])
        th = np.unique(np.quantile(s[dev_sem], np.linspace(.01, .99, 197))); t_ = th[np.argmax([np.mean((s[dev_sem] > x) == y_sem[dev_sem]) for x in th])]
        r["wic_acc"] = float(np.mean((s[~dev_sem] > t_) == y_sem[~dev_sem])); r["wic_auc"] = auc(s[~dev_sem], y_sem[~dev_sem])
        s = np.array([Wv[SEM_IX[a]] @ Wv[SEM_IX[b]] for a, b in CSEM]); r["coarse_auc"] = auc(s[~dev_c], y_c[~dev_c])
    return r

def fmt(r): return "  ".join(f"{k} {v*100:5.1f}" if k != "neg" else f"neg {int(round(v*22)):2d}/22" for k, v in r.items())

def sif_w(freq, a=1e-3):
    tw = (a / (a + freq)).astype(np.float32); return lambda t, st, F: tw[t]

def evaluate(name, feats, ref, freq, posts=("raw", "pc1", "white"), weights=None, log=None, a=1e-3):
    """Full scorecard for one model across poolings x post-processing."""
    weights = weights or {"mean": lambda t, st, F: np.ones(len(t), np.float32), "sif": sif_w(freq, a)}
    te, semv = run_bank(TEST, feats, weights, extra_tok=SEM_TOK)
    rf, _ = run_bank(ref, feats, weights)
    rows = {}
    for wk in weights:
        for pk in posts:
            if pk == "raw": f = lambda X: X
            elif pk == "pc1": f = Post(rf[wk], k=1)
            elif pk == "pc0": f = Post(rf[wk], k=0)
            elif pk == "white": f = Post(rf[wk], white=True)
            r = metrics(f(te[wk]), semv if (wk, pk) == (list(weights)[0], posts[0]) else None); rows[f"{wk}+{pk}"] = r
            line = f"  {name:34s} {wk+'+'+pk:13s} {fmt(r)}"; print(line, flush=True)
            if log: open(log, "a").write(line + "\n")
    return rows, te, rf, semv

# ---------------- closed-form readout ----------------
class RidgeFit:
    """Accumulate X'X and X'Y chunk by chunk (float64), standardise features, pick lambda on held-out cosine."""
    def __init__(self): self.G = None; self.n = 0
    def add(self, X, Y):
        X = X.astype(np.float32)
        if self.G is None:
            self.mu = X.mean(0); self.sd = X.std(0) + 1e-6; d = X.shape[1]
            self.G = np.zeros((d + 1, d + 1)); self.XY = np.zeros((d + 1, Y.shape[1]))
        Xs = np.c_[(X - self.mu) / self.sd, np.ones(len(X), np.float32)]
        self.G += Xs.T.astype(np.float64) @ Xs; self.XY += Xs.T.astype(np.float64) @ Y; self.n += len(X)
    def solve(self, Xv, Yv, lams=(1e-4, 1e-3, 1e-2, 1e-1)):
        d = len(self.G); best = None; Xs = np.c_[(Xv - self.mu) / self.sd, np.ones(len(Xv), np.float32)]
        for lam in lams:
            W = np.linalg.solve(self.G + lam * np.trace(self.G) / d * np.eye(d), self.XY).astype(np.float32)
            c = float(np.mean(np.sum(nz(Xs @ W) * nz(Yv), 1)))
            if best is None or c > best[0]: best = (c, W, lam)
        self.cos, self.W, self.lam = best
        # fold standardisation into one matrix + bias
        self.Wf = (self.W[:-1] / self.sd[:, None]).astype(np.float32); self.bf = (self.W[-1] - (self.mu / self.sd) @ self.W[:-1]).astype(np.float32)
        return best[0]
    def __call__(self, X): return X @ self.Wf + self.bf

# ---------------- sentence-level closed-form objective ----------------
def neighbour_bank(ids, n=60000, seed=1, hi=None):
    """Pairs of consecutive corpus sentences (k, k+1), stored interleaved: bank sentence 2i = k, 2i+1 = k+1."""
    s, e = sentence_bounds(ids, 0, hi or int(0.9 * len(ids))); adj = np.flatnonzero(e[:-1] == s[1:])
    r = np.random.default_rng(seed); k = np.sort(r.choice(adj, n, replace=False))
    toks = []
    for i in k: toks += [ids[s[i]:e[i]], ids[s[i + 1]:e[i + 1]]]
    return Bank(toks=toks)

class NCCA:
    """Symmetric CCA between a sentence and its neighbour. Keeps directions along which neighbouring sentences agree,
    weighted by canonical correlation ** p. (Whitening = every direction weighted 1; this learns which matter.)"""
    def __init__(self, P, p=1.0, k=None, eps=1e-4):
        X, Y = P[0::2].astype(np.float64), P[1::2].astype(np.float64); self.m = np.r_[X, Y].mean(0); X = X - self.m; Y = Y - self.m
        C = (X.T @ X + Y.T @ Y) / (2 * len(X)); Cxy = (X.T @ Y + Y.T @ X) / (2 * len(X))
        e, U = np.linalg.eigh(C); Wh = U / np.sqrt(e + eps * e.max())
        rho, Q = np.linalg.eigh(Wh.T @ Cxy @ Wh); o = np.argsort(-rho); rho, Q = rho[o], Q[:, o]
        keep = rho > 0 if k is None else np.arange(len(rho)) < k
        self.rho = rho; self.W = (Wh @ Q[:, keep] * np.maximum(rho[keep], 1e-6) ** p).astype(np.float32)
    def __call__(self, X): return (X - self.m.astype(np.float32)) @ self.W

def wic_eval(feats, name="", log=None):
    """Only the word-in-context scores (fine SemCor + coarse), encoding just the sentences that hold target words."""
    need = np.zeros(TEST.n, bool); need[[k[0] for k in _sem_keys]] = True
    _, semv = run_bank(TEST, feats, {"m": lambda t, st, F: np.ones(len(t), np.float32)}, need=need, extra_tok=SEM_TOK)
    Wv = nz(np.array([semv[SEM_IX[k]] for k in _sem_keys]))
    return wic_scores(lambda k: Wv[SEM_IX[k]], name, log)

def macro_auc(s, y, lem):
    out = []
    for l in np.unique(lem):
        m = lem == l
        if 0 < y[m].sum() < m.sum(): out.append(auc(s[m], y[m]))
    return float(np.mean(out))
LEM_F = np.array([p["lemma"] for p in PAIRS]); LEM_C = np.array([p["lemma"] for p in CPAIRS])
def wic_scores(vec, name="", log=None):
    """pooled AUC (all test pairs together) and per-lemma AUC averaged over lemmas (removes lemma-level offsets)."""
    s = np.array([vec(a) @ vec(b) for a, b in SEM]); f = auc(s[~dev_sem], y_sem[~dev_sem]); fm = macro_auc(s[~dev_sem], y_sem[~dev_sem], LEM_F[~dev_sem])
    s = np.array([vec(a) @ vec(b) for a, b in CSEM]); c = auc(s[~dev_c], y_c[~dev_c]); cm = macro_auc(s[~dev_c], y_c[~dev_c], LEM_C[~dev_c])
    line = f"  {name:40s} fine WiC AUC {f*100:5.1f} (per-lemma {fm*100:5.1f})   coarse WiC AUC {c*100:5.1f} (per-lemma {cm*100:5.1f})"; print(line, flush=True)
    if log: open(log, "a").write(line + "\n")
    return f, fm, c, cm
