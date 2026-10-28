"""Round O: a LEARNED contextual layer in closed form.
Input at every (visible) token: [meaning ; word x context products], products = RandomBilinear(meaning, [gap guess ; attention ; guess-query attention]).
Self-supervised targets: the NEXT token (full 16k-vocabulary readout) and the meanings of the previous / next tokens.
The layer's output = the features projected onto the top-256 directions the readouts use (a learned 256-d 'hidden state').
Then: probe it (POS / supersense on ambiguous words) and pool it (STS / SICK / negation)."""
from hx import *
import pickle, gc
from scipy.linalg import cho_factor, cho_solve
from zoo import Zoo
from att import PMIAttn, query_attend
from nl import RandomBilinear
LOG = "roundO.log"; open(LOG, "w")
E = np.load(f"{BIG}/E.npy"); E768 = np.load(f"{BIG}/E768.npy"); d = 128; sc = pickle.load(open("tables128.pkl", "rb")); z = Zoo(E)
Zs = np.load("roundJ_stats.npz"); ORDER = ["P", "LIN", "SEL", "HIP", "ESN", "LINX"]; dims = Zs["dims"]; off = np.cumsum(np.r_[0, dims])
idx = {k: np.arange(off[i], off[i + 1]) for i, k in enumerate(ORDER)}; mu0, sd0 = Zs["mu"], Zs["sd"]; D0 = Zs["G"].shape[0] - 1
S1 = ["P", "LIN", "SEL", "ESN"]; cols1 = np.r_[np.concatenate([idx[k] for k in S1]), D0]; Gs = Zs["G"][np.ix_(cols1, cols1)]
W1 = cho_solve(cho_factor(Gs + 1e-3 * np.trace(Gs) / len(cols1) * np.eye(len(cols1))), Zs["XYm"][cols1]).astype(np.float32); del Zs, Gs; gc.collect()
ids = load_corpus().astype(np.int64); n = len(ids); hi = int(0.9 * n); freq = token_freq(ids)
Zp = np.load(f"{BIG}/pmi_W8.npz"); KA, KB = np.ascontiguousarray(Zp["A"]), np.ascontiguousarray(Zp["B"]); w = np.sqrt(freq + 1e-9)[:, None]
MA = np.linalg.solve((E * w).T @ (E * w) + 1e-3 * np.eye(d), (E * w).T @ (KA * w)).astype(np.float32)
MB = np.linalg.solve((E * w).T @ (E * w) + 1e-3 * np.eye(d), (E * w).T @ (KB * w)).astype(np.float32)
at = PMIAttn(KA, KB, E, tau=0.4, sink=1.0); xp4 = RandomBilinear(d, 3 * d, 3072, 32)
def layer_in(t, st, loo):
    sid = np.repeat(np.arange(len(st) - 1), np.diff(st)); pos = np.arange(len(t)); Pm = []
    for g, T in sc.tables.items(): M, lc = T.lookup(t, sid, pos, t if loo else None, E); Pm += [M, lc]
    B = {"P": np.concatenate(Pm, 1)}
    for k in ["LIN", "SEL", "ESN"]: B[k] = z.run(k, t, st)
    c1 = cols1[:-1]; X = np.concatenate([B[k] for k in S1], 1); g = nz(((X - mu0[c1]) / sd0[c1]) @ W1[:-1] + W1[-1]); del X, B
    A1 = nz(at(t, st)); o = np.empty((len(t), d), np.float32)
    query_attend(t, st, (g @ MA).astype(np.float32), (g @ MB).astype(np.float32), KA, KB, E, np.float32(0.2), np.float32(0.0), o)
    return np.c_[E[t], nz(xp4(E[t], np.c_[g, A1, nz(o)])) * 8].astype(np.float32), g
# ---- fit the layer on corpus sentences
r = np.random.default_rng(5); s_, e_ = sentence_bounds(ids, 0, hi); k = np.sort(r.choice(len(s_), 30000, replace=False))
bank = Bank(toks=[ids[s_[i]:e_[i]] for i in k]); del ids; gc.collect()
G = None; t0 = time.time(); NP = 0
for c in range(0, bank.n, 500):
    lo, hi_ = bank.starts[c], bank.starts[min(bank.n, c + 500)]; t = bank.tok[lo:hi_].astype(np.int64); st = (bank.starts[c:min(bank.n, c + 500) + 1] - lo).astype(np.int64)
    F, _ = layer_in(t, st, True); L = np.diff(st); sid = np.repeat(np.arange(len(L)), L)
    nxt = np.r_[t[1:], -1]; nxt[st[1:] - 1] = -1; prv = np.r_[-1, t[:-1]]; prv[st[:-1]] = -1; m = nxt >= 0
    F, y, Ym = F[m], nxt[m], np.c_[E[nxt[m]], np.where((prv[m] >= 0)[:, None], E[np.maximum(prv[m], 0)], 0)]
    if G is None:
        D = F.shape[1]; mu = F.mean(0); sd = F.std(0) + 1e-6; G = np.zeros((D + 1, D + 1)); XYm = np.zeros((D + 1, Ym.shape[1])); XYv = np.zeros((V, D + 1))
    Xs = np.c_[(F - mu) / sd, np.ones(len(F), np.float32)].astype(np.float32); del F
    G += (Xs.T @ Xs).astype(np.float64); XYm += (Xs.T @ Ym).astype(np.float64)
    o = np.argsort(y, kind="stable"); u, s0 = np.unique(y[o], return_index=True); XYv[u] += np.add.reduceat(Xs[o], s0, 0); NP += len(y); del Xs; gc.collect()
tick(f"layer statistics over {NP:,} tokens ({time.time()-t0:.0f}s)")
lam = 1e-3 * np.trace(G) / len(G); cf = cho_factor(G + lam * np.eye(len(G)))
Wv = cho_solve(cf, XYv.T).astype(np.float32); Wm = cho_solve(cf, XYm).astype(np.float32); del XYv, cf; gc.collect()
Wall = np.c_[Wv[:-1] / np.linalg.norm(Wv[:-1]), Wm[:-1] / np.linalg.norm(Wm[:-1])]          # both readouts, equal weight
U, S, _ = np.linalg.svd(Wall.astype(np.float64), full_matrices=False); R = 256; Pr = (U[:, :R] * S[:R]).astype(np.float32); del Wv, Wall, U; gc.collect()
tick(f"learned layer: top-{R} readout directions keep {np.sum(S[:R]**2)/np.sum(S**2)*100:.0f}% of readout energy")
def hidden(t, st):
    F, g = layer_in(t, st, False); return ((F - mu) / sd) @ Pr, g
# ---- probe
import probe as PR
WIDX = PR.W_IDX; out = None; wi = 0
for c in range(0, PR.BANK.n, 600):
    s0, s1 = c, min(PR.BANK.n, c + 600); a, b = PR.BANK.starts[s0], PR.BANK.starts[s1]
    H, g = hidden(PR.BANK.tok[a:b].astype(np.int64), (PR.BANK.starts[s0:s1 + 1] - a).astype(np.int64)); Fc = np.c_[E768[PR.BANK.tok[a:b]], H, g]
    if out is None: out = np.zeros((len(WIDX), Fc.shape[1]), np.float32)
    while wi < len(WIDX) and WIDX[wi][0] < s1: out[wi] = Fc[WIDX[wi][2] - a].mean(0); wi += 1
open(LOG, "a").write(PR.baseline_line() + "\n"); print(PR.baseline_line())
PR.probe(None, "E768 + learned layer (256)", LOG, X=out[:, :768 + R])
PR.probe(None, "E768 + learned layer + gap guess", LOG, X=out)
PR.probe(None, "learned layer only (256)", LOG, X=out[:, 768:768 + R])
del out; gc.collect()
# ---- sentences
ref = make_ref(load_corpus(), n=10000)
def sent_feats(t, st):
    H, g = hidden(t.astype(np.int64), st.astype(np.int64)); return np.c_[E768[t], g, nz(H)].astype(np.float32)
Wt = {"mean": lambda t, s, F: np.ones(len(t), np.float32)}
te, _ = run_bank(TEST, sent_feats, Wt, need=NEED_SENT); rf, _ = run_bank(ref, sent_feats, Wt)
for nm, sl in {"meaning+guess": slice(0, 896), "meaning+guess+learned layer": slice(0, 896 + R), "learned layer only": slice(896, 896 + R)}.items():
    f = Post(rf["mean"][:, sl], white=True); rr = metrics(f(te["mean"][:, sl])); line = f"  {nm:34s} mean+white    {fmt(rr)}"; print(line, flush=True); open(LOG, "a").write(line + "\n")
tick("done")
