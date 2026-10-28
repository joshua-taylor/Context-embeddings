"""Round N: do the word x context products help SENTENCE vectors (STS, SICK, negation, entail/contradict)?"""
from hx import *
import pickle, gc
from scipy.linalg import cho_factor, cho_solve
from zoo import Zoo
from att import PMIAttn, query_attend
from nl import RandomBilinear
LOG = "roundN.log"; open(LOG, "w")
E = np.load(f"{BIG}/E.npy"); E768 = np.load(f"{BIG}/E768.npy"); d = 128; sc = pickle.load(open("tables128.pkl", "rb")); z = Zoo(E)
Zs = np.load("roundJ_stats.npz"); ORDER = ["P", "LIN", "SEL", "HIP", "ESN", "LINX"]; dims = Zs["dims"]; off = np.cumsum(np.r_[0, dims])
idx = {k: np.arange(off[i], off[i + 1]) for i, k in enumerate(ORDER)}; mu0, sd0 = Zs["mu"], Zs["sd"]; D0 = Zs["G"].shape[0] - 1
S1 = ["P", "LIN", "SEL", "ESN"]; cols1 = np.r_[np.concatenate([idx[k] for k in S1]), D0]; Gs = Zs["G"][np.ix_(cols1, cols1)]
W1 = cho_solve(cho_factor(Gs + 1e-3 * np.trace(Gs) / len(cols1) * np.eye(len(cols1))), Zs["XYm"][cols1]).astype(np.float32); del Zs, Gs; gc.collect()
ids = load_corpus(); freq = token_freq(ids); ref = make_ref(ids, n=10000); del ids
Zp = np.load(f"{BIG}/pmi_W8.npz"); KA, KB = np.ascontiguousarray(Zp["A"]), np.ascontiguousarray(Zp["B"]); w = np.sqrt(freq + 1e-9)[:, None]
MA = np.linalg.solve((E * w).T @ (E * w) + 1e-3 * np.eye(d), (E * w).T @ (KA * w)).astype(np.float32)
MB = np.linalg.solve((E * w).T @ (E * w) + 1e-3 * np.eye(d), (E * w).T @ (KB * w)).astype(np.float32)
at = PMIAttn(KA, KB, E, tau=0.4, sink=1.0); xp4 = RandomBilinear(d, 3 * d, 3072, 32)
def feats(t, st):
    t = t.astype(np.int64); st = st.astype(np.int64); sid = np.repeat(np.arange(len(st) - 1), np.diff(st)); pos = np.arange(len(t)); Pm = []
    for g, T in sc.tables.items(): M, lc = T.lookup(t, sid, pos, None, E); Pm += [M, lc]
    B = {"P": np.concatenate(Pm, 1)}
    for k in ["LIN", "SEL", "ESN"]: B[k] = z.run(k, t, st)
    c1 = cols1[:-1]; X = np.concatenate([B[k] for k in S1], 1); g = nz(((X - mu0[c1]) / sd0[c1]) @ W1[:-1] + W1[-1]); del X, B
    A1 = nz(at(t, st)); o = np.empty((len(t), d), np.float32)
    query_attend(t, st, (g @ MA).astype(np.float32), (g @ MB).astype(np.float32), KA, KB, E, np.float32(0.2), np.float32(0.0), o)
    XP = xp4(E[t], np.c_[g, A1, nz(o)])
    return np.c_[E768[t], g, nz(XP) * 2].astype(np.float32)
W = {"mean": lambda t, s, F: np.ones(len(t), np.float32)}
te, _ = run_bank(TEST, feats, W, need=NEED_SENT); rf, _ = run_bank(ref, feats, W); del sc; gc.collect()
for nm, sl in {"meaning+guess": slice(0, 896), "meaning+guess+products": slice(0, 896 + 3072), "products only": slice(896, 896 + 3072)}.items():
    f = Post(rf["mean"][:, sl], white=True); r = metrics(f(te["mean"][:, sl]))
    line = f"  {nm:34s} mean+white    {fmt(r)}"; print(line, flush=True); open(LOG, "a").write(line + "\n")
tick("done")
