"""Train SPARC from scratch on the first FRAC of the training stream; evaluate; log stage timings.
usage: python train_sparc.py FRAC [NSLOTS]
All stages are closed form: counting (meanings, pattern tables), one SVD, two ridge solves."""
from hx import *
import pickle, gc
from scipy import sparse as sp
from sklearn.utils.extmath import randomized_svd
from sparse import SparseCloze
from ctx import corpus_flags
from mlm2 import VocabRidge
from sparc2 import SPARC2
FRAC = float(sys.argv[1]); NTR = int(sys.argv[2]) if len(sys.argv) > 2 else 600_000; TAG = f"f{FRAC:g}"
LOG = "scaling.log"; TM = {}
ids = load_corpus().astype(np.int64); n = len(ids); hold = int(0.9 * n); ntok = int(FRAC * hold)
def ppmi_svd(seg_ids, dim=128, W=4):
    M = sp.csr_matrix((V, V), dtype=np.float32)
    for c in range(0, len(seg_ids), 5_000_000):
        seg = seg_ids[c:c + 5_000_000 + W]; r_, c_, v_ = [], [], []
        for d in range(1, W + 1):
            a, b = seg[:-d], seg[d:]; w = np.full(len(a), 1.0 / d, np.float32); r_ += [a, b]; c_ += [b, a]; v_ += [w, w]
        M = M + sp.csr_matrix((np.concatenate(v_), (np.concatenate(r_), np.concatenate(c_))), shape=(V, V))
    tot = M.sum(); rs = np.asarray(M.sum(1)).ravel(); cs = np.asarray(M.sum(0)).ravel() ** 0.75; cs = cs / cs.sum() * tot
    M = M.tocoo(); pm = np.log(M.data * tot / (rs[M.row] * cs[M.col] + 1e-12)); k = pm > 0
    U, S, _ = randomized_svd(sp.csr_matrix((pm[k], (M.row[k], M.col[k])), shape=(V, V)), dim, n_iter=5, random_state=0)
    return nz(U * np.sqrt(S)).astype(np.float32)
t0 = time.time(); E = ppmi_svd(ids[:ntok]); TM["meanings"] = time.time() - t0; tick(f"[{TAG}] meanings over {ntok:,} tokens: {TM['meanings']:.0f}s")
t0 = time.time(); sc = SparseCloze(E, ids, DOT, hi=ntok); TM["pattern tables"] = time.time() - t0
nkeys = sum(len(T.keys) for T in sc.tables.values()); tick(f"[{TAG}] pattern tables: {TM['pattern tables']:.0f}s, {nkeys/1e6:.2f}M pattern values")
st, en, sid = corpus_flags(ids, DOT); freq = token_freq(ids[:ntok]); r = np.random.default_rng(0)
keep_p = np.minimum(1.0, np.sqrt(1e-4 / (freq + 1e-12)) + 1e-4 / (freq + 1e-12))
def sample(lo, hi_, m):
    c = r.integers(lo, hi_, int(m * 4)); c = c[r.random(len(c)) < keep_p[ids[c]]]; return np.sort(np.unique(c)[:m])
ptr = sample(2, ntok - 2, NTR); pva = sample(hold + 2, n - 2, 10000)
t0 = time.time(); rf = RidgeFit(); vr = VocabRidge(V)
for c in range(0, len(ptr), 50000):
    p = ptr[c:c + 50000]; F = sc.feats(ids, st, en, sid, p, target=ids[p]); rf.add(F, E[ids[p]]); vr.add(F, ids[p]); del F; gc.collect()
Fv = sc.feats(ids, st, en, sid, pva); cos = rf.solve(Fv, E[ids[pva]]); v = vr.solve(Fv, ids[pva], lams=(1e-4, 1e-3, 1e-2)); TM["readouts"] = time.time() - t0
G = nz(rf(Fv)) @ E.T; rank = (G > G[np.arange(len(pva)), ids[pva]][:, None]).sum(1); del Fv, G
vr.XYt = None; vr.G = None; ref = make_ref(ids, hi=ntok); del st, en, sid, ids; gc.collect()
m = SPARC2(E, sc, rf, vr); del sc; gc.collect()
t_, s_ = TEST.tok, TEST.starts; m.encode_tokens(t_[:3000].astype(np.int64), s_[:np.searchsorted(s_, 3000, "right")])
t0 = time.time(); m.encode_tokens(t_, s_); TM["encode tok/s"] = len(t_) / (time.time() - t0)
W = {"ctx-sif": lambda t, s, F: (0.01 / (0.01 + np.clip(m.last_score, 1e-5, 1.0))).astype(np.float32)}
rows, *_ = evaluate(f"SPARC {TAG}", m.token_vectors, ref, freq, posts=("white",), weights=W)
rows0, *_ = evaluate(f"static meanings {TAG}", lambda t, s: E[t], ref, freq, posts=("white",), weights={"mean": lambda t, s, F: np.ones(len(t), np.float32)})
res = dict(frac=FRAC, tokens=ntok, slots=NTR, pattern_values=nkeys, times=TM, cloze=dict(meaning_cos=cos, meaning_mrr=float(np.mean(1 / (rank + 1))), **v),
           sparc=rows["ctx-sif+white"], static=rows0["mean+white"])
line = json.dumps(res); open(LOG, "a").write(line + "\n"); print(line)
pickle.dump(m, open(f"sparc_{TAG}.pkl", "wb")); tick("done")
