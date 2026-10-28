"""Round H: masked-slot model from sparse products (pattern lookups) + memories."""
from hx import *
from sparse import SparseCloze
from ctx import corpus_flags
from mlm2 import VocabRidge
import pickle, gc
LOG = "roundH.log"; open(LOG, "w")
ids = load_corpus().astype(np.int64); freq = token_freq(ids); ref = make_ref(ids); E = np.load(f"{BIG}/E.npy"); n = len(ids); hi = int(0.9 * n)
NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 600_000
t0 = time.time(); sc = SparseCloze(E, ids, DOT, hi=hi); tick(f"pattern tables built in {time.time()-t0:.0f}s: " + ", ".join(f"{g} {len(T.keys)/1e6:.2f}M" for g, T in sc.tables.items()))
st, en, sid = corpus_flags(ids, DOT); r = np.random.default_rng(0)
keep_p = np.minimum(1.0, np.sqrt(1e-4 / (freq + 1e-12)) + 1e-4 / (freq + 1e-12))
def sample(lo, hi_, m):
    c = r.integers(lo, hi_, int(m * 4)); c = c[r.random(len(c)) < keep_p[ids[c]]]; return np.sort(np.unique(c)[:m])
ptr = sample(2, hi - 2, NTR); pva = sample(hi + 2, n - 2, 10000)
rf = RidgeFit(); vr = VocabRidge(V)
for c in range(0, len(ptr), 50000):
    p = ptr[c:c + 50000]; F = sc.feats(ids, st, en, sid, p, target=ids[p]); rf.add(F, E[ids[p]]); vr.add(F, ids[p]); del F; gc.collect()
Fv = sc.feats(ids, st, en, sid, pva); cos = rf.solve(Fv, E[ids[pva]]); v = vr.solve(Fv, ids[pva], lams=(1e-4, 1e-3, 1e-2))
G = nz(rf(Fv)) @ E.T; rank = (G > G[np.arange(len(pva)), ids[pva]][:, None]).sum(1)
line = (f"sparse-product cloze ({NTR:,} slots, {Fv.shape[1]} features): meaning readout cos {cos:.3f} MRR {np.mean(1/(rank+1)):.3f} | "
        f"vocab readout MRR {v['mrr']:.3f} top1 {v['top1']:.3f} top10 {v['top10']:.3f}  (fit {time.time()-t0:.0f}s)")
tick(line); open(LOG, "a").write(line + "\n")
vr.XYt = None; vr.G = None; gc.collect()
LAST = {}
def guess(t, st_):
    F = sc.bank_feats(t, st_); Fs = vr.std(F)
    LAST["sc"] = np.concatenate([np.einsum("nd,dn->n", Fs[c:c + 20000], vr.W[:, t[c:c + 20000]]) for c in range(0, len(t), 20000)]); LAST["n"] = len(t)
    return rf(F)
def csif(a):
    def w(t, st_, F): p = np.clip(LAST["sc"], 1e-5, 1.0); return (a / (a + p)).astype(np.float32)
    return w
W3 = {"mean": lambda t, s, F: np.ones(len(t), np.float32), "sif": sif_w(freq), "ctx-sif": csif(0.01)}
evaluate("sparse guess", guess, ref, freq, posts=("white",), weights=W3, log=LOG)
evaluate("[x ; sparse guess]", lambda t, s: np.c_[E[t], nz(guess(t, s))], ref, freq, posts=("white",), weights=W3, log=LOG)
evaluate("static x, ctx-sif (sparse model)", lambda t, s: (guess(t, s), E[t])[1], ref, freq, posts=("white",), weights={"ctx-sif": csif(0.01)}, log=LOG)
pickle.dump(dict(rf=rf, vr=vr), open("roundH_readouts.pkl", "wb"))
tick("done")
