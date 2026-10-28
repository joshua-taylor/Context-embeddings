"""Round I: combined candidate.
cloze features = sparse products (pattern lookups) + memories + leave-one-out sentence topic (mean meaning of the
other tokens in the sentence) -> ridge to the hidden token's meaning (guess) and to the vocabulary (predictability).
token vector = [meaning ; guess ; PMI-attention context]; sentence = contextual-SIF weighted mean, whitened."""
from hx import *
from sparse import SparseCloze
from ctx import corpus_flags
from mlm2 import VocabRidge
from att import PMIAttn
import pickle, gc
LOG = "roundI.log"; open(LOG, "w")
ids = load_corpus().astype(np.int64); freq = token_freq(ids); ref = make_ref(ids); E = np.load(f"{BIG}/E.npy"); n = len(ids); hi = int(0.9 * n)
NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 600_000
sc = SparseCloze(E, ids, DOT, hi=hi); st, en, sid = corpus_flags(ids, DOT); tick("tables")
# sentence sums for leave-one-out topic
SS = np.zeros((sid[-1] + 1, E.shape[1]), np.float32); SL = np.bincount(sid).astype(np.float32)
for c in range(0, n, 2_000_000): np.add.at(SS, sid[c:c + 2_000_000], E[ids[c:c + 2_000_000]])
def topic_corpus(p): return nz((SS[sid[p]] - E[ids[p]]) / np.maximum(SL[sid[p]] - 1, 1)[:, None])
def topic_bank(t, starts):
    S = np.add.reduceat(E[t], starts[:-1], axis=0); L = np.diff(starts); s_ = np.repeat(np.arange(len(L)), L)
    return nz((S[s_] - E[t]) / np.maximum(L[s_] - 1, 1)[:, None])
r = np.random.default_rng(0); keep_p = np.minimum(1.0, np.sqrt(1e-4 / (freq + 1e-12)) + 1e-4 / (freq + 1e-12))
def sample(lo, hi_, m):
    c = r.integers(lo, hi_, int(m * 4)); c = c[r.random(len(c)) < keep_p[ids[c]]]; return np.sort(np.unique(c)[:m])
ptr = sample(2, hi - 2, NTR); pva = sample(hi + 2, n - 2, 10000)
rf = RidgeFit(); vr = VocabRidge(V)
for c in range(0, len(ptr), 50000):
    p = ptr[c:c + 50000]; F = np.c_[sc.feats(ids, st, en, sid, p, target=ids[p]), topic_corpus(p)]; rf.add(F, E[ids[p]]); vr.add(F, ids[p]); del F; gc.collect()
Fv = np.c_[sc.feats(ids, st, en, sid, pva), topic_corpus(pva)]; cos = rf.solve(Fv, E[ids[pva]]); v = vr.solve(Fv, ids[pva], lams=(1e-4, 1e-3, 1e-2))
G = nz(rf(Fv)) @ E.T; rank = (G > G[np.arange(len(pva)), ids[pva]][:, None]).sum(1)
line = f"cloze + topic ({NTR:,} slots): meaning cos {cos:.3f} MRR {np.mean(1/(rank+1)):.3f} | vocab MRR {v['mrr']:.3f} top1 {v['top1']:.3f} top10 {v['top10']:.3f}"
tick(line); open(LOG, "a").write(line + "\n")
vr.XYt = None; vr.G = None; del SS, sid, st, en, ids; gc.collect()
pickle.dump(dict(rf=rf, vr=vr), open("roundI_readouts.pkl", "wb"))
Z = np.load(f"{BIG}/pmi_W8.npz"); at = PMIAttn(Z["A"], Z["B"], E, tau=0.4, sink=1.0)
LAST = {}
def enc(t, s):
    F = np.c_[sc.bank_feats(t, s), topic_bank(t, s)]; Fs = vr.std(F)
    LAST["sc"] = np.concatenate([np.einsum("nd,dn->n", Fs[c:c + 20000], vr.W[:, t[c:c + 20000]]) for c in range(0, len(t), 20000)])
    return E[t], nz(rf(F)), nz(at(t, s))
def csif(a):
    return lambda t, s, F: (a / (a + np.clip(LAST["sc"], 1e-5, 1.0))).astype(np.float32)
W = {"mean": lambda t, s, F: np.ones(len(t), np.float32), "ctx-sif .01": csif(0.01), "ctx-sif .03": csif(0.03), "ctx-sif .1": csif(0.1)}
for nm, f in [("[x ; guess]", lambda t, s: np.c_[enc(t, s)[:2]]), ("[x ; guess ; attention]", lambda t, s: np.c_[enc(t, s)]), ("x only", lambda t, s: enc(t, s)[0])]:
    evaluate(nm, f, ref, freq, posts=("white",), weights=W, log=LOG)
wic_eval(lambda t, s: np.c_[enc(t, s)], "[x ; guess ; attention]", LOG)
wic_eval(lambda t, s: enc(t, s)[1], "guess only (with topic)", LOG)
tick("done")
