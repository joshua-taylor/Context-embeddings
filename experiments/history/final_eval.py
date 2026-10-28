"""Final candidate: SPARC context model + higher-dimensional token meanings for the pooled sentence vector."""
from hx import *
import pickle, gc
from scipy import sparse as sp
from sklearn.utils.extmath import randomized_svd
LOG = "final.log"; open(LOG, "w")
ids = load_corpus().astype(np.int64); n = int(0.9 * len(ids)); freq = token_freq(ids); ref = make_ref(ids)
def meanings(W, dims, p=0.5):
    M = sp.csr_matrix((V, V), dtype=np.float32)
    for c in range(0, n, 5_000_000):
        seg = ids[c:min(n, c + 5_000_000 + W)]; r_, c_, v_ = [], [], []
        for d in range(1, W + 1):
            a, b = seg[:-d], seg[d:]; w = np.full(len(a), 1.0 / d, np.float32); r_ += [a, b]; c_ += [b, a]; v_ += [w, w]
        M = M + sp.csr_matrix((np.concatenate(v_), (np.concatenate(r_), np.concatenate(c_))), shape=(V, V))
    tot = M.sum(); rs = np.asarray(M.sum(1)).ravel(); cs = np.asarray(M.sum(0)).ravel() ** 0.75; cs = cs / cs.sum() * tot
    C = M.tocoo(); pm = np.log(C.data * tot / (rs[C.row] * cs[C.col] + 1e-12)); k = pm > 0
    P = sp.csr_matrix((pm[k].astype(np.float32), (C.row[k], C.col[k])), shape=(V, V)); del M, C, pm
    U, S, _ = randomized_svd(P, max(dims), n_iter=5, random_state=0)
    return {d: nz(U[:, :d] * S[:d] ** p).astype(np.float32) for d in dims}
t0 = time.time(); EB = meanings(8, (300, 500, 768)); tick(f"meanings (W=8, 768-d SVD) {time.time()-t0:.0f}s"); del ids; gc.collect()
m = pickle.load(open("sparc_f1.pkl", "rb"))
W = {"mean": lambda t, s, F: np.ones(len(t), np.float32), "ctx-sif": lambda t, s, F: (0.01 / (0.01 + np.clip(m.last_score, 1e-5, 1.0))).astype(np.float32)}
for d, Eb in EB.items():
    def xonly(t, s, Eb=Eb): m.token_vectors(t, s); return Eb[t]
    def full(t, s, Eb=Eb): X = m.token_vectors(t, s); return np.c_[Eb[t], X[:, 128:]]
    evaluate(f"meanings {d}d only", xonly, ref, freq, posts=("white",), weights=W, log=LOG)
    evaluate(f"SPARC: [meanings {d}d ; gap guess]", full, ref, freq, posts=("white",), weights=W, log=LOG)
np.save("E300.npy", EB[300]); np.save("E768.npy", EB[768])
tick("done")
