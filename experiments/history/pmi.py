"""Directional association factors: PMI(token a, then token b within W tokens) ~= A[a] . B[b].
Counted over the whole corpus (closed form), PPMI with context-distribution smoothing, randomized SVD."""
import sys, numpy as np, time
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
W = int(sys.argv[1]) if len(sys.argv) > 1 else 8; K = 256; V = 16000
ids = np.load("/home/claude/big/tokens.npz")["ids"]; n = int(0.9 * len(ids)); t0 = time.time()
M = sparse.csr_matrix((V, V), dtype=np.float32)
for c in range(0, n, 4_000_000):
    seg = ids[c:min(n, c + 4_000_000 + W)].astype(np.int64)
    for d in range(1, W + 1):
        a, b = seg[:-d], seg[d:]; w = np.float32(1.0 / np.sqrt(d))
        M = M + sparse.csr_matrix((np.full(len(a), w, np.float32), (a, b)), shape=(V, V))
    print(f"{c/1e6:.0f}M {time.time()-t0:.0f}s", flush=True)
tot = M.sum(); rs = np.asarray(M.sum(1)).ravel(); cs = np.asarray(M.sum(0)).ravel()
rs75 = rs ** 0.75; rs75 = rs75 / rs75.sum() * tot; cs75 = cs ** 0.75; cs75 = cs75 / cs75.sum() * tot
M = M.tocoo(); pm = np.log(M.data * tot / (np.sqrt(rs[M.row] * rs75[M.row]) * np.sqrt(cs[M.col] * cs75[M.col]) + 1e-12)); k = pm > 0
P = sparse.csr_matrix((pm[k].astype(np.float32), (M.row[k], M.col[k])), shape=(V, V)); del M, pm
U, S, Vt = randomized_svd(P, K, n_iter=4, random_state=0)
np.savez(f"/home/claude/big/pmi_W{W}.npz", A=(U * np.sqrt(S)).astype(np.float32), B=(Vt.T * np.sqrt(S)).astype(np.float32), S=S, rs=rs, cs=cs)
print(f"done {time.time()-t0:.0f}s; top singular values {S[:5]}", flush=True)
