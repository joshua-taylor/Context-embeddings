"""Token meanings and directional PMI factors from ONE co-occurrence count over the corpus.

M[a, b] = sum over pairs where token a occurs d <= W tokens before token b of weight 1/d.
  meanings    : PPMI of the symmetric counts (M + M^T), context-distribution smoothing 0.75, truncated SVD,
                rows = U * S^0.5, L2-normalised.  768-d for output vectors; the first 128 dims for internal use.
  PMI factors : PPMI of the directional counts M, rank-256 SVD:  PMI(a before b) ~= A[a] . B[b]
                (used as the fixed relevance scores of the counting attention).
"""
import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from .utils import nz, tick


def cooccurrence(ids, V, W=8, chunk=4_000_000):
    ids = ids.astype(np.int64)
    M = sparse.csr_matrix((V, V), dtype=np.float32)
    for c in range(0, len(ids), chunk):
        seg = ids[c:c + chunk + W]
        rows, cols, vals = [], [], []
        for d in range(1, W + 1):
            rows.append(seg[:-d]); cols.append(seg[d:]); vals.append(np.full(len(seg) - d, 1.0 / d, np.float32))
        M = M + sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(V, V))
    return M


def _ppmi(M, cds=0.75):
    tot = M.sum()
    rs = np.asarray(M.sum(1)).ravel()
    cs = np.asarray(M.sum(0)).ravel() ** cds
    cs = cs / cs.sum() * tot
    C = M.tocoo()
    pm = np.log(C.data * tot / (rs[C.row] * cs[C.col] + 1e-12))
    k = pm > 0
    return sparse.csr_matrix((pm[k].astype(np.float32), (C.row[k], C.col[k])), shape=M.shape)


def meanings(M, dim=768, p=0.5, seed=0):
    """returns (E_out [V, dim], E_in [V, 128])"""
    P = _ppmi(M + M.T)
    U, S, _ = randomized_svd(P, dim, n_iter=5, random_state=seed)
    E_out = nz(U * S ** p).astype(np.float32)
    E_in = nz(U[:, :128] * S[:128] ** p).astype(np.float32)
    return E_out, E_in


def pmi_factors(M, rank=256, seed=0):
    P = _ppmi(M)
    U, S, Vt = randomized_svd(P, rank, n_iter=4, random_state=seed)
    return (U * np.sqrt(S)).astype(np.float32), (Vt.T * np.sqrt(S)).astype(np.float32)
