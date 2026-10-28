"""Which closed-form token-meaning recipe is best? word similarity + whitened-average STS for each."""
from hx import *
from evals import word_sim
from scipy import sparse as sp
from sklearn.utils.extmath import randomized_svd
LOG = "evariants.log"; open(LOG, "w")
ids = load_corpus().astype(np.int64); n = int(0.9 * len(ids)); freq = token_freq(ids); ref = make_ref(ids)
def cooc(W, weight):
    M = sp.csr_matrix((V, V), dtype=np.float32)
    for c in range(0, n, 5_000_000):
        seg = ids[c:min(n, c + 5_000_000 + W)]; r_, c_, v_ = [], [], []
        for d in range(1, W + 1):
            a, b = seg[:-d], seg[d:]; w = np.full(len(a), weight(d), np.float32); r_ += [a, b]; c_ += [b, a]; v_ += [w, w]
        M = M + sp.csr_matrix((np.concatenate(v_), (np.concatenate(r_), np.concatenate(c_))), shape=(V, V))
    return M
def ppmi(M, cds=0.75, shift=0.0):
    tot = M.sum(); rs = np.asarray(M.sum(1)).ravel(); cs = np.asarray(M.sum(0)).ravel() ** cds; cs = cs / cs.sum() * tot
    C = M.tocoo(); pm = np.log(C.data * tot / (rs[C.row] * cs[C.col] + 1e-12)) - shift; k = pm > 0
    return sp.csr_matrix((pm[k].astype(np.float32), (C.row[k], C.col[k])), shape=(V, V))
def score(name, E):
    def vec(w):
        t = tok.encode(w).ids; return E[t].mean(0) if t else None
    ws = word_sim(vec); rows, *_ = evaluate(name, lambda t, s: E[t], ref, freq, posts=("white",), weights={"mean": lambda t, s, F: np.ones(len(t), np.float32)})
    r = rows["mean+white"]; line = f"{name:44s} SimLex {ws['SimLex-999'][0]:.3f} WS353 {ws['WordSim-353'][0]:.3f} MEN {ws['MEN'][0]:.3f} RW {ws['RareWords'][0]:.3f} | STS-B {r['STS-B']*100:.1f} SICK {r['SICK']*100:.1f}"
    tick(line); open(LOG, "a").write(line + "\n")
for W, wname, wf in [(4, "1/d", lambda d: 1.0 / d), (2, "1/d", lambda d: 1.0 / d), (8, "1/d", lambda d: 1.0 / d)]:
    M = cooc(W, wf)
    for shift in (0.0, np.log(5)):
        P = ppmi(M, shift=shift); U, S, Vt = randomized_svd(P, 300, n_iter=5, random_state=0); del P
        for dim in (128, 300):
            for pw in (0.0, 0.5):
                E = nz(U[:, :dim] * S[:dim] ** pw).astype(np.float32); score(f"W={W} {wname} shift={shift:.1f} dim={dim} p={pw}", E)
            E = nz(U[:, :dim] * np.sqrt(S[:dim]) + Vt[:dim].T * np.sqrt(S[:dim])).astype(np.float32); score(f"W={W} {wname} shift={shift:.1f} dim={dim} W+C", E)
        gc_ = None
tick("done")
