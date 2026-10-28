"""Round D: neighbour-sentence CCA as the closed-form sentence-level objective, on static token meanings."""
from hx import *
LOG = "roundD.log"; open(LOG, "w")
ids = load_corpus(); freq = token_freq(ids); ref = make_ref(ids); nb_ = neighbour_bank(ids); del ids
E = np.load(f"{BIG}/E.npy"); tick("loaded")
W2 = {"mean": lambda t, st, F: np.ones(len(t), np.float32), "sif": sif_w(freq)}
te, semv = run_bank(TEST, lambda t, st: E[t], W2, extra_tok=SEM_TOK); pr, _ = run_bank(nb_, lambda t, st: E[t], W2)
for wk in W2:
    for p in (0.0, 0.5, 1.0, 2.0):
        for k in (None, 64):
            f = NCCA(pr[wk], p=p, k=k); line = f"  static E NCCA p={p} k={k}  {wk:5s} {fmt(metrics(f(te[wk])))}"; print(line, flush=True); open(LOG, "a").write(line + "\n")
print("top canonical correlations", np.round(f.rho[:10], 3), "n>0.1:", int((f.rho > 0.1).sum()))
tick("done")
