"""Round C: learned composition from distributional context (closed form).

For every adjacent token pair (u = x_{t-1}, v = x_t) in the corpus, learn by ridge regression
    f(u, v) = Wu u + Wv v + Wp [(u Pa) * (v Pb)]          (random bilinear products -> pair interactions)
to predict the meaning of the SURROUNDING context (tokens t-5..t-2 and t+1..t+4, frequent tokens down-weighted).
f is a learned composition function: 'not hot' is mapped to where the contexts of 'not hot' live.
Token-in-context vector at t = f(x_{t-1}, x_t)  (and the mirror pair (x_t, x_{t+1}) for the right side).
"""
from hx import *
from nl import RandomBilinear
LOG = "roundC.log"; open(LOG, "w")
ids = load_corpus(); freq = token_freq(ids); ref = make_ref(ids); n = len(ids); split = int(0.9 * n)
E = np.load(f"{BIG}/E.npy"); d = E.shape[1]; tw = (1e-3 / (1e-3 + freq)).astype(np.float32)
NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000; DP = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
rb = RandomBilinear(d, d, DP, 7)
def phi(u, v): return np.c_[u, v, rb(u, v)]
OFF = [-5, -4, -3, -2, 1, 2, 3, 4]
def target(p):
    W = np.stack([tw[ids[p + o]] for o in OFF], 1); X = np.stack([E[ids[p + o]] for o in OFF], 1)
    return nz((X * W[..., None]).sum(1) / W.sum(1, keepdims=True))
r = np.random.default_rng(0); ptr = np.sort(r.choice(np.arange(6, split - 6), NTR, replace=False)); pva = r.choice(np.arange(split + 6, n - 6), 20000, replace=False)
rf = RidgeFit()
for c in range(0, NTR, 50000):
    p = ptr[c:c + 50000]; rf.add(phi(E[ids[p - 1]], E[ids[p]]), target(p))
pv = pva; cos = rf.solve(phi(E[ids[pv - 1]], E[ids[pv]]), target(pv)); tick(f"fit pair composer on {NTR:,} pairs, held-out cos {cos:.3f} lam {rf.lam}")
# linear-only control (no products)
rl = RidgeFit()
for c in range(0, NTR, 50000):
    p = ptr[c:c + 50000]; rl.add(np.c_[E[ids[p - 1]], E[ids[p]]], target(p))
cosl = rl.solve(np.c_[E[ids[pv - 1]], E[ids[pv]]], target(pv)); tick(f"linear-only control held-out cos {cosl:.3f}")

def prev_tok_feats(t, st, f):
    """left pair (x_{t-1}, x_t), x_{-1}=0 at sentence start"""
    X = E[t]; U = np.zeros_like(X); U[1:] = X[:-1]; U[st[:-1]] = 0; return f(U, X)
def comp(f):
    def g(t, st):
        X = E[t]; U = np.zeros_like(X); U[1:] = X[:-1]; U[st[:-1]] = 0
        Vn = np.zeros_like(X); Vn[:-1] = X[1:]; Vn[st[1:] - 1] = 0
        return nz(f(U, X)) + nz(f(X, Vn))            # left pair + right pair
    return g
pairs = lambda u, v: rf(phi(u, v)); lin = lambda u, v: rl(np.c_[u, v])
W2 = {"mean": lambda t, st, F: np.ones(len(t), np.float32), "sif": sif_w(freq)}
evaluate("linear composer", comp(lin), ref, freq, posts=("white",), weights=W2, log=LOG)
evaluate("product composer", comp(pairs), ref, freq, posts=("white",), weights=W2, log=LOG)
evaluate("[x ; product composer]", lambda t, st: np.c_[E[t], nz(comp(pairs)(t, st))], ref, freq, posts=("white",), weights=W2, log=LOG)
import pickle; pickle.dump(dict(rb=rb, rf=rf, rl=rl), open("roundC_model.pkl", "wb"))
tick("done")
