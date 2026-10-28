"""Round B: closed-form PMI attention as a contextualiser."""
from hx import *
from att import PMIAttn
LOG = "roundB.log"; open(LOG, "w")
ids = load_corpus(); freq = token_freq(ids); ref = make_ref(ids); del ids
E = np.load(f"{BIG}/E.npy"); Z = np.load(f"{BIG}/pmi_W8.npz"); A, B = Z["A"], Z["B"]
sifl = np.log(1e-3 / (1e-3 + freq)).astype(np.float32)
tick("loaded")
mean_w = {"mean": lambda t, st, F: np.ones(len(t), np.float32)}
grid = [(0.2, 0.5, 0, 0), (0.2, 1.0, 0, 0), (0.1, 1.0, 0, 0), (0.4, 1.0, 0, 0), (0.2, 1.0, 0, 0.1), (0.2, 1.0, 0, 0.2), (0.2, 1.0, 0.2, 0.1), (0.2, 1.5, 0, 0.1)]
for tau, sink, lam, beta in grid:
    at = PMIAttn(A, B, E, tau=tau, sink=sink, lam=lam, kb=beta * sifl)
    nm = f"att tau={tau} sink={sink} lam={lam} kb={beta}"
    evaluate(nm, at, ref, freq, posts=("white",), weights=mean_w, log=LOG)
    evaluate(nm + " [x;c]", lambda t, st: np.c_[E[t], nz(at(t, st))], ref, freq, posts=("white",), weights=mean_w, log=LOG)
tick("done")
