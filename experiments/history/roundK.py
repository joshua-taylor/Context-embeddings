"""Round K: which contextual features are richest? Linear probes (POS / supersense on ambiguous words) + word-in-context.
Representations are built from the same blocks as round J; gap guesses use the round-J meaning readouts."""
from hx import *
from hx import _sem_keys
import pickle, gc
from scipy.linalg import cho_factor, cho_solve
from zoo import Zoo
from att import PMIAttn
import probe as PR
LOG = "roundK.log"; open(LOG, "w"); open(LOG, "a").write(PR.baseline_line() + "\n"); print(PR.baseline_line())
E = np.load(f"{BIG}/E.npy"); E768 = np.load(f"{BIG}/E768.npy"); sc = pickle.load(open("tables128.pkl", "rb")); z = Zoo(E)
Zs = np.load("roundJ_stats.npz"); G, XYm, mu, sd, dims = Zs["G"], Zs["XYm"], Zs["mu"], Zs["sd"], Zs["dims"]; D = G.shape[0] - 1
ORDER = ["P", "LIN", "SEL", "HIP", "ESN", "LINX"]; off = np.cumsum(np.r_[0, dims]); idx = {k: np.arange(off[i], off[i + 1]) for i, k in enumerate(ORDER)}
z.lx_mu = None
# LINX standardisation must match training: refit on a corpus-like batch (the probe bank) is NOT allowed -> use stored mu/sd of LIN block
lin_mu, lin_sd = mu[idx["LIN"]], sd[idx["LIN"]]; z.lx_mu, z.lx_sd = lin_mu, lin_sd
READ = {}
def readout(names, lam=1e-3):
    cols = np.r_[np.concatenate([idx[k] for k in names]), D]; Gs = G[np.ix_(cols, cols)]; dd = len(cols)
    W = cho_solve(cho_factor(Gs + lam * np.trace(Gs) / dd * np.eye(dd)), XYm[cols]).astype(np.float32); return cols, W
Zat = np.load(f"{BIG}/pmi_W8.npz"); at = PMIAttn(Zat["A"], Zat["B"], E, tau=0.4, sink=1.0)
class Lazy(dict):
    """blocks for one chunk, computed only when asked for"""
    def __init__(self, t, st): super().__init__(); self.t, self.st = t, st
    def __missing__(self, k):
        t, st = self.t, self.st
        if k == "P":
            sid = np.repeat(np.arange(len(st) - 1), np.diff(st)); pos = np.arange(len(t)); Pm = []
            for g, T in sc.tables.items(): M, lc = T.lookup(t, sid, pos, None, E); Pm += [M, lc]
            v = np.concatenate(Pm, 1)
        elif k == "ATT": v = at(t, st)
        elif k == "LINX": v = z.run("LINX", t, st, Hlin=self["LIN"])
        else: v = z.run(k, t, st)
        self[k] = v; return v
def blocks(t, st): return Lazy(t, st)
def guess(B, names):
    if tuple(names) not in READ: READ[tuple(names)] = readout(names)
    cols, W = READ[tuple(names)]; X = np.concatenate([B[k] if k in names else np.zeros((len(B.t), dims[i]), np.float32) for i, k in enumerate(ORDER)], 1); Xs = np.c_[(X - mu) / sd, np.ones(len(X), np.float32)][:, cols]
    return Xs @ W
def rep(spec):
    """spec: list of parts: 'E', 'E768', reservoir names, 'ATT', 'G:P+LIN' (gap guess from those blocks). each part normalised."""
    def f(t, st):
        B = blocks(t, st); out = []
        for p in spec:
            if p == "E": out.append(E[t])
            elif p == "E768": out.append(E768[t])
            elif p.startswith("G:"): out.append(nz(guess(B, p[2:].split("+"))))
            else:
                X = B[p]; out.append(nz((X - X.mean(0)) / (X.std(0) + 1e-6)) if p not in ("ATT",) else nz(X))
        return np.concatenate(out, 1).astype(np.float32)
    return f
SPECS = [["E"], ["E768"], ["E", "LIN"], ["E", "HIP"], ["E", "ESN"], ["E", "SEL"], ["E", "ATT"], ["E", "G:P+LIN"], ["E", "G:P+LIN+SEL+HIP+ESN+LINX", "LIN", "ESN", "ATT"]]
if len(sys.argv) > 1: SPECS = [s.split(",") for s in sys.argv[1].split(";")]
for spec in SPECS:
    name = " + ".join(spec); f = rep(spec)
    PR.probe(f, name, LOG)
    wic_eval(f, name, LOG)
tick("done")
