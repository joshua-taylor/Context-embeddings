"""Word-in-context comparison on fine (SemCor synsets) and coarse (different WordNet classes) tests."""
from hx import *
import pickle, gc
from att import PMIAttn
from sparse import SparseCloze
LOG = "roundWiC.log"; open(LOG, "w")
E = np.load(f"{BIG}/E.npy")
wic_eval(lambda t, s: E[t], "static token meanings", LOG)
Z = np.load(f"{BIG}/pmi_W8.npz"); at = PMIAttn(Z["A"], Z["B"], E, tau=0.4, sink=1.0)
wic_eval(lambda t, s: np.c_[E[t], nz(at(t, s))], "[x ; PMI attention]", LOG)
wic_eval(lambda t, s: at(t, s), "PMI attention only", LOG)
from mlm2 import Hidden
m = pickle.load(open("mlm2_full.pkl", "rb")); H = Hidden(m, 256)
wic_eval(lambda t, s: H(t, s), "dense MLM hidden z", LOG)
wic_eval(lambda t, s: np.c_[E[t], nz(H(t, s))], "[x ; dense MLM hidden z]", LOG)
del m, H; gc.collect()
ids = load_corpus().astype(np.int64); sc = SparseCloze(E, ids, DOT); del ids; gc.collect()
R = pickle.load(open("roundH_readouts.pkl", "rb")); rf = R["rf"]
g = lambda t, s: rf(sc.bank_feats(t, s))
wic_eval(g, "sparse-product guess", LOG)
for w in (0.5, 1.0, 2.0):
    wic_eval(lambda t, s: np.c_[E[t], w * nz(g(t, s))], f"[x ; {w} x sparse guess]", LOG)
tick("done")
