"""Round G: embeddings from the full-rank masked-token model (hidden state = readout subspace)."""
from hx import *
from mlm2 import MLM2, Hidden
import pickle, gc
LOG = "roundG.log"; open(LOG, "a")
ids = load_corpus(); freq = token_freq(ids); ref = make_ref(ids); E = np.load(f"{BIG}/E.npy"); NTR = int(sys.argv[1])
m = pickle.load(open("mlm2_full.pkl", "rb")); del ids; gc.collect()
H = Hidden(m, 256)
cache = {}
LAST = {}
def hz(t, st):
    z, sc = H(t, st, True); LAST["sc"] = sc; LAST["n"] = len(t); return z
def feat_z(t, st): return hz(t, st)
def feat_xz(t, st): return np.c_[E[t], nz(hz(t, st))]
def feat_x(t, st): hz(t, st); return E[t]
def csif(a):
    def w(t, st, F): assert LAST["n"] == len(t); p = np.clip(LAST["sc"], 1e-5, 1.0); return (a / (a + p)).astype(np.float32)
    return w
W3 = {"mean": lambda t, st, F: np.ones(len(t), np.float32), "sif": sif_w(freq), "ctx-sif": csif(0.01)}
evaluate("hidden z (r=256)", feat_z, ref, freq, posts=("white",), weights=W3, log=LOG)
evaluate("[x ; hidden z]", feat_xz, ref, freq, posts=("white",), weights=W3, log=LOG)
evaluate("static x, ctx-sif pooling", feat_x, ref, freq, posts=("white",), weights={"ctx-sif": csif(0.01), "ctx-sif .1": csif(0.1)}, log=LOG)
tick("done")
