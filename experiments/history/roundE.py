"""Round E: closed-form masked-token model; ablate the feature families; score the embeddings."""
from hx import *
from mlm import MLM
import pickle
LOG = "roundE.log"; open(LOG, "w")
ids = load_corpus(); freq = token_freq(ids); ref = make_ref(ids)
E = np.load(f"{BIG}/E.npy"); NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 600_000
W2 = {"mean": lambda t, st, F: np.ones(len(t), np.float32), "sif": sif_w(freq)}
for parts in [("res",), ("win", "res"), ("win", "res", "prod")]:
    m = MLM(E, parts); v = m.fit(ids, DOT, freq, ntr=NTR)
    line = f"MLM {'+'.join(parts):14s} held-out masked-token: cos {v['cos']:.3f}  MRR {v['mrr']:.3f}  top1 {v['top1']:.3f}  top10 {v['top10']:.3f}"
    tick(line); open(LOG, "a").write(line + "\n")
    evaluate(f"guess [{'+'.join(parts)}]", m.guess, ref, freq, posts=("white",), weights=W2, log=LOG)
    evaluate(f"[x ; guess] [{'+'.join(parts)}]", lambda t, st: np.c_[E[t], nz(m.guess(t, st))], ref, freq, posts=("white",), weights=W2, log=LOG)
m.E = None; pickle.dump(m, open("mlm_full.pkl", "wb"))
tick("done")
