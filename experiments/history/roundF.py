"""Round F: masked-token model with a full-rank vocabulary readout (vs the 128-d meaning bottleneck)."""
from hx import *
from mlm2 import MLM2
import pickle
LOG = "roundF.log"; open(LOG, "w")
ids = load_corpus(); freq = token_freq(ids); ref = make_ref(ids)
E = np.load(f"{BIG}/E.npy"); NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
PARTS = [tuple(p.split("+")) for p in (sys.argv[2] if len(sys.argv) > 2 else "res,win+res,win+res+prod").split(",")]
W2 = {"mean": lambda t, st, F: np.ones(len(t), np.float32), "sif": sif_w(freq)}
for parts in PARTS:
    m = MLM2(E, parts); v = m.fit(ids, DOT, freq, ntr=NTR)
    line = f"MLM2 {'+'.join(parts):14s} vocab readout, held-out masked-token: MRR {v['mrr']:.3f}  top1 {v['top1']:.3f}  top10 {v['top10']:.3f} (lam {v['lam']})"
    tick(line); open(LOG, "a").write(line + "\n")
    evaluate(f"substitutes [{'+'.join(parts)}]", m.substitute, ref, freq, posts=("white",), weights=W2, log=LOG)
    evaluate(f"[x ; substitutes] [{'+'.join(parts)}]", lambda t, st: np.c_[E[t], nz(m.substitute(t, st))], ref, freq, posts=("white",), weights=W2, log=LOG)
    pickle.dump(m, open(f"mlm2_{'+'.join(parts)}.pkl", "wb")); del m; import gc; gc.collect()
tick("done")
