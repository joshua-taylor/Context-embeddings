"""Round F2: does the full-rank vocabulary readout let the nonlinear features help? (masked-token accuracy only)"""
from hx import *
from mlm2 import MLM2
ids = load_corpus(); freq = token_freq(ids); E = np.load(f"{BIG}/E.npy"); NTR = int(sys.argv[1])
for parts in [tuple(p.split("+")) for p in sys.argv[2].split(",")]:
    m = MLM2(E, parts); v = m.fit(ids, DOT, freq, ntr=NTR)
    line = f"MLM2 {'+'.join(parts):14s} vocab readout ({NTR:,} slots), held-out masked-token: MRR {v['mrr']:.3f}  top1 {v['top1']:.3f}  top10 {v['top10']:.3f} (lam {v['lam']})"
    tick(line); open("roundF.log", "a").write(line + "\n"); del m; import gc; gc.collect()
