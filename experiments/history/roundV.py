"""Round V: SPARC-3 streams (cloze / fwd / bwd), 3 layers each, timed. usage: python roundV.py MODE"""
from hx import *
import pickle, gc
from sparc3 import Stream, make_reservoirs
MODE = sys.argv[1]; NL = int(sys.argv[2]) if len(sys.argv) > 2 else 3; LOG = "roundV.log"
E = np.load(f"{BIG}/E.npy"); sc = pickle.load(open("tables128.pkl", "rb")); tables = sc.tables; del sc
for g in ("LLR", "LRR"): tables.pop(g, None)
ids = load_corpus().astype(np.int64); n = len(ids); hi = int(0.9 * n); freq = token_freq(ids)
rr = np.random.default_rng(0); s_, e_ = sentence_bounds(ids, 0, hi); ktr = np.sort(rr.choice(len(s_), 100_000, replace=False))
sv, ev = sentence_bounds(ids, hi, n); kva = np.sort(rr.choice(len(sv), 3000, replace=False))
train = Bank(toks=[ids[s_[i]:e_[i]] for i in ktr]); val = Bank(toks=[ids[sv[i]:ev[i]] for i in kva]); del ids; gc.collect()
keep_p = np.minimum(1.0, np.sqrt(1e-4 / (freq + 1e-12)) + 1e-4 / (freq + 1e-12)).astype(np.float32)
Zp = np.load(f"{BIG}/pmi_W8.npz"); pmi = (np.ascontiguousarray(Zp["A"]), np.ascontiguousarray(Zp["B"]))
s = Stream(MODE, E, tables, make_reservoirs(E), freq, pmi, seed={"cloze": 1, "fwd": 2, "bwd": 3}[MODE])
for k in range(1, NL + 1):
    info = s.add_layer(train, keep_p); r = s.evaluate(val, keep_p)
    line = (f"  {MODE:5s} layer {k}: vocab MRR {r['vocab_mrr']:.3f} top1 {r['top1']*100:4.1f}% | meaning-space MRR {r['meaning_mrr']:.3f} "
            f"| {info['dims']} feats, fit {info['t_total']:.0f}s (features {info['t_feat']:.0f}s)")
    tick(line); open(LOG, "a").write(line + "\n")
s.strip(); s.tabs = None; pickle.dump(s, open(f"stream_{MODE}.pkl", "wb")); tick("saved")
