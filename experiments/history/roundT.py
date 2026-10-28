"""Round T: how deep can the closed-form gap-filler go? (nonlinear expansion between every pair of layers)"""
from hx import *
import pickle, gc
from layers import Deep
LOG = "roundT.log"; open(LOG, "a")
E = np.load(f"{BIG}/E.npy"); sc = pickle.load(open("tables128.pkl", "rb"))
for g in ("LLR", "LRR"): del sc.tables[g]
ids = load_corpus().astype(np.int64); n = len(ids); hi = int(0.9 * n); freq = token_freq(ids)
rr = np.random.default_rng(0); s_, e_ = sentence_bounds(ids, 0, hi); ktr = np.sort(rr.choice(len(s_), 100_000, replace=False))
sv, ev = sentence_bounds(ids, hi, n); kva = np.sort(rr.choice(len(sv), 3000, replace=False))
train = Bank(toks=[ids[s_[i]:e_[i]] for i in ktr]); val = Bank(toks=[ids[sv[i]:ev[i]] for i in kva]); del ids; gc.collect()
keep_p = np.minimum(1.0, np.sqrt(1e-4 / (freq + 1e-12)) + 1e-4 / (freq + 1e-12)).astype(np.float32)
import os
START = int(sys.argv[1]) if len(sys.argv) > 1 else 1
if START > 1: m = pickle.load(open(f"deep_ckpt_L{START-1}.pkl", "rb")); m.sc = sc
else: m = Deep(E, sc, freq=freq)
t0 = time.time()
PLAN = [(None, None, False), ("g", True, False), ("h", True, True)]
for k, (carry, prod, resid) in enumerate(PLAN, 1):
    if k < START: continue
    tl = time.time(); m.add_layer(train, keep_p, carry=carry or "g", prod=prod if prod is not None else True, seed=k, residual=resid); r = m.evaluate(val, keep_p)
    desc = "patterns + memories" if k == 1 else f"+ layer-{k-1} {'guess' if carry == 'g' else 'hidden state'} x context products{' (residual)' if resid else ''}"
    line = f"  layer {k} ({desc:46s}) vocab MRR {r['vocab_mrr']:.3f} top1 {r['top1']*100:4.1f}% | meaning-space MRR {r['meaning_mrr']:.3f}  (layer fit {time.time()-tl:.0f}s)"
    tick(line); open(LOG, "a").write(line + "\n")
    m.sc = None; Wv_ = m.layers[-1]["Wv"]; m.layers[-1]["Wv"] = None; pickle.dump(m, open(f"deep_ckpt_L{k}.pkl", "wb")); m.sc = sc; m.layers[-1]["Wv"] = Wv_; del Wv_
m.sc = None
for L in m.layers: L["Wv"] = None
pickle.dump(m, open("deep_model_T.pkl", "wb")); tick("done")
