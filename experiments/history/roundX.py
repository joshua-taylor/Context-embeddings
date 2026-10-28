"""Round X: engineered SPARC-3 (cached training, fused inference) at a chosen product width. usage: python roundX.py MODE DPROD"""
from hx import *
import pickle, gc, os
from sparc3 import make_reservoirs, GROUPS
from fast3 import FastStream, FastStream2
MODE = sys.argv[1]; DP = int(sys.argv[2]); NL = 3; LOG = "roundX.log"
E = np.load(f"{BIG}/E.npy"); sc = pickle.load(open("tables128.pkl", "rb")); tables = sc.tables; del sc
ids = load_corpus().astype(np.int64); n = len(ids); hi = int(0.9 * n); freq = token_freq(ids)
rr = np.random.default_rng(0); s_, e_ = sentence_bounds(ids, 0, hi); ktr = np.sort(rr.choice(len(s_), 100_000, replace=False))
sv, ev = sentence_bounds(ids, hi, n); kva = np.sort(rr.choice(len(sv), 3000, replace=False))
train = Bank(toks=[ids[s_[i]:e_[i]] for i in ktr]); val = Bank(toks=[ids[sv[i]:ev[i]] for i in kva]); del ids; gc.collect()
keep_p = np.minimum(1.0, np.sqrt(1e-4 / (freq + 1e-12)) + 1e-4 / (freq + 1e-12)).astype(np.float32)
Zp = np.load(f"{BIG}/pmi_W8.npz"); pmi = (np.ascontiguousarray(Zp["A"]), np.ascontiguousarray(Zp["B"]))
s = (FastStream2 if os.environ.get("INCR") else FastStream)(MODE, E, {g: tables[g] for g in GROUPS[MODE]}, make_reservoirs(E), freq, pmi, dprod=DP, seed={"cloze": 1, "fwd": 2, "bwd": 3}[MODE])
T0 = time.time()
for k in range(1, NL + 1):
    info = s.add_layer(train, keep_p); r = s.evaluate(val, keep_p)
    line = (f"  [products {DP:4d}] {MODE:5s} layer {k}: vocab MRR {r['vocab_mrr']:.3f} top1 {r['top1']*100:4.1f}% | {info['dims']} feats, fit {info['t_total']:.0f}s (features {info["t_feat"]:.0f}s, stats {info.get("t_stats", 0):.0f}s)")
    tick(line); open(LOG, "a").write(line + "\n")
line = f"  [products {DP:4d}] {MODE:5s} total training time for {NL} layers: {time.time()-T0:.0f}s"; tick(line); open(LOG, "a").write(line + "\n")
s.clear_cache(); s.compile()
st = TEST.starts[:np.searchsorted(TEST.starts, 30000, "right")].astype(np.int64); t = TEST.tok[:st[-1]].astype(np.int64)
s.encode(t[:st[5]], st[:6]); s.encode_fast(t[:st[5]], st[:6])
t0 = time.time(); o1 = s.encode(t, st); T1 = time.time() - t0; t0 = time.time(); o2 = s.encode_fast(t, st); T2 = time.time() - t0
agree = min(float(np.mean(np.sum(a[v] * b[v], 1))) for a, b in zip(o1, o2) for v in ("g", "h"))
line = f"  [products {DP:4d}] {MODE:5s} inference: plain {len(t)/T1:,.0f} tok/s -> fused {len(t)/T2:,.0f} tok/s ({T1/T2:.1f}x), outputs agree (min cos) {agree:.5f}"
tick(line); open(LOG, "a").write(line + "\n")
s.strip(); s.tabs = None; pickle.dump(s, open(f"fstream_{MODE}_{DP}.pkl", "wb")); tick("saved")
