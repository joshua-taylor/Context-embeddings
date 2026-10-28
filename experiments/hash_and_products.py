"""Cloze stream under different pattern-table memory budgets and interaction designs (shared components built once).

  python experiments/hash_and_products.py tokens.npz bpe.json
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tokenizers import Tokenizer
from sparc.utils import tick, sentence_ids, sample_bank, keep_prob
from sparc.meanings import cooccurrence, meanings, pmi_factors
from sparc.patterns import Patterns
from sparc.reservoir import Memory
from sparc.attention import CountingAttention
from sparc.stream import Stream

ids = np.load(sys.argv[1])["ids"].astype(np.int64); tk = Tokenizer.from_file(sys.argv[2]); V = tk.get_vocab_size(); dot = tk.token_to_id(".")
n = len(ids); hi = int(0.9 * n); freq = np.bincount(ids[:hi], minlength=V) / hi
M = cooccurrence(ids[:hi], V, 8); E_out, E = meanings(M, 768); KA, KB = pmi_factors(M, 256); del M
sid = sentence_ids(ids, dot); mem = Memory(E); att = CountingAttention(KA, KB, E, freq)
train = sample_bank(ids, dot, 100_000, 0, hi, 0); val = sample_bank(ids, dot, 3000, hi, n, 1); kp = keep_prob(freq)
LOG = os.path.join(os.path.dirname(__file__), "logs", "hash_and_products.log"); out = open(LOG, "a")
tables = {}
CONFIGS = [(2 ** 21, "random", 2048), (2 ** 19, "random", 2048), (2 ** 17, "random", 2048), (2 ** 21, "subspace", 2048), (2 ** 21, "subspace", 4608)]
for B, kind, width in CONFIGS:
    if B not in tables:
        t0 = time.time(); tables = {B: Patterns(ids, sid, E, hi, buckets=B)}; tb = time.time() - t0
    P = tables[B]; s = Stream("cloze", dict(E=E, patterns=P, memory=mem, attention=att, V=V), 3, width, kind, seed=1)
    t0 = time.time(); rep = s.fit(train, kp, val=val, log=tick); tt = time.time() - t0
    st = val.starts[:np.searchsorted(val.starts, 30000, "right")]; tok = val.tok[:st[-1]]
    s.encode(tok[:st[3]], st[:4]); t0 = time.time(); s.encode(tok, st); sp = len(tok) / (time.time() - t0)
    used = sum(G.used for g, G in P.groups.items() if not G.direct); nbytes = P.nbytes()
    line = (f"buckets {B:>9,} ({used/1e6:.2f}M patterns kept, {nbytes/1e9:.2f} GB, built {tb:.0f}s) | products {kind:8s} {width:5d} | "
            + " ".join(f"L{i+1} {r['mrr']:.3f}" for i, r in enumerate(rep)) + f" (top1 {rep[-1]['top1']*100:.1f}%) | fit {tt:.0f}s | encode {sp:,.0f} tok/s")
    tick(line); out.write(line + "\n"); out.flush()
