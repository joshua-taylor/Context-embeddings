"""Train SPARC end to end from a BPE token stream.

  python scripts/train.py --tokens tokens.npz --tokenizer bpe.json --out model_dir [--buckets 2097152] [--products random]

tokens.npz must hold an int array `ids` (the corpus as BPE token ids, sentences ending in '.').
Use scripts/prepare_corpus.py to make it from raw text.
"""
import argparse, json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sparc import SPARC
from sparc.utils import tick

p = argparse.ArgumentParser()
p.add_argument("--tokens", required=True); p.add_argument("--tokenizer", required=True); p.add_argument("--out", required=True)
p.add_argument("--buckets", type=int, default=2 ** 21); p.add_argument("--products", default="random", choices=["random", "subspace"])
p.add_argument("--width", type=int, default=2048); p.add_argument("--layers", type=int, default=3); p.add_argument("--train-targets", type=int, default=200_000)
a = p.parse_args()
ids = np.load(a.tokens)["ids"]
m = SPARC(a.tokenizer, buckets=a.buckets, products=a.products, width=a.width, n_layers=a.layers, train_targets=a.train_targets)
t0 = time.time(); T = m.build(ids)
tick("build finished in %.0fs" % (time.time() - t0))
for k, v in T.items(): print(f"  {k:36s} {v:7.1f}s")
m.save(a.out); json.dump(dict(times=T, report=m.report), open(os.path.join(a.out, "report.json"), "w"), indent=1, default=float)
