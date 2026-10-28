"""Round A: does smarter pooling help? Static token meanings vs word2vec / fastText, same pooling for all."""
from hx import *
from gensim.models import KeyedVectors
LOG = "roundA.log"; open(LOG, "w")
ids = load_corpus(); freq = token_freq(ids); ref = make_ref(ids); tick("loaded")
E = np.load(f"{BIG}/E.npy")
evaluate("static E (PPMI-SVD 128d)", lambda t, st: E[t], ref, freq, log=LOG)
for a in (1e-4, 3e-4, 3e-3):
    evaluate(f"static E, SIF a={a}", lambda t, st: E[t], ref, freq, posts=("pc1",), weights={"sif": sif_w(freq, a)}, log=LOG)

# word-level baselines: same sentences, same pooling, word frequencies from the training corpus
Wd = json.load(open(f"{BIG}/words.json")); wc = dict(zip(Wd["words"], Wd["counts"])); tot = sum(Wd["counts"])
inv = {i: s for s, i in tok.get_vocab().items()}
def to_words(t):
    out = []
    for x in t:
        p = inv[int(x)]
        if p.startswith("##") and out: out[-1] += p[2:]
        else: out.append(p)
    return out
ref_words = [to_words(ref.tok[ref.starts[i]:ref.starts[i + 1]]) for i in range(ref.n)]
for nm in ("word2vec", "fastText"):
    kv = KeyedVectors.load(f"{BIG}/{nm}.kv")
    def vec(w): return kv[w] if (w in kv.key_to_index or nm == "fastText") else None
    def sents_vecs(S, wt):
        out = np.zeros((len(S), kv.vector_size), np.float32)
        for i, ws in enumerate(S):
            v = [(vec(w), wt(w)) for w in ws]; v = [(x, y) for x, y in v if x is not None]
            if v: out[i] = np.sum([x * y for x, y in v], 0) / sum(y for _, y in v)
        return out
    for wn, wt in (("mean", lambda w: 1.0), ("sif", lambda w: 1e-3 / (1e-3 + wc.get(w, 1) / tot))):
        te = sents_vecs(SENTS, wt); rf = sents_vecs(ref_words, wt)
        for pk, f in (("raw", lambda X: X), ("pc1", Post(rf, k=1)), ("white", Post(rf, white=True))):
            line = f"  {nm+' (avg words)':34s} {wn+'+'+pk:13s} {fmt(metrics(f(te)))}"; print(line, flush=True); open(LOG, "a").write(line + "\n")
tick("done")
