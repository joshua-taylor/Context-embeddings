"""Calibration: a real pretrained transformer (RoBERTa-base, 125M params, 160GB of text, via spaCy en_core_web_trf)
on exactly our tests and pooling. Word vector = mean of its pieces (last layer). No fine-tuning."""
from hx import *
import spacy, warnings; warnings.filterwarnings("ignore")
from spacy.tokens import Doc
nlp = spacy.load("en_core_web_trf"); tr = nlp.get_pipe("transformer")
def word_vecs(sents, bs=32):
    out = []
    docs = (Doc(nlp.vocab, words=list(s) if len(s) else ["."]) for s in sents)
    for d in tr.pipe(docs, batch_size=bs):
        R = d._.trf_data.last_hidden_layer_state; X = R.dataXd; L = R.lengths; o = np.r_[0, np.cumsum(L)]
        out.append(np.array([X[o[i]:o[i + 1]].mean(0) if L[i] else np.zeros(X.shape[1]) for i in range(len(L))], np.float32))
    return out
ids = load_corpus(); ref = make_ref(ids, n=5000); del ids
inv = {i: s for s, i in tok.get_vocab().items()}
def to_words(t):
    out = []
    for x in t:
        p = inv[int(x)]
        if p.startswith("##") and out: out[-1] += p[2:]
        else: out.append(p)
    return out
t0 = time.time(); RW = word_vecs([to_words(ref.tok[ref.starts[i]:ref.starts[i + 1]]) for i in range(ref.n)]); tick(f"ref encoded {time.time()-t0:.0f}s")
TW = word_vecs(SENTS); tick(f"test encoded {time.time()-t0:.0f}s ({sum(len(s) for s in SENTS)/(time.time()-t0):.0f} words/s)")
Wd = json.load(open(f"{BIG}/words.json")); wc = dict(zip(Wd["words"], Wd["counts"])); tot = sum(Wd["counts"])
semv = {}
for (si, wi), j in SEM_IX.items(): semv[j] = TW[si][min(wi, len(TW[si]) - 1)]
LOG = "calib.log"; open(LOG, "w")
for wn, wt in (("mean", lambda w: 1.0), ("sif", lambda w: 1e-3 / (1e-3 + wc.get(w, 1) / tot))):
    pl = lambda V, S: np.array([(v * np.array([wt(w) for w in s])[:, None]).sum(0) / sum(wt(w) for w in s) for v, s in zip(V, S)], np.float32)
    te = pl(TW, SENTS); rf = pl(RW, [to_words(ref.tok[ref.starts[i]:ref.starts[i + 1]]) for i in range(ref.n)])
    for pk, f in (("raw", lambda X: X), ("white", Post(rf, white=True))):
        line = f"  {'RoBERTa-base (last layer)':34s} {wn+'+'+pk:13s} {fmt(metrics(f(te), semv if (wn, pk) == ('mean', 'raw') else None))}"; print(line, flush=True); open(LOG, "a").write(line + "\n")
