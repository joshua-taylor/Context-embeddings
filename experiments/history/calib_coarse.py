"""RoBERTa-base and static baselines on the coarse word-in-context test (test-split lemmas only)."""
from hx import *
import spacy, warnings; warnings.filterwarnings("ignore")
from spacy.tokens import Doc
E = np.load(f"{BIG}/E.npy")
test_pairs = [(a, b, p["label"]) for (a, b), p in zip(CSEM, CPAIRS) if p["split"] == "test"]
need = sorted({x[0] for a, b, _ in test_pairs for x in (a, b)}); print(len(test_pairs), "test pairs,", len(need), "sentences")
# static token meanings: mean of the word's pieces
def static(k):
    si, wi = k; return TEST.tok[SEM_TOK[SEM_IX[k]]]
sv = lambda k: nz(E[static(k)].mean(0))
s = np.array([sv(a) @ sv(b) for a, b, _ in test_pairs]); y = np.array([l for *_, l in test_pairs])
print(f"static token meanings  coarse AUC {auc(s, y)*100:.1f}")
nlp = spacy.load("en_core_web_trf"); tr = nlp.get_pipe("transformer"); V = {}
docs = (Doc(nlp.vocab, words=SENTS[i]) for i in need)
for i, d in zip(need, tr.pipe(docs, batch_size=32)):
    R = d._.trf_data.last_hidden_layer_state; X = R.dataXd; L = R.lengths; o = np.r_[0, np.cumsum(L)]; V[i] = (X, o)
def rv(k):
    X, o = V[k[0]]; w = min(k[1], len(o) - 2); return nz(X[o[w]:o[w + 1]].mean(0))
s = np.array([rv(a) @ rv(b) for a, b, _ in test_pairs]); print(f"RoBERTa-base last layer coarse AUC {auc(s, y)*100:.1f}")
open("calib.log", "a").write(f"  coarse WiC (test lemmas): static {auc(np.array([sv(a) @ sv(b) for a, b, _ in test_pairs]), y)*100:.1f}  RoBERTa {auc(s, y)*100:.1f}\n")
