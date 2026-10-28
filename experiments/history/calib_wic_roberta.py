"""RoBERTa-base on both word-in-context tests, pooled and per-lemma AUC. Saves the target-word vectors."""
from hx import *
from hx import _sem_keys
import spacy, warnings; warnings.filterwarnings("ignore")
from spacy.tokens import Doc
nlp = spacy.load("en_core_web_trf"); tr = nlp.get_pipe("transformer")
need = sorted({k[0] for k in _sem_keys}); want = {}
for k in _sem_keys: want.setdefault(k[0], []).append(k[1])
out = np.zeros((len(_sem_keys), 768), np.float32)
for i, d in zip(need, tr.pipe((Doc(nlp.vocab, words=SENTS[i]) for i in need), batch_size=32)):
    R = d._.trf_data.last_hidden_layer_state; X = R.dataXd; o = np.r_[0, np.cumsum(R.lengths)]
    for w in want[i]: ww = min(w, len(o) - 2); out[SEM_IX[(i, w)]] = X[o[ww]:o[ww + 1]].mean(0)
np.save("roberta_wic_vecs.npy", out); Wv = nz(out)
wic_scores(lambda k: Wv[SEM_IX[k]], "RoBERTa-base (spaCy trf, last layer)", "roundWiC.log")
