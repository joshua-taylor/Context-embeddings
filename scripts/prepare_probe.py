"""Contextual probing data from SemCor: every word gets its Penn POS tag; sense-tagged nouns/verbs also get their
WordNet supersense (e.g. noun.artifact, verb.motion). Split by document (80/20) so test sentences are unseen."""
import re, json, random
from nltk.corpus import semcor
from nltk.tree import Tree
TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")
files = semcor.fileids(); rng = random.Random(0); rng.shuffle(files); test_files = set(files[: len(files) // 5])
import os; os.makedirs("data", exist_ok=True)
out = []
def pos_of(t):
    while isinstance(t, Tree) and isinstance(t[0], Tree): t = t[0]
    return t.label() if isinstance(t, Tree) else None
for f in files:
    for sent in semcor.tagged_sents(fileids=f, tag="both"):
        words, pos, ss = [], [], []
        for ch in sent:
            lab = ch.label() if isinstance(ch, Tree) else None
            sense = lab.synset().lexname() if hasattr(lab, "synset") and lab.synset() is not None else None
            inner = ch[0] if (isinstance(ch, Tree) and hasattr(lab, "synset")) else ch
            p = pos_of(inner) if isinstance(inner, Tree) else None
            if p == "NE" and isinstance(inner, Tree): p = pos_of(inner[0]) if isinstance(inner[0], Tree) else "NNP"
            leaves = [w.lower() for w in (ch.leaves() if isinstance(ch, Tree) else [ch])]
            pieces = [t for w in leaves for t in TOK.findall(w)]
            single = len(pieces) == 1
            for t in pieces:
                words.append(t); pos.append(p if (p and re.match(r"^[A-Z]", p) and t.isalnum()) else None); ss.append(sense if single else None)
        if 4 <= len(words) <= 60: out.append(dict(w=words, pos=pos, ss=ss, split="test" if f in test_files else "train"))
json.dump(out, open("data/probe.json", "w"))
n_pos = sum(p is not None for s in out for p in s["pos"]); n_ss = sum(x is not None for s in out for x in s["ss"])
print(len(out), "sentences", n_pos, "POS-tagged words", n_ss, "supersense-tagged words", sum(s["split"] == "test" for s in out), "test sentences")
