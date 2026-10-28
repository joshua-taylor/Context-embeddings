"""Coarse word-in-context test from SemCor: same synset (label 1) vs senses in DIFFERENT WordNet lexicographer
classes (label 0), e.g. bank = financial institution (noun.group) vs sloping land (noun.object).
Fine-grained near-duplicate senses are excluded, so a model that really tracks context should separate these."""
import re, json, random
from collections import defaultdict
from nltk.corpus import semcor
from nltk.tree import Tree
TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")
occ = defaultdict(list)
for sent in semcor.tagged_sents(tag="sem"):
    words, targets = [], []
    for ch in sent:
        if isinstance(ch, Tree):
            leaves = [w.lower() for w in ch.leaves()]; lab = ch.label()
            if len(leaves) == 1 and hasattr(lab, "synset"):
                ss = lab.synset()
                if ss.pos() in ("n", "v") and leaves[0].isalpha(): targets.append((len(words), lab.name().lower() + "." + ss.pos(), ss.name(), ss.lexname()))
            words += [t for w in leaves for t in TOK.findall(w)] if len(leaves) > 1 else [leaves[0]]
        else:
            words += [t for w in ch for t in TOK.findall(w.lower())]
    for i, lem, ss, lx in targets: occ[lem].append((words, i, ss, lx))
rng = random.Random(1); pairs = []
for lem, L in occ.items():
    by = defaultdict(list)
    for o in L: by[o[2]].append(o)
    senses = [s for s, v in by.items() if len(v) >= 2]
    lexof = {s: by[s][0][3] for s in senses}
    cross = [(a, b) for a in senses for b in senses if a < b and lexof[a] != lexof[b]]
    if not cross: continue
    k = 6
    for _ in range(k):
        s = rng.choice([x for x, _ in cross] + [y for _, y in cross]); a, b = rng.sample(by[s], 2); pairs.append(dict(lemma=lem, label=1, a=[a[0], a[1]], b=[b[0], b[1]]))
        s1, s2 = rng.choice(cross); pairs.append(dict(lemma=lem, label=0, a=[rng.choice(by[s1])[0], 0], b=[None, 0]) if False else dict(lemma=lem, label=0, a=list(rng.choice(by[s1])[:2]), b=list(rng.choice(by[s2])[:2])))
lemmas = sorted({p["lemma"] for p in pairs}); rng.shuffle(lemmas); dev = set(lemmas[: len(lemmas) // 2])
for p in pairs: p["split"] = "dev" if p["lemma"] in dev else "test"
json.dump(pairs, open("data/coarse_pairs.json", "w"))
print(len(pairs), "pairs", len(lemmas), "lemmas")
for p in [p for p in pairs if p["lemma"] == "bank.n"][:6]: print(p["label"], " ".join(p["a"][0][max(0, p["a"][1]-5):p["a"][1]+5]), "||", " ".join(p["b"][0][max(0, p["b"][1]-5):p["b"][1]+5]))
