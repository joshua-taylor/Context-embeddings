"""Have some fun with a trained SPARC model: associations, fill-in-the-blank, next words, sense in context, similarity.

  python scripts/demo.py --model model_dir [--tokens tokens.npz]   (tokens: corpus, for word-in-context retrieval)
"""
import argparse, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sparc import SPARC, Bank, nz
from sparc.utils import sentence_bounds

p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--tokens"); p.add_argument("--out", default=None); a = p.parse_args()
m = SPARC.load(a.model); L = []
def say(s=""): print(s); L.append(s)
K = m.cfg["n_layers"]
CTX = [f"{s}_{v}{k}" for s in ("cloze", "fwd", "bwd") for v in ("g", "h") for k in range(1, K + 1)] + ["word_x_context"]

say("## 1. Word associations (nearest token meanings)\n")
for w in ["river", "doctor", "guitar", "happy", "london", "war", "money", "running"]:
    say(f"- **{w}**: " + ", ".join(t for t, _ in m.neighbours(w, 8)))

say("\n## 2. Fill in the blank (cloze stream, top guesses)\n")
for s in ["she sat on the river [MASK] and watched the water", "he deposited the money at the [MASK] on monday",
          "the [MASK] barked at the postman", "the doctor gave the patient some [MASK]", "paris is the capital of [MASK]",
          "the band played a new [MASK] at the concert", "the war ended in [MASK]", "she was elected [MASK] of the united states"]:
    say(f"- {s.replace('[MASK]', '___')}  →  " + ", ".join(m.fill_mask(s, 8)))

say("\n## 3. What comes next (forward stream)\n")
for s in ["the river flows into the", "she was born in", "the album was released", "the united states", "the battle of"]:
    toks, _ = m.next_tokens(s, 8); say(f"- {s} …  →  " + ", ".join(toks))
say("\nSampled continuations (sampled from the top 4 guesses at each step):\n")
for s in ["the city is", "the river", "in the early years of the war", "the band"]:
    say(f"- {m.generate(s, 14, seed=1)}")

say("\n## 4. One word, different senses (contextual vectors, meaning excluded)\n")
SENSES = {"bank": ["she sat on the river bank and watched the water", "the boat drifted towards the muddy bank of the stream",
                   "the bank raised its interest rates again", "he borrowed money from the bank to buy a house"],
          "plant": ["the plant grew tall in the sunny garden", "water the plant every day and it will flower",
                    "the car plant employs two thousand workers", "the power plant was closed after the accident"],
          "bass": ["he caught a large bass in the lake", "the fishermen landed bass and trout",
                   "he plays bass guitar in a rock band", "the song has a heavy bass line"]}
for w, sents in SENSES.items():
    Vv = np.array([m.word_vector(s, w, CTX) for s in sents]); C = Vv @ Vv.T
    say(f"**{w}** (cosine; rows = sentences)\n"); say("| | " + " | ".join(f"s{i+1}" for i in range(4)) + " |"); say("|---|" + "---|" * 4)
    for i, s in enumerate(sents): say(f"| s{i+1}: {s} | " + " | ".join(f"{C[i, j]:.2f}" for j in range(4)) + " |")
    say("\nwhat fits in that slot (cloze guesses with the word hidden):\n")
    for s in sents: say(f"- {s.replace(w, '___', 1)}  →  " + ", ".join(m.fill_mask(s.replace(w, '[MASK]', 1), 6)))
    say("")

if a.tokens:
    say("## 5. Word-in-context retrieval: nearest corpus occurrences to a query use\n")
    ids = np.load(a.tokens)["ids"].astype(np.int64); s_, e_ = sentence_bounds(ids, m.dot, int(0.9 * len(ids)), len(ids), 6, 40)
    for w, qs in [("bank", ["she sat on the river bank and watched the water", "the bank raised its interest rates again"]),
                  ("plant", ["the plant grew tall in the sunny garden", "the power plant was closed after the accident"])]:
        wid = m.tok.token_to_id(w); hits = [i for i in range(len(s_)) if wid in ids[s_[i]:e_[i]]][:400]
        bank = Bank([ids[s_[i]:e_[i]] for i in hits]); vecs = []
        for tok, starts in bank.chunks(200):
            B = m.encode_tokens(tok, starts); X = np.concatenate([nz(B[p]) for p in CTX], 1)
            for j in range(len(starts) - 1):
                pos = np.flatnonzero(tok[starts[j]:starts[j + 1]] == wid)[0] + starts[j]; vecs.append(nz(X[pos]))
        vecs = np.array(vecs); texts = [m.tok.decode(list(ids[s_[i]:e_[i]])).replace(" ##", "") for i in hits]
        for qtext in qs:
            qv = m.word_vector(qtext, w, CTX); top = np.argsort(-(vecs @ qv))[:3]
            say(f"- query: *{qtext}*"); [say(f"    - {texts[i]}") for i in top]
    say("")

say("## 6. Sentence similarity (whitened sentence vectors): closest match in the pool\n")
POOL = ["a man is playing a guitar", "a man plays the guitar on stage", "a woman is slicing an onion", "someone is cutting vegetables in the kitchen",
        "the stock market fell sharply today", "share prices dropped steeply this morning", "the doctor examined the patient",
        "the physician checked the sick man", "two dogs are running in a field", "a pair of dogs run across the grass",
        "the government announced new taxes", "the parliament approved a new tax law"]
S = m.sentence_vectors(POOL); C = S @ S.T; np.fill_diagonal(C, -9)
for i, s in enumerate(POOL): j = int(np.argmax(C[i])); say(f"- {s}  →  {POOL[j]}  ({C[i, j]:.2f})")
if a.out: open(a.out, "w").write("\n".join(L) + "\n")
