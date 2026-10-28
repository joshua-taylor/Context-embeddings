"""Scorecard for a trained SPARC model.

  python scripts/evaluate.py --model model_dir --tests data/tests.json --probe data/probe.json

tests.json : {"sts": [[words_a, words_b, score], ...], "sick": [[words_a, words_b, score, label], ...]}   (STS-B / SICK test)
probe.json : SemCor sentences with POS and WordNet supersense tags (scripts/prepare_probe.py)
Reports: STS-B / SICK Spearman, SICK entailment>contradiction AUC, and linear-probe accuracy on AMBIGUOUS words
(word types seen with >= 2 tags), next to the 'most frequent tag of this word' baseline that needs no context.
"""
import argparse, json, sys, os
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sparc import SPARC, Bank, nz


def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); p = y.sum()
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * (len(y) - p)))


def sentence_scores(m, T):
    out = {}
    for name in ("sts", "sick"):
        rows = T[name]; A = m.sentence_vectors([" ".join(r[0]) for r in rows]); B = m.sentence_vectors([" ".join(r[1]) for r in rows])
        s = np.sum(A * B, 1); out[name] = float(spearmanr(s, [r[2] for r in rows]).correlation)
        if name == "sick":
            lab = np.array([r[3] for r in rows]); k = np.isin(lab, ["ENTAILMENT", "CONTRADICTION"]); out["entail>contradict"] = auc(s[k], (lab[k] == "ENTAILMENT").astype(int))
    return out


def probe(m, P, n_train=6000, n_test=2500, seed=0):
    r = np.random.default_rng(seed); tr = [s for s in P if s["split"] == "train"]; te = [s for s in P if s["split"] == "test"]
    S = [tr[i] for i in r.choice(len(tr), min(n_train, len(tr)), replace=False)] + [te[i] for i in r.choice(len(te), min(n_test, len(te)), replace=False)]
    ntr = min(n_train, len(tr)); enc = m.tok.encode_batch([" ".join(s["w"]) for s in S])
    bank = Bank([e.ids for e in enc]); wmaps = [np.array([w if w is not None else -1 for w in e.word_ids]) for e in enc]
    rows = [(si, wi) for si in range(len(S)) for wi in range(len(S[si]["w"])) if (wmaps[si] == wi).any()]
    X = None; ri = 0; names = None
    for ci, (tok, starts) in enumerate(bank.chunks(500)):
        B = m.encode_tokens(tok, starts)
        if X is None: names = list(B); dims = [B[k].shape[1] for k in names]; X = np.zeros((len(rows), sum(dims)), np.float16)
        Fc = np.concatenate([B[k] for k in names], 1); del B
        while ri < len(rows) and rows[ri][0] < ci * 500 + len(starts) - 1:
            si, wi = rows[ri]; j = si - ci * 500; pos = np.flatnonzero(wmaps[si] == wi) + starts[j]; X[ri] = Fc[pos].mean(0); ri += 1
    off = np.cumsum([0] + dims); col = {k: np.arange(off[i], off[i + 1]) for i, k in enumerate(names)}
    is_tr = np.array([si < ntr for si, _ in rows]); words = [S[si]["w"][wi] for si, wi in rows]
    K = m.cfg["n_layers"]; streams = [f"{s}_{v}{k}" for s in ("cloze", "fwd", "bwd") for v in ("g", "h") for k in range(1, K + 1)]
    REPS = {"meaning only": ["meaning"], "meaning + all stream layers": ["meaning"] + streams, "everything (+ word x context)": ["meaning"] + streams + ["word_x_context"]}
    res = {}; D = X.shape[1]
    for task, key in (("POS", "pos"), ("supersense", "ss")):
        lab = [S[si][key][wi] for si, wi in rows]; keep = np.array([l is not None for l in lab]); cls = sorted({l for l in lab if l is not None}); cmap = {c: i for i, c in enumerate(cls)}
        y = np.array([cmap[l] if l is not None else -1 for l in lab]); by = defaultdict(Counter)
        for w, l, t in zip(words, y, is_tr):
            if t and l >= 0: by[w][l] += 1
        amb = np.array([len(by[w]) >= 2 for w in words]); glob = Counter(y[is_tr & keep]).most_common(1)[0][0]
        mfs = np.array([by[w].most_common(1)[0][0] if by[w] else glob for w in words]); te_ = np.flatnonzero((~is_tr) & keep)
        res[f"{task}: no-context baseline (ambiguous words)"] = float((mfs == y)[te_][amb[te_]].mean())
        itr = np.flatnonzero(is_tr & keep); cut = int(0.9 * len(itr)); fit, hold = itr[:cut], itr[cut:]
        s1 = np.zeros(D); s2 = np.zeros(D)
        for c in range(0, len(itr), 10000): x = X[itr[c:c + 10000]].astype(np.float64); s1 += x.sum(0); s2 += (x ** 2).sum(0)
        mu = (s1 / len(itr)).astype(np.float32); sd = (np.sqrt(np.maximum(s2 / len(itr) - (s1 / len(itr)) ** 2, 0)) + 1e-6).astype(np.float32)
        std = lambda rr: np.c_[(X[rr].astype(np.float32) - mu) / sd, np.ones(len(rr), np.float32)]
        def stats(rr):
            G = np.zeros((D + 1, D + 1)); XY = np.zeros((D + 1, len(cls)))
            for c in range(0, len(rr), 5000):
                Xs = std(rr[c:c + 5000]); G += (Xs.T @ Xs).astype(np.float64); Y = np.zeros((len(Xs), len(cls)), np.float32); Y[np.arange(len(Xs)), y[rr[c:c + 5000]]] = 1; XY += Xs.T @ Y
            return G, XY
        Gf, XYf = stats(fit); Gh, XYh = stats(hold)
        pred = lambda rr, cols, W: np.concatenate([(std(rr[c:c + 5000])[:, cols] @ W).argmax(1) for c in range(0, len(rr), 5000)])
        for rep, parts in REPS.items():
            cols = np.r_[np.concatenate([col[p] for p in parts]), D]; g = Gf[np.ix_(cols, cols)]; sol = lambda G_, XY_, l: np.linalg.solve(G_ + l * np.trace(G_) / len(G_) * np.eye(len(G_)), XY_)
            lam = max((1e-3, 1e-2, 1e-1), key=lambda l: np.mean(pred(hold, cols, sol(g, XYf[cols], l)) == y[hold]))
            W = sol(g + Gh[np.ix_(cols, cols)], XYf[cols] + XYh[cols], lam); ok = pred(te_, cols, W) == y[te_]
            res[f"{task}: {rep} (ambiguous words)"] = float(ok[amb[te_]].mean())
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--tests"); p.add_argument("--probe"); a = p.parse_args()
    m = SPARC.load(a.model); out = {}
    if a.tests: out.update(sentence_scores(m, json.load(open(a.tests))))
    if a.probe: out.update(probe(m, json.load(open(a.probe))))
    for k, v in out.items(): print(f"  {k:60s} {v*100:6.1f}")
    json.dump(out, open(os.path.join(a.model, "scores.json"), "w"), indent=1)
