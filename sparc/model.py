"""SPARC: a deep, closed-form contextual encoder. Build from a token stream; encode sentences; play with it.

Pipeline (every step is counting, an SVD, or a ridge solve):
  1. one co-occurrence count  -> token meanings (768-d out, 128-d internal) and directional PMI factors (attention)
  2. pattern tables (hashed, fixed memory)
  3. three streams (cloze / fwd / bwd), 3 layers each
  4. whitening statistics for sentence vectors (fitted on ordinary corpus sentences)
"""
import os, json, time, pickle
import numpy as np
from tokenizers import Tokenizer
from .utils import nz, tick, sentence_ids, sample_bank, keep_prob
from .meanings import cooccurrence, meanings, pmi_factors
from .patterns import Patterns
from .reservoir import Memory
from .attention import CountingAttention
from .products import RandomProducts
from .stream import Stream

DEFAULTS = dict(window=8, dim_out=768, pmi_rank=256, buckets=2 ** 21, min_count=3, memory_units=256, n_layers=3, width=2048,
                products="random", train_sentences=100_000, train_targets=200_000, val_sentences=3000, white_sentences=10_000, tau=0.2)


class SPARC:
    def __init__(self, tokenizer_path, **cfg):
        self.cfg = {**DEFAULTS, **cfg}; self.tokenizer_path = tokenizer_path; self.tok = Tokenizer.from_file(tokenizer_path)
        self.V = self.tok.get_vocab_size(); self.dot = self.tok.token_to_id(".")

    # ------------------------------------------------------------------ build
    def build(self, ids, holdout=0.1, log=tick):
        c = self.cfg; ids = ids.astype(np.int64); n = len(ids); hi = int((1 - holdout) * n); T = {}; t0 = time.time()
        self.freq = (np.bincount(ids[:hi], minlength=self.V) / hi).astype(np.float64)
        M = cooccurrence(ids[:hi], self.V, c["window"]); T["co-occurrence counts"] = time.time() - t0; t0 = time.time()
        self.E_out, self.E = meanings(M, c["dim_out"]); KA, KB = pmi_factors(M, c["pmi_rank"]); del M
        T["meanings + PMI factors (2 SVDs)"] = time.time() - t0; t0 = time.time(); log(f"meanings and PMI factors ready ({sum(T.values()):.0f}s)")
        sid = sentence_ids(ids, self.dot)
        self.patterns = Patterns(ids, sid, self.E, hi, buckets=c["buckets"], minc=c["min_count"], V=self.V); del sid
        T["pattern tables (hashed)"] = time.time() - t0; t0 = time.time()
        log("pattern tables: " + ", ".join(f"{g} {G.used/1e6:.2f}M used / {G.B/1e6:.2f}M buckets" for g, G in self.patterns.groups.items()) + f"  ({self.patterns.nbytes()/1e9:.2f} GB)")
        self.memory = Memory(self.E, c["memory_units"]); self.attention = CountingAttention(KA, KB, self.E, self.freq, c["tau"])
        train = sample_bank(ids, self.dot, c["train_sentences"], 0, hi, 0); val = sample_bank(ids, self.dot, c["val_sentences"], hi, n, 1)
        kp = keep_prob(self.freq); self.streams = {}; self.report = {}
        for i, mode in enumerate(("cloze", "fwd", "bwd")):
            s = Stream(mode, self.shared(), c["n_layers"], c["width"], c["products"], seed=i + 1)
            self.report[mode] = s.fit(train, kp, ntr=c["train_targets"], val=val, log=log); self.streams[mode] = s
            T[f"stream {mode} ({c['n_layers']} layers)"] = time.time() - t0; t0 = time.time()
        self.xp = RandomProducts(self.E.shape[1], 2 * self.E.shape[1], 2048, 77)
        ref = sample_bank(ids, self.dot, c["white_sentences"], 0, hi, 2); self.fit_whitening(ref); T["whitening"] = time.time() - t0
        self.times = T; return T

    def shared(self):
        return dict(E=self.E, patterns=self.patterns, memory=self.memory, attention=self.attention, V=self.V)

    # ------------------------------------------------------------------ encoding
    def tokenize(self, sentences, close=True):
        """lower-case + BPE. Training sentences always end in '.', so by default one is appended if missing
        (otherwise the right edge of a sentence looks like a gap that must be filled by '.')."""
        sentences = [s.lower().strip() for s in sentences]
        if close: sentences = [s if s.endswith((".", "?", "!")) else s + " ." for s in sentences]
        enc = self.tok.encode_batch(sentences)
        toks = [np.array(e.ids, np.int64) for e in enc]; starts = np.r_[0, np.cumsum([len(t) for t in toks])].astype(np.int64)
        words = [np.array([w if w is not None else -1 for w in e.word_ids]) for e in enc]
        return np.concatenate(toks), starts, words, enc

    def encode_tokens(self, tok, starts):
        """dict of per-token blocks: meaning, every stream's guess/hidden at every layer, word x context products."""
        out = {"meaning": self.E_out[tok]}
        for m, s in self.streams.items():
            for k, o in enumerate(s.encode(tok, starts), 1): out[f"{m}_g{k}"] = o["g"]; out[f"{m}_h{k}"] = o["h"]
        K = self.cfg["n_layers"]; out["word_x_context"] = nz(self.xp(self.E[tok], np.c_[out[f"fwd_g{K}"], out[f"bwd_g{K}"]]))
        return out

    def token_vectors(self, sentences, parts=None):
        """per sentence: (tokens, [n_tokens x dim] contextual vectors). parts: which blocks (default: all)."""
        tok, starts, _, enc = self.tokenize(sentences); B = self.encode_tokens(tok, starts); parts = parts or list(B)
        X = np.concatenate([nz(B[p]) for p in parts], 1)
        return [(enc[i].tokens, X[starts[i]:starts[i + 1]]) for i in range(len(sentences))]

    def word_vector(self, sentence, word, parts=None):
        """contextual vector of `word` (first occurrence) inside `sentence` (mean over its BPE pieces)."""
        tok, starts, words, _ = self.tokenize([sentence]); B = self.encode_tokens(tok, starts); parts = parts or list(B)
        wi = sentence.lower().split().index(word.lower()); m = np.flatnonzero(words[0] == wi)
        return nz(np.concatenate([nz(B[p][m].mean(0, keepdims=True)) for p in parts], 1))[0]

    def _pool(self, tok, starts):
        B = self.encode_tokens(tok, starts); X = np.c_[B["meaning"], B["cloze_g1"]]
        return np.add.reduceat(X, starts[:-1], 0) / np.diff(starts)[:, None]

    def fit_whitening(self, bank):
        S = np.concatenate([self._pool(t, s) for t, s in bank.chunks(1000)]); self.w_mu = S.mean(0)
        U, Sv, Vt = np.linalg.svd((S - self.w_mu).astype(np.float64), full_matrices=False)
        self.w_W = (Vt.T / (Sv / np.sqrt(len(S)) + 1e-6 * Sv[0])).astype(np.float32)

    def sentence_vectors(self, sentences):
        tok, starts, _, _ = self.tokenize(sentences); return nz((self._pool(tok, starts) - self.w_mu) @ self.w_W)

    # ------------------------------------------------------------------ fun
    def fill_mask(self, text, k=10):
        """'she sat on the river [MASK] and watched the water' -> top tokens from the cloze stream"""
        left, right = text.lower().strip().split("[mask]")
        if not right.strip().endswith((".", "?", "!")): right = right.rstrip() + " ."
        a = self.tok.encode(left.strip()).ids if left.strip() else []; b = self.tok.encode(right.strip()).ids if right.strip() else []
        tok = np.array(a + [0] + b, np.int64); starts = np.array([0, len(tok)], np.int64)       # the gap's own id is never read
        sc = self.streams["cloze"].scores(tok, starts, [len(a)])[0]; top = np.argsort(-sc)[:k]
        return [self.tok.id_to_token(int(i)) for i in top]

    def next_tokens(self, text, k=10):
        tok = np.array(self.tok.encode(text.lower()).ids, np.int64); starts = np.array([0, len(tok)], np.int64)
        sc = self.streams["fwd"].scores(tok, starts, [len(tok) - 1])[0]; top = np.argsort(-sc)[:k]
        return [self.tok.id_to_token(int(i)) for i in top], sc[top]

    def generate(self, text, n=15, top=4, seed=0, ban=("[UNK]",)):
        r = np.random.default_rng(seed); ids = list(self.tok.encode(text.lower()).ids); bad = {self.tok.token_to_id(b) for b in ban}
        for _ in range(n):
            tok = np.array(ids, np.int64); sc = self.streams["fwd"].scores(tok, np.array([0, len(tok)], np.int64), [len(tok) - 1])[0]
            for b in bad: sc[b] = -1e9
            for j in set(ids[-6:]): sc[j] -= 0.5 * abs(sc[j])                                   # mild repetition penalty
            cand = np.argsort(-sc)[:top]; p = np.maximum(sc[cand], 1e-6) ** 4; ids.append(int(r.choice(cand, p=p / p.sum())))
            if ids[-1] == self.dot: break
        return self.tok.decode(ids).replace(" ##", "")

    def neighbours(self, word, k=10):
        ids = self.tok.encode(word.lower()).ids; v = nz(self.E_out[ids].mean(0, keepdims=True))[0]; s = self.E_out @ v; o = np.argsort(-s)
        return [(self.tok.id_to_token(int(i)), round(float(s[i]), 2)) for i in o[1:k + 1] if not self.tok.id_to_token(int(i)).startswith("##")][:k]

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for s in self.streams.values(): s.detach()
        tk = self.tok; self.tok = None
        with open(f"{path}/model.pkl", "wb") as f: pickle.dump(self, f, protocol=4)
        self.tok = tk
        for s in self.streams.values(): s.attach(self.shared())
        json.dump(dict(cfg=self.cfg, times=self.times), open(f"{path}/info.json", "w"), indent=1)

    @staticmethod
    def load(path):
        with open(f"{path}/model.pkl", "rb") as f: m = pickle.load(f)
        m.tok = Tokenizer.from_file(m.tokenizer_path)
        for s in m.streams.values(): s.attach(m.shared())
        return m
