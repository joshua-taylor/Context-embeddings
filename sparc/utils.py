"""Small shared helpers: normalisation, sentence banks, corpus sentence ids, logging."""
import time
import numpy as np

_T0 = time.time()


def tick(msg):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def nz(X):
    """L2-normalise rows."""
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)


class Bank:
    """A set of sentences stored as one token array plus sentence start offsets (len n+1)."""

    def __init__(self, toks):
        toks = [np.asarray(t, np.int64) for t in toks]
        self.tok = np.concatenate(toks) if toks else np.zeros(0, np.int64)
        self.starts = np.r_[0, np.cumsum([len(t) for t in toks])].astype(np.int64)
        self.n = len(toks)

    def chunks(self, size):
        """yield (tokens, starts) for groups of `size` sentences; starts are local to the chunk."""
        for c in range(0, self.n, size):
            s0, s1 = c, min(self.n, c + size)
            lo, hi = self.starts[s0], self.starts[s1]
            yield self.tok[lo:hi], self.starts[s0:s1 + 1] - lo


def sentence_ids(ids, dot):
    """corpus sentence id per token (a sentence ends with the '.' token)."""
    st = np.zeros(len(ids), np.int32)
    st[1:] = ids[:-1] == dot
    return np.cumsum(st, dtype=np.int32)


def sentence_bounds(ids, dot, lo=0, hi=None, minlen=4, maxlen=64):
    """(start, end) of corpus sentences ending in '.', inside [lo, hi)."""
    hi = len(ids) if hi is None else hi
    d = np.flatnonzero(ids[lo:hi] == dot) + lo
    s = np.r_[lo, d[:-1] + 1]
    e = d + 1
    k = ((e - s) >= minlen) & ((e - s) <= maxlen)
    return s[k], e[k]


def sample_bank(ids, dot, n, lo, hi, seed):
    s, e = sentence_bounds(ids, dot, lo, hi)
    k = np.sort(np.random.default_rng(seed).choice(len(s), min(n, len(s)), replace=False))
    return Bank([ids[s[i]:e[i]] for i in k])


def keep_prob(freq):
    """word2vec-style subsampling of frequent tokens when choosing training targets."""
    return np.minimum(1.0, np.sqrt(1e-4 / (freq + 1e-12)) + 1e-4 / (freq + 1e-12)).astype(np.float32)
