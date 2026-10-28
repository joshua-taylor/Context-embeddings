"""Pattern tables ('sparse products') in fixed-size hashed storage.

For a context pattern around a gap (e.g. LR = the tokens left and right of the gap), store the average meaning of the
tokens that filled the gap, over the whole corpus, plus how often the pattern occurred.

Storage scales with a chosen number of buckets, not with the data:
  bucket      = hash(pattern) mod B
  fingerprint = 16 other bits of the hash. Three cheap passes over the corpus:
                1. a count-min sketch estimates every pattern's frequency (upper bound, 64 MB);
                2. per bucket, a Boyer-Moore majority vote among patterns whose estimate reaches min_count
                   (so a frequent pattern is not out-voted by a crowd of one-off patterns hashing to the same bucket);
                3. only the winning pattern's occurrences are accumulated.
                A lookup is a hit only if the fingerprint matches, so a bucket never returns a mixture of unrelated
                patterns; colliding minority patterns simply become misses (a cache keeping one frequent key per slot).
  value       = int8 mean meaning (per-bucket scale) + float count;  ~140 bytes per bucket.
Single-token patterns (L, R) use a direct table indexed by token id (no hashing needed).
Lookups are O(1) and run in a fused numba kernel.
"""
import numpy as np
import numba as nb

OFFS = {"L": (-1,), "R": (1,), "LR": (-1, 1), "LL": (-2, -1), "RR": (1, 2)}
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


@nb.njit(cache=True, inline="always")
def _mix(x):
    x = np.uint64(x)
    x ^= x >> np.uint64(30); x *= _M1
    x ^= x >> np.uint64(27); x *= _M2
    x ^= x >> np.uint64(31)
    return x


@nb.njit(cache=True)
def _code(tok, sid, gap, ref, offs, V):
    """pattern code for a gap at position `gap`, all key tokens must lie in the sentence of position `ref`."""
    n = len(tok); code = 0
    for o in offs:
        q = gap + o
        if q < 0 or q >= n or sid[q] != sid[ref]:
            return -1
        code = code * V + tok[q]
    return code


@nb.njit(cache=True)
def _sketch(tok, sid, offs, V, cm, hi):
    """count-min sketch (2 rows) of pattern frequencies: an upper bound on each pattern's count."""
    W = cm.shape[1]
    for p in range(hi):
        c = _code(tok, sid, p, p, offs, V)
        if c < 0: continue
        h1 = _mix(c ^ 0x5bd1e995); h2 = _mix(c + 0x27d4eb2f)
        b1 = np.int64(h1 & np.uint64(W - 1)); b2 = np.int64(h2 & np.uint64(W - 1))
        if cm[0, b1] < 65535: cm[0, b1] += 1
        if cm[1, b2] < 65535: cm[1, b2] += 1


@nb.njit(cache=True)
def _vote(tok, sid, offs, V, B, direct, cand, votes, hi, cm, minc):
    """majority vote per bucket among patterns whose sketched count reaches minc (rare patterns cannot win a bucket)."""
    W = cm.shape[1]
    for p in range(hi):
        c = _code(tok, sid, p, p, offs, V)
        if c < 0: continue
        h1 = _mix(c ^ 0x5bd1e995); h2 = _mix(c + 0x27d4eb2f)
        if min(cm[0, np.int64(h1 & np.uint64(W - 1))], cm[1, np.int64(h2 & np.uint64(W - 1))]) < minc: continue
        h = _mix(c); b = np.int64(h & np.uint64(B - 1)); fp = np.uint16((h >> np.uint64(48)) | np.uint64(1))
        if votes[b] == 0: cand[b] = fp; votes[b] = 1
        elif cand[b] == fp: votes[b] += 1
        else: votes[b] -= 1


@nb.njit(cache=True)
def _accumulate(tok, sid, offs, V, B, direct, cand, E, sums, cnt, hi):
    d = E.shape[1]
    for p in range(hi):
        c = _code(tok, sid, p, p, offs, V)
        if c < 0: continue
        if direct:
            b = c
        else:
            h = _mix(c); b = np.int64(h & np.uint64(B - 1)); fp = np.uint16((h >> np.uint64(48)) | np.uint64(1))
            if cand[b] != fp: continue
        cnt[b] += 1
        y = tok[p]
        for j in range(d): sums[b, j] += E[y, j]


@nb.njit(parallel=True, cache=True)
def _finalize(sums, cnt, minc, q, scale):
    """mean = sums / count, quantised to int8 per row, in place (no full-size temporaries)."""
    B, d = sums.shape
    for b in nb.prange(B):
        if cnt[b] < minc:
            cnt[b] = 0; scale[b] = 1.0
            for j in range(d): q[b, j] = 0
            continue
        mx = 0.0
        for j in range(d):
            v = abs(sums[b, j] / cnt[b])
            if v > mx: mx = v
        s = mx / 127.0 + 1e-12; scale[b] = s
        for j in range(d): q[b, j] = np.int8(round(sums[b, j] / cnt[b] / s))


class PatternGroup:
    def __init__(self, name, tok, sid, E, hi, buckets=2 ** 21, minc=3, V=16000):
        self.name = name; self.offs = np.array(OFFS[name], np.int64); self.V = V
        self.direct = len(self.offs) == 1
        B = V if self.direct else buckets
        assert self.direct or (B & (B - 1)) == 0, "bucket count must be a power of two"
        self.B = B
        cand = np.zeros(B, np.uint16); votes = np.zeros(B, np.int32)
        if not self.direct:
            cm = np.zeros((2, 2 ** 24), np.uint16); _sketch(tok, sid, self.offs, V, cm, hi)
            _vote(tok, sid, self.offs, V, B, False, cand, votes, hi, cm, minc); del cm
        del votes
        sums = np.zeros((B, E.shape[1]), np.float32); cnt = np.zeros(B, np.int64)
        _accumulate(tok, sid, self.offs, V, B, self.direct, cand, E, sums, cnt, hi)
        self.q = np.empty((B, E.shape[1]), np.int8); self.scale = np.empty(B, np.float32)
        _finalize(sums, cnt, minc, self.q, self.scale); del sums
        self.cnt = cnt.astype(np.float32); self.fp = cand; self.used = int((cnt > 0).sum())

    def nbytes(self):
        return self.q.nbytes + self.scale.nbytes + self.cnt.nbytes + self.fp.nbytes


@nb.njit(parallel=True, cache=True)
def _lookup(tok, sid, gaps, refs, offs, V, B, direct, fpt, q, scale, cnt, loo, E, out, col):
    """out[i, col:col+d] = mean filler meaning, out[i, col+d] = log(1+count); leave-one-out removes the gap's own token."""
    d = q.shape[1]
    for i in nb.prange(len(gaps)):
        for j in range(d + 1): out[i, col + j] = 0.0
        g = gaps[i]; c = _code(tok, sid, g, refs[i], offs, V)
        if c < 0: continue
        if direct:
            b = c
        else:
            h = _mix(c); b = np.int64(h & np.uint64(B - 1)); fp = np.uint16((h >> np.uint64(48)) | np.uint64(1))
            if fpt[b] != fp: continue
        k = cnt[b]
        if k <= 0: continue
        s = scale[b]
        if loo:
            if k - 1 < 1: continue
            y = tok[g]
            for j in range(d): out[i, col + j] = (q[b, j] * s * k - E[y, j]) / (k - 1)
            out[i, col + d] = np.log1p(k - 1)
        else:
            for j in range(d): out[i, col + j] = q[b, j] * s
            out[i, col + d] = np.log1p(k)


class Patterns:
    """all pattern groups; lookup writes [mean (d), log count] per group into a feature matrix."""

    def __init__(self, tok, sid, E, hi, groups=("L", "R", "LR", "LL", "RR"), buckets=2 ** 21, minc=3, V=16000):
        self.groups = {g: PatternGroup(g, tok, sid, E, hi, buckets, minc, V) for g in groups}
        self.d = E.shape[1]

    def lookup(self, names, tok, sid, gaps, refs, loo, E, out, col):
        for g in names:
            G = self.groups[g]
            _lookup(tok, sid, gaps, refs, G.offs, G.V, G.B, G.direct, G.fp, G.q, G.scale, G.cnt, loo, E, out, col)
            col += self.d + 1
        return col

    def nbytes(self):
        return sum(G.nbytes() for G in self.groups.values())
