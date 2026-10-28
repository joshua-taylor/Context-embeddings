"""Context features for a closed-form masked-token model (sentence-reset, bias-corrected reservoirs + exact window).

Everything is defined per sentence: states reset at each sentence start, in training and at test time.
Bias correction: a leaky unit h <- a h + (1-a) x started at 0 underestimates its average by the factor (1 - a^k)
after k steps; we divide that out (same trick as Adam's bias correction), so short sentences are not washed out.
"""
import numpy as np, numba as nb

@nb.njit(cache=True, fastmath=True)
def scan_collect(tok, reset, T, a, pos, out):
    """Run the leaky reservoir over tok; out[j] = bias-corrected state BEFORE token pos[j] (pos sorted)."""
    U = T.shape[1]; h = np.zeros(U, np.float32); c = np.zeros(U, np.float32); j = 0; npos = len(pos)
    for t in range(len(tok)):
        if reset[t]:
            for u in range(U): h[u] = 0.0; c[u] = 0.0
        while j < npos and pos[j] == t:
            for u in range(U): out[j, u] = h[u] / c[u] if c[u] > 0 else 0.0
            j += 1
        x = tok[t]
        for u in range(U):
            h[u] = a[u] * h[u] + (1.0 - a[u]) * T[x, u]; c[u] = a[u] * c[u] + (1.0 - a[u])
        if j >= npos: break

class Res:
    def __init__(self, E, U=256, seed=3):
        r = np.random.default_rng(seed); d = E.shape[1]
        self.T = (E @ (r.standard_normal((d, U)) / np.sqrt(d))).astype(np.float32)
        self.a = np.r_[r.uniform(0.1, 0.7, U // 2), r.uniform(0.8, 0.98, U - U // 2)].astype(np.float32)
    def before(self, tok, reset, pos):
        out = np.zeros((len(pos), self.T.shape[1]), np.float32); scan_collect(tok, reset, self.T, self.a, pos, out); return out
    def after(self, tok, reset_end, pos):
        """state of the backward reservoir AFTER pos (reads tokens to the right, reset at sentence ends)."""
        n = len(tok); rtok = tok[::-1].copy(); rres = reset_end[::-1].copy(); rp = (n - 1 - pos)[::-1].copy()
        return self.before(rtok, rres, rp)[::-1]

def corpus_flags(ids, dot):
    st = np.zeros(len(ids), np.bool_); st[0] = True; st[1:] = ids[:-1] == dot      # sentence starts
    en = np.zeros(len(ids), np.bool_); en[-1] = True; en[:-1] = st[1:]                 # last token of a sentence
    rev_reset = np.zeros(len(ids), np.bool_); rev_reset[:] = en                        # backward scan resets at sentence ends
    sid = np.cumsum(st) - 1
    return st, rev_reset, sid
