"""Round R: depth.
(a) Stacking at the same position: stage k's gap guess (and the counting attention it queries) feeds stage k+1.
(b) Depth along the sequence (leak-free, ELMo-style): a FORWARD layer-1 model guesses every token from its left
    context only; a layer-2 linear reservoir then runs over that stream of guesses (and the backward mirror).
    For the gap at t, every input was computed without seeing token t.
All readouts are closed-form ridge; stages are fitted one after another on the same training gaps."""
from hx import *
import pickle, gc
from scipy.linalg import cho_factor, cho_solve
from zoo import Zoo2, k_lin
from att import query_attend
LOG = "roundS.log"; open(LOG, "w")
from nl import RandomBilinear
XPD = RandomBilinear(128, 256, 2048, 41); RPROJ = None
RES = (sys.argv[1] if len(sys.argv) > 1 else "LIN+SEL+OSC").split("+")
E = np.load(f"{BIG}/E.npy"); d = 128; sc = pickle.load(open("tables128.pkl", "rb"))
for g in ("LLR", "LRR"): del sc.tables[g]
z = Zoo2(E)
# ---- layer-1 directional models from round-Q statistics (sub-matrix solves)
Zs = np.load("roundQ_stats.npz"); NAMES = list(Zs["names"]); dims = Zs["dims"]; off = np.cumsum(np.r_[0, dims]); mu0, sd0 = Zs["mu"], Zs["sd"]
idx = {k: np.arange(off[i], off[i + 1]) for i, k in enumerate(NAMES)}; D0 = Zs["G"].shape[0] - 1
GR = ["L", "R", "LR", "LL", "RR"]; gidx = {g: idx["P"][i * 129:(i + 1) * 129] for i, g in enumerate(GR)}
def half(k, fwd):
    ix = idx[k]; h = len(ix) // 2                      # every reservoir stores [forward half ; backward half]
    return ix[:h] if fwd else ix[h:]
def solve_m(cols, lam=1e-3):
    cols = np.r_[cols, D0]; Gs = Zs["G"][np.ix_(cols, cols)]; return cols, cho_solve(cho_factor(Gs + lam * np.trace(Gs) / len(cols) * np.eye(len(cols))), Zs["XYm"][cols]).astype(np.float32)
FW = solve_m(np.concatenate([gidx["L"], gidx["LL"]] + [half(k, True) for k in RES]))
BW = solve_m(np.concatenate([gidx["R"], gidx["RR"]] + [half(k, False) for k in RES]))
del Zs; gc.collect()
r = np.random.default_rng(9); U2 = 256
T2 = (r.standard_normal((2 * d, U2)) / np.sqrt(2 * d)).astype(np.float32); a2 = np.r_[r.uniform(0.1, 0.7, U2 // 2), r.uniform(0.8, 0.98, U2 - U2 // 2)].astype(np.float32)
Zp = np.load(f"{BIG}/pmi_W8.npz"); KA, KB = np.ascontiguousarray(Zp["A"]), np.ascontiguousarray(Zp["B"])
ids = load_corpus().astype(np.int64); n = len(ids); hi = int(0.9 * n); freq = token_freq(ids); w = np.sqrt(freq + 1e-9)[:, None]
MA = np.linalg.solve((E * w).T @ (E * w) + 1e-3 * np.eye(d), (E * w).T @ (KA * w)).astype(np.float32)
MB = np.linalg.solve((E * w).T @ (E * w) + 1e-3 * np.eye(d), (E * w).T @ (KB * w)).astype(np.float32)
rr = np.random.default_rng(0); s_, e_ = sentence_bounds(ids, 0, hi); ktr = np.sort(rr.choice(len(s_), 100_000, replace=False))
sv, ev = sentence_bounds(ids, hi, n); kva = np.sort(rr.choice(len(sv), 3000, replace=False))
train = Bank(toks=[ids[s_[i]:e_[i]] for i in ktr]); val = Bank(toks=[ids[sv[i]:ev[i]] for i in kva]); del ids; gc.collect()
keep_p = np.minimum(1.0, np.sqrt(1e-4 / (freq + 1e-12)) + 1e-4 / (freq + 1e-12)).astype(np.float32)
def base(t, st, loo):
    """per-token: pattern block, reservoir blocks, standardised with round-Q statistics"""
    sid = np.repeat(np.arange(len(st) - 1), np.diff(st)); allp = np.arange(len(t)); Pm = []
    for g, T in sc.tables.items(): M, lc = T.lookup(t, sid, allp, t if loo else None, E); Pm += [M, lc]
    X = np.zeros((len(t), D0), np.float32); X[:, idx["P"]] = np.concatenate(Pm, 1)
    for k in RES: X[:, idx[k]] = z.run(k, t, st)
    return (X - mu0) / sd0
def apply(Xs, cw): cols, W = cw; return Xs[:, cols[:-1]] @ W[:-1] + W[-1]
def seq_depth(t, st, Xs):
    """layer 1 directional guesses, then a layer-2 linear reservoir over [guess ; surprise] streams."""
    f1 = nz(apply(Xs, FW)); b1 = nz(apply(Xs, BW))
    # forward stream: drive at s = [f1_s ; E[t_s] - f1_s]  (what was expected, and how wrong it was). state BEFORE t.
    Df = np.c_[f1, E[t] - f1] @ T2; Db = np.c_[b1, E[t] - b1] @ T2
    Hf = np.empty((len(t), 2 * U2), np.float32); Hb = np.empty((len(t), 2 * U2), np.float32)
    # k_lin indexes drives by token id; here drives are per position -> pass positions as the 'tokens'
    pos = np.arange(len(t), dtype=np.int64); k_lin(pos, st, Df.astype(np.float32), a2, Hf); k_lin(pos, st, Db.astype(np.float32), a2, Hb)
    return np.c_[Hf[:, :U2], f1, Hb[:, U2:], b1]                                   # fwd-before, f1_t, bwd-after, b1_t
STAGES = {}
def prods(F, upto):
    """nonlinear expansion BETWEEN layers: previous guess x [its attention ; a random view of the memories]"""
    global RPROJ
    if RPROJ is None: RPROJ = (np.random.default_rng(42).standard_normal((F["BASE"].shape[1] - 645, 128)) / np.sqrt(F["BASE"].shape[1] - 645)).astype(np.float32)
    g = F[f"G{upto-1}"]; return XPD(g, np.c_[F[f"A{upto-1}"], nz(F["BASE"][:, 645:] @ RPROJ)])
def stage_feats(t, st, Xs, upto):
    """features for stage `upto` (1, 2, 3) and the sequence-depth block"""
    F = {"BASE": Xs[:, np.r_[idx["P"], np.concatenate([idx[k] for k in RES])]]}
    g_prev = None
    for k in range(1, upto):
        X_in = np.c_[F["BASE"]] if k == 1 else np.c_[F["BASE"], F[f"G{k-1}"], F[f"A{k-1}"]]
        W = STAGES[k]; g = nz((X_in - W["mu"]) / W["sd"] @ W["W"][:-1] + W["W"][-1]); F[f"G{k}"] = g
        o = np.empty((len(t), d), np.float32); query_attend(t, st, (g @ MA).astype(np.float32), (g @ MB).astype(np.float32), KA, KB, E, np.float32(0.2), np.float32(0.0), o); F[f"A{k}"] = nz(o)
    return F
def pass_stats(upto, with_seq):
    G = None; NTR = 0; rng = np.random.default_rng(1)
    for c in range(0, train.n, 800):
        s0, s1 = c, min(train.n, c + 800); lo, hi_ = train.starts[s0], train.starts[s1]; t = train.tok[lo:hi_].astype(np.int64); st = (train.starts[s0:s1 + 1] - lo).astype(np.int64)
        pos = np.flatnonzero(rng.random(len(t)) < keep_p[t]); Xs = base(t, st, True); F = stage_feats(t, st, Xs, upto)
        parts = [F["BASE"]] + ([F[f"G{upto-1}"], F[f"A{upto-1}"]] if upto > 1 else []) + ([prods(F, upto)] if with_seq else [])
        X = np.concatenate(parts, 1)[pos]; y = t[pos]; del Xs, F, parts
        if G is None: Dd = X.shape[1]; mu = X.mean(0); sd = X.std(0) + 1e-6; G = np.zeros((Dd + 1, Dd + 1)); XYm = np.zeros((Dd + 1, d)); XYv = np.zeros((V, Dd + 1))
        Xq = np.c_[(X - mu) / sd, np.ones(len(X), np.float32)].astype(np.float32); G += (Xq.T @ Xq).astype(np.float64); XYm += (Xq.T @ E[y]).astype(np.float64); NTR += len(y)
        o_ = np.argsort(y, kind='stable'); u_, s_i = np.unique(y[o_], return_index=True); XYv[u_] += np.add.reduceat(Xq[o_], s_i, 0); del Xq
        if NTR >= 200_000: break
    cf = cho_factor(G + 1e-3 * np.trace(G) / len(G) * np.eye(len(G))); W = cho_solve(cf, XYm).astype(np.float32)
    cf2 = cho_factor(G + 1e-4 * np.trace(G) / len(G) * np.eye(len(G))); Wv = cho_solve(cf2, XYv.T).astype(np.float32); del XYv, G
    return dict(W=W, Wv=Wv, mu=mu, sd=sd)
def evaluate_stage(upto, with_seq, W):
    rng = np.random.default_rng(2); ranks = []; vr = []
    for c in range(0, val.n, 500):
        s0, s1 = c, min(val.n, c + 500); lo, hi_ = val.starts[s0], val.starts[s1]; t = val.tok[lo:hi_].astype(np.int64); st = (val.starts[s0:s1 + 1] - lo).astype(np.int64)
        pos = np.flatnonzero(rng.random(len(t)) < keep_p[t]); Xs = base(t, st, False); F = stage_feats(t, st, Xs, upto)
        parts = [F["BASE"]] + ([F[f"G{upto-1}"], F[f"A{upto-1}"]] if upto > 1 else []) + ([prods(F, upto)] if with_seq else [])
        X = np.concatenate(parts, 1)[pos]; Xq = (X - W["mu"]) / W["sd"]; g = nz(Xq @ W["W"][:-1] + W["W"][-1]); S = g @ E.T; y = t[pos]
        ranks.append((S > S[np.arange(len(S)), y][:, None]).sum(1)); S = Xq @ W["Wv"][:-1] + W["Wv"][-1]; vr.append((S > S[np.arange(len(S)), y][:, None]).sum(1))
    rk = np.concatenate(ranks); vk = np.concatenate(vr); return float(np.mean(1 / (rk + 1))), float(np.mean(rk == 0)), float(np.mean(1 / (vk + 1))), float(np.mean(vk == 0))
t0 = time.time()
for upto, wp, nm in [(1, False, "stage 1"), (2, False, "stage 2 = stage-1 guess + its attention, stacked linearly"), (2, True, "stage 2 + NONLINEAR expansion between layers (guess x context products)")]:
    W = pass_stats(upto, wp); m, t1, vm, vt1 = evaluate_stage(upto, wp, W)
    if not wp: STAGES[upto] = W
    line = f"  {nm:72s} vocab MRR {vm:.3f} top1 {vt1*100:4.1f}% | meaning-space MRR {m:.3f}  [{time.time()-t0:.0f}s]"; tick(line); open(LOG, "a").write(line + "\n")
    if upto == 1: STAGES[1] = W
pickle.dump(dict(STAGES=STAGES, FW=FW, BW=BW, T2=T2, a2=a2, RES=RES), open("roundS_model.pkl", "wb"))
tick("done")
