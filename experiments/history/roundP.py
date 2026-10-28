"""Round P: which pattern groups are worth their memory? (sub-matrix solves on round-J statistics, no re-runs)
P block layout: groups L, R, LR, LL, RR, LLR, LRR, each = 128 mean + 1 log-count."""
import sys, numpy as np, time, gc
sys.path.insert(0, "/home/claude/big")
from scipy.linalg import cho_factor, cho_solve
T0 = time.time(); LOG = "roundP.log"; open(LOG, "w")
Z = np.load("roundJ_stats.npz"); G, XYm, XYv, Xv, yv, dims = Z["G"], Z["XYm"], Z["XYv"], Z["Xv"], Z["yv"], Z["dims"]
E = np.load("E.npy"); D = G.shape[0] - 1; nz = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
GR = ["L", "R", "LR", "LL", "RR", "LLR", "LRR"]; gi = {g: np.arange(i * 129, (i + 1) * 129) for i, g in enumerate(GR)}
off = np.cumsum(np.r_[0, dims]); LIN = np.arange(off[1], off[2]); SEL = np.arange(off[2], off[3])
keys_M = {"L": 0.01, "R": 0.01, "LR": 1.31, "LL": 0.98, "RR": 0.97, "LLR": 1.35, "LRR": 1.36}       # millions of stored pattern values
def solve(groups, extra, vocab=True, lams=(1e-4, 1e-3)):
    cols = np.r_[np.concatenate([gi[g] for g in groups] + extra), D]; Gs = G[np.ix_(cols, cols)]; Xs = Xv[:, cols]; dd = len(cols); res = {}
    best = None
    for lam in lams:
        cf = cho_factor(Gs + lam * np.trace(Gs) / dd * np.eye(dd)); Wm = cho_solve(cf, XYm[cols]).astype(np.float32)
        rk = np.concatenate([(lambda Gm, y_: (Gm > Gm[np.arange(len(Gm)), y_][:, None]).sum(1))(nz(Xs[c:c + 2000] @ Wm) @ E.T, yv[c:c + 2000]) for c in range(0, len(Xs), 2000)]); m = float(np.mean(1 / (rk + 1)))
        if best is None or m > best[0]: best = (m, lam, cf)
    res["mm"] = best[0]
    if vocab:
        W = cho_solve(best[2], XYv[:, cols].T).astype(np.float32); rk = []
        for c in range(0, len(Xs), 2000):
            S = Xs[c:c + 2000] @ W; rk.append((S > S[np.arange(len(S)), yv[c:c + 2000]][:, None]).sum(1))
        rk = np.concatenate(rk); res["vm"] = float(np.mean(1 / (rk + 1))); res["t1"] = float(np.mean(rk == 0)); del W
    mem = sum(keys_M[g] for g in groups)
    line = f"  patterns {'+'.join(groups):24s} {'+ LIN+SEL' if len(extra) else '         '} ({mem:4.2f}M values) meaning-MRR {res['mm']:.3f}" + (f"  vocab-MRR {res['vm']:.3f} top1 {res['t1']*100:4.1f}%" if vocab else "")
    print(f"[{time.time()-T0:4.0f}s] {line}", flush=True); open(LOG, "a").write(line + "\n"); gc.collect()
for groups in [GR, ["L", "R", "LR", "LL", "RR"], ["L", "R", "LR"], ["L", "R", "LR", "LLR", "LRR"], ["LR", "LL", "RR", "LLR", "LRR"], ["L", "R"]]:
    solve(groups, [LIN, SEL], vocab=groups in (GR, ["L", "R", "LR", "LL", "RR"], ["L", "R", "LR"]))
