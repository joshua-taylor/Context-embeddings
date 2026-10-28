"""Solve every block combination from the saved statistics of round J (no tables in memory)."""
import sys, numpy as np, time, gc
sys.path.insert(0, "/home/claude/big")
from scipy.linalg import cho_factor, cho_solve
T0 = time.time(); LOG = sys.argv[1] if len(sys.argv) > 1 else "roundJ.log"
STATS = sys.argv[3] if len(sys.argv) > 3 else "roundJ_stats.npz"; Z = np.load(STATS); G, XYm, XYv, Xv, yv, dims = Z["G"], Z["XYm"], Z["XYv"], Z["Xv"], Z["yv"], Z["dims"]
E = np.load("E.npy"); D = G.shape[0] - 1
ORDER = list(Z["names"]) if "names" in Z.files else ["P", "LIN", "SEL", "HIP", "ESN", "LINX"]; off = np.cumsum(np.r_[0, dims]); idx = {k: np.arange(off[i], off[i + 1]) for i, k in enumerate(ORDER)}
nz = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
def solve(names, lams=(1e-4, 1e-3)):
    cols = np.r_[np.concatenate([idx[k] for k in names]), D]; Gs = G[np.ix_(cols, cols)]; Xs = Xv[:, cols]; dd = len(cols); best = None
    for lam in lams:
        cf = cho_factor(Gs + lam * np.trace(Gs) / dd * np.eye(dd)); rk = []
        W = cho_solve(cf, XYv[:, cols].T, overwrite_b=True).astype(np.float32); gc.collect()
        for c in range(0, len(Xs), 2000):
            S = Xs[c:c + 2000] @ W; rk.append((S > S[np.arange(len(S)), yv[c:c + 2000]][:, None]).sum(1))
        rk = np.concatenate(rk); mrr = float(np.mean(1 / (rk + 1)))
        if best is None or mrr > best[0]: best = (mrr, float(np.mean(rk == 0)), float(np.mean(rk < 10)), lam)
        del W, cf; gc.collect()
    Wm = cho_solve(cho_factor(Gs + best[3] * np.trace(Gs) / dd * np.eye(dd)), XYm[cols]).astype(np.float32)
    rk = np.concatenate([(lambda Gm, y_: (Gm > Gm[np.arange(len(Gm)), y_][:, None]).sum(1))(nz(Xs[c:c + 2000] @ Wm) @ E.T, yv[c:c + 2000]) for c in range(0, len(Xs), 2000)])
    line = f"  {'+'.join(names):44s} ({dd-1:5d} feats)  vocab MRR {best[0]:.3f} top1 {best[1]*100:5.1f}% top10 {best[2]*100:5.1f}% | meaning-space MRR {np.mean(1/(rk+1)):.3f}"
    print(f"[{time.time()-T0:5.0f}s] {line}", flush=True); open(LOG, "a").write(line + "\n")
COMBOS = [["LIN"], ["SEL"], ["HIP"], ["ESN"], ["LINX"], ["LIN", "SEL", "HIP", "ESN", "LINX"],
          ["P"], ["P", "LIN"], ["P", "SEL"], ["P", "HIP"], ["P", "ESN"], ["P", "LINX"],
          ["P", "LIN", "HIP"], ["P", "LIN", "ESN"], ["P", "LIN", "SEL"], ["P", "HIP", "ESN"], ["P", "LIN", "SEL", "HIP", "ESN", "LINX"]]
for c in (COMBOS if len(sys.argv) < 3 else [x.split("+") for x in sys.argv[2].split(",")]): solve(c)
