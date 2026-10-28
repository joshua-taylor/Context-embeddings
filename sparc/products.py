"""Nonlinear expansion between layers: interactions between a layer's guess g and its context c.

RandomProducts   : (g Wa) * (c Wb), D random directions each.   Cost (dg + dc) * D per token.
SubspaceProducts : EXACT outer product of g and c restricted to their top principal directions,
                   (g Pa)[k] x (c Pb)[m] -> k*m features.        Cost dg*k + dc*m + k*m per token
                   (about 35x cheaper than random products at the same width). Directions are fitted in closed
                   form (PCA) on the training rows, so every feature is an interpretable 'guess-axis x context-axis'
                   interaction instead of a random mixture.
"""
import numpy as np


class RandomProducts:
    def __init__(self, dg, dc, D, seed):
        r = np.random.default_rng(seed); self.D = D
        self.Wa = (r.standard_normal((dg, D)) / np.sqrt(dg)).astype(np.float32)
        self.Wb = (r.standard_normal((dc, D)) / np.sqrt(dc)).astype(np.float32)

    def fit(self, g, c):
        return self

    def __call__(self, g, c):
        return (g @ self.Wa) * (c @ self.Wb)


class SubspaceProducts:
    def __init__(self, k=32, m=64):
        self.k, self.m = k, m; self.D = k * m

    def fit(self, g, c):
        def pca(X, r):
            mu = X.mean(0); _, S, Vt = np.linalg.svd((X - mu).astype(np.float64), full_matrices=False)
            return mu.astype(np.float32), (Vt[:r].T / (S[:r] / np.sqrt(len(X)) + 1e-6)).astype(np.float32)   # whitened axes
        self.mg, self.Pa = pca(g, self.k); self.mc, self.Pb = pca(c, self.m); return self

    def __call__(self, g, c):
        a = (g - self.mg) @ self.Pa; b = (c - self.mc) @ self.Pb
        return (a[:, :, None] * b[:, None, :]).reshape(len(g), self.D)


def make_products(kind, dg, dc, width, seed):
    if kind == "random": return RandomProducts(dg, dc, width, seed)
    k = int(round(np.sqrt(width / 2))); return SubspaceProducts(k, width // k)
