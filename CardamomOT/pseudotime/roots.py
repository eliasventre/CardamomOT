"""
Root cells of the pseudotime (PROTOTYPE_PLAN.md §2.1).

1. **User-defined**: the boolean (or 0/1) obs column ``key`` (default ``'is_root'``).
2. **Fallback** (a suboptimal default): a fraction (2 %) of the cells at one end of the first
   diffusion component of the expression kNN graph.  The end is the one with the lowest mean
   log-expression (``side='auto'``: in a stimulus response, the cells before the stimulus are the
   least expressed), or chosen by the user (``side='low'`` / ``'high'`` end of the component).

Palantir needs roots before any pseudotime exists, which is why the fallback cannot use the
pseudotime itself.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

ROOT_SIDES = ("auto", "low", "high")


def first_diffusion_component(counts: np.ndarray, n_neighbors: int = 30, n_pcs: int = 20,
                              seed: int = 0) -> np.ndarray:
    """
    First non-trivial eigenvector of the diffusion operator of the kNN graph on log1p counts
    (adaptive Gaussian kernel, symmetric normalisation), shape (N,).  Its sign is arbitrary.
    """
    x = np.log1p(np.asarray(counts, dtype=float))
    n = x.shape[0]
    if n < 3:
        raise ValueError("at least 3 cells are needed")
    x = x - x.mean(axis=0)
    if x.shape[1] > n_pcs:
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        x = u[:, :n_pcs] * s[:n_pcs]
    k = int(min(max(n_neighbors, 2), n - 1))
    dist, ind = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)
    dist, ind = dist[:, 1:], ind[:, 1:]
    sigma = np.maximum(dist[:, k // 2], 1e-12)                      # adaptive bandwidth
    w = np.exp(-(dist ** 2) / (sigma[:, None] * sigma[ind]))
    rows = np.repeat(np.arange(n), k)
    W = csr_matrix((w.ravel(), (rows, ind.ravel())), shape=(n, n))
    W = W.maximum(W.T)                                              # symmetric graph
    deg = np.asarray(W.sum(axis=1)).ravel()
    d_inv = diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    A = d_inv @ W @ d_inv                                           # symmetric normalisation
    rng = np.random.default_rng(seed)
    vals, vecs = eigsh(A, k=2, which="LA", v0=rng.standard_normal(n))
    order = np.argsort(vals)[::-1]
    psi = vecs[:, order[1]] / np.sqrt(np.maximum(deg, 1e-12))      # right eigenvector
    return psi


def select_roots(data, key: Optional[str] = "is_root", fraction: float = 0.02, side: str = "auto",
                 n_neighbors: int = 30, seed: int = 0) -> Tuple[np.ndarray, dict]:
    """
    Boolean mask of the root cells and a description of the choice.

    ``data``: AnnData (``obs[key]`` read when present, ``X`` used by the fallback).  Returns
    ``(mask, info)`` with ``info['source']`` in ``{'user', 'fallback'}``.
    """
    obs = getattr(data, "obs", None)
    if key and obs is not None and key in obs:
        flag = np.asarray(obs[key])
        mask = flag.astype(bool)
        if not mask.any():
            raise ValueError(f"obs['{key}'] marks no root cell")
        return mask, dict(source="user", key=key, n_roots=int(mask.sum()))

    if side not in ROOT_SIDES:
        raise ValueError(f"side must be one of {ROOT_SIDES}, got {side!r}")
    X = data.X
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=float)
    n = X.shape[0]
    n_roots = int(max(1, np.ceil(float(fraction) * n)))
    psi = first_diffusion_component(X, n_neighbors=n_neighbors, seed=seed)
    order = np.argsort(psi, kind="stable")
    low, high = order[:n_roots], order[::-1][:n_roots]
    expr = np.log1p(X).mean(axis=1)
    if side == "auto":
        chosen = "low" if expr[low].mean() <= expr[high].mean() else "high"
    else:
        chosen = side
    mask = np.zeros(n, dtype=bool)
    mask[low if chosen == "low" else high] = True
    info = dict(source="fallback", fraction=float(fraction), n_roots=n_roots, side=chosen,
                mean_log_expression=dict(low=float(expr[low].mean()), high=float(expr[high].mean())))
    return mask, info
