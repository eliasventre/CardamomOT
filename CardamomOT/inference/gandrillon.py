"""
Entropy-based gene selection of the From_fastq_to_CardamomOT workflow (O. Gandrillon,
github.com/ogandril/From_fastq_to_CardamomOT, Define_genes.ipynb), on raw counts:

- KD: 1-D Wasserstein distance of the counts of each gene between consecutive timepoints;
- MDE: absolute change of the discrete entropy of the counts between consecutive timepoints,
  the entropy being estimated per timepoint with the BUB estimator (Paninski 2003, R code of
  M. Gaillard ported here);
- selected genes: union over transitions of the top_n KD genes, intersected with the union over
  transitions of the top_n MDE genes.
"""
from functools import lru_cache
import numpy as np
from scipy.special import gammaln
from scipy.stats import wasserstein_distance


@lru_cache(maxsize=4)
def _bernstein_matrix(N):
    """B[j, k] = C(N, j) p_k^j (1 - p_k)^(N - j) at p_k = k / N (mat_bernouilli_polynom)."""
    p = np.arange(N + 1) / N
    B = np.zeros((N + 1, N + 1))
    j = np.arange(N + 1)[:, None]
    lc = gammaln(N + 1) - gammaln(j + 1) - gammaln(N - j + 1)
    pin = p[1:-1][None, :]
    B[:, 1:-1] = np.exp(lc + j * np.log(pin) + (N - j) * np.log(1 - pin))
    B[0, 0] = B[N, N] = 1.0
    return B


@lru_cache(maxsize=4096)
def _bub_coefficients(N, m, k_max=11):
    """BUB coefficients a_j (j = 0..N) for N samples and m bins (bub_opti)."""
    k_max = min(k_max, N)
    B = _bernstein_matrix(N)
    p = np.arange(N + 1) / N
    g = np.where(p < 1.0 / m, float(m), 1.0 / np.maximum(p, 1e-300))
    Y_tot = np.concatenate([[0.0], -p[1:] * np.log(p[1:])])
    a = p.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.nan_to_num(-a * np.log(a)) + (1 - a) / (2 * N)
    best_MM, best_a = np.inf, a.copy()
    for i in range(1, k_max + 1):
        h_exp = a[i:] @ B[i:, :]
        Y = g * (Y_tot - h_exp)
        X = (g[None, :] * B[:i, :]).T                       # (N+1, i)
        D = -np.eye(i) + np.eye(i, k=1)
        U = X.T @ X + (N / 4) * (D.T @ D)
        U[i - 1, i - 1] += N / 4
        XY = X.T @ Y
        XY[i - 1] += (N / 4) * a[i]
        a[:i] = np.linalg.pinv(U) @ XY
        bias = np.max(np.abs(2 * g * (Y_tot - a @ B)))
        var_bound = np.max((a - np.append(a[1:], 0.0)) ** 2)
        MM = np.sqrt(bias ** 2 + N * var_bound)
        if MM < best_MM:
            best_MM, best_a = MM, a.copy()
    return best_a


def bub_entropy(x, k_max=11):
    """BUB estimate (nats) of the entropy of the discrete distribution of integer counts x."""
    x = np.asarray(x).astype(int)
    N = len(x)
    h = np.bincount(x)                       # cells per count value, values 0..max
    hist_eff = np.bincount(h, minlength=N + 1)[:N + 1]   # number of values seen j times
    return float(_bub_coefficients(N, len(h), k_max) @ hist_eff)


def gandrillon_scores(X, times, n_cells=1000, rng=None):
    """
    KD and MDE scores per transition (T-1, G) of raw counts X (cells x genes); entropies use at
    most n_cells cells per timepoint (the BUB matrices are (N+1)^2).
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X)
    tu = np.sort(np.unique(times))
    idx = [np.flatnonzero(times == t) for t in tu]
    H = []
    for i in idx:
        sub = rng.choice(i, n_cells, replace=False) if len(i) > n_cells else i
        Xs = np.rint(X[sub]).astype(int)
        H.append([bub_entropy(Xs[:, g]) for g in range(X.shape[1])])
    H = np.array(H)
    mde = np.abs(np.diff(H, axis=0))
    kd = np.array([[wasserstein_distance(X[i0, g], X[i1, g]) for g in range(X.shape[1])]
                   for i0, i1 in zip(idx[:-1], idx[1:])])
    return kd, mde, H


def gandrillon_genes(kd, mde, top_n=200):
    """Indices of genes in the union of the per-transition top_n KD genes AND of the per-transition top_n MDE genes."""
    top = lambda S: set(np.concatenate([np.argsort(-row)[:top_n] for row in S])) if len(S) else set()
    return np.array(sorted(top(kd) & top(mde)), dtype=int)
