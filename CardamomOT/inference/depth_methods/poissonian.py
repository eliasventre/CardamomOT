"""
Cell size factors from "Poisson" genes (Fang & Pachter, Extrinsic biological stochasticity and
technical noise normalization of scRNA-seq data, bioRxiv 2025; github.com/pachterlab/FP_2025),
computed within homogeneous groups (sample, time, cell type).

Under the extrinsic noise model (counts ~ Bernoulli sampling of in vivo counts whose means scale
with a cell factor c), the normalized covariance Cov(Xa, Xb) / (E[Xa] E[Xb]) of intrinsically
independent genes and the normalized variance (Var[X] - E[X]) / E[X]^2 of intrinsically Poisson
genes both equal the extrinsic noise s = Var[c] / E[c]^2. Procedure:
1. s = mode (bins of 0.01) of the normalized covariances between genes of mean > min_mean;
2. "Poisson" genes = those whose 95% bootstrap interval of normalized variance contains s;
   s is re-estimated on them, until it changes by less than 10%;
3. size factor c_i = sum over Poisson genes of X_ij / sum of their means (Poisson MLE).
Unlike the paper (one population), the procedure is run within each homogeneous group, whose
extrinsic noise may differ (e.g. proliferating vs quiescent cells), so that differences between
cell types or times do not inflate the covariances; c_i is relative to the means of the cell's
group (reference='group', as group_median), or to the global means on the union of the groups'
Poisson genes (reference='global'). Groups of fewer than 50 cells keep c = 1.
"""
import numpy as np
import scipy.sparse

DEFAULTS = dict(min_mean=0.1, n_cells=1024, n_boot=100, max_genes_cov=500, max_iter=5, reference='group',
                seed=0)
BINS = np.arange(-0.5, 1.5, 0.01) - 0.005


def _mode(eta):
    hist, edges = np.histogram(eta, bins=BINS)
    return float((edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2)


def _group_poisson(Y, rng, min_mean, n_boot, max_genes_cov, max_iter):
    """Paper procedure on the cells of one group: (extrinsic noise s, indices of Poisson genes)."""
    mu = Y.mean(axis=0)
    cand = np.flatnonzero(mu > min_mean)
    if len(cand) < 10:
        return np.nan, cand

    def noise(genes):
        g = genes if len(genes) <= max_genes_cov else rng.choice(genes, max_genes_cov, replace=False)
        C = np.cov(Y[:, g], rowvar=False) / np.outer(mu[g], mu[g])
        return _mode(C[np.triu_indices(len(g), 1)])

    Yc = Y[:, cand]
    Wb = rng.poisson(1.0, (n_boot, Y.shape[0])).astype(float)
    Wb /= np.maximum(Wb.sum(axis=1, keepdims=True), 1)
    m1, m2 = Wb @ Yc, Wb @ (Yc ** 2)
    boot = (m2 - m1 ** 2 - m1) / np.maximum(m1, 1e-12) ** 2
    lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
    s, poisson = noise(cand), cand
    for _ in range(max_iter):
        sel = cand[(lo <= s) & (s <= hi)]
        if len(sel) < 10:
            break
        poisson = sel
        s_new = noise(poisson)
        converged = abs(s_new - s) <= 0.1 * max(abs(s), 1e-12)
        s = s_new
        if converged:
            break
    return s, poisson


def compute(X, lib, groups, min_mean=0.1, n_cells=1024, n_boot=100, max_genes_cov=500, max_iter=5,
            reference='group', seed=0):
    rng = np.random.default_rng(seed)
    X = X.tocsr() if scipy.sparse.issparse(X) else scipy.sparse.csr_matrix(X)
    c = np.ones(X.shape[0])
    union, report = set(), []
    for g in np.unique(groups):
        m = np.flatnonzero(groups == g)
        if len(m) < 50:
            continue
        sub = rng.choice(m, min(n_cells, len(m)), replace=False)
        s_g, P = _group_poisson(X[sub].toarray().astype(float), rng, min_mean, n_boot, max_genes_cov, max_iter)
        if len(P) < 10:
            continue
        union.update(P.tolist())
        report.append((g, s_g, len(P)))
        if reference != 'global':
            # Poisson MLE of the cell size on the group's Poisson genes, relative to the group means
            XP = X[m][:, P]
            c[m] = np.asarray(XP.sum(axis=1)).ravel() / max(np.asarray(XP.mean(axis=0)).ravel().sum(), 1e-12)
    if reference == 'global' and union:
        P = np.array(sorted(union))
        XP = X[:, P]
        c = np.asarray(XP.sum(axis=1)).ravel() / max(np.asarray(XP.mean(axis=0)).ravel().sum(), 1e-12)
    for g, s_g, n in report:
        print(f"[poissonian depth] {g}: extrinsic noise s = {s_g:.2f}, {n} Poisson genes")
    return np.maximum(c, 1e-3)
