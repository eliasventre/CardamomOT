"""
Cell size factors from "Poisson" genes (Fang & Pachter, Extrinsic biological stochasticity and
technical noise normalization of scRNA-seq data, bioRxiv 2025; github.com/pachterlab/FP_2025).

Under the extrinsic noise model (counts ~ Bernoulli sampling of in vivo counts whose means scale
with a cell factor c), the normalized covariance Cov(Xa, Xb) / (E[Xa] E[Xb]) of intrinsically
independent genes and the normalized variance (Var[X] - E[X]) / E[X]^2 of intrinsically Poisson
genes both equal the extrinsic noise s = Var[c] / E[c]^2. Procedure on a population of cells:
1. s = mode (bins of 0.01) of the normalized covariances between genes of mean > min_mean;
2. "Poisson" genes = those whose 95% bootstrap interval of normalized variance contains s;
   s is re-estimated on them, until it changes by less than 10%;
3. size factor c_i = sum over Poisson genes of X_ij / sum of their means (Poisson MLE).

reference (groups = 'sample|time[|cell type]' labels, sample = part before the first '|'):
- 'sample_any': run within each group (time-resolved): c relative to the means of the cell's group (mean 1 per group;
  depth differences between the groups, e.g. separate libraries, are not corrected);
- 'sample_all': run on the cells of the whole sample at once, as in the paper (one population): Poisson genes stable
  over all its times, c relative to the sample means (few genes when the sample is heterogeneous);
- 'sample_pairwise' (default): the 'sample_any' factors times a scale per group, lambda_g^alpha: for each pair of
  groups of the sample, r = median over the genes Poisson in both of the log ratio of their means, and log lambda
  fitted by weighted least squares on log lambda_g - log lambda_h = r (weights: shared genes; mean of log lambda 0
  in the sample; with every pair, the mean of the ratios to the other groups). alpha in [0, 1]: 0 = 'sample_any',
  1 = full correction of the depth differences between the groups.
Groups (or samples) of fewer than 50 cells keep c = 1.
"""
import numpy as np
import scipy.sparse

DEFAULTS = dict(reference='sample_pairwise', alpha=1.0, min_mean=0.1, n_cells=1024, n_boot=100, max_genes_cov=500,
                max_iter=5, seed=0)
REFERENCES = ('sample_any', 'sample_all', 'sample_pairwise')
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


def pairwise_scales(X, groups, members, genes, min_shared=10):
    """
    log lambda of the groups `members` (one sample): r(g, h) = median over the genes Poisson in g and h of
    log(mean in g / mean in h), then weighted least squares of log lambda_g - log lambda_h = r (weights: number of
    shared genes), mean of log lambda 0. Groups linked to no other keep 0. Returns ({group: log lambda}, pairs used).
    """
    rows, rhs, w, used = [], [], [], []
    idx = {g: k for k, g in enumerate(members)}
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            g, h = members[a], members[b]
            if g not in genes or h not in genes:
                continue
            P = np.array(sorted(set(genes[g].tolist()) & set(genes[h].tolist())))
            if len(P) < min_shared:
                continue
            mg = np.asarray(X[np.flatnonzero(groups == g)][:, P].mean(axis=0)).ravel()
            mh = np.asarray(X[np.flatnonzero(groups == h)][:, P].mean(axis=0)).ravel()
            ok = (mg > 0) & (mh > 0)
            if ok.sum() < min_shared:
                continue
            row = np.zeros(len(members))
            row[idx[g]], row[idx[h]] = 1.0, -1.0
            rows.append(row)
            rhs.append(float(np.median(np.log(mg[ok] / mh[ok]))))
            w.append(float(ok.sum()))
            used.append((g, h, int(ok.sum())))
    out = {g: 0.0 for g in members}
    if not rows:
        return out, used
    A, y, sw = np.array(rows), np.array(rhs), np.sqrt(np.array(w))
    linked = np.flatnonzero(np.abs(A).sum(axis=0) > 0)
    C = np.zeros(len(members))
    C[linked] = 1.0   # mean of log lambda over the linked groups set to 0 (heavily weighted constraint row)
    sol = np.linalg.lstsq(np.vstack([A * sw[:, None], 1e3 * C])[:, linked], np.r_[y * sw, 0.0], rcond=None)[0]
    for k, j in enumerate(linked):
        out[members[j]] = float(sol[k])
    return out, used


def _size(X, rows, P):
    """Poisson MLE of the cell sizes of `rows` on the genes P, relative to their means over these cells."""
    XP = X[rows][:, P]
    return np.asarray(XP.sum(axis=1)).ravel() / max(np.asarray(XP.mean(axis=0)).ravel().sum(), 1e-12)


def compute(X, lib, groups, reference='sample_pairwise', alpha=1.0, min_mean=0.1, n_cells=1024, n_boot=100,
            max_genes_cov=500, max_iter=5, seed=0):
    if reference not in REFERENCES:
        raise ValueError(f"poissonian depth: reference '{reference}' (use one of {REFERENCES})")
    rng = np.random.default_rng(seed)
    X = X.tocsr() if scipy.sparse.issparse(X) else scipy.sparse.csr_matrix(X)
    groups = np.asarray(groups)
    smp = np.array([str(g).split('|')[0] for g in groups])
    c = np.ones(X.shape[0])
    args = (rng, min_mean, n_boot, max_genes_cov, max_iter)
    if reference == 'sample_all':
        # One population per sample: at most n_cells per group of the sample, pooled
        for s_ in np.unique(smp):
            m = np.flatnonzero(smp == s_)
            if len(m) < 50:
                continue
            sub = np.concatenate([rng.choice(np.flatnonzero(groups == g), min(n_cells, int(np.sum(groups == g))),
                                             replace=False) for g in np.unique(groups[m])])
            s_s, P = _group_poisson(X[sub].toarray().astype(float), *args)
            print(f"[poissonian depth] sample {s_} (sample_all): extrinsic noise s = {s_s:.2f}, {len(P)} Poisson genes")
            if len(P) >= 10:
                c[m] = _size(X, m, P)
        return np.maximum(c, 1e-3)
    # Within each group (sample_any), then the scales between the groups of each sample (sample_pairwise)
    genes = {}
    for g in np.unique(groups):
        m = np.flatnonzero(groups == g)
        if len(m) < 50:
            continue
        sub = rng.choice(m, min(n_cells, len(m)), replace=False)
        s_g, P = _group_poisson(X[sub].toarray().astype(float), *args)
        if len(P) < 10:
            continue
        genes[g] = P
        c[m] = _size(X, m, P)
        print(f"[poissonian depth] {g}: extrinsic noise s = {s_g:.2f}, {len(P)} Poisson genes")
    if reference == 'sample_pairwise':
        for s_ in np.unique(smp):
            members = [g for g in np.unique(groups[smp == s_]) if g in genes]
            loglam, used = pairwise_scales(X, groups, members, genes)
            for g, ll in loglam.items():
                c[groups == g] *= np.exp(alpha * ll)
            print(f"[poissonian depth] sample {s_} (sample_pairwise, alpha {alpha:g}, {len(used)} pairs, median "
                  f"{np.median([u[2] for u in used]) if used else 0:.0f} shared genes): scales "
                  + ', '.join(f"{g.split('|', 1)[1] if '|' in g else g} {np.exp(ll):.2f}" for g, ll in loglam.items()))
    return np.maximum(c, 1e-3)
