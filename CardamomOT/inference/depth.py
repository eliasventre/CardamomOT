"""
Per-cell depth: library sizes, homogeneous groups and the diagnostic deciding whether a depth
factor s_i is needed (estimate_cell_depth.py). Counts are then modelled as NB(k, c / s_i): the
counts stay integers, k (basins, kon) is read at the reference depth of the cell's group.
"""
import numpy as np
import scipy.sparse

from .mixture import group_init, large_groups

# Genes needed to estimate a depth from the transcriptome (data_complete, or data if large enough)
MIN_GENES_DEPTH = 10000


def library(X, n_top=50):
    """Per-cell total counts without the n_top most expressed genes (robust to dominant transcripts)."""
    X = scipy.sparse.csr_matrix(X) if not scipy.sparse.issparse(X) else X.tocsr()
    tot = np.asarray(X.sum(axis=0)).ravel()
    keep = np.ones(X.shape[1], bool)
    keep[np.argsort(-tot)[:n_top]] = False
    return np.asarray(X[:, keep].sum(axis=1)).ravel().astype(float)


def depth_groups(times, samples=None, cell_types=None):
    """Labels (sample|time|cell type) of the homogeneous groups; cell types too small within their
    (sample, time) (large_groups) are pooled with the rest of it."""
    n = len(times)
    samples = np.zeros(n) if samples is None else np.asarray(samples)
    base = np.array([f'{s}|{t:g}' for s, t in zip(samples, np.asarray(times, dtype=float))], dtype=object)
    if cell_types is None:
        return base
    cts = np.asarray(cell_types).astype(str)
    out = base.copy()
    for b in np.unique(base):
        m = np.flatnonzero(base == b)
        big = large_groups(cts[m])
        out[m] = np.where(big, base[m] + '|' + cts[m], base[m] + '|other')
    return out


def _corr_share(Y, z, groups, min_cells=200):
    """Mean within-group pairwise correlation of log1p counts, raw and with log depth regressed out."""
    raw = res = w = 0.0
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < min_cells:
            continue
        L = np.log1p(Y[m]); L = L - L.mean(axis=0)
        zz = z[m] - z[m].mean()
        R = L - np.outer(zz, (zz @ L) / max(zz @ zz, 1e-12))
        for M, acc in ((L, 'raw'), (R, 'res')):
            ok = M.std(axis=0) > 0
            C = np.corrcoef(M[:, ok].T)
            v = np.mean(C[np.triu_indices(C.shape[0], 1)])
            if acc == 'raw':
                raw += v * m.sum()
            else:
                res += v * m.sum()
        w += m.sum()
    return (raw / w, res / w) if w else (np.nan, np.nan)


def diagnose_depth(X, lib, groups, times, cell_types=None, s=None, doublet=None, n_ref=1000,
                   n_cells=1024, seuil=0.01, seed=0, corr_max=0.5, close_max=0.5):
    """
    Depth diagnostic on raw counts X (cells x genes, whole transcriptome).
    s: depth factors of the chosen method (group_median if None) giving the spread within groups.
    Reference genes: the n_ref most variable genes (log1p of x / s) expressed in > 5% of the cells,
    on at most n_cells cells per group. Returns (per-group DataFrame, dict of global indicators,
    with 'recommended' = share of the within-group correlation due to depth > corr_max, or share of
    genes whose NB-initialization modes are closer than the depth spread > close_max).
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    if s is None:
        from .depth_methods.group_median import compute
        s = compute(None, lib, groups)
    # Extrinsic noise of each group (Fang & Pachter: mode of the normalized covariances, refined on
    # the Poisson genes), whatever the depth method
    from .depth_methods.poissonian import _group_poisson
    rows = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 20:
            continue
        noise, n_poisson = np.nan, 0
        if m.sum() >= 50:
            idx = rng.choice(np.flatnonzero(m), min(n_cells, int(m.sum())), replace=False)
            Yg = X[idx]
            Yg = Yg.toarray().astype(float) if scipy.sparse.issparse(Yg) else np.asarray(Yg, dtype=float)
            noise, P = _group_poisson(Yg, rng, 0.1, 50, 500, 5)
            n_poisson = int(len(P))
        rows.append(dict(group=g, n_cells=int(m.sum()), median_counts=float(np.median(lib[m])),
                         cv=float(lib[m].std() / max(lib[m].mean(), 1e-12)),
                         s_q05=float(np.quantile(s[m], .05)), s_q95=float(np.quantile(s[m], .95)),
                         extrinsic_noise=float(noise), n_poisson_genes=n_poisson))
    per_group = pd.DataFrame(rows)
    spread = float(np.median(per_group.s_q95 / per_group.s_q05)) if len(per_group) else np.nan

    # Reference genes and cells
    sub = np.concatenate([rng.choice(np.flatnonzero(groups == g), min(n_cells, int(np.sum(groups == g))), replace=False)
                          for g in np.unique(groups)])
    Xs = X[sub]
    Xs = Xs.toarray() if scipy.sparse.issparse(Xs) else np.asarray(Xs)
    expressed = (Xs > 0).mean(axis=0) > 0.05
    Ln = np.log1p(Xs[:, expressed] / s[sub, None])
    ref = np.flatnonzero(expressed)[np.argsort(-Ln.var(axis=0))[:n_ref]]
    Y = Xs[:, ref].astype(float)
    z = np.log(np.maximum(lib[sub], 1))
    c_raw, c_res = _corr_share(Y, z, groups[sub])
    share = float((c_raw - c_res) / abs(c_raw)) if np.isfinite(c_raw) and c_raw != 0 else np.nan
    gene_corr = []
    for g in np.unique(groups[sub]):
        m = groups[sub] == g
        if m.sum() < 200:
            continue
        L = np.log1p(Y[m])
        gene_corr.append([np.corrcoef(z[m], L[:, j])[0, 1] if L[:, j].std() > 0 else np.nan for j in range(L.shape[1])])
    gene_corr = float(np.nanmedian(np.nanmedian(np.array(gene_corr), axis=0))) if gene_corr else np.nan

    # NB-initialization modes (times / cell types) vs the depth spread
    labels = [np.asarray(times)[sub], None if cell_types is None else np.asarray(cell_types)[sub]]
    ratios = []
    for j in range(Y.shape[1]):
        x = np.rint(Y[:, j]).astype(int)
        res = group_init(x, labels, seuil) if x.min() < x.max() else None
        if res is not None and res[0] > 0:
            ratios.append(res[1] / res[0])
    ratios = np.array(ratios)
    close = float(np.mean(ratios < spread)) if len(ratios) else np.nan

    r = lib / max(lib.mean(), 1e-12)
    out = dict(n_cells=int(len(lib)), n_reference_genes=int(len(ref)), depth_spread_q95_q05=spread,
               median_cv=float(per_group.cv.median()) if len(per_group) else np.nan,
               median_extrinsic_noise=float(np.nanmedian(per_group.extrinsic_noise)) if len(per_group) else np.nan,
               corr_raw=float(c_raw), corr_without_depth=float(c_res), corr_share_depth=share,
               median_gene_depth_corr=gene_corr, median_mode_ratio=float(np.median(ratios)) if len(ratios) else np.nan,
               share_modes_closer_than_spread=close,
               extrinsic_phi=float((r.var() - r.mean() / max(lib.mean(), 1e-12)) / max(r.mean() ** 2, 1e-12)))
    if doublet is not None:
        d = np.asarray(doublet, dtype=float)
        cc = [np.corrcoef(d[groups == g], np.log(np.maximum(lib[groups == g], 1)))[0, 1]
              for g in np.unique(groups) if np.sum(groups == g) >= 50 and np.std(d[groups == g]) > 0]
        out['doublet_depth_corr'] = float(np.nanmedian(cc)) if cc else np.nan
    out['recommended'] = bool((np.isfinite(share) and share > corr_max) or (np.isfinite(close) and close > close_max))
    return per_group, out


def state_depth(depth_cells, real_idx):
    """Depth factor of each trajectory state: that of the real cell behind it (traj_real_idx; 1 if
    none). None without depth factors."""
    if depth_cells is None or real_idx is None:
        return None
    real_idx = np.asarray(real_idx, dtype=int)
    s = np.ones(len(real_idx))
    ok = (real_idx >= 0) & (real_idx < len(depth_cells))
    s[ok] = np.asarray(depth_cells, dtype=float)[real_idx[ok]]
    return s


def simulation_depth(depth_cells, real_idx, times_states, times_sim):
    """
    Depth factor of each simulated cell: simulated trajectory j mimics trajectory j of the
    inference, so at each simulated time it takes the depth of the real cell behind state j at the
    nearest inferred time. None without depth factors.
    """
    s_states = state_depth(depth_cells, real_idx)
    if s_states is None:
        return None
    tu = np.sort(np.unique(times_states))
    N = len(s_states) // len(tu)
    ts = np.asarray(times_sim, dtype=float)
    out = np.ones(len(ts))
    for t in np.unique(ts):
        rows = np.flatnonzero(ts == t)
        ti = int(np.argmin(np.abs(tu - t)))
        out[rows] = np.resize(s_states[ti * N:(ti + 1) * N], len(rows))
    return out
