"""
Classical optimal-transport analysis (Waddington-OT / moscot style), run before CardamomOT on every gene.

Per sample and pair of consecutive timepoints: entropic unbalanced OT (Chizat et al. scaling algorithm) on
the squared Euclidean distances of a PCA of log1p(counts / depth factor), divided by their median; source
marginal ∝ exp(r Δt) with r the net proliferation rate (obs['proliferation_net_rate']) when available,
relaxed (lambda1) so that cells may grow or die, target marginal nearly strict (lambda2). Defaults of WOT:
epsilon = 0.05, lambda1 = 1, lambda2 = 50.

From the couplings: cell-type transitions, genes differentially expressed in the ancestors of each cell
type (its immediate ancestors against those of the other cells, weighted Welch t) next to the genes
differentially expressed in the cell type itself; velocities drawn by the report (tools/velocity.py).
"""
import numpy as np
import pandas as pd
import scipy.sparse

EPSILON, LAMBDA1, LAMBDA2 = 0.05, 1.0, 50.0


def unbalanced_sinkhorn(a, b, C, eps=EPSILON, lam1=LAMBDA1, lam2=LAMBDA2, n_iter=3000, tol=1e-7):
    """Entropic unbalanced OT plan diag(u) K diag(v), K = exp(-C / eps) (scaling algorithm, KL marginal penalties)."""
    K = np.exp(-C / eps)
    u, v = np.ones(len(a)), np.ones(len(b))
    f1, f2 = lam1 / (lam1 + eps), lam2 / (lam2 + eps)
    for _ in range(n_iter):
        u_prev = u
        u = (a / np.maximum(K @ v, 1e-300)) ** f1
        v = (b / np.maximum(K.T @ u, 1e-300)) ** f2
        if np.max(np.abs(u - u_prev) / np.maximum(u, 1e-300)) < tol:
            break
    return u[:, None] * K * v[None, :]


def sq_dist(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def weighted_stats(X, w):
    """Weighted mean and variance of every column of X (CSR or dense) with weights w >= 0, and n_eff."""
    w = np.asarray(w, dtype=float)
    s = w.sum()
    if s <= 0:
        return None
    wn = w / s
    mean = np.asarray(X.T @ wn).ravel()
    X2 = X.multiply(X) if scipy.sparse.issparse(X) else X ** 2
    var = np.maximum(np.asarray(X2.T @ wn).ravel() - mean ** 2, 0.0)
    return mean, var, s ** 2 / np.sum(w ** 2)


def welch(X, w1, w2):
    """(t, mean difference) of the columns of X between two weighted populations."""
    a, b = weighted_stats(X, w1), weighted_stats(X, w2)
    if a is None or b is None:
        return None, None
    (m1, v1, n1), (m2, v2, n2) = a, b
    return (m1 - m2) / np.sqrt(v1 / n1 + v2 / n2 + 1e-12), m1 - m2


def run_couplings(Z, times, samples, growth=None, max_cells=5000, eps=EPSILON, lam1=LAMBDA1, lam2=LAMBDA2,
                  seed=0, verb=True):
    """
    Couplings of every sample between its consecutive timepoints, on the rows of Z. Cells beyond max_cells
    per (sample, time) are left out (velocity by kernel regression). Returns a list of
    dict(sample, t_from, t_to, src, tgt, P) with P dense (len(src) x len(tgt)), total mass 1.
    """
    rng = np.random.default_rng(seed)
    out = []
    for s in np.unique(samples):
        ts = np.sort(np.unique(times[samples == s]))
        cells = {}
        for t in ts:
            idx = np.flatnonzero((samples == s) & (times == t))
            cells[t] = np.sort(rng.choice(idx, max_cells, replace=False)) if len(idx) > max_cells else idx
        for t0, t1 in zip(ts[:-1], ts[1:]):
            src, tgt = cells[t0], cells[t1]
            dt = t1 - t0
            C = sq_dist(Z[src], Z[tgt])
            C /= max(np.median(C), 1e-12)
            a = np.exp(growth[src] * dt) if growth is not None else np.ones(len(src))
            a /= a.sum()
            b = np.ones(len(tgt)) / len(tgt)
            P = unbalanced_sinkhorn(a, b, C, eps, lam1, lam2)
            P /= max(P.sum(), 1e-300)
            out.append(dict(sample=s, t_from=float(t0), t_to=float(t1), src=src, tgt=tgt, P=P))
            if verb:
                print(f"[classical_OT] sample {s}: {t0:g} -> {t1:g}, {len(src)} x {len(tgt)} cells")
    return out


def transitions(blocks, labels):
    """Cell-type flows per block: DataFrame (sample, t_from, t_to, from, to, mass, fate, ancestry)."""
    cats = sorted(set(labels))
    rows = []
    for blk in blocks:
        ls, lt = labels[blk['src']], labels[blk['tgt']]
        M = np.array([[blk['P'][np.ix_(ls == a, lt == b)].sum() for b in cats] for a in cats])
        fate = M / np.maximum(M.sum(axis=1, keepdims=True), 1e-300)
        anc = M / np.maximum(M.sum(axis=0, keepdims=True), 1e-300)
        for i, a in enumerate(cats):
            for j, b in enumerate(cats):
                rows.append(dict(sample=blk['sample'], t_from=blk['t_from'], t_to=blk['t_to'], cell_type_from=a,
                                 cell_type_to=b, mass=M[i, j], fate_probability=fate[i, j],
                                 ancestor_probability=anc[i, j]))
    return pd.DataFrame(rows)


def fate_genes(X, genes, blocks, labels, fit, n_top=100):
    """
    For each cell type C: genes up in the immediate ancestors of C against the ancestors of the other cells
    (weights of the couplings, pooled over intervals and samples, each block of mass 1), and genes up in C
    against the other cells (rows fit); weighted Welch t. Returns a DataFrame of the n_top genes of
    either list per cell type.
    """
    n = X.shape[0]
    out = []
    for c in sorted(set(labels[fit])):
        w_c, w_o = np.zeros(n), np.zeros(n)
        for blk in blocks:
            in_c = labels[blk['tgt']] == c
            w_c[blk['src']] += blk['P'][:, in_c].sum(axis=1)
            w_o[blk['src']] += blk['P'][:, ~in_c].sum(axis=1)
        t_anc, d_anc = welch(X, w_c, w_o) if w_c.sum() > 0 else (None, None)
        member = fit & (labels == c)
        t_ct, d_ct = welch(X, member.astype(float), (fit & ~member).astype(float))
        df = pd.DataFrame(dict(cell_type=c, gene=genes))
        for name, t, d in (('ancestors', t_anc, d_anc), ('cell_type', t_ct, d_ct)):
            df[f'{name}_t'] = t if t is not None else np.nan
            df[f'{name}_log_fc'] = d if d is not None else np.nan
            df[f'{name}_rank'] = df[f'{name}_t'].rank(ascending=False, method='first')
        keep = (df['ancestors_rank'] <= n_top) | (df['cell_type_rank'] <= n_top)
        df = df[keep].copy()
        df['top10_both'] = (df['ancestors_rank'] <= 10) & (df['cell_type_rank'] <= 10)
        df['n_ancestor_cells_eff'] = (w_c.sum() ** 2 / np.sum(w_c ** 2)) if w_c.sum() > 0 else 0.0
        out.append(df.sort_values('ancestors_rank'))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def pooled_transitions(tr):
    """Flows pooled over the samples for each interval (each sample has mass 1)."""
    if tr.empty:
        return tr
    g = tr.groupby(['t_from', 't_to', 'cell_type_from', 'cell_type_to'], as_index=False)['mass'].sum()
    n = tr.groupby(['t_from', 't_to'])['sample'].nunique().rename('n').reset_index()
    g = g.merge(n, on=['t_from', 't_to'])
    g['mass'] /= g['n']
    g = g.drop(columns='n')
    tot_from = g.groupby(['t_from', 't_to', 'cell_type_from'])['mass'].transform('sum')
    tot_to = g.groupby(['t_from', 't_to', 'cell_type_to'])['mass'].transform('sum')
    g['fate_probability'] = g['mass'] / tot_from.where(tot_from > 0)
    g['ancestor_probability'] = g['mass'] / tot_to.where(tot_to > 0)
    g.insert(0, 'sample', 'all')
    return g


def dense_blocks(blocks):
    """Sparse couplings {(sample, t_from, t_to): CSR} as blocks dict(sample, t_from, t_to, src, tgt, P) of mass 1."""
    out = []
    for (s, t0, t1), M in blocks.items():
        src = np.flatnonzero(np.asarray(M.sum(axis=1)).ravel() > 0)
        tgt = np.flatnonzero(np.asarray(M.sum(axis=0)).ravel() > 0)
        if not len(src) or not len(tgt):
            continue
        P = M[src][:, tgt].toarray()
        out.append(dict(sample=s, t_from=t0, t_to=t1, src=src, tgt=tgt, P=P / P.sum()))
    return out


def classical_couplings(A, fit, use_depth, max_cells=5000, seed=0, min_cells=10, verb=True):
    """
    Preprocessing and couplings of the classical OT on the cells fit of A (raw counts; obs time, dataset_id,
    proliferation_net_rate): log1p(counts / depth factor if use_depth, else normalize_total), genes expressed
    in >= min_cells fit cells, PCA (50) fitted on the fit cells. Returns (X log, genes, Z PCA of every cell,
    blocks with src / tgt = rows of A).
    """
    import anndata as ad
    import scanpy as sc
    from .embedding import log_normalised
    times = A.obs['time'].astype(float).values if 'time' in A.obs else np.zeros(A.n_obs)
    samples = A.obs['dataset_id'].astype(str).values if 'dataset_id' in A.obs else np.full(A.n_obs, '0')
    growth = A.obs['proliferation_net_rate'].astype(float).values if 'proliferation_net_rate' in A.obs else None
    X = log_normalised(A, use_depth)
    keep = np.asarray((X[fit] > 0).sum(axis=0)).ravel() >= min_cells
    X, genes = X[:, keep].tocsr(), A.var_names.values[keep].astype(str)
    B = ad.AnnData(X=X[fit])
    sc.pp.pca(B, n_comps=min(50, B.n_vars - 1, B.n_obs - 1), random_state=seed)
    PCs = B.varm['PCs']
    mean = np.asarray(X[fit].mean(axis=0)).ravel()
    Z = np.asarray(X @ PCs) - mean @ PCs
    idx = np.flatnonzero(fit)
    blocks = run_couplings(Z[idx], times[idx], samples[idx], None if growth is None else growth[idx],
                           max_cells=max_cells, seed=seed, verb=verb)
    for blk in blocks:
        blk['src'], blk['tgt'] = idx[blk['src']], idx[blk['tgt']]
    return X, genes, Z, blocks


def _weighted_cosine(A, B):
    """Cosine similarity per row, averaged with weights |a|·|b|."""
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    w = na * nb
    return float(np.sum((A * B).sum(axis=1)) / w.sum()) if w.sum() > 0 else np.nan


def _velocity(Y, blocks):
    """Barycentric velocities (descendants, else ancestors) in Y of the dense blocks; NaN elsewhere."""
    from .velocity import barycentric_velocity
    n = len(Y)
    sp = {(b['sample'], b['t_from'], b['t_to']): scipy.sparse.coo_matrix(
        (b['P'].ravel(), (np.repeat(b['src'], len(b['tgt'])), np.tile(b['tgt'], len(b['src'])))),
        shape=(n, n)).tocsr() for b in blocks}
    return barycentric_velocity(Y, sp)[0]


def _fate_distance(blocks_a, blocks_b, labels):
    """Mean Jensen-Shannon distance between the fate distributions (pooled over samples) of two couplings, per
    interval and source cell type, weighted by the source mass."""
    from scipy.spatial.distance import jensenshannon
    fa, fb = (pooled_transitions(transitions(b, labels)) for b in (blocks_a, blocks_b))
    key = ['t_from', 't_to', 'cell_type_from']
    d, w = [], []
    for k, ga in fa.groupby(key):
        gb = fb.set_index(key).loc[[k]] if k in fb.set_index(key).index else None
        if gb is None or ga['mass'].sum() <= 0:
            continue
        pa = ga.set_index('cell_type_to')['mass']
        pb = gb.set_index('cell_type_to')['mass'].reindex(pa.index).fillna(0)
        if pb.sum() <= 0:
            continue
        d.append(jensenshannon(pa / pa.sum(), pb / pb.sum(), base=2))
        w.append(pa.sum())
    return float(np.average(d, weights=w)) if w else np.nan


def trajectory_preservation(A, genes, Z_full, blocks_full, fit, use_depth, max_cells=5000, seed=0, n_random=3):
    """
    How much a gene set preserves the trajectories of the classical OT on every gene: the same OT (same cells,
    preprocessing and parameters) on the gene set only, compared with the full one by (i) the weighted cosine
    of the velocities of the transported cells in the full PCA space (1 = same directions) and (ii) the mean
    Jensen-Shannon distance of the cell-type fate distributions (0 = same transitions; needs obs['cell_type']).
    Baseline: n_random random sets of as many genes. Returns a dict.
    """
    labels = A.obs['cell_type'].astype(str).values if 'cell_type' in A.obs else None
    V_full = _velocity(Z_full, blocks_full)

    def score(gs):
        sub = classical_couplings(A[:, list(gs)], fit, use_depth, max_cells, seed, min_cells=1, verb=False)[3]
        V = _velocity(Z_full, sub)
        ok = np.isfinite(V).all(axis=1) & np.isfinite(V_full).all(axis=1)
        return (_weighted_cosine(V[ok], V_full[ok]),
                _fate_distance(blocks_full, sub, labels) if labels is not None else np.nan)

    cos, js = score(genes)
    rng = np.random.default_rng(seed)
    pool = A.var_names[np.asarray((A.X[fit] > 0).sum(axis=0)).ravel() >= 10]
    rand = [score(rng.choice(pool, len(genes), replace=False)) for _ in range(n_random)]
    return dict(n_genes=len(genes), velocity_cosine=cos, fate_js=js,
                random_velocity_cosine=float(np.mean([r[0] for r in rand])),
                random_fate_js=float(np.nanmean([r[1] for r in rand])) if labels is not None else np.nan)


def fate_probabilities(blocks, labels, n_cells):
    """
    Fate of every transported cell: probability that its descendants at the last time of its sample belong to
    each cell type, by chaining the row-normalised couplings backwards (CellRank-style absorption, along the real
    times); the cells of the last time have their own type. Returns (F (n_cells x K), cell types); NaN rows
    for the cells outside the couplings.
    """
    cats = sorted(set(labels))
    F = np.full((n_cells, len(cats)), np.nan)
    for s in sorted({b['sample'] for b in blocks}, key=str):
        bs = sorted([b for b in blocks if b['sample'] == s], key=lambda b: b['t_from'])
        last = bs[-1]['tgt']
        F[last] = (labels[last][:, None] == np.array(cats)[None, :]).astype(float)
        for b in bs[::-1]:
            Ft = np.nan_to_num(F[b['tgt']])
            P = b['P'] / np.maximum(b['P'].sum(axis=1, keepdims=True), 1e-300)
            G = P @ Ft
            tot = G.sum(axis=1, keepdims=True)
            F[b['src']] = np.where(tot > 0, G / np.maximum(tot, 1e-300), np.nan)
    return F, cats


def fate_drivers(X, genes, F, cats, labels, times, last_times, min_cells=30):
    """
    Driver genes of each fate: Pearson correlation, within each timepoint before the last one, between the
    expression of a gene (X log, cells x genes) and the probability of the fate, over the cells not yet of that
    type (priming rather than markers); combined over the timepoints by Fisher's z (weights n - 3), BH q-values
    per fate. Returns a DataFrame (cell_type, gene, corr, z, qval, rank) of the positive drivers.
    """
    from scipy.stats import norm
    out = []
    for k, c in enumerate(cats):
        zs, ws = 0.0, 0.0
        for t in np.unique(times):
            m = (times == t) & ~last_times & np.isfinite(F[:, k]) & (labels != c)
            n = int(m.sum())
            f = F[m, k]
            if n < min_cells or f.std() == 0:
                continue
            Xm = X[m]
            mean = np.asarray(Xm.mean(axis=0)).ravel()
            X2 = Xm.multiply(Xm) if scipy.sparse.issparse(Xm) else Xm ** 2
            sd = np.sqrt(np.maximum(np.asarray(X2.mean(axis=0)).ravel() - mean ** 2, 0.0))
            fz = (f - f.mean()) / f.std()
            r = np.asarray(Xm.T @ fz).ravel() / n / np.where(sd > 0, sd, np.inf)
            zs = zs + (n - 3) * np.arctanh(np.clip(r, -0.999999, 0.999999))
            ws += n - 3
        if ws == 0:
            continue
        zbar = zs / ws
        z = zbar * np.sqrt(ws)
        p = norm.sf(z)  # one-sided: expression up with the fate
        order = np.argsort(p)
        q = np.empty_like(p)
        q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        df = pd.DataFrame(dict(cell_type=c, gene=genes, corr=np.tanh(zbar), z=z, qval=np.minimum(q, 1.0)))
        df = df[df['corr'] > 0].sort_values('z', ascending=False)
        df['rank'] = np.arange(1, len(df) + 1)
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def driver_order(drivers, q_max=0.05, representatives=True):
    """
    Driver genes in round robin over the fates (best of each fate in turn), q <= q_max, module representatives
    only (column 'representative' of mark_redundant) by default: [(gene, 'fate=<type>')].
    """
    d = drivers[drivers['qval'] <= q_max]
    if representatives and 'representative' in d:
        d = d[d['representative'].astype(bool)]
    groups = list(d.groupby('cell_type'))
    lists = [list(g.sort_values('rank')['gene']) for _, g in groups]
    out, seen = [], set()
    for i in range(max((len(l) for l in lists), default=0)):
        for (c, _), l in zip(groups, lists):
            if i < len(l) and l[i] not in seen:
                seen.add(l[i])
                out.append((l[i], f'fate={c}'))
    return out


def mark_redundant(drivers, X, genes, rows, n_top=200, max_corr=0.3, q_max=0.05, max_linked=2):
    """
    Redundancy of the drivers: in the round-robin order over the fates (top n_top per fate), a gene whose expression
    correlates (|r| >= max_corr, over the cells rows of X) with at least max_linked earlier accepted ones at once is
    redundant: each co-expressed module (e.g. the cell cycle) keeps max_linked genes, and a gene linked to a single
    accepted one stays (finer signals). Adds the columns representative and redundant_with.
    """
    d = drivers.copy()
    d['representative'], d['redundant_with'] = False, ''
    top = d[(d['qval'] <= q_max) & (d['rank'] <= n_top)]
    order = [g for g, _ in driver_order(top, q_max, representatives=False)]
    if not order:
        return d
    pos = {g: i for i, g in enumerate(genes)}
    M = X[rows][:, [pos[g] for g in order]]
    M = M.toarray() if scipy.sparse.issparse(M) else np.asarray(M)
    sd = M.std(axis=0)
    R = np.corrcoef(M[:, sd > 0].T)
    ok_idx = np.flatnonzero(sd > 0)
    col = {g: k for k, g in zip(range(len(ok_idx)), [order[i] for i in ok_idx])}
    accepted, status = [], {}
    for g in order:
        if g not in col:
            status[g] = ''
            continue
        hit = [a for a in accepted if abs(R[col[g], col[a]]) >= max_corr]
        status[g] = ', '.join(hit) if len(hit) >= max_linked else None
        if len(hit) < max_linked:
            accepted.append(g)
    d['representative'] = d['gene'].map(lambda g: status.get(g, 0) is None)
    d['redundant_with'] = d['gene'].map(lambda g: status.get(g) or '')
    return d


def fate_prediction(X, var_names, genes, F, rows, times, Z_full=None, pool=None, n_random=3, n_folds=5, seed=0):
    """
    Capacity of a gene set to predict the fates: cross-validated R² (KFold, ridge with internal alpha choice) of the
    fate probabilities F of the cells rows (classical OT on every gene, before the last time) from the expression
    at t of the genes (X log, cells x var_names) with the timepoint as covariate. References: time only, random
    gene sets of the same size, every gene (PCA Z_full). Returns a dict.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    Y = F[rows]
    tu = np.unique(times[rows])
    T = (times[rows][:, None] == tu[None, :]).astype(float)
    cv = KFold(n_folds, shuffle=True, random_state=seed)
    pos = {g: i for i, g in enumerate(var_names)}

    def r2(feat):
        M = np.hstack([feat, T]) if feat is not None else T
        sd = M.std(axis=0)
        M = (M - M.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
        pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), M, Y, cv=cv)
        return float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(axis=0)) ** 2).sum())

    def expr(gs):
        M = X[rows][:, [pos[g] for g in gs if g in pos]]
        return M.toarray() if scipy.sparse.issparse(M) else np.asarray(M)

    rng = np.random.default_rng(seed)
    pool = list(var_names) if pool is None else list(pool)  # genes the random sets are drawn from
    out = dict(fate_r2=r2(expr(genes)), fate_r2_time=r2(None),
               fate_r2_random=float(np.mean([r2(expr(rng.choice(pool, len(genes), replace=False)))
                                             for _ in range(n_random)])))
    if Z_full is not None:
        out['fate_r2_all_genes'] = r2(Z_full[rows])
    return out
