"""
Genes selected per network condition (select_genes_per_condition).

The data files hold the union of the per-condition selections, and uns['condition_genes'] = {condition: [genes]};
a gene outside the set of a condition is inactive in its samples: not fitted (mixture, OT, network, kinetics),
frozen in the simulations and left out of the comparisons. Without uns['condition_genes'] every gene is active.
Masks are (S, ns + G) bool over the sorted dataset_id (sample indices of the run), stimuli always active.
"""
import json
import os

import numpy as np

KEY = 'condition_genes'


def condition_gene_sets(adata):
    """{condition: [genes]} of an AnnData (uns['condition_genes']), None if every gene is shared."""
    sets = getattr(adata, 'uns', {}).get(KEY)
    if sets is None or len(sets) == 0:
        return None
    return {str(c): [str(g) for g in np.atleast_1d(v)] for c, v in dict(sets).items()}


def sample_conditions(obs):
    """{dataset_id: network_condition} of the cells of obs (one condition per sample; '0' without columns)."""
    if 'dataset_id' not in obs:
        return {'0': str(obs['network_condition'].iloc[0]) if 'network_condition' in obs and len(obs) else '0'}
    sid = obs['dataset_id'].astype(str).values
    cond = obs['network_condition'].astype(str).values if 'network_condition' in obs else np.full(len(sid), '0')
    return {s: str(cond[np.flatnonzero(sid == s)[0]]) for s in np.unique(sid)}


def gene_mask(adata, sample_ids=None, ns=0, genes=None):
    """
    (S, ns + G) bool: gene active in each sample (sorted dataset_id, or sample_ids), stimuli always active; None when
    every gene is shared (no uns['condition_genes']). genes: column names (default adata.var_names).
    """
    sets = condition_gene_sets(adata)
    if sets is None:
        return None
    genes = [str(g) for g in (adata.var_names if genes is None else genes)]
    s2c = sample_conditions(adata.obs)
    ids = sorted(s2c) if sample_ids is None else [str(s) for s in sample_ids]
    mask = np.ones((len(ids), ns + len(genes)), dtype=bool)
    for k, s in enumerate(ids):
        c = s2c.get(s)
        if c is not None and c in sets:
            mask[k, ns:] = np.isin(genes, sets[c])
    return mask


def condition_genes_of(adata, condition):
    """Genes of a condition (every gene without per-condition selection)."""
    sets = condition_gene_sets(adata)
    return list(adata.var_names) if sets is None or str(condition) not in sets else sets[str(condition)]


def save_sets(path, sets):
    with open(path, 'w') as fh:
        json.dump(sets, fh, indent=1)


def load_sets(cardamom_dir):
    path = os.path.join(cardamom_dir, 'condition_genes.json')
    return json.load(open(path)) if os.path.exists(path) else None


def row_mask(adata, row_samples, genes=None):
    """(N, G) bool: gene active for each row of dataset_id row_samples; None when every gene is shared."""
    sets = condition_gene_sets(adata)
    if sets is None:
        return None
    genes = [str(g) for g in (adata.var_names if genes is None else genes)]
    s2c = sample_conditions(adata.obs)
    conds = sorted(sets)
    by_cond = {c: np.isin(genes, sets[c]) for c in conds}
    rows = np.asarray(row_samples).astype(str)
    out = np.ones((len(rows), len(genes)), dtype=bool)
    for s in np.unique(rows):
        c = s2c.get(s)
        if c in by_cond:
            out[rows == s] = by_cond[c]
    return out


def mask_stage(A, data, model_draws=True):
    """Stage AnnData A (obs['dataset_id']) of the data `data`: uns['condition_genes'] copied and, for model draws,
    counts of the genes outside each row's condition set to 0 (they are not modelled there). In place; returns A."""
    sets = condition_gene_sets(data)
    if A is None or sets is None:
        return A
    A.uns[KEY] = sets
    if model_draws and 'dataset_id' in A.obs:
        m = row_mask(data, A.obs['dataset_id'].astype(str).values, genes=A.var_names)
        X = A.X.toarray() if hasattr(A.X, 'toarray') else np.array(A.X, dtype=float)
        X[~m] = 0
        A.X = X
    return A


def project_data(p, split=None):
    """Data AnnData of a project (data_<split>, else data_full / data_train), backed: obs, var and uns only."""
    import anndata as ad
    for name in ([split] if split else []) + ['full', 'train']:
        path = os.path.join(p, 'Data', f'data_{name}.h5ad')
        if os.path.exists(path):
            return ad.read_h5ad(path, backed='r')
    raise FileNotFoundError(f"no Data/data_full.h5ad nor data_train.h5ad in {p}")


def by_condition(data, adatas, samples_traj=None):
    """
    [(condition label, AnnData list)]: with >= 2 network conditions in data.obs, one entry per condition with the
    cells of its samples (obs['dataset_id'], attached from samples_traj, the data_samples.npy of the run, when absent)
    and its genes (all genes without select_genes_per_condition); otherwise [('', adatas)].
    """
    s2c = sample_conditions(data.obs)
    conds = sorted(set(s2c.values()))
    if len(conds) < 2:
        return [('', adatas)]
    names = sorted(s2c)
    out_adatas = []
    for A in adatas:
        if 'dataset_id' not in A.obs and samples_traj is not None and 'time' in A.obs:
            t = np.asarray(A.obs['time'], dtype=float)
            N = int(np.sum(t == t.min()))
            idx = (np.asarray(samples_traj) if A.n_obs == len(samples_traj)
                   else np.tile(np.asarray(samples_traj)[:N], A.n_obs // max(N, 1)))
            if len(idx) == A.n_obs:
                A.obs['dataset_id'] = np.asarray(names)[np.minimum(idx.astype(int), len(names) - 1)]
        out_adatas.append(A)
    out = []
    for c in conds:
        samples = [s for s in names if s2c[s] == c]
        genes = condition_genes_of(data, c)
        sub = []
        for A in out_adatas:
            if 'dataset_id' not in A.obs:
                raise ValueError("by_condition: AnnData without obs['dataset_id'] (pass samples_traj)")
            B = A[A.obs['dataset_id'].astype(str).isin(samples).to_numpy()]
            sub.append(B[:, B.var_names.isin(genes)].copy())
        out.append((str(c), sub))
    return out
