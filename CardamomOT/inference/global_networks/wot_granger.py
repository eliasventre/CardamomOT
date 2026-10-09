"""
WOT-Granger global network: the Granger regression of OTVelo-Granger (otvelo_granger.granger_network) through
the couplings of the classical OT analysis (run_classical_OT.py: Waddington-OT style, every gene, train cells,
growth from the net proliferation rate) instead of the OTVelo couplings. A gene regulates another when its
velocity at t predicts the velocity of the other at t + 1 along the inferred trajectories (both directions are
read from the network: regulators and targets of the genes of interest).

The couplings are fixed: the null networks (cells permuted gene by gene, gene_selection.permuted_counts) are
regressions through the same couplings, which keeps the trajectories and breaks the gene-gene dependences.
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse

from .otvelo_granger import granger_network

DEFAULTS = dict(n_cells=2000, k_candidates=50, en_alpha='auto', l1_ratio=0.5, scale=True, stim_weight=10.0)


def load_classical_couplings(project_path):
    """(cell names of the rows, {(t_from, t_to): CSR coupling summed over the samples}) of run_classical_OT."""
    from ...tools.velocity import coupling_blocks
    cdir = os.path.join(project_path or '', 'cardamomOT', 'classical_OT')
    vel, cp = os.path.join(cdir, 'velocity.npz'), os.path.join(cdir, 'couplings.npz')
    if not (os.path.exists(vel) and os.path.exists(cp)):
        raise FileNotFoundError("network_method = 'wot_granger' needs the classical OT couplings "
                                "(cardamomOT/classical_OT/): run run_classical_OT.py before select_genes.py")
    names = np.load(vel, allow_pickle=True)['obs_names'].astype(str)
    out = {}
    for (_, t0, t1), M in coupling_blocks(cp, len(names)).items():
        out[(t0, t1)] = out[(t0, t1)] + M if (t0, t1) in out else M
    return names, out


def build_network(adata, stim, seed=0, sample_key='dataset_id', project_path=None, n_cells=2000, k_candidates=50,
                  en_alpha='auto', l1_ratio=0.5, scale=True, stim_weight=10.0):
    """
    Global network interface (see global_networks): Granger regression through the classical OT couplings on
    log(x+1) counts of adata, one network per sample (obs[sample_key]) averaged with cell weights; the cells
    outside the couplings are left out. Returns C, stimuli first.
    """
    names, blocks = load_classical_couplings(project_path)
    pos = pd.Index(names).get_indexer(adata.obs_names.astype(str))
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.asarray(adata.X)
    X = np.log1p(X.astype(np.float32))
    times = adata.obs['time'].values.astype(float)
    tu = np.sort(np.unique(times))
    samples = adata.obs[sample_key].values if sample_key in adata.obs else np.zeros(adata.n_obs)
    rng = np.random.default_rng(seed)
    C, w_tot = 0, 0
    for s in np.unique(samples):
        ts = np.sort(np.unique(times[samples == s]))
        if len(ts) < 2 or any((a, b) not in blocks for a, b in zip(ts[:-1], ts[1:])):
            continue
        # Cells of the sample in the couplings, at most n_cells per time (those carrying most mass first)
        idx = []
        for k, t in enumerate(ts):
            cand = np.flatnonzero((samples == s) & (times == t) & (pos >= 0))
            mass = np.zeros(len(cand))
            if k < len(ts) - 1:
                mass += np.asarray(blocks[(t, ts[k + 1])][pos[cand]].sum(axis=1)).ravel()
            if k > 0:
                mass += np.asarray(blocks[(ts[k - 1], t)][:, pos[cand]].sum(axis=0)).ravel()
            cand = cand[mass > 0]
            if len(cand) > n_cells:
                cand = rng.choice(cand, n_cells, replace=False, p=mass[mass > 0] / mass[mass > 0].sum())
            idx.append(np.sort(cand))
        if any(len(i) == 0 for i in idx):
            continue
        Ts = []
        for k in range(len(ts) - 1):
            T = blocks[(ts[k], ts[k + 1])][pos[idx[k]]][:, pos[idx[k + 1]]].toarray()
            Ts.append(T + 1e-9 * max(T.mean(), 1e-300))  # every cell keeps a (tiny) row and column mass
        stim_s = None if stim is None else np.asarray(stim, dtype=float).reshape(len(tu), -1)[np.searchsorted(tu, ts)]
        n = sum(len(i) for i in idx)
        C = C + n * granger_network([X[i] for i in idx], Ts, ts, stim_s, k_candidates, en_alpha, l1_ratio, scale,
                                    stim_weight)
        w_tot += n
    if w_tot == 0:
        raise ValueError("WOT-Granger: no sample with classical OT couplings between consecutive timepoints")
    return C / w_tot
