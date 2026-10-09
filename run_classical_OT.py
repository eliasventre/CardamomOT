"""
run_classical_OT.py
-------------------
Classical optimal-transport analysis of the data before CardamomOT (Waddington-OT / moscot style), on every
gene and the cells of CardamomOT's inference (data_<split>.h5ad: train cells, or every inference cell with
split = 'full'; obs['split'] of split_dataset.py), per sample (CardamomOT.tools.classical_ot).

Usage:
    python run_classical_OT.py -i <project_path>
(run_classical_OT, classical_ot_max_cells, seed: model_parameters sheet)

Cells: log1p(counts / obs['depth_factor'] with use_depth_factor, else normalize_total), genes expressed in >= 10
train cells, PCA (50 components) fitted on the train cells. Couplings between consecutive timepoints of each
sample (growth: obs['proliferation_net_rate'] if present). The report draws the velocities on the embedding
of Data/data.h5ad: barycentric (descendants, else ancestors) in its PCA space, extended to the other cells
(test, unused, beyond classical_ot_max_cells) by Gaussian-kernel regression in the PCA space of the transport
(velocity.npz), within their sample and timepoint.

Required input files:
    - Data/data.h5ad (obs['time'], obs['split'] from split_dataset.py; optional obs['cell_type'],
      obs['dataset_id'], obs['proliferation_net_rate'], obs['depth_factor'])

Output (cardamomOT/classical_OT/):
    - velocity.npz: obs_names (every cell of data.h5ad), Z (PCA space of the transport), groups
      (sample|time), transported (cells of the couplings)
    - couplings.npz: couplings (sparse, top 50 targets per cell; rows of data.h5ad), see tools/velocity.py
    - transitions.csv: cell-type flows per sample and interval (sample 'all': pooled over the samples)
    - fate_genes.csv: per cell type, genes up in its ancestors vs the ancestors of the other cells, and
      genes up in the cell type vs the other cells (weighted Welch t, log fold change, ranks)
    - fate.npz: fate probabilities of the transported cells (cell types of the last time, chained couplings)
    - fate_drivers.csv: per fate, genes whose expression predicts it (correlation with the fate probability
      within each earlier timepoint, cells not yet of that type; Fisher-combined z, BH q-value, rank;
      representative: not correlated, |r| >= 0.3, with two earlier drivers at once: two genes per co-expression
      module), used as
      terminals of the gene selection (n_driver_genes)
"""
import sys; sys.path += ['../']
import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from CardamomOT import NetworkModel, ensure_raw_counts, harmonize_obs
from CardamomOT.run_options import parse_step_options, configure
from CardamomOT.tools.embedding import log_normalised
from CardamomOT.tools.classical_ot import (classical_couplings, transitions, pooled_transitions, fate_genes,
                                          fate_probabilities, fate_drivers, mark_redundant)
from CardamomOT.tools.velocity import save_couplings


def main(argv):
    opts = parse_step_options(argv, 'run_classical_OT', __doc__)
    p = opts.p
    model = configure(NetworkModel(1), opts)
    if not model.run_classical_OT:
        print("[run_classical_OT] run_classical_OT = False: skipped")
        return
    seed = 0 if model.seed is None else int(model.seed)
    out_dir = os.path.join(p, 'cardamomOT', 'classical_OT')
    os.makedirs(out_dir, exist_ok=True)

    data_path = os.path.join(p, 'Data', 'data.h5ad')
    A = ensure_raw_counts(ad.read_h5ad(data_path), data_path)
    harmonize_obs(A)
    obs_names = A.obs_names.values.astype(str)
    times = A.obs['time'].astype(float).values if 'time' in A.obs else np.zeros(A.n_obs)
    if len(np.unique(times)) < 2:
        print("[run_classical_OT] Single timepoint: no transport, skipped")
        return
    samples = A.obs['dataset_id'].astype(str).values if 'dataset_id' in A.obs else np.full(A.n_obs, '0')
    # Transported cells: those of CardamomOT's inference, i.e. obs['split'] of data.h5ad (split_dataset.py, which
    # runs just before; data_<split>.h5ad is written later by select_genes and may be from an earlier split)
    ref_path = os.path.join(p, 'Data', f'data_{model.split}.h5ad')
    if 'split' in A.obs:
        fit = A.obs['split'].astype(str).isin(['train', 'full']).to_numpy()
        print("[run_classical_OT] Transported cells: obs['split'] of Data/data.h5ad (train / full)")
    elif os.path.exists(ref_path):
        fit = A.obs_names.isin(ad.read_h5ad(ref_path, backed='r').obs_names)
        print(f"[run_classical_OT] Transported cells: those of data_{model.split}.h5ad (no obs['split'])")
    else:
        print(f"[run_classical_OT] Error: no data_{model.split}.h5ad nor obs['split']: run split_dataset.py and "
              "select_genes.py first")
        sys.exit(1)
    labels = A.obs['cell_type'].astype(str).values if 'cell_type' in A.obs else None
    growth = (A.obs['proliferation_net_rate'].astype(float).values
              if 'proliferation_net_rate' in A.obs else None)
    print(f"[run_classical_OT] {int(fit.sum())} transported cells (of {A.n_obs}), {len(np.unique(samples[fit]))} "
          f"sample(s); growth from obs['proliferation_net_rate']: {growth is not None}; depth factor: "
          f"{bool(model.use_depth_factor and 'depth_factor' in A.obs)} (use_depth_factor)")

    # log1p(normalised counts) of the genes expressed in >= 10 transported cells, PCA fitted on those cells,
    # couplings (indices: rows of data.h5ad)
    X, genes, Z, blocks = classical_couplings(A, fit, bool(model.use_depth_factor),
                                              int(model.classical_ot_max_cells), seed)
    del A
    print(f"[run_classical_OT] {len(genes)} genes, PCA {Z.shape[1]} components")

    # Space of the kernel regression of the velocities (cells outside the couplings), drawn by the report
    np.savez_compressed(os.path.join(out_dir, 'velocity.npz'), obs_names=obs_names, Z=Z.astype(np.float32),
                        groups=np.array([f'{s}|{t:g}' for s, t in zip(samples, times)]), transported=fit)

    # Sparse couplings (top 50 targets per source cell)
    sparse = []
    for blk in blocks:
        P, k = blk['P'], min(50, blk['P'].shape[1])
        cols = np.argpartition(-P, k - 1, axis=1)[:, :k]
        w = np.take_along_axis(P, cols, axis=1)
        rows = np.broadcast_to(np.arange(P.shape[0])[:, None], cols.shape)
        sparse.append(dict(src=blk['src'][rows.ravel()], tgt=blk['tgt'][cols.ravel()], w=w.ravel(),
                           sample=int(np.flatnonzero(np.unique(samples) == blk['sample'])[0]),
                           t_from=blk['t_from'], t_to=blk['t_to']))
    save_couplings(os.path.join(out_dir, 'couplings.npz'), sparse)

    if labels is None:
        print("[run_classical_OT] No obs['cell_type']: no transitions nor fate genes")
    else:
        tr = transitions(blocks, labels)
        tr = pd.concat([tr, pooled_transitions(tr)], ignore_index=True)
        tr.to_csv(os.path.join(out_dir, 'transitions.csv'), index=False)
        fg = fate_genes(X, genes, blocks, labels, fit)
        fg.to_csv(os.path.join(out_dir, 'fate_genes.csv'), index=False)
        # Fate drivers: genes whose expression predicts the fate (last time) within each earlier timepoint
        F, cats = fate_probabilities(blocks, labels, len(times))
        last = np.zeros(len(times), bool)
        for s in np.unique(samples[fit]):
            m = fit & (samples == s)
            last |= m & (times == times[m].max())
        dr = mark_redundant(fate_drivers(X, genes, F, cats, labels, times, last), X, genes, np.flatnonzero(fit))
        dr.to_csv(os.path.join(out_dir, 'fate_drivers.csv'), index=False)
        np.savez_compressed(os.path.join(out_dir, 'fate.npz'), obs_names=obs_names, F=F, cell_types=np.array(cats))
        for c, df in dr.groupby('cell_type'):
            rep = df[df['representative'].astype(bool)]
            print(f"[run_classical_OT] Drivers of fate {c}: {int((df['qval'] <= 0.05).sum())} (q <= 0.05), top "
                  f"{list(df['gene'][:4])}; module representatives {list(rep['gene'][:6])}")
        for c, df in fg.groupby('cell_type'):
            top_a = list(df.nsmallest(10, 'ancestors_rank')['gene'])
            top_c = list(df.nsmallest(10, 'cell_type_rank')['gene'])
            print(f"[run_classical_OT] {c}: ancestors {top_a[:5]}..., cell type {top_c[:5]}..., "
                  f"{len(set(top_a) & set(top_c))} common in the top 10")
    print(f"[run_classical_OT] Saved velocity.npz, couplings.npz, transitions.csv, fate_genes.csv to {out_dir}")
    print(f"[run_classical_OT] Cells in the couplings: {int(sum(len(np.unique(b['src'])) for b in blocks))} sources, "
          f"{int(sum(len(np.unique(b['tgt'])) for b in blocks))} targets; the others by kernel regression in the report")


if __name__ == "__main__":
    main(sys.argv[1:])
