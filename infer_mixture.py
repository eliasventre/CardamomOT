"""
infer_mixture.py
----------------
Main CARDAMOM mixture model inference pipeline.

This script infers the burst kinetics parameters (mixture model) from temporal
scRNA-seq data.

Usage:
    python infer_mixture.py -i <project_path> [--mean-forcing <float>]
(split, max_iter_softEM, integrate_samples, ref_sample_integration: model_parameters sheet)

Several samples (obs['dataset_id'] with >= 2 samples): the mixture is fitted per sample and kept per
sample (mixture_parameters.npy (S, M+1, G)); with integrate_samples, the counts of each sample are
projected, mode by mode (level by level for more than two levels), on the mixture of the reference sample
(ref_sample_integration, default the first fitted sample) by quantile matching, and every sample takes the
reference parameters. Data/data_{full,train,test}.h5ad are then rewritten with projected counts in X and the raw
counts in layers['counts_raw'] (a rerun always restarts from the raw counts); without projection they are restored.
Genes listed for a sample in Data/KO_OV_inference.txt are never projected for that sample.
"""

import sys
sys.path += ['../']
import numpy as np
import pandas as pd
import scipy.sparse
from CardamomOT import NetworkModel as NetworkModel_beta, check_stationary
from CardamomOT.inputs import input_dir
import anndata as ad
from CardamomOT.run_options import parse_step_options, settings, configure
from CardamomOT.config import find_stimulus_schedule
import os
import pickle

verb = 1
RAW_LAYER = 'counts_raw'


def _restore_raw(adata):
    """X <- raw counts if a previous integration stored them."""
    if RAW_LAYER in adata.layers:
        adata.X = adata.layers[RAW_LAYER].copy()
    return adata


def _write_counts(adata, X, path):
    """Write integrated counts in X (same sparsity as the raw ones) and keep the raw counts in a layer."""
    if RAW_LAYER not in adata.layers:
        adata.layers[RAW_LAYER] = adata.X.copy()
    X = X.astype(np.float32)
    adata.X = scipy.sparse.csr_matrix(X) if scipy.sparse.issparse(adata.layers[RAW_LAYER]) else X
    adata.write(path)


def _load_kov_genes(p, gene_names):
    """{dataset_id: set of genes} perturbed per sample in Data/KO_OV_inference.txt (sample_id | KO | OV)."""
    path = os.path.join(input_dir(p), 'KO_OV_inference.txt')
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep='\t', dtype=str).fillna('')
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.rename(columns={'DATASET_ID': 'SAMPLE_ID'})
    upper = {g.upper(): g for g in gene_names}
    out = {}
    for _, row in df.iterrows():
        genes = [upper[g.strip().upper()] for col in ('KO', 'OV') for g in str(row.get(col, '')).split(',')
                 if g.strip().upper() in upper]
        out.setdefault(str(row.get('SAMPLE_ID', '')).strip(), set()).update(genes)
    return out


def main(argv):
    """
    Main function to run the mixture model inference pipeline.

    Args:
        argv: Command-line arguments (-i <project>, --mean-forcing).
    """
    opts = parse_step_options(argv, 'infer_mixture', __doc__)
    p = opts.p
    split = settings(opts).split

    data_path = os.path.join(p, 'Data', 'data_{}.h5ad'.format(split))
    if os.path.exists(data_path):
        adata = _restore_raw(ad.read_h5ad(data_path))
        if verb:
            print(f"[infer_mixture] Loaded data from {data_path}")
        # Depth factors of data_<split> must be those of Data/data.h5ad (estimate_cell_depth rerun after select_genes?)
        full_path = os.path.join(p, 'Data', 'data.h5ad')
        if 'depth_factor' in adata.obs and os.path.exists(full_path):
            obs_full = ad.read_h5ad(full_path, backed='r').obs
            if 'depth_factor' not in obs_full:
                print("[infer_mixture] Warning: obs['depth_factor'] in data_{split} but no longer in Data/data.h5ad: "
                      "rerun select_genes (data files written before estimate_cell_depth)".format(split=split))
            else:
                common = adata.obs_names.intersection(obs_full.index[~obs_full.index.duplicated()])
                diff = np.abs(np.log(np.maximum(adata.obs.loc[common, 'depth_factor'].to_numpy(float), 1e-12)
                                     / np.maximum(obs_full.loc[common, 'depth_factor'].to_numpy(float), 1e-12)))
                if len(common) and diff.max() > 1e-6:
                    print(f"[infer_mixture] Warning: the depth factors of data_{split} differ from those of "
                          f"Data/data.h5ad for {int((diff > 1e-6).sum())} cells: rerun select_genes (or split_dataset) "
                          "after estimate_cell_depth")
    else:
        error_msg = (
            f"Error: Data file not found at {data_path}.\n"
            "Create a 'Data' folder in your project directory and place "
            f"a count table named 'data_{split}.h5ad'."
        )
        print(error_msg)
        raise FileNotFoundError(error_msg)

    # ─── CHECK TEMPORAL INFORMATION ──────────────────────────────────────
    # Absent or single timepoint: stationary setting, mixture fitted without temporal constraints
    if check_stationary(adata) and verb:
        print("[infer_mixture] Stationary data (no or single timepoint): "
              "fitting mixture without temporal constraints")

    # ─── LOAD STIMULUS SCHEDULE (optional) ──────────────────────────────
    stim_sched = None
    sched_path = (find_stimulus_schedule(input_dir(p))
                  or os.path.join(input_dir(p), 'stimulus_schedule_inference.txt'))
    if os.path.exists(sched_path):
        stim_sched = np.loadtxt(sched_path)
        if verb:
            print(f"[infer_mixture] Loaded stimulus schedule from {sched_path}")

    # ─── DETECT n_stimuli FROM SCHEDULE ─────────────────────────────────
    _stim_arr = np.asarray(stim_sched) if stim_sched is not None else None
    n_stimuli = int(_stim_arr.shape[1]) if (_stim_arr is not None and _stim_arr.ndim == 2) else 1
    if verb:
        print(f"[infer_mixture] n_stimuli detected: {n_stimuli}")

    # ─── INFER MIXTURE MODEL ────────────────────────────────────────────
    model = NetworkModel_beta(adata.shape[1], n_stimuli=n_stimuli)
    configure(model, opts)
    if verb:
        print(f"[infer_mixture] mean_forcing_em={model.mean_forcing_em}, "
              f"max_iter_softEM={model.max_iter_softEM}, integrate_samples={model.integrate_samples}")

    if verb:
        print(f"[infer_mixture] Starting mixture model inference ({adata.shape[1]} genes)...")

    X_int = model.fit_mixture_samples(
        adata,
        kov_genes=_load_kov_genes(p, list(adata.var_names)),
        gene_names=list(adata.var_names),
        verb=verb,
        stimulus_schedule=stim_sched,
    )

    # ─── WRITE INTEGRATED DATA (or restore raw counts) ──────────────────
    other_files = [os.path.join(p, 'Data', f'data_{s}.h5ad') for s in ('full', 'train', 'test') if s != split]
    if X_int is not None:
        _write_counts(adata, X_int, data_path)
        if verb:
            print(f"[infer_mixture] Integrated counts written to {data_path} (raw counts in layers['{RAW_LAYER}'])")
        for path in other_files:
            if not os.path.exists(path):
                continue
            other = _restore_raw(ad.read_h5ad(path))
            if list(other.var_names) != list(adata.var_names):
                print(f"[infer_mixture] Warning: {path} has other genes than data_{split}; not integrated")
                continue
            _write_counts(other, model.integrate_data(other), path)
            if verb:
                print(f"[infer_mixture] Integrated counts written to {path}")
    else:
        # No integration (single sample or disabled): undo a previous one
        for path in [data_path] + other_files:
            if os.path.exists(path):
                d = ad.read_h5ad(path)
                if RAW_LAYER in d.layers:
                    _restore_raw(d)
                    del d.layers[RAW_LAYER]
                    d.write(path)
                    print(f"[infer_mixture] Raw counts restored in {path}")

    # ─── SAVE RESULTS ───────────────────────────────────────────────────
    out_dir = os.path.join(p, 'cardamomOT')
    os.makedirs(out_dir, exist_ok=True)

    if verb:
        print(f"[infer_mixture] Saving results to {out_dir}...")

    np.save(os.path.join(out_dir, 'modes'),              model.modes)
    np.save(os.path.join(out_dir, 'basins'),             model.basins)
    np.save(os.path.join(out_dir, 'proba_init'),         model.proba_init)
    np.save(os.path.join(out_dir, 'weights'),            model.weights)
    np.save(os.path.join(out_dir, 'mixture_parameters'), model.a)
    # Depth settings of the fit: every later step uses them (the NB parameters depend on them)
    import json
    with open(os.path.join(out_dir, 'mixture_depth.json'), 'w') as fh:
        json.dump({'use_depth_factor': bool(model.use_depth_factor), 'depth_model': str(model.depth_model)}, fh)

    with open(os.path.join(out_dir, 'pi_init.pkl'), 'wb') as f:
        pickle.dump(model.pi_init, f)

    # Per-sample mixtures of the integration (removed if no integration)
    for name in ('mixture_parameters_samples.npy', 'integration_report.csv'):
        if os.path.exists(os.path.join(out_dir, name)):
            os.remove(os.path.join(out_dir, name))
    if model.integration is not None:
        np.save(os.path.join(out_dir, 'mixture_parameters_samples'), model.integration['a_samples'])
        model.integration_report(list(adata.var_names)).to_csv(
            os.path.join(out_dir, 'integration_report.csv'), index=False)

    if verb:
        print("[infer_mixture] Inference complete. Results saved.")

if __name__ == "__main__":
    main(sys.argv[1:])
