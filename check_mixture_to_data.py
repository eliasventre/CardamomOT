"""
check_mixture_to_data.py
------------------------
Validate mixture model predictions against real data.

Compares the inferred mixture model (burst kinetics) predictions with 
observed expression data through distribution analysis and optimal
transport distance metrics.

Usage:
    python check_mixture_to_data.py -i <project_path>   (split: Model_parameters sheet)

Required input files:
    - Data/data_<split>.h5ad: count matrix with temporal information
    - cardamomOT/mixture_parameters.npy: inferred burst kinetics parameters
    - cardamomOT/modes.npy: mode of burst frequency distribution
    - cardamomOT/pi_zinb.npy: zero-inflation probabilities

Output files:
    - cardamomOT/adata_beta.h5ad: simulated data from mixture model
    - Check/mixture_vs_data/ directory: comparison plots
"""

import numpy as np
import sys
from CardamomOT.run_options import parse_step_options, settings, configure
import anndata as ad
from CardamomOT import plot_data_umap_toref, plot_data_distrib, check_stationary
from CardamomOT.inference.integration import nb_cell_parameters
import scipy.sparse
import os
from CardamomOT.inputs import depth_factor_used
import ot

plot_in_script = 0
compute_ot = 0

def main(argv):
    """
    Compare mixture model predictions with observed data distribution.

    Generates synthetic data from the inferred mixture parameters and
    computes optimal transport (Wasserstein) distance to quantify the
    quality of the burst kinetics inference.

    Args:
        argv: Command-line arguments (-i <project> and the options of run_options.STEP_OPTIONS).
    
    Returns:
        None. Saves comparison data and prints OT distance metric.
    """
    opts = parse_step_options(argv, 'check_mixture_to_data', __doc__)
    p = opts.p
    split = settings(opts).split
    inputfile = p  # plots write to <project>/Check

    outputfile = 'Check'
    complement1 = 'mixture_vs_data'

    # Load observed expression data
    data_path = os.path.join(p, 'Data', 'data_{}.h5ad'.format(split))
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[check_mixture_to_data] Loaded data from {data_path}")
    except FileNotFoundError as e:
        print(f"[check_mixture_to_data] Error: {e}")
        print(f"[check_mixture_to_data] Please ensure Data/data_{split}.h5ad exists in {p}")
        sys.exit(1)
    
    # Extract count matrix
    if scipy.sparse.issparse(adata.X):
        data_rna_extracted = adata.X.T.toarray().astype(float)
    else:
        data_rna_extracted = np.asarray(adata.X.T, dtype=float)

    # Temporal information (absent or single timepoint = stationary setting)
    stationary = check_stationary(adata)
    times = adata.obs['time'].values.astype(float)
    if stationary:
        print("[check_mixture_to_data] Stationary data (no or single timepoint)")
    else:
        print(f"[check_mixture_to_data] Detected {len(np.unique(times))} timepoints")
    
    data_real = np.vstack([times, data_rna_extracted]).astype(float)

    # Load mixture model parameters
    print("[check_mixture_to_data] Loading mixture model parameters...")
    try:
        mixture_parameters = np.load(os.path.join(p, 'cardamomOT', 'mixture_parameters.npy'))
        pi_zinb = np.load(os.path.join(p, 'cardamomOT', 'pi_zinb.npy'))
        vect_kon_beta = np.load(os.path.join(p, 'cardamomOT', 'modes.npy')) + 1e-6
        print("[check_mixture_to_data] Successfully loaded mixture parameters")
    except FileNotFoundError as e:
        print(f"[check_mixture_to_data] Error: Missing parameter file: {e}")
        print("[check_mixture_to_data] Please ensure mixture inference has been completed")
        sys.exit(1)

    times_data = times.copy()
    names = adata.var_names
    t_data = list(set(times_data))
    t_data.sort()
    print(f"[check_mixture_to_data] Using {len(t_data)} unique timepoints: {t_data}")

    # Generate synthetic data from mixture model
    print("[check_mixture_to_data] Generating synthetic data from mixture model...")
    G = np.size(data_real, 0)-1
    # n_stimuli inferred from mixture_parameters: columns 0..ns-1 are stimulus slots
    ns = mixture_parameters.shape[-1] - G
    # NB parameters of each cell (its sample's mixture if per-sample)
    ids = adata.obs['dataset_id'].values if 'dataset_id' in adata.obs else None
    sample_idx = np.searchsorted(np.unique(ids), ids) if ids is not None else None
    k1c, cc, pzc = nb_cell_parameters(mixture_parameters, pi_zinb, sample_idx)
    print(f"[check_mixture_to_data] n_stimuli inferred: {ns}")
    data_beta = np.zeros((G+1, np.size(vect_kon_beta, 0)))
    data_beta[0, :] = times_data[:]

    # Apply zero-inflation
    zero_mask = (np.random.uniform(0, 1, (data_beta[1:, :].shape)) < pzc.T)
    zero_ratio = np.sum(zero_mask == 1)/np.size(data_beta[1:, :])
    print(f"[check_mixture_to_data] Applied zero-inflation with ratio: {zero_ratio:.4f}")

    # Sample from negative binomial distribution (exclude stimulus columns ns:)
    # Depth factors: each cell drawn at its own depth, NB(k, c / s)
    s = (adata.obs['depth_factor'].values.astype(float)[:, None]
         if (depth_factor_used(p) and 'depth_factor' in adata.obs) else 1.0)
    data_beta[1:, :] = np.random.negative_binomial(((k1c + 1e-6)*vect_kon_beta)[:, ns:].T, (cc / (cc + s))[:, ns:].T)
    data_beta[1:, :] = np.where(zero_mask, 0, data_beta[1:, :])

    # Save synthetic data 
    adata_beta = ad.AnnData(X=data_beta[1:, :].T)
    adata_beta.var = adata.var.copy()
    adata_beta.obs['time'] = times_data
    adata_beta.write(os.path.join(p, 'cardamomOT', 'adata_beta.h5ad'))
    print(f"[check_mixture_to_data] Saved synthetic data to {os.path.join(p, 'cardamomOT', 'adata_beta.h5ad')}")

    if compute_ot:
        # Compute optimal transport distance (Wasserstein)
        print("[check_mixture_to_data] Computing optimal transport distance...")
        N_cells = len(times_data)
        try:
            ot_distance = ot.emd2(np.ones(N_cells)/N_cells, np.ones(N_cells)/N_cells, 
                                ot.dist(data_beta[1:].T, data_real[1:].T), numItermax=100000)
            print(f"[check_mixture_to_data] Optimal transport distance (Wasserstein): {ot_distance:.6f}")
        except Exception as e:
            print(f"[check_mixture_to_data] Error computing OT distance: {e}")

    if plot_in_script:
        print("[check_mixture_to_data] Generating comparison plots...")
        plot_data_distrib(data_real, data_beta, t_data, t_data, names, inputfile, outputfile, complement1)
        plot_data_umap_toref(data_real, data_beta, t_data, inputfile, 'Check', 'umap_mixture')
        print("[check_mixture_to_data] Plots saved")

if __name__ == "__main__":
   main(sys.argv[1:])
