"""
check_mixture_to_data.py
------------------------
Validate mixture model predictions against real data.

Compares the inferred mixture model (burst kinetics) predictions with 
observed expression data through distribution analysis and optimal
transport distance metrics.

Usage:
    python check_mixture_to_data.py -i <project_path> -s <split>

Required input files:
    - Data/data_<split>.h5ad: count matrix with temporal information
    - cardamom/mixture_parameters.npy: inferred burst kinetics parameters
    - cardamom/modes.npy: mode of burst frequency distribution
    - cardamom/pi_zinb.npy: zero-inflation probabilities

Output files:
    - cardamom/adata_beta.h5ad: simulated data from mixture model
    - Check/mixture_vs_data/ directory: comparison plots
"""

import numpy as np
import sys, getopt
import anndata as ad
from CardamomOT import plot_data_distrib
import scipy.sparse
import os
import ot

plot_in_script = 0

def main(argv):
    """
    Compare mixture model predictions with observed data distribution.

    Generates synthetic data from the inferred mixture parameters and
    computes optimal transport (Wasserstein) distance to quantify the
    quality of the burst kinetics inference.

    Args:
        argv: Command-line arguments (--input, --split).
    
    Returns:
        None. Saves comparison data and prints OT distance metric.
    """
    inputfile = ''
    split = ''
    try:
        opts, args = getopt.getopt(argv, "hi:s:", ["input=", "split="])
    except getopt.GetoptError:
        print("[check_mixture_to_data] Error: Invalid command-line arguments")
        print("[check_mixture_to_data] Usage: python check_mixture_to_data.py -i <project_path> -s <split>")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-s", "--split"):
            split = '{}'.format(arg)
        elif opt == "-h":
            print(__doc__)
            sys.exit(0)

    if not inputfile or not split:
        print("[check_mixture_to_data] Error: Missing required arguments --input and --split")
        sys.exit(1)

    p = '{}/'.format(inputfile)

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
    
    # Load read depth correction (optional)
    try:
        cell_rd = np.load(os.path.join(p, 'cardamom', 'cell_rd.npy'))
        print(f"[check_mixture_to_data] Loaded read depth corrections for {len(cell_rd)} cells")
    except FileNotFoundError:
        print("[check_mixture_to_data] Read depth file not found, assuming uniform depth")
        cell_rd = np.ones(adata.shape[0], dtype=float)

    # Apply read depth normalization
    if cell_rd is not None:
        if scipy.sparse.issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = np.asarray(adata.X, dtype=float)
        adata.X[:, :] = np.round(X / cell_rd.reshape(-1, 1)).astype(int)[:, :]
        adata.write(data_path)
        
    # Extract count matrix
    if scipy.sparse.issparse(adata.X):
        data_rna_extracted = adata.X.T.toarray()
    else:
        data_rna_extracted = adata.X.T
    
    # Validate temporal information
    try:
        times = adata.obs['time'].values 
        if len(np.unique(times)) <= 1:
            raise ValueError("Data must contain multiple timepoints in obs['time']")
        print(f"[check_mixture_to_data] Detected {len(np.unique(times))} timepoints")
    except KeyError:
        print("[check_mixture_to_data] Error: data.obs['time'] not found")
        sys.exit(1)
    except ValueError as e:
        print(f"[check_mixture_to_data] Error: {e}")
        sys.exit(1)
    
    data_real = np.vstack([times, data_rna_extracted]).astype(float)

    # Load mixture model parameters
    print("[check_mixture_to_data] Loading mixture model parameters...")
    try:
        mixture_parameters = np.load(os.path.join(p, 'cardamom', 'mixture_parameters.npy'))
        c = mixture_parameters[-1, :]
        kz = mixture_parameters[:-1, :] + 1e-6
        pi_zinb = np.load(os.path.join(p, 'cardamom', 'pi_zinb.npy'))
        vect_kon_beta = np.load(os.path.join(p, 'cardamom', 'modes.npy')) + 1e-6
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
    data_beta = np.zeros((G+1, np.size(vect_kon_beta, 0)))
    data_beta[0, :] = times_data[:]

    # Apply zero-inflation
    zero_mask = (np.random.uniform(0, 1, (data_beta[1:, :].shape)) < pi_zinb.reshape((G, 1)))
    zero_ratio = np.sum(zero_mask == 1)/np.size(data_beta[1:, :])
    print(f"[check_mixture_to_data] Applied zero-inflation with ratio: {zero_ratio:.4f}")
    
    # Sample from negative binomial distribution
    data_beta[1:, :] = np.random.negative_binomial((np.max(kz, 0)*vect_kon_beta)[:, 1:].T, (c / (c+1))[1:].reshape(G, 1))
    data_beta[1:, :] = np.where(zero_mask, 0, data_beta[1:, :])

    # Save synthetic data 
    adata_beta = ad.AnnData(X=data_beta[1:, :].T)
    adata_beta.var = adata.var.copy()
    adata_beta.obs['time'] = times_data
    adata_beta.write(os.path.join(p, 'cardamom', 'adata_beta.h5ad'))
    print(f"[check_mixture_to_data] Saved synthetic data to {os.path.join(p, 'cardamom', 'adata_beta.h5ad')}")

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
        print("[check_mixture_to_data] Plots saved")

if __name__ == "__main__":
   main(sys.argv[1:])
