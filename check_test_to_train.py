"""
check_test_to_train.py
----------------------
Validate test-set inference by comparing predicted dynamics with observations.

Loads the AnnData objects produced by infer_test.py and generates distribution
comparison plots between real test data and the inferred/simulated trajectories.

Usage:
    python check_test_to_train.py -i <project_path> -s <split> [-t <stimulus>] [-p <prior>]

Required input files (produced by infer_test.py):
    - Data/data_<split>.h5ad: observed test data
    - cardamomOT/adata_beta_test_stim<s>_prior<p>.h5ad
    - cardamomOT/adata_theta_test_stim<s>_prior<p>.h5ad
    - cardamomOT/adata_sim_test_stim<s>_prior<p>.h5ad
    - cardamomOT/adata_rna_traj_test_stim<s>_prior<p>.h5ad

Output files:
    - Check/test_to_train_stim<s>_prior<p>/: distribution comparison plots
    - Check/altogether_test_stim<s>_prior<p>/: UMAP comparison plots
"""
import numpy as np
import sys
import getopt
import anndata as ad
from CardamomOT import plot_data_distrib, plot_data_umap_altogether
import scipy.sparse
import os


def main(argv):
    """
    Validate test-set inference using pre-built AnnData objects from infer_test.py.

    Loads trajectory and simulation AnnData objects, converts them to array format,
    and generates comparison plots against real observed test data.

    Args:
        argv: Command-line arguments (--input, --split, --stim, --prior).
    """
    inputfile = ''
    split = ''
    stim = 1.0
    prior = 1.0
    try:
        opts, args = getopt.getopt(argv, "hi:s:t:p:", ["input=", "split=", "stimulus=", "prior="])
    except getopt.GetoptError:
        print("[check_test_to_train] Error: Invalid command-line arguments")
        print("[check_test_to_train] Usage: python check_test_to_train.py -i <project_path>"
              " -s <split> [-t <stimulus>] [-p <prior>]")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-s", "--split"):
            split = '{}'.format(arg)
        elif opt in ("-t", "--stimulus"):
            stim = float(arg)
        elif opt in ("-p", "--prior"):
            prior = float(arg)
        elif opt == "-h":
            print(__doc__)
            sys.exit(0)

    if not inputfile or not split:
        print("[check_test_to_train] Error: Missing required arguments --input and --split")
        sys.exit(1)

    p = '{}/'.format(inputfile)
    cardamom_dir = os.path.join(p, 'cardamomOT')

    # ─── LOAD OBSERVED TEST DATA ─────────────────────────────────────────
    data_path = os.path.join(p, 'Data', 'data_{}.h5ad'.format(split))
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Test data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[check_test_to_train] Loaded test data from {data_path}")
        print(f"[check_test_to_train] Dataset: {adata.shape[0]} cells, {adata.shape[1]} genes")
    except FileNotFoundError as e:
        print(f"[check_test_to_train] Error: {e}")
        sys.exit(1)

    # Read depth normalization: divide data_real in memory, never touch adata
    if 'rd' in adata.obs.columns:
        cell_rd = np.clip(np.asarray(adata.obs['rd'].values, dtype=float), 1e-6, None)
        print(f"[check_test_to_train] Read depth correction loaded from adata.obs['rd']")
    else:
        cell_rd = np.ones(adata.shape[0], dtype=float)

    if scipy.sparse.issparse(adata.X):
        data_rna_extracted = adata.X.T.toarray().astype(float)
    else:
        data_rna_extracted = np.asarray(adata.X.T, dtype=float)
    data_rna_extracted /= cell_rd[np.newaxis, :]   # (G, N) / (1, N)

    try:
        times = adata.obs['time'].values
        if len(np.unique(times)) <= 1:
            raise ValueError("Dataset must contain temporal information with multiple timepoints")
        print(f"[check_test_to_train] Found {len(np.unique(times))} unique timepoints")
    except (KeyError, ValueError) as e:
        print(f"[check_test_to_train] Error: {e}")
        sys.exit(1)

    data_real = np.vstack([times, data_rna_extracted]).astype(float)

    # ─── LOAD PRE-BUILT ANNDATA OBJECTS ──────────────────────────────────
    tag = f'stim{stim}_prior{prior}'
    adata_files = {
        'beta':  f'adata_beta_test_{tag}.h5ad',
        'theta': f'adata_theta_test_{tag}.h5ad',
        'sim':   f'adata_sim_test_{tag}.h5ad',
    }

    adatas = {}
    try:
        for key, fname in adata_files.items():
            fpath = os.path.join(cardamom_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Required AnnData file not found: {fpath}\n"
                                        "Run infer_test.py first.")
            adatas[key] = ad.read_h5ad(fpath)
        print(f"[check_test_to_train] Loaded pre-built AnnData objects")
    except FileNotFoundError as e:
        print(f"[check_test_to_train] Error: {e}")
        sys.exit(1)

    # Convert adatas to array format expected by plotting functions
    # data_* shape: (G+1, N_cells) — row 0 = times, rows 1: = gene expressions
    def _adata_to_array(a):
        X = a.X.toarray() if scipy.sparse.issparse(a.X) else np.asarray(a.X)
        return np.vstack([a.obs['time'].values, X.T])

    data_ref = data_real  # use original observed data (not deconvolved trajectories)
    data_beta = _adata_to_array(adatas['beta'])
    data_netw_theta = _adata_to_array(adatas['theta'])
    data_sim = _adata_to_array(adatas['sim'])

    names = adata.var_names
    t_data = sorted(np.unique(adatas['beta'].obs['time'].values).tolist())
    t_simul = sorted(np.unique(adatas['sim'].obs['time'].values).tolist())
    print(f"[check_test_to_train] Trajectory timepoints: {t_data}")
    print(f"[check_test_to_train] Simulation timepoints: {t_simul}")

    # ─── GENERATE PLOTS ──────────────────────────────────────────────────
    tag = f'stim{stim}_prior{prior}'
    print(f"[check_test_to_train] Generating validation plots (stim={stim}, prior={prior})...")
    try:
        plot_data_distrib(data_real, data_sim, t_data, t_simul, names,
                          inputfile, 'Check', f'test_to_train_{tag}')
        plot_data_umap_altogether(data_real, data_ref, data_beta, data_netw_theta,
                                  data_sim, t_data, t_simul, inputfile, 'Check',
                                  f'altogether_test_{tag}',
                                  cell_rd=cell_rd)
        print(f"[check_test_to_train] Generated comparison plots in Check/test_to_train_{tag}/")
    except Exception as e:
        print(f"[check_test_to_train] Warning: Error generating plots: {e}")

    print("[check_test_to_train] Test set validation completed successfully")


if __name__ == "__main__":
    main(sys.argv[1:])
