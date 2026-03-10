"""
infer_mixture.py
----------------
Main CARDAMOM mixture model inference pipeline.

This script infers the burst kinetics parameters (mixture model) from temporal
scRNA-seq data. When adata.obs contains a 'rd' column (computed by infer_rd.py),
it is automatically used as a cell-level read-depth correction factor in the
negative binomial inference, addressing the bias from variable sequencing depth
across cells (see Fang, Gorin & Pachter 2024).

Usage:
    python infer_mixture.py -i <project_path> -s <split> [-m <mean_threshold>]

Recommended workflow:
    1) python infer_rd.py      -i <project_path> -s <split>  # estimate read depth
    2) python infer_mixture.py -i <project_path> -s <split>  # uses read depth
"""

import sys
sys.path += ['../']
import numpy as np
from CardamomOT import NetworkModel as NetworkModel_beta
import anndata as ad
import getopt
import scipy.sparse
import os
import pickle

verb = 1


def main(argv):
    """
    Main function to run the mixture model inference pipeline.

    Loads temporal scRNA-seq data, estimates burst kinetics using a Poisson-gamma
    or zero-inflated negative binomial mixture model, and saves calibrated parameters.

    Args:
        argv: Command-line arguments (--input, --split, --mean).
    """
    inputfile  = ''
    split      = ''
    mean_forcing = -1

    try:
        opts, args = getopt.getopt(argv, "hi:s:m:", ["input=", "split=", "mean="])
    except getopt.GetoptError:
        print("Error: Invalid arguments. Use: infer_mixture.py -i <project> -s <split> [-m <threshold>]")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        if opt in ("-s", "--split"):
            split = '{}'.format(arg)
        if opt in ("-m", "--mean"):
            mean_forcing = float(arg)

    p = '{}/'.format(inputfile)

    data_path = os.path.join(p, 'Data', 'data_{}.h5ad'.format(split))
    if os.path.exists(data_path):
        adata = ad.read_h5ad(data_path)
        if verb:
            print(f"[infer_mixture] Loaded data from {data_path}")
    else:
        error_msg = (
            f"Error: Data file not found at {data_path}.\n"
            "Create a 'Data' folder in your project directory and place "
            f"a count table named 'data_{split}.h5ad'."
        )
        print(error_msg)
        raise FileNotFoundError(error_msg)

    if scipy.sparse.issparse(adata.X):
        data_rna_extracted = adata.X.T.toarray()
    else:
        data_rna_extracted = adata.X.T

    # ─── CHECK TEMPORAL INFORMATION ──────────────────────────────────────
    try:
        times = adata.obs['time'].values
        if len(np.unique(times)) <= 1:
            raise ValueError(
                "Data must contain temporal information with at least 2 distinct timepoints. "
                "Ensure adata.obs['time'] is present."
            )
    except KeyError as e:
        error_msg = f"Error: 'time' column not found in adata.obs. {e}"
        print(error_msg)
        raise SystemExit(error_msg)
    except ValueError as e:
        print(f"Error: {e}")
        raise SystemExit(e)

    data_rna = np.vstack([times, data_rna_extracted]).T
    G = np.size(data_rna, 1)

    # ─── LOAD READ DEPTH CORRECTION (optional) ──────────────────────────
    if 'rd' in adata.obs.columns:
        cell_rd = np.asarray(adata.obs['rd'].values, dtype=float)
        # Safety: normalize to unit mean
        cell_rd = cell_rd / (np.mean(cell_rd) + 1e-16)
        if verb:
            min_rd, med_rd, max_rd = cell_rd.min(), np.median(cell_rd), cell_rd.max()
            cv2_rd = np.var(cell_rd) / (np.mean(cell_rd)**2)
            print(f"[infer_mixture] Read-depth correction enabled:")
            print(f"  min={min_rd:.3f}  median={med_rd:.3f}  max={max_rd:.3f}  CV²={cv2_rd:.4f}")
    else:
        cell_rd = np.ones(adata.shape[0], dtype=float)
        if verb:
            print("[infer_mixture] No 'rd' column in adata.obs.")
            print("  Tip: Run infer_rd.py first to estimate read-depth factors.")
            print("  Using standard NB inference without read-depth correction.")

    # ─── INFER MIXTURE MODEL ────────────────────────────────────────────
    model = NetworkModel_beta(G - 1)
    if mean_forcing >= 0:
        model.mean_forcing_em = mean_forcing
        if verb:
            print(f"[infer_mixture] Mean forcing threshold set to {mean_forcing}")
    
    if verb:
        print(f"[infer_mixture] Starting mixture model inference ({G-1} genes)...")
    
    model.fit_mixture(
        data_rna,
        gene_names=list(adata.var_names),
        min_components=2,
        max_components=2,
        max_iter_kinetics=0,
        cell_rd=cell_rd,
        verb=verb,
    )

    # ─── SAVE RESULTS ───────────────────────────────────────────────────
    out_dir = os.path.join(p, 'cardamomOT')
    os.makedirs(out_dir, exist_ok=True)

    if verb:
        print(f"[infer_mixture] Saving results to {out_dir}...")

    np.save(os.path.join(out_dir, 'modes'),              model.modes)
    np.save(os.path.join(out_dir, 'proba'),              model.proba)
    np.save(os.path.join(out_dir, 'proba_init'),         model.proba_init)
    np.save(os.path.join(out_dir, 'n_networks'),         model.n_networks)
    np.save(os.path.join(out_dir, 'weights'),            model.weights)
    np.save(os.path.join(out_dir, 'mixture_parameters'), model.a)
    np.save(os.path.join(out_dir, 'pi_zinb'),            model.pi)
    np.save(os.path.join(out_dir, 'cell_rd'),            cell_rd)

    with open(os.path.join(out_dir, 'pi_init.pkl'), 'wb') as f:
        pickle.dump(model.pi_init, f)

    if verb:
        print("[infer_mixture] Inference complete. Results saved.")

if __name__ == "__main__":
    main(sys.argv[1:])
