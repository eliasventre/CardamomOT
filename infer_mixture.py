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
import os
import pickle

verb = 1


def main(argv):
    """
    Main function to run the mixture model inference pipeline.

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

    # ─── LOAD STIMULUS SCHEDULE (optional) ──────────────────────────────
    stim_sched = None
    sched_path = os.path.join(p, 'Data', 'stimulus_schedule.txt')
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
    if mean_forcing >= 0:
        model.mean_forcing_em = mean_forcing
        if verb:
            print(f"[infer_mixture] Mean forcing threshold set to {mean_forcing}")

    if verb:
        print(f"[infer_mixture] Starting mixture model inference ({adata.shape[1]} genes)...")

    model.fit_mixture(
        adata,
        gene_names=list(adata.var_names),
        min_components=2,
        max_components=2,
        max_iter_kinetics=0,
        verb=verb,
        stimulus_schedule=stim_sched,
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
    np.save(os.path.join(out_dir, 'pi_zinb'),            model.pi_zinb)

    with open(os.path.join(out_dir, 'pi_init.pkl'), 'wb') as f:
        pickle.dump(model.pi_init, f)

    if verb:
        print("[infer_mixture] Inference complete. Results saved.")

if __name__ == "__main__":
    main(sys.argv[1:])
