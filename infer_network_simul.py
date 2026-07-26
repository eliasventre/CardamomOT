"""
infer_network_simul.py
----------------------
Adapt inferred network parameters for simulation.

Loads the inferred network structure and adapts parameters to enable
simulation of gene expression dynamics. Handles reference network loading
and parameter transformation for downstream simulation steps.

Usage:
    python infer_network_simul.py -i <project_path> -s <split>

Required input files:
    - Data/data_<split>.h5ad: count matrix with temporal information
    - cardamom/inter.npy, basal.npy: inferred network parameters
    - cardamom/mixture_parameters.npy: burst kinetics parameters
    - cardamom/ref_network.csv: optional reference network (tab-separated)

Output files:
    - cardamom/data_prot_unitary.npy: adapted protein abundance
    - cardamom/data_kon_unitary.npy: adapted bursting parameters
    - cardamom/basal_simul.npy, inter_simul.npy: simulation-ready parameters
    - cardamom/basal_t_simul.npy, inter_t_simul.npy: temporal parameters
    - cardamom/ratios.npy: parameter ratios
    - cardamom/degradations_temporal.npy: temporal degradation rates
"""
import sys; sys.path += ['../']
import numpy as np
from CardamomOT import NetworkModel as NetworkModel_beta
import getopt
import anndata as ad
import pandas as pd
import os
import torch

verb = 1

def main(argv):
    """
    Adapt inferred network parameters for simulation.

    Args:
        argv: Command-line arguments (--input, --split).
    """
    inputfile = ''
    split = ''
    stimulus = -1.0
    prior = -1.0
    recompute_proliferations = False
    try:
        opts, args = getopt.getopt(argv, "hi:s:t:p:",
                                   ["input=", "split=", "stimulus=", "prior=", "compute-proliferation"])
    except getopt.GetoptError:
        print("[infer_network_simul] Error: Invalid command-line arguments")
        print("[infer_network_simul] Usage: python infer_network_simul.py -i <project_path> -s <split> "
              "[--stimulus <float>] [--prior <float>] [--compute-proliferation]")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-s", "--split"):
            split = '{}'.format(arg)
        elif opt in ("-t", "--stimulus"):
            stimulus = float(arg)
        elif opt in ("-p", "--prior"):
            prior = float(arg)
        elif opt == "--compute-proliferation":
            recompute_proliferations = True
        elif opt == "-h":
            print(__doc__)
            sys.exit(0)

    if not inputfile or not split:
        print("[infer_network_simul] Error: Missing required arguments --input and --split")
        sys.exit(1)

    p = '{}/'.format(inputfile)

    # Load gene expression data
    data_path = os.path.join(p, 'Data', 'data_{}.h5ad'.format(split))
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[infer_network_simul] Loaded data from {data_path}")
    except FileNotFoundError as e:
        print(f"[infer_network_simul] Error: {e}")
        sys.exit(1)

    # ─── LOAD STIMULUS SCHEDULE (optional) ──────────────────────────────
    stim_sched = None
    sched_path = os.path.join(p, 'Data', 'stimulus_schedule.txt')
    if os.path.exists(sched_path):
        stim_sched = np.loadtxt(sched_path)
        print(f"[infer_network_simul] Loaded stimulus schedule from {sched_path}")
    else:
        print("[infer_network_simul] No stimulus schedule found, using default")

    # ─── DETECT n_stimuli FROM SCHEDULE ─────────────────────────────────
    _stim_arr = np.asarray(stim_sched) if stim_sched is not None else None
    n_stimuli = int(_stim_arr.shape[1]) if (_stim_arr is not None and _stim_arr.ndim == 2) else 1
    print(f"[infer_network_simul] n_stimuli detected: {n_stimuli}")

    model = NetworkModel_beta(adata.shape[1], n_stimuli=n_stimuli)
    if stimulus >= 0:
        model.stimulus = stimulus
    if prior >= 0:
        model.prior_network_pen = prior
    print(f"[infer_network_simul] stimulus={model.stimulus}, prior_network_pen={model.prior_network_pen}")

    # Load inferred network parameters
    print("[infer_network_simul] Loading inferred network parameters...")
    try:
        model.d = np.load(os.path.join(p, 'cardamomOT', 'degradations.npy'))
        model.basal = np.load(os.path.join(p, 'cardamomOT', 'basal.npy'))
        model.inter = np.load(os.path.join(p, 'cardamomOT', 'inter.npy'))
        model.a = np.load(os.path.join(p, 'cardamomOT', 'mixture_parameters.npy'))
        model.modes = np.load(os.path.join(p, 'cardamomOT', 'modes.npy'))
        model.prot = np.load(os.path.join(p, 'cardamomOT', 'data_prot.npy'))
        model.rna = np.load(os.path.join(p, 'cardamomOT', 'data_rna.npy'))
        model.times_data = np.load(os.path.join(p, 'cardamomOT', 'data_times.npy'))
        model.samples_data = np.load(os.path.join(p, 'cardamomOT', 'data_samples.npy'))
        model.kon_theta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_theta.npy'))
        model.kon_beta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_beta.npy'))
        model.alpha = np.load(os.path.join(p, 'cardamomOT', 'alpha.npy'))
        model.proba_traj = np.load(os.path.join(p, 'cardamomOT', 'proba_traj.npy'))
        model.n_networks = np.load(os.path.join(p, 'cardamomOT', 'n_networks.npy'))
        kon_beta_h_path = os.path.join(p, 'cardamomOT', 'data_kon_beta_harissa.npy')
        if os.path.exists(kon_beta_h_path):
            model.kon_beta_harissa = np.load(kon_beta_h_path)
            print("[infer_network_simul] Loaded kon_beta_harissa for Harissa-mode network re-inference")
        R_opt_path = os.path.join(p, 'cardamomOT', 'data_R_opt.npy')
        if os.path.exists(R_opt_path):
            model.R_opt = np.load(R_opt_path)
            print("[infer_network_simul] Loaded R_opt for proliferation MLP training")
        print("[infer_network_simul] Successfully loaded all network parameters")
    except FileNotFoundError as e:
        print(f"[infer_network_simul] Error: Missing parameter file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[infer_network_simul] Error loading parameters: {e}")
        sys.exit(1)

    # Load reference network if available
    G_tot = model.inter.shape[0]
    ns = model.n_stimuli
    genes_only = [g.upper() for g in adata.var_names]   # no stimulus prefix
    model.ref_network = np.ones((G_tot, G_tot, model.n_networks))
    ref_path = os.path.join(p, 'cardamomOT', 'ref_network.csv')
    if os.path.exists(ref_path):
        print(f"[infer_network_simul] Loading reference network from {ref_path}")
        try:
            ref_df = pd.read_csv(ref_path, index_col=0)
            ref_df.columns = ref_df.columns.astype(str)
            ref_df.index = ref_df.index.astype(str)
            # CSV contains only genes (no stimulus rows/cols)
            common_genes = [g for g in genes_only if g in ref_df.index]
            if common_genes:
                sub_df = ref_df.loc[common_genes, [c for c in common_genes if c in ref_df.columns]]
                print(f"[infer_network_simul] ref_network gene block = {sub_df.shape}")
                ref_mat = np.abs(sub_df.to_numpy())
                row_idxs = [ns + genes_only.index(g) for g in sub_df.index]
                col_idxs = [ns + genes_only.index(g) for g in sub_df.columns]
                for n in range(model.n_networks):
                    for ii, ri in enumerate(row_idxs):
                        model.ref_network[ri, col_idxs, n] = ref_mat[ii, :]
                print(f"[infer_network_simul] Incorporated reference network with {len(common_genes)} genes")
            else:
                print("[infer_network_simul] Warning: no common genes found between CSV and data")
        except Exception as e:
            print(f"[infer_network_simul] Warning: Could not load reference network: {e}")
    else:
        print("[infer_network_simul] No reference network found, using inferred network only")

    model.ref_network = np.maximum(model.prior_network_pen, model.ref_network)
    model.ref_network[:ns, :] = model.stimulus

    # ─── LOAD INTER_SIMUL_REF (optional) ────────────────────────────────
    # Data/inter_simul_ref.npy or .csv — used as inter_ref in
    # refine_network_degradations (with final=0) instead of self.inter.
    # CSV: gene names in index/columns → named mapping; no gene names → positional
    # (assumes same gene order as adata.var_names, emits a warning).
    stim_labels = ['Stimulus'] if ns == 1 else [f'Stimulus_{i}' for i in range(ns)]
    genes_list = stim_labels + genes_only
    n_genes = G_tot - ns

    _sr_npy = os.path.join(p, 'Data', 'inter_simul_ref.npy')
    _sr_csv = os.path.join(p, 'Data', 'inter_simul_ref.csv')

    if os.path.exists(_sr_npy):
        arr_sr = np.load(_sr_npy)
        model.inter_simul_ref = arr_sr
        print(f"[infer_network_simul] Loaded inter_simul_ref from {_sr_npy} shape={arr_sr.shape} "
              f"— will be used as inter_ref with final=0")
    elif os.path.exists(_sr_csv):
        try:
            df_sr = pd.read_csv(_sr_csv, index_col=0)
            df_sr.columns = df_sr.columns.astype(str).str.upper()
            df_sr.index = df_sr.index.astype(str).str.upper()
            common_sr = [g for g in df_sr.index if g in genes_list]
            if common_sr:
                arr_sr = np.zeros((G_tot, G_tot))
                for row_g in common_sr:
                    for col_g in [c for c in df_sr.columns if c in genes_list]:
                        arr_sr[genes_list.index(row_g), genes_list.index(col_g)] = df_sr.loc[row_g, col_g]
                model.inter_simul_ref = arr_sr
                print(f"[infer_network_simul] Loaded inter_simul_ref from {_sr_csv} "
                      f"({len(common_sr)} genes matched) — will be used as inter_ref with final=0")
            else:
                # Positional fallback: no gene names recognised
                def _is_float(s):
                    try:
                        float(str(s).strip())
                        return True
                    except ValueError:
                        return False
                raw = pd.read_csv(_sr_csv, header=None, dtype=str)
                data = raw.values
                r0 = [str(v).strip() for v in data[0]]
                has_header = (not r0[0]) or not all(_is_float(v) for v in r0 if v)
                sr = 1 if has_header else 0
                if has_header and not r0[0]:
                    has_idx = True
                else:
                    c0 = [str(v).strip() for v in data[sr:, 0]]
                    has_idx = not all(_is_float(v) for v in c0 if v)
                sc = 1 if has_idx else 0
                vals = data[sr:, sc:].astype(float)
                if vals.shape == (G_tot, G_tot):
                    arr_sr = vals
                elif vals.shape == (n_genes, n_genes):
                    arr_sr = np.zeros((G_tot, G_tot))
                    arr_sr[ns:ns + n_genes, ns:ns + n_genes] = vals
                else:
                    print(
                        f"[infer_network_simul] ERROR: Data/inter_simul_ref.csv has no recognisable "
                        f"gene names and its shape {vals.shape} is inconsistent with {n_genes} genes "
                        f"(expected {n_genes}×{n_genes} without stimulus or {G_tot}×{G_tot} with "
                        f"stimulus). Add gene names as row/column headers or provide a matrix of the "
                        f"correct size."
                    )
                    arr_sr = None
                if arr_sr is not None:
                    import warnings as _warnings
                    _warnings.warn(
                        "Data/inter_simul_ref.csv has no gene names — assuming positional order "
                        "matches adata.var_names.",
                        UserWarning, stacklevel=2
                    )
                    print(
                        f"[infer_network_simul] WARNING: Data/inter_simul_ref.csv has no gene names "
                        f"— assuming gene order matches adata.var_names."
                    )
                    model.inter_simul_ref = arr_sr
                    print(f"[infer_network_simul] Loaded inter_simul_ref (positional) shape={arr_sr.shape} "
                          f"— will be used as inter_ref with final=0")
        except Exception as e:
            print(f"[infer_network_simul] Warning: could not load inter_simul_ref.csv: {e}")
    else:
        print("[infer_network_simul] No inter_simul_ref found in Data/, using inferred inter as reference")

    # Adapt parameters for simulation
    print("[infer_network_simul] Adapting parameters for simulation...")
    model.recompute_proliferations = recompute_proliferations
    model.refine_network_degradations(stimulus_schedule=stim_sched)
    print("[infer_network_simul] Parameter adaptation completed")

    # Save adapted parameters
    cardamom_dir = os.path.join(p, 'cardamomOT')
    try:
        np.save(os.path.join(cardamom_dir, 'data_prot_forsimul'), model.prot)
        np.save(os.path.join(cardamom_dir, 'data_kon_forsimul'), model.kon_theta)
        np.save(os.path.join(cardamom_dir, 'basal_simul'), model.basal)
        np.save(os.path.join(cardamom_dir, 'inter_simul'), model.inter)
        np.save(os.path.join(cardamom_dir, 'basal_t_simul'), model.basal_t)
        np.save(os.path.join(cardamom_dir, 'inter_t_simul'), model.inter_t)
        np.save(os.path.join(cardamom_dir, 'ratios'), model.ratios)
        np.save(os.path.join(cardamom_dir, 'degradations_temporal.npy'), model.d_t)
        if model.prolif_network is not None:
            torch.save(model.prolif_network.state_dict(),
                       os.path.join(cardamom_dir, 'prolif_network.pt'))
            np.save(os.path.join(cardamom_dir, 'prolif_network_n_proteins'),
                    np.array([model.prolif_network.net[0].in_features]))
            print("[infer_network_simul] Saved proliferation network to prolif_network.pt")
        print(f"[infer_network_simul] Successfully saved adapted parameters to {cardamom_dir}")
    except Exception as e:
        print(f"[infer_network_simul] Error saving parameters: {e}")
        sys.exit(1)

if __name__ == "__main__":
   main(sys.argv[1:])
