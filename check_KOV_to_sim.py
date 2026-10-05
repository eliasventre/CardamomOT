"""
check_KOV_to_sim.py
-------------------
Validate knockouts/overexpressions by comparing simulations to observations.

Compares simulated gene expression under various perturbations (KO/OV)
with observed wildtype data. Generates AnnData objects for each perturbation
condition for downstream analysis and visualization.

Usage:
    python check_KOV_to_sim.py -i <project_path> [--stimulus <float>] [--prior <float>]

Required input files:
    - Data/data_<split>.h5ad: observed count matrix (wildtype)
    - the perturbation_simulation sheet: perturbations
    - cardamomOT/data_prot_simul_KO_*.npy: simulated proteins for each perturbation
    - cardamomOT/data_kon_simul_KO_*.npy: simulated bursting for each perturbation

Output files:
    - cardamomOT/adata_sim_KO_*.h5ad: AnnData objects for each perturbation
    - cardamomOT/adata_prot_simul_KO_*.h5ad: Protein trajectories for each perturbation
"""
import re
import numpy as np
import sys
from CardamomOT.run_options import parse_step_options, settings, configure
import anndata as ad
from CardamomOT import NetworkModel
from CardamomOT.inputs import input_dir
from CardamomOT.inference.integration import nb_cell_parameters
from CardamomOT.inference.depth import simulation_depth
import scipy.sparse
import os
from CardamomOT.inputs import depth_factor_used


# Shared loader (KO / OV / STIM columns), kept under the old names for other scripts
from CardamomOT.tools.perturbations import (find_perturbation_file, load_perturbations, combo_label,
                                            parse_gene_with_pct as _parse_gene_with_pct, gene_label as _gene_label)


def load_ko_ov_combinations(file_path, genes=None):
    return load_perturbations(file_path, genes)


def _sim_sample_idx(samples_traj, times_simulation):
    """Sample index of each simulated cell: the N initial trajectories, repeated at each simulated time."""
    if samples_traj is None:
        return None
    N = int(np.sum(times_simulation == times_simulation[0]))
    return np.tile(np.asarray(samples_traj)[:N], len(times_simulation) // N)


def main(argv):
    """
    Validate simulated perturbations against observed data.

    Loads simulated expression data for each KO/OV combination and
    creates AnnData objects for visualization and comparison with
    wildtype observations.

    Args:
        argv: Command-line arguments (-i <project> and the options of run_options.STEP_OPTIONS).
    
    Returns:
        None. Saves validation datasets to cardamomOT/ directory.
    """
    opts = parse_step_options(argv, 'check_KOV_to_sim', __doc__)
    p = opts.p
    split = settings(opts).split

    # Load perturbations (KO / OV / STIM)
    ko_ov_file = find_perturbation_file(input_dir(p))
    try:
        # Gene names (to read stimulus targets as in simulate_network_KOV)
        _genes = list(ad.read_h5ad(os.path.join(p, 'Data', f'data_{split}.h5ad'), backed='r').var_names)
        combos = load_ko_ov_combinations(ko_ov_file, _genes)
        if len(combos) == 0:
            print(f"[check_KOV_to_sim] No perturbation found in {ko_ov_file}")
            sys.exit(0)
        print(f"[check_KOV_to_sim] Loaded {len(combos)} KO/OV combinations")
    except FileNotFoundError as e:
        print(f"[check_KOV_to_sim] Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[check_KOV_to_sim] Error loading KO/OV combinations: {e}")
        sys.exit(1)

    # Load observed data
    data_path = os.path.join(p, 'Data', f'data_{split}.h5ad')
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Observed data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[check_KOV_to_sim] Loaded observed data from {data_path}")
        print(f"[check_KOV_to_sim] Dataset contains {adata.shape[0]} cells and {adata.shape[1]} genes")
    except FileNotFoundError as e:
        print(f"[check_KOV_to_sim] Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[check_KOV_to_sim] Error loading observed data: {e}")
        sys.exit(1)

    if scipy.sparse.issparse(adata.X):
        data_rna_extracted = adata.X.T.toarray()
    else:
        data_rna_extracted = adata.X.T

    # Validate temporal information
    try:
        times = adata.obs['time'].values 
        if len(np.unique(times)) <= 1:
            raise ValueError("Dataset must contain temporal information with multiple timepoints")
        print(f"[check_KOV_to_sim] Found {len(np.unique(times))} unique timepoints: {sorted(np.unique(times))}")
    except (KeyError, ValueError) as e:
        print(f"[check_KOV_to_sim] Error: {e}")
        sys.exit(1)

    data_real = np.vstack([times, data_rna_extracted]).astype(float)

    # Load model parameters
    try:
        mixture_parameters = np.load(os.path.join(p, 'cardamomOT', 'mixture_parameters.npy'))
        pi_zinb = np.load(os.path.join(p, 'cardamomOT', 'pi_zinb.npy'))
        samples_path = os.path.join(p, 'cardamomOT', 'data_samples.npy')
        samples_traj = np.load(samples_path).astype(int) if os.path.exists(samples_path) else None
        times_simulation = np.load(os.path.join(p, 'cardamomOT', 'simulation_times.npy'))
        t_simul = list(set(times_simulation))
        t_simul.sort()
        print(f"[check_KOV_to_sim] Loaded model parameters and simulation times")
    except FileNotFoundError as e:
        print(f"[check_KOV_to_sim] Error: Missing parameter file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[check_KOV_to_sim] Error loading model parameters: {e}")
        sys.exit(1)

    G = np.size(data_real, 0) - 1
    # n_stimuli inferred from mixture_parameters: columns 0..ns-1 are stimulus slots
    ns = mixture_parameters.shape[-1] - G
    # NB parameters of each simulated cell: its sample's mixture if per-sample (identical otherwise)
    k1_sim, c_sim, pz_sim = nb_cell_parameters(mixture_parameters, pi_zinb, _sim_sample_idx(samples_traj, times_simulation))
    # Depth factors: each simulated cell drawn at the depth of the real cell it mimics (p = c / (c + s))
    depth_cells = (adata.obs['depth_factor'].values.astype(float)
                   if (depth_factor_used(p) and 'depth_factor' in adata.obs) else None)
    idx_path = os.path.join(p, 'cardamomOT', 'data_traj_real_idx.npy')
    times_path = os.path.join(p, 'cardamomOT', 'data_times.npy')
    s_sim = (simulation_depth(depth_cells, np.load(idx_path), np.load(times_path), times_simulation)
             if depth_cells is not None and os.path.exists(idx_path) and os.path.exists(times_path) else None)
    s_sim = 1.0 if s_sim is None else s_sim[:, None]

    model = NetworkModel(G)
    configure(model, opts)  # workbook, then the command-line options
    print(f"[check_KOV_to_sim] stimulus={model.stimulus}, prior_network_pen={model.prior_network_pen}")

    # Create AnnData objects for each perturbation combination
    print(f"[check_KOV_to_sim] Creating AnnData objects for {len(combos)} KO/OV combinations")

    for idx, combo in enumerate(combos, start=1):
        kos = combo["KO"]
        ovs = combo["OV"]

        label = combo_label(combo)
        print(f"[check_KOV_to_sim] Processing combination {idx}/{len(combos)}: {label}")
        
        file_prefix = os.path.join(p, f"cardamomOT/data_kon_simul_{label}.npy")
        prot_prefix = os.path.join(p, f"cardamomOT/data_prot_simul_{label}.npy")

        if not os.path.exists(file_prefix):
            print(f"[check_KOV_to_sim] Warning: Simulation data missing for {label}, file {file_prefix} not found. Skipping.")
            continue

        try:
            vect_kon_sim = np.load(file_prefix)
            print(f"[check_KOV_to_sim] Loaded simulation data for {label}")
        except Exception as e:
            print(f"[check_KOV_to_sim] Error loading simulation data for {label}: {e}")
            continue
        # Generate simulated expression data
        data_sim = np.zeros((G+1, np.size(vect_kon_sim, 0)))
        data_sim[0, :] = times_simulation[:]

        # Generate negative binomial noise + sparsity
        zero_mask = (np.random.uniform(0, 1, data_sim[1:, :].shape) < pz_sim.T)
        print(f'[check_KOV_to_sim] New zeros ratio for {label}: {np.sum(zero_mask == 1)/np.size(data_sim[1:, :]):.3f}')

        data_sim[1:, :] = np.random.negative_binomial(
            (k1_sim * vect_kon_sim)[:, ns:].T,
            (c_sim / (c_sim + s_sim))[:, ns:].T
        )
        data_sim[1:, :] = np.where(zero_mask, 0, data_sim[1:, :])

        # Create AnnData object for simulated RNA
        adata_sim = ad.AnnData(X=data_sim[1:, ].T)
        adata_sim.var = adata.var.copy()
        adata_sim.obs["combo_label"] = label
        adata_sim.obs['time'] = times_simulation
        # Population size of the branching simulation (log, relative to t0, per simulated time)
        pop_path = os.path.join(p, 'cardamomOT', f'data_log_population_{label}.npy')
        if os.path.exists(pop_path):
            adata_sim.uns['log_population'] = np.load(pop_path)

        # Save simulated RNA data
        sim_rna_path = os.path.join(p, f'cardamomOT/adata_sim_{label}_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad')
        try:
            adata_sim.write(sim_rna_path)
            print(f"[check_KOV_to_sim] Saved simulated RNA data: {os.path.basename(sim_rna_path)}")
        except Exception as e:
            print(f"[check_KOV_to_sim] Error saving simulated RNA data: {e}")
            continue

        # Load and save simulated protein data
        if os.path.exists(prot_prefix):
            try:
                data_prot_simul = np.load(prot_prefix)
                adata_prot_simul = ad.AnnData(X=data_prot_simul[:, ns:])
                adata_prot_simul.var = adata.var.copy()
                adata_prot_simul.obs['time'] = times_simulation
                
                prot_path = os.path.join(p, f'cardamomOT/adata_prot_simul_{label}_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad')
                adata_prot_simul.write(prot_path)
                print(f"[check_KOV_to_sim] Saved simulated protein data: {os.path.basename(prot_path)}")
            except Exception as e:
                print(f"[check_KOV_to_sim] Error processing protein data for {label}: {e}")
        else:
            print(f"[check_KOV_to_sim] Warning: Protein simulation data not found for {label}")

    print("[check_KOV_to_sim] KO/OV validation completed successfully")


if __name__ == "__main__":
    main(sys.argv[1:])
