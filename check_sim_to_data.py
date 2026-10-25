"""
check_sim_to_data.py
--------------------
Compare network simulations with observed data.

Validates the quality of the inferred gene regulatory network by comparing
simulated expression dynamics with the real data used for inference. Draws NB
counts from the mixture and generates comparison plots.

Usage:
    python check_sim_to_data.py -i <project_path> [--stimulus <float>] [--prior <float>]

Required input files:
    - Data/data_<split>.h5ad: observed count matrix
    - cardamomOT/data_rna.npy: RNA expression from network inference
    - cardamomOT/data_prot_simul.npy: simulated protein abundance
    - cardamomOT/simulation_times.npy: timepoints used for simulation

Output files:
    - cardamomOT/adata_*_stim*.h5ad: generated AnnData objects for visualization
    - Check/sim_vs_data/ directory: distribution comparison plots
"""

import numpy as np
import sys
from CardamomOT.run_options import parse_step_options, settings, configure
import anndata as ad
from CardamomOT import NetworkModel, plot_data_umap_altogether, plot_data_distrib
from CardamomOT.inference.integration import nb_cell_parameters
from CardamomOT.inference.depth_model import nb_numpy_params
from CardamomOT.inference.depth import state_depth, simulation_depth
import scipy.sparse
import os

plot_in_script = 0


def _sim_sample_idx(samples_traj, times_simulation):
    """Sample index of each simulated cell: the N initial trajectories, repeated at each simulated time."""
    if samples_traj is None:
        return None
    N = int(np.sum(times_simulation == times_simulation[0]))
    return np.tile(np.asarray(samples_traj)[:N], len(times_simulation) // N)


def main(argv):
    """
    Compare network simulation results with observed data.

    Generates synthetic data from simulated network dynamics and compares
    expression distributions with the real observed data to assess network
    inference quality. Saves comparison datasets and generates visualizations.

    Args:
        argv: Command-line arguments (-i <project> and the options of run_options.STEP_OPTIONS).
    
    Returns:
        None. Saves comparison datasets and comparison plots.
    """
    opts = parse_step_options(argv, 'check_sim_to_data', __doc__)
    p = opts.p
    split = settings(opts).split
    inputfile = p  # plots write to <project>/Check

    outputfile = 'Check'
    complement1 = 'sim_vs_data'

    # Load observed expression data
    data_path = os.path.join(p, 'Data', 'data_{}.h5ad'.format(split))
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[check_sim_to_data] Loaded data from {data_path}")
    except FileNotFoundError as e:
        print(f"[check_sim_to_data] Error: {e}")
        sys.exit(1)
    
    # Extract count matrix
    if scipy.sparse.issparse(adata.X):
        data_rna_extracted = adata.X.T.toarray().astype(float)
    else:
        data_rna_extracted = np.asarray(adata.X.T, dtype=float)

    # Validate temporal information
    try:
        times = adata.obs['time'].values 
        if len(np.unique(times)) <= 1:
            raise ValueError("Data must contain multiple timepoints in obs['time']")
        print(f"[check_sim_to_data] Detected {len(np.unique(times))} observable timepoints")
    except KeyError:
        print("[check_sim_to_data] Error: data.obs['time'] not found")
        sys.exit(1)
    except ValueError as e:
        print(f"[check_sim_to_data] Error: {e}")
        sys.exit(1)
    
    data_real = np.vstack([times, data_rna_extracted]).astype(float)
    G = np.size(data_real, 0)-1
    model = NetworkModel(G)
    configure(model, opts)  # workbook, then the command-line options
    print(f"[check_sim_to_data] stimulus={model.stimulus}, prior_network_pen={model.prior_network_pen}")

    # Load mixture and simulation parameters
    print("[check_sim_to_data] Loading mixture and simulation parameters...")
    try:
        mixture_parameters = np.load(os.path.join(p, 'cardamomOT', 'mixture_parameters.npy'))
        samples_path = os.path.join(p, 'cardamomOT', 'data_samples.npy')
        samples_traj = np.load(samples_path).astype(int) if os.path.exists(samples_path) else None
        
        vect_kon_beta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_beta.npy')) + 1e-6
        # Network sample: kon of the network of fit_network on its trajectories
        vect_kon_theta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_theta.npy')) + 1e-6
        vect_kon_sim = np.load(os.path.join(p, 'cardamomOT', 'data_kon_simul.npy')) + 1e-6
        times_data = np.load(os.path.join(p, 'cardamomOT', 'data_times.npy'))
        times_simulation = np.load(os.path.join(p, 'cardamomOT', 'simulation_times.npy'))
        rna_ref = np.load(os.path.join(p, 'cardamomOT', 'data_rna.npy'))
        mrna_simul_path = os.path.join(p, 'cardamomOT', 'data_mrna_simul.npy')
        mrna_simul = np.load(mrna_simul_path) if os.path.exists(mrna_simul_path) and model.simulation_mrnas_direct else None
        if mrna_simul is not None:
            print("[check_sim_to_data] Explicit mRNA simulation found — used directly (no NB sampling)")
        print("[check_sim_to_data] Successfully loaded all parameters")
    except FileNotFoundError as e:
        print(f"[check_sim_to_data] Error: Missing parameter file: {e}")
        print("[check_sim_to_data] Please ensure network inference and simulation have been completed")
        sys.exit(1)
   
    names = adata.var_names
    t_data = list(set(times_data))
    t_data.sort()
    t_simul = list(set(times_simulation))
    t_simul.sort()
    print(f"[check_sim_to_data] Real data timepoints: {t_data}")
    print(f"[check_sim_to_data] Simulation timepoints: {t_simul}")

    # Generate synthetic data from mixture and simulation models
    print("[check_sim_to_data] Generating synthetic data for distribution comparison...")
    # n_stimuli inferred from rna_ref: columns 0..ns-1 are stimulus slots, ns..G_tot-1 are genes
    ns = rna_ref.shape[1] - G
    data_ref = np.zeros((G+1, np.size(rna_ref, 0)))
    N_traj_rna = data_ref.shape[1] // len(np.unique(times_data))
    for t, time in enumerate(np.sort(np.unique(times_data))):
        data_ref[0, t*N_traj_rna:(t+1)*N_traj_rna] = time
    data_ref[0, N_traj_rna*len(np.unique(times_data)):] = times_data[-1]
    data_ref[1:, :] = rna_ref[:, ns:].T

    data_beta = np.zeros((G+1, np.size(vect_kon_beta, 0)))
    data_netw_theta = np.zeros((G+1, np.size(vect_kon_theta, 0)))
    data_sim = np.zeros((G+1, np.size(vect_kon_sim, 0)))
    data_beta[0, :] = times_data[:]
    data_netw_theta[0, :] = times_data[:]
    data_sim[0, :] = times_simulation[:]

    # NB parameters of each state: its sample's mixture if per-sample (identical otherwise)
    k1_tr, c_tr = nb_cell_parameters(mixture_parameters, samples_traj)
    k1_sim, c_sim = nb_cell_parameters(mixture_parameters, _sim_sample_idx(samples_traj, times_simulation))
    k1_tr, k1_sim = k1_tr + 1e-6, k1_sim + 1e-6
    # Depth factors (estimate_cell_depth.py): each state / simulated cell drawn at the depth of the
    # real cell it mimics (depth_model: NB(k s, c) or NB(k, c / s))
    depth_cells = (adata.obs['depth_factor'].values.astype(float)
                   if (model.use_depth_factor and 'depth_factor' in adata.obs) else None)
    idx_path = os.path.join(p, 'cardamomOT', 'data_traj_real_idx.npy')
    real_idx = np.load(idx_path) if os.path.exists(idx_path) else None
    s_tr = state_depth(depth_cells, real_idx)
    s_sim = simulation_depth(depth_cells, real_idx, times_data, times_simulation)
    s_tr = 1.0 if s_tr is None else s_tr[:, None]
    s_sim = 1.0 if s_sim is None else s_sim[:, None]
    if depth_cells is not None:
        print("[check_sim_to_data] Counts drawn at the depth of the cells mimicked (obs['depth_factor'])")
    # RNA of the trajectories (slot n at t -> slot n at t+1): counts of the real cell behind each state, else the
    # trajectory mRNA, already at the reference depth
    ref_real = real_idx is not None and len(real_idx) == data_ref.shape[1] and np.all(real_idx >= 0)
    if ref_real:
        data_ref[1:, :] = data_rna_extracted[:, real_idx]

    # Generate data_sim: either directly from the simulated mRNAs (simulation_mrnas_direct) or via NB sampling
    if mrna_simul is not None:
        data_sim[1:, :] = mrna_simul[:, ns:].T
        print("[check_sim_to_data] data_sim built from the explicit mRNA simulation (no NB sampling)")
    else:
        # Standard mode: sample counts from the NB burst model
        n_nb, p_nb = nb_numpy_params(k1_sim * vect_kon_sim, c_sim, s_sim, model.depth_model)
        data_sim[1:, :] = np.random.negative_binomial(n_nb[:, ns:].T, p_nb[:, ns:].T)

    n_nb, p_nb = nb_numpy_params(k1_tr * vect_kon_beta, c_tr, s_tr, model.depth_model)
    data_beta[1:, :] = np.random.negative_binomial(n_nb[:, ns:].T, p_nb[:, ns:].T)

    n_nb, p_nb = nb_numpy_params(k1_tr * vect_kon_theta, c_tr, s_tr, model.depth_model)
    data_netw_theta[1:, :] = np.random.negative_binomial(n_nb[:, ns:].T, p_nb[:, ns:].T)

    # Depth each model draw was taken at (obs['depth_factor']): the report divides data and draws by it
    # (cell_depth_for_representation), the explicit mRNA simulation being at the reference depth
    depth_of = {}
    if depth_cells is not None:
        if np.ndim(s_tr):
            depth_of.update(beta=np.ravel(s_tr), theta=np.ravel(s_tr))
        if mrna_simul is not None:
            depth_of['sim'] = np.ones(data_sim.shape[1])
        elif np.ndim(s_sim):
            depth_of['sim'] = np.ravel(s_sim)

    cardamom_dir = os.path.join(p, 'cardamomOT')

    # Trajectory states are drawn at each time among the observed cells (target marginal of the OT): they follow the
    # observed composition, the reference of the simulations with or without proliferation (no reweighting)
    flag_path = os.path.join(cardamom_dir, 'simulation_with_proliferation.npy')
    sim_prolif = bool(np.load(flag_path)[0]) if os.path.exists(flag_path) else False
    valid_path = os.path.join(cardamom_dir, 'data_traj_valid.npy')
    traj_valid = np.load(valid_path) if os.path.exists(valid_path) else np.ones(len(times_data), dtype=bool)

    sample_names = (sorted(adata.obs['dataset_id'].astype(str).unique()) if 'dataset_id' in adata.obs else None)

    def _with_samples(A, idx):
        # obs['dataset_id'] of each state / simulated cell (the classifier of cell_type is per sample)
        if sample_names is not None and idx is not None and len(idx) == A.n_obs:
            A.obs['dataset_id'] = np.asarray(sample_names)[np.minimum(np.asarray(idx, dtype=int), len(sample_names) - 1)]
        return A

    def _write_states(X, name, var_names=None):
        # Trajectory-state AnnData
        A = ad.AnnData(X=X)
        A.var = adata.var.copy()
        A.obs['time'] = times_data
        _with_samples(A, samples_traj)
        A.obs['observed'] = traj_valid  # False: virtual state (time not observed for its sample)
        if name in depth_of:
            A.obs['depth_factor'] = depth_of[name]
        from CardamomOT.condition_genes import mask_stage
        mask_stage(A, adata, model_draws=name != 'rna_traj')  # genes outside each sample's condition: 0 (draws)
        if name == 'beta':
            # kon of each state (its mode per gene): mode-shuffle ceiling of the gene-pair correlations (report)
            A.layers['kon'] = np.asarray(vect_kon_beta[:, ns:], dtype=np.float32)
        if name == 'rna_traj' and 'depth_factor' in adata.obs and ref_real:
            # Depth of the real cell behind each state (counts of the real cells), even if the run did not use it;
            # none for the trajectory mRNA (reference depth)
            A.obs['depth_factor'] = state_depth(adata.obs['depth_factor'].values.astype(float), real_idx)
        if name == 'rna_traj' and real_idx is not None and len(real_idx) == A.n_obs:
            # Real cell behind each state (row of data_<split>): its out-of-fold cell type labels the state (report)
            A.obs['real_idx'] = np.asarray(real_idx, dtype=int)
            A.uns['real_counts'] = bool(ref_real)
        A.write(os.path.join(cardamom_dir, f'adata_{name}_{tag}.h5ad'))
        stale = os.path.join(cardamom_dir, f'adata_{name}_growth_{tag}.h5ad')  # growth-resampled states, no longer used
        if os.path.exists(stale):
            os.remove(stale)

    tag = f'stim{model.stimulus}_prior{model.prior_network_pen}'

    # Save comparison datasets
    print("[check_sim_to_data] Saving comparison datasets...")
    try:
        _write_states(data_beta[1:, ].T, 'beta')
        _write_states(data_netw_theta[1:, ].T, 'theta')

        adata_sim = ad.AnnData(X=data_sim[1:, ].T)
        adata_sim.var = adata.var.copy()
        adata_sim.obs['time'] = times_simulation
        _with_samples(adata_sim, _sim_sample_idx(samples_traj, times_simulation))
        if 'sim' in depth_of:
            adata_sim.obs['depth_factor'] = depth_of['sim']
        from CardamomOT.condition_genes import mask_stage
        mask_stage(adata_sim, adata)
        adata_sim.uns['proliferation'] = sim_prolif
        pop_path = os.path.join(cardamom_dir, 'data_log_population.npy')
        if sim_prolif and os.path.exists(pop_path):
            adata_sim.uns['log_population'] = np.load(pop_path)
        adata_sim.write(os.path.join(cardamom_dir, f'adata_sim_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad'))

        # Trajectory RNA
        _write_states(data_ref[1:, :].T, 'rna_traj')

        data_prot_traj = np.load(os.path.join(cardamom_dir, 'data_prot.npy'))  # trajectories of fit_network
        _write_states(data_prot_traj[:, ns:], 'prot_traj')

        data_prot_simul = np.load(os.path.join(cardamom_dir, 'data_prot_simul.npy'))
        adata_prot_simul = ad.AnnData(X=data_prot_simul[:, ns:])
        adata_prot_simul.var = adata.var.copy()
        adata_prot_simul.obs['time'] = times_simulation
        adata_prot_simul.write(os.path.join(cardamom_dir, f'adata_prot_simul_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad'))
        
        print(f"[check_sim_to_data] Successfully saved comparison datasets to {cardamom_dir}")
    except Exception as e:
        print(f"[check_sim_to_data] Error saving datasets: {e}")

    if plot_in_script:
        print("[check_sim_to_data] Generating distribution comparison plots...")
        try:
            ref_sc, beta_sc, theta_sc = data_ref, data_beta, data_netw_theta
            plot_data_distrib(ref_sc, data_sim, t_data, t_simul, names, inputfile, outputfile, complement1)
            plot_data_umap_altogether(data_real, ref_sc, beta_sc, theta_sc,
                                  data_sim, t_data, t_simul, inputfile, 'Check', 'altogether_sim')
            print("[check_sim_to_data] Plots successfully generated")
        except Exception as e:
            print(f"[check_sim_to_data] Warning: Could not generate plots: {e}")

if __name__ == "__main__":
   main(sys.argv[1:])
