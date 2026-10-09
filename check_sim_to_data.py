"""
check_sim_to_data.py
--------------------
Compare network simulations with observed data.

Validates the quality of the inferred gene regulatory network by comparing
simulated expression dynamics with the real data used for inference. Computes
zero-inflation adjusted distributions and generates comparison plots.

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


def growth_log_weights(R_opt, times_data):
    """
    Cumulative log mass of each trajectory state, L_n(t_k) = sum_{j<k} R_opt[j, n] dt_j: the growth
    the OT pass attributes to the path of slot n up to t_k (0 at the first time).
    """
    tu = np.sort(np.unique(times_data))
    T = len(tu)
    N = len(times_data) // T
    gain = np.nan_to_num(np.asarray(R_opt, dtype=float)[:T * N].reshape(T, N)[:-1] * np.diff(tu)[:, None])
    return np.vstack([np.zeros((1, N)), np.cumsum(gain, axis=0)]).ravel()


def growth_resample(L, times_data, samples, seed=0, valid=None):
    """
    Indices of the trajectory states drawn with weights exp(L) within each (sample, time): the
    trajectories with the expansion of their population, i.e. what a simulation with proliferation
    should reproduce (multinomial resampling, as in the branching simulation). Virtual states
    (valid = False: time not observed for their sample) are left as they are.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(L))
    samples = np.zeros(len(L), dtype=int) if samples is None else np.asarray(samples)[:len(L)]
    valid = np.ones(len(L), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    for t in np.unique(times_data):
        for s in np.unique(samples):
            g = np.flatnonzero((times_data == t) & (samples == s) & valid)
            if len(g) and np.ptp(L[g]) > 1e-12:   # uniform weights (first time): states kept as they are
                w = np.exp(L[g] - L[g].max())
                idx[g] = g[rng.choice(len(g), len(g), replace=True, p=w / w.sum())]
    return idx


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
        pi_zinb = np.load(os.path.join(p, 'cardamomOT', 'pi_zinb.npy'))
        samples_path = os.path.join(p, 'cardamomOT', 'data_samples.npy')
        samples_traj = np.load(samples_path).astype(int) if os.path.exists(samples_path) else None
        
        vect_kon_beta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_beta.npy')) + 1e-6
        vect_kon_theta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_forsimul.npy')) + 1e-6
        vect_kon_sim = np.load(os.path.join(p, 'cardamomOT', 'data_kon_simul.npy')) + 1e-6
        times_data = np.load(os.path.join(p, 'cardamomOT', 'data_times.npy'))
        times_simulation = np.load(os.path.join(p, 'cardamomOT', 'simulation_times.npy'))
        rna_ref = np.load(os.path.join(p, 'cardamomOT', 'data_rna.npy'))
        mrna_simul_path = os.path.join(p, 'cardamomOT', 'data_mrna_simul.npy')
        mrna_simul = np.load(mrna_simul_path) if os.path.exists(mrna_simul_path) and model.simulate_full_with_harissa else None
        if mrna_simul is not None:
            print("[check_sim_to_data] Harissa mRNA simulation found — will use directly (no NB sampling)")
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
    k1_tr, c_tr, pz_tr = nb_cell_parameters(mixture_parameters, pi_zinb, samples_traj)
    k1_sim, c_sim, pz_sim = nb_cell_parameters(mixture_parameters, pi_zinb, _sim_sample_idx(samples_traj, times_simulation))
    k1_tr, k1_sim = k1_tr + 1e-6, k1_sim + 1e-6
    # Depth factors (estimate_cell_depth.py): each state / simulated cell drawn at the depth of the
    # real cell it mimics, NB(k, c / s): p = c / (c + s)
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
    # RNA of the trajectories (slot n at t -> slot n at t+1): counts of the real cell behind each state
    if real_idx is not None and len(real_idx) == data_ref.shape[1] and np.all(real_idx >= 0):
        data_ref[1:, :] = data_rna_extracted[:, real_idx]

    # Generate data_sim: either directly from Harissa mRNAs or via NB sampling
    if mrna_simul is not None:
        # Harissa mode: mRNAs are already simulated — use them directly
        data_sim[1:, :] = mrna_simul[:, ns:].T
        print("[check_sim_to_data] data_sim built from Harissa mRNA simulation (no NB sampling)")
    else:
        # Standard mode: sample counts from the NB burst model
        zero_mask = (np.random.uniform(0, 1, (data_sim[1:, :].shape)) < pz_sim.T)
        zero_ratio_sim = np.sum(zero_mask == 1)/np.size(data_sim[1:, :])
        print(f"[check_sim_to_data] Simulation zero-inflation ratio: {zero_ratio_sim:.4f}")
        data_sim[1:, :] = np.random.negative_binomial((k1_sim*vect_kon_sim)[:, ns:].T, (c_sim / (c_sim + s_sim))[:, ns:].T)
        data_sim[1:, :] = np.where(zero_mask, 0, data_sim[1:, :])

    zero_mask = (np.random.uniform(0, 1, (data_beta[1:, :].shape)) < pz_tr.T)
    zero_ratio_beta = np.sum(zero_mask == 1)/np.size(data_beta[1:, :])
    print(f"[check_sim_to_data] Beta (mixture) zero-inflation ratio: {zero_ratio_beta:.4f}")
    data_beta[1:, :] = np.random.negative_binomial((k1_tr*vect_kon_beta)[:, ns:].T, (c_tr / (c_tr + s_tr))[:, ns:].T)
    data_beta[1:, :] = np.where(zero_mask, 0, data_beta[1:, :])

    zero_mask = (np.random.uniform(0, 1, (data_netw_theta[1:, :].shape)) < pz_tr.T)
    zero_ratio_theta = np.sum(zero_mask == 1)/np.size(data_netw_theta[1:, :])
    print(f"[check_sim_to_data] Theta (network) zero-inflation ratio: {zero_ratio_theta:.4f}")
    data_netw_theta[1:, :] = np.random.negative_binomial((k1_tr*vect_kon_theta)[:, ns:].T, (c_tr / (c_tr + s_tr))[:, ns:].T)
    data_netw_theta[1:, :] = np.where(zero_mask, 0, data_netw_theta[1:, :])

    # Same draws at the reference depth (s = 1): layer 'reference_depth', shown by the report with
    # cell_depth_for_representation (the observed data then divided by their depth factor)
    ref_depth = {}
    if depth_cells is not None:
        def _draw_ref(k1, kon, c, pz):
            x = np.random.negative_binomial((k1 * kon)[:, ns:].T, (c / (c + 1.0))[:, ns:].T)
            return np.where(np.random.uniform(0, 1, x.shape) < pz.T, 0, x).T
        ref_depth['beta'] = _draw_ref(k1_tr, vect_kon_beta, c_tr, pz_tr)
        ref_depth['theta'] = _draw_ref(k1_tr, vect_kon_theta, c_tr, pz_tr)
        if mrna_simul is None:
            ref_depth['sim'] = _draw_ref(k1_sim, vect_kon_sim, c_sim, pz_sim)

    cardamom_dir = os.path.join(p, 'cardamomOT')

    # Growth of the trajectories: the OT selection removes the proliferation (one descendant per
    # ancestor); a simulation with proliferation is compared with the growth-weighted trajectories
    flag_path = os.path.join(cardamom_dir, 'simulation_with_proliferation.npy')
    sim_prolif = bool(np.load(flag_path)[0]) if os.path.exists(flag_path) else False
    R_opt_path = os.path.join(cardamom_dir, 'data_R_opt.npy')
    R_opt = np.load(R_opt_path) if os.path.exists(R_opt_path) else None
    L_growth = growth_log_weights(R_opt, times_data) if (R_opt is not None and len(R_opt) == len(times_data)) else None
    valid_path = os.path.join(cardamom_dir, 'data_traj_valid.npy')
    traj_valid = np.load(valid_path) if os.path.exists(valid_path) else np.ones(len(times_data), dtype=bool)
    growth_idx = (growth_resample(L_growth, times_data, samples_traj, seed=model.seed or 0, valid=traj_valid)
                  if L_growth is not None else None)
    if sim_prolif:
        print("[check_sim_to_data] Simulation with proliferation: growth-weighted trajectories written "
              + ("(adata_*_growth_*.h5ad)" if growth_idx is not None else "— skipped, data_R_opt.npy missing"))

    sample_names = (sorted(adata.obs['dataset_id'].astype(str).unique()) if 'dataset_id' in adata.obs else None)

    def _with_samples(A, idx):
        # obs['dataset_id'] of each state / simulated cell (the classifier of cell_type is per sample)
        if sample_names is not None and idx is not None and len(idx) == A.n_obs:
            A.obs['dataset_id'] = np.asarray(sample_names)[np.minimum(np.asarray(idx, dtype=int), len(sample_names) - 1)]
        return A

    def _write_states(X, name, var_names=None):
        # Trajectory-state AnnData (+ growth weights), and its growth-resampled version if needed
        A = ad.AnnData(X=X)
        A.var = adata.var.copy()
        A.obs['time'] = times_data
        _with_samples(A, samples_traj)
        A.obs['observed'] = traj_valid  # False: virtual state (time not observed for its sample)
        if name in ref_depth:
            A.layers['reference_depth'] = ref_depth[name]
        if name == 'rna_traj' and 'depth_factor' in adata.obs and real_idx is not None:
            # Depth of the real cell behind each state (counts of the real cells), even if the run did not use it
            A.obs['depth_factor'] = state_depth(adata.obs['depth_factor'].values.astype(float), real_idx)
        if L_growth is not None:
            A.obs['growth_log_weight'] = L_growth
        A.write(os.path.join(cardamom_dir, f'adata_{name}_{tag}.h5ad'))
        if sim_prolif and growth_idx is not None:
            B = A[growth_idx].copy()
            B.obs_names_make_unique()
            B.obs['state'] = growth_idx
            B.uns['growth_resampled'] = True
            B.write(os.path.join(cardamom_dir, f'adata_{name}_growth_{tag}.h5ad'))

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
        if 'sim' in ref_depth:
            adata_sim.layers['reference_depth'] = ref_depth['sim']
        adata_sim.uns['proliferation'] = sim_prolif
        pop_path = os.path.join(cardamom_dir, 'data_log_population.npy')
        if sim_prolif and os.path.exists(pop_path):
            adata_sim.uns['log_population'] = np.load(pop_path)
        adata_sim.write(os.path.join(cardamom_dir, f'adata_sim_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad'))

        # Trajectory RNA (without proliferation), and growth-weighted if the simulation has proliferation
        _write_states(data_ref[1:, :].T, 'rna_traj')

        data_prot_traj = np.load(os.path.join(cardamom_dir, 'data_prot_forsimul.npy'))
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
            # Reference equivalent to the simulation: trajectories, growth-weighted with proliferation
            g = growth_idx if (sim_prolif and growth_idx is not None) else np.arange(data_ref.shape[1])
            ref_sc, beta_sc, theta_sc = data_ref[:, g], data_beta[:, g], data_netw_theta[:, g]
            plot_data_distrib(ref_sc, data_sim, t_data, t_simul, names, inputfile, outputfile, complement1)
            plot_data_umap_altogether(data_real, ref_sc, beta_sc, theta_sc,
                                  data_sim, t_data, t_simul, inputfile, 'Check', 'altogether_sim')
            print("[check_sim_to_data] Plots successfully generated")
        except Exception as e:
            print(f"[check_sim_to_data] Warning: Could not generate plots: {e}")

if __name__ == "__main__":
   main(sys.argv[1:])
