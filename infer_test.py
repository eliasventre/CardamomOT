"""
infer_test.py
-------------
Infer trajectories and simulate on test set using pre-learned model parameters.

Held-out validation with everything learned on the training cells kept fixed:
1. basins of the test cells from the training mixture parameters (per sample);
2. protein trajectories by the inference loop with the core network fixed (inter.npy /
   basal.npy): OT couplings and basin updates combining EMD and network, continuing the
   training schedule (last training iteration n_iter_inference.npy: same basin weights and
   low Sinkhorn regularization as at the end of the training);
3. trajectories and kon_theta recomputed with the simulation network (*_simul.npy), then
   simulation from the test cells at the first timepoint;
4. AnnData objects equivalent to the training ones (compared to Data/data_test.h5ad by
   check_test_to_train.py and in the final report).
Per-sample parameters (mixtures, basals) are routed to the cells of each sample, never averaged.

Usage:
    python infer_test.py -i <project_path>

Required input files (from training pipeline):
    - Data/data_test.h5ad: test count matrix with temporal information
    - cardamomOT/mixture_parameters.npy, n_networks.npy: mixture model parameters
    - cardamomOT/pi_zinb.npy: zero-inflation parameters
    - cardamomOT/inter.npy, basal.npy: core inferred network (used for infer_test)
    - cardamomOT/basal_simul.npy, inter_simul.npy: adapted network parameters
    - cardamomOT/basal_t_simul.npy, inter_t_simul.npy: temporal network parameters
    - cardamomOT/ratios.npy, degradations.npy, degradations_temporal.npy: kinetics

Output files:
    - cardamomOT/data_prot_test.npy: protein trajectories for test cells
    - cardamomOT/data_rna_test.npy: RNA trajectories for test cells
    - cardamomOT/data_times_test.npy, data_samples_test.npy: trajectory metadata
    - cardamomOT/data_kon_beta_test.npy, data_kon_theta_test.npy: kon parameters
    - cardamomOT/proba_traj_test.npy: trajectory mode probabilities
    - cardamomOT/data_prot_simul_test.npy, data_kon_simul_test.npy: simulations
    - cardamomOT/simulation_times_test.npy: simulation timepoints
    - cardamomOT/adata_*_test_stim*_prior*.h5ad: AnnData comparison objects
"""
import sys; sys.path += ['../']
import os
import pickle
import numpy as np
import anndata as ad
import pandas as pd
from CardamomOT import NetworkModel as NetworkModel_beta, find_data_file
from CardamomOT.inputs import input_dir
from CardamomOT.inference.integration import nb_cell_parameters
from CardamomOT.inference.depth import state_depth, simulation_depth
import getopt
from CardamomOT.config import find_stimulus_schedule, simulation_schedule


def main(argv):
    """
    Infer test-set trajectories and produce all comparison AnnData objects.

    Args:
        argv: Command-line arguments (--input).
    """
    inputfile = ''
    stimulus = -1.0
    prior = -1.0
    force_basins = -1
    temporal_basins = -1
    try:
        opts, args = getopt.getopt(argv, "hi:t:p:f:b:",
                                   ["input=", "stimulus=", "prior=",
                                    "force-basins=", "temporal-basins="])
    except getopt.GetoptError:
        print("[infer_test] Error: Invalid command-line arguments")
        print("[infer_test] Usage: python infer_test.py -i <project_path> "
              "[--stimulus <float>] [--prior <float>] [--force-basins <int>] [--temporal-basins <int>]")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-t", "--stimulus"):
            stimulus = float(arg)
        elif opt in ("-p", "--prior"):
            prior = float(arg)
        elif opt in ("-f", "--force-basins"):
            force_basins = int(arg)
        elif opt in ("-b", "--temporal-basins"):
            temporal_basins = int(arg)
        elif opt == "-h":
            print(__doc__)
            sys.exit(0)

    if not inputfile:
        print("[infer_test] Error: Missing required argument --input")
        sys.exit(1)

    p = '{}/'.format(inputfile)
    cardamom_dir = os.path.join(p, 'cardamomOT')

    # ─── LOAD TEST DATA ──────────────────────────────────────────────────
    data_path = os.path.join(p, 'Data', 'data_test.h5ad')
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Test data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[infer_test] Loaded test data from {data_path}")
        print(f"[infer_test] Dataset: {adata.shape[0]} cells, {adata.shape[1]} genes")
    except FileNotFoundError as e:
        print(f"[infer_test] Error: {e}")
        sys.exit(1)

    try:
        times_obs = adata.obs['time'].values
        if len(np.unique(times_obs)) <= 1:
            raise ValueError("Dataset must contain multiple timepoints")
        print(f"[infer_test] Found {len(np.unique(times_obs))} timepoints: {sorted(np.unique(times_obs))}")
    except (KeyError, ValueError) as e:
        print(f"[infer_test] Error: {e}")
        sys.exit(1)

    # ─── STIMULUS SCHEDULES ──────────────────────────────────────────────
    stim_sched = None
    sched_path = (find_stimulus_schedule(input_dir(p))
                  or os.path.join(input_dir(p), 'stimulus_schedule_inference.txt'))
    if os.path.exists(sched_path):
        stim_sched = np.loadtxt(sched_path)
        print(f"[infer_test] Loaded stimulus schedule from {sched_path}")

    # Inference stimuli in simulation: first columns of stimulus_schedule_simulate.txt
    _ns = int(np.asarray(stim_sched).shape[1]) if stim_sched is not None and np.ndim(stim_sched) == 2 else 1
    stim_sched_simul, _ = simulation_schedule(input_dir(p), _ns)
    if stim_sched_simul is None:
        stim_sched_simul = stim_sched

    # ─── DETECT n_stimuli FROM SCHEDULE ─────────────────────────────────
    _stim_arr = np.asarray(stim_sched) if stim_sched is not None else None
    n_stimuli = int(_stim_arr.shape[1]) if (_stim_arr is not None and _stim_arr.ndim == 2) else 1
    print(f"[infer_test] n_stimuli detected: {n_stimuli}")

    # ─── INITIALIZE MODEL AND LOAD TRAINING PARAMETERS ──────────────────
    model = NetworkModel_beta(adata.shape[1], n_stimuli=n_stimuli)
    if stimulus >= 0:
        model.stimulus = stimulus
    if prior >= 0:
        model.prior_network_pen = prior
    if force_basins >= 0:
        model.force_basins = force_basins
    if temporal_basins >= 0:
        model.temporal_basins = temporal_basins
    model.apply_project_parameters(p)  # Data/CardamomOT_inputs.xlsx dominates the options
    print(f"[infer_test] Initialized model with {adata.shape[1]} genes, "
          f"stimulus={model.stimulus}, prior_network_pen={model.prior_network_pen}, "
          f"force_basins={model.force_basins}, temporal_basins={model.temporal_basins}")

    try:
        model.a = np.load(os.path.join(cardamom_dir, 'mixture_parameters.npy'))
        model.n_networks = int(np.load(os.path.join(cardamom_dir, 'n_networks.npy')))
        model.d = np.load(os.path.join(cardamom_dir, 'degradations.npy'))
        pi_zinb = np.load(os.path.join(cardamom_dir, 'pi_zinb.npy'))
        model.pi_zinb = pi_zinb   # per-gene zero-inflation (used by fit_mixture_test as ZINB prior)
        pi_init_path = os.path.join(cardamom_dir, 'pi_init.pkl')
        if os.path.exists(pi_init_path):
            with open(pi_init_path, 'rb') as f:
                model.pi_init = pickle.load(f)
            print(f"[infer_test] Loaded pi_init (training mode proportions) from {pi_init_path}")
        print(f"[infer_test] Loaded mixture and degradation parameters")
    except FileNotFoundError as e:
        print(f"[infer_test] Error: Missing mixture parameter file: {e}")
        sys.exit(1)

    # ─── TRAINING SAMPLES OF THE TEST CELLS (per-sample parameters) ────────
    # Per-sample arrays follow the sorted dataset_id of the training cells; keep the test samples' rows
    test_ids = np.sort(adata.obs['dataset_id'].unique()) if 'dataset_id' in adata.obs else np.array([0])
    train_ids = test_ids
    train_path = os.path.join(p, 'Data', 'data_train.h5ad')
    if os.path.exists(train_path) and 'dataset_id' in adata.obs:
        train_ids = np.sort(ad.read_h5ad(train_path, backed='r').obs['dataset_id'].unique())
    missing_s = [s for s in test_ids if s not in set(train_ids)]
    if missing_s:
        print(f"[infer_test] Error: test samples {missing_s} absent from the training cells")
        sys.exit(1)
    rows_s = [int(np.flatnonzero(train_ids == s)[0]) for s in test_ids]

    def _sample_rows(arr, axis=0):
        # Rows of the test samples when arr has one entry per training sample along axis
        arr = np.asarray(arr)
        if arr.ndim > axis and arr.shape[axis] == len(train_ids) and len(train_ids) > 1:
            return np.take(arr, rows_s, axis=axis)
        return arr

    if model.a.ndim == 3:
        model.a = _sample_rows(model.a)
        pi_zinb = _sample_rows(pi_zinb)
        model.pi_zinb = pi_zinb

    # ─── LOAD CORE NETWORK (raw CardamomOT infer_network output) ────────────
    # inter.npy / basal.npy are the main result of CardamomOT inference and
    # are used for trajectory coupling on test cells.  The "simul" variants are
    # post-processed versions adapted for simulation and are loaded separately.
    try:
        inter_core = np.load(os.path.join(cardamom_dir, 'inter.npy'))
        basal_core = np.load(os.path.join(cardamom_dir, 'basal.npy'))
        print(f"[infer_test] Loaded core inferred network (inter.npy, basal.npy)")
    except FileNotFoundError:
        print(f"[infer_test] Warning: core network files (inter.npy / basal.npy) not found; "
              f"falling back to simul parameters for trajectory inference")
        inter_core = None
        basal_core = None

    # ─── LOAD POST-PROCESSING SIMULATION PARAMETERS ─────────────────────────
    try:
        basal_simul = np.load(os.path.join(cardamom_dir, 'basal_simul.npy'))
        inter_simul = np.load(os.path.join(cardamom_dir, 'inter_simul.npy'))
        basal_t_simul = np.load(os.path.join(cardamom_dir, 'basal_t_simul.npy'))
        inter_t_simul = np.load(os.path.join(cardamom_dir, 'inter_t_simul.npy'))
        ratios = np.load(os.path.join(cardamom_dir, 'ratios.npy'))
        d_t = np.load(os.path.join(cardamom_dir, 'degradations_temporal.npy'))
        print(f"[infer_test] Loaded post-processing simulation parameters")
    except FileNotFoundError as e:
        print(f"[infer_test] Error: Missing simulation parameter file: {e}")
        sys.exit(1)

    # ─── SET CORE NETWORK FOR TRAJECTORY INFERENCE ──────────────────────────
    # infer_test uses the raw inferred network (inter/basal), not the simul ones;
    # basal (n_samples, G_tot, n_networks): rows of the test samples
    if inter_core is not None and basal_core is not None:
        model.basal = _sample_rows(basal_core)
        model.inter = inter_core
    else:
        model.basal = _sample_rows(basal_simul)
        model.inter = inter_simul
    basal_simul = _sample_rows(basal_simul)
    basal_t_simul = _sample_rows(basal_t_simul, axis=1)  # (T-1, n_samples, G_tot, n_networks)

    # Last training iteration: the test loop continues its schedule with the network fixed
    it_path = os.path.join(cardamom_dir, 'n_iter_inference.npy')
    n_iter_offset = int(np.load(it_path)[0]) if os.path.exists(it_path) else None
    if n_iter_offset is None:
        print("[infer_test] Warning: n_iter_inference.npy not found (older run): "
              f"test loop starts at iteration {model.min_n_loops}")

    # d_t and ratios are needed by estimate_trajectories inside infer_test
    model.ratios = ratios
    model.d_t = d_t

    # Set ref_network (all ones = no structural prior for test)
    G_tot = adata.shape[1] + model.n_stimuli
    model.ref_network = np.ones((G_tot, G_tot, model.n_networks))

    # Build stimulus schedule
    times_unique = np.sort(np.unique(times_obs))
    model._stim_schedule = model._build_stimulus_schedule(times_unique, stim_sched)

    # ─── LOAD OPTIONAL PER-SAMPLE KO/OV PRIOR ───────────────────────────
    basal_ref_test = None
    kov_path = os.path.join(input_dir(p), 'KO_OV_inference.txt')
    if os.path.exists(kov_path) and 'dataset_id' in adata.obs:
        try:
            ns = model.n_stimuli
            G_tot_test = adata.shape[1] + ns
            n_nw = model.n_networks
            stim_labels = ['Stimulus'] if ns == 1 else [f'Stimulus_{i}' for i in range(ns)]
            genes_only = [g.upper() for g in adata.var_names]

            df_kov = pd.read_csv(kov_path, sep='\t', dtype=str).fillna('')
            df_kov.columns = [c.strip().upper() for c in df_kov.columns]
            if 'DATASET_ID' in df_kov.columns:
                df_kov = df_kov.rename(columns={'DATASET_ID': 'SAMPLE_ID'})

            unique_samples = np.sort(adata.obs['dataset_id'].unique())
            n_samp = len(unique_samples)
            sample_to_idx = {str(s): i for i, s in enumerate(unique_samples)}

            basal_ref_test = np.zeros((n_samp, G_tot_test, n_nw))

            def _parse_genes(cell):
                if not cell or cell in ('0', 'nan', 'NAN'):
                    return []
                return [g.strip().upper() for g in cell.split(',') if g.strip()]

            for _, row in df_kov.iterrows():
                sid = str(row.get('SAMPLE_ID', '')).strip()
                if sid not in sample_to_idx:
                    continue
                s_idx = sample_to_idx[sid]
                for gene in _parse_genes(row.get('KO', '')):
                    if gene in genes_only:
                        gi = ns + genes_only.index(gene)
                        basal_ref_test[s_idx, gi, :] = -100.0
                for gene in _parse_genes(row.get('OV', '')):
                    if gene in genes_only:
                        gi = ns + genes_only.index(gene)
                        basal_ref_test[s_idx, gi, :] = 100.0
            print(f"[infer_test] Loaded per-sample KOV prior from {kov_path}")
        except Exception as e:
            print(f"[infer_test] Warning: could not load KO_OV_inference.txt: {e}")
            basal_ref_test = None

    # ─── LOAD OPTIONAL TRANSITION RATES ─────────────────────────────────
    transition_rates_test = None
    tr_path = find_data_file(input_dir(p), 'transition_rates')
    if tr_path is not None:
        transition_rates_test = pd.read_csv(tr_path, sep=None, engine='python', index_col=0)
        transition_rates_test.index = transition_rates_test.index.astype(str)
        transition_rates_test.columns = transition_rates_test.columns.astype(str)
        print(f"[infer_test] Loaded transition rates from {tr_path} shape={transition_rates_test.shape}")

    # ─── TRAJECTORY INFERENCE ON TEST SET ────────────────────────────────
    # Classifies cells into modes (fixed kz/c) then infers OT couplings with
    # the core inferred network (compute_theta=False, update_modes=True).
    print(f"[infer_test] Running mixture classification and trajectory inference...")
    try:
        model.infer_test(adata, verb=1, stimulus_schedule=stim_sched,
                         basal_ref=basal_ref_test,
                         transition_rates=transition_rates_test,
                         n_iter_offset=n_iter_offset)
        print(f"[infer_test] Test trajectory inference completed")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[infer_test] Error during inference: {e}")
        sys.exit(1)

    # ─── RESTORE SIMUL NETWORK FOR POST-PROCESSING ───────────────────────
    # Trajectory couplings were inferred with the core network; now load the
    # post-processed simul parameters so that estimate_trajectories and
    # kon_theta are computed with the correct unitary-scale network.
    model.basal = basal_simul
    model.inter = inter_simul
    model.basal_t = basal_t_simul
    model.inter_t = inter_t_simul
    model.ratios = ratios
    model.d_t = d_t

    # ─── RESTRICTED POST-PROCESSING: ESTIMATE TRAJECTORIES + KON_THETA ──
    # Runs only the estimate_trajectories step (updates protein paths along the
    # inferred OT couplings) and recomputes kon_theta with the simul network.
    # No network inference, MLP training, or degradation update is performed.
    print(f"[infer_test] Running restricted post-processing (estimate_trajectories + kon_theta)...")
    try:
        model.refine_network_degradations(test=True, stimulus_schedule=stim_sched)
        print(f"[infer_test] Restricted post-processing completed")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[infer_test] Error during restricted post-processing: {e}")
        sys.exit(1)

    # ─── SAVE TRAJECTORY OUTPUTS ─────────────────────────────────────────
    try:
        np.save(os.path.join(cardamom_dir, 'data_prot_test'), model.prot)
        np.save(os.path.join(cardamom_dir, 'data_rna_test'), model.rna)
        np.save(os.path.join(cardamom_dir, 'data_times_test'), model.times_data)
        np.save(os.path.join(cardamom_dir, 'data_samples_test'), model.samples_data)
        np.save(os.path.join(cardamom_dir, 'data_kon_beta_test'), model.kon_beta)
        np.save(os.path.join(cardamom_dir, 'data_kon_theta_test'), model.kon_theta)
        np.save(os.path.join(cardamom_dir, 'proba_traj_test'), model.proba_traj)
        print(f"[infer_test] Saved test trajectory outputs")
    except Exception as e:
        print(f"[infer_test] Error saving trajectory outputs: {e}")
        sys.exit(1)

    # Keep copies before simulate_network overwrites model.prot / model.kon_theta
    prot_traj_test = model.prot.copy()
    times_data_test = model.times_data.copy()
    kon_beta_test = model.kon_beta.copy()
    kon_theta_test = model.kon_theta.copy()
    rna_test = model.rna.copy()
    real_idx_test = np.asarray(model.traj_real_idx).copy()  # test cell behind each trajectory state

    # ─── SIMULATION TIMES ────────────────────────────────────────────────
    times_file = os.path.join(input_dir(p), 'times_to_simulate.txt')
    if os.path.exists(times_file):
        with open(times_file, "r") as f:
            sim_times = [float(line.strip()) for line in f if line.strip()]
        if sim_times[0] != 0:
            sim_times = [0.0] + sim_times
        print(f"[infer_test] Loaded simulation times from file: {sim_times}")
    else:
        sim_times = sorted(np.unique(model.times_data).tolist())
        print(f"[infer_test] Using inferred timepoints for simulation: {sim_times}")

    # ─── NETWORK SIMULATION ───────────────────────────────────────────────
    print(f"[infer_test] Simulating network dynamics on test set...")
    try:
        model.simulate_network(sim_times, stimulus_schedule=stim_sched_simul)
        print(f"[infer_test] Network simulation completed")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[infer_test] Error during simulation: {e}")
        sys.exit(1)

    times_simulation_test = model.times_simul

    try:
        np.save(os.path.join(cardamom_dir, 'data_prot_simul_test'), model.prot)
        np.save(os.path.join(cardamom_dir, 'data_kon_simul_test'), model.kon_theta)
        np.save(os.path.join(cardamom_dir, 'simulation_times_test'), times_simulation_test)
        print(f"[infer_test] Saved simulation outputs")
    except Exception as e:
        print(f"[infer_test] Error saving simulation outputs: {e}")
        sys.exit(1)

    # ─── BUILD ANNDATA COMPARISON OBJECTS ────────────────────────────────
    print("[infer_test] Building AnnData comparison objects...")
    try:
        ns = model.n_stimuli
        G = adata.shape[1]   # number of genes (no stimulus)


        vect_kon_beta = kon_beta_test + 1e-6        # (N_traj, G_tot)
        vect_kon_theta = kon_theta_test + 1e-6
        vect_kon_sim = model.kon_theta + 1e-6       # (N_sim, G_tot) after simulation

        # Helper: NB sample matrix of shape (G, N) using pi_zinb sparsity (per-sample mixtures if any)
        def _nb_sample(kon, times_vec, sample_idx=None, depth=None):
            n_cells = kon.shape[0]
            k1c, cc, pzc = nb_cell_parameters(model.a, pi_zinb, sample_idx)
            n_param = ((k1c + 1e-6) * kon)[:, ns:].T  # (G, N)
            sd = 1.0 if depth is None else depth[:, None]  # NB(k, c / s): p = c / (c + s)
            p_param = (cc / (cc + sd))[:, ns:].T
            n_param = np.maximum(n_param, 1e-6)
            p_param = np.clip(p_param, 1e-6, 1 - 1e-6)
            zero_mask = np.random.uniform(0, 1, (G, n_cells)) < pzc.T
            counts = np.random.negative_binomial(n_param, p_param)
            counts = np.where(zero_mask, 0, counts)
            out = np.zeros((G + 1, n_cells))
            out[0, :] = times_vec
            out[1:, :] = counts
            return out

        s_test = np.asarray(model.samples_data).astype(int) if model.samples_data is not None else None
        N0 = int(np.sum(times_simulation_test == times_simulation_test[0]))
        s_sim = None if s_test is None else np.tile(s_test[:N0], len(times_simulation_test) // N0)
        # Depth factors: states / simulated cells drawn at the depth of the test cells they mimic
        depth_cells = adata.obs['depth_factor'].values.astype(float) if 'depth_factor' in adata.obs else None
        d_tr = state_depth(depth_cells, real_idx_test)
        d_sim = simulation_depth(depth_cells, real_idx_test, times_data_test, times_simulation_test)
        data_beta = _nb_sample(vect_kon_beta, times_data_test, s_test[:len(vect_kon_beta)] if s_test is not None else None, d_tr)
        data_netw_theta = _nb_sample(vect_kon_theta, times_data_test, s_test[:len(vect_kon_theta)] if s_test is not None else None, d_tr)
        data_sim = _nb_sample(vect_kon_sim, times_simulation_test, s_sim, d_sim)

        # RNA trajectory data
        data_rna_traj = np.zeros((G + 1, rna_test.shape[0]))
        data_rna_traj[0, :] = times_data_test
        data_rna_traj[1:, :] = rna_test[:, ns:].T

        stim = model.stimulus
        prior = model.prior_network_pen

        def _make_adata(matrix_2d, obs_times, suffix):
            """matrix_2d: (G, N) — rows=genes, cols=cells."""
            a = ad.AnnData(X=matrix_2d.T)
            a.var = adata.var.copy()
            a.obs['time'] = obs_times
            a.write(os.path.join(cardamom_dir,
                                 f'adata_{suffix}_test_stim{stim}_prior{prior}.h5ad'))
            print(f"[infer_test] Saved adata_{suffix}_test_stim{stim}_prior{prior}.h5ad")

        _make_adata(data_beta[1:], times_data_test, 'beta')
        _make_adata(data_netw_theta[1:], times_data_test, 'theta')
        _make_adata(data_sim[1:], times_simulation_test, 'sim')
        _make_adata(data_rna_traj[1:], times_data_test, 'rna_traj')
        _make_adata(prot_traj_test[:, ns:].T, times_data_test, 'prot_traj')
        _make_adata(model.prot[:, ns:].T, times_simulation_test, 'prot_simul')

        print(f"[infer_test] All AnnData objects saved to {cardamom_dir}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[infer_test] Error building AnnData objects: {e}")
        sys.exit(1)

    print("[infer_test] Test set inference and simulation completed successfully")


if __name__ == "__main__":
    main(sys.argv[1:])
