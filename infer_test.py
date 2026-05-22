"""
infer_test.py
-------------
Infer trajectories and simulate on test set using pre-learned model parameters.

Classifies test cells into mixture modes with fixed kinetic parameters, then
infers protein trajectories with a fixed regulatory network, simulates expression
dynamics, and builds AnnData output objects equivalent to the training pipeline.

Usage:
    python infer_test.py -i <project_path>

Required input files (from training pipeline):
    - Data/data_test.h5ad: test count matrix with temporal information
    - cardamomOT/mixture_parameters.npy, n_networks.npy: mixture model parameters
    - cardamomOT/pi_zinb.npy: zero-inflation parameters
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
from CardamomOT import NetworkModel as NetworkModel_beta
import getopt


def main(argv):
    """
    Infer test-set trajectories and produce all comparison AnnData objects.

    Args:
        argv: Command-line arguments (--input).
    """
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:", ["input="])
    except getopt.GetoptError:
        print("[infer_test] Error: Invalid command-line arguments")
        print("[infer_test] Usage: python infer_test.py -i <project_path>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
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
    sched_path = os.path.join(p, 'Data', 'stimulus_schedule.txt')
    if os.path.exists(sched_path):
        stim_sched = np.loadtxt(sched_path)
        print(f"[infer_test] Loaded stimulus schedule from {sched_path}")

    stim_sched_simul = None
    sched_simul_path = os.path.join(p, 'Data', 'stimulus_schedule_simul.txt')
    if os.path.exists(sched_simul_path):
        stim_sched_simul = np.loadtxt(sched_simul_path)
        print(f"[infer_test] Loaded simulation stimulus schedule from {sched_simul_path}")
    else:
        stim_sched_simul = stim_sched

    # ─── DETECT n_stimuli FROM SCHEDULE ─────────────────────────────────
    _stim_arr = np.asarray(stim_sched) if stim_sched is not None else None
    n_stimuli = int(_stim_arr.shape[1]) if (_stim_arr is not None and _stim_arr.ndim == 2) else 1
    print(f"[infer_test] n_stimuli detected: {n_stimuli}")

    # ─── INITIALIZE MODEL AND LOAD TRAINING PARAMETERS ──────────────────
    model = NetworkModel_beta(adata.shape[1], n_stimuli=n_stimuli)
    print(f"[infer_test] Initialized model with {adata.shape[1]} genes")

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

    # Load the unitary-scale network (from infer_network_simul.py)
    try:
        basal_simul = np.load(os.path.join(cardamom_dir, 'basal_simul.npy'))
        inter_simul = np.load(os.path.join(cardamom_dir, 'inter_simul.npy'))
        basal_t_simul = np.load(os.path.join(cardamom_dir, 'basal_t_simul.npy'))
        inter_t_simul = np.load(os.path.join(cardamom_dir, 'inter_t_simul.npy'))
        ratios = np.load(os.path.join(cardamom_dir, 'ratios.npy'))
        d_t = np.load(os.path.join(cardamom_dir, 'degradations_temporal.npy'))
        print(f"[infer_test] Loaded unitary-scale simulation parameters")
    except FileNotFoundError as e:
        print(f"[infer_test] Error: Missing simulation parameter file: {e}")
        sys.exit(1)

    # Use mean basal over training samples for test trajectory inference
    # (avoids sample-count mismatch between training and test sets)
    if basal_simul.ndim == 3:
        model.basal = basal_simul.mean(axis=0)   # (G_tot, n_networks)
    else:
        model.basal = basal_simul
    model.inter = inter_simul
    model.basal_t = basal_t_simul
    model.inter_t = inter_t_simul
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
    kov_path = os.path.join(p, 'Data', 'KO_OV_inference.txt')
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
    tr_path = os.path.join(p, 'Data', 'transition_rates.csv')
    if os.path.exists(tr_path):
        transition_rates_test = pd.read_csv(tr_path, index_col=0)
        transition_rates_test.index = transition_rates_test.index.astype(str)
        transition_rates_test.columns = transition_rates_test.columns.astype(str)
        print(f"[infer_test] Loaded transition rates from {tr_path} shape={transition_rates_test.shape}")

    # ─── TRAJECTORY INFERENCE ON TEST SET ────────────────────────────────
    # Classifies cells into modes (fixed kz/c) then infers trajectories with
    # the pre-learned network (compute_theta=False, update_modes=True).
    print(f"[infer_test] Running mixture classification and trajectory inference...")
    try:
        model.infer_test(adata, verb=1, stimulus_schedule=None,
                         basal_ref=basal_ref_test,
                         transition_rates=transition_rates_test)  # schedule already built
        print(f"[infer_test] Test trajectory inference completed")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[infer_test] Error during inference: {e}")
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

    # ─── SIMULATION TIMES ────────────────────────────────────────────────
    times_file = os.path.join(p, 'Data', 'times_to_simulate.txt')
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

        c = model.a[-1, :]
        kz = model.a[:-1, :] + 1e-6

        vect_kon_beta = kon_beta_test + 1e-6        # (N_traj, G_tot)
        vect_kon_theta = kon_theta_test + 1e-6
        vect_kon_sim = model.kon_theta + 1e-6       # (N_sim, G_tot) after simulation

        # Helper: NB sample matrix of shape (G, N) using pi_zinb sparsity
        def _nb_sample(kon, times_vec):
            n_cells = kon.shape[0]
            n_param = (np.max(kz, axis=0) * kon)[:, ns:].T  # (G, N)
            p_param = (c / (c + 1))[ns:].reshape(G, 1)
            n_param = np.maximum(n_param, 1e-6)
            p_param = np.clip(p_param, 1e-6, 1 - 1e-6)
            zero_mask = np.random.uniform(0, 1, (G, n_cells)) < pi_zinb.reshape(G, 1)
            counts = np.random.negative_binomial(n_param, p_param)
            counts = np.where(zero_mask, 0, counts)
            out = np.zeros((G + 1, n_cells))
            out[0, :] = times_vec
            out[1:, :] = counts
            return out

        data_beta = _nb_sample(vect_kon_beta, times_data_test)
        data_netw_theta = _nb_sample(vect_kon_theta, times_data_test)
        data_sim = _nb_sample(vect_kon_sim, times_simulation_test)

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
