"""
infer_test.py
-------------
Infer trajectories and simulate on test set using pre-learned model parameters.

Held-out validation with everything learned on the training cells kept fixed:
1. basins of the test cells from the training mixture parameters (per sample): one EMD per gene with the
   masses of the test cells (force_basins, mean_forcing_em), the network term of the training basin update
   replaced by a logistic regression of the final training basins (basins_final.npy) on the mixture probabilities;
2. protein trajectories in a single pass with the core network fixed (inter.npy / basal.npy): OT couplings
   with the final training regularization (n_iter_inference.npy), the alpha of each state copied from the
   nearest training state of the same time and sample (alpha.npy, data_prot.npy, data_kon_beta.npy);
3. trajectories and kon_theta recomputed with the simulation network (*_simul.npy), then
   simulation from the test cells at the first timepoint;
4. AnnData objects equivalent to the training ones (compared to Data/data_test.h5ad by
   check_test_to_train.py and in the final report).
Per-sample parameters (mixtures, basals) are routed to the cells of each sample, never averaged.
Stimuli: stimulus_test_schedule (per-sample rows), else the inference schedule.

Samples with remove_from_inference (perturbation_inference) are validated separately: simulation from the
first-timepoint training states of their reference_sample (default: the largest training sample), with that
sample's mixture and basal and the removed sample's test schedule (e.g. an untreated control measured at a
single time), saved as cardamomOT/adata_sim_validation_<sample>_stim*_prior*.h5ad and compared to its cells
in the report (section 6).

Usage:
    python infer_test.py -i <project_path> [--stimulus <float>] [--prior <float>] [--force-basins <float>]
                         [--temporal-basins <0|1>]

Required input files (from training pipeline):
    - Data/data_test.h5ad: test count matrix with temporal information
    - cardamomOT/mixture_parameters.npy, n_networks.npy: mixture model parameters
    - cardamomOT/pi_zinb.npy: zero-inflation parameters
    - cardamomOT/inter.npy, basal.npy: core inferred network (used for infer_test)
    - cardamomOT/basins_final.npy, proba_init.npy, alpha.npy, data_prot.npy, data_kon_beta.npy, data_times.npy,
      data_samples.npy, data_traj_valid.npy: training basins and trajectories
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
from CardamomOT.inputs import input_dir, removed_samples
from CardamomOT.inference.integration import nb_cell_parameters
from CardamomOT.inference.depth import state_depth, simulation_depth
from CardamomOT.run_options import parse_step_options, settings, configure
from CardamomOT.config import n_inference_stimuli
from CardamomOT.schedules import StimulusSchedule, sample_names, test_schedule


def main(argv):
    """
    Infer test-set trajectories and produce all comparison AnnData objects.

    Args:
        argv: Command-line arguments (--input).
    """
    opts = parse_step_options(argv, 'infer_test', __doc__)
    p = opts.p
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
    if 'time' not in adata.obs:
        print("[infer_test] Error: data_test.h5ad has no obs['time']")
        sys.exit(1)

    # Samples removed from the inference (perturbation_inference): validation pass; the others: test pass
    removed, refs = ([], {})
    if 'dataset_id' in adata.obs:
        removed, refs = removed_samples(p, present=adata.obs['dataset_id'].astype(str).unique())
    is_removed = (adata.obs['dataset_id'].astype(str).isin(removed).to_numpy() if removed
                  else np.zeros(adata.n_obs, dtype=bool))
    adata_removed = adata[is_removed].copy()
    adata = adata[~is_removed].copy()

    if adata.n_obs and len(np.unique(adata.obs['time'])) > 1:
        test_pass(p, opts, adata, cardamom_dir, removed)
    elif adata.n_obs:
        print("[infer_test] Warning: held-out cells of the training samples at a single timepoint: test pass skipped")
    if removed:
        validation_pass(p, opts, adata_removed, removed, refs, cardamom_dir)
    elif not adata.n_obs or len(np.unique(adata.obs['time'])) <= 1:
        print("[infer_test] Error: no held-out cells over several timepoints and no sample removed from the inference")
        sys.exit(1)
    print("[infer_test] Test set inference and simulation completed successfully")


def test_pass(p, opts, adata, cardamom_dir, removed=()):
    """Trajectories and simulation of the held-out cells of the training samples (network fixed)."""
    times_obs = adata.obs['time'].values
    print(f"[infer_test] Test pass: {adata.n_obs} cells, timepoints {sorted(np.unique(times_obs))}")

    # ─── STIMULUS SCHEDULES ──────────────────────────────────────────────
    # Held-out cells: stimulus_test_schedule (per sample), else the inference schedule; also for the simulation
    n_stimuli = n_inference_stimuli(input_dir(p))
    stim_sched, test_overrides = test_schedule(p, n_stimuli)
    test_overrides = {s: v for s, v in test_overrides.items() if s not in removed}  # validation pass
    stim_sched_simul = stim_sched

    # ─── INITIALIZE MODEL AND LOAD TRAINING PARAMETERS ──────────────────
    model = NetworkModel_beta(adata.shape[1], n_stimuli=n_stimuli)
    configure(model, opts)  # workbook, then the command-line options
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
    # Per-sample schedules of the held-out cells (sample indices follow the sorted test samples)
    model._stim_overrides = test_overrides
    model.set_sample_names(test_ids)

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
    def _inter_rows(arr, axis=0):
        # Per-sample interactions (network conditions: one more axis) take the test samples' rows too
        return _sample_rows(arr, axis) if np.ndim(arr) == 4 + axis else arr

    if inter_core is not None and basal_core is not None:
        model.basal = _sample_rows(basal_core)
        model.inter = _inter_rows(inter_core)
    else:
        model.basal = _sample_rows(basal_simul)
        model.inter = _inter_rows(inter_simul)
    basal_simul = _sample_rows(basal_simul)
    basal_t_simul = _sample_rows(basal_t_simul, axis=1)  # (T-1, n_samples, G_tot, n_networks)
    inter_simul = _inter_rows(inter_simul)
    inter_t_simul = _inter_rows(inter_t_simul, axis=1)   # (T-1, [n_samples,] G_tot, G_tot, n_networks)

    # Context of the last training iteration (regularization, mode-to-mode OT weight, basin probability weight)
    it_path = os.path.join(cardamom_dir, 'n_iter_inference.npy')
    context = None
    if os.path.exists(it_path):
        it_saved = np.atleast_1d(np.load(it_path)).astype(float)
        n_reg = int(it_saved[0])
        context = {'n_iter_reg': n_reg, 'weight_init': it_saved[1] if len(it_saved) > 1 else 0.0,
                   'weight_prob': it_saved[2] if len(it_saved) > 2 else max(.96**(n_reg - 1), .1)}
    else:
        print("[infer_test] Warning: n_iter_inference.npy not found (older run): "
              f"context of iteration {model.min_n_loops}")

    # Training arrays of the single-pass test inference (basin calibration, alphas of the nearest states)
    train = load_train_arrays(p, cardamom_dir, train_ids if 'dataset_id' in adata.obs else None)

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
    from CardamomOT.inputs import load_transition_rates
    transition_rates_test = load_transition_rates(p)  # default matrix, or {'default': ..., dataset_id: ...} with matrices per sample
    if transition_rates_test is not None:
        _m = transition_rates_test
        print(f"[infer_test] Loaded transition rates: " + (f"shape={_m.shape}" if hasattr(_m, 'shape') else
              "default " + ('yes' if _m.get('default') is not None else 'no') + ", own matrix for " +
              (', '.join(k for k in _m if k != 'default') or 'no sample')))

    # ─── TRAJECTORY INFERENCE ON TEST SET ────────────────────────────────
    # Classifies cells into modes (fixed kz/c) then infers OT couplings with
    # the core inferred network (compute_theta=False, update_modes=True).
    print(f"[infer_test] Running mixture classification and trajectory inference...")
    try:
        model.infer_test(adata, verb=1, stimulus_schedule=stim_sched,
                         basal_ref=basal_ref_test,
                         transition_rates=transition_rates_test,
                         context=context, train=train)
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
    times_file = os.path.join(input_dir(p), 'times_simulation.txt')
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
    # Same simulation as for the training cells: proliferation MLP (branching, birth rate of the dilution)
    if model.simulate_with_proliferation:
        from simulate_network import attach_proliferation
        attach_proliferation(model, p, tag='[infer_test]')
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
        depth_cells = (adata.obs['depth_factor'].values.astype(float)
                       if (model.use_depth_factor and 'depth_factor' in adata.obs) else None)
        d_tr = state_depth(depth_cells, real_idx_test)
        d_sim = simulation_depth(depth_cells, real_idx_test, times_data_test, times_simulation_test)
        data_beta = _nb_sample(vect_kon_beta, times_data_test, s_test[:len(vect_kon_beta)] if s_test is not None else None, d_tr)
        data_netw_theta = _nb_sample(vect_kon_theta, times_data_test, s_test[:len(vect_kon_theta)] if s_test is not None else None, d_tr)
        data_sim = _nb_sample(vect_kon_sim, times_simulation_test, s_sim, d_sim)
        # Same draws at the reference depth (s = 1), shown by the report with cell_depth_for_representation
        ref_depth = {}
        if depth_cells is not None:
            ref_depth = dict(beta=_nb_sample(vect_kon_beta, times_data_test, s_test[:len(vect_kon_beta)] if s_test is not None else None)[1:],
                             theta=_nb_sample(vect_kon_theta, times_data_test, s_test[:len(vect_kon_theta)] if s_test is not None else None)[1:],
                             sim=_nb_sample(vect_kon_sim, times_simulation_test, s_sim)[1:])

        # RNA trajectory data
        data_rna_traj = np.zeros((G + 1, rna_test.shape[0]))
        data_rna_traj[0, :] = times_data_test
        data_rna_traj[1:, :] = rna_test[:, ns:].T
        if d_tr is not None:  # counts of the real test cells (the inference works at the reference depth)
            data_rna_traj[1:, :] *= np.ravel(d_tr)[None, :]

        stim = model.stimulus
        prior = model.prior_network_pen

        def _make_adata(matrix_2d, obs_times, suffix):
            """matrix_2d: (G, N) — rows=genes, cols=cells."""
            a = ad.AnnData(X=matrix_2d.T)
            a.var = adata.var.copy()
            a.obs['time'] = obs_times
            if suffix in ref_depth:
                a.layers['reference_depth'] = ref_depth[suffix].T
            if suffix == 'rna_traj' and d_tr is not None:
                a.obs['depth_factor'] = np.ravel(d_tr)  # depth of the real test cell behind each state
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


def load_train_arrays(p, cardamom_dir, train_ids):
    """
    Training arrays used by the test inference (final basins, trajectories, alphas); exits if the run predates
    them or does not match data_train.h5ad.
    """
    names = {'prot': 'data_prot', 'kon_beta': 'data_kon_beta', 'alpha': 'alpha', 'times_data': 'data_times',
             'samples_data': 'data_samples', 'valid': 'data_traj_valid', 'proba_init': 'proba_init',
             'basins_final': 'basins_final'}
    paths = {k: os.path.join(cardamom_dir, v + '.npy') for k, v in names.items()}
    missing = [os.path.basename(v) for v in paths.values() if not os.path.exists(v)]
    train_path = os.path.join(p, 'Data', 'data_train.h5ad')
    if missing or not os.path.exists(train_path):
        print(f"[infer_test] Error: {', '.join(missing) or 'data_train.h5ad'} missing (rerun infer_network_structure)")
        sys.exit(1)
    train = {k: np.load(v) for k, v in paths.items()}
    obs = ad.read_h5ad(train_path, backed='r').obs
    train['cell_times'] = obs['time'].to_numpy(dtype=float)
    if len(train['cell_times']) != train['basins_final'].shape[0]:
        print("[infer_test] Error: training basins do not match data_train.h5ad (rerun infer_network_structure)")
        sys.exit(1)
    train['sample_ids'] = list(np.sort(obs['dataset_id'].unique())) if 'dataset_id' in obs and train_ids is not None else [0]
    return train


def _restrict_slots(model, keep):
    """Model trajectories restricted to the slots `keep` (bool (N,)), every timepoint."""
    T = len(np.unique(model.times_data))
    n = len(model.times_data)  # length before any restriction (times_data itself is restricted in the loop)
    for attr in ('prot', 'rna', 'kon_beta', 'kon_theta', 'traj_real_idx', 'R_opt', 'R_stim_offset', 'samples_data',
                 'times_data'):
        arr = getattr(model, attr, None)
        if arr is not None and np.ndim(arr) >= 1 and len(arr) == n:
            arr = np.asarray(arr)
            setattr(model, attr, arr.reshape((T, -1) + arr.shape[1:])[:, keep].reshape((-1,) + arr.shape[1:]))


def validation_pass(p, opts, adata_rem, removed, refs, cardamom_dir):
    """
    Samples removed from the inference: simulation from the first-timepoint training states of their reference
    sample (its mixture and basal), with their stimulus_test_schedule, compared to their observed cells.
    """
    import copy
    from simulate_network import load_simulation_model
    split = settings(opts).split
    train = ad.read_h5ad(os.path.join(p, 'Data', f'data_{split}.h5ad'))
    names = sample_names(train)
    base, _ = load_simulation_model(p, opts, train, tag='[infer_test]')
    ns = base.n_stimuli
    default, overrides = test_schedule(p, ns)
    pi_zinb = np.load(os.path.join(cardamom_dir, 'pi_zinb.npy'))
    T_tr = np.sort(np.unique(base.times_data))
    N = int(np.sum(base.times_data == T_tr[0]))
    sd = (np.asarray(base.samples_data).astype(int)[:N] if base.samples_data is not None
          else np.zeros(N, dtype=int))
    counts = train.obs['dataset_id'].astype(str).value_counts() if 'dataset_id' in train.obs else None
    tag = f'stim{base.stimulus}_prior{base.prior_network_pen}'

    for r in removed:
        cells = adata_rem[adata_rem.obs['dataset_id'].astype(str) == r]
        ref = refs.get(r)
        if ref is None:
            ref = str(counts.index[0]) if counts is not None else names[0]
            print(f"[infer_test] Warning: no reference_sample for removed sample {r}: '{ref}' (largest training sample)")
        if ref not in names:
            print(f"[infer_test] Warning: reference sample '{ref}' of {r} not in the training samples: {r} skipped")
            continue
        ref_idx = names.index(ref)
        keep = sd == ref_idx if len(names) > 1 else np.ones(N, dtype=bool)
        if not keep.any():
            print(f"[infer_test] Warning: no training trajectory of sample '{ref}': {r} skipped")
            continue
        m = copy.deepcopy(base)
        _restrict_slots(m, keep)

        # Simulated times: training times up to the last observed time of r, and the observed times of r
        t_obs = np.sort(np.unique(cells.obs['time'].astype(float)))
        sim_times = sorted({float(t) for t in T_tr if t <= t_obs[-1]} | {float(t) for t in t_obs} | {float(T_tr[0])})
        sched = StimulusSchedule(m._build_default_schedule(np.array(sim_times), default, times_ref=T_tr), overrides)
        m._stim_schedule = StimulusSchedule({t: sched.at(t, r) for t in sim_times})
        print(f"[infer_test] Validation of removed sample {r} ({cells.n_obs} cells at {t_obs.tolist()}) from the "
              f"first-timepoint states of '{ref}' ({int(keep.sum())} trajectories); schedule "
              + ', '.join(f'{t:g}: {np.round(v, 3).tolist()}' for t, v in m._stim_schedule.items()))
        try:
            m.simulate_network(list(sim_times))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[infer_test] Error during the validation simulation of {r}: {e}")
            continue

        # mRNA drawn from the simulated kon with the reference sample's mixture (depth of the observed cells)
        kon = m.kon_theta + 1e-6
        n_sim = kon.shape[0]
        k1c, cc, pzc = nb_cell_parameters(m.a, pi_zinb, np.full(n_sim, ref_idx))
        depth = None
        if m.use_depth_factor and 'depth_factor' in cells.obs:
            depth = np.random.choice(cells.obs['depth_factor'].to_numpy(dtype=float), n_sim)
        sd_ = 1.0 if depth is None else depth[:, None]
        n_param = np.maximum(((k1c + 1e-6) * kon)[:, ns:], 1e-6)
        p_param = np.clip((cc / (cc + sd_))[:, ns:], 1e-6, 1 - 1e-6)
        x = np.random.negative_binomial(n_param, p_param)
        x = np.where(np.random.uniform(0, 1, x.shape) < pzc, 0, x)  # pzc (1 or N, genes)
        a = ad.AnnData(X=x.astype(float))
        if depth is not None:  # same draw at the reference depth (s = 1)
            x_ref = np.random.negative_binomial(n_param, np.clip((cc / (cc + 1.0))[:, ns:], 1e-6, 1 - 1e-6))
            a.layers['reference_depth'] = np.where(np.random.uniform(0, 1, x_ref.shape) < pzc, 0, x_ref).astype(float)
        a.var = train.var.copy()
        a.obs['time'] = m.times_simul
        a.obs['dataset_id'] = r
        a.uns['reference_sample'] = ref
        a.uns['observed_times'] = t_obs
        if m.log_population is not None:
            a.uns['log_population'] = np.asarray(m.log_population)
            a.uns['simulated_times'] = np.array(sim_times)
        out = os.path.join(cardamom_dir, f'adata_sim_validation_{r}_{tag}.h5ad')
        a.write(out)
        print(f"[infer_test] Saved {os.path.basename(out)}")


if __name__ == "__main__":
    main(sys.argv[1:])
