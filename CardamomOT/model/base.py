
"""
Core implementation of the NetworkModel used for inference and simulation.

This module defines the :class:`NetworkModel` class which encapsulates
parameters, state, and algorithms for fitting gene regulatory networks
from single-cell expression data, performing stochastic or deterministic
simulations, and managing mixture models. All documentation and comments
are maintained in English.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import ot
import seaborn as sns
import multiprocessing as mp
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from ..config import resolve_cell_type_obs, CELL_TYPE_OBS_KEYS
from ..inference.trajectory import ks_of, s1_of, s1_rows
from ..inference import (inference_network_multi, active_regulators, PrevProt, signed_floor, filter_network,
                        minimal_repetition_choice, find_next_prot, my_otdistance, count_errors,
                        kon_ref_vector, inference_alpha, inference_alpha_1thread,
                        NegativeBinomialMixtureEM, predict_resp,
                        simulate_next_prot_ode, simulate_next_prot_pdmp,
                        train_kon_correction_mlp, infer_ratio_d0_d1_full, infer_ratio_d0_d1_unitary, inference_degradation_prot,
                        train_proliferation_mlp, quadrature, fit_scale_theta,
                        seed_everything, seeded_call, task_seed,
                        stratified_order, stratified_choice, grouped_partition)

np.set_printoptions(precision=3, suppress=True)
EPS=1e-16


def _sample_values(entry, t, samples):
    """Value at time t of a perturbation stimulus for each sample index (its own schedule if given)."""
    samples = np.asarray(samples)
    out = np.full(len(samples), float(entry['schedule'](t)))
    for s, fn in (entry.get('sample_schedules') or {}).items():
        out[samples == s] = float(fn(t))
    return out


class NetworkModel:
    """
    Encapsulates the state and parameters of a regulatory network.

    The class stores kinetic, mixture and network parameters as well as
    trajectories produced during inference. It provides methods for
    initialization, calibration and simulation used by the higher-level
    pipeline script.
    """
    def __init__(self, n_genes=None, n_stimuli=1, times=None):
        # Infos
        self.loss_trajectory = []
        self.theta_trajectory = []
        # Kinetic parameters
        self.d = None
        self.d_t = None # temporal cinetic parameters
        # Mixture parameters
        self.weights = None
        self.n_networks = None
        self.adapt_size_network = None
        self.rna = None
        self.kon_beta = None
        self.modes = None
        self.alpha = None
        self.pi_init = None
        self.pi_zinb = None
        # Network parameters
        self.kon_theta = None
        self.a = None
        self.ref_network = None
        self.stimulus_targets = None  # (n_stimuli, n_genes) allowed stimulus -> gene edges (Data/stimulus_targets.txt)
        self.perturbation_stimulus = None  # KO/OV/Stim simulations: list of dict(signs=(G_tot,), schedule=t -> value)
        self.rate_perturbation = None      # RATE effects of the KO/OV/Stim simulations: list of dict(weights=(G,) or None, scale=(G,), delta, schedule)
        self.log_population = None         # log population size per simulated time (branching simulation), relative to t0
        self.basal = None
        self.inter = None
        self.inter_t = None
        self.basal_tmp = None
        self.inter_tmp = None
        self.ratios = None
        self.times_data = None
        self.times_simul = None
        self.samples_data = None
        self.prot = None
        self.proba_init = None
        self.proba = None
        self.proba_traj = None

        self.n_stimuli = n_stimuli
        self._stim_schedule = None
        self._stim_overrides = {}   # per-sample stimulus schedules {dataset_id: (times or None, values)}
        self._sample_names = None   # sorted dataset_id of the run (index of a sample -> its label)

        ### Pipeline (run.sh / cardamomot pipeline): data and steps, fixed per project in Data/CardamomOT_inputs.xlsx
        self.split = 'train'                     # 'train': train/test split of the cells (train_rate per sample and time); 'full': all cells
        self.train_rate = 0.7                    # share of the cells of each (sample, time) in the train split (at least 100); the test keeps at most as many
        self.select_genes = False                # select_genes_and_split: gene selection (queries, entropy genes, global network, Steiner tree); False = all genes kept
        self.build_prior_network = False         # literature prior cardamomOT/ref_network.csv: built by the gene selection if select_genes, literature_selection and prior_network_pen = 0, else by build_reference_network
        self.estimate_proliferation_rates = True  # get_proliferation_rates: obs['proliferation_net_rate'] from gene signatures, anchored to Data/proliferation_rates
        self.run_test = True                    # infer_test + check_test_to_train on the held-out cells (needs split = 'train')
        self.simulate_perturbations = True       # simulate_network_KOV + check_KOV_to_sim (perturbation_simulation sheet)
        self.species = 'auto'                    # 'auto' (from gene names), 'human' or 'mouse': degradation rates, proliferation signatures, literature prior
        self.senescence_gating = True            # get_proliferation_rates: the senescence signature gates the proliferation score
        self.overwrite_degradation_rates = False  # get_degradation_rates: replace the d0/d1 already stored in the AnnData files
        self.report_net_index = 0                # report: network shown when n_networks > 1
        self.report_normalize = False            # report UMAPs: counts normalised per cell
        self.report_log1p = True                 # report UMAPs: log1p of the counts
        self.report_n_umap = 4000                # report: maximal number of cells per stage in the UMAPs (0 = all)

        ### Default behaviour
        self.seed = None # Random seed for reproducibility (main process and parallel workers); None = not seeded (runs vary)

        ## Compute mixture parameters
        self.hard_em = 1 # Do we initialize with a hard_em ?
        self.preserve_mean_values = 1 # Do we ensure temporal constraints when fitting the basins in the hard_em ?
        self.mean_forcing_em = 0.5 # at which point we force the mean correction: the higher the more
        self.force_basins = 1.0 # Do we want to ensure the means to be preserved by the NB mixture ? It may not preserve multistability
        self.temporal_basins = 1 # Is it preserved temporally ?
        self.transform_proba = 0 # Do we want to force probas to be steep for compatibility with sigmoid model?
        self.seuil = 1e-2 # minimum for beta mixture parameters (second parameters)
        self.batch_size_mixture = 1024 # Maximum number of cells per time used for mixture calibration in the inference.
        self.use_depth_factor = False   # use obs['depth_factor'] (estimate_cell_depth) in the NB model and simulations; False = s_i = 1 (an existing factor is kept but ignored)
        self.compute_depth_factor = False # estimate_cell_depth computes the diagnostic and writes obs['depth_factor'] if needed; False = only reads an existing factor (never removed)
        self.allow_depth_correction = False  # estimate_cell_depth: apply the per-cell depth factor s_i when the diagnostic recommends it
        self.depth_method = 'poissonian'  # depth factor: 'group_median', 'poissonian' (Fang & Pachter, 2025), or <project>/depth_methods/<name>.py
        self.depth_method_params = {}       # parameters of the depth method
        self.depth_by_cell_type = False     # estimate s_i within (sample, time, cell type) groups; False = (sample, time) only:
                                            # cell types derive from expression, normalising within them pulls cells to their type (circular)
        self.soft_em_refinement = False # NB mixture: refine (ks, c) by soft EM with fixed basin masses
        self.use_scBoolSeq = False # If True, initialize NB mixture with scBoolSeq binarization instead of hard EM
        self.scboolseq_confidence = 0.6 # GMM posterior probability threshold for cell label assignment (lower → fewer NaN)
        self.scboolseq_min_cells_per_label = 10 # min cells per label (0 and 1) required to use scBoolSeq path; otherwise falls back to normal EM
        self.scboolseq_dropout = False

        ## Infer network
        self.n_networks = 1
        self.adapt_size_network = 0
        # Loop for the inference
        self.min_n_loops = 10 # minimal number of iterations in inference loops
        self.count_max = 5 # Stopping criteria
        self.max_iter = 40 # max iteration for main loop
        # Trajectory inference with OT
        self.stopThr_init = 1e-7 # initial tolerance for sinkhorn algorithm
        self.batch_size_traj = 1024 # Maximum number of cells used per time point per sample for solving EOT problems in the inference.
        self.batch_size_traj_exp = 1024 # Target size of the real (experimental) cell batches matched against each reconstruction batch; decoupled from batch_size_traj so the OT target pool can stay well-sized even when batch_size_traj is kept small for cost.
        self.n_strata_traj = 10 # Real-cell batches are stratified by cell type if available, else by this number of k-means clusters (0 = uniform batches)
        self.unbalanced_reg = 5 # Unbalanced regularization parameter for UOT if > 0, OT if 0
        self.init_entropic_noise = 1.5 # Initial entropic penalization for OT
        self.quant_samples = .95 # Quantile of cells number per sample to use for inference
        # General parameters to calibrate protein reconstruction
        self.scale_proteins = 1 # Eventually rescale protein values (recommended:1-2)
        self.scale_mrnas = 100 # Eventually rescale mRNA values (recommended:100)
        self.fact_simple = 2 # slight transformation for constrative modes in learning phase
        # Network inference with scipy
        self.loss_norm = 'CE'
        self.scale_pen = 20 # Error that is expected = 1/scale_pen
        self.compute_with_proba = 0 # Determine if compute with proba or kon values in network inference (recommended:1)
        self.weight_prev = .4 # max = .5 to not withdrawn the inference on timepoints, allows the calibration to incorporate some "flow-matching" method
        self.batch_size_network = 100 # Cells per network sub-sample (stratified by time and sample); raised to 10 x the parameters of a target-gene fit if lower; None = that floor
        self.n_network_fits = 10 # Theta = mean of min(n_network_fits, 1 + n_states // batch_size) fits on disjoint subsamples, at every network update
        # Inference of alpha = switch moment between each timepoint and modes
        self.update_modes = 1
        self.alpha_threshold = .6 # max = 1, thershold for important transition to update alpha full
        self.n_pas = 25 # number of timesteps between timepoints for inference of alpha
        self.force_n_pas = False # if True, use n_pas as-is instead of auto-flooring it to the inter-timepoint interval length
        # Penalization/prior information
        self.stimulus = 1.0 # 1 if we simulate with a stimulus. If not we can penalize the stimulus with a value between 1 and 0: 0 = no sitmulus
        self.prior_network_pen = 1.0 # 1 if we don't use prior information. If not we can penalize the non-existing age in prior network with values between 1 and 0: 0 = impossible edge
        self.constrain_basal_uniform = 1.0 # >= 0 penalty strength that pushes per-sample basals to be equal (ignores samples pinned by KO/OV basal_ref)
        self.hard_forcing_ref = False # if True, constrain all network params to ±ref_constraint_pct around inter_ref
        self.ref_constraint_pct = 0.01 # fractional tolerance around inter_ref values for bounds (used when hard_forcing_ref=True)
        self.seuil_zero_min_ref = 5e-2 # reference values (inter_ref) with |v| <= this are read as absent edges; also min |theta| of sign-forced edges
        self.lambda_mlp    = .5  # Mix weight for training-data ratios vs MLP in simulate_full_with_harissa:
                                  # 1 = pure linear interpolation of observed g, 0 = pure MLP g(P, kon(P))
        # Filtering
        self.filter_network = 1 # Do we filter the network ? It also builds a temporal network using the filter criterium
        self.seuil_min_network_intensity = 1e-2 # post-inference filter: edges with |theta| below are removed (max with seuil_zero_min_ref if hard_forcing_ref)
        self.seuil_min_network_variations = 0 # post-inference filter: min kon variation when removing an edge (0 = intensity threshold only)

        ## Compute degradations after inference
        self.recompute_degradations = 1 # Do we want to recompute degradation rates for simulations ?
        self.traj_cell_types = None # Cell type of the real cell behind each trajectory state (set by fit_network)
        self.batch_size_degradations = 256 # Trajectories drawn per interval at each optimizer step of the degradation inference (slow without gpu)
        self.use_temporal_degradations = 1 # If so, compute temporal degradation rates for simulations ?
        self.lambda_scale  = 1e-3  # L2 penalty on scale[ns:] around 1 (large = scale stays ~1; 0 = free)
        self.lambda_deg0   = 1  # L2 penalty on d0 around d_init (0 = free; large = stays close to prior)
        self.lambda_deg1   = 1e-3  # L2 penalty on d1 around 0 (0 = free; large = stays close to non-temporal)
        self.smooth_degradations_sigma = None  # None=auto KDE+CV, 0=off, float>0=fixed sigma (in time-step units)
        self.smooth_degradations_strength = 0.5  # blend weight in [0,1]: 0=no smoothing, 1=full smoothing

        ## Simulations
        self.simulation_stochastic = True # 1 if we simulate Bursty-like proteins, 0 if deterministic limit for proteins
        self.finish_by_determinist = False # 1 if we simulate with deterministic limit for the last timepoint
        self.min_ratio = .05
        self.max_ratio = 50
        self.simulate_full_with_harissa = False  # use Harissa PDMP to jointly simulate proteins+mRNAs
        self.kon_beta_harissa = None  # continuous adaptive_shrinkage burst-rate estimates (set by loop_trajectories)
        self.kon_mlp = None           # KonCorrectionMLP trained in refine_network_degradations (Harissa branch)

        ## Gene selection (select_genes_and_split.py, change=1): terminals + global network + directed Steiner tree
        self.num_max_genes = 100         # budget: number of selected genes (stimuli excluded)
        self.n_query_genes = 40          # at most this many genes of genes_queries (gene_lists sheet) (round robin over time/cell-type DE groups)
        self.n_entropy_genes = 30        # at least this many entropy genes (Gandrillon KD & MDE), same round robin; n_query + n_entropy < num_max_genes
        self.n_top_entropy = 400         # top genes per transition for KD and for MDE (Gandrillon's TOP_N)
        self.n_cells_entropy = 1000      # cells per timepoint for the BUB entropy (its matrices are (N+1)^2)
        self.n_hvg_selection = 5000      # highly variable genes forming the network universe (terminals always kept)
        self.network_method = 'otvelo_granger'  # global network: 'otvelo_granger', 'otvelo_corr', or <project>/network_methods/<name>.py
        self.network_method_params = {}  # parameters of the network method (otvelo: n_cells, n_pcs, eps, alpha; granger: + k_candidates, en_alpha, l1_ratio)
        self.k_in_steiner = 20           # strongest incoming edges kept per gene in the Steiner graph
        self.k_stim_steiner = 20         # direct targets kept per stimulus in the Steiner graph
        self.min_edge_prob = 0.05        # edges with probability (1 - FDR against the permuted-data network) below are dropped
        self.edge_prior = 0.9            # each Steiner edge costs -log(prob * edge_prior): favours short paths among equally probable ones
        self.null_network = 'hybrid'     # edge probabilities vs permuted data: 'hybrid' (gene edges within (sample, time), stimulus all cells), 'within_time', 'all_cells'
        self.closure_min = 0.5           # closure: add the gene bringing the most probable regulation (sum of w) to the selection while >= this
        self.literature_selection = True  # gene selection: edge probabilities reweighted by OmniPath feasibility, whatever the prior
        self.literature_depth = 3        # max path length in the literature graph (last edge TF -> target)
        self.literature_weight = 1.0     # exponent on the data-calibrated literature likelihood ratio (0 = data only)
        self.min_entropy_change = 0.1    # variability floor of the selection: max BUB-entropy change between consecutive times (bits, Gandrillon MDE)
        self.min_nb_separation = 0.15    # ... and separation of the extreme NB modes at the mixture initialization (times / cell types, batch_size_mixture cells per time)
        self.max_free_params = 10000     # change=1 with ref=1 and prior 0 (hard prior): gene budget set so that the literature prior leaves this many free network parameters
        self.literature_resources = 'extended'  # 'extended' (OmniPath, CollecTRI + PathwayExtra, KinaseExtra, LigRecExtra, DoRothEA A-D, TFtarget) or 'core' (OmniPath + CollecTRI)

        ## Multi-sample integration of the mixture (obs['dataset_id']; no effect with a single sample)
        self.integrate_samples = 1.0             # in [0, 1]: per-sample mixtures pushed towards the common target (1 = equal parameters, 0 = own fits)
        self.ref_sample_integration = None       # dataset_id of the reference sample; None = cell-weighted average of mode means (keeps total counts)
        self.min_cells_integration = 200         # smaller samples are not fitted alone (raw counts, classified with global modes)
        self.min_mode_weight_integration = 0.01  # a sample mode is supported if its weight is at least this...
        self.min_mode_ratio_integration = 1.0    # ...and the ON/OFF mean ratio at least this; else the gene is not integrated for that sample
        self.integration = None                  # per-sample parameters and integrated genes, set by fit_mixture_samples

        ## Proliferation
        self.recompute_proliferations = False    # train a ProliferationMLP on R_opt in refine_network_degradations
        self.simulate_with_proliferation = False # apply branching process in simulate_trajectories_unitary
        self.prolif_uses_stimulus = True         # inference stimuli are inputs of the ProliferationMLP, R(u, P) (no effect if constant over the intervals)
        self.prolif_network = None               # ProliferationMLP trained in refine_network_degradations
        self.R_opt = None                        # net growth rate of each trajectory state over the next interval (NaN at last time), from the final growth OT pass
        self.R_stim_offset = None                # part of R_opt due to the inference stimuli (RATEk of perturbation_inference), removed before training the proliferation MLP
        self.stimulus_rate_model = None          # StimulusRateModel: stimulus part of the net rate from mRNA, added back in the branching simulations
        self.growth_reg_source = 2.0             # source-marginal relaxation (x log G) of the growth OT pass: small = data-driven but noisy, large = prior kept
        self.n_growth_iter = 1                   # WOT-style growth iterations (source weights <- row marginals); more iterations amplify the noise
        self.n_growth_nodes = 5                  # quadrature nodes per interval to integrate R along paths (MLP training and branching simulation)
        self.population_sizes = None             # optional {time: total cell number}: absolute population growth per interval (else the prior one)
        self.inter_simul_ref = None              # optional inter reference for refine_network_degradations (forces final=0)

        if n_genes is not None:
            G = n_genes + n_stimuli
            # Default degradation rates
            self.d = np.zeros((2,G))
            self.d[0] = np.log(2)/9 # mRNA degradation rates
            self.d[1] = np.log(2)/46 # protein degradation rates
            # Default network parameters
            self.a = np.zeros((3,G))
            self.basal = np.zeros((1, G, 1))
            self.inter = np.zeros((G, G, 1))
            self.inter_t = np.zeros((1, G, G, 1))
            self.ref_network = np.ones((G, G, 1))
        

    def _parse_input(self, data, time_key='time', scale_depth=False):
        """(N, 1 + G_tot) array [time, stimuli padding, counts]; scale_depth: counts divided by the
        per-cell depth factors (obs['depth_factor']), i.e. brought to the reference depth."""
        try:
            import anndata
            import scipy.sparse
            if isinstance(data, anndata.AnnData):
                # Missing time column = stationary data (single timepoint 0)
                vect_t = (data.obs[time_key].values.astype(float) if time_key in data.obs
                          else np.zeros(data.n_obs))
                X = data.X.toarray() if scipy.sparse.issparse(data.X) else np.asarray(data.X, dtype=float)
                depth = self._depth_factors(data) if scale_depth else None
                if depth is not None:
                    X = X / depth[:, None]
                if self.n_stimuli > 1:
                    X = np.hstack([np.zeros((X.shape[0], self.n_stimuli - 1), dtype=float), X])
                return np.column_stack([vect_t, X])
        except (ImportError, AttributeError):
            pass
        return data

    @staticmethod
    def _cell_type_labels(data):
        """Per-cell cell-type labels (str) used to stratify batches, or None if unavailable."""
        obs = getattr(data, 'obs', None)
        if obs is None:
            return None
        key = resolve_cell_type_obs(data, 'transition')
        return None if key is None else obs[key].values.astype(str)

    def set_sample_names(self, names):
        """Sorted dataset_id of the run: labels of the sample indices (per-sample schedules); overrides of
        samples absent from the data are dropped with a warning."""
        from ..schedules import keep_present
        self._sample_names = [str(s) for s in names]
        self._stim_overrides = keep_present(self._stim_overrides, self._sample_names, 'stimulus schedule')
        if self._stim_schedule is not None:
            self._stim_schedule.sample_names = self._sample_names
            self._stim_schedule.overrides = self._stim_overrides

    def apply_project_parameters(self, project, verb=True):
        """
        Override attributes with the values filled in the model_parameters sheet of
        Data/CardamomOT_inputs.xlsx; the command-line options are applied after it and
        dominate (run_options.configure). Values are cast to the type of the current value.
        Returns the set of overridden attributes.
        """
        import ast
        from ..inputs import project_parameters
        values = project_parameters(project)
        done = {}
        for name, v in values.items():
            if not hasattr(self, name):
                print(f"[CardamomOT] Warning: model_parameters: unknown parameter '{name}' ignored")
                continue
            cur = getattr(self, name)
            try:
                if isinstance(cur, bool):
                    val = v if isinstance(v, bool) else str(v).strip().lower() in ('true', '1', 'yes')
                elif isinstance(cur, int) and not isinstance(cur, bool):
                    val = int(float(v))
                elif isinstance(cur, float):
                    val = float(v)
                elif isinstance(cur, str):
                    val = str(v)
                else:  # None, dict, list...: Python literal, else string
                    try:
                        val = ast.literal_eval(v) if isinstance(v, str) else v
                    except (ValueError, SyntaxError):
                        val = v
            except (TypeError, ValueError):
                print(f"[CardamomOT] Warning: model_parameters: invalid value {v!r} for '{name}', ignored")
                continue
            setattr(self, name, val)
            done[name] = val
        self._project_overrides = set(done)
        if done and verb and not getattr(NetworkModel, '_printed_overrides', False):
            print(f"[CardamomOT] Parameters of Data/CardamomOT_inputs.xlsx (override the defaults): {done}")
            NetworkModel._printed_overrides = True
        return set(done)

    def overridden(self, name):
        """True if the attribute was set by the model_parameters sheet of the project."""
        return name in getattr(self, '_project_overrides', set())

    def _add_perturbation_stimulus(self, basal_t, inter_t, times):
        """
        Perturbation stimuli of the KO/OV/Stim simulations (self.perturbation_stimulus = list of
        dict(signs=(G_tot,) in {-1, 0, 1}, schedule=t -> value)): on each simulated interval, the
        value of each stimulus at the end of the interval times w_g is added to the input of each
        of its targets g, with w_g = sign_g (100 + sum of the |interactions| received by g), i.e.
        stimuli whose interactions dominate the others (as a KO/OV), each with its own schedule;
        their effects add up.
        """
        perts = getattr(self, 'perturbation_stimulus', None) or []
        if isinstance(perts, dict):
            perts = [perts]
        for pert in perts:
            signs = np.asarray(pert['signs'], dtype=float)
            targets = np.flatnonzero(signs)
            if pert.get('sample_schedules') and basal_t.ndim != 4:
                print("[simulate] Warning: per-sample perturbation schedules need per-sample basal: default schedule used")
            for cnt in range(len(times) - 1):
                if not len(targets):
                    continue
                w = signs[targets][:, None] * (100 + np.abs(inter_t[cnt][:, targets, :]).sum(axis=0))  # (n_targets, n_networks)
                if basal_t.ndim == 4:
                    # Each sample with its own schedule of the stimulus
                    u = _sample_values(pert, times[cnt + 1], np.arange(basal_t.shape[1]))
                    basal_t[cnt][:, targets, :] += u[:, None, None] * w[None]
                else:
                    basal_t[cnt][targets, :] += float(pert['schedule'](times[cnt + 1])) * w

    def _apply_stimulus_targets(self):
        """Forbid the stimulus -> gene edges outside self.stimulus_targets ((n_stimuli, n_genes) mask)."""
        mask = getattr(self, 'stimulus_targets', None)
        if mask is None:
            return
        ns = self.n_stimuli
        self.ref_network[:ns, ns:] = self.ref_network[:ns, ns:] * np.asarray(mask, dtype=float)[:, :, None]

    def _depth_factors(self, data):
        """Per-cell depth factors s_i (obs['depth_factor'], estimate_cell_depth.py), or None (also if not use_depth_factor)."""
        obs = getattr(data, 'obs', None)
        if not self.use_depth_factor or obs is None or 'depth_factor' not in obs:
            return None
        return np.asarray(obs['depth_factor'].values, dtype=float)

    def _t0_cell_types(self, vect_t, vect_samples_id, sample):
        """Cell types of the first-timepoint cells of a sample (None if unavailable)."""
        ct = getattr(self, '_strata_labels', None)
        if ct is None:
            return None
        return ct[(vect_t == np.min(vect_t)) & (vect_samples_id == sample)]

    def _load_ot_constraints(self, data, transition_rates=None):
        """
        Load the optional OT constraints from AnnData: per-cell net proliferation
        rates, lineage barcodes, and cell-type transition rates with their grouping
        (`resolve_cell_type_obs(data, 'transition')`, read only if
        `transition_rates` is given). Transitions are all-or-nothing: if the
        grouping is missing or any cell type is absent from the matrix, the OT
        runs without transition constraint.
        """
        self._prolif_net_rate = None
        self._cell_types = None
        self._transition_rates = None
        self._transition_type_labels = None
        self._lineage = None
        self._lineage_known = None
        # Cell types (if any) stratify every batch built from the data
        self._strata_labels = self._cell_type_labels(data)
        obs = getattr(data, 'obs', None)
        if obs is not None:
            if 'proliferation_net_rate' in obs:
                self._prolif_net_rate = obs['proliferation_net_rate'].values.astype(float)
            if 'lineage' in obs:
                self._lineage_known = obs['lineage'].notna().values
                self._lineage = obs['lineage'].astype(str).values

        if transition_rates is None:
            return
        ct_col = resolve_cell_type_obs(data, 'transition') if obs is not None else None
        if ct_col is None:
            print("Warning: transition_rates given but adata.obs has none of "
                  f"{list(CELL_TYPE_OBS_KEYS['transition'])}; OT run without transition constraint")
            return
        cell_types = obs[ct_col].values.astype(str)
        types = set(cell_types)
        _Tr = np.asarray(transition_rates, dtype=float)
        if hasattr(transition_rates, 'index') and hasattr(transition_rates, 'columns'):
            tr_df = transition_rates.copy()
            tr_df.index, tr_df.columns = tr_df.index.astype(str), tr_df.columns.astype(str)
            labels = [l for l in tr_df.index if l in set(tr_df.columns)]
            # Partial anchoring is worse than none: every type must be in rows and columns
            missing = sorted(types - set(labels))
            if missing:
                print(f"Warning: cell type(s) {missing} of adata.obs['{ct_col}'] not found in "
                      "transition_rates rows/columns; OT run without transition constraint")
                return
            # Align columns on rows so that index i means the same type on both axes
            _Tr = tr_df.loc[labels, labels].to_numpy().astype(float)
        else:
            labels = None
            if _Tr.shape != (len(types), len(types)):
                print(f"Warning: transition_rates shape {_Tr.shape} does not match the "
                      f"{len(types)} types of adata.obs['{ct_col}']; OT run without transition constraint")
                return
        self._cell_types = cell_types
        self._transition_type_labels = labels
        self._transition_rates = np.clip(_Tr, 0.0, None)  # store raw non-negative rates
        print(f"Transition rates anchored on adata.obs['{ct_col}']")

    def _build_stimulus_schedule(self, times_unique, stimulus_schedule=None, times_ref=None):
        """Default schedule {t: values} with the per-sample overrides (StimulusSchedule)."""
        from ..schedules import StimulusSchedule
        return StimulusSchedule(self._build_default_schedule(times_unique, stimulus_schedule, times_ref),
                                self._stim_overrides, self._sample_names)

    def _build_default_schedule(self, times_unique, stimulus_schedule=None, times_ref=None):
        if stimulus_schedule is None:
            t_min = times_unique[0]
            return {t: (np.zeros(self.n_stimuli) if t == t_min else np.ones(self.n_stimuli))
                    for t in times_unique}
        stim = np.asarray(stimulus_schedule, dtype=float)
        if stim.ndim == 1:
            stim = stim[:, None]

        if times_ref is not None and len(times_ref) > 0:
            # Step-function interpolation: each row of stim corresponds to times_ref[i].
            # For each simulation time, use the value from the most recent reference time
            # that is <= simulation time (hold-last semantics).
            times_ref_sorted = np.sort(np.asarray(times_ref, dtype=float))
            n_ref = len(times_ref_sorted)
            if stim.shape[0] < n_ref:
                stim = np.vstack([stim, np.tile(stim[-1], (n_ref - stim.shape[0], 1))])
            stim_mapped = np.empty((len(times_unique), stim.shape[1]))
            for i, t_sim in enumerate(times_unique):
                idx = int(np.searchsorted(times_ref_sorted, t_sim + 1e-9, side='right')) - 1
                idx = max(0, min(idx, n_ref - 1))
                stim_mapped[i] = stim[idx]
            stim = stim_mapped
        else:
            # Direct mapping: rows correspond 1-to-1 to times_unique (inference mode)
            n_rows, n_tp = stim.shape[0], len(times_unique)
            if n_rows < n_tp:
                stim = np.vstack([stim, np.tile(stim[-1], (n_tp - n_rows, 1))])
            elif n_rows > n_tp:
                raise ValueError(
                    f"stimulus_schedule has {n_rows} rows but only {n_tp} unique timepoints"
                )

        if stim.shape[1] == 1 and self.n_stimuli > 1:
            stim = np.tile(stim, (1, self.n_stimuli))
        if stim.shape[1] != self.n_stimuli:
            raise ValueError(
                f"stimulus_schedule has {stim.shape[1]} column(s) but model was created with "
                f"n_stimuli={self.n_stimuli}. Pass n_stimuli={stim.shape[1]} to the model "
                f"constructor (e.g. NetworkModel(n_genes, n_stimuli={stim.shape[1]}))."
            )
        return {t: stim[i] for i, t in enumerate(times_unique)}


    def _compute_scboolseq_matrix(self, data_rna, gene_names, G_tot):
        """
        Log-transform raw counts and run scBoolSeq to obtain:
          - a binarization matrix  (N_cells, n_genes) with values 0.0 / 1.0 / NaN
          - a dropout-rate vector  (n_genes,) with per-gene structural-zero probability

        The dropout rate is read from ``scbs.criteria_['DropOutRate']`` — the
        fraction of log-expression values that are essentially zero, which maps
        directly to the ``pi_zero`` parameter of the ZINB mixture model.
        """
        try:
            import pandas as pd
            from scboolseq import scBoolSeq
        except ImportError:
            raise ImportError(
                "scBoolSeq is required when use_scBoolSeq=True. "
                "Install it with:  conda install -c conda-forge -c colomoto scboolseq"
            )
        ns = self.n_stimuli
        gene_expr = data_rna[:, ns:].astype(float)
        log_expr  = np.log1p(gene_expr)
        n_genes   = log_expr.shape[1]
        col_names = [str(gn) for gn in (gene_names[:n_genes] if len(gene_names) >= n_genes
                                        else range(n_genes))]
        log_df = pd.DataFrame(log_expr, columns=col_names)

        scbs = scBoolSeq(confidence=self.scboolseq_confidence)
        scbs.fit(log_df)
        binarized = scbs.binarize(log_df)

        # Extract per-gene dropout rate from scBoolSeq criteria
        dropout_rates = None
        if self.scboolseq_dropout and hasattr(scbs, 'criteria_') and 'DropOutRate' in scbs.criteria_.columns:
            dropout_rates = np.zeros(n_genes, dtype=float)
            for j, col in enumerate(col_names):
                if col in scbs.criteria_.index:
                    dropout_rates[j] = float(
                        np.clip(scbs.criteria_.loc[col, 'DropOutRate'], 0.0, 0.95)
                    )

        return binarized.to_numpy().astype(float), dropout_rates


    def core_binarization(self, data_rna, gene_names, vect_t, G_tot, min_components=1, max_components=5, refilter=0, max_iter_kinetics=100,
                          verb=True, kov_cell_mask=None, scboolseq_matrix=None, scboolseq_dropouts=None, strata=None,
                          depth=None):
        """
        Parameters
        ----------
        strata : (N_cells,) array or None
            Cell types stratifying the per-timepoint sub-sample of the mixture fit.
        depth : (N_cells,) array or None
            Per-cell depth factors: counts modelled as NB(k, c / s_i), parameters at the reference depth.
        kov_cell_mask : (N_cells, G_tot) int8 array or None
            Per-cell KO/OV constraints derived from KO_OV_inference.

            - -1: gene is KO for this cell (force to lowest mode)
            - +1: gene is OV for this cell (force to highest mode)
            - 0: no constraint
        """

        # Get kinetic parameters
        N_cells = np.size(data_rna, 0)
        ns = self.n_stimuli
        frequency_modes_smooth = np.zeros((N_cells, G_tot), dtype=float)
        for t in np.unique(vect_t):
            frequency_modes_smooth[vect_t == t, :ns] = self._stim_schedule[t]

        ks = []
        proba_init = []
        proba_modif = []
        pi_init = []
        c = np.ones(G_tot)
        pi_zeros = np.ones(G_tot - ns)
        n_components = 0
        kinetics = NegativeBinomialMixtureEM(min_components=min_components,
                                                 max_components=max_components, zi=None,
                                                 max_iter_em=max_iter_kinetics,
                                                 refilter=refilter, hard_em=self.hard_em,
                                                 preserve_mean_values=self.preserve_mean_values, mean_forcing_em=self.mean_forcing_em,
                                                 use_scBoolSeq=(scboolseq_matrix is not None),
                                                 soft_em_refinement=self.soft_em_refinement)

        def run_main_loop_for_gene(g):
            if verb: print("Calibrating gene", g)
            x = data_rna[:, g]
            scbs_labels   = scboolseq_matrix[:, g - ns]  if scboolseq_matrix  is not None else None
            scbs_dropout  = scboolseq_dropouts[g - ns]   if scboolseq_dropouts is not None else None
            if scbs_labels is not None:
                _min_n = self.scboolseq_min_cells_per_label
                _valid = ~np.isnan(scbs_labels.astype(float))
                if (np.sum(_valid & (scbs_labels == 0)) < _min_n or
                        np.sum(_valid & (scbs_labels == 1)) < _min_n):
                    scbs_labels = None
                    scbs_dropout = None
            model = kinetics.fit(x, vect_t=vect_t, vect_celltypes=strata, seuil=self.seuil, s=depth,
                                 batch_size_mixture=self.batch_size_mixture, strata=strata,
                                 scboolseq_labels=scbs_labels,
                                 scboolseq_dropout=scbs_dropout)
            ks, c, pi0, proba, pi = np.sort(model['ks']), model['c'], np.mean(np.asarray(model['pi_zero'])), model['resp'], model['pi']
            ## Transform proba to be steepers
            tmp = proba.copy()
            if self.transform_proba:
                tmp = np.exp(self.transform_proba * ((len(ks)-1))*np.log(G_tot)*(proba - 1/len(ks))) # self.transform_proba is the typical size of parameters that are expected, np.log(G) the number of regulators), and the difference to the mean max proba scales the protein level
                tmp /= (1 + tmp)
                tmp /= np.sum(tmp, 1).reshape(N_cells, 1)
                for cell in range(N_cells):
                    if np.max(proba[cell]) > np.max(tmp[cell]):
                        tmp[cell, :] = proba[cell, :]
                proba[:, :] = tmp[:, :]
            if self.update_modes or self.loss_norm == 'CE':
                # Initial basins: argmax of the mixture posteriors (masses are imposed later, in loop_trajectories)
                tmp = np.zeros_like(proba)
                tmp[np.arange(proba.shape[0]), np.argmax(proba, axis=1)] = 1
        
            return ks, c, pi0, proba, tmp, pi

        results = Parallel(n_jobs=-1)(
        delayed(seeded_call)(task_seed(self.seed, 1, g), run_main_loop_for_gene, g) for g in range(ns, G_tot)
        )

        for idx, g in enumerate(range(ns, G_tot)):

            kg, cg, pi_zerog, probag, tmpg, pi_initg = results[idx]
            cg_old, cg = cg, np.minimum(9, cg) # No need of having a variance too low
            kg *= cg / cg_old
            frequency_modes_smooth[:, g] = np.sum(kg * tmpg, axis=1)
            if verb and g - ns < len(gene_names): print('Gene {}-{} calibrated...'.format(g, gene_names[g - ns]), kg, cg)
            if len(kg) > n_components:
                n_components = len(kg)
            ks.append(kg)
            proba_init.append(probag)
            proba_modif.append(tmpg)
            c[g] = cg
            pi_zeros[g - ns] = pi_zerog
            pi_init.append(pi_initg)

        self.a = np.zeros((n_components+1, G_tot)) + self.seuil / 10
        frequency_proba_init = np.zeros((N_cells, G_tot, n_components))
        for s in range(ns):
            for t in np.sort(np.unique(vect_t)):
                mask = vect_t == t
                if self._stim_schedule[t][s] < 0.5:
                    frequency_proba_init[mask, s, 0] = 1
                else:
                    frequency_proba_init[mask, s, -1] = 1
        frequency_proba_modif = frequency_proba_init.copy()
        for s in range(ns):
            self.a[:, s] = 1
        for g in range(ns, G_tot):
            g_idx = g - ns
            self.a[:len(ks[g_idx]), g] = ks[g_idx][:]
            frequency_proba_init[:, g, :len(ks[g_idx])] = proba_init[g_idx]
            frequency_proba_init[:, g, len(ks[g_idx]):] = 0
            frequency_proba_modif[:, g, :len(ks[g_idx])] = proba_modif[g_idx]
            frequency_proba_modif[:, g, len(ks[g_idx]):] = 0
        self.a[-1, :] = c[:]
        self.pi_init = pi_init

        scale_max = np.max(self.a[:-1, :], axis=0)
        frequency_modes_smooth /= np.maximum(scale_max, EPS)  # all-zero modes would give 0/0

        # ── Force KO/OV cells to the correct mode after binarization ─────────
        if kov_cell_mask is not None:
            for g in range(ns, G_tot):
                g_idx = g - ns
                n_modes = len(ks[g_idx])
                ko_cells = kov_cell_mask[:, g] < 0
                ov_cells = kov_cell_mask[:, g] > 0
                if np.any(ko_cells):
                    frequency_proba_init[ko_cells, g, :] = 0
                    frequency_proba_init[ko_cells, g, 0] = 1
                    frequency_proba_modif[ko_cells, g, :] = 0
                    frequency_proba_modif[ko_cells, g, 0] = 1
                    frequency_modes_smooth[ko_cells, g] = self.a[0, g] / (scale_max[g] + EPS)
                if np.any(ov_cells):
                    frequency_proba_init[ov_cells, g, :] = 0
                    frequency_proba_init[ov_cells, g, n_modes - 1] = 1
                    frequency_proba_modif[ov_cells, g, :] = 0
                    frequency_proba_modif[ov_cells, g, n_modes - 1] = 1
                    frequency_modes_smooth[ov_cells, g] = self.a[n_modes - 1, g] / (scale_max[g] + EPS)

        if verb: print('Mean proba = ', np.mean(np.max(frequency_proba_init[:, ns:, :], axis=-1)),
              np.mean(np.max(frequency_proba_modif[:, ns:, :], axis=-1)))

        return frequency_modes_smooth, frequency_proba_init, frequency_proba_modif, pi_zeros



    def fit_mixture(self, data, refilter=0, gene_names=np.arange(1, 50000), min_components=2, max_components=2, max_iter_kinetics=0, verb=True, stimulus_schedule=None, time_key='time', kov_cell_mask=None):
        """
        Fit the mixture model parameters to the data.
        """
        seed_everything(self.seed)
        data_rna = self._parse_input(data, time_key)
        N_cells, G_tot = data_rna.shape
        vect_t = data_rna[:, 0]
        self._stim_schedule = self._build_stimulus_schedule(np.sort(np.unique(vect_t)), stimulus_schedule)

        # ── scBoolSeq pre-computation (once for all genes) ──────────────────
        scboolseq_matrix  = None
        scboolseq_dropouts = None
        if self.use_scBoolSeq:
            scboolseq_matrix, scboolseq_dropouts = self._compute_scboolseq_matrix(
                data_rna, gene_names, G_tot
            )

        frequency_modes_smooth, frequency_proba_init, frequency_proba_modif, pi_zeros = self.core_binarization(
                                        data_rna, gene_names, vect_t, G_tot,
                                        min_components=min_components,
                                        max_components=max_components,
                                        refilter=refilter,
                                        max_iter_kinetics=max_iter_kinetics,
                                        verb=verb,
                                        kov_cell_mask=kov_cell_mask,
                                        scboolseq_matrix=scboolseq_matrix,
                                        scboolseq_dropouts=scboolseq_dropouts,
                                        strata=self._cell_type_labels(data),
                                        depth=self._depth_factors(data))

        self.pi_zinb = pi_zeros
        self.modes = frequency_modes_smooth
        self.proba_init = frequency_proba_init
        self.proba = frequency_proba_modif
        self._finalize_components(G_tot)

    def _finalize_components(self, G_tot):
        """Number of networks from the number of mixture components (or components merged by quantiles)."""
        n_components = self.a.shape[-2] - 1  # a: (M+1, G) shared or (S, M+1, G) per sample
        if self.adapt_size_network:
            self.n_networks = n_components - 1
        else:
            if n_components > self.n_networks+1:
                qs = np.linspace(0, 1, self.n_networks+1)

                def merge(a):
                    a_new = np.zeros((self.n_networks+2, G_tot))
                    for g in range(G_tot):
                        l_max = len(np.unique(self.modes[:, g]))
                        a_new[:-1, g] = np.quantile(a[:l_max, g], qs)
                    a_new[-1, :] = a[-1, :]
                    return a_new
                self.a = np.stack([merge(a) for a in self.a]) if self.a.ndim == 3 else merge(self.a)

    def _shared_mixture(self, arr):
        """Per-sample mixture term collapsed to its shared value (Harissa paths): only when all samples are equal."""
        if np.ndim(self.a) < 3 or np.ndim(arr) < np.ndim(self.a) - 1:
            return arr
        if not np.allclose(self.a, self.a[:1]):
            raise NotImplementedError("Sample-specific mixtures (integrate_samples < 1) are not supported "
                                      "with simulate_full_with_harissa")
        return arr[0]

    def _mixture_terms(self):
        """
        Normalised mixture amplitudes ks (G, M), mRNA-to-protein factors s1 (G - ns,) and max burst
        rates k1 (G,) from self.a; with per-sample mixtures (a 3-D), each gets a leading sample axis.
        """
        ns = self.n_stimuli
        kz, c = self.a[..., :-1, :], self.a[..., -1, :]
        k1 = np.max(kz, axis=-2)
        ks = np.swapaxes(kz / np.clip(k1[..., None, :], EPS, None), -1, -2)
        s1 = self.fact_simple * c[..., ns:] / np.maximum(k1[..., ns:], EPS)
        return ks, s1, k1

    def _steepen_proba(self, proba, G_tot):
        """Same post-processing of mixture responsibilities as core_binarization: (proba_init, proba)."""
        proba = proba.copy()
        n = proba.shape[1]
        if self.transform_proba:
            tmp = np.exp(self.transform_proba * (n - 1) * np.log(G_tot) * (proba - 1 / n))
            tmp /= (1 + tmp)
            tmp /= np.sum(tmp, 1, keepdims=True)
            keep = np.max(proba, axis=1) > np.max(tmp, axis=1)
            tmp[keep] = proba[keep]
            proba = tmp.copy()
        tmp = proba.copy()
        if self.update_modes or self.loss_norm == 'CE':
            tmp = np.zeros_like(proba)
            tmp[np.arange(len(proba)), np.argmax(proba, axis=1)] = 1
        return proba, tmp

    def fit_mixture_samples(self, data, sample_key='dataset_id', kov_genes=None, stimulus_schedule=None,
                            time_key='time', verb=True, **kwargs):
        """
        Mixture fit with several samples (>= 2 in obs[sample_key]); with a single sample,
        identical to fit_mixture.

        Each sample (>= min_cells_integration cells) gets its own mixture fit, i.e. identifies its
        own modes. Per gene, the common target comes from the reference sample
        (ref_sample_integration, a dataset_id), or is the cell-weighted average of the sample mode
        means recalibrated to keep the mean and, when possible, the variance of the gene over all
        cells; it uses the samples whose modes are supported (see integration.supported_modes) and
        where the gene is not perturbed (kov_genes: {dataset_id: set of genes}).

        integrate_samples = lam in [0, 1] (True = 1, False = 0): the parameters of these (sample,
        gene) pairs are interpolated from the sample's own fit (lam = 0) to the target (lam = 1)
        (integration.interpolate_parameters, recalibrated when the target is the average), and
        their counts quantile-matched accordingly (unchanged at lam = 0); the other pairs keep raw
        counts and the target parameters, and are classified with them (genes never supported
        anywhere take the pooled fit). self.a is always (S, M+1, G) and self.pi_zinb (S, G - ns),
        samples ordered as np.unique(obs[sample_key]) (= sample index of fit_network); at lam = 1
        all samples share the same parameters.

        Returns the integrated count matrix (N, n_genes), or None if counts were not changed (lam = 0).
        """
        from ..inference.integration import (supported_modes, consistent_modes, global_parameters,
                                             interpolate_parameters, integrate_counts)
        self.integration = None
        samples = data.obs[sample_key].values if sample_key in data.obs else None
        sample_ids = list(np.unique(samples)) if samples is not None else []
        if len(sample_ids) < 2:
            if len(sample_ids) == 1 and str(sample_ids[0]) in self._stim_overrides:
                # Single sample with its own schedule (stimulus_inference_schedule, sample_id rows)
                tt = np.sort(np.unique(np.asarray(data.obs[time_key], dtype=float)))
                sched = self._build_stimulus_schedule(tt, stimulus_schedule)
                stimulus_schedule = np.array([sched.at(t, sample_ids[0]) for t in tt])
            self.fit_mixture(data, stimulus_schedule=stimulus_schedule, time_key=time_key, verb=verb, **kwargs)
            return None
        lam = float(np.clip(float(self.integrate_samples), 0.0, 1.0))

        ns = self.n_stimuli
        data_rna = self._parse_input(data, time_key)
        N, G_tot = data_rna.shape
        vect_t = data_rna[:, 0]
        times_all = np.sort(np.unique(vect_t))
        sched_full = self._build_stimulus_schedule(times_all, stimulus_schedule)
        gene_names = list(data.var_names)

        # --- 1. Independent fit per sample ---
        fits = []
        for s in sample_ids:
            m = samples == s
            if m.sum() < self.min_cells_integration:
                if verb:
                    print(f"[integration] sample {s}: {m.sum()} cells < {self.min_cells_integration}, not fitted alone")
                continue
            if verb:
                print(f"[integration] Fitting the mixture of sample {s} ({m.sum()} cells)")
            sub = data[m]
            t_sub = np.sort(np.unique(vect_t[m]))
            self.fit_mixture(sub, stimulus_schedule=np.array([sched_full.at(t, s) for t in t_sub]),
                             time_key=time_key, verb=verb, **kwargs)
            fits.append(dict(id=s, mask=m, a=self.a.copy(), proba_init=self.proba_init.copy(),
                             proba=self.proba.copy(), pi_init=self.pi_init,
                             pi0=np.concatenate([np.zeros(ns), self.pi_zinb])))
        if not fits:
            self.fit_mixture(data, stimulus_schedule=stimulus_schedule, time_key=time_key, verb=verb, **kwargs)
            return None

        # Same number of modes for all samples (pad mode rows before the dispersion row)
        M = max(f['a'].shape[0] - 1 for f in fits)
        for f in fits:
            k = f['a'].shape[0] - 1
            if k < M:
                f['a'] = np.vstack([f['a'][:-1], np.full((M - k, G_tot), self.seuil / 10), f['a'][-1:]])
                f['proba_init'] = np.concatenate([f['proba_init'], np.zeros((f['mask'].sum(), G_tot, M - k))], axis=2)
                f['proba'] = np.concatenate([f['proba'], np.zeros((f['mask'].sum(), G_tot, M - k))], axis=2)
        S = len(fits)
        a_samples = np.stack([f['a'] for f in fits])
        pi0_samples = np.stack([f['pi0'] for f in fits])

        # --- 2. Eligible (sample, gene) pairs and global parameters ---
        eligible = np.zeros((S, G_tot), dtype=bool)
        for i, f in enumerate(fits):
            eligible[i] = supported_modes(f['a'], f['proba_init'], ns, self.min_mode_weight_integration,
                                          self.min_mode_ratio_integration)
            for gene in (kov_genes or {}).get(str(f['id']), ()):
                if gene in gene_names:
                    eligible[i, ns + gene_names.index(gene)] = False
        eligible[:, :ns] = False
        eligible = consistent_modes(a_samples, [f['proba_init'] for f in fits], eligible, ns)
        ref = None
        if self.ref_sample_integration is not None:
            ids = [str(f['id']) for f in fits]
            if str(self.ref_sample_integration) in ids:
                ref = ids.index(str(self.ref_sample_integration))
            elif verb:
                print(f"[integration] Warning: reference sample {self.ref_sample_integration} not among the fitted "
                      f"samples {ids}; using the average of the samples")
        a, pi0, has = global_parameters(a_samples, pi0_samples, [f['proba_init'] for f in fits], eligible, ref)

        # Genes supported in no sample: parameters of the pooled fit
        no_global = np.flatnonzero(~has[ns:]) + ns
        if len(no_global):
            if verb:
                print(f"[integration] {len(no_global)} genes supported in no sample: pooled mixture parameters")
            self.fit_mixture(data, stimulus_schedule=stimulus_schedule, time_key=time_key, verb=verb, **kwargs)
            k = min(M, self.a.shape[0] - 1)
            a[:, no_global] = self.seuil / 10
            a[:k, no_global] = self.a[:k, no_global]
            a[-1, no_global] = self.a[-1, no_global]
            pi0[no_global] = np.concatenate([np.zeros(ns), self.pi_zinb])[no_global]
        a[:, :ns] = 1
        # Target recalibrated (average target only), then per-sample parameters at level lam
        proba_fits = [f['proba_init'] for f in fits]
        if ref is None:
            a1, p1 = interpolate_parameters(a_samples, pi0_samples, proba_fits, eligible, a, pi0, 1.0, True, ns)
            for g in np.flatnonzero(eligible.any(axis=0)):
                e0 = np.flatnonzero(eligible[:, g])[0]
                a[:, g], pi0[g] = a1[e0, :, g], p1[e0, g]
        a_lam, pi_lam = interpolate_parameters(a_samples, pi0_samples, proba_fits, eligible, a, pi0, lam, ref is None, ns)
        n_modes = np.full(G_tot, M)
        for g in range(ns, G_tot):
            n_modes[g] = max(2, int(np.sum(a[:-1, g] > self.seuil / 10 + EPS))) if np.all(np.isfinite(a[:, g])) else M

        # --- 3. Integrated counts and global responsibilities ---
        rng = np.random.default_rng(self.seed)
        depth = self._depth_factors(data)
        X = data_rna.copy()
        proba_init = np.zeros((N, G_tot, M))
        proba = np.zeros((N, G_tot, M))
        fitted = np.zeros(N, dtype=bool)
        for i, f in enumerate(fits):
            m = f['mask']
            fitted |= m
            for g in range(ns, G_tot):
                if eligible[i, g]:
                    z = np.argmax(f['proba_init'][:, g, :], axis=1)
                    if lam > 0:
                        X[m, g] = integrate_counts(data_rna[m, g], z, (a_samples[i, :-1, g], a_samples[i, -1, g], pi0_samples[i, g]),
                                                   (a_lam[i, :-1, g], a_lam[i, -1, g], pi_lam[i, g]), rng,
                                                   s=None if depth is None else depth[m])
                    proba_init[m, g] = f['proba_init'][:, g]
                    proba[m, g] = f['proba'][:, g]
        # Raw (sample, gene) pairs: classify the raw counts with the global modes
        for g in range(ns, G_tot):
            ng = n_modes[g]
            raw = ~fitted.copy()
            for i, f in enumerate(fits):
                if not eligible[i, g]:
                    raw |= f['mask']
            if raw.any():
                resp, _ = predict_resp(data_rna[raw, g], a[:ng, g], a[-1, g],
                                       pi_zero=pi0[g] if pi0[g] > 0 else None, zi=True if pi0[g] > 0 else None,
                                       s=None if depth is None else depth[raw])
                proba_init[raw, g, :ng], proba[raw, g, :ng] = self._steepen_proba(resp, G_tot)
        for s in range(ns):
            for t in times_all:
                mt = vect_t == t
                proba_init[mt, s, :] = 0
                proba_init[mt, s, 0 if sched_full[t][s] < 0.5 else -1] = 1
        proba[:, :ns] = proba_init[:, :ns]

        # --- 4. Mixture outputs, as fit_mixture would give on the integrated data ---
        self._stim_schedule = sched_full
        # Per-sample parameters at level lam; target ones for unsupported/perturbed pairs and unfitted samples
        a_per = np.repeat(a[None], len(sample_ids), axis=0)
        pi0_per = np.repeat(pi0[None], len(sample_ids), axis=0)
        for i, f in enumerate(fits):
            k = sample_ids.index(f['id'])
            a_per[k][:, eligible[i]] = a_lam[i][:, eligible[i]]
            pi0_per[k][eligible[i]] = pi_lam[i][eligible[i]]
        self.a = a_per
        self.pi_zinb = pi0_per[:, ns:]
        a_cells = a_per
        pos = np.searchsorted(np.asarray(sample_ids), samples)
        # Modes in units of the max burst rate of each cell's sample
        kz_cells = a_cells[pos, :-1]                                   # (N, M, G)
        modes = np.einsum('ngm,nmg->ng', proba, kz_cells) / np.maximum(kz_cells.max(axis=1), EPS)
        for t in times_all:
            modes[vect_t == t, :ns] = sched_full[t]
        self.modes = modes
        self.proba_init = proba_init
        self.proba = proba
        self.pi_init = []
        for g in range(ns, G_tot):
            ng = n_modes[g]
            pi_g = {}
            for t in times_all:
                p = proba_init[vect_t == t, g, :ng].mean(axis=0)
                pi_g[t] = p / (p.sum() + EPS)
            self.pi_init.append(pi_g)
        self._finalize_components(G_tot)

        # Per-sample parameters, kept to integrate other files of the same samples
        self.integration = dict(
            sample_ids=[f['id'] for f in fits], a_samples=a_samples, pi0_samples=pi0_samples,
            pi_init_samples=[f['pi_init'] for f in fits], eligible=eligible, a=a, pi0=pi0,
            n_modes=n_modes, ref=None if ref is None else fits[ref]['id'], sample_key=sample_key,
            lam=lam, a_dst=a_lam, pi0_dst=pi_lam, all_sample_ids=sample_ids)
        if verb:
            n_int = eligible[:, ns:].sum(axis=1)
            print(f"[integration] integrate_samples = {lam:g}; genes with sample-specific parameters per sample: "
                  + ", ".join(f"{f['id']}: {n}/{G_tot - ns}" for f, n in zip(fits, n_int)))
        return X[:, ns:] if lam > 0 else None

    def integrate_data(self, data, time_key='time'):
        """
        Integrated counts (N, n_genes) of another AnnData of the same samples (e.g. data_test):
        each cell is classified with the mixture of its own sample, then quantile-matched as in
        fit_mixture_samples. Cells of unfitted samples and non-integrated genes keep raw counts.
        """
        from ..inference.integration import integrate_counts
        info = self.integration
        ns = self.n_stimuli
        data_rna = self._parse_input(data, time_key)
        X = data_rna.copy()
        vect_t = data_rna[:, 0]
        key = info['sample_key']
        if key not in data.obs:
            return X[:, ns:]
        samples = data.obs[key].values.astype(str)
        depth = self._depth_factors(data)
        rng = np.random.default_rng(None if self.seed is None else self.seed + 1)
        for i, sid in enumerate(info['sample_ids']):
            m = samples == str(sid)
            if not m.any():
                continue
            a_s, pi0_s = info['a_samples'][i], info['pi0_samples'][i]
            if info['lam'] == 0:
                continue
            a_d, pi_d = info['a_dst'][i], info['pi0_dst'][i]
            for g in np.flatnonzero(info['eligible'][i]):
                ng = info['n_modes'][g]
                pi_t = info['pi_init_samples'][i][g - ns]
                z = np.zeros(m.sum(), dtype=int)
                for t in np.unique(vect_t[m]):
                    mt = vect_t[m] == t
                    prior = np.asarray(pi_t[t], dtype=float)[:ng] if t in pi_t else None
                    resp, _ = predict_resp(data_rna[m, g][mt], a_s[:ng, g], a_s[-1, g], pi=prior,
                                           pi_zero=pi0_s[g] if pi0_s[g] > 0 else None,
                                           zi=True if pi0_s[g] > 0 else None,
                                           s=None if depth is None else depth[m][mt])
                    z[mt] = np.argmax(resp, axis=1)
                X[m, g] = integrate_counts(data_rna[m, g], z, (a_s[:-1, g], a_s[-1, g], pi0_s[g]),
                                           (a_d[:-1, g], a_d[-1, g], pi_d[g]), rng,
                                           s=None if depth is None else depth[m])
        return X[:, ns:]

    def integration_report(self, gene_names):
        """DataFrame (sample, gene): mode means of the sample and global ones, and whether the pair was integrated."""
        import pandas as pd
        info = self.integration
        ns = self.n_stimuli
        rows = []
        for i, sid in enumerate(info['sample_ids']):
            a_s = info['a_samples'][i]
            for g in range(ns, a_s.shape[1]):
                ng = info['n_modes'][g]
                row = {'sample': sid, 'gene': gene_names[g - ns], 'integrated': bool(info['eligible'][i, g]),
                       'integrate_samples': info['lam']}
                a_d = info['a_dst'][i] if info['eligible'][i, g] else info['a']
                for z in range(ng):
                    row[f'mean_mode{z}_sample'] = a_s[z, g] / a_s[-1, g]
                    row[f'mean_mode{z}_global'] = info['a'][z, g] / info['a'][-1, g]
                    row[f'mean_mode{z}_used'] = a_d[z, g] / a_d[-1, g]
                rows.append(row)
        return pd.DataFrame(rows)


    def adaptive_shrinkage(self, x, mu, fact=2, p=2):
        d = x - mu
        alpha = (np.abs(d) / (fact * (EPS + mu)))**p
        weight = alpha / (1 + alpha)
        res = x * (1 - weight) + mu * weight
        return res * self.scale_proteins
    
    
    def adaptive_shrinkage_init(self, x, mu, p=.5):
        G = mu.shape[1]
        res = mu * self.scale_proteins
        xs = self.adaptive_shrinkage(x, mu)
        a = np.min(mu, axis=0) * self.scale_proteins
        b = np.max(mu, axis=0) * self.scale_proteins
        for g in range(G):
            ks = np.sort(np.unique(mu[:, g]))
            x_min = np.min(xs[:, g])
            x_max = np.max(xs[:, g]) 
            xs[:, g] -= x_min
            lmax = len(ks)
            for cnt_z, z in enumerate(ks):
                indices = (mu[:, g] == z)
                x_ming = min(np.min(xs[indices, g]), x_max / (1 + lmax - cnt_z))
                x_maxg = max(np.max(xs[indices, g]), x_max / (lmax - cnt_z))
                res[indices, g] = a[g] + (b[g]-a[g]) * (cnt_z + 
                                                (np.clip(xs[indices, g], x_ming, x_maxg) / (x_maxg + EPS))**p) / lmax
        return res 
    

    def estimate_trajectories_given_model(self, vect_t, times, vect_samples_id,
                                      samples_id, vect_rna, y_prot_old, prot_formodes, y_kon_old, y_rna_old, y_proba_old,
                                      alpha_old, vect_samples_id_modified,
                                      basal, inter, s1, ks, init_cells, R_opt_traj, to_keep_for_update, offset_init=[0],
                                      n_iter=1, N_full=[100], N_samples=[100], intensity_prior=10,
                                      real_cell_batches=None, batch_idx=None, sim_real_idx=None,
                                      growth_only=False):
        """
        Infer the protein trajectories when d1 is known and theta is not.

        real_cell_batches[s_idx][t_idx] : list of index arrays partitioning the real
            cells available at times[t_idx + 1] for sample s_idx into batches, so that
            each batch of the reconstruction ensemble only targets its corresponding
            batch of real cells instead of the full experimental set.
        batch_idx : current batch index per sample (into real_cell_batches[s_idx]).
        sim_real_idx : (T * N_total,) int array or None, updated in place with the real
            cell behind each trajectory state (-1 = not yet assigned).
        growth_only : if True, trajectories are left untouched and only R_opt_traj is
            filled, from the growth OT pass (see _growth_log_mass) on the same costs.
        """

        G = vect_rna.shape[1]
        T = len(times)
        N_total = np.sum(N_full)
        ns = self.n_stimuli

        rna_modified = y_rna_old
        prot_modified = y_prot_old
        kon_modified = y_kon_old
        proba_modified = y_proba_old
        alpha_modified = alpha_old
        prot_old_is_nonzero = y_prot_old.any()

        # Track which real cell (index into vect_rna) each simulated slot corresponds to.
        if sim_real_idx is None:
            sim_real_idx = np.full(rna_modified.shape[0], -1, dtype=int)

        # Growth pass: the final trajectories (t = 0 included) are kept as they are
        if not growth_only:
            # Set stimulus values per timepoint (schedule-based, per sample if it has its own schedule)
            sched = self._stim_schedule
            per_sample = getattr(sched, 'has_overrides', lambda: False)()
            slot_s = np.repeat(np.arange(len(samples_id)), N_full)[:N_total] if per_sample else None
            for t_idx, t_i in enumerate(times):
                sl = slice(t_idx * N_total, (t_idx + 1) * N_total)
                m_t = vect_t == t_i
                if per_sample:
                    val = sched.per_cell(t_i, slot_s)
                    prot_formodes[m_t, :ns] = sched.per_cell(t_i, vect_samples_id[m_t]) * self.scale_proteins
                else:
                    val = np.asarray(sched[t_i], dtype=float)
                    prot_formodes[m_t, :ns] = val * self.scale_proteins
                rna_modified[sl, :ns] = val * self.scale_mrnas
                prot_modified[sl, :ns] = val * self.scale_proteins
                kon_modified[sl, :ns] = (val >= 0.5)

            # Fill initial state (t = 0) and initialize sim_real_idx for those cells
            offset = 0
            for s, sample in enumerate(samples_id):
                cell_indices = (vect_t == times[0]) & (vect_samples_id == sample)
                global_cell_idx = np.flatnonzero(cell_indices)
                selected_init = init_cells[s]

                kon_modified[offset+offset_init[s]:offset+offset_init[s] + N_samples[s], ns:] = self.modes[cell_indices][selected_init, ns:]
                if self.compute_with_proba:
                    proba_modified[offset+offset_init[s]:offset+offset_init[s] + N_samples[s]] = self.proba[cell_indices][selected_init]
                rna_modified[offset+offset_init[s]:offset+offset_init[s] + N_samples[s], ns:] = vect_rna[cell_indices][selected_init, ns:]
                sim_real_idx[offset+offset_init[s]:offset+offset_init[s] + N_samples[s]] = global_cell_idx[selected_init]
                # Sample label of every trajectory slot, last timepoint included
                for t_idx in range(T):
                    vect_samples_id_modified[N_total * t_idx + offset:N_total * t_idx + offset + N_full[s]] = s

                offset += N_full[s]

            prot_modified[:N_total, ns:] = self.adaptive_shrinkage_init(
                rna_modified[:N_total, ns:] * s1_rows(s1, vect_samples_id_modified[:N_total]), kon_modified[:N_total, ns:])
            if n_iter == 1:
                N_cells_0 = np.sum(vect_t == times[0])
                prot_formodes[:N_cells_0, ns:] = self.adaptive_shrinkage_init(
                    vect_rna[:N_cells_0, ns:] * s1_rows(s1, vect_samples_id)[:N_cells_0], self.modes[:N_cells_0, ns:])

        for t_idx, time in enumerate(times[:-1]):
            offset = 0
            for s_idx, sample in enumerate(samples_id):
                cell_idx = real_cell_batches[s_idx][t_idx][batch_idx[s_idx]]
                offset_init_s = offset + offset_init[s_idx]
                start_index = N_total * t_idx + offset_init_s
                next_index = N_total * (t_idx + 1) + offset_init_s
                N_sample = N_samples[s_idx]
                N_cells = len(cell_idx)

                if N_sample and N_cells:

                    current_indices = np.arange(start_index, start_index + N_sample)
                    next_indices = np.arange(next_index, next_index + N_sample)
                    alpha_indices = np.arange(offset_init_s, offset_init_s + N_sample)

                    # Old trajectories of this batch block at t+1 (not yet overwritten):
                    # candidate pool for the alpha re-assignment below
                    if prot_old_is_nonzero and time != times[-2]:
                        prot_old_blk = y_prot_old[next_indices, ns:]
                        kon_old_blk = y_kon_old[next_indices, ns:]
                        alpha_old_blk = alpha_old[t_idx + 1, alpha_indices]
                    else:
                        prot_old_blk = None

                    prot_init = prot_modified[current_indices, ns:]
                    alpha_init = alpha_modified[t_idx, alpha_indices]
                    s1_s, ks_s = s1_of(s1, s_idx), ks_of(ks, s_idx)
                    mode_init = self.adaptive_shrinkage(rna_modified[current_indices, ns:] * s1_s, kon_modified[current_indices, ns:]) / s1_s
                    mode_end = self.adaptive_shrinkage(vect_rna[cell_idx, ns:] * s1_s, self.modes[cell_idx, ns:]) / s1_s

                    basal_s = basal[min(s_idx, basal.shape[0] - 1)] if basal.ndim == 3 else basal
                    pairwise_dist = my_otdistance(
                        kon_modified[current_indices, ns:], self.modes[cell_idx, ns:],
                        prot_init,
                        rna_modified[current_indices, ns:], vect_rna[cell_idx, ns:],
                        proba_modified[current_indices, ns:], self.proba[cell_idx, ns:, :],
                        mode_init, mode_end,
                        alpha_init,
                        s1_s, ks_s, self.d[1, ns:], times[t_idx + 1] - time, basal_s, inter, loss=self.loss_norm,
                        n_iter=n_iter, intensity_prior=intensity_prior,
                        compute_with_proba=self.compute_with_proba,
                        n_stimuli=ns, stim_vals=np.asarray(self._stim_schedule.at(times[t_idx + 1], s_idx), dtype=np.float64),
                        scale_proteins=self.scale_proteins
                    )

                    delta_t = times[t_idx + 1] - time
                    src_real = sim_real_idx[current_indices]

                    # --- Transition rate cost adjustment ---
                    _tr = getattr(self, '_transition_rates', None)
                    _ct = getattr(self, '_cell_types', None)
                    if _tr is not None and _ct is not None:
                        _labels = getattr(self, '_transition_type_labels', None)
                        if _labels is not None:
                            _lbl_to_i = {l: i for i, l in enumerate(_labels)}
                        else:
                            _lbl_to_i = {c: i for i, c in enumerate(np.unique(_ct))}
                        # Every cell type is in the matrix (checked in _load_ot_constraints)
                        src_ti = np.array([_lbl_to_i[str(_ct[r])] for r in src_real])
                        tgt_ti = np.array([_lbl_to_i[str(_ct[j])] for j in cell_idx])

                        tr_prob = np.exp(_tr * delta_t)
                        tr_prob = tr_prob / tr_prob.sum(axis=1, keepdims=True) * _tr.shape[1]
                        tr_w = tr_prob[np.ix_(src_ti, tgt_ti)]
                        pairwise_dist = pairwise_dist / np.maximum(tr_w, 1e-10)

                    # --- Lineage constraint ---
                    # Cells sharing a clonal lineage barcode at t must map to a cell of
                    # the same lineage at t+1. Cells with no known lineage (NaN/empty)
                    # are left unconstrained. Cross-lineage pairs get a very large but
                    # finite penalty rather than np.inf, so Sinkhorn stays numerically
                    # stable even if a batch happens to contain no same-lineage target
                    # (real_cell_batches sub-samples the target cells per batch) —
                    # the constraint degrades gracefully instead of breaking convergence.
                    _lin = getattr(self, '_lineage', None)
                    _lin_known = getattr(self, '_lineage_known', None)
                    if _lin is not None:
                        lin_src = _lin[src_real]
                        lin_tgt = _lin[cell_idx]
                        known_src = _lin_known[src_real]
                        known_tgt = _lin_known[cell_idx]
                        mismatch = (lin_src[:, None] != lin_tgt[None, :]) & known_src[:, None] & known_tgt[None, :]
                        if mismatch.any():
                            penalty = pairwise_dist.max() * 1e3 + 1e6
                            pairwise_dist = np.where(mismatch, pairwise_dist + penalty, pairwise_dist)

                    tmp = np.log(G)
                    reg = max(self.init_entropic_noise * tmp * (1 / n_iter)**(1 - 1/n_iter), .01)
                    if growth_only:
                        R_opt_traj[start_index:start_index + N_sample] = self._growth_log_mass(
                            pairwise_dist, src_real, delta_t, reg, tmp, time, times[t_idx + 1]) / delta_t

                # Trajectory update (skipped by the growth pass)
                if N_sample and N_cells and not growth_only:
                    # --- Growth-weighted OT marginals ---
                    # WOT convention (Schiebinger et al. 2019): both the source AND
                    # target marginals are corrected by exp(±R·Δt/2), not just the
                    # source by exp(R·Δt) — see docs/advanced.md#net-proliferation-rate--default-behaviour.
                    _r = getattr(self, '_prolif_net_rate', None)
                    if _r is not None:
                        mu = np.exp(_r[src_real] * delta_t / 2)
                        nu = np.exp(-_r[cell_idx] * delta_t / 2)
                        mu /= mu.sum()
                        nu /= nu.sum()
                    else:
                        mu = np.ones(N_sample) / N_sample
                        nu = np.ones(N_cells) / N_cells

                    reg_m = np.array([1e3, self.unbalanced_reg * tmp]) if self.unbalanced_reg else None
                    coupling = self._solve_ot(mu, nu, pairwise_dist, reg, reg_m)

                    # Draw one target per trajectory from its coupling row (inverse CDF)
                    cdf = np.cumsum(coupling, axis=1)
                    u = np.random.random(N_sample) * cdf[:, -1]
                    m_idx = np.minimum((cdf < u[:, None]).sum(axis=1), N_cells - 1)
                    tgt = cell_idx[m_idx]

                    # End states recomputed for the sampled pairs only
                    next_prot = find_next_prot(
                        self.d[1, ns:], prot_init, rna_modified[current_indices, ns:],
                        vect_rna[tgt, ns:], mode_init, mode_end[m_idx], alpha_init, s1_s, delta_t)

                    kon_modified[next_indices] = self.modes[tgt]
                    if self.compute_with_proba:
                        proba_modified[next_indices] = self.proba[tgt]
                    rna_modified[next_indices, ns:] = vect_rna[tgt, ns:]
                    prot_modified[next_indices, ns:] = next_prot
                    prot_formodes[tgt, ns:] = next_prot
                    to_keep_for_update[tgt] = 1
                    sim_real_idx[next_indices] = tgt

                    # Re-assign alpha from the nearest old trajectory of the block (weighted L1)
                    if prot_old_blk is not None:
                        w_p, w_k = 1 / G, (G - 1) / G
                        d_match = cdist(np.hstack([next_prot * w_p, kon_modified[next_indices, ns:] * w_k]),
                                        np.hstack([prot_old_blk * w_p, kon_old_blk * w_k]), 'cityblock')
                        alpha_modified[t_idx + 1, alpha_indices] = alpha_old_blk[np.argmin(d_match, axis=1)]

                offset += N_full[s_idx]

        return prot_modified, prot_formodes, rna_modified, kon_modified, \
                proba_modified, alpha_modified, vect_samples_id_modified, \
                  R_opt_traj, to_keep_for_update

    def _solve_ot(self, mu, nu, C, reg, reg_m=None):
        """Entropic OT plan (unbalanced if reg_m is given), relaxing the tolerance until Sinkhorn succeeds."""
        stopThr, numItermax = self.stopThr_init, int(10000 / min(1, reg))
        while stopThr <= self.stopThr_init * 100:
            try:
                if reg_m is None:
                    return ot.bregman.sinkhorn(mu, nu, C, reg=reg, numItermax=numItermax, stopThr=stopThr)
                return ot.unbalanced.sinkhorn_unbalanced(mu, nu, C, reg=reg, reg_m=reg_m,
                                                         numItermax=numItermax, stopThr=stopThr)
            except Exception:
                stopThr *= 2
                numItermax *= 2
        print('Warning, main Sinkhorn did not converge')
        if reg_m is None:
            return ot.bregman.sinkhorn(mu, nu, C, reg=reg, numItermax=numItermax, stopThr=stopThr)
        return ot.unbalanced.sinkhorn_unbalanced(mu, nu, C, reg=reg, reg_m=reg_m,
                                                 numItermax=numItermax, stopThr=stopThr)

    def _growth_log_mass(self, C, src_real, delta_t, reg, log_G, t_from, t_to):
        """
        Log mass gain of each source state over one interval (Waddington-OT growth
        estimation): the source marginal starts from the prior growth exp(R_prior·Δt)
        and is relaxed (growth_reg_source) while the target (observed cells) stays
        nearly hard; n_growth_iter times, source weights <- row marginals. The mean
        population growth is set by population_sizes if given, else by the prior.
        """
        N_src, N_tgt = C.shape
        r = getattr(self, '_prolif_net_rate', None)
        log_prior = r[src_real] * delta_t if r is not None else np.zeros(N_src)
        mu = np.exp(log_prior - log_prior.max())
        mu /= mu.sum()
        nu = np.ones(N_tgt) / N_tgt
        reg_m = np.array([self.growth_reg_source * log_G, 1e3])
        for _ in range(max(1, self.n_growth_iter)):
            row = np.maximum(self._solve_ot(mu, nu, C, reg, reg_m).sum(axis=1), EPS)
            mu = row / row.sum()
        log_m = np.log(mu * N_src)  # relative gain: mean(exp(log_m)) = 1
        sizes = self.population_sizes
        if sizes is not None and t_from in sizes and t_to in sizes:
            return log_m + np.log(sizes[t_to] / sizes[t_from])
        return log_m + np.log(np.mean(np.exp(log_prior)))

    def _traj_strata(self, data_rna, vect_t, vect_samples_id, times, samples_id):
        """
        Stratum label of each real cell, used to build balanced OT batches (source
        trajectories by their t0 cell, real target cells): cell types if provided, else
        k-means clusters of log1p(rna) per (sample, time) where cells are split into
        several batches. None = uniform batches.
        """
        _ct = getattr(self, '_strata_labels', None)
        if _ct is not None:
            return _ct
        if not self.n_strata_traj:
            return None
        ns = self.n_stimuli
        strata = np.zeros(len(vect_t), dtype=int)
        for s_idx, sample in enumerate(samples_id):
            for t_idx, t in enumerate(times):
                idx = np.flatnonzero((vect_t == t) & (vect_samples_id == sample))
                if len(idx) <= min(self.batch_size_traj, self.batch_size_traj_exp):
                    continue
                km = MiniBatchKMeans(n_clusters=min(self.n_strata_traj, len(idx)), n_init=3,
                                     random_state=task_seed(self.seed, 7, s_idx, t_idx))
                strata[idx] = km.fit_predict(np.log1p(data_rna[idx, ns:]))
        return strata

    def _n_params_per_target(self, active_cols, n_samples):
        """Largest parameter count of one target-gene fit: (active regulators + per-sample basals) x n_networks."""
        k_max = max((len(c) for c in active_cols[self.n_stimuli:]), default=0)
        return (k_max + n_samples) * int(self.n_networks)

    def _network_batch_size(self, n_params):
        """Network sub-sample size: at least 10 states per parameter; batch_size_network (if not None) can only raise it."""
        floor = 10 * n_params
        return floor if self.batch_size_network is None else max(self.batch_size_network, floor)

    def _fit_theta_averaged(self, fit_fn, times_vec, samples_vec, labels, n_fits, n_params):
        """
        Theta from fit_fn(sels) (one fit per sub-sample, run jointly) on sub-samples of
        _network_batch_size(n_params) trajectory states,
        stratified by (time, sample) and cell type (labels, if not None). Mean over
        min(n_fits, 1 + n_states // batch_size) disjoint sub-samples (covering every state
        about once, at most n_fits); a single fit when one sub-sample holds every state.
        Returns (basal, inter, basal_tmp, inter_tmp).
        """
        batch_size = self._network_batch_size(n_params)
        # Disjoint sub-samples covering as many states as possible (a single one if it holds them all)
        n_fits = max(5, min(n_fits, 1 + len(times_vec) // batch_size))
        sels, _ = grouped_partition([times_vec, samples_vec], batch_size, n_fits, labels)
        self._net_batch_info = (int(np.mean([len(sel) for sel in sels])), len(sels), len(times_vec))  # for the iteration log
        fits = fit_fn(sels)
        return tuple(np.mean([f[i] for f in fits], axis=0) for i in range(4))

    def loop_trajectories(
        self,
        data_rna,
        vect_t,
        vect_samples_id,
        times,
        samples_id,
        ks,
        s1,
        init_cells_full,
        N_full,
        N_samples,
        G_tot,
        min_n_loops,
        count_max,
        intensity_prior,
        basal_init=None,
        inter_init=None,
        basal_ref=None,
        inter_ref=None,
        verb=True,
        compute_theta=True,
        initialize_alpha=True,
        kov_cell_mask=None,
        hard_forcing_ref=False,
        ref_constraint_pct=0.1,
        n_iter_offset=None,
    ):
        """
        Alternating optimization of trajectories and network (theta).

        n_iter_offset : int or None
            Iterations already done (test set: those of the training inference), added to the
            counter of the Sinkhorn regularization and of the basin-update weight, so that a fixed
            network is used with the final formulas of the training. None: training schedule.

        basal_init / inter_init : (G_tot, n_networks) / (G_tot, G_tot, n_networks) or None
            Starting point for theta. Zeros if None.
        kov_cell_mask : (N_cells, G_tot) int8 array or None
            Per-cell KO/OV constraints. -1 → KO (force lowest mode),
            +1 → OV (force highest mode), 0 → no constraint.
            Applied after each update_modes step as a hard override.
        basal_ref / inter_ref : same shape or None
            Regularization target passed to inference_network. Zeros if None (no prior).
        """

        n_iter = 1
        errors = [1e12]
        count_end = 0
        N_tot = np.sum(N_full)
        # Iteration counter of the OT regularization and of the basin weights
        it_shift = (n_iter_offset if n_iter_offset is not None
                    else min_n_loops * min(1, 1 - compute_theta + hard_forcing_ref))
        it_basins = n_iter_offset or 0

        n_samples_local = len(samples_id)
        if compute_theta:
            # --- Initialize theta parameters ---
            # basal: (n_samples, G_tot, n_networks); inter: (G_tot, G_tot, n_networks)
            self.basal = np.zeros((n_samples_local, G_tot, self.n_networks))
            self.inter = np.zeros((G_tot, G_tot, self.n_networks))
            if basal_init is not None:
                # basal_init already (n_samples, G, n_networks) from _normalize_theta
                self.basal[:, :, :] = basal_init[:n_samples_local]
            if inter_init is not None:
                self.inter[:, :, :] = inter_init

        # Ensure basal is always 3-D (n_samples, G_tot, n_networks) — promote 2-D for compat
        if self.basal.ndim == 2:
            self.basal = self.basal[np.newaxis, :, :]
        basal = self.basal.copy()      # (n_samples, G_tot, n_networks)
        inter = self.inter.copy()      # (G_tot, G_tot, n_networks)
        basal_tmp = self.basal.copy()
        inter_tmp = self.inter.copy()
        # Regularization targets: provided prior, or zeros (no penalization)
        _basal_ref = basal_ref if basal_ref is not None else np.zeros((n_samples_local, G_tot, self.n_networks))
        _inter_ref = inter_ref if inter_ref is not None else np.zeros((G_tot, G_tot, self.n_networks))
        basal_ref, inter_ref = _basal_ref, _inter_ref

        # --- Initialize switching probabilities alpha ---
        if initialize_alpha:
            ns = self.n_stimuli
            self.alpha = np.random.uniform(.1, .9, size=(len(times) - 1, N_tot, G_tot - ns))
            if np.linalg.norm(self.ref_network[:ns, :]):
                self.alpha[0] = .01

        # --- Time vector for full and reduced datasets ---
        vect_t_sim = np.repeat(times, N_tot)

        # --- Initialize placeholders ---
        y_prot = np.zeros((len(times) * N_tot, G_tot))
        # Flow-matching protein states, stored per target gene on its active regulators only
        y_prot_prev = None
        if compute_theta:
            _, prev_cols = active_regulators(self.ref_network, inter_ref, hard_forcing_ref)
            y_prot_prev = PrevProt.zeros(prev_cols, len(times) * N_tot)
            slot_s = np.repeat(np.arange(len(N_full)), N_full)[:N_tot]  # sample of each trajectory slot
            for t_idx, t_i in enumerate(times):
                sl = slice(t_idx * N_tot, (t_idx + 1) * N_tot)
                # Same scaling as the stimulus columns of y_prot (per sample if it has its own schedule)
                stim_val = self._stim_schedule.per_cell(t_i, slot_s) * self.scale_proteins
                for c, v in zip(y_prot_prev.cols, y_prot_prev.values):
                    is_stim = c < self.n_stimuli
                    v[sl, is_stim] = stim_val[:, c[is_stim]]
        y_kon = np.zeros_like(y_prot)
        y_rna = np.zeros_like(y_prot)
        y_proba = np.zeros((len(times) * N_tot, G_tot, self.n_networks + 1))
        y_alpha = self.alpha.copy()
        y_samples = np.zeros(len(vect_t_sim), dtype=int)
        R_opt_traj = np.zeros(len(vect_t_sim))

        # Number of reconstruction batches per sample (ceil(N_full[s] / N_samples[s])) —
        # this also fixes the number of real-cell batches built below, so that batch b of
        # the reconstruction ensemble always targets batch b of the real cells.
        n_batches_per_sample = [
            int(np.ceil(N_full[s] / N_samples[s])) if N_samples[s] > 0 else 1
            for s in range(n_samples_local)
        ]
        # Trajectory states carry sample indices (y_samples), not dataset_id labels
        sample_idx = np.arange(len(samples_id))

        def refresh_prev_prot():
            # Flow-matching states at t+1 from the current trajectories and alphas
            s1r = s1_rows(s1, y_samples)
            modes = self.adaptive_shrinkage(y_rna[:, ns:] * s1r, y_kon[:, ns:]) / s1r
            for cnt, time in enumerate(times[:-1]):
                idx_prev = slice(N_tot * cnt, N_tot * (cnt + 1))
                idx_next = slice(N_tot * (cnt + 1), N_tot * (cnt + 2))
                self._fill_prev_prot(
                    y_prot_prev, idx_next, y_alpha[cnt], times[cnt + 1] - time,
                    self.d[1, ns:], y_prot[idx_prev, ns:],
                    y_rna[idx_prev, ns:] * self.scale_proteins,
                    y_rna[idx_next, ns:] * self.scale_proteins,
                    modes[idx_prev], modes[idx_next], s1r[idx_prev] if np.ndim(s1r) == 2 else s1r)

        # Real cell behind each trajectory state, and its cell type (None if unavailable)
        y_real = np.full(len(vect_t_sim), -1, dtype=int)
        cell_types = getattr(self, '_strata_labels', None)

        def traj_cell_types():
            if cell_types is None:
                return None
            return np.where(y_real >= 0, cell_types[np.maximum(y_real, 0)], '')

        def fit_theta(weight_prev, basal_start, inter_start, n_fits):
            # Network fit(s) on (time, sample, cell type)-stratified subsamples of the trajectories
            def fit_on(sels):
                return inference_network_multi(
                    sels, y_samples, y_kon, y_proba, y_prot, y_prot_prev,
                    ks, n_stimuli=ns, samples_id=samples_id,
                    ref_network=self.ref_network, basal_init=basal_start, inter_init=inter_start,
                    basal_ref=basal_ref, inter_ref=inter_ref,
                    proba=self.compute_with_proba, scale=self.scale_pen,
                    weight_prev=weight_prev, loss=self.loss_norm,
                    final=0, constrain_basal_uniform=self.constrain_basal_uniform,
                    hard_forcing_ref=hard_forcing_ref, ref_constraint_pct=ref_constraint_pct,
                    seuil_zero_min_ref=self.seuil_zero_min_ref)
            return self._fit_theta_averaged(fit_on, vect_t_sim, y_samples, traj_cell_types(), n_fits,
                                            self._n_params_per_target(prev_cols, n_samples_local))

        # Stratum of each real cell, for balanced OT batches; real cells at t0 per sample
        strata = self._traj_strata(data_rna, vect_t, vect_samples_id, times, samples_id)
        t0_real = [np.flatnonzero((vect_t == times[0]) & (vect_samples_id == sample)) for sample in samples_id]

        # === Main loop ===
        while count_end <= count_max:

            weight_prev = self.weight_prev * min(1, (n_iter-1)/min_n_loops) # Flow matching from second iteration and small at early ones

            if count_end == count_max or n_iter > self.max_iter:
                break

            # --- Shuffle order of cells for each sample ---
            # Source batches are consecutive slot blocks: with strata, the order is a stratified
            # interleaving on the t0 cell of each trajectory, so every batch holds each stratum in proportion
            indices_shuffled = [
                np.random.permutation(N_full[s]) if strata is None
                else stratified_order(np.arange(N_full[s]), strata[t0_real[s][init_cells_full[s]]])
                for s in range(len(samples_id))
            ]
            init_cells_full = [
                init_cells_full[s][indices_shuffled[s]]
                for s in range(len(samples_id))
            ]
            if n_iter > 1:

                # Reorder alpha, y_prot, y_kon according to new cell order
                offset = 0
                for cnt, time in enumerate(times):
                    offset = 0
                    for s, N in enumerate(N_full):
                        if time != times[-1]:
                            y_alpha[cnt, offset:offset+N] = y_alpha[cnt, offset + indices_shuffled[s]]
                        y_prot[cnt * N_tot + offset : cnt * N_tot + offset + N] = y_prot[cnt * N_tot + offset + indices_shuffled[s]]
                        y_kon[cnt * N_tot + offset : cnt * N_tot + offset + N] = y_kon[cnt * N_tot + offset + indices_shuffled[s]]
                        y_rna[cnt * N_tot + offset : cnt * N_tot + offset + N] = y_rna[cnt * N_tot + offset + indices_shuffled[s]]
                        y_proba[cnt * N_tot + offset : cnt * N_tot + offset + N] = y_proba[cnt * N_tot + offset + indices_shuffled[s]]
                        offset += N

            # --- Redraw per-sample, per-time batches of real (experimental) cells ---
            # Redrawn every outer iteration (fresh randomness, same cadence as the
            # reconstruction-batch reshuffle above) to maximize mixing over the run.
            # For sample s at time times[t_idx + 1], with K_nom = batch_size_traj_exp the
            # target real-cell batch size (decoupled from N_samples[s], which only sizes
            # the reconstruction/departure batches) and B_s = n_batches_per_sample[s] the
            # number of batches to build, each batch gets size K = min(N_real, max(K_nom, ceil(N_real / B_s))):
            #   - N_real <= K_nom (few real cells): K == N_real, so every batch is simply
            #     the full real-cell set (no split at all);
            #   - N_real >= B_s * K_nom (as many/more real cells than B_s * batch_size_traj_exp):
            #     K == ceil(N_real / B_s), a plain partition with ~zero redundancy;
            #   - in between: batches are pinned to K_nom, which requires some random
            #     redundancy to still cover all real cells via their union — achieved by
            #     concatenating independent shuffles of the real cells and slicing
            #     consecutive chunks, which spreads the redundancy as evenly as possible
            #     while guaranteeing full coverage.
            # With strata, each shuffle is a stratified interleaving, so every batch holds
            # each cell type / cluster in proportion (rare states are not left out).
            def shuffle_real(idx):
                return np.random.permutation(idx) if strata is None else stratified_order(idx, strata[idx])

            real_cell_batches = [[None] * (len(times) - 1) for _ in range(n_samples_local)]
            for s_idx, sample in enumerate(samples_id):
                K_nom = max(self.batch_size_traj_exp, N_samples[s_idx])
                B_s = n_batches_per_sample[s_idx]
                for t_idx in range(len(times) - 1):
                    idx = np.flatnonzero((vect_t == times[t_idx + 1]) & (vect_samples_id == sample))
                    N_real = len(idx)
                    if N_real == 0:
                        real_cell_batches[s_idx][t_idx] = [idx for _ in range(B_s)]
                        continue
                    K = min(N_real, max(K_nom, int(np.ceil(N_real / B_s))))
                    L = B_s * K
                    reps = int(np.ceil(L / N_real))
                    tiled = np.concatenate([shuffle_real(idx) for _ in range(reps)])[:L]
                    real_cell_batches[s_idx][t_idx] = [tiled[b * K:(b + 1) * K] for b in range(B_s)]

            offset_init = [0] * len(samples_id)
            to_keep_for_update = np.zeros(len(vect_t), dtype=bool)
            to_keep_for_update[vect_t == times[0]] = True
            y_prot_formodes = np.zeros_like(self.modes)
            while not np.array_equal(offset_init, N_full):
                N_tmp = [min(N_samples[s], N_full[s] - offset_init[s]) for s in range(len(samples_id))]
                init_cells = [init_cells_full[s][offset_init[s]:offset_init[s] + N_tmp[s]] for s in range(len(samples_id))]
                batch_idx = [
                    min(offset_init[s] // N_samples[s], n_batches_per_sample[s] - 1) if N_samples[s] > 0 else 0
                    for s in range(len(samples_id))
                ]
                y_prot, y_prot_formodes, y_rna, y_kon, y_proba, y_alpha, y_samples, R_opt_traj, to_keep_for_update = \
                    self.estimate_trajectories_given_model(
                        vect_t, times, vect_samples_id, samples_id,
                        data_rna, y_prot, y_prot_formodes, y_kon, y_rna, y_proba, y_alpha, y_samples,
                        basal, inter, s1, ks, init_cells,
                        R_opt_traj, to_keep_for_update,
                        offset_init=offset_init,
                        N_full=N_full, N_samples=N_tmp,
                        n_iter=n_iter + it_shift,
                        intensity_prior=intensity_prior * compute_theta * (1 - hard_forcing_ref),
                        real_cell_batches=real_cell_batches, batch_idx=batch_idx,
                        sim_real_idx=y_real
                    )

                offset_init = [offset_init[s] + N_tmp[s] for s in range(len(samples_id))]

            if y_prot_prev is not None:
                for c, v in zip(y_prot_prev.cols, y_prot_prev.values):
                    is_gene = c >= ns
                    v[:N_tot, is_gene] = y_prot[:N_tot, c[is_gene]]

            # --- Evaluate error before and after inference ---
            error = self._count_errors_per_sample(y_prot, y_kon, y_proba, ks, inter, basal,
                                                   samples_id=sample_idx, samples_data=y_samples)
            if compute_theta and len(times) > 1:
                if self.weight_prev > 0:
                    refresh_prev_prot()
                # Mean of fits on independent stratified subsamples covering the trajectory states
                basal, inter, basal_tmp, inter_tmp = fit_theta(weight_prev, basal, inter, self.n_network_fits)

            error_2 = self._count_errors_per_sample(y_prot, y_kon, y_proba, ks, inter, basal,
                                                    samples_id=sample_idx, samples_data=y_samples)
            errors.append(error_2)
            self.loss_trajectory.append(error_2)
            self.theta_trajectory.append(inter_tmp)

            if verb:
                print(f"{n_iter}", f"{count_end} | Errors (before, after): {error:.5f}, {error_2:.5f} | alpha mean: {np.mean(y_alpha[0]):.4f}")
                one = len(N_samples) == 1
                traj_bs, traj_tot = (N_samples[0], N_full[0]) if one else (list(N_samples), list(N_full))
                net_log = ""
                if compute_theta and len(times) > 1:
                    net_bs, n_net, net_tot = self._net_batch_info
                    net_log = f"; net = {net_bs}" + (f" x{n_net} fits" if n_net > 1 else "") + f" (tot = {net_tot})"
                print(f"  n_cells per batch: traj = {traj_bs} (tot = {traj_tot}){net_log}")

            # --- Update counts for stopping condition if n_iter is high enough ---
            if count_end >= 1:
                if (errors[-2] - errors[-1]) < 1e-3:
                    count_end += 1
                ### If we compute theta, the absence of difference before and after update of theta is also taken into account
                if compute_theta:
                    if np.abs(error - error_2) < 2e-4:
                        count_end += 1
            # Unblock the counter
            if (count_end < 1) and (n_iter > min_n_loops) and (errors[-1] - errors[-2]) > 0:
                count_end += 1
            n_iter += 1

            # --- Update kon_theta for alpha ---
            kon_vector = y_kon.copy()
            kon_vector[:, ns:] = self._kon_ref_per_sample(y_prot, ks, inter, basal, samples_id=sample_idx, samples_data=y_samples)[:, ns:]

            # --- Update alphas ---
            s1r = s1_rows(s1, y_samples)
            modes = self.adaptive_shrinkage(y_rna[:, ns:] * s1r, y_kon[:, ns:]) / s1r
            if len(times) > 1:
                # Intervals are independent: one parallel task each (numba on 1 thread per task)
                alpha_fn = inference_alpha_1thread if len(times) > 2 else inference_alpha
                def alpha_task(cnt, time):
                    return delayed(alpha_fn)(
                            self.d[1, ns:], s1,
                            y_alpha[cnt],
                            y_kon[vect_t_sim == time],
                            kon_vector[vect_t_sim == time],
                            y_prot[vect_t_sim == time],
                            y_rna[vect_t_sim == time],
                            y_kon[vect_t_sim == times[cnt + 1]],
                            kon_vector[vect_t_sim == times[cnt + 1]],
                            y_prot[vect_t_sim == times[cnt + 1]],
                            y_rna[vect_t_sim == times[cnt + 1]],
                            modes[vect_t_sim == time], modes[vect_t_sim == times[cnt + 1]],
                            basal, inter, ks, times[cnt + 1] - time,
                            tol=self.alpha_threshold,
                            n_pas = self.n_pas if self.force_n_pas else max(self.n_pas, int(times[cnt + 1] - time)),
                            samples_data=y_samples[vect_t_sim == time],
                            stim_vals=self._stim_schedule.per_cell(times[cnt + 1], y_samples[vect_t_sim == time]),
                            scale_proteins=self.scale_proteins
                        )
                # Same pool size as every other Parallel call: a different n_jobs makes loky respawn workers
                alphas = Parallel(n_jobs=-1)(
                    alpha_task(cnt, time) for cnt, time in enumerate(times[:-1]))
                for cnt, alpha_cnt in enumerate(alphas):
                    y_alpha[cnt] = alpha_cnt
            
            # --- Update kon_theta values for modes ---
            # y_prot_formodes is indexed like the original data (shape = N_original_cells),
            # so we must use vect_samples_id (original) — not y_samples which has N_traj_cells rows.
            kon_vector_formodes = y_prot_formodes.copy()
            kon_vector_formodes[:, ns:] = self._kon_ref_per_sample(y_prot_formodes, ks, inter, basal, 
                                    samples_id=samples_id, samples_data=vect_samples_id)[:, ns:]
            print("number of non reached cells", np.sum(to_keep_for_update == 0))

            # --- Update modes ---
            if self.update_modes:
                n_cells = self.proba_init.shape[0]
                weight_prob = max(.96**(n_iter + it_basins - 1), .1) # the weight of the network increases slowly because it aims to get the right attribution given probabilities that are close
                # Mass constraint grows from the argmax masses (lam=0: plain argmax) to nu (lam=1: full OT) by min_n_loops
                lam = min(1.0, (n_iter - 1) / max(min_n_loops - 1, 1)) if compute_theta else 1.0

                # Mode amplitudes of each real cell's sample (per-sample mixtures), and their max
                ks_cells = (ks[np.searchsorted(np.asarray(samples_id), vect_samples_id)]
                            if ks.ndim == 3 else None)
                ks_max = ks.max(axis=0) if ks.ndim == 3 else ks

                def assign_basins(mu, nu, dist):
                    # EMD with target masses interpolated between those of the argmax and nu
                    nu_argmax = np.bincount(np.argmin(dist, axis=1), minlength=dist.shape[1]) / dist.shape[0]
                    coupling = ot.emd(mu, (1 - lam) * nu_argmax + lam * nu, dist, numItermax=int(1e7))
                    return np.argmax(coupling, axis=1)
                
                def run_main_loop_for_gene(g, temporal=self.temporal_basins):
                    l_max = 1 + np.argmax(ks_max[g, :])
                    obj = np.zeros((n_cells, l_max), dtype=float)
                    obj[:, :] = ks_cells[:, g, :l_max] if ks_cells is not None else ks[g, :l_max][None, :]
                    tmp_proba = np.zeros_like(self.proba[:, g])
                    tmp_modes = np.zeros_like(self.modes[:, g])
                    if temporal:
                        for t_i in times:
                            indices = (vect_t == t_i)
                            tmp_proba_i = np.zeros_like(self.proba[indices, g])
                            tmp_modes_i = np.zeros_like(self.modes[indices, g])
                            proba_i = self.proba_init[indices, g, :l_max].copy()
                            # Target masses: expected counts from posteriors normalised per cell
                            post_i = proba_i / np.maximum(np.sum(proba_i, axis=1, keepdims=True), EPS)
                            proba_i /= np.max(proba_i, axis=1, keepdims=True)
                            obj_i = obj[indices].copy()
                            n_cells_i = np.sum(indices)
                            mu = np.ones(n_cells_i)/n_cells_i
                            nu = self.pi_init[g - ns][t_i][:l_max] * self.force_basins + np.sum(
                                                        post_i, axis=0) * (1 - self.force_basins)
                            nu /= np.sum(nu)
                            diff_k = np.maximum(1 - np.abs(kon_vector_formodes[indices, g, None] - obj_i), 1e-3)
                            diff_k /= np.max(diff_k, axis=1, keepdims=True)
                            dist = - (np.log(proba_i) + (1 - weight_prob) * 
                                      to_keep_for_update[indices, None] * np.log(diff_k))
                            dist = np.clip(dist, 0, 100)
                            idx = assign_basins(mu, nu, dist)
                            for cell in range(n_cells_i):
                                tmp_proba_i[cell, idx[cell]] = 1
                                tmp_modes_i[cell] = obj_i[cell, idx[cell]]
                            tmp_proba[indices, :] = tmp_proba_i[:, :]
                            tmp_modes[indices] = tmp_modes_i[:]
                    else:
                        proba = self.proba_init[:, g, :l_max].copy()
                        post = proba / np.maximum(np.sum(proba, axis=1, keepdims=True), EPS)
                        proba /= np.max(proba, axis=1, keepdims=True)
                        mu = np.ones(n_cells)/n_cells
                        nu = np.sum([self.pi_init[g - ns][t_i] * np.sum(vect_t == t_i)/n_cells
                                                    for t_i in times], axis=0)[:l_max] * self.force_basins + np.sum(
                                                        post, axis=0) * (1 - self.force_basins)
                        nu /= np.sum(nu)
                        diff_k = np.maximum(1 - np.abs(kon_vector_formodes[:, g, None] - obj), 1e-3)
                        diff_k /= np.max(diff_k, axis=1, keepdims=True)
                        dist = - (np.log(proba) + (1 - weight_prob) * 
                                  to_keep_for_update[:, None] * np.log(diff_k))
                        dist = np.clip(dist, 0, 100)
                        idx = assign_basins(mu, nu, dist)
                        for cell in range(n_cells):
                            tmp_proba[cell, idx[cell]] = 1
                            tmp_modes[cell] = obj[cell, idx[cell]]
                    return tmp_proba, tmp_modes
                results = Parallel(n_jobs=-1)(
                    delayed(run_main_loop_for_gene)(g) for g in range(ns, G_tot))

                for idx, g in enumerate(range(ns, G_tot)):
                    tmp_proba, tmp_modes = results[idx]
                    self.proba[:, g, :], self.modes[:, g] = tmp_proba[:, :], tmp_modes[:]

                # ── Force KO/OV cells to the correct mode after update ────────
                if kov_cell_mask is not None:
                    for g in range(ns, G_tot):
                        l_max = 1 + int(np.argmax(ks_max[g, :]))
                        ko_cells = kov_cell_mask[:, g] < 0
                        ov_cells = kov_cell_mask[:, g] > 0
                        if np.any(ko_cells):
                            self.proba[ko_cells, g, :] = 0
                            self.proba[ko_cells, g, 0] = 1
                            self.modes[ko_cells, g] = ks_cells[ko_cells, g, 0] if ks_cells is not None else ks[g, 0]
                        if np.any(ov_cells):
                            self.proba[ov_cells, g, :] = 0
                            self.proba[ov_cells, g, l_max - 1] = 1
                            self.modes[ov_cells, g] = ks_cells[ov_cells, g, l_max - 1] if ks_cells is not None else ks[g, l_max - 1]

        # --- Growth pass: net growth of the final trajectories given the final network ---
        # Done once, out of the loop: the growth correction is not fed back into the inference
        R_opt_traj = np.full(len(vect_t_sim), np.nan)
        if len(times) > 1:
            offset_init = [0] * len(samples_id)
            while not np.array_equal(offset_init, N_full):
                N_tmp = [min(N_samples[s], N_full[s] - offset_init[s]) for s in range(len(samples_id))]
                init_cells = [init_cells_full[s][offset_init[s]:offset_init[s] + N_tmp[s]] for s in range(len(samples_id))]
                batch_idx = [
                    min(offset_init[s] // N_samples[s], n_batches_per_sample[s] - 1) if N_samples[s] > 0 else 0
                    for s in range(len(samples_id))
                ]
                R_opt_traj = self.estimate_trajectories_given_model(
                    vect_t, times, vect_samples_id, samples_id,
                    data_rna, y_prot, y_prot_formodes, y_kon, y_rna, y_proba, y_alpha, y_samples,
                    basal, inter, s1, ks, init_cells,
                    R_opt_traj, to_keep_for_update,
                    offset_init=offset_init,
                    N_full=N_full, N_samples=N_tmp,
                    n_iter=n_iter - 1 + it_shift,
                    intensity_prior=intensity_prior * compute_theta * (1 - hard_forcing_ref),
                    real_cell_batches=real_cell_batches, batch_idx=batch_idx,
                    sim_real_idx=y_real, growth_only=True
                )[7]
                offset_init = [offset_init[s] + N_tmp[s] for s in range(len(samples_id))]
            if verb:
                print(f"[fit_network] Growth pass: net rate per state in "
                      f"[{np.nanmin(R_opt_traj):.3g}, {np.nanmax(R_opt_traj):.3g}], mean {np.nanmean(R_opt_traj):.3g}")
        self.R_opt = R_opt_traj
        # Last iteration of the basin weights (continued by the test-set inference)
        self.n_iter_final = n_iter - 1 + it_basins

        # --- Updating the networks ---
        if compute_theta:
            self.basal = basal
            self.inter = inter
            self.basal_tmp = basal_tmp
            self.inter_tmp = inter_tmp

        # --- Store results ---
        self.kon_theta = kon_vector
        self.kon_beta = y_kon
        self.rna = y_rna
        self.prot = y_prot
        self.proba_traj = y_proba
        self.samples_data = y_samples
        self.times_data = vect_t_sim
        self.alpha = y_alpha
        # Real cell behind each trajectory state and its cell type (stratify later batches)
        self.traj_real_idx = y_real
        self.traj_cell_types = traj_cell_types()

        # Harissa: continuous adaptive_shrinkage burst-rate estimates, used in
        # refine_network_degradations in place of discrete mode assignments (kon_beta).
        self.kon_beta_harissa = y_kon.copy()
        self.kon_beta_harissa[:, ns:] = (
            self.adaptive_shrinkage(y_rna[:, ns:] * s1_rows(s1, y_samples), y_kon[:, ns:])
        )


    @staticmethod
    def _normalize_theta(arr, G_tot, n_networks, n_samples=1, is_inter=False):
        """Normalize basal/inter init or ref arrays.

        For inter (is_inter=True):
            (G, G)             → (G, G, n_networks)  broadcast to all networks
            (G, G, 1)          → (G, G, n_networks)  broadcast to all networks
            (G, G, n_networks) → as-is

        For basal (is_inter=False):
            (G,)                     → (1, G, n_networks)  broadcast to 1 sample
            (G, n_networks)          → (1, G, n_networks)  broadcast to 1 sample
            (n_samples, G, n_networks) → as-is

        Returns None if arr is None.
        """
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        if is_inter:
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            # arr is now 3D; expand single-layer to all networks
            if arr.shape[2] == 1 and n_networks > 1:
                arr = np.repeat(arr, n_networks, axis=2)
            return arr
        else:
            if arr.ndim == 1:                          # (G,) → (1, G, n_networks)
                arr = np.repeat(arr[:, np.newaxis], n_networks, axis=1)
                return arr[np.newaxis, :, :]           # (1, G, n_networks)
            if arr.ndim == 2:                          # (G, n_networks) → (1, G, n_networks)
                return arr[np.newaxis, :, :]
            return arr                                  # assumed (n_samples, G, n_networks)

    def fit_network(
        self,
        data,
        intensity_prior=10,
        vect_samples_id=None,
        basal_init=None,
        inter_init=None,
        basal_ref=None,
        inter_ref=None,
        verb=True,
        stimulus_schedule=None,
        transition_rates=None,
        time_key='time',
        hard_forcing_ref=None,
        ref_constraint_pct=None,
    ):
        """
        Fit the gene regulatory network to the RNA expression data.

        Parameters
        ----------
        data : ndarray or AnnData
            RNA expression matrix (cells × genes).
        intensity_prior : float
            Regularization intensity for optimal transport.
        vect_samples_id : ndarray or None
            Array of sample labels (same size as data), or None if only one sample.
        basal_init : ndarray or None
            Initial basal rates: shape (G,) broadcast to all networks, or (G, n_networks).
        inter_init : ndarray or None
            Initial interaction matrix: shape (G, G) or (G, G, n_networks).
        basal_ref : ndarray or None
            Regularization target for basal rates, same shape rules as basal_init.
            Defaults to zeros (no penalization towards a prior).
        inter_ref : ndarray or None
            Regularization target for interactions, same shape rules as inter_init.
            Defaults to zeros.
        verb : bool
            Whether to print progress.
        """
        seed_everything(self.seed)

        # --- Initialization ---
        data_rna = self._parse_input(data, time_key, scale_depth=True)  # counts at the reference depth
        if stimulus_schedule is not None or self._stim_schedule is None:
            self._stim_schedule = self._build_stimulus_schedule(
                np.sort(np.unique(data_rna[:, 0])), stimulus_schedule)
        G_tot = data_rna.shape[1]
        vect_t = data_rna[:, 0]
        ns = self.n_stimuli

        # --- Adapt ref_network ---
        self.ref_network = signed_floor(self.ref_network, self.prior_network_pen)  # keeps signed priors
        self.ref_network[:ns, :] = self.stimulus
        self._apply_stimulus_targets()
        print(self._stim_schedule)
        for g in range(ns, G_tot):
            l_max = len(np.unique(self.modes[:, g]))
            if l_max < 2:
                self.ref_network[g, :], self.ref_network[:, g] = 0, 0
            if l_max > len(np.unique(self.a[..., :-1, g])):
                self.compute_with_proba = 0

        # Auto-extract vect_samples_id from AnnData if not provided
        try:
            import anndata
            if isinstance(data, anndata.AnnData) and vect_samples_id is None and 'dataset_id' in data.obs:
                vect_samples_id = data.obs['dataset_id'].values
        except ImportError:
            pass

        self._load_ot_constraints(data, transition_rates)

        # If no sample ID provided, assume one global sample
        if vect_samples_id is None:
            vect_samples_id = np.zeros_like(vect_t)

        # Unique time points and sample IDs
        times = np.sort(np.unique(vect_t))
        samples_id = np.sort(np.unique(vect_samples_id))
        self.set_sample_names(samples_id)  # per-sample stimulus schedules

        # --- Compute number of real cells per time/sample ---
        nb_cells = np.zeros((len(samples_id), len(times)), dtype=int)
        for s, sample in enumerate(samples_id):
            for t_idx, t in enumerate(times):
                nb_cells[s, t_idx] = np.sum((vect_t == t) & (vect_samples_id == sample))

        if verb:
            print("[fit_network] Cell counts per sample/timepoint and genes:\n", nb_cells, G_tot)

        # --- Define number of cells used for inference ---
        N_samples = []
        for s in range(len(samples_id)):
            n = int(np.quantile(nb_cells[s], self.quant_samples)) 
            q, r = divmod(n, self.batch_size_traj) 
            if q == 0: N_samples.append(n)
            else: N_samples.append(min(self.batch_size_traj + 1+int(r/q), n))

        N_full = [int(np.quantile(nb_cells[s], self.quant_samples)) for s in range(len(samples_id))]

        if verb:
            print("[fit_network] Number of simulated cells per sample:", N_samples)
            print("[fit_network] Number of total cells per sample:", N_full)

        # --- Choose initial cells per sample ---
        init_cells_full = [
            minimal_repetition_choice(nb_cells[s, 0], N_full[s], labels=self._t0_cell_types(vect_t, vect_samples_id, sample))
            for s, sample in enumerate(samples_id)
        ]

        # --- Extract kinetic parameters ---
        ks, s1, _ = self._mixture_terms()

        # --- Normalize init/ref arrays to (n_samples, G_tot, n_networks) / (G_tot, G_tot, n_networks) ---
        n_samples = len(samples_id)
        nn = self.n_networks
        basal_init = self._normalize_theta(basal_init, G_tot, nn, n_samples, is_inter=False)
        inter_init = self._normalize_theta(inter_init, G_tot, nn, n_samples, is_inter=True)
        basal_ref  = self._normalize_theta(basal_ref,  G_tot, nn, n_samples, is_inter=False)
        inter_ref  = self._normalize_theta(inter_ref,  G_tot, nn, n_samples, is_inter=True)

        # --- Build per-cell KO/OV mask from basal_ref (±100 entries) ─────────
        # kov_cell_mask[cell, g] = -1 (KO) / +1 (OV) / 0 (unconstrained)
        kov_cell_mask = None
        if basal_ref is not None:
            br = np.asarray(basal_ref, dtype=float)
            if br.ndim == 3 and np.any(np.abs(br) > 50):
                cm = np.zeros((len(vect_t), G_tot), dtype=np.int8)
                for s_idx, s in enumerate(samples_id):
                    cell_idx = np.where(vect_samples_id == s)[0]
                    ko_genes = np.where(br[s_idx, :, 0] < -50)[0]
                    ov_genes = np.where(br[s_idx, :, 0] >  50)[0]
                    if len(cell_idx) and len(ko_genes):
                        cm[np.ix_(cell_idx, ko_genes)] = -1
                    if len(cell_idx) and len(ov_genes):
                        cm[np.ix_(cell_idx, ov_genes)] =  1
                if np.any(cm != 0):
                    kov_cell_mask = cm

        # --- Infer theta (basal/interactions) on reduced simulations with mixing ---
        self.loop_trajectories(
            data_rna=data_rna,
            vect_t=vect_t,
            vect_samples_id=vect_samples_id,
            times=times,
            samples_id=samples_id,
            ks=ks,
            s1=s1,
            init_cells_full=init_cells_full,
            N_full=N_full,
            N_samples=N_samples,
            G_tot=G_tot,
            min_n_loops=self.min_n_loops,
            count_max=self.count_max,
            intensity_prior=intensity_prior,
            basal_init=basal_init,
            inter_init=inter_init,
            basal_ref=basal_ref,
            inter_ref=inter_ref,
            verb=verb,
            compute_theta=True,
            initialize_alpha=True,
            kov_cell_mask=kov_cell_mask,
            hard_forcing_ref=hard_forcing_ref if hard_forcing_ref is not None else self.hard_forcing_ref,
            ref_constraint_pct=ref_constraint_pct if ref_constraint_pct is not None else self.ref_constraint_pct,
        )


        # --- Print results (optional) ---
        if verb:
            print("\n[fit_network] Final network:")
            for n in range(self.n_networks):
                print(f"  Network {n} | Interactions:\n", self.inter[:, :, n].T)
                print(f"  Network {n} | Basal:\n", self.basal.mean(axis=0)[:, n])

            print("\n[fit_network] Intermediate network:")
            for n in range(self.n_networks):
                print(f"  Network {n} | Interactions:\n", self.inter_tmp[:, :, n].T)
                print(f"  Network {n} | Basal:\n", self.basal_tmp.mean(axis=0)[:, n])
            

    def _fill_prev_prot(self, prev, idx_next, alpha, delta_t, d1, P0, M0, M1,
                        mode_init, mode_end, s):
        """
        Fill the flow-matching states prev (PrevProt) at rows idx_next from the
        previous-time states P0, for all cells at once. For target g, regulators follow the flow
        over delta_t * alpha_mod, alpha_mod = min(alpha[:, g] + .1, 1) (+.1 lets
        the mode stabilize). Arrays P0..mode_end and alpha are (N, G_genes).
        """
        ns = self.n_stimuli
        for g in range(ns, len(prev.cols)):
            c = prev.cols[g]
            is_gene = c >= ns
            r = c[is_gene] - ns
            if not len(r):
                continue
            alpha_mod = np.minimum(alpha[:, g - ns] + .1, 1)[:, None]
            prev.values[g][idx_next, is_gene] = find_next_prot(
                d1[r], P0[:, r], M0[:, r], M1[:, r], mode_init[:, r], mode_end[:, r],
                np.minimum(alpha[:, r] / alpha_mod, 1),
                s[..., r] if np.ndim(s) else s, delta_t * alpha_mod)

    def estimate_trajectories(self, y_prot, times, d1, N=100, kon_beta=None, s=None, prev_cols=None):
        """
        Estimate protein trajectories when d1, theta, and alpha are known.

        Parameters
        ----------
        kon_beta : array of shape (T*N, G_tot), optional
            Pre-computed burst frequencies. If None, uses ``self.kon_beta``.
        s : float or array of shape (G_genes,), optional
            Per-gene protein scale. Defaults to ``self.scale_proteins``.
            Must match the s used in my_otdistance when building y_prot; pass
            ``s`` to reproduce protein trajectories exactly.
        prev_cols : list of index arrays, optional
            Active regulators per target (see active_regulators). If given, the
            flow-matching states are also returned as a PrevProt, else None.
        """
        if kon_beta is None:
            kon_beta = self.kon_beta
        if s is None:
            s = self.scale_proteins
        ns = self.n_stimuli
        prot_modified = y_prot.copy()
        # Flow-matching states start from the input trajectories
        prot_modified_prev = None
        if prev_cols is not None:
            prot_modified_prev = PrevProt(prev_cols, [prot_modified[:, c] for c in prev_cols])

        for cnt, time in enumerate(times[:-1]):
            delta_t = times[cnt + 1] - time
            rows_prev = slice(N * cnt, N * (cnt + 1))
            rows_next = slice(N * (cnt + 1), N * (cnt + 2))
            kb_prev = np.ascontiguousarray(kon_beta[rows_prev, ns:], dtype=np.float64)
            kb_next = np.ascontiguousarray(kon_beta[rows_next, ns:], dtype=np.float64)

            # All N trajectories at once
            prot_modified[rows_next, ns:] = find_next_prot(
                np.asarray(d1, dtype=np.float64),
                np.ascontiguousarray(prot_modified[rows_prev, ns:], dtype=np.float64),
                kb_prev, kb_next, kb_prev, kb_next,
                np.ascontiguousarray(self.alpha[cnt, :N], dtype=np.float64),
                s, float(delta_t))

            if prot_modified_prev is not None and self.weight_prev > 0:
                self._fill_prev_prot(
                    prot_modified_prev, rows_next, self.alpha[cnt, :N], delta_t,
                    d1, prot_modified[rows_prev, ns:], kb_prev, kb_next, kb_prev, kb_next, s)

        return prot_modified, prot_modified_prev
    

    def select_cells_to_use(self):

        n_samples = len(np.unique(self.samples_data))
        t0 = np.min(self.times_data)
        N_t = np.sum(self.times_data == t0)
        cells_to_use = np.zeros_like(self.times_data, dtype=int)
        times = np.unique(self.times_data)

        for s in range(n_samples):
            # Trajectories of sample s at the first timepoint
            idx_first = (self.samples_data == s) & (self.times_data == t0)
            idx_first_indices = np.where(idx_first)[0]
            N_s = len(idx_first_indices)

            if N_s == 0:
                continue

            # Random subset of trajectories, cell-type proportional at the first timepoint
            chosen_idx = stratified_choice(
                idx_first_indices, self.batch_size_degradations,
                None if self.traj_cell_types is None else self.traj_cell_types[idx_first_indices])

            # Same trajectories at every timepoint
            for cnt, t in enumerate(times):
                idx_cnt = chosen_idx+N_t*cnt
                cells_to_use[idx_cnt] = 1

        return cells_to_use

    def _kon_ref_per_sample(self, y_prot, ks, inter, basal, samples_id=None, samples_data=None):
        """
        Compute kon_ref_vector respecting per-sample basal when basal is 3-D.

        When basal is 2-D (G, n_networks), falls back to a single kon_ref call.
        When basal is 3-D (n_samples, G, n_networks), loops over samples and
        assembles the result using samples_data (or self.samples_data) as the
        routing key so that each cell uses its own sample's basal.
        """
        if basal.ndim < 3 and ks.ndim < 3:
            return kon_ref_vector(y_prot, ks, inter, basal)
        sd = samples_data if samples_data is not None else self.samples_data
        if samples_id is None:
            samples_id = np.sort(np.unique(sd))
        out = np.zeros((y_prot.shape[0], y_prot.shape[1]))
        for s_idx, s in enumerate(samples_id):
            mask = (sd == s)
            if not np.any(mask):
                continue
            basal_s = basal[min(s_idx, basal.shape[0] - 1)] if basal.ndim == 3 else basal
            out[mask] = kon_ref_vector(y_prot[mask], ks_of(ks, s_idx), inter, basal_s)
        return out

    def _count_errors_per_sample(self, y_prot, kon_beta, proba_traj, ks, inter, basal,
                                  samples_id=None, samples_data=None):
        """
        Weighted-average count_errors respecting per-sample basal.
        When basal is 2-D, delegates to count_errors directly.

        samples_data : per-cell sample assignment; defaults to self.samples_data.
        """
        if basal.ndim < 3 and ks.ndim < 3:
            return count_errors(y_prot, kon_beta, proba_traj, ks, basal, inter,
                                loss=self.loss_norm,
                                compute_with_proba=self.compute_with_proba,
                                n_stimuli=self.n_stimuli)
        if samples_data is None:
            samples_data = self.samples_data
        if samples_id is None:
            samples_id = np.sort(np.unique(samples_data))
        total_err, total_cells = 0.0, 0
        for s_idx, s in enumerate(samples_id):
            mask = (samples_data == s)
            n_s = int(mask.sum())
            if n_s == 0:
                continue
            basal_s = basal[min(s_idx, basal.shape[0] - 1)] if basal.ndim == 3 else basal
            err_s = count_errors(y_prot[mask], kon_beta[mask], proba_traj[mask],
                                 ks_of(ks, s_idx), basal_s, inter,
                                 loss=self.loss_norm,
                                 compute_with_proba=self.compute_with_proba,
                                 n_stimuli=self.n_stimuli)
            total_err += err_s * n_s
            total_cells += n_s
        return total_err / total_cells if total_cells > 0 else 0.0

    def refine_network_degradations(self, verb=True, stimulus_schedule=None, test=False):
        """
        Refine network parameters and infer degradation rates for simulation.

        When ``test=True``, only runs the trajectory estimation step and recomputes
        ``kon_theta`` using the current (pre-loaded simul) network. No inference,
        MLP training, or parameter update is performed.
        """
        seed_everything(self.seed)

        times = np.sort(np.unique(self.times_data))
        N_tot = np.sum(self.times_data == times[0])

        if stimulus_schedule is not None or self._stim_schedule is None:
            self._stim_schedule = self._build_stimulus_schedule(times, stimulus_schedule)
        
        if self.simulate_full_with_harissa:
            self.scale_proteins = 1

        ns = self.n_stimuli
        # --- Adapt ref_network ---
        for g in range(ns, self.ref_network.shape[0]):
            l_max = len(np.unique(self.modes[:, g]))
            if l_max < 2: # If only one mode
                self.ref_network[g, :], self.ref_network[:, g] = 0, 0
            if l_max > len(np.unique(self.a[..., :-1, g])): # compute with proba = 0 si more modes than ks
                self.compute_with_proba = 0

        ks, _, k1 = self._mixture_terms()
        if self.simulate_full_with_harissa:
            ks, k1 = self._shared_mixture(ks), self._shared_mixture(k1)
        ksT = np.swapaxes(ks, -1, -2)  # (n_modes, G) or (S, n_modes, G) for the degradation inference
        # basal: (n_samples, G_tot, n_networks); inter: (G_tot, G_tot, n_networks)
        basal, inter = self.basal.copy(), self.inter.copy()

        if test:
            # In test mode: estimate protein trajectories along pre-inferred OT couplings
            # and recompute kon_theta using the simul network. No inference is run.
            if not self.simulate_full_with_harissa:
                y_prot, _ = self.estimate_trajectories(
                    self.prot, times, self.d[1, ns:], N=N_tot,
                    kon_beta=self.kon_beta, s=self.scale_proteins)
            else:
                y_prot, _ = self.estimate_trajectories(
                    self.prot, times, self.d[1, ns:], N=N_tot,
                    kon_beta=self.kon_beta_harissa, s=1)
            self.prot = y_prot
            kon_vector = self.kon_beta.copy()
            kon_vector[:, ns:] = self._kon_ref_per_sample(
                y_prot, ks, inter, basal, samples_data=self.samples_data)[:, ns:]
            self.kon_theta = kon_vector
            return

        basal_ref, inter_ref = self.basal.copy(), self.inter.copy()
        if self.inter_simul_ref is not None:
            inter_ref = self._normalize_theta(
                self.inter_simul_ref, self.inter.shape[0], self.n_networks, is_inter=True)
        samples_id = np.sort(np.unique(self.samples_data))
        # Same active regulators as in the inference_network call below
        _, prev_cols = active_regulators(self.ref_network, inter_ref, self.hard_forcing_ref)

        if self.simulate_full_with_harissa:
            y_prot, y_prot_prev = self.estimate_trajectories(self.prot, times, self.d[1, ns:], N=N_tot, kon_beta=self.kon_beta_harissa, s=1, prev_cols=prev_cols)
            error = self._count_errors_per_sample(y_prot, self.kon_beta, self.proba_traj, ks,
                                                inter, basal, samples_id=samples_id, samples_data=self.samples_data)
        else:
            y_prot, y_prot_prev = self.estimate_trajectories(self.prot, times, self.d[1, ns:], N=N_tot, kon_beta=self.kon_beta, s=self.scale_proteins, prev_cols=prev_cols)
            error = self._count_errors_per_sample(y_prot, self.kon_beta, self.proba_traj, ks,
                                                inter, basal, samples_id=samples_id, samples_data=self.samples_data)

        # inference_network returns (n_samples, G_tot, n_networks) for basal
        _final_call = 0 if (self.hard_forcing_ref or self.inter_simul_ref is not None) else 1

        def fit_on(sels):
            return inference_network_multi(
                sels, self.samples_data, self.kon_beta, self.proba_traj,
                y_prot, y_prot_prev, ks, n_stimuli=ns, proba=self.compute_with_proba,
                ref_network=self.ref_network, basal_init=basal_ref, inter_init=inter_ref,
                basal_ref=basal_ref, inter_ref=inter_ref,
                scale=self.scale_pen * 2, # # slightly stronger regularization for network
                weight_prev=self.weight_prev, loss=self.loss_norm, final=_final_call,
                samples_id=samples_id,
                constrain_basal_uniform=self.constrain_basal_uniform,
                hard_forcing_ref=self.hard_forcing_ref, ref_constraint_pct=self.ref_constraint_pct,
                seuil_zero_min_ref=self.seuil_zero_min_ref,
            )

        # Mean theta over fits on (time, sample, cell type)-stratified subsamples (single fit if one holds all)
        basal, inter, _, _ = self._fit_theta_averaged(
            fit_on, self.times_data, self.samples_data, self.traj_cell_types, self.n_network_fits,
            self._n_params_per_target(prev_cols, len(samples_id)))

        ### filter_edges
        if self.filter_network:
            # With hard_forcing_ref, absent reference edges may reach ±seuil_zero_min_ref: filter them out
            seuil_intensity = self.seuil_min_network_intensity
            if self.hard_forcing_ref:
                seuil_intensity = max(seuil_intensity, self.seuil_zero_min_ref)
            inter, _ = filter_network(len(times), N_tot, y_prot, ks, basal, inter, 
                                      samples_data=self.samples_data, 
                                      seuil_intensity=seuil_intensity, seuil_variations=self.seuil_min_network_variations,
                                      seed=task_seed(self.seed, 2))

        error_corrected = self._count_errors_per_sample(y_prot, self.kon_beta, self.proba_traj, ks,
                                                        inter, basal, samples_id=samples_id, samples_data=self.samples_data)
        if verb:
            print("[refine_network_degradations] ratio errors", error, error_corrected)

        # Pre-scale basal/inter to best fit kon_beta across all cells before ODE inference.
        scale_theta_pre = fit_scale_theta(
            y_prot, self.kon_beta, basal, inter,
            ksT * self.scale_proteins, ns, samples_data=self.samples_data,
        )
        basal *= scale_theta_pre       
        inter *= scale_theta_pre  

        print(np.mean(scale_theta_pre))

        self.prot = y_prot
        kon_vector = self.kon_beta.copy()
        kon_vector[:, ns:] = self._kon_ref_per_sample(y_prot, ks, inter, basal, samples_data=self.samples_data)[:, ns:]
        self.kon_theta = kon_vector

        # --- Train proliferation MLP along the paths of the recomputed trajectories ---
        if self.recompute_proliferations and self.R_opt is not None:
            if verb:
                print("[refine_network_degradations] Training ProliferationMLP on R_opt...")
            # Rate without the inference stimuli when their effects are given (added back in the simulations);
            # the MLP then takes no stimulus input
            R_target = self.R_opt if self.R_stim_offset is None else self.R_opt - self.R_stim_offset
            self.prolif_network = train_proliferation_mlp(
                self.prot, R_target, self.times_data, ns=ns, n_nodes=self.n_growth_nodes,
                with_stim=self.prolif_uses_stimulus and self.R_stim_offset is None, seed=self.seed, verb=verb,
            )
            if verb:
                print("[refine_network_degradations] ProliferationMLP training done.")

        ### Adapt degradation rates
        self.ratios = np.tile(self.d[0, :] / self.d[1, :], (len(times)-1, 1))
        self.d_t = np.tile(self.d, (len(times)-1, 1, 1))
        # When basal is 3-D (n_samples, G, n_networks) keep per-sample structure:
        # basal_t → 4-D (T-1, n_samples, G, n_networks)
        if basal.ndim == 3:
            basal_t = np.tile(basal, (len(times)-1, 1, 1, 1))  # (T-1, n_samples, G, n_networks)
        else:
            basal_t = np.tile(basal, (len(times)-1, 1, 1))  # (T-1, G, n_networks)
        inter_t = np.tile(inter, (len(times)-1, 1, 1, 1))

        if self.recompute_degradations:
            # ── Harissa: train per-gene MLP correction before degradation inference ──
            self.kon_mlp = None
            if self.simulate_full_with_harissa:
                print("[refine_network_degradations] Training kon correction MLP (Harissa branch)...")
                self.kon_mlp = train_kon_correction_mlp(
                    self.prot, self.kon_beta_harissa, self.kon_beta * self.scale_proteins, ns, seuil=self.seuil,
                    seed=task_seed(self.seed, 6)
                )

                prot_np    = self.prot if isinstance(self.prot, np.ndarray) else self.prot.cpu().numpy()
                kb_genes   = self.kon_beta[:, ns:] * self.scale_proteins
                kh_genes   = self.kon_beta_harissa[:, ns:]

                ratio_pred = self.kon_mlp(prot_np, kb_genes)          # (N, G_genes)
                pred       = kb_genes * ratio_pred
                residual   = np.linalg.norm(pred - kh_genes)
                residual_prior = np.linalg.norm(kb_genes - kh_genes)  # baseline g=1

                print(f"Fit residual norm       = {residual:.4f}")
                print(f"Prior residual (g=1)    = {residual_prior:.4f}")
                print(f"Gain                    = {residual_prior / residual:.2f}x")

            # Per-cell training ratios g = kon_harissa / kon_beta (genes only).
            # Passed to inference functions for the lambda_mlp interpolation mix;
            # None when simulate_full_with_harissa is off or no MLP was trained.
            g_obs_all = (
                self.kon_beta_harissa[:, ns:].clip(self.seuil, None) / (self.kon_beta[:, ns:].clip(self.seuil, None))
                if self.kon_mlp is not None else None
            )  # shape (N, G_genes), same row-indexing as self.prot / self.times_data

            # Subset used by infer_ratio_d0_d1_unitary; d1 inference uses all trajectories
            # with a random minibatch of batch_size_degradations per interval and step
            cells_to_use = self.select_cells_to_use()
            if not self.use_temporal_degradations:
                # Single job: torch may use every core
                import torch
                n_threads_prev = torch.get_num_threads()
                torch.set_num_threads(os.cpu_count() or 1)
                d1, scale_theta = inference_degradation_prot(
                            self.prot,
                            self.times_data,
                            basal,   # 3-D (n_samples, G, n_networks) — triggers per-sample ODE
                            inter, ksT * self.scale_proteins,
                            d=self.d[1], lr=1e-2,
                            batch_size=self.batch_size_degradations,
                            n_stimuli=ns, stim_schedule=self._stim_schedule,
                            scale_proteins = self.scale_proteins,
                            samples_data=self.samples_data,
                            strata=self.traj_cell_types,
                            kon_mlp=self.kon_mlp,
                            lambda_scale=self.lambda_scale,
                            lambda_deg=self.lambda_deg1,
                            lambda_mlp=self.lambda_mlp,
                            g_obs_train=g_obs_all)
                torch.set_num_threads(n_threads_prev)
                self.d_t[:, 1, :] = np.tile(d1, (len(times)-1, 1))
                basal *= scale_theta[None, :, None]   # (n_samples, G, n_networks) * (1, G, 1)
                inter *= scale_theta[None, :, None]
                if basal_t.ndim == 4:
                    basal_t *= scale_theta[None, None, :, None]  # (T-1, n_samples, G, n_networks)
                    inter_t *= scale_theta[None, None, :, None]
                else:
                    basal_t *= scale_theta[None, :, None]  # (T-1, G, n_networks)
                    inter_t *= scale_theta[None, None, :, None]

            n_intervals = len(times) - 1
            _sigma = None  # set below in the temporal branch if smoothing is active

            if self.use_temporal_degradations:
                # Split the cores between the interval jobs (torch threads per job)
                n_cpu = os.cpu_count() or 1
                n_jobs_deg = max(1, min(len(times) - 1, n_cpu))
                n_threads_deg = max(1, n_cpu // n_jobs_deg)

                def run_main_inference_degradation_prot(t):
                    import torch
                    n_threads_prev = torch.get_num_threads()
                    torch.set_num_threads(n_threads_deg)
                    idx = (self.times_data == times[t]) | (self.times_data == times[t+1])
                    try:
                        return run_degradation_interval(idx)
                    finally:
                        torch.set_num_threads(n_threads_prev)

                def run_degradation_interval(idx):
                    return inference_degradation_prot(
                        self.prot[idx], self.times_data[idx],
                        basal,   # 3-D
                        inter, ksT * self.scale_proteins,
                        d=self.d[1], lr=1e-2,
                        batch_size=self.batch_size_degradations,
                        n_stimuli=ns, stim_schedule=self._stim_schedule,
                        scale_proteins = self.scale_proteins,
                        samples_data=self.samples_data[idx],
                        strata=None if self.traj_cell_types is None else self.traj_cell_types[idx],
                        kon_mlp=self.kon_mlp,
                        lambda_scale=self.lambda_scale,
                        lambda_deg=self.lambda_deg1,
                        lambda_mlp=self.lambda_mlp,
                        g_obs_train=g_obs_all[idx] if g_obs_all is not None else None)

                results = Parallel(n_jobs=-1)(  # same pool size everywhere (no loky worker respawn)
                    delayed(seeded_call)(task_seed(self.seed, 3, t), run_main_inference_degradation_prot, t)
                    for t in range(0, len(times)-1)
                )

                # ── Phase 1: collect d1 and scale_theta ──────────────────────
                scale_theta = np.ones_like(self.d_t[:, 1, :])
                for cnt in range(0, len(times)-1):
                    self.d_t[cnt, 1, :], scale_theta[cnt] = results[cnt]

                # ── Phase 2: compute sigma; smooth d1 and scale_theta ─────────
                if n_intervals > 2 and self.smooth_degradations_sigma != 0:
                    ns_s = self.n_stimuli
                    strength = float(np.clip(self.smooth_degradations_strength, 0.0, 1.0))
                    if self.smooth_degradations_sigma is None:
                        t_idx = np.arange(n_intervals, dtype=float).reshape(-1, 1)
                        bw_grid = np.logspace(-1, np.log10(n_intervals / 2.0 + 0.1), 30)
                        cv = LeaveOneOut() if n_intervals <= 5 else 5
                        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                                            {'bandwidth': bw_grid}, cv=cv)
                        grid.fit(t_idx)
                        _sigma = grid.best_params_['bandwidth']
                        if verb:
                            print(f'[refine_network_degradations] temporal smoothing sigma (auto KDE): {_sigma:.3f} steps, strength={strength:.2f}')
                    else:
                        _sigma = float(self.smooth_degradations_sigma)
                        if verb:
                            print(f'[refine_network_degradations] temporal smoothing sigma (fixed): {_sigma:.3f} steps, strength={strength:.2f}')
                    for g in range(ns_s, self.d_t.shape[2]):
                        orig = self.d_t[:, 1, g].copy()
                        self.d_t[:, 1, g] = (1 - strength) * orig + strength * gaussian_filter1d(orig, sigma=_sigma)
                    self.d_t[:, 1, :] = np.clip(self.d_t[:, 1, :], 1e-6, None)
                    for g in range(ns_s, scale_theta.shape[1]):
                        orig = scale_theta[:, g].copy()
                        scale_theta[:, g] = (1 - strength) * orig + strength * gaussian_filter1d(orig, sigma=_sigma)
                    scale_theta = np.clip(scale_theta, 1e-6, None)

                # ── Phase 3: apply (smoothed) scale_theta to basal/inter ──────
                for cnt in range(0, len(times)-1):
                    if basal_t.ndim == 4:
                        basal_t[cnt] = basal * scale_theta[cnt, None, :, None]
                    else:
                        basal_t[cnt] = basal * scale_theta[cnt, :, None]
                    inter_t[cnt] = inter * scale_theta[cnt, None, :, None]

            # ── Infer d0/d1 = ε (mRNA/protein timescale ratio) ──────────────────
            #
            # Two branches depending on whether the Harissa MLP correction is
            # available:
            #
            #  A) Harissa branch (simulate_full_with_harissa=True,
            #     unitary_for_deg=False, kon_mlp trained):
            #     Uses the ratio g = kon_harissa / kon_beta as a proxy for the
            #     mRNA lag.  Calls infer_ratio_d0_d1_full (MLP-based LS).
            #
            #  B) kon_beta branch (simulate_full_with_harissa=False OR
            #     unitary_for_deg=True):
            #     Uses ODE residuals as a proxy for PDMP stochastic variance.
            #     Calls infer_ratio_d0_d1_unitary (variance-matching MoM).
            #
            # In both cases: self.ratios[cnt] = 1/ε = d0/d1, so that
            #   d0_sim = d1 * ratios = d1 * (d0/d1) = d0  ✓

            # prior_d1d0 = d1/d0 from the literature (initial self.ratios)
            prior_d1d0 = self.d[1, :] / self.d[0, :]   # shape (G,)

            if self.simulate_full_with_harissa:
                # ── Branch A: Harissa / MLP ───────────────────────────────────
                ratios_temporal, ratios_global = infer_ratio_d0_d1_full(
                    self.prot,
                    self.times_data,
                    basal_t,
                    inter_t,
                    ksT,   # ks    : (n_modes, G)
                    d_learned=self.d_t[:, 1, :],  # (T-1, G) — per-interval like epsilon branch
                    k1_vec=k1,
                    kon_mlp=self.kon_mlp,
                    prior_d1d0=prior_d1d0,
                    n_stimuli=ns,
                    stim_schedule=self._stim_schedule,
                    samples_data=self.samples_data,
                    lambda_deg=self.lambda_deg0,
                    lambda_mlp=self.lambda_mlp,
                    g_obs_train=g_obs_all if g_obs_all is not None else None,
                    verbose=verb,
                )  # ratios_temporal (T-1, G), ratios_global (G,) — all d1/d0

                if self.use_temporal_degradations:
                    for cnt in range(len(times) - 1):
                        self.ratios[cnt, :] = 1.0 / ratios_temporal[cnt]
                else:
                    self.ratios[:] = (1.0 / ratios_global)[None, :]

            else:
                # ── Branch B: ODE residuals / variance matching ───────────────
                ratios_temporal, ratios_global = infer_ratio_d0_d1_unitary(
                    self.prot[cells_to_use == 1],
                    self.times_data[cells_to_use == 1],
                    basal_t,
                    inter_t,
                    ksT * self.scale_proteins,
                    self.d_t[:, 1, :],              # (T-1, G) learned d1 per interval
                    k1 * self.scale_proteins,       # (G,) max burst rate × scale
                    n_stimuli=ns,
                    stim_schedule=self._stim_schedule,
                    samples_data=self.samples_data[cells_to_use == 1],
                    lambda_deg=self.lambda_deg0,
                    prior_eps=prior_d1d0,
                    scale=self.scale_proteins,
                    verbose=verb,
                )  # eps_temporal (T-1, G), eps_global (G,) — all d1/d0

                if self.use_temporal_degradations:
                    for cnt in range(len(times) - 1):
                        self.ratios[cnt, :] = 1.0 / ratios_temporal[cnt]
                else:
                    self.ratios[:] = (1.0 / ratios_global)[None, :]

            # ── Smooth ratios after d0/d1 computation ────────────────────────
            if self.use_temporal_degradations and n_intervals > 2 and self.smooth_degradations_sigma != 0:
                ns_s = self.n_stimuli
                for g in range(ns_s, self.ratios.shape[1]):
                    orig = self.ratios[:, g].copy()
                    self.ratios[:, g] = (1 - strength) * orig + strength * gaussian_filter1d(orig, sigma=_sigma)

        self.basal, self.inter = basal, inter
        self.basal_t, self.inter_t = basal_t, inter_t

        if verb:
            basal_mean = basal.mean(axis=0) if basal.ndim == 3 else basal
            print('[refine_network_degradations]  Static network unitary', [self.inter.transpose(1, 0, 2)[:, :, n] for n in range(self.n_networks)],
                    [basal_mean[:, n] for n in range(self.n_networks)])
            
        self.d[0, :ns], self.d_t[:, 0, :ns] = 1.0, 1.0
        self.d[1, :ns], self.d_t[:, 1, :ns] = 0.2, 0.2
        self.d[0, np.where(self.d[0, :] == self.d[1, :])], \
            self.d_t[:, 0, np.where(self.d_t[:, 0, :] == self.d_t[:, 1, :])] = self.d[1, np.where(self.d[0, :] == self.d[1, :])] + 1e-6,\
            self.d_t[:, 1, np.where(self.d_t[:, 0, :] == self.d_t[:, 1, :])] + 1e-6


    def simulate_trajectories_unitary(self, times, times_train, ks, N=100, verb=True, samples_data=None):
        """
        Simulate protein trajectories with unitary scale
        """
        ns = self.n_stimuli

        prot_modified = np.ones((N * len(times), self.prot.shape[1]))
        kon_vector = np.ones((N * len(times), self.prot.shape[1]))
        prot_modified[:N, :] = self.prot[:N, :]
        kon_vector[:N, :] = self.kon_beta[:N, :]
        kon_vector[:N, ns:] = self._kon_ref_per_sample(
            self.prot[:N, :], ks, self.inter, self.basal, samples_data=samples_data)[:, ns:]
        start_time=0
        # We want to capture automatically if we simulate from after last timepoint
        if times_train[-1] < times[1]: # times[0] = 0 by construction
            times = [0, times_train[-1]] + list(times[1:])
            l = len(times_train)
            prot_modified = np.ones((N * len(times), self.prot.shape[1]))
            kon_vector = np.ones((N * len(times), self.prot.shape[1]))
            prot_modified[:N, :] = self.prot[:N, :]
            kon_vector[:N, :] = self.kon_beta[:N, :]
            kon_vector[:N, ns:] = self._kon_ref_per_sample(
                self.prot[:N, :], ks, self.inter, self.basal, samples_data=samples_data)[:, ns:]
            ### Add last timepoints as starting timepoints for simulation
            prot_modified[N:2*N, :] = self.prot[N*(l-1):N*l, :]
            kon_vector[N:2*N, :] = self.kon_beta[N*(l-1):N*l, :]
            kon_vector[N:2*N, ns:] = self._kon_ref_per_sample(
                    self.prot[N*(l-1):N*l, :], ks, self.inter_t[-1], self.basal_t[-1], 
                    samples_data=samples_data)[:, ns:]
            start_time=1

        ### Actualize times_simulation
        times.sort()
        times_simulation = np.zeros(len(times)*N)
        for t in range(0, len(times)):
            times_simulation[t*N:(t+1)*N] = times[t]

        d_t_train = self.d_t.copy()
        ratios_train = self.ratios.copy()
        basal_t_train = self.basal_t.copy()
        inter_t_train = self.inter_t.copy()
        simulation_stochastic_orig = self.simulation_stochastic
        self.d_t = np.zeros((len(times)-1, 2, self.prot.shape[1]), dtype=float)
        self.ratios = np.zeros((len(times)-1, self.prot.shape[1]), dtype=float)
        G_sim = self.prot.shape[1]
        # Preserve 4-D structure when basal_t is per-sample
        if basal_t_train.ndim == 4:
            n_samp = basal_t_train.shape[1]
            basal_t = np.zeros((len(times)-1, n_samp, G_sim, self.n_networks), dtype=float)
        else:
            basal_t = np.zeros((len(times)-1, G_sim, self.n_networks), dtype=float)
        inter_t = np.zeros((len(times)-1, G_sim, G_sim, self.n_networks), dtype=float)
        for cnt, time in enumerate(times[:-1]):
            index = np.argmin(np.abs(times_train[:-1] - time))
            self.d_t[cnt]    = d_t_train[index]
            self.ratios[cnt] = ratios_train[index]
            basal_t[cnt]    = basal_t_train[index]
            inter_t[cnt]    = inter_t_train[index]
        self._add_perturbation_stimulus(basal_t, inter_t, times)

        ### Rescale kz (per sample with per-sample mixtures: rescale (S, G), kz (S, G, n_modes))
        _, _, k1 = self._mixture_terms()
        rescale = np.ones_like(k1)
        rescale[..., ns:] = k1[..., ns:]
        kz = ks * rescale[..., None]

        # Determine per-cell basal strategy once before the loop.
        # If basal_t is 4-D (per-sample) but no samples_data was provided, that means
        # the caller forgot to set model.samples_data → route everything to sample 0
        # and emit a warning rather than crashing with a cryptic shape error.
        if basal_t.ndim == 4 and samples_data is None:
            print("[simulate_trajectories_unitary] WARNING: basal_t is 4-D (per-sample) "
                  "but samples_data is None. "
                  "Did you forget to load data_samples.npy into model.samples_data? "
                  "Falling back to sample index 0 for all cells.")
            samples_data = np.zeros(N, dtype=int)
        _basal_is_per_sample = (samples_data is not None and basal_t.ndim == 4)
        if ks.ndim == 3 and samples_data is None:
            raise ValueError("Per-sample mixtures need samples_data (data_samples.npy) to simulate")
        # Sample index of each simulated cell (as for the per-sample basal)
        s_cells = np.asarray(samples_data)[:N].astype(int) if samples_data is not None else np.zeros(N, dtype=int)

        # Proliferation: R(P) integrated along each simulated path (states recorded at the quadrature nodes)
        _prolif_fn = None
        if self.simulate_with_proliferation and self.prolif_network is not None:
            _prolif_fn = self.prolif_network.predict  # (..., n_proteins), stimuli of the interval -> (...)
        # RATE effects of perturbation stimuli (added to R); without the MLP, R = 0 + effects
        _rates = list(getattr(self, 'rate_perturbation', None) or [])
        if _rates and _prolif_fn is None and verb:
            print("[simulate] Warning: RATE effects without the proliferation MLP: base net rate 0")
        _branching = _prolif_fn is not None or bool(_rates)
        self.log_population = np.zeros(len(times)) if _branching else None
        if _branching:
            u_growth, w_growth = quadrature(self.n_growth_nodes)
            # Resampling groups: cells of a sample only replace cells of the same sample
            groups = ([np.flatnonzero(np.asarray(samples_data)[:N] == s) for s in np.unique(np.asarray(samples_data)[:N])]
                      if samples_data is not None else [np.arange(N)])

        for cnt, time in enumerate(times[start_time:-1], start=start_time):
            delta_t = times[cnt + 1] - time
            # Recording times within the interval (end state last)
            t_rec = delta_t * u_growth[1:] if _branching else delta_t

            degradations = self.d_t[cnt].copy()
            if self.simulation_stochastic:
                degradations[0, :] = degradations[1, :] * self.ratios[cnt] # * (1 + np.sqrt(cnt))
                degradations[0, :] = np.clip(degradations[0, :], degradations[1, :] * self.min_ratio, degradations[1, :] * self.max_ratio)

            if self.finish_by_determinist:
                if time >= times[-2] or time > times[-1] * (len(times)-1) / len(times):
                    self.simulation_stochastic = 0

            start_index = N * cnt
            end_index = N * (cnt+1)

            if _basal_is_per_sample:
                # 4-D basal_t: per-sample basal directly available
                n_samp = basal_t.shape[1]
                basal_cells = np.array([
                    basal_t[cnt, min(int(samples_data[n]), n_samp - 1)]
                    for n in range(N)
                ])
            else:
                basal_cells = None

            # Stimulus values of each simulated cell over the interval (its sample's schedule)
            stim_cells = self._stim_schedule.per_cell(times[cnt + 1], s_cells) * self.scale_proteins

            def run_main_loop_for_cell(n, _basal_cells=basal_cells, _basal_t_cnt=basal_t[cnt],
                                       _stim_cells=stim_cells):
                _stim_vals = _stim_cells[n]
                basal_n = _basal_cells[n] if _basal_cells is not None else _basal_t_cnt
                s_n = s_cells[n]
                if self.simulation_stochastic:
                    return simulate_next_prot_pdmp(
                            degradations[1, :],
                            ks_of(kz, s_n) * degradations[0][:, None],
                            s1_of(rescale, s_n) * (degradations[0, :] / degradations[1, :]),
                            basal_n, inter_t[cnt], t_rec,
                            self.scale_proteins, P0=prot_modified[start_index + n, :],
                            ns=ns, stim_vals=_stim_vals,
                        )
                else:
                    return simulate_next_prot_ode(
                        degradations[1, :], ks_of(ks, s_n),
                        basal_n, inter_t[cnt], t_rec,
                        self.scale_proteins, P0=prot_modified[start_index + n, :],
                        ns=ns, stim_vals=_stim_vals
                    )

            results = Parallel(n_jobs=-1)(
            delayed(seeded_call)(task_seed(self.seed, 4, cnt, n), run_main_loop_for_cell, n) for n in range(0, N)
            )

            for idx, n in enumerate(range(0, N)):
                # result.p[-1] has shape (G_tot-1,): indices 1..G_tot-1 of the state
                # (index 0 = stim1 is excluded by the simulation).
                # For ns>1, indices 1..ns-1 are stim2..stimN — skip them.
                prot_modified[end_index + n, ns:] = results[idx].p[-1][ns - 1:]

            # --- Branching process resampling ---
            # log_weight_n = ∫ R(P_n(s)) ds over the simulated path (trapezoidal rule on the
            # recorded states), the same quadrature as in the MLP training. Multinomial
            # resampling within each sample keeps N cells (no Poisson extinction at R ≈ 0).
            if _branching:
                path = np.stack([prot_modified[start_index:start_index + N, ns:]]
                                + [np.array([results[n].p[q][ns - 1:] for n in range(N)])
                                   for q in range(len(t_rec))], axis=1)       # (N, Q, G)
                R_path = _prolif_fn(path, stim_cells[:, None, :]) if _prolif_fn is not None else np.zeros(path.shape[:2])
                for eff in _rates:
                    # delta x score along the path, scaled by the stimulus value over the interval
                    u = _sample_values(eff, times[cnt + 1], s_cells)  # (N,) value of each cell's sample
                    if not u.any():
                        continue
                    if eff['weights'] is None:
                        score = np.ones(path.shape[:2])
                    else:
                        score = np.clip(path * eff['scale'], 0, 1) @ eff['weights']
                    R_path = R_path + (u * eff['delta'])[:, None] * score
                srm = getattr(self, 'stimulus_rate_model', None)
                if srm is not None and _prolif_fn is not None:
                    # Inference stimuli: their part of the rate (removed when the MLP was trained), with the
                    # simulated schedule, from mRNA drawn at the recorded states (reference depth)
                    Qn = path.shape[1]
                    n_samp_k = basal_t.shape[1] if basal_t.ndim == 4 else (ks.shape[0] if ks.ndim == 3 else 1)
                    prot_nodes = np.hstack([np.repeat(stim_cells, Qn, axis=0), path.reshape(N * Qn, -1)])
                    kon_nodes = self._kon_ref_per_sample(prot_nodes, ks, inter_t[cnt], basal_t[cnt],
                                                         samples_id=np.arange(n_samp_k),
                                                         samples_data=np.repeat(s_cells, Qn))[:, ns:]
                    k1_nb = k1[..., ns:] if k1.ndim == 1 else k1[np.minimum(np.repeat(s_cells, Qn), len(k1) - 1), ns:]
                    c_nb = self.a[..., -1, :][..., ns:]
                    if c_nb.ndim == 2:
                        c_nb = c_nb[np.minimum(np.repeat(s_cells, Qn), len(c_nb) - 1)]
                    counts = np.random.negative_binomial(np.maximum(k1_nb * kon_nodes, 1e-8), c_nb / (c_nb + 1.0))
                    u_sim = self._stim_schedule.per_cell(times[cnt + 1], s_cells)[:, :srm.n_stimuli]  # (N, K)
                    R_path = R_path + (srm.effect(counts).reshape(N, Qn, -1) * u_sim[:, None, :]).sum(axis=-1)
                log_weights = (R_path * w_growth).sum(axis=1) * delta_t
                # Population size: mean growth factor of the cells over the interval
                self.log_population[cnt + 1] = self.log_population[cnt] + float(
                    np.log(np.mean(np.exp(log_weights - log_weights.max()))) + log_weights.max())
                P_end = prot_modified[end_index:end_index + N, ns:].copy()
                for grp in groups:
                    weights = np.exp(log_weights[grp] - log_weights[grp].max())
                    src = grp[np.random.choice(len(grp), len(grp), replace=True, p=weights / weights.sum())]
                    prot_modified[end_index + grp, ns:] = P_end[src]

            # Set stim values at step boundary from schedule (authoritative source).
            prot_modified[end_index:end_index + N, :ns] = stim_cells

            # Each cell with its own sample's basal and mixture (index-based, as basal_cells)
            n_samp = basal_t.shape[1] if basal_t.ndim == 4 else (ks.shape[0] if ks.ndim == 3 else 1)
            kon_vector[end_index:end_index+N, ns:] = self._kon_ref_per_sample(
                prot_modified[end_index:end_index+N, :], ks, inter_t[cnt], basal_t[cnt],
                samples_id=np.arange(n_samp), samples_data=s_cells)[:, ns:]

            if verb:
                print(f'timepoints {cnt} done', delta_t, time)

        self.simulation_stochastic = simulation_stochastic_orig
        return prot_modified, kon_vector, times_simulation


    def simulate_trajectories_full(self, times, times_train, ks, N=100, verb=True, samples_data=None):
        """
        Simulate protein AND mRNA trajectories using the Harissa bursty PDMP.

        Uses the inferred basal_t / inter_t (in absolute burst-rate units when
        simulate_full_with_harissa=True) as the Harissa network.
        Mimics simulate_trajectories_unitary but returns mRNA levels in addition
        to proteins.  Only supports ns == 1.

        Returns
        -------
        prot_modified : (N * len(times), G_tot)
        mrna_modified : (N * len(times), G_tot)
        kon_vector    : (N * len(times), G_tot)
        times_simulation : (N * len(times),)
        """
        ks = self._shared_mixture(ks)
        try:
            from harissa.model import NetworkModel as HarissaNetworkModel
        except ImportError:
            raise ImportError(
                "Harissa is required for simulate_full_with_harissa=True. "
                "Install it with: pip install harissa"
            )

        ns = self.n_stimuli
        if ns != 1:
            raise NotImplementedError(
                "simulate_full_with_harissa currently supports only ns=1. "
                f"Got ns={ns}."
            )

        G_tot   = self.prot.shape[1]
        G_genes = G_tot - ns   # number of actual genes (Harissa's G parameter)

        # Build Harissa model — kinetic params (a, d[0]) are constant over time;
        # basal, inter and d[1] will be updated at each interval inside the loop.
        h_model = HarissaNetworkModel(G_genes)
        h_model.a  = self._shared_mixture(self.a)

        # --- Mirror simulate_trajectories_unitary initialisation ---
        prot_modified = np.ones((N * len(times), G_tot))
        mrna_modified = np.ones((N * len(times), G_tot))
        kon_vector    = np.ones((N * len(times), G_tot))

        prot_modified[:N, :] = self.prot[:N, :]
        kon_vector[:N, :] = self.kon_beta[:N, :]
        mrna_modified[:N, :] = self.rna[:N, :]

        kon_vector[:N, ns:] = self._kon_ref_per_sample(
            self.prot[:N, :], ks, self.inter, self.basal, samples_data=samples_data)[:, ns:]

        start_time = 0
        if times_train[-1] < times[1]:
            times = [0, times_train[-1]] + list(times[1:])
            l = len(times_train)
            mrna_modified[N:2*N, :] = self.rna[N*(l-1):N*l, :]
            prot_modified[N:2*N, :]  = self.prot[N*(l-1):N*l, :]
            kon_vector[N:2*N, ns:] = self._kon_ref_per_sample(
                    self.prot[N*(l-1):N*l, :], ks, self.inter_t[-1], self.basal_t[-1],
                    samples_data=samples_data)[:, ns:]
            start_time = 1

        times.sort()
        times_simulation = np.zeros(len(times) * N)
        for t in range(len(times)):
            times_simulation[t*N:(t+1)*N] = times[t]

        # Interpolate d_t / basal_t / inter_t to simulation times
        d_t_train     = self.d_t.copy()
        ratios_train  = self.ratios.copy()
        basal_t_train = self.basal_t.copy()
        inter_t_train = self.inter_t.copy()
        self.d_t   = np.zeros((len(times)-1, 2, G_tot), dtype=float)
        self.ratios = np.zeros((len(times)-1, G_tot), dtype=float)
        if basal_t_train.ndim == 4:
            n_samp = basal_t_train.shape[1]
            basal_t = np.zeros((len(times)-1, n_samp, G_tot, self.n_networks), dtype=float)
        else:
            basal_t = np.zeros((len(times)-1, G_tot, self.n_networks), dtype=float)
        inter_t = np.zeros((len(times)-1, G_tot, G_tot, self.n_networks), dtype=float)
        for cnt, time in enumerate(times[:-1]):
            index = np.argmin(np.abs(times_train[:-1] - time))
            self.d_t[cnt]    = d_t_train[index]
            self.ratios[cnt]  = ratios_train[index]
            basal_t[cnt]     = basal_t_train[index]
            inter_t[cnt]     = inter_t_train[index]
        self._add_perturbation_stimulus(basal_t, inter_t, times)

        rescale = np.ones(G_tot)
        rescale[ns:] = self._shared_mixture(self._mixture_terms()[2])[ns:]

        # Simulation loop
        for cnt, time in enumerate(times[start_time:-1], start=start_time):
            delta_t    = times[cnt + 1] - time
            start_index = N * cnt
            end_index   = N * (cnt + 1)

            # Update Harissa network and degradation for this interval
            # Harissa takes a single (G_tot,) basal — average over samples and networks
            basal_h_cnt = basal_t[cnt].mean(axis=0).mean(axis=-1) if basal_t.ndim == 4 else basal_t[cnt].mean(axis=-1)
            inter_h_cnt = inter_t[cnt].mean(axis=-1)   # (G_tot, G_tot)
            h_model.d = self.d_t[cnt].copy()
            h_model.d[0, ns:] = h_model.d[1, ns:] * self.ratios[cnt, ns:] # * (1 + np.sqrt(cnt)) 
            h_model.d[0, :] = np.clip(h_model.d[0, :], h_model.d[1, :] * self.min_ratio, h_model.d[1, :] * self.max_ratio)
            h_model.basal  = basal_h_cnt
            h_model.inter  = inter_h_cnt

            cur_stim_vals = self._stim_schedule[times[cnt + 1]]

            def run_harissa_cell(n, _h=h_model, _dt=delta_t, _si=start_index, _stim_vals=cur_stim_vals):
                P0 = prot_modified[_si + n].copy()
                M0 = mrna_modified[_si + n].copy()
                P0[:ns] = _stim_vals * self.scale_proteins
                M0[:ns] = _stim_vals * self.scale_mrnas
                sim = _h.simulate(np.array([_dt]), M0=M0, P0=P0, burnin=None)
                # sim.p/m already exclude index 0 (stimulus), shape (1, G_genes)
                return sim.p[-1, :], sim.m[-1, :]

            results = Parallel(n_jobs=-1)(
                delayed(seeded_call)(task_seed(self.seed, 5, cnt, n), run_harissa_cell, n) for n in range(N)
            )
            for n, (p_end, m_end) in enumerate(results):
                prot_modified[end_index + n, ns:] = p_end
                mrna_modified[end_index + n, ns:] = np.random.poisson(m_end)

            # Stim dims: set from schedule
            prot_modified[end_index:end_index+N, :ns] = cur_stim_vals * self.scale_proteins
            mrna_modified[end_index:end_index+N, :ns] = cur_stim_vals * self.scale_mrnas

            basal_t_cnt_2d = basal_t[cnt].mean(axis=0) if basal_t.ndim == 4 else basal_t[cnt]
            kon_vector[end_index:end_index+N, ns:] = kon_ref_vector(
                prot_modified[end_index:end_index+N, :], ks, inter_t[cnt], basal_t_cnt_2d)[:, ns:]

            if verb:
                print(f'[simulate_full] timepoint {cnt} done  delta_t={delta_t}')

        return prot_modified, mrna_modified, kon_vector, times_simulation


    def simulate_network(self, times, verb=True, stimulus_schedule=None):
        """
        Simulate the protein trajectories using the final inferred network.
        """
        seed_everything(self.seed)

        if self.simulate_full_with_harissa:
            self.scale_proteins = 1

        times.sort()
        times_train = np.sort(np.unique(self.times_data))
        if stimulus_schedule is not None or self._stim_schedule is None:
            # Pass times_ref=times_train so the schedule is step-function interpolated
            # when simulation times differ from training times (e.g. times_simulation.txt).
            self._stim_schedule = self._build_stimulus_schedule(
                np.sort(np.unique(times)), stimulus_schedule, times_ref=times_train)
        N = np.sum(self.times_data == times_train[0])
        ks, _, _ = self._mixture_terms()

        samples_data_sim = (self.samples_data[:N]
                            if (self.samples_data is not None
                                and ((self.basal is not None and self.basal.ndim == 3) or self.a.ndim == 3))
                            else None)

        # Harissa PDMP always forces stimulus=1; only use it when the schedule is the
        # default (ns==1, stimulus active at every non-initial timepoint).
        t_min = min(self._stim_schedule.keys())
        _stim_is_default = (
            not self._stim_schedule.has_overrides()
            and self.n_stimuli == 1
            and all(
                np.all(np.asarray(v) == 1.0)
                for t, v in self._stim_schedule.items() if t > t_min
            )
        )
        if self.simulate_full_with_harissa and _stim_is_default:
            y_prot, y_mrna, kon_vector, times_simul = self.simulate_trajectories_full(
                times, times_train, ks, N=N, verb=verb, samples_data=samples_data_sim)
            self.mrna_simul = y_mrna
        else:
            if self.simulate_full_with_harissa and not _stim_is_default:
                print("[simulate_network] Harissa disabled: non-default stimulus schedule detected, using unitary simulation")
            y_prot, kon_vector, times_simul = self.simulate_trajectories_unitary(
                times, times_train, ks, N=N, verb=verb, samples_data=samples_data_sim)

        self.prot = y_prot
        self.kon_theta = kon_vector
        self.times_simul = times_simul



    def fit_mixture_test(self, data_rna, ks, c, verb=False, depth=None):
        """Classify test cells into mixture modes using fixed kinetic parameters.

        Sets self.modes, self.proba, self.proba_init, and self.pi_init so that
        update_modes in loop_trajectories works on test data without re-fitting kz/c.
        """
        ns = self.n_stimuli
        N_cells, G_tot = data_rna.shape
        vect_t = data_rna[:, 0]
        times = np.sort(np.unique(vect_t))

        # Preserve training priors before overwriting self.pi_init at the end
        training_pi_init = self.pi_init         # list[dict{t: array(ng,)}] or None
        training_pi_zinb = self.pi_zinb        # array(G_tot - ns,) of pi_zero, or None

        frequency_modes_smooth = np.ones_like(data_rna, dtype=float)
        frequency_proba = np.ones((N_cells, G_tot, self.n_networks + 1), dtype=float)
        frequency_proba_init = np.zeros((N_cells, G_tot, self.n_networks + 1), dtype=float)
        pi_init_test = []

        for g in range(ns, G_tot):
            g_idx = g - ns
            ng = np.argmax(ks[1:, g]) + 2

            # ZINB: use training pi_zero for this gene if available and positive
            pi_zero_g = None
            zi_flag = None
            if training_pi_zinb is not None and g_idx < len(training_pi_zinb):
                pzg = float(training_pi_zinb[g_idx])
                if pzg > 0:
                    pi_zero_g = pzg
                    zi_flag = True

            # Compute responsibilities per timepoint using training pi_init as prior
            proba = np.zeros((N_cells, ng))
            for t_i in times:
                idx_t = (vect_t == t_i)
                if not np.any(idx_t):
                    continue
                pi_prior = None
                if training_pi_init is not None and g_idx < len(training_pi_init):
                    raw = training_pi_init[g_idx].get(t_i, None)
                    if raw is not None:
                        arr = np.asarray(raw, dtype=float)[:ng]
                        s = arr.sum()
                        pi_prior = arr / (s + EPS) if s > 0 else np.ones(ng) / ng
                proba[idx_t], _ = predict_resp(
                    data_rna[idx_t, g], ks[:ng, g], c[g],
                    pi=pi_prior, pi_zero=pi_zero_g, zi=zi_flag,
                    s=None if depth is None else depth[idx_t]
                )
            tmp = proba
            if self.transform_proba:
                tmp = np.exp(self.transform_proba * ((len(ks)-1))*np.log(G_tot)*(proba - 1/len(ks))) # self.transform_proba is the typical size of parameters that are expected, np.log(G) the number of regulators), and the difference to the mean max proba scales the protein level
                tmp /= (1 + tmp)
                tmp /= np.sum(tmp, 1).reshape(N_cells, 1)
                for cell in range(N_cells):
                    if np.max(proba[cell]) > np.max(tmp[cell]):
                        tmp[cell, :] = proba[cell, :]
                proba[:, :] = tmp[:, :]

            if self.update_modes or self.loss_norm == 'CE':
                # Initial basins: argmax of the mixture posteriors, as for the training set
                tmp = np.zeros_like(proba)
                tmp[np.arange(proba.shape[0]), np.argmax(proba, axis=1)] = 1

            frequency_proba[:, g, :ng] = tmp
            frequency_proba[:, g, ng:] = 0
            frequency_proba_init[:, g, :ng] = proba
            frequency_proba_init[:, g, ng:] = 0
            frequency_modes_smooth[:, g] = np.sum(ks[:ng, g] * tmp, axis=1)
            if verb:
                print('[infer_test]', f'Gene {g} calibrated...', ks[:ng, g], c[g])

            # Per-timepoint mode proportions used by update_modes in loop_trajectories
            pi_g = {}
            for t_i in times:
                idx_t = (vect_t == t_i)
                if np.any(idx_t):
                    pi_g_t = np.mean(proba[idx_t], axis=0)
                    pi_g_t = pi_g_t / (np.sum(pi_g_t) + 1e-16)
                else:
                    pi_g_t = np.ones(ng) / ng
                pi_g[t_i] = pi_g_t
            pi_init_test.append(pi_g)

        scale_max = np.max(self.a[:-1, :], axis=0)
        frequency_modes_smooth /= np.maximum(scale_max, EPS)  # all-zero modes would give 0/0
        self.pi_zinb = training_pi_zinb   # keep training ZINB zero-inflation values
        self.modes = frequency_modes_smooth
        self.proba = frequency_proba
        self.proba_init = frequency_proba_init
        self.pi_init = pi_init_test



    def _fit_mixture_test_per_sample(self, data_rna, vect_samples_id, samples_id, depth=None):
        """fit_mixture_test with per-sample mixtures (a 3-D): each sample's cells with its own parameters."""
        a3, pz3, pi_train = self.a, self.pi_zinb, self.pi_init
        N, G_tot = data_rna.shape
        M = a3.shape[1] - 1
        modes, proba, proba_init = np.zeros((N, G_tot)), np.zeros((N, G_tot, M)), np.zeros((N, G_tot, M))
        vect_t = data_rna[:, 0]
        times = np.sort(np.unique(vect_t))
        pi_sum = None
        for s_idx, sid in enumerate(samples_id):
            m = vect_samples_id == sid
            self.a, self.pi_zinb, self.pi_init = a3[min(s_idx, len(a3) - 1)], pz3[min(s_idx, len(pz3) - 1)], pi_train
            self.fit_mixture_test(data_rna[m], self.a[:-1], self.a[-1], depth=None if depth is None else depth[m])
            modes[m], proba[m], proba_init[m] = self.modes, self.proba, self.proba_init
            # Mode proportions per time, weighted by the cells of each sample
            w = {t: np.sum(vect_t[m] == t) for t in times}
            if pi_sum is None:
                pi_sum = [{t: np.zeros(M) for t in times} for _ in self.pi_init]
            for g, pi_g in enumerate(self.pi_init):
                for t in times:
                    if t in pi_g:
                        p = np.asarray(pi_g[t], dtype=float)
                        pi_sum[g][t][:len(p)] += w[t] * p
        self.a, self.pi_zinb = a3, pz3
        self.modes, self.proba, self.proba_init = modes, proba, proba_init
        self.pi_init = [{t: p / (p.sum() + EPS) for t, p in pi_g.items()} for pi_g in pi_sum]

    def infer_test(self, data, vect_samples_id=None, verb=True, stimulus_schedule=None,
                   basal_ref=None, transition_rates=None, time_key='time', n_iter_offset=None):
        """
        Run inference pipeline on test data with the network fixed: basins of the test cells from
        the mixture, then trajectory loop (OT + basin updates combining EMD and network) continuing
        the training schedule (n_iter_offset: last training iteration, see loop_trajectories).

        basal_ref : (n_samples, G_tot, n_networks) array or None
            Per-sample KO/OV prior (±100 entries) used to build kov_cell_mask.
        transition_rates : DataFrame or array or None
            Cell-type transition rate matrix for OT cost adjustment.
        """
        seed_everything(self.seed)
        data_rna = self._parse_input(data, time_key)
        if stimulus_schedule is not None or self._stim_schedule is None:
            self._stim_schedule = self._build_stimulus_schedule(
                np.sort(np.unique(data_rna[:, 0])), stimulus_schedule)
        N_cells, G_tot = data_rna.shape
        ns = self.n_stimuli
        vect_t = data_rna[:, 0]
        try:
            import anndata
            if isinstance(data, anndata.AnnData) and vect_samples_id is None and 'dataset_id' in data.obs:
                vect_samples_id = data.obs['dataset_id'].values
        except ImportError:
            pass
        if vect_samples_id is None:
            vect_samples_id = np.zeros_like(vect_t)

        self._load_ot_constraints(data, transition_rates)

        times = np.sort(np.unique(vect_t))
        samples_id = np.sort(np.unique(vect_samples_id))

        if self.a.ndim == 3:
            self._fit_mixture_test_per_sample(data_rna, vect_samples_id, samples_id, depth=self._depth_factors(data))
        else:
            self.fit_mixture_test(data_rna, self.a[:-1], self.a[-1], depth=self._depth_factors(data))
        # Trajectories on counts at the reference depth (the basins above used the raw counts and s)
        if self._depth_factors(data) is not None:
            data_rna = self._parse_input(data, time_key, scale_depth=True)

        print('[infer_test] Mean proba = ', np.mean(np.max(self.proba[:, ns:, :], axis=-1)))

        ks, s1, _ = self._mixture_terms()
        # Mode amplitudes of each test cell's sample (per-sample mixtures)
        ks_cells = ks[np.searchsorted(samples_id, vect_samples_id)] if ks.ndim == 3 else None
        ks_max = ks.max(axis=0) if ks.ndim == 3 else ks

        # --- Build per-cell KO/OV mask and apply initial mode forcing ---
        kov_cell_mask = None
        if basal_ref is not None:
            br = np.asarray(basal_ref, dtype=float)
            if br.ndim == 3 and np.any(np.abs(br) > 50):
                cm = np.zeros((len(vect_t), G_tot), dtype=np.int8)
                for s_idx, s in enumerate(samples_id):
                    cell_idx = np.where(vect_samples_id == s)[0]
                    ko_genes = np.where(br[s_idx, :, 0] < -50)[0]
                    ov_genes = np.where(br[s_idx, :, 0] >  50)[0]
                    if len(cell_idx) and len(ko_genes):
                        cm[np.ix_(cell_idx, ko_genes)] = -1
                    if len(cell_idx) and len(ov_genes):
                        cm[np.ix_(cell_idx, ov_genes)] =  1
                if np.any(cm != 0):
                    kov_cell_mask = cm
                    # Force initial modes from fit_mixture_test
                    for g in range(ns, G_tot):
                        l_max = 1 + int(np.argmax(ks_max[g, :]))
                        ko_cells = kov_cell_mask[:, g] < 0
                        ov_cells = kov_cell_mask[:, g] > 0
                        if np.any(ko_cells):
                            self.proba[ko_cells, g, :] = 0
                            self.proba[ko_cells, g, 0] = 1
                            self.modes[ko_cells, g] = ks_cells[ko_cells, g, 0] if ks_cells is not None else ks[g, 0]
                        if np.any(ov_cells):
                            self.proba[ov_cells, g, :] = 0
                            self.proba[ov_cells, g, l_max - 1] = 1
                            self.modes[ov_cells, g] = ks_cells[ov_cells, g, l_max - 1] if ks_cells is not None else ks[g, l_max - 1]

        nb_cells = np.zeros((len(samples_id), len(times)), dtype=int)
        for s, sid in enumerate(samples_id):
            for t, time in enumerate(times):
                nb_cells[s, t] = np.sum((vect_t[vect_samples_id == sid] == time))

        if verb:
            print("[infer_test] Cell counts per sample/timepoint and genes:\n", nb_cells, G_tot)

        # --- Define number of cells used for inference ---
        N_samples = []
        for s in range(len(samples_id)):
            n = int(np.max(nb_cells[s]))
            q, r = divmod(n, self.batch_size_traj)
            if q == 0: N_samples.append(n)
            else: N_samples.append(min(self.batch_size_traj + 1+int(r/q), n))

        N_full = [int(np.max(nb_cells[s])) for s in range(len(samples_id))]

        if verb:
            print("[infer_test] Number of simulated cells per sample:", N_samples)
            print("[infer_test] Number of total cells per sample:", N_full)

        # --- Choose initial cells per sample ---
        init_cells_full = [
            minimal_repetition_choice(nb_cells[s, 0], N_full[s], labels=self._t0_cell_types(vect_t, vect_samples_id, sample))
            for s, sample in enumerate(samples_id)
        ]

        # --- Infer trajectories on full simulations given theta ---
        self.loop_trajectories(
            data_rna=data_rna,
            vect_t=vect_t,
            vect_samples_id=vect_samples_id,
            times=times,
            samples_id=samples_id,
            ks=ks,
            s1=s1,
            init_cells_full=init_cells_full,
            N_full=N_full,
            N_samples=N_samples,
            G_tot=G_tot,
            min_n_loops=self.min_n_loops,
            count_max=self.count_max,
            intensity_prior=0,
            basal_init=None,
            inter_init=None,
            verb=verb,
            compute_theta=False,
            initialize_alpha=True,
            kov_cell_mask=kov_cell_mask,
            n_iter_offset=n_iter_offset if n_iter_offset is not None else self.min_n_loops,
        )


    def fit(self, data_rna, intensity_prior=100, refilter=5.0, max_iter_kinetics=100, verb=True):

        self.fit_mixture(data_rna, min_components=2, max_components=2, refilter=refilter, max_iter_kinetics=max_iter_kinetics)
        self.fit_network(data_rna, intensity_prior=intensity_prior, verb=verb)
        # self.refine_network_degradations()