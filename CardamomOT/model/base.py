
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
from scipy.linalg import expm
from ..config import resolve_cell_type_obs, CELL_TYPE_OBS_KEYS
from ..inference.trajectory import ks_of, s1_of, s1_rows, inter_of
from ..inference.mixture import _compute_nu_with_temporal_constraint
from ..inference import (inference_network_multi, active_regulators, PrevProt, signed_floor, filter_network,
                        minimal_repetition_choice, find_next_prot, my_otdistance, count_errors,
                        kon_ref_vector, inference_alpha, inference_alpha_1thread,
                        NegativeBinomialMixtureEM, predict_resp,
                        simulate_next_prot_ode, simulate_next_prot_pdmp,
                        infer_ratio_d0_d1_unitary, inference_degradation_prot,
                        train_proliferation_mlp, quadrature, fit_scale_theta,
                        seed_everything, seeded_call, task_seed,
                        stratified_order, stratified_choice, grouped_partition)

np.set_printoptions(precision=3, suppress=True)
EPS=1e-16

# Parameters that moved: a value for all samples would claim an information that is not known for all
# Former names of parameters (workbooks written before the renaming are still read)
RENAMED_PARAMETERS = {'cell_depth_for_embeddings': 'cell_depth_for_representation'}

REMOVED_PARAMETERS = {
}


def _rows(x, idx):
    """Rows idx of a per-row array (2-D), or x itself when shared (1-D)."""
    return x[idx] if np.ndim(x) == 2 else x


def observed_times(vect_t, vect_samples_id, times, samples_id):
    """(S, T) bool: sample s has cells at times[t] (samples may have different timepoints)."""
    return np.array([[np.any((vect_t == t) & (vect_samples_id == s)) for t in times] for s in samples_id])


def valid_rows(observed, N_full):
    """(T * N_tot,) bool: trajectory state of a sample at a time where it has cells (else a virtual state)."""
    S, T = observed.shape
    return np.concatenate([np.repeat(observed[:, t], N_full) for t in range(T)])


def trajectory_pairs(valid, T):
    """
    [(a, b, slots)]: trajectory slots whose consecutive observed timepoints are times[a] -> times[b]
    (b = a + 1 when every sample is observed at every time); valid: (T * N,) bool.
    """
    V = np.asarray(valid, dtype=bool).reshape(T, -1)
    groups = {}
    for n in range(V.shape[1]):
        obs = np.flatnonzero(V[:, n])
        for a, b in zip(obs[:-1], obs[1:]):
            groups.setdefault((int(a), int(b)), []).append(n)
    return [(a, b, np.array(sl)) for (a, b), sl in sorted(groups.items())]


def prot_along(d1, P0, mode_init, mode_end, alpha, s, delta_t, tau):
    """Protein at time tau in [0, delta_t] of the flow of find_next_prot (switch of the modes at alpha * delta_t)."""
    t_sw = alpha * delta_t
    p_sw = mode_init * s + (P0 - mode_init * s) * np.exp(-d1 * np.minimum(tau, t_sw))
    return np.where(tau <= t_sw, p_sw, mode_end * s + (p_sw - mode_end * s) * np.exp(-d1 * np.maximum(tau - t_sw, 0)))


def fill_virtual(arrays, valid, T, inside=True):
    """States of the virtual rows copied from the previous observed row of their slot (else the next one);
    inside=False: only the virtual rows before the first or after the last observed time of their slot."""
    V = np.asarray(valid, dtype=bool).reshape(T, -1)
    if V.all():
        return
    N = V.shape[1]
    for n in np.flatnonzero(~V.all(axis=0)):
        obs = np.flatnonzero(V[:, n])
        if not len(obs):
            continue
        for t in np.flatnonzero(~V[:, n]):
            if not inside and obs[0] < t < obs[-1]:
                continue
            prev = obs[obs < t]
            src = prev[-1] if len(prev) else obs[0]
            for arr in arrays:
                arr[t * N + n] = arr[src * N + n]


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
        self.traj_valid = None      # (T * N,) bool: trajectory state at a time observed for its sample (None = all)
        self.couplings = None       # final soft couplings of the trajectories (list of sparse blocks, growth pass)
        self._sample_names = None   # sorted dataset_id of the run (index of a sample -> its label)
        self.sample_conditions = None  # (n_samples,) network condition index of each sample (None: one common network)
        self.condition_names = None    # labels of the network conditions (obs['network_condition'])

        ### Pipeline (run.sh / cardamomot pipeline): data and steps, fixed per project in Data/CardamomOT_inputs.xlsx
        self.split = 'train'                     # 'train': train/test split of the cells (train_rate per sample and time); 'full': all cells
        self.train_rate = 0.7                    # share of the cells of each (sample, time) in the train split (at least 100); the test keeps at most as many
        self.select_genes = False                # select_genes (on the train cells of split_dataset): gene selection (queries, entropy genes, global network, Steiner tree); False = all genes kept
        self.build_prior_network = False         # literature prior cardamomOT/ref_network.csv: built by the gene selection if select_genes, literature_selection and prior_network_pen = 0, else by build_reference_network
        self.estimate_proliferation_rates = True  # get_proliferation_rates: obs['proliferation_net_rate'] from gene signatures, anchored to Data/proliferation_rates
        self.run_classical_OT = True             # run_classical_OT: Waddington-OT-style analysis (every gene, train cells) before CardamomOT, 2 report pages
        self.classical_ot_max_cells = 5000       # run_classical_OT: at most this many cells per (sample, time) in the couplings (others: kernel regression)
        self.run_test = True                    # infer_test + check_test_to_train on the held-out cells (needs split = 'train')
        self.simulate_perturbations = True       # simulate_network_KOV + check_KOV_to_sim (perturbation_simulation sheet)
        self.species = 'auto'                    # 'auto' (from gene names), 'human' or 'mouse': degradation rates, proliferation signatures, literature prior
        self.senescence_gating = True            # get_proliferation_rates: the senescence signature gates the proliferation score
        self.overwrite_degradation_rates = False  # get_degradation_rates: replace the d0/d1 already stored in the AnnData files
        self.report_net_index = 0                # report: network shown when n_networks > 1
        self.report_normalize = False            # report UMAPs: counts normalised per cell
        self.report_log1p = True                 # report UMAPs: log1p of the counts
        self.report_n_umap = 4000                # report: maximal number of cells per stage in the UMAPs (0 = all)
        self.cell_depth_for_representation = True    # report, cell-type classifiers: mRNA / depth factor (if computed), model draws at the reference depth; False = raw counts, draws at the cells' depth
        self.embedding_method_visualization = 'umap'           # report: 2-D embeddings with 'umap', 'pca' or 'phate'
        self.classifier_method = 'logistic'  # cell types of the model outputs (report, notebooks): 'random_forest' or 'logistic'

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
        self.mrna_driven_proteins = False # True: degradations on the mRNA-driven trajectories of fit_network (not bounded to the modes); False: proteins re-estimated on the modes and network refitted. Forced to True with simulate_full_with_harissa
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
        self.network_condition_pen = 1.0 # >= 0, with >= 2 values of obs['network_condition']: fused L1 penalty on the deviations of each condition's network from the shared one, relative to the sparsity penalty (large = common network, 0 = independent networks)
        self.hard_forcing_ref = False # if True, constrain all network params to ±ref_constraint_pct around inter_ref
        self.ref_constraint_pct = 0.01 # fractional tolerance around inter_ref values for bounds (used when hard_forcing_ref=True)
        self.seuil_zero_min_ref = 5e-2 # reference values (inter_ref) with |v| <= this are read as absent edges; also min |theta| of sign-forced edges
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
        self.smooth_degradations_sigma = None  # None=auto KDE+CV, 0=off, float>0=fixed sigma (in time-step units); interior intervals only
        self.smooth_degradations_strength = 0.5  # blend weight in [0,1]: 0=no smoothing, 1=full smoothing

        ## Simulations
        self.simulation_stochastic = True # 1 if we simulate Bursty-like proteins, 0 if deterministic limit for proteins
        self.finish_by_determinist = False # 1 if we simulate with deterministic limit for the last timepoint
        self.min_ratio = .05
        self.max_ratio = 50
        self.simulate_full_with_harissa = False  # use Harissa PDMP to jointly simulate proteins+mRNAs (simulations only)

        ## Gene selection (select_genes.py, select_genes = True): terminals + global network + directed Steiner tree
        self.num_max_genes = 100         # budget: number of selected genes (stimuli excluded)
        self.n_query_genes = 30          # at most this many genes of genes_queries (gene_lists sheet) (round robin over time/cell-type DE groups)
        self.n_driver_genes = 30         # fate drivers of run_classical_OT (classical_OT/fate_drivers.csv) required in the selection (0 = none); n_query + n_driver + n_entropy < num_max_genes
        self.entropy_preselection = True  # queries, fate drivers and network genes (Steiner, closure) kept only among the entropy candidates (n_top_entropy) and the perturbed genes
        self.n_entropy_genes = 10        # at least this many entropy genes (Gandrillon KD & MDE), same round robin; n_query + n_entropy < num_max_genes
        self.n_top_entropy = 500         # top genes per transition for KD and for MDE (Gandrillon's TOP_N)
        self.n_cells_entropy = 1000      # cells per timepoint for the BUB entropy (its matrices are (N+1)^2)
        self.n_hvg_selection = 5000      # highly variable genes forming the network universe (terminals always kept)
        self.network_method = 'otvelo_granger'  # global network: 'otvelo_granger', 'otvelo_corr', 'wot_granger' (through the run_classical_OT couplings), or <project>/network_methods/<name>.py
        self.network_method_params = {}  # parameters of the network method (otvelo: n_cells, n_pcs, eps, alpha; granger: + k_candidates, en_alpha ('auto' = min(1, sqrt(50 / G))), l1_ratio, scale, stim_weight)
        self.k_in_steiner = 20           # strongest incoming edges kept per gene in the Steiner graph
        self.k_stim_steiner = 20         # direct targets kept per stimulus in the Steiner graph
        self.min_edge_prob = 0.05        # edges with probability (1 - FDR against the permuted-data network) below are dropped
        self.edge_prior = 0.9            # each Steiner edge costs -log(prob * edge_prior): favours short paths among equally probable ones
        self.null_network = 'hybrid'     # edge probabilities vs permuted data: 'hybrid' (gene edges within (sample, time), stimulus all cells), 'within_time', 'all_cells'
        self.sample_network_combination = 'auto'  # several samples, gene selection: 'consensus' (edge probabilities averaged, shared network), 'any' (probabilistic OR + closure balanced over the samples, independent condition networks); 'auto' = 'any' with >= 2 network conditions and network_condition_pen = 0
        self.selection_edge_prob = 0.6   # gene_selection_report: is_regulated_by / regulates list the edges inside the selection with probability >= this (and literature-feasible)
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
        self.simulate_with_proliferation = False # apply branching process in simulate_trajectories_unitary
        self.protein_dilution = True             # proteins diluted at the birth rate b of each cell: dP/dt = d1 u - (d1 + b) P (obs['proliferation_birth_rate']; independent of simulate_with_proliferation)
        self.prolif_uses_stimulus = True         # inference stimuli are inputs of the ProliferationMLP, R(u, P) (no effect if constant over the intervals)
        self.prolif_network = None               # ProliferationMLP trained in refine_network_degradations (if simulate_with_proliferation), not a parameter
        self.R_opt = None                        # net growth rate of each trajectory state over the next interval (NaN at last time), from the final growth OT pass
        self.R_stim_offset = None                # part of R_opt due to the inference stimuli (RATEk of perturbation_inference), removed before training the proliferation MLP
        self.stimulus_rate_model = None          # StimulusRateModel: stimulus part of the net rate from mRNA, added back in the branching simulations
        self.growth_reg_source = 2.0             # source-marginal relaxation (x log G) of the growth OT pass: small = data-driven but noisy, large = prior kept
        self.n_growth_iter = 1                   # WOT-style growth iterations (source weights <- row marginals); more iterations amplify the noise
        self.n_growth_nodes = 5                  # quadrature nodes per interval to integrate R along paths (MLP training and branching simulation)
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

    def set_network_conditions(self, vect_samples_id, samples_id, vect_conditions=None, verb=True):
        """
        Network condition of each sample (obs['network_condition'] per cell; several dataset_id may share
        one). With >= 2 conditions, each has its own network (inter per sample, see network_condition_pen);
        otherwise sample_conditions = None (one common network).
        """
        self.sample_conditions, self.condition_names = None, None
        if vect_conditions is None:
            return
        vect_conditions = np.asarray(vect_conditions).astype(str)
        labels = []
        for sample in samples_id:
            c = np.unique(vect_conditions[np.asarray(vect_samples_id) == sample])
            if len(c) != 1:
                raise ValueError(f"dataset_id {sample} has several network_condition values: {list(c)}")
            labels.append(c[0])
        names, idx = np.unique(labels, return_inverse=True)
        if len(names) < 2:
            return
        self.sample_conditions, self.condition_names = idx.astype(int), [str(n) for n in names]
        if verb:
            print("[fit_network] Network conditions (network_condition_pen = "
                  f"{self.network_condition_pen}): " + '; '.join(
                      f"{n}: {[str(s) for s, i in zip(samples_id, idx) if i == k]}" for k, n in enumerate(names)))

    def condition_networks(self, inter):
        """{condition label: (G, G, n_networks) network} of a per-sample inter; {None: inter} if common."""
        if np.ndim(inter) < 4:
            return {None: inter}
        if self.sample_conditions is None:
            return {str(s): inter[k] for k, s in enumerate(self._sample_names or range(len(inter)))}
        first = {c: int(np.flatnonzero(self.sample_conditions == k)[0]) for k, c in enumerate(self.condition_names)}
        return {c: inter[k] for c, k in first.items()}

    def save_network_conditions(self, cardamom_dir, gene_names=None, inter=None, suffix=''):
        """
        With network conditions: network_conditions.json (condition of each sample), inter_shared<suffix>.npy
        (median network over the conditions) and network_differences<suffix>.csv (edges whose value differs
        between conditions, one column per condition). Removes them otherwise.
        """
        import json
        import pandas as pd
        inter = self.inter if inter is None else inter
        files = [os.path.join(cardamom_dir, f) for f in
                 ('network_conditions.json', f'inter_shared{suffix}.npy', f'network_differences{suffix}.csv')]
        if self.sample_conditions is None or np.ndim(inter) < 4:
            for f in files[(1 if suffix else 0):]:
                if os.path.exists(f):
                    os.remove(f)
            return
        if not suffix:   # written by the inference (infer_network_structure) only
            with open(files[0], 'w') as fh:
                json.dump({'samples': list(self._sample_names or []), 'conditions': self.condition_names,
                           'sample_conditions': [int(c) for c in self.sample_conditions],
                           'network_condition_pen': float(self.network_condition_pen)}, fh, indent=1)
        nets = self.condition_networks(inter)
        stack = np.stack(list(nets.values()))          # (C, G, G, n_networks)
        np.save(files[1], np.median(stack, axis=0))
        names = list(gene_names) if gene_names is not None else [str(g) for g in range(stack.shape[1])]
        rows = []
        for i, j, n in zip(*np.nonzero(np.ptp(stack, axis=0) > 0)):
            if i == j:  # self-regulations (often artefacts) are not compared
                continue
            rows.append({'regulator': names[i], 'target': names[j], 'network': int(n),
                         **{f'{c}': float(v) for c, v in zip(nets, stack[:, i, j, n])},
                         'sign_change': bool(stack[:, i, j, n].min() < 0 < stack[:, i, j, n].max())})
        pd.DataFrame(rows).to_csv(files[2], index=False)

    def load_network_conditions(self, cardamom_dir):
        """Network conditions saved by save_network_conditions (None if the run had one common network)."""
        import json
        path = os.path.join(cardamom_dir, 'network_conditions.json')
        self.sample_conditions, self.condition_names = None, None
        if os.path.exists(path):
            info = json.load(open(path))
            self.sample_conditions = np.asarray(info['sample_conditions'], dtype=int)
            self.condition_names = list(info['conditions'])

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
            if name in RENAMED_PARAMETERS:
                print(f"[CardamomOT] Warning: model_parameters: '{name}' is now '{RENAMED_PARAMETERS[name]}'; "
                      "rename the row of the workbook")
                name = RENAMED_PARAMETERS[name]
            if name in REMOVED_PARAMETERS:
                print(f"[CardamomOT] Warning: model_parameters: '{name}' was removed ({REMOVED_PARAMETERS[name]}); ignored")
                continue
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
                # (n_targets, n_networks), or ([n_samples,] ...) with per-sample interactions
                w = signs[targets][:, None] * (100 + np.abs(inter_t[cnt][..., targets, :]).sum(axis=-3))
                if basal_t.ndim == 4:
                    # Each sample with its own schedule of the stimulus
                    u = _sample_values(pert, times[cnt + 1], np.arange(basal_t.shape[1]))
                    basal_t[cnt][:, targets, :] += u[:, None, None] * (w if w.ndim == 3 else w[None])
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

    def _t0_cell_types(self, vect_t, vect_samples_id, sample, t0=None):
        """Cell types of the first-timepoint cells of a sample (its own first time by default; None if unavailable)."""
        ct = getattr(self, '_strata_labels', None)
        if ct is None:
            return None
        m = vect_samples_id == sample
        t0 = np.min(vect_t[m]) if t0 is None else t0
        return ct[(vect_t == t0) & m]

    def _valid(self):
        """Validity of the trajectory states (all valid when no sample misses a timepoint)."""
        return np.ones(len(self.times_data), dtype=bool) if self.traj_valid is None else np.asarray(self.traj_valid, bool)

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
        self._birth_rate = self._death_rate = self._birth_rate_base = self._death_rate_base = None
        self._cell_types = None
        self._transition_rates = None
        self._transition_rates_samples = {}   # {dataset_id: matrix}: transition rates of the samples with their own
        self._transition_type_labels = None
        self._lineage = None
        self._lineage_known = None
        # Cell types (if any) stratify every batch built from the data
        self._strata_labels = self._cell_type_labels(data)
        obs = getattr(data, 'obs', None)
        if obs is not None:
            if 'proliferation_net_rate' in obs:
                self._prolif_net_rate = obs['proliferation_net_rate'].values.astype(float)
            # Birth and death rates of each cell (dilution of its proteins); without them, max(±net, 0)
            if 'proliferation_birth_rate' in obs:
                self._birth_rate = np.maximum(obs['proliferation_birth_rate'].values.astype(float), 0.0)
                self._death_rate = np.maximum(obs['proliferation_death_rate'].values.astype(float), 0.0) \
                    if 'proliferation_death_rate' in obs else np.maximum(self._birth_rate - self._prolif_net_rate, 0.0)
            elif self._prolif_net_rate is not None:
                self._birth_rate = np.maximum(self._prolif_net_rate, 0.0)
                self._death_rate = np.maximum(-self._prolif_net_rate, 0.0)
            # Without the stimulus (rates of the proliferation MLP when the stimulus effects are given apart)
            self._birth_rate_base, self._death_rate_base = self._birth_rate, self._death_rate
            if 'proliferation_birth_rate_base' in obs and 'proliferation_death_rate_base' in obs:
                self._birth_rate_base = np.maximum(obs['proliferation_birth_rate_base'].values.astype(float), 0.0)
                self._death_rate_base = np.maximum(obs['proliferation_death_rate_base'].values.astype(float), 0.0)
            if 'lineage' in obs:
                self._lineage_known = obs['lineage'].notna().values
                self._lineage = obs['lineage'].astype(str).values

        if transition_rates is None:
            return
        if isinstance(transition_rates, dict):
            self._load_sample_transitions(data, transition_rates)
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
        self._transition_rates = np.clip(_Tr, 0.0, None)  # off-diagonal rates (h^-1); diagonal ignored
        print(f"Transition rates anchored on adata.obs['{ct_col}']")

    def _load_sample_transitions(self, data, transition_rates):
        """
        Transition rate matrices {'default': DataFrame or None, dataset_id: DataFrame} (inputs.load_transition_rates):
        each matrix must hold every cell type of the grouping (else it is ignored with a warning, the sample then
        takes the default matrix, or no transition constraint without one). All matrices are aligned on the sorted
        cell types.
        """
        ct_col = resolve_cell_type_obs(data, 'transition')
        if ct_col is None:
            print("Warning: transition_rates given but adata.obs has none of "
                  f"{list(CELL_TYPE_OBS_KEYS['transition'])}; OT run without transition constraint")
            return
        cell_types = data.obs[ct_col].values.astype(str)
        labels = list(np.unique(cell_types))
        mats = {}
        for name, df in transition_rates.items():
            if df is None:
                continue
            df = df.copy()
            df.index, df.columns = df.index.astype(str), df.columns.astype(str)
            missing = sorted(set(labels) - set(df.index) | set(labels) - set(df.columns))
            if missing:
                print(f"Warning: cell type(s) {missing} of adata.obs['{ct_col}'] not found in the transition_rates of "
                      f"{'the default' if name == 'default' else 'sample ' + str(name)}: ignored")
                continue
            mats[name] = np.clip(df.loc[labels, labels].to_numpy().astype(float), 0.0, None)
        present = {str(s) for s in (data.obs['dataset_id'].astype(str).unique() if 'dataset_id' in data.obs else [])}
        absent = sorted(set(mats) - {'default'} - present)
        if absent:
            print(f"Warning: transition_rates given for sample(s) {absent} absent from the data: ignored")
        self._cell_types = cell_types
        self._transition_type_labels = labels
        self._transition_rates = mats.get('default')
        self._transition_rates_samples = {k: v for k, v in mats.items() if k != 'default' and k in present}
        if self._transition_rates is not None or self._transition_rates_samples:
            print(f"Transition rates anchored on adata.obs['{ct_col}']: default "
                  f"{'yes' if self._transition_rates is not None else 'no'}, own matrix for "
                  f"{sorted(self._transition_rates_samples) or 'no sample'}")

    def _transition_matrix(self, s_idx):
        """Transition rate matrix of sample index s_idx (its own, else the default, else None)."""
        own = getattr(self, '_transition_rates_samples', None)
        if own:
            names = getattr(self, '_sample_names', None)
            if names is not None and 0 <= int(s_idx) < len(names) and names[int(s_idx)] in own:
                return own[names[int(s_idx)]]
        return getattr(self, '_transition_rates', None)

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
        counts and their own fit (modes not supported in the sample: the target would describe another
        sample's expression), except the perturbed ones (kov_genes), which take the target parameters
        so that the perturbation stays visible against the common level; cells of unfitted samples take
        the target (genes never supported anywhere: the pooled fit). self.a is always (S, M+1, G) and
        self.pi_zinb (S, G - ns), samples ordered as np.unique(obs[sample_key]) (= sample index of
        fit_network); at lam = 1 all samples share the parameters of their integrated pairs.

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
        perturbed = np.zeros((S, G_tot), dtype=bool)  # KO/OV pairs of perturbation_inference: target parameters
        for i, f in enumerate(fits):
            eligible[i] = supported_modes(f['a'], f['proba_init'], ns, self.min_mode_weight_integration,
                                          self.min_mode_ratio_integration)
            for gene in (kov_genes or {}).get(str(f['id']), ()):
                if gene in gene_names:
                    eligible[i, ns + gene_names.index(gene)] = False
                    perturbed[i, ns + gene_names.index(gene)] = True
        eligible[:, :ns] = False
        eligible = consistent_modes(a_samples, [f['proba_init'] for f in fits], eligible, ns)
        # Pairs keeping their own fit: not integrated and not perturbed
        own = ~eligible & ~perturbed
        own[:, :ns] = False
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
                if own[i, g]:  # raw counts, responsibilities of the sample's own fit
                    proba_init[m, g] = f['proba_init'][:, g]
                    proba[m, g] = f['proba'][:, g]
                elif eligible[i, g]:
                    z = np.argmax(f['proba_init'][:, g, :], axis=1)
                    if lam > 0:
                        X[m, g] = integrate_counts(data_rna[m, g], z, (a_samples[i, :-1, g], a_samples[i, -1, g], pi0_samples[i, g]),
                                                   (a_lam[i, :-1, g], a_lam[i, -1, g], pi_lam[i, g]), rng,
                                                   s=None if depth is None else depth[m])
                    proba_init[m, g] = f['proba_init'][:, g]
                    proba[m, g] = f['proba'][:, g]
        # Raw (sample, gene) pairs: classify the raw counts with the global modes (posteriors and masses as a fit)
        raw_cells, raw_masses = {}, {}
        for g in range(ns, G_tot):
            ng = n_modes[g]
            raw = ~fitted.copy()
            for i, f in enumerate(fits):
                if perturbed[i, g]:
                    raw |= f['mask']
            raw_cells[g] = raw
            if raw.any():
                resp, raw_masses[g] = self._posteriors_and_masses(
                    data_rna[raw, g], vect_t[raw], a[:ng, g], a[-1, g],
                    pi_zero=pi0[g] if pi0[g] > 0 else None, zi=True if pi0[g] > 0 else None,
                    depth=None if depth is None else depth[raw])
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
            a_per[k][:, own[i]] = a_samples[i][:, own[i]]
            pi0_per[k][own[i]] = pi0_samples[i][own[i]]
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
        # Mode masses per time: those of each sample's own fit (mean_forcing_em), and of the raw cells, by cells
        def masses_at(pi, t):
            p = np.asarray(pi[t] if isinstance(pi, dict) else pi, dtype=float)
            return np.pad(p, (0, M - len(p)))[:M]
        self.pi_init = []
        for g in range(ns, G_tot):
            ng = n_modes[g]
            pi_g = {}
            for t in times_all:
                mt = vect_t == t
                p = np.zeros(M)
                for f in fits:
                    n = np.sum(f['mask'] & mt & ~raw_cells[g])
                    if n:
                        p += n * masses_at(f['pi_init'][g - ns], t)
                n_raw = np.sum(raw_cells[g] & mt)
                if n_raw:
                    p += n_raw * masses_at(raw_masses[g], t)
                p = p[:ng]
                pi_g[t] = p / (p.sum() + EPS)
            self.pi_init.append(pi_g)
        self._finalize_components(G_tot)

        # Per-sample parameters, kept to integrate other files of the same samples
        self.integration = dict(
            sample_ids=[f['id'] for f in fits], a_samples=a_samples, pi0_samples=pi0_samples,
            pi_init_samples=[f['pi_init'] for f in fits], eligible=eligible, a=a, pi0=pi0,
            n_modes=n_modes, ref=None if ref is None else fits[ref]['id'], sample_key=sample_key,
            lam=lam, a_dst=a_lam, pi0_dst=pi_lam, all_sample_ids=sample_ids, own=own)
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
                own = info.get('own')
                a_d = (info['a_dst'][i] if info['eligible'][i, g]
                       else a_s if own is not None and own[i, g] else info['a'])
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
                                      growth_only=False, observed=None, context=None, alpha_pool=None):
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
        observed : (S, T) bool or None: timepoints of each sample (None: from vect_t). Each sample is
            transported between its consecutive observed timepoints; its states at the other
            timepoints are virtual (copies of its previous observed state).
        alpha_pool : dict or None (see _alpha_pool). Test set: the alphas are copied from the nearest
            training state of the same time and sample instead of the previous iteration's trajectories.
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
        if observed is None:
            observed = observed_times(vect_t, vect_samples_id, times, samples_id)
        first_t = [int(np.argmax(observed[s])) for s in range(len(samples_id))]
        # Next observed timepoint of each sample after each timepoint (None at its last one)
        next_t = [[next((b for b in range(a + 1, T) if observed[s, b]), None) for a in range(T)]
                  for s in range(len(samples_id))]

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

            # Fill the initial state (first observed time of each sample) and sim_real_idx of those cells
            offset = 0
            start_rows, start_cells = [], np.zeros(len(vect_t), dtype=bool)
            for s, sample in enumerate(samples_id):
                cell_indices = (vect_t == times[first_t[s]]) & (vect_samples_id == sample)
                start_cells |= cell_indices
                global_cell_idx = np.flatnonzero(cell_indices)
                selected_init = init_cells[s]
                r0 = N_total * first_t[s] + offset + offset_init[s]
                rows = slice(r0, r0 + N_samples[s])
                start_rows.append(np.arange(r0, r0 + N_samples[s]))

                kon_modified[rows, ns:] = self.modes[cell_indices][selected_init, ns:]
                if self.compute_with_proba:
                    proba_modified[rows] = self.proba[cell_indices][selected_init]
                rna_modified[rows, ns:] = vect_rna[cell_indices][selected_init, ns:]
                sim_real_idx[rows] = global_cell_idx[selected_init]
                # Sample label of every trajectory slot, last timepoint included
                for t_idx in range(T):
                    vect_samples_id_modified[N_total * t_idx + offset:N_total * t_idx + offset + N_full[s]] = s

                offset += N_full[s]

            start_rows = np.concatenate(start_rows)
            s1_traj = s1_rows(s1, vect_samples_id_modified)
            # Initial proteins at the equilibrium compressed by the dilution, c = d1 / (d1 + b) (1 without)
            prot_modified[start_rows, ns:] = self.adaptive_shrinkage_init(
                rna_modified[start_rows, ns:] * _rows(s1_traj, start_rows), kon_modified[start_rows, ns:]) \
                * self._dilution(sim_real_idx[start_rows])[1]
            if n_iter == 1:
                s1_cells = s1_rows(s1, vect_samples_id)
                prot_formodes[start_cells, ns:] = self.adaptive_shrinkage_init(
                    vect_rna[start_cells, ns:] * _rows(s1_cells, start_cells), self.modes[start_cells, ns:]) \
                    * self._dilution(np.flatnonzero(start_cells))[1]

        for t_idx, time in enumerate(times[:-1]):
            offset = 0
            for s_idx, sample in enumerate(samples_id):
                # Interval of this sample: from an observed time to its next observed time b
                b = next_t[s_idx][t_idx] if observed[s_idx, t_idx] else None
                if b is None:
                    offset += N_full[s_idx]
                    continue
                cell_idx = real_cell_batches[s_idx][b - 1][batch_idx[s_idx]]
                offset_init_s = offset + offset_init[s_idx]
                start_index = N_total * t_idx + offset_init_s
                next_index = N_total * b + offset_init_s
                N_sample = N_samples[s_idx]
                N_cells = len(cell_idx)

                if N_sample and N_cells:

                    current_indices = np.arange(start_index, start_index + N_sample)
                    next_indices = np.arange(next_index, next_index + N_sample)
                    alpha_indices = np.arange(offset_init_s, offset_init_s + N_sample)

                    # Old trajectories of this batch block at t+1 (not yet overwritten):
                    # candidate pool for the alpha re-assignment below
                    if alpha_pool is None and prot_old_is_nonzero and next_t[s_idx][b] is not None:
                        prot_old_blk = y_prot_old[next_indices, ns:]
                        kon_old_blk = y_kon_old[next_indices, ns:]
                        alpha_old_blk = alpha_old[b, alpha_indices]
                    else:
                        prot_old_blk = None

                    prot_init = prot_modified[current_indices, ns:]
                    alpha_init = alpha_modified[t_idx, alpha_indices]
                    if alpha_pool is not None and t_idx == first_t[s_idx]:
                        # Initial states: alpha of the nearest training state (later ones are matched below)
                        alpha_init = alpha_modified[t_idx, alpha_indices] = self._match_alpha(
                            alpha_pool, time, s_idx, prot_init, kon_modified[current_indices, ns:], G)
                    s1_s, ks_s = s1_of(s1, s_idx), ks_of(ks, s_idx)
                    mode_init = self.adaptive_shrinkage(rna_modified[current_indices, ns:] * s1_s, kon_modified[current_indices, ns:]) / s1_s
                    mode_end = self.adaptive_shrinkage(vect_rna[cell_idx, ns:] * s1_s, self.modes[cell_idx, ns:]) / s1_s

                    basal_s = basal[min(s_idx, basal.shape[0] - 1)] if basal.ndim == 3 else basal
                    # Dilution of the source states at the birth rate of their real cell (rate d1 + b, targets c u)
                    rate_src, c_src, b_src = self._dilution(sim_real_idx[current_indices])
                    b_src = np.zeros(0) if b_src is None else b_src
                    pairwise_dist = my_otdistance(
                        kon_modified[current_indices, ns:], self.modes[cell_idx, ns:],
                        prot_init,
                        rna_modified[current_indices, ns:], vect_rna[cell_idx, ns:],
                        proba_modified[current_indices, ns:], self.proba[cell_idx, ns:, :],
                        mode_init, mode_end,
                        alpha_init,
                        s1_s, ks_s, self.d[1, ns:], times[b] - time, basal_s, inter_of(inter, s_idx), loss=self.loss_norm,
                        n_iter=n_iter, intensity_prior=intensity_prior,
                        compute_with_proba=self.compute_with_proba,
                        n_stimuli=ns, stim_vals=np.asarray(self._stim_schedule.at(times[b], s_idx), dtype=np.float64),
                        scale_proteins=self.scale_proteins,
                        weight_fixed=-1.0 if context is None else context['weight_init'],
                        b_init=b_src,
                    )

                    delta_t = times[b] - time
                    src_real = sim_real_idx[current_indices]
                    tmp = np.log(G)
                    it_reg = n_iter if context is None else context['n_iter_reg']  # final training regularization
                    reg = max(self.init_entropic_noise * tmp * (1 / it_reg)**(1 - 1/it_reg), .01)

                    # --- Transition prior: Gibbs kernel times type-to-type probabilities ---
                    _tr = self._transition_matrix(s_idx)
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
                        pairwise_dist = pairwise_dist + self._transition_log_penalty(delta_t, reg, _tr)[np.ix_(src_ti, tgt_ti)]

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

                    if growth_only:
                        R_opt_traj[start_index:start_index + N_sample] = self._growth_log_mass(
                            pairwise_dist, src_real, delta_t, reg, tmp, time, times[b], sample) / delta_t
                        if self.couplings is not None:
                            self._record_coupling(pairwise_dist, src_real, cell_idx, delta_t, reg, tmp,
                                                  s_idx, time, times[b])

                # Trajectory update (skipped by the growth pass)
                if N_sample and N_cells and not growth_only:
                    # --- Growth-weighted OT marginals ---
                    # WOT convention (Schiebinger et al. 2019): both the source AND
                    # target marginals are corrected by exp(±R·Δt/2), not just the
                    # source by exp(R·Δt) — see docs/advanced.md#net-proliferation-rate--default-behaviour.
                    mu, nu, reg_m = self._traj_marginals(src_real, cell_idx, delta_t, tmp)
                    coupling = self._solve_ot(mu, nu, pairwise_dist, reg, reg_m)

                    # Draw one target per trajectory from its coupling row (inverse CDF)
                    cdf = np.cumsum(coupling, axis=1)
                    u = np.random.random(N_sample) * cdf[:, -1]
                    m_idx = np.minimum((cdf < u[:, None]).sum(axis=1), N_cells - 1)
                    tgt = cell_idx[m_idx]

                    # End states recomputed for the sampled pairs only
                    next_prot = find_next_prot(
                        rate_src, prot_init, rna_modified[current_indices, ns:],
                        vect_rna[tgt, ns:], mode_init * c_src, mode_end[m_idx] * c_src, alpha_init, s1_s, delta_t)

                    # Timepoints missed by the sample inside the interval: proteins along the same flow
                    for k in range(t_idx + 1, b):
                        prot_modified[N_total * k + offset_init_s + np.arange(N_sample), ns:] = prot_along(
                            rate_src, prot_init, mode_init * c_src, mode_end[m_idx] * c_src, alpha_init, s1_s,
                            delta_t, times[k] - time)

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
                        alpha_modified[b, alpha_indices] = alpha_old_blk[np.argmin(d_match, axis=1)]
                    elif alpha_pool is not None and next_t[s_idx][b] is not None:
                        alpha_modified[b, alpha_indices] = self._match_alpha(
                            alpha_pool, times[b], s_idx, next_prot, kon_modified[next_indices, ns:], G)

                offset += N_full[s_idx]

        if not growth_only and not observed.all():
            # Virtual states (timepoints a sample misses): copies of its previous observed state, proteins
            # along the flow inside an interval (above), copies before / after its observed times
            valid = valid_rows(observed, N_full)
            arrays = [rna_modified, kon_modified, sim_real_idx] + ([proba_modified] if self.compute_with_proba else [])
            fill_virtual(arrays, valid, T)
            fill_virtual([prot_modified], valid, T, inside=False)

        return prot_modified, prot_formodes, rna_modified, kon_modified, \
                proba_modified, alpha_modified, vect_samples_id_modified, \
                  R_opt_traj, to_keep_for_update

    def _alpha_pool(self, train, samples_id, train_sample_ids):
        """
        Training trajectories as a pool of (protein, kon, alpha) states per time, to give the test states
        the alpha of their nearest training state. train: dict of the saved training arrays (prot, kon_beta,
        alpha, times_data, samples_data, valid).
        """
        t = np.sort(np.unique(train['times_data']))
        T = len(t)
        shape = lambda a: np.asarray(a).reshape((T, -1) + np.shape(a)[1:])
        train_sample_ids = list(train_sample_ids)
        return dict(t=t, prot=shape(train['prot']), kon=shape(train['kon_beta']), alpha=np.asarray(train['alpha']),
                    slot_sample=np.asarray(train['samples_data']).reshape(T, -1)[0].astype(int),
                    valid=shape(train['valid']).astype(bool),
                    sample_map=[train_sample_ids.index(s) if s in train_sample_ids else 0 for s in samples_id])

    def _match_alpha(self, pool, t, s_idx, prot, kon, G):
        """Alpha of the nearest training state (weighted L1 on proteins and kon) at the time and sample of the states."""
        ns = self.n_stimuli
        k = min(int(np.argmin(np.abs(pool['t'] - t))), pool['alpha'].shape[0] - 1)
        rows = np.flatnonzero((pool['slot_sample'] == pool['sample_map'][s_idx]) & pool['valid'][k])
        if not len(rows):
            rows = np.flatnonzero(pool['valid'][k])
        w_p, w_k = 1 / G, (G - 1) / G
        d_match = cdist(np.hstack([prot * w_p, kon * w_k]),
                        np.hstack([pool['prot'][k][rows, ns:] * w_p, pool['kon'][k][rows, ns:] * w_k]), 'cityblock')
        return pool['alpha'][k][rows[np.argmin(d_match, axis=1)]]

    def _traj_marginals(self, src_real, cell_idx, delta_t, log_G):
        """Marginals (mu, nu) and unbalanced relaxation of the trajectory OT between source states and real cells."""
        r = getattr(self, '_prolif_net_rate', None)
        if r is not None:
            mu = np.exp(r[src_real] * delta_t / 2)
            nu = np.exp(-r[cell_idx] * delta_t / 2)
            mu, nu = mu / mu.sum(), nu / nu.sum()
        else:
            mu = np.ones(len(src_real)) / len(src_real)
            nu = np.ones(len(cell_idx)) / len(cell_idx)
        reg_m = np.array([1e3, self.unbalanced_reg * log_G]) if self.unbalanced_reg else None
        return mu, nu, reg_m

    def _record_coupling(self, C, src_real, cell_idx, delta_t, reg, log_G, s_idx, t_from, t_to, top_k=50):
        """
        Soft coupling of the final trajectories (same costs and marginals as the trajectory OT), kept sparse:
        the top_k targets of each source state; entries (real source cell, real target cell, mass), the
        mass of each state scaled to ~1 (batches of different sizes), summed later over duplicated states.
        """
        mu, nu, reg_m = self._traj_marginals(src_real, cell_idx, delta_t, log_G)
        P = self._solve_ot(mu, nu, C, reg, reg_m) * len(src_real)
        k = min(top_k, P.shape[1])
        cols = np.argpartition(-P, k - 1, axis=1)[:, :k]
        w = np.take_along_axis(P, cols, axis=1)
        keep = w > 1e-3 * np.maximum(w.max(axis=1, keepdims=True), 1e-300)
        rows = np.broadcast_to(np.arange(len(src_real))[:, None], cols.shape)
        self.couplings.append(dict(src=src_real[rows[keep]], tgt=cell_idx[cols[keep]], w=w[keep].astype(np.float32),
                                   sample=s_idx, t_from=float(t_from), t_to=float(t_to)))

    def _dilution(self, real_idx):
        """
        Protein relaxation of trajectory states with dilution at the birth rate b of their real cell (real_idx, -1:
        b = 0): dP/dt = d1 u - (d1 + b) P = (d1 + b)(c u - P). Returns (rate d1 + b (N, G), factor c = d1 / (d1 + b)
        (N, G), b (N,)) to use in place of (d1, 1) with the modes as targets; (d1, 1.0, None) exactly without dilution.
        """
        ns = self.n_stimuli
        d1 = self.d[1, ns:]
        b = getattr(self, '_birth_rate', None)
        if not self.protein_dilution or b is None or real_idx is None:
            return d1, 1.0, None
        real_idx = np.asarray(real_idx, dtype=int)
        bs = np.where(real_idx >= 0, b[np.maximum(real_idx, 0)], 0.0)
        rate = d1[None, :] + bs[:, None]
        return rate, np.where(rate > 0, d1[None, :] / np.maximum(rate, 1e-300), 1.0), bs

    def _prior_state_rate(self, attr='_birth_rate'):
        """Per-cell rate attr (birth, death, with or without stimulus) of the real cell of each trajectory state
        (traj_real_idx; 0 if none), or None if unavailable."""
        r, idx = getattr(self, attr, None), getattr(self, 'traj_real_idx', None)
        if r is None or idx is None or len(idx) != len(self.times_data):
            return None
        idx = np.asarray(idx, dtype=int)
        return np.where(idx >= 0, r[np.maximum(idx, 0)], 0.0)

    def _prior_state_birth(self):
        """Prior birth rate of each trajectory state, or None without dilution."""
        return self._prior_state_rate('_birth_rate') if self.protein_dilution else None

    def _state_birth(self, verb=False):
        """
        Birth rate of each trajectory state used to refit d1: the prior birth of its real cell, the same as in the
        inferred trajectories (the simulations use its regression on the state, MLP birth part, or the prior of
        the slot). With b explicit in the refitted ODE, d1 stays a pure degradation rate. None without dilution.
        """
        b = self._prior_state_birth()
        if verb and b is not None:
            print(f"[refine_network_degradations] Dilution of the proteins: prior birth rate of the real cells, "
                  f"mean {np.mean(b):.4f} h^-1")
        return b

    def _transition_log_penalty(self, delta_t, reg, rates=None):
        """
        Cost penalty -reg·log Π(Δt) between cell types, with Π = expm(Q·Δt) the type-to-type
        transition probabilities of the generator Q (off-diagonal rates of the transition_rates
        matrix, diagonal = minus their row sum): the Gibbs kernel exp(-C/reg) is multiplied by Π.
        """
        Q = (self._transition_rates if rates is None else rates).copy()
        np.fill_diagonal(Q, 0.0)
        np.fill_diagonal(Q, -Q.sum(axis=1))
        Pi = expm(Q * delta_t)
        # Floor: unreachable transitions get a factor 1e-12 instead of 0 (Sinkhorn stays stable)
        return -reg * np.log(np.clip(Pi, 1e-12, None))

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

    def _growth_log_mass(self, C, src_real, delta_t, reg, log_G, t_from, t_to, sample=None):
        """
        Log mass gain of each source state over one interval (Waddington-OT growth
        estimation): the source marginal starts from the prior growth exp(R_prior·Δt)
        and is relaxed (growth_reg_source) while the target (observed cells) stays
        nearly hard; n_growth_iter times, source weights <- row marginals. The mean
        population growth is that of the prior (anchored by fit_population_anchors on the population sizes, if given).
        sample is unused (kept for the call).
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
        """Largest parameter count of one target-gene fit: (active regulators x network conditions + per-sample basals) x n_networks."""
        k_max = max((len(c) for c in active_cols[self.n_stimuli:]), default=0)
        n_cond = 1 if self.sample_conditions is None else int(np.max(self.sample_conditions)) + 1
        return (k_max * n_cond + n_samples) * int(self.n_networks)

    def _network_batch_size(self, n_params):
        """Network sub-sample size: at least 10 states per parameter; batch_size_network (if not None) can only raise it."""
        floor = 10 * n_params
        return floor if self.batch_size_network is None else max(self.batch_size_network, floor)

    def _fit_theta_averaged(self, fit_fn, times_vec, samples_vec, labels, n_fits, n_params, valid=None):
        """
        Theta from fit_fn(sels) (one fit per sub-sample, run jointly) on sub-samples of
        _network_batch_size(n_params) trajectory states,
        stratified by (time, sample) and cell type (labels, if not None). Mean over
        min(n_fits, 1 + n_states // batch_size) disjoint sub-samples (covering every state
        about once, at most n_fits); a single fit when one sub-sample holds every state.
        Returns (basal, inter, basal_tmp, inter_tmp).
        """
        batch_size = self._network_batch_size(n_params)
        # Virtual states (time not observed for their sample) are left out
        rows = np.arange(len(times_vec)) if valid is None else np.flatnonzero(valid)
        times_vec, samples_vec = np.asarray(times_vec)[rows], np.asarray(samples_vec)[rows]
        labels = None if labels is None else np.asarray(labels)[rows]
        # Disjoint sub-samples covering as many states as possible (a single one if it holds them all)
        n_fits = max(5, min(n_fits, 1 + len(times_vec) // batch_size))
        sels, _ = grouped_partition([times_vec, samples_vec], batch_size, n_fits, labels)
        sels = [rows[sel] for sel in sels]
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
        context=None,
        alpha_pool=None,
    ):
        """
        Alternating optimization of trajectories and network (theta).

        context : dict or None
            Test set: context of the last training iteration, kept fixed ('n_iter_reg': counter of the
            Sinkhorn regularization, 'weight_init': weight of the mode-to-mode OT cost; 'weight_prob': final weight
            of the mixture probabilities in the basin updates, used by _assign_calibrated_basins). None: training.

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
        # Iteration counter of the OT regularization (unused with a fixed context, test set)
        it_shift = 0 if context is not None else min_n_loops * min(1, 1 - compute_theta + hard_forcing_ref)

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
                # (G, G, n_networks), or per sample (n_samples, G, G, n_networks) with network conditions
                self.inter = np.array(inter_init, dtype=float)

        # Ensure basal is always 3-D (n_samples, G_tot, n_networks) — promote 2-D for compat
        if self.basal.ndim == 2:
            self.basal = self.basal[np.newaxis, :, :]
        basal = self.basal.copy()      # (n_samples, G_tot, n_networks)
        inter = self.inter.copy()      # (G_tot, G_tot, n_networks), or (n_samples, ...) with network conditions
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
        # Timepoints of each sample: states at the other timepoints are virtual (excluded from the fits),
        # and every interval of the trajectories joins consecutive observed timepoints of its sample
        observed = observed_times(vect_t, vect_samples_id, times, samples_id)
        valid = valid_rows(observed, N_full)
        pairs = trajectory_pairs(valid, len(times))
        first_t = [int(np.argmax(observed[s])) for s in range(len(samples_id))]

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
            # Flow-matching states at the end of each interval (consecutive observed times of each sample)
            s1r = s1_rows(s1, y_samples)
            modes = self.adaptive_shrinkage(y_rna[:, ns:] * s1r, y_kon[:, ns:]) / s1r
            for a, b, slots in pairs:
                idx_prev, idx_next = N_tot * a + slots, N_tot * b + slots
                rate_p, c_p, _ = self._dilution(y_real[idx_prev])  # dilution of the source states
                self._fill_prev_prot(
                    y_prot_prev, idx_next, y_alpha[a][slots], times[b] - times[a],
                    rate_p, y_prot[idx_prev, ns:],
                    y_rna[idx_prev, ns:] * self.scale_proteins,
                    y_rna[idx_next, ns:] * self.scale_proteins,
                    modes[idx_prev] * c_p, modes[idx_next] * c_p, _rows(s1r, idx_prev))

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
                    seuil_zero_min_ref=self.seuil_zero_min_ref,
                    sample_conditions=self.sample_conditions, condition_pen=self.network_condition_pen)
            return self._fit_theta_averaged(fit_on, vect_t_sim, y_samples, traj_cell_types(), n_fits,
                                            self._n_params_per_target(prev_cols, n_samples_local), valid=valid)

        # Stratum of each real cell, for balanced OT batches; real cells at t0 per sample
        strata = self._traj_strata(data_rna, vect_t, vect_samples_id, times, samples_id)
        t0_real = [np.flatnonzero((vect_t == times[first_t[s]]) & (vect_samples_id == sample))
                   for s, sample in enumerate(samples_id)]

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
            for s_, sample in enumerate(samples_id):  # first observed cells of each sample
                to_keep_for_update[(vect_t == times[first_t[s_]]) & (vect_samples_id == sample)] = True
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
                        sim_real_idx=y_real, observed=observed, context=context, alpha_pool=alpha_pool
                    )

                offset_init = [offset_init[s] + N_tmp[s] for s in range(len(samples_id))]

            if y_prot_prev is not None:
                # Starting states of each sample (first observed time): flow-matching state = state
                start = np.concatenate([N_tot * first_t[s_] + np.arange(sum(N_full[:s_]), sum(N_full[:s_ + 1]))
                                        for s_ in range(len(samples_id))])
                for c, v in zip(y_prot_prev.cols, y_prot_prev.values):
                    is_gene = np.flatnonzero(c >= ns)
                    v[np.ix_(start, is_gene)] = y_prot[np.ix_(start, c[is_gene])]

            if alpha_pool is not None:
                # Single pass: trajectories and alphas are final, only kon_theta of the network remains
                kon_vector = y_kon.copy()
                kon_vector[:, ns:] = self._kon_ref_per_sample(y_prot, ks, inter, basal, samples_id=sample_idx, samples_data=y_samples)[:, ns:]
                n_iter += 1
                break

            # --- Evaluate error before and after inference (observed states) ---
            error = self._count_errors_per_sample(y_prot[valid], y_kon[valid], y_proba[valid], ks, inter, basal,
                                                   samples_id=sample_idx, samples_data=y_samples[valid])
            if compute_theta and len(times) > 1:
                if self.weight_prev > 0:
                    refresh_prev_prot()
                # Mean of fits on independent stratified subsamples covering the trajectory states
                basal, inter, basal_tmp, inter_tmp = fit_theta(weight_prev, basal, inter, self.n_network_fits)

            error_2 = self._count_errors_per_sample(y_prot[valid], y_kon[valid], y_proba[valid], ks, inter, basal,
                                                    samples_id=sample_idx, samples_data=y_samples[valid])
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
                # Intervals are independent: one parallel task per (pair of observed times, slots)
                alpha_fn = inference_alpha_1thread if len(pairs) > 1 else inference_alpha
                def alpha_task(a, b, slots):
                    r0, r1 = N_tot * a + slots, N_tot * b + slots
                    rate_a, c_a, _ = self._dilution(y_real[r0])  # dilution of the source states
                    return delayed(alpha_fn)(
                            rate_a, s1,
                            y_alpha[a][slots],
                            y_kon[r0], kon_vector[r0], y_prot[r0], y_rna[r0],
                            y_kon[r1], kon_vector[r1], y_prot[r1], y_rna[r1],
                            modes[r0] * c_a, modes[r1] * c_a,
                            basal, inter, ks, times[b] - times[a],
                            tol=self.alpha_threshold,
                            n_pas = self.n_pas if self.force_n_pas else max(self.n_pas, int(times[b] - times[a])),
                            samples_data=y_samples[r0],
                            stim_vals=self._stim_schedule.per_cell(times[b], y_samples[r0]),
                            scale_proteins=self.scale_proteins
                        )
                # Same pool size as every other Parallel call: a different n_jobs makes loky respawn workers
                alphas = Parallel(n_jobs=-1)(alpha_task(a, b, slots) for a, b, slots in pairs)
                for (a, b, slots), alpha_ab in zip(pairs, alphas):
                    y_alpha[a][slots] = alpha_ab
            
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
                # Mass constraint grows from the argmax masses (lam=0: plain argmax) to nu (lam=1: full OT) by min_n_loops
                lam = min(1.0, (n_iter - 1) / max(min_n_loops - 1, 1))
                # The weight of the network increases slowly because it aims to get the right attribution given probabilities that are close
                weight_prob = max(.96**(n_iter - 1), .1)

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
        self.couplings = [] if len(times) > 1 else None  # final soft couplings, filled by the growth pass
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
                    sim_real_idx=y_real, growth_only=True, observed=observed, context=context, alpha_pool=alpha_pool
                )[7]
                offset_init = [offset_init[s] + N_tmp[s] for s in range(len(samples_id))]
            # Virtual states: rate of the interval of observed times that contains them (0 outside)
            R2 = R_opt_traj.reshape(len(times), N_tot)
            V = valid.reshape(len(times), N_tot)
            for a, b, slots in pairs:
                R2[a + 1:b][:, slots] = R2[a, slots][None, :]
            R2[~V & np.isnan(R2)] = 0.0
            if verb:
                print(f"[fit_network] Growth pass: net rate per state in "
                      f"[{np.nanmin(R_opt_traj):.3g}, {np.nanmax(R_opt_traj):.3g}], mean {np.nanmean(R_opt_traj):.3g}")
        self.R_opt = R_opt_traj
        # Context of the last iteration, kept fixed by the test-set inference
        if context is None:
            n_reg = n_iter - 1 + it_shift
            ip = intensity_prior * compute_theta * (1 - hard_forcing_ref)
            context = {'n_iter_reg': n_reg,
                       'weight_init': float(n_reg < ip) * (1 / n_reg)**(1 - 1 / n_reg),
                       'weight_prob': max(.96**(n_iter - 1), .1)}
        self.final_context = context

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
        self.traj_valid = valid


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
        vect_conditions=None,
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
        vect_conditions : ndarray or None
            Network condition of each cell (obs['network_condition'] of an AnnData if None); with
            >= 2 conditions each has its own network, see network_condition_pen. Every sample
            (dataset_id) must belong to one condition.
        """
        seed_everything(self.seed)
        if self.simulate_full_with_harissa:
            self.scale_proteins = 1  # Harissa simulates unit-scaled proteins

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
            if isinstance(data, anndata.AnnData) and vect_conditions is None and 'network_condition' in data.obs:
                vect_conditions = data.obs['network_condition'].values
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
        self.set_network_conditions(vect_samples_id, samples_id, vect_conditions, verb=verb)
        if self.sample_conditions is not None and self.simulate_full_with_harissa:
            raise ValueError("network conditions (per-sample networks) are not supported by the Harissa simulation")

        # --- Compute number of real cells per time/sample ---
        nb_cells = np.zeros((len(samples_id), len(times)), dtype=int)
        for s, sample in enumerate(samples_id):
            for t_idx, t in enumerate(times):
                nb_cells[s, t_idx] = np.sum((vect_t == t) & (vect_samples_id == sample))

        if verb:
            print("[fit_network] Cell counts per sample/timepoint and genes:\n", nb_cells, G_tot)

        # Timepoints of each sample (samples may miss some of the timepoints)
        observed = nb_cells > 0
        first_t = [int(np.argmax(observed[s])) for s in range(len(samples_id))]
        if not observed.all() and verb:
            print("[fit_network] Samples with their own timepoints: "
                  + '; '.join(f"{sample}: {times[observed[s]].tolist()}" for s, sample in enumerate(samples_id)))
        if any(f > 0 for f in first_t):
            print("[fit_network] Warning: samples starting after the first timepoint; their simulations start at "
                  "the first timepoint from their first observed states")

        # --- Define number of cells used for inference ---
        N_samples = []
        for s in range(len(samples_id)):
            n = int(np.quantile(nb_cells[s][observed[s]], self.quant_samples))
            q, r = divmod(n, self.batch_size_traj) 
            if q == 0: N_samples.append(n)
            else: N_samples.append(min(self.batch_size_traj + 1+int(r/q), n))

        N_full = [int(np.quantile(nb_cells[s][observed[s]], self.quant_samples)) for s in range(len(samples_id))]

        if verb:
            print("[fit_network] Number of simulated cells per sample:", N_samples)
            print("[fit_network] Number of total cells per sample:", N_full)

        # --- Choose initial cells per sample ---
        init_cells_full = [
            minimal_repetition_choice(nb_cells[s, first_t[s]], N_full[s],
                                      labels=self._t0_cell_types(vect_t, vect_samples_id, sample, times[first_t[s]]))
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
            for title, inter, basal in (("Final", self.inter, self.basal),
                                        ("Intermediate", self.inter_tmp, self.basal_tmp)):
                print(f"\n[fit_network] {title} network:")
                for c, inter_c in self.condition_networks(inter).items():
                    for n in range(self.n_networks):
                        print(f"  Network {n}{'' if c is None else f' | condition {c}'} | Interactions:\n", inter_c[:, :, n].T)
                for n in range(self.n_networks):
                    print(f"  Network {n} | Basal:\n", basal.mean(axis=0)[:, n])
            

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
            # Rows as a slice or an index array (pairs of observed times of each sample)
            dest = (idx_next, is_gene) if isinstance(idx_next, slice) else np.ix_(idx_next, np.flatnonzero(is_gene))
            prev.values[g][dest] = find_next_prot(
                d1[..., r], P0[:, r], M0[:, r], M1[:, r], mode_init[:, r], mode_end[:, r],
                np.minimum(alpha[:, r] / alpha_mod, 1),
                s[..., r] if np.ndim(s) else s, delta_t * alpha_mod)

    def estimate_trajectories(self, y_prot, times, d1, N=100, kon_beta=None, s=None, prev_cols=None, birth=None):
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
        birth : array of shape (T*N,), optional
            Birth rate of each state (dilution: rate d1 + b, mode targets times d1 / (d1 + b)); None: no dilution.
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

        # Intervals between consecutive observed times of each trajectory (in time order)
        valid = self._valid() if len(self._valid()) == len(y_prot) else np.ones(len(y_prot), dtype=bool)
        for a, b, slots in trajectory_pairs(valid, len(times)):
            delta_t = times[b] - times[a]
            rows_prev, rows_next = N * a + slots, N * b + slots
            kb_prev = np.ascontiguousarray(kon_beta[rows_prev, ns:], dtype=np.float64)
            kb_next = np.ascontiguousarray(kon_beta[rows_next, ns:], dtype=np.float64)

            P0 = np.ascontiguousarray(prot_modified[rows_prev, ns:], dtype=np.float64)
            alpha_ab = np.ascontiguousarray(self.alpha[a][slots], dtype=np.float64)
            # Dilution of the start states: rate d1 + b, mode targets compressed by c = d1 / (d1 + b)
            rate, c = np.asarray(d1, dtype=np.float64), 1.0
            if birth is not None:
                rate = rate[None, :] + np.asarray(birth, dtype=np.float64)[rows_prev][:, None]
                c = np.asarray(d1, dtype=np.float64)[None, :] / rate
            prot_modified[rows_next, ns:] = find_next_prot(
                rate, P0, kb_prev, kb_next, kb_prev * c, kb_next * c, alpha_ab, s, float(delta_t))
            for k in range(a + 1, b):  # timepoints missed inside the interval: along the same flow
                prot_modified[N * k + slots, ns:] = prot_along(
                    rate, P0, kb_prev * c, kb_next * c, alpha_ab, s, delta_t, times[k] - times[a])

            if prot_modified_prev is not None and self.weight_prev > 0:
                self._fill_prev_prot(
                    prot_modified_prev, rows_next, self.alpha[a][slots], delta_t,
                    rate, prot_modified[rows_prev, ns:], kb_prev, kb_next, kb_prev * c, kb_next * c, s)
        if not valid.all():
            fill_virtual([prot_modified], valid, len(times), inside=False)

        return prot_modified, prot_modified_prev
    

    def _mrna_ratio(self):
        """Per-row g = kon_beta_nonscaled / kon_beta on the genes, kon_beta_nonscaled = adaptive_shrinkage(rna * s1, kon_beta)
        being the (unbounded) target that drives the trajectories of fit_network; recomputed from the saved rna/kon_beta."""
        ns = self.n_stimuli
        s1 = self._mixture_terms()[1]
        kon_beta_nonscaled = self.adaptive_shrinkage(self.rna[:, ns:] * s1_rows(s1, self.samples_data), self.kon_beta[:, ns:])
        kon_beta = np.clip(self.kon_beta[:, ns:] * self.scale_proteins, self.seuil, None)
        return np.clip(kon_beta_nonscaled / kon_beta, 0.1, 10.0)

    def select_cells_to_use(self):

        n_samples = len(np.unique(self.samples_data))
        t0 = np.min(self.times_data)
        N_t = np.sum(self.times_data == t0)
        cells_to_use = np.zeros_like(self.times_data, dtype=int)
        times = np.unique(self.times_data)

        for s in range(n_samples):
            # Trajectories of sample s at the first timepoint (virtual states copy its first observed one)
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
        if basal.ndim < 3 and ks.ndim < 3 and inter.ndim < 4:
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
            out[mask] = kon_ref_vector(y_prot[mask], ks_of(ks, s_idx), inter_of(inter, s_idx), basal_s)
        return out

    def _count_errors_per_sample(self, y_prot, kon_beta, proba_traj, ks, inter, basal,
                                  samples_id=None, samples_data=None):
        """
        Weighted-average count_errors respecting per-sample basal.
        When basal is 2-D, delegates to count_errors directly.

        samples_data : per-cell sample assignment; defaults to self.samples_data.
        """
        if basal.ndim < 3 and ks.ndim < 3 and inter.ndim < 4:
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
                                 ks_of(ks, s_idx), basal_s, inter_of(inter, s_idx),
                                 loss=self.loss_norm,
                                 compute_with_proba=self.compute_with_proba,
                                 n_stimuli=self.n_stimuli)
            total_err += err_s * n_s
            total_cells += n_s
        return total_err / total_cells if total_cells > 0 else 0.0

    def refine_network_degradations(self, verb=True, stimulus_schedule=None, test=False):
        """
        Refine network parameters and infer degradation rates for simulation.

        With ``mrna_driven_proteins``, the trajectories and network of fit_network are kept;
        otherwise proteins are re-estimated on the modes and the network is refitted.
        Degradations are fitted on the mean field of the simulated model: d1 on dP/dt = d1(kon(P) - P),
        with kon scaled by the per-cell ratio g = kon_beta_nonscaled/kon_beta (observed mRNA) for Harissa,
        and d1/d0 by variance matching (mRNA filtering of the noise for Harissa).

        When ``test=True``, only runs the trajectory estimation step and recomputes
        ``kon_theta`` using the current (pre-loaded simul) network. No inference
        or parameter update is performed.
        """
        seed_everything(self.seed)

        times = np.sort(np.unique(self.times_data))
        N_tot = np.sum(self.times_data == times[0])

        if stimulus_schedule is not None or self._stim_schedule is None:
            self._stim_schedule = self._build_stimulus_schedule(times, stimulus_schedule)
        
        if self.simulate_full_with_harissa:
            self.scale_proteins = 1
            self.mrna_driven_proteins = True  # Harissa: proteins driven by the mRNAs

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
            if not self.mrna_driven_proteins:
                self.prot, _ = self.estimate_trajectories(
                    self.prot, times, self.d[1, ns:], N=N_tot,
                    kon_beta=self.kon_beta, s=self.scale_proteins)
            y_prot = self.prot
            kon_vector = self.kon_beta.copy()
            kon_vector[:, ns:] = self._kon_ref_per_sample(
                y_prot, ks, inter, basal, samples_data=self.samples_data)[:, ns:]
            self.kon_theta = kon_vector
            return

        samples_id = np.sort(np.unique(self.samples_data))
        # Observed states (virtual ones: timepoints a sample misses) for the fits and errors
        valid = self._valid()

        if self.mrna_driven_proteins:
            # Trajectories and network of fit_network kept as is
            y_prot = self.prot
            if self.inter_simul_ref is not None:
                print("[refine_network_degradations] inter_simul_ref ignored: no network refit with mrna_driven_proteins")
        else:
            basal_ref, inter_ref = self.basal.copy(), self.inter.copy()
            if self.inter_simul_ref is not None:
                inter_ref = self._normalize_theta(
                    self.inter_simul_ref, self.inter.shape[-2], self.n_networks, is_inter=True)
            # Same active regulators as in the inference_network call below
            _, prev_cols = active_regulators(self.ref_network, inter_ref, self.hard_forcing_ref)
            # Proteins bounded by the modes, along the couplings of fit_network
            y_prot, y_prot_prev = self.estimate_trajectories(self.prot, times, self.d[1, ns:], N=N_tot, kon_beta=self.kon_beta, s=self.scale_proteins, prev_cols=prev_cols,
                                                             birth=self._prior_state_birth())

        error = self._count_errors_per_sample(y_prot[valid], self.kon_beta[valid], self.proba_traj[valid], ks,
                                              inter, basal, samples_id=samples_id, samples_data=self.samples_data[valid])

        if not self.mrna_driven_proteins:
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
                    sample_conditions=self.sample_conditions, condition_pen=self.network_condition_pen,
                )

            # Mean theta over fits on (time, sample, cell type)-stratified subsamples (single fit if one holds all)
            basal, inter, _, _ = self._fit_theta_averaged(
                fit_on, self.times_data, self.samples_data, self.traj_cell_types, self.n_network_fits,
                self._n_params_per_target(prev_cols, len(samples_id)), valid=valid)

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

        error_corrected = self._count_errors_per_sample(y_prot[valid], self.kon_beta[valid], self.proba_traj[valid], ks,
                                                        inter, basal, samples_id=samples_id,
                                                        samples_data=self.samples_data[valid])
        if verb:
            print("[refine_network_degradations] ratio errors", error, error_corrected)

        # Pre-scale basal/inter to best fit kon_beta across all cells before ODE inference.
        scale_theta_pre = fit_scale_theta(
            y_prot[valid], self.kon_beta[valid], basal, inter,
            ksT * self.scale_proteins, ns, samples_data=self.samples_data[valid],
        )
        basal *= scale_theta_pre       
        inter *= scale_theta_pre  

        print(np.mean(scale_theta_pre))

        self.prot = y_prot
        kon_vector = self.kon_beta.copy()
        kon_vector[:, ns:] = self._kon_ref_per_sample(y_prot, ks, inter, basal, samples_data=self.samples_data)[:, ns:]
        self.kon_theta = kon_vector

        # --- Train proliferation MLP along the paths of the recomputed trajectories ---
        if self.simulate_with_proliferation and self.R_opt is not None:  # proliferation MLP for the simulations
            if verb:
                print("[refine_network_degradations] Training ProliferationMLP on R_opt...")
            # Rate without the inference stimuli when their effects are given (added back in the simulations);
            # the MLP then takes no stimulus input
            R_target = self.R_opt if self.R_stim_offset is None else self.R_opt - self.R_stim_offset
            # Two heads (birth, death) with the prior split of the states, without the stimulus if its effects
            # are given apart (R_stim_offset)
            base = self.R_stim_offset is not None
            b_prior = self._prior_state_rate('_birth_rate_base' if base else '_birth_rate')
            d_prior = self._prior_state_rate('_death_rate_base' if base else '_death_rate')
            self.prolif_network = train_proliferation_mlp(
                self.prot, R_target, self.times_data, ns=ns, n_nodes=self.n_growth_nodes,
                with_stim=self.prolif_uses_stimulus and self.R_stim_offset is None, seed=self.seed, verb=verb,
                birth_prior=b_prior, death_prior=d_prior,
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
        inter_t = np.tile(inter, (len(times)-1,) + (1,) * inter.ndim)  # (T-1, [n_samples,] G, G, n_networks)

        # Birth rate of each state for the dilution, as in the simulations (MLP birth head or prior birth)
        b_states = self._state_birth(verb=verb)
        if self.recompute_degradations:
            # Harissa mean field: mRNA state s1*M = g * kon_beta from the data, interpolated per cell in the d1 ODE
            g_all = self._mrna_ratio() if self.simulate_full_with_harissa else None

            # Subset used by infer_ratio_d0_d1_unitary; d1 inference uses all trajectories
            # with a random minibatch of batch_size_degradations per interval and step
            cells_to_use = self.select_cells_to_use()
            if not self.use_temporal_degradations:
                # Single job: torch may use every core
                import torch
                n_threads_prev = torch.get_num_threads()
                torch.set_num_threads(os.cpu_count() or 1)
                d1, scale_theta = inference_degradation_prot(
                            self.prot[valid],
                            self.times_data[valid],
                            basal,   # 3-D (n_samples, G, n_networks) — triggers per-sample ODE
                            inter, ksT * self.scale_proteins,
                            d=self.d[1], lr=1e-2,
                            batch_size=self.batch_size_degradations,
                            n_stimuli=ns, stim_schedule=self._stim_schedule,
                            scale_proteins = self.scale_proteins,
                            samples_data=self.samples_data[valid],
                            strata=None if self.traj_cell_types is None else self.traj_cell_types[valid],
                            lambda_scale=self.lambda_scale,
                            lambda_deg=self.lambda_deg1,
                            g_ratio=g_all[valid] if g_all is not None else None,
                            birth=None if b_states is None else b_states[valid])
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

                # Interval t: for each trajectory, its pair of consecutive observed times containing [t, t+1]
                pairs_deg = trajectory_pairs(valid, len(times))

                def interval_rows(t):
                    idx = np.zeros(len(self.times_data), dtype=bool)
                    for a, b, slots in pairs_deg:
                        if a <= t < b:
                            idx[N_tot * a + slots] = True
                            idx[N_tot * b + slots] = True
                    return idx

                def run_main_inference_degradation_prot(t):
                    import torch
                    n_threads_prev = torch.get_num_threads()
                    torch.set_num_threads(n_threads_deg)
                    idx = interval_rows(t)
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
                        lambda_scale=self.lambda_scale,
                        lambda_deg=self.lambda_deg1,
                        g_ratio=g_all[idx] if g_all is not None else None,
                        birth=None if b_states is None else b_states[idx])

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
                    _sigma = self._smoothing_sigma(n_intervals)
                    if verb:
                        print(f"[refine_network_degradations] temporal smoothing sigma "
                              f"({'auto KDE' if self.smooth_degradations_sigma is None else 'fixed'}): "
                              f"{_sigma:.3f} steps, strength={strength:.2f}")
                    self.d_t[:, 1, ns_s:] = self._smooth_interior(self.d_t[:, 1, ns_s:], _sigma, strength)
                    self.d_t[:, 1, :] = np.clip(self.d_t[:, 1, :], 1e-6, None)
                    scale_theta[:, ns_s:] = self._smooth_interior(scale_theta[:, ns_s:], _sigma, strength)
                    scale_theta = np.clip(scale_theta, 1e-6, None)

                # ── Phase 3: apply (smoothed) scale_theta to basal/inter ──────
                for cnt in range(0, len(times)-1):
                    if basal_t.ndim == 4:
                        basal_t[cnt] = basal * scale_theta[cnt, None, :, None]
                    else:
                        basal_t[cnt] = basal * scale_theta[cnt, :, None]
                    inter_t[cnt] = inter * scale_theta[cnt, None, :, None]

            # ── Infer d0/d1 = ε (mRNA/protein timescale ratio) ──────────────────
            # ODE residuals as a proxy for the PDMP stochastic variance (variance-matching MoM);
            # self.ratios[cnt] = 1/ε = d0/d1, so that d0_sim = d1 * ratios = d0

            # prior_d1d0 = d1/d0 from the literature (initial self.ratios)
            prior_d1d0 = self.d[1, :] / self.d[0, :]   # shape (G,)

            use = (cells_to_use == 1) & valid
            ratios_temporal, ratios_global = infer_ratio_d0_d1_unitary(
                self.prot[use],
                self.times_data[use],
                basal_t,
                inter_t,
                ksT * self.scale_proteins,
                self.d_t[:, 1, :],              # (T-1, G) learned d1 per interval
                k1 * self.scale_proteins,       # (G,) max burst rate × scale
                n_stimuli=ns,
                stim_schedule=self._stim_schedule,
                samples_data=self.samples_data[use],
                lambda_deg=self.lambda_deg0,
                prior_eps=prior_d1d0,
                scale=self.scale_proteins,
                two_stage=self.simulate_full_with_harissa,
                verbose=verb,
                birth=None if b_states is None else b_states[use],
            )  # eps_temporal (T-1, G), eps_global (G,) — all d1/d0

            if self.use_temporal_degradations:
                for cnt in range(len(times) - 1):
                    self.ratios[cnt, :] = 1.0 / ratios_temporal[cnt]
            else:
                self.ratios[:] = (1.0 / ratios_global)[None, :]

            # ── Smooth ratios after d0/d1 computation ────────────────────────
            if self.use_temporal_degradations and n_intervals > 2 and self.smooth_degradations_sigma != 0:
                ns_s = self.n_stimuli
                self.ratios[:, ns_s:] = self._smooth_interior(self.ratios[:, ns_s:], _sigma, strength)

        self.basal, self.inter = basal, inter
        self.basal_t, self.inter_t = basal_t, inter_t

        if verb:
            basal_mean = basal.mean(axis=0) if basal.ndim == 3 else basal
            print('[refine_network_degradations]  Static network unitary',
                  [np.swapaxes(self.inter, -3, -2)[..., n] for n in range(self.n_networks)],
                  [basal_mean[:, n] for n in range(self.n_networks)])
            
        self.d[0, :ns], self.d_t[:, 0, :ns] = 1.0, 1.0
        self.d[1, :ns], self.d_t[:, 1, :ns] = 0.2, 0.2
        self.d[0, np.where(self.d[0, :] == self.d[1, :])], \
            self.d_t[:, 0, np.where(self.d_t[:, 0, :] == self.d_t[:, 1, :])] = self.d[1, np.where(self.d[0, :] == self.d[1, :])] + 1e-6,\
            self.d_t[:, 1, np.where(self.d_t[:, 0, :] == self.d_t[:, 1, :])] + 1e-6


    def _smoothing_sigma(self, n_intervals, verb=False):
        """Heat-kernel width (in inference intervals) of the temporal rates: smooth_degradations_sigma, or
        automatic (KDE bandwidth by cross-validation, as for the smoothing of the inferred rates); 0 = none."""
        if self.smooth_degradations_sigma is not None:
            return float(self.smooth_degradations_sigma)
        if n_intervals < 2:
            return 0.0
        t_idx = np.arange(n_intervals, dtype=float).reshape(-1, 1)
        bw_grid = np.logspace(-1, np.log10(n_intervals / 2.0 + 0.1), 30)
        cv = LeaveOneOut() if n_intervals <= 5 else 5
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bw_grid}, cv=cv)
        grid.fit(t_idx)
        return float(grid.best_params_['bandwidth'])

    @staticmethod
    def _smooth_interior(x, sigma, strength):
        """Temporal smoothing (axis 0) of per-interval rates; the first and last intervals keep their
        inferred values (a one-sided kernel would pull them towards their only neighbours)."""
        out = (1 - strength) * x + strength * gaussian_filter1d(x, sigma=sigma, axis=0, mode='nearest')
        out[[0, -1]] = x[[0, -1]]
        return out

    def _transfer_weights(self, times_train, times_sim):
        """
        (n_sim_intervals, n_train_intervals) weights giving the rates of the simulated intervals from the
        piecewise-constant inference rates: each simulated interval takes the rates of the inference
        intervals it overlaps, weighted by the overlap duration (a subdivision keeps the rates of its
        inference interval); constant extrapolation outside the inference times. Rows sum to 1.
        """
        tt = np.asarray(times_train, dtype=float)
        ts = np.asarray(times_sim, dtype=float)
        J = len(tt) - 1
        lo, hi = tt[:-1].copy(), tt[1:].copy()
        lo[0], hi[-1] = -np.inf, np.inf  # constant extrapolation
        W = np.zeros((len(ts) - 1, J))
        for k in range(len(ts) - 1):
            a, b = ts[k], ts[k + 1]
            if b > a:
                W[k] = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None)
            else:  # zero-length interval: rates in force at a
                W[k, min(int(np.searchsorted(tt, a, side='right')) - 1, J - 1) if a >= tt[0] else 0] = 1.0
            W[k] /= W[k].sum()
        return W

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
        inter_t = np.zeros((len(times)-1,) + inter_t_train.shape[1:], dtype=float)  # [n_samples,] G, G, n_networks
        # Rates of each simulated interval from those of the inference intervals (overlap-weighted)
        W = self._transfer_weights(times_train, times)
        self.d_t[:] = np.tensordot(W, d_t_train, axes=1)
        self.ratios[:] = np.tensordot(W, ratios_train, axes=1)
        basal_t[:] = np.tensordot(W, basal_t_train, axes=1)
        inter_t[:] = np.tensordot(W, inter_t_train, axes=1)
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
        if (ks.ndim == 3 or inter_t.ndim == 5) and samples_data is None:
            raise ValueError("Per-sample mixtures / networks need samples_data (data_samples.npy) to simulate")
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
        # Dilution of the proteins at the birth rate of each simulated cell, the prior birth as in the trajectories
        # and the refit of d1: its regression on the state (MLP) if the simulation uses it, else prior birth of the
        # trajectory slot of the cell at the inference interval containing the simulated one (slot followed through
        # the resampling)
        from ..tools.estimate_proliferation import split_net_change
        _mlp_birth = _prolif_fn is not None and hasattr(self.prolif_network, 'predict_birth')
        b_traj = d_traj = slot = None
        if self.protein_dilution and not _mlp_birth:
            bp, dp = self._prior_state_rate('_birth_rate'), self._prior_state_rate('_death_rate')
            T_tr = len(times_train)
            if bp is not None and len(bp) % T_tr == 0 and len(bp) // T_tr >= N:
                b_traj = bp.reshape(T_tr, -1)[:, :N]
                d_traj = (dp if dp is not None else np.zeros_like(bp)).reshape(T_tr, -1)[:, :N]
                slot = np.arange(N)
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

            def rate_effects(path):
                """(N, Q) part of the net rate from the RATE effects of the perturbation stimuli and, with the
                MLP, from the inference stimuli (their effects given apart), at the states path (N, Q, G)."""
                out = np.zeros(path.shape[:2])
                for eff in _rates:
                    # delta x score along the path, scaled by the stimulus value over the interval
                    u = _sample_values(eff, times[cnt + 1], s_cells)  # (N,) value of each cell's sample
                    if not u.any():
                        continue
                    if eff['weights'] is None:
                        score = np.ones(path.shape[:2])
                    else:
                        score = np.clip(path * eff['scale'], 0, 1) @ eff['weights']
                    out = out + (u * eff['delta'])[:, None] * score
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
                    out = out + (srm.effect(counts).reshape(N, Qn, -1) * u_sim[:, None, :]).sum(axis=-1)
                return out

            # Birth rate of each cell over the interval (dilution), from its state at the start
            b_cells = None
            if self.protein_dilution and (_mlp_birth or b_traj is not None):
                P_start = prot_modified[start_index:start_index + N, ns:]
                if _mlp_birth:
                    b0 = self.prolif_network.predict_birth(P_start, stim_cells)
                    d0 = self.prolif_network.predict_death(P_start, stim_cells)
                else:
                    # Inference state at the start of the interval containing time (last one beyond)
                    k = int(np.clip(np.searchsorted(times_train, time, side='right') - 1, 0, len(times_train) - 1))
                    b0, d0 = b_traj[k, slot], d_traj[k, slot]
                delta = rate_effects(P_start[:, None, :])[:, 0]
                b_cells = split_net_change(b0, d0, delta)[0] if np.any(delta) else np.asarray(b0, dtype=float)

            def run_main_loop_for_cell(n, _basal_cells=basal_cells, _basal_t_cnt=basal_t[cnt],
                                       _stim_cells=stim_cells, _b_cells=b_cells):
                _stim_vals = _stim_cells[n]
                basal_n = _basal_cells[n] if _basal_cells is not None else _basal_t_cnt
                s_n = s_cells[n]
                # Dilution: protein decay d1 + b of the cell (PDMP: flow only, bursts unchanged; ODE: target x c)
                b_vec = np.zeros(degradations.shape[1])
                if _b_cells is not None:
                    b_vec[ns:] = _b_cells[n]
                if self.simulation_stochastic:
                    return simulate_next_prot_pdmp(
                            degradations[1, :] + b_vec,
                            ks_of(kz, s_n) * degradations[0][:, None],
                            s1_of(rescale, s_n) * (degradations[0, :] / degradations[1, :]),
                            basal_n, inter_of(inter_t[cnt], s_n), t_rec,
                            self.scale_proteins, P0=prot_modified[start_index + n, :],
                            ns=ns, stim_vals=_stim_vals,
                        )
                else:
                    return simulate_next_prot_ode(
                        degradations[1, :] + b_vec, ks_of(ks, s_n),
                        basal_n, inter_of(inter_t[cnt], s_n), t_rec,
                        self.scale_proteins, P0=prot_modified[start_index + n, :],
                        ns=ns, stim_vals=_stim_vals,
                        cfac=degradations[1, :] / np.maximum(degradations[1, :] + b_vec, 1e-300)
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
                R_path = R_path + rate_effects(path)
                log_weights = (R_path * w_growth).sum(axis=1) * delta_t
                # Population size: mean growth factor of the cells over the interval
                self.log_population[cnt + 1] = self.log_population[cnt] + float(
                    np.log(np.mean(np.exp(log_weights - log_weights.max()))) + log_weights.max())
                P_end = prot_modified[end_index:end_index + N, ns:].copy()
                for grp in groups:
                    weights = np.exp(log_weights[grp] - log_weights[grp].max())
                    src = grp[np.random.choice(len(grp), len(grp), replace=True, p=weights / weights.sum())]
                    prot_modified[end_index + grp, ns:] = P_end[src]
                    if slot is not None:  # the trajectory slot (prior birth) follows the resampled cells
                        slot[grp] = slot[src]

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
        if np.ndim(self.inter) == 4:
            raise ValueError("network conditions (per-sample networks) are not supported by the Harissa simulation")
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
        # Rates of each simulated interval from those of the inference intervals (overlap-weighted)
        W = self._transfer_weights(times_train, times)
        self.d_t[:] = np.tensordot(W, d_t_train, axes=1)
        self.ratios[:] = np.tensordot(W, ratios_train, axes=1)
        basal_t[:] = np.tensordot(W, basal_t_train, axes=1)
        inter_t[:] = np.tensordot(W, inter_t_train, axes=1)
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


    def simulate_network(self, times, verb=True, stimulus_schedule=None, schedule_times=None):
        """
        Simulate the protein trajectories using the final inferred network.

        schedule_times : times of the rows of stimulus_schedule (step function, a value holding from its
            time on); None = the inference times (rows of the inference schedule).
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
                np.sort(np.unique(times)), stimulus_schedule,
                times_ref=times_train if schedule_times is None else schedule_times)
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



    def _mode_masses(self, resp, x, vect_t, times, ks, c, depth):
        """Mode masses per time, as _assign_basins(final=True) of the mixture (mean-preserving with mean_forcing_em)."""
        out = {}
        for t in times:
            m = vect_t == t
            if not self.preserve_mean_values:
                p = resp[m].sum(axis=0)
                out[t] = p / (p.sum() + EPS)
            else:
                out[t] = _compute_nu_with_temporal_constraint(resp[m], x[m], ks, c, len(ks), self.mean_forcing_em,
                                                               s=None if depth is None else depth[m])
        return out

    def _posteriors_and_masses(self, x, vect_t, ks, c, pi_zero=None, zi=None, depth=None):
        """
        Posteriors and mode masses per time of cells classified with fixed modes (ks, c), as in the mixture fit:
        masses from uniform-prior posteriors, their time mean as prior of the posteriors, final masses per time.
        """
        times = np.sort(np.unique(vect_t))
        resp0, _ = predict_resp(x, ks, c, pi_zero=pi_zero, zi=zi, s=depth)
        nu0 = self._mode_masses(resp0, x, vect_t, times, ks, c, depth)
        pi_glob = np.sum([nu0[t] * np.mean(vect_t == t) for t in times], axis=0)
        resp, _ = predict_resp(x, ks, c, pi=pi_glob, pi_zero=pi_zero, zi=zi, forcing=self.mean_forcing_em, s=depth)
        return resp, self._mode_masses(resp, x, vect_t, times, ks, c, depth)

    def fit_mixture_test(self, data_rna, ks, c, verb=False, depth=None):
        """Classify test cells into mixture modes using fixed kinetic parameters (ks, c, pi_zinb of the training).

        Sets self.modes, self.proba, self.proba_init, and self.pi_init (mode masses per time, computed on the test
        cells as on the training ones: mean-preserving with mean_forcing_em, never the training masses).
        """
        ns = self.n_stimuli
        N_cells, G_tot = data_rna.shape
        vect_t = data_rna[:, 0]
        times = np.sort(np.unique(vect_t))

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

            # Posteriors and mode masses as in the training mixture, from the test cells only
            proba, pi_g = self._posteriors_and_masses(data_rna[:, g], vect_t, ks[:ng, g], c[g],
                                                      pi_zero=pi_zero_g, zi=zi_flag, depth=depth)
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
        a3, pz3 = self.a, self.pi_zinb
        N, G_tot = data_rna.shape
        M = a3.shape[1] - 1
        modes, proba, proba_init = np.zeros((N, G_tot)), np.zeros((N, G_tot, M)), np.zeros((N, G_tot, M))
        vect_t = data_rna[:, 0]
        times = np.sort(np.unique(vect_t))
        pi_sum = None
        for s_idx, sid in enumerate(samples_id):
            m = vect_samples_id == sid
            self.a, self.pi_zinb = a3[min(s_idx, len(a3) - 1)], pz3[min(s_idx, len(pz3) - 1)]
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

    def _assign_calibrated_basins(self, train, vect_t, times, ks, ks_max, ks_cells, G_tot, weight_prob):
        """
        Basins of the test cells, as the last basin update of the training with the network term replaced: one EMD
        per gene (per time with temporal_basins), modes imposed, cost -(log p + (1 - weight_prob) log q) with p the
        mixture probabilities (/ max) and q (/ max) the mode predicted by a multinomial logistic regression of the
        final training basins on the training mixture probabilities (per time with temporal_basins), i.e. how
        the network settled undecided cells in the training. Target masses as in the training (force_basins), from
        the test cells. train: proba_init (cells, G_tot, modes), basins_final (cells, G_tot), cell_times.
        """
        from sklearn.linear_model import LogisticRegression
        ns, n_cells = self.n_stimuli, len(vect_t)
        t_tr = np.asarray(train['cell_times'])
        t_tr_u = np.unique(t_tr)

        def posterior(p):
            return p / np.maximum(np.sum(p, axis=1, keepdims=True), EPS)

        def fit_lr(X, y):
            # None when the training assignment cannot be learned (too few cells or a single basin)
            if len(y) < 30 or len(np.unique(y)) < 2 or np.bincount(y).max() > len(y) - 3:
                return None
            return LogisticRegression(C=1.0, max_iter=300).fit(np.log(np.clip(X, 1e-4, 1)), y)

        def predicted(lr, Xnew):
            # Modes predicted by the regression (/ max, floor 1e-3 as the network term); 1: no constraint
            if lr is None:
                return np.ones_like(Xnew)
            P = np.zeros_like(Xnew)
            P[:, lr.classes_] = lr.predict_proba(np.log(np.clip(Xnew, 1e-4, 1)))
            P = np.maximum(P / np.max(P, axis=1, keepdims=True), 1e-3)
            return P / np.max(P, axis=1, keepdims=True)

        def run_gene(g):
            l_max = 1 + int(np.argmax(ks_max[g, :]))
            obj = (ks_cells[:, g, :l_max] if ks_cells is not None
                   else np.repeat(ks[g, :l_max][None, :], n_cells, axis=0))
            Xtr = posterior(train['proba_init'][:, g, :l_max])
            ytr = np.minimum(np.asarray(train['basins_final'][:, g], dtype=int), l_max - 1)
            lr_all = fit_lr(Xtr, ytr)
            tmp_proba = np.zeros_like(self.proba[:, g])
            tmp_modes = np.zeros(n_cells)
            for t_i in (times if self.temporal_basins else [None]):
                idx = np.flatnonzero(np.ones(n_cells, bool) if t_i is None else vect_t == t_i)
                if not len(idx):
                    continue
                proba = self.proba_init[idx, g, :l_max].copy()
                post = posterior(proba)
                proba /= np.max(proba, axis=1, keepdims=True)
                lr = lr_all
                if t_i is not None:
                    nu = self.pi_init[g - ns][t_i][:l_max] * self.force_basins + post.sum(axis=0) * (1 - self.force_basins)
                    # Regression at the nearest training time, unless it misses basins of the pooled one
                    m = t_tr == t_tr_u[np.argmin(np.abs(t_tr_u - t_i))]
                    lr_t = fit_lr(Xtr[m], ytr[m])
                    if lr_t is not None and (lr_all is None or len(lr_t.classes_) == len(lr_all.classes_)):
                        lr = lr_t
                else:
                    nu = np.sum([self.pi_init[g - ns][t] * np.sum(vect_t == t) / n_cells for t in times], axis=0)[:l_max] \
                        * self.force_basins + post.sum(axis=0) * (1 - self.force_basins)
                nu /= np.sum(nu)
                dist = np.clip(-(np.log(proba) + (1 - weight_prob) * np.log(predicted(lr, post))), 0, 100)
                mu = np.ones(len(idx)) / len(idx)
                k = np.argmax(ot.emd(mu, nu, dist, numItermax=int(1e7)), axis=1)
                tmp_proba[idx, k] = 1
                tmp_modes[idx] = obj[idx, k]
            return tmp_proba, tmp_modes

        results = Parallel(n_jobs=-1)(delayed(run_gene)(g) for g in range(ns, G_tot))
        for g, (tmp_proba, tmp_modes) in zip(range(ns, G_tot), results):
            self.proba[:, g, :], self.modes[:, g] = tmp_proba, tmp_modes

    def infer_test(self, data, vect_samples_id=None, verb=True, stimulus_schedule=None,
                   basal_ref=None, transition_rates=None, time_key='time', context=None, train=None):
        """
        Test-set inference with the network fixed, in a single pass. train: dict of the training arrays (see
        _alpha_pool and _assign_calibrated_basins). The basins come from one EMD of the mixture probabilities,
        calibrated on how the training assigned them (network included); the trajectories are computed once with
        the final regularization of the training (context), the alphas being copied from the nearest training
        states.

        basal_ref : (n_samples, G_tot, n_networks) array or None
            Per-sample KO/OV prior (±100 entries) used to build kov_cell_mask.
        transition_rates : DataFrame or array or None
            Cell-type transition rate matrix for OT cost adjustment.
        """
        if train is None:
            raise ValueError("infer_test needs the training arrays (rerun infer_network_structure to save basins_final)")
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

        def force_kov():
            # KO cells at the lowest mode, OV cells at the highest one
            if kov_cell_mask is None:
                return
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

        if context is None:
            context = {'n_iter_reg': self.min_n_loops, 'weight_init': 0.0,
                       'weight_prob': max(.96**(self.min_n_loops - 1), .1)}
        self._assign_calibrated_basins(train, vect_t, times, ks, ks_max, ks_cells, G_tot, context['weight_prob'])
        force_kov()

        nb_cells = np.zeros((len(samples_id), len(times)), dtype=int)
        for s, sid in enumerate(samples_id):
            for t, time in enumerate(times):
                nb_cells[s, t] = np.sum((vect_t[vect_samples_id == sid] == time))

        if verb:
            print("[infer_test] Cell counts per sample/timepoint and genes:\n", nb_cells, G_tot)

        # Timepoints of each sample (samples may miss some of the timepoints)
        first_t = [int(np.argmax(nb_cells[s] > 0)) for s in range(len(samples_id))]

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
            minimal_repetition_choice(nb_cells[s, first_t[s]], N_full[s],
                                      labels=self._t0_cell_types(vect_t, vect_samples_id, sample, times[first_t[s]]))
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
            context=context,
            alpha_pool=self._alpha_pool(train, samples_id, train['sample_ids']),
        )


    def fit(self, data_rna, intensity_prior=100, refilter=5.0, max_iter_kinetics=100, verb=True):

        self.fit_mixture(data_rna, min_components=2, max_components=2, refilter=refilter, max_iter_kinetics=max_iter_kinetics)
        self.fit_network(data_rna, intensity_prior=intensity_prior, verb=verb)
        # self.refine_network_degradations()