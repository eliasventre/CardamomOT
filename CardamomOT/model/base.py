
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

import copy
from functools import partial
import numpy as np
import ot
import seaborn as sns
import multiprocessing as mp
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from scipy.linalg import expm
from ..config import resolve_cell_type_obs, CELL_TYPE_OBS_KEYS
from ..inference.trajectory import ks_of, s1_of, s1_rows, inter_of, birth_alpha, prot_bridge, prot_bridge_end
from ..inference.mixture import _compute_nu_with_temporal_constraint
from ..inference.simulations import simulate_next_mrna_pdmp, ChromatinModel, chrom_gate, chrom_values
from ..inference import (inference_network_multi, active_regulators, PrevProt, PathProt, path_weights, signed_floor, sigmoid_amplitudes,
                        sigmoid_target, gene_weights, filter_network,
                        minimal_repetition_choice, find_next_prot, find_next_prot_dilution, my_otdistance, count_errors,
                        kon_ref_vector, inference_alpha, inference_alpha_1thread,
                        NegativeBinomialMixtureEM, predict_resp,
                        simulate_next_prot_ode, simulate_next_prot_pdmp,
                        train_proliferation_mlp, quadrature,
                        seed_everything, seeded_call, task_seed,
                        stratified_order, grouped_partition)
from ..inference.depth_model import set_depth_model

np.set_printoptions(precision=3, suppress=True)
EPS=1e-16

# Parameters that moved: a value for all samples would claim an information that is not known for all
# Former names of parameters (workbooks written before the renaming are still read)
RENAMED_PARAMETERS = {'cell_depth_for_embeddings': 'cell_depth_for_representation',
                      'simulate_full_with_harissa': 'simulation_mrnas_direct',
                      'condition_kinetics': 'degradation_rates_per_condition'}

REMOVED_PARAMETERS = {
    'n_networks': 'a single network: kon is a sigmoid between the lowest and the highest mode',
    'adapt_size_network': 'a single network: kon is a sigmoid between the lowest and the highest mode',
    'compute_with_proba': 'the network is always fitted on the kon values',
    'report_net_index': 'a single network',
    'loss_norm': 'the network is always fitted with the cross-entropy on kon',
    'update_modes': 'the basins are always updated',
    'transform_proba': 'the mixture posteriors are never modified',
    'recompute_degradations': 'the degradation rates are always refitted for the simulations',
    'mrna_driven_proteins': 'set by simulation_stochastic: stochastic simulations keep the trajectories and network of fit_network',
    'finish_by_determinist': 'simulations are entirely stochastic or entirely deterministic (simulation_stochastic)',
    'bridge_mode_ridge': 'the equivalent modes of the bridges are no longer used by refine_network_degradations',
    'lambda_ratio_lit': 'd0 / d1 is learned by the free-running fit of refine_network_degradations',
    'noise_aware_degradations': 'always used for the stochastic simulations',
    'fit_degradations_in_loop': 'renamed refine_in_loop (free-running fit of the rates, and the network with refine_theta_with_sde)',
    'deg_loop_period': 'renamed refine_in_loop_period',
    'deg_loop_fixed': 'd1 and d0 / d1 are both learned by refine_in_loop',
    'lambda_deg_loop1': 'rappel of the rates of refine_in_loop: sde_lambda_rates',
    'network_interval_scale': 'the scale of theta is learned with d1 in the free-running fit (fit_scale_theta), never in the network fit',
    'batch_size_degradations': 'the free-running fit of refine_network_degradations draws sde_batch_size trajectories',
    'lambda_scale': 'rappel of the scale of theta: sde_lambda_rates',
    'lambda_deg0': 'rappel of d0 / d1: sde_lambda_rates',
    'lambda_deg1': 'rappel of d1: sde_lambda_rates',
    'smooth_degradations_sigma': 'the temporal rates of the free-running fit are not smoothed (rappel sde_lambda_rates)',
    'smooth_degradations_strength': 'the temporal rates of the free-running fit are not smoothed (rappel sde_lambda_rates)',
}


def drop_network_axis(arr):
    """Network array without the former trailing network axis of size 1 (basal/inter files of earlier runs)."""
    arr = np.asarray(arr)
    return arr[..., 0] if arr.ndim >= 2 and arr.shape[-1] == 1 else arr


def load_network_array(path):
    """np.load of a basal / inter file, compatible with the files of earlier runs (trailing network axis)."""
    return drop_network_axis(np.load(path))


def load_simul_parameters(cardamom_dir):
    """Simulation parameters of infer_network_simul: {basal, inter, basal_t, inter_t, d_t, ratios}, with the trajectories
    of fit_network (prot, kon_theta: its network on them), the only saved ones."""
    path = lambda name: os.path.join(cardamom_dir, name)
    return dict(basal=load_network_array(path('basal_simul.npy')), inter=load_network_array(path('inter_simul.npy')),
                basal_t=load_network_array(path('basal_t_simul.npy')),
                inter_t=load_network_array(path('inter_t_simul.npy')),
                d_t=np.load(path('degradations_temporal.npy')), ratios=np.load(path('ratios.npy')),
                prot=np.load(path('data_prot.npy')), kon_theta=np.load(path('data_kon_theta.npy')))


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


def prot_along(d1, P0, mode_init, mode_end, alpha, s, delta_t, tau, d1_end=None, alpha_b=None):
    """Protein at time tau in [0, delta_t] of the flow of find_next_prot (switch of the modes at alpha * delta_t), or
    with dilution (d1_end given) of find_next_prot_dilution (rate d1 -> d1_end at alpha_b * delta_t)."""
    if d1_end is None:
        t_sw = alpha * delta_t
        p_sw = mode_init * s + (P0 - mode_init * s) * np.exp(-d1 * np.minimum(tau, t_sw))
        return np.where(tau <= t_sw, p_sw, mode_end * s + (p_sw - mode_end * s) * np.exp(-d1 * np.maximum(tau - t_sw, 0)))
    alpha_b = alpha if alpha_b is None else alpha_b
    t1, t2 = np.minimum(alpha, alpha_b) * delta_t, np.maximum(alpha, alpha_b) * delta_t
    f = 1.0 * (alpha < alpha_b)  # 1: the modes switch first
    mode_mid = f * (mode_end * d1_end / d1) + (1.0 - f) * (mode_init * d1 / d1_end)
    rate_mid = f * d1 + (1.0 - f) * d1_end
    p1 = mode_init * s + (P0 - mode_init * s) * np.exp(-d1 * np.minimum(tau, t1))
    p2 = mode_mid * s + (p1 - mode_mid * s) * np.exp(-rate_mid * np.clip(tau - t1, 0, t2 - t1))
    return mode_end * s + (p2 - mode_end * s) * np.exp(-d1_end * np.maximum(tau - t2, 0))


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
        self.d_literature = None # literature (d0, d1) (2, G), rappel target of refine_in_loop in fit_network (set at its start)
        self.d1_eff_t = None # stochastic simulations: effective d1 (T-1, G) fitted on the ODE
        self.train_times = None # infer_test: times of the training intervals of d_t (transfer to the test intervals)
        # Mixture parameters
        self.weights = None
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
        self.basins = None   # basin (mode index) of each cell and gene, (N_cells, G_tot) int

        self.n_stimuli = n_stimuli
        self._stim_schedule = None
        self._stim_overrides = {}   # per-sample stimulus schedules {dataset_id: (times or None, values)}
        self.traj_valid = None      # (T * N,) bool: trajectory state at a time observed for its sample (None = all)
        self.couplings = None       # final soft couplings of the trajectories (list of sparse blocks, growth pass)
        self._sample_names = None   # sorted dataset_id of the run (index of a sample -> its label)
        self.sample_conditions = None  # (n_samples,) network condition index of each sample (None: one common network)
        self.condition_names = None    # labels of the network conditions (obs['network_condition'])
        self.gene_mask = None          # (n_samples, G_tot) bool: genes active in each sample (select_genes_per_condition); None = all
        self.d_samples = None          # (n_samples, 2, G_tot) rates of each sample in fit_network (refine_in_loop per condition); None = self.d

        ### Pipeline (run.sh / cardamomot pipeline): data and steps, fixed per project in Data/CardamomOT_inputs.xlsx
        self.CardamomOT_method = 'temporal'      # 'temporal': current pipeline; 'stat': CardamomOT-stat (stationary snapshot per sample, times only for the mixture; automatic without obs['time']); 'stat+temporal': to be specified
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
        self.report_normalize = False            # report UMAPs: counts normalised per cell
        self.report_log1p = True                 # report UMAPs: log1p of the counts
        self.report_n_umap = 4000                # report: maximal number of cells per stage in the UMAPs (0 = all)
        self.cell_depth_for_representation = True    # report, cell-type classifiers: counts / depth factor (if computed) for data and model draws (drawn at the depth of the cells they mimic); False = raw counts
        self.embedding_method_visualization = 'umap'           # report: 2-D embeddings with 'umap', 'pca' or 'phate'
        self.classifier_method = 'logistic'  # cell types of the model outputs (report, notebooks): 'random_forest' or 'logistic'

        ### Default behaviour
        self.seed = None # Random seed for reproducibility (main process and parallel workers); None = not seeded (runs vary)

        ## Compute mixture parameters
        self.hard_em = 1 # Do we initialize with a hard_em ?
        self.preserve_mean_values = 1 # Do we ensure temporal constraints when fitting the basins in the hard_em ?
        self.mean_forcing_em = 1.0 # at which point we force the mean correction: the higher the more
        self.force_basins = 1.0 # Do we want to ensure the means to be preserved by the NB mixture ? It may not preserve multistability
        self.temporal_basins = 1 # Is it preserved temporally ?
        self.seuil = 1e-2 # minimum for beta mixture parameters (second parameters)
        self.batch_size_mixture = 4096 # Maximum number of cells per time used for mixture calibration in the inference.
        self.depth_model = 'mean' # with depth factors: 'burst_frequency' = the depth multiplies the burst frequency, NB(k s, c) (Fano factor independent of the depth); 'mean' = it multiplies only the mean, NB(k, c / s) (dispersion k fixed, as scVI)
        self.use_depth_factor = False   # use obs['depth_factor'] (estimate_cell_depth) in the NB model and simulations; False = s_i = 1 (an existing factor is kept but ignored)
        self.compute_depth_factor = True # estimate_cell_depth computes the diagnostic and writes obs['depth_factor'] if needed; False = only reads an existing factor (never removed)
        self.allow_depth_correction = True  # estimate_cell_depth: apply the per-cell depth factor s_i when the diagnostic recommends it
        self.depth_method = 'poissonian'  # depth factor: 'group_median', 'poissonian' (Fang & Pachter, 2025), or <project>/depth_methods/<name>.py
        self.depth_method_params = {}       # parameters of the depth method
        self.depth_by_cell_type = False     # estimate s_i within (sample, time, cell type) groups; False = (sample, time) only:
                                            # cell types derive from expression, normalising within them pulls cells to their type (circular)
        self.soft_em_refinement = False # NB mixture: refine (ks, c) by soft EM with fixed basin masses
        self.use_scBoolSeq = False # If True, initialize NB mixture with scBoolSeq binarization instead of hard EM
        self.scboolseq_confidence = 0.6 # GMM posterior probability threshold for cell label assignment (lower → fewer NaN)
        self.scboolseq_min_cells_per_label = 10 # min cells per label (0 and 1) required to use scBoolSeq path; otherwise falls back to normal EM
        self.scboolseq_dropout = False

        ## Infer network (single network: kon = sigmoid between the lowest and the highest mode of each gene)
        # Loop for the inference
        self.min_n_loops = 10 # minimal number of iterations in inference loops
        self.count_max = 5 # Stopping criteria
        self.max_iter = 40 # max iteration for main loop
        # Trajectory inference with OT
        self.stopThr_init = 1e-7 # initial tolerance for sinkhorn algorithm
        self.batch_size_traj = 1024 # Maximum number of cells used per time point per sample for solving EOT problems in the inference.
        self.batch_size_traj_exp = 1024 # Target size of the real (experimental) cell batches matched against each reconstruction batch; decoupled from batch_size_traj so the OT target pool can stay well-sized even when batch_size_traj is kept small for cost.
        self.n_strata_traj = 10 # Real-cell batches are stratified by cell type if available, else by this number of k-means clusters (0 = uniform batches)
        self.unbalanced_reg = 10 # UOT relaxation of the target marginal if > 0 (reg_m = unbalanced_reg x typical excess cost, its exponent then frozen from min_n_loops on), reached gradually from balanced OT at the first iteration; OT if 0. Higher = more constrained marginals (uncertainty of the proliferation rates)
        self.init_entropic_noise = 1.0 # Initial entropic regularization of the OT, relative to the typical excess cost of a pair (decreasing with the iteration)
        self.intensity_prior = 100 # mRNA term of the OT cost while n_iter <= intensity_prior (decreasing weight); < 1: random couplings at the first iteration, then network term only
        self.quant_samples = .95 # Quantile of cells number per sample to use for inference
        # General parameters to calibrate protein reconstruction
        self.scale_proteins = 1 # Eventually rescale protein values (recommended:1-2)
        self.scale_mrnas = 100 # Eventually rescale mRNA values (recommended:100)
        self.fact_simple = 2 # slight transformation for constrative modes in learning phase
        # Network inference with scipy
        self.scale_pen = 20 # Error that is expected = 1/scale_pen
        self.weight_prev = .4 # max = .5 to not withdrawn the inference on timepoints, allows the calibration to incorporate some "flow-matching" method
        self.exact_flow_matching = False # True: the flow-matching term (weight weight_prev, same schedule) compares the target mode with the sigmoid filtered by the mRNA memory d0 exp(-d0 (t_b - s)) along the bridge of the interval (flow_matching_nodes nodes), instead of a single state after the switch; the end-state term is kept
        self.flow_matching_nodes = 5 # with exact_flow_matching: nodes per interval (start + interior nodes of the bridge, the end state being the row itself)
        self.batch_size_network = 100 # Cells per network sub-sample (stratified by time and sample); raised to 10 x the parameters of a target-gene fit if lower; None = that floor
        self.n_network_fits = 10 # Theta = mean of min(n_network_fits, 1 + n_states // batch_size) fits on disjoint subsamples, at every network update
        # Inference of alpha = switch moment between each timepoint and modes
        self.alpha_threshold = .6 # max = 1, thershold for important transition to update alpha full
        self.n_pas = 25 # number of timesteps between timepoints for inference of alpha
        self.force_n_pas = False # if True, use n_pas as-is instead of auto-flooring it to the inter-timepoint interval length
        self.refine_in_loop = False # True: in fit_network, between the network fit and alpha, refine_in_loop_epochs steps of the free-running fit of refine_network_degradations on its longest windows (model of the simulations: ODE, protein-only PDMP with simulation_stochastic, explicit mRNA with simulation_mrnas_direct): constant d1 (d1_eff) and d0 / d1 (rappel sde_lambda_rates towards the literature), no scale, the network too with refine_theta_with_sde; the rates (converted to the full d0, d1) then drive the trajectories
        self.refine_in_loop_start = 2 # with refine_in_loop: first iteration of the updates
        self.refine_in_loop_period = 1 # with refine_in_loop: iterations between two updates (the last cycle is completed before stopping)
        self.refine_in_loop_epochs = 1 # with refine_in_loop: optimizer steps at each update (windows of sde_max_horizon only)
        self.refine_in_loop_decay = 0.95 # with refine_in_loop: rappel towards the literature sde_lambda_rates x decay^k at the k-th update (k = 0 first), so that the network learned along the iterations can correct wrong literature rates; 1 = constant rappel
        self.fit_scale_theta = True # refine: scale of theta per target gene (and interval with use_temporal_degradations) learned in the free-running fit (within [1/5, 5]); False: scale 1
        self.refine_theta_with_sde = True # refine: True = network of fit_network refined (support and signs kept) with d1, d0 / d1 and the scale of theta by the free-running fit of the simulation model (ODE / SDE) on the observed modes; False = network refitted on the trajectories on the modes (noise-aware when stochastic), then fixed in that fit
        self.sde_epochs = 30 # free-running fit: optimizer steps per horizon of the curriculum (windows of 1 to T-1 intervals)
        self.sde_lr = 1e-2 # free-running fit: Adam learning rate
        self.sde_lambda_theta = 0.1 # free-running fit: rappel of theta / basal towards the starting network (relative L2)
        self.sde_lambda_rates = 1.0 # free-running fit: rappel of log d1, log(d1 / d0) and log scale towards their starting values (literature)
        self.sde_n_sub = 20 # free-running fit: integration steps per interval
        self.degradation_rates_per_condition = True # with network conditions: network, d1, d0 / d1 and the scale of theta refined separately per condition (its samples and genes, its network as rappel; stored per sample); False: one shared fit from the mean network (forced True with select_genes_per_condition)
        self.sde_max_horizon = 9 # free-running fit: longest window of the curriculum (intervals; 9 = chunks of 10 timepoints, starting at every timepoint)
        self.sde_windows = 10 # free-running fit: window starts drawn at each step (None = all; bounds the cost with many timepoints)
        self.sde_batch_size = 256 # free-running fit: trajectories per sample and window at each step (and observed cells per time)
        self.inference_stochastic = False # without inference_mrnas_direct: the network fits of fit_network average kon over the protein noise of the PDMP at d1_eff (literature d0 / d1, one stage)
        self.inference_mrnas_direct = True # True: fit_network trajectories driven by the observed mRNA (shrunk modes, adaptive_shrinkage); False: by the modes only
        self.mrna_bridge = True # True: proteins from the linear mRNA bridges between observed mRNAs (prot_bridge_end, d0 >= min_ratio d1); False: shrunk modes (adaptive_shrinkage)
        self.bridge_sampling = True # with mrna_bridge: end proteins sampled from the bridges (noise shared by the targets of a source, in the OT cost and the trajectories); False: bridge means
        # Penalization/prior information
        self.stimulus = 1.0 # 1 if we simulate with a stimulus. If not we can penalize the stimulus with a value between 1 and 0: 0 = no sitmulus
        self.prior_network_pen = 1.0 # 1 if we don't use prior information. If not we can penalize the non-existing age in prior network with values between 1 and 0: 0 = impossible edge
        self.constrain_basal_uniform = 1.0 # >= 0 penalty strength that pushes per-sample basals to be equal (ignores samples pinned by KO/OV basal_ref)
        self.network_condition_pen = 1.0 # >= 0, with >= 2 values of obs['network_condition']: fused L1 penalty on the deviations of each condition's network from the shared one minus those of inter_ref from its mean, relative to the sparsity penalty (large = reference deviations, i.e. a common network without per-condition inter_ref; 0 = independent networks)
        self.hard_forcing_ref = False # if True, constrain all network params to ±ref_constraint_pct around inter_ref
        self.ref_constraint_pct = 0.01 # fractional tolerance around inter_ref values for bounds (used when hard_forcing_ref=True)
        self.seuil_zero_min_ref = 5e-2 # reference values (inter_ref) with |v| <= this are read as absent edges; also min |theta| of sign-forced edges
        # Filtering
        self.filter_network = 1 # Do we filter the network ? It also builds a temporal network using the filter criterium
        self.seuil_min_network_intensity = 1e-2 # post-inference filter: edges with |theta| below are removed (max with seuil_zero_min_ref if hard_forcing_ref)
        self.seuil_min_network_variations = 0 # post-inference filter: min kon variation when removing an edge (0 = intensity threshold only)

        ## Compute degradations after inference
        self.traj_cell_types = None # Cell type of the real cell behind each trajectory state (set by fit_network)
        self.use_temporal_degradations = 1 # If so, compute temporal degradation rates for simulations ?

        ## Simulations
        self.simulation_stochastic = True # 1 if we simulate Bursty-like proteins, 0 if deterministic limit for proteins
        self.min_ratio = 1.5
        self.max_ratio = 50.0
        self.simulation_mrnas_direct = False  # True: the PDMP simulations carry explicit mRNA (mRNA bursts, proteins translated from them) and return mRNA counts; refine then fits d1 on mRNA-driven trajectories and d0 from d1_eff (protein-only fit on the modes). The ODE (simulation_stochastic = 0) stays on the proteins
        self.chromatin = None  # Chromatin regions of the simulations: dict(w (G, R), basal_c (R,), inter_c (G, R), kbar_c (R,), optional k_close_const, c_levels, diploid, c0), see ChromatinModel; None = no region

        ## Gene selection (select_genes.py, select_genes = True): terminals + global network + directed Steiner tree
        self.num_max_genes = 100         # budget: number of selected genes (stimuli excluded)
        self.n_query_genes = 10          # at most this many genes of genes_queries (gene_lists sheet) (round robin over time/cell-type DE groups)
        self.n_driver_genes = 10         # fate drivers of run_classical_OT (classical_OT/fate_drivers.csv) required in the selection (0 = none); n_query + n_driver + n_entropy < num_max_genes
        self.entropy_preselection = True  # queries, fate drivers and network genes (Steiner, closure) kept only among the entropy candidates (n_top_entropy) and the perturbed genes
        self.n_entropy_genes = 30        # at least this many entropy genes (Gandrillon KD & MDE), same round robin; n_query + n_entropy < num_max_genes
        self.n_top_entropy = 200         # top genes per transition for KD and for MDE (Gandrillon's TOP_N)
        self.n_cells_entropy = 1000      # cells per timepoint for the BUB entropy (its matrices are (N+1)^2)
        self.n_hvg_selection = 5000      # highly variable genes forming the network universe (terminals always kept)
        self.network_method = 'otvelo_granger'  # global network: 'otvelo_granger', 'otvelo_corr', 'wot_granger' (through the run_classical_OT couplings), or <project>/network_methods/<name>.py
        self.network_method_params = {}  # parameters of the network method (otvelo: n_cells, n_pcs, eps, alpha; granger: + k_candidates, en_alpha ('auto' = min(1, sqrt(50 / G))), l1_ratio, scale, stim_weight)
        self.k_in_steiner = 20           # strongest incoming edges kept per gene in the Steiner graph
        self.k_stim_steiner = 20         # direct targets kept per stimulus in the Steiner graph
        self.min_edge_prob = 0.05        # edges with probability (1 - FDR against the permuted-data network) below are dropped
        self.edge_prior = 0.9            # each Steiner edge costs -log(prob * edge_prior): favours short paths among equally probable ones
        self.null_network = 'hybrid'     # edge probabilities vs permuted data: 'hybrid' (gene edges within (sample, time), stimulus all cells), 'within_time', 'all_cells'
        self.select_genes_per_condition = False  # with >= 2 network conditions: genes selected separately per condition (data = union, uns['condition_genes']); each condition fitted on its own genes, fused penalty on the shared edges only
        self.sample_network_combination = 'auto'  # several samples, gene selection: 'consensus' (edge probabilities averaged, shared network), 'any' (probabilistic OR + closure balanced over the samples, independent condition networks); 'auto' = 'any' with >= 2 network conditions and network_condition_pen = 0
        self.selection_edge_prob = 0.6   # gene_selection_report: is_regulated_by / regulates list the edges inside the selection with probability >= this (and literature-feasible)
        self.closure_min = 0.5           # closure: add the gene bringing the most probable regulation (sum of w) to the selection while >= this
        self.literature_selection = True  # gene selection: edge probabilities reweighted by OmniPath feasibility, whatever the prior
        self.literature_depth = 3        # max path length in the literature graph (last edge TF -> target)
        self.literature_weight = 1.0     # exponent on the data-calibrated literature likelihood ratio (0 = data only)
        self.min_entropy_change = 0.1    # variability floor of the selection: max BUB-entropy change between consecutive times (bits, Gandrillon MDE)
        self.min_nb_separation = 0.1    # ... and separation of the extreme NB modes at the mixture initialization (times / cell types, batch_size_mixture cells per time); also the minimal separation of consecutive modes of the fitted mixture (else fewer modes, down to one)
        self.max_free_params = 10000     # change=1 with ref=1 and prior 0 (hard prior): gene budget set so that the literature prior leaves this many free network parameters
        self.literature_resources = 'extended'  # 'extended' (OmniPath, CollecTRI + PathwayExtra, KinaseExtra, LigRecExtra, DoRothEA A-D, TFtarget) or 'core' (OmniPath + CollecTRI)

        ## Multi-sample integration of the mixture (obs['dataset_id']; no effect with a single sample)
        self.integrate_samples = 0.0             # in [0, 1]: per-sample mixtures pushed towards the common target (1 = equal parameters, 0 = own fits)
        self.ref_sample_integration = None       # dataset_id of the reference sample; None = cell-weighted average of mode means (keeps total counts)
        self.min_cells_integration = 200         # smaller samples are not fitted alone (raw counts, classified with global modes)
        self.min_mode_weight_integration = 0.001  # a sample mode is supported if its weight is at least this (also the minimal weight of a mode of the fitted mixture)...
        self.min_mode_ratio_integration = 0.6    # ...and the ON/OFF mean ratio at least this; else the gene is not integrated for that sample (also required by the fitted mixture)
        self.integration = None                  # per-sample parameters and integrated genes, set by fit_mixture_samples

        ## Proliferation
        self.simulate_with_proliferation = True # apply branching process in simulate_trajectories
        self.ProliferationMLP_per_condition = True  # with network conditions: one proliferation MLP per condition (its samples and genes); False: one shared MLP (forced True with select_genes_per_condition)
        self.protein_dilution = True             # proteins diluted at the birth rate b of each cell: dP/dt = d1 u - (d1 + b) P (obs['proliferation_birth_rate']; independent of simulate_with_proliferation)
        self.prolif_uses_stimulus = True         # inference stimuli are inputs of the ProliferationMLP, R(u, P) (no effect if constant over the intervals)
        self.prolif_network = None               # ProliferationMLP trained in refine_network_degradations (if simulate_with_proliferation or protein_dilution), not a parameter
        self.R_opt = None                        # net growth rate of each trajectory state over the next interval (NaN at last time), from the final growth OT pass
        self.R_stim_offset = None                # part of R_opt due to the inference stimuli (RATEk of perturbation_inference), removed before training the proliferation MLP
        self.stimulus_rate_model = None          # StimulusRateModel: stimulus part of the net rate from mRNA, added back in the branching simulations
        self.growth_reg_source = 2.0             # source-marginal relaxation (x typical excess cost of the batch) of the growth OT pass: small = data-driven but noisy, large = prior kept
        self.n_growth_iter = 1                   # WOT-style growth iterations (source weights <- row marginals); more iterations amplify the noise
        self.n_growth_nodes = 5                  # quadrature nodes per interval to integrate R along paths (MLP training and branching simulation)
        self.inter_simul_ref = None              # optional inter reference for the network refits of refine_network_degradations

        if n_genes is not None:
            G = n_genes + n_stimuli
            # Default degradation rates
            self.d = np.zeros((2,G))
            self.d[0] = np.log(2)/9 # mRNA degradation rates
            self.d[1] = np.log(2)/46 # protein degradation rates
            # Default network parameters
            self.a = np.zeros((3,G))
            self.basal = np.zeros((1, G))
            self.inter = np.zeros((G, G))
            self.inter_t = np.zeros((1, G, G))
            self.ref_network = np.ones((G, G))
        

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

    def set_gene_mask(self, adata, sample_ids=None, verb=True):
        """Genes active in each sample (uns['condition_genes'] of select_genes_per_condition, sorted dataset_id or
        sample_ids); None when every gene is shared."""
        from ..condition_genes import gene_mask
        self.gene_mask = gene_mask(adata, sample_ids, self.n_stimuli)
        if verb and self.gene_mask is not None:
            print("[CardamomOT] Genes per condition: active genes per sample "
                  f"{(self.gene_mask[:, self.n_stimuli:]).sum(axis=1).tolist()} of {self.gene_mask.shape[1] - self.n_stimuli}")
        return self.gene_mask

    def active_genes(self, s_idx):
        """(G_tot,) bool: genes active in sample s_idx (all without per-condition genes)."""
        if self.gene_mask is None:
            return None
        return self.gene_mask[min(int(s_idx), len(self.gene_mask) - 1)]

    def cell_gene_mask(self, samples):
        """(N, G_tot) bool: genes active for each row of sample indices samples; None without per-condition genes."""
        if self.gene_mask is None:
            return None
        return self.gene_mask[np.minimum(np.asarray(samples).astype(int), len(self.gene_mask) - 1)]

    def per_condition_fits(self):
        """(degradation_rates_per_condition, ProliferationMLP_per_condition) in force: both with select_genes_per_condition."""
        forced = bool(self.select_genes_per_condition) or self.gene_mask is not None
        return forced or bool(self.degradation_rates_per_condition), forced or bool(self.ProliferationMLP_per_condition)

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
        """{condition label: (G, G) network} of a per-sample inter (S, G, G); {None: inter} if common."""
        if np.ndim(inter) < 3:
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
        if self.sample_conditions is None or np.ndim(inter) < 3:
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
        stack = np.stack(list(nets.values()))          # (C, G, G)
        # Edges existing in each condition (both genes selected there, select_genes_per_condition); NaN elsewhere
        if self.gene_mask is not None:
            first = [int(np.flatnonzero(self.sample_conditions == k)[0]) for k in range(len(nets))]
            act = self.gene_mask[first]
            stack = np.where(act[:, :, None] & act[:, None, :], stack, np.nan)
        exists = np.isfinite(stack)
        np.save(files[1], np.nan_to_num(np.nanmedian(np.where(exists.any(axis=0), stack, 0.0), axis=0)))
        names = list(gene_names) if gene_names is not None else [str(g) for g in range(stack.shape[1])]
        rows = []
        # Edges compared over the conditions where they exist (at least two)
        lo, hi = np.nanmin(np.where(exists, stack, np.inf), axis=0), np.nanmax(np.where(exists, stack, -np.inf), axis=0)
        differ = (exists.sum(axis=0) >= 2) & (hi - lo > 0)
        for i, j in zip(*np.nonzero(differ)):
            if i == j:  # self-regulations (often artefacts) are not compared
                continue
            rows.append({'regulator': names[i], 'target': names[j],
                         **{f'{c}': float(v) for c, v in zip(nets, stack[:, i, j])},
                         'sign_change': bool(lo[i, j] < 0 < hi[i, j])})
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
            if pert.get('sample_schedules') and basal_t.ndim != 3:
                print("[simulate] Warning: per-sample perturbation schedules need per-sample basal: default schedule used")
            for cnt in range(len(times) - 1):
                if not len(targets):
                    continue
                # (n_targets,), or (n_samples, n_targets) with per-sample interactions
                w = signs[targets] * (100 + np.abs(inter_t[cnt][..., targets]).sum(axis=-2))
                if basal_t.ndim == 3:
                    # Each sample with its own schedule of the stimulus
                    u = _sample_values(pert, times[cnt + 1], np.arange(basal_t.shape[1]))
                    basal_t[cnt][:, targets] += u[:, None] * (w if w.ndim == 2 else w[None])
                else:
                    basal_t[cnt][targets] += float(pert['schedule'](times[cnt + 1])) * w

    def _apply_stimulus_targets(self):
        """Forbid the stimulus -> gene edges outside self.stimulus_targets ((n_stimuli, n_genes) mask)."""
        mask = getattr(self, 'stimulus_targets', None)
        if mask is None:
            return
        ns = self.n_stimuli
        self.ref_network[:ns, ns:] = self.ref_network[:ns, ns:] * np.asarray(mask, dtype=float)

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


    def core_binarization(self, data_rna, gene_names, vect_t, G_tot, min_components=1, max_components=5, max_iter_kinetics=100,
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
        basins_genes = []
        pi_init = []
        c = np.ones(G_tot)
        pi_zeros = np.ones(G_tot - ns)
        n_components = 0
        kinetics = NegativeBinomialMixtureEM(min_components=min_components,
                                                 max_components=max_components, zi=None,
                                                 max_iter_em=max_iter_kinetics,
                                                 hard_em=self.hard_em,
                                                 preserve_mean_values=self.preserve_mean_values, mean_forcing_em=self.mean_forcing_em,
                                                 use_scBoolSeq=(scboolseq_matrix is not None),
                                                 soft_em_refinement=self.soft_em_refinement,
                                                 depth_model=self.depth_model,
                                                 min_separation=self.min_nb_separation,
                                                 min_mode_weight=self.min_mode_weight_integration,
                                                 min_mode_ratio=self.min_mode_ratio_integration)

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
            # Initial basins: argmax of the mixture posteriors (masses are imposed later, in loop_trajectories)
            return ks, c, pi0, proba, np.argmax(proba, axis=1), pi

        results = Parallel(n_jobs=-1)(
        delayed(seeded_call)(task_seed(self.seed, 1, g), run_main_loop_for_gene, g) for g in range(ns, G_tot)
        )

        for idx, g in enumerate(range(ns, G_tot)):

            kg, cg, pi_zerog, probag, basinsg, pi_initg = results[idx]
            cg_old, cg = cg, np.minimum(9, cg) # No need of having a variance too low
            kg *= cg / cg_old
            frequency_modes_smooth[:, g] = kg[basinsg]
            if verb and g - ns < len(gene_names): print('Gene {}-{} calibrated...'.format(g, gene_names[g - ns]), kg, cg)
            if len(kg) > n_components:
                n_components = len(kg)
            ks.append(kg)
            proba_init.append(probag)
            basins_genes.append(basinsg)
            c[g] = cg
            pi_zeros[g - ns] = pi_zerog
            pi_init.append(pi_initg)

        n_components = max(n_components, min_components)  # same number of mode rows whatever the genes kept
        self.a = np.zeros((n_components+1, G_tot)) + self.seuil / 10
        frequency_proba_init = np.zeros((N_cells, G_tot, n_components))
        for s in range(ns):
            for t in np.sort(np.unique(vect_t)):
                mask = vect_t == t
                if self._stim_schedule[t][s] < 0.5:
                    frequency_proba_init[mask, s, 0] = 1
                else:
                    frequency_proba_init[mask, s, -1] = 1
        frequency_basins = np.argmax(frequency_proba_init, axis=-1).astype(np.int8)
        for s in range(ns):
            self.a[:, s] = 1
        for g in range(ns, G_tot):
            g_idx = g - ns
            self.a[:len(ks[g_idx]), g] = ks[g_idx][:]
            # Fewer modes (e.g. a single one when the modes are not separated): the missing ones at the highest mode,
            # without cells (lowest = highest: no information on kon)
            self.a[len(ks[g_idx]):n_components, g] = ks[g_idx][-1]
            frequency_proba_init[:, g, :len(ks[g_idx])] = proba_init[g_idx]
            frequency_proba_init[:, g, len(ks[g_idx]):] = 0
            frequency_basins[:, g] = basins_genes[g_idx]
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
                    frequency_basins[ko_cells, g] = 0
                    frequency_modes_smooth[ko_cells, g] = self.a[0, g] / (scale_max[g] + EPS)
                if np.any(ov_cells):
                    frequency_proba_init[ov_cells, g, :] = 0
                    frequency_proba_init[ov_cells, g, n_modes - 1] = 1
                    frequency_basins[ov_cells, g] = n_modes - 1
                    frequency_modes_smooth[ov_cells, g] = self.a[n_modes - 1, g] / (scale_max[g] + EPS)

        if verb: print('Mean proba = ', np.mean(np.max(frequency_proba_init[:, ns:, :], axis=-1)))

        return frequency_modes_smooth, frequency_proba_init, frequency_basins, pi_zeros



    def fit_mixture(self, data, gene_names=np.arange(1, 50000), min_components=2, max_components=2, max_iter_kinetics=0, verb=True, stimulus_schedule=None, time_key='time', kov_cell_mask=None):
        """
        Fit the mixture model parameters to the data.
        """
        set_depth_model(self.depth_model)
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

        frequency_modes_smooth, frequency_proba_init, frequency_basins, pi_zeros = self.core_binarization(
                                        data_rna, gene_names, vect_t, G_tot,
                                        min_components=min_components,
                                        max_components=max_components,
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
        self.basins = frequency_basins

    def _chromatin_model(self, G_tot):
        """ChromatinModel of self.chromatin (template of every simulated cell, levels at c0), None without regions.
        Default c0: regions open by default (basal_c > 0) at their top level, others closed."""
        par = self.chromatin
        if par is None:
            return None
        par = dict(par)
        w = np.asarray(par['w'], dtype=float)
        if w.shape[0] != G_tot:
            raise ValueError(f"chromatin['w'] must have {G_tot} rows (stimuli and genes), got {w.shape}")
        if w.shape[1] == 0:
            return None
        c0 = par.get('c0')
        if c0 is None:
            c0 = np.where(np.asarray(par['basal_c']) > 0, 10 ** 6, 0)
        return ChromatinModel(w, par['basal_c'], par['inter_c'], par['kbar_c'], c_levels=par.get('c_levels'),
                              k_close_const=par.get('k_close_const'), diploid=bool(par.get('diploid', False)),
                              c0=np.asarray(c0).astype(np.int64))

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

    @staticmethod
    def _initial_basins(proba):
        """Posteriors (unchanged) and initial basins (their argmax), as in core_binarization."""
        return proba, np.argmax(proba, axis=1)

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
        set_depth_model(self.depth_model)
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
        # Genes of each sample's condition (select_genes_per_condition); None: every gene in every sample
        gm = self.set_gene_mask(data, [str(x) for x in sample_ids], verb=verb)

        # --- 1. Independent fit per sample (on the genes of its condition) ---
        fits = []
        for s in sample_ids:
            m = samples == s
            if m.sum() < self.min_cells_integration:
                if verb:
                    print(f"[integration] sample {s}: {m.sum()} cells < {self.min_cells_integration}, not fitted alone")
                continue
            if verb:
                print(f"[integration] Fitting the mixture of sample {s} ({m.sum()} cells)")
            active = None if gm is None else gm[sample_ids.index(s)]
            sub = data[m] if active is None else data[m][:, active[ns:]]
            t_sub = np.sort(np.unique(vect_t[m]))
            self.fit_mixture(sub, stimulus_schedule=np.array([sched_full.at(t, s) for t in t_sub]),
                             time_key=time_key, verb=verb, **kwargs)
            f = dict(id=s, mask=m, a=self.a.copy(), proba_init=self.proba_init.copy(), basins=self.basins.copy(),
                     pi_init=self.pi_init, pi0=np.concatenate([np.zeros(ns), self.pi_zinb]), active=active)
            if active is not None:
                self._expand_inactive(f, active, t_sub)
            fits.append(f)
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
            if f['active'] is not None:  # inactive genes: equal modes (no information on kon) after the padding
                f['a'][:, ~f['active']] = 1.0
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
            if f['active'] is not None:  # genes outside the sample's condition: never integrated nor perturbed
                eligible[i, ~f['active']] = False
                perturbed[i, ~f['active']] = False
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
            # One pooled fit per pattern of samples where the genes are active (all samples without per-condition genes)
            patterns = {}
            for g in no_global:
                patterns.setdefault(tuple(np.ones(len(sample_ids), bool) if gm is None else gm[:, g]), []).append(g)
            for key, genes_p in patterns.items():
                cells = np.isin(samples, [sample_ids[k] for k in np.flatnonzero(key)])
                a[:, genes_p] = 1.0
                if not cells.any():
                    continue
                t_p = np.sort(np.unique(vect_t[cells]))
                sub = data if gm is None else data[cells][:, [gene_names[g - ns] for g in genes_p]]
                self.fit_mixture(sub, stimulus_schedule=stimulus_schedule if gm is None
                                 else np.array([sched_full[t] for t in t_p]), time_key=time_key, verb=verb, **kwargs)
                cols = genes_p if gm is None else list(range(ns, ns + len(genes_p)))
                k = min(M, self.a.shape[0] - 1)
                a[:, genes_p] = self.seuil / 10
                a[:k, genes_p] = self.a[:k, cols]
                a[-1, genes_p] = self.a[-1, cols]
                pi0[genes_p] = np.concatenate([np.zeros(ns), self.pi_zinb])[cols]
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
        basins = np.zeros((N, G_tot), dtype=np.int8)
        fitted = np.zeros(N, dtype=bool)
        for i, f in enumerate(fits):
            m = f['mask']
            fitted |= m
            for g in range(ns, G_tot):
                if own[i, g]:  # raw counts, responsibilities of the sample's own fit
                    proba_init[m, g] = f['proba_init'][:, g]
                    basins[m, g] = f['basins'][:, g]
                elif eligible[i, g]:
                    z = np.argmax(f['proba_init'][:, g, :], axis=1)
                    if lam > 0:
                        X[m, g] = integrate_counts(data_rna[m, g], z, (a_samples[i, :-1, g], a_samples[i, -1, g], pi0_samples[i, g]),
                                                   (a_lam[i, :-1, g], a_lam[i, -1, g], pi_lam[i, g]), rng,
                                                   s=None if depth is None else depth[m])
                    proba_init[m, g] = f['proba_init'][:, g]
                    basins[m, g] = f['basins'][:, g]
        # Raw (sample, gene) pairs: classify the raw counts with the global modes (posteriors and masses as a fit)
        raw_cells, raw_masses = {}, {}
        pos = np.searchsorted(np.asarray(sample_ids), samples)
        cell_active = None if gm is None else gm[pos]  # (N, G_tot)
        for g in range(ns, G_tot):
            ng = n_modes[g]
            raw = ~fitted.copy()
            for i, f in enumerate(fits):
                if perturbed[i, g]:
                    raw |= f['mask']
            if cell_active is not None:
                raw &= cell_active[:, g]
            raw_cells[g] = raw
            if raw.any():
                resp, raw_masses[g] = self._posteriors_and_masses(
                    data_rna[raw, g], vect_t[raw], a[:ng, g], a[-1, g],
                    pi_zero=pi0[g] if pi0[g] > 0 else None, zi=True if pi0[g] > 0 else None,
                    depth=None if depth is None else depth[raw])
                proba_init[raw, g, :ng], basins[raw, g] = self._initial_basins(resp)
        for s in range(ns):
            for t in times_all:
                mt = vect_t == t
                proba_init[mt, s, :] = 0
                proba_init[mt, s, 0 if sched_full[t][s] < 0.5 else -1] = 1
        basins[:, :ns] = np.argmax(proba_init[:, :ns], axis=-1)
        if cell_active is not None:  # inactive pairs: a single mode
            off = ~cell_active
            proba_init[off] = 0.0
            proba_init[..., 0][off] = 1.0
            basins[off] = 0

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
        if gm is not None:  # genes outside each sample's condition: equal modes, no zero inflation
            for k in range(len(sample_ids)):
                a_per[k][:, ~gm[k]] = 1.0
                pi0_per[k][~gm[k]] = 0.0
        self.a = a_per
        self.pi_zinb = pi0_per[:, ns:]
        a_cells = a_per
        # Modes in units of the max burst rate of each cell's sample
        kz_cells = a_cells[pos, :-1]                                   # (N, M, G)
        modes = (np.take_along_axis(kz_cells, basins[:, None, :].astype(int), axis=1)[:, 0]
                 / np.maximum(kz_cells.max(axis=1), EPS))
        for t in times_all:
            modes[vect_t == t, :ns] = sched_full[t]
        self.modes = modes
        self.proba_init = proba_init
        self.basins = basins
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
                    if f['active'] is not None and not f['active'][g]:
                        continue  # gene outside the sample's condition
                    n = np.sum(f['mask'] & mt & ~raw_cells[g])
                    if n:
                        p += n * masses_at(f['pi_init'][g - ns], t)
                n_raw = np.sum(raw_cells[g] & mt)
                if n_raw:
                    p += n_raw * masses_at(raw_masses[g], t)
                p = p[:ng]
                if p.sum() <= 0:  # no active cell at t: single mode
                    p[0] = 1.0
                pi_g[t] = p / (p.sum() + EPS)
            self.pi_init.append(pi_g)

        # Per-sample parameters, kept to integrate other files of the same samples
        self.integration = dict(
            sample_ids=[f['id'] for f in fits], a_samples=a_samples, pi0_samples=pi0_samples,
            pi_init_samples=[f['pi_init'] for f in fits], eligible=eligible, a=a, pi0=pi0,
            n_modes=n_modes, ref=None if ref is None else fits[ref]['id'], sample_key=sample_key,
            lam=lam, a_dst=a_lam, pi0_dst=pi_lam, all_sample_ids=sample_ids, own=own, gene_mask=gm)
        if verb:
            n_int = eligible[:, ns:].sum(axis=1)
            print(f"[integration] integrate_samples = {lam:g}; genes with sample-specific parameters per sample: "
                  + ", ".join(f"{f['id']}: {n}/{G_tot - ns}" for f, n in zip(fits, n_int)))
        return X[:, ns:] if lam > 0 else None

    def _expand_inactive(self, f, active, times):
        """Fit f of a sample on its active genes brought to every gene (G_tot): inactive genes with a single mode
        (equal modes, posterior on the first one, basin 0, masses [1, 0, ...], no zero inflation)."""
        G_tot, idx = len(active), np.flatnonzero(active)
        n, M = f['proba_init'].shape[0], f['proba_init'].shape[2]
        a = np.ones((f['a'].shape[0], G_tot))
        a[:, idx] = f['a']
        proba = np.zeros((n, G_tot, M))
        proba[:, ~active, 0] = 1.0
        proba[:, idx] = f['proba_init']
        basins = np.zeros((n, G_tot), dtype=f['basins'].dtype)
        basins[:, idx] = f['basins']
        pi0 = np.zeros(G_tot)
        pi0[idx] = f['pi0']
        ns = self.n_stimuli
        single = {t: np.eye(M)[0] for t in times}
        it = iter(f['pi_init'])
        f.update(a=a, proba_init=proba, basins=basins, pi0=pi0,
                 pi_init=[next(it) if active[g] else single for g in range(ns, G_tot)])

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
        gm = info.get('gene_mask')
        all_ids = list(info.get('all_sample_ids', info['sample_ids']))
        for i, sid in enumerate(info['sample_ids']):
            a_s = info['a_samples'][i]
            for g in range(ns, a_s.shape[1]):
                if gm is not None and not gm[all_ids.index(sid), g]:
                    continue  # gene outside the sample's condition
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


    def adaptive_shrinkage(self, x, mu, fact=2, p=2, raw=False):
        """Modes mu shrunk towards the observed x (scaled); the modes only without inference_mrnas_direct (raw: always shrunk)."""
        if not (self.inference_mrnas_direct or raw):
            return mu * self.scale_proteins
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
    

    def _bridge_d0(self):
        """mRNA degradation rates of the bridges, floored at min_ratio x d1 (None without mrna_bridge)."""
        return self._bridge_rates()[0]

    def _bridge_rates(self, s_idx=None, samples=None):
        """(d0, d1 of production) of the mRNA bridges, d0 floored at min_ratio x d1 ((None, None) without mrna_bridge):
        common (G,), of sample s_idx (G,), or per row of sample indices samples (N, G), with the rates of each sample
        (d_samples, refine_in_loop per condition) when they exist."""
        if not self.mrna_bridge:
            return None, None
        ns = self.n_stimuli
        of = lambda d: (np.maximum(d[0, ns:], self.min_ratio * d[1, ns:]), d[1, ns:].copy())
        if self.d_samples is None or (s_idx is None and samples is None):
            return of(self.d)
        if s_idx is not None:
            return of(self.d_samples[min(int(s_idx), len(self.d_samples) - 1)])
        per = [of(d) for d in self.d_samples]
        rows = np.minimum(np.asarray(samples).astype(int), len(per) - 1)
        return np.stack([p[0] for p in per])[rows], np.stack([p[1] for p in per])[rows]

    def _bridge_beta(self, s_idx):
        """Mean burst size of the genes of sample s_idx in kon units (scale_proteins / k1), empty without sampling."""
        if not self.bridge_sampling:
            return np.zeros(0)
        k1 = self._mixture_terms()[2]
        k1 = k1[min(s_idx, k1.shape[0] - 1)] if k1.ndim == 2 else k1
        return self.scale_proteins / np.maximum(k1[self.n_stimuli:], EPS)

    def estimate_trajectories_given_model(self, vect_t, times, vect_samples_id,
                                      samples_id, vect_rna, y_prot_old, prot_formodes, y_kon_old, y_rna_old,
                                      alpha_old, vect_samples_id_modified,
                                      basal, inter, s1, ks, init_cells, R_opt_traj, to_keep_for_update, offset_init=[0],
                                      n_iter=1, N_full=[100], N_samples=[100], intensity_prior=10,
                                      real_cell_batches=None, batch_idx=None, sim_real_idx=None,
                                      growth_only=False, observed=None, context=None, alpha_pool=None, reg_factor=1.0):
        """
        Infer the protein trajectories when d1 is known and theta is not.
        reg_factor: factor of the entropic regularization of the OT (0.1 at the last iteration, max_iter).

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
                    # Sigmoid bounds and weights of the genes for the network term of the OT cost
                    kb_s, w_s = sigmoid_amplitudes(ks_s), gene_weights(ks_s)[ns:]
                    (d0_br, d1p_br), rna_to_x = self._bridge_rates(s_idx=s_idx), self.scale_proteins / self.fact_simple
                    # Bridge noise of each source trajectory, shared by its targets (OT cost and sampled end)
                    beta_br = np.zeros(0) if d0_br is None else self._bridge_beta(s_idx)
                    z_br = np.random.standard_normal((N_sample, len(beta_br))) if len(beta_br) else np.zeros((0, 0))
                    if d0_br is not None:  # bridges: kon levels of the modes, mRNA in the same units
                        mode_init = kon_modified[current_indices, ns:] * self.scale_proteins
                        mode_end = self.modes[cell_idx, ns:] * self.scale_proteins
                    else:
                        mode_init = self.adaptive_shrinkage(rna_modified[current_indices, ns:] * s1_s, kon_modified[current_indices, ns:]) / s1_s
                        mode_end = self.adaptive_shrinkage(vect_rna[cell_idx, ns:] * s1_s, self.modes[cell_idx, ns:]) / s1_s

                    basal_s = basal[min(s_idx, basal.shape[0] - 1)] if basal.ndim == 2 else basal
                    # Dilution at the birth rate of the real cell of the source before alpha, of the target after
                    # (rate d1 + b, targets c u)
                    rate_src, c_src, b_src = self._dilution(sim_real_idx[current_indices])
                    rate_tgt, c_tgt, b_tgt = self._dilution(cell_idx)
                    b_src = np.zeros(0) if b_src is None else b_src
                    b_tgt = np.zeros(0) if b_tgt is None else b_tgt
                    pairwise_dist = my_otdistance(
                        kon_modified[current_indices, ns:], self.modes[cell_idx, ns:],
                        prot_init,
                        rna_modified[current_indices, ns:], vect_rna[cell_idx, ns:],
                        mode_init, mode_end,
                        alpha_init,
                        s1_s, self._d1_traj(s_idx=s_idx)[ns:], times[b] - time, basal_s, inter_of(inter, s_idx),
                        sigmoid_target(self.modes[cell_idx, ns:], kb_s[ns:, 0], kb_s[ns:, 1]), w_s,
                        n_iter=n_iter, intensity_prior=intensity_prior,
                        n_stimuli=ns, stim_vals=np.asarray(self._stim_schedule.at(times[b], s_idx), dtype=np.float64),
                        scale_proteins=self.scale_proteins,
                        weight_fixed=-1.0 if context is None else context['weight_init'],
                        b_init=b_src, b_end=b_tgt, alpha_tol=self.alpha_threshold,
                        d0=np.zeros(0) if d0_br is None else d0_br, rna_to_x=rna_to_x,
                        beta=beta_br, z=z_br,
                        g_active=np.zeros(0) if self.gene_mask is None
                        else self.active_genes(s_idx)[ns:].astype(np.float64),
                    )

                    delta_t = times[b] - time
                    src_real = sim_real_idx[current_indices]
                    # Cost relative to the best source of each target (no target dropped for being poorly explained);
                    # scale tmp = typical excess cost: entropic and unbalanced regularizations relative to it
                    pairwise_dist = pairwise_dist - pairwise_dist.min(axis=0, keepdims=True)
                    tmp = float(np.median(pairwise_dist))
                    if tmp <= 1e-12:
                        tmp = float(np.mean(pairwise_dist)) if np.mean(pairwise_dist) > 1e-12 else 1.0
                    it_reg = n_iter if context is None else context['n_iter_reg']  # final training regularization
                    reg = max(self.init_entropic_noise * tmp * (1 / it_reg)**(1 - 1/it_reg), 1e-6 * tmp) * reg_factor
                    # Unbalanced relaxation: none at the first iteration, full from min_n_loops (at once with constraints)
                    ramp_uot = self._uot_ramp(it_reg, s_idx)
                    m_ref = max(self.min_n_loops, 1)
                    reg_ref = max(self.init_entropic_noise * tmp * (1 / m_ref)**(1 - 1/m_ref), 1e-6 * tmp)  # reg at min_n_loops

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
                            self._record_coupling(pairwise_dist, src_real, cell_idx, delta_t, reg, tmp, ramp_uot, reg_ref,
                                                  s_idx, time, times[b])

                # Trajectory update (skipped by the growth pass)
                if N_sample and N_cells and not growth_only:
                    # --- Growth-weighted OT marginals ---
                    # Symmetric convention (ours, not WOT's, which weights the source by exp(R·Δt) and keeps
                    # the target uniform): source and target corrected by exp(±R·Δt/2), the midpoint rule for
                    # the growth along the step — see docs/advanced.md#net-proliferation-rate--default-behaviour.
                    mu, nu, reg_m = self._traj_marginals(src_real, cell_idx, delta_t, reg, tmp, ramp_uot, reg_ref)
                    coupling = self._solve_ot(mu, nu, pairwise_dist, reg, reg_m)

                    # Draw one target per trajectory from its coupling row (inverse CDF)
                    cdf = np.cumsum(coupling, axis=1)
                    u = np.random.random(N_sample) * cdf[:, -1]
                    m_idx = np.minimum((cdf < u[:, None]).sum(axis=1), N_cells - 1)
                    tgt = cell_idx[m_idx]

                    # End states recomputed for the sampled pairs only
                    if d0_br is not None:
                        rate_end = rate_src if b_tgt.shape[0] == 0 else rate_tgt[m_idx]
                        alpha_b = alpha_init if b_tgt.shape[0] == 0 else birth_alpha(
                            alpha_init, kon_modified[current_indices, ns:], self.modes[tgt, ns:], self.alpha_threshold)
                        x0 = rna_modified[current_indices, ns:] * s1_s * rna_to_x
                        x1 = vect_rna[tgt, ns:] * s1_s * rna_to_x
                        br_args = (d1p_br, d0_br, rate_src, rate_end, prot_init, x0, x1,
                                   mode_init, mode_end[m_idx], alpha_init, alpha_b, delta_t)
                        next_prot = prot_bridge_end(*br_args, beta_br if len(beta_br) else 0.0 * d0_br,
                                                    z_br if len(beta_br) else 0.0 * prot_init)
                        bridge_at = lambda tau: prot_bridge(*br_args, tau, next_prot)
                    elif b_tgt.shape[0] == 0:  # no dilution
                        rate_end, c_end, alpha_b = None, c_src, None
                        next_prot = find_next_prot(
                            rate_src, prot_init, rna_modified[current_indices, ns:], vect_rna[tgt, ns:],
                            mode_init * c_src, mode_end[m_idx] * c_src, alpha_init, s1_s, delta_t)
                    else:
                        # Birth rate switched at birth_alpha in the equation of each gene (alpha itself unchanged)
                        rate_end, c_end = rate_tgt[m_idx], c_tgt[m_idx]
                        alpha_b = birth_alpha(alpha_init, kon_modified[current_indices, ns:], self.modes[tgt, ns:],
                                              self.alpha_threshold)
                        next_prot = find_next_prot_dilution(
                            rate_src, rate_end, prot_init, mode_init * c_src, mode_end[m_idx] * c_end, alpha_init, alpha_b,
                            s1_s, delta_t)

                    # Timepoints missed by the sample inside the interval: proteins along the same flow
                    for k in range(t_idx + 1, b):
                        if d0_br is not None:
                            prot_modified[N_total * k + offset_init_s + np.arange(N_sample), ns:] = bridge_at(times[k] - time)
                            continue
                        prot_modified[N_total * k + offset_init_s + np.arange(N_sample), ns:] = prot_along(
                            rate_src, prot_init, mode_init * c_src, mode_end[m_idx] * c_end, alpha_init, s1_s,
                            delta_t, times[k] - time, d1_end=rate_end, alpha_b=alpha_b)

                    kon_modified[next_indices] = self.modes[tgt]
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
            arrays = [rna_modified, kon_modified, sim_real_idx]
            fill_virtual(arrays, valid, T)
            fill_virtual([prot_modified], valid, T, inside=False)

        return prot_modified, prot_formodes, rna_modified, kon_modified, \
                alpha_modified, vect_samples_id_modified, R_opt_traj, to_keep_for_update

    def _reg_factor(self, n_iter, context):
        """Factor of the entropic regularization of the OT: 0.1 at the last iteration (n_iter = max_iter) only when
        max_iter < min_n_loops (the regularization ramp then not completed), else 1; that of the last training
        iteration for the test set (context)."""
        if context is not None:
            return context['reg_factor']
        return 0.1 if (n_iter == self.max_iter and self.max_iter < self.min_n_loops) else 1.0

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

    def _uot_ramp(self, n_iter, s_idx):
        """Fraction of the unbalanced relaxation at iteration n_iter: 0 (balanced) at the first iteration, 1 from
        min_n_loops, as the mass constraint of the basin updates; 1 at once with lineage or transition-rate constraints,
        which rely on the unbalanced relaxation."""
        if getattr(self, '_lineage', None) is not None or self._transition_matrix(s_idx) is not None:
            return 1.0
        return min(1.0, (n_iter - 1) / max(self.min_n_loops - 1, 1))

    def _traj_marginals(self, src_real, cell_idx, delta_t, reg, log_G, ramp=1.0, reg_ref=None):
        """
        Marginals (mu, nu) and unbalanced relaxation of the trajectory OT between source states and real cells. The
        target marginal is enforced with the exponent f = reg_m / (reg_m + reg) of the unbalanced Sinkhorn: from 1
        (balanced, ramp = 0) to that of reg_m = unbalanced_reg * log_G at min_n_loops (log_G: typical excess cost of
        the batch, the scale of the cost); once the entropic regularization
        is below reg_ref (its value at min_n_loops), or with full relaxation from the start (lineage / transition
        rates, ramp = 1), reg_m follows reg so that f is constant at its min_n_loops value.
        """
        r = getattr(self, '_prolif_net_rate', None)
        if r is not None:
            mu = np.exp(r[src_real] * delta_t / 2)
            nu = np.exp(-r[cell_idx] * delta_t / 2)
            mu, nu = mu / mu.sum(), nu / nu.sum()
        else:
            mu = np.ones(len(src_real)) / len(src_real)
            nu = np.ones(len(cell_idx)) / len(cell_idx)
        reg_m = None
        if self.unbalanced_reg and ramp > 0:
            rm = self.unbalanced_reg * log_G
            if reg_ref is not None and (reg < reg_ref or ramp >= 1):
                rm *= reg / reg_ref  # f frozen at its value at min_n_loops (from the start with full relaxation)
            if ramp < 1:
                # Exponent of the target marginal interpolated between 1 (balanced) and that of rm
                f = 1 - ramp * reg / (rm + reg)
                rm = reg * f / (1 - f)
            reg_m = np.array([1e6, rm])
        return mu, nu, reg_m

    def _record_coupling(self, C, src_real, cell_idx, delta_t, reg, log_G, ramp_uot, reg_ref, s_idx, t_from, t_to, top_k=50):
        """
        Soft coupling of the final trajectories (same costs and marginals as the trajectory OT), kept sparse:
        the top_k targets of each source state; entries (real source cell, real target cell, mass), the
        mass of each state scaled to ~1 (batches of different sizes), summed later over duplicated states.
        """
        mu, nu, reg_m = self._traj_marginals(src_real, cell_idx, delta_t, reg, log_G, ramp_uot, reg_ref)
        P = self._solve_ot(mu, nu, C, reg, reg_m) * len(src_real)
        k = min(top_k, P.shape[1])
        cols = np.argpartition(-P, k - 1, axis=1)[:, :k]
        w = np.take_along_axis(P, cols, axis=1)
        keep = w > 1e-3 * np.maximum(w.max(axis=1, keepdims=True), 1e-300)
        rows = np.broadcast_to(np.arange(len(src_real))[:, None], cols.shape)
        self.couplings.append(dict(src=src_real[rows[keep]], tgt=cell_idx[cols[keep]], w=w[keep].astype(np.float32),
                                   sample=s_idx, t_from=float(t_from), t_to=float(t_to)))

    def _inference_noise_coef(self):
        """noise_coef of the network fits of fit_network (inference_stochastic, trajectories on the modes without
        inference_mrnas_direct): protein noise of the PDMP at d1_eff, variance eps s P / k1 with eps = d1_eff / d0 =
        1 / (1 + r), r the literature d0 / d1 (one stage); None otherwise."""
        if not self.inference_stochastic or self.inference_mrnas_direct or self.mrna_bridge:
            return None
        ns = self.n_stimuli
        k1 = self._mixture_terms()[2]
        k1 = k1.mean(axis=0) if np.ndim(k1) == 2 else k1
        r = np.maximum(self.d[0] / self.d[1], self.min_ratio)
        coef = self.scale_proteins / (1 + r) / np.maximum(k1, 1e-12)
        coef[:ns] = 0.0
        return coef

    def _d1_traj(self, d=None, s_idx=None):
        """Protein rates (G,) of the fit_network trajectories: d1_eff = d0 d1 / (d0 + d1) without inference_mrnas_direct
        (trajectories on the modes: mean delay 1/d0 + 1/d1 of the mRNA-protein chain), else d1 (the shrunk modes,
        read on the observed mRNA, already carry the delay); d1 with mrna_bridge (explicit mRNA). s_idx: rates of that
        sample (d_samples, refine_in_loop per condition)."""
        if d is None:
            d = self.d if self.d_samples is None or s_idx is None else self.d_samples[min(int(s_idx), len(self.d_samples) - 1)]
        if self.inference_mrnas_direct or self.mrna_bridge:
            return d[1].copy()
        return d[0] * d[1] / (d[0] + d[1])

    def _dilution(self, real_idx):
        """
        Protein relaxation of trajectory states with dilution at the birth rate b of their real cell (real_idx, -1:
        b = 0): dP/dt = d1 u - (d1 + b) P = (d1 + b)(c u - P). Returns (rate d1 + b (N, G), factor c = d1 / (d1 + b)
        (N, G), b (N,)) to use in place of (d1, 1) with the modes as targets; (d1, 1.0, None) exactly without dilution.
        """
        ns = self.n_stimuli
        d1 = self._d1_traj()[ns:]
        b = getattr(self, '_birth_rate', None)
        rs = getattr(self, '_real_sample_idx', None)
        if self.d_samples is not None and real_idx is not None and rs is not None:
            # Rates of each state's sample (refine_in_loop per condition), through the dilution form (b = 0 if none)
            real_idx = np.asarray(real_idx, dtype=int)
            per = np.stack([self._d1_traj(s_idx=k)[ns:] for k in range(len(self.d_samples))])
            d1 = np.where((real_idx >= 0)[:, None], per[rs[np.maximum(real_idx, 0)]], per.mean(axis=0)[None])
            bs = (np.where(real_idx >= 0, b[np.maximum(real_idx, 0)], 0.0) if self.protein_dilution and b is not None
                  else np.zeros(len(real_idx)))
            rate = d1 + bs[:, None]
            return rate, np.where(rate > 0, d1 / np.maximum(rate, 1e-300), 1.0), bs
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

    STATE_RATES = ('state_birth', 'state_death', 'state_birth_base', 'state_death_base')

    def fit_birth_death_to_growth(self, verb=True):
        """
        Birth and death rates of each trajectory state consistent with its net growth R_opt (growth OT pass of
        fit_network) and the closest to the prior rates of its real cell: the change of the net rate is shared in
        proportion to the prior birth and death (split_net_change), so that a non-dividing cell stays non-dividing.
        States without R_opt (last time, no real cell) keep the prior. Without the stimulus too (R_opt - R_stim_offset
        and the base priors) when its effects are given apart. Sets the attributes STATE_RATES, used by
        refine_network_degradations (trajectories, network, d1 / d0, heads of the proliferation MLP) and saved for
        the simulations; prior rates (or None) when R_opt is unavailable.
        """
        from ..tools.estimate_proliferation import split_net_change
        b0, d0 = self._prior_state_rate('_birth_rate'), self._prior_state_rate('_death_rate')
        bb0, db0 = self._prior_state_rate('_birth_rate_base'), self._prior_state_rate('_death_rate_base')
        self.state_birth, self.state_death, self.state_birth_base, self.state_death_base = b0, d0, bb0, db0
        R = getattr(self, 'R_opt', None)
        if not self.protein_dilution or b0 is None or d0 is None or R is None or len(R) != len(b0):
            if verb and self.protein_dilution and b0 is not None:
                print("[refine_network_degradations] No growth estimate (R_opt): prior birth and death rates kept")
            return
        R = np.asarray(R, dtype=float)
        ok = np.isfinite(R) & (np.asarray(self.traj_real_idx) >= 0)

        def fit(b, d, net):
            b, d = np.array(b, dtype=float), np.array(d, dtype=float)
            b[ok], d[ok] = split_net_change(b[ok], d[ok], net[ok] - (b[ok] - d[ok]))
            return b, d

        self.state_birth, self.state_death = fit(b0, d0, R)
        if self.R_stim_offset is not None and bb0 is not None and db0 is not None:
            self.state_birth_base, self.state_death_base = fit(bb0, db0, R - np.asarray(self.R_stim_offset, dtype=float))
        else:
            self.state_birth_base, self.state_death_base = self.state_birth, self.state_death
        if verb:
            gap = np.abs(R[ok] - (b0[ok] - d0[ok]))
            print(f"[refine_network_degradations] Birth and death rates consistent with R_opt ({int(ok.sum())} states): "
                  f"birth {np.mean(b0[ok]):.4f} -> {np.mean(self.state_birth[ok]):.4f} h^-1, death "
                  f"{np.mean(d0[ok]):.4f} -> {np.mean(self.state_death[ok]):.4f} h^-1 (mean |R_opt - prior net| "
                  f"{np.mean(gap):.4f} h^-1)")

    def _state_rates(self, base=False):
        """(birth, death) of each trajectory state: those of fit_birth_death_to_growth (or loaded for the simulations)
        when they match the states, else the prior of the real cells; (None, None) without dilution."""
        if not self.protein_dilution:
            return None, None
        b, d = (getattr(self, 'state_birth_base', None), getattr(self, 'state_death_base', None)) if base else \
            (getattr(self, 'state_birth', None), getattr(self, 'state_death', None))
        if b is not None and d is not None and len(b) == len(self.times_data):
            return b, d
        sfx = '_base' if base else ''
        return self._prior_state_rate(f'_birth_rate{sfx}'), self._prior_state_rate(f'_death_rate{sfx}')

    def save_state_rates(self, cardamom_dir):
        """Birth and death rates of the trajectory states (cardamomOT/data_state_rates.npz) for the simulations."""
        path = os.path.join(cardamom_dir, 'data_state_rates.npz')
        arrs = {k: np.asarray(getattr(self, k)) for k in self.STATE_RATES if getattr(self, k, None) is not None}
        if arrs:
            np.savez(path, **arrs)
        elif os.path.exists(path):
            os.remove(path)

    def load_state_rates(self, cardamom_dir):
        """Birth and death rates of the trajectory states saved by infer_network_simul (dilution of the simulations)."""
        path = os.path.join(cardamom_dir, 'data_state_rates.npz')
        if os.path.exists(path):
            with np.load(path) as f:
                for k in self.STATE_RATES:
                    setattr(self, k, f[k] if k in f else None)

    def load_chromatin(self, cardamom_dir):
        """Chromatin regions of the simulations (self.chromatin) from chromatin_simul.npz, if present."""
        path = os.path.join(cardamom_dir, 'chromatin_simul.npz')
        if os.path.exists(path):
            with np.load(path, allow_pickle=False) as f:
                self.chromatin = {k: f[k] for k in f.files}
            print(f"[CardamomOT] Chromatin regions of the simulations: {self.chromatin['w'].shape[1]} (chromatin_simul.npz)")

    def _state_birth(self, verb=False):
        """
        Birth rate of each trajectory state for the dilution in refine_network_degradations (trajectories, d1 refit):
        consistent with R_opt (fit_birth_death_to_growth), else the prior of its real cell. With b explicit in the
        refitted ODE, d1 stays a pure degradation rate. None without dilution.
        """
        b = self._state_rates()[0]
        if verb and b is not None:
            print(f"[refine_network_degradations] Dilution of the proteins: birth rate of the states, "
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
        Log mass gain of each source state over one interval (growth estimation of Waddington-OT,
        Schiebinger et al. 2019, Cell 176:928): the source marginal starts from the prior growth exp(R_prior·Δt)
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
                act = self.active_genes(s_idx)
                cols = np.arange(ns, data_rna.shape[1]) if act is None else np.flatnonzero(act[ns:]) + ns
                strata[idx] = km.fit_predict(np.log1p(data_rna[np.ix_(idx, cols)]))
        return strata

    def _n_params_per_target(self, active_cols, n_samples):
        """Largest parameter count of one target-gene fit: active regulators x network conditions + per-sample basals."""
        k_max = max((len(c) for c in active_cols[self.n_stimuli:]), default=0)
        n_cond = 1 if self.sample_conditions is None else int(np.max(self.sample_conditions)) + 1
        return k_max * n_cond + n_samples

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

        basal_init / inter_init : (n_samples, G_tot) / (G_tot, G_tot) or None
            Starting point for theta. Zeros if None.
        kov_cell_mask : (N_cells, G_tot) int8 array or None
            Per-cell KO/OV constraints. -1 → KO (force lowest mode),
            +1 → OV (force highest mode), 0 → no constraint.
            Applied after each basin update as a hard override.
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
            # basal: (n_samples, G_tot); inter: (G_tot, G_tot)
            self.basal = np.zeros((n_samples_local, G_tot))
            self.inter = np.zeros((G_tot, G_tot))
            if basal_init is not None:
                # basal_init already (n_samples, G) from _normalize_theta
                self.basal[:, :] = basal_init[:n_samples_local]
            if inter_init is not None:
                # (G, G), or per sample (n_samples, G, G) with network conditions
                self.inter = np.array(inter_init, dtype=float)

        # Ensure basal is always 2-D (n_samples, G_tot) — promote 1-D for compat
        if self.basal.ndim == 1:
            self.basal = self.basal[np.newaxis, :]
        basal = self.basal.copy()      # (n_samples, G_tot)
        inter = self.inter.copy()      # (G_tot, G_tot), or (n_samples, G_tot, G_tot) with network conditions
        basal_tmp = self.basal.copy()
        inter_tmp = self.inter.copy()
        # Regularization targets: provided prior, or zeros (no penalization)
        _basal_ref = basal_ref if basal_ref is not None else np.zeros((n_samples_local, G_tot))
        _inter_ref = inter_ref if inter_ref is not None else np.zeros((G_tot, G_tot))
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
        exact_fm = compute_theta and self.exact_flow_matching
        if compute_theta:
            _, prev_cols = active_regulators(self.ref_network, inter_ref, hard_forcing_ref)
            y_prot_prev = PathProt.zeros(len(times) * N_tot, self.flow_matching_nodes, G_tot) if exact_fm \
                else PrevProt.zeros(prev_cols, len(times) * N_tot)
        if compute_theta and not exact_fm:
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
            d0_br, d1p_br = self._bridge_rates(samples=y_samples)  # per row with rates per sample
            if d0_br is not None:  # bridges: kon levels of the modes, mRNA in the same units
                modes = y_kon[:, ns:] * self.scale_proteins
                x_rna = y_rna[:, ns:] * s1r * (self.scale_proteins / self.fact_simple)
            else:
                modes = self.adaptive_shrinkage(y_rna[:, ns:] * s1r, y_kon[:, ns:]) / s1r
                x_rna = y_rna[:, ns:] * self.scale_proteins
            for a, b, slots in pairs:
                idx_prev, idx_next = N_tot * a + slots, N_tot * b + slots
                rate_p, c_p, b_p = self._dilution(y_real[idx_prev])  # dilution of the source states before alpha
                rate_n, c_n, _ = self._dilution(y_real[idx_next])  # and of the target states after
                if d0_br is not None:
                    c_p, c_n = 1.0, 1.0
                fill = self._fill_prev_prot
                if exact_fm:  # path states from the start state of the interval (start row given in full)
                    fill = partial(self._fill_path_prot, P_start=y_prot[idx_prev], stim_end=y_prot[idx_next, :ns])
                fill(
                    y_prot_prev, idx_next, y_alpha[a][slots], times[b] - times[a],
                    rate_p, y_prot[idx_prev, ns:],
                    x_rna[idx_prev], x_rna[idx_next],
                    modes[idx_prev] * c_p, modes[idx_next] * c_n, _rows(s1r, idx_prev),
                    d1_end=None if b_p is None else rate_n,
                    alpha_b=None if b_p is None else birth_alpha(y_alpha[a][slots], y_kon[idx_prev, ns:],
                                                                 y_kon[idx_next, ns:], self.alpha_threshold),
                    d0=_rows(d0_br, idx_next), d1_prod=_rows(d1p_br, idx_next), P1=y_prot[idx_next, ns:])

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
                    sels, y_samples, y_kon, y_prot, y_prot_prev,
                    ks, n_stimuli=ns, samples_id=samples_id,
                    ref_network=self.ref_network, basal_init=basal_start, inter_init=inter_start,
                    basal_ref=basal_ref, inter_ref=inter_ref,
                    scale=self.scale_pen, weight_prev=weight_prev,
                    final=0, constrain_basal_uniform=self.constrain_basal_uniform,
                    hard_forcing_ref=hard_forcing_ref, ref_constraint_pct=ref_constraint_pct,
                    seuil_zero_min_ref=self.seuil_zero_min_ref,
                    sample_conditions=self.sample_conditions, condition_pen=self.network_condition_pen,
                gene_mask=self.gene_mask,
                    noise_coef=self._inference_noise_coef())  # current ratio (refine_in_loop)
            return self._fit_theta_averaged(fit_on, vect_t_sim, y_samples, traj_cell_types(), n_fits,
                                            self._n_params_per_target(prev_cols, n_samples_local), valid=valid)

        # Stratum of each real cell, for balanced OT batches; real cells at t0 per sample
        strata = self._traj_strata(data_rna, vect_t, vect_samples_id, times, samples_id)
        t0_real = [np.flatnonzero((vect_t == times[first_t[s]]) & (vect_samples_id == sample))
                   for s, sample in enumerate(samples_id)]

        # Rates (and network) refined in the loop (training only), with a rappel towards the literature
        deg_fit = self.refine_in_loop and compute_theta and context is None and alpha_pool is None and len(times) > 1
        last_deg, self._n_refine_in_loop = None, 0
        if deg_fit and self.d_literature is None:
            self.d_literature = self.d.copy()

        # === Main loop ===
        while True:

            weight_prev = self.weight_prev * min(1, (n_iter-1)/min_n_loops) # Flow matching from second iteration and small at early ones

            # Stop only at the end of a cycle of the degradation updates
            cycle_done = last_deg is None or n_iter - last_deg >= self.refine_in_loop_period
            if (count_end >= count_max and cycle_done) or n_iter > self.max_iter:
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
                y_prot, y_prot_formodes, y_rna, y_kon, y_alpha, y_samples, R_opt_traj, to_keep_for_update = \
                    self.estimate_trajectories_given_model(
                        vect_t, times, vect_samples_id, samples_id,
                        data_rna, y_prot, y_prot_formodes, y_kon, y_rna, y_alpha, y_samples,
                        basal, inter, s1, ks, init_cells,
                        R_opt_traj, to_keep_for_update,
                        offset_init=offset_init,
                        N_full=N_full, N_samples=N_tmp,
                        n_iter=n_iter + it_shift,
                        intensity_prior=intensity_prior * compute_theta * (1 - hard_forcing_ref),
                        real_cell_batches=real_cell_batches, batch_idx=batch_idx,
                        sim_real_idx=y_real, observed=observed, context=context, alpha_pool=alpha_pool,
                        reg_factor=self._reg_factor(n_iter, context)
                    )

                offset_init = [offset_init[s] + N_tmp[s] for s in range(len(samples_id))]

            if y_prot_prev is not None and not exact_fm:
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
            error = self._count_errors_per_sample(y_prot[valid], y_kon[valid], ks, inter, basal,
                                                   samples_id=sample_idx, samples_data=y_samples[valid])
            if compute_theta and len(times) > 1:
                if self.weight_prev > 0 or exact_fm:
                    refresh_prev_prot()
                # Mean of fits on independent stratified subsamples covering the trajectory states
                basal, inter, basal_tmp, inter_tmp = fit_theta(weight_prev, basal, inter, self.n_network_fits)

            error_2 = self._count_errors_per_sample(y_prot[valid], y_kon[valid], ks, inter, basal,
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

            if deg_fit and n_iter >= self.refine_in_loop_start and \
                    (n_iter - self.refine_in_loop_start) % self.refine_in_loop_period == 0:
                # Rates (and network) adjusted to the dynamics they generate, before alpha and the modes
                basal, inter = self._refine_in_loop(y_prot, y_rna, y_kon, y_samples, y_real, valid, times, N_tot,
                                                    basal, inter, ks, sample_idx, verb=verb)
                last_deg = n_iter

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
            d0_br, d1p_br = self._bridge_rates(samples=y_samples)  # per row with rates per sample
            if d0_br is not None:  # bridges: kon levels of the modes (no compression, production d1)
                modes = y_kon[:, ns:] * self.scale_proteins
            else:
                modes = self.adaptive_shrinkage(y_rna[:, ns:] * s1r, y_kon[:, ns:]) / s1r
            if len(times) > 1:
                # Intervals are independent: one parallel task per (pair of observed times, slots)
                alpha_fn = inference_alpha_1thread if len(pairs) > 1 else inference_alpha
                def alpha_task(a, b, slots):
                    r0, r1 = N_tot * a + slots, N_tot * b + slots
                    rate_a, c_a, b_a = self._dilution(y_real[r0])  # dilution of the source states before alpha
                    rate_b, c_b, _ = self._dilution(y_real[r1])  # and of the target states after
                    if d0_br is not None:
                        c_a, c_b = 1.0, 1.0
                    return delayed(alpha_fn)(
                            rate_a, s1,
                            y_alpha[a][slots],
                            y_kon[r0], kon_vector[r0], y_prot[r0], y_rna[r0],
                            y_kon[r1], kon_vector[r1], y_prot[r1], y_rna[r1],
                            modes[r0] * c_a, modes[r1] * c_b,
                            basal, inter, ks, times[b] - times[a],
                            tol=self.alpha_threshold,
                            n_pas = self.n_pas if self.force_n_pas else max(self.n_pas, int(times[b] - times[a])),
                            samples_data=y_samples[r0],
                            stim_vals=self._stim_schedule.per_cell(times[b], y_samples[r0]),
                            scale_proteins=self.scale_proteins, d1_end=None if b_a is None else rate_b,
                            d0=_rows(d0_br, r0), d1_prod=_rows(d1p_br, r0),
                            rna_to_x=self.scale_proteins / self.fact_simple
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
            n_cells = self.proba_init.shape[0]
            # Mass constraint grows from the argmax masses (lam=0: plain argmax) to nu (lam=1: full OT) by min_n_loops
            lam = min(1.0, (n_iter - 1) / max(min_n_loops - 1, 1))
            # The weight of the network increases slowly because it aims to get the right attribution given probabilities that are close
            weight_prob = max(.96**(n_iter - 1), .1)

            # Mode amplitudes of each real cell's sample (per-sample mixtures), and their max
            ks_cells = (ks[np.searchsorted(np.asarray(samples_id), vect_samples_id)]
                        if ks.ndim == 3 else None)
            ks_max = ks.max(axis=0) if ks.ndim == 3 else ks
            # Genes of each real cell's condition (select_genes_per_condition): the others keep their single mode
            act_cells = self.cell_gene_mask(np.searchsorted(np.asarray(samples_id), vect_samples_id))

            def assign_basins(mu, nu, dist):
                # EMD with target masses interpolated between those of the argmax and nu
                nu_argmax = np.bincount(np.argmin(dist, axis=1), minlength=dist.shape[1]) / dist.shape[0]
                coupling = ot.emd(mu, (1 - lam) * nu_argmax + lam * nu, dist, numItermax=int(1e7))
                return np.argmax(coupling, axis=1)
            
            def run_main_loop_for_gene(g, temporal=self.temporal_basins):
                l_max = 1 + np.argmax(ks_max[g, :])
                obj = np.zeros((n_cells, l_max), dtype=float)
                obj[:, :] = ks_cells[:, g, :l_max] if ks_cells is not None else ks[g, :l_max][None, :]
                # Distance network / mode on kon: weakly separated genes keep a weak network term against the posteriors
                tmp_basins = np.zeros(n_cells, dtype=np.int8)
                tmp_modes = np.zeros_like(self.modes[:, g])
                act = np.ones(n_cells, bool) if act_cells is None else act_cells[:, g]
                tmp_modes[~act] = obj[~act, 0]
                if temporal:
                    for t_i in times:
                        indices = (vect_t == t_i) & act
                        if not indices.any():
                            continue
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
                            tmp_modes_i[cell] = obj_i[cell, idx[cell]]
                        tmp_basins[indices] = idx
                        tmp_modes[indices] = tmp_modes_i[:]
                elif act.any():
                    proba = self.proba_init[act, g, :l_max].copy()
                    post = proba / np.maximum(np.sum(proba, axis=1, keepdims=True), EPS)
                    proba /= np.max(proba, axis=1, keepdims=True)
                    n_act = int(act.sum())
                    mu = np.ones(n_act) / n_act
                    nu = np.sum([self.pi_init[g - ns][t_i] * np.sum((vect_t == t_i) & act) / n_act
                                                for t_i in times], axis=0)[:l_max] * self.force_basins + np.sum(
                                                    post, axis=0) * (1 - self.force_basins)
                    nu /= np.sum(nu)
                    diff_k = np.maximum(1 - np.abs(kon_vector_formodes[act, g, None] - obj[act]), 1e-3)
                    diff_k /= np.max(diff_k, axis=1, keepdims=True)
                    dist = - (np.log(proba) + (1 - weight_prob) * 
                              to_keep_for_update[act, None] * np.log(diff_k))
                    dist = np.clip(dist, 0, 100)
                    idx = assign_basins(mu, nu, dist)
                    tmp_modes[act] = obj[act][np.arange(n_act), idx]
                    tmp_basins[act] = idx
                return tmp_basins, tmp_modes
            results = Parallel(n_jobs=-1)(
                delayed(run_main_loop_for_gene)(g) for g in range(ns, G_tot))

            for idx, g in enumerate(range(ns, G_tot)):
                tmp_basins, tmp_modes = results[idx]
                self.basins[:, g], self.modes[:, g] = tmp_basins, tmp_modes[:]

            # ── Force KO/OV cells to the correct mode after update ────────
            if kov_cell_mask is not None:
                for g in range(ns, G_tot):
                    l_max = 1 + int(np.argmax(ks_max[g, :]))
                    ko_cells = kov_cell_mask[:, g] < 0
                    ov_cells = kov_cell_mask[:, g] > 0
                    if np.any(ko_cells):
                        self.basins[ko_cells, g] = 0
                        self.modes[ko_cells, g] = ks_cells[ko_cells, g, 0] if ks_cells is not None else ks[g, 0]
                    if np.any(ov_cells):
                        self.basins[ov_cells, g] = l_max - 1
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
                    data_rna, y_prot, y_prot_formodes, y_kon, y_rna, y_alpha, y_samples,
                    basal, inter, s1, ks, init_cells,
                    R_opt_traj, to_keep_for_update,
                    offset_init=offset_init,
                    N_full=N_full, N_samples=N_tmp,
                    n_iter=n_iter - 1 + it_shift,
                    intensity_prior=intensity_prior * compute_theta * (1 - hard_forcing_ref),
                    real_cell_batches=real_cell_batches, batch_idx=batch_idx,
                    sim_real_idx=y_real, growth_only=True, observed=observed, context=context, alpha_pool=alpha_pool,
                    reg_factor=self._reg_factor(n_iter - 1, context)
                )[6]
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
                       'weight_init': float(n_reg <= ip) * (1 / n_reg)**(1 - 1 / n_reg),
                       'weight_prob': max(.96**(n_iter - 1), .1),
                       'reg_factor': self._reg_factor(n_iter - 1, None)}
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
        self.samples_data = y_samples
        self.times_data = vect_t_sim
        self.alpha = y_alpha
        # Real cell behind each trajectory state and its cell type (stratify later batches)
        self.traj_real_idx = y_real
        self.traj_cell_types = traj_cell_types()
        self.traj_valid = valid


    def _refine_in_loop(self, y_prot, y_rna, y_kon, y_samples, y_real, valid, times, N_tot, basal, inter, ks,
                        sample_idx, verb=True):
        """
        refine_in_loop: refine_in_loop_epochs optimizer steps of the free-running fit (_fit_free_running) on the
        windows of sde_max_horizon only (whole time course from the first timepoint when it fits), on the modes of
        the current trajectories: the model of the simulations (explicit mRNA with simulation_mrnas_direct, protein-
        only PDMP with simulation_stochastic, else ODE), whatever built the trajectories (inference_mrnas_direct:
        full d1 and d0 / d1). Constant rates of that model from the current self.d (protein-only models: d1_eff and
        eps = 1 / (1 + r)), rappel towards the literature (d_literature) in the same units, decreasing along the
        updates (sde_lambda_rates x refine_in_loop_decay^k), no scale; the network
        too with refine_theta_with_sde (rappel towards the current fit). The learned rates are converted back to
        the full (d0, d1) of self.d (ODE: d0 / d1 kept); returns (basal, inter).
        """
        ns = self.n_stimuli
        mode = 'sde2' if (self.simulation_mrnas_direct and self.simulation_stochastic) else \
            ('sde1' if self.simulation_stochastic else 'ode')
        _, _, k1 = self._mixture_terms()
        r = np.maximum(self.d[0] / self.d[1], self.min_ratio)
        d1_s, eps_s = (self.d[1].copy(), 1.0 / r) if mode == 'sde2' else (self.d[1] * r / (1 + r), 1.0 / (1 + r))
        d1_ref, eps_ref = self._literature_rates(mode)
        b_states = self._dilution(y_real)[2]
        basal_n, inter_n, d1_c, eps_c, _, _ = self._fit_free_running(
            y_prot, basal, inter, mode, d1_s, eps_s, valid, times, N_tot, ks, k1, sample_idx, b_states, verb=verb,
            samples_rows=y_samples, kon_targets=y_kon, rna=y_rna, prolif=None, temporal=False, learn_scale=False,
            learn_theta=self.refine_theta_with_sde, epochs=self.refine_in_loop_epochs, per_condition=True,
            d1_ref=d1_ref, eps_ref=eps_ref, only_max_horizon=True,
            lam_rates=self.sde_lambda_rates * self.refine_in_loop_decay ** self._n_refine_in_loop, tag='fit_network')
        self._n_refine_in_loop += 1

        def full_rates(c):
            # Full (d0, d1) of kinetics c from the fitted rates of the model
            d = self.d.copy()
            d1n, epsn = d1_c[c, 0, ns:], eps_c[c, 0, ns:]
            d0n = d1n / epsn
            if mode == 'sde2':
                d[1, ns:], d[0, ns:] = d1n, d0n
            elif mode == 'sde1':  # d1_eff and d0: d1 = d0 d1_eff / (d0 - d1_eff)
                d[0, ns:] = d0n
                d[1, ns:] = d0n * d1n / np.maximum(d0n - d1n, 1e-12)
            else:  # ODE: d1_eff only, the ratio d0 / d1 kept
                d[1, ns:] = d1n * (1 + r[ns:]) / r[ns:]
                d[0, ns:] = r[ns:] * d[1, ns:]
            return d

        if d1_c.shape[0] > 1:  # one fit per condition (degradation_rates_per_condition): rates of each sample
            cond = np.asarray(self.sample_conditions, dtype=int)
            self.d_samples = np.stack([full_rates(c) for c in cond])
            self.d = self.d_samples.mean(axis=0)
        else:
            self.d = full_rates(0)
        if verb:
            lit = self.d_literature
            print(f"[fit_network] refine_in_loop ({mode}): d1 x{np.exp(np.mean(np.log(self.d[1, ns:] / lit[1, ns:]))):.3f}, "
                  f"d0 / d1 median {np.median(self.d[0, ns:] / self.d[1, ns:]):.3g} "
                  f"(literature {np.median(lit[0, ns:] / lit[1, ns:]):.3g})")
        return basal_n, inter_n

    @staticmethod
    def _normalize_theta(arr, G_tot, n_samples=1, is_inter=False):
        """Normalize basal/inter init or ref arrays (former trailing network axis of size 1 dropped).

        For inter (is_inter=True): (G, G) common, or (n_samples, G, G) per sample → as-is.
        For basal (is_inter=False): (G,) → (1, G) broadcast to 1 sample; (n_samples, G) → as-is.

        Returns None if arr is None.
        """
        if arr is None:
            return None
        arr = drop_network_axis(np.asarray(arr, dtype=float))
        if not is_inter and arr.ndim == 1:
            return arr[np.newaxis, :]
        return arr

    def fit_network(
        self,
        data,
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
        vect_samples_id : ndarray or None
            Array of sample labels (same size as data), or None if only one sample.
        basal_init : ndarray or None
            Initial basal rates: shape (G,) broadcast to all samples, or (n_samples, G).
        inter_init : ndarray or None
            Initial interaction matrix: shape (G, G), or (n_samples, G, G) with network conditions.
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
        # Genes of each sample's condition (select_genes_per_condition)
        self.gene_mask = None
        if hasattr(data, 'uns'):
            self.set_gene_mask(data, [str(s) for s in samples_id], verb=verb)
        self.d_samples = None
        self._real_sample_idx = np.searchsorted(samples_id, vect_samples_id)  # sample index of each real cell

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

        # --- Normalize init/ref arrays to (n_samples, G_tot) / (G_tot, G_tot) ---
        n_samples = len(samples_id)
        basal_init = self._normalize_theta(basal_init, G_tot, n_samples, is_inter=False)
        inter_init = self._normalize_theta(inter_init, G_tot, n_samples, is_inter=True)
        basal_ref  = self._normalize_theta(basal_ref,  G_tot, n_samples, is_inter=False)
        inter_ref  = self._normalize_theta(inter_ref,  G_tot, n_samples, is_inter=True)

        # --- Build per-cell KO/OV mask from basal_ref (±100 entries) ─────────
        # kov_cell_mask[cell, g] = -1 (KO) / +1 (OV) / 0 (unconstrained)
        kov_cell_mask = None
        if basal_ref is not None:
            br = np.asarray(basal_ref, dtype=float)
            if br.ndim == 2 and np.any(np.abs(br) > 50):
                cm = np.zeros((len(vect_t), G_tot), dtype=np.int8)
                for s_idx, s in enumerate(samples_id):
                    cell_idx = np.where(vect_samples_id == s)[0]
                    ko_genes = np.where(br[s_idx] < -50)[0]
                    ov_genes = np.where(br[s_idx] >  50)[0]
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
            intensity_prior=self.intensity_prior,
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
                    print(f"  {'' if c is None else f'Condition {c} | '}Interactions:\n", inter_c.T)
                print("  Basal:\n", basal.mean(axis=0))
            

    def _fill_prev_prot(self, prev, idx_next, alpha, delta_t, d1, P0, M0, M1,
                        mode_init, mode_end, s, d1_end=None, alpha_b=None, d0=None, P1=None, d1_prod=None):
        """
        Fill the flow-matching states prev (PrevProt) at rows idx_next from the
        previous-time states P0, for all cells at once. For target g, regulators follow the flow
        over delta_t * alpha_mod, alpha_mod = min(alpha[:, g] + .1, 1) (+.1 lets
        the mode stabilize). Arrays P0..mode_end and alpha are (N, G_genes). d1 / d1_end: relaxation
        rates before / after the switch of the birth rate of each regulator, at alpha_b (birth_alpha); d1_end=None:
        no dilution switch (find_next_prot). d0 given: mRNA bridges (prot_bridge, production self.d[1]) conditioned
        on the end proteins P1, M0 / M1 being the mRNA in kon units and mode_init / mode_end the kon levels.
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
            if d0 is not None:
                r_end = d1 if d1_end is None else d1_end
                a_b = alpha if alpha_b is None else alpha_b
                prod = self.d[1, ns:] if d1_prod is None else d1_prod  # production d1 ((G,) or per row)
                prev.values[g][dest] = prot_bridge(
                    prod[..., r], d0[..., r], d1[..., r], r_end[..., r], P0[:, r], M0[:, r], M1[:, r],
                    mode_init[:, r], mode_end[:, r], alpha[:, r], a_b[:, r], delta_t, delta_t * alpha_mod, P1[:, r])
            elif d1_end is None:
                prev.values[g][dest] = find_next_prot(
                    d1[..., r], P0[:, r], M0[:, r], M1[:, r], mode_init[:, r], mode_end[:, r],
                    np.minimum(alpha[:, r] / alpha_mod, 1),
                    s[..., r] if np.ndim(s) else s, delta_t * alpha_mod)
            else:
                prev.values[g][dest] = find_next_prot_dilution(
                    d1[..., r], d1_end[..., r], P0[:, r], mode_init[:, r], mode_end[:, r],
                    np.minimum(alpha[:, r] / alpha_mod, 1), np.minimum(alpha_b[:, r] / alpha_mod, 1),
                    s[..., r] if np.ndim(s) else s, delta_t * alpha_mod)

    def _fill_path_prot(self, path, idx_next, alpha, delta_t, d1, P0, M0, M1,
                        mode_init, mode_end, s, d1_end=None, alpha_b=None, d0=None, P1=None, P_start=None, stim_end=None,
                        d1_prod=None):
        """
        Exact flow matching: fill the path states path (PathProt) at rows idx_next, with the arguments of
        _fill_prev_prot: node 0 = start state P_start (full row, stimuli at the start time), node q = the bridge at
        q / Q of the interval for every gene (same flows as _fill_prev_prot, each gene switching at its own alpha;
        stimuli at their end value stim_end), and the weights of the mRNA memory of each target (path_weights, d0).
        """
        ns = self.n_stimuli
        Qn = path.nodes.shape[1]
        rows = np.asarray(idx_next)
        path.nodes[rows, 0] = P_start
        for q in range(1, Qn):
            f = q / Qn
            if d0 is not None:
                P = prot_bridge(self.d[1, ns:] if d1_prod is None else d1_prod, d0, d1, d1 if d1_end is None else d1_end,
                                P0, M0, M1,
                                mode_init, mode_end, alpha, alpha if alpha_b is None else alpha_b,
                                delta_t, delta_t * f, P1)
            elif d1_end is None:
                P = find_next_prot(d1, P0, M0, M1, mode_init, mode_end, np.minimum(alpha / f, 1), s, delta_t * f)
            else:
                P = find_next_prot_dilution(d1, d1_end, P0, mode_init, mode_end, np.minimum(alpha / f, 1),
                                            np.minimum(alpha_b / f, 1), s, delta_t * f)
            path.nodes[rows, q, :ns] = stim_end
            path.nodes[rows, q, ns:] = P
        path.weights[rows] = path_weights(self.d[0], float(delta_t), Qn)[None]

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
        d1 : (G_genes,), or (T-1, G_genes) per interval (mean over the intervals of each pair of observed times),
            or (T-1, S, G_genes) per sample (kinetics per network condition, routed by samples_data).
        """
        if kon_beta is None:
            kon_beta = self.kon_beta
        if s is None:
            s = self.scale_proteins
        ns = self.n_stimuli
        prot_modified = y_prot.copy()
        # Flow-matching states start from the input trajectories
        prot_modified_prev = None
        exact_fm = prev_cols is not None and self.exact_flow_matching
        if exact_fm:
            prot_modified_prev = PathProt.from_states(prot_modified, self.flow_matching_nodes)
        elif prev_cols is not None:
            prot_modified_prev = PrevProt(prev_cols, [prot_modified[:, c] for c in prev_cols])

        # Intervals between consecutive observed times of each trajectory (in time order)
        valid = self._valid() if len(self._valid()) == len(y_prot) else np.ones(len(y_prot), dtype=bool)
        d1_all = np.asarray(d1, dtype=np.float64)
        for a, b, slots in trajectory_pairs(valid, len(times)):
            delta_t = times[b] - times[a]
            d1 = d1_all if d1_all.ndim == 1 else d1_all[a:b].mean(axis=0)
            rows_prev, rows_next = N * a + slots, N * b + slots
            if d1.ndim == 2:  # per sample: rate of each trajectory (rows)
                pos = np.minimum(np.asarray(self.samples_data)[rows_prev].astype(int), d1.shape[0] - 1)
                d1 = d1[pos]
            kb_prev = np.ascontiguousarray(kon_beta[rows_prev, ns:], dtype=np.float64)
            kb_next = np.ascontiguousarray(kon_beta[rows_next, ns:], dtype=np.float64)

            P0 = np.ascontiguousarray(prot_modified[rows_prev, ns:], dtype=np.float64)
            alpha_ab = np.ascontiguousarray(self.alpha[a][slots], dtype=np.float64)
            # Dilution at the birth rate of the start state before alpha, of the end state after:
            # rate d1 + b, mode targets compressed by c = d1 / (d1 + b)
            rate, c = np.asarray(d1, dtype=np.float64), 1.0
            rate_n, c_n, alpha_b = None, c, None  # no dilution: find_next_prot
            if birth is not None:
                b_all = np.asarray(birth, dtype=np.float64)
                d1r = np.asarray(d1, dtype=np.float64)
                d1r = d1r if d1r.ndim == 2 else d1r[None, :]
                rate_n = d1r + b_all[rows_next][:, None]
                rate = d1r + b_all[rows_prev][:, None]
                c = d1r / rate
                c_n = d1r / rate_n
                # Birth rate switched at birth_alpha in the equation of each gene (alpha itself unchanged)
                alpha_b = birth_alpha(alpha_ab, kb_prev, kb_next, self.alpha_threshold)
                prot_modified[rows_next, ns:] = find_next_prot_dilution(
                    rate, rate_n, P0, kb_prev * c, kb_next * c_n, alpha_ab, alpha_b, s, float(delta_t))
            else:
                prot_modified[rows_next, ns:] = find_next_prot(
                    rate, P0, kb_prev, kb_next, kb_prev, kb_next, alpha_ab, s, float(delta_t))
            for k in range(a + 1, b):  # timepoints missed inside the interval: along the same flow
                prot_modified[N * k + slots, ns:] = prot_along(
                    rate, P0, kb_prev * c, kb_next * c_n, alpha_ab, s, delta_t, times[k] - times[a], d1_end=rate_n,
                    alpha_b=alpha_b)

            if prot_modified_prev is not None and (self.weight_prev > 0 or exact_fm):
                fill = self._fill_prev_prot
                if exact_fm:
                    fill = partial(self._fill_path_prot, P_start=prot_modified[rows_prev],
                                   stim_end=prot_modified[rows_next, :ns])
                fill(
                    prot_modified_prev, rows_next, self.alpha[a][slots], delta_t,
                    rate, prot_modified[rows_prev, ns:], kb_prev, kb_next, kb_prev * c, kb_next * c_n, s,
                    d1_end=rate_n, alpha_b=alpha_b)
        if not valid.all():
            fill_virtual([prot_modified], valid, len(times), inside=False)

        return prot_modified, prot_modified_prev
    

    def _kon_ref_per_sample(self, y_prot, ks, inter, basal, samples_id=None, samples_data=None):
        """
        Compute kon_ref_vector respecting per-sample basal when basal is 2-D.

        When basal is 1-D (G,), falls back to a single kon_ref call.
        When basal is 2-D (n_samples, G), loops over samples and
        assembles the result using samples_data (or self.samples_data) as the
        routing key so that each cell uses its own sample's basal.
        """
        if basal.ndim < 2 and ks.ndim < 3 and inter.ndim < 3:
            return kon_ref_vector(y_prot, ks, inter, basal)
        sd = samples_data if samples_data is not None else self.samples_data
        if samples_id is None:
            samples_id = np.sort(np.unique(sd))
        out = np.zeros((y_prot.shape[0], y_prot.shape[1]))
        for s_idx, s in enumerate(samples_id):
            mask = (sd == s)
            if not np.any(mask):
                continue
            basal_s = basal[min(s_idx, basal.shape[0] - 1)] if basal.ndim == 2 else basal
            out[mask] = kon_ref_vector(y_prot[mask], ks_of(ks, s_idx), inter_of(inter, s_idx), basal_s)
        return out

    def _count_errors_per_sample(self, y_prot, kon_beta, ks, inter, basal,
                                  samples_id=None, samples_data=None):
        """
        Weighted-average count_errors respecting per-sample basal.
        When basal is 1-D, delegates to count_errors directly.

        samples_data : per-cell sample assignment; defaults to self.samples_data.
        """
        if basal.ndim < 2 and ks.ndim < 3 and inter.ndim < 3:
            return count_errors(y_prot, kon_beta, ks, basal, inter,
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
            basal_s = basal[min(s_idx, basal.shape[0] - 1)] if basal.ndim == 2 else basal
            err_s = count_errors(y_prot[mask], kon_beta[mask],
                                 ks_of(ks, s_idx), basal_s, inter_of(inter, s_idx),
                                 n_stimuli=self.n_stimuli)
            total_err += err_s * n_s
            total_cells += n_s
        return total_err / total_cells if total_cells > 0 else 0.0

    def refine_network_degradations(self, verb=True, stimulus_schedule=None, test=False):
        """
        Refine network parameters and infer degradation rates for simulation.

        The simulation model (ODE without simulation_stochastic, protein-only bursty PDMP, explicit-mRNA PDMP with
        simulation_mrnas_direct) is fitted free-running (no teacher forcing, fit_free_running) from the trajectory
        states on the kon of the observed modes (and the observed mRNA with explicit mRNA): d1, eps = d1 / d0 and
        the scale of theta per target gene (per interval with use_temporal_degradations), from the literature
        (d1_eff = d0 d1 / (d0 + d1) and d0 / d1_eff = 1 + r for the ODE and the protein-only PDMP, d1 and r = d0 / d1
        with explicit mRNA) of the rates of fit_network (start and rappel), with the birth rates of the states (protein
        dilution) and the branching. The trajectories and kon_theta of fit_network are kept (never replaced).
        Windows start from the unitary proteins (modes, literature d1_eff) for the ODE and the protein-only PDMP, from
        the proteins of fit_network with explicit mRNA.
        - refine_theta_with_sde = True: network of fit_network, refined in that fit (support and signs kept, rappel
          sde_lambda_theta).
        - refine_theta_with_sde = False: network refitted (final = 1) on these proteins (kon averaged over the
          literature protein noise for the protein-only PDMP), then fixed.
        With network conditions and degradation_rates_per_condition, one fit per condition (stored per sample).
        d1_eff kept in d1_eff_t.

        When ``test=True``, only runs the trajectory estimation step and recomputes
        ``kon_theta`` using the current (pre-loaded simul) network. No inference
        or parameter update is performed.
        """
        seed_everything(self.seed)

        times = np.sort(np.unique(self.times_data))
        N_tot = np.sum(self.times_data == times[0])

        if stimulus_schedule is not None or self._stim_schedule is None:
            self._stim_schedule = self._build_stimulus_schedule(times, stimulus_schedule)
        # Birth / death of the states consistent with R_opt (growth pass of fit_network, or of infer_test along its
        # couplings): dilution of the trajectories, network, d1 / d0, MLP heads
        self.fit_birth_death_to_growth(verb=verb)

        ns = self.n_stimuli
        # --- Adapt ref_network ---
        for g in range(ns, self.ref_network.shape[0]):
            l_max = len(np.unique(self.modes[:, g]))
            if l_max < 2: # If only one mode
                self.ref_network[g, :], self.ref_network[:, g] = 0, 0

        ks, _, k1 = self._mixture_terms()
        # basal: (n_samples, G_tot); inter: (G_tot, G_tot), or (n_samples, G_tot, G_tot) with network conditions
        basal, inter = self.basal.copy(), self.inter.copy()

        mrna_sim = bool(self.simulation_mrnas_direct)
        stoch = bool(self.simulation_stochastic) or mrna_sim
        d0_lit, d1_lit = self.d[0].copy(), self.d[1].copy()
        r_lit = np.maximum(d0_lit / d1_lit, self.min_ratio)
        # Rate of the protein-only model on the modes: d1_eff (mean delay 1/d0 + 1/d1 of the mRNA-protein chain)
        d1_unit = d1_lit * r_lit / (1 + r_lit)

        if test:
            # In test mode: estimate protein trajectories along pre-inferred OT couplings
            # and recompute kon_theta using the simul network. No inference is run.
            # Trajectories on the modes, as those of the simulation network: literature d1_eff (deterministic),
            # d1 of the simulations (stochastic), transferred from the training intervals (train_times) to the test
            # ones as in the simulations (_transfer_weights)
            if stoch or self.inference_mrnas_direct:
                d1_test = d1_unit
                if stoch and self.d_t is not None:
                    d1_sim = np.asarray(self.d_t)[:, 1]
                    train_times = getattr(self, 'train_times', None)
                    if train_times is None:
                        if len(d1_sim) != len(times) - 1:
                            raise ValueError("refine_network_degradations(test=True): train_times needed to transfer "
                                             "the d1 of the simulations to the test intervals")
                        d1_test = d1_sim
                    else:
                        d1_test = np.tensordot(self._transfer_weights(np.sort(np.unique(train_times)), times),
                                               d1_sim, axes=1)  # (T-1, [S,] G)
                self.prot, _ = self.estimate_trajectories(
                    self.prot, times, d1_test[..., ns:], N=N_tot, kon_beta=self.kon_beta, s=self.scale_proteins,
                    birth=self._state_birth())
            y_prot = self.prot
            kon_vector = self.kon_beta.copy()
            kon_vector[:, ns:] = self._kon_ref_per_sample(
                y_prot, ks, inter, basal, samples_data=self.samples_data)[:, ns:]
            self.kon_theta = kon_vector
            return

        samples_id = np.sort(np.unique(self.samples_data))
        # Observed states (virtual ones: timepoints a sample misses) for the fits and errors
        valid = self._valid()
        # Birth rate of each state for the dilution, as in the simulations (MLP birth head or the same rates)
        b_states = self._state_birth(verb=verb)

        # Network conditions without separate fits (degradation_rates_per_condition = False): one shared fit from the
        # mean network of the conditions (kept per sample)
        if self.sample_conditions is not None and not self.per_condition_fits()[0] and np.ndim(inter) == 3:
            nets = np.stack(list(self.condition_networks(inter).values()))
            inter = np.repeat(nets.mean(axis=0)[None], inter.shape[0], axis=0)
            if verb:
                print("[refine_network_degradations] One shared refinement from the mean network of the conditions")
        # Simulation model and its literature rates: ODE and protein-only PDMP at d1_eff (eps = d1_eff / d0 =
        # 1 / (1 + r)), explicit mRNA at d1 (eps = d1 / d0 = 1 / r)
        mode = 'sde2' if mrna_sim else ('sde1' if stoch else 'ode')
        d1_start = d1_lit if mode == 'sde2' else d1_unit
        eps_start = 1.0 / r_lit if mode == 'sde2' else 1.0 / (1 + r_lit)
        # Start and rappel: the rates of fit_network (learned with refine_in_loop, else the literature)
        # Window starts: unitary proteins (modes, literature d1_eff) for the ODE and the protein-only PDMP, the
        # proteins of fit_network (mRNA-driven) with explicit mRNA
        unitary = mode != 'sde2'
        if self.refine_theta_with_sde:
            # Network of fit_network, refined in the free-running fit (no refit here)
            y_prot = self.estimate_trajectories(self.prot, times, d1_unit[ns:], N=N_tot, kon_beta=self.kon_beta,
                                                s=self.scale_proteins, birth=self._state_birth())[0] \
                if unitary else self.prot
        else:
            # Network refitted (final = 1) on these proteins, kon averaged over the literature protein noise for the
            # protein-only PDMP (q = 1 / (1 + r)); not with explicit mRNA (the mRNA-driven proteins already carry
            # the noise), then fixed in the free-running fit
            noise_coef = None
            if stoch and unitary:
                k1m = k1.mean(axis=0) if np.ndim(k1) == 2 else k1
                noise_coef = self.scale_proteins / (1 + r_lit) / np.maximum(k1m, 1e-12)
                noise_coef[:ns] = 0.0
            y_prot, basal, inter = self._refine_trajectories_network(
                self.kon_beta, d1_unit[ns:], times, N_tot, ks, valid, samples_id, verb=verb, noise_coef=noise_coef,
                y_prot_given=None if unitary else self.prot)
        # These proteins only start the windows of the fit: self.prot and self.kon_theta stay those of fit_network
        # Proliferation MLP first (on the trajectories of fit_network): dilution and branching of the fit
        self._train_proliferation_mlp(verb=verb)
        basal, inter, d1_c, eps_c, scale_c, cond = self._fit_free_running(
            y_prot, basal, inter, mode, d1_start, eps_start, valid, times, N_tot, ks, k1, samples_id, b_states,
            verb=verb)
        T1 = len(times) - 1
        if d1_c.shape[0] > 1:
            # Kinetics per network condition, stored per sample (as inter): d_t (T-1, 2, S, G), ratios (T-1, S, G)
            per_s = lambda v: np.swapaxes(v[cond], 0, 1)
            d1_t, eps_t, scale_t = per_s(d1_c), per_s(eps_c), per_s(scale_c)
            self.d_t = np.tile(self.d[None, :, None, :], (T1, 1, len(cond), 1))
            basal_t = basal[None] * scale_t                          # (T-1, S, G)
            inter_t = inter[None] * scale_t[:, :, None, :]           # (T-1, S, G, G)
        else:
            d1_t, eps_t, scale_t = d1_c[0], eps_c[0], scale_c[0]
            self.d_t = np.tile(self.d, (T1, 1, 1))
            # Scale of each target gene: last axis of basal (G) and of inter (regulator, target)
            basal_t = basal[None] * scale_t[:, None, :]
            inter_t = inter[None] * (scale_t[:, None, :] if inter.ndim == 2 else scale_t[:, None, None, :])
        self.d_t[:, 1] = d1_t
        self.ratios = 1.0 / eps_t if stoch else np.broadcast_to(r_lit, d1_t.shape).copy()  # unused by the ODE
        self.d1_eff_t = d1_t / (1 + eps_t) if mode == 'sde2' else d1_t
        if verb:
            print(f"[refine_network_degradations] free-running {mode} fit ({d1_c.shape[0]} kinetics): d1 median "
                  f"{np.median(d1_t[..., ns:]):.4g}, d0 / d1 median {np.median(self.ratios[..., ns:]):.3g} "
                  f"(fit_network {np.median(1.0 / eps_start[ns:]):.3g}), scale median {np.median(scale_t[..., ns:]):.3g}")

        self.basal, self.inter = basal, inter
        self.basal_t, self.inter_t = basal_t, inter_t

        if verb:
            basal_mean = basal.mean(axis=0) if basal.ndim == 2 else basal
            print('[refine_network_degradations]  Static simulation network',
                  np.swapaxes(self.inter, -2, -1), basal_mean)

        # Stimulus dimensions; d0 kept distinct from d1 (temporal rates with or without a sample axis)
        self.d[0, :ns], self.d_t[:, 0, ..., :ns] = 1.0, 1.0
        self.d[1, :ns], self.d_t[:, 1, ..., :ns] = 0.2, 0.2
        eq = self.d[0] == self.d[1]
        self.d[0, eq] = self.d[1, eq] + 1e-6
        d0_t, d1_t = self.d_t[:, 0], self.d_t[:, 1]
        eq = d0_t == d1_t
        d0_t[eq] = d1_t[eq] + 1e-6

    def _literature_rates(self, mode):
        """Literature (d1, eps = d1 / d0) of the model of mode, rappel targets of refine_in_loop: d1_eff and
        1 / (1 + r) for the ODE and the protein-only PDMP, d1 and 1 / r with explicit mRNA (r = d0 / d1)."""
        d = self.d if self.d_literature is None else np.asarray(self.d_literature, dtype=float)
        r = np.maximum(d[0] / d[1], self.min_ratio)
        if mode == 'sde2':
            return d[1].copy(), 1.0 / r
        return d[1] * r / (1 + r), 1.0 / (1 + r)

    def _train_proliferation_mlp(self, verb=True):
        """Proliferation MLP along the trajectories self.prot (branching and birth rate of the dilution)."""
        ns = self.n_stimuli
        # Proliferation MLP for the simulations: branching (simulate_with_proliferation) and birth rate of the dilution
        if (self.simulate_with_proliferation or self.protein_dilution) and self.R_opt is not None:
            if verb:
                print("[refine_network_degradations] Training ProliferationMLP on R_opt...")
            # Rate without the inference stimuli when their effects are given (added back in the simulations);
            # the MLP then takes no stimulus input
            R_target = self.R_opt if self.R_stim_offset is None else self.R_opt - self.R_stim_offset
            # Two heads (birth, death) on the split of the states consistent with R_opt, without the stimulus if its
            # effects are given apart (R_stim_offset)
            base = self.R_stim_offset is not None
            b_prior, d_prior = self._state_rates(base=base)  # consistent with R_opt (fit_birth_death_to_growth)
            kw = dict(ns=ns, n_nodes=self.n_growth_nodes, with_stim=self.prolif_uses_stimulus and self.R_stim_offset is None,
                      seed=self.seed, verb=verb)
            if self.sample_conditions is not None and self.per_condition_fits()[1]:
                # One MLP per condition, on its trajectories and the proteins of its genes
                from ..inference.proliferations import ConditionProliferation
                T = len(np.unique(self.times_data))
                N_tot = len(self.times_data) // T
                cond = np.asarray(self.sample_conditions, dtype=int)
                slot_cond = cond[np.minimum(np.asarray(self.samples_data[:N_tot]).astype(int), len(cond) - 1)]
                nets, cols = [], []
                for c in range(int(cond.max()) + 1):
                    first = int(np.flatnonzero(cond == c)[0])
                    g_idx = (np.arange(self.prot.shape[1] - ns) if self.gene_mask is None
                             else np.flatnonzero(self.gene_mask[first, ns:]))
                    slots = np.flatnonzero(slot_cond == c)
                    if not len(slots):
                        nets.append(None)
                        cols.append(g_idx)
                        continue
                    rows = (np.arange(T)[:, None] * N_tot + slots[None, :]).ravel()
                    take = lambda v: None if v is None else np.asarray(v)[rows]
                    if verb:
                        print(f"[refine_network_degradations] ProliferationMLP of condition {self.condition_names[c]} "
                              f"({len(slots)} trajectories, {len(g_idx)} genes)")
                    nets.append(train_proliferation_mlp(
                        self.prot[rows][:, np.concatenate([np.arange(ns), ns + g_idx])], R_target[rows],
                        self.times_data[rows], birth_prior=take(b_prior), death_prior=take(d_prior), **kw))
                    cols.append(g_idx)
                self.prolif_network = ConditionProliferation(nets, cols, cond, list(self.condition_names))
            else:
                self.prolif_network = train_proliferation_mlp(
                    self.prot, R_target, self.times_data, birth_prior=b_prior, death_prior=d_prior, **kw)
            if verb:
                print("[refine_network_degradations] ProliferationMLP training done.")

    def _fit_free_running(self, y_prot, basal, inter, mode, d1_start, eps_start, valid, times, N_tot, ks, k1,
                          samples_id, b_states, verb=True, samples_rows=None, kon_targets=None, rna=None,
                          prolif='model', temporal=None, learn_scale=None, learn_theta=None, epochs=None,
                          per_condition=True, d1_ref=None, eps_ref=None, only_max_horizon=False, lam_rates=None,
                          tag='refine_network_degradations'):
        """
        d1, eps = d1 / d0 and the scale of theta per target gene (per interval with use_temporal_degradations) by
        the free-running fit of the simulation model (fit_free_running: 'ode', 'sde1' protein-only PDMP, 'sde2'
        explicit mRNA) from the trajectory states, on the kon of the observed modes; the network (support and
        signs kept) refined with refine_theta_with_sde, else fixed. Dilution and branching as in the simulations.
        The keyword arguments override the attributes for the fit inside fit_network (refine_in_loop): trajectory
        arrays (samples_rows, kon_targets, rna), prolif (None: no MLP), temporal / learn_scale / learn_theta /
        epochs, per_condition, rappel targets d1_ref / eps_ref (default the starting values) and strength lam_rates
        (default sde_lambda_rates), only_max_horizon.
        Returns (basal, inter, d1, eps, scale (C, T-1, G), condition of each sample position (S,)).
        """
        import torch
        from ..inference.sde import fit_free_running
        ns = self.n_stimuli
        T, G = len(times), y_prot.shape[1]
        kb = sigmoid_amplitudes(ks)  # (G, 2) or (S, G, 2): lowest / highest normalised mode
        c_nb = self.a[..., -1, :]
        samples_rows = np.asarray(self.samples_data if samples_rows is None else samples_rows)
        kon_targets = self.kon_beta if kon_targets is None else kon_targets
        X = y_prot.reshape(T, N_tot, G)
        R = None if mode != 'sde2' else np.asarray(self.rna if rna is None else rna, dtype=float).reshape(T, N_tot, G)
        prolif = self.prolif_network if prolif == 'model' else prolif
        bt_all = None if b_states is None else np.asarray(b_states, dtype=float).reshape(T, N_tot)
        # Kinetics per network condition (index of each sample position): one separate fit per condition (its samples,
        # its genes, its network as rappel), else one shared fit
        separate = (per_condition and self.per_condition_fits()[0] and self.sample_conditions is not None
                    and np.ndim(inter) == 3)
        if separate:
            cond = np.asarray(self.sample_conditions, dtype=int)
        else:
            cond = np.zeros(len(samples_id), dtype=int)
        groups = []
        for s_idx, s in enumerate(samples_id):
            slots = np.flatnonzero(samples_rows[:N_tot] == s)
            if not len(slots):
                continue
            kb_s = kb[min(s_idx, kb.shape[0] - 1)] if kb.ndim == 3 else kb
            k1_s = np.asarray(s1_of(k1, s_idx), dtype=float)
            stim = np.array([self._stim_schedule.per_cell(times[t + 1], np.array([s]))[0]
                             for t in range(T - 1)], dtype=float) * self.scale_proteins
            g = dict(X=X[:, slots].astype(float), k_lo=kb_s[:, 0], k_hi=kb_s[:, 1], k1=k1_s, stim=stim,
                     cond=int(cond[min(s_idx, len(cond) - 1)]),
                     idx=min(s_idx, basal.shape[0] - 1) if inter.ndim < 3 else min(s_idx, inter.shape[0] - 1),
                     targets=[None if not np.any(valid[N_tot * t + slots]) else
                              kon_targets[N_tot * t + slots][valid[N_tot * t + slots]] for t in range(T)],
                     start_ok=np.array([np.any(valid[N_tot * t + slots]) for t in range(T)]))
            if mode == 'sde2':  # normalised mRNA m = M c s / k1 (mean kon s, as the bridges)
                M = R[:, slots] * (self.scale_proteins * np.asarray(s1_of(c_nb, s_idx)) / np.maximum(k1_s, 1e-12))[None, None, :]
                M[..., :ns] = 0.0
                g['M'] = M
                g['M_targets'] = [None if g['targets'][t] is None else
                                  M[t][valid[N_tot * t + slots]] for t in range(T)]
            # Birth rate of the dilution (MLP birth head on the simulated states, else the rates of the slots)
            # and net rate of the branching, as in simulate_trajectories
            g['birth'], g['rate'] = None, None
            if self.protein_dilution and prolif is not None:
                g['birth'] = lambda P, t, sl, st=stim, si=s_idx: prolif.predict_birth(
                    P[:, ns:], np.repeat(st[t][None], len(P), 0), samples=np.full(len(P), si))
            elif self.protein_dilution and bt_all is not None:
                bt = bt_all[:, slots]
                g['birth'] = lambda P, t, sl, bt=bt: 0.5 * (bt[t, sl] + bt[t + 1, sl])
            if self.simulate_with_proliferation and prolif is not None:
                g['rate'] = lambda P, t, st=stim, si=s_idx: prolif.predict(
                    P[:, ns:], np.repeat(st[t][None], len(P), 0), samples=np.full(len(P), si))
            act = self.active_genes(s_idx)
            g['gmask'] = None if act is None else act.astype(float)  # genes of the sample's condition
            groups.append(g)
        basal_rows = np.asarray(basal, dtype=float)
        T1 = T - 1
        n_threads_prev = torch.get_num_threads()
        torch.set_num_threads(os.cpu_count() or 1)
        pick = lambda v, default: default if v is None else v
        tile = lambda v: None if v is None else np.tile(v, (T1, 1))
        def run(gs, theta0, basal0, n_cond, key):
            return fit_free_running(
                gs, times, ns, np.asarray(theta0, dtype=float), basal0,
                np.tile(d1_start, (T1, 1)), np.tile(eps_start, (T1, 1)), np.ones((T1, G)), mode=mode,
                learn_theta=bool(pick(learn_theta, self.refine_theta_with_sde)), learn_d=True,
                learn_eps=mode != 'ode', learn_scale=bool(pick(learn_scale, self.fit_scale_theta)),
                temporal=bool(pick(temporal, self.use_temporal_degradations)),
                lam_theta=self.sde_lambda_theta, lam_rates=pick(lam_rates, self.sde_lambda_rates),
                eps_bounds=(1.0 / self.max_ratio, 1.0 / self.min_ratio), epochs=pick(epochs, self.sde_epochs),
                lr=self.sde_lr, n_sub=self.sde_n_sub, batch_size=self.sde_batch_size,
                scale_p=float(self.scale_proteins), max_horizon=self.sde_max_horizon, n_windows=self.sde_windows,
                n_cond=n_cond, d1_ref=tile(d1_ref), eps_ref=tile(eps_ref),
                only_max_horizon=only_max_horizon,
                seed=task_seed(self.seed, *key), verb=verb > 1 if tag == 'fit_network' else verb)

        try:
            if not separate:
                theta, basal_new, d1_t, eps_t, scale_t, hist = run(groups, inter, basal_rows, 1, (3,))
                hists = [hist]
            else:
                # One fit per condition on its samples (rows of inter / basal remapped), stacked per condition
                theta, basal_new = np.array(inter, dtype=float), basal_rows.copy()
                rates, hists = [], []
                for c in range(int(cond.max()) + 1):
                    rows = np.flatnonzero(cond == c)
                    gs = [dict(g, idx=int(np.flatnonzero(rows == g['idx'])[0]), cond=0)
                          for g in groups if g['cond'] == c]
                    if not gs:
                        rates.append(None)
                        continue
                    th_c, ba_c, d1_c, eps_c, sc_c, hist = run(gs, theta[rows], basal_rows[rows], 1, (3, c))
                    theta[rows], basal_new[rows] = th_c, ba_c
                    rates.append((d1_c[0], eps_c[0], sc_c[0]))
                    hists.append(hist)
                # Conditions without trajectories: the starting rates
                start = (np.tile(d1_start, (T1, 1)), np.tile(eps_start, (T1, 1)), np.ones((T1, G)))
                d1_t, eps_t, scale_t = (np.stack([(r if r is not None else start)[k] for r in rates]) for k in range(3))
        finally:
            torch.set_num_threads(n_threads_prev)
        if verb:
            print(f"[{tag}] free-running {mode} loss " + '; '.join(f"{h['before']:.4f} -> {h['after']:.4f}" for h in hists)
                  + (" (one fit per condition)" if separate else ""))
        return basal_new, theta, d1_t, eps_t, scale_t, cond

    def _refine_trajectories_network(self, kon_targets, d1, times, N_tot, ks, valid, samples_id, verb=True,
                                     noise_coef=None, y_prot_given=None):
        """Proteins along the couplings of fit_network towards kon_targets (relaxation d1) and network refitted on
        them (inter_simul_ref as reference if given; noise_coef: kon averaged over the protein noise, see
        inference_network_multi; y_prot_given: these proteins instead, without flow-matching term). Returns
        (y_prot, basal, inter)."""
        ns = self.n_stimuli
        basal_ref, inter_ref = self.basal.copy(), self.inter.copy()
        if self.inter_simul_ref is not None:
            inter_ref = self._normalize_theta(self.inter_simul_ref, self.inter.shape[-1], is_inter=True)
        # Network conditions refitted together (degradation_rates_per_condition = False): mean network as reference
        common = self.sample_conditions is not None and not self.per_condition_fits()[0]
        n_rows = np.shape(self.inter)[0] if np.ndim(self.inter) == 3 else None
        if common and np.ndim(inter_ref) == 3:
            inter_ref = np.stack(list(self.condition_networks(inter_ref).values())).mean(axis=0)
        # Same active regulators as in the inference_network call below
        _, prev_cols = active_regulators(self.ref_network, inter_ref, self.hard_forcing_ref)
        if y_prot_given is None:
            y_prot, y_prot_prev = self.estimate_trajectories(self.prot, times, d1, N=N_tot, kon_beta=kon_targets,
                                                             s=self.scale_proteins, prev_cols=prev_cols,
                                                             birth=self._state_birth())
        else:
            y_prot = np.asarray(y_prot_given, dtype=float)
            y_prot_prev = PathProt.from_states(y_prot, self.flow_matching_nodes) if self.exact_flow_matching \
                else PrevProt(prev_cols, [y_prot[:, c] for c in prev_cols])
        error = self._count_errors_per_sample(y_prot[valid], self.kon_beta[valid], ks, self.inter, self.basal,
                                              samples_id=samples_id, samples_data=self.samples_data[valid])

        def fit_on(sels):
            return inference_network_multi(
                sels, self.samples_data, self.kon_beta,
                y_prot, y_prot_prev, ks, n_stimuli=ns,
                ref_network=self.ref_network, basal_init=basal_ref, inter_init=inter_ref,
                basal_ref=basal_ref, inter_ref=inter_ref,
                scale=self.scale_pen * 2, # # slightly stronger regularization for network
                weight_prev=self.weight_prev, final=1,
                samples_id=samples_id,
                constrain_basal_uniform=self.constrain_basal_uniform,
                hard_forcing_ref=self.hard_forcing_ref, ref_constraint_pct=self.ref_constraint_pct,
                seuil_zero_min_ref=self.seuil_zero_min_ref,
                # Separate per condition (no fused term) with degradation_rates_per_condition, else a common network
                sample_conditions=self.sample_conditions if self.per_condition_fits()[0] else None,
                condition_pen=0.0, gene_mask=self.gene_mask,
                noise_coef=noise_coef,
            )

        # Mean theta over fits on (time, sample, cell type)-stratified subsamples (single fit if one holds all)
        basal, inter, _, _ = self._fit_theta_averaged(
            fit_on, self.times_data, self.samples_data, self.traj_cell_types, self.n_network_fits,
            self._n_params_per_target(prev_cols, len(samples_id)), valid=valid)
        if common and n_rows is not None and np.ndim(inter) == 2:  # kept per sample (one network for every condition)
            inter = np.repeat(np.asarray(inter)[None], n_rows, axis=0)

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

        error_corrected = self._count_errors_per_sample(y_prot[valid], self.kon_beta[valid], ks, inter, basal,
                                                        samples_id=samples_id, samples_data=self.samples_data[valid])
        if verb:
            print("[refine_network_degradations] ratio errors", error, error_corrected)
        return y_prot, basal, inter

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

    def simulate_trajectories(self, times, times_train, ks, N=100, verb=True, samples_data=None):
        """
        Simulate the trajectories of N cells from the first-timepoint states. Only the extreme modes of ks (bounds
        of the sigmoid) are used: intermediate modes only serve the inference.

        Proteins: ODE (simulation_stochastic = 0) or bursty PDMP; with simulation_mrnas_direct, the PDMP carries
        explicit mRNA (mRNA bursts, proteins translated from them) and returns mRNA counts (Poisson of the mRNA level;
        NB of the modes on the ODE intervals). With self.chromatin, each cell carries the levels of the chromatin
        regions (Markov chain regulated by the proteins, gating kon).

        Returns (proteins, mRNA counts or None, kon, chromatin accessibility or None, times), rows = N cells per time.
        """
        ns = self.n_stimuli
        ks = sigmoid_amplitudes(ks)
        G_tot = self.prot.shape[1]
        mrna_mode = bool(self.simulation_mrnas_direct)
        chrom0 = self._chromatin_model(G_tot)

        start_time = 0
        # We want to capture automatically if we simulate from after last timepoint
        if times_train[-1] < times[1]: # times[0] = 0 by construction
            times = [0, times_train[-1]] + list(times[1:])
            start_time = 1
        prot_modified = np.ones((N * len(times), G_tot))
        kon_vector = np.ones((N * len(times), G_tot))
        # Starting states: first timepoint, and the last one when simulating beyond the data
        # Network in force at the first timepoint: that of the first interval (scale of theta included)
        starts = [(0, 0, self.inter_t[0], self.basal_t[0])]
        if start_time:
            l = len(times_train)
            starts.append((N, N * (l - 1), self.inter_t[-1], self.basal_t[-1]))
        for dst, src, inter_0, basal_0 in starts:
            prot_modified[dst:dst + N, :] = self.prot[src:src + N, :]
            kon_vector[dst:dst + N, :] = self.kon_beta[src:src + N, :]
            kon_vector[dst:dst + N, ns:] = self._kon_ref_per_sample(
                self.prot[src:src + N, :], ks, inter_0, basal_0, samples_data=samples_data)[:, ns:]
        # mRNA levels (continuous, count units) and counts; chromatin levels (allele, region) of each row
        m_state = m_counts = chrom_lvl = None
        if mrna_mode:
            m_state = np.zeros((N * len(times), G_tot))
            for dst, src, _, _ in starts:
                m_state[dst:dst + N, :] = self.rna[src:src + N, :]
            m_counts = m_state.copy()
        if chrom0 is not None:
            chrom_lvl = np.repeat(chrom0.lvl[None], N * len(times), axis=0)

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
        # Rates of the simulated intervals ([S,] axis kept: kinetics per network condition)
        self.d_t = np.zeros((len(times)-1,) + d_t_train.shape[1:], dtype=float)
        self.ratios = np.zeros((len(times)-1,) + ratios_train.shape[1:], dtype=float)
        # Per-sample basal_t (T-1, n_samples, G) keeps its structure
        basal_t = np.zeros((len(times)-1,) + basal_t_train.shape[1:], dtype=float)
        inter_t = np.zeros((len(times)-1,) + inter_t_train.shape[1:], dtype=float)  # [n_samples,] G, G
        # Rates of each simulated interval from those of the inference intervals (overlap-weighted)
        # Intervals of the rates: the training ones (train_times in infer_test, whose times_data are the test ones)
        W = self._transfer_weights(times_train if self.train_times is None else np.sort(self.train_times), times)
        self.d_t[:] = np.tensordot(W, d_t_train, axes=1)
        self.ratios[:] = np.tensordot(W, ratios_train, axes=1)
        basal_t[:] = np.tensordot(W, basal_t_train, axes=1)
        inter_t[:] = np.tensordot(W, inter_t_train, axes=1)
        self._add_perturbation_stimulus(basal_t, inter_t, times)

        ### Rescale kz (per sample with per-sample mixtures: rescale (S, G), kz (S, G, 2))
        _, _, k1 = self._mixture_terms()
        rescale = np.ones_like(k1)
        rescale[..., ns:] = k1[..., ns:]
        kz = ks * rescale[..., None]
        c_nb = self.a[..., -1, :]  # (G,) or (S, G): inverse mRNA burst size

        # Determine per-cell basal strategy once before the loop.
        # If basal_t is 3-D (per-sample) but no samples_data was provided, that means
        # the caller forgot to set model.samples_data → route everything to sample 0
        # and emit a warning rather than crashing with a cryptic shape error.
        if basal_t.ndim == 3 and samples_data is None:
            print("[simulate_trajectories] WARNING: basal_t is 3-D (per-sample) "
                  "but samples_data is None. "
                  "Did you forget to load data_samples.npy into model.samples_data? "
                  "Falling back to sample index 0 for all cells.")
            samples_data = np.zeros(N, dtype=int)
        _basal_is_per_sample = (samples_data is not None and basal_t.ndim == 3)
        if (ks.ndim == 3 or inter_t.ndim == 4) and samples_data is None:
            raise ValueError("Per-sample mixtures / networks need samples_data (data_samples.npy) to simulate")
        # Sample index of each simulated cell (as for the per-sample basal)
        s_cells = np.asarray(samples_data)[:N].astype(int) if samples_data is not None else np.zeros(N, dtype=int)
        # Per-cell mixture terms (shared or per sample)
        k1_cells = np.array([s1_of(rescale, s) for s in s_cells])     # (N, G)
        c_cells = np.array([s1_of(c_nb, s) for s in s_cells])         # (N, G)
        low_cells = np.array([ks_of(ks, s)[:, 0] for s in s_cells])   # (N, G) lowest normalised mode

        def gate_rows(rows):
            """kon of the rows gated by their chromatin: k_low + cbar (kon - k_low)."""
            if chrom0 is None:
                return
            cbar = np.array([chrom_gate(chrom_values(chrom_lvl[r], chrom0.c_levels), chrom0.w, chrom0.has_region)
                             for r in rows])
            kon_vector[rows, ns:] = low_cells[:, ns:] + cbar[:, ns:] * (kon_vector[rows, ns:] - low_cells[:, ns:])

        for dst, _, _, _ in starts:
            gate_rows(np.arange(dst, dst + N))

        # Proliferation: R(P) integrated along each simulated path (states recorded at the quadrature nodes)
        _prolif_fn = None
        if self.simulate_with_proliferation and self.prolif_network is not None:
            # (N, ..., n_proteins), stimuli of the interval -> (N, ...); rows routed by sample (per-condition MLPs)
            _prolif_fn = lambda path, st: self.prolif_network.predict(path, st, samples=s_cells)
        # RATE effects of perturbation stimuli (added to R); without the MLP, R = 0 + effects
        _rates = list(getattr(self, 'rate_perturbation', None) or [])
        if _rates and _prolif_fn is None and verb:
            print("[simulate] Warning: RATE effects without the proliferation MLP: base net rate 0")
        _branching = _prolif_fn is not None or bool(_rates)
        self.log_population = np.zeros(len(times)) if _branching else None
        # Dilution of the proteins at the birth rate of each simulated cell, as in the trajectories and the refit of
        # d1: its regression on the simulated state (birth head of the MLP, with or without branching), else the birth
        # (consistent with R_opt, saved by infer_network_simul) of the trajectory slot of the cell at the inference
        # interval containing the simulated one (slot followed through the resampling)
        from ..tools.estimate_proliferation import split_net_change
        _mlp_birth = (self.protein_dilution and self.prolif_network is not None
                      and hasattr(self.prolif_network, 'predict_birth'))
        b_traj = d_traj = slot = None
        if self.protein_dilution and not _mlp_birth:
            bp, dp = self._state_rates()  # those of infer_network_simul (load_state_rates), else the prior
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

            start_index = N * cnt
            end_index = N * (cnt+1)

            if _basal_is_per_sample:
                # 3-D basal_t: per-sample basal directly available
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
                    n_samp_k = basal_t.shape[1] if basal_t.ndim == 3 else (ks.shape[0] if ks.ndim == 3 else 1)
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
                    b0 = self.prolif_network.predict_birth(P_start, stim_cells, samples=s_cells)
                    d0 = self.prolif_network.predict_death(P_start, stim_cells, samples=s_cells)
                else:
                    # Inference interval containing time (last one beyond): mean of the rates of its start and end
                    # states, as the birth linear in time of the refit (same dilution integral)
                    k = int(np.clip(np.searchsorted(times_train, time, side='right') - 1, 0, len(times_train) - 1))
                    k_end = min(k + 1, len(times_train) - 1)
                    b0 = 0.5 * (b_traj[k, slot] + b_traj[k_end, slot])
                    d0 = 0.5 * (d_traj[k, slot] + d_traj[k_end, slot])
                delta = rate_effects(P_start[:, None, :])[:, 0]
                b_cells = split_net_change(b0, d0, delta)[0] if np.any(delta) else np.asarray(b0, dtype=float)

            mrna_pdmp = mrna_mode and bool(self.simulation_stochastic)
            stoch, scale_p = bool(self.simulation_stochastic), self.scale_proteins

            # The worker gets only its cell's rows (no self, no full arrays): joblib pickles the closure with every
            # batch of tasks, so capturing the model or the state arrays made the simulation several times slower
            def run_main_loop_for_cell(n, P0, M0, lvl0, _basal_cells=basal_cells, _basal_t_cnt=basal_t[cnt],
                                       _stim_cells=stim_cells, _b_cells=b_cells, _inter_cnt=inter_t[cnt]):
                _stim_vals = _stim_cells[n]
                basal_n = _basal_cells[n] if _basal_cells is not None else _basal_t_cnt
                s_n = s_cells[n]
                # Rates of the cell's kinetics (per network condition: rows of its sample)
                deg = degradations if degradations.ndim == 2 else degradations[:, min(s_n, degradations.shape[1] - 1)]
                # Dilution: protein decay d1 + b of the cell (PDMP: flow only, bursts unchanged; ODE: target x c)
                b_vec = np.zeros(deg.shape[1])
                if _b_cells is not None:
                    b_vec[ns:] = _b_cells[n]
                ch = None
                if chrom0 is not None:  # chromatin of the cell (levels updated in place by the simulation)
                    ch = copy.copy(chrom0)
                    ch.lvl = lvl0.copy()
                if mrna_pdmp:
                    sim = simulate_next_mrna_pdmp(
                        deg[0, :], deg[1, :],
                        ks_of(kz, s_n) * deg[0][:, None], c_cells[n],
                        basal_n, inter_of(_inter_cnt, s_n), t_rec, scale_p,
                        P0=P0, M0=M0, ns=ns, stim_vals=_stim_vals, chrom=ch, dilution=b_vec)
                elif stoch:
                    sim = simulate_next_prot_pdmp(
                            deg[1, :] + b_vec,
                            ks_of(kz, s_n) * deg[0][:, None],
                            s1_of(rescale, s_n) * (deg[0, :] / deg[1, :]),
                            basal_n, inter_of(_inter_cnt, s_n), t_rec,
                            scale_p, P0=P0, ns=ns, stim_vals=_stim_vals, chrom=ch,
                        )
                else:
                    sim = simulate_next_prot_ode(
                        deg[1, :] + b_vec, ks_of(ks, s_n),
                        basal_n, inter_of(_inter_cnt, s_n), t_rec,
                        scale_p, P0=P0, ns=ns, stim_vals=_stim_vals, chrom=ch,
                        cfac=deg[1, :] / np.maximum(deg[1, :] + b_vec, 1e-300)
                    )
                return sim, (None if ch is None else ch.lvl)

            outputs = Parallel(n_jobs=-1)(
                delayed(seeded_call)(task_seed(self.seed, 4, cnt, n), run_main_loop_for_cell, n,
                                     prot_modified[start_index + n, :],
                                     None if m_state is None else m_state[start_index + n, :],
                                     None if chrom_lvl is None else chrom_lvl[start_index + n])
                for n in range(0, N))
            results = [o[0] for o in outputs]

            for idx, n in enumerate(range(0, N)):
                # result.p[-1] has shape (G_tot-1,): indices 1..G_tot-1 of the state
                # (index 0 = stim1 is excluded by the simulation).
                # For ns>1, indices 1..ns-1 are stim2..stimN — skip them.
                prot_modified[end_index + n, ns:] = results[idx].p[-1][ns - 1:]
                if mrna_pdmp:
                    m_state[end_index + n, ns:] = results[idx].m[-1][ns - 1:]
                if chrom0 is not None:
                    chrom_lvl[end_index + n] = outputs[idx][1]
            if self.gene_mask is not None:
                # Genes outside each cell's condition (select_genes_per_condition): frozen
                off = ~self.cell_gene_mask(s_cells)
                off[:, :ns] = False
                prot_modified[end_index:end_index + N][off] = prot_modified[start_index:start_index + N][off]
                if mrna_pdmp:
                    m_state[end_index:end_index + N][off] = m_state[start_index:start_index + N][off]

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
                M_end = None if not mrna_pdmp else m_state[end_index:end_index + N, ns:].copy()
                L_end = None if chrom0 is None else chrom_lvl[end_index:end_index + N].copy()
                for grp in groups:
                    weights = np.exp(log_weights[grp] - log_weights[grp].max())
                    src = grp[np.random.choice(len(grp), len(grp), replace=True, p=weights / weights.sum())]
                    prot_modified[end_index + grp, ns:] = P_end[src]
                    if M_end is not None:
                        m_state[end_index + grp, ns:] = M_end[src]
                    if L_end is not None:
                        chrom_lvl[end_index + grp] = L_end[src]
                    if slot is not None:  # the trajectory slot (prior birth) follows the resampled cells
                        slot[grp] = slot[src]

            # Set stim values at step boundary from schedule (authoritative source).
            prot_modified[end_index:end_index + N, :ns] = stim_cells

            # Each cell with its own sample's basal and mixture (index-based, as basal_cells)
            n_samp = basal_t.shape[1] if basal_t.ndim == 3 else (ks.shape[0] if ks.ndim == 3 else 1)
            kon_vector[end_index:end_index+N, ns:] = self._kon_ref_per_sample(
                prot_modified[end_index:end_index+N, :], ks, inter_t[cnt], basal_t[cnt],
                samples_id=np.arange(n_samp), samples_data=s_cells)[:, ns:]
            gate_rows(np.arange(end_index, end_index + N))

            if mrna_mode:
                rows = slice(end_index, end_index + N)
                if mrna_pdmp:
                    m_counts[rows, ns:] = np.random.poisson(m_state[rows, ns:])
                else:
                    # ODE interval: mRNA at its mean given the proteins, counts from the NB of the modes
                    mean_kon = k1_cells[:, ns:] * kon_vector[rows, ns:]
                    m_state[rows, ns:] = mean_kon / c_cells[:, ns:]
                    m_counts[rows, ns:] = np.random.negative_binomial(
                        np.maximum(mean_kon, 1e-8), c_cells[:, ns:] / (c_cells[:, ns:] + 1.0))
                m_counts[rows, :ns] = stim_cells / self.scale_proteins * self.scale_mrnas

            if verb:
                print(f'timepoints {cnt} done', delta_t, time)

        self.simulation_stochastic = simulation_stochastic_orig
        chrom_values_rows = None
        if chrom0 is not None:
            chrom_values_rows = np.array([chrom_values(lv, chrom0.c_levels) for lv in chrom_lvl])
        return prot_modified, m_counts, kon_vector, chrom_values_rows, times_simulation


    def simulate_network(self, times, verb=True, stimulus_schedule=None, schedule_times=None):
        """
        Simulate the protein trajectories using the final inferred network.

        schedule_times : times of the rows of stimulus_schedule (step function, a value holding from its
            time on); None = the inference times (rows of the inference schedule).
        """
        seed_everything(self.seed)

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
                                and ((self.basal is not None and self.basal.ndim == 2) or self.a.ndim == 3))
                            else None)

        y_prot, y_mrna, kon_vector, y_chrom, times_simul = self.simulate_trajectories(
            times, times_train, ks, N=N, verb=verb, samples_data=samples_data_sim)
        self.mrna_simul = y_mrna    # explicit mRNA counts (simulation_mrnas_direct), else None
        self.chrom_simul = y_chrom  # chromatin accessibility (N * T, R) with self.chromatin, else None

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
        set_depth_model(self.depth_model)
        times = np.sort(np.unique(vect_t))
        resp0, _ = predict_resp(x, ks, c, pi_zero=pi_zero, zi=zi, s=depth)
        nu0 = self._mode_masses(resp0, x, vect_t, times, ks, c, depth)
        pi_glob = np.sum([nu0[t] * np.mean(vect_t == t) for t in times], axis=0)
        resp, _ = predict_resp(x, ks, c, pi=pi_glob, pi_zero=pi_zero, zi=zi, forcing=self.mean_forcing_em, s=depth)
        return resp, self._mode_masses(resp, x, vect_t, times, ks, c, depth)

    def fit_mixture_test(self, data_rna, ks, c, verb=False, depth=None):
        """Classify test cells into mixture modes using fixed kinetic parameters (ks, c, pi_zinb of the training).

        Sets self.modes, self.basins, self.proba_init, and self.pi_init (mode masses per time, computed on the test
        cells as on the training ones: mean-preserving with mean_forcing_em, never the training masses).
        """
        ns = self.n_stimuli
        N_cells, G_tot = data_rna.shape
        vect_t = data_rna[:, 0]
        times = np.sort(np.unique(vect_t))

        training_pi_zinb = self.pi_zinb        # array(G_tot - ns,) of pi_zero, or None

        frequency_modes_smooth = np.ones_like(data_rna, dtype=float)
        M = self.a.shape[-2] - 1  # number of mixture modes
        frequency_basins = np.zeros((N_cells, G_tot), dtype=np.int8)
        frequency_proba_init = np.zeros((N_cells, G_tot, M), dtype=float)
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
            # Initial basins: argmax of the mixture posteriors, as for the training set
            basins_g = np.argmax(proba, axis=1)
            frequency_basins[:, g] = basins_g
            frequency_proba_init[:, g, :ng] = proba
            frequency_proba_init[:, g, ng:] = 0
            frequency_modes_smooth[:, g] = ks[:ng, g][basins_g]
            if verb:
                print('[infer_test]', f'Gene {g} calibrated...', ks[:ng, g], c[g])

            pi_init_test.append(pi_g)

        scale_max = np.max(self.a[:-1, :], axis=0)
        frequency_modes_smooth /= np.maximum(scale_max, EPS)  # all-zero modes would give 0/0
        self.pi_zinb = training_pi_zinb   # keep training ZINB zero-inflation values
        self.modes = frequency_modes_smooth
        self.basins = frequency_basins
        self.proba_init = frequency_proba_init
        self.pi_init = pi_init_test



    def _fit_mixture_test_per_sample(self, data_rna, vect_samples_id, samples_id, depth=None):
        """fit_mixture_test with per-sample mixtures (a 3-D): each sample's cells with its own parameters."""
        a3, pz3 = self.a, self.pi_zinb
        N, G_tot = data_rna.shape
        M = a3.shape[1] - 1
        modes, basins, proba_init = np.zeros((N, G_tot)), np.zeros((N, G_tot), dtype=np.int8), np.zeros((N, G_tot, M))
        vect_t = data_rna[:, 0]
        times = np.sort(np.unique(vect_t))
        pi_sum = None
        for s_idx, sid in enumerate(samples_id):
            m = vect_samples_id == sid
            self.a, self.pi_zinb = a3[min(s_idx, len(a3) - 1)], pz3[min(s_idx, len(pz3) - 1)]
            self.fit_mixture_test(data_rna[m], self.a[:-1], self.a[-1], depth=None if depth is None else depth[m])
            modes[m], basins[m], proba_init[m] = self.modes, self.basins, self.proba_init
            act = None if self.gene_mask is None else self.active_genes(s_idx)
            if act is not None:  # genes outside the sample's condition: single mode
                modes[np.ix_(m, ~act)], basins[np.ix_(m, ~act)] = 1.0, 0
                proba_init[np.ix_(m, ~act)] = np.eye(M)[0]
            # Mode proportions per time, weighted by the cells of each sample
            w = {t: np.sum(vect_t[m] == t) for t in times}
            if pi_sum is None:
                pi_sum = [{t: np.zeros(M) for t in times} for _ in self.pi_init]
            for g, pi_g in enumerate(self.pi_init):
                if act is not None and not act[self.n_stimuli + g]:
                    continue  # gene outside the sample's condition
                for t in times:
                    if t in pi_g:
                        p = np.asarray(pi_g[t], dtype=float)
                        pi_sum[g][t][:len(p)] += w[t] * p
        self.a, self.pi_zinb = a3, pz3
        self.modes, self.basins, self.proba_init = modes, basins, proba_init
        self.pi_init = [{t: (p if p.sum() > 0 else np.eye(M)[0]) / (p.sum() + EPS) for t, p in pi_g.items()}
                        for pi_g in pi_sum]

    def _assign_calibrated_basins(self, train, vect_t, times, ks, ks_max, ks_cells, G_tot, weight_prob, act_cells=None):
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
            tmp_basins = np.zeros(n_cells, dtype=np.int8)
            tmp_modes = np.zeros(n_cells)
            # Cells whose condition selected g (select_genes_per_condition); the others keep their single mode
            act = np.ones(n_cells, bool) if act_cells is None else act_cells[:, g]
            tmp_modes[~act] = obj[~act, 0]
            for t_i in (times if self.temporal_basins else [None]):
                idx = np.flatnonzero(act & (np.ones(n_cells, bool) if t_i is None else vect_t == t_i))
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
                    nu = np.sum([self.pi_init[g - ns][t] * np.sum((vect_t == t) & act) / act.sum() for t in times], axis=0)[:l_max] \
                        * self.force_basins + post.sum(axis=0) * (1 - self.force_basins)
                nu /= np.sum(nu)
                dist = np.clip(-(np.log(proba) + (1 - weight_prob) * np.log(predicted(lr, post))), 0, 100)
                mu = np.ones(len(idx)) / len(idx)
                k = np.argmax(ot.emd(mu, nu, dist, numItermax=int(1e7)), axis=1)
                tmp_basins[idx] = k
                tmp_modes[idx] = obj[idx, k]
            return tmp_basins, tmp_modes

        results = Parallel(n_jobs=-1)(delayed(run_gene)(g) for g in range(ns, G_tot))
        for g, (tmp_basins, tmp_modes) in zip(range(ns, G_tot), results):
            self.basins[:, g], self.modes[:, g] = tmp_basins, tmp_modes

    def infer_test(self, data, vect_samples_id=None, verb=True, stimulus_schedule=None,
                   basal_ref=None, transition_rates=None, time_key='time', context=None, train=None):
        """
        Test-set inference with the network fixed, in a single pass. train: dict of the training arrays (see
        _alpha_pool and _assign_calibrated_basins). The basins come from one EMD of the mixture probabilities,
        calibrated on how the training assigned them (network included); the trajectories are computed once with
        the final regularization of the training (context), the alphas being copied from the nearest training
        states.

        basal_ref : (n_samples, G_tot) array or None
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
        self.gene_mask = None
        if hasattr(data, 'uns'):  # genes of each test sample's condition (select_genes_per_condition)
            self.set_gene_mask(data, [str(s) for s in samples_id], verb=verb)

        if self.a.ndim == 3:
            self._fit_mixture_test_per_sample(data_rna, vect_samples_id, samples_id, depth=self._depth_factors(data))
        else:
            self.fit_mixture_test(data_rna, self.a[:-1], self.a[-1], depth=self._depth_factors(data))
        # Trajectories on counts at the reference depth (the basins above used the raw counts and s)
        if self._depth_factors(data) is not None:
            data_rna = self._parse_input(data, time_key, scale_depth=True)

        print('[infer_test] Mean proba = ', np.mean(np.max(self.proba_init[:, ns:, :], axis=-1)))

        ks, s1, _ = self._mixture_terms()
        # Mode amplitudes of each test cell's sample (per-sample mixtures)
        ks_cells = ks[np.searchsorted(samples_id, vect_samples_id)] if ks.ndim == 3 else None
        ks_max = ks.max(axis=0) if ks.ndim == 3 else ks

        # --- Build per-cell KO/OV mask and apply initial mode forcing ---
        kov_cell_mask = None
        if basal_ref is not None:
            br = np.asarray(basal_ref, dtype=float)
            if br.ndim == 2 and np.any(np.abs(br) > 50):
                cm = np.zeros((len(vect_t), G_tot), dtype=np.int8)
                for s_idx, s in enumerate(samples_id):
                    cell_idx = np.where(vect_samples_id == s)[0]
                    ko_genes = np.where(br[s_idx] < -50)[0]
                    ov_genes = np.where(br[s_idx] >  50)[0]
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
                    self.basins[ko_cells, g] = 0
                    self.modes[ko_cells, g] = ks_cells[ko_cells, g, 0] if ks_cells is not None else ks[g, 0]
                if np.any(ov_cells):
                    self.basins[ov_cells, g] = l_max - 1
                    self.modes[ov_cells, g] = ks_cells[ov_cells, g, l_max - 1] if ks_cells is not None else ks[g, l_max - 1]

        if context is None:
            context = {'n_iter_reg': self.min_n_loops, 'weight_init': 0.0,
                       'weight_prob': max(.96**(self.min_n_loops - 1), .1), 'reg_factor': 1.0}
        self._assign_calibrated_basins(train, vect_t, times, ks, ks_max, ks_cells, G_tot, context['weight_prob'],
                                       act_cells=self.cell_gene_mask(np.searchsorted(np.asarray(samples_id),
                                                                                     vect_samples_id)))
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


    def fit(self, data_rna, max_iter_kinetics=100, verb=True):

        self.fit_mixture(data_rna, min_components=2, max_components=2, max_iter_kinetics=max_iter_kinetics)
        self.fit_network(data_rna, verb=verb)
        # self.refine_network_degradations()