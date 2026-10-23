
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
import warnings
from functools import partial
import numpy as np
import ot
import seaborn as sns
import multiprocessing as mp
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.linalg import expm
from ..config import resolve_cell_type_obs, CELL_TYPE_OBS_KEYS
from ..inference.trajectory import ks_of, s1_of, s1_rows, inter_of, birth_alpha, prot_bridge, prot_bridge_end
from ..inference.mixture import _compute_nu_with_temporal_constraint, fit_grid_mixture, grid_weights_fixed, mixture_criterion
from ..inference.network import cross_entropy, dist_term, _dist_loss_grad
from ..inference.trajectory import kon_switch_fraction
from ..inference.sigma import (SPEC_PHI, SPEC_LINK, PHI_IDENTITY, LINK_LOGISTIC, PHI_NAMES, LINK_NAMES, phi_code, link_code,
                               default_spec)
from ..inference.simulations import simulate_next_mrna_pdmp, ChromatinModel, chrom_gate, chrom_values, Simulation
from ..inference import (inference_network_multi, active_regulators, PathProt, signed_floor, sigmoid_amplitudes,
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
RENAMED_PARAMETERS = {'max_iter_kinetics': 'max_iter_softEM', 'cell_depth_for_embeddings': 'cell_depth_for_representation',
                      'simulate_full_with_harissa': 'simulation_mrnas_direct',
                      'condition_kinetics': 'degradation_rates_per_condition',
                      'grid_levels': 'max_grid_levels'}

REMOVED_PARAMETERS = {
    'continuous_targets': 'the mixture is always the continuous model of the counts with 2..max_grid_levels levels (max_grid_levels = 2: two modes)',
    'nb_likelihood': 'the mixture is always the continuous model of the counts (max_grid_levels)',
    'min_mode_weight': 'the number of modes is chosen by mixture_selection_criterion (min_mixture_components = 1 lets a gene keep one mode)',
    'min_mode_ratio': 'the number of modes is chosen by mixture_selection_criterion (min_mixture_components = 1 lets a gene keep one mode)',
    'min_mode_weight_integration': 'the number of modes is chosen by mixture_selection_criterion',
    'min_mode_ratio_integration': 'the number of modes is chosen by mixture_selection_criterion',
    'soft_em_refinement': 'the soft EM with masses per time (max_iter_softEM) removes the truncation bias of the CEM',
    'late_protein_for_autoregulation': 'removed (the mRNA memory, targets_accounting_late_mrna, lags every regulator)',
    'late_protein_lag': 'removed: the lag of the own protein alone biased the self-regulation of the simulations; the mRNA memory (targets_accounting_late_mrna) lags every regulator',
    'inference_stochastic': 'the network fits never average kon over the protein noise (modes, or inference_mrnas_direct)',
    'model_learning': 'renamed model_learning_grn (model_learning_chrom for the chromatin opening function)',
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
    'use_scBoolSeq': 'the NB mixture is always fitted by EM',
    'bridge_sampling': 'the end proteins of the mRNA bridges are always their conditional means',
    'seuil_zero_min_ref': 'single threshold seuil_min_network_intensity (absent references, min |theta| of the kept signs)',
    'weight_prev': 'renamed weight_flow_matching',
    'exact_flow_matching': 'single flow matching: flow_matching_nodes nodes drawn uniformly along each interval',
    'scale_pen': 'sparsity of the network fits set by qut_level (quantile universal threshold), scale of the effects by network_ridge',
    'pi_zinb': 'the mixture is a pure NB (no zero inflation)',
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


def load_state_growth(cardamom_dir, use_recomputed=True):
    """Net growth rate of each trajectory state for the steps after infer_network_structure: R_opt of the growth
    pass (data_R_opt.npy) with use_recomputed_proliferation, else the prior rates of the real cells
    (data_R_prior.npy); None if absent."""
    path = os.path.join(cardamom_dir, 'data_R_opt.npy' if use_recomputed else 'data_R_prior.npy')
    return np.load(path) if os.path.exists(path) else None


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


def match_distance(prot, kon, prot_ref, kon_ref):
    """Distance between states (proteins and kon) and reference states: half the L1 distance of the proteins and
    half that of the kon, each brought to its median over the pairs (units of the proteins and of kon not mixed)."""
    dp = cdist(prot, prot_ref, 'cityblock')
    dk = cdist(kon, kon_ref, 'cityblock')
    return 0.5 * dp / max(np.median(dp), 1e-12) + 0.5 * dk / max(np.median(dk), 1e-12)


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



def _sinkhorn_log(mu, nu, C, reg, reg_m=None, tol=1e-7, n_iter=100000):
    """Sinkhorn in the log domain, same formulation as POT: balanced (kernel exp(-C / reg)), or unbalanced with KL
    marginals of weights reg_m = (source, target) and entropy relative to mu nu^T; stops when the log-potentials
    change by less than tol. Fallback of _solve_ot (slower, never underflows)."""
    la, lb = np.log(np.maximum(mu, 1e-300)), np.log(np.maximum(nu, 1e-300))
    lK = -C / reg
    f1, f2 = 1.0, 1.0
    if reg_m is not None:
        lK = lK + la[:, None] + lb[None, :]
        f1, f2 = reg_m[0] / (reg_m[0] + reg), reg_m[1] / (reg_m[1] + reg)
    f, g = np.zeros(len(mu)), np.zeros(len(nu))
    for _ in range(n_iter):
        f_new = f1 * (la - logsumexp(lK + g[None, :], axis=1))
        g_new = f2 * (lb - logsumexp(lK + f_new[:, None], axis=0))
        done = max(np.max(np.abs(f_new - f)), np.max(np.abs(g_new - g))) < tol
        f, g = f_new, g_new
        if done:
            break
    return np.exp(f[:, None] + lK + g[None, :])

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
        self.modes_soft = None  # one_hot_probas = False: expected mode of each cell and gene under the posteriors updated by the network (targets of the network fit and of the network term of the OT)
        self._target_mix = 0.0  # one_hot_probas = False: weight of the EMD basins in the network targets (1 at the first iteration of fit_network, 0 from min_n_loops)
        self.posteriors_soft = None # one_hot_probas = False: posteriors updated by the network (N_cells, G_tot, M), targets of the free-running fit (levels drawn from them)

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
        self.select_genes = False                # select_genes (on the train cells of split_dataset): gene selection (queries, entropy genes, global network, Steiner selection); False = all genes kept
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
        self.hard_em = True # Do we initialize with a hard_em ?
        self.max_iter_softEM = 100 # NB mixture: iterations of the soft EM (masses per time) that refines the CEM (removes its truncation bias: rate c overestimated, low mode too low); 0 / False: CEM only
        self.preserve_mean_values = 1 # Do we ensure temporal constraints when fitting the basins in the hard_em ?
        self.mean_forcing_em = 1.0 # at which point we force the mean correction: the higher the more
        self.force_basins = 1.0 # Do we want to ensure the means to be preserved by the NB mixture ? It may not preserve multistability
        self.temporal_basins = 1 # Is it preserved temporally ?
        self.seuil = 1e-2 # minimum for beta mixture parameters (second parameters)
        self.batch_size_mixture = 4096 # Maximum number of cells per time used for mixture calibration in the inference.

        # Observation model of the network terms
        self.min_mixture_components = 1 # NB mixture: minimal number of modes (1: a gene without separated modes keeps a single one)
        self.criterion_weights_per_time = False # mixture_selection_criterion: False = the level weights counted once (permissive: the weights of the times are pulled together by their Dirichlet prior), True = once per time
        self.mixture_selection_criterion = 'bic' # number of levels of each gene (max_grid_levels): 'icl' = BIC + 2 x entropy of the posteriors (penalises overlapping levels: a gene that switches keeps two modes), 'bic' = -2 loglik + p log n
        self.max_grid_levels = 4 # the mixture of each gene is the continuous model of the counts, NB of the burst frequency k_low + (k_high - k_low) sigma at the depth of the cell, sigma on K uniform levels of [0, 1] with weights per time (fit_grid_mixture, from the extremes of the CEM + soft EM); its levels are the modes of the whole inference (basins, forcing, targets of the network); K = 2..max_grid_levels chosen per gene by mixture_selection_criterion (2: two modes).
        self.grid_network_precision_max = 100 # genes of more than two levels: upper bound of the precision kappa of the network prior on the levels in the basin updates, exp(-kappa KL(tau_l || sigma)) of width ~ sigma (1 - sigma) / kappa, kappa estimated per gene at each update by moment matching (mean sigma (1 - sigma) / mean (expected level - sigma)^2, at least 1 = the Bernoulli prior of two modes)

        ## Basins of the cells (EM of fit_network)
        self.one_hot_probas = False # basin of each cell (forcing of the trajectories) = mass-constrained assignment (EMD) of the mixture posteriors under the mode masses, then of the costs with the network term (ramped, weighted per gene) in fit_network, in both cases; True: network fit on these basins (classification EM); False: network fit on the expected modes under the posteriors updated by the network, entropic OT of regularization 1 under the same masses with the network term at full weight (E-step: posterior x network prior x mass factors), the targets going from the EMD basins at the first iteration to these expected modes at min_n_loops
        self.weight_network_distribution = 0.1 # one_hot_probas = False, targets of more than two levels: weight of the distribution term of the network fit and of the stopping error, sum over (sample, time) of sum_i (sigma_(i) - q_i)^2 (sorted sigmas against the quantiles of the levels under the mean posterior of the cells: 1-D Wasserstein-2), so that the network keeps the spread of the targets and not only their mean per cell; ~2 x the weight of the cross-entropy near sigma = 0.5; 0: not computed

        self.targets_accounting_late_mrna = False # True: the mRNA of a cell reflects its burst frequency over ~1 / d0 before the observation (d0 >= min_ratio x d1): every comparison of sigma with the mode of a cell (network fit, OT, basin updates, stopping error, score of the model learning, free-running fit, kon_theta, simulated counts) takes sigma along the memory of its mRNA (late_mrna_nodes); the forcing switches (alpha) when the network commits, delayed by the equivalent step of the mRNA when the forcing is the mRNA level (adaptive shrinkage, no mrna_bridge)
        self.late_mrna_nodes = 4 # with targets_accounting_late_mrna, for the network fit, the stopping error, the basin updates and the score of the model learning: 1 = single lagged node per class of targets (late_mrna_classes): sigma at the proteins (1 - exp(-d0 dt)) / d0 before the state, the mean of the memory kernel truncated at the start of the interval; Q >= 2 = sigma averaged on Q nodes over the last min(interval, 5 / d0) hours with the weights d0 exp(-d0 (t - s)) of each target (mass before the window on the earliest node), Q + 1 evaluations of sigma. The OT, the free-running fit, kon_theta and the simulated counts, which evaluate sigma many times, always use the single lagged nodes
        self.late_mrna_classes = 3 # with targets_accounting_late_mrna: classes of the targets by d0 (equal sizes), each with its single lagged node (the mean lag of its genes): sigma of each target at the lag of its class, cost x classes of the single node
        self.weight_flow_matching = 0 # weight of the flow-matching term of the network fits (end-state term: 1 - weight), ramped from 0 at the first iteration to this value at min_n_loops
        self.flow_matching_nodes = 1 # flow matching: nodes in each interval around the switch a of the state (mean of the alphas of the genes that change mode): just after it (a + 0.1, at a when a >= 1/2), then equally spaced on [0, a] and [a, 1] alternately; equally weighted, the regulators at their protein states there and the target at its kon in force (mode of the start state before its switch alpha, of the end state after it)
                   
        # Per-cell sequencing depth
        self.depth_model = 'mean' # with depth factors: 'burst_frequency' = the depth multiplies the burst frequency, NB(k s, c) (Fano factor independent of the depth); 'mean' = it multiplies only the mean, NB(k, c / s) (dispersion k fixed, as scVI)
        self.use_depth_factor = True   # use obs['depth_factor'] (estimate_cell_depth) in the NB model and simulations; False = s_i = 1 (an existing factor is kept but ignored)
        self.compute_depth_factor = True # estimate_cell_depth computes the diagnostic and writes obs['depth_factor'] if needed; False = only reads an existing factor (never removed)
        self.allow_depth_correction = True  # estimate_cell_depth: apply the per-cell depth factor s_i when the diagnostic recommends it
        self.depth_method = 'poissonian'  # depth factor: 'group_median', 'poissonian' (Fang & Pachter, 2025), or <project>/depth_methods/<name>.py
        self.depth_method_params = {'reference': 'sample_pairwise'}       # parameters of the depth method
        self.depth_by_cell_type = False     # estimate s_i within (sample, time, cell type) groups; False = (sample, time) only:
                                            # cell types derive from expression, normalising within them pulls cells to their type (circular)
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
        self.unbalanced_reg = 10 # UOT relaxation of the target marginal if > 0: exponent f = reg_m / (reg_m + reg) of its value for reg_m = unbalanced_reg x typical excess cost and the entropic regularization of min_n_loops, fixed at every iteration; OT if 0. Higher = more constrained marginals (uncertainty of the proliferation rates)
        self.init_entropic_noise = 1.0 # Initial entropic regularization of the OT, relative to the typical excess cost of a pair (decreasing with the iteration)
        self.intensity_prior = 100 # mRNA term of the OT cost while n_iter <= intensity_prior (decreasing weight); < 1: random couplings at the first iteration, then network term only
        self.quant_samples = .95 # Quantile of cells number per sample to use for inference
        # Protein trajectories: forcing by the modes or the observed mRNA, scales
        self.inference_mrnas_direct = True # True: fit_network trajectories driven by the observed mRNA (shrunk modes, adaptive_shrinkage); False: by the modes only
        self.mrna_bridge = False # True: proteins from the linear mRNA bridges between observed mRNAs (prot_bridge_end, d0 >= min_ratio d1); False: shrunk modes (adaptive_shrinkage)
        self.scale_proteins = 1 # Eventually rescale protein values (recommended:1-2)
        self.scale_mrnas = 100 # Eventually rescale mRNA values (recommended:100)
        self.fact_simple = 2 # slight transformation for constrative modes in learning phase
        # Inference of alpha = switch moment between each timepoint and modes
        self.alpha_threshold = .6 # max = 1, thershold for important transition to update alpha full
        self.n_pas = 25 # number of timesteps between timepoints for inference of alpha
        self.force_n_pas = False # if True, use n_pas as-is instead of auto-flooring it to the inter-timepoint interval length
        # Network inference with scipy
        self.qut_level = 0.5 # network fits: sparsity level = quantile universal threshold, level-quantile of the largest gradient at the null model under permuted targets (probability that no null edge enters a target without regulator); first fit, and refinement (garrote: its own threshold, theta of the first fit fixed)
        self.qut_n_perm = 20 # network fits: permutations of the target per fit for the quantile universal threshold
        self.network_ridge = 20 # network fits: Gaussian prior on the effect of each edge on the logit (w theta sd of the regulator), sd 2 logit(1 - 1/network_ridge); bounds the edges of nearly separable targets
        self.batch_size_network = 100 # Cells per network sub-sample (stratified by time and sample); raised to 10 x the parameters of a target-gene fit if lower; None = that floor
        self.n_network_fits = 10 # Theta = mean of min(n_network_fits, 1 + n_states // batch_size) fits on disjoint subsamples, at every network update
        # Penalization/prior information
        self.stimulus = 1.0 # 1 if we simulate with a stimulus. If not we can penalize the stimulus with a value between 1 and 0: 0 = no sitmulus
        self.prior_network_pen = 1.0 # 1 if we don't use prior information. If not we can penalize the non-existing age in prior network with values between 1 and 0: 0 = impossible edge
        self.constrain_basal_uniform = 1.0 # >= 0 penalty strength that pushes per-sample basals to be equal (ignores samples pinned by KO/OV basal_ref)
        self.network_condition_pen = 1.0 # >= 0, with >= 2 values of obs['network_condition']: fused L1 penalty on the deviations of each condition's network from the shared one minus those of inter_ref from its mean, relative to the sparsity penalty (large = reference deviations, i.e. a common network without per-condition inter_ref; 0 = independent networks)
        self.hard_forcing_ref = False # if True, constrain all network params to ±ref_constraint_pct around inter_ref
        self.ref_constraint_pct = 0.01 # fractional tolerance around inter_ref values for bounds (used when hard_forcing_ref=True)
        # Filtering
        self.filter_network = 1 # Do we filter the network ? It also builds a temporal network using the filter criterium
        self.seuil_min_network_intensity = 1e-2 # |effective theta| below = absent edge: post-inference filter, support of the free-running fit, references (inter_ref) below are ignored; min |theta| of the kept signs (inter_ref above it, ref_network < -1 or > 1)
        self.seuil_min_network_variations = 0 # post-inference filter: min kon variation when removing an edge (0 = intensity threshold only)

        ## Model learning of the regulation functions (inference/sigma.py)
        # GRN: regulation function sigma of the burst frequency
        self.model_learning_grn = False # True (forces one_hot_probas = False): the family of the GRN regulation function sigma of each target gene is selected in fit_network among model_learning_link x model_learning_phi, and sigma_spec_grn (saved as cardamomOT/sigma_spec_grn.npy) is used by every later step; False: logistic of a linear input for every gene (default model, sigma_spec_grn ignored)
        self.model_learning_link = ['logistic'] # with model_learning_grn: candidate links of sigma (logistic only for now)
        self.model_learning_phi = ['identity', 'logarithm'] # with model_learning_grn: candidate transformations of the regulators, phi(P) = P (sigmoid of a linear input) or log(1 + P / kappa) (Hill type); a single link and phi: that family is forced for every gene and the whole pipeline (no selection, no extra cost), e.g. ['logarithm'] for Hill functions
        self.model_learning_kappa = 0.1 # with model_learning_grn: kappa of every regulator (phi = log) relative to the protein level Pbar of its highest mode (scale_proteins); phi is normalised so that phi(Pbar) = Pbar
        self.model_learning_start = None # with model_learning_grn: first iteration of the selection (None = min_n_loops), before it every gene is logistic
        self.model_learning_freeze = None # with model_learning_grn: families fixed from this iteration on (None = start + 5)
        self.model_learning_margin = 0.01 # with model_learning_grn: a target changes family when the held-out score of another one is lower by this relative margin at two consecutive selections
        self.model_learning_score = 'ce' # with model_learning_grn: held-out score of the families, 'ce' = cross-entropy against the targets of the mixture alone (E[sigma | counts] with the grid), 'nb' = negative log-likelihood of the counts under the continuous model (exact only if the counts follow its NB: in the oracle with explicit mRNA, chance level against 0.65 for 'ce')
        self.sigma_spec_grn = None # with model_learning_grn: (G_tot, N_SPEC) phi and link of each gene as a target, scale kappa as a regulator (inference/sigma.py), set by the selection; None = all logistic
        # Chromatin: opening function of the regions (independent of the GRN sigma)
        self.model_learning_chrom = False # True: the chromatin opening function sigma_c of each region follows sigma_spec_chrom (sigma.py, independent of the GRN sigma); False: logistic (default)
        self.sigma_spec_chrom = None # with model_learning_chrom: (R, N_SPEC_CHROM) family of each region (sigma.py); None = all logistic

        ## Refine: kinetic rates (and network) by the free-running fit of the simulation model (refine_network_degradations, inference/sde.py)
        self.traj_cell_types = None # Cell type of the real cell behind each trajectory state (set by fit_network)
        self.use_temporal_degradations = 1 # If so, compute temporal degradation rates for simulations ?
        self.degradation_rates_per_condition = True # with network conditions: network, d1, d0 / d1 and the scale of theta refined separately per condition (its samples and genes, its network as rappel; stored per sample); False: one shared fit from the mean network (forced True with select_genes_per_condition)
        self.fit_scale_theta = True # refine: scale of theta per target gene (and interval with use_temporal_degradations) learned in the free-running fit (within [1/5, 5]); False: scale 1
        self.refine_theta_with_sde = True # refine: True = network of fit_network refined (support and signs kept) with d1, d0 / d1 and the scale of theta by the free-running fit of the simulation model (ODE / SDE) on the observed modes; False = network refitted (final = 1) on the trajectories on the modes, then fixed in that fit
        self.min_ratio = 2.0 # lower bound of d0 / d1: bursting regime, imposed on the literature rates at the start of fit_network / infer_test (with a warning), then kept by the refine
        self.max_ratio = 50.0 # upper bound of d0 / d1 in the refine and the simulations: beyond, the simulations are nearly deterministic and only slower
        # Free-running fit (ODE / SDE)
        self.sde_epochs = 30 # free-running fit: optimizer steps per horizon of the curriculum (windows of 1 to T-1 intervals)
        self.sde_lr = 1e-2 # free-running fit: Adam learning rate
        self.sde_lambda_theta = 0.1 # free-running fit: rappel of theta / basal towards the starting network (relative L2)
        self.sde_lambda_rates = 1.0 # free-running fit: rappel of log d1, log(d1 / d0) and log scale towards their starting values (literature)
        self.sde_n_sub = 20 # free-running fit: integration steps per interval
        self.sde_max_horizon = 9 # free-running fit: longest window of the curriculum (intervals; 9 = chunks of 10 timepoints, starting at every timepoint)
        self.sde_windows = 10 # free-running fit: window starts drawn at each step (None = all; bounds the cost with many timepoints)
        self.sde_batch_size = 256 # free-running fit: trajectories per sample and window at each step (and observed cells per time)
        self.sde_memory_filter = False # free-running fit: observed modes compared with the kon read on the simulated mRNA, Kb' = d0 (kon - Kb) at the learned d0 = d1 / eps, carried along the windows from the kon of the start states (exact memory kernel), instead of the lagged nodes of targets_accounting_late_mrna; always on with simulation_mrnas_direct without targets_accounting_late_mrna
        self.ode_ce_weight = 0.1 # free-running fit of the ODE (simulation_stochastic = False): weight of the cross-entropy of each trajectory against the mode of its state (pairing of the OT), added to the Sinkhorn between distributions (which alone sets the bimodality of the deterministic cells); 0 = Sinkhorn only
        # Within the inference loop (refine_in_loop)
        self.refine_in_loop = False # True: in fit_network, between the network fit and alpha, refine_in_loop_epochs steps of the free-running fit of refine_network_degradations on its longest windows (model of the simulations: ODE, protein-only PDMP with simulation_stochastic, explicit mRNA with simulation_mrnas_direct): constant d1 (d1_eff) and d0 / d1 (rappel sde_lambda_rates towards the literature), no scale, the network too with refine_theta_with_sde; the rates (converted to the full d0, d1) then drive the trajectories
        self.refine_in_loop_start = 2 # with refine_in_loop: first iteration of the updates
        self.refine_in_loop_period = 1 # with refine_in_loop: iterations between two updates (the last cycle is completed before stopping)
        self.refine_in_loop_epochs = 10 # with refine_in_loop: optimizer steps at each update (windows of sde_max_horizon only)
        self.refine_in_loop_decay = 0.95 # with refine_in_loop: rappel towards the literature sde_lambda_rates x decay^k at the k-th update (k = 0 first), so that the network learned along the iterations can correct wrong literature rates; 1 = constant rappel

        ## Simulations
        self.simulation_stochastic = True # 1 if we simulate Bursty-like proteins, 0 if deterministic limit for proteins
        self.simulation_mrnas_direct = True  # True: the PDMP simulations carry explicit mRNA (mRNA bursts, proteins translated from them) and return mRNA counts; refine then fits d1 on mRNA-driven trajectories and d0 from d1_eff (protein-only fit on the modes). The ODE (simulation_stochastic = 0) stays on the proteins
        self.chromatin = None  # Chromatin regions of the simulations: dict(w (G, R), basal_c (R,), inter_c (G, R), kbar_c (R,), optional k_close_const, c_levels, diploid, c0), see ChromatinModel; None = no region

        ## Gene selection (select_genes.py, select_genes = True): terminals + global network + prize-collecting Steiner selection
        self.num_max_genes = 100         # budget: number of selected genes (stimuli excluded)
        self.n_query_genes = 10          # at most this many genes of genes_queries (gene_lists sheet) (round robin over time/cell-type DE groups)
        self.n_driver_genes = 10         # fate drivers of run_classical_OT (classical_OT/fate_drivers.csv) required in the selection (0 = none); n_query + n_driver + n_entropy < num_max_genes
        self.entropy_preselection = True  # queries, fate drivers and network genes (Steiner paths, growth) kept only among the entropy candidates (n_top_entropy) and the perturbed genes
        self.n_entropy_genes = 30        # at least this many entropy genes (Gandrillon KD & MDE), same round robin; n_query + n_entropy < num_max_genes
        self.n_top_entropy = 200         # top genes per transition for KD and for MDE (Gandrillon's TOP_N)
        self.n_cells_entropy = 1000      # cells per timepoint for the BUB entropy (its matrices are (N+1)^2)
        self.n_hvg_selection = 5000      # highly variable genes forming the network universe (terminals always kept)
        self.network_method = 'otvelo_granger'  # global network: 'otvelo_granger', 'otvelo_corr', 'wot_granger' (through the run_classical_OT couplings), or <project>/network_methods/<name>.py
        self.network_method_params = {}  # parameters of the network method (otvelo: n_cells, n_pcs, eps, alpha; granger: + k_candidates, en_alpha ('auto' = min(1, sqrt(50 / G))), l1_ratio, scale, stim_weight)
        self.k_in_steiner = 20           # strongest incoming edges kept per gene in the Steiner graph
        self.k_stim_steiner = 20         # direct targets kept per stimulus in the Steiner graph
        self.min_edge_prob = 0.05        # edges with probability (1 - FDR against the permuted-data network) below are dropped
        self.edge_prior = 0.9            # each Steiner edge is real with probability prob * edge_prior (paths and anchoring): favours short paths among equally probable ones
        self.null_network = 'hybrid'     # edge probabilities vs permuted data: 'hybrid' (gene edges within (sample, time), stimulus all cells), 'within_time', 'all_cells'
        self.select_genes_per_condition = False  # with >= 2 network conditions: genes selected separately per condition (data = union, uns['condition_genes']); each condition fitted on its own genes, fused penalty on the shared edges only
        self.sample_network_combination = 'auto'  # several samples, gene selection: 'consensus' (edge probabilities averaged, shared network), 'any' (probabilistic OR + closure balanced over the samples, independent condition networks); 'auto' = 'any' with >= 2 network conditions and network_condition_pen = 0
        self.selection_edge_prob = 0.6   # gene_selection_report: is_regulated_by / regulates list the edges inside the selection with probability >= this (and literature-feasible)
        self.closure_min = 0.5           # cost of a gene in the growth of the Steiner selection: the gene worth the most (sum of its w towards the selected genes + anchoring it brings to the terminals) is added while worth >= this
        self.literature_selection = True  # gene selection: edge probabilities reweighted by OmniPath feasibility, whatever the prior
        self.literature_depth = 3        # max path length in the literature graph (last edge TF -> target)
        self.literature_weight = 1.0     # exponent on the data-calibrated literature likelihood ratio (0 = data only)
        self.min_entropy_change = 0.1    # variability floor of the selection: max BUB-entropy change between consecutive times (bits, Gandrillon MDE)
        self.min_nb_separation = 0.1    # ... and separation of the extreme NB modes at the mixture initialization (times / cell types, batch_size_mixture cells per time)
        self.max_free_params = 10000     # change=1 with ref=1 and prior 0 (hard prior): gene budget set so that the literature prior leaves this many free network parameters
        self.literature_resources = 'extended'  # 'extended' (OmniPath, CollecTRI + PathwayExtra, KinaseExtra, LigRecExtra, DoRothEA A-D, TFtarget) or 'core' (OmniPath + CollecTRI)

        ## Multi-sample integration of the mixture (obs['dataset_id']; no effect with a single sample)
        self.integrate_samples = False           # True: the counts of each sample are projected, mode by mode (level by level for more than two levels), on the mixture of the reference sample, whose parameters every sample takes; False: each sample keeps its own fit and raw counts
        self.ref_sample_integration = None       # dataset_id of the reference sample; None = the first fitted sample (sorted dataset_id)
        self.min_cells_integration = 200         # smaller samples are not fitted alone (raw counts, classified with the reference modes)
        self.integration = None                  # per-sample parameters and integrated genes, set by fit_mixture_samples

        ## Proliferation
        self.simulate_with_proliferation = True # apply branching process in simulate_trajectories
        self.n_proliferation_markers = 5  # with protein_dilution or simulate_with_proliferation: genes always selected that best predict the prior net rate of the cells (non-redundant), so that R(P) of the proliferation MLP can tell dividing and arrested cells apart; 0 = none
        self.ProliferationMLP_per_condition = True  # with network conditions: one proliferation MLP per condition (its samples and genes); False: one shared MLP (forced True with select_genes_per_condition)
        self.protein_dilution = True             # proteins diluted at the birth rate b of each cell: dP/dt = d1 u - (d1 + b) P (obs['proliferation_birth_rate']; independent of simulate_with_proliferation)
        self.prolif_uses_stimulus = True         # inference stimuli are inputs of the ProliferationMLP, R(u, P) (no effect if constant over the intervals)
        self.prolif_network = None               # ProliferationMLP trained in refine_network_degradations (if simulate_with_proliferation or protein_dilution), not a parameter
        self.use_recomputed_proliferation = False # after infer_network_structure: net growth of the trajectory states = R_opt of the growth pass (True) or the prior rates of their real cells (False, obs['proliferation_net_rate'], data_R_prior.npy): proliferation MLP, birth / death of the states (dilution), growth-weighted checks and report
        self.R_prior = None                      # prior net rate of each trajectory state (rate of its real cell, same layout as R_opt)
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


    def core_binarization(self, data_rna, gene_names, vect_t, G_tot, min_components=2,
                          verb=True, kov_cell_mask=None, strata=None,
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
        n_components = 0
        kinetics = NegativeBinomialMixtureEM(min_components=2, max_components=2,
                                                 max_iter_em=int(self.max_iter_softEM or 0),
                                                 hard_em=self.hard_em,
                                                 preserve_mean_values=self.preserve_mean_values, mean_forcing_em=self.mean_forcing_em,
                                                 depth_model=self.depth_model)
        single = NegativeBinomialMixtureEM(min_components=1, max_components=1,
                                           max_iter_em=int(self.max_iter_softEM or 0), hard_em=self.hard_em, preserve_mean_values=self.preserve_mean_values,
                                           mean_forcing_em=self.mean_forcing_em, depth_model=self.depth_model)
        crit_kw = dict(criterion=self.mixture_selection_criterion, weights_per_time=self.criterion_weights_per_time)

        def run_main_loop_for_gene(g):
            if verb: print("Calibrating gene", g)
            x = data_rna[:, g]
            fit_kw = dict(vect_t=vect_t, vect_celltypes=strata, seuil=self.seuil, s=depth,
                          batch_size_mixture=self.batch_size_mixture, strata=strata)
            # Candidates: one mode (min_components = 1), uniform grids of K = 2..max_grid_levels levels between the
            # extremes of the CEM + soft EM (fit_grid_mixture), compared by mixture_criterion on their soft likelihood
            model = kinetics.fit(x, **fit_kw)
            cands = []
            if model['n_components'] >= 2:
                ks2, c2 = np.sort(model['ks']), model['c']
                for K in range(max(2, min_components), int(self.max_grid_levels) + 1):
                    grid = fit_grid_mixture(x, vect_t, ks2[0], ks2[-1], c2, s=depth, seuil=self.seuil, n_grid=K)
                    if grid is not None:
                        cands.append(grid)
                if not cands and min_components <= 2:
                    cands.append((ks2, c2, model['pi'], model['resp']))
            if min_components <= 1 or not cands:
                m1 = model if model['n_components'] < 2 else single.fit(x, **fit_kw)
                cands.append((np.sort(m1['ks']), m1['c'], m1['pi'], m1['resp']))
            scores = [mixture_criterion(x, vect_t, lv, cc, w, s=depth, **crit_kw)[0] for lv, cc, w, _ in cands]
            ks, c, pi, proba = cands[int(np.argmin(scores))]
            # Initial basins: EMD of the mixture posteriors under the mode masses (forcing modes in both modes)
            return ks, c, proba, self._basins_from_posteriors(proba, vect_t, pi), pi

        results = Parallel(n_jobs=-1)(
        delayed(seeded_call)(task_seed(self.seed, 1, g), run_main_loop_for_gene, g) for g in range(ns, G_tot)
        )

        for idx, g in enumerate(range(ns, G_tot)):

            kg, cg, probag, basinsg, pi_initg = results[idx]
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

        return frequency_modes_smooth, frequency_proba_init, frequency_basins



    def fit_mixture(self, data, gene_names=np.arange(1, 50000), min_components=None, verb=True, stimulus_schedule=None,
                    time_key='time', kov_cell_mask=None):
        """
        Fit the mixture model parameters to the data: from min_components (None: min_mixture_components) to two modes
        or to max_grid_levels levels (grids initialised by the CEM + soft EM), chosen by mixture_selection_criterion.
        """
        min_components = self.min_mixture_components if min_components is None else min_components
        set_depth_model(self.depth_model)
        self._enforce_soft_basins(verb=verb)
        seed_everything(self.seed)
        data_rna = self._parse_input(data, time_key)
        N_cells, G_tot = data_rna.shape
        vect_t = data_rna[:, 0]
        self._stim_schedule = self._build_stimulus_schedule(np.sort(np.unique(vect_t)), stimulus_schedule)

        frequency_modes_smooth, frequency_proba_init, frequency_basins = self.core_binarization(
                                        data_rna, gene_names, vect_t, G_tot,
                                        min_components=min_components,
                                        verb=verb,
                                        kov_cell_mask=kov_cell_mask,
                                        strata=self._cell_type_labels(data),
                                        depth=self._depth_factors(data))

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
                              c0=np.asarray(c0).astype(np.int64), spec_c=self._spec_chrom())

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

    def _initial_basins(self, proba, vect_t=None, masses=None):
        """Posteriors (unchanged) and initial basins (_basins_from_posteriors), as in core_binarization."""
        return proba, self._basins_from_posteriors(proba, vect_t, masses)

    def _target_masses(self, post, pi_t):
        """Target mode masses of a group of cells: masses pi_t of the mixture (force_basins) mixed with the mean of the
        posteriors post (cells x modes), normalised."""
        K = post.shape[1]
        p = post / np.maximum(post.sum(axis=1, keepdims=True), EPS)
        nu = np.asarray(pi_t, dtype=float)[:K] * self.force_basins + p.mean(axis=0) * (1 - self.force_basins)
        return nu / np.maximum(nu.sum(), EPS)

    def _emd_basins(self, proba, vect_t, masses):
        """Basins from the posteriors (cells x modes) under the mode masses (dict time -> masses): EMD of the costs
        -log posterior, per time (temporal_basins) or on all the cells (time-weighted masses)."""
        P = np.maximum(np.asarray(proba, dtype=float), 0.0)
        dist = np.clip(-np.log(np.maximum(P / np.maximum(P.max(axis=1, keepdims=True), EPS), EPS)), 0, 100)
        times = np.sort(np.unique(vect_t))
        groups = [vect_t == t for t in times] if self.temporal_basins else [np.ones(len(P), bool)]
        out = np.zeros(len(P), dtype=int)
        for m in groups:
            n = int(m.sum())
            if not n:
                continue
            pi_t = masses[vect_t[m][0]] if self.temporal_basins else \
                np.sum([np.asarray(masses[t], dtype=float)[:P.shape[1]] * np.mean(vect_t == t) for t in times], axis=0)
            nu = self._target_masses(P[m], pi_t)
            out[m] = np.argmax(ot.emd(np.ones(n) / n, nu, dist[m], numItermax=int(1e7)), axis=1)
        return out

    def _basins_from_posteriors(self, proba, vect_t=None, masses=None):
        """Initial basin of each cell from its posteriors (cells x modes): EMD under the mode masses per time
        (_emd_basins; argmax without masses)."""
        if masses is not None:
            return self._emd_basins(proba, np.asarray(vect_t), masses)
        return np.argmax(proba, axis=1)

    def _assign_from_costs(self, mu, nu, dist, dist_soft=None):
        """Basins of the cells (rows) from the costs dist = -log of their posterior + network term, with mode masses
        nu: argmax of the EMD coupling (forcing of the trajectories: right masses and ranks, deterministic), and, with
        one_hot_probas = False, the rows of the entropic coupling of regularization 1 of the costs dist_soft (dist if
        None): posteriors updated by the network, posterior x one factor per mode so that the masses are nu; else
        None."""
        idx = np.argmax(ot.emd(mu, nu, dist, numItermax=int(1e7)), axis=1)
        if self.one_hot_probas:
            return idx, None
        # Marginal tolerance 1e-7 (the default 1e-9 of POT is not reached within the iterations with levels of small
        # masses, for differences of the posteriors below 1e-4)
        coupling = ot.sinkhorn(mu, nu, dist if dist_soft is None else dist_soft, reg=1.0, method='sinkhorn_log',
                               numItermax=10000, stopThr=1e-7)
        return idx, coupling / np.maximum(coupling.sum(axis=1, keepdims=True), EPS)

    def _enforce_soft_basins(self, verb=True):
        """Model learning of the GRN sigma relies on the EM with soft posteriors and on the likelihood of the counts:
        one_hot_probas forced to False."""
        if self.model_learning_grn and self.one_hot_probas:
            if verb:
                print("[CardamomOT] model_learning_grn: one_hot_probas forced to False (soft posteriors)")
            self.one_hot_probas = False

    def _floor_mrna_degradation(self, verb=True):
        """Bursting regime: d0 >= min_ratio x d1 for every gene (and sample rates), set once on the rates of the
        literature before any use of d0 (trajectories, lags, refine, simulations), with a warning."""
        ns = self.n_stimuli
        for name in ('d', 'd_samples', 'd_literature'):
            d = getattr(self, name, None)
            if d is None:
                continue
            d = np.array(d, dtype=float)
            low = d[..., 0, ns:] < self.min_ratio * d[..., 1, ns:]
            if not low.any():
                continue
            d[..., 0, ns:] = np.where(low, self.min_ratio * d[..., 1, ns:], d[..., 0, ns:])
            setattr(self, name, d)
            if verb and name == 'd':
                genes = np.flatnonzero(np.any(low.reshape(-1, low.shape[-1]), axis=0)) + ns
                print(f"[CardamomOT] Warning: d0 < min_ratio x d1 (min_ratio = {self.min_ratio}) for {len(genes)} "
                      f"gene(s) (indices {genes.tolist()}): d0 set to min_ratio x d1 (bursting regime)")

    def _spec_grn(self):
        """Families of the GRN sigma used everywhere (inference and simulations): sigma_spec_grn with
        model_learning_grn, else None (logistic)."""
        return self.sigma_spec_grn if self.model_learning_grn else None

    def _penalty_kw(self):
        """Penalties of the network fits (inference_network_multi)."""
        return dict(qut_level=self.qut_level, qut_n_perm=self.qut_n_perm, network_ridge=self.network_ridge)

    def _spec_chrom(self):
        """Families of the chromatin opening function: sigma_spec_chrom with model_learning_chrom, else None
        (logistic)."""
        return self.sigma_spec_chrom if self.model_learning_chrom else None

    @staticmethod
    def _mode_network_cost(kon_pred, obj):
        """Network term of the basin costs, -log p(mode | network) = KL(tau_m || sigma) (cells x modes), tau_m and
        sigma the sigmoid targets of the modes obj and of the predicted kon (same cross-entropy as the network fit),
        and the gene weight w = 1 - k_low / k_high (cells x 1) of the EMD basins, so that weakly separated genes keep
        a weak network term in the forcing. Returns (KL, w)."""
        k_lo, k_hi = obj[:, :1], obj.max(axis=1, keepdims=True)
        sig = sigmoid_target(np.asarray(kon_pred, dtype=float)[:, None], k_lo, k_hi)
        tau = sigmoid_target(obj, k_lo, k_hi)
        sig_b = np.ascontiguousarray(np.broadcast_to(sig, tau.shape), dtype=float).ravel()
        kl = cross_entropy(sig_b, np.ascontiguousarray(tau, dtype=float).ravel()).reshape(tau.shape)
        w = np.clip(1.0 - k_lo / np.maximum(k_hi, EPS), 0.0, 1.0)
        return kl, w

    def _set_mode_soft(self, cells, g, m_idx):
        """Forced mode m_idx of gene g for the cells (KO/OV): soft mode and one-hot posteriors (one_hot_probas = False)."""
        if self.modes_soft is None or not np.any(cells):
            return
        self.modes_soft[cells, g] = self.modes[cells, g]
        self.posteriors_soft[cells, g, :] = 0.0
        self.posteriors_soft[cells, g, m_idx] = 1.0

    def _posterior_modes(self, P, ks, vect_samples_id, samples_id):
        """Expected mode (normalised kon) of each cell and gene under the posteriors P (N, G, M) (one_hot_probas =
        False: soft targets of the network fit)."""
        ks_c = ks[np.searchsorted(np.asarray(samples_id), vect_samples_id)] if np.ndim(ks) == 3 else np.asarray(ks)[None]
        M = min(P.shape[2], ks_c.shape[2])
        Pn = P[..., :M] / np.maximum(P[..., :M].sum(axis=2, keepdims=True), EPS)
        return (Pn * ks_c[..., :M]).sum(axis=2)

    def fit_mixture_samples(self, data, sample_key='dataset_id', kov_genes=None, stimulus_schedule=None,
                            time_key='time', verb=True, **kwargs):
        """
        Mixture fit with several samples (>= 2 in obs[sample_key]); with a single sample, identical to fit_mixture.

        Each sample (>= min_cells_integration cells) gets its own mixture fit (CEM + soft EM; levels of the continuous
        model). The reference sample is ref_sample_integration (a dataset_id), else the first
        fitted sample; per gene, the first fitted sample where the gene is active when it is not active in the reference
        (select_genes_per_condition). With integrate_samples, the counts of every (sample, gene) pair are projected on
        the reference: each mode of the sample goes to the mode of the reference that best explains its cells
        (integration.projection_targets; groups of levels for more than two levels), and each cell, in the mode (level)
        of largest posterior in its sample, is quantile-matched from the NB of that mode to the NB of its target
        (integration.integrate_counts); every sample takes the parameters of the reference. Pairs left raw and classified with the parameters of the
        reference (_posteriors_and_masses): the perturbed ones (kov_genes: {dataset_id: set of genes}, so that the
        perturbation stays visible against the reference level), with integrate_samples the genes with a single mode in
        the reference (nothing to project on), and the cells of unfitted samples. Without
        integrate_samples, each sample keeps its own fit and raw counts. self.a is always (S, M+1, G), samples ordered
        as np.unique(obs[sample_key]) (= sample index of fit_network).

        Returns the integrated count matrix (N, n_genes), or None if counts were not changed.
        """
        set_depth_model(self.depth_model)
        from ..inference.integration import integrate_counts, projection_targets
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
        integrate = bool(self.integrate_samples)

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
                     pi_init=self.pi_init, active=active)
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

        # --- 2. Reference sample of each gene, and the fate of each (sample, gene) pair ---
        ids = [str(f['id']) for f in fits]
        ref = 0
        if self.ref_sample_integration is not None:
            if str(self.ref_sample_integration) in ids:
                ref = ids.index(str(self.ref_sample_integration))
            elif verb:
                print(f"[integration] Warning: reference sample {self.ref_sample_integration} not among the fitted "
                      f"samples {ids}; using the first one ({ids[0]})")
        act_fit = np.stack([np.ones(G_tot, bool) if f['active'] is None else f['active'] for f in fits])
        ref_of = np.full(G_tot, -1)
        for g in range(ns, G_tot):
            cand = [ref] + [i for i in range(S) if i != ref]
            ref_of[g] = next((i for i in cand if act_fit[i, g]), -1)
        a = np.ones_like(a_samples[0])
        for g in np.flatnonzero(ref_of >= 0):
            a[:, g] = a_samples[ref_of[g], :, g]
        one_mode = lambda a_s, g: a_s[:-1, g].max() <= a_s[:-1, g].min() * (1 + 1e-9)
        perturbed = np.zeros((S, G_tot), dtype=bool)  # KO/OV pairs of perturbation_inference: raw, reference modes
        proj = np.zeros((S, G_tot), dtype=bool)       # counts projected on the reference (integrate_samples)
        own = np.zeros((S, G_tot), dtype=bool)        # own fit and raw counts
        for i, f in enumerate(fits):
            for gene in (kov_genes or {}).get(str(f['id']), ()):
                if gene in gene_names:
                    perturbed[i, ns + gene_names.index(gene)] = True
            for g in range(ns, G_tot):
                if not act_fit[i, g] or perturbed[i, g]:
                    continue
                if not integrate or i == ref_of[g]:
                    own[i, g] = True
                elif not one_mode(a, g):
                    proj[i, g] = True

        # Genes active in no fitted sample: parameters of the pooled fit
        no_ref = np.flatnonzero(ref_of[ns:] < 0) + ns
        if len(no_ref):
            if verb:
                print(f"[integration] {len(no_ref)} genes active in no fitted sample: pooled mixture parameters")
            # One pooled fit per pattern of samples where the genes are active (all samples without per-condition genes)
            patterns = {}
            for g in no_ref:
                patterns.setdefault(tuple(np.ones(len(sample_ids), bool) if gm is None else gm[:, g]), []).append(g)
            for key, genes_p in patterns.items():
                cells = np.isin(samples, [sample_ids[k] for k in np.flatnonzero(key)])
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
        a[:, :ns] = 1
        # Modes up to the highest one (padding rows are below it; the lowest level may be ~0)
        n_modes = np.full(G_tot, M)
        for g in range(ns, G_tot):
            n_modes[g] = max(2, 1 + int(np.argmax(a[:-1, g]))) if np.all(np.isfinite(a[:, g])) else M

        # --- 3. Projected counts and responsibilities ---
        rng = np.random.default_rng(self.seed)
        depth = self._depth_factors(data)
        X = data_rna.copy()
        proba_init = np.zeros((N, G_tot, M))
        basins = np.zeros((N, G_tot), dtype=np.int8)
        fitted = np.zeros(N, dtype=bool)
        dst = np.repeat(a[None, :-1], S, axis=0)  # target burst frequency of each mode of each sample
        tgt_all = np.repeat(np.arange(M)[None, :, None], S, axis=0).repeat(G_tot, axis=2)  # and its target mode
        for i, f in enumerate(fits):
            m = f['mask']
            fitted |= m
            for g in range(ns, G_tot):
                if proj[i, g]:
                    # Each mode (group of levels) of the sample goes to the reference mode that best explains its cells
                    z = np.argmax(f['proba_init'][:, g, :], axis=1)
                    n_src, n_ref = max(2, 1 + int(np.argmax(a_samples[i, :-1, g]))), n_modes[g]
                    pi_ref = fits[ref_of[g]]['pi_init'][g - ns]
                    w_ref = np.mean([np.pad(np.asarray(p, dtype=float), (0, M))[:n_ref] for p in pi_ref.values()], axis=0)
                    tgt = projection_targets(data_rna[m, g], z, n_src, a[:n_ref, g], a[-1, g], w_ref,
                                             s=None if depth is None else depth[m],
                                             grid=n_src > 2)
                    dst[i, :n_src, g], tgt_all[i, :n_src, g] = a[tgt, g], tgt
                    X[m, g] = integrate_counts(data_rna[m, g], z, (a_samples[i, :-1, g], a_samples[i, -1, g]),
                                               (dst[i, :, g], a[-1, g]), rng, s=None if depth is None else depth[m])
                    # Posteriors and basins carried to the target modes
                    for l in range(n_src):
                        proba_init[m, g, tgt[l]] += f['proba_init'][:, g, l]
                    basins[m, g] = tgt[np.minimum(f['basins'][:, g], n_src - 1)]
                elif own[i, g]:
                    proba_init[m, g] = f['proba_init'][:, g]
                    basins[m, g] = f['basins'][:, g]
        # Raw pairs: classify the raw counts with the reference modes (posteriors and masses as a fit)
        raw_cells, raw_masses = {}, {}
        pos = np.searchsorted(np.asarray(sample_ids), samples)
        cell_active = None if gm is None else gm[pos]  # (N, G_tot)
        for g in range(ns, G_tot):
            ng = n_modes[g]
            raw = ~fitted.copy()
            for i, f in enumerate(fits):
                if act_fit[i, g] and not (proj[i, g] or own[i, g]):
                    raw |= f['mask']
            if cell_active is not None:
                raw &= cell_active[:, g]
            raw_cells[g] = raw
            if raw.any():
                resp, raw_masses[g] = self._posteriors_and_masses(
                    data_rna[raw, g], vect_t[raw], a[:ng, g], a[-1, g],
                    depth=None if depth is None else depth[raw])
                proba_init[raw, g, :ng], basins[raw, g] = self._initial_basins(resp, vect_t[raw], raw_masses[g])
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

        # --- 4. Mixture outputs, as fit_mixture would give on the projected data ---
        self._stim_schedule = sched_full
        # Reference parameters, except the pairs keeping their own fit
        a_per = np.repeat(a[None], len(sample_ids), axis=0)
        for i, f in enumerate(fits):
            k = sample_ids.index(f['id'])
            a_per[k][:, own[i]] = a_samples[i][:, own[i]]
        if gm is not None:  # genes outside each sample's condition: equal modes
            for k in range(len(sample_ids)):
                a_per[k][:, ~gm[k]] = 1.0
        self.a = a_per
        # Modes in units of the max burst rate of each cell's sample
        kz_cells = a_per[pos, :-1]                                   # (N, M, G)
        modes = (np.take_along_axis(kz_cells, basins[:, None, :].astype(int), axis=1)[:, 0]
                 / np.maximum(kz_cells.max(axis=1), EPS))
        for t in times_all:
            modes[vect_t == t, :ns] = sched_full[t]
        self.modes = modes
        self.proba_init = proba_init
        self.basins = basins
        # Mode masses per time: those of each sample's own fit, and of the raw cells, by cells
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
                for i, f in enumerate(fits):
                    if f['active'] is not None and not f['active'][g]:
                        continue  # gene outside the sample's condition
                    n = np.sum(f['mask'] & mt & ~raw_cells[g])
                    if n:  # masses carried to the target modes of the projected pairs
                        p += n * np.bincount(tgt_all[i, :, g], weights=masses_at(f['pi_init'][g - ns], t), minlength=M)
                n_raw = np.sum(raw_cells[g] & mt)
                if n_raw:
                    p += n_raw * masses_at(raw_masses[g], t)
                p = p[:ng]
                if p.sum() <= 0:  # no active cell at t: single mode
                    p[0] = 1.0
                pi_g[t] = p / (p.sum() + EPS)
            self.pi_init.append(pi_g)

        # Per-sample parameters, kept to project other files of the same samples
        self.integration = dict(
            sample_ids=[f['id'] for f in fits], a_samples=a_samples, pi_init_samples=[f['pi_init'] for f in fits],
            a=a, dst=dst, proj=proj, own=own, n_modes=n_modes, ref=fits[ref]['id'], sample_key=sample_key,
            integrate=integrate, all_sample_ids=sample_ids, gene_mask=gm)
        if verb:
            print(f"[integration] integrate_samples = {integrate}, reference sample {fits[ref]['id']}; genes projected "
                  "per sample: " + ", ".join(f"{f['id']}: {n}/{G_tot - ns}" for f, n in zip(fits, proj[:, ns:].sum(1))))
        return X[:, ns:] if proj.any() else None

    def _expand_inactive(self, f, active, times):
        """Fit f of a sample on its active genes brought to every gene (G_tot): inactive genes with a single mode
        (equal modes, posterior on the first one, basin 0, masses [1, 0, ...])."""
        G_tot, idx = len(active), np.flatnonzero(active)
        n, M = f['proba_init'].shape[0], f['proba_init'].shape[2]
        a = np.ones((f['a'].shape[0], G_tot))
        a[:, idx] = f['a']
        proba = np.zeros((n, G_tot, M))
        proba[:, ~active, 0] = 1.0
        proba[:, idx] = f['proba_init']
        basins = np.zeros((n, G_tot), dtype=f['basins'].dtype)
        basins[:, idx] = f['basins']
        ns = self.n_stimuli
        single = {t: np.eye(M)[0] for t in times}
        it = iter(f['pi_init'])
        f.update(a=a, proba_init=proba, basins=basins,
                 pi_init=[next(it) if active[g] else single for g in range(ns, G_tot)])

    def integrate_data(self, data, time_key='time'):
        """
        Projected counts (N, n_genes) of another AnnData of the same samples (e.g. data_test): each cell is classified
        with the mixture of its own sample (masses of its time), then quantile-matched on the target of its mode in
        the reference as in fit_mixture_samples. Cells of unfitted samples and pairs not projected keep raw counts.
        """
        from ..inference.integration import integrate_counts
        info = self.integration
        ns = self.n_stimuli
        data_rna = self._parse_input(data, time_key)
        X = data_rna.copy()
        vect_t = data_rna[:, 0]
        key = info['sample_key']
        if key not in data.obs or not info['integrate']:
            return X[:, ns:]
        samples = data.obs[key].values.astype(str)
        depth = self._depth_factors(data)
        rng = np.random.default_rng(None if self.seed is None else self.seed + 1)
        a_ref = info['a']
        for i, sid in enumerate(info['sample_ids']):
            m = samples == str(sid)
            if not m.any():
                continue
            a_s = info['a_samples'][i]
            for g in np.flatnonzero(info['proj'][i]):
                ng = max(2, 1 + int(np.argmax(a_s[:-1, g])))
                pi_t = info['pi_init_samples'][i][g - ns]
                z = np.zeros(m.sum(), dtype=int)
                for t in np.unique(vect_t[m]):
                    mt = vect_t[m] == t
                    prior = np.asarray(pi_t[t], dtype=float)[:ng] if t in pi_t else None
                    resp, _ = predict_resp(data_rna[m, g][mt], a_s[:ng, g], a_s[-1, g], pi=prior,
                                           s=None if depth is None else depth[m][mt])
                    z[mt] = np.argmax(resp, axis=1)
                X[m, g] = integrate_counts(data_rna[m, g], z, (a_s[:-1, g], a_s[-1, g]),
                                           (info['dst'][i, :, g], a_ref[-1, g]), rng,
                                           s=None if depth is None else depth[m])
        return X[:, ns:]

    def integration_report(self, gene_names):
        """DataFrame (sample, gene): mode means of the sample, of the reference and used, and whether the pair was
        projected on the reference."""
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
                row = {'sample': sid, 'gene': gene_names[g - ns], 'integrated': bool(info['proj'][i, g]),
                       'integrate_samples': float(info['integrate']), 'reference': info['ref']}
                a_d = a_s if info['own'][i, g] else info['a']
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
    
    
    def adaptive_shrinkage_init(self, x, mu, p=.5, levels=None, groups=None):
        """
        Initial proteins, per group of cells (sample of per-sample mixtures; one group by default) and gene: each mode
        index gets a band (by rank among the levels of the group, between its lowest and highest one), each cell is
        placed in the band of its mode by its shrunk mRNA among the cells of that mode. levels (N or 1, G, M): levels of
        the mode indices of each cell (default: distinct values of mu over the group), the mode of a cell being the level
        nearest to mu (default band for a mode without cell).
        """
        N, G = mu.shape
        res = mu * self.scale_proteins
        groups = np.zeros(N, dtype=int) if groups is None else np.asarray(groups)
        for grp in np.unique(groups):
            rows = np.flatnonzero(groups == grp)
            for g in range(G):
                if levels is None:
                    lv = np.sort(np.unique(mu[rows, g]))
                else:
                    lv = np.asarray(levels[rows[0] if len(levels) > 1 else 0, g], dtype=float)
                    lv = lv[:1 + int(np.argmax(lv))]
                lmax = len(lv)
                assign = np.argmin(np.abs(mu[rows, g, None] - lv[None, :]), axis=1)
                a, b = lv[0] * self.scale_proteins, lv[-1] * self.scale_proteins
                xr = x[rows, g]
                xs = self.adaptive_shrinkage(xr, lv[assign])
                x_min, x_max = np.min(xs), np.max(xs)
                xs = xs - x_min
                bands = []
                for m in range(lmax):
                    ind = assign == m
                    lo, hi = x_max / (1 + lmax - m), x_max / (lmax - m)
                    if ind.any():
                        lo, hi = min(np.min(xs[ind]), lo), max(np.max(xs[ind]), hi)
                    bands.append((lo, hi))

                def value(m, xs_m):
                    lo, hi = bands[m]
                    return a + (b - a) * (m + (np.clip(xs_m, lo, hi) / (hi + EPS))**p) / lmax

                for m in range(lmax):
                    ind = assign == m
                    res[rows[ind], g] = value(m, xs[ind])
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
            # Bands per sample and mode index (levels of the sample of the real cell), at the forcing mode of each cell
            def init_prot(x, mu, real):
                s_real = np.searchsorted(np.asarray(samples_id), vect_samples_id[real])
                levels = (ks[s_real] if np.ndim(ks) == 3 else np.asarray(ks)[None])[:, ns:]
                return self.adaptive_shrinkage_init(x, mu, levels=levels, groups=s_real)

            real_start = sim_real_idx[start_rows]
            prot_modified[start_rows, ns:] = init_prot(
                rna_modified[start_rows, ns:] * _rows(s1_traj, start_rows), kon_modified[start_rows, ns:], real_start) \
                * self._dilution(real_start)[1]
            if n_iter == 1:
                s1_cells = s1_rows(s1, vect_samples_id)
                cells = np.flatnonzero(start_cells)
                prot_formodes[start_cells, ns:] = init_prot(
                    vect_rna[start_cells, ns:] * _rows(s1_cells, start_cells), self.modes[start_cells, ns:], cells) \
                    * self._dilution(cells)[1]

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
                        # Network term: same target as the network fit (soft with one_hot_probas = False), the forcing
                        # of the bridge and alpha keeping the forcing modes (EMD basins)
                        sigmoid_target((self.modes[cell_idx] if self.modes_soft is None
                                        else self._soft_target_modes(cell_idx))[:, ns:],
                                       kb_s[ns:, 0], kb_s[ns:, 1]), w_s,
                        n_iter=n_iter, intensity_prior=intensity_prior,
                        n_stimuli=ns, stim_vals=np.asarray(self._stim_schedule.at(times[b], s_idx), dtype=np.float64),
                        scale_proteins=self.scale_proteins,
                        weight_fixed=-1.0 if context is None else context['weight_init'],
                        b_init=b_src, b_end=b_tgt, alpha_tol=self.alpha_threshold,
                        d0=np.zeros(0) if d0_br is None else d0_br, rna_to_x=rna_to_x,
                        g_active=np.zeros(0) if self.gene_mask is None
                        else self.active_genes(s_idx)[ns:].astype(np.float64),
                        spec=self._spec_grn(), **self._memory_ot_kw(s_idx, times[b] - time),
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
                    # Unbalanced relaxation: exponent of the target marginal fixed at its value for the regularization
                    # of min_n_loops (reg_m follows the annealed reg)
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
                            self._record_coupling(pairwise_dist, src_real, cell_idx, delta_t, reg, tmp, reg_ref,
                                                  s_idx, time, times[b])

                # Trajectory update (skipped by the growth pass)
                if N_sample and N_cells and not growth_only:
                    # --- Growth-weighted OT marginals ---
                    # Symmetric convention (ours, not WOT's, which weights the source by exp(R·Δt) and keeps
                    # the target uniform): source and target corrected by exp(±R·Δt/2), the midpoint rule for
                    # the growth along the step — see docs/advanced.md#net-proliferation-rate--default-behaviour.
                    mu, nu, reg_m = self._traj_marginals(src_real, cell_idx, delta_t, reg, tmp, reg_ref)
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
                        next_prot = prot_bridge_end(*br_args)
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

                    # Re-assign alpha from the nearest old trajectory of the block (proteins and kon, half each)
                    if prot_old_blk is not None:
                        d_match = match_distance(next_prot, kon_modified[next_indices, ns:], prot_old_blk, kon_old_blk)
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
        """Alpha of the nearest training state (match_distance on proteins and kon, as in training) at the time and
        sample of the states."""
        ns = self.n_stimuli
        k = min(int(np.argmin(np.abs(pool['t'] - t))), pool['alpha'].shape[0] - 1)
        rows = np.flatnonzero((pool['slot_sample'] == pool['sample_map'][s_idx]) & pool['valid'][k])
        if not len(rows):
            rows = np.flatnonzero(pool['valid'][k])
        d_match = match_distance(prot, kon, pool['prot'][k][rows, ns:], pool['kon'][k][rows, ns:])
        return pool['alpha'][k][rows[np.argmin(d_match, axis=1)]]

    def _traj_marginals(self, src_real, cell_idx, delta_t, reg, log_G, reg_ref=None):
        """
        Marginals (mu, nu) and unbalanced relaxation of the trajectory OT between source states and real cells. The
        target marginal is enforced with the exponent f = reg_m / (reg_m + reg) of the unbalanced Sinkhorn, fixed at
        every iteration at its value for reg_m = unbalanced_reg * log_G (log_G: typical excess cost of the batch, the
        scale of the cost) and reg = reg_ref (the entropic regularization of min_n_loops): reg_m follows the annealed
        reg, so that the relaxation does not vanish with it.
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
        if self.unbalanced_reg:
            rm = self.unbalanced_reg * log_G
            if reg_ref is not None:
                rm *= reg / reg_ref  # f fixed at its value at min_n_loops
            reg_m = np.array([1e6, rm])
        return mu, nu, reg_m

    def _record_coupling(self, C, src_real, cell_idx, delta_t, reg, log_G, reg_ref, s_idx, t_from, t_to, top_k=50):
        """
        Soft coupling of the final trajectories (same costs and marginals as the trajectory OT), kept sparse:
        the top_k targets of each source state; entries (real source cell, real target cell, mass), the
        mass of each state scaled to ~1 (batches of different sizes), summed later over duplicated states.
        """
        mu, nu, reg_m = self._traj_marginals(src_real, cell_idx, delta_t, reg, log_G, reg_ref)
        P = self._solve_ot(mu, nu, C, reg, reg_m) * len(src_real)
        k = min(top_k, P.shape[1])
        cols = np.argpartition(-P, k - 1, axis=1)[:, :k]
        w = np.take_along_axis(P, cols, axis=1)
        keep = w > 1e-3 * np.maximum(w.max(axis=1, keepdims=True), 1e-300)
        rows = np.broadcast_to(np.arange(len(src_real))[:, None], cols.shape)
        self.couplings.append(dict(src=src_real[rows[keep]], tgt=cell_idx[cols[keep]], w=w[keep].astype(np.float32),
                                   sample=s_idx, t_from=float(t_from), t_to=float(t_to)))

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
        """
        Entropic OT plan (unbalanced if reg_m is given), as POT (sinkhorn / sinkhorn_unbalanced, entropy relative to
        mu nu^T). The cost of each source row is shifted by its minimum c, so that the Gibbs kernel keeps an entry of 1
        per row (a row whose pairs all exceed ~745 reg would underflow to 0: Sinkhorn stopped at an early iterate, empty
        rows); the plan is unchanged, the source masses compensating exactly (unbalanced: mu exp(c / reg_m[0]), the
        reference measure kept at mu nu^T). The rare remaining failures (numerical warnings, non-finite or empty rows,
        compensation out of range with a relaxed source marginal) are solved again in the log domain (_sinkhorn_log).
        """
        c_r = C.min(axis=1)
        Cc = C - c_r[:, None]
        numItermax = int(10000 / min(1, reg))
        P = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if reg_m is None:
                P = ot.bregman.sinkhorn(mu, nu, Cc, reg=reg, numItermax=numItermax, stopThr=self.stopThr_init)
            elif np.max(c_r) / reg_m[0] < 500:
                P = ot.unbalanced.sinkhorn_unbalanced(mu * np.exp(c_r / reg_m[0]), nu, Cc, reg=reg, reg_m=reg_m,
                                                      c=mu[:, None] * nu[None, :], numItermax=numItermax,
                                                      stopThr=self.stopThr_init)
        numerical = any(k in str(w.message) for w in caught
                        for k in ('Numerical errors', 'divide by zero', 'overflow', 'invalid value'))
        if P is None or numerical or not np.all(np.isfinite(P)) or np.any(P.sum(axis=1) <= 0):
            P = _sinkhorn_log(mu, nu, C, reg, reg_m, tol=self.stopThr_init)
        return P

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
        sels = self._network_subsamples(times_vec, samples_vec, labels, n_fits, n_params, valid=valid)
        return self._mean_fit(fit_fn(sels))

    @staticmethod
    def _mean_fit(fits):
        """(basal, inter, basal_tmp, inter_tmp): mean of the fits on the sub-samples."""
        return tuple(np.mean([f[i] for f in fits], axis=0) for i in range(4))

    def _network_subsamples(self, times_vec, samples_vec, labels, n_fits, n_params, valid=None):
        """Disjoint sub-samples (index arrays) of the network fits, see _fit_theta_averaged."""
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
        return sels

    def _model_learning_candidates(self):
        """Candidate families (phi, link) of the GRN sigma: model_learning_phi x model_learning_link."""
        as_list = lambda v: [v] if isinstance(v, str) else list(v)
        links = [link_code(name) for name in as_list(self.model_learning_link)]
        if any(link != LINK_LOGISTIC for link in links):
            raise ValueError("model_learning_link: the network fit supports the logistic link only")
        return [(phi_code(p), link) for link in links for p in as_list(self.model_learning_phi)]

    def _model_learning_window(self, min_n_loops):
        """(first, last + 1) iterations of the family selection of model_learning_grn."""
        start = min_n_loops if self.model_learning_start is None else int(self.model_learning_start)
        freeze = start + 5 if self.model_learning_freeze is None else int(self.model_learning_freeze)
        return start, freeze

    def _reset_model_learning(self, G_tot):
        """Families of the targets at the start of fit_network: the single candidate if there is one (sigma forced for
        the whole pipeline, no selection), else the sigmoid of a linear input until the selection; kappa and Pbar of the
        regulators (phi = log); stimuli always identity."""
        cands = self._model_learning_candidates()
        phi, link = cands[0] if len(cands) == 1 else (PHI_IDENTITY, LINK_LOGISTIC)
        self.sigma_spec_grn = default_spec(G_tot, kappa=self.model_learning_kappa * self.scale_proteins,
                                           pbar=self.scale_proteins, phi=phi, link=link)
        self.sigma_spec_grn[:self.n_stimuli, SPEC_PHI] = PHI_IDENTITY
        self.sigma_spec_grn[:self.n_stimuli, SPEC_LINK] = LINK_LOGISTIC
        self._ml_theta, self._ml_pending = {}, {}
        self.model_learning_history = []

    def _sigma_bar(self, y_prot, mem, ks, inter, basal, samples_id, samples_data, sigma_spec=None):
        """sigma (n, G_tot) at the states y_prot, averaged on the mRNA memory of each target if mem (PathProt of the same
        rows, targets_accounting_late_mrna) is given."""
        kb = sigmoid_amplitudes(ks)
        kb = kb[np.searchsorted(np.asarray(samples_id), np.asarray(samples_data))] if np.ndim(ks) == 3 else kb[None]
        sig_of = lambda Y, q: sigmoid_target(self._kon_ref_per_sample(
            Y, ks, inter, basal, samples_id=samples_id, samples_data=samples_data, sigma_spec=sigma_spec),
            kb[..., 0], kb[..., 1])
        if mem is None:
            return sig_of(y_prot, 0)
        # Each term on the rows where it has a weight (e.g. a single lagged node: the end state of the first rows only)
        out = np.zeros((len(y_prot), y_prot.shape[1]))
        terms = [(mem.weights[:, 0], y_prot)] + [(mem.weights[:, q + 1], mem.nodes[:, q]) for q in range(mem.nodes.shape[1])]
        for q, (w, Y) in enumerate(terms):
            rows = np.flatnonzero(np.max(w, axis=1) > 0)
            if len(rows) == len(y_prot):
                out += w * sig_of(Y, q)
            elif len(rows):
                sub = self._sigma_bar(Y[rows], None, ks, inter, basal, samples_id, np.asarray(samples_data)[rows],
                                      sigma_spec)
                out[rows] += w[rows] * sub
        return np.clip(out, 0.0, 1.0)

    def _kon_read(self, y_prot, mem, ks, inter, basal, samples_id, samples_data):
        """kon (n, G_tot) read on the mRNA: sigma along the mRNA memory mem (_sigma_bar) between the bounds of the
        sigmoid of the sample of each row (stimuli columns: those of y_prot)."""
        ns = self.n_stimuli
        sig = self._sigma_bar(y_prot, mem, ks, inter, basal, samples_id, samples_data)
        kb = sigmoid_amplitudes(ks)
        kb = kb[np.searchsorted(np.asarray(samples_id), np.asarray(samples_data))] if np.ndim(ks) == 3 else kb[None]
        kon = np.array(y_prot, dtype=float)
        kon[:, ns:] = (kb[..., 0] + (kb[..., 1] - kb[..., 0]) * sig)[:, ns:]
        return kon

    def _nb_nll(self, sig, x, samples, depth=None):
        """Negative log-likelihood (n, G_tot) of the counts x (n, G_tot) under the continuous model of the mixture:
        NB of the burst frequency k_low + (k_high - k_low) sigma (lowest / highest mode and rate of the mixture of the
        sample of each row, samples = sample indices), at the depth of each cell (depth_model)."""
        from scipy.special import gammaln
        from ..inference.depth_model import nb_shape_rate, set_depth_model
        set_depth_model(self.depth_model)
        a = np.asarray(self.a, dtype=float)
        a_r = a[np.minimum(np.asarray(samples).astype(int), a.shape[0] - 1)] if a.ndim == 3 else a[None]
        k_lo, k_hi, c = a_r[:, :-1].min(axis=1), a_r[:, :-1].max(axis=1), a_r[:, -1]   # (n or 1, G_tot)
        r, rate = nb_shape_rate(k_lo + (k_hi - k_lo) * sig, c, None if depth is None else np.asarray(depth)[:, None])
        r, rate = np.maximum(np.asarray(r, dtype=float), 1e-10), np.asarray(rate, dtype=float)
        x = np.asarray(x, dtype=float)
        return -(gammaln(x + r) - gammaln(r) - gammaln(x + 1) + r * np.log(rate / (1 + rate)) - x * np.log1p(rate))

    def _held_out_ce(self, fits, sels, spec, y_prot, y_samples, y_real, tau_cells, ks, sample_idx, mem=None):
        """Score (G_tot,) of each target, of the fit of each sub-sample on the states of the other ones (states with a
        real cell; sigma averaged on the mRNA memory with mem): cross-entropy against tau_cells (N_cells, G_tot),
        sigmoid targets of the mixture alone, or (model_learning_score = 'nb') negative log-likelihood of the counts
        of their real cells under the continuous model of the mixture (_nb_nll)."""
        counts = getattr(self, '_nb_counts', None) if self.model_learning_score == 'nb' else None
        depth = getattr(self, '_nb_depth', None)
        acc, n = np.zeros(y_prot.shape[1]), 0
        for k, fit in enumerate(fits):
            held = np.concatenate([sel for m, sel in enumerate(sels) if m != k])
            held = held[y_real[held] >= 0]
            if not len(held):
                continue
            sig = self._sigma_bar(y_prot[held], None if mem is None else mem.select_cells(held), ks, fit[1], fit[0],
                                  sample_idx, y_samples[held], sigma_spec=spec)
            if counts is not None:
                cells = y_real[held]
                ce = self._nb_nll(sig, counts[cells], y_samples[held], None if depth is None else depth[cells])
            else:
                tau = tau_cells[y_real[held]]
                ce = cross_entropy(np.ascontiguousarray(sig, dtype=float).ravel(),
                                   np.ascontiguousarray(tau, dtype=float).ravel()).reshape(sig.shape)
            acc += ce.mean(axis=0)
            n += 1
        return acc / max(n, 1)

    def _select_sigma_families(self, fit_on_spec, sels, basal_start, inter_start, y_prot, y_samples, y_real,
                               tau_cells, ks, sample_idx, n_iter, verb=True, mem=None):
        """
        Model learning of the GRN sigma at one network update: the network is fitted with every candidate family for
        every target (same sub-samples, warm start from the last fit of the family), each target is scored by the
        held-out cross-entropy of the fits against the mixture alone (_held_out_ce), and changes family when another
        one is better by model_learning_margin at two consecutive selections. Returns (basal, inter, basal_tmp,
        inter_tmp) assembled per target from the fits of its family; updates sigma_spec_grn.
        """
        ns = self.n_stimuli
        cands = self._model_learning_candidates()
        spec = self.sigma_spec_grn
        G = spec.shape[0]
        # Warm start of a family fitted for the first time: theta of the current fit (same units for every phi: change
        # of the logit between 0 and the high-mode level)
        means, scores = [], np.zeros((len(cands), G))
        for f, (phi, link) in enumerate(cands):
            spec_f = spec.copy()
            spec_f[ns:, SPEC_PHI], spec_f[ns:, SPEC_LINK] = phi, link
            b0, i0 = self._ml_theta.get(f, (basal_start, inter_start))
            fits = fit_on_spec(spec_f, b0, i0)(sels)
            means.append(self._mean_fit(fits))
            self._ml_theta[f] = means[-1][:2]
            scores[f] = self._held_out_ce(fits, sels, spec_f, y_prot, y_samples, y_real, tau_cells, ks, sample_idx,
                                          mem=mem)
        # Current family of each target, then the switches
        cur = np.zeros(G, dtype=int)
        for f, (phi, link) in enumerate(cands):
            cur[(spec[:, SPEC_PHI] == phi) & (spec[:, SPEC_LINK] == link)] = f
        best = np.argmin(scores, axis=0)
        for g in range(ns, G):
            gain = (scores[cur[g], g] - scores[best[g], g]) / max(scores[cur[g], g], EPS)
            if best[g] != cur[g] and gain > self.model_learning_margin:
                prev = self._ml_pending.get(g)
                self._ml_pending[g] = (best[g], prev[1] + 1 if prev is not None and prev[0] == best[g] else 1)
                if self._ml_pending[g][1] >= 2:
                    cur[g] = best[g]
                    del self._ml_pending[g]
            else:
                self._ml_pending.pop(g, None)
        for g in range(ns, G):
            spec[g, SPEC_PHI], spec[g, SPEC_LINK] = cands[cur[g]]
        self.model_learning_history.append(dict(n_iter=n_iter, scores=scores, family=cur.copy()))
        if verb:
            counts = ', '.join(f"{PHI_NAMES[p]}/{LINK_NAMES[l]}: {int(np.sum(cur[ns:] == f))}"
                               for f, (p, l) in enumerate(cands))
            print(f"[model_learning] iteration {n_iter}: families of the targets {counts}")
        # Parameters of each target from the fit of its family
        out = [np.array(m, dtype=float, copy=True) for m in means[0]]
        for g in range(ns, G):
            for i in range(4):
                out[i][..., g] = means[cur[g]][i][..., g]
        return tuple(out)

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
        # Flow-matching nodes along the interval ending at each state (rows without previous interval: the state)
        y_prot_prev = None
        if compute_theta:
            _, prev_cols = active_regulators(self.ref_network, inter_ref, hard_forcing_ref)
            y_prot_prev = PathProt.zeros(len(times) * N_tot, self.flow_matching_nodes, G_tot)
        # Nodes of the mRNA memory of each target along the interval ending at each state (targets_accounting_late_mrna)
        y_prot_mem = PathProt.zeros(len(times) * N_tot, self._mem_node_count(G_tot), G_tot) \
            if self.targets_accounting_late_mrna and len(times) > 1 else None
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

        def refresh_prev_prot(path=None, memory=False, n_mem=None):
            # Flow-matching nodes along each interval (consecutive observed times of each sample), or (memory) the
            # nodes of the mRNA memory of each target (_memory_nodes)
            path = y_prot_prev if path is None else path
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
                frac, w_mem = self._memory_nodes(self._mrna_d0(y_samples[idx_next]), times[b] - times[a],
                                                 n_nodes=n_mem) if memory else (None, None)
                # Flow matching: kon in force at the nodes switches at the commitment of the network, before the
                # switch of an mRNA-level forcing (alpha) by the delay of its equivalent step
                d0_sw = self._switch_d0(y_samples[idx_next]) if path is y_prot_prev else None
                kon_sw = None if d0_sw is None else \
                    kon_switch_fraction(y_alpha[a][slots], d0_sw, times[b] - times[a])
                self._fill_path_prot(
                    path, idx_next, y_alpha[a][slots], times[b] - times[a],
                    rate_p, y_prot[idx_prev, ns:],
                    x_rna[idx_prev], x_rna[idx_next],
                    modes[idx_prev] * c_p, modes[idx_next] * c_n, _rows(s1r, idx_prev),
                    d1_end=None if b_p is None else rate_n,
                    alpha_b=None if b_p is None else birth_alpha(y_alpha[a][slots], y_kon[idx_prev, ns:],
                                                                 y_kon[idx_next, ns:], self.alpha_threshold),
                    d0=_rows(d0_br, idx_next), d1_prod=_rows(d1p_br, idx_next), P1=y_prot[idx_next, ns:],
                    stim_end=y_prot[idx_next, :ns], kon_start=y_kon[idx_prev], kon_stop=y_kon[idx_next],
                    fractions=frac, weights=w_mem, kon_switch=kon_sw)

        # Real cell behind each trajectory state, and its cell type (None if unavailable)
        y_real = np.full(len(vect_t_sim), -1, dtype=int)
        cell_types = getattr(self, '_strata_labels', None)

        def traj_cell_types():
            if cell_types is None:
                return None
            return np.where(y_real >= 0, cell_types[np.maximum(y_real, 0)], '')

        def fit_targets():
            # Network targets (_network_targets); the trajectories keep the forcing modes (EMD basins)
            return self._network_targets(y_kon, y_real)

        def fit_theta(weight_fm, basal_start, inter_start, n_fits):
            # Network fit(s) on (time, sample, cell type)-stratified subsamples of the trajectories
            def fit_on_spec(spec, b0, i0):
                def fit_on(sels):
                    return inference_network_multi(
                        sels, y_samples, fit_targets(), y_prot, y_prot_prev,
                        ks, n_stimuli=ns, samples_id=samples_id,
                        ref_network=self.ref_network, basal_init=b0, inter_init=i0,
                        basal_ref=basal_ref, inter_ref=inter_ref,
                        weight_flow_matching=weight_fm,
                        final=0, constrain_basal_uniform=self.constrain_basal_uniform,
                        hard_forcing_ref=hard_forcing_ref, ref_constraint_pct=ref_constraint_pct,
                        seuil_min_network_intensity=self.seuil_min_network_intensity,
                        sample_conditions=self.sample_conditions, condition_pen=self.network_condition_pen,
                        gene_mask=self.gene_mask, sigma_spec_grn=spec, y_prot_mem=y_prot_mem,
                        dist_targets=self._distribution_targets(y_kon, y_real, y_samples, vect_t_sim, ks, valid),
                        **self._penalty_kw())
                return fit_on
            n_params = self._n_params_per_target(prev_cols, n_samples_local)
            if not (model_learning and ml_start <= n_iter < ml_freeze):
                return self._fit_theta_averaged(fit_on_spec(self._spec_grn(), basal_start, inter_start), vect_t_sim,
                                                y_samples, traj_cell_types(), n_fits, n_params, valid=valid)
            # Model learning: every candidate family on the same sub-samples, scored against the mixture alone
            sels = self._network_subsamples(vect_t_sim, y_samples, traj_cell_types(), n_fits, n_params, valid=valid)
            return self._select_sigma_families(fit_on_spec, sels, basal_start, inter_start, y_prot, y_samples, y_real,
                                               tau_mixture, ks, sample_idx, n_iter, verb=verb, mem=y_prot_mem)

        # Stratum of each real cell, for balanced OT batches; real cells at t0 per sample
        strata = self._traj_strata(data_rna, vect_t, vect_samples_id, times, samples_id)
        t0_real = [np.flatnonzero((vect_t == times[first_t[s]]) & (vect_samples_id == sample))
                   for s, sample in enumerate(samples_id)]

        # Rates (and network) refined in the loop (training only), with a rappel towards the literature
        deg_fit = self.refine_in_loop and compute_theta and context is None and alpha_pool is None and len(times) > 1
        last_deg, self._n_refine_in_loop = None, 0
        self.grid_network_precision = None  # precision of the network prior on the levels (more than two levels)
        if deg_fit and self.d_literature is None:
            self.d_literature = self.d.copy()

        # Soft targets (one_hot_probas = False): expected modes under the mixture posteriors, then updated by the network
        # (test set: those of the calibrated assignment, set by _assign_calibrated_basins)
        if context is None:
            self.modes_soft, self.posteriors_soft = None, None
        if context is None and not self.one_hot_probas and self.proba_init is not None:
            self.posteriors_soft = self.proba_init / np.maximum(self.proba_init.sum(axis=2, keepdims=True), EPS)
            self.modes_soft = self.modes.copy()
            self.modes_soft[:, ns:] = self._posterior_modes(self.proba_init, ks, vect_samples_id, samples_id)[:, ns:]
        # Targets ramped from the EMD basins only when the network is fitted (training)
        ramp_targets = self.modes_soft is not None and compute_theta and context is None and len(times) > 1

        # Model learning of the GRN sigma (training): logistic until the selection window, scored against the sigmoid
        # targets of the mixture alone (expected modes under proba_init), not the posteriors updated by the network
        # A single candidate family is forced from the first iteration, without selection
        model_learning = False
        ml_start, ml_freeze, tau_mixture = 0, 0, None
        if self.model_learning_grn and compute_theta and context is None:
            self._reset_model_learning(G_tot)
            model_learning = len(self._model_learning_candidates()) > 1 and len(times) > 1
        if model_learning:
            ml_start, ml_freeze = self._model_learning_window(min_n_loops)
            kb_cells = sigmoid_amplitudes(ks)
            kb_cells = kb_cells[np.searchsorted(np.asarray(samples_id), vect_samples_id)] if kb_cells.ndim == 3 \
                else kb_cells[None]
            tau_mixture = sigmoid_target(self._posterior_modes(self.proba_init, ks, vect_samples_id, samples_id),
                                         kb_cells[..., 0], kb_cells[..., 1])

        # === Main loop ===
        while True:

            weight_fm = self.weight_flow_matching * min(1, (n_iter-1)/min_n_loops) # Flow matching from second iteration and small at early ones
            # Network targets (one_hot_probas = False) from the EMD basins at the first iteration to the expected modes
            # at min_n_loops: the EM started from the classification EM, not from the flat network of the mixture alone
            self._target_mix = max(0.0, 1.0 - (n_iter - 1) / max(min_n_loops - 1, 1)) if ramp_targets else 0.0

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

            if alpha_pool is not None:
                # Single pass: trajectories and alphas are final, only kon_theta of the network remains
                kon_vector = y_kon.copy()
                kon_vector[:, ns:] = self._kon_ref_per_sample(y_prot, ks, inter, basal, samples_id=sample_idx, samples_data=y_samples)[:, ns:]
                n_iter += 1
                break

            # --- Evaluate error before and after inference (observed states) ---
            # Stopping error = network term of the OT: cross-entropy against the same targets (those of the network fit:
            # expected modes in soft, else the modes), weighted by gene_weights
            kon_err = fit_targets()[valid]
            mem_err = None
            if y_prot_mem is not None:  # nodes of the mRNA memory along the current trajectories
                refresh_prev_prot(y_prot_mem, memory=True)
                mem_err = y_prot_mem.select_cells(valid)
            # Flow-matching nodes of the iteration (same nodes for the errors before / after the fit and for the fit)
            fit_net = compute_theta and len(times) > 1
            if fit_net and weight_fm > 0:
                refresh_prev_prot()
            fm_err = (y_prot_prev.select_cells(valid), weight_fm) if fit_net and weight_fm > 0 else None
            # Error minimised by the network fit: (1 - weight_fm) end-state term + weight_fm flow-matching term
            dist_err = self._distribution_targets(y_kon, y_real, y_samples, vect_t_sim, ks, valid)
            dist_err = None if dist_err is None else (dist_err[0][valid], dist_err[1][valid], dist_err[2][valid],
                                                      dist_err[3])
            err_kw = dict(samples_id=sample_idx, samples_data=y_samples[valid], mem=mem_err, fm=fm_err, dist=dist_err)
            error = self._count_errors_per_sample(y_prot[valid], kon_err, ks, inter, basal, **err_kw)
            if fit_net:
                # Mean of fits on independent stratified subsamples covering the trajectory states
                basal, inter, basal_tmp, inter_tmp = fit_theta(weight_fm, basal, inter, self.n_network_fits)

            error_2 = self._count_errors_per_sample(y_prot[valid], kon_err, ks, inter, basal, **err_kw)
            errors.append(error_2)
            self.loss_trajectory.append(error_2)
            self.theta_trajectory.append(inter_tmp)

            if verb:
                kap = getattr(self, 'grid_network_precision', None)
                kap_log = f" | network precision on the levels (median): {np.median(kap):.1f}" \
                    if kap is not None and len(kap) and np.any(kap > 1) else ""
                print(f"{n_iter}", f"{count_end} | Errors (before, after): {error:.5f}, {error_2:.5f} | alpha mean: {np.mean(y_alpha[0]):.4f}{kap_log}")
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
                            rna_to_x=self.scale_proteins / self.fact_simple, spec=self._spec_grn(),
                            d0_mem=self._switch_d0(y_samples[r1])
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
            if y_prot_mem is not None:
                # kon on the mRNA memory of each target, mean over the trajectory states of each real cell
                refresh_prev_prot(y_prot_mem, memory=True)
                rr = np.flatnonzero(valid & (y_real >= 0))
                sig_b = self._sigma_bar(y_prot[rr], y_prot_mem.select_cells(rr), ks, inter, basal, sample_idx,
                                        y_samples[rr])
                cnt = np.bincount(y_real[rr], minlength=len(vect_t))
                reached = cnt > 0
                sig_c = np.stack([np.bincount(y_real[rr], weights=sig_b[:, g], minlength=len(vect_t))
                                  for g in range(G_tot)], axis=1)[reached] / cnt[reached][:, None]
                kb_c = sigmoid_amplitudes(ks)
                kb_c = kb_c[np.searchsorted(np.asarray(samples_id), vect_samples_id[reached])] if ks.ndim == 3 \
                    else kb_c[None]
                kon_vector_formodes[reached, ns:] = (kb_c[..., 0] + (kb_c[..., 1] - kb_c[..., 0]) * sig_c)[:, ns:]
            print("number of non reached cells", np.sum(to_keep_for_update == 0))

            # --- Update modes ---
            n_cells = self.proba_init.shape[0]
            # The weight of the network increases slowly because it aims to get the right attribution given probabilities that are close
            weight_prob = max(.95**(n_iter - 1), .1)

            # Mode amplitudes of each real cell's sample (per-sample mixtures), and their max
            ks_cells = (ks[np.searchsorted(np.asarray(samples_id), vect_samples_id)]
                        if ks.ndim == 3 else None)
            ks_max = ks.max(axis=0) if ks.ndim == 3 else ks
            # Genes of each real cell's condition (select_genes_per_condition): the others keep their single mode
            act_cells = self.cell_gene_mask(np.searchsorted(np.asarray(samples_id), vect_samples_id))

            def assign_basins(mu, nu, dist, dist_soft):
                # Target masses nu: EMD basins (forcing), and the posteriors updated by the network (one_hot_probas = False)
                return self._assign_from_costs(mu, nu, dist, dist_soft)

            def basin_costs(proba, keep, kappa, kl, wg):
                # EMD basins: network term ramped (weight_prob) and weighted per gene, as a label set by the network
                # would lock in; soft posteriors: network term at full weight (E-step), so that the cells whose counts
                # do not discriminate the modes follow the network instead of the mean of their time
                log_p = -np.log(proba)
                dist = np.clip(log_p + (1 - weight_prob) * keep * kappa * wg * kl, 0, 100)
                soft = None if self.one_hot_probas else np.clip(log_p + keep * kappa * kl, 0, 100)
                return dist, soft
        
            def run_main_loop_for_gene(g, temporal=self.temporal_basins):
                l_max = 1 + np.argmax(ks_max[g, :])
                obj = np.zeros((n_cells, l_max), dtype=float)
                obj[:, :] = ks_cells[:, g, :l_max] if ks_cells is not None else ks[g, :l_max][None, :]
                # Distance network / mode on kon: weakly separated genes keep a weak network term against the posteriors
                tmp_basins = np.zeros(n_cells, dtype=np.int8)
                tmp_modes = np.zeros_like(self.modes[:, g])
                act = np.ones(n_cells, bool) if act_cells is None else act_cells[:, g]
                tmp_modes[~act] = obj[~act, 0]
                tmp_soft = tmp_modes.copy()  # expected mode under the updated posteriors (one_hot_probas = False)
                tmp_post = np.zeros((n_cells, l_max))  # updated posteriors (one_hot_probas = False)
                tmp_post[~act, 0] = 1.0
                # Precision kappa of the network prior on the levels (more than two levels): exp(-kappa KL(tau_l || sigma))
                # has a width ~ sigma (1 - sigma) / kappa, flat for kappa = 1 (the Bernoulli prior of two modes) while a
                # cell predicted at sigma lies at the level sigma; kappa by moment matching on the reached cells, mean
                # sigma (1 - sigma) / mean E_q[(level - sigma)^2] (EM update of the precision of the prior: second
                # moment under the posteriors q, one_hot_probas = False; (basin level - sigma)^2 otherwise, CEM)
                kappa = 1.0
                sel = act & (to_keep_for_update > 0)
                if l_max > 2 and sel.sum() > 1:
                    lo, hi = obj[sel, 0], obj[sel].max(axis=1)
                    sig = sigmoid_target(kon_vector_formodes[sel, g], lo, hi)
                    if self.posteriors_soft is None:
                        dev2 = (sigmoid_target(self.modes[sel, g], lo, hi) - sig) ** 2
                    else:
                        q = self.posteriors_soft[sel, g, :l_max]
                        tau_l = sigmoid_target(obj[sel], lo[:, None], hi[:, None])
                        dev2 = (q * (tau_l - sig[:, None]) ** 2).sum(axis=1) / np.maximum(q.sum(axis=1), EPS)
                    kappa = float(np.clip(np.mean(sig * (1 - sig)) / max(np.mean(dev2), EPS), 1.0,
                                          self.grid_network_precision_max))
                if temporal:
                    for t_i in times:
                        indices = (vect_t == t_i) & act
                        if not indices.any():
                            continue
                        tmp_modes_i = np.zeros_like(self.modes[indices, g])
                        proba_i = self.proba_init[indices, g, :l_max].copy()
                        nu = self._target_masses(proba_i, self.pi_init[g - ns][t_i])
                        proba_i /= np.max(proba_i, axis=1, keepdims=True)
                        obj_i = obj[indices].copy()
                        n_cells_i = np.sum(indices)
                        mu = np.ones(n_cells_i)/n_cells_i
                        kl, wg = self._mode_network_cost(kon_vector_formodes[indices, g], obj_i)
                        dist, dist_soft = basin_costs(proba_i, to_keep_for_update[indices, None], kappa, kl, wg)
                        idx, probs = assign_basins(mu, nu, dist, dist_soft)
                        for cell in range(n_cells_i):
                            tmp_modes_i[cell] = obj_i[cell, idx[cell]]
                        tmp_basins[indices] = idx
                        tmp_modes[indices] = tmp_modes_i[:]
                        tmp_soft[indices] = tmp_modes_i if probs is None else (probs * obj_i).sum(axis=1)
                        if probs is not None:
                            tmp_post[indices] = probs
                elif act.any():
                    proba = self.proba_init[act, g, :l_max].copy()
                    n_act = int(act.sum())
                    nu = self._target_masses(proba, np.sum([np.asarray(self.pi_init[g - ns][t_i])[:l_max] *
                                                            np.sum((vect_t == t_i) & act) / n_act for t_i in times],
                                                           axis=0))
                    proba /= np.max(proba, axis=1, keepdims=True)
                    mu = np.ones(n_act) / n_act
                    kl, wg = self._mode_network_cost(kon_vector_formodes[act, g], obj[act])
                    dist, dist_soft = basin_costs(proba, to_keep_for_update[act, None], kappa, kl, wg)
                    idx, probs = assign_basins(mu, nu, dist, dist_soft)
                    tmp_modes[act] = obj[act][np.arange(n_act), idx]
                    tmp_basins[act] = idx
                    tmp_soft[act] = tmp_modes[act] if probs is None else (probs * obj[act]).sum(axis=1)
                    if probs is not None:
                        tmp_post[act] = probs
                return tmp_basins, tmp_modes, tmp_soft, tmp_post, kappa
            results = Parallel(n_jobs=-1)(
                delayed(seeded_call)(task_seed(self.seed, 8, n_iter, g), run_main_loop_for_gene, g)
                for g in range(ns, G_tot))

            # Forcing modes: the EMD basins (right masses and ranks, deterministic); network targets (one_hot_probas =
            # False): the expected modes under the posteriors updated by the network
            self.grid_network_precision = np.array([res[4] for res in results])  # kappa of each gene (diagnostic)
            for idx, g in enumerate(range(ns, G_tot)):
                tmp_basins, tmp_modes, tmp_soft, tmp_post, _ = results[idx]
                self.basins[:, g] = tmp_basins
                self.modes[:, g] = tmp_modes
                if self.modes_soft is None:
                    continue
                self.modes_soft[:, g] = tmp_soft
                self.posteriors_soft[:, g, :] = 0.0
                self.posteriors_soft[:, g, :tmp_post.shape[1]] = tmp_post

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
                    self._set_mode_soft(ko_cells, g, 0)
                    self._set_mode_soft(ov_cells, g, l_max - 1)

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
        # Prior net rate of each state (rate of its real cell), same layout as R_opt (use_recomputed_proliferation)
        r_prior = getattr(self, '_prolif_net_rate', None)
        self.R_prior = None
        if r_prior is not None and len(times) > 1:
            Rp = np.where(y_real >= 0, np.asarray(r_prior, dtype=float)[np.maximum(y_real, 0)], np.nan)
            Rp2 = Rp.reshape(len(times), N_tot)
            for a, b, slots in pairs:
                Rp2[a + 1:b][:, slots] = Rp2[a, slots][None, :]
            Rp[np.isnan(R_opt_traj)] = np.nan
            Rp[np.isfinite(R_opt_traj) & np.isnan(Rp)] = 0.0
            self.R_prior = Rp
        self._target_mix = 0.0  # later steps (final refit, free-running fit) use the expected modes
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
        if self.targets_accounting_late_mrna and len(times) > 1:
            # kon read on the mRNA (reports, NB draws): single lagged node of the class of each target (as the OT, the
            # free-running fit and the simulations)
            lag_path = PathProt.zeros(len(vect_t_sim), self._mem_node_count(G_tot, 1), G_tot)
            refresh_prev_prot(lag_path, memory=True, n_mem=1)
            kon_vector = kon_vector.copy()
            kon_vector[:, ns:] = self._kon_read(y_prot, lag_path, ks, inter, basal, sample_idx, y_samples)[:, ns:]
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


    def _distribution_targets(self, y_kon, y_real, y_samples, vect_t_sim, ks, valid=None):
        """Targets of the distribution term (weight_network_distribution, one_hot_probas = False) of the trajectory
        states: (posteriors (N, G, L), sigmoid targets of the levels (N, G, L) of their real cell, group (N,) =
        sample x time of each state, -1 without real cell or for a virtual state, weight); None if inactive."""
        if not self.weight_network_distribution or self.one_hot_probas or y_real is None:
            return None
        out = self._posterior_levels(y_kon, y_real, y_samples, ks)
        if out is None:
            return None
        P, lev = out
        kb = sigmoid_amplitudes(ks)
        kb = kb[np.asarray(y_samples, dtype=int)] if np.ndim(ks) == 3 else kb[None]
        tau = sigmoid_target(lev, kb[..., :1], kb[..., 1:])
        _, ti = np.unique(vect_t_sim, return_inverse=True)
        keep = np.asarray(y_real) >= 0
        if valid is not None:
            keep &= np.asarray(valid, dtype=bool)
        grp = np.where(keep, np.asarray(y_samples, dtype=int) * (ti.max() + 1) + ti, -1)
        return P, tau, grp, float(self.weight_network_distribution)

    def _distribution_error(self, y_prot, mem, ks, inter, basal, samples_id, samples_data, dist):
        """Distribution term of the stopping error: per cell, sum over the genes of more than two levels (gene
        weights of the cross-entropy, normalised per cell) of the term of the network fit (dist_term)."""
        ns = self.n_stimuli
        sd = self.samples_data if samples_data is None else samples_data
        sid = np.sort(np.unique(sd)) if samples_id is None else np.asarray(samples_id)
        sig = self._sigma_bar(y_prot, mem, ks, inter, basal, sid, sd)
        gw = gene_weights(ks)
        gw = gw[np.searchsorted(sid, sd)] if np.ndim(ks) == 3 else np.broadcast_to(gw[None], (len(sig), gw.shape[-1]))
        wn = gw[:, ns:] / np.maximum(gw[:, ns:].sum(axis=1, keepdims=True), EPS)
        post, tau, grp, lam = dist
        err = 0.0
        for j, g in enumerate(range(ns, sig.shape[1])):
            d = dist_term(grp, post[:, g], tau[:, g], lam)
            if d is not None:
                err += _dist_loss_grad(sig[:, g], d)[0] * np.mean(wn[:, j])
        return err / max(len(sig), 1)

    def _network_targets(self, y_kon, y_real):
        """Targets of the network fit and of the free-running fit: kon of the modes of the states, or (one_hot_probas =
        False) their expectation under the posteriors updated by the network, of the real cell y_real of each state
        (k_low + (k_high - k_low) E[sigma]); y_kon where unavailable."""
        if self.one_hot_probas or self.modes_soft is None or y_real is None or len(y_real) != len(y_kon):
            return y_kon
        ns = self.n_stimuli
        y_real = np.asarray(y_real, dtype=int)
        yk = np.array(y_kon, copy=True)
        r = y_real >= 0
        yk[r, ns:] = self._soft_target_modes(y_real[r])[:, ns:]
        return yk

    def _soft_target_modes(self, rows):
        """Network targets (kon) of the real cells rows with one_hot_probas = False: expected modes under the
        posteriors updated by the network, started from the EMD basins (classification EM), _target_mix x basin +
        (1 - _target_mix) x expected mode."""
        lam = self._target_mix
        if lam <= 0:
            return self.modes_soft[rows]
        return lam * self.modes[rows] + (1 - lam) * self.modes_soft[rows]

    def _soft_targets_missing(self, y_real, where):
        """True (with a warning) when one_hot_probas = False but the posteriors updated by the network are not
        available for the states y_real (e.g. a model reloaded without soft_modes.npz)."""
        ok = self.one_hot_probas or (self.posteriors_soft is not None and y_real is not None)
        if not ok:
            warnings.warn(f"{where}: one_hot_probas = False but no posteriors updated by the network "
                          "(soft_modes.npz): targets = forcing modes kon_beta")
        return not ok

    def _posterior_levels(self, y_kon, y_real, rows_samples, ks):
        """Posteriors (N, G, L) and levels (N, G, L) (normalised kon) of the real cell of each state, for targets
        drawn from them (one_hot_probas = False); states without real cell, and the stimuli, at their y_kon.
        rows_samples: sample index of each state. None without posteriors."""
        if self.one_hot_probas or self.posteriors_soft is None or y_real is None or len(y_real) != len(y_kon):
            return None
        ns = self.n_stimuli
        y_real = np.asarray(y_real, dtype=int)
        ks = np.asarray(ks, dtype=float)
        L = min(self.posteriors_soft.shape[2], ks.shape[-1])
        N, G = np.shape(y_kon)
        P = np.zeros((N, G, L))
        P[..., 0] = 1.0
        lev = np.repeat(np.asarray(y_kon, dtype=float)[..., None], L, axis=2)
        r = y_real >= 0
        Pr = self.posteriors_soft[y_real[r], ns:, :L]
        P[r, ns:] = Pr / np.maximum(Pr.sum(axis=2, keepdims=True), EPS)
        lev[r, ns:] = (ks[np.asarray(rows_samples, dtype=int)[r]] if ks.ndim == 3 else ks[None])[:, ns:, :L]
        return P, lev

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
            samples_rows=y_samples, kon_targets=y_kon, real_idx=y_real, rna=y_rna, prolif=None, temporal=False, learn_scale=False,
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
        self._enforce_soft_basins(verb=verb)
        self._floor_mrna_degradation(verb=verb)
        seed_everything(self.seed)

        # --- Initialization ---
        data_rna = self._parse_input(data, time_key, scale_depth=True)  # counts at the reference depth
        # Raw counts and depth factors of the real cells: NB likelihood of the score of the model learning
        self._nb_counts, self._nb_depth = self._parse_input(data, time_key), self._depth_factors(data)
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
            

    def _fm_fractions(self, alpha, kon_start, kon_stop, Q):
        """
        Flow-matching nodes (n, Q) of the intervals (time fractions), around the switch of the state, a (n,) = mean of
        the switching fractions alpha (n, G_genes) of the genes weighted by the size of their change of mode (as
        birth_alpha; 1/2 if none changes): the first node just after it, a + 0.1 (at a when a >= 1/2), then the others
        equally spaced on [0, a] and on [a, 1], alternately from [0, a] (2: middle of [0, a]; 3: and of [a, 1]; 4: [0, a]
        at 1/3 and 2/3; ...): deterministic, without the variance of uniform draws.
        """
        ns, tol = self.n_stimuli, self.alpha_threshold
        alpha = np.asarray(alpha, dtype=float)
        w = np.minimum((np.abs(np.asarray(kon_stop)[:, ns:] - np.asarray(kon_start)[:, ns:]) / max(tol, EPS))
                       ** (1 / tol), 1.0)
        a = np.where(w.sum(axis=1) > 0, (w * alpha).sum(axis=1) / np.maximum(w.sum(axis=1), EPS), 0.5)
        f = np.empty((len(a), Q))
        f[:, 0] = a + 0.1 * (a < 0.5)
        n_a, n_b = (Q - 1 + 1) // 2, (Q - 1) // 2
        for k in range(1, n_a + 1):
            f[:, k] = a * k / (n_a + 1)
        for k in range(1, n_b + 1):
            f[:, n_a + k] = a + (1 - a) * k / (n_b + 1)
        return f

    def _fill_path_prot(self, path, idx_next, alpha, delta_t, d1, P0, M0, M1, mode_init, mode_end, s, d1_end=None,
                        alpha_b=None, d0=None, P1=None, d1_prod=None, stim_end=None, kon_start=None, kon_stop=None,
                        fractions=None, weights=None, kon_switch=None):
        """
        Flow matching: path.nodes.shape[1] nodes in each interval ending at rows idx_next (_fm_fractions) and fill the
        protein states of every gene there (each gene switching at its own alpha, stimuli at their end value
        stim_end), the kon of each target in force at each node (kon_start, mode of the start state, before its
        switch; kon_stop, that of the end state, after it), and equal weights 1 / Q on the nodes (0 on the end
        state). Arrays P0..mode_end and alpha are (N, G_genes); d1 / d1_end: relaxation rates before / after the
        switch of the birth rate at alpha_b (birth_alpha), d1_end=None: no dilution. d0 given: mRNA bridges
        (prot_bridge, production d1_prod) conditioned on the end proteins P1, M0 / M1 being the mRNA in kon units
        and mode_init / mode_end the kon levels; else the flow of find_next_prot (prot_along).
        fractions (n, Q) and weights (n, Q + 1, G) given (mRNA memory, _memory_nodes): nodes at these time fractions,
        with these weights, instead of _fm_fractions and equal weights; fractions (n, Q, G_genes): per gene.
        kon_switch (n, G_genes): fractions at which kon switches, before alpha when the forcing is the mRNA level
        (kon_switch_fraction, targets_accounting_late_mrna); None: at alpha.
        """
        ns = self.n_stimuli
        Qn = path.nodes.shape[1]
        rows = np.asarray(idx_next)
        f = self._fm_fractions(alpha, kon_start, kon_stop, Qn) if fractions is None else fractions  # time fractions
        for q in range(Qn):
            tau = delta_t * (f[:, q:q + 1] if f.ndim == 2 else f[:, q])
            if d0 is not None:
                P = prot_bridge(self.d[1, ns:] if d1_prod is None else d1_prod, d0, d1, d1 if d1_end is None else d1_end,
                                P0, M0, M1, mode_init, mode_end, alpha, alpha if alpha_b is None else alpha_b,
                                delta_t, tau, P1)
            else:
                P = prot_along(d1, P0, mode_init, mode_end, alpha, s, delta_t, tau, d1_end=d1_end, alpha_b=alpha_b)
            path.nodes[rows, q, :ns] = stim_end
            path.nodes[rows, q, ns:] = P
        sw = np.asarray(alpha if kon_switch is None else kon_switch, dtype=float)
        after = (f[:, :, None] if f.ndim == 2 else f) >= sw[:, None, :]  # kon switched
        path.kon[rows, :, ns:] = np.where(after, np.asarray(kon_stop)[:, None, ns:], np.asarray(kon_start)[:, None, ns:])
        if weights is not None:
            path.weights[rows] = weights
            return
        path.weights[rows, 0] = 0.0
        path.weights[rows, 1:] = 1.0 / Qn

    def estimate_trajectories(self, y_prot, times, d1, N=100, kon_beta=None, s=None, birth=None, lag_d0=None):
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
        birth : array of shape (T*N,), optional
            Birth rate of each state (dilution: rate d1 + b, mode targets times d1 / (d1 + b)); None: no dilution.
        d1 : (G_genes,), or (T-1, G_genes) per interval (mean over the intervals of each pair of observed times),
            or (T-1, S, G_genes) per sample (kinetics per network condition, routed by samples_data).
        lag_d0 : (T*N, G_tot) mRNA rates or None: also return the single lagged nodes of the mRNA memory of each state
            along the same flow (_memory_nodes, one per class of targets; the state itself without previous interval),
            (proteins, PathProt of these nodes) for _kon_read.
        The forcing is kon (the modes): the alpha of an mRNA-level forcing of fit_network (_switch_d0) is brought back
        to the switch of kon (kon_switch_fraction).
        """
        if kon_beta is None:
            kon_beta = self.kon_beta
        if s is None:
            s = self.scale_proteins
        ns = self.n_stimuli
        prot_modified = y_prot.copy()
        G_tot = y_prot.shape[1]
        mem = None if lag_d0 is None else PathProt.zeros(len(y_prot), self._mem_node_count(G_tot, 1), G_tot)
        sd = getattr(self, 'samples_data', None)
        d0_sw = self._switch_d0(sd) if sd is not None and len(sd) == len(y_prot) else None

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
            if d0_sw is not None:  # kon forcing: switch of kon, before that of the mRNA-level forcing
                alpha_ab = kon_switch_fraction(alpha_ab, d0_sw[rows_next], float(delta_t))
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
            if mem is not None:  # single lagged nodes of the mRNA memory, along the same flow
                frac, w = self._memory_nodes(np.asarray(lag_d0)[rows_next], delta_t, n_nodes=1)
                for q in range(frac.shape[1]):
                    mem.nodes[rows_next, q, :ns] = prot_modified[rows_next, :ns]
                    mem.nodes[rows_next, q, ns:] = prot_along(rate, P0, kb_prev * c, kb_next * c_n, alpha_ab, s,
                                                              delta_t, delta_t * frac[:, q:q + 1], d1_end=rate_n,
                                                              alpha_b=alpha_b)
                mem.weights[rows_next] = w
            for k in range(a + 1, b):  # timepoints missed inside the interval: along the same flow
                prot_modified[N * k + slots, ns:] = prot_along(
                    rate, P0, kb_prev * c, kb_next * c_n, alpha_ab, s, delta_t, times[k] - times[a], d1_end=rate_n,
                    alpha_b=alpha_b)

        if not valid.all():
            fill_virtual([prot_modified] + ([] if mem is None else [mem.nodes, mem.weights]), valid, len(times),
                         inside=False)

        return prot_modified if mem is None else (prot_modified, mem)
    

    def _kon_ref_per_sample(self, y_prot, ks, inter, basal, samples_id=None, samples_data=None, sigma_spec=None):
        """
        Compute kon_ref_vector respecting per-sample basal when basal is 2-D.

        When basal is 1-D (G,), falls back to a single kon_ref call.
        When basal is 2-D (n_samples, G), loops over samples and
        assembles the result using samples_data (or self.samples_data) as the
        routing key so that each cell uses its own sample's basal. sigma_spec: families of sigma (None: those of the
        model, _spec_grn).
        """
        spec = self._spec_grn() if sigma_spec is None else sigma_spec
        if basal.ndim < 2 and ks.ndim < 3 and inter.ndim < 3:
            return kon_ref_vector(y_prot, ks, inter, basal, spec)
        sd = samples_data if samples_data is not None else self.samples_data
        if samples_id is None:
            samples_id = np.sort(np.unique(sd))
        out = np.zeros((y_prot.shape[0], y_prot.shape[1]))
        for s_idx, s in enumerate(samples_id):
            mask = (sd == s)
            if not np.any(mask):
                continue
            basal_s = basal[min(s_idx, basal.shape[0] - 1)] if basal.ndim == 2 else basal
            out[mask] = kon_ref_vector(y_prot[mask], ks_of(ks, s_idx), inter_of(inter, s_idx), basal_s, spec)
        return out

    def _mrna_d0(self, samples):
        """mRNA degradation rate d0 (n, G_tot) of each row (sample indices samples; rates of each sample if any), at
        least min_ratio x d1 of the same rates (as the bridges, the refine and the simulations)."""
        rows = np.asarray(samples).astype(int)
        if self.d_samples is not None:
            d = np.stack([np.asarray(d_s, dtype=float) for d_s in self.d_samples])[np.minimum(rows, len(self.d_samples) - 1)]
            return np.maximum(d[:, 0], self.min_ratio * d[:, 1])
        d = np.asarray(self.d, dtype=float)
        return np.broadcast_to(np.maximum(d[0], self.min_ratio * d[1]), (len(rows), d.shape[-1]))

    def _switch_d0(self, samples):
        """mRNA rates (n, G_genes) of the delay of the switch of the forcing (inference_alpha) with
        targets_accounting_late_mrna when the forcing is the mRNA level (adaptive shrinkage of the observed mRNA:
        inference_mrnas_direct without mrna_bridge); None when it is kon (bridges, modes), which switches when the
        network commits, the mRNA delay being in its dynamics (bridge d0, d1_eff)."""
        if not (self.targets_accounting_late_mrna and self.inference_mrnas_direct and not self.mrna_bridge):
            return None
        return np.ascontiguousarray(self._mrna_d0(samples)[:, self.n_stimuli:])

    def _memory_filter(self):
        """Modes of the free-running fit read on the mRNA filter (sde_memory_filter); always with explicit mRNA
        without targets_accounting_late_mrna, where an instantaneous reading would contradict the simulated mRNA."""
        return bool(self.sde_memory_filter or (self.simulation_mrnas_direct and not self.targets_accounting_late_mrna))

    def _mem_node_count(self, G, n_nodes=None):
        """Number of nodes of the mRNA memory (_memory_nodes): late_mrna_nodes (or n_nodes) if >= 2, else the number
        of classes of the single lagged nodes."""
        Q = int(self.late_mrna_nodes if n_nodes is None else n_nodes)
        return Q if Q >= 2 else max(1, min(int(self.late_mrna_classes), G - self.n_stimuli))

    def _lag_classes(self, d0, delta_t):
        """Single lagged nodes of the mRNA memory per class of targets (late_mrna_classes classes of equal sizes by
        d0, rates d0 (n, G)): lag (n, K) hours of each class, mean over its genes of (1 - exp(-d0 delta_t)) / d0, the
        mean of the memory kernel truncated at the start of the interval (its mass before on the start state), and
        one-hot weights (n, K, G) of the class of each gene (stimuli in the first one)."""
        ns = self.n_stimuli
        d0 = np.maximum(np.asarray(d0, dtype=float), 1e-12)
        n, G = d0.shape
        K = self._mem_node_count(G, 1)
        lag_g = -np.expm1(-d0 * float(delta_t)) / d0
        order = np.argsort(d0[:, ns:], axis=1, kind='stable')
        w = np.zeros((n, K, G))
        w[:, 0, :ns] = 1.0
        rows = np.arange(n)[:, None]
        for k, part in enumerate(np.array_split(np.arange(G - ns), K)):
            w[rows, k, ns + order[:, part]] = 1.0
        lag = (w[..., ns:] * lag_g[:, None, ns:]).sum(axis=2) / np.maximum(w[..., ns:].sum(axis=2), 1.0)
        return lag, w

    def _memory_nodes(self, d0, delta_t, n_nodes=None):
        """Nodes of the mRNA memory of the intervals of length delta_t ending at rows with rates d0 (n, G): time
        fractions (n, Q) of the late_mrna_nodes nodes over the last H = min(delta_t, 5 / min d0) hours, and weights
        (n, Q + 1, G) of the end state (column 0) and of the nodes, trapezoidal rule of d0 exp(-d0 h), the mass before
        the window on the earliest node (sum 1 per gene). late_mrna_nodes = 1: the single lagged node of the class of
        each target (_lag_classes), of weight 1 for its genes. n_nodes: number of nodes instead of late_mrna_nodes."""
        Q = int(self.late_mrna_nodes if n_nodes is None else n_nodes)
        d0 = np.maximum(np.asarray(d0, dtype=float), 1e-12)
        if Q == 1:
            lag, w_cls = self._lag_classes(d0, delta_t)
            w = np.zeros((len(d0), w_cls.shape[1] + 1, d0.shape[1]))
            w[:, 1:] = w_cls
            return 1.0 - lag / float(delta_t), w
        H = np.minimum(float(delta_t), 5.0 / d0[:, self.n_stimuli:].min(axis=1))    # (n,)
        h = H[:, None] * np.arange(Q + 1)[None, :] / Q                              # (n, Q + 1), h_0 = 0 = end
        dh = (H / Q)[:, None, None]
        w = d0[:, None, :] * np.exp(-d0[:, None, :] * h[:, :, None]) * dh           # (n, Q + 1, G)
        w[:, 0] *= 0.5
        w[:, -1] *= 0.5
        w[:, -1] += np.exp(-d0 * H[:, None])
        w /= w.sum(axis=1, keepdims=True)
        return 1.0 - h[:, 1:] / float(delta_t), w

    def _memory_ot_kw(self, s_idx, delta_t):
        """Nodes and weights of the mRNA memory (targets_accounting_late_mrna) of an interval of the trajectory OT
        (my_otdistance). The OT evaluates sigma for every pair of states: always the single lagged nodes of the classes
        of targets (late_mrna_nodes = 1), the average on the memory being kept for the network fit, the stopping error,
        the basin updates and the score of the model learning."""
        kw, d0 = {}, self._mrna_d0(np.array([s_idx]))
        if self.targets_accounting_late_mrna:
            frac, w = self._memory_nodes(d0, delta_t, n_nodes=1)
            kw.update(mem_frac=np.ascontiguousarray(frac[0]), mem_w=np.ascontiguousarray(w[0][:, self.n_stimuli:]))
        return kw

    def _fm_error(self, path, y_prot, kon_beta, ks, inter, basal, samples_id=None, samples_data=None):
        """Flow-matching error of the rows, as the flow-matching term of the network fit: sum over the end state
        (weight path.weights[:, 0]) and the nodes (weights path.weights[:, q + 1]) of the cross-entropy between sigma at
        their states and the sigmoid target of the kon in force there, weighted by gene_weights, mean over the rows."""
        ns = self.n_stimuli
        sd = self.samples_data if samples_data is None else samples_data
        sid = np.sort(np.unique(sd)) if samples_id is None else np.asarray(samples_id)
        kb, gw = sigmoid_amplitudes(ks), gene_weights(ks)
        pos = np.searchsorted(sid, sd)
        kb, gw = (kb[pos], gw[pos]) if np.ndim(ks) == 3 else (kb[None], gw[None])
        w_g = np.broadcast_to(gw[..., ns:], (len(y_prot), gw.shape[-1] - ns))
        w_g = w_g / np.maximum(w_g.sum(axis=1, keepdims=True), EPS)
        terms = [(path.weights[:, 0], y_prot, kon_beta)] + [(path.weights[:, q + 1], path.nodes[:, q], path.kon[:, q])
                                                           for q in range(path.nodes.shape[1])]
        total = np.zeros(len(y_prot))
        for w, Y, kon in terms:
            rows = np.flatnonzero(np.max(w, axis=1) > 0) if w.ndim == 2 else np.flatnonzero(w > 0)
            if not len(rows):
                continue
            sig = sigmoid_target(self._kon_ref_per_sample(Y[rows], ks, inter, basal, samples_id=sid,
                                                          samples_data=sd[rows]), kb[rows if len(kb) > 1 else [0], ..., 0],
                                 kb[rows if len(kb) > 1 else [0], ..., 1])
            tau = sigmoid_target(np.asarray(kon)[rows], kb[rows if len(kb) > 1 else [0], ..., 0],
                                 kb[rows if len(kb) > 1 else [0], ..., 1])
            ce = cross_entropy(np.ascontiguousarray(sig[:, ns:]).ravel(),
                               np.ascontiguousarray(tau[:, ns:]).ravel()).reshape(len(rows), -1)
            ww = w[rows] if w.ndim == 1 else w[rows][:, ns:]
            ww = ww[:, None] if ww.ndim == 1 else ww
            total[rows] += np.sum(ww * ce * w_g[rows], axis=1)
        return float(total.mean())

    def _count_errors_per_sample(self, y_prot, kon_beta, ks, inter, basal,
                                  samples_id=None, samples_data=None, mem=None, fm=None, dist=None):
        """
        Weighted-average count_errors respecting per-sample basal.
        When basal is 1-D, delegates to count_errors directly.

        samples_data : per-cell sample assignment; defaults to self.samples_data.
        mem : PathProt of the rows (targets_accounting_late_mrna): sigma averaged on the mRNA memory of each target.
        fm : (PathProt of the rows, weight_fm) (flow matching): (1 - weight_fm) times the end-state error + weight_fm
            times the flow-matching error (_fm_error), as the objective of the network fit.
        dist : distribution targets of the rows (_distribution_targets) or None: the distribution term of the network
            fit is added to the end-state error, per cell, with the gene weights of the cross-entropy.
        """
        if fm is not None and fm[1] > 0:
            end = self._count_errors_per_sample(y_prot, kon_beta, ks, inter, basal, samples_id, samples_data, mem,
                                                dist=dist)
            return (1 - fm[1]) * end + fm[1] * self._fm_error(fm[0], y_prot, kon_beta, ks, inter, basal, samples_id,
                                                                samples_data)
        if dist is not None:
            end = self._count_errors_per_sample(y_prot, kon_beta, ks, inter, basal, samples_id, samples_data, mem)
            return end + self._distribution_error(y_prot, mem, ks, inter, basal, samples_id, samples_data, dist)
        if mem is not None:
            ns = self.n_stimuli
            sd = self.samples_data if samples_data is None else samples_data
            sid = np.sort(np.unique(sd)) if samples_id is None else np.asarray(samples_id)
            sig = self._sigma_bar(y_prot, mem, ks, inter, basal, sid, sd)
            kb, gw = sigmoid_amplitudes(ks), gene_weights(ks)
            pos = np.searchsorted(sid, sd)
            kb, gw = (kb[pos], gw[pos]) if np.ndim(ks) == 3 else (kb[None], gw[None])
            tau = sigmoid_target(kon_beta, kb[..., 0], kb[..., 1])
            ce = cross_entropy(np.ascontiguousarray(sig[:, ns:]).ravel(),
                               np.ascontiguousarray(tau[:, ns:]).ravel()).reshape(len(sig), -1)
            w = np.broadcast_to(gw[..., ns:], ce.shape)
            return float(np.sum(ce * w / np.maximum(w.sum(axis=1, keepdims=True), EPS)) / max(len(sig), 1))
        if basal.ndim < 2 and ks.ndim < 3 and inter.ndim < 3:
            return count_errors(y_prot, kon_beta, ks, basal, inter,
                                n_stimuli=self.n_stimuli, spec=self._spec_grn())
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
                                 n_stimuli=self.n_stimuli, spec=self._spec_grn())
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
        - refine_theta_with_sde = False: network refitted (final = 1) on these proteins, then fixed.
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
            lag_d0 = self._mrna_d0(self.samples_data) if self.targets_accounting_late_mrna else None
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
                out = self.estimate_trajectories(
                    self.prot, times, d1_test[..., ns:], N=N_tot, kon_beta=self.kon_beta, s=self.scale_proteins,
                    birth=self._state_birth(), lag_d0=lag_d0)
                self.prot, mem = (out, None) if lag_d0 is None else out
            elif lag_d0 is not None:  # trajectories kept, read at the lagged nodes along the flows on the modes
                mem = self.estimate_trajectories(self.prot, times, d1_unit[ns:], N=N_tot, kon_beta=self.kon_beta,
                                                 s=self.scale_proteins, birth=self._state_birth(), lag_d0=lag_d0)[1]
            else:
                mem = None
            # kon read on the mRNA: single lagged node of the class of each target (as fit_network)
            kon_vector = self.kon_beta.copy()
            kon_vector[:, ns:] = self._kon_read(self.prot, mem, ks, inter, basal, np.sort(np.unique(self.samples_data)),
                                                self.samples_data)[:, ns:]
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
                                                s=self.scale_proteins, birth=self._state_birth()) \
                if unitary else self.prot
        else:
            # Network refitted (final = 1) on these proteins, then fixed in the free-running fit
            y_prot, basal, inter = self._refine_trajectories_network(
                self.kon_beta, d1_unit[ns:], times, N_tot, ks, valid, samples_id, verb=verb,
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

    def _birth_bounds(self):
        """(min, max) birth rate of the trajectory states (those consistent with R_opt, else the MLP on the
        trajectories): bounds of the birth head of the MLP on simulated states (no extrapolation run-away)."""
        ns = self.n_stimuli
        b = self._state_rates()[0]
        if b is None and self.prolif_network is not None and getattr(self, 'prot', None) is not None:
            b = self.prolif_network.predict_birth(self.prot[:, ns:], self.prot[:, :ns], samples=self.samples_data)
        b = None if b is None else np.asarray(b, dtype=float)
        return None if b is None or not np.isfinite(b).any() else (float(np.nanmin(b)), float(np.nanmax(b)))

    def _nearest_state_birth(self, X, K, B, V):
        """
        Birth rate of simulated states without the MLP: that of the nearest trajectory state at the start of the
        interval (mean of its rates at both ends), with match_distance on proteins and kon (as the re-assignment
        of alpha). X, K (T, n, G) trajectory proteins (units of the fit) and kon, B (T, n) birth rates, V (T, n)
        observed. Returns f(P, kon, t, slots).
        """
        ns = self.n_stimuli

        def birth(P, kon, t, sl):
            ok = np.flatnonzero(V[t]) if V[t].any() else np.arange(X.shape[1])
            dist = match_distance(np.asarray(P)[:, ns:], np.asarray(kon)[:, ns:], X[t, ok, ns:], K[t, ok, ns:])
            nn = ok[np.argmin(dist, axis=1)]
            return 0.5 * (B[t, nn] + B[t + 1, nn])
        return birth

    def _fit_free_running(self, y_prot, basal, inter, mode, d1_start, eps_start, valid, times, N_tot, ks, k1,
                          samples_id, b_states, verb=True, samples_rows=None, kon_targets=None, real_idx=None,
                          rna=None,
                          prolif='model', temporal=None, learn_scale=None, learn_theta=None, epochs=None,
                          per_condition=True, d1_ref=None, eps_ref=None, only_max_horizon=False, lam_rates=None,
                          tag='refine_network_degradations'):
        """
        d1, eps = d1 / d0 and the scale of theta per target gene (per interval with use_temporal_degradations) by
        the free-running fit of the simulation model (fit_free_running: 'ode', 'sde1' protein-only PDMP, 'sde2'
        explicit mRNA) from the trajectory states, on the kon of the observed modes (one_hot_probas = False: levels
        drawn from the posteriors updated by the network at every minibatch); the network (support and
        signs kept) refined with refine_theta_with_sde, else fixed. Dilution and branching as in the simulations.
        The keyword arguments override the attributes for the fit inside fit_network (refine_in_loop): trajectory
        arrays (samples_rows, kon_targets, real_idx, rna), prolif (None: no MLP), temporal / learn_scale / learn_theta /
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
        real_idx = getattr(self, 'traj_real_idx', None) if real_idx is None else real_idx
        # Targets: levels drawn from the posteriors of the real cells (distribution not contracted, as their mean)
        post = None if self._soft_targets_missing(real_idx, 'free-running fit') else \
            self._posterior_levels(kon_targets, real_idx, samples_rows[:len(kon_targets)], ks)
        # ODE: modes of the states (EMD basins of the forcing: right histogram, bimodality of the deterministic cells)
        net_targets = None if mode != 'ode' else np.asarray(kon_targets, dtype=float)
        X = y_prot.reshape(T, N_tot, G)
        R = None if mode != 'sde2' else np.asarray(self.rna if rna is None else rna, dtype=float).reshape(T, N_tot, G)
        prolif = self.prolif_network if prolif == 'model' else prolif
        bt_all = None if b_states is None else np.asarray(b_states, dtype=float).reshape(T, N_tot)
        b_bounds = None if prolif is None else self._birth_bounds()
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
            ks_s = np.asarray(ks)[min(s_idx, np.shape(ks)[0] - 1)] if np.ndim(ks) == 3 else np.asarray(ks)
            k1_s = np.asarray(s1_of(k1, s_idx), dtype=float)
            stim = np.array([self._stim_schedule.per_cell(times[t + 1], np.array([s]))[0]
                             for t in range(T - 1)], dtype=float) * self.scale_proteins
            # Genes of two modes: sigma is the probability of the high mode, compared as a Bernoulli draw
            g = dict(X=X[:, slots].astype(float), k_lo=kb_s[:, 0], k_hi=kb_s[:, 1], k1=k1_s, stim=stim,
                     binary=1 + np.argmax(ks_s, axis=-1) == 2,
                     cond=int(cond[min(s_idx, len(cond) - 1)]),
                     idx=min(s_idx, basal.shape[0] - 1) if inter.ndim < 3 else min(s_idx, inter.shape[0] - 1),
                     targets=[None if not np.any(valid[N_tot * t + slots]) else
                              kon_targets[N_tot * t + slots][valid[N_tot * t + slots]] for t in range(T)],
                     start_ok=np.array([np.any(valid[N_tot * t + slots]) for t in range(T)]))
            if post is not None:  # posteriors and levels of the observed states, targets drawn from them
                rows = [N_tot * t + slots[valid[N_tot * t + slots]] for t in range(T)]
                g['post'] = [None if g['targets'][t] is None else post[0][rows[t]] for t in range(T)]
                g['levels'] = [None if g['targets'][t] is None else post[1][rows[t]] for t in range(T)]
            if mode == 'ode' and self.ode_ce_weight > 0:  # deterministic: + cross-entropy of each trajectory
                tau = sigmoid_target(net_targets.reshape(T, N_tot, G)[:, slots], kb_s[:, 0], kb_s[:, 1])
                g['tau'], g['tau_ok'] = tau, valid.reshape(T, N_tot)[:, slots]
                g['gw'] = gene_weights(ks_s)
            if mode == 'sde2':  # normalised mRNA m = M c s / k1 (mean kon s, as the bridges)
                M = R[:, slots] * (self.scale_proteins * np.asarray(s1_of(c_nb, s_idx)) / np.maximum(k1_s, 1e-12))[None, None, :]
                M[..., :ns] = 0.0
                g['M'] = M
                g['M_targets'] = [None if g['targets'][t] is None else
                                  M[t][valid[N_tot * t + slots]] for t in range(T)]
            # Birth rate of the dilution on the simulated states: MLP birth head at every integration step, else at
            # the start of each interval that of the nearest trajectory state of the sample (kon and proteins);
            # net rate of the branching, as in simulate_trajectories
            g['birth'], g['rate'], g['birth_every_step'] = None, None, True
            if self.protein_dilution and prolif is not None:
                lo, hi = b_bounds if b_bounds is not None else (-np.inf, np.inf)
                g['birth'] = lambda P, kon, t, sl, st=stim, si=s_idx, lo=lo, hi=hi: np.clip(prolif.predict_birth(
                    P[:, ns:], np.repeat(st[t][None], len(P), 0), samples=np.full(len(P), si)), lo, hi)
            elif self.protein_dilution and bt_all is not None:
                g['birth'] = self._nearest_state_birth(X[:, slots], kon_targets.reshape(T, N_tot, G)[:, slots],
                                                       bt_all[:, slots], valid.reshape(T, N_tot)[:, slots])
                g['birth_every_step'] = False
            if self.simulate_with_proliferation and prolif is not None:
                g['rate'] = lambda P, t, st=stim, si=s_idx: prolif.predict(
                    P[:, ns:], np.repeat(st[t][None], len(P), 0), samples=np.full(len(P), si))
            act = self.active_genes(s_idx)
            g['gmask'] = None if act is None else act.astype(float)  # genes of the sample's condition
            if self._memory_filter():  # kon of the start states: initial kon read on the mRNA
                g['K0'] = np.asarray(kon_targets, dtype=float).reshape(T, N_tot, G)[:, slots]
            if self.targets_accounting_late_mrna:  # modes read on the mRNA: kon at the lagged node of each class
                d0_s = self._mrna_d0(np.array([s_idx]))
                lc = [self._lag_classes(d0_s, times[t + 1] - times[t]) for t in range(T - 1)]
                g['lag'] = np.stack([lag[0] for lag, _ in lc])  # (T - 1, K) hours
                g['lag_w'] = lc[0][1][0]                         # (K, G) classes of the targets

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
                seed=task_seed(self.seed, *key), verb=verb > 1 if tag == 'fit_network' else verb,
                spec=self._spec_grn(), support_tol=self.seuil_min_network_intensity,
                memory_filter=self._memory_filter(), ce_weight=float(self.ode_ce_weight) if mode == 'ode' else 0.0)

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
                                     y_prot_given=None):
        """Proteins along the couplings of fit_network towards kon_targets (relaxation d1) and network refitted on
        them (final fit, without flow-matching term; inter_simul_ref as reference if given; y_prot_given: these
        proteins instead). Returns (y_prot, basal, inter)."""
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
        y_prot = self.estimate_trajectories(self.prot, times, d1, N=N_tot, kon_beta=kon_targets,
                                            s=self.scale_proteins, birth=self._state_birth()) \
            if y_prot_given is None else np.asarray(y_prot_given, dtype=float)
        error = self._count_errors_per_sample(y_prot[valid], self.kon_beta[valid], ks, self.inter, self.basal,
                                              samples_id=samples_id, samples_data=self.samples_data[valid])

        # Network targets as in fit_network (expected modes); the trajectories keep the forcing modes
        real_idx = getattr(self, 'traj_real_idx', None)
        targets = self.kon_beta if self._soft_targets_missing(real_idx, 'network refit') else \
            self._network_targets(self.kon_beta, real_idx)

        def fit_on(sels):
            return inference_network_multi(
                sels, self.samples_data, targets,
                y_prot, None, ks, n_stimuli=ns,  # final fit: no flow-matching term
                ref_network=self.ref_network, basal_init=basal_ref, inter_init=inter_ref,
                basal_ref=basal_ref, inter_ref=inter_ref,
                final=1,
                samples_id=samples_id,
                constrain_basal_uniform=self.constrain_basal_uniform,
                hard_forcing_ref=self.hard_forcing_ref, ref_constraint_pct=self.ref_constraint_pct,
                seuil_min_network_intensity=self.seuil_min_network_intensity,
                # Separate per condition (no fused term) with degradation_rates_per_condition, else a common network
                sample_conditions=self.sample_conditions if self.per_condition_fits()[0] else None,
                condition_pen=0.0, gene_mask=self.gene_mask,
                sigma_spec_grn=self._spec_grn(), **self._penalty_kw(),
                dist_targets=None if targets is self.kon_beta else self._distribution_targets(
                    self.kon_beta, real_idx, self.samples_data, self.times_data, ks, valid),
            )

        # Mean theta over fits on (time, sample, cell type)-stratified subsamples (single fit if one holds all)
        basal, inter, _, _ = self._fit_theta_averaged(
            fit_on, self.times_data, self.samples_data, self.traj_cell_types, self.n_network_fits,
            self._n_params_per_target(prev_cols, len(samples_id)), valid=valid)
        if common and n_rows is not None and np.ndim(inter) == 2:  # kept per sample (one network for every condition)
            inter = np.repeat(np.asarray(inter)[None], n_rows, axis=0)

        ### filter_edges
        if self.filter_network:
            seuil_intensity = self.seuil_min_network_intensity
            inter, _ = filter_network(len(times), N_tot, y_prot, ks, basal, inter,
                                      samples_data=self.samples_data, spec=self._spec_grn(),
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
        spec = self._spec_grn()

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
        # d1: its regression on the simulated state (birth head of the MLP, numpy copy re-evaluated by each worker at
        # every quadrature node, bounded by the birth rates of the trajectory states), else at the start of each
        # interval the birth (consistent with R_opt, saved by infer_network_simul) of the nearest trajectory state of
        # the inference interval containing the simulated one (match_distance on proteins and kon, same sample)
        from ..tools.estimate_proliferation import split_net_change
        from ..inference.proliferations import birth_heads, head_rates
        _mlp_birth = (self.protein_dilution and self.prolif_network is not None
                      and hasattr(self.prolif_network, 'predict_birth'))
        heads = head_of_cell = b_bounds = None
        if _mlp_birth:
            heads, hcond = birth_heads(self.prolif_network)
            head_of_cell = np.zeros(N, dtype=int) if hcond is None else \
                hcond[np.minimum(np.asarray(s_cells, dtype=int), len(hcond) - 1)]
            b_bounds = self._birth_bounds()
        b_traj = d_traj = None
        if self.protein_dilution and not _mlp_birth:
            bp, dp = self._state_rates()  # those of infer_network_simul (load_state_rates), else the prior
            T_tr = len(times_train)
            if bp is not None and len(bp) % T_tr == 0 and len(bp) == len(self.prot):
                N_tr = len(bp) // T_tr
                b_traj = bp.reshape(T_tr, N_tr)
                d_traj = (dp if dp is not None else np.zeros_like(bp)).reshape(T_tr, N_tr)
                ref_P = self.prot.reshape(T_tr, N_tr, -1)[..., ns:]
                ref_K = self.kon_beta.reshape(T_tr, N_tr, -1)[..., ns:]
                ref_s = np.asarray(self.samples_data).reshape(T_tr, N_tr)[0]  # sample of each trajectory slot
        if _branching or _mlp_birth:
            u_growth, w_growth = quadrature(self.n_growth_nodes)
        if _branching:
            # Resampling groups: cells of a sample only replace cells of the same sample
            groups = ([np.flatnonzero(np.asarray(samples_data)[:N] == s) for s in np.unique(np.asarray(samples_data)[:N])]
                      if samples_data is not None else [np.arange(N)])

        for cnt, time in enumerate(times[start_time:-1], start=start_time):
            delta_t = times[cnt + 1] - time
            # Recording times within the interval (end state last): quadrature nodes with the branching or the
            # birth rates re-evaluated along the path
            t_rec = delta_t * u_growth[1:] if (_branching or _mlp_birth) else delta_t

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

            # Birth rate of each cell (dilution): with the MLP, re-evaluated by the workers along the path (RATE
            # effects of the start state); else from the nearest trajectory state at the start of the interval
            b_cells = delta_cells = None
            if self.protein_dilution and (_mlp_birth or b_traj is not None):
                P_start = prot_modified[start_index:start_index + N, ns:]
                delta_cells = rate_effects(P_start[:, None, :])[:, 0]
                if b_traj is not None:
                    # Inference interval containing time (last one beyond): mean of the rates of the start and end
                    # states of the nearest trajectory, as the birth linear in time of the refit
                    k = int(np.clip(np.searchsorted(times_train, time, side='right') - 1, 0, len(times_train) - 1))
                    k_end = min(k + 1, len(times_train) - 1)
                    K_start = kon_vector[start_index:start_index + N, ns:]
                    b0, d0 = np.zeros(N), np.zeros(N)
                    for s_ in np.unique(s_cells):
                        rows = np.flatnonzero(s_cells == s_)
                        ref = np.flatnonzero(ref_s == s_)
                        ref = ref if len(ref) else np.arange(len(ref_s))
                        nn = ref[np.argmin(match_distance(P_start[rows], K_start[rows], ref_P[k, ref], ref_K[k, ref]),
                                           axis=1)]
                        b0[rows] = 0.5 * (b_traj[k, nn] + b_traj[k_end, nn])
                        d0[rows] = 0.5 * (d_traj[k, nn] + d_traj[k_end, nn])
                    b_cells = split_net_change(b0, d0, delta_cells)[0] if np.any(delta_cells) else b0

            mrna_pdmp = mrna_mode and bool(self.simulation_stochastic)
            stoch, scale_p = bool(self.simulation_stochastic), self.scale_proteins
            # mRNA memory (targets_accounting_late_mrna), counts drawn from kon (no explicit mRNA PDMP): the kon of each
            # target of the end state is read at the lagged node of its class (_lag_classes, its sample's d0), recorded
            # per cell
            lag_on = self.targets_accounting_late_mrna and not mrna_pdmp
            t_rec_base = np.atleast_1d(t_rec).astype(float)
            t_rec_cells, i_orig, i_lag = [t_rec] * N, [np.arange(len(t_rec_base))] * N, [None] * N
            lag_w = None
            if lag_on:
                lag_c, lag_w = self._lag_classes(self._mrna_d0(s_cells), delta_t)  # (N, K), (N, K, G_tot)
                t_lag = delta_t - lag_c
                for n in range(N):
                    ok = (t_lag[n] > 0) & (t_lag[n] < delta_t)
                    tr = np.unique(np.concatenate([t_rec_base, t_lag[n][ok]]))
                    t_rec_cells[n] = tr
                    i_orig[n] = np.searchsorted(tr, t_rec_base)
                    i_lag[n] = [int(np.searchsorted(tr, tl)) if o else None for tl, o in zip(t_lag[n], ok)]

            # The worker gets only its cell's rows (no self, no full arrays): joblib pickles the closure with every
            # batch of tasks, so capturing the model or the state arrays made the simulation several times slower
            def run_main_loop_for_cell(n, P0, M0, lvl0, _basal_cells=basal_cells, _basal_t_cnt=basal_t[cnt],
                                       _stim_cells=stim_cells, _b_cells=b_cells, _inter_cnt=inter_t[cnt],
                                       _heads=heads, _head_of_cell=head_of_cell, _delta=delta_cells,
                                       _b_bounds=b_bounds, _t_rec=t_rec):
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
                def run(P_, M_, b_, t_):
                    if mrna_pdmp:
                        return simulate_next_mrna_pdmp(
                            deg[0, :], deg[1, :],
                            ks_of(kz, s_n) * deg[0][:, None], c_cells[n],
                            basal_n, inter_of(_inter_cnt, s_n), t_, scale_p,
                            P0=P_, M0=M_, ns=ns, stim_vals=_stim_vals, chrom=ch, dilution=b_, spec=spec)
                    if stoch:
                        return simulate_next_prot_pdmp(
                            deg[1, :] + b_,
                            ks_of(kz, s_n) * deg[0][:, None],
                            s1_of(rescale, s_n) * (deg[0, :] / deg[1, :]),
                            basal_n, inter_of(_inter_cnt, s_n), t_,
                            scale_p, P0=P_, ns=ns, stim_vals=_stim_vals, chrom=ch, spec=spec)
                    return simulate_next_prot_ode(
                        deg[1, :] + b_, ks_of(ks, s_n),
                        basal_n, inter_of(_inter_cnt, s_n), t_,
                        scale_p, P0=P_, ns=ns, stim_vals=_stim_vals, chrom=ch, spec=spec,
                        cfac=deg[1, :] / np.maximum(deg[1, :] + b_, 1e-300))

                if _heads is None:
                    sim = run(P0, M0, b_vec, _t_rec)
                else:
                    # Birth rate re-evaluated on the cell's state at every node (exact restart: memoryless bursts)
                    head = _heads[_head_of_cell[n]]
                    P_, M_ = np.array(P0, dtype=float), None if M0 is None else np.array(M0, dtype=float)
                    ps, ms, t_prev = [], [], 0.0
                    for t_q in np.atleast_1d(_t_rec):
                        b_vec = np.zeros(deg.shape[1])
                        if head is not None:
                            b0_, d0_ = head_rates(head, P_[ns:], _stim_vals)
                            if _b_bounds is not None:
                                b0_ = float(np.clip(b0_, *_b_bounds))
                            b_vec[ns:] = split_net_change(b0_, d0_, _delta[n])[0] if _delta[n] else b0_
                        seg = run(P_, M_, b_vec, np.array([t_q - t_prev]))
                        ps.append(seg.p[-1])
                        P_ = P_.copy()
                        P_[1:] = seg.p[-1]
                        if mrna_pdmp:
                            ms.append(seg.m[-1])
                            M_ = M_.copy()
                            M_[1:] = seg.m[-1]
                        t_prev = t_q
                    sim = Simulation(np.atleast_1d(_t_rec), np.array(ps), m=np.array(ms) if mrna_pdmp else None)
                return sim, (None if ch is None else ch.lvl)

            outputs = Parallel(n_jobs=-1)(
                delayed(seeded_call)(task_seed(self.seed, 4, cnt, n), run_main_loop_for_cell, n,
                                     prot_modified[start_index + n, :],
                                     None if m_state is None else m_state[start_index + n, :],
                                     None if chrom_lvl is None else chrom_lvl[start_index + n],
                                     _t_rec=t_rec_cells[n])
                for n in range(0, N))
            results = [o[0] for o in outputs]
            # Proteins (K, N, G_genes) the lag of each class before the end state (the start of the interval at most)
            P_lag = None
            if lag_on:
                P_lag = np.array([[results[n].p[i_lag[n][k]][ns - 1:] if i_lag[n][k] is not None
                                   else prot_modified[start_index + n, ns:] for n in range(N)]
                                  for k in range(lag_w.shape[1])])

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
                if P_lag is not None:
                    P_lag[:, off[:, ns:]] = prot_modified[start_index:start_index + N, ns:][off[:, ns:]]
                if mrna_pdmp:
                    m_state[end_index:end_index + N][off] = m_state[start_index:start_index + N][off]

            # --- Branching process resampling ---
            # log_weight_n = ∫ R(P_n(s)) ds over the simulated path (trapezoidal rule on the
            # recorded states), the same quadrature as in the MLP training. Multinomial
            # resampling within each sample keeps N cells (no Poisson extinction at R ≈ 0).
            if _branching:
                path = np.stack([prot_modified[start_index:start_index + N, ns:]]
                                + [np.array([results[n].p[i_orig[n][q]][ns - 1:] for n in range(N)])
                                   for q in range(len(t_rec_base))], axis=1)  # (N, Q, G)
                R_path = _prolif_fn(path, stim_cells[:, None, :]) if _prolif_fn is not None else np.zeros(path.shape[:2])
                R_path = R_path + rate_effects(path)
                log_weights = (R_path * w_growth).sum(axis=1) * delta_t
                # Population size: mean growth factor of the cells over the interval
                self.log_population[cnt + 1] = self.log_population[cnt] + float(
                    np.log(np.mean(np.exp(log_weights - log_weights.max()))) + log_weights.max())
                P_end = prot_modified[end_index:end_index + N, ns:].copy()
                M_end = None if not mrna_pdmp else m_state[end_index:end_index + N, ns:].copy()
                L_end = None if chrom0 is None else chrom_lvl[end_index:end_index + N].copy()
                P_lag_end = None if P_lag is None else P_lag.copy()
                for grp in groups:
                    weights = np.exp(log_weights[grp] - log_weights[grp].max())
                    src = grp[np.random.choice(len(grp), len(grp), replace=True, p=weights / weights.sum())]
                    prot_modified[end_index + grp, ns:] = P_end[src]
                    if M_end is not None:
                        m_state[end_index + grp, ns:] = M_end[src]
                    if L_end is not None:
                        chrom_lvl[end_index + grp] = L_end[src]
                    if P_lag_end is not None:
                        P_lag[:, grp] = P_lag_end[:, src]

            # Set stim values at step boundary from schedule (authoritative source).
            prot_modified[end_index:end_index + N, :ns] = stim_cells

            # Each cell with its own sample's basal and mixture (index-based, as basal_cells)
            n_samp = basal_t.shape[1] if basal_t.ndim == 3 else (ks.shape[0] if ks.ndim == 3 else 1)
            prot_read = prot_modified[end_index:end_index + N, :]
            mem = None
            if P_lag is not None:  # kon read on the mRNA: single lagged node of the class of each target
                mem = PathProt.zeros(N, P_lag.shape[0], prot_read.shape[1])
                mem.nodes[:, :, :ns] = prot_read[:, None, :ns]
                mem.nodes[:, :, ns:] = np.swapaxes(P_lag, 0, 1)
                mem.weights[:, 0] = 0.0
                mem.weights[:, 1:] = lag_w
            kon_vector[end_index:end_index+N, ns:] = self._kon_read(
                prot_read, mem, ks, inter_t[cnt], basal_t[cnt], np.arange(n_samp), s_cells)[:, ns:]
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

    def _posteriors_and_masses(self, x, vect_t, ks, c, depth=None):
        """
        Posteriors and mode masses per time of cells classified with fixed modes (ks, c), as in the mixture fit:
        masses from uniform-prior posteriors, their time mean as prior of the posteriors, final masses per time; with
        the levels of the continuous model, the MAP weights per time of the grid.
        """
        set_depth_model(self.depth_model)
        if len(ks) >= 2:
            return grid_weights_fixed(x, vect_t, ks, c, s=depth)
        times = np.sort(np.unique(vect_t))
        resp0, _ = predict_resp(x, ks, c, s=depth)
        nu0 = self._mode_masses(resp0, x, vect_t, times, ks, c, depth)
        pi_glob = np.sum([nu0[t] * np.mean(vect_t == t) for t in times], axis=0)
        resp, _ = predict_resp(x, ks, c, pi=pi_glob, forcing=self.mean_forcing_em, s=depth)
        return resp, self._mode_masses(resp, x, vect_t, times, ks, c, depth)

    def fit_mixture_test(self, data_rna, ks, c, verb=False, depth=None):
        """Classify test cells into mixture modes using fixed kinetic parameters (ks, c of the training).

        Sets self.modes, self.basins, self.proba_init, and self.pi_init (mode masses per time, computed on the test
        cells as on the training ones: mean-preserving with mean_forcing_em, never the training masses).
        """
        ns = self.n_stimuli
        N_cells, G_tot = data_rna.shape
        vect_t = data_rna[:, 0]
        times = np.sort(np.unique(vect_t))

        frequency_modes_smooth = np.ones_like(data_rna, dtype=float)
        M = self.a.shape[-2] - 1  # number of mixture modes
        frequency_basins = np.zeros((N_cells, G_tot), dtype=np.int8)
        frequency_proba_init = np.zeros((N_cells, G_tot, M), dtype=float)
        pi_init_test = []

        for g in range(ns, G_tot):
            ng = np.argmax(ks[1:, g]) + 2

            # Posteriors and mode masses as in the training mixture, from the test cells only
            proba, pi_g = self._posteriors_and_masses(data_rna[:, g], vect_t, ks[:ng, g], c[g], depth=depth)
            # Initial basins: EMD under the test masses (or a draw), as for the training set
            basins_g = self._basins_from_posteriors(proba, vect_t, pi_g)
            frequency_basins[:, g] = basins_g
            frequency_proba_init[:, g, :ng] = proba
            frequency_proba_init[:, g, ng:] = 0
            frequency_modes_smooth[:, g] = ks[:ng, g][basins_g]
            if verb:
                print('[infer_test]', f'Gene {g} calibrated...', ks[:ng, g], c[g])

            pi_init_test.append(pi_g)

        scale_max = np.max(self.a[:-1, :], axis=0)
        frequency_modes_smooth /= np.maximum(scale_max, EPS)  # all-zero modes would give 0/0
        self.modes = frequency_modes_smooth
        self.basins = frequency_basins
        self.proba_init = frequency_proba_init
        self.pi_init = pi_init_test



    def _fit_mixture_test_per_sample(self, data_rna, vect_samples_id, samples_id, depth=None):
        """fit_mixture_test with per-sample mixtures (a 3-D): each sample's cells with its own parameters."""
        a3 = self.a
        N, G_tot = data_rna.shape
        M = a3.shape[1] - 1
        modes, basins, proba_init = np.zeros((N, G_tot)), np.zeros((N, G_tot), dtype=np.int8), np.zeros((N, G_tot, M))
        vect_t = data_rna[:, 0]
        times = np.sort(np.unique(vect_t))
        pi_sum = None
        for s_idx, sid in enumerate(samples_id):
            m = vect_samples_id == sid
            self.a = a3[min(s_idx, len(a3) - 1)]
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
        self.a = a3
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
        The EMD basins force the trajectories, as in the training; with one_hot_probas = False, the entropic coupling
        of the same costs gives the calibrated posteriors (posteriors_soft) and their expected modes (modes_soft).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        ns, n_cells = self.n_stimuli, len(vect_t)
        t_tr = np.asarray(train['cell_times'])
        t_tr_u = np.unique(t_tr)

        def posterior(p):
            return p / np.maximum(np.sum(p, axis=1, keepdims=True), EPS)

        def fit_lr(X, y):
            # None when the training assignment cannot be learned (too few cells or a single basin)
            if len(y) < 30 or len(np.unique(y)) < 2 or np.bincount(y).max() > len(y) - 3:
                return None
            # Standardised log-posteriors: the multinomial fit converges with many modes (levels of the continuous model)
            return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(
                np.log(np.clip(X, 1e-4, 1)), y)

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
            tmp_post = np.zeros((n_cells, l_max))
            # Cells whose condition selected g (select_genes_per_condition); the others keep their single mode
            act = np.ones(n_cells, bool) if act_cells is None else act_cells[:, g]
            tmp_modes[~act] = obj[~act, 0]
            tmp_post[~act, 0] = 1.0
            for t_i in (times if self.temporal_basins else [None]):
                idx = np.flatnonzero(act & (np.ones(n_cells, bool) if t_i is None else vect_t == t_i))
                if not len(idx):
                    continue
                proba = self.proba_init[idx, g, :l_max].copy()
                post = posterior(proba)
                proba /= np.max(proba, axis=1, keepdims=True)
                lr = lr_all
                if t_i is not None:
                    nu = self._target_masses(post, self.pi_init[g - ns][t_i])
                    # Regression at the nearest training time, unless it misses basins of the pooled one
                    m = t_tr == t_tr_u[np.argmin(np.abs(t_tr_u - t_i))]
                    lr_t = fit_lr(Xtr[m], ytr[m])
                    if lr_t is not None and (lr_all is None or len(lr_t.classes_) == len(lr_all.classes_)):
                        lr = lr_t
                else:
                    nu = self._target_masses(post, np.sum([np.asarray(self.pi_init[g - ns][t])[:l_max] *
                                                           np.sum((vect_t == t) & act) / act.sum() for t in times], axis=0))
                dist = np.clip(-(np.log(proba) + (1 - weight_prob) * np.log(predicted(lr, post))), 0, 100)
                mu = np.ones(len(idx)) / len(idx)
                k, probs = self._assign_from_costs(mu, nu, dist)
                tmp_basins[idx] = k
                tmp_modes[idx] = obj[idx, k]
                tmp_post[idx] = np.eye(l_max)[k] if probs is None else probs
            return tmp_basins, tmp_modes, tmp_post, (tmp_post * obj).sum(axis=1)

        results = Parallel(n_jobs=-1)(delayed(seeded_call)(task_seed(self.seed, 9, g), run_gene, g)
                                      for g in range(ns, G_tot))
        soft = not self.one_hot_probas
        self.modes_soft = self.modes.copy() if soft else None
        self.posteriors_soft = np.zeros(self.proba_init.shape) if soft else None
        for g, (tmp_basins, tmp_modes, tmp_post, tmp_soft) in zip(range(ns, G_tot), results):
            self.basins[:, g] = tmp_basins
            self.modes[:, g] = tmp_modes
            if soft:
                self.modes_soft[:, g] = tmp_soft
                self.posteriors_soft[:, g, :tmp_post.shape[1]] = tmp_post

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
        self._enforce_soft_basins(verb=verb)
        self._floor_mrna_degradation(verb=verb)
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
                self._set_mode_soft(ko_cells, g, 0)
                self._set_mode_soft(ov_cells, g, l_max - 1)

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


    def fit(self, data_rna, verb=True):

        self.fit_mixture(data_rna)
        self.fit_network(data_rna, verb=verb)
        # self.refine_network_degradations()