
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
from scipy.ndimage import gaussian_filter1d
from ..inference import (inference_network, filter_network,
                        minimal_repetition_choice, find_next_prot, my_otdistance, count_errors, kon_ref_vector, inference_alpha,
                        NegativeBinomialMixtureEM, predict_resp,
                        simulate_next_prot_ode, simulate_next_prot_pdmp,
                        train_kon_correction_mlp, infer_ratio_d0_d1_full, infer_ratio_d0_d1_unitary, inference_degradation_prot,
                        train_proliferation_mlp, fit_scale_theta)

np.set_printoptions(precision=3, suppress=True)
EPS=1e-16


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

        ### Default behaviour

        ## Compute mixture parameters
        self.hard_em = 1 # Do we initialize with a hard_em ?
        self.preserve_mean_values = 1 # Do we ensure temporal constraints when fitting the basins in the hard_em ?
        self.mean_forcing_em = 0.5 # at which point we force the mean correction: the higher the more
        self.force_basins = 1.0 # Do we want to ensure the means to be preserved by the NB mixture ? It may not preserve multistability
        self.temporal_basins = 1 # Is it preserved temporally ?
        self.transform_proba = 0 # Do we want to force probas to be steep for compatibility with sigmoid model?
        self.seuil = 1e-2 # minimum for beta mixture parameters (second parameters)
        self.batch_size_mixture = 512 # Maximum number of cells per time used for mixture calibration in the inference.
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
        self.batch_size_traj = 512 # Maximum number of cells used per time point per sample for solving EOT problems in the inference.
        self.unbalanced_reg = 5 # Unbalanced regularization parameter for UOT if > 0, OT if 0
        self.init_entropic_noise = 1.5 # Initial entropic penalization for OT
        self.quant_samples = .8 # Quantile of cells number per sample to use for inference
        # General parameters to calibrate protein reconstruction
        self.scale_proteins = 1 # Eventually rescale protein values (recommended:1-2)
        self.scale_mrnas = 100 # Eventually rescale mRNA values (recommended:100)
        self.fact_simple = 2 # slight transformation for constrative modes in learning phase
        # Network inference with scipy
        self.loss_norm = 'CE'
        self.scale_pen = 20 # Error that is expected = 1/scale_pen
        self.compute_with_proba = 0 # Determine if compute with proba or kon values in network inference (recommended:1)
        self.weight_prev = .4 # max = .5 to not withdrawn the inference on timepoints, allows the calibration to incorporate some "flow-matching" method
        self.batch_size_network = 2048 # Maximum number of cells used for network calibration in the inference.
        # Inference of alpha = switch moment between each timepoint and modes
        self.update_modes = 1
        self.alpha_threshold = .6 # max = 1, thershold for important transition to update alpha full
        self.n_pas = 25 # number of timesteps between timepoints for inference of alpha
        # Penalization/prior information
        self.stimulus = 1.0 # 1 if we simulate with a stimulus. If not we can penalize the stimulus with a value between 1 and 0: 0 = no sitmulus
        self.prior_network_pen = 1.0 # 1 if we don't use prior information. If not we can penalize the non-existing age in prior network with values between 1 and 0: 0 = impossible edge
        self.constrain_basal_uniform = 1.0 # >= 0 penalty strength that pushes per-sample basals to be equal (ignores samples pinned by KO/OV basal_ref)
        self.hard_forcing_ref = True # if True, constrain all network params to ±ref_constraint_pct around inter_ref
        self.ref_constraint_pct = 0.1 # fractional tolerance around inter_ref values for bounds (used when hard_forcing_ref=True)
        self.lambda_mlp    = .5  # Mix weight for training-data ratios vs MLP in simulate_full_with_harissa:
                                  # 1 = pure linear interpolation of observed g, 0 = pure MLP g(P, kon(P))
        # Filtering
        self.filter_network = 1 # Do we filter the network ? It also builds a temporal network using the filter criterium
        self.seuil_min_network = 1e-1 # minimum absolute value for network interaction bounds and filtering

        ## Compute degradations after inference
        self.recompute_degradations = 1 # Do we want to recompute degradation rates for simulations ?
        self.batch_size_degradations = 256 # number of trajectories to take to make the inference (slow without gpu)
        self.use_temporal_degradations = 1 # If so, compute temporal degradation rates for simulations ?
        self.lambda_scale  = 1e-3  # L2 penalty on scale[ns:] around 1 (large = scale stays ~1; 0 = free)
        self.lambda_deg0   = 1     # L2 penalty on d around d_init (0 = free; large = stays close to prior)
        self.lambda_deg1   = 1e-3  # L2 penalty on d_t around 0 (0 = free; large = stays close to non-temporal)
        self.smooth_degradations_sigma = None  # None=auto KDE+CV, 0=off, float>0=fixed sigma (in time-step units)
        self.smooth_degradations_strength = 0  # blend weight in [0,1]: 0=no smoothing, 1=full smoothing

        ## Simulations
        self.simulation_stochastic = True # 1 if we simulate Bursty-like proteins, 0 if deterministic limit for proteins
        self.finish_by_determinist = False # 1 if we simulate with deterministic limit for the last timepoint
        self.min_ratio = .05
        self.max_ratio = 50
        self.simulate_full_with_harissa = False  # use Harissa PDMP to jointly simulate proteins+mRNAs
        self.kon_beta_harissa = None  # continuous adaptive_shrinkage burst-rate estimates (set by loop_trajectories)
        self.kon_mlp = None           # KonCorrectionMLP trained in refine_network_degradations (Harissa branch)

        ## Proliferation
        self.recompute_proliferations = False    # train a ProliferationMLP on R_opt in refine_network_degradations
        self.simulate_with_proliferation = False # apply branching process in simulate_trajectories_unitary
        self.prolif_network = None               # ProliferationMLP trained in refine_network_degradations
        self.R_opt = None                        # per-cell optimal proliferation rates from OT coupling marginals
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
        

    def _parse_input(self, data, time_key='time'):
        try:
            import anndata
            import scipy.sparse
            if isinstance(data, anndata.AnnData):
                vect_t = data.obs[time_key].values.astype(float)
                X = data.X.toarray() if scipy.sparse.issparse(data.X) else np.asarray(data.X, dtype=float)
                if self.n_stimuli > 1:
                    X = np.hstack([np.zeros((X.shape[0], self.n_stimuli - 1), dtype=float), X])
                return np.column_stack([vect_t, X])
        except (ImportError, AttributeError):
            pass
        return data

    def _build_stimulus_schedule(self, times_unique, stimulus_schedule=None, times_ref=None):
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
                          cell_rd=None, verb=True, kov_cell_mask=None, scboolseq_matrix=None, scboolseq_dropouts=None):
        """
        Parameters
        ----------
        cell_rd : (N_cells,) array or None
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
                                                 use_scBoolSeq=(scboolseq_matrix is not None))

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
            model = kinetics.fit(x, vect_t=vect_t, seuil=self.seuil,
                                 s=cell_rd,
                                 batch_size_mixture=self.batch_size_mixture,
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
                tmp = np.zeros_like(proba)
                if self.temporal_basins:
                    for t_i in np.unique(vect_t):
                        indices = (vect_t == t_i)
                        tmp_proba_i = np.zeros_like(proba[indices])
                        proba_i = proba[indices]
                        n_cells_i = np.sum(indices)
                        mu = np.ones(n_cells_i)/n_cells_i
                        nu = pi[t_i] * self.force_basins + np.sum(proba_i, axis=0) * (1 - self.force_basins) 
                        nu /= np.sum(nu)
                        dist = - np.log(proba_i)
                        coupling = ot.bregman.sinkhorn(mu, nu, dist, reg=1, numItermax=10000)
                        idx = np.argmax(coupling, axis=1)
                        for cell in range(n_cells_i):
                            tmp_proba_i[cell, idx[cell]] = 1
                        tmp[indices, :] = tmp_proba_i[:, :]
                else:
                    mu = np.ones(N_cells)/N_cells
                    nu = np.sum([pi[t_i] * np.sum(vect_t == t_i)/N_cells 
                                        for t_i in np.unique(vect_t)], axis=0) * self.force_basins + np.sum(
                                                           proba, axis=0) * (1 - self.force_basins)
                    nu /= np.sum(nu)
                    dist = - np.log(proba) 
                    coupling = ot.bregman.sinkhorn(mu, nu, dist, reg=1, numItermax=10000)
                    idx = np.argmax(coupling, axis=1)
                    for cell in range(N_cells):
                        tmp[cell, idx[cell]] = 1
        
            return ks, c, pi0, proba, tmp, pi

        results = Parallel(n_jobs=-1)(
        delayed(run_main_loop_for_gene)(g) for g in range(ns, G_tot)
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
        frequency_modes_smooth /= scale_max

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



    def fit_mixture(self, data, refilter=0, gene_names=np.arange(1, 50000), min_components=2, max_components=2, max_iter_kinetics=0, cell_rd=None, verb=True, stimulus_schedule=None, time_key='time', kov_cell_mask=None):
        """
        Fit the mixture model parameters to the data.

        Parameters
        ----------
        cell_rd : (N_cells,) array, pd.Series, ou None
            Facteurs de read depth par cellule, typiquement issus de
            adata.obs['rd'] (calculés par infer_rd.py).
            Si None, le modèle NB classique sans correction est utilisé.

        Exemple d'appel avec correction de read depth::

            rd = np.asarray(adata.obs['rd'])
            model.fit_mixture(data_rna, ..., cell_rd=rd)
        """
        data_rna = self._parse_input(data, time_key)
        N_cells, G_tot = data_rna.shape
        vect_t = data_rna[:, 0]
        self._stim_schedule = self._build_stimulus_schedule(np.sort(np.unique(vect_t)), stimulus_schedule)

        # Auto-extract cell_rd from AnnData if not provided
        try:
            import anndata
            if isinstance(data, anndata.AnnData) and cell_rd is None and 'rd' in data.obs:
                cell_rd = np.asarray(data.obs['rd'].values, dtype=float)
                cell_rd = cell_rd / (np.mean(cell_rd) + 1e-16)
        except ImportError:
            pass

        # Conversion propre du read depth (supporte pd.Series et np.array)
        if cell_rd is not None:
            cell_rd = np.asarray(cell_rd, dtype=float).reshape(-1)
            assert len(cell_rd) == N_cells, (
                f"cell_rd a {len(cell_rd)} entrées mais data a {N_cells} cellules."
            )

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
                                        cell_rd=cell_rd,
                                        verb=verb,
                                        kov_cell_mask=kov_cell_mask,
                                        scboolseq_matrix=scboolseq_matrix,
                                        scboolseq_dropouts=scboolseq_dropouts)

        self.pi_zinb = pi_zeros
        self.modes = frequency_modes_smooth
        self.proba_init = frequency_proba_init
        self.proba = frequency_proba_modif
        n_components = np.size(self.a[:-1], 0)
        if self.adapt_size_network:
            self.n_networks = n_components - 1
        else:
            if n_components > self.n_networks+1:
                a_new = np.zeros((self.n_networks+2, G_tot))
                qs = np.linspace(0, 1, self.n_networks+1)
                for g in range(G_tot):
                    l_max = len(np.unique(self.modes[:, g]))
                    a_new[:-1, g] = np.quantile(self.a[:l_max, g], qs)
                a_new[-1, :] = self.a[-1, :]
                self.a = a_new.copy()


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
                                      basal, inter, s1, ks, nb_cells, init_cells, R_opt_traj, to_keep_for_update, offset_init=[0],
                                      n_iter=1, N_full=[100], N_samples=[100], intensity_prior=10):
        """
        Infer the protein trajectories when d1 is known and theta is not.
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
        sim_real_idx = np.full(rna_modified.shape[0], -1, dtype=int)

        # Set stimulus values per timepoint (schedule-based)
        for t_idx, t_i in enumerate(times):
            sl = slice(t_idx * N_total, (t_idx + 1) * N_total)
            val = np.asarray(self._stim_schedule[t_i], dtype=float)
            prot_formodes[vect_t == t_i, :ns] = val * self.scale_proteins
            rna_modified[sl, :ns] = val * self.scale_mrnas
            prot_modified[sl, :ns] = val * self.scale_proteins
            kon_modified[sl, :ns] = (val >= 0.5)

        # Fill initial state (t = 0) and initialize sim_real_idx for those cells
        offset = 0
        for s, sample in enumerate(samples_id):
            cell_indices = (vect_samples_id == sample)
            global_cell_idx = np.flatnonzero(cell_indices)
            selected_init = init_cells[s]

            kon_modified[offset+offset_init[s]:offset+offset_init[s] + N_samples[s], ns:] = self.modes[cell_indices][selected_init, ns:]
            if self.compute_with_proba:
                proba_modified[offset+offset_init[s]:offset+offset_init[s] + N_samples[s]] = self.proba[cell_indices][selected_init]
            rna_modified[offset+offset_init[s]:offset+offset_init[s] + N_samples[s], ns:] = vect_rna[cell_indices][selected_init, ns:]
            sim_real_idx[offset+offset_init[s]:offset+offset_init[s] + N_samples[s]] = global_cell_idx[selected_init]

            offset += N_full[s]

        prot_modified[:N_total, ns:] = self.adaptive_shrinkage_init(rna_modified[:N_total, ns:] * s1, kon_modified[:N_total, ns:])
        if n_iter == 1: 
            N_cells_0 = np.sum(vect_t == times[0])
            prot_formodes[:N_cells_0, ns:] = self.adaptive_shrinkage_init(vect_rna[:N_cells_0, ns:] * s1, self.modes[:N_cells_0, ns:])

        for t_idx, time in enumerate(times[:-1]):
            offset = 0
            for s_idx, sample in enumerate(samples_id):
                start_next = (vect_t == times[t_idx + 1]) & (vect_samples_id == sample)
                offset_init_s = offset + offset_init[s_idx]
                start_index = N_total * t_idx + offset_init_s
                next_index = N_total * (t_idx + 1) + offset_init_s
                N_sample = N_samples[s_idx]
                N_cells = nb_cells[s_idx, t_idx + 1]
                start_index_full = N_total * (t_idx + 1) + offset
                next_index_full = N_total * (t_idx + 1) + offset + N_full[s_idx]

                vect_samples_id_modified[N_total * t_idx + offset:N_total * t_idx + offset + N_full[s_idx]] = s_idx

                if N_sample:

                    # Snapshot minimal des anciennes valeurs pour le bloc alpha
                    if prot_old_is_nonzero and time != times[-2]:
                        prot_old_snapshot = y_prot_old[start_index_full:next_index_full, ns:].copy()
                        kon_old_snapshot  = y_kon_old[start_index_full:next_index_full, ns:].copy()
                    else:
                        prot_old_snapshot = None
                        kon_old_snapshot  = None

                    current_indices = np.arange(start_index, start_index + N_sample)
                    alpha_indices = np.arange(offset_init_s, offset_init_s + N_sample)
                    mode_init = self.adaptive_shrinkage(rna_modified[current_indices, ns:] * s1, kon_modified[current_indices, ns:]) / s1
                    mode_end = self.adaptive_shrinkage(vect_rna[start_next, ns:] * s1, self.modes[start_next, ns:]) / s1

                    basal_s = basal[min(s_idx, basal.shape[0] - 1)] if basal.ndim == 3 else basal
                    pairwise_dist, next_prot = my_otdistance(
                        kon_modified[current_indices, ns:], self.modes[start_next, ns:],
                        prot_modified[current_indices, ns:],
                        rna_modified[current_indices, ns:], vect_rna[start_next, ns:],
                        proba_modified[current_indices, ns:], self.proba[start_next, ns:, :],
                        mode_init, mode_end,
                        alpha_modified[t_idx, alpha_indices],
                        s1, ks, self.d[1, ns:], times[t_idx + 1] - time, basal_s, inter, loss=self.loss_norm,
                        n_iter=n_iter, intensity_prior=intensity_prior,
                        compute_with_proba=self.compute_with_proba,
                        n_stimuli=ns, stim_vals=np.asarray(self._stim_schedule[times[t_idx + 1]], dtype=np.float64),
                        scale_proteins=self.scale_proteins
                    )

                    cell_idx = np.flatnonzero(start_next)
                    delta_t = times[t_idx + 1] - time
                    src_real = sim_real_idx[current_indices]

                    # --- Transition rate cost adjustment ---
                    _tr = getattr(self, '_transition_rates', None)
                    _ct = getattr(self, '_cell_types', None)
                    if _tr is not None and _ct is not None:
                        _labels = getattr(self, '_transition_type_labels', None)
                        if _labels is not None:
                            _lbl_to_i = {l: i for i, l in enumerate(_labels)}
                            src_ti = np.array([_lbl_to_i.get(str(_ct[r]), 0) for r in src_real])
                            tgt_ti = np.array([_lbl_to_i.get(str(_ct[j]), 0) for j in cell_idx])
                        else:
                            _uniq = np.unique(_ct)
                            _ct_to_i = {c: i for i, c in enumerate(_uniq)}
                            src_ti = np.array([_ct_to_i.get(_ct[r], 0) for r in src_real])
                            tgt_ti = np.array([_ct_to_i.get(_ct[j], 0) for j in cell_idx])
                        n_types_tr = _tr.shape[0]
                        src_ti = np.clip(src_ti, 0, n_types_tr - 1)
                        tgt_ti = np.clip(tgt_ti, 0, n_types_tr - 1)
                        tr_prob = np.exp(_tr * delta_t)
                        tr_prob = tr_prob / tr_prob.sum(axis=1, keepdims=True) * _tr.shape[1]
                        tr_w = tr_prob[np.ix_(src_ti, tgt_ti)]
                        pairwise_dist = pairwise_dist / np.maximum(tr_w, 1e-10)

                    # --- Growth-weighted OT marginals ---
                    _pr = getattr(self, '_prolif_rate', None)
                    _dr = getattr(self, '_death_rate', None)
                    _has_growth = _pr is not None or _dr is not None
                    if _has_growth:
                        _p = _pr if _pr is not None else np.zeros(len(vect_t))
                        _d = _dr if _dr is not None else np.zeros(len(vect_t))
                        mu = np.exp((_p[src_real] - _d[src_real]) * delta_t / 2)
                        mu /= mu.sum()
                        nu = np.exp((_d[cell_idx] - _p[cell_idx]) * delta_t / 2)
                        nu /= nu.sum()
                    else:
                        mu = np.ones(N_sample) / N_sample
                        nu = np.ones(N_cells) / N_cells

                    tmp = np.log(G)
                    reg = max(self.init_entropic_noise * tmp * (1 / n_iter)**(1 - 1/n_iter), .01)
                    stopThr, numItermax = self.stopThr_init, int(10000 / min(1, reg))
                    if not self.unbalanced_reg:
                        while stopThr <= self.stopThr_init*100:
                            try:
                                coupling = ot.bregman.sinkhorn(
                                    mu, nu, pairwise_dist,
                                    reg=reg,
                                    numItermax=numItermax,
                                    stopThr=stopThr
                                )
                                break
                            except Exception:
                                stopThr *= 2
                                numItermax *= 2
                        else:
                            print('Warning, main Sinkhorn did not converge')
                    else:
                        reg_m = np.array([1e3, self.unbalanced_reg*tmp])
                        while stopThr <= self.stopThr_init*100:
                            try:
                                coupling = ot.unbalanced.sinkhorn_unbalanced(
                                    mu, nu, pairwise_dist,
                                    reg=reg,
                                    reg_m=reg_m,
                                    numItermax=numItermax,
                                    stopThr=stopThr
                                )
                                break
                            except Exception:
                                stopThr *= 2
                                numItermax *= 2
                        else:
                            print('Warning, main Sinkhorn did not converge')

                    row_marginals = coupling.sum(axis=1)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        log_m = np.log(np.maximum(row_marginals, EPS))
                        R_n = (2.0 / delta_t) * (log_m - log_m.mean())
                    R_opt_traj[start_index:start_index + N_sample] = R_n

                    # coupling /= coupling.sum(axis=0, keepdims=True)
                    # row_marginals = coupling.sum(axis=1)
                    # col_marginals = coupling.sum(axis=0)
                    # coupling = ot.emd(row_marginals, col_marginals, -np.log(coupling + EPS))
                    coupling_norm = coupling / coupling.sum(axis=1, keepdims=True)
                    for n in range(N_sample):
                        m = np.random.choice(N_cells, p=coupling_norm[n])
                        target_index = next_index + n
                        kon_modified[target_index] = self.modes[cell_idx[m]]
                        if self.compute_with_proba:
                            proba_modified[target_index] = self.proba[cell_idx[m]]
                        rna_modified[target_index, ns:] = vect_rna[cell_idx[m], ns:]
                        prot_modified[target_index, ns:] = next_prot[n, m, ns:]
                        prot_formodes[cell_idx[m], ns:] = next_prot[n, m, ns:]
                        to_keep_for_update[cell_idx[m]] = 1
                        sim_real_idx[target_index] = cell_idx[m]

                        # Re-assign alpha — 
                        if prot_old_snapshot is not None:
                            target_prot = prot_modified[target_index, ns:]
                            distances_prot = np.linalg.norm(prot_old_snapshot - target_prot, 1, axis=1)
                            target_kon = kon_modified[target_index, ns:]
                            distances_kon = np.linalg.norm(kon_old_snapshot - target_kon, 1, axis=1)
                            match = np.argmin((1/G) * distances_prot + ((G-1)/G) * distances_kon)
                            alpha_modified[t_idx+1, offset_init_s + n] = alpha_old[t_idx+1, offset + match]

                offset += N_full[s_idx]

        return prot_modified, prot_formodes, rna_modified, kon_modified, \
                proba_modified, alpha_modified, vect_samples_id_modified, \
                  R_opt_traj, to_keep_for_update
    

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
        nb_cells,
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
    ):
        """
        Alternating optimization of trajectories and network (theta).

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
        y_prot_prev = np.zeros((G_tot, len(times) * N_tot, G_tot))
        for stim_s in range(self.n_stimuli):
            for t_idx, t_i in enumerate(times):
                sl = slice(t_idx * N_tot, (t_idx + 1) * N_tot)
                y_prot_prev[:, sl, stim_s] = self._stim_schedule[t_i][stim_s]
        y_kon = np.zeros_like(y_prot)
        y_rna = np.zeros_like(y_prot)
        y_proba = np.zeros((len(times) * N_tot, G_tot, self.n_networks + 1))
        y_alpha = self.alpha.copy()
        y_samples = np.zeros(len(vect_t_sim), dtype=int)
        R_opt_traj = np.zeros(len(vect_t_sim))

        # === Main loop ===
        while count_end <= count_max:

            if count_end == count_max or n_iter > self.max_iter:
                if N_tot * len(times) > self.batch_size_network:
                    basal, inter, basal_tmp, inter_tmp = inference_network(
                        y_samples, y_kon, y_proba, y_prot, y_prot_prev,
                        ks, n_stimuli=ns, samples_id=samples_id,
                        ref_network=self.ref_network, basal_init=basal, inter_init=inter,
                        basal_ref=basal_ref, inter_ref=inter_ref,
                        proba=self.compute_with_proba, scale=self.scale_pen,
                        weight_prev=weight_prev, loss=self.loss_norm,
                        final=0, constrain_basal_uniform=self.constrain_basal_uniform,
                        hard_forcing_ref=hard_forcing_ref, ref_constraint_pct=ref_constraint_pct,
                        seuil_min_network=self.seuil_min_network)
                    # --- Update kon_theta for alpha ---
                    kon_vector[:, ns:] = self._kon_ref_per_sample(y_prot, ks, inter, basal, samples_id=samples_id, samples_data=y_samples)[:, ns:]
                break
            
            weight_prev = self.weight_prev * min(1, (n_iter-1)/min_n_loops) # Flow matching from second iteration and small at early ones

            # --- Shuffle order of cells for each sample ---
            if n_iter > 1:
                indices_shuffled = [
                    np.random.permutation(N_full[s]) for s in range(len(samples_id))
                ]
                init_cells_full = [
                    init_cells_full[s][indices_shuffled[s]]
                    for s in range(len(samples_id))
                ]

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
            else:
                indices_shuffled = [np.arange(N_full[s]) for s in range(len(samples_id))]

            offset_init = [0] * len(samples_id)
            to_keep_for_update = np.zeros(len(vect_t), dtype=bool)
            to_keep_for_update[vect_t == times[0]] = True
            y_prot_formodes = np.zeros_like(self.modes)
            while not np.array_equal(offset_init, N_full):
                N_tmp = [min(N_samples[s], N_full[s] - offset_init[s]) for s in range(len(samples_id))]
                init_cells = [init_cells_full[s][offset_init[s]:offset_init[s] + N_tmp[s]] for s in range(len(samples_id))]
                y_prot, y_prot_formodes, y_rna, y_kon, y_proba, y_alpha, y_samples, R_opt_traj, to_keep_for_update = \
                    self.estimate_trajectories_given_model(
                        vect_t, times, vect_samples_id, samples_id,
                        data_rna, y_prot, y_prot_formodes, y_kon, y_rna, y_proba, y_alpha, y_samples,
                        basal, inter, s1, ks, nb_cells, init_cells,
                        R_opt_traj, to_keep_for_update,
                        offset_init=offset_init,
                        N_full=N_full, N_samples=N_tmp,
                        n_iter=n_iter + min_n_loops * min(1, 1 - compute_theta + hard_forcing_ref),
                        intensity_prior=intensity_prior * compute_theta * (1 - hard_forcing_ref)
                    )

                offset_init = [offset_init[s] + N_tmp[s] for s in range(len(samples_id))]

            y_prot_prev[:, :N_tot, ns:] = y_prot[:N_tot, ns:]
            self.R_opt = R_opt_traj
    
            # --- Evaluate error before and after inference ---
            error = self._count_errors_per_sample(y_prot, y_kon, y_proba, ks, inter, basal,
                                                   samples_id=samples_id, samples_data=y_samples)
            if compute_theta and len(times) > 1:
                if self.weight_prev > 0:
                    modes = self.adaptive_shrinkage(y_rna[:, ns:] * s1, y_kon[:, ns:]) / s1
                    for cnt, time in enumerate(times[:-1]):
                        delta_t = times[cnt + 1] - time
                        for n in range(N_tot):
                            idx_prev = N_tot * cnt + n
                            idx_next = N_tot * (cnt + 1) + n
                            alpha_n = y_alpha[cnt, n]

                            for g in range(ns, G_tot):
                                alpha_n_mod = min(alpha_n[g-ns]+.1, 1) # +.1 for letting some time for mode to stabilize
                                y_prot_prev[g, idx_next, ns:] = find_next_prot(self.d[1, ns:],
                                                        y_prot[idx_prev, ns:],
                                                        y_rna[idx_prev, ns:] * self.scale_proteins,
                                                        y_rna[idx_next, ns:] * self.scale_proteins,
                                                        modes[idx_prev],
                                                        modes[idx_next],
                                                        np.minimum(alpha_n / alpha_n_mod, 1),
                                                        s1, delta_t * alpha_n_mod
                                                    )
                                
                # --- Subsample at most batch_size cells per timepoint ---
                batch_per_time = 1 + self.batch_size_network // len(times)
                _sel = np.concatenate([
                                (lambda idx: np.random.choice(idx, size=batch_per_time, replace=False) 
                                if len(idx) > batch_per_time else idx)(np.where(vect_t_sim == t)[0])
                    for t in times
                ])

                basal, inter, basal_tmp, inter_tmp = inference_network(
                    y_samples[_sel], y_kon[_sel], y_proba[_sel], y_prot[_sel], y_prot_prev[:, _sel, :],
                    ks, n_stimuli=ns, samples_id=samples_id,
                    ref_network=self.ref_network, basal_init=basal, inter_init=inter,
                    basal_ref=basal_ref, inter_ref=inter_ref,
                    proba=self.compute_with_proba, scale=self.scale_pen,
                    weight_prev=weight_prev, loss=self.loss_norm,
                    final=0, constrain_basal_uniform=self.constrain_basal_uniform,
                    hard_forcing_ref=hard_forcing_ref, ref_constraint_pct=ref_constraint_pct,
                    seuil_min_network=self.seuil_min_network)

            error_2 = self._count_errors_per_sample(y_prot, y_kon, y_proba, ks, inter, basal,
                                                    samples_id=samples_id, samples_data=y_samples)
            errors.append(error_2)
            self.loss_trajectory.append(error_2)
            self.theta_trajectory.append(inter_tmp)

            if verb:
                print(f"{n_iter}", f"{count_end} | Errors (before, after): {error:.5f}, {error_2:.5f} | alpha mean: {np.mean(y_alpha[0]):.4f}")

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
            kon_vector[:, ns:] = self._kon_ref_per_sample(y_prot, ks, inter, basal, samples_id=samples_id, samples_data=y_samples)[:, ns:]

            # --- Update alphas ---
            modes = self.adaptive_shrinkage(y_rna[:, ns:] * s1, y_kon[:, ns:]) / s1
            if len(times) > 1:
                for cnt, time in enumerate(times[:-1]):
                    y_alpha[cnt] = inference_alpha(
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
                            n_pas = max(self.n_pas, int(times[cnt + 1] - time)),
                            samples_data=y_samples[vect_t_sim == time],
                            stim_vals=np.asarray(self._stim_schedule[times[cnt + 1]], dtype=float),
                            scale_proteins=self.scale_proteins
                        )
            
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
                reg = 1 - (1/min_n_loops) * (n_iter - 1)
                weight_prob = max(.96**(n_iter-1), .1) # the weight of the network increases slowly because it aims to get the right attribution given probabilities that are close
                
                def run_main_loop_for_gene(g, temporal=self.temporal_basins):
                    l_max = 1 + np.argmax(ks[g, :])
                    obj = np.zeros((n_cells, l_max), dtype=float)
                    obj[:, :] = ks[g, :l_max][None, :]
                    tmp_proba = np.zeros_like(self.proba[:, g])
                    tmp_modes = np.zeros_like(self.modes[:, g])
                    if temporal:
                        for t_i in times:
                            indices = (vect_t == t_i)
                            tmp_proba_i = np.zeros_like(self.proba[indices, g])
                            tmp_modes_i = np.zeros_like(self.modes[indices, g])
                            proba_i = self.proba_init[indices, g, :l_max].copy()
                            proba_i /= np.max(proba_i, axis=1, keepdims=True)
                            obj_i = obj[indices].copy()
                            n_cells_i = np.sum(indices)
                            mu = np.ones(n_cells_i)/n_cells_i
                            nu = self.pi_init[g - ns][t_i][:l_max] * self.force_basins + np.sum(
                                                        proba_i[:, :l_max], axis=0) * (1 - self.force_basins)
                            nu /= np.sum(nu)
                            diff_k = np.maximum(1 - np.abs(kon_vector_formodes[indices, g, None] - obj_i), 1e-3)
                            diff_k /= np.max(diff_k, axis=1, keepdims=True)
                            dist = - (np.log(proba_i) + (1 - weight_prob) * 
                                      to_keep_for_update[indices, None] * np.log(diff_k))
                            dist = np.clip(dist, 0, 100)
                            if reg > 1/min_n_loops and compute_theta: 
                                try: 
                                    coupling = ot.bregman.sinkhorn(mu, nu, dist, reg=reg, numItermax=int(10000/reg), stopThr=self.stopThr_init*10)
                                except Exception: 
                                    coupling = ot.emd(mu, nu, dist, numItermax=int(1e-7))
                            else: coupling = ot.emd(mu, nu, dist, numItermax=int(1e-7))
                            idx = np.argmax(coupling, axis=1)
                            for cell in range(n_cells_i):
                                tmp_proba_i[cell, idx[cell]] = 1
                                tmp_modes_i[cell] = obj_i[cell, idx[cell]]
                            tmp_proba[indices, :] = tmp_proba_i[:, :]
                            tmp_modes[indices] = tmp_modes_i[:]
                    else:
                        proba = self.proba_init[:, g, :l_max].copy()
                        proba /= np.max(proba, axis=1, keepdims=True)
                        mu = np.ones(n_cells)/n_cells
                        nu = np.sum([self.pi_init[g - ns][t_i] * np.sum(vect_t == t_i)/n_cells
                                                    for t_i in times], axis=0)[:l_max] * self.force_basins + np.sum(
                                                        proba[:, :l_max], axis=0) * (1 - self.force_basins)
                        nu /= np.sum(nu)
                        diff_k = np.maximum(1 - np.abs(kon_vector_formodes[:, g, None] - obj), 1e-3)
                        diff_k /= np.max(diff_k, axis=1, keepdims=True)
                        dist = - (np.log(proba) + (1 - weight_prob) * 
                                  to_keep_for_update[:, None] * np.log(diff_k))
                        dist = np.clip(dist, 0, 100)
                        if reg > 1/min_n_loops and compute_theta: 
                            try: 
                                coupling = ot.bregman.sinkhorn(mu, nu, dist, reg=reg, numItermax=int(10000/reg), stopThr=self.stopThr_init*10)
                            except Exception: 
                                coupling = ot.emd(mu, nu, dist, numItermax=int(1e-7))
                        else: 
                            coupling = ot.emd(mu, nu, dist, numItermax=int(1e-7))
                        idx = np.argmax(coupling, axis=1)
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
                        l_max = 1 + int(np.argmax(ks[g, :]))
                        ko_cells = kov_cell_mask[:, g] < 0
                        ov_cells = kov_cell_mask[:, g] > 0
                        if np.any(ko_cells):
                            self.proba[ko_cells, g, :] = 0
                            self.proba[ko_cells, g, 0] = 1
                            self.modes[ko_cells, g] = ks[g, 0]
                        if np.any(ov_cells):
                            self.proba[ov_cells, g, :] = 0
                            self.proba[ov_cells, g, l_max - 1] = 1
                            self.modes[ov_cells, g] = ks[g, l_max - 1]

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

        # Harissa: continuous adaptive_shrinkage burst-rate estimates, used in
        # refine_network_degradations in place of discrete mode assignments (kon_beta).
        self.kon_beta_harissa = y_kon.copy()
        self.kon_beta_harissa[:, ns:] = (
            self.adaptive_shrinkage(y_rna[:, ns:] * s1, y_kon[:, ns:]) 
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

        # --- Initialization ---
        data_rna = self._parse_input(data, time_key)
        if stimulus_schedule is not None or self._stim_schedule is None:
            self._stim_schedule = self._build_stimulus_schedule(
                np.sort(np.unique(data_rna[:, 0])), stimulus_schedule)
        G_tot = data_rna.shape[1]
        vect_t = data_rna[:, 0]
        ns = self.n_stimuli

        # --- Adapt ref_network ---
        self.ref_network = np.maximum(self.prior_network_pen, self.ref_network)
        self.ref_network[:ns, :] = self.stimulus
        print(self._stim_schedule)
        for g in range(ns, G_tot):
            l_max = len(np.unique(self.modes[:, g]))
            if l_max < 2:
                self.ref_network[g, :], self.ref_network[:, g] = 0, 0
            if l_max > len(np.unique(self.a[:-1, g])):
                self.compute_with_proba = 0

        # Auto-extract vect_samples_id from AnnData if not provided
        try:
            import anndata
            if isinstance(data, anndata.AnnData) and vect_samples_id is None and 'dataset_id' in data.obs:
                vect_samples_id = data.obs['dataset_id'].values
        except ImportError:
            pass

        # --- Load per-cell growth rates and cell types from AnnData ---
        self._prolif_rate = None
        self._death_rate = None
        self._cell_types = None
        self._transition_rates = None
        self._transition_type_labels = None
        try:
            import anndata as _ad
            if isinstance(data, _ad.AnnData):
                if 'prolif_rate' in data.obs:
                    self._prolif_rate = data.obs['prolif_rate'].values.astype(float)
                if 'death_rate' in data.obs:
                    self._death_rate = data.obs['death_rate'].values.astype(float)
                for _ct_col in ('cell_type', 'cell_types', 'celltype'):
                    if _ct_col in data.obs:
                        self._cell_types = data.obs[_ct_col].values.astype(str)
                        break
        except ImportError:
            pass

        if transition_rates is not None:
            if hasattr(transition_rates, 'index') and hasattr(transition_rates, 'to_numpy'):
                self._transition_type_labels = list(transition_rates.index.astype(str))
                _Tr = transition_rates.to_numpy().astype(float)
            else:
                _Tr = np.asarray(transition_rates, dtype=float)
            self._transition_rates = np.clip(_Tr, 0.0, None)  # store raw non-negative rates

        # If no sample ID provided, assume one global sample
        if vect_samples_id is None:
            vect_samples_id = np.zeros_like(vect_t)

        # Unique time points and sample IDs
        times = np.sort(np.unique(vect_t))
        samples_id = np.sort(np.unique(vect_samples_id))

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
            minimal_repetition_choice(nb_cells[s, 0], N_full[s])
            for s in range(len(samples_id))
        ]

        # --- Extract kinetic parameters ---
        ks = (self.a[:-1] / np.clip(np.max(self.a[:-1], axis=0), EPS, None)).T
        s1 = self.fact_simple * (self.a[-1, ns:] / np.maximum(np.max(self.a[:-1, ns:], axis=0), EPS))

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
            nb_cells=nb_cells,
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
            

    def estimate_trajectories(self, y_prot, times, d1, N=100, kon_beta=None, s=None):
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
        """
        if kon_beta is None:
            kon_beta = self.kon_beta
        if s is None:
            s = self.scale_proteins
        ns = self.n_stimuli
        prot_modified = y_prot.copy()
        N_tot, G = prot_modified.shape
        prot_modified_prev = np.ones((G, N_tot, G), dtype=float) 
        prot_modified_prev[:, :, :] = prot_modified[None, :, :]

        for cnt, time in enumerate(times[:-1]):
            delta_t = times[cnt + 1] - time

            for n in range(N):
                idx_prev = N * cnt + n
                idx_next = N * (cnt + 1) + n
                alpha_n = self.alpha[cnt, n]

                prot_modified[idx_next, ns:] = find_next_prot(
                    d1,
                    prot_modified[idx_prev, ns:],
                    kon_beta[idx_prev, ns:],
                    kon_beta[idx_next, ns:],
                    kon_beta[idx_prev, ns:],
                    kon_beta[idx_next, ns:],
                    alpha_n,
                    s,
                    delta_t
                )

                if self.weight_prev > 0:
                    for g in range(ns, G):
                        alpha_n_mod = min(alpha_n[g-ns]+.1, 1)
                        prot_modified_prev[g, idx_next, ns:] = find_next_prot(d1,
                                                prot_modified[idx_prev, ns:],
                                                kon_beta[idx_prev, ns:],
                                                kon_beta[idx_next, ns:],
                                                kon_beta[idx_prev, ns:],
                                                kon_beta[idx_next, ns:],
                                                np.minimum(alpha_n / alpha_n_mod, 1),
                                                s,
                                                delta_t * alpha_n_mod
                                            )


        return prot_modified, prot_modified_prev
    

    def select_cells_to_use(self):

        n_samples = len(np.unique(self.samples_data))
        N_t = np.sum(self.times_data == 0)
        cells_to_use = np.zeros_like(self.times_data, dtype=int)
        times = np.unique(self.times_data)

        for s in range(n_samples):
            # cellules appartenant à l’échantillon s au temps 0
            idx_first = (self.samples_data == s) & (self.times_data == 0)
            idx_first_indices = np.where(idx_first)[0]
            N_s = len(idx_first_indices)

            # tirage aléatoire d’un sous-ensemble
            n_pick = min(N_s, self.batch_size_degradations)
            if n_pick == 0:
                continue

            chosen_idx = np.random.choice(idx_first_indices, n_pick, replace=False)

            # marquer les mêmes cellules à travers tous les temps
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
        if basal.ndim < 3:
            return kon_ref_vector(y_prot, ks, inter, basal)
        sd = samples_data if samples_data is not None else self.samples_data
        if samples_id is None:
            samples_id = np.sort(np.unique(sd))
        out = np.zeros((y_prot.shape[0], y_prot.shape[1]))
        for s_idx, s in enumerate(samples_id):
            mask = (sd == s)
            if not np.any(mask):
                continue
            basal_s = basal[min(s_idx, basal.shape[0] - 1)]
            out[mask] = kon_ref_vector(y_prot[mask], ks, inter, basal_s)
        return out

    def _count_errors_per_sample(self, y_prot, kon_beta, proba_traj, ks, inter, basal,
                                  samples_id=None, samples_data=None):
        """
        Weighted-average count_errors respecting per-sample basal.
        When basal is 2-D, delegates to count_errors directly.

        samples_data : per-cell sample assignment; defaults to self.samples_data.
        """
        if basal.ndim < 3:
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
            basal_s = basal[min(s_idx, basal.shape[0] - 1)]
            err_s = count_errors(y_prot[mask], kon_beta[mask], proba_traj[mask],
                                 ks, basal_s, inter,
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

        times = np.sort(np.unique(self.times_data))
        N_tot = np.sum(self.times_data == 0)

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
            if l_max > len(np.unique(self.a[:-1, g])): # compute with proba = 0 si more modes than ks
                self.compute_with_proba = 0

        k1 = np.max(self.a[:-1], axis=0)
        ks = (self.a[:-1] / k1).T
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

        if self.simulate_full_with_harissa:
            y_prot, y_prot_prev = self.estimate_trajectories(self.prot, times, self.d[1, ns:], N=N_tot, kon_beta=self.kon_beta_harissa, s=1)
            error = self._count_errors_per_sample(y_prot, self.kon_beta, self.proba_traj, ks,
                                                inter, basal, samples_id=samples_id, samples_data=self.samples_data)
        else:
            y_prot, y_prot_prev = self.estimate_trajectories(self.prot, times, self.d[1, ns:], N=N_tot, kon_beta=self.kon_beta, s=self.scale_proteins)
            error = self._count_errors_per_sample(y_prot, self.kon_beta, self.proba_traj, ks,
                                                inter, basal, samples_id=samples_id, samples_data=self.samples_data)

        # inference_network returns (n_samples, G_tot, n_networks) for basal
        _final_call = 0 if (self.hard_forcing_ref or self.inter_simul_ref is not None) else 1
        basal, inter, _, _ = inference_network(
            self.samples_data, self.kon_beta, self.proba_traj,
            y_prot, y_prot_prev, ks, n_stimuli=ns, proba=self.compute_with_proba,
            ref_network=self.ref_network, basal_init=basal_ref, inter_init=inter_ref,
            basal_ref=basal_ref, inter_ref=inter_ref,
            scale=self.scale_pen * 2, # # slightly stronger regularization for network
            weight_prev=self.weight_prev, loss=self.loss_norm, final=_final_call,
            samples_id=samples_id,
            constrain_basal_uniform=self.constrain_basal_uniform,
            hard_forcing_ref=self.hard_forcing_ref, ref_constraint_pct=self.ref_constraint_pct,
            seuil_min_network=self.seuil_min_network,
        )

        ### filter_edges
        if self.filter_network:
            inter, _ = filter_network(len(times), N_tot, y_prot, ks, basal, inter, 
                                      samples_data=self.samples_data, seuil_variations=self.seuil_min_network)

        error_corrected = self._count_errors_per_sample(y_prot, self.kon_beta, self.proba_traj, ks,
                                                        inter, basal, samples_id=samples_id, samples_data=self.samples_data)
        if verb:
            print("[refine_network_degradations] ratio errors", error, error_corrected)

        # Pre-scale basal/inter to best fit kon_beta across all cells before ODE inference.
        scale_theta_pre = fit_scale_theta(
            y_prot, self.kon_beta, basal, inter,
            ks.T * self.scale_proteins, ns, samples_data=self.samples_data,
        )
        basal *= scale_theta_pre       
        inter *= scale_theta_pre  

        print(np.mean(scale_theta_pre))

        self.prot = y_prot
        kon_vector = self.kon_beta.copy()
        kon_vector[:, ns:] = self._kon_ref_per_sample(y_prot, ks, inter, basal, samples_data=self.samples_data)[:, ns:]
        self.kon_theta = kon_vector

        # --- Train proliferation MLP on R_opt (after prot recomputation) ---
        if self.recompute_proliferations and self.R_opt is not None:
            if verb:
                print("[refine_network_degradations] Training ProliferationMLP on R_opt...")
            self.prolif_network = train_proliferation_mlp(
                self.prot, self.R_opt, ns=ns, verb=verb,
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
                    self.prot, self.kon_beta_harissa, self.kon_beta * self.scale_proteins, ns, seuil=self.seuil
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

            cells_to_use = self.select_cells_to_use()
            if not self.use_temporal_degradations:
                d1, scale_theta = inference_degradation_prot(
                            self.prot[cells_to_use == 1],
                            self.times_data[cells_to_use == 1],
                            basal,   # 3-D (n_samples, G, n_networks) — triggers per-sample ODE
                            inter, ks.T * self.scale_proteins,
                            d=self.d[1], lr=1e-2,
                            n_stimuli=ns, stim_schedule=self._stim_schedule,
                            scale_proteins = self.scale_proteins,
                            samples_data=self.samples_data[cells_to_use == 1],
                            kon_mlp=self.kon_mlp,
                            lambda_scale=self.lambda_scale,
                            lambda_deg=self.lambda_deg1,
                            lambda_mlp=self.lambda_mlp,
                            g_obs_train=g_obs_all[cells_to_use == 1] if g_obs_all is not None else None)
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
                def run_main_inference_degradation_prot(t):
                    idx = ((self.times_data == times[t]) | (self.times_data == times[t+1])) & (cells_to_use == 1)
                    return inference_degradation_prot(
                        self.prot[idx], self.times_data[idx],
                        basal,   # 3-D
                        inter, ks.T * self.scale_proteins,
                        d=self.d[1], lr=1e-2,
                        n_stimuli=ns, stim_schedule=self._stim_schedule,
                        scale_proteins = self.scale_proteins,
                        samples_data=self.samples_data[idx],
                        kon_mlp=self.kon_mlp,
                        lambda_scale=self.lambda_scale,
                        lambda_deg=self.lambda_deg1,
                        lambda_mlp=self.lambda_mlp,
                        g_obs_train=g_obs_all[idx] if g_obs_all is not None else None)

                results = Parallel(n_jobs=-1)(
                    delayed(run_main_inference_degradation_prot)(t) for t in range(0, len(times)-1)
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
                    ks.T,   # ks    : (n_modes, G)
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
                    ks.T * self.scale_proteins,
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

        ### Rescale kz
        rescale = np.ones(prot_modified.shape[1])
        rescale[ns:] = np.max(self.a[:-1, ns:], axis=0)
        kz = ks * rescale.reshape(ks.shape[0], 1)

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

        # Proliferation: build a callable wrapper around self.prolif_network once.
        _prolif_fn = None
        if self.simulate_with_proliferation and self.prolif_network is not None:
            _prolif_fn = self.prolif_network.predict  # (1, n_proteins) -> (1,)

        for cnt, time in enumerate(times[start_time:-1], start=start_time):
            delta_t = times[cnt + 1] - time

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

            cur_stim_vals = self._stim_schedule[times[cnt + 1]] * self.scale_proteins

            def run_main_loop_for_cell(n, _basal_cells=basal_cells, _basal_t_cnt=basal_t[cnt], 
                                       _stim_vals=cur_stim_vals):
                basal_n = _basal_cells[n] if _basal_cells is not None else _basal_t_cnt
                if self.simulation_stochastic:
                    return simulate_next_prot_pdmp(
                            degradations[1, :],
                            kz * degradations[0][:, None],
                            rescale * (degradations[0, :] / degradations[1, :]),
                            basal_n, inter_t[cnt], delta_t,
                            self.scale_proteins, P0=prot_modified[start_index + n, :],
                            ns=ns, stim_vals=_stim_vals,
                        )
                else:
                    return simulate_next_prot_ode(
                        degradations[1, :], ks,
                        basal_n, inter_t[cnt], delta_t,
                        self.scale_proteins, P0=prot_modified[start_index + n, :],
                        ns=ns, stim_vals=_stim_vals
                    )

            results = Parallel(n_jobs=-1)(
            delayed(run_main_loop_for_cell)(n) for n in range(0, N)
            )

            for idx, n in enumerate(range(0, N)):
                # result.p[-1] has shape (G_tot-1,): indices 1..G_tot-1 of the state
                # (index 0 = stim1 is excluded by the simulation).
                # For ns>1, indices 1..ns-1 are stim2..stimN — skip them.
                prot_modified[end_index + n, ns:] = results[idx].p[-1][ns - 1:]

            # --- Branching process resampling (trapezoidal forward+backward) ---
            # Trapezoidal rule: log_weight_n = (R(P_start_n) + R(P_end_n)) / 2 * delta_t
            # R_opt was estimated with factor 2 (WOT corrects both source AND target
            # marginals). During simulation we know both endpoints, so we replicate
            # the same forward+backward structure: each endpoint contributes R·Δt/2.
            # Multinomial resampling avoids the catastrophic variance of Poisson(λ≈1)
            # which kills ~37% of cells per step even for R ≈ 0.
            if _prolif_fn is not None:
                P_start = prot_modified[start_index:start_index + N, ns:]
                P_end   = prot_modified[end_index:end_index + N, ns:]
                R_start = _prolif_fn(P_start)  # (N,)
                R_end   = _prolif_fn(P_end)    # (N,)
                log_weights = (R_start + R_end) * 0.5 * delta_t
                weights = np.exp(log_weights - log_weights.max())  # numerically stable
                weights /= weights.sum()
                src_for_new = np.random.choice(N, N, replace=True, p=weights)
                prot_modified[end_index:end_index + N, ns:] = P_end[src_for_new]

            # Set stim values at step boundary from schedule (authoritative source).
            prot_modified[end_index:end_index + N, :ns] = cur_stim_vals

            basal_t_cnt_2d = basal_t[cnt].mean(axis=0) if basal_t.ndim == 4 else basal_t[cnt]
            kon_vector[end_index:end_index+N, ns:] = kon_ref_vector(
                prot_modified[end_index:end_index+N, :], ks, inter_t[cnt], basal_t_cnt_2d)[:, ns:]

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
        h_model.a  = self.a      

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

        rescale = np.ones(G_tot)
        rescale[ns:] = np.max(self.a[:-1, ns:], axis=0)

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
                delayed(run_harissa_cell)(n) for n in range(N)
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

        if self.simulate_full_with_harissa:
            self.scale_proteins = 1

        times.sort()
        times_train = np.sort(np.unique(self.times_data))
        if stimulus_schedule is not None or self._stim_schedule is None:
            # Pass times_ref=times_train so the schedule is step-function interpolated
            # when simulation times differ from training times (e.g. times_to_simulate.txt).
            self._stim_schedule = self._build_stimulus_schedule(
                np.sort(np.unique(times)), stimulus_schedule, times_ref=times_train)
        N = np.sum(self.times_data == 0)
        ks = (self.a[:-1] / np.max(self.a[:-1], axis=0)).T

        samples_data_sim = (self.samples_data[:N]
                            if (self.basal is not None and self.basal.ndim == 3
                                and self.samples_data is not None)
                            else None)

        # Harissa PDMP always forces stimulus=1; only use it when the schedule is the
        # default (ns==1, stimulus active at every non-initial timepoint).
        t_min = min(self._stim_schedule.keys())
        _stim_is_default = (
            self.n_stimuli == 1
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



    def fit_mixture_test(self, data_rna, ks, c, verb=False):
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
                    pi=pi_prior, pi_zero=pi_zero_g, zi=zi_flag
                )
            if self.transform_proba:
                tmp = np.exp(self.transform_proba * ((len(ks)-1))*np.log(G_tot)*(proba - 1/len(ks))) # self.transform_proba is the typical size of parameters that are expected, np.log(G) the number of regulators), and the difference to the mean max proba scales the protein level
                tmp /= (1 + tmp)
                tmp /= np.sum(tmp, 1).reshape(N_cells, 1)
                for cell in range(N_cells):
                    if np.max(proba[cell]) > np.max(tmp[cell]):
                        tmp[cell, :] = proba[cell, :]
                proba[:, :] = tmp[:, :]

            if self.update_modes or self.loss_norm == 'CE':
                tmp = np.zeros_like(proba)
                pi_g_train = (training_pi_init[g_idx]
                              if training_pi_init is not None and g_idx < len(training_pi_init)
                              else None)
                if self.temporal_basins:
                    for t_i in np.unique(vect_t):
                        indices = (vect_t == t_i)
                        tmp_proba_i = np.zeros_like(proba[indices])
                        proba_i = proba[indices]
                        n_cells_i = np.sum(indices)
                        mu = np.ones(n_cells_i)/n_cells_i
                        nu = np.asarray(pi_g_train[t_i], dtype=float)[:ng] * self.force_basins + np.sum(
                            proba_i, axis=0) * (1 - self.force_basins) if pi_g_train is not None else np.sum(proba_i, axis=0)
                        nu /= np.sum(nu)
                        dist = - np.log(proba_i)
                        coupling = ot.bregman.sinkhorn(mu, nu, dist, reg=1, numItermax=10000)
                        idx = np.argmax(coupling, axis=1)
                        for cell in range(n_cells_i):
                            tmp_proba_i[cell, idx[cell]] = 1
                        tmp[indices, :] = tmp_proba_i[:, :]
                else:
                    mu = np.ones(N_cells)/N_cells
                    nu = np.sum([np.asarray(pi_g_train.get(t_i, np.ones(ng)/ng), dtype=float)[:ng]
                                     * np.sum(vect_t == t_i)/N_cells
                                     for t_i in np.unique(vect_t)], axis=0) * self.force_basins + np.sum(
                                         proba, axis=0) * (1 - self.force_basins) if pi_g_train is not None else np.sum(proba, axis=0)
                    nu /= np.sum(nu)
                    dist = - np.log(proba)
                    coupling = ot.bregman.sinkhorn(mu, nu, dist, reg=1, numItermax=10000)
                    idx = np.argmax(coupling, axis=1)
                    for cell in range(N_cells):
                        tmp[cell, idx[cell]] = 1

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
        frequency_modes_smooth /= scale_max
        self.pi_zinb = training_pi_zinb   # keep training ZINB zero-inflation values
        self.modes = frequency_modes_smooth
        self.proba = frequency_proba
        self.proba_init = frequency_proba_init
        self.pi_init = pi_init_test



    def infer_test(self, data, vect_samples_id=None, verb=True, stimulus_schedule=None,
                   basal_ref=None, transition_rates=None, time_key='time'):
        """
        Run inference pipeline on test data: kon estimation, trajectory inference, and alpha initialization.

        basal_ref : (n_samples, G_tot, n_networks) array or None
            Per-sample KO/OV prior (±100 entries) used to build kov_cell_mask.
        transition_rates : DataFrame or array or None
            Cell-type transition rate matrix for OT cost adjustment.
        """
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

        # --- Load per-cell growth rates and cell types from AnnData ---
        self._prolif_rate = None
        self._death_rate = None
        self._cell_types = None
        self._transition_rates = None
        self._transition_type_labels = None
        try:
            import anndata as _ad
            if isinstance(data, _ad.AnnData):
                if 'prolif_rate' in data.obs:
                    self._prolif_rate = data.obs['prolif_rate'].values.astype(float)
                if 'death_rate' in data.obs:
                    self._death_rate = data.obs['death_rate'].values.astype(float)
                for _ct_col in ('cell_type', 'cell_types', 'celltype'):
                    if _ct_col in data.obs:
                        self._cell_types = data.obs[_ct_col].values.astype(str)
                        break
        except ImportError:
            pass

        if transition_rates is not None:
            if hasattr(transition_rates, 'index') and hasattr(transition_rates, 'to_numpy'):
                self._transition_type_labels = list(transition_rates.index.astype(str))
                _Tr = transition_rates.to_numpy().astype(float)
            else:
                _Tr = np.asarray(transition_rates, dtype=float)
            self._transition_rates = np.clip(_Tr, 0.0, None)  # store raw non-negative rates

        times = np.sort(np.unique(vect_t))
        kz = self.a[:-1]
        c = self.a[-1]

        self.fit_mixture_test(data_rna, kz, c)

        print('[infer_test] Mean proba = ', np.mean(np.max(self.proba[:, ns:, :], axis=-1)))

        ks = (kz / np.max(self.a[:-1], axis=0)).T
        s1 = self.fact_simple * self.a[-1, ns:] / np.maximum(np.max(self.a[:-1, ns:], axis=0), EPS)

        samples_id = np.sort(np.unique(vect_samples_id))

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
                        l_max = 1 + int(np.argmax(ks[g, :]))
                        ko_cells = kov_cell_mask[:, g] < 0
                        ov_cells = kov_cell_mask[:, g] > 0
                        if np.any(ko_cells):
                            self.proba[ko_cells, g, :] = 0
                            self.proba[ko_cells, g, 0] = 1
                            self.modes[ko_cells, g] = ks[g, 0]
                        if np.any(ov_cells):
                            self.proba[ov_cells, g, :] = 0
                            self.proba[ov_cells, g, l_max - 1] = 1
                            self.modes[ov_cells, g] = ks[g, l_max - 1]

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
            minimal_repetition_choice(nb_cells[s, 0], N_full[s])
            for s in range(len(samples_id))
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
            nb_cells=nb_cells,
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
        )


    def fit(self, data_rna, intensity_prior=100, refilter=5.0, max_iter_kinetics=100, verb=True):

        self.fit_mixture(data_rna, min_components=2, max_components=2, refilter=refilter, max_iter_kinetics=max_iter_kinetics)
        self.fit_network(data_rna, intensity_prior=intensity_prior, verb=verb)
        # self.refine_network_degradations()