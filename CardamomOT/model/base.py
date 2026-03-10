
"""
Core implementation of the NetworkModel used for inference and simulation.

This module defines the :class:`NetworkModel` class which encapsulates
parameters, state, and algorithms for fitting gene regulatory networks
from single-cell expression data, performing stochastic or deterministic
simulations, and managing mixture models. All documentation and comments
are maintained in English.
"""
import numpy as np
import ot
from scipy.stats import gamma, rankdata
import seaborn as sns
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from joblib import Parallel, delayed
from ..inference import (inference_network, filter_network,
                        minimal_repetition_choice, find_next_prot, my_otdistance, count_errors, kon_ref_vector, inference_alpha, 
                        NegativeBinomialMixtureEM, 
                        simulate_next_prot_ode, simulate_next_prot_pdmp, 
                        inference_degradation_prot, inference_epsilon_temporal)

np.set_printoptions(precision=3, suppress=True)

class NetworkModel:
    """
    Encapsulates the state and parameters of a regulatory network.

    The class stores kinetic, mixture and network parameters as well as
    trajectories produced during inference. It provides methods for
    initialization, calibration and simulation used by the higher-level
    pipeline script.
    """
    def __init__(self, n_genes=None, times=None):
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
        # Network parameters
        self.kon_theta = None
        self.a = None
        self.ref_network = None
        self.basal = None
        self.inter = None
        self.basal_t = None
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

        ### Default behaviour

        ## Compute mixture parameters
        self.hard_em = 1 # Do we initialize with a hard_em ?
        self.preserve_mean_values = 1 # Do we ensure temporal constraints when fitting the basins in the hard_em ?
        self.mean_forcing_em = 0.5 # at which point we force the mean correction: the higher the more
        self.force_basins = 1 # Do we want to ensure the means to be preserved by the NB mixture ? It may not preserve multistability
        self.temporal_basins = 1 # Is it preserved temporally ?
        self.transform_proba = 0 # Do we want to force probas to be steep for compatibility with sigmoid model?
        self.seuil = 1e-2 # minimum for beta mixture parameters (second parameters)

        ## Infer network
        self.n_networks = 1
        self.adapt_size_network = 0
        # Loop for the inference
        self.n_loops = 10 # minimal number of iterations in inference loops
        self.count_max = 5 # Stopping criteria
        self.max_iter = 40 # max iteration for main loop
        # Trajectory inference with OT
        self.stopThr_init = 1e-7 # initial tolerance for sinkhorn algorithm
        self.batch_size = 1024 # Maximum number of cells used per time point per sample for inference.
        self.unbalanced_reg = 5 # Unbalanced regularization parameter for UOT if > 0, OT if 0
        self.init_entropic_noise = 1.5 # Initial entropic penalization for OT
        self.quant_samples = .8 # Quantile of cells number per sample to use for inference
        # General parameters to calibrate protein reconstruction
        self.scale_proteins = 1 # Eventually rescale protein values when switching to simple=1 (recommended:1)
        self.fact_simple = 2 # slight transformation for constrative modes in learning phase
        # Network inference with scipy
        self.loss_norm = 'CE'
        self.scale_pen = 20 # Error that is expected = 1/scale_pen
        self.compute_with_proba = 0 # Determine if compute with proba or kon values in network inference (recommended:1)
        self.weight_prev = .4 # max = .5 to not withdrawn the inference on timepoints, allows the calibration to incorporate some "flow-matching" method
        # Inference of alpha = switch moment between each timepoint and modes
        self.update_modes = 1
        self.alpha_threshold= .4 # max = .5 to update alpha at least for important transition
        # Penalization/prior information
        self.stimulus = 1.0 # 1 if we simulate with a stimulus. If not we can penalize the stimulus with a value between 1 and 0: 0 = no sitmulus
        self.prior_network_pen = 1.0 # 1 if we don't use prior information. If not we can penalize the non-existing age in prior network with values between 1 and 0: 0 = impossible edge
        # Filtering
        self.filter_network = 0 # Do we filter the network ? It also builds a temporal network using the filter criterium

        ## Compute degradations after inference
        self.recompute_degradations = 1 # Do we want to recompute degradation rates for simulations ?
        self.nb_traj_for_degradations_inference = 200 # number of trajectories to take to make the inference (slow without gpu)
        self.use_temporal_degradations = 1 # If so, compute temporal degradation rates for simulations ?

        ## Simulations
        self.simulation_stochastic = 1 # 1 if we simulate Bursty-like proteins, 0 if deterministic limit for proteins
        self.finish_by_determinist = 1 # 1 if we simulate with deterministic limit for the last timepoint
        self.min_ratio = 1
        self.max_ratio = 50

        if n_genes is not None:
            G = n_genes + 1 # Genes + stimulus
            # Default degradation rates
            self.d = np.zeros((2,G))
            self.d[0] = np.log(2)/9 # mRNA degradation rates
            self.d[1] = np.log(2)/46 # protein degradation rates
            # Default network parameters
            self.a = np.zeros((3,G))
            self.basal = np.zeros((G, 1))
            self.inter = np.zeros((G, G, 1))
            self.ref_network = np.ones((G, G, 1))
        

    def core_binarization(self, data_rna, gene_names, vect_t, G_tot, min_components=1, max_components=5, refilter=0, max_iter_kinetics=100, cell_rd=None, verb=True):
        """
        Parameters
        ----------
        cell_rd : (N_cells,) array or None
            Read depth scaling factors per cell (median = 1), typically
            stored in ``adata.obs['rd']``.  If ``None``, the standard
            Negative Binomial model without scaling is applied.
        """

        # Get kinetic parameters
        N_cells = np.size(data_rna, 0)
        N_cells_0 = np.sum(vect_t == 0)
        frequency_modes_smooth = np.zeros_like(data_rna, dtype="float")
        frequency_modes_smooth[N_cells_0:, 0] = 1

        ks = []
        proba_init = []
        proba_modif = []
        pi_init = []
        c = np.ones(G_tot)
        pi_zeros = np.ones(G_tot-1)
        n_components = 0

        def run_main_loop_for_gene(g, temporal=self.temporal_basins):
            if verb: print("Calibrating gene", g)
            kinetics = NegativeBinomialMixtureEM(min_components=min_components, 
                                                 max_components=max_components, zi=None, 
                                                 max_iter_em=max_iter_kinetics,
                                                 refilter=refilter, hard_em=self.hard_em, 
                                                 preserve_mean_values=self.preserve_mean_values, mean_forcing_em=self.mean_forcing_em)
            x = data_rna[:, g]
            # Pass the read depth factor if it is available
            model = kinetics.fit(x, vect_t=vect_t, seuil=self.seuil, s=cell_rd)
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
                if temporal:
                    for t_i in np.unique(vect_t):
                        indices = (vect_t == t_i)
                        tmp_proba_i = np.zeros_like(proba[indices])
                        proba_i = proba[indices]
                        n_cells_i = np.sum(indices)
                        mu = np.ones(n_cells_i)/n_cells_i
                        if self.force_basins: nu = pi[t_i]
                        else: nu = np.sum(proba_i, axis=0)
                        nu /= np.sum(nu)
                        dist = - np.log(proba_i)
                        coupling = ot.bregman.sinkhorn(mu, nu, dist, reg=1, numItermax=10000)
                        idx = np.argmax(coupling, axis=1)
                        for cell in range(n_cells_i):
                            tmp_proba_i[cell, idx[cell]] = 1
                        tmp[indices, :] = tmp_proba_i[:, :]
                else:
                    mu = np.ones(N_cells)/N_cells
                    if self.force_basins: nu = np.sum([pi[t_i] * np.sum(vect_t == t_i)/N_cells 
                                                       for t_i in np.unique(vect_t)], axis=0)
                    else: nu = np.sum(proba, axis=0)
                    nu /= np.sum(nu)
                    dist = - np.log(proba) 
                    coupling = ot.bregman.sinkhorn(mu, nu, dist, reg=1, numItermax=10000)
                    idx = np.argmax(coupling, axis=1)
                    for cell in range(N_cells):
                        tmp[cell, idx[cell]] = 1
        
            return ks, c, pi0, proba, tmp, pi

        results = Parallel(n_jobs=-1)(
        delayed(run_main_loop_for_gene)(g) for g in range(1, G_tot)
        )

        for idx, g in enumerate(range(1, G_tot)):

            kg, cg, pi_zerog, probag, tmpg, pi_initg = results[idx]
            cg_old, cg = cg, np.minimum(9, cg) # No need of having a variance too low
            kg *= cg / cg_old
            frequency_modes_smooth[:, g] = np.sum(kg * tmpg, axis=1)
            if verb and g <= len(gene_names): print('Gene {}-{} calibrated...'.format(g, gene_names[g-1]), kg, cg)
            if len(kg) > n_components:
                n_components = len(kg)
            ks.append(kg)
            proba_init.append(probag)
            proba_modif.append(tmpg)
            c[g] = cg
            pi_zeros[g-1] = pi_zerog
            pi_init.append(pi_initg)

        self.a = np.zeros((n_components+1, G_tot)) + self.seuil
        frequency_proba_init = np.zeros((N_cells, G_tot, n_components))
        frequency_proba_init[:N_cells_0, 0, 0], frequency_proba_init[N_cells_0:, 0, -1] = 1, 1 # the stimulus is in high mode after 0
        frequency_proba_modif = frequency_proba_init.copy()
        self.a[:, 0] = 1
        for g in range(1, G_tot):
            self.a[:len(ks[g-1]), g] = ks[g-1][:]
            frequency_proba_init[:, g, :len(ks[g-1])] = proba_init[g-1]
            frequency_proba_init[:, g, len(ks[g-1]):] = 0
            frequency_proba_modif[:, g, :len(ks[g-1])] = proba_modif[g-1]
            frequency_proba_modif[:, g, len(ks[g-1]):] = 0
        self.a[-1, :] = c[:]
        self.pi_init = pi_init
        
        scale_max = np.max(self.a[:-1, :], 0)
        frequency_modes_smooth /= scale_max

        if verb: print('Mean proba = ', np.mean(np.max(frequency_proba_init[:, 1:, :], axis=-1)), 
              np.mean(np.max(frequency_proba_modif[:, 1:, :], axis=-1)))
        
        return frequency_modes_smooth, frequency_proba_init, frequency_proba_modif, pi_zeros



    def fit_mixture(self, data_rna, refilter=0, gene_names=np.arange(1, 50000), min_components=2, max_components=2, max_iter_kinetics=0, cell_rd=None, verb=True):
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
        N_cells, G_tot = data_rna.shape
        vect_t = data_rna[:, 0]

        # Conversion propre du read depth (supporte pd.Series et np.array)
        if cell_rd is not None:
            cell_rd = np.asarray(cell_rd, dtype=float).reshape(-1)
            assert len(cell_rd) == N_cells, (
                f"cell_rd a {len(cell_rd)} entrées mais data_rna a {N_cells} cellules."
            )

        frequency_modes_smooth, frequency_proba_init, frequency_proba_modif, pi_zeros = self.core_binarization(
                                        data_rna, gene_names, vect_t, G_tot, 
                                        min_components=min_components, 
                                        max_components=max_components, 
                                        refilter=refilter, 
                                        max_iter_kinetics=max_iter_kinetics,
                                        cell_rd=cell_rd,
                                        verb=verb)

        self.pi = pi_zeros
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
        alpha = (np.abs(d) / (fact * (1e-12 + mu)))**p
        weight = alpha / (1 + alpha)
        res = x * (1 - weight) + mu * weight
        return res * self.scale_proteins
    
    
    def adaptive_shrinkage_init(self, x, mu, p=.5):
        G = mu.shape[1]
        res = mu.copy()
        xs = self.adaptive_shrinkage(x, mu)
        a = np.min(mu, axis=0)
        b = np.max(mu, axis=0)
        for g in range(G):
            ks = np.sort(np.unique(mu[:, g]))
            x_min = np.min(xs[:, g])
            x_max = np.max(xs[:, g]) - x_min
            xs[:, g] -= x_min
            lmax = len(ks)
            for cnt_z, z in enumerate(ks):
                indices = (mu[:, g] == z)
                x_ming = min(np.min(xs[indices, g]), x_max / (1 + lmax - cnt_z))
                x_maxg = max(np.max(xs[indices, g]), x_max / (lmax - cnt_z))
                res[indices, g] = a[g] + (b[g]-a[g]) * (cnt_z + 
                                                (np.clip(xs[indices, g], x_ming, x_maxg) / (x_maxg + 1e-16))**p) / lmax
        # print(np.mean(res, axis=0), np.mean(mu, axis=0))
        return res * self.scale_proteins
    

    def estimate_trajectories_given_model(self, vect_t, times, vect_samples_id, 
                                      samples_id, vect_rna, y_prot_old, y_kon_old, y_rna_old, y_proba_old,
                                      basal, inter, s1, ks, nb_cells, init_cells, offset_init=[0],  
                                      n_iter=1, N_full=[100], N_samples=[100], intensity_prior=10):
        """
        Infer the protein trajectories when d1 is known and theta is not.
        """

        G = vect_rna.shape[1]
        T = len(times)
        N_total = np.sum(N_full)
        N_cells_0 = np.sum(vect_t == 0)
        to_keep_for_update = np.zeros_like(vect_t, dtype=int)

        # Initialize simulation arrays
        rna_modified = y_rna_old.copy()
        prot_modified = y_prot_old.copy()
        prot_formodes = np.zeros_like(self.modes)
        kon_modified = y_kon_old.copy()
        proba_modified = y_proba_old.copy()
        vect_samples_id_modified = np.zeros(rna_modified.shape[0])

        # Stimulus at t>0
        prot_formodes[N_cells_0:, 0] = self.scale_proteins
        rna_modified[N_total:, 0] = self.scale_proteins
        prot_modified[N_total:, 0] = self.scale_proteins
        kon_modified[N_total:, 0] = 1

        # Fill initial state (t = 0)
        offset = 0
        for s, sample in enumerate(samples_id):
            cell_indices = (vect_samples_id == sample)
            selected_init = init_cells[s]

            kon_modified[offset+offset_init[s]:offset+offset_init[s] + N_samples[s], 1:] = self.modes[cell_indices][selected_init, 1:]
            if self.compute_with_proba:
                proba_modified[offset+offset_init[s]:offset+offset_init[s] + N_samples[s]] = self.proba[cell_indices][selected_init]
            rna_modified[offset+offset_init[s]:offset+offset_init[s] + N_samples[s], 1:] = vect_rna[cell_indices][selected_init, 1:]

            offset += N_full[s]

        # Protein initialization without stimulus
        prot_modified[:N_total, 1:] = self.adaptive_shrinkage_init(rna_modified[:N_total, 1:] * s1, kon_modified[:N_total, 1:])
        prot_formodes[:N_cells_0, 1:] = self.adaptive_shrinkage_init(vect_rna[:N_cells_0, 1:] * s1, self.modes[:N_cells_0, 1:])
        to_keep_for_update[:N_cells_0] = 1

        alpha_prev = self.alpha.copy()

        for t_idx, time in enumerate(times[:-1]):
            offset = 0
            for s_idx, sample in enumerate(samples_id):
                start_next = (vect_t == times[t_idx + 1]) & (vect_samples_id == sample)
                offset_init_s = offset + offset_init[s]
                start_index = N_total * t_idx + offset_init_s
                next_index = N_total * (t_idx + 1) + offset_init_s
                N_sample = N_samples[s_idx]
                N_cells = nb_cells[s_idx, t_idx + 1]
                start_index_full = N_total * (t_idx + 1) + offset 
                next_index_full = N_total * (t_idx + 1) + offset + N_full[s_idx]

                vect_samples_id_modified[N_total * t_idx + offset:N_total * t_idx + offset + N_full[s_idx]] = s_idx

                if N_sample: # Don't calculate if there no cells in this sample for this round

                    # Prepare indices
                    current_indices = np.arange(start_index, start_index + N_sample)
                    alpha_indices = np.arange(offset + offset_init[s], offset + offset_init[s] + N_sample)
                    mode_init = self.adaptive_shrinkage(rna_modified[current_indices, 1:] * s1, kon_modified[current_indices, 1:] * self.scale_proteins) / s1
                    mode_end = self.adaptive_shrinkage(vect_rna[start_next, 1:] * s1, self.modes[start_next, 1:] * self.scale_proteins) / s1
                    # Compute cost matrix and transitions
                    pairwise_dist, next_prot = my_otdistance(
                        kon_modified[current_indices, 1:], self.modes[start_next, 1:],
                        prot_modified[current_indices, 1:], 
                        rna_modified[current_indices, 1:], vect_rna[start_next, 1:], 
                        proba_modified[current_indices, 1:], self.proba[start_next, 1:, :],
                        mode_init, mode_end,
                        self.alpha[t_idx, alpha_indices],
                        s1, ks, self.d[1, 1:], times[t_idx + 1] - time, basal, inter, loss=self.loss_norm,
                        n_iter=n_iter, intensity_prior=intensity_prior,
                        compute_with_proba=self.compute_with_proba,
                    )

                    mu = np.ones(N_sample)/N_sample
                    nu = np.ones(N_cells)/N_cells
                    tmp = np.log(G) # approximate number of errors expected by gene in the reconstructed modes
                    reg = self.init_entropic_noise * tmp / n_iter # decrease entropic regularization with iterations - on purpose the slope is in 1/x for fast convergence
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
                        while stopThr <= 1e-5:
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

                    cell_idx = np.flatnonzero(start_next)
                    for n in range(0, N_sample):
                        m = np.random.choice(N_cells, p=coupling[n] / coupling[n].sum())
                        target_index = next_index + n

                        kon_modified[target_index] = self.modes[cell_idx[m]]
                        if self.compute_with_proba:
                            proba_modified[target_index] = self.proba[cell_idx[m]]
                        rna_modified[target_index, 1:] = vect_rna[cell_idx[m], 1:]
                        prot_modified[target_index, 1:] = next_prot[n, m, 1:]
                        prot_formodes[cell_idx[m], 1:] = next_prot[n, m, 1:]
                        to_keep_for_update[cell_idx[m]] = 1

                        ### Re-assign the alpha to the new cells if alpha is associated to cells and not trajectories
                        if y_prot_old.sum() and time != times[-2]:
                            target = prot_modified[target_index, 1:]
                            subarray = y_prot_old[start_index_full:next_index_full, 1:]
                            distances_prot = np.linalg.norm(subarray - target, 1, axis=1)
                            target = kon_modified[target_index, 1:]
                            subarray = y_kon_old[start_index_full:next_index_full, 1:]
                            distances_kon = np.linalg.norm(subarray - target, 1, axis=1)
                            match = np.argmin((1/G)*distances_prot+((G-1)/G)*distances_kon)
                            self.alpha[t_idx+1, offset_init_s + n] = alpha_prev[t_idx+1, offset + match]
                    

                offset += N_full[s_idx]

        return prot_modified, prot_formodes, rna_modified, kon_modified, proba_modified, vect_samples_id_modified, to_keep_for_update
    
    

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
        n_loops,
        count_max,
        intensity_prior,
        basal_init=None,
        inter_init=None,
        verb=True,
        compute_theta=True,
        initialize_alpha=True
    ):
        """
        Alternating optimization of trajectories and network (theta).
        """

        n_iter = 1
        errors = [1e12]
        count_end = 0
        N_tot = np.sum(N_full)

        if compute_theta:
            # --- Initialize theta parameters (basal, inter) ---
            self.basal = np.zeros((G_tot, self.n_networks))
            self.inter = np.zeros((G_tot, G_tot, self.n_networks))
            if basal_init is not None and inter_init is not None:
                for n in range(self.n_networks):
                    self.basal[:, n] = basal_init
                    self.inter[:, :, n] = inter_init

        basal, inter = self.basal.copy(), self.inter.copy()
        basal_tmp, inter_tmp = self.basal.copy(), self.inter.copy()
        basal_ref, inter_ref = self.basal.copy(), self.inter.copy()

        # --- Initialize switching probabilities alpha ---
        if initialize_alpha:
            self.alpha = np.random.uniform(.1, .9, size=(len(times) - 1, N_tot, G_tot - 1)) 
            if np.linalg.norm(self.ref_network[0, :]):
                self.alpha[0] = .01 # stimulus at t=0 has fast effect

        # --- Time vector for full and reduced datasets ---
        vect_t_sim = np.repeat(times, N_tot)

        # --- Initialize placeholders ---
        y_prot = np.zeros((len(times) * N_tot, G_tot))
        y_prot_prev = np.zeros((G_tot, len(times) * N_tot, G_tot))
        y_prot_prev[:, N_tot:, 0] = 1 # Define stimulus for the one which will not be updated in estimate_trajectories
        y_kon = np.zeros_like(y_prot)
        y_rna = np.zeros_like(y_prot)
        y_proba = np.zeros((len(times) * N_tot, G_tot, self.n_networks + 1))

        # === Main loop ===
        while (n_iter < n_loops) or ((n_iter >= n_loops) and count_end < count_max):
            if n_iter > self.max_iter:
                break
            
            weight_prev = self.weight_prev * min(1, (n_iter-1)/n_loops) # Flow matching from second iteration and small at early ones

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
                            self.alpha[cnt, offset:offset+N] = self.alpha[cnt, offset + indices_shuffled[s]]
                        y_prot[cnt * N_tot + offset : cnt * N_tot + offset + N] = y_prot[cnt * N_tot + offset + indices_shuffled[s]]
                        y_kon[cnt * N_tot + offset : cnt * N_tot + offset + N] = y_kon[cnt * N_tot + offset + indices_shuffled[s]]
                        y_rna[cnt * N_tot + offset : cnt * N_tot + offset + N] = y_rna[cnt * N_tot + offset + indices_shuffled[s]]
                        y_proba[cnt * N_tot + offset : cnt * N_tot + offset + N] = y_proba[cnt * N_tot + offset + indices_shuffled[s]]
                        offset += N
            else:
                indices_shuffled = [np.arange(N_full[s]) for s in range(len(samples_id))]

            offset_init = [0 for _ in range(len(samples_id))]
            while not np.array_equal(offset_init, N_full):
                N_tmp = [min(N_samples[s], N_full[s] - offset_init[s]) for s in range(len(samples_id))]
                init_cells = [init_cells_full[s][offset_init[s]:offset_init[s] + N_tmp[s]] for s in range(len(samples_id))]
                # --- Estimate trajectories given current model ---
                y_prot, y_prot_formodes, y_rna, y_kon, y_proba, y_samples, to_keep_for_update = self.estimate_trajectories_given_model(
                    vect_t, times, vect_samples_id, samples_id,
                    data_rna, y_prot, y_kon, y_rna, y_proba, basal, inter, s1, ks,
                    nb_cells, init_cells, offset_init=offset_init,
                    N_full=N_full, N_samples=N_tmp, 
                    n_iter=n_iter + 10 * (1-compute_theta) + 10 * (np.linalg.norm(inter_ref) > 1), # we use a small regularization if we just update trajectories or have a reference network
                    intensity_prior=intensity_prior * compute_theta # we don't care about the prior if we just update trajectories
                )
                offset_init = [offset_init[s] + N_tmp[s] for s in range(len(samples_id))]

                y_prot_prev[:, :N_tot, 1:] = y_prot[:N_tot, 1:]
    
            # --- Evaluate error before and after inference ---
            error = count_errors(y_prot, y_kon, y_proba, ks, basal, inter, loss=self.loss_norm, compute_with_proba=self.compute_with_proba)
            if compute_theta and len(times) > 1:
                if self.weight_prev > 0:
                    modes = self.adaptive_shrinkage(y_rna[:, 1:] * s1, y_kon[:, 1:] * self.scale_proteins) / s1
                    for cnt, time in enumerate(times[:-1]):
                        delta_t = times[cnt + 1] - time
                        for n in range(N_tot):
                            idx_prev = N_tot * cnt + n
                            idx_next = N_tot * (cnt + 1) + n
                            alpha_n = self.alpha[cnt, n]
                            
                            for g in range(1, G_tot):
                                alpha_n_mod = min(alpha_n[g-1]+.1, 1) # Letting some time for mode to stabilize
                                y_prot_prev[g, idx_next, 1:] = find_next_prot(self.d[1, 1:],
                                                        y_prot[idx_prev, 1:],
                                                        y_rna[idx_prev, 1:], y_rna[idx_next, 1:],
                                                        modes[idx_prev],
                                                        modes[idx_next],
                                                        np.minimum(alpha_n / alpha_n_mod, 1),
                                                        s1, delta_t * alpha_n_mod
                                                    )
                                
                basal, inter, basal_tmp, inter_tmp = inference_network(
                    vect_t_sim, times, y_samples, y_kon, y_proba, y_prot, y_prot_prev, 
                    ks, ref_network = self.ref_network, basal_init=basal, inter_init=inter,
                    basal_ref = basal_ref, inter_ref = inter_ref,
                    proba=self.compute_with_proba, scale=self.scale_pen, weight_prev=weight_prev, loss=self.loss_norm,
                    final=int(np.linalg.norm(inter_ref) > 1))


            error_2 = count_errors(y_prot, y_kon, y_proba, ks, basal, inter, loss=self.loss_norm, compute_with_proba=self.compute_with_proba)
            errors.append(error_2)
            self.loss_trajectory.append(error_2)
            self.theta_trajectory.append(inter_tmp)

            if verb:
                print(f"{n_iter}", f"{count_end} | Errors (before, after): {error:.5f}, {error_2:.5f} | alpha mean: {np.mean(self.alpha[0]):.4f}")

            # --- Update counts for stopping condition if n_iter is high enough ---
            n_iter += 1
            if (n_iter >= n_loops) and (count_end >= 2):
                if (errors[-2] - errors[-1]) < 1e-3:
                    count_end += 1
                ### If we compute theta, the absence of difference before and after update of theta is also taken into account
                if compute_theta:
                    if np.abs(error - error_2) < 2e-4:
                        count_end += 1
            # Unblock the counter
            if (count_end < 2) and (n_iter > n_loops/2) and (errors[-1] - errors[-2]) > 0:
                count_end += 1

            # --- Update kon_theta for alpha ---
            kon_vector = y_kon.copy()
            kon_vector[:, 1:] = kon_ref_vector(y_prot, ks, inter, basal)[:, 1:]

            # --- Update alphas ---
            modes = self.adaptive_shrinkage(y_rna[:, 1:] * s1, y_kon[:, 1:] * self.scale_proteins) / s1
            if len(times) > 1:
                for cnt, time in enumerate(times[:-1]):
                    self.alpha[cnt] = inference_alpha(
                            self.d[1, 1:], s1,
                            self.alpha[cnt],
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
                            tol=self.alpha_threshold
                        )
            
            # --- Update kon_theta values for modes ---
            kon_vector_formodes = y_prot_formodes.copy()
            kon_vector_formodes[:, 1:] = kon_ref_vector(y_prot_formodes, ks, inter, basal)[:, 1:]
            print("number of non reached cells", np.sum(to_keep_for_update == 0))

            # --- Update modes ---
            if self.update_modes:
                n_cells = self.proba_init.shape[0]
                reg = (1 + n_loops - n_iter)/n_loops
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
                            obj_i = obj[indices].copy()
                            n_cells_i = np.sum(indices)
                            mu = np.ones(n_cells_i)/n_cells_i
                            if self.force_basins: nu = self.pi_init[g-1][t_i]
                            else: nu = np.sum(proba_i[:, :l_max], axis=0)
                            nu /= np.sum(nu)
                            diff_k = np.maximum(1 - np.abs(kon_vector_formodes[indices, g, None] - obj_i), self.seuil)
                            dist = - (np.log(proba_i) + (1 - weight_prob) * to_keep_for_update[indices, None] * 
                                                        np.log(diff_k/np.max(diff_k, axis=1, keepdims=True)))
                            if n_iter <= n_loops: 
                                try: 
                                    coupling = ot.bregman.sinkhorn(mu, nu, dist, reg=reg, numItermax=int(10000/reg), stopThr=self.stopThr_init*10)
                                except Exception: 
                                    coupling = ot.emd(mu, nu, dist)
                            else: coupling = ot.emd(mu, nu, dist)
                            idx = np.argmax(coupling, axis=1)
                            for cell in range(n_cells_i):
                                tmp_proba_i[cell, idx[cell]] = 1
                                tmp_modes_i[cell] = obj_i[cell, idx[cell]]
                            tmp_proba[indices, :] = tmp_proba_i[:, :]
                            tmp_modes[indices] = tmp_modes_i[:]
                    else:
                        proba = self.proba_init[:, g, :l_max].copy()
                        mu = np.ones(n_cells)/n_cells
                        if self.force_basins: nu = np.sum([self.pi_init[g-1][t_i] * np.sum(vect_t == t_i)/n_cells 
                                                       for t_i in times], axis=0)
                        else: nu = np.sum(proba[:, :l_max], axis=0)
                        nu /= np.sum(nu)
                        diff_k = np.maximum(1 - np.abs(kon_vector_formodes[:, g, None] - obj), self.seuil)
                        dist = - (np.log(proba) + (1 - weight_prob) * to_keep_for_update[:, None] * 
                                                    np.log(diff_k/np.max(diff_k, axis=1, keepdims=True)))
                        if n_iter <= n_loops: 
                            try: 
                                coupling = ot.bregman.sinkhorn(mu, nu, dist, reg=reg, numItermax=int(10000/reg), stopThr=self.stopThr_init*10)
                            except Exception: 
                                coupling = ot.emd(mu, nu, dist)
                        else: 
                            coupling = ot.emd(mu, nu, dist)
                        idx = np.argmax(coupling, axis=1)
                        for cell in range(n_cells):
                            tmp_proba[cell, idx[cell]] = 1
                            tmp_modes[cell] = obj[cell, idx[cell]]
                    return tmp_proba, tmp_modes

                results = Parallel(n_jobs=-1)(
                    delayed(run_main_loop_for_gene)(g) for g in range(1, G_tot))

                for idx, g in enumerate(range(1, G_tot)):
                    tmp_proba, tmp_modes = results[idx]
                    self.proba[:, g, :], self.modes[:, g] = tmp_proba[:, :], tmp_modes[:]

        # --- Updating the networks ---
        if compute_theta:
            if self.filter_network:
                inter, inter_tmp = filter_network(len(times), N_tot, y_prot, ks, basal, basal_tmp, inter, inter_tmp)
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


    def fit_network(
        self,
        data_rna,
        intensity_prior=10,
        vect_samples_id=None,
        basal_init=None,
        inter_init=None,
        verb=True
    ):
        """
        Fit the gene regulatory network to the RNA expression data.

        Parameters
        ----------
        data_rna : ndarray
            RNA expression matrix (cells × genes), including a time column at index 0.
        intensity_prior : float
            Regularization intensity for optimal transport.
        vect_samples_id : ndarray or None
            Array of sample labels (same size as data_rna), or None if only one sample.
        n_loops : int
            Number of full inference cycles (trajectories <-> network).
        count_max : int
            Max number of convergence steps (tolerance checks).
        basal_init : ndarray or None
            Optional initial basal activity (G × networks).
        inter_init : ndarray or None
            Optional initial interaction matrix (G × G × networks).
        verb : bool
            Whether to print progress.
        """

        # --- Initialization ---
        G_tot = data_rna.shape[1]  # total number of genes
        vect_t = data_rna[:, 0]

        # --- Adapt ref_network ---
        self.ref_network = np.maximum(self.prior_network_pen, self.ref_network)
        self.ref_network[0, :] = self.stimulus
        for g in range(1, G_tot):
            l_max = len(np.unique(self.modes[:, g]))
            if l_max < 2:
                self.ref_network[g, :], self.ref_network[:, g] = 0, 0
            if l_max > len(np.unique(self.a[:-1, g])): # compuet with proba = 0 si more modes than ks
                self.compute_with_proba = 0

        # If no sample ID provided, assume one global sample
        if vect_samples_id is None or not np.linalg.norm(vect_samples_id):
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
            q, r = divmod(n, self.batch_size) 
            if q == 0: N_samples.append(n)
            else: N_samples.append(min(self.batch_size + 1+int(r/q), n))

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
        ks = (self.a[:-1] / np.clip(np.max(self.a[:-1], axis=0), 1e-12, None)).T
        s1 = self.fact_simple * self.a[-1, 1:] / np.clip(np.max(self.a[:-1, 1:], axis=0), 1e-12, None)
        s1 *= self.scale_proteins

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
            n_loops=self.n_loops,
            count_max=self.count_max,
            intensity_prior=intensity_prior,
            basal_init=basal_init,
            inter_init=inter_init,
            verb=verb,
            compute_theta=True,
            initialize_alpha=True,
        )


        # --- Print results (optional) ---
        if verb:
            print("\n[fit_network] Final network:")
            for n in range(self.n_networks):
                print(f"  Network {n} | Interactions:\n", self.inter[:, :, n].T)
                print(f"  Network {n} | Basal:\n", self.basal[:, n])

            print("\n[fit_network] Intermediate network:")
            for n in range(self.n_networks):
                print(f"  Network {n} | Interactions:\n", self.inter_tmp[:, :, n].T)
                print(f"  Network {n} | Basal:\n", self.basal_tmp[:, n])
            

    def estimate_trajectories_unitary(self, y_prot, times, d1, N=100):
        """
        Estimate protein trajectories when d1, theta, and alpha are known.
        """
        prot_modified = y_prot.copy()
        N_tot, G = prot_modified.shape
        prot_modified_prev = np.ones((G, N_tot, G))
        prot_modified_prev[:, :N, 0] = 0
        prot_modified_prev[:, :N, 1:] = prot_modified[:N, 1:]

        for cnt, time in enumerate(times[:-1]):
            delta_t = times[cnt + 1] - time

            for n in range(N):
                idx_prev = N * cnt + n
                idx_next = N * (cnt + 1) + n
                alpha_n = self.alpha[cnt, n]

                prot_modified[idx_next, 1:] = find_next_prot(
                    d1,
                    prot_modified[idx_prev, 1:],
                    self.kon_beta[idx_prev, 1:],
                    self.kon_beta[idx_next, 1:],
                    self.kon_beta[idx_prev, 1:],
                    self.kon_beta[idx_next, 1:],
                    alpha_n,
                    self.scale_proteins,
                    delta_t
                )

                if self.weight_prev > 0:
                    for g in range(1, G):
                        alpha_n_mod = min(alpha_n[g-1]+.1, 1) # letting some time for mode to stabilize
                        prot_modified_prev[g, idx_next, 1:] = find_next_prot(d1,
                                                prot_modified[idx_prev, 1:],
                                                self.kon_beta[idx_prev, 1:],
                                                self.kon_beta[idx_next, 1:],
                                                self.kon_beta[idx_prev, 1:],
                                                self.kon_beta[idx_next, 1:],
                                                np.minimum(alpha_n / alpha_n_mod, 1),
                                                self.scale_proteins,
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
            n_pick = min(N_s, self.nb_traj_for_degradations_inference)
            if n_pick == 0:
                continue

            chosen_idx = np.random.choice(idx_first_indices, n_pick, replace=False)

            # marquer les mêmes cellules à travers tous les temps
            for cnt, t in enumerate(times):
                idx_cnt = chosen_idx+N_t*cnt
                cells_to_use[idx_cnt] = 1

        return cells_to_use


    def adapt_to_unitary(self, verb=True):
        """
        Adapt learned parameters to unitary protein scale.
        """
        times = np.sort(np.unique(self.times_data))
        N_tot = np.sum(self.times_data == 0)

        # --- Adapt ref_network ---
        self.ref_network = np.maximum(self.prior_network_pen, self.ref_network)
        self.ref_network[0, :] = self.stimulus
        for g in range(1, self.ref_network.shape[0]):
            l_max = len(np.unique(self.modes[:, g]))
            if l_max < 2: # If only one mode
                self.ref_network[g, :], self.ref_network[:, g] = 0, 0
            if l_max > len(np.unique(self.a[:-1, g])): # compute with proba = 0 si more modes than ks
                self.compute_with_proba = 0

        ks = (self.a[:-1] / np.max(self.a[:-1], axis=0)).T
        k1 = np.max(self.a[:-1], axis=0)
        y_prot = self.prot.copy()
        basal, inter = self.basal.copy(), self.inter.copy()
        basal_ref, inter_ref = self.basal.copy(), self.inter.copy()

        y_prot, y_prot_prev = self.estimate_trajectories_unitary(y_prot, times, self.d[1, 1:], N=N_tot)
        error = count_errors(y_prot, self.kon_beta, self.proba_traj, ks, self.basal, self.inter, loss=self.loss_norm, compute_with_proba=self.compute_with_proba)

        basal, inter, _, _ = inference_network(
            self.times_data, times, self.samples_data, self.kon_beta, self.proba_traj,
            y_prot, y_prot_prev, ks, proba=self.compute_with_proba,
            ref_network = self.ref_network, basal_init=basal_ref, inter_init=inter_ref,
            basal_ref=basal_ref, inter_ref=inter_ref,
            scale=self.scale_pen * self.fact_simple, weight_prev=self.weight_prev, loss=self.loss_norm, final=1
        )

        error_corrected = count_errors(y_prot, self.kon_beta, self.proba_traj, ks, basal, inter, loss=self.loss_norm, compute_with_proba=self.compute_with_proba)
        if verb:
            print("[adapt_unitary] ratio errors unitary", error, error_corrected)

        self.prot[:, :] = y_prot[:, :]
        kon_vector = self.kon_beta.copy()
        kon_vector[:, 1:] = kon_ref_vector(y_prot, ks, inter, basal)[:, 1:]
        self.kon_theta[:, :] = kon_vector[:, :]

        ### Adapt degradation rates
        self.ratios = np.tile(self.d[0, :] / self.d[1, :], (len(times)-1, 1))
        self.d_t = np.tile(self.d, (len(times)-1, 1, 1))
        basal_t = np.tile(basal, (len(times)-1, 1, 1))
        inter_t = np.tile(inter, (len(times)-1, 1, 1, 1))

        if self.recompute_degradations:
            cells_to_use = self.select_cells_to_use()
            if not self.use_temporal_degradations:
                d1, scale_theta = inference_degradation_prot(y_prot[cells_to_use == 1], 
                            self.times_data[cells_to_use == 1], basal, inter, ks.T * self.scale_proteins, d=self.d[1], lr=1e-2*self.scale_proteins)
                self.d_t[:, 1, :] = np.tile(d1, (len(times)-1, 1))
                basal *= scale_theta[:, None]
                inter *= scale_theta[None, :, None]
                basal_t *= scale_theta[None, :, None]
                inter_t *= scale_theta[None, None, :, None]


            if self.use_temporal_degradations:
                def run_main_inference_degradation_prot(t):
                    idx = ((self.times_data == times[t]) | (self.times_data == times[t+1])) & (cells_to_use == 1)
                    return inference_degradation_prot(y_prot[idx], self.times_data[idx], basal, inter, ks.T * self.scale_proteins, d=self.d[1], lr=1e-2*self.scale_proteins)

                results = Parallel(n_jobs=-1)(
                delayed(run_main_inference_degradation_prot)(t) for t in range(0, len(times)-1)
                )

                scale_theta = np.ones_like(self.d_t[:, 1, :])
                for cnt in range(0, len(times)-1):
                    res_t = results[cnt]
                    self.d_t[cnt, 1, :], scale_theta[cnt] = res_t
                    basal_t[cnt] = basal * scale_theta[cnt, :, None]
                    inter_t[cnt] = inter * scale_theta[cnt, None, :, None]
                basal *= np.mean(scale_theta, axis=0)[:, None]
                inter *= np.mean(scale_theta, axis=0)[None, :, None]


            eps_temporal, eps_global = inference_epsilon_temporal(y_prot, 
                                                                  self.times_data, 
                                                                  basal_t, inter_t, ks.T * self.scale_proteins, self.d_t[:, 1, :], 
                                                                  k1 * self.scale_proteins, self.ratios, self.alpha, verbose=False)

            for cnt in range(0, len(times)-1):
                if self.use_temporal_degradations: 
                    self.ratios[cnt, :] = 1/eps_temporal[cnt]
                else: 
                    self.ratios[cnt, :] = 1/eps_global

        self.basal, self.inter = basal, inter
        self.basal_t, self.inter_t = basal_t, inter_t
        
        if verb:
            print('[adapt_unitary]  Static network unitary', [self.inter.transpose(1, 0, 2)[:, :, n] for n in range(self.n_networks)],
                    [self.basal[:, n] for n in range(self.n_networks)])


    def simulate_trajectories_unitary(self, times, times_train, ks, N=100, verb=True):
        """
        Simulate protein trajectories assuming d1 is known, and theta is not.
        
        This assumes the change in dynamics occurs halfway between timepoints.
        """
        # By default
        prot_modified = np.ones((N * len(times), self.prot.shape[1]))
        kon_vector = np.ones((N * len(times), self.prot.shape[1]))
        prot_modified[:N, :] = self.prot[:N, :]
        kon_vector[:N, :] = self.kon_beta[:N, :]
        kon_vector[:N, 1:] = kon_ref_vector(self.prot[:N, :], ks, self.inter, self.basal)[:, 1:]
        start_time=0
        # We want to capture automatically if we simulate from after last timepoint
        if times_train[-1] < times[1]: # times[0] = 0 by construction
            times = [0, times_train[-1]] + list(times[1:])
            l = len(times_train)
            prot_modified = np.ones((N * len(times), self.prot.shape[1]))
            kon_vector = np.ones((N * len(times), self.prot.shape[1]))
            prot_modified[:N, :] = self.prot[:N, :]
            kon_vector[:N, :] = self.kon_beta[:N, :]
            kon_vector[:N, 1:] = kon_ref_vector(self.prot[:N, :], ks, self.inter, self.basal)[:, 1:]
            ### Add last timepoints as starting timepoints for simulation
            prot_modified[N:2*N, :] = self.prot[N*(l-1):N*l, :]
            kon_vector[N:2*N, 1:] = kon_ref_vector(self.prot[N*(l-1):N*l, :], ks, self.inter_t[-1], self.basal_t[-1])[:, 1:]
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
        self.d_t = np.zeros((len(times)-1, 2, self.prot.shape[1]), dtype=float)
        self.ratios = np.zeros((len(times)-1, self.prot.shape[1]), dtype=float)
        basal_t = np.zeros((len(times)-1, self.prot.shape[1], self.n_networks), dtype=float)
        inter_t = np.zeros((len(times)-1, self.prot.shape[1], self.prot.shape[1], self.n_networks), dtype=float)
        for cnt, time in enumerate(times[:-1]):
            index = np.argmin(np.abs(times_train[:-1] - time))
            self.d_t[cnt, :, :] = d_t_train[index, :, :]
            self.ratios[cnt, :] = ratios_train[index, :]
            basal_t[cnt, :, :] = basal_t_train[index, :, :]
            inter_t[cnt, :, :, :] = inter_t_train[index, :, :, :]

        ### Rescale kz
        rescale = np.ones(prot_modified.shape[1])
        rescale[1:] = np.max(self.a[:-1, 1:], axis=0)
        kz = ks * rescale.reshape(ks.shape[0], 1)

        for cnt, time in enumerate(times[start_time:-1], start=start_time):
            delta_t = times[cnt + 1] - time

            degradations = self.d_t[cnt].copy()
            if self.simulation_stochastic:
                self.ratios[cnt] = np.clip(self.ratios[cnt], self.min_ratio, self.max_ratio)
                degradations[0, :] = degradations[1, :] * self.ratios[cnt] * min(1 + np.sqrt(cnt), self.max_ratio)
                # degradations[0, :] = np.clip(degradations[0, :], self.min_ratio * degradations[1, :], self.max_ratio * degradations[1, :])

            if self.finish_by_determinist:
                if time >= times[-2] or time > times[-1] * (len(times)-1) / len(times): # We finish by a deterministic simulation to reach final equilibrium
                    self.simulation_stochastic = 0

            start_index = N * cnt
            end_index = N * (cnt+1)

            def run_main_loop_for_cell(n):
                if self.simulation_stochastic:
                    return simulate_next_prot_pdmp(
                            degradations[1, :],
                            kz * degradations[0, :].reshape(kz.shape[0], 1),
                            rescale * (degradations[0, :] / degradations[1, :]),
                            basal_t[cnt], inter_t[cnt], delta_t,
                            self.scale_proteins, P0=prot_modified[start_index + n, :]
                        )
                else: 
                    return simulate_next_prot_ode(
                        degradations[1, :], ks, basal_t[cnt], inter_t[cnt], delta_t,
                        self.scale_proteins, P0=prot_modified[start_index + n, :]
                    )

            results = Parallel(n_jobs=-1)(
            delayed(run_main_loop_for_cell)(n) for n in range(0, N)
            )

            for idx, n in enumerate(range(0, N)):
                prot_modified[end_index + n, 1:] = results[idx].p[-1]
            
            kon_vector[end_index:end_index+N, 1:] = kon_ref_vector(prot_modified[end_index:end_index+N, :], ks, inter_t[cnt], basal_t[cnt])[:, 1:]

            if verb:
                print(f'timepoints {cnt} done', delta_t, time)

        return prot_modified, kon_vector, times_simulation
    

    def simulate_network(self, times, verb=True):
        """
        Simulate the protein trajectories using the final inferred network.
        """
        times.sort()
        times_train = np.sort(np.unique(self.times_data))
        N = np.sum(self.times_data == 0) # We generate as many trajectories as initial cells
        ks = (self.a[:-1] / np.max(self.a[:-1], axis=0)).T

        y_prot, kon_vector, times_simul = self.simulate_trajectories_unitary(times, times_train, ks, N=N, verb=verb)

        self.prot = y_prot
        self.kon_theta = kon_vector
        self.times_simul = times_simul


    def fit_mixture_test(self, data_rna, kz, c, verb=False):

        N_cells, G_tot = data_rna.shape

        frequency_modes_smooth = np.ones_like(data_rna, dtype=float)
        frequency_proba = np.ones((N_cells, G_tot, self.n_networks + 1), dtype=float)

        kinetics = NegativeBinomialMixtureEM()
        for g in range(1, G_tot):
            ng = np.argmax(kz[1:, g]) + 2
            probag = kinetics.compute_proba(data_rna[:, g], kz[:ng, g], c[g])
            if self.transform_proba > 10:
                tmpg = np.where(probag == np.max(probag, axis=1, keepdims=True), 1, 0)
            else:
                tmpg = np.exp(self.transform_proba * (ng - 1) * np.log(G_tot) * (probag - 1 / ng))
                tmpg /= (1 + tmpg)
                tmpg /= np.sum(tmpg, axis=1, keepdims=True)
                for cell in range(probag.shape[0]):
                    if np.max(probag[cell]) > np.max(tmpg[cell]):
                        tmpg[cell, :] = probag[cell, :]
            if self.transform_proba: frequency_proba[:, g, :ng] = tmpg
            else: frequency_proba[:, g, :ng] = probag
            frequency_proba[:, g, ng:] = 0
            frequency_modes_smooth[:, g] = np.sum(kz[:ng, g] * tmpg, axis=1)
            if verb:
                print('[infer_test]', f'Gene {g} calibrated...', kz[:ng, g], c[g])

        scale_max = np.max(self.a[:-1, :], axis=0)
        frequency_modes_smooth /= scale_max
        self.pi = np.zeros(G_tot)
        self.modes = frequency_modes_smooth
        self.proba = frequency_proba



    def infer_test(self, data_rna, vect_samples_id=np.zeros(2), verb=True):
        """
        Run inference pipeline on test data: kon estimation, trajectory inference, and alpha initialization.
        """
        N_cells, G_tot = data_rna.shape
        vect_t = data_rna[:, 0]
        if not np.linalg.norm(vect_samples_id):
            vect_samples_id = np.zeros_like(vect_t)

        times = np.sort(np.unique(vect_t))
        kz = self.a[:-1]
        c = self.a[-1]

        self.fit_mixture_test(data_rna, kz, c)
        
        print('[infer_test] Mean proba = ', np.mean(np.max(self.proba[:, 1:, :], axis=-1)))

        ks = (kz / np.max(self.a[:-1], axis=0)).T
        s1 = self.a[-1, 1:] / np.maximum(np.max(self.a[:-1, 1:], axis=0), 1e-9)

        samples_id = np.sort(np.unique(vect_samples_id))
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
            q, r = divmod(n, self.batch_size) 
            if q == 0: N_samples.append(n)
            else: N_samples.append(min(self.batch_size + 1+int(r/q), n))

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
            N_samples=N_samples, # N_samples if we reconstruct only subtrajectories
            G_tot=G_tot,
            n_loops=self.n_loops,
            count_max=self.count_max,
            basal_init=None,
            inter_init=None,
            verb=verb,
            intensity_prior=0,
            compute_theta=False,
            initialize_alpha=True,
        )


    def fit(self, data_rna, intensity_prior=100, verb=True):

        self.fit_mixture(data_rna, min_components=2, max_components=2, refilter=5.0, max_iter_kinetics=100)
        self.fit_network(data_rna, intensity_prior=intensity_prior, verb=verb)
        # self.adapt_to_unitary()
