CardamomOT.model
================

.. py:module:: CardamomOT.model

.. autoapi-nested-parse::

   Main interface for the package.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/CardamomOT/model/base/index


Classes
-------

.. autoapisummary::

   CardamomOT.model.NetworkModel


Package Contents
----------------

.. py:class:: NetworkModel(n_genes=None, n_stimuli=1, times=None)

   Encapsulates the state and parameters of a regulatory network.

   The class stores kinetic, mixture and network parameters as well as
   trajectories produced during inference. It provides methods for
   initialization, calibration and simulation used by the higher-level
   pipeline script.


   .. py:attribute:: loss_trajectory
      :value: []



   .. py:attribute:: theta_trajectory
      :value: []



   .. py:attribute:: d
      :value: None



   .. py:attribute:: d_t
      :value: None



   .. py:attribute:: weights
      :value: None



   .. py:attribute:: n_networks
      :value: None



   .. py:attribute:: adapt_size_network
      :value: None



   .. py:attribute:: rna
      :value: None



   .. py:attribute:: kon_beta
      :value: None



   .. py:attribute:: modes
      :value: None



   .. py:attribute:: alpha
      :value: None



   .. py:attribute:: alpha_traj
      :value: None



   .. py:attribute:: pi_init
      :value: None



   .. py:attribute:: pi_zinb
      :value: None



   .. py:attribute:: kon_theta
      :value: None



   .. py:attribute:: a
      :value: None



   .. py:attribute:: ref_network
      :value: None



   .. py:attribute:: basal
      :value: None



   .. py:attribute:: inter
      :value: None



   .. py:attribute:: inter_t
      :value: None



   .. py:attribute:: basal_tmp
      :value: None



   .. py:attribute:: inter_tmp
      :value: None



   .. py:attribute:: ratios
      :value: None



   .. py:attribute:: times_data
      :value: None



   .. py:attribute:: times_simul
      :value: None



   .. py:attribute:: samples_data
      :value: None



   .. py:attribute:: prot
      :value: None



   .. py:attribute:: proba_init
      :value: None



   .. py:attribute:: proba
      :value: None



   .. py:attribute:: proba_traj
      :value: None



   .. py:attribute:: n_stimuli
      :value: 1



   .. py:attribute:: hard_em
      :value: 1



   .. py:attribute:: preserve_mean_values
      :value: 1



   .. py:attribute:: mean_forcing_em
      :value: 0.5



   .. py:attribute:: force_basins
      :value: 1.0



   .. py:attribute:: temporal_basins
      :value: 1



   .. py:attribute:: transform_proba
      :value: 0



   .. py:attribute:: seuil
      :value: 0.01



   .. py:attribute:: n_loops
      :value: 10



   .. py:attribute:: count_max
      :value: 5



   .. py:attribute:: max_iter
      :value: 35



   .. py:attribute:: stopThr_init
      :value: 1e-07



   .. py:attribute:: batch_size
      :value: 5096



   .. py:attribute:: unbalanced_reg
      :value: 5



   .. py:attribute:: init_entropic_noise
      :value: 1.5



   .. py:attribute:: quant_samples
      :value: 0.8



   .. py:attribute:: scale_proteins
      :value: 1



   .. py:attribute:: fact_simple
      :value: 2



   .. py:attribute:: s1
      :value: None



   .. py:attribute:: loss_norm
      :value: 'CE'



   .. py:attribute:: scale_pen
      :value: 20



   .. py:attribute:: compute_with_proba
      :value: 0



   .. py:attribute:: weight_prev
      :value: 0.4



   .. py:attribute:: update_modes
      :value: 1



   .. py:attribute:: alpha_threshold
      :value: 0.4



   .. py:attribute:: stimulus
      :value: 1.0



   .. py:attribute:: prior_network_pen
      :value: 1.0



   .. py:attribute:: constrain_basal_uniform
      :value: 0.0



   .. py:attribute:: lambda_scale
      :value: 0.01



   .. py:attribute:: lambda_deg0
      :value: 1



   .. py:attribute:: lambda_deg1
      :value: 0.01



   .. py:attribute:: lambda_mlp
      :value: 0.5



   .. py:attribute:: filter_network
      :value: 0



   .. py:attribute:: recompute_degradations
      :value: 1



   .. py:attribute:: nb_traj_for_degradations_inference
      :value: 300



   .. py:attribute:: use_temporal_degradations
      :value: 1



   .. py:attribute:: smooth_degradations_sigma
      :value: None



   .. py:attribute:: smooth_degradations_strength
      :value: 0.5



   .. py:attribute:: simulation_stochastic
      :value: True



   .. py:attribute:: finish_by_determinist
      :value: False



   .. py:attribute:: min_ratio
      :value: 0.05



   .. py:attribute:: max_ratio
      :value: 20



   .. py:attribute:: production_factor
      :value: None



   .. py:attribute:: simulate_full_with_harissa
      :value: False



   .. py:attribute:: kon_beta_harissa
      :value: None



   .. py:attribute:: kon_mlp
      :value: None



   .. py:method:: core_binarization(data_rna, gene_names, vect_t, G_tot, min_components=1, max_components=5, refilter=0, max_iter_kinetics=100, cell_rd=None, verb=True, kov_cell_mask=None)

      :param cell_rd:
      :type cell_rd: (N_cells,) array or None
      :param kov_cell_mask: Per-cell KO/OV constraints derived from KO_OV_inference.

                            - -1: gene is KO for this cell (force to lowest mode)
                            - +1: gene is OV for this cell (force to highest mode)
                            - 0: no constraint
      :type kov_cell_mask: (N_cells, G_tot) int8 array or None



   .. py:method:: fit_mixture(data, refilter=0, gene_names=np.arange(1, 50000), min_components=2, max_components=2, max_iter_kinetics=0, cell_rd=None, verb=True, stimulus_schedule=None, time_key='time', kov_cell_mask=None)

      Fit the mixture model parameters to the data.

      :param cell_rd: Facteurs de read depth par cellule, typiquement issus de
                      adata.obs['rd'] (calculés par infer_rd.py).
                      Si None, le modèle NB classique sans correction est utilisé.
      :type cell_rd: (N_cells,) array, pd.Series, ou None
      :param Exemple d'appel avec correction de read depth::: rd = np.asarray(adata.obs['rd'])
                                                              model.fit_mixture(data_rna, ..., cell_rd=rd)



   .. py:method:: adaptive_shrinkage(x, mu, fact=2, p=2)


   .. py:method:: adaptive_shrinkage_init(x, mu, p=0.5)


   .. py:method:: estimate_trajectories_given_model(vect_t, times, vect_samples_id, samples_id, vect_rna, y_prot_old, y_kon_old, y_rna_old, y_proba_old, basal, inter, s1, ks, nb_cells, init_cells, offset_init=[0], n_iter=1, N_full=[100], N_samples=[100], intensity_prior=10)

      Infer the protein trajectories when d1 is known and theta is not.



   .. py:method:: loop_trajectories(data_rna, vect_t, vect_samples_id, times, samples_id, ks, s1, init_cells_full, nb_cells, N_full, N_samples, G_tot, n_loops, count_max, intensity_prior, basal_init=None, inter_init=None, basal_ref=None, inter_ref=None, verb=True, compute_theta=True, initialize_alpha=True, kov_cell_mask=None)

      Alternating optimization of trajectories and network (theta).

      basal_init / inter_init : (G_tot, n_networks) / (G_tot, G_tot, n_networks) or None
          Starting point for theta. Zeros if None.
      kov_cell_mask : (N_cells, G_tot) int8 array or None
          Per-cell KO/OV constraints. -1 → KO (force lowest mode),
          +1 → OV (force highest mode), 0 → no constraint.
          Applied after each update_modes step as a hard override.
      basal_ref / inter_ref : same shape or None
          Regularization target passed to inference_network. Zeros if None (no prior).



   .. py:method:: fit_network(data, intensity_prior=10, vect_samples_id=None, basal_init=None, inter_init=None, basal_ref=None, inter_ref=None, verb=True, stimulus_schedule=None, transition_rates=None, time_key='time')

      Fit the gene regulatory network to the RNA expression data.

      :param data: RNA expression matrix (cells × genes).
      :type data: ndarray or AnnData
      :param intensity_prior: Regularization intensity for optimal transport.
      :type intensity_prior: float
      :param vect_samples_id: Array of sample labels (same size as data), or None if only one sample.
      :type vect_samples_id: ndarray or None
      :param basal_init: Initial basal rates: shape (G,) broadcast to all networks, or (G, n_networks).
      :type basal_init: ndarray or None
      :param inter_init: Initial interaction matrix: shape (G, G) or (G, G, n_networks).
      :type inter_init: ndarray or None
      :param basal_ref: Regularization target for basal rates, same shape rules as basal_init.
                        Defaults to zeros (no penalization towards a prior).
      :type basal_ref: ndarray or None
      :param inter_ref: Regularization target for interactions, same shape rules as inter_init.
                        Defaults to zeros.
      :type inter_ref: ndarray or None
      :param verb: Whether to print progress.
      :type verb: bool



   .. py:method:: estimate_trajectories(y_prot, times, d1, N=100, kon_beta=None, s=None)

      Estimate protein trajectories when d1, theta, and alpha are known.

      :param kon_beta: Pre-computed burst frequencies. If None, uses ``self.kon_beta``.
      :type kon_beta: array of shape (T*N, G_tot), optional
      :param s: Per-gene protein scale. Defaults to ``self.scale_proteins``.
                Must match the s1 used in my_otdistance when building y_prot; pass
                ``self.s1`` to reproduce protein trajectories exactly.
      :type s: float or array of shape (G_genes,), optional



   .. py:method:: select_cells_to_use()


   .. py:method:: refine_network_degradations(verb=True)

      Refine network parameters and infer degradation rates for simulation.



   .. py:method:: simulate_trajectories_unitary(times, times_train, ks, N=100, verb=True, samples_data=None)

      Simulate protein trajectories with unitary scale



   .. py:method:: simulate_trajectories_full(times, times_train, ks, N=100, verb=True)

      Simulate protein AND mRNA trajectories using the Harissa bursty PDMP.

      Uses the inferred basal_t / inter_t (in absolute burst-rate units when
      simulate_full_with_harissa=True) as the Harissa network.
      Mimics simulate_trajectories_unitary but returns mRNA levels in addition
      to proteins.  Only supports ns == 1.

      :returns: * **prot_modified** (*(N * len(times), G_tot)*)
                * **mrna_modified** (*(N * len(times), G_tot)*)
                * **kon_vector** (*(N * len(times), G_tot)*)
                * **times_simulation** (*(N * len(times),)*)



   .. py:method:: simulate_network(times, verb=True, stimulus_schedule=None)

      Simulate the protein trajectories using the final inferred network.



   .. py:method:: fit_mixture_test(data_rna, ks, c, verb=False)

      Classify test cells into mixture modes using fixed kinetic parameters.

      Sets self.modes, self.proba, self.proba_init, and self.pi_init so that
      update_modes in loop_trajectories works on test data without re-fitting kz/c.



   .. py:method:: infer_test(data, vect_samples_id=None, verb=True, stimulus_schedule=None, basal_ref=None, transition_rates=None, time_key='time')

      Run inference pipeline on test data: kon estimation, trajectory inference, and alpha initialization.

      basal_ref : (n_samples, G_tot, n_networks) array or None
          Per-sample KO/OV prior (±100 entries) used to build kov_cell_mask.
      transition_rates : DataFrame or array or None
          Cell-type transition rate matrix for OT cost adjustment.



   .. py:method:: fit(data_rna, intensity_prior=100, verb=True)


