CardamomOT
==========

.. py:module:: CardamomOT

.. autoapi-nested-parse::

   CardamomOT
   =====================================================================================

   A gene regulatory network inference method adapted to time-course scRNA-seq datasets.

   The algorithm consists of calibrating the parameters of a mechanistic model of gene
   expression. The calibrated model can then be simulated to reproduce the dataset used
   for inference. The simulation part is based on the Harissa package.

   Key Features
   ------------
   - Mechanistic inference of gene regulatory networks from scRNA-seq data
   - Supports time-course and trajectory data
   - Integration with mechanistic simulation models
   - Network visualization and analysis tools

   .. rubric:: References

   1. Ventre E, et al. (2021).
   2. Ventre E, et al. (2023).
   2. Mauge Y, et Ventre, E. (2026).

   Author
   ------
   Elias Ventre
   Yann Mauge

   License
   -------
   MIT License



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/CardamomOT/config/index
   /autoapi/CardamomOT/inference/index
   /autoapi/CardamomOT/model/index
   /autoapi/CardamomOT/tools/index


Attributes
----------

.. autoapisummary::

   CardamomOT.DEFAULT_DATA_FOLDER
   CardamomOT.DEFAULT_CARDAMOM_FOLDER


Classes
-------

.. autoapisummary::

   CardamomOT.NetworkModel


Functions
---------

.. autoapisummary::

   CardamomOT.get_project_directories
   CardamomOT.get_default_parameters
   CardamomOT.kon_ref_vector
   CardamomOT.select_DEgenes
   CardamomOT.extract_degradation_rates
   CardamomOT.plot_data_distrib
   CardamomOT.plot_data_pmf_temporal
   CardamomOT.plot_data_pmf_total
   CardamomOT.compare_marginals
   CardamomOT.plot_data_umap_toref
   CardamomOT.plot_data_umap_altogether
   CardamomOT.animate_dynamic_grns
   CardamomOT.interactive_edit_positions
   CardamomOT.enforce_min_distance
   CardamomOT.filter_edges
   CardamomOT.compute_max_variation_times
   CardamomOT.predict_cell_types
   CardamomOT.train_classifier
   CardamomOT.plot_cell_type_proportions


Package Contents
----------------

.. py:function:: get_project_directories(project_path: pathlib.Path) -> Dict[str, pathlib.Path]

   Get all standard subdirectories for a CARDAMOM project.

   :param project_path: Root directory of the project.

   :returns: data, cardamom, results.
   :rtype: Dictionary with keys


.. py:function:: get_default_parameters() -> Dict[str, Any]

   Get all default parameters as a dictionary.

   :returns: Dictionary of all default parameter values.


.. py:data:: DEFAULT_DATA_FOLDER
   :value: 'Data'


.. py:data:: DEFAULT_CARDAMOM_FOLDER
   :value: 'cardamom'


.. py:function:: kon_ref_vector(y_prot, kz, theta_inter, theta_basal) -> numpy.ndarray

.. py:function:: select_DEgenes(vect_t, vect_samples_id, vect_celltype_id, proba, list_genes, n_genes_tokeep_temporal=[1000], n_genes_tokeep_celltype=[1000], limit_min=0.01, verb=0)

.. py:function:: extract_degradation_rates(df, gene_list, cell_line=None, similarity_threshold=np.linspace(0.99, 0.01, 10))

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


.. py:function:: plot_data_distrib(data_reference, data_simulated, t_data, t_sim, names, file_from, file_to, complement)

.. py:function:: plot_data_pmf_temporal(data_reference, kz, c, data_simulated, t_real, t_netw, names, file_from, file_to, complement)

   Affiche les histogrammes et superpose le mélange de binomiales négatives.


.. py:function:: plot_data_pmf_total(weights, data_reference, kz, c, data_simulated, names, file_from, file_to, complement)

   Affiche les histogrammes et superpose le mélange de binomiales négatives.


.. py:function:: compare_marginals(data_real, data_netw, t_real, t_netw, genes, file_from, file_to, complement)

.. py:function:: plot_data_umap_toref(data_ref_base, data_sim_base, times, file_from, file_to, complement, logscale=True, cell_rd=None)

.. py:function:: plot_data_umap_altogether(data_real_base, data_ref_base, data_beta_base, data_theta_base, data_sim_base, times_data, times_simul, file_from, file_to, complement, logscale=True, cell_rd=None)

.. py:function:: animate_dynamic_grns(pos, inter_t, gene_names, timepoints, output_path, G)

.. py:function:: interactive_edit_positions(G, pos, labels, save_path)

.. py:function:: enforce_min_distance(pos, min_dist=0.1)

.. py:function:: filter_edges(matrix, ref, abs_thresh=1, rel_thresh=0.1)

.. py:function:: compute_max_variation_times(inter_t)

.. py:function:: predict_cell_types(adata_new, clf, label_key='cell_type')

.. py:function:: train_classifier(adata, label_key='cell_type')

.. py:function:: plot_cell_type_proportions(adatas, labels, label_key='cell_type', colors=None)

