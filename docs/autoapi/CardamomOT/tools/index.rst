CardamomOT.tools
================

.. py:module:: CardamomOT.tools

.. autoapi-nested-parse::

   Tools for verifying the quality of the inference.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/CardamomOT/tools/characterize_cell_type/index
   /autoapi/CardamomOT/tools/joint_distances/index
   /autoapi/CardamomOT/tools/marginals/index
   /autoapi/CardamomOT/tools/umap/index
   /autoapi/CardamomOT/tools/visualize_network/index


Functions
---------

.. autoapisummary::

   CardamomOT.tools.plot_data_distrib
   CardamomOT.tools.plot_data_pmf_temporal
   CardamomOT.tools.plot_data_pmf_total
   CardamomOT.tools.compare_marginals
   CardamomOT.tools.plot_data_umap_toref
   CardamomOT.tools.plot_data_umap_altogether
   CardamomOT.tools.animate_dynamic_grns
   CardamomOT.tools.interactive_edit_positions
   CardamomOT.tools.enforce_min_distance
   CardamomOT.tools.filter_edges
   CardamomOT.tools.compute_max_variation_times
   CardamomOT.tools.predict_cell_types
   CardamomOT.tools.train_classifier
   CardamomOT.tools.plot_cell_type_proportions


Package Contents
----------------

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

