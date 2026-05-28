CardamomOT.tools.marginals
==========================

.. py:module:: CardamomOT.tools.marginals


Functions
---------

.. autoapisummary::

   CardamomOT.tools.marginals.negative_binomial_pmf
   CardamomOT.tools.marginals.compute_mixture_weights
   CardamomOT.tools.marginals.plot_data_pmf_temporal
   CardamomOT.tools.marginals.plot_data_pmf_total
   CardamomOT.tools.marginals.plot_data_distrib
   CardamomOT.tools.marginals.compare_marginals


Module Contents
---------------

.. py:function:: negative_binomial_pmf(x, k, c)

.. py:function:: compute_mixture_weights(data_reference, kz, c)

   Calcule les poids de chaque composante pour chaque cellule et chaque gène.


.. py:function:: plot_data_pmf_temporal(data_reference, kz, c, data_simulated, t_real, t_netw, names, file_from, file_to, complement)

   Affiche les histogrammes et superpose le mélange de binomiales négatives.


.. py:function:: plot_data_pmf_total(weights, data_reference, kz, c, data_simulated, names, file_from, file_to, complement)

   Affiche les histogrammes et superpose le mélange de binomiales négatives.


.. py:function:: plot_data_distrib(data_reference, data_simulated, t_data, t_sim, names, file_from, file_to, complement)

.. py:function:: compare_marginals(data_real, data_netw, t_real, t_netw, genes, file_from, file_to, complement)

