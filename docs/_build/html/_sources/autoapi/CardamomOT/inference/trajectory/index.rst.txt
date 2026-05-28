CardamomOT.inference.trajectory
===============================

.. py:module:: CardamomOT.inference.trajectory

.. autoapi-nested-parse::

   Core functions for the inference of trajectories, mainly used in loop_trajectories



Functions
---------

.. autoapisummary::

   CardamomOT.inference.trajectory.minimal_repetition_choice
   CardamomOT.inference.trajectory.base_kon_vector
   CardamomOT.inference.trajectory.find_next_prot_mixed
   CardamomOT.inference.trajectory.find_next_prot
   CardamomOT.inference.trajectory.count_errors
   CardamomOT.inference.trajectory.kon_ref_vector
   CardamomOT.inference.trajectory.my_otdistance
   CardamomOT.inference.trajectory.inference_alpha
   CardamomOT.inference.trajectory.filter_network


Module Contents
---------------

.. py:function:: minimal_repetition_choice(N, M, seed=None)

.. py:function:: base_kon_vector(theta_basal, theta_inter, y_prot) -> numpy.ndarray

.. py:function:: find_next_prot_mixed(d1, P0, M0, M1, mode_init, mode_end, alpha, s, delta_t)

   Deterministic flow interpolating between two points


.. py:function:: find_next_prot(d1, P0, M0, M1, mode_init, mode_end, alpha, s, delta_t)

   Deterministic flow interpolating between two points


.. py:function:: count_errors(vect_prot, vect_kon, vect_proba, ks, Y, X, loss='CE', compute_with_proba=0, n_stimuli=1)

.. py:function:: kon_ref_vector(y_prot, kz, theta_inter, theta_basal) -> numpy.ndarray

.. py:function:: my_otdistance(vect_kon_init, vect_kon_end, vect_prot_init, vect_rna_init, vect_rna_end, vect_proba_init, vect_proba_end, mode_init, mode_end, alpha, s1, ks, d1, delta_t, basal, inter, loss='CE', compute_with_proba=1, n_iter=1, intensity_prior=1, q=0.9, n_stimuli=1, stim_vals=np.ones(1)) -> tuple[numpy.ndarray, numpy.ndarray]

.. py:function:: inference_alpha(d1, s1, alpha_init, y_kon_init_true, y_kon_init, y_prot_init, y_rna_init, y_kon_end_true, y_kon_end, y_prot_end, y_rna_end, mode_init, mode_end, basal, inter, ks, delta_t, tol=0.5, n_pas=25, samples_data=None)

.. py:function:: filter_network(T, N_traj, prot_traj, ks, basal_ref, inter_ref, seuil_intensity=0.05, seuil_variations=0.01, n_order=10, samples_data=None)

