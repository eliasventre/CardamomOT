CardamomOT.inference.network
============================

.. py:module:: CardamomOT.inference.network

.. autoapi-nested-parse::

   Core routines for network inference used in trajectory-based loops.

   This module implements the optimization objectives, gradients, and
   penalization schemes required to learn regulatory interactions from
   multi-modal single-cell data.  It is designed to be imported lazily in
   order to avoid heavy dependencies when only other parts of the package
   are needed.



Attributes
----------

.. autoapisummary::

   CardamomOT.inference.network.logger
   CardamomOT.inference.network.seuil
   CardamomOT.inference.network.check_gradient
   CardamomOT.inference.network.alpha
   CardamomOT.inference.network.eps_CE
   CardamomOT.inference.network.EPS
   CardamomOT.inference.network.sc
   CardamomOT.inference.network.r_elasticnet


Functions
---------

.. autoapisummary::

   CardamomOT.inference.network.smoothed_l1_penalization
   CardamomOT.inference.network.grad_smoothed_l1_penalization
   CardamomOT.inference.network.theta_penalization
   CardamomOT.inference.network.grad_theta_penalization
   CardamomOT.inference.network.l2_penalization
   CardamomOT.inference.network.grad_l2_penalization
   CardamomOT.inference.network.final_theta_penalization
   CardamomOT.inference.network.grad_final_theta_penalization
   CardamomOT.inference.network.main_loss
   CardamomOT.inference.network.grad_main_loss
   CardamomOT.inference.network.base_kon
   CardamomOT.inference.network.objective
   CardamomOT.inference.network.grad_theta
   CardamomOT.inference.network.objective_refinement
   CardamomOT.inference.network.grad_correc
   CardamomOT.inference.network.core_inference
   CardamomOT.inference.network.refine_inference
   CardamomOT.inference.network.main_loop_inference
   CardamomOT.inference.network.inference_network


Module Contents
---------------

.. py:data:: logger

.. py:data:: seuil
   :value: 1e-05


.. py:data:: check_gradient
   :value: 0


.. py:data:: alpha
   :value: 1


.. py:data:: eps_CE
   :value: 1e-06


.. py:data:: EPS
   :value: 1e-16


.. py:data:: sc
   :value: 0.001


.. py:data:: r_elasticnet
   :value: 0.5


.. py:function:: smoothed_l1_penalization(array, l1, sc=sc)

.. py:function:: grad_smoothed_l1_penalization(array, l1, sc=sc) -> numpy.ndarray

.. py:function:: theta_penalization(theta, l)

.. py:function:: grad_theta_penalization(theta, l) -> numpy.ndarray

.. py:function:: l2_penalization(theta, l2)

.. py:function:: grad_l2_penalization(theta, l2)

.. py:function:: final_theta_penalization(theta, l)

.. py:function:: grad_final_theta_penalization(theta, l)

.. py:function:: main_loss(y_pred, y_true, l, loss, sc=sc, eps=eps_CE)

.. py:function:: grad_main_loss(y_pred, y_true, l, loss, sc=sc, eps=eps_CE)

.. py:function:: base_kon(theta_basal, theta_inter, y_prot) -> numpy.ndarray

   More robust version with clipping of exponentials to avoid overflow.


.. py:function:: objective(X, weights_samples, ys, ypr, yp, ypm, yk, ks, G, g, n_networks, n_samples, theta_ref, ref_network, l_pen, proba, weight_prev, loss, final, constrain_basal_uniform=0.0, basal_free_mask=None)

   Loss function for a single gene — joint over all samples.

   theta shape: (G + n_samples, n_networks)
     rows 0..G-1      : interaction weights
     rows G..G+ns-1   : per-sample basal values

   constrain_basal_uniform : float >= 0
       Penalty strength that pushes free-sample basals toward their mean.
       Samples whose basal is pinned by a KO/OV prior are excluded (basal_free_mask).
   basal_free_mask : (n_samples,) bool array or None
       True for samples that are NOT pinned by basal_ref for this gene.


.. py:function:: grad_theta(X, weights_samples, ys, ypr, yp, ypm, yk, ks, G, g, n_networks, n_samples, theta_ref, ref_network, l_pen, proba, weight_prev, loss, final, constrain_basal_uniform=0.0, basal_free_mask=None)

   Gradient of objective w.r.t. X = theta.ravel().


.. py:function:: objective_refinement(X, correc_ref, inter, basal, weights_samples, ys, ypr, yp, ypm, yk, ks, diag, G, g, n_networks, n_samples, l_pen, proba, weight_prev, loss, final)

   Loss for the refinement step with per-sample basal.

   correc shape: (G + n_samples, n_networks)
     rows 0..G-1    : multiplicative corrections for inter
     rows G..G+ns-1 : multiplicative corrections for per-sample basal
   basal : (n_samples, n_networks)


.. py:function:: grad_correc(X, correc_ref, inter, basal, weights_samples, ys, ypr, yp, ypm, yk, ks, diag, G, g, n_networks, n_samples, l_pen, proba, weight_prev, loss, final)

   Gradient of objective_refinement w.r.t. correction factors X.


.. py:function:: core_inference(y_samples, y_proba, y_prot, y_prot_mod, y_kon, theta_init, theta_ref, ref_network, ks, G, g, n_networks, n_samples, proba, l_pen, weight_prev=0.5, loss='CE', final=0, constrain_basal_uniform=0.0, basal_free_mask=None, G_tol=None)

   Joint L-BFGS-B optimisation of interactions + per-sample basals for one gene.

   theta_init shape: (G + n_samples, n_networks)
     rows 0..G-1    : interaction weights
     rows G..G+ns-1 : per-sample basal values
   Returns theta_final of same shape.


.. py:function:: refine_inference(y_samples, y_proba, y_prot, y_prot_mod, y_kon, inter, basal, theta_ref, ks, G, g, n_networks, n_samples, proba, l_pen, weight_prev=0.5, loss='CE', correc_ref=0, final=0, G_tol=None)

   Refinement step: optimise multiplicative correction factors over inter and per-sample basal.

   basal : (n_samples, n_networks)
   Returns updated (inter, basal).


.. py:function:: main_loop_inference(g, y_samples, y_proba, y_prot, y_prot_mod, y_kon, theta_init, theta_ref, ks, G, n_networks, n_samples, proba, l_gen, scale, inter_tmp, basal_tmp, inter, basal, ref_network, weight_prev=0.5, loss='CE', final=0, constrain_basal_uniform=0.0, basal_free_mask=None, G_tol=None)

   Joint inference (inter + per-sample basal) for a single gene g.

   theta_init : (G + n_samples, n_networks)
   basal      : (n_samples, n_networks)  — will be updated in-place (copy returned)
   inter      : (G, n_networks)          — will be updated in-place (copy returned)
   basal_free_mask : (n_samples,) bool — True for samples NOT pinned by basal_ref for gene g
   Returns (basal, inter, basal_tmp, inter_tmp).


.. py:function:: inference_network(y_samples, y_kon, y_proba, y_prot, y_prot_mod, ks, n_stimuli=1, proba=0, ref_network=None, basal_init=None, inter_init=None, basal_ref=None, inter_ref=None, scale=100, weight_prev=0.5, loss='CE', final=0, samples_id=None, constrain_basal_uniform=0.0) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]

   Joint network inference: interactions shared across samples, basal per sample.

   Returns (basal, inter, basal_tmp, inter_tmp).
   basal shape: (n_samples, G, n_networks)

   constrain_basal_uniform : float >= 0
       When > 0, adds a penalty that pushes per-sample basals toward their common
       mean for each gene. Samples whose basal is pinned by a non-zero basal_ref
       (KO/OV priors) are excluded from the penalty for that gene.


