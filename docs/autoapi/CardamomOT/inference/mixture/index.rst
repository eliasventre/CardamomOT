CardamomOT.inference.mixture
============================

.. py:module:: CardamomOT.inference.mixture

.. autoapi-nested-parse::

   mixture.py
   -----------
   Core routines for mixture model inference used in the CARDAMOM pipeline.

   This module contains functions for estimating negative-binomial mixture
   parameters, kinetics inference, and related utilities. It was originally
   ported from legacy scripts; the refactored version centralizes logging,
   adds type hints, and provides comprehensive documentation.

   Public functions include:

   * ``infer_mixture``        – primary routine for mixture parameter learning
   * ``infer_kinetics_temporal`` – estimate gamma-Poisson kinetics over time
   * ``infer_kinetics_temporal_scaled`` – scaled kinetics version

   Auxiliary helpers and legacy utilities are retained for compatibility.



Attributes
----------

.. autoapisummary::

   CardamomOT.inference.mixture.logger
   CardamomOT.inference.mixture.EPS


Classes
-------

.. autoapisummary::

   CardamomOT.inference.mixture.NegativeBinomialMixtureEM


Functions
---------

.. autoapisummary::

   CardamomOT.inference.mixture.estim_gamma_poisson
   CardamomOT.inference.mixture.infer_kinetics_temporal
   CardamomOT.inference.mixture.infer_kinetics_temporal_scaled
   CardamomOT.inference.mixture.infer_kinetics_preserve_mean_values_assignment
   CardamomOT.inference.mixture.nb_logpmf_vectorized
   CardamomOT.inference.mixture.zinb_logpmf_vectorized
   CardamomOT.inference.mixture.predict_resp
   CardamomOT.inference.mixture.hard_em_scaled
   CardamomOT.inference.mixture.hard_em
   CardamomOT.inference.mixture.neg_log_likelihood_logits
   CardamomOT.inference.mixture.compute_nu_star
   CardamomOT.inference.mixture.ks_statistic
   CardamomOT.inference.mixture.infer_kinetics_scaled
   CardamomOT.inference.mixture.em_vectorized_nb_zinb_scaled
   CardamomOT.inference.mixture.compute_aic_for_params_scaled
   CardamomOT.inference.mixture.em_vectorized_nb_zinb
   CardamomOT.inference.mixture.compute_aic_for_params


Module Contents
---------------

.. py:data:: logger

.. py:data:: EPS
   :value: 1e-12


.. py:function:: estim_gamma_poisson(x, mod=0, a_init=0, b_init=0)

   Estimate parameters a and b of the Gamma-Poisson(a,b) distribution.


.. py:function:: infer_kinetics_temporal(x, times, a_init=np.ones(100), b_init=1, max_iter=100, seuil=0.001, tol=1e-06, verb=False) -> tuple[Any, Any]

   Original version used for initialization with discrete timepoints.


.. py:function:: infer_kinetics_temporal_scaled(x, s, times, a_init=np.ones(100), b_init=1, max_iter=100, seuil=0.001, tol=1e-06, verb=False) -> tuple[numpy.ndarray, numpy.floating[Any]]

   Scaled version of :func:`infer_kinetics_temporal`.
   Model: X_i | basin k ∼ NB(a_k, b / s_i).
   The only differences with the original are:

   - the gradient uses log(b/s_i) instead of log(b)
   - the analytic update for b uses Σ x_i/s_i instead of Σ x_i


.. py:function:: infer_kinetics_preserve_mean_values_assignment(x, resp, seuil=0.01, a_init=None, b_init=None, tol=1e-06, max_iter=100, damping=0.7, verb=False) -> tuple[Any, Any]

   Adapted version of ``infer_kinetics_temporal`` for EM when
   *preserve_mean_values* assignments are used.

   Analytically optimizes parameters ``a[0],...,a[K-1]`` and ``b`` (called
   ``c`` in other parts of the code) by maximizing the weighted log-
   likelihood under the provided responsibilities.

   Parameters:
   -----------
   x : array (N,)
       Observations (counts)
   resp : array (N, K)
       Responsibilities (probabilities of membership in each component)
   seuil : float
       Lower bound for ``a`` and ``b``
   a_init : array (K,) or None
       Initialization for ``a``
   b_init : float or None
       Initialization for ``b``
   damping : float
       Damping factor for Newton–Raphson (0 < damping <= 1);
       smaller values increase stability at the cost of speed.

   Returns:
   --------
   a : array (K,)
       Optimal shape parameters (ks in our notation)
   b : float
       Common dispersion parameter (c in our notation)


.. py:function:: nb_logpmf_vectorized(x, ks, c, s=None)

   Vectorized log-PMF of a Negative Binomial with optional cell-specific scaling.

   Model: X_i | z=k ∼ NB(ks_k, c / s_i)
   so that E[X_i | k] = s_i * ks_k / c (scaling multiplies the mean).

   :param x:
   :type x: (N,)   observations (integer counts)
   :param ks:
   :type ks: (K,)   shape parameters
   :param c:
   :type c: float  dispersion parameter for the gene (shared across components)
   :param s:
   :type s: (N,)   cell read-depth factors (median = 1)

   :returns: **logpmf**
   :rtype: (N, K)


.. py:function:: zinb_logpmf_vectorized(x, ks, c, pi_zero, s=None)

   ZINB log-pmf matrix.


.. py:function:: predict_resp(x, ks, c, s=None, pi=None, pi_zero=None, zi=None, forcing=1.0) -> tuple[Any, Any]

   Compute the responsibilities.


.. py:function:: hard_em_scaled(data, s, n_components, ks_init, c_init, seuil, tol=1e-06, max_iter_loop=200, basins_temporal=None, vect_t=None, preserve_mean_values=0, mean_forcing=1.0)

   Hard EM avec scaling cellulaire.
   Seuls changements vs hard_em:

   - E-step via predict_resp
   - M-step via infer_kinetics_temporal_scaled
   - _apply_temporal_constraints travaille sur x/s pour les moyennes


.. py:function:: hard_em(data, n_components, ks_init, c_init, seuil, tol=1e-06, max_iter_loop=200, basins_temporal=None, vect_t=None, preserve_mean_values=0, mean_forcing=1.0)

   Hard EM for a Negative Binomial mixture with temporal constraints.

   :param mean_forcing: Weight of the temporal constraint (0 = no constraint, 1 = strict constraint)
   :type mean_forcing: float


.. py:function:: neg_log_likelihood_logits(logits, data, r, p)

   logits : array (K,)
   data   : array (n,)
   r, p   : NB parameters (K,)


.. py:function:: compute_nu_star(data, r_components, p_components, nu_init=None, method='L-BFGS-B')

.. py:function:: ks_statistic(data_full, nu, r, p, repet=10, n_cells_init=200) -> Any | float

.. py:function:: infer_kinetics_scaled(x, s, resp, seuil=0.01, a_init=None, b_init=None, tol=1e-06, max_iter=100, damping=0.7, verb=False) -> tuple[Any, Any]

   Analytical M-step for NB mixture with cell-specific read-depth factors ``s_i``.

   Model: X_i | k  ~  NB(a_k,  c/s_i)
   The weighted log-likelihood under responsibilities ``resp`` is::

     ℓ(a, c) = Σ_i Σ_k r_{ik} [
         log Γ(x_i + a_k) - log Γ(a_k) - log Γ(x_i+1)
         + a_k * log(c/s_i) - (x_i + a_k) * log(1 + c/s_i)
     ]

   We optimize using Newton–Raphson on ``a_k`` and analytically close-form
   update for ``c``.

   :param x:
   :type x: (N,)   counts
   :param s:
   :type s: (N,)   read-depth factors (median = 1)
   :param resp:
   :type resp: (N, K) responsibilities

   :returns: * **a** (*(K,)  shape parameters*)
             * **b** (float dispersion ``c``  (such that mean_k = s_i * a_k / c))


.. py:function:: em_vectorized_nb_zinb_scaled(x, s, ks_init, c_init, pi_init=None, pi_zero_init=None, zi_mode=None, max_iter=200, tol=1e-06, seuil=0.01, damping=0.7, verbose=False)

   EM for NB mixture with cellular read depth factors ``s_i``.

   Same as :func:`em_vectorized_nb_zinb` but uses
   :func:`nb_logpmf_vectorized` in the E-step and
   :func:`infer_kinetics_scaled` in the M-step.

   :param x:
   :type x: (N,)   counts (integers)
   :param s:
   :type s: (N,)   per-cell read depth (median normalized to 1)
   :param (other parameters are identical to ``em_vectorized_nb_zinb``):


.. py:function:: compute_aic_for_params_scaled(x, s, ks, c, pi, pi_zero, zi_mode) -> tuple[Any, float]

   Compute the AIC with read depth for a given set of parameters.


.. py:function:: em_vectorized_nb_zinb(x, ks_init, c_init, pi_init=None, pi_zero_init=None, zi_mode=None, max_iter=200, tol=1e-06, seuil=0.01, damping=0.7, verbose=False) -> Any

   EM for NB/ZINB mixture with ANALYTICAL M-step via Newton-Raphson.

   Replaces moment-based estimates with direct optimization of the
   weighted log-likelihood, as in :func:`infer_kinetics_temporal`.


.. py:function:: compute_aic_for_params(x, ks, c, pi, pi_zero, zi_mode) -> tuple[Any, numpy.floating[Any]]

   Compute the AIC for a given set of parameters.


.. py:class:: NegativeBinomialMixtureEM(min_components=1, max_components=3, zi=None, refilter=0.0, hard_em=1, mean_forcing_em=1.0, tol=1e-05, max_iter_em=200, verbose=False, preserve_mean_values=0, compare_init_aic=True, damping=1.0)

   .. py:attribute:: min_components
      :type:  int
      :value: 1



   .. py:attribute:: max_components
      :type:  int
      :value: 3



   .. py:attribute:: zi
      :value: None



   .. py:attribute:: preserve_mean_values
      :type:  int
      :value: 0



   .. py:attribute:: hard_em
      :type:  int
      :value: 1



   .. py:attribute:: mean_forcing_em
      :type:  float
      :value: 1.0



   .. py:attribute:: refilter
      :type:  float
      :value: 0.0



   .. py:attribute:: tol
      :type:  float
      :value: 1e-05



   .. py:attribute:: max_iter_em
      :type:  int
      :value: 200



   .. py:attribute:: verbose
      :type:  bool
      :value: False



   .. py:attribute:: compare_init_aic
      :type:  bool
      :value: True



   .. py:attribute:: damping
      :type:  float
      :value: 1.0



   .. py:attribute:: best_model
      :value: None



   .. py:method:: fit(x, vect_t=None, vect_celltypes=None, quant_init=None, seuil=0.001, s=None)

      Fit the NB mixture model to data ``x``.

      :param x:
      :type x: (N,)  counts (integers)
      :param vect_t:
      :type vect_t: (N,)  optional time label per cell
      :param s: If ``None`` all factors are set to 1 (original behavior).
                If provided (e.g. from ``adata.obs['rd']``), the model
                accounts for scaling: X_i|k ~ NB(ks_k, c/s_i).
      :type s: (N,)  optional cell-specific read depth factors.



