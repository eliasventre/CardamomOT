CardamomOT.inference.degradations
=================================

.. py:module:: CardamomOT.inference.degradations

.. autoapi-nested-parse::

   Utilities for degradation rate inference and temporal epsilon estimation.

   This module provides PyTorch models and helper functions used by the
   CARDAMOM pipeline when learning gene-specific degradation parameters
   from protein dynamics.  It includes the
   :class:`GeneRegulatoryODE_softmax` neural ODE model and the
   :func:`infer_ratio_d0_d1_unitary` routine among other utilities.



Attributes
----------

.. autoapisummary::

   CardamomOT.inference.degradations.logger
   CardamomOT.inference.degradations.times


Classes
-------

.. autoapisummary::

   CardamomOT.inference.degradations.KonCorrectionMLP
   CardamomOT.inference.degradations.GeneRegulatoryODE_softmax


Functions
---------

.. autoapisummary::

   CardamomOT.inference.degradations.build_kon_fn
   CardamomOT.inference.degradations.train_kon_correction_mlp
   CardamomOT.inference.degradations.infer_ratio_d0_d1_unitary
   CardamomOT.inference.degradations.inference_degradation_prot
   CardamomOT.inference.degradations.infer_ratio_d0_d1_full
   CardamomOT.inference.degradations.predict_trajectory
   CardamomOT.inference.degradations.compare_trajectories_umap


Module Contents
---------------

.. py:data:: logger

.. py:function:: build_kon_fn(ks, theta_inter, bias, device='cpu')

   Return a function kon(X_numpy_or_torch) -> numpy array (batch, G)
   The function accepts either numpy arrays or torch tensors; it returns numpy.


.. py:class:: KonCorrectionMLP(G_tot: int, G_genes: int, hidden_dim: int = 8)

   Bases: :py:obj:`torch.nn.Module`


   G_genes MLPs indépendants entraînés en parallèle via joblib.
   Gène i prend [P_i, kon_beta_i] → delta_mu_i.
   At init: delta_mu = 1 everywhere.


   .. py:attribute:: G_tot


   .. py:attribute:: G_genes


   .. py:attribute:: ns


   .. py:attribute:: hidden_dim
      :value: 8



   .. py:attribute:: W1


   .. py:attribute:: b1


   .. py:attribute:: W2


   .. py:attribute:: b2


   .. py:method:: forward(y_prot: numpy.ndarray, kon_beta: numpy.ndarray) -> numpy.ndarray


.. py:function:: train_kon_correction_mlp(y_prot: numpy.ndarray, kon_harissa: numpy.ndarray, kon_beta: numpy.ndarray, ns: int, hidden_dim: int = 8, n_epochs: int = 1000, lr: float = 0.01, seuil: float = 1e-08, fixed_r: float = 10.0, n_jobs: int = -1) -> KonCorrectionMLP

.. py:class:: GeneRegulatoryODE_softmax(G, d_init, ks, theta_inter, bias, n_stimuli=1, stim_vals=None, device='cpu', kon_mlp=None, lambda_scale=1000.0)

   Bases: :py:obj:`torch.nn.Module`


   ODE model for gene regulatory dynamics with generalized softmax-based kon.
   Learns gene-specific degradation rates (d) and scale factors.
   Optionally applies a KonCorrectionMLP multiplicative correction (Harissa branch).


   .. py:attribute:: G


   .. py:attribute:: n_stimuli
      :value: 1



   .. py:attribute:: device


   .. py:attribute:: d_param


   .. py:attribute:: scale_param


   .. py:attribute:: n_modes


   .. py:attribute:: lambda_scale


   .. py:attribute:: kon_mlp
      :value: None



   .. py:method:: set_ratio_interpolation(t_start, t_end, ratio_t0, ratio_t1, lambda_mlp)

      Configure time-interpolated ratio mix for the next odeint call.

      :param t_start: boundaries of the current interval (floats).
      :param t_end: boundaries of the current interval (floats).
      :param ratio_t0: mean observed g = kon_harissa/kon_beta at
                       ``t_start`` / ``t_end``, shape ``(G_genes,)``.
      :param ratio_t1: mean observed g = kon_harissa/kon_beta at
                       ``t_start`` / ``t_end``, shape ``(G_genes,)``.
      :param lambda_mlp: mix weight (1 = pure interpolation, 0 = pure MLP).



   .. py:method:: forward(t, X)

      Compute dX/dt for a given state X at time t.
      Includes learned scaling of theta_inter and bias.
      When kon_mlp is set, applies a frozen per-gene multiplicative correction.
      If ``set_ratio_interpolation`` has been called, the correction is mixed
      with the linearly interpolated observed training ratios at weight
      ``lambda_mlp``.



.. py:function:: infer_ratio_d0_d1_unitary(X_prot, times, bias, theta_inter, ks, d_learned_temporal, k1_vec, n_stimuli=1, stim_schedule=None, samples_data=None, method='dopri5', rtol=1e-06, atol=1e-08, lambda_deg=0.0, prior_eps=None, outlier_quantile=0.95, min_kon=1e-06, eps_min=0.01, eps_max=100.0, verbose=True, kon_mlp=None, scale=1.0) -> tuple[numpy.ndarray, numpy.ndarray]

   Estimate ε_i = d1_i/d0_i from ODE residuals (bursty-PDMP variance matching).

   **Theoretical background.**

   In the bursty PDMP actually simulated (see :class:`BurstyPDMP`), the burst
   parameters at interval ``cnt`` are:

   * burst **rate**  : ``λ_i = k1_i · kon_norm_i(P) · d0_i``
   * burst **size**  : ``Exp(k1_i · d0_i / (scale · d1_i))``
     i.e. mean burst = ``scale · d1_i / (k1_i · d0_i)``

   This ensures the mean-field ODE limit is ``dP_i/dt = d1_i(kon_i/k1_i − P_i)``.

   By Campbell's theorem (leading term for small dt):

   .. math::

       \operatorname{Var}[\Delta X_i] \approx
       \lambda_i \cdot E[B_i^2] \cdot dt
       = 2\,dt\cdot k_{on,i}(P)\cdot\mathrm{scale}\,
         \frac{d_{1,i}^2}{k_{1,i}^2\,d_{0,i}}
       = \varepsilon_i \cdot h_{ic}

   where :math:`\varepsilon_i = d_{1,i}/d_{0,i}` and

   .. math::

       h_{ic} = 2\,dt\cdot\mathrm{scale}\cdot
                \frac{d_{1,i}}{k_{1,i}^2}\cdot k_{on,i}(P_c)

   The regularised method-of-moments estimator is:

   .. math::

       \varepsilon_i =
       \frac{\sum_c (X_{\mathrm{pred},c} - X_{1,c})^2
              + \lambda\,\varepsilon_{\mathrm{prior},i}}
             {\sum_c h_{ic} + \lambda}

   :param X_prot: ``(N, G)`` protein observations (all timepoints).
   :param times: ``(N,)`` time label for each row.
   :param bias: GRN bias — ``(T-1, G, n_nets)`` or
                ``(T-1, n_samples, G, n_nets)``.
   :param theta_inter: ``(T-1, G, G, n_nets)`` GRN interaction tensor.
   :param ks: ``(n_modes, G)`` softmax burst-rate amplitudes
              (already multiplied by ``scale_proteins``).
   :param d_learned_temporal: ``(T-1, G)`` learned protein degradation rates d1.
   :param k1_vec: ``(G,)`` max burst rate × scale (= ``k1 * scale``).
   :param n_stimuli: Number of stimulus columns (default 1).
   :param stim_schedule: ``{float: np.ndarray(ns)}`` stimulus values per time.
   :param samples_data: ``(N,)`` per-cell sample index (optional).
   :param method: ODE solver settings.
   :param rtol: ODE solver settings.
   :param atol: ODE solver settings.
   :param lambda_deg: Tikhonov weight; 0 = pure MoM, large → prior.
   :param prior_eps: ``(G,)`` prior for ε = d1/d0. Defaults to all-ones.
   :param outlier_quantile: Fraction of cells to keep (sorted by residual).
   :param min_kon: Minimum kon value for a cell to contribute.
   :param eps_min: Output clipping bounds for ε = d1/d0.
   :param eps_max: Output clipping bounds for ε = d1/d0.
   :param verbose: Log per-interval diagnostics.
   :param kon_mlp: Optional :class:`KonCorrectionMLP` (Harissa branch).
   :param scale: Protein scale (``self.scale_proteins``). Default 1.0.

   :returns: ``(T-1, G)`` per-interval ε = d1/d0 estimates.
             eps_global   : ``(G,)`` global ε pooled over all intervals.
   :rtype: eps_temporal


.. py:function:: inference_degradation_prot(X_prot, times, bias, theta_inter, ks, d=None, n_epochs=500, lr=0.01, method='dopri5', rtol=1e-06, atol=1e-08, print_every=50, batch_size=None, verbose=True, n_stimuli=1, stim_schedule=None, samples_data=None, kon_mlp=None, lambda_scale=1000.0, lambda_deg=0.0, lambda_mlp=0.0, g_obs_train=None) -> tuple[numpy.ndarray, numpy.ndarray, float, GeneRegulatoryODE_softmax]

   Estimate degradation rates and scaling factors from protein time-course data.

   When ``bias`` is 3-D (n_samples, G, n_modes-1) and ``samples_data`` is provided,
   one ODE module is created per sample with its own bias while ``d_param`` and
   ``scale_param`` are shared, so a single optimizer refines the shared kinetics
   from all samples jointly.

   :returns: Learned degradation rates, shape (G,).
             ``scale_learned``: Learned scaling parameters, shape (G,).
   :rtype: ``d_learned``


.. py:function:: infer_ratio_d0_d1_full(X_prot, times, bias, theta_inter, ks, d_learned, k1_vec, kon_mlp, prior_d1d0=None, n_stimuli=1, stim_schedule=None, samples_data=None, method='dopri5', rtol=1e-05, atol=1e-07, n_steps=10, lambda_deg=0.0, lambda_mlp=0.5, g_obs_train=None, min_h=0.0001, verbose=True, clip_lo=0.01, clip_hi=100.0) -> tuple[numpy.ndarray, numpy.ndarray]

   Infer d1/d0 from full simulated trajectories via regularised LS.

   **Theoretical background.**

   To first order in ``d1/d0``, the MLP correction factor satisfies:

   .. math::

       1 - g_i(P(t)) \approx \frac{d_{1,i}}{d_{0,i}} \cdot h_i(P(t))

   where ``g_i = kon_harissa_i / kon_beta_i`` (learned by *kon_mlp*) and

   .. math::

       h_i(P) = \sum_j \frac{\partial\ln kon_i}{\partial X_j}
                \cdot (kon^{\rm eff}_j - X_j)

   where the ``d_{1,j}`` factor is *not* included in ``h_i`` (uniform-d1
   approximation: ``d_{1,j} \approx d_{1,i}`` for all j, so it becomes
   part of the ratio ``d_{1,i}/d_{0,i}`` that the LS estimates directly).

   **Fitting strategy.**

   Instead of evaluating only at the terminal state, this function simulates
   full trajectories with ``n_steps`` intermediate evaluation points and fits
   ``r_i = d1_i/d0_i`` by minimising over all *(cell, timestep)* pairs:

   .. math::

       \mathcal{L}(r_i) = \sum_{c,t} \bigl[y_{ict} - r_i\,h_{ict}\bigr]^2
                          + \lambda\,(r_i - r_{\rm prior,i})^2

   with ``y_{ict} = 1 - g_i(P_c(t))``.  The closed-form solution is:

   .. math::

       r_i^* = \frac{\sum h\,y + \lambda\,r_{\rm prior}}{\sum h^2 + \lambda}

   Only *(cell, timestep)* pairs where ``h_i`` and ``y_i`` have the same
   sign (i.e. where the formula predicts ``r_i > 0``) and where
   ``|h_i| > min_h`` are included.

   **Usage in base.py**::

       ratios_temporal, ratios_global = infer_ratio_d0_d1_full(...)
       # temporal (one ratio per interval):
       self.ratios[cnt] = 1.0 / ratios_temporal[cnt]
       # global (one shared ratio):
       self.ratios[:] = (1.0 / ratios_global)[None, :]

   :param X_prot: ``(N, G)`` protein observations across all timepoints.
   :param times: ``(N,)`` time label for each observation.
   :param bias: GRN bias — ``(G, n_modes-1)`` or ``(n_samples, G, n_modes-1)``.
   :param theta_inter: GRN interactions — ``(G, G, n_modes-1)`` or
                       ``(n_samples, G, G, n_modes-1)``.
   :param ks: Mode amplitudes, shape ``(n_modes, G)``.
   :param d_learned: Protein degradation rates, shape ``(G,)`` (constant) or
                     ``(T-1, G)`` (temporal). When 2-D, interval ``cnt``
                     uses ``d_learned[cnt]`` for the ODE simulation.
   :param k1_vec: Kept for interface compatibility (not used internally).
   :param kon_mlp: Trained :class:`KonCorrectionMLP`.
   :param prior_d1d0: ``(G,)`` prior for d1/d0 used when ``lambda_deg > 0``.
                      Defaults to all-ones (quasi-stationary limit).
   :param n_stimuli: Number of stimulus columns (default 1).
   :param stim_schedule: Dict ``{float_time: np.ndarray(ns,)}``; defaults to
                         all-ones when ``None``.
   :param samples_data: ``(N,)`` per-cell sample index.
   :param method: ODE solver method (default ``"dopri5"``).
   :param rtol: ODE solver tolerances.
   :param atol: ODE solver tolerances.
   :param n_steps: Number of intermediate ODE evaluation points per
                   interval (default 10; total = n_steps+1 incl. t0, t1).
   :param lambda_deg: Tikhonov regularisation weight — same role as in
                      :func:`inference_degradation_prot`.  ``0`` = pure LS;
                      large values → prior.
   :param lambda_mlp: Mix weight between the linearly interpolated observed
                      training ratios and the MLP prediction.  For a simulated
                      state at time ``t`` in ``[t1, t2]`` the effective ratio
                      used as ``y_i = 1 - g_eff`` is:

                      .. math::

                          g_{\rm eff} = \lambda_{\rm mlp}
                              \Bigl[
                                  \frac{t_2-t}{t_2-t_1}\,r_1
                                + \frac{t-t_1}{t_2-t_1}\,r_2
                              \Bigr]
                            + (1-\lambda_{\rm mlp})\,g(P,\,k_{\rm on}(P))

                      where ``r_1`` / ``r_2`` are the mean observed MLP ratios
                      at the training timepoints ``t_1`` / ``t_2``
                      (shape ``(G_genes,)``).
                      ``1.0`` = pure linear interpolation of observed g;
                      ``0.0`` = pure MLP prediction (original behaviour).
   :param min_h: Minimum ``|h_i|`` to include a *(cell, timestep)* pair.
   :param verbose: Log per-interval diagnostics.
   :param clip_lo: Final clamp on the estimated d1/d0.
   :param clip_hi: Final clamp on the estimated d1/d0.

   :returns: ``(T-1, G)`` per-interval d1/d0 estimates.
             ratios_global   : ``(G,)``    global d1/d0 pooled over all intervals.

             Stimulus columns (``[:ns]``) are fixed to 1.0 in both outputs.
   :rtype: ratios_temporal


.. py:function:: predict_trajectory(ode_func, X0, t_span, method='dopri5', rtol=1e-06, atol=1e-08, stim_vals=None)

   Simulate a trajectory given an initial state and trained ODE model.


.. py:function:: compare_trajectories_umap(ode_func, X_prot, times, method='dopri5')

   Compare real and simulated trajectories using UMAP projection.


.. py:data:: times

