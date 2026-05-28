CardamomOT.inference.simulations
================================

.. py:module:: CardamomOT.inference.simulations

.. autoapi-nested-parse::

   Numerical solvers and utilities for gene regulatory network simulations.

   This module provides both Numba-accelerated low-level functions and
   higher-level classes for the two simulation regimes used by CARDAMOM:

   * deterministic ordinary differential equation (ODE) dynamics handled by
     :class:`ApproxODE`;
   * stochastic piecewise-deterministic Markov process (PDMP) modelling
     bursty protein production encapsulated in :class:`BurstyPDMP`.

   Helper routines such as ``base_kon_vector`` and ``kon_ref`` are also
   included, since they are reused across inference and trajectory modules.



Attributes
----------

.. autoapisummary::

   CardamomOT.inference.simulations.logger


Classes
-------

.. autoapisummary::

   CardamomOT.inference.simulations.ApproxODE
   CardamomOT.inference.simulations.BurstyPDMP
   CardamomOT.inference.simulations.Simulation


Functions
---------

.. autoapisummary::

   CardamomOT.inference.simulations.base_kon_vector
   CardamomOT.inference.simulations.kon_ref
   CardamomOT.inference.simulations.flow
   CardamomOT.inference.simulations.step_ode
   CardamomOT.inference.simulations.simulate_next_prot_ode
   CardamomOT.inference.simulations.simulate_next_prot_pdmp


Module Contents
---------------

.. py:data:: logger

.. py:function:: base_kon_vector(theta_basal, theta_inter, y_prot) -> numpy.ndarray

.. py:function:: kon_ref(y_prot, kz, theta_inter, theta_basal)

.. py:function:: flow(time, d1, P, ns=1)

   Deterministic flow for the bursty model.


.. py:function:: step_ode(d1, ks, inter, basal, dt, scale, P, ns=1)

   Euler step for the deterministic limit model.


.. py:class:: ApproxODE(d, basal, inter)

   ODE version of the network model


   .. py:attribute:: basal
      :type:  Any


   .. py:attribute:: inter
      :type:  Any


   .. py:attribute:: state
      :type:  numpy.ndarray[Any, numpy.dtype[Any]]


   .. py:attribute:: euler_step


   .. py:method:: simulation(d1, ks, timepoints, scale, ns=1, verb=False)

      Simulation of the deterministic limit model, which is relevant when
      promoters and mRNA are much faster than proteins.
      1. Nonlinear ODE system involving proteins only
      2. Mean level of mRNA given protein levels



.. py:class:: BurstyPDMP(ks, basal, inter, ns=1)

   Bursty PDMP version of the network model (promoters not described)


   .. py:attribute:: basal
      :type:  Any


   .. py:attribute:: inter
      :type:  Any


   .. py:attribute:: state
      :type:  numpy.ndarray[Any, numpy.dtype[Any]]


   .. py:attribute:: thin_cst


   .. py:method:: step(d1, ks, c, scale, ns=1)

      Compute the next jump and the next step of the
      thinning method, in the case of the bursty model.



   .. py:method:: simulation(d1, ks, c, timepoints, scale, ns=1, verb=False)

      Exact simulation of the network in the bursty PDMP case.



.. py:class:: Simulation(t, p)

   Basic object to store simulations.


   .. py:attribute:: t
      :type:  Any


   .. py:attribute:: p
      :type:  Any


.. py:function:: simulate_next_prot_ode(d, a, basal, inter, t, scale, **kwargs) -> Simulation

   Perform simulation of the network model (ODE version).


.. py:function:: simulate_next_prot_pdmp(d, a, c, basal, inter, t, scale, **kwargs) -> Simulation

   Perform simulation of the network model (PDMP version).


