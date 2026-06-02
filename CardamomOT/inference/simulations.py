"""
Numerical solvers and utilities for gene regulatory network simulations.

This module provides both Numba-accelerated low-level functions and
higher-level classes for the two simulation regimes used by CARDAMOM:

* deterministic ordinary differential equation (ODE) dynamics handled by
  :class:`ApproxODE`;
* stochastic piecewise-deterministic Markov process (PDMP) modelling
  bursty protein production encapsulated in :class:`BurstyPDMP`.

Helper routines such as ``base_kon_vector`` and ``kon_ref`` are also
included, since they are reused across inference and trajectory modules.
"""
from typing import Any
import numpy as np
from numba import njit
import logging

from CardamomOT.logging import get_logger

# module logger
logger = get_logger(__name__)


@njit
def base_kon_vector(theta_basal, theta_inter, y_prot) -> np.ndarray:
    n_cells, G = y_prot.shape
    Gm1, n_net = theta_basal.shape[0], theta_basal.shape[1]
    Z = np.zeros((n_cells, Gm1, n_net))
    result = np.empty((n_cells, Gm1, n_net + 1))
    for i in range(n_cells):
        for j in range(Gm1):
            z_max: float = -np.inf
            for k in range(n_net):
                Z[i, j, k] = theta_basal[j, k]
                for g in range(G):
                    Z[i, j, k] += y_prot[i, g] * theta_inter[g, j, k]
                if Z[i, j, k] > z_max:
                    z_max = Z[i, j, k]
            denom = np.exp(-z_max)
            result[i, j, 0] = np.exp(-z_max)
            for k in range(n_net):
                val = np.exp(Z[i, j, k] - z_max)
                denom += val
                result[i, j, k + 1] = val
            result[i, j] /= denom

    return result


@njit
def kon_ref(y_prot, kz, theta_inter, theta_basal):
    result = base_kon_vector(theta_basal, theta_inter, y_prot)
    res = np.zeros(theta_basal.shape[0])
    for j in range(theta_basal.shape[0]):
        for k in range(result.shape[-1]):
            res[j] += kz[j, k] * result[-1, j, k]
    return res


@njit
def flow(time, d1, P, ns=1):
    """
    Deterministic flow for the bursty model.
    """
    Pnew = P*np.exp(-time*d1)
    for s in range(ns):
        Pnew[s] = P[s]  # preserve stimulus dimensions 0..ns-1
    return Pnew

@njit
def step_ode(d1, ks, inter, basal, dt, scale, P, ns=1):
        """
        Euler step for the deterministic limit model.
        """
        a = kon_ref(P, ks, inter, basal)
        Pnew = scale*a + (P[-1, :] - scale*a)*np.exp(-d1*dt)
        for s in range(ns):
            Pnew[s] = P[-1, s]  # preserve stimulus dimensions 0..ns-1
        return Pnew


class ApproxODE:
    """
    ODE version of the network model
    """
    def __init__(self, d, basal, inter) -> None:
        # Kinetic parameters
        G = basal.shape[0]
        # Network parameters
        self.basal: Any = basal
        self.inter: Any = inter
        # Default state
        type: list[tuple[str, str]] = [('P', 'float')]
        self.state: np.ndarray[Any, np.dtype[Any]] = np.array([(0) for i in range(G)], dtype=type)
        # Simulation parameter
        self.euler_step = 1e-2/np.max(d)


    def simulation(self, d1, ks, timepoints, scale, ns=1, verb=False):
        """
        Simulation of the deterministic limit model, which is relevant when
        promoters and mRNA are much faster than proteins.
        1. Nonlinear ODE system involving proteins only
        2. Mean level of mRNA given protein levels
        """
        G = d1.size
        dt = self.euler_step
        if np.size(timepoints) > 1:
            dt = np.min([dt, np.min(timepoints[1:] - timepoints[:-1])])
        type: list[tuple[str, str]] = [('P', 'float64')]
        sim = []
        T, c = 0, 0
        # Core loop for simulation and recording
        for t in timepoints:
            while T < t:
                self.state['P'] = step_ode(d1, ks, self.inter, self.basal, dt, scale, self.state['P'].reshape((1, -1)), ns)
                T += dt
                c += 1
            sim += [np.array([(self.state['P'][i]) for i in range(1,G)], dtype=type)]
        # Display info about steps
        if verb:
            if c > 0:
                logger.info("ODE simulation used %d steps (step size = %.5f)", c, dt)
            else:
                logger.debug("ODE simulation used no step")
        return np.array(sim)
    

class BurstyPDMP:
    """
    Bursty PDMP version of the network model (promoters not described)
    """
    def __init__(self, ks, basal, inter, ns=1) -> None:
        # Kinetic parameters
        G = basal.shape[0]
        # Network parameters
        self.basal: Any = basal
        self.inter: Any = inter
        # Default state
        type: list[tuple[str, str]] = [('P', 'float')]
        self.state: np.ndarray[Any, np.dtype[Any]] = np.array([(0) for i in range(G)], dtype=type)
        # Thinning constant: sum of max burst rates over gene dims only (exclude stim dims 0..ns-1)
        self.thin_cst = np.sum(np.max(ks[ns:, :], axis=1))


    def step(self, d1, ks, c, scale, ns=1):
        """
        Compute the next jump and the next step of the
        thinning method, in the case of the bursty model.
        """
        tau = self.thin_cst

        # 0. Draw the waiting time before the next jump
        U = np.random.exponential(scale=1/tau)

        # 1. Update the continuous states (stimulus dims preserved inside flow)
        P = flow(U, d1, self.state['P'], ns)
        self.state['P'] = P

        # 2. Compute the next jump
        v = kon_ref(P.reshape((1, -1)), ks, self.inter, self.basal)/tau # i = 1, ..., G-1 : burst of prot i
        for s in range(1, ns):
            v[s] = 0.0  # stim2..stimN must not receive bursts
        v[0] = 1.0 - np.sum(v[1:]) # i = 0 : no change (phantom jump)
        # Deal robustly with precision errors
        i: np.signedinteger[Any] = np.searchsorted(np.cumsum(v), np.random.random(), side='right')
        jump: np.bool_ = i > 0 # Test if jump is a true (i > 0) or phantom jump (i == 0)
        if jump:
            r = (c/scale)[i]
            self.state['P'][i] += np.random.exponential(1/r)

        return U, jump


    def simulation(self, d1, ks, c, timepoints, scale, ns=1, verb=False):
        """
        Exact simulation of the network in the bursty PDMP case.
        """
        G = self.basal.shape[0]
        types: list[tuple[str, str]] = [('P', 'float64')]
        sim = [] # List of states to be recorded
        c0, c1 = 0, 0 # Jump counts (phantom and true)
        T = 0
        # Core loop for simulation and recording
        Told, state_old = T, self.state.copy()
        for t in timepoints:
            while T < t:
                Told, state_old = T, self.state.copy()
                U, jump = self.step(d1, ks, c, scale, ns)
                T += U
                if jump:
                    c1 += 1
                else:
                    c0 += 1
            P = flow(t - Told, d1, state_old['P'], ns)
            sim += [np.array([(P[i]) for i in range(1,G)], dtype=types)]
        # Update the current state
        self.state['P'] = P
        # Display info about jumps
        if verb:
            ctot: int = c0 + c1
            if ctot > 0:
                logger.info(
                    "Exact simulation used %d jumps (including %d phantom jumps %.2f%%)",
                    ctot,
                    c0,
                    100 * c0 / ctot,
                )
            else:
                logger.debug("Exact simulation used no jump")
        return np.array(sim)

    def simulation_with_growth(self, d1, ks, c, timepoints, scale, prolif_fn, ns=1, verb=False):
        """
        Same as simulation but also accumulates the growth log-weight:
            log_weight = integral R(P(t)) dt   (forward Riemann sum over PDMP steps)

        This is the forward-only estimate of the full growth log-weight R·Δt.
        For a combined forward+backward correction (trapezoidal rule using both
        start and end protein states), see the resampling step in
        simulate_trajectories_unitary, which does not use this method.

        prolif_fn : callable (1, n_proteins) -> (1,) array of net growth rates.
        Returns (sim_array, log_weight).
        """
        G = self.basal.shape[0]
        types: list[tuple[str, str]] = [('P', 'float64')]
        sim = []
        c0, c1 = 0, 0
        T = 0
        log_weight: float = 0.0
        Told, state_old = T, self.state.copy()
        for t in timepoints:
            while T < t:
                Told, state_old = T, self.state.copy()
                U, jump = self.step(d1, ks, c, scale, ns)
                T += U
                # Accumulate R * U (full rate R, not divided by 2)
                P_genes = self.state['P'][ns:].reshape(1, -1)
                R = prolif_fn(P_genes)
                log_weight += float(R[0]) * U
                if jump:
                    c1 += 1
                else:
                    c0 += 1
            P = flow(t - Told, d1, state_old['P'], ns)
            sim += [np.array([(P[i]) for i in range(1, G)], dtype=types)]
        self.state['P'] = P
        if verb:
            ctot: int = c0 + c1
            if ctot > 0:
                logger.info(
                    "Exact simulation (with growth) used %d jumps (%d phantom %.2f%%)",
                    ctot, c0, 100 * c0 / ctot,
                )
            else:
                logger.debug("Exact simulation (with growth) used no jump")
        return np.array(sim), log_weight
    

class Simulation:
    """
    Basic object to store simulations.
    """
    def __init__(self, t, p, log_weight: float = 0.0) -> None:
        self.t: Any = t          # Time points
        self.p: Any = p          # Proteins
        self.log_weight: float = log_weight  # Accumulated integral of R(P) dt


def simulate_next_prot_ode(d, a, basal, inter, t, scale, **kwargs) -> Simulation:
        """
        Perform simulation of the network model (ODE version).
        """
        p0 = kwargs.get('P0')
        ns = kwargs.get('ns', 1)
        stim_vals = kwargs.get('stim_vals', None)  # schedule values at current time ti
        verb = kwargs.get('verb', False)
        if np.size(t) == 1:
            t = np.array([t])
        if np.any(t != np.sort(t)):
            msg = 'Time points must appear in increasing order'
            raise ValueError(msg)
        network: ApproxODE = ApproxODE(d, basal, inter)
        if p0 is not None:
            network.state['P'][ns:] = p0[ns:]  # gene dims only
        # Stimulus dims: schedule is authoritative; fall back to p0 if not provided
        if stim_vals is not None:
            for s in range(ns):
                network.state['P'][s] = stim_vals[s]
        else:
            if p0 is not None:
                network.state['P'][1:ns] = p0[1:ns]  # stim2..stimN from p0
            network.state['P'][0] = 1  # backward-compat: stim1 always active
        sim = network.simulation(d, a, t, scale, ns=ns, verb=verb)
        p = sim['P']
        return Simulation(t, p)


def simulate_next_prot_pdmp(d, a, c, basal, inter, t, scale, **kwargs) -> Simulation:
        """
        Perform simulation of the network model (PDMP version).

        Optional kwargs:
          prolif_fn : callable (1, n_proteins) -> (1,) or None.
              When provided, the PDMP micro-steps are used to accumulate
              log_weight = integral R(P(t)) dt stored in Simulation.log_weight.
        """
        p0 = kwargs.get('P0')
        ns = kwargs.get('ns', 1)
        stim_vals = kwargs.get('stim_vals', None)  # schedule values at current time ti
        verb = kwargs.get('verb', False)
        prolif_fn = kwargs.get('prolif_fn', None)
        if np.size(t) == 1:
            t = np.array([t])
        if np.any(t != np.sort(t)):
            msg = 'Time points must appear in increasing order'
            raise ValueError(msg)
        network: BurstyPDMP = BurstyPDMP(a, basal, inter, ns=ns)
        if p0 is not None:
            network.state['P'][ns:] = p0[ns:]  # gene dims only
        # Stimulus dims: schedule is authoritative; fall back to p0 if not provided
        if stim_vals is not None:
            for s in range(ns):
                network.state['P'][s] = stim_vals[s]
        else:
            if p0 is not None:
                network.state['P'][1:ns] = p0[1:ns]  # stim2..stimN from p0
            network.state['P'][0] = 1  # backward-compat: stim1 always active
        if prolif_fn is not None:
            sim_arr, log_weight = network.simulation_with_growth(d, a, c, t, scale, prolif_fn, ns=ns, verb=verb)
            p = sim_arr['P']
            return Simulation(t, p, log_weight=log_weight)
        sim = network.simulation(d, a, c, t, scale, ns=ns, verb=verb)
        p = sim['P']
        return Simulation(t, p)