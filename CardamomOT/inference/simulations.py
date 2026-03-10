"""
Core functions for ODE and PDMP simulations.
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
def flow(time, d1, P):
    """
    Deterministic flow for the bursty model.
    """
    # Explicit solution of the ODE generating the flow
    Pnew = P*np.exp(-time*d1)
    Pnew[0] = P[0] # Discard stimulus
    return Pnew

@njit
def step_ode(d1, ks, inter, basal, dt, scale, P):
        """
        Euler step for the deterministic limit model.
        """
        a = kon_ref(P, ks, inter, basal) 
        Pnew = scale*a + (P[-1, :] - scale*a)*np.exp(-d1*dt)
        Pnew[0] = P[-1, 0] # Discard stimulus
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


    def simulation(self, d1, ks, timepoints, scale, verb=False):
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
                self.state['P'] = step_ode(d1, ks, self.inter, self.basal, dt, scale, self.state['P'].reshape((1, -1)))
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
    def __init__(self, ks, basal, inter) -> None:
        # Kinetic parameters
        G = basal.shape[0]
        # Network parameters
        self.basal: Any = basal
        self.inter: Any = inter
        # Default state
        type: list[tuple[str, str]] = [('P', 'float')]
        self.state: np.ndarray[Any, np.dtype[Any]] = np.array([(0) for i in range(G)], dtype=type)
        # Simulation parameter
        self.thin_cst = np.sum(np.max(ks[1:, :], axis=1))


    def step(self, d1, ks, c, scale):
        """
        Compute the next jump and the next step of the
        thinning method, in the case of the bursty model.
        """
        tau = self.thin_cst

        # 0. Draw the waiting time before the next jump
        U = np.random.exponential(scale=1/tau)

        # 1. Update the continuous states
        P = flow(U, d1, self.state['P'])
        self.state['P'] = P

        # 2. Compute the next jump
        v = kon_ref(P.reshape((1, -1)), ks, self.inter, self.basal)/tau # i = 1, ..., G-1 : burst of prot i
        v[0] = 1.0 - np.sum(v[1:]) # i = 0 : no change (phantom jump)
        # Deal robustly with precision errors
        i: np.signedinteger[Any] = np.searchsorted(np.cumsum(v), np.random.random(), side='right')
        jump: np.bool_ = i > 0 # Test if jump is a true (i > 0) or phantom jump (i == 0)
        if jump:
            r = (c/scale)[i]
            self.state['P'][i] += np.random.exponential(1/r)

        return U, jump


    def simulation(self, d1, ks, c, timepoints, scale, verb=False):
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
                U, jump = self.step(d1, ks, c, scale)
                T += U
                if jump:
                    c1 += 1
                else:
                    c0 += 1
            P = flow(t - Told, d1, state_old['P'])
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
    

class Simulation:
    """
    Basic object to store simulations.
    """
    def __init__(self, t, p) -> None:
        self.t: Any = t # Time points
        self.p: Any = p # Proteins


def simulate_next_prot_ode(d, a, basal, inter, t, scale, **kwargs) -> Simulation:
        """
        Perform simulation of the network model (ODE version).
        """
        # Get keyword arguments
        p0 = kwargs.get('P0')
        verb = kwargs.get('verb', False)
        if np.size(t) == 1:
            t = np.array([t])
        if np.any(t != np.sort(t)):
            msg = 'Time points must appear in increasing order'
            raise ValueError(msg)
        network: ApproxODE = ApproxODE(d, basal, inter)
        # Burnin simulation without stimulus
        if p0 is not None:
            network.state['P'][1:] = p0[1:]
        # Activate the stimulus
        network.state['P'][0] = 1
        # Final simulation with stimulus
        sim = network.simulation(d, a, t, scale, verb)
        p = sim['P']
        return Simulation(t, p)


def simulate_next_prot_pdmp(d, a, c, basal, inter, t, scale, **kwargs) -> Simulation:
        """
        Perform simulation of the network model (ODE version).
        """
        # Get keyword arguments
        p0 = kwargs.get('P0')
        verb = kwargs.get('verb', False)
        if np.size(t) == 1:
            t = np.array([t])
        if np.any(t != np.sort(t)):
            msg = 'Time points must appear in increasing order'
            raise ValueError(msg)
        network: BurstyPDMP = BurstyPDMP(a, basal, inter)
        # Burnin simulation without stimulus
        if p0 is not None:
            network.state['P'][1:] = p0[1:]
        # Activate the stimulus
        network.state['P'][0] = 1
        # Final simulation with stimulus
        sim = network.simulation(d, a, c, t, scale, verb)
        p = sim['P']
        return Simulation(t, p)