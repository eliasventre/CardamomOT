"""
Numerical solvers and utilities for gene regulatory network simulations.

This module provides both Numba-accelerated low-level functions and
higher-level classes for the two simulation regimes used by CARDAMOM:

* deterministic ordinary differential equation (ODE) dynamics handled by
  :class:`ApproxODE`;
* stochastic piecewise-deterministic Markov process (PDMP) modelling
  bursty protein production encapsulated in :class:`BurstyPDMP`;
* the bursty PDMP with explicit mRNA (bursts of mRNA, proteins translated from them),
  :class:`BurstyPDMPmRNA`.

The network model and its simulators reproduce those of the Harissa package
(U. Herbach, https://github.com/ulysseherbach/harissa; model of Herbach et al. 2017): the
bursty PDMP (bursts at rate kon(P), exponential burst sizes, exact deterministic flow between
the jumps), its exact simulation by adaptive thinning on an upper bound of kon along the flow,
and the mean-field ODE. They are re-implemented here and adapted to the CardamomOT setting:
sigmoid kon between the extreme modes of the mixture, several stimuli, per-sample parameters,
protein dilution at the birth rate of the cells, time-dependent rates, and the chromatin regions.

References:
    Herbach U, Bonnaffoux A, Espinasse T, Gandrillon O (2017). Inferring gene regulatory networks
    from single-cell data: a mechanistic approach. BMC Systems Biology 11:105.
    Harissa: https://github.com/ulysseherbach/harissa

Each of them optionally carries the accessibility of chromatin regions (:class:`ChromatinModel`), a
Markov chain on discrete levels that gates the regulated part of kon and is regulated by the proteins.

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


@njit(cache=True)
def base_kon_vector(theta_basal, theta_inter, y_prot) -> np.ndarray:
    """(1 - sigmoid(Z), sigmoid(Z)) per cell and gene, Z = basal + prot @ inter (basal (G,), inter (G, G))."""
    n_cells, G = y_prot.shape
    Gm1 = theta_basal.shape[0]
    result = np.empty((n_cells, Gm1, 2))
    for i in range(n_cells):
        for j in range(Gm1):
            z = theta_basal[j]
            for g in range(G):
                z += y_prot[i, g] * theta_inter[g, j]
            # Stable sigmoid: exp of a negative argument only (no inf / inf for |z| > 709)
            e = np.exp(-abs(z))
            denom = e + 1.0
            if z >= 0:
                result[i, j, 0] = e / denom
                result[i, j, 1] = 1.0 / denom
            else:
                result[i, j, 0] = 1.0 / denom
                result[i, j, 1] = e / denom

    return result


@njit(cache=True)
def kon_ref(y_prot, kz, theta_inter, theta_basal):
    # kon = k_low * (1 - sigmoid) + k_high * sigmoid (lowest and highest modes)
    result = base_kon_vector(theta_basal, theta_inter, y_prot)
    res = np.zeros(theta_basal.shape[0])
    for j in range(theta_basal.shape[0]):
        k_high = kz[j, 0]
        for k in range(1, kz.shape[1]):
            if kz[j, k] > k_high:
                k_high = kz[j, k]
        res[j] += kz[j, 0] * result[-1, j, 0]
        res[j] += k_high * result[-1, j, 1]
    return res


@njit(cache=True)
def kon_chrom(y_prot, kz, theta_inter, theta_basal, cbar):
    # kon = k_low + (k_high - k_low) * cbar * sigmoid, written so that cbar = 1 gives kon_ref bit for bit
    result = base_kon_vector(theta_basal, theta_inter, y_prot)
    res = np.zeros(theta_basal.shape[0])
    for j in range(theta_basal.shape[0]):
        k_high = kz[j, 0]
        for k in range(1, kz.shape[1]):
            if kz[j, k] > k_high:
                k_high = kz[j, k]
        s1 = result[-1, j, 1]
        res[j] += kz[j, 0] * (result[-1, j, 0] + (1.0 - cbar[j]) * s1)
        res[j] += k_high * (cbar[j] * s1)
    return res


@njit(cache=True)
def chrom_values(lvl, c_levels):
    """Accessibility c (R,) = mean over alleles of the levels lvl (A, R) (indices into c_levels (R, Kmax))."""
    A, R = lvl.shape
    c = np.zeros(R)
    for r in range(R):
        for a in range(A):
            c[r] += c_levels[r, lvl[a, r]]
        c[r] /= A
    return c


@njit(cache=True)
def chrom_gate(c, w, has_region):
    """Gate cbar_g = sum_r w[g, r] c_r, 1 for genes without region (and stimuli)."""
    G = w.shape[0]
    cbar = np.ones(G)
    for g in range(G):
        if has_region[g]:
            z = 0.0
            for r in range(w.shape[1]):
                z += w[g, r] * c[r]
            cbar[g] = z
    return cbar


@njit(cache=True)
def chrom_open_close(P, inter_c, basal_c, kbar_c, k_close_const):
    """Opening / closing rates (R,): k_open = kbar sigmoid(Zc); k_close = kbar (1 - sigmoid(Zc)) (mirror,
    k_close_const NaN) or k_close_const."""
    R = basal_c.shape[0]
    ko = np.empty(R)
    kc = np.empty(R)
    for r in range(R):
        z = basal_c[r]
        for g in range(P.shape[0]):
            z += P[g] * inter_c[g, r]
        e = np.exp(-abs(z))  # stable sigmoid
        s_open = 1.0 / (1.0 + e) if z >= 0 else e / (1.0 + e)
        ko[r] = kbar_c[r] * s_open
        if np.isnan(k_close_const[r]):
            kc[r] = kbar_c[r] * (e / (1.0 + e) if z >= 0 else 1.0 / (1.0 + e))
        else:
            kc[r] = k_close_const[r]
    return ko, kc


@njit(cache=True)
def chrom_jump_rates(lvl, c_levels, n_levels, ko, kc):
    """Rates (2 A R,) of the jumps to adjacent levels, index 2 (a R + r) (+0 up, +1 down):
    up k_open (1 - c_k) / (c_k+1 - c_k), down k_close c_k / (c_k - c_k-1), so that dE[c]/dt = k_open (1 - c) - k_close c."""
    A, R = lvl.shape
    rates = np.zeros(2 * A * R)
    for a in range(A):
        for r in range(R):
            k = lvl[a, r]
            i = 2 * (a * R + r)
            if k < n_levels[r] - 1:
                rates[i] = ko[r] * (1.0 - c_levels[r, k]) / (c_levels[r, k + 1] - c_levels[r, k])
            if k > 0:
                rates[i + 1] = kc[r] * c_levels[r, k] / (c_levels[r, k] - c_levels[r, k - 1])
    return rates


@njit(cache=True)
def step_ode_chrom(d1, ks, inter, basal, dt, scale, P, ns, cfac, c, w, has_region, inter_c, basal_c,
                   kbar_c, k_close_const):
    """Exact step of (P, c) with rates frozen over dt; c follows dc/dt = k_open (1 - c) - k_close c."""
    a = kon_chrom(P, ks, inter, basal, chrom_gate(c, w, has_region)) * cfac
    Pnew = scale*a + (P[-1, :] - scale*a)*np.exp(-d1*dt)
    for s in range(ns):
        Pnew[s] = P[-1, s]
    ko, kc = chrom_open_close(P[-1, :], inter_c, basal_c, kbar_c, k_close_const)
    cnew = c.copy()
    for r in range(c.shape[0]):
        k = ko[r] + kc[r]
        if k > 0:
            ceq = ko[r] / k
            cnew[r] = ceq + (c[r] - ceq) * np.exp(-k * dt)
    return Pnew, cnew


@njit(cache=True)
def step_ode_gate(d1, ks, inter, basal, dt, scale, P, ns, cfac, cbar):
    """step_ode with the kon gated by cbar (chromatin levels fixed over dt)."""
    a = kon_chrom(P, ks, inter, basal, cbar) * cfac
    Pnew = scale*a + (P[-1, :] - scale*a)*np.exp(-d1*dt)
    for s in range(ns):
        Pnew[s] = P[-1, s]
    return Pnew


class ChromatinModel:
    """
    Accessibility of R regulatory regions and their regulation by the proteins.

    w (G, R): weights of the regions of each gene (rows summing to 1, zero rows = gene without region, gate 1);
    basal_c (R,), inter_c (G, R): Zc = basal_c + P @ inter_c (rows = regulators, stimuli included);
    kbar_c (R,): chromatin speed; k_close_const (R,): constant closing rate, NaN = mirror (default);
    c_levels: levels of each region, (K,) shared or (R, Kmax) NaN-padded, first 0 and last 1 (default [0, 1]);
    diploid: two independent chains per region, c_r = mean of the alleles;
    c0: initial level index (int, or (A, R)), 0 = closed. The current levels are kept in lvl (A, R).
    """
    def __init__(self, w, basal_c, inter_c, kbar_c, c_levels=None, k_close_const=None, diploid=False, c0=0):
        self.w = np.ascontiguousarray(w, dtype=np.float64)
        G, R = self.w.shape
        self.R = R
        self.basal_c = np.asarray(basal_c, dtype=np.float64).reshape(R)
        self.inter_c = np.ascontiguousarray(inter_c, dtype=np.float64).reshape(G, R)
        self.kbar_c = np.asarray(kbar_c, dtype=np.float64).reshape(R)
        self.k_close_const = (np.full(R, np.nan) if k_close_const is None
                              else np.asarray(k_close_const, dtype=np.float64).reshape(R))
        self.has_region = np.abs(self.w).sum(axis=1) > 0
        lev = np.array([0.0, 1.0]) if c_levels is None else np.asarray(c_levels, dtype=np.float64)
        if lev.ndim == 1:
            lev = np.tile(lev, (R, 1))
        self.c_levels = np.ascontiguousarray(lev.reshape(R, -1) if R > 0 else np.zeros((0, 2)))
        self.n_levels = np.sum(~np.isnan(self.c_levels), axis=1).astype(np.int64)
        for r in range(R):
            v = self.c_levels[r, :self.n_levels[r]]
            if (self.n_levels[r] < 2 or np.any(np.isnan(self.c_levels[r, self.n_levels[r]:]))
                    or v[0] != 0.0 or v[-1] != 1.0 or np.any(np.diff(v) <= 0)):
                raise ValueError(f"c_levels of region {r} must increase from 0 to 1 (NaN padding at the end): {v}")
        self.n_alleles = 2 if diploid else 1
        self.lvl = np.zeros((self.n_alleles, R), dtype=np.int64)
        self.lvl[:] = c0
        self.lvl = np.minimum(self.lvl, self.n_levels[None, :] - 1)
        self.thin_bound = self._thin_bound()

    def _thin_bound(self):
        """Upper bound of the total jump rate: per allele and region, max over levels of the up + down rates."""
        tot = 0.0
        for r in range(self.R):
            v = self.c_levels[r, :self.n_levels[r]]
            fu = np.zeros(len(v))
            fd = np.zeros(len(v))
            fu[:-1] = (1.0 - v[:-1]) / np.diff(v)
            fd[1:] = v[1:] / np.diff(v)
            kb = self.kbar_c[r]
            if np.isnan(self.k_close_const[r]):
                b = kb * np.max(np.maximum(fu, fd))  # k_open + k_close = kbar
            else:
                b = np.max(kb * fu + self.k_close_const[r] * fd)
            tot += b
        return self.n_alleles * tot

    def values(self):
        return chrom_values(self.lvl, self.c_levels)

    def jump_rates(self, P):
        ko, kc = chrom_open_close(P, self.inter_c, self.basal_c, self.kbar_c, self.k_close_const)
        return chrom_jump_rates(self.lvl, self.c_levels, self.n_levels, ko, kc)

    def apply_jump(self, j):
        """Jump j of chrom_jump_rates: allele / region j // 2, up (j even) or down."""
        a, r = divmod(j // 2, self.R)
        self.lvl[a, r] += 1 if j % 2 == 0 else -1

    def gate(self):
        return chrom_gate(self.values(), self.w, self.has_region)

    def thinned_jump(self, P):
        """Accept / reject a candidate jump at the thinning rate thin_bound (one uniform draw)."""
        v = self.jump_rates(P) / self.thin_bound
        j = np.searchsorted(np.cumsum(v), np.random.random(), side='right')
        if j < v.shape[0] and v[j] > 0:
            self.apply_jump(j)
            return True
        return False

    def alleles(self):
        """Levels (A, R) of each allele."""
        return self.c_levels[np.arange(self.R)[None, :], self.lvl]


@njit(cache=True)
def flow(time, d1, P, ns=1):
    """
    Deterministic flow for the bursty model.
    """
    Pnew = P*np.exp(-time*d1)
    for s in range(ns):
        Pnew[s] = P[s]  # preserve stimulus dimensions 0..ns-1
    return Pnew

@njit(cache=True)
def step_ode(d1, ks, inter, basal, dt, scale, P, ns=1, cfac=np.ones(1)):
        """
        Euler step for the deterministic limit model; cfac: target factor d1 / (d1 + b) with dilution at the
        birth rate b (d1 then being d1 + b), ones otherwise.
        """
        a = kon_ref(P, ks, inter, basal) * cfac
        Pnew = scale*a + (P[-1, :] - scale*a)*np.exp(-d1*dt)
        for s in range(ns):
            Pnew[s] = P[-1, s]  # preserve stimulus dimensions 0..ns-1
        return Pnew


class ApproxODE:
    """
    ODE version of the network model (mean field of the bursty PDMP, as the ODE simulation of Harissa,
    Herbach et al. 2017).
    """
    def __init__(self, d, basal, inter, chrom=None, chrom_stochastic=True) -> None:
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
        # Chromatin (ChromatinModel or None): Markov chain on the levels (exact thinning along the ODE),
        # or its mean accessibility c following dc/dt = k_open (1 - c) - k_close c (chrom_stochastic=False)
        self.chrom = chrom if (chrom is not None and chrom.R > 0) else None
        self.chrom_stochastic = chrom_stochastic
        if self.chrom is not None:
            self.c = self.chrom.values()
            kmax = np.max(np.concatenate([self.chrom.kbar_c, np.nan_to_num(self.chrom.k_close_const)]))
            if kmax > 0:
                self.euler_step = min(self.euler_step, 1e-2/kmax)


    def simulation(self, d1, ks, timepoints, scale, ns=1, verb=False, cfac=None):
        """
        Simulation of the deterministic limit model, which is relevant when
        promoters and mRNA are much faster than proteins.
        1. Nonlinear ODE system involving proteins only
        2. Mean level of mRNA given protein levels
        Dilution at the birth rate b: d1 = d1 + b and cfac = d1 / (d1 + b) (target factor).
        """
        G = d1.size
        cfac = np.ones(G) if cfac is None else np.asarray(cfac, dtype=np.float64)
        dt = self.euler_step
        if np.size(timepoints) > 1:
            dt = np.min([dt, np.min(timepoints[1:] - timepoints[:-1])])
        type: list[tuple[str, str]] = [('P', 'float64')]
        sim = []
        T, c = 0, 0
        ch = self.chrom
        c_rec, ca_rec = [], []
        if ch is not None and self.chrom_stochastic:
            sim = self._simulation_chrom_jumps(d1, ks, timepoints, scale, ns, cfac, dt, c_rec, ca_rec)
            self.c_rec, self.c_alleles_rec = np.array(c_rec), np.array(ca_rec)
            return sim
        # Core loop for simulation and recording
        for t in timepoints:
            while T < t:
                if ch is None:
                    self.state['P'] = step_ode(d1, ks, self.inter, self.basal, dt, scale, self.state['P'].reshape((1, -1)), ns, cfac)
                else:
                    self.state['P'], self.c = step_ode_chrom(
                        d1, ks, self.inter, self.basal, dt, scale, self.state['P'].reshape((1, -1)), ns, cfac,
                        self.c, ch.w, ch.has_region, ch.inter_c, ch.basal_c, ch.kbar_c, ch.k_close_const)
                T += dt
                c += 1
            sim += [np.array([(self.state['P'][i]) for i in range(1,G)], dtype=type)]
            if ch is not None:
                c_rec.append(self.c.copy())
        self.c_rec = np.array(c_rec)
        # Display info about steps
        if verb:
            if c > 0:
                logger.info("ODE simulation used %d steps (step size = %.5f)", c, dt)
            else:
                logger.debug("ODE simulation used no step")
        return np.array(sim)

    def _simulation_chrom_jumps(self, d1, ks, timepoints, scale, ns, cfac, dt, c_rec, ca_rec):
        """ODE for P between the chromatin jumps, candidate jumps at rate thin_bound (exact thinning)."""
        ch = self.chrom
        G = d1.size
        type: list[tuple[str, str]] = [('P', 'float64')]
        sim = []
        bound = ch.thin_bound
        T = 0.0
        T_jump = np.random.exponential(1 / bound) if bound > 0 else np.inf
        cbar = ch.gate()
        for t in timepoints:
            while T < t:
                hit = T + dt >= T_jump
                h = T_jump - T if hit else dt
                self.state['P'] = step_ode_gate(d1, ks, self.inter, self.basal, h, scale,
                                                self.state['P'].reshape((1, -1)), ns, cfac, cbar)
                if hit:
                    T = T_jump
                    if ch.thinned_jump(self.state['P']):
                        cbar = ch.gate()
                    T_jump = T + np.random.exponential(1 / bound)
                else:
                    T += h
            sim += [np.array([(self.state['P'][i]) for i in range(1, G)], dtype=type)]
            c_rec.append(ch.values())
            ca_rec.append(ch.alleles())
        return np.array(sim)


class BurstyPDMP:
    """
    Bursty PDMP version of the network model (promoters not described), protein-only: reproduces the bursty
    PDMP and the adaptive thinning of Harissa (Herbach et al. 2017; https://github.com/ulysseherbach/harissa),
    with the bursts acting directly on the proteins.
    """
    def __init__(self, ks, basal, inter, ns=1, chrom=None) -> None:
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
        # Chromatin (ChromatinModel or None): jumps of the region levels added to the thinning
        self.chrom = chrom if (chrom is not None and chrom.R > 0) else None
        if self.chrom is not None:
            self.thin_cst = self.thin_cst + self.chrom.thin_bound


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

        if self.chrom is not None:
            return U, self._jump_chrom(P, ks, c, scale, tau, ns)

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


    def _jump_chrom(self, P, ks, c, scale, tau, ns):
        """Next jump with chromatin: bursts with the gated kon (i = 1..G-1), then the region jumps (i >= G);
        same draws as without chromatin (one uniform, one exponential for a burst)."""
        ch = self.chrom
        G = P.shape[0]
        cbar = chrom_gate(ch.values(), ch.w, ch.has_region)
        v = kon_chrom(P.reshape((1, -1)), ks, self.inter, self.basal, cbar)/tau
        for s in range(1, ns):
            v[s] = 0.0
        vc = ch.jump_rates(P)/tau
        v[0] = 1.0 - (np.sum(v[1:]) + np.sum(vc))
        w = np.concatenate((v, vc))
        i = np.searchsorted(np.cumsum(w), np.random.random(), side='right')
        if 0 < i < G:
            r = (c/scale)[i]
            self.state['P'][i] += np.random.exponential(1/r)
            return True
        if G <= i < w.shape[0] and w[i] > 0:
            ch.apply_jump(i - G)
            return True
        return False


    def simulation(self, d1, ks, c, timepoints, scale, ns=1, verb=False):
        """
        Exact simulation of the network in the bursty PDMP case.
        """
        G = self.basal.shape[0]
        types: list[tuple[str, str]] = [('P', 'float64')]
        sim = [] # List of states to be recorded
        c0, c1 = 0, 0 # Jump counts (phantom and true)
        T = 0
        ch = self.chrom
        lvl_old, c_rec, lvl_rec = None, [], []
        # Core loop for simulation and recording
        Told, state_old = T, self.state.copy()
        if ch is not None:
            lvl_old = ch.lvl.copy()
        for t in timepoints:
            while T < t:
                Told, state_old = T, self.state.copy()
                if ch is not None:
                    lvl_old = ch.lvl.copy()
                U, jump = self.step(d1, ks, c, scale, ns)
                T += U
                if jump:
                    c1 += 1
                else:
                    c0 += 1
            P = flow(t - Told, d1, state_old['P'], ns)
            sim += [np.array([(P[i]) for i in range(1,G)], dtype=types)]
            if ch is not None:
                lvl_rec.append(lvl_old.copy())
                c_rec.append(chrom_values(lvl_old, ch.c_levels))
        # Update the current state
        self.state['P'] = P
        if ch is not None:
            ch.lvl[:] = lvl_old  # chromatin state at the last recorded time
            self.c_rec = np.array(c_rec)
            self.c_alleles_rec = np.array([ch.c_levels[np.arange(ch.R)[None, :], l] for l in lvl_rec])
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
    def __init__(self, t, p, c=None, c_alleles=None, m=None) -> None:
        self.t: Any = t          # Time points
        self.p: Any = p          # Proteins
        self.m: Any = m          # mRNA (explicit mRNA PDMP), continuous levels in count units
        self.c: Any = c          # Region accessibility (n_t, R), with chromatin
        self.c_alleles: Any = c_alleles  # Levels of each allele (n_t, A, R), PDMP with chromatin


def simulate_next_prot_ode(d, a, basal, inter, t, scale, **kwargs) -> Simulation:
        """
        Perform simulation of the network model (ODE version).
        """
        p0 = kwargs.get('P0')
        ns = kwargs.get('ns', 1)
        stim_vals = kwargs.get('stim_vals', None)  # schedule values at current time ti
        verb = kwargs.get('verb', False)
        cfac = kwargs.get('cfac', None)  # dilution: target factor d1 / (d1 + b), d being d1 + b
        if np.size(t) == 1:
            t = np.array([t])
        if np.any(t != np.sort(t)):
            msg = 'Time points must appear in increasing order'
            raise ValueError(msg)
        chrom = kwargs.get('chrom', None)  # ChromatinModel: (P, c) system
        network: ApproxODE = ApproxODE(d, basal, inter, chrom=chrom,
                                       chrom_stochastic=kwargs.get('chrom_stochastic', True))
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
        sim = network.simulation(d, a, t, scale, ns=ns, verb=verb, cfac=cfac)
        p = sim['P']
        if network.chrom is not None:
            return Simulation(t, p, c=network.c_rec, c_alleles=getattr(network, 'c_alleles_rec', None))
        return Simulation(t, p)


def simulate_next_prot_pdmp(d, a, c, basal, inter, t, scale, **kwargs) -> Simulation:
        """
        Perform simulation of the network model (PDMP version).
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
        chrom = kwargs.get('chrom', None)  # ChromatinModel (state updated in place)
        network: BurstyPDMP = BurstyPDMP(a, basal, inter, ns=ns, chrom=chrom)
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
        sim = network.simulation(d, a, c, t, scale, ns=ns, verb=verb)
        p = sim['P']
        if network.chrom is not None:
            return Simulation(t, p, c=network.c_rec, c_alleles=network.c_alleles_rec)
        return Simulation(t, p)

# ----------------------------------------------------------------------------------------------------------------
# Bursty PDMP with explicit mRNA (adapted from harissa.simulation.pdmp.BurstyPDMP, U. Herbach):
# mRNA bursts M_g += Exp(1 / c_g) at rate kon_g(P, c), dM/dt = -d0 M, dP/dt = s1 M - d1 P between the jumps.

@njit(cache=True)
def _mrna_kernel(time, d0, d1):
    """(exp(-d1 t) - exp(-d0 t)) / (d0 - d1), t exp(-d1 t) in the limit d0 = d1."""
    if abs(d0 - d1) <= 1e-12 * max(d0, d1):
        return time * np.exp(-d1 * time)
    return (np.exp(-d1 * time) - np.exp(-d0 * time)) / (d0 - d1)


@njit(cache=True)
def flow_mrna(time, d0, d1, s1, M, P, ns=1):
    """Exact flow of (M, P) between the jumps (stimulus dimensions 0..ns-1 preserved)."""
    G = M.shape[0]
    Mnew = M.copy()
    Pnew = P.copy()
    for g in range(ns, G):
        Mnew[g] = M[g] * np.exp(-time * d0[g])
        Pnew[g] = s1[g] * M[g] * _mrna_kernel(time, d0[g], d1[g]) + P[g] * np.exp(-time * d1[g])
    return Mnew, Pnew


@njit(cache=True)
def pmax_mrna(d0, d1, s1, M, P, ns=1):
    """Upper bound of P along the flow from (M, P): P + s1 M max_t kernel(t)."""
    G = M.shape[0]
    pmax = P.copy()
    for g in range(ns, G):
        if abs(d0[g] - d1[g]) <= 1e-12 * max(d0[g], d1[g]):
            tc = 1.0 / d1[g]
        else:
            tc = np.log(d0[g] / d1[g]) / (d0[g] - d1[g])
        pmax[g] = P[g] + s1[g] * M[g] * _mrna_kernel(tc, d0[g], d1[g])
    return pmax


@njit(cache=True)
def kon_bound_mrna(pmax, ks, inter, basal, ns=1):
    """Sum over the genes of an upper bound of kon along the flow: positive interactions at pmax, gate 1."""
    G = pmax.shape[0]
    tot = 0.0
    for j in range(ns, basal.shape[0]):
        z = basal[j]
        for g in range(G):
            if inter[g, j] > 0:
                z += pmax[g] * inter[g, j]
        k_high = ks[j, 0]
        for k in range(1, ks.shape[1]):
            if ks[j, k] > k_high:
                k_high = ks[j, k]
        sig = 1.0 / (1.0 + np.exp(-z))
        tot += (1 - sig) * ks[j, 0] + sig * k_high + 1e-10  # precision margin
    return tot


class BurstyPDMPmRNA:
    """
    Bursty PDMP with explicit mRNA (promoters not described), reproducing harissa.simulation.pdmp.BurstyPDMP
    (Herbach et al. 2017; https://github.com/ulysseherbach/harissa): same flow, burst sizes and adaptive
    thinning, adapted to the extensions below.

    ks (G, n_modes): burst rates (per hour, kz d0; extreme modes used); c_nb (G,): inverse mRNA burst size;
    d0, d1 (G,): mRNA and protein degradation; scale: protein level of the highest mode, s1 = scale d0 d1 c / k_high;
    dilution (G,): birth rate b added to the protein decay (d1 + b in the flow, s1 unchanged);
    stimulus dimensions 0..ns-1 constant, without bursts; chrom: ChromatinModel (state updated in place).
    Adaptive thinning as Harissa (bound of kon along the flow), plus the bound of the chromatin jumps.
    """
    def __init__(self, ks, c_nb, d0, d1, basal, inter, scale=1.0, ns=1, chrom=None, dilution=None) -> None:
        G = basal.shape[0]
        self.ks = np.ascontiguousarray(ks, dtype=np.float64)
        self.c_nb = np.asarray(c_nb, dtype=np.float64)
        self.d0 = np.asarray(d0, dtype=np.float64)
        d1 = np.asarray(d1, dtype=np.float64)
        self.d1 = d1 + (0.0 if dilution is None else np.asarray(dilution, dtype=np.float64))
        k_high = np.max(self.ks, axis=1)
        self.s1 = np.zeros(G)
        self.s1[ns:] = scale * self.d0[ns:] * d1[ns:] * self.c_nb[ns:] / np.maximum(k_high[ns:], 1e-300)
        self.basal = np.asarray(basal, dtype=np.float64)
        self.inter = np.ascontiguousarray(inter, dtype=np.float64)
        self.ns = ns
        self.M = np.zeros(G)
        self.P = np.zeros(G)
        self.chrom = chrom if (chrom is not None and chrom.R > 0) else None
        self.chrom_bound = 0.0 if self.chrom is None else self.chrom.thin_bound

    def step(self):
        """Next jump of the thinning method: waiting time, flow, then burst / chromatin jump / phantom."""
        ns = self.ns
        tau = kon_bound_mrna(pmax_mrna(self.d0, self.d1, self.s1, self.M, self.P, ns),
                             self.ks, self.inter, self.basal, ns) + self.chrom_bound
        U = np.random.exponential(scale=1 / tau)
        self.M, self.P = flow_mrna(U, self.d0, self.d1, self.s1, self.M, self.P, ns)
        G = self.P.shape[0]
        P2 = self.P.reshape((1, -1))
        if self.chrom is None:
            v = kon_ref(P2, self.ks, self.inter, self.basal) / tau
            vc = np.zeros(0)
        else:
            v = kon_chrom(P2, self.ks, self.inter, self.basal, self.chrom.gate()) / tau
            vc = self.chrom.jump_rates(self.P) / tau
        for s in range(1, ns):
            v[s] = 0.0
        v[0] = 1.0 - (np.sum(v[1:]) + np.sum(vc))
        w = np.concatenate((v, vc)) if vc.shape[0] else v
        i = np.searchsorted(np.cumsum(w), np.random.random(), side='right')
        if 0 < i < G:
            self.M[i] += np.random.exponential(1 / self.c_nb[i])
            return U, True
        if G <= i < w.shape[0] and w[i] > 0:
            self.chrom.apply_jump(i - G)
            return U, True
        return U, False

    def simulation(self, timepoints, verb=False):
        """Exact simulation; records (M, P) of the dimensions 1..G-1 and the chromatin at the time points."""
        ns = self.ns
        ch = self.chrom
        m_rec, p_rec, c_rec, ca_rec = [], [], [], []
        c0, c1 = 0, 0
        T = 0
        Told, M_old, P_old = T, self.M.copy(), self.P.copy()
        lvl_old = None if ch is None else ch.lvl.copy()
        for t in timepoints:
            while T < t:
                Told, M_old, P_old = T, self.M.copy(), self.P.copy()
                if ch is not None:
                    lvl_old = ch.lvl.copy()
                U, jump = self.step()
                T += U
                if jump:
                    c1 += 1
                else:
                    c0 += 1
            M, P = flow_mrna(t - Told, self.d0, self.d1, self.s1, M_old, P_old, ns)
            m_rec.append(M[1:].copy())
            p_rec.append(P[1:].copy())
            if ch is not None:
                c_rec.append(chrom_values(lvl_old, ch.c_levels))
                ca_rec.append(ch.c_levels[np.arange(ch.R)[None, :], lvl_old])
        # Current state: the one at the last recorded time
        self.M, self.P = M, P
        if ch is not None:
            ch.lvl[:] = lvl_old
        self.c_rec = np.array(c_rec) if ch is not None else None
        self.c_alleles_rec = np.array(ca_rec) if ch is not None else None
        if verb:
            ctot = c0 + c1
            if ctot > 0:
                logger.info("Exact simulation (mRNA) used %d jumps (including %d phantom jumps %.2f%%)",
                            ctot, c0, 100 * c0 / ctot)
        return np.array(m_rec), np.array(p_rec)


def simulate_next_mrna_pdmp(d0, d1, ks, c_nb, basal, inter, t, scale, **kwargs) -> Simulation:
    """
    Perform simulation of the network model (bursty PDMP with explicit mRNA).

    kwargs: P0, M0 (G,) initial states; ns; stim_vals (ns,) stimulus values (authoritative over P0);
    chrom (ChromatinModel, updated in place); dilution (G,) birth rates added to the protein decay; verb.
    Returns Simulation with p and m (dimensions 1..G-1 at the time points) and the chromatin if any.
    """
    p0, m0 = kwargs.get('P0'), kwargs.get('M0')
    ns = kwargs.get('ns', 1)
    stim_vals = kwargs.get('stim_vals', None)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(t != np.sort(t)):
        raise ValueError('Time points must appear in increasing order')
    network = BurstyPDMPmRNA(ks, c_nb, d0, d1, basal, inter, scale=scale, ns=ns, chrom=kwargs.get('chrom'),
                             dilution=kwargs.get('dilution'))
    if p0 is not None:
        network.P[ns:] = p0[ns:]
    if m0 is not None:
        network.M[ns:] = m0[ns:]
    if stim_vals is not None:
        network.P[:ns] = stim_vals[:ns]
    else:
        if p0 is not None:
            network.P[1:ns] = p0[1:ns]
        network.P[0] = 1
    m, p = network.simulation(t, verb=kwargs.get('verb', False))
    return Simulation(t, p, c=network.c_rec, c_alleles=network.c_alleles_rec, m=m)
