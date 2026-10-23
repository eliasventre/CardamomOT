"""
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

Auxiliary helpers and legacy utilities are retained for compatibility.
"""

from typing import Any

import numpy as np
from numpy import floating
from scipy.special import gammaln, logsumexp, psi, polygamma, zeta
from scipy.stats import nbinom
from scipy.optimize import minimize
import ot

from CardamomOT.logging import get_logger
from .sampling import stratified_choice
from .depth_model import burst_frequency, set_depth_model

# module-level logger
logger = get_logger(__name__)

EPS = 1e-16

# ---------------------------
# Original utility functions (retained for initialization)
# ---------------------------

def estim_gamma_poisson(x, mod=0, a_init=0, b_init=0):
    """Estimate parameters a and b of the Gamma-Poisson(a,b) distribution."""
    m = np.mean(x)
    v = np.var(x)
    if m == 0: return 0, 1
    if v == 0: return m, 1
    r = v - m
    if r > 0: 
        a = (m**2) / r
        b = a / m
    else:
        if a_init and b_init:
            if a_init/b_init < v:
                a = ((a_init/b_init)**2) / (v - a_init/b_init)
                b = a / m
            else:
                a: int = a_init
                b: int = b_init
        else:
            a = (m**2) / v
            b = a / m
    if mod == -1 and a_init:
        if a_init/b_init < a/b:
            a, b = a_init, b_init
    elif mod == 1 and a_init:
        if a_init/b_init > a/b:
            a, b = a_init, b_init
    return a, b


def infer_kinetics_temporal(x, times, a_init=np.ones(100), b_init=1, max_iter=100, seuil=0.001, tol=1e-6, verb=False,
                            s=None) -> tuple[Any, Any]:
    """
    Original version used for initialization with discrete timepoints.
    s: per-cell depth factors (counts NB(a s_i, b) with depth_model 'burst_frequency', NB(a, b / s_i) with 'mean');
    None = 1.
    """
    t = np.sort(list(set(times)))
    m: int = t.size
    n = np.zeros(m)
    a = np.zeros(m)
    b = np.zeros(m)
    
    # Distinct counts and their multiplicities per group: sums over cells become sums over values
    vals, mult = [], []
    for i in range(m):
        cells = (times == t[i])
        n[i] = np.sum(cells)
        a[i], b[i] = estim_gamma_poisson(x[cells] if s is None else x[cells] / s[cells], mod=(i==m)-(i==0),
                                         a_init=a_init[i], b_init=b_init)
        v, w = np.unique(x[cells], return_counts=True)
        vals.append(v); mult.append(w)
    b = np.mean(b)
    # Groups as rows of a padded (m, n_values) table; zero multiplicity on the padding
    V = np.zeros((m, max(len(v) for v in vals)))
    Wm = np.zeros_like(V)
    for i in range(m):
        V[i, :len(vals[i])], Wm[i, :len(vals[i])] = vals[i], mult[i]
    
    # Moment equation for b on the raw counts: Σ_i (a_g(i) / b) s_i = Σ_i x_i, i.e. b = Σ_g a_g S_g / Σ x
    # with S_g the sum of the depth factors of group g (S_g = n_g without depth factors): the counts
    # expected at the cells' depths keep the observed total
    sx: float = max(np.sum(x), EPS)
    gidx = np.searchsorted(t, times) if s is not None else None
    w_b = n if s is None else np.bincount(gidx, weights=s, minlength=m)
    bf = s is not None and burst_frequency()

    k, c = 0, 0
    while (k == 0) or (k < max_iter and c > tol):
        # Newton step on every a_i > 0 at once (trigamma = zeta(2, .)):
        # gradient Σ_i [ψ(x_i+a) + log(b/s_i) - log(1+b/s_i) - ψ(a)]
        pos = a > 0
        if not pos.any():  # no burst in any group (gene ~ unexpressed): nothing to fit, b as after an update
            b = np.sum(w_b * a) / sx
            break
        a_s = np.where(pos, a, 1.0)
        if bf:
            # Shape a s_i per cell: gradient Σ_i s_i [ψ(x_i + a s_i) - ψ(a s_i) + log(b / (1 + b))]
            r = a_s[gidx] * s
            d = (np.bincount(gidx, weights=s * (psi(x + r) - psi(r)), minlength=m)
                 + w_b * (np.log(b) - np.log(1 + b)))
            h = np.bincount(gidx, weights=s ** 2 * (zeta(2, x + r) - zeta(2, r)), minlength=m)
        else:
            z = a_s[:, None] + V
            p0 = np.sum(Wm * psi(z), axis=1)
            p1 = np.sum(Wm * zeta(2, z), axis=1)
            if s is None:
                lb = n * (np.log(b) - np.log(1 + b))
            else:  # Σ_i log(b / (b + s_i)) per group
                lb = np.bincount(gidx, weights=np.log(b) - np.log(b + s), minlength=m)
            d = p0 + lb - n * psi(a_s)
            h = p1 - n * zeta(2, a_s)
        ok = pos & (h != 0)
        da = np.where(ok, -d / np.where(ok, h, 1.0), 0.0)
        
        a += np.maximum(da, -a)
        b = np.sum(w_b * a) / sx
        c = np.max(np.abs(da))
        k += 1
        if (k > 100) and (b > 1/seuil or b < seuil): break
    
    if (k == max_iter) or (c > tol):
        if verb:
            logger.warning("Bad convergence (b = %s), progress %.3f, delta %.3e", b, k/max_iter, c)
        if b > 1/seuil: a, b = a/(b*seuil), 1/seuil
    
    if np.sum(a < 0) > 0 and verb:
        logger.warning("a < 0 detected during kinetics inference")
    if b <= 0 and verb:
        logger.warning("b <= 0 detected during kinetics inference")
    if np.all(a == 0) and verb:
        logger.warning("All a coefficients are zero")
    
    b_old = b
    b = np.clip(b, seuil, 1/seuil)
    a *= b/(b_old + EPS)
    # Keep a > 0 even when all a[i] == 0 (nbinom.logpmf is NaN for a zero rate).
    floor = b/2 if np.max(a) <= 0 else np.minimum(b/2, np.max(a)/100)
    a = np.maximum(a, floor)

    return a, b


def nb_separation(kmin, kmax, c):
    """Separation of two NB modes of shapes kmin, kmax sharing the rate c (mean k / c, variance
    k (1 + c) / c^2): gap between the means over their pooled standard deviation."""
    return (kmax - kmin) * np.sqrt(2 / max((kmin + kmax) * (c + 1), EPS))


# Groups smaller than this are left out of the initialization (their extreme modes may be chance)
MIN_GROUP_CELLS, MIN_GROUP_FRAC = 50, 0.05


def large_groups(groups):
    """Cells of the groups holding at least MIN_GROUP_CELLS cells and MIN_GROUP_FRAC of the cells."""
    labels, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    ok = counts >= max(MIN_GROUP_CELLS, MIN_GROUP_FRAC * len(groups))
    return ok[inv]


def group_init(x, groups_list, seuil=0.001, s=None):
    """
    Initialization of the mixture from labelled groups of cells (times, cell types): for each
    labelling, one NB shape per group with a common rate (infer_kinetics_temporal), small groups
    left out (large_groups); the labelling whose extreme modes are the most separated
    (nb_separation) is kept.
    Returns (kmin, kmax, c, separation, index of the labelling kept).
    """
    best = None
    for i, groups in enumerate(groups_list):
        if groups is None:
            continue
        keep = large_groups(groups)
        if len(np.unique(groups[keep])) <= 1:
            continue
        a, b = infer_kinetics_temporal(x[keep], groups[keep], seuil=seuil, max_iter=1e5,
                                       s=None if s is None else s[keep])
        cand = (np.min(a), np.max(a), b, nb_separation(np.min(a), np.max(a), b), i)
        if best is None or cand[3] > best[3]:
            best = cand
    return best


def infer_kinetics_preserve_mean_values_assignment(x, resp, seuil=0.01, a_init=None, b_init=None,
                                   tol=1e-6, max_iter=100,  
                                   damping=0.7, verb=False, s=None, floor=None) -> tuple[Any, Any]:
    """
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
    floor : float or None
        Lower bound of ``a``; None: min(b / 2, max(a) / 100) (the CEM basins), a small value for the observation model.

    Returns:
    --------
    a : array (K,)
        Optimal shape parameters (ks in our notation)
    b : float
        Common dispersion parameter (c in our notation)
    """
    x = np.asarray(x).reshape(-1)
    resp = np.asarray(resp)
    N, K = resp.shape
    
    # weighted counts per component
    n = resp.sum(axis=0) + EPS  # (K,)
    
    # Initialisation
    if a_init is None:
        # initialize by weighted mean
        a = np.zeros(K)
        for k in range(K):
            weighted_mean = np.sum(resp[:, k] * (x if s is None else x / s)) / n[k]
            a[k] = max(seuil, weighted_mean)
    else:
        a = np.array(a_init).copy()
    
    if b_init is None:
        b = 1.0
    else:
        b = float(b_init)
    
    # Moment equation for b on the raw counts (as the hard M-step): b = Σ_k a_k S_k / Σ_i r_ik x_i,
    # S_k = Σ_i r_ik s_i (= n_k without depth factors)
    sx: np.bool_ | float = max(np.sum(resp * x[:, None]), EPS)
    w_b = n if s is None else resp.T @ np.asarray(s, dtype=float)
    bf = s is not None and burst_frequency()
    
    # Newton-Raphson with damping
    iteration, conv_metric = 0, 0
    
    while (iteration == 0) or (iteration < max_iter and conv_metric > tol):
        da = np.zeros(K)
        
        for k in range(K):
            if a[k] > seuil * 0.1 and bf:  # shape a s_i per cell (depth_model 'burst_frequency')
                r = a[k] * s
                gradient = np.sum(resp[:, k] * s * (psi(x + r) - psi(r) + np.log(b) - np.log(1 + b)))
                hessian = np.sum(resp[:, k] * s ** 2 * (polygamma(1, x + r) - polygamma(1, r)))
            elif a[k] > seuil * 0.1:  # avoid divisions by zero
                # z = a[k] + x for all observations
                z = a[k] + x  # (N,)
                
                # weighted gradient and Hessian by resp[:, k]
                # weighted d/da log p(x|a,b)
                p0 = np.sum(resp[:, k] * psi(z))
                p1: np.bool_ = np.sum(resp[:, k] * polygamma(1, z))
                
                if s is None:
                    gradient = n[k] * (np.log(b) - np.log(b + 1) - psi(a[k])) + p0
                else:  # Σ_i r_ik log(b / (b + s_i))
                    gradient = np.sum(resp[:, k] * (np.log(b) - np.log(b + s))) - n[k] * psi(a[k]) + p0
                hessian = p1 - n[k] * polygamma(1, a[k])
            if a[k] > seuil * 0.1:
                # Newton step avec damping
                if abs(hessian) != 0:
                    da[k] = -damping * gradient / hessian
                else:
                    da[k] = 0.0
        
        a += np.maximum(da, -a)
        b = np.sum(w_b * a) / sx
        
        # convergence metric
        conv_metric = np.max(np.abs(da))
        iteration += 1
        
        # Safety check: break if b diverges
        if (iteration > 100) and (b > 1/seuil or b < seuil): break
    
    # final checks
    if (iteration == max_iter) or (conv_metric > tol):
        if verb:
            logger.warning("Soft kinetics: convergence warning (iter=%d, conv=%.2e, b=%.4f)", iteration, conv_metric, b)
    
    if np.any(a < 0) and verb:
        logger.warning('WARNING: negative a detected, clipping')
    if b <= 0 and verb:
        logger.warning('WARNING: non-positive b')
    
    # Clipping final conservateur
    b_old: Any | float = b
    b = np.clip(b, seuil, 1/seuil)
    a *= b/(b_old + EPS)
    a = np.maximum(a, np.minimum(b/2, np.max(a)/100) if floor is None else floor)
    
    return a, b


def nb_logpmf_vectorized(x, ks, c, s=None):
    """
    Vectorized log-PMF of a Negative Binomial.

    Model: X_i | z=k ∼ NB(ks_k, c); depth factors s: NB(ks_k s_i, c) (depth_model 'burst_frequency') or
    NB(ks_k, c / s_i) ('mean').

    Parameters
    ----------
    x  : (N,)   observations (integer counts)
    ks : (K,)   shape parameters
    c  : float  dispersion parameter for the gene (shared across components)

    Returns
    -------
    logpmf : (N, K)
    """
    x  = np.asarray(x).reshape(-1)
    ks = np.clip(np.asarray(ks).reshape(-1), 1e-8, 1e5)
    c  = float(np.clip(c, 1e-8, 1e5))

    X  = x[:, None]                         # (N, 1)
    K  = ks[None, :]                        # (1, K)

    ln_c: np.ndarray[Any, np.dtype[Any]]   = np.log(c + EPS)               # scalar
    ln_1pc: np.ndarray[Any, np.dtype[Any]] = np.log(1.0 + c + EPS)         # scalar
    if s is not None and burst_frequency():  # NB(k s_i, c): shape k s_i per cell
        K = K * np.asarray(s, dtype=float).reshape(-1, 1)
    elif s is not None:  # NB(k, c / s_i): rate c / s_i per cell
        s = np.asarray(s, dtype=float).reshape(-1, 1)
        ln_c, ln_1pc = np.log(c / s + EPS), np.log(1.0 + c / s + EPS)

    return (gammaln(X + K) - gammaln(K) - gammaln(X + 1.0)
            + K * ln_c - (X + K) * ln_1pc)


def predict_resp(x, ks, c, pi=None, forcing=1.0, s=None) -> tuple[Any, Any]:
        """
        Compute the responsibilities.
        """
        n_components: int = len(ks)
        if pi is not None and np.size(pi) != n_components:
            # A size mismatch would silently broadcast resp to the wrong width.
            logger.warning(
                "predict_resp: pi has %d entries but ks has %d components; "
                "falling back to a uniform pi", np.size(pi), n_components
            )
            pi = None
        if pi is None:
            pi = np.ones(n_components) / n_components
        else:
            if forcing < 1:
                pi = pi * forcing + (np.ones(n_components) / n_components) * (1 - forcing)
            pi = pi / (pi.sum() + EPS)

        logpmf = nb_logpmf_vectorized(x, ks, c, s)

        log_joint    = logpmf + np.log(pi + EPS)[None, :]
        log_evidence = logsumexp(log_joint, axis=1, keepdims=True)
        resp         = np.exp(log_joint - log_evidence)
        resp         = np.clip(resp, EPS, 1.0)
        resp        /= resp.sum(axis=1, keepdims=True)
        return resp, log_joint


def hard_em(data, n_components, ks_init, c_init, seuil, tol=1e-6, max_iter_loop=200,
            vect_t=None, preserve_mean_values=0, mean_forcing=1.0, s=None):
    """
    Hard EM for a Negative Binomial mixture with temporal constraints.
    
    Parameters
    ----------
    mean_forcing : float
        Weight of the temporal constraint (0 = no constraint, 1 = strict constraint)
    """
    
    n_cells = data.size 
    basins: np.ndarray[Any, np.dtype[Any]] = np.zeros(n_cells, dtype=int)
    ks, c = ks_init.copy(), c_init
    
    # Initialisation
    resp, log_proba = predict_resp(data, ks, c, s=s)
    basins, pi = _assign_basins(resp, data, ks, c, vect_t, preserve_mean_values, n_components, mean_forcing,
                                s=s)

    if len(np.unique(basins)) < n_components:
        basins, pi = _assign_basins(resp, data, ks, c, vect_t, 0, n_components, mean_forcing, s=s)
    
    log_likelihood_old = np.sum([log_proba[cell, basins[cell]] for cell in range(n_cells)])
    
    for it in range(max_iter_loop):
        # M-step: update parameters
        ks_new, c_new = infer_kinetics_temporal(data, basins, a_init=ks, b_init=c,
                                                 seuil=seuil, max_iter=1e5, tol=tol, s=s)

        if np.size(ks_new) != n_components:
            # All cells collapsed into fewer basins, so ks shrank: keep the last consistent fit.
            return ks, c, pi, basins

        # E-step: reassign
        resp_new, log_proba = predict_resp(
            data, ks_new, c_new, pi=pi, forcing=mean_forcing, s=s
            )
        basins_new, pi_new = _assign_basins(resp_new, data, ks_new, c_new, 
                                            vect_t, preserve_mean_values, 
                                            n_components, mean_forcing, s=s)
        
        if len(np.unique(basins_new)) < n_components:
            return ks, c, pi, basins
        
        # convergence check
        if it:
            log_likelihood_new = np.sum([log_proba[cell, basins_new[cell]] for cell in range(n_cells)])
            if (log_likelihood_new - log_likelihood_old) < 1/max_iter_loop:
                break
            log_likelihood_old = log_likelihood_new
        
        ks, c, pi, basins = ks_new.copy(), c_new, pi_new.copy(), basins_new.copy()

    return ks, c, pi, basins


def _ot_assign(mu, nu, cost):
    """Hard assignment of cells to basins with target masses nu by exact OT (EMD)."""
    return np.argmax(ot.emd(mu, nu, cost, numItermax=int(1e7)), axis=1)


def _assign_basins(resp, data, ks, c, vect_t, preserve_mean_values, n_components, mean_forcing, final=False,
                   s=None):
    """Assign cells to basins with optional temporal constraints."""
    
    n_cells = data.size

    if not preserve_mean_values:
        return np.argmax(resp, axis=1), resp.sum(axis=0) / (resp.sum() + EPS)
    
    if vect_t is None:
        mu = np.ones(n_cells) / n_cells
        nu = _compute_nu_with_temporal_constraint(
            resp, data, ks, c, n_components, mean_forcing, s=s
        )
        return _ot_assign(mu, nu, -np.log(resp)), nu
    
    # Soft clustering with temporal constraints
    basins: np.ndarray[Any, np.dtype[Any]] = np.zeros(n_cells, dtype=int)
    pi_final = {}
    pi = np.zeros(n_components)
    for t_i in np.unique(vect_t):
        indices = (vect_t == t_i)
        resp_i = resp[indices]
        n_cells_i = np.sum(indices)
        
        # Uniform cell distribution
        mu = np.ones(n_cells_i) / n_cells_i
        
        # Compute nu with temporal constraint
        nu = _compute_nu_with_temporal_constraint(
            resp_i, data[indices], ks, c, n_components, mean_forcing,
            s=None if s is None else s[indices]
        )
        
        # Transport optimal
        basins[indices] = _ot_assign(mu, nu, -np.log(resp_i))
        pi_final[t_i] = nu
        pi += nu * n_cells_i / n_cells
    
    if final: return basins, pi_final
    return basins, pi


def _compute_nu_with_temporal_constraint(resp_i, data_t, ks, c, n_components, mean_forcing, s=None):
    """
    Compute the target distribution nu by balancing likelihood and a temporal
    mean constraint.
    
    The temporal constraint enforces: sum_k (nu_k * ks_k / c) ≈ mean_t
    """
    # proportions based on the likelihood
    nu_likelihood = np.sum(resp_i, axis=0)
    nu_likelihood /= np.sum(nu_likelihood)
    
    # proportions based on the mean constraint
    means_components = ks / c  # Moyenne de chaque NB
    nu = _solve_mean_constraint(means_components, data_t, ks, c, nu_likelihood, n_components, mean_forcing,
                                s=s)
    nu = np.clip(nu, EPS, 1)  # avoid zero values
    nu /= np.sum(nu)
    
    return nu


def neg_log_likelihood_logits(logits, data, r, p):
    # r[k], p: scalars, or per cell (depth factors: shapes r_k s_i, or p_i = c / (c + s_i))
    """
    logits : array (K,)
    data   : array (n,)
    r, p   : NB parameters (K,)
    """
    # softmax -> nu
    logits = logits - np.max(logits)  # numerical stability
    exp_logits = np.exp(logits)
    nu = exp_logits / np.sum(exp_logits)

    # log-probabilities of the mixture
    log_probs = []
    for k in range(len(nu)):
        log_probs.append(
            np.log(nu[k]) + nbinom.logpmf(data, r[k], p)
        )

    log_probs = np.vstack(log_probs)           # (K, n)
    log_mix = np.logaddexp.reduce(log_probs, axis=0)

    return -np.sum(log_mix)


def compute_nu_star(
    data,
    r_components,
    p_components,
    nu_init=None,
    method="L-BFGS-B"
):
    K: int = len(r_components)

    # initialisation
    if nu_init is not None and np.size(nu_init) != K:
        logger.warning(
            "compute_nu_star: nu_init has %d entries but %d components were "
            "given; ignoring nu_init", np.size(nu_init), K
        )
        nu_init = None
    if nu_init is None:
        logits0 = np.zeros(K)
    else:
        nu_init = np.clip(nu_init, 1e-12, None)
        nu_init = nu_init / nu_init.sum()
        logits0 = np.log(nu_init)

    res = minimize(
        neg_log_likelihood_logits,
        logits0,
        args=(data, r_components, p_components),
        method=method
    )

    # retrieval of nu*
    logits_opt = res.x
    logits_opt -= np.max(logits_opt)
    nu_star = np.clip(np.exp(logits_opt), 1e-12, 1)
    nu_star /= nu_star.sum()

    return nu_star


def ks_statistic(data_full, nu, r, p, repet=10, n_cells_init=200) -> Any | float:

    n_cells: int = min(data_full.size, n_cells_init)
    stat = 0
    for _ in range(repet):
        # sorted data
        data = data_full[np.random.choice(data_full.size, n_cells, replace=False)]
        data_sorted = np.sort(data)
        n: int = len(data_sorted)

        # CDF empirique
        emp_cdf = np.arange(1, n + 1) / n

        # theoretical mixture CDF
        model_cdf = np.zeros(n)
        for k in range(len(nu)):
            model_cdf += nu[k] * nbinom.cdf(data_sorted, r[k], p)

        # statistique KS
        stat += np.max(np.abs(emp_cdf - model_cdf))
    
    return stat / repet


def _solve_mean_constraint(means_components, data_t, ks, c, nu_init, n_components, mean_forcing, s=None):
    """
    while minimizing the distance to the uniform distribution_k) = target_mean
    while minimizing the distance to the uniform distribution.
    
    Solves a constrained quadratic optimization problem.
    """
    # Depth factors: mean at the reference depth; NB per cell, shapes ks s_i ('burst_frequency') or p_i = c / (c + s_i)
    mean_t = np.mean(data_t) if s is None else np.mean(data_t) / np.mean(s)  # ratio of means, as the M-step
    r_cells = ks
    if s is not None and burst_frequency():
        r_cells, p_cells = np.asarray(ks, dtype=float)[:, None] * np.asarray(s, dtype=float)[None, :], c / (1 + c)
    else:
        p_cells = c / (1 + c) if s is None else c / (c + np.asarray(s, dtype=float))

    # Size everything from the actual components, not a possibly stale n_components.
    n_components = len(means_components)
    if np.size(nu_init) != n_components:
        nu_init = np.ones(n_components) / n_components

    # Adjust the level of regularization
    nu_star = compute_nu_star(data=data_t,
                            r_components=r_cells,
                            p_components=p_cells,
                            nu_init=nu_init
                            )
    # KS statistic at the reference depth (counts rescaled by s and rounded)
    stat_KS = ks_statistic(data_t if s is None else np.rint(data_t / s), nu_star, ks, c / (1 + c))
    if mean_forcing > 0: alpha_reg = np.clip(stat_KS / mean_forcing, 0.0, 1.0)
    else: alpha_reg = 1.0

    # Likelihood term: maximum-likelihood masses (sums of posteriors are pulled toward uniform
    # when components overlap, e.g. at zero counts of sparse genes)
    nu_init = nu_star

    # Objective function: distance to the uniform distribution
    def objective(nu):
        mean_pred = np.dot(nu, means_components)
        mean_erreur = (mean_pred - mean_t) / (EPS + mean_t)
        err = mean_erreur**2
        return (1 - alpha_reg) * err + alpha_reg * np.sum((nu - nu_init)**2)
    
    # Constraint: sum = 1
    def constraint_sum(nu):
        return np.sum(nu) - 1
    
    constraints = [
        {'type': 'eq', 'fun': constraint_sum}
    ]
    
    bounds: list[tuple[int, int]] = [(0, 1) for _ in range(n_components)]
    
    result = minimize(objective, nu_init, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x
    

def _masses_per_time(resp, vect_t):
    """Masses (mean posteriors) of each time {time: (K,)}."""
    return {t: resp[vect_t == t].mean(axis=0) for t in np.unique(vect_t)}


def _log_masses_rows(masses, vect_t, n_cells, K):
    """(N, K) log masses of each cell: per time (dict, vect_t) or global (K,)."""
    if isinstance(masses, dict):
        return np.stack([np.log(np.asarray(masses[t], dtype=float) + EPS) for t in vect_t])
    return np.tile(np.log(np.asarray(masses, dtype=float) + EPS), (n_cells, 1))


def em_vectorized_nb(x, ks_init, c_init, pi_init=None, max_iter=200, tol=1e-6, seuil=0.01,
                     damping=0.7, verbose=False, s=None, vect_t=None, floor=None) -> Any:
    """
    EM for NB mixture with ANALYTICAL M-step via Newton-Raphson.
    
    Replaces moment-based estimates with direct optimization of the
    weighted log-likelihood, as in :func:`infer_kinetics_temporal`. vect_t given: masses per time (the modes are
    shared by the times, their weights are not), pi returned as their cell-weighted mean.
    """
    x = np.asarray(x).reshape(-1)
    N: int = x.size
    ks: np.ndarray[Any, np.dtype[Any]] = np.asarray(ks_init).astype(float).reshape(-1)
    K: int = ks.size
    c = float(c_init)
    
    if pi_init is None:
        pi = np.ones(K) / K
    else:
        pi: np.ndarray[Any, np.dtype[Any]] = np.asarray(pi_init).astype(float).reshape(-1)
        pi = pi / (pi.sum() + EPS)

    loglik_old: float = -np.inf
    resp = np.ones((N, K)) / K
    masses = pi if vect_t is None else {t: pi for t in np.unique(vect_t)}

    for it in range(max_iter):
        # ==================
        # E-STEP (unchanged)
        # ==================
        logpmf = nb_logpmf_vectorized(x, ks, c, s)

        log_joint = logpmf + _log_masses_rows(masses, vect_t, N, K)
        
        log_evidence = logsumexp(log_joint, axis=1, keepdims=True)
        resp = np.exp(log_joint - log_evidence)
        resp = np.clip(resp, 1e-12, 1.0)
        resp /= resp.sum(axis=1, keepdims=True)
        
        loglik = np.sum(log_evidence)

        if verbose and it % 20 == 0:
            logger.info("EM iter %d: loglik=%.4f, ks=%s, c=%.4f", it, loglik, ks, c)

        # Convergence check
        if np.isfinite(loglik_old) and abs(loglik - loglik_old) < tol:
            if verbose:
                logger.info("Converged at iteration %d", it)
            break
        loglik_old = loglik

        # ==================
        # M-STEP ANALYTIQUE
        # ==================
        
        # 1) Update the masses (per time with vect_t)
        Nk = resp.sum(axis=0) + EPS
        pi = Nk / N
        masses = pi if vect_t is None else _masses_per_time(resp, vect_t)

        # 2) Update ks and c via analytical optimization
        ks, c = infer_kinetics_preserve_mean_values_assignment(
            x, resp, 
            seuil=seuil,
            a_init=ks, 
            b_init=c,
            tol=tol,
            max_iter=1e5,  # Newton-Raphson sub-iterations
            damping=damping,
            verb=verbose,
            s=s,
            floor=floor
        )

    # Final likelihood
    final_logpmf = nb_logpmf_vectorized(x, ks, c, s)

    final_joint = final_logpmf + _log_masses_rows(masses, vect_t, N, K)
    final_logev = logsumexp(final_joint, axis=1)
    final_loglik = np.sum(final_logev)

    return ks, c, pi, resp, final_loglik


# Continuous model of the counts (max_grid_levels): sigma on n_grid levels of [0, 1], NB(k_low + (k_high - k_low)
# sigma_l, c), weights per time with a Dirichlet prior GRID_ALPHA on every level + GRID_ALPHA_EXT on the two extremes
# (anchors k_low and k_high on occupied levels: without it, the extremes drift outwards with little mass); rate c at
# most GRID_C_MAX (as the cap of the CEM rate in core_binarization)
GRID_LEVELS, GRID_ALPHA, GRID_ALPHA_EXT, GRID_C_MAX = 11, 0.01, 1.0, 9.0


def grid_weights_update(counts, alpha=GRID_ALPHA, alpha_ext=GRID_ALPHA_EXT):
    """MAP weights of the levels from their expected counts (..., L) under the Dirichlet prior of the grid."""
    prior = np.full(counts.shape[-1], alpha)
    prior[[0, -1]] += alpha_ext
    w = counts + prior
    return w / w.sum(axis=-1, keepdims=True)


def grid_levels(k_low, k_high, n_grid=GRID_LEVELS):
    """Burst frequencies (n_grid,) of the levels of the continuous model."""
    return k_low + (k_high - k_low) * np.linspace(0.0, 1.0, n_grid)


def grid_weights_fixed(x, vect_t, levels, c, s=None, n_em=500, tol=1e-7):
    """Posteriors (N, L) and MAP weights per time {t: (L,)} of cells under fixed levels (continuous model)."""
    lp = nb_logpmf_vectorized(x, levels, c, s)
    resp, W = np.zeros_like(lp), {}
    for t in np.unique(vect_t):
        rows = np.asarray(vect_t) == t
        L = np.exp(lp[rows] - lp[rows].max(axis=1, keepdims=True))
        w = np.full(lp.shape[1], 1.0 / lp.shape[1])
        for _ in range(n_em):
            post = L * w
            post /= post.sum(axis=1, keepdims=True)
            w_new = grid_weights_update(post.sum(axis=0))
            done = np.max(np.abs(w_new - w)) < tol
            w = w_new
            if done:
                break
        post = L * w
        resp[rows] = post / post.sum(axis=1, keepdims=True)
        W[t] = w
    return resp, W


def mixture_criterion(x, vect_t, levels, c, weights, s=None, criterion='icl', weights_per_time=False):
    """
    Selection criterion of a NB mixture of burst frequencies levels (K,), rate c, weights per time {t: (K,)} (or
    common): BIC = -2 loglik + p log n, p = min(K, 2) + 1 + (K - 1) (two shapes or k_low, k_high, and c; the weights,
    counted once since the weights of the times are pulled together by their Dirichlet prior, or once per time with
    weights_per_time); ICL (Biernacki, Celeux and Govaert 2000) = BIC + 2 entropy of the posteriors, which
    penalises overlapping levels, so that a gene that switches keeps two modes and levels are added only when the
    counts separate them. Returns (criterion, posteriors).
    """
    x = np.asarray(x).astype(int)
    vt = np.asarray(vect_t)
    K = len(levels)
    of = (lambda t: weights[t]) if isinstance(weights, dict) else (lambda t: weights)  # per time, or common
    W = np.stack([np.pad(np.asarray(of(t), dtype=float), (0, max(0, K - len(of(t)))))[:K] for t in vt])
    lj = nb_logpmf_vectorized(x, levels, c, s) + np.log(np.maximum(W, 1e-300))
    le = logsumexp(lj, axis=1, keepdims=True)
    post = np.exp(lj - le)
    ent = -np.sum(post * np.log(np.maximum(post, 1e-300)))
    p = min(K, 2) + 1 + (K - 1) * (len(np.unique(vt)) if weights_per_time else 1)
    bic = float(-2 * np.sum(le) + p * np.log(len(x)))
    return (bic + 2 * ent if criterion == 'icl' else bic), post


def fit_grid_mixture(x, vect_t, k_low, k_high, c, s=None, seuil=0.01, n_grid=GRID_LEVELS, n_em=300, tol=1e-8,
                     depth_model=None):
    """
    Continuous model of the counts of one gene (max_grid_levels levels at most), started from the extreme modes of the CEM (+ soft
    EM): x ~ NB(k_low + (k_high - k_low) sigma_l, c) at the depth s of the cell (depth_model), sigma_l on n_grid levels
    with weights per time (grid_weights_update), by EM on the weights and on (k_low, k_high, c) (M-step by L-BFGS on
    their logarithms, k_low floored at min(c / 2, k_high / 100) as in the CEM), on the distinct (time, count, depth)
    triples. Two modes read the cells in a transient as part of
    the extremes, which pulls them inwards; here they sit on intermediate levels. Returns (levels (n_grid,), c, weights
    {t: (n_grid,)}, posteriors (N, n_grid)), or None for a constant gene. depth_model: set in the process that runs the
    fit (parallel workers), None: current setting.
    """
    if depth_model is not None:
        set_depth_model(depth_model)
    x = np.asarray(x).astype(int)
    if x.size == 0 or x.min() == x.max() or not k_high > k_low:
        return None
    sig = np.linspace(0.0, 1.0, n_grid)
    times, t_cell = np.unique(np.asarray(vect_t), return_inverse=True)
    sd = np.ones(len(x)) if s is None else np.round(np.asarray(s, dtype=float), 3)
    uk, mult = np.unique(np.stack([t_cell, x.astype(float), sd], axis=1), axis=0, return_counts=True)
    ti, X, su = uk[:, 0].astype(int), uk[:, 1][:, None], uk[:, 2][:, None]
    bf = s is not None and burst_frequency()

    def terms(th):
        kl, dk, cc = np.exp(th)
        K = (kl + dk * sig)[None, :]
        r = np.maximum(K * su if bf else np.broadcast_to(K, (len(X), n_grid)), 1e-12)
        rate = cc if bf else cc / su
        lp = gammaln(X + r) - gammaln(r) - gammaln(X + 1) + r * np.log(rate) - (X + r) * np.log1p(rate)
        return lp, r, rate, kl, dk, cc

    def neg_ll(th, R):
        lp, r, rate, kl, dk, cc = terms(th)
        dK = -R * (psi(X + r) - psi(r) + np.log(rate) - np.log1p(rate)) * (su if bf else 1.0)
        drate = -R * (r / rate - (X + r) / (1 + rate)) * (1.0 if bf else 1.0 / su)
        return -np.sum(R * lp), np.array([np.sum(dK) * kl, np.sum(dK * sig) * dk, np.sum(drate) * cc])

    c_hi = min(1.0 / seuil, GRID_C_MAX)
    th = np.log([max(k_low, 1e-8), max(k_high - k_low, 1e-8), np.clip(c, seuil, c_hi)])
    bounds = [(np.log(1e-8), np.log(1e5))] * 2 + [(np.log(seuil), np.log(c_hi))]
    W = np.full((len(times), n_grid), 1.0 / n_grid)
    prev = -np.inf
    for _ in range(n_em):
        lj = terms(th)[0] + np.log(W[ti])
        le = logsumexp(lj, axis=1, keepdims=True)
        R = np.exp(lj - le) * mult[:, None]
        ll = float(np.sum(le[:, 0] * mult))
        W = grid_weights_update(np.stack([R[ti == j].sum(axis=0) for j in range(len(times))]))
        th = minimize(neg_ll, th, args=(R,), jac=True, method='L-BFGS-B', bounds=bounds).x
        # Floor of the low level, as in the CEM: k_low >= min(c / 2, k_high / 100), k_high kept
        kl, dk, cc = np.exp(th)
        floor = min(cc / 2, (kl + dk) / 100)
        if kl < floor:
            th = np.log([floor, max(kl + dk - floor, 1e-8), cc])
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    kl, dk, cc = np.exp(th)
    if not (np.all(np.isfinite([kl, dk, cc])) and dk > 0):
        return None
    levels = kl + dk * sig
    weights = {t: W[j] for j, t in enumerate(times)}
    lj = nb_logpmf_vectorized(x, levels, cc, s) + np.stack([np.log(weights[t]) for t in np.asarray(vect_t)])
    resp = np.exp(lj - logsumexp(lj, axis=1, keepdims=True))
    return levels, float(cc), weights, resp


# ---------------------------
# Fonction pour calculer l'AIC
# ---------------------------

def compute_aic_for_params(x, ks, c, pi, s=None, vect_t=None) -> tuple[Any, floating[Any]]:
    """Compute the AIC for a given set of parameters (vect_t given: masses per time, the mean posteriors of each time
    under pi)."""
    logpmf = nb_logpmf_vectorized(x, ks, c, s)
    num_params: int = len(ks) + 1 + (len(ks) - 1)

    if vect_t is not None:
        lj = logpmf + np.log(np.asarray(pi, dtype=float) + EPS)[None, :]
        resp = np.exp(lj - logsumexp(lj, axis=1, keepdims=True))
        log_joint = logpmf + _log_masses_rows(_masses_per_time(resp, vect_t), vect_t, len(x), len(ks))
        log_evidence = logsumexp(log_joint, axis=1)
        return np.log(x.size) * num_params - 2 * np.sum(log_evidence), np.sum(log_evidence)
    log_joint = logpmf + np.log(pi + EPS)[None, :]
    log_evidence = logsumexp(log_joint, axis=1)
    loglik = np.sum(log_evidence)
    
    aic = np.log(x.size) * num_params - 2 * loglik
    return aic, loglik


# ---------------------------
# Main class (refactored)
# ---------------------------

class NegativeBinomialMixtureEM:
    def __init__(self, min_components=1, max_components=3, hard_em=1, mean_forcing_em=1.0,
                 tol=1e-5, max_iter_em=200, verbose=False, preserve_mean_values=0,
                 compare_init_aic=True, damping=1.0,
                 depth_model=None) -> None:
        """
        NB mixture with optimal analytical M-step.

        New parameters:
        ---------------
        compare_init_aic : bool
            If True, compare the AIC of the initialization with the AIC after EM
        damping : float (0, 1]
            Damping factor for Newton-Raphson (0.5-0.8 = stable, 1.0 = fast)
        depth_model : 'burst_frequency', 'mean' or None (current setting): how depth factors enter the NB
            (depth_model.py), set in the process that runs fit (parallel workers included).
        """
        assert min_components >= 1 and max_components >= min_components
        self.min_components: int = min_components
        self.max_components: int = max_components
        self.preserve_mean_values: int = preserve_mean_values
        self.hard_em: int = hard_em
        self.mean_forcing_em: float = mean_forcing_em
        self.tol: float = tol
        self.max_iter_em: int = max_iter_em
        self.verbose: bool = verbose
        self.compare_init_aic: bool = compare_init_aic
        self.damping: float = damping
        self.depth_model = depth_model
        self.best_model = None


    def _init_for_K(self, x, K, vect_t=None, vect_celltypes=None, quant_init=None, seuil=0.01, s=None):
        # Moments at the reference depth (depth factors s)
        xr = x if s is None else x / s
        mean = np.mean(xr)
        var = np.var(xr)
        
        if mean <= 0:
            k_glob = 1.0
            c_glob = 1.0
        else:
            if var <= mean + 1e-8:
                k_glob = max(50.0, mean)
            else:
                k_glob = max(1e-3, (mean**2) / (var - mean))
            c_glob = max(1e-6, k_glob / (mean + 1e-16))

        self._init_groups = 'time'
        init = group_init(x, [vect_t, vect_celltypes], seuil, s) if vect_t is not None else None
        if init is not None:
            # Extreme modes of the most diverse labelling (times or cell types), common rate
            kmin, kmax, c_init, _, i = init
            ks_init = np.linspace(kmin, kmax, K)
            self._init_groups = ('time', 'celltype')[i]
        elif vect_t is not None:
            a_per_time, b_est = infer_kinetics_temporal(x, vect_t, seuil=seuil, max_iter=1e5, s=s)
            ks_init = np.linspace(np.min(a_per_time), np.max(a_per_time), K)
            c_init = b_est
        else:
            if vect_celltypes is None or (vect_celltypes is not None and len(np.unique(vect_celltypes)) <= 1):
                if quant_init is None:
                    quant_init = np.linspace(0, 1, K+1)
                elif len(quant_init) != K+1:
                    print("Quant init has not the right size")
                    quant_init = np.linspace(0, 1, K+1)
                centers = np.quantile(x, quant_init)
                ks_init = np.array([np.mean(x[(centers[k] <= x) & 
                                          (x <= centers[k+1])]) * c_glob 
                                          for k in range(0, K)]) 
                c_init = c_glob
            else:
                if quant_init is None:
                    a_per_type, b_est = infer_kinetics_temporal(x, vect_celltypes, seuil=seuil, max_iter=1e5)
                    ks_init = np.linspace(np.min(a_per_type), np.max(a_per_type), K)
                    c_init = b_est
                else:
                    if len(quant_init) != K+1:
                        print("Quant init has not the right size")
                        quant_init = np.linspace(0, 1, K+1)
                    centers = np.quantile(x, quant_init)
                    a_per_quant = np.array([np.mean(x[(centers[k] <= x) & 
                                            (x <= centers[k+1])]) 
                                            for k in range(0, K)]) 
                    a_per_type, b_type = infer_kinetics_temporal(x, vect_celltypes, seuil=seuil, max_iter=1e5)
                    ks_init = np.linspace(min(np.min(a_per_quant), np.min(a_per_type/b_type)), max(np.max(a_per_quant), np.max(a_per_type/b_type)), K)
                    c_init = b_type
                    ks_init *= c_glob

        
        n = 1/2
        m, M = np.min(ks_init), np.max(ks_init)
        while (len(np.unique((ks_init/c_init).astype(int))) < K) and (n > 1/x.size):
            mn = min(m, np.quantile(seuil + xr*c_init, n))
            Mn = max(M, np.quantile(seuil + xr*c_init, 1-n))
            ks_init = np.linspace(mn, Mn, K)
            n /= 2
        # A zero rate makes nbinom.logpmf/cdf return NaN.
        ks_init = np.maximum(ks_init, seuil * c_init)
        if self.verbose: print(f"  Init: ks_init={ks_init}")

        pi_init = np.ones(K) / K
        return ks_init.astype(float), float(c_init), pi_init.astype(float)




    def _count_params(self, K):
        return K + 1 + (K - 1)
    

    def _fit_constant_gene(self, x_all, vect_t_all, seuil):
        """Single-mode model for a gene constant across all cells; its single mode
        makes the network inference zero its ref_network row and column."""
        N: int = x_all.size
        value = float(x_all[0]) if N else 0.0
        c = 1.0
        ks: np.ndarray[Any, np.dtype[Any]] = np.array([max(value * c, seuil)], dtype=float)
        resp: np.ndarray[Any, np.dtype[Any]] = np.ones((N, 1))
        basins: np.ndarray[Any, np.dtype[Any]] = np.zeros(N, dtype=int)

        # Same pi structure as _assign_basins(final=True).
        if self.preserve_mean_values and vect_t_all is not None:
            pi: Any = {t: np.ones(1) for t in np.unique(vect_t_all)}
        else:
            pi = np.ones(1)

        aic, loglik = compute_aic_for_params(x_all, ks, c, np.ones(1))

        best_model = {
            'ks':              ks,
            'c':               c,
            'pi':              pi,
            'basins':          basins,
            'resp':            resp,
            'loglik':          float(loglik),
            'n_components':    1,
            'aic':             float(aic),
            'initial_K_tried': 1,
        }
        self.best_model = best_model
        return best_model


    def fit(self, x, vect_t=None, vect_celltypes=None, quant_init=None, seuil=0.001,
            batch_size_mixture=None, strata=None, s=None):
        """
        Fit the NB mixture model to data ``x``.

        Parameters
        ----------
        x      : (N,)  counts (integers)
        vect_t : (N,)  optional time label per cell
        batch_size_mixture : int or None
               If set, parameter learning is done on a random sub-sample of at
               most ``batch_size_mixture`` cells per timepoint; responsibilities
               (resp) and basin assignments are then recomputed on ALL cells
               with the learned parameters.
        strata : (N,) cell types or None
               If given, the sub-sample is cell-type proportional within each timepoint.
        s      : (N,) per-cell depth factors or None: counts modelled as NB(k s_i, c) or NB(k, c / s_i)
               (depth_model), the parameters ks, c being those of a cell at the reference depth (s = 1).
        """
        set_depth_model(self.depth_model)
        # ── Keep full-data references for the final resp computation ────────
        x_all: np.ndarray[Any, np.dtype[Any]] = np.asarray(x).astype(int)
        N_all: int = x_all.size
        vect_t_all = vect_t

        # ── Constant gene: nothing to fit, return a neutral 1-mode model ────
        if N_all == 0 or x_all.min() == x_all.max():
            return self._fit_constant_gene(x_all, vect_t_all, seuil)

        # ── Optional mini-batch sub-sampling for parameter learning ─────────
        if batch_size_mixture is not None:
            if vect_t_all is not None:
                idx_list = []
                for time in np.unique(vect_t_all):
                    idx_t = np.where(vect_t_all == time)[0]
                    idx_list.append(stratified_choice(
                        idx_t, batch_size_mixture, None if strata is None else strata[idx_t]))
                cells_to_use = np.concatenate(idx_list)
            else:
                cells_to_use = stratified_choice(np.arange(N_all), batch_size_mixture, strata)
        else:
            cells_to_use = np.arange(N_all)

        x: np.ndarray[Any, np.dtype[Any]] = x_all[cells_to_use]
        vect_t   = vect_t_all[cells_to_use]   if vect_t_all is not None else None
        vect_celltypes = np.asarray(vect_celltypes)[cells_to_use] if vect_celltypes is not None else None
        s_all = None if s is None else np.asarray(s, dtype=float)
        s = None if s_all is None else s_all[cells_to_use]
        x_init = x

        best_aic: float = np.inf
        best_model = None

        # Candidate numbers of modes
        K_list = list(range(self.min_components, self.max_components + 1))
        for K_try in K_list:
            if self.verbose:
                logger.info("=== Trying K_init = %s ===", K_try)
            ks_init, c_init, pi_init = self._init_for_K(
                x_init, K_try, vect_t=vect_t, vect_celltypes=vect_celltypes, quant_init=quant_init, seuil=seuil, s=s
            )

            if self.hard_em:
                ks_init, c_init, pi_init, basins = hard_em(
                    x, K_try, ks_init, c_init, seuil, tol=self.tol, vect_t=vect_t,
                    preserve_mean_values=self.preserve_mean_values,
                    mean_forcing=self.mean_forcing_em, s=s
                )

            # ── AIC de l'initialisation ───────────────────────────────────
            if self.compare_init_aic:
                # Same footing as the EM below: masses per time (mean posteriors under the initial modes)
                aic_init, loglik_init = compute_aic_for_params(
                    x, ks_init, c_init, pi_init, s, vect_t=vect_t
                )
                if self.verbose:
                    logger.info("Init AIC: %.3f, loglik: %.3f", aic_init, loglik_init)
            else:
                aic_init: float = np.inf
                loglik_init: float = -np.inf

            ks_final, c_final, pi_final, loglik_final = \
                ks_init.copy(), c_init, pi_init.copy(), loglik_init
            stable = False
            iter_count = 0

            while not stable and iter_count < 10:
                iter_count += 1

                ks_fit, c_fit, pi_fit, resp_fit, loglik_fit = \
                    em_vectorized_nb(
                        x, ks_final, c_final, pi_final, max_iter=self.max_iter_em,
                        tol=self.tol, seuil=seuil, damping=self.damping,
                        verbose=self.verbose, s=s, vect_t=vect_t
                    )

                if self.verbose:
                    logger.info("EM iter %d: K=%d, loglik=%.3f", iter_count, ks_fit.size, loglik_fit)

                ks_final, c_final, pi_final, resp_final, loglik_final = (
                    ks_fit, c_fit, pi_fit, resp_fit, loglik_fit
                )
                stable = True

            K_final   = ks_final.size
            num_params = self._count_params(K_final)
            aic_final  = np.log(x.size) * num_params - 2 * loglik_final

            # ── Comparer init vs EM ───────────────────────────────────────
            if self.compare_init_aic and self.max_iter_em:
                keep_init: np.bool_ | bool = loglik_init >= loglik_final
                if keep_init:
                    if self.verbose:
                        logger.info(">> Keeping init (loglik %.3f >= %.3f)", loglik_init, loglik_final)
                    ks_final, c_final, pi_final = ks_init, c_init, pi_init
                    aic_final: Any | float    = aic_init
                    loglik_final = loglik_init
                    K_final      = ks_init.size
                    # Recalculer resp
                    logpmf = nb_logpmf_vectorized(x, ks_final, c_final, s)
                    log_joint    = logpmf + np.log(pi_final + EPS)[None, :]
                    log_evidence = logsumexp(log_joint, axis=1, keepdims=True)
                    resp_final   = np.exp(log_joint - log_evidence)

            if self.verbose:
                logger.info("Final: K=%d, loglik=%.3f, AIC=%.3f", K_final, loglik_final, aic_final)

            if aic_final < best_aic:
                # Recompute resp and basins on ALL cells (not just the mini-batch)
                resp_final, _ = predict_resp(
                    x_all, ks_final, c_final, pi=pi_final, forcing=self.mean_forcing_em, s=s_all
                )
                basins, pi_final = _assign_basins(
                    resp_final, x_all, ks_final, c_final, vect_t_all,
                    self.preserve_mean_values, len(ks_final),
                    self.mean_forcing_em, final=True, s=s_all
                )
                best_aic = aic_final
                best_model = {
                    'ks':              ks_final,
                    'c':               c_final,
                    'pi':              pi_final,
                    'basins':          basins,
                    'resp':            resp_final,
                    'loglik':          loglik_final,
                    'n_components':    K_final,
                    'aic':             aic_final,
                    'initial_K_tried': K_try,
                }

        self.best_model = best_model
        return best_model