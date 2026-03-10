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
* ``infer_kinetics_temporal_scaled`` – scaled kinetics version

Auxiliary helpers and legacy utilities are retained for compatibility.
"""

from typing import Any, Literal

import numpy as np
from numpy import ndarray, floating
from scipy.special import gammaln, logsumexp, psi, polygamma
from scipy.stats import nbinom
from scipy.optimize import minimize
from scipy import stats
import ot
import logging

from CardamomOT.logging import get_logger

# module-level logger
logger = get_logger(__name__)

EPS = 1e-12

# ---------------------------
# Fonctions utilitaires d'origine (conservées pour l'init)
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


def infer_kinetics_temporal(x, times, a_init=np.ones(100), b_init=1, max_iter=100, seuil=0.001, tol=1e-6, verb=False) -> tuple[Any, Any]:
    """
    Version originale pour l'initialisation avec des temps discrets.
    """
    t = np.sort(list(set(times)))
    m: int = t.size
    n = np.zeros(m)
    a = np.zeros(m)
    b = np.zeros(m)
    
    for i in range(m):
        cells = (times == t[i])
        n[i] = np.sum(cells)
        a[i], b[i] = estim_gamma_poisson(x[cells], mod=(i==m)-(i==0), 
                                         a_init=a_init[i], b_init=b_init)
    b = np.mean(b)
    
    # Σ x_i / s_i  (dénominateur pour b)
    sx: float = max(np.sum(x), EPS)

    k, c = 0, 0
    while (k == 0) or (k < max_iter and c > tol):
        da = np.zeros(m)
        for i in range(m):
            if a[i] > 0:
                cells = (times == t[i])
                z = a[i] + x[cells]
                p0 = np.sum(psi(z))
                p1 = np.sum(polygamma(1, z))

                # gradient : Σ_i [ψ(x_i+a) + log(b/s_i) - log(1+b/s_i) - ψ(a)]
                log_b   = np.log(b)
                log_1pb = np.log(1 + b)
                d = p0 + n[i] * (log_b - log_1pb) - n[i] * psi(a[i])
                h = p1 - n[i] * polygamma(1, a[i])
                if h != 0:
                    da[i] = -d / h
        
        a += np.maximum(da, -a)
        b = np.sum(n*a)/sx
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
    a = np.maximum(a, np.minimum(b/2, np.max(a)/100))

    return a, b


def infer_kinetics_temporal_scaled(x, s, times, a_init=np.ones(100), b_init=1,
                                    max_iter=100, seuil=0.001, tol=1e-6, verb=False) -> tuple[np.ndarray, floating[Any]]:
    """
    Version scalée de infer_kinetics_temporal.
    Modèle : X_i | bassin k  ~  NB(a_k,  b/s_i)
    La seule différence avec la version originale :
      - le gradient utilise log(b/s_i) au lieu de log(b)
      - la fermeture de b utilise Σ x_i/s_i au lieu de Σ x_i
    """
    t = np.sort(list(set(times)))
    m: int = t.size
    n = np.zeros(m)
    a = np.zeros(m)
    b = np.zeros(m)

    for i in range(m):
        cells = (times == t[i])
        n[i] = np.sum(cells)
        a[i], b[i] = estim_gamma_poisson(x[cells]/s[cells], mod=(i==m)-(i==0),
                                       a_init=a_init[i], b_init=b_init)
    b = np.mean(b) 

    # Σ x_i / s_i  (dénominateur pour b)
    sx: float = max(np.sum(x / s), EPS)

    k, c = 0, 0
    while (k == 0) or (k < max_iter and c > tol):
        da = np.zeros(m)
        for i in range(m):
            if a[i] > 0:
                cells = (times == t[i])
                x_c = x[cells]
                s_c = s[cells]
                z   = a[i] + x_c                        # (n_i,)
                p0  = np.sum(psi(z))
                p1  = np.sum(polygamma(1, z))

                # gradient : Σ_i [ψ(x_i+a) + log(b/s_i) - log(1+b/s_i) - ψ(a)]
                log_b_s   = np.log(b) - np.log(s_c)
                log_1pb_s = np.log(1 + b / s_c)
                d = p0 + np.sum(log_b_s - log_1pb_s) - n[i] * psi(a[i])
                h = p1 - n[i] * polygamma(1, a[i])
                if h != 0:
                    da[i] = -d / h

        a += np.maximum(da, -a)
        b = np.sum(n * a) / sx          # fermeture analytique corrigée
        c  = np.max(np.abs(da))
        k += 1
        if (k > 100) and (b > 1/seuil or b < seuil):
            break

    b_old = b
    b = np.clip(b, seuil, 1/seuil)
    a *= b / (b_old + EPS)
    a: np.ndarray[Any, np.dtype[Any]]  = np.maximum(a, np.minimum(b/2, np.max(a)/100))
    return a, b


def infer_kinetics_preserve_mean_values_assignment(x, resp, seuil=0.01, a_init=None, b_init=None,
                                   tol=1e-6, max_iter=100,  
                                   damping=0.7, verb=False) -> tuple[Any, Any]:
    """
    Version adaptée de infer_kinetics_temporal pour l'EM avec preserve_mean_values assignments.
    
    Optimise analytiquement les paramètres a[0],...,a[K-1] et b (c dans notre notation)
    en maximisant la log-vraisemblance pondérée par les responsabilités.
    
    Parameters:
    -----------
    x : array (N,)
        Observations (counts)
    resp : array (N, K)
        Responsabilités (probabilités d'appartenance aux K composantes)
    seuil : float
        Borne inférieure pour a et b
    a_init : array (K,) ou None
        Initialisation des a
    b_init : float ou None
        Initialisation de b
    damping : float
        Facteur d'amortissement pour Newton-Raphson (0 < damping <= 1)
        Plus petit = plus stable mais plus lent
    
    Returns:
    --------
    a : array (K,)
        Paramètres de forme optimaux (ks dans notre notation)
    b : float
        Paramètre de dispersion commun (c dans notre notation)
    """
    x = np.asarray(x).reshape(-1)
    resp = np.asarray(resp)
    N, K = resp.shape
    
    # Effectifs pondérés par composante
    n = resp.sum(axis=0) + EPS  # (K,)
    
    # Initialisation
    if a_init is None:
        # Init par moyenne pondérée
        a = np.zeros(K)
        for k in range(K):
            weighted_mean = np.sum(resp[:, k] * x) / n[k]
            a[k] = max(seuil, weighted_mean)
    else:
        a = np.array(a_init).copy()
    
    if b_init is None:
        b = 1.0
    else:
        b = float(b_init)
    
    # Somme totale pondérée (pour mise à jour de b)
    sx: np.bool_ | float = max(np.sum(resp * x[:, None]), EPS)
    
    # Newton-Raphson avec damping
    iteration, conv_metric = 0, 0
    
    while (iteration == 0) or (iteration < max_iter and conv_metric > tol):
        da = np.zeros(K)
        
        for k in range(K):
            if a[k] > seuil * 0.1:  # Éviter divisions par 0
                # z = a[k] + x pour toutes les observations
                z = a[k] + x  # (N,)
                
                # Gradient et Hessian pondérés par resp[:, k]
                # d/da log p(x|a,b) pondéré
                p0 = np.sum(resp[:, k] * psi(z))
                p1: np.bool_ = np.sum(resp[:, k] * polygamma(1, z))
                
                gradient = n[k] * (np.log(b) - np.log(b + 1) - psi(a[k])) + p0
                hessian = p1 - n[k] * polygamma(1, a[k])
                
                # Newton step avec damping
                if abs(hessian) != 0:
                    da[k] = -damping * gradient / hessian
                else:
                    da[k] = 0.0
        
        a += np.maximum(da, -a)
        b = np.sum(n*a)/sx
        
        # Métrique de convergence
        conv_metric = np.max(np.abs(da))
        iteration += 1
        
        # Sécurité : sortir si b diverge
        if (iteration > 100) and (b > 1/seuil or b < seuil): break
    
    # Vérifications finales
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
    a = np.maximum(a, np.minimum(b/2, np.max(a)/100))
    
    return a, b


def nb_logpmf_vectorized(x, ks, c, s=None):
    """
    Log-PMF NB vectorisée avec scaling cellulaire (read depth).

    Modèle : X_i | z=k  ~  NB(ks_k,  c / s_i)
    soit    E[X_i | k] = s_i * ks_k / c   (le scaling multiplie la moyenne).

    Parameters
    ----------
    x  : (N,)   observations (comptages entiers)
    ks : (K,)   shape parameters
    c  : float  dispersion du gène  (partagée entre composantes)
    s  : (N,)   facteurs de read depth cellulaires (médiane = 1)

    Returns
    -------
    logpmf : (N, K)
    """
    x  = np.asarray(x).reshape(-1)
    ks = np.clip(np.asarray(ks).reshape(-1), 1e-8, 1e5)
    if s is not None:
        s  = np.clip(np.asarray(s).reshape(-1),  1e-8, 1e8)
    else:
        s = np.ones_like(x)
    c  = float(np.clip(c, 1e-8, 1e5))

    # c_eff[i] = c / s[i]  →  (N,)
    c_eff = c / s                           # (N,)

    X  = x[:, None]                         # (N, 1)
    K  = ks[None, :]                        # (1, K)
    C = c_eff[:, None]                     # (N, 1)

    ln_c: np.ndarray[Any, np.dtype[Any]]   = np.log(C + EPS)               # (N, 1)
    ln_1pc: np.ndarray[Any, np.dtype[Any]] = np.log(1.0 + C + EPS)         # (N, 1)

    return (gammaln(X + K) - gammaln(K) - gammaln(X + 1.0)
            + K * ln_c - (X + K) * ln_1pc)


def zinb_logpmf_vectorized(x, ks, c, pi_zero, s=None):
    """ZINB log-pmf matrix."""
    log_nb = nb_logpmf_vectorized(x, ks, c, s=s)
    X = np.asarray(x)
    zeros_mask = (X == 0)
    N: int = X.size
    K = ks.size

    if np.isscalar(pi_zero):
        pis: np.ndarray[Any, np.dtype[Any]] = np.full(K, float(pi_zero))
    else:
        pis: np.ndarray[Any, np.dtype[Any]] = np.asarray(pi_zero).astype(float).reshape(-1)
    pis: np.ndarray[Any, np.dtype[Any]] = np.clip(pis, 0.0, 1.0 - 1e-12)
    logpis: np.ndarray[Any, np.dtype[Any]] = np.log(pis + EPS)
    log1mpis: np.ndarray[Any, np.dtype[Any]] = np.log(1.0 - pis + EPS)

    logpmf = np.empty_like(log_nb)

    if N == 0 or K == 0:
        return logpmf

    mask_nonzero = ~zeros_mask
    if np.any(mask_nonzero):
        logpmf[mask_nonzero, :] = log_nb[mask_nonzero, :] + log1mpis[None, :]

    if np.any(zeros_mask):
        log_nb0 = log_nb[zeros_mask, :]
        a = log1mpis[None, :] + log_nb0
        b: np.ndarray[Any, np.dtype[Any]] = logpis[None, :]
        M: np.ndarray[Any, np.dtype[Any]] = np.maximum(a, b)
        logpmf[zeros_mask, :] = M + np.log(np.exp(a - M) + np.exp(b - M) + EPS)

    return logpmf


def predict_resp(x, ks, c, s=None, pi=None, pi_zero=None, zi=None) -> tuple[Any, Any]:
        """
        Calcule les responsabilités.
        """

        # Test : impose pi=None (we don't modify the priori on the proportion of basins)
        # pi = None

        n_components: int = len(ks)
        if pi is None:
            pi = np.ones(n_components) / n_components
        else:
            pi = pi / (pi.sum() + EPS)

        if zi is None:
            logpmf = nb_logpmf_vectorized(x, ks, c, s=s)
        else:
            logpmf = zinb_logpmf_vectorized(x, ks, c, pi_zero, s=s)

        log_joint    = logpmf + np.log(pi + EPS)[None, :]
        log_evidence = logsumexp(log_joint, axis=1, keepdims=True)
        resp         = np.exp(log_joint - log_evidence)
        resp         = np.clip(resp, EPS, 1.0)
        resp        /= resp.sum(axis=1, keepdims=True)
        return resp, log_joint


def hard_em_scaled(data, s, n_components, ks_init, c_init, seuil,
                   tol=1e-6, max_iter_loop=200,
                   basins_temporal=None, vect_t=None,
                   preserve_mean_values=0, mean_forcing=1.0):
    """
    Hard EM avec scaling cellulaire.
    Seuls changements vs hard_em :
      - E-step via predict_resp
      - M-step via infer_kinetics_temporal_scaled
      - _apply_temporal_constraints travaille sur x/s pour les moyennes
    """
    n_cells = data.size
    ks, c   = ks_init.copy(), c_init

    resp, log_proba = predict_resp(data, ks, c, s=s)
    basins, pi = _assign_basins(resp, data, ks, c, vect_t,
                                preserve_mean_values, n_components, mean_forcing)
    if len(np.unique(basins)) < n_components:
        basins, pi = _assign_basins(resp, data, ks, c, vect_t,
                                    0, n_components, mean_forcing)

    log_likelihood_old = np.sum([log_proba[cell, basins[cell]]
                                  for cell in range(n_cells)])

    for it in range(max_iter_loop):
        ks_new, c_new = infer_kinetics_temporal_scaled(
            data, s, basins, a_init=ks, b_init=c, seuil=seuil,
            max_iter=int(1e5), tol=tol
        )

        if basins_temporal is not None:
            # _apply_temporal_constraints : les moyennes cibles sont sur x/s
            ks_new = _apply_temporal_constraints(
                data / s, basins_temporal, ks_new, c_new, n_components
            )

        resp_new, log_proba = predict_resp(
            data, ks_new, c_new, s=s, pi=pi
            )
        basins_new, pi_new  = _assign_basins(resp_new, data, ks_new, c_new, vect_t,
                                              preserve_mean_values, n_components,
                                              mean_forcing)
        if len(np.unique(basins_new)) < n_components:
            return ks, c, pi, basins

        if it:
            log_likelihood_new = np.sum([log_proba[cell, basins_new[cell]]
                                          for cell in range(n_cells)])
            if (log_likelihood_new - log_likelihood_old) < 1/max_iter_loop:
                break
            log_likelihood_old = log_likelihood_new

        ks, c, pi, basins = ks_new.copy(), c_new, pi_new.copy(), basins_new.copy()

    return ks, c, pi, basins


def hard_em(data, n_components, ks_init, c_init, seuil, tol=1e-6, max_iter_loop=200, 
            basins_temporal=None, vect_t=None, preserve_mean_values=0, mean_forcing=1.0):
    """
    Hard EM pour mélange de Binomiales Négatives avec contraintes temporelles.
    
    Parameters:
    -----------
    mean_forcing : float
        Poids de la contrainte temporelle (0 = pas de contrainte, 1 = contrainte forte)
    """
    
    n_cells = data.size 
    basins: np.ndarray[Any, np.dtype[Any]] = np.zeros(n_cells, dtype=int)
    ks, c = ks_init.copy(), c_init
    
    # Initialisation
    resp, log_proba = predict_resp(data, ks, c)
    basins, pi = _assign_basins(resp, data, ks, c, vect_t, preserve_mean_values, n_components, mean_forcing)

    if len(np.unique(basins)) < n_components:
        basins, pi = _assign_basins(resp, data, ks, c, vect_t, 0, n_components, mean_forcing)
    
    log_likelihood_old = np.sum([log_proba[cell, basins[cell]] for cell in range(n_cells)])
    
    for it in range(max_iter_loop):
        # M-step : mise à jour des paramètres
        ks_new, c_new = infer_kinetics_temporal(data, basins, a_init=ks, b_init=c, 
                                                 seuil=seuil, max_iter=1e5, tol=tol)
        
        # Contraintes sur les paramètres basées sur basins_temporal
        if basins_temporal is not None:
            ks_new = _apply_temporal_constraints(data, basins_temporal, ks_new, c_new, n_components)
        
        # E-step : réassignation
        resp_new, log_proba = predict_resp(
            data, ks_new, c_new, pi=pi
            )
        basins_new, pi_new = _assign_basins(resp_new, data, ks_new, c_new, 
                                            vect_t, preserve_mean_values, 
                                            n_components, mean_forcing)
        
        if len(np.unique(basins_new)) < n_components:
            return ks, c, pi, basins
        
        # Vérification convergence
        if it:
            log_likelihood_new = np.sum([log_proba[cell, basins_new[cell]] for cell in range(n_cells)])
            if (log_likelihood_new - log_likelihood_old) < 1/max_iter_loop:
                break
            log_likelihood_old = log_likelihood_new
        
        ks, c, pi, basins = ks_new.copy(), c_new, pi_new.copy(), basins_new.copy()

    return ks, c, pi, basins


def _assign_basins(resp, data, ks, c, vect_t, preserve_mean_values, n_components, mean_forcing, final=False):
    """Assigne les cellules aux bassins avec contraintes temporelles optionnelles."""
    
    n_cells = data.size

    if not preserve_mean_values:
        return np.argmax(resp, axis=1), resp.sum(axis=0) / (resp.sum() + EPS)
    
    if vect_t is None:
        mu = np.ones(n_cells) / n_cells
        nu = _compute_nu_with_temporal_constraint(
            resp, data, ks, c, n_components, mean_forcing
        )
        coupling = ot.bregman.sinkhorn(mu, nu, -np.log(resp), reg=1)
        return np.argmax(coupling, axis=1), nu
    
    # Soft clustering avec contraintes temporelles
    basins: np.ndarray[Any, np.dtype[Any]] = np.zeros(n_cells, dtype=int)
    pi_final = {}
    pi = np.zeros(n_components)
    for t_i in np.unique(vect_t):
        indices = (vect_t == t_i)
        resp_i = resp[indices]
        n_cells_i = np.sum(indices)
        
        # Distribution uniforme des cellules
        mu = np.ones(n_cells_i) / n_cells_i
        
        # Calcul de nu avec contrainte temporelle
        nu = _compute_nu_with_temporal_constraint(
            resp_i, data[indices], ks, c, n_components, mean_forcing
        )
        
        # Transport optimal
        coupling = ot.bregman.sinkhorn(mu, nu, -np.log(resp_i), reg=1)
        basins[indices] = np.argmax(coupling, axis=1)
        pi_final[t_i] = nu
        pi += nu * n_cells_i / n_cells
    
    if final: return basins, pi_final
    return basins, pi


def _compute_nu_with_temporal_constraint(resp_i, data_t, ks, c, n_components, mean_forcing):
    """
    Calcule la distribution cible nu en équilibrant likelihood et contrainte temporelle.
    
    La contrainte temporelle force: sum_k (nu_k * ks_k / c) ≈ mean_t
    """
    # Proportions basées sur la likelihood
    nu_likelihood = np.sum(resp_i, axis=0)
    nu_likelihood /= np.sum(nu_likelihood)
    
    # Proportions basées sur la contrainte de moyenne
    means_components = ks / c  # Moyenne de chaque NB
    nu = _solve_mean_constraint(means_components, data_t, ks, c, nu_likelihood, n_components, mean_forcing)
    nu = np.clip(nu, EPS, 1)  # Éviter les valeurs nulles
    nu /= np.sum(nu)
    
    return nu


def neg_log_likelihood_logits(logits, data, r, p):
    """
    logits : array (K,)
    data   : array (n,)
    r, p   : NB parameters (K,)
    """
    # softmax -> nu
    logits = logits - np.max(logits)  # stabilité numérique
    exp_logits = np.exp(logits)
    nu = exp_logits / np.sum(exp_logits)

    # log-probas du mélange
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

    # récupération de nu*
    logits_opt = res.x
    logits_opt -= np.max(logits_opt)
    nu_star = np.clip(np.exp(logits_opt), 1e-12, 1)
    nu_star /= nu_star.sum()

    return nu_star


def ks_statistic(data_full, nu, r, p, repet=10, n_cells_init=200) -> Any | float:

    n_cells: int = min(data_full.size, n_cells_init)
    stat = 0
    for _ in range(repet):
        # données triées
        data = data_full[np.random.choice(data_full.size, n_cells, replace=False)]
        data_sorted = np.sort(data)
        n: int = len(data_sorted)

        # CDF empirique
        emp_cdf = np.arange(1, n + 1) / n

        # CDF théorique du mélange
        model_cdf = np.zeros(n)
        for k in range(len(nu)):
            model_cdf += nu[k] * nbinom.cdf(data_sorted, r[k], p)

        # statistique KS
        stat += np.max(np.abs(emp_cdf - model_cdf))
    
    return stat / repet


def _solve_mean_constraint(means_components, data_t, ks, c, nu_init, n_components, mean_forcing):
    """
    Trouve les proportions nu qui satisfont: sum(nu_k * means_k) = target_mean
    tout en minimisant la distance à la distribution uniforme.
    
    Résout un problème d'optimisation quadratique sous contrainte.
    """
    mean_t = np.mean(data_t)

    # Adjust the level of regularization
    nu_star = compute_nu_star(data=data_t,
                            r_components=ks,
                            p_components=c/(1 + c),
                            nu_init=nu_init
                            )
    stat_KS: Any | float = ks_statistic(data_t, nu_star, ks , c / (1 + c))
    if mean_forcing > 0: alpha_reg = np.clip(stat_KS / mean_forcing, 0.0, 1.0)
    else: alpha_reg = 1.0

    # Fonction objectif : distance à la distribution uniforme
    def objective(nu):
        mean_pred = np.dot(nu, means_components)
        mean_erreur = (mean_pred - mean_t) / (EPS + mean_t)
        err = mean_erreur**2
        return (1 - alpha_reg) * err + alpha_reg * np.sum((nu - nu_init)**2)
    
    # Contrainte : somme = 1
    def constraint_sum(nu):
        return np.sum(nu) - 1
    
    constraints = [
        {'type': 'eq', 'fun': constraint_sum}
    ]
    
    bounds: list[tuple[int, int]] = [(0, 1) for _ in range(n_components)]
    
    result = minimize(objective, nu_init, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x
    

def _apply_temporal_constraints(data, basins_temporal, ks, c, n_components):
    """Applique des contraintes sur les paramètres basées sur basins_temporal."""
    mean_min = np.mean(data[basins_temporal == 0])
    mean_max = np.mean(data[basins_temporal == n_components-1])
    ks[0] = np.minimum(mean_min * c, ks[0])
    ks[-1] = np.maximum(mean_max * c, ks[-1])

    ks = np.maximum(ks, np.minimum(c/2, np.max(ks)/100))
    
    return ks


def infer_kinetics_scaled(x, s, resp, seuil=0.01, a_init=None, b_init=None,
                          tol=1e-6, max_iter=100, damping=0.7, verb=False) -> tuple[Any, Any]:
    """
    M-step analytique pour mélange NB avec read depth cellulaire s_i.

    Modèle : X_i | k  ~  NB(a_k,  c/s_i)
    La log-vraisemblance pondérée par resp est :

      ℓ(a, c) = Σ_i Σ_k r_{ik} [
          log Γ(x_i + a_k) - log Γ(a_k) - log Γ(x_i+1)
          + a_k * log(c/s_i) - (x_i + a_k) * log(1 + c/s_i)
      ]

    On maximise via Newton-Raphson sur a_k et fermeture analytique pour c.

    Parameters
    ----------
    x    : (N,)   comptages
    s    : (N,)   read depth (médiane = 1)
    resp : (N, K) responsabilités

    Returns
    -------
    a : (K,)  shape params
    b : float  dispersion c  (tel que mean_k = s_i * a_k / c)
    """
    x: np.ndarray[Any, np.dtype[Any]]    = np.asarray(x,    dtype=float).reshape(-1)
    s: np.ndarray[Any, np.dtype[Any]]    = np.clip(np.asarray(s, dtype=float).reshape(-1), 1e-8, 1e8)
    resp = np.asarray(resp)
    N, K = resp.shape

    n  = resp.sum(axis=0) + EPS           # (K,)  effectifs pondérés

    # Initialisation
    if a_init is None:
        a: np.ndarray[Any, np.dtype[Any]] = np.array([max(seuil, np.sum(resp[:, k] * x / s) / n[k])
                      for k in range(K)])
    else:
        a: np.ndarray[Any, np.dtype[Any]] = np.array(a_init, dtype=float).copy()

    b: float = float(b_init) if b_init is not None else 1.0

    # Somme pondérée de x / s_i  (pour la fermeture analytique de b)
    # E[X/s | k] = a_k / b  →  b = Σ_k n_k * a_k / Σ_i Σ_k r_{ik} * x_i/s_i
    sx = max(np.sum(resp * (x / s)[:, None]), EPS)

    iteration, conv_metric = 0, 0.0

    while (iteration == 0) or (iteration < max_iter and conv_metric > tol):
        da = np.zeros(K)

        for k in range(K):
            if a[k] > seuil * 0.1:
                z  = a[k] + x                          # (N,)
                p0 = np.sum(resp[:, k] * psi(z))
                p1: np.bool_ = np.sum(resp[:, k] * polygamma(1, z))

                # log(c/s_i) = log b - log s_i  ;  log(1 + c/s_i) = log(1 + b/s_i)
                # gradient w.r.t. a_k :
                #   Σ_i r_{ik} [ψ(x_i+a_k) + log(b/s_i) - log(1+b/s_i) - ψ(a_k)]
                log_b_over_s   = np.log(b + EPS) - np.log(s + EPS)           # (N,)
                log_1p_b_over_s: np.ndarray[Any, np.dtype[Any]] = np.log(1.0 + b / s + EPS)                  # (N,)
                grad_base = np.sum(resp[:, k] * (log_b_over_s - log_1p_b_over_s))

                gradient = n[k] * (-psi(a[k])) + p0 + grad_base
                hessian  = p1 - n[k] * polygamma(1, a[k])

                if abs(hessian) > EPS:
                    da[k] = -damping * gradient / hessian

        a += np.maximum(da, -a)
        b  = np.sum(n * a) / sx               # fermeture analytique

        conv_metric = np.max(np.abs(da))
        iteration  += 1
        if iteration > 100 and (b > 1.0/seuil or b < seuil):
            break

    if (iteration >= max_iter or conv_metric > tol) and verb:
        logger.warning("[scaled_kinetics] conv warning iter=%d conv=%.2e b=%.4f",
                       iteration, conv_metric, b)

    b_old: Any | float = b
    b = np.clip(b, seuil, 1.0/seuil)
    a *= b / (b_old + EPS)
    a  = np.maximum(a, np.minimum(b / 2, np.max(a) / 100))
    return a, b


def em_vectorized_nb_zinb_scaled(x, s, ks_init, c_init, pi_init=None,
                                  pi_zero_init=None, zi_mode=None,
                                  max_iter=200, tol=1e-6, seuil=0.01,
                                  damping=0.7, verbose=False):
    """
    EM pour mélange NB avec read depth cellulaire s_i.

    Identique à em_vectorized_nb_zinb mais utilise nb_logpmf_vectorized dans
    l'E-step et infer_kinetics_scaled dans le M-step.

    Parameters
    ----------
    x      : (N,)   comptages (entiers)
    s      : (N,)   read depth par cellule (médiane normalisée à 1)
    (autres paramètres identiques à em_vectorized_nb_zinb)
    """
    x: np.ndarray[Any, np.dtype[Any]] = np.asarray(x, dtype=float).reshape(-1)
    s: np.ndarray[Any, np.dtype[Any]] = np.clip(np.asarray(s, dtype=float).reshape(-1), 1e-8, 1e8)
    N: int = x.size
    ks: np.ndarray[Any, np.dtype[Any]] = np.asarray(ks_init, dtype=float).reshape(-1)
    K: int  = ks.size
    c  = float(c_init)

    pi = (np.ones(K) / K if pi_init is None
          else np.asarray(pi_init, dtype=float) / (np.sum(pi_init) + EPS))

    # Zero-inflation (reprise directe de la version sans scaling)
    pi_zero = 0
    if zi_mode == 'global':
        pi_zero: float = float(pi_zero_init) if pi_zero_init is not None else 0.05
    elif zi_mode == 'component':
        pi_zero = (np.full(K, 0.05) if pi_zero_init is None
                   else np.asarray(pi_zero_init, dtype=float).reshape(-1))
    elif zi_mode is not None:
        raise ValueError("zi_mode must be None, 'global', or 'component'")

    loglik_old: float = -np.inf
    resp = np.ones((N, K)) / K

    for it in range(max_iter):
        # ── E-step ────────────────────────────────────────────────────────
        if zi_mode is None:
            logpmf = nb_logpmf_vectorized(x, ks, c, s)            # (N, K)
        else:
            # ZINB avec scaling : composante NB scalée + spike à zéro
            log_nb = nb_logpmf_vectorized(x, ks, c, s)
            X: np.ndarray[Any, np.dtype[Any]] = np.asarray(x)
            zeros_mask = (X == 0)
            if np.isscalar(pi_zero):
                pis: np.ndarray[Any, np.dtype[Any]] = np.full(K, float(pi_zero))
            else:
                pis: np.ndarray[Any, np.dtype[Any]] = np.asarray(pi_zero, dtype=float).reshape(-1)
            pis: np.ndarray[Any, np.dtype[Any]]      = np.clip(pis, 0.0, 1.0 - EPS)
            logpis: np.ndarray[Any, np.dtype[Any]]   = np.log(pis + EPS)
            log1mpis: np.ndarray[Any, np.dtype[Any]] = np.log(1.0 - pis + EPS)

            logpmf = np.empty_like(log_nb)
            if np.any(~zeros_mask):
                logpmf[~zeros_mask] = log_nb[~zeros_mask] + log1mpis[None, :]
            if np.any(zeros_mask):
                a_ = log1mpis[None, :] + log_nb[zeros_mask]
                b_: np.ndarray[Any, np.dtype[Any]] = logpis[None, :]
                M: np.ndarray[Any, np.dtype[Any]]  = np.maximum(a_, b_)
                logpmf[zeros_mask] = M + np.log(np.exp(a_ - M) + np.exp(b_ - M) + EPS)

        log_joint    = logpmf + np.log(pi + EPS)[None, :]
        log_evidence = logsumexp(log_joint, axis=1, keepdims=True)
        resp         = np.exp(log_joint - log_evidence)
        resp         = np.clip(resp, EPS, 1.0)
        resp        /= resp.sum(axis=1, keepdims=True)

        loglik = float(np.sum(log_evidence))
        if verbose and it % 20 == 0:
            logger.info("[scaled EM] iter %d: loglik=%.4f, c=%.4f", it, loglik, c)

        if np.isfinite(loglik_old) and abs(loglik - loglik_old) < tol:
            if verbose:
                logger.info("[scaled EM] Converged at iter %d", it)
            break
        loglik_old: float = loglik

        # ── M-step ────────────────────────────────────────────────────────
        Nk = resp.sum(axis=0) + EPS
        pi = Nk / N

        ks, c = infer_kinetics_scaled(
            x, s, resp,
            seuil=seuil, a_init=ks, b_init=c,
            tol=tol, max_iter=int(1e5), damping=damping, verb=verbose
        )

        # Zero-inflation update (identique à la version non-scaled)
        if zi_mode == 'global':
            frac_zeros     = np.mean(x == 0)
            log_nb0        = nb_logpmf_vectorized(np.zeros(1), ks, c,
                                              np.ones(1)).ravel()
            nb0            = np.exp(log_nb0)
            expected_zero  = (pi * nb0).sum()
            pi_zero        = float(np.clip(frac_zeros - expected_zero, 0.0, 0.95))
        elif zi_mode == 'component':
            log_nb0 = nb_logpmf_vectorized(np.zeros(1), ks, c, np.ones(1)).ravel()
            nb0     = np.exp(log_nb0)
            frac_zeros_j = (resp[x == 0].sum(axis=0) / (N + EPS)
                            if np.any(x == 0) else np.zeros(K))
            pi_zero = np.clip((frac_zeros_j - pi * nb0) / (pi + EPS), 0.0, 0.95)

    # ── Log-vraisemblance finale ───────────────────────────────────────────
    final_logpmf = nb_logpmf_vectorized(x, ks, c, s)
    final_joint  = final_logpmf + np.log(pi + EPS)[None, :]
    final_loglik = float(np.sum(logsumexp(final_joint, axis=1)))

    return ks, c, pi, pi_zero, resp, final_loglik


def compute_aic_for_params_scaled(x, s, ks, c, pi, pi_zero, zi_mode) -> tuple[Any, float]:
    """Calcule l'AIC avec read depth pour un jeu de paramètres donné."""
    if zi_mode is None:
        logpmf     = nb_logpmf_vectorized(x, ks, c, s)
        num_params: int = len(ks) + 1 + (len(ks) - 1)
    else:
        # ZINB scaled : même logique que la version non-scaled
        log_nb = nb_logpmf_vectorized(x, ks, c, s)
        zeros_mask = (np.asarray(x) == 0)
        if np.isscalar(pi_zero):
            pis: np.ndarray[Any, np.dtype[Any]] = np.full(len(ks), float(pi_zero))
        else:
            pis: np.ndarray[Any, np.dtype[Any]] = np.asarray(pi_zero, dtype=float).reshape(-1)
        pis: np.ndarray[Any, np.dtype[Any]]      = np.clip(pis, 0.0, 1.0 - EPS)
        log1mpis: np.ndarray[Any, np.dtype[Any]] = np.log(1.0 - pis + EPS)
        logpis: np.ndarray[Any, np.dtype[Any]]   = np.log(pis + EPS)
        logpmf   = np.empty_like(log_nb)
        if np.any(~zeros_mask):
            logpmf[~zeros_mask] = log_nb[~zeros_mask] + log1mpis[None, :]
        if np.any(zeros_mask):
            a_ = log1mpis[None, :] + log_nb[zeros_mask]
            b_: np.ndarray[Any, np.dtype[Any]] = logpis[None, :]
            M: np.ndarray[Any, np.dtype[Any]]  = np.maximum(a_, b_)
            logpmf[zeros_mask] = M + np.log(np.exp(a_ - M) + np.exp(b_ - M) + EPS)
        num_params: int = (len(ks) + 1 + (len(ks) - 1) + 1 if zi_mode == 'global'
                      else len(ks) + 1 + (len(ks) - 1) + len(ks))

    log_joint    = logpmf + np.log(pi + EPS)[None, :]
    log_evidence = logsumexp(log_joint, axis=1)
    loglik       = float(np.sum(log_evidence))
    aic          = np.log(len(x)) * num_params - 2.0 * loglik
    return aic, loglik


def em_vectorized_nb_zinb(x, ks_init, c_init, pi_init=None, pi_zero_init=None,
                        zi_mode=None, max_iter=200, tol=1e-6, seuil=0.01, 
                        damping=0.7, verbose=False) -> Any:
    """
    EM pour mélange NB/ZINB avec M-step ANALYTIQUE via Newton-Raphson.
    
    Remplace l'estimation par moments par une optimisation directe de la 
    log-vraisemblance pondérée, comme dans infer_kinetics_temporal.
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

    # Zero-inflation setup
    pi_zero = 0
    if zi_mode == 'global':
        pi_zero: float = float(pi_zero_init) if pi_zero_init is not None else 0.05
    elif zi_mode == 'component':
        if pi_zero_init is None:
            pi_zero: np.ndarray[Any, np.dtype[Any]] = np.full(K, 0.05)
        else:
            pi_zero: np.ndarray[Any, np.dtype[Any]] = np.asarray(pi_zero_init).astype(float).reshape(-1)
    elif zi_mode is not None:
        raise ValueError("zi_mode must be None, 'global', or 'component'")

    loglik_old: float = -np.inf
    resp = np.ones((N, K)) / K

    for it in range(max_iter):
        # ==================
        # E-STEP (inchangé)
        # ==================
        if zi_mode is None:
            logpmf = nb_logpmf_vectorized(x, ks, c)
        else:
            logpmf = zinb_logpmf_vectorized(x, ks, c, pi_zero)

        logpi: np.ndarray[Any, np.dtype[Any]] = np.log(pi + EPS)[None, :]
        log_joint = logpmf + logpi
        
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
        
        # 1) Mise à jour de pi 
        Nk = resp.sum(axis=0) + EPS
        pi = Nk / N

        # 2) Mise à jour de ks et c via optimisation analytique
        ks, c = infer_kinetics_preserve_mean_values_assignment(
            x, resp, 
            seuil=seuil,
            a_init=ks, 
            b_init=c,
            tol=tol,
            max_iter=1e5,  # Sous-itérations Newton-Raphson
            damping=damping,
            verb=verbose
        )

        # 3) Mise à jour de pi_zero (ZINB uniquement)
        if zi_mode == 'global':
            frac_zeros = np.mean(x == 0)
            log_nb0 = nb_logpmf_vectorized(np.array([0]), ks, c).ravel()
            nb0 = np.exp(log_nb0)
            expected_nb_zero = (pi * nb0).sum()
            pi_zero = float(np.clip(frac_zeros - expected_nb_zero, 0.0, 0.95))
            
        elif zi_mode == 'component':
            log_nb0 = nb_logpmf_vectorized(np.array([0]), ks, c).ravel()
            nb0 = np.exp(log_nb0)
            if np.any(x == 0):
                frac_zeros_j = resp[x == 0].sum(axis=0) / (N + EPS)
            else:
                frac_zeros_j = np.zeros(K)
            pi_zero = np.clip((frac_zeros_j - pi * nb0) / (pi + EPS), 0.0, 0.95)

    # Final likelihood
    if zi_mode is None:
        final_logpmf = nb_logpmf_vectorized(x, ks, c)
    else:
        final_logpmf = zinb_logpmf_vectorized(x, ks, c, pi_zero)
    
    final_joint = final_logpmf + np.log(pi + EPS)[None, :]
    final_logev = logsumexp(final_joint, axis=1)
    final_loglik = np.sum(final_logev)

    return ks, c, pi, pi_zero, resp, final_loglik


# ---------------------------
# Fonction pour calculer l'AIC
# ---------------------------

def compute_aic_for_params(x, ks, c, pi, pi_zero, zi_mode) -> tuple[Any, floating[Any]]:
    """Calcule l'AIC pour un jeu de paramètres donné."""
    if zi_mode is None:
        logpmf = nb_logpmf_vectorized(x, ks, c)
        num_params: int = len(ks) + 1 + (len(ks) - 1)
    else:
        logpmf = zinb_logpmf_vectorized(x, ks, c, pi_zero)
        if zi_mode == 'global':
            num_params: int = len(ks) + 1 + (len(ks) - 1) + 1
        else:  # component
            num_params: int = len(ks) + 1 + (len(ks) - 1) + len(ks)
    
    log_joint = logpmf + np.log(pi + EPS)[None, :]
    log_evidence = logsumexp(log_joint, axis=1)
    loglik = np.sum(log_evidence)
    
    aic = np.log(x.size) * num_params - 2 * loglik
    return aic, loglik


# ---------------------------
# Classe principale (refactorisée)
# ---------------------------

class NegativeBinomialMixtureEM:
    def __init__(self, min_components=1, max_components=3, zi=None, refilter=0.0, hard_em=1, mean_forcing_em=1.0,
                 tol=1e-5, max_iter_em=200, verbose=False, preserve_mean_values=0,
                 compare_init_aic=True, damping=1.0) -> None:
        """
        Mélange NB/ZINB avec M-step analytique optimal.
        
        New parameters:
        ---------------
        compare_init_aic : bool
            Si True, compare l'AIC de l'initialisation avec l'AIC post-EM
        damping : float (0, 1]
            Facteur d'amortissement pour Newton-Raphson (0.5-0.8 = stable, 1.0 = rapide)
        """
        assert min_components >= 1 and max_components >= min_components
        self.min_components: int = min_components
        self.max_components: int = max_components
        self.zi = zi
        self.preserve_mean_values: int = preserve_mean_values
        self.hard_em: int = hard_em
        self.mean_forcing_em: float = mean_forcing_em
        self.refilter: float = refilter
        self.tol: float = tol
        self.max_iter_em: int = max_iter_em
        self.verbose: bool = verbose
        self.compare_init_aic: bool = compare_init_aic
        self.damping: float = damping
        self.best_model = None


    def _init_for_K(self, x, K, vect_t=None, quant_init=None, seuil=0.01):
        mean = np.mean(x)
        var = np.var(x)
        
        if mean <= 0:
            k_glob = 1.0
            c_glob = 1.0
        else:
            if var <= mean + 1e-8:
                k_glob: float = max(50.0, mean)
            else:
                k_glob: float = max(1e-3, (mean**2) / (var - mean))
            c_glob: float = max(1e-6, k_glob / (mean + EPS))

        if vect_t is not None:
            try:
                a_per_time, b_est = infer_kinetics_temporal(x, vect_t, seuil, max_iter=1e5)
                ks_init = np.linspace(np.min(a_per_time), np.max(a_per_time), K)
                if a_per_time.size < K:
                    qs = np.linspace(0, 1, K)
                    ks_init += np.quantile(a_per_time, qs)
                    ks_init /= 2
                c_init = b_est
                if self.verbose:
                    logger.info("Warm start: ks_init=%s, c_init=%s", ks_init, c_init)
            except Exception as e:
                if self.verbose:
                    logger.warning("Warm start failed: %s, using fallback", e)
                ks_init = k_glob * np.linspace(0.8, 1.2, K)
                c_init: float = c_glob
        else:
            if len(quant_init) != 2*K:
                quant_init = np.linspace(0, 1, 2*K)
            centers = np.quantile(x, quant_init)
            ks_init = np.array([np.mean(x[(centers[2*k] <= x) & 
                                          (x <= np.maximum(centers[2*k+1], centers[2*k]+1))]) * c_glob 
                                          for k in range(0, K)]) 
            ks_init = np.clip(ks_init, seuil, 1e5)
            c_init: float = c_glob
        
        n: float = 1/2
        m, M = np.min(ks_init), np.max(ks_init)
        while (len(np.unique((ks_init/c_init).astype(int))) < K) and (n > 1/x.size):
            mn = min(m, np.quantile(seuil + x*c_init, n))
            Mn = max(M, np.quantile(seuil + x*c_init, 1-n))
            ks_init = np.linspace(mn, Mn, K)
            n /= 2 
        if self.verbose: logger.info("Init: ks_init=%s", ks_init)

        pi_init = np.ones(K) / K
        if self.zi is None:
            pi_zero_init = 0
        elif self.zi == 'global':
            frac_zeros = np.mean(x == 0)
            pi_zero_init: float = min(0.95, max(0.0, frac_zeros - 0.05))
        else:
            frac_zeros = np.mean(x == 0)
            pi_zero_init: np.ndarray[Any, np.dtype[Any]] = np.full(K, max(0.0, frac_zeros - 0.05))

        return ks_init.astype(float), float(c_init), pi_init.astype(float), pi_zero_init




    def _refilter_merge(self, ks, c, pi, pi_zero):
        modified = False
        order: np.ndarray[Any, np.dtype[np.signedinteger[Any]]] = np.argsort(ks)
        ks = ks[order].astype(float).copy()
        pi = pi[order].astype(float).copy()
        
        if pi_zero is not None and self.zi == 'component':
            pi_zero: np.ndarray[Any, np.dtype[Any]] = np.asarray(pi_zero)[order].astype(float).copy()
        elif pi_zero is not None:
            pi_zero = float(pi_zero)
        else:
            pi_zero = 0

        K = ks.size
        i = 0
        while i < K - 1:
            if abs(ks[i+1] - ks[i]) < self.refilter * c:
                modified = True
                weight_i = pi[i]
                weight_ip1 = pi[i+1]
                total_weight = weight_i + weight_ip1
                
                new_k = (ks[i] * weight_i + ks[i+1] * weight_ip1) / total_weight
                new_pi = total_weight
                
                if self.zi == 'component' and isinstance(pi_zero, np.ndarray):
                    new_pi_zero = (pi_zero[i] * weight_i + pi_zero[i+1] * weight_ip1) / total_weight
                    pi_zero: np.ndarray[Any, np.dtype[Any]] = np.delete(pi_zero, i+1)
                    pi_zero[i] = new_pi_zero
                
                ks = np.delete(ks, i+1)
                ks[i] = new_k
                pi = np.delete(pi, i+1)
                pi[i] = new_pi
                
                K -= 1
            else:
                i += 1
        
        pi = pi / (pi.sum() + EPS)
        return ks, c, pi, pi_zero, modified


    def _count_params(self, K):
        if self.zi == 'component':
            zi_p = K
        else:
            zi_p = 1
        return K + 1 + (K - 1) + zi_p
    

    def fit(self, x, vect_t=None, quant_init=None, seuil=0.001, s=None):
        """
        Ajuste le mélange NB sur les données x.

        Parameters
        ----------
        x      : (N,)  comptages (entiers)
        vect_t : (N,)  temps par cellule (optionnel)
        s      : (N,)  facteurs de read depth cellulaires (optionnel).
                       Si None, on utilise s=1 pour toutes les cellules
                       (comportement original).
                       Si fourni (issu de adata.obs['rd']), le modèle tient
                       compte du scaling : X_i|k ~ NB(ks_k, c/s_i).
        """
        x: np.ndarray[Any, np.dtype[Any]] = np.asarray(x).astype(int)
        N: int = x.size

        # ── Gestion du read depth ─────────────────────────────────────────
        use_scaling: bool = (s is not None)
        if use_scaling:
            # Pseudo-comptages pour l'initialisation (hard EM et _init_for_K)
            x_init = np.round(x / s).astype(int)
            x_init = np.clip(x_init, 0, None)
        else:
            s: np.ndarray[Any, np.dtype[Any]] = np.ones(N, dtype=float)
            x_init = x

        best_aic: float = np.inf
        best_model = None

        for K_try in range(self.min_components, self.max_components + 1):
            if self.verbose:
                logger.info("=== Trying K_init = %s ===", K_try)
            ks_init, c_init, pi_init, pi_zero_init = self._init_for_K(
                x_init, K_try, vect_t=vect_t, quant_init=quant_init, seuil=seuil
            )

            if self.hard_em:
                if vect_t is not None:
                    basins_temporal = np.ones_like(vect_t) * K_try
                    list_t = np.unique(vect_t)
                    mean_list: list[floating[Any]] = [np.mean(x[vect_t == time]) for time in list_t]
                    ml: np.signedinteger[Any] = np.argmin(mean_list)
                    Ml: np.signedinteger[Any] = np.argmax(mean_list)
                    basins_temporal[vect_t == list_t[ml]] = 0
                    basins_temporal[vect_t == list_t[Ml]] = K_try - 1
                    if use_scaling:
                        ks_init, c_init, pi_init, basins = hard_em_scaled(
                            x, s, K_try, ks_init, c_init, seuil, tol=self.tol,
                            basins_temporal=basins_temporal, vect_t=vect_t,
                            preserve_mean_values=self.preserve_mean_values,
                            mean_forcing=self.mean_forcing_em
                        )
                    else:
                        ks_init, c_init, pi_init, basins = hard_em(
                            x, K_try, ks_init, c_init, seuil, tol=self.tol,
                            basins_temporal=basins_temporal, vect_t=vect_t,
                            preserve_mean_values=self.preserve_mean_values,
                            mean_forcing=self.mean_forcing_em
                        )
                else:
                    if use_scaling:
                        ks_init, c_init, pi_init, basins = hard_em_scaled(
                            x, s, K_try, ks_init, c_init, seuil, tol=self.tol,
                            preserve_mean_values=self.preserve_mean_values,
                            mean_forcing=self.mean_forcing_em
                        )
                    else:
                        ks_init, c_init, pi_init, basins = hard_em(
                            x, K_try, ks_init, c_init, seuil, tol=self.tol,
                            preserve_mean_values=self.preserve_mean_values,
                            mean_forcing=self.mean_forcing_em
                        )

            if (self.refilter > 0.0) and (ks_init.size > 1):
                ks_merged, c_merged, pi_merged, pi_zero_merged, modified = \
                    self._refilter_merge(ks_init, c_init, pi_init, pi_zero_init)
                if modified and ks_merged.size < ks_init.size and ks_merged.size >= 1:
                    if self.verbose:
                        logger.info("Merged -> %d, re-fitting", ks_merged.size)
                    ks_init, c_init, pi_init, pi_zero_init = \
                        ks_merged, c_merged, pi_merged, pi_zero_merged

            # ── AIC de l'initialisation ───────────────────────────────────
            if self.compare_init_aic:
                if use_scaling:
                    aic_init, loglik_init = compute_aic_for_params_scaled(
                        x, s, ks_init, c_init, pi_init, pi_zero_init, self.zi
                    )
                else:
                    aic_init, loglik_init = compute_aic_for_params(
                        x, ks_init, c_init, pi_init, pi_zero_init, self.zi
                    )
                if self.verbose:
                    logger.info("Init AIC: %.3f, loglik: %.3f", aic_init, loglik_init)
            else:
                aic_init: float = np.inf
                loglik_init: float = -np.inf

            ks_final, c_final, pi_final, pi_zero_final, loglik_final = \
                ks_init.copy(), c_init, pi_init.copy(), pi_zero_init, loglik_init
            stable = False
            iter_count = 0

            while not stable and iter_count < 10:
                iter_count += 1

                # Choisir la version EM selon la présence ou non du scaling
                if use_scaling:
                    ks_fit, c_fit, pi_fit, pi_zero_fit, resp_fit, loglik_fit = \
                        em_vectorized_nb_zinb_scaled(
                            x, s, ks_final, c_final, pi_final, pi_zero_final,
                            zi_mode=self.zi, max_iter=self.max_iter_em,
                            tol=self.tol, seuil=seuil, damping=self.damping,
                            verbose=self.verbose
                        )
                else:
                    ks_fit, c_fit, pi_fit, pi_zero_fit, resp_fit, loglik_fit = \
                        em_vectorized_nb_zinb(
                            x, ks_final, c_final, pi_final, pi_zero_final,
                            zi_mode=self.zi, max_iter=self.max_iter_em,
                            tol=self.tol, seuil=seuil, damping=self.damping,
                            verbose=self.verbose
                        )

                if self.verbose:
                    logger.info("EM iter %d: K=%d, loglik=%.3f", iter_count, ks_fit.size, loglik_fit)

                if (self.refilter > 0.0) and (ks_fit.size > 1):
                    ks_merged, c_merged, pi_merged, pi_zero_merged, modified = \
                        self._refilter_merge(ks_fit, c_fit, pi_fit, pi_zero_fit)
                    if modified and ks_merged.size < ks_fit.size and ks_merged.size >= 1:
                        if self.verbose:
                            logger.info("Merged -> %d", ks_merged.size)
                        ks_final, c_final, pi_final, pi_zero_final = \
                            ks_merged, c_merged, pi_merged, pi_zero_merged
                        continue

                ks_final, c_final, pi_final, pi_zero_final, resp_final, loglik_final = (
                    ks_fit, c_fit, pi_fit, pi_zero_fit, resp_fit, loglik_fit
                )
                stable = True

            K_final   = ks_final.size
            num_params = self._count_params(K_final)
            aic_final  = np.log(x.size) * num_params - 2 * loglik_final

            # ── Comparer init vs EM ───────────────────────────────────────
            if self.compare_init_aic and self.max_iter_em:
                keep_init: np.bool_ | bool = (loglik_init >= loglik_final) or (
                    not self.refilter and vect_t is not None and
                    np.std(ks_init / c_init) > 2 * np.std(ks_final / c_final)
                )
                if keep_init:
                    if self.verbose:
                        logger.info(">> Keeping init (loglik %.3f >= %.3f)", loglik_init, loglik_final)
                    ks_final, c_final, pi_final, pi_zero_final = \
                        ks_init, c_init, pi_init, pi_zero_init
                    aic_final: Any | float    = aic_init
                    loglik_final = loglik_init
                    K_final      = ks_init.size
                    # Recalculer resp
                    if use_scaling:
                        logpmf = nb_logpmf_vectorized(x, ks_final, c_final, s)
                    elif self.zi is None:
                        logpmf = nb_logpmf_vectorized(x, ks_final, c_final)
                    else:
                        logpmf = zinb_logpmf_vectorized(
                            x, ks_final, c_final, pi_zero_final
                        )
                    log_joint    = logpmf + np.log(pi_final + EPS)[None, :]
                    log_evidence = logsumexp(log_joint, axis=1, keepdims=True)
                    resp_final   = np.exp(log_joint - log_evidence)

            if self.verbose:
                logger.info("Final: K=%d, loglik=%.3f, AIC=%.3f", K_final, loglik_final, aic_final)

            if aic_final < best_aic:
                resp_final, _ = predict_resp(
                    x, ks_final, c_final, s=s,
                    pi_zero=pi_zero_final, zi=self.zi, pi=pi_final
                )
                basins, pi_final = _assign_basins(
                    resp_final, x, ks_final, c_final, vect_t,
                    self.preserve_mean_values, len(ks_final),
                    self.mean_forcing_em, final=True
                )
                best_aic = aic_final
                best_model = {
                    'ks':              ks_final,
                    'c':               c_final,
                    'pi':              pi_final,
                    'pi_zero':         pi_zero_final,
                    'basins':          basins,
                    'resp':            resp_final,
                    'loglik':          loglik_final,
                    'n_components':    K_final,
                    'aic':             aic_final,
                    'initial_K_tried': K_try,
                }

        self.best_model = best_model
        return best_model
    

    def predict_proba(self, x, ks, c, pi_zero=None, zi=None):
        
        if zi is None:
            logpmf = nb_logpmf_vectorized(x, ks, c)
        else:
            logpmf = zinb_logpmf_vectorized(x, ks, c, pi_zero)
        
        log_evidence = logsumexp(logpmf, axis=1, keepdims=True)
        
        return np.sum(log_evidence)

