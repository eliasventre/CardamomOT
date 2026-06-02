"""
Core routines for network inference used in trajectory-based loops.

This module implements the optimization objectives, gradients, and
penalization schemes required to learn regulatory interactions from
multi-modal single-cell data.  It is designed to be imported lazily in
order to avoid heavy dependencies when only other parts of the package
are needed.
"""

from typing import Any
import multiprocessing as mp
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import check_grad
from numba import njit
from functools import partial
import logging

from CardamomOT.logging import get_logger

# module-level logger
logger = get_logger(__name__)

# import warnings
# warnings.filterwarnings("error")


# --- Hyperparameters ---
seuil = 1e-5
check_gradient = 0
alpha = 1 # How to balance differents sources when they are unbalanced (ex: RMS2V3 and RD136 with different number of cells, 0 if not rescaled (each cell equal), 1 if rescaled (each model equal)))
eps_CE = 1e-6
EPS = 1e-16
sc = 1e-3
r_elasticnet = 0.5


### FUNCTIONS FOR NETWORK INFERENCE

@njit(fastmath=True, cache=True)
def smoothed_l1_penalization(array, l1, sc=sc):
    # Apply a smoothed version of the L1 norm element-wise
    smoothed_array: np.ndarray[Any, np.dtype[Any]] = np.where(
        np.abs(array) < sc,  # check if absolute value is below threshold
        l1 * 0.5 * (array ** 2) / sc,  # quadratic region when |x| < sc
        l1 * (np.abs(array) - 0.5 * sc)  # linear region otherwise
    )
    return np.sum(smoothed_array)


@njit(fastmath=True, cache=True)
def grad_smoothed_l1_penalization(array, l1, sc=sc) -> np.ndarray:
    # Apply a smoothed version of the L1 norm element-wise
    smoothed_array = np.where(
        np.abs(array) < sc,  # check if absolute value is below threshold
        l1 * array / sc,  # quadratic gradient portion when |x| < sc
        l1 * np.sign(array)  # linear gradient otherwise
    )
    return smoothed_array


@njit(fastmath=True, cache=True)
def theta_penalization(theta, l):
    return smoothed_l1_penalization(theta, l)

@njit(fastmath=True, cache=True)
def grad_theta_penalization(theta, l) -> np.ndarray:
    return grad_smoothed_l1_penalization(theta, l)

@njit(fastmath=True, cache=True)
def l2_penalization(theta, l2):
    return np.sum(l2 * np.square(theta))

@njit(fastmath=True, cache=True)
def grad_l2_penalization(theta, l2):
    return l2 * 2 * theta

@njit(fastmath=True, cache=True)
def final_theta_penalization(theta, l):
    return r_elasticnet*l2_penalization(theta, l) + (1-r_elasticnet)*smoothed_l1_penalization(theta, l)

@njit(fastmath=True, cache=True)
def grad_final_theta_penalization(theta, l):
    return r_elasticnet*grad_l2_penalization(theta, l) + (1-r_elasticnet)*grad_smoothed_l1_penalization(theta, l)

### Define the norm of main_loss
@njit(fastmath=True, cache=True)
def main_loss(y_pred, y_true, l, loss, sc=sc, eps=eps_CE):

    if loss == 'l1':
        smoothed_array: np.ndarray[Any, np.dtype[Any]] = np.where(
            np.abs(y_pred - y_true) < sc,
            l * .5 * ((y_pred - y_true) ** 2) / sc,
            l * (np.abs(y_pred - y_true) - .5 * sc)
        )
        return np.sum(smoothed_array)

    elif loss == 'l2':
        return l * np.sum(np.square(y_pred - y_true))

    else:  # Cross-Entropy
        y_pred_c = np.clip(y_pred, eps, 1 - eps)
        y_true_c = np.clip(y_true, eps, 1 - eps)
        return -l * np.sum(
            y_true * np.log(y_pred_c / y_true_c) +
            (1 - y_true) * np.log((1-y_pred_c)/(1-y_true_c))
        )


@njit(fastmath=True, cache=True)
def grad_main_loss(y_pred, y_true, l, loss, sc=sc, eps=eps_CE):

    if loss == 'l1':
        smoothed_array: np.ndarray[Any, np.dtype[Any]] = np.where(
            np.abs(y_pred - y_true) < sc,
            l * (y_pred - y_true) / sc,
            l * np.sign(y_pred - y_true)
        )
        return smoothed_array

    elif loss == 'l2':
        return l * 2 * (y_pred - y_true)

    else:  # Cross-Entropy - VERSION ROBUSTE
        y_clipped = np.clip(y_pred, eps, 1 - eps)

        # Gradient: -(y_true/y - (1-y_true)/(1-y))
        denominator = y_clipped * (1 - y_clipped)
        grad = l * (y_clipped - y_true) / denominator
        mask = (y_pred >= eps) & (y_pred <= 1 - eps)

        return grad * mask


@njit(fastmath=True, cache=True)
def base_kon(theta_basal, theta_inter, y_prot) -> np.ndarray:
    """More robust version with clipping of exponentials to avoid overflow."""
    n_cells, G = y_prot.shape
    n_net = theta_basal.size
    Z = np.zeros((n_cells, n_net))
    result = np.empty((n_cells, n_net + 1))

    # Limit to avoid overflow of the exponential function
    MAX_EXP = 50.0  # exp(50) ≈ 5e21, largement suffisant

    for i in range(n_cells):
        # Calcul des logits
        for k in range(n_net):
            Z[i, k] = theta_basal[k]
            for g in range(G):
                Z[i, k] += y_prot[i, g] * theta_inter[g, k]

        # Find max for numerical stability
        z_max = np.max(Z[i])

        # Clip to avoid extreme values
        z_max = min(z_max, MAX_EXP)

        # Softmax stable
        denom = np.exp(min(-z_max, MAX_EXP))  # Classe 0
        result[i, 0] = denom

        for k in range(n_net):
            exp_val = np.exp(min(Z[i, k] - z_max, MAX_EXP))
            denom += exp_val
            result[i, k + 1] = exp_val

        # Normalisation
        result[i] /= denom

    return result


def _compute_sigma_per_sample(theta_inter, theta_basal_all, ref_network, yp, ys, us, n_networks):
    """Compute sigma for all cells, using per-sample basal from theta_basal_all.

    theta_basal_all : (n_samples, n_networks)
    Returns sigma : (N_cells, n_networks+1)
    """
    N_cells = yp.shape[0]
    sigma = np.empty((N_cells, n_networks + 1))
    inter_ref = theta_inter * ref_network  # (G, n_networks)
    for s in us:
        s_idx = int(s)
        mask = (ys == s)
        if np.any(mask):
            sigma[mask] = base_kon(theta_basal_all[s_idx], inter_ref, yp[mask])
    return sigma


def objective(X, weights_samples, ys, ypr, yp, ypm, yk, ks, G, g, n_networks, n_samples,
              theta_ref, ref_network, l_pen, proba, weight_prev, loss, final,
              constrain_basal_uniform=0.0, basal_free_mask=None):
    """
    Loss function for a single gene — joint over all samples.

    theta shape: (G + n_samples, n_networks)
      rows 0..G-1      : interaction weights
      rows G..G+ns-1   : per-sample basal values

    constrain_basal_uniform : float >= 0
        Penalty strength that pushes free-sample basals toward their mean.
        Samples whose basal is pinned by a KO/OV prior are excluded (basal_free_mask).
    basal_free_mask : (n_samples,) bool array or None
        True for samples that are NOT pinned by basal_ref for this gene.
    """
    theta = X.reshape(G + n_samples, n_networks)
    theta_inter = theta[:G]           # (G, n_networks)
    theta_basal = theta[G:]           # (n_samples, n_networks)

    Q = 0
    us = np.unique(ys.ravel())
    ns_unique: int = len(us)

    sigma = _compute_sigma_per_sample(theta_inter, theta_basal, ref_network, yp, ys, us, n_networks)

    for cnt, s in enumerate(us):
        weight_s = np.sum(weights_samples) / weights_samples[cnt]
        mask = (ys == s)
        if proba:
            Q += (1 - weight_prev) * main_loss(sigma[mask], ypr[mask], 1, loss) * weight_s / ns_unique
        else:
            Q += (1 - weight_prev) * main_loss(np.sum(ks * sigma[mask], axis=-1), yk[mask], 1, loss) * weight_s / ns_unique

    if weight_prev:
        sigma_mod = _compute_sigma_per_sample(theta_inter, theta_basal, ref_network, ypm, ys, us, n_networks)
        for cnt, s in enumerate(us):
            weight_s = np.sum(weights_samples) / weights_samples[cnt]
            mask = (ys == s)
            if proba:
                Q += weight_prev * main_loss(sigma_mod[mask], ypr[mask], 1, loss) * weight_s / ns_unique
            else:
                Q += weight_prev * main_loss(np.sum(ks * sigma_mod[mask], axis=-1), yk[mask], 1, loss) * weight_s / ns_unique

    if not final:
        Q += theta_penalization((theta_inter - theta_ref[:G]) * (ref_network > 0), l_pen)
    else:
        Q += final_theta_penalization((theta_inter - theta_ref[:G]) * (ref_network > 0), l_pen)

    # Uniformity penalty: push free-sample basals toward their common mean
    if constrain_basal_uniform > 0.0 and basal_free_mask is not None:
        free_idx = np.where(basal_free_mask)[0]
        if len(free_idx) >= 2:
            theta_free = theta_basal[free_idx, :]          # (n_free, n_networks)
            mean_free  = theta_free.mean(axis=0)            # (n_networks,)
            Q += constrain_basal_uniform * np.sum((theta_free - mean_free) ** 2)

    return Q


def grad_theta(X, weights_samples, ys, ypr, yp, ypm, yk, ks, G, g, n_networks, n_samples,
               theta_ref, ref_network, l_pen, proba, weight_prev, loss, final,
               constrain_basal_uniform=0.0, basal_free_mask=None):
    """Gradient of objective w.r.t. X = theta.ravel()."""
    theta = X.reshape(G + n_samples, n_networks)
    theta_inter = theta[:G]
    theta_basal = theta[G:]
    dq = np.zeros_like(theta)
    us = np.unique(ys.ravel())
    ns_unique: int = len(us)

    sigma = _compute_sigma_per_sample(theta_inter, theta_basal, ref_network, yp, ys, us, n_networks)

    for n in range(n_networks):
        for cnt, s in enumerate(us):
            s_idx = int(s)
            weight_s = np.sum(weights_samples) / weights_samples[cnt]
            mask = (ys == s)
            # Softmax Jacobian column n for cells of sample s
            grad_sigma_s = -sigma[mask] * sigma[mask, n + 1, np.newaxis]
            grad_sigma_s[:, n + 1] += sigma[mask, n + 1]
            if proba:
                tmp = grad_main_loss(sigma[mask], ypr[mask], 1, loss) * grad_sigma_s
                sum_tmp = np.sum(tmp, axis=-1)  # (N_s,)
                dq[:G, n] += (1 - weight_prev) * ref_network[:, n] * (yp[mask].T @ sum_tmp[:, np.newaxis]).ravel() * weight_s / ns_unique
                dq[G + s_idx, n] += (1 - weight_prev) * np.sum(sum_tmp) * weight_s / ns_unique
            else:
                tmp = grad_main_loss(np.sum(ks * sigma[mask], axis=-1), yk[mask], 1, loss) * np.sum(ks * grad_sigma_s, axis=-1)
                res = (yp[mask].T @ tmp[:, None]).reshape(-1)
                dq[:G, n] += (1 - weight_prev) * ref_network[:, n] * res * weight_s / ns_unique
                dq[G + s_idx, n] += (1 - weight_prev) * np.sum(tmp) * weight_s / ns_unique

    if weight_prev:
        sigma_mod = _compute_sigma_per_sample(theta_inter, theta_basal, ref_network, ypm, ys, us, n_networks)
        for n in range(n_networks):
            for cnt, s in enumerate(us):
                s_idx = int(s)
                weight_s = np.sum(weights_samples) / weights_samples[cnt]
                mask = (ys == s)
                grad_sigma_mod_s = -sigma_mod[mask] * sigma_mod[mask, n + 1, np.newaxis]
                grad_sigma_mod_s[:, n + 1] += sigma_mod[mask, n + 1]
                if proba:
                    tmp_mod = grad_main_loss(sigma_mod[mask], ypr[mask], 1, loss) * grad_sigma_mod_s
                    sum_tmp_mod = np.sum(tmp_mod, axis=-1)
                    dq[:G, n] += weight_prev * ref_network[:, n] * (ypm[mask].T @ sum_tmp_mod[:, np.newaxis]).ravel() * weight_s / ns_unique
                    dq[G + s_idx, n] += weight_prev * np.sum(sum_tmp_mod) * weight_s / ns_unique
                else:
                    tmp_mod = grad_main_loss(np.sum(ks * sigma_mod[mask], axis=-1), yk[mask], 1, loss) * np.sum(ks * grad_sigma_mod_s, axis=-1)
                    res_mod = (ypm[mask].T @ tmp_mod[:, None]).reshape(-1)
                    dq[:G, n] += weight_prev * ref_network[:, n] * res_mod * weight_s / ns_unique
                    dq[G + s_idx, n] += weight_prev * np.sum(tmp_mod) * weight_s / ns_unique

    if not final:
        dq[:G] += grad_theta_penalization(theta_inter - theta_ref[:G], l_pen) * (ref_network > 0)
    else:
        dq[:G] += grad_final_theta_penalization(theta_inter - theta_ref[:G], l_pen) * (ref_network > 0)

    # Gradient of uniformity penalty: 2λ(θ_basal[s] − mean) for free samples
    if constrain_basal_uniform > 0.0 and basal_free_mask is not None:
        free_idx = np.where(basal_free_mask)[0]
        if len(free_idx) >= 2:
            theta_free = theta_basal[free_idx, :]       # (n_free, n_networks)
            mean_free  = theta_free.mean(axis=0)         # (n_networks,)
            dq[G + free_idx, :] += 2.0 * constrain_basal_uniform * (theta_free - mean_free)

    return dq.ravel()


def objective_refinement(X, correc_ref, inter, basal, weights_samples, ys, ypr, yp, ypm, yk, ks,
                          diag, G, g, n_networks, n_samples, l_pen, proba, weight_prev, loss, final):
    """
    Loss for the refinement step with per-sample basal.

    correc shape: (G + n_samples, n_networks)
      rows 0..G-1    : multiplicative corrections for inter
      rows G..G+ns-1 : multiplicative corrections for per-sample basal
    basal : (n_samples, n_networks)
    """
    correc = X.reshape(G + n_samples, n_networks)
    theta_inter = inter.copy() * correc[:G, :]          # (G, n_networks)
    theta_basal = basal.copy() * correc[G:, :]           # (n_samples, n_networks)

    Q = 0
    us = np.unique(ys.ravel())
    ns_unique: int = len(us)

    inter_with_diag = theta_inter + diag
    sigma = np.empty((yp.shape[0], n_networks + 1))
    for s in us:
        s_idx = int(s)
        mask = (ys == s)
        if np.any(mask):
            sigma[mask] = base_kon(theta_basal[s_idx], inter_with_diag, yp[mask])

    for cnt, s in enumerate(us):
        weight_s = np.sum(weights_samples) / weights_samples[cnt]
        mask = (ys == s)
        if proba:
            Q += (1 - weight_prev) * main_loss(sigma[mask], ypr[mask], 1, loss) * weight_s / ns_unique
        else:
            Q += (1 - weight_prev) * main_loss(np.sum(ks * sigma[mask], axis=-1), yk[mask], 1, loss) * weight_s / ns_unique

    if weight_prev:
        sigma_mod = np.empty((ypm.shape[0], n_networks + 1))
        for s in us:
            s_idx = int(s)
            mask = (ys == s)
            if np.any(mask):
                sigma_mod[mask] = base_kon(theta_basal[s_idx], inter_with_diag, ypm[mask])
        for cnt, s in enumerate(us):
            weight_s = np.sum(weights_samples) / weights_samples[cnt]
            mask = (ys == s)
            if proba:
                Q += weight_prev * main_loss(sigma_mod[mask], ypr[mask], 1, loss) * weight_s / ns_unique
            else:
                Q += weight_prev * main_loss(np.sum(ks * sigma_mod[mask], axis=-1), yk[mask], 1, loss) * weight_s / ns_unique

    if not final:
        Q += theta_penalization(correc[:g] - correc_ref, l_pen)
        Q += theta_penalization(correc[g + 1:G] - correc_ref, l_pen)
    else:
        Q += final_theta_penalization(correc[:g] - correc_ref, l_pen)
        Q += final_theta_penalization(correc[g + 1:G] - correc_ref, l_pen)
    return Q


def grad_correc(X, correc_ref, inter, basal, weights_samples, ys, ypr, yp, ypm, yk, ks,
                diag, G, g, n_networks, n_samples, l_pen, proba, weight_prev, loss, final):
    """Gradient of objective_refinement w.r.t. correction factors X."""
    correc = X.reshape(G + n_samples, n_networks)
    theta_inter = inter.copy() * correc[:G, :]
    theta_basal = basal.copy() * correc[G:, :]  # (n_samples, n_networks)
    inter_with_diag = theta_inter + diag

    dq = np.zeros_like(correc)
    us = np.unique(ys.ravel())
    ns_unique: int = len(us)

    sigma = np.empty((yp.shape[0], n_networks + 1))
    for s in us:
        s_idx = int(s)
        mask = (ys == s)
        if np.any(mask):
            sigma[mask] = base_kon(theta_basal[s_idx], inter_with_diag, yp[mask])

    for n in range(n_networks):
        for cnt, s in enumerate(us):
            s_idx = int(s)
            weight_s = np.sum(weights_samples) / weights_samples[cnt]
            mask = (ys == s)
            grad_sigma_s = -sigma[mask] * sigma[mask, n + 1, np.newaxis]
            grad_sigma_s[:, n + 1] += sigma[mask, n + 1]
            if proba:
                tmp = grad_main_loss(sigma[mask], ypr[mask], 1, loss) * grad_sigma_s
                sum_tmp = np.sum(tmp, axis=-1)
                dq[:G, n] += (1 - weight_prev) * inter[:, n] * (yp[mask].T @ sum_tmp[:, np.newaxis]).ravel() * weight_s / ns_unique
                dq[G + s_idx, n] += (1 - weight_prev) * basal[s_idx, n] * np.sum(sum_tmp) * weight_s / ns_unique
            else:
                tmp = grad_main_loss(np.sum(ks * sigma[mask], axis=-1), yk[mask], 1, loss) * np.sum(ks * grad_sigma_s, axis=-1)
                res = (yp[mask].T @ tmp[:, None]).reshape(-1)
                dq[:G, n] += (1 - weight_prev) * inter[:, n] * res * weight_s / ns_unique
                dq[G + s_idx, n] += (1 - weight_prev) * basal[s_idx, n] * np.sum(tmp) * weight_s / ns_unique

    if weight_prev:
        sigma_mod = np.empty((ypm.shape[0], n_networks + 1))
        for s in us:
            s_idx = int(s)
            mask = (ys == s)
            if np.any(mask):
                sigma_mod[mask] = base_kon(theta_basal[s_idx], inter_with_diag, ypm[mask])
        for n in range(n_networks):
            for cnt, s in enumerate(us):
                s_idx = int(s)
                weight_s = np.sum(weights_samples) / weights_samples[cnt]
                mask = (ys == s)
                grad_sigma_mod_s = -sigma_mod[mask] * sigma_mod[mask, n + 1, np.newaxis]
                grad_sigma_mod_s[:, n + 1] += sigma_mod[mask, n + 1]
                if proba:
                    tmp_mod = grad_main_loss(sigma_mod[mask], ypr[mask], 1, loss) * grad_sigma_mod_s
                    sum_tmp_mod = np.sum(tmp_mod, axis=-1)
                    dq[:G, n] += weight_prev * inter[:, n] * (ypm[mask].T @ sum_tmp_mod[:, np.newaxis]).ravel() * weight_s / ns_unique
                    dq[G + s_idx, n] += weight_prev * basal[s_idx, n] * np.sum(sum_tmp_mod) * weight_s / ns_unique
                else:
                    tmp_mod = grad_main_loss(np.sum(ks * sigma_mod[mask], axis=-1), yk[mask], 1, loss) * np.sum(ks * grad_sigma_mod_s, axis=-1)
                    res_mod = (ypm[mask].T @ tmp_mod[:, None]).reshape(-1)
                    dq[:G, n] += weight_prev * inter[:, n] * res_mod * weight_s / ns_unique
                    dq[G + s_idx, n] += weight_prev * basal[s_idx, n] * np.sum(tmp_mod) * weight_s / ns_unique

    if not final:
        dq[:g] += grad_theta_penalization(correc[:g] - correc_ref, l_pen)
        dq[g + 1:G] += grad_theta_penalization(correc[g + 1:G] - correc_ref, l_pen)
    else:
        dq[:g] += grad_final_theta_penalization(correc[:g] - correc_ref, l_pen)
        dq[g + 1:G] += grad_final_theta_penalization(correc[g + 1:G] - correc_ref, l_pen)
    return dq.ravel()


def core_inference(y_samples, y_proba, y_prot, y_prot_mod, y_kon, theta_init, theta_ref,
                   ref_network, ks, G, g, n_networks, n_samples, proba,
                   l_pen, weight_prev=.5, loss='CE', final=0,
                   constrain_basal_uniform=0.0, basal_free_mask=None, G_tol=None):
    """
    Joint L-BFGS-B optimisation of interactions + per-sample basals for one gene.

    theta_init shape: (G + n_samples, n_networks)
      rows 0..G-1    : interaction weights
      rows G..G+ns-1 : per-sample basal values
    Returns theta_final of same shape.
    """
    weights_samples = [np.sum(y_samples == s) ** alpha for s in np.unique(y_samples)]
    y_prot_mod = y_prot.copy()

    theta_init_ = np.ascontiguousarray(theta_init, dtype=float)    # (G + n_samples, n_networks)
    theta_ref_  = np.ascontiguousarray(theta_ref,  dtype=float)
    ref_network_ = np.ascontiguousarray(ref_network, dtype=float)  # (G, n_networks)

    X_flat = theta_init_.ravel(order='C')
    theta_ref_flat = theta_ref_.ravel(order='C')
    ref_network_flat = ref_network_.ravel(order='C')

    bounds = [(None, None)] * len(X_flat)
    n_inter_flat = G * n_networks
    # Only apply sign/reference constraints to interaction rows (first G rows of theta)
    for idx in range(n_inter_flat):
        if theta_ref_flat[idx] != 0.0:
            v = theta_ref_flat[idx]
            bounds[idx] = (v, None) if v > 0 else (None, v)
        elif ref_network_flat[idx] != 0.0:
            v = ref_network_flat[idx]
            if v < 1: bounds[idx] = (None, 0)
            elif v > 1: bounds[idx] = (0, None)
            else: bounds[idx] = (None, None)
    # Basal rows (G..G+n_samples-1): unconstrained — leave as (None, None)

    loss_fn = partial(objective, weights_samples=weights_samples, ys=y_samples,
                      ypr=y_proba, yp=y_prot, ypm=y_prot_mod, yk=y_kon,
                      ks=ks, G=G, g=g, n_networks=n_networks, n_samples=n_samples,
                      theta_ref=theta_ref, ref_network=np.abs(ref_network),
                      l_pen=l_pen, proba=proba, weight_prev=weight_prev, loss=loss, final=final,
                      constrain_basal_uniform=constrain_basal_uniform, basal_free_mask=basal_free_mask)

    grad_fn = partial(grad_theta, weights_samples=weights_samples, ys=y_samples,
                      ypr=y_proba, yp=y_prot, ypm=y_prot_mod, yk=y_kon,
                      ks=ks, G=G, g=g, n_networks=n_networks, n_samples=n_samples,
                      theta_ref=theta_ref, ref_network=np.abs(ref_network),
                      l_pen=l_pen, proba=proba, weight_prev=weight_prev, loss=loss, final=final,
                      constrain_basal_uniform=constrain_basal_uniform, basal_free_mask=basal_free_mask)

    _G_tol = G_tol if G_tol is not None else G
    res = minimize(loss_fn, X_flat, jac=grad_fn, method="L-BFGS-B", bounds=bounds, tol=seuil / _G_tol)
    if not res.success:
        logger.error('Minimization failed for inference: %s', res.message)

    if check_gradient:
        error = check_grad(loss_fn, grad_fn, res.x)
        if error > .05:
            logger.debug("Gradient theta inference check error for gene %s: %s", g, error)
            res = minimize(loss_fn, X_flat, bounds=bounds, method="L-BFGS-B", tol=seuil / _G_tol)

    theta_final = res.x.reshape(G + n_samples, n_networks)
    theta_final[:G] *= np.abs(ref_network)   # apply ref_network mask to interactions only

    return theta_final


def refine_inference(y_samples, y_proba, y_prot, y_prot_mod, y_kon, inter, basal, theta_ref,
                     ks, G, g, n_networks, n_samples, proba,
                     l_pen, weight_prev=.5, loss='CE', correc_ref=0, final=0, G_tol=None):
    """
    Refinement step: optimise multiplicative correction factors over inter and per-sample basal.

    basal : (n_samples, n_networks)
    Returns updated (inter, basal).
    """
    correc = np.ones((G + n_samples, n_networks))
    diag = np.zeros((G, n_networks))
    if g < G:   # g == G is the sentinel meaning "no self-regulation in active set"
        diag[g, :] = inter[g, :]
    inter = inter.copy()
    inter -= diag
    basal = basal.copy()  # (n_samples, n_networks)

    weights_samples = [np.sum(y_samples == s) ** alpha for s in np.unique(y_samples)]

    correc_init_ = np.ascontiguousarray(correc, dtype=float)
    theta_ref_   = np.ascontiguousarray(theta_ref, dtype=float)
    X_flat = correc_init_.ravel(order='C')
    theta_ref_flat = theta_ref_.ravel(order='C')

    bounds = [(0, None)] * len(X_flat)
    n_inter_flat = G * n_networks
    if not final:
        # Tighten bounds for non-zero reference interactions only
        for idx in range(n_inter_flat):
            if theta_ref_flat[idx] != 0.0:
                bounds[idx] = (1, None)
    # Basal correction rows start at n_inter_flat; keep (0, None)

    loss_fn = partial(objective_refinement,
                      correc_ref=correc_ref, inter=inter, basal=basal,
                      weights_samples=weights_samples, ys=y_samples,
                      ypr=y_proba, yp=y_prot, ypm=y_prot_mod, yk=y_kon,
                      ks=ks, diag=diag, G=G, g=g, n_networks=n_networks, n_samples=n_samples,
                      proba=proba, l_pen=l_pen, weight_prev=weight_prev, loss=loss, final=final)
    grad_fn = partial(grad_correc,
                      correc_ref=correc_ref, inter=inter, basal=basal,
                      weights_samples=weights_samples, ys=y_samples,
                      ypr=y_proba, yp=y_prot, ypm=y_prot_mod, yk=y_kon,
                      ks=ks, diag=diag, G=G, g=g, n_networks=n_networks, n_samples=n_samples,
                      proba=proba, l_pen=l_pen, weight_prev=weight_prev, loss=loss, final=final)

    _G_tol = G_tol if G_tol is not None else G
    res = minimize(loss_fn, X_flat, jac=grad_fn, method="L-BFGS-B", bounds=bounds, tol=seuil / _G_tol)
    if not res.success:
        logger.error('Minimization failed for refining: %s', res.message)

    if check_gradient:
        error = check_grad(loss_fn, grad_fn, res.x)
        if error > .05:
            logger.debug("Gradient theta refining check error for gene %s: %s", g, error)
            res = minimize(loss_fn, X_flat, bounds=bounds, method="L-BFGS-B", tol=seuil / _G_tol)

    correc = res.x.reshape(G + n_samples, n_networks)
    inter[:, :] *= correc[:G, :]
    basal *= correc[G:, :]     # (n_samples, n_networks) *= (n_samples, n_networks)
    inter += diag

    return inter, basal


def main_loop_inference(g, y_samples, y_proba, y_prot, y_prot_mod, y_kon, theta_init, theta_ref,
                         ks, G, n_networks, n_samples, proba, l_gen, scale,
                         inter_tmp, basal_tmp, inter, basal, ref_network,
                         weight_prev=.5, loss='CE', final=0,
                         constrain_basal_uniform=0.0, basal_free_mask=None, G_tol=None):
    """
    Joint inference (inter + per-sample basal) for a single gene g.

    theta_init : (G + n_samples, n_networks)
    basal      : (n_samples, n_networks)  — will be updated in-place (copy returned)
    inter      : (G, n_networks)          — will be updated in-place (copy returned)
    basal_free_mask : (n_samples,) bool — True for samples NOT pinned by basal_ref for gene g
    Returns (basal, inter, basal_tmp, inter_tmp).
    """
    n_networks_tmp: int = int(1 + np.argmax(ks[1:]))

    l_pen1 = l_gen * np.size(y_prot, 0) / (n_networks_tmp * scale * (1 + G**(1/2)))
    theta = core_inference(
        y_samples, y_proba[:, :n_networks_tmp + 1], y_prot, y_prot_mod, y_kon,
        theta_init[:, :n_networks_tmp],
        theta_ref[:, :n_networks_tmp],
        ref_network[:, :n_networks_tmp],
        ks[:n_networks_tmp + 1], G, g, n_networks_tmp, n_samples, proba,
        l_pen1, weight_prev=weight_prev * (1 - final), loss=loss, final=final,
        constrain_basal_uniform=constrain_basal_uniform, basal_free_mask=basal_free_mask,
        G_tol=G_tol,
    )
    # theta shape: (G + n_samples, n_networks_tmp)
    inter[:, :n_networks_tmp]    = theta[:G, :]
    basal[:, :n_networks_tmp]    = theta[G:G + n_samples, :]
    inter_tmp[:, :n_networks_tmp] = theta[:G, :]
    basal_tmp[:, :n_networks_tmp] = theta[G:G + n_samples, :]

    # Refinement step
    l_pen2 = l_gen / (n_networks_tmp * (1 + np.log(G)))
    inter[:, :n_networks_tmp], basal[:, :n_networks_tmp] = refine_inference(
        y_samples, y_proba[:, :n_networks_tmp + 1], y_prot, y_prot_mod, y_kon,
        inter, basal[:, :n_networks_tmp],
        theta_ref[:, :n_networks_tmp],
        ks[:n_networks_tmp + 1], G, g, n_networks_tmp, n_samples, proba,
        l_pen2, weight_prev=weight_prev * (1 - final), loss=loss,
        correc_ref=final, final=final, G_tol=G_tol,
    )

    if n_networks_tmp < n_networks:
        basal[:, n_networks_tmp:] = -100
        basal_tmp[:, n_networks_tmp:] = -100

    return basal, inter, basal_tmp, inter_tmp


def inference_network(y_samples, y_kon, y_proba, y_prot, y_prot_mod, ks, n_stimuli=1, proba=0,
                      ref_network=None,
                      basal_init=None, inter_init=None,
                      basal_ref=None, inter_ref=None,
                      scale=100, weight_prev=.5, loss='CE', final=0,
                      samples_id=None,
                      constrain_basal_uniform=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Joint network inference: interactions shared across samples, basal per sample.

    Returns (basal, inter, basal_tmp, inter_tmp).
    basal shape: (n_samples, G, n_networks)

    constrain_basal_uniform : float >= 0
        When > 0, adds a penalty that pushes per-sample basals toward their common
        mean for each gene. Samples whose basal is pinned by a non-zero basal_ref
        (KO/OV priors) are excluded from the penalty for that gene.
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        Parallel = None
        delayed = None
        logging.getLogger(__name__).warning("joblib not available; parallel loops will run sequentially")

    G: int = np.size(y_prot, 1)
    n_networks: int = np.size(ks, 1) - 1
    unique_samples = np.asarray(samples_id) if samples_id is not None else np.unique(y_samples)
    n_samples: int = len(unique_samples)

    # ── Build theta_init (G + n_samples, G, n_networks) ─────────────────────
    # rows 0..G-1: interactions per target gene; rows G..G+ns-1: per-sample basal per target gene
    theta_init_mat = np.zeros((G + n_samples, G, n_networks))
    if inter_init is not None:
        theta_init_mat[:G, :, :] = np.asarray(inter_init, dtype=float)

    if basal_init is not None:
        bi = np.asarray(basal_init, dtype=float)
        if bi.ndim == 3:                    # (n_samples, G, n_networks)
            for s_idx in range(min(n_samples, bi.shape[0])):
                theta_init_mat[G + s_idx, :, :] = bi[s_idx]
        else:                               # (G, n_networks) — broadcast
            for s_idx in range(n_samples):
                theta_init_mat[G + s_idx, :, :] = bi

    # ── Build theta_ref (G + n_samples, G, n_networks) ──────────────────────
    theta_ref_mat = np.zeros((G + n_samples, G, n_networks))
    if inter_ref is not None:
        theta_ref_mat[:G, :, :] = np.asarray(inter_ref, dtype=float)

    if basal_ref is not None:
        br = np.asarray(basal_ref, dtype=float)
        if br.ndim == 3:
            for s_idx in range(min(n_samples, br.shape[0])):
                theta_ref_mat[G + s_idx, :, :] = br[s_idx]
        else:
            for s_idx in range(n_samples):
                theta_ref_mat[G + s_idx, :, :] = br

    if ref_network is None:
        ref_network = np.ones((G, G, n_networks))

    # ── Compute free-sample mask from basal_ref ─────────────────────────────
    # free_mask[s, g] = True  iff sample s is NOT pinned for gene g
    # (i.e. basal_ref[s, g, :] is all-zero → no KO/OV prior)
    if basal_ref is not None:
        br_arr = np.asarray(basal_ref, dtype=float)
        if br_arr.ndim == 3:                         # (n_samples, G, n_networks)
            free_mask_2d = ~np.any(br_arr != 0.0, axis=-1)  # (n_samples, G)
        else:                                        # 2-D: no per-sample pinning
            free_mask_2d = np.ones((n_samples, G), dtype=bool)
    else:
        free_mask_2d = np.ones((n_samples, G), dtype=bool)

    l_gen: int = (1 + proba)
    inter_tmp = np.zeros((G, G, n_networks))
    basal_tmp  = np.zeros((n_samples, G, n_networks))
    inter      = np.zeros((G, G, n_networks))
    basal      = np.zeros((n_samples, G, n_networks))

    def run_main_loop_for_gene(g_tgt):
        # Find active (non-zero) regulators for this target gene
        active_src = np.where(np.any(ref_network[:, g_tgt, :] != 0, axis=-1))[0]
        K_g = len(active_src)

        if K_g == 0 or K_g == G:
            # Dense path: no sub-selection, identical to previous behaviour.
            # K_g == 0 case: no active regulators, so the dense path is used to
            # safely optimize the basal only (interactions stay zero via ref_network masking).
            return main_loop_inference(
                g_tgt, y_samples, y_proba[:, g_tgt], y_prot,
                y_prot_mod[g_tgt, :, :], y_kon[:, g_tgt],
                theta_init_mat[:, g_tgt, :], theta_ref_mat[:, g_tgt, :],
                ks[g_tgt], G, n_networks, n_samples, proba, l_gen, scale,
                inter_tmp[:, g_tgt, :], basal_tmp[:, g_tgt, :],
                inter[:, g_tgt, :], basal[:, g_tgt, :],
                ref_network[:, g_tgt, :],
                weight_prev=weight_prev, loss=loss, final=final,
                constrain_basal_uniform=constrain_basal_uniform,
                basal_free_mask=free_mask_2d[:, g_tgt],
            )

        # Sparse path: sub-select to K_g active regulators
        yp_sub  = y_prot[:, active_src]                    # (N_cells, K_g)
        ypm_sub = y_prot_mod[g_tgt, :, :][:, active_src]  # (N_cells, K_g)
        rn_sub  = ref_network[active_src, g_tgt, :]        # (K_g, n_networks)

        # Sub-select theta rows: active_src rows + per-sample basal rows (G..G+n_samples-1)
        ti_g = theta_init_mat[:, g_tgt, :]
        theta_init_sub = np.concatenate([ti_g[active_src], ti_g[G:]], axis=0)  # (K_g + n_samples, n_networks)
        tr_g = theta_ref_mat[:, g_tgt, :]
        theta_ref_sub  = np.concatenate([tr_g[active_src], tr_g[G:]], axis=0)  # (K_g + n_samples, n_networks)

        # Position of g_tgt in active_src (sentinel K_g = "no self-regulation")
        g_sub_matches = np.where(active_src == g_tgt)[0]
        g_sub = int(g_sub_matches[0]) if len(g_sub_matches) > 0 else K_g

        # Local arrays for the reduced sub-problem
        inter_sub_loc     = np.zeros((K_g, n_networks))
        basal_sub_loc     = np.zeros((n_samples, n_networks))
        inter_tmp_sub_loc = np.zeros((K_g, n_networks))
        basal_tmp_sub_loc = np.zeros((n_samples, n_networks))

        basal_r, inter_sub_r, basal_tmp_r, inter_tmp_sub_r = main_loop_inference(
            g_sub, y_samples, y_proba[:, g_tgt], yp_sub, ypm_sub, y_kon[:, g_tgt],
            theta_init_sub, theta_ref_sub,
            ks[g_tgt], K_g, n_networks, n_samples, proba, l_gen, scale,
            inter_tmp_sub_loc, basal_tmp_sub_loc, inter_sub_loc, basal_sub_loc, rn_sub,
            weight_prev=weight_prev, loss=loss, final=final,
            constrain_basal_uniform=constrain_basal_uniform,
            basal_free_mask=free_mask_2d[:, g_tgt],
            G_tol=K_g,
        )

        # Expand interaction results back to full G-dimensional space
        inter_full     = np.zeros((G, n_networks))
        inter_tmp_full = np.zeros((G, n_networks))
        inter_full[active_src]     = inter_sub_r
        inter_tmp_full[active_src] = inter_tmp_sub_r

        return basal_r, inter_full, basal_tmp_r, inter_tmp_full

    if Parallel is not None:
        results = Parallel(n_jobs=-1)(
            delayed(run_main_loop_for_gene)(g) for g in range(n_stimuli, G)
        )
    else:
        results = [run_main_loop_for_gene(g) for g in range(n_stimuli, G)]

    for idx, g in enumerate(range(n_stimuli, G)):
        basal[:, g, :], inter[:, g, :], basal_tmp[:, g, :], inter_tmp[:, g, :] = results[idx]

    return basal, inter, basal_tmp, inter_tmp
