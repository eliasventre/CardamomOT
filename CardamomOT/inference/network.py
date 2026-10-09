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
max_inter = 100  # Capacity: bound on each interaction (logit units), independent of the number of genes


def _basal_bound(n_regulators):
    # Basal must be able to offset all regulators at full capacity (e.g. KO/OV samples)
    return max_inter * (1 + n_regulators)


def _lbfgs_options(n_inter):
    # Relative loss decrease must resolve each of the n_inter interactions; gradient test is per coordinate
    return {'ftol': seuil / max(n_inter, 1), 'gtol': seuil}


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


def _kss(ks, s):
    """Amplitudes of the target gene for sample s: ks is (n_samples, n_modes) if per-sample, else (n_modes,)."""
    return ks[int(s)] if ks.ndim == 2 else ks


def _of_cond(arr, cond, s_idx):
    """Slice of the condition of sample s_idx when arr is per condition (C, ...), else arr."""
    return arr if cond is None else arr[cond[int(s_idx)]]


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
            Q += (1 - weight_prev) * main_loss(np.sum(_kss(ks, s) * sigma[mask], axis=-1), yk[mask], 1, loss) * weight_s / ns_unique

    if weight_prev:
        sigma_mod = _compute_sigma_per_sample(theta_inter, theta_basal, ref_network, ypm, ys, us, n_networks)
        for cnt, s in enumerate(us):
            weight_s = np.sum(weights_samples) / weights_samples[cnt]
            mask = (ys == s)
            if proba:
                Q += weight_prev * main_loss(sigma_mod[mask], ypr[mask], 1, loss) * weight_s / ns_unique
            else:
                Q += weight_prev * main_loss(np.sum(_kss(ks, s) * sigma_mod[mask], axis=-1), yk[mask], 1, loss) * weight_s / ns_unique

    d_inter = theta_inter - theta_ref[:G]
    if not final:
        Q += theta_penalization(d_inter * (ref_network > 0), l_pen)
    else:
        # L2 on sqrt(w) * theta so that, like L1, it costs 1/w on the effective interaction
        Q += r_elasticnet * l2_penalization(np.sqrt(ref_network) * d_inter, l_pen) \
            + (1 - r_elasticnet) * smoothed_l1_penalization(d_inter * (ref_network > 0), l_pen)

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
                tmp = grad_main_loss(np.sum(_kss(ks, s) * sigma[mask], axis=-1), yk[mask], 1, loss) * np.sum(_kss(ks, s) * grad_sigma_s, axis=-1)
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
                    tmp_mod = grad_main_loss(np.sum(_kss(ks, s) * sigma_mod[mask], axis=-1), yk[mask], 1, loss) * np.sum(_kss(ks, s) * grad_sigma_mod_s, axis=-1)
                    res_mod = (ypm[mask].T @ tmp_mod[:, None]).reshape(-1)
                    dq[:G, n] += weight_prev * ref_network[:, n] * res_mod * weight_s / ns_unique
                    dq[G + s_idx, n] += weight_prev * np.sum(tmp_mod) * weight_s / ns_unique

    d_inter = theta_inter - theta_ref[:G]
    if not final:
        dq[:G] += grad_theta_penalization(d_inter, l_pen) * (ref_network > 0)
    else:
        dq[:G] += r_elasticnet * grad_l2_penalization(d_inter, l_pen) * ref_network \
            + (1 - r_elasticnet) * grad_smoothed_l1_penalization(d_inter, l_pen) * (ref_network > 0)

    # Gradient of uniformity penalty: 2λ(θ_basal[s] − mean) for free samples
    if constrain_basal_uniform > 0.0 and basal_free_mask is not None:
        free_idx = np.where(basal_free_mask)[0]
        if len(free_idx) >= 2:
            theta_free = theta_basal[free_idx, :]       # (n_free, n_networks)
            mean_free  = theta_free.mean(axis=0)         # (n_networks,)
            dq[G + free_idx, :] += 2.0 * constrain_basal_uniform * (theta_free - mean_free)

    return dq.ravel()


def objective_refinement(X, correc_ref, inter, basal, weights_samples, ys, ypr, yp, ypm, yk, ks,
                          diag, G, g, n_networks, n_samples, l_pen, proba, weight_prev, loss, final, cond=None):
    """
    Loss for the refinement step with per-sample basal.

    correc shape: (G + n_samples, n_networks)
      rows 0..G-1    : multiplicative corrections for inter
      rows G..G+ns-1 : multiplicative corrections for per-sample basal
    basal : (n_samples, n_networks)
    inter, diag : (G, n_networks), or (C, G, n_networks) with cond[s] = condition of sample s
    (one correction per edge, shared by the conditions)
    """
    correc = X.reshape(G + n_samples, n_networks)
    theta_inter = inter.copy() * correc[:G, :]          # (G, n_networks) or (C, G, n_networks)
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
            sigma[mask] = base_kon(theta_basal[s_idx], _of_cond(inter_with_diag, cond, s_idx), yp[mask])

    for cnt, s in enumerate(us):
        weight_s = np.sum(weights_samples) / weights_samples[cnt]
        mask = (ys == s)
        if proba:
            Q += (1 - weight_prev) * main_loss(sigma[mask], ypr[mask], 1, loss) * weight_s / ns_unique
        else:
            Q += (1 - weight_prev) * main_loss(np.sum(_kss(ks, s) * sigma[mask], axis=-1), yk[mask], 1, loss) * weight_s / ns_unique

    if weight_prev:
        sigma_mod = np.empty((ypm.shape[0], n_networks + 1))
        for s in us:
            s_idx = int(s)
            mask = (ys == s)
            if np.any(mask):
                sigma_mod[mask] = base_kon(theta_basal[s_idx], _of_cond(inter_with_diag, cond, s_idx), ypm[mask])
        for cnt, s in enumerate(us):
            weight_s = np.sum(weights_samples) / weights_samples[cnt]
            mask = (ys == s)
            if proba:
                Q += weight_prev * main_loss(sigma_mod[mask], ypr[mask], 1, loss) * weight_s / ns_unique
            else:
                Q += weight_prev * main_loss(np.sum(_kss(ks, s) * sigma_mod[mask], axis=-1), yk[mask], 1, loss) * weight_s / ns_unique

    # l_pen: (G, n_networks) per-edge penalty (1/w times the base one, see refine_inference)
    if not final:
        Q += theta_penalization(correc[:g] - correc_ref, l_pen[:g])
        Q += theta_penalization(correc[g + 1:G] - correc_ref, l_pen[g + 1:G])
    else:
        Q += final_theta_penalization(correc[:g] - correc_ref, l_pen[:g])
        Q += final_theta_penalization(correc[g + 1:G] - correc_ref, l_pen[g + 1:G])
    return Q


def grad_correc(X, correc_ref, inter, basal, weights_samples, ys, ypr, yp, ypm, yk, ks,
                diag, G, g, n_networks, n_samples, l_pen, proba, weight_prev, loss, final, cond=None):
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
            sigma[mask] = base_kon(theta_basal[s_idx], _of_cond(inter_with_diag, cond, s_idx), yp[mask])

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
                dq[:G, n] += (1 - weight_prev) * _of_cond(inter, cond, s_idx)[:, n] * (yp[mask].T @ sum_tmp[:, np.newaxis]).ravel() * weight_s / ns_unique
                dq[G + s_idx, n] += (1 - weight_prev) * basal[s_idx, n] * np.sum(sum_tmp) * weight_s / ns_unique
            else:
                tmp = grad_main_loss(np.sum(_kss(ks, s) * sigma[mask], axis=-1), yk[mask], 1, loss) * np.sum(_kss(ks, s) * grad_sigma_s, axis=-1)
                res = (yp[mask].T @ tmp[:, None]).reshape(-1)
                dq[:G, n] += (1 - weight_prev) * _of_cond(inter, cond, s_idx)[:, n] * res * weight_s / ns_unique
                dq[G + s_idx, n] += (1 - weight_prev) * basal[s_idx, n] * np.sum(tmp) * weight_s / ns_unique

    if weight_prev:
        sigma_mod = np.empty((ypm.shape[0], n_networks + 1))
        for s in us:
            s_idx = int(s)
            mask = (ys == s)
            if np.any(mask):
                sigma_mod[mask] = base_kon(theta_basal[s_idx], _of_cond(inter_with_diag, cond, s_idx), ypm[mask])
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
                    dq[:G, n] += weight_prev * _of_cond(inter, cond, s_idx)[:, n] * (ypm[mask].T @ sum_tmp_mod[:, np.newaxis]).ravel() * weight_s / ns_unique
                    dq[G + s_idx, n] += weight_prev * basal[s_idx, n] * np.sum(sum_tmp_mod) * weight_s / ns_unique
                else:
                    tmp_mod = grad_main_loss(np.sum(_kss(ks, s) * sigma_mod[mask], axis=-1), yk[mask], 1, loss) * np.sum(_kss(ks, s) * grad_sigma_mod_s, axis=-1)
                    res_mod = (ypm[mask].T @ tmp_mod[:, None]).reshape(-1)
                    dq[:G, n] += weight_prev * _of_cond(inter, cond, s_idx)[:, n] * res_mod * weight_s / ns_unique
                    dq[G + s_idx, n] += weight_prev * basal[s_idx, n] * np.sum(tmp_mod) * weight_s / ns_unique

    if not final:
        dq[:g] += grad_theta_penalization(correc[:g] - correc_ref, l_pen[:g])
        dq[g + 1:G] += grad_theta_penalization(correc[g + 1:G] - correc_ref, l_pen[g + 1:G])
    else:
        dq[:g] += grad_final_theta_penalization(correc[:g] - correc_ref, l_pen[:g])
        dq[g + 1:G] += grad_final_theta_penalization(correc[g + 1:G] - correc_ref, l_pen[g + 1:G])
    return dq.ravel()


def core_inference(y_samples, y_proba, y_prot, y_prot_mod, y_kon, theta_init, theta_ref,
                   ref_network, ks, G, g, n_networks, n_samples, proba,
                   l_pen, weight_prev=.5, loss='CE', final=0,
                   constrain_basal_uniform=0.0, basal_free_mask=None,
                   hard_forcing_ref=False, ref_constraint_pct=0.1, seuil_zero_min_ref=1e-2):
    """
    Joint L-BFGS-B optimisation of interactions + per-sample basals for one gene.
    Effective interactions theta * w (w = prior_weight) are bounded by max_inter, basals by
    _basal_bound(G); theta_init / theta_ref are in theta units.

    theta_init shape: (G + n_samples, n_networks)
      rows 0..G-1    : interaction weights
      rows G..G+ns-1 : per-sample basal values
    Returns theta_final of same shape.
    """
    weights_samples = [np.sum(y_samples == s) ** alpha for s in np.unique(y_samples)]

    theta_init_ = np.ascontiguousarray(theta_init, dtype=float)    # (G + n_samples, n_networks)
    theta_ref_  = np.ascontiguousarray(theta_ref,  dtype=float)
    ref_network_ = np.ascontiguousarray(ref_network, dtype=float)  # (G, n_networks)

    X_flat = theta_init_.ravel(order='C')
    theta_ref_flat = theta_ref_.ravel(order='C')
    ref_network_flat = ref_network_.ravel(order='C')

    bounds = _interaction_bounds(theta_ref_flat, ref_network_flat, G, n_networks, final,
                                 hard_forcing_ref, ref_constraint_pct, seuil_zero_min_ref)
    bounds += [(-_basal_bound(G), _basal_bound(G))] * (n_samples * n_networks)

    loss_fn = partial(objective, weights_samples=weights_samples, ys=y_samples,
                      ypr=y_proba, yp=y_prot, ypm=y_prot_mod, yk=y_kon,
                      ks=ks, G=G, g=g, n_networks=n_networks, n_samples=n_samples,
                      theta_ref=theta_ref, ref_network=prior_weight(ref_network),
                      l_pen=l_pen, proba=proba, weight_prev=weight_prev, loss=loss, final=final,
                      constrain_basal_uniform=constrain_basal_uniform, basal_free_mask=basal_free_mask)

    grad_fn = partial(grad_theta, weights_samples=weights_samples, ys=y_samples,
                      ypr=y_proba, yp=y_prot, ypm=y_prot_mod, yk=y_kon,
                      ks=ks, G=G, g=g, n_networks=n_networks, n_samples=n_samples,
                      theta_ref=theta_ref, ref_network=prior_weight(ref_network),
                      l_pen=l_pen, proba=proba, weight_prev=weight_prev, loss=loss, final=final,
                      constrain_basal_uniform=constrain_basal_uniform, basal_free_mask=basal_free_mask)

    res = minimize(loss_fn, X_flat, jac=grad_fn, method="L-BFGS-B", bounds=bounds, options=_lbfgs_options(G))
    if not res.success:
        logger.error('Minimization failed for inference: %s', res.message)

    if check_gradient:
        error = check_grad(loss_fn, grad_fn, res.x)
        if error > .05:
            logger.debug("Gradient theta inference check error for gene %s: %s", g, error)
            res = minimize(loss_fn, X_flat, bounds=bounds, method="L-BFGS-B", options=_lbfgs_options(G))

    theta_final = res.x.reshape(G + n_samples, n_networks)
    theta_final[:G] *= prior_weight(ref_network)   # apply ref_network mask to interactions only

    return theta_final


def _interaction_bounds(theta_ref_flat, ref_network_flat, G, n_networks, final,
                        hard_forcing_ref, ref_constraint_pct, seuil_zero_min_ref):
    """L-BFGS-B bounds of the G * n_networks interaction variables (theta units, effective = theta * w)."""
    # Capacity applies to the effective interaction theta * w, not to theta
    w_flat = prior_weight(ref_network_flat)
    mb = np.where(w_flat > 0, max_inter / np.maximum(w_flat, EPS), max_inter)
    bounds = [(-b, b) for b in mb]
    if not final:
        r_plus = 1 + ref_constraint_pct / max(EPS, (1 - ref_constraint_pct))
        r_minus = 1 - ref_constraint_pct
        n_inter_flat = G * n_networks
        if hard_forcing_ref:
            for idx in range(n_inter_flat):
                v = theta_ref_flat[idx]
                max_bounds = mb[idx]
                if v * w_flat[idx] > seuil_zero_min_ref:
                    bounds[idx] = (r_minus * v, max(r_minus * v, min(r_plus * v, max_bounds)))
                elif v * w_flat[idx] < -seuil_zero_min_ref:
                    bounds[idx] = (min(r_minus * v, max(-max_bounds, r_plus * v)), r_minus * v)
                else:
                    zb = seuil_zero_min_ref / max(w_flat[idx], EPS)
                    bounds[idx] = (-zb, zb)
        else:
            # Only apply sign/reference constraints to interaction rows (first G rows of theta)
            for idx in range(n_inter_flat):
                max_bounds = mb[idx]
                if theta_ref_flat[idx] != 0.0:
                    v = theta_ref_flat[idx]
                    if v * w_flat[idx] > seuil_zero_min_ref:
                        bounds[idx] = (r_minus * v, max(r_minus * v, max_bounds))
                    elif v * w_flat[idx] < -seuil_zero_min_ref:
                        bounds[idx] =  (min(-max_bounds, r_minus * v), r_minus * v)
                elif ref_network_flat[idx] != 0.0:
                    v = ref_network_flat[idx]
                    if v < -1: bounds[idx] = (-max_bounds, -seuil_zero_min_ref)
                    elif v > 1: bounds[idx] = (seuil_zero_min_ref, max_bounds)
    else:
        # Signed priors (|v| > 1, weight 1) stay enforced in the final fit
        for idx in range(G * n_networks):
            v = ref_network_flat[idx]
            if v < -1: bounds[idx] = (-max_inter, -seuil_zero_min_ref)
            elif v > 1: bounds[idx] = (seuil_zero_min_ref, max_inter)

    return bounds


### NETWORK CONDITIONS: theta_c = theta_shared + delta_c (fused penalty on delta)

def _loss_grad_conditions(X, weights_samples, ys, ypr, yp, ypm, yk, ks, G, n_networks, n_samples,
                          cond, omega, theta_ref, w, l_pen, l_fuse, proba, weight_prev, loss, final,
                          constrain_basal_uniform=0.0, basal_free_mask=None):
    """
    Loss and gradient for one gene with one network per condition.

    X = [shared (G, nn), delta (C, G, nn), basal (n_samples, nn)] in theta units; the network of
    condition c is theta_c = shared + delta_c (effective = w * theta_c); sample s uses cond[s].
    Penalties: sparsity sum_c omega_c pen(theta_c - theta_ref_c) (omega_c = share of the samples,
    so a common network costs as now) + l_fuse * L1(w * delta_c) (0 = independent networks).
    """
    C, nn = len(omega), n_networks
    shared = X[:G * nn].reshape(G, nn)
    delta = X[G * nn:(C + 1) * G * nn].reshape(C, G, nn)
    basal = X[(C + 1) * G * nn:].reshape(n_samples, nn)
    theta_c = shared[None] + delta
    eff = theta_c * w[None]

    Q = 0.0
    g_eff = np.zeros((C, G, nn))
    g_basal = np.zeros((n_samples, nn))
    us = np.unique(ys.ravel())
    w_tot = np.sum(weights_samples)
    passes = [(1 - weight_prev, yp)] + ([(weight_prev, ypm)] if weight_prev else [])
    for pass_w, Y in passes:
        for cnt, s in enumerate(us):
            s_idx = int(s)
            c, mask = cond[s_idx], (ys == s)
            ws = pass_w * w_tot / weights_samples[cnt] / len(us)
            sig = base_kon(basal[s_idx], eff[c], Y[mask])
            if proba:
                Q += ws * main_loss(sig, ypr[mask], 1, loss)
                gl = grad_main_loss(sig, ypr[mask], 1, loss)
            else:
                ks_s = _kss(ks, s)
                Q += ws * main_loss(np.sum(ks_s * sig, axis=-1), yk[mask], 1, loss)
                gl = grad_main_loss(np.sum(ks_s * sig, axis=-1), yk[mask], 1, loss)
            for n in range(nn):
                # Softmax Jacobian column n
                grad_sigma = -sig * sig[:, n + 1, np.newaxis]
                grad_sigma[:, n + 1] += sig[:, n + 1]
                tmp = np.sum(gl * grad_sigma, axis=-1) if proba else gl * np.sum(ks_s * grad_sigma, axis=-1)
                g_eff[c, :, n] += ws * (Y[mask].T @ tmp)
                g_basal[s_idx, n] += ws * np.sum(tmp)

    g_theta = g_eff * w[None]
    on = (w > 0)
    for c in range(C):
        if omega[c] == 0:
            continue
        d = theta_c[c] - theta_ref[c]
        if not final:
            Q += omega[c] * theta_penalization(d * on, l_pen)
            g_theta[c] += omega[c] * grad_theta_penalization(d, l_pen) * on
        else:
            Q += omega[c] * (r_elasticnet * l2_penalization(np.sqrt(w) * d, l_pen)
                             + (1 - r_elasticnet) * smoothed_l1_penalization(d * on, l_pen))
            g_theta[c] += omega[c] * (r_elasticnet * grad_l2_penalization(d, l_pen) * w
                                      + (1 - r_elasticnet) * grad_smoothed_l1_penalization(d, l_pen) * on)

    # Fused penalty on the effective deviations from the shared network
    g_delta = g_theta.copy()
    if l_fuse > 0:
        Q += smoothed_l1_penalization(delta * w[None], l_fuse)
        g_delta += grad_smoothed_l1_penalization(delta * w[None], l_fuse) * w[None]

    if constrain_basal_uniform > 0.0 and basal_free_mask is not None:
        free_idx = np.where(basal_free_mask)[0]
        if len(free_idx) >= 2:
            theta_free = basal[free_idx, :]
            mean_free = theta_free.mean(axis=0)
            Q += constrain_basal_uniform * np.sum((theta_free - mean_free) ** 2)
            g_basal[free_idx, :] += 2.0 * constrain_basal_uniform * (theta_free - mean_free)

    grad = np.concatenate([g_theta.sum(axis=0).ravel(), g_delta.ravel(), g_basal.ravel()])
    return Q, grad


def fuse_conditions(theta_c, tol=1e-2):
    """Snap each condition's value to the median over conditions where it is within tol of it (exact fusion)."""
    med = np.median(theta_c, axis=0)
    return np.where(np.abs(theta_c - med[None]) < tol, med[None], theta_c)


def core_inference_conditions(y_samples, y_proba, y_prot, y_prot_mod, y_kon, theta_init, theta_ref,
                              ref_network, ks, G, g, n_networks, n_samples, cond, proba,
                              l_pen, l_fuse, weight_prev=.5, loss='CE', final=0,
                              constrain_basal_uniform=0.0, basal_free_mask=None,
                              hard_forcing_ref=False, ref_constraint_pct=0.1, seuil_zero_min_ref=1e-2):
    """
    core_inference with one network per condition (cond[s] = condition of sample s).

    theta_init / theta_ref : (C * G + n_samples, n_networks), condition c in rows c*G..(c+1)*G-1
    (theta units). The reference / sign bounds apply to the shared network; with
    hard_forcing_ref the deviations are pinned to 0. Returns the effective (C * G + n_samples, n_networks).
    """
    C = int(np.max(cond)) + 1
    nn = n_networks
    w = prior_weight(np.asarray(ref_network, dtype=float))
    us = np.unique(y_samples)
    weights_samples = [np.sum(y_samples == s) ** alpha for s in us]
    # Sparsity weight of each condition: its share of the samples of the fit
    omega = np.bincount(cond[us.astype(int)], minlength=C) / len(us)

    init_c = np.asarray(theta_init[:C * G], dtype=float).reshape(C, G, nn)
    ref_c = np.asarray(theta_ref[:C * G], dtype=float).reshape(C, G, nn)
    shared0 = np.median(init_c, axis=0)
    X_flat = np.concatenate([shared0.ravel(), (init_c - shared0[None]).ravel(),
                             np.asarray(theta_init[C * G:], dtype=float).ravel()])

    ref_flat = ref_c.mean(axis=0).ravel()
    ref_network_flat = np.ascontiguousarray(ref_network, dtype=float).ravel()
    bounds = _interaction_bounds(ref_flat, ref_network_flat, G, nn, final,
                                 hard_forcing_ref, ref_constraint_pct, seuil_zero_min_ref)
    if hard_forcing_ref and not final:
        bounds += [(0.0, 0.0)] * (C * G * nn)
    else:
        bounds += [(-2 * max(-lo, hi), 2 * max(-lo, hi)) for _ in range(C) for lo, hi in bounds[:G * nn]]
    bounds += [(-_basal_bound(G), _basal_bound(G))] * (n_samples * nn)

    fn = partial(_loss_grad_conditions, weights_samples=weights_samples, ys=y_samples,
                 ypr=y_proba, yp=y_prot, ypm=y_prot_mod, yk=y_kon, ks=ks, G=G, n_networks=nn,
                 n_samples=n_samples, cond=cond, omega=omega, theta_ref=ref_c, w=w,
                 l_pen=l_pen, l_fuse=l_fuse, proba=proba, weight_prev=weight_prev, loss=loss, final=final,
                 constrain_basal_uniform=constrain_basal_uniform, basal_free_mask=basal_free_mask)
    # Stage 1: common network (delta = 0), well conditioned; stage 2: deviations from it, started
    # from the previous deviations or from none (the stiff fused term alone stalls L-BFGS-B)
    d_sl = slice(G * nn, (C + 1) * G * nn)
    X_common = X_flat.copy()
    X_common[d_sl] = 0.0
    bounds_common = bounds[:G * nn] + [(0.0, 0.0)] * (C * G * nn) + bounds[(C + 1) * G * nn:]
    res = minimize(fn, X_common, jac=True, method="L-BFGS-B", bounds=bounds_common, options=_lbfgs_options(G))
    if l_fuse < np.inf and not (hard_forcing_ref and not final):
        starts = [res.x, np.concatenate([res.x[:G * nn], X_flat[d_sl], res.x[(C + 1) * G * nn:]])]
        x0 = min(starts, key=lambda x: fn(x)[0])
        res = minimize(fn, x0, jac=True, method="L-BFGS-B", bounds=bounds, options=_lbfgs_options(C * G))
    if not res.success:
        logger.error('Minimization failed for inference (network conditions): %s', res.message)

    if check_gradient:
        error = check_grad(lambda x: fn(x)[0], lambda x: fn(x)[1], res.x)
        if error > .05:
            logger.debug("Gradient theta inference (conditions) check error for gene %s: %s", g, error)

    shared = res.x[:G * nn].reshape(G, nn)
    delta = res.x[G * nn:(C + 1) * G * nn].reshape(C, G, nn)
    theta_c = fuse_conditions((shared[None] + delta) * w[None])
    return np.concatenate([theta_c.reshape(C * G, nn), res.x[(C + 1) * G * nn:].reshape(n_samples, nn)])


def refine_inference(y_samples, y_proba, y_prot, y_prot_mod, y_kon, inter, basal, theta_ref,
                     ks, G, g, n_networks, n_samples, proba,
                     l_pen, weight_prev=.5, loss='CE', correc_ref=0, final=0,
                     hard_forcing_ref=False, ref_constraint_pct=0.1, ref_network=None, seuil_zero_min_ref=1e-2,
                     cond=None):
    """
    Refinement step: optimise multiplicative correction factors over inter and per-sample basal.
    Corrections are bounded so that refined values stay within max_inter and _basal_bound(G);
    sign-forced edges (|ref_network| > 1) keep |theta| >= seuil_zero_min_ref.

    basal : (n_samples, n_networks)
    inter : (G, n_networks), or (C, G, n_networks) per condition (cond[s] = condition of sample s):
        one correction per edge, shared by the conditions, keeps their differences.
    Returns updated (inter, basal).
    """
    correc = np.ones((G + n_samples, n_networks))
    diag = np.zeros(inter.shape)
    if g < G:   # g == G is the sentinel meaning "no self-regulation in active set"
        diag[..., g, :] = inter[..., g, :]
    inter = inter.copy()
    inter -= diag
    basal = basal.copy()  # (n_samples, n_networks)

    weights_samples = [np.sum(y_samples == s) ** alpha for s in np.unique(y_samples)]

    correc_init_ = np.ascontiguousarray(correc, dtype=float)
    theta_ref_   = np.ascontiguousarray(theta_ref, dtype=float)
    X_flat = correc_init_.ravel(order='C')
    theta_ref_flat = theta_ref_.ravel(order='C')

    max_bounds = 10
    bounds = [(0, max_bounds)] * len(X_flat)
    n_inter_flat = G * n_networks
    if not final:
        r_plus = min(max_bounds, 1 + ref_constraint_pct / max(EPS, (1 - ref_constraint_pct)))
        r_minus = 1 - ref_constraint_pct
        if hard_forcing_ref:
            for idx in range(n_inter_flat):
                bounds[idx] = (r_minus, max(r_minus, r_plus))
        else:
            # Tighten bounds for non-zero reference interactions only
            for idx in range(n_inter_flat):
                if theta_ref_flat[idx] != 0.0:
                    bounds[idx] = (r_minus, max(r_minus, r_plus))

    # Cap corrections so that |value * correction| stays within capacity (reference lower bounds win)
    forced = (np.abs(ref_network) > 1).ravel() if ref_network is not None else np.zeros(n_inter_flat, bool)
    # Edges of prior weight w cost 1/w more, as in core_inference
    w = prior_weight(ref_network) if ref_network is not None else np.ones((G, n_networks))
    l_pen = np.where(w > 0, l_pen / np.maximum(w, EPS), l_pen) * np.ones((G, n_networks))
    # Per correction: largest |value| for the capacity, smallest for the forced magnitude (over conditions)
    a_inter = np.abs(inter).reshape(-1, n_inter_flat) if n_inter_flat else np.zeros((1, 0))
    v_max = np.concatenate([a_inter.max(axis=0), np.abs(basal).ravel()])
    v_min = np.concatenate([a_inter.min(axis=0), np.abs(basal).ravel()])
    caps = np.concatenate([np.full(n_inter_flat, max_inter), np.full(basal.size, _basal_bound(G))])
    for idx, (v, v_lo, cap) in enumerate(zip(v_max, v_min, caps)):
        if v != 0.0:
            lo, hi = bounds[idx]
            # Ratios written without dividing by a near-zero |v| (overflow; the min picks the other term anyway)
            if idx < n_inter_flat and forced[idx]:
                lo = max(lo, 1.0 if v_lo <= seuil_zero_min_ref else seuil_zero_min_ref / v_lo)  # keep the forced-sign magnitude
            bounds[idx] = (lo, max(lo, hi if v * hi <= cap else cap / v))

    loss_fn = partial(objective_refinement,
                      correc_ref=correc_ref, inter=inter, basal=basal,
                      weights_samples=weights_samples, ys=y_samples,
                      ypr=y_proba, yp=y_prot, ypm=y_prot_mod, yk=y_kon,
                      ks=ks, diag=diag, G=G, g=g, n_networks=n_networks, n_samples=n_samples,
                      proba=proba, l_pen=l_pen, weight_prev=weight_prev, loss=loss, final=final, cond=cond)
    grad_fn = partial(grad_correc,
                      correc_ref=correc_ref, inter=inter, basal=basal,
                      weights_samples=weights_samples, ys=y_samples,
                      ypr=y_proba, yp=y_prot, ypm=y_prot_mod, yk=y_kon,
                      ks=ks, diag=diag, G=G, g=g, n_networks=n_networks, n_samples=n_samples,
                      proba=proba, l_pen=l_pen, weight_prev=weight_prev, loss=loss, final=final, cond=cond)

    res = minimize(loss_fn, X_flat, jac=grad_fn, method="L-BFGS-B", bounds=bounds, options=_lbfgs_options(G))
    if not res.success:
        logger.error('Minimization failed for refining: %s', res.message)

    if check_gradient:
        error = check_grad(loss_fn, grad_fn, res.x)
        if error > .05:
            logger.debug("Gradient theta refining check error for gene %s: %s", g, error)
            res = minimize(loss_fn, X_flat, bounds=bounds, method="L-BFGS-B", options=_lbfgs_options(G))

    correc = res.x.reshape(G + n_samples, n_networks)
    inter *= correc[:G, :]
    basal *= correc[G:, :]     # (n_samples, n_networks) *= (n_samples, n_networks)
    inter += diag

    return inter, basal


def main_loop_inference(g, y_samples, y_proba, y_prot, y_prot_mod, y_kon, theta_init, theta_ref,
                         ks, G, n_networks, n_samples, proba, l_gen, scale,
                         inter_tmp, basal_tmp, inter, basal, ref_network,
                         weight_prev=.5, loss='CE', final=0,
                         constrain_basal_uniform=0.0, basal_free_mask=None,
                         hard_forcing_ref=False, ref_constraint_pct=0.1, seuil_zero_min_ref=1e-2,
                         cond=None, condition_pen=1.0):
    """
    Joint inference (inter + per-sample basal) for a single gene g.

    theta_init : (G + n_samples, n_networks), or (C * G + n_samples, n_networks) with one
                 network per condition (cond[s] = condition of sample s, see core_inference_conditions)
    basal      : (n_samples, n_networks)  — will be updated in-place (copy returned)
    inter      : (G, n_networks), or (C, G, n_networks) with cond — will be updated in-place (copy returned)
    basal_free_mask : (n_samples,) bool — True for samples NOT pinned by basal_ref for gene g
    condition_pen : fused penalty on the deviations between conditions, relative to the sparsity one
    Returns (basal, inter, basal_tmp, inter_tmp).
    """
    # Active networks from the largest mode amplitudes over samples
    n_networks_tmp: int = int(1 + np.argmax((ks.max(axis=0) if ks.ndim == 2 else ks)[1:]))
    C = 1 if cond is None else int(np.max(cond)) + 1
    n_rows = C * G   # interaction rows of theta

    # Interactions are optimised as theta = e / w (w = prior weight), so an edge of weight q
    # costs 1/q more; init and ref are given as effective values e and converted here.
    w = prior_weight(ref_network)
    w_rows = np.tile(w, (C, 1))
    theta_init, theta_ref = theta_init.copy(), theta_ref.copy()
    theta_init[:n_rows] = np.where(w_rows > 0, theta_init[:n_rows] / np.maximum(w_rows, EPS), 0.0)
    theta_ref[:n_rows] = np.where(w_rows > 0, theta_ref[:n_rows] / np.maximum(w_rows, EPS), 0.0)

    l_pen1 = l_gen * np.size(y_prot, 0) / (n_networks_tmp * scale * (1 + G**(1/2)))
    common = dict(weight_prev=weight_prev * (1 - final), loss=loss, final=final,
                  constrain_basal_uniform=constrain_basal_uniform, basal_free_mask=basal_free_mask,
                  hard_forcing_ref=hard_forcing_ref, ref_constraint_pct=ref_constraint_pct,
                  seuil_zero_min_ref=seuil_zero_min_ref)
    if cond is None:
        theta = core_inference(
            y_samples, y_proba[:, :n_networks_tmp + 1], y_prot, y_prot_mod, y_kon,
            theta_init[:, :n_networks_tmp],
            theta_ref[:, :n_networks_tmp],
            ref_network[:, :n_networks_tmp],
            ks[..., :n_networks_tmp + 1], G, g, n_networks_tmp, n_samples, proba,
            l_pen1, **common,
        )
    else:
        theta = core_inference_conditions(
            y_samples, y_proba[:, :n_networks_tmp + 1], y_prot, y_prot_mod, y_kon,
            theta_init[:, :n_networks_tmp], theta_ref[:, :n_networks_tmp],
            ref_network[:, :n_networks_tmp],
            ks[..., :n_networks_tmp + 1], G, g, n_networks_tmp, n_samples, cond, proba,
            l_pen1, condition_pen * l_pen1, **common,
        )
    # theta shape: (C * G + n_samples, n_networks_tmp)
    inter_new = theta[:n_rows].reshape(inter[..., :n_networks_tmp].shape)
    inter[..., :n_networks_tmp] = inter_new
    basal[:, :n_networks_tmp] = theta[n_rows:n_rows + n_samples, :]
    inter_tmp[..., :n_networks_tmp] = inter_new
    basal_tmp[:, :n_networks_tmp] = theta[n_rows:n_rows + n_samples, :]

    # Refinement step (reference of the shared network for the conditions)
    ref_refine = theta_ref if cond is None else np.concatenate(
        [theta_ref[:n_rows].reshape(C, G, theta_ref.shape[1]).mean(axis=0), theta_ref[n_rows:]])
    l_pen2 = l_gen / (n_networks_tmp * (1 + np.log(max(G, 1))))  # G == 0: no regulator
    inter[..., :n_networks_tmp], basal[:, :n_networks_tmp] = refine_inference(
        y_samples, y_proba[:, :n_networks_tmp + 1], y_prot, y_prot_mod, y_kon,
        inter[..., :n_networks_tmp], basal[:, :n_networks_tmp],
        ref_refine[:, :n_networks_tmp],
        ks[..., :n_networks_tmp + 1], G, g, n_networks_tmp, n_samples, proba,
        l_pen2, weight_prev=weight_prev * (1 - final), loss=loss,
        correc_ref=final, final=final, hard_forcing_ref=hard_forcing_ref, ref_constraint_pct=ref_constraint_pct,
        ref_network=ref_network[:, :n_networks_tmp], seuil_zero_min_ref=seuil_zero_min_ref, cond=cond,
    )

    if n_networks_tmp < n_networks:
        basal[:, n_networks_tmp:] = -100
        basal_tmp[:, n_networks_tmp:] = -100

    return basal, inter, basal_tmp, inter_tmp


def prior_weight(ref_network):
    """Prior weight w of each edge: |v| capped at 1. An edge costs 1/w more; |v| > 1 only adds a forced sign."""
    return np.minimum(np.abs(ref_network), 1.0)


def signed_floor(ref_network, floor):
    """
    Raise |ref_network| to at least floor, keeping the sign (0 counts as positive).
    ref_network values: |v| <= 1 is the soft prior weight, |v| > 1 forces the sign (see prior_weight).
    """
    ref_network = np.asarray(ref_network, dtype=float)
    return np.where(ref_network < 0, -1.0, 1.0) * np.maximum(np.abs(ref_network), floor)


def active_regulators(ref_network, inter_ref=None, hard_forcing_ref=False):
    """
    Effective structural prior used by inference_network and, for each target
    gene g, the sorted indices of its active (non-zero) regulators.

    ref_network : (G, G, n_networks); with hard_forcing_ref, edges of inter_ref are added
    (inter_ref per sample (n_samples, G, G, n_networks): edges of any sample).
    Returns (ref_network, cols) with cols[g] an int array.
    """
    ref_network = np.asarray(ref_network, dtype=float)
    if hard_forcing_ref and inter_ref is not None:
        present = np.asarray(inter_ref, dtype=float) != 0
        if present.ndim == 4:
            present = present.any(axis=0)
        ref_network = signed_floor(ref_network, present.astype(float))
    cols = [np.flatnonzero(np.any(ref_network[:, g, :] != 0, axis=-1)) for g in range(ref_network.shape[1])]
    return ref_network, cols


class PrevProt:
    """
    Flow-matching protein states per target gene, stored only on its active
    regulators (sparse replacement of a dense (G, N_cells, G) tensor).

    cols[g]   : sorted regulator indices of target g (see active_regulators)
    values[g] : (N_cells, len(cols[g])) protein states of these regulators
    """

    def __init__(self, cols, values):
        self.cols = cols
        self.values = values

    @classmethod
    def zeros(cls, cols, n_cells):
        return cls(cols, [np.zeros((n_cells, len(c))) for c in cols])

    def select_cells(self, idx):
        return PrevProt(self.cols, [v[idx] for v in self.values])

    def get(self, g, src):
        """(N_cells, len(src)) states of regulators src (a subset of cols[g]) for target g."""
        cols = self.cols[g]
        pos = np.searchsorted(cols, src)
        if not (np.all(pos < len(cols)) and np.array_equal(cols[np.minimum(pos, len(cols) - 1)], src)):
            raise ValueError(f"Regulators of target {g} were not stored in PrevProt")
        return self.values[g] if len(src) == len(cols) else self.values[g][:, pos]


def _infer_gene(g_tgt, G, active_src, y_samples, y_proba_g, yp, ypm, y_kon_g, ti_g, tr_g,
                ks_g, n_networks, n_samples, proba, l_gen, scale, rn, free_mask_g, gene_kw, cond=None):
    """
    Network inference for one target gene (worker task of inference_network).
    yp / ypm / rn are already restricted to the active regulators active_src
    (full when every regulator is active). With cond (condition of each sample), ti_g / tr_g hold
    one block of G interaction rows per condition and the interactions returned are (C, G, n_networks).
    """
    K_g = len(active_src)
    C = 1 if cond is None else int(np.max(cond)) + 1
    shape = (G, n_networks) if cond is None else (C, G, n_networks)
    if K_g == G:
        # Dense path: no sub-selection
        return main_loop_inference(
            g_tgt, y_samples, y_proba_g, yp, ypm, y_kon_g, ti_g, tr_g,
            ks_g, G, n_networks, n_samples, proba, l_gen, scale,
            np.zeros(shape), np.zeros((n_samples, n_networks)),
            np.zeros(shape), np.zeros((n_samples, n_networks)),
            rn, basal_free_mask=free_mask_g, cond=cond, **gene_kw,
        )

    # Sparse path: sub-select to K_g active regulators. K_g == 0: no interaction
    # to infer, only the per-sample basal is fitted (O(N_cells) problem).
    # Sub-select theta rows: active_src rows (of each condition) + per-sample basal rows
    rows = np.concatenate([c * G + active_src for c in range(C)] + [np.arange(C * G, C * G + n_samples)])
    theta_init_sub = ti_g[rows]  # (C * K_g + n_samples, n_networks)
    theta_ref_sub  = tr_g[rows]

    # Position of g_tgt in active_src (sentinel K_g = "no self-regulation")
    g_sub_matches = np.where(active_src == g_tgt)[0]
    g_sub = int(g_sub_matches[0]) if len(g_sub_matches) > 0 else K_g

    shape_sub = (K_g, n_networks) if cond is None else (C, K_g, n_networks)
    basal_r, inter_sub_r, basal_tmp_r, inter_tmp_sub_r = main_loop_inference(
        g_sub, y_samples, y_proba_g, yp, ypm, y_kon_g,
        theta_init_sub, theta_ref_sub,
        ks_g, K_g, n_networks, n_samples, proba, l_gen, scale,
        np.zeros(shape_sub), np.zeros((n_samples, n_networks)),
        np.zeros(shape_sub), np.zeros((n_samples, n_networks)), rn,
        basal_free_mask=free_mask_g, cond=cond, **gene_kw,
    )

    # Expand interaction results back to full G-dimensional space
    inter_full     = np.zeros(shape)
    inter_tmp_full = np.zeros(shape)
    inter_full[..., active_src, :]     = inter_sub_r
    inter_tmp_full[..., active_src, :] = inter_tmp_sub_r
    return basal_r, inter_full, basal_tmp_r, inter_tmp_full


def inference_network_multi(subsets, y_samples, y_kon, y_proba, y_prot, y_prot_mod, ks, n_stimuli=1, proba=0,
                      ref_network=None,
                      basal_init=None, inter_init=None,
                      basal_ref=None, inter_ref=None,
                      scale=100, weight_prev=.5, loss='CE', final=0,
                      samples_id=None,
                      constrain_basal_uniform=0.0,
                      hard_forcing_ref=False, ref_constraint_pct=0.1,
                      seuil_zero_min_ref=1e-2,
                      sample_conditions=None, condition_pen=1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Joint network inference: interactions shared across samples, basal per sample.
    One independent fit per row subset in subsets (index arrays, None = all rows);
    all fits x target genes are run in a single parallel pool.

    y_prot_mod : PrevProt built on active_regulators(ref_network, inter_ref,
        hard_forcing_ref), or a dense (G, N_cells, G) array.

    Returns a list with, per subset, (basal, inter, basal_tmp, inter_tmp).
    basal shape: (n_samples, G, n_networks)

    constrain_basal_uniform : float >= 0
        When > 0, adds a penalty that pushes per-sample basals toward their common
        mean for each gene. Samples whose basal is pinned by a non-zero basal_ref
        (KO/OV priors) are excluded from the penalty for that gene.
    sample_conditions : (n_samples,) int array or None
        Network condition of each sample (index). With >= 2 conditions, each condition has its
        own network theta_c = theta_shared + delta_c, with a fused L1 penalty on delta_c of
        condition_pen times the sparsity penalty (0: independent networks); inter is then
        returned per sample, (n_samples, G, G, n_networks). inter_init / inter_ref may be
        (G, G, n_networks) (common) or per sample.
    """
    if ref_constraint_pct < 0:
        raise ValueError(f"ref_constraint_pct must be >= 0, got {ref_constraint_pct}")
    try:
        from joblib import Parallel, delayed
    except ImportError:
        Parallel = None
        delayed = None
        logging.getLogger(__name__).warning("joblib not available; parallel loops will run sequentially")

    G: int = np.size(y_prot, 1)
    n_networks: int = ks.shape[-1] - 1  # ks: (G, n_modes) shared or (n_samples, G, n_modes) per sample
    unique_samples = np.asarray(samples_id) if samples_id is not None else np.unique(y_samples)
    n_samples: int = len(unique_samples)

    # ── Network conditions: one network per condition when there are >= 2 ──
    cond = None
    if sample_conditions is not None and len(np.unique(sample_conditions)) > 1:
        cond = np.unique(np.asarray(sample_conditions), return_inverse=True)[1].astype(int)
        if len(cond) != n_samples:
            raise ValueError(f"sample_conditions has {len(cond)} entries for {n_samples} samples")
    C = 1 if cond is None else int(cond.max()) + 1
    # First sample of each condition (its slice of per-sample init / ref)
    first_of = np.arange(1) if cond is None else np.array([np.flatnonzero(cond == c)[0] for c in range(C)])

    def inter_rows(arr):
        # (C * G, G, n_networks) interaction rows: common (G, G, nn) or per sample (n_samples, G, G, nn)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 4:
            if cond is None and not np.allclose(arr, arr[:1]):
                raise ValueError("per-sample interactions given without network conditions")
            arr = arr[first_of]
        else:
            arr = np.repeat(arr[None], C, axis=0)
        return arr.reshape(C * G, G, n_networks)

    # ── Build theta_init (C * G + n_samples, G, n_networks) ─────────────────
    # rows 0..C*G-1: interactions per target gene (one block per condition);
    # rows C*G..C*G+ns-1: per-sample basal per target gene
    theta_init_mat = np.zeros((C * G + n_samples, G, n_networks))
    if inter_init is not None:
        theta_init_mat[:C * G, :, :] = inter_rows(inter_init)

    if basal_init is not None:
        bi = np.asarray(basal_init, dtype=float)
        if bi.ndim == 3:                    # (n_samples, G, n_networks)
            for s_idx in range(min(n_samples, bi.shape[0])):
                theta_init_mat[C * G + s_idx, :, :] = bi[s_idx]
        else:                               # (G, n_networks) — broadcast
            for s_idx in range(n_samples):
                theta_init_mat[C * G + s_idx, :, :] = bi

    # ── Build theta_ref (C * G + n_samples, G, n_networks) ──────────────────
    theta_ref_mat = np.zeros((C * G + n_samples, G, n_networks))
    if inter_ref is not None:
        theta_ref_mat[:C * G, :, :] = inter_rows(inter_ref)

    if basal_ref is not None:
        br = np.asarray(basal_ref, dtype=float)
        if br.ndim == 3:
            for s_idx in range(min(n_samples, br.shape[0])):
                theta_ref_mat[C * G + s_idx, :, :] = br[s_idx]
        else:
            for s_idx in range(n_samples):
                theta_ref_mat[C * G + s_idx, :, :] = br

    if ref_network is None:
        ref_network = np.ones((G, G, n_networks))

    # When hard_forcing_ref, the inter_ref matrix defines the allowed structure in addition
    # to ref_network: interactions specified by inter_ref must not be zeroed out by the
    # structural prior mask at the end of core_inference.
    ref_network, active_cols = active_regulators(ref_network, inter_ref, hard_forcing_ref)

    def prev_prot(g_tgt, src):
        # Flow-matching protein states of regulators src for target g_tgt
        if isinstance(y_prot_mod, PrevProt):
            return y_prot_mod.get(g_tgt, src)
        return y_prot_mod[g_tgt][:, src]

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

    gene_kw = dict(weight_prev=weight_prev, loss=loss, final=final,
                   constrain_basal_uniform=constrain_basal_uniform,
                   hard_forcing_ref=hard_forcing_ref, ref_constraint_pct=ref_constraint_pct,
                   seuil_zero_min_ref=seuil_zero_min_ref, condition_pen=condition_pen)

    def gene_args(rows, g_tgt):
        # Only the rows and regulators target g_tgt needs are sent to the worker (built lazily)
        rows = slice(None) if rows is None else rows
        active_src = active_cols[g_tgt]
        dense = len(active_src) == G
        return (g_tgt, G, active_src, y_samples[rows], y_proba[rows, g_tgt],
                y_prot[rows] if dense else y_prot[rows][:, active_src], prev_prot(g_tgt, active_src)[rows],
                y_kon[rows, g_tgt], theta_init_mat[:, g_tgt, :], theta_ref_mat[:, g_tgt, :],
                ks[:, g_tgt] if ks.ndim == 3 else ks[g_tgt], n_networks, n_samples, proba, l_gen, scale,
                ref_network[:, g_tgt, :] if dense else ref_network[active_src, g_tgt, :],
                free_mask_2d[:, g_tgt], gene_kw, cond)

    genes = range(n_stimuli, G)
    tasks = [(k, g) for k in range(len(subsets)) for g in genes]
    if Parallel is not None:
        results = Parallel(n_jobs=-1)(
            delayed(_infer_gene)(*gene_args(subsets[k], g)) for k, g in tasks
        )
    else:
        results = [_infer_gene(*gene_args(subsets[k], g)) for k, g in tasks]

    fits = []
    shape = (G, G, n_networks) if cond is None else (C, G, G, n_networks)
    for k in range(len(subsets)):
        basal, basal_tmp = np.zeros((n_samples, G, n_networks)), np.zeros((n_samples, G, n_networks))
        inter, inter_tmp = np.zeros(shape), np.zeros(shape)
        for idx, g in enumerate(genes):
            basal[:, g, :], inter[..., g, :], basal_tmp[:, g, :], inter_tmp[..., g, :] = results[k * len(genes) + idx]
        if cond is not None:
            # Per sample: each sample takes the network of its condition
            inter, inter_tmp = inter[cond], inter_tmp[cond]
        fits.append((basal, inter, basal_tmp, inter_tmp))
    return fits


def inference_network(y_samples, y_kon, y_proba, y_prot, y_prot_mod, ks, **kwargs):
    """Single joint network fit on all rows (see inference_network_multi). Returns (basal, inter, basal_tmp, inter_tmp)."""
    return inference_network_multi([None], y_samples, y_kon, y_proba, y_prot, y_prot_mod, ks, **kwargs)[0]
