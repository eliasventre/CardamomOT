"""
Core functions for network inference, mainly use in loop_trajectories
"""

from typing import Any
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import numpy as np
# joblib is only required for parallel loops; delay import to runtime to
# avoid failing module import when the package is missing.  The functions
# that need it will import inside themselves.

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
sc = 1e-3
r_elasticnet = 0.5


### FUNCTIONS FOR NETWORK INFERENCE

@njit(fastmath=True, cache=True)
def smoothed_l1_penalization(array, l1, sc=sc):
    # Applique une version lissée de la norme L1 pour chaque élément
    smoothed_array: np.ndarray[Any, np.dtype[Any]] = np.where(
        np.abs(array) < sc,  # Vérifie si la valeur absolue est sous le seuil
        l1 * 0.5 * (array ** 2) / sc,  # Quadratique si |x| < sc
        l1 * (np.abs(array) - 0.5 * sc)  # Linéaire autrement
    )
    return np.sum(smoothed_array)


@njit(fastmath=True, cache=True)
def grad_smoothed_l1_penalization(array, l1, sc=sc) -> np.ndarray:
    # Applique une version lissée de la norme L1 pour chaque élément
    smoothed_array = np.where(
        np.abs(array) < sc,  # Vérifie si la valeur absolue est sous le seuil
        l1 * array / sc,  # Quadratique si |x| < sc
        l1 * np.sign(array)  # Linéaire autrement
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
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -l * np.sum(
            y_true * np.log(y_pred) + 
            (1 - y_true) * np.log1p(-y_pred)  
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
    """Version plus robuste avec clipping des exponentielles"""
    n_cells, G = y_prot.shape
    n_net = theta_basal.size
    Z = np.zeros((n_cells, n_net))
    result = np.empty((n_cells, n_net + 1))
    
    # Limite pour éviter overflow dans exp()
    MAX_EXP = 50.0  # exp(50) ≈ 5e21, largement suffisant
    
    for i in range(n_cells):
        # Calcul des logits
        for k in range(n_net):
            Z[i, k] = theta_basal[k]
            for g in range(G):
                Z[i, k] += y_prot[i, g] * theta_inter[g, k]
        
        # Trouver le max pour stabilité numérique
        z_max = np.max(Z[i])
        
        # Clipping pour éviter les valeurs extrêmes
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


def objective(X, weights_samples, ys, ypr, yp, ypm, yk, ks, G, g, n_networks, theta_ref, ref_network, l_pen, proba, weight_prev, loss, final):
    """
    Objective function to be minimized (all cells, all genes, one timepoints).
    """ 

    theta = X.reshape(G+1, n_networks) 
    Q = 0
    us = np.unique(ys.ravel())
    ns: int = len(us)
    sigma = base_kon(theta[-1], theta[:-1] * ref_network, yp)
    for cnt, s in enumerate(us):
        weight_s = np.sum(weights_samples)/weights_samples[cnt] # Ensure that each batch is considered with the same weight
        if proba: 
            Q += (1-weight_prev) * main_loss(sigma[ys==s], ypr[ys==s], 1, loss) * weight_s/ns
        else: 
            Q += (1-weight_prev) * main_loss(np.sum(ks * sigma[ys==s], axis=-1), yk[ys==s], 1, loss) * weight_s/ns
    if weight_prev:
        sigma_mod = base_kon(theta[-1], theta[:-1] * ref_network, ypm)
        for cnt, s in enumerate(us):
            weight_s = np.sum(weights_samples)/weights_samples[cnt] # Ensure that each batch is considered with the same weight
            if proba: 
                Q += weight_prev * main_loss(sigma_mod[ys==s], ypr[ys==s], 1, loss) * weight_s/ns
            else: 
                Q += weight_prev * main_loss(np.sum(ks * sigma_mod[ys==s], axis=-1), yk[ys==s], 1, loss) * weight_s/ns
    if not final: 
        Q += theta_penalization((theta[:-1] - theta_ref[:-1]) * (ref_network > 0), l_pen)
    else: 
        Q += final_theta_penalization((theta[:-1] - theta_ref[:-1]) * (ref_network > 0), l_pen)
    return Q


def grad_theta(X, weights_samples, ys, ypr, yp, ypm, yk, ks, G, g, n_networks, theta_ref, ref_network, l_pen, proba, weight_prev, loss, final):
    """
    Objective gradient
    """

    theta = X.reshape(G+1, n_networks) 
    dq = np.zeros_like(theta)
    us = np.unique(ys.ravel())
    ns: int = len(us)
    sigma = base_kon(theta[-1], theta[:-1] * ref_network, yp)
    for n in range(n_networks):
        grad_sigma = -sigma*sigma[:,n+1,np.newaxis]
        for cnt, s in enumerate(us):
            weight_s = np.sum(weights_samples)/weights_samples[cnt]
            grad_sigma[ys==s,n+1] += sigma[ys==s,n+1]
            if proba:
                tmp = grad_main_loss(sigma[ys==s], ypr[ys==s], 1, loss) * grad_sigma[ys==s]
                dq[:-1,n] += (1-weight_prev) * ref_network[:, n] * (yp[ys==s].T @ np.sum(tmp, axis=-1, keepdims=True)) * weight_s/ns
                dq[-1,n] += (1-weight_prev) * np.sum(np.sum(tmp, axis=-1)) * weight_s/ns
            else: 
                tmp = grad_main_loss(np.sum(ks * sigma[ys==s], axis=-1), yk[ys==s], 1, loss) * np.sum(ks * grad_sigma[ys==s], axis=-1)
                res = (yp[ys == s].T @ tmp[:, None]).reshape(-1)
                dq[:-1,n] += (1-weight_prev) * ref_network[:, n] * res * weight_s/ns
                dq[-1,n] += (1-weight_prev) * np.sum(tmp) * weight_s/ns
    if weight_prev:
        sigma_mod = base_kon(theta[-1], theta[:-1] * ref_network, ypm)
        for n in range(n_networks):
            grad_sigma_mod = -sigma_mod*sigma_mod[:,n+1,np.newaxis]
            for cnt, s in enumerate(us):
                weight_s = np.sum(weights_samples)/weights_samples[cnt]
                grad_sigma_mod[ys==s,n+1] += sigma_mod[ys==s,n+1]
                if proba:
                    tmp_mod = grad_main_loss(sigma_mod[ys==s], ypr[ys==s], 1, loss) * grad_sigma_mod[ys==s]
                    dq[:-1,n] += weight_prev * ref_network[:, n] * (ypm[ys==s].T @ np.sum(tmp_mod, axis=-1, keepdims=True)) * weight_s/ns
                    dq[-1,n] += weight_prev * np.sum(np.sum(tmp_mod, axis=-1)) * weight_s/ns
                else: 
                    tmp_mod = grad_main_loss(np.sum(ks * sigma_mod[ys==s], axis=-1), yk[ys==s], 1, loss) * np.sum(ks * grad_sigma_mod[ys==s], axis=-1)
                    res_mod = (ypm[ys == s].T @ tmp_mod[:, None]).reshape(-1)
                    dq[:-1,n] += weight_prev * ref_network[:, n] * res_mod * weight_s/ns
                    dq[-1,n] += weight_prev * np.sum(tmp_mod) * weight_s/ns
    if not final: 
        dq[:-1] += grad_theta_penalization(theta[:-1] - theta_ref[:-1], l_pen) * (ref_network > 0)
    else: 
        dq[:-1] += grad_final_theta_penalization(theta[:-1] - theta_ref[:-1], l_pen) * (ref_network > 0)
    return dq.ravel()


def objective_refinement(X, correc_ref, inter, basal, weights_samples, ys, ypr, yp, ypm, yk, ks, diag, G, g, n_networks, l_pen, proba, weight_prev, loss, final):
    """
    Objective function to be minimized (all cells, all genes).
    """
    correc = X.reshape(G+1, n_networks)
    theta_inter = inter.copy()
    theta_basal = basal.copy()
    theta_inter *= correc[:-1, :]
    theta_basal *= correc[-1, :]
    Q = 0
    us = np.unique(ys.ravel())
    ns: int = len(us)
    sigma = base_kon(theta_basal, theta_inter + diag, yp)
    for cnt, s in enumerate(us):
        weight_s = np.sum(weights_samples)/weights_samples[cnt]
        if proba: 
            Q += (1-weight_prev) * main_loss(sigma[ys==s], ypr[ys==s], 1, loss) * weight_s/ns
        else: 
            Q += (1-weight_prev) * main_loss(np.sum(ks * sigma[ys==s], axis=-1), yk[ys==s], 1, loss) * weight_s/ns
    if weight_prev:
        sigma_mod = base_kon(theta_basal, theta_inter + diag, ypm)
        for cnt, s in enumerate(us):
            weight_s = np.sum(weights_samples)/weights_samples[cnt]
            if proba: 
                Q += weight_prev * main_loss(sigma_mod[ys==s], ypr[ys==s], 1, loss) * weight_s/ns
            else: 
                Q += weight_prev * main_loss(np.sum(ks * sigma_mod[ys==s], axis=-1), yk[ys==s], 1, loss) * weight_s/ns
    if not final: 
        Q += theta_penalization(correc[:g] - correc_ref, l_pen)
        Q += theta_penalization(correc[g+1:-1] - correc_ref, l_pen)
    else: 
        Q += final_theta_penalization(correc[:g] - correc_ref, l_pen)
        Q += final_theta_penalization(correc[g+1:-1] - correc_ref, l_pen)
    return Q


def grad_correc(X, correc_ref, inter, basal, weights_samples, ys, ypr, yp, ypm, yk, ks, diag, G, g, n_networks, l_pen, proba, weight_prev, loss, final):
    """
    Objective gradient
    """

    correc = X.reshape(G+1, n_networks)
    theta_inter = inter.copy()
    theta_basal = basal.copy()
    theta_inter *= correc[:-1, :]
    theta_basal *= correc[-1, :]
    dq = np.zeros_like(correc)
    us = np.unique(ys.ravel())
    ns: int = len(us)
    sigma = base_kon(theta_basal, theta_inter + diag, yp)
    for n in range(n_networks):
        grad_sigma = -sigma*sigma[:,n+1,np.newaxis]
        for cnt, s in enumerate(us):
            weight_s = np.sum(weights_samples)/weights_samples[cnt]
            grad_sigma[ys == s,n+1] += sigma[ys == s,n+1]
            if proba: 
                tmp = grad_main_loss(sigma[ys == s], ypr[ys == s], 1, loss) * grad_sigma[ys == s]
                dq[:-1,n] += (1-weight_prev) * inter[:,n] * (yp[ys == s].T @ np.sum(tmp, axis=-1, keepdims=True)) * weight_s/ns
                dq[-1,n] += (1-weight_prev) * basal[n] * np.sum(np.sum(tmp, axis=-1)) * weight_s/ns
            else:
                tmp = grad_main_loss(np.sum(ks * sigma[ys == s], axis=-1), yk[ys == s], 1, loss) * np.sum(ks * grad_sigma[ys == s], axis=-1) 
                res = (yp[ys == s].T @ tmp[:, None]).reshape(-1)
                dq[:-1,n] += (1-weight_prev) * inter[:,n] * res * weight_s/ns
                dq[-1,n] += (1-weight_prev) * basal[n] * np.sum(tmp) * weight_s/ns
    if weight_prev:
        sigma_mod = base_kon(theta_basal, theta_inter + diag, ypm)
        for n in range(n_networks):
            grad_sigma_mod = -sigma_mod*sigma_mod[:,n+1,np.newaxis]
            for cnt, s in enumerate(us):
                weight_s = np.sum(weights_samples)/weights_samples[cnt]
                grad_sigma_mod[ys == s,n+1] += sigma_mod[ys == s,n+1]
                if proba: 
                    tmp_mod = grad_main_loss(sigma_mod[ys == s], ypr[ys == s], 1, loss) * grad_sigma_mod[ys == s]
                    dq[:-1,n] += weight_prev * inter[:,n] * (ypm[ys == s].T @ np.sum(tmp_mod, axis=-1, keepdims=True)) * weight_s/ns
                    dq[-1,n] += weight_prev * basal[n] * np.sum(np.sum(tmp_mod, axis=-1)) * weight_s/ns
                else:
                    tmp_mod = grad_main_loss(np.sum(ks * sigma_mod[ys == s], axis=-1), yk[ys == s], 1, loss) * np.sum(ks * grad_sigma_mod[ys == s], axis=-1) 
                    res_mod = (ypm[ys == s].T @ tmp_mod[:, None]).reshape(-1)
                    dq[:-1,n] += weight_prev * inter[:,n] * res_mod * weight_s/ns
                    dq[-1,n] += weight_prev * basal[n] * np.sum(tmp_mod) * weight_s/ns
    if not final: 
        dq[:g] += grad_theta_penalization(correc[:g] - correc_ref, l_pen)
        dq[g+1:-1] += grad_theta_penalization(correc[g+1:-1] - correc_ref, l_pen)
    else: 
        dq[:g] += grad_final_theta_penalization(correc[:g] - correc_ref, l_pen)
        dq[g+1:-1] += grad_final_theta_penalization(correc[g+1:-1] - correc_ref, l_pen)
    return dq.ravel()


def core_inference(y_samples, y_proba, y_prot, y_prot_mod, y_kon, theta_init, theta_ref, ref_network, ks, G, g, n_networks, proba, l_pen, weight_prev=.5, loss='CE', final=0):
    
    X = theta_init.ravel()
    weights_samples =[np.sum(y_samples == s)**alpha for s in np.unique(y_samples)]
    y_prot_mod = y_prot.copy()

    loss_fn = partial(objective, weights_samples=weights_samples, ys=y_samples,
                        ypr=y_proba, yp=y_prot, ypm = y_prot_mod, yk=y_kon,
                        ks=ks, G=G, g=g, n_networks=n_networks,
                        theta_ref=theta_ref, ref_network=ref_network, l_pen=l_pen, proba=proba, weight_prev=weight_prev, loss=loss, final=final)

    grad_fn = partial(grad_theta, weights_samples=weights_samples, ys=y_samples,
                        ypr=y_proba, yp=y_prot, ypm=y_prot_mod, yk=y_kon,
                        ks=ks, G=G, g=g, n_networks=n_networks,
                        theta_ref=theta_ref, ref_network=ref_network, l_pen=l_pen, proba=proba, weight_prev=weight_prev, loss=loss, final=final)

    res = minimize(loss_fn, X, jac=grad_fn, method="L-BFGS-B", tol=seuil/G)
    if not res.success:
        logger.error('Minimization failed for inference: %s', res.message)

    if check_gradient:
        error = check_grad(loss_fn, grad_fn, res.x)
        if error > .05:
            logger.debug("Gradient theta inference check error for gene %s: %s", g, error)
            res = minimize(loss_fn, X, method="L-BFGS-B", tol=seuil/G)

    theta_final = res.x.reshape(G+1, n_networks)
    theta_final[:-1] *= ref_network

    return theta_final
    

def refine_inference(y_samples, y_proba, y_prot, y_prot_mod, y_kon, inter, basal, ks, G, g, n_networks, proba, l_pen, weight_prev=.5, loss='CE', correc_ref=0, final=0):

        correc = np.ones((G+1, n_networks))
        diag = np.zeros((G, n_networks))
        diag[g, :] = inter[g, :]
        inter -= diag

        weights_samples = [np.sum(y_samples == s)**alpha for s in np.unique(y_samples)]

        loss_fn = partial(objective_refinement, correc_ref=correc_ref, inter=inter, basal=basal, 
                                weights_samples=weights_samples, ys=y_samples, 
                                ypr=y_proba, yp=y_prot, ypm = y_prot_mod, yk=y_kon, 
                                ks=ks, diag=diag, G=G, g=g, n_networks=n_networks, proba=proba, l_pen=l_pen, weight_prev=weight_prev, loss=loss, final=final)
        grad_fn = partial(grad_correc, correc_ref=correc_ref, inter=inter, basal=basal, 
                                weights_samples=weights_samples, ys=y_samples, 
                                ypr=y_proba, yp=y_prot, ypm = y_prot_mod, yk=y_kon, 
                                ks=ks, diag=diag, G=G, g=g, n_networks=n_networks, proba=proba, l_pen=l_pen, weight_prev=weight_prev, loss=loss, final=final)

        res = minimize(loss_fn, correc.ravel(), jac=grad_fn, method="L-BFGS-B", tol=seuil/G)
        if not res.success:
            logger.error('Minimization failed for refining: %s', res.message)
            
        if check_gradient:
            error = check_grad(loss_fn, grad_fn, res.x)
            if error > .05:
                logger.debug("Gradient theta refining check error for gene %s: %s", g, error)
                res = minimize(loss_fn, correc.ravel(), method="L-BFGS-B", tol=seuil/G)

        correc = res.x.reshape(G+1, n_networks)
        inter[:, :] *= correc[:-1, :]
        basal[:] *= correc[-1, :]
        inter += diag

        return inter, basal



def main_loop_inference(g, vect_t, y_samples, y_proba, y_prot, y_prot_mod, y_kon, theta_init, theta_ref, 
                  ks, G, n_networks, proba, l_gen, scale, inter_tmp, basal_tmp, inter, basal, ref_network,
                  weight_prev=.5, loss='CE', final=0):

    n_networks_tmp: np.signedinteger[Any] = 1+np.argmax(ks[1:]) 

    l_pen1 = l_gen * np.size(y_prot, 0) / ((n_networks_tmp)*scale*(1+np.sqrt(G)))
    theta = core_inference(y_samples, y_proba[:, :n_networks_tmp+1], y_prot, y_prot_mod, y_kon, 
                theta_init[:, :n_networks_tmp], theta_ref[:, :n_networks_tmp], ref_network[:, :n_networks_tmp], 
                ks[:n_networks_tmp+1], G, g, n_networks_tmp, proba, 
                l_pen1, weight_prev=weight_prev*(1-final), loss=loss, final=final)
    inter[:, :n_networks_tmp], basal[:n_networks_tmp] = theta[:-1, :], theta[-1, :]
    inter_tmp[:, :n_networks_tmp], basal_tmp[:n_networks_tmp] = theta[:-1, :], theta[-1, :]

    ### Refine inference
    l_pen2 = l_gen / (n_networks_tmp*(1+np.log(G)))
    inter[:, :n_networks_tmp], basal[:n_networks_tmp] = refine_inference(y_samples, 
                            y_proba[:, :n_networks_tmp+1], y_prot, y_prot_mod, y_kon, 
                            inter, basal, ks[:n_networks_tmp+1], G, g, n_networks_tmp, proba, 
                            l_pen2, weight_prev=weight_prev*(1-final), loss=loss, correc_ref=final, final=final)
            
    if n_networks_tmp < n_networks:
        basal[n_networks_tmp:] = -100
        basal_tmp[n_networks_tmp:] = -100

    return basal, inter, basal_tmp, inter_tmp


def inference_network(vect_t, times, y_samples, y_kon, y_proba, y_prot, y_prot_mod, ks, proba=1,
                      ref_network = np.zeros(2),
                      basal_init = np.zeros(2), inter_init = np.zeros(2), 
                      basal_ref = np.zeros(2), inter_ref = np.zeros(2), 
                      scale=100, weight_prev=.5, loss='CE', final=0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Network inference procedure.
    Return the inferred network (basal + network) and the time at which each edge has been detected with strongest intensity.
    """
    # import joblib lazily to avoid errors when the package is not installed
    try:
        from joblib import Parallel, delayed
    except ImportError:
        Parallel = None
        delayed = None
        logging.getLogger(__name__).warning("joblib not available; parallel loops will run sequentially")

    G: int = np.size(y_prot, 1)
    n_networks: int = np.size(ks, 1) - 1

    ### Initialize network
    theta_init = np.zeros((G+1, G, n_networks))
    if np.linalg.norm(inter_init):
        theta_init[:-1, :, :] = inter_init[:, :, :]
    if np.linalg.norm(basal_init):
        theta_init[-1, :, :] = basal_init[:, :]
    
    theta_ref = np.zeros((G+1, G, n_networks))
    if np.linalg.norm(inter_ref):
        theta_ref[:-1, :, :] = inter_init[:, :, :]
    if np.linalg.norm(basal_ref):
        theta_ref[-1, :, :] = basal_init[:, :]

    if np.linalg.norm(ref_network) <= 0:
        ref_network = np.ones_like(inter_init)


    ### Initialize penalization
    l_gen: int = (1+proba)
    inter_tmp, basal_tmp = np.zeros((G, G, n_networks)), np.zeros((G, n_networks))
    inter, basal = np.zeros((G, G, n_networks)), np.zeros((G, n_networks))

    def run_main_loop_for_gene(g):
        return main_loop_inference(g,
            vect_t, 
            y_samples,
            y_proba[:, g],
            y_prot, 
            y_prot_mod[g, :, :],
            y_kon[:, g],
            theta_init[:, g, :],
            theta_ref[:, g, :],
            ks[g], G, n_networks, proba, l_gen,
            scale, 
            inter_tmp[:, g, :],
            basal_tmp[g, :],
            inter[:, g, :],
            basal[g, :],
            ref_network[:, g, :], 
            weight_prev=weight_prev,
            loss=loss,
            final=final,
        )

    if Parallel is not None:
        results = Parallel(n_jobs=-1)(
            delayed(run_main_loop_for_gene)(g) for g in range(1, G)
        )
    else:
        # fallback sequential loop when joblib is unavailable
        results = [run_main_loop_for_gene(g) for g in range(1, G)]

    for idx, g in enumerate(range(1, G)):
        basal[g, :], inter[:, g, :], basal_tmp[g, :], inter_tmp[:, g, :] = results[idx]

    return basal, inter, basal_tmp, inter_tmp
