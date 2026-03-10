
"""
Core functions for the inference of trajectories, mainly used in loop_trajectories
"""

from typing import Any
import numpy as np
from numba import njit, prange
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
# joblib is optional: import lazily to avoid breaking imports when it's absent
try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None
    import logging
    logging.getLogger(__name__).warning("joblib missing: parallel loops disabled")
from .network import main_loss
from .simulations import simulate_next_prot_ode


def minimal_repetition_choice(N, M, seed=None):
    if seed is not None:
        np.random.seed(seed)

    result = []
    full_cycles = M // N
    remainder = M % N
    
    for _ in range(full_cycles):
        cycle = np.arange(N)
        np.random.shuffle(cycle)
        result.append(cycle)
    
    if remainder > 0:
        remainder_sample = np.random.choice(N, remainder, replace=False)
        result.append(remainder_sample)
    
    return np.concatenate(result)


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
def find_next_prot_mixed(d1, P0, M0, M1, mode_init, mode_end, alpha, s, delta_t):
    """
    Deterministic flow interpolating between two points
    """

    P_int = (P0 - s * (M0 + (M0 - mode_init) / d1)) * np.exp(-d1 * delta_t * alpha) + s * (M0 + (mode_init - M0) * (alpha * delta_t * d1 - 1) / d1)
    return (P_int - s * (mode_end + (mode_end - M1) / d1)) * np.exp(-d1 * delta_t * (1-alpha)) + s * (mode_end + (M1 - mode_end) * ((1-alpha) * delta_t * d1 - 1) / d1)


@njit
def find_next_prot(d1, P0, M0, M1, mode_init, mode_end, alpha, s, delta_t):
    """
    Deterministic flow interpolating between two points
    """
    
    Pint = mode_init * s + (P0 - mode_init * s) * np.exp(-d1*delta_t*alpha)
    return mode_end * s + (Pint - mode_end * s) * np.exp(-d1*delta_t*(1-alpha))


def count_errors(vect_prot, vect_kon, vect_proba, ks, Y, X, loss='CE', compute_with_proba=0):

    N, G = vect_prot.shape
    if compute_with_proba:
        proba = base_kon_vector(Y, X, vect_prot)
        cnt_errors = main_loss(proba[:, 1:], vect_proba[:, 1:], 1, loss)
    else:
        kon = kon_ref_vector(vect_prot, ks, X, Y)
        cnt_errors = main_loss(kon[:, 1:], vect_kon[:, 1:], 1, loss)
    return cnt_errors/(N*(G-1))


@njit(fastmath=True, parallel=True)
def kon_ref_vector(y_prot, kz, theta_inter, theta_basal) -> np.ndarray:
    sigma = base_kon_vector(theta_basal, theta_inter, y_prot)
    out = np.zeros(sigma.shape[:2])  # shape: (n_cells, Gm1)

    for i in prange(sigma.shape[0]):
        for j in prange(sigma.shape[1]):
            for k in prange(sigma.shape[2]):
                out[i, j] += kz[j, k] * sigma[i, j, k]
    
    return out

    
@njit(fastmath=True, parallel=True)
def my_otdistance(vect_kon_init, vect_kon_end, vect_prot_init, vect_rna_init, vect_rna_end,
                            vect_proba_init, vect_proba_end, mode_init, mode_end, alpha, s1, ks, d1, delta_t, basal, inter, loss='CE',
                            compute_with_proba=1, n_iter=1, intensity_prior=1) -> tuple[np.ndarray, np.ndarray]:
    n1, G = vect_rna_init.shape
    n2 = vect_rna_end.shape[0]

    # Préallocation (important pour Numba)
    dist = np.ones((n1, n2))
    vect_prot_end = np.ones((n1, n2, G + 1))

    weight_init: float = 1 / n_iter

    for i in prange(n1):  # parallélisation cellule par cellule
        prot_init_i = vect_prot_init[i]
        rna_init_i = vect_rna_init[i]
        mode_init_i = mode_init[i]
        proba_init_i = vect_proba_init[i]
        kon_init_i = vect_kon_init[i]
        alpha_i = alpha[i]

        local_prot_end = np.empty((n2, G))
        local_dist = np.zeros(n2)

        # --- Boucle sur les cibles j ---
        for j in range(n2):
            prot_end = find_next_prot(d1, prot_init_i, rna_init_i, vect_rna_end[j], mode_init_i, mode_end[j], alpha_i, s1, delta_t)
            local_prot_end[j] = prot_end

            if n_iter <= intensity_prior:
                # distances sur proba ou modes et protéines
                diff_prot = prot_end - prot_init_i
                if compute_with_proba:
                    diff_p = vect_proba_end[j] - proba_init_i
                    local_dist[j] += ((1.0 - 1.0 / G) * np.sum(diff_p * diff_p) +
                                (1.0 / G) * np.sum(diff_prot * diff_prot)) * weight_init
                else:
                    diff_k = vect_kon_end[j] - kon_init_i
                    local_dist[j] += ((1.0 - 1.0 / G) * np.sum(diff_k * diff_k) +
                                (1.0 / G) * np.sum(diff_prot * diff_prot)) * weight_init

        # --- Stockage ---
        vect_prot_end[i, :, 1:] = local_prot_end

        # --- Correction avec main_loss ---
        if n_iter > 1:
            if compute_with_proba:
                sigma = base_kon_vector(basal, inter, vect_prot_end[i]) 
                for j in range(n2):
                    local_dist[j] += main_loss(sigma[j, 1:], vect_proba_end[j], 1, loss) * (1 - weight_init)
            else:
                sigma = kon_ref_vector(vect_prot_end[i], ks, inter, basal) 
                for j in range(n2):
                    local_dist[j] += main_loss(sigma[j, 1:], vect_kon_end[j], 1, loss) * (1 - weight_init)

        # --- Clamp et copie ---
        for j in range(n2):
            val = local_dist[j]
            dist[i, j] = 100.0 if val > 100.0 else val

    return dist, vect_prot_end



def my_otdistance_simulated(vect_prot_init, vect_rna_init, vect_rna_end,
                            vect_proba_end, s1, ks, d1, delta_t, basal, inter) -> tuple[np.ndarray, np.ndarray]:
    
    G = vect_rna_init.shape[1]
    n1, n2 = vect_rna_init.shape[0], vect_rna_end.shape[0]
    dist = np.ones((n1, n2))
    vect_prot_end = np.ones((n1, n2, G + 1))

    def run_main_loop_for_cell(i):
        d1_aug = np.concatenate(([1.0], d1))   # shape (G+1,)
        P0_aug = np.concatenate(([1.0], vect_prot_init[i]))   # shape (G+1,)
        prot_end = simulate_next_prot_ode(d1_aug, ks, basal, inter, delta_t, 1, P0=P0_aug).p[-1]
        return prot_end

    if Parallel is not None:
        results = Parallel(n_jobs=-1)(
            delayed(run_main_loop_for_cell)(i) for i in range(0, n1)
        )
    else:
        # fallback sequential loop
        results = [run_main_loop_for_cell(i) for i in range(0, n1)]

    for i in range(n1):
        prot_end = results[i]
        for j in range(n2):
            vect_prot_end[i, j, 1:] = prot_end[:]
        sigma = base_kon_vector(basal, inter, vect_prot_end[i])
        for j in range(n2):
            dist[i, j] += np.sum((sigma[j, 1:] - vect_proba_end[j]) ** 2)

    return dist, vect_prot_end



def inference_alpha(d1, s1, alpha_init, y_kon_init_true, y_kon_init, y_prot_init, y_rna_init, 
                    y_kon_end_true, y_kon_end, y_prot_end, y_rna_end, mode_init, mode_end,
                    basal, inter, ks, delta_t, tol=.5, n_pas=25):
    
    N, G = y_prot_init.shape
    alpha = alpha_init.copy()
    cnt = np.zeros((N, G-1))
    t: float = 1 / n_pas

    y_prot = np.ones_like(y_rna_end)

    ### We don't modify alphas that are not characterizing an enough important difference or for which the inference was bad
    diff_kon = (np.abs(y_kon_end_true[:, 1:] - y_kon_init_true[:, 1:]) < tol)
    cnt[diff_kon] = 1

    while t < 1 and np.min(cnt) < 1:
        alpha_tmp = alpha.copy() / t
        alpha_tmp = np.minimum(alpha_tmp, 1.0)  # shape (N, G)

        y_prot[:, 1:] = find_next_prot(d1, y_prot_init[:, 1:], y_rna_init[:, 1:], y_rna_end[:, 1:], mode_init, mode_end, alpha_tmp, s1, t * delta_t)

        kon_new = kon_ref_vector(y_prot, ks, inter, basal)  # shape (N, G)

        # Condition: if abs(kon_new[i] - y_kon_init[i]) > abs(kon_new[i] - y_kon_end[i])
        # Only apply where cnt < 1
        diff_init = np.abs(kon_new[:, 1:] - y_kon_init[:, 1:])
        diff_end = np.abs(kon_new[:, 1:] - y_kon_end[:, 1:])
        bool_var = (cnt < 1) & (diff_init > diff_end)
        alpha[bool_var] = t
        cnt[bool_var] = 1

        t += 1 / n_pas

    # print(np.sum(np.abs(alpha  - alpha_init))/y_prot_init[:, 1:].size, np.mean(alpha), np.mean(alpha_init), delta_t)

    return alpha


@njit(fastmath=True, parallel=True)
def argmax_numba(a) -> int:
    """Manual argmax compatible with Numba."""
    max_val = a[0]
    max_idx = 0
    for i in prange(1, a.shape[0]):
        if a[i] > max_val:
            max_val = a[i]
            max_idx: int = i
    return max_idx


# @njit(fastmath=True, parallel=True)
def filter_network(T, N_traj, prot_traj, ks, basal_init, basal_t_init, inter_init, inter_t_init,
                   seuil_intensity=5e-2, seuil_variations=.2, n_order=10):
    
    G, n_networks = basal_init.shape

    inter_ref = inter_t_init[-1].copy()
    basal_ref = basal_t_init[-1].copy()
    inter_temp = inter_t_init[-1].copy()

    kon_vector = kon_ref_vector(prot_traj, ks, inter_ref, basal_ref)
    genes_list = np.arange(G)

    inter_t = np.zeros((T, G, G, n_networks))

    @njit(fastmath=True, parallel=True)
    def core_filter(inter_ref, inter_temp, inter_t, kon_vector, genes_list):
        variations = np.zeros((n_networks, G, G, T))
        variations_ref = np.zeros((n_networks, G, G, T))
        inter_tmp = inter_ref.copy()
        for g1 in genes_list:
            for n in range(n_networks):
                inter_tmp[g1, :, n] = 0
                kon_vector_nog1 = kon_ref_vector(prot_traj, ks, inter_tmp, basal_ref)
                for g2 in prange(0, G):
                    val = abs(inter_ref[g1, g2, n])
                    if val >= seuil_intensity:
                        inter_tmp[g1, g2, n] = 0
                        for t in range(T):
                            start = N_traj * t
                            end = N_traj * (t + 1)
                            diff = (kon_vector_nog1[start:end, g2] - kon_vector[start:end, g2])**2
                            variations_ref[n, g1, g2, t] = np.max(diff)
                            if t > 0:
                                variations[n, g1, g2, t] = abs(variations_ref[n, g1, g2, t] - variations_ref[n, g1, g2, t-1])
                        max_val = np.max(variations[n, g1, g2, :])
                        if max_val >= seuil_variations / G:#(1 + np.sum(np.minimum(1, np.abs(inter_tmp[:, g2, :]))) + np.sum(np.minimum(1, np.abs(basal_ref[g2])))):
                            tmax: int = argmax_numba(variations[n, g1, g2, :])
                            for t in range(tmax, T):
                                inter_t[t, g1, g2, n] = inter_temp[g1, g2, n]
                            inter_tmp[g1, g2, n] = inter_ref[g1, g2, n]
        return inter_t

    for _ in range(0, min(n_order, G)):  # G attempts to not depend too much on order
        genes_list = np.random.choice(genes_list, G, replace=False)
        inter_t = core_filter(inter_ref, inter_temp, inter_t, kon_vector, genes_list)
    
    return inter_init, inter_t