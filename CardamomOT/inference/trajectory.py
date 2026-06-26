
"""
Core functions for the inference of trajectories, mainly used in loop_trajectories
"""

import numpy as np
from numba import njit, prange
import multiprocessing as mp
from joblib import Parallel, delayed
from .network import main_loss
from .simulations import simulate_next_prot_ode

EPS = 1e-16

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
    result = np.zeros((n_cells, Gm1, n_net + 1))
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


def count_errors(vect_prot, vect_kon, vect_proba, ks, Y, X, loss='CE', compute_with_proba=0, n_stimuli=1):

    N, G = vect_prot.shape
    ns = n_stimuli
    if compute_with_proba:
        proba = base_kon_vector(Y, X, vect_prot)
        cnt_errors = main_loss(proba[:, ns:], vect_proba[:, ns:], 1, loss)
    else:
        kon = kon_ref_vector(vect_prot, ks, X, Y)
        cnt_errors = main_loss(kon[:, ns:], vect_kon[:, ns:], 1, loss)
    return cnt_errors/(N*(G-ns))


@njit(fastmath=True, parallel=True)
def kon_ref_vector(y_prot, kz, theta_inter, theta_basal) -> np.ndarray:
    sigma = base_kon_vector(theta_basal, theta_inter, y_prot)
    out = np.zeros(sigma.shape[:2])  # shape: (n_cells, Gm1)

    for i in prange(sigma.shape[0]):
        for j in prange(sigma.shape[1]):
            for k in prange(sigma.shape[2]):
                out[i, j] += kz[j, k] * sigma[i, j, k]

    return out


def _kon_per_sample(y_prot, ks, inter, basal, samples_data=None):
    """
    kon_ref_vector with per-sample basal support.

    When basal is 2-D (G, n_networks) or samples_data is None, delegates to
    kon_ref_vector directly.  When basal is 3-D (n_samples, G, n_networks),
    routes each cell to its sample's basal slice via samples_data.
    """
    if basal.ndim < 3 or samples_data is None:
        return kon_ref_vector(y_prot, ks, inter, basal)
    samples_id = np.sort(np.unique(samples_data))
    out = np.zeros((y_prot.shape[0], y_prot.shape[1]))
    for s_idx, s in enumerate(samples_id):
        mask = (samples_data == s)
        if not np.any(mask):
            continue
        basal_s = basal[min(s_idx, basal.shape[0] - 1)]
        out[mask] = kon_ref_vector(y_prot[mask], ks, inter, basal_s)
    return out

    
@njit(fastmath=True, parallel=True)
def my_otdistance(vect_kon_init, vect_kon_end, vect_prot_init, vect_rna_init, vect_rna_end,
                            vect_proba_init, vect_proba_end, mode_init, mode_end, alpha, s1, ks, d1, delta_t, basal, inter, loss='CE',
                            compute_with_proba=1, n_iter=1, intensity_prior=1, q=.9,
                            n_stimuli=1, stim_vals=np.ones(1), scale_proteins=1) -> tuple[np.ndarray, np.ndarray]:
    n1, G = vect_rna_init.shape
    n2 = vect_rna_end.shape[0]
    ns = n_stimuli

    # Preallocation (important for Numba)
    dist = np.zeros((n1, n2))
    vect_prot_end = np.ones((n1, n2, G + ns))
    vect_prot_end[:, :, :ns] = stim_vals * scale_proteins
    log_vect_rna_end = np.log1p(vect_rna_end)
    log_vect_rna_init = np.log1p(vect_rna_init)

    combined = np.vstack((log_vect_rna_end, log_vect_rna_init))
    for g in range(G):
        sorted_combined = np.sort(combined[:, g])
        idx = int(q * (n1 + n2 - 1))
        scale_rna = sorted_combined[idx]
        log_vect_rna_init[:, g] /= max(scale_rna, 1)
        log_vect_rna_end[:, g] /= max(scale_rna, 1)

    weight_init: float = (n_iter < intensity_prior) * (1 / n_iter)**(1 - 1/n_iter)

    for i in prange(n1):  # parallelize cell-by-cell

        prot_init_i = vect_prot_init[i]
        log_rna_init_i = log_vect_rna_init[i]
        rna_init_i = vect_rna_init[i]
        mode_init_i = mode_init[i]
        proba_init_i = vect_proba_init[i]
        kon_init_i = vect_kon_init[i]
        alpha_i = alpha[i]

        prot_end_i = np.zeros((n2, G))
        local_dist_i = np.zeros(n2)

        # --- Loop over target cells j ---
        for j in range(0, n2):
            prot_end_i[j, :] = find_next_prot(d1, prot_init_i, rna_init_i * scale_proteins, vect_rna_end[j] * scale_proteins, mode_init_i, mode_end[j], alpha_i, s1, delta_t)

        # --- Storage ---
        vect_prot_end[i, :, ns:] = prot_end_i[:, :]

        for g in range(G):
            sorted_prot = np.sort(prot_end_i[:, g])
            idx = int(q * (n2 - 1))
            scale_prot = sorted_prot[idx]
            prot_end_i[:, g] /= max(scale_prot, 1)
            prot_init_i /= max(scale_prot, 1)

        # --- Main loss ---
        if compute_with_proba:
            sigma = base_kon_vector(basal, inter, vect_prot_end[i])
            for j in range(0, n2):
                diff_proba = vect_proba_end[j] - proba_init_i
                diff_prot = (prot_end_i[j] - prot_init_i) / scale_proteins
                diff_rna = log_vect_rna_end[j] - log_rna_init_i
                local_dist_i[j] = ((1.0 - 1.0 / G) * (np.sum(diff_proba * diff_proba) * weight_init +
                                    main_loss(sigma[j, ns:], vect_proba_end[j], 1, loss) * (1 - weight_init)) +
                                    (0.5 / G) * (np.sum(diff_prot * diff_prot) + np.sum(diff_rna * diff_rna)))
        else:
            sigma = kon_ref_vector(vect_prot_end[i], ks, inter, basal)
            for j in range(0, n2):
                diff_k = vect_kon_end[j] - kon_init_i
                diff_prot = (prot_end_i[j] - prot_init_i) / scale_proteins
                diff_rna = log_vect_rna_end[j] - log_rna_init_i
                local_dist_i[j] = ((1.0 - 1.0 / G) * (np.sum(diff_k * diff_k) * weight_init +
                                    main_loss(sigma[j, ns:], vect_kon_end[j], 1, loss) * (1 - weight_init)) +
                                    (0.5 / G) * (np.sum(diff_prot * diff_prot) + np.sum(diff_rna * diff_rna)))
        
        dist[i, :] = np.minimum(local_dist_i, 100.0)

    return dist, vect_prot_end


def inference_alpha(d1, s1, alpha_init, y_kon_init_true, y_kon_init, y_prot_init, y_rna_init,
                    y_kon_end_true, y_kon_end, y_prot_end, y_rna_end, mode_init, mode_end,
                    basal, inter, ks, delta_t, tol=1.0, n_pas=25, samples_data=None, stim_vals=np.ones(1), scale_proteins=1):
    
    ns = len(stim_vals)
    y_prot = np.ones_like(y_rna_end)
    y_prot[:, :ns] = stim_vals * scale_proteins
    diff_kon = np.abs(y_kon_end_true[:, ns:] - y_kon_init_true[:, ns:]) / max(tol, EPS)
    weights = np.minimum(diff_kon ** (1/tol), 1)

    alpha = alpha_init.copy()
    cnt = np.zeros_like(alpha_init, dtype=int)
    t = 1 / n_pas

    while t < 1 and np.min(cnt) < 1:
        y_prot[:, ns:] = find_next_prot(d1, y_prot_init[:, ns:], y_rna_init[:, ns:] * scale_proteins, 
                    y_rna_end[:, ns:] * scale_proteins, mode_init, mode_end, 
                    np.minimum(alpha / t, 1.0), s1, t * delta_t)
        kon_new = _kon_per_sample(y_prot, ks, inter, basal, samples_data=samples_data)  # shape (N, G)

        # Condition: if abs(kon_new[i] - y_kon_init[i]) > abs(kon_new[i] - y_kon_end[i])
        # Only apply where cnt < 1
        diff_init = np.abs(kon_new[:, ns:] - y_kon_init[:, ns:])
        diff_end = np.abs(kon_new[:, ns:] - y_kon_end[:, ns:])
        bool_var = (cnt == 0) & (diff_init > diff_end)
        alpha[bool_var] = alpha[bool_var] * (1 - weights[bool_var]) + t * weights[bool_var]
        cnt[bool_var] = 1

        t += 1 / n_pas

    return alpha


def filter_network(T, N_traj, prot_traj, ks, basal_ref, inter_ref,
                   seuil_intensity=5e-2, seuil_variations=1e-2, n_order=10, samples_data=None):

    if basal_ref.ndim == 3:
        n_samples, G, n_networks = basal_ref.shape
    else:
        G, n_networks = basal_ref.shape

    kon_vector = _kon_per_sample(prot_traj, ks, inter_ref, basal_ref, samples_data=samples_data)

    def core_filter(inter_ref, kon_vector, genes_list):
        inter_t = np.zeros((T, G, G, n_networks))
        variations = np.zeros((n_networks, G, G, T))
        variations_ref = np.zeros((n_networks, G, G, T))
        inter_tmp = inter_ref.copy()
        for g1 in genes_list:
            for n in range(n_networks):
                inter_tmp[g1, :, n] = 0
                kon_vector_nog1 = _kon_per_sample(prot_traj, ks, inter_tmp, basal_ref, samples_data=samples_data)
                for g2 in range(0, G):
                    val = abs(inter_ref[g1, g2, n])
                    if val >= seuil_intensity:
                        diff = (kon_vector_nog1.reshape(T, N_traj, G) - 
                                            kon_vector.reshape(T, N_traj, G))**2
                        variations_ref[n, g1, g2, :] = diff[:, :, g2].max(axis=1)
                        variations[n, g1, g2, 1:] = np.abs(np.diff(variations_ref[n, g1, g2, :]))
                        variations[n, g1, g2, 0] = abs(variations_ref[n, g1, g2, 0])
                        max_val = np.max(variations[n, g1, g2, :])
                        if max_val >= seuil_variations/np.sqrt(G):
                            tmax: int = np.argmax(variations[n, g1, g2, :])
                            inter_t[tmax:, g1, g2, n] = inter_ref[g1, g2, n]
                            inter_tmp[g1, g2, n] = inter_ref[g1, g2, n]
        return inter_t

    try:
        from joblib import Parallel, delayed
    except ImportError:
        Parallel = None
        delayed = None
        logging.getLogger(__name__).warning("joblib not available; parallel loops will run sequentially")

    def single_run(_):
        genes_list = np.random.permutation(G)
        inter_t_run = core_filter(inter_ref, kon_vector, genes_list)
        return inter_t_run

    n_order = max(n_order, G)
    results = Parallel(n_jobs=-1)(delayed(single_run)(i) for i in range(n_order))

    # Agregation
    stacked = np.stack(results, axis=0)  # (n_order, T, G, G, n_networks)
    inter_t = np.zeros((T, G, G, n_networks))
    pos_mask = stacked != 0                         # (n_order, T, G, G, n_networks)
    last_positive_i = np.where(pos_mask, np.arange(n_order)[:, None, None, None, None], -1).argmax(axis=0)
    inter_t = stacked[last_positive_i, 
                    np.arange(T)[:, None, None, None],
                    np.arange(G)[None, :, None, None],
                    np.arange(G)[None, None, :, None],
                    np.arange(n_networks)[None, None, None, :]]
    inter_t[~pos_mask.any(axis=0)] = 0

    # Filtre
    inter = inter_ref * (inter_t[-1] != 0)
    
    return inter, inter_t


def filter_network_fancy(
    T, N_traj, prot_traj, ks, basal_ref, inter_ref,
    seuil_intensity=5e-2, seuil_variations=5e-2,
    seuil_synergy=1.5,
    samples_data=None,
    n_jobs=-1,
):
    """
    Structured three-pass edge filtering that avoids both the greedy
    order-dependence problem and the silent-synergy problem.

    Pass 1 — Canonical individual scoring
        Each edge is evaluated independently against the full reference
        network. Produces a continuous impact score per edge.

    Pass 2 — Conditional re-evaluation of rejected edges
        Edges rejected in pass 1 are re-evaluated inside the filtered
        network (without other rejected edges). An edge that was silent
        in the full network but becomes significant once dominant edges
        are removed was being masked — it is reinstated.

    Pass 3 — Pairwise synergy check among weak edges
        Among edges with low but non-zero individual scores, pairs are
        tested jointly. If removing both edges together produces an impact
        significantly greater than the sum of their individual impacts
        (super-additivity), both edges are retained.

    Parameters
    ----------
    T : int
        Number of time steps.
    N_traj : int
        Number of trajectories.
    prot_traj : array
        Protein trajectories, shape (T * N_traj, G).
    ks : array
        Kinetic parameters.
    basal_ref : array
        Basal transcription rates, shape (G, n_networks) or
        (n_samples, G, n_networks).
    inter_ref : array
        Reference interaction matrix, shape (G, G, n_networks).
    seuil_intensity : float
        Minimum absolute edge weight to enter evaluation at all.
    seuil_variations : float
        Minimum variation score to retain an edge (main threshold).
    seuil_synergy : float
        Multiplicative factor above which joint impact is considered
        super-additive. An edge pair (e1, e2) is flagged as synergistic
        if impact(e1 + e2 removed) > seuil_synergy * (score(e1) + score(e2)).
        Default 2.0 (joint impact must be at least twice the additive sum).
    samples_data : optional
        Additional per-sample data passed to _kon_per_sample.
    n_jobs : int
        Number of parallel jobs for joblib (-1 = all cores).

    Returns
    -------
    inter : array, shape (G, G, n_networks)
        Filtered interaction matrix (binary mask applied to inter_ref).
    inter_t : array, shape (T, G, G, n_networks)
        Time-resolved interaction tensor.
    report : dict
        Diagnostic report with keys:
        - 'scores_pass1'   : raw scores from pass 1, shape (G, G, n_networks)
        - 'reinstated'     : list of edges reinstated by pass 2
        - 'synergistic'    : list of edge pairs flagged as synergistic in pass 3
        - 'retained_mask'  : final boolean retention mask
    """
    import numpy as np
    import logging
    from itertools import combinations

    log = logging.getLogger(__name__)

    try:
        from joblib import Parallel, delayed
    except ImportError:
        Parallel = delayed = None
        log.warning("joblib not available; falling back to sequential evaluation")

    if basal_ref.ndim == 3:
        n_samples, G, n_networks = basal_ref.shape
    else:
        G, n_networks = basal_ref.shape

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _compute_impact(g1, g2, n, context_ref, kon_context):
        """
        Measure the functional impact of edge (g1 -> g2, network n) within
        context_ref by comparing kon trajectories with and without the edge.

        Returns
        -------
        score : float
            Maximum temporal variation of the squared kon difference.
            Zero if the edge does not pass seuil_intensity.
        tmax : int or None
            Time index of maximum variation (used to build inter_t).
        """
        if abs(context_ref[g1, g2, n]) < seuil_intensity:
            return 0.0, None

        inter_tmp = context_ref.copy()
        inter_tmp[g1, g2, n] = 0

        kon_no_edge = _kon_per_sample(
            prot_traj, ks, inter_tmp, basal_ref, samples_data=samples_data
        )

        diff_sq = (
            kon_no_edge.reshape(T, N_traj, G)
            - kon_context.reshape(T, N_traj, G)
        ) ** 2
        impact_over_time = diff_sq[:, :, g2].max(axis=1)  # (T,)

        variations = np.empty(T)
        variations[0]  = abs(impact_over_time[0])
        variations[1:] = np.abs(np.diff(impact_over_time))

        score = float(np.max(variations))
        tmax  = int(np.argmax(variations))
        return score, tmax

    def _compute_joint_impact(edge_a, edge_b, context_ref, kon_context):
        """
        Measure the joint impact of simultaneously removing two edges.
        Used in pass 3 to detect super-additive (synergistic) pairs.

        Returns
        -------
        joint_score : float
            Max temporal variation when both edges are absent.
        """
        (g1a, g2a, na), (g1b, g2b, nb) = edge_a, edge_b

        inter_tmp = context_ref.copy()
        inter_tmp[g1a, g2a, na] = 0
        inter_tmp[g1b, g2b, nb] = 0

        kon_no_pair = _kon_per_sample(
            prot_traj, ks, inter_tmp, basal_ref, samples_data=samples_data
        )

        # Measure impact on both target genes and take the max
        joint_score = 0.0
        for g2 in {g2a, g2b}:
            diff_sq = (
                kon_no_pair.reshape(T, N_traj, G)
                - kon_context.reshape(T, N_traj, G)
            ) ** 2
            impact = diff_sq[:, :, g2].max(axis=1)
            variations = np.empty(T)
            variations[0]  = abs(impact[0])
            variations[1:] = np.abs(np.diff(impact))
            joint_score = max(joint_score, float(np.max(variations)))

        return joint_score

    def _par(fn, items):
        """Run fn over items in parallel (or sequentially if joblib missing)."""
        if Parallel is not None:
            return Parallel(n_jobs=n_jobs)(delayed(fn)(x) for x in items)
        return [fn(x) for x in items]

    # ------------------------------------------------------------------
    # Pass 1 — canonical individual scoring (full reference context)
    # ------------------------------------------------------------------
    log.debug("Pass 1: canonical individual scoring")

    kon_full = _kon_per_sample(
        prot_traj, ks, inter_ref, basal_ref, samples_data=samples_data
    )

    all_edges = [
        (g1, g2, n)
        for g1 in range(G)
        for g2 in range(G)
        for n  in range(n_networks)
    ]

    pass1_results = _par(
        lambda e: _compute_impact(*e, inter_ref, kon_full),
        all_edges,
    )

    scores_p1 = np.zeros((G, G, n_networks))
    tmax_p1   = {}
    for (g1, g2, n), (score, tmax) in zip(all_edges, pass1_results):
        scores_p1[g1, g2, n] = score
        if tmax is not None:
            tmax_p1[(g1, g2, n)] = tmax

    retained_p1 = {e for e, (s, _) in zip(all_edges, pass1_results)
                   if s >= seuil_variations/G}
    rejected_p1 = {e for e, (s, _) in zip(all_edges, pass1_results)
                   if 0 < s < seuil_variations/G}   # weak but non-zero
    silent_p1   = {e for e, (s, _) in zip(all_edges, pass1_results)
                   if s == 0.0}                    # zero score (intensity gate)

    log.debug(
        f"Pass 1 — retained: {len(retained_p1)}, "
        f"weak: {len(rejected_p1)}, silent: {len(silent_p1)}"
    )

    # ------------------------------------------------------------------
    # Pass 2 — conditional re-evaluation of rejected edges
    #
    # Context: the filtered network (only pass-1 retained edges active).
    # An edge that was dominated in the full network may become significant
    # once stronger edges are removed.
    # ------------------------------------------------------------------
    log.debug("Pass 2: conditional re-evaluation of rejected edges")

    # Build the reduced context: only edges retained in pass 1
    inter_filtered = inter_ref * np.isin(
        np.arange(inter_ref.size).reshape(inter_ref.shape),
        [all_edges.index(e) for e in retained_p1],
    ).astype(float)
    # Cleaner equivalent: mask directly
    mask_p1 = np.zeros((G, G, n_networks), dtype=bool)
    for e in retained_p1:
        mask_p1[e] = True
    inter_filtered = inter_ref * mask_p1

    kon_filtered = _kon_per_sample(
        prot_traj, ks, inter_filtered, basal_ref, samples_data=samples_data
    )

    # Re-evaluate only the rejected (non-silent) edges in this reduced context
    rejected_list = list(rejected_p1)
    pass2_results = _par(
        lambda e: _compute_impact(*e, inter_filtered, kon_filtered),
        rejected_list,
    )

    reinstated   = []
    scores_p2    = {}
    tmax_p2      = {}
    for e, (score, tmax) in zip(rejected_list, pass2_results):
        scores_p2[e] = score
        if score >= seuil_variations/G:
            reinstated.append(e)
            retained_p1.add(e)          # promote to retained set
            if tmax is not None:
                tmax_p2[e] = tmax
            log.debug(f"  Reinstated edge {e} (score: {score:.4f})")

    log.debug(f"Pass 2 — reinstated: {len(reinstated)}")

    # ------------------------------------------------------------------
    # Pass 3 — pairwise synergy check among still-weak edges
    #
    # Among edges not yet retained, test all pairs jointly. A pair is
    # synergistic (both retained) if their joint impact exceeds
    # seuil_synergy × (sum of individual scores).
    # ------------------------------------------------------------------
    log.debug("Pass 3: pairwise synergy check")

    # Still-weak edges: rejected in pass 1, not reinstated in pass 2,
    # but with a non-zero individual score (some signal to work with)
    still_weak = [
        e for e in rejected_p1
        if e not in retained_p1 and scores_p1[e] > 0
    ]

    synergistic_pairs = []

    if len(still_weak) >= 2:
        # Evaluate all pairs against the full reference context (kon_full),
        # so the joint impact is measured in the richest possible setting
        pairs = list(combinations(still_weak, 2))

        joint_scores = _par(
            lambda pair: _compute_joint_impact(pair[0], pair[1], inter_ref, kon_full),
            pairs,
        )

        for (ea, eb), joint in zip(pairs, joint_scores):
            additive_sum = scores_p1[ea] + scores_p1[eb]
            if additive_sum > 0 and joint >= seuil_synergy * additive_sum:
                synergistic_pairs.append((ea, eb))
                retained_p1.add(ea)
                retained_p1.add(eb)
                log.debug(
                    f"  Synergistic pair {ea} + {eb}: "
                    f"joint={joint:.4f}, additive={additive_sum:.4f}"
                )

    log.debug(f"Pass 3 — synergistic pairs: {len(synergistic_pairs)}")

    # ------------------------------------------------------------------
    # Build the final inter_t tensor using the best available tmax
    # ------------------------------------------------------------------
    inter_t = np.zeros((T, G, G, n_networks))

    for e in retained_p1:
        g1, g2, n = e
        # Priority: pass-1 tmax > pass-2 tmax > fallback to 0
        tmax = tmax_p1.get(e, tmax_p2.get(e, 0))
        inter_t[tmax:, g1, g2, n] = inter_ref[g1, g2, n]

    # Final binary mask applied to inter_ref
    inter = inter_ref * (inter_t[-1] != 0)

    report = {
        "scores_pass1"  : scores_p1,
        "reinstated"    : reinstated,
        "synergistic"   : synergistic_pairs,
        "retained_mask" : inter_t[-1] != 0,
    }

    return inter, inter_t