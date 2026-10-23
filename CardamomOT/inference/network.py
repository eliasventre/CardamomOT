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
from .sigma import SPEC_PHI, SPEC_LINK, SPEC_KAPPA, SPEC_PBAR, PHI_IDENTITY, LINK_LOGISTIC, logistic, logistic_pair, features

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
FLAT_TARGET_TOL = 1e-6  # a sample whose sigmoid target of a gene spans less is single-mode for it (left out of its fit)


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

### Cross-entropy between predicted and target sigmoids (both in [0, 1])
@njit(fastmath=True, cache=True)
def cross_entropy(y_pred, y_true, eps=eps_CE):
    """Element-wise cross-entropy relative to the target (0 when y_pred = y_true)."""
    y_pred_c = np.clip(y_pred, eps, 1 - eps)
    y_true_c = np.clip(y_true, eps, 1 - eps)
    return -(y_true * np.log(y_pred_c / y_true_c) + (1 - y_true) * np.log((1-y_pred_c)/(1-y_true_c)))


@njit(fastmath=True, cache=True)
def main_loss(y_pred, y_true, l, eps=eps_CE):
    return l * np.sum(cross_entropy(y_pred, y_true, eps))


@njit(fastmath=True, cache=True)
def grad_main_loss(y_pred, y_true, l, eps=eps_CE):
    y_clipped = np.clip(y_pred, eps, 1 - eps)

    # Gradient: -(y_true/y - (1-y_true)/(1-y))
    denominator = y_clipped * (1 - y_clipped)
    grad = l * (y_clipped - y_true) / denominator
    mask = (y_pred >= eps) & (y_pred <= 1 - eps)

    return grad * mask


@njit(fastmath=True, cache=True)
def base_kon(theta_basal, theta_inter, y_prot) -> np.ndarray:
    """sigma per cell of one target, logistic of Z = basal + Y @ inter (scalar basal, inter (K,)), Y (N, K) the
    regulator features (phi of sigma.py, the protein levels for a logistic target)."""
    n_cells, G = y_prot.shape
    result = np.empty(n_cells)
    for i in range(n_cells):
        z = theta_basal
        for g in range(G):
            z += y_prot[i, g] * theta_inter[g]
        result[i] = logistic_pair(z)[1]
    return result


def sigmoid_amplitudes(ks):
    """Sigmoid bounds (..., 2) = (lowest mode, highest mode) of mixture amplitudes ks (..., n_modes):
    kon = k_low + (k_high - k_low) * sigmoid(Z), intermediate modes lie on the slope (unstable)."""
    ks = np.asarray(ks, dtype=float)
    return np.stack([ks[..., 0], ks.max(axis=-1)], axis=-1)


def sigmoid_target(kon, k_low, k_high):
    """Target sigmoid of a kon: its position between the lowest (0) and the highest (1) mode, clipped to [0, 1];
    0.5 when both modes are equal (no information)."""
    span = k_high - k_low
    return np.where(span > EPS, np.clip((kon - k_low) / np.maximum(span, EPS), 0.0, 1.0), 0.5)


def gene_weights(ks):
    """Weight of each gene in the trajectory costs, w = 1 - k_low / k_high (0 when both modes are equal);
    ks (..., n_modes) mixture amplitudes."""
    kb = sigmoid_amplitudes(ks)
    return np.clip(1.0 - kb[..., 0] / np.maximum(kb[..., 1], EPS), 0.0, 1.0)


def _of_cond(arr, cond, s_idx):
    """Slice of the condition of sample s_idx when arr is per condition (C, ...), else arr."""
    return arr if cond is None else arr[cond[int(s_idx)]]


def _compute_sigma_per_sample(theta_inter, theta_basal_all, ref_network, yp, ys, us):
    """Compute sigma = sigmoid(Z) for all cells, using per-sample basal from theta_basal_all.

    theta_basal_all : (n_samples,)
    Returns sigma : (N_cells,)
    """
    N_cells = yp.shape[0]
    sigma = np.empty(N_cells)
    inter_ref = theta_inter * ref_network  # (G,)
    for s in us:
        s_idx = int(s)
        mask = (ys == s)
        if np.any(mask):
            sigma[mask] = base_kon(theta_basal_all[s_idx], inter_ref, yp[mask])
    return sigma


def _sigma_loss(sigma, yt):
    """Cross-entropy of the sigmoid against its target yt (sigmoid_target of the target kon): symmetric ON / OFF."""
    return main_loss(sigma, yt, 1)


class DistTerm:
    """
    Distribution term of the end-state fit (weight_network_distribution): lam x sum over the groups (sample, time) of
    sum_i (sigma_(i) - q_i)^2, sigma_(i) the sorted sigmas of the group and q_i the quantiles (i - 1/2) / n of the
    levels of its cells under their mean posterior, i.e. the 1-D Wasserstein-2 between the predicted sigmas and the
    levels of the cells (the network keeps the spread of the targets, not only their mean per cell).
    grp (N,): group of each row (-1: no real cell, left out); dists {group: (sorted levels, cdf)}.
    """

    def __init__(self, grp, dists, lam):
        self.grp, self.dists, self.lam = grp, dists, lam

    def __getitem__(self, idx):
        return DistTerm(self.grp[idx], self.dists, self.lam)

    def quantiles(self, k, n):
        lv, cdf = self.dists[k]
        return lv[np.minimum(np.searchsorted(cdf, (np.arange(n) + 0.5) / n), len(lv) - 1)]


def dist_term(grp, post, levels, lam):
    """DistTerm of one target from the group (N,), posteriors (N, L) and sigmoid targets of the levels (N, L) of the
    real cell of each row; None for a target of at most two levels (its distribution is the fraction ON, already
    given by the cross-entropy)."""
    on = grp >= 0
    if not np.any(on) or lam <= 0 or len(np.unique(np.round(levels[on], 6))) <= 2:
        return None
    dists = {}
    for k in np.unique(grp[on]):
        m = grp == k
        lv, p = levels[m][0], post[m].mean(axis=0)
        o = np.argsort(lv)
        cdf = np.cumsum(p[o])
        dists[k] = (lv[o], cdf / max(cdf[-1], EPS))
    return DistTerm(np.asarray(grp), dists, float(lam))


def _sl(d, mask):
    """Rows mask of a DistTerm (None stays None)."""
    return None if d is None else d[mask]


def _dist_loss_grad(sig, d):
    """Distribution term (DistTerm d) of the sigmas sig (n,): (loss, d loss / d sig (n,))."""
    loss, dl = 0.0, np.zeros(len(sig))
    for k in np.unique(d.grp):
        if k < 0:
            continue
        m = np.flatnonzero(d.grp == k)
        o = m[np.argsort(sig[m], kind='stable')]
        r = sig[o] - d.quantiles(k, len(m))
        loss += float(r @ r)
        dl[o] = 2.0 * r
    return d.lam * loss, d.lam * dl


def _end_loss(sigma, yt, d=None):
    """End-state term: cross-entropy (+ distribution term)."""
    return _sigma_loss(sigma, yt) + (0.0 if d is None else _dist_loss_grad(sigma, d)[0])


def _end_grad_terms(sigma, yt, d=None):
    """d end-state term / d logit per cell."""
    tmp = _sigma_grad_terms(sigma, yt)
    return tmp if d is None else tmp + _dist_loss_grad(sigma, d)[1] * sigma * (1 - sigma)


class PathNodes:
    """
    Flow matching for one target (term of weight weight_flow_matching, the end-state term keeping
    1 - weight_flow_matching): states of its regulators at nodes drawn uniformly along the interval ending at each row,
    nodes (N, Q, K), weights (N, Q + 1) (column 0 for the end state, the row itself, then the nodes: 0 then 1 / Q, or
    1 on the end state for the rows without previous interval), and the kon of the target in force at each node,
    kon (N, Q) (mode of the start state before the switch alpha of the target, of the end state after it);
    targets (N, Q): their sigmoid targets (set by inference_network).
    """

    def __init__(self, nodes, weights, kon=None, targets=None):
        self.nodes = nodes
        self.weights = weights
        self.kon = kon
        self.targets = targets

    def __getitem__(self, idx):
        return PathNodes(self.nodes[idx], self.weights[idx], None if self.kon is None else self.kon[idx],
                         None if self.targets is None else self.targets[idx])

    def __len__(self):
        return len(self.weights)


def _fm_weight(weight_flow_matching, ypm):
    """Weight of the flow-matching term: 0 without flow-matching states (ypm None, e.g. the final fit)."""
    return weight_flow_matching if ypm is not None else 0.0


class MemStates:
    """
    Regulator states of the end-state term with the mRNA memory of the target (targets_accounting_late_mrna): end
    (N, K) states of the rows, nodes (N, Q, K) earlier states along the interval ending at each row, weights (N, Q + 1)
    of the end state (column 0) and of the nodes in the memory d0 exp(-d0 (t - s)) of the target's mRNA (sum 1, 1 on
    the end state for the rows without previous interval): the network term compares sum_q w_q sigma(Y_q) to the target.
    """

    def __init__(self, end, nodes, weights):
        self.end, self.nodes, self.weights = end, nodes, weights

    def __getitem__(self, idx):
        return MemStates(self.end[idx], self.nodes[idx], self.weights[idx])

    def __len__(self):
        return len(self.end)


def _end(Y):
    """End states of Y (MemStates) or Y itself."""
    return Y.end if isinstance(Y, MemStates) else Y


def _mem_terms(basal_s, th, Y):
    """sigma averaged on the mRNA memory and its terms: (sigma_bar (n,), [(w_q (n,), sigma_q (n,), Y_q (n, K))])."""
    terms = [(Y.weights[:, 0], Y.end)] + [(Y.weights[:, q + 1], Y.nodes[:, q]) for q in range(Y.nodes.shape[1])]
    out, sig_bar = [], np.zeros(len(Y))
    for w, Yq in terms:
        sq = logistic(basal_s + Yq @ th)
        sig_bar += w * sq
        out.append((w, sq, Yq))
    return np.clip(sig_bar, 0.0, 1.0), out


def _mem_loss_grad(basal_s, th, Y, yt, grad=True, dist=None):
    """Cross-entropy of sigma_bar against yt (+ distribution term) and its gradients (d / d th (K,), d / d basal)."""
    sig_bar, terms = _mem_terms(basal_s, th, Y)
    loss = _sigma_loss(sig_bar, yt)
    lw, dlw = (0.0, 0.0) if dist is None else _dist_loss_grad(sig_bar, dist)
    loss += lw
    if not grad:
        return loss, None, None
    dl = grad_main_loss(sig_bar, yt, 1) + dlw
    g_th, g_b = np.zeros(len(th)), 0.0
    for w, sq, Yq in terms:
        tmp = dl * w * sq * (1 - sq)
        g_th = g_th + Yq.T @ tmp
        g_b += float(np.sum(tmp))
    return loss, g_th, g_b


def _path_terms(yp, path):
    """[(weights (N,), states (N, K), sigmoid targets (N,) or None = those of the end states)] of the end states yp
    and of the nodes of path (target in force at each node along the bridge)."""
    return [(path.weights[:, 0], _end(yp), None)] + [
        (path.weights[:, q + 1], path.nodes[:, q], None if path.targets is None else path.targets[:, q])
        for q in range(path.nodes.shape[1])]


def _path_fit(theta_of, basal, ys, us, terms, yt, weights_samples, grad=True):
    """
    Flow matching: cross-entropy along the interval, sum_q w_q CE(sigmoid(basal + Y_q theta), yt_q), yt_q the
    target in force at node q (mode of the start state before the switch alpha of the target, of the end state
    after it; yt for the end state), nodes drawn uniformly over the interval with equal weights: the network
    reproduces the switching path of the trajectories. theta_of(s): effective interactions (K,) of sample s.
    Returns (loss, [(s, weight of the sample, d / d theta (K,), d / d basal)]).
    """
    Q = 0.0
    out = []
    w_tot = np.sum(weights_samples)
    for cnt, s in enumerate(us):
        s_idx = int(s)
        mask = (ys == s)
        if not np.any(mask):
            continue
        ws = w_tot / weights_samples[cnt] / len(us)
        th = theta_of(s_idx)
        yt_s = yt[mask]
        g_th, g_b = np.zeros(len(th)), 0.0
        for w, Y, tq in terms:
            w_s, Y_s = w[mask], Y[mask]
            if not np.any(w_s > 0):
                continue
            t_s = yt_s if tq is None else tq[mask]
            sig = logistic(basal[s_idx] + Y_s @ th)
            Q += ws * np.sum(w_s * cross_entropy(sig, t_s, eps_CE))
            if grad:
                tmp = grad_main_loss(sig, t_s, 1) * w_s * sig * (1 - sig)
                g_th += Y_s.T @ tmp
                g_b += float(np.sum(tmp))
        if grad:
            out.append((s_idx, ws, g_th, g_b))
    return Q, out


def objective(X, weights_samples, ys, yp, ypm, yt, G, g, n_samples,
              theta_ref, ref_network, l_pen, weight_flow_matching, final,
              constrain_basal_uniform=0.0, basal_free_mask=None, ridge=None, dist=None):
    """
    Loss function for a single gene — joint over all samples.
    ridge : (G,) or None; Gaussian prior 0.5 * sum ridge * (theta - theta_ref)^2 (network_ridge).

    X = theta, shape (G + n_samples,)
      0..G-1      : interaction weights
      G..G+ns-1   : per-sample basal values

    constrain_basal_uniform : float >= 0
        Penalty strength that pushes free-sample basals toward their mean.
        Samples whose basal is pinned by a KO/OV prior are excluded (basal_free_mask).
    basal_free_mask : (n_samples,) bool array or None
        True for samples that are NOT pinned by basal_ref for this gene.
    """
    theta_inter = X[:G]           # (G,)
    theta_basal = X[G:]           # (n_samples,)

    Q = 0
    us = np.unique(ys.ravel())
    ns_unique: int = len(us)
    w_fm = _fm_weight(weight_flow_matching, ypm)

    if isinstance(yp, MemStates):  # sigma averaged on the mRNA memory of the target
        theta_eff = theta_inter * ref_network
        for cnt, s in enumerate(us):
            weight_s = np.sum(weights_samples) / weights_samples[cnt]
            mask = (ys == s)
            Q += (1 - w_fm) * _mem_loss_grad(theta_basal[int(s)], theta_eff, yp[mask], yt[mask], grad=False,
                                             dist=_sl(dist, mask))[0] \
                * weight_s / ns_unique
    else:
        sigma = _compute_sigma_per_sample(theta_inter, theta_basal, ref_network, yp, ys, us)
        for cnt, s in enumerate(us):
            weight_s = np.sum(weights_samples) / weights_samples[cnt]
            mask = (ys == s)
            Q += (1 - w_fm) * _end_loss(sigma[mask], yt[mask], _sl(dist, mask)) * weight_s / ns_unique

    if w_fm:
        theta_eff = theta_inter * ref_network
        Q += w_fm * _path_fit(lambda s: theta_eff, theta_basal, ys, us, _path_terms(yp, ypm), yt,
                              weights_samples, grad=False)[0]

    d_inter = theta_inter - theta_ref[:G]
    Q += theta_penalization(d_inter * (ref_network > 0), l_pen)
    if final:
        # L2 towards the reference on sqrt(w) * theta (costs 1/w on the effective interaction, as L1); the L1 keeps
        # its full level, so the QUT holds for the edges absent from the reference
        Q += r_elasticnet * l2_penalization(np.sqrt(ref_network) * d_inter, l_pen)
    if ridge is not None:
        Q += 0.5 * np.sum(ridge * d_inter ** 2)

    # Uniformity penalty: push free-sample basals toward their common mean
    if constrain_basal_uniform > 0.0 and basal_free_mask is not None:
        free_idx = np.where(basal_free_mask)[0]
        if len(free_idx) >= 2:
            theta_free = theta_basal[free_idx]
            mean_free  = theta_free.mean()
            Q += constrain_basal_uniform * np.sum((theta_free - mean_free) ** 2)

    return Q


def _sigma_grad_terms(sigma, yt):
    """d loss / d logit per cell, for _sigma_loss."""
    return grad_main_loss(sigma, yt, 1) * sigma * (1 - sigma)


def grad_theta(X, weights_samples, ys, yp, ypm, yt, G, g, n_samples,
               theta_ref, ref_network, l_pen, weight_flow_matching, final,
               constrain_basal_uniform=0.0, basal_free_mask=None, ridge=None, dist=None):
    """Gradient of objective w.r.t. X = theta (dist: DistTerm of the end-state term or None)."""
    theta_inter = X[:G]
    theta_basal = X[G:]
    dq = np.zeros_like(X)
    us = np.unique(ys.ravel())
    ns_unique: int = len(us)

    w_fm = _fm_weight(weight_flow_matching, ypm)
    if w_fm:
        theta_eff = theta_inter * ref_network
        for s_idx, ws, gt, gb in _path_fit(lambda s: theta_eff, theta_basal, ys, us, _path_terms(yp, ypm), yt,
                                           weights_samples)[1]:
            dq[:G] += w_fm * ref_network * gt * ws
            dq[G + s_idx] += w_fm * gb * ws
    for pass_w, Y in [(1 - w_fm, yp)]:
        if isinstance(Y, MemStates):  # sigma averaged on the mRNA memory of the target
            theta_eff = theta_inter * ref_network
            for cnt, s in enumerate(us):
                s_idx = int(s)
                weight_s = np.sum(weights_samples) / weights_samples[cnt]
                mask = (ys == s)
                _, res, rb = _mem_loss_grad(theta_basal[s_idx], theta_eff, Y[mask], yt[mask], dist=_sl(dist, mask))
                dq[:G] += pass_w * ref_network * res * weight_s / ns_unique
                dq[G + s_idx] += pass_w * rb * weight_s / ns_unique
            continue
        sigma = _compute_sigma_per_sample(theta_inter, theta_basal, ref_network, Y, ys, us)
        for cnt, s in enumerate(us):
            s_idx = int(s)
            weight_s = np.sum(weights_samples) / weights_samples[cnt]
            mask = (ys == s)
            tmp = _end_grad_terms(sigma[mask], yt[mask], _sl(dist, mask))
            res = (Y[mask].T @ tmp[:, None]).reshape(-1)
            dq[:G] += pass_w * ref_network * res * weight_s / ns_unique
            dq[G + s_idx] += pass_w * np.sum(tmp) * weight_s / ns_unique

    d_inter = theta_inter - theta_ref[:G]
    dq[:G] += grad_theta_penalization(d_inter, l_pen) * (ref_network > 0)
    if final:
        dq[:G] += r_elasticnet * grad_l2_penalization(d_inter, l_pen) * ref_network
    if ridge is not None:
        dq[:G] += ridge * d_inter

    # Gradient of uniformity penalty: 2λ(θ_basal[s] − mean) for free samples
    if constrain_basal_uniform > 0.0 and basal_free_mask is not None:
        free_idx = np.where(basal_free_mask)[0]
        if len(free_idx) >= 2:
            theta_free = theta_basal[free_idx]
            mean_free  = theta_free.mean()
            dq[G + free_idx] += 2.0 * constrain_basal_uniform * (theta_free - mean_free)

    return dq


def _sigma_refinement(theta_basal, inter_with_diag, cond, Y, ys, us):
    """sigma = sigmoid(Z) of the refinement step, per-sample basal and per-condition network."""
    sigma = np.empty(Y.shape[0])
    for s in us:
        s_idx = int(s)
        mask = (ys == s)
        if np.any(mask):
            sigma[mask] = base_kon(theta_basal[s_idx], _of_cond(inter_with_diag, cond, s_idx), Y[mask])
    return sigma


def objective_refinement(X, inter, weights_samples, ys, yp, ypm, yt,
                          diag, G, g, n_samples, l_pen, weight_flow_matching, cond=None, dist=None):
    """
    Loss for the refinement step with per-sample basal.

    X shape (G + n_samples,)
      0..G-1    : multiplicative corrections for inter
      G..G+ns-1 : per-sample basal values (free, additive)
    inter, diag : (G,), or (C, G) with cond[s] = condition of sample s
    (one correction per edge, shared by the conditions)
    """
    theta_inter = inter.copy() * X[:G]          # (G,) or (C, G)
    theta_basal = X[G:]                          # (n_samples,)

    Q = 0
    us = np.unique(ys.ravel())
    ns_unique: int = len(us)
    inter_with_diag = theta_inter + diag

    w_fm = _fm_weight(weight_flow_matching, ypm)
    if w_fm:
        Q += w_fm * _path_fit(lambda s: _of_cond(inter_with_diag, cond, s), theta_basal, ys, us,
                              _path_terms(yp, ypm), yt, weights_samples, grad=False)[0]
    for pass_w, Y in [(1 - w_fm, yp)]:
        if isinstance(Y, MemStates):  # sigma averaged on the mRNA memory of the target
            for cnt, s in enumerate(us):
                weight_s = np.sum(weights_samples) / weights_samples[cnt]
                mask = (ys == s)
                Q += pass_w * _mem_loss_grad(theta_basal[int(s)], _of_cond(inter_with_diag, cond, int(s)), Y[mask],
                                             yt[mask], grad=False, dist=_sl(dist, mask))[0] * weight_s / ns_unique
            continue
        sigma = _sigma_refinement(theta_basal, inter_with_diag, cond, Y, ys, us)
        for cnt, s in enumerate(us):
            weight_s = np.sum(weights_samples) / weights_samples[cnt]
            mask = (ys == s)
            Q += pass_w * _end_loss(sigma[mask], yt[mask], _sl(dist, mask)) * weight_s / ns_unique

    # l_pen: (G,) per-edge penalty (1/w times the base one, see refine_inference)
    Q += theta_penalization(X[:g], l_pen[:g])
    Q += theta_penalization(X[g + 1:G], l_pen[g + 1:G])
    return Q


def grad_correc(X, inter, weights_samples, ys, yp, ypm, yt,
                diag, G, g, n_samples, l_pen, weight_flow_matching, cond=None, dist=None):
    """Gradient of objective_refinement w.r.t. X (interaction corrections, basals)."""
    theta_inter = inter.copy() * X[:G]
    theta_basal = X[G:]  # (n_samples,)
    inter_with_diag = theta_inter + diag

    dq = np.zeros_like(X)
    us = np.unique(ys.ravel())
    ns_unique: int = len(us)

    w_fm = _fm_weight(weight_flow_matching, ypm)
    if w_fm:
        for s_idx, ws, gt, gb in _path_fit(lambda s: _of_cond(inter_with_diag, cond, s), theta_basal, ys, us,
                                           _path_terms(yp, ypm), yt, weights_samples)[1]:
            dq[:G] += w_fm * _of_cond(inter, cond, s_idx) * gt * ws
            dq[G + s_idx] += w_fm * gb * ws
    for pass_w, Y in [(1 - w_fm, yp)]:
        if isinstance(Y, MemStates):  # sigma averaged on the mRNA memory of the target
            for cnt, s in enumerate(us):
                s_idx = int(s)
                weight_s = np.sum(weights_samples) / weights_samples[cnt]
                mask = (ys == s)
                _, res, rb = _mem_loss_grad(theta_basal[s_idx], _of_cond(inter_with_diag, cond, s_idx), Y[mask],
                                            yt[mask], dist=_sl(dist, mask))
                dq[:G] += pass_w * _of_cond(inter, cond, s_idx) * res * weight_s / ns_unique
                dq[G + s_idx] += pass_w * rb * weight_s / ns_unique
            continue
        sigma = _sigma_refinement(theta_basal, inter_with_diag, cond, Y, ys, us)
        for cnt, s in enumerate(us):
            s_idx = int(s)
            weight_s = np.sum(weights_samples) / weights_samples[cnt]
            mask = (ys == s)
            tmp = _end_grad_terms(sigma[mask], yt[mask], _sl(dist, mask))
            res = (Y[mask].T @ tmp[:, None]).reshape(-1)
            dq[:G] += pass_w * _of_cond(inter, cond, s_idx) * res * weight_s / ns_unique
            dq[G + s_idx] += pass_w * np.sum(tmp) * weight_s / ns_unique

    dq[:g] += grad_theta_penalization(X[:g], l_pen[:g])
    dq[g + 1:G] += grad_theta_penalization(X[g + 1:G], l_pen[g + 1:G])
    return dq


def core_inference(y_samples, y_prot, y_prot_mod, y_target, theta_init, theta_ref,
                   ref_network, G, g, n_samples,
                   l_pen, weight_flow_matching=.5, final=0,
                   constrain_basal_uniform=0.0, basal_free_mask=None,
                   hard_forcing_ref=False, ref_constraint_pct=0.1, seuil_min_network_intensity=1e-2, ridge=None,
                   dist=None):
    """
    Joint L-BFGS-B optimisation of interactions + per-sample basals for one gene.
    Effective interactions theta * w (w = prior_weight) are bounded by max_inter, basals by
    _basal_bound(G); theta_init / theta_ref are in theta units.

    theta_init shape: (G + n_samples,)
      0..G-1    : interaction weights
      G..G+ns-1 : per-sample basal values
    Returns theta_final of same shape.
    """
    weights_samples = [np.sum(y_samples == s) ** alpha for s in np.unique(y_samples)]

    X_flat = np.ascontiguousarray(theta_init, dtype=float).copy()   # (G + n_samples,)
    theta_ref_flat = np.ascontiguousarray(theta_ref, dtype=float)
    ref_network_flat = np.ascontiguousarray(ref_network, dtype=float)  # (G,)

    bounds = _interaction_bounds(theta_ref_flat, ref_network_flat, G, final,
                                 hard_forcing_ref, ref_constraint_pct, seuil_min_network_intensity)
    bounds += [(-_basal_bound(G), _basal_bound(G))] * n_samples

    kw = dict(weights_samples=weights_samples, ys=y_samples, yp=y_prot, ypm=y_prot_mod, yt=y_target,
              G=G, g=g, n_samples=n_samples, theta_ref=theta_ref, ref_network=prior_weight(ref_network),
              l_pen=l_pen, weight_flow_matching=weight_flow_matching, final=final,
              constrain_basal_uniform=constrain_basal_uniform, basal_free_mask=basal_free_mask, ridge=ridge,
              dist=dist)
    loss_fn = partial(objective, **kw)
    grad_fn = partial(grad_theta, **kw)

    res = minimize(loss_fn, X_flat, jac=grad_fn, method="L-BFGS-B", bounds=bounds, options=_lbfgs_options(G))
    if not res.success:
        logger.error('Minimization failed for inference: %s', res.message)

    if check_gradient:
        error = check_grad(loss_fn, grad_fn, res.x)
        if error > .05:
            logger.debug("Gradient theta inference check error for gene %s: %s", g, error)
            res = minimize(loss_fn, X_flat, bounds=bounds, method="L-BFGS-B", options=_lbfgs_options(G))

    theta_final = res.x.copy()
    theta_final[:G] *= prior_weight(ref_network)   # apply ref_network mask to interactions only

    return theta_final


def _interaction_bounds(theta_ref_flat, ref_network_flat, G, final,
                        hard_forcing_ref, ref_constraint_pct, seuil_min_network_intensity):
    """
    L-BFGS-B bounds of the G interaction variables (theta units, effective = theta * w), smin =
    seuil_min_network_intensity on the effective interaction:
    - signed priors (ref_network < -1 or > 1): sign forced, |effective| >= smin (every fit; against the sign of the
      reference they win, except over a reference pinned by hard_forcing_ref);
    - final = 0, references (theta_ref, effective |v| > smin): sign kept, |effective| >= smin; with hard_forcing_ref
      each interaction pinned to +- ref_constraint_pct around its reference (|.| <= smin for absent ones);
    - final = 1: signed priors only.
    """
    # Capacity applies to the effective interaction theta * w, not to theta
    w_flat = prior_weight(ref_network_flat)
    mb = np.where(w_flat > 0, max_inter / np.maximum(w_flat, EPS), max_inter)
    lo = np.where(w_flat > 0, seuil_min_network_intensity / np.maximum(w_flat, EPS), 0.0)  # smin in theta units
    bounds = [(-b, b) for b in mb]
    for idx in range(G):
        eff = theta_ref_flat[idx] * w_flat[idx]
        if not final and hard_forcing_ref:
            v = theta_ref_flat[idx]
            r_plus = 1 + ref_constraint_pct / max(EPS, (1 - ref_constraint_pct))
            r_minus = 1 - ref_constraint_pct
            if eff > seuil_min_network_intensity:
                bounds[idx] = (r_minus * v, max(r_minus * v, min(r_plus * v, mb[idx])))
            elif eff < -seuil_min_network_intensity:
                bounds[idx] = (min(r_minus * v, max(-mb[idx], r_plus * v)), r_minus * v)
            else:
                bounds[idx] = (-lo[idx], lo[idx])
        elif not final and eff > seuil_min_network_intensity:
            bounds[idx] = (lo[idx], mb[idx])
        elif not final and eff < -seuil_min_network_intensity:
            bounds[idx] = (-mb[idx], -lo[idx])
        # Signed priors, whatever the reference; against the sign of the reference, the prior wins (the pinned
        # reference with hard_forcing_ref)
        v = ref_network_flat[idx]
        if abs(v) > 1:
            sb = (lo[idx], mb[idx]) if v > 1 else (-mb[idx], -lo[idx])
            cut = (max(bounds[idx][0], sb[0]), min(bounds[idx][1], sb[1]))
            if cut[0] <= cut[1]:
                bounds[idx] = cut
            elif final or not hard_forcing_ref:
                bounds[idx] = sb
    return bounds


### NETWORK CONDITIONS: theta_c = theta_shared + delta_c (fused penalty on delta - delta_ref)

def _loss_grad_conditions(X, weights_samples, ys, yp, ypm, yt, G, n_samples,
                          cond, omega, theta_ref, delta_ref, w, l_pen, l_fuse, weight_flow_matching, final,
                          constrain_basal_uniform=0.0, basal_free_mask=None, act=None, ridge=None, dist=None):
    """
    Loss and gradient for one gene with one network per condition.

    X = [shared (G,), delta (C, G), basal (n_samples,)] in theta units; the network of
    condition c is theta_c = shared + delta_c (effective = w * theta_c); sample s uses cond[s].
    Penalties: sparsity sum_c omega_c pen(theta_c - theta_ref_c) (omega_c = share of the samples,
    so a common network costs as now) + l_fuse * L1(w * (delta_c - delta_ref_c)) (0 = independent
    networks); delta_ref_c = theta_ref_c - mean_c theta_ref_c (0 without reference: common network).
    act (C, G) or None: 1 for the regulators selected in each condition (select_genes_per_condition), 0 for the
    others (no effect, no sparsity, no fused penalty: only the edges shared by conditions are fused).
    """
    C = len(omega)
    shared = X[:G]
    delta = X[G:(C + 1) * G].reshape(C, G)
    basal = X[(C + 1) * G:]
    theta_c = shared[None] + delta
    wa = w[None] if act is None else w[None] * act  # (1 or C, G)
    eff = theta_c * wa

    Q = 0.0
    g_eff = np.zeros((C, G))
    g_basal = np.zeros(n_samples)
    us = np.unique(ys.ravel())
    w_tot = np.sum(weights_samples)
    w_fm = _fm_weight(weight_flow_matching, ypm)
    if w_fm:
        q_path, out = _path_fit(lambda s: eff[cond[s]], basal, ys, us, _path_terms(yp, ypm), yt, weights_samples)
        Q += w_fm * q_path
        for s_idx, ws, gt, gb in out:
            g_eff[cond[s_idx]] += w_fm * ws * gt
            g_basal[s_idx] += w_fm * ws * gb
    for pass_w, Y in [(1 - w_fm, yp)]:
        for cnt, s in enumerate(us):
            s_idx = int(s)
            c, mask = cond[s_idx], (ys == s)
            ws = pass_w * w_tot / weights_samples[cnt] / len(us)
            if isinstance(Y, MemStates):  # sigma averaged on the mRNA memory of the target
                q_s, gt, gb = _mem_loss_grad(basal[s_idx], eff[c], Y[mask], yt[mask], dist=_sl(dist, mask))
                Q += ws * q_s
                g_eff[c] += ws * gt
                g_basal[s_idx] += ws * gb
                continue
            sig = base_kon(basal[s_idx], eff[c], Y[mask])
            Q += ws * _end_loss(sig, yt[mask], _sl(dist, mask))
            tmp = _end_grad_terms(sig, yt[mask], _sl(dist, mask))
            g_eff[c] += ws * (Y[mask].T @ tmp)
            g_basal[s_idx] += ws * np.sum(tmp)

    g_theta = g_eff * wa
    for c in range(C):
        if omega[c] == 0:
            continue
        on = (w > 0) if act is None else (w > 0) & (act[c] > 0)
        d = theta_c[c] - theta_ref[c]
        Q += omega[c] * theta_penalization(d * on, l_pen)
        g_theta[c] += omega[c] * grad_theta_penalization(d, l_pen) * on
        if final:
            w_c = w if act is None else w * act[c]
            Q += omega[c] * r_elasticnet * l2_penalization(np.sqrt(w_c) * d, l_pen)
            g_theta[c] += omega[c] * r_elasticnet * grad_l2_penalization(d, l_pen) * w_c
        if ridge is not None:
            Q += omega[c] * 0.5 * np.sum(ridge * d ** 2)
            g_theta[c] += omega[c] * ridge * d

    # Fused penalty: effective deviations from the shared network pulled towards the reference ones
    g_delta = g_theta.copy()
    if l_fuse > 0:
        dd = (delta - delta_ref) * wa
        Q += smoothed_l1_penalization(dd, l_fuse)
        g_delta += grad_smoothed_l1_penalization(dd, l_fuse) * wa

    if constrain_basal_uniform > 0.0 and basal_free_mask is not None:
        free_idx = np.where(basal_free_mask)[0]
        if len(free_idx) >= 2:
            theta_free = basal[free_idx]
            mean_free = theta_free.mean()
            Q += constrain_basal_uniform * np.sum((theta_free - mean_free) ** 2)
            g_basal[free_idx] += 2.0 * constrain_basal_uniform * (theta_free - mean_free)

    grad = np.concatenate([g_theta.sum(axis=0), g_delta.ravel(), g_basal])
    return Q, grad


def fuse_conditions(theta_c, tol=1e-2, act=None):
    """Snap each condition's value to the median over conditions where it is within tol of it (exact fusion); act
    (C, G): median over the conditions where the entry exists, the others set to 0."""
    if act is None:
        med = np.median(theta_c, axis=0)
        return np.where(np.abs(theta_c - med[None]) < tol, med[None], theta_c)
    on = act > 0
    med = _masked_median(theta_c, on)
    return np.where(on, np.where(np.abs(theta_c - med[None]) < tol, med[None], theta_c), 0.0)


def _masked_median(x, on):
    """Median over axis 0 of the entries where on (0 where none)."""
    m = np.nanmedian(np.where(on, x, np.nan), axis=0) if on.any() else np.zeros(x.shape[1:])
    return np.nan_to_num(m)


def core_inference_conditions(y_samples, y_prot, y_prot_mod, y_target, theta_init, theta_ref,
                              ref_network, G, g, n_samples, cond,
                              l_pen, l_fuse, weight_flow_matching=.5, final=0,
                              constrain_basal_uniform=0.0, basal_free_mask=None,
                              hard_forcing_ref=False, ref_constraint_pct=0.1, seuil_min_network_intensity=1e-2, act=None,
                              ridge=None, dist=None):
    """
    core_inference with one network per condition (cond[s] = condition of sample s). act (C, G) or None: regulators
    selected in each condition (select_genes_per_condition); the shared network, the reference deviations and the
    fusion are then taken over the conditions where each regulator exists, and its other entries are 0.

    theta_init / theta_ref : (C * G + n_samples,), condition c in c*G..(c+1)*G-1
    (theta units). The reference / sign bounds apply to the shared network (mean of the references), the
    fused penalty pulls the deviations towards those of the references (theta_ref_c - mean); with
    hard_forcing_ref the deviations are pinned to the reference ones. Returns the effective (C * G + n_samples,).
    """
    C = int(np.max(cond)) + 1
    w = prior_weight(np.asarray(ref_network, dtype=float))
    us = np.unique(y_samples)
    weights_samples = [np.sum(y_samples == s) ** alpha for s in us]
    # Sparsity weight of each condition: its share of the samples of the fit
    omega = np.bincount(cond[us.astype(int)], minlength=C) / len(us)

    init_c = np.asarray(theta_init[:C * G], dtype=float).reshape(C, G)
    ref_c = np.asarray(theta_ref[:C * G], dtype=float).reshape(C, G)
    on = None if act is None else np.asarray(act, dtype=float) > 0
    shared0 = np.median(init_c, axis=0) if on is None else _masked_median(init_c, on)
    X_flat = np.concatenate([shared0, (init_c - shared0[None]).ravel(),
                             np.asarray(theta_init[C * G:], dtype=float)])

    if on is None:
        ref_flat = ref_c.mean(axis=0)
        delta_ref = ref_c - ref_flat[None]   # reference deviations from the mean reference
    else:  # mean over the conditions where the regulator exists
        ref_flat = np.sum(ref_c * on, axis=0) / np.maximum(on.sum(axis=0), 1)
        delta_ref = (ref_c - ref_flat[None]) * on
    ref_network_flat = np.ascontiguousarray(ref_network, dtype=float)
    bounds = _interaction_bounds(ref_flat, ref_network_flat, G, final,
                                 hard_forcing_ref, ref_constraint_pct, seuil_min_network_intensity)
    pinned = [(v, v) for v in delta_ref.ravel()]
    if hard_forcing_ref and not final:
        bounds += pinned
    else:
        bounds += [(min(-2 * max(-lo, hi), dr), max(2 * max(-lo, hi), dr))
                   for dr_c in delta_ref for (lo, hi), dr in zip(bounds[:G], dr_c)]
    bounds += [(-_basal_bound(G), _basal_bound(G))] * n_samples

    fn = partial(_loss_grad_conditions, weights_samples=weights_samples, ys=y_samples,
                 yp=y_prot, ypm=y_prot_mod, yt=y_target, G=G,
                 n_samples=n_samples, cond=cond, omega=omega, theta_ref=ref_c, delta_ref=delta_ref, w=w,
                 l_pen=l_pen, l_fuse=l_fuse, weight_flow_matching=weight_flow_matching, final=final,
                 constrain_basal_uniform=constrain_basal_uniform, basal_free_mask=basal_free_mask,
                 act=None if act is None else np.asarray(act, dtype=float), ridge=ridge, dist=dist)
    # Stage 1: deviations pinned to the reference ones (0 without reference), well conditioned; stage 2:
    # free deviations, started from stage 1 or the initial deviations (the stiff fused term alone stalls L-BFGS-B)
    d_sl = slice(G, (C + 1) * G)
    X_common = X_flat.copy()
    X_common[d_sl] = delta_ref.ravel()
    bounds_common = bounds[:G] + pinned + bounds[(C + 1) * G:]
    res = minimize(fn, X_common, jac=True, method="L-BFGS-B", bounds=bounds_common, options=_lbfgs_options(G))
    if l_fuse < np.inf and not (hard_forcing_ref and not final):
        starts = [res.x, np.concatenate([res.x[:G], X_flat[d_sl], res.x[(C + 1) * G:]])]
        x0 = min(starts, key=lambda x: fn(x)[0])
        res = minimize(fn, x0, jac=True, method="L-BFGS-B", bounds=bounds, options=_lbfgs_options(C * G))
    if not res.success:
        logger.error('Minimization failed for inference (network conditions): %s', res.message)

    if check_gradient:
        error = check_grad(lambda x: fn(x)[0], lambda x: fn(x)[1], res.x)
        if error > .05:
            logger.debug("Gradient theta inference (conditions) check error for gene %s: %s", g, error)

    shared = res.x[:G]
    delta = res.x[G:(C + 1) * G].reshape(C, G)
    theta_c = fuse_conditions((shared[None] + delta) * w[None], act=act)
    return np.concatenate([theta_c.ravel(), res.x[(C + 1) * G:]])


def refine_inference(y_samples, y_prot, y_prot_mod, y_target, inter, basal, theta_ref,
                     G, g, n_samples,
                     l_pen, weight_flow_matching=.5, final=0,
                     hard_forcing_ref=False, ref_constraint_pct=0.1, ref_network=None, seuil_min_network_intensity=1e-2,
                     cond=None, dist=None):
    """
    Refinement step: optimise multiplicative correction factors over inter, per-sample basals refitted freely
    (additive, any sign). Refined values stay within max_inter and _basal_bound(G);
    sign-forced edges (|ref_network| > 1) keep |theta| >= seuil_min_network_intensity.

    Adaptive lasso (Zou 2006) with weights 1 / |theta_tmp|, i.e. Breiman's non-negative garrote:
    theta_e = c_e theta_tmp_e with c_e >= 0, so l_pen |c_e| = l_pen |theta_e| / |theta_tmp_e|
    (l_pen: quantile universal threshold of the garrote itself, qut_garrote),
    within the support and signs of theta_tmp (self-regulation and basals unpenalized). final: no bounds around the
    reference (signed priors only), as in core_inference.

    basal : (n_samples,)
    inter : (G,), or (C, G) per condition (cond[s] = condition of sample s):
        one correction per edge, shared by the conditions, keeps their differences.
    Returns updated (inter, basal).
    """
    diag = np.zeros(inter.shape)
    if g < G:   # g == G is the sentinel meaning "no self-regulation in active set"
        diag[..., g] = inter[..., g]
    inter = inter.copy()
    inter -= diag
    basal = basal.copy()  # (n_samples,)

    weights_samples = [np.sum(y_samples == s) ** alpha for s in np.unique(y_samples)]

    X_flat = np.r_[np.ones(G), np.clip(basal, -_basal_bound(G), _basal_bound(G))]

    # Corrections: interactions in [0, 2] (selection, little expansion); basals free within their capacity
    max_bounds = 10
    bounds = [(0, max_bounds)] * G
    if not final and hard_forcing_ref:
        r_plus = min(max_bounds, 1 + ref_constraint_pct / max(EPS, (1 - ref_constraint_pct)))
        r_minus = 1 - ref_constraint_pct
        for idx in range(G):
            bounds[idx] = (r_minus, max(r_minus, r_plus))

    # Edges of prior weight w cost 1/w more, as in core_inference
    w = prior_weight(ref_network) if ref_network is not None else np.ones(G)
    # Kept edges (|effective| >= seuil_min_network_intensity, the garrote keeping their signs): signed priors, and
    # references above the threshold out of the final fit (as in _interaction_bounds)
    forced = np.abs(ref_network) > 1 if ref_network is not None else np.zeros(G, bool)
    if not final:
        forced = forced | (np.abs(np.asarray(theta_ref, dtype=float)[:G] * w) > seuil_min_network_intensity)
    # Cap corrections so that |value * correction| stays within capacity (reference lower bounds win)
    l_pen = np.where(w > 0, l_pen / np.maximum(w, EPS), l_pen) * np.ones(G)
    # Per correction: largest |value| for the capacity, smallest for the forced magnitude (over conditions)
    a_inter = np.abs(inter).reshape(-1, G) if G else np.zeros((1, 0))
    v_max, v_min = a_inter.max(axis=0), a_inter.min(axis=0)
    for idx, (v, v_lo) in enumerate(zip(v_max, v_min)):
        if v != 0.0:
            lo, hi = bounds[idx]
            # Ratios written without dividing by a near-zero |v| (overflow; the min picks the other term anyway)
            if forced[idx]:
                lo = max(lo, 1.0 if v_lo <= seuil_min_network_intensity else seuil_min_network_intensity / v_lo)  # keep the kept-edge magnitude
            bounds[idx] = (lo, max(lo, hi if v * hi <= max_inter else max_inter / v))
    bounds += [(-_basal_bound(G), _basal_bound(G))] * n_samples

    kw = dict(inter=inter,
              weights_samples=weights_samples, ys=y_samples, yp=y_prot, ypm=y_prot_mod, yt=y_target,
              diag=diag, G=G, g=g, n_samples=n_samples,
              l_pen=l_pen, weight_flow_matching=weight_flow_matching, cond=cond, dist=dist)
    loss_fn = partial(objective_refinement, **kw)
    grad_fn = partial(grad_correc, **kw)

    res = minimize(loss_fn, X_flat, jac=grad_fn, method="L-BFGS-B", bounds=bounds, options=_lbfgs_options(G))
    if not res.success:
        logger.error('Minimization failed for refining: %s', res.message)

    if check_gradient:
        error = check_grad(loss_fn, grad_fn, res.x)
        if error > .05:
            logger.debug("Gradient theta refining check error for gene %s: %s", g, error)
            res = minimize(loss_fn, X_flat, bounds=bounds, method="L-BFGS-B", options=_lbfgs_options(G))

    correc = res.x
    inter *= correc[:G]
    basal = correc[G:].copy()
    inter += diag

    return inter, basal


def ridge_coefficients(y_prot, ref_network, network_ridge):
    """
    Gaussian prior on the effect of each edge on the logit, w_e theta_e sd_e ~ N(0, s^2) (sd_e: standard deviation
    of regulator e), s = 2 logit(1 - 1/network_ridge): the logit span from a target at 1/network_ridge of OFF to
    1/network_ridge of ON. Bounds the edges of nearly separable targets; zero gradient at 0, so the QUT is unchanged.
    Returns the coefficients (w sd / s)^2 of 0.5 sum c theta^2.
    """
    eps = 1.0 / max(network_ridge, 2.0 + EPS)
    span = 2.0 * np.log((1.0 - eps) / eps)
    sd = np.std(np.asarray(y_prot, dtype=float), axis=0)
    return (prior_weight(ref_network) * sd / span) ** 2


def qut_garrote(y_samples, y_prot, y_prot_mod, y_target, inter, basal, G, g, n_samples, weight_flow_matching, ref_network,
                cond=None, level=0.5, n_perm=20):
    """
    Quantile universal threshold of the refinement step (non-negative garrote theta_e = c_e theta_tmp_e, c_e >= 0),
    theta_tmp = inter held fixed (conditional QUT): level-quantile of max_e w_e (-dL/dc_e)_+ at the null model
    (c = 0 out of self-regulation, per-sample basals fitted freely), the target being permuted within each sample.
    dL/dc_e = theta_tmp_e dL/dtheta_e, so the level follows the shrinkage of the first fit.
    """
    if G == 0:  # no candidate regulator
        return 0.0
    diag = np.zeros(inter.shape)
    if g < G:
        diag[..., g] = inter[..., g]
    inter = inter - diag
    w = prior_weight(ref_network) if ref_network is not None else np.ones(G)
    on = np.any(np.abs(inter).reshape(-1, G) > 0, axis=0) & (w > 0)
    if not on.any():
        return 0.0
    kw = dict(inter=inter,
              weights_samples=[np.sum(y_samples == s) ** alpha for s in np.unique(y_samples)],
              ys=y_samples, yp=y_prot, diag=diag, G=G, g=g, n_samples=n_samples, l_pen=np.zeros(G),
              weight_flow_matching=weight_flow_matching, cond=cond)
    # Null basals: invariant under within-sample permutations of the target
    null = dict(kw, ypm=y_prot_mod, yt=y_target)
    bb = _basal_bound(G)
    X0 = minimize(partial(objective_refinement, **null), np.r_[np.zeros(G), np.asarray(basal, dtype=float)],
                  jac=partial(grad_correc, **null), method="L-BFGS-B",
                  bounds=[(0.0, 0.0)] * G + [(-bb, bb)] * n_samples, options=_lbfgs_options(G)).x
    ys = np.asarray(y_samples).astype(int)
    groups = [np.flatnonzero(ys == s) for s in np.unique(ys)]
    rng = np.random.default_rng(1000 * g + len(y_target) + 1)
    stats = np.empty(n_perm)
    for k in range(n_perm):
        p = np.arange(len(y_target))
        for idx in groups:
            p[idx] = rng.permutation(idx)
        ypm = y_prot_mod
        if ypm is not None:
            ypm = PathNodes(ypm.nodes, ypm.weights, None if ypm.kon is None else ypm.kon[p],
                            None if ypm.targets is None else ypm.targets[p])
        dq = grad_correc(X0, ypm=ypm, yt=y_target[p], **kw)[:G]
        stats[k] = np.max(w[on] * np.maximum(-dq[on], 0.0))
    return float(np.quantile(stats, level))


def qut_penalty(y_samples, y_prot, y_prot_mod, y_target, ref_network, G, g, n_samples, weight_flow_matching,
                level=0.95, n_perm=20, constrain_basal_uniform=0.0, basal_free_mask=None):
    """
    Quantile universal threshold (Giacobino et al. 2017) of the sparsity penalty of target g: level-quantile of
    max_j |dL/d theta_j| at the null model (no interaction, per-sample basals fitted), the target being permuted
    within each sample. Under the null no edge enters with probability ~ level (L1 KKT condition at 0).
    """
    w = prior_weight(ref_network)
    if not np.any(w > 0):  # no candidate regulator
        return 0.0
    kw = dict(ys=y_samples, yp=y_prot, G=G, g=g, n_samples=n_samples, theta_ref=np.zeros(G + n_samples),
              ref_network=w, l_pen=0.0, weight_flow_matching=weight_flow_matching, final=0,
              constrain_basal_uniform=constrain_basal_uniform, basal_free_mask=basal_free_mask)
    kw['weights_samples'] = [np.sum(y_samples == s) ** alpha for s in np.unique(y_samples)]
    # Null basals: invariant under within-sample permutations of the target
    bounds = [(0.0, 0.0)] * G + [(-_basal_bound(G), _basal_bound(G))] * n_samples
    null = dict(kw, ypm=y_prot_mod, yt=y_target)
    X0 = minimize(partial(objective, **null), np.zeros(G + n_samples), jac=partial(grad_theta, **null),
                  method="L-BFGS-B", bounds=bounds, options=_lbfgs_options(G)).x
    ys = np.asarray(y_samples).astype(int)
    groups = [np.flatnonzero(ys == s) for s in np.unique(ys)]
    rng = np.random.default_rng(1000 * g + len(y_target))
    on = w > 0
    if not on.any():
        return 0.0
    stats = np.empty(n_perm)
    for k in range(n_perm):
        p = np.arange(len(y_target))
        for idx in groups:
            p[idx] = rng.permutation(idx)
        ypm = y_prot_mod
        if ypm is not None:
            ypm = PathNodes(ypm.nodes, ypm.weights, None if ypm.kon is None else ypm.kon[p],
                            None if ypm.targets is None else ypm.targets[p])
        dq = grad_theta(X0, ypm=ypm, yt=y_target[p], **kw)[:G]
        stats[k] = np.max(np.abs(dq[on]))
    return float(np.quantile(stats, level))


def main_loop_inference(g, y_samples, y_prot, y_prot_mod, y_kon, theta_init, theta_ref,
                        ks, G, n_samples, ref_network,
                        weight_flow_matching=.5, final=0,
                        constrain_basal_uniform=0.0, basal_free_mask=None,
                        hard_forcing_ref=False, ref_constraint_pct=0.1, seuil_min_network_intensity=1e-2,
                        cond=None, condition_pen=1.0, act=None, act_tgt=None,
                        qut_level=0.5, qut_n_perm=20, network_ridge=20.0, dist=None):
    """
    Joint inference (inter + per-sample basal) for a single gene g.
    dist: DistTerm of the end-state term (weight_network_distribution) or None; the sparsity levels (QUT) are those of
    the cross-entropy alone (at the null model the sigmas are tied and the distribution term is not defined).
    Sparsity levels: quantile universal thresholds at qut_level, of the first fit (qut_penalty) and of the
    refinement (qut_garrote, theta of the first fit fixed);
    network_ridge: Gaussian prior on the effects of the edges in the first fit (ridge_coefficients).
    final : L2 towards theta_ref added to the L1, no bounds around the reference (signed priors only).
    act (C, G), act_tgt (C,) or None: regulators and target selected in each condition (select_genes_per_condition);
    the samples of a condition without the target are left out (as single-mode ones) and its interactions are 0.

    theta_init : (G + n_samples,), or (C * G + n_samples,) with one network per condition
                 (cond[s] = condition of sample s, see core_inference_conditions)
    y_kon      : target kon of g per cell, fitted as its sigmoid target (sigmoid_target): cross-entropy on the
                 sigmoid, symmetric between ON and OFF
    ks         : mixture amplitudes of g, (n_modes,) or (n_samples, n_modes): only the lowest
                 and highest modes are used (sigmoid bounds)
    basal_free_mask : (n_samples,) bool — True for samples NOT pinned by basal_ref for gene g
    condition_pen : fused penalty on the deviations between conditions, relative to the sparsity one
    Returns (basal, inter, basal_tmp, inter_tmp): basal (n_samples,), inter (G,) or (C, G);
    *_tmp are the values before the refinement step.
    """
    # Sigmoid between the lowest and the highest mode of each cell's sample (intermediate modes lie on the slope)
    kb = sigmoid_amplitudes(ks)
    if kb.ndim == 2:
        kb = kb[np.asarray(y_samples).astype(int)]
    y_target = sigmoid_target(y_kon, kb[..., 0], kb[..., 1])
    if y_prot_mod is not None:  # targets in force at the flow-matching nodes
        lo, hi = np.asarray(kb[..., 0]), np.asarray(kb[..., 1])  # scalars, or one per row (per-sample mixtures)
        lo, hi = (lo[:, None], hi[:, None]) if lo.ndim else (lo, hi)
        y_prot_mod = PathNodes(y_prot_mod.nodes, y_prot_mod.weights, y_prot_mod.kon,
                               sigmoid_target(y_prot_mod.kon, lo, hi))
    C = 1 if cond is None else int(np.max(cond)) + 1
    n_rows = C * G   # interaction entries of theta

    # Samples where g has a single mode (constant target): no information on its regulation, left out of the
    # network fit (and of the basal uniformity); their basal is fitted alone on the final network
    ys_all = np.asarray(y_samples).astype(int)
    flat = np.array([np.any(ys_all == s) and np.ptp(y_target[ys_all == s]) <= FLAT_TARGET_TOL
                     for s in range(n_samples)], dtype=bool)
    if act is not None:
        act = np.asarray(act, dtype=float) * (1.0 if act_tgt is None else np.asarray(act_tgt, dtype=float)[:, None])
        if act_tgt is not None and cond is not None:
            flat |= ~np.asarray(act_tgt, dtype=bool)[np.asarray(cond)[:n_samples]]  # condition without the target
    full = (ys_all, _end(y_prot), y_target)
    if flat.any():
        keep = ~flat[ys_all]
        y_samples, y_prot, y_target = y_samples[keep], y_prot[keep], y_target[keep]
        y_prot_mod = None if y_prot_mod is None else y_prot_mod[keep]
        dist = _sl(dist, keep)
        basal_free_mask = (np.ones(n_samples, bool) if basal_free_mask is None else basal_free_mask) & ~flat
    if len(y_target) == 0:
        # g has a single mode in every sample: network kept at its initial value, basals fitted alone
        inter = np.asarray(theta_init[:n_rows], dtype=float).reshape((G,) if cond is None else (C, G))
        if act is not None and cond is not None:
            inter = inter * act
        basal = _fit_flat_basals(np.array(theta_init[n_rows:n_rows + n_samples], dtype=float), inter, flat, *full,
                                 g, cond, _basal_bound(G))
        return basal, inter, basal.copy(), inter.copy()

    # Interactions are optimised as theta = e / w (w = prior weight), so an edge of weight q
    # costs 1/q more; init and ref are given as effective values e and converted here.
    w = prior_weight(ref_network)
    w_rows = np.tile(w, C)
    theta_init, theta_ref = np.array(theta_init, dtype=float), np.array(theta_ref, dtype=float)
    theta_init[:n_rows] = np.where(w_rows > 0, theta_init[:n_rows] / np.maximum(w_rows, EPS), 0.0)
    theta_ref[:n_rows] = np.where(w_rows > 0, theta_ref[:n_rows] / np.maximum(w_rows, EPS), 0.0)

    l_pen = qut_penalty(y_samples, y_prot, y_prot_mod, y_target, ref_network, G, g, n_samples,
                        weight_flow_matching * (1 - final), qut_level, qut_n_perm, constrain_basal_uniform, basal_free_mask)

    common = dict(weight_flow_matching=weight_flow_matching * (1 - final), final=final,
                  constrain_basal_uniform=constrain_basal_uniform, basal_free_mask=basal_free_mask,
                  hard_forcing_ref=hard_forcing_ref, ref_constraint_pct=ref_constraint_pct,
                  seuil_min_network_intensity=seuil_min_network_intensity,
                  ridge=ridge_coefficients(_end(y_prot), ref_network, network_ridge), dist=dist)
    if cond is None:
        theta = core_inference(y_samples, y_prot, y_prot_mod, y_target, theta_init, theta_ref, ref_network,
                               G, g, n_samples, l_pen, **common)
    else:
        theta = core_inference_conditions(y_samples, y_prot, y_prot_mod, y_target, theta_init, theta_ref, ref_network,
                                          G, g, n_samples, cond, l_pen, condition_pen * l_pen, act=act, **common)
    inter_tmp = theta[:n_rows].reshape((G,) if cond is None else (C, G))
    basal_tmp = theta[n_rows:n_rows + n_samples].copy()

    # Refinement step (reference of the shared network for the conditions)
    ref_refine = theta_ref if cond is None else np.concatenate(
        [theta_ref[:n_rows].reshape(C, G).mean(axis=0), theta_ref[n_rows:]])

    inter, basal = refine_inference(
        y_samples, y_prot, y_prot_mod, y_target, inter_tmp.copy(), basal_tmp.copy(), ref_refine,
        G, g, n_samples,
        qut_garrote(y_samples, y_prot, y_prot_mod, y_target, inter_tmp, basal_tmp, G, g, n_samples,
                    weight_flow_matching * (1 - final), ref_network, cond, qut_level, qut_n_perm),
        weight_flow_matching=weight_flow_matching * (1 - final),  # adaptive lasso, weights 1 / |theta_tmp|
        final=final, hard_forcing_ref=hard_forcing_ref, ref_constraint_pct=ref_constraint_pct,
        ref_network=ref_network, seuil_min_network_intensity=seuil_min_network_intensity, cond=cond, dist=dist,
    )
    if act is not None and cond is not None:
        inter = inter * act  # entries of regulators / targets outside each condition stay 0
    if flat.any():
        basal = _fit_flat_basals(basal, inter, flat, *full, g, cond, _basal_bound(G))
        basal_tmp[flat] = basal[flat]

    return basal, inter, basal_tmp, inter_tmp


def _fit_flat_basals(basal, inter, flat, ys, Y, yt, g, cond, bound):
    """Basal of each flat sample (constant target of g) fitted alone on the network inter ((G,) or (C, G),
    self-regulation g included), bounded by bound; the other basals are returned unchanged."""
    from scipy.optimize import minimize_scalar
    basal = np.array(basal, dtype=float)
    for s in np.flatnonzero(flat):
        mask = ys == s
        th = np.asarray(_of_cond(inter, cond, s), dtype=float)
        # Constant target 0 or 1 has no finite optimum: clip it so that the basal saturates at FLAT_TARGET_TOL
        yt_s = np.clip(yt[mask], FLAT_TARGET_TOL, 1 - FLAT_TARGET_TOL)
        def loss(b):
            # Cross-entropy written on the logit x (convex, not flat where the clipped sigmoid saturates)
            x = b + Y[mask] @ th
            return float(np.sum(np.logaddexp(0.0, x) - yt_s * x))
        res = minimize_scalar(loss, bounds=(-bound, bound), method='bounded')
        basal[s] = res.x
    return basal


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

    ref_network : (G, G); with hard_forcing_ref, edges of inter_ref are added
    (inter_ref per sample (n_samples, G, G): edges of any sample).
    Returns (ref_network, cols) with cols[g] an int array.
    """
    ref_network = np.asarray(ref_network, dtype=float)
    if hard_forcing_ref and inter_ref is not None:
        present = np.asarray(inter_ref, dtype=float) != 0
        if present.ndim == 3:
            present = present.any(axis=0)
        ref_network = signed_floor(ref_network, present.astype(float))
    cols = [np.flatnonzero(ref_network[:, g] != 0) for g in range(ref_network.shape[1])]
    return ref_network, cols


class PathProt:
    """
    Flow matching: states of every gene at flow_matching_nodes nodes drawn uniformly along the interval ending at each
    row, nodes (N_cells, Q, G) (they do not depend on the target), weights (N_cells, Q + 1, G) (column 0 = the row
    itself: 0 and 1 / Q on the nodes, 1 on the row for the rows without previous interval), and kon (N_cells, Q, G):
    kon of each target in force at each node (mode of the start state before its switch alpha, of the end state
    after it).
    """

    def __init__(self, nodes, weights, kon=None):
        self.nodes = nodes
        self.weights = weights
        self.kon = np.zeros(nodes.shape) if kon is None else kon

    @classmethod
    def zeros(cls, n_cells, n_nodes, G):
        weights = np.zeros((n_cells, n_nodes + 1, G))
        weights[:, 0] = 1.0
        return cls(np.zeros((n_cells, n_nodes, G)), weights)

    @classmethod
    def from_states(cls, y_prot, n_nodes):
        """No path: every row on its own state (weight 1), i.e. the main term alone."""
        path = cls.zeros(len(y_prot), n_nodes, y_prot.shape[1])
        path.nodes[:] = y_prot[:, None, :]
        return path

    def select_cells(self, idx):
        return PathProt(self.nodes[idx], self.weights[idx], self.kon[idx])

    def get(self, g, src):
        """PathNodes of the regulators src for target g (with its kon in force at the nodes)."""
        return PathNodes(self.nodes[:, :, src], self.weights[:, :, g], self.kon[:, :, g])


def _infer_gene(g_tgt, G, active_src, y_samples, yp, ypm, y_kon_g, ti_g, tr_g,
                ks_g, n_samples, rn, free_mask_g, gene_kw, cond=None, act=None,
                act_tgt=None, dist=None):
    """
    Network inference for one target gene (worker task of inference_network).
    yp / ypm / rn are already restricted to the active regulators active_src
    (full when every regulator is active). With cond (condition of each sample), ti_g / tr_g hold
    one block of G interactions per condition and the interactions returned are (C, G).
    """
    K_g = len(active_src)
    C = 1 if cond is None else int(np.max(cond)) + 1
    mask_kw = {} if act is None else dict(act=act, act_tgt=act_tgt)
    if K_g == G:
        # Dense path: no sub-selection
        return main_loop_inference(
            g_tgt, y_samples, yp, ypm, y_kon_g, ti_g, tr_g, ks_g, G, n_samples, rn,
            basal_free_mask=free_mask_g, cond=cond, dist=dist, **mask_kw, **gene_kw,
        )

    # Sparse path: sub-select to K_g active regulators. K_g == 0: no interaction
    # to infer, only the per-sample basal is fitted (O(N_cells) problem).
    # Sub-select theta: active_src entries (of each condition) + per-sample basals
    rows = np.concatenate([c * G + active_src for c in range(C)] + [np.arange(C * G, C * G + n_samples)])
    theta_init_sub = ti_g[rows]  # (C * K_g + n_samples,)
    theta_ref_sub  = tr_g[rows]

    # Position of g_tgt in active_src (sentinel K_g = "no self-regulation")
    g_sub_matches = np.where(active_src == g_tgt)[0]
    g_sub = int(g_sub_matches[0]) if len(g_sub_matches) > 0 else K_g

    basal_r, inter_sub_r, basal_tmp_r, inter_tmp_sub_r = main_loop_inference(
        g_sub, y_samples, yp, ypm, y_kon_g, theta_init_sub, theta_ref_sub,
        ks_g, K_g, n_samples, rn,
        basal_free_mask=free_mask_g, cond=cond, dist=dist, **mask_kw, **gene_kw,
    )

    # Expand interaction results back to full G-dimensional space
    shape = (G,) if cond is None else (C, G)
    inter_full     = np.zeros(shape)
    inter_tmp_full = np.zeros(shape)
    inter_full[..., active_src]     = inter_sub_r
    inter_tmp_full[..., active_src] = inter_tmp_sub_r
    return basal_r, inter_full, basal_tmp_r, inter_tmp_full


def inference_network_multi(subsets, y_samples, y_kon, y_prot, y_prot_mod, ks, n_stimuli=1,
                      ref_network=None,
                      basal_init=None, inter_init=None,
                      basal_ref=None, inter_ref=None,
                      network_ridge=20.0, weight_flow_matching=.5, final=0,
                      samples_id=None,
                      constrain_basal_uniform=0.0,
                      hard_forcing_ref=False, ref_constraint_pct=0.1,
                      seuil_min_network_intensity=1e-2,
                      sample_conditions=None, condition_pen=1.0, gene_mask=None,
                      sigma_spec_grn=None, qut_level=0.5, qut_n_perm=20, y_prot_mem=None,
                      dist_targets=None) -> list:
    """
    Joint network inference: interactions shared across samples, basal per sample.
    One independent fit per row subset in subsets (index arrays, None = all rows);
    all fits x target genes are run in a single parallel pool.

    y_prot_mod : PathProt (flow-matching nodes), or None without flow-matching term (final fit).
    ks : mixture amplitudes (G, n_modes), or per sample (n_samples, G, n_modes); kon is a sigmoid
        between the lowest and the highest mode.

    Returns a list with, per subset, (basal, inter, basal_tmp, inter_tmp):
    basal (n_samples, G), inter (G, G).

    constrain_basal_uniform : float >= 0
        When > 0, adds a penalty that pushes per-sample basals toward their common
        mean for each gene. Samples whose basal is pinned by a non-zero basal_ref
        (KO/OV priors) are excluded from the penalty for that gene.
    sample_conditions : (n_samples,) int array or None
        Network condition of each sample (index). With >= 2 conditions, each condition has its
        own network theta_c = theta_shared + delta_c, with a fused L1 penalty on delta_c of
        condition_pen times the sparsity penalty (0: independent networks); inter is then
        returned per sample, (n_samples, G, G). inter_init / inter_ref may be
        (G, G) (common) or per sample.
    gene_mask : (n_samples, G) bool or None: genes selected in each sample's condition (select_genes_per_condition);
        a gene outside a condition is neither a target nor a regulator there (its interactions are 0) and the fused
        penalty only acts on the edges shared by conditions. Without conditions, the mask of the (single) gene set.
    sigma_spec_grn : (G, N_SPEC) or None: families of the GRN sigma (sigma.py, model_learning_grn); a target with a
        non-identity phi is fitted on the features phi of its regulators (same fit, theta then acting on phi; logistic
        link only). None: every target logistic of a linear input.
    y_prot_mem : PathProt or None (targets_accounting_late_mrna): states of every gene at nodes along the interval
        ending at each row (nodes (N, Q, G)) and weights (N, Q + 1, G) of each target's mRNA memory (column 0 = the row);
        the end-state term then compares sigma averaged on the memory of the target to its target (MemStates).
    dist_targets : (post (N, G, L), levels (N, G, L), grp (N,), weight) or None (weight_network_distribution): posteriors
        and sigmoid targets of the levels of the real cell of each row, group (sample, time) of each row (-1 without
        real cell); targets of more than two levels get the distribution term (dist_term) in their end-state term.
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
        # (C * G, G) interaction rows: common (G, G) or per sample (n_samples, G, G)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 3:
            if cond is None and not np.allclose(arr, arr[:1]):
                raise ValueError("per-sample interactions given without network conditions")
            arr = arr[first_of]
        else:
            arr = np.repeat(arr[None], C, axis=0)
        return arr.reshape(C * G, G)

    def basal_rows(arr):
        # (n_samples, G) basal rows: per sample (n_samples, G) or common (G,)
        arr = np.asarray(arr, dtype=float)
        out = np.zeros((n_samples, G))
        if arr.ndim == 2:
            out[:min(n_samples, arr.shape[0])] = arr[:n_samples]
        else:
            out[:] = arr
        return out

    # ── theta_init / theta_ref (C * G + n_samples, G): rows 0..C*G-1 interactions per target gene
    # (one block per condition), rows C*G..C*G+ns-1 per-sample basal per target gene ──
    theta_init_mat = np.zeros((C * G + n_samples, G))
    if inter_init is not None:
        theta_init_mat[:C * G] = inter_rows(inter_init)
    if basal_init is not None:
        theta_init_mat[C * G:] = basal_rows(basal_init)

    theta_ref_mat = np.zeros((C * G + n_samples, G))
    if inter_ref is not None:
        theta_ref_mat[:C * G] = inter_rows(inter_ref)
    if basal_ref is not None:
        theta_ref_mat[C * G:] = basal_rows(basal_ref)

    if ref_network is None:
        ref_network = np.ones((G, G))

    # When hard_forcing_ref, the inter_ref matrix defines the allowed structure in addition
    # to ref_network: interactions specified by inter_ref must not be zeroed out by the
    # structural prior mask at the end of core_inference.
    ref_network, active_cols = active_regulators(ref_network, inter_ref, hard_forcing_ref)

    def prev_prot(g_tgt, src):
        # Flow-matching nodes of regulators src for target g_tgt (None without flow matching)
        return None if y_prot_mod is None else y_prot_mod.get(g_tgt, src)

    # ── Free-sample mask from basal_ref: free_mask[s, g] = True iff sample s is NOT pinned
    # for gene g (basal_ref[s, g] == 0, no KO/OV prior); a common basal_ref pins no sample ──
    br_arr = None if basal_ref is None else np.asarray(basal_ref, dtype=float)
    if br_arr is not None and br_arr.ndim == 2:
        free_mask_2d = br_arr == 0.0
    else:
        free_mask_2d = np.ones((n_samples, G), dtype=bool)

    gene_kw = dict(weight_flow_matching=weight_flow_matching, final=final,
                   constrain_basal_uniform=constrain_basal_uniform,
                   hard_forcing_ref=hard_forcing_ref, ref_constraint_pct=ref_constraint_pct,
                   seuil_min_network_intensity=seuil_min_network_intensity, condition_pen=condition_pen,
                   qut_level=qut_level, qut_n_perm=qut_n_perm, network_ridge=network_ridge)

    # Genes of each condition (rows of the first sample of each condition); a single set without conditions
    gm = None if gene_mask is None else np.asarray(gene_mask, dtype=bool)
    act_c = None if gm is None else gm[first_of]   # (C, G)
    if act_c is not None and cond is None:
        # One gene set: regulators outside it are dropped from every target
        active_cols = [c[act_c[0][c]] for c in active_cols]

    def gene_args(rows, g_tgt):
        # Only the rows and regulators target g_tgt needs are sent to the worker (built lazily)
        rows = slice(None) if rows is None else rows
        active_src = active_cols[g_tgt]
        dense = len(active_src) == G
        yp = y_prot[rows] if dense else y_prot[rows][:, active_src]
        ypm = prev_prot(g_tgt, active_src)
        ypm = None if ypm is None else ypm[rows]
        mem = None if y_prot_mem is None else (y_prot_mem.nodes[rows][:, :, active_src], y_prot_mem.weights[rows][:, :, g_tgt])
        if sigma_spec_grn is not None and sigma_spec_grn[g_tgt, SPEC_LINK] != LINK_LOGISTIC:
            raise ValueError("the network fit supports the logistic link only")
        if sigma_spec_grn is not None and sigma_spec_grn[g_tgt, SPEC_PHI] != PHI_IDENTITY:
            # Non-identity phi: the fit acts on the features phi of the regulators of the target
            fam = int(sigma_spec_grn[g_tgt, SPEC_PHI])
            kappa = np.ascontiguousarray(sigma_spec_grn[active_src, SPEC_KAPPA], dtype=float)
            pbar = np.ascontiguousarray(sigma_spec_grn[active_src, SPEC_PBAR], dtype=float)
            phi = lambda Y: features(np.ascontiguousarray(Y, dtype=float), fam, kappa, pbar)
            yp = phi(yp)
            if mem is not None:
                n, Q, K = mem[0].shape
                mem = (phi(mem[0].reshape(n * Q, K)).reshape(n, Q, K), mem[1])
            if ypm is not None:
                n, Q, K = ypm.nodes.shape
                ypm = PathNodes(phi(ypm.nodes.reshape(n * Q, K)).reshape(n, Q, K), ypm.weights, ypm.kon, ypm.targets)
        if mem is not None and mem[0].shape[1] == 1 and np.all((mem[1] == 0) | (mem[1] == 1)):
            # Single node of weight 1 (late_mrna_nodes = 1): the end-state term at the lagged states, without cost
            yp = np.where(mem[1][:, :1] > 0.5, yp, mem[0][:, 0])
        elif mem is not None:  # end-state term on sigma averaged on the target's mRNA memory
            yp = MemStates(yp, mem[0], mem[1])
        return (g_tgt, G, active_src, y_samples[rows], yp, ypm,
                y_kon[rows, g_tgt], theta_init_mat[:, g_tgt], theta_ref_mat[:, g_tgt],
                ks[:, g_tgt] if ks.ndim == 3 else ks[g_tgt], n_samples,
                ref_network[:, g_tgt] if dense else ref_network[active_src, g_tgt],
                free_mask_2d[:, g_tgt], gene_kw, cond,
                None if act_c is None or cond is None else act_c[:, active_src].astype(float),
                None if act_c is None or cond is None else act_c[:, g_tgt],
                None if dist_targets is None else dist_term(dist_targets[2][rows], dist_targets[0][rows, g_tgt],
                                                            dist_targets[1][rows, g_tgt], dist_targets[3]))

    # Targets outside every condition's genes (single gene set: outside it) are not fitted
    genes = [g for g in range(n_stimuli, G) if act_c is None or act_c[:, g].any()]
    tasks = [(k, g) for k in range(len(subsets)) for g in genes]
    if Parallel is not None:
        results = Parallel(n_jobs=-1)(
            delayed(_infer_gene)(*gene_args(subsets[k], g)) for k, g in tasks
        )
    else:
        results = [_infer_gene(*gene_args(subsets[k], g)) for k, g in tasks]

    fits = []
    shape = (G, G) if cond is None else (C, G, G)
    for k in range(len(subsets)):
        basal, basal_tmp = np.zeros((n_samples, G)), np.zeros((n_samples, G))
        inter, inter_tmp = np.zeros(shape), np.zeros(shape)
        if act_c is not None and basal_init is not None:  # targets outside every gene set keep their initial basal
            basal[:, n_stimuli:], basal_tmp[:, n_stimuli:] = (basal_rows(basal_init)[:, n_stimuli:],) * 2
        for idx, g in enumerate(genes):
            basal[:, g], inter[..., g], basal_tmp[:, g], inter_tmp[..., g] = results[k * len(genes) + idx]
        if cond is not None:
            # Per sample: each sample takes the network of its condition
            inter, inter_tmp = inter[cond], inter_tmp[cond]
        fits.append((basal, inter, basal_tmp, inter_tmp))
    return fits


def inference_network(y_samples, y_kon, y_prot, y_prot_mod, ks, **kwargs):
    """Single joint network fit on all rows (see inference_network_multi). Returns (basal, inter, basal_tmp, inter_tmp)."""
    return inference_network_multi([None], y_samples, y_kon, y_prot, y_prot_mod, ks, **kwargs)[0]
