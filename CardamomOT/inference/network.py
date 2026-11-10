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
from scipy.special import expit, ndtri
from scipy.optimize import check_grad
from numba import njit
from functools import partial
from collections import namedtuple
import logging

from CardamomOT.logging import get_logger
from .sigma import (SPEC_PHI, SPEC_LINK, SPEC_KAPPA, SPEC_PBAR, SPEC_THRESH, SPEC_M, N_SPEC, PHI_IDENTITY, PHI_HERBACH,
                    LINK_LOGISTIC, HERBACH_U_MIN, logistic, logistic_pair, features)

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


def _log_failure(res, loss0, what):
    """Error for a failed L-BFGS-B fit; debug for a line-search stop after a decrease of the loss from loss0 (a kink of
    the objective, e.g. an exact Herbach edge at its repression bound: the point reached is the optimum to ~1 %)."""
    if res.success:
        return
    msg = str(res.message)
    if 'ABNORMAL_TERMINATION_IN_LNSRCH' in msg and res.fun < loss0:
        logger.debug('Line search stopped at a kink of the objective (%s): %s', what, msg)
    else:
        logger.error('Minimization failed for %s: %s', what, msg)


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


# Network costs of sigma against its target: 0 = cross-entropy, 1 = euclidean 2 (sigma - target)^2 (equal to the
# cross-entropy to second order at a target of 0.5), 2 = ordinal (cross-entropy per segment between the main modes)
COST_CODES = {'ce': 0, 'euclidean': 1, 'ordinal': 2}
ORD_EPS = 0.05  # ordinal cost: the central mode is sent to 1 - ORD_EPS (lower warp) and ORD_EPS (upper warp)
ORD_H = 0.02  # ordinal cost: width of the smoothing of the bend of the warps at the central mode

# Data term of the network fit (fit_network_cost), set in each worker by main_loop_inference; _LIK: on the rows with a
# real cell, the likelihood of its counts under the prior exp(-kappa C) of that cost (fit_likelihood)
_MODE = 0
_LIK = False


def cost_code(cost, allowed=('ce', 'euclidean', 'ordinal')):
    """Code (COST_CODES) of a network cost among allowed."""
    if cost not in allowed:
        raise ValueError(f"network cost must be one of {allowed}, got {cost!r}")
    return COST_CODES[cost]


def set_network_cost(cost, likelihood=False):
    """Data term of the network fits of this process: 'ce', 'euclidean' or 'ordinal' (and its prior on the levels
    with likelihood)."""
    global _MODE, _LIK
    _MODE, _LIK = cost_code(cost), bool(likelihood)


BALANCE_MIN_FREQ = 0.05  # fit_network_balance: floor of the frequency of a main mode (weights at most (1 / 0.05)^gamma)


TARGET_ROWS = ('w', 'mid', 'u1', 'u2', 'll', 'tl', 'kap', 'obs', 'lu1', 'lu2', 'lent')


class TargetArray(np.ndarray):
    """Sigmoid targets (N,) carrying row arrays (TARGET_ROWS), kept through slices and permutations, so that every
    term and the QUT use those of the target of each row: w, weight (fit_network_balance); mid, position of the
    central mode (ordinal cost; -1: two main modes); u1, u2, warped targets of the ordinal cost (expectations under
    the posteriors of the levels); ll (N, L), tl (N, L), kap (N,), obs (N,), log-likelihoods of the counts at the
    levels, positions of the levels, precision of the prior and rows with a real cell (likelihood cost); lu1, lu2,
    lent (N, L), targets of the levels in the network cost (lik_levels)."""

    def __array_finalize__(self, obj):
        for k in TARGET_ROWS:
            setattr(self, k, getattr(obj, k, None))

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        if isinstance(out, TargetArray):
            for k in TARGET_ROWS:
                v = getattr(self, k)
                setattr(out, k, None if v is None else v[idx])
        return out


def with_targets(y_target, **rows):
    """TargetArray of y_target with the row arrays rows (TARGET_ROWS; the others kept from y_target)."""
    out = np.asarray(y_target, dtype=float).view(TargetArray)
    for k in TARGET_ROWS:
        v = rows.get(k)
        setattr(out, k, getattr(y_target, k, None) if v is None else np.asarray(v, dtype=float))
    return out


def balance_weights(y_target, ys, ks, transit=None, gamma=1.0):
    """TargetArray of y_target (N,) with the weights of fit_network_balance: per sample, 1 / frequency^gamma of the
    main mode of each row (nearest main mode of the mixture ks, (n_modes,) or (n_samples, n_modes); transit
    (n_samples,): transition levels per interval between its main modes, a transitional row counting for its nearest
    main mode), frequency floored at BALANCE_MIN_FREQ, mean 1 over the rows of the sample: each main mode of a sample
    (with the transitions nearest to it, in their proportions) weighs frequency^(1 - gamma) in the fit, the same for
    all at gamma = 1."""
    ks = np.asarray(ks, dtype=float)
    ys = np.asarray(ys).astype(int)
    w = np.ones(len(y_target))
    for s in np.unique(ys):
        m = np.flatnonzero(ys == s)
        k = ks[min(s, len(ks) - 1)] if ks.ndim == 2 else ks
        kb = sigmoid_amplitudes(k)
        n_tr = 0 if transit is None else int(np.asarray(transit).ravel()[min(s, np.size(transit) - 1)])
        main = k[:1 + int(np.argmax(k))][::n_tr + 1]  # main modes: every (n_tr + 1)-th level (transit_groups)
        lv = np.unique(np.round(sigmoid_target(main, kb[0], kb[1]), 6))
        near = np.argmin(np.abs(np.asarray(y_target)[m, None] - lv[None]), axis=1)
        freq = np.bincount(near, minlength=len(lv)) / len(m)
        ws = np.maximum(freq[near], BALANCE_MIN_FREQ) ** -gamma
        w[m] = ws / ws.mean()
    return with_targets(y_target, w=w)


def _null_start(G, n_samples, ys, yt):
    """Start (G + n_samples,) of the null fits of the QUT: no interaction, basal of each sample at the logit of the
    weighted mean of its targets, the null optimum of the cross-entropy and of the euclidean cost (a start at 0 can
    stall the non-convex costs on a saturated plateau, where their gradient vanishes)."""
    x = np.zeros(G + n_samples)
    ys = np.asarray(ys).astype(int)
    w, t = np.broadcast_to(_row_w(yt), len(ys)), np.asarray(yt, dtype=float)
    for s in np.unique(ys):
        m = ys == s
        mu = np.clip(np.sum(w[m] * t[m]) / max(np.sum(w[m]), EPS), 1e-3, 1 - 1e-3)
        if s < n_samples:
            x[G + s] = np.log(mu / (1 - mu))
    return x


def _row_w(yt):
    """Weights of the rows of the targets yt (1 without fit_network_balance)."""
    w = getattr(yt, 'w', None)
    return 1.0 if w is None else w


def _row_mid(yt, n):
    """Central modes (n,) of the rows of the targets yt (-1: none)."""
    mid = getattr(yt, 'mid', None)
    return np.full(n, -1.0) if mid is None else np.ascontiguousarray(mid, dtype=float)


LIK_BIG = 1e6  # log-likelihood gap of the other levels of a row whose level is known (onehot_levels)


def onehot_levels(t, tl, ll):
    """Log-likelihoods (..., L) of rows whose level is known (target t (...)): 0 at the level of tl nearest to t,
    -LIK_BIG at the other levels of the gene (finite ll), -inf beyond: the likelihood of such a row is then
    -log p(level | sigma), in the units of that of the counts."""
    ok = np.isfinite(ll)
    near = np.argmin(np.where(ok, np.abs(np.asarray(tl) - np.asarray(t)[..., None]), np.inf), axis=-1)
    out = np.where(ok, -LIK_BIG, -np.inf)
    np.put_along_axis(out, near[..., None], 0.0, axis=-1)
    return out


def lik_cost_any(sig, ll, tl, kap, mode=0, mid=None):
    """Likelihood term (lik_terms) shifted by the best log-likelihood of each element (>= 0; 0 for a gene of a single
    level), for any leading shape: sig, kap, mid (...), ll, tl (..., L)."""
    shape = np.broadcast_shapes(np.shape(sig), np.shape(ll)[:-1])
    n, L = int(np.prod(shape)), np.shape(ll)[-1]

    def flat(x, last=()):
        # Writable contiguous rows (n, *last) of x broadcast to the shape (numba rejects read-only views)
        return np.array(np.broadcast_to(np.asarray(x, dtype=float), shape + last), dtype=float).reshape((n,) + last)
    ll2, tl2 = flat(ll, (L,)), flat(tl, (L,))
    ok = np.isfinite(ll2).astype(float)
    ll2 = np.where(ok > 0, ll2, 0.0)  # no infinity in the kernels
    C = ordinal_consts(flat(-1.0 if mid is None else mid)) if mode == 2 else np.zeros((0, 9))
    c = lik_terms(flat(sig), ll2, ok, *lik_levels(tl2, ok, mode, C), flat(kap), mode, C, False)[0]
    return (best_loglik(ll2, ok) + c).reshape(shape)


def lik_null_deviance(ll, tl, kap, mode=0, mid=None, n_grid=199, max_cells=2000):
    """Null deviance (G,) of the likelihood term (lik_cost_any) of the cells of one sample (ll, tl (n, G, L), kap
    (n, G): the same levels and kappa for every cell of the sample; central modes mid (G,)): mean cost of the best
    constant sigma, min_c mean_i lik_cost(c), on a grid (at most max_cells cells, evenly spaced); floored at that of a
    gene of standard deviation NULL_DEV_SD whose counts give the level of each cell (cells at the lowest and highest
    levels), as null_deviance."""
    ll, tl, kap = np.asarray(ll, float), np.asarray(tl, float), np.asarray(kap, float)
    ll = ll[::max(1, len(ll) // max_cells)]
    grid = np.linspace(1e-3, 1 - 1e-3, n_grid)
    p = 0.5 * (1 - np.sqrt(1 - 4 * NULL_DEV_SD ** 2))
    out = np.ones(ll.shape[1])
    for g in range(ll.shape[1]):
        ok = np.isfinite(ll[0, g])
        if ok.sum() < 2:
            continue
        lv = tl[0, g, ok]
        C = cost_any(np.repeat(grid[:, None], len(lv), axis=1), np.broadcast_to(lv, (n_grid, len(lv))), mode,
                     None if mid is None else np.full((n_grid, len(lv)), float(mid[g])))
        lp = -kap[0, g] * C
        lp = lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)  # log p(l | c), (n_grid, L_g)
        lg = ll[:, g, ok]
        dev = (lg.max(axis=1)[:, None] - np.logaddexp.reduce(lg[:, None, :] + lp[None], axis=2)).mean(axis=0)
        floor = -(1 - p) * lp[:, np.argmin(lv)] - p * lp[:, np.argmax(lv)]
        out[g] = max(dev.min(), floor.min())
    return out


def _loss_grad_elem(sig, yt, t=None, grad=True):
    """Element-wise data term of sigma against the targets t (yt by default; nodes of the flow matching), with the row
    arrays of yt (weight, central mode, likelihood of the counts) and the warped targets of t (yt by default; the
    warps of t when it carries none), times the weight of the row, and with grad its derivative in sigma (None
    otherwise), in one pass: (loss (n,), grad (n,) or None). With the likelihood (_LIK), every row in it, in the same
    units: the counts of the rows with a real cell (end states), the level of the target (onehot_levels) on the other
    rows and on the nodes."""
    t0 = np.ascontiguousarray(yt if t is None else t, dtype=float)
    sig = np.ascontiguousarray(sig, dtype=float)
    w = _row_w(yt)
    if _LIK and getattr(yt, 'll', None) is not None:
        ll = np.array(yt.ll, dtype=float)
        known = np.ones(len(t0), bool) if t is not None else ~(np.asarray(yt.obs) > 0)
        if np.any(known):
            ll[known] = onehot_levels(t0[known], yt.tl[known], ll[known])
        ok = np.isfinite(ll).astype(float)
        ll = np.where(ok > 0, ll, 0.0)  # no infinity in the kernels
        C = ordinal_consts(_row_mid(yt, len(t0))) if _MODE == 2 else np.zeros((0, 9))
        lev = (yt.lu1, yt.lu2, yt.lent) if getattr(yt, 'lu1', None) is not None else \
            lik_levels(np.ascontiguousarray(yt.tl, dtype=float), ok, _MODE, C)  # targets of the levels
        c, d = lik_terms(sig, ll, ok, *(np.ascontiguousarray(a, dtype=float) for a in lev),
                         np.ascontiguousarray(yt.kap, dtype=float), _MODE, C, grad)
        return w * c, (w * d if grad else None)
    if _MODE == 2:
        C = ordinal_consts(_row_mid(yt, len(t0)))
        src = yt if t is None else t
        u1, u2 = getattr(src, 'u1', None), getattr(src, 'u2', None)
        if u1 is None:
            u1, u2 = ordinal_targets_c(t0, C)
        c, d = ordinal_terms(sig, np.ascontiguousarray(u1, dtype=float), np.ascontiguousarray(u2, dtype=float), C,
                             grad)
        return w * c, (w * d if grad else None)
    if _MODE == 1:
        return w * net_cost(sig, t0, 1), (w * 4.0 * (sig - t0) if grad else None)
    return w * net_cost(sig, t0, 0), (w * grad_main_loss(sig, t0, 1) if grad else None)


def _d2loss(sig):
    """Gauss-Newton curvature of the data term in sigma (cross-entropy and euclidean cost)."""
    if _MODE == 2 or _LIK:
        raise ValueError("the ordinal network cost and the likelihood have no Gauss-Newton curvature "
                         "(mixed_effect_model_conditions)")
    return np.full(np.shape(sig), 4.0) if _MODE == 1 else 1.0 / np.maximum(sig * (1 - sig), EPS)


@njit(fastmath=True, cache=True)
def net_cost(y_pred, y_true, mode=0):
    """Element-wise cost of sigma against its target: cross-entropy (mode 0), or 2 (y_pred - y_true)^2 (mode 1; equal
    to the cross-entropy to second order at a target of 0.5); the ordinal cost is ordinal_cost."""
    if mode == 1:
        return 2.0 * (y_pred - y_true) ** 2
    return cross_entropy(y_pred, y_true)


@njit(fastmath=True, cache=True)
def _softplus(x):
    return max(x, 0.0) + np.log1p(np.exp(-abs(x)))


@njit(fastmath=True, cache=True)
def _ce1(p, t):
    """Cross-entropy of one prediction p against one target t (cross_entropy)."""
    pc, tc = min(max(p, eps_CE), 1 - eps_CE), min(max(t, eps_CE), 1 - eps_CE)
    return -(t * np.log(pc / tc) + (1 - t) * np.log((1 - pc) / (1 - tc)))


@njit(fastmath=True, cache=True)
def _dce1(p, t):
    """d _ce1 / d p (0 where p is clipped, as grad_main_loss)."""
    if p < eps_CE or p > 1 - eps_CE:
        return 0.0
    return (p - t) / (p * (1 - p))


@njit(fastmath=True, cache=True)
def ordinal_consts(mid):
    """Constants (n, 9) of the two warps of the ordinal cost at the central modes mid (n,): monotone maps of [0, 1]
    onto itself (0 and 1 fixed), of slope sa below m and sb above it, the bend smoothed over ORD_H, T(x) proportional
    to sa x + (sb - sa) ORD_H softplus((x - m) / ORD_H) - T(0) (_warps). Columns: m, then for each warp sa, sb, T(0)
    and 1 / (T(1) - T(0)) before normalisation; computed once per run of equal central modes (rows of one sample); m
    outside (0, 1) (no central mode) leaves the other columns at 0."""
    C = np.zeros((mid.shape[0], 9))
    for i in range(mid.shape[0]):
        m = mid[i]
        C[i, 0] = m
        if not 0.0 < m < 1.0:
            continue
        if i > 0 and m == mid[i - 1]:
            C[i] = C[i - 1]
            continue
        s0, s1 = ORD_H * _softplus(-m / ORD_H), ORD_H * _softplus((1.0 - m) / ORD_H)
        a1, b1, a2, b2 = (1.0 - ORD_EPS) / m, ORD_EPS / (1.0 - m), ORD_EPS / m, (1.0 - ORD_EPS) / (1.0 - m)
        C[i, 1], C[i, 2], C[i, 3], C[i, 4] = a1, b1, (b1 - a1) * s0, 1.0 / (a1 + (b1 - a1) * (s1 - s0))
        C[i, 5], C[i, 6], C[i, 7], C[i, 8] = a2, b2, (b2 - a2) * s0, 1.0 / (a2 + (b2 - a2) * (s1 - s0))
    return C


@njit(fastmath=True, cache=True)
def _warps(x, C, i, grad):
    """Both warps (T_1(x), T_2(x)) of the ordinal cost with the constants C[i] (ordinal_consts; one softplus shared by
    the two), and with grad their slopes (0 otherwise)."""
    z = (x - C[i, 0]) / ORD_H
    e = np.exp(-abs(z))
    sp = ORD_H * (max(z, 0.0) + np.log1p(e))
    t1 = (C[i, 1] * x + (C[i, 2] - C[i, 1]) * sp - C[i, 3]) * C[i, 4]
    t2 = (C[i, 5] * x + (C[i, 6] - C[i, 5]) * sp - C[i, 7]) * C[i, 8]
    if not grad:
        return t1, t2, 0.0, 0.0
    ex = 1.0 / (1.0 + e) if z >= 0 else e / (1.0 + e)
    return t1, t2, (C[i, 1] + (C[i, 2] - C[i, 1]) * ex) * C[i, 4], (C[i, 5] + (C[i, 6] - C[i, 5]) * ex) * C[i, 8]


@njit(fastmath=True, cache=True)
def ordinal_targets_c(t, C):
    """ordinal_targets with the constants C (n, 9) of the warps (ordinal_consts)."""
    u1, u2 = t.copy(), t.copy()
    for i in range(t.shape[0]):
        if 0.0 < C[i, 0] < 1.0:
            w1, w2, _, _ = _warps(t[i], C, i, False)
            u1[i], u2[i] = w1, w2
    return u1, u2


@njit(fastmath=True, cache=True)
def ordinal_targets(t, mid):
    """Warped targets (u1, u2) (1-D arrays) of the ordinal cost: the two warps of t (t itself without central mode)."""
    return ordinal_targets_c(t, ordinal_consts(mid))


@njit(fastmath=True, cache=True)
def ordinal_terms(p, u1, u2, C, grad):
    """Ordinal cost (n,) of p against the warped targets (u1, u2) with the constants C (n, 9) of the warps
    (ordinal_consts), and with grad its derivative in p (empty otherwise), in one pass; the cross-entropy against u1
    without central mode."""
    n = p.shape[0]
    out, d = np.empty(n), np.empty(n if grad else 0)
    for i in range(n):
        if not 0.0 < C[i, 0] < 1.0:
            out[i] = _ce1(p[i], u1[i])
            if grad:
                d[i] = _dce1(p[i], u1[i])
            continue
        t1, t2, s1, s2 = _warps(p[i], C, i, grad)
        out[i] = _ce1(t1, u1[i]) + _ce1(t2, u2[i])
        if grad:
            d[i] = _dce1(t1, u1[i]) * s1 + _dce1(t2, u2[i]) * s2
    return out, d


@njit(fastmath=True, cache=True)
def ordinal_cost_w(p, u1, u2, mid):
    """Ordinal cost (1-D arrays) of p against the warped targets (u1, u2) (ordinal_targets, or their expectations
    under the posteriors of the levels: the M-step of the EM, the cost being linear in its warped targets); the
    cross-entropy against u1 without central mode."""
    return ordinal_terms(p, u1, u2, ordinal_consts(mid), False)[0]


@njit(fastmath=True, cache=True)
def ordinal_cost(p, t, mid):
    """Ordinal cost (1-D arrays): with a central main mode at mid (0 < mid < 1), sum of the cross-entropies of p and
    t through two monotone warps of [0, 1] (0 and 1 fixed, ordinal_consts), sending the central mode to 1 - ORD_EPS and to
    ORD_EPS: the central mode is resolved as an end of both warps, as finely as the extreme modes, with the tails of
    the cross-entropy (finite cost, non-vanishing gradient) and 0 at p = t; the cross-entropy without central mode
    (mid outside (0, 1)), and for two main modes."""
    C = ordinal_consts(mid)
    u1, u2 = ordinal_targets_c(t, C)
    return ordinal_terms(p, u1, u2, C, False)[0]


@njit(fastmath=True, cache=True)
def _xlogx(t):
    """t log t + (1 - t) log(1 - t), clipped as _ce1: the part of the cross-entropy against t that depends on t only."""
    tc = min(max(t, eps_CE), 1 - eps_CE)
    return t * np.log(tc) + (1 - t) * np.log(1 - tc)


# The likelihood kernels take the levels of each row as a mask ok (n, L) (1: level of the gene, np.isfinite of the
# log-likelihoods, computed in numpy) and use no infinity: they may be inlined into fastmath callers (my_otdistance),
# where tests on -inf are not reliable.


@njit(fastmath=True, cache=True)
def lik_levels(tl, ok, mode, C):
    """Targets (u1, u2, ent) (n, L) of the levels tl (n, L) of the rows (mask ok (n, L)) in the network cost of code
    mode, computed once: the positions (u1 = tl, u2 = 0) for the cross-entropy and the euclidean cost, their two
    warps for the ordinal cost (constants C (n, 9), ordinal_consts; unused otherwise), and ent the part of the
    cross-entropies that depends on the levels only (_xlogx)."""
    n, L = ok.shape
    u1, u2, ent = np.zeros((n, L)), np.zeros((n, L)), np.zeros((n, L))
    for i in range(n):
        warped = mode == 2 and 0.0 < C[i, 0] < 1.0
        for l in range(L):
            if ok[i, l] <= 0.0:
                continue
            if warped:
                w1, w2, _, _ = _warps(tl[i, l], C, i, False)
                u1[i, l], u2[i, l], ent[i, l] = w1, w2, _xlogx(w1) + _xlogx(w2)
            else:
                u1[i, l] = tl[i, l]
                ent[i, l] = 0.0 if mode == 1 else _xlogx(tl[i, l])
    return u1, u2, ent


@njit(fastmath=True, cache=True)
def best_loglik(ll, ok):
    """Best log-likelihood (n,) of each row of ll (n, L) over its levels (mask ok), 0 for a row without."""
    out = np.zeros(ll.shape[0])
    for i in range(ll.shape[0]):
        found = False
        for l in range(ll.shape[1]):
            if ok[i, l] > 0.0 and (not found or ll[i, l] > out[i]):
                out[i], found = ll[i, l], True
    return out


@njit(fastmath=True, cache=True)
def lik_terms(sig, ll, ok, u1, u2, ent, kap, mode, C, grad):
    """Likelihood cost (n,) of the rows: -log sum_l exp(ll_l) p(l | sigma), p(l | sigma) proportional to
    exp(-kap C(sigma, tau_l)) on the levels of the mask ok (n, L) (0 for a row without), C the network cost of code
    mode (the prior of the basin updates) against the targets of the levels (lik_levels: u1, u2, ent (n, L);
    constants C (n, 9) of the ordinal cost), the logs of sigma (or of its two warps) computed once per row; with grad
    its derivative in sigma, kap sum_l (r_l - p_l) dC_l / d sigma (r the posteriors, p the prior), empty otherwise."""
    n, L = ll.shape
    out, d = np.zeros(n), np.zeros(n if grad else 0)
    lp, dc = np.zeros(L), np.zeros(L)
    for i in range(n):
        p = sig[i]
        warped = mode == 2 and 0.0 < C[i, 0] < 1.0
        x1, x2, s1, s2 = p, 0.5, 1.0, 0.0
        if warped:
            x1, x2, s1, s2 = _warps(p, C, i, grad)
        c1, c2 = min(max(x1, eps_CE), 1 - eps_CE), min(max(x2, eps_CE), 1 - eps_CE)
        a1, b1 = np.log(c1), np.log(1 - c1)
        a2, b2 = (np.log(c2), np.log(1 - c2)) if warped else (0.0, 0.0)
        # d CE / d sigma = (x - u) k, k = slope / (x (1 - x)), 0 where x is clipped (_dce1)
        k1 = s1 / (x1 * (1 - x1)) if eps_CE <= x1 <= 1 - eps_CE else 0.0
        k2 = s2 / (x2 * (1 - x2)) if warped and eps_CE <= x2 <= 1 - eps_CE else 0.0
        found, mz, ma = False, 0.0, 0.0  # max of the log-prior and of the log-joint over the levels
        for l in range(L):
            if ok[i, l] <= 0.0:
                continue
            if mode == 1:
                c, dc[l] = 2.0 * (p - u1[i, l]) ** 2, 4.0 * (p - u1[i, l])
            else:
                c = ent[i, l] - u1[i, l] * a1 - (1 - u1[i, l]) * b1 - u2[i, l] * a2 - (1 - u2[i, l]) * b2
                dc[l] = (x1 - u1[i, l]) * k1 + (x2 - u2[i, l]) * k2
            lp[l] = -kap[i] * c
            if not found or lp[l] > mz:
                mz = lp[l]
            if not found or ll[i, l] + lp[l] > ma:
                ma = ll[i, l] + lp[l]
            found = True
        if not found:
            continue
        sz, sa = 0.0, 0.0
        for l in range(L):
            if ok[i, l] > 0.0:
                sz += np.exp(lp[l] - mz)
                sa += np.exp(ll[i, l] + lp[l] - ma)
        lz, lm = mz + np.log(sz), ma + np.log(sa)
        out[i] = lz - lm
        if grad:
            g = 0.0
            for l in range(L):
                if ok[i, l] > 0.0:
                    g += (np.exp(ll[i, l] + lp[l] - lm) - np.exp(lp[l] - lz)) * dc[l]
            d[i] = kap[i] * g
    return out, d


def cost_curvature(sig, mode, mid=None, h=1e-3):
    """Curvature (any shape) of the network cost of code mode in its target at the target sigma, d^2 C(sigma, tau) /
    d tau^2 at tau = sigma: 1 / (sigma (1 - sigma)) for the cross-entropy, 4 for the euclidean cost (numerical for
    the ordinal cost): exp(-kappa C) is locally a Gaussian of variance 1 / (kappa curvature)."""
    s = np.clip(np.asarray(sig, dtype=float), 2 * h, 1 - 2 * h)
    if mode == 1:
        return np.full(s.shape, 4.0)
    if mode != 2:
        return 1.0 / (s * (1 - s))
    return (cost_any(s, s + h, 2, mid) + cost_any(s, s - h, 2, mid)) / h ** 2


def ordinal_params(mid):
    """Parameters (m, sa_1, sb_1, sa_2, sb_2, has) of the two warps of the ordinal cost (ordinal_consts) for the central modes
    mid (any shape); identity warps (sa = sb = 1) without central mode (has False: the second one is left out)."""
    mid = np.asarray(mid, dtype=float)
    has = (mid > 0.0) & (mid < 1.0)
    m = np.where(has, mid, 0.5)
    one = np.ones_like(m)
    return (m, np.where(has, (1 - ORD_EPS) / m, one), np.where(has, ORD_EPS / (1 - m), one),
            np.where(has, ORD_EPS / m, one), np.where(has, (1 - ORD_EPS) / (1 - m), one), has)


NULL_DEV_SD = 0.05  # floor of the null deviance: that of a gene of standard deviation 0.05 (cells at 0 or 1)


def null_deviance(tau, mode, mid=None):
    """Null deviance (G,) of each gene in the network cost of code mode (central modes mid (G,) for the ordinal
    cost): mean cost of the best constant prediction, min_c mean_i cost(c, tau_i), over the positions tau (n, G)
    (2 Var(tau) for the euclidean cost, the deviance of the logistic regression for the cross-entropy); floored at
    that of a gene of standard deviation NULL_DEV_SD."""
    tau = np.asarray(tau, dtype=float)
    mid = np.full(tau.shape[1], -1.0) if mid is None else np.asarray(mid, dtype=float)
    grid = np.linspace(1e-3, 1 - 1e-3, 999)
    p = 0.5 * (1 - np.sqrt(1 - 4 * NULL_DEV_SD ** 2))  # share at 1 of the floor gene, p (1 - p) = sd^2

    def dev(vals, f, m):
        c = cost_any(np.broadcast_to(grid[:, None], (len(grid), len(vals))),
                     np.broadcast_to(vals[None], (len(grid), len(vals))), mode, m)
        return float(np.min(c @ f))
    out = np.empty(tau.shape[1])
    for g in range(tau.shape[1]):
        vals, cnt = np.unique(np.round(tau[:, g], 6), return_counts=True)
        out[g] = max(dev(vals, cnt / cnt.sum(), mid[g]), dev(np.array([0.0, 1.0]), np.array([1 - p, p]), mid[g]))
    return out


def cost_any(y_pred, y_true, mode=0, mid=None):
    """Element-wise network cost of any shape: net_cost, or ordinal_cost (mode 2) with the central modes mid,
    broadcast to the shape (None: no central mode)."""
    if mode != 2:
        return net_cost(np.asarray(y_pred, dtype=float), np.asarray(y_true, dtype=float), mode)
    shape = np.shape(y_pred)
    mid = np.full(shape, -1.0) if mid is None else np.broadcast_to(np.asarray(mid, dtype=float), shape)
    # Writable copies (a broadcast view of one element stays read-only through ascontiguousarray: numba rejects it)
    return ordinal_cost(np.array(y_pred, dtype=float).ravel(),
                        np.array(np.broadcast_to(y_true, shape), dtype=float).ravel(),
                        np.array(mid, dtype=float).ravel()).reshape(shape)


def cost_gene_weights(ks, mode=0):
    """Gene weights of a network cost: w = 1 - k_low / k_high for the cross-entropy and the ordinal cost, w^2 for the
    euclidean cost (2 (sigma - tau)^2 w^2 = 2 ((kon_pred - kon) / k_high)^2: kon in units of the highest mode)."""
    w = gene_weights(ks)
    return w * w if mode == 1 else w


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


class HerbachArray(np.ndarray):
    """Inputs L = m log(P_k / s_k) (..., K) of the regulators of an exact Herbach target (sigma.py), with lbar (K,) =
    m log(Pbar_k / s_k), pbar (K,) of the columns and m the Hill coefficient of the target: its logit is basal +
    sum_k psi(t_k, P_k / s_k), t_k = herbach_theta(theta_k Pbar_k, lbar_k) (effect theta_k Pbar_k at the high mode),
    nonlinear in theta; the fits use _logit / _logit_grad on these arrays. learn: _logit_grad also returns the
    gradient w.r.t. the log-thresholds of the columns (_shift_thresholds). h: occupancies expit(L) = x^m / (1 + x^m),
    independent of theta, computed once (_as_herbach) and carried by the slices and reshapes of the array."""

    def __array_finalize__(self, obj):
        self.lbar, self.pbar = getattr(obj, 'lbar', None), getattr(obj, 'pbar', None)
        self.m, self.learn = getattr(obj, 'm', None), getattr(obj, 'learn', False)
        h = getattr(obj, 'h', None)  # reshapes keep it; slices set it in __getitem__
        self.h = h.reshape(self.shape) if h is not None and h.size == self.size else None

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        if isinstance(out, HerbachArray):
            h = getattr(self, 'h', None)
            out.h = None if h is None else h[idx]
        return out

    def __reduce__(self):  # keep the attributes through pickling (joblib workers)
        rec, args, state = super().__reduce__()
        return rec, args, state + (self.lbar, self.pbar, self.m, self.learn, getattr(self, 'h', None))

    def __setstate__(self, state):
        self.lbar, self.pbar, self.m, self.learn, self.h = state[-5:]
        super().__setstate__(state[:-5])


THRESH_BOUNDS = (0.01, 2.0)  # learned thresholds of the edges of the exact Herbach targets, x Pbar of the regulator


def herbach_inputs(Y, thresh, pbar, m):
    """HerbachArray m log(max(Y / thresh, tiny)) of the regulator levels Y (..., K), thresholds thresh and high-mode
    levels pbar (K,)."""
    thresh, pbar = np.asarray(thresh, dtype=float), np.asarray(pbar, dtype=float)
    x = np.maximum(np.asarray(Y, dtype=float) / thresh, 1e-12)
    return _as_herbach(float(m) * np.log(x), float(m) * np.log(pbar / thresh), pbar, float(m))


def _as_herbach(L, lbar, pbar, m=None, learn=False, h=None):
    out = np.asarray(L).view(HerbachArray)
    out.lbar, out.pbar, out.m, out.learn = lbar, pbar, m, learn
    out.h = expit(np.asarray(L)) if h is None else h
    return out


def _hill(Y):
    """Occupancies expit(L) (..., K) of the inputs Y (HerbachArray), computed once."""
    return Y.h if getattr(Y, 'h', None) is not None else expit(np.asarray(Y))


def _shift_thresholds(Y, d):
    """Inputs Y (HerbachArray, MemStates, PathNodes or None) with the log-thresholds of the columns shifted by d (K,),
    L - m d and lbar - m d, the gradients w.r.t. d returned by _logit_grad (learn); other inputs unchanged."""
    if isinstance(Y, HerbachArray):
        L = np.asarray(Y) - Y.m * d
        h = _hill(Y).copy()
        moved = np.asarray(d) != 0  # occupancies recomputed on the shifted columns only
        h[..., moved] = expit(L[..., moved])
        return _as_herbach(L, Y.lbar - Y.m * d, Y.pbar, Y.m, True, h)
    if isinstance(Y, MemStates):
        return MemStates(_shift_thresholds(Y.end, d), _shift_thresholds(Y.nodes, d), Y.weights)
    if isinstance(Y, PathNodes):
        return PathNodes(_shift_thresholds(Y.nodes, d), Y.weights, Y.kon, Y.targets)
    return Y


def _herbach_theta(Y, th):
    """Fold-change exponents t (K,) of the effects th Pbar of the columns of Y (HerbachArray), dt / d th and
    1 + dt / d lbar (both 0 at the repression bound)."""
    u = th * Y.pbar + np.logaddexp(0.0, Y.lbar)
    uc = np.maximum(u, HERBACH_U_MIN)
    inv = 1.0 / -np.expm1(-uc)
    on = u > HERBACH_U_MIN
    return uc + np.log(-np.expm1(-uc)) - Y.lbar, np.where(on, Y.pbar * inv, 0.0), np.where(on, inv * expit(Y.lbar), 0.0)


@njit(cache=True)
def _psi_sum(h, c, act):
    """sum_k log1p(c_k h_ik) (n,) over the columns act of h (n, K) (exact Herbach logit, single pass)."""
    out = np.zeros(h.shape[0])
    for i in range(h.shape[0]):
        s = 0.0
        for k in act:
            s += np.log1p(c[k] * h[i, k])
        out[i] = s
    return out


@njit(cache=True)
def _psi_grad(h, c, act, r, e):
    """e[k] = sum_i r_i (1 + c_k) h_ik / (1 + c_k h_ik) for the columns act, in place (expit(L + t).T @ r)."""
    for k in act:
        e[k] = 0.0
    for i in range(h.shape[0]):
        for k in act:
            e[k] += r[i] * (1.0 + c[k]) * h[i, k] / (1.0 + c[k] * h[i, k])


def _logit(Y, th):
    """Logit without basal (n,) of the inputs Y (n, K): Y @ th, or sum_k psi(t_k, .) for a Herbach target, psi =
    log((1 + e^t x^m) / (1 + x^m)) = log1p((e^t - 1) h), on the edges with theta != 0 only (psi = 0 at theta = 0)."""
    if isinstance(Y, HerbachArray):
        act = np.flatnonzero(np.asarray(th) != 0)
        if not len(act):
            return np.zeros(Y.shape[:-1])
        c = np.expm1(_herbach_theta(Y, th)[0])
        h = _hill(Y)
        if h.ndim == 2:
            return _psi_sum(np.ascontiguousarray(h), c, act)
        return np.log1p(h[..., act] * c[act]).sum(axis=-1)
    return Y @ th


def _logit_grad(Y, th, r):
    """sum_i r_i d logit_i / d th (K,) of the inputs Y (n, K): Y.T @ r, or (expit(t + L).T @ r) dt / d th
    (Herbach); with Y.learn, followed by the gradient (K,) w.r.t. the log-thresholds, -m (expit(t + L).T @ r
    (1 + dt / d lbar) - expit(L).T @ r)."""
    if isinstance(Y, HerbachArray):
        t, dt, c1 = _herbach_theta(Y, th)
        h = _hill(Y)
        hr = h.T @ r  # expit(L + t) = h at theta = 0 (t = 0)
        e = hr.copy()
        act = np.flatnonzero(np.asarray(th) != 0)
        if len(act):  # expit(L + t) = e^t h / (1 + (e^t - 1) h)
            _psi_grad(np.ascontiguousarray(h), np.expm1(t), act, np.ascontiguousarray(r, dtype=float), e)
        if not Y.learn:
            return e * dt
        return np.r_[e * dt, -Y.m * (e * c1 - hr)]
    return Y.T @ r


def _split_grad(gr, G):
    """(d / d theta (G,), d / d log-thresholds (G,) or 0) of a gradient of _logit_grad (scalar 0: no term)."""
    gr = np.asarray(gr)
    return (gr, 0.0) if gr.ndim == 0 or gr.shape[0] == G else (gr[:G], gr[G:])


def _sigma_of(basal, th, Y):
    """sigma (n,) of one target at the inputs Y (n, K), scalar basal, interactions th (K,)."""
    if isinstance(Y, HerbachArray):
        return logistic(basal + _logit(Y, th))
    return base_kon(basal, th, Y)


def _input_features(Y):
    """Linear-equivalent features of the inputs (the gradient of the logit at theta = 0): Y, or the normalised Hill
    ratios Pbar h(x) / h(xbar) of a Herbach target, those of the approximate form (ridge, null model)."""
    if isinstance(Y, HerbachArray):
        return Y.pbar * _hill(Y) / expit(Y.lbar)
    return np.asarray(Y, dtype=float)


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
        if mask.all():  # single sample: no copy of the inputs
            sigma[:] = _sigma_of(theta_basal_all[s_idx], inter_ref, yp)
        elif np.any(mask):
            sigma[mask] = _sigma_of(theta_basal_all[s_idx], inter_ref, yp[mask])
    return sigma


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


def _end_terms(sigma, yt, d=None, grad=True):
    """End-state term: data term of the sigmoid against its target yt (+ distribution term), and with grad its
    derivative in the logit per cell (None otherwise): (loss, d loss / d logit (n,))."""
    lv, dl = _loss_grad_elem(sigma, yt, grad=grad)
    loss = float(np.sum(lv))
    if d is not None:
        lw, dlw = _dist_loss_grad(sigma, d)
        loss += lw
        dl = dl + dlw if grad else None
    return loss, (dl * sigma * (1 - sigma) if grad else None)


class PathNodes:
    """
    Flow matching for one target (term of weight weight_flow_matching, the end-state term keeping
    1 - weight_flow_matching): states of its regulators at nodes drawn uniformly along the interval ending at each row,
    nodes (N, Q, K), weights (N, Q + 1) (column 0 for the end state, the row itself, then the nodes: 0 then 1 / Q, or
    1 on the end state for the rows without previous interval), and the kon of the target in force at each node,
    kon (N, Q) (mode of the start state before the switch alpha of the target, of the end state after it);
    targets (N, Q): their sigmoid targets (set by main_loop_inference; with the ordinal cost, a TargetArray carrying
    their warped targets).
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
        sq = logistic(basal_s + _logit(Yq, th))
        sig_bar += w * sq
        out.append((w, sq, Yq))
    return np.clip(sig_bar, 0.0, 1.0), out


def _mem_loss_grad(basal_s, th, Y, yt, grad=True, dist=None):
    """Cross-entropy of sigma_bar against yt (+ distribution term) and its gradients (d / d th (K,), d / d basal)."""
    sig_bar, terms = _mem_terms(basal_s, th, Y)
    lv, dl = _loss_grad_elem(sig_bar, yt, grad=grad)
    lw, dlw = (0.0, 0.0) if dist is None else _dist_loss_grad(sig_bar, dist)
    loss = float(np.sum(lv)) + lw
    if not grad:
        return loss, None, None
    dl = dl + dlw
    g_th, g_b = 0.0, 0.0  # (K,) after the first term, (2K,) with the log-thresholds (learn)
    for w, sq, Yq in terms:
        tmp = dl * w * sq * (1 - sq)
        g_th = g_th + _logit_grad(Yq, th, tmp)
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
        g_th, g_b = 0.0, 0.0  # (K,) after the first term, (2K,) with the log-thresholds (learn)
        for w, Y, tq in terms:
            w_s, Y_s = w[mask], Y[mask]
            if not np.any(w_s > 0):
                continue
            t_s = None if tq is None else tq[mask]  # weight and central mode of the row on its nodes too
            sig = logistic(basal[s_idx] + _logit(Y_s, th))
            lv, dl = _loss_grad_elem(sig, yt_s, t_s, grad=grad)
            Q += ws * np.sum(w_s * lv)
            if grad:
                tmp = dl * w_s * sig * (1 - sig)
                g_th += _logit_grad(Y_s, th, tmp)
                g_b += float(np.sum(tmp))
        if grad:
            out.append((s_idx, ws, g_th, g_b))
    return Q, out


def objective_grad(X, weights_samples, ys, yp, ypm, yt, G, g, n_samples,
                   theta_ref, ref_network, l_pen, weight_flow_matching, final,
                   constrain_basal_uniform=0.0, basal_free_mask=None, ridge=None, dist=None, grad=True):
    """
    Loss function for a single gene — joint over all samples — and with grad its gradient w.r.t. X, in one pass (the
    terms of each row computed once): (Q, dQ / dX or None).
    ridge : (G,) or None; Gaussian prior 0.5 * sum ridge * (theta - theta_ref)^2 (network_ridge).
    dist : DistTerm of the end-state term or None.

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
    theta_eff = theta_inter * ref_network

    Q = 0.0
    dq = np.zeros_like(X) if grad else None
    us = np.unique(ys.ravel())
    ns_unique: int = len(us)
    w_fm = _fm_weight(weight_flow_matching, ypm)

    if w_fm:
        q, out = _path_fit(lambda s: theta_eff, theta_basal, ys, us, _path_terms(yp, ypm), yt, weights_samples,
                           grad=grad)
        Q += w_fm * q
        for s_idx, ws, gt, gb in out:
            dq[:G] += w_fm * ref_network * gt * ws
            dq[G + s_idx] += w_fm * gb * ws
    mem = isinstance(yp, MemStates)  # sigma averaged on the mRNA memory of the target
    sigma = None if mem else _compute_sigma_per_sample(theta_inter, theta_basal, ref_network, yp, ys, us)
    for cnt, s in enumerate(us):
        s_idx = int(s)
        weight_s = (1 - w_fm) * np.sum(weights_samples) / weights_samples[cnt] / ns_unique
        mask = (ys == s)
        if mem:
            q, res, rb = _mem_loss_grad(theta_basal[s_idx], theta_eff, yp[mask], yt[mask], grad=grad,
                                        dist=_sl(dist, mask))
        else:
            q, tmp = _end_terms(sigma[mask], yt[mask], _sl(dist, mask), grad)
            if grad:
                res, rb = _logit_grad(yp[mask], theta_eff, tmp), np.sum(tmp)
        Q += weight_s * q
        if grad:
            dq[:G] += ref_network * res * weight_s
            dq[G + s_idx] += rb * weight_s

    d_inter = theta_inter - theta_ref[:G]
    Q += theta_penalization(d_inter * (ref_network > 0), l_pen)
    if grad:
        dq[:G] += grad_theta_penalization(d_inter, l_pen) * (ref_network > 0)
    if final:
        # L2 towards the reference on sqrt(w) * theta (costs 1/w on the effective interaction, as L1); the L1 keeps
        # its full level, so the QUT holds for the edges absent from the reference
        Q += r_elasticnet * l2_penalization(np.sqrt(ref_network) * d_inter, l_pen)
        if grad:
            dq[:G] += r_elasticnet * grad_l2_penalization(d_inter, l_pen) * ref_network
    if ridge is not None:
        Q += 0.5 * np.sum(ridge * d_inter ** 2)
        if grad:
            dq[:G] += ridge * d_inter

    # Uniformity penalty: push free-sample basals toward their common mean (gradient 2λ(θ_basal[s] − mean))
    if constrain_basal_uniform > 0.0 and basal_free_mask is not None:
        free_idx = np.where(basal_free_mask)[0]
        if len(free_idx) >= 2:
            theta_free = theta_basal[free_idx]
            mean_free  = theta_free.mean()
            Q += constrain_basal_uniform * np.sum((theta_free - mean_free) ** 2)
            if grad:
                dq[G + free_idx] += 2.0 * constrain_basal_uniform * (theta_free - mean_free)

    return Q, dq


def objective(X, *args, **kwargs):
    """Loss of objective_grad."""
    return objective_grad(X, *args, grad=False, **kwargs)[0]


def grad_theta(X, *args, **kwargs):
    """Gradient of objective w.r.t. X = theta (objective_grad)."""
    return objective_grad(X, *args, **kwargs)[1]


def _sigma_refinement(theta_basal, inter_with_diag, cond, Y, ys, us):
    """sigma = sigmoid(Z) of the refinement step, per-sample basal and per-condition network."""
    sigma = np.empty(Y.shape[0])
    for s in us:
        s_idx = int(s)
        mask = (ys == s)
        if np.any(mask):
            sigma[mask] = _sigma_of(theta_basal[s_idx], _of_cond(inter_with_diag, cond, s_idx), Y[mask])
    return sigma


def objective_refinement_grad(X, inter, weights_samples, ys, yp, ypm, yt,
                              diag, G, g, n_samples, l_pen, weight_flow_matching, cond=None, dist=None, ridge=None,
                              thresh=None, grad=True):
    """
    Loss for the refinement step with per-sample basal, and with grad its gradient w.r.t. X (interaction
    corrections, basals, log-threshold shifts), in one pass: (Q, dQ / dX or None).

    X shape (G + n_samples,), (2 G + n_samples,) with thresh
      0..G-1    : multiplicative corrections for inter
      G..G+ns-1 : per-sample basal values (free, additive)
      G+ns..    : shifts d of the log-thresholds of the edges (exact Herbach target, thresh = (log s (G,) current,
                  log s_0 (G,) of the regulators, prior sd tau): Gaussian prior on log s + d around log s_0)
    inter, diag : (G,), or (C, G) with cond[s] = condition of sample s
    (one correction per edge, shared by the conditions)
    """
    theta_inter = inter.copy() * X[:G]          # (G,) or (C, G)
    theta_basal = X[G:G + n_samples]             # (n_samples,)
    if thresh is not None:
        d = X[G + n_samples:]
        yp, ypm = _shift_thresholds(yp, d), _shift_thresholds(ypm, d)
    inter_with_diag = theta_inter + diag
    sl_d = slice(G + n_samples, None)

    Q = 0.0
    dq = np.zeros_like(X) if grad else None
    us = np.unique(ys.ravel())
    ns_unique: int = len(us)

    w_fm = _fm_weight(weight_flow_matching, ypm)
    if w_fm:
        q, out = _path_fit(lambda s: _of_cond(inter_with_diag, cond, s), theta_basal, ys, us,
                           _path_terms(yp, ypm), yt, weights_samples, grad=grad)
        Q += w_fm * q
        for s_idx, ws, gt, gb in out:
            gt, gd = _split_grad(gt, G)
            dq[:G] += w_fm * _of_cond(inter, cond, s_idx) * gt * ws
            dq[G + s_idx] += w_fm * gb * ws
            dq[sl_d] += w_fm * gd * ws
    mem = isinstance(yp, MemStates)  # sigma averaged on the mRNA memory of the target
    sigma = None if mem else _sigma_refinement(theta_basal, inter_with_diag, cond, yp, ys, us)
    for cnt, s in enumerate(us):
        s_idx = int(s)
        weight_s = (1 - w_fm) * np.sum(weights_samples) / weights_samples[cnt] / ns_unique
        mask = (ys == s)
        th = _of_cond(inter_with_diag, cond, s_idx)
        if mem:
            q, res, rb = _mem_loss_grad(theta_basal[s_idx], th, yp[mask], yt[mask], grad=grad, dist=_sl(dist, mask))
        else:
            q, tmp = _end_terms(sigma[mask], yt[mask], _sl(dist, mask), grad)
            if grad:
                res, rb = _logit_grad(yp[mask], th, tmp), np.sum(tmp)
        Q += weight_s * q
        if grad:
            res, rd = _split_grad(res, G)
            dq[:G] += _of_cond(inter, cond, s_idx) * res * weight_s
            dq[G + s_idx] += rb * weight_s
            dq[sl_d] += rd * weight_s

    # l_pen: (G,) per-edge penalty (1/w times the base one, see refine_inference)
    Q += theta_penalization(X[:g], l_pen[:g]) + theta_penalization(X[g + 1:G], l_pen[g + 1:G])
    if grad:
        dq[:g] += grad_theta_penalization(X[:g], l_pen[:g])
        dq[g + 1:G] += grad_theta_penalization(X[g + 1:G], l_pen[g + 1:G])
    if ridge is not None:  # Gaussian prior on the effects c_e theta_e (relaxed refinement)
        Q += 0.5 * np.sum(ridge * theta_inter ** 2)
        if grad:
            dq[:G] += (ridge * theta_inter * inter).reshape(-1, G).sum(axis=0)
    if thresh is not None:
        Q += np.sum((thresh[0] + d - thresh[1]) ** 2) / (2 * thresh[2] ** 2)
        if grad:
            dq[sl_d] += (thresh[0] + d - thresh[1]) / thresh[2] ** 2
    return Q, dq


def objective_refinement(X, *args, **kwargs):
    """Loss of objective_refinement_grad."""
    return objective_refinement_grad(X, *args, grad=False, **kwargs)[0]


def grad_correc(X, *args, **kwargs):
    """Gradient of objective_refinement w.r.t. X (objective_refinement_grad)."""
    return objective_refinement_grad(X, *args, **kwargs)[1]


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

    res = minimize(partial(objective_grad, **kw), X_flat, jac=True, method="L-BFGS-B", bounds=bounds,
                   options=_lbfgs_options(G))
    _log_failure(res, loss_fn(X_flat), 'inference')

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


### NETWORK CONDITIONS: theta_c = theta_shared + delta_c (L1 on the shared network and on delta - delta_ref)

# Penalty levels of the fit with network conditions (qut_conditions)
ConditionLevels = namedtuple('ConditionLevels', 'l_pen l_shared l_cond l_fuse omega ridge_w')
# Gaussian priors of the mixed model: precision of the deviations (G,) and of the patient means of the basals, ridge of
# the mean network (G,), patient of each sample (mixed_effect_model_conditions)
MixedPrior = namedtuple('MixedPrior', 'prec prec_basal ridge patient')


def _patient_mean_term(basal, free_mask, patient, prec):
    """Random effect on the patient means of the basals: 0.5 prec sum_p (mean_p - mean of the means)^2, the contrasts
    between the samples of a patient being free. Returns (value, gradient (n_samples,))."""
    g = np.zeros(len(basal))
    free = np.flatnonzero(free_mask)
    pats = np.unique(patient[free])
    if len(pats) < 2 or prec <= 0:
        return 0.0, g
    means = np.array([basal[free[patient[free] == p]].mean() for p in pats])
    dev = means - means.mean()
    for p, d in zip(pats, dev):
        idx = free[patient[free] == p]
        g[idx] += prec * d / len(idx)  # the derivative of the mean of the means vanishes (sum of dev = 0)
    return 0.5 * prec * float(np.sum(dev ** 2)), g


def _loss_grad_conditions(X, weights_samples, ys, yp, ypm, yt, G, n_samples,
                          cond, levels, theta_ref, ref_shared, delta_ref, w, weight_flow_matching, final,
                          constrain_basal_uniform=0.0, basal_free_mask=None, act=None, ridge=None, dist=None, mixed=None):
    """
    Loss and gradient for one gene with one network per condition.

    X = [shared (G,), delta (C, G), basal (n_samples,)] in theta units; the network of
    condition c is theta_c = shared + delta_c (effective = w * theta_c); sample s uses cond[s].
    Penalties (levels: ConditionLevels, see qut_conditions): l_shared L1(shared - ref_shared)
    + sum_c l_cond_c L1(theta_c - theta_ref_c) + sum_c l_fuse_c L1(w * (delta_c - delta_ref_c));
    delta_ref_c = theta_ref_c - mean_c theta_ref_c (0 without reference: common network).
    act (C, G) or None: 1 for the regulators selected in each condition (select_genes_per_condition), 0 for the
    others (no effect, no sparsity, no fused penalty: only the edges shared by conditions are fused).
    mixed: MixedPrior of the mixed model, adding its Gaussian terms on delta, on shared and on the patient means of the
    basals (see core_inference_mixed).
    """
    C = len(levels.l_cond)
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
            sig = _sigma_of(basal[s_idx], eff[c], Y[mask])
            q_s, tmp = _end_terms(sig, yt[mask], _sl(dist, mask))
            Q += ws * q_s
            g_eff[c] += ws * _logit_grad(Y[mask], eff[c], tmp)
            g_basal[s_idx] += ws * np.sum(tmp)

    g_theta = g_eff * wa
    for c in range(C):
        if levels.omega[c] == 0:  # condition absent from the fit
            continue
        on = (w > 0) if act is None else (w > 0) & (act[c] > 0)
        d = theta_c[c] - theta_ref[c]
        Q += theta_penalization(d * on, levels.l_cond[c])
        g_theta[c] += grad_theta_penalization(d, levels.l_cond[c]) * on
        if final:
            w_c = w if act is None else w * act[c]
            Q += levels.omega[c] * r_elasticnet * l2_penalization(np.sqrt(w_c) * d, levels.l_pen)
            g_theta[c] += levels.omega[c] * r_elasticnet * grad_l2_penalization(d, levels.l_pen) * w_c
        if ridge is not None:
            Q += levels.ridge_w[c] * 0.5 * np.sum(ridge * d ** 2)
            g_theta[c] += levels.ridge_w[c] * ridge * d

    # Effective deviations from the shared network pulled towards the reference ones
    g_delta = g_theta.copy()
    dd = (delta - delta_ref) * wa
    for c in range(C):
        if levels.l_fuse[c] > 0:
            Q += smoothed_l1_penalization(dd[c], levels.l_fuse[c])
            g_delta[c] += grad_smoothed_l1_penalization(dd[c], levels.l_fuse[c]) * wa[min(c, len(wa) - 1)]

    if constrain_basal_uniform > 0.0 and basal_free_mask is not None:
        free_idx = np.where(basal_free_mask)[0]
        if len(free_idx) >= 2:
            theta_free = basal[free_idx]
            mean_free = theta_free.mean()
            Q += constrain_basal_uniform * np.sum((theta_free - mean_free) ** 2)
            g_basal[free_idx] += 2.0 * constrain_basal_uniform * (theta_free - mean_free)

    d_shared = shared - ref_shared
    on_shared = (w > 0) if act is None else (w > 0) & np.any(act > 0, axis=0)
    Q += theta_penalization(d_shared * on_shared, levels.l_shared)
    g_shared = g_theta.sum(axis=0) + grad_theta_penalization(d_shared, levels.l_shared) * on_shared

    if mixed is not None:
        d_delta = delta - delta_ref
        Q += 0.5 * np.sum(mixed.prec * d_delta ** 2)
        g_delta += mixed.prec * d_delta
        Q += 0.5 * np.sum(mixed.ridge * d_shared ** 2)
        g_shared += mixed.ridge * d_shared
        q_b, g_b = _patient_mean_term(basal, np.ones(n_samples, bool) if basal_free_mask is None else basal_free_mask,
                                      mixed.patient, mixed.prec_basal)
        Q += q_b
        g_basal += g_b

    grad = np.concatenate([g_shared, g_delta.ravel(), g_basal])
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
    m, has = np.zeros(x.shape[1:]), on.any(axis=0)   # regulators absent from every condition of the target: 0
    if has.any():
        m[has] = np.nanmedian(np.where(on, x, np.nan)[:, has], axis=0)
    return m


def core_inference_conditions(y_samples, y_prot, y_prot_mod, y_target, theta_init, theta_ref,
                              ref_network, G, g, n_samples, cond,
                              levels, weight_flow_matching=.5, final=0,
                              constrain_basal_uniform=0.0, basal_free_mask=None,
                              hard_forcing_ref=False, ref_constraint_pct=0.1, seuil_min_network_intensity=1e-2, act=None,
                              ridge=None, dist=None):
    """
    core_inference with one network per condition (cond[s] = condition of sample s). act (C, G) or None: regulators
    selected in each condition (select_genes_per_condition); the shared network, the reference deviations and the
    fusion are then taken over the conditions where each regulator exists, and its other entries are 0.
    levels: penalty levels (ConditionLevels, see qut_conditions).

    theta_init / theta_ref : (C * G + n_samples,), condition c in c*G..(c+1)*G-1
    (theta units). The reference / sign bounds apply to the shared network (mean of the references), the
    penalty on the deviations pulls them towards those of the references (theta_ref_c - mean); with
    hard_forcing_ref the deviations are pinned to the reference ones. Returns the effective (C * G + n_samples,).
    """
    C = int(np.max(cond)) + 1
    w = prior_weight(np.asarray(ref_network, dtype=float))
    us = np.unique(y_samples)
    weights_samples = [np.sum(y_samples == s) ** alpha for s in us]

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
                 n_samples=n_samples, cond=cond, levels=levels, theta_ref=ref_c, ref_shared=ref_flat,
                 delta_ref=delta_ref, w=w, weight_flow_matching=weight_flow_matching, final=final,
                 constrain_basal_uniform=constrain_basal_uniform, basal_free_mask=basal_free_mask,
                 act=None if act is None else np.asarray(act, dtype=float), ridge=ridge, dist=dist)
    # Stage 1: deviations pinned to the reference ones (0 without reference), well conditioned; stage 2:
    # free deviations, started from stage 1 or the initial deviations (the stiff term on delta alone stalls L-BFGS-B)
    d_sl = slice(G, (C + 1) * G)
    X_common = X_flat.copy()
    X_common[d_sl] = delta_ref.ravel()
    bounds_common = bounds[:G] + pinned + bounds[(C + 1) * G:]
    res = minimize(fn, X_common, jac=True, method="L-BFGS-B", bounds=bounds_common, options=_lbfgs_options(G))
    if np.all(np.isfinite(levels.l_fuse)) and not (hard_forcing_ref and not final):
        starts = [res.x, np.concatenate([res.x[:G], X_flat[d_sl], res.x[(C + 1) * G:]])]
        x0 = min(starts, key=lambda x: fn(x)[0])
        res = minimize(fn, x0, jac=True, method="L-BFGS-B", bounds=bounds, options=_lbfgs_options(C * G))
    _log_failure(res, fn(X_common)[0], 'inference (network conditions)')

    if check_gradient:
        error = check_grad(lambda x: fn(x)[0], lambda x: fn(x)[1], res.x)
        if error > .05:
            logger.debug("Gradient theta inference (conditions) check error for gene %s: %s", g, error)

    shared = res.x[:G]
    delta = res.x[G:(C + 1) * G].reshape(C, G)
    theta_c = fuse_conditions((shared[None] + delta) * w[None], act=act)
    return np.concatenate([theta_c.ravel(), res.x[(C + 1) * G:]])


### MIXED MODEL: psi_p = theta_mean + eps_p, eps_p ~ N(delta_ref_p, tau2 / (sd w)^2), one patient = one network condition

def _mixed_weights(y_samples):
    """Weights of the samples of a fit (weights_samples) and scale m of the data term: the cells of a sample count once,
    up to n_ref = median of the sizes of the samples (no sample is inflated, cells beyond n_ref bring no information).
    The loss weighs sample s by m c_s, with c_s = min(1, n_ref / n_s) and m = mean(1 / c_s)."""
    n_s = np.array([np.sum(y_samples == s) for s in np.unique(y_samples)], dtype=float)
    c_s = np.minimum(1.0, np.median(n_s) / n_s)
    return list(1.0 / c_s), float(np.mean(1.0 / c_s))


def _logit_jacobian(Y, th):
    """d logit / d th (n, K) of the inputs Y (n, K) at th (see _logit_grad)."""
    if isinstance(Y, HerbachArray):
        t, dt, _ = _herbach_theta(Y, th)
        c, h = np.expm1(t), _hill(Y)
        return (1.0 + c) * h / (1.0 + c * h) * dt
    return np.asarray(Y, dtype=float)


def _gn_blocks(basal_s, th, Y, weights=None):
    """Gauss-Newton blocks (Hxx (K, K), hbx (K,), hbb) of the data term of one sample at (basal_s, th), rows weighted
    by weights; Y MemStates: of sigma averaged on the memory of the target."""
    if isinstance(Y, MemStates):
        sig_bar, terms = _mem_terms(basal_s, th, Y)
        D, Db = 0.0, 0.0
        for w_q, s_q, Y_q in terms:
            v = w_q * s_q * (1 - s_q)
            D = D + v[:, None] * _logit_jacobian(Y_q, th)
            Db = Db + v
        inv = _d2loss(sig_bar)
        if weights is not None:
            inv = inv * weights
        return (D * inv[:, None]).T @ D, D.T @ (Db * inv), float(np.sum(Db ** 2 * inv))
    s = logistic(basal_s + _logit(Y, th))
    v = _d2loss(s) * (s * (1 - s)) ** 2
    v = v if weights is None else weights * v
    J = _logit_jacobian(Y, th)
    return (J * v[:, None]).T @ J, J.T @ v, float(np.sum(v))


def _sample_blocks(basal_s, eff, Y, ypm, w_fm, ws, w):
    """Gauss-Newton blocks of the loss of one sample in theta units (effective = w theta): end-state and path terms."""
    a, b, d = _gn_blocks(basal_s, eff, Y)
    hxx, hbx, hbb = (1 - w_fm) * ws * a, (1 - w_fm) * ws * b, (1 - w_fm) * ws * d
    if w_fm:
        for wq, Yq, _ in _path_terms(Y, ypm):
            if np.any(wq > 0):
                a, b, d = _gn_blocks(basal_s, eff, Yq, wq)
                hxx, hbx, hbb = hxx + w_fm * ws * a, hbx + w_fm * ws * b, hbb + w_fm * ws * d
    return hxx * np.outer(w, w), hbx * w, hbb


def qut_mixed(y_samples, y_prot, y_prot_mod, y_target, ref_network, G, g, n_samples, weight_flow_matching, cond, prec,
              weights, m, level=0.5, n_perm=20):
    """
    Quantile universal threshold of the L1 on the mean network of the mixed model. At theta_mean_j = 0 with the
    deviations fitted, the gradient of the objective is sum_p R_p s_p, with R_p = (I + H_p P^-1)^-1 (H_p: Gauss-Newton
    matrix of patient p at the null, basals eliminated, P = diag(prec)); level-quantile of its max norm under the null
    of the model: no edge on average, s_p the score of the target permuted within each sample + H_p e_p, e_p ~ N(0,
    m / prec) the deviations of the patient. tau2 -> 0 gives the common network QUT.
    """
    C = int(np.max(cond)) + 1
    w = prior_weight(ref_network)
    on = (w > 0) & (prec > 0)
    if not on.any():
        return 0.0
    ys = np.asarray(y_samples).astype(int)
    us = np.unique(ys)
    kw = dict(G=G, g=g, n_samples=n_samples, theta_ref=np.zeros(G + n_samples), ref_network=w, l_pen=0.0,
              weight_flow_matching=weight_flow_matching, final=0)
    null = dict(kw, ys=y_samples, yp=y_prot, ypm=y_prot_mod, yt=y_target, weights_samples=weights)
    bounds = [(0.0, 0.0)] * G + [(-_basal_bound(G), _basal_bound(G))] * n_samples
    X0 = minimize(partial(objective_grad, **null), _null_start(G, n_samples, y_samples, y_target),
                  jac=True,
                  method="L-BFGS-B", bounds=bounds, options=_lbfgs_options(G)).x
    w_fm = _fm_weight(weight_flow_matching, y_prot_mod)
    ws_of = {s: np.mean(weights) / weights[i] for i, s in enumerate(us)}
    H = np.zeros((C, G, G))
    for s in us:
        r = ys == s
        hxx, hbx, hbb = _sample_blocks(X0[G + s], np.zeros(G), y_prot[r], None if y_prot_mod is None else y_prot_mod[r],
                                       w_fm, ws_of[s], w)
        H[cond[s]] += hxx - np.outer(hbx, hbx) / max(hbb, EPS)
    idx = np.flatnonzero(on)
    H = H[:, idx][:, :, idx]
    P_inv = 1.0 / prec[idx]
    R = np.linalg.inv(np.eye(len(idx))[None] + H * P_inv[None, :])  # (C, K, K)
    groups = [np.flatnonzero(ys == s) for s in us]
    conds = [c for c in range(C) if np.any(cond[us] == c)]
    rows_c = {c: np.isin(ys, us[cond[us] == c]) for c in conds}
    w_c = {c: [weights[i] for i, s in enumerate(us) if cond[s] == c] for c in conds}
    scale = {c: np.mean(weights) / np.mean(w_c[c]) for c in conds}
    rng = np.random.default_rng(1000 * g + len(y_target))
    stats = np.empty(n_perm)
    for k in range(n_perm):
        p = np.arange(len(y_target))
        for ind in groups:
            p[ind] = rng.permutation(ind)
        ypm = y_prot_mod
        if ypm is not None:
            ypm = PathNodes(ypm.nodes, ypm.weights, None if ypm.kon is None else ypm.kon[p],
                            None if ypm.targets is None else ypm.targets[p])
        yt = y_target[p]
        tot = np.zeros(len(idx))
        for c in conds:
            r = rows_c[c]
            s_c = scale[c] * grad_theta(X0, ys=y_samples[r], yp=y_prot[r], ypm=None if ypm is None else ypm[r], yt=yt[r],
                                        weights_samples=w_c[c], **kw)[:G][idx]
            e_c = rng.normal(0.0, np.sqrt(m * P_inv))
            tot += R[c] @ (s_c + H[c] @ e_c)
        stats[k] = np.max(np.abs(tot))
    return float(np.quantile(stats, level))


def core_inference_mixed(y_samples, y_prot, y_prot_mod, y_target, theta_init, theta_ref, ref_network, G, g, n_samples,
                         cond, tau2, tau2_basal, weights, m, weight_flow_matching=.5, basal_free_mask=None,
                         ref_constraint_pct=0.1, seuil_min_network_intensity=1e-2, ridge=None, dist=None,
                         level=0.5, n_perm=20):
    """
    Mixed model of one target, one patient per network condition: psi_p = theta_mean + eps_p, with eps_p (effective
    units) ~ N(delta_ref_p, tau2) and a random effect of variance tau2_basal on the patient means of the basals.
    L1 on theta_mean at its QUT (qut_mixed), Gaussian priors otherwise. Edges out of the support of theta_mean: eps = 0,
    except those of a patient whose posterior z exceeds the universal threshold at level. Returns the effective
    networks and basals (C * G + n_samples,) and (s_theta, n_theta, s_basal, n_basal, z): statistics of the Laplace-EM
    of the variances (sum of E[eps^2] and their count on the support of theta_mean, REML; sum over the patients
    of E[(mean_p - mean)^2] and P - 1) and posterior z (C, G) of the deviations.
    """
    C = int(np.max(cond)) + 1
    w = prior_weight(np.asarray(ref_network, dtype=float))
    us = np.unique(y_samples).astype(int)
    prec = m * w ** 2 / tau2
    free_mask = np.ones(n_samples, bool) if basal_free_mask is None else np.asarray(basal_free_mask, dtype=bool)
    free_mask = free_mask & np.isin(np.arange(n_samples), us)
    mixed = MixedPrior(prec, m / tau2_basal, (m * ridge) if ridge is not None else np.zeros(G), np.asarray(cond))

    init_c = np.asarray(theta_init[:C * G], dtype=float).reshape(C, G)
    ref_c = np.asarray(theta_ref[:C * G], dtype=float).reshape(C, G)
    ref_flat = ref_c.mean(axis=0)
    delta_ref = ref_c - ref_flat[None]
    shared0 = init_c.mean(axis=0)

    wfm = weight_flow_matching
    lam = qut_mixed(y_samples, y_prot, y_prot_mod, y_target, ref_network, G, g, n_samples, wfm, cond, prec, weights, m,
                    level, n_perm)
    omega = np.bincount(cond[us], minlength=C) / len(us)
    levels = ConditionLevels(lam, lam, np.zeros(C), np.zeros(C), omega, np.zeros(C))
    fn = partial(_loss_grad_conditions, weights_samples=weights, ys=y_samples, yp=y_prot, ypm=y_prot_mod, yt=y_target,
                 G=G, n_samples=n_samples, cond=cond, levels=levels, theta_ref=ref_c, ref_shared=ref_flat,
                 delta_ref=delta_ref, w=w, weight_flow_matching=wfm, final=0, basal_free_mask=free_mask, dist=dist,
                 mixed=mixed)
    bounds_s = _interaction_bounds(ref_flat, np.ascontiguousarray(ref_network, dtype=float), G, 0, False,
                                   ref_constraint_pct, seuil_min_network_intensity)
    bounds_d = [(min(-2 * max(-lo, hi), dr), max(2 * max(-lo, hi), dr))
                for dr_c in delta_ref for (lo, hi), dr in zip(bounds_s, dr_c)]
    bounds_b = [(-_basal_bound(G), _basal_bound(G))] * n_samples
    pinned = [(v, v) for v in delta_ref.ravel()]
    b0 = np.asarray(theta_init[C * G:], dtype=float)
    # Stage 1: deviations at the reference ones, well conditioned; stage 2: deviations free on every candidate edge
    res = minimize(fn, np.concatenate([shared0, delta_ref.ravel(), b0]), jac=True, method="L-BFGS-B",
                   bounds=bounds_s + pinned + bounds_b, options=_lbfgs_options(G))
    x1 = res.x
    res = minimize(fn, x1, jac=True, method="L-BFGS-B", bounds=bounds_s + bounds_d + bounds_b,
                   options=_lbfgs_options(C * G))
    _log_failure(res, fn(x1)[0], 'inference (mixed model)')

    def blocks(x):
        # Gauss-Newton blocks per sample at x, theta units
        sh, de, ba = x[:G], x[G:(C + 1) * G].reshape(C, G), x[(C + 1) * G:]
        w_fm = _fm_weight(wfm, y_prot_mod)
        out = {}
        for i, s in enumerate(us):
            r = y_samples == s
            out[s] = _sample_blocks(ba[s], (sh + de[cond[s]]) * w, y_prot[r],
                                    None if y_prot_mod is None else y_prot_mod[r], w_fm, np.mean(weights) / weights[i], w)
        return out

    candidates = (w > 0) & (prec > 0)
    support = (np.abs(res.x[:G] * w) >= seuil_min_network_intensity) & candidates
    # Patient-specific edges: posterior z of the deviations, given theta_mean and the basals
    blk = blocks(res.x)
    delta = res.x[G:(C + 1) * G].reshape(C, G) - delta_ref
    idx = np.flatnonzero(candidates)
    z = np.zeros((C, G))
    for c in range(C):
        ss = [s for s in us if cond[s] == c]
        if not ss or not len(idx):
            continue
        hc = sum(blk[s][0] - np.outer(blk[s][1], blk[s][1]) / max(blk[s][2], EPS) for s in ss)
        var = m * np.diag(np.linalg.inv(hc[np.ix_(idx, idx)] + np.diag(prec[idx])))
        z[c, idx] = delta[c, idx] / np.sqrt(np.maximum(var, EPS))
    n_tests = max(len(idx) * int(np.sum(omega > 0)), 1)
    z_star = ndtri(1.0 - (1.0 - level ** (1.0 / n_tests)) / 2.0)
    keep = support[None] | (np.abs(z) > z_star)

    # Stage 3: kept entries only, the network returned
    lo_hi_s = [b if on else (r, r) for b, on, r in zip(bounds_s, support, ref_flat)]
    lo_hi_d = [b if k else (dr, dr) for b, k, dr in zip(bounds_d, keep.ravel(), delta_ref.ravel())]
    x0 = np.clip(res.x, [b[0] for b in lo_hi_s + lo_hi_d + bounds_b], [b[1] for b in lo_hi_s + lo_hi_d + bounds_b])
    res = minimize(fn, x0, jac=True, method="L-BFGS-B", bounds=lo_hi_s + lo_hi_d + bounds_b, options=_lbfgs_options(C * G))
    _log_failure(res, fn(x0)[0], 'inference (mixed model, kept entries)')
    x_net = res.x
    # Statistics of the variances only: L1 relaxed on the support, as it pulls the mean network towards 0 and the
    # deviations of all the patients the other way, inflating tau2 (relaxed in the network, it misleads the refinement)
    fn = partial(fn.func, **{**fn.keywords, 'levels': levels._replace(l_shared=0.0)})
    res = minimize(fn, x_net, jac=True, method="L-BFGS-B", bounds=lo_hi_s + lo_hi_d + bounds_b, options=_lbfgs_options(C * G))
    _log_failure(res, fn(x_net)[0], 'inference (mixed model, statistics)')

    # Laplace-EM statistics of tau2 and tau2_basal, REML: joint Gauss-Newton matrix of the kept entries
    blk = blocks(res.x)
    delta = res.x[G:(C + 1) * G].reshape(C, G) - delta_ref
    basal = res.x[(C + 1) * G:]
    s_idx = np.flatnonzero(support)
    d_idx = [np.flatnonzero(keep[c] & candidates) for c in range(C)]
    pos_s = np.arange(len(s_idx))
    pos_b = len(s_idx) + np.arange(len(us))
    dim, pos_d = pos_b[-1] + 1, []
    for c in range(C):
        pos_d.append(dim + np.arange(len(d_idx[c])))
        dim += len(d_idx[c])
    H = np.eye(dim) * 1e-8
    H[pos_s, pos_s] += mixed.ridge[s_idx]
    for c in range(C):
        H[pos_d[c], pos_d[c]] += prec[d_idx[c]]
    for i, s in enumerate(us):
        hxx, hbx, hbb = blk[s]
        c = cond[s]
        ps, js = np.r_[pos_s, pos_d[c]], np.r_[s_idx, d_idx[c]]
        H[np.ix_(ps, ps)] += hxx[np.ix_(js, js)]
        H[pos_b[i], ps] += hbx[js]
        H[ps, pos_b[i]] += hbx[js]
        H[pos_b[i], pos_b[i]] += hbb
    free = np.flatnonzero(free_mask)
    pats = np.unique(cond[free]) if len(free) else np.array([], dtype=int)
    contrasts = None
    if len(pats) >= 2:
        contrasts = np.zeros((len(pats), len(us)))
        mean_row = np.zeros(len(us))
        for k, p in enumerate(pats):
            mem = free[cond[free] == p]
            for s in mem:
                contrasts[k, np.searchsorted(us, s)] = 1.0 / len(mem)
            mean_row += contrasts[k] / len(pats)
        contrasts = contrasts - mean_row[None]
        H[np.ix_(pos_b, pos_b)] += mixed.prec_basal * contrasts.T @ contrasts
    V = m * np.linalg.inv(H)
    s_theta, n_theta = 0.0, 0
    for c in range(C):
        if omega[c] == 0:
            continue
        on = np.isin(d_idx[c], s_idx)
        js, ps = d_idx[c][on], pos_d[c][on]
        s_theta += float(np.sum(w[js] ** 2 * (delta[c, js] ** 2 + V[ps, ps])))
        n_theta += len(js)
    s_basal, n_basal = 0.0, 0
    if contrasts is not None:
        vb = V[np.ix_(pos_b, pos_b)]
        means = contrasts @ basal[us]
        s_basal = float(np.sum(means ** 2) + np.trace(contrasts @ vb @ contrasts.T))
        n_basal = len(pats) - 1

    shared, delta_full = x_net[:G], x_net[G:(C + 1) * G].reshape(C, G)
    theta_c = (shared[None] + delta_full) * w[None]
    return np.concatenate([theta_c.ravel(), x_net[(C + 1) * G:]]), (s_theta, n_theta, s_basal, n_basal, z)


def learn_mixed_variances(fit_stats, G, d0, n_steps=7, bracket=(1e-4, 100.0)):
    """
    Variances of the mixed model on frozen trajectories. fit_stats(tau2, tau2_b): (G, 4) statistics of a network fit
    (s_theta, n_theta, s_basal, n_basal per target). Their REML score equations tau2 = s / n are solved by bisection on
    log tau2, EM steps being too slow near 0: tau2_0 and tau2_b pooled over the targets, then tau2 of each target with
    the inverse-gamma prior of d0 entries centred on tau2_0, tau2 = (s + d0 tau2_0) / (n + d0) (a target without edge
    gets tau2_0). Returns (tau2 (G,), tau2_b, tau2_0).
    """
    def bisect(above, x0):
        # above(x): fixed-point map of x is above x; no root in the bracket: its end
        lo, hi = np.full_like(x0, bracket[0], dtype=float), np.full_like(x0, bracket[1], dtype=float)
        for _ in range(n_steps):
            mid = np.sqrt(lo * hi)
            up = above(mid)
            lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
        return np.sqrt(lo * hi)

    def pooled(x):
        s = np.sum(fit_stats(x[0], x[1]), axis=0)
        return np.array([s[0] > x[0] * s[1] and s[1] > 0, s[2] > x[1] * s[3] and s[3] > 0])

    tau2_0, tau2_b = bisect(pooled, np.ones(2))

    def per_target(x):
        s = fit_stats(x, tau2_b)
        return s[:, 0] + d0 * tau2_0 > x * (s[:, 1] + d0)

    return bisect(per_target, np.ones(G)), float(tau2_b), float(tau2_0)


def refine_inference(y_samples, y_prot, y_prot_mod, y_target, inter, basal, theta_ref,
                     G, g, n_samples,
                     l_pen, weight_flow_matching=.5, final=0,
                     hard_forcing_ref=False, ref_constraint_pct=0.1, ref_network=None, seuil_min_network_intensity=1e-2,
                     cond=None, dist=None, ridge=None, thresh_prior=None, sample_weights=None):
    """
    Refinement step: optimise multiplicative correction factors over inter, per-sample basals refitted freely
    (additive, any sign); for an exact Herbach target with thresh_prior = (log s_0 (G,), tau), the log-thresholds of
    its edges too (kept edges, |theta| >= seuil_min_network_intensity, out of self-regulation, Gaussian prior of sd
    tau around log s_0, thresholds within
    THRESH_BOUNDS x Pbar): a lower threshold lifts the repression bound. Refined values stay within max_inter and _basal_bound(G);
    sign-forced edges (|ref_network| > 1) keep |theta| >= seuil_min_network_intensity.

    Adaptive lasso (Zou 2006) with weights 1 / |theta_tmp|, i.e. Breiman's non-negative garrote:
    theta_e = c_e theta_tmp_e with c_e >= 0, so l_pen |c_e| = l_pen |theta_e| / |theta_tmp_e|
    (l_pen: quantile universal threshold of the garrote itself, qut_garrote),
    within the support and signs of theta_tmp (self-regulation held at its first-fit value, which anchors the scale of
    the corrections to the inferred structure; basals unpenalized). final: no bounds around the
    reference (signed priors only), as in core_inference. ridge (G,) or None: Gaussian prior on the effects
    c_e theta_tmp_e (ridge_coefficients), with l_pen = 0 in the relaxed refinement (network_relaxed).

    basal : (n_samples,)
    inter : (G,), or (C, G) per condition (cond[s] = condition of sample s):
        one correction per edge, shared by the conditions, keeps their differences.
    Returns updated (inter, basal, log-thresholds (G,) of the edges or None).
    """
    diag = np.zeros(inter.shape)
    if g < G:   # g == G is the sentinel meaning "no self-regulation in active set"
        diag[..., g] = inter[..., g]
    inter = inter.copy()
    inter -= diag
    basal = basal.copy()  # (n_samples,)

    weights_samples = sample_weights if sample_weights is not None else \
        [np.sum(y_samples == s) ** alpha for s in np.unique(y_samples)]

    X_flat = np.r_[np.ones(G), np.clip(basal, -_basal_bound(G), _basal_bound(G))]

    # Corrections: interactions in [0, 2] (selection, little expansion); basals free within their capacity
    max_bounds = 3
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

    # Log-thresholds of the edges (exact Herbach target): shifts d from the current ones, the others fixed
    thresh = None
    Y = _end(y_prot)
    if thresh_prior is not None and isinstance(Y, HerbachArray) and G:
        log_s = np.log(Y.pbar) - Y.lbar / Y.m
        lo = np.minimum(np.log(THRESH_BOUNDS[0] * Y.pbar) - log_s, 0.0)
        hi = np.maximum(np.log(THRESH_BOUNDS[1] * Y.pbar) - log_s, 0.0)
        # Kept edges (self-regulation removed above): the thresholds of the negligible ones have no meaning
        on = np.any(np.abs(inter.reshape(-1, G)) >= seuil_min_network_intensity, axis=0)
        X_flat = np.r_[X_flat, np.zeros(G)]
        bounds += [(l, h) if o else (0.0, 0.0) for l, h, o in zip(lo, hi, on)]
        thresh = (log_s, np.asarray(thresh_prior[0], dtype=float), float(thresh_prior[1]))

    kw = dict(inter=inter,
              weights_samples=weights_samples, ys=y_samples, yp=y_prot, ypm=y_prot_mod, yt=y_target,
              diag=diag, G=G, g=g, n_samples=n_samples,
              l_pen=l_pen, weight_flow_matching=weight_flow_matching, cond=cond, dist=dist, ridge=ridge,
              thresh=thresh)
    loss_fn = partial(objective_refinement, **kw)
    grad_fn = partial(grad_correc, **kw)

    res = minimize(partial(objective_refinement_grad, **kw), X_flat, jac=True, method="L-BFGS-B", bounds=bounds,
                   options=_lbfgs_options(G))
    _log_failure(res, loss_fn(X_flat), 'refining')

    if check_gradient:
        error = check_grad(loss_fn, grad_fn, res.x)
        if error > .05:
            logger.debug("Gradient theta refining check error for gene %s: %s", g, error)
            res = minimize(loss_fn, X_flat, bounds=bounds, method="L-BFGS-B", options=_lbfgs_options(G))

    correc = res.x
    inter *= correc[:G]
    basal = correc[G:G + n_samples].copy()
    inter += diag

    return inter, basal, None if thresh is None else thresh[0] + correc[G + n_samples:]


def ridge_coefficients(y_prot, ref_network, network_ridge):
    """
    Gaussian prior on the effect of each edge on the logit, w_e theta_e sd_e ~ N(0, s^2) (sd_e: standard deviation
    of regulator e), s = 2 logit(1 - 1/network_ridge): the logit span from a target at 1/network_ridge of OFF to
    1/network_ridge of ON. Bounds the edges of nearly separable targets; zero gradient at 0, so the QUT is unchanged.
    Returns the coefficients (w sd / s)^2 of 0.5 sum c theta^2.
    """
    eps = 1.0 / max(network_ridge, 2.0 + EPS)
    span = 2.0 * np.log((1.0 - eps) / eps)
    sd = np.std(_input_features(y_prot), axis=0)  # Herbach target: of its Hill ratios (effect of theta at 0)
    return (prior_weight(ref_network) * sd / span) ** 2


def qut_garrote(y_samples, y_prot, y_prot_mod, y_target, inter, basal, G, g, n_samples, weight_flow_matching, ref_network,
                cond=None, level=0.5, n_perm=20, sample_weights=None):
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
              weights_samples=sample_weights if sample_weights is not None else
              [np.sum(y_samples == s) ** alpha for s in np.unique(y_samples)],
              ys=y_samples, yp=y_prot, diag=diag, G=G, g=g, n_samples=n_samples, l_pen=np.zeros(G),
              weight_flow_matching=weight_flow_matching, cond=cond)
    # Null basals: invariant under within-sample permutations of the target
    null = dict(kw, ypm=y_prot_mod, yt=y_target)
    bb = _basal_bound(G)
    X0 = minimize(partial(objective_refinement_grad, **null), np.r_[np.zeros(G), np.asarray(basal, dtype=float)],
                  jac=True, method="L-BFGS-B",
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
    X0 = minimize(partial(objective_grad, **null), _null_start(G, n_samples, y_samples, y_target),
                  jac=True,
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


def qut_conditions(y_samples, y_prot, y_prot_mod, y_target, ref_network, G, g, n_samples, weight_flow_matching, cond, pen,
                   level=0.5, n_perm=20, constrain_basal_uniform=0.0, basal_free_mask=None, act=None):
    """
    Penalty levels of the fit with network conditions, generalising qut_penalty (same null model and permutations).
    lam: QUT of the gradient summed over the conditions (that of a common network); lam_c: QUT of the gradient of the
    rows of condition c, in the scale of the joint objective. With rho = min(pen, 1):
    l_shared = kappa rho lam, l_cond_c = kappa (1 - rho) lam_c, l_fuse_c = kappa pen lam_c (effective units for delta).
    pen = 0: each condition at its own QUT; pen = 1: shared network and deviations each at their QUT; pen -> inf: common
    network at lam. An edge can enter through the shared network or one condition only: kappa >= 1 scales the levels so
    that the network of a condition stays at 0 with probability ~ level under the null, as with one network.
    Returns ConditionLevels; ridge_w: weights of the ridge per condition, omega_c (share of the samples of the fit) at
    pen >= 1, the scale of its data term in the joint objective at pen = 0.
    """
    C = int(np.max(cond)) + 1
    w = prior_weight(ref_network)
    ys = np.asarray(y_samples).astype(int)
    us = np.unique(ys)
    weights = [np.sum(ys == s) ** alpha for s in us]
    omega = np.bincount(cond[us], minlength=C) / len(us)
    rows_c = [np.isin(ys, us[cond[us] == c]) for c in range(C)]
    w_c = [[np.sum(ys == s) ** alpha for s in us[cond[us] == c]] for c in range(C)]
    # Scale of the data term of each condition in the joint objective, relative to fitting it alone
    scale = np.array([np.mean(weights) / np.mean(w_c[c]) if len(w_c[c]) else 0.0 for c in range(C)])
    rho = min(pen, 1.0)
    ridge_w = (1 - rho) * scale + rho * omega
    empty = ConditionLevels(0.0, 0.0, np.zeros(C), np.zeros(C), omega, ridge_w)
    on = np.tile(w > 0, (C, 1)) if act is None else (w > 0) & (np.asarray(act, dtype=float) > 0)
    if not on.any():
        return empty
    kw = dict(G=G, g=g, n_samples=n_samples, theta_ref=np.zeros(G + n_samples), ref_network=w, l_pen=0.0,
              weight_flow_matching=weight_flow_matching, final=0)
    # Null basals: invariant under within-sample permutations of the target
    null = dict(kw, ys=y_samples, yp=y_prot, ypm=y_prot_mod, yt=y_target, weights_samples=weights,
                constrain_basal_uniform=constrain_basal_uniform, basal_free_mask=basal_free_mask)
    bounds = [(0.0, 0.0)] * G + [(-_basal_bound(G), _basal_bound(G))] * n_samples
    X0 = minimize(partial(objective_grad, **null), _null_start(G, n_samples, y_samples, y_target),
                  jac=True,
                  method="L-BFGS-B", bounds=bounds, options=_lbfgs_options(G)).x
    groups = [np.flatnonzero(ys == s) for s in us]
    rng = np.random.default_rng(1000 * g + len(y_target))
    grads = np.zeros((n_perm, C, G))  # d loss / d theta_c at the null, theta units
    for k in range(n_perm):
        p = np.arange(len(y_target))
        for idx in groups:
            p[idx] = rng.permutation(idx)
        ypm = y_prot_mod
        if ypm is not None:
            ypm = PathNodes(ypm.nodes, ypm.weights, None if ypm.kon is None else ypm.kon[p],
                            None if ypm.targets is None else ypm.targets[p])
        yt = y_target[p]
        for c in np.flatnonzero(omega > 0):
            r = rows_c[c]
            grads[k, c] = scale[c] * grad_theta(X0, ys=y_samples[r], yp=y_prot[r], ypm=None if ypm is None else ypm[r],
                                                yt=yt[r], weights_samples=w_c[c], **kw)[:G] * on[c]
    pool = np.max(np.abs(grads.sum(axis=1)), axis=1)  # shared network
    wmax = np.where(w > 0, w, 1.0)
    per_c = np.array([[np.max(np.abs(grads[k, c])[on[c]]) if on[c].any() else 0.0 for c in range(C)]
                      for k in range(n_perm)])
    per_c_eff = np.array([[np.max((np.abs(grads[k, c]) / wmax)[on[c]]) if on[c].any() else 0.0 for c in range(C)]
                          for k in range(n_perm)])
    lam = float(np.quantile(pool, level))
    lam_c, lam_e = np.quantile(per_c, level, axis=0), np.quantile(per_c_eff, level, axis=0)
    l_shared, l_cond, l_fuse = rho * lam, (1 - rho) * lam_c, pen * lam_e
    # Scale of the levels that keeps each condition at 0 through the shared network or its own deviations
    door_shared = max(l_shared + float(np.sum(l_cond)), EPS)
    active = np.flatnonzero((omega > 0) & on.any(axis=1))
    kappa = np.empty(len(active))
    for i, c in enumerate(active):
        door_c = np.maximum((l_cond[c] + l_fuse[c] * w)[on[c]], EPS)
        ratio = np.array([np.max(np.abs(grads[k, c])[on[c]] / door_c) for k in range(n_perm)])
        kappa[i] = np.quantile(np.maximum(pool / door_shared, ratio), level)
    kappa = float(np.mean(kappa)) if len(kappa) else 1.0
    return ConditionLevels(lam, kappa * l_shared, kappa * l_cond, kappa * l_fuse, omega, ridge_w)


def main_loop_inference(g, y_samples, y_prot, y_prot_mod, y_kon, theta_init, theta_ref,
                        ks, G, n_samples, ref_network,
                        weight_flow_matching=.5, final=0,
                        constrain_basal_uniform=0.0, basal_free_mask=None,
                        hard_forcing_ref=False, ref_constraint_pct=0.1, seuil_min_network_intensity=1e-2,
                        cond=None, condition_pen=1.0, act=None, act_tgt=None,
                        qut_level=0.5, qut_n_perm=20, network_ridge=20.0, network_relaxed=False, dist=None,
                        thresh_prior=None, mixed=None, network_cost='ce', network_balance=False, balance_transit=None,
                        ordinal_mid=None, ordinal_u=None, level_data=None, network_likelihood=False):
    """
    Joint inference (inter + per-sample basal) for a single gene g.
    network_cost: data term of the fit, 'ce' (cross-entropy), 'euclidean' (2 (sigma - target)^2) or 'ordinal'
    (ordinal_cost, central mode of each sample in ordinal_mid (n_samples,) or (1,)), QUT included.
    network_likelihood: on the rows with a real cell, the negative log-likelihood of its counts under the prior
    exp(-kappa C) of that cost on the levels (lik_terms: the exact M-step; the cost itself on the rows without real
    cell and on the nodes of the flow matching), the QUT permuting the counts with the targets.
    level_data (A (N, L), tau_l (N, L), kappa (N,), obs (N,)) of the rows, with network_likelihood: A the
    log-likelihoods of the counts at the levels of the rows with a real cell (obs), 0 at the levels of the gene in
    the sample of the other rows (-inf beyond), whose level is that of their target (onehot_levels), as on the nodes.
    ordinal_u: (u1, u2) (N,) warped targets of the ordinal cost of the rows (soft targets: their expectations under
    the posteriors of the levels, E_q[T_k(tau_l)], the M-step of the EM), None: the warps of the targets.
    network_balance: exponent gamma in [0, 1] (False: none, True: 1): rows weighted by 1 / frequency^gamma of their main
    mode in their sample (balance_weights; transition levels per interval of each sample in balance_transit
    (n_samples,) or None), QUT included.
    mixed: (tau2, tau2_basal) of the mixed model (core_inference_mixed), with cond = patient of each sample; replaces
    condition_pen, constrain_basal_uniform and alpha, and adds the statistics of its variances as a sixth element.
    dist: DistTerm of the end-state term (weight_network_distribution) or None; the sparsity levels (QUT) are those of
    the cross-entropy alone (at the null model the sigmas are tied and the distribution term is not defined).
    Sparsity levels: quantile universal thresholds at qut_level, of the first fit (qut_penalty) and of the
    refinement (qut_garrote, theta of the first fit fixed);
    network_ridge: Gaussian prior on the effects of the edges in the first fit (ridge_coefficients).
    network_relaxed: refinement without L1 penalty (relaxed lasso: the first fit sets the support and signs, the
    refinement re-estimates the effects without the shrinkage of the L1, within the bounds), keeping the ridge of the
    first fit, which bounds the edges of nearly separable targets.
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
    condition_pen : penalty on the deviations between conditions, relative to their QUT (qut_conditions)
    Returns (basal, inter, basal_tmp, inter_tmp): basal (n_samples,), inter (G,) or (C, G);
    *_tmp are the values before the refinement step.
    thresh_prior: (log s_0 (G,), tau) of an exact Herbach target, its log-thresholds learned in the refinement
    (refine_inference); returned as a fifth element (G,), None otherwise.
    """
    set_network_cost(network_cost, network_likelihood and level_data is not None)
    # Sigmoid between the lowest and the highest mode of each cell's sample (intermediate modes lie on the slope)
    kb = sigmoid_amplitudes(ks)
    if kb.ndim == 2:
        kb = kb[np.asarray(y_samples).astype(int)]
    y_target = sigmoid_target(y_kon, kb[..., 0], kb[..., 1])
    if (network_cost == 'ordinal' or network_likelihood) and mixed is not None:
        raise ValueError("the ordinal network cost and fit_likelihood are not supported with "
                         "mixed_effect_model_conditions")
    lik = network_likelihood and level_data is not None
    if network_cost == 'ordinal':
        om = np.full(1, -1.0) if ordinal_mid is None else np.asarray(ordinal_mid, dtype=float)
        mid = om[np.minimum(np.asarray(y_samples).astype(int), len(om) - 1)]
        u1, u2 = ordinal_targets(np.ascontiguousarray(y_target, dtype=float), np.ascontiguousarray(mid)) \
            if ordinal_u is None else ordinal_u  # soft targets: expected warped targets of the levels
        y_target = with_targets(y_target, mid=mid, u1=u1, u2=u2)
    if lik:
        ll, tl, kap, obs = (np.ascontiguousarray(a, dtype=float) for a in level_data)
        C = ordinal_consts(np.ascontiguousarray(mid)) if network_cost == 'ordinal' else np.zeros((0, 9))
        lu1, lu2, lent = lik_levels(tl, np.isfinite(ll).astype(float), cost_code(network_cost), C)  # levels, once
        y_target = with_targets(y_target, ll=ll, tl=tl, kap=kap, obs=obs, lu1=lu1, lu2=lu2, lent=lent)
    if network_balance:
        if mixed is not None:
            raise ValueError("fit_network_balance is not supported with mixed_effect_model_conditions")
        y_target = balance_weights(y_target, y_samples, ks, balance_transit, float(network_balance))
    if y_prot_mod is not None:  # targets in force at the flow-matching nodes
        lo, hi = np.asarray(kb[..., 0]), np.asarray(kb[..., 1])  # scalars, or one per row (per-sample mixtures)
        lo, hi = (lo[:, None], hi[:, None]) if lo.ndim else (lo, hi)
        y_prot_mod = PathNodes(y_prot_mod.nodes, y_prot_mod.weights, y_prot_mod.kon,
                               sigmoid_target(y_prot_mod.kon, lo, hi))
        if network_cost == 'ordinal':  # warped targets of the nodes, computed once
            tn = np.ascontiguousarray(y_prot_mod.targets, dtype=float)
            w1, w2 = ordinal_targets(tn.ravel(), np.repeat(mid, tn.shape[1]))
            y_prot_mod.targets = with_targets(tn, u1=w1.reshape(tn.shape), u2=w2.reshape(tn.shape))
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
        return basal, inter, basal.copy(), inter.copy(), None

    # Interactions are optimised as theta = e / w (w = prior weight), so an edge of weight q
    # costs 1/q more; init and ref are given as effective values e and converted here.
    w = prior_weight(ref_network)
    w_rows = np.tile(w, C)
    theta_init, theta_ref = np.array(theta_init, dtype=float), np.array(theta_ref, dtype=float)
    theta_init[:n_rows] = np.where(w_rows > 0, theta_init[:n_rows] / np.maximum(w_rows, EPS), 0.0)
    theta_ref[:n_rows] = np.where(w_rows > 0, theta_ref[:n_rows] / np.maximum(w_rows, EPS), 0.0)
    if _MODE and not np.any(theta_init[:n_rows]):
        # Non-convex cost from no network: basals at the null optimum (a start at 0 can stall on a saturated plateau)
        theta_init[n_rows:n_rows + n_samples] = _null_start(0, n_samples, y_samples, y_target)

    common = dict(weight_flow_matching=weight_flow_matching * (1 - final), final=final,
                  constrain_basal_uniform=constrain_basal_uniform, basal_free_mask=basal_free_mask,
                  hard_forcing_ref=hard_forcing_ref, ref_constraint_pct=ref_constraint_pct,
                  seuil_min_network_intensity=seuil_min_network_intensity,
                  ridge=ridge_coefficients(_end(y_prot), ref_network, network_ridge), dist=dist)
    sample_weights, mixed_stats = None, None
    if mixed is not None:
        sample_weights, m = _mixed_weights(y_samples)
        theta, mixed_stats = core_inference_mixed(
            y_samples, y_prot, y_prot_mod, y_target, theta_init, theta_ref, ref_network, G, g, n_samples, cond,
            mixed[0], mixed[1], sample_weights, m, weight_flow_matching=common['weight_flow_matching'],
            basal_free_mask=basal_free_mask, ref_constraint_pct=ref_constraint_pct,
            seuil_min_network_intensity=seuil_min_network_intensity, ridge=common['ridge'], dist=dist,
            level=qut_level, n_perm=qut_n_perm)
    elif cond is None:
        l_pen = qut_penalty(y_samples, y_prot, y_prot_mod, y_target, ref_network, G, g, n_samples,
                            weight_flow_matching * (1 - final), qut_level, qut_n_perm, constrain_basal_uniform,
                            basal_free_mask)
        theta = core_inference(y_samples, y_prot, y_prot_mod, y_target, theta_init, theta_ref, ref_network,
                               G, g, n_samples, l_pen, **common)
    else:
        levels = qut_conditions(y_samples, y_prot, y_prot_mod, y_target, ref_network, G, g, n_samples,
                                weight_flow_matching * (1 - final), cond, condition_pen, qut_level, qut_n_perm,
                                constrain_basal_uniform, basal_free_mask, act)
        theta = core_inference_conditions(y_samples, y_prot, y_prot_mod, y_target, theta_init, theta_ref, ref_network,
                                          G, g, n_samples, cond, levels, act=act, **common)
    inter_tmp = theta[:n_rows].reshape((G,) if cond is None else (C, G))
    basal_tmp = theta[n_rows:n_rows + n_samples].copy()

    # Refinement step (reference of the shared network for the conditions)
    ref_refine = theta_ref if cond is None else np.concatenate(
        [theta_ref[:n_rows].reshape(C, G).mean(axis=0), theta_ref[n_rows:]])

    inter, basal, log_thresh = refine_inference(
        y_samples, y_prot, y_prot_mod, y_target, inter_tmp.copy(), basal_tmp.copy(), ref_refine,
        G, g, n_samples,
        0.0 if network_relaxed else
        qut_garrote(y_samples, y_prot, y_prot_mod, y_target, inter_tmp, basal_tmp, G, g, n_samples,
                    weight_flow_matching * (1 - final), ref_network, cond, qut_level, qut_n_perm, sample_weights),
        weight_flow_matching=weight_flow_matching * (1 - final),  # adaptive lasso, weights 1 / |theta_tmp|
        final=final, hard_forcing_ref=hard_forcing_ref, ref_constraint_pct=ref_constraint_pct,
        ref_network=ref_network, seuil_min_network_intensity=seuil_min_network_intensity, cond=cond, dist=dist,
        ridge=common['ridge'] if network_relaxed else None, thresh_prior=thresh_prior, sample_weights=sample_weights,
    )
    if act is not None and cond is not None:
        inter = inter * act  # entries of regulators / targets outside each condition stay 0
    if flat.any():
        basal = _fit_flat_basals(basal, inter, flat, *full, g, cond, _basal_bound(G))
        basal_tmp[flat] = basal[flat]

    if mixed is not None:
        return basal, inter, basal_tmp, inter_tmp, log_thresh, mixed_stats
    return basal, inter, basal_tmp, inter_tmp, log_thresh


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
            x = b + _logit(Y[mask], th)
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
                act_tgt=None, dist=None, thresh_prior=None, level_data=None, ordinal_u=None):
    """
    Network inference for one target gene (worker task of inference_network).
    yp / ypm / rn are already restricted to the active regulators active_src
    (full when every regulator is active). With cond (condition of each sample), ti_g / tr_g hold
    one block of G interactions per condition and the interactions returned are (C, G).
    """
    if gene_kw.get('ordinal_mid') is not None:  # central mode of this target per sample (ordinal cost)
        gene_kw = dict(gene_kw, ordinal_mid=np.asarray(gene_kw['ordinal_mid'], dtype=float)[:, g_tgt])
    if gene_kw.get('balance_transit') is not None:  # transition levels of this target per sample (balance)
        gene_kw = dict(gene_kw, balance_transit=np.asarray(gene_kw['balance_transit'])[:, g_tgt])
    if gene_kw.get('mixed') is not None:  # tau2 of this target
        tau2, tau2_b = gene_kw['mixed']
        gene_kw = dict(gene_kw, mixed=(float(np.broadcast_to(tau2, (G,))[g_tgt]), tau2_b))
    K_g = len(active_src)
    C = 1 if cond is None else int(np.max(cond)) + 1
    mask_kw = {} if act is None else dict(act=act, act_tgt=act_tgt)
    if K_g == G:
        # Dense path: no sub-selection
        return main_loop_inference(
            g_tgt, y_samples, yp, ypm, y_kon_g, ti_g, tr_g, ks_g, G, n_samples, rn,
            basal_free_mask=free_mask_g, cond=cond, dist=dist, thresh_prior=thresh_prior, level_data=level_data, ordinal_u=ordinal_u,
            **mask_kw, **gene_kw,
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

    out = main_loop_inference(
        g_sub, y_samples, yp, ypm, y_kon_g, theta_init_sub, theta_ref_sub,
        ks_g, K_g, n_samples, rn,
        basal_free_mask=free_mask_g, cond=cond, dist=dist, thresh_prior=thresh_prior, level_data=level_data, ordinal_u=ordinal_u,
        **mask_kw, **gene_kw,
    )
    basal_r, inter_sub_r, basal_tmp_r, inter_tmp_sub_r, thr_sub = out[:5]

    # Expand interaction results back to full G-dimensional space
    shape = (G,) if cond is None else (C, G)
    inter_full     = np.zeros(shape)
    inter_tmp_full = np.zeros(shape)
    inter_full[..., active_src]     = inter_sub_r
    inter_tmp_full[..., active_src] = inter_tmp_sub_r
    thr_full = None
    if thr_sub is not None:  # log-thresholds of the active regulators, NaN elsewhere
        thr_full = np.full(G, np.nan)
        thr_full[active_src] = thr_sub
    extra = out[5:]
    if extra:  # mixed model: statistics of the variances, posterior z of the deviations
        z_full = np.zeros(shape)
        z_full[..., active_src] = extra[0][4]
        extra = (tuple(extra[0][:4]) + (z_full,),)
    return (basal_r, inter_full, basal_tmp_r, inter_tmp_full, thr_full) + extra


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
                      sigma_spec_grn=None, qut_level=0.5, qut_n_perm=20, network_relaxed=False, y_prot_mem=None,
                      dist_targets=None, herbach_threshold_sd=None, mixed=None, network_cost='ce',
                      network_balance=False, balance_transit=None, ordinal_mid=None, ordinal_u=None, level_data=None,
                      network_likelihood=False) -> list:
    """
    Joint network inference: interactions shared across samples, basal per sample.
    One independent fit per row subset in subsets (index arrays, None = all rows);
    all fits x target genes are run in a single parallel pool.

    y_prot_mod : PathProt (flow-matching nodes), or None without flow-matching term (final fit).
    ks : mixture amplitudes (G, n_modes), or per sample (n_samples, G, n_modes); kon is a sigmoid
        between the lowest and the highest mode.

    Returns a list with, per subset, (basal, inter, basal_tmp, inter_tmp, edges):
    basal (n_samples, G), inter (G, G), edges (G, G) the thresholds [target, regulator] of the edges of that fit
    (sigma_spec_grn[:, N_SPEC:] with the thresholds learned on the subset, herbach_threshold_sd), None without spec.

    constrain_basal_uniform : float >= 0
        When > 0, adds a penalty that pushes per-sample basals toward their common
        mean for each gene. Samples whose basal is pinned by a non-zero basal_ref
        (KO/OV priors) are excluded from the penalty for that gene.
    sample_conditions : (n_samples,) int array or None
        Network condition of each sample (index). With >= 2 conditions, each condition has its
        own network theta_c = theta_shared + delta_c, with L1 penalties on theta_shared and delta_c at their
        quantile universal thresholds (qut_conditions, condition_pen scales that of delta_c); condition_pen = 0:
        independent networks, each condition fitted alone on its rows, samples and genes (single-network fit, as if inferred alone; NaN for a
        condition without rows in a subset); inter is then returned per sample, (n_samples, G, G).
        inter_init / inter_ref may be (G, G) (common) or per sample.
    gene_mask : (n_samples, G) bool or None: genes selected in each sample's condition (select_genes_per_condition);
        a gene outside a condition is neither a target nor a regulator there (its interactions are 0) and the fused
        penalty only acts on the edges shared by conditions. Without conditions, the mask of the (single) gene set.
    sigma_spec_grn : (G, N_SPEC + G) or None: families of the GRN sigma (sigma.py, model_learning_grn); a target with a
        non-identity phi (Hill or Herbach type) is fitted on the features phi of its regulators (same fit, theta then
        acting on phi; logistic link only). None: every target logistic of a linear input.
    y_prot_mem : PathProt or None (targets_accounting_late_mrna): states of every gene at nodes along the interval
        ending at each row (nodes (N, Q, G)) and weights (N, Q + 1, G) of each target's mRNA memory (column 0 = the row);
        the end-state term then compares sigma averaged on the memory of the target to its target (MemStates).
    dist_targets : (post (N, G, L), levels (N, G, L), grp (N,), weight) or None (weight_network_distribution): posteriors
        and sigmoid targets of the levels of the real cell of each row, group (sample, time) of each row (-1 without
        real cell); targets of more than two levels get the distribution term (dist_term) in their end-state term.
    herbach_threshold_sd : prior sd of the log-thresholds of the edges of the exact Herbach targets around those of
        their regulators (spec[:, SPEC_THRESH]), learned in the refinement; their geometric mean over the subsets is
        written into sigma_spec_grn[:, N_SPEC:] (in place). None / 0: thresholds of sigma_spec_grn fixed.
    mixed : (tau2, tau2_basal) or None: mixed model with one patient per network condition (core_inference_mixed), tau2
        a value or one per target (G,); condition_pen and constrain_basal_uniform are then ignored, and each fit has a
        sixth element: statistics of the variances (G, 4) (s_theta, n_theta, s_basal, n_basal per target) and posterior
        z (C, G, G).
    network_cost, network_likelihood, network_balance, balance_transit, ordinal_mid: main_loop_inference.
    ordinal_u : (u1, u2) (N, G) or None: warped targets of the ordinal cost of the rows (main_loop_inference).
    level_data : with network_likelihood, (A, tau_l (N_cells, G, L), kappa (N_cells, G), real (N,), tau_l (S, G, L),
        A0 (S, G, L), kappa (S, G)): log-likelihoods of the counts A, positions of the levels and kappa of the real
        cells, real cell of each row (-1: none), positions, 0 / -inf on the levels and kappa of each sample (rows
        without real cell).
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
    if mixed is not None and cond is None:
        mixed = None  # a single patient: no random effect
    if mixed is not None and (gene_mask is not None or hard_forcing_ref):
        raise ValueError("the mixed model does not support select_genes_per_condition nor hard_forcing_ref yet")
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
                   qut_level=qut_level, qut_n_perm=qut_n_perm, network_ridge=network_ridge,
                   network_relaxed=network_relaxed, mixed=mixed, network_cost=network_cost,
                   network_balance=network_balance, balance_transit=balance_transit, ordinal_mid=ordinal_mid,
                   network_likelihood=network_likelihood)

    # Genes of each condition (rows of the first sample of each condition); a single set without conditions
    gm = None if gene_mask is None else np.asarray(gene_mask, dtype=bool)
    act_c = None if gm is None else gm[first_of]   # (C, G)
    if act_c is not None and cond is None:
        # One gene set: regulators outside it are dropped from every target
        active_cols = [c[act_c[0][c]] for c in active_cols]
    # Independent networks (condition_pen = 0): one single-network fit per condition, on its samples and genes, as if
    # the condition were inferred alone (its own sparsity level, ridge and refinement)
    apart = cond is not None and condition_pen == 0 and mixed is None
    rows_of = None if not apart else [np.isin(y_samples, np.flatnonzero(cond == c)) for c in range(C)]

    def level_rows(rows, g_tgt):
        # Level data of the rows for target g_tgt (level_data: log-likelihoods A, positions tau_l (N_cells, G, L) and
        # kappa (N_cells, G) of the real cells, real cell of each row (N,), -1 for none; positions tau_l (S, G, L),
        # 0 / -inf on the levels of each sample and kappa (S, G) of the samples, for the rows without real cell)
        if level_data is None:
            return None
        A, tl, kap, real, tl_s, ll0_s, kap_s = level_data
        rr = np.asarray(real)[rows].astype(int)
        obs = rr >= 0
        rc = np.where(obs, rr, 0)
        ys = np.minimum(np.asarray(y_samples)[rows].astype(int), len(tl_s) - 1)
        return (np.where(obs[:, None], A[rc, g_tgt], ll0_s[ys, g_tgt]),
                np.where(obs[:, None], tl[rc, g_tgt], tl_s[ys, g_tgt]),
                np.where(obs, kap[rc, g_tgt], kap_s[ys, g_tgt]), obs.astype(float)) if A.shape[2] else None

    def ordinal_rows(rows, g_tgt):
        # Warped targets (u1, u2) of the rows for target g_tgt (ordinal_u: (N, G) each), None without
        return None if ordinal_u is None else (np.ascontiguousarray(ordinal_u[0][rows, g_tgt], dtype=float),
                                               np.ascontiguousarray(ordinal_u[1][rows, g_tgt], dtype=float))

    def gene_args(rows, g_tgt, c=None):
        # Only the rows and regulators target g_tgt needs are sent to the worker (built lazily); c: condition fitted
        # apart (its rows and regulators, its block of theta, no condition structure)
        if c is not None:
            rows = np.flatnonzero(rows_of[c]) if rows is None else rows[rows_of[c][rows]]
        rows = slice(None) if rows is None else rows
        active_src = active_cols[g_tgt]
        if c is not None and act_c is not None:
            active_src = active_src[act_c[c][active_src]]
        dense = len(active_src) == G
        yp = y_prot[rows] if dense else y_prot[rows][:, active_src]
        ypm = prev_prot(g_tgt, active_src)
        ypm = None if ypm is None else ypm[rows]
        mem = None if y_prot_mem is None else (y_prot_mem.nodes[rows][:, :, active_src], y_prot_mem.weights[rows][:, :, g_tgt])
        thresh_prior = None
        if sigma_spec_grn is not None and sigma_spec_grn[g_tgt, SPEC_LINK] != LINK_LOGISTIC:
            raise ValueError("the network fit supports the logistic link only")
        if sigma_spec_grn is not None and sigma_spec_grn[g_tgt, SPEC_PHI] != PHI_IDENTITY:
            # Non-identity phi: the fit acts on the features phi of the regulators of the target
            fam = int(sigma_spec_grn[g_tgt, SPEC_PHI])
            kappa = np.ascontiguousarray(sigma_spec_grn[active_src, SPEC_KAPPA], dtype=float)
            pbar = np.ascontiguousarray(sigma_spec_grn[active_src, SPEC_PBAR], dtype=float)
            thresh = np.ascontiguousarray(sigma_spec_grn[active_src, SPEC_THRESH], dtype=float)
            m_tgt = float(sigma_spec_grn[g_tgt, SPEC_M])
            if fam == PHI_HERBACH:  # exact Herbach contributions: inputs m log(P / s_ij), logit nonlinear in theta
                thr_e = np.ascontiguousarray(sigma_spec_grn[g_tgt, N_SPEC + active_src], dtype=float)
                phi = lambda Y: herbach_inputs(Y, thr_e, pbar, m_tgt)
                if herbach_threshold_sd:
                    thresh_prior = (np.log(thresh), float(herbach_threshold_sd))
            else:
                phi = lambda Y: features(np.ascontiguousarray(Y, dtype=float), fam, kappa, pbar, thresh, m_tgt)
            yp = phi(yp)
            if mem is not None:
                n, Q, K = mem[0].shape
                mem = (phi(mem[0].reshape(n * Q, K)).reshape(n, Q, K), mem[1])
            if ypm is not None:
                n, Q, K = ypm.nodes.shape
                ypm = PathNodes(phi(ypm.nodes.reshape(n * Q, K)).reshape(n, Q, K), ypm.weights, ypm.kon, ypm.targets)
        if mem is not None and mem[0].shape[1] == 1 and np.all((mem[1] == 0) | (mem[1] == 1)):
            # Single node of weight 1 (late_mrna_nodes = 1): the end-state term at the lagged states, without cost
            lagged = np.where(mem[1][:, :1] > 0.5, yp, mem[0][:, 0])
            yp = _as_herbach(lagged, yp.lbar, yp.pbar, yp.m) if isinstance(yp, HerbachArray) else lagged
        elif mem is not None:  # end-state term on sigma averaged on the target's mRNA memory
            yp = MemStates(yp, mem[0], mem[1])
        theta_rows = slice(None) if c is None else np.r_[c * G:(c + 1) * G, C * G:C * G + n_samples]
        cond_g = None if c is not None else cond
        return (g_tgt, G, active_src, y_samples[rows], yp, ypm,
                y_kon[rows, g_tgt], theta_init_mat[theta_rows, g_tgt], theta_ref_mat[theta_rows, g_tgt],
                ks[:, g_tgt] if ks.ndim == 3 else ks[g_tgt], n_samples,
                ref_network[:, g_tgt] if dense else ref_network[active_src, g_tgt],
                free_mask_2d[:, g_tgt], gene_kw, cond_g,
                None if act_c is None or cond_g is None else act_c[:, active_src].astype(float),
                None if act_c is None or cond_g is None else act_c[:, g_tgt],
                None if dist_targets is None else dist_term(dist_targets[2][rows], dist_targets[0][rows, g_tgt],
                                                            dist_targets[1][rows, g_tgt], dist_targets[3]),
                thresh_prior, level_rows(rows, g_tgt), ordinal_rows(rows, g_tgt))

    # Targets outside every condition's genes (single gene set: outside it) are not fitted
    genes = [g for g in range(n_stimuli, G) if act_c is None or act_c[:, g].any()]
    # Conditions fitted apart: those of each subset where the target is selected (a condition may be absent from a
    # subset when its sub-samples are drawn alone: NaN there, left out of the mean over the subsets)
    present = [[apart and (rows_of[c] if sub is None else rows_of[c][sub]).any() for c in range(C)] for sub in subsets]
    tasks = [(k, g, c) for k in range(len(subsets)) for g in genes
             for c in ([None] if not apart else [c for c in range(C) if present[k][c] and (act_c is None or act_c[c, g])])]
    if Parallel is not None:
        results = Parallel(n_jobs=-1)(
            delayed(_infer_gene)(*gene_args(subsets[k], g, c)) for k, g, c in tasks
        )
    else:
        results = [_infer_gene(*gene_args(subsets[k], g, c)) for k, g, c in tasks]

    # Learned log-thresholds of the edges of the exact Herbach targets: per subset (geometric mean over the conditions
    # fitted apart), and their geometric mean over the subsets written into sigma_spec_grn
    thr, thr_k = {}, {}
    for (k_, g, c), res in zip(tasks, results):
        if res[4] is not None:
            thr.setdefault(g, []).append(res[4])
            thr_k.setdefault((k_, g), []).append(res[4])
    edge_fit = None if sigma_spec_grn is None else [sigma_spec_grn[:, N_SPEC:].copy() for _ in subsets]
    for (k_, g), lst in thr_k.items():
        lt = np.array(lst)
        ok = np.isfinite(lt).any(axis=0)
        edge_fit[k_][g, np.flatnonzero(ok)] = np.exp(np.nanmean(lt[:, ok], axis=0))
    for g, lst in thr.items():
        lt = np.array(lst)
        ok = np.isfinite(lt).any(axis=0)
        sigma_spec_grn[g, N_SPEC + np.flatnonzero(ok)] = np.exp(np.nanmean(lt[:, ok], axis=0))

    fits = []
    shape = (G, G) if cond is None else (C, G, G)
    for k in range(len(subsets)):
        basal, basal_tmp = np.zeros((n_samples, G)), np.zeros((n_samples, G))
        inter, inter_tmp = np.zeros(shape), np.zeros(shape)
        if act_c is not None and basal_init is not None:  # targets outside every gene set keep their initial basal
            basal[:, n_stimuli:], basal_tmp[:, n_stimuli:] = (basal_rows(basal_init)[:, n_stimuli:],) * 2
        for (k_, g, c), res in zip(tasks, results):
            if k_ != k:
                continue
            if c is None:
                basal[:, g], inter[..., g], basal_tmp[:, g], inter_tmp[..., g] = res[:4]
                continue
            # Condition fitted apart: its network and the basals of its samples
            in_c = cond == c
            basal[in_c, g], inter[c, :, g], basal_tmp[in_c, g], inter_tmp[c, :, g] = \
                res[0][in_c], res[1], res[2][in_c], res[3]
        for c in range(C):
            if apart and not present[k][c]:
                for arr in (basal, basal_tmp):
                    arr[cond == c] = np.nan
                inter[c], inter_tmp[c] = np.nan, np.nan
        if cond is not None:
            # Per sample: each sample takes the network of its condition
            inter, inter_tmp = inter[cond], inter_tmp[cond]
        stats_k = None
        if mixed is not None:
            s_k, z_k = np.zeros((G, 4)), np.zeros((C, G, G))
            for (k_, g, _), res in zip(tasks, results):
                if k_ == k and len(res) > 5:
                    s_k[g] = res[5][:4]
                    z_k[:, :, g] = res[5][4]
            stats_k = (s_k, z_k)
        fits.append((basal, inter, basal_tmp, inter_tmp, None if edge_fit is None else edge_fit[k], stats_k))
    return fits


def inference_network(y_samples, y_kon, y_prot, y_prot_mod, ks, **kwargs):
    """Single joint network fit on all rows (see inference_network_multi). Returns (basal, inter, basal_tmp, inter_tmp)."""
    return inference_network_multi([None], y_samples, y_kon, y_prot, y_prot_mod, ks, **kwargs)[0][:4]
