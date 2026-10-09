"""
Small population models: growth and transitions between K types (cell_type_proliferation, or the types of the
transition matrix) fitted on the proportions of the types over time of one sample, and on its population sizes if
given, to make prescribed rates consistent with them (fit_population_anchors.py).

The abundances n of the types follow dn/dt = M(t) n with M = diag(r(t)) + Q^T - diag(Q 1): net rates r_a(t) = r0_a +
sum_k effects[a, k] u_k(t) (r0: rates without stimulus, effects: RATEk of perturbation_inference, u_k: inference
schedule of the sample, constant over each interval) and transition rates Q[a, b] (a -> b, h^-1). Only the proportions
n / sum(n) are observed: they identify r_b - r_a and the transitions only jointly, so each is held by a prior (the
prescribed rates; transitions small unless prescribed). Population sizes N(t) (relative), when given, constrain the
absolute growth: log N(t) - log N(t_ref) = log sum n(t) - log sum n(t_ref).
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

LN2 = np.log(2.0)

DEFAULTS = dict(
    sd_r=0.002,        # h^-1: how far the corrected rates may move from the prescribed ones (about +-0.05 / day)
    sd_q=0.003,        # h^-1: scale of the free transitions (prior 0), also the floor of the prior of prescribed ones
    q_rel_sd=0.5,      # prescribed transition rates may change by this fraction of their value
    sd_logn=0.1,       # standard error of a log population size (counting, plating)
    sd_r_free=0.02,    # h^-1: prior sd of the rates fitted from the population sizes alone (no prescribed rate)
    sd_r_sizes=0.01,   # h^-1: prior sd of prescribed rates when population sizes are given (measured: they dominate)
    se_floor=0.02,     # floor of the standard error of a proportion (sampling is not the only noise between libraries)
)


def generator(r, Q):
    """M of dn/dt = M n for net rates r (K,) and transition rates Q[a, b] (a -> b; diagonal ignored)."""
    Q = np.array(Q, dtype=float)
    np.fill_diagonal(Q, 0.0)
    return np.diag(r) + Q.T - np.diag(Q.sum(axis=1))


def predict(f0, times, r0, Q, effects, u, log_size=False):
    """
    Proportions (T, K) at `times` from f0 at times[0]: effects (K, S) and u (T - 1, S) = value of each stimulus over
    each interval (times[i], times[i + 1]]. With log_size, also the log population size (T,) relative to times[0].
    """
    f = np.asarray(f0, dtype=float)
    f = f / f.sum()
    out, logn = [f], [0.0]
    for i in range(len(times) - 1):
        n = np.clip(expm(generator(r0 + effects @ u[i], Q) * (times[i + 1] - times[i])) @ f, 0, None)
        logn.append(logn[-1] + np.log(max(n.sum(), 1e-300)))
        f = n / max(n.sum(), 1e-300)
        out.append(f)
    return (np.array(out), np.array(logn)) if log_size else np.array(out)


def fit_population(f_obs, n_cells, times, effects, u, r_prior=None, r_fixed=None, q_prior=None, sizes=None,
                   r_sd=None, **kw):
    """
    Rates and transitions of one sample from its proportions f_obs (T, K), observed on n_cells (T,) cells at `times`.

    r_prior (K,) : prescribed net rates without stimulus, which are corrected (fitted, prior sd_r); with r_fixed (K,)
        instead they are not fitted. q_prior (K, K) : prescribed transition rates (prior sd max(q_rel_sd q, sd_q)), else
        the transitions are free with a prior 0 of sd sd_q. sizes (T,): population sizes at `times` (NaN = unknown,
        any unit; partial anchors allowed, two known at least): the predicted log growth between the known ones must
        match (sd_logn). r_sd: prior sd of r (default sd_r, sd_r_sizes with population sizes: measured sizes dominate
        prescribed rates).
    Returns dict(r0, Q, pred, rmse, cost, size_rmse): rmse = root mean square error of the proportions (fraction, over
    times > 0); size_rmse = that of the log sizes (None without sizes).
    """
    p = {**DEFAULTS, **kw}
    f_obs = np.asarray(f_obs, dtype=float)
    T, K = f_obs.shape
    effects = np.zeros((K, 0)) if effects is None else np.asarray(effects, dtype=float)
    u = np.zeros((T - 1, effects.shape[1])) if u is None else np.asarray(u, dtype=float)
    fit_r = r_prior is not None
    r_ref = np.asarray(r_prior if fit_r else r_fixed, dtype=float)
    known = np.zeros(T, bool) if sizes is None else np.isfinite(np.asarray(sizes, dtype=float))
    log_obs = np.log(np.where(known, np.asarray(sizes if sizes is not None else np.ones(T), dtype=float), 1.0))
    use_sizes = known.sum() >= 2
    sd_r = float(r_sd) if r_sd is not None else (p['sd_r_sizes'] if use_sizes else p['sd_r'])
    off = ~np.eye(K, dtype=bool)
    q_ref = np.zeros((K, K)) if q_prior is None else np.asarray(q_prior, dtype=float)
    q_sd = np.full((K, K), p['sd_q']) if q_prior is None else np.maximum(p['q_rel_sd'] * q_ref, p['sd_q'])
    se = np.maximum(np.sqrt(np.clip(f_obs * (1 - f_obs), 1e-12, None) / np.maximum(np.asarray(n_cells, float), 1)[:, None]),
                    p['se_floor'])

    def unpack(x):
        r0 = x[:K] if fit_r else r_ref
        Q = np.zeros((K, K))
        Q[off] = x[K if fit_r else 0:]
        return r0, Q

    def size_residuals(logn):
        # Log growth between the known sizes, relative to the first known one
        k = np.flatnonzero(known)
        return ((logn[k[1:]] - logn[k[0]]) - (log_obs[k[1:]] - log_obs[k[0]])) / p['sd_logn']

    def residuals(x):
        r0, Q = unpack(x)
        pred, logn = predict(f_obs[0], times, r0, Q, effects, u, log_size=True)
        res = [((pred[1:] - f_obs[1:]) / se[1:] / np.sqrt(max(K - 1, 1))).ravel()]
        if fit_r:
            res.append((r0 - r_ref) / sd_r)
        res.append(((Q - q_ref) / q_sd)[off])
        if use_sizes:
            res.append(size_residuals(logn))
        return np.concatenate([np.atleast_1d(np.asarray(r, dtype=float)) for r in res])

    q0 = np.clip(q_ref[off], 1e-5, None)
    x0 = np.concatenate([r_ref if fit_r else [], q0])
    lo = np.concatenate([np.full(K, -0.2) if fit_r else [], np.zeros(len(q0))])
    hi = np.concatenate([np.full(K, 0.2) if fit_r else [], np.full(len(q0), 1.0)])
    best = None
    for scale in (1.0, 10.0, 0.1):                       # a few starts for the transitions
        x_start = np.clip(np.concatenate([x0[:K] if fit_r else [], x0[K if fit_r else 0:] * scale]), lo + 1e-12, hi - 1e-12)
        sol = least_squares(residuals, x_start, bounds=(lo, hi), x_scale=np.concatenate(
            [np.full(K, 0.003) if fit_r else [], np.full(len(q0), 0.003)]))
        if best is None or sol.cost < best.cost:
            best = sol
    r0, Q = unpack(best.x)
    pred, logn = predict(f_obs[0], times, r0, Q, effects, u, log_size=True)
    size_rmse = float(np.sqrt(np.mean((size_residuals(logn) * p['sd_logn']) ** 2))) if use_sizes else None
    return dict(r0=r0, Q=Q, pred=pred, rmse=float(np.sqrt(np.mean((pred[1:] - f_obs[1:]) ** 2))) if T > 1 else 0.0,
                cost=float(best.cost), size_rmse=size_rmse, log_size=logn)
