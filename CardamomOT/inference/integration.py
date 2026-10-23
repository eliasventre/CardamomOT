"""
Multi-sample integration of the NB mixture model: raw projection on the reference sample.

Each sample identifies its own mixture modes (or levels of the continuous model); only their VALUES may differ
between samples. Each mode of a sample is sent on the mode of the reference that best explains its cells taken as a
whole (projection_targets: cells grouped in a sample stay grouped in the reference, and a mode of a gene expressed only in
another sample is not sent on the expressed mode of the reference), and the counts of each cell are transported from
the NB of its mode in its sample to the NB of its target by randomized quantile matching, so they stay integer counts.

NB convention of the package: X ~ NB(k, c) with pmf ∝ c^k / (1 + c)^(x + k),
i.e. scipy nbinom(n=k, p=c/(1+c)), mean k/c.
"""
import numpy as np
from scipy.stats import nbinom

from .depth_model import nb_numpy_params, nb_shape_rate


def nb_cdf(x, k, c, s=1.0):
    """CDF of the NB at integer x (-1 → 0); s: per-cell depth factors (depth_model)."""
    x = np.asarray(x, dtype=float)
    return np.where(x < 0, 0.0, nbinom.cdf(x, *nb_numpy_params(k, c, s)))


def nb_ppf(u, k, c, s=1.0):
    """Quantile function of the NB (s: per-cell depth factors)."""
    # scipy returns -1 at u = 0
    return np.maximum(nbinom.ppf(np.clip(np.asarray(u, dtype=float), 0.0, 1 - 1e-12), *nb_numpy_params(k, c, s)), 0.0)


def quantile_match(x, src, dst, rng, s=1.0):
    """
    Randomized probability integral transform of counts x from NB src=(k, c) to NB dst=(k, c):
    u ~ U[F(x-1), F(x)], then F_dst^{-1}(u). s: per-cell depth factors, kept by the transform
    (both distributions at the cell's depth).
    """
    lo, hi = nb_cdf(x - 1, *src, s=s), nb_cdf(x, *src, s=s)
    u = lo + rng.random(np.shape(x)) * (hi - lo)
    return nb_ppf(u, *dst, s=s)


def nb_cell_parameters(a, sample_idx=None):
    """
    NB parameters per cell from shared (M+1, G) or per-sample (S, M+1, G) mixtures: max burst
    rate k1 and dispersion c, (1 or N, G_tot). sample_idx: (N,) sample index of each cell,
    needed with per-sample mixtures.
    """
    a = np.asarray(a)
    if a.ndim == 2:
        return a[:-1].max(axis=0)[None], a[-1][None]
    idx = np.zeros(1, dtype=int) if sample_idx is None else np.minimum(np.asarray(sample_idx).astype(int), len(a) - 1)
    a_c = a[idx]
    return a_c[:, :-1].max(axis=1), a_c[:, -1]


def integrate_counts(x, z, src, dst, rng, s=None):
    """Quantile-match counts x (N,) of one gene/sample, cell n being in mode z[n]; src/dst = (k (M,), c);
    s: (N,) depth factors or None."""
    out = np.asarray(x, dtype=float).copy()
    for m in np.unique(z):
        sel = z == m
        out[sel] = quantile_match(out[sel], (src[0][m], src[1]), (dst[0][m], dst[1]), rng,
                                  s=1.0 if s is None else s[sel])
    return out


def _group_scores(x, groups, cands, c_ref, s=None):
    """
    (n_groups, n_cands) log-likelihood of the mean count of each group of cells (boolean masks) under each candidate of
    the reference, + log of its mass. A candidate is (burst frequencies (L,), masses (L,)): one mode, or a mixture of
    levels. The mean of a group of n cells is ~ N(mean of the candidate, its variance / n) at the depths of its cells:
    the group is taken as a whole (a mode with a broad NB explains each low count, not a group of them).
    """
    out = np.full((len(groups), len(cands)), -np.inf)
    for j, sel in enumerate(groups):
        n = int(np.sum(sel))
        if not n:
            continue
        s_g = np.ones(n) if s is None else np.asarray(s, dtype=float)[sel]
        for m, (ks, w) in enumerate(cands):
            ks, w = np.atleast_1d(np.asarray(ks, dtype=float)), np.atleast_1d(np.asarray(w, dtype=float))
            mass = w.sum()
            if mass <= 0:
                continue
            r, rate = nb_shape_rate(ks[None, :] * np.ones((n, 1)), c_ref, s_g[:, None])
            r, rate = np.maximum(np.asarray(r, dtype=float), 1e-12), np.asarray(rate, dtype=float)
            mu_l, v_l = r / rate, r * (1 + rate) / rate ** 2                    # (n, L) per cell and level
            p = w / mass
            mu_c = mu_l @ p
            v_c = (v_l + mu_l ** 2) @ p - mu_c ** 2                             # mixture of the levels, per cell
            var = max(np.sum(v_c) / n ** 2, 1e-12)
            out[j, m] = -0.5 * (np.mean(x[sel]) - np.mean(mu_c)) ** 2 / var - 0.5 * np.log(var) + np.log(mass)
    return out


def projection_targets(x, z, n_src, ks_ref, c_ref, w_ref, s=None, grid=False):
    """
    Target mode (index in the reference mixture (ks_ref (K,), c_ref, masses w_ref (K,)), (n_src,)) of each of the
    n_src modes of a sample, from its cells (counts x, mode z of largest posterior, depth s): each mode goes to the
    reference mode that best explains its cells taken as a whole (_group_scores), the targets kept non-decreasing (the
    order of the modes is kept). Grid (levels of the continuous model): the rule acts on the two halves of the levels
    (sigma <= 1/2, > 1/2) of the sample, with the two halves of the reference (mixtures of their levels) as candidates,
    a half going to the lowest or highest level (level by level, the rule would send each level on the reference
    level of the same counts and undo the projection); the levels are carried linearly between the two targets (the
    nearest reference level).
    """
    K = len(ks_ref)
    x = np.asarray(x, dtype=float)
    ks_ref, w_ref = np.asarray(ks_ref, dtype=float), np.asarray(w_ref, dtype=float)
    if not grid:
        sc = _group_scores(x, [z == m for m in range(n_src)], [(k, w) for k, w in zip(ks_ref, w_ref)], c_ref, s)
        tgt = np.array([int(np.argmax(sc[m])) if np.isfinite(sc[m]).any() else min(m, K - 1) for m in range(n_src)])
        return np.maximum.accumulate(tgt)
    low, half = z <= (n_src - 1) / 2, np.arange(K) <= (K - 1) / 2
    sc = _group_scores(x, [low, ~low], [(ks_ref[half], w_ref[half]), (ks_ref[~half], w_ref[~half])], c_ref, s)
    t_lo = (K - 1) * int(np.argmax(sc[0])) if low.any() else 0
    t_hi = (K - 1) * int(np.argmax(sc[1])) if (~low).any() else K - 1
    t_hi = max(t_hi, t_lo)
    return np.rint(t_lo + (t_hi - t_lo) * np.linspace(0.0, 1.0, n_src)).astype(int)
