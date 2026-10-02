"""
Minimal multi-sample integration of the NB mixture model.

Each sample identifies its own mixture modes; only the VALUES of the modes may
differ between samples. Per gene, global mode values are taken from a reference
sample, or are the cell-weighted average of the sample mode means (which keeps
the total expected counts). Counts are then transported, mode by mode, from the
sample (ZI)NB to the global one by randomized quantile matching, so they stay
integer counts.

NB convention of the package: X ~ NB(k, c) with pmf ∝ c^k / (1 + c)^(x + k),
i.e. scipy nbinom(n=k, p=c/(1+c)), mean k/c.
"""
import numpy as np
from scipy.stats import nbinom


def zinb_cdf(x, k, c, pi0, s=1.0):
    """CDF of the zero-inflated NB at integer x (-1 → 0); s: per-cell depth factors (rate c / s)."""
    x = np.asarray(x, dtype=float)
    F = pi0 + (1 - pi0) * nbinom.cdf(x, k, c / (s + c))
    return np.where(x < 0, 0.0, F)


def zinb_ppf(u, k, c, pi0, s=1.0):
    """Quantile function of the zero-inflated NB (s: per-cell depth factors)."""
    u = np.asarray(u, dtype=float)
    v = np.clip((u - pi0) / max(1 - pi0, 1e-12), 0.0, 1 - 1e-12)
    return np.where(u <= pi0, 0.0, nbinom.ppf(v, k, c / (s + c)))


def quantile_match(x, src, dst, rng, s=1.0):
    """
    Randomized probability integral transform of counts x from ZINB src=(k, c, pi0)
    to ZINB dst=(k, c, pi0): u ~ U[F(x-1), F(x)], then F_dst^{-1}(u). s: per-cell depth
    factors, kept by the transform (both distributions at the cell's depth).
    """
    lo, hi = zinb_cdf(x - 1, *src, s=s), zinb_cdf(x, *src, s=s)
    u = lo + rng.random(np.shape(x)) * (hi - lo)
    return zinb_ppf(np.clip(u, 0.0, 1 - 1e-12), *dst, s=s)


def supported_modes(a_s, proba_s, ns, min_weight=0.05, min_ratio=2.0):
    """
    (G_tot,) bool: the mixture of a sample really separates OFF/ON for each gene,
    i.e. every mode has weight >= min_weight and the highest/lowest mean ratio >= min_ratio
    (otherwise its "ON" mode is a phantom, e.g. a gene never activated in that sample).
    """
    G_tot = a_s.shape[1]
    out = np.zeros(G_tot, dtype=bool)
    for g in range(ns, G_tot):
        k = a_s[:-1, g]
        n_modes = int(np.sum(proba_s[:, g, :].sum(axis=0) > 0))
        n_modes = max(n_modes, 1)
        if n_modes < 2:
            continue
        w = proba_s[:, g, :n_modes].mean(axis=0)
        out[g] = (w.min() >= min_weight) and (k[n_modes - 1] >= min_ratio * max(k[0], 1e-12))
    return out


def consistent_modes(a_samples, proba_samples, ok, ns):
    """
    Cross-sample check of the ON mode: for each (sample, gene) still ok, the highest mode mean of
    the sample must be closer (in log) to the median ON level of the other ok samples than to their
    median OFF level. Catches samples splitting zeros from low counts of a gene never activated
    there (large ON/OFF ratio, yet an "ON" mode at the OFF level of the others).
    """
    S, _, G_tot = a_samples.shape
    out = ok.copy()
    for g in range(ns, G_tot):
        idx = np.flatnonzero(ok[:, g])
        if len(idx) < 2:
            continue
        lo, hi = np.zeros(S), np.zeros(S)
        for s in idx:
            n_modes = max(2, int(np.sum(proba_samples[s][:, g, :].sum(axis=0) > 0)))
            mean = np.maximum(a_samples[s, :n_modes, g] / a_samples[s, -1, g], 1e-6)
            lo[s], hi[s] = np.log(mean[0]), np.log(mean[-1])
        for s in idx:
            others = idx[idx != s]
            on, off = np.median(hi[others]), np.median(lo[others])
            if abs(hi[s] - on) > abs(hi[s] - off):
                out[s, g] = False
    return out


def global_parameters(a_samples, pi0_samples, proba_samples, eligible, ref=None):
    """
    Global mode values per gene from the eligible samples.

    a_samples : (S, M+1, G_tot) mode values (rows :-1) and dispersion c (last row) per sample
    pi0_samples : (S, G_tot) zero-inflation per sample (0 for stimulus columns)
    proba_samples : list of (N_s, G_tot, M) responsibilities
    eligible : (S, G_tot) bool, samples used for each gene
    ref : index of the reference sample or None (cell-weighted average of the mode means)

    Returns a (M+1, G_tot), pi0 (G_tot,) and has_global (G_tot,) = at least one eligible sample.
    """
    S, M1, G_tot = a_samples.shape
    a = np.full((M1, G_tot), np.nan)
    pi0 = np.zeros(G_tot)
    has = eligible.any(axis=0)
    n_cells = np.array([len(p) for p in proba_samples], dtype=float)
    for g in np.flatnonzero(has):
        if ref is not None and eligible[ref, g]:
            a[:, g] = a_samples[ref, :, g]
            pi0[g] = pi0_samples[ref, g]
            continue
        e = np.flatnonzero(eligible[:, g])
        c_s = a_samples[e, -1, g]
        c = np.sum(n_cells[e] * c_s) / n_cells[e].sum()
        # Cells of each sample in each mode, and mode means k/c
        n_z = np.stack([proba_samples[s][:, g, :].sum(axis=0) for s in e])          # (|e|, M)
        mean_z = a_samples[e, :-1, g] / c_s[:, None]
        mean = np.sum(n_z * mean_z, axis=0) / np.maximum(n_z.sum(axis=0), 1e-12)
        a[:-1, g] = c * mean
        a[-1, g] = c
        pi0[g] = np.sum(n_cells[e] * pi0_samples[e, g]) / n_cells[e].sum()
    return a, pi0, has


def _moments(mu, c, p, w):
    """Mean and variance of the gene over all cells: mixture of ZINB(mu, c, p) components with weights w."""
    m1 = (1 - p) * mu
    m2 = (1 - p) * (mu + mu ** 2 / c + mu ** 2)
    E = np.sum(w * m1)
    return E, np.sum(w * m2) - E ** 2


def interpolate_parameters(a_samples, pi0_samples, proba_samples, eligible, a_target, pi0_target,
                           lam, calibrate, ns, c_max=9.0):
    """
    Per-sample mixture parameters at integration level lam in [0, 1]: for each eligible
    (sample, gene) pair, mode means and dispersion are interpolated in log scale (zero inflation
    linearly) from the sample's own fit (lam = 0) to the target (lam = 1).

    calibrate: then, per gene, a factor common to all samples and modes rescales the means so that
    the mean of the gene over all cells (mode occupancies of the fits, ZINB moments) is that of
    the own fits, and a common factor on 1/c restores their total variance when possible
    (c clipped to [1e-3, c_max]).

    Returns a (S, M+1, G) and pi0 (S, G); non-eligible pairs are left as in a_samples.
    """
    a_out, pi_out = a_samples.copy(), pi0_samples.copy()
    for g in range(ns, a_samples.shape[2]):
        e = np.flatnonzero(eligible[:, g])
        if not len(e):
            continue
        c_s = a_samples[e, -1, g]
        mu_s = np.maximum(a_samples[e, :-1, g] / c_s[:, None], 1e-12)          # (|e|, M)
        p_s = pi0_samples[e, g][:, None]
        w = np.stack([proba_samples[s][:, g, :].sum(axis=0) for s in e])        # mode occupancies
        w = w / max(w.sum(), 1e-12)
        mu_t = np.maximum(a_target[:-1, g] / a_target[-1, g], 1e-12)
        mu = np.exp((1 - lam) * np.log(mu_s) + lam * np.log(mu_t)[None])
        c = np.exp((1 - lam) * np.log(c_s) + lam * np.log(a_target[-1, g]))[:, None]
        p = (1 - lam) * p_s + lam * pi0_target[g]
        if calibrate and lam > 0:
            E0, V0 = _moments(mu_s, c_s[:, None], p_s, w)
            mu = mu * E0 / max(np.sum(w * (1 - p) * mu), 1e-12)
            A = np.sum(w * (1 - p) * (mu + mu ** 2))
            B = np.sum(w * (1 - p) * mu ** 2 / c)
            beta = (V0 + E0 ** 2 - A) / B if B > 0 else -1.0
            if beta > 0:
                c = np.clip(c / beta, 1e-3, c_max)
        a_out[e, :-1, g] = c * mu
        a_out[e, -1, g] = c[:, 0]
        pi_out[e, g] = p[:, 0]
    return a_out, pi_out


def nb_cell_parameters(a, pi_zinb, sample_idx=None):
    """
    NB parameters per cell from shared (M+1, G) or per-sample (S, M+1, G) mixtures: max burst
    rate k1 and dispersion c, (1 or N, G_tot), and zero inflation (1 or N, G_tot - ns).
    sample_idx: (N,) sample index of each cell, needed with per-sample mixtures.
    """
    a, pz = np.asarray(a), np.asarray(pi_zinb)
    if a.ndim == 2:
        return a[:-1].max(axis=0)[None], a[-1][None], pz.reshape(1, -1)
    idx = np.zeros(1, dtype=int) if sample_idx is None else np.minimum(np.asarray(sample_idx).astype(int), len(a) - 1)
    a_c = a[idx]
    return a_c[:, :-1].max(axis=1), a_c[:, -1], (pz[np.minimum(idx, len(pz) - 1)] if pz.ndim == 2 else pz[None])


def integrate_counts(x, z, src, dst, rng, s=None):
    """Quantile-match counts x (N,) of one gene/sample, cell n being in mode z[n]; src/dst = (k (M,), c, pi0);
    s: (N,) depth factors or None."""
    out = np.asarray(x, dtype=float).copy()
    for m in np.unique(z):
        sel = z == m
        out[sel] = quantile_match(out[sel], (src[0][m], src[1], src[2]), (dst[0][m], dst[1], dst[2]), rng,
                                  s=1.0 if s is None else s[sel])
    return out
