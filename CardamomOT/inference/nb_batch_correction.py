"""
nb_batch_correction.py
-----------------------
Per-gene Negative-Binomial batch & library-size correction for the CARDAMOM /
CardamomOT pipeline, designed to run *before* gene selection so that the whole
pipeline is coherent with the NB assumption.

Rationale
=========
The historical pipeline did library-size / batch correction with ad-hoc
upstream preprocessing, *then* selected genes without reference to the NB
model, *then* fitted a per-gene NB mixture. This module folds the
normalization step into a count model instead, exactly like ComBat-seq
(Zhang, Parmigiani & Johnson, NAR Genomics & Bioinformatics 2020), and it is
built to be consistent with the NB parameterization used in ``mixture.py``:

    X | component k  ~  NB(mean = ks_k / c,  Fano = 1 + 1/c)

i.e. the dispersion ``c`` is shared across components, so a library-size or
batch effect is a purely *multiplicative* factor on the mean. That is why it
enters cleanly as a GLM **offset** (library size) and a **batch coefficient**,
without touching ``c``.

What it does, per gene g
========================
1. Fit a Poisson GLM   log E[y_ng] = log(ell_n) + D_n . beta_g
   where ``ell_n`` is the library size (offset), and the design matrix ``D``
   contains an intercept, the *biological covariates to preserve* (e.g. time,
   cell_type) as factors, and the *batch* factor ``dataset_id`` to remove.
   Poisson point estimates of the mean coefficients are consistent under
   over-dispersion (quasi-Poisson argument), which is all we need here.
2. Estimate a per-gene NB over-dispersion ``phi_g`` (var = mu + phi*mu^2) by
   method of moments on the Pearson residuals, with optional empirical-Bayes
   shrinkage toward the cross-gene trend.
3. Map every raw count through matched NB quantiles from its *observed* batch
   (and depth) to a common *reference* batch and reference depth
   (ComBat-seq ``match_quantiles``). This yields **integer** corrected counts.

Why this preserves biology (your main worry)
=============================================
Conditional on the preserved covariates, the quantile map is a *monotone*
transformation of the counts within each (covariate) group. A monotone map
cannot create or destroy modes: bistability / multiple basins within a
timepoint are preserved; only the batch-induced location/scale and the depth
are harmonized. Contrast this with a low-rank latent model (scVI), which would
collapse per-gene modes into a shared continuous manifold.

IMPORTANT CAVEAT
================
Batch correction is only valid if ``dataset_id`` is a *technical* axis (samples
/ replicates of the same biological design). Any biology that differs across
``dataset_id`` and that you want to keep MUST be listed in ``preserve``,
otherwise it will be removed. Because in this setting timepoints are captured
*together* within a sample, ``time`` is never confounded with ``dataset_id`` --
but if two ``dataset_id`` correspond to genuinely different conditions, do not
batch-correct across them.

Dependencies: numpy, scipy, pandas (all already used by CardamomOT). Optional:
joblib for parallelism, anndata for the convenience wrapper.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import nbinom

# Reuse primitives from the mixture engine so the two modules stay in sync.
try:
    from CardamomOT.mixture import EPS  # type: ignore
    from CardamomOT.logging import get_logger  # type: ignore
    logger = get_logger(__name__)
except Exception:  # pragma: no cover - standalone / testing fallback
    import logging
    EPS = 1e-16
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:  # pragma: no cover
    _HAS_JOBLIB = False


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------
def _one_hot(values: np.ndarray, drop_level: Any = None) -> tuple[np.ndarray, list]:
    """One-hot encode a categorical vector, dropping ``drop_level`` (reference).

    Returns the (N, L-1) indicator matrix and the ordered list of *kept* levels.
    If ``drop_level`` is None the first level (sorted) is dropped.
    """
    values = np.asarray(values)
    levels = sorted(pd.unique(values).tolist(), key=lambda v: str(v))
    if drop_level is None:
        drop_level = levels[0]
    kept = [lv for lv in levels if lv != drop_level]
    if not kept:
        return np.zeros((values.shape[0], 0)), []
    mat = np.zeros((values.shape[0], len(kept)), dtype=float)
    for j, lv in enumerate(kept):
        mat[:, j] = (values == lv).astype(float)
    return mat, kept


def _build_design(
    obs: pd.DataFrame,
    preserve_keys: Sequence[str],
    batch_key: str,
    ref_batch: Any,
) -> tuple[np.ndarray, np.ndarray, list[str], list]:
    """Build the design matrix ``D`` (with batch) and ``D_ref`` (batch removed).

    Both matrices share the intercept + preserved-covariate columns; ``D_ref``
    has the batch indicator columns set to zero, i.e. it evaluates every cell as
    if it belonged to the reference batch.
    """
    n = obs.shape[0]
    cols: list[np.ndarray] = [np.ones((n, 1))]
    names: list[str] = ["intercept"]
    batch_col_idx: list[int] = []

    # Preserved biological covariates (kept identical in D and D_ref).
    for key in preserve_keys:
        if key is None or key not in obs:
            continue
        vals = obs[key].values
        # Numeric non-categorical covariate with many levels -> keep as-is (centered);
        # otherwise treat as a factor.
        if np.issubdtype(np.asarray(vals).dtype, np.number) and len(pd.unique(vals)) > 12:
            v = np.asarray(vals, dtype=float)
            cols.append(((v - np.nanmean(v)) / (np.nanstd(v) + EPS)).reshape(-1, 1))
            names.append(f"{key}[num]")
        else:
            mat, kept = _one_hot(vals)
            for j, lv in enumerate(kept):
                cols.append(mat[:, [j]])
                names.append(f"{key}={lv}")

    # Batch factor (removed in D_ref).
    batch_mat, batch_kept = _one_hot(obs[batch_key].values, drop_level=ref_batch)
    start = sum(c.shape[1] for c in cols)
    for j, lv in enumerate(batch_kept):
        cols.append(batch_mat[:, [j]])
        names.append(f"{batch_key}={lv}")
        batch_col_idx.append(start + j)

    D = np.hstack(cols) if cols else np.ones((n, 1))
    D_ref = D.copy()
    if batch_col_idx:
        D_ref[:, batch_col_idx] = 0.0
    return D, D_ref, names, batch_kept


# ---------------------------------------------------------------------------
# Per-gene Poisson IRLS (offset = log library size) + NB dispersion
# ---------------------------------------------------------------------------
def _poisson_irls(
    y: np.ndarray,
    D: np.ndarray,
    offset: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
    ridge: float = 1e-6,
    eta_clip: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Poisson GLM with an offset via ridge-stabilized IRLS.

    Returns ``(beta, mu_fitted)``. Robust to separation / all-zero batches
    thanks to the small ridge and eta clipping.
    """
    n, p = D.shape
    beta = np.zeros(p)
    # Warm start the intercept at the marginal log-rate.
    beta[0] = np.log(max(y.mean(), 1e-3)) - float(np.mean(offset))
    R = ridge * np.eye(p)
    mu = np.exp(np.clip(D @ beta + offset, -eta_clip, eta_clip))
    for _ in range(max_iter):
        w = np.maximum(mu, 1e-8)                       # Poisson weights = mu
        z = (D @ beta) + (y - mu) / w                  # working response (offset drops out)
        Dw = D * w[:, None]
        A = D.T @ Dw + R
        b = D.T @ (w * z)
        try:
            beta_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(A, b, rcond=None)[0]
        eta = np.clip(D @ beta_new + offset, -eta_clip, eta_clip)
        mu_new = np.exp(eta)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta, mu = beta_new, mu_new
            break
        beta, mu = beta_new, mu_new
    return beta, mu


def _estimate_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """Method-of-moments NB over-dispersion phi with var = mu + phi*mu^2."""
    num = np.sum((y - mu) ** 2 - mu)
    den = np.sum(mu ** 2) + EPS
    return float(max(num / den, 1e-6))


def _match_quantiles(
    y: np.ndarray,
    mu_old: np.ndarray,
    mu_new: np.ndarray,
    phi: float,
) -> np.ndarray:
    """ComBat-seq matched-quantile mapping between two NB distributions.

    Maps count ``y`` (mean ``mu_old``) to the count with the same lower-tail
    probability under mean ``mu_new`` (shared over-dispersion ``phi``). Counts
    0 and 1 are left untouched (as in ComBat-seq) to avoid inflating structural
    zeros.
    """
    r = 1.0 / max(phi, 1e-6)                       # NB "size" (number of failures)
    out = y.astype(float).copy()
    mask = y >= 2
    if not np.any(mask):
        return np.rint(out).astype(int)

    p_old = r / (r + np.maximum(mu_old[mask], EPS))
    p_new = r / (r + np.maximum(mu_new[mask], EPS))
    tmp = nbinom.cdf(y[mask] - 1, r, p_old)
    tmp = np.clip(tmp, 0.0, 1.0 - 1e-9)            # guard outliers (cdf ~ 1 -> inf)
    mapped = 1.0 + nbinom.ppf(tmp, r, p_new)

    bad = ~np.isfinite(mapped)
    mapped[bad] = y[mask][bad]                      # fall back to original on failure
    out[mask] = mapped
    return np.rint(out).astype(int)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def correct_counts(
    counts: np.ndarray,
    obs: pd.DataFrame,
    batch_key: str = "dataset_id",
    preserve: Sequence[str] = ("time",),
    library_size: Optional[np.ndarray] = None,
    correct_library_size: bool = True,
    reference_depth: Optional[float] = None,
    ref_batch: Any = None,
    shrink_dispersion: bool = True,
    max_fit_cells: Optional[int] = 20000,
    n_jobs: int = 1,
    random_state: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Batch- and (optionally) depth-correct a raw count matrix, gene by gene.

    Parameters
    ----------
    counts : (N, G) array of non-negative integer counts (dense).
    obs    : DataFrame with N rows containing ``batch_key`` and ``preserve``.
    batch_key : nuisance factor to remove (default 'dataset_id').
    preserve  : biological covariates to KEEP (kept in the design so they are
                not absorbed into the batch correction). Typically ('time',) or
                ('time', 'cell_type').
    library_size : (N,) per-cell offset. If None, uses total counts per cell.
    correct_library_size : if True, also harmonize sequencing depth to
                ``reference_depth`` (default = MAX library size, see below). If
                False, only the batch effect is removed and per-cell depth is
                kept.
    reference_depth : common depth to map to. Default: the maximum observed
                library size (NOT the median). Rationale: quantile-mapping a
                cell DOWN to a smaller depth means rounding its (already
                discrete, often small) counts down -- for a weakly-expressed
                gene this can erase the only nonzero counts it has, silently
                deleting real signal and, in the worst case, collapsing a
                gene's per-timepoint means to an exact tie it did not have in
                the raw data. Mapping UP to the max depth only ever inflates
                counts, which is lossless for an NB mixture fit downstream
                (it changes scale, not shape) and never destroys information.
                The trade-off is that absolute count values become larger and
                less directly interpretable as "raw-like" counts -- pass an
                explicit value (e.g. the median) if you specifically need
                corrected counts on a realistic depth scale and accept the
                small risk of eroding rare low counts.
    ref_batch : batch level used as the reference (default: the largest batch).
    shrink_dispersion : empirical-Bayes-lite shrinkage of log-dispersion toward
                the cross-gene median (stabilizes small/rare genes).
    max_fit_cells : cap on cells used to *fit* the GLM per batch-balanced
                subsample (the quantile map is still applied to ALL cells).
    n_jobs : parallel workers (requires joblib).

    Returns
    -------
    corrected : (N, G) integer array.
    report    : per-gene DataFrame with dispersion and per-batch log2 fold
                changes (the estimated technical shifts that were removed).
    """
    counts = np.asarray(counts)
    N, G = counts.shape
    rng = np.random.default_rng(random_state)

    # No batch column, or a single level: there is nothing to remove on the
    # "batch" axis, but library-size harmonization is orthogonal and still
    # makes sense. We materialize a dummy constant batch column so the design
    # matrix below has zero batch indicator columns (D_ref == D on that axis),
    # and the only thing correct_counts does is the depth normalization.
    if batch_key not in obs or len(pd.unique(obs[batch_key].values)) <= 1:
        if not correct_library_size:
            logger.warning(
                "No batch factor to remove ('%s' missing or single-valued) and "
                "correct_library_size=False -> nothing to correct, returning raw counts.",
                batch_key,
            )
            return counts.astype(int), pd.DataFrame({"gene_index": np.arange(G)})
        logger.info(
            "No batch factor to remove ('%s' missing or single-valued) -> "
            "applying LIBRARY-SIZE harmonization only (no batch correction).",
            batch_key,
        )
        obs = obs.copy()
        obs[batch_key] = "single_batch"  # dummy: yields zero batch indicator columns
    n_batches = len(pd.unique(obs[batch_key].values))

    if library_size is None:
        library_size = counts.sum(axis=1).astype(float)
    library_size = np.asarray(library_size, dtype=float)
    library_size = np.maximum(library_size, 1.0)
    offset = np.log(library_size)

    if ref_batch is None:
        vc = pd.Series(obs[batch_key].values).value_counts()
        ref_batch = vc.index[0]
    if reference_depth is None:
        reference_depth = float(np.max(library_size))
    offset_ref_const = np.log(max(reference_depth, 1.0))

    D, D_ref, names, batch_levels = _build_design(obs, preserve, batch_key, ref_batch)
    batch_name_idx = [i for i, nm in enumerate(names) if nm.startswith(f"{batch_key}=")]

    # Optional batch-balanced subsample for *fitting* only.
    if max_fit_cells is not None and N > max_fit_cells:
        per = max(1, max_fit_cells // n_batches)
        fit_idx = []
        bvals = obs[batch_key].values
        for lv in pd.unique(bvals):
            idx = np.where(bvals == lv)[0]
            fit_idx.append(rng.choice(idx, min(per, idx.size), replace=False))
        fit_idx = np.concatenate(fit_idx)
    else:
        fit_idx = np.arange(N)

    Dfit, offfit = D[fit_idx], offset[fit_idx]

    def _fit_one(g: int):
        print(g)
        y_all = counts[:, g].astype(float)
        y_fit = y_all[fit_idx]
        if y_fit.max() <= 1 or np.all(y_fit == y_fit[0]):
            return y_all.astype(int), np.nan, np.zeros(len(batch_levels))
        beta, mu_fit = _poisson_irls(y_fit, Dfit, offfit)
        phi = _estimate_dispersion(y_fit, mu_fit)
        # Fitted mean WITH batch (observed) and WITHOUT batch at ref depth.
        eta_old = np.clip(D @ beta + offset, -30, 30)
        mu_old = np.exp(eta_old)
        off_ref = np.full(N, offset_ref_const) if correct_library_size else offset
        eta_new = np.clip(D_ref @ beta + off_ref, -30, 30)
        mu_new = np.exp(eta_new)
        corrected_g = _match_quantiles(counts[:, g].astype(int), mu_old, mu_new, phi)
        batch_lfc = (beta[batch_name_idx] / np.log(2)) if batch_name_idx else np.zeros(0)
        return corrected_g, phi, batch_lfc

    if n_jobs != 1 and _HAS_JOBLIB:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_fit_one)(g) for g in range(G)
        )
    else:
        results = [_fit_one(g) for g in range(G)]

    corrected = np.empty((N, G), dtype=int)
    phis = np.empty(G)
    lfcs = np.zeros((G, len(batch_levels)))
    for g, (cg, phi, blfc) in enumerate(results):
        corrected[:, g] = cg
        phis[g] = phi
        if blfc.size:
            lfcs[g, :] = blfc

    # EB-lite shrinkage of dispersion toward the cross-gene median (log-space).
    if shrink_dispersion:
        valid = np.isfinite(phis) & (phis > 0)
        if valid.sum() > 10:
            log_phi = np.log(phis[valid])
            target = np.median(log_phi)
            w = 0.3  # shrinkage weight; light by design
            phis[valid] = np.exp((1 - w) * log_phi + w * target)

    report = pd.DataFrame({"gene_index": np.arange(G), "dispersion_phi": phis})
    for j, lv in enumerate(batch_levels):
        report[f"log2FC[{batch_key}={lv}]"] = lfcs[:, j]
    report[f"reference_batch"] = str(ref_batch)
    report["reference_depth"] = reference_depth
    return corrected, report


def correct_adata(
    adata,
    batch_key: str = "dataset_id",
    preserve: Sequence[str] = ("time",),
    layer: Optional[str] = None,
    out_layer: str = "corrected",
    inplace_X: bool = True,
    **kwargs,
):
    """Convenience wrapper around :func:`correct_counts` for an AnnData.

    Reads raw counts from ``adata.layers[layer]`` (or ``adata.X``), computes the
    library size from the *full* matrix, stores corrected integer counts in
    ``adata.layers[out_layer]`` and (if ``inplace_X``) also in ``adata.X``.
    Returns the per-gene report DataFrame.
    """
    import scipy.sparse as sp

    X = adata.layers[layer] if layer is not None else adata.X
    dense = X.toarray() if sp.issparse(X) else np.asarray(X)
    dense = np.rint(dense).astype(float)          # ensure counts
    lib = dense.sum(axis=1)

    preserve = tuple(k for k in preserve if k in adata.obs)
    corrected, report = correct_counts(
        dense, adata.obs, batch_key=batch_key, preserve=preserve,
        library_size=lib, **kwargs,
    )
    adata.layers[out_layer] = corrected
    if inplace_X:
        adata.X = corrected
    report.insert(0, "gene", np.asarray(adata.var_names))
    return report