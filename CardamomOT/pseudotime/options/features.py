"""
Feature space of the cells, shared by the ordering options (``PLAN.md`` §5.1).

Option 3 builds a kNN graph and Option 1 looks for the candidate descendants of a frontier cell in
this space, so both need the same embedding of a cell.  Three choices:

* ``'posterior'`` (default): expected burst mode of each gene,
  :math:`\\sum_m \\pi_{c,g,m}\\,k_{s,g,m}` — the *expected* promoter activity.  The hard modes
  ``model.modes`` are discrete (one value per basin): with 2 modes and many genes most cells are
  exact duplicates, the kNN graph breaks into one component per basin pattern and most cells become
  unreachable;
* ``'hard'``: the assigned burst modes (``model.modes``), for comparison;
* ``'expression'``: log1p of the counts (no mixture at all).

The features are standardised per gene, optionally weighted by the separation of the two modes of
each gene (``gene_weights``), optionally PCA-reduced (large ``G``), then standardised again.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from ...inference.network import gene_weights

EPS = 1e-16
FEATURE_WEIGHTS = ("gene_weights", "std", "none")
FEATURE_SPACES = ("posterior", "hard", "expression")


def feature_matrix(model, feature: str = "posterior", feature_weight: str = "gene_weights",
                   n_pcs: int = 0, seed: int = 0) -> np.ndarray:
    """
    (N, G') feature matrix of the cells of ``model`` (a PseudotimeNetworkModel with a fitted
    mixture; ``model.counts`` is needed by ``feature='expression'``).
    """
    if feature not in FEATURE_SPACES:
        raise ValueError(f"feature must be one of {FEATURE_SPACES}, got {feature!r}")
    if feature_weight not in FEATURE_WEIGHTS:
        raise ValueError(f"feature_weight must be one of {FEATURE_WEIGHTS}, "
                         f"got {feature_weight!r}")
    ns = int(model.n_stimuli)
    proba = getattr(model, "proba_init", None)
    if feature == "expression" and model.counts is not None:
        F = np.log1p(np.asarray(model.counts, dtype=float))
    elif feature == "posterior" and proba is not None:
        ks, _, _ = model._mixture_terms()
        ks = ks[0] if np.ndim(ks) == 3 else ks
        M = int(min(np.shape(proba)[2], ks.shape[1]))
        F = (np.asarray(proba, dtype=float)[:, ns:, :M] * ks[None, ns:, :M]).sum(axis=-1)
    else:
        F = np.asarray(model.modes[:, ns:], dtype=float)

    gene_w = gene_weights(_current_modes(model))[ns:]

    sd = F.std(axis=0)
    keep = sd > EPS
    F = (F[:, keep] - F[:, keep].mean(axis=0)) / sd[keep]
    if feature_weight == "gene_weights":
        F = F * gene_w[keep][None, :]
    if n_pcs and int(n_pcs) < F.shape[1]:
        F = PCA(n_components=int(n_pcs), random_state=seed).fit_transform(F)
    sd = F.std(axis=0)
    return (F - F.mean(axis=0)) / np.maximum(sd, EPS)


def _current_modes(model) -> np.ndarray:
    """``(G_tot, M)`` burst modes of the model (shape depends on the mixture version)."""
    ks, _, _ = model._mixture_terms()
    return ks[0] if np.ndim(ks) == 3 else ks
