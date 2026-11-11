"""
Small helpers shared by the ordering options of the pseudotime cycle (``PLAN.md`` §5).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-16


def root_mask(model, roots, n_cells: int) -> np.ndarray:
    """
    Boolean mask of the root cells.

    ``roots`` is either ``None`` (read the origins from the current pseudotime, i.e. the cells at
    ``tau == 0``), a boolean mask, or an array of cell indices.
    """
    if roots is None:
        if getattr(model, "tau", None) is None:
            raise ValueError("no roots given and no pseudotime to read them from")
        tau = np.asarray(model.tau, dtype=float)
        return tau <= np.min(tau) + 1e-12
    roots = np.asarray(roots)
    if roots.dtype == bool:
        if roots.size != n_cells:
            raise ValueError(f"{roots.size} root flags for {n_cells} cells")
        return roots.copy()
    mask = np.zeros(n_cells, dtype=bool)
    mask[roots.astype(int)] = True
    return mask
