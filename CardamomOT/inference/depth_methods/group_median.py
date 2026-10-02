"""s_i = library of the cell / median library of its group (sample, time, cell type)."""
import numpy as np


def compute(X, lib, groups):
    lib = np.asarray(lib, dtype=float)
    s = np.ones_like(lib)
    for g in np.unique(groups):
        m = groups == g
        s[m] = lib[m] / max(np.median(lib[m]), 1e-12)
    return np.maximum(s, 1e-3)
