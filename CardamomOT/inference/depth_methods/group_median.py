"""
s_i = library of the cell / median library of its reference population (groups = 'sample|time[|cell type]' labels):
- reference='sample_any' (default): its group (time-resolved; depth differences between the groups not corrected);
- reference='sample_all': its whole sample (standard library-size normalisation within a sample).
"""
import numpy as np

DEFAULTS = dict(reference='sample_any')
REFERENCES = ('sample_any', 'sample_all')


def compute(X, lib, groups, reference='sample_any'):
    if reference not in REFERENCES:
        raise ValueError(f"group_median depth: reference '{reference}' (use one of {REFERENCES})")
    lib = np.asarray(lib, dtype=float)
    groups = np.asarray(groups)
    pops = groups if reference == 'sample_any' else np.array([str(g).split('|')[0] for g in groups])
    s = np.ones_like(lib)
    for g in np.unique(pops):
        m = pops == g
        s[m] = lib[m] / max(np.median(lib[m]), 1e-12)
    return np.maximum(s, 1e-3)
