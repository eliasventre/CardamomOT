"""
Stratified sub-sampling shared by every batch construction (mixture, OT,
network, degradations): strata are cell types when available.
"""
import numpy as np


def stratified_order(idx, labels):
    """
    Random order of idx such that any contiguous window holds each stratum of
    labels in proportion (systematic sampling with a random offset per stratum).
    """
    keys = np.empty(len(idx))
    for lab in np.unique(labels):
        pos = np.flatnonzero(labels == lab)
        n = len(pos)
        keys[pos[np.random.permutation(n)]] = (np.arange(n) + np.random.random()) / n
    return idx[np.argsort(keys, kind='stable')]


def stratified_choice(idx, n, labels=None):
    """n elements of idx without replacement, in proportion to labels if given (all of idx if n >= len(idx))."""
    idx = np.asarray(idx)
    if n >= len(idx):
        return idx
    if labels is None:
        return np.random.choice(idx, n, replace=False)
    return stratified_order(idx, np.asarray(labels))[:n]


def grouped_partition(groups, budget, n_parts, labels=None):
    """
    n_parts sub-samples of ~budget rows: equal quota per group (e.g. time x sample),
    cell-type proportional within each group, and jointly covering as many rows as
    possible (disjoint when a group holds >= n_parts x quota rows, otherwise the
    repetitions are spread evenly). groups is a list of per-row key arrays.
    Returns (list of row arrays, full) with full = True when one sub-sample holds
    every row (then a single sub-sample is returned).
    """
    keys = np.stack([np.asarray(g) for g in groups], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    per_group = 1 + budget // len(uniq)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    if all(len(idx) <= per_group for idx in members):
        return [np.concatenate(members)], True
    parts = [[] for _ in range(n_parts)]
    for idx in members:
        q = min(per_group, len(idx))
        L = n_parts * q
        reps = int(np.ceil(L / len(idx)))
        # Consecutive chunks of stratified orders: disjoint parts, proportional in cell types
        order = np.concatenate([stratified_order(idx, labels[idx]) if labels is not None
                                else np.random.permutation(idx) for _ in range(reps)])[:L]
        for k in range(n_parts):
            parts[k].append(np.unique(order[k * q:(k + 1) * q]))
    return [np.concatenate(p) for p in parts], False


def grouped_subsample(groups, budget, labels=None):
    """
    Sub-sample of at most ~budget rows: equal quota per group (e.g. time x sample),
    cell-type proportional within each group. groups is a list of per-row key arrays.
    Returns (selected rows, True if every row was kept).
    """
    keys = np.stack([np.asarray(g) for g in groups], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    per_group = 1 + budget // len(uniq)
    sel, full = [], True
    for k in range(len(uniq)):
        idx = np.flatnonzero(inv == k)
        full &= len(idx) <= per_group
        sel.append(stratified_choice(idx, per_group, None if labels is None else labels[idx]))
    return np.concatenate(sel), full
