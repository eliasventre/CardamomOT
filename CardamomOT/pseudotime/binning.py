"""
Pseudotime -> bins: the time labels the CardamomOT cycle expects.

The whole CardamomOT inference works on *groups of cells* (``times``), with optimal transport
between consecutive groups.  In the pseudotime setting the groups are bins of the inferred
pseudotime; this module turns a per-cell pseudotime ``tau`` into that bin structure, with the two
rules of ``PLAN.md`` §4.2:

* **bin 0 is the root population** (and only it): bin 0 is a source bin, never a target, so a
  non-root cell placed there would never be reached by any transport;
* ``equal_width`` bins keep the *density* of cells along the pseudotime (this is what the growth
  pass reads, ``PLAN.md`` §6.3), while ``equal_count`` bins have a constant density by
  construction (ablation only).

Bins with fewer than ``min_cells`` cells are merged greedily with their successor, so that no
optimal transport problem is solved on a handful of cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

BIN_RULES = ("equal_width", "equal_count")


@dataclass
class BinScheme:
    """Cell -> pseudotime bin assignment, and the model time of each bin."""

    bin_of_cell: np.ndarray          # (N,) int: 0 = roots, 1..K-1 = the other cells
    bin_times: np.ndarray            # (K,) model time of each bin: k * dtau
    n_bins: int                      # K (bins actually used, after merging)
    dtau: float                      # time step between two bins
    counts: np.ndarray               # (K,) cells per bin
    edges: np.ndarray                # (K,) upper edge of each bin in normalised tau (diagnostics)
    rule: str
    total_duration: float            # K - 1 steps of dtau
    n_roots: int = 0
    n_merged: int = 0                # bins removed by the min_cells merging
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("bin_of_cell", "bin_times", "counts", "edges"):
            setattr(self, name, np.asarray(getattr(self, name)))

    def masses(self) -> np.ndarray:
        """Normalised mass of each bin (its share of the cells)."""
        n = self.counts.sum()
        return self.counts / n if n > 0 else self.counts.astype(float)

    @property
    def cell_time(self) -> np.ndarray:
        """Model time of each cell (the ``vect_t`` of the CardamomOT cycle)."""
        return self.bin_times[self.bin_of_cell]

    def as_obs(self, time_key: str = "pseudotime_bin",
               index_key: str = "pseudotime_bin_index") -> dict:
        """obs columns holding the bin time and the bin index of each cell."""
        return {time_key: self.cell_time, index_key: self.bin_of_cell.copy()}

    def summary(self) -> dict:
        return dict(n_bins=int(self.n_bins), dtau=float(self.dtau),
                    total_duration=float(self.total_duration), rule=self.rule,
                    n_roots=int(self.n_roots), counts=[int(c) for c in self.counts],
                    masses=[round(float(m), 4) for m in self.masses()],
                    n_merged=int(self.n_merged),
                    edges=[round(float(e), 4) for e in self.edges])


def tau_from_time(times: Sequence[float]) -> np.ndarray:
    """Pseudotime from known times (oracle validation): 0 at the first time, 1 at the last."""
    t = np.asarray(times, dtype=float).ravel()
    if t.size == 0:
        return t
    t = t - np.nanmin(t)
    m = np.nanmax(t)
    return t / m if m > 0 else t


def _root_mask(tau: np.ndarray, roots: Optional[Union[np.ndarray, Sequence[int]]],
               tol: float) -> np.ndarray:
    """Boolean mask of the root cells (``roots`` as a mask or as indices; else the smallest tau)."""
    if roots is None:
        return tau <= tau.min() + tol
    roots = np.asarray(roots)
    if roots.dtype == bool:
        if roots.size != tau.size:
            raise ValueError(f"{roots.size} root flags for {tau.size} cells")
        return roots.copy()
    mask = np.zeros(tau.size, dtype=bool)
    mask[roots.astype(int)] = True
    return mask


def _merge_small_bins(labels: np.ndarray, min_cells: int) -> tuple[np.ndarray, int]:
    """
    Relabel the non-root bins (labels >= 1) so that each holds at least ``min_cells`` cells:
    greedy merge with the following bins; a trailing remainder joins the last group.  Returns the
    new labels (0 = roots unchanged, others in 1..K-1, contiguous) and the number of bins removed.
    """
    values = np.unique(labels)
    order = [int(v) for v in values if v > 0]
    if not order:
        raise ValueError("no non-root cell: the pseudotime has a single value")

    groups, current, size = [], [], 0
    for v in order:
        n = int(np.sum(labels == v))
        current.append(v)
        size += n
        if size >= min_cells:
            groups.append(current)
            current, size = [], 0
    if current:
        if groups:                 # too small alone: attached to the previous group
            groups[-1].extend(current)
        else:
            groups.append(current)

    new = np.zeros(labels.size, dtype=int)
    for k, group in enumerate(groups, start=1):
        new[np.isin(labels, group)] = k
    return new, max(len(order) - len(groups), 0)


def bin_from_tau(tau: Sequence[float], n_bins: int = 10,
                 roots: Optional[Union[np.ndarray, Sequence[int]]] = None,
                 rule: str = "equal_width", min_cells: int = 50,
                 total_duration: Optional[float] = None, tol: float = 1e-12) -> BinScheme:
    """
    Assign the cells to ``n_bins`` pseudotime bins.

    Parameters
    ----------
    tau : (N,) pseudotime, non-negative, 0 for the root cells (a distance or a step count: the
        values are divided by their maximum, which keeps the *relative* distances — and therefore
        the density of cells along the pseudotime, which the growth pass reads).
    n_bins : number of bins K, root bin included (K >= 2).
    roots : boolean mask or indices of the root cells; default: the cells with the smallest tau.
    rule : 'equal_width' (constant tau width, cells per bin proportional to the density) or
        'equal_count' (quantile edges, constant number of cells: growth cannot be read, ablation).
    min_cells : bins with fewer cells are merged with their successor.
    total_duration : duration of the pseudotime axis in model time units (hours); None or 1.0
        means the axis is normalised to [0, 1] and ``dtau = 1 / (K - 1)``.

    Returns
    -------
    BinScheme with ``bin_of_cell`` (0 = roots, 1..K-1), ``bin_times`` (k * dtau, model time) and
    the diagnostics.
    """
    tau = np.asarray(tau, dtype=float).ravel()
    if tau.size == 0:
        raise ValueError("empty pseudotime")
    if not np.all(np.isfinite(tau)):
        raise ValueError("the pseudotime has non-finite values")
    if tau.min() < -tol:
        raise ValueError("the pseudotime must be non-negative (it is a distance or a step count)")
    if rule not in BIN_RULES:
        raise ValueError(f"rule must be one of {BIN_RULES}, got {rule!r}")
    n_bins = int(n_bins)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2 (the root bin and one other)")

    root_mask = _root_mask(tau, roots, tol)
    n_roots = int(root_mask.sum())
    if not n_roots:
        raise ValueError("no root cell")
    idx = np.flatnonzero(~root_mask)
    if idx.size == 0:
        raise ValueError("every cell is a root: nothing to order")

    # Normalise by the maximum (not by the rank): the relative distances, hence the density, are kept
    t_max = float(tau[idx].max())
    if not t_max > tol:
        raise ValueError("all the non-root cells have the same pseudotime: no bin can be built")
    x = tau[idx] / t_max

    # Right edges of the K-1 non-root bins, in normalised tau
    if rule == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins)[1:]
    else:
        edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins)[1:])
        edges = np.maximum.accumulate(np.maximum(edges, 1e-12))
        edges[-1] = max(edges[-1], edges[-2] * (1 + 1e-9), 1e-12)

    # searchsorted 'right': number of edges <= x -> the index of the bin holding x
    labels = np.zeros(tau.size, dtype=int)
    labels[idx] = np.clip(np.searchsorted(edges, x, side="right") + 1, 1, n_bins - 1)

    labels, n_merged = _merge_small_bins(labels, max(int(min_cells), 1))
    n_used = int(labels.max()) + 1

    # Upper edge of each merged bin (normalised tau), for the diagnostics: max tau of the bin
    upper = np.zeros(n_used)
    for k in range(1, n_used):
        m = labels == k
        if m.any():
            upper[k] = float(np.max(tau[m]) / t_max)
    upper[0] = 0.0

    duration = 1.0 if total_duration is None else float(total_duration)
    if not duration > 0:
        raise ValueError("total_duration must be positive")
    dtau = duration / (n_used - 1)

    return BinScheme(
        bin_of_cell=labels,
        bin_times=np.arange(n_used) * dtau,
        n_bins=n_used,
        dtau=dtau,
        counts=np.bincount(labels, minlength=n_used),
        edges=upper,
        rule=rule,
        total_duration=duration,
        n_roots=n_roots,
        n_merged=n_merged,
        info=dict(tau_max=t_max, min_cells=int(min_cells)),
    )


# ----------------------------------------------------------------------------------------------
# Prototype (PROTOTYPE_PLAN.md §2.3–§2.5): bins sized by the roots and the growth, cut by rank
# ----------------------------------------------------------------------------------------------
BY_SIZE_RULE = "by_size"


def growth_sizes(n_roots: int, n_cells: int, gaps: Optional[Sequence[float]] = None,
                 rates: Optional[Sequence[float]] = None, n_min: int = 50, k_min: int = 4,
                 k_max: int = 30) -> np.ndarray:
    r"""
    Number of cells of each bin: ``n_0 = n_roots`` and :math:`n_{k+1} = n_k \exp(\bar R_k \Delta t_k)`.

    Parameters
    ----------
    n_roots : n_0, the root bin (kept exactly).
    n_cells : N, total number of cells; the sizes sum to N.
    gaps, rates : (K_prev - 1,) gaps Δt_k and mean net rates R̄_k of the previous iteration (the
        growth factor of the step leaving bin k).  None, or missing entries beyond their length,
        mean no growth (the last available factor is reused when ``gaps`` and ``rates`` are given).
    n_min, k_min, k_max : guards — every non-root bin has at least ``n_min`` cells (when possible)
        and the number of bins K is in ``[k_min, k_max]``.

    The non-root sizes follow the growth sequence :math:`s_k = n_0\prod_{j<k} f_j` up to one common
    factor, so that they sum to ``N - n_0`` exactly: the growth *ratios* are kept when a guard binds.
    """
    n_roots, n_cells = int(n_roots), int(n_cells)
    if n_roots < 1:
        raise ValueError("at least one root cell is needed")
    rest = n_cells - n_roots
    if rest < 1:
        raise ValueError("every cell is a root: nothing to order")
    k_min, k_max = max(int(k_min), 2), max(int(k_max), 2)
    if k_min > k_max:
        raise ValueError("k_min must not exceed k_max")

    factors = np.ones(0)
    if gaps is not None and rates is not None:
        g = np.asarray(gaps, dtype=float).ravel()
        r = np.asarray(rates, dtype=float).ravel()
        m = min(g.size, r.size)
        factors = np.exp(np.nan_to_num(r[:m]) * g[:m])

    def sequence(m: int) -> np.ndarray:
        """Unscaled sizes s_1..s_m of the non-root bins."""
        f = np.ones(m)
        if factors.size:
            f = np.array([factors[min(k, factors.size - 1)] for k in range(m)])
        return n_roots * np.cumprod(f)

    # natural number of non-root bins: the growth sequence that best covers N - n_0
    m, total = 0, 0.0
    while total < rest and m < 10 * k_max:
        m += 1
        total = float(sequence(m).sum())
    if m > 1 and abs(float(sequence(m - 1).sum()) - rest) < abs(total - rest):
        m -= 1
    m = int(np.clip(m, k_min - 1, k_max - 1))
    m = min(m, rest)                                   # at least one cell per bin

    while True:
        s = sequence(m)
        bounds = np.round(np.cumsum(s) * rest / s.sum()).astype(int)
        bounds[-1] = rest
        sizes = np.diff(np.concatenate([[0], bounds]))
        if m <= max(k_min - 1, 1) or sizes.min() >= min(int(n_min), rest):
            break
        m -= 1                                         # a bin below n_min: one bin fewer
    if sizes.min() < 1:
        raise ValueError("a bin would be empty: too many bins for the cells")
    return np.concatenate([[n_roots], sizes]).astype(int)


def bin_by_sizes(tau: Sequence[float], sizes: Sequence[int],
                 roots: Union[np.ndarray, Sequence[int]],
                 bin_times: Optional[Sequence[float]] = None) -> BinScheme:
    """
    Cut the cells into bins of the given ``sizes`` by **rank** of ``tau``.

    Bin 0 is the root population (``sizes[0]`` must equal the number of roots); the other cells
    are sorted by ``tau`` (ties broken by cell index) and cut into consecutive blocks of
    ``sizes[1:]``.  Only the order of ``tau`` is used, not its values.

    ``bin_times`` (K,) gives the time of each bin (default: ``0, 1, …, K-1``); they may be unequally
    spaced and must be increasing.
    """
    tau = np.asarray(tau, dtype=float).ravel()
    sizes = np.asarray(sizes, dtype=int).ravel()
    root_mask = _root_mask(tau, roots, 1e-12)
    n_roots = int(root_mask.sum())
    if sizes.size < 2:
        raise ValueError("at least two bins (the roots and one other)")
    if sizes[0] != n_roots:
        raise ValueError(f"sizes[0] = {sizes[0]} but there are {n_roots} roots")
    if int(sizes.sum()) != tau.size:
        raise ValueError(f"the sizes sum to {int(sizes.sum())} for {tau.size} cells")
    if np.any(sizes[1:] < 1):
        raise ValueError("empty bin")

    idx = np.flatnonzero(~root_mask)
    order = idx[np.lexsort((idx, tau[idx]))]
    labels = np.zeros(tau.size, dtype=int)
    bounds = np.cumsum(sizes[1:])
    for k, (lo, hi) in enumerate(zip(np.concatenate([[0], bounds[:-1]]), bounds), start=1):
        labels[order[lo:hi]] = k

    K = int(sizes.size)
    times = np.arange(K, dtype=float) if bin_times is None else np.asarray(bin_times, dtype=float)
    if times.size != K:
        raise ValueError(f"{times.size} bin times for {K} bins")
    if np.any(np.diff(times) <= 0):
        raise ValueError("the bin times must be increasing")
    total = float(times[-1] - times[0])
    rank_edges = np.concatenate([[0.0], bounds / max(idx.size, 1)])
    return BinScheme(
        bin_of_cell=labels, bin_times=times, n_bins=K, dtau=total / (K - 1),
        counts=np.bincount(labels, minlength=K), edges=rank_edges, rule=BY_SIZE_RULE,
        total_duration=total, n_roots=n_roots, n_merged=0,
        info=dict(sizes=[int(s) for s in sizes]),
    )


def cell_times(scheme: BinScheme) -> np.ndarray:
    """Time of each cell: the time of its bin (PROTOTYPE_PLAN.md §2.4)."""
    return np.asarray(scheme.bin_times, dtype=float)[np.asarray(scheme.bin_of_cell)]


def gaps_from_cell_times(bin_of_cell: Sequence[int], cell_time: Sequence[float],
                         min_gap: float) -> np.ndarray:
    """
    Starting gaps of new bins from the time of each cell at the previous iteration
    (PROTOTYPE_PLAN.md §2.4): the time of a new bin is the median old time of its cells (0 for the
    root bin), and the gaps are the differences of these times, floored at ``min_gap`` (a re-ordering
    can make two medians equal or decreasing).
    """
    labels = np.asarray(bin_of_cell, dtype=int)
    t = np.asarray(cell_time, dtype=float)
    if labels.size != t.size:
        raise ValueError("one time per cell is needed")
    K = int(labels.max()) + 1
    med = np.zeros(K)
    for k in range(1, K):
        m = labels == k
        med[k] = float(np.median(t[m])) if m.any() else np.nan
    # empty bins: interpolate the medians of their neighbours
    bad = ~np.isfinite(med)
    if bad.any():
        good = np.flatnonzero(~bad)
        med[bad] = np.interp(np.flatnonzero(bad), good, med[good])
    return np.maximum(np.diff(med), float(min_gap))
