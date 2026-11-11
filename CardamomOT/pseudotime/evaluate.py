"""
Metrics of the pseudotime inference (``PLAN.md`` §7).

* pseudotime vs the true sampling time (the ground truth of the exercise):
  Spearman / Kendall, globally and **within each branch** (a single scalar τ conflates the two
  attractors of a bifurcating network such as BN8, so the within-branch rank is the meaningful one);
* network vs the ground-truth graph: AUPRC, AUROC, top-k precision, sign accuracy.

The ``evaluate`` entry point collects both from a fitted ``PseudotimeNetworkModel``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import kendalltau, spearmanr


def _valid(*arrays) -> np.ndarray:
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        ok &= np.isfinite(np.asarray(a, dtype=float))
    return ok


def spearman(x, y) -> float:
    """Spearman correlation (0.0 when undefined)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = _valid(x, y)
    if ok.sum() < 3 or np.ptp(x[ok]) <= 1e-12 or np.ptp(y[ok]) <= 1e-12:
        return 0.0
    return float(spearmanr(x[ok], y[ok]).statistic)


def kendall(x, y) -> float:
    """Kendall tau (0.0 when undefined)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = _valid(x, y)
    if ok.sum() < 3 or np.ptp(x[ok]) <= 1e-12 or np.ptp(y[ok]) <= 1e-12:
        return 0.0
    return float(kendalltau(x[ok], y[ok]).statistic)


def branch_labels(counts, late_mask, n_branches: int = 2, seed: int = 0) -> np.ndarray:
    """
    Attractor of each cell: k-means on the standardised log1p counts of the late cells;
    -1 for the other cells (no branch assigned).
    """
    from sklearn.cluster import KMeans

    counts = np.asarray(counts, dtype=float)
    labels = np.full(counts.shape[0], -1, dtype=int)
    late = np.flatnonzero(np.asarray(late_mask, dtype=bool))
    if late.size < max(3 * n_branches, 10):
        return labels
    Z = np.log1p(counts[late])
    Z = (Z - Z.mean(axis=0)) / np.maximum(Z.std(axis=0), 1e-12)
    labels[late] = KMeans(n_clusters=int(n_branches), n_init=10,
                          random_state=seed).fit_predict(Z)
    return labels


def pseudotime_metrics(tau, true_time, counts=None, n_branches: int = 2,
                       late_quantile: float = 0.8, transient_quantile: float = 0.5,
                       seed: int = 0) -> dict:
    """
    Rank agreement of the inferred pseudotime with the true sampling time.

    ``counts`` given (and ``n_branches >= 2``): also the Spearman inside each branch, the late
    cells (``true_time >= late_quantile`` quantile) being clustered into ``n_branches`` attractors.

    ``spearman_transient``: the same on the cells of the first ``transient_quantile`` of the true
    time.  Cells that have reached the steady state are (nearly) identical, so **no** pseudotime can
    order them: on the simulated datasets the global Spearman is limited by that saturation
    (0.6 while the transient alone reaches 0.7-0.8, and the diffusion pseudotime baseline is
    below 0.5), so both numbers are reported.
    """
    tau = np.asarray(tau, dtype=float)
    t = np.asarray(true_time, dtype=float)
    ok = _valid(tau, t)
    out = dict(n_cells=int(ok.sum()), spearman=spearman(tau[ok], t[ok]),
               kendall=kendall(tau[ok], t[ok]))
    if transient_quantile is not None and ok.any():
        early = ok & (t <= np.quantile(t[ok], float(transient_quantile)))
        out.update(spearman_transient=spearman(tau[early], t[early]),
                   n_cells_transient=int(early.sum()))
    if counts is not None and n_branches >= 2 and ok.sum() > 10:
        late = t >= np.quantile(t[ok], late_quantile)
        labels = branch_labels(np.asarray(counts), late, n_branches, seed=seed)
        per_branch, sizes = [], []
        for b in range(n_branches):
            m = ok & (labels == b)
            sizes.append(int(m.sum()))
            if m.sum() > 10:
                per_branch.append(spearman(tau[m], t[m]))
        out["branch_sizes"] = sizes
        out["spearman_per_branch"] = per_branch
        out["spearman_branch_mean"] = float(np.mean(per_branch)) if per_branch else float("nan")
    return out


def grn_metrics(inter, true_inter, ns: int = 1, exclude_diagonal: bool = True,
                topk: Optional[int] = None) -> dict:
    """AUPRC / AUROC / top-k precision / sign accuracy of ``inter`` against a binary truth."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    score = np.asarray(inter, dtype=float)
    truth = np.asarray(true_inter, dtype=float)
    if score.ndim == 3:                      # per-sample networks: first sample
        score = score[0]
    if truth.ndim == 3:
        truth = truth[0]
    score, truth = score[ns:, ns:], truth[ns:, ns:]
    if exclude_diagonal:
        off = ~np.eye(score.shape[0], dtype=bool)
        score, truth = score[off], truth[off]

    positive = np.abs(truth) > 0
    flat = np.abs(score).ravel()
    pos = positive.ravel()
    out = dict(n_edges=int(pos.sum()), n_pairs=int(pos.size))
    if pos.sum() == 0 or pos.all():
        out.update(auprc=float("nan"), auroc=float("nan"), topk_precision=float("nan"),
                   sign_accuracy=float("nan"))
        return out
    out["auprc"] = float(average_precision_score(pos, flat))
    out["auroc"] = float(roc_auc_score(pos, flat))
    k = int(topk if topk is not None else pos.sum())
    order = np.argsort(-flat)[:k]
    out["topk_precision"] = float(pos[order].mean())
    out["sign_accuracy"] = float((np.sign(score.ravel()[pos]) == np.sign(truth.ravel()[pos])).mean())
    return out


def marginal_agreement(model, data) -> dict:
    """
    Comparison of the **simulated** mRNA marginals with the observed ones (the M5 gate of
    ``PLAN.md`` §8), per pseudotime bin and per gene.

    Needs ``model.simulate_network(...)`` to have run (it sets ``kon_theta``).  The expected count of
    a simulated state is ``kon * kz / c`` with ``kz = sigmoid_amplitudes(ks).max(mode) * k1`` and
    ``c`` the overdispersion — the mean of ``negative_binomial(kz*kon, c/(c+1))``, i.e. the same
    formula the simulation uses (``simulate_trajectories_unitary``) and the figure code
    (``_prot_to_adata``).

    Returns ``per_bin`` / ``per_gene`` ratios and ``frac_within_2x``.
    """
    from ..inference.network import sigmoid_amplitudes

    ns = int(model.n_stimuli)
    ks, _, k1 = model.mixture_terms()
    ks = ks[0] if np.ndim(ks) == 3 else ks
    k1 = k1[0] if np.ndim(k1) == 2 else k1
    c = np.asarray(model.a)[-1]
    kz = (sigmoid_amplitudes(ks) * np.asarray(k1)[:, None]).max(axis=1)
    expected = np.asarray(model.kon_theta)[:, ns:] * kz[ns:][None, :] / c[ns:][None, :]

    X = data.X.toarray() if hasattr(data.X, "toarray") else np.asarray(data.X)
    if X.shape[1] != expected.shape[1]:
        raise ValueError(f"{X.shape[1]} genes in the data, {expected.shape[1]} in the model")
    bin_of_cell = np.asarray(model.bins.bin_of_cell)
    bin_time = np.asarray(model.bins.bin_times)[bin_of_cell]

    bins = np.sort(np.unique(np.asarray(model.times_data)))
    per_bin = []
    for b in bins:
        sim = expected[np.asarray(model.times_data) == b].mean()
        obs = X[bin_time == b].mean()
        per_bin.append(dict(bin=float(b), n_cells=int((bin_time == b).sum()),
                            observed=float(obs), simulated=float(sim),
                            ratio=float(sim / obs) if obs > 0 else np.nan))
    sim_g, obs_g = expected.mean(axis=0), X.mean(axis=0)
    per_gene = dict(observed=obs_g.tolist(), simulated=sim_g.tolist(),
                    ratio=np.where(obs_g > 0, sim_g / np.maximum(obs_g, 1e-12), np.nan).tolist())
    ratios = np.array([r["ratio"] for r in per_bin] + per_gene["ratio"], dtype=float)
    ratios = ratios[np.isfinite(ratios)]
    return dict(per_bin=per_bin, per_gene=per_gene,
                frac_within_2x=float(((ratios > 0.5) & (ratios < 2.0)).mean()) if ratios.size else np.nan,
                ratio_min=float(ratios.min()) if ratios.size else np.nan,
                ratio_max=float(ratios.max()) if ratios.size else np.nan)


def degradation_metrics(model, true_d) -> dict:
    """
    Recovery of the degradation rates by ``refine_network_degradations`` (``PLAN.md`` §10.1).

    ``true_d`` is the ``(2, G_tot)`` array of the simulations (``uns['true_d']``): row 0 = d0,
    row 1 = d1.  Only ``d1`` is refitted by the base code (``model.d_t[:, 1, :]``); the ratio
    ``d0 / d1`` is ``model.ratios``.
    """
    ns = int(model.n_stimuli)
    true_d = np.asarray(true_d, dtype=float)
    true_d1 = true_d[1, ns:]
    d1 = np.asarray(model.d_t, dtype=float)[:, 1, ns:].mean(axis=0)
    out = dict(d1_inferred=d1.tolist(), d1_true=true_d1.tolist(),
               d1_ratio=(d1 / np.maximum(true_d1, 1e-12)).tolist(),
               d1_within_2x=float(np.mean((d1 / np.maximum(true_d1, 1e-12) > 0.5)
                                          & (d1 / np.maximum(true_d1, 1e-12) < 2.0))))
    ratios = getattr(model, "ratios", None)
    if ratios is not None:
        true_ratio = true_d[0, ns:] / np.maximum(true_d1, 1e-12)
        got = np.asarray(ratios, dtype=float)[:, ns:].mean(axis=0)
        out.update(d0_d1_inferred=got.tolist(), d0_d1_true=true_ratio.tolist())
    return out


def snapshot_metrics(tau, true_time, n_max: int = 50) -> dict:
    """
    Metrics for a **snapshot** ground truth, i.e. when the cells come from a few library times
    (``PLAN.md`` §0.11).  With such a design the true time is a step function, so Spearman is
    tie-attenuated; what matters is:

    * ``order_accuracy``: fraction of snapshot pairs ``(i, j)``, ``t_i < t_j``, whose **median**
      :math:`\\tau` is also increasing — "are the libraries in the right order?";
    * ``separation``: fraction of pairs whose :math:`\\tau` **ranges do not overlap**
      (:math:`\\min \\tau_j > \\max \\tau_i`) — a strict, tie-free notion of a correct ordering;
    * ``gap_spearman`` / ``gap_ratio_*``: the inferred **spacing**.  With
      :math:`m_i` the median :math:`\\tau` of snapshot :math:`i`, the recovered gap ratio is
      :math:`\\frac{m_{i+1}-m_i}{\\sum_j (m_{j+1}-m_j)} \\big/ \\frac{t_{i+1}-t_i}{t_K-t_1}`.
      A method that only recovers the *order* reconciles the gaps with the ranks (ratios far from 1);
      a method that recovers the dynamics should return ratios near 1.

    Requires the number of distinct true times to be small (<= ``n_max``).
    """
    tau = np.asarray(tau, dtype=float).ravel()
    true_time = np.asarray(true_time, dtype=float).ravel()
    if tau.size != true_time.size:
        raise ValueError(f"{tau.size} pseudotimes for {true_time.size} times")
    ok = np.isfinite(tau) & np.isfinite(true_time)
    tau, true_time = tau[ok], true_time[ok]
    times = np.unique(true_time)
    if times.size < 3:
        raise ValueError("snapshot_metrics needs at least 3 distinct true times")
    if times.size > n_max:
        raise ValueError(f"{times.size} distinct true times (> {n_max}): that is not a snapshot "
                         f"design, use pseudotime_metrics instead")

    med = np.array([np.median(tau[true_time == t]) for t in times])
    lo = np.array([tau[true_time == t].min() for t in times])
    hi = np.array([tau[true_time == t].max() for t in times])

    n_pairs = n_ordered = n_split = 0
    for i in range(times.size):
        for j in range(i + 1, times.size):
            n_pairs += 1
            n_ordered += int(med[j] > med[i])
            n_split += int(lo[j] > hi[i])

    true_gaps = np.diff(times)
    tau_gaps = np.diff(med)
    out = dict(
        n_snapshots=int(times.size),
        order_accuracy=float(n_ordered / n_pairs),
        separation=float(n_split / n_pairs),
        n_pairs=int(n_pairs),
    )
    if np.all(tau_gaps > 0) and tau_gaps.sum() > 0:
        ratio = (tau_gaps / tau_gaps.sum()) / (true_gaps / true_gaps.sum())
        out.update(
            gap_spearman=spearman(tau_gaps, true_gaps),
            gap_ratio_min=float(ratio.min()),
            gap_ratio_max=float(ratio.max()),
            gap_ratio_within_2x=float(np.mean((ratio > 0.5) & (ratio < 2.0))),
            gap_ratio=ratio.tolist(),
        )
    else:
        out.update(gap_spearman=np.nan, gap_ratio_min=np.nan, gap_ratio_max=np.nan,
                   gap_ratio_within_2x=0.0, gap_ratio=[])
    return out


def evaluate(model, true_time=None, true_inter=None, n_branches: int = 2) -> dict:
    """Metrics of a fitted ``PseudotimeNetworkModel`` (pseudotime and/or network)."""
    out = {}
    tau = getattr(model, "tau", None)
    if true_time is not None and tau is not None:
        out["pseudotime"] = pseudotime_metrics(tau, true_time, counts=getattr(model, "counts", None),
                                               n_branches=n_branches)
    inter = getattr(model, "inter", None)
    if true_inter is not None and inter is not None:
        out["grn"] = grn_metrics(inter, true_inter, ns=model.n_stimuli)
    return out


__all__ = ["spearman", "kendall", "branch_labels", "pseudotime_metrics", "grn_metrics",
           "snapshot_metrics", "marginal_agreement", "degradation_metrics", "evaluate"]
