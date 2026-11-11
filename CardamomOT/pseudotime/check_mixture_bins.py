#!/usr/bin/env python3
r"""
Check the NB mixture fitted on pseudotime bins against the pooled mixture.

Fits, on the same simulated dataset (``obs['time']`` = true time, used here as an *oracle*
pseudotime):

* ``pooled`` : one mixture over all the cells (one group, no time label);
* ``binned`` : ``PseudotimeNetworkModel.fit_mixture_on_bins``, the bins of the true time;

and measures the three gains expected in ``PLAN.md`` §10.6:

(a) **the mode masses evolve along the axis** :math:`\nu_g(k)` (the target marginal of the basin
    EMD) and match each bin's mean (the mean-preservation constraint becomes local);
(b) **the mode amplitudes span the observed range** (``_apply_temporal_constraints`` calibrates the
    extreme modes on the extreme bins);
(c) **the basin labels are distribution-matched per bin** (a different share of the cells than the
    pooled argmax).

Usage
-----
    conda run -n cardamom_env python CardamomOT/pseudotime/check_mixture_bins.py \
        --networks CN5 BN8 FN8 --runs 1 --bins 10

Writes ``<outdir>/mixture_bins_<NET>_r<run>.png`` and ``.json`` and prints one line per dataset.
The ``force_basins`` ramp of the plan (0.3 -> 1.0) only acts inside the EM loop, not here: this
script validates the *mixture* step alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

if __package__ in (None, ""):   # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CardamomOT.pseudotime import PseudotimeNetworkModel, tau_from_time

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data"
DEFAULT_OUT = HERE / "data" / "_mixture_bins"


def load_counts(path: Path):
    """(AnnData, counts as float array)."""
    import anndata as ad
    data = ad.read_h5ad(path)
    X = data.X.toarray() if sp.issparse(data.X) else np.asarray(data.X)
    return data, np.asarray(X, dtype=float)


def oracle_tau(data, root_quantile: float):
    """Oracle pseudotime (= normalised true time) and the root mask (earliest ``root_quantile``)."""
    t = np.asarray(data.obs["time"].values, dtype=float)
    roots = t <= np.quantile(t, root_quantile)
    tau = tau_from_time(t)
    tau[roots] = 0.0                      # the root population sits exactly at tau = 0
    return tau, roots


def _spearman_columns(y: np.ndarray) -> np.ndarray:
    """Spearman correlation of each column of y (K, G) with the row index (0..K-1)."""
    x = np.arange(y.shape[0])
    out = np.zeros(y.shape[1])
    for g in range(y.shape[1]):
        col = y[:, g]
        out[g] = 0.0 if np.ptp(col) <= 1e-12 else float(spearmanr(x, col).statistic)
    return out


def analyse(pooled, binned, X: np.ndarray, ns: int = 1) -> dict:
    """Metrics of the binned fit against the pooled one (X: (N, G) counts, gene columns only)."""
    labels = binned.bins.bin_of_cell
    K = binned.bins.n_bins
    G = X.shape[1]

    # --- (a) per-bin masses vs the observed mean of the bin (mean-preservation constraint) ----
    obs_mean = np.stack([X[labels == k].mean(axis=0) if np.any(labels == k) else np.zeros(G)
                         for k in range(K)])
    implied = np.asarray(binned.bin_expected_means())[:, ns:]
    mean_err = np.abs(implied - obs_mean) / np.maximum(obs_mean, 1.0)

    nu_b = np.asarray(binned.bin_masses())[:, ns:, :]      # (K, G, M)
    nu_p = np.asarray(pooled.mixture_masses())[ns:, :]     # (G, M_p)
    hi = nu_b[:, :, -1]                                     # highest mode of each bin
    rho = _spearman_columns(hi)

    # --- (b) range of the mode amplitudes vs the range observed along the axis -----------------
    c = np.asarray(binned.a[-1])[ns:]
    ks_p, ks_b = np.asarray(pooled.mode_amplitudes())[:, ns:], np.asarray(binned.mode_amplitudes())[:, ns:]
    range_p = ks_p.max(axis=0) - ks_p.min(axis=0)
    range_b = ks_b.max(axis=0) - ks_b.min(axis=0)
    observed_range = c * (obs_mean.max(axis=0) - obs_mean.min(axis=0))
    bimodal_p, bimodal_b = range_p > 1e-12, range_b > 1e-12
    both = bimodal_p & bimodal_b

    # --- (c) share of the cells whose basin differs from the pooled argmax ---------------------
    # measured on the genes bimodal in the binned fit: those the pooled fit collapsed to a single
    # mode (range 0) count as fully reassigned, which is the strongest gain of the binned fit
    pooled_basins = np.asarray(pooled.basins)[:, ns:]
    binned_basins = np.asarray(binned.basins)[:, ns:]
    label_change = (float(np.mean(pooled_basins[:, bimodal_b] != binned_basins[:, bimodal_b]))
                    if bimodal_b.any() else 0.0)

    return dict(
        n_bins=int(K), counts=[int(v) for v in binned.bins.counts],
        masses=[round(float(v), 4) for v in binned.bins.masses()],
        dtau=float(binned.bins.dtau),
        mean_error_max=float(np.nanmax(mean_err)),
        mean_error_p90=float(np.nanpercentile(mean_err, 90)),
        mean_error_median=float(np.nanmedian(mean_err)),
        trend_rho=rho.round(4).tolist(),
        n_genes_trend=int(np.sum(np.abs(rho) >= 0.8)),
        n_genes=int(G),
        n_genes_bimodal_pooled=int(bimodal_p.sum()),
        n_genes_bimodal_binned=int(bimodal_b.sum()),
        range_pooled=range_p.round(3).tolist(), range_binned=range_b.round(3).tolist(),
        range_observed=observed_range.round(3).tolist(),
        range_ratio_median_both=float(np.median(range_b[both] / range_p[both])) if both.any() else float("nan"),
        label_change=label_change,
        obs_mean=obs_mean.round(3), implied_mean=implied.round(3),
        nu_high_binned=hi.round(4), nu_high_pooled=nu_p[:, -1].round(4),
        gene_names=[str(g) for g in range(G)],
    )


def plot(metrics: dict, title: str, path: Path, n_genes: int = 4) -> None:
    """(A) per-bin high-mode masses, (B) observed vs implied mean, (C) mode ranges pooled/binned."""
    hi, rho = np.asarray(metrics["nu_high_binned"]), np.abs(np.asarray(metrics["trend_rho"]))
    order = np.argsort(-rho)[:n_genes]
    K = hi.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))

    ax = axes[0]
    for g in order:
        ax.plot(np.arange(K), hi[:, g], marker="o", label=f"G{g + 1} ($\\rho$={metrics['trend_rho'][g]:.2f})")
        ax.axhline(metrics["nu_high_pooled"][g], ls="--", lw=1, color=ax.lines[-1].get_color(), alpha=.6)
    ax.set_xlabel("pseudotime bin"); ax.set_ylabel("mass of the highest mode")
    ax.set_title("(A) per-bin mode masses (dashed: pooled)")
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    ax = axes[1]
    obs, imp = np.asarray(metrics["obs_mean"]), np.asarray(metrics["implied_mean"])
    for g in order[:2]:
        ax.plot(np.arange(K), obs[:, g], "o-", label=f"G{g + 1} observed")
        ax.plot(np.arange(K), imp[:, g], "x--", label=f"G{g + 1} mixture")
    ax.set_xlabel("pseudotime bin"); ax.set_ylabel("mean counts")
    ax.set_title("(B) mean-preservation per bin")
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    ax = axes[2]
    rp, rb, ro = (np.asarray(metrics[k]) for k in ("range_pooled", "range_binned", "range_observed"))
    ax.scatter(rp, rb, s=30, label="binned")
    ax.scatter(rp, ro, s=18, marker="^", alpha=.6, label="observed")
    lim = max(rp.max(), rb.max(), ro.max()) * 1.05 + 1e-9
    ax.plot([0, lim], [0, lim], "k-", lw=.8, alpha=.5)
    ax.set_xlabel("mode range, pooled"); ax.set_ylabel("mode range")
    ax.set_title("(B') dynamic range of the modes")
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_one(data_path: Path, outdir: Path, n_bins: int, rule: str, min_cells: int,
            root_quantile: float, n_plot: int, seed: int = 0, verb: bool = True) -> dict:
    data, X = load_counts(data_path)
    tau, roots = oracle_tau(data, root_quantile)
    G = X.shape[1]
    kw = dict(gene_names=[str(g) for g in data.var_names],
              min_components=2, max_components=2, max_iter_kinetics=0)

    pooled = PseudotimeNetworkModel(n_genes=G)
    pooled.seed = seed                     # the mixture EM is unseeded by default
    pooled.fit_mixture_pooled(data, verb=verb, **kw)

    binned = PseudotimeNetworkModel(n_genes=G)
    binned.seed = seed
    binned.fit_mixture_on_bins(data, tau=tau, roots=roots, n_bins=n_bins, rule=rule,
                               min_cells=min_cells, verb=verb, **kw)

    metrics = analyse(pooled, binned, X[:, :], ns=1)
    metrics.update(dataset=str(data_path), rule=rule, root_quantile=float(root_quantile),
                   bins_requested=int(n_bins), seed=int(seed))

    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"mixture_bins_{data_path.parent.parent.name}_r{data_path.stem.split('_')[-1]}"
    plot(metrics, f"{data_path.parent.parent.name} (oracle pseudotime, {metrics['n_bins']} bins)",
         outdir / f"{stem}.png", n_genes=n_plot)
    with open(outdir / f"{stem}.json", "w") as fh:
        json.dump({k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}, fh, indent=2)

    if verb:
        print(f"  {stem}: bins={metrics['n_bins']} counts={metrics['counts']} "
              f"| mean err (median/p90) {metrics['mean_error_median']:.3f}/{metrics['mean_error_p90']:.3f} "
              f"| trend {metrics['n_genes_trend']}/{metrics['n_genes']} genes "
              f"| bimodal pooled/binned {metrics['n_genes_bimodal_pooled']}/{metrics['n_genes_bimodal_binned']} "
              f"| labels changed {100 * metrics['label_change']:.0f}%")
    return metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--networks", nargs="+", default=["CN5", "BN8", "FN8"])
    p.add_argument("--runs", nargs="+", type=int, default=[1], help="run indices of the datasets")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--bins", type=int, default=10, help="number of pseudotime bins")
    p.add_argument("--rule", choices=("equal_width", "equal_count"), default="equal_width")
    p.add_argument("--min-cells", type=int, default=50, help="bins with fewer cells are merged")
    p.add_argument("--root-quantile", type=float, default=0.02,
                   help="oracle roots: cells with a true time below this quantile")
    p.add_argument("--n-plot", type=int, default=4, help="genes drawn in the figure")
    p.add_argument("--seed", type=int, default=0, help="seed of the mixture fits (unseeded by default)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"Mixture on bins vs pooled mixture ({args.bins} bins, {args.rule}, "
          f"roots = first {100 * args.root_quantile:.1f}% of the true time)")
    for net in args.networks:
        for run in args.runs:
            path = args.data_root / net / "Data" / f"data_{run}.h5ad"
            if not path.exists():
                print(f"  missing {path}, skipped")
                continue
            print(f" * {net} run {run}")
            run_one(path, args.outdir, args.bins, args.rule, args.min_cells,
                    args.root_quantile, args.n_plot, seed=args.seed, verb=not args.quiet)
    print(f"Figures and metrics in {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
