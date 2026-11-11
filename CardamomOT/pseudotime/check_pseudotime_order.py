#!/usr/bin/env python3
r"""
Check the ordering options on the simulated datasets (Stage M1 of ``PLAN.md`` §8).

For each dataset, the mixture is fitted **pooled** (no time label), the protein state of each cell is
initialised from its modes, and each edge-weight variant of Option 3 (mechanistic Palantir) computes a
pseudotime, which is scored against the **true** simulation time (the ground truth).  A second pass
re-fits the mixture on the bins of that pseudotime and recomputes it — the first iteration of the
joint cycle, with the mixture guard of ``PLAN.md`` §10.6.

Variants (``plan.tex`` §mech-palantir):

* ``geometric``    — kNN graph in the log1p expression space (Palantir without the GRN);
* ``plausibility`` — GRN-based transition plausibility (no protein state needed);
* ``mechanistic``  — bridge + network cross-entropy + geometric term.

Usage
-----
    conda run -n cardamom_env python CardamomOT/pseudotime/check_pseudotime_order.py \
        --networks CN5 BN8 FN8 --runs 1 --iterations 0 1

Writes ``<outdir>/pseudotime_order_<NET>_r<run>.png`` and ``.json``, and prints one line per variant.
CN5 is an oscillator (its ON-fraction is periodic), so its τ cannot be monotone along the true time:
judge the ordering on BN8/FN8 (see ``PLAN.md`` §0.1).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

if __package__ in (None, ""):   # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CardamomOT.pseudotime import PseudotimeNetworkModel
from CardamomOT.pseudotime.evaluate import pseudotime_metrics

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data"
DEFAULT_OUT = HERE / "data" / "_pseudotime_order"


def load_dataset(path: Path, root_quantile: float):
    """(AnnData, counts, true time, root mask) of a simulated dataset."""
    import anndata as ad
    data = ad.read_h5ad(path)
    X = data.X.toarray() if sp.issparse(data.X) else np.asarray(data.X)
    t = np.asarray(data.obs["time"].values, dtype=float)
    roots = t <= np.quantile(t, root_quantile)
    data.obs["is_root"] = roots
    return data, np.asarray(X, dtype=float), t, roots


def clone(template: PseudotimeNetworkModel, **params) -> PseudotimeNetworkModel:
    """Model with the mixture / cell states of ``template`` and new parameters (no re-fit)."""
    model = PseudotimeNetworkModel(n_genes=int(template.a.shape[-1]) - template.n_stimuli,
                                   n_stimuli=template.n_stimuli)
    model.seed = template.seed
    model.d = np.array(template.d, copy=True)
    model._stim_schedule = template._stim_schedule
    for attr in ("a", "modes", "proba_init", "basins", "pi_init", "pi_zinb",
                 "counts", "cell_prot", "roots", "tau", "bins"):
        value = getattr(template, attr, None)
        if isinstance(value, (list, dict)):
            value = copy.deepcopy(value)
        elif value is not None:
            value = np.array(value, copy=True)
        setattr(model, attr, value)
    for key, value in params.items():
        if not hasattr(model, key):
            raise AttributeError(f"unknown parameter {key!r}")
        setattr(model, key, value)
    return model


def dpt_baseline(X: np.ndarray, roots: np.ndarray, n_pcs: int = 5, n_neighbors: int = 30):
    """Diffusion pseudotime of scanpy (expression space, no mechanism) — the standard baseline."""
    import anndata as ad
    import scanpy as sc

    A = ad.AnnData(X=np.asarray(X, dtype=float))
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    sc.pp.pca(A, n_comps=int(min(n_pcs, X.shape[1] - 1)))
    sc.pp.neighbors(A, n_neighbors=int(n_neighbors))
    A.uns["iroot"] = int(np.flatnonzero(roots)[0])
    sc.tl.dpt(A)
    return np.asarray(A.obs["dpt_pseudotime"].values, dtype=float)


def run_dataset(data_path: Path, outdir: Path, weights, iterations, n_bins: int, min_cells: int,
                root_quantile: float, n_neighbors: int, n_waypoints: int, seed: int,
                with_dpt: bool = True, verb: bool = True) -> dict:
    data, X, true_time, roots = load_dataset(data_path, root_quantile)
    G = X.shape[1]
    kw = dict(gene_names=[str(g) for g in data.var_names], min_components=2, max_components=2,
              max_iter_kinetics=0)

    base = PseudotimeNetworkModel(n_genes=G)
    base.seed = seed
    base.fit_mixture_pooled(data, verb=verb, **kw)
    base.initialise_cell_states(data)

    results, models = {}, {}
    for weight in weights:
        model = clone(base, palantir_edge_weight=weight, palantir_n_neighbors=n_neighbors,
                      palantir_n_waypoints=n_waypoints)
        tau = model.update_pseudotime(n_iter=0, roots=roots, verb=verb)
        metrics = pseudotime_metrics(tau, true_time, counts=X)
        metrics.update(model.pseudotime_diagnostics)
        results[f"{weight}|iter0"] = metrics
        models[weight] = model

    if 1 in iterations:
        for weight in weights:
            model = clone(models[weight], palantir_edge_weight=weight,
                          palantir_n_neighbors=n_neighbors, palantir_n_waypoints=n_waypoints)
            model.fit_mixture_on_bins(data, tau=model.tau, roots=roots, n_bins=n_bins,
                                      min_cells=min_cells, verb=False, n_iter=1, **kw)
            model.initialise_cell_states(data)
            tau = model.update_pseudotime(n_iter=1, roots=roots, verb=verb)
            metrics = pseudotime_metrics(tau, true_time, counts=X)
            metrics.update(model.pseudotime_diagnostics)
            results[f"{weight}|iter1"] = metrics
            models[weight] = model

    if with_dpt:
        try:
            tau_dpt = dpt_baseline(X, roots, n_pcs=5, n_neighbors=n_neighbors)
            metrics = pseudotime_metrics(tau_dpt, true_time, counts=X)
            metrics.update(n_unreached=int(np.sum(~np.isfinite(tau_dpt))), n_neighbors=n_neighbors)
            results["dpt|base"] = metrics
            models["dpt"] = type("Baseline", (), {"tau": tau_dpt})()
        except Exception as exc:      # scanpy unavailable, or too few genes for a PCA
            print(f"   (DPT baseline skipped: {exc})")

    stem = f"pseudotime_order_{data_path.parent.parent.name}_r{data_path.stem.split('_')[-1]}"
    panel_keys = [f"{w}|iter{max(iterations)}" for w in weights if f"{w}|iter{max(iterations)}" in results]
    if "dpt|base" in results:
        panel_keys.append("dpt|base")
    plot(results, models, true_time, panel_keys,
         f"{data_path.parent.parent.name} ({n_bins} bins, seed {seed})", outdir / f"{stem}.png")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{stem}.json", "w") as fh:
        json.dump({"dataset": str(data_path), "results": results}, fh, indent=2)

    if verb:
        for key, m in results.items():
            print(f"   {key:22s} spearman {m['spearman']:+.3f} "
                  f"| transient {m.get('spearman_transient', float('nan')):+.3f} "
                  f"| branch {m.get('spearman_branch_mean', float('nan')):+.3f} "
                  f"| unreached {m.get('n_unreached', 0)}")
    return dict(stem=stem, results=results)


def plot(results: dict, models: dict, true_time, keys, title: str, path: Path) -> None:
    """One scatter of tau vs the true time per method, then the summary bars of all the runs."""
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(3.6 * (len(keys) + 1), 3.6))

    for ax, key in zip(np.atleast_1d(axes)[:-1], keys):
        model = models[key.split("|")[0]] if "|" in key else models.get(key)
        metrics = results[key]
        ax.scatter(true_time, np.asarray(model.tau, dtype=float), s=4, alpha=.5)
        ax.set_xlabel("true time"); ax.set_ylabel(f"pseudotime ({key})")
        ax.set_title(f"{key}: $\\rho$={metrics['spearman']:+.2f}, "
                     f"transient {metrics.get('spearman_transient', float('nan')):+.2f}", fontsize=9)
        ax.grid(alpha=.3)

    ax = np.atleast_1d(axes)[-1]
    labels = [k.replace("|", "\n") for k in results]
    values = [results[k]["spearman"] for k in results]
    colors = ["C3" if k.startswith("dpt") else "C0" for k in results]
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values))); ax.set_xticklabels(labels, fontsize=6)
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("Spearman(tau, true time)")
    ax.set_title("summary"); ax.grid(alpha=.3, axis="y")

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--networks", nargs="+", default=["CN5", "BN8", "FN8"])
    p.add_argument("--runs", nargs="+", type=int, default=[1])
    p.add_argument("--weights", nargs="+", default=["geometric", "plausibility", "mechanistic"],
                   choices=["geometric", "plausibility", "mechanistic"])
    p.add_argument("--iterations", nargs="+", type=int, default=[0, 1],
                   help="0: pooled mixture; 1: mixture re-fitted on the bins of the first tau")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--min-cells", type=int, default=50)
    p.add_argument("--root-quantile", type=float, default=0.02)
    p.add_argument("--n-neighbors", type=int, default=60)
    p.add_argument("--n-waypoints", type=int, default=50)
    p.add_argument("--no-dpt", action="store_true", help="skip the diffusion pseudotime baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"Pseudotime ordering ({', '.join(args.weights)}) — roots: first "
          f"{100 * args.root_quantile:.1f}% of the true time, {args.bins} bins")
    for net in args.networks:
        for run in args.runs:
            path = args.data_root / net / "Data" / f"data_{run}.h5ad"
            if not path.exists():
                print(f"  missing {path}, skipped")
                continue
            print(f" * {net} run {run}")
            run_dataset(path, args.outdir, args.weights, args.iterations, args.bins,
                        args.min_cells, args.root_quantile, args.n_neighbors, args.n_waypoints,
                        args.seed, with_dpt=not args.no_dpt, verb=not args.quiet)
    print(f"Figures and metrics in {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
