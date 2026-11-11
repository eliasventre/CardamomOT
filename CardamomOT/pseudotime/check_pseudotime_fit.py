#!/usr/bin/env python3
r"""
Run the joint pseudotime cycle on the simulated datasets (``PLAN.md`` §4.4).

Two modes per dataset:

* ``oracle`` — the pseudotime is frozen to the *true* time (``pseudotime_fixed_tau=True``): the
  cycle then only sees correct time bins, which validates the wiring (it must approach the temporal
  CardamomOT) and gives the ceiling of the pipeline;
* ``free`` — the pseudotime is re-estimated by Option 3 at every iteration (the actual method).

Reported per run: the network (AUPRC / AUROC / top-k precision / sign accuracy vs the true graph),
the pseudotime (Spearman with the true time, globally and on the transient), the final error, the
number of iterations, the growth rate and the wall time.

Usage
-----
    conda run -n cardamom_env python CardamomOT/pseudotime/check_pseudotime_fit.py \
        --networks CN5 BN8 FN8 --runs 1 --bins 10 --iterations 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

if __package__ in (None, ""):   # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CardamomOT.pseudotime import PseudotimeNetworkModel, tau_from_time
from CardamomOT.pseudotime.evaluate import grn_metrics, pseudotime_metrics

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data"
DEFAULT_OUT = HERE / "data" / "_pseudotime_fit"


def load(path: Path, root_quantile: float):
    """(AnnData, counts, true time, true network, root mask)."""
    import anndata as ad
    data = ad.read_h5ad(path)
    X = data.X.toarray() if sp.issparse(data.X) else np.asarray(data.X)
    t = np.asarray(data.obs["time"].values, dtype=float)
    roots = t <= np.quantile(t, root_quantile)
    data.obs["is_root"] = roots
    true_inter = np.load(path.parent.parent / "True" / f"inter_{path.stem.split('_')[-1]}.npy")
    # benchmark protocol of the figure scripts: the true degradation rates are given in var
    if "true_d" in data.uns:
        true_d = np.asarray(data.uns["true_d"], dtype=float)      # (2, G_tot), node 0 = stimulus
        data.var["d1"] = true_d[1, 1:]
        data.var["d0"] = true_d[0, 1:]
    return data, np.asarray(X, dtype=float), t, true_inter, roots


def run_mode(data, X, t, true_inter, roots, mode: str, n_bins: int, max_iter: int, seed: int,
             growth: bool, verb: bool, duration: str = "prior") -> dict:
    model = PseudotimeNetworkModel(n_genes=X.shape[1])
    model.seed = seed
    model.pseudotime_bins = n_bins
    model.pseudotime_growth = growth
    model.pseudotime_duration_mode = duration
    if mode == "oracle":
        model.set_tau(tau_from_time(t), roots=roots)   # true-time bins, frozen
        model.pseudotime_fixed_tau = True
    start = time.time()
    model.fit_pseudotime(data, verb=verb, max_iter=max_iter)
    runtime = time.time() - start

    out = dict(mode=mode, runtime=float(runtime), n_iter=len(model.pseudotime_history),
               n_bins=int(model.bins.n_bins), dtau=float(model.bins.dtau),
               total_duration=float(model.bins.total_duration),
               error=float(model.pseudotime_history[-1]["error_after"]),
               force_basins=float(model.pseudotime_history[-1]["force_basins"]))
    out["pseudotime"] = pseudotime_metrics(model.tau, t, counts=X)
    out["grn"] = grn_metrics(model.inter, true_inter)
    R = np.asarray(model.R_opt, dtype=float)
    out["growth_mean"] = float(np.nanmean(R)) if np.any(np.isfinite(R)) else float("nan")
    out["tau"] = model.tau
    return out


def plot(results: dict, t: np.ndarray, title: str, path: Path) -> None:
    """Pseudotime vs true time (one panel per mode) and the AUPRC / Spearman summary."""
    modes = list(results)
    fig, axes = plt.subplots(1, len(modes) + 1, figsize=(4 * (len(modes) + 1), 3.6))
    for ax, mode in zip(np.atleast_1d(axes)[:-1], modes):
        r = results[mode]
        ax.scatter(t, r["tau"], s=4, alpha=.5)
        ax.set_xlabel("true time"); ax.set_ylabel(f"pseudotime ({mode})")
        ax.set_title(f"{mode}: $\\rho$={r['pseudotime']['spearman']:+.2f}, "
                     f"transient {r['pseudotime'].get('spearman_transient', float('nan')):+.2f}",
                     fontsize=9)
        ax.grid(alpha=.3)
    ax = np.atleast_1d(axes)[-1]
    labels = [f"{m}\nAUPRC" for m in modes] + [f"{m}\n$\\rho$" for m in modes]
    values = [results[m]["grn"]["auprc"] for m in modes] + \
             [results[m]["pseudotime"]["spearman"] for m in modes]
    ax.bar(range(len(values)), values, color=["C0"] * len(modes) + ["C1"] * len(modes))
    ax.set_xticks(range(len(values))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("summary"); ax.grid(alpha=.3, axis="y")
    fig.suptitle(title); fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--networks", nargs="+", default=["CN5", "BN8", "FN8"])
    p.add_argument("--runs", nargs="+", type=int, default=[1])
    p.add_argument("--modes", nargs="+", default=["oracle", "free"], choices=["oracle", "free"])
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--root-quantile", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", default="prior", choices=["prior", "fixed", "grid"],
                   help="time unit of the pseudotime axis (PLAN.md §4.3)")
    p.add_argument("--no-growth", action="store_true", help="skip the growth pass")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    verb = not args.quiet
    print(f"Joint pseudotime cycle: {', '.join(args.modes)} | {args.bins} bins, "
          f"max {args.iterations} iterations, seed {args.seed}")
    for net in args.networks:
        for run in args.runs:
            path = args.data_root / net / "Data" / f"data_{run}.h5ad"
            if not path.exists():
                print(f"  missing {path}, skipped")
                continue
            data, X, t, true_inter, roots = load(path, args.root_quantile)
            print(f" * {net} run {run}")
            results = {}
            for mode in args.modes:
                results[mode] = run_mode(data, X, t, true_inter, roots, mode, args.bins,
                                         args.iterations, args.seed, not args.no_growth, verb,
                                         duration=args.duration)
                r = results[mode]
                print(f"   {mode:6s} iters {r['n_iter']:2d} | AUPRC {r['grn']['auprc']:.3f} "
                      f"(AUROC {r['grn']['auroc']:.3f}, top-k {r['grn']['topk_precision']:.2f}) "
                      f"| rho {r['pseudotime']['spearman']:+.3f} "
                      f"(transient {r['pseudotime'].get('spearman_transient', float('nan')):+.3f}) "
                      f"| err {r['error']:.4f} | dtau {r['dtau']:.2f} "
                      f"| R {r['growth_mean']:+.4f} | {r['runtime']:.0f}s")
            stem = f"pseudotime_fit_{net}_r{run}"
            plot(results, t, f"{net} run {run} ({args.bins} bins, seed {args.seed})",
                 args.outdir / f"{stem}.png")
            args.outdir.mkdir(parents=True, exist_ok=True)
            with open(args.outdir / f"{stem}.json", "w") as fh:
                json.dump({"dataset": str(path),
                           "results": {k: {kk: vv for kk, vv in v.items() if kk != "tau"}
                                       for k, v in results.items()}}, fh, indent=2)
    print(f"Figures and metrics in {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
