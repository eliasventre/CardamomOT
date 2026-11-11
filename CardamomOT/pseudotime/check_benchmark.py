"""
M6 — benchmark of the three ordering options and the four baselines (``PLAN.md`` §0.12).

Two sampling designs are compared:

* ``continuous``  — each cell has its own true time (``data/<NET>/Data/``);
* ``grid``        — the cells come from 10 **unequally spaced** library times
  (``data/<NET>/Data_grid/``, ``simulate_pseudotime_data.py --time-design grid``), so the true
  time is a step function.  Spearman is then tie-attenuated, which is why
  ``evaluate.snapshot_metrics`` reports the pairwise order accuracy, the separation of the
  snapshots and the **recovered gap ratios** (does the method see that 40→96 h is ~3x longer
  than 0→3 h?).

Methods
-------
* the three options: ``palantir``, ``partial_ot``, ``tsp`` (full cycle, τ inferred);
* the four baselines, where τ is given and only the E/M cycle runs (``pseudotime_fixed_tau``):
  ``oracle`` (true time), ``palantir_pca`` (geometric Palantir on log counts, no network),
  ``random``, and ``stat`` (CardamomOT without any time information).

Usage::

    conda run -n cardamom_env python -m CardamomOT.pseudotime.check_benchmark \\
        --networks CN5 BN8 FN8 --runs 1 2 3 --designs continuous grid --iterations 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np

from . import PseudotimeNetworkModel
from .evaluate import grn_metrics, kendall, pseudotime_metrics, snapshot_metrics, spearman
from .options.palantir_like import PalantirLike

DEFAULT_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_OUT = DEFAULT_DATA / "_benchmark"
OPTIONS = ("palantir", "partial_ot", "tsp")
BASELINES = ("oracle", "palantir_pca", "random")
METHODS = OPTIONS + BASELINES


def load_dataset(root: Path, network: str, run: int, design: str, root_quantile: float):
    folder = "Data_grid" if design == "grid" else "Data"
    data = ad.read_h5ad(root / network / folder / f"data_{run}.h5ad")
    true_d = np.asarray(data.uns["true_d"], dtype=float)
    data.var["d1"] = true_d[1, 1:]
    data.var["d0"] = true_d[0, 1:]
    true_inter = np.load(root / network / "True" / f"inter_{run}.npy")
    true_time = np.asarray(data.obs["time"].values, dtype=float)
    data.obs["is_root"] = true_time <= np.quantile(true_time, root_quantile)
    return data, true_inter, true_time


def metrics_of(model, true_time, true_inter, design: str, k: int) -> dict:
    out = dict(
        auprc=np.nan, auroc=np.nan, topk_precision=np.nan,
        spearman=np.nan, spearman_transient=np.nan, kendall=np.nan,
        total_duration=np.nan,
    )
    g = grn_metrics(model.inter, true_inter, ns=model.n_stimuli)
    out.update(auprc=g["auprc"], auroc=g["auroc"], topk_precision=g["topk_precision"])
    tau = getattr(model, "tau", None)
    if tau is not None:
        p = pseudotime_metrics(tau, true_time, counts=getattr(model, "counts", None))
        out.update(spearman=p["spearman"], spearman_transient=p["spearman_transient"])
        out["kendall"] = kendall(tau, true_time)
        if design == "grid":
            out.update(snapshot_metrics(tau, true_time))
    bins = getattr(model, "bins", None)
    if bins is not None:
        out["total_duration"] = float(bins.total_duration)
        out["n_bins"] = int(bins.n_bins)
    return out


def run_option(data, method: str, args) -> PseudotimeNetworkModel:
    model = PseudotimeNetworkModel(n_genes=data.n_vars)
    model.seed = args.seed
    model.pseudotime_bins = args.bins
    model.pseudotime_growth = False
    model.pseudotime_method = method
    model.fit_pseudotime(data, verb=False, max_iter=args.iterations)
    return model


def run_baseline(data, method: str, args, true_time, roots) -> PseudotimeNetworkModel:
    """τ is imposed (baseline) and frozen; only the E/M cycle runs."""
    model = PseudotimeNetworkModel(n_genes=data.n_vars)
    model.seed = args.seed
    model.pseudotime_bins = args.bins
    model.pseudotime_growth = False
    model.prepare_pseudotime(data, verb=False)      # mixture + a first ordering (needed for modes)
    n = int(model.modes.shape[0])
    if method == "oracle":
        tau = np.asarray(true_time, dtype=float)
    elif method == "random":
        tau = np.random.default_rng(args.seed).random(n)
    elif method == "palantir_pca":
        tau = PalantirLike(feature="expression", edge_weight="geometric", seed=args.seed).run(
            model, roots=roots).tau
    else:
        raise ValueError(f"unknown baseline {method!r}")
    model.set_tau(tau, roots=roots)
    model.pseudotime_fixed_tau = True
    model.loop_pseudotime(data, verb=False, roots=roots, max_iter=args.iterations)
    return model


def aggregate_table(records: dict) -> str:
    """Mean over the runs, one block per (network, design): the table of ``PLAN.md`` §0.12."""
    lines = []
    for network in sorted({r["network"] for r in records.values()}):
        for design in ("continuous", "grid"):
            rows = [r for r in records.values() if r["network"] == network and r["design"] == design]
            if not rows:
                continue
            lines.append(f"#### {network} / {design}")
            lines.append("| method | AUPRC (sd) | AUROC | rho (sd) | kendall | order | sep | gap2x "
                         "| duration | s |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for method in METHODS:
                sel = [r for r in rows if r["method"] == method and "metrics" in r]
                if not sel:
                    continue
                m = [r["metrics"] for r in sel]
                sec = [r["seconds"] for r in sel]

                def stat(key, fmt="{:.3f}"):
                    v = np.array([d.get(key, np.nan) for d in m], dtype=float)
                    return "n/a" if np.all(~np.isfinite(v)) else fmt.format(np.nanmean(v))

                def sd(key):
                    v = np.array([d.get(key, np.nan) for d in m], dtype=float)
                    return "" if np.all(~np.isfinite(v)) else f" ({np.nanstd(v):.3f})"

                lines.append(
                    f"| {method} | {stat('auprc')}{sd('auprc')} | {stat('auroc')} "
                    f"| {stat('spearman', '{:+.3f}')}{sd('spearman')} | {stat('kendall', '{:+.3f}')} "
                    f"| {stat('order_accuracy')} | {stat('separation')} "
                    f"| {stat('gap_ratio_within_2x')} | {stat('total_duration', '{:.1f}')} "
                    f"| {np.mean(sec):.1f} |")
            lines.append("")
    return "\n".join(lines)


def summarise(records: dict) -> str:
    """Markdown table of the records, one row per (dataset, method)."""
    cols = ["auprc", "auroc", "spearman", "kendall", "order_acc", "separation",
            "gap_ratio_2x", "duration", "seconds"]
    head = "| dataset | method | " + " | ".join(cols) + " |"
    sep = "|---" * (len(cols) + 2) + "|"
    lines = [head, sep]
    for key, rec in records.items():
        m = rec.get("metrics", {})
        row = [
            f"{m.get('auprc', np.nan):.3f}", f"{m.get('auroc', np.nan):.3f}",
            f"{m.get('spearman', np.nan):+.3f}", f"{m.get('kendall', np.nan):+.3f}",
            f"{m.get('order_accuracy', np.nan):.3f}", f"{m.get('separation', np.nan):.3f}",
            f"{m.get('gap_ratio_within_2x', np.nan):.2f}",
            f"{m.get('total_duration', np.nan):.1f}", f"{rec.get('seconds', np.nan):.1f}",
        ]
        lines.append(f"| {rec['dataset']} | {rec['method']} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--networks", nargs="+", default=["CN5", "BN8", "FN8"])
    p.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--designs", nargs="+", default=["continuous", "grid"],
                   choices=["continuous", "grid"])
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--root-quantile", type=float, default=0.02)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--append", action="store_true",
                   help="merge with the records of an existing results.json")
    args = p.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    results_path = args.outdir / "results.json"
    records = {}
    if args.append and results_path.exists():
        records = json.loads(results_path.read_text())

    for network in args.networks:
        for run in args.runs:
            for design in args.designs:
                data, true_inter, true_time = load_dataset(
                    args.data_root, network, run, design, args.root_quantile)
                roots = np.asarray(data.obs["is_root"].values, dtype=bool)
                for method in args.methods:
                    key = f"{network}_run{run}_{design}_{method}"
                    if key in records:
                        print(f"  {key}: already done, skipped")
                        continue
                    start = time.time()
                    try:
                        if method in OPTIONS:
                            model = run_option(data, method, args)
                        else:
                            model = run_baseline(data, method, args, true_time, roots)
                        rec = dict(dataset=f"{network}/r{run}/{design}", method=method,
                                   design=design, network=network, run=run,
                                   seconds=time.time() - start,
                                   metrics=metrics_of(model, true_time, true_inter, design, args.bins))
                    except Exception as exc:                       # keep the benchmark going
                        rec = dict(dataset=f"{network}/r{run}/{design}", method=method,
                                   design=design, network=network, run=run,
                                   seconds=time.time() - start, error=repr(exc))
                    records[key] = rec
                    results_path.write_text(json.dumps(records, indent=2))
                    m = rec.get("metrics", {})
                    print(f"  {key:44s} AUPRC {m.get('auprc', np.nan):.3f} "
                          f"rho {m.get('spearman', np.nan):+.3f} "
                          f"{'' if 'error' not in rec else 'ERROR ' + rec['error'][:60]}",
                          flush=True)

    table = summarise(records)
    (args.outdir / "table.md").write_text(table + "\n")
    agg = aggregate_table(records)
    (args.outdir / "aggregate.md").write_text(agg + "\n")
    print("\n" + agg)
    print(f"\nresults -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
