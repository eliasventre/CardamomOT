"""
Gate G0 of the prototype (PROTOTYPE_PLAN.md §8.2): does the OT cost between two bins carry the
time gap between them?

Datasets (``--networks``):

* simulated networks (BN8, FN8, CH20) in the snapshot design (``data/<NET>/Data_grid/``), whose
  10 libraries are the bins (first library = roots) with known, unequal gaps (3.0 … 24.5 h);
* ``Schiebinger`` (real data, ``experimental_datasets/Schiebinger/Data/data_full.h5ad``,
  ``obs['time']`` in hours): the Dox (0–192 h) then serum (204–432 h) trajectory, at 10 real
  timepoints of growing spacing (``SCHIEBINGER_TIMES``), at most 300 cells per timepoint drawn
  with the seed of the run, roots = the cells at 0 h, d1 / d0 of ``var``.

For each dataset and run:

* **G0a (best case)**: the network is fitted by the oracle cycle (``loop_pseudotime`` with the
  pseudotime frozen) on the snapshot bins **with their true times**;
* **G0b (realistic)**: the same, but with **equal gaps at the prior** ``1 / d1`` (what the loop
  starts from: the network does not know the times);
* **null**: the G0a network, with the non-root cells shuffled between the non-root bins (no time
  signal between two consecutive bins, except from the roots).

Then one E-step of the prototype with the gap search recording the curves ``W_k(Δ)`` of every
formulation (F1 network only; F2 + protein term at three weights; F3 + mRNA term), the trajectories
being propagated with fixed gaps (``propagate='given'``, fixed sources) and the cost scales fixed
at the prior gap.  By default the propagation uses the **prior gap** for every pair
(``--propagation prior``): propagating with the true gaps (``--propagation true``) leaks the true
gaps into the source states (the null then still follows them).

Outputs in ``data/_prototype/g0/``: ``results.json``, ``summary.md`` and one figure of the curves
per network, run and setting.

    conda run -n cardamom_env python CardamomOT/pseudotime/check_prototype.py --networks BN8 FN8
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np

from CardamomOT.pseudotime import PseudotimeNetworkModel
from CardamomOT.pseudotime.binning import bin_by_sizes
from CardamomOT.pseudotime.estep import AlphaPool, estep_with_gaps
from CardamomOT.pseudotime.evaluate import spearman
from CardamomOT.pseudotime.gaps import GapSearch, choose_gap, curve_flags, joint_gaps_dp

DEFAULT_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_OUT = DEFAULT_DATA / "_prototype" / "g0"
REPO = Path(__file__).resolve().parents[2]
SCHIEBINGER = REPO / "experimental_datasets" / "Schiebinger" / "Data" / "data_full.h5ad"
SCHIEBINGER_TIMES = (0.0, 12.0, 36.0, 60.0, 84.0, 120.0, 168.0, 240.0, 324.0, 432.0)
SCHIEBINGER_TREATMENTS = ("Dox", "serum")
OFFLINE_NULL = "F1 - shuffled null (offline)"
FORMULATIONS = {
    "F1": (0.0, 0.0),
    "F6 forecast": "forecast",
    "F1 - self null": "self_null",
    "F2 lP=0.25": (0.25, 0.0),
    "F2 lP=1": (1.0, 0.0),
    "F2 lP=4": (4.0, 0.0),
    "F3 lX=1": (0.0, 1.0),
    "F3 lP=1 lX=1": (1.0, 1.0),
}
SETTINGS = ("G0a", "G0b", "null")


def load_grid(root: Path, network: str, run: int):
    """Snapshot dataset with the true degradation rates as d1/d0 priors (as the benchmark)."""
    data = ad.read_h5ad(root / network / "Data_grid" / f"data_{run}.h5ad")
    true_d = np.asarray(data.uns["true_d"], dtype=float)
    data.var["d1"] = true_d[1, 1:]
    data.var["d0"] = true_d[0, 1:]
    snap = np.asarray(data.obs["snapshot"]).astype(int)
    data.obs["is_root"] = snap == 0
    return data, snap, np.asarray(data.uns["snapshot_times"], dtype=float)


def load_schiebinger(run: int, n_per_time: int = 300, path: Path = SCHIEBINGER):
    """Dox + serum trajectory of Schiebinger at SCHIEBINGER_TIMES, <= n_per_time cells per time."""
    import pandas as pd

    full = ad.read_h5ad(path)
    obs = full.obs
    keep = np.flatnonzero(obs["treatment"].astype(str).isin(SCHIEBINGER_TREATMENTS).values
                          & obs["time"].isin(SCHIEBINGER_TIMES).values)
    rng = np.random.default_rng(run)
    chosen = []
    t_all = obs["time"].values[keep]
    for t in SCHIEBINGER_TIMES:
        idx = keep[t_all == t]
        if idx.size == 0:
            raise ValueError(f"no Schiebinger cell at {t} h")
        chosen.append(np.sort(rng.choice(idx, size=min(n_per_time, idx.size), replace=False)))
    chosen = np.concatenate(chosen)
    X = full.X[chosen]
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=float)
    times = np.asarray(obs["time"].values[chosen], dtype=float)
    snap = np.searchsorted(np.asarray(SCHIEBINGER_TIMES), times)
    data = ad.AnnData(X=X, obs=pd.DataFrame({"time": times, "snapshot": snap, "is_root": snap == 0},
                                            index=obs.index[chosen].astype(str)),
                      var=full.var[["d0", "d1"]].copy())
    return data, snap, np.asarray(SCHIEBINGER_TIMES, dtype=float)


def load_dataset(root: Path, name: str, run: int):
    """(data, snapshot index of each cell, snapshot times) of a simulated network or Schiebinger."""
    if name == "Schiebinger":
        return load_schiebinger(run)
    return load_grid(root, name, run)


def oracle_model(data, snap, bin_times, iterations: int, seed: int) -> PseudotimeNetworkModel:
    """
    The existing oracle cycle (``pseudotime_fixed_tau``) on the snapshot bins with the given bin
    times: mixture, then E/M/alpha/basins on frozen bins.
    """
    model = PseudotimeNetworkModel(n_genes=data.n_vars)
    model.seed = seed
    model.pseudotime_growth = False
    model.prepare_pseudotime(data, verb=False)
    tau = np.asarray(data.obs["time"], dtype=float)
    model.tau = tau
    model.set_bins(bin_by_sizes(tau, np.bincount(snap), snap == 0, bin_times=bin_times))
    model.pseudotime_fixed_tau = True
    model.pseudotime_duration_mode = "grid"         # no re-binning of the frozen bins in the loop
    model.loop_pseudotime(data, verb=False, roots=model.roots, max_iter=iterations)
    return model


def record_curves(model, data, prop_gaps, n_iter: int, seed: int, shuffle: bool = False):
    """One E-step with the gap search recording every formulation (sources propagated with ``prop_gaps``)."""
    data_rna = model._parse_input(data)
    ks, s1, _ = model.mixture_terms()
    bins = model.bins
    if shuffle:                                     # null: non-root cells shuffled between bins
        labels = bins.bin_of_cell.copy()
        rng = np.random.default_rng(seed)
        non_root = np.flatnonzero(labels > 0)
        labels[non_root] = rng.permutation(labels[non_root])
        tau = np.zeros(labels.size)
        tau[non_root] = labels[non_root].astype(float)
        model.tau = tau
        model.set_bins(bin_by_sizes(tau, np.bincount(labels), labels == 0,
                                    bin_times=np.asarray(bins.bin_times)))
    times = np.concatenate([[0.0], np.cumsum(prop_gaps)])
    S = model._scaffolding(data_rna)
    pool = AlphaPool.from_arrays(model.prot, model.kon_beta, model.alpha,
                                 np.asarray(bins.bin_times), int(model.alpha.shape[1]),
                                 int(model.n_stimuli)) \
        if model.alpha is not None and model.alpha.shape[1] == S["n_full"] else None
    search = GapSearch(seed=seed, start=1, scale_ref="prior", extra=dict(FORMULATIONS))
    np.random.seed(seed)
    E = estep_with_gaps(model, data_rna, S, model.basal, model.inter, ks, s1, n_iter,
                        np.diff(times), search=search, pool=pool, propagate="given")
    return E["gap_diag"], search


def gap_metrics(est: np.ndarray, true: np.ndarray) -> dict:
    """Absolute ratio, shape ratio (each gap as a share of the total) and rank agreement."""
    ratio = est / true
    shape = (est / est.sum()) / (true / true.sum())
    return dict(
        abs_within_2x=float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
        shape_within_2x=float(np.mean((shape >= 0.5) & (shape <= 2.0))),
        median_ratio=float(np.median(ratio)),
        spearman=float(spearman(est, true)),
        estimated=[round(float(e), 3) for e in est],
    )


def analyse(diags, true_gaps, log_prior: float, flat_tol: float) -> dict:
    """Per formulation: interior/flat fractions, raw argmin, DP and forward choices, metrics."""
    grid = np.asarray(diags[0]["grid"])
    out = {}
    names = list(FORMULATIONS) + ([OFFLINE_NULL] if OFFLINE_NULL in diags[0]["curves"] else [])
    for name in names:
        curves = np.stack([np.asarray(d["curves"][name]) for d in diags])
        flags = [curve_flags(c, flat_tol) for c in curves]
        raw = np.array([grid[np.argmin(c)] for c in curves])
        _, dp = joint_gaps_dp(grid, curves, log_prior, prior_weight=0.01, smooth_weight=0.01)
        fwd, prev = [], None
        for c in curves:                            # forward: previous chosen gap only
            ch = choose_gap(grid, c, log_prior, 0.01, [] if prev is None else [np.log(prev)], 0.01)
            prev = ch["gap"]
            fwd.append(ch["gap"])
        out[name] = dict(
            interior=float(np.mean([not f["boundary"] for f in flags])),
            flat=float(np.mean([f["flat"] for f in flags])),
            relative_depth=[round(f["relative_depth"], 4) for f in flags],
            raw=gap_metrics(raw, true_gaps), dp=gap_metrics(dp, true_gaps),
            forward=gap_metrics(np.asarray(fwd), true_gaps),
            curves=curves.tolist(),
        )
    return out


def plot(diags, true_gaps, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(diags)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows), squeeze=False)
    for ax, d in zip(axes.ravel(), diags):
        grid = np.asarray(d["grid"])
        for name in d["curves"]:
            c = np.asarray(d["curves"][name])
            ax.plot(grid, c / max(float(np.mean(np.abs(c))), 1e-12), label=name, lw=1.2)
        ax.axvline(true_gaps[d["pair"]], color="k", ls="--", lw=1, label="true gap")
        ax.set_xscale("log")
        ax.set_title(f"pair {d['pair']} -> {d['pair'] + 1}", fontsize=9)
        ax.set_xlabel("gap (h)", fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(title + "  (curves divided by their mean absolute value)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summary_table(records: dict) -> str:
    lines = ["| dataset | setting | formulation | interior | flat | raw shape2x | raw abs2x | "
             "raw median ratio | raw rho | DP shape2x | forward shape2x |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, rec in records.items():
        for setting in SETTINGS:
            res = rec.get(setting)
            if not res:
                continue
            for name, r in res["formulations"].items():
                lines.append(
                    f"| {rec['dataset']} | {setting} | {name} | {r['interior']:.2f} | {r['flat']:.2f} | "
                    f"{r['raw']['shape_within_2x']:.2f} | {r['raw']['abs_within_2x']:.2f} | "
                    f"{r['raw']['median_ratio']:.2f} | {r['raw']['spearman']:+.2f} | "
                    f"{r['dp']['shape_within_2x']:.2f} | {r['forward']['shape_within_2x']:.2f} |")
    return "\n".join(lines)


def aggregate_table(records: dict) -> str:
    """Mean over the runs, per network, setting and formulation (raw argmin of the curves)."""
    rows = {}
    for rec in records.values():
        net = rec["dataset"].split("/")[0]
        for setting in SETTINGS:
            for name, r in rec.get(setting, {}).get("formulations", {}).items():
                rows.setdefault((net, setting, name), []).append(r)
    lines = ["| network | setting | formulation | interior | rho(est, true) | median est/true | "
             "shape2x | DP rho |",
             "|---|---|---|---|---|---|---|---|"]
    for (net, setting, name), rs in rows.items():
        mean = lambda f: float(np.mean([f(r) for r in rs]))
        lines.append(f"| {net} | {setting} | {name} | {mean(lambda r: r['interior']):.2f} | "
                     f"{mean(lambda r: r['raw']['spearman']):+.2f} | "
                     f"{float(np.median([r['raw']['median_ratio'] for r in rs])):.2f} | "
                     f"{mean(lambda r: r['raw']['shape_within_2x']):.2f} | "
                     f"{mean(lambda r: r['dp']['spearman']):+.2f} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--networks", nargs="+", default=["BN8", "FN8"])
    p.add_argument("--runs", nargs="+", type=int, default=[1])
    p.add_argument("--iterations", type=int, default=3, help="iterations of the oracle cycle")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--propagation", choices=["prior", "true"], default="prior",
                   help="gaps of the propagation of the sources (true: leaks the truth)")
    args = p.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    records = {}
    for network in args.networks:
        for run in args.runs:
            start = time.time()
            data, snap, snap_times = load_dataset(args.data_root, network, run)
            true_gaps = np.diff(snap_times)
            K = snap_times.size
            prior_gap = 1.0 / float(np.median(data.var["d1"]))
            prior_times = np.arange(K) * prior_gap
            prop_gaps = true_gaps if args.propagation == "true" else np.full(K - 1, prior_gap)
            rec = dict(dataset=f"{network}/r{run}", true_gaps=true_gaps.round(3).tolist(),
                       d1=float(np.median(data.var["d1"])), propagation=args.propagation,
                       prior_gap=prior_gap)
            model_a = None
            all_diags = {}
            for setting in SETTINGS:
                if setting == "null":                   # the G0a network, shuffled bins
                    model = copy.deepcopy(model_a)
                else:
                    times = prior_times if setting == "G0b" else snap_times
                    model = oracle_model(data, snap, times, args.iterations, args.seed)
                    if setting == "G0a":
                        model_a = copy.deepcopy(model)
                diags, search = record_curves(model, data, prop_gaps, args.iterations + 1,
                                              args.seed, shuffle=(setting == "null"))
                all_diags[setting] = diags
            # the offline correction of the first G0 analysis: F1 minus the F1 curve of the shuffled
            # null of the same dataset (needs a refitted network: not available in a real run)
            for setting in ("G0a", "G0b"):
                for d, dn in zip(all_diags[setting], all_diags["null"]):
                    c, n = np.asarray(d["curves"]["F1"]), np.asarray(dn["curves"]["F1"])
                    d["curves"][OFFLINE_NULL] = c / np.mean(np.abs(c)) - n / np.mean(np.abs(n))
            d1 = np.asarray(data.var["d1"], dtype=float)
            log_prior = float(np.log(1.0 / np.median(d1[d1 > 0])))
            for setting in SETTINGS:
                diags = all_diags[setting]
                rec[setting] = dict(
                    grid=np.asarray(diags[0]["grid"]).round(3).tolist(),
                    formulations=analyse(diags, true_gaps, log_prior, search.flat_tol))
                plot(diags, true_gaps, f"{network} run {run} — {setting}",
                     args.outdir / f"{network}_r{run}_{setting}.png")
                for name in ("F1", "F6 forecast", "F1 - self null"):
                    f = rec[setting]["formulations"][name]
                    print(f"  {network} r{run} {setting} {name:15s}: interior {f['interior']:.2f}, "
                          f"bias x{f['raw']['median_ratio']:.2f}, rho {f['raw']['spearman']:+.2f}",
                          flush=True)
            rec["seconds"] = time.time() - start
            records[f"{network}_r{run}"] = rec
            (args.outdir / "results.json").write_text(json.dumps(records, indent=1))

    table = summary_table(records)
    agg = aggregate_table(records)
    (args.outdir / "summary.md").write_text(table + "\n")
    (args.outdir / "aggregate.md").write_text(agg + "\n")
    print("\n" + agg)
    print(f"\nresults -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
