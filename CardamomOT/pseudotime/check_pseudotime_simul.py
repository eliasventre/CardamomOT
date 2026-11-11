"""
M5 — growth, degradations and simulations in pseudotime mode (``PLAN.md`` §0.10).

Runs the full chain on the simulated networks and reports the gate of ``PLAN.md`` §8:

    mixture + ordering cycle -> growth pass -> refine_network_degradations -> simulate_network

* **$R$**: the net proliferation rate read along the pseudotime axis (the simulations do not
  proliferate, so the target is $R \\approx 0$);
* **$d_1$**: the refinement is the only place where it is *inferred* rather than taken from the
  prior ``var['d1']`` that sets the duration of the axis — the gate is "within a factor 2";
* **marginals**: mean mRNA of the re-simulated cells vs the observed cells, per bin and per gene
  (``evaluate.marginal_agreement``), gate "within 2x".

Usage::

    conda run -n cardamom_env python CardamomOT/pseudotime/check_pseudotime_simul.py \\
        --networks CN5 BN8 --runs 1 --iterations 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np

from . import PseudotimeNetworkModel
from .evaluate import degradation_metrics, marginal_agreement

DEFAULT_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_OUT = DEFAULT_DATA / "_pseudotime_simul"


def load_dataset(root: Path, network: str, run: int, root_quantile: float):
    """AnnData + the true network / degradation rates of one simulated run."""
    data = ad.read_h5ad(root / network / "Data" / f"data_{run}.h5ad")
    true_d = np.asarray(data.uns["true_d"], dtype=float)
    data.var["d1"] = true_d[1, 1:]
    data.var["d0"] = true_d[0, 1:]
    true_inter = np.load(root / network / "True" / f"inter_{run}.npy")
    time = np.asarray(data.obs["time"].values, dtype=float)
    data.obs["is_root"] = time <= np.quantile(time, root_quantile)
    return data, true_inter, true_d, time


def run_one(data, true_d, true_inter, time, args, method: str) -> dict:
    model = PseudotimeNetworkModel(n_genes=data.n_vars)
    model.seed = args.seed
    model.pseudotime_bins = args.bins
    model.pseudotime_method = method
    model.pseudotime_growth = True
    model.fit_pseudotime(data, verb=False, max_iter=args.iterations, refine=True)

    bins = np.sort(np.unique(model.times_data))
    model.simulate_network(list(bins), verb=False)

    rec = dict(
        method=method,
        R_mean=float(np.nanmean(model.R_opt)),
        R_abs_max=float(np.nanmax(np.abs(model.R_opt))),
        total_duration=float(model.bins.total_duration),
        dtau=float(model.bins.dtau),
        degradations=degradation_metrics(model, true_d),
        marginals=marginal_agreement(model, data),
        n_bins=int(model.bins.n_bins),
    )
    if true_inter is not None:
        from .evaluate import grn_metrics, pseudotime_metrics
        rec["grn"] = grn_metrics(model.inter, true_inter, ns=model.n_stimuli)
        rec["pseudotime"] = pseudotime_metrics(model.tau, time)
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--networks", nargs="+", default=["CN5", "BN8", "FN8"])
    p.add_argument("--runs", nargs="+", type=int, default=[1])
    p.add_argument("--methods", nargs="+", default=["palantir"],
                   help="ordering options to run (palantir | partial_ot | tsp)")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--root-quantile", type=float, default=0.02)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    print(f"{'net':6s}{'method':11s} {'R (net)':>10s} {'d1 ratio':>18s} {'2x':>4s} "
          f"{'marg. ratio':>16s} {'2x':>5s}")
    for network in args.networks:
        for run in args.runs:
            data, true_inter, true_d, time = load_dataset(
                args.data_root, network, run, args.root_quantile)
            for method in args.methods:
                rec = run_one(data, true_d, true_inter, time, args, method)
                results[f"{network}_run{run}_{method}"] = rec
                deg, marg = rec["degradations"], rec["marginals"]
                r = np.asarray(deg["d1_ratio"], dtype=float)
                print(f"{network:6s}{method:11s} {rec['R_mean']:+10.2e} "
                      f"[{r.min():.2f},{r.max():.2f}] {deg['d1_within_2x']:4.2f} "
                      f"[{marg['ratio_min']:.2f},{marg['ratio_max']:.2f}]".ljust(60)
                      + f" {marg['frac_within_2x']:5.2f}")

    out = args.outdir / "metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nmetrics -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
