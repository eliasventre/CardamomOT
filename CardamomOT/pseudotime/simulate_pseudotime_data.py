#!/usr/bin/env python3
r"""
Simulate time-stamped single-cell datasets for the pseudotime experiments.

The datasets are generated with the bursty PDMP model of Harissa (``harissa.NetworkModel``),
using exactly the same ground-truth networks and kinetic parameters as
``results_article/figures/simulate_data_{CN5,BN8,FN8}.py``.

The only difference with these figure scripts is the way the cells are sampled. The figure
scripts draw ``n_cells / 10`` cells at each of the 10 fixed time points ``{0, 6, ..., 96}``;
here every cell gets its own continuous time, drawn uniformly in ``[t_min, t_max]``:

* there are no replicates of the same time point, since the time of a cell is a continuous
  quantity. This is the situation we want to be able to solve: infer the dynamics (and
  progressively the pseudotime itself) from unordered cells;
* the true time of each cell is kept (``obs['time']``), as the ground truth that an
  inference of the pseudotime without time information will be compared to;
* the cells are shuffled, so their order does not leak the time.

Each cell is simulated independently: the network is first run for ``burnin`` without
stimulus, the stimulus is then activated, and the model is simulated up to the time of the
cell. Counts are Poisson draws of the mRNA means at that time, exactly as in the figure
scripts.

Outputs, for each network ``NET`` and each run ``r`` (mirrors the layout of
``results_article/figures/<NET>_/``)::

    <outdir>/<NET>/Data/data_{r}.h5ad        AnnData: X = raw counts (cells x genes),
                                             obs['time'] (true time), obs['stimulus'],
                                             obs['run'], var_names = G1..GG, plus the
                                             ground-truth parameters in uns[...]
    <outdir>/<NET>/Data/data_{r}.txt         PIDC-like table (same layout as the figure
                                             datasets): row 0 = time, row 1 = stimulus,
                                             rows 2.. = counts of genes 1..G, column 0 =
                                             labels. Only row 0 holds floats (cell times).
    <outdir>/<NET>/True/inter_{r}.npy        binarised ground-truth network, (G+1) x (G+1),
                                             node 0 = stimulus
    <outdir>/<NET>/True/degradation_rates.txt   true degradation rates (d.T), as in the
                                             figure datasets
    <outdir>/<NET>/True/network.npz          true kinetic parameters (a, d, basal, inter)
                                             and the node names
    <outdir>/<NET>/metadata.json             simulation settings (times, cells, seeds, ...)

Examples
--------
Simulate the three networks with the default settings (2000 cells, 5 runs, times in [0, 96])::

    conda run -n cardamom_env python CardamomOT/pseudotime/simulate_pseudotime_data.py

A quick small dataset (e.g. for tests)::

    conda run -n cardamom_env python CardamomOT/pseudotime/simulate_pseudotime_data.py \
        --networks CN5 --cells 200 --runs 1 --tmax 24

Also usable as a module::

    python -m CardamomOT.pseudotime.simulate_pseudotime_data --networks BN8 FN8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_OUTDIR = HERE / "data"

# Name of the stimulus node (index 0 of the network matrices) in the saved ground truth
STIMULUS_NAME = "Stimulus"

def _cascade(model, nodes, sign: int = 10) -> None:
    """``nodes[0] -> nodes[1] -> ...`` (``inter[src, tgt]`` = effect of src on tgt)."""
    for a, b in zip(nodes[:-1], nodes[1:]):
        model.inter[a, b] = sign

# --------------------------------------------------------------------------------------
# Ground-truth networks
# --------------------------------------------------------------------------------------
def _new_model(n_genes: int):
    """``harissa.NetworkModel(n_genes)`` with a helpful message when harissa is missing."""
    try:
        from harissa import NetworkModel
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "harissa is required to simulate the datasets but is not installed in this "
            "Python environment.\nInstall it (pip install harissa) or run the script from an "
            "environment that has it, e.g.:\n    conda run -n cardamom_env python "
            f"{Path(__file__).as_posix()}"
        ) from exc
    return NetworkModel(n_genes)


def build_cn5():
    """5-gene cycle, as in results_article/figures/simulate_data_CN5.py."""
    model = _new_model(5)
    model.d[0] = 0.5
    model.d[1] = 0.1

    model.basal[1:] = [-5, 4, 4, -5, -5]
    model.inter[0, 1] = 10
    model.inter[1, 2] = -10
    model.inter[2, 3] = -10
    model.inter[3, 4] = 10
    model.inter[4, 5] = 10
    model.inter[5, 1] = -10

    return model


def build_bn8():
    """8-gene bifurcation, as in results_article/figures/simulate_data_BN8.py."""
    model = _new_model(8)
    model.d[0] = 0.25
    model.d[1] = 0.05

    model.basal[1:] = [-4, -4, -4, -4, -4, -4, -4, -4]
    model.inter[0, 1] = 10
    model.inter[1, 2] = 10
    model.inter[1, 3] = 10
    model.inter[3, 2] = -10
    model.inter[2, 3] = -10
    model.inter[2, 2] = 5
    model.inter[3, 3] = 5
    model.inter[2, 4] = 10
    model.inter[3, 5] = 10
    model.inter[2, 5] = -10
    model.inter[3, 4] = -10
    model.inter[4, 7] = -10
    model.inter[5, 6] = -10
    model.inter[4, 6] = 10
    model.inter[5, 7] = 10
    model.inter[7, 8] = 10
    model.inter[6, 8] = -10

    return model


def build_fn8():
    """8-gene feed-forward network, as in results_article/figures/simulate_data_FN8.py."""
    model = _new_model(8)
    model.d[0] = 0.4
    model.d[1] = 0.08

    model.basal[1:] = [-5, -5, -5, -5, -5, -5, -5, -5]
    model.inter[0, 1] = 10
    model.inter[1, 2] = 10
    model.inter[2, 3] = 10
    model.inter[3, 4] = 10
    model.inter[3, 5] = 10
    model.inter[3, 6] = 10
    model.inter[4, 1] = -10
    model.inter[5, 1] = -10
    model.inter[6, 1] = -10
    model.inter[4, 4] = 10
    model.inter[5, 5] = 10
    model.inter[6, 6] = 10
    model.inter[4, 8] = -10
    model.inter[4, 7] = -10
    model.inter[6, 7] = 10
    model.inter[7, 6] = 10
    model.inter[8, 8] = 10

    return model


def build_ch20():
    """
    20 genes: stimulus -> 4 hub genes, each hub activating a chain of 4 genes (a 4-arm tree), with
    the third gene of each arm repressing its hub (a delayed negative feedback).
    """
    model = _new_model(20)
    model.d[0] = 0.4
    model.d[1] = 0.08

    model.basal[1:] = -4.0
    arms = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10), (11, 12, 13, 14, 15), (16, 17, 18, 19, 20)]
    for arm in arms:
        model.inter[0, arm[0]] = 10            # stimulus -> hub of the arm
        _cascade(model, arm)                   # hub -> its chain
        model.inter[arm[2], arm[0]] = -10      # delayed negative feedback on the hub

    return model


def build_rg50():
    """
    50 genes: sparse random DAG in topological order (the depth of a gene sets its timing), the
    stimulus driving three root genes, mixed signs and a few self-activations.
    """
    rng = np.random.default_rng(0)             # fixed seed: the benchmark is reproducible
    n = 50
    model = _new_model(n)
    model.d[0] = 0.4
    model.d[1] = 0.08

    model.basal[1:] = -4.5
    for root in (1, 2, 3):
        model.inter[0, root] = 10
    for src in range(1, n + 1):
        for tgt in range(src + 1, n + 1):
            if rng.random() < 0.12:
                model.inter[src, tgt] = 10 if rng.random() < 0.85 else -10
    for tgt in range(4, n + 1):                # every gene has at least one regulator
        if not np.any(model.inter[1:tgt, tgt]):
            src = int(rng.integers(1, tgt))
            model.inter[src, tgt] = 10 if rng.random() < 0.9 else -10
    for g in rng.choice(np.arange(1, n + 1), size=5, replace=False):
        model.inter[g, g] = 5                  # a few self-activations
    return model


def build_md100():
    """
    100 genes: five modules of 20 (a hub + 19 targets), the modules switched on in cascade
    (stimulus -> hub 1 -> hub 2 -> ...), so that the expression changes are staggered in time.
    """
    model = _new_model(100)
    model.d[0] = 0.4
    model.d[1] = 0.08

    model.basal[1:] = -5.0
    modules = [list(range(1 + 20 * m, 1 + 20 * (m + 1))) for m in range(5)]
    for m, genes in enumerate(modules):
        hub = genes[0]
        if m == 0:
            model.inter[0, hub] = 10           # only the first module sees the stimulus
        else:
            model.inter[modules[m - 1][0], hub] = 10
        for g in genes[1:]:
            model.inter[hub, g] = 10
        # a single weak repression back on the hub, at the end of the module (its targets add up:
        # one edge per target would switch the hub off and collapse the whole module)
        model.inter[genes[-1], hub] = -3
    return model


#: Available ground-truth networks (add new builders here, e.g. "FN4": build_fn4)
NETWORKS: Dict[str, Callable[[], object]] = {
    "CN5": build_cn5,
    "BN8": build_bn8,
    "FN8": build_fn8,
    "CH20": build_ch20,
    "RG50": build_rg50,
    "MD100": build_md100,
}

#: Per-network simulation defaults: the larger networks are simulated with fewer cells / runs
#: (one harissa simulation per cell, whose cost grows with the number of genes).
NETWORK_DEFAULTS: Dict[str, dict] = {
    "CN5": dict(cells=2000, runs=5),
    "BN8": dict(cells=2000, runs=5),
    "FN8": dict(cells=2000, runs=5),
    "CH20": dict(cells=1000, runs=3),
    "RG50": dict(cells=800, runs=3),
    "MD100": dict(cells=500, runs=3),
}


def gene_names(n_genes: int) -> List[str]:
    """Names of the genes (the stimulus node excluded), used as ``var_names`` of the AnnData."""
    return [f"G{i}" for i in range(1, n_genes + 1)]


# --------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------
@dataclass
class SimConfig:
    """Settings of the simulation, shared by all networks and runs."""

    cells: int = 2000          # number of cells per run
    runs: int = 5              # number of independent runs (replicates) per network
    t_min: float = 0.0         # cells are sampled uniformly in [t_min, t_max]
    t_max: float = 96.0
    burnin: float = 5.0        # duration of the simulation without stimulus before t = 0
    seed: int = 0
    # sampling design of the cell times (PLAN.md §0.11)
    time_design: str = "continuous"   # 'continuous' | 'grid' (unequally spaced snapshots)
    n_snapshots: int = 10             # 'grid' only: number of library times
    grid_ratio: float = 1.3           # 'grid' only: ratio of the geometric progression of gaps


TIME_DESIGNS = ("continuous", "grid")


def snapshot_times(t_min: float, t_max: float, n_snapshots: int = 10,
                   ratio: float = 1.3) -> np.ndarray:
    """
    Unequally spaced snapshot times over ``[t_min, t_max]``.

    The gaps follow a geometric progression of ratio ``ratio``: the last one is
    ``ratio ** (n_snapshots - 2)`` times the first (8.2x for the default 10 snapshots and
    ``ratio = 1.3``).  This mimics the library times of a real sequencing design — dense in
    the transient, sparse at the end — and, unlike a uniform grid, it makes the *spacing*
    itself an unknown that a dynamical model can in principle recover.
    """
    if not t_max > t_min:
        raise ValueError(f"t_max ({t_max}) must be larger than t_min ({t_min})")
    n_gap = int(n_snapshots) - 1
    if n_gap < 1:
        raise ValueError("n_snapshots must be >= 2")
    if ratio <= 0:
        raise ValueError("grid_ratio must be > 0")
    gaps = np.power(float(ratio), np.arange(n_gap))
    gaps = gaps / gaps.sum()
    return t_min + (t_max - t_min) * np.concatenate(([0.0], np.cumsum(gaps)))


def sample_times(n_cells: int, t_min: float, t_max: float, rng: np.random.Generator,
                 design: str = "continuous", n_snapshots: int = 10,
                 grid_ratio: float = 1.3) -> np.ndarray:
    """
    Sampling time of each cell, shape ``(n_cells,)``.

    * ``'continuous'``: uniform in ``[t_min, t_max]`` — every cell has its own time;
    * ``'grid'``: the unequal snapshots of :func:`snapshot_times`, with the same number of
      cells at each one — the design of a real experiment, where a cell is only known up to
      the library it comes from.
    """
    if not t_max > t_min:
        raise ValueError(f"t_max ({t_max}) must be larger than t_min ({t_min})")
    if design == "continuous":
        return rng.uniform(t_min, t_max, size=n_cells)
    if design != "grid":
        raise ValueError(f"design must be one of {TIME_DESIGNS}, got {design!r}")
    snaps = snapshot_times(t_min, t_max, n_snapshots, grid_ratio)
    per = np.full(snaps.size, n_cells // snaps.size)
    per[: n_cells % snaps.size] += 1            # equal cells per snapshot (as even as possible)
    return np.repeat(snaps, per)


def simulate_counts(model, times: Sequence[float], rng: np.random.Generator,
                    burnin: float = 5.0, verbose: bool = True) -> np.ndarray:
    """
    Counts of one independent cell per time, shape ``(len(times), n_genes)``.

    As in the figure scripts, each cell is obtained by simulating the model up to its own
    time after a burn-in without stimulus, then drawing the counts from a Poisson law of
    mean the simulated mRNA level at that time.
    """
    n_genes = model.inter.shape[0] - 1
    counts = np.zeros((len(times), n_genes), dtype=np.int64)
    n_cells = len(times)
    step = max(1, n_cells // 10)

    for i, t in enumerate(times):
        sim = model.simulate(float(t), burnin=burnin)
        counts[i] = rng.poisson(sim.m[-1])
        if verbose and ((i + 1) % step == 0 or i + 1 == n_cells):
            print(f"    {i + 1}/{n_cells} cells", flush=True)

    return counts


# --------------------------------------------------------------------------------------
# Writing the datasets
# --------------------------------------------------------------------------------------
def pidc_table(times: np.ndarray, stimulus: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Table in the format of the figure datasets (before transposition): shape
    ``(n_cells + 1, n_genes + 2)``, row 0 = labels, column 0 = time, column 1 = stimulus,
    columns 2.. = counts.
    """
    data = np.zeros((len(times) + 1, counts.shape[1] + 2), dtype=float)
    data[0, 1:] = np.arange(counts.shape[1] + 1)
    data[1:, 0] = times
    data[1:, 1] = stimulus
    data[1:, 2:] = counts
    return data


def write_pidc_txt(path: Path, times: np.ndarray, stimulus: np.ndarray,
                   counts: np.ndarray) -> None:
    """
    PIDC-like text file, same layout as ``results_article/figures/<NET>_/Data/data_{r}.txt``:
    row 0 = time, row 1 = stimulus, rows 2.. = counts, first column = labels.

    The time row is written as floats (the cells have continuous times), the other rows as
    integers; readers must therefore load row 0 with ``dtype=float`` and the rest with
    ``dtype=int``.
    """
    table = pidc_table(times, stimulus, counts).T
    with open(path, "w") as fh:
        np.savetxt(fh, table[:1], fmt="%.10g", delimiter="\t")   # time row (floats)
        np.savetxt(fh, table[1:], fmt="%d", delimiter="\t")      # stimulus + counts (ints)


def write_h5ad(path: Path, network: str, run: int, times: np.ndarray, stimulus: np.ndarray,
               counts: np.ndarray, model, snapshot_times=None) -> None:
    """AnnData with the raw counts in ``X``, the true time in ``obs['time']``, the ground truth in ``uns``."""
    import anndata as ad
    import pandas as pd

    n_genes = counts.shape[1]
    names = gene_names(n_genes)

    obs = pd.DataFrame(
        {"time": times.astype(float),
         "stimulus": stimulus.astype(float),
         "run": np.full(len(times), run, dtype=int)},
        index=[f"cell_{i:05d}" for i in range(len(times))],
    )
    adata = ad.AnnData(X=counts.astype(np.int32), obs=obs, var=pd.DataFrame(index=names))

    adata.uns["network"] = network
    adata.uns["run"] = run
    adata.uns["gene_names"] = np.array(names)
    adata.uns["stimulus_name"] = STIMULUS_NAME
    adata.uns["node_names"] = np.array([STIMULUS_NAME] + names)
    # Ground-truth parameters, on the full network (node 0 = stimulus)
    adata.uns["true_inter"] = np.asarray(model.inter, dtype=float)
    adata.uns["true_network"] = (np.abs(model.inter) > 0).astype(np.int8)
    adata.uns["true_a"] = np.asarray(model.a, dtype=float)
    adata.uns["true_d"] = np.asarray(model.d, dtype=float)
    adata.uns["true_basal"] = np.asarray(model.basal, dtype=float)
    adata.uns["time_unit"] = "model time (dimensionless)"
    if snapshot_times is not None:
        # 'grid' design: the library times, and the timepoint of each cell as a category
        snaps = np.asarray(snapshot_times, dtype=float)
        adata.uns["snapshot_times"] = snaps
        adata.uns["time_design"] = "grid"
        adata.obs["snapshot"] = np.searchsorted(snaps, times).astype(int)
    else:
        adata.uns["time_design"] = "continuous"

    adata.write_h5ad(path)


def write_true_inter(net_dir: Path, run: int, model) -> None:
    """Binarised ground-truth network of one run, in ``<NET>/True/inter_{run}.npy``."""
    true_dir = net_dir / "True"
    true_dir.mkdir(parents=True, exist_ok=True)
    np.save(true_dir / f"inter_{run}.npy", 1 * (np.abs(model.inter) > 0))


def write_true_parameters(net_dir: Path, model) -> None:
    """Kinetic parameters, shared by all the runs, in ``<NET>/True/``."""
    true_dir = net_dir / "True"
    true_dir.mkdir(parents=True, exist_ok=True)

    # Same layout as the figure datasets: d.T, i.e. one row per node and one column per
    # degradation parameter (mRNA, protein)
    np.savetxt(true_dir / "degradation_rates.txt", np.asarray(model.d).T, fmt="%.6g",
               delimiter="\t")
    n_genes = model.inter.shape[0] - 1
    np.savez(
        true_dir / "network.npz",
        a=np.asarray(model.a, dtype=float),
        d=np.asarray(model.d, dtype=float),
        basal=np.asarray(model.basal, dtype=float),
        inter=np.asarray(model.inter, dtype=float),
        inter_binary=1 * (np.abs(model.inter) > 0),
        node_names=np.array([STIMULUS_NAME] + gene_names(n_genes)),
    )


def write_metadata(net_dir: Path, network: str, cfg: SimConfig, n_genes: int) -> None:
    """Simulation settings, to keep the datasets self-describing."""
    metadata = {
        "network": network,
        "n_genes": n_genes,
        "n_nodes": n_genes + 1,
        "gene_names": gene_names(n_genes),
        "stimulus_name": STIMULUS_NAME,
        "n_cells_per_run": cfg.cells,
        "n_runs": cfg.runs,
        "time_sampling": "uniform",
        "t_min": cfg.t_min,
        "t_max": cfg.t_max,
        "burnin": cfg.burnin,
        "seed": cfg.seed,
        "model": "harissa bursty PDMP; counts = Poisson(mRNA mean at the cell time)",
        "cell_order": "shuffled",
        "obs": {"time": "true simulated time of the cell (ground truth pseudotime)",
                "stimulus": "stimulus level, 100 * (time > 0) as in the figure datasets",
                "run": "run (replicate) index"},
        "files": {"anndata": "Data/data_{r}.h5ad",
                  "pidc_table": "Data/data_{r}.txt (row 0 = float times)",
                  "true_network": "True/inter_{r}.npy",
                  "degradation_rates": "True/degradation_rates.txt",
                  "kinetic_parameters": "True/network.npz"},
    }
    with open(net_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def simulate_network(network: str, cfg: SimConfig, outdir: Path, net_index: int,
                     overwrite: bool = False, verbose: bool = True) -> List[Path]:
    """Simulate and save the ``cfg.runs`` runs of one network. Returns the written files."""
    builder = NETWORKS[network]
    net_dir = outdir / network
    data_dir = net_dir / ("Data_grid" if cfg.time_design == "grid" else "Data")
    data_dir.mkdir(parents=True, exist_ok=True)

    model = builder()
    n_genes = model.inter.shape[0] - 1
    write_metadata(net_dir, network, cfg, n_genes)
    write_true_parameters(net_dir, model)

    # One independent random stream per (network, run), so that runs can be added or re-run
    # without changing the others
    run_sequences = np.random.SeedSequence([cfg.seed, net_index]).spawn(cfg.runs)
    written: List[Path] = []

    for run, sequence in enumerate(run_sequences, start=1):
        h5ad_path = data_dir / f"data_{run}.h5ad"
        txt_path = data_dir / f"data_{run}.txt"
        if not overwrite and h5ad_path.exists() and txt_path.exists():
            print(f"  [{network}] run {run}/{cfg.runs}: already simulated, skipped "
                  f"(use --overwrite to replace it)")
            continue

        if verbose:
            print(f"  [{network}] run {run}/{cfg.runs}: {cfg.cells} cells, times in "
                  f"[{cfg.t_min:g}, {cfg.t_max:g}]", flush=True)
        start = time.time()

        sim_sequence, global_sequence = sequence.spawn(2)
        rng = np.random.default_rng(sim_sequence)
        # harissa uses the global numpy RNG: seeding it makes each run reproducible
        np.random.seed(int(global_sequence.generate_state(1)[0]))

        # Cells are drawn already in a shuffled order: their index does not carry the time
        times = sample_times(cfg.cells, cfg.t_min, cfg.t_max, rng, design=cfg.time_design,
                             n_snapshots=cfg.n_snapshots, grid_ratio=cfg.grid_ratio)
        times = times[rng.permutation(cfg.cells)]
        counts = simulate_counts(model, times, rng, burnin=cfg.burnin, verbose=verbose)
        stimulus = 100.0 * (times > 0)   # same convention as the figure datasets
        snaps = (snapshot_times(cfg.t_min, cfg.t_max, cfg.n_snapshots, cfg.grid_ratio)
                 if cfg.time_design == "grid" else None)

        write_h5ad(h5ad_path, network, run, times, stimulus, counts, model, snapshot_times=snaps)
        write_pidc_txt(txt_path, times, stimulus, counts)
        write_true_inter(net_dir, run, model)
        written += [h5ad_path, txt_path]

        if verbose:
            nonzero = float(np.mean(counts.sum(axis=1) > 0))
            print(f"  [{network}] run {run}/{cfg.runs}: saved in {time.time() - start:.1f}s "
                  f"(mean counts/gene: {counts.mean(axis=0).round(1).tolist()}, "
                  f"cells with signal: {100 * nonzero:.0f}%)", flush=True)

    if verbose:
        print(f"  [{network}] ground truth and metadata in {net_dir}", flush=True)
    return written


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate time-stamped single-cell datasets (CN5, BN8, FN8) with a "
                    "continuous, pseudotime-like sampling of the cell times.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--networks", nargs="+", choices=list(NETWORKS),
                        default=list(NETWORKS), metavar="NET",
                        help=f"networks to simulate, among {list(NETWORKS)}")
    parser.add_argument("--cells", type=int, default=None,
                        help="number of cells per run (default: per network, see NETWORK_DEFAULTS)")
    parser.add_argument("--runs", type=int, default=None,
                        help="number of runs (replicates) per network (default: per network)")
    parser.add_argument("--tmin", type=float, default=SimConfig.t_min,
                        help="lower bound of the uniform sampling of the cell times")
    parser.add_argument("--tmax", type=float, default=SimConfig.t_max,
                        help="upper bound of the uniform sampling of the cell times "
                             "(lower it, e.g. 24, to keep the cells in the transient regime)")
    parser.add_argument("--burnin", type=float, default=SimConfig.burnin,
                        help="duration of the simulation without stimulus before t = 0")
    parser.add_argument("--time-design", choices=list(TIME_DESIGNS),
                        default=SimConfig.time_design,
                        help="'continuous' (one time per cell) or 'grid' (a few unequally "
                             "spaced snapshots, written to Data_grid/)")
    parser.add_argument("--snapshots", type=int, default=SimConfig.n_snapshots,
                        help="'grid' only: number of library times")
    parser.add_argument("--grid-ratio", type=float, default=SimConfig.grid_ratio,
                        help="'grid' only: ratio of the geometric progression of the gaps "
                             "(the last gap is ratio**(snapshots-2) times the first)")
    parser.add_argument("--seed", type=int, default=SimConfig.seed,
                        help="master seed (one independent stream per network and run)")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help="output folder, one sub-folder per network")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-simulate the runs whose files already exist")
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    print(f"Simulating {', '.join(args.networks)} (times uniform in "
          f"[{args.tmin:g}, {args.tmax:g}], seed {args.seed}) -> {outdir}")
    for net_index, network in enumerate(NETWORKS):   # fixed order -> stable seeds
        if network not in args.networks:
            continue
        defaults = NETWORK_DEFAULTS.get(network, {})
        cfg = SimConfig(cells=int(args.cells or defaults.get("cells", SimConfig.cells)),
                        runs=int(args.runs or defaults.get("runs", SimConfig.runs)),
                        t_min=args.tmin, t_max=args.tmax, burnin=args.burnin, seed=args.seed,
                        time_design=args.time_design, n_snapshots=args.snapshots,
                        grid_ratio=args.grid_ratio)
        if cfg.cells < 1 or cfg.runs < 1:
            raise SystemExit("--cells and --runs must be positive")
        if verbose:
            print(f" * {network}: {cfg.cells} cells x {cfg.runs} runs")
        simulate_network(network, cfg, outdir, net_index,
                         overwrite=args.overwrite, verbose=verbose)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
