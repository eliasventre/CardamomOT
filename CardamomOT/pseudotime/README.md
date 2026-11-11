# Pseudotime datasets

Simulated single-cell datasets with **continuous, per-cell sampling times**, used to study
network inference along a pseudotime (i.e. when the time of each cell is not available, or
is only known through an estimated pseudotime).

> The models that infer the network *without* the time labels (three options: partial OT from the
> root population, path-dependent TSP, mechanistic Palantir) are specified in
> [`PLAN.md`](PLAN.md).

## Inference side (work in progress)

`binning.py` turns a per-cell pseudotime into the bins the CardamomOT cycle expects, and
`base_pseudotime.py` provides `PseudotimeNetworkModel` (a `NetworkModel` subclass, base code
untouched) whose time labels are those bins:

```python
import anndata as ad
from CardamomOT.pseudotime import PseudotimeNetworkModel

data = ad.read_h5ad("CardamomOT/pseudotime/data/BN8/Data/data_1.h5ad")
tau = PseudotimeNetworkModel.tau_from_time(data.obs["time"])   # oracle; the options will replace it
model = PseudotimeNetworkModel(n_genes=data.n_vars)
model.fit_mixture_on_bins(data, tau=tau, n_bins=10, n_iter=1)   # NB mixture fitted on the bins
model.bin_masses()          # (K, G+1, M) per-bin mode masses -> target marginal of the basin EMD
model.bin_expected_means()  # mean implied by the masses, to check the mean-preservation per bin
model.force_basins_at(1)    # ramp of force_basins (0.3 -> force_basins over min_n_loops)
```

The three guards against the bins -> mixture -> bins circularity are the parameters
`pseudotime_refit_mixture` / `pseudotime_refit_every` (re-fit every 5 iterations via
`refit_mixture_if_due`) and `pseudotime_force_basins_start` (`force_basins_at`).

### Ordering step (Option 3: mechanistic Palantir)

```python
model.initialise_cell_states(data)                 # counts + per-cell proteins (equilibrium of the modes)
tau = model.update_pseudotime(n_iter=1, roots=data.obs['is_root'].values)
model.bins, model.pseudotime_diagnostics            # bins, and the run diagnostics
```

Parameters: `palantir_feature` (`'posterior'` = expected burst modes, the default; `'hard'`;
`'expression'`), `palantir_edge_weight` (`'geometric'` default - best while the network is still
being fitted, `'plausibility'`, `'mechanistic'`, `'auto'`), `palantir_n_neighbors` (60; a denser graph
gives a smoother tau and markedly better edges, PLAN.md §0.7), `palantir_n_waypoints` (50),
`palantir_consensus` (`'min'` = tightest perspective, default; `'mean'` = Palantir weighted mean),
`palantir_geom_space` (`'expression'` default; `'modes'`). The mechanistic
weights use the same bridge as the E-step (`costs.bridge_end` = `find_next_prot`) and are meant for
iterations where the network is already fitted.

Validation of the ordering against the true time (with a diffusion-pseudotime baseline):

```bash
conda run -n cardamom_env python CardamomOT/pseudotime/check_pseudotime_order.py \
    --networks CN5 BN8 FN8 --runs 1 --iterations 0 1
# figures + metrics in CardamomOT/pseudotime/data/_pseudotime_order/
conda run -n cardamom_env python -m pytest tests/test_pseudotime_binning.py tests/test_pseudotime_order.py -q
```

### Option 1: partial OT from the root population

`pseudotime_method` selects the ordering step (`PLAN.md` §5):

| `pseudotime_method` | file | pseudotime |
|---|---|---|
| `'palantir'` (default) | `options/palantir_like.py` | shortest path in the GRN-weighted kNN graph |
| `'partial_ot'` | `options/partial_ot.py` | waves of partial OT from the root population |
| `'tsp'` | `options/tsp.py` | path-dependent TSP refined by a windowed 2-opt |

Option 1 moves a fraction `mass_per_step` (β) of the mass of the current frontier onto the still
unexplained cells, one wave at a time, with the same mechanistic cost as the E-step:

```python
model.pseudotime_method = 'partial_ot'
model.partial_ot_mass_per_step = 0.25        # beta: width of the wave
model.partial_ot_edge_weight = 'mechanistic' # 'plausibility' | 'geometric'
model.partial_ot_alpha_geom = 1.0
model.partial_ot_restrict = 'knn'            # candidate pool: kNN of the frontier | 'all'
model.partial_ot_partial = 'mass'            # 'mass' (fixed beta) | 'lagrange' (phantom column)
model.partial_ot_frontier_cap = 200          # leading-edge cells solved per wave (cost |F| x |T|)
model.fit_pseudotime(data, verb=True, max_iter=12)
model.pseudotime_diagnostics['n_steps'], ['n_unreached'], ['wave_sizes']
```

Every cell receives a pseudotime (`n_unreached` counts the cells that only got the final
neighbour fill, `n_fallback` the waves where the candidate pool had to be widened), and `tau` stays
an absolute coordinate: the step is mass-weighted, `tau_{c'} = sum_c gamma(c,c')(tau_c + dtau) /
sum_c gamma(c,c')`, not a rank.

`alpha_geom` is the important knob: its geometric term is what keeps the waves anchored to
*expression continuity*, so a small value lets the freshly fitted network override the ordering
(see `PLAN.md` §0.8). Measured on the simulated networks (free τ, 3 iterations): Option 1 gives a
slightly worse pseudotime than Option 3 on every dataset but **better edges on the branching ones**
(CH20 AUROC 0.923 vs 0.887, RG50 0.584 vs 0.563), so `palantir` remains the default and
`partial_ot` is the method of choice for a tree-like dataset.

```bash
conda run -n cardamom_env python -m pytest tests/test_pseudotime_partial_ot.py -q
```

### Option 2: path-dependent TSP

A tour through the cells, refined by a windowed 2-opt, whose step cost depends on the state reached
at the previous step:

```python
model.pseudotime_method = 'tsp'
model.tsp_init = 'palantir'          # 'palantir' | 'partial_ot' | 'previous'
model.tsp_n_moves = 1000             # random windows tried per EM iteration
model.tsp_window = 40                # longest segment reversed by one move
model.tsp_edge_weight = 'mechanistic'
model.tsp_use_path_length = True     # tau = cumulative cost (False: the tour rank)
model.fit_pseudotime(data, verb=True, max_iter=12)
model.pseudotime_diagnostics['n_moves_accepted'], ['tour_improvement']
```

The tour always starts from the pseudotime of the previous iteration (warm start), and from another
option when there is none yet — a TSP started from scratch has no chance of finding the direction.
Reversing a segment changes every edge inside it, so the proteins are re-propagated from the start
of the window: that is why the moves are windowed (`tsp_window`).

The pseudotime is the **cumulative path length** (a pseudo-metric, which the bins and the growth
pass need) rather than the tour rank. A single tour cannot represent a branching trajectory — the
arms of a tree are interleaved along one path — so Option 2 is a *refinement* of Options 1/3 rather
than a competitor: measured on the simulated networks it shortens the warm-started tour by only
1–4 % and ends below Option 3 on every dataset, though on the cyclic `CN5` it gives the best
transient correlation of the three (a tour is a cycle). See `PLAN.md` §0.9.

```bash
conda run -n cardamom_env python -m pytest tests/test_pseudotime_tsp.py -q
```

### Growth, degradations and simulations

The rest of the pipeline (growth pass, `refine_network_degradations`, `simulate_network`) runs
unchanged on the pseudotime bins, because the loop leaves a *balanced* design: `times_data =
np.repeat(bin_times, n_full)` with `prot` / `kon_beta` / `alpha` on the same slots.

```python
model.fit_pseudotime(data, verb=True, max_iter=12, refine=True)
# ... growth (R_opt) -> refine (network refit, d1/d0, ProliferationMLP) -> simulate
model.simulate_network(list(np.sort(np.unique(model.times_data))), verb=False)

from CardamomOT.pseudotime.evaluate import marginal_agreement, degradation_metrics
decl = degradation_metrics(model, data.uns['true_d'])   # d1 recovered, d0/d1 ratio
marg = marginal_agreement(model, data)                 # simulated vs observed mRNA, per bin/gene
marg['frac_within_2x']
```

`marginal_agreement` compares the expected count of every simulated state,
`kon * kz / c` with `kz = max(sigmoid_amplitudes(ks)) * k1` (the mean of
`negative_binomial(kz*kon, c/(c+1))`), with the observed counts of the same bin. Note that the base
refinement only re-estimates `d1`; the ratio `d0/d1` is `model.ratios`, and `model.d_t[:, 0, :]`
keeps its prior.

```bash
conda run -n cardamom_env python -m CardamomOT.pseudotime.check_pseudotime_simul \
    --networks CN5 BN8 FN8 --runs 1 --iterations 3
# d1 / marginals vs the truth -> CardamomOT/pseudotime/data/_pseudotime_simul/metrics.json
```

### Benchmark: two sampling designs

`simulate_pseudotime_data.py --time-design grid` writes a second dataset per network, where the
cells come from **10 unequally spaced library times** (gaps in a geometric progression of ratio 1.3,
equal cells per snapshot) instead of one continuous time per cell:

```bash
conda run -n cardamom_env python -m CardamomOT.pseudotime.simulate_pseudotime_data \
    --networks CN5 BN8 FN8 --runs 3 --time-design grid --snapshots 10 --grid-ratio 1.3
# -> data/<NET>/Data_grid/  (+ uns['snapshot_times'], obs['snapshot'])
```

With a discrete truth, Spearman is tie-attenuated, so `evaluate.snapshot_metrics` reports the
pairwise **order accuracy**, the **separation** of the snapshots (do their tau ranges overlap?) and
the **recovered gap ratios** (is the spacing recovered, not just the order?).

```bash
conda run -n cardamom_env python -m CardamomOT.pseudotime.check_benchmark \
    --networks CN5 BN8 FN8 --runs 1 2 3 --designs continuous grid --iterations 3
# results.json + table.md + aggregate.md -> data/_benchmark/
# add --append to regenerate the tables from an existing results.json without refitting
```

Methods compared: the 3 options (full cycle) and the baselines `oracle` (true time), `palantir_pca`
(geometric Palantir on log counts, no network) and `random`, all three with τ imposed and the E/M
cycle run with `pseudotime_fixed_tau`. The `stat` (no-time) baseline of `PLAN.md` is not available —
see `PLAN.md` §0.12 finding 8.

**What the benchmark says** (§0.12): on an irregular design the methods recover the **order** of the
libraries (0.76–0.99 of the pairs, vs 0.49–0.57 for random) but **not their spacing** (0 gap ratios
within 2x, against 1.00 for the oracle), and no two non-root snapshots are separated by their tau
ranges. Across 3 runs the three options are statistically indistinguishable on the network, so the
default stays `palantir` and the choice should be made on runtime and on the dataset structure.

### Method schematic

`figure_methods.py` draws the four variants side by side — the calculation steps of each, the
machinery they share, and the two loops that link them (τ → bins; Θ and the cell proteins → the
ordering of the next iteration):

```bash
conda run -n cardamom_env python CardamomOT/pseudotime/figure_methods.py
# -> data/_figures/methods_schematic.pdf (+ .png), vector, for the article
```

### The joint cycle

```python
model = PseudotimeNetworkModel(n_genes=data.n_vars)
model.seed = 0
model.pseudotime_bins = 10
model.fit_pseudotime(data, verb=True, max_iter=12)   # mixture -> ordering -> E/M/alpha/basins
model.tau            # inferred pseudotime
model.inter          # inferred network
model.pseudotime_history   # per-iteration error, tau stability, weights, force_basins
model.R_opt          # net growth per unit pseudotime (growth pass)
```

`pseudotime_fixed_tau=True` freezes the pseudotime (oracle validation on the true time). The edge
weights are `plausibility` while the network is unknown and `mechanistic` from the first fitted
network on (`palantir_edge_weight='auto'`).

The **time unit** of the axis (`PLAN.md` §4.3) is set by `pseudotime_duration_mode`: `'prior'`
(default, each bin spans one protein relaxation time `1/d1`), `'fixed'` (uses
`pseudotime_total_duration`) or `'grid'` (selects the bin width by the model error). The degradation
rates are read from `adata.var['d1']` / `['d0']` when present, as in the figure pipelines.

Six simulated networks are available: `CN5`, `BN8`, `FN8` (2000 cells × 5 runs), and the larger
`CH20` (20 genes, 4-arm tree), `RG50` (50 genes, sparse random DAG), `MD100` (100 genes, 5 modules
in cascade) with per-network cell/run defaults.

```bash
conda run -n cardamom_env python CardamomOT/pseudotime/check_pseudotime_fit.py \
    --networks CN5 BN8 FN8 --runs 1 --bins 10 --iterations 12
# AUPRC / AUROC / top-k, tau vs true time, growth rate -> data/_pseudotime_fit/
```

Validation of the mixture-on-bins against a pooled mixture on the simulated datasets:

```bash
conda run -n cardamom_env python CardamomOT/pseudotime/check_mixture_bins.py \
    --networks CN5 BN8 FN8 --runs 1 --bins 10
# figures + metrics in CardamomOT/pseudotime/data/_mixture_bins/
conda run -n cardamom_env python -m pytest tests/test_pseudotime_binning.py -q
```

Everything is produced by [`simulate_pseudotime_data.py`](simulate_pseudotime_data.py), which
reuses the ground-truth networks and kinetic parameters of
`results_article/figures/simulate_data_{CN5,BN8,FN8}.py` and the bursty PDMP model of
[harissa](https://github.com/eliasventre/harissa).

## Difference with the figure datasets

| | `results_article/figures` | here |
|---|---|---|
| cell times | 10 fixed time points `{0, 6, ..., 96}`, 100 cells each | each cell has its own time, drawn uniformly in `[t_min, t_max]` |
| replicates | exact replicates of the same time point | none (continuous times) |
| cell order | sorted by time | shuffled |
| true time | binned (and used as a label) | kept per cell in `obs['time']` |

The true time of each cell is the **ground truth** that an inference of the pseudotime
without time information will be compared to.

## Usage

`harissa` is needed for the simulation (it is available in the `cardamom_env` and `CardaWM`
conda environments, not in `base`):

```bash
# default: CN5, BN8 and FN8, 2000 cells x 5 runs, times uniform in [0, 96], seed 0
conda run -n cardamom_env python CardamomOT/pseudotime/simulate_pseudotime_data.py

# a subset, with custom settings
conda run -n cardamom_env python CardamomOT/pseudotime/simulate_pseudotime_data.py \
    --networks CN5 BN8 --cells 500 --runs 2 --tmax 48 --seed 1

# quick test dataset
conda run -n cardamom_env python CardamomOT/pseudotime/simulate_pseudotime_data.py \
    --networks CN5 --cells 200 --runs 1 --tmax 24
```

Useful options: `--cells`, `--runs`, `--tmin`, `--tmax`, `--burnin`, `--seed`, `--outdir`,
`--overwrite` (by default, runs whose files already exist are skipped), `--quiet`.
The script is also runnable as a module:
`python -m CardamomOT.pseudotime.simulate_pseudotime_data`.

Reproducibility: each (network, run) has its own random stream derived from `--seed`, and
harissa's global RNG is seeded before each run, so a given command always produces the same
files.

## Outputs

For each network `NET` (`CN5`, `BN8`, `FN8`) and run `r = 1..N`, in `data/<NET>/`:

```
Data/data_{r}.h5ad          AnnData, cells x genes
    X                       raw counts (int32), genes only (the stimulus node is not a gene)
    var_names               G1..GG
    obs['time']             true simulated time of the cell (ground-truth pseudotime)
    obs['stimulus']         stimulus level, 100 * (time > 0) as in the figure datasets
    obs['run']              run index
    uns['true_inter']       true interaction matrix, (G+1) x (G+1), node 0 = stimulus
    uns['true_network']     binarised version of true_inter
    uns['true_a'], uns['true_d'], uns['true_basal']   true kinetic parameters
    uns['node_names']       ['Stimulus', 'G1', ..., 'GG']

Data/data_{r}.txt           PIDC-like table, same layout as results_article/figures/<NET>/Data
    row 0                   time (floats)
    row 1                   stimulus
    rows 2..                counts of genes 1..G
    column 0                labels
    -> read the time row with dtype=float and the other rows with dtype=int

True/inter_{r}.npy          binarised ground-truth network, (G+1) x (G+1)
True/degradation_rates.txt  true degradation rates, d.T (one row per node, 2 columns)
True/network.npz            a, d, basal, inter, inter_binary, node_names
metadata.json               all the simulation settings of the dataset
```

Example:

```python
import anndata as ad
import numpy as np

adata = ad.read_h5ad("CardamomOT/pseudotime/data/BN8/Data/data_1.h5ad")
adata.obs["time"]            # true time of each (shuffled) cell
adata.uns["true_network"]    # ground-truth network to score the inference against

# the .txt files keep the format of the figure datasets
path = "CardamomOT/pseudotime/data/BN8/Data/data_1.txt"
time = np.loadtxt(path, delimiter="\t")[0, 1:]            # float times
data = np.loadtxt(path, delimiter="\t")[1:, 1:].astype(int)  # stimulus + counts
```

## Notes

* Each cell is simulated independently: burn-in without stimulus (`--burnin`, 5 by default),
  then stimulus activation, then simulation up to the time of the cell, and counts drawn
  from a Poisson law of mean the simulated mRNA level at that time.
* With `--tmax 96` (the range of the figure datasets) the cells span both the transient
  response and the steady state; lower `--tmax` to concentrate on the transient, raise it to
  put more cells at stationarity.
* Adding a network: write a `build_<name>()` function returning a configured
  `harissa.NetworkModel` and register it in `NETWORKS` of `simulate_pseudotime_data.py`.
