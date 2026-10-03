# Quick Start

This guide walks through a complete CardamomOT analysis on your own time-course scRNA-seq dataset.

## Prepare your data

CardamomOT reads a single **AnnData** file (`h5ad` format). The required metadata fields are:

| Field | Type | Description |
|---|---|---|
| `adata.obs['time']` | float | Measurement time for each cell. If absent or unique, the data are treated as stationary: gene selection (on cell types only) and mixture inference run, then the pipeline stops — network inference will be handled by CardamomOT-stat (in prep.) |
| `adata.obs['cell_type']` | str | Cell type label (optional but recommended), used for DE gene selection |
| `adata.obs['cell_type_selection']` | str | Optional override of `cell_type` for DE gene selection |
| `adata.obs['cell_type_proliferation']` | str | Optional, only used with `Data/proliferation_rates` to anchor the literature proliferation estimate ([details](advanced.md)) |
| `adata.obs['cell_type_transition']` | str | Optional, only used with `Data/transition_rates` to structure the OT ([details](advanced.md)) |

`cell_type_proliferation` and `cell_type_transition` may differ (e.g. a finer grouping for proliferation). If only one of them is defined, it is used for both; if neither is, `cell_type` is used. Both anchorings are all-or-nothing: if any cell type is missing from the corresponding file, a warning is printed and the anchoring is skipped for all cells.
| `adata.X` | matrix | Raw or normalised count matrix |

Organise your project folder as follows:

```
my_project/
└── Data/
    └── data.h5ad
```

## Run the interactive pipeline

The simplest entry point is the interactive `run` command: it asks once for the hard-to-calibrate parameters (empty = workbook value or default), then for each step (Enter keeps the answer preselected from the project parameters):

```bash
cardamomot run my_project/
```

Steps (preselected according to the parameters in parentheses):

| Step | Description | Preselected |
|---|---|---|
| **Cell depth** | Per-cell depth diagnostic | ✓ |
| Proliferation rates | Net proliferation rate per cell from literature gene signatures (on the full gene set) | `estimate_proliferation_rates` |
| **Gene selection** | Select genes (`select_genes`); split cells into train/test (`split`, `train_rate`) | ✓ |
| Network constraint | Build prior network from databases | `build_prior_network` and not `select_genes` |
| **Kinetics** | Assign literature mRNA/protein degradation rates (h⁻¹), species auto-detected | ✓ |
| **Mixture model** | Fit negative-binomial burst parameters per gene | ✓ |
| Check mixture | Validate mixture against data | ✓ |
| **Network inference** | Learn regulatory interactions via optimal transport | ✓ |
| **Network adaptation** | Prepare network parameters for simulation | ✓ |
| **Simulation** | Generate synthetic single-cell trajectories | ✓ |
| Check simulation | Validate simulations vs data | ✓ |
| Test — inference | Infer and simulate on held-out test set | `run_test` (with `split = 'train'`) |
| Test — check | Compare test predictions to training observations | `run_test` |
| Perturb (KO/OV) | Simulate in-silico knock-outs / over-expressions / stimuli | `simulate_perturbations` |
| Check KO/OV | Compare perturbations to wild-type simulation | `simulate_perturbations` |
| **Report** | Final PDF | ✓ |

To run the preselected steps with the project parameters, without any prompt:

```bash
cardamomot run my_project/ --default
```

```{note}
The **Proliferation rates** step (`estimate_proliferation_rates = True` in the `Model_parameters`
sheet) scores each cell against built-in proliferation/death marker genes and writes
`adata.obs['proliferation_net_rate']`, which the network-inference step then uses to
correct the optimal-transport marginals for cell growth/death. It runs on the full,
unfiltered dataset — *before* gene selection — so that DE gene filtering doesn't
discard the literature marker genes needed to score the signature. It always
(re)computes and overwrites `adata.obs['proliferation_net_rate']`; keep the parameter
`False` (default) to use your own values (e.g. from EdU staining). See
[Advanced Features](advanced.md#refining-proliferation-rates) to use mouse gene
sets, supply your own marker genes, or anchor the estimate to a known
population-level rate.
```

## Run in batch mode

For scripting or cluster submission, use the `pipeline` sub-command (or `run.sh`). **Only `-i` is required**,
and the only options are the hard-to-calibrate parameters, in this order:

```bash
# Minimal call — workbook values, else defaults of base.py
cardamomot pipeline -i my_project

# Full explicit call
cardamomot pipeline \
    -i my_project \
    --stimulus 1.0 \          # stimulus-edge penalisation in [0,1]            (model.stimulus)
    --prior 1.0 \             # weight of edges absent from the prior in [0,1] (model.prior_network_pen)
    --mean-forcing 0.5 \      # mean-forcing intensity of the NB mixture       (model.mean_forcing_em)
    --force-basins 1.0 \      # basin weights kept in the network fit, [0,1]   (model.force_basins)
    --temporal-basins 1        # basin weights per timepoint (0 or 1)           (model.temporal_basins)

# Same with run.sh (positional, same order; empty or -1 = workbook value / default)
./run.sh my_project 1.0 1.0 0.5 1.0 1
```

**Everything else is a parameter** of `NetworkModel` (`CardamomOT/model/base.py`), fixed per project in the
`Model_parameters` sheet of `Data/CardamomOT_inputs.xlsx` (precedence: default < workbook < command line):

| Parameter | Default | Steps |
|---|---|---|
| `split` (`'train'` / `'full'`), `train_rate` | `'train'`, `0.7` | train/test split of the cells (all steps read `data_<split>.h5ad`) |
| `select_genes` | `False` | gene selection in `select_genes_and_split` (otherwise all genes kept) |
| `build_prior_network` | `False` | literature prior: by the selection, or `build_reference_network` if `select_genes = False` |
| `estimate_proliferation_rates` | `False` | `get_proliferation_rates` |
| `run_test` | `False` | `infer_test` + `check_test_to_train` (needs `split = 'train'`) |
| `simulate_perturbations` | `True` | `simulate_network_KOV` + `check_KOV_to_sim` |
| `simulate_with_proliferation` | `False` | proliferation MLP (`infer_network_simul`) and branching simulations |
| `species` | `'auto'` | degradation rates, proliferation signatures, literature prior |

## Run individual steps

Each step takes `-i` and only the hard-to-calibrate options it uses (`cardamomot step <script_name> [args]`,
script name without `.py`); a removed option stops the step with the parameter to set instead:

```bash
cardamomot step estimate_cell_depth     -i my_project
cardamomot step get_proliferation_rates -i my_project            # if estimate_proliferation_rates
cardamomot step select_genes_and_split  -i my_project --prior 1.0
cardamomot step build_reference_network -i my_project            # if build_prior_network and not select_genes
cardamomot step get_degradation_rates   -i my_project            # d0/d1 of the species (overwrite_degradation_rates)
cardamomot step infer_mixture           -i my_project --mean-forcing 0.5
cardamomot step check_mixture_to_data   -i my_project

# --stimulus and --prior must be identical in the network steps, the checks and the report (file tags)
cardamomot step infer_network_structure -i my_project --stimulus 1.0 --prior 1.0 --force-basins 1.0 --temporal-basins 1
cardamomot step infer_network_simul     -i my_project --stimulus 1.0 --prior 1.0   # + MLP if simulate_with_proliferation
cardamomot step simulate_network        -i my_project
cardamomot step check_sim_to_data       -i my_project --stimulus 1.0 --prior 1.0

# Held-out validation (run_test, split = 'train'): test cells classified with the training mixtures, trajectory
# loop with the network fixed continuing the training schedule, simulation from the test cells at t0, compared
# to Data/data_test.h5ad (Check/ and section 6 of the report); test cells per (time, sample) capped at the train ones.
cardamomot step infer_test              -i my_project --stimulus 1.0 --prior 1.0 --force-basins 1.0 --temporal-basins 1
cardamomot step check_test_to_train     -i my_project --stimulus 1.0 --prior 1.0

# Perturbations (simulate_perturbations)
cardamomot step simulate_network_KOV    -i my_project
cardamomot step check_KOV_to_sim        -i my_project --stimulus 1.0 --prior 1.0
cardamomot step report_results          -i my_project --stimulus 1.0 --prior 1.0
```

## Examine results

Results land in `my_project/cardamomOT/`:

```
my_project/
├── cardamomOT/
│   ├── adata_beta_stim<s>_prior<p>.h5ad      # kinetic + network parameters
│   ├── adata_rna_traj_stim<s>_prior<p>.h5ad  # inferred RNA trajectories
│   ├── adata_prot_simul_stim<s>_prior<p>.h5ad # simulated protein levels
│   └── adata_prot_simul_KO_<gene>_*.h5ad     # in-silico perturbation outputs
└── Check/                                     # diagnostic figures
```

## Post-analysis

The `utils/` directory contains Jupyter notebooks that call the post-analysis functions
exported by the package. You can also call these functions directly in your own scripts.

| Notebook | What it does |
|---|---|
| `plot_networks.ipynb` | Inferred GRN — per-regulator subgraphs and reduced network (`plot_network`) |
| `plot_data_to_sim.ipynb` | Compare data, NB mixture, trajectories and simulation (UMAPs) |
| `plot_data_to_sim_KOV.ipynb` | Compare wild-type simulation to KO/OV perturbations (`plot_results_sim_kov`) |
| `compare_cell_types.ipynb` | Train cell-type classifier and compare proportions across stages |
| `compare_cell_types_across_KOV.ipynb` | Cell-type proportions under each in-silico perturbation (`compare_cell_types`) |

### Typical workflow

```python
import anndata as ad
from CardamomOT import (
    train_classifier,
    check_cell_types_mixture,
    check_cell_types_full,
    plot_results_rna_mixture,
    plot_results_rna_clean,
    plot_results_prot,
    plot_network,
    plot_results_sim_kov,
    compare_cell_types,
)

p = "my_project/"   # trailing slash required
split = "full"      # dataset split used during the pipeline ("full" or "train")
stim, prior = 1.0, 1.0  # match the values passed to the pipeline
label = "cell_type"

# ── Cell-type characterisation ───────────────────────────────────────────────
adata_full = ad.read_h5ad(p + f"Data/data_{split}.h5ad")
clf = train_classifier(adata_full, label_key=label)
check_cell_types_mixture(clf, p, adata_full)
check_cell_types_full(clf, p, stim=stim, prior=prior)

# ── UMAP comparisons ─────────────────────────────────────────────────────────
plot_results_rna_mixture(split, p)
plot_results_rna_clean(split, p, stim=stim, prior=prior,
                       normtransform=False, logtransform=True)
plot_results_prot(p, stim=stim, prior=prior)

# ── Inferred GRN ─────────────────────────────────────────────────────────────
plot_network(p, seuil=0, network=0, train=split)

# ── KO/OV comparison (if perturbation steps were run) ────────────────────────
combo = "KO_Gene_OV_none"
compare_cell_types(p, combo, split=split)
plot_results_sim_kov(p, combo, stim=stim, prior=prior)
```

See the [API reference](api.md) for all parameters.
