# Quick Start

This guide walks through a complete CardamomOT analysis on your own time-course scRNA-seq dataset.

## Prepare your data

CardamomOT reads a single **AnnData** file (`h5ad` format). The required metadata fields are:

| Field | Type | Description |
|---|---|---|
| `adata.obs['time']` | float | Measurement time for each cell |
| `adata.obs['cell_type']` | str | Cell type label (optional but recommended) |
| `adata.X` | matrix | Raw or normalised count matrix |

Organise your project folder as follows:

```
my_project/
└── Data/
    └── data.h5ad
```

## Run the interactive pipeline

The simplest entry point is the interactive `run` command, which presents a checkbox menu for each analysis step:

```bash
cardamomot run my_project/
```

Steps (checked by default unless marked *optional*):

| Step | Description | Default |
|---|---|---|
| Read-depth correction | Compute per-cell read-depth factors | optional |
| **Gene selection** | Filter DE genes; split cells into train/test | ✓ |
| Network constraint | Build prior network from databases | optional |
| **Kinetics** | Estimate mRNA degradation and synthesis rates | ✓ |
| **Mixture model** | Fit negative-binomial burst parameters per gene | ✓ |
| Check mixture | Validate mixture against data | ✓ |
| **Network inference** | Learn regulatory interactions via optimal transport | ✓ |
| **Network adaptation** | Prepare network parameters for simulation | ✓ |
| **Simulation** | Generate synthetic single-cell trajectories | ✓ |
| Check simulation | Validate simulations vs data | ✓ |
| Test — inference | Infer and simulate on held-out test set | optional |
| Test — check | Compare test predictions to training observations | optional |
| **Perturb (KO/OV)** | Simulate in-silico knock-outs / over-expressions | ✓ |
| Check KO/OV | Compare perturbations to wild-type simulation | ✓ |

To run all steps with default parameters without any prompt:

```bash
cardamomot run my_project/ --default
```

## Run in batch mode

For scripting or cluster submission, use the `pipeline` sub-command. **Only `-i` is required**; every other argument has a default inherited from the model (`base.py`):

```bash
# Minimal call — all parameters use model defaults
cardamomot pipeline -i my_project

# Full explicit call
cardamomot pipeline \
    -i my_project \
    -s full \                       # dataset split: full | train  (default: full)
    -c 0 \                          # differential gene selection (0=off, 1=on)  (default: 0)
    -r 1 \                          # cell-selection split rate  (default: 1)
    --mean-forcing 0.5 \            # mean-forcing intensity for NB mixture  (default: 0.5)
    --stimulus 1.0 \                # stimulus-edge penalisation in [0,1]
    --prior 1.0 \                   # prior-network weighting in [0,1]
    --force-basins 1.0 \            # preserve NB mode means in [0,1]
    --temporal-basins 1             # enforce temporal mode consistency (0 or 1)
```

**Optional-section flags** — these are switches with no value; just add the flag to change the behaviour:

| Flag | Step(s) triggered | Behaviour without flag | Behaviour with flag |
|---|---|---|---|
| `--rd` | `infer_rd` | skipped | enabled |
| `--ref` | `prepare_reference_network` | skipped | enabled |
| `--ref-depth N` | *(used with `--ref`)* | `3` (default) | path length set to `N` |
| `--test` | `infer_test` + `check_test_to_train` | skipped | enabled |
| `--no-kov` | `simulate_network_KOV` + `check_KOV_to_sim` | enabled | skipped |

## Run individual steps

Each step can be run independently with `cardamomot step <script_name> [args]`, using the exact same arguments as in `run.sh`. The script name is the filename without `.py`:

```bash
# ── Optional: read-depth correction (run before gene selection) ───────────────
cardamomot step infer_rd -i my_project

# ── Gene selection and cell split ─────────────────────────────────────────────
cardamomot step select_DEgenes_and_split \
    -i my_project -s full -r 1 -c 0 --mean-forcing 0.5 --force-basins 1.0 --temporal-basins 1

# ── Optional: prior network (run after gene selection) ────────────────────────
cardamomot step prepare_reference_network -i my_project -d 3

# ── Kinetics ──────────────────────────────────────────────────────────────────
cardamomot step get_kinetic_rates -i my_project -s full

# ── Mixture model ─────────────────────────────────────────────────────────────
cardamomot step infer_mixture \
    -i my_project -s full --mean-forcing 0.5 --force-basins 1.0 --temporal-basins 1
cardamomot step check_mixture_to_data -i my_project -s full

# ── Network inference ─────────────────────────────────────────────────────────
# --stimulus and --prior must be identical in both commands:
#   infer_network_structure uses them to constrain what edges are *learned*
#   infer_network_simul     uses them to build the *simulation* reference network
cardamomot step infer_network_structure \
    -i my_project -s full --stimulus 1.0 --prior 1.0 --force-basins 1.0 --temporal-basins 1
cardamomot step infer_network_simul \
    -i my_project -s full --stimulus 1.0 --prior 1.0

# ── Simulation ────────────────────────────────────────────────────────────────
cardamomot step simulate_network -i my_project -s full
cardamomot step check_sim_to_data \
    -i my_project -s full --stimulus 1.0 --prior 1.0

# ── Optional: test set (requires -s train) ────────────────────────────────────
cardamomot step infer_test \
    -i my_project --stimulus 1.0 --prior 1.0 --force-basins 1.0 --temporal-basins 1
cardamomot step check_test_to_train \
    -i my_project -s train --stimulus 1.0 --prior 1.0

# ── Perturbations (default) ───────────────────────────────────────────────────
cardamomot step simulate_network_KOV -i my_project -s full
cardamomot step check_KOV_to_sim \
    -i my_project -s full --stimulus 1.0 --prior 1.0
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
