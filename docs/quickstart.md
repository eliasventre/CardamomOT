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

For scripting or cluster submission, use the `pipeline` sub-command:

```bash
cardamomot pipeline \
    -i my_project \
    -s full \                  # dataset split: full | train
    -c 0 \                     # differential gene selection (0=off, 1=on)
    -r 1.0 \                   # cell-selection split rate (default 1)
    -m 1.0 \                   # mean expression threshold (-1=auto)
    --stimulus 1.0 \           # stimulus-edge penalisation in [0,1]
    --prior 1.0 \              # prior-network weighting in [0,1]
    --force-basins 1.0 \       # preserve NB mode means in [0,1]
    --temporal-basins 1        # enforce temporal mode consistency (0 or 1)
```

**Optional-section flags** (add any combination):

| Flag | Effect |
|---|---|
| `--rd` | Enable read-depth correction step |
| `--ref` | Enable prior-network preparation step |
| `--test` | Enable test-set inference steps |
| `--no-kov` | Disable KO/OV perturbation steps |

## Run individual steps

```bash
# Example: re-run only network inference
cardamomot step -i my_project network_inference
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

The `utils/` directory contains Jupyter notebooks for the standard post-pipeline analyses:

| Notebook | What it does |
|---|---|
| `plot_networks.ipynb` | Visualise the inferred GRN (edge weights, thresholding) |
| `plot_data_to_sim.ipynb` | Compare observed data to simulated trajectories (UMAPs, marginals) |
| `plot_data_to_sim_KOV.ipynb` | Compare wild-type and KO/OV simulations |
| `compare_cell_types.ipynb` | Train a cell-type classifier and compare proportions across conditions |
| `compare_cell_types_across_KOV.ipynb` | Cell-type proportions under each in-silico perturbation |

These notebooks use the high-level functions exported by the package:

```python
import anndata as ad
from CardamomOT import (
    train_classifier, predict_cell_types, plot_cell_type_proportions,
    compare_marginals, plot_data_pmf_temporal,
    plot_data_umap_toref, plot_data_umap_altogether,
    animate_dynamic_grns,
)

p = "my_project/"

# Load observed data and simulations
adata      = ad.read_h5ad(p + "Data/data_full.h5ad")
adata_sim  = ad.read_h5ad(p + "cardamomOT/adata_prot_simul_stim1.0_prior1.0.h5ad")

# Compare marginal distributions per gene and time point
compare_marginals(adata, adata_sim)

# Cell-type classification and proportion comparison
clf = train_classifier(adata, label_key="cell_type")
predict_cell_types(clf, adata_sim)
plot_cell_type_proportions(adata, adata_sim)
```

See the [API reference](api.md) for all parameters.
