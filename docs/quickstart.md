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

Results land in `my_project/cardamom_output/`:

```
cardamom_output/
├── cardamomOT/
│   ├── adata_beta.h5ad           # kinetic parameters
│   ├── adata_rna_traj_*.h5ad     # inferred RNA trajectories
│   └── adata_prot_simul_*.h5ad   # simulated protein levels
└── *.log
```

Load and inspect the inferred model from Python:

```python
import anndata as ad
from CardamomOT import NetworkModel

# Load inferred parameters
adata_beta = ad.read_h5ad("my_project/cardamom_output/cardamomOT/adata_beta.h5ad")

# Reconstruct and simulate the network
model = NetworkModel(adata_beta)
simulation = model.simulate(n_cells=500)
```

## Python API

For fine-grained control, CardamomOT can be used entirely from Python:

```python
import anndata as ad
from CardamomOT import NetworkModel
from CardamomOT.inference import kon_ref_vector, extract_degradation_rates

adata = ad.read_h5ad("Data/data.h5ad")

# --- step 1: burst kinetics ---
adata_beta = kon_ref_vector(adata, ...)

# --- step 2: degradation rates ---
adata_beta = extract_degradation_rates(adata_beta, ...)

# --- step 3: network inference ---
model = NetworkModel(adata_beta)
model.fit(adata, ...)

# --- step 4: simulation ---
simulated = model.simulate(n_cells=1000, time_points=[0, 2, 4, 7])
```

See the [API reference](api.md) for all parameters.
