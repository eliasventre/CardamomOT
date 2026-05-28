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

Steps available:

1. **Read-depth correction** — normalise sequencing depth across cells
2. **Mixture model** — fit negative-binomial burst parameters per gene
3. **Gene selection** — retain the most informative genes
4. **Network structure** — inject prior knowledge (optional)
5. **Network inference** — learn regulatory interactions via optimal transport
6. **Network adaptation** — prepare the network for simulation
7. **Simulation** — generate synthetic single-cell trajectories

To run all steps with default parameters without any prompt:

```bash
cardamomot run my_project/ --default
```

## Run in batch mode

For scripting or cluster submission, use the `pipeline` sub-command:

```bash
cardamomot pipeline \
    -i my_project \
    -s full \       # dataset split: full | train
    -c 1 \          # differential gene selection
    -r 0.6 \        # kinetics rate parameter
    -m 0.5          # mean expression threshold
```

To activate stimulus-schedule handling (e.g. for datasets with non-trivial perturbation timing):

```bash
cardamomot pipeline -i my_project --stimulus 0.5
```

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
