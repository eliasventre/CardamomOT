# Running CardamomOT on Your Own Data

This guide explains how to use the CardamomOT pipeline to analyze your single-cell RNA-seq dataset.

## Project Structure

First, organize your project as follows:

```
my_project/
├── Data/
│   └── data.h5ad              # Your H5AD format dataset
├── config.yaml                # (optional) Custom parameters
└── cardamom_output/           # (auto-created) Results
    ├── cardamom/              # Inferred network files
    ├── simulations/           # Simulated data
    └── figures/               # Plots and visualizations
```

## Quick Start

### 1. Interactive Mode (Recommended)

Run the pipeline interactively to select steps and customize parameters:

```bash
cardamomot run /path/to/my_project
```

This will:
1. ✓ Validate your project structure
2. ✓ Show checkboxes to select which analysis steps you want
3. ✓ Prompt for custom hyperparameters (or use defaults)
4. ✓ Execute the selected pipeline steps

### 2. Default Mode

Run the full pipeline with all default parameters:

```bash
cardamomot run /path/to/my_project --default
```

This will execute all standard steps without any interaction.

## Pipeline Steps Explained

| Step | Purpose | When to skip |
|------|---------|------------|
| **Estimate read depth** | Correct for sequencing depth bias | If your data is already normalized |
| **Infer mixture model** | Learn cell state kinetic parameters | Always include |
| **Select DE genes** | Filter genes by temporal/celltype variability | To keep all genes, skip |
| **Infer network structure** | Build reference network from prior knowledge | If no prior network available |
| **Infer network** | Learn gene regulatory interactions | Always include |
| **Adapt network** | Prepare network for simulation | If not simulating |
| **Simulate network** | Generate synthetic trajectories | For validation and visualization |

## Hyperparameters

When running interactively, you can customize:

- **`-i` / Input path**: Project directory (auto-filled)
- **`-s` / Split name**: Data split identifier (default: 'train')
- **`-n` / Number of genes**: Genes to keep after filtering (default: 1000)
- **`-c` / Change flag**: Computational setting (default: 1)
- **`-r` / Rate parameter**: Inference learning rate (default: 1.0)
- **`-m` / Mean parameter**: Mean constraint strength (default: 0.5)

For more control, copy `config_template.yaml` to your project and customize:

```bash
cp config_template.yaml /path/to/my_project/config.yaml
```

Then edit the YAML file before running the pipeline.

## Example Workflow

```bash
# 1. Initialize project
mkdir my_study
cd my_study
mkdir Data
# → Place your data.h5ad here

# 2. Run pipeline (interactive)
cardamomot run .

# 3. Select steps (checkbox prompt)
# → Choose: mixture, select genes, infer network, simulate
# → Uncheck: read depth, network structure

# 4. Enter parameters when prompted
# → Keep defaults or customize

# 5. Wait for pipeline to complete...
# ✅ Results in ./cardamom_output/
```

## Troubleshooting

### Error: "Data/ directory not found"
Make sure your project has a `Data/` subfolder with your `.h5ad` file:
```bash
mkdir -p /path/to/my_project/Data
cp my_data.h5ad /path/to/my_project/Data/
```

### Error: "Script not found: infer_*.py"
This usually means CardamomOT is not properly installed. Install from repo:
```bash
cd /path/to/CardamomOT
pip install -e .
```

### A step fails mid-pipeline
The CLI will ask if you want to continue. Choose `Y` to skip to the next step, or `N` to stop.

## Output Files

After running the pipeline, check `cardamom_output/`:

- **`cardamom/data_*.npy`**: Processed expression matrices
- **`cardamom/network*.npy`**: Inferred network interactions
- **`simulations/*.npy`**: Simulated gene expression trajectories
- **`*.log`**: Detailed execution logs
- **`figures/`**: Plots if visualization steps were included

## Next Steps

- Load and visualize results in a notebook (examples in `utils/`)
- Compare inferred and simulated data distributions
- Extract top-ranked network interactions
- Generate publication-ready figures

For detailed methods and theory, see the [CardamomOT paper](https://doi.org/10.1101/2023.).

---

Need help? Check the [CardamomOT GitHub](https://github.com/your-username/CardamomOT) or submit an issue.
