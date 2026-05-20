# CARDAMOM

CARDAMOM is an executable gene regulatory network (GRN) inference method, adapted to time-course scRNA-seq datasets. The algorithm consists in calibrating the parameters of a mechanistic model of gene expression: the calibrated model can then be simulated, which allows to reproduce the dataset used for inference. The first inference method has been introduced in [[1](#Ventre2021)]. It has been benchmarked along with other GRN inference tools and applied to a real dataset in [[2](#Ventre2023)]. The second version is presented in [[2](#Mauge2026)] and combines GRN and trajectory inference method and shows a strong improvement over the first version.  The simulation part is based on the [Harissa](https://github.com/ulysseherbach/harissa) package.

## 🚀 Quick Start

### 1. Installation

#### Create a virtual environment (recommended)

```bash
# Create a new conda environment
conda create -n cardamom_env python=3.12 -y
conda activate cardamom_env

# IMPORTANT on macOS Apple Silicon (arm64): install numba + threading runtimes from conda-forge
# This avoids pip wheels without TBB backend support.
conda install -c conda-forge numba llvmlite llvm-openmp tbb tbb-devel -y

# OR with venv (standard Python)
python -m venv cardamom_env
source cardamom_env/bin/activate  # Linux/Mac
# cardamom_env\Scripts\activate   # Windows
```

#### Install CARDAMOM

```bash
# Clone the repository
git clone https://github.com/eliasventre/CardamomOT.git
cd CardamomOT

# Install the package in development mode
pip install -e .

# For development (tests, linting):
pip install -e ".[dev]"

# For Jupyter notebooks:
pip install -e ".[notebooks]"
```

### 2. Prepare your data

CARDAMOM requires a project directory containing your scRNA-seq data. Create a folder `my_project/` with:

```
my_project/
└── Data/
    └── data.h5ad          # Your count matrix (required)
    └── gene_list.txt      # Gene list (optional)
```

**Required format for `data.h5ad`:**
- Gene counts (rows = genes, columns = cells)
- `data.obs['time']`: measurement time for each cell
- `data.obs['cell_type']`: cell types (optional)

### 3. Run the full analysis

#### Full pipeline (recommended for beginners)

```bash
# Activate the environment
conda activate cardamom_env

# Run the full pipeline
python -m CardamomOT.cli pipeline -i my_project -s full -c 1 -r 0.6 -m 0.5
```

**Important parameters:**
- `-i my_project`: path to your project directory
- `-s full`: dataset split for inference (full or train)
- `-c 1`: change flag (e.g. stimulus)
- `-r 0.6`: rate parameter for kinetics
- `-m 0.5`: mean expression threshold

#### Results

The pipeline automatically creates these directories:
- `cardamom/`: calibrated model parameters
- `Check/`: inference vs data comparisons
- `visualization/`: regulatory network analyses

## 📋 Detailed Usage

### Step-by-step (for advanced users)

Instead of using the full pipeline, you can run each step individually:

```bash
# 1. Compute degradation rates
python -m CardamomOT.cli step get_kinetic_rates -i my_project

# 2. Select differentially expressed genes
python -m CardamomOT.cli step select_DEgenes_and_split -i my_project -s full -m 0.5

# 3. Infer mixture parameters (burst kinetics)
python -m CardamomOT.cli step infer_mixture -i my_project -s full -m 0.5

# 4. Check mixture vs data consistency
python -m CardamomOT.cli step check_mixture_to_data -i my_project -s full

# 5. Infer network structure
python -m CardamomOT.cli step infer_network_structure -i my_project -s full

# 6. Simulate network
python -m CardamomOT.cli step infer_network_simul -i my_project -s full

# 7. Full simulation
python -m CardamomOT.cli step simulate_network -i my_project -s full

# 8. Final checks
python -m CardamomOT.cli step check_sim_to_data -i my_project -s full
python -m CardamomOT.cli step simulate_network_KOV -i my_project -s full
python -m CardamomOT.cli step check_KOV_to_sim -i my_project -s full
```

### Individual scripts (expert mode)

You can also run the Python scripts directly:

```bash
# Example: infer network structure only
python infer_network_structure.py -i my_project -s full

# With verbose output for debugging
python infer_mixture.py -i my_project -s full -m 0.5 --verbose
```

## 🧬 Advanced Features

This section describes optional input files that activate advanced modes of the algorithm. All optional files are placed either in `my_project/Data/` or directly in `adata.obs`.

---

### Read depth correction (`infer_rd.py`)

When cells have heterogeneous sequencing depths, CARDAMOM can correct for this before inference. Run:

```bash
python infer_rd.py -i my_project
```

This script identifies Poisson-like genes (following Chronocell, Fang et al. 2024) and computes a per-cell read depth factor stored in `adata.obs['rd']`. Subsequent steps (`infer_mixture`, `infer_network_structure`) automatically use this column when present.

**Requirements:** at least 1,000 genes. Below this threshold the step is silently skipped (no `rd` column is added).

**Key parameters:**
- `--var_threshold 1.2`: Poisson variance threshold (default 1.2)
- `--min_mean 0.1`: minimum gene mean for Poisson gene selection

---

### Multiple experimental samples (`dataset_id`)

If your experiment contains several biological conditions that should share a common gene regulatory network but may have **different basal transcription rates** (e.g. different cell lines, donors, or perturbation backgrounds), annotate each cell with a sample label:

```python
adata.obs['dataset_id'] = ...   # string or integer label per cell
```

When `dataset_id` is present, CARDAMOM automatically:
- Identifies unique samples and builds one set of **per-sample basal parameters** `θ_basal(s)` for each sample `s`.
- Solves a **single joint optimisation** over shared interaction weights `inter` and all per-sample basals simultaneously — so perturbation information is exploited during network inference.
- Saves `basal.npy` with shape `(n_samples, G, n_networks)` instead of the standard `(1, G, n_networks)`.

> Without `dataset_id`, all cells are treated as a single sample (`n_samples = 1`).

**Keeping per-sample basals close to each other (`constrain_basal_uniform`):**
By default the per-sample basals are free to diverge, which gives maximum flexibility but may overfit when samples differ only by targeted perturbations. Setting `model.constrain_basal_uniform = λ` (e.g. `λ = 100–1000`) adds an L2 penalty that pushes each gene's free-sample basals toward their common mean. The penalty is applied **per (sample, gene) pair**: a sample's basal for a given gene is excluded from the penalty if and only if that specific gene has a non-zero `basal_ref` for that sample (i.e. a KO or OV prior — see below). Concretely, for a `KO_CHGA` sample, only the CHGA basal is excluded; all other genes in that sample are still constrained to stay close to the wild-type values.

---

### Stimulus / exogenous signal (`n_stimuli`, `stimulus_schedule.txt`)

CARDAMOM supports one or several **exogenous inputs** (stimuli) that are not inferred but act as known regulators of the network. Stimuli occupy the first `n_stimuli` columns of the full gene-plus-stimulus state vector.

**Default behaviour (no file needed):** one stimulus that is `0` at the first timepoint and `1` at all subsequent timepoints.

**Custom schedule:** place `Data/stimulus_schedule.txt` in the project folder. Each row corresponds to a timepoint (in chronological order); each column to one stimulus:

```
# stimulus_schedule.txt  (tab or space separated, no header)
# rows = timepoints, cols = stimulus channels
0.0    0.0
1.0    0.0
1.0    1.0
```

For a single stimulus channel, a single-column file suffices. Values between 0 and 1 are allowed and interpreted as partial stimulus strength.

**Fewer rows than timepoints:** if the file contains fewer rows than the number of unique timepoints in the data, the missing timepoints automatically inherit the value of the **last row**. This is useful when a stimulus reaches a plateau and you only want to specify the transition rows explicitly. Providing *more* rows than timepoints raises an error.

**Simulation-specific schedule:** to use a *different* schedule during forward simulation (e.g. to test a new stimulus protocol after training), place `Data/stimulus_schedule_simul.txt`. `simulate_network.py` and `simulate_network_KOV.py` look for this file first, falling back to `stimulus_schedule.txt` if absent.

---

### Initialisation and reference arrays (`Data/`)

The network inference step accepts optional arrays to **warm-start** the optimiser or to **anchor** the solution towards a prior network. All files go in `my_project/Data/` and can be provided as `.npy` or gene-indexed `.csv`.

| File | Shape | Role |
|---|---|---|
| `basal_init.npy` / `.csv` | `(G,)` or `(n_samples, G, n_networks)` | Initial values for basal parameters |
| `inter_init.npy` / `.csv` | `(G, G)` or `(G, G, n_networks)` | Initial values for interaction matrix |
| `basal_ref.npy` / `.csv` | same as `basal_init` | Regularisation target for basal (penalises deviations; entries ≠ 0 also exclude that sample/gene from the `constrain_basal_uniform` penalty) |
| `inter_ref.npy` / `.csv` | same as `inter_init` | Regularisation target for interactions |
| `ref_network.csv` | `(G, G)` gene-indexed CSV | Binary/real prior interaction graph (sparsity mask) |

**CSV format for `basal_ref` / `inter_ref`:** rows and columns must be gene names matching `adata.var_names` (upper-cased). Stimulus rows/columns (`Stimulus`, `Stimulus_0`, …) are handled automatically.

A 3-D `basal_init` of shape `(n_samples, G, n_networks)` warm-starts each sample's basal independently. If a 2-D array `(G, n_networks)` is provided it is broadcast to all samples.

---

### Per-sample KO / OV prior (`Data/KO_OV_inference.txt`)

To encode prior knowledge about **which genes are knocked out (KO) or overexpressed (OV)** in specific samples, create a tab-separated file:

```
# KO_OV_inference.txt
sample_id	KO	OV
wt	        	
ko_CHGA	CHGA	
ov_STMN2	 	STMN2
ko_CHGA_ov_STMN2	CHGA	STMN2
```

- `sample_id` values must match `adata.obs['dataset_id']`.
- `KO` column: comma-separated gene names forced to **basal = −100** (silent during inference).
- `OV` column: comma-separated gene names forced to **basal = +100** (always active during inference).
- Multiple genes per cell: `CHGA,POSTN`.
- Missing or `0` entries mean no constraint for that sample.

When this file is present, `infer_network_structure.py` replaces `basal_ref` for the affected genes and samples, and saves `cardamomOT/basal_ref_mask.npy` (a boolean `(n_samples, G)` array) so that `simulate_network_KOV.py` can compute a clean wild-type baseline when running new perturbations.

---

### In-silico perturbation simulation (`Data/KO_OV_simulate.txt`)

After training, you can simulate arbitrary knock-out / over-expression combinations with `simulate_network_KOV.py`. Define the combinations in:

```
# KO_OV_simulate.txt  (tab-separated, header required)
KO	        OV
            		        # wild-type (no perturbation)
CHGA	
		STMN2
CHGA	STMN2
POSTN	S100B,STMN2
```

Each row produces one independent simulation. Results are saved as `cardamomOT/adata_prot_simul_KO_*_OV_*_stim*.h5ad`.

If `KO_OV_inference.txt` was used during training and `basal.npy` is therefore 3-D (per-sample), `simulate_network_KOV.py` automatically reconstructs a clean wild-type basal by averaging the unconstrained samples before applying each new perturbation.

#### Partial KO / OV via degradation rates (`GENE-X` syntax)

By default a KO silences a gene by setting its basal transcription to −∞ (complete silencing). For a **partial** perturbation of strength X% (0 < X < 100) append `-X` to the gene name:

```
# KO_OV_simulate.txt
KO	        OV
CHGA-80	                    # 80 % KO of CHGA
		STMN2-60            # 60 % OV of STMN2
CHGA-80	STMN2-60
POSTN	S100B               # full KO / full OV (no suffix = 100 %, existing behaviour)
```

**Mechanism:** instead of modifying the basal transcription, partial perturbations scale the **per-gene creation rate** (burst rate in the stochastic PDMP, effective `ks` in the ODE) without touching the degradation rates:

| Mode | Creation rate factor | Effect on steady-state |
|---|---|---|
| KO `X`% | `× (1 − X/100)` | production reduced → lower steady-state |
| OV `X`% | `× 1 / (1 − X/100)` | production increased → higher steady-state |

This is the correct approach for both simulation modes:
- **ODE**: steady state `P* = scale × ks_eff / d1` scales proportionally with the creation rate.
- **PDMP** (stochastic): only the burst *rate* is scaled; the burst *size* (`∝ d1/d0`) is unchanged, so the production rate scales by the factor without altering convergence speed.

Modifying degradation rates would leave the PDMP steady state unchanged (`P* = ks × scale_proteins` is independent of `d1` in the PDMP regime) and would only speed up or slow down dynamics rather than changing expression levels.

At X = 0 the factor is 1 (no effect). As X → 100 the KO factor → 0 (full silencing) and the OV factor → ∞. The output file is labelled `KO_CHGApct80_OV_STMN2pct60` to distinguish it from a complete perturbation.

> **Note on gene names with hyphens:** the `-X` suffix is recognised as a percentage only when `X` is a number strictly between 0 and 100. Gene names such as `HIF-1A` are therefore parsed correctly (the suffix `1` is outside the 0–100 exclusive range).

---

### Population dynamics: proliferation, death and cell-type transition rates

CARDAMOM's optimal transport step assumes a static population by default. If cells proliferate or die between timepoints, or if some cell-type transitions are more likely than others, you can provide this information to correct the OT marginals and cost matrix.

#### Proliferation and death rates (`adata.obs`)

Add per-cell rates directly to the AnnData object before running `infer_network_structure.py`:

```python
adata.obs['prolif_rate'] = ...   # float, net proliferation rate per cell (e.g. from EdU staining)
adata.obs['death_rate']  = ...   # float, net death rate per cell
```

When both columns are present, the OT marginals between consecutive timepoints t₁ and t₂ are modified:
- **Source marginal** µᵢ ∝ exp(+(prolif_i − death_i) · Δt/2) — cells with higher net growth carry more weight as trajectory sources
- **Target marginal** νⱼ ∝ exp(−(prolif_j − death_j) · Δt/2) — fast-growing cells at t₂ are down-weighted (they represent fewer distinct lineages)

This is equivalent to computing a **demographically corrected** optimal transport (as in Waddington OT, Schiebinger et al. 2019): the resulting coupling captures intrinsic lineage transitions independently of population-level growth effects. If either column is absent it is treated as 0 (neutral, no correction).

#### Cell-type transition rates (`Data/transition_rates.csv`)

To bias the OT cost toward biologically plausible cell-type transitions, place a square CSV in the project's `Data/` folder:

```
# transition_rates.csv — rows = source type at t1, cols = target type at t2
# values are relative likelihoods (need not sum to 1; will be normalised)
         ,TypeA,TypeB,TypeC
TypeA    ,  3.0,  1.0,  0.1
TypeB    ,  0.5,  2.0,  0.5
TypeC    ,  0.1,  0.5,  3.0
```

Row/column names must match the values of `adata.obs['cell_type']` (or `cell_types` / `celltype`). The matrix is **bistochastically normalised** (Sinkhorn–Knopp, 500 iterations) and rescaled to mean = 1 so the overall cost scale is preserved.

The OT pairwise distance is then divided element-wise by the normalised transition weight: a transition with weight > 1 becomes cheaper (preferred), and a transition with weight < 1 becomes more expensive (penalised). Missing cell types default to the lowest transition weight (index 0).

Both corrections are active simultaneously when the corresponding files are present. They apply during training (`infer_network_structure.py`) and on the test set (`infer_test.py`).

---

### Summary: project directory structure

```
my_project/
├── Data/
│   ├── data.h5ad                  # required — obs['time'], obs['dataset_id'] (opt.), obs['rd'] (opt.)
│   │                              #            obs['prolif_rate'] (opt.), obs['death_rate'] (opt.)
│   │                              #            obs['cell_type'] (opt.)
│   ├── gene_list.txt              # optional — subset of genes to use
│   ├── stimulus_schedule.txt      # optional — stimulus values per timepoint
│   ├── stimulus_schedule_simul.txt# optional — overrides stimulus schedule for simulation only
│   ├── ref_network.csv            # optional — prior interaction graph (sparsity mask)
│   ├── basal_init.npy / .csv      # optional — warm-start for basal parameters
│   ├── inter_init.npy / .csv      # optional — warm-start for interactions
│   ├── basal_ref.npy / .csv       # optional — regularisation target for basal
│   ├── inter_ref.npy / .csv       # optional — regularisation target for interactions
│   ├── KO_OV_inference.txt          # optional — per-sample KO/OV prior (requires dataset_id)
│   ├── KO_OV_simulate.txt             # optional — in-silico perturbations to simulate
│   └── transition_rates.csv       # optional — cell-type transition cost matrix for OT
└── cardamomOT/                    # generated by the pipeline
    ├── basal.npy                  # (n_samples, G, n_networks)
    ├── inter.npy                  # (G, G, n_networks)
    ├── basal_ref_mask.npy         # (n_samples, G) bool — generated from KO_OV_inference.txt
    └── ...
```

---

## 📊 Understanding Results

### Main generated files

**In `my_project/cardamomOT/`:**
- `inter.npy`: gene regulatory interaction matrix `(G, G, n_networks)`
- `basal.npy`: basal transcription parameters `(n_samples, G, n_networks)`
- `mixture_parameters.npy`: burst kinetics parameters `(G+1, G)`

**In `my_project/Check/`:**
- Visual comparisons between real data and simulations

### Parameter Interpretation

- **Interactions (`inter[i, j, k]`)**: weight of gene `i` on gene `j` in network state `k`; positive = activation, negative = repression
- **Basal (`basal[s, g, k]`)**: constitutive transcription of gene `g` in sample `s` under network state `k`
- **Mixture parameters**: frequency and size of transcriptional bursts per gene

## 🔧 Troubleshooting

### Common Issues

**Import error (numpy/scipy):**
```bash
# Verify active environment
conda activate cardamom_env
python -c "import numpy, scipy; print('OK')"
```

**Non-conforming data:**
- Verify that `data.obs['time']` exists
- Ensure counts are positive integers

**Insufficient memory:**
- Reduce the number of genes in `gene_list.txt`
- Use subsampling of cells

### Debug Commands

```bash
# Verify installation
python -c "import CardamomOT; print('CARDAMOM imported')"

# Test a specific module
python -c "from CardamomOT.inference import mixture; print('Module OK')"

# Check dependencies
python -m CardamomOT.cli --help
```

## 📚 Advanced Tutorials

### Converting data from CARDAMOM v1

If you have data in CARDAMOM v1 format:

```bash
# Convert old .txt format to .h5ad
python ./utils/old_to_new/convert_old_data_to_ad.py -i my_project
python ./utils/old_to_new/add_degradations_to_ad.py -i my_project
```

### Customizing Parameters

See source files to modify:
- `select_DEgenes_and_split.py`: gene selection criteria
- `infer_mixture.py`: burst kinetics parameters
- `infer_network_*.py`: network inference algorithms

## 📖 Références

[3] Y. Mauge, E. Ventre. [Mechanistic optimal transport reveals gene regulatory networks and cellular trajectories from temporal single-cell transcriptomics]. *bioRxiv*, 2026.

[2] E. Ventre, U. Herbach et al. [One model fits all: Combining inference and simulation of gene regulatory networks](https://doi.org/10.1371/journal.pcbi.1010962). *PLOS Computational Biology*, 2023.

[1] E. Ventre. [Reverse engineering of a mechanistic model of gene expression using metastability and temporal dynamics](https://content.iospress.com/articles/in-silico-biology/isb210226). *In Silico Biology*, 2021.

