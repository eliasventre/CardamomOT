# CARDAMOM

CARDAMOM is an executable gene regulatory network (GRN) inference method, adapted to time-course scRNA-seq datasets. The algorithm consists in calibrating the parameters of a mechanistic model of gene expression: the calibrated model can then be simulated, which allows to reproduce the dataset used for inference. The first inference method has been introduced in [[1](#Ventre2021)]. It has been benchmarked along with other GRN inference tools and applied to a real dataset in [[2](#Ventre2023)]. The second version is presented in [[2](#Mauge2026)] and combines GRN and trajectory inference method and shows a strong improvement over the first version.  The simulation part is based on the [Harissa](https://github.com/ulysseherbach/harissa) package.

## 🚀 Quick Start

### 1. Installation

#### Create a virtual environment (recommended)

```bash
# Create a new conda environment
conda create -n cardamom_env python=3.12 -y
conda activate cardamom_env

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

## 📊 Understanding Results

### Main generated files

**In `my_project/cardamom/`:**
- `inter.npy`: gene regulatory interaction matrix (G×G)
- `basal.npy`: basal expression parameters (G×1)
- `mixture_parameters.npy`: burst kinetics parameters (G×3)

**In `my_project/Check/`:**
- Visual comparisons between real data and simulations

**In `my_project/visualization/`:**
- Analyses of the inferred regulatory network

### Parameter Interpretation

- **Interactions (`inter.npy`)**: positive values = activation, negative = repression
- **Basal parameters (`basal.npy`)**: constitutive gene expression
- **Mixture parameters**: frequency and size of transcriptional bursts

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

