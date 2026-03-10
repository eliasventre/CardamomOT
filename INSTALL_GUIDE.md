# Installation and Quick Start Guide

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/CardamomOT.git
cd CardamomOT
```

### 2. Install CardamomOT

#### Option A: Minimal Installation (core functionality only)
```bash
pip install -e .
```

#### Option B: Full Installation (recommended - includes interactive CLI)
```bash
pip install -e ".[cli]"
```

This installs questionary for interactive checkboxes and pyyaml for config file support.

#### Option C: Development Setup (if you want to contribute)
```bash
pip install -e ".[cli,dev,notebooks]"
```

This installs everything for development, testing, and running example notebooks.

### 3. Verify Installation
```bash
cardamomot --help
```

You should see the help output with available commands:
- `run` — Interactive pipeline runner (recommended for new users)
- `pipeline` — Full pipeline with direct arguments (power users)
- `step` — Run individual analysis steps (debugging/advanced)

## Quick Start: Your First Analysis

### Step 1: Prepare Your Project

```bash
mkdir my_cardamom_study
cd my_cardamom_study
mkdir Data
# Copy your expression data (in H5AD format) to Data/
cp /path/to/your_data.h5ad Data/
```

Expected structure:
```
my_cardamom_study/
├── Data/
│   └── data.h5ad          # Your expression matrix
└── config.yaml            # (optional) Custom parameters
```

### Step 2: Run the Pipeline Interactively

```bash
cardamomot run .
```

This will:
1. ✓ Validate your project structure
2. ✓ Show checkboxes for each analysis step (select which ones to run)
3. ✓ Prompt for custom hyperparameters
4. ✓ Execute the pipeline with your choices

### Step 3: Use Default Parameters (Advanced)

To run all steps with default parameters without interaction:

```bash
cardamomot run . --default
```

### Step 4: Review Results

Results are automatically saved to `cardamom_output/`:
```
my_cardamom_study/
├── cardamom_output/
│   ├── cardamom/          # Inferred network files
│   ├── simulations/       # Simulated trajectories
│   ├── figures/           # Visualizations
│   └── *.log              # Execution logs
```

## Pipeline Steps Explained

When running interactively, you'll see these steps:

1. **Read depth correction** → Normalize sequencing depth
2. **Mixture model** → Learn kinetic parameters
3. **Gene selection** → Filter to top variable genes
4. **Network structure** → Add prior knowledge (optional)
5. **Network inference** → Learn gene interactions
6. **Network adaptation** → Prepare for simulation
7. **Network simulation** → Generate synthetic data

Each step has sensible defaults, but you can customize hyperparameters when prompted.

## Troubleshooting

### "command not found: cardamomot"
This means CardamomOT wasn't installed as a console command. Try:
```bash
pip install -e .
```
Or add to PATH:
```bash
export PATH="$PATH:$(python -c 'import CardamomOT; print(CardamomOT.__file__[:-12])')"
```

### "Data/ directory not found"
Make sure your project has a `Data/` subfolder with your `.h5ad` file:
```bash
mkdir -p ./Data
cp your_data.h5ad ./Data/
```

### Missing questionary library
If interactive checkboxes don't work, install the optional CLI dependencies:
```bash
pip install questionary pyyaml
```
(The tool will fall back to Y/n prompts if questionary is unavailable.)

### Python version incompatibility
CardamomOT requires Python 3.8 or higher. Check your version:
```bash
python --version
```

## Next Steps

- See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for detailed workflow documentation
- Check [example notebooks](utils/) for result visualization and interpretation
- Read [the paper](https://doi.org/10.1101/2023.) for methods and theory

## Getting Help

- Open an issue on [GitHub](https://github.com/yourusername/CardamomOT/issues)
- Check the [documentation](https://github.com/yourusername/CardamomOT/wiki)
- Email: authors@email.com

---

**Welcome to CardamomOT!** 🌿
