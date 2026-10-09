# CardamomOT CLI Quick Reference

## Commands Summary

### Interactive Pipeline (Recommended)
```bash
cardamomot run /path/to/project
```
- Shows checkboxes to select analysis steps
- Prompts for hyperparameter customization
- Best for first-time users

### Automated Pipeline (Skip all prompts)
```bash
cardamomot run /path/to/project --default
```
- Runs the steps given by the project parameters (model_parameters sheet), with their values
- No interaction required
- Useful for scripting or batch processing

### Full pipeline (direct arguments)
```bash
cardamomot pipeline -i /path/to/project --stimulus 1 --prior 1 --mean-forcing 0.5
./run.sh /path/to/project 1 1 0.5          # same, positional
```
- Only the hard-to-calibrate parameters are options, in this order: `--stimulus`, `--prior`,
  `--mean-forcing`, `--force-basins`, `--temporal-basins` (absent or -1 = workbook value, else default)
- Everything else (split, train_rate, select_genes, build_prior_network, estimate_proliferation_rates,
  run_test, simulate_perturbations, simulate_with_proliferation, species...) is set in the
  `model_parameters` sheet of `Data/CardamomOT_inputs.xlsx` (default of `CardamomOT/model/base.py`)
- Precedence: default < workbook < command line

### Individual Steps (Debugging/Advanced)
```bash
cardamomot step infer_mixture -i /path/to/project --mean-forcing 0.5
```
- Run a single analysis step
- Useful for debugging or re-running specific steps
- Available steps: `infer_mixture`, `select_genes`, `infer_network_structure`, etc.

## Common Workflows

### First-Time User (Simplest)
```bash
# 1. Prepare data
mkdir my_project/Data
cp my_data.h5ad my_project/Data/

# 2. Run interactive pipeline
cd my_project
cardamomot run .

# 3. Select steps and parameters when prompted
```

### Batch Processing (Scripting)
```bash
# Run multiple projects with defaults
for project in project1 project2 project3; do
  cardamomot run "$project" --default
done
```

### Custom Configuration
```bash
# 1. Copy template
cp config_template.yaml my_project/config.yaml

# 2. Edit custom parameters
nano my_project/config.yaml

# 3. Run with defaults
cardamomot run my_project --default
```

### Debugging a Failed Step
```bash
# Check which steps ran
ls my_project/cardamom_output/

# Re-run a specific step
cardamomot step infer_mixture -i my_project
```

## Project Structure

```
my_project/
├── Data/
│   └── data.h5ad              # Expression matrix (required)
├── config.yaml                # Configuration (optional)
└── cardamom_output/           # Results (auto-created)
    ├── cardamom/
    │   ├── data_train.npy
    │   ├── network_final.npy
    │   └── ...
    ├── simulations/
    │   ├── trajectories.npy
    │   └── ...
    └── logs/
        └── *.log
```

## Hyperparameter Quick Guide

| Parameter | Flag (`run.sh` position) | Default (`base.py`) | Description |
|-----------|------|---------|-------------|
| Input path | `-i` (1) | required | Project directory |
| Stimulus | `--stimulus` (2) | 1.0 | Stimulus edge penalisation in [0, 1] |
| Prior | `--prior` (3) | 1.0 | Weight of edges absent from the literature prior (0 = hard mask) |
| Mean forcing | `--mean-forcing` (4) | 0.5 | Mean-forcing intensity of the NB mixture |
| Force basins | `--force-basins` (5) | 1.0 | Basin weights kept from the mixture in the network fit |
| Temporal basins | `--temporal-basins` (6) | 1 | Basin weights per timepoint (0/1) |
| Split, genes, steps... | workbook | see `base.py` | `split`, `train_rate`, `select_genes`, `num_max_genes`, `run_test`... |

## Help and Information

```bash
# General help
cardamomot --help

# Help for specific command
cardamomot run --help
cardamomot pipeline --help
cardamomot step --help

# Check installation
python -c "import CardamomOT; print(CardamomOT.__version__)"
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `command not found: cardamomot` | Run `pip install -e .` from repo root |
| `Data/ directory not found` | Create `Data/` folder and add `.h5ad` file |
| `No module named 'questionary'` | Run `pip install ".[cli]"` or `pip install questionary` |
| `AttributeError: module has no attribute 'X'` | Update CardamomOT: `pip install -e . --upgrade` |
| Step fails with error | Check logs: `ls *project*/cardamom_output/logs/` |

## Tips and Tricks

1. **Save your selections**: After answering prompts once, you can edit `config.yaml` to repeat the same configuration
2. **Run in background**: Use `nohup cardamom run project &` to run in the background
3. **Parallel processing**: Different projects can be run in parallel (each gets its own Python kernel)
4. **Monitor progress**: Check logs in real-time: `tail -f project/cardamom_output/logs/pipeline.log`
5. **Skip steps silently**: Edit `config.yaml` to disable specific steps, then use `--default`

---

For more details, see:
- [INSTALL_GUIDE.md](INSTALL_GUIDE.md) — Installation and setup
- [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) — Detailed workflow documentation
- [README.md](README.md) — Main documentation and methods
