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
- Runs all steps with default parameters
- No interaction required
- Useful for scripting or batch processing

### Traditional Pipeline (Direct arguments)
```bash
cardamomot pipeline \
  -i /path/to/project \
  -s train \
  -c 1 \
  -r 1.0 \
  -m 0.5
```
- Legacy interface for experienced users
- All parameters specified on command line
- Common flags:
  - `-i, --input`: Project directory
  - `-s, --split`: Data split name (default: 'full')
  - `-c, --change`: Change detection flag (0 or 1)
  - `-r, --rate`: Learning rate (default: 1.0)
  - `-m, --mean`: Mean constraint strength (default: 1.0)

### Individual Steps (Debugging/Advanced)
```bash
cardamomot step infer_mixture -i /path/to/project
```
- Run a single analysis step
- Useful for debugging or re-running specific steps
- Available steps: `infer_rd`, `infer_mixture`, `select_DEgenes_and_split`, `infer_network_structure`, etc.

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

# Re-run a specific step with verbose output
cardamomot step infer_mixture -i my_project --verbose
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

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Input path | `-i` | required | Project directory |
| Split name | `-s` | 'full' | Data partition to analyze |
| Change flag | `-c` | 0 | Enable change detection (0=off, 1=on) |
| Rate | `-r` | 1.0 | Learning rate for inference (0.1-10) |
| Mean | `-m` | 1.0 | Strength of mean constraint (0.1-10) |
| N genes | `-n` | 1000 | Number of genes to keep |
| Verbose | `--verbose` | off | Enable debug logging |
| Quiet | `--quiet` | off | Suppress info messages |
| Log file | `--log-file` | none | Save logs to file |

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
