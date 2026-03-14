# CardamomOT Setup Verification Checklist

Use this checklist to verify that CardamomOT is properly installed and ready to use.

## Installation Verification

- [ ] **Python version OK**
  ```bash
  python --version  # Should show Python 3.8 or higher
  ```

- [ ] **CardamomOT installed**
  ```bash
  pip show cardamomOT  # Should show package info
  ```

- [ ] **CLI command available**
  ```bash
  cardamomot --help  # Should show help output
  ```

- [ ] **Run subcommand works**
  ```bash
  cardamomot run --help  # Should show run command help
  ```

## Project Structure Verification

For a project you want to analyze:

- [ ] **Project directory exists**
  ```bash
  ls -la /path/to/my_project/  # Should list files
  ```

- [ ] **Data/ subdirectory exists**
  ```bash
  ls /path/to/my_project/Data/  # Should not be empty
  ```

- [ ] **H5AD file present**
  ```bash
  ls -lh /path/to/my_project/Data/*.h5ad  # Should show file(s)
  ```

- [ ] **H5AD file is readable**
  ```bash
  python -c "
  import anndata
  adata = anndata.read_h5ad('/path/to/my_project/Data/data.h5ad')
  print(f'Loaded: {adata.n_obs} cells x {adata.n_vars} genes')
  "
  ```

## Optional Dependencies

- [ ] **Questionary installed** (for interactive checkboxes)
  ```bash
  python -c "import questionary; print('OK')"
  # If fails, run: pip install questionary
  ```

- [ ] **PyYAML installed** (for config file support)
  ```bash
  python -c "import yaml; print('OK')"
  # If fails, run: pip install pyyaml
  ```

- [ ] **All recommended packages**
  ```bash
  pip install -e ".[cli]"  # In CardamomOT repo root
  ```

## Runtime Verification

### Test 1: Dry Run (No Execution)

```bash
cd /path/to/my_project
cardamomot run . --help  # Should show command help
```

### Test 2: Validation Only

```bash
# This will validate structure but shouldn't execute anything yet
python -c "
from CardamomOT.cli_pipeline import validate_project_structure
from pathlib import Path
is_valid = validate_project_structure(Path('.'))
print(f'Project valid: {is_valid}')
"
```

### Test 3: Interactive Mode (Single Step)

```bash
cd /path/to/my_project/

# Run in interactive mode
# When prompted, select ONLY ONE step (e.g., infer_rd)
# Press Enter to use default parameters
cardamomot run .
```

**Expected outcome**:
- See checkboxes for each step
- Can select/deselect steps
- Prompted for parameters
- One step executes successfully
- Output directory created: `cardamom_output/`

### Test 4: Default Mode

```bash
cd /path/to/my_project/

# Run with all defaults (no prompts)
cardamomot run . --default
```

**Expected outcome**:
- No interactive prompts
- All pipeline steps execute
- Results saved to `cardamom_output/`

## Output Verification

After running analysis, check:

- [ ] **Output directory created**
  ```bash
  ls /path/to/my_project/cardamom_output/
  ```

- [ ] **Results present**
  ```bash
  ls -la /path/to/my_project/cardamom_output/cardamom/
  ls -la /path/to/my_project/cardamom_output/simulations/
  ```

- [ ] **Log files generated**
  ```bash
  ls /path/to/my_project/cardamom_output/*.log
  ```

- [ ] **Check log for errors**
  ```bash
  tail -100 /path/to/my_project/cardamom_output/pipeline.log
  ```

## Troubleshooting

### If any check fails:

**"command not found: cardamomot"**
```bash
# Reinstall from source
cd /path/to/CardamomOT
pip install -e .
```

**"Data/ directory not found"**
```bash
# Create and populate Data/
mkdir -p /path/to/my_project/Data
cp my_data.h5ad /path/to/my_project/Data/
```

**"No module named 'questionary'"**
```bash
# Install optional dependencies
pip install questionary pyyaml
# OR
cd /path/to/CardamomOT
pip install -e ".[cli]"
```

**"H5AD file not found or unreadable"**
```bash
# Verify file
file /path/to/my_project/Data/*.h5ad  # Should show HDF5 format

# Test reading
python -c "
import anndata
try:
    adata = anndata.read_h5ad('/path/to/file.h5ad')
    print(f'File OK: {adata.shape}')
except Exception as e:
    print(f'Error: {e}')
"
```

**"Pipeline step fails"**
```bash
# Check detailed log
cat /path/to/my_project/cardamom_output/pipeline.log

# Try individual step
cardamomot step infer_mixture -i /path/to/my_project --verbose
```

## System Configuration

### Memory Requirements

CardamomOT typically needs:
- **Minimum**: 8 GB RAM (for small datasets)
- **Recommended**: 16 GB RAM (for typical datasets)
- **Large datasets**: 32+ GB RAM

Check available memory:
```bash
python -c "
import psutil
mem = psutil.virtual_memory()
print(f'Available RAM: {mem.available / 1e9:.1f} GB')
"
```

### Disk Space Requirements

Typical pipeline needs:
- **Input data** (H5AD): 100 MB - 1 GB
- **Intermediate files**: 500 MB - 5 GB
- **Output**: 1 GB - 10 GB
- **Total**: 2 GB - 20 GB (depends on dataset size)

Check available space:
```bash
df -h /path/to/my_project/  # Shows available disk space
```

### Python Environment

Recommended setup:
```bash
# Use virtual environment
python3 -m venv cardamom_env
source cardamom_env/bin/activate  # On Windows: cardamom_env\Scripts\activate

# Install CARDAMOM
pip install --upgrade pip
pip install -e ".[cli]"

# Verify
cardamomot --version
```

## Platform-Specific Notes

### macOS
- M1/M2 chips may need PyTorch compiled for ARM64
- Install: `pip install torch::py310_m1` or similar
- See PyTorch website for your specific version

### Linux (Ubuntu/Debian)
- Install system dependencies: `sudo apt install python3-dev`
- Consider using `conda` instead of `pip` for complex dependencies

### Windows
- Use WSL2 (Windows Subsystem for Linux) for best compatibility
- Or use Anaconda distribution
- Be careful with path separators (use forward slashes or raw strings)

## Getting Help

If checks fail or you encounter issues:

1. **Consult documentation**
   - [INSTALL_GUIDE.md](INSTALL_GUIDE.md) — Installation steps
   - [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) — Usage workflow
   - [QUICK_REFERENCE.md](QUICK_REFERENCE.md) — Command reference
   - [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) — Technical details

2. **Check logs**
   ```bash
   cat /path/to/project/cardamom_output/*.log
   tail -f /path/to/project/cardamom_output/pipeline.log  # Monitor in real-time
   ```

3. **Search existing issues**
   - https://github.com/yourusername/CardamomOT/issues

4. **Open a new issue**
   - Include output of: `cardamomot --help`
   - Include relevant section of log file (lines leading to error)
   - Include system info: OS, Python version, available RAM

5. **Email support**
   - developers@email.com

---

**✅ If all checks pass, you're ready to run CardamomOT!**

Start with: `cardamomot run /path/to/my_project`
