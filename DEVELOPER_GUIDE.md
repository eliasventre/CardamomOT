# Developer Guide: Interactive Pipeline Architecture

## Overview

The CardamomOT interactive pipeline system consists of three main components:

```
┌─────────────────────────────────────────────────┐
│        cardamom run (CLI entry point)           │
│              cli.py main()                      │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│   Interactive Pipeline Orchestrator             │
│      cli_pipeline.run_pipeline_interactive()    │
│  - Validates project structure                  │
│  - Prompts for step selection (checkboxes)      │
│  - Gathers hyperparameters                      │
│  - Executes steps sequentially                  │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│     Individual Pipeline Scripts                 │
│  (infer_rd.py, infer_mixture.py, etc.)         │
│  Executed as subprocess calls                   │
└─────────────────────────────────────────────────┘
```

## Key Modules

### 1. cli.py - Command-Line Interface Entry Point

**Location**: `CardamomOT/cli.py`

**Purpose**: Main CLI dispatcher with argparse-based command routing.

**Key Functions**:
- `main()` - Entry point; defines subparsers for `run`, `pipeline`, and `step` commands
- `_run_pipeline_interactive(args)` - Handler for `cardamom run` command; delegates to cli_pipeline

**Responsibilities**:
- Parse command-line arguments
- Route to appropriate handler function
- Maintain backward compatibility with existing scripts

**Integration Points**:
- Imports `cli_pipeline` module
- Called via console_scripts entry point in setup.py/pyproject.toml

```python
entry_points={
    'console_scripts': [
        'cardamom=CardamomOT.cli:main',
    ],
}
```

### 2. cli_pipeline.py - Interactive Pipeline Orchestration

**Location**: `CardamomOT/cli_pipeline.py`

**Purpose**: Core interactive workflow engine for step selection and execution.

**Key Data Structure**: PIPELINE_STEPS list
```python
PIPELINE_STEPS = [
    {
        "id": "infer_rd",
        "name": "Estimate read depth correction",
        "script": "infer_rd.py",
        "description": "...",
    },
    # ... more steps
]
```

**Key Functions**:

#### `validate_project_structure(project_path: Path) -> bool`
- Checks for required `Data/` directory
- Verifies presence of at least one `.h5ad` file
- Returns boolean success status
- Called at start of `run_pipeline_interactive()`

```python
if not validate_project_structure(Path(project_path)):
    print("ERROR: Invalid project structure")
    return False
```

#### `interactive_step_selection() -> List[str]`
- Presents UI for step selection
- Returns list of selected step script names
- Uses questionary for checkboxes if available (HAS_QUESTIONARY=True)
- Falls back to Y/n prompts if questionary missing

**Behavior**:
```
┌─ HAS_QUESTIONARY? ──┐
└─ Yes: Show checkboxes with all steps pre-selected
└─ No:  Prompt Y/n for each step
```

#### `interactive_parameter_input(step_id: str, project_path: str) -> Dict`
- Gathers step-specific hyperparameters
- Called before executing each step
- Displays simple prompts for each relevant parameter
- Returns dict of parameter names/values

**Step-to-Parameters Mapping**:
- `select_DEgenes` → `n_genes`, `temporal_quantile`
- `infer_mixture` → `mean_forcing`
- `infer_network` → `scale_penalty`, `max_iterations`
- Other steps → Basic options

#### `run_step(script_name: str, params: Dict, repo_root: Path) -> bool`
- Executes single pipeline script as subprocess
- Builds command: `python script.py -i project -s split -param value`
- Handles subprocess errors gracefully
- Prompts user to continue/abort on failure

```python
def run_step(script_name, params, repo_root):
    cmd = [sys.executable, repo_root / script_name, ...]
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        # Handle error, prompt user
        return False
```

#### `run_pipeline_interactive(project_path: str, use_defaults: bool = False) -> None`
- **Main orchestrator function**
- Called by `_run_pipeline_interactive()` in cli.py
- Workflow:
  1. Validate project structure
  2. Select steps (or skip if use_defaults=True)
  3. For each selected step:
     - Get hyperparameters (or use defaults if use_defaults=True)
     - Execute via `run_step()`
     - Handle errors
  4. Print completion summary

**Pseudocode**:
```python
def run_pipeline_interactive(project_path, use_defaults=False):
    if not validate_project_structure(project_path):
        return False
    
    if use_defaults:
        selected_scripts = [step["script"] for step in PIPELINE_STEPS]
    else:
        selected_scripts = interactive_step_selection()
    
    for script in selected_scripts:
        params = interactive_parameter_input(step_id, project_path)
        if not run_step(script, params, repo_root):
            if not user_continues():
                return False
    
    print("Pipeline complete!")
    return True
```

#### `main() -> None`
- Standalone entry point for direct module invocation
- Uses argparse to parse `--project` and `--default` arguments
- Calls `run_pipeline_interactive()`

## Configuration System

### config_template.yaml

**Location**: `CardamomOT/config_template.yaml`

**Purpose**: Template for user configuration with documented hyperparameters

**Sections**:
```yaml
project:
  name: "My Analysis"
  description: "..."
data:
  input_file: "data.h5ad"
  split: "train"
mixture:
  mean_forcing: 0.5
selection:
  n_genes: 1000
  temporal_quantile: 0.8
# ... more sections
```

**Status**: Currently template only
**Future**: Can be loaded in `run_pipeline_interactive()` to pre-populate defaults

**Integration Proposal**:
```python
def run_pipeline_interactive(project_path, use_defaults=False):
    config = load_config(Path(project_path) / "config.yaml")
    # Use config values as defaults
```

## Error Handling & Resilience

### Step-Level Error Recovery

When `run_step()` fails:
1. Print error message
2. Ask user: "Continue with next step? (y/n)"
3. User decides whether to abort or skip current step
4. Continue to next step if user chooses

### Fallback Mechanisms

1. **Questionary unavailable**:
   - HAS_QUESTIONARY flag checked at module load time
   - Falls back to simple Y/n prompts
   - Functionality preserved, UX degrades gracefully

2. **Missing Data/ directory**:
   - Caught by `validate_project_structure()`
   - User told which directory/files are missing
   - Pipeline aborts before running any steps

3. **Script execution fails**:
   - Subprocess error caught
   - User prompted to continue/abort
   - Allows skipping problematic steps

## Extending the Pipeline

### Adding a New Step

1. **Add to PIPELINE_STEPS** in cli_pipeline.py:
```python
PIPELINE_STEPS.append({
    "id": "my_new_step",
    "name": "Description for UI",
    "script": "my_new_script.py",
    "description": "Detailed description",
})
```

2. **Add parameter handling** (if needed):
```python
def interactive_parameter_input(step_id, project_path):
    # ...
    elif step_id == "my_new_step":
        params = {
            "my_param": input("Enter value: "),
        }
    return params
```

3. **Ensure script exists** in repo root as `my_new_script.py`

### Modifying Parameter Defaults

Edit `PIPELINE_STEPS` dict:
```python
{
    "id": "select_DEgenes",
    "defaults": {
        "n_genes": 500,  # Changed from 1000
        "temporal_quantile": 0.5,
    }
}
```

Then update `interactive_parameter_input()` to use these defaults:
```python
defaults = step["defaults"]
n_genes = input(f"Number of genes [{defaults.get('n_genes')}]: ") \
          or defaults.get('n_genes')
```

### Supporting Config Files

To integrate YAML config support:

1. Import pyyaml: `import yaml`
2. Load config in `run_pipeline_interactive()`:
```python
def load_config(config_path):
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}

config = load_config(Path(project_path) / "config.yaml")
```
3. Use config values as parameter defaults

## Testing Checklist

When modifying cli_pipeline.py:

- [ ] Validate project structure detection works
- [ ] Questionary fallback to Y/n prompts works
- [ ] Step selection saves selected steps correctly
- [ ] Parameter prompts accept user input
- [ ] Subprocess execution succeeds for valid steps
- [ ] Error handling works (skip/continue/abort)
- [ ] `--default` flag bypasses all prompts
- [ ] Summary printed at end

## Integration with Existing Code

### Backward Compatibility

The new `cardamom run` command is **additive only**:
- Existing `cardamom pipeline` command unchanged
- Existing `cardamom step` command unchanged
- All existing scripts (infer_*.py) unchanged
- Users can continue using shell scripts if preferred

### Setup/Installation

Entry points properly configured:
```python
# setup.py and pyproject.toml both specify:
[project.scripts]
cardamom = "CardamomOT.cli:main"
```

Package can be installed with:
- `pip install -e .` (CLI with fallback behavior)
- `pip install -e ".[cli]"` (CLI with questionary)
- `pip install -e ".[cli,dev,notebooks]"` (Full dev setup)

## Performance Considerations

1. **Subprocess overhead**: Each step runs in separate Python process
   - Necessary for isolation and memory management
   - Acceptable for analysis pipeline (steps take minutes/hours)
   
2. **Questionary rendering**: Minimal overhead (<100ms)
   - Only runs at start during step selection
   - Fallback simple Y/n prompts are instant

3. **Project validation**: Single directory scan
   - O(n) where n = files in Data/ directory
   - Typically <100ms even for large projects

## Future Enhancements

1. **Config file persistence**: Load/save user preferences in YAML
2. **Step dependencies**: Validate step ordering, skip unnecessary steps
3. **Dry-run mode**: Print commands without executing
4. **Progress bar**: Show overall pipeline progress across all steps
5. **Parallel execution**: Run independent steps in parallel (with locking)
6. **Visualization dashboard**: Web UI for monitoring and interaction
7. **Result inspection**: Built-in tools to view/compare outputs

---

## Contact & Contributions

For questions or to contribute improvements:
- Open an issue on [GitHub](https://github.com/yourusername/CardamomOT)
- Submit pull requests with new features or fixes
- Email: developers@email.com
