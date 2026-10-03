# CardamomOT CLI Architecture

## Overview

The CardamomOT command-line interface (CLI) is properly structured with clear separation of concerns:

```
cardamomot (console_scripts entry point in setup.py/pyproject.toml)
    ↓
CardamomOT/cli.py:main()
    ├── cardamomot run      → CardamomOT/cli.py:_run_pipeline_interactive()
    │                          → CardamomOT/cli_pipeline.py:run_pipeline_interactive()
    │
    ├── cardamomot pipeline → CardamomOT/cli.py:_pipeline()   (run.sh calls it)
    │                          [steps from the project parameters: cli.pipeline_steps]
    │
    └── cardamomot step     → CardamomOT/cli.py:_run_script()
                              [runs an individual script by name]
```

## Options and parameters (CardamomOT/run_options.py)

- Every step takes `-i <project>` and, among the hard-to-calibrate parameters (in this order
  `--stimulus`, `--prior`, `--mean-forcing`, `--force-basins`, `--temporal-basins`), only those it uses
  (`STEP_OPTIONS`); `parse_step_options` rejects the others, and names the workbook parameter replacing a
  removed option (`REMOVED_OPTIONS`).
- Everything else (split, train_rate, select_genes, build_prior_network, estimate_proliferation_rates,
  run_test, simulate_perturbations, simulate_with_proliferation, species, step settings) is a NetworkModel
  parameter (`CardamomOT/model/base.py`), fixed per project in the Model_parameters sheet of
  `Data/CardamomOT_inputs.xlsx`.
- `configure(model, opts)`: workbook, then command line — precedence default < workbook < command line.
  `settings(opts)` returns a configured NetworkModel(1) to read the pipeline parameters before a step's model;
  `step_arguments(step, p, values)` builds the command line of a step for the runners.

## Module Responsibilities

### CardamomOT/cli.py
**Purpose**: Main CLI dispatcher and legacy command handler

**Responsibilities**:
- Parse command-line arguments via argparse
- Route to appropriate handler function
- Maintain backward compatibility with old scripts
- Provide helper functions used by standalone scripts

**Key Functions**:
- `create_pipeline_parser()` - Build standardized argument parser for scripts
- `add_*_argument()` - Helpers for adding common arguments
- `handle_common_args()` - Configure logging from parsed args
- `_run_script()` - Execute individual Python script as subprocess
- `_pipeline()` - Run full "traditional" pipeline (all scripts in sequence)
- `_run_pipeline_interactive()` - Delegate to cli_pipeline for new interactive mode
- `main()` - Argparse entry point, register all subcommands

**Does NOT handle**:
- ~~Interactive step selection~~ → cli_pipeline.py
- ~~Project validation~~  → cli_pipeline.py
- ~~Parameter prompts~~ → cli_pipeline.py

### CardamomOT/cli_pipeline.py
**Purpose**: Interactive pipeline orchestration engine

**Responsibilities**:
- Validate project directory structure
- Present interactive step selection interface
- Prompt for step-specific hyperparameters
- Execute pipeline steps sequentially
- Handle errors and recovery
- Provide default mode (skip all prompts)

**Key Data/Functions**:
- `PIPELINE_STEPS` - List of pipeline step definitions
- `validate_project_structure()` - Check for Data/ directory and .h5ad files
- `interactive_step_selection()` - Present checkboxes or Y/n prompts
- `interactive_parameter_input()` - Gather step-specific parameters
- `run_step()` - Execute single script as subprocess
- `run_pipeline_interactive()` - Main orchestrator function
- `main()` - Standalone entry point (for direct module execution)

**Does NOT handle**:
- ~~General argparse setup~~ → cli.py
- ~~Other subcommands (pipeline/step)~~ → cli.py
- ~~Logging configuration~~ → cli.py

## Command Examples

### Interactive Pipeline (New, Recommended)
```bash
# User-friendly checkbox-based step selection
cardamomot run /path/to/project

# Skip prompts, use defaults
cardamomot run /path/to/project --default
```

Flow:
1. User runs: `cardamomot run /path/to/project`
2. cli.py main() parses args, routes to `_run_pipeline_interactive()`
3. `_run_pipeline_interactive()` calls `cli_pipeline.run_pipeline_interactive()`
4. cli_pipeline handles all interaction and execution

### Traditional Pipeline (Legacy)
```bash
# Direct parameter specification, no interaction
cardamomot pipeline -i /path/to/project --stimulus 1 --prior 1 --mean-forcing 0.5
```

Flow:
1. User runs: `cardamomot pipeline -i /path/to/project ...`
2. cli.py main() parses args, routes to `_pipeline()`
3. `_pipeline()` calls `_run_script()` for each step in sequence
4. No cli_pipeline involvement

### Individual Steps (Debug/Advanced)
```bash
# Run one step at a time
cardamomot step infer_mixture -i /path/to/project --verbose
```

Flow:
1. User runs: `cardamomot step infer_mixture -i /path/to/project`
2. cli.py main() parses args, routes to lambda calling `_run_script()`
3. `_run_script()` executes named script
4. No cli_pipeline involvement

## No Redundancy

**Why separate files?**
- **cli.py**: Provides foundation for existing scripts + argparse utilities they import
- **cli_pipeline.py**: Focused on interactive workflow (step selection, validation, params)
- They serve different purposes and don't duplicate logic

**Division of concerns**:
```
cli.py (Low-level)        cli_pipeline.py (High-level Interactive)
├─ Argparse              ├─ Project validation
├─ Script execution      ├─ Step selection
├─ Logging config        ├─ Parameter gathering
└─ Utilities             └─ Workflow orchestration
```

No script logic in cli_pipeline, no interaction in cli.py.

## Entry Point Configuration

Both setup.py and pyproject.toml register the same console entry point:

```python
[project.scripts]
cardamomot = "CardamomOT.cli:main"
```

This means `cardamomot` command → always calls `CardamomOT.cli:main()` → which dispatches to appropriate subcommand handler.

## Alternative Usage Patterns

### As standalone module:
```python
from CardamomOT.cli_pipeline import run_pipeline_interactive

# Programmatic usage without CLI
run_pipeline_interactive(
    project_path="/path/to/project",
    use_defaults=True
)
```

### Direct script invocation (legacy):
```bash
python infer_mixture.py -i /path/to/project -m 0.5
```

Still works because these scripts import helpers from cli.py.

## Naming Consistency

All references updated to use **CardamomOT** (not CARDAMOM or cardamom):
- Package name: `CardamomOT`
- Console command: `cardamomot` (not `cardamom`)
- Module imports: `from CardamomOT import ...`
- Documentation: "CardamomOT" consistently

This avoids conflicts with your earlier "cardamom" package.

---

**Summary**: Clear separation, no redundancy, single entry point, three modular subcommands.
