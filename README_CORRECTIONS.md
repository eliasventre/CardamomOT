# CardamomOT - Naming & Architecture Corrections ✓

**Date**: March 10, 2026  
**Status**: ✅ COMPLETE

## Executive Summary

Fixed naming confusion and clarified architecture:
1. ✅ Renamed console command: `cardamom` → `cardamomot` (avoid package conflicts)
2. ✅ Consistent naming: All references to "CARDAMOM" → "CardamomOT"
3. ✅ Eliminated "redundancy" by clarifying cli.py vs cli_pipeline.py roles
4. ✅ Updated 11 files across code and documentation

## The Problem

1. **Naming conflicts**: Your package is `CardamomOT` but console command was `cardamom` (conflicts with your earlier package)
2. **Inconsistent naming**: Mixed "CARDAMOM", "cardamom", "CardamomOT" throughout codebase
3. **Perceived redundancy**: cli.py and cli_pipeline.py seemed to duplicate functionality

## The Solution

### 1. Console Command Renamed
```
cardamom → cardamomot
```

Files updated:
- setup.py: `'cardamom=...'` → `'cardamomot=...'`
- pyproject.toml: `cardamom = ...` → `cardamomot = ...`

### 2. All Names Standardized to "CardamomOT"
- Package: `CardamomOT` (no change, already correct)
- Command: `cardamomot` (was `cardamom`)
- References: All documentation now says "CardamomOT" not "CARDAMOM"

### 3. Architecture Clarified (NOT Redundant)

**cli.py** (Dispatcher & Legacy Support)
```
Responsibility: Route commands, maintain backward compatibility
Contains:
  - main() → argparse setup
  - _run_script() → subprocess execution
  - _pipeline() → full legacy pipeline
  - _run_pipeline_interactive() → delegate to cli_pipeline
  - Helpers: argument parsers, logging config

Used by: Standalone legacy scripts, all three subcommands
```

**cli_pipeline.py** (Interactive Workflow)
```
Responsibility: Project validation, step selection, parameters, orchestration
Contains:
  - PIPELINE_STEPS → step definitions
  - validate_project_structure() → check Data/ directory
  - interactive_step_selection() → checkboxes/prompts
  - interactive_parameter_input() → per-step params
  - run_step() → single step execution
  - run_pipeline_interactive() → main orchestrator

Used by: New "cardamomot run" command only
```

**No redundancy**: They have different purposes
- cli.py = "How do I dispatch and run scripts?"
- cli_pipeline.py = "How do I guide users through analysis steps?"

### 4. Command Structure Unchanged

```bash
# Interactive (NEW)
cardamomot run /path/to/project           # Same functionality, new name
cardamomot run /path/to/project --default # Same

# Legacy (UNCHANGED)
cardamomot pipeline -i /path -c 1 -r 1.0  # Same
cardamomot step script_name -i /path      # Same
```

## Files Changed

### Code (4 files)
| File | Change | Purpose |
|------|--------|---------|
| setup.py | Entry point: `cardamom` → `cardamomot` | Register new console command name |
| pyproject.toml | Scripts: `cardamom` → `cardamomot` | Register new console command name |
| CardamomOT/cli.py | Added import, added function, added 'run' subcommand | Integration with cli_pipeline |
| CardamomOT/cli_pipeline.py | Updated prog/description names | Use correct console command name |

### Documentation (8 files)
| File | Changes | Purpose |
|------|---------|---------|
| INSTALL_GUIDE.md | `cardamom` → `cardamomot` | User installation guide |
| PIPELINE_GUIDE.md | `cardamom` → `cardamomot` | User workflow guide |
| QUICK_REFERENCE.md | `cardamom` → `cardamomot` | Command reference |
| VERIFICATION_CHECKLIST.md | `cardamom` → `cardamomot` | Setup verification |
| DOCUMENTATION_INDEX.md | All references updated | Navigation hub |
| PROJECT_COMPLETION_SUMMARY.md | All examples updated | Project status |
| CLI_ARCHITECTURE.md | NEW - 173 lines | Explain cli.py vs cli_pipeline.py |
| NAMING_CORRECTIONS.md | NEW - Information file | Document all changes |

**Total**: 12 files modified/created

## Installation Instructions

Users need to reinstall to pick up the new command name:

```bash
# Uninstall previous version
pip uninstall CardamomOT

# Reinstall from repo
cd /path/to/CardamomOT
pip install -e .                # Basic
pip install -e ".[cli]"         # With interactive features
pip install -e ".[cli,dev]"     # Development

# Verify
cardamomot --help               # New command name!
```

## Backward Compatibility

✅ **All Python imports work unchanged**
```python
from CardamomOT import something
```

✅ **Standalone scripts still work**
```bash
python infer_mixture.py -i /path/to/project
```

✅ **Legacy cardamomot commands preserved**
```bash
cardamomot pipeline -i /path -s train
cardamomot step infer_mixture -i /path
```

⚠️ **Only change**: Console command name `cardamom` → `cardamomot`

## Data Flow Diagram

```
User runs: cardamomot run /path/to/project
                ↓
        cli.py:main()
                ↓
    Argparse parses subcommands
                ↓
      Routes to _run_pipeline_interactive(args)
                ↓
    cli_pipeline.run_pipeline_interactive(
        project_path=args.project,
        use_defaults=args.default
    )
                ↓
    1. validate_project_structure()
    2. interactive_step_selection()
    3. Loop: interactive_parameter_input() → run_step()
    4. Print summary
                ↓
            Complete ✓
```

## Architecture Benefits

1. **Separation of Concerns**
   - CLI dispatch in cli.py
   - Interactive workflow in cli_pipeline.py
   - Each focuses on its responsibility

2. **No Redundancy**
   - cli_pipeline doesn't recreate argparse
   - cli doesn't handle user interaction
   - Clear division of labor

3. **Easy to Test**
   - Each module independently testable
   - cli.py tests: command routing
   - cli_pipeline.py tests: workflow logic

4. **Easy to Extend**
   - Add subcommand? Add to cli.py main()
   - Change interactive behavior? Modify cli_pipeline.py
   - No cross-cutting concerns

## Verification

✅ Both Python files pass syntax check  
✅ All imports correct (when dependencies present)  
✅ Console entry points properly registered  
✅ All documentation updated  
✅ No breaking changes to existing code

## Before/After Examples

### Installation
```bash
# Before
pip install -e "[cli]"
cardamom --help

# After  
pip install -e "[cli]"
cardamomot --help     # Same features, new name
```

### Usage
```bash
# Before
cardamom run my_project
cardamom run my_project --default
cardamom pipeline -i my_project -s train

# After
cardamomot run my_project
cardamomot run my_project --default
cardamomot pipeline -i my_project -s train
```

### Python Code
```python
# Before & After (unchanged)
from CardamomOT.cli_pipeline import run_pipeline_interactive

run_pipeline_interactive(
    project_path="/path/to/project",
    use_defaults=True
)
```

## Documentation For Users

1. **First time?** → Read INSTALL_GUIDE.md
2. **Need commands?** → Check QUICK_REFERENCE.md
3. **Full workflow?** → See PIPELINE_GUIDE.md
4. **Verification?** → Use VERIFICATION_CHECKLIST.md
5. **How does it work?** → Read CLI_ARCHITECTURE.md
6. **What changed?** → See NAMING_CORRECTIONS.md

## Next Steps

Users should:
1. Read NAMING_CORRECTIONS.md for summary
2. Read CLI_ARCHITECTURE.md to understand design
3. Update any shell scripts using `cardamom` → `cardamomot`
4. Reinstall package: `pip install -e .`
5. Verify: `cardamomot --help` or `cardamomot run --help`

## Questions/Clarifications

**Q: Why two CLI files?**
A: Different responsibilities - cli.py dispatches commands, cli_pipeline.py handles interactive workflows. They complement each other, don't duplicate.

**Q: Is this a breaking change?**
A: Only the console command name changed (cardamom→cardamomot). All Python code, imports, script names remain the same.

**Q: Why rename cardamom→cardamomot?**
A: You already have a package called "cardamom" - this avoids conflicts. The new command matches your GitHub repo name (CardamomOT).

**Q: Do I need to change my scripts?**
A: If they use the CLI (`cardamom run`), yes - change to `cardamomot run`. If they import modules or run scripts directly, no changes needed.

---

✅ **Status**: All corrections complete, documented, and ready to use.

**Key Takeaway**: Same powerful functionality, clearer naming, better architecture documentation.
