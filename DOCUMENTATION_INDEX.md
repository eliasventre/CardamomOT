# CardamomOT Documentation Index

Complete reference to all documentation available in this package.

## Quick Start (Choose Your Path)

### 👤 New User - Never Used CardamomOT Before
**Start here:**
1. [INSTALL_GUIDE.md](INSTALL_GUIDE.md) - Install and verify installation
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md#first-time-user-simplest) - Simplest workflow
3. [VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md) - Verify everything works
4. [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - Understand the workflow

**Expected time:** 15 minutes setup + 2-5 hours first analysis run

---

### 👨‍💼 Experienced User - Used CardamomOT Before
**Start here:**
1. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command reference and workflows
2. [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - Review details as needed
3. [config_template.yaml](config_template.yaml) - Customize parameters

**Expected time:** 5 minutes + analysis runtime

---

### 👨‍💻 Developer - Want to Extend/Modify CardamomOT
**Start here:**
1. [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Architecture and internals
2. [CardamomOT/cli_pipeline.py](CardamomOT/cli_pipeline.py) - Implementation details
3. [CardamomOT/cli.py](CardamomOT/cli.py) - CLI integration
4. [CardamomOT/inference/](CardamomOT/inference/) - Core analysis modules (fully documented)

**Expected time:** 30 minutes understanding + implementation time

---

## Complete Documentation Guide

### Installation & Setup

#### [**INSTALL_GUIDE.md**](INSTALL_GUIDE.md)
**Who should read:** Everyone on first use
**What it covers:**
- Step-by-step installation instructions (3 options: minimal, full, dev)
- Verification of installation
- Quick start workflow
- First analysis example
- Troubleshooting common installation issues

**Sections:**
- Installation (minimal, full, dev options)
- Verify installation command
- Quick start (prepare project, run pipeline, review results)
- First analysis example with expected output
- Troubleshooting (command not found, missing data, etc.)

**Time to read:** 10 minutes

---

### User Guides

#### [**PIPELINE_GUIDE.md**](PIPELINE_GUIDE.md)
**Who should read:** All users running analyses
**What it covers:**
- Detailed project structure setup
- Interactive vs. default mode usage
- Pipeline step explanations
- Hyperparameter descriptions
- Example workflow
- Troubleshooting
- Output file descriptions
- Next steps for result analysis

**Sections:**
- Project structure (recommended directory layout)
- Quick start (interactive and default modes)
- Pipeline steps explained (what each step does and when to skip)
- Hyperparameters guide
- Example complete workflow
- Troubleshooting
- Understanding output files

**Time to read:** 15 minutes

---

#### [**QUICK_REFERENCE.md**](QUICK_REFERENCE.md)
**Who should read:** Users wanting fast command reference
**What it covers:**
- All CLI commands (interactive, automated, traditional, individual steps)
- Common workflows (first-time, batch processing, custom config, debugging)
- Project structure visual
- Hyperparameter quick table
- Help commands
- Troubleshooting quick lookup
- Tips and tricks

**Sections:**
- Commands summary (with example usage)
- Common workflows (copy-paste ready)
- Project structure diagram
- Hyperparameter table
- Help and information commands
- Troubleshooting table
- Tips and tricks

**Time to read:** 5 minutes (reference, not linear)

---

#### [**VERIFICATION_CHECKLIST.md**](VERIFICATION_CHECKLIST.md)
**Who should read:** Users verifying proper setup
**What it covers:**
- Installation verification checks
- Project structure verification
- Optional dependencies verification
- Runtime verification (4 test levels)
- Output verification
- Troubleshooting by symptom
- System configuration (memory, disk, Python environment)
- Platform-specific notes (macOS, Linux, Windows)
- Getting help resources

**Sections:**
- Installation verification (Python, package, CLI)
- Project structure verification (Data/ directory, H5AD files)
- Optional dependencies (questionary, PyYAML)
- Runtime verification (4 tests from dry-run to full pipeline)
- Output verification (check results were created)
- Troubleshooting lookup
- System requirements
- Platform-specific guidance

**Time to work through:** 20-30 minutes

---

### Configuration

#### [**config_template.yaml**](config_template.yaml)
**Who should read:** Users with custom analysis needs
**What it covers:**
- Template configuration file structure
- All available hyperparameters
- Default values with documentation
- Sections for each pipeline stage

**Sections:**
- Project metadata (name, description)
- Data settings (input file, split)
- Per-step hyperparameters (read depth, mixture, selection, network, simulation, logging)

**How to use:**
1. Copy to your project: `cp config_template.yaml ~/my_project/config.yaml`
2. Edit values as needed
3. Run with defaults: `cardamomot run ~/my_project --default`

**Time to customize:** 5-10 minutes

---

### Developer Documentation

#### [**DEVELOPER_GUIDE.md**](DEVELOPER_GUIDE.md)
**Who should read:** Developers, maintainers, advanced users
**What it covers:**
- Architecture overview (diagram of components)
- Module descriptions (cli.py, cli_pipeline.py)
- Data structures (PIPELINE_STEPS definition)
- Function documentation (purpose, parameters, behavior)
- Configuration system (current and future)
- Error handling & resilience
- How to extend (adding steps, modifying parameters, config file support)
- Testing checklist
- Integration with existing code
- Performance considerations
- Future enhancement ideas

**Sections:**
- High-level architecture diagram
- CLI entry point (cli.py)
- Interactive pipeline orchestrator (cli_pipeline.py)
- Key functions (validate, select, input, run, orchestrate, main)
- Configuration system (current and planned)
- Error handling strategies
- Extending the pipeline (step-by-step guide)
- Testing verification checklist
- Backward compatibility notes
- Performance analysis
- Future enhancement timeline

**Time to read:** 30-45 minutes

---

## Source Code Documentation

### Core Modules (Fully English-Translated)

All source files in `CardamomOT/inference/` have been thoroughly documented and translated to English:

- **[network.py](CardamomOT/inference/network.py)** - Network classes and functions
  - Network inference core algorithm
  - Gradient-based optimization
  - Penalization schemes

- **[base.py](CardamomOT/inference/base.py)** - Base inference class
  - Template for inference algorithms
  - Parameter validation
  - Output formatting

- **[mixture.py](CardamomOT/inference/mixture.py)** - Mixture model inference
  - Cell state kinetic parameters
  - EM algorithm implementation
  - Parameter estimation

- **[degradations.py](CardamomOT/inference/degradations.py)** - Degradation rate inference
  - PyTorch ODE models
  - Epsilon parameter fitting
  - Stability analysis

- **[pretreatment.py](CardamomOT/inference/pretreatment.py)** - Data preprocessing
  - Gene filtering and selection
  - Normalization routines
  - Quality control checks

- **[trajectory.py](CardamomOT/inference/trajectory.py)** - Trajectory inference
  - Temporal alignment
  - Cell trajectory reconstruction
  - Curve fitting methods

- **[simulations.py](CardamomOT/inference/simulations.py)** - Simulation engines
  - ODE-based simulations (ApproxODE)
  - PDMP simulations (BurstyPDMP)
  - Synthetic trajectory generation

- **[network_final.py](CardamomOT/inference/network_final.py)** - Final network processing
  - Network cleanup and validation
  - Edge ranking and filtering
  - Result export

---

## File Organization

```
CardamomOT/
├── INSTALL_GUIDE.md              ← Installation and first-time setup
├── PIPELINE_GUIDE.md             ← Detailed usage workflow
├── QUICK_REFERENCE.md            ← Command cheatsheet
├── VERIFICATION_CHECKLIST.md     ← Setup verification tests
├── DEVELOPER_GUIDE.md            ← Technical internals (developers)
├── config_template.yaml          ← Configuration template
├── DOCUMENTATION_INDEX.md        ← This file
│
├── CardamomOT/
│   ├── cli.py                    ← CLI entry point
│   ├── cli_pipeline.py           ← Interactive pipeline (new)
│   ├── config.py                 ← Configuration utilities
│   ├── logging.py                ← Logging setup
│   │
│   ├── inference/                ← Core analysis modules (fully documented)
│   │   ├── network.py
│   │   ├── base.py
│   │   ├── mixture.py
│   │   ├── degradations.py
│   │   ├── pretreatment.py
│   │   ├── trajectory.py
│   │   ├── simulations.py
│   │   └── network_final.py
│   │
│   ├── model/                    ← Model definitions
│   └── tools/                    ← Visualization and analysis tools
│
├── README.md                     ← Main project README
├── setup.py                      ← Package installation config
├── pyproject.toml                ← Modern Python project config
│
└── scripts/
    ├── infer_rd.py               ← Individual pipeline scripts
    ├── infer_mixture.py
    ├── select_DEgenes_and_split.py
    ├── infer_network_structure.py
    ├── infer_network_simul.py
    ├── simulate_network.py
    └── ... (other scripts)
```

---

## Recommended Reading Order by Use Case

### 📊 "I want to run CardamomOT on my data NOW"
1. INSTALL_GUIDE.md (5 min)
2. QUICK_REFERENCE.md → First-Time User section (3 min)
3. VERIFICATION_CHECKLIST.md → Test 1-2 (10 min)
4. Run: `cardamomot run /path/to/project`

**Total:** ~30 minutes setup + analysis runtime

---

### 📚 "I want to understand what CardamomOT does"
1. README.md (main project documentation)
2. PIPELINE_GUIDE.md → Project Structure & Pipeline Steps sections (10 min)
3. config_template.yaml (understand parameters)
4. PIPELINE_GUIDE.md → Remaining sections (10 min)

**Total:** ~30 minutes learning

---

### 🔧 "I want to extend or modify CARDAMOM"
1. DEVELOPER_GUIDE.md (full read) (30 min)
2. CardamomOT/cli_pipeline.py (code review) (15 min)
3. CardamomOT/cli.py (integration review) (10 min)
4. Review relevant inference module (10-20 min)
5. Implement changes

**Total:** ~1-2 hours understanding + implementation time

---

### ❌ "Something is broken, I need help"
1. QUICK_REFERENCE.md → Troubleshooting section (2 min)
2. VERIFICATION_CHECKLIST.md → Troubleshooting section (5 min)
3. Check relevant log file:
   ```bash
   tail -50 /path/to/project/cardamom_output/pipeline.log
   ```
4. If still stuck:
   - Post issue with log excerpt + command run
   - Email support with full log file

**Total:** 5-10 minutes initial diagnosis

---

## Key Features Explained

### Interactive mode with checkboxes
```bash
cardamomot run /path/to/project
```
- Shows checkboxes for step selection (questionary)
- Falls back to Y/n prompts if questionary not available
- Prompts for custom hyperparameters
- See: QUICK_REFERENCE.md, PIPELINE_GUIDE.md

### Default mode (skip all prompts)
```bash
cardamomot run /path/to/project --default
```
- Runs all steps automatically
- Uses default hyperparameters
- Useful for scripting/batch processing
- See: QUICK_REFERENCE.md → Batch Processing section

### Configuration file support
```bash
# Create config.yaml from template
cp config_template.yaml /path/to/project/config.yaml

# Edit values as needed
# Run with defaults
cardamomot run /path/to/project --default
```
- Pre-define all parameters in YAML
- Repeatable analyses
- See: PIPELINE_GUIDE.md → Custom Configuration section

### Full English documentation
All `.py` files in `CardamomOT/inference/` are fully English-documented with:
- Module-level docstrings explaining purpose and usage
- Function docstrings with parameters and returns
- Inline comments for complex logic
- NumPy-style documentation format

See: Any file in CardamomOT/inference/

---

## Getting Help

### 📖 First Check
1. This index (you are here!)
2. Relevant guide matching your use case
3. QUICK_REFERENCE.md troubleshooting section
4. Relevant source file docstrings

### 💬 Community Support
- **GitHub Issues:** https://github.com/yourusername/CardamomOT/issues
- **Email:** support@email.com
- **Discussion:** https://github.com/yourusername/CardamomOT/discussions

### 📚 Additional Resources
- Original paper: [Insert citation]
- Method preprint: [Insert URL]
- Example notebooks: `utils/` folder
- Data availability: [Insert link]

---

## Document Status & Maintenance

| Document | Status | Last Updated |
|----------|---------|--------------|
| INSTALL_GUIDE.md | ✅ Current | 2024 |
| PIPELINE_GUIDE.md | ✅ Current | 2024 |
| QUICK_REFERENCE.md | ✅ Current | 2024 |
| VERIFICATION_CHECKLIST.md | ✅ Current | 2024 |
| DEVELOPER_GUIDE.md | ✅ Current | 2024 |
| config_template.yaml | ✅ Current | 2024 |
| Inference module docs | ✅ Full English | 2024 |
| CLI modules (cli.py, cli_pipeline.py) | ✅ Current | 2024 |

---

## Glossary

- **H5AD** - HDF5-based data format for annotated data matrices (AnnData)
- **Pipeline** - Sequence of analysis steps from raw data to network inference and simulation
- **Hyperparameter** - User-configurable parameter (learning rate, number of genes, etc.)
- **Default mode** - Running pipeline without interactive prompts (--default flag)
- **Interactive mode** - Running with checkboxes for step selection and parameter prompts
- **Subprocess** - Individual pipeline script executed as separate Python process
- **Config file** - YAML file with pre-defined analysis parameters

---

**Welcome to CardamomOT! 🌿 Choose your starting point above and begin.**
