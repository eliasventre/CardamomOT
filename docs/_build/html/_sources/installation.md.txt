# Installation

## Requirements

CardamomOT requires **Python ≥ 3.8**. Core dependencies (NumPy, SciPy, PyTorch, AnnData, POT, …) are installed automatically.

## Step 1 — Create a virtual environment

### With conda (recommended)

```bash
conda create -n cardamom_env python=3.12 -y
conda activate cardamom_env
```

:::{important}
**macOS Apple Silicon (arm64)** — install Numba and its threading runtimes from conda-forge *before* pip-installing CardamomOT, to avoid the `No threading layer` error:

```bash
conda install -c conda-forge numba llvmlite llvm-openmp tbb tbb-devel -y
```
:::

### With venv

```bash
python -m venv cardamom_env
source cardamom_env/bin/activate      # Linux / macOS
# cardamom_env\Scripts\activate       # Windows
```

## Step 2 — Install CardamomOT

Clone the repository and install in editable mode:

```bash
git clone https://github.com/eliasventre/CardamomOT.git
cd CardamomOT
```

| Install variant | Command | When to use |
|---|---|---|
| **Minimal** | `pip install -e .` | Core inference only |
| **Full** | `pip install -e ".[cli]"` | Interactive CLI (`cardamomot run`) |
| **Development** | `pip install -e ".[cli,dev,notebooks]"` | Testing + Jupyter notebooks |

## Step 3 — Verify

```bash
cardamomot --help
```

You should see the three sub-commands: `run`, `pipeline`, and `step`.

## Troubleshooting

**`command not found: cardamomot`**
: The entry-point was not placed on your PATH. Re-install with `pip install -e .` inside the activated environment.

**`No threading layer could be loaded` (Numba)**
: Install Numba from conda-forge as described in the Apple Silicon note above.

**`ModuleNotFoundError: questionary`**
: The interactive CLI needs the `cli` extra: `pip install -e ".[cli]"`. The tool falls back to plain `y/n` prompts if the library is absent.
