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
pip install -e .
```

The `cardamomot` command and all dependencies (including visualization tools) are installed by default. Optional extras:

| Extra | Command | Adds |
|---|---|---|
| `dev` | `pip install -e ".[dev]"` | pytest, black, flake8 |
| `notebooks` | `pip install -e ".[notebooks]"` | Jupyter, scVelo |

## Step 3 — Verify

```bash
cardamomot --help
```

You should see the three sub-commands: `run`, `pipeline`, and `step`.

## Troubleshooting

**`command not found: cardamomot`**
: The `cardamomot` script is installed into your environment's `bin/` directory. Make sure the environment is **activated** before running it.

  ```bash
  # With conda — always activate first:
  conda activate cardamom_env
  cardamomot --help

  # Verify where the script was installed:
  python -c "import sys; print(sys.prefix + '/bin/cardamomot')"

  # Fallback that always works regardless of PATH:
  python -m CardamomOT.cli --help
  ```

  On HPC clusters, if `~/.local/bin` is not in your `PATH`, add it:
  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  ```

**`No threading layer could be loaded` (Numba)**
: Install Numba from conda-forge as described in the Apple Silicon note above.

**`ModuleNotFoundError: No module named 'scanpy'`**
: Run `pip install -e .` again — `scanpy` is a core dependency and should be installed automatically. If the error persists, install it explicitly: `pip install scanpy`.
