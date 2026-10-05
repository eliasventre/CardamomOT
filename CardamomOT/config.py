"""
Configuration and constants for CARDAMOM pipeline.

Centralizes all constants, default parameters, and configuration options
used throughout the CARDAMOM pipeline for easy maintenance and consistency.
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence

# ============================================================================
# Directory Structure Defaults
# ============================================================================

DEFAULT_DATA_FOLDER = "Data"
DEFAULT_CARDAMOM_FOLDER = "cardamom"
DEFAULT_RESULTS_FOLDER = "results"

# Standard filenames
DEFAULT_DATA_FILE = "data.h5ad"
DEFAULT_GENE_LIST_FILE = "gene_list.txt"
DEFAULT_HALFLIFE_TABLES = {"mouse": "halflife_mouse.tsv", "human": "halflife_human.tsv"}  # in CardamomOT/data/halflife

# ============================================================================
# Data and Processing Parameters
# ============================================================================

# Default gene selection parameters
DEFAULT_N_GENES_TEMPORAL = 5  # Genes to select per timepoint
DEFAULT_N_GENES_CELLTYPE = 3  # Genes to select per cell type
DEFAULT_MIN_MEAN_EXPRESSION = 0.01  # Minimum mean expression threshold
DEFAULT_VAR_THRESHOLD = 1.2  # Coefficient of variation threshold for Poisson filtering

# ============================================================================
# Anatomical and Biological Constants
# ============================================================================

# Keys used in AnnData objects
REQUIRED_OBS_KEYS = {
    "time": "Measurement timepoint for each cell",
}

OPTIONAL_OBS_KEYS = {
    "cell_type": "Cell type classification (used for gene selection)",
    "d0": "mRNA degradation rate",
    "d1": "Protein degradation rate",
}

# Standard observation/variable keys in processed AnnData
STANDARD_OBS = [
    "time",
    "cell_type",
    "d0",
    "d1",
]

# ============================================================================
# Inference and Simulation Parameters
# ============================================================================

# Network inference defaults
DEFAULT_PRIOR_STRENGTH = 1.0  # Prior weighting in inference (0-1)
DEFAULT_STIM_LEVEL = 1.0  # Stimulus strength (0-1)

# Mixture model inference
DEFAULT_MIXTURE_TOLERANCE = 1e-6  # Convergence tolerance
DEFAULT_MIXTURE_MAX_ITER = 1000  # Maximum iterations

# Kinetic parameters
DEFAULT_PROTEIN_HALFLIFE_MIN = 30  # minutes
DEFAULT_PROTEIN_HALFLIFE_MAX = 720  # minutes (12 hours)
DEFAULT_MRNA_HALFLIFE_MIN = 5  # minutes
DEFAULT_MRNA_HALFLIFE_MAX = 120  # minutes

# ============================================================================
# Visualization Defaults
# ============================================================================

# Colormap defaults
CMAP_GENE_EXPRESSION = "viridis"
CMAP_NETWORK = "coolwarm"
CMAP_CELL_TYPES = "Dark2"

# Figure size defaults (in inches)
DEFAULT_FIGURE_WIDTH = 10
DEFAULT_FIGURE_HEIGHT = 8

# ============================================================================
# Error Messages and Warnings
# ============================================================================

ERROR_MSG_NO_DATA = (
    "No data file found. Create a subfolder 'Data' in your project directory "
    "and place a count table named 'data.h5ad' inside. "
    "The AnnData object must have 'time' in adata.obs."
)

ERROR_MSG_NO_TIMES = (
    "The input data has no temporal information or only one timepoint. "
    "Please ensure 'time' column exists in adata.obs with at least one value=0 "
    "and at least one value>0."
)

ERROR_MSG_INVALID_SPLIT = (
    "Invalid data split specified. Expected splits in: "
    "{available_splits}"
)

WARNING_MSG_NO_CELL_TYPES = (
    "No cell type information found in adata.obs['cell_type']. "
    "Gene selection will use only temporal information."
)

WARNING_MSG_NO_GENE_LIST = (
    "No external gene list found at {gene_list_path}. "
    "Using only data-driven gene selection."
)

# ============================================================================
# Configuration Helper Functions
# ============================================================================

def get_project_directories(project_path: Path) -> Dict[str, Path]:
    """
    Get all standard subdirectories for a CARDAMOM project.

    Args:
        project_path: Root directory of the project.

    Returns:
        Dictionary with keys: data, cardamom, results.
    """
    project_path = Path(project_path)
    return {
        "data": project_path / DEFAULT_DATA_FOLDER,
        "cardamom": project_path / DEFAULT_CARDAMOM_FOLDER,
        "results": project_path / DEFAULT_RESULTS_FOLDER,
    }


def find_data_file(data_dir: Path, basename: str,
                    extensions: Sequence[str] = (".csv", ".txt")) -> Optional[Path]:
    """
    Look up an optional data file that may be provided as either a .csv or a
    .txt table of the exported inputs (e.g. transition_rates.csv, proliferation_rates.txt).

    Args:
        data_dir: Directory to look in (e.g. project_dir / "Data").
        basename: Filename without extension (e.g. "proliferation_rates").
        extensions: Extensions to try, in priority order.

    Returns:
        Path to the first existing file, or None if none of the extensions match.
    """
    data_dir = Path(data_dir)
    for ext in extensions:
        candidate = data_dir / f"{basename}{ext}"
        if candidate.exists():
            return candidate
    return None


# Obs columns tried in order for each task-specific cell-type grouping.
# Proliferation and transition groupings fall back on each other before `cell_type`.
CELL_TYPE_OBS_KEYS = {
    "selection": ("cell_type_selection", "cell_type"),
    "proliferation": ("cell_type_proliferation", "cell_type_transition", "cell_type"),
    "transition": ("cell_type_transition", "cell_type_proliferation", "cell_type"),
}


def resolve_cell_type_obs(adata, task: str) -> Optional[str]:
    """
    Pick which adata.obs column to use as the cell-type grouping for a given
    task ("selection", "proliferation" or "transition"), letting preprocessing
    override the generic `cell_type` with a task-specific labeling.

    Returns the first column of `CELL_TYPE_OBS_KEYS[task]` present in
    adata.obs, else None.
    """
    for key in CELL_TYPE_OBS_KEYS[task]:
        if key in adata.obs.columns:
            return key
    return None


# Exit code of scripts that stop on stationary data (no/single timepoint)
STATIONARY_EXIT_CODE = 3
STATIONARY_MESSAGE = "Stationary data (no or single timepoint): switch to method CardamomOT-stat, in prep."


def check_stationary(adata, time_key: str = "time") -> bool:
    """
    Fill a missing `adata.obs[time_key]` with 0 and return True when the data
    have at most one timepoint (stationary setting, handled by CardamomOT-stat).
    """
    if time_key not in adata.obs.columns:
        adata.obs[time_key] = 0.0
    return adata.obs[time_key].nunique() <= 1


# obs columns read by CardamomOT, most specific first (matched in this order by harmonize_obs)
EXPECTED_OBS = ("cell_type_proliferation", "cell_type_transition", "cell_type_selection",
                "proliferation_net_rate", "cell_type", "dataset_id", "lineage", "time")


def harmonize_obs(adata, min_ratio: float = 0.8) -> Dict[str, str]:
    """
    Rename in place the obs columns close to a column CardamomOT expects (EXPECTED_OBS) but
    absent: same name up to case, the words of one name containing those of the other (e.g.
    'cell_type_annotation' -> 'cell_type'), or a string similarity >= min_ratio (< 20% apart).
    Each column is renamed at most once; ambiguous matches are left as they are. Matches by words
    are made for all keys before matches by similarity (so that 'cell_type_annotation' becomes
    'cell_type', not the similar 'cell_type_transition').
    Returns {old name: new name}.
    """
    from difflib import SequenceMatcher
    words = lambda s: set(w for w in re.split(r'[^a-z0-9]+', s.lower()) if w)
    ratio = lambda key, col: SequenceMatcher(None, key, col.lower()).ratio()

    def by_words(key, col):
        if col.lower() == key:
            return 2.0
        kw, cw, r = words(key), words(col), ratio(key, col)
        # A shorter name inside the key must be close to it ('dataset' yes; 'id', 'cell' no)
        return 1.0 + r if (kw <= cw or (cw <= kw and r >= 0.65)) else None

    def by_similarity(key, col):
        r = ratio(key, col)
        return r if r >= min_ratio else None

    import pandas as pd
    numeric_keys = {"time", "proliferation_net_rate"}

    def compatible(key, col):
        # Numeric keys take numeric columns, categorical keys (cell types, samples, lineages) the others
        return pd.api.types.is_numeric_dtype(adata.obs[col]) == (key in numeric_keys) or key == "dataset_id"

    renamed, word_candidates = {}, set()
    for score in (by_words, by_similarity):
        for key in EXPECTED_OBS:
            if key in adata.obs.columns:
                continue
            scores = {col: s for col in adata.obs.columns
                      if col not in EXPECTED_OBS and compatible(key, col)
                      and not (score is by_similarity and col in word_candidates)
                      and (s := score(key, col)) is not None}
            if score is by_words:
                word_candidates.update(scores)
            if not scores:
                continue
            best = sorted(scores.items(), key=lambda kv: -kv[1])
            if len(best) > 1 and best[1][1] == best[0][1]:
                print(f"[CardamomOT] obs '{key}' absent; ambiguous candidates {[c for c, _ in best[:3]]}, none renamed")
                continue
            col = best[0][0]
            adata.obs.rename(columns={col: key}, inplace=True)
            renamed[col] = key
            print(f"[CardamomOT] obs '{col}' renamed '{key}' (expected by CardamomOT)")
    return renamed


# Layers tried first for raw counts when adata.X is not made of counts
RAW_COUNT_LAYERS = ("counts_raw", "counts", "raw_counts", "raw", "spliced")


def _is_counts(X, n_rows: int = 2000) -> bool:
    """True if the first n_rows rows of X hold non-negative integers."""
    import numpy as np
    import scipy.sparse
    sub = X[:n_rows]
    vals = sub.data if scipy.sparse.issparse(sub) else np.asarray(sub).ravel()
    vals = vals[vals != 0]
    return bool(vals.size == 0 or (vals.min() >= 0 and np.all(np.abs(vals - np.round(vals)) < 1e-6)))


def ensure_raw_counts(adata, label: str = "data"):
    """
    adata with raw counts in X: adata itself if X holds counts, else a new AnnData (same obs,
    var, obsm, uns; no layers) whose X is the first layer of raw counts found (RAW_COUNT_LAYERS
    first, then any other layer, then adata.raw). The input object is not modified. Raises
    ValueError if X is not counts (e.g. log-normalised) and no raw counts are found.
    """
    import anndata as ad
    if _is_counts(adata.X):
        return adata
    # The log1p record of the transformed X no longer applies to the raw counts
    uns = {k: v for k, v in adata.uns.items() if k != 'log1p'}
    names = [k for k in RAW_COUNT_LAYERS if k in adata.layers] + \
            [k for k in adata.layers.keys() if k not in RAW_COUNT_LAYERS]
    for k in names:
        if adata.layers[k].shape == adata.shape and _is_counts(adata.layers[k]):
            print(f"[CardamomOT] {label}: X is not raw counts (e.g. log-normalised); using layers['{k}']")
            return ad.AnnData(X=adata.layers[k], obs=adata.obs, var=adata.var, obsm=dict(adata.obsm),
                              uns=uns)
    if adata.raw is not None and adata.raw.shape == adata.shape and _is_counts(adata.raw.X):
        print(f"[CardamomOT] {label}: X is not raw counts (e.g. log-normalised); using adata.raw.X")
        return ad.AnnData(X=adata.raw.X, obs=adata.obs, var=adata.var, obsm=dict(adata.obsm),
                          uns=uns)
    raise ValueError(f"{label}: X is not made of raw counts (non-integer or negative values, e.g. "
                     f"log-normalised) and no layer of raw counts was found (layers: "
                     f"{list(adata.layers.keys())}, raw: {adata.raw is not None}). CardamomOT needs "
                     f"raw counts: put them in X or in layers['counts_raw'].")


def find_stimulus_schedule(data_dir) -> Optional[str]:
    """Stimulus schedule of the inference (stimulus_schedule_inference.txt of the exported inputs), or None."""
    path = Path(data_dir) / "stimulus_schedule_inference.txt"
    return str(path) if path.exists() else None


def n_inference_stimuli(data_dir) -> int:
    """Number of stimuli of the inference: columns of its schedule (1 without schedule)."""
    path = find_stimulus_schedule(data_dir)
    if path is None:
        return 1
    arr = np.loadtxt(path, ndmin=2)
    return int(arr.shape[1])


def simulation_schedule(data_dir, n_stimuli):
    """
    Stimulus schedules of the simulations, stimulus_schedule_simulate.txt of the exported inputs: one row per
    simulated time, first the n_stimuli columns of the inference stimuli, then one column per
    perturbation stimulus of KO_OV_Stim_simulate.txt (STIM1, STIM2...). Returns (inference
    stimuli (rows, n_stimuli) or None, perturbation stimuli (rows, k) or None). Without file, the
    inference schedule is used and the perturbation stimuli take their default (0 at the first time, 1 after).
    """
    path = Path(data_dir) / "stimulus_schedule_simulate.txt"
    if path.exists():
        arr = np.loadtxt(path, ndmin=2)
        if arr.shape[1] < n_stimuli:
            raise ValueError(f"{path} has {arr.shape[1]} column(s), fewer than the {n_stimuli} inference stimuli")
        print(f"[CardamomOT] Simulation schedule from {path}: {n_stimuli} inference stimuli"
              + (f", {arr.shape[1] - n_stimuli} perturbation stimuli" if arr.shape[1] > n_stimuli else ""))
        return arr[:, :n_stimuli], (arr[:, n_stimuli:] if arr.shape[1] > n_stimuli else None)
    inf = find_stimulus_schedule(data_dir)
    return (np.loadtxt(inf, ndmin=2) if inf is not None else None), None


def read_stimulus_targets(data_dir: Path) -> Optional[List[List[str]]]:
    """
    Possible targets of each stimulus (stimulus_targets.txt of the exported inputs): one column per
    stimulus, in the order of the columns of stimulus_schedule_inference.txt, one gene per row (columns
    separated by tabs, empty cells allowed; with a single stimulus, any separator). '#' starts a
    comment. Returns one gene list per column, or None without file.
    """
    path = find_data_file(Path(data_dir), "stimulus_targets")
    if path is None:
        return None
    lines = [line.split("#", 1)[0].rstrip("\n") for line in Path(path).read_text().splitlines()]
    lines = [line for line in lines if line.strip()]
    if not any("\t" in line for line in lines):
        return [[g for g in re.split(r"[,\s]+", "\n".join(lines)) if g]]
    rows = [line.split("\t") for line in lines]
    n_cols = max(len(r) for r in rows)
    return [[r[j].strip() for r in rows if j < len(r) and r[j].strip()] for j in range(n_cols)]


def stimulus_target_mask(targets, genes, n_stimuli):
    """
    (n_stimuli, n_genes) mask of the allowed stimulus -> gene edges from read_stimulus_targets:
    a stimulus whose column names no gene of `genes` (or has no column) is unconstrained (all
    True), not deprived of targets. Gene names are matched case-insensitively. None if no
    stimulus is constrained.
    """
    if not targets:
        return None
    up = [str(g).upper() for g in genes]
    mask = np.ones((n_stimuli, len(genes)), dtype=bool)
    constrained = False
    for s in range(n_stimuli):
        listed = {str(g).upper() for g in targets[s]} if s < len(targets) else set()
        hit = np.array([g in listed for g in up])
        if hit.any():
            mask[s] = hit
            constrained = True
            print(f"[CardamomOT] stimulus {s}: {int(hit.sum())} possible targets (Data/stimulus_targets)")
        elif listed:
            print(f"[CardamomOT] stimulus {s}: none of its {len(listed)} listed targets is in the data, unconstrained")
    return mask if constrained else None


def read_gene_list(path: Path) -> List[str]:
    """
    Read a flat gene list from a .csv or .txt file (one gene per line, or
    comma-separated — both are accepted since either just needs splitting on
    whitespace/commas).

    Args:
        path: Path to the gene list file.

    Returns:
        List of gene symbols, in file order, blank entries removed; text after '#' on a line is a comment.
    """
    text = "\n".join(line.split("#", 1)[0] for line in Path(path).read_text().splitlines())
    return [g for g in re.split(r"[,\s]+", text) if g]


def get_default_parameters() -> Dict[str, Any]:
    """
    Get all default parameters as a dictionary.

    Returns:
        Dictionary of all default parameter values.
    """
    return {
        "n_genes_temporal": DEFAULT_N_GENES_TEMPORAL,
        "n_genes_celltype": DEFAULT_N_GENES_CELLTYPE,
        "min_mean_expression": DEFAULT_MIN_MEAN_EXPRESSION,
        "var_threshold": DEFAULT_VAR_THRESHOLD,
        "prior_strength": DEFAULT_PRIOR_STRENGTH,
        "stim_level": DEFAULT_STIM_LEVEL,
        "mixture_tolerance": DEFAULT_MIXTURE_TOLERANCE,
        "mixture_max_iter": DEFAULT_MIXTURE_MAX_ITER,
    }
