"""
Configuration and constants for CARDAMOM pipeline.

Centralizes all constants, default parameters, and configuration options
used throughout the CARDAMOM pipeline for easy maintenance and consistency.
"""

from pathlib import Path
from typing import Dict, Any

# ============================================================================
# Directory Structure Defaults
# ============================================================================

DEFAULT_DATA_FOLDER = "Data"
DEFAULT_CARDAMOM_FOLDER = "cardamom"
DEFAULT_RESULTS_FOLDER = "results"

# Standard filenames
DEFAULT_DATA_FILE = "data.h5ad"
DEFAULT_GENE_LIST_FILE = "gene_list.txt"
DEFAULT_HALFLIFE_TABLE = "table_halflife_mammalian.csv"

# ============================================================================
# Data and Processing Parameters
# ============================================================================

# Default gene selection parameters
DEFAULT_N_GENES_TEMPORAL = 5  # Genes to select per timepoint
DEFAULT_N_GENES_CELLTYPE = 3  # Genes to select per cell type
DEFAULT_MIN_MEAN_EXPRESSION = 0.01  # Minimum mean expression threshold
DEFAULT_VAR_THRESHOLD = 1.2  # Coefficient of variation threshold for Poisson filtering

# Read depth normalization
DEFAULT_RD_MEDIAN_TOLERANCE = 1e-16  # Small epsilon for numerical stability

# ============================================================================
# Anatomical and Biological Constants
# ============================================================================

# Keys used in AnnData objects
REQUIRED_OBS_KEYS = {
    "time": "Measurement timepoint for each cell",
}

OPTIONAL_OBS_KEYS = {
    "cell_type": "Cell type classification (used for gene selection)",
    "rd": "Read depth normalization factor per cell",
    "d0": "mRNA degradation rate",
    "d1": "Protein degradation rate",
}

# Standard observation/variable keys in processed AnnData
STANDARD_OBS = [
    "time",
    "cell_type",
    "rd",
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
