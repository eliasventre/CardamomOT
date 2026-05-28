"""
CardamomOT
=====================================================================================

A gene regulatory network inference method adapted to time-course scRNA-seq datasets.

The algorithm consists of calibrating the parameters of a mechanistic model of gene
expression. The calibrated model can then be simulated to reproduce the dataset used
for inference. The simulation part is based on the Harissa package.

Key Features
------------
- Mechanistic inference of gene regulatory networks from scRNA-seq data
- Supports time-course and trajectory data
- Integration with mechanistic simulation models
- Network visualization and analysis tools

References
----------
1. Ventre E, et al. (2021). 
2. Ventre E, et al. (2023). 
3. Maugé Y. and Ventre E. (2026). CardamomOT: a mechanistic optimal transport-based framework for gene regulatory network inference, trajectory reconstruction and generative modeling. doi: 10.64898/2026.03.31.715390

Author
------
Elias Ventre
Yann Mauge

License
-------
MIT License
"""

from .logging import configure_logging, get_logger
from .cli import create_pipeline_parser, handle_common_args
from .config import (
    get_project_directories,
    get_default_parameters,
    DEFAULT_DATA_FOLDER,
    DEFAULT_CARDAMOM_FOLDER,
)

# Package version
__version__ = "2.0.0"

# Initialize logging with default configuration
logger = get_logger(__name__)

# ============================================================================
# Core inference functionality
# ============================================================================
from .inference import (
    kon_ref_vector,
    select_DEgenes,
    extract_degradation_rates,
)

# ============================================================================
# Core model classes
# ============================================================================
from .model import NetworkModel

# ============================================================================
# Visualization and analysis tools
# ============================================================================
from .tools import (
    plot_data_distrib,
    plot_data_pmf_temporal,
    plot_data_pmf_total,
    compare_marginals,
    plot_data_umap_toref,
    plot_data_umap_altogether,
    predict_cell_types,
    train_classifier,
    plot_cell_type_proportions,
    check_cell_types_full,
    check_cell_types_mixture,
    plot_results_rna_mixture,
    plot_results_rna_clean,
    plot_results_prot,
    plot_results_sim_kov,
    compare_cell_types,
    analyse_reseau,
    reseau_top_regulateurs,
    plot_network,
)

# ============================================================================
# Public API
# ============================================================================
__all__ = [
    # Logging utilities
    "configure_logging",
    "get_logger",
    # CLI utilities
    "create_pipeline_parser",
    "handle_common_args",
    # Configuration
    "get_project_directories",
    "get_default_parameters",
    "DEFAULT_DATA_FOLDER",
    "DEFAULT_CARDAMOM_FOLDER",
    # Inference functions
    "kon_ref_vector",
    "select_DEgenes",
    "extract_degradation_rates",
    # Model
    "NetworkModel",
    # Visualization
    "plot_data_distrib",
    "plot_data_pmf_temporal",
    "plot_data_pmf_total",
    "compare_marginals",
    "plot_data_umap_toref",
    "plot_data_umap_altogether",
    # Analysis tools
    "predict_cell_types",
    "train_classifier",
    "plot_cell_type_proportions",
    "check_cell_types_full",
    "check_cell_types_mixture",
    # Post-analysis plotting
    "plot_results_rna_mixture",
    "plot_results_rna_clean",
    "plot_results_prot",
    # KO/OV perturbation analysis
    "plot_results_sim_kov",
    "compare_cell_types",
    # Network visualisation
    "analyse_reseau",
    "reseau_top_regulateurs",
    "plot_network",
]

logger.debug(f"CARDAMOM {__version__} loaded successfully")
