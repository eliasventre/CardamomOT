"""
Tools for verifying the quality of the inference and post-analysis.
"""
from .marginals import plot_data_distrib, plot_data_pmf_temporal, plot_data_pmf_total, compare_marginals
from .umap import plot_data_umap_toref, plot_data_umap_altogether
from .characterize_cell_type import (
    predict_cell_types, train_classifier, plot_cell_type_proportions,
    check_cell_types_full, check_cell_types_mixture,
)
from .plot_results import (
    plot_results_rna_mixture, plot_results_rna_clean, plot_results_prot,
    plot_results_rna_clean_kov,
)
from .plot_networks import (
    add_edges_from_matrix, analyse_reseau, reseau_top_regulateurs, plot_networks,
)


__all__ = [
    'plot_data_distrib', 'plot_data_pmf_temporal', 'plot_data_pmf_total',
    'compare_marginals', 'plot_data_umap_toref', 'plot_data_umap_altogether',
    'predict_cell_types', 'train_classifier', 'plot_cell_type_proportions',
    'check_cell_types_full', 'check_cell_types_mixture',
    'plot_results_rna_mixture', 'plot_results_rna_clean', 'plot_results_prot',
    'plot_results_rna_clean_kov',
    'add_edges_from_matrix', 'analyse_reseau', 'reseau_top_regulateurs', 'plot_networks',
]
