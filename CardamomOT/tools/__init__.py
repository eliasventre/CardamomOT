"""
Tools for verifying the quality of the inference and post-analysis.
"""
from .marginals import plot_data_distrib, plot_data_pmf_temporal, plot_data_pmf_total, compare_marginals
from .umap import plot_data_umap_toref, plot_data_umap_altogether
from .characterize_cell_type import (
    predict_cell_types, train_classifier, plot_cell_type_proportions,
    check_cell_types_full, check_cell_types_mixture,
)
from .plot_results_sim import (
    plot_results_rna_mixture, plot_results_rna_clean, plot_results_prot,
)
from .plot_results_kov import (
    plot_results_sim_kov, compare_cell_types,
)
from .plot_networks import (
    analyse_reseau, reseau_top_regulateurs, plot_network,
)
from .estimate_proliferation import (
    score_gene_sets, scores_to_rates, estimate_growth_rates,
    combine_growth_rates_with_reference,
)


__all__ = [
    'plot_data_distrib', 'plot_data_pmf_temporal', 'plot_data_pmf_total',
    'compare_marginals', 'plot_data_umap_toref', 'plot_data_umap_altogether',
    'predict_cell_types', 'train_classifier', 'plot_cell_type_proportions',
    'check_cell_types_full', 'check_cell_types_mixture',
    'plot_results_rna_mixture', 'plot_results_rna_clean', 'plot_results_prot',
    'plot_results_sim_kov', 'compare_cell_types',
    'analyse_reseau', 'reseau_top_regulateurs', 'plot_network',
    'score_gene_sets', 'scores_to_rates', 'estimate_growth_rates',
    'combine_growth_rates_with_reference',
]
