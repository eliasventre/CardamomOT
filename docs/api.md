# API Reference

The CardamomOT API is organised into three sub-packages:

| Sub-package | Description |
|---|---|
| `CardamomOT.model` | Core `NetworkModel` class — parameters, fitting, simulation |
| `CardamomOT.inference` | Kinetics, network inference, trajectory algorithms |
| `CardamomOT.tools` | Visualisation and analysis utilities |

The full auto-generated reference is available below.

```{toctree}
:maxdepth: 3

autoapi/CardamomOT/index
```

## Top-level imports

The most commonly used objects are re-exported at package level:

```python
from CardamomOT import (
    NetworkModel,                  # core model class
    kon_ref_vector,                # burst kinetics fitting
    extract_degradation_rates,
    select_DEgenes,
    # --- marginals & UMAP ---
    plot_data_distrib,
    plot_data_pmf_temporal,
    plot_data_umap_toref,
    compare_marginals,
    # --- cell-type analysis ---
    train_classifier,
    predict_cell_types,
    check_cell_types_full,
    check_cell_types_mixture,
    # --- result comparison ---
    plot_results_rna_mixture,
    plot_results_rna_clean,
    plot_results_prot,
    # --- KO/OV perturbations ---
    plot_results_sim_kov,
    compare_cell_types,
    # --- network visualisation ---
    analyse_reseau,
    reseau_top_regulateurs,
    plot_network,
    # --- literature-based proliferation rate estimation ---
    score_gene_sets,
    estimate_growth_rates,
    combine_growth_rates_with_reference,
    # --- data file helpers ---
    find_data_file,
    read_gene_list,
)
```
