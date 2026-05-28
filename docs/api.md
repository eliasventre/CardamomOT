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
    NetworkModel,          # core model class
    kon_ref_vector,        # burst kinetics fitting
    extract_degradation_rates,
    select_DEgenes,
    # --- visualisation ---
    plot_data_distrib,
    plot_data_pmf_temporal,
    plot_data_umap_toref,
    compare_marginals,
    animate_dynamic_grns,
    predict_cell_types,
)
```
