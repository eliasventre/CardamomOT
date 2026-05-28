# Schiebinger et al. (2019) — iPSC reprogramming with stimulus schedule

> **Dataset:** Schiebinger G. et al., *Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming*, Cell 2019.  
> **Organism / tissue:** Mouse embryonic fibroblasts (MEF) reprogrammed toward iPSC.  
> **Time points:** 18 time points over 18 days (days 0–18) with varying cell densities.  
> **Genes analysed:** ~20 selected pluripotency and mesenchymal markers.

## Biological context

iPSC reprogramming is induced by the transient expression of four transcription factors (Oct4, Sox2, Klf4, c-Myc — the Yamanaka factors). The reprogramming stimulus is therefore **not constant over time**: it is active during an early induction window and then fades or is withdrawn, creating a non-trivial temporal structure that most GRN inference tools ignore.

## The non-trivial stimulus schedule

CardamomOT explicitly models the stimulus as a time-varying external signal. The `--stimulus` parameter controls how strongly the stimulus edges are penalised in the network inference objective.

The time schedule of reprogramming factor expression must be provided alongside the data. CardamomOT reads this schedule from the `time` field in `adata.obs` and accounts for it when computing the optimal-transport coupling between consecutive time points.

## CardamomOT configuration

```bash
cardamomot pipeline \
    -i experimental_datasets/Schiebinger \
    -s train \
    -c 0 \
    -r 0.3 \
    -m 0.5 \
    --stimulus 1.0 \
    --prior 1.0 \
    --force-basins 0.0 \
    --temporal-basins 0 \
    --test
```

Key differences from the other datasets:
- `-s train` splits cells into train/test sets, enabling held-out evaluation.
- `-r 0.3` uses a lower rate parameter suited to the longer time range (18 days).
- `--force-basins 0.0 --temporal-basins 0` disables NB mode forcing, appropriate for the continuous reprogramming dynamics.
- `--test` activates the test-set inference steps (`infer_test` + `check_test_to_train`).

Pre-computed outputs for both `--prior 0.5` and `--prior 1.0` are stored in `experimental_datasets/Schiebinger/cardamomOT/`.

## Loading pre-computed results

```python
import anndata as ad
from CardamomOT import NetworkModel, plot_data_pmf_temporal
import matplotlib.pyplot as plt

# Load inferred kinetic parameters
adata_beta = ad.read_h5ad(
    "experimental_datasets/Schiebinger/cardamomOT/adata_beta_stim1.0_prior1.0.h5ad"
)

# Load simulated trajectories
adata_sim = ad.read_h5ad(
    "experimental_datasets/Schiebinger/cardamomOT/adata_prot_simul_stim1.0_prior1.0.h5ad"
)

# Load real RNA trajectories for comparison
adata_rna = ad.read_h5ad(
    "experimental_datasets/Schiebinger/cardamomOT/adata_rna_traj_stim1.0_prior1.0.h5ad"
)

# Compare marginals: simulated vs observed
plot_data_pmf_temporal(adata_sim, adata_rna, genes=["Klf4", "Esrrb", "Col5a2"])
plt.tight_layout()
plt.show()
```

## In-silico KO experiments

Silencing the reprogramming factors in-silico and observing whether the system can still reach the iPSC attractor:

```python
ko_obox6_zfp42 = ad.read_h5ad(
    "experimental_datasets/Schiebinger/cardamomOT/"
    "adata_prot_simul_KO_none_OV_Obox6-Zfp42_stim1.0_prior1.0.h5ad"
)
```

## Key methodological point

The non-trivial stimulus schedule in this dataset illustrates a unique feature of CardamomOT: the optimal-transport coupling adapts to **time-varying external signals**, so the inferred regulatory network distinguishes between stimulus-driven and autonomous regulatory interactions. This is not possible with snapshot-based or time-aggregated GRN inference methods.
