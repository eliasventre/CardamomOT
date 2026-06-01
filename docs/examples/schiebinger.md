# Schiebinger et al. (2019) — iPSC reprogramming with stimulus schedule

> **Dataset:** Schiebinger G, Shu J, Tabaka M, Cleary B, Subramanian V, Solomon A, et al. *Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming.* Cell 2019; 176:928–43.  
> **Organism / tissue:** Mouse embryonic fibroblasts (MEF) reprogrammed toward iPSC.  
> **Time points:** 18 time points over 18 days (days 0–18) with varying cell densities.  
> **Genes analysed:** 108 genes selected by `select_DEgenes_and_split`, using a list of 60 reference genes identified in the original Schiebinger et al. (2019) article as a seed for differential expression selection.

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
    --mean-forcing 0.5 \
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

## Post-analysis notebooks

Pre-computed outputs are visualised with the notebooks in `utils/`:

| Notebook | Content |
|---|---|
| `plot_networks.ipynb` | Inferred GRN (stimulus-driven vs autonomous edges) |
| `plot_data_to_sim.ipynb` | Data vs simulation comparison across 18 time points |
| `plot_data_to_sim_KOV.ipynb` | `KO_none_OV_Obox6-Zfp42` and other perturbations |
| `compare_cell_types.ipynb` | MEF/iPSC proportions in data vs simulation |

## In-silico KO experiments

Silencing reprogramming factors in-silico tests whether the system can reach the iPSC attractor autonomously. Pre-computed outputs (e.g. `KO_none_OV_Obox6-Zfp42`, `KO_none_OV_Zfp42`) are stored in `experimental_datasets/Schiebinger/cardamomOT/` and compared to the wild-type simulation using `utils/plot_data_to_sim_KOV.ipynb`.

## Key methodological point

The non-trivial stimulus schedule in this dataset illustrates a unique feature of CardamomOT: the optimal-transport coupling adapts to **time-varying external signals**, so the inferred regulatory network distinguishes between stimulus-driven and autonomous regulatory interactions. This is not possible with snapshot-based or time-aggregated GRN inference methods.
